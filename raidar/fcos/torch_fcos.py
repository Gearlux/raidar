"""``TorchFcos`` — the shared tiny-FCOS spec in ``torch.nn``.

The torch entry in the cross-framework comparison :mod:`raidar.fcos.spec` describes, and the
opposite kind of model from the zoo wrappers it sits beside: the torchvision builders, RF-DETR
and the Hub wrapper reach pretrained detectors, this one wraps nothing at all. For real work
prefer a zoo model — a tiny net trained from scratch loses to a pretrained detector every time.

It serves ALL THREE torch backends. ``LightningDetector``, ``TorchDetector`` and
``FastaiDetector`` differ in the training loop, not in what a model is, so each takes any module
speaking the torchvision-native contract and this is one: ``forward(images, targets)`` returns
the loss dict in train mode, ``forward(images)`` the per-image prediction dicts in eval mode.

**Only the LOSS is written in torch** — it must be differentiable. Target assignment runs
through the spec's numpy reference per image (no gradients flow through targets), and the eval
path IS :func:`raidar.fcos.spec.decode_predictions` over the raw head maps — so the sigmoid/exp
decode rules live in exactly one place and every framework's implementation decodes identically
by construction.

**The channels question:** the ``input_shape`` knob reads ``(H, W, C)`` channels-LAST like every
implementation's (one config line drives any of them), while the tensors ``forward`` consumes
are the native contract's — a list of ``[C, H, W]`` float images, exactly what the shared
``ToTensor`` preprocessing already produces. Every image must already be ``input_shape``-sized:
resize upstream with the joint image+boxes resize op; a mismatch raises naming that rule.

Labels index the class map's channels directly, so the model is convention-agnostic: under
torchvision's background-at-0 convention (``label_offset: 1``, ``num_classes: K+1``) channel 0
simply never fires, and under the 0-indexed convention (``label_offset: 0``, ``num_classes: K``)
every channel is a real class. Predictions come back with the same ids the targets used.
"""

from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from confluid import configurable
from loggair import get_logger
from torch import Tensor, nn
from torchvision.ops import sigmoid_focal_loss

from raidar.fcos.spec import (
    BACKBONE_WIDTHS,
    CENTER_RADIUS,
    HEAD_WIDTH,
    INPUT_SHAPE,
    KERNEL_SIZE,
    assign_targets,
    decode_predictions,
    stride,
)

logger = get_logger(__name__)

__all__ = ["TorchFcos"]


def _conv_bn_relu(in_channels: int, width: int, kernel_size: int, conv_stride: int = 1) -> nn.Sequential:
    """``conv -> BN -> ReLU`` — the spec's repeated unit, in torch's vocabulary."""
    return nn.Sequential(
        nn.Conv2d(in_channels, width, kernel_size, stride=conv_stride, padding=kernel_size // 2),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
    )


class _Fcos(nn.Module):
    """The spec's layer graph. Private: the ``@configurable`` wrapper is the public surface."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        widths: Tuple[int, ...],
        head_width: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        stages: List[nn.Module] = []
        channels = in_channels
        for width in widths:
            stages.append(_conv_bn_relu(channels, width, kernel_size))
            stages.append(_conv_bn_relu(width, width, kernel_size, conv_stride=2))
            channels = width
        self.backbone = nn.Sequential(*stages)
        self.trunk = _conv_bn_relu(channels, head_width, kernel_size)
        # The three 1x1 heads: raw outputs, NO activation — the spec's decode/loss own the
        # sigmoid/exp transforms.
        self.cls_head = nn.Conv2d(head_width, num_classes, kernel_size=1)
        self.box_head = nn.Conv2d(head_width, 4, kernel_size=1)
        self.ctr_head = nn.Conv2d(head_width, 1, kernel_size=1)

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.trunk(self.backbone(images))
        return self.cls_head(x), self.box_head(x), self.ctr_head(x)


@configurable(task="detection", role="model", lazy=True, framework="torch")
class TorchFcos(nn.Module):
    """The shared tiny-FCOS spec (:mod:`raidar.fcos.spec`), implemented in ``torch.nn``.

    Args:
        num_classes: Width of the per-cell class head. Sentinel ``0`` is invalid but permitted so
            the class is zero-arg constructible; a detector flows the resolved count in at run
            time, so it is not written in YAML.
        input_shape: ``(H, W, C)`` — the SAME channels-last knob every implementation takes.
            Images must arrive resized to it (the joint image+boxes resize op, upstream).
        widths: Channels per backbone stage. Changing it breaks comparability with the other
            implementations — see :mod:`raidar.fcos.spec`.
        head_width: Channels of the shared head trunk.
        score_threshold: Eval-time decode floor on ``sqrt(cls_prob * ctr_prob)``.
        iou_threshold: Eval-time per-class NMS overlap threshold.
        max_detections: Eval-time cap on detections per image, best-first.
        seed: Seed for the parameter initializers, so a config reproduces its weights.
    """

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = BACKBONE_WIDTHS,
        head_width: int = HEAD_WIDTH,
        score_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
        seed: int = 0,
    ) -> None:
        # `nn.Module.__init__` first — a torch module refuses submodule assignment before it.
        super().__init__()
        # Partial: store only. No parameter allocation — the workspace zero-arg mandate.
        self.num_classes = num_classes
        self.input_shape = cast(Tuple[int, int, int], tuple(input_shape))
        self.widths = tuple(widths)
        self.head_width = head_width
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.seed = seed
        # Public submodule name so a state_dict round-trips as `net.*` — the shared-spec
        # convention (a leading underscore reads as "private" for something checkpoints name).
        self.net: Optional[nn.Module] = None

    def solidify(self) -> nn.Module:
        """Build (once) and return the live network — confluid's ``flow()``-time hook.

        Required configuration is validated HERE rather than in ``__init__``, which is what
        keeps zero-arg construction working. ``flow()`` calls this, so ``self.parameters()`` is
        populated before an optimizer reads it.
        """
        if self.net is None:
            if not self.num_classes:
                raise ValueError(
                    "TorchFcos.num_classes is not set — a detector flows the resolved count in "
                    "at run time, or set it explicitly."
                )
            torch.manual_seed(self.seed)
            self.net = _Fcos(
                num_classes=self.num_classes,
                in_channels=int(self.input_shape[-1]),
                widths=self.widths,
                head_width=self.head_width,
                kernel_size=KERNEL_SIZE,
            )
            logger.info(f"Built the shared FCOS spec in torch.nn ({self.num_classes} classes, widths={self.widths}).")
        return cast(nn.Module, self.net)

    @property
    def backbone(self) -> nn.Module:
        """The live network, built on first access.

        ``flow()`` never touches property getters (confluid's lazy-init rule 3), which is why
        :meth:`solidify` exists as a method as well; this is what makes a directly constructed,
        never-flowed instance usable.
        """
        return self.solidify()

    @property
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`raidar.fcos.spec` predicts.

        BatchNorm's running mean/variance are torch BUFFERS rather than parameters, so they are
        excluded by construction here — an MLX implementation has to filter them by name.
        """
        return int(sum(p.numel() for p in self.backbone.parameters() if p.requires_grad))

    # ------------------------------------------------------------------ #
    # The native contract
    # ------------------------------------------------------------------ #
    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """Loss dict in train mode, per-image prediction dicts in eval mode."""
        batch = self._stack(images)
        cls_map, box_map, ctr_map = self.backbone(batch)
        if self.training:
            if targets is None:
                raise ValueError("TorchFcos needs targets in training mode (the native contract)")
            return self._loss(cls_map, box_map, ctr_map, targets)
        return self._decode(cls_map, box_map, ctr_map)

    def _stack(self, images: List[Tensor]) -> Tensor:
        """The list-of-CHW native input as one ``[N, C, H, W]`` batch, size-checked.

        Checked here rather than left to fail: dense targets only line up when every image is
        exactly ``input_shape``, so the error must name the resize rule, not a shape mismatch.
        """
        height, width = int(self.input_shape[0]), int(self.input_shape[1])
        for image in images:
            if tuple(image.shape[-2:]) != (height, width):
                raise ValueError(
                    f"TorchFcos needs every image resized to {height}x{width} (dense per-cell "
                    f"targets need one fixed grid), got {tuple(image.shape[-2:])}. Resize "
                    "upstream with the joint image+boxes resize op."
                )
        return torch.stack(list(images))

    def _loss(
        self, cls_map: Tensor, box_map: Tensor, ctr_map: Tensor, targets: List[Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        """The spec's loss recipe, differentiable: focal + (1 - IoU) + centerness BCE.

        Assignment runs through the spec's NUMPY reference per image — no gradient flows through
        targets, so numpy is correct there and keeps one assignment rule for every framework.
        """
        device = cls_map.device
        assigned = [
            assign_targets(
                boxes=target["boxes"].detach().cpu().numpy(),
                labels=target["labels"].detach().cpu().numpy(),
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                widths=self.widths,
                center_radius=CENTER_RADIUS,
            )
            for target in targets
        ]
        cls_target = torch.from_numpy(np.stack([a["cls"] for a in assigned])).to(device)
        box_target = torch.from_numpy(np.stack([a["box"] for a in assigned])).to(device)
        ctr_target = torch.from_numpy(np.stack([a["centerness"] for a in assigned])).to(device)
        fg = torch.from_numpy(np.stack([a["foreground"] for a in assigned])).to(device)
        num_fg = fg.sum().clamp(min=1).to(cls_map.dtype)

        # Heads come out [N, C', rows, cols]; the maps were assigned [N, rows, cols, C'].
        cls_logits = cls_map.permute(0, 2, 3, 1)
        box_logits = box_map.permute(0, 2, 3, 1)
        ctr_logits = ctr_map.squeeze(1)

        loss_cls = sigmoid_focal_loss(cls_logits, cls_target, alpha=0.25, gamma=2.0, reduction="sum") / num_fg

        if bool(fg.any()):
            predicted = torch.exp(box_logits[fg]) * float(stride(self.widths))
            expected = box_target[fg]
            lr = torch.minimum(predicted[:, 0], expected[:, 0]) + torch.minimum(predicted[:, 2], expected[:, 2])
            tb = torch.minimum(predicted[:, 1], expected[:, 1]) + torch.minimum(predicted[:, 3], expected[:, 3])
            intersection = lr.clamp(min=0) * tb.clamp(min=0)
            area_p = (predicted[:, 0] + predicted[:, 2]) * (predicted[:, 1] + predicted[:, 3])
            area_e = (expected[:, 0] + expected[:, 2]) * (expected[:, 1] + expected[:, 3])
            iou = intersection / (area_p + area_e - intersection).clamp(min=1e-9)
            loss_box = (1.0 - iou).sum() / num_fg
            loss_ctr = F.binary_cross_entropy_with_logits(ctr_logits[fg], ctr_target[fg], reduction="sum") / num_fg
        else:
            # No positive cell in the batch: the box/centerness heads have nothing to learn
            # from, and indexing with an all-False mask would still be fine — but a literal zero
            # keeps the loss dict's meaning honest.
            loss_box = cls_map.sum() * 0.0
            loss_ctr = cls_map.sum() * 0.0
        return {"loss_cls": loss_cls, "loss_box": loss_box, "loss_ctr": loss_ctr}

    def _decode(self, cls_map: Tensor, box_map: Tensor, ctr_map: Tensor) -> List[Dict[str, Tensor]]:
        """Eval predictions — the spec's numpy decode over the raw maps, per image."""
        predictions: List[Dict[str, Tensor]] = []
        for n in range(cls_map.shape[0]):
            decoded = decode_predictions(
                cls_logits=cls_map[n].permute(1, 2, 0).detach().cpu().numpy(),
                box_raw=box_map[n].permute(1, 2, 0).detach().cpu().numpy(),
                ctr_logit=ctr_map[n, 0].detach().cpu().numpy(),
                input_shape=self.input_shape,
                widths=self.widths,
                score_threshold=self.score_threshold,
                iou_threshold=self.iou_threshold,
                max_detections=self.max_detections,
            )
            predictions.append(
                {
                    "boxes": torch.from_numpy(decoded["boxes"]),
                    "scores": torch.from_numpy(decoded["scores"]),
                    "labels": torch.from_numpy(decoded["labels"]),
                }
            )
        return predictions

    def __repr__(self) -> str:
        """Short and honest before the build — the default would print an empty module."""
        if self.net is None:
            return f"TorchFcos(num_classes={self.num_classes}, widths={self.widths}, unbuilt)"
        return cast(str, super().__repr__())
