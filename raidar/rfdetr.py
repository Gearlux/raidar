"""Roboflow RF-DETR — torchvision-native wrapper.

Exposes the same forward contract as torchvision's detection models:

* ``model.train(); model(images, targets) → Dict[str, Tensor]`` — single-key
  loss dict ``{"loss": total}`` (the weighted sum
  ``sum(loss_dict[k] * weight_dict[k])`` over the criterion's matched keys).
* ``model.eval();  model(images)         → List[Dict[str, Tensor]]`` — per-image
  ``{boxes, scores, labels}`` in xyxy-pixel space.

RF-DETR (Roboflow's Real-time DETR, 2025) is a DINOv2-backed transformer
detector. We instantiate the high-level ``rfdetr.RFDETR`` class to benefit
from its config validation, pretrained-weight download, and detection-head
re-init logic, then keep references to the inner ``LWDETR`` ``nn.Module``
as a submodule so PyTorch's training / device transfer / state-dict mechanics
flow through naturally.

Image preprocessing: assumes all images are already
``[3, resolution, resolution]`` in ``[0, 1]`` float range. ImageNet
normalisation is applied internally.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Union, cast

import torch
import torch.nn as nn
from confluid import configurable
from loggair import get_logger
from matrainer.torch import load_lightning_checkpoint
from matrainer.uri import ModelRef, file_model_ref
from recordstream import DetectionOutput
from torch import Tensor

logger = get_logger(__name__)

#: Scheme naming the RF-DETR zoo in a model reference. The library downloads a variant's
#: COCO-pretrained checkpoint from its own release assets, so the variant IS the identifier and
#: there is no repository page to link — see :attr:`RFDETRModel.model_refs`.
RFDETR_URI_PREFIX = "rfdetr://"

_VARIANT_CLASSES: Dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def _build_rfdetr(
    variant: Literal["nano", "small", "medium", "large"],
    num_classes: int,
    pretrained_detector: bool,
    resolution: Optional[int],
) -> Any:
    """Instantiate a high-level ``RFDETR`` variant and return it."""
    import rfdetr

    cls_name = _VARIANT_CLASSES[variant]
    rfdetr_cls = getattr(rfdetr, cls_name)
    kwargs: Dict[str, Any] = {"num_classes": int(num_classes)}
    if not pretrained_detector:
        kwargs["pretrain_weights"] = None
    if resolution is not None:
        kwargs["resolution"] = int(resolution)
    return rfdetr_cls(**kwargs)


def _build_criterion_args(model_config: Any, device: Any) -> SimpleNamespace:
    """Build a minimal namespace for :func:`build_criterion_and_postprocessors`."""
    return SimpleNamespace(
        num_classes=model_config.num_classes,
        # A LITERAL, like the other loss coefficients below, because rfdetr 1.9.0 moved this off
        # ModelConfig — `rfdetr/_namespace.py` now calls TrainConfig "its sole owner" (default 1.0).
        # Reading it from `model_config` raised `AttributeError: 'RFDETRNanoConfig' object has no
        # attribute 'cls_loss_coef'` at criterion-build time; this function is handed only a model
        # config, so the default belongs here beside `bbox_loss_coef` / `giou_loss_coef`.
        cls_loss_coef=1.0,
        dec_layers=model_config.dec_layers,
        two_stage=model_config.two_stage,
        group_detr=model_config.group_detr,
        ia_bce_loss=model_config.ia_bce_loss,
        segmentation_head=model_config.segmentation_head,
        num_select=model_config.num_select,
        device=device,
        set_cost_class=2,
        set_cost_bbox=5,
        set_cost_giou=2,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        focal_alpha=0.25,
        aux_loss=True,
        sum_group_losses=False,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        mask_ce_loss_coef=5.0,
        mask_dice_loss_coef=5.0,
        # `build_criterion_and_postprocessors` reads `args.mask_point_sample_ratio`; the old spelling
        # here matched nothing and survived only because it is reached solely on a segmentation head,
        # which this detection path never sets.
        mask_point_sample_ratio=16,
    )


def _build_rfdetr_targets(
    targets: List[Dict[str, Tensor]],
    resolution: int,
    device: torch.device,
) -> List[Dict[str, Tensor]]:
    """Convert raidar's xyxy-pixel targets to RF-DETR's cxcywh-normalized format."""
    out: List[Dict[str, Tensor]] = []
    for t in targets:
        boxes_xyxy = t["boxes"].to(device).float()
        labels = t["labels"].to(device).long()
        n = boxes_xyxy.shape[0]
        if n == 0:
            cxcywh = torch.zeros((0, 4), device=device, dtype=torch.float32)
        else:
            cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / (2.0 * resolution)
            cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / (2.0 * resolution)
            w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / resolution
            h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / resolution
            cxcywh = torch.stack([cx, cy, w, h], dim=1)
        out.append({"boxes": cxcywh, "labels": labels})
    return out


def _decode_rfdetr_predictions(
    postprocess_results: List[Dict[str, Tensor]],
    num_classes: int,
    score_threshold: float,
) -> List[DetectionOutput]:
    """Convert RF-DETR's ``PostProcess`` output into per-image
    :class:`~recordstream.DetectionOutput` dicts. Drops the no-object
    slot at index ``num_classes`` and filters by ``score_threshold``.
    """
    predictions: List[DetectionOutput] = []
    for r in postprocess_results:
        scores = r["scores"]
        labels = r["labels"]
        boxes = r["boxes"]
        keep = (scores >= score_threshold) & (labels < num_classes)
        predictions.append(
            DetectionOutput(
                boxes=boxes[keep].float(),
                scores=scores[keep].float(),
                labels=labels[keep].long(),
            )
        )
    return predictions


@configurable(task="detection", role="model")
class RFDETRModel(nn.Module):
    """Torchvision-native RF-DETR wrapper for
    :class:`~raidar.lightning_detector.LightningDetector`.

    Forward contract:

    * **Training**: ``forward(images, targets) → {"loss": total}``.
    * **Eval**: ``forward(images) → List[Dict]`` of per-image predictions.

    Args:
        num_classes: Foreground class count. Internally an extra
            ``num_classes`` slot is reserved for the no-object class and any
            prediction with that label is dropped during decode.
        variant: ``"nano"`` (default), ``"small"``, ``"medium"``, ``"large"``.
        pretrained_detector: Load full COCO-pretrained RF-DETR weights.
        resolution: Square input size in pixels. Must be a positive integer
            divisible by 32. ``None`` uses the variant's default resolution.
        score_threshold: Eval-time confidence cutoff applied after
            background-class filtering.
        checkpoint_path / checkpoint_strict: Optional Lightning ``.ckpt`` load.
    """

    def __init__(
        self,
        num_classes: int = 1,
        variant: Literal["nano", "small", "medium", "large"] = "nano",
        pretrained_detector: bool = True,
        resolution: Optional[int] = None,
        score_threshold: float = 0.5,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
    ) -> None:
        from rfdetr.detr import RFDETR
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors

        super().__init__()
        if variant not in _VARIANT_CLASSES:
            raise ValueError(f"variant must be one of {sorted(_VARIANT_CLASSES)}, got {variant!r}.")
        self.num_classes = int(num_classes)
        self.variant = variant
        self.score_threshold = float(score_threshold)
        # Kept so the model can name its own weight artifacts (`model_refs`) — a value the
        # run record needs cannot be one the constructor consumes and forgets.
        self.pretrained_detector = bool(pretrained_detector)
        self.checkpoint_path = checkpoint_path

        rfdetr = _build_rfdetr(
            variant=variant,
            num_classes=self.num_classes,
            pretrained_detector=pretrained_detector,
            resolution=resolution,
        )
        self.detector = rfdetr.model.model
        self.postprocess = rfdetr.model.postprocess
        self.resolution = int(rfdetr.model_config.resolution)
        criterion_args = _build_criterion_args(rfdetr.model_config, rfdetr.model.device)
        criterion, _ = build_criterion_and_postprocessors(criterion_args)
        self.criterion = criterion
        self.register_buffer("_mean", torch.tensor(RFDETR.means).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(RFDETR.stds).view(1, 3, 1, 1))

        if checkpoint_path:
            load_lightning_checkpoint(self, checkpoint_path, strict=checkpoint_strict)

    def _preprocess(self, images: List[Tensor]) -> Tensor:
        """Stack images and apply ImageNet normalization on-device."""
        for i, img in enumerate(images):
            if img.shape[-1] != self.resolution or img.shape[-2] != self.resolution:
                raise ValueError(
                    f"RFDETRModel requires all images to be ({self.resolution}, {self.resolution}); "
                    f"got image[{i}].shape={tuple(img.shape)}. Resize upstream."
                )
        stacked = torch.stack(images)
        mean = cast(Tensor, self._mean)
        std = cast(Tensor, self._std)
        return (stacked - mean) / std

    def _criterion_loss(self, outputs: Dict[str, Any], targets_detr: List[Dict[str, Tensor]]) -> Tensor:
        loss_dict = self.criterion(outputs, targets_detr)
        weight_dict = self.criterion.weight_dict
        device = outputs["pred_logits"].device
        zero = torch.zeros((), device=device)
        total: Tensor = sum(
            (loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict),
            zero,
        )
        return total

    @property
    def model_refs(self) -> List[ModelRef]:
        """The weight artifacts this detector loads — see :mod:`matrainer.uri`.

        A variant names one COCO-pretrained checkpoint the library fetches from its own release
        assets; there is no repository URL to open, so none is invented.
        """
        refs: List[ModelRef] = []
        if self.pretrained_detector and self.variant:
            refs.append(ModelRef(uri=f"{RFDETR_URI_PREFIX}{self.variant}"))
        if self.checkpoint_path:
            refs.append(file_model_ref(self.checkpoint_path))
        return refs

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[Dict[str, Tensor], List[DetectionOutput]]:
        if self.training:
            if targets is None:
                raise ValueError("RFDETRModel in train mode requires targets.")
            normalised = self._preprocess(images)
            targets_detr = _build_rfdetr_targets(targets, self.resolution, normalised.device)
            outputs = self.detector(normalised, targets_detr)
            loss_total = self._criterion_loss(outputs, targets_detr)
            return {"loss": loss_total}

        normalised = self._preprocess(images)
        outputs = self.detector(normalised)
        target_sizes = torch.tensor(
            [[self.resolution, self.resolution]] * normalised.shape[0],
            device=normalised.device,
        )
        results = self.postprocess(outputs, target_sizes)
        predictions = _decode_rfdetr_predictions(results, self.num_classes, self.score_threshold)
        return predictions


__all__ = [
    "RFDETRModel",
    "_build_rfdetr_targets",
    "_decode_rfdetr_predictions",
]
