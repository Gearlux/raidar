"""Roboflow RF-DETR wrapped for the Marainer Trainer flow.

RF-DETR (Roboflow's Real-time DETR, 2025) is a DINOv2-backed transformer
detector. This module wraps the high-level ``RFDETR`` class hierarchy
(``RFDETRNano`` / ``RFDETRSmall`` / ``RFDETRMedium`` / ``RFDETRLarge``) into
the same ``DetectionStepOutput`` contract as :class:`raidar.yolo26.YOLO26Model`
and :class:`raidar.rtdetr.RTDETRModel`, so the existing generic adapters
(:class:`~raidar.detection.TorchvisionDetectionLoss`,
:class:`~raidar.detection.DetectionMetricAdapter`,
:func:`~raidar.detection.detection_collate_fn`) work unchanged.

Unlike Ultralytics' detector classes, ``rfdetr.RFDETR`` is **not** an
``nn.Module`` — it's a high-level wrapper that holds the underlying
``LWDETR`` ``nn.Module`` at ``self.model.model`` plus a ``PostProcess`` at
``self.model.postprocess``. We instantiate the high-level class to benefit
from its config validation, pretrained-weight download, and detection-head
re-init logic, then keep references to the inner ``nn.Module``s as
submodules of this wrapper so PyTorch's training / device transfer / state
dict mechanics work as expected.

Loss contract: RF-DETR's training loop calls
``criterion(outputs, targets) -> loss_dict`` and sums
``loss_dict[k] * weight_dict[k]`` over the matched keys. We mirror that
formula and pack the scalar as ``losses={"loss": total}`` so
:class:`~raidar.detection.TorchvisionDetectionLoss` passes it through
unchanged at default weight 1.0.

Inference contract: ``LWDETR(img_tensor)`` returns
``{"pred_logits": [B, num_queries, num_classes_with_bg], "pred_boxes": [B,
num_queries, 4]}`` (cxcywh, normalized to [0, 1]). ``PostProcess`` does
top-K-by-flat-logit selection, converts cxcywh→xyxy, and scales boxes to a
``target_sizes`` tensor (we pass the model resolution, since this wrapper
already requires pre-resized square images). The internal class index
``num_classes`` is the no-object slot — those predictions are filtered out.

Image preprocessing: this wrapper assumes all images are already the same
square size (``resolution`` × ``resolution``) in ``[0, 1]`` float range.
ImageNet normalization is applied internally (matching RF-DETR's
``predict()`` pipeline). Variable-size images must be resized upstream.

Typical wiring (YAML)::

    model: !class:raidar.rfdetr.RFDETRModel
      num_classes: 2
      variant: nano
      pretrained_detector: true
    loss_fn: !class:raidar.detection.TorchvisionDetectionLoss
      # No `weights` block — single "loss" key passes through with weight 1.0.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, cast

import torch
import torch.nn as nn
from torch import Tensor

from confluid import configurable
from logflow import get_logger
from marainer.outputs import DetectionOutput
from raidar.checkpoint import load_lightning_checkpoint
from raidar.detection import DetectionBatchInput, DetectionStepOutput

logger = get_logger(__name__)

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
    """Instantiate a high-level ``RFDETR`` variant and return it.

    The returned object is **not** an ``nn.Module`` — it's the high-level
    Roboflow wrapper. The underlying ``nn.Module`` lives at
    ``rfdetr.model.model``, and the post-processor at
    ``rfdetr.model.postprocess``. We let the wrapper handle pretrained-weight
    download (controlled by ``pretrain_weights=None`` for offline mode) and
    detection-head reinitialization to ``num_classes`` (handled inside
    ``RFDETR.__init__`` via :func:`_load_pretrain_weights_into`).
    """
    import rfdetr

    cls_name = _VARIANT_CLASSES[variant]
    rfdetr_cls = getattr(rfdetr, cls_name)
    kwargs: Dict[str, Any] = {"num_classes": int(num_classes)}
    if not pretrained_detector:
        # Skip the detector-level COCO checkpoint. The DINOv2 backbone still
        # loads its own pretrained weights from HuggingFace (RF-DETR's encoder
        # is bootstrapped from DINOv2 regardless of the detector checkpoint).
        kwargs["pretrain_weights"] = None
    if resolution is not None:
        kwargs["resolution"] = int(resolution)
    return rfdetr_cls(**kwargs)


def _build_criterion_args(model_config: Any, device: Any) -> SimpleNamespace:
    """Build a minimal namespace for :func:`build_criterion_and_postprocessors`.

    Mirrors the loss-related fields of ``rfdetr._namespace.build_namespace``
    without instantiating a full ``TrainConfig`` (which would require a
    ``dataset_dir`` we don't have). The hardcoded values are RF-DETR's
    standard detection-loss defaults (set_cost / loss_coef / focal_alpha)
    and match the ones :func:`build_namespace` injects for the default
    training pipeline. Only fields actually read by the criterion or
    matcher are populated.
    """
    return SimpleNamespace(
        # From model_config
        num_classes=model_config.num_classes,
        cls_loss_coef=model_config.cls_loss_coef,
        dec_layers=model_config.dec_layers,
        two_stage=model_config.two_stage,
        group_detr=model_config.group_detr,
        ia_bce_loss=model_config.ia_bce_loss,
        segmentation_head=model_config.segmentation_head,
        num_select=model_config.num_select,
        device=device,
        # Detection-loss defaults — mirror rfdetr._namespace.build_namespace.
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
        # Segmentation defaults — only consumed when segmentation_head=True.
        mask_ce_loss_coef=5.0,
        mask_dice_loss_coef=5.0,
        mask_point_sample_ratio=16,
    )


def _build_rfdetr_targets(
    targets: List[Dict[str, Tensor]],
    resolution: int,
    device: torch.device,
) -> List[Dict[str, Tensor]]:
    """Convert raidar's xyxy-pixel targets to RF-DETR's cxcywh-normalized format.

    Inputs:

    * ``targets``: list of per-image ``{"boxes": [N, 4] xyxy-pixel float32,
      "labels": [N] int64}`` dicts (raidar / torchvision convention).

    Output (DETR convention consumed by ``SetCriterion``):

    * list of ``{"boxes": [N, 4] cxcywh-normalized float32, "labels": [N]
      int64}`` dicts. Empty targets become zero-row tensors so the criterion
      can iterate without IndexErrors.
    """
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
    :class:`~marainer.outputs.DetectionOutput` dicts.

    ``PostProcess`` returns one dict per image with
    ``{"scores": [K], "labels": [K], "boxes": [K, 4] xyxy-pixel}`` where the
    label index ``num_classes`` is the no-object slot — those predictions
    are dropped here, then the remainder is filtered by ``score_threshold``.
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


@configurable
class RFDETRModel(nn.Module):
    """Roboflow RF-DETR wrapped for the Marainer Trainer flow.

    Returns a :class:`~raidar.detection.DetectionStepOutput` in every mode:

    * **Training**: ``{losses: {"loss": total}, predictions: None}`` — the
      single ``"loss"`` key holds the canonical weighted sum
      ``sum(loss_dict[k] * weight_dict[k])`` over the matched criterion keys
      (loss_ce / loss_bbox / loss_giou plus their per-decoder-layer aux
      variants).
    * **Validation / test** with targets: ``{losses: {"loss": total},
      predictions: [...]}``. The val-loss recompute runs under
      ``torch.no_grad()`` (RF-DETR's transformer has no BatchNorm — the
      BN-freeze trick used by YOLO26 / RT-DETR / Faster R-CNN doesn't apply).
    * **Pure prediction** (no targets): ``{losses: None, predictions: [...]}``.

    Args:
        num_classes: Foreground class count. RF-DETR labels classes
            ``0..num_classes-1``; internally an extra ``num_classes`` slot
            is reserved for the no-object class and any prediction with
            that label is dropped during decode.
        variant: Model size — ``"nano"`` (smallest, default), ``"small"``,
            ``"medium"``, or ``"large"``. Larger variants are heavier but
            more accurate; each ships with a different default ``resolution``
            (Nano=384, Small=512, Medium=576, Large=704).
        pretrained_detector: Load full COCO-pretrained RF-DETR weights
            (downloads on first use). When False, the detector head starts
            random — but the DINOv2 backbone still loads its own pretrained
            weights from HuggingFace, so the model is never fully cold-start.
            Strongly recommended True for small-dataset training.
        resolution: Square input size in pixels. Must be a positive integer
            divisible by ``patch_size * num_windows`` for the chosen variant
            (32 for Nano/Small/Medium/Large). All images in the batch must
            already be ``[3, resolution, resolution]``; resize upstream.
            ``None`` uses the variant's default resolution.
        score_threshold: Eval-time confidence cutoff applied after
            background-class filtering.
        compute_val_loss: When False, skips the loss recomputation during
            eval. Defaults to True.
        checkpoint_path: Optional path to a Lightning ``.ckpt`` (or plain
            torch state-dict) saved by a previous training run. Loaded after
            ``pretrained_detector`` weights so checkpoint values win.
        checkpoint_strict: Forwarded to ``load_state_dict(strict=...)``.
            Defaults to False.
    """

    def __init__(
        self,
        num_classes: int = 1,
        variant: Literal["nano", "small", "medium", "large"] = "nano",
        pretrained_detector: bool = True,
        resolution: Optional[int] = None,
        score_threshold: float = 0.5,
        compute_val_loss: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
    ) -> None:
        from rfdetr.detr import RFDETR
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors

        super().__init__()
        if variant not in _VARIANT_CLASSES:
            raise ValueError(
                f"variant must be one of {sorted(_VARIANT_CLASSES)}, got {variant!r}."
            )
        self.num_classes = int(num_classes)
        self.variant = variant
        self.score_threshold = float(score_threshold)
        self.compute_val_loss = bool(compute_val_loss)

        rfdetr = _build_rfdetr(
            variant=variant,
            num_classes=self.num_classes,
            pretrained_detector=pretrained_detector,
            resolution=resolution,
        )
        # Register the inner LWDETR as a real submodule so .train()/.eval(),
        # .to(device), and state_dict() flow through it. Same for criterion.
        self.detector = rfdetr.model.model
        self.postprocess = rfdetr.model.postprocess
        self.resolution = int(rfdetr.model_config.resolution)
        criterion_args = _build_criterion_args(rfdetr.model_config, rfdetr.model.device)
        criterion, _ = build_criterion_and_postprocessors(criterion_args)
        self.criterion = criterion
        # ImageNet normalization constants — match RFDETR.predict() preprocessing.
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
        # Buffers are typed Tensor|Module by nn.Module — narrow back to Tensor.
        mean = cast(Tensor, self._mean)
        std = cast(Tensor, self._std)
        return (stacked - mean) / std

    def _criterion_loss(
        self, outputs: Dict[str, Any], targets_detr: List[Dict[str, Tensor]]
    ) -> Tensor:
        loss_dict = self.criterion(outputs, targets_detr)
        weight_dict = self.criterion.weight_dict
        device = outputs["pred_logits"].device
        zero = torch.zeros((), device=device)
        total: Tensor = sum(
            (loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict),
            zero,
        )
        return total

    def forward(self, x: DetectionBatchInput) -> DetectionStepOutput:
        if self.training:
            if x.targets is None:
                raise ValueError(
                    "RFDETRModel in train mode requires targets on DetectionBatchInput."
                )
            images = self._preprocess(x.images)
            targets_detr = _build_rfdetr_targets(
                x.targets, self.resolution, images.device
            )
            outputs = self.detector(images, targets_detr)
            loss_total = self._criterion_loss(outputs, targets_detr)
            n_imgs = len(x.images)
            box_counts = [int(t["boxes"].shape[0]) for t in x.targets]
            total_boxes = sum(box_counts)
            if total_boxes == 0:
                logger.warning(
                    f"[diag][rfdetr][train] batch sees ZERO target boxes across "
                    f"{n_imgs} image(s) — gradients on classifier head will be "
                    f"trivial. loss={float(loss_total):.4f}"
                )
            else:
                logger.trace(
                    f"[diag][rfdetr][train] images={n_imgs} target_boxes/img={box_counts} "
                    f"total_boxes={total_boxes} loss={float(loss_total):.4f}"
                )
            return DetectionStepOutput(losses={"loss": loss_total}, predictions=None)

        images = self._preprocess(x.images)
        outputs = self.detector(images)
        losses: Optional[Dict[str, Tensor]] = None
        if x.targets is not None and self.compute_val_loss:
            targets_detr = _build_rfdetr_targets(
                x.targets, self.resolution, images.device
            )
            with torch.no_grad():
                loss_total = self._criterion_loss(outputs, targets_detr)
            losses = {"loss": loss_total}

        target_sizes = torch.tensor(
            [[self.resolution, self.resolution]] * images.shape[0],
            device=images.device,
        )
        results = self.postprocess(outputs, target_sizes)
        predictions = _decode_rfdetr_predictions(
            results, self.num_classes, self.score_threshold
        )

        per_image_summaries = []
        for i, p in enumerate(predictions):
            n = int(p["boxes"].shape[0])
            if n == 0:
                per_image_summaries.append(f"img{i}=0")
                continue
            scores = p["scores"]
            per_image_summaries.append(
                f"img{i}={n}@s∈[{float(scores.min()):.3f},{float(scores.max()):.3f}]"
                f"(mean={float(scores.mean()):.3f})"
            )
        total_pred = sum(int(p["boxes"].shape[0]) for p in predictions)
        if total_pred == 0:
            logger.warning(
                f"[diag][rfdetr][eval] {len(predictions)} image(s): no predictions "
                f"survived score_threshold={self.score_threshold} — classifier head "
                f"may be undertrained, or threshold may be too high."
            )
        else:
            loss_summary = f"loss={float(losses['loss']):.4f}" if losses else "loss=n/a"
            logger.debug(
                f"[diag][rfdetr][eval] {' '.join(per_image_summaries)} | {loss_summary}"
            )
        return DetectionStepOutput(losses=losses, predictions=predictions)


__all__ = ["RFDETRModel"]
