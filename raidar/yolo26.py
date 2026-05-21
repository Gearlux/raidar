"""Ultralytics YOLO26 wrapped for the Marainer Trainer flow.

YOLO26 (released January 2026) is Ultralytics' end-to-end NMS-free detector
optimized for edge inference. This module wraps the underlying
``DetectionModel`` (a plain ``nn.Module``) into the same ``DetectionStepOutput``
contract as :class:`raidar.faster_rcnn.FasterRCNNModel`, so the existing
generic adapters (:class:`~raidar.detection.TorchvisionDetectionLoss`,
:class:`~raidar.detection.DetectionMetricAdapter`,
:func:`~raidar.detection.detection_collate_fn`) work unchanged.

The high-level ``ultralytics.YOLO`` class bundles its own trainer / validator /
predictor — none of which we want, since raidar's whole value-add is plugging
into the unchanged base :class:`~marainer.trainer.LightningTrainer`. We therefore drive
the underlying ``DetectionModel`` directly.

Loss contract: Ultralytics' ``DetectionModel.loss(batch_dict)`` returns a
``(per_component_tensor[3], detached_items[3])`` pair where
``per_component_tensor.sum()`` is the canonical scalar to backprop (the
trainer's own ``self.loss = loss.sum()`` line). We pack this scalar as
``losses={"loss": total}`` so :class:`~raidar.detection.TorchvisionDetectionLoss`
passes it through unchanged at default weight 1.0 — no YOLO-specific loss
class needed.

Image preprocessing: this wrapper assumes all images in a
:class:`~raidar.detection.DetectionBatchInput` are already the same square
size (``imgsz`` × ``imgsz``). Variable-size images must be resized upstream
(e.g. via a Waivefront op) — keeping that responsibility outside the model
keeps the wrapper small and avoids hiding letterbox math inside ``forward``.

Typical wiring (YAML)::

    model: !class:raidar.yolo26.YOLO26Model
      num_classes: 2
      variant: n
      pretrained_detector: true
      imgsz: 640
    loss_fn: !class:raidar.detection.TorchvisionDetectionLoss
      # No `weights` block — single "loss" key passes through with weight 1.0.
"""

from typing import Dict, List, Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor

from confluid import configurable
from logflow import get_logger
from marainer.outputs import DetectionOutput
from raidar.checkpoint import load_lightning_checkpoint
from raidar.detection import DetectionBatchInput, DetectionStepOutput
from raidar.ultralytics_common import build_yolo26 as _build_yolo26

logger = get_logger(__name__)


def _compute_yolo_loss_without_bn_drift(
    detector: nn.Module,
    batch_dict: Dict[str, Tensor],
) -> Tuple[Tensor, Tensor]:
    """Run YOLO's training loss path under eval-style BN behavior.

    Mirrors :func:`raidar.faster_rcnn._compute_losses_without_bn_drift`:
    Ultralytics' ``DetectionModel.loss`` only emits its loss tensors after
    a forward pass, and that forward pass updates BN running stats when the
    model is in train mode. To make ``val/loss`` directly comparable to
    ``train/loss`` without polluting the running stats, we briefly flip to
    train mode but force every BN layer back to ``eval()`` for the duration.
    Grad is disabled — we only want the scalar.
    """
    was_training = detector.training
    detector.train()
    bn_layers = [
        m for m in detector.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    for m in bn_layers:
        m.eval()
    try:
        with torch.no_grad():
            return cast(Tuple[Tensor, Tensor], detector(batch_dict))
    finally:
        if not was_training:
            detector.eval()


def _build_yolo_batch_dict(
    images: List[Tensor],
    targets: List[Dict[str, Tensor]],
    imgsz: int,
) -> Dict[str, Tensor]:
    """Convert raidar's per-image image / target lists into Ultralytics' batch dict.

    Inputs:

    * ``images``: list of ``[3, imgsz, imgsz]`` float tensors (this wrapper
      requires already-square pre-resized images; see module docstring).
    * ``targets``: list of per-image ``{"boxes": [N, 4] xyxy-pixel float32,
      "labels": [N] int64}`` dicts (raidar / torchvision convention).

    Output (Ultralytics convention):

    * ``img``: ``[B, 3, imgsz, imgsz]`` float tensor.
    * ``cls``: ``[total_boxes, 1]`` float32 class indices.
    * ``bboxes``: ``[total_boxes, 4]`` float32 cxcywh **normalized** to [0, 1].
    * ``batch_idx``: ``[total_boxes]`` float32 image indices.
    """
    for i, img in enumerate(images):
        if img.shape[-1] != imgsz or img.shape[-2] != imgsz:
            raise ValueError(
                f"YOLO26Model requires all images to be ({imgsz}, {imgsz}); "
                f"got image[{i}].shape={tuple(img.shape)}. Resize upstream."
            )

    cls_list: List[Tensor] = []
    box_list: List[Tensor] = []
    idx_list: List[Tensor] = []
    for i, t in enumerate(targets):
        boxes_xyxy = t["boxes"].float()
        labels = t["labels"].float().unsqueeze(-1)
        n = boxes_xyxy.shape[0]
        if n == 0:
            continue
        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / (2.0 * imgsz)
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / (2.0 * imgsz)
        w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / imgsz
        h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / imgsz
        cls_list.append(labels)
        box_list.append(torch.stack([cx, cy, w, h], dim=1))
        idx_list.append(torch.full((n,), float(i)))

    img_tensor = torch.stack(images)
    if cls_list:
        return {
            "img": img_tensor,
            "cls": torch.cat(cls_list, dim=0).to(img_tensor.device),
            "bboxes": torch.cat(box_list, dim=0).to(img_tensor.device),
            "batch_idx": torch.cat(idx_list, dim=0).to(img_tensor.device),
        }
    # Empty-target batch — model's loss path still tolerates zero-row tensors.
    return {
        "img": img_tensor,
        "cls": torch.zeros((0, 1), device=img_tensor.device),
        "bboxes": torch.zeros((0, 4), device=img_tensor.device),
        "batch_idx": torch.zeros((0,), device=img_tensor.device),
    }


def _decode_yolo_predictions(
    raw: Tensor,
    score_threshold: float,
    iou_threshold: float = 0.7,
    max_det: int = 300,
) -> List[DetectionOutput]:
    """Convert Ultralytics' detection-head output to per-image
    :class:`~marainer.outputs.DetectionOutput` dicts.

    Handles both head modes via :func:`ultralytics.utils.nms.non_max_suppression`:

    * **end2end=True**: ``raw`` is ``(B, max_det, 6)`` — already top-K'd
      inside the head. NMS short-circuits to a confidence filter.
    * **end2end=False**: ``raw`` is ``(B, 4 + nc, num_anchors)`` — the
      classic pre-NMS layout. NMS does the full xywh→xyxy + IoU dedup pass.

    Each output row is ``[x1, y1, x2, y2, score, class_id]`` in input-pixel
    space (the ``imgsz``-square space the model saw). Coordinates are not
    remapped to any "original" image size because this wrapper requires
    images to arrive pre-resized to ``imgsz``.
    """
    from ultralytics.utils.nms import non_max_suppression

    nms_out = non_max_suppression(
        raw, conf_thres=score_threshold, iou_thres=iou_threshold, max_det=max_det
    )
    predictions: List[DetectionOutput] = []
    for rows in nms_out:
        predictions.append(
            DetectionOutput(
                boxes=rows[:, :4].float(),
                scores=rows[:, 4].float(),
                labels=rows[:, 5].long(),
            )
        )
    return predictions


@configurable
class YOLO26Model(nn.Module):
    """Ultralytics YOLO26 wrapped for the Marainer Trainer flow.

    Returns a :class:`~raidar.detection.DetectionStepOutput` in every mode:

    * **Training**: ``{losses: {"loss": total}, predictions: None}`` — the
      single ``"loss"`` key holds Ultralytics' canonical balanced scalar
      (``per_component_tensor.sum()``), already weighted by Ultralytics'
      internal box / cls / dfl gain hyperparameters.
    * **Validation / test** with targets: ``{losses: {"loss": total},
      predictions: [...]}`` via a BN-frozen dual-forward (mirrors
      :class:`raidar.faster_rcnn.FasterRCNNModel`). ``val/loss`` is
      directly comparable to ``train/loss``.
    * **Pure prediction** (no targets): ``{losses: None, predictions: [...]}``.

    Args:
        num_classes: Foreground class count. YOLO uses ``0..nc-1`` directly
            **without** a background class — unlike Faster R-CNN where class
            0 is reserved for background.
        variant: Model size — ``"n"`` (nano, ~2.5M params) through ``"x"``
            (xlarge). Default ``"n"``.
        pretrained_detector: Load full COCO-pretrained YOLO26 weights
            (downloads on first use). When ``num_classes != 80``, the
            classification convolutions in the head don't shape-match and
            are skipped — backbone + neck + regression head stay
            pretrained, classifier starts random. Strongly recommended for
            small-dataset training.
        imgsz: Square input size (must be a multiple of 32). All images in
            the batch must already be ``[3, imgsz, imgsz]``; resize upstream.
        score_threshold: Eval-time confidence cutoff applied to the
            ``[B, N, 6]`` end-to-end output before packing into
            :class:`DetectionOutput`.
        agnostic_nms: Class-agnostic top-k selection in the end-to-end head
            — pick the top ``max_det`` boxes by their best class score
            regardless of class id, instead of considering all classes
            jointly. Forwarded to ``Detect.agnostic_nms`` via
            ``DetectionModel.set_head_attr`` so it applies to both the
            internal ``end2end`` path and the legacy multi-class top-k path.
            Defaults to False.
        compute_val_loss: When False, skips the loss recomputation during
            eval. Defaults to True.
        checkpoint_path: Optional path to a Lightning ``.ckpt`` (or plain
            torch state-dict) saved by a previous training run. Loaded after
            ``pretrained_detector`` weights so checkpoint values win.
            ``"model."`` prefix from Lightning wrapping is stripped
            automatically. Use this to resume training from a checkpoint,
            or to plug a trained model into the Evaluator without going
            through Marainer.
        checkpoint_strict: Forwarded to ``load_state_dict(strict=...)``.
            Defaults to False so e.g. a re-trained classifier head with a
            different ``num_classes`` doesn't reject the whole load.
        end2end: Toggle YOLO26's NMS-free one2one detection head. ``None``
            (default) leaves whatever the model YAML specifies — for
            ``yolo26{variant}.yaml`` that's ``True``. **Must match how the
            checkpoint was trained.** Training with ``end2end=False`` (e.g.
            to dodge the MPS bool-indexing bug in ``E2EDetectLoss``) only
            updates the one2many head; the one2one head stays at its
            random init. Predicting from such a checkpoint with
            ``end2end=True`` (the YAML default) routes inference through
            that random head and silently collapses confidences to ~0.
    """

    def __init__(
        self,
        num_classes: int = 1,
        variant: Literal["n", "s", "m", "l", "x"] = "n",
        pretrained_detector: bool = True,
        imgsz: int = 640,
        score_threshold: float = 0.25,
        agnostic_nms: bool = False,
        compute_val_loss: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
        end2end: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if imgsz % 32 != 0:
            raise ValueError(
                f"imgsz must be a multiple of 32 (YOLO stride), got {imgsz}."
            )
        self.num_classes = int(num_classes)
        self.variant = variant
        self.imgsz = int(imgsz)
        self.score_threshold = float(score_threshold)
        self.agnostic_nms = bool(agnostic_nms)
        self.compute_val_loss = bool(compute_val_loss)
        self.end2end = end2end
        self.detector = _build_yolo26(
            variant=variant,
            num_classes=self.num_classes,
            pretrained_detector=pretrained_detector,
            agnostic_nms=self.agnostic_nms,
            end2end=end2end,
        )
        if checkpoint_path:
            load_lightning_checkpoint(self, checkpoint_path, strict=checkpoint_strict)

    def forward(self, x: DetectionBatchInput) -> DetectionStepOutput:
        if self.training:
            if x.targets is None:
                raise ValueError(
                    "YOLO26Model in train mode requires targets on DetectionBatchInput."
                )
            batch_dict = _build_yolo_batch_dict(x.images, x.targets, self.imgsz)
            loss_components, _items = self.detector(batch_dict)
            loss_total = loss_components.sum()
            n_imgs = len(x.images)
            box_counts = [int(t["boxes"].shape[0]) for t in x.targets]
            total_boxes = sum(box_counts)
            loss_value = loss_total.detach().item()
            if total_boxes == 0:
                logger.warning(
                    f"[diag][yolo26][train] batch sees ZERO target boxes across "
                    f"{n_imgs} image(s) — gradients on classifier head will be "
                    f"trivial. loss={loss_value:.4f}"
                )
            else:
                logger.trace(
                    f"[diag][yolo26][train] images={n_imgs} target_boxes/img={box_counts} "
                    f"total_boxes={total_boxes} loss={loss_value:.4f}"
                )
            return DetectionStepOutput(losses={"loss": loss_total}, predictions=None)

        losses: Optional[Dict[str, Tensor]] = None
        if x.targets is not None and self.compute_val_loss:
            batch_dict = _build_yolo_batch_dict(x.images, x.targets, self.imgsz)
            loss_components, _items = _compute_yolo_loss_without_bn_drift(
                self.detector, batch_dict
            )
            losses = {"loss": loss_components.sum()}

        img_tensor = torch.stack(x.images)
        raw, _intermediate = self.detector(img_tensor)
        predictions = _decode_yolo_predictions(raw, self.score_threshold)

        per_image_summaries = []
        for i, p in enumerate(predictions):
            n = int(p["boxes"].shape[0])
            if n == 0:
                per_image_summaries.append(f"img{i}=0")
                continue
            scores = p["scores"].detach()
            per_image_summaries.append(
                f"img{i}={n}@s∈[{scores.min().item():.3f},{scores.max().item():.3f}]"
                f"(mean={scores.mean().item():.3f})"
            )
        total_pred = sum(int(p["boxes"].shape[0]) for p in predictions)
        if total_pred == 0:
            logger.trace(
                f"[diag][yolo26][eval] {len(predictions)} image(s): no predictions "
                f"survived score_threshold={self.score_threshold} — classifier head "
                f"may be undertrained, or threshold may be too high."
            )
        else:
            loss_summary = f"loss={float(losses['loss']):.4f}" if losses else "loss=n/a"
            logger.trace(
                f"[diag][yolo26][eval] {' '.join(per_image_summaries)} | {loss_summary}"
            )
        return DetectionStepOutput(losses=losses, predictions=predictions)


__all__ = ["YOLO26Model"]
