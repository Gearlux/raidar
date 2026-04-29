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
into the unchanged base :class:`~marainer.trainer.Trainer`. We therefore drive
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

from types import SimpleNamespace
from typing import Dict, List, Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
from confluid import configurable
from logflow import get_logger
from marainer.outputs import DetectionOutput
from torch import Tensor

from raidar.checkpoint import load_lightning_checkpoint
from raidar.detection import DetectionBatchInput, DetectionStepOutput

logger = get_logger(__name__)


def _build_yolo26(
    variant: Literal["n", "s", "m", "l", "x"],
    num_classes: int,
    pretrained_detector: bool,
) -> nn.Module:
    """Instantiate a YOLO26 ``DetectionModel`` with ``num_classes`` outputs.

    Building from the YAML (``yolo26{variant}.yaml``) gives us full control
    over ``nc``; if ``pretrained_detector=True`` we then load the COCO-trained
    .pt and copy every shape-compatible parameter into the new model. The
    head's classification convolutions don't match (the .pt was trained for
    ``nc=80``) and are silently skipped, leaving the backbone, FPN, and
    regression head fully pretrained while the classifier starts random —
    the same body-pretrained / head-fresh pattern used by
    :func:`raidar.faster_rcnn._build_faster_rcnn`.
    """
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG_DICT
    from ultralytics.utils.downloads import attempt_download_asset

    detector = DetectionModel(f"yolo26{variant}.yaml", nc=num_classes, ch=3)

    if pretrained_detector:
        weights_path = attempt_download_asset(f"yolo26{variant}.pt")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        src_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        src_state = src_model.float().state_dict()
        dst_state = detector.state_dict()
        for k, v in src_state.items():
            if k in dst_state and v.shape == dst_state[k].shape:
                dst_state[k] = v
        detector.load_state_dict(dst_state)

    # The criterion accesses self.hyp.{box, cls, dfl, ...} via attribute lookup.
    # A YAML-built DetectionModel may have no .args at all; a .pt-loaded one
    # has .args as a dict from the task block. Either way, normalize to a
    # namespace populated from DEFAULT_CFG_DICT + whatever was present.
    existing = getattr(detector, "args", None)
    if existing is None:
        raw_args: Dict[str, object] = {}
    elif isinstance(existing, dict):
        raw_args = existing
    else:
        raw_args = vars(existing)
    merged: Dict[str, object] = {**DEFAULT_CFG_DICT, **raw_args}
    # nn.Module.__setattr__ only types Tensor / Module / Parameter assignments;
    # bypass it via object.__setattr__ to attach the namespace cleanly.
    namespace = SimpleNamespace(**merged)
    object.__setattr__(detector, "args", namespace)
    object.__setattr__(detector, "hyp", namespace)

    return detector


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
    bn_layers = [m for m in detector.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
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
) -> List[DetectionOutput]:
    """Convert Ultralytics' ``[B, N, 6]`` end-to-end output to per-image
    :class:`~marainer.outputs.DetectionOutput` dicts.

    Each row is ``[x1, y1, x2, y2, score, class_id]`` in input-pixel space
    (i.e. the ``imgsz``-square space the model saw). Boxes are clipped to
    ``[0, imgsz]`` and filtered by ``score_threshold``; coordinates are not
    remapped to any "original" image size because this wrapper requires
    images to arrive pre-resized to ``imgsz``.
    """
    predictions: List[DetectionOutput] = []
    for b in range(raw.shape[0]):
        rows = raw[b]
        scores = rows[:, 4]
        keep = scores >= score_threshold
        rows = rows[keep]
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
    ) -> None:
        super().__init__()
        if imgsz % 32 != 0:
            raise ValueError(f"imgsz must be a multiple of 32 (YOLO stride), got {imgsz}.")
        self.num_classes = int(num_classes)
        self.variant = variant
        self.imgsz = int(imgsz)
        self.score_threshold = float(score_threshold)
        self.agnostic_nms = bool(agnostic_nms)
        self.compute_val_loss = bool(compute_val_loss)
        self.detector = _build_yolo26(
            variant=variant,
            num_classes=self.num_classes,
            pretrained_detector=pretrained_detector,
        )
        self.detector.set_head_attr(agnostic_nms=self.agnostic_nms)
        if checkpoint_path:
            load_lightning_checkpoint(self, checkpoint_path, strict=checkpoint_strict)

    def forward(self, x: DetectionBatchInput) -> DetectionStepOutput:
        if self.training:
            if x.targets is None:
                raise ValueError("YOLO26Model in train mode requires targets on DetectionBatchInput.")
            batch_dict = _build_yolo_batch_dict(x.images, x.targets, self.imgsz)
            loss_components, _items = self.detector(batch_dict)
            loss_total = loss_components.sum()
            n_imgs = len(x.images)
            box_counts = [int(t["boxes"].shape[0]) for t in x.targets]
            total_boxes = sum(box_counts)
            if total_boxes == 0:
                logger.warning(
                    f"[diag][yolo26][train] batch sees ZERO target boxes across "
                    f"{n_imgs} image(s) — gradients on classifier head will be "
                    f"trivial. loss={float(loss_total):.4f}"
                )
            else:
                logger.trace(
                    f"[diag][yolo26][train] images={n_imgs} target_boxes/img={box_counts} "
                    f"total_boxes={total_boxes} loss={float(loss_total):.4f}"
                )
            return DetectionStepOutput(losses={"loss": loss_total}, predictions=None)

        losses: Optional[Dict[str, Tensor]] = None
        if x.targets is not None and self.compute_val_loss:
            batch_dict = _build_yolo_batch_dict(x.images, x.targets, self.imgsz)
            loss_components, _items = _compute_yolo_loss_without_bn_drift(self.detector, batch_dict)
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
            scores = p["scores"]
            per_image_summaries.append(
                f"img{i}={n}@s∈[{float(scores.min()):.3f},{float(scores.max()):.3f}]"
                f"(mean={float(scores.mean()):.3f})"
            )
        total_pred = sum(int(p["boxes"].shape[0]) for p in predictions)
        if total_pred == 0:
            logger.warning(
                f"[diag][yolo26][eval] {len(predictions)} image(s): no predictions "
                f"survived score_threshold={self.score_threshold} — classifier head "
                f"may be undertrained, or threshold may be too high."
            )
        else:
            loss_summary = f"loss={float(losses['loss']):.4f}" if losses else "loss=n/a"
            logger.debug(f"[diag][yolo26][eval] {' '.join(per_image_summaries)} | {loss_summary}")
        return DetectionStepOutput(losses=losses, predictions=predictions)


__all__ = ["YOLO26Model"]
