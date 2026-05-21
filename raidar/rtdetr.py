"""Ultralytics RT-DETR wrapped for the Marainer Trainer flow.

RT-DETR (Real-Time DEtection TRansformer) is Ultralytics' transformer-based
NMS-free detector. This module wraps the underlying ``RTDETRDetectionModel``
(a plain ``nn.Module``) into the same ``DetectionStepOutput`` contract as
:class:`raidar.yolo26.YOLO26Model` and :class:`raidar.faster_rcnn.FasterRCNNModel`,
so the existing generic adapters
(:class:`~raidar.detection.TorchvisionDetectionLoss`,
:class:`~raidar.detection.DetectionMetricAdapter`,
:func:`~raidar.detection.detection_collate_fn`) work unchanged.

The high-level ``ultralytics.RTDETR`` class bundles its own trainer / validator /
predictor — none of which we want, since raidar's whole value-add is plugging
into the unchanged base :class:`~marainer.trainer.LightningTrainer`. We therefore drive
the underlying ``RTDETRDetectionModel`` directly.

Loss contract: unlike YOLO26's ``(per_component_tensor[3], detached_items[3])``,
``RTDETRDetectionModel.loss(batch_dict)`` already returns a 0-dim scalar as
the first element (``sum(loss.values())`` over the giou/class/bbox dict) plus
a 3-vector of detached items for logging. We pack that scalar as
``losses={"loss": total}`` so :class:`~raidar.detection.TorchvisionDetectionLoss`
passes it through unchanged at default weight 1.0.

Inference contract: ``RTDETRDetectionModel(img_tensor)`` returns
``(y, x)`` where ``y`` has shape ``[B, 300, 4 + nc]`` — normalized cxcywh
boxes concatenated with per-class sigmoid probabilities. We split, argmax
over the class dim, threshold, and lift normalized → pixel via ``imgsz``,
mirroring :class:`ultralytics.models.rtdetr.predict.RTDETRPredictor.postprocess`.

Image preprocessing: this wrapper assumes all images in a
:class:`~raidar.detection.DetectionBatchInput` are already the same square
size (``imgsz`` × ``imgsz``). Variable-size images must be resized upstream
(e.g. via a Waivefront op) — keeping that responsibility outside the model
keeps the wrapper small and avoids hiding letterbox math inside ``forward``.

Typical wiring (YAML)::

    model: !class:raidar.rtdetr.RTDETRModel
      num_classes: 2
      variant: l
      pretrained_detector: true
      imgsz: 640
    loss_fn: !class:raidar.detection.TorchvisionDetectionLoss
      # No `weights` block — single "loss" key passes through with weight 1.0.
"""

from types import SimpleNamespace
from typing import Dict, List, Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor

from confluid import configurable
from logflow import get_logger
from marainer.outputs import DetectionOutput
from raidar.checkpoint import load_lightning_checkpoint
from raidar.detection import DetectionBatchInput, DetectionStepOutput

logger = get_logger(__name__)

_RESNET_VARIANTS = {"resnet50", "resnet101"}


def _build_rtdetr(
    variant: Literal["l", "x", "resnet50", "resnet101"],
    num_classes: int,
    pretrained_detector: bool,
) -> nn.Module:
    """Instantiate an ``RTDETRDetectionModel`` with ``num_classes`` outputs.

    Building from the YAML (``rtdetr-{variant}.yaml``) gives us full control
    over ``nc``; if ``pretrained_detector=True`` we then load the COCO-trained
    .pt and copy every shape-compatible parameter into the new model. The
    decoder's classification heads don't match (the .pt was trained for
    ``nc=80``) and are silently skipped, leaving the backbone, encoder, and
    bbox heads fully pretrained while the classifier starts random — the same
    body-pretrained / head-fresh pattern used by
    :func:`raidar.yolo26._build_yolo26` and
    :func:`raidar.faster_rcnn._build_faster_rcnn`.

    Only the ``l`` and ``x`` variants ship with public pretrained weights.
    Requesting ``pretrained_detector=True`` for ``resnet50`` / ``resnet101``
    raises ``ValueError`` upfront rather than failing later inside
    :func:`ultralytics.utils.downloads.attempt_download_asset` with a generic
    404.
    """
    from ultralytics.nn.tasks import RTDETRDetectionModel
    from ultralytics.utils import DEFAULT_CFG_DICT
    from ultralytics.utils.downloads import attempt_download_asset

    if pretrained_detector and variant in _RESNET_VARIANTS:
        raise ValueError(
            f"RT-DETR '{variant}' has no public pretrained weights — "
            "pass pretrained_detector=False, or use variant='l' or 'x'."
        )

    detector = RTDETRDetectionModel(f"rtdetr-{variant}.yaml", nc=num_classes, ch=3)
    # DetectionModel only stores nc inside self.yaml["nc"]; RTDETRDetectionModel.init_criterion
    # reads self.nc directly, which would otherwise AttributeError at first loss call.
    object.__setattr__(detector, "nc", int(num_classes))

    if pretrained_detector:
        weights_path = attempt_download_asset(f"rtdetr-{variant}.pt")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        src_model = (
            ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        )
        src_state = src_model.float().state_dict()
        dst_state = detector.state_dict()
        for k, v in src_state.items():
            if k in dst_state and v.shape == dst_state[k].shape:
                dst_state[k] = v
        detector.load_state_dict(dst_state)

    # RT-DETR's loss is self-contained (no self.hyp lookups) but other
    # ultralytics utility code paths (info(), fuse(), ...) opportunistically
    # reach for model.args. Attach the same DEFAULT_CFG_DICT-merged namespace
    # YOLO26 uses for structural parity and defensive consistency.
    existing = getattr(detector, "args", None)
    if existing is None:
        raw_args: Dict[str, object] = {}
    elif isinstance(existing, dict):
        raw_args = existing
    else:
        raw_args = vars(existing)
    merged: Dict[str, object] = {**DEFAULT_CFG_DICT, **raw_args}
    namespace = SimpleNamespace(**merged)
    object.__setattr__(detector, "args", namespace)
    object.__setattr__(detector, "hyp", namespace)

    return detector


def _compute_rtdetr_loss_without_bn_drift(
    detector: nn.Module,
    batch_dict: Dict[str, Tensor],
) -> Tuple[Tensor, Tensor]:
    """Run RT-DETR's training loss path under eval-style BN behavior.

    Mirrors :func:`raidar.yolo26._compute_yolo_loss_without_bn_drift`:
    the loss path runs a forward pass which would update BN running stats
    when the model is in train mode. To make ``val/loss`` directly comparable
    to ``train/loss`` without polluting the running stats, we briefly flip to
    train mode but force every BN layer back to ``eval()`` for the duration.
    Grad is disabled — we only want the scalar.

    Return shape differs from YOLO26: the first element is **already a
    0-dim scalar** (``sum(loss.values())`` from ``RTDETRDetectionModel.loss``),
    not a per-component ``[3]`` tensor.
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


def _build_rtdetr_batch_dict(
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

    Output (Ultralytics convention — identical for YOLO and RT-DETR):

    * ``img``: ``[B, 3, imgsz, imgsz]`` float tensor.
    * ``cls``: ``[total_boxes, 1]`` float32 class indices.
    * ``bboxes``: ``[total_boxes, 4]`` float32 cxcywh **normalized** to [0, 1].
    * ``batch_idx``: ``[total_boxes]`` float32 image indices.
    """
    for i, img in enumerate(images):
        if img.shape[-1] != imgsz or img.shape[-2] != imgsz:
            raise ValueError(
                f"RTDETRModel requires all images to be ({imgsz}, {imgsz}); "
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


def _decode_rtdetr_predictions(
    raw: Tensor,
    imgsz: int,
    score_threshold: float,
) -> List[DetectionOutput]:
    """Convert RT-DETR's ``[B, 300, 4 + nc]`` decoder output to per-image
    :class:`~marainer.outputs.DetectionOutput` dicts.

    The last dim splits into 4 normalized-cxcywh box coords plus ``nc``
    per-class sigmoid probabilities. We argmax over the class dim, filter
    by ``score_threshold``, convert cxcywh → xyxy via
    :func:`ultralytics.utils.ops.xywh2xyxy`, and lift normalized → pixel
    space by multiplying by ``imgsz`` (this wrapper requires images to
    arrive pre-resized to ``imgsz``-square so the lift is exact).

    Mirrors the canonical decode in
    :class:`ultralytics.models.rtdetr.predict.RTDETRPredictor.postprocess`.
    """
    from ultralytics.utils.ops import xywh2xyxy

    nc_dim = raw.shape[-1] - 4
    bboxes_norm_cxcywh, scores_per_class = raw.split((4, nc_dim), dim=-1)
    predictions: List[DetectionOutput] = []
    for b in range(raw.shape[0]):
        bbox = xywh2xyxy(bboxes_norm_cxcywh[b])
        max_score, cls = scores_per_class[b].max(-1)
        keep = max_score >= score_threshold
        bbox = (bbox[keep] * float(imgsz)).clamp_(0.0, float(imgsz))
        predictions.append(
            DetectionOutput(
                boxes=bbox.float(),
                scores=max_score[keep].float(),
                labels=cls[keep].long(),
            )
        )
    return predictions


@configurable
class RTDETRModel(nn.Module):
    """Ultralytics RT-DETR wrapped for the Marainer Trainer flow.

    Returns a :class:`~raidar.detection.DetectionStepOutput` in every mode:

    * **Training**: ``{losses: {"loss": total}, predictions: None}`` — the
      single ``"loss"`` key holds RT-DETR's canonical scalar
      (``sum(loss.values())`` over giou / class / bbox), already weighted by
      the loss's internal gain hyperparameters.
    * **Validation / test** with targets: ``{losses: {"loss": total},
      predictions: [...]}`` via a BN-frozen dual-forward (mirrors
      :class:`raidar.yolo26.YOLO26Model`). ``val/loss`` is directly
      comparable to ``train/loss``.
    * **Pure prediction** (no targets): ``{losses: None, predictions: [...]}``.

    Args:
        num_classes: Foreground class count. RT-DETR uses ``0..nc-1``
            directly **without** a background class — same convention as
            YOLO, unlike Faster R-CNN where class 0 is reserved.
        variant: Model size — ``"l"`` (large, default) or ``"x"`` (xlarge),
            both with public COCO weights; ``"resnet50"`` / ``"resnet101"``
            for the original Baidu RT-DETR backbones (architecture only,
            no public pretrained .pt — set ``pretrained_detector=False``).
        pretrained_detector: Load full COCO-pretrained RT-DETR weights
            (downloads on first use). When ``num_classes != 80``, the
            decoder's classification heads don't shape-match and are
            skipped — backbone + encoder + bbox heads stay pretrained,
            classifier starts random. Strongly recommended for small-dataset
            training. Must be False for resnet variants.
        imgsz: Square input size (must be a multiple of 32). All images in
            the batch must already be ``[3, imgsz, imgsz]``; resize upstream.
        score_threshold: Eval-time confidence cutoff applied to the per-query
            max class probability before packing into :class:`DetectionOutput`.
            Important: RT-DETR always emits 300 raw queries per image, so
            this threshold is the only mechanism keeping ``MeanAveragePrecision``
            tractable on real data.
        compute_val_loss: When False, skips the loss recomputation during
            eval. Defaults to True.
        checkpoint_path: Optional path to a Lightning ``.ckpt`` (or plain
            torch state-dict) saved by a previous training run. Loaded after
            ``pretrained_detector`` weights so checkpoint values win.
            ``"model."`` prefix from Lightning wrapping is stripped
            automatically.
        checkpoint_strict: Forwarded to ``load_state_dict(strict=...)``.
            Defaults to False so e.g. a re-trained classifier head with a
            different ``num_classes`` doesn't reject the whole load.
    """

    def __init__(
        self,
        num_classes: int = 1,
        variant: Literal["l", "x", "resnet50", "resnet101"] = "l",
        pretrained_detector: bool = True,
        imgsz: int = 640,
        score_threshold: float = 0.25,
        compute_val_loss: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
    ) -> None:
        super().__init__()
        if imgsz % 32 != 0:
            raise ValueError(
                f"imgsz must be a multiple of 32 (RT-DETR stride), got {imgsz}."
            )
        self.num_classes = int(num_classes)
        self.variant = variant
        self.imgsz = int(imgsz)
        self.score_threshold = float(score_threshold)
        self.compute_val_loss = bool(compute_val_loss)
        self.detector = _build_rtdetr(
            variant=variant,
            num_classes=self.num_classes,
            pretrained_detector=pretrained_detector,
        )
        if checkpoint_path:
            load_lightning_checkpoint(self, checkpoint_path, strict=checkpoint_strict)

    def forward(self, x: DetectionBatchInput) -> DetectionStepOutput:
        if self.training:
            if x.targets is None:
                raise ValueError(
                    "RTDETRModel in train mode requires targets on DetectionBatchInput."
                )
            batch_dict = _build_rtdetr_batch_dict(x.images, x.targets, self.imgsz)
            loss_total, _items = self.detector(batch_dict)
            n_imgs = len(x.images)
            box_counts = [int(t["boxes"].shape[0]) for t in x.targets]
            total_boxes = sum(box_counts)
            if total_boxes == 0:
                logger.warning(
                    f"[diag][rtdetr][train] batch sees ZERO target boxes across "
                    f"{n_imgs} image(s) — gradients on classifier head will be "
                    f"trivial. loss={float(loss_total):.4f}"
                )
            else:
                logger.trace(
                    f"[diag][rtdetr][train] images={n_imgs} target_boxes/img={box_counts} "
                    f"total_boxes={total_boxes} loss={float(loss_total):.4f}"
                )
            return DetectionStepOutput(losses={"loss": loss_total}, predictions=None)

        losses: Optional[Dict[str, Tensor]] = None
        if x.targets is not None and self.compute_val_loss:
            batch_dict = _build_rtdetr_batch_dict(x.images, x.targets, self.imgsz)
            loss_total, _items = _compute_rtdetr_loss_without_bn_drift(
                self.detector, batch_dict
            )
            losses = {"loss": loss_total}

        img_tensor = torch.stack(x.images)
        raw, _intermediate = self.detector(img_tensor)
        predictions = _decode_rtdetr_predictions(raw, self.imgsz, self.score_threshold)

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
                f"[diag][rtdetr][eval] {len(predictions)} image(s): no predictions "
                f"survived score_threshold={self.score_threshold} — classifier head "
                f"may be undertrained, or threshold may be too high."
            )
        else:
            loss_summary = f"loss={float(losses['loss']):.4f}" if losses else "loss=n/a"
            logger.debug(
                f"[diag][rtdetr][eval] {' '.join(per_image_summaries)} | {loss_summary}"
            )
        return DetectionStepOutput(losses=losses, predictions=predictions)


__all__ = ["RTDETRModel"]
