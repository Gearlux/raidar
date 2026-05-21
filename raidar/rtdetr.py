"""Ultralytics RT-DETR — torchvision-native wrapper.

Exposes the same forward contract as torchvision's detection models:

* ``model.train(); model(images, targets) → Dict[str, Tensor]`` — single-key
  loss dict ``{"loss": total}``.
* ``model.eval();  model(images)         → List[Dict[str, Tensor]]`` — per-image
  ``{boxes, scores, labels}`` in xyxy-pixel space.

RT-DETR (Real-Time DEtection TRansformer) is Ultralytics' transformer-based
NMS-free detector. We drive the underlying ``RTDETRDetectionModel`` directly
(rather than the high-level ``ultralytics.RTDETR`` class) so raidar plugs into
Lightning instead of ultralytics' bundled trainer / validator / predictor.

Inference contract: ``RTDETRDetectionModel(img_tensor)`` returns
``(y, x)`` where ``y`` has shape ``[B, 300, 4 + nc]`` — normalized cxcywh
boxes concatenated with per-class sigmoid probabilities. We argmax over
class, threshold by score, convert cxcywh→xyxy, and lift normalized→pixel via
``imgsz``.

Image preprocessing: assumes all images are already ``[3, imgsz, imgsz]``.

Marainer-generic flow (``DetectionStepOutput`` + ``compute_val_loss``) lives
in :mod:`raidar.rtdetr_marainer`.
"""

from types import SimpleNamespace
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn
from confluid import configurable
from logflow import get_logger
from marainer.outputs import DetectionOutput
from torch import Tensor

from raidar.checkpoint import load_lightning_checkpoint

logger = get_logger(__name__)

_RESNET_VARIANTS = {"resnet50", "resnet101"}


def _build_rtdetr(
    variant: Literal["l", "x", "resnet50", "resnet101"],
    num_classes: int,
    pretrained_detector: bool,
) -> nn.Module:
    """Instantiate an ``RTDETRDetectionModel`` with ``num_classes`` outputs.

    Building from the YAML (``rtdetr-{variant}.yaml``) gives us full control
    over ``nc``; if ``pretrained_detector=True`` we load the COCO-trained .pt
    and copy every shape-compatible parameter into the new model.
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
    object.__setattr__(detector, "nc", int(num_classes))

    if pretrained_detector:
        weights_path = attempt_download_asset(f"rtdetr-{variant}.pt")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        src_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        src_state = src_model.float().state_dict()
        dst_state = detector.state_dict()
        for k, v in src_state.items():
            if k in dst_state and v.shape == dst_state[k].shape:
                dst_state[k] = v
        detector.load_state_dict(dst_state)

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


def _build_rtdetr_batch_dict(
    images: List[Tensor],
    targets: List[Dict[str, Tensor]],
    imgsz: int,
) -> Dict[str, Tensor]:
    """Convert raidar's per-image image / target lists into Ultralytics' batch dict.

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
    """Torchvision-native RT-DETR wrapper for
    :class:`~raidar.lightning_trainer.LightningDetectionTrainer`.

    Forward contract:

    * **Training**: ``forward(images, targets) → {"loss": total}``.
    * **Eval**: ``forward(images) → List[Dict]`` of per-image predictions.

    Args:
        num_classes: Foreground class count. RT-DETR uses ``0..nc-1`` directly
            without a background class.
        variant: ``"l"`` (default) / ``"x"`` ship public COCO weights;
            ``"resnet50"`` / ``"resnet101"`` are architecture-only.
        pretrained_detector: Load full COCO-pretrained RT-DETR weights.
            Must be False for resnet variants.
        imgsz: Square input size (must be a multiple of 32).
        score_threshold: Eval-time confidence cutoff (RT-DETR always emits
            300 raw queries per image — this threshold is the only mechanism
            keeping MAP tractable on real data).
        checkpoint_path / checkpoint_strict: Optional Lightning ``.ckpt`` load.
    """

    def __init__(
        self,
        num_classes: int = 1,
        variant: Literal["l", "x", "resnet50", "resnet101"] = "l",
        pretrained_detector: bool = True,
        imgsz: int = 640,
        score_threshold: float = 0.25,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
    ) -> None:
        super().__init__()
        if imgsz % 32 != 0:
            raise ValueError(f"imgsz must be a multiple of 32 (RT-DETR stride), got {imgsz}.")
        self.num_classes = int(num_classes)
        self.variant = variant
        self.imgsz = int(imgsz)
        self.score_threshold = float(score_threshold)
        self.detector = _build_rtdetr(
            variant=variant,
            num_classes=self.num_classes,
            pretrained_detector=pretrained_detector,
        )
        if checkpoint_path:
            load_lightning_checkpoint(self, checkpoint_path, strict=checkpoint_strict)

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[Dict[str, Tensor], List[DetectionOutput]]:
        if self.training:
            if targets is None:
                raise ValueError("RTDETRModel in train mode requires targets.")
            batch_dict = _build_rtdetr_batch_dict(images, targets, self.imgsz)
            loss_total, _items = self.detector(batch_dict)
            box_counts = [int(t["boxes"].shape[0]) for t in targets]
            total_boxes = sum(box_counts)
            if total_boxes == 0:
                logger.trace(
                    f"[diag][rtdetr][train] batch sees ZERO target boxes across "
                    f"{len(images)} image(s). loss={float(loss_total):.4f}"
                )
            else:
                logger.trace(
                    f"[diag][rtdetr][train] images={len(images)} target_boxes/img={box_counts} "
                    f"total_boxes={total_boxes} loss={float(loss_total):.4f}"
                )
            return {"loss": loss_total}

        img_tensor = torch.stack(images)
        raw, _ = self.detector(img_tensor)
        predictions = _decode_rtdetr_predictions(raw, self.imgsz, self.score_threshold)
        return predictions


__all__ = ["RTDETRModel", "_build_rtdetr_batch_dict", "_decode_rtdetr_predictions"]
