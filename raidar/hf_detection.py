"""Hugging Face transformers object detectors — torchvision-native wrappers.

Wraps any `transformers` object-detection model (`AutoModelForObjectDetection`)
so the specialised :class:`raidar.detector.LightningDetector` /
:class:`raidar.detector.LightningDetector` can drive it without an
adapter layer, exactly like :class:`raidar.rfdetr.RFDETRModel` and
``raidar_ultralytics.YOLO26Model``:

* ``model.train(); model(images, targets) → {"loss": total}`` — the model's own
  loss (HF detectors compute the Hungarian-matched DETR loss inside ``forward``
  when given ``labels``; we return the single scalar ``outputs.loss``).
* ``model.eval();  model(images)          → List[DetectionOutput]`` — per-image
  ``{boxes, scores, labels}`` in xyxy-pixel space, decoded by the model's own
  image processor (``post_process_object_detection``).

Two ergonomic presets ship here — :class:`DFINEModel` (D-FINE) and
:class:`RTDETRv2Model` (RT-DETRv2) — both Apache-2.0 and available straight from
the installed ``transformers`` (so they live in raidar *core*, unlike the AGPL
ultralytics family). The generic :class:`HFDetectionModel` accepts ANY HF
object-detection checkpoint by ``model_id`` (DETR, Deformable-DETR, RT-DETR,
YOLOS, …).

Design notes grounded by an empirical probe of ``transformers`` 5.x:

* **Input is plain ``[0, 1]`` — NO ImageNet normalisation.** The RT-DETR/D-FINE
  image processors run with ``do_normalize=False`` (only ``do_rescale`` 1/255),
  so the model expects ``[0, 1]`` pixels. raidar's op chain already produces that
  (``sampleflux.ops.torch.ToTensorOp(normalize=True)``); we feed it through
  unchanged. Do NOT subtract ImageNet mean/std here — that would double-process.
* **Images are resized inside the wrapper** (to a square ``image_size``, default
  640 — what these models were trained at), so the SAME op chain as the
  torchvision Faster R-CNN example drops in with no detection-aware resize op.
  Variable input sizes are fine: predictions are decoded back into each image's
  ORIGINAL pixel space via per-image ``target_sizes``, so they align with the
  ground-truth boxes the metric sees.
* **Labels are 0-indexed ``0..K-1``** (the model adds no background slot — like
  RF-DETR, the OPPOSITE of torchvision's background-at-0). Wire the target op
  with ``label_offset=0``, NOT ``+1``.
* **Custom class counts re-init the head:** ``from_pretrained(model_id,
  num_labels=K, ignore_mismatched_sizes=True)`` keeps the backbone / encoder /
  decoder pretrained and re-inits only the class head when ``K != 80``.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union, get_args

import torch
import torch.nn as nn
import torch.nn.functional as F
from confluid import configurable
from loggair import get_logger
from marainer.torch import load_lightning_checkpoint
from marainer.torch.outputs import DetectionOutput
from torch import Tensor

logger = get_logger(__name__)

#: D-FINE COCO checkpoints (Apache-2.0, org ``ustc-community``), keyed by the
#: closed ``DFINEModel.variant`` Literal. Verified loadable in ``transformers`` 5.x.
_DFINE_CHECKPOINTS: Dict[str, str] = {
    "nano": "ustc-community/dfine-nano-coco",
    "small": "ustc-community/dfine-small-coco",
    "medium": "ustc-community/dfine-medium-coco",
    "large": "ustc-community/dfine-large-coco",
    "xlarge": "ustc-community/dfine-xlarge-coco",
}

#: RT-DETRv2 checkpoints (Apache-2.0, org ``PekingU``), keyed by the closed
#: ``RTDETRv2Model.variant`` Literal (ResNet backbone depth + ``vd`` variant).
_RTDETRV2_CHECKPOINTS: Dict[str, str] = {
    "r18vd": "PekingU/rtdetr_v2_r18vd",
    "r34vd": "PekingU/rtdetr_v2_r34vd",
    "r50vd": "PekingU/rtdetr_v2_r50vd",
    "r101vd": "PekingU/rtdetr_v2_r101vd",
}

_DfineVariant = Literal["nano", "small", "medium", "large", "xlarge"]
_RTDETRv2Variant = Literal["r18vd", "r34vd", "r50vd", "r101vd"]


def _build_hf_detection_targets(
    targets: List[Dict[str, Tensor]],
    sizes: List[Tuple[int, int]],
    device: torch.device,
) -> List[Dict[str, Tensor]]:
    """Convert raidar xyxy-pixel targets to HF's ``{class_labels, boxes}`` format.

    HF detection models want, per image, ``class_labels`` (``[N]`` long) and
    ``boxes`` (``[N, 4]`` cxcywh **normalised** to ``[0, 1]``). Normalisation is
    scale-invariant, so we divide by each image's own ``(W, H)`` — independent of
    the square ``image_size`` the pixels are later resized to.

    Args:
        targets: Per-image ``{"boxes": [N, 4] xyxy-pixel, "labels": [N]}`` dicts.
        sizes: Per-image original ``(height, width)`` in pixels (aligned to ``targets``).
        device: Device for the produced tensors.
    """
    out: List[Dict[str, Tensor]] = []
    for t, (height, width) in zip(targets, sizes):
        boxes_xyxy = t["boxes"].to(device).float()
        labels = t["labels"].to(device).long()
        n = boxes_xyxy.shape[0]
        if n == 0:
            cxcywh = torch.zeros((0, 4), device=device, dtype=torch.float32)
        else:
            cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / (2.0 * width)
            cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / (2.0 * height)
            w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / width
            h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / height
            cxcywh = torch.stack([cx, cy, w, h], dim=1)
        out.append({"class_labels": labels, "boxes": cxcywh})
    return out


def _decode_hf_predictions(results: List[Dict[str, Tensor]]) -> List[DetectionOutput]:
    """Map ``post_process_object_detection`` output to per-image :class:`DetectionOutput`.

    The image processor has already applied the score threshold and rescaled the
    boxes to the ``target_sizes`` pixel space, so this is a typed repackage.
    """
    predictions: List[DetectionOutput] = []
    for r in results:
        predictions.append(
            DetectionOutput(
                boxes=r["boxes"].float(),
                scores=r["scores"].float(),
                labels=r["labels"].long(),
            )
        )
    return predictions


@configurable(task="detection", role="model")
class HFDetectionModel(nn.Module):
    """Torchvision-native wrapper around any ``transformers`` object detector.

    Forward contract (matches torchvision detection models):

    * **Training**: ``forward(images, targets) → {"loss": total}``.
    * **Eval**: ``forward(images) → List[DetectionOutput]`` of per-image
      ``{boxes, scores, labels}`` in each image's original xyxy-pixel space.

    Images may be any size — they are resized to a square ``image_size`` for the
    model and predictions are decoded back to the input size. Pixels must be
    ``[0, 1]`` float (the model's processor uses ``do_normalize=False``); raidar's
    ``ToTensorOp(normalize=True)`` produces exactly that. Labels are 0-indexed
    ``0..num_classes-1`` (``label_offset=0`` in the target op).

    Args:
        model_id: Any HF object-detection checkpoint id (e.g.
            ``"ustc-community/dfine-small-coco"``). Open-ended, so a bare ``str``.
        num_classes: Foreground class count K. When it differs from the
            checkpoint's, the class head is re-initialised
            (``ignore_mismatched_sizes=True``) and the rest stays pretrained.
        pretrained: Load the checkpoint's pretrained weights. When ``False`` the
            architecture is built from its config with random weights.
        image_size: Square side (pixels, multiple of 32) the model runs at.
        score_threshold: Eval-time confidence cutoff (applied by the processor's
            ``post_process_object_detection``).
        checkpoint_path: Optional Lightning ``.ckpt`` / state-dict to load after
            the pretrained weights (checkpoint values win).
        checkpoint_strict: Forwarded to ``load_state_dict(strict=...)``.
    """

    def __init__(
        self,
        model_id: str = "ustc-community/dfine-small-coco",
        num_classes: int = 1,
        pretrained: bool = True,
        image_size: int = 640,
        score_threshold: float = 0.5,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
    ) -> None:
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection

        super().__init__()
        if image_size % 32 != 0:
            raise ValueError(f"image_size must be a multiple of 32, got {image_size}.")
        self.model_id = model_id
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.score_threshold = float(score_threshold)

        if pretrained:
            self.model = AutoModelForObjectDetection.from_pretrained(
                model_id, num_labels=self.num_classes, ignore_mismatched_sizes=True
            )
        else:
            config = AutoConfig.from_pretrained(model_id, num_labels=self.num_classes)
            self.model = AutoModelForObjectDetection.from_config(config)
        # The image processor is a plain (non-Module) helper used only to decode
        # predictions; nn.Module stores it as an ordinary attribute.
        self.image_processor = AutoImageProcessor.from_pretrained(model_id)

        if checkpoint_path:
            load_lightning_checkpoint(self, checkpoint_path, strict=checkpoint_strict)

    def _preprocess(self, images: List[Tensor]) -> Tuple[Tensor, List[Tuple[int, int]]]:
        """Resize each image to a square ``image_size`` and stack; return original sizes.

        Returns ``(pixel_values [B, 3, S, S], sizes [(H, W), ...])``. Already-square
        ``image_size`` images skip the interpolation.
        """
        sizes: List[Tuple[int, int]] = []
        resized: List[Tensor] = []
        s = self.image_size
        for img in images:
            img = img.float()
            sizes.append((int(img.shape[-2]), int(img.shape[-1])))
            if img.shape[-2] != s or img.shape[-1] != s:
                img = F.interpolate(img.unsqueeze(0), size=(s, s), mode="bilinear", align_corners=False).squeeze(0)
            resized.append(img)
        return torch.stack(resized), sizes

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[Dict[str, Tensor], List[DetectionOutput]]:
        pixel_values, sizes = self._preprocess(images)

        if self.training:
            if targets is None:
                raise ValueError("HFDetectionModel in train mode requires targets.")
            labels = _build_hf_detection_targets(targets, sizes, pixel_values.device)
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            total_boxes = sum(int(t["boxes"].shape[0]) for t in targets)
            logger.trace(
                f"[diag][hf_detection][train] images={len(images)} total_boxes={total_boxes} "
                f"loss={float(outputs.loss.detach()):.4f}"
            )
            return {"loss": outputs.loss}

        outputs = self.model(pixel_values=pixel_values)
        target_sizes = torch.tensor(sizes, device=pixel_values.device)
        results = self.image_processor.post_process_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=target_sizes
        )
        predictions = _decode_hf_predictions(results)
        total_pred = sum(int(p["boxes"].shape[0]) for p in predictions)
        if total_pred == 0:
            logger.trace(
                f"[diag][hf_detection][eval] {len(predictions)} image(s): no predictions survived "
                f"score_threshold={self.score_threshold}."
            )
        return predictions


@configurable(task="detection", role="model")
class DFINEModel(HFDetectionModel):
    """D-FINE detector (Apache-2.0) preset over :class:`HFDetectionModel`.

    D-FINE (2024) is a real-time DETR with fine-grained distribution refinement —
    a strong, permissively-licensed alternative to the AGPL YOLO family.

    Args:
        num_classes: Foreground class count K (0-indexed labels; ``label_offset=0``).
        variant: Model size — ``"nano"`` / ``"small"`` (default) / ``"medium"`` /
            ``"large"`` / ``"xlarge"``, mapped to the ``ustc-community/dfine-*-coco``
            checkpoints.
        pretrained: Load COCO-pretrained weights (head re-init when ``K != 80``).
        image_size: Square input side (multiple of 32; D-FINE trains at 640).
        score_threshold: Eval-time confidence cutoff.
        checkpoint_path / checkpoint_strict: Optional Lightning ``.ckpt`` load.
    """

    def __init__(
        self,
        num_classes: int = 1,
        variant: _DfineVariant = "small",
        pretrained: bool = True,
        image_size: int = 640,
        score_threshold: float = 0.5,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
    ) -> None:
        if variant not in get_args(_DfineVariant):
            raise ValueError(f"variant must be one of {get_args(_DfineVariant)}, got {variant!r}.")
        self.variant = variant
        super().__init__(
            model_id=_DFINE_CHECKPOINTS[variant],
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=image_size,
            score_threshold=score_threshold,
            checkpoint_path=checkpoint_path,
            checkpoint_strict=checkpoint_strict,
        )


@configurable(task="detection", role="model")
class RTDETRv2Model(HFDetectionModel):
    """RT-DETRv2 detector (Apache-2.0) preset over :class:`HFDetectionModel`.

    RT-DETRv2 ("DETRs Beat YOLOs on Real-time Object Detection", improved
    baseline) is an NMS-free real-time transformer detector.

    Args:
        num_classes: Foreground class count K (0-indexed labels; ``label_offset=0``).
        variant: ResNet backbone — ``"r18vd"`` / ``"r34vd"`` / ``"r50vd"``
            (default) / ``"r101vd"``, mapped to the ``PekingU/rtdetr_v2_*`` checkpoints.
        pretrained: Load COCO-pretrained weights (head re-init when ``K != 80``).
        image_size: Square input side (multiple of 32; RT-DETRv2 trains at 640).
        score_threshold: Eval-time confidence cutoff.
        checkpoint_path / checkpoint_strict: Optional Lightning ``.ckpt`` load.
    """

    def __init__(
        self,
        num_classes: int = 1,
        variant: _RTDETRv2Variant = "r50vd",
        pretrained: bool = True,
        image_size: int = 640,
        score_threshold: float = 0.5,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = False,
    ) -> None:
        if variant not in get_args(_RTDETRv2Variant):
            raise ValueError(f"variant must be one of {get_args(_RTDETRv2Variant)}, got {variant!r}.")
        self.variant = variant
        super().__init__(
            model_id=_RTDETRV2_CHECKPOINTS[variant],
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=image_size,
            score_threshold=score_threshold,
            checkpoint_path=checkpoint_path,
            checkpoint_strict=checkpoint_strict,
        )


__all__ = [
    "HFDetectionModel",
    "DFINEModel",
    "RTDETRv2Model",
    "_build_hf_detection_targets",
    "_decode_hf_predictions",
]
