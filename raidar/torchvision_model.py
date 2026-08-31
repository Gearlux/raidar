"""The torchvision detection zoo behind ONE configured wrapper class.

The raw builder functions (:mod:`raidar.torchvision_detection`) stay the TRAINING spelling — a
``!lazy:`` slot the trainer flows with the dataset-derived ``num_classes``. What they cannot
offer a config-facing surface is exactly what this wrapper adds: the pretrained ``weights`` as a
plain enum value instead of a wire-only torchvision enum object, a ``score_threshold`` (torchvision
reports every box above its internal 0.05 — noise in a review UI), a ``label_indexing`` knob (the
zoo emits 1-indexed foreground labels, background reserved at 0, while the rest of the workspace's
detection wrappers emit 0-indexed), and a ``checkpoint_path`` to load a trained model back for
inference. Rationale: ``docs/architecture.md``.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast, get_args

import torch
from confluid import configurable
from loggair import get_logger
from matrainer.torch import load_lightning_checkpoint
from matrainer.uri import ModelRef, file_model_ref
from recordstream.outputs import DetectionOutput
from torch import Tensor, nn

logger = get_logger(__name__)

#: Scheme naming the torchvision zoo in a model reference — the query pins which weights loaded.
TORCHVISION_URI_PREFIX = "torchvision://"

#: The zoo members raidar supports — the same nine builders `raidar.torchvision_detection`
#: registers for the training slot. All conform to the native detection forward contract.
TorchVisionVariant = Literal[
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "retinanet_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
]
TORCHVISION_VARIANTS: Tuple[str, ...] = get_args(TorchVisionVariant)

#: The pretrained checkpoints torchvision publishes for these detectors. Every current variant
#: ships exactly ``COCO_V1`` with ``DEFAULT`` aliasing it (verified across all nine, torchvision
#: 0.2x); ``none`` builds fully random — the shape a ``checkpoint_path`` load starts from.
TorchVisionWeights = Literal["none", "DEFAULT", "COCO_V1"]
TORCHVISION_WEIGHTS: Tuple[str, ...] = get_args(TorchVisionWeights)

#: What numbering the EMITTED labels use. torchvision reserves background at index 0 and emits
#: foreground as ``1..K``; ``zero_based`` (the default) shifts that to ``0..K-1`` — the convention
#: the workspace's other detection wrappers emit — while ``one_based`` passes it through verbatim.
LabelIndexing = Literal["zero_based", "one_based"]
LABEL_INDEXINGS: Tuple[str, ...] = get_args(LabelIndexing)


#: variant -> (classification-head weight key, anchors-per-location divisor): the rule that reads
#: the class count OFF a checkpoint's head shape. MEASURED at num_classes=7 (2026-08-31):
#: Faster R-CNN's box predictor is a linear ``[K, 1024]``; RetinaNet's cls_logits conv is
#: ``[9*K, 256, 3, 3]`` (nine anchors per location); FCOS is anchor-free, ``[K, 256, 3, 3]``;
#: SSD300's level-0 head is ``[4*K, 512, 3, 3]``; SSDLite's level-0 1x1 conv is ``[6*K, 672, 1, 1]``.
#: Pinned per variant by tests, so a torchvision release moving an anchor default fails there.
_HEAD_RULES: Dict[str, "Tuple[str, int]"] = {
    "fasterrcnn_resnet50_fpn": ("roi_heads.box_predictor.cls_score.weight", 1),
    "fasterrcnn_resnet50_fpn_v2": ("roi_heads.box_predictor.cls_score.weight", 1),
    "fasterrcnn_mobilenet_v3_large_fpn": ("roi_heads.box_predictor.cls_score.weight", 1),
    "fasterrcnn_mobilenet_v3_large_320_fpn": ("roi_heads.box_predictor.cls_score.weight", 1),
    "retinanet_resnet50_fpn": ("head.classification_head.cls_logits.weight", 9),
    "retinanet_resnet50_fpn_v2": ("head.classification_head.cls_logits.weight", 9),
    "fcos_resnet50_fpn": ("head.classification_head.cls_logits.weight", 1),
    "ssd300_vgg16": ("head.classification_head.module_list.0.weight", 4),
    "ssdlite320_mobilenet_v3_large": ("head.classification_head.module_list.0.1.weight", 6),
}


def _checkpoint_num_classes(variant: str, path: str) -> Optional[int]:
    """The class count BAKED INTO a checkpoint, read off its classification head's shape.

    Suffix-matched, so Lightning's ``model.`` prefix and the wrapper's own ``detector.`` level
    both resolve; a Lightning ``.ckpt``'s state dict nests under ``state_dict``. ``None`` when
    the file is absent (the loader owns that error) or the head key is not in the dict — the
    latter WARNS, because the build then falls back to the config default and the load that
    follows will almost certainly refuse the shapes.
    """
    import os

    if not os.path.exists(path):
        return None
    suffix, divisor = _HEAD_RULES[variant]
    raw: Any = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if isinstance(state, dict):
        for key, value in state.items():
            if isinstance(key, str) and (key == suffix or key.endswith("." + suffix)) and hasattr(value, "shape"):
                return int(value.shape[0]) // divisor
    logger.warning(
        f"TorchVisionDetectionModel: could not read the class count from {path!r} — "
        f"no key ending in {suffix!r} (is this a {variant!r} checkpoint?). "
        f"Building with the configured/default head instead."
    )
    return None


def _weights_member(variant: str, weights: str) -> Optional[Any]:
    """The variant's weights-enum member named ``weights``, or ``None`` — silently.

    ``None`` for ``"none"`` and for a name the variant's enum lacks; the caller decides whether
    that absence deserves a warning (the build does) or not (a provenance read does not).
    """
    if weights == "none":
        return None
    from torchvision.models import get_model_weights

    return getattr(get_model_weights(variant), weights, None)


def _resolve_weights(variant: str, weights: str) -> Optional[Any]:
    """The build-path resolution: a missing member WARNS and falls back to random init.

    The constructor already validates ``weights`` against :data:`TorchVisionWeights`, so this
    fires only on torchvision skew — a future version dropping or renaming a member for one
    variant. That is an operator-actionable condition (the model runs with random weights), hence
    ``warning``, but never an error: the checkpoint-load flow legitimately follows.
    """
    member = _weights_member(variant, weights)
    if member is None and weights != "none":
        from torchvision.models import get_model_weights

        available = ", ".join(m.name for m in get_model_weights(variant))
        logger.warning(
            f"TorchVisionDetectionModel: {variant!r} has no {weights!r} weights "
            f"(available: {available}). Building with random initialization."
        )
    return member


@configurable(task="detection", role="model")
class TorchVisionDetectionModel(nn.Module):
    """One torchvision detector, configured by value — weights, score cut, label numbering.

    Forward contract (the zoo's own, adapted only on the way out):

    * **Training**: ``forward(images, targets) → loss dict`` — passed through verbatim.
    * **Eval**: ``forward(images) → List[Dict]`` of per-image predictions, after dropping rows
      below ``score_threshold`` and background rows (label 0), and renumbering labels per
      ``label_indexing``.

    Args:
        variant: Which zoo member to build — one of :data:`TORCHVISION_VARIANTS`.
        weights: Pretrained checkpoint to load — ``DEFAULT``/``COCO_V1`` (COCO, 91 classes) or
            ``none`` (random init, the shape a ``checkpoint_path`` load starts from).
        num_classes: Head size incl. the background slot (a K-class dataset needs K+1). ``None``
            (the default) reads the size BAKED INTO ``checkpoint_path``'s own head when one is
            set, else keeps the pretrained head; setting it beside pretrained ``weights`` is
            refused by torchvision itself, and a value disagreeing with the checkpoint fails
            the load loudly.
        score_threshold: Eval-time confidence cutoff — torchvision itself reports every box
            above its internal 0.05.
        label_indexing: Numbering of EMITTED labels — ``zero_based`` (default; foreground
            ``0..K-1``, matching the workspace's other detection wrappers) or ``one_based``
            (torchvision's native ``1..K``).
        checkpoint_path: Optional trained checkpoint — a Lightning ``.ckpt`` or a plain state
            dict; keys retarget to the wrapped detector automatically, missing keys are logged,
            and a head-size disagreement fails the load loudly.
    """

    def __init__(
        self,
        variant: TorchVisionVariant = "fasterrcnn_resnet50_fpn",
        weights: TorchVisionWeights = "DEFAULT",
        num_classes: Optional[int] = None,
        score_threshold: float = 0.5,
        label_indexing: LabelIndexing = "zero_based",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if variant not in TORCHVISION_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(TORCHVISION_VARIANTS)}, got {variant!r}.")
        if weights not in TORCHVISION_WEIGHTS:
            raise ValueError(f"weights must be one of {sorted(TORCHVISION_WEIGHTS)}, got {weights!r}.")
        if label_indexing not in LABEL_INDEXINGS:
            raise ValueError(f"label_indexing must be one of {sorted(LABEL_INDEXINGS)}, got {label_indexing!r}.")
        self.variant = variant
        self.weights = weights
        self.num_classes = None if num_classes is None else int(num_classes)
        self.score_threshold = float(score_threshold)
        self.label_indexing = label_indexing
        self.checkpoint_path = checkpoint_path
        # Partial: config only. The builder call (and behind pretrained weights, a download)
        # happens in solidify(), so constructing for a schema read costs nothing.
        self.detector: Optional[nn.Module] = None

    def solidify(self) -> nn.Module:
        """Build (once) and return the live detector — confluid's ``flow()``-time hook.

        The checkpoint load stays last, so its values win over whatever ``weights`` loaded —
        the same order the RF-DETR wrapper uses.
        """
        if self.detector is None:
            from torchvision.models import get_model_builder

            builder = get_model_builder(self.variant)
            member = _resolve_weights(self.variant, self.weights)
            kwargs: Dict[str, Any] = {"weights": member}
            if member is None:
                # Fully random — the checkpoint-load shape. Leaving the backbone default here
                # would silently download ImageNet weights the checkpoint overwrites anyway.
                kwargs["weights_backbone"] = None
            num_classes = self.num_classes
            if num_classes is None and self.checkpoint_path:
                # The head size is baked into a trained checkpoint — config need not restate it.
                num_classes = _checkpoint_num_classes(self.variant, self.checkpoint_path)
            if num_classes is not None:
                kwargs["num_classes"] = num_classes
            self.detector = builder(**kwargs)
            if self.checkpoint_path:
                load_lightning_checkpoint(self, self.checkpoint_path)
        return cast(nn.Module, self.detector)

    @property
    def model_refs(self) -> List[ModelRef]:
        """The weight artifacts this detector loads — see :mod:`matrainer.uri`."""
        refs: List[ModelRef] = []
        if self.weights != "none":
            member = _weights_member(self.variant, self.weights)
            refs.append(
                ModelRef(
                    uri=f"{TORCHVISION_URI_PREFIX}{self.variant}?weights={self.weights}",
                    url=getattr(member, "url", None),
                )
            )
        if self.checkpoint_path:
            refs.append(file_model_ref(self.checkpoint_path))
        return refs

    def _decode(self, result: Dict[str, Tensor]) -> DetectionOutput:
        scores = result["scores"]
        labels = result["labels"]
        # Label 0 is torchvision's reserved background slot in BOTH numbering modes — a row
        # claiming it is a non-detection, so it is dropped rather than renumbered to -1.
        keep = (scores >= self.score_threshold) & (labels >= 1)
        labels = labels[keep]
        if self.label_indexing == "zero_based":
            labels = labels - 1
        return DetectionOutput(
            boxes=result["boxes"][keep].float(),
            scores=scores[keep].float(),
            labels=labels.long(),
        )

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[Dict[str, Tensor], List[DetectionOutput]]:
        detector = self.solidify()
        if self.training:
            if targets is None:
                raise ValueError("TorchVisionDetectionModel in train mode requires targets.")
            return cast(Dict[str, Tensor], detector(list(images), targets))
        with torch.no_grad():
            results = detector(list(images))
        return [self._decode(result) for result in results]


__all__ = [
    "LABEL_INDEXINGS",
    "LabelIndexing",
    "TORCHVISION_URI_PREFIX",
    "TORCHVISION_VARIANTS",
    "TORCHVISION_WEIGHTS",
    "TorchVisionDetectionModel",
    "TorchVisionVariant",
    "TorchVisionWeights",
]
