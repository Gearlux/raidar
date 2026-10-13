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

import os
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated, Any, Dict, List, Literal, Optional, Union, cast, get_args

import numpy as np
import torch
import torch.nn as nn
from annotated_types import Interval
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

#: Every RF-DETR class a config may ask for, by the name the ``variant`` knob takes: the four detectors
#: and the five instance-segmentation models (``seg-*``), which train on the same box dataset — RF-DETR's
#: own reader turns a box label into its four-corner polygon — through the ``rfdetr`` backend
#: (``RFDETRDetector``). The Literals below are read off this table.
VariantName = Literal[
    "nano", "small", "medium", "large", "seg-nano", "seg-small", "seg-medium", "seg-large", "seg-xlarge"
]
_VARIANT_CLASSES: Dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "seg-nano": "RFDETRSegNano",
    "seg-small": "RFDETRSegSmall",
    "seg-medium": "RFDETRSegMedium",
    "seg-large": "RFDETRSegLarge",
    "seg-xlarge": "RFDETRSegXLarge",
}


def _build_rfdetr(
    variant: VariantName,
    num_classes: int,
    pretrained_detector: bool,
    resolution: Optional[int],
    weights_file: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Instantiate a high-level ``RFDETR`` variant and return it.

    ``weights_file`` names one of RF-DETR's OWN checkpoints to build from instead of the COCO
    weights — the library's ``pretrain_weights`` route, which reads both of its checkpoint
    shapes and aligns the head. It wins over ``pretrained_detector``: the file replaces every
    weight the COCO checkpoint would have supplied.
    """
    import rfdetr

    cls_name = _VARIANT_CLASSES[variant]
    rfdetr_cls = getattr(rfdetr, cls_name)
    taken = sorted({"num_classes", "resolution", "pretrain_weights"} & set(model_config or {}))
    if taken:
        raise ValueError(f"model_config must not set {taken}: each has a parameter of its own")
    kwargs: Dict[str, Any] = dict(model_config or {})
    kwargs["num_classes"] = int(num_classes)
    if weights_file is not None:
        kwargs["pretrain_weights"] = weights_file
    elif not pretrained_detector:
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


#: The pretrained RF-DETR checkpoints are COCO-trained — 90 foreground classes. Resolving an
#: unset ``num_classes`` to this is what lets the DEFAULT node detect out of the box instead of
#: silently re-heading the checkpoint to a random 1-class head (measured 2026-08-31: the library
#: itself warns "Checkpoint has 90 classes but model is configured for 1").
_PRETRAINED_NUM_CLASSES = 90


def _checkpoint_num_classes(path: str) -> Optional[int]:
    """The FOREGROUND class count baked into a checkpoint — its ``class_embed`` rows minus one.

    RF-DETR's head is ``[K + 1, hidden]`` (the no-object slot rides at index K). The state dict
    is unwrapped from the containers both training routes write (``state_dict`` for a Lightning
    ``.ckpt``, ``model`` for RF-DETR's own ``checkpoint_best_total.pth``) and the key is
    suffix-matched, so ``model.``/``detector.`` prefixes resolve too. ``None`` when the file is
    absent (the loader owns that error) or carries no such head.
    """
    if not os.path.exists(path):
        return None
    raw: Any = torch.load(path, map_location="cpu", weights_only=False)
    state = raw
    for container in ("state_dict", "model"):
        if isinstance(state, dict) and isinstance(state.get(container), dict):
            state = state[container]
    if isinstance(state, dict):
        for key, value in state.items():
            if (
                isinstance(key, str)
                and (key == "class_embed.weight" or key.endswith(".class_embed.weight"))
                and hasattr(value, "shape")
            ):
                return int(value.shape[0]) - 1
    return None


def _is_rfdetr_checkpoint(path: str) -> bool:
    """Whether ``path`` is one of RF-DETR's OWN checkpoints — a top-level ``model`` MAPPING.

    RF-DETR writes two shapes and both carry it: the older ``{model, optimizer, lr_scheduler,
    epoch, args}`` and rfdetr 1.9's ``{model, args, state_dict, ...}``. What this project
    writes carries none — the lightning backend's ``.ckpt`` keeps the wrapper under
    ``state_dict``, and a plain ``state_dict()`` is keyed ``detector.*`` — and an Ultralytics
    ``.pt``'s ``model`` is a live ``nn.Module``, not a mapping. ``False`` for a missing file:
    the loader that runs next owns that error.

    Why it matters (measured 2026-09-27): matrainer's generic loader reads the older shape as
    a five-key state dict and lands 1 of 465 tensors, with only its INFO lines to show for it;
    ``rfdetr``'s own loader lands all 465. Teaching the generic loader RF-DETR's shapes was
    ruled out — a library's format is loaded by the wrapper that owns the library.
    """
    if not os.path.exists(path):
        return False
    raw: Any = torch.load(path, map_location="cpu", weights_only=False)
    return isinstance(raw, Mapping) and isinstance(raw.get("model"), Mapping)


@configurable(task="detection", role="model")
class RFDETRModel(nn.Module):
    """Torchvision-native RF-DETR wrapper for
    :class:`~raidar.lightning_detector.LightningDetector`.

    Forward contract:

    * **Training**: ``forward(images, targets) → {"loss": total}``.
    * **Eval**: ``forward(images) → List[Dict]`` of per-image predictions.

    Args:
        num_classes: Foreground class count. ``None`` (the default) reads it
            off ``checkpoint_path``'s own head when one is set, else uses the
            pretrained COCO detector's 90 — set it only to build a fresh random
            head. Internally an extra slot is reserved for the no-object class
            and any prediction with that label is dropped during decode.
        variant: ``"nano"`` (default), ``"small"``, ``"medium"``, ``"large"``.
        pretrained_detector: Load full COCO-pretrained RF-DETR weights.
        resolution: Square input size in pixels. Must be a positive integer
            divisible by 32. ``None`` uses the variant's default resolution.
        score_threshold: Eval-time confidence cutoff applied after
            background-class filtering.
        checkpoint_path: Optional trained-checkpoint load (a Lightning ``.ckpt``, RF-DETR's
            own ``.pth`` in either of its shapes, or a plain state dict); its values win over
            the pretrained ones. RF-DETR's own files load through ``rfdetr``'s loader, the
            rest through matrainer's generic one.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        variant: Literal["nano", "small", "medium", "large"] = "nano",
        pretrained_detector: bool = True,
        resolution: Optional[int] = None,
        score_threshold: float = 0.5,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if variant not in _VARIANT_CLASSES:
            raise ValueError(f"variant must be one of {sorted(_VARIANT_CLASSES)}, got {variant!r}.")
        self.num_classes = None if num_classes is None else int(num_classes)
        self.variant = variant
        self.score_threshold = float(score_threshold)
        # Kept so the model can name its own weight artifacts (`model_refs`) — a value the
        # run record needs cannot be one the constructor consumes and forgets.
        self.pretrained_detector = bool(pretrained_detector)
        self.checkpoint_path = checkpoint_path
        # Partial: store config only. Building here allocated 30.1M parameters AND read the
        # pretrained checkpoint off disk (~/.roboflow) at CONSTRUCTION — 2.1 s before this
        # object could even be asked what its parameters are named, which the Partial-Init /
        # Zero-Arg mandate forbids. Everything the build derives is resolved in solidify().
        self._requested_resolution = resolution
        self._resolution: Optional[int] = None
        self._resolved_num_classes: Optional[int] = None
        self.detector: Optional[nn.Module] = None
        self.postprocess: Optional[Any] = None
        self.criterion: Optional[nn.Module] = None

    def solidify(self) -> nn.Module:
        """Build (once) and return the live detector — confluid's ``flow()``-time hook.

        Resolves everything the eager constructor used to derive from the built model: the
        detector itself, its postprocessor, the EFFECTIVE resolution (which the library picks
        from the variant when none was requested) and the criterion. Idempotent; ``flow()``
        calls it, so ``self.parameters()`` is populated before an optimizer reads it, and
        :meth:`forward` calls it for a directly constructed instance. The checkpoint's values win
        over the pretrained ones: RF-DETR's own file is handed to the library AS the weights to
        build from, any other file is loaded after the build — the same order the eager
        constructor had.
        """
        if self._resolved_num_classes is None:
            self._resolved_num_classes = self._resolve_num_classes()
        if self.detector is None:
            from rfdetr.detr import RFDETR
            from rfdetr.models.lwdetr import build_criterion_and_postprocessors

            # The ImageNet normalization constants are the LIBRARY's. Registering the buffers
            # HERE rather than in __init__ is what keeps construction free of the `rfdetr`
            # import itself — measured at 0.3-0.5 s for the first instance in a process, which
            # a schema read of this class should not pay. They land before the checkpoint load
            # below, so a `.ckpt` carrying them still wins.
            self.register_buffer("_mean", torch.tensor(RFDETR.means).view(1, 3, 1, 1))
            self.register_buffer("_std", torch.tensor(RFDETR.stds).view(1, 3, 1, 1))
            rfdetr_file = self._rfdetr_checkpoint()
            rfdetr = _build_rfdetr(
                variant=self.variant,
                num_classes=self._resolved_num_classes,
                pretrained_detector=self.pretrained_detector,
                resolution=self._requested_resolution,
                weights_file=rfdetr_file,
            )
            self.detector = rfdetr.model.model
            self.postprocess = rfdetr.model.postprocess
            self._resolution = int(rfdetr.model_config.resolution)
            criterion_args = _build_criterion_args(rfdetr.model_config, rfdetr.model.device)
            criterion, _ = build_criterion_and_postprocessors(criterion_args)
            self.criterion = criterion
            if self.checkpoint_path and rfdetr_file is None:
                load_lightning_checkpoint(self, self.checkpoint_path)
        return cast(nn.Module, self.detector)

    def _rfdetr_checkpoint(self) -> Optional[str]:
        """``checkpoint_path`` when it is RF-DETR's own file, checked against an explicit head size.

        An explicit ``num_classes`` wins and a checkpoint that disagrees fails the load (the
        2026-08-31 ruling). ``rfdetr``'s loader would instead re-size the head quietly, so the
        disagreement is refused HERE, before the library is asked.
        """
        if not self.checkpoint_path or not _is_rfdetr_checkpoint(self.checkpoint_path):
            return None
        if self.num_classes is not None:
            baked = _checkpoint_num_classes(self.checkpoint_path)
            if baked is not None and baked != self.num_classes:
                raise RuntimeError(
                    f"RFDETRModel: num_classes={self.num_classes} but {self.checkpoint_path!r} carries a "
                    f"{baked}-class head — leave num_classes unset to use the checkpoint's own count."
                )
        logger.info(f"RFDETRModel: loading RF-DETR checkpoint {self.checkpoint_path} through rfdetr's own loader")
        return self.checkpoint_path

    def _resolve_num_classes(self) -> int:
        """The head size this build uses — explicit, else BAKED INTO the weights being loaded.

        Order (user ruling 2026-08-31, "only use checkpoint_path"): an explicit ``num_classes``
        wins; unset, a ``checkpoint_path``'s own ``class_embed`` shape answers; unset with only
        the pretrained COCO detector, its known 90 — so the checkpoint loads WITHOUT the silent
        random re-head. Reads config + at most one file; never builds.
        """
        if self.num_classes is not None:
            return self.num_classes
        if self.checkpoint_path:
            derived = _checkpoint_num_classes(self.checkpoint_path)
            if derived is not None:
                return derived
            logger.warning(
                f"RFDETRModel: could not read the class count from {self.checkpoint_path!r} "
                f"(no class_embed head found) — falling back to the pretrained default."
            )
        if self.pretrained_detector:
            return _PRETRAINED_NUM_CLASSES
        raise ValueError(
            "RFDETRModel: num_classes is not set and there is no checkpoint_path or pretrained "
            "detector to read it from — set num_classes to build a fresh head."
        )

    @property
    def backbone(self) -> nn.Module:
        """The live detector, built on first access.

        ``flow()`` never touches property getters (confluid's lazy-init rule 3), which is why
        :meth:`solidify` exists as a method as well; this is what makes a directly constructed,
        never-flowed instance usable.
        """
        return self.solidify()

    @property
    def resolution(self) -> int:
        """The EFFECTIVE square input size, which only the built model knows.

        ``resolution=None`` (the default) lets the library choose per variant, so this cannot be
        answered from stored config alone — reading it builds. Callers that only want to know
        what was REQUESTED read ``_requested_resolution`` instead and get no build.
        """
        self.solidify()
        return cast(int, self._resolution)

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
        # Built alongside the detector in solidify(); only ever reached from forward(), which
        # solidifies first — the cast documents that rather than re-checking it here.
        criterion = cast(Any, self.criterion)
        loss_dict = criterion(outputs, targets_detr)
        weight_dict = criterion.weight_dict
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
        # Build up front rather than relying on `_preprocess` happening to read the
        # resolution property first — this is the one call that must not be order-sensitive.
        detector = self.solidify()

        if self.training:
            if targets is None:
                raise ValueError("RFDETRModel in train mode requires targets.")
            normalised = self._preprocess(images)
            targets_detr = _build_rfdetr_targets(targets, self.resolution, normalised.device)
            outputs = detector(normalised, targets_detr)
            loss_total = self._criterion_loss(outputs, targets_detr)
            return {"loss": loss_total}

        normalised = self._preprocess(images)
        outputs = detector(normalised)
        target_sizes = torch.tensor(
            [[self.resolution, self.resolution]] * normalised.shape[0],
            device=normalised.device,
        )
        results = cast(Any, self.postprocess)(outputs, target_sizes)
        predictions = _decode_rfdetr_predictions(results, cast(int, self._resolved_num_classes), self.score_threshold)
        return predictions


#: A score cut, between 0 and 1 — the widget range a form or a picker offers.
Probability = Annotated[float, Interval(ge=0.0, le=1.0)]

#: How :class:`RFDETRPredictor` scores a query: ``class`` — RF-DETR's own, one detection per (query, class)
#: pair above the cut, scored by that class; ``any_class`` — one detection per query, scored by the chance
#: that ANY of its classes is right and named by its best one.
ScoreMode = Literal["class", "any_class"]


def _any_class_logits(logits: Tensor, counted: Tensor) -> Tensor:
    """Rewrite ``[B, Q, C]`` class logits so RF-DETR's own top-k ranks each QUERY by its any-class chance.

    RF-DETR scores every class of a query with its own sigmoid, so a query that is surely a signal but
    torn between classes (0.27, 0.13, 0.13, ...) never passes a per-class cut. Here each query keeps ONE
    slot — its best counted class — holding ``logit(1 - prod(1 - p))`` over the counted classes, and every
    other slot is ``-inf`` (score 0). The library's post-processing then selects, boxes and masks exactly
    as it does for real logits; nothing of it is patched or copied.

    Args:
        logits: The head's raw class logits, ``[B, Q, C]`` (the no-object slot included).
        counted: ``[C]`` bool — the slots that count as a class (False: the no-object slot, an excluded class).
    """
    prob = logits.sigmoid() * counted.to(logits.dtype)
    # 1 - prod(1 - p) as -expm1(sum(log1p(-p))): no underflow over many small scores; p = 1 gives -inf -> 1.
    any_class = -torch.expm1(torch.log1p(-prob).sum(dim=-1))
    best = prob.argmax(dim=-1, keepdim=True)
    rewritten = torch.full_like(logits, float("-inf"))
    return rewritten.scatter(-1, best, torch.logit(any_class, eps=1e-7).unsqueeze(-1).to(logits.dtype))


class _AnyClassPostProcess:
    """RF-DETR's ``PostProcess`` with the logits rewritten first (:func:`_any_class_logits`).

    ``RFDETR.predict`` only CALLS its post-process (``self.model.postprocess(outputs, target_sizes=...,
    score_threshold=...)``), so wrapping that call is the whole change: the rest of ``predict`` — the
    threshold, the class names, the masks — runs as the library wrote it.
    """

    def __init__(self, inner: Any, counted: Tensor) -> None:
        self.inner = inner  # the library's PostProcess (an nn.Module; its type is the library's private one)
        self.counted = counted

    def __call__(self, outputs: Dict[str, Tensor], *args: Any, **kwargs: Any) -> Any:
        logits = outputs["pred_logits"]
        rewritten = dict(outputs, pred_logits=_any_class_logits(logits, self.counted.to(logits.device)))
        return self.inner(rewritten, *args, **kwargs)


@configurable(task="detection", role="model")
class RFDETRPredictor:
    """Prediction from one of RF-DETR's OWN checkpoints — boxes, and instance masks for a ``seg-*`` model.

    The PREDICTION twin of :class:`RFDETRModel`, which stays the Lightning backend's TRAINING wrapper and
    builds box targets only (its variants are the four detectors on purpose). A checkpoint written by
    RF-DETR's training (``checkpoint_best_total.pth``) carries everything its model is: the class, the
    resolution, the query groups (``group_detr``), the head and its class names. This reads all of it from
    the file through ``rfdetr.RFDETR.from_checkpoint`` — so a seg model trained with ``group_detr: 4``
    loads without anything restated; the variant's own default (13 groups) would refuse its weights.

    The contract is the torchvision eval one the detection ops call (``eval()``, ``to(device)``,
    ``model(images) -> [{boxes, scores, labels}]``) plus ``masks`` when the model segments:

    * ``images``: a list of pictures of ANY size — ``[3, H, W]`` floats in ``[0, 1]``, ``H x W x 3``
      uint8 arrays, or a grey picture (``[H, W]`` / ``[1, H, W]``) read as its three equal channels, the
      way training read a one-channel picture (``convert("RGB")``). RF-DETR resizes each to its resolution.
    * per picture: ``boxes`` ``[N, 4]`` xyxy and ``masks`` ``[N, H, W]`` bool, both in the INPUT
      picture's pixels; ``scores`` ``[N]``; ``labels`` ``[N]`` int64, 0-based into :attr:`class_names`.
      A label outside the vocabulary (RF-DETR's no-object slot) is dropped.

    Not an ``nn.Module``: the network is the library's, held by its ``RFDETR`` object, and ``to()`` must
    move that object's device too (its ``predict`` places the input by it), which a module's ``to``
    would not.

    Args:
        checkpoint_path: The RF-DETR checkpoint to predict with (``checkpoint_best_total.pth`` or
            ``checkpoint_best_ema.pth``; a Lightning ``.ckpt`` of rfdetr <= 1.11 carries no ``args`` and
            is refused by the library). Required to predict; read on the first prediction.
        score_threshold: A detection scoring at or below this is dropped.
        score_mode: ``class`` (default) — RF-DETR's own scoring: a detection per (query, class) pair, scored by
            that class, so one query may answer under two classes. ``any_class`` — one detection per query,
            scored ``1 - prod(1 - p)`` over its classes (the chance that ANY of them is right) and named by its
            best class: a query that is surely a signal but torn between classes is kept, its name a best guess.
        any_class_excludes: Class names left out of the ``any_class`` score and never used as a name (read
            only in ``any_class`` mode); a name the checkpoint does not have is refused when the model is built.
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        score_threshold: Probability = 0.5,
        score_mode: ScoreMode = "class",
        any_class_excludes: Optional[List[str]] = None,
    ) -> None:
        if score_mode not in get_args(ScoreMode):
            raise ValueError(
                f"RFDETRPredictor: score_mode {score_mode!r} is not one of "
                + ", ".join(repr(mode) for mode in get_args(ScoreMode))
            )
        self.checkpoint_path = checkpoint_path
        self.score_threshold = float(score_threshold)
        self.score_mode: ScoreMode = score_mode
        self.any_class_excludes: List[str] = list(any_class_excludes or [])
        # Lazy-init: the library and the file are touched on the first prediction (or `solidify()`).
        self._rf: Any = None
        self._device: str = "cpu"

    def solidify(self) -> Any:
        """Build (once) and return the library's ``RFDETR`` object, on the device last asked for."""
        if self._rf is None:
            if not self.checkpoint_path:
                raise ValueError("RFDETRPredictor: checkpoint_path is not set — name an RF-DETR checkpoint file")
            path = os.path.expanduser(os.path.expandvars(self.checkpoint_path))
            if not os.path.exists(path):
                raise FileNotFoundError(f"RFDETRPredictor: checkpoint {path} does not exist")
            from rfdetr.detr import RFDETR

            logger.debug(f"RFDETRPredictor: loading {path} on {self._device}")
            rf = RFDETR.from_checkpoint(path, device=self._device)
            if self.score_mode == "any_class":
                # Typed `PostProcess` by the library; the wrapper only has to be CALLED like one (predict's one use).
                rf.model.postprocess = cast(Any, _AnyClassPostProcess(rf.model.postprocess, self._counted_slots(rf)))
            self._rf = rf
        return self._rf

    def _counted_slots(self, rf: Any) -> Tensor:
        """The class-logit slots ``any_class`` counts: every class of the checkpoint but the excluded ones.

        The head has one slot more than the vocabulary (the no-object slot), and it never counts.
        """
        names = list(rf.class_names)
        unknown = [name for name in self.any_class_excludes if name not in names]
        if unknown:
            raise ValueError(
                f"RFDETRPredictor: any_class_excludes names {', '.join(repr(n) for n in unknown)}, which the "
                f"checkpoint {self.checkpoint_path} does not have — its classes: {', '.join(names)}"
            )
        slots = int(getattr(rf.model.args, "num_classes", len(names))) + 1
        counted = torch.zeros(max(slots, len(names)), dtype=torch.bool)
        counted[: len(names)] = True
        for name in self.any_class_excludes:
            counted[names.index(name)] = False
        return counted

    def eval(self) -> "RFDETRPredictor":
        """Prediction only — there is no training mode; kept for the eval contract callers use."""
        return self

    def to(self, device: Union[str, torch.device]) -> "RFDETRPredictor":
        """Predict on ``device`` — remembered for the build, applied at once to a model already built."""
        self._device = str(device)
        if self._rf is not None:
            self._rf.model.model.to(self._device)
            self._rf.model.device = torch.device(self._device)
        return self

    @property
    def class_names(self) -> List[str]:
        """The checkpoint's class names, by label id (reading it builds the model)."""
        return list(self.solidify().class_names)

    @property
    def model_refs(self) -> List[ModelRef]:
        """The weight artifact this predictor loads — the checkpoint file (stored config only, no build)."""
        return [file_model_ref(self.checkpoint_path)] if self.checkpoint_path else []

    def __call__(self, images: List[Any]) -> List[Dict[str, Tensor]]:
        rf = self.solidify()
        detections = rf.predict(
            [_rgb_picture(image) for image in images], threshold=self.score_threshold, include_source_image=False
        )
        if not isinstance(detections, list):
            detections = [detections]
        known = len(rf.class_names)
        outputs: List[Dict[str, Tensor]] = []
        for found in detections:
            keep = np.asarray(found.class_id) < known
            out = {
                "boxes": torch.as_tensor(np.asarray(found.xyxy, dtype=np.float32)[keep]).reshape(-1, 4),
                "scores": torch.as_tensor(np.asarray(found.confidence, dtype=np.float32)[keep]),
                "labels": torch.as_tensor(np.asarray(found.class_id, dtype=np.int64)[keep]),
            }
            if found.mask is not None:
                out["masks"] = torch.as_tensor(np.asarray(found.mask, dtype=bool)[keep])
            outputs.append(out)
        return outputs


def _rgb_picture(image: Any) -> Any:
    """One picture as RF-DETR's ``predict`` takes it: a ``[3, H, W]`` float tensor, or an ``H x W x 3`` uint8 array.

    A grey picture becomes its three equal channels — what training's reader made of a one-channel PNG.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image if image.dtype == np.uint8 else torch.as_tensor(np.moveaxis(image, -1, 0)).float()
    tensor = torch.as_tensor(image)
    if tensor.dim() == 2:
        tensor = tensor[None]
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, *tensor.shape[1:])
    return tensor.float()


__all__ = [
    "RFDETRModel",
    "RFDETRPredictor",
    "ScoreMode",
    "VariantName",
    "_build_rfdetr_targets",
    "_decode_rfdetr_predictions",
]
