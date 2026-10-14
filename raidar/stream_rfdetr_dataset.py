"""Bridge a RecordStream ``Stream`` into RF-DETR's native in-memory dataset contract.

RF-DETR's :class:`rfdetr.training.module_data.RFDETRDataModule` builds its
datasets via ``build_dataset(split, ns, resolution)`` — which reads a COCO/YOLO
directory off disk. This module supplies the *in-memory* counterpart so the
native ``RFDETRModelModule`` / ``build_trainer`` stack can train/eval straight
from a :class:`recordstream.Stream` with **no disk materialization** — the analogue of
``raidar.ultralytics.stream_yolo_dataset.StreamYOLODataset`` for RF-DETR.

How it stays byte-faithful to the disk path: rfdetr's ``CocoDetection.__getitem__``
does ``img(PIL), target = ConvertCoco(img, coco_target)`` then
``img, target = self._transforms(img, target)``. We reuse the SAME
:class:`rfdetr.datasets.coco.ConvertCoco` converter + the SAME
``make_coco_transforms`` / ``make_coco_transforms_square_div_64`` pipeline (and
select between them exactly as ``build_roboflow_from_coco`` does), feeding a PIL
image + an xyxy-absolute-pixel target — the pre-``Normalize`` form, whose
``Normalize`` does the xyxy-abs → cxcywh-normalized conversion the model expects.
So our per-item output is identical to the disk dataset's, and the inherited
``collate_fn`` (NestedTensor), DataLoaders and ``transfer_batch_to_device`` work
unchanged.

Per-record contract (read via ``item_data`` over the configured keys — the
torchvision-detection convention raidar uses elsewhere,
:class:`raidar.rfdetr.RFDETRModel`):

* The ``input_key`` value's payload: a torch tensor / numpy array / PIL
  image, ``[C, H, W]`` / ``[H, W, C]`` / ``[H, W]`` in **RGB** channel order.
  Floats in ``[0, 1]`` are rescaled to ``uint8 [0, 255]``; uint8 passes through.
  (No BGR flip — RF-DETR's ``ToImage``/``Normalize`` expect RGB, unlike
  ultralytics.)
* The ``target_key`` :class:`recordstream.Boxes` value — ``boxes`` ``[N, 4]``
  xyxy-pixel + ``labels`` ``[N]`` int — with labels **0-indexed ``0..K-1``**
  (RF-DETR reserves index ``K`` for the background class internally — the
  OPPOSITE of torchvision's background-at-0, so do NOT ``+1``-shift labels for
  RF-DETR). A record with no target key is fine.
* For a seg model (``segmentation_head``), the ``Boxes``' ``extras[POLYGONS]``
  (:mod:`raidar.yolo_sink`'s contract: per box its outlines in pixels) — each instance's mask,
  rasterised by RF-DETR's own ``ConvertCoco``; a box without an outline is masked by its
  rectangle.

The labels are read with the picture, per item — never walked ahead of it — so a random step in
the stream (``raidar.crop.RandomCropDetection``) cuts the picture and its boxes alike.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loggair import get_logger
from PIL import Image
from recordstream import Boxes, Stream, item_data

# PRIVATE rfdetr module, and it moves: `build_namespace` was renamed
# `_namespace_from_configs` in rfdetr 1.9.0, which broke this import outright. rfdetr exposes
# no public way to build the merged namespace its transforms need, so the private path is the
# only one available — pinned by tests/test_stream_rfdetr_dataset.py, which fails loudly on the
# next rename rather than at training time.
from rfdetr._namespace import _namespace_from_configs
from rfdetr.datasets._torchvision import Compose, RandomHorizontalFlip
from rfdetr.datasets.coco import ConvertCoco, make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize
from rfdetr.training.module_data import RFDETRDataModule
from torchvision.transforms.v2 import ToDtype, ToImage

from raidar.yolo_sink import POLYGONS

logger = get_logger(__name__)


def _record_input_to_rgb_pil(arr_in: Any) -> Image.Image:
    """Normalize a record's input-item payload into an RGB :class:`PIL.Image`.

    Accepts a PIL image (returned as RGB), or a torch tensor / numpy array in
    ``[C, H, W]`` / ``[H, W, C]`` / ``[H, W]`` layout. Float arrays in ``[0, 1]``
    are rescaled to ``uint8 [0, 255]``; grayscale is expanded to 3 channels.
    Channel order is assumed RGB (no BGR flip — RF-DETR wants RGB).
    """
    if isinstance(arr_in, Image.Image):
        return arr_in.convert("RGB")

    if isinstance(arr_in, torch.Tensor):
        arr = arr_in.detach().cpu().numpy()
    else:
        arr = np.asarray(arr_in)

    # CHW -> HWC (only when the leading dim looks like channels and the trailing one doesn't).
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.dtype != np.uint8:
        if arr.size and float(arr.max()) <= 1.0 + 1e-6 and float(arr.min()) >= -1e-6:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    return Image.fromarray(np.ascontiguousarray(arr), mode="RGB")


def _target_boxes_labels(target: Any) -> Any:
    """Extract ``(boxes, labels)`` from a projected detection target — a typed
    :class:`~recordstream.Boxes` (the ``CocoToTorchVisionDetection`` op output) or a raw
    ``{"boxes", "labels"}`` dict. Returns ``(None, None)`` when either is absent."""
    if target is None:
        return None, None
    if isinstance(target, Boxes):
        return target.boxes, target.labels
    if hasattr(target, "boxes") and hasattr(target, "labels"):
        return target.boxes, target.labels
    if isinstance(target, dict):
        return target.get("boxes"), target.get("labels")
    return None, None


def _record_input(record: Any, input_key: str) -> Any:
    """The image payload (tensor / array / PIL) under ``input_key`` in a record."""
    return item_data(record[input_key])


def _as_f32(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _as_i64(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.int64)
    return torch.as_tensor(np.asarray(value), dtype=torch.int64)


def _target_to_coco_annotations(
    boxes_xyxy: torch.Tensor, labels: torch.Tensor, polygons: Optional[Sequence[Any]] = None
) -> List[Dict[str, Any]]:
    """``boxes [N,4] xyxy-pixel`` + ``labels [N]`` -> the COCO annotation-list ``ConvertCoco`` consumes.

    Boxes are emitted as ``bbox = [x, y, w, h]`` (COCO format — ``ConvertCoco``
    reconstructs xyxy + clamps); ``category_id`` is the label verbatim (already
    0-indexed, ``cat2label=None``); ``area`` is ``w*h`` (cosmetic — torchmetrics
    box-mAP ignores it); ``iscrowd=0``.

    Given ``polygons`` (``Boxes.extras[POLYGONS]``, per box its outlines in pixels), each annotation also carries
    ``segmentation``: the outlines as COCO's flat ``[x0, y0, x1, y1, ...]`` lists — or, for a box with none, its
    rectangle, the YOLO convention that a box without an outline is its own mask.
    """
    boxes = boxes_xyxy.reshape(-1, 4)
    anns: List[Dict[str, Any]] = []
    for i in range(boxes.shape[0]):
        x0, y0, x1, y1 = (float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3]))
        w, h = (x1 - x0), (y1 - y0)
        ann: Dict[str, Any] = {
            "bbox": [x0, y0, w, h],
            "category_id": int(labels[i]),
            "area": float(max(w, 0.0) * max(h, 0.0)),
            "iscrowd": 0,
        }
        if polygons is not None:
            rings = polygons[i] if i < len(polygons) and polygons[i] else []
            flat = [[float(v) for point in ring for v in point] for ring in rings]
            ann["segmentation"] = flat or [[x0, y0, x1, y0, x1, y1, x0, y1]]
        anns.append(ann)
    return anns


def _build_transforms(image_set: str, resolution: int, ns: Any) -> Any:
    """Select + build the transform pipeline exactly as ``build_roboflow_from_coco`` does.

    Mirrors ``rfdetr/datasets/coco.py`` (``build_roboflow_from_coco``, the
    ``square_resize_div_64`` branch + kwargs) so the in-memory path applies the
    identical resize / augmentation / ImageNet-normalize the disk path would.
    """
    square_resize_div_64 = getattr(ns, "square_resize_div_64", False)
    builder = make_coco_transforms_square_div_64 if square_resize_div_64 else make_coco_transforms
    return builder(
        image_set,
        resolution,
        multi_scale=getattr(ns, "multi_scale", False),
        expanded_scales=getattr(ns, "expanded_scales", False),
        skip_random_resize=not getattr(ns, "do_random_resize_via_padding", False),
        patch_size=getattr(ns, "patch_size", 16),
        num_windows=getattr(ns, "num_windows", 4),
        aug_config=getattr(ns, "aug_config", None),
    )


def _unresized_transforms(image_set: str, ns: Any) -> Any:
    """RF-DETR's pipeline for ``image_set`` WITHOUT its resize: each picture reaches the model at its own size.

    RF-DETR's own pipelines always resize — the square one to ``resolution`` x ``resolution``, the other to a short
    side of ``resolution`` and a long side of at most 1333 (its ``_COCO_MAX_SIZE``, a module constant) — so a
    504 x 2040 picture trains as 330 x 1333. This keeps what they do besides: for training, the augmentation RF-DETR
    routes ``aug_config`` to (``None``: its horizontal flip at 0.5; ``{}``: none; anything else: its Albumentations
    stack), then the tensor, the float scale and ImageNet normalization. The batch still pads every picture to a
    multiple of the backbone's block (patch x windows) — RF-DETR's collate does that, not this pipeline.
    """
    aug_config = getattr(ns, "aug_config", None)
    steps: List[Any] = []
    if image_set == "train":
        if aug_config is None:
            steps.append(RandomHorizontalFlip(p=0.5))
        elif aug_config:
            steps.extend(AlbumentationsWrapper.from_config(aug_config))
    steps += [ToImage(), ToDtype(torch.float32, scale=True), Normalize()]
    return Compose(steps)


class StreamRFDETRDataset(torch.utils.data.Dataset):
    """In-memory ``Dataset`` yielding RF-DETR's ``(normalized-image, target)`` pairs from a Stream.

    Args:
        stream: The :class:`recordstream.Stream` of record dicts.
        image_set: ``"train"`` (augmented) or ``"val"`` / ``"test"`` (no aug).
        resolution: Model input resolution (``model_config.resolution``).
        ns: The merged ``rfdetr`` namespace (``_namespace_from_configs(model_config,
            train_config)``) — carries ``square_resize_div_64`` / ``multi_scale`` /
            ``patch_size`` / ``num_windows`` / ``aug_config`` that drive the transform.
        input_key: Record key carrying the input image (default ``"image"``).
        target_key: Record key carrying the target ``Boxes`` (default ``"target"``).
        keep_picture_size: Hand every picture to the model at its own size — RF-DETR's resize skipped
            (:func:`_unresized_transforms`). Every picture must then be ONE size: a second size is refused where it is
            read, naming both, since a seg batch stops on masks of two sizes.
    """

    def __init__(
        self,
        stream: Stream,
        image_set: str,
        resolution: int,
        ns: Any,
        input_key: str = "image",
        target_key: str = "target",
        keep_picture_size: bool = False,
    ) -> None:
        self._stream = stream
        self._image_set = image_set
        self._input_key = str(input_key)
        self._target_key = str(target_key)
        self._keep_picture_size = bool(keep_picture_size)
        self._transforms = (
            _unresized_transforms(image_set, ns)
            if self._keep_picture_size
            else _build_transforms(image_set, resolution, ns)
        )
        #: The first picture's (width, height) once read — every kept picture must be that size.
        self._kept_size: Optional[Tuple[int, int]] = None
        # A seg model (`segmentation_head`) trains on masks: each instance's outline, rasterised by RF-DETR's own
        # converter at the picture's size, so the masks follow every later resize and crop as the disk path's do.
        self._masks = bool(getattr(ns, "segmentation_head", False))
        self._convert = ConvertCoco(include_masks=self._masks, cat2label=None)
        # Nothing is read here: a picture's labels come from the SAME read as the picture (__getitem__), so a random
        # step in the stream — a crop — cuts both alike. Until 2026-10-12 every target was walked once at
        # construction and reused each epoch; a random crop then trained on the labels of another cut.

    def __len__(self) -> int:
        return len(self._stream)

    def _labels_of(self, record: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[Sequence[Any]]]:
        """``(boxes [N, 4] xyxy px, labels [N], outlines or None)`` of a record's target."""
        target = record.get(self._target_key) if isinstance(record, dict) else None
        boxes, labels = _target_boxes_labels(target)
        if boxes is None or labels is None:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64), None
        outlines = None
        if self._masks:
            extras = getattr(target, "extras", None) or {}
            outlines = extras.get(POLYGONS) or []
        return _as_f32(boxes).reshape(-1, 4), _as_i64(labels), outlines

    def _check_kept_size(self, i: int, record: Any, size: Tuple[int, int]) -> None:
        """Refuse a kept picture whose size is not the first one's, naming both and the record's file."""
        if self._kept_size is None:
            self._kept_size = size
            return
        if size != self._kept_size:
            name = record.get("file") if isinstance(record, dict) else None
            where = f"picture {i}" + (f" ({name})" if name else "")
            raise ValueError(
                f"keep_picture_size: {where} is {size[0]} x {size[1]} (width x height), the first was "
                f"{self._kept_size[0]} x {self._kept_size[1]} — a picture kept at its own size must be the one size "
                "every other is (a seg batch joins masks of one size): make the pictures one size, or crop them to one"
            )

    def __getitem__(self, i: int) -> Any:
        record = self._stream[i]
        pil = _record_input_to_rgb_pil(_record_input(record, self._input_key))
        if self._keep_picture_size:
            self._check_kept_size(i, record, pil.size)
        boxes, labels, outlines = self._labels_of(record)
        coco_target = {
            "image_id": i,
            "annotations": _target_to_coco_annotations(boxes, labels, outlines),
        }
        # ConvertCoco: COCO-annotation-list -> pre-transform tensor target (xyxy-abs boxes,
        # int64 labels, image_id, area, iscrowd, orig_size, size) — identical to the disk path.
        img, target = self._convert(pil, coco_target)
        # Compose: resize/augment/ToImage/ToDtype/Normalize -> [C,H,W] float32 + cxcywh-norm boxes.
        img, target = self._transforms(img, target)
        return img, target


class StreamRFDETRDataModule(RFDETRDataModule):
    """``RFDETRDataModule`` that injects :class:`StreamRFDETRDataset` instead of ``build_dataset``.

    Overrides :meth:`setup` to assign ``self._dataset_{train,val,test}`` from
    in-memory Streams (bypassing the disk ``build_dataset``) and the
    :attr:`class_names` property to return the supplied names (so RF-DETR's
    ``COCOEvalCallback``, which reads ``dm.class_names``, names per-class metrics
    without needing a ``.coco`` annotation object). All DataLoader / ``collate_fn``
    (NestedTensor) / ``transfer_batch_to_device`` plumbing is inherited unchanged.

    Args:
        model_config / train_config: as the base ``RFDETRDataModule``. ``aug_config``
            and the resize knobs flow through ``train_config`` (the namespace), so
            there is no separate augmentation parameter here.
        train_stream / val_stream / test_stream: the wired Streams (``test`` falls back to
            ``val`` when unset).
        class_names: ordered list of class names (index = class id).
        keep_picture_size: Every dataset it makes hands each picture to the model at its own size
            (:class:`StreamRFDETRDataset`'s setting of the same name).
    """

    def __init__(
        self,
        model_config: Any,
        train_config: Any,
        *,
        train_stream: Optional[Stream] = None,
        val_stream: Optional[Stream] = None,
        test_stream: Optional[Stream] = None,
        class_names: Optional[List[str]] = None,
        keep_picture_size: bool = False,
    ) -> None:
        super().__init__(model_config, train_config)
        self._keep_picture_size = bool(keep_picture_size)
        self._train_stream = train_stream
        self._val_stream = val_stream
        self._test_stream = test_stream
        self._stream_class_names = list(class_names) if class_names else None

    def _make(self, stream: Stream, image_set: str) -> StreamRFDETRDataset:
        ns = _namespace_from_configs(self.model_config, self.train_config)
        return StreamRFDETRDataset(
            stream, image_set, self.model_config.resolution, ns, keep_picture_size=self._keep_picture_size
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self._dataset_train is None and self._train_stream is not None:
                self._dataset_train = self._make(self._train_stream, "train")
            if self._dataset_val is None and self._val_stream is not None:
                self._dataset_val = self._make(self._val_stream, "val")
        elif stage == "validate":
            if self._dataset_val is None and self._val_stream is not None:
                self._dataset_val = self._make(self._val_stream, "val")
        elif stage == "test":
            src = self._test_stream if self._test_stream is not None else self._val_stream
            if self._dataset_test is None and src is not None:
                self._dataset_test = self._make(src, "val")  # val transforms — no augmentation
        elif stage == "predict":
            if self._dataset_val is None and self._val_stream is not None:
                self._dataset_val = self._make(self._val_stream, "val")

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._stream_class_names


__all__ = ["StreamRFDETRDataset", "StreamRFDETRDataModule"]
