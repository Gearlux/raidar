"""Bridge a DataFlux ``Flux`` into Ultralytics' ``YOLODataset`` contract.

Ultralytics' native trainer (``DetectionTrainer``) and validator
(``DetectionValidator``) build their dataloaders from a ``YOLODataset`` that
walks a directory of images + per-image ``.txt`` label files. The mosaic +
albumentations augmentation pipeline they ship is implemented inside
``YOLODataset.build_transforms``; reimplementing it from scratch would lose
the very thing we want from the native path.

This module subclasses ``YOLODataset`` with three targeted overrides
(``get_img_files``, ``get_labels``, ``load_image``) so the dataset reads from
a :class:`dataflux.Flux` of :class:`dataflux.Sample` triplets instead of
disk. Mosaic, mixup, cutmix and albumentations all keep working unchanged
because they only ever index into ``self.labels`` and call
``self.load_image(i)`` — neither path touches the file system once these
three methods are overridden.

Per-Sample contract:

* ``Sample.input``: a torch tensor or numpy array of shape ``[C, H, W]``,
  ``[H, W, C]`` or ``[H, W]`` in RGB channel order. Floats in ``[0, 1]`` are
  rescaled to ``uint8 [0, 255]`` automatically; uint8 is passed through.
  The image is converted to BGR ``HWC uint8`` to match Ultralytics' standard
  ``cv2.imread`` convention before resizing.
* ``Sample.target``: ``{"boxes": [N, 4] xyxy-pixel, "labels": [N] int}`` — the
  same torchvision-detection convention raidar already uses elsewhere
  (:class:`raidar.faster_rcnn.FasterRCNNModel`,
  :class:`raidar.yolo26.YOLO26Model`). Boxes are normalized to
  ``[N, 4] cxcywh ∈ [0, 1]`` (Ultralytics' wire format) on the way through.
  Empty-target samples are stored with zero-row ``cls`` / ``bboxes``; mosaic
  handles those fine.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from dataflux import Flux
from logflow import get_logger
from ultralytics.data.dataset import YOLODataset

logger = get_logger(__name__)


def _sample_image_to_bgr_hwc_uint8(arr_in: Any, target_channels: int) -> np.ndarray:
    """Normalize a Sample.input to ``HWC BGR uint8`` numpy."""
    if isinstance(arr_in, torch.Tensor):
        arr = arr_in.detach().cpu().numpy()
    else:
        arr = np.asarray(arr_in)

    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.dtype != np.uint8:
        if arr.size and float(arr.max()) <= 1.0 + 1e-6 and float(arr.min()) >= -1e-6:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    if arr.shape[-1] == 1 and target_channels == 3:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.shape[-1] == 3:
        # Flux convention is RGB (torchvision); ultralytics expects BGR.
        arr = np.ascontiguousarray(arr[..., ::-1])

    return arr


def _target_to_yolo_label(target: Any, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ``{"boxes": xyxy-pixel, "labels": int}`` to YOLO
    ``(cls[N,1] float32, bboxes[N,4] cxcywh-normalized float32)``.
    """
    if not target:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    boxes = target.get("boxes")
    labels = target.get("labels")
    if boxes is None or labels is None:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    if isinstance(boxes, torch.Tensor):
        boxes_arr = boxes.detach().cpu().numpy().astype(np.float32)
    else:
        boxes_arr = np.asarray(boxes, dtype=np.float32)
    boxes_arr = boxes_arr.reshape(-1, 4)

    if isinstance(labels, torch.Tensor):
        labels_arr = labels.detach().cpu().numpy().astype(np.float32)
    else:
        labels_arr = np.asarray(labels, dtype=np.float32)
    labels_arr = labels_arr.reshape(-1)

    if boxes_arr.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    cx = (boxes_arr[:, 0] + boxes_arr[:, 2]) / (2.0 * w)
    cy = (boxes_arr[:, 1] + boxes_arr[:, 3]) / (2.0 * h)
    bw = (boxes_arr[:, 2] - boxes_arr[:, 0]) / w
    bh = (boxes_arr[:, 3] - boxes_arr[:, 1]) / h
    bboxes = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)
    cls = labels_arr.astype(np.float32).reshape(-1, 1)
    return cls, bboxes


def _input_hw(arr_in: Any) -> Tuple[int, int]:
    """Return (H, W) of a Sample.input regardless of CHW / HWC / HW layout."""
    shape = getattr(arr_in, "shape", None)
    if shape is None:
        raise ValueError("Sample.input has no .shape attribute — expected a tensor or numpy array.")
    if len(shape) == 3 and shape[0] in (1, 3) and shape[-1] not in (1, 3):
        return int(shape[1]), int(shape[2])  # CHW
    if len(shape) == 3:
        return int(shape[0]), int(shape[1])  # HWC
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])  # HW
    raise ValueError(f"Sample.input has unexpected shape {tuple(shape)}; expected 2D or 3D array.")


class FluxYOLODataset(YOLODataset):  # type: ignore[misc]
    """``YOLODataset`` subclass that reads images and labels from a DataFlux ``Flux``."""

    def __init__(
        self,
        *,
        flux: Flux,
        data: Dict[str, Any],
        imgsz: int,
        augment: bool,
        hyp: Any,
        batch_size: int,
        stride: int = 32,
        rect: bool = False,
        prefix: str = "",
        task: str = "detect",
    ) -> None:
        # Stash flux ref BEFORE super().__init__: BaseDataset.__init__ calls
        # self.get_img_files() and self.get_labels() during init.
        self._flux = flux
        super().__init__(
            "<flux>",  # img_path positional — ignored by our overrides
            data=data,
            task=task,
            imgsz=imgsz,
            cache=False,
            augment=augment,
            hyp=hyp,
            prefix=prefix,
            rect=rect,
            batch_size=batch_size,
            stride=stride,
        )

    def get_img_files(self, img_path: Any) -> List[str]:  # noqa: ARG002
        return [f"flux://{i}" for i in range(len(self._flux))]

    def get_labels(self) -> List[Dict[str, Any]]:
        labels: List[Dict[str, Any]] = []
        for i in range(len(self._flux)):
            sample = self._flux[i]
            h, w = _input_hw(sample.input)
            cls, bboxes = _target_to_yolo_label(sample.target, h, w)
            labels.append(
                {
                    "im_file": f"flux://{i}",
                    "shape": (h, w),
                    "cls": cls,
                    "bboxes": bboxes,
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        if not labels:
            raise RuntimeError(f"FluxYOLODataset: source flux is empty (len={len(self._flux)}).")
        return labels

    def _flux_index(self, i: int) -> int:
        """Resolve the position into the underlying Flux from the dataset slot ``i``.

        ``BaseDataset.set_rectangle`` (rect=True, used in val mode) reorders
        ``self.im_files`` and ``self.labels`` together by aspect-ratio sort,
        so the slot index ``i`` no longer matches the original Flux index.
        ``self.im_files[i]`` carries the synthetic ``flux://<j>`` path emitted
        by :meth:`get_img_files` and is reordered in lockstep with the labels,
        so it's the single source of truth for "which Flux sample does this
        slot describe?". Parsing back from there keeps the image we load
        paired with the labels the dataset hands to the loss / validator.
        """
        return int(self.im_files[i].rsplit("/", 1)[-1])

    def load_image(
        self,
        i: int,
        rect_mode: bool = True,
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        cached = self.ims[i]
        if cached is not None:
            return cached, self.im_hw0[i], self.im_hw[i]

        sample = self._flux[self._flux_index(i)]
        arr = _sample_image_to_bgr_hwc_uint8(sample.input, target_channels=self.channels)
        h0, w0 = arr.shape[:2]

        if rect_mode:
            r = self.imgsz / max(h0, w0)
            if r != 1:
                w_new = min(math.ceil(w0 * r), self.imgsz)
                h_new = min(math.ceil(h0 * r), self.imgsz)
                arr = cv2.resize(arr, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            arr = cv2.resize(arr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        if arr.ndim == 2:
            arr = arr[..., None]

        if self.augment:
            # BaseDataset types self.ims/im_hw0/im_hw as list[None] at init;
            # ultralytics' own load_image populates them with arrays/tuples
            # at runtime — mirror that here.
            self.ims[i], self.im_hw0[i], self.im_hw[i] = arr, (h0, w0), arr.shape[:2]  # type: ignore[call-overload]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
        return arr, (h0, w0), arr.shape[:2]

    def cache_labels(self, path: Any = None) -> Dict[str, Any]:  # noqa: ARG002
        raise NotImplementedError(
            "FluxYOLODataset builds labels lazily from the Flux source — there is no on-disk label cache."
        )

    def cache_images(self) -> None:
        # No-op: load_image already caches into self.ims when augment=True.
        return


__all__ = ["FluxYOLODataset"]
