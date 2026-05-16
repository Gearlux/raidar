"""Tests for :class:`raidar.flux_yolo_dataset.FluxYOLODataset`.

Covers the three things the FluxYOLODataset must get right for the native
Ultralytics path to work:

1. Sample.input → BGR HWC uint8 numpy (CHW float, HWC uint8, 1-channel
   inputs all normalize to the same canonical shape).
2. Sample.target xyxy-pixel boxes → cxcywh-normalized float32, with
   empty-target samples producing zero-row arrays mosaic can splice.
3. The dataset object integrates cleanly with the parent ``YOLODataset``
   ``__getitem__`` path (Format transform produces ``img`` / ``cls`` /
   ``bboxes`` / ``batch_idx`` keys with the right dtypes).
"""

from typing import List, cast

import numpy as np
import pytest
import torch
from dataflux.core import Flux
from dataflux.sample import Sample
from ultralytics.cfg import get_cfg

from raidar.flux_yolo_dataset import FluxYOLODataset, _input_hw, _sample_image_to_bgr_hwc_uint8, _target_to_yolo_label


def _tiny_flux(n: int = 4, h: int = 64, w: int = 64) -> Flux:
    """Construct a tiny in-memory Flux of (image, target) Samples.

    Two of every four samples have a single box; the others are empty.
    Produces inputs as CHW float tensors (raidar / waivefront convention).
    """
    samples: List[Sample] = []
    for i in range(n):
        img = torch.rand(3, h, w)
        if i % 2 == 0:
            boxes = torch.tensor([[10.0, 12.0, 30.0, 40.0]], dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}
        else:
            target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}
        samples.append(Sample(img, target, {}))
    return Flux(source=samples)


# --- pure helper tests -------------------------------------------------------


def test_sample_image_chw_float_to_hwc_bgr_uint8() -> None:
    chw_float = torch.rand(3, 32, 48)
    arr = _sample_image_to_bgr_hwc_uint8(chw_float, target_channels=3)
    assert arr.shape == (32, 48, 3)
    assert arr.dtype == np.uint8


def test_sample_image_hwc_uint8_passthrough_in_bgr() -> None:
    hwc_uint8 = (np.random.rand(32, 48, 3) * 255).astype(np.uint8)
    out = _sample_image_to_bgr_hwc_uint8(hwc_uint8, target_channels=3)
    assert out.shape == (32, 48, 3)
    assert out.dtype == np.uint8
    # RGB -> BGR swap means channel 0 of out equals channel 2 of input.
    assert (out[..., 0] == hwc_uint8[..., 2]).all()
    assert (out[..., 2] == hwc_uint8[..., 0]).all()


def test_sample_image_grayscale_expanded_to_3_channels() -> None:
    gray_hw = torch.rand(32, 32)
    arr = _sample_image_to_bgr_hwc_uint8(gray_hw, target_channels=3)
    assert arr.shape == (32, 32, 3)


def test_input_hw_chw_vs_hwc() -> None:
    assert _input_hw(torch.zeros(3, 64, 96)) == (64, 96)
    assert _input_hw(np.zeros((64, 96, 3))) == (64, 96)
    assert _input_hw(torch.zeros(64, 96)) == (64, 96)


def test_input_hw_rejects_unexpected_shape() -> None:
    with pytest.raises(ValueError):
        _input_hw(torch.zeros(2, 3, 4, 5))


def test_target_to_yolo_label_xyxy_to_cxcywh_normalized() -> None:
    target = {
        "boxes": torch.tensor([[10.0, 12.0, 30.0, 40.0]], dtype=torch.float32),
        "labels": torch.tensor([2], dtype=torch.int64),
    }
    cls, bboxes = _target_to_yolo_label(target, h=80, w=100)
    assert cls.shape == (1, 1)
    assert cls.dtype == np.float32
    assert cls[0, 0] == 2.0
    assert bboxes.shape == (1, 4)
    assert bboxes.dtype == np.float32
    # cx = 20/100 = 0.2, cy = 26/80 = 0.325, w = 20/100 = 0.2, h = 28/80 = 0.35
    np.testing.assert_allclose(bboxes[0], [0.2, 26.0 / 80.0, 0.2, 28.0 / 80.0], rtol=1e-5)


def test_target_to_yolo_label_empty_target() -> None:
    cls, bboxes = _target_to_yolo_label({}, h=64, w=64)
    assert cls.shape == (0, 1) and cls.dtype == np.float32
    assert bboxes.shape == (0, 4) and bboxes.dtype == np.float32


def test_target_to_yolo_label_none_returns_empty() -> None:
    cls, bboxes = _target_to_yolo_label(None, h=64, w=64)
    assert cls.shape == (0, 1)
    assert bboxes.shape == (0, 4)


# --- FluxYOLODataset integration --------------------------------------------


def _make_dataset(augment: bool = False) -> FluxYOLODataset:
    flux = _tiny_flux(n=4, h=64, w=64)
    args = get_cfg(
        overrides={
            "task": "detect",
            "data": "<flux>",
            "imgsz": 64,
            "mode": "train" if augment else "val",
            "mosaic": 1.0 if augment else 0.0,
            "plots": False,
        }
    )
    return FluxYOLODataset(
        flux=flux,
        data={"path": ".", "train": ".", "val": ".", "names": {0: "thing"}, "nc": 1, "channels": 3},
        imgsz=64,
        augment=augment,
        hyp=args,
        batch_size=2,
        stride=32,
        rect=not augment,
    )


def test_dataset_get_labels_shape_keys() -> None:
    ds = _make_dataset(augment=False)
    assert len(ds.labels) == 4
    for lb in ds.labels:
        assert {"im_file", "cls", "bboxes", "segments", "keypoints", "normalized", "bbox_format"} <= set(lb.keys()) | {
            "shape",
            "rect_shape",
        }
        assert lb["bbox_format"] == "xywh"
        assert lb["normalized"] is True
        assert lb["cls"].dtype == np.float32
        assert lb["bboxes"].dtype == np.float32
        assert lb["im_file"].startswith("flux://")


def test_dataset_getitem_format_pipeline_yields_yolo_keys() -> None:
    """Indexing into the dataset must run the YOLODataset Format transform
    chain — that's what produces the ``img``/``cls``/``bboxes``/``batch_idx``
    contract the trainer collate_fn expects.
    """
    ds = _make_dataset(augment=False)
    sample = ds[0]
    assert isinstance(sample, dict)
    assert "img" in sample
    assert "cls" in sample
    assert "bboxes" in sample
    assert "batch_idx" in sample
    img = sample["img"]
    assert isinstance(img, torch.Tensor)
    # YOLO Format normalizes images to CHW float (or uint8 depending on Format args).
    assert img.ndim == 3
    assert img.shape[0] == 3  # 3 channels
    cls = sample["cls"]
    bboxes = sample["bboxes"]
    assert isinstance(cls, torch.Tensor)
    assert isinstance(bboxes, torch.Tensor)


def test_dataset_load_image_returns_bgr_hwc_uint8() -> None:
    ds = _make_dataset(augment=False)
    arr, hw_orig, hw_resized = ds.load_image(0, rect_mode=True)
    assert arr.dtype == np.uint8
    assert arr.shape[-1] == 3  # 3 channels
    assert hw_orig == (64, 64)
    # imgsz=64 and orig=64 → no resize happens; resized == orig.
    assert hw_resized == (64, 64)


def test_dataset_collate_fn_produces_stacked_batch() -> None:
    ds = _make_dataset(augment=False)
    items = [ds[i] for i in range(2)]
    batch = ds.collate_fn(items)
    assert "img" in batch
    assert batch["img"].shape[0] == 2
    assert batch["batch_idx"].ndim == 1


def test_dataset_cache_labels_raises() -> None:
    ds = _make_dataset(augment=False)
    with pytest.raises(NotImplementedError):
        cast(object, ds.cache_labels("/tmp/dummy"))


def test_load_image_follows_rect_mode_permutation() -> None:
    """In rect mode, ``BaseDataset.set_rectangle`` reorders ``self.im_files``
    and ``self.labels`` together by aspect-ratio sort. ``load_image(i)`` must
    return the sample that ``self.labels[i]`` describes — i.e. it must look
    up the Flux index via ``self.im_files[i]`` (which carries ``flux://j``)
    rather than indexing the Flux directly with ``i``.

    Regression for a silent val-mode-only bug: training (rect=False) was
    correct, but validation paired random images with random labels because
    ``np.argsort`` permutes equal-AR slots non-trivially → every val mAP run
    came back near zero.
    """
    n = 12
    samples: List[Sample] = []
    for i in range(n):
        # Slightly varied (h, w) → argsort guaranteed to permute slots.
        h, w = 60 + (i % 4), 64 - (i % 4)
        # Tag each image with its source-flux index in pixel (0, 0, 0).
        img = torch.full((3, h, w), float(i) / 255.0)
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)
        samples.append(Sample(img, {"boxes": boxes, "labels": labels}, {}))
    flux = Flux(source=samples)

    args = get_cfg(overrides={"task": "detect", "data": "<flux>", "imgsz": 64, "mode": "val", "plots": False})
    ds = FluxYOLODataset(
        flux=flux,
        data={"path": ".", "train": ".", "val": ".", "names": {0: "bg", 1: "fg"}, "nc": 2, "channels": 3},
        imgsz=64,
        augment=False,
        hyp=args,
        batch_size=4,
        stride=32,
        rect=True,
    )
    # set_rectangle should have permuted at least some slots even with all-equal AR.
    permuted_slots = sum(1 for i, f in enumerate(ds.im_files) if int(f.rsplit("/", 1)[-1]) != i)
    assert permuted_slots > 0, (
        "test precondition failed — set_rectangle did not permute any slot. "
        "Adjust n until argsort returns a non-identity permutation, otherwise "
        "this regression cannot bite."
    )

    for i in range(n):
        arr, _, _ = ds.load_image(i)
        decoded = int(arr[0, 0, 0])  # tag we wrote into the source image
        expected = int(ds.im_files[i].rsplit("/", 1)[-1])
        assert decoded == expected, (
            f"slot i={i}: load_image returned image tagged source={decoded} "
            f"but labels[{i}] describes flux index {expected}"
        )


def test_dataset_empty_flux_raises() -> None:
    flux = Flux(source=[])
    args = get_cfg(overrides={"task": "detect", "data": "<flux>", "imgsz": 64, "mode": "val", "plots": False})
    with pytest.raises(RuntimeError, match="empty"):
        FluxYOLODataset(
            flux=flux,
            data={"path": ".", "train": ".", "val": ".", "names": {0: "thing"}, "nc": 1, "channels": 3},
            imgsz=64,
            augment=False,
            hyp=args,
            batch_size=2,
            stride=32,
            rect=True,
        )
