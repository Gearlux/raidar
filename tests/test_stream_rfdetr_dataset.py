"""Tests for the in-memory RF-DETR dataset bridge (``raidar.stream_rfdetr_dataset``).

The Stream dataset feeds RF-DETR's native PTL stack with NO disk materialization,
mirroring the per-item contract the disk ``CocoDetection`` produces: a
``([C,H,W] float32 ImageNet-normalized, target)`` pair with cxcywh-normalized,
0-indexed-label targets. ``rfdetr[train]`` extras are required, so the whole
module skips when they're absent.
"""

from typing import Any

import pytest

pytest.importorskip("rfdetr.training")  # the [train,loggers] extras

import torch  # noqa: E402
from recordstream import Image, Record, Regions  # noqa: E402
from recordstream.core import Stream  # noqa: E402
from rfdetr.config import RFDETRNanoConfig, TrainConfig  # noqa: E402

import raidar.stream_rfdetr_dataset as frd  # noqa: E402
from raidar.stream_rfdetr_dataset import StreamRFDETRDataModule, StreamRFDETRDataset  # noqa: E402


def _det_record(image: Any, boxes: Any, labels: Any, idx: int) -> Record:
    """A detection record: Image under "image" + Regions under "target" (the CocoToTorchVisionDetection shape)."""
    return {"image": Image(image.numpy(), layout="CHW"), "target": Regions(boxes=boxes, labels=labels), "idx": idx}


def _stream(n: int = 4, *, h: int = 40, w: int = 60, empty: bool = False) -> Stream:
    records = []
    for i in range(n):
        if empty:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor([[4.0, 4.0, 20.0, 24.0]])
            labels = torch.tensor([1])
        records.append(_det_record(torch.rand(3, h, w), boxes, labels, i))
    return Stream(source=records)


def _val_dataset(stream: Stream, resolution: int = 256) -> StreamRFDETRDataset:
    # square_resize_div_64 path (RF-DETR's default); a plain namespace stands in for build_namespace.
    from types import SimpleNamespace

    ns = SimpleNamespace(
        square_resize_div_64=True,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        patch_size=16,
        num_windows=4,
        aug_config=None,
    )
    return StreamRFDETRDataset(stream, "val", resolution, ns)


# ---------------------------------------------------------------------------
# Per-item contract
# ---------------------------------------------------------------------------


def test_item_contract() -> None:
    ds = _val_dataset(_stream(2, h=40, w=60), resolution=256)
    img, target = ds[0]
    assert isinstance(img, torch.Tensor) and img.dtype == torch.float32 and img.shape[0] == 3
    # ImageNet-normalize ran → values escape [0,1].
    assert float(img.min()) < 0.0 < 1.0 < float(img.max())
    assert set(target) == {"boxes", "labels", "image_id", "area", "iscrowd", "orig_size", "size"}
    boxes = target["boxes"]
    assert boxes.shape == (1, 4) and boxes.dtype == torch.float32
    # cxcywh normalized to [0, 1].
    assert float(boxes.min()) >= 0.0 and float(boxes.max()) <= 1.0
    assert target["labels"].dtype == torch.int64 and target["labels"].tolist() == [1]
    assert target["orig_size"].tolist() == [40, 60]  # [H, W]


def test_labels_stay_zero_indexed_below_num_classes() -> None:
    # RF-DETR labels are 0-indexed 0..K-1 (background at K); the dataset must not shift them.
    ds = _val_dataset(_stream(1), resolution=256)
    _, target = ds[0]
    assert int(target["labels"].max()) < 2  # num_classes=2 → labels in {0, 1}


def test_empty_target_record() -> None:
    ds = _val_dataset(_stream(1, empty=True), resolution=256)
    img, target = ds[0]
    assert img.shape[0] == 3
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].tolist() == []


def test_len_matches_stream() -> None:
    ds = _val_dataset(_stream(5), resolution=256)
    assert len(ds) == 5


# ---------------------------------------------------------------------------
# Projection-optimized label cache (no image decode at construction)
# ---------------------------------------------------------------------------


def test_labels_cached_via_projection_without_image_decode(monkeypatch: Any) -> None:
    stream = _stream(4)

    # Spy: image decode must NOT fire while caching labels (only per __getitem__).
    decode_calls = {"n": 0}
    real_decode = frd._record_input_to_rgb_pil

    def _counting_decode(arr: Any) -> Any:
        decode_calls["n"] += 1
        return real_decode(arr)

    monkeypatch.setattr(frd, "_record_input_to_rgb_pil", _counting_decode)

    # Spy: labels are cached via the key-restricted projection helper.
    projection_calls = {"n": 0}
    real_iter_key = frd.iter_key

    def _counting_iter_key(src: Any, key: str) -> Any:
        projection_calls["n"] += 1
        return real_iter_key(src, key)

    monkeypatch.setattr(frd, "iter_key", _counting_iter_key)

    ds = _val_dataset(stream, resolution=256)
    assert projection_calls["n"] == 1, "labels must be cached via iter_key (target-key-only projection)"
    assert decode_calls["n"] == 0, "no image should be decoded while caching labels"
    assert len(ds._labels) == 4

    _ = ds[0]  # now an image IS decoded
    assert decode_calls["n"] == 1


# ---------------------------------------------------------------------------
# DataModule injection (no disk build_dataset)
# ---------------------------------------------------------------------------


def _configs(num_classes: int = 2) -> Any:
    return RFDETRNanoConfig(num_classes=num_classes), TrainConfig(dataset_dir=".", batch_size=2, num_workers=0)


def test_datamodule_injects_stream_dataset_and_never_builds_from_disk(monkeypatch: Any) -> None:
    import rfdetr.training.module_data as md

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("build_dataset must NOT be called in stream mode")

    monkeypatch.setattr(md, "build_dataset", _boom)

    mc, tc = _configs()
    dm = StreamRFDETRDataModule(mc, tc, train_stream=_stream(4), val_stream=_stream(2), class_names=["cat", "dog"])
    dm.setup("fit")
    assert isinstance(dm._dataset_train, StreamRFDETRDataset) and len(dm._dataset_train) == 4
    assert isinstance(dm._dataset_val, StreamRFDETRDataset)


def test_datamodule_class_names_override_without_coco() -> None:
    mc, tc = _configs()
    dm = StreamRFDETRDataModule(mc, tc, train_stream=_stream(4), val_stream=_stream(2), class_names=["cat", "dog"])
    dm.setup("fit")
    # No `.coco` attribute on the in-memory dataset; class_names comes from the knob.
    assert getattr(dm._dataset_train, "coco", None) is None
    assert dm.class_names == ["cat", "dog"]


def test_datamodule_dataloader_yields_nested_tensor_batch() -> None:
    mc, tc = _configs()
    dm = StreamRFDETRDataModule(mc, tc, train_stream=_stream(4), val_stream=_stream(2), class_names=["cat", "dog"])
    dm.setup("fit")
    samples, targets = next(iter(dm.train_dataloader()))
    # RF-DETR's collate builds a NestedTensor (.tensors [B,C,H,W], .mask) + a list of targets.
    assert hasattr(samples, "tensors") and samples.tensors.ndim == 4
    assert isinstance(targets, (list, tuple)) and len(targets) == 2
    assert set(targets[0]) >= {"boxes", "labels", "orig_size"}


def test_datamodule_test_falls_back_to_val() -> None:
    mc, tc = _configs()
    dm = StreamRFDETRDataModule(mc, tc, train_stream=_stream(4), val_stream=_stream(2), class_names=["cat", "dog"])
    dm.setup("test")
    assert isinstance(dm._dataset_test, StreamRFDETRDataset)


# ---------------------------------------------------------------------------
# Image normalization helper
# ---------------------------------------------------------------------------


def test_record_input_variants_to_rgb_pil() -> None:
    import numpy as np
    from PIL import Image

    chw = frd._record_input_to_rgb_pil(torch.rand(3, 8, 10))
    assert isinstance(chw, Image.Image) and chw.mode == "RGB" and chw.size == (10, 8)  # (W, H)
    hw_gray = frd._record_input_to_rgb_pil(np.zeros((8, 10), dtype=np.uint8))
    assert hw_gray.mode == "RGB" and hw_gray.size == (10, 8)
    pil_in = frd._record_input_to_rgb_pil(Image.new("L", (10, 8)))
    assert pil_in.mode == "RGB"
