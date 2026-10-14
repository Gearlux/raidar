"""Tests for the in-memory RF-DETR dataset bridge (``raidar.stream_rfdetr_dataset``).

The Stream dataset feeds RF-DETR's native PTL stack with NO disk materialization,
mirroring the per-item contract the disk ``CocoDetection`` produces: a
``([C,H,W] float32 ImageNet-normalized, target)`` pair with cxcywh-normalized,
0-indexed-label targets. ``rfdetr[train]`` extras are required, so the whole
module skips when they're absent.
"""

import random
from typing import Any, Tuple

import numpy as np
import pytest

pytest.importorskip("rfdetr.training")  # the [train,loggers] extras

import torch  # noqa: E402
from recordstream import Boxes, Image, Record  # noqa: E402
from recordstream.core import Stream  # noqa: E402
from rfdetr.config import RFDETRNanoConfig, TrainConfig  # noqa: E402

import raidar.stream_rfdetr_dataset as frd  # noqa: E402
from raidar.stream_rfdetr_dataset import StreamRFDETRDataModule, StreamRFDETRDataset  # noqa: E402
from raidar.yolo_sink import POLYGONS  # noqa: E402


def _det_record(image: Any, boxes: Any, labels: Any, idx: int) -> Record:
    """A detection record: Image under "image" + Boxes under "target" (the CocoToTorchVisionDetection shape)."""
    return {"image": Image(image.numpy(), layout="CHW"), "target": Boxes(boxes=boxes, labels=labels), "idx": idx}


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
    # square_resize_div_64 path (RF-DETR's default); a plain namespace stands in for _namespace_from_configs.
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
# Nothing read up front; a picture and its labels come from ONE read
# ---------------------------------------------------------------------------


def test_nothing_is_read_up_front_and_each_item_reads_its_record_once(monkeypatch: Any) -> None:
    """The labels are read with the picture, per item: a random step in the stream (a crop) then cuts both alike.
    Before 2026-10-12 every target was walked once at construction and reused for every epoch."""
    decode_calls = {"n": 0}
    real_decode = frd._record_input_to_rgb_pil

    def _counting_decode(arr: Any) -> Any:
        decode_calls["n"] += 1
        return real_decode(arr)

    monkeypatch.setattr(frd, "_record_input_to_rgb_pil", _counting_decode)
    ds = _val_dataset(_stream(4), resolution=256)
    assert decode_calls["n"] == 0
    _ = ds[0]
    assert decode_calls["n"] == 1


class _Counter:
    """A stream step that paints every pixel 10 * k and labels the one box k, k counting the calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, record: Record) -> Record:
        self.calls += 1
        pixels = np.full((40, 60, 3), 10 * self.calls, dtype=np.uint8)
        return {
            **record,
            "image": Image(pixels, layout="HWC"),
            "target": Boxes(boxes=torch.tensor([[4.0, 4.0, 20.0, 24.0]]), labels=torch.tensor([self.calls])),
        }


def test_a_pictures_labels_come_from_the_read_that_gave_the_picture() -> None:
    counter = _Counter()
    ds = _val_dataset(Stream(source=[{"idx": 0}], ops=[counter]), resolution=64)
    for _ in range(3):
        img, target = ds[0]
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        value = float(((img * std + mean) * 255.0).mean())
        assert round(value / 10.0) == int(target["labels"][0])


# ---------------------------------------------------------------------------
# A seg model's masks, from the instances' outlines
# ---------------------------------------------------------------------------


def _seg_ns(*, square: bool = True, multi_scale: bool = False) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        square_resize_div_64=square,
        multi_scale=multi_scale,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        patch_size=16,
        num_windows=4,
        aug_config=None,
        segmentation_head=True,
    )


def _seg_record(height: int = 40, width: int = 60) -> Record:
    ring = [[30.0, 10.0], [50.0, 10.0], [50.0, 30.0], [30.0, 30.0]]
    return {
        "image": Image(np.zeros((height, width, 3), dtype=np.uint8), layout="HWC"),
        "target": Boxes(
            boxes=np.array([[30.0, 10.0, 50.0, 30.0], [4.0, 4.0, 20.0, 24.0]], dtype=np.float32),
            labels=np.array([0, 1], dtype=np.int64),
            canvas=(height, width),
            extras={POLYGONS: [[ring], []]},
        ),
    }


def _box_share(mask: torch.Tensor, box: torch.Tensor) -> Tuple[float, float]:
    """The share of the box (cxcywh, normalized) the mask lights, and the share of the mask inside the box."""
    height, width = mask.shape
    cx, cy, w, h = box.tolist()
    x0, x1 = int(round((cx - w / 2) * width)), int(round((cx + w / 2) * width))
    y0, y1 = int(round((cy - h / 2) * height)), int(round((cy + h / 2) * height))
    inside = mask[y0:y1, x0:x1]
    return float(inside.float().mean()), float(inside.sum()) / max(float(mask.sum()), 1.0)


def test_a_seg_model_gets_each_instances_mask_from_its_outline_and_a_box_without_one_is_its_box() -> None:
    ds = StreamRFDETRDataset(Stream(source=[_seg_record()]), "val", 64, _seg_ns())
    img, target = ds[0]
    masks = target["masks"]
    assert masks.shape == (2,) + tuple(img.shape[-2:])
    for mask, box in zip(masks, target["boxes"]):
        filled, inside = _box_share(mask, box)
        assert filled > 0.9 and inside > 0.9


def test_a_box_only_model_gets_no_masks() -> None:
    ds = _val_dataset(Stream(source=[_seg_record()]), resolution=64)
    assert "masks" not in ds[0][1]


def test_cut_pictures_of_any_height_batch_into_masks_of_one_size() -> None:
    """Pictures 100, 1008 and 3000 rows tall, as RF-DETR's shape-keeping training resize sees them: each cut to one
    48 x 48 piece first, their masks join into one batch; uncut, the batch stops on the masks' sizes — the run that
    stopped on 2026-10-12 (``Sizes of tensors must match except in dimension 0``)."""
    from rfdetr.utilities.tensors import collate_fn

    from raidar.crop import RandomCropDetection

    records = [_seg_record(height, 48) for height in (100, 1008, 3000, 1008)]
    random.seed(0)
    torch.manual_seed(0)
    cut = StreamRFDETRDataset(
        Stream(source=records, ops=[RandomCropDetection(height=48, width=48)]), "train", 48, _seg_ns(square=False)
    )
    _, targets = collate_fn([cut[i] for i in range(len(records))])
    assert torch.cat([t["masks"] for t in targets]).shape[-2:] == targets[0]["masks"].shape[-2:]
    uncut = StreamRFDETRDataset(Stream(source=records), "train", 48, _seg_ns(square=False))
    _, targets = collate_fn([uncut[i] for i in range(len(records))])
    with pytest.raises(RuntimeError, match="Sizes of tensors must match except in dimension 0"):
        torch.cat([t["masks"] for t in targets])


# ---------------------------------------------------------------------------
# keep_picture_size: each picture at its own size, RF-DETR's resize skipped
# ---------------------------------------------------------------------------


def _tall_record(height: int = 204, width: int = 48, value: int = 0) -> Record:
    """A ``height`` x ``width`` grey picture whose left half is ``value`` and right half 200, one box and outline."""
    pixels = np.full((height, width, 3), 200, dtype=np.uint8)
    pixels[:, : width // 2] = value
    ring = [[4.0, 10.0], [20.0, 10.0], [20.0, 150.0], [4.0, 150.0]]
    return {
        "image": Image(pixels, layout="HWC"),
        "target": Boxes(
            boxes=np.array([[4.0, 10.0, 20.0, 150.0]], dtype=np.float32),
            labels=np.array([1], dtype=np.int64),
            canvas=(height, width),
            extras={POLYGONS: [[ring]]},
        ),
        "file": f"picture_{height}x{width}.png",
    }


@pytest.mark.parametrize("image_set", ["train", "val"])
def test_a_kept_picture_reaches_the_model_at_its_own_size_where_the_resize_would_change_it(image_set: str) -> None:
    """A 48 x 204 picture (width x height) at resolution 64: RF-DETR's resize makes it 64 tall at most; kept, it stays
    48 x 204, its box normalized by its own size and its mask the picture's size."""
    kept = StreamRFDETRDataset(Stream(source=[_tall_record()]), image_set, 64, _seg_ns(), keep_picture_size=True)
    img, target = kept[0]
    assert tuple(img.shape) == (3, 204, 48)
    assert tuple(target["masks"].shape) == (1, 204, 48)
    cx, cy, w, h = target["boxes"][0].tolist()
    assert (w, h) == (pytest.approx(16 / 48), pytest.approx(140 / 204))
    assert cy == pytest.approx(80 / 204)
    assert cx in (pytest.approx(12 / 48), pytest.approx(36 / 48)), "the box, flipped or not"
    resized, _ = StreamRFDETRDataset(Stream(source=[_tall_record()]), image_set, 64, _seg_ns())[0]
    assert tuple(resized.shape[-2:]) != (204, 48)


def test_a_kept_val_picture_is_the_picture_normalized_and_nothing_else() -> None:
    img, _ = StreamRFDETRDataset(Stream(source=[_tall_record()]), "val", 64, _seg_ns(), keep_picture_size=True)[0]
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    expected_left = (0.0 - mean) / std
    expected_right = (200.0 / 255.0 - mean) / std
    assert torch.allclose(img[:, :, :24], expected_left[:, None, None].expand(3, 204, 24), atol=1e-6)
    assert torch.allclose(img[:, :, 24:], expected_right[:, None, None].expand(3, 204, 24), atol=1e-6)


def test_kept_training_pictures_keep_rf_detrs_flip_and_an_empty_aug_config_turns_it_off() -> None:
    """RF-DETR's default augmentation (``aug_config`` None) is a horizontal flip at 0.5: over 40 draws both sides turn
    up; ``aug_config: {}`` — RF-DETR's spelling of no augmentation — never flips."""
    stream = Stream(source=[_tall_record()])
    torch.manual_seed(0)
    random.seed(0)
    flipped = StreamRFDETRDataset(stream, "train", 64, _seg_ns(), keep_picture_size=True)
    lefts = {round(float(flipped[0][0][0, 0, 0]), 3) for _ in range(40)}
    assert len(lefts) == 2, "dark on the left in some draws, bright in others"
    ns = _seg_ns()
    ns.aug_config = {}
    unflipped = StreamRFDETRDataset(stream, "train", 64, ns, keep_picture_size=True)
    assert {round(float(unflipped[0][0][0, 0, 0]), 3) for _ in range(20)} == {round((0.0 - 0.485) / 0.229, 3)}


def test_a_kept_picture_of_another_size_is_refused_naming_both_sizes_and_the_file() -> None:
    """A seg batch joins its masks as they come, so kept pictures of two sizes would stop the run at the batch with
    ``Sizes of tensors must match``; the dataset refuses the second size where it is read."""
    stream = Stream(source=[_tall_record(), _tall_record(height=216)])
    dataset = StreamRFDETRDataset(stream, "val", 64, _seg_ns(), keep_picture_size=True)
    dataset[0]
    with pytest.raises(
        ValueError,
        match=r"keep_picture_size: picture 1 \(picture_216x48\.png\) is 48 x 216 \(width x height\), "
        r"the first was 48 x 204",
    ):
        dataset[1]


def test_the_kept_size_is_not_a_default() -> None:
    dataset = StreamRFDETRDataset(Stream(source=[_tall_record(), _tall_record(height=216)]), "val", 64, _seg_ns())
    assert tuple(dataset[1][0].shape[-2:]) != (216, 48), "resized, and two sizes pass as they always did"


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
    # RF-DETR's collate builds a NestedTensor (.tensors [B,C,H,W], .mask) + a SEQUENCE of per-sample
    # target dicts. Its type is rfdetr's to choose — a plain list up to 1.9, a `PackedTargets` from
    # 1.11 (one tensor per field, indexable and iterable as the same dicts) — so only the sequence
    # behaviour raidar reads (`len`, `[i]`, `for t in targets`) is pinned here.
    assert hasattr(samples, "tensors") and samples.tensors.ndim == 4
    assert len(targets) == 2
    assert set(targets[0]) >= {"boxes", "labels", "orig_size"}
    assert [set(t) >= {"boxes", "labels"} for t in targets] == [True, True]


def test_the_datamodule_hands_keep_picture_size_to_every_dataset_it_makes() -> None:
    mc, tc = _configs()
    records = [_tall_record() for _ in range(2)]
    dm = StreamRFDETRDataModule(
        mc,
        tc,
        train_stream=Stream(source=records),
        val_stream=Stream(source=records),
        class_names=["cat", "dog"],
        keep_picture_size=True,
    )
    dm.setup("fit")
    dm.setup("test")
    for dataset in (dm._dataset_train, dm._dataset_val, dm._dataset_test):
        assert isinstance(dataset, StreamRFDETRDataset)
        assert tuple(dataset[0][0].shape[-2:]) == (204, 48)


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
