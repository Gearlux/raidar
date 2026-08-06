"""The torch-contract batch adapter — what is left after the collate moved upstream.

``collate_records`` (recordstream's own) batches a detection record correctly on its own: a
variable-size image column stays a per-image list, a variable-N ``Regions`` keeps per-record
columns, and ``batch_regions`` transposes those back. What raidar still owns is ONE fact —
the torchvision-native forward takes ``List[Tensor]`` — so this file tests exactly that
adapter and nothing about batching itself (recordstream's ``tests/test_batch.py`` owns that).
"""

from typing import List

import numpy as np
import pytest
import torch
from recordstream import Image, Record, Regions, collate_records

from raidar.torch_detection import native_batch


def _records(counts: tuple = (1, 3), sizes: tuple = (8, 12)) -> List[Record]:
    return [
        {
            "image": torch.rand(3, size, size),
            "target": Regions(boxes=torch.rand(n, 4), labels=torch.zeros(n, dtype=torch.int64)),
            "idx": i,
        }
        for i, (n, size) in enumerate(zip(counts, sizes))
    ]


def test_it_reads_the_native_pair_off_a_plain_collated_record() -> None:
    """The whole point: NO raidar collate — `collate_records` in, the native contract out."""
    images, targets = native_batch(collate_records(_records()))
    assert [tuple(i.shape) for i in images] == [(3, 8, 8), (3, 12, 12)]
    assert targets is not None
    assert [tuple(t["boxes"].shape) for t in targets] == [(1, 4), (3, 4)]
    assert [tuple(t["labels"].shape) for t in targets] == [(1,), (3,)]


def test_variable_sized_images_survive_as_a_list() -> None:
    """Never padded, never dropped — the detection batch convention, now the engine's default."""
    images, _ = native_batch(collate_records(_records(sizes=(8, 32))))
    assert [tuple(i.shape) for i in images] == [(3, 8, 8), (3, 32, 32)]


def test_a_stacked_uniform_image_column_still_yields_per_image_tensors() -> None:
    """A config without `ToTensor` leaves `Image` items, which the collate STACKS — the adapter
    normalizes both shapes, because torchvision's forward takes a list either way."""
    records: List[Record] = [
        {
            "image": Image(np.random.rand(3, 8, 8).astype("float32"), layout="CHW"),
            "target": Regions(boxes=torch.rand(1, 4), labels=torch.zeros(1, dtype=torch.int64)),
        }
        for _ in range(2)
    ]
    images, _ = native_batch(collate_records(records))
    assert len(images) == 2 and all(tuple(i.shape) == (3, 8, 8) for i in images)


def test_a_batch_without_targets_yields_none() -> None:
    """The prediction path: no ground-truth column, so no targets to read."""
    images, targets = native_batch(collate_records([{"image": torch.rand(3, 8, 8)}]))
    assert len(images) == 1 and targets is None


def test_configured_keys_are_honoured() -> None:
    batch = collate_records([{"pixels": torch.rand(3, 8, 8), "boxes": Regions(boxes=torch.rand(2, 4))}])
    images, targets = native_batch(batch, input_key="pixels", target_key="boxes")
    assert len(images) == 1 and targets is not None and tuple(targets[0]["boxes"].shape) == (2, 4)


def test_metadata_stays_readable_through_recordstreams_transpose() -> None:
    """The sink path: the columns the model did not consume, per record — `batch_metadata`."""
    from recordstream import batch_metadata

    batch = collate_records(_records())
    assert batch_metadata(batch, exclude=("image", "target")) == [{"idx": 0}, {"idx": 1}]


def test_there_is_no_raidar_collate_left_to_import() -> None:
    """The deleted surface must not come back: batching a record is recordstream's job."""
    import raidar.torch_detection as mod

    for gone in ("detection_collate_fn", "DetectionBatchInput", "batch_detection"):
        assert not hasattr(mod, gone), f"{gone} is back — batching belongs to recordstream"
    with pytest.raises(ImportError):
        from raidar import detection_collate_fn  # noqa: F401
