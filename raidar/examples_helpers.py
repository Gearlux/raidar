"""Tiny test/example helpers — kept inside the package so example YAMLs
can reference them via ``!class:raidar.examples_helpers.X`` without forcing
users to add their own scaffolding for a smoke run.

These helpers exist ONLY to make the in-tree example YAMLs runnable
end-to-end. Real training jobs should provide a real Stream pipeline
(typically via waivefront op chains over annotated data).
"""

from typing import Any, List

import torch
from confluid import configurable
from recordstream import Image, Record, Regions
from torch import Tensor
from torch.utils.data import Dataset


@configurable
class SyntheticDetectionDataset(Dataset):
    """Synthetic detection dataset producing one box per image at a fixed location.

    Useful only for smoke-testing the wiring of LightningDetector +
    raidar's detector wrappers. Replace with a real Stream pipeline for actual
    training.

    Args:
        n: Number of records to generate.
        num_classes: Number of detection classes (label drawn modulo this).
        image_size: Square image side in pixels.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        n: int = 16,
        num_classes: int = 91,
        image_size: int = 256,
        seed: int = 0,
    ) -> None:
        self.n = int(n)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self._rng = torch.Generator().manual_seed(int(seed))
        # Pre-generate consistent image content so __getitem__ is deterministic.
        self._images: List[Tensor] = [
            torch.rand(3, self.image_size, self.image_size, generator=self._rng) for _ in range(self.n)
        ]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Record:
        if idx < 0:
            idx += self.n
        if not 0 <= idx < self.n:
            raise IndexError(idx)
        image = self._images[idx]
        # One foreground box centred in the image, class id varies by index.
        s = self.image_size
        boxes: Any = torch.tensor([[s * 0.25, s * 0.25, s * 0.75, s * 0.75]], dtype=torch.float32)
        labels: Any = torch.tensor([1 + (idx % max(self.num_classes - 1, 1))], dtype=torch.int64)
        # "target" matches the detection pipeline's convention key (the default the
        # CocoToTorchVisionDetection / MasksToDetectionBoxes ops write and the collate reads).
        return {
            "image": Image(image.numpy(), layout="CHW"),
            "target": Regions(boxes=boxes, labels=labels),
            "idx": idx,
        }
