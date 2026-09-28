"""Tests for the specialised :class:`raidar.lightning_detector.LightningDetector`.

These tests cover the **native** path — overridden ``training_step`` /
``validation_step`` / ``test_step`` consume torchvision-native models
(``forward(images, targets) → loss dict`` / ``forward(images) → predictions
list``) and torchmetrics detection metrics directly, with no
``TorchvisionDetectionLoss`` / ``DetectionMetricAdapter`` layer involved.
"""

from typing import Any, Dict, List, Optional, Union, cast, get_args

import pytest
import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import PartialClass
from recordstream import Boxes, Image, Record, collate_list
from recordstream.core import Stream
from recordstream.runnable import runnable_entrypoints
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.detection import MeanAveragePrecision

from raidar.lightning_detector import DetectorTask, LightningDetector, _model_num_classes
from raidar.torch_detection import DetectionConfusionMatrix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NativeDetector(nn.Module):
    """Stub detector that conforms to the torchvision-native contract.

    * Train mode: ``forward(images, targets) → {"loss_a": .., "loss_b": ..}``
      where each loss depends on a learnable parameter (so gradients flow).
    * Eval mode: ``forward(images) → List[Dict[str, Tensor]]`` with one empty
      prediction per image (enough for ``MeanAveragePrecision.update``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            device = images[0].device
            return {
                "loss_a": (self.alpha * torch.ones((), device=device)).sum(),
                "loss_b": (0.5 * self.alpha * torch.ones((), device=device)).sum(),
            }
        out: List[Dict[str, Tensor]] = []
        for _ in images:
            out.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            )
        return out


def _regions(boxes: Any, labels: Any) -> Boxes:
    """A detection-target Boxes carrying tensor boxes/labels (as the typed detection op produces)."""
    return Boxes(boxes=boxes, labels=labels)


def _make_records(n: int = 4, size: int = 32) -> List[Record]:
    """Per-record records shaped like a detection source: Image + Boxes target + plain meta key."""
    records: List[Record] = []
    for i in range(n):
        records.append(
            {
                "image": Image(torch.rand(3, size, size).numpy(), layout="CHW"),
                "target": _regions(
                    torch.tensor([[4.0, 4.0, 20.0, 20.0]], dtype=torch.float32),
                    torch.tensor([1], dtype=torch.int64),
                ),
                "idx": i,
            }
        )
    return records


def _make_batch(n: int = 2, size: int = 32) -> Record:
    return cast(Dict[str, Any], collate_list(_make_records(n=n, size=size)))


def _make_stream(n: int = 4, size: int = 32) -> Stream:
    # A Stream (torch Dataset) — the trainer's train_set/val_set/test_set are typed
    # Dataset[Any] | Fluid; a bare list[Record] is normalized to a Stream only at run
    # time (ensure_record_dataset), so wire datasets as a Stream up front.
    return Stream(source=_make_records(n=n, size=size))


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------


def test_loss_fn_is_none() -> None:
    """Specialised trainer drops the loss_fn parameter entirely — native
    detection models compute loss inside forward."""
    trainer = LightningDetector(
        model=_NativeDetector(),
        train_set=_make_stream(4),
        num_classes=2,
    )
    assert trainer.loss_fn is None


def test_there_is_deliberately_no_train_metrics_slot() -> None:
    """A torchvision-native model in train mode returns ONLY the loss dict — there are no
    train-split predictions to score, so a `train_metrics` slot would accept a wired set and
    silently never call it (the deltaid huggingface-backend precedent)."""
    import inspect

    assert "train_metrics" not in inspect.signature(LightningDetector.__init__).parameters
    trainer = LightningDetector(model=_NativeDetector(), train_set=_make_stream(4), num_classes=2)
    # Since the 2026-08-06 consolidation the shared base CARRIES the slot for the tasks that
    # have one; here it is pinned to None and wiring one is a LOUD error — stronger than the
    # old silent absence, where a top-level `train_metrics:` key simply never landed.
    assert trainer.train_metrics is None
    with pytest.raises(ValueError, match="no train_metrics slot"):
        trainer.train_metrics = {"map": None}


def test_metric_slots_default_to_sentinels_resolved_at_initialize() -> None:
    """`None` means "not chosen", never "no metric" — the COCO-mAP default arrives in
    `initialize()`, built per split and prefixed, so an explicitly wired metric is never
    silently replaced."""
    trainer = LightningDetector(
        model=_NativeDetector(),
        train_set=_make_stream(4),
        num_classes=2,
    )
    assert trainer.metrics is None and trainer.val_metrics is None and trainer.test_metrics is None
    trainer.initialize()
    assert isinstance(trainer.val_metrics, MetricCollection)
    assert isinstance(trainer.test_metrics, MetricCollection)
    assert trainer.val_metrics is not trainer.test_metrics
    assert any(str(k).startswith("val/") for k in trainer.val_metrics.keys())
    assert any(isinstance(m, MeanAveragePrecision) for m in trainer.val_metrics.values())


def test_a_split_slot_adds_to_the_shared_set_for_that_split_only() -> None:
    """`metrics:` is what every eval split measures; `val_metrics:` is what validation measures on
    top — merged by name through `matrainer.metric_slots.merge_metric_specs`, never replaced."""
    trainer = LightningDetector(
        model=_NativeDetector(),
        train_set=_make_stream(4),
        num_classes=2,
        val_metrics={"confusion": DetectionConfusionMatrix(iou_threshold=0.5)},
    )
    trainer.initialize()
    val_keys = set(cast(MetricCollection, trainer.val_metrics).keys())
    test_keys = set(cast(MetricCollection, trainer.test_metrics).keys())
    assert "val/confusion" in val_keys, val_keys
    assert any(str(k).startswith("val/map") for k in val_keys), "the shared default must survive the merge"
    assert "test/confusion" not in test_keys, "a val-only metric must not widen to the test split"


def test_a_second_initialize_does_not_reprefix(caplog: Any) -> None:
    """`run()` calls `initialize()` every time; `_metrics_built` is what stops a second pass
    re-prefixing `val/` into `val/val/`."""
    trainer = LightningDetector(model=_NativeDetector(), train_set=_make_stream(4), num_classes=2)
    trainer.initialize()
    first = list(cast(MetricCollection, trainer.val_metrics).keys())
    trainer.initialize()
    assert list(cast(MetricCollection, trainer.val_metrics).keys()) == first


def test_there_is_no_logger_knob_and_trackers_is_a_ctor_param() -> None:
    """The `logger:` slot is DELETED — `trackers:` is the whole of a run's output wiring, and it
    MUST be a constructor param: confluid gates LIST broadcast on the receiving slot's
    annotation, so a post-construction attribute would make `trackers: [...]` in YAML silently
    do nothing."""
    import inspect

    params = inspect.signature(LightningDetector.__init__).parameters
    assert "logger" not in params
    assert "trackers" in params and "metric_names" in params
    # Stored VERBATIM (lazy-init): the ctor must not iterate or normalize the value.
    marker = object()
    trainer = LightningDetector(model=_NativeDetector(), trackers=cast(Any, marker))
    assert trainer.trackers is marker


def test_reportable_metrics_filters_the_classes_bookkeeping_key() -> None:
    """MAP's `classes` entry is the OBSERVED class ids, not a metric — left in, it is logged as a
    bogus scalar (one class seen) or rendered as a bogus per-class table (several seen)."""
    values = {"val/map": torch.tensor(0.5), "val/classes": torch.tensor([1, 2]), "val/map_50": torch.tensor(0.6)}
    from raidar.torch_detection import reportable_metrics

    assert set(reportable_metrics(values)) == {"val/map", "val/map_50"}


def test_user_supplied_instance_metric_passes_through() -> None:
    """An already-instantiated metric is held verbatim — no Partial wrapping needed."""
    custom = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    trainer = LightningDetector(model=_NativeDetector(), train_set=_make_stream(4), num_classes=2)
    trainer.val_metrics = custom
    assert trainer.val_metrics is custom


def test_metric_collection_with_native_confusion_matrix_construction() -> None:
    """A MetricCollection of native detection metrics (MAP + confusion matrix)
    constructs cleanly — proving the specialised trainer accepts compositions
    without adapter wrapping."""
    coll = MetricCollection(
        {
            "map": MeanAveragePrecision(iou_type="bbox"),
            "confusion": DetectionConfusionMatrix(iou_threshold=0.5),
        }
    )
    trainer = LightningDetector(model=_NativeDetector(), train_set=_make_stream(4), num_classes=2)
    trainer.val_metrics = coll
    assert trainer.val_metrics is coll


def test_default_loader_DECLARES_the_list_collate() -> None:

    from confluid.fluid import Target as ConfluidClass

    trainer = LightningDetector(
        model=_NativeDetector(),
        train_set=_make_stream(4),
        num_classes=2,
    )
    for loader in (trainer.train_loader, trainer.val_loader):
        assert isinstance(loader, ConfluidClass)
        collate = loader.kwargs.get("collate_fn")
        # DECLARED, not inherited: the native forward takes List[Tensor], so the loaders ask
        # recordstream for the unstacked batch rather than relying on ToTensor's plain
        # tensors happening to skip stacking.
        assert collate is collate_list


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------


def test_training_step_sums_native_loss_dict_with_grad() -> None:
    trainer = LightningDetector(
        model=_NativeDetector(),
        train_set=_make_stream(4),
        num_classes=2,
    )
    trainer.train()
    batch = _make_batch(n=2)
    loss = trainer.training_step(batch, 0)
    assert isinstance(loss, Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_training_step_rejects_missing_targets() -> None:
    trainer = LightningDetector(
        model=_NativeDetector(),
        train_set=_make_stream(4),
        num_classes=2,
    )
    trainer.train()
    batch: Record = {
        "image": [torch.rand(3, 32, 32)],
        "targets": [],
        "metadata": [{}],
    }
    with pytest.raises(ValueError, match="requires targets"):
        trainer.training_step(batch, 0)


def test_training_step_rejects_non_dict_model_output() -> None:
    """If the model returns a list (eval-shape) in train mode, fail loudly."""

    class _BrokenDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.alpha = nn.Parameter(torch.tensor(1.0))

        def forward(
            self,
            images: List[Tensor],
            targets: Optional[List[Dict[str, Tensor]]] = None,
        ) -> Any:
            return [{"boxes": torch.zeros((0, 4))} for _ in images]

    trainer = LightningDetector(
        model=_BrokenDetector(),
        train_set=_make_stream(4),
        num_classes=2,
    )
    trainer.train()
    batch = _make_batch(n=2)
    with pytest.raises(TypeError, match="must return Dict"):
        trainer.training_step(batch, 0)


def test_validation_step_updates_metric_with_predictions() -> None:
    trainer = LightningDetector(model=_NativeDetector(), train_set=_make_stream(4), num_classes=2)
    trainer.val_set = _make_stream(2)
    trainer.val_metrics = MeanAveragePrecision(iou_type="bbox")
    trainer.eval()
    batch = _make_batch(n=2)
    # validation_step doesn't return a loss in the native path; just executes.
    trainer.validation_step(batch, 0)
    # After update, the metric holds state — compute() returns a dict with map keys.
    assert isinstance(trainer.val_metrics, MeanAveragePrecision)


def test_test_step_uses_test_metric() -> None:
    trainer = LightningDetector(model=_NativeDetector(), train_set=_make_stream(4), num_classes=2)
    trainer.test_set = _make_stream(2)
    trainer.test_metrics = MeanAveragePrecision(iou_type="bbox")
    trainer.eval()
    batch = _make_batch(n=2)
    trainer.test_step(batch, 0)
    assert isinstance(trainer.test_metrics, MeanAveragePrecision)


# ---------------------------------------------------------------------------
# End-to-end Lightning fit() against an in-memory Stream — no raidar adapters
# ---------------------------------------------------------------------------


def test_end_to_end_native_trainer_fit(tmp_path: Any) -> None:
    """Specialised trainer runs through fit() with no raidar loss/metric adapter:
    pass raw native detector + raw torchmetrics MAP, observe train/loss and
    val/map.
    """
    train_stream = _make_stream(4)
    val_stream = _make_stream(2)

    trainer = LightningDetector(model=_NativeDetector(), train_set=train_stream, num_classes=2)
    trainer.val_set = val_stream

    lightning_trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        accelerator="cpu",
    )
    # run() resolves the metric sentinels + builds the per-split collections in initialize().
    # Mirror that here so the metrics materialise before fit().
    trainer.initialize()
    # `cast`: a Stream is map-style (`__len__`/`__getitem__`) and DataLoader duck-types its
    # argument at runtime, but torch's STUB declares `Dataset[T]`. Stream deliberately no
    # longer inherits that base — it made torch a hard dependency of a numpy engine — so the
    # stub's stricter view is bridged here rather than by re-coupling the engine.
    trainer.train_loader = torch.utils.data.DataLoader(cast(Any, train_stream), batch_size=2, collate_fn=collate_list)
    trainer.val_loader = torch.utils.data.DataLoader(cast(Any, val_stream), batch_size=2, collate_fn=collate_list)

    lightning_trainer.fit(
        model=trainer,
        train_dataloaders=trainer.train_loader,
        val_dataloaders=trainer.val_loader,
    )
    metrics = lightning_trainer.callback_metrics
    assert any(k.startswith("train/loss") for k in metrics), f"expected train/loss in {list(metrics)}"
    train_loss = next(v for k, v in metrics.items() if k.startswith("train/loss"))
    assert torch.isfinite(train_loss)
    # MAP scalars should appear under val/ prefix.
    assert any(k.startswith("val/map") for k in metrics), f"expected val/map* in {list(metrics)}"


# ---------------------------------------------------------------------------
# initialize(): num_classes resolution + lazy-model flow (sonair-style)
# ---------------------------------------------------------------------------


class _NumClassesDetector(nn.Module):
    """Stub whose ctor records ``num_classes`` — stands in for a ``!lazy:`` detector
    builder (e.g. torchvision's ``fasterrcnn_resnet50_fpn``), so ``initialize()``'s
    flow-injection of the resolved count is testable without a heavy download."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = nn.Parameter(torch.tensor(1.0))


def _labelled_stream(label_sets: List[List[int]], size: int = 16) -> Stream:
    """A detection Stream whose per-image targets carry the given label lists (under "target")."""
    records = [
        {
            "image": Image(torch.rand(3, size, size).numpy(), layout="CHW"),
            "target": _regions(
                torch.zeros((len(lbls), 4), dtype=torch.float32),
                torch.tensor(lbls, dtype=torch.int64),
            ),
        }
        for lbls in label_sets
    ]
    return Stream(source=records)


def test_initialize_derives_num_classes_from_targets_and_flows_lazy_model() -> None:
    # Labels up to 2 → derived num_classes = max + 1 = 3; injected into the lazy model.
    trainer = LightningDetector(model=PartialClass(_NumClassesDetector), train_set=_labelled_stream([[1, 2], [1]]))
    trainer.initialize()
    assert trainer.num_classes == 3
    assert isinstance(trainer.model, _NumClassesDetector)
    assert trainer.model.num_classes == 3


def test_initialize_explicit_num_classes_injected_into_lazy_model() -> None:
    trainer = LightningDetector(
        model=PartialClass(_NumClassesDetector), train_set=_labelled_stream([[1]]), num_classes=7
    )
    trainer.initialize()
    assert isinstance(trainer.model, _NumClassesDetector)
    assert trainer.num_classes == 7 and trainer.model.num_classes == 7


def test_initialize_lazy_model_pinned_num_classes_used() -> None:
    # num_classes pinned on the lazy model's kwargs is used as-is.
    trainer = LightningDetector(
        model=PartialClass(_NumClassesDetector, num_classes=9), train_set=_labelled_stream([[1]])
    )
    trainer.initialize()
    assert isinstance(trainer.model, _NumClassesDetector)
    assert trainer.num_classes == 9 and trainer.model.num_classes == 9


def test_initialize_concrete_model_num_classes_is_authoritative() -> None:
    # A concrete model's head wins over a (mismatched) explicit trainer arg.
    model = _NumClassesDetector(num_classes=4)
    trainer = LightningDetector(model=model, train_set=_labelled_stream([[1]]), num_classes=99)
    trainer.initialize()
    assert trainer.num_classes == 4
    assert trainer.model is model


def test_initialize_model_without_num_classes_param_flowed_plainly() -> None:
    # A lazy model whose builder takes no num_classes is flowed plainly (no kwarg error).
    trainer = LightningDetector(model=PartialClass(_NativeDetector), train_set=_labelled_stream([[1]]), num_classes=3)
    trainer.initialize()
    assert isinstance(trainer.model, _NativeDetector)
    assert trainer.num_classes == 3


def test_trained_model_output_raises_before_run_returns_after_flow() -> None:
    trainer = LightningDetector(model=PartialClass(_NumClassesDetector), train_set=_labelled_stream([[1]]))
    with pytest.raises(RuntimeError, match="only available after run"):
        _ = trainer.trained_model
    trainer.initialize()  # flows the model into a live instance
    assert isinstance(trainer.trained_model, _NumClassesDetector)


def test_model_num_classes_reads_attr_and_torchvision_head() -> None:
    # Explicit num_classes attribute (raidar/ultralytics wrappers).
    assert _model_num_classes(_NumClassesDetector(num_classes=6)) == 6
    # torchvision two-stage box-predictor head shape (Faster R-CNN).
    m = nn.Module()
    m.roi_heads = nn.Module()
    m.roi_heads.box_predictor = nn.Module()
    m.roi_heads.box_predictor.cls_score = nn.Linear(10, 5)
    assert _model_num_classes(m) == 5
    # Neither present → None.
    assert _model_num_classes(nn.Linear(3, 3)) is None


# ---------------------------------------------------------------------------
# run() dispatch — the @entrypoint markers ARE the table (recordstream run_entrypoint)
# ---------------------------------------------------------------------------


def test_entrypoint_markers_cover_every_task_literal_value() -> None:
    """`run()` dispatches through the markers, so a task the Literal offers must be declared."""
    declared = {str(meta["task"]) for meta in runnable_entrypoints(LightningDetector).values()}
    assert declared == set(get_args(DetectorTask))


@pytest.mark.parametrize("task", get_args(DetectorTask))
def test_run_dispatches_to_the_entry_point_declaring_the_task(task: str) -> None:
    calls: List[str] = []
    # A non-empty stand-in: this test drives DISPATCH only, but run() guards the
    # Mandatory model slot first (matrainer TrainerRunnable._require_model).
    trainer = LightningDetector(model=cast(Any, object()), task=cast(Any, task))
    trainer.initialize = lambda: None  # type: ignore[method-assign]
    for name in get_args(DetectorTask):  # instance attrs win over the class's bound methods
        setattr(trainer, name, lambda n=name: calls.append(n))
    trainer.run()
    assert calls == [task]


def test_persistent_workers_can_be_named_instead_of_derived() -> None:
    """The HOST-scoped knob: a config says `persistent_workers: false` and keeps its workers.

    macOS terminates persistent workers slowly enough that a short run spends longer stopping
    than training, which no derivation from `num_workers` can see."""
    runnable = LightningDetector(model=_NativeDetector(), num_classes=2, num_workers=2, persistent_workers=False)
    for split in ("train_loader", "val_loader", "test_loader"):
        marker = getattr(runnable, split)
        assert marker.kwargs["persistent_workers"] is False, split
        assert marker.kwargs["num_workers"] == 2, split
