"""``DetectionBase`` — the task half of every raidar backend, written once.

The detection logic that used to exist per backend module: ``num_classes`` resolution
(concrete model head → explicit → lazy kwarg → the announced target walk over the ``Boxes``
column), the injected-only-when-accepted model flow (torchvision ships its detectors as
builder FUNCTIONS with differing signatures), the fork-safe split preparation, and the
fit/eval initialization split — including detection's own asymmetry, the missing train
metrics split (a native model in train mode returns only a loss dict).

It subclasses the trainer API base (`matrainer.runnable.TrainerRunnable`) and is deliberately
**torch-free** so the four spec engines (keras/tensorflow/jax/mlx) can import it. The
torch-family hook IMPLEMENTATIONS live in :mod:`raidar.detection_torch` — NOT to be confused
with :mod:`raidar.torch_detection`, which is the torch ENGINE's registered metric set + the
``native_batch`` contract adapter; that one registers/adapts, this one carries task logic.
"""

import inspect
from typing import TYPE_CHECKING, Any, List, Optional

from confluid import Fluid, flow
from confluid.registry import resolve_class
from loggair import get_logger
from matrainer.runnable import TrainerRunnable
from recordstream import Boxes, iter_key
from tqdm import tqdm

if TYPE_CHECKING:  # annotation-only — this module must not import torch
    import torch.nn as nn

logger = get_logger(__name__)

__all__ = ["DetectionBase", "_model_num_classes"]


def _model_num_classes(model: "nn.Module") -> Optional[int]:
    """Best-effort read of a BUILT detector's class count (explicit attr or torchvision head), or ``None``."""
    nc = getattr(model, "num_classes", None)
    if nc:
        return int(nc)
    box_predictor = getattr(getattr(model, "roi_heads", None), "box_predictor", None)
    out_features = getattr(getattr(box_predictor, "cls_score", None), "out_features", None)
    if out_features:
        return int(out_features)
    return None


class DetectionBase(TrainerRunnable):
    """The detection task base: ``num_classes`` resolution + the fit/eval initialization.

    The attributes below are STORED by the concrete backend's own ``__init__`` (they are its
    config surface); this base only declares their names/types for the methods that read them.
    There is deliberately NO ``train_metrics`` attribute in any detection config surface — a
    torchvision-native model in train mode returns ONLY the loss dict, so a slot would accept
    a wired set and silently never call it.
    """

    log_label = "raidar"

    @property
    def train_metrics(self) -> None:
        """Always ``None`` — detection has NO train-split metrics, structurally.

        A torchvision-native model in train mode returns ONLY the loss dict, so there are no
        train-split predictions to score. The shared trainer base carries the slot for the
        tasks that have one; this property pins it to ``None`` here and makes WIRING one a
        loud error instead of a set that would silently never be called.
        """
        return None

    @train_metrics.setter
    def train_metrics(self, value: Any) -> None:
        if value is not None:
            raise ValueError(
                "detection has deliberately no train_metrics slot: a torchvision-native model "
                "in train mode returns only its loss dict, so a wired train-split metric set "
                "could never be called. Use val_metrics / test_metrics."
            )

    num_classes: Optional[int]
    #: Labels for the per-class metric report, in class-id order (used when a
    #: ``MeanAveragePrecision(class_metrics=True)`` emits per-class vectors). A detection
    #: target carries ids, not names, so there is no label map to derive them from.
    class_names: Optional[List[str]]
    plot_metrics: bool

    def initialize(self) -> None:
        """Resolve ``num_classes``, flow the model, and build the per-split metric collections."""
        if self.task == "fit":
            self._initialize_fit()
        else:
            self._initialize_eval()

    def _initialize_fit(self) -> None:
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        nc = self._resolve_num_classes()
        if nc is not None:
            self.num_classes = int(nc)
        if isinstance(self.model, Fluid):
            # Injected ONLY when the deferred builder accepts it — torchvision ships its
            # detectors as builder FUNCTIONS with differing signatures, so a builder without
            # the param is flowed plainly rather than handed a kwarg it would reject.
            if self.num_classes is not None and self._model_accepts_num_classes():
                self.model = flow(self.model, num_classes=self.num_classes)
            else:
                self.model = flow(self.model)
        self._resolve_task_defaults()
        self._build_fit_metrics(include_train=False)

    def _initialize_eval(self) -> None:
        # The trained model's head is authoritative for eval — flow it first, read num_classes
        # off it. Best-effort: the default MAP metric is class-count-agnostic, so an
        # undeterminable count leaves `num_classes` as-is instead of raising.
        self.model = flow(self.model)
        nc = _model_num_classes(self.model)
        if nc:
            self.num_classes = int(nc)
        self._resolve_task_defaults()
        eval_slot = "val_set" if self.task == "evaluate" else "test_set"
        setattr(self, eval_slot, self._prepare(getattr(self, eval_slot)))
        self._build_eval_metrics()

    def _model_accepts_num_classes(self) -> bool:
        if not isinstance(self.model, Fluid):
            return False
        target = self.model.target
        resolved = resolve_class(target) if isinstance(target, str) else target
        if resolved is None or not callable(resolved):
            return False
        try:
            params = inspect.signature(resolved).parameters
        except (ValueError, TypeError):
            return False
        return "num_classes" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def _resolve_num_classes(self) -> Optional[int]:
        if not isinstance(self.model, Fluid):
            model_nc = _model_num_classes(self.model)
            if model_nc:
                if self.num_classes is not None and int(self.num_classes) != model_nc:
                    logger.warning(f"Trainer num_classes={self.num_classes} != model {model_nc}; using the model's.")
                return model_nc
        if self.num_classes is not None:
            return int(self.num_classes)
        if isinstance(self.model, Fluid):
            pinned = self.model.kwargs.get("num_classes")
            if pinned:
                return int(pinned)
        return self._num_classes_from_targets(self.train_set)

    def _num_classes_from_targets(self, source: Any) -> Optional[int]:
        if source is None:
            return None
        src = flow(source)
        total = len(src) if hasattr(src, "__len__") else None
        logger.info("raidar: deriving num_classes from detection targets (pin num_classes to skip)")
        highest = -1
        for target in tqdm(
            iter_key(src, self.target_key), total=total, desc="raidar: deriving num_classes", unit="record"
        ):
            labels: Any
            if isinstance(target, Boxes):
                labels = target.labels
            elif isinstance(target, dict):
                labels = target.get("labels")
            else:
                continue
            if labels is None:
                continue
            if hasattr(labels, "numel"):
                if int(labels.numel()) == 0:
                    continue
                highest = max(highest, int(labels.max()))
            else:
                for x in labels:
                    highest = max(highest, int(x))
        return highest + 1 if highest >= 0 else None
