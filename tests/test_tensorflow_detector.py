# mypy: disable-error-code="attr-defined"
"""``TensorFlowDetector`` — the native ``tf.data`` + ``GradientTape`` detection backend.

Two groups, split by what they need (the deltaid TF-suite pattern):

* **Source-only** — the import ordering and the sibling contract, read as TEXT. They run
  everywhere, because the ordering they guard decides which engine keras binds and the symptom
  of getting it wrong (``tf.GradientTape`` failing on a torch-backed model) is far from its cause.
* **Behavioural** — everything that imports the module, skipped unless keras is on the
  TensorFlow engine in THIS process. **Keras 3 binds exactly one engine per process, at first
  import** — a full-suite run loads keras on torch first (alphabetical test order), so this loop
  genuinely cannot run there. Run it as its own invocation::

    pytest tests/test_tensorflow_detector.py

  Running this file alone needs no environment variable: it imports ``raidar._tensorflow``
  before anything else can claim the engine.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

SOURCE = Path(__file__).resolve().parent.parent / "raidar" / "tensorflow_detector.py"

# Claim the engine before any other import can — what lets a solo run need no env var.
if importlib.util.find_spec("tensorflow") is not None and "keras" not in sys.modules:
    import raidar._tensorflow  # noqa: F401


def _keras_is_on_tensorflow() -> bool:
    """Whether keras is bound to the TensorFlow engine in this process (never importing it)."""
    keras = sys.modules.get("keras")
    return keras is not None and keras.backend.backend() == "tensorflow"


requires_tensorflow = pytest.mark.skipif(
    not _keras_is_on_tensorflow(),
    reason="needs keras on the TensorFlow engine: run `pytest tests/test_tensorflow_detector.py` alone",
)


# --------------------------------------------------------------------------- #
# Source-only — runs without TensorFlow too
# --------------------------------------------------------------------------- #
def test_keras_is_reached_through_the_module_that_pins_the_engine() -> None:
    """``from raidar._tensorflow import …``, NEVER ``from recordstream.keras import …`` —
    recordstream defaults KERAS_BACKEND to the first engine installed (torch on this workspace),
    and isort would sort its line ABOVE every raidar line, beating the setdefault into place."""
    source = SOURCE.read_text()
    assert "from raidar._tensorflow import" in source
    tree = ast.parse(source)
    roots = {(node.module or "") for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    roots |= {a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    assert "recordstream.keras" not in roots
    assert "keras" not in roots and "tensorflow" not in roots


def test_the_environment_default_precedes_every_import_that_reads_it() -> None:
    text = (SOURCE.parent / "_tensorflow.py").read_text()
    setdefault = text.index('os.environ.setdefault("KERAS_BACKEND"')
    assert setdefault < text.index("import tensorflow as tf")
    assert setdefault < text.index("from recordstream.keras import")


def test_it_is_a_sibling_of_the_keras_backend_not_a_subclass() -> None:
    """It NARROWS a precondition (``_require_tensorflow_backend``) its would-be parent lacks."""
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "TensorFlowDetector")
    assert "KerasDetector" not in [ast.unparse(b) for b in cls.bases]
    assert "TensorFlowRunnable" in SOURCE.read_text()


@requires_tensorflow
def test_the_narrowed_precondition_lives_on_the_machinery_base() -> None:
    """The other half of the sibling claim — the engine guard `KerasDetector` does not have.

    Split from the source-only test above because reading the machinery's body needs the module
    IMPORTED, and it imports TensorFlow at import time.
    """
    import inspect

    from matrainer.tensorflow.loop import TensorFlowRunnable

    assert "_require_tensorflow_backend" in inspect.getsource(TensorFlowRunnable.run)


def test_the_raw_tf_model_reaches_no_keras_initializer() -> None:
    """`tf.initializers` IS Keras since TF 2.16 — bound to torch it returns torch tensors, and a
    raw-`tf.nn` module reaching for it dies converting them (the deltaid measurement)."""
    source = (SOURCE.parent / "fcos" / "tensorflow_fcos.py").read_text()
    # AST rather than substrings — the module's own docstring NAMES the banned reach-throughs.
    accesses = {ast.unparse(node) for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Attribute)}
    for banned in ("tf.initializers", "tf.keras", "tf.losses", "tf.metrics"):
        assert not any(access.startswith(banned) for access in accesses), f"{banned} is Keras wearing a tf name"
    assert "stateless_uniform" in source


# --------------------------------------------------------------------------- #
# Behavioural — needs keras on the TensorFlow engine in this process
# --------------------------------------------------------------------------- #
def _make_stream(n: int = 4, size: int = 64) -> Any:
    import numpy as np
    import torch
    from recordstream import Boxes, Image
    from recordstream.core import Stream

    from raidar.fcos.targets import FcosTargets

    records: List[Dict[str, Any]] = [
        {
            "image": Image(np.random.default_rng(i).random((3, size, size)).astype("float32"), layout="CHW"),
            "target": Boxes(
                boxes=torch.tensor([[8.0, 8.0, 40.0, 40.0]], dtype=torch.float32),
                labels=torch.tensor([1], dtype=torch.int64),
            ),
            "idx": i,
        }
        for i in range(n)
    ]
    return Stream(source=records, ops=[FcosTargets(num_classes=3, input_shape=(size, size, 3))])


@requires_tensorflow
def test_fit_end_to_end_with_the_raw_tf_model(tmp_path: Any) -> None:
    import json

    from raidar.fcos.tensorflow_fcos import TensorFlowFcos
    from raidar.tensorflow_detector import TensorFlowDetector

    sidecar = tmp_path / "metrics.json"
    detector = TensorFlowDetector(
        model=TensorFlowFcos(num_classes=3, input_shape=(64, 64, 3)),
        train_set=_make_stream(6),
        num_classes=3,
        task="fit",
        max_epochs=2,
        batch_size=2,
        export_dir=str(tmp_path / "saved_model"),
    )
    detector.val_set = _make_stream(2)
    detector.metrics_sidecar_path = str(sidecar)
    detector.run()

    metrics = detector.computed_metrics
    assert "train/loss" in metrics and "val/map_50" in metrics, metrics
    assert json.loads(sidecar.read_text())["metrics"]
    assert (tmp_path / "saved_model").exists(), "the SavedModel export is this backend's artifact"


@requires_tensorflow
def test_test_scores_through_the_numpy_path() -> None:
    from raidar.fcos.tensorflow_fcos import TensorFlowFcos
    from raidar.tensorflow_detector import TensorFlowDetector

    detector = TensorFlowDetector(model=TensorFlowFcos(num_classes=3, input_shape=(64, 64, 3)), task="test")
    detector.test_set = _make_stream(4)
    detector.run()
    assert "test/map_50" in detector.computed_metrics


@requires_tensorflow
def test_a_wrong_engine_raises_naming_the_variable() -> None:
    # The guard reads the LIVE backend; here it is tensorflow, so patch the probe instead.
    import raidar.tensorflow_detector as mod
    from raidar.tensorflow_detector import TensorFlowDetector

    original = mod.keras_backend
    mod.keras_backend = lambda: "torch"  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="KERAS_BACKEND"):
            TensorFlowDetector._require_tensorflow_backend()
    finally:
        mod.keras_backend = original  # type: ignore[assignment]
