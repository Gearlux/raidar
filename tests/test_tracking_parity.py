"""Both detection backends hand matrainer the RUNNABLE, and neither derives its own record.

AST-based (not import-based) for the fleet-test reason established in the classification and
segmentation projects: an import-based check silently reduces to whichever frameworks the local
venv carries — exactly the machine where a copy-pasted omission lands.
"""

import ast
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent / "raidar"

#: Every framework runnable module in this project, and how it hooks tracking up:
#: `lightning_detector` through `TrackingCallback` (the Trainer owns the loop), `rfdetr_detector`
#: through the plain functions (the upstream loop owns everything, so provenance wraps it).
BACKENDS = {
    "lightning_detector.py": "TrackingCallback",
    "torch_detector.py": "log_run_context",
    "fastai_detector.py": "log_run_context",
    "huggingface_detector.py": "log_run_context",
    "rfdetr_detector.py": "log_run_context",
    # The Keras-family loops' hookup is matrainer's callback adapter.
    "keras_detector.py": "keras_tracking_callback",
    "tensorflow_detector.py": "keras_tracking_callback",
    # The JAX and MLX loops own everything, so provenance is the plain-function hookup.
    "jax_detector.py": "log_run_context",
    "mlx_detector.py": "log_run_context",
}


def _source(name: str) -> str:
    return (SOURCE_DIR / name).read_text()


def test_the_checked_list_is_every_backend_on_disk() -> None:
    """The guard that stops a third backend skipping this file: every `*_detector.py` is listed."""
    on_disk = {p.name for p in SOURCE_DIR.glob("*_detector.py")}
    assert on_disk == set(BACKENDS), f"backends on disk {on_disk} != checked {set(BACKENDS)}"


def test_every_backend_hands_matrainer_the_runnable() -> None:
    """`runnable=self` — matrainer owns the ONE hyperparameter derivation, so no two backends can
    answer "what were the hyperparameters?" differently."""
    for name, entry in BACKENDS.items():
        source = _source(name)
        assert entry in source, f"{name}: expected the {entry} hookup"
        assert "runnable=self" in source, f"{name}: tracking must pass runnable=self"


def test_no_backend_keeps_a_private_derivation() -> None:
    """The retired local helpers must not return: params dicts and hand-rolled sidecars each made
    a second answer to a question matrainer owns."""
    banned = {"_config_params", "_tracked_params", "_finalize_metrics", "_metrics_to_payload"}
    for name in BACKENDS:
        tree = ast.parse(_source(name))
        defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert not (defined & banned), f"{name}: private derivation {defined & banned} has returned"


def test_the_rfdetr_backend_finishes_its_manifest() -> None:
    """The plain-function hookup must close the loop: an unfinished manifest stays `running` and
    matrainer's atexit hook then labels a successful run `interrupted`."""
    source = _source("rfdetr_detector.py")
    assert "capture_log_file" in source and 'status="success"' in source


#: How each backend gets its METRICS onto the wired trackers — a DIFFERENT question from the
#: provenance hookup above, and the distinction is what let a real bug through: raidar's
#: `rfdetr_detector` had `log_run_context` (so the provenance table passed) while never calling
#: `tracker.log_metrics` at all, so a tracked run recorded its git state and its log file and NOT
#: its mAP. Its evaluate path had no tracker wiring whatsoever.
#:
#: The sibling restoration project found the same shape on its `transformers.Trainer` backend, and
#: only because a backend SWEEP put eleven runs in one table and one came out blank. A backend read
#: ALONE looks perfect — it prints its score and writes its metrics sidecar.
METRIC_BRIDGES = {
    "fastai_detector.py": "log_metrics",
    "huggingface_detector.py": "_log_trainer_history",
    "jax_detector.py": "log_metrics",
    "keras_detector.py": "keras_tracking_callback",
    "lightning_detector.py": "TrackerLogger",
    "mlx_detector.py": "log_metrics",
    "rfdetr_detector.py": "log_metrics",
    "tensorflow_detector.py": "keras_tracking_callback",
    "torch_detector.py": "log_metrics",
}


def test_every_backend_puts_its_METRICS_on_the_trackers() -> None:
    """Provenance is not enough: a run whose numbers reach only the log is invisible to every
    comparison, every tracker UI and every sweep."""
    for name, marker in METRIC_BRIDGES.items():
        source = (SOURCE_DIR / name).read_text()
        assert marker in source, f"{name} never reaches a tracker's metric channel via {marker}"


def test_the_metric_bridge_table_covers_every_backend_on_disk() -> None:
    """The converse — a new backend must not be added without a metric channel."""
    on_disk = {p.name for p in SOURCE_DIR.glob("*_detector.py")}
    assert on_disk == set(METRIC_BRIDGES)
