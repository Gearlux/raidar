"""The backend sweep's CONFIG surface — the project's first (2026-08-07).

Config-level rather than run-level, for the sibling projects' reason: no venv carries all nine
frameworks, so the checks load and RESOLVE the overlay instead of training it — and they fail
on the machine that introduces a mistake rather than only where an extra happens to be
installed. The sweep MACHINERY is matrainer's (`matrainer.sweep`, pinned in its own suite);
what belongs here is this project's half — the spec's matrix and the overlay's tracker
dimension.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import confluid
import pytest

from raidar.cli import FRAMEWORKS

SWEEP = Path(__file__).resolve().parent.parent / "examples" / "sweep" / "all_backends.yaml"

#: tracker name -> the matrainer class its `!scope:tracker=` block must wire.
TRACKERS = {
    "mlflow": "MlflowTracker",
    "tensorboard": "TensorBoardTracker",
    "csv": "CsvTracker",
    "aim": "AimTracker",
    "clearml": "ClearMLTracker",
    "wandb": "WandbTracker",
}


def resolve(framework: str, tracker: str = "mlflow") -> Dict[str, Any]:
    document = confluid.load(
        confluid.load_config(SWEEP), flow=False, scopes=[f"framework={framework}", f"tracker={tracker}"]
    )
    assert isinstance(document, dict)
    return document


def target_name(value: Any) -> Optional[str]:
    """The class name behind a `!class:` / `!lazy:` marker, without importing it."""
    target = getattr(value, "target", None)
    if target is None:
        return None
    return str(target).rsplit(".", 1)[-1]


def test_the_sweep_spec_covers_every_backend_the_cli_advertises() -> None:
    """A backend added to `FRAMEWORKS` must join the sweep in the same change — except the
    AGPL sibling, whose install is deliberately absent on a default checkout (and whose
    trainer is train-only besides)."""
    from matrainer.sweep import load_spec

    spec = load_spec(SWEEP.parent / "sweep.yaml")
    assert {row.name for row in spec.frameworks} == set(FRAMEWORKS) - {"ultralytics"}
    assert not spec.models, "the example wires FCOS per framework block — there is no model dimension to flag"


@pytest.mark.parametrize("tracker", sorted(TRACKERS))
def test_every_tracker_block_wires_its_backend_with_the_run_identity(tracker: str) -> None:
    """Each `!scope:tracker=` block must carry the `!ref:experiment_name` / `!ref:run_name`
    pair — a `!class:` in a LIST stays a deferred marker that `_live_trackers()` flows WITHOUT
    the document, so the refs are the ONLY way the sweep driver's per-cell identity reaches
    the tracker."""
    document = resolve("torch", tracker=tracker)
    wired = document.get("trackers")
    assert isinstance(wired, list) and len(wired) == 1
    assert target_name(wired[0]) == TRACKERS[tracker]
    kwargs = getattr(wired[0], "kwargs", {})
    assert "experiment_name" in kwargs and "run_name" in kwargs


@pytest.mark.parametrize("framework", sorted(set(FRAMEWORKS) - {"ultralytics"}))
def test_the_overlay_resolves_for_every_swept_framework(framework: str) -> None:
    """`include:` deep-merges the example, so every framework block the example declares must
    still activate under the overlay — a scope block REPLACES the key it lands on, and a
    malformed overlay block would silently discard a backend's runnable."""
    document = resolve(framework, tracker="csv")
    assert document.get("runnable") is not None, f"{framework}: the overlay lost the runnable"
