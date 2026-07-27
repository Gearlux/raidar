"""Every example config must LOAD + FLOW to a runnable — the CI safety net.

CI runs ``python examples/*.py`` (an aisland-scaffolded glob), but raidar ships
its examples as ``.yaml`` configs, which the glob never executes. Without this
test a stale example (a renamed class, a bad ``!class:`` path) would fail only
at run time and sail through CI green.

This test mirrors what the runners do short of training: it loads each
``examples/*.yaml``, resolves + flows the object bound under its root key —
``runnable:`` (the merged ``LightningDetector`` for ``recordstream run`` /
``raidar train|evaluate``) or ``trainer:`` / ``evaluator:`` (the framework-native
``RFDETRTrainer`` / ``RFDETREvaluator`` for ``raidar-rfdetr``) — importing its
``!class:``/``!lazy:`` targets and constructing it (cheap: every configurable is
zero-arg / lazy, so no dataset download or training runs), and asserts a callable
``run()``. A dangling class path raises ``confluid.UnknownClassError`` at flow
time, failing the test.
"""

from pathlib import Path

import confluid
import pytest
from confluid import flow
from confluid.fluid import Fluid

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_CONFIGS = sorted(EXAMPLES_DIR.glob("*.yaml"))
# Root keys a raidar example may bind its runnable under, in resolution order.
ROOT_KEYS = ("runnable", "trainer", "evaluator", "processor")
VALID_TASKS = {"fit", "evaluate", "test", "predict"}


def test_examples_dir_is_not_empty() -> None:
    """Guard: if the examples move/rename, this test must not silently cover nothing."""
    assert EXAMPLE_CONFIGS, f"no example configs found under {EXAMPLES_DIR}"


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_config_loads_and_flows_to_a_runnable(config_path: Path) -> None:
    """The config binds a runnable whose ``!class:`` resolves + constructs into a ``run()``-able object."""
    document = confluid.load(config_path.read_text(), flow=True)

    root_key = next(
        (k for k in ROOT_KEYS if (isinstance(document, dict) and k in document) or hasattr(document, k)),
        None,
    )
    assert root_key is not None, f"{config_path.name}: no runnable root key (one of {ROOT_KEYS})"

    runnable = document[root_key] if isinstance(document, dict) else getattr(document, root_key)
    if isinstance(runnable, Fluid):
        runnable = flow(runnable)  # imports the !class:/!lazy: target — dangling paths raise here

    assert callable(getattr(runnable, "run", None)), f"{config_path.name}: runnable has no run() method"
    # The merged LightningDetector (runnable: root) dispatches on a `task`; assert it is valid.
    if root_key == "runnable":
        task = getattr(runnable, "task", None)
        assert task in VALID_TASKS, f"{config_path.name}: invalid task {task!r} (expected one of {VALID_TASKS})"
