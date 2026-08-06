"""The CLI axis: three verbs, a ``--framework`` FLAG, and a config that stays complete under it.

The framework is a property of the CONFIG, not of the CLI, so it is a flag the CLI passes through
rather than a subcommand. The checks here are deliberately CONFIG-level — they load the example
YAML once per declared scope — so a block that drops a slot fails on the machine that introduces
the mistake, rather than only where that backend's framework happens to be installed.
"""

import inspect
from pathlib import Path
from typing import List, Optional

import confluid
import pytest

from raidar.cli import FRAMEWORKS, VERB_TASKS, _dispatch, _normalize_framework_flag, _requested_framework, app

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "train_cppe5.yaml"


# --------------------------------------------------------------------------- #
# The verbs
# --------------------------------------------------------------------------- #
def test_three_verbs_mapping_to_the_tasks_they_force() -> None:
    """The verb wins over the YAML's `task:`, so ONE config serves all three."""
    assert VERB_TASKS == {"train": "fit", "evaluate": "test", "predict": "predict"}


@pytest.mark.parametrize("verb", sorted(VERB_TASKS))
def test_every_verb_is_registered_as_a_command(verb: str) -> None:
    assert verb in app._commands


@pytest.mark.parametrize("verb", sorted(VERB_TASKS))
def test_a_verb_handler_exposes_exactly_the_runnable_parameter(verb: str) -> None:
    """liquifai reads the SIGNATURE to build the command, and binds by parameter NAME.

    A default-argument closure smuggling the loop variable would advertise a `--_verb` option in
    help and TAB completion — which is why the handlers come from a factory.
    """
    params = list(inspect.signature(app._commands[verb]).parameters)
    assert params == ["runnable"], f"{verb} exposes {params}"


def test_the_verbs_build_through_materialize_runnable_not_a_bare_flow() -> None:
    """A bare `flow()` drops EVERY top-level key silently.

    liquifai's DI only materializes a bound node against its document when the command parameter
    is annotated with a *configurable class*; these annotate `runnable: Any` because the runnable
    is polymorphic. So an auto flow would leave `train_set` as None (surfacing much later as a
    failure to derive num_classes) and `max_epochs: 10` as the constructor default (never
    surfacing at all).
    """
    import raidar.cli as cli

    source = inspect.getsource(cli)
    assert "materialize_runnable" in source
    assert 'flow_mode="manual"' in source


# --------------------------------------------------------------------------- #
# Dispatching a runnable that has no `task` knob
# --------------------------------------------------------------------------- #
class _TaskLess:
    """A runnable with a single behaviour and no `task` attribute (the ultralytics sibling's
    current trainer shape)."""

    def __init__(self) -> None:
        self.ran = False

    def run(self) -> None:
        self.ran = True


def test_a_taskless_runnable_still_trains() -> None:
    runnable = _TaskLess()
    _dispatch(runnable, "fit")
    assert runnable.ran


@pytest.mark.parametrize("task", ["test", "predict"])
def test_a_taskless_runnable_raises_on_any_other_verb(task: str) -> None:
    """`raidar evaluate` over a train-only runnable must not silently TRAIN — the class's one
    behaviour under the wrong verb is the failure a silent `hasattr` guard would hide."""
    runnable = _TaskLess()
    with pytest.raises(ValueError, match="task"):
        _dispatch(runnable, task)
    assert not runnable.ran


# --------------------------------------------------------------------------- #
# The --framework flag
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "argv,expected",
    [
        (["raidar", "train", "c.yaml", "--framework", "rfdetr"], "rfdetr"),
        (["raidar", "train", "c.yaml", "--framework=rfdetr"], "rfdetr"),
        (["raidar", "train", "c.yaml", "--scope", "framework=rfdetr"], "rfdetr"),
        (["raidar", "train", "c.yaml", "-s", "framework=rfdetr"], "rfdetr"),
        (["raidar", "train", "c.yaml"], None),
    ],
)
def test_the_requested_framework_is_read_from_argv_in_every_spelling(argv: List[str], expected: Optional[str]) -> None:
    """Read from argv rather than liquifai because the registrations must be imported BEFORE the
    config loads — the eager `!class:Foo()` form is built during load, earlier than any handler."""
    assert _requested_framework(argv) == expected


def test_the_flag_is_rewritten_to_a_scope_never_appended() -> None:
    """liquifai binds the ergonomic `--KEY VAL` flag only for dimensions the config DECLARES.

    Against a config with no `!scope:framework=…` block, a bare `--framework rfdetr` would be
    read as a config OVERRIDE and try to set a `framework` attribute. Rewriting makes the flag
    mean one thing either way, and an undeclared dimension stays the inert no-op that lets a
    config grow into a backend it does not have yet.
    """
    argv = ["raidar", "train", "c.yaml", "--framework", "rfdetr"]
    _normalize_framework_flag(argv)
    assert argv == ["raidar", "train", "c.yaml", "--scope", "framework=rfdetr"]


def test_an_explicit_scope_spelling_is_left_alone() -> None:
    argv = ["raidar", "train", "c.yaml", "--scope", "framework=rfdetr"]
    _normalize_framework_flag(argv)
    assert argv == ["raidar", "train", "c.yaml", "--scope", "framework=rfdetr"]


# --------------------------------------------------------------------------- #
# The config stays complete under every declared framework
# --------------------------------------------------------------------------- #
def test_the_example_config_declares_every_implemented_framework() -> None:
    """A backend absent from the config cannot be selected; one in the config but not in
    FRAMEWORKS is advertised by neither `raidar frameworks` nor the registrations."""
    text = EXAMPLE.read_text()
    for name in FRAMEWORKS:
        assert f"!scope:framework={name}" in text, f"the example config declares no block for {name!r}"


@pytest.mark.parametrize("framework", sorted(FRAMEWORKS))
def test_every_declared_framework_resolves_to_a_complete_runnable(framework: str) -> None:
    """The check that catches a scope block erasing a slot it did not mean to.

    A scope block replaces a KEY and takes its whole subtree with it, so a key nested inside
    `runnable:` would VANISH the moment a block swapped the runnable — a failure that surfaces as
    a missing-argument TypeError minutes into a run. Keeping every framework-specific key a
    top-level SIBLING is what prevents it, and this asserts the result rather than the layout.
    """
    doc = confluid.load(confluid.load_config(EXAMPLE), flow=False, scopes=[f"framework={framework}"])
    for key in ("runnable", "train_set", "val_set", "detection_ops", "num_classes"):
        assert doc.get(key) is not None, f"--framework {framework} leaves {key!r} unset"


def test_the_zero_indexed_backends_restate_the_label_convention() -> None:
    """THE semantic difference between the backends, and silent when wrong.

    torchvision reserves background at index 0 (`label_offset: 1`, `num_classes: 6` on cppe-5);
    RF-DETR and ultralytics are 0-indexed (`label_offset: 0`, `num_classes: 5`). A block
    inheriting the torchvision shift would train with every label off by one and nothing raising.
    """
    for framework, expected_nc in (("lightning", 6), ("rfdetr", 5), ("ultralytics", 5)):
        doc = confluid.load(confluid.load_config(EXAMPLE), flow=False, scopes=[f"framework={framework}"])
        assert doc["num_classes"] == expected_nc, f"{framework}: num_classes"
        ops = doc["detection_ops"]
        offsets = [op.kwargs.get("label_offset") for op in ops if "CocoToTorchVision" in str(op.target)]
        assert offsets == [1 if framework == "lightning" else 0], f"{framework}: label_offset {offsets}"


def test_the_default_backend_has_a_block_of_its_own() -> None:
    """The top level already IS lightning, but confluid raises on an activation matching no
    block — so an undeclared default would make naming it explicitly an ERROR. That raise is the
    check a backend subcommand could never do: a typo now lists the declared backends instead of
    silently training the default one."""
    with pytest.raises(confluid.ScopeError) as excinfo:
        confluid.load(confluid.load_config(EXAMPLE), flow=False, scopes=["framework=lightnin"])
    assert "lightning" in str(excinfo.value), "the error must LIST the values the config declares"


def test_every_framework_names_registrations_that_import() -> None:
    """A `registrations` module that does not import leaves every bare registered name
    unresolvable. `ultralytics` is best-effort by design — its module lives in the AGPL sibling
    package, so it is only checked where that package is installed."""
    import importlib
    import importlib.util

    for name, spec in FRAMEWORKS.items():
        for module in spec.registrations:
            if importlib.util.find_spec(module.split(".")[0]) is None:
                assert name == "ultralytics", f"{name}: registrations module {module!r} is not installed"
                continue
            importlib.import_module(module)  # raises if the path is wrong
