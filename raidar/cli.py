"""Raidar CLI — object detection across training backends.

Three verbs, and the training backend is a *config* dimension the CLI passes through::

    raidar train    cfg.yaml                       # the config's default backend
    raidar train    cfg.yaml --framework rfdetr    # a `!scope:framework=rfdetr` block
    raidar evaluate cfg.yaml
    raidar predict  cfg.yaml
    raidar frameworks                              # what the backends are

What stays constant across backends is the point: the same ``runnable:`` + ``task:`` config, the
same three verbs, the same tracking. Only the runnable class named in the YAML changes. (A generic
``recordstream run cfg.yaml`` also works — it calls ``.run()`` with whatever ``task`` the YAML
declares, with no verb at all.)

Adding a backend: write ``raidar/<name>_detector.py``, then add a row to ``FRAMEWORKS`` below.
"""

import sys
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from liquifai import LiquifyApp
from loggair import get_logger

logger = get_logger(__name__)

app = LiquifyApp(name="raidar", description="Object detection across training backends")


def _dispatch(runnable: Any, task: str) -> None:
    """Build the bound runnable, force ``task``, and run it.

    The verb wins over any ``task:`` set in YAML, so one config serves all three verbs.
    Backend-agnostic: needs only a ``task`` attribute and ``run()``.

    Building goes through ``recordstream.cli.materialize_runnable`` — NOT a bare ``flow()`` —
    because the flat config's top-level keys (``train_set`` / ``batch_size`` / ``max_epochs`` /
    ``model`` …) only broadcast into the constructor when the node is built against the document
    it came from. A bare flow drops them SILENTLY: `train_set` becomes None (surfacing much later
    as a failure to derive num_classes) and `max_epochs: 3` quietly falls back to the constructor
    default, which never surfaces at all. Same helper as `recordstream run`, so both runners share
    one answer.

    A runnable with NO ``task`` attribute cannot take the verb, and the guard is not deltaid's
    silent ``hasattr``: for anything but ``fit`` this RAISES rather than running the class's one
    behaviour under the wrong verb — ``raidar evaluate`` over a train-only runnable (the
    ultralytics sibling's current trainer) must not silently TRAIN.
    """
    if runnable is None:
        raise ValueError("No runnable was bound — provide one under 'runnable:' in your YAML.")

    from recordstream.cli import materialize_runnable

    runnable = materialize_runnable(runnable)
    if hasattr(runnable, "task"):
        runnable.task = task  # the verb wins over the YAML's task
    elif task != "fit":
        raise ValueError(
            f"{type(runnable).__name__} has no `task` knob, so this verb cannot drive it "
            f"(task={task!r}); it only knows how to run its single behaviour. Use `raidar train`, "
            "or wire a runnable that dispatches on `task`."
        )
    else:
        logger.info(f"{type(runnable).__name__} has no `task` knob — running it as-is.")
    runnable.run()


# --------------------------------------------------------------------------- #
# The backends
# --------------------------------------------------------------------------- #
class Framework(NamedTuple):
    """One training backend: what to say about it, what to import for it."""

    description: str
    #: The modules whose import registers this engine's classes under a plain NAME. A TUPLE
    #: because an engine may register from more than one module (the torch engine registers its
    #: metrics from `torch_detection` and the torchvision builders from `torchvision_detection`)
    #: — listing both HERE keeps the coupling readable in the table instead of hiding it in a
    #: side-effect import inside one of them. Empty means the backend has NOTHING to register,
    #: which is a real state rather than an oversight: a backend whose models are wired by dotted
    #: path leaves nothing for a bare registered name to resolve.
    registrations: Tuple[str, ...] = ()


FRAMEWORKS: Dict[str, Framework] = {
    "lightning": Framework(
        "PyTorch Lightning — the reference backend (torchvision natives, RFDETRModel, TorchFcos).",
        # The torch engine's TWO registration modules: the torchmetrics detection metric set +
        # DetectionConfusionMatrix, and the torchvision detection BUILDERS — so a config may name
        # any of them bare (the eager `!class:Foo()` form is built during config load, before any
        # handler runs).
        ("raidar.torch_detection", "raidar.torchvision_detection"),
    ),
    # Shares `lightning`'s registration module, and that is the point rather than an economy: a
    # torchmetrics detection metric does not care whether a Trainer or a hand-written loop feeds
    # it, so the two backends differ ONLY in the loop.
    "torch": Framework(
        "PyTorch — a hand-written loop, no training framework at all (the control experiment).",
        ("raidar.torch_detection", "raidar.torchvision_detection"),
    ),
    "fastai": Framework(
        "fastai — a Learner with one-cycle scheduling over the same torch stack (no-op-loss "
        "bridge for the loss-inside-forward contract).",
        # The same torch registrations reach it — a torchmetrics collection and a torchvision
        # builder do not care which loop feeds them.
        ("raidar.torch_detection", "raidar.torchvision_detection"),
    ),
    "huggingface": Framework(
        "Transformers `Trainer` + Accelerate — fine-tuning a Hub detection checkpoint by name.",
        # No registrations: an `AutoModelForObjectDetection` computes its own loss from
        # `labels=`, so there is nothing a config could wire.
    ),
    "rfdetr": Framework(
        "RF-DETR's own PyTorch-Lightning loop (Apache-2.0) — COCO-mAP eval, EMA, LR schedule.",
        # Nothing to register: the runnable and the model are wired by dotted path.
    ),
    "keras": Framework(
        "Keras 3 — the shared FCOS spec on whichever engine KERAS_BACKEND names (torch / TF / jax).",
        # Nothing to register: the spec model and the dense-target op are wired by dotted path,
        # the loss is the spec's own (no compile-slot vocabulary), and scoring is numpy.
    ),
    "tensorflow": Framework(
        "Native TensorFlow — tf.data + GradientTape steps under tf.function, SavedModel export.",
        # Nothing to register, for the keras row's reason — and no second registration module:
        # TF >= 2.16's tf.keras IS Keras 3, so one would only duplicate tags.
    ),
    "jax": Framework(
        "JAX — an nnx.jit-traced optax loop over the shared FCOS spec in flax.nnx.",
        # Nothing to register: optax ships plain functions/factories — no class for a tag.
    ),
    "mlx": Framework(
        "Apple MLX — unified-memory training of the shared FCOS spec (Apple silicon only).",
        # Nothing to register, for the jax row's reason.
    ),
    "ultralytics": Framework(
        "Ultralytics YOLO (AGPL) — lives in the raidar-ultralytics sibling package; "
        "`pip install raidar-ultralytics`.",
        # Best-effort import — registers the sibling's models/trainer when it is installed;
        # absent on a default checkout, which is the correct state for an AGPL opt-in.
        ("raidar_ultralytics",),
    ),
}

#: Verb -> the ``task`` it forces. The verb wins over any ``task:`` in the YAML.
VERB_TASKS = {"train": "fit", "evaluate": "test", "predict": "predict"}

_VERB_DOCS = {
    "train": "Train an object-detection model",
    "evaluate": "Evaluate a trained detector over the test set",
    "predict": "Stream detection predictions from a trained detector to its sink",
}


def _verb_command(verb: str) -> Callable[[Any], None]:
    """One verb.

    A factory, not a default-argument closure: liquifai reads the handler's SIGNATURE to build the
    command, so smuggling the loop variable in as a default would advertise it as a ``--_verb``
    option in help and TAB completion. The signature it must expose is exactly ``(runnable)`` —
    the parameter NAME is what liquifai's DI binds to the config's ``runnable:`` key.
    """

    def command(runnable: Any) -> None:
        _dispatch(runnable, VERB_TASKS[verb])

    command.__name__ = verb
    command.__doc__ = f"{_VERB_DOCS[verb]} (task={VERB_TASKS[verb]}). Pick a backend with --framework."
    return command


for _verb in VERB_TASKS:
    app.script_command(name=_verb, flow_mode="manual")(_verb_command(_verb))


@app.command(name="frameworks")
def frameworks() -> None:
    """List the training backends `--framework` accepts."""
    width = max(len(name) for name in FRAMEWORKS)
    print("Training backends — pass one as `--framework <name>`:\n")
    for name, spec in FRAMEWORKS.items():
        print(f"  {name:<{width}}  {spec.description}")
    print(
        "\nThe name must also match a `!scope:framework=<name>` block in the config;"
        "\nconfluid lists the values a config declares when it does not."
    )


def _register_detection(framework: Optional[str]) -> None:
    """Import a backend's registrations so a config may NAME its classes bare.

    A ``!class:`` target that is a registered name resolves only once the module that registers it
    has been imported; a dotted path resolves through importlib either way. Discovery services
    bootstrap every ``confluid.configurables`` entry point, but a CLI does not.

    It happens from ``main()``, before ``app.run()``, because the EAGER instance form
    (``!class:Foo()`` with parens) is built during config load — earlier than any handler runs.
    With no ``--framework`` this imports nothing: the flag is what names the engine, and there is
    no safe guess. A config that names no engine-specific set, or spells its classes as dotted
    paths, needs none of this.

    Best-effort: a missing backend is a normal state on a partial install, and its absent
    registrations are the correct outcome, not an error.
    """
    if framework is None:
        return
    import importlib

    for module in FRAMEWORKS.get(framework, Framework("")).registrations:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            logger.debug(f"{framework}: registrations unavailable ({module}: {exc})")


def _normalize_framework_flag(argv: List[str]) -> None:
    """Rewrite ``--framework <name>`` into the ``--scope framework=<name>`` liquifai reads.

    liquifai binds ``--KEY VAL`` to scope activation only for dimensions the config DECLARES;
    against a config with no ``!scope:framework=…`` block at all, a bare ``--framework lightning``
    would instead be read as a config override and try to set a ``framework`` attribute on
    something. Rewriting here makes the flag mean one thing regardless of what the config happens
    to declare — an undeclared dimension stays an inert no-op, which is what lets a config grow
    into a backend it does not have yet.

    Mutates ``argv`` in place because scopes are parsed from the command line, so this must land
    before ``run()``. The explicit ``--scope framework=…`` spelling is left exactly as typed.
    """
    out: List[str] = []
    i, n = 0, len(argv)
    while i < n:
        arg = argv[i]
        if arg.startswith("--framework="):
            out += ["--scope", f"framework={arg.split('=', 1)[1]}"]
        elif arg == "--framework" and i + 1 < n:
            out += ["--scope", f"framework={argv[i + 1]}"]
            i += 1
        else:
            out.append(arg)
        i += 1
    argv[:] = out


def _requested_framework(argv: List[str]) -> Optional[str]:
    """The backend named on the command line, in any spelling, or ``None``.

    Read from ``argv`` rather than from liquifai because the caller needs it BEFORE the config is
    loaded — see :func:`_register_detection`.
    """
    for i, arg in enumerate(argv):
        if arg.startswith("--framework="):
            return arg.split("=", 1)[1]
        if arg.startswith(("--scope=", "-s=")) and "framework=" in arg:
            return arg.split("framework=", 1)[1]
        if arg in ("--framework", "--scope", "-s") and i + 1 < len(argv):
            following = argv[i + 1]
            if arg == "--framework":
                return following
            if following.startswith("framework="):
                return following.split("=", 1)[1]
    return None


def main() -> None:
    framework = _requested_framework(sys.argv)
    _register_detection(framework)
    _normalize_framework_flag(sys.argv)
    app.run()


if __name__ == "__main__":
    main()
