"""Raidar CLI — object detection across training backends.

Three verbs, and the training backend is a *config* dimension the CLI passes through::

    raidar train    cfg.yaml                       # the config's default backend
    raidar train    cfg.yaml --framework rfdetr    # a `!scope:framework=rfdetr` block
    raidar evaluate cfg.yaml
    raidar predict  cfg.yaml
    raidar frameworks                              # what the backends are

What stays constant across backends is the point: the same ``runnable:`` + ``task:`` config, the
same three verbs, the same tracking. Only the runnable class named in the YAML changes. (A generic
``matrainer train cfg.yaml`` or ``recordstream run cfg.yaml`` also works — the first is the same
machinery without this project's table.)

The MACHINERY is :mod:`matrainer.cli` (2026-08-07, the P4 lift) — the verb dispatch through
``materialize_runnable``, the ``--framework`` → ``--scope`` rewrite, the keras engine pin, the
registration import and their load-bearing ORDER all live on the factory's own docstrings. Two
raidar notes on that machinery: the taskless-fit guard (a runnable with NO ``task`` attribute —
the ultralytics sibling's current trainer — trains as-is and RAISES on the other verbs) started
HERE and is now the shared default dispatch; and the keras engine pin, which this CLI never
carried (its tensorflow row registers nothing, and ``raidar._tensorflow`` sets the same default
at import), now rides along harmlessly — both are ``setdefault``, so they cannot disagree. What
this module owns is raidar's DATA: the FRAMEWORKS table and the verb help texts.

Adding a backend: write ``raidar/<name>_detector.py``, then add a row to ``FRAMEWORKS`` below.
"""

from typing import Dict

from matrainer.cli import Framework, make_trainer_app, trainer_main

FRAMEWORKS: Dict[str, Framework] = {
    "lightning": Framework(
        "PyTorch Lightning — the reference backend (torchvision natives, RFDETRModel, TorchFcos).",
        # The torch engine's TWO registration modules: the torchmetrics detection metric set +
        # DetectionConfusionMatrix, and the torchvision detection BUILDERS — so a config may name
        # any of them bare (the eager `!class:Foo()` form is built during config load, before any
        # handler runs). A TUPLE keeps the coupling readable in the table instead of hiding it in
        # a side-effect import inside one of them.
        ("raidar.torch_detection", "raidar.torchvision_detection"),
    ),
    # Shares `lightning`'s registration modules, and that is the point rather than an economy: a
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
        ("raidar.ultralytics",),
    ),
}

app = make_trainer_app(
    name="raidar",
    description="Object detection across training backends",
    frameworks=FRAMEWORKS,
    verb_docs={
        "train": "Train an object-detection model",
        "evaluate": "Evaluate a trained detector over the test set",
        "predict": "Stream detection predictions from a trained detector to its sink",
    },
)


def main() -> None:
    trainer_main(app, FRAMEWORKS)


if __name__ == "__main__":
    main()
