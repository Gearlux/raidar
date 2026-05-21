"""Marainer plugin entry point for raidar.

Registers two CLI verbs on the marainer ``LiquifyApp``:

* ``ultralytics-train``  → :class:`raidar.ultralytics_trainer.UltralyticsTrainer`
* ``ultralytics-evaluate`` → :class:`raidar.ultralytics_trainer.UltralyticsEvaluator`

Wired in :file:`pyproject.toml` under
``[project.entry-points."marainer.commands"]``. The marainer CLI calls
``register(app)`` once during bootstrap (see :mod:`marainer.plugins`).
"""

from typing import TYPE_CHECKING

from logflow import get_logger

from raidar.ultralytics_trainer import UltralyticsEvaluator, UltralyticsTrainer

if TYPE_CHECKING:
    from liquifai import LiquifyApp

logger = get_logger(__name__)


def register(app: "LiquifyApp") -> None:
    """Decorate raidar's two ultralytics-native commands onto ``app``."""

    @app.script_command(flow_mode="auto")
    def ultralytics_train(trainer: UltralyticsTrainer) -> None:
        """Train a YOLO26 model via Ultralytics' native trainer
        (mosaic + albumentations + LOGGER) instead of PyTorch Lightning.
        """
        if trainer is None:
            logger.error("'trainer' was not bound — provide one under 'trainer:' in your YAML.")
            return
        logger.info(f"raidar ultralytics-train: {trainer.experiment_name}")
        trainer.run()

    @app.script_command(flow_mode="auto")
    def ultralytics_evaluate(evaluator: UltralyticsEvaluator) -> None:
        """Evaluate a trained YOLO26 checkpoint via Ultralytics' native validator."""
        if evaluator is None:
            logger.error("'evaluator' was not bound — provide one under 'evaluator:' in your YAML.")
            return
        logger.info(f"raidar ultralytics-evaluate: {evaluator.experiment_name}")
        evaluator.run()


__all__ = ["register"]
