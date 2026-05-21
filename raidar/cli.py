"""Raidar CLI — object-detection training and evaluation commands.

Provides ``raidar train`` / ``raidar evaluate`` (Lightning path) and
``raidar ultralytics-train`` / ``raidar ultralytics-evaluate`` (native
ultralytics escape hatch). The Lightning verbs are equivalent to
``marainer detection train`` / ``marainer detection evaluate``; the
ultralytics verbs are equivalent to ``marainer ultralytics-train`` /
``marainer ultralytics-evaluate``.
"""

import sys

from liquifai import LiquifyApp
from logflow import get_logger

from raidar.evaluator import LightningDetectionEvaluator
from raidar.lightning_trainer import LightningDetectionTrainer
from raidar.ultralytics_trainer import UltralyticsEvaluator, UltralyticsTrainer

logger = get_logger(__name__)

if "." not in sys.path:
    sys.path.insert(0, ".")

app = LiquifyApp(name="raidar", description="Raidar object-detection commands")


@app.script_command(flow_mode="auto")
def train(trainer: LightningDetectionTrainer) -> None:
    """Train an object-detection model via the Lightning trainer subclass."""
    trainer.run()


@app.script_command(flow_mode="auto")
def evaluate(evaluator: LightningDetectionEvaluator) -> None:
    """Evaluate a trained object-detection model via the Lightning evaluator subclass."""
    evaluator.run()


@app.script_command(flow_mode="auto")
def ultralytics_train(trainer: UltralyticsTrainer) -> None:
    """Train a YOLO26 model via Ultralytics' native trainer (mosaic + albumentations + LOGGER)."""
    if trainer is None:
        logger.error("'trainer' was not bound — provide one under 'trainer:' in your YAML.")
        return
    trainer.run()


@app.script_command(flow_mode="auto")
def ultralytics_evaluate(evaluator: UltralyticsEvaluator) -> None:
    """Evaluate a trained YOLO26 checkpoint via Ultralytics' native validator."""
    if evaluator is None:
        logger.error("'evaluator' was not bound — provide one under 'evaluator:' in your YAML.")
        return
    evaluator.run()


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
