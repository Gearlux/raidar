"""Raidar CLI — object-detection training and evaluation commands.

Provides ``raidar train`` / ``raidar evaluate`` for the Lightning detection
flow. The native-ultralytics escape hatch lives in the
[raidar-ultralytics](https://github.com/Gearlux/raidar-ultralytics) sibling
package as ``raidar-ultralytics train`` / ``raidar-ultralytics evaluate``
(or ``marainer ultralytics-train`` / ``marainer ultralytics-evaluate``).
"""

from liquifai import LiquifyApp
from logflow import get_logger

from raidar.evaluator import LightningDetectionEvaluator
from raidar.lightning_trainer import LightningDetectionTrainer

logger = get_logger(__name__)

app = LiquifyApp(name="raidar", description="Raidar object-detection commands")


@app.script_command(flow_mode="auto")
def train(trainer: LightningDetectionTrainer) -> None:
    """Train an object-detection model via the Lightning trainer subclass."""
    trainer.run()


@app.script_command(flow_mode="auto")
def evaluate(evaluator: LightningDetectionEvaluator) -> None:
    """Evaluate a trained object-detection model via the Lightning evaluator subclass."""
    evaluator.run()


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
