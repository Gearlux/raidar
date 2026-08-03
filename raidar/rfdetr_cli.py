"""Raidar-RFDETR CLI — native RF-DETR training/evaluation via RF-DETR's own PTL stack.

Provides ``raidar-rfdetr train`` / ``raidar-rfdetr evaluate`` (the native RF-DETR
escape hatch, parallel to ``raidar train`` which drives the generic Lightning
detection path). The same verbs mount under ``matrainer rfdetr-train`` /
``matrainer rfdetr-evaluate`` via :mod:`raidar.rfdetr_plugin`.
"""

from liquifai import LiquifyApp
from loggair import get_logger

from raidar.rfdetr_trainer import RFDETREvaluator, RFDETRTrainer

logger = get_logger(__name__)

app = LiquifyApp(name="raidar-rfdetr", description="Native RF-DETR detection training and evaluation.")


@app.script_command(flow_mode="auto")
def train(trainer: RFDETRTrainer) -> None:
    """Train a detection model via RF-DETR's native PyTorch-Lightning trainer."""
    if trainer is None:
        logger.error("'trainer' was not bound — provide one under 'trainer:' in your YAML.")
        return
    trainer.run()


@app.script_command(flow_mode="auto")
def evaluate(evaluator: RFDETREvaluator) -> None:
    """Evaluate a trained RF-DETR checkpoint via RF-DETR's native validation loop."""
    if evaluator is None:
        logger.error("'evaluator' was not bound — provide one under 'evaluator:' in your YAML.")
        return
    evaluator.run()


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
