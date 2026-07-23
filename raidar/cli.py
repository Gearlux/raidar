"""Raidar CLI — object-detection training and evaluation.

``raidar train`` / ``raidar evaluate`` / ``raidar predict`` bind the single
:class:`~raidar.detector.LightningDetector` (a standard Lightning module) and run it
with the matching ``task`` (the verb wins over the YAML's ``task:``). The RF-DETR
framework-native trainer keeps its own ``raidar-rfdetr`` CLI; the ultralytics escape
hatch lives in the raidar-ultralytics sibling. A generic ``sampleflux run`` also works.
"""

from typing import Any

from liquifai import LiquifyApp
from loggair import get_logger

logger = get_logger(__name__)

app = LiquifyApp(name="raidar", description="Raidar object-detection commands")


def _dispatch(detector: Any, task: str) -> None:
    if detector is None:
        logger.error("No detector was bound — provide one under 'runnable:' in your YAML.")
        return
    from confluid import flow
    from confluid.fluid import Fluid

    if isinstance(detector, Fluid):
        detector = flow(detector)
    if hasattr(detector, "task"):
        detector.task = task
    detector.run()


@app.script_command(flow_mode="auto")
def train(runnable: Any) -> None:
    """Train an object-detection model (task=fit)."""
    _dispatch(runnable, "fit")


@app.script_command(flow_mode="auto")
def evaluate(runnable: Any) -> None:
    """Evaluate a trained detector over the test set (task=test)."""
    _dispatch(runnable, "test")


@app.script_command(flow_mode="auto")
def predict(runnable: Any) -> None:
    """Stream detection predictions to the sink (task=predict)."""
    _dispatch(runnable, "predict")


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
