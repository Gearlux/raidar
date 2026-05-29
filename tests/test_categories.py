# mypy: disable-error-code="attr-defined"
"""Discovery-category coverage for raidar ``@configurable`` classes.

These ``category=`` tags drive navigaitor's ``list_configurable_classes(category=...)``
and the visual-editor form-spec picker (``get_node_form_spec``). Pinned here so a
dropped tag fails CI instead of silently emptying a detection picker.
"""

from confluid.registry import get_registry

from raidar.detection import TorchvisionDetectionLoss
from raidar.faster_rcnn import FasterRCNNModel
from raidar.rfdetr import RFDETRModel


def test_detection_models_tagged() -> None:
    assert FasterRCNNModel.__confluid_category__ == "detection_model"
    assert RFDETRModel.__confluid_category__ == "detection_model"


def test_detection_loss_tagged() -> None:
    assert TorchvisionDetectionLoss.__confluid_category__ == "detection_loss"


def test_categories_enumerable_via_registry() -> None:
    registry = get_registry()
    assert {"FasterRCNNModel", "RFDETRModel"} <= registry.list_classes(category="detection_model")
    assert "TorchvisionDetectionLoss" in registry.list_classes(category="detection_loss")


def test_task_role_tags() -> None:
    """task/role drive navigaitor's scan-and-generate task surfaces."""
    from raidar.evaluator import LightningDetectionEvaluator
    from raidar.lightning_trainer import LightningDetectionTrainer

    assert FasterRCNNModel.__confluid_task__ == "detection"
    assert FasterRCNNModel.__confluid_role__ == "model"
    assert TorchvisionDetectionLoss.__confluid_role__ == "loss"
    assert LightningDetectionTrainer.__confluid_task__ == "detection"
    assert LightningDetectionTrainer.__confluid_role__ == "trainer"
    assert LightningDetectionEvaluator.__confluid_role__ == "evaluator"
    registry = get_registry()
    assert {"FasterRCNNModel", "RFDETRModel"} <= registry.list_classes(task="detection", role="model")
    assert "LightningDetectionTrainer" in registry.list_classes(task="detection", role="trainer")
