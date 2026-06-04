# mypy: disable-error-code="attr-defined"
"""Discovery-category coverage for raidar ``@configurable`` classes.

These ``category=`` tags drive navigaitor's ``list_configurable_classes(category=...)``
and the visual-editor form-spec picker (``get_node_form_spec``). Pinned here so a
dropped tag fails CI instead of silently emptying a detection picker.
"""

import pytest
from confluid.registry import get_registry

from raidar.rfdetr import RFDETRModel


def test_detection_models_tagged() -> None:
    assert RFDETRModel.__confluid_category__ == "detection_model"


def test_categories_enumerable_via_registry() -> None:
    registry = get_registry()
    assert "RFDETRModel" in registry.list_classes(category="detection_model")


def test_task_role_tags() -> None:
    """task/role drive navigaitor's scan-and-generate task surfaces."""
    from raidar.evaluator import LightningDetectionEvaluator
    from raidar.lightning_trainer import LightningDetectionTrainer

    assert RFDETRModel.__confluid_task__ == "detection"
    assert RFDETRModel.__confluid_role__ == "model"
    assert LightningDetectionTrainer.__confluid_task__ == "detection"
    assert LightningDetectionTrainer.__confluid_role__ == "trainer"
    assert LightningDetectionEvaluator.__confluid_role__ == "evaluator"
    registry = get_registry()
    assert "RFDETRModel" in registry.list_classes(task="detection", role="model")
    assert "LightningDetectionTrainer" in registry.list_classes(task="detection", role="trainer")


# torchvision detection BUILDER FUNCTIONS registered by raidar.configurables.
_BUILDER_NAMES = [
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "retinanet_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
]


def test_torchvision_builders_registered_as_detection_models() -> None:
    """The raw torchvision builder FUNCTIONS register as ``detection_model`` (role="model"),
    so a FluxStudio object node + every navigaitor model picker offers them beside RFDETRModel."""
    import raidar.configurables  # noqa: F401 - registration side-effect

    registry = get_registry()
    listed = registry.list_classes(category="detection_model")
    for builder in _BUILDER_NAMES:
        assert builder in listed, f"{builder} not registered as detection_model"
    # Marked lazy (a deferred runtime-injection slot — the trainer flows num_classes at run()).
    fn = raidar.configurables.fasterrcnn_mobilenet_v3_large_320_fpn
    assert fn.__confluid_lazy__ is True  # type: ignore[attr-defined]
    assert fn.__confluid_role__ == "model"  # type: ignore[attr-defined]
    assert fn.__confluid_group__ == "detection"  # type: ignore[attr-defined]


@pytest.mark.filterwarnings("ignore::pydantic.json_schema.PydanticJsonSchemaWarning")
def test_torchvision_builder_generates_config_schema() -> None:
    """``to_pydantic`` is callable-aware, so a registered builder FUNCTION produces a JSON-Schema-able
    config (navigaitor's form-spec / MCP schema path) — ``num_classes`` enumerated from its signature.

    (A benign ``PydanticJsonSchemaWarning`` fires: the non-JSON-serializable ``weights_backbone`` enum
    DEFAULT is excluded from the schema. The field still generates; ``flow()`` uses torchvision's real
    default at runtime, so this is informational-only and filtered here.)"""
    from confluid import to_pydantic

    import raidar.configurables

    model = to_pydantic(raidar.configurables.fasterrcnn_resnet50_fpn)
    assert "num_classes" in model.model_fields
    model.model_json_schema()  # must not raise (Weights enum / Callable params coerced to Any)
