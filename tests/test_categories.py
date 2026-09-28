# mypy: disable-error-code="attr-defined"
"""Discovery-category coverage for raidar ``@configurable`` classes.

These ``category=`` tags drive navigaitor's ``list_configurable_classes(category=...)``
and the visual-editor form-spec picker (``get_node_form_spec``). Pinned here so a
dropped tag fails CI instead of silently emptying a detection picker.
"""

import pytest
from confluid import marks
from confluid.registry import get_registry

from raidar.rfdetr import RFDETRModel


def test_detection_models_tagged() -> None:
    from raidar.huggingface_model import HuggingFaceDetectionModel

    assert RFDETRModel.__confluid_category__ == "detection_model"
    # The ONE Hub zoo wrapper shares the detection_model tag so the model picker offers it —
    # a checkpoint is a `model_name` STRING, not a preset class.
    assert HuggingFaceDetectionModel.__confluid_category__ == "detection_model"


def test_categories_enumerable_via_registry() -> None:
    registry = get_registry()
    listed = registry.list_classes(category="detection_model")
    assert "RFDETRModel" in listed
    assert "HuggingFaceDetectionModel" in listed


def test_yolo_sink_tagged() -> None:
    """``YoloSink`` carries ``category="sink"`` so StreamStudio surfaces it as a ``DatasetProcessor``
    sink node (``RECORDSTREAM_OBJECT:sink``) — ``matrainer convert`` streams a detection Stream into it on a
    canvas just as the YAML config does."""
    from raidar.yolo_sink import YoloSink

    assert YoloSink.__confluid_category__ == "sink"
    assert "YoloSink" in get_registry().list_classes(category="sink")


#: The METRIC half of `torch_detection.__all__` — the rest of that list is the batch plumbing
#: (`native_batch` / the prediction contracts), which is library
#: surface rather than registered classes. Stated here so the assertion below reads as "every
#: metric this module advertises is registered" and cannot silently pass by shrinking.
_TORCH_METRIC_NAMES = {
    "CompleteIntersectionOverUnion",
    "DetectionConfusionMatrix",
    "DistanceIntersectionOverUnion",
    "GeneralizedIntersectionOverUnion",
    "IntersectionOverUnion",
    "MeanAveragePrecision",
}


def test_the_torch_engine_metric_set_is_registered() -> None:
    """The engine's metrics live in its OWN module (`torch_detection.py`) with a literal
    ``__all__`` — the deltaid one-module-per-engine rule. The torchmetrics detection family is
    registered upstream-class-as-is; ``DetectionConfusionMatrix`` is the one metric raidar owns."""
    import raidar.torch_detection as td

    registry = get_registry()
    metrics = set(registry.list_classes(task="detection", role="metric"))
    assert _TORCH_METRIC_NAMES <= set(td.__all__), "a metric vanished from the module's palette list"
    assert _TORCH_METRIC_NAMES <= metrics, metrics
    assert td.DetectionConfusionMatrix.__confluid_framework__ == "torch"


def test_the_batch_plumbing_shares_the_engine_module_and_is_not_registered() -> None:
    """The collate + its container are LIBRARY surface, not configurables: they are wired in code
    by every torch-family backend, never named in a config, so tagging them would put plumbing in
    a model/metric picker."""
    import raidar.torch_detection as td

    plumbing = {"native_batch"}
    assert plumbing <= set(td.__all__)
    listed = set(get_registry().list_classes(task="detection"))
    assert not (plumbing & listed), "batch plumbing must not be registered"


def test_task_role_tags() -> None:
    """task/role drive navigaitor's scan-and-generate task surfaces."""
    from raidar.lightning_detector import LightningDetector

    assert RFDETRModel.__confluid_task__ == "detection"
    assert RFDETRModel.__confluid_role__ == "model"
    # The merged LightningDetector (train + eval in one standard class) is tagged role="trainer";
    # `framework` is the third discovery axis (what a class can be WIRED TO).
    assert LightningDetector.__confluid_task__ == "detection"
    assert LightningDetector.__confluid_role__ == "trainer"
    assert LightningDetector.__confluid_framework__ == "torch"
    registry = get_registry()
    assert "RFDETRModel" in registry.list_classes(task="detection", role="model")
    assert "LightningDetector" in registry.list_classes(task="detection", role="trainer")


def test_detection_is_a_multi_trainer_task() -> None:
    """The merged RF-DETR runnable is tagged for detection ALONGSIDE the Lightning detector, so
    navigaitor can navigate multiple trainers per task. Each class's evaluator capability is
    declared by its ``@entrypoint`` markers, not by a second class — there is no separate
    evaluator class on either side."""
    from recordstream.runnable import entrypoint_tasks

    from raidar.lightning_detector import LightningDetector
    from raidar.rfdetr_detector import RFDETRDetector

    assert RFDETRDetector.__confluid_task__ == "detection" and RFDETRDetector.__confluid_role__ == "trainer"
    assert RFDETRDetector.__confluid_framework__ == "rfdetr"

    registry = get_registry()
    trainers = registry.list_classes(task="detection", role="trainer")
    assert {"LightningDetector", "RFDETRDetector"} <= set(trainers)
    # The evaluator capability comes from the markers on both classes.
    assert entrypoint_tasks(LightningDetector, "evaluator")
    assert entrypoint_tasks(RFDETRDetector, "evaluator")


# torchvision detection BUILDER FUNCTIONS registered by raidar.torchvision_detection.
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
    so a StreamStudio object node + every navigaitor model picker offers them beside RFDETRModel."""
    import raidar.torchvision_detection  # noqa: F401 - registration side-effect

    registry = get_registry()
    listed = registry.list_classes(category="detection_model")
    for builder in _BUILDER_NAMES:
        assert builder in listed, f"{builder} not registered as detection_model"
    # Marked lazy (a deferred runtime-injection slot — the trainer flows num_classes at run()).
    # Read through `marks()`, the ONE public stamp surface — the dunder names are confluid-internal
    # (the stamp was renamed once already, which is what broke the raw read this replaced).
    fn_marks = marks(raidar.torchvision_detection.fasterrcnn_mobilenet_v3_large_320_fpn)
    assert fn_marks.lazy is True
    assert fn_marks.role == "model"
    assert fn_marks.group == "detection"


@pytest.mark.filterwarnings("ignore::pydantic.json_schema.PydanticJsonSchemaWarning")
def test_torchvision_builder_generates_config_schema() -> None:
    """``to_pydantic`` is callable-aware, so a registered builder FUNCTION produces a JSON-Schema-able
    config (navigaitor's form-spec / MCP schema path) — ``num_classes`` enumerated from its signature.

    (A benign ``PydanticJsonSchemaWarning`` fires: the non-JSON-serializable ``weights_backbone`` enum
    DEFAULT is excluded from the schema. The field still generates; ``flow()`` uses torchvision's real
    default at runtime, so this is informational-only and filtered here.)"""
    from confluid import to_pydantic

    import raidar.torchvision_detection

    model = to_pydantic(raidar.torchvision_detection.fasterrcnn_resnet50_fpn)
    assert "num_classes" in model.model_fields
    model.model_json_schema()  # must not raise (Weights enum / Callable params coerced to Any)
