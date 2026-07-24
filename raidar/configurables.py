"""Register torchvision's detection model BUILDER FUNCTIONS as confluid configurables.

torchvision ships its detection models as builder FUNCTIONS (``fasterrcnn_resnet50_fpn(...)``,
``ssdlite320_mobilenet_v3_large(...)``, …), NOT classes — and they already conform to the native
detection contract :class:`raidar.detector.LightningDetector` consumes
(``model(images, targets) → loss dict`` in train, ``model(images) → predictions list`` in eval). We
``register`` each here with ``task="detection", role="model"`` (→ derived ``category="detection_model"``,
the same bucket as ``RFDETRModel``) + ``lazy=True`` so:

* confluid stamps ``__confluid_lazy__`` on the function (a deferred / runtime-injection slot — the
  trainer flows it with the dataset-derived ``num_classes`` at ``run()``);
* navigaitor's ``list_configurable_classes(category="detection_model")`` / ``list_classes(task=…,
  role=…)`` and the form-spec enumerate them alongside ``RFDETRModel`` — confluid's ``to_pydantic``
  is callable-aware (introspects the builder's OWN signature), so the function generates a valid
  config schema with no wrapper class; and
* FluxStudio generates an OBJECT node for it (``fluxstudio.bridge._build_object_node`` is
  callable-aware): widgets come from the builder's signature (``num_classes`` / ``weights_backbone``
  / …) and the node's role-qualified output socket (``SAMPLEFLUX_OBJECT:model``, carrying a
  ``LazyClass(builder, …)``) wires STRAIGHT into a detection trainer's ``model`` slot — no Retag.

Registered with a presentation-only ``group="detection"`` for palette nesting (mirroring how
``marainer.lightning.configurables`` groups the torch optimizers). The ``role="model"`` is what lets these raw
builders sit beside the raidar wrapper class (``RFDETRModel``) in every model picker.

Mask R-CNN / Keypoint R-CNN are intentionally excluded — they need instance masks / keypoints in the
target, not the ``{boxes, labels}`` shape ``raidar.detection.detection_collate_fn`` produces.

Exposed via the ``confluid.configurables`` entry point + ``__all__`` so FluxStudio's discovery second
pass surfaces them — ``scan_module`` alone skips them (their ``__module__`` is ``torchvision``).
"""

from confluid import register
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fcos_resnet50_fpn,
    retinanet_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large,
)

# All conform to the native detection forward contract and all take ``num_classes``.
# ``task="detection", role="model"`` → ``category="detection_model"``; ``lazy=True`` keeps the
# constructed value deferred; ``group`` is palette-nesting only (see the module docstring).
for _builder in (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    retinanet_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
    fcos_resnet50_fpn,
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large,
):
    register(_builder, task="detection", role="model", group="detection", lazy=True)

# Re-exported names FluxStudio discovery surfaces (the second, ``__all__``-gated pass).
__all__ = [
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
