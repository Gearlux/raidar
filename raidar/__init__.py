"""Raidar: object detection across training backends, driven by one config.

**The core is backend-free.** ``pip install raidar`` gives you the CLI and the config surface;
a training framework arrives with the backend you select::

    pip install 'raidar[lightning]'          # PyTorch Lightning — the reference backend
    pip install 'raidar[rfdetr]'             # RF-DETR's own PTL loop (Apache-2.0)
    pip install 'raidar[huggingface]'        # + the D-FINE / RT-DETRv2 model wrappers

Which is why every backend-owned name below resolves LAZILY (PEP 562): importing ``raidar`` must
not require a framework the operator did not choose. Touching a name whose backend is missing
raises an :class:`ImportError` naming the exact extra, instead of a traceback from three
libraries down.
"""

from importlib.metadata import PackageNotFoundError, version

from matrainer._exports import lazy_exports

try:
    # Single source of truth: the installed distribution's metadata, i.e. pyproject.toml's
    # `version` (the distribution is `raidar`). Never type the number in here as well.
    __version__ = version("raidar")
except PackageNotFoundError:  # pragma: no cover - uninstalled source checkout
    __version__ = "0.0.0.dev0"


#: Backend-owned name -> (defining module, the extra that installs it). Resolved on first
#: attribute access. Keep in step with the extras in ``pyproject.toml``: a name here whose extra
#: does not exist would advertise an install command that fails.
_BACKEND_NAMES = {
    "LightningDetector": ("raidar.lightning_detector", "lightning"),
    "TorchDetector": ("raidar.torch_detector", "torch"),
    "FastaiDetector": ("raidar.fastai_detector", "fastai"),
    "HuggingFaceDetector": ("raidar.huggingface_detector", "huggingface"),
    "RFDETRDetector": ("raidar.rfdetr_detector", "rfdetr"),
    # The native RF-DETR model module (runs through the LIGHTNING backend; building it needs the
    # `rfdetr` library, so its extra is the one that supplies the weights + architecture).
    "RFDETRModel": ("raidar.rfdetr", "rfdetr"),
    # The ONE Hub zoo wrapper — a `model_name` string reaches any object-detection checkpoint;
    # it is the huggingface BACKEND's model (the model computes its own loss from `labels=`).
    "HuggingFaceDetectionModel": ("raidar.huggingface_model", "huggingface"),
    # The torchvision zoo behind ONE configured wrapper (weights/score cut/label numbering/
    # checkpoint load) — the inference twin of the raw builder functions the trainers wire.
    "TorchVisionDetectionModel": ("raidar.torchvision_model", "torch"),
    # The torch engine's registered metric set (one module per engine). Batching is
    # recordstream's — `collate_records` + `batch_boxes`, no raidar collate.
    "DetectionConfusionMatrix": ("raidar.torch_detection", "torch"),
    # The shared tiny-FCOS spec's torch implementation (serves the lightning/torch/fastai loops).
    "TorchFcos": ("raidar.fcos.torch_fcos", "torch"),
    # The Keras 3 backend + the spec's Keras implementation and its dense-target op.
    "KerasDetector": ("raidar.keras_detector", "keras"),
    "KerasFcos": ("raidar.fcos.keras_fcos", "keras"),
    # The native TensorFlow loop + the spec in raw tf.nn ops.
    "TensorFlowDetector": ("raidar.tensorflow_detector", "tensorflow"),
    "TensorFlowFcos": ("raidar.fcos.tensorflow_fcos", "tensorflow"),
    # The jitted optax loop + the spec in flax.nnx.
    "JaxDetector": ("raidar.jax_detector", "jax"),
    "FlaxFcos": ("raidar.fcos.flax_fcos", "jax"),
    # Apple's MLX loop + the spec in mlx.nn (Apple silicon only).
    "MlxDetector": ("raidar.mlx_detector", "mlx"),
    "MlxFcos": ("raidar.fcos.mlx_fcos", "mlx"),
    # FcosTargets is numpy + the spec only, but it reads Boxes targets the torch-shaped chain
    # produced — its extra is torch (the smallest install that can feed it).
    "FcosTargets": ("raidar.fcos.targets", "torch"),
    # The on-disk YOLO-layout sink is numpy/PIL/yaml only — no framework — but it stays lazy so
    # `import raidar` imports NOTHING. Its "extra" is the core install itself.
    "YoloSink": ("raidar.yolo_sink", ""),
}

__all__ = [*_BACKEND_NAMES]

# The resolver is matrainer's `lazy_exports` (the machinery was identical in every training
# project's root); the `_BACKEND_NAMES` table above is the project DATA it runs on. An EMPTY
# extra (YoloSink) passes its ImportError through untranslated — a core name's failure is real.
__getattr__, __dir__ = lazy_exports(__name__, _BACKEND_NAMES)
