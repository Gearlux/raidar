"""``HuggingFaceDetectionModel`` — an ``AutoModelForObjectDetection`` behind the model contract.

The Hub is the zoo here, so this is the same shape as the workspace's other Hub wrappers: ONE
light generic class taking a ``model_name`` — ``"ustc-community/dfine-small-coco"``,
``"PekingU/rtdetr_v2_r18vd"``, ``"facebook/detr-resnet-50"``, any Hub object-detection checkpoint
— with ``__init__`` storing only, the network built on first use by :meth:`solidify` (confluid's
``flow()``-time hook), and required configuration validated at build time so zero-arg
construction keeps working. There are deliberately NO per-checkpoint preset classes: a checkpoint
is a STRING, not a class.

Three things are specific to the Hub and are why this class exists rather than a config wiring
``transformers`` directly:

* **``num_classes`` arrives at run time.** A pretrained checkpoint carries its own head (80 COCO
  classes for most detectors), so pointing it at a 5-class dataset is a shape mismatch. The build
  passes ``num_labels`` plus ``ignore_mismatched_sizes=True``, which tells transformers to
  DISCARD the pretrained head and start a fresh one at the requested width while backbone /
  encoder / decoder stay pretrained.
* **The model computes its own loss**, from ``labels=`` (the DETR-family Hungarian matcher), so
  the huggingface BACKEND drives it with no loss slot — see
  :mod:`raidar.huggingface_detector`.
* **The image processor is part of the checkpoint.** Resizing, padding, normalization and the
  decode back to pixel-space boxes are the PROCESSOR's job, so the wrapper exposes it
  (:meth:`image_processor`) and the backend's collator and eval path use it — raidar reimplements
  none of that shaping.

Labels are 0-indexed ``0..K-1`` with no background slot (the transformers detection convention —
wire ``CocoToTorchVisionDetection(label_offset=0)``, the opposite of torchvision's
background-at-0).
"""

from typing import Any, List, Optional

from confluid import configurable
from loggair import get_logger
from matrainer.uri import ModelRef, hf_model_ref

logger = get_logger(__name__)

__all__ = ["HuggingFaceDetectionModel"]


@configurable(task="detection", role="model", lazy=True, framework="huggingface")
class HuggingFaceDetectionModel:
    """A Hub object-detection checkpoint with a head sized to your dataset.

    Args:
        model_name: A Hub model id, e.g. ``"ustc-community/dfine-small-coco"`` or
            ``"PekingU/rtdetr_v2_r18vd"``. Sentinel ``""`` is invalid but permitted so the class
            is zero-arg constructible; it is rejected at build time with a clear error.
        num_classes: Width of the class head (K foreground classes, 0-indexed labels). Sentinel
            ``0`` likewise — a detector backend injects the resolved count via
            ``flow(model, num_classes=N)`` at run time, so it is not written in YAML.
        pretrained: Load the checkpoint's weights. ``False`` builds the same architecture from
            its config with fresh weights, which keeps a first run offline and reproducible.
        trust_remote_code: Allow a Hub repo to execute its own modelling code. Off by default
            because it runs code from the internet; turn it on only for a repo you trust.
    """

    def __init__(
        self,
        model_name: str = "",
        num_classes: int = 0,
        pretrained: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        # Lazy: store only. No Hub call, no download — the workspace zero-arg mandate.
        self.model_name = model_name
        self.num_classes = int(num_classes)
        self.pretrained = pretrained
        self.trust_remote_code = trust_remote_code
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    def solidify(self) -> Any:
        """Build (once) and return the live ``transformers`` model."""
        if self._model is not None:
            return self._model
        if not self.model_name:
            raise ValueError("HuggingFaceDetectionModel.model_name is required (got empty).")
        if self.num_classes < 1:
            raise ValueError(
                f"HuggingFaceDetectionModel.num_classes must be >= 1 to build (got {self.num_classes}) — "
                "set it explicitly, or let the detector backend derive it."
            )
        from transformers import AutoConfig, AutoModelForObjectDetection

        if self.pretrained:
            # `ignore_mismatched_sizes` is what makes fine-tuning possible at all: the checkpoint's
            # head is sized for ITS dataset, and without this the load raises on the mismatch
            # instead of reinitialising the head.
            self._model = AutoModelForObjectDetection.from_pretrained(
                self.model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
                trust_remote_code=self.trust_remote_code,
            )
        else:
            config = AutoConfig.from_pretrained(
                self.model_name, num_labels=self.num_classes, trust_remote_code=self.trust_remote_code
            )
            self._model = AutoModelForObjectDetection.from_config(config)
        logger.info(f"Built {self.model_name} with a {self.num_classes}-class head (pretrained={self.pretrained}).")
        return self._model

    @property
    def model(self) -> Any:
        """The live ``transformers`` model, built on first access."""
        return self.solidify()

    @property
    def image_processor(self) -> Any:
        """The checkpoint's OWN image processor (resize / pad / normalize / box decode), built once.

        Shaping belongs to the checkpoint, not to raidar: the backend's collator calls this with
        images + COCO-style annotations, and its eval path decodes predictions back into
        pixel-space boxes through ``post_process_object_detection``.
        """
        if self._processor is None:
            if not self.model_name:
                raise ValueError("HuggingFaceDetectionModel.model_name is required (got empty).")
            from transformers import AutoImageProcessor

            self._processor = AutoImageProcessor.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
        return self._processor

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the live model, so the wrapper is callable wherever the model is."""
        return self.model(*args, **kwargs)

    @property
    def model_refs(self) -> List[ModelRef]:
        """The weights this wrapper loads, for the run manifest's ``models`` section.

        Reads STORED CONFIG ONLY — never builds, downloads or opens anything, because the
        property is asked of a wrapper that may never run. An EMPTY list when
        ``pretrained=False``: random weights referenced no artifact, and the architecture is
        already recorded in the run's params, so naming one would name something nobody loaded.
        """
        if not self.model_name or not self.pretrained:
            return []
        return [hf_model_ref(self.model_name, role="pretrained")]
