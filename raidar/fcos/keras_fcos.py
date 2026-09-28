"""``KerasFcos`` — the shared tiny-FCOS spec in ``keras.layers``, plus the spec loss in ``keras.ops``.

The Keras entry in the comparison :mod:`raidar.fcos.spec` describes. It exists because **every
detection zoo raidar wraps is torch-only** — without this the Keras backend would have no
detection model and the cross-framework comparison would stop at torch.

Being written in ``keras.layers``/``keras.ops`` it runs on whichever engine Keras is bound to,
so BOTH the ``keras`` and (later) ``tensorflow`` backends wire it. **Channels-LAST**: ``call``
takes ``[N, H, W, C]`` and returns the three raw head maps ``(cls, box, ctr)`` as a list —
``[N, rows, cols, num_classes]`` / ``[N, rows, cols, 4]`` / ``[N, rows, cols, 1]``.

**The loss functions live beside the model** because they are the spec's differentiable half in
this API's vocabulary (the torch twin writes them inside ``TorchFcos._loss``): three per-output
functions matching the model's three outputs, each deriving the positive-cell mask from its OWN
dense target (a positive cell has a nonzero class row / strictly positive distances / strictly
positive centerness — the same cells, three spellings) and normalizing by that image's positive
count. Written in ``keras.ops`` so they trace on any engine.
"""

from typing import Any, Callable, Optional, Tuple

from confluid import configurable
from loggair import get_logger

from raidar._keras import keras, keras_backend
from raidar.fcos.spec import BACKBONE_WIDTHS, HEAD_WIDTH, INPUT_SHAPE, KERNEL_SIZE, grid_size, stride

logger = get_logger(__name__)

__all__ = ["KerasFcos", "fcos_box_loss", "fcos_cls_loss", "fcos_ctr_loss"]


# --------------------------------------------------------------------------- #
# The spec loss, per output — keras.ops, engine-agnostic
# --------------------------------------------------------------------------- #
def fcos_cls_loss(y_true: Any, y_pred: Any) -> Any:
    """Sigmoid focal loss (α=0.25, γ=2) over ALL cells, per image / its positive count."""
    ops = keras.ops
    p = ops.sigmoid(y_pred)
    ce = ops.binary_crossentropy(y_true, y_pred, from_logits=True)
    p_t = y_true * p + (1.0 - y_true) * (1.0 - p)
    alpha_t = y_true * 0.25 + (1.0 - y_true) * 0.75
    focal = alpha_t * ops.power(1.0 - p_t, 2.0) * ce
    per_image = ops.sum(focal, axis=(1, 2, 3))
    num_fg = ops.sum(ops.max(y_true, axis=-1), axis=(1, 2))
    return per_image / ops.maximum(num_fg, 1.0)


def fcos_box_loss(box_stride: float) -> Callable[[Any, Any], Any]:
    """``1 - IoU`` of predicted vs target ``(l, t, r, b)`` on the positive cells.

    A factory because the raw-to-distance rule is ``exp(raw) * stride`` and the stride follows
    from the model's widths — the backend closes it over the configured model.
    """

    def loss(y_true: Any, y_pred: Any) -> Any:
        ops = keras.ops
        fg = ops.cast(ops.min(y_true, axis=-1) > 0.0, y_pred.dtype)  # [B, rows, cols]
        predicted = ops.exp(y_pred) * box_stride
        lr = ops.minimum(predicted[..., 0], y_true[..., 0]) + ops.minimum(predicted[..., 2], y_true[..., 2])
        tb = ops.minimum(predicted[..., 1], y_true[..., 1]) + ops.minimum(predicted[..., 3], y_true[..., 3])
        intersection = ops.maximum(lr, 0.0) * ops.maximum(tb, 0.0)
        area_p = (predicted[..., 0] + predicted[..., 2]) * (predicted[..., 1] + predicted[..., 3])
        area_t = (y_true[..., 0] + y_true[..., 2]) * (y_true[..., 1] + y_true[..., 3])
        iou = intersection / ops.maximum(area_p + area_t - intersection, 1e-9)
        per_image = ops.sum((1.0 - iou) * fg, axis=(1, 2))
        num_fg = ops.sum(fg, axis=(1, 2))
        return per_image / ops.maximum(num_fg, 1.0)

    return loss


def fcos_ctr_loss(y_true: Any, y_pred: Any) -> Any:
    """Binary cross-entropy on the centerness logit, positive cells only.

    A positive cell's centerness target is strictly positive (its distances are), so the target
    itself carries the mask.
    """
    ops = keras.ops
    logit = ops.squeeze(y_pred, axis=-1)
    fg = ops.cast(y_true > 0.0, logit.dtype)
    bce = ops.binary_crossentropy(y_true, logit, from_logits=True)
    per_image = ops.sum(bce * fg, axis=(1, 2))
    num_fg = ops.sum(fg, axis=(1, 2))
    return per_image / ops.maximum(num_fg, 1.0)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
def _conv_bn_relu(x: Any, width: int, conv_stride: int = 1) -> Any:
    """``conv -> BN -> ReLU`` — the spec's repeated unit, in Keras's vocabulary."""
    x = keras.layers.Conv2D(width, KERNEL_SIZE, strides=conv_stride, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.ReLU()(x)


@configurable(task="detection", role="model", lazy=True, framework="keras")
class KerasFcos(keras.Model):
    """The shared tiny-FCOS spec (:mod:`raidar.fcos.spec`), implemented in ``keras.layers``.

    Args:
        num_classes: Width of the per-cell class head. Sentinel ``0`` is invalid but permitted so
            the class is zero-arg constructible; a detector flows the resolved count in at run time.
        input_shape: ``(H, W, C)`` — channels LAST, which is also this engine's tensor layout.
            Unlike a fully-convolutional segmenter this model PINS the spatial dims: dense FCOS
            targets need one fixed grid, so accepting other sizes would only defer the error.
        widths: Channels per backbone stage. Changing it breaks comparability with the other
            implementations — see :mod:`raidar.fcos.spec`.
        head_width: Channels of the shared head trunk.
        score_threshold: Eval-time decode floor — read by the backend's numpy decode.
        iou_threshold: Eval-time per-class NMS overlap threshold.
        max_detections: Eval-time cap on detections per image.
        seed: Seed applied before the build, so a config reproduces its weights — the same knob
            its four sibling implementations carry, and the reason this engine was the one whose
            final loss moved between identical runs.
    """

    def __init__(
        self,
        num_classes: int = 0,
        input_shape: Tuple[int, int, int] = INPUT_SHAPE,
        widths: Tuple[int, ...] = BACKBONE_WIDTHS,
        head_width: int = HEAD_WIDTH,
        score_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
        seed: int = 0,
    ) -> None:
        # Partial: store only, and `super().__init__()` FIRST — a `keras.Layer` refuses attribute
        # assignment before it.
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.head_width = head_width
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.seed = int(seed)
        self._model: Optional[Any] = None

    def solidify(self) -> Any:
        """Build (once) and return the live ``keras.Model`` — confluid's ``flow()``-time hook."""
        if self._model is not None:
            return self._model
        if not self.num_classes:
            raise ValueError(
                "KerasFcos.num_classes is not set — a detector flows the resolved count in at "
                "run time, or set it explicitly."
            )
        grid_size(self.input_shape, self.widths)  # type: ignore[arg-type]  # validate divisibility EARLY
        # Seeded like every sibling implementation, and for the same reason: an unseeded run is
        # not comparable to itself, so a refactor cannot be checked by re-running it. Keras owns
        # no per-layer seed argument the way raw TF does, so this is the global call its own
        # docs prescribe — it covers the initializers of every layer built below.
        keras.utils.set_random_seed(self.seed)
        inputs = keras.Input(shape=(int(self.input_shape[0]), int(self.input_shape[1]), int(self.input_shape[2])))
        x = inputs
        for width in self.widths:
            x = _conv_bn_relu(x, width)
            x = _conv_bn_relu(x, width, conv_stride=2)  # each stage halves H and W
        x = _conv_bn_relu(x, self.head_width)
        # The three 1x1 heads: raw outputs, NO activation — the spec's decode/loss own the
        # sigmoid/exp transforms.
        cls_map = keras.layers.Conv2D(self.num_classes, 1, name="cls")(x)
        box_map = keras.layers.Conv2D(4, 1, name="box")(x)
        ctr_map = keras.layers.Conv2D(1, 1, name="ctr")(x)
        self._model = keras.Model(inputs, [cls_map, box_map, ctr_map], name="raidar_fcos")
        logger.info(
            f"Built the shared FCOS spec in keras ({keras_backend()} backend, "
            f"{self.num_classes} classes, widths={self.widths})."
        )
        return self._model

    @property
    def model(self) -> Any:
        """The live ``keras.Model``, built on first access."""
        return self.solidify()

    @property
    def spec_stride(self) -> int:
        """The raw-to-distance scale the loss/decode need — derived from the configured widths."""
        return stride(self.widths)

    def build(self, input_shape: Any = None) -> None:
        """Keras's build hook — the ONE place a subclassed model may create state."""
        self.solidify()
        super().build(input_shape)

    def call(self, x: Any, training: bool = False) -> Any:
        """The three raw head maps from an ``[N, H, W, C]`` batch — channels LAST."""
        return self.model(x, training=training)

    @property
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`raidar.fcos.spec` predicts.

        BatchNorm's moving mean/variance are Keras NON-trainable weights, so reading
        ``trainable_weights`` excludes them by construction — the same exclusion the spec makes.
        """
        return int(sum(int(w.size if hasattr(w, "size") else w.numpy().size) for w in self.model.trainable_weights))
