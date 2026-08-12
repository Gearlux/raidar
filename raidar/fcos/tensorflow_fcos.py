"""``TensorFlowFcos`` — the shared tiny-FCOS spec in raw ``tf.nn`` ops.

The TensorFlow entry in the comparison :mod:`raidar.fcos.spec` describes, and it wraps nothing at
all. It is what makes that comparison include a real TensorFlow implementation rather than
Keras-on-TF standing in for one — :class:`raidar.fcos.keras_fcos.KerasFcos` is already the Keras
entry, and it runs on whichever engine Keras is bound to.

Tagged ``framework="keras"`` rather than ``"tensorflow"``, which looks wrong and is deliberate:
the framework axis names the API a class is **wired to**, and the TensorFlow backend — the only
trainer that can consume this — declares ``"keras"`` itself (its slots are Keras-shaped).

**It is a ``tf.Module``, not a ``keras.Model``** — ``trainable_variables`` and ``__call__`` are
everything a traced loop needs. BatchNorm is written out rather than borrowed (borrowing would
mean ``keras.layers.BatchNormalization`` and this would be a Keras model wearing a TensorFlow
name); its moving statistics are ``trainable=False`` variables — STATE, not parameters — which is
what lets the parameter count match the other implementations.
"""

from typing import Any, List, Optional, Tuple

from confluid import configurable
from loggair import get_logger

# `raidar._tensorflow` pins KERAS_BACKEND=tensorflow BEFORE importing keras; importing `tf`
# through it keeps that ordering in one place.
from raidar._tensorflow import tf
from raidar.fcos.spec import BACKBONE_WIDTHS, HEAD_WIDTH, INPUT_SHAPE, KERNEL_SIZE, grid_size, stride

logger = get_logger(__name__)

__all__ = ["TensorFlowFcos"]


def _glorot_uniform(shape: Tuple[int, ...], seed: int) -> Any:
    """Glorot-uniform weights from ``tf.random``, written out rather than taken from an initializer.

    **`tf.initializers` IS Keras** (since TF 2.16 `tf.keras` is Keras 3): in a process where
    Keras is bound to the TORCH engine — this workspace's default — it returns a torch tensor and
    `tf.Variable` dies converting it. So a raw-`tf.nn` module cannot reach for it, and these four
    lines are what "raw TensorFlow" costs. ``stateless_uniform`` so the seed is the whole
    argument; fans computed as Keras's ``compute_fans`` does, so the draw matches the other
    implementations' initializers.
    """
    receptive_field = 1
    for dim in shape[:-2]:
        receptive_field *= int(dim)
    fan_in, fan_out = int(shape[-2]) * receptive_field, int(shape[-1]) * receptive_field
    limit = (6.0 / float(fan_in + fan_out)) ** 0.5
    return tf.random.stateless_uniform(shape, seed=(seed, 0), minval=-limit, maxval=limit, dtype=tf.float32)


class _ConvBnRelu(tf.Module):
    """``conv -> BN -> ReLU`` — the spec's repeated unit, in raw ``tf.nn`` ops."""

    def __init__(self, in_channels: int, width: int, seed: int, conv_stride: int = 1, momentum: float = 0.99) -> None:
        super().__init__(name=f"conv_{in_channels}_{width}_s{conv_stride}")
        self.conv_stride = conv_stride
        self.momentum = momentum
        self.kernel = tf.Variable(_glorot_uniform((KERNEL_SIZE, KERNEL_SIZE, in_channels, width), seed), name="kernel")
        self.bias = tf.Variable(tf.zeros((width,)), name="bias")
        self.gamma = tf.Variable(tf.ones((width,)), name="gamma")
        self.beta = tf.Variable(tf.zeros((width,)), name="beta")
        self.moving_mean = tf.Variable(tf.zeros((width,)), trainable=False, name="moving_mean")
        self.moving_var = tf.Variable(tf.ones((width,)), trainable=False, name="moving_var")

    def __call__(self, x: Any, training: bool = False) -> Any:
        x = tf.nn.conv2d(x, self.kernel, strides=self.conv_stride, padding="SAME") + self.bias
        if training:
            # Batch statistics while training, folded into the moving estimates the eval path
            # uses — the update Keras/torch do for you inside their layer.
            mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
            self.moving_mean.assign(self.moving_mean * self.momentum + mean * (1.0 - self.momentum))
            self.moving_var.assign(self.moving_var * self.momentum + variance * (1.0 - self.momentum))
        else:
            mean, variance = self.moving_mean, self.moving_var
        x = tf.nn.batch_normalization(x, mean, variance, self.beta, self.gamma, variance_epsilon=1e-3)
        return tf.nn.relu(x)


class _Head(tf.Module):
    """A 1x1 conv head — raw output, NO activation (the spec's decode/loss own the transforms)."""

    def __init__(self, in_channels: int, out_channels: int, seed: int, name: str) -> None:
        super().__init__(name=name)
        self.kernel = tf.Variable(_glorot_uniform((1, 1, in_channels, out_channels), seed), name="kernel")
        self.bias = tf.Variable(tf.zeros((out_channels,)), name="bias")

    def __call__(self, x: Any) -> Any:
        return tf.nn.conv2d(x, self.kernel, strides=1, padding="SAME") + self.bias


class _Fcos(tf.Module):
    """The spec's layer graph: strided backbone → shared trunk → the three 1x1 heads."""

    def __init__(self, num_classes: int, in_channels: int, widths: Tuple[int, ...], head_width: int, seed: int) -> None:
        super().__init__(name="raidar_fcos")
        self.stages: List[_ConvBnRelu] = []
        channels = in_channels
        for index, width in enumerate(widths):
            self.stages.append(_ConvBnRelu(channels, width, seed=seed + index * 2))
            self.stages.append(_ConvBnRelu(width, width, seed=seed + index * 2 + 1, conv_stride=2))
            channels = width
        self.trunk = _ConvBnRelu(channels, head_width, seed=seed + len(widths) * 2)
        self.cls_head = _Head(head_width, num_classes, seed=seed + 101, name="cls")
        self.box_head = _Head(head_width, 4, seed=seed + 102, name="box")
        self.ctr_head = _Head(head_width, 1, seed=seed + 103, name="ctr")

    def __call__(self, x: Any, training: bool = False) -> Any:
        for stage in self.stages:
            x = stage(x, training=training)
        x = self.trunk(x, training=training)
        return self.cls_head(x), self.box_head(x), self.ctr_head(x)


@configurable(task="detection", role="model", lazy=True, framework="keras")
class TensorFlowFcos(tf.Module):
    """The shared tiny-FCOS spec (:mod:`raidar.fcos.spec`), implemented in raw ``tf.nn`` ops.

    Args:
        num_classes: Width of the per-cell class head. Sentinel ``0`` is invalid but permitted so
            the class is zero-arg constructible; a detector flows the resolved count in at run time.
        input_shape: ``(H, W, C)`` — channels LAST, which is also this engine's tensor layout.
        widths: Channels per backbone stage. Changing it breaks comparability with the other
            implementations — see :mod:`raidar.fcos.spec`.
        head_width: Channels of the shared head trunk.
        score_threshold: Eval-time decode floor — read by the backend's numpy decode.
        iou_threshold: Eval-time per-class NMS overlap threshold.
        max_detections: Eval-time cap on detections per image.
        seed: Seed for the variable initializers, so a config reproduces its weights.
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
        # `tf.Module.__init__` first — it sets up the name scope the variables register in.
        super().__init__(name="raidar_tensorflow_fcos")
        # Partial: store only. No variables allocated until `solidify()`.
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.head_width = head_width
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.seed = seed
        self._model: Optional[_Fcos] = None

    def solidify(self) -> Any:
        """Build (once) and return the live network — confluid's ``flow()``-time hook."""
        if self._model is None:
            if not self.num_classes:
                raise ValueError(
                    "TensorFlowFcos.num_classes is not set — a detector flows the resolved "
                    "count in at run time, or set it explicitly."
                )
            grid_size(self.input_shape, self.widths)  # type: ignore[arg-type]  # validate divisibility EARLY
            self._model = _Fcos(
                num_classes=self.num_classes,
                in_channels=int(self.input_shape[-1]),
                widths=self.widths,
                head_width=self.head_width,
                seed=self.seed,
            )
            logger.info(f"Built the shared FCOS spec in tf.nn ({self.num_classes} classes, widths={self.widths}).")
        return self._model

    @property
    def model(self) -> Any:
        """The live network, built on first access."""
        return self.solidify()

    @property
    def spec_stride(self) -> int:
        """The raw-to-distance scale the loss/decode need — derived from the configured widths."""
        return stride(self.widths)

    def __call__(self, x: Any, training: bool = False) -> Any:
        """The three raw head maps from an ``[N, H, W, C]`` batch — channels LAST."""
        return self.model(x, training=training)

    @property
    def trainable_variables(self) -> Any:
        """The variables the optimizer updates — building on ACCESS, so ``optimizer.build`` works."""
        return self.model.trainable_variables

    @property
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`raidar.fcos.spec` predicts.

        The moving statistics are ``trainable=False``, so they are excluded by construction.
        """
        return int(sum(int(tf.size(v)) for v in self.trainable_variables))
