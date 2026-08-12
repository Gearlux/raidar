"""``FlaxFcos`` — the shared tiny-FCOS spec in ``flax.nnx``, the JAX backend's model.

The JAX entry in the comparison :mod:`raidar.fcos.spec` describes, and it wraps no zoo — which
here is not a design choice but the only option: there is no Flax detection zoo to wrap
(``transformers`` dropped Flax entirely in v5). So the JAX backend implements the spec natively,
which is exactly why the spec exists.

It IS an ``nnx.Module``, the same decision as its siblings being their framework's own model
type: the detector drives whatever is wired into its ``model`` slot without asking what it got,
and ``nnx.state`` / ``nnx.Optimizer`` reach the parameters straight through the wrapper.

**Channels-LAST** (``[N, H, W, C]``), like every implementation except the torch one.

One Flax fact shapes the lazy build (measured in the segmentation project): an attribute first
assigned ``None`` in ``__init__`` becomes STATIC in nnx's pytree, and assigning a Module to it
later raises — ``nnx.data`` is the fix flax itself names, and it is what keeps zero-arg
construction compatible with building at ``flow()`` time.
"""

from typing import Any, Tuple

from confluid import configurable
from flax import nnx
from loggair import get_logger

from raidar.fcos.spec import BACKBONE_WIDTHS, HEAD_WIDTH, INPUT_SHAPE, KERNEL_SIZE, grid_size, stride

logger = get_logger(__name__)

__all__ = ["FlaxFcos"]


class _ConvBnRelu(nnx.Module):
    """``conv -> BN -> ReLU`` — the spec's repeated unit, in ``flax.nnx``."""

    def __init__(self, in_channels: int, width: int, rngs: Any, conv_stride: int = 1) -> None:
        self.conv = nnx.Conv(
            in_channels,
            width,
            kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
            strides=(conv_stride, conv_stride),
            rngs=rngs,
        )
        self.norm = nnx.BatchNorm(width, rngs=rngs)

    def __call__(self, x: Any, training: bool = False) -> Any:
        # `use_running_average` is the INVERSE of "training": in eval we consume the running
        # statistics rather than the batch's, exactly as keras and torch do.
        return nnx.relu(self.norm(self.conv(x), use_running_average=not training))


class _Fcos(nnx.Module):
    """The spec's layer graph: strided backbone → shared trunk → the three 1x1 heads."""

    def __init__(self, num_classes: int, in_channels: int, widths: Tuple[int, ...], head_width: int, seed: int) -> None:
        rngs = nnx.Rngs(seed)
        self.stages: Any = nnx.List([])
        channels = in_channels
        for width in widths:
            self.stages.append(_ConvBnRelu(channels, width, rngs))
            self.stages.append(_ConvBnRelu(width, width, rngs, conv_stride=2))  # halves H and W
            channels = width
        self.trunk = _ConvBnRelu(channels, head_width, rngs)
        # The three 1x1 heads: raw outputs, NO activation — the spec's decode/loss own the
        # sigmoid/exp transforms.
        self.cls_head = nnx.Conv(head_width, num_classes, kernel_size=(1, 1), rngs=rngs)
        self.box_head = nnx.Conv(head_width, 4, kernel_size=(1, 1), rngs=rngs)
        self.ctr_head = nnx.Conv(head_width, 1, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: Any, training: bool = False) -> Any:
        for stage in self.stages:
            x = stage(x, training=training)
        x = self.trunk(x, training=training)
        return self.cls_head(x), self.box_head(x), self.ctr_head(x)


@configurable(task="detection", role="model", lazy=True, framework="jax")
class FlaxFcos(nnx.Module):
    """The shared tiny-FCOS spec (:mod:`raidar.fcos.spec`), implemented in ``flax.nnx``.

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
        seed: Seed for the ``nnx.Rngs`` the initializers draw from — explicit randomness is the
            JAX story, and this knob is what makes a config reproduce its weights.
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
        # Partial: store only. No parameter allocation — the workspace zero-arg mandate.
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.head_width = head_width
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.seed = seed
        # `nnx.data`: a bare `None` would become a STATIC pytree attribute, and assigning the
        # built network to it later RAISES (measured in the segmentation project).
        self._model: Any = nnx.data(None)

    def solidify(self) -> Any:
        """Build (once) and return the live network — confluid's ``flow()``-time hook."""
        if self._model is None:
            if not self.num_classes:
                raise ValueError(
                    "FlaxFcos.num_classes is not set — a detector flows the resolved count in "
                    "at run time, or set it explicitly."
                )
            grid_size(self.input_shape, self.widths)  # type: ignore[arg-type]  # validate divisibility EARLY
            self._model = nnx.data(
                _Fcos(
                    num_classes=self.num_classes,
                    in_channels=int(self.input_shape[-1]),
                    widths=self.widths,
                    head_width=self.head_width,
                    seed=self.seed,
                )
            )
            logger.info(f"Built the shared FCOS spec in flax.nnx ({self.num_classes} classes, widths={self.widths}).")
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
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`raidar.fcos.spec` predicts.

        BatchNorm's running statistics are ``nnx.BatchStat`` — a different variable TYPE in the
        same state tree — so counting ``nnx.Param`` only excludes them, the same exclusion the
        spec makes.
        """
        import jax

        params = nnx.state(self.model, nnx.Param)
        return int(sum(leaf.size for leaf in jax.tree.leaves(params)))
