"""``MlxFcos`` — the shared tiny-FCOS spec in ``mlx.nn``, the MLX backend's model.

Like the Flax one this wraps no zoo, and for a comparable reason: the only real MLX vision zoo
(``mlx-image``) hard-pins a ``huggingface_hub`` this workspace cannot take, and ships
classification backbones anyway. The MLX backend implements :mod:`raidar.fcos.spec` natively.

MLX is **Apple-silicon only** — a property of the framework; the extra carries the platform
marker, and this module is only importable where it holds.

Three MLX facts shape the code (all measured in the segmentation project):

* **Channels-last is native.** ``mlx.nn.Conv2d`` takes ``NHWC`` directly, so this consumes the
  same batches as the Keras / TensorFlow / JAX implementations with no transpose.
* **Arrays are lazy.** Nothing computes until ``mx.eval`` — the training loop's concern, not
  this module's.
* **``parameters()`` skips underscore-prefixed attributes** (``valid_parameter_filter`` ends in
  ``not key.startswith("_")``). The lazily built network is therefore held in the PUBLIC
  attribute ``net`` — as ``_model`` the whole network would be invisible to the optimizer, and
  the loop would run forever applying gradients to nothing.
"""

from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from confluid import configurable
from loggair import get_logger

from raidar.fcos.spec import BACKBONE_WIDTHS, HEAD_WIDTH, INPUT_SHAPE, KERNEL_SIZE, grid_size, stride

logger = get_logger(__name__)

__all__ = ["MlxFcos"]


class _ConvBnRelu(nn.Module):
    """``conv -> BN -> ReLU`` — the spec's repeated unit, in ``mlx.nn``."""

    def __init__(self, in_channels: int, width: int, conv_stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, width, kernel_size=KERNEL_SIZE, stride=conv_stride, padding=KERNEL_SIZE // 2)
        self.norm = nn.BatchNorm(width)

    def __call__(self, x: Any) -> Any:
        return nn.relu(self.norm(self.conv(x)))


class _Fcos(nn.Module):
    """The spec's layer graph: strided backbone → shared trunk → the three 1x1 heads."""

    def __init__(self, num_classes: int, in_channels: int, widths: Tuple[int, ...], head_width: int) -> None:
        super().__init__()
        stages: List[nn.Module] = []
        channels = in_channels
        for width in widths:
            stages.append(_ConvBnRelu(channels, width))
            stages.append(_ConvBnRelu(width, width, conv_stride=2))  # halves H and W
            channels = width
        self.stages = stages
        self.trunk = _ConvBnRelu(channels, head_width)
        # The three 1x1 heads: raw outputs, NO activation — the spec's decode/loss own the
        # sigmoid/exp transforms.
        self.cls_head = nn.Conv2d(head_width, num_classes, kernel_size=1)
        self.box_head = nn.Conv2d(head_width, 4, kernel_size=1)
        self.ctr_head = nn.Conv2d(head_width, 1, kernel_size=1)

    def __call__(self, x: Any) -> Any:
        for stage in self.stages:
            x = stage(x)
        x = self.trunk(x)
        return self.cls_head(x), self.box_head(x), self.ctr_head(x)


@configurable(task="detection", role="model", lazy=True, framework="mlx")
class MlxFcos(nn.Module):
    """The shared tiny-FCOS spec (:mod:`raidar.fcos.spec`), implemented in ``mlx.nn``.

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
        seed: Seed for ``mx.random`` before the build, so a config reproduces its weights.
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
        super().__init__()
        # Partial: store only. No parameter allocation — the workspace zero-arg mandate.
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.widths = tuple(widths)
        self.head_width = head_width
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.seed = seed
        # PUBLIC on purpose — see the module docstring's `parameters()` fact.
        self.net: Optional[nn.Module] = None

    def solidify(self) -> Any:
        """Build (once) and return the live network — confluid's ``flow()``-time hook."""
        if self.net is None:
            if not self.num_classes:
                raise ValueError(
                    "MlxFcos.num_classes is not set — a detector flows the resolved count in "
                    "at run time, or set it explicitly."
                )
            grid_size(self.input_shape, self.widths)  # type: ignore[arg-type]  # validate divisibility EARLY
            mx.random.seed(self.seed)
            self.net = _Fcos(
                num_classes=self.num_classes,
                in_channels=int(self.input_shape[-1]),
                widths=self.widths,
                head_width=self.head_width,
            )
            logger.info(f"Built the shared FCOS spec in mlx.nn ({self.num_classes} classes, widths={self.widths}).")
        return self.net

    @property
    def model(self) -> Any:
        """The live network, built on first access."""
        return self.solidify()

    @property
    def spec_stride(self) -> int:
        """The raw-to-distance scale the loss/decode need — derived from the configured widths."""
        return stride(self.widths)

    def __call__(self, x: Any) -> Any:
        """The three raw head maps from an ``[N, H, W, C]`` batch — channels LAST.

        No ``training=`` argument: MLX's train/eval is module STATE (``.train(bool)``), set by
        the loop — a default here would silently flip the network on every forward.
        """
        return self.model(x)

    @property
    def parameter_count(self) -> int:
        """Trainable parameters — the number :mod:`raidar.fcos.spec` predicts.

        MLX keeps BatchNorm's running statistics in the SAME parameter tree as the weights, so
        the exclusion is a NAME filter (``running_mean`` / ``running_var``) — without it the
        cross-implementation match breaks.
        """
        from typing import cast

        from mlx.utils import tree_flatten

        entries = cast(List[Tuple[str, Any]], tree_flatten(self.model.parameters()))
        return int(sum(value.size for key, value in entries if "running_mean" not in key and "running_var" not in key))
