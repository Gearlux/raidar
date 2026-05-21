"""Shared helpers for raidar's Ultralytics-backed model paths.

The Lightning-driven :class:`raidar.yolo26.YOLO26Model` and the native-Ultralytics
:class:`raidar.ultralytics_trainer.UltralyticsTrainer` both need the same
"build a ``DetectionModel`` with the right ``nc`` and optionally graft on the
COCO-pretrained .pt's shape-compatible weights" routine. Centralizing it here
keeps the two paths in lockstep and avoids the silent drift that comes with
copy-pasted scaffolding.
"""

from types import SimpleNamespace
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn


def build_yolo26(
    variant: Literal["n", "s", "m", "l", "x"],
    num_classes: int,
    pretrained_detector: bool,
    agnostic_nms: bool = False,
    end2end: Optional[bool] = None,
) -> nn.Module:
    """Instantiate a YOLO26 ``DetectionModel`` with ``num_classes`` outputs.

    Building from the YAML (``yolo26{variant}.yaml``) gives us full control
    over ``nc``; if ``pretrained_detector=True`` we then load the COCO-trained
    .pt and copy every shape-compatible parameter into the new model. The
    head's classification convolutions don't match (the .pt was trained for
    ``nc=80``) and are silently skipped, leaving the backbone, FPN, and
    regression head fully pretrained while the classifier starts random.

    Args:
        end2end: If ``None`` (default), leave the head's NMS-free toggle as
            the YAML specifies (YOLO26 ships ``end2end=True``). Set ``False``
            to disable the ``one2one`` branch — useful on Apple MPS, where
            the TAL assigner inside ``E2EDetectLoss`` hits a bool-indexing
            regression on expanded views and crashes mid-batch. The
            ``one2many`` branch still trains and predicts; you trade
            NMS-free inference for a working MPS run.
    """
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG_DICT
    from ultralytics.utils.downloads import attempt_download_asset

    detector = DetectionModel(f"yolo26{variant}.yaml", nc=num_classes, ch=3)

    if pretrained_detector:
        weights_path = attempt_download_asset(f"yolo26{variant}.pt")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        src_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        src_state = src_model.float().state_dict()
        dst_state = detector.state_dict()
        for k, v in src_state.items():
            if k in dst_state and v.shape == dst_state[k].shape:
                dst_state[k] = v
        detector.load_state_dict(dst_state)

    # The criterion accesses self.hyp.{box, cls, dfl, ...} via attribute lookup.
    # A YAML-built DetectionModel may have no .args at all; a .pt-loaded one
    # has .args as a dict from the task block. Either way, normalize to a
    # namespace populated from DEFAULT_CFG_DICT + whatever was present.
    existing = getattr(detector, "args", None)
    if existing is None:
        raw_args: Dict[str, object] = {}
    elif isinstance(existing, dict):
        raw_args = existing
    else:
        raw_args = vars(existing)
    merged: Dict[str, object] = {**DEFAULT_CFG_DICT, **raw_args}
    # nn.Module.__setattr__ only types Tensor / Module / Parameter assignments;
    # bypass it via object.__setattr__ to attach the namespace cleanly.
    namespace = SimpleNamespace(**merged)
    object.__setattr__(detector, "args", namespace)
    object.__setattr__(detector, "hyp", namespace)

    detector.set_head_attr(agnostic_nms=agnostic_nms)
    if end2end is not None:
        # The Detect head has end2end as a property with a setter; routing it
        # through set_head_attr keeps the head/model coupling consistent with
        # how agnostic_nms is wired (and emits a clean LOGGER warning if a
        # future ultralytics version renames it).
        detector.set_head_attr(end2end=end2end)

    return detector


__all__ = ["build_yolo26"]
