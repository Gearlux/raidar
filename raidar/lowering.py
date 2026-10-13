"""The compiler's view of raidar's RF-DETR predictor: the library's own inference path as torch, fixed-size.

Installed by the ``[compile]`` extra and listed under the entry-point group ``torpaido.lowerings``. The lowering
mirrors ``RFDETR.predict`` step for step — the bilinear resize to the model's resolution without antialiasing,
the ImageNet normalization, the network, the library's own top-k selection and box scaling — and replaces the
one thing an exported program cannot do, the score filter that changes the row count, with a ``keep`` mask over
the fixed ``num_select`` rows. The ``any_class`` score mode is the predictor's own logit rewrite, applied before
the selection exactly as the predictor applies it.

The contract is the one a detector op's lowering states: ``[B, 3, H, W]`` pictures in ``[0, 1]`` in; ``boxes``
``[B, K, 4]`` xyxy in the picture's pixels, ``scores`` ``[B, K]``, ``labels`` ``[B, K]``, ``keep`` ``[B, K]`` and
``class_count`` out.
"""

from typing import Any, Dict

import torch
from torch import Tensor
from torpaido.lowering import Context, lowers_model

from raidar.rfdetr import RFDETRPredictor, _any_class_logits

__all__ = ["lower_rfdetr_predictor"]


@lowers_model(RFDETRPredictor)
def lower_rfdetr_predictor(model: RFDETRPredictor, pictures: Tensor, ctx: Context) -> Dict[str, Any]:
    rf = model.solidify()
    network = ctx.module("network", rf.model.model)
    resolution = int(rf.model.resolution)
    batch = pictures.shape[0]
    resized = torch.nn.functional.interpolate(
        pictures, size=(resolution, resolution), mode="bilinear", align_corners=False, antialias=False
    )
    mean = ctx.buffer("mean", lambda: torch.tensor(list(rf.means), dtype=torch.float32).view(1, -1, 1, 1))
    std = ctx.buffer("std", lambda: torch.tensor(list(rf.stds), dtype=torch.float32).view(1, -1, 1, 1))
    outputs = network((resized - mean) / std)
    if isinstance(outputs, tuple):
        logits, pred_boxes = outputs[1], outputs[0]
    else:
        logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
    if model.score_mode == "any_class":
        counted = ctx.buffer("counted", lambda: model._counted_slots(rf))
        logits = _any_class_logits(logits, counted)
    postprocess = rf.model.postprocess
    postprocess = getattr(postprocess, "inner", postprocess)  # the predictor's any-class wrapper holds the library's
    scores, labels, selected = postprocess._select_topk(logits)
    height = torch.zeros(batch, dtype=torch.float32, device=pictures.device) + pictures.shape[-2]
    width = torch.zeros(batch, dtype=torch.float32, device=pictures.device) + pictures.shape[-1]
    target_sizes = torch.stack([height, width], dim=1)
    boxes = type(postprocess)._gather_and_scale_boxes(pred_boxes, selected, target_sizes)
    known = len(rf.class_names)
    keep = (scores > float(model.score_threshold)) & (labels < known)
    return {"boxes": boxes, "scores": scores, "labels": labels, "keep": keep, "class_count": known}
