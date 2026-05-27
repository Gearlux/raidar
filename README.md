# Raidar

Object-detection models and adapters for the [Marainer](https://github.com/Gearlux/marainer) training framework.

Raidar provides thin, Confluid-configurable wrappers around two detector
families — torchvision (Faster R-CNN) and Roboflow's RF-DETR — so they all
plug into the **unchanged** base `marainer.trainer.LightningTrainer` step
flow. No subclassing the Trainer, no overridden step functions, no
per-detector Lightning module.

The ultralytics-backed detectors (YOLO26, RT-DETR) plus the native
ultralytics trainer/evaluator live in the
[raidar-ultralytics](https://github.com/Gearlux/raidar-ultralytics) sibling
package — that's where ``ultralytics`` is pulled in as a dependency. Both
packages share the adapters in ``raidar.detection``:

Marainer's standard
`preds = self(batch.input); loss = self.loss_fn(preds, batch.target)` fits
detection because raidar adapts the pieces on either side:

| Adapter | Role |
|---|---|
| `DetectionBatchInput` | Carries images + targets through `batch.input` (torchvision detectors need both during training) and exposes `.shape` so Lightning's batch-size logging still works. |
| `DetectionStepOutput` | Unified `{losses, predictions}` dict every raidar model returns from `forward`. |
| `TorchvisionDetectionLoss` | Sums the loss dict from the step output — a real scalar in both train and val, never a `tensor(0.0)` placeholder. |
| `DetectionMetricAdapter` | Unwraps `predictions` before forwarding to any torchmetrics detection metric (e.g. `MeanAveragePrecision`). |
| `detection_collate_fn` | Preserves variable-length per-image targets as lists. |

Per-image predictions use the task-agnostic `marainer.outputs.DetectionOutput`
schema, re-exported from `raidar.detection` for convenience.

## Installation

```bash
pip install git+https://github.com/Gearlux/raidar.git@main
```

For the ultralytics-backed detectors (YOLO26 / RT-DETR / native ultralytics
training):

```bash
pip install git+https://github.com/Gearlux/raidar-ultralytics.git@main
```

### Development

```bash
git clone https://github.com/Gearlux/raidar.git
cd raidar
pip install -e ".[dev]"
```

## Quick Start

### YAML wiring (Lightning-driven)

```yaml
model: !class:raidar.faster_rcnn.FasterRCNNModel
  num_classes: 2          # 0 = background, 1 = foreground
  variant: v2
  pretrained_detector: true

loss_fn: !class:raidar.detection.TorchvisionDetectionLoss
  weights:
    loss_classifier: 1.0
    loss_box_reg: 1.0

val_metrics: !class:raidar.detection.DetectionMetricAdapter
  inner: !class:torchmetrics.detection.MeanAveragePrecision
    iou_type: bbox

train_loader: !class:torch.utils.data.DataLoader
  batch_size: 4
  shuffle: true
  collate_fn: !ref:raidar.detection.detection_collate_fn

trainer: !class:marainer.trainer.LightningTrainer
  experiment_name: my_detection
  run_name: faster_rcnn_v2
  model: !ref:model
  loss_fn: !ref:loss_fn
  val_metrics: !ref:val_metrics
  ...
```

Run:

```bash
marainer train path/to/config.yaml
```

### Detector catalogue

| Module | Class | Notes |
|---|---|---|
| `raidar.faster_rcnn` | `FasterRCNNModel` | torchvision Faster R-CNN. `variant: v1` / `v2`, optional pretrained backbone + detector, BN-frozen dual-forward in eval mode so val loss is directly comparable to train loss. |
| `raidar.rfdetr` | `RFDETRModel` | Roboflow's RF-DETR family (native torchvision-style forward). |
| `raidar.rfdetr_marainer` | `RFDETRMarainerModel` | Marainer-generic RF-DETR wrapper. |

The Ultralytics-backed YOLO26 + RT-DETR wrappers — plus the native
ultralytics trainer/evaluator (mosaic + albumentations + Ultralytics
`LOGGER`) — live in
[raidar-ultralytics](https://github.com/Gearlux/raidar-ultralytics).

## Manifests

Models, losses, and metrics live under `manifests/` so navigaitor's
scanner can discover them:

```
raidar/manifests/
├── losses/
├── metrics/
└── models/
    ├── faster_rcnn.yaml
    ├── rfdetr.yaml
    └── rfdetr_marainer.yaml
```

## Examples

End-to-end YAMLs in `examples/`:

- `lightning_train_disk.yaml` — Faster R-CNN on a disk dataset.
- `lightning_train_native.yaml` — direct torchvision Faster R-CNN.

## License

MIT.
