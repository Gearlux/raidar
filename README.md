# Raidar

Object-detection models and adapters for the [Marainer](https://github.com/Gearlux/marainer) training framework.

Raidar provides thin, Confluid-configurable wrappers around four detector
families — torchvision (Faster R-CNN), Ultralytics (YOLO26), RT-DETR, and
RF-DETR — so they all plug into the **unchanged** base
`marainer.trainer.LightningTrainer` step flow. No subclassing the Trainer,
no overridden step functions, no per-detector Lightning module.

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
| `raidar.yolo26` | `YOLO26Model` | Ultralytics YOLO26 wrapped for the Lightning flow. Use this when you want the unified Trainer surface. |
| `raidar.rtdetr` | `RTDETRModel` | Real-time DETR (Ultralytics-hosted weights). |
| `raidar.rfdetr` | `RFDETRModel` | Roboflow's RF-DETR family. |

## Ultralytics-native training

Lightning's standard step flow bypasses Ultralytics' built-in augmentation
pipeline (mosaic, mixup, cutmix, albumentations) and its `LOGGER`. When you
want those, raidar exposes a second pair of CLI verbs that delegate to
Ultralytics' native trainer:

```bash
marainer ultralytics-train     raidar/examples/ultralytics_train_disk.yaml
marainer ultralytics-evaluate  raidar/examples/ultralytics_evaluate_flux.yaml
```

These verbs are registered onto the marainer CLI via
`[project.entry-points."marainer.commands"]` (see `raidar.cli_plugin`).

### Disk mode

`UltralyticsTrainer` takes the same `data` path you would pass to
`yolo train data=...` — a YOLO `data.yaml` with `path:`, `train:`, `val:`,
`names:`, `nc:`:

```yaml
trainer: !class:raidar.ultralytics_runner.UltralyticsTrainer
  data: ./datasets/coco8/coco8.yaml
  variant: n
  pretrained: true
  imgsz: 640
  epochs: 100
  batch: 16
  experiment_name: yolo26n_coco8_disk
  metrics_output_path: ./runs/yolo26n_coco8_disk/metrics.json
```

### Flux mode

To drive the Ultralytics trainer over in-memory `dataflux.Flux` objects —
the same Confluid-wired Flux pipeline the Lightning flow uses, with no
on-disk YOLO data.yaml — pass `data: { train: !ref:train_set, val: !ref:val_set, names: ..., nc: ... }`:

```yaml
train_set: !class:dataflux.core.Flux()
  source: []   # e.g. RegionsJsonSource
  ops: []      # IQ → spectrogram → detection-target op chain

val_set: !class:dataflux.core.Flux()
  source: []
  ops: []

trainer: !class:raidar.ultralytics_runner.UltralyticsTrainer
  data:
    train: !ref:train_set
    val: !ref:val_set
    names: { 0: drone-signal }
    nc: 1
  variant: n
  pretrained: true
  imgsz: 640
  epochs: 50
  batch: 16
  experiment_name: yolo26n_rfuav_flux
  metrics_output_path: ./runs/yolo26n_rfuav_flux/metrics.json
```

Flux mode swaps the trainer for `FluxDetectionTrainer`, whose
`build_dataset` returns a `FluxYOLODataset(YOLODataset)` reading samples
from the supplied `Flux`. Mosaic, albumentations, and the Ultralytics
`LOGGER` all stay live — only the on-disk `YOLODataset` is replaced.

### Metrics sidecar

Both `UltralyticsTrainer` and `UltralyticsEvaluator` write a
`metrics.json` sidecar at `metrics_output_path`, matching
`marainer.evaluator.LightningEvaluator.metrics_sidecar_path`. This lets
[Navigaitor](https://github.com/Gearlux/navigaitor)'s `read_run_metrics`
tool read metrics uniformly across Lightning- and Ultralytics-driven runs.

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
    ├── rtdetr.yaml
    └── yolo26.yaml
```

## Examples

End-to-end YAMLs in `examples/`:

- `ultralytics_train_disk.yaml` — YOLO26 + COCO8 on disk.
- `ultralytics_train_flux.yaml` — YOLO26 + in-memory DataFlux pipeline.
- `ultralytics_evaluate_flux.yaml` — evaluate a trained checkpoint over a Flux.

For a real RFUAV drone-signal detection pipeline, see
`waivefront-rfuav/config/train_yolo26.yaml` (and friends), which wires
`RegionsJsonSource` → IQ → spectrogram → detection-target op chain →
`YOLO26Model` end-to-end.

## License

MIT.
