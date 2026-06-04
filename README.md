# Raidar

Object-detection models and trainers for the [Marainer](https://github.com/Gearlux/marainer) training framework.

Raidar drives torchvision-native detectors — torchvision (Faster R-CNN) and
Roboflow's RF-DETR — through a **single specialised path**: the
`LightningDetectionTrainer` / `LightningDetectionEvaluator` (subclasses of
marainer's base) **override** the step functions and talk to native detection
models and raw `torchmetrics` metrics directly. There is **no** adapter layer
between trainer and loss/metric.

The ultralytics-backed detectors (YOLO26, RT-DETR) plus the native
ultralytics trainer/evaluator live in the
[raidar-ultralytics](https://github.com/Gearlux/raidar-ultralytics) sibling
package — that's where ``ultralytics`` is pulled in as a dependency.

The native model contract is the shape torchvision detectors already speak:

* `model.train(); model(images, targets) → Dict[str, Tensor]` — loss dict
  (the trainer sums it for the optimisation scalar).
* `model.eval();  model(images)         → List[Dict[str, Tensor]]` — per-image
  `{boxes, scores, labels}` (a `marainer.outputs.DetectionOutput` list, fed
  straight to the metric).

`raidar.detection` holds the small native plumbing both paths use:

| Symbol | Role |
|---|---|
| `DetectionBatchInput` | Carries images + targets through `batch.input` so variable-size per-image tensors survive collation; the trainer/evaluator unwrap `(images, targets)` from it. Exposes `.shape` for Lightning's batch-size logging. |
| `detection_collate_fn` | DataLoader collate that builds a `DetectionBatchInput` and keeps per-image targets as lists. |
| `DetectionConfusionMatrix` | Native `torchmetrics.Metric` accumulating TP / FP / FN (+ precision / recall / F1); drops into a `MetricCollection` next to `MeanAveragePrecision`. |
| `DetectionOutput` / `DetectionPredictions` | Task-agnostic per-image prediction schema, re-exported from `marainer.outputs`. |

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
num_classes: 2          # 0 = background, 1 = foreground

# Raw torchvision Faster R-CNN — no raidar wrapper. Its native forward already
# returns a loss dict (train) / predictions list (eval).
model: !class:torchvision.models.detection.fasterrcnn_resnet50_fpn_v2
  num_classes: !ref:num_classes
  weights_backbone: DEFAULT

trainer: !class:raidar.lightning_trainer.LightningDetectionTrainer
  experiment_name: my_detection
  run_name: faster_rcnn_v2
  num_classes: !ref:num_classes
  model: !ref:model
  train_set: !ref:train_set
  val_set: !ref:val_set
  # val_metrics / test_metrics default to MeanAveragePrecision(iou_type="bbox").
  # Combine MAP with TP/FP/FN via a plain MetricCollection (no raidar adapter):
  val_metrics: !class:torchmetrics.MetricCollection
    metrics:
      - !class:torchmetrics.detection.MeanAveragePrecision
        iou_type: bbox
      - !class:raidar.detection.DetectionConfusionMatrix
        iou_threshold: 0.5
  # train/val/test loaders default to a DataLoader with detection_collate_fn.
```

Run:

```bash
marainer train path/to/config.yaml   # or: raidar train path/to/config.yaml
```

### Dataset-derived `num_classes` (`!lazy:` model)

Like `sonair.LightningClassificationTrainer`, the trainer resolves `num_classes`
in an `initialize()` hook at run time and injects it into a **`!lazy:`** model —
so you can omit `num_classes` entirely and have it derived from the data
(`max(label) + 1`). Resolution order: a concrete model's head → explicit trainer
arg → a lazy model's pinned `num_classes` → a target walk. This works for a raw
torchvision builder *function* (confluid flows it with the derived count):

```yaml
# No num_classes anywhere — derived from the dataset, injected into the !lazy: builder.
model: !lazy:torchvision.models.detection.fasterrcnn_resnet50_fpn
  weights_backbone: DEFAULT

trainer: !class:raidar.lightning_trainer.LightningDetectionTrainer
  model: !ref:model
  train_set: !ref:train_set
  val_set: !ref:val_set
  batch_size: 2          # ergonomic flat knobs (also num_workers / max_epochs / logger)
  max_epochs: 10
```

A full flat HuggingFace example is
[`examples/train_fasterrcnn_cppe5_hf.yaml`](examples/train_fasterrcnn_cppe5_hf.yaml)
(cppe-5 — the "MNIST of object detection").

### Detector catalogue

| Model | Notes |
|---|---|
| `torchvision.models.detection.fasterrcnn_*` / `retinanet_*` / `fcos_*` / `ssd*` | Used **raw** — conform to the native forward contract directly, no raidar wrapper. **Registered** as `detection_model` (see *Registered builders* below). |
| `raidar.rfdetr.RFDETRModel` | Roboflow's RF-DETR family (native torchvision-style forward). |

The Ultralytics-backed YOLO26 + RT-DETR native models — plus the native
ultralytics trainer/evaluator (mosaic + albumentations + Ultralytics
`LOGGER`) — live in
[raidar-ultralytics](https://github.com/Gearlux/raidar-ultralytics).

### Registered builders (discovery + FluxStudio)

Torchvision ships its detectors as builder **functions**, not classes.
`raidar.configurables` registers the full detection family as `detection_model`
so they're discoverable everywhere a raidar detector is:

```python
# raidar/configurables.py (loaded via the confluid.configurables entry point)
register(fasterrcnn_resnet50_fpn,            task="detection", role="model", group="detection", lazy=True)
register(fasterrcnn_resnet50_fpn_v2,         task="detection", role="model", group="detection", lazy=True)
register(fasterrcnn_mobilenet_v3_large_fpn,  task="detection", role="model", group="detection", lazy=True)
register(fasterrcnn_mobilenet_v3_large_320_fpn, ...)
register(retinanet_resnet50_fpn, ...);  register(retinanet_resnet50_fpn_v2, ...)
register(fcos_resnet50_fpn, ...)
register(ssd300_vgg16, ...);            register(ssdlite320_mobilenet_v3_large, ...)
```

| Family | Builders |
|---|---|
| Faster R-CNN | `fasterrcnn_resnet50_fpn`, `_v2`, `fasterrcnn_mobilenet_v3_large_fpn`, `_320_fpn` |
| RetinaNet | `retinanet_resnet50_fpn`, `_v2` |
| FCOS | `fcos_resnet50_fpn` |
| SSD | `ssd300_vgg16`, `ssdlite320_mobilenet_v3_large` |

What registration buys (no behaviour change for raw dotted-path YAML — the
`!lazy:torchvision...` form still works):

* **navigaitor** lists each builder in `list_configurable_classes(category="detection_model")`
  (and the visual-editor form-spec), beside `RFDETRModel` — `num_classes` and
  the other knobs are enumerated from the builder's own signature.
* **FluxStudio** generates a `DATAFLUX_OBJECT:model` **object node** per builder
  (palette `Taidal/Model/Detection`), with widgets introspected from the
  function signature. Its output wires **straight into a detection trainer's
  `model` slot** — no Retag node. `lazy=True` keeps the model deferred so the
  trainer injects the dataset-derived `num_classes` at run time.

Mask R-CNN / Keypoint R-CNN are intentionally **not** registered — they need
instance-mask / keypoint targets, not the `{boxes, labels}` shape
`detection_collate_fn` produces.

## Discovery

Detection `@configurable` classes carry `task`/`role` tags — `RFDETRModel`
(`detection_model`), `LightningDetectionTrainer` (`role="trainer"`),
`LightningDetectionEvaluator` (`role="evaluator"`) — so navigaitor's
`list_configurable_classes(category=...)` / `list_classes(task=…, role=…)` and
the visual-editor form-spec enumerate them by kind. The native trainer computes
loss inside the model's `forward`, so there is no detection-loss config.
Discovery is automatic from the `@configurable` registry; there are no
hand-written manifest files.

## Inherited from marainer

`LightningDetectionTrainer` / `LightningDetectionEvaluator` subclass marainer's
base trainer/evaluator, so they automatically pick up marainer's default
`ModelCheckpoint` (`save_last=True` + best by the first val metric) and
YAML-config artifact logging. Wire stock `pytorch_lightning.loggers.*` classes
under `lightning.logger:`. See marainer's README → *Default checkpointing* for
the full semantics.

## Examples

End-to-end YAML in `examples/`:

- `lightning_train_native.yaml` — `LightningDetectionTrainer` + raw torchvision
  Faster R-CNN + raw `MeanAveragePrecision`, on a synthetic in-memory dataset.
- `train_fasterrcnn_cppe5_hf.yaml` — the flat HuggingFace example (the
  classification-side analogue of sonair's `train_efficientvit_b0_sonair.yaml`):
  `HuggingFaceSource("cppe-5")` → `dataflux.ops.target.CocoToTorchVisionDetectionOp`
  → `ToTensorOp` → `LightningDetectionTrainer` with a `!lazy:` torchvision
  Faster R-CNN (the fast MobileNet-320 variant) whose `num_classes` is derived.
- `train_pennfudan_hf.yaml` — the **fast/tiny** example: Penn-Fudan Pedestrian
  (160 train / 10 test, 1 class — the torchvision detection-tutorial dataset).
  Penn-Fudan ships masks not boxes, so it uses
  `dataflux.ops.target.MasksToDetectionBoxesOp` (per-instance bbox from the
  `instance_id` mask) → `ToTensorOp(mode="RGB")` → MobileNet-320 Faster R-CNN.

## License

MIT.
