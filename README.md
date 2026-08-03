# Raidar

Standard PyTorch Lightning object-detection models and trainers.

Raidar drives torchvision-native detectors — torchvision (Faster R-CNN),
Roboflow's RF-DETR, and the Hugging Face transformers real-time detectors
(D-FINE, RT-DETRv2) — through a **single specialised path**:
`raidar.detector.LightningDetector` — one self-contained standard
`pytorch_lightning.LightningModule` that **overrides** the step functions and
talks to native detection models and raw `torchmetrics` metrics directly, with a
`task` knob (`fit`/`evaluate`/`test`/`predict`) its `run()` dispatches on. There
is **no** adapter layer between trainer and loss/metric. It uses
[Matrainer](https://github.com/Gearlux/matrainer) only for optional experiment
tracking + `matrainer.torch` utilities.

The ultralytics-backed detectors (YOLO26, RT-DETR) plus the native
ultralytics trainer/evaluator live in the
[raidar-ultralytics](https://github.com/Gearlux/raidar-ultralytics) sibling
package — that's where ``ultralytics`` is pulled in as a dependency.

The native model contract is the shape torchvision detectors already speak:

* `model.train(); model(images, targets) → Dict[str, Tensor]` — loss dict
  (the trainer sums it for the optimisation scalar).
* `model.eval();  model(images)         → List[Dict[str, Tensor]]` — per-image
  `{boxes, scores, labels}` (a `recordstream.DetectionOutput` list, fed
  straight to the metric).

`raidar.detection` holds the small native plumbing both paths use:

| Symbol | Role |
|---|---|
| `DetectionBatchInput` | Carries images + targets through `batch.input` so variable-size per-image tensors survive collation; the trainer/evaluator unwrap `(images, targets)` from it. Exposes `.shape` for Lightning's batch-size logging. |
| `detection_collate_fn` | DataLoader collate that builds a `DetectionBatchInput` and keeps per-image targets as lists. |
| `DetectionConfusionMatrix` | Native `torchmetrics.Metric` accumulating TP / FP / FN (+ precision / recall / F1); drops into a `MetricCollection` next to `MeanAveragePrecision`. |
| `DetectionOutput` / `DetectionPredictions` | Task-agnostic per-image prediction schema, re-exported from `recordstream.outputs`. |

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

runnable: !class:raidar.detector.LightningDetector
  task: fit
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
raidar train path/to/config.yaml   # or the generic runner: recordstream run path/to/config.yaml
```

### Dataset-derived `num_classes` (`!lazy:` model)

Like `sonair.LightningClassifier`, `LightningDetector` resolves
`num_classes` in an `initialize()` hook at run time and injects it into a
**`!lazy:`** model —
so you can omit `num_classes` entirely and have it derived from the data
(`max(label) + 1`). Resolution order: a concrete model's head → explicit trainer
arg → a lazy model's pinned `num_classes` → a target walk. This works for a raw
torchvision builder *function* (confluid flows it with the derived count):

```yaml
# No num_classes anywhere — derived from the dataset, injected into the !lazy: builder.
model: !lazy:torchvision.models.detection.fasterrcnn_resnet50_fpn
  weights_backbone: DEFAULT

runnable: !class:raidar.detector.LightningDetector
  task: fit
  model: !ref:model
  train_set: !ref:train_set
  val_set: !ref:val_set
  batch_size: 2          # ergonomic flat knobs (also num_workers / max_epochs / logger)
  max_epochs: 10
```

A full flat HuggingFace example is
[`examples/train_fasterrcnn_cppe5_hf.yaml`](examples/train_fasterrcnn_cppe5_hf.yaml)
(cppe-5 — the "MNIST of object detection").

### Evaluation — model-driven `num_classes`

Evaluation is the SAME `LightningDetector` run with `task: test` (or `evaluate`).
On an eval task its `initialize()` hook (run at the top of `run()`, before the
eval loop) **flows the model first** and reads `num_classes` straight off the live
trained model — the model loaded its own checkpoint, so its head is the single
source of truth (not the eval dataset, which may be a subset). The read is
*best-effort*: the default `MeanAveragePrecision` is class-count-agnostic, so a
detector that doesn't expose its count just leaves `num_classes` unset without
erroring. `initialize()` also normalizes the wired dataset to a record-yielding `Dataset`
via recordstream's `ensure_record_dataset`.

```yaml
# task: test — num_classes is read off the trained model (which carries it).
runnable: !class:raidar.detector.LightningDetector
  task: test
  model: !class:raidar.rfdetr.RFDETRModel
    num_classes: 2                             # the model carries the count
    checkpoint_path: ./checkpoints/last.ckpt   # the model loads itself
  test_set: !ref:eval_set
  metrics_sidecar_path: ./eval_metrics.json    # optional logger-agnostic dump
```

### Detector catalogue

| Model | Notes |
|---|---|
| `torchvision.models.detection.fasterrcnn_*` / `retinanet_*` / `fcos_*` / `ssd*` | Used **raw** — conform to the native forward contract directly, no raidar wrapper. **Registered** as `detection_model` (see *Registered builders* below). |
| `raidar.rfdetr.RFDETRModel` | Roboflow's RF-DETR family (native torchvision-style forward). |
| `raidar.hf_detection.DFINEModel` | D-FINE (Apache-2.0) via Hugging Face `transformers` — real-time DETR, `variant` ∈ nano/small/medium/large/xlarge. |
| `raidar.hf_detection.RTDETRv2Model` | RT-DETRv2 (Apache-2.0) via `transformers` — NMS-free real-time DETR, `variant` ∈ r18vd/r34vd/r50vd/r101vd. |
| `raidar.hf_detection.HFDetectionModel` | Generic wrapper around **any** `transformers` object-detection checkpoint by `model_id` (DETR, Deformable-DETR, YOLOS, …). |

The Ultralytics-backed YOLO26 + RT-DETR native models — plus the native
ultralytics trainer/evaluator (mosaic + albumentations + Ultralytics
`LOGGER`) — live in
[raidar-ultralytics](https://github.com/Gearlux/raidar-ultralytics).

### Registered builders (discovery + StreamStudio)

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
* **StreamStudio** generates a `RECORDSTREAM_OBJECT:model` **object node** per builder
  (palette `Taidal/Model/Detection`), with widgets introspected from the
  function signature. Its output wires **straight into a detection trainer's
  `model` slot** — no Retag node. `lazy=True` keeps the model deferred so the
  trainer injects the dataset-derived `num_classes` at run time.

Mask R-CNN / Keypoint R-CNN are intentionally **not** registered — they need
instance-mask / keypoint targets, not the `{boxes, labels}` shape
`detection_collate_fn` produces.

## Native RF-DETR (Apache-2.0)

Two detection training paths ship in core raidar:

- the **generic Lightning path** above (`raidar train`, `LightningDetector`),
  which drives bare torchvision-native models through the standard Lightning loop; and
- the **native RF-DETR path** (`raidar-rfdetr train`, `RFDETRTrainer`), which hands
  off to RF-DETR's **own** PyTorch-Lightning recipe — COCO-mAP eval, EMA,
  best-checkpoint saving, LR schedule. It's the in-core analogue of
  `raidar-ultralytics`'s `UltralyticsTrainer`; RF-DETR is Apache-2.0 so (unlike
  AGPL ultralytics) it needs no sibling package.

`RFDETRTrainer` / `RFDETREvaluator` are tagged `@configurable(task="detection",
role=…)`, so navigaitor lists them **alongside** the Lightning detector (detection is a
multi-trainer task — pick either). Hard deps: `pip install 'rfdetr[train,loggers]'`
+ `faster-coco-eval` (RF-DETR's mAP backend).

Two input modes (mutually exclusive), mirroring `UltralyticsTrainer`:

```yaml
# Disk mode — a COCO/YOLO/Roboflow dataset directory → RF-DETR's native .train()
trainer: !class:raidar.rfdetr_trainer.RFDETRTrainer
  data: /data/my_coco_dataset      # dir with train/ valid/ + _annotations.coco.json
  dataset_file: roboflow
  variant: small                   # nano | small | medium | large
  epochs: 50
  batch_size: 4
  output_dir: ./runs/rfdetr_small
```

```yaml
# Stream mode — a RecordStream pipeline, fed IN-MEMORY (no disk materialization)
num_classes: 6                     # K foreground classes; labels are 0-indexed 0..K-1
trainer: !class:raidar.rfdetr_trainer.RFDETRTrainer
  variant: small
  num_classes: !ref:num_classes
  class_names: {0: coverall, 1: face_shield, 2: gloves, 3: goggles, 4: mask, 5: apron}
  train_set: !ref:train_set        # any recordstream Stream yielding {boxes xyxy-px, labels}
  val_set: !ref:val_set
  epochs: 50
  batch_size: 4
```

> ⚠️ **0-indexed labels.** RF-DETR reserves the background class at index `K`, the
> **opposite** of torchvision's background-at-0. Wire a 0-indexed target op
> (`CocoToTorchVisionDetectionOp(label_offset=0)`), **not** the `+1` shift the
> Faster R-CNN example uses.

`raidar-rfdetr evaluate` runs RF-DETR's native
validation/test loop over a trained checkpoint (`weights:`). See
`examples/rfdetr_train_disk.yaml`, `examples/rfdetr_train_stream.yaml`,
`examples/rfdetr_evaluate_stream.yaml`.

## Hugging Face detectors — D-FINE / RT-DETRv2 (Apache-2.0)

`raidar.hf_detection` wraps Hugging Face `transformers` object detectors into the
native forward contract, so they run through the **generic Lightning path** above
(`raidar train` / `raidar evaluate`) with no bespoke trainer — exactly like a
torchvision builder or `YOLO26Model`. Both shipped presets are **Apache-2.0** (so
they live in raidar *core*, unlike the AGPL ultralytics family) and come straight
from the installed `transformers` — no extra/unmaintained dependency:

- **`DFINEModel`** — D-FINE (2024), `variant` ∈ `nano`/`small`/`medium`/`large`/`xlarge`.
- **`RTDETRv2Model`** — RT-DETRv2 ("DETRs Beat YOLOs"), `variant` ∈ `r18vd`/`r34vd`/`r50vd`/`r101vd`.
- **`HFDetectionModel`** — the generic base; point `model_id` at any HF object-detection checkpoint.

```yaml
runnable: !class:raidar.detector.LightningDetector
  task: fit
  num_classes: 5                         # K foreground classes; injected into the !lazy: model
model: !lazy:raidar.hf_detection.DFINEModel
  variant: small
  image_size: 640                        # square side, multiple of 32 (these models train at 640)
detection_ops:
  - !class:recordstream.ops.target.CocoToTorchVisionDetectionOp
    bbox_format: xywh
    label_offset: 0                      # ⚠️ HF detectors are 0-indexed — do NOT +1
  - !class:recordstream.ops.torch.ToTensorOp
    mode: RGB
    normalize: true                      # [0,255] → [0,1]; HF processor is do_normalize=False
train_set: { ... }                       # any recordstream Stream yielding {boxes xyxy-px, labels}
val_set:   { ... }
```

> ⚠️ **0-indexed labels** (like RF-DETR, the **opposite** of torchvision's
> background-at-0): `label_offset: 0`, `num_classes = K` foreground classes.

Two convenience behaviours of the wrapper, grounded in the models' processors:

* **Images are resized internally** to the square `image_size` (default 640), and
  predictions are decoded back into each image's **original** pixel space — so the
  same op chain as the Faster R-CNN example drops in (no detection-aware resize op).
* **Input must be `[0, 1]`, NOT ImageNet-normalised** — the RT-DETR/D-FINE
  processors run with `do_normalize=False`. `ToTensorOp(normalize=True)` produces
  exactly that; do not add a `StandardizeOp`.

When `num_classes != 80`, the COCO-pretrained class head is re-initialised
(`ignore_mismatched_sizes=True`) while the backbone / encoder / decoder stay
pretrained. Evaluate with `LightningDetector` run at `task: test` (model-driven
`num_classes`; pass the trained `.ckpt` via the model's `checkpoint_path`). See
`examples/dfine_train_stream.yaml`, `examples/dfine_evaluate_stream.yaml`,
`examples/rtdetrv2_train_stream.yaml`, `examples/rtdetrv2_evaluate_stream.yaml`.

## Discovery

Detection `@configurable` classes carry `task`/`role` tags — `RFDETRModel`
(`detection_model`), `LightningDetector` **and** `RFDETRTrainer`
(`role="trainer"`; `LightningDetector` also serves evaluate/test/predict via
`@entrypoint`-annotated methods), and `RFDETREvaluator` (`role="evaluator"`) — so
navigaitor's `list_configurable_classes(category=...)` /
`list_classes(task=…, role=…)` and the visual-editor form-spec enumerate them by
kind. Detection is a **multi-trainer task**: `(detection, trainer)` resolves to both
trainers, and navigaitor's `train_detection` config is a `Union` over them (pick
either). The Lightning detector computes loss inside the model's `forward`, so there
is no detection-loss config.
Discovery is automatic from the `@configurable` registry; there are no
hand-written manifest files.

## Tracking & checkpointing

`LightningDetector` is a self-contained standard `pytorch_lightning.LightningModule`,
so checkpointing and logging are plain Lightning concerns. Wire stock
`pytorch_lightning.loggers.*` classes via the `logger:` flat knob (or under the
`lightning:` `L.Trainer` block), and add a `pytorch_lightning.callbacks.ModelCheckpoint`
to the trainer's callbacks for best/last saving. For optional experiment tracking,
wire `matrainer.tracking` backends under `trackers:` — a `TrackingCallback` is
attached automatically, and system-metric + progress callbacks come for free. See
matrainer's README for the tracking backends.

## Examples

End-to-end YAML in `examples/`:

- `lightning_train_native.yaml` — `LightningDetector` + raw torchvision
  Faster R-CNN + raw `MeanAveragePrecision`, on a synthetic in-memory dataset.
- `train_fasterrcnn_cppe5_hf.yaml` — the flat HuggingFace example (the
  classification-side analogue of sonair's `train_efficientvit_b0_sonair.yaml`):
  `HuggingFaceSource("cppe-5")` → `recordstream.ops.target.CocoToTorchVisionDetectionOp`
  → `ToTensorOp` → `LightningDetector` with a `!lazy:` torchvision
  Faster R-CNN (the fast MobileNet-320 variant) whose `num_classes` is derived.
- `train_pennfudan_hf.yaml` — the **fast/tiny** example: Penn-Fudan Pedestrian
  (160 train / 10 test, 1 class — the torchvision detection-tutorial dataset).
  Penn-Fudan ships masks not boxes, so it uses
  `recordstream.ops.target.MasksToDetectionBoxesOp` (per-instance bbox from the
  `instance_id` mask) → `ToTensorOp(mode="RGB")` → MobileNet-320 Faster R-CNN.
- `dfine_train_stream.yaml` / `dfine_evaluate_stream.yaml` — D-FINE on cppe-5 through
  the generic Lightning path (HF transformers, `label_offset: 0`).
- `rtdetrv2_train_stream.yaml` / `rtdetrv2_evaluate_stream.yaml` — RT-DETRv2 on cppe-5,
  same wiring.

## License

MIT.

## What the run record says about the model

Every model wrapper here names the weight artifacts it loads, so a run's manifest points at them
instead of leaving a reader to work out which registry resolves a bare architecture name. Nothing
has to be called for it — the tracking layer walks the configured graph and records what it finds:

```yaml
models:
- {kind: hf, name: ustc-community/dfine-small-coco, role: pretrained, slot: model,
   url: 'https://huggingface.co/ustc-community/dfine-small-coco'}
```

**One wrapper may report several**, each with the `role` it played: `pretrained` for weights that
came from outside the run, `checkpoint` for weights a previous run wrote. A model trained from
scratch loaded no artifact and records nothing — the architecture it used is configuration, and is
already in the manifest's `params`.

Reading the identity is free and never builds the model: the properties consult stored
configuration only. Adding one to your own wrapper is a single `model_refs` property — see
[matrainer's `uri` module](https://github.com/Gearlux/matrainer).
