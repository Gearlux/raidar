# Raidar

Object detection across **ten training backends**, driven by **one config**.

```bash
raidar train    examples/train_cppe5.yaml                      # the default backend (lightning)
raidar train    examples/train_cppe5.yaml --framework torch    # the hand-written control loop
raidar train    examples/train_cppe5.yaml --framework keras    # the shared FCOS spec on Keras 3
raidar evaluate examples/train_cppe5.yaml
raidar predict  examples/train_cppe5.yaml
raidar frameworks                                              # what the backends are
```

The training backend is a property of the **config**, not of the CLI: an alternative backend is a
`!scope:framework=<name>` block in the YAML that overrides only the framework-specific keys, and
`--framework <name>` activates it. The same flag works on the generic runner
(`recordstream run cfg.yaml --framework rfdetr`).

| framework | what it is | extra |
|---|---|---|
| `lightning` | The reference backend — a standard `pytorch_lightning.LightningModule` (`raidar.LightningDetector`) driving torchvision-native models: the torchvision builders, `RFDETRModel`, `TorchFcos`. | `raidar[lightning]` |
| `torch` | The control experiment — a hand-written loop (`raidar.TorchDetector`), no training framework at all: `zero_grad`/`backward`/`step` in the open, same models, same config. | `raidar[torch]` |
| `fastai` | A `Learner` with one-cycle scheduling (`raidar.FastaiDetector`) over the same torch stack — the loss-inside-forward contract bridged by a no-op loss. | `raidar[fastai]` |
| `huggingface` | `transformers.Trainer` (`raidar.HuggingFaceDetector`) fine-tuning any Hub object-detection checkpoint by name through ONE light wrapper (`raidar.HuggingFaceDetectionModel`). | `raidar[huggingface]` |
| `keras` | Keras 3 (`raidar.KerasDetector`) training the shared FCOS spec on whichever engine `KERAS_BACKEND` names (torch / TF / jax). | `raidar[keras]` |
| `tensorflow` | A native `tf.data` + `GradientTape` loop (`raidar.TensorFlowDetector`) — traced steps, XLA knob, SavedModel export; drives the spec in **raw `tf.nn`** (`TensorFlowFcos`). | `raidar[tensorflow]` |
| `jax` | An `nnx.jit`-traced optax loop (`raidar.JaxDetector`) over the spec in `flax.nnx` (`FlaxFcos`). | `raidar[jax]` |
| `mlx` | Apple's MLX (`raidar.MlxDetector`) — unified memory, lazy arrays; Apple silicon only. | `raidar[mlx]` |
| `rfdetr` | RF-DETR's **own** PyTorch-Lightning recipe (`raidar.RFDETRDetector`) — COCO-mAP eval, EMA, best-checkpoint saving, LR schedule. Apache-2.0, so it lives in core. | `raidar[rfdetr]` |
| `ultralytics` | Ultralytics YOLO (AGPL) — lives in the [raidar-ultralytics](https://github.com/Gearlux/raidar-ultralytics) sibling package; raidar carries only the table row and the config block. | sibling package |

## Installation

**The core is backend-free** — `pip install raidar` gives you the CLI and the config surface only.
A training framework arrives with the backend you select:

```bash
pip install "git+https://github.com/Gearlux/raidar.git@main#egg=raidar[lightning]"    # the reference backend
pip install "git+https://github.com/Gearlux/raidar.git@main#egg=raidar[huggingface]"  # transformers.Trainer
pip install "git+https://github.com/Gearlux/raidar.git@main#egg=raidar[keras]"        # the Keras 3 loop
pip install "git+https://github.com/Gearlux/raidar.git@main#egg=raidar[rfdetr]"       # RF-DETR's stack
```

`[torch]` is the base of the torch-family backends (`lightning` / `fastai` build on it). For the
ultralytics backend:

```bash
pip install git+https://github.com/Gearlux/raidar-ultralytics.git@main
```

Every backend-owned name in `raidar/__init__.py` resolves lazily, so `import raidar` pulls no
framework and touching a missing one raises an `ImportError` naming the exact extra.

### Development

```bash
git clone https://github.com/Gearlux/raidar.git
cd raidar
pip install -e ".[lightning,fastai,huggingface,keras,tensorflow,jax,rfdetr,dev]"
```

## The one config

[`examples/train_cppe5.yaml`](examples/train_cppe5.yaml) trains on **cppe-5** (the "MNIST of
object detection": 1000 medical-PPE images, 5 classes, COCO boxes) on every backend, changing only
the flag. Its shape is the demo:

* everything **above** the framework divider is engine-agnostic — the dataset splits, the
  batch/epoch knobs, the run identity;
* every framework-specific key (`runnable`, `model`, `detection_ops`, `num_classes`,
  `class_names`) is a **top-level sibling**, and each backend's `!scope:framework=` block
  overrides only the ones it must. (Flat on purpose: a scope block replaces a key and takes its
  whole subtree, so a key nested inside `runnable:` would vanish the moment a block swapped the
  runnable.)

> ⚠️ **The label convention is the real semantic difference between the backends**, and it is
> silent when wrong. torchvision (the lightning default) reserves **background at index 0** —
> `CocoToTorchVisionDetection(label_offset=1)` and `num_classes: 6` on cppe-5. Every other
> convention here is **0-indexed** (`0..K-1`): RF-DETR (background at index `K`), the HF
> detectors, ultralytics, and the FCOS spec engines — `label_offset: 0` and `num_classes: 5`.
> Each backend's block restates both keys; inheriting the wrong shift trains with every label
> off by one and nothing raising.

A typo'd backend is an error, not a silent fall-through: `--framework lightnin` raises a
`ScopeError` listing the frameworks the config declares.

## The torch-family backends — lightning, torch, fastai

`raidar.LightningDetector` is one self-contained standard `pytorch_lightning.LightningModule` —
its own detection step functions, `configure_optimizers`, and a `run()` that builds the loaders +
`pl.Trainer` and dispatches on the `task` knob (`fit`/`evaluate`/`test`/`predict`; the CLI verb
wins over the YAML's `task:`). There is **no adapter layer**: it talks to native detection models
and raw `torchmetrics` metrics directly.

The native model contract is the shape torchvision detectors already speak:

* `model.train(); model(images, targets) → Dict[str, Tensor]` — loss dict (summed for the
  optimisation scalar).
* `model.eval();  model(images)          → List[Dict[str, Tensor]]` — per-image
  `{boxes, scores, labels}` (a `recordstream.DetectionOutput` list, fed straight to the metric).

`raidar.TorchDetector` removes the training framework entirely and the config still does not
change — the same dataset, ops, model, torchmetrics and artifacts over a hand-written loop with a
`device:` knob, a plain-`state_dict` `checkpoint_dir:` and `grad_clip:`. `raidar.FastaiDetector`
swaps in fastai's `Learner` (one-cycle scheduling): the native contract computes loss INSIDE the
model's forward, which the backend bridges with a **no-op loss** — the training forward returns
the summed loss dict as the "prediction" and `loss_func` passes it through, so fastai backprops
it unchanged in one forward. The recorded `valid_loss` is therefore a meaningless 0; validation
quality comes from the same torchmetrics as the torch backend, via a callback.

### `num_classes` resolution

`initialize()` (run at the top of `run()`) resolves `num_classes` and injects it into a `!lazy:`
model — so a config can omit it entirely. On a fit: a concrete model's head → the explicit arg →
a lazy model's pinned kwarg → derived from the train targets (`max(label) + 1`, a full pass —
announced). On an eval task it is **model-driven**: the model is flowed first (it loaded its own
checkpoint) and the count is read off the live head, best-effort — the default mAP metric is
class-count-agnostic.

### Metrics — shared set + per-split additions

`metrics:` is what every **eval** split measures (defaulting to
`MeanAveragePrecision(iou_type="bbox")` — a sentinel `None` means "not chosen", so an explicitly
wired metric is never replaced); `val_metrics:` / `test_metrics:` add to one split, merged over
the shared set by name (`matrainer.metric_slots.merge_metric_specs` — the same rule as every
other task in the workspace). There is deliberately **no `train_metrics` slot**: a native model in
train mode returns only the loss dict, so there are no train-split predictions to score.

```yaml
val_metrics:
  # Per-class AP/AR — [C] vectors, reported through the trackers' TEXT channel as a
  # worst-first table (plot_metrics: true adds a bar chart; class_names: labels the rows).
  map: !lazy:torchmetrics.detection.MeanAveragePrecision { iou_type: bbox, class_metrics: true }
  # TP/FP/FN + precision/recall/F1 at fixed IoU/score thresholds:
  confusion: !lazy:raidar.DetectionConfusionMatrix { iou_threshold: 0.5 }
```

A non-scalar metric value never goes through Lightning's `log_dict` (which raises on it): scalars
are logged under the canonical `<split>/<name>` names (`train/loss`, `val/map`, …), per-class
vectors are rendered as tables/figures to the log and to every tracker's text/figure channel.

### Where the results go

`trackers:` is the whole of a run's output wiring — there is **no `logger:` slot**. Each matrainer
tracker is adapted to Lightning's logger interface (or hooked through the plain provenance
functions / the Keras callback, per backend), so per-step metrics land in the same backends
as the git provenance, the resolved config, the run manifest and the metrics sidecar
(`metrics_sidecar_path:`). `metric_names:` re-spells the recorded names for another engine's
vocabulary when you need it.

```yaml
trackers:
  - !class:matrainer.MlflowTracker
    experiment_name: !ref:experiment_name
  - !class:matrainer.TensorBoardTracker
```

## The huggingface backend

`raidar.HuggingFaceDetector` runs `transformers.Trainer` over any Hub object-detection checkpoint
named as a STRING through the one generic wrapper — the model computes its own DETR-family loss
from `labels=`, so there is no loss to wire:

```yaml
huggingface: !scope:framework=huggingface
  runnable: !class:raidar.HuggingFaceDetector
    learning_rate: 5e-5              # a fine-tuning LR — we start from real weights
  model: !lazy:raidar.HuggingFaceDetectionModel
    model_name: ustc-community/dfine-small-coco    # any Hub detection checkpoint
  detection_ops:
    - !class:recordstream.ops.target.CocoToTorchVisionDetection { bbox_format: xywh, label_offset: 0 }
  num_classes: 5
```

The ops only shape the TARGET (0-indexed labels): images stay RAW — the checkpoint's own image
processor does the resize / pad / normalize and the box conversion. `TrainingArguments` defaults
are derived at run time as a FLOOR (`remove_unused_columns=False` included); a key your config
pins wins. With `num_classes != 80` the pretrained class head is re-initialised
(`ignore_mismatched_sizes=True`) while everything else stays pretrained.

## The shared FCOS spec — keras, tensorflow, jax, mlx

Every detection zoo raidar wraps is torch-only, so a cross-framework comparison built on zoos
would compare different networks. The [`raidar/fcos/`](raidar/fcos/) package is the answer: a
tiny single-level anchor-free FCOS-style detector (~110K parameters) **specified once**
([`spec.py`](raidar/fcos/spec.py)) and implemented natively per framework — `TorchFcos` (serves
lightning / torch / fastai too), `KerasFcos`, `TensorFlowFcos` (raw `tf.nn`), `FlaxFcos`,
`MlxFcos` — each checked against the spec's DERIVED parameter count, so "the same network" is a
claim the tests enforce rather than an aspiration. For real work prefer a zoo model; this is the
control, not the recommendation.

The spec also ships the **numpy reference algorithms** every implementation shares: FCOS
center-sampling target assignment, dense-grid decode + NMS, and AP50/precision/recall — the
scoring the non-torch engines report (torchmetrics is torch-only). Only the LOSS is per-engine
(it must be differentiable): sigmoid focal + `1 − IoU` + centerness BCE.

Two structural changes in a spec engine's config block:

* the ops chain gains `ResizeDetection` (dense targets need ONE fixed grid) and ends in
  `raidar.fcos.targets.FcosTargets`, which encodes each record's boxes into fixed-shape dense
  maps — that is what lets `collate_records` stack detection batches with no ragged tensors;
* labels are 0-indexed (the class map has no background slot).

```yaml
detection_ops:
  - !class:recordstream.ops.target.CocoToTorchVisionDetection { bbox_format: xywh, label_offset: 0 }
  - !class:recordstream.ops.target.ResizeDetection { width: 256, height: 256 }
  - !class:recordstream.ops.torch.ToTensor { mode: RGB, normalize: true }
  - !class:raidar.fcos.targets.FcosTargets { num_classes: 5, input_shape: [256, 256, 3] }
```

**Resize with `ResizeDetection`, not an image-only resize.** Three knobs must agree — the resize's
`width`/`height`, the op's `input_shape`, and the model's — and note the ordering: the resize takes
`(width, height)` while `input_shape` is `(H, W, C)`. `ResizeDetection` moves the image AND the
boxes; an image-only resize (`ConvertToImage` and friends) moves only the pixels and leaves the
boxes in the source frame, which every shape check downstream would accept. Both mistakes are
caught for you at the first record:

```
the boxes reach (426.0, 426.0) but the frame is 128x128 (W x H) — they are not in this frame …
FcosTargets: the target's boxes are in a (256, 256) (H, W) frame but this op assigns against (128, 128) …
```

Engine notes: `keras` runs on whichever engine `KERAS_BACKEND` names and never sets it;
`tensorflow` pins it (via `raidar._tensorflow`) and must run in a fresh process — Keras 3 binds
one engine per process at first import; `jax` jits its steps ONCE (`jit: false` to read a
traceback); `mlx` is Apple-silicon only, with the extra's platform marker keeping the install
valid elsewhere.

Every spec model takes a `seed` (default `0`) applied before its weights are built, so a config
reproduces its run — two identical runs land on the same loss to the last digit. Change it to
sample a different initialization:

```yaml
model: !lazy:raidar.fcos.keras_fcos.KerasFcos { input_shape: [256, 256, 3], seed: 7 }
```

## The rfdetr backend

`raidar.RFDETRDetector` hands off to RF-DETR's **own** training recipe (`RFDETRModelModule` +
`build_trainer`) — one class, train and eval, dispatched on the same `task` knob. Two input modes
(mutually exclusive):

```yaml
# Disk mode — a COCO/YOLO/Roboflow dataset directory → RF-DETR's native loaders
runnable: !class:raidar.RFDETRDetector
  data: /data/my_coco_dataset
  dataset_file: roboflow
  variant: small                   # nano | small | medium | large
  epochs: 50
```

```yaml
# Stream mode — a RecordStream pipeline, fed IN-MEMORY (no disk materialization)
runnable: !class:raidar.RFDETRDetector
  variant: small
  num_classes: 5                       # K foreground classes; labels 0-indexed 0..K-1
  class_names: {0: Coverall, 1: Face_Shield, 2: Gloves, 3: Goggles, 4: Mask}
  train_set: !ref:train_set            # any recordstream Stream yielding {boxes xyxy-px, labels}
  val_set: !ref:val_set
  epochs: 50
```

Beyond the ergonomic knobs, **`train_config:` reaches every RF-DETR parameter**: the mapping is
handed verbatim to RF-DETR's own config builder, whose pydantic `TrainConfig` validates it — so
any knob an RF-DETR release adds is immediately configurable with no raidar edit. Raidar-derived
values are a FLOOR; a key you pin wins.

**Pin a seed if you want a run you can compare against itself** — RF-DETR's `TrainConfig.seed`
defaults to `None`, i.e. it seeds nothing:

```yaml
runnable: !class:raidar.RFDETRDetector
  train_config: { seed: 7 }
  device: cpu                          # see below
```

Seeded, two runs land identically **on cpu**. On mps they still differ — that residual is Apple's
MPS backend rather than the config, so mps is the faster option and cpu is the reproducible one.

`raidar evaluate --framework rfdetr` runs RF-DETR's native validation/test loop over a trained
checkpoint (`weights:`). The wired `trackers:` receive the run's **record** (provenance, config,
manifest, log file, final-metrics sidecar); RF-DETR's own per-step curves stay with its native
`mlflow:` / `tensorboard:` knobs. There is no `predict` here — prediction runs through the
lightning backend with `raidar.RFDETRModel` (the native model module, which also loads RF-DETR
checkpoints for evaluation under `LightningDetector`).

## Detector catalogue

| Model | Notes |
|---|---|
| `torchvision.models.detection.fasterrcnn_*` / `retinanet_*` / `fcos_*` / `ssd*` | Used **raw** — they conform to the native forward contract directly. Registered as `detection_model` builders (below). |
| `raidar.RFDETRModel` | RF-DETR family as a native torchvision-style model module. |
| `raidar.HuggingFaceDetectionModel` | ONE light wrapper — `model_name:` reaches any Hub object-detection checkpoint (D-FINE, RT-DETRv2, DETR, Deformable-DETR, …). |
| `raidar.fcos.*` | The shared spec's five native implementations (above). |

The Ultralytics-backed YOLO26 + RT-DETR native models live in
[raidar-ultralytics](https://github.com/Gearlux/raidar-ultralytics).

`raidar.torch_detection` is the torch ENGINE's module — the batch plumbing every torch-family
backend shares, plus that engine's registered metrics:

| Symbol | Role |
|---|---|
| `native_batch` | The ONE torch-contract adapter: reads a collated batch into the `(List[Tensor], per-image targets)` pair torchvision's forward takes. **Batching itself is recordstream's** — the loaders declare `collate_fn=collate_list` (the unstacked batch the native contract needs), and `batch_boxes` / `batch_metadata` read it back. Override `collate_fn` for a model wanting one stacked tensor. |
| `DetectionOutput` / `DetectionPredictions` | Task-agnostic per-image prediction schema (re-exported from recordstream). |
| the `torchmetrics.detection` family + `DetectionConfusionMatrix` | The engine's registered metrics (TP / FP / FN + precision / recall / F1 for the last). |

`raidar.torchvision_detection` registers the torchvision zoo's builder **functions** separately —
one module per upstream library, both named in the `lightning` / `torch` / `fastai` rows of the
`FRAMEWORKS` table.

## Discovery

Detection `@configurable` classes carry `task` / `role` / `framework` tags — every backend
runnable is `(detection, trainer, <framework>)` (their evaluate/test/predict capabilities are
declared by `@entrypoint` markers on the same class), the model wrappers and spec implementations
are `(detection, model)`, and `raidar.torchvision_detection` registers the torchvision builder
**functions** (Faster R-CNN ×4, RetinaNet ×2, FCOS, SSD ×2) as `detection_model` with
`lazy=True`, so a discovery service or visual editor enumerates them beside the wrapper classes
and a deferred builder still gets the dataset-derived `num_classes` injected at run time.
Mask R-CNN / Keypoint R-CNN are intentionally not registered — they need instance-mask /
keypoint targets, not the `{boxes, labels}` shape `recordstream.batch_boxes` yields. Discovery runs
off the `confluid.configurables` entry points (one per registering module); there are no
hand-written manifest files.

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
scratch loaded no artifact and records nothing. Reading the identity is free and never builds the
model; adding one to your own wrapper is a single `model_refs` property — see
[matrainer's `uri` module](https://github.com/Gearlux/matrainer).

## License

MIT.
