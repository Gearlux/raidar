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
| `ultralytics` | Ultralytics YOLO (AGPL) — lives in a separate package that is not part of this distribution; raidar carries only the table row and the config block. | separate package |

## Installation

**The core is backend-free** — `pip install raidar` gives you the CLI and the config surface only.
A training framework arrives with the backend you select:

```bash
pip install "raidar[lightning]"     # the reference backend
pip install "raidar[huggingface]"   # transformers.Trainer
pip install "raidar[keras]"         # the Keras 3 loop
pip install "raidar[rfdetr]"        # RF-DETR's stack
```

raidar is on PyPI as a pre-release; pip installs it without `--pre`, because no final release
exists yet. Leave `--pre` off: it lets pip take pre-releases of *every* dependency too.

`[torch]` is the base of the torch-family backends (`lightning` / `fastai` build on it).

Every backend-owned name in `raidar/__init__.py` resolves lazily, so `import raidar` pulls no
framework and touching a missing one raises an `ImportError` naming the exact extra.

### Development

```bash
git clone https://github.com/Gearlux/raidar.git
cd raidar
pip install -e ".[lightning,fastai,huggingface,keras,tensorflow,jax,rfdetr,dev]"
```

## The one config

[`examples/train_cppe5.yaml`](https://github.com/Gearlux/raidar/blob/main/examples/train_cppe5.yaml) trains on **cppe-5** (the "MNIST of
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

`raidar.LightningDetector` is a standard `pytorch_lightning.LightningModule` built on matrainer's
`LightningRunnable` — the loaders, the deferred `pl.Trainer`, the tracking, and a `run()` that
dispatches on the `task` knob (`fit`/`evaluate`/`test`/`predict`; the CLI verb wins over the YAML's
`task:`) — plus the detection task hooks shared by the torch-family backends. What the class adds
is its own native detection step functions. There is **no adapter layer**: it talks to native
detection models and raw `torchmetrics` metrics directly.

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
would compare different networks. The [`raidar/fcos/`](https://github.com/Gearlux/raidar/tree/main/raidar/fcos) package is the answer: a
tiny single-level anchor-free FCOS-style detector (~110K parameters) **specified once**
([`spec.py`](https://github.com/Gearlux/raidar/blob/main/raidar/fcos/spec.py)) and implemented natively per framework — `TorchFcos` (serves
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
  - !class:recordstream.ops.image.ConvertMode { mode: RGB }
  - !class:recordstream.ops.numpy.Scale {}
  - !class:recordstream.ops.torch.ToTensor {}
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
  variant: small                   # nano | small | medium | large — or seg-nano | seg-small | seg-medium | seg-large | seg-xlarge
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
values are a FLOOR; a key you pin wins. **`model_config:` is the same pass-through for the MODEL's
settings, which are fixed when it is built** (`group_detr`, `num_queries`, …); `num_classes`,
`resolution` and the weights have parameters of their own and are refused there:

```yaml
runnable: !class:raidar.RFDETRDetector
  variant: seg-medium
  model_config: { group_detr: 8 }   # default 13; the mask loss's memory scales with it
```

A lower `group_detr` loads the published weights only partly (the dropped groups' tensors are skipped).

**Pin a seed if you want a run you can compare against itself** — RF-DETR's `TrainConfig.seed`
defaults to `None`, i.e. it seeds nothing:

```yaml
runnable: !class:raidar.RFDETRDetector
  train_config: { seed: 7 }
  device: cpu                          # see below
```

Seeded, two runs land identically **on cpu**. On mps they still differ — that residual is Apple's
MPS backend rather than the config, so mps is the faster option and cpu is the reproducible one.

**A segmentation variant trains from the same boxes.** `variant: seg-nano` … `seg-xlarge` build
RF-DETR's instance-segmentation models; in disk mode with `dataset_file: yolo` RF-DETR reads a box
label as its four-corner polygon, so a box dataset is enough (a label line with ≥ 3 points is read
as that polygon — `YoloSink` writes such lines when a record's target carries
`extras["polygons"]`, below). In stream mode a `seg-*` variant takes each instance's mask from its target's
`extras["polygons"]` (a box without an outline is masked by its rectangle), rasterised by RF-DETR's own converter,
and a picture's labels are read with the picture — so a random step in the stream cuts both alike (next section).
The `RFDETRModel` wrapper (the lightning backend) offers the four detection variants: it builds box targets and no
masks.

### One picture size for any picture — `raidar.RandomCropDetection`

RF-DETR batches a seg model's masks only when every picture of a batch comes out of its own resize the same size;
pictures of different shapes stop the run (`Sizes of tensors must match except in dimension 0`), and its own crop
(`scale_jitter`) and shape-keeping resize (`square_resize_div_64: false`) cannot fix that — they run after the resize,
and the long side is capped at 1333 px. `RandomCropDetection` cuts each picture to ONE `height` x `width` piece in
its own pixels, before RF-DETR sees it: its boxes moved and clipped (a box left under `min_size_px`, 3, in either
direction dropped), its outlines clipped exactly, a picture smaller than the piece padded with its median — never
stretched. `placement: random` cuts anywhere (Python's `random`, which the data loader seeds per worker and epoch: a
new piece each epoch); `centre` the middle piece, every call — for validation. Read a folder with `YoloSource`:

```yaml
runnable: !class:raidar.RFDETRDetector
  variant: seg-nano
  resolution: 504
  num_classes: 2
  class_names: {0: signal, 1: ignore}
  train_set: !class:recordstream.core.stream.Stream
    source: !class:raidar.YoloSource {root: ./runs/pictures, split: train}
    ops:
      - !class:raidar.crop.RandomCropDetection {height: 504, width: 504, placement: random}
  val_set: !class:recordstream.core.stream.Stream
    source: !class:raidar.YoloSource {root: ./runs/pictures, split: val}
    ops:
      - !class:raidar.crop.RandomCropDetection {height: 504, width: 504, placement: centre}
```

Measured (2026-10-12, seg-nano at 504, one epoch over 130 + 65 pictures 504 px wide and 100, 1 008 or 3 000 rows
tall): it trains — mask losses and `val/segm_mAP_*` written. Without the cut, the same folder read by RF-DETR's
shape-keeping resize stops on the masks' sizes, and a 504 x 3000 picture reaches the model 224 px wide.


`raidar evaluate --framework rfdetr` runs RF-DETR's native validation/test loop over a trained
checkpoint (`weights:`). The wired `trackers:` receive the run's **record** (provenance, config,
manifest, log file, final-metrics sidecar); RF-DETR's own per-step curves stay with its native
`mlflow:` / `tensorboard:` knobs. There is no `predict` here — prediction runs through the
lightning backend with `raidar.RFDETRModel` (the native model module, which also loads RF-DETR
checkpoints for evaluation under `LightningDetector`), or, for a checkpoint this backend trained —
a `seg-*` one included, masks and all — through `raidar.RFDETRPredictor` (below).

### Writing a detection Stream to disk (`raidar.YoloSink`)

A `DataSink` for `recordstream run`: `DatasetProcessor(stream=…, sink=YoloSink(…))` writes each
record's image and its `{boxes, labels}` target to a YOLO dataset folder, one split per run into the
SAME `output_dir` (the sink merges `data.yaml` and appends `metadata.jsonl`). `layout` chooses whose
folder it is:

| `layout` | folder | reader |
|---|---|---|
| `ultralytics` (default) | `images/<split>/`, `labels/<split>/`, `annotations/<split>.json` | `YOLO.train(data=<output_dir>/data.yaml)` |
| `rfdetr` | `<split>/images/`, `<split>/labels/`, `<split>/_annotations.coco.json` | `RFDETRDetector(data=<output_dir>, train_config={dataset_file: yolo})` |

Measured with RF-DETR 1.9.3's own `is_valid_yolo_dataset`: the default layout answers `False` (it
wants `train/` and `val/` folders holding `images` and `labels`), the `rfdetr` layout `True`.

```yaml
runnable: !class:recordstream.processing.DatasetProcessor
  stream: !ref:detection_stream              # records with an image and a {boxes xyxy-px, labels} target
  sink: !class:raidar.YoloSink
    output_dir: /data/my_dataset
    split: train                              # run once per split: train, then val
    class_names: {0: signal}
    layout: rfdetr
    mode: replace                             # replace (default): wipe this split first | append: continue it
    facts: [scale]                            # list the distinct values of these record entries in data.yaml
    image_mode: rgb                           # rgb (default) | grey: one channel a picture whose three are equal
    image_threads: 4                          # threads encoding and writing the pictures (the files are the same)
```

| `mode` | what a run does to a split that exists |
|---|---|
| `replace` (default) | wipes the split's folders first, so a rerun is idempotent |
| `append` | continues it: the index goes on after the last picture, `metadata.jsonl` and the COCO sidecar grow, `data.yaml` must already name every class the same way (`class 0 is 'drone-signal' in the folder and 'other' here` is refused before anything is written) |

`facts` names record entries whose DISTINCT values `data.yaml` lists under `facts:` — a picture's
scale, for instance: a folder of mixed scales is recorded, never refused. Every record's entries
ride `metadata.jsonl` regardless.

`image_mode: grey` stores a grey picture (its three channels equal, a spectrogram's grey scale) as
ONE channel: measured on 60 spectrogram pictures of 1024 x 1024, 12.5 ms and 0.52 MB a picture
against 27.6 ms and 0.80 MB in RGB. The readers turn it back into the same three channels — RF-DETR's
YOLO reader and `YoloSource` call `convert("RGB")`, Ultralytics reads with OpenCV's colour flag — so
`data.yaml` keeps `channels: 3`. A picture whose channels differ is refused, naming its record:
`YoloSink: image_mode 'grey' writes one channel, and the picture of record 1 has three that differ —
write this folder with image_mode: rgb`.

The pictures are encoded on `image_threads` threads (PIL's encoder runs outside Python's lock: 60
PNGs took 1.74 s on one thread, 0.23 s on eight), each under the index its record was given, so the
folder is the one a single thread writes; a write that fails stops the run.

#### Instance masks — `Boxes.extras["polygons"]`

A record's target is a `recordstream.Boxes` (`boxes` `[N, 4]` xyxy pixels, `labels` `[N]`). A
producer MAY add each instance's polygons under `target.extras["polygons"]` — `raidar.POLYGONS`
names the key — and the sink then writes masks, which RF-DETR's `seg-*` models train on:

```python
from raidar import POLYGONS
from recordstream import Boxes

target = Boxes(
    boxes=[[2, 2, 56, 28], [44, 2, 60, 10]],
    labels=[1, 0],
    extras={POLYGONS: [
        [[[2, 2], [10, 2], [10, 10], [2, 10]], [[40, 16], [56, 16], [56, 28], [40, 28]]],  # instance 0: two polygons
        [],                                                                              # instance 1: no mask of its own
    ]},
)
```

The contract: a list of length N; element `i` is the LIST of polygons of box `i`; each polygon is a
list of `[x, y]` pixel points — at least 3, a closed ring, the first point not repeated. The
polygons of one instance do not overlap (they may touch). `[]` or `None` means the instance has no
mask of its own, and its mask is its box rectangle. A list of the wrong length, or a polygon of
fewer than 3 points, is refused naming the record. Without the key — or with one that holds no
polygon at all — the sink writes exactly what it wrote before.

On a 64 x 32 picture, the target above writes:

| file | before (box lines only) | with the polygons |
|---|---|---|
| `labels/<split>/000000.txt` | `1 0.453125 0.468750 0.843750 0.812500`<br>`0 0.812500 0.187500 0.250000 0.250000` | `1 0.156250 0.312500 0.031250 0.312500 0.031250 0.062500 0.156250 0.062500 0.156250 0.312500 0.625000 0.500000 0.875000 0.500000 0.875000 0.875000 0.625000 0.875000 0.625000 0.500000`<br>`0 0.687500 0.062500 0.937500 0.062500 0.937500 0.312500 0.687500 0.312500` |
| COCO `segmentation` / `area` of instance 0 | `[]` / `1404.0` (its box) | `[[2,2,10,2,10,10,2,10], [40,16,56,16,56,28,40,28]]` / `256.0` (the polygons' summed area) |
| COCO `segmentation` of instance 1 | `[]` | `[[44,2,60,2,60,10,44,10]]` (its box as a polygon) |

- **One ring per instance.** The YOLO-seg line, `class x1 y1 x2 y2 ...` (normalised by the picture,
  six decimals like a box line), has room for one polygon, so several are joined into one ring by
  zero-width bridges: the ring walks the first polygon round from the vertex nearest the next
  polygon, crosses to that polygon's nearest vertex, walks it round, and so on, then retraces the
  same bridge vertices back to its start. Each bridge is walked once each way, so it encloses no
  area under even-odd or non-zero fill; the next polygon is always the one nearest the vertex the
  walk stands on. In the line above, the ring runs `(10,10)` round R1 and back to `(10,10)`, across
  to `(40,16)`, round R3 and back to `(40,16)`, and closes along the same bridge. An instance without
  polygons is its box's four corners as a ring.
- **COCO keeps the polygons apart.** `segmentation` holds one flat `[x1, y1, x2, y2, ...]` list per
  polygon, in pixels, and `area` their summed shoelace area.
- **A split of masks has no box line.** Ultralytics 8.4.126 refuses a label file that mixes the two
  forms (`labels mix segment and detection rows`) and a segment model refuses a dataset with fewer
  segments than boxes (`Segment dataset requires equal numbers of boxes and segments`). As soon as any
  record of a split carries polygons, the sink writes every line of the split as a ring when it
  flushes — the box-only records before it, and (with `mode: append`) the records an earlier run
  left — and gives every COCO annotation a polygon. Nothing about this is written to `data.yaml`:
  RF-DETR's YOLO reader tells a box line from a polygon line line by line, and an Ultralytics
  segment model asks for the segments.
- **RF-DETR reads it as a segmentation dataset.** Measured with RF-DETR 1.9.3 over the target
  above. Its YOLO reader (`YoloDetection(..., include_masks=True)`) gives instance 0 the two
  rectangles as `ImageDraw.polygon` fills them (302 pixels, none lost) plus 9 pixels along the
  bridge — a line one pixel wide, which is what a doubled edge fills — and gives instance 1 its box
  rectangle. Its COCO reader (`ConvertCoco(include_masks=True)`) decodes the sidecar's
  `segmentation` to 256 pixels for instance 0, exactly the polygons' `area` and no bridge, and 128
  for instance 1.

#### Reading a folder back — `raidar.YoloSource`

The sink's mirror: one split of a folder as records of `image` (HWC uint8) and `target` (`Boxes` in
pixels, the folder's names as `classes`), each record's `metadata.jsonl` row re-attached (its
`source_metadata` entries become record entries) plus `yolo_source: {root, split, image}`. Both
layouts are read; `classes:` remaps the ids by NAME to a vocabulary and refuses a name it lacks. A
polygon line is read as the instance's mask: its box is the ring's bounding box and
`target.extras["polygons"]` holds `[ring]` for it (the one bridged ring, in pixels) and `[]` for an
instance read from a box line; a record with no polygon line has no `polygons` entry. So the masks
survive a merge. Indexable, so `ConcatSource` / `DatasetSplit` / `RangeSource` wrap it — which is
how two folders merge, `examples/merge_yolo.yaml`:

```yaml
source: !class:recordstream.sources.concat.ConcatSource
  sources:
    - !class:raidar.YoloSource {root: ./runs/dataset_a, split: train, classes: {0: signal, 1: ignore}}
    - !class:raidar.YoloSource {root: ./runs/dataset_b, split: train, classes: {0: signal, 1: ignore}}
runnable: !class:recordstream.processing.DatasetProcessor
  stream: !class:recordstream.core.stream.Stream {source: !ref:source}
  sink: !class:raidar.YoloSink {output_dir: ./runs/dataset_merged, split: train, class_names: {0: signal, 1: ignore}, layout: rfdetr, mode: append, facts: [scale]}
```

```bash
recordstream run examples/merge_yolo.yaml               # then --scope val into the same folder
```

Measured: a folder of 3 pictures named `drone-signal` / `ignore` and one of 6 named `signal` /
`ignore`, merged under `{0: drone-signal, 1: ignore, 2: signal}` → 9 pictures, 44 / 2 / 71 boxes
by class, the union of both folders' picture scales in `data.yaml`, each record's origin folder in
`metadata.jsonl`.

## Compiling the RF-DETR predictor

`pip install raidar[compile]` adds the predictor's lowering for the workspace's package compiler: `RFDETRPredictor`
becomes a fixed-size torch program — the library's own bilinear resize to the model's resolution, its ImageNet
normalization, the network, its top-k selection and box scaling — with the score cut turned into a `keep` mask over
the `num_select` rows, and the `any_class` score mode applied as the predictor applies it. A detection op that holds
the predictor compiles through it; nothing in `raidar.rfdetr` changes. The lowering is held to the library's own
pieces on a stand-in network (`tests/test_lowering.py`) and, with `RAIDAR_RFDETR_CHECKPOINT` naming a checkpoint, to
the predictor itself.

## Detector catalogue

| Model | Notes |
|---|---|
| `torchvision.models.detection.fasterrcnn_*` / `retinanet_*` / `fcos_*` / `ssd*` | Used **raw** — they conform to the native forward contract directly. Registered as `detection_model` builders (below). |
| `raidar.TorchVisionDetectionModel` | The same torchvision zoo behind ONE configured wrapper — pretrained `weights` as a plain value, an eval-time `score_threshold`, a `label_indexing` knob, and a `checkpoint_path` to load a trained model back for inference (below). |
| `raidar.RFDETRModel` | RF-DETR family as a native torchvision-style model module. |
| `raidar.RFDETRPredictor` | Prediction from one of RF-DETR's own checkpoint files — any variant, the model rebuilt from what the file records; boxes, and instance masks for a `seg-*` model (below). |
| `raidar.HuggingFaceDetectionModel` | ONE light wrapper — `model_name:` reaches any Hub object-detection checkpoint (D-FINE, RT-DETRv2, DETR, Deformable-DETR, …). |
| `raidar.fcos.*` | The shared spec's five native implementations (above). |

The Ultralytics-backed YOLO26 + RT-DETR native models live in a separate package that is not
part of this distribution (Ultralytics is AGPL). Installed, it adds the `raidar.ultralytics`
subpackage, so a config names its classes as `raidar.ultralytics.<module>.<Class>`.

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

### Using a torchvision detector for inference — `TorchVisionDetectionModel`

The raw builders are the TRAINING spelling (a `!lazy:` slot the trainer flows with the
dataset-derived `num_classes`). For inference — a prediction chain, a visual review pipeline —
`raidar.TorchVisionDetectionModel` puts every decision in plain config values:

```yaml
model: !class:raidar.torchvision_model.TorchVisionDetectionModel()
  variant: fasterrcnn_resnet50_fpn     # one of the nine registered zoo members
  weights: DEFAULT                     # DEFAULT / COCO_V1 (COCO, 91 classes) / none
  score_threshold: 0.5                 # torchvision itself reports every box above 0.05
  label_indexing: zero_based           # emitted labels 0..K-1 (default) — one_based keeps 1..K
```

For a model you trained yourself, start from random init and load the checkpoint —
`weights: none` and `checkpoint_path: <a Lightning .ckpt or plain state dict>`. The head size
is read off the checkpoint itself (the class count is baked into its classification-head
shapes), so `num_classes` stays unset; set it only to build a FRESH random head without a
checkpoint, and note an explicit value that disagrees with the checkpoint fails the load
loudly. Keys retarget to the wrapped detector automatically; missing keys are logged.
`RFDETRModel` follows the same rule: `checkpoint_path` alone is enough — its head size comes
off the checkpoint's own `class_embed` shape, and with no checkpoint at all the pretrained
COCO detector keeps its own 90-class head instead of being silently re-headed. It accepts
RF-DETR's own `.pth` in either shape rfdetr writes (the older `model`/`optimizer`/`epoch` file
and the current one) — those load through rfdetr's own loader — as well as a `.ckpt` from the
lightning backend or a plain state dict.
Eval output follows the native contract after two adaptations: rows below `score_threshold`
and background rows (label 0) are dropped, and labels are renumbered per `label_indexing` —
`zero_based` matches what the other model wrappers here emit. A `weights` name the chosen
variant's zoo does not publish logs a warning and builds with random init instead (the
constructor already refuses names outside the declared set). Training still works through the
wrapper (the loss dict passes through untouched), but the example config deliberately keeps
the raw builders for that role.

### Predicting from an RF-DETR checkpoint — `RFDETRPredictor`

A checkpoint the rfdetr backend wrote (`checkpoint_best_total.pth`, `checkpoint_best_ema.pth`) records
the model it is: the variant, the resolution, the query groups (`group_detr`), the head and its class
names. `raidar.RFDETRPredictor` rebuilds the model from that record (rfdetr's own
`RFDETR.from_checkpoint`), so the checkpoint path is the only thing a config states — a seg model
trained with `model_config: {group_detr: 4}` loads without the setting repeated (a model built from
the variant's own defaults has 13 groups and refuses those weights):

```yaml
model: !class:raidar.rfdetr.RFDETRPredictor
  checkpoint_path: runs/seg_medium/checkpoint_best_total.pth
  score_threshold: 0.3          # a detection scoring at or below this is dropped
```

Two settings choose how a query is scored:

| setting | values | effect |
|---|---|---|
| `score_mode` | `class` (default) | RF-DETR's own: one detection per (query, class) pair above the cut, scored by that class — a query may answer under two classes |
| | `any_class` | one detection per query, scored `1 - prod(1 - p)` over its classes — the chance that ANY of them is right — and named by its best class |
| `any_class_excludes` | class names, default none | left out of the `any_class` score and never used as a name; a name the checkpoint lacks is refused when the model is built |

RF-DETR scores each class with its own sigmoid, so a query the model is sure holds an object but that is
torn between classes never passes a per-class cut. Measured on one query of a trained model:

```
class scores           0.274, 0.127, 0.126, 0.092, ... (28 more, together 0.373)
score_mode: class      best class 0.274 — dropped at a 0.3 cut
score_mode: any_class  0.685, named by the 0.274 class — kept at a 0.5 cut
```

`any_class` finds more objects; its name is a best guess. On a test set of 81 captures (1 724 objects) the
same model went from 182 missed / 152 false alarms (`class`, cut 0.3) to 104 / 128 (`any_class`, cut 0.5):

```yaml
model: !class:raidar.rfdetr.RFDETRPredictor
  checkpoint_path: runs/seg_medium/checkpoint_best_total.pth
  score_mode: any_class
  score_threshold: 0.5
  # any_class_excludes: [ignore]   # a class that must neither add to the score nor name a box
```

It answers the torchvision eval contract the prediction ops call — `eval()`, `to(device)`,
`model(images) -> [{boxes, scores, labels}]` — plus `masks` for a `seg-*` model:

| key | shape | meaning |
|---|---|---|
| `boxes` | `[N, 4]` float | xyxy in the INPUT picture's pixels (any size: RF-DETR resizes to its resolution and back) |
| `scores` | `[N]` float | confidence |
| `labels` | `[N]` int64 | 0-based into `predictor.class_names` (the checkpoint's own names); RF-DETR's no-object slot is dropped |
| `masks` | `[N, H, W]` bool | each instance's mask at the input picture's size — a `seg-*` checkpoint only |

A picture is `[3, H, W]` floats in `[0, 1]`, an `H x W x 3` uint8 array, or a grey one (`[H, W]`,
`[1, H, W]`) read as its three equal channels — what training's reader made of a one-channel PNG.
The file is read on the first prediction, not at construction. It predicts only: training goes
through the rfdetr backend, or `RFDETRModel` under the lightning backend.

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
