# Architecture decisions

Short decision records — **Context → Decision → Consequences → Example → What you may change** —
for the mechanisms whose existence or shape is not obvious from their names and the user docs.
Agent-facing rules live in `AGENTS.md`; this file is the human-readable rationale, and the two
cross-link but never duplicate.

---

## 1. The training backend is a config dimension, not a CLI surface (2026-08-04)

**Context.** Detection has more than one legitimate training loop: the generic Lightning path
(torchvision-native models, our steps), RF-DETR's own PTL recipe (its EMA / COCO-mAP / LR
schedule are part of what the model IS), and the AGPL ultralytics loop in a sibling package.
The first arrangement gave each loop its own surface — `raidar train` for one, a separate
`raidar-rfdetr` binary rooted at a different YAML key (`trainer:` / `evaluator:`) for another,
and the sibling's own CLI for the third — so "the same experiment on another backend" meant a
different command, a different config shape, and a different tracking story.

**Decision.** One CLI of three verbs; the backend is a `!scope:framework=<name>` block in ONE
config, activated by `--framework <name>` (rewritten to `--scope framework=…` so the flag means
one thing whether or not the config declares the dimension). A `FRAMEWORKS` table in `cli.py` is
the whole registry: name → description + best-effort registrations import. Each backend is a
merged train/eval runnable dispatching on a `task` knob through its `@entrypoint` markers. This
is the same decision the workspace's classification and segmentation projects took; the rationale
for the flag-not-subcommand half lives in their architecture docs and is not restated here.

**Consequences.** One config trains on every installed backend and the diff between backends is
readable IN the config (the scope blocks). A typo'd backend raises listing the declared ones.
The cost is discipline: every framework-specific key must stay a top-level sibling (a scope block
replaces a key wholesale), and semantic differences between backends — here, the label
convention — must be restated per block rather than inherited. `tests/test_cli_frameworks.py`
pins both config-level.

**Example.**

```yaml
runnable: !class:raidar.LightningDetector        # top level = the default backend
detection_ops:
  - !class:recordstream.ops.target.CocoToTorchVisionDetection { label_offset: 1 }
num_classes: 6

rfdetr: !scope:framework=rfdetr                  # `raidar train cfg.yaml --framework rfdetr`
  runnable: !class:raidar.RFDETRDetector { variant: nano }
  detection_ops:
    - !class:recordstream.ops.target.CocoToTorchVisionDetection { label_offset: 0 }
  num_classes: 5
```

**What you may change.** Add a backend by writing `raidar/<name>_detector.py` (a merged runnable
with a `task` knob) and a `FRAMEWORKS` row, then declaring its scope block in the example config
— nothing else. What you may NOT do: reintroduce a per-backend binary or subcommand, or nest a
framework-specific key inside `runnable:`.

---

## 2. A backend earns its module by the LOOP upstream ships (2026-08-04, revised 2026-08-05)

**Context.** RF-DETR and the Hugging Face detectors both arrive with strong upstream opinions,
and each could plausibly have become either "a model the Lightning path drives" or "a backend of
its own". The first arrangement wrapped the HF detectors into the torchvision-native forward
contract (preset classes with variant tables, internal resize/decode gymnastics) and trained
them under Lightning — which duplicated, badly, what `transformers.Trainer` already does.

**Decision.** The criterion is whether upstream ships a TRAINING LOOP worth driving — and both
answers are now "yes, drive it". RF-DETR ships `RFDETRModelModule` + `build_trainer` (EMA, the
COCO-mAP callback, best-checkpoint saving, the LR schedule: reproducing its published results
means running ITS loop), so `RFDETRDetector` wraps that loop. `transformers` ships `Trainer`
(accelerate placement, the collator protocol, the eval loop), so `HuggingFaceDetector` drives it
over ONE light generic wrapper — `HuggingFaceDetectionModel`, where a `model_name` STRING
reaches any Hub object-detection checkpoint and the checkpoint's own image processor does the
shaping. The preset classes are DELETED: a per-checkpoint wrapper class is a variant table
restating what the Hub already indexes. (`RFDETRModel` still exists as a native-contract model —
the bare LWDETR network for when you want OUR loop over RF-DETR's architecture.)

**Consequences.** Detection is a MULTI-trainer task — `(detection, trainer)` resolves to many
classes — and a discovery service must navigate that rather than assume one trainer per task.
Tracking differs honestly per loop ownership: the Lightning backend's trackers ARE its metric
sinks; the RF-DETR and HF backends' trackers carry the run RECORD while per-step curves stay
with the upstream loop's own reporting. Neither backend has a loss slot — the loss lives inside
the upstream model/loop, and a slot would accept a wired value that silently never fires.

**Example.**

```yaml
# The SAME checkpoint family, both loops:
runnable: !class:raidar.HuggingFaceDetector          # transformers.Trainer — --framework huggingface
model: !lazy:raidar.HuggingFaceDetectionModel { model_name: ustc-community/dfine-small-coco }

runnable: !class:raidar.RFDETRDetector { epochs: 50 } # RF-DETR's own loop — --framework rfdetr
```

**What you may change.** Apply the same criterion to the next arrival: upstream loop worth
driving → a `<framework>_detector.py` runnable + FRAMEWORKS row; loss-inside-forward with NO
upstream loop → a native-contract model wrapper under the torch-family backends. What you may
NOT do: reintroduce per-checkpoint preset classes, give a model wrapper its own trainer, or
re-split a merged runnable into trainer/evaluator classes.

---

## 3. The shared FCOS spec is a package, apart from the zoo wrappers (2026-08-05)

**Context.** Every detection zoo raidar wraps — torchvision, RF-DETR, `transformers` — is
torch-only. A cross-framework comparison built on zoos would compare different networks on
different engines and tell you nothing about the engines; the keras/tensorflow/jax/mlx backends
would otherwise have NO detection model at all.

**Decision.** `raidar/fcos/` is the deltaid-`unet/` analogue, extended with what detection needs
and segmentation did not: `spec.py` owns the constants, DERIVES `trainable_parameter_count`
(110,234 for 5 classes — the claim every implementation is tested against, including under
custom widths), and ships the NUMPY REFERENCE algorithms — FCOS center-sampling assignment,
dense decode + NMS, and AP50/precision/recall. The transforms (sigmoid/exp) live in the decode
in exactly ONE place, so every engine decodes identically by construction; only the LOSS is
per-engine, because it must be differentiable. Five implementations — `TorchFcos` (serving all
three torch-family backends), `KerasFcos`, `TensorFlowFcos` (raw `tf.nn`), `FlaxFcos`,
`MlxFcos` — each its own framework's model type, checked against the derivation rather than
against whichever was written first.

**Consequences.** The comparison means something: the five final losses on the same cppe-5 run
land within a factor of ~1.2 of each other. The costs: the spec needs a FIXED input size (dense
targets need one grid), so its configs carry the joint image+boxes resize; and a spec
implementation's knobs are the architecture itself — a pretrained-weights or variant knob would
break the one claim the package makes.

**Example.**

```python
from raidar.fcos import spec
assert spec.trainable_parameter_count(5) == 110_234       # what every implementation must report
maps = spec.assign_targets(boxes, labels, num_classes=5)  # the numpy reference the engines share
```

**What you may change.** A sixth implementation joins by matching the derivation
(`tests/test_fcos_spec.py::IMPLEMENTATIONS`) and speaking its backend's batch contract. What you
may NOT do: add zoo-style knobs to a spec implementation, fork the decode rules into an engine,
or "optimize" an implementation in a way that changes its parameter count.

---

## 4. Detection batches for the non-torch engines are DENSE, encoded before collation (2026-08-05)

**Context.** Detection targets are variable-length — N boxes per image. `collate_records`
handles that natively for the TORCH family (a `Regions` column keeps per-record boxes/labels,
which `batch_regions` reads back — raidar owns no collate), but the keras/tf/jax/mlx loops need
REGULAR stacking: their models take dense arrays, a ragged `[N, 4]` column is not one, and the
collate mandate forbids padding.

**Decision.** Encode EARLY: `raidar.fcos.targets.FcosTargets` runs at the END of the ops chain
and turns each record's `Regions` into the spec's fixed-shape dense maps (class one-hot, ltrb
distances, centerness) via the numpy assignment — so every batch stacks regularly and the
per-engine batch adapter only reads arrays. It **adds** those maps and destroys nothing: the
`Regions` rides along for the eval pass, because a region column never stacks anyway (its
boxes/labels are declared item attrs, which the collate lists per record) and `batch_regions`
reads it back.

**Consequences.** The four non-torch backends share one batching story and zero adapter code
beyond a transpose. The cost is a config obligation: THREE knobs must agree — `ResizeDetection`'s
`width`/`height`, the op's `input_shape`, and the model's. The op can only see two of them, and
since the `Regions` survives it (2026-08-05) it checks the one it can: the boxes' own `canvas`
against its `input_shape`, raising at the first record and naming the knobs. That is an earlier,
better-aimed error rather than a correctness fix — a real mismatch also dies in the model, which
reports tensor shapes several frames from the config lines responsible.

**The frame obligation has ONE genuinely silent failure, and it is guarded in the spec** (added
2026-08-05). The knob check above compares the frame the boxes RECORD against the frame this op
assigns against — but an image-only resize desyncs the boxes without touching either number, so
that check cannot see it by construction. The chain: an image-only resize (`ConvertToImage` and
friends) moves pixels without moving boxes, so the image matches `input_shape` — passing
`TorchFcos`'s own size check — while the boxes stay in the source frame. Nothing downstream
notices, because every SHAPE is valid; only the numbers are meaningless. Measured: every box
falls outside the grid, no cell is positive, and training proceeds against pure background with
`loss_box`/`loss_ctr` pinned at `-0.0` while `loss_cls` climbs. So `spec.assign_targets` rejects
boxes that cannot be in the frame it was handed. It sits THERE rather than in either caller
because both callers reach assignment through it — the op for the four non-torch engines, and
`TorchFcos` for the three torch ones — so one implementation covers every engine. The slack is
deliberate (a box may legitimately sit a pixel or two past an edge; annotations are not always
clipped), and it is a hard error rather than a warning because there is no recovery: the run
would train to completion and report nothing wrong. Note what
this record does NOT claim: that a detection target needs flattening to batch. It does not (see
the batching mandate) — only the DENSE MAPS need regular shapes, which is why the op adds them
beside the target instead of replacing it.

**Example.**

```yaml
detection_ops:
  - !class:recordstream.ops.target.CocoToTorchVisionDetection { bbox_format: xywh, label_offset: 0 }
  - !class:recordstream.ops.target.ResizeDetection { width: 256, height: 256 }
  - !class:recordstream.ops.torch.ToTensor { mode: RGB, normalize: true }
  - !class:raidar.fcos.targets.FcosTargets { num_classes: 5, input_shape: [256, 256, 3] }
```

**What you may change.** The op's keys (`cls_key`/`box_key`/`ctr_key`) are knobs shared with
every spec backend's ctor. What you may NOT do: pad ragged boxes through the collate, or teach
`collate_records` a detection special case — the encoding belongs to the spec, upstream of the
batch boundary.

---

## 5. The Lightning backend runs on the workspace trainer bases (2026-08-06)

**Context.** The four task projects were a four-way mirror: the same merged
train/evaluate/test/predict runnable per engine, with the Lightning modules 40–61%
byte-identical ACROSS projects (tracker wiring, trainer construction, step skeletons, epoch
reporting, sidecars, outputs) and the task logic near-identical WITHIN each project across its
engines. Both projects' own comments called the duplication deliberate-and-temporary, deferred
until the fleet existed to design a base against. By 2026-08-06 it did.

**Decision.** The trainer machinery moved UP to matrainer (its `docs/architecture.md` record 9
carries the full shape): a neutral abstract API (`TrainerRunnable`), a torch family layer
pinning the types (`TorchTrainerRunnable`), and the Lightning machinery base
(`LightningRunnable`). THIS project keeps only its task half, written once for the whole engine
fleet — `detection_base.py` (torch-free, importable by every backend) + `detection_torch.py` (the
torch-typed hook implementations) — and the concrete Lightning class becomes the composition
plus its own full flat constructor (the confluid introspection surface, signature unchanged).

**Consequences.** One task behaviour = one hook implementation instead of one copy per engine;
the fleet drift tests now check "inherits the shared base and does not shadow its wiring"
instead of per-module source patterns; matrainer is load-bearing for training (the old
"removing it keeps the class training" claim is dead by design). The other engine backends
still carry their private copies until their family's migration phase lands (root workspace
backlog).

**Example.**

```python
@configurable(task="detection", role="trainer", framework="torch")
class LightningDetector(DetectionTorchHooks, LightningRunnable):
    def __init__(self, model, train_set=None, ..., num_classes=None):
        super().__init__(model=model, train_set=train_set, ...)  # shared slots
        self.num_classes = num_classes                                          # task-specific state
```

**Amendment (2026-08-07).** The bare-torch backend joined the same shape: ``TorchDetector`` is `(DetectionTorchHooks, matrainer.torch.TorchLoopRunnable)` — the same task hooks, a different machinery base, plus the three BATCH-hook overrides (`_train_batch_loss` summing the forward's loss dict, `_eval_batch` returning `None` because eval mode yields predictions rather than losses, `_predict_batch`) around `native_batch`, with `collate_fn=collate_list` declared at the base ctor — the loop-base analogue of the Lightning concrete's four step-function overrides.

**What you may change.** Hook implementations and task state freely; the concrete signature
(it is the config surface). What you may not do: shadow the inherited machinery
(`_build_trainer` / `_live_trackers` / `_run_eval` / `_report_epoch_metrics`) — extend through
the hooks, or change the base in matrainer with every consumer updated in the same workspace
change.
