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
default_scopes: [framework=lightning]            # what a bare `raidar train cfg.yaml` picks
detection_ops:                                   # the torchvision-family defaults every block inherits
  - !class:recordstream.ops.target.CocoToTorchVisionDetection { label_offset: 1 }
num_classes: 6

lightning: !scope:framework=lightning            # the default, a block like every other
  runnable: !class:raidar.LightningDetector

rfdetr: !scope:framework=rfdetr                  # `raidar train cfg.yaml --framework rfdetr`
  runnable: !class:raidar.RFDETRDetector { variant: nano }
  detection_ops:
    - !class:recordstream.ops.target.CocoToTorchVisionDetection { label_offset: 0 }
  num_classes: 5
```

**What you may change.** Add a backend by writing `raidar/<name>_detector.py` (a merged runnable
with a `task` knob) and a `FRAMEWORKS` row, then declaring its scope block in the example config.
The fleet tests name the rest: an entry point in `pyproject.toml`, rows in `BACKENDS` and
`METRIC_BRIDGES` in `tests/test_tracking_parity.py`, and a row in `examples/sweep/sweep.yaml`. What
you may NOT do: reintroduce a per-backend binary or subcommand, or nest a framework-specific key
inside `runnable:`.

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
handles that natively for the TORCH family (a `Boxes` column keeps per-record boxes/labels,
which `batch_boxes` reads back — raidar owns no collate), but the keras/tf/jax/mlx loops need
REGULAR stacking: their models take dense arrays, a ragged `[N, 4]` column is not one, and the
collate mandate forbids padding.

**Decision.** Encode EARLY: `raidar.fcos.targets.FcosTargets` runs at the END of the ops chain
and turns each record's `Boxes` into the spec's fixed-shape dense maps (class one-hot, ltrb
distances, centerness) via the numpy assignment — so every batch stacks regularly and the
per-engine batch adapter only reads arrays. It **adds** those maps and destroys nothing: the
`Boxes` rides along for the eval pass, because a region column never stacks anyway (its
boxes/labels are declared item attrs, which the collate lists per record) and `batch_boxes`
reads it back.

**Consequences.** The four non-torch backends share one batching story and zero adapter code
beyond a transpose. The cost is a config obligation: THREE knobs must agree — `ResizeDetection`'s
`width`/`height`, the op's `input_shape`, and the model's. The op can only see two of them, and
since the `Boxes` survives it (2026-08-05) it checks the one it can: the boxes' own `canvas`
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
  - !class:recordstream.ops.image.ConvertMode { mode: RGB }
  - !class:recordstream.ops.numpy.Scale {}
  - !class:recordstream.ops.torch.ToTensor {}
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
joined family by family through 2026-08-08, when the migration COMPLETED — every backend
now composes its task hooks with a shared machinery base (see the closing amendment below).

**Example.**

```python
@configurable(task="detection", role="trainer", framework="torch")
class LightningDetector(DetectionTorchHooks, LightningRunnable):
    def __init__(self, model, train_set=None, ..., num_classes=None):
        super().__init__(model=model, train_set=train_set, ...)  # shared slots
        self.num_classes = num_classes                                          # task-specific state
```

**Amendment (2026-08-07).** The bare-torch backend joined the same shape: ``TorchDetector`` is `(DetectionTorchHooks, matrainer.torch.TorchLoopRunnable)` — the same task hooks, a different machinery base, plus the three BATCH-hook overrides (`_train_batch_loss` summing the forward's loss dict, `_eval_batch` returning `None` because eval mode yields predictions rather than losses, `_predict_batch`) around `native_batch`, with `collate_fn=collate_list` declared at the base ctor — the loop-base analogue of the Lightning concrete's four step-function overrides.

**Amendment (2026-08-07, second).** The fastai backend joined: ``FastaiDetector`` is `(DetectionTorchHooks, matrainer.fastai.FastaiRunnable)` — the no-op-loss bridge now rides the base's Learner seams (`_learner_model` wraps the adapter, `_learner_loss` returns the pass-through, `_learner_callbacks` prepends the metrics callback, `_place` pins the device) and the family's torchmetrics builders are RESTORED over the fastai base's no-ops — the loss-inside-forward analogue of the loop base's three batch-hook overrides.

**Amendment (2026-08-07/08, closing — P5 + P6, the migration completes).** The remaining five
in-core backends joined and the fleet is DONE. P5 (keras family): ``KerasDetector`` and
``TensorFlowDetector`` compose `detection_keras.py`'s hooks (the dense-map collate, the spec's
positional loss list, the numpy AP path behind ONE `_batch_predictor` seam) with the two
keras-family machinery siblings. P6: ``HuggingFaceDetector`` is ``(DetectionBase,
matrainer.huggingface.HuggingFaceRunnable)`` — the processor collate, the seed-before-build
ordering, the RESTORED family torchmetrics builders and the decode+torchmetrics scoring behind
the machinery's eval seams — and ``JaxDetector`` / ``MlxDetector`` are ``(DetectionBase,
matrainer.jax.JaxLoopRunnable)`` / ``(…, matrainer.mlx.MlxLoopRunnable)``, their 4-arity
bare-loss spec steps driving the machinery's splat-tolerant epoch drivers and the numpy AP
riding `_fit_scores`/`_evaluate_split`. The seeded spec cells reproduced their canonical losses
BIT-EXACTLY through every new base (keras 2.30023455619812, tensorflow 2.3113211567203202,
jax 2.1331496040026345, mlx 1.953120619058609) — the strongest refactor evidence the fleet
records. (The keras canonical later moved to 2.285114049911499 — matrainer's `shuffle=False`
fix of 2026-08-08, which made the cell tracker-neutral; see record 8 and matrainer
`docs/architecture.md` record 9 amendment 6.)

**What you may change.** Hook implementations and task state freely; the concrete signature
(it is the config surface). What you may not do: shadow the inherited machinery
(`_build_trainer` / `_live_trackers` / `_run_eval` / `_report_epoch_metrics`) — extend through
the hooks, or change the base in matrainer with every consumer updated in the same workspace
change.

## 6. The torchvision zoo has TWO spellings — raw builders for training, ONE wrapper for inference (2026-08-31)

**Context.** The zoo's builder functions are registered as `detection_model` configurables and
serve the trainers well: a `!lazy:` slot the runnable flows with the dataset-derived
`num_classes` at `run()`. But three inference decisions cannot be stated on a builder as plain
config values: the pretrained `weights` parameter is a torchvision **enum object** (in a
generated form it renders as a wired object socket with no producer — so a canvas-drawn
fasterrcnn silently came up with RANDOM weights and detected nothing); torchvision reports
every box above its internal 0.05 (measured on a real cppe-5 frame: 11 detections, scores
0.95 down to 0.27 — noise in any review surface); and the zoo emits 1-indexed foreground
labels (background reserved at index 0) while the workspace's other detection wrappers
(RF-DETR, the Hub wrapper) emit 0-indexed, so cross-model label comparisons were off by one.
There is also no builder knob for loading a TRAINED checkpoint back for inference.

**Decision.** One wrapper class, `raidar.torchvision_model.TorchVisionDetectionModel`,
mirroring `RFDETRModel`'s shape: partial-init (`__init__` stores config, `solidify()` builds
once, checkpoint load LAST so its values win), closed `Literal` knobs (`variant` — the same
nine builders `torchvision_detection.py` registers; `weights` — `none`/`DEFAULT`/`COCO_V1`,
verified identical across all nine zoo enums; `label_indexing` — `zero_based`/`one_based`),
and an eval decode that drops sub-threshold and background rows and renumbers labels. A
`weights` name the variant's enum lacks (future torchvision skew — the constructor already
refuses names outside the Literal) WARNS and builds with random init, because the
checkpoint-load flow legitimately follows. The raw builders stay registered and stay the
training spelling — the example config is untouched.

**Consequences.** Two spellings for one zoo, deliberately: the builder is the trainer's slot,
the wrapper is the inference slot (it also trains — the loss dict passes through — but nothing
ships configured that way). `zero_based` is the default, so a wrapper-emitted label matches
the other wrappers' vocabulary with no extra op; the cost is that COCO-pretrained output uses
0-indexed COCO ids (person = 0, not the familiar 1) — `one_based` restores the native numbering.

**Example.** The measured before/after on one cppe-5 frame (fasterrcnn_resnet50_fpn, COCO
weights): raw forward — 11 detections, labels `[1, 1, 1, ...]`, scores `[0.95, 0.85, 0.57,
0.44, 0.27, ...]`; through the wrapper (`score_threshold: 0.5`, `zero_based`) — 3 detections,
labels `[0, 0, 0]`, scores `[0.95, 0.85, 0.57]`, boxes unchanged in the original frame.

```yaml
model: !class:raidar.torchvision_model.TorchVisionDetectionModel()
  variant: fasterrcnn_resnet50_fpn
  weights: none                                 # random init …
  checkpoint_path: runs/frcnn_cppe5/best.ckpt   # … head sized FROM the checkpoint, weights loaded
```

(Amended 2026-08-31, user question "why do we need to set num_classes for a trained
checkpoint": the class count is BAKED INTO a checkpoint's classification-head shapes, so
`_checkpoint_num_classes` reads it per variant — Faster R-CNN's linear head is `[K, …]`,
RetinaNet's cls_logits conv `[9K, …]`, FCOS `[K, …]`, SSD300 level-0 `[4K, …]`, SSDLite
level-0 `[6K, …]`, all measured — suffix-matched so Lightning's `model.` and the wrapper's
`detector.` prefixes both resolve. An explicit `num_classes` still wins, and a disagreement
fails the load loudly — `load_state_dict` raises on a size mismatch even with
`strict=False`, measured.)

**What you may change.** The decode's vocabulary (a new knob appends to the signature — the
widget order is APPEND-ONLY once shipped) and the `Literal` sets when torchvision's zoo moves.
What you may not do: make the label shift unconditional again (the knob was a user decision,
2026-08-31), turn the weights-skew warning into an error, or grow zoo-style knobs on the spec
models (record 3).

(Amended 2026-08-31, user ruling "only use checkpoint_path": `checkpoint_strict` is REMOVED
from BOTH wrappers — loads are always `strict=False`, where missing keys are logged and a
head-size disagreement still raises on its own — and the baked-in `num_classes` landed on
`RFDETRModel` too: unset, it reads the checkpoint's own `class_embed` rows − 1, and with no
checkpoint keeps the pretrained COCO detector's 90 instead of silently re-heading it to a
random 1-class head. Measured end to end on a real trained checkpoint (`last_ema.pth`,
cppe-5): `checkpoint_path` alone → 3 of 5 annotations matched at mean IoU 0.927, 0 FP, where
the same graph with the old explicit `num_classes: 1` died on a state-dict size mismatch.)

## 7. RF-DETR's own checkpoints load through RF-DETR's own loader (2026-09-27)

**Context.** `RFDETRModel(checkpoint_path=…)` loaded every file through matrainer's generic
checkpoint loader, which recognises the generic shapes: a Lightning `.ckpt` (`state_dict`), an
Ultralytics `.pt` (a live module under `model`), a plain state dict. RF-DETR writes TWO shapes
of its own. The current one (rfdetr 1.9: `model` + `args` + a Lightning-style `state_dict`)
loaded by luck, through its `state_dict` key. The older one — `{model, optimizer,
lr_scheduler, epoch, args}`, the shape of `rf-detr-nano.pth` and of any checkpoint an older
rfdetr trained — has no `state_dict`, so the generic loader took the five top-level entries AS
the state dict, matched none of them, and the model ran on random weights with nothing but its
routine INFO lines (`checkpoint missing 468 key(s)`) to show for it.

**Decision.** The wrapper routes by the file's own shape (`_is_rfdetr_checkpoint`): a top-level
`model` MAPPING is RF-DETR's, and it is handed to the library as the weights to build from —
the `pretrain_weights` route the rfdetr runnable's evaluation already uses, which reads both
shapes and aligns the head. Anything else (the lightning backend's `.ckpt`, a plain state dict)
keeps the generic loader, which knows the wrapper's `model.` / `detector.` prefixes. Rejected
(user ruling 2026-09-27): teaching the generic loader RF-DETR's shapes, or making a load that
matches nothing raise there — a library's format belongs to the wrapper that owns the library,
and the generic loader's INFO line stays what it is.

**Consequences.** The checkpoint's values still win: an RF-DETR file replaces the COCO weights
at build time instead of overwriting them after. The library would quietly re-size a head that
disagrees with an explicit `num_classes`; the wrapper refuses that first (`RuntimeError` naming
both counts), keeping record 6's "an explicit value that disagrees fails the load". The file is
read once more to decide the route — measured 0.01–0.06 s for a 366 MB checkpoint.

**Example.** Measured on the real files, loading into `RFDETRModel(checkpoint_path=…)`:

| file | before | after |
|---|---|---|
| `rf-detr-nano.pth` (older RF-DETR shape) | 1 of 465 tensors | 465 of 465 |
| `runs/rfdetr_cppe5/checkpoint_best_total.pth` (current shape) | 466 of 466 | 466 of 466, through the library |
| lightning backend `.ckpt` / plain state dict | loads | loads, through the generic loader as before |

```yaml
model: !class:raidar.rfdetr.RFDETRModel()
  variant: nano
  checkpoint_path: rf-detr-nano.pth   # RF-DETR's own file -> rfdetr's loader; head size read off it
```

**What you may change.** The shape test, if RF-DETR ships a third checkpoint shape. What you may
not do: send an RF-DETR file back through the generic loader, or move RF-DETR's shapes into it.
Pins: `tests/test_rfdetr.py` (`test_every_checkpoint_shape_loads_every_weight`,
`test_rfdetr_files_go_to_the_library_and_the_rest_to_the_generic_loader`,
`test_an_explicit_count_the_file_disagrees_with_fails_loudly`,
`test_only_a_top_level_model_mapping_marks_an_rfdetr_file`).

---

## 8. Every random draw that shapes the network is seeded before the build (2026-08-06, extended 2026-08-09)

**Context.** A backend that cannot reproduce itself cannot confirm a refactor, and six of
them could not:

- The keras FCOS cell had no seed. Three runs of the identical config gave final train/loss
  2.27, 2.18 and 2.12.
- `huggingface` and `rfdetr` did seed, but too late. `transformers.Trainer` applies
  `TrainingArguments.seed` when training starts, and RF-DETR's module seeds in `on_fit_start`.
  Both happen after the wrapper has built the network and randomly initialized its
  `num_classes` head. The data order was seeded and the weights were not. Two runs of one
  config gave:

  | backend | run 1 | run 2 |
  |---|---|---|
  | rfdetr, mAP50:95 | 0.3156 | 0.3819 |
  | huggingface, val/map_50 | 0.0685 | 0.1302 |

- The torchvision builders re-initialize their head on every build, and no framework covered
  that draw. Identical sweeps moved train/loss: torch 0.5115 → 0.5143, lightning
  0.5366 → 0.5246, fastai 0.5181 → 0.5430.

**Decision.** Seed at BUILD time, in each framework's own idiom, and do not restore the RNG
afterwards. This is a run-level seed applied at the earliest useful moment, so it also covers
the loader shuffle.

| what | knob | applied |
|---|---|---|
| each FCOS implementation | `seed: int = 0` | before its build: `torch.manual_seed`, `keras.utils.set_random_seed` (Keras has no per-layer seed), per-initializer `stateless_uniform` (raw TF), `nnx.Rngs`, `mx.random.seed` |
| lightning · torch · fastai | `seed: int = 0` on each flat constructor | once, in `DetectionTorchHooks.initialize()`, before the task base builds the model |
| huggingface | `seed: int = 42` (the transformers default) | `transformers.set_seed` before the build; the same value joins the `TrainingArguments` floor |
| rfdetr | `train_config: {seed: …}` (RF-DETR defaults it to `None`) | `seed_everything(seed, workers=True)` at the top of both fit paths; no seed, no call |

The FCOS seed and the torchvision-family seed are separate on purpose: the first binds the
architecture comparison, the second binds the run.

**Consequences.**

- **Runs reproduce.** Paired 1-epoch runs of torch, lightning and fastai matched on every
  metric. torch at 5 epochs landed on train/loss 0.5018 / val/map_50 0.3485 twice, with
  `metrics.csv` identical. huggingface landed on train/loss 40.41484375 twice.
- **rfdetr reproduces on cpu only.** Seeded, two cpu runs gave mAP50:95 0.355725 both times;
  two mps runs gave 0.29446 and 0.333115. That residual is Apple's MPS backend, not the seed.
  Pin `device: cpu` on the rfdetr block for a run you will compare against itself.
- **Seed first, then vary the device.** A cpu-vs-mps comparison made before the seeding fix
  cannot isolate the device, because the unseeded head dominates both arms. One such
  comparison (0.3777 / 0.3453 on cpu) was briefly read as clearing the device. The seeded runs
  above show mps does add variance.
- **The FCOS cells are bit-comparable across a refactor.** The seeded sweep of the example
  config (5 epochs, the 96-row cppe-5 slice), 2026-08-06; the keras row as re-measured
  2026-08-08, identical under the csv and mlflow trackers and across reruns:

  | engine | final train/loss | final val/map_50 |
  |---|---|---|
  | keras | 2.285114049911499 | 0.00020625352105024217 |
  | tensorflow | 2.3113211567203202 | 0.001234 |
  | jax | 2.1331496040026345 | 0.001068 |
  | mlx | 1.953120619058609 | 0.002274 |

  tensorflow, jax and mlx reproduced these through the whole base migration (record 5). The
  keras number was 2.30023455619812 until matrainer's Keras base stopped letting Keras
  shuffle batch order through Python's global `random` (`shuffle=False`; matrainer
  `docs/architecture.md` record 9, amendment 6). A tracker that drew from that global RNG had
  shifted the order. Keras numbers from before that change are not comparable with later ones.

**Example.**

```yaml
lightning: !scope:framework=lightning
  runnable: !class:raidar.LightningDetector
    seed: 0                          # the default; torch.manual_seed before the model is built
keras: !scope:framework=keras
  model: !lazy:raidar.fcos.keras_fcos.KerasFcos
    input_shape: [256, 256, 3]
    seed: 7                          # a different initialization of the same network
rfdetr: !scope:framework=rfdetr
  runnable: !class:raidar.RFDETRDetector
    train_config: { seed: 7 }        # without it RF-DETR seeds nothing
    device: cpu                      # reproducible; mps is faster but not reproducible
```

**What you may change.** You may add `seed` to a new backend or model whose build draws random
numbers, applied before the build in that framework's own idiom. Do not restore the RNG after
the build, do not move the torchvision-family seed out of the shared hook, and do not add a
seed to a wrapper that only loads weights. Pins:
- `tests/test_fcos_spec.py::test_every_implementation_is_SEEDED`
- `tests/test_fcos_spec.py::test_the_keras_seed_actually_reproduces_the_weights`
- `tests/test_torch_detector.py::TestTheSeedCoversTheModelBuild`
- `tests/test_huggingface_detector.py::TestTheRunIsSeeded`
- `tests/test_rfdetr_detector.py::TestTheSeedCoversTheMODEL`

---

## 9. An instance's mask is ONE bridged ring, and a split holds one label form (2026-10-07)

**Context.** A producer that knows an instance's shape hands the sink its polygons, and an
instance can be several separate pieces. The two targets the folder must feed hold less:

- A YOLO-seg label line is `class x1 y1 x2 y2 ...` — ONE polygon per line.
- Ultralytics 8.4.126 refuses a label file that mixes the two forms (`labels mix segment and
  detection rows`) and a segment model refuses a dataset with fewer segments than boxes
  (`Segment dataset requires equal numbers of boxes and segments`).
- RF-DETR 1.9.3's YOLO reader tells a box line (5 fields) from a polygon line (7 or more) line by
  line, rasterizes a polygon with `ImageDraw.polygon(fill=1)` and reads a box line as its
  four-corner polygon; its COCO reader decodes a list of polygons per instance to their union.

Writing one line per piece would turn one instance into several: the count, the class
statistics and every metric that matches instances to predictions would change with it. Writing
only COCO would leave the YOLO folder without masks.

**Decision.** Per instance, ONE ring. Several polygons are joined by zero-width bridges: walk the
first polygon round from the vertex nearest its closest neighbour and back to that vertex, cross
to the neighbour's nearest vertex, walk it round and back, and so on, each next polygon being the
one nearest the vertex the walk stands on; then retrace the bridge vertices in reverse to the
start. A bridge is walked once each way, so it encloses no area under even-odd or non-zero fill.
The COCO sidecar needs no such trick and keeps the polygons apart, with `area` their summed
shoelace area. An instance without polygons is its box's four corners, in both. The form is
decided for the SPLIT, not the record: as soon as one record carries a polygon, every line (and
every COCO annotation) becomes a ring when the sink flushes, the earlier box-only records and the
ones an `append` continues included. Nothing is written to `data.yaml` — neither reader asks for a
marker. The ring code is written for this sink; it copies no Ultralytics code (AGPL).

**Consequences.**

- **The bridges light a pixel line.** Measured with RF-DETR's YOLO reader on two rectangles with
  one bridge: 302 pixels in the union of the rectangles, 311 in the ring's mask — nothing lost,
  9 pixels along the bridge. The COCO reader's mask of the same instance is exactly the polygons'
  `area`, 256 pixels, with no bridge.
- **A round trip keeps the ring, not the pieces.** `YoloSource` returns the ring as the instance's one
  polygon, so a merge writes COCO `segmentation` as that one ring. Its `area` is unchanged (the bridges
  add none); the pieces are not recoverable from the YOLO folder.
- **Coordinates are six decimals**, like a box line; RF-DETR clips polygon points to the picture.
- **An `append` reads the split's label files at flush** to learn whether the earlier records are
  rings. A `replace` run without polygons reads none and writes what it always wrote.

**Example.**

```python
from raidar import POLYGONS
from recordstream import Boxes

target = Boxes(
    boxes=[[2, 2, 56, 28], [44, 2, 60, 10]],
    labels=[1, 0],
    extras={POLYGONS: [
        [[[2, 2], [10, 2], [10, 10], [2, 10]], [[40, 16], [56, 16], [56, 28], [40, 28]]],  # two pieces
        [],                                                                              # its box
    ]},
)
# 64 x 32 picture, labels/<split>/000000.txt:
# 1 0.156250 0.312500 0.031250 0.312500 0.031250 0.062500 0.156250 0.062500 0.156250 0.312500 0.625000 0.500000 0.875000 0.500000 0.875000 0.875000 0.625000 0.875000 0.625000 0.500000
# 0 0.687500 0.062500 0.937500 0.062500 0.937500 0.312500 0.687500 0.312500
```

**What you may change.** You may pick the bridge vertices and the order of the pieces differently,
as long as the ring stays closed, its first point is not repeated, every bridge is walked twice
and the polygons' union is what it fills. Do not write one line per piece, do not leave a box line
in a split that holds a ring, do not add a `data.yaml` marker or a sink knob for it, and do not
change what a record without the key writes. Pins:
- `tests/test_yolo_sink.py::TestBoxOnlyRecordsAreUnchanged`
- `tests/test_yolo_sink.py::TestSeveralPolygonsOfOneInstance`
- `tests/test_yolo_sink.py::TestOneFormPerSplit`
- `tests/test_yolo_sink.py::TestRfdetrReadsTheFolderAsSegmentation`
- `tests/test_yolo_source.py::TestPolygons`
- `tests/test_yolo_source.py::TestMasksSurviveTheRoundTrip`

---

## 10. Prediction from RF-DETR's own checkpoint is a class of its own, described by the file (2026-10-11)

**Context.** A `seg-*` model the rfdetr backend trained could not be loaded to predict. `RFDETRModel`, the
lightning backend's model, offers the four detection variants on purpose (it builds box targets and no masks, so a
seg variant would train nothing it can score), and builds from the variant's defaults: measured on a seg-medium
trained with `model_config: {group_detr: 4}` —

```
RFDETRModel(variant="seg-medium", ...)   → ValidationError: Input should be 'nano', 'small', 'medium' or 'large'
(the variant forced past that check)     → size mismatch for refpoint_embed.weight: [800, 4] in the checkpoint,
                                            [2600, 4] in the model (13 query groups, the variant's default)
```

Nothing returned a mask, either. But the checkpoint records the model it is — `model_name` (`RFDETRSegMedium`),
`model_config` (resolution 1008, `group_detr` 4, 33 classes), the class names — and `rfdetr`'s own
`RFDETR.from_checkpoint` rebuilds it from that record.

**Decision.** A prediction-only class, `RFDETRPredictor(checkpoint_path, score_threshold)`, beside `RFDETRModel` in
`rfdetr.py`: built from the file through `from_checkpoint` at the first prediction, it predicts through the library's
own `predict` (its resize, normalisation and post-processing) and answers the torchvision eval contract plus `masks`,
everything in the INPUT picture's pixels. It takes no architecture setting at all. Rejected: opening `RFDETRModel`'s
variants to `seg-*` with a `model_config` pass-through — it breaks the training wrapper's box-only rule, makes a
config restate settings the file already holds (and fail when they drift), and still returns no masks; and a
`variant` knob on the predictor — the file names its class.

**Consequences.** A config names a checkpoint and a score cut, nothing else. Any input size: RF-DETR resizes to its
resolution (or, for a model trained keeping the aspect, to that aspect — §13) and maps everything back, so a pipeline feeds the picture it trained on (a 1024 px picture to a 1008 px
model) instead of rendering at the model's size. It is not an `nn.Module`: the network is the library's, held by its
`RFDETR` object, and `to()` moves that object's device too — its `predict` places the input by it. A Lightning
`.ckpt` of rfdetr <= 1.11 carries no `args` and is refused by the library: predict from the `.pth` files.

**Example.** Measured on that seg-medium on a 1024 x 1024 validation picture of an LTE capture (score cut 0.3, CUDA,
0.05 s a picture warm):

```yaml
model: !class:raidar.rfdetr.RFDETRPredictor
  checkpoint_path: runs/signal_detection/a/checkpoint_best_total.pth
  score_threshold: 0.3
```

```
RFDETRSegMedium, resolution 1008, group_detr 4, 33 class names — all read off the file
boxes [7, 4] · scores [7] · labels [7] · masks [7, 1024, 1024] bool
  label 1 0.97 box [343, 0, 590, 1024]  mask 200 740 px
```

**What you may change.** The input forms `_rgb_picture` accepts, and the output keys, appending only. What you may
not do: give it an architecture setting, or teach `RFDETRModel` the seg variants to predict masks.
Pins: `tests/test_rfdetr_predictor.py`.

## 11. A query scored by ANY of its classes rewrites the logits RF-DETR ranks (2026-10-12)

**Context.** RF-DETR scores every class of a query with its own sigmoid and its post-processing keeps the
(query, class) pairs above the cut. A model that has learnt WHERE objects are but not always WHAT they are spreads
a query over several classes, none over the cut: measured on a signal-detection model, a query read 0.274, 0.127,
0.126, 0.092 and 28 smaller scores — dropped at 0.3, though `1 - prod(1 - p)` is 0.685. Across 81 captures, 115 of
1 724 objects had no detection at all, while lowering the cut to 0.05 found every object at 125 false alarms on one
capture, most of them one query repeated under several classes.

**Decision.** `RFDETRPredictor(score_mode="any_class")` keeps ONE detection per query, scored by its any-class
chance and named by its best class. It is done by wrapping the post-processing call that `RFDETR.predict` makes
(`self.model.postprocess(outputs, target_sizes=..., score_threshold=...)`): before the library's own step runs,
each query's logits become one slot — its best counted class — holding `logit(1 - prod(1 - p))`, every other slot
`-inf`. The library then selects, scales the boxes, cuts and upsamples the masks exactly as for real logits.
Rejected: overriding `PostProcess._select_topk` (a private method, renamed or reshaped across releases — this rewrite
depends only on the `pred_logits` key and the call's signature); re-implementing `predict` past the network (its
resize, normalisation, COCO-id mapping and mask path would be a copy to keep in step); and summing or taking the max
of the class scores (the max is `class` mode again; a sum is no probability and passes 1).

**Consequences.** `class` stays the default and gives the library's answer unchanged. In `any_class` mode a query
never answers twice and its score is never below its best class's, so at one cut it keeps every query `class` mode
keeps and possibly more; the name of a kept query may be wrong, and only the any-class score is reported. The
no-object slot never counts; `any_class_excludes` leaves named classes out the same way (an unknown name is refused
at the build, naming the checkpoint's classes). The sum is taken as `-expm1(sum(log1p(-p)))`, so 32 small scores
do not underflow and a certain class (`p = 1`) gives 1, not NaN.

**Example.** The query above, and the test set's totals (cut 0.3 per class, 0.5 any class):

```yaml
model: !class:raidar.rfdetr.RFDETRPredictor
  checkpoint_path: runs/signal_detection/a/checkpoint_best_total.pth
  score_mode: any_class
  score_threshold: 0.5
```

```
query     class 0.274 (best) → dropped        any_class 0.685 → kept, named by the 0.274 class
81 captures, 1 724 objects    class:     1 640 found, 115 missed, 200 false alarms, 1 097 accurate
                              any_class: 1 697 found,  73 missed, 199 false alarms, 1 158 accurate
```

**What you may change.** The aggregation (`_any_class_logits`) for another probability rule, keeping one slot per
query; the modes, appending to `ScoreMode`. What you may not do: change `class` mode's answer, or patch the library's
post-processing in place. Pins: `tests/test_rfdetr_predictor.py::TestTheAnyClassRewrite`, `::TestAnyClassScoring`.

## 12. A picture is cut to one piece before RF-DETR's resize; stream mode reads masks and labels per item (2026-10-12)

**Context.** An RF-DETR seg model's batch pads its pictures to one size but joins their masks as they come, so a batch
of pictures of different shapes stops the run (`RuntimeError: Sizes of tensors must match except in dimension 0.
Expected size 564 but got size 1333`). Pictures of a whole capture — 504 columns of band, as many rows as the capture
has time, 100 to 7 143 rows — never share a shape. RF-DETR's own levers cannot fix it, measured on such pictures:
its square resize (the default) makes every picture one shape by squeezing it (1 008 rows to 564); with
`square_resize_div_64: false` the shape is kept but its zoom-crop (`scale_jitter`) still cuts a SQUARE beside the
uncut 1:2 pictures (`Expected size 1128 but got size 564`); a custom crop in `aug_config` runs AFTER its resize,
which caps the long side at 1 333 px — a 504 x 3 000 picture reaches it 224 px wide, and any crop wider than that
fails (`CropSizeError`). Stream mode gave no masks at all (`ConvertCoco(include_masks=False)`), and read every
target once at construction (a projection walk) and reused it each epoch.

**Decision.** `raidar.crop.RandomCropDetection` cuts each picture to one `height` x `width` piece in its OWN pixels,
in the stream, so RF-DETR only ever sees squares and keeps its defaults (square resize, zoom-crop, flip): boxes
moved and clipped (a box under `min_size_px` dropped), outlines clipped exactly (Sutherland–Hodgman against the
piece), a smaller picture padded with its median, never stretched — the tiled pictures' rule. `random` placement
draws from Python's `random`, which the data loader seeds per worker and per epoch; `centre` is validation's.
`StreamRFDETRDataset` reads a record's labels in the same read as its picture and, for a seg model
(`segmentation_head`), hands RF-DETR each instance's outlines as COCO polygons (its box rectangle when it has none),
rasterised by RF-DETR's own `ConvertCoco`, so the masks follow its later resize and crop as the disk path's do. The
step lives in raidar because it is an augmentation (recordstream ships none) and the outline contract is raidar's.

**Consequences.** A folder of any picture heights trains a seg model: one epoch of seg-nano at 504 over 130 + 65
pictures 504 wide and 100 / 1 008 / 3 000 tall wrote its mask losses and `val/segm_mAP_*` (2026-10-12). A model
trained on pieces should be RUN on pieces — the review workspaces still run it on the whole picture (a follow-up
in the consuming project's backlog). Stream mode starts sooner (nothing is walked up front); a picture is decoded once per item,
as before.

**Example.**

```python
import random
from raidar.crop import RandomCropDetection

crop = RandomCropDetection(height=504, width=504, placement="random")
random.seed(0)
piece = crop(record)          # record: YoloSource's {image: 504 x 3000, target: Boxes + extras["polygons"]}
piece["image"].data.shape     # (504, 504, 3); piece["target"].canvas == (504, 504)
```

**What you may change.** The placements (a new one is a `Placement` value with its test), `min_size_px`, the pad
value with a measurement. Not: a crop moved into RF-DETR's `aug_config` (it runs after the resize and its cap), a
stream that caches labels apart from the picture it read, or stretching a picture to the piece.

(2026-10-14) The zoom-crop the context names became a lever: the stream now passes `scale_jitter` to RF-DETR's
transforms, so pictures of ONE shape that keep their aspect train with it off (§13).

## 13. The checkpoint says how its pictures were resized; the stream passes the zoom-crop switch (2026-10-14)

**Context.** A detection set's pictures became 504 x 1008 — the whole capture, the band's bins as columns — trained
keeping that aspect (`square_resize_div_64: false`, short side at a 504 resolution). Two things stood in the way.
Training: one epoch of seg-nano stopped in RF-DETR's mask matcher (`torch.cat` of the batch's masks: `Expected size
372 but got size 744`) — its zoom-crop (`scale_jitter`, on by default) cuts a SQUARE from half the pictures and the
other half keep their 1:2 shape, and the batch pads pictures but not masks. RF-DETR's own folder loader passes
`scale_jitter` from its config to the transforms; the stream's builder, which mirrors that loader, did not, so
`train_config: {scale_jitter: false}` changed nothing. Measured on a 504 x 1008 picture at 504: on, 1248 x 624 x55
and 624 x 624 x45 of 100 draws; off, 1248 x 624 x100. Prediction: `RFDETR.predict` resizes to the square of the
resolution unless handed a `shape`, so the model trained at 1008 x 504 was shown 504 x 504. Whether a model trained
keeping the aspect is recorded in the checkpoint's `args` only — `from_checkpoint` builds the loaded model's args
from its model config, and they report the library's default (True) whatever the file says (measured, rfdetr 1.13).

**Decision.** (1) The stream's transform builder passes `scale_jitter` (default on, as RF-DETR's loader does). (2)
`RFDETRPredictor` reads `square_resize_div_64` off the file's `args` when it builds — a tensor-free read
(`weights_only`, memory-mapped: 0.02 s on a 142 MB checkpoint) — and resizes each picture as that training validated:
the square of the resolution, or the short side at the resolution with the long side scaled and capped at 1 333 px,
both rounded UP to whole blocks of `patch_size x num_windows` (where validation padded the batch; `predict` takes only
whole blocks). Pictures needing different sizes go in one `predict` call per size, their answers back in order. The
lowering resizes through the same `input_size`. Rejected: a `shape` setting on the predictor — an input size is the
model's training, which the file records, and a setting could disagree with it (§10's rule, no architecture setting).

**Consequences.** A model trained keeping the aspect needs no setting to run: its package hands the network a 1008 x
504 picture at 1008 x 504 (624 x 312 for a 312 smoke model, measured end to end 2026-10-14). A square-trained
checkpoint, and every file without the key, predicts as before. Turning the zoom-crop off loses that augmentation for
the configs that do; a set of pictures of SEVERAL shapes still needs the piece cut (§12). The compiled lowering reads
the picture's size at the trace, so a compiled package keeps the picture size it was compiled at — the picture step
fixes it.

**Example.**

```yaml
train_config:
  square_resize_div_64: false   # the aspect kept; recorded in the checkpoint, read back by the predictor
  scale_jitter: false           # no square zoom-crop beside the 1:2 pictures
```

```python
predictor = RFDETRPredictor(checkpoint_path="runs/a/checkpoint_best_total.pth")
predictor.input_size(1008, 504)   # (1008, 504) at a 504 resolution trained keeping the aspect; (504, 504) square
```

**What you may change.** The rounding (up to a block) with a measurement against the training's validation. Not: a
size setting on the predictor, the flag read from the loaded model's args (they hold the default), or a predict call
mixing pictures of two sizes. Pins: `tests/test_rfdetr_predictor.py::TestTheTrainingsResize`,
`tests/test_lowering.py::test_a_model_trained_keeping_the_aspect_is_resized_as_the_predictor_resizes`,
`tests/test_stream_rfdetr_dataset.py::test_the_scale_jitter_setting_reaches_the_training_transforms_as_on_rfdetrs_own_loader`.

## 14. A picture may reach RF-DETR at its own size: the resize skipped, nothing else (2026-10-13)

**Context.** RF-DETR resizes every picture: to a `resolution` square, or (`square_resize_div_64: false`) to a short
side of `resolution` with the long side capped at 1 333 px — `_COCO_MAX_SIZE`, a module constant of
`rfdetr/datasets/coco.py`, not a setting. A picture made tall on purpose — a capture's rows at a fixed time step,
504 x 2 040 — never reaches the model whole (330 x 1 333). The network itself takes a non-square input whose sides are
multiples of its block (patch x windows: measured 2026-10-13, seg-medium at 504 x 2 040 ran, 512 x 2 048 was refused
`Backbone requires input shape to be divisible by 24`), and RF-DETR's collate pads a batch up to that block.

**Decision.** `RFDETRDetector(keep_picture_size=True)` hands `StreamRFDETRDataset` a pipeline of RF-DETR's own
pieces without the resize (`_unresized_transforms`): the augmentation RF-DETR's router picks for `aug_config` (`None`
its horizontal flip, `{}` none, otherwise its Albumentations stack), `ToImage`, the float scale, its `Normalize`. The
dataset refuses a picture whose size differs from the first it read, naming both and the record's `file`, because a
seg batch joins masks of one size and would fail later with `Sizes of tensors must match`. Stream mode only: a disk
dataset goes through RF-DETR's `build_dataset`, which this runnable does not build.

**Consequences.** Pictures of one non-square size train whole; `resolution` only sizes the model. The predictor
(`RFDETRPredictor`, §10) runs such a model the same way through `keep_picture_size` (addendum), which wins over the
size the checkpoint records (§13): a model trained keeping the aspect needs no setting, one trained on whole pictures
needs this one.

**Example.**

```python
from raidar.stream_rfdetr_dataset import StreamRFDETRDataset

dataset = StreamRFDETRDataset(stream, "val", 504, ns, keep_picture_size=True)
image, target = dataset[0]    # a 504 x 2040 picture -> image.shape == (3, 2040, 504)
```

**What you may change.** The augmentation kept, if RF-DETR's router changes (follow it, never diverge). Not: a resize
or a pad added to this path to make sizes agree — a crop in the stream does that (§12).

**Addendum (2026-10-14): prediction at the picture's size.** `RFDETRPredictor(keep_picture_size=True)` makes
`input_size(H, W)` (§13) answer `(H, W)`, so each picture's own `(H, W)` goes to `predict` as its `shape` argument — the
library's resize then changes nothing — and the one-call-per-size grouping of §13 applies unchanged. Nothing of
`predict` is copied, so its normalization, post-processing and mask upsampling stay the library's. The compiler's
lowering reads the same `input_size`, so its resize to the picture's own size is the identity.

## 15. A dataset folder is read whichever way it is labelled, and loaded by its settings alone (2026-10-16)

**Context.** The detection pictures a sink writes hold YOLO labels and a COCO sidecar for the same pictures, and
outside datasets come as either. A person reviewing one in the annotation workspace had to draw a source graph — a
reader node, a stream, the class list wired into the graph's root — before seeing a single picture. Two facts shaped
the answer, both measured: an annotation workspace adds only FILES through a tool (each file then read by its files
graph, which has no labels), so a tool listing pictures shows no boxes; and its source graph, applied from its
settings alone, lists records with their boxes and class names (a two-picture copy of a real split: `detection
source: 2 recording(s), 33 class(es)`, six named boxes on the first).

**Decision.** `CocoSource` reads one split of a COCO folder in the three places a COCO file sits, numbering classes
by the categories' order (the official ids have gaps). `CocoOrYoloSource` picks the reader from what the folder holds
— `labels` decides only when both are there — and DECLARES `class_names`, as `YoloSource` and `CocoSource` now do, so
a graph's class list is the source itself (`classes: !ref:dataset`) rather than a list typed or evaluated by an
editor. raidar ships the workspace's `detection` source graph wrapping it: three settings (folder, split, labels), and
an `app: {tools: …}` line the workspace lists in its tools menu, which opens those settings. Every folder refusal is a
`DatasetFolderError`, a `ValueError`: the workspace reports a `ValueError` from a source as a wrong setting, and
anything else escapes its sync.

Rejected: a tool adding the pictures as files plus a files graph that finds each picture's labels by its path —
every dropped picture would run raidar's files graph, and a COCO file would be searched per picture; the person chose
the source graph.

**Consequences.** A detection workspace loads one split of one dataset at a time, replacing what its source listed.
With raidar installed, the detection type's source is this graph instead of an empty starter; its canvas still opens.
Run from inside `raidar/`, the folder's gitignored `raidar.egg-info` shadows the installed metadata — an entry point
added without regenerating it is invisible there (measured: the workspace ran the type's empty source).

**Example.**

```python
from raidar.coco_or_yolo_source import CocoOrYoloSource

source = CocoOrYoloSource(folder="$DATA_ROOT/signal_detection/pictures/d", split="val")
len(source), source.class_names[:2]     # (1880, ['5GNR', 'LTE'])
source.labels = "coco"                   # the same pictures, from val/_annotations.coco.json
```

**What you may change.** Another place a COCO file sits (one more candidate, in order), another label kind (one more
branch, with its refusal). Not: a refusal that is not a `ValueError` (the workspace would crash on a wrong folder), or
a class numbering by the categories' ids.
