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
