# Raidar Mandates

Rules only. The reasons are in [`docs/architecture.md`](docs/architecture.md) (cited as §N),
and usage is in the [README](README.md) and the comments of `examples/train_cppe5.yaml`. Root
`AGENTS.md` rules are not repeated here.

## Current state

Object detection on ten backends, all driven by one config: `lightning`, `torch` (a
hand-written loop), `fastai`, `huggingface`, `keras`, `tensorflow`, `jax`, `mlx`, `rfdetr`
(RF-DETR's own loop), and `ultralytics`, which ships in a separate distribution. On PyPI as
pre-release `0.1.0a1`; the repo is private.

Each runnable `<backend>_detector.py` = task hooks (here) + a matrainer machinery base + its
own full flat `__init__` (§5):

| backends | task hooks | matrainer base |
|---|---|---|
| lightning · torch · fastai | `DetectionTorchHooks` | `LightningRunnable` · `TorchLoopRunnable` · `FastaiRunnable` |
| keras · tensorflow | `DetectionKerasHooks` | `KerasRunnable` · `TensorFlowRunnable` |
| huggingface · jax · mlx | `DetectionBase` | `HuggingFaceRunnable` · `JaxLoopRunnable` · `MlxLoopRunnable` |
| rfdetr | none | none: `RFDETRDetector` drives RF-DETR's loop, plain-function tracking |

Other modules: `detection_base.py` (torch-free); `torch_detection.py` (the torch engine's
registered metrics + `native_batch`, not to be confused with `detection_torch.py`);
`torchvision_detection.py` (builder registrations); the model wrappers `rfdetr.py`,
`huggingface_model.py`, `torchvision_model.py`; `fcos/` (spec, five implementations,
`FcosTargets`); `crop.py` (`RandomCropDetection`, one piece size before a detector's own resize); `cli.py` (the `FRAMEWORKS` table only); `__init__.py` (lazy exports over
`_BACKEND_NAMES`). Example: `examples/train_cppe5.yaml`. Sweep: `examples/sweep/`.

**Who owns what:** recordstream owns records, collate, `batch_boxes`/`batch_metadata`, the
target ops and `DetectionOutput`. matrainer owns the trainer machinery, tracking, metric slots,
the generic checkpoint loader, the CLI factory and the lazy exports. raidar owns the detection
hooks, model wrappers, the FCOS spec, registrations and configs.

## Backends

- Keep each concrete backend's FULL flat `__init__` (introspection reads it). Extend through
  the hooks; never shadow the inherited machinery (`_build_trainer`, `_live_trackers`,
  `_run_eval`, `_report_epoch_metrics`, `_log_run_context`, `fit`, `_score`; in the keras family
  also `_callbacks`, `_compile`, `initialize`). Machinery changes go into matrainer with every
  consumer updated. (`tests/test_tracking_parity.py`) §5.
- A new backend touches: `raidar/<name>_detector.py`, a `FRAMEWORKS` row, a scope block in the
  example, an entry point, `_BACKEND_NAMES` + an extra + `[tool.aisland] frameworks`,
  `BACKENDS`/`METRIC_BRIDGES` in `tests/test_tracking_parity.py`, and a `sweep.yaml` row. §1.
- A backend gets its own module only when upstream ships a training loop worth driving; it
  drives that loop and adds only a Dataset/DataModule bridge. No adapter layer, no
  `{losses, predictions}` step contract, no per-checkpoint preset classes. §2.
- License decides core admission: permissive (RF-DETR) in core, AGPL (ultralytics) outside.
- Every backend puts its final metrics on the trackers, not only in the sidecar
  (`tests/test_tracking_parity.py::test_every_backend_puts_its_METRICS_on_the_trackers`).
- `TorchDetector` imports no training framework (`tests/test_torch_detector.py::test_it_needs_no_training_framework`).
- `TensorFlowDetector` is a sibling of `KerasDetector` (shared `DetectionKerasHooks`), never a
  subclass, and reaches keras/tf only through `raidar._tensorflow`, which sets
  `KERAS_BACKEND=tensorflow` first. (`tests/test_tensorflow_detector.py`)
- fastai: the recorded `valid_loss` is meaningless under the no-op-loss bridge and is dropped;
  the `device:` knob writes the private `dataloaders._device` on purpose.
- huggingface: no loss slot, `registrations` is `None`, and `transformers` objects stay deferred
  dotted strings (`tests/test_huggingface_detector.py::test_transformers_stays_a_deferred_dotted_target`).
- rfdetr: no `predict` (prediction is `LightningDetector` + `RFDETRModel`, or `RFDETRPredictor` for one of
  RF-DETR's own checkpoint files); the run-name knob is
  `run_name`, never `run`, which would shadow `run()`
  (`tests/test_rfdetr_detector.py::test_run_name_does_not_shadow_run_method`);
  `train_config:` merges verbatim over the floor, `model_config:` is the same pass-through to the model's build-time settings (`group_detr`, …; keys with a parameter of their own are refused: `tests/test_rfdetr_detector.py::test_model_config_refuses_a_key_with_its_own_parameter`); `fit` finishes its manifest with
  `capture_log_file(..., status="success")`; Stream mode is drift-guarded by the gated
  `::test_stream_train_smoke`.
- The RF-DETR variants are ONE table, `raidar.rfdetr._VARIANT_CLASSES`, and the two Literals are
  read off it: `VariantName` (the backend, every entry — the `seg-*` models train from the dataset
  on disk, a box label read as its four-corner polygon, a polygon line as that polygon, or from a stream whose
  targets carry `extras[POLYGONS]`) and the four detection names on
  `RFDETRModel`, which builds box targets and no masks. A new variant is a table row
  (`tests/test_rfdetr.py::TestTheVariantTable`).
- `RFDETRPredictor` predicts only, from RF-DETR's own checkpoint file, and takes NO architecture setting: the variant,
  resolution, `group_detr`, head and class names come from the file (`RFDETR.from_checkpoint`) — never add a
  `variant`/`model_config` knob to it, and never open `RFDETRModel`'s variants to `seg-*` to predict masks (it is the
  lightning backend's training wrapper). Boxes and masks are in the INPUT picture's pixels, a label outside the
  vocabulary (the no-object slot) is dropped, a grey picture is read as three equal channels; the file is read at the
  first prediction. §10. (`tests/test_rfdetr_predictor.py`)
- `RFDETRPredictor.score_mode`: `class` (default) is the library's answer unchanged; `any_class` rewrites the
  `pred_logits` handed to the library's OWN post-processing (one slot per query: its best counted class carrying
  `1 - prod(1 - p)`), so selection, boxes and masks stay the library's — never override `PostProcess` internals or
  copy `predict`'s tail. The no-object slot never counts; an unknown `any_class_excludes` name is refused at the build.
  §11. (`tests/test_rfdetr_predictor.py::TestTheAnyClassRewrite`, `::TestAnyClassScoring`)
- RF-DETR stream mode reads a record's labels in the SAME read as its picture, never walked ahead or cached, and a
  `seg-*` model gets each instance's mask from `Boxes.extras[POLYGONS]` through RF-DETR's own `ConvertCoco` (a box
  without an outline masked by its rectangle). §12. (`tests/test_stream_rfdetr_dataset.py::test_a_pictures_labels_come_from_the_read_that_gave_the_picture`,
  `::test_a_seg_model_gets_each_instances_mask_from_its_outline_and_a_box_without_one_is_its_box`)
- `RandomCropDetection` (`raidar/crop.py`) cuts in the picture's OWN pixels before RF-DETR's resize: exactly
  `height` x `width` out, boxes clipped and a box under `min_size_px` dropped, outlines clipped exactly, a smaller
  picture padded with its median and never stretched; `random` draws from Python's `random` (seeded per loader
  worker and epoch), never an instance RNG (pickled into every worker it would repeat). Never move the cut into
  RF-DETR's `aug_config` — it runs after the resize and its 1 333 px cap. §12.
  (`tests/test_crop.py`, `tests/test_stream_rfdetr_dataset.py::test_cut_pictures_of_any_height_batch_into_masks_of_one_size`)
- `YoloSink.layout` is the only difference between the Ultralytics folder (`images/<split>`) and
  RF-DETR's (`<split>/images`, the COCO sidecar at `<split>/_annotations.coco.json`); the default
  stays `ultralytics`. (`tests/test_yolo_sink.py::test_the_rfdetr_layout_is_what_rfdetr_reads_and_the_default_is_not`)
- `YoloSink.mode`: `replace` (default) wipes the split, `append` continues it after the last stem and
  refuses a vocabulary that disagrees id for id BEFORE writing; `facts` lists distinct record entries
  in `data.yaml` (mixed picture scales are recorded, never refused — user ruling 2026-10-06).
  (`tests/test_yolo_sink.py`, the "append mode and facts" block)
- `YoloSource` is the sink's mirror and reads both layouts; class ids are remapped by NAME to the
  vocabulary given and an unknown name is refused, never renumbered; construction touches no disk.
  Merging folders is readers + `ConcatSource` + the sink in `append` (`examples/merge_yolo.yaml`).
  (`tests/test_yolo_source.py`)
- Instance masks ride `Boxes.extras[raidar.POLYGONS]`: per box a LIST of non-overlapping polygons
  (≥ 3 `[x, y]` pixel points, first point not repeated), `[]`/`None` meaning the box rectangle; the
  contract is `yolo_sink`'s module docstring. `YoloSink` writes ONE YOLO-seg ring per instance (YOLO-seg
  holds one polygon a line; several are joined by zero-width bridges, each walked once each way) and
  COCO `segmentation` as the polygons apart; a split holding any ring holds NO box line, rewritten at
  `flush` (an `append` too), because Ultralytics refuses a file that mixes the forms; nothing goes
  in `data.yaml`; no key → byte-identical to the box-only output; `YoloSource` reads a ring back as
  `[ring]`. §9. (`tests/test_yolo_sink.py::TestBoxOnlyRecordsAreUnchanged`, `::TestSeveralPolygonsOfOneInstance`,
  `::TestOneFormPerSplit`, `::TestRfdetrReadsTheFolderAsSegmentation`; `tests/test_yolo_source.py::TestPolygons`)
- A dataset-processing example (`DatasetProcessor`) is a `runnable:` without a `task`; `tests/
  test_examples_load.py` holds only a runnable that has one to the task vocabulary.
- Spec engines (keras, tensorflow, jax, mlx) have no `loss:` slot and score with
  `spec.average_precision_50` over `spec.decode_predictions` (`<split>/map_50`, `/precision`,
  `/recall`). jax and mlx `initialize()` call `solidify()` explicitly. jax jits its steps once;
  `MlxFcos` keeps its network in the PUBLIC attribute `net`. (`tests/test_jax_detector.py`,
  `tests/test_mlx_detector.py`)

## Config and CLI

- The backend is a flag: `--framework X` → `--scope framework=X`. No backend subcommand, no
  separate binary. §1.
- `cli.py` is data only. The dispatch through `materialize_runnable`, the flag rewrite and the
  registration-import order are `matrainer.cli`'s; never copy a helper back.
- A runnable with no `task` attribute (the ultralytics trainer) runs under `train` and must
  RAISE under `evaluate`/`predict`. The guard is `matrainer.cli.default_dispatch`
  (`tests/test_cli_frameworks.py::test_a_taskless_runnable_raises_on_any_other_verb`).
- `runnable:` sits inside each `!scope:framework=` block, `default_scopes:
  [framework=lightning]` picks the default, and `model`, `detection_ops`, `num_classes`,
  `class_names` are TOP-LEVEL siblings
  (`::test_the_default_backend_is_a_block_named_by_default_scopes`,
  `::test_every_declared_framework_resolves_to_a_complete_runnable`).
- Labels: torchvision blocks use `label_offset: 1` and count background (`num_classes: 6` on
  cppe-5). Every other backend is 0-indexed (`label_offset: 0`, `num_classes: 5`) and its block
  RESTATES both keys (`::test_the_zero_indexed_backends_restate_the_label_convention`).
- `Framework.registrations` names every registration module a backend needs; one registration
  module never imports another for its side effect
  (`::test_every_framework_names_registrations_that_import`).
- Device pins are config, not code: the torchvision blocks pin cpu (MPS is ~30x slower) and
  huggingface pins `use_cpu: true` (D-FINE cannot run on MPS). Never add a silent device guard
  in library code.

## The ultralytics boundary

- raidar never imports `ultralytics`. Code that only writes the YOLO format (`YoloSink`) may live
  here.
- The backend ships in the `raidar-ultralytics` distribution as `raidar.ultralytics` (the root
  "ships INSIDE another package" rule). raidar carries only its `FRAMEWORKS` row, imported
  best-effort, and the example's scope block, wired by dotted path. Code that tolerates its
  absence checks the dotted name `raidar.ultralytics`, never `raidar`
  (`tests/test_cli_frameworks.py::test_every_framework_names_registrations_that_import`).
  The sweep leaves it out.

## Registrations and discovery

- `import raidar` registers nothing. Every registering module gets its own
  `confluid.configurables` entry point in the SAME change (a module, never `module:attr`), then
  `uv pip install -e ./raidar --no-deps`. (`tests/test_entry_points.py`)
- A registration module is named for its content, `<engine|library>_<task>.py`.
- Every backend runnable is `(detection, trainer, <framework>)`, so detection has nine trainers
  in core. Evaluate/test/predict are `@entrypoint` markers, never evaluator classes. `framework`
  is what a class is wired to (fastai → `torch`, TensorFlow → `keras`). Wrappers are
  `role="model"`; there is no `role="loss"`. (`tests/test_categories.py`)
- The nine torchvision builders are `register`-ed with `lazy=True`; Mask/Keypoint R-CNN stay out.
  `native_batch` is not `@configurable`. (`tests/test_categories.py`)
- A new constructor param on `LightningDetector` or `RFDETRModel` stales navigaitor's
  `examples/detection_train.yaml`: run `python -m navigaitor.scripts.regen_examples` in the same
  change (navigaitor `tests/test_examples_are_fresh.py`).

## Task pipeline

- `run()` is `run_entrypoint(self, self.task)`; never a hand-written `{task: method}` dict
  (`tests/test_lightning_trainer.py::test_entrypoint_markers_cover_every_task_literal_value`).
- `num_classes` on fit: concrete model head → explicit → lazy model's pinned kwarg → derived from
  train targets. Injected into a `!lazy:` model only when the builder accepts it; read off the
  model on eval. (`tests/test_lightning_trainer.py::test_initialize_*`,
  `tests/test_detection_evaluator.py::test_initialize_*`)
- No `train_metrics` slot; wiring one raises
  (`tests/test_lightning_trainer.py::test_there_is_deliberately_no_train_metrics_slot`).
- `metrics:` covers every eval split; `val_metrics`/`test_metrics` merge over it through
  `matrainer.metric_slots.merge_metric_specs`. Defaults are `None` sentinels resolved in
  `_resolve_task_defaults()`; a wired metric is never replaced
  (`::test_metric_slots_default_to_sentinels_resolved_at_initialize`).
- `_reportable()` drops MAP's `classes` key (`::test_reportable_metrics_filters_the_classes_bookkeeping_key`).
- Predictions are `recordstream.DetectionOutput`/`DetectionPredictions`; never a parallel schema.

## Batching

- raidar owns no collate. Torch-family loaders declare `collate_fn=collate_list` and read the
  batch with `native_batch` + `batch_boxes`/`batch_metadata`. No raidar collate, container class
  or key convention; the batch keeps the configured `input_key`/`target_key`.
  (`test_default_loader_DECLARES_the_list_collate` in the lightning, torch and evaluator tests;
  `tests/test_torch_detection_batching.py`)
- Spec-engine chains end in `FcosTargets`, which ADDS the dense maps and keeps the `Boxes`
  (`tests/test_keras_detector.py::TestFcosTargets`). The boxes-outside-the-frame guard lives
  ONLY in `spec.assign_targets`; never copy it into a caller
  (`tests/test_fcos_spec.py::TestBoxesMustBeInTheFrame`). §4.

## Slots and models

- Every slot flowed at run time is `Partial[T]` (`Mandatory[Partial[T]]` when required), never
  `Any` or a hand-written union (`tests/test_lazy_slots.py`).
- `trackers` is a constructor param, `Optional[Union[Partial[Tracker],
  Iterable[Partial[Tracker]]]]`, stored verbatim; there is no `logger:` slot. `metric_names` is
  declared on every backend too.
  (`tests/test_lightning_trainer.py::test_there_is_no_logger_knob_and_trackers_is_a_ctor_param`)
- `model` is `Mandatory`; `trained_model` and `computed_metrics` are outputs that raise before a
  run (`::test_trained_model_output_raises_before_run_returns_after_flow`).
- The native contract: train `forward(images, targets) → loss dict`, eval `forward(images) →
  [{boxes, scores, labels}]`. torchvision detectors conform raw.
- `model_refs` reads stored config only, returns `[]` when nothing external was loaded, and uses
  `hf_model_ref`/`file_model_ref` (`tests/test_model_refs.py`).
- `RFDETRModel` builds, imports `rfdetr` and registers its buffers in `solidify()` only; config
  readers use `_requested_resolution` (`tests/test_rfdetr.py::test_construction_materializes_nothing_and_does_not_import_the_library`).
- `RFDETRModel` sends RF-DETR's own checkpoint files to `rfdetr`'s loader; never teach matrainer
  their shapes (user ruling 2026-09-27, do not re-litigate). §7.
- In both wrappers `num_classes` is optional and, unset, is read off `checkpoint_path`'s head
  (user ruling 2026-08-31, "only use checkpoint_path"). An explicit value wins, a disagreement
  fails the load, and there is no `checkpoint_strict`
  (`TestNumClassesIsBakedIntoTheCheckpoint` in `test_torchvision_model.py` and `test_rfdetr.py`). §6.
- `TorchVisionDetectionModel` is the inference spelling and the raw builders the training one.
  `label_indexing` is a user decision (never make the shift unconditional), an unknown `weights`
  name warns, and the signature is append-only. (`tests/test_torchvision_model.py`) §6.

## FCOS spec and seeding

- `fcos/__init__.py` imports nothing. An implementation's knobs ARE the architecture: no
  pretrained or variant knobs. A new one joins `tests/test_fcos_spec.py::IMPLEMENTATIONS`. §3.
- Decoding exists once, in `spec.decode_predictions`; only the loss is per engine.
- `TensorFlowFcos` never touches `tf.initializers`/`tf.keras`/`tf.losses`/`tf.metrics`
  (`tests/test_tensorflow_detector.py::test_the_raw_tf_model_reaches_no_keras_initializer`).
- Every random draw that shapes the network is seeded BEFORE the build and the RNG is never
  restored: the FCOS `seed`, the torchvision family's `seed` in `DetectionTorchHooks.initialize()`,
  huggingface's `seed`, rfdetr's `train_config.seed`. Fix seeding before comparing devices or
  refactors. Pins and canonical numbers: §8.

## YoloSink

- `_jsonable_metadata` must never raise (`tests/test_yolo_sink.py::test_jsonable_metadata_*`).
- `image_mode: grey` writes ONE channel only for a picture whose three are equal, and refuses any
  other (never reduces colour); `data.yaml` keeps `channels: 3`, what every reader turns it into.
  Pictures are written on `image_threads` threads under the index their record got in `write()`, at most
  `2 × image_threads` waiting; `flush()` drains them before any sidecar. The files equal a one-thread
  run's. (`tests/test_yolo_sink.py::TestGreyPictures`, `::TestPicturesWrittenOnThreads`)
- `_bridged_ring` is held vertex for vertex to the walk its docstring describes, ties included
  (`tests/test_yolo_sink.py::test_the_ring_is_the_walk_its_description_gives_ties_included`); a faster
  form changes how the distances are gathered, never the walk.

## Packaging, release, tests

- `pip install raidar` installs no framework. The spec-engine extras name their own engine plus
  bare `torch`, not `raidar[torch]`. Keep `_BACKEND_NAMES` in step with the extras.
  (`release.yml` smoke test)
- `fastcore<2` and `fastprogress<1.1` are upstream breaks; see `pyproject.toml`.
- Releases: hand-written `release.yml`; workspace specs read `>=0.1.0a1` (aisland
  `tests/test_workspace_dependency_specs.py`); README links are absolute
  (`tests/test_docs_links.py`); the newest CHANGELOG entry is the version (`tests/test_packaging.py`).
- CI runs `mypy .` here with only its own extras: list extra-gated imports in
  `[[tool.mypy.overrides]]`. `--cov-branch` is required (`tests/test_coverage_config.py`).
- CI installs no tensorflow, fastai, jax or mlx, so those suites skip there; install the extra
  before trusting a change. Run the TF behaviour tests alone: `pytest tests/test_tensorflow_detector.py`.

## Sweep

- `examples/sweep/sweep.yaml` is the matrix (nine frameworks, no model dimension) and
  `all_backends.yaml` the overlay adding only tracker blocks; each restates its whole
  `trackers:` list with `!ref:experiment_name`/`!ref:run_name`. The driver is
  `matrainer sweep`. (`tests/test_sweep_config.py`)
