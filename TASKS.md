# raidar — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.



- [ ] **`test_every_flowed_slot_is_declared_lazy` passes vacuously for `LightningDetector`** @small @tests — `raidar/tests/test_lazy_slots.py:39-43` reads `flow(self.<slot>` only from the concrete class's own module, but the flows moved to `raidar/raidar/detection_base.py:105-115` and `raidar/raidar/detection_keras.py:102,192` (and into matrainer): `grep -n "flow(self\." raidar/raidar/lightning_detector.py` → no hit, so the check finds nothing to verify. `RUNNABLES` (`:29-34`) also omits `KerasDetector`, `TensorFlowDetector`, `JaxDetector` and `MlxDetector`. Scan the whole class hierarchy and list every runnable.
- [ ] **Docs and comments describe the old surface** @docs @small — nine places still describe raidar before the merged runnables and the retired matrainer verbs:
  - `raidar/README.md:87-91` — "`LightningDetector` is one self-contained standard `LightningModule` — its own detection step functions, `configure_optimizers`, and a `run()` that builds the loaders + `pl.Trainer`"; it is `(DetectionTorchHooks, LightningRunnable)` since architecture §5, and `configure_optimizers` has no hit in `raidar/raidar`.
  - `raidar/docs/architecture.md:192` (record 4, Example) — writes `ToTensor { mode: RGB, normalize: true }`; `recordstream.ops.torch.ToTensor.__init__(self, field="", output="")` (`recordstream/recordstream/ops/torch.py:62`) takes neither; the example config uses `ConvertMode {mode: RGB}` + `Scale {}` + `ToTensor {}`.
  - `raidar/pyproject.toml:101-103`, `raidar/raidar/detection_keras.py:14-15` and `raidar/examples/train_cppe5.yaml:210-211` — say the shared `ToTensor` carries the RGB coercion; `ToTensor` "CONVERTS and nothing else" (`recordstream/recordstream/ops/torch.py:38-41`), the coercion is `ConvertMode`.
  - `raidar/raidar/yolo_sink.py:4,20,239`, `raidar/tests/test_yolo_sink.py:5` and `raidar/tests/test_categories.py:34` — name `matrainer convert`, which does not exist (`matrainer.cli.VERB_TASKS` is train / evaluate / predict; every run is `recordstream run`).
  - `raidar/raidar/_keras.py:10-11` — "Every raidar module writes `from raidar._keras import keras`"; `raidar/keras_detector.py:32` and `raidar/detection_keras.py:39` import it from `recordstream.keras`.
  - `raidar/raidar/rfdetr_detector.py:364-366` — says the device hypothesis "was tested and rejected"; seeded runs later showed mps still varying (0.29446 / 0.333115) while cpu reproduces (the rfdetr comment in `examples/train_cppe5.yaml`, the README rfdetr section).
  - `raidar/docs/architecture.md:53-55` (record 1, "What you may change") — says a backend needs only a `<name>_detector.py`, a `FRAMEWORKS` row and a scope block, "nothing else"; the tests also require an entry point, `BACKENDS` / `METRIC_BRIDGES` rows in `tests/test_tracking_parity.py` and a `sweep.yaml` row.
  - `raidar/tests/test_entry_points.py:41` — the docstring names `configurables.py`, renamed `torchvision_detection.py`.
  - `raidar/tests/test_tracking_parity.py:1` — the docstring says "Both detection backends…"; the file checks nine.
