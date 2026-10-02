# raidar — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.



- [ ] **`test_every_flowed_slot_is_declared_lazy` passes vacuously for `LightningDetector`** @small @tests — `raidar/tests/test_lazy_slots.py:39-43` reads `flow(self.<slot>` only from the concrete class's own module, but the flows moved to `raidar/raidar/detection_base.py:105-115` and `raidar/raidar/detection_keras.py:102,192` (and into matrainer): `grep -n "flow(self\." raidar/raidar/lightning_detector.py` → no hit, so the check finds nothing to verify. `RUNNABLES` (`:29-34`) also omits `KerasDetector`, `TensorFlowDetector`, `JaxDetector` and `MlxDetector`. Scan the whole class hierarchy and list every runnable.
