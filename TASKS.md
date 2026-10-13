# raidar — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.



- [ ] **`test_every_flowed_slot_is_declared_lazy` passes vacuously for `LightningDetector`** @small @tests — `raidar/tests/test_lazy_slots.py:39-43` reads `flow(self.<slot>` only from the concrete class's own module, but the flows moved to `raidar/raidar/detection_base.py:105-115` and `raidar/raidar/detection_keras.py:102,192` (and into matrainer): `grep -n "flow(self\." raidar/raidar/lightning_detector.py` → no hit, so the check finds nothing to verify. `RUNNABLES` (`:29-34`) also omits `KerasDetector`, `TensorFlowDetector`, `JaxDetector` and `MlxDetector`. Scan the whole class hierarchy and list every runnable.

- [ ] **`YoloSink(layout="rfdetr")` writes a COCO sidecar RF-DETR's COCO reader cannot open the pictures of** @small @bug — the sidecar sits at `<split>/_annotations.coco.json` (where `build_roboflow_from_coco` looks) but its `file_name` is the bare `000000.png` while the pictures are in `<split>/images/`: `rfdetr.datasets.coco.CocoDetection(<out>/train, <out>/train/_annotations.coco.json)[0]` → `FileNotFoundError: .../train/000000.png`. The YOLO reader (`dataset_file: yolo`, the one the layout is for) is unaffected. Decide: `file_name: images/000000.png` in this layout, or document the sidecar as not trainable from.
