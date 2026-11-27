# raidar — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

- [ ] **RF-DETR seg on Apple's GPU (`mps`) holds 9–11 GB for a 3 GB job and swaps a laptop** @medium @perf — measured 2026-10-14 on an 18 GB M3 Pro, a signal-detection training run (`smoke_pieces`, `batch_size=1`) (seg-nano, 130 pictures of 504 x 504, one epoch): GPU 16-bit (the default) 11 GB from 102 s, training 8 min 42 s, over 12 GB after the epoch; GPU 32-bit (`runnable.model_config.amp=false`) 10–11 GB, 4 min 3 s; GPU 16-bit with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5` 8.8–9.7 GB, **2 min 47 s**; CPU (`runnable.device=cpu`) 2.1–3.1 GB, 6 min 56 s. At batch size 2 every GPU run reached 13–20 GB and stalled in swap. The limit only bounds torch's cache, so 9 GB is still three times the CPU's: find what the GPU path keeps (the mask loss's buffers, the cache). `RFDETRDetector.mps_low_watermark` (and every matrainer torch trainer's) now sets the low ratio from a config or the command line; the smoke modes of a signal project's RF-DETR seg training config set 0.5 (2 min 45 s, 9.1-9.5 GB). A low ratio above the high one is refused at start (`RuntimeError: invalid low watermark ratio 1.4`). @2026-10-14

- [ ] **`test_every_flowed_slot_is_declared_lazy` passes vacuously for `LightningDetector`** @small @tests — `raidar/tests/test_lazy_slots.py:39-43` reads `flow(self.<slot>` only from the concrete class's own module, but the flows moved to `raidar/raidar/detection_base.py:105-115` and `raidar/raidar/detection_keras.py:102,192` (and into matrainer): `grep -n "flow(self\." raidar/raidar/lightning_detector.py` → no hit, so the check finds nothing to verify. `RUNNABLES` (`:29-34`) also omits `KerasDetector`, `TensorFlowDetector`, `JaxDetector` and `MlxDetector`. Scan the whole class hierarchy and list every runnable.

- [ ] **`YoloSink(layout="rfdetr")` writes a COCO sidecar RF-DETR's COCO reader cannot open the pictures of** @small @bug — the sidecar sits at `<split>/_annotations.coco.json` (where `build_roboflow_from_coco` looks) but its `file_name` is the bare `000000.png` while the pictures are in `<split>/images/`: `rfdetr.datasets.coco.CocoDetection(<out>/train, <out>/train/_annotations.coco.json)[0]` → `FileNotFoundError: .../train/000000.png`. The YOLO reader (`dataset_file: yolo`, the one the layout is for) is unaffected. Decide: `file_name: images/000000.png` in this layout, or document the sidecar as not trainable from.

- [ ] **The energy finder: a class-free dense detector for spectrogram pictures** @medium @feature — a fully convolutional
  model with a U-Net trunk (the shape of deltaid's U-Net spec, at the picture's resolution) and two heads per cell — occupancy
  (focal binary loss) and the log distances to the instance's four edges (the FCOS rectangle head, one class, an overlap
  loss on occupied cells) — decoded by fusing the occupied cells' rectangle votes (overlap ≥ 0.5, occupancy-weighted
  mean, the mask the voters) and dropping instances under a cell count; output in `DetectionModelPredict`'s contract
  (`boxes`, `scores`, `labels`, `masks`). Torch first, the other engines when asked. The design and its measurement are
  a signal project's two-stage detection design; scope as code first (root rule). @2026-10-18
