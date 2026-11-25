# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`YoloSink.split_in_name`** — `true` names every file after its split (`train_000000.png`, `val_000000.png`), so
  two splits of one folder never hold a picture of the same name; off by default (`README.md`).
- **`RandomCropDetection` placement `densest`** — the piece that keeps the most boxes, the same one every call: for a
  dataset of smaller pictures, one piece a picture. Pieces keeping as many boxes are told apart by the box area they
  keep, then by how near the picture's middle they start; a picture without boxes gets the middle piece (`README.md`).
- **`raidar.crop.PieceSource`** — a validation set of every piece of every picture, side by side (no overlap, the
  piece past an edge padded with the median), each piece cut as `RandomCropDetection` cuts it; indexable, so RF-DETR's
  loader reads it. `RandomCropDetection.cut_at(record, top, left)` is the cut both make (`README.md`).
- **The wheel ships the example configs** (`examples/*.yaml`) under `share/raidar/examples/`, where
  `pkg://raidar/examples/<file>.yaml` names them from any install.
- **`raidar.relabel.RelabelLabels`** remaps a YOLO or COCO dataset folder's classes by name, many to one, always from
  the original labels the first remap keeps (`labels.orig/`, `data.orig.yaml`, `<stem>.orig.json`; each remapped
  COCO annotation names its `original_label`); `mode: restore` puts the folder back byte for byte
  (`docs/relabel.md`, `examples/relabel.yaml`).
- **`raidar.CocoSource`** reads one split of a COCO dataset folder as detection records — the COCO file in
  `<split>/_annotations.coco.json`, `annotations/<split>.json` or the official `annotations/instances_<split>*.json`,
  classes numbered by the categories' order, polygon segmentations as `extras["polygons"]` (`README.md`).
- **`raidar.CocoOrYoloSource`** reads a dataset folder labelled either way: `labels: yolo | coco` picks when the
  folder holds both; a wrong folder or a split it lacks is a `DatasetFolderError` (a `ValueError`).
- **`YoloSource.class_names`** — the folder's class list in label order, declared on the source.
- **The annotation workspace's detection source**: in a detection workspace, *tools ▾ → load a COCO or YOLO
  dataset…* opens a folder, split and labels setting, and the list shows that split with its boxes and classes.
- **`RFDETRDetector.mps_low_watermark`** — on Apple's GPU, torch gives its cached memory back above that share of the
  GPU's working set (`PYTORCH_MPS_LOW_WATERMARK_RATIO`, set before the model is built); settable in a config or on
  the command line (`runnable.mps_low_watermark=0.5`). Unset leaves torch's default.
- **`RFDETRDetector(keep_picture_size=True)`** hands every picture to the model at its own size in stream mode —
  RF-DETR's resize (a square, or a long side capped at 1333 px) skipped, its augmentation and normalization kept;
  every picture must be one size, a second size refused naming both (`README.md`).
- **`RFDETRPredictor(keep_picture_size=True)`** predicts each picture at its own size (RF-DETR's `predict` told the
  picture's shape), as a model trained with `keep_picture_size` saw them; its lowering for the package compiler
  skips the resize to the square the same way.
- **`raidar.crop.RandomCropDetection`** cuts a picture to one `height` x `width` piece with its boxes and mask
  outlines, before a detector's own resize: `placement: random` (a new piece each epoch) or `centre`; a smaller
  picture is padded with its median, a box cut under `min_size_px` dropped. Pictures of any height then train an
  RF-DETR seg model, whose batch needs every picture one size (`README.md`).
- **RF-DETR stream mode trains `seg-*` models**: each instance's mask from its target's `extras["polygons"]`, a
  box without one masked by its rectangle; a picture's labels are read with the picture, per item.
- **`raidar.RFDETRPredictor`** predicts from one of RF-DETR's own checkpoint files, the model rebuilt
  from what the file records (variant, resolution, query groups, head, class names): boxes, scores and
  labels in the input picture's pixels, and each instance's mask for a `seg-*` checkpoint. A model trained keeping its pictures' aspect
  (`square_resize_div_64: false`, read off the file's `args`) is shown each picture at that aspect — the short side
  at its resolution — and `input_size(height, width)` says the size.
- **RF-DETR stream mode passes `train_config`'s `scale_jitter`** to RF-DETR's transforms, as RF-DETR's own folder
  loader does: off, pictures that keep a non-square aspect train a `seg-*` model in batches.
- **`RFDETRPredictor(score_mode="any_class")`** keeps one detection per query, scored by the chance that any of
  its classes is right (`1 - prod(1 - p)`) and named by its best class — a query torn between classes is found
  where a per-class cut drops it. `any_class_excludes` leaves named classes out of the score. `score_mode="class"`,
  the default, is RF-DETR's own scoring.
- **`YoloSink(image_mode="grey")`** writes a grey picture (three equal channels) as one channel — 2.2x
  faster to encode and a third smaller; every reader turns it back into the same three channels, and a
  picture whose channels differ is refused.
- **`YoloSink(image_threads=4)`** encodes and writes the pictures on threads, each under its record's
  index — the same files as one thread writes.
- **Instance masks through `YoloSink`.** A record's `Boxes` target may carry each instance's
  polygons in `extras["polygons"]` (`raidar.POLYGONS` names the key; per box a list of non-overlapping
  polygons of `[x, y]` pixel points, `[]` for a box with no mask of its own). The sink writes the
  YOLO-seg line `class x1 y1 x2 y2 ...` — one ring per instance, several polygons joined by
  zero-width bridges — and the COCO sidecar's `segmentation` (the polygons apart) and `area` (their
  shoelace sum); a split holding any ring holds no box line. `raidar.YoloSource` reads a ring back
  as the instance's polygon, so a merge keeps the masks. RF-DETR's own YOLO reader yields the masks
  and its COCO reader decodes the sidecar. Records without the key are written as before.
- **`YoloSink(mode="append")`** continues a split after its last record — the index, `metadata.jsonl`
  and the COCO sidecar grow; a vocabulary that disagrees id for id is refused before anything is
  written. `replace`, the default, wipes the split first as before.
- **`YoloSink(facts=[...])`** lists the distinct values of the named record entries in `data.yaml`
  (a picture's scale: a folder of mixed scales is recorded, never refused).
- **`raidar.YoloSource`** reads one split of a YOLO folder back as records — either layout, the
  labels scaled by the picture, the `metadata.jsonl` row re-attached, class ids remapped by name to a
  vocabulary — and `examples/merge_yolo.yaml` merges folders with it (readers → the sink in `append`).
- **`YoloSink(layout="rfdetr")`** writes the folder RF-DETR's own reader accepts — `<split>/images`,
  `<split>/labels`, the COCO sidecar at `<split>/_annotations.coco.json` — beside the default
  Ultralytics layout (`images/<split>`), which is unchanged.
- **The segmentation variants** `seg-nano` / `seg-small` / `seg-medium` / `seg-large` / `seg-xlarge`
  on the `rfdetr` backend (`RFDETRDetector`): RF-DETR's instance-segmentation models, trained from a
  box dataset on disk (`dataset_file: yolo`, a box label read as its four-corner polygon). One table,
  `raidar.rfdetr._VARIANT_CLASSES`, names every variant; `RFDETRModel` keeps the four detectors.

### Changed
- `--framework ultralytics` loads the separate Ultralytics package as the `raidar.ultralytics`
  subpackage, and `examples/train_cppe5.yaml` names its trainer
  `raidar.ultralytics.ultralytics_trainer.UltralyticsTrainer`.

## [0.1.0a1] — 2026-09-28

First public pre-release. The surface it ships:

### Backends
- **Nine training engines over one config**, each a runnable class behind its own extra:
  `LightningDetector` (`[lightning]`, the reference), `TorchDetector` (`[torch]`, a
  hand-written loop), `FastaiDetector` (`[fastai]`), `HuggingFaceDetector` (`[huggingface]`,
  `transformers.Trainer`), `KerasDetector` (`[keras]`), `TensorFlowDetector` (`[tensorflow]`),
  `JaxDetector` (`[jax]`), `MlxDetector` (`[mlx]`, Apple silicon) and `RFDETRDetector`
  (`[rfdetr]`, RF-DETR's own training recipe). Each runnable dispatches on `task`: `fit` /
  `evaluate` / `test` / `predict`.
- **Backend-free core:** `import raidar` loads no ML framework. Touching a backend that is not
  installed raises an `ImportError` naming the extra to install.

### Models
- The torchvision detectors (Faster R-CNN, RetinaNet, FCOS, SSD) used as-is, plus
  `TorchVisionDetectionModel` — the same zoo behind one configured wrapper for inference from a
  trained checkpoint.
- `RFDETRModel` (the RF-DETR family as a torchvision-style module) and
  `HuggingFaceDetectionModel` (any Hub object-detection checkpoint by name).
- The `raidar.fcos` package: one small FCOS spec with five native implementations
  (`TorchFcos`, `KerasFcos`, `TensorFlowFcos`, `FlaxFcos`, `MlxFcos`), for comparing engines on
  one network.

### Data, metrics and results
- COCO-style detection metrics (mAP) with per-split additions and a detection confusion matrix.
- `YoloSink` writes a dataset in the on-disk YOLO layout.
- RF-DETR trains either from a dataset directory or straight from a record stream in memory.

### Command line
- `raidar train | evaluate | predict <config.yaml> [--framework <name>]` and `raidar frameworks`;
  the same config also runs under `recordstream run`.
