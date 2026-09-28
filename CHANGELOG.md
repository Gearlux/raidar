# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
