"""Raidar marainer plugin — mounts ``marainer detection`` sub-app AND
preserves the legacy ``ultralytics-train`` / ``ultralytics-evaluate`` verbs.

This module is discovered by marainer's plugin system via the
``marainer.commands`` entry-point group (declared in ``raidar/pyproject.toml``).
It adds:

* ``marainer detection train``      → :func:`raidar.cli.train`
  (sonair-style Lightning detection trainer subclass).
* ``marainer detection evaluate``   → :func:`raidar.cli.evaluate`.
* ``marainer ultralytics-train``    → :class:`raidar.ultralytics_trainer.UltralyticsTrainer`
  (native Ultralytics trainer; back-compat with prior entry-point).
* ``marainer ultralytics-evaluate`` → :class:`raidar.ultralytics_trainer.UltralyticsEvaluator`.

The two ultralytics-native verbs are kept registered via
:func:`raidar.cli_plugin.register` for back-compat — any user invoking
``marainer ultralytics-train`` against an existing YAML continues to work.
"""

from liquifai import LiquifyApp

from raidar.cli import evaluate, train
from raidar.cli_plugin import register as _register_ultralytics_verbs


def register(app: LiquifyApp) -> None:
    """Mount the ``detection`` sub-app onto *app* and re-register the
    legacy ultralytics-native verbs.
    """
    # Legacy: ultralytics-train / ultralytics-evaluate top-level verbs.
    _register_ultralytics_verbs(app)

    # New: detection sub-app (Lightning path, sonair-style).
    detection = LiquifyApp(
        name="detection",
        description="Object-detection workflow commands (raidar specialisation).",
    )
    detection.script_command(flow_mode="auto")(train)
    detection.script_command(flow_mode="auto")(evaluate)
    app.add_app(detection, name="detection")
