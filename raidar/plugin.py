"""Raidar marainer plugin — mounts the ``marainer detection`` sub-app.

This module is discovered by marainer's plugin system via the
``marainer.commands`` entry-point group (declared in ``raidar/pyproject.toml``).
It adds:

* ``marainer detection train``    → :func:`raidar.cli.train`
  (sonair-style Lightning detection trainer subclass).
* ``marainer detection evaluate`` → :func:`raidar.cli.evaluate`.

The native-ultralytics verbs (``marainer ultralytics-train`` /
``marainer ultralytics-evaluate``) live in the
[raidar-ultralytics](https://github.com/Gearlux/raidar-ultralytics)
sibling package, registered via its own ``marainer.commands`` entry point.
"""

from liquifai import LiquifyApp

from raidar.cli import evaluate, train


def register(app: LiquifyApp) -> None:
    """Mount the ``detection`` sub-app onto *app*."""
    detection = LiquifyApp(
        name="detection",
        description="Object-detection workflow commands (raidar specialisation).",
    )
    detection.script_command(flow_mode="auto")(train)
    detection.script_command(flow_mode="auto")(evaluate)
    app.add_app(detection, name="detection")
