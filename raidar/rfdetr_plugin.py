"""Raidar-RFDETR marainer plugin — registers ``marainer rfdetr-train`` / ``rfdetr-evaluate``.

Discovered by marainer's plugin system via the ``marainer.commands`` entry-point
group (declared in ``raidar/pyproject.toml``). It adds the native RF-DETR verbs as
a flat escape hatch — parallel to the generic ``marainer detection`` sub-app
(:mod:`raidar.plugin`), and mirroring how raidar-ultralytics keeps its
``ultralytics-train`` / ``ultralytics-evaluate`` verbs in a separate plugin.
"""

from liquifai import LiquifyApp

from raidar.rfdetr_cli import evaluate, train


def register(app: LiquifyApp) -> None:
    """Mount the native RF-DETR verbs onto *app*."""
    app.script_command(name="rfdetr-train", flow_mode="auto")(train)
    app.script_command(name="rfdetr-evaluate", flow_mode="auto")(evaluate)
