"""Tests for the raidar marainer plugin — verifies BOTH the legacy
ultralytics-* verbs AND the new `detection` sub-app land on the host app.
"""

from liquifai import LiquifyApp
from raidar.plugin import register


def _surface_repr(app: LiquifyApp) -> str:
    return repr(vars(app))


def test_register_mounts_detection_subapp() -> None:
    app = LiquifyApp(name="marainer", description="test marainer host")
    register(app)
    sub_app_names = {a.name for a in getattr(app, "sub_apps", []) if hasattr(a, "name")}
    if sub_app_names:
        assert "detection" in sub_app_names
    else:
        assert "detection" in _surface_repr(
            app
        ), "register() must mount a `detection` sub-app on the host LiquifyApp."


def test_register_keeps_legacy_ultralytics_verbs() -> None:
    app = LiquifyApp(name="marainer", description="test marainer host")
    register(app)
    payload = _surface_repr(app)
    assert "ultralytics_train" in payload or "ultralytics-train" in payload
    assert "ultralytics_evaluate" in payload or "ultralytics-evaluate" in payload


def test_register_exposes_detection_train_and_evaluate_commands() -> None:
    app = LiquifyApp(name="marainer", description="test marainer host")
    register(app)

    # Locate the detection sub-app via best-effort traversal of LiquifyApp internals.
    seg_app = None
    for attr_value in vars(app).values():
        if isinstance(attr_value, list):
            for entry in attr_value:
                if getattr(entry, "name", None) == "detection":
                    seg_app = entry
                    break
        elif isinstance(attr_value, dict):
            seg_app = attr_value.get("detection", seg_app)
        elif getattr(attr_value, "name", None) == "detection":
            seg_app = attr_value
        if seg_app is not None:
            break

    assert seg_app is not None, "could not locate the registered `detection` sub-app"
    sub_payload = repr(vars(seg_app))
    assert "train" in sub_payload
    assert "evaluate" in sub_payload
