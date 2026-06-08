"""Tests for raidar's native RF-DETR marainer CLI plugin (``raidar.rfdetr_plugin``)."""

from liquifai import LiquifyApp

from raidar.rfdetr_plugin import register


def test_register_adds_two_rfdetr_commands() -> None:
    app = LiquifyApp(name="marainer")
    register(app)
    assert "rfdetr-train" in app._commands
    assert "rfdetr-evaluate" in app._commands


def test_registered_commands_are_script_commands_with_auto_flow() -> None:
    # script_command(flow_mode="auto") sets __liquifai_flow_mode__ so the YAML's
    # deferred Class stubs get flowed into live RFDETRTrainer/RFDETREvaluator instances.
    app = LiquifyApp(name="marainer")
    register(app)
    assert getattr(app._commands["rfdetr-train"], "__liquifai_flow_mode__", None) == "auto"
    assert getattr(app._commands["rfdetr-evaluate"], "__liquifai_flow_mode__", None) == "auto"
    assert "rfdetr-train" in app._script_cmds
    assert "rfdetr-evaluate" in app._script_cmds
