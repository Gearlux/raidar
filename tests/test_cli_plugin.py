"""Tests for raidar's marainer CLI plugin registration."""

from liquifai import LiquifyApp

from raidar.cli_plugin import register


def test_register_adds_two_ultralytics_commands_to_app() -> None:
    app = LiquifyApp(name="marainer")
    register(app)
    assert "ultralytics-train" in app._commands
    assert "ultralytics-evaluate" in app._commands


def test_registered_commands_are_script_commands_with_auto_flow() -> None:
    """script_command sets ``__liquifai_flow_mode__`` to "auto" — without
    that, deferred ``Class`` stubs in the YAML wouldn't get flowed into
    live ``UltralyticsTrainer`` / ``UltralyticsEvaluator`` instances before
    the command body runs.
    """
    app = LiquifyApp(name="marainer")
    register(app)
    train_cmd = app._commands["ultralytics-train"]
    eval_cmd = app._commands["ultralytics-evaluate"]
    assert getattr(train_cmd, "__liquifai_flow_mode__", None) == "auto"
    assert getattr(eval_cmd, "__liquifai_flow_mode__", None) == "auto"
    assert "ultralytics-train" in app._script_cmds
    assert "ultralytics-evaluate" in app._script_cmds
