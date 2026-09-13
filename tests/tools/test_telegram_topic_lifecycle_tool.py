import inspect
import json
from unittest.mock import patch

import pytest


def _tool():
    try:
        from tools import telegram_topic_lifecycle_tool
    except ModuleNotFoundError:
        pytest.fail("tools.telegram_topic_lifecycle_tool is missing")
    return telegram_topic_lifecycle_tool


def _contains_forbidden_field(value):
    if isinstance(value, dict):
        return any("pin" in str(key).lower() or _contains_forbidden_field(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_field(item) for item in value)
    return False


def test_schemas_expose_only_status_and_create_or_rename_title():
    tool = _tool()

    assert tool.TELEGRAM_TOPIC_STATUS_SCHEMA["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    parameters = tool.TELEGRAM_TOPIC_MANAGE_SCHEMA["parameters"]
    assert set(parameters["properties"]) == {"action", "title"}
    assert parameters["properties"]["action"]["enum"] == [
        "create_topic",
        "rename_topic",
    ]
    assert set(parameters["required"]) == {"action", "title"}
    assert parameters["additionalProperties"] is False
    assert list(inspect.signature(tool.telegram_topic_manage).parameters) == ["args"]
    assert not _contains_forbidden_field(tool.TELEGRAM_TOPIC_STATUS_SCHEMA)
    assert not _contains_forbidden_field(tool.TELEGRAM_TOPIC_MANAGE_SCHEMA)


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"action": "create_topic"},
        {"action": "rename_topic", "title": ""},
        {"action": "delete_topic", "title": "Release"},
        {"action": "create_topic", "title": "Release", "chat_id": "123"},
    ],
)
def test_invalid_arguments_fail_before_approval_or_mutation(args):
    tool = _tool()

    with patch.object(tool, "request_elicitation_consent") as consent, patch.object(
        tool, "create_current_telegram_topic"
    ) as create, patch.object(tool, "rename_current_telegram_topic") as rename:
        result = json.loads(tool.telegram_topic_manage(args))

    assert result["success"] is False
    consent.assert_not_called()
    create.assert_not_called()
    rename.assert_not_called()


@pytest.mark.parametrize("decision", ["decline", "cancel", TimeoutError("timeout")])
def test_non_acceptance_or_approval_error_prevents_mutation(decision):
    tool = _tool()
    consent_kwargs = (
        {"side_effect": decision} if isinstance(decision, Exception) else {"return_value": decision}
    )

    with patch.object(tool, "request_elicitation_consent", **consent_kwargs), patch.object(
        tool, "rename_current_telegram_topic"
    ) as rename:
        result = json.loads(
            tool.telegram_topic_manage(
                {"action": "rename_topic", "title": "Release planning"}
            )
        )

    assert result == {
        "success": False,
        "error": "Telegram topic change was not approved.",
    }
    rename.assert_not_called()


@pytest.mark.parametrize(
    ("action", "service_name"),
    [
        ("create_topic", "create_current_telegram_topic"),
        ("rename_topic", "rename_current_telegram_topic"),
    ],
)
def test_acceptance_calls_only_matching_service(action, service_name):
    tool = _tool()
    other_name = (
        "rename_current_telegram_topic"
        if service_name == "create_current_telegram_topic"
        else "create_current_telegram_topic"
    )

    with patch.object(
        tool, "request_elicitation_consent", return_value="accept"
    ) as consent, patch.object(
        tool, service_name, return_value={"success": True, "operation": action}
    ) as service, patch.object(tool, other_name) as other:
        result = json.loads(
            tool.telegram_topic_manage(
                {"action": action, "title": "Release planning"}
            )
        )

    assert result == {"success": True, "operation": action}
    service.assert_called_once_with("Release planning")
    other.assert_not_called()
    message, description = consent.call_args.args
    assert "chat_id" not in message + description
    assert "thread_id" not in message + description
    assert consent.call_args.kwargs == {"surface": "telegram-topic-lifecycle"}


@pytest.mark.parametrize("unsafe", ["Release\x1b[31m", "Safe\u202etxt.exe"])
def test_control_or_bidi_title_fails_before_approval(unsafe):
    tool = _tool()

    with patch.object(tool, "request_elicitation_consent") as consent, patch.object(
        tool, "rename_current_telegram_topic"
    ) as rename:
        result = json.loads(
            tool.telegram_topic_manage({"action": "rename_topic", "title": unsafe})
        )

    assert result["success"] is False
    consent.assert_not_called()
    rename.assert_not_called()


def test_approval_preview_is_bounded_but_service_receives_verbatim_title():
    tool = _tool()
    title = "Release\nplanning " + ("x" * 140)

    with patch.object(
        tool, "request_elicitation_consent", return_value="accept"
    ) as consent, patch.object(
        tool,
        "rename_current_telegram_topic",
        return_value={"success": True},
    ) as rename:
        tool.telegram_topic_manage({"action": "rename_topic", "title": title})

    message, _description = consent.call_args.args
    assert "\n" not in message
    assert '..."' in message
    assert len(message) < 220
    rename.assert_called_once_with(title)
