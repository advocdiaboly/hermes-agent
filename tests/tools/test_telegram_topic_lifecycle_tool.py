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


def test_status_schema_has_no_arguments_and_delegates_once():
    tool = _tool()

    assert tool.TELEGRAM_TOPIC_STATUS_SCHEMA["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    with patch.object(
        tool,
        "get_current_telegram_topic_status",
        return_value={"success": True, "surface": "private_topic"},
    ) as status:
        result = json.loads(tool.telegram_topic_status({}))

    assert result == {"success": True, "surface": "private_topic"}
    status.assert_called_once_with()


def test_registry_dispatch_accepts_execution_context_kwargs():
    tool = _tool()

    with patch.object(
        tool,
        "get_current_telegram_topic_status",
        return_value={"success": True},
    ):
        result = json.loads(
            tool.registry.dispatch(
                "telegram_topic_status",
                {},
                task_id="task-1",
                session_id="session-1",
            )
        )

    assert result == {"success": True}


def test_manage_schema_exposes_only_action_and_required_title():
    tool = _tool()
    parameters = tool.TELEGRAM_TOPIC_MANAGE_SCHEMA["parameters"]

    assert set(parameters["properties"]) == {"action", "title"}
    assert parameters["properties"]["action"]["enum"] == [
        "create_topic",
        "rename_topic",
    ]
    assert set(parameters["required"]) == {"action", "title"}
    assert parameters["additionalProperties"] is False
    assert list(inspect.signature(tool.telegram_topic_manage).parameters) == [
        "args"
    ]


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"action": "create_topic"},
        {"action": "rename_topic", "title": ""},
        {"action": "rename_topic", "title": "   "},
        {"action": "delete_topic", "title": "Release"},
        {"action": "create_topic", "title": 42},
        {"action": "create_topic", "title": "Release", "chat_id": "123"},
    ],
)
def test_invalid_manage_arguments_fail_before_approval_or_mutation(args):
    tool = _tool()

    with patch.object(
        tool, "request_elicitation_consent"
    ) as consent, patch.object(
        tool, "create_current_telegram_topic"
    ) as create, patch.object(
        tool, "rename_current_telegram_topic"
    ) as rename:
        result = json.loads(tool.telegram_topic_manage(args))

    assert result["success"] is False
    assert "error" in result
    consent.assert_not_called()
    create.assert_not_called()
    rename.assert_not_called()


@pytest.mark.parametrize("decision", ["decline", "cancel"])
def test_non_acceptance_fails_before_mutation(decision):
    tool = _tool()

    with patch.object(
        tool, "request_elicitation_consent", return_value=decision
    ) as consent, patch.object(
        tool, "create_current_telegram_topic"
    ) as create:
        result = json.loads(
            tool.telegram_topic_manage(
                {"action": "create_topic", "title": "Release planning"}
            )
        )

    assert result["success"] is False
    assert result["error"] == "Telegram topic change was not approved."
    consent.assert_called_once()
    create.assert_not_called()


def test_approval_exception_fails_before_mutation():
    tool = _tool()

    with patch.object(
        tool,
        "request_elicitation_consent",
        side_effect=TimeoutError("approval timed out"),
    ) as consent, patch.object(
        tool, "rename_current_telegram_topic"
    ) as rename:
        result = json.loads(
            tool.telegram_topic_manage(
                {"action": "rename_topic", "title": "Release planning"}
            )
        )

    assert result["success"] is False
    assert result["error"] == "Telegram topic change was not approved."
    consent.assert_called_once()
    rename.assert_not_called()


@pytest.mark.parametrize(
    ("action", "service_name"),
    [
        ("create_topic", "create_current_telegram_topic"),
        ("rename_topic", "rename_current_telegram_topic"),
    ],
)
def test_acceptance_calls_only_the_matching_service(action, service_name):
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
    ) as service, patch.object(
        tool, other_name
    ) as other:
        result = json.loads(
            tool.telegram_topic_manage(
                {"action": action, "title": "Release planning"}
            )
        )

    assert result == {"success": True, "operation": action}
    service.assert_called_once_with("Release planning")
    other.assert_not_called()
    consent.assert_called_once()
    message, description = consent.call_args.args
    assert "chat_id" not in message + description
    assert "thread_id" not in message + description
    assert action.removesuffix("_topic").capitalize() in message
    assert '"Release planning"' in message
    assert consent.call_args.kwargs == {"surface": "telegram-topic-lifecycle"}


def test_approval_uses_bounded_control_safe_effective_title_preview():
    tool = _tool()
    title = ("Release\nplanning " + ("x" * 140))

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
    assert '"Release planning ' in message
    assert '..."' in message
    assert len(message) < 220
    rename.assert_called_once_with(title)


def test_control_character_title_fails_before_approval_or_mutation():
    tool = _tool()

    with patch.object(
        tool, "request_elicitation_consent"
    ) as consent, patch.object(
        tool, "rename_current_telegram_topic"
    ) as rename:
        result = json.loads(
            tool.telegram_topic_manage(
                {"action": "rename_topic", "title": "Release\x1b[31m"}
            )
        )

    assert result["success"] is False
    consent.assert_not_called()
    rename.assert_not_called()


@pytest.mark.parametrize(
    "control",
    [
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    ],
)
def test_bidi_format_control_title_fails_before_approval_or_mutation(control):
    tool = _tool()

    with patch.object(
        tool, "request_elicitation_consent"
    ) as consent, patch.object(
        tool, "rename_current_telegram_topic"
    ) as rename:
        result = json.loads(
            tool.telegram_topic_manage(
                {"action": "rename_topic", "title": f"Safe{control}txt.exe"}
            )
        )

    assert result["success"] is False
    consent.assert_not_called()
    rename.assert_not_called()


def test_normal_emoji_zwj_title_is_approved_and_applied_verbatim():
    tool = _tool()
    title = "Family 👨\u200d👩\u200d👧 planning"

    with patch.object(
        tool, "request_elicitation_consent", return_value="accept"
    ) as consent, patch.object(
        tool,
        "rename_current_telegram_topic",
        return_value={"success": True},
    ) as rename:
        result = json.loads(
            tool.telegram_topic_manage({"action": "rename_topic", "title": title})
        )

    assert result["success"] is True
    assert title in consent.call_args.args[0]
    rename.assert_called_once_with(title)


def test_each_mutation_requests_fresh_consent():
    tool = _tool()

    with patch.object(
        tool,
        "request_elicitation_consent",
        side_effect=["accept", "decline"],
    ) as consent, patch.object(
        tool,
        "rename_current_telegram_topic",
        return_value={"success": True},
    ) as rename:
        first = json.loads(
            tool.telegram_topic_manage(
                {"action": "rename_topic", "title": "First title"}
            )
        )
        second = json.loads(
            tool.telegram_topic_manage(
                {"action": "rename_topic", "title": "Second title"}
            )
        )

    assert first["success"] is True
    assert second["success"] is False
    assert consent.call_count == 2
    rename.assert_called_once_with("First title")
