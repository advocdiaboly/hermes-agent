"""Thin tools for the current Telegram topic lifecycle."""

from __future__ import annotations

import json
import unicodedata
from typing import Any

from gateway.telegram_topic_lifecycle import (
    create_current_telegram_topic,
    get_current_telegram_topic_status,
    rename_current_telegram_topic,
)
from tools.approval import request_elicitation_consent
from tools.registry import registry


_TOOLSET = "telegram_topic_lifecycle"
_APPROVAL_SURFACE = "telegram-topic-lifecycle"
_UNSAFE_BIDI_CONTROLS = frozenset(
    {
        "\u061c",  # ARABIC LETTER MARK
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\u202a",  # LEFT-TO-RIGHT EMBEDDING
        "\u202b",  # RIGHT-TO-LEFT EMBEDDING
        "\u202c",  # POP DIRECTIONAL FORMATTING
        "\u202d",  # LEFT-TO-RIGHT OVERRIDE
        "\u202e",  # RIGHT-TO-LEFT OVERRIDE
        "\u2066",  # LEFT-TO-RIGHT ISOLATE
        "\u2067",  # RIGHT-TO-LEFT ISOLATE
        "\u2068",  # FIRST STRONG ISOLATE
        "\u2069",  # POP DIRECTIONAL ISOLATE
    }
)
_ACTION_VERBS = {
    "create_topic": "Create",
    "rename_topic": "Rename",
}


TELEGRAM_TOPIC_STATUS_SCHEMA = {
    "name": "telegram_topic_status",
    "description": (
        "Inspect topic lifecycle status for the authoritative current Telegram "
        "session. The gateway owns all routing and profile selection."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


TELEGRAM_TOPIC_MANAGE_SCHEMA = {
    "name": "telegram_topic_manage",
    "description": (
        "Create or rename the topic for the authoritative current Telegram "
        "session after a fresh user approval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create_topic", "rename_topic"],
                "description": "Topic lifecycle operation.",
            },
            "title": {
                "type": "string",
                "minLength": 1,
                "description": "Proposed Telegram topic title.",
            },
        },
        "required": ["action", "title"],
        "additionalProperties": False,
    },
}


def _error(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _approval_title_preview(title: str) -> str:
    """Mirror the gateway's whitespace/length normalization for safe display."""
    preview = " ".join(title.split())
    if len(preview) > 120:
        preview = preview[:117].rstrip() + "..."
    return json.dumps(preview, ensure_ascii=False)


def telegram_topic_status(args: dict[str, Any]) -> str:
    """Return structured status for the current Telegram context."""
    if args:
        return _error("telegram_topic_status does not accept arguments.")
    return json.dumps(get_current_telegram_topic_status(), ensure_ascii=False)


def telegram_topic_manage(args: dict[str, Any]) -> str:
    """Request one-operation consent, then perform the selected mutation."""
    if not isinstance(args, dict) or set(args) != {"action", "title"}:
        return _error("Both action and title are required; no other arguments are allowed.")

    action = args.get("action")
    title = args.get("title")
    if action not in _ACTION_VERBS:
        return _error("action must be create_topic or rename_topic.")
    if not isinstance(title, str) or not title.strip():
        return _error("title must be a non-empty string.")
    if any(
        unicodedata.category(character) == "Cc" and not character.isspace()
        for character in title
    ):
        return _error("title must not contain control characters.")
    if any(character in _UNSAFE_BIDI_CONTROLS for character in title):
        return _error("title must not contain bidirectional format controls.")

    verb = _ACTION_VERBS[action]
    try:
        decision = request_elicitation_consent(
            (
                f"{verb} the current Telegram topic with proposed title "
                f"{_approval_title_preview(title)}."
            ),
            "Telegram topic lifecycle change for the current session.",
            surface=_APPROVAL_SURFACE,
        )
    except Exception:
        decision = "decline"
    if decision != "accept":
        return _error("Telegram topic change was not approved.")

    if action == "create_topic":
        result = create_current_telegram_topic(title)
    else:
        result = rename_current_telegram_topic(title)
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="telegram_topic_status",
    toolset=_TOOLSET,
    schema=TELEGRAM_TOPIC_STATUS_SCHEMA,
    handler=lambda args, **_kwargs: telegram_topic_status(args),
    emoji="💬",
)

registry.register(
    name="telegram_topic_manage",
    toolset=_TOOLSET,
    schema=TELEGRAM_TOPIC_MANAGE_SCHEMA,
    handler=lambda args, **_kwargs: telegram_topic_manage(args),
    emoji="💬",
)
