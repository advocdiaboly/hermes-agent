"""Managed Telegram topic lifecycle tools.

These tools are intentionally narrow. Discussion, naming policy, and consent
live in skills/operator conversation; this module only performs auditable
Telegram topic/session primitives.
"""

from __future__ import annotations

import json
import inspect
import os
from typing import Any, Dict, Optional

from hermes_state import SessionDB
from tools.registry import registry


TOOLSET = "telegram_topic_lifecycle"


TELEGRAM_TOPIC_STATUS_SCHEMA = {
    "name": "telegram_topic_status",
    "description": (
        "Inspect the current Telegram topic lifecycle context. Read-only. "
        "Returns current chat/thread/session identifiers, existing Hermes "
        "topic binding, and routing-vs-metadata status."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "Optional Telegram chat id. Defaults to the current gateway session context.",
            },
            "thread_id": {
                "type": "string",
                "description": "Optional Telegram topic/thread id. Defaults to the current gateway session context.",
            },
            "chat_type": {
                "type": "string",
                "description": "Optional Telegram chat type. Defaults to the current gateway session context.",
            },
        },
        "required": [],
    },
}


TELEGRAM_TOPIC_MANAGE_SCHEMA = {
    "name": "telegram_topic_manage",
    "description": (
        "Perform one approved managed Telegram topic/session primitive. "
        "Mutating actions require approval metadata from the operator "
        "conversation. This is not a raw Telegram Bot API surface."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "create_topic",
                    "rename_topic",
                    "bind_session",
                ],
            },
            "chat_id": {"type": "string"},
            "thread_id": {"type": "string"},
            "title": {"type": "string"},
            "session_id": {"type": "string"},
            "session_key": {"type": "string"},
            "approval": {
                "type": "object",
                "description": (
                    "Required for mutating actions. Include approved_by and "
                    "approval_text_excerpt; approval_ref is recommended."
                ),
                "properties": {
                    "approved_by": {"type": "string"},
                    "approval_ref": {"type": "string"},
                    "approval_text_excerpt": {"type": "string"},
                },
            },
        },
        "required": ["action"],
    },
}


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _error(message: str, **extra: Any) -> str:
    return _json({"success": False, "error": message, **extra})


def _get_session_db(kwargs: Dict[str, Any]) -> SessionDB:
    db = kwargs.get("session_db")
    if db is not None:
        return db
    return SessionDB()


def _run_maybe_awaitable(value):
    if not inspect.isawaitable(value):
        return value
    try:
        from model_tools import _run_async
    except ImportError:
        import asyncio

        return asyncio.run(value)
    return _run_async(value)


def _coerce_chat_id(value: str):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


class TelegramBotApiLifecycleClient:
    """Small Bot API wrapper for managed-topic lifecycle primitives."""

    def __init__(self, token: str):
        from telegram import Bot

        self._bot = Bot(token=token)

    @classmethod
    def from_env(cls) -> Optional["TelegramBotApiLifecycleClient"]:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            return None
        try:
            return cls(token)
        except Exception:
            return None

    def create_topic(self, *, chat_id: str, title: str) -> str:
        topic = _run_maybe_awaitable(
            self._bot.create_forum_topic(
                chat_id=_coerce_chat_id(chat_id),
                name=title,
            )
        )
        return str(getattr(topic, "message_thread_id"))

    def rename_topic(self, *, chat_id: str, thread_id: str, title: str) -> bool:
        _run_maybe_awaitable(
            self._bot.edit_forum_topic(
                chat_id=_coerce_chat_id(chat_id),
                message_thread_id=int(thread_id),
                name=title,
            )
        )
        return True


def _get_telegram_client(kwargs: Dict[str, Any]):
    client = kwargs.get("telegram_client")
    if client is not None:
        return client
    return TelegramBotApiLifecycleClient.from_env()


def _session_env(name: str, default: str = "") -> str:
    try:
        from gateway.session_context import get_session_env

        return get_session_env(name, default) or ""
    except Exception:
        return default


def _context(args: Dict[str, Any]) -> Dict[str, Optional[str]]:
    return {
        "platform": _session_env("HERMES_SESSION_PLATFORM"),
        "chat_id": str(args.get("chat_id") or _session_env("HERMES_SESSION_CHAT_ID") or "") or None,
        "chat_type": str(args.get("chat_type") or _session_env("HERMES_SESSION_CHAT_TYPE") or "") or None,
        "thread_id": str(args.get("thread_id") or _session_env("HERMES_SESSION_THREAD_ID") or "") or None,
        "user_id": str(args.get("user_id") or _session_env("HERMES_SESSION_USER_ID") or "") or None,
        "session_key": str(args.get("session_key") or _session_env("HERMES_SESSION_KEY") or "") or None,
        "session_id": str(args.get("session_id") or _session_env("HERMES_SESSION_ID") or "") or None,
        "message_id": str(args.get("message_id") or _session_env("HERMES_SESSION_MESSAGE_ID") or "") or None,
    }


def _require_telegram_context(ctx: Dict[str, Optional[str]]) -> Optional[str]:
    if ctx.get("platform") != "telegram":
        return "telegram_topic_lifecycle tools require a current Telegram session context"
    if not ctx.get("chat_id"):
        return "telegram chat_id is required"
    return None


def _approval(args: Dict[str, Any]) -> Optional[Dict[str, Optional[str]]]:
    raw = args.get("approval")
    if not isinstance(raw, dict):
        return None
    approved_by = str(raw.get("approved_by") or "").strip()
    excerpt = str(raw.get("approval_text_excerpt") or "").strip()
    if not approved_by or not excerpt:
        return None
    return {
        "approved_by": approved_by,
        "approval_ref": str(raw.get("approval_ref") or "").strip() or None,
        "approval_text_excerpt": excerpt,
    }


def _topic_surface(ctx: Dict[str, Optional[str]]) -> str:
    chat_type = str(ctx.get("chat_type") or "").strip().lower()
    chat_id = str(ctx.get("chat_id") or "").strip()
    session_key = str(ctx.get("session_key") or "").strip().lower()
    has_thread = bool(ctx.get("thread_id"))
    if not has_thread:
        return "plain_chat"
    is_group_key = ":telegram:group:" in session_key or session_key.startswith("telegram:group:")
    is_dm_key = ":telegram:dm:" in session_key or session_key.startswith("telegram:dm:")
    if chat_type in {"group", "forum", "supergroup"} or is_group_key:
        return "supergroup_forum_topic"
    if chat_type in {"dm", "private"} or is_dm_key:
        return "private_topic"
    if chat_id.startswith("-"):
        return "supergroup_forum_topic"
    if chat_id:
        return "private_topic"
    return "plain_chat"


def _routing_context(ctx: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    return {
        "chat_id": ctx.get("chat_id"),
        "thread_id": ctx.get("thread_id"),
        "session_id": ctx.get("session_id"),
        "session_key": ctx.get("session_key"),
    }


def _binding_matches_routing(
    binding: Optional[Dict[str, Any]],
    routing: Dict[str, Optional[str]],
) -> bool:
    if not binding:
        return False
    checks = []
    if routing.get("session_id"):
        checks.append(str(binding.get("session_id") or "") == routing["session_id"])
    if routing.get("session_key"):
        checks.append(str(binding.get("session_key") or "") == routing["session_key"])
    return bool(checks) and all(checks)


def _binding_status(
    binding: Optional[Dict[str, Any]],
    *,
    matches_routing: bool,
    surface: str,
) -> str:
    if binding is None:
        return "missing"
    if matches_routing:
        return "matches_routing"
    if surface == "supergroup_forum_topic":
        return "metadata_only_mismatch"
    return "mismatch"


def _binding_authority(surface: str) -> str:
    if surface == "supergroup_forum_topic":
        return "metadata_only"
    if surface == "private_topic":
        return "private_topic_binding"
    return "none"


def _bind_session_routing_effect(surface: str) -> str:
    if surface == "private_topic":
        return "private_topic_binding"
    if surface == "supergroup_forum_topic":
        return "metadata_only"
    return "metadata_only"


def telegram_topic_status(args: Dict[str, Any], **kwargs) -> str:
    db = _get_session_db(kwargs)
    ctx = _context(args)
    context_error = _require_telegram_context(ctx)
    if context_error:
        return _error(context_error, context=ctx)

    chat_id = ctx["chat_id"]
    thread_id = ctx.get("thread_id")
    user_id = ctx.get("user_id")
    binding = None
    if chat_id and thread_id:
        binding = db.get_telegram_topic_binding(chat_id=chat_id, thread_id=thread_id)

    topic_mode_enabled = False
    if chat_id and user_id:
        topic_mode_enabled = db.is_telegram_topic_mode_enabled(
            chat_id=chat_id,
            user_id=user_id,
        )
    surface = _topic_surface(ctx)
    routing = _routing_context(ctx)
    binding_matches = _binding_matches_routing(binding, routing)

    return _json(
        {
            "success": True,
            "context": ctx,
            "surface": surface,
            "routing": routing,
            "topic_mode_enabled": topic_mode_enabled,
            "private_topic_mode_enabled": topic_mode_enabled,
            "binding": binding,
            "binding_status": _binding_status(
                binding,
                matches_routing=binding_matches,
                surface=surface,
            ),
            "binding_matches_routing": binding_matches,
            "binding_authority": _binding_authority(surface),
            "limitations": [
                "supergroup/forum topic bindings are lifecycle metadata unless binding_status is matches_routing",
                "Telegram pinned-message memory is not exposed as a supported topic lifecycle primitive",
            ],
        }
    )


def telegram_topic_manage(args: Dict[str, Any], **kwargs) -> str:
    action = str(args.get("action") or "").strip()
    db = _get_session_db(kwargs)
    client = _get_telegram_client(kwargs)
    ctx = _context(args)
    context_error = _require_telegram_context(ctx)
    if context_error:
        return _error(context_error, action=action, context=ctx)

    approval = _approval(args)
    if approval is None:
        return _error("approval metadata is required for mutating Telegram topic actions", action=action)

    if action == "bind_session":
        if not ctx.get("thread_id"):
            return _error("thread_id is required to bind a Telegram topic", action=action)
        session_id = str(args.get("session_id") or ctx.get("session_id") or "").strip()
        if not session_id:
            return _error("session_id is required to bind a Telegram topic", action=action)
        session_key = str(args.get("session_key") or ctx.get("session_key") or "").strip()
        if not session_key:
            session_key = f"telegram:dm:{ctx['chat_id']}:{ctx['thread_id']}"
        db.bind_telegram_topic(
            chat_id=ctx["chat_id"],
            thread_id=ctx["thread_id"],
            user_id=ctx.get("user_id") or ctx["chat_id"],
            session_key=session_key,
            session_id=session_id,
            managed_mode="managed",
        )
        binding = db.get_telegram_topic_binding(
            chat_id=ctx["chat_id"],
            thread_id=ctx["thread_id"],
        )
        surface = _topic_surface(ctx)
        response = {
            "success": True,
            "action": action,
            "surface": surface,
            "binding": binding,
            "binding_authority": _binding_authority(surface),
            "routing_effect": _bind_session_routing_effect(surface),
            "approval": approval,
        }
        if surface == "supergroup_forum_topic":
            response["warning"] = (
                "supergroup/forum bind_session records lifecycle metadata only; "
                "it does not change gateway routing"
            )
        return _json(
            response
        )

    if action == "create_topic":
        title = str(args.get("title") or "").strip()
        if not title:
            return _error("title is required to create a Telegram topic", action=action)
        if client is None:
            return _error("telegram client is not available for create_topic", action=action)
        thread_id = str(
            client.create_topic(
                chat_id=ctx["chat_id"],
                title=title,
            )
        )
        return _json(
            {
                "success": True,
                "action": action,
                "chat_id": ctx["chat_id"],
                "thread_id": thread_id,
                "title": title,
                "binding_created": False,
                "routing_effect": "topic_created_only",
                "approval": approval,
            }
        )

    if action == "rename_topic":
        title = str(args.get("title") or "").strip()
        thread_id = str(args.get("thread_id") or ctx.get("thread_id") or "").strip()
        if not title:
            return _error("title is required to rename a Telegram topic", action=action)
        if not thread_id:
            return _error("thread_id is required to rename a Telegram topic", action=action)
        if client is None:
            return _error("telegram client is not available for rename_topic", action=action)
        client.rename_topic(
            chat_id=ctx["chat_id"],
            thread_id=thread_id,
            title=title,
        )
        return _json(
            {
                "success": True,
                "action": action,
                "chat_id": ctx["chat_id"],
                "thread_id": thread_id,
                "title": title,
                "approval": approval,
            }
        )

    return _error(f"unknown telegram_topic_manage action: {action}", action=action)


registry.register(
    name="telegram_topic_status",
    toolset=TOOLSET,
    schema=TELEGRAM_TOPIC_STATUS_SCHEMA,
    handler=telegram_topic_status,
    description=TELEGRAM_TOPIC_STATUS_SCHEMA["description"],
)

registry.register(
    name="telegram_topic_manage",
    toolset=TOOLSET,
    schema=TELEGRAM_TOPIC_MANAGE_SCHEMA,
    handler=telegram_topic_manage,
    description=TELEGRAM_TOPIC_MANAGE_SCHEMA["description"],
)
