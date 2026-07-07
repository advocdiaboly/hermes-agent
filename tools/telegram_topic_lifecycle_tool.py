"""Managed Telegram topic lifecycle tools.

These tools are intentionally narrow. Discussion, naming policy, and consent
live in skills/operator conversation; this module only performs auditable
Telegram topic/session and pinned-marker primitives.
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
        "topic binding, and Hermes-known pinned marker metadata. The known "
        "pins list is an index/cache, not a complete Telegram pinned history."
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
            "include_unpinned": {
                "type": "boolean",
                "description": "Include known markers that Hermes has marked unpinned.",
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
                    "update_metadata",
                ],
            },
            "chat_id": {"type": "string"},
            "thread_id": {"type": "string"},
            "title": {"type": "string"},
            "session_id": {"type": "string"},
            "session_key": {"type": "string"},
            "message_id": {"type": "string"},
            "kind": {"type": "string"},
            "label": {"type": "string"},
            "text_snapshot": {"type": "string"},
            "metadata": {"type": "object"},
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


TELEGRAM_PIN_MANAGE_SCHEMA = {
    "name": "telegram_pin_manage",
    "description": (
        "Manage Hermes-known Telegram pinned markers for the current managed "
        "topic. Mutating actions require approval metadata. list_known_pins "
        "returns only markers Hermes created or explicitly recorded."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "post_and_pin",
                    "pin_existing",
                    "unpin",
                    "record_pin_metadata",
                    "list_known_pins",
                ],
            },
            "chat_id": {"type": "string"},
            "thread_id": {"type": "string"},
            "message_id": {"type": "string"},
            "message": {"type": "string"},
            "kind": {"type": "string"},
            "label": {"type": "string"},
            "text_snapshot": {"type": "string"},
            "metadata": {"type": "object"},
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

    def send_message(self, *, chat_id: str, thread_id: Optional[str], message: str) -> str:
        kwargs: Dict[str, Any] = {
            "chat_id": _coerce_chat_id(chat_id),
            "text": message,
        }
        if thread_id:
            try:
                kwargs["direct_messages_topic_id"] = int(thread_id)
                sent = _run_maybe_awaitable(self._bot.send_message(**kwargs))
            except TypeError:
                kwargs.pop("direct_messages_topic_id", None)
                if str(thread_id) != "1":
                    kwargs["message_thread_id"] = int(thread_id)
                sent = _run_maybe_awaitable(self._bot.send_message(**kwargs))
        else:
            sent = _run_maybe_awaitable(self._bot.send_message(**kwargs))
        return str(getattr(sent, "message_id"))

    def pin_message(self, *, chat_id: str, message_id: str) -> bool:
        _run_maybe_awaitable(
            self._bot.pin_chat_message(
                chat_id=_coerce_chat_id(chat_id),
                message_id=int(message_id),
                disable_notification=True,
            )
        )
        return True

    def unpin_message(self, *, chat_id: str, message_id: str) -> bool:
        _run_maybe_awaitable(
            self._bot.unpin_chat_message(
                chat_id=_coerce_chat_id(chat_id),
                message_id=int(message_id),
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


def _known_pins(db: SessionDB, ctx: Dict[str, Optional[str]], *, include_unpinned: bool = False):
    chat_id = ctx.get("chat_id")
    if not chat_id:
        return []
    return db.list_telegram_pinned_markers(
        chat_id=chat_id,
        thread_id=ctx.get("thread_id"),
        include_unpinned=include_unpinned,
    )


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

    return _json(
        {
            "success": True,
            "context": ctx,
            "topic_mode_enabled": topic_mode_enabled,
            "binding": binding,
            "known_pins": _known_pins(
                db,
                ctx,
                include_unpinned=bool(args.get("include_unpinned")),
            ),
            "known_pins_complete": False,
            "limitations": [
                "known_pins is a Hermes-maintained index/cache, not the complete Telegram pinned-message list"
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
        return _json(
            {
                "success": True,
                "action": action,
                "binding": binding,
                "approval": approval,
            }
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

    if action == "update_metadata":
        message_id = str(args.get("message_id") or ctx.get("message_id") or "").strip()
        kind = str(args.get("kind") or "").strip()
        if not message_id:
            return _error("message_id is required for update_metadata", action=action)
        if not kind:
            return _error("kind is required for update_metadata", action=action)
        marker = db.record_telegram_pinned_marker(
            chat_id=ctx["chat_id"],
            thread_id=ctx.get("thread_id"),
            message_id=message_id,
            kind=kind,
            label=args.get("label"),
            text_snapshot=args.get("text_snapshot"),
            session_id=ctx.get("session_id"),
            metadata=args.get("metadata") if isinstance(args.get("metadata"), dict) else None,
            approved_by=approval["approved_by"],
            approval_ref=approval["approval_ref"],
            approval_text_excerpt=approval["approval_text_excerpt"],
        )
        return _json(
            {
                "success": True,
                "action": action,
                "marker": marker,
                "approval": approval,
                "known_pins_complete": False,
            }
        )

    return _error(f"unknown telegram_topic_manage action: {action}", action=action)


def telegram_pin_manage(args: Dict[str, Any], **kwargs) -> str:
    action = str(args.get("action") or "").strip()
    db = _get_session_db(kwargs)
    client = _get_telegram_client(kwargs)
    ctx = _context(args)
    context_error = _require_telegram_context(ctx)
    if context_error:
        return _error(context_error, action=action, context=ctx)

    if action == "list_known_pins":
        return _json(
            {
                "success": True,
                "action": action,
                "known_pins": _known_pins(
                    db,
                    ctx,
                    include_unpinned=bool(args.get("include_unpinned")),
                ),
                "known_pins_complete": False,
            }
        )

    approval = _approval(args)
    if approval is None:
        return _error("approval metadata is required for mutating Telegram pinned-marker actions", action=action)

    if action == "record_pin_metadata":
        message_id = str(args.get("message_id") or "").strip()
        kind = str(args.get("kind") or "").strip()
        if not message_id:
            return _error("message_id is required for record_pin_metadata", action=action)
        if not kind:
            return _error("kind is required for record_pin_metadata", action=action)
        marker = db.record_telegram_pinned_marker(
            chat_id=ctx["chat_id"],
            thread_id=ctx.get("thread_id"),
            message_id=message_id,
            kind=kind,
            label=args.get("label"),
            text_snapshot=args.get("text_snapshot"),
            session_id=ctx.get("session_id"),
            metadata=args.get("metadata") if isinstance(args.get("metadata"), dict) else None,
            approved_by=approval["approved_by"],
            approval_ref=approval["approval_ref"],
            approval_text_excerpt=approval["approval_text_excerpt"],
        )
        return _json(
            {
                "success": True,
                "action": action,
                "marker": marker,
                "approval": approval,
                "known_pins_complete": False,
            }
        )

    if action == "post_and_pin":
        if client is None:
            return _error("telegram client is not available for post_and_pin", action=action)
        message = str(args.get("message") or "").strip()
        kind = str(args.get("kind") or "").strip()
        if not message:
            return _error("message is required for post_and_pin", action=action)
        if not kind:
            return _error("kind is required for post_and_pin", action=action)
        message_id = str(
            client.send_message(
                chat_id=ctx["chat_id"],
                thread_id=ctx.get("thread_id"),
                message=message,
            )
        )
        client.pin_message(chat_id=ctx["chat_id"], message_id=message_id)
        marker = db.record_telegram_pinned_marker(
            chat_id=ctx["chat_id"],
            thread_id=ctx.get("thread_id"),
            message_id=message_id,
            kind=kind,
            label=args.get("label"),
            text_snapshot=args.get("text_snapshot") or message,
            session_id=ctx.get("session_id"),
            metadata=args.get("metadata") if isinstance(args.get("metadata"), dict) else None,
            approved_by=approval["approved_by"],
            approval_ref=approval["approval_ref"],
            approval_text_excerpt=approval["approval_text_excerpt"],
        )
        return _json(
            {
                "success": True,
                "action": action,
                "marker": marker,
                "approval": approval,
                "known_pins_complete": False,
            }
        )

    if action == "pin_existing":
        if client is None:
            return _error("telegram client is not available for pin_existing", action=action)
        message_id = str(args.get("message_id") or "").strip()
        kind = str(args.get("kind") or "").strip()
        if not message_id:
            return _error("message_id is required for pin_existing", action=action)
        if not kind:
            return _error("kind is required for pin_existing", action=action)
        client.pin_message(chat_id=ctx["chat_id"], message_id=message_id)
        marker = db.record_telegram_pinned_marker(
            chat_id=ctx["chat_id"],
            thread_id=ctx.get("thread_id"),
            message_id=message_id,
            kind=kind,
            label=args.get("label"),
            text_snapshot=args.get("text_snapshot"),
            session_id=ctx.get("session_id"),
            metadata=args.get("metadata") if isinstance(args.get("metadata"), dict) else None,
            approved_by=approval["approved_by"],
            approval_ref=approval["approval_ref"],
            approval_text_excerpt=approval["approval_text_excerpt"],
        )
        return _json(
            {
                "success": True,
                "action": action,
                "marker": marker,
                "approval": approval,
                "known_pins_complete": False,
            }
        )

    if action == "unpin":
        if client is None:
            return _error("telegram client is not available for unpin", action=action)
        message_id = str(args.get("message_id") or "").strip()
        if not message_id:
            return _error("message_id is required for unpin", action=action)
        client.unpin_message(chat_id=ctx["chat_id"], message_id=message_id)
        marker = db.update_telegram_pinned_marker_state(
            chat_id=ctx["chat_id"],
            message_id=message_id,
            state="unpinned",
        )
        return _json(
            {
                "success": True,
                "action": action,
                "marker": marker,
                "approval": approval,
                "known_pins_complete": False,
            }
        )

    return _error(f"unknown telegram_pin_manage action: {action}", action=action)


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

registry.register(
    name="telegram_pin_manage",
    toolset=TOOLSET,
    schema=TELEGRAM_PIN_MANAGE_SCHEMA,
    handler=telegram_pin_manage,
    description=TELEGRAM_PIN_MANAGE_SCHEMA["description"],
)
