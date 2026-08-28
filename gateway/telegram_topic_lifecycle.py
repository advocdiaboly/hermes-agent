"""Gateway-owned Telegram topic lifecycle for the current session."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_context import get_session_context_value


_GATEWAY_CALL_TIMEOUT_SECONDS = 15.0
_GATEWAY_CANCEL_GRACE_SECONDS = 0.25
_PRIVATE_CHAT_TYPES = frozenset({"dm", "private"})
_GROUP_CHAT_TYPES = frozenset({"group", "forum", "supergroup"})


@dataclass(frozen=True)
class _CurrentTelegramContext:
    platform: str
    chat_id: str
    chat_type: str
    thread_id: str
    user_id: str
    message_id: str
    session_id: str
    session_key: str
    profile: str

    @classmethod
    def capture(cls) -> "_CurrentTelegramContext":
        return cls(
            platform=get_session_context_value("HERMES_SESSION_PLATFORM", ""),
            chat_id=get_session_context_value("HERMES_SESSION_CHAT_ID", ""),
            chat_type=get_session_context_value("HERMES_SESSION_CHAT_TYPE", ""),
            thread_id=get_session_context_value("HERMES_SESSION_THREAD_ID", ""),
            user_id=get_session_context_value("HERMES_SESSION_USER_ID", ""),
            message_id=get_session_context_value("HERMES_SESSION_MESSAGE_ID", ""),
            session_id=get_session_context_value("HERMES_SESSION_ID", ""),
            session_key=get_session_context_value("HERMES_SESSION_KEY", ""),
            profile=get_session_context_value("HERMES_SESSION_PROFILE", ""),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "platform": self.platform,
            "chat_id": self.chat_id,
            "chat_type": self.chat_type,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "session_key": self.session_key,
            "profile": self.profile,
        }

    def to_source(self) -> SessionSource:
        return SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=self.chat_id,
            chat_type=self.chat_type,
            thread_id=self.thread_id or None,
            user_id=self.user_id or None,
            message_id=self.message_id or None,
            profile=self.profile or None,
        )


def _failure(operation: str, code: str, message: str) -> dict[str, Any]:
    return {
        "success": False,
        "operation": operation,
        "error": {"code": code, "message": message},
    }


def _surface(context: _CurrentTelegramContext) -> str:
    if not context.thread_id:
        return "plain_chat"
    chat_type = context.chat_type.strip().lower()
    if chat_type in _PRIVATE_CHAT_TYPES:
        return "private_topic"
    if chat_type in _GROUP_CHAT_TYPES:
        return "supergroup_forum_topic"
    return "plain_chat"


def _binding_matches_routing(
    binding: Optional[dict[str, Any]],
    context: _CurrentTelegramContext,
) -> bool:
    if not binding:
        return False
    checks = []
    if context.session_id:
        checks.append(str(binding.get("session_id") or "") == context.session_id)
    if context.session_key:
        checks.append(str(binding.get("session_key") or "") == context.session_key)
    return bool(checks) and all(checks)


def _binding_semantics(surface: str) -> tuple[str, str]:
    if surface == "private_topic":
        return "private_topic_binding", "native_private_topic_routing"
    if surface == "supergroup_forum_topic":
        return "metadata_only", "metadata_only"
    return "none", "none"


def _binding_status(
    binding: Optional[dict[str, Any]], *, matches: bool, surface: str
) -> str:
    if binding is None:
        return "missing"
    if matches:
        return "matches_routing"
    if surface == "supergroup_forum_topic":
        return "metadata_only_mismatch"
    return "mismatch"


def _coerce_chat_id(value: str) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _static_value(owner: Any, name: str) -> Any:
    """Read a real attribute without MagicMock synthesizing a capability."""
    if owner is None:
        return None
    try:
        inspect.getattr_static(owner, name)
    except AttributeError:
        return None
    try:
        return getattr(owner, name)
    except Exception:
        return None


def _static_callable(owner: Any, *names: str):
    for name in names:
        candidate = _static_value(owner, name)
        if callable(candidate):
            return candidate
    return None


def _active_runner():
    from gateway import run as gateway_run

    ref = getattr(gateway_run, "_gateway_runner_ref", None)
    return ref() if callable(ref) else None


def _validate_context(
    operation: str,
    context: _CurrentTelegramContext,
    *,
    require_thread: bool = False,
) -> Optional[dict[str, Any]]:
    if context.platform != "telegram":
        return _failure(
            operation,
            "not_telegram_context",
            "A current Telegram session context is required.",
        )
    if not context.chat_id:
        return _failure(
            operation,
            "missing_chat_id",
            "The current Telegram context has no authoritative chat ID.",
        )
    if require_thread and not context.thread_id:
        return _failure(
            operation,
            "missing_thread_id",
            "The current Telegram context has no authoritative topic thread ID.",
        )
    return None


def _lifecycle_task_registry(runner: Any) -> set[asyncio.Task]:
    registry = getattr(runner, "_telegram_lifecycle_tasks", None)
    if registry is None:
        registry = set()
        runner._telegram_lifecycle_tasks = registry
    return registry


class _GatewayLoopSubmission:
    """Own one worker-thread submission until loop-side cleanup completes."""

    def __init__(
        self,
        operation: str,
        runner: Any,
        loop: asyncio.AbstractEventLoop,
        work: Callable[[Any, SessionSource, Any], Awaitable[dict[str, Any]]],
        source: SessionSource,
        context: _CurrentTelegramContext,
    ) -> None:
        self.operation = operation
        self.runner = runner
        self.loop = loop
        self.work = work
        self.source = source
        self.context = context
        self.result: concurrent.futures.Future = concurrent.futures.Future()
        self.task: Optional[asyncio.Task] = None
        self.cleanup_complete = threading.Event()
        self._cancel_requested = False
        self._state_lock = threading.Lock()

    def submit(self) -> bool:
        try:
            self.loop.call_soon_threadsafe(self._start_on_loop)
            return True
        except Exception:
            self.cleanup_complete.set()
            return False

    def _start_on_loop(self) -> None:
        with self._state_lock:
            if self._cancel_requested:
                self.cleanup_complete.set()
                return
        if not getattr(self.runner, "_telegram_lifecycle_accepting", True):
            if not self.result.done():
                self.result.set_result(
                    _failure(
                        self.operation,
                        "gateway_runner_unavailable",
                        "The live gateway runner is shutting down.",
                    )
                )
            self.cleanup_complete.set()
            return

        coroutine = None
        try:
            coroutine = self.work(self.runner, self.source, self.context)
            task = self.loop.create_task(coroutine)
        except Exception as exc:
            if inspect.iscoroutine(coroutine):
                coroutine.close()
            if not self.result.done():
                self.result.set_exception(exc)
            self.cleanup_complete.set()
            return

        self.task = task
        _lifecycle_task_registry(self.runner).add(task)
        task.add_done_callback(self._task_done_on_loop)

    def _task_done_on_loop(self, task: asyncio.Task) -> None:
        _lifecycle_task_registry(self.runner).discard(task)
        try:
            value = task.result()
        except asyncio.CancelledError:
            if not self.result.done():
                self.result.set_exception(concurrent.futures.CancelledError())
        except Exception as exc:
            if not self.result.done():
                self.result.set_exception(exc)
        else:
            if not self.result.done():
                self.result.set_result(value)
        finally:
            self.cleanup_complete.set()

    def request_cancel(self) -> None:
        with self._state_lock:
            self._cancel_requested = True
        if not self.loop.is_closed() and self.loop.is_running():
            try:
                self.loop.call_soon_threadsafe(self._cancel_on_loop)
            except Exception:
                pass
        self.cleanup_complete.wait(timeout=_GATEWAY_CANCEL_GRACE_SECONDS)

    def _cancel_on_loop(self) -> None:
        if self.task is None:
            self.cleanup_complete.set()
        elif not self.task.done():
            self.task.cancel()


async def cancel_gateway_telegram_lifecycle_tasks(runner: Any) -> None:
    """Stop admission, then cancel and reap runner-owned lifecycle tasks."""
    runner._telegram_lifecycle_accepting = False
    registry = _lifecycle_task_registry(runner)
    tasks = [task for task in list(registry) if not task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    registry.difference_update(tasks)


def _run_on_gateway_loop(
    operation: str,
    context: _CurrentTelegramContext,
    work: Callable[[Any, SessionSource, Any], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    context_error = _validate_context(
        operation,
        context,
        require_thread=operation == "rename_topic",
    )
    if context_error is not None:
        return context_error

    runner = _active_runner()
    if runner is None or (
        getattr(runner, "_draining", False)
        or not getattr(runner, "_telegram_lifecycle_accepting", True)
    ):
        return _failure(
            operation,
            "gateway_runner_unavailable",
            "The live gateway runner is unavailable or shutting down.",
        )

    loop = getattr(runner, "_gateway_loop", None)
    if loop is None or loop.is_closed() or not loop.is_running():
        return _failure(
            operation,
            "gateway_loop_unavailable",
            "The live gateway event loop is unavailable.",
        )
    try:
        if asyncio.get_running_loop() is loop:
            return _failure(
                operation,
                "gateway_loop_thread",
                "A synchronous lifecycle call cannot block the gateway loop thread.",
            )
    except RuntimeError:
        pass

    submission = _GatewayLoopSubmission(
        operation,
        runner,
        loop,
        work,
        context.to_source(),
        context,
    )
    if not submission.submit():
        return _failure(
            operation,
            "gateway_loop_unavailable",
            "The lifecycle operation could not be scheduled on the gateway loop.",
        )

    try:
        return submission.result.result(timeout=_GATEWAY_CALL_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        submission.request_cancel()
        return _failure(
            operation,
            "gateway_timeout",
            "The Telegram lifecycle operation timed out.",
        )
    except Exception:
        return _failure(
            operation,
            "gateway_operation_error",
            "The gateway lifecycle operation failed.",
        )


def _adapter_for_source(runner: Any, source: SessionSource):
    try:
        return runner._adapter_for_source(source)
    except Exception:
        return None


async def _status_on_gateway_loop(
    runner: Any,
    source: SessionSource,
    context: _CurrentTelegramContext,
) -> dict[str, Any]:
    if _adapter_for_source(runner, source) is None:
        return _failure(
            "status",
            "telegram_adapter_unavailable",
            "The profile-specific Telegram adapter is unavailable.",
        )

    surface = _surface(context)
    binding = None
    topic_mode_enabled: Optional[bool] = None
    session_db = getattr(runner, "_session_db", None)
    try:
        if session_db is not None and context.thread_id:
            binding = await session_db.get_telegram_topic_binding(
                chat_id=context.chat_id,
                thread_id=context.thread_id,
            )
        if (
            session_db is not None
            and context.user_id
            and context.chat_type.strip().lower() in _PRIVATE_CHAT_TYPES
        ):
            topic_mode_enabled = bool(
                await session_db.is_telegram_topic_mode_enabled(
                    chat_id=context.chat_id,
                    user_id=context.user_id,
                )
            )
    except Exception:
        return _failure(
            "status",
            "session_state_unavailable",
            "Native Telegram session state is unavailable.",
        )

    matches = _binding_matches_routing(binding, context)
    authority, routing_effect = _binding_semantics(surface)
    context_dict = context.to_dict()
    return {
        "success": True,
        "operation": "status",
        "context": context_dict,
        "surface": surface,
        "routing": dict(context_dict),
        "binding": binding,
        "binding_status": _binding_status(binding, matches=matches, surface=surface),
        "binding_matches_routing": matches,
        "binding_authority": authority,
        "binding_routing_effect": routing_effect,
        "private_topic_mode_enabled": topic_mode_enabled,
    }


def _native_title(runner: Any, title: str) -> str:
    sanitize = getattr(runner, "_sanitize_telegram_topic_title", None)
    return str(sanitize(title) or "") if callable(sanitize) else ""


async def _create_on_gateway_loop(
    runner: Any,
    source: SessionSource,
    context: _CurrentTelegramContext,
    title: str,
) -> dict[str, Any]:
    adapter = _adapter_for_source(runner, source)
    if adapter is None:
        return _failure(
            "create_topic",
            "telegram_adapter_unavailable",
            "The profile-specific Telegram adapter is unavailable.",
        )
    title = _native_title(runner, title)
    if not title:
        return _failure(
            "create_topic",
            "unsupported_capability",
            "The gateway has no native Telegram title sanitizer.",
        )

    chat_type = context.chat_type.strip().lower()
    try:
        if chat_type in _PRIVATE_CHAT_TYPES:
            bot = _static_value(adapter, "_bot")
            if _static_callable(bot, "create_forum_topic", "createForumTopic") is None:
                create_topic = None
            else:
                create_topic = _static_callable(adapter, "_create_dm_topic")
            if create_topic is None:
                return _failure(
                    "create_topic",
                    "unsupported_capability",
                    "The active Telegram adapter cannot create private topics.",
                )
            thread_id = await create_topic(_coerce_chat_id(context.chat_id), title)
        elif chat_type in _GROUP_CHAT_TYPES:
            create_topic = _static_callable(
                _static_value(adapter, "_bot"),
                "create_forum_topic",
                "createForumTopic",
            )
            if create_topic is None:
                return _failure(
                    "create_topic",
                    "unsupported_capability",
                    "The active Telegram adapter cannot create forum topics.",
                )
            topic = await create_topic(
                chat_id=_coerce_chat_id(context.chat_id),
                name=title,
            )
            thread_id = getattr(topic, "message_thread_id", None)
        else:
            return _failure(
                "create_topic",
                "unsupported_chat_type",
                "The current Telegram chat type does not support topic creation.",
            )
    except Exception:
        return _failure(
            "create_topic",
            "telegram_api_error",
            "Telegram rejected the topic creation request.",
        )

    if not thread_id:
        return _failure(
            "create_topic",
            "telegram_api_error",
            "Telegram did not return a topic thread ID.",
        )
    return {
        "success": True,
        "operation": "create_topic",
        "surface": _surface(context),
        "chat_id": context.chat_id,
        "thread_id": str(thread_id),
        "title": title,
        "binding_created": False,
        "routing_effect": "topic_created_only",
    }


async def _rename_on_gateway_loop(
    runner: Any,
    source: SessionSource,
    context: _CurrentTelegramContext,
    title: str,
) -> dict[str, Any]:
    adapter = _adapter_for_source(runner, source)
    if adapter is None:
        return _failure(
            "rename_topic",
            "telegram_adapter_unavailable",
            "The profile-specific Telegram adapter is unavailable.",
        )
    title = _native_title(runner, title)
    if not title:
        return _failure(
            "rename_topic",
            "unsupported_capability",
            "The gateway has no native Telegram title sanitizer.",
        )

    chat_type = context.chat_type.strip().lower()
    try:
        if chat_type in _PRIVATE_CHAT_TYPES:
            bot = _static_value(adapter, "_bot")
            if _static_callable(bot, "edit_forum_topic", "editForumTopic") is None:
                rename_topic = None
            else:
                rename_topic = _static_callable(adapter, "rename_dm_topic")
            if rename_topic is None:
                return _failure(
                    "rename_topic",
                    "unsupported_capability",
                    "The active Telegram adapter cannot rename private topics.",
                )
            await rename_topic(
                chat_id=context.chat_id,
                thread_id=context.thread_id,
                name=title,
            )
        elif chat_type in _GROUP_CHAT_TYPES:
            rename_topic = _static_callable(
                _static_value(adapter, "_bot"),
                "edit_forum_topic",
                "editForumTopic",
            )
            if rename_topic is None:
                return _failure(
                    "rename_topic",
                    "unsupported_capability",
                    "The active Telegram adapter cannot rename forum topics.",
                )
            await rename_topic(
                chat_id=_coerce_chat_id(context.chat_id),
                message_thread_id=_coerce_chat_id(context.thread_id),
                name=title,
            )
        else:
            return _failure(
                "rename_topic",
                "unsupported_chat_type",
                "The current Telegram chat type does not support topic rename.",
            )
    except Exception:
        return _failure(
            "rename_topic",
            "telegram_api_error",
            "Telegram rejected the topic rename request.",
        )

    return {
        "success": True,
        "operation": "rename_topic",
        "surface": _surface(context),
        "chat_id": context.chat_id,
        "thread_id": context.thread_id,
        "title": title,
        "binding_created": False,
        "routing_effect": "topic_renamed_only",
    }


def get_current_telegram_topic_status() -> dict[str, Any]:
    context = _CurrentTelegramContext.capture()
    return _run_on_gateway_loop("status", context, _status_on_gateway_loop)


def create_current_telegram_topic(title: str) -> dict[str, Any]:
    context = _CurrentTelegramContext.capture()

    async def create(
        runner: Any,
        source: SessionSource,
        current: _CurrentTelegramContext,
    ) -> dict[str, Any]:
        return await _create_on_gateway_loop(runner, source, current, title)

    return _run_on_gateway_loop("create_topic", context, create)


def rename_current_telegram_topic(title: str) -> dict[str, Any]:
    context = _CurrentTelegramContext.capture()

    async def rename(
        runner: Any,
        source: SessionSource,
        current: _CurrentTelegramContext,
    ) -> dict[str, Any]:
        return await _rename_on_gateway_loop(runner, source, current, title)

    return _run_on_gateway_loop("rename_topic", context, rename)
