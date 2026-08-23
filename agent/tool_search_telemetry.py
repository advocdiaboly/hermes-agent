"""Private, best-effort per-turn telemetry for native tool search."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import hashlib
import json
import logging
import os
import re
import stat
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_EVENT_KIND = "tool_search_turn_finish"
_MAX_JSONL_RECORD_BYTES = 16 * 1024
_TURN_ID_MAX_CHARS = 512
_WRITE_LOCK = threading.Lock()
_TRACKERS_LOCK = threading.Lock()
_TRACKERS_BY_TURN_ID: dict[str, "_TurnTracker"] = {}
_CURRENT_TRACKER: contextvars.ContextVar[Optional["_TurnTracker"]] = (
    contextvars.ContextVar("tool_search_turn_telemetry", default=None)
)
_SAFE_REASON_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


@dataclass(frozen=True)
class ToolSearchAssemblySnapshot:
    """Metadata for the exact tool-schema assembly used by a turn."""

    state: str
    activated: bool
    deferred_count: int
    deferred_schema_tokens: int
    activation_threshold_tokens: int

    @classmethod
    def active(
        cls,
        *,
        deferred_count: int,
        deferred_schema_tokens: int,
        activation_threshold_tokens: int,
    ) -> "ToolSearchAssemblySnapshot":
        return cls(
            state="active",
            activated=True,
            deferred_count=max(0, int(deferred_count)),
            deferred_schema_tokens=max(0, int(deferred_schema_tokens)),
            activation_threshold_tokens=max(0, int(activation_threshold_tokens)),
        )

    @classmethod
    def control(
        cls,
        *,
        deferred_count: int,
        deferred_schema_tokens: int,
        activation_threshold_tokens: int,
    ) -> "ToolSearchAssemblySnapshot":
        return cls(
            state="control",
            activated=False,
            deferred_count=max(0, int(deferred_count)),
            deferred_schema_tokens=max(0, int(deferred_schema_tokens)),
            activation_threshold_tokens=max(0, int(activation_threshold_tokens)),
        )

    @classmethod
    def off(
        cls,
        *,
        deferred_count: int = 0,
        deferred_schema_tokens: int = 0,
        activation_threshold_tokens: int = 0,
    ) -> "ToolSearchAssemblySnapshot":
        return cls(
            state="off",
            activated=False,
            deferred_count=max(0, int(deferred_count)),
            deferred_schema_tokens=max(0, int(deferred_schema_tokens)),
            activation_threshold_tokens=max(0, int(activation_threshold_tokens)),
        )

    @classmethod
    def unknown(cls) -> "ToolSearchAssemblySnapshot":
        return cls(
            state="unknown",
            activated=False,
            deferred_count=0,
            deferred_schema_tokens=0,
            activation_threshold_tokens=0,
        )


class _AnnotatedToolDefinition(dict):
    def __init__(
        self,
        value: Mapping[str, Any],
        snapshot: ToolSearchAssemblySnapshot,
    ) -> None:
        super().__init__(value)
        self._tool_search_assembly = snapshot


class _AnnotatedToolDefinitions(list):
    def __init__(
        self,
        values: Iterable[Mapping[str, Any]],
        snapshot: ToolSearchAssemblySnapshot,
    ) -> None:
        super().__init__(values)
        self._tool_search_assembly = snapshot


def attach_assembly_snapshot(
    tool_defs: Iterable[Mapping[str, Any]],
    snapshot: ToolSearchAssemblySnapshot,
) -> list[dict[str, Any]]:
    """Attach metadata without adding provider-visible schema keys."""

    annotated = [_AnnotatedToolDefinition(tool_def, snapshot) for tool_def in tool_defs]
    return _AnnotatedToolDefinitions(annotated, snapshot)


def extract_assembly_snapshot(
    tool_defs: Optional[Iterable[Mapping[str, Any]]],
) -> ToolSearchAssemblySnapshot:
    """Read metadata from the list or a surviving schema element."""

    snapshot = getattr(tool_defs, "_tool_search_assembly", None)
    if isinstance(snapshot, ToolSearchAssemblySnapshot):
        return snapshot
    for tool_def in tool_defs or ():
        snapshot = getattr(tool_def, "_tool_search_assembly", None)
        if isinstance(snapshot, ToolSearchAssemblySnapshot):
            return snapshot
    return ToolSearchAssemblySnapshot.unknown()


def copy_tool_definitions(
    tool_defs: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy a tool list while preserving out-of-band metadata."""

    copied = list(tool_defs)
    snapshot = extract_assembly_snapshot(tool_defs)
    if snapshot.state == "unknown":
        return copied
    return _AnnotatedToolDefinitions(copied, snapshot)


@dataclass
class _Usage:
    api_calls: int = 0
    first_prompt_tokens: int = 0
    prompt_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class _TurnTracker:
    registry_turn_id: str
    turn_id: str
    session_id: str
    user_hash: str
    model: str
    provider: str
    assembly: ToolSearchAssemblySnapshot
    path: Path
    usage: _Usage = field(default_factory=_Usage)
    bridge_calls: dict[str, int] = field(
        default_factory=lambda: {
            "tool_search": 0,
            "tool_describe": 0,
            "tool_call": 0,
        }
    )
    finalized: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


def begin_turn_scope() -> contextvars.Token:
    return _CURRENT_TRACKER.set(None)


def bind_turn_scope(turn_id: str) -> Optional[contextvars.Token]:
    tracker = _tracker_for(turn_id)
    if tracker is None:
        return None
    return _CURRENT_TRACKER.set(tracker)


def current_turn_id() -> Optional[str]:
    tracker = _CURRENT_TRACKER.get()
    return tracker.registry_turn_id if tracker is not None else None


def end_turn_scope(token: contextvars.Token) -> None:
    try:
        _CURRENT_TRACKER.reset(token)
    except Exception:
        logger.debug("tool-search telemetry scope reset failed", exc_info=True)


def _bounded_text(value: Any, limit: int = 256) -> str:
    return str(value or "")[:limit]


def _bounded_identifier(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return f"{text[: limit - len(digest) - 1]}~{digest}"


def _normalized_user_hash(user_message: Any) -> str:
    if isinstance(user_message, str):
        normalized = unicodedata.normalize("NFKC", user_message)
        normalized = " ".join(normalized.split()).casefold()
    else:
        normalized = f"<{type(user_message).__name__}>"
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def default_path() -> Path:
    override = os.environ.get("HERMES_TOOL_SEARCH_TELEMETRY_PATH")
    if override:
        return Path(os.path.expandvars(os.path.expanduser(override)))
    return get_hermes_home() / "logs" / "tool-search-turns.jsonl"


def start_turn(
    *,
    turn_id: str,
    session_id: str,
    user_message: Any,
    tools: Optional[Iterable[Mapping[str, Any]]],
    model: str,
    provider: str,
    path: Optional[os.PathLike[str] | str] = None,
) -> None:
    registry_turn_id = str(turn_id or "")
    tracker = _TurnTracker(
        registry_turn_id=registry_turn_id,
        turn_id=_bounded_identifier(registry_turn_id, _TURN_ID_MAX_CHARS),
        session_id=_bounded_text(session_id),
        user_hash=_normalized_user_hash(user_message),
        model=_bounded_text(model, 160),
        provider=_bounded_text(provider, 80),
        assembly=extract_assembly_snapshot(tools),
        path=Path(path) if path is not None else default_path(),
    )
    _CURRENT_TRACKER.set(tracker)
    with _TRACKERS_LOCK:
        _TRACKERS_BY_TURN_ID[tracker.registry_turn_id] = tracker


def _tracker_for(turn_id: Optional[str]) -> Optional[_TurnTracker]:
    if turn_id:
        with _TRACKERS_LOCK:
            return _TRACKERS_BY_TURN_ID.get(str(turn_id))
    return _CURRENT_TRACKER.get()


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def note_api_call(*, turn_id: Optional[str] = None) -> None:
    tracker = _tracker_for(turn_id)
    if tracker is None:
        return
    with tracker.lock:
        if not tracker.finalized:
            tracker.usage.api_calls += 1


def note_api_usage(
    *,
    prompt_tokens: Any,
    input_tokens: Any,
    output_tokens: Any,
    completion_tokens: Any,
    total_tokens: Any,
    cache_read_tokens: Any,
    cache_write_tokens: Any,
    reasoning_tokens: Any,
    turn_id: Optional[str] = None,
) -> None:
    tracker = _tracker_for(turn_id)
    if tracker is None:
        return
    values = {
        "prompt_tokens": _nonnegative_int(prompt_tokens),
        "input_tokens": _nonnegative_int(input_tokens),
        "output_tokens": _nonnegative_int(output_tokens),
        "completion_tokens": _nonnegative_int(completion_tokens),
        "total_tokens": _nonnegative_int(total_tokens),
        "cache_read_tokens": _nonnegative_int(cache_read_tokens),
        "cache_write_tokens": _nonnegative_int(cache_write_tokens),
        "reasoning_tokens": _nonnegative_int(reasoning_tokens),
    }
    with tracker.lock:
        if tracker.finalized:
            return
        if tracker.usage.first_prompt_tokens == 0 and values["prompt_tokens"] > 0:
            tracker.usage.first_prompt_tokens = values["prompt_tokens"]
        for name, value in values.items():
            setattr(tracker.usage, name, getattr(tracker.usage, name) + value)


def note_bridge_call(name: str, *, turn_id: Optional[str] = None) -> None:
    if name not in {"tool_search", "tool_describe", "tool_call"}:
        return
    tracker = _tracker_for(turn_id)
    if tracker is None:
        return
    with tracker.lock:
        if not tracker.finalized:
            tracker.bridge_calls[name] += 1


def _safe_exit_reason(value: Any) -> str:
    reason = str(value or "unknown").split("(", 1)[0].strip()
    reason = _SAFE_REASON_RE.sub("_", reason).strip("_")
    return (reason or "unknown")[:80]


def _event_id(turn_id: str) -> str:
    digest = hashlib.sha256(turn_id.encode("utf-8", errors="replace")).hexdigest()
    return f"turn_finish:{digest}"


def _is_interruption_exception(exception: BaseException) -> bool:
    return isinstance(
        exception,
        (
            KeyboardInterrupt,
            InterruptedError,
            asyncio.CancelledError,
            concurrent.futures.CancelledError,
        ),
    )


def _build_record(
    tracker: _TurnTracker,
    result: Optional[Mapping[str, Any]],
    exception: Optional[BaseException],
) -> dict[str, Any]:
    result = result if isinstance(result, Mapping) else {}
    if exception is not None:
        interrupted = _is_interruption_exception(exception)
        outcome = {
            "completed": False,
            "failed": not interrupted,
            "interrupted": interrupted,
            "partial": True,
            "exit_reason": f"exception:{type(exception).__name__}"[:80],
        }
    else:
        completed = bool(result.get("completed", False))
        interrupted = bool(result.get("interrupted", False))
        partial = bool(result.get("partial", False))
        has_error = bool(result.get("error"))
        failed = bool(result.get("failed", has_error and not interrupted))
        explicit_reason = result.get("turn_exit_reason")
        if explicit_reason:
            exit_reason = _safe_exit_reason(explicit_reason)
        elif interrupted:
            exit_reason = "interrupted"
        elif failed and partial:
            exit_reason = "partial_error"
        elif failed:
            exit_reason = "failed"
        elif partial:
            exit_reason = "partial"
        elif completed:
            exit_reason = "completed"
        else:
            exit_reason = "unknown"
        outcome = {
            "completed": completed,
            "failed": failed,
            "interrupted": interrupted,
            "partial": partial,
            "exit_reason": exit_reason,
        }
    return {
        "schema_version": _SCHEMA_VERSION,
        "event": _EVENT_KIND,
        "event_id": _event_id(tracker.registry_turn_id),
        "timestamp": time.time(),
        "turn_id": tracker.turn_id,
        "session_id": tracker.session_id,
        "user_hash": tracker.user_hash,
        "model": tracker.model,
        "provider": tracker.provider,
        "tool_search": {
            "state": tracker.assembly.state,
            "activated": tracker.assembly.activated,
            "deferred_count": tracker.assembly.deferred_count,
            "deferred_schema_tokens": tracker.assembly.deferred_schema_tokens,
            "activation_threshold_tokens": tracker.assembly.activation_threshold_tokens,
            "calls": dict(tracker.bridge_calls),
        },
        "usage": {
            "first_prompt_tokens": tracker.usage.first_prompt_tokens,
            "prompt_tokens": tracker.usage.prompt_tokens,
            "input_tokens": tracker.usage.input_tokens,
            "output_tokens": tracker.usage.output_tokens,
            "completion_tokens": tracker.usage.completion_tokens,
            "total_tokens": tracker.usage.total_tokens,
            "cache_read_tokens": tracker.usage.cache_read_tokens,
            "cache_write_tokens": tracker.usage.cache_write_tokens,
            "reasoning_tokens": tracker.usage.reasoning_tokens,
            "api_calls": tracker.usage.api_calls,
        },
        "outcome": outcome,
    }


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_JSONL_RECORD_BYTES:
        raise ValueError("tool-search telemetry record exceeded bounded size")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("telemetry append requires O_NOFOLLOW support")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    with _WRITE_LOCK:
        fd = os.open(path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("telemetry target is not a regular file")
            os.fchmod(fd, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("telemetry append made no progress")
                view = view[written:]
        finally:
            os.close(fd)


def finish_current_turn(
    result: Optional[Mapping[str, Any]] = None,
    *,
    exception: Optional[BaseException] = None,
) -> bool:
    tracker = _CURRENT_TRACKER.get()
    if tracker is None:
        return False
    try:
        with tracker.lock:
            if tracker.finalized:
                return False
            tracker.finalized = True
            record = _build_record(tracker, result, exception)
        _append_record(tracker.path, record)
        return True
    except Exception:
        logger.debug(
            "tool-search telemetry append failed for turn=%s",
            tracker.turn_id,
            exc_info=True,
        )
        return False
    finally:
        with _TRACKERS_LOCK:
            if _TRACKERS_BY_TURN_ID.get(tracker.registry_turn_id) is tracker:
                _TRACKERS_BY_TURN_ID.pop(tracker.registry_turn_id, None)


def read_records(path: os.PathLike[str] | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with Path(path).open("rb") as stream:
            while True:
                line = stream.readline(_MAX_JSONL_RECORD_BYTES + 1)
                if not line:
                    break
                if len(line) > _MAX_JSONL_RECORD_BYTES or not line.endswith(b"\n"):
                    while line and not line.endswith(b"\n"):
                        line = stream.readline(_MAX_JSONL_RECORD_BYTES + 1)
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if (
                    not isinstance(record, dict)
                    or record.get("event") != _EVENT_KIND
                    or not isinstance(record.get("event_id"), str)
                    or record["event_id"] in seen
                ):
                    continue
                seen.add(record["event_id"])
                records.append(record)
    except OSError:
        return []
    return records


def summarize(path: os.PathLike[str] | str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "turns": 0,
        "api_calls": 0,
        "prompt_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "tool_search_calls": 0,
        "assembly_states": {},
    }
    for record in read_records(path):
        usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
        tool_search = (
            record.get("tool_search")
            if isinstance(record.get("tool_search"), dict)
            else {}
        )
        calls = (
            tool_search.get("calls")
            if isinstance(tool_search.get("calls"), dict)
            else {}
        )
        state = str(tool_search.get("state") or "unknown")
        summary["turns"] += 1
        summary["api_calls"] += _nonnegative_int(usage.get("api_calls"))
        summary["prompt_tokens"] += _nonnegative_int(usage.get("prompt_tokens"))
        summary["input_tokens"] += _nonnegative_int(usage.get("input_tokens"))
        summary["output_tokens"] += _nonnegative_int(usage.get("output_tokens"))
        summary["total_tokens"] += _nonnegative_int(usage.get("total_tokens"))
        summary["tool_search_calls"] += _nonnegative_int(calls.get("tool_search"))
        summary["assembly_states"][state] = summary["assembly_states"].get(state, 0) + 1
    return summary
