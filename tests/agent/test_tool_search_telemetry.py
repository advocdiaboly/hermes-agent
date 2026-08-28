"""Behavior contracts for private per-turn tool-search telemetry."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} description",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _finish_turn(
    telemetry,
    *,
    path: Path,
    turn_id: str,
    prompt_tokens: list[int],
    assembly,
    user_message: str = "ordinary user message",
    result: dict | None = None,
) -> None:
    token = telemetry.begin_turn_scope()
    try:
        tools = telemetry.attach_assembly_snapshot([_tool("terminal")], assembly)
        telemetry.start_turn(
            turn_id=turn_id,
            session_id="session-1",
            user_message=user_message,
            tools=tools,
            model="test/model",
            provider="test",
            path=path,
        )
        for prompt in prompt_tokens:
            telemetry.note_api_call()
            telemetry.note_api_usage(
                prompt_tokens=prompt,
                input_tokens=max(0, prompt - 1),
                output_tokens=2,
                completion_tokens=2,
                total_tokens=prompt + 2,
                cache_read_tokens=1 if prompt else 0,
                cache_write_tokens=0,
                reasoning_tokens=1,
            )
        assert telemetry.finish_current_turn(
            result
            or {
                "completed": True,
                "interrupted": False,
                "failed": False,
                "partial": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(token)


def test_usage_is_turn_local_and_first_positive_prompt_is_recorded(tmp_path):
    """A stale tracker must not leak usage into the next turn."""
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "turns.jsonl"
    assembly = telemetry.ToolSearchAssemblySnapshot.control(
        deferred_count=4,
        deferred_schema_tokens=800,
        activation_threshold_tokens=2_000,
    )
    _finish_turn(
        telemetry,
        path=path,
        turn_id="turn-1",
        prompt_tokens=[0, 100, 130],
        assembly=assembly,
    )
    _finish_turn(
        telemetry,
        path=path,
        turn_id="turn-2",
        prompt_tokens=[40],
        assembly=assembly,
    )

    records = telemetry.read_records(path)
    assert [record["turn_id"] for record in records] == ["turn-1", "turn-2"]
    assert records[0]["usage"]["first_prompt_tokens"] == 100
    assert records[0]["usage"]["prompt_tokens"] == 230
    assert records[0]["usage"]["api_calls"] == 3
    assert records[1]["usage"]["prompt_tokens"] == 40
    assert records[1]["usage"]["api_calls"] == 1


def test_assembly_metadata_survives_copy_without_entering_provider_json():
    """Removing out-of-band metadata or leaking it into schemas is a bug."""
    from agent import tool_search_telemetry as telemetry

    snapshot = telemetry.ToolSearchAssemblySnapshot.active(
        deferred_count=12,
        deferred_schema_tokens=7_000,
        activation_threshold_tokens=5_000,
    )
    tools = telemetry.attach_assembly_snapshot([_tool("tool_search")], snapshot)
    copied = telemetry.copy_tool_definitions(tools)

    assert telemetry.extract_assembly_snapshot(tools) == snapshot
    assert telemetry.extract_assembly_snapshot(copied) == snapshot
    assert copied == [_tool("tool_search")]
    assert "_tool_search_assembly" not in json.dumps(copied)


def test_concurrent_turns_keep_independent_assembly_and_usage(tmp_path):
    """Context loss in worker threads must not cross-attribute turn data."""
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "turns.jsonl"
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker(turn_id: str, prompt_tokens: int, assembly) -> None:
        token = telemetry.begin_turn_scope()
        try:
            tools = telemetry.attach_assembly_snapshot([_tool("terminal")], assembly)
            telemetry.start_turn(
                turn_id=turn_id,
                session_id="shared-session",
                user_message=turn_id,
                tools=tools,
                model="test/model",
                provider="test",
                path=path,
            )
            barrier.wait(timeout=5)
            telemetry.note_api_call()
            telemetry.note_api_usage(
                prompt_tokens=prompt_tokens,
                input_tokens=prompt_tokens,
                output_tokens=1,
                completion_tokens=1,
                total_tokens=prompt_tokens + 1,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            )
            assert telemetry.finish_current_turn({"completed": True})
        except BaseException as exc:
            errors.append(exc)
        finally:
            telemetry.end_turn_scope(token)

    threads = [
        threading.Thread(
            target=worker,
            args=(
                "active",
                10,
                telemetry.ToolSearchAssemblySnapshot.active(
                    deferred_count=20,
                    deferred_schema_tokens=8_000,
                    activation_threshold_tokens=4_000,
                ),
            ),
        ),
        threading.Thread(
            target=worker,
            args=(
                "control",
                20,
                telemetry.ToolSearchAssemblySnapshot.control(
                    deferred_count=2,
                    deferred_schema_tokens=500,
                    activation_threshold_tokens=4_000,
                ),
            ),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    records = {record["turn_id"]: record for record in telemetry.read_records(path)}
    assert records["active"]["tool_search"]["state"] == "active"
    assert records["active"]["usage"]["prompt_tokens"] == 10
    assert records["control"]["tool_search"]["state"] == "control"
    assert records["control"]["usage"]["prompt_tokens"] == 20


def test_bridge_calls_require_the_authoritative_turn_id(tmp_path):
    """A bridge call from another turn must not increment this tracker."""
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "turns.jsonl"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-bridge",
            session_id="session-1",
            user_message="find a tool",
            tools=[],
            model="test/model",
            provider="test",
            path=path,
        )
        telemetry.note_bridge_call("tool_search", turn_id="other-turn")
        telemetry.note_bridge_call("tool_search", turn_id="turn-bridge")
        telemetry.note_bridge_call("tool_describe", turn_id="turn-bridge")
        telemetry.note_bridge_call("tool_call", turn_id="turn-bridge")
        assert telemetry.finish_current_turn({"completed": True})
        assert telemetry.finish_current_turn({"completed": True}) is False
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["tool_search"]["calls"] == {
        "tool_search": 1,
        "tool_describe": 1,
        "tool_call": 1,
    }


def test_record_is_private_and_contains_no_raw_text_or_error(tmp_path):
    """Raw prompts, secrets, and provider error details must never persist."""
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "private" / "turns.jsonl"
    secret = "sk-sentinel-secret-DO-NOT-WRITE"
    _finish_turn(
        telemetry,
        path=path,
        turn_id="turn-private",
        prompt_tokens=[25],
        assembly=telemetry.ToolSearchAssemblySnapshot.control(
            deferred_count=1,
            deferred_schema_tokens=50,
            activation_threshold_tokens=500,
        ),
        user_message=f"Please inspect confidential payload {secret}",
        result={
            "completed": False,
            "failed": True,
            "partial": True,
            "error": f"provider leaked {secret}",
            "turn_exit_reason": f"provider_error({secret})",
        },
    )

    payload = path.read_text(encoding="utf-8")
    assert "Please inspect confidential payload" not in payload
    assert secret not in payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    [record] = telemetry.read_records(path)
    assert len(record["user_hash"]) == 64
    assert record["outcome"]["exit_reason"] == "provider_error"


def test_long_turn_id_routes_by_full_value_but_persists_bounded(tmp_path):
    """A caller-controlled task ID must not make the finish record unwritable."""
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "bounded-id.jsonl"
    turn_id = "turn:" + ("x" * telemetry._MAX_JSONL_RECORD_BYTES)
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id=turn_id,
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="test",
            path=path,
        )
        telemetry.note_api_call(turn_id=turn_id)
        assert telemetry.finish_current_turn({"completed": True})
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 1
    assert len(record["turn_id"]) <= telemetry._TURN_ID_MAX_CHARS
    assert record["turn_id"] != turn_id
    assert len(path.read_bytes()) <= telemetry._MAX_JSONL_RECORD_BYTES


def test_short_writes_are_retried_until_one_record_is_complete(
    tmp_path,
    monkeypatch,
):
    """A short os.write must not leave a truncated JSONL record."""
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "short-write.jsonl"
    real_write = os.write
    writes: list[int] = []

    def short_write(fd, payload):
        writes.append(len(payload))
        return real_write(fd, payload[: max(1, len(payload) // 2)])

    monkeypatch.setattr(telemetry.os, "write", short_write)
    _finish_turn(
        telemetry,
        path=path,
        turn_id="turn-short-write",
        prompt_tokens=[],
        assembly=telemetry.ToolSearchAssemblySnapshot.off(),
    )

    assert len(writes) > 1
    assert telemetry.read_records(path)[0]["turn_id"] == "turn-short-write"


def test_symlink_target_fails_closed_without_touching_target(tmp_path):
    """Following a telemetry symlink would turn diagnostics into a file clobber."""
    from agent import tool_search_telemetry as telemetry

    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform has no O_NOFOLLOW")
    target = tmp_path / "target.txt"
    target.write_text("ORIGINAL\n", encoding="utf-8")
    target.chmod(0o644)
    link = tmp_path / "turns.jsonl"
    link.symlink_to(target)
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-symlink",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="test",
            path=link,
        )
        assert telemetry.finish_current_turn({"completed": True}) is False
    finally:
        telemetry.end_turn_scope(token)

    assert target.read_text(encoding="utf-8") == "ORIGINAL\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_reader_ignores_malformed_truncated_and_duplicate_lines(tmp_path):
    """One damaged append must not poison later offline summaries."""
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "turns.jsonl"
    _finish_turn(
        telemetry,
        path=path,
        turn_id="turn-summary",
        prompt_tokens=[50],
        assembly=telemetry.ToolSearchAssemblySnapshot.active(
            deferred_count=5,
            deferred_schema_tokens=1_000,
            activation_threshold_tokens=500,
        ),
    )
    valid_line = path.read_text(encoding="utf-8")
    with path.open("a", encoding="utf-8") as stream:
        stream.write("{truncated\n")
        stream.write(valid_line)

    assert telemetry.summarize(path) == {
        "turns": 1,
        "api_calls": 1,
        "prompt_tokens": 50,
        "input_tokens": 49,
        "output_tokens": 2,
        "total_tokens": 52,
        "tool_search_calls": 0,
        "assembly_states": {"active": 1},
    }


@pytest.mark.parametrize(
    ("exception", "failed", "interrupted"),
    [
        (RuntimeError("boom"), True, False),
        (KeyboardInterrupt(), False, True),
        (asyncio.CancelledError(), False, True),
    ],
)
def test_exception_outcomes_are_bounded_and_classified(
    tmp_path,
    exception,
    failed,
    interrupted,
):
    """Cancellation must not be reported as failure or persist exception text."""
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / f"{type(exception).__name__}.jsonl"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id=f"turn-{type(exception).__name__}",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="test",
            path=path,
        )
        telemetry.note_api_call()
        assert telemetry.finish_current_turn(exception=exception)
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["outcome"]["failed"] is failed
    assert record["outcome"]["interrupted"] is interrupted
    assert record["outcome"]["partial"] is True
    assert record["usage"]["api_calls"] == 1
    assert "boom" not in path.read_text(encoding="utf-8")


def test_model_tools_attaches_exact_active_and_control_assembly(monkeypatch):
    """Losing assembly metadata makes active and control turns indistinguishable."""
    import model_tools
    from agent import tool_search_telemetry as telemetry
    from tools import tool_search

    source_defs = [_tool("plugin_example")]
    monkeypatch.setattr(model_tools, "validate_toolset", lambda _name: True)
    monkeypatch.setattr(model_tools, "resolve_toolset", lambda _name: {"plugin_example"})
    monkeypatch.setattr(
        model_tools.registry,
        "get_definitions",
        lambda *_args, **_kwargs: list(source_defs),
    )
    monkeypatch.setattr(model_tools, "_resolve_active_context_length", lambda: 50_000)
    monkeypatch.setattr(
        tool_search,
        "load_config",
        lambda: tool_search.ToolSearchConfig.from_raw(
            {"enabled": "auto", "threshold_pct": 10}
        ),
    )
    monkeypatch.setattr(
        tool_search,
        "assemble_tool_defs",
        lambda *_args, **_kwargs: tool_search.AssemblyResult(
            tool_defs=[_tool("tool_search")],
            activated=True,
            deferred_count=9,
            deferred_tokens=6_000,
            threshold_tokens=5_000,
        ),
    )

    active_defs = model_tools._compute_tool_definitions(
        enabled_toolsets=["plugin-test"],
        quiet_mode=True,
    )
    active = telemetry.extract_assembly_snapshot(active_defs)
    assert active == telemetry.ToolSearchAssemblySnapshot.active(
        deferred_count=9,
        deferred_schema_tokens=6_000,
        activation_threshold_tokens=5_000,
    )

    monkeypatch.setattr(
        tool_search,
        "assemble_tool_defs",
        lambda *_args, **_kwargs: tool_search.AssemblyResult(
            tool_defs=source_defs,
            activated=False,
            deferred_count=2,
            deferred_tokens=400,
            threshold_tokens=5_000,
        ),
    )
    control_defs = model_tools._compute_tool_definitions(
        enabled_toolsets=["plugin-test"],
        quiet_mode=True,
    )
    control = telemetry.extract_assembly_snapshot(control_defs)
    assert control == telemetry.ToolSearchAssemblySnapshot.control(
        deferred_count=2,
        deferred_schema_tokens=400,
        activation_threshold_tokens=5_000,
    )


def test_model_tools_cache_preserves_empty_control_assembly(monkeypatch):
    """A shallow copy of an empty cached list must not erase cohort metadata."""
    import model_tools
    from agent import tool_search_telemetry as telemetry

    snapshot = telemetry.ToolSearchAssemblySnapshot.control(
        deferred_count=0,
        deferred_schema_tokens=0,
        activation_threshold_tokens=5_000,
    )
    monkeypatch.setattr(model_tools, "check_fn_cache_scope", lambda: "test-scope")
    monkeypatch.setattr(model_tools.registry, "current_scope_key", lambda: "test")
    monkeypatch.setattr(
        model_tools,
        "_compute_tool_definitions",
        lambda *_args, **_kwargs: telemetry.attach_assembly_snapshot([], snapshot),
    )
    model_tools._clear_tool_defs_cache()

    first = model_tools.get_tool_definitions(quiet_mode=True)
    second = model_tools.get_tool_definitions(quiet_mode=True)

    assert telemetry.extract_assembly_snapshot(first) == snapshot
    assert telemetry.extract_assembly_snapshot(second) == snapshot


def test_bridge_dispatch_records_the_matching_turn(monkeypatch):
    """Removing dispatch instrumentation must make bridge usage disappear."""
    import model_tools
    from agent import tool_search_telemetry as telemetry
    from tools import tool_search

    note_bridge_call = MagicMock()
    monkeypatch.setattr(telemetry, "note_bridge_call", note_bridge_call)
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_kwargs: [])
    monkeypatch.setattr(
        tool_search,
        "dispatch_tool_search",
        lambda *_args, **_kwargs: json.dumps({"results": []}),
    )

    result = model_tools.handle_function_call(
        function_name="tool_search",
        function_args={"query": "calendar"},
        turn_id="turn-dispatch",
    )

    assert json.loads(result) == {"results": []}
    note_bridge_call.assert_called_once_with(
        "tool_search",
        turn_id="turn-dispatch",
    )


def test_turn_context_starts_tracker_with_authoritative_turn_id(
    tmp_path,
    monkeypatch,
):
    """Starting telemetry after the prologue would lose early failures."""
    from agent import tool_search_telemetry as telemetry
    from tests.agent.test_turn_context import _FakeAgent, _build

    path = tmp_path / "turn-context.jsonl"
    monkeypatch.setenv("HERMES_TOOL_SEARCH_TELEMETRY_PATH", str(path))
    agent = _FakeAgent()
    agent.tools = telemetry.attach_assembly_snapshot(
        [_tool("terminal")],
        telemetry.ToolSearchAssemblySnapshot.control(
            deferred_count=3,
            deferred_schema_tokens=300,
            activation_threshold_tokens=1_000,
        ),
    )
    token = telemetry.begin_turn_scope()
    try:
        ctx = _build(agent, user_message="raw user text")
        assert telemetry.current_turn_id() == ctx.turn_id
        assert telemetry.finish_current_turn(
            {
                "completed": False,
                "interrupted": True,
                "failed": False,
                "partial": False,
                "turn_exit_reason": "interrupted_by_user",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["turn_id"] == ctx.turn_id
    assert record["session_id"] == agent.session_id
    assert record["tool_search"]["deferred_count"] == 3


def test_run_conversation_finalizes_partial_early_return(tmp_path, monkeypatch):
    """An early result must still emit exactly one bounded finish record."""
    from agent import conversation_loop
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "wrapped.jsonl"

    def fake_impl(agent, user_message, **_kwargs):
        telemetry.start_turn(
            turn_id="turn-wrapped-partial",
            session_id="session-1",
            user_message=user_message,
            tools=[],
            model="test/model",
            provider="test",
            path=path,
        )
        return {
            "completed": False,
            "failed": False,
            "interrupted": False,
            "partial": True,
            "turn_exit_reason": "truncated_response(provider detail)",
        }

    monkeypatch.setattr(conversation_loop, "_run_conversation_impl", fake_impl)
    result = conversation_loop.run_conversation(object(), "secret raw prompt")

    assert result["partial"] is True
    [record] = telemetry.read_records(path)
    assert record["outcome"]["exit_reason"] == "truncated_response"


def test_run_conversation_finalizes_failure_then_reraises(tmp_path, monkeypatch):
    """An exception must retain its identity while telemetry remains private."""
    from agent import conversation_loop
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "wrapped-failure.jsonl"
    raised = RuntimeError("sentinel secret must not persist")

    def fake_impl(agent, user_message, **_kwargs):
        telemetry.start_turn(
            turn_id="turn-wrapped-failure",
            session_id="session-1",
            user_message=user_message,
            tools=[],
            model="test/model",
            provider="test",
            path=path,
        )
        raise raised

    monkeypatch.setattr(conversation_loop, "_run_conversation_impl", fake_impl)
    with pytest.raises(RuntimeError) as caught:
        conversation_loop.run_conversation(object(), "raw prompt")

    assert caught.value is raised
    [record] = telemetry.read_records(path)
    assert record["outcome"]["exit_reason"] == "exception:RuntimeError"
    assert "sentinel secret" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "api_mode", "usage"),
    [
        (
            "openrouter",
            "chat_completions",
            SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
                prompt_tokens_details=SimpleNamespace(cached_tokens=20),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
            ),
        ),
        (
            "codex",
            "codex_responses",
            SimpleNamespace(
                input_tokens=100,
                output_tokens=10,
                total_tokens=110,
                input_tokens_details=SimpleNamespace(cached_tokens=20),
                output_tokens_details=SimpleNamespace(reasoning_tokens=3),
            ),
        ),
    ],
)
def test_relay_sync_attempt_normalizes_response_usage(
    tmp_path,
    monkeypatch,
    name,
    api_mode,
    usage,
):
    """The common provider boundary owns both attempts and normalized usage."""
    from agent import relay_llm, relay_runtime
    from agent import tool_search_telemetry as telemetry

    monkeypatch.setattr(
        relay_runtime,
        "resolve_execution_context",
        lambda _session_id: (None, None, None),
    )
    path = tmp_path / f"relay-{api_mode}.jsonl"
    response = SimpleNamespace(usage=usage, choices=[])
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id=f"turn-{api_mode}",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider=name,
            path=path,
        )
        assert relay_llm.execute(
            {"model": "test/model"},
            lambda _request: response,
            session_id="session-1",
            name=name,
            model_name="test/model",
            metadata={"api_mode": api_mode},
        ) is response
        assert telemetry.finish_current_turn({"completed": True})
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["usage"] == {
        "first_prompt_tokens": 100,
        "prompt_tokens": 100,
        "input_tokens": 80,
        "output_tokens": 10,
        "completion_tokens": 10,
        "total_tokens": 110,
        "cache_read_tokens": 20,
        "cache_write_tokens": 0,
        "reasoning_tokens": 3,
        "api_calls": 1,
    }


@pytest.mark.asyncio
async def test_relay_async_attempt_is_counted_once(tmp_path, monkeypatch):
    from agent import relay_llm, relay_runtime
    from agent import tool_search_telemetry as telemetry

    monkeypatch.setattr(
        relay_runtime,
        "resolve_execution_context",
        lambda _session_id: (None, None, None),
    )

    async def provider(_request):
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=2)
        )

    path = tmp_path / "relay-async.jsonl"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-async",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="auxiliary",
            path=path,
        )
        await relay_llm.execute_async(
            {"model": "test/model"},
            provider,
            session_id="session-1",
            name="auxiliary",
            model_name="test/model",
            metadata={"api_mode": "chat_completions"},
        )
        assert telemetry.finish_current_turn({"completed": True})
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 1
    assert record["usage"]["prompt_tokens"] == 12


def test_relay_stream_attempt_uses_terminal_chunk_usage(tmp_path, monkeypatch):
    from agent import relay_llm, relay_runtime
    from agent import tool_search_telemetry as telemetry

    monkeypatch.setattr(
        relay_runtime,
        "resolve_execution_context",
        lambda _session_id: (None, None, None),
    )
    usage = SimpleNamespace(prompt_tokens=21, completion_tokens=4)
    path = tmp_path / "relay-stream.jsonl"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-stream",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="bedrock",
            path=path,
        )
        stream = relay_llm.stream(
            {"model": "test/model"},
            lambda _request: iter(
                [SimpleNamespace(delta="done", usage=usage)]
            ),
            session_id="session-1",
            name="bedrock",
            model_name="test/model",
            finalizer=lambda: SimpleNamespace(usage=usage),
            metadata={"api_mode": "chat_completions"},
        )
        assert len(list(stream)) == 1
        assert telemetry.finish_current_turn({"completed": True})
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 1
    assert record["usage"]["prompt_tokens"] == 21
    assert record["usage"]["output_tokens"] == 4


def test_relay_retry_counts_each_callback_attempt(tmp_path, monkeypatch):
    from agent import relay_llm, relay_runtime
    from agent import tool_search_telemetry as telemetry

    monkeypatch.setattr(
        relay_runtime,
        "resolve_execution_context",
        lambda _session_id: (None, None, None),
    )
    attempts = iter(
        [
            ConnectionError("transient"),
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=8, completion_tokens=1)
            ),
        ]
    )

    def provider(_request):
        outcome = next(attempts)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    path = tmp_path / "relay-retry.jsonl"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-retry",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="moa",
            path=path,
        )
        with pytest.raises(ConnectionError, match="transient"):
            relay_llm.execute(
                {},
                provider,
                session_id="session-1",
                name="openrouter",
                model_name="test/model",
            )
        relay_llm.execute(
            {},
            provider,
            session_id="session-1",
            name="openrouter",
            model_name="test/model",
        )
        assert telemetry.finish_current_turn({"completed": True})
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 2
    assert record["usage"]["prompt_tokens"] == 8


def test_relay_virtual_moa_facade_does_not_double_count_internal_attempt(
    tmp_path,
    monkeypatch,
):
    from agent import relay_llm, relay_runtime
    from agent import tool_search_telemetry as telemetry

    monkeypatch.setattr(
        relay_runtime,
        "resolve_execution_context",
        lambda _session_id: (None, None, None),
    )
    monkeypatch.setattr(relay_runtime, "active_turn", lambda *_args: None)
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=13, completion_tokens=2)
    )

    def virtual_facade(_request):
        relay_llm.execute_current(
            {},
            lambda _inner_request: response,
            name="openrouter",
            model_name="advisor/test",
            metadata={"api_mode": "chat_completions"},
        )
        return iter(())

    path = tmp_path / "relay-moa-virtual.jsonl"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-moa-virtual",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="moa",
            provider="moa",
            path=path,
        )
        assert list(
            relay_llm.stream(
                {},
                virtual_facade,
                session_id="session-1",
                name="moa",
                model_name="moa",
                finalizer=lambda: SimpleNamespace(usage=None),
            )
        ) == []
        assert telemetry.finish_current_turn({"completed": True})
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 1
    assert record["usage"]["prompt_tokens"] == 13


def test_relay_middleware_short_circuit_counts_no_provider_attempt(
    tmp_path,
    monkeypatch,
):
    from agent import relay_llm, relay_runtime
    from agent import tool_search_telemetry as telemetry

    class _Llm:
        @staticmethod
        async def execute(_name, _request, _invoke, **_kwargs):
            return {"content": "cached"}

    class _Runtime:
        relay = SimpleNamespace(
            LLMRequest=lambda _headers, content: SimpleNamespace(content=content),
            llm=_Llm(),
            codecs=None,
        )

        @staticmethod
        def managed_execution_enabled():
            return True

        @staticmethod
        async def run_in_session_async(_session, operation, *args, **kwargs):
            return await operation(*args, **kwargs)

    runtime = _Runtime()
    session = SimpleNamespace(session_id="session-1")
    monkeypatch.setattr(
        relay_runtime,
        "resolve_execution_context",
        lambda _session_id: (runtime, session, None),
    )
    monkeypatch.setattr(relay_runtime, "active_turn", lambda *_args: None)
    provider = MagicMock(side_effect=AssertionError("provider must not run"))
    path = tmp_path / "relay-short-circuit.jsonl"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-short-circuit",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="test",
            path=path,
        )
        result = relay_llm.execute(
            {},
            provider,
            session_id="session-1",
            name="test",
            model_name="test/model",
        )
        assert result.content == "cached"
        assert telemetry.finish_current_turn({"completed": True})
    finally:
        telemetry.end_turn_scope(token)

    provider.assert_not_called()
    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 0
