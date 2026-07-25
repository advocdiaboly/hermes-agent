"""Focused contract tests for native tool-search per-turn telemetry."""

from __future__ import annotations

import asyncio
import builtins
import json
import os
import stat
import threading
from copy import deepcopy
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
    completed: bool = True,
    interrupted: bool = False,
    failed: bool = False,
    partial: bool = False,
    exit_reason: str = "completed",
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
                input_tokens=prompt - 1 if prompt else 0,
                output_tokens=2,
                completion_tokens=2,
                total_tokens=prompt + 2,
                cache_read_tokens=1 if prompt else 0,
                cache_write_tokens=0,
                reasoning_tokens=1,
            )
        assert telemetry.finish_current_turn(
            {
                "api_calls": len(prompt_tokens),
                "completed": completed,
                "interrupted": interrupted,
                "failed": failed,
                "partial": partial,
                "turn_exit_reason": exit_reason,
            }
        )
    finally:
        telemetry.end_turn_scope(token)


def test_two_consecutive_turns_have_independent_usage_deltas(tmp_path):
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
        prompt_tokens=[100, 130],
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
    assert records[0]["usage"]["prompt_tokens"] == 230
    assert records[1]["usage"]["prompt_tokens"] == 40
    assert records[0]["usage"]["api_calls"] == 2
    assert records[1]["usage"]["api_calls"] == 1


def test_first_prompt_tokens_are_first_positive_api_usage(tmp_path):
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "turns.jsonl"
    _finish_turn(
        telemetry,
        path=path,
        turn_id="turn-first-positive",
        prompt_tokens=[0, 321, 654],
        assembly=telemetry.ToolSearchAssemblySnapshot.control(
            deferred_count=0,
            deferred_schema_tokens=0,
            activation_threshold_tokens=20_000,
        ),
    )

    [record] = telemetry.read_records(path)
    assert record["usage"]["first_prompt_tokens"] == 321
    assert record["usage"]["prompt_tokens"] == 975


def test_active_and_control_assemblies_are_distinguished_without_schema_leak():
    from agent import tool_search_telemetry as telemetry

    active = telemetry.ToolSearchAssemblySnapshot.active(
        deferred_count=12,
        deferred_schema_tokens=7_000,
        activation_threshold_tokens=5_000,
    )
    control = telemetry.ToolSearchAssemblySnapshot.control(
        deferred_count=3,
        deferred_schema_tokens=900,
        activation_threshold_tokens=5_000,
    )

    active_tools = telemetry.attach_assembly_snapshot([_tool("tool_search")], active)
    control_tools = telemetry.attach_assembly_snapshot([_tool("terminal")], control)

    assert telemetry.extract_assembly_snapshot(active_tools) == active
    assert telemetry.extract_assembly_snapshot(control_tools) == control
    assert active_tools == [_tool("tool_search")]
    assert control_tools == [_tool("terminal")]
    assert "_tool_search_assembly" not in json.dumps(active_tools)


def test_concurrent_profiles_keep_their_own_assembly_snapshot(tmp_path):
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "turns.jsonl"
    barrier = threading.Barrier(2)

    def worker(turn_id: str, assembly) -> None:
        token = telemetry.begin_turn_scope()
        try:
            tools = telemetry.attach_assembly_snapshot([_tool("terminal")], assembly)
            telemetry.start_turn(
                turn_id=turn_id,
                session_id=f"session-{turn_id}",
                user_message=f"message-{turn_id}",
                tools=tools,
                model=f"model-{turn_id}",
                provider="test",
                path=path,
            )
            barrier.wait(timeout=5)
            telemetry.note_api_usage(
                prompt_tokens=10 if turn_id == "active" else 20,
                input_tokens=9 if turn_id == "active" else 19,
                output_tokens=1,
                completion_tokens=1,
                total_tokens=11 if turn_id == "active" else 21,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            )
            assert telemetry.finish_current_turn(
                {
                    "api_calls": 1,
                    "completed": True,
                    "interrupted": False,
                    "failed": False,
                    "partial": False,
                    "turn_exit_reason": "completed",
                }
            )
        finally:
            telemetry.end_turn_scope(token)

    active = telemetry.ToolSearchAssemblySnapshot.active(
        deferred_count=20,
        deferred_schema_tokens=8_000,
        activation_threshold_tokens=4_000,
    )
    control = telemetry.ToolSearchAssemblySnapshot.control(
        deferred_count=2,
        deferred_schema_tokens=500,
        activation_threshold_tokens=4_000,
    )
    threads = [
        threading.Thread(target=worker, args=("active", active)),
        threading.Thread(target=worker, args=("control", control)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    by_turn = {record["turn_id"]: record for record in telemetry.read_records(path)}
    assert by_turn["active"]["tool_search"]["state"] == "active"
    assert by_turn["active"]["tool_search"]["deferred_count"] == 20
    assert by_turn["control"]["tool_search"]["state"] == "control"
    assert by_turn["control"]["tool_search"]["deferred_count"] == 2


def test_malformed_jsonl_is_ignored_and_duplicate_events_are_not_counted(tmp_path):
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "turns.jsonl"
    assembly = telemetry.ToolSearchAssemblySnapshot.active(
        deferred_count=5,
        deferred_schema_tokens=1_000,
        activation_threshold_tokens=500,
    )
    _finish_turn(
        telemetry,
        path=path,
        turn_id="turn-summary",
        prompt_tokens=[50],
        assembly=assembly,
    )
    valid_line = path.read_text(encoding="utf-8")
    with path.open("a", encoding="utf-8") as stream:
        stream.write("{truncated\n")
        stream.write(valid_line)

    summary = telemetry.summarize(path)
    assert summary["turns"] == 1
    assert summary["api_calls"] == 1
    assert summary["prompt_tokens"] == 50
    assert summary["assembly_states"] == {"active": 1}


def test_file_contains_no_raw_text_or_secret_and_is_private(tmp_path):
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "private" / "turns.jsonl"
    raw_text = "Please inspect confidential payload"
    sentinel_secret = "sk-sentinel-secret-DO-NOT-WRITE"
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
        user_message=f"{raw_text} {sentinel_secret}",
        exit_reason=f"provider_error({sentinel_secret})",
        failed=True,
        completed=False,
    )

    payload = path.read_text(encoding="utf-8")
    assert raw_text not in payload
    assert sentinel_secret not in payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    [record] = telemetry.read_records(path)
    assert len(record["user_hash"]) == 64
    assert record["outcome"]["exit_reason"] == "provider_error"


def test_io_failure_never_escapes_turn_finalization(tmp_path):
    from agent import tool_search_telemetry as telemetry

    unwritable_target = tmp_path / "directory-not-file"
    unwritable_target.mkdir()
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-io-failure",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="test",
            path=unwritable_target,
        )
        assert (
            telemetry.finish_current_turn(
                {
                    "api_calls": 0,
                    "completed": False,
                    "interrupted": True,
                    "failed": False,
                    "partial": True,
                    "turn_exit_reason": "interrupted_by_user",
                }
            )
            is False
        )
        # A repeated finalization attempt must not emit a second event.
        assert telemetry.finish_current_turn({}) is False
    finally:
        telemetry.end_turn_scope(token)


def test_symlink_telemetry_path_fails_closed_without_touching_target(tmp_path):
    import pytest

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
        assert telemetry.finish_current_turn(
            {
                "completed": True,
                "partial": False,
                "failed": False,
                "interrupted": False,
                "turn_exit_reason": "completed",
            }
        ) is False
    finally:
        telemetry.end_turn_scope(token)

    assert target.read_text(encoding="utf-8") == "ORIGINAL\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_short_writes_are_retried_to_complete_one_jsonl_line(
    tmp_path,
    monkeypatch,
):
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "short-write.jsonl"
    real_write = os.write
    calls = []

    def short_write(fd, payload):
        calls.append(len(payload))
        chunk_size = max(1, len(payload) // 2)
        return real_write(fd, payload[:chunk_size])

    monkeypatch.setattr(telemetry.os, "write", short_write)
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-short-write",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="test",
            path=path,
        )
        assert telemetry.finish_current_turn(
            {
                "completed": True,
                "partial": False,
                "failed": False,
                "interrupted": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    assert len(calls) > 1
    [record] = telemetry.read_records(path)
    assert record["turn_id"] == "turn-short-write"


def test_record_construction_failure_is_swallowed_and_registry_is_cleaned(
    tmp_path,
    monkeypatch,
):
    from agent import tool_search_telemetry as telemetry

    turn_id = "turn-build-record-failure"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id=turn_id,
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="test",
            path=tmp_path / "turns.jsonl",
        )
        monkeypatch.setattr(
            telemetry,
            "_build_record",
            MagicMock(side_effect=RuntimeError("record construction failed")),
        )
        assert telemetry.finish_current_turn({"completed": True}) is False
        assert turn_id not in telemetry._TRACKERS_BY_TURN_ID
        assert telemetry.finish_current_turn({"completed": True}) is False
    finally:
        telemetry.end_turn_scope(token)


def test_wrapper_swallows_unexpected_finalization_exception(monkeypatch):
    from agent import conversation_loop
    from agent import tool_search_telemetry as telemetry

    monkeypatch.setattr(
        conversation_loop,
        "_run_conversation_impl",
        lambda *_args, **_kwargs: {"completed": True},
    )
    monkeypatch.setattr(
        telemetry,
        "finish_current_turn",
        MagicMock(side_effect=RuntimeError("telemetry bug")),
    )

    assert conversation_loop.run_conversation(object(), "hello") == {
        "completed": True
    }


def test_bridge_usage_is_scoped_by_authoritative_turn_id(tmp_path):
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
        assert telemetry.finish_current_turn(
            {
                "api_calls": 1,
                "completed": True,
                "interrupted": False,
                "failed": False,
                "partial": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["tool_search"]["calls"] == {
        "tool_search": 1,
        "tool_describe": 1,
        "tool_call": 1,
    }


def test_long_authoritative_turn_id_still_routes_usage_to_its_tracker(tmp_path):
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "long-turn-id.jsonl"
    turn_id = "session:task:" + ("x" * 400)
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
        telemetry.note_api_usage(
            prompt_tokens=12,
            input_tokens=10,
            output_tokens=2,
            completion_tokens=2,
            total_tokens=14,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
            turn_id=turn_id,
        )
        assert telemetry.finish_current_turn(
            {
                "api_calls": 1,
                "completed": True,
                "interrupted": False,
                "failed": False,
                "partial": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["usage"]["prompt_tokens"] == 12
    assert record["turn_id"] == turn_id


def test_tracker_exists_when_prologue_fails_immediately_after_turn_id(
    tmp_path,
    monkeypatch,
):
    import pytest

    from agent import tool_search_telemetry as telemetry
    from tests.agent.test_turn_context import _FakeAgent, _build

    path = tmp_path / "prologue-failure.jsonl"
    monkeypatch.setenv("HERMES_TOOL_SEARCH_TELEMETRY_PATH", str(path))
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.note_turn_start",
        MagicMock(side_effect=RuntimeError("injected prologue failure")),
    )
    token = telemetry.begin_turn_scope()
    try:
        with pytest.raises(RuntimeError, match="injected prologue failure"):
            _build(_FakeAgent(), user_message="private input")
        assert telemetry.finish_current_turn(
            exception=RuntimeError("injected prologue failure")
        )
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["outcome"]["failed"] is True
    assert record["outcome"]["exit_reason"] == "exception:RuntimeError"


def test_real_partial_early_exit_shape_is_normalized_without_raw_error(tmp_path):
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "partial-shape.jsonl"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-partial-shape",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="test",
            path=path,
        )
        telemetry.note_api_call()
        assert telemetry.finish_current_turn(
            {
                "final_response": "Response truncated",
                "messages": [],
                "api_calls": 99,
                "completed": False,
                "partial": True,
                "error": "SECRET provider detail that must not persist",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["outcome"] == {
        "completed": False,
        "failed": True,
        "interrupted": False,
        "partial": True,
        "exit_reason": "partial_error",
    }
    assert record["usage"]["api_calls"] == 1
    assert "SECRET" not in path.read_text(encoding="utf-8")


def test_exception_after_usage_keeps_tracker_api_call_count(tmp_path):
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "exception-after-usage.jsonl"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-exception-after-usage",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="test",
            path=path,
        )
        telemetry.note_api_call()
        telemetry.note_api_usage(
            prompt_tokens=41,
            input_tokens=40,
            output_tokens=1,
            completion_tokens=1,
            total_tokens=42,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
        )
        assert telemetry.finish_current_turn(exception=RuntimeError("boom"))
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 1
    assert record["usage"]["prompt_tokens"] == 41


def test_codex_interrupted_turn_reports_real_outcome_usage_and_call_count(
    tmp_path,
    monkeypatch,
):
    from agent import tool_search_telemetry as telemetry
    from agent.codex_runtime import run_codex_app_server_turn
    from tests.agent.test_codex_app_server_persist import _make_agent, _make_turn

    path = tmp_path / "codex-interrupted.jsonl"
    turn = _make_turn()
    turn.interrupted = True
    turn.error = None
    turn.final_text = "partial"
    turn.token_usage_last = {
        "inputTokens": 50,
        "cachedInputTokens": 10,
        "outputTokens": 5,
        "reasoningOutputTokens": 2,
        "totalTokens": 67,
    }
    turn.model_context_window = 100_000
    agent = _make_agent(session_db=None)
    agent._codex_session.run_turn.return_value = turn
    agent.model = "codex/test"
    agent.provider = "openai-codex"
    agent.base_url = "https://example.invalid"
    agent.api_key = "not-persisted"
    agent.context_compressor = MagicMock()
    for name in (
        "session_prompt_tokens",
        "session_completion_tokens",
        "session_total_tokens",
        "session_api_calls",
        "session_input_tokens",
        "session_output_tokens",
        "session_cache_read_tokens",
        "session_cache_write_tokens",
        "session_reasoning_tokens",
    ):
        setattr(agent, name, 0)
    agent.session_estimated_cost_usd = 0.0
    monkeypatch.setattr(
        "agent.usage_pricing.estimate_usage_cost",
        lambda *_args, **_kwargs: SimpleNamespace(
            amount_usd=None,
            status="included",
            source="test",
        ),
    )

    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-codex-interrupted",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model=agent.model,
            provider=agent.provider,
            path=path,
        )
        result = run_codex_app_server_turn(
            agent,
            user_message="hello",
            original_user_message="hello",
            messages=[{"role": "user", "content": "hello"}],
            effective_task_id="task-1",
        )
        assert result["interrupted"] is True
        assert result["failed"] is False
        assert result["turn_exit_reason"] == "interrupted"
        assert telemetry.finish_current_turn(result)
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 1
    assert record["usage"]["first_prompt_tokens"] == 60
    assert record["usage"]["prompt_tokens"] == 60
    assert record["outcome"]["interrupted"] is True
    assert record["outcome"]["exit_reason"] == "interrupted"


def test_moa_reference_records_its_own_prompt_before_aggregate(
    tmp_path,
    monkeypatch,
):
    from agent import tool_search_telemetry as telemetry
    from agent import moa_loop

    path = tmp_path / "moa-reference.jsonl"
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=123,
            completion_tokens=7,
            total_tokens=130,
        )
    )
    def fake_call_llm(**_kwargs):
        from agent.auxiliary_client import _call_provider_create
        return _call_provider_create(lambda **_call_kwargs: response)

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda _slot: {
            "provider": "openai",
            "api_mode": "chat_completions",
            "model": "advisor/test",
            "base_url": "https://example.invalid",
            "api_key": "not-persisted",
        },
    )
    monkeypatch.setattr(moa_loop, "_extract_text", lambda _response: "advice")
    monkeypatch.setattr(
        "agent.usage_pricing.estimate_usage_cost",
        lambda *_args, **_kwargs: SimpleNamespace(
            amount_usd=None,
            status="unknown",
            source="test",
        ),
    )

    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-moa-reference",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="moa",
            provider="moa",
            path=path,
        )
        label, text, _accounting = moa_loop._run_reference(
            {"provider": "openai", "model": "advisor/test"},
            [{"role": "user", "content": "hello"}],
        )
        assert label
        assert text == "advice"
        assert telemetry.finish_current_turn(
            {
                "completed": True,
                "partial": False,
                "failed": False,
                "interrupted": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 1
    assert record["usage"]["first_prompt_tokens"] == 123
    assert record["usage"]["prompt_tokens"] == 123


def test_model_tools_attaches_exact_active_and_control_assembly(monkeypatch):
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

    config = tool_search.ToolSearchConfig.from_raw(
        {"enabled": "auto", "threshold_pct": 10}
    )
    monkeypatch.setattr(tool_search, "load_config", lambda: config)

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
    assert active.state == "active"
    assert active.deferred_count == 9
    assert active.deferred_schema_tokens == 6_000
    assert active.activation_threshold_tokens == 5_000

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
    assert control.state == "control"
    assert control.deferred_count == 2


def test_control_snapshot_keeps_configured_threshold_with_no_deferred_tools(
    monkeypatch,
):
    import model_tools
    from agent import tool_search_telemetry as telemetry
    from tools import tool_search

    monkeypatch.setattr(model_tools, "validate_toolset", lambda _name: True)
    monkeypatch.setattr(model_tools, "resolve_toolset", lambda _name: {"terminal"})
    monkeypatch.setattr(
        model_tools.registry,
        "get_definitions",
        lambda *_args, **_kwargs: [_tool("terminal")],
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
            tool_defs=[_tool("terminal")],
            activated=False,
            deferred_count=0,
            deferred_tokens=0,
            threshold_tokens=0,
        ),
    )

    definitions = model_tools._compute_tool_definitions(
        enabled_toolsets=["terminal"],
        quiet_mode=True,
    )

    snapshot = telemetry.extract_assembly_snapshot(definitions)
    assert snapshot.state == "control"
    assert snapshot.activation_threshold_tokens == 5_000


def test_empty_quiet_cache_preserves_snapshot_on_miss_hit_and_deepcopy(
    monkeypatch,
):
    import model_tools
    from agent import tool_search_telemetry as telemetry

    snapshot = telemetry.ToolSearchAssemblySnapshot.control(
        deferred_count=0,
        deferred_schema_tokens=0,
        activation_threshold_tokens=5_000,
    )
    computed = telemetry.attach_assembly_snapshot([], snapshot)
    compute = MagicMock(return_value=computed)
    monkeypatch.setattr(model_tools, "_compute_tool_definitions", compute)
    model_tools._clear_tool_defs_cache()
    try:
        miss = model_tools.get_tool_definitions(quiet_mode=True)
        hit = model_tools.get_tool_definitions(quiet_mode=True)
        copied = deepcopy(hit)
    finally:
        model_tools._clear_tool_defs_cache()

    assert compute.call_count == 1
    assert telemetry.extract_assembly_snapshot(miss) == snapshot
    assert telemetry.extract_assembly_snapshot(hit) == snapshot
    assert telemetry.extract_assembly_snapshot(copied) == snapshot


def test_model_tools_bridge_dispatch_records_matching_turn(monkeypatch):
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


def test_turn_context_starts_tracker_with_generated_turn_id(tmp_path, monkeypatch):
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
        assert telemetry.finish_current_turn(
            {
                "api_calls": 0,
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


def test_run_conversation_wrapper_finalizes_partial_early_return(
    tmp_path,
    monkeypatch,
):
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
        telemetry.note_api_usage(
            prompt_tokens=77,
            input_tokens=70,
            output_tokens=7,
            completion_tokens=7,
            total_tokens=84,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
        )
        return {
            "api_calls": 1,
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
    assert record["outcome"] == {
        "completed": False,
        "failed": False,
        "interrupted": False,
        "partial": True,
        "exit_reason": "truncated_response",
    }
    assert record["usage"]["prompt_tokens"] == 77


def test_run_conversation_wrapper_records_failure_then_reraises(
    tmp_path,
    monkeypatch,
):
    import pytest

    from agent import conversation_loop
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "wrapped-failure.jsonl"

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
        raise RuntimeError("sentinel secret must not be persisted")

    monkeypatch.setattr(conversation_loop, "_run_conversation_impl", fake_impl)
    with pytest.raises(RuntimeError, match="sentinel secret"):
        conversation_loop.run_conversation(object(), "raw prompt")

    [record] = telemetry.read_records(path)
    assert record["outcome"]["failed"] is True
    assert record["outcome"]["partial"] is True
    assert record["outcome"]["exit_reason"] == "exception:RuntimeError"
    assert "sentinel secret" not in path.read_text(encoding="utf-8")


def test_llm_execution_middleware_short_circuit_counts_no_provider_attempt(
    tmp_path,
    monkeypatch,
):
    from agent import tool_search_telemetry as telemetry
    from tests.run_agent.test_run_agent_codex_responses import (
        _build_agent,
        _codex_message_response,
    )

    path = tmp_path / "middleware-short-circuit.jsonl"
    monkeypatch.setenv("HERMES_TOOL_SEARCH_TELEMETRY_PATH", str(path))
    agent = _build_agent(monkeypatch)
    agent._disable_streaming = True
    terminal_call = MagicMock(side_effect=AssertionError("provider must not run"))
    monkeypatch.setattr(agent, "_interruptible_api_call", terminal_call)
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        lambda _request, _next_call, **_context: _codex_message_response("cached"),
    )

    result = agent.run_conversation("use middleware response")

    assert result["completed"] is True
    terminal_call.assert_not_called()
    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 0


def test_nonstreaming_openai_and_bedrock_terminal_calls_count_attempts(
    tmp_path,
    monkeypatch,
):
    from agent import bedrock_adapter
    from agent import chat_completion_helpers as helpers
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "nonstream-attempts.jsonl"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-nonstream-attempts",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="test",
            path=path,
        )

        openai_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: SimpleNamespace(choices=[])
                )
            )
        )
        openai_agent = SimpleNamespace(
            api_mode="chat_completions",
            provider="openrouter",
        )
        helpers._dispatch_nonstreaming_api_request(
            openai_agent,
            {},
            make_client=lambda *_args, **_kwargs: openai_client,
        )

        bedrock_client = SimpleNamespace(
            converse=lambda **_kwargs: {"output": {"message": {"content": []}}}
        )
        monkeypatch.setattr(
            bedrock_adapter,
            "_get_bedrock_runtime_client",
            lambda _region: bedrock_client,
        )
        monkeypatch.setattr(
            bedrock_adapter,
            "normalize_converse_response",
            lambda response: response,
        )
        bedrock_agent = SimpleNamespace(
            api_mode="bedrock_converse",
            provider="bedrock",
        )
        helpers._dispatch_nonstreaming_api_request(
            bedrock_agent,
            {"__bedrock_region__": "us-east-1", "__bedrock_converse__": True},
            make_client=lambda *_args, **_kwargs: None,
        )

        moa_facade = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: SimpleNamespace(choices=[])
                )
            )
        )
        moa_agent = SimpleNamespace(
            api_mode="chat_completions",
            provider="moa",
            client=moa_facade,
        )
        helpers._dispatch_nonstreaming_api_request(
            moa_agent,
            {},
            make_client=lambda *_args, **_kwargs: None,
        )

        assert telemetry.finish_current_turn(
            {
                "completed": True,
                "partial": False,
                "failed": False,
                "interrupted": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    # The MoA facade is virtual; only its internal reference/aggregator SDK
    # attempts count. The two direct calls above are OpenAI + Bedrock.
    assert record["usage"]["api_calls"] == 2


def test_bedrock_streaming_iam_fallback_counts_both_provider_attempts(
    tmp_path,
    monkeypatch,
):
    from agent import bedrock_adapter
    from agent import chat_completion_helpers as helpers
    from agent import tool_search_telemetry as telemetry
    from tests.agent.test_bedrock_interrupt_post_worker import _FakeAgent

    path = tmp_path / "bedrock-stream-fallback.jsonl"
    response = SimpleNamespace(choices=[], usage=None, stop_reason="end_turn")

    def deny_stream(**_kwargs):
        raise RuntimeError("stream action denied")

    client = SimpleNamespace(
        converse_stream=deny_stream,
        converse=lambda **_kwargs: response,
    )
    monkeypatch.setattr(
        bedrock_adapter,
        "_get_bedrock_runtime_client",
        lambda _region: client,
    )
    monkeypatch.setattr(
        bedrock_adapter,
        "is_streaming_access_denied_error",
        lambda _exc: True,
    )
    monkeypatch.setattr(
        bedrock_adapter,
        "normalize_converse_response",
        lambda value: value,
    )

    agent = _FakeAgent()
    agent._current_turn_id = "turn-bedrock-stream-fallback"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-bedrock-stream-fallback",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model=agent.model,
            provider=agent.provider,
            path=path,
        )
        result = helpers.interruptible_streaming_api_call(
            agent,
            {
                "modelId": agent.model,
                "messages": [],
                "__bedrock_region__": "us-east-1",
                "__bedrock_converse__": True,
            },
        )
        assert result is response
        assert telemetry.finish_current_turn(
            {
                "completed": True,
                "partial": False,
                "failed": False,
                "interrupted": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 2


def test_streaming_transport_retry_counts_each_actual_provider_attempt(
    tmp_path,
    monkeypatch,
):
    import httpx

    from agent import tool_search_telemetry as telemetry
    from run_agent import AIAgent

    path = tmp_path / "stream-retries.jsonl"
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
    client = MagicMock()
    client.chat.completions.create.side_effect = httpx.ConnectError("dropped")

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent._current_turn_id = "turn-stream-retries"
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **_kwargs: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *_a, **_k: None)
    monkeypatch.setattr(agent, "_replace_primary_openai_client", lambda **_kwargs: None)

    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-stream-retries",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model=agent.model,
            provider=agent.provider,
            path=path,
        )
        with __import__("pytest").raises(httpx.ConnectError, match="dropped"):
            agent._interruptible_streaming_api_call({})
        assert telemetry.finish_current_turn(exception=httpx.ConnectError("dropped"))
    finally:
        telemetry.end_turn_scope(token)

    assert client.chat.completions.create.call_count == 2
    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 2


def test_streaming_virtual_moa_facade_does_not_double_count_internal_calls(
    tmp_path,
):
    from agent import chat_completion_helpers as helpers
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "moa-virtual-stream.jsonl"
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-moa-virtual-stream",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="moa",
            provider="moa",
            path=path,
        )
        helpers._note_provider_attempt(
            SimpleNamespace(
                provider="moa",
                _current_turn_id="turn-moa-virtual-stream",
            )
        )
        telemetry.note_api_call(turn_id="turn-moa-virtual-stream")
        assert telemetry.finish_current_turn(
            {
                "completed": True,
                "partial": False,
                "failed": False,
                "interrupted": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    # One internal aggregator SDK call, zero for the outer virtual facade.
    assert record["usage"]["api_calls"] == 1


def test_virtual_moa_worker_binds_turn_for_internal_provider_attempts(
    tmp_path,
):
    from agent import auxiliary_client
    from agent import chat_completion_helpers as helpers
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "moa-worker-context.jsonl"

    class _Facade:
        @staticmethod
        def create(**_kwargs):
            return auxiliary_client._call_provider_create(
                lambda **_call_kwargs: SimpleNamespace(choices=[])
            )

    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="moa",
        _current_turn_id="wrong-shared-turn",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=_Facade()),
        ),
    )
    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-moa-worker-context",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="moa",
            provider="moa",
            path=path,
        )
        worker = threading.Thread(
            target=lambda: helpers._dispatch_nonstreaming_api_request(
                agent,
                {},
                make_client=lambda *_args, **_kwargs: None,
                telemetry_turn_id="turn-moa-worker-context",
            )
        )
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert telemetry.finish_current_turn(
            {
                "completed": True,
                "partial": False,
                "failed": False,
                "interrupted": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 1


@pytest.mark.parametrize("provider_route", ["ordinary", "moa", "codex"])
def test_overlapping_turns_on_shared_agent_do_not_cross_attribute_provider_attempts(
    tmp_path,
    provider_route,
):
    from agent import auxiliary_client
    from agent import chat_completion_helpers as helpers
    from agent import codex_runtime
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / f"shared-agent-{provider_route}.jsonl"
    start_barrier = threading.Barrier(2)
    finish_barrier = threading.Barrier(2)
    errors = []

    if provider_route == "ordinary":
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: SimpleNamespace(choices=[])
                )
            )
        )
        agent = SimpleNamespace(
            api_mode="chat_completions",
            provider="openrouter",
        )

        def invoke():
            helpers._dispatch_nonstreaming_api_request(
                agent,
                {},
                make_client=lambda *_args, **_kwargs: client,
            )

    elif provider_route == "moa":
        class _Facade:
            @staticmethod
            def create(**_kwargs):
                return auxiliary_client._call_provider_create(
                    lambda **_call_kwargs: SimpleNamespace(choices=[])
                )

        agent = SimpleNamespace(
            api_mode="chat_completions",
            provider="moa",
            client=SimpleNamespace(
                chat=SimpleNamespace(completions=_Facade()),
            ),
        )

        def invoke():
            helpers._dispatch_nonstreaming_api_request(
                agent,
                {},
                make_client=lambda *_args, **_kwargs: None,
            )

    else:
        completed = SimpleNamespace(
            output=[],
            output_text="done",
            usage=None,
            status="completed",
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_kwargs: completed)
        )
        agent = SimpleNamespace(
            _ensure_primary_openai_client=lambda **_kwargs: client,
            _codex_streamed_text_parts=[],
            _fire_stream_delta=lambda _text: None,
            _fire_reasoning_delta=lambda _text: None,
            _fire_streamed_codex_commentary=lambda _text: None,
            _touch_activity=lambda _status: None,
            _interrupt_requested=False,
            _client_log_context=lambda: "test",
            interim_assistant_callback=None,
            show_commentary=True,
        )

        def invoke():
            codex_runtime.run_codex_stream(agent, {"model": "codex/test"})

    def run_turn(turn_id):
        token = telemetry.begin_turn_scope()
        try:
            telemetry.start_turn(
                turn_id=turn_id,
                session_id="shared-session",
                user_message=turn_id,
                tools=[],
                model="test/model",
                provider=provider_route,
                path=path,
            )
            agent._current_turn_id = turn_id
            start_barrier.wait(timeout=5)
            invoke()
            finish_barrier.wait(timeout=5)
            assert telemetry.finish_current_turn(
                {
                    "completed": True,
                    "partial": False,
                    "failed": False,
                    "interrupted": False,
                    "turn_exit_reason": "completed",
                }
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            telemetry.end_turn_scope(token)

    workers = [
        threading.Thread(target=run_turn, args=(f"turn-{suffix}",))
        for suffix in ("a", "b")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)

    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    records = sorted(telemetry.read_records(path), key=lambda record: record["turn_id"])
    assert [record["turn_id"] for record in records] == ["turn-a", "turn-b"]
    assert [record["usage"]["api_calls"] for record in records] == [1, 1]


def test_moa_auxiliary_transient_retry_counts_each_real_attempt(
    tmp_path,
    monkeypatch,
):
    from agent import auxiliary_client
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "moa-aux-retry.jsonl"
    complete = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="advisor"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=1,
            total_tokens=6,
        ),
    )
    client = MagicMock()
    client.base_url = "https://example.invalid/v1"
    client.chat.completions.create.side_effect = [
        ConnectionError("transient"),
        complete,
    ]
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_task_provider_model",
        lambda *_args, **_kwargs: (
            "openrouter",
            "advisor/test",
            "https://example.invalid/v1",
            "not-persisted",
            "chat_completions",
        ),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *_args, **_kwargs: (client, "advisor/test"),
    )
    monkeypatch.setattr(auxiliary_client, "_transient_retry_count", lambda: 1)
    monkeypatch.setattr(auxiliary_client.time, "sleep", lambda _seconds: None)

    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-moa-aux-retry",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="moa",
            provider="moa",
            path=path,
        )
        result = auxiliary_client.call_llm(
            task="moa_reference",
            provider="openrouter",
            model="advisor/test",
            base_url="https://example.invalid/v1",
            api_key="not-persisted",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert result is complete
        assert telemetry.finish_current_turn(
            {
                "completed": True,
                "partial": False,
                "failed": False,
                "interrupted": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    assert client.chat.completions.create.call_count == 2
    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 2


def test_codex_internal_retry_counts_each_responses_create_attempt(
    tmp_path,
    monkeypatch,
):
    import httpx

    from agent import codex_runtime
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / "codex-retries.jsonl"
    completed = SimpleNamespace(
        output=[],
        output_text="done",
        usage=None,
        status="completed",
    )
    client = MagicMock()
    client.responses.create.side_effect = [
        httpx.ConnectError("first"),
        completed,
    ]
    agent = SimpleNamespace(
        _ensure_primary_openai_client=lambda **_kwargs: client,
        _codex_streamed_text_parts=[],
        _fire_stream_delta=lambda _text: None,
        _fire_reasoning_delta=lambda _text: None,
        _fire_streamed_codex_commentary=lambda _text: None,
        _touch_activity=lambda _status: None,
        _interrupt_requested=False,
        _client_log_context=lambda: "test",
        interim_assistant_callback=None,
        show_commentary=True,
        _current_turn_id="turn-codex-retries",
    )

    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-codex-retries",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="codex/test",
            provider="openai-codex",
            path=path,
        )
        assert codex_runtime.run_codex_stream(agent, {"model": "codex/test"}) is completed
        assert telemetry.finish_current_turn(
            {
                "completed": True,
                "partial": False,
                "failed": False,
                "interrupted": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    assert client.responses.create.call_count == 2
    [record] = telemetry.read_records(path)
    assert record["usage"]["api_calls"] == 2


def test_authoritative_turn_id_bounds_immediately_over_limit_and_emits_record(
    tmp_path,
    monkeypatch,
):
    from agent import tool_search_telemetry as telemetry
    from agent.turn_context import (
        TURN_ID_COMPONENT_MAX_CHARS,
        TURN_ID_MAX_CHARS,
    )
    from tests.agent.test_turn_context import _FakeAgent, _build

    path = tmp_path / "bounded-turn-ids.jsonl"
    monkeypatch.setenv("HERMES_TOOL_SEARCH_TELEMETRY_PATH", str(path))
    exact_task_id = "t" * TURN_ID_COMPONENT_MAX_CHARS
    over_task_id = exact_task_id + "x"
    generated_ids = []

    for task_id in (exact_task_id, over_task_id):
        agent = _FakeAgent()
        token = telemetry.begin_turn_scope()
        try:
            ctx = _build(agent, task_id=task_id)
            generated_ids.append(ctx.turn_id)
            assert ctx.effective_task_id == task_id
            assert agent._current_task_id == task_id
            assert ctx.turn_id.isascii()
            assert len(ctx.turn_id) <= TURN_ID_MAX_CHARS
            assert telemetry.finish_current_turn(
                {
                    "completed": True,
                    "partial": False,
                    "failed": False,
                    "interrupted": False,
                    "turn_exit_reason": "completed",
                }
            )
        finally:
            telemetry.end_turn_scope(token)

    assert f":{exact_task_id}:" in generated_ids[0]
    assert over_task_id not in generated_ids[1]
    assert generated_ids[0] != generated_ids[1]
    records = telemetry.read_records(path)
    assert [record["turn_id"] for record in records] == generated_ids
    assert all(len(line) <= telemetry._MAX_JSONL_RECORD_BYTES for line in path.read_bytes().splitlines(True))


def test_authoritative_turn_id_uses_full_nonce_entropy_for_registry_and_event(
    tmp_path,
):
    from agent import tool_search_telemetry as telemetry
    from agent.turn_context import (
        TURN_ID_MAX_CHARS,
        build_authoritative_turn_id,
    )

    path = tmp_path / "nonce-distinct.jsonl"
    first_id = build_authoritative_turn_id(
        "session",
        "task",
        nonce="deadbeef00000000",
    )
    second_id = build_authoritative_turn_id(
        "session",
        "task",
        nonce="deadbeefffffffff",
    )

    assert first_id != second_id
    assert len(first_id) <= TURN_ID_MAX_CHARS
    assert len(second_id) <= TURN_ID_MAX_CHARS

    first_scope = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id=first_id,
            session_id="session",
            user_message="first",
            tools=[],
            model="test/model",
            provider="test",
            path=path,
        )
        second_scope = telemetry.begin_turn_scope()
        try:
            telemetry.start_turn(
                turn_id=second_id,
                session_id="session",
                user_message="second",
                tools=[],
                model="test/model",
                provider="test",
                path=path,
            )
            assert first_id in telemetry._TRACKERS_BY_TURN_ID
            assert second_id in telemetry._TRACKERS_BY_TURN_ID
            assert telemetry.finish_current_turn(
                {
                    "completed": True,
                    "partial": False,
                    "failed": False,
                    "interrupted": False,
                    "turn_exit_reason": "completed",
                }
            )
        finally:
            telemetry.end_turn_scope(second_scope)
        assert telemetry.finish_current_turn(
            {
                "completed": True,
                "partial": False,
                "failed": False,
                "interrupted": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(first_scope)

    records = telemetry.read_records(path)
    assert {record["turn_id"] for record in records} == {first_id, second_id}
    assert len({record["event_id"] for record in records}) == 2
    assert all(
        len(line) <= telemetry._MAX_JSONL_RECORD_BYTES
        for line in path.read_bytes().splitlines(True)
    )


@__import__("pytest").mark.parametrize(
    "exception_factory",
    [KeyboardInterrupt, asyncio.CancelledError],
)
def test_wrapper_classifies_base_interruptions_and_reraises_original(
    tmp_path,
    monkeypatch,
    exception_factory,
):
    import pytest

    from agent import conversation_loop
    from agent import tool_search_telemetry as telemetry

    path = tmp_path / f"{exception_factory.__name__}.jsonl"
    raised = exception_factory()

    def fake_impl(agent, user_message, **_kwargs):
        telemetry.start_turn(
            turn_id=f"turn-{exception_factory.__name__}",
            session_id="session-1",
            user_message=user_message,
            tools=[],
            model="test/model",
            provider="test",
            path=path,
        )
        raise raised

    monkeypatch.setattr(conversation_loop, "_run_conversation_impl", fake_impl)
    with pytest.raises(exception_factory) as caught:
        conversation_loop.run_conversation(object(), "raw prompt")

    assert caught.value is raised
    [record] = telemetry.read_records(path)
    assert record["outcome"] == {
        "completed": False,
        "failed": False,
        "interrupted": True,
        "partial": True,
        "exit_reason": f"exception:{exception_factory.__name__}",
    }


def test_symlink_leaf_fails_closed_when_o_nofollow_is_unavailable(
    tmp_path,
    monkeypatch,
):
    from agent import tool_search_telemetry as telemetry

    target = tmp_path / "target.txt"
    target.write_text("original", encoding="utf-8")
    link = tmp_path / "turns.jsonl"
    link.symlink_to(target)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    token = telemetry.begin_turn_scope()
    try:
        telemetry.start_turn(
            turn_id="turn-portable-symlink",
            session_id="session-1",
            user_message="hello",
            tools=[],
            model="test/model",
            provider="test",
            path=link,
        )
        assert not telemetry.finish_current_turn(
            {
                "completed": True,
                "partial": False,
                "failed": False,
                "interrupted": False,
                "turn_exit_reason": "completed",
            }
        )
    finally:
        telemetry.end_turn_scope(token)

    assert target.read_text(encoding="utf-8") == "original"


def test_telemetry_setup_import_failure_degrades_to_uninstrumented_turn(
    monkeypatch,
):
    from agent import conversation_loop

    expected = {
        "completed": True,
        "partial": False,
        "failed": False,
        "interrupted": False,
    }
    monkeypatch.setattr(
        conversation_loop,
        "_run_conversation_impl",
        lambda *_args, **_kwargs: expected,
    )
    real_import = builtins.__import__

    def fail_telemetry_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "agent.tool_search_telemetry"
            and "begin_turn_scope" in fromlist
        ):
            raise ImportError("simulated partial upgrade")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_telemetry_import)

    assert conversation_loop.run_conversation(object(), "hello") is expected
