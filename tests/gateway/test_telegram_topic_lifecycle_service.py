import asyncio
import gc
import importlib
import inspect
import queue
import threading
import warnings
from contextvars import copy_context
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    clear_session_vars,
    set_session_vars,
)


def _service():
    try:
        return importlib.import_module("gateway.telegram_topic_lifecycle")
    except ModuleNotFoundError:
        pytest.fail("gateway.telegram_topic_lifecycle service is missing")


class FakeSessionDB:
    def __init__(self, *, binding=None, topic_mode=False):
        self.binding = binding
        self.topic_mode = topic_mode
        self.reads = []
        self.writes = []

    async def get_telegram_topic_binding(self, *, chat_id, thread_id):
        self.reads.append(("binding", chat_id, thread_id))
        return self.binding

    async def is_telegram_topic_mode_enabled(self, *, chat_id, user_id):
        self.reads.append(("topic_mode", chat_id, user_id))
        return self.topic_mode


class FakeAdapter:
    def __init__(self):
        self.dm_creates = []
        self.dm_renames = []
        self.group_creates = []
        self.group_renames = []
        self.operation_loops = []
        self._bot = SimpleNamespace(
            create_forum_topic=self._create_forum_topic,
            edit_forum_topic=self._edit_forum_topic,
        )

    async def _create_dm_topic(self, chat_id, name):
        self.operation_loops.append(asyncio.get_running_loop())
        self.dm_creates.append((chat_id, name))
        return 221

    async def rename_dm_topic(self, *, chat_id, thread_id, name):
        self.operation_loops.append(asyncio.get_running_loop())
        self.dm_renames.append((chat_id, thread_id, name))

    async def _create_forum_topic(self, *, chat_id, name):
        self.operation_loops.append(asyncio.get_running_loop())
        self.group_creates.append((chat_id, name))
        return SimpleNamespace(message_thread_id=331)

    async def _edit_forum_topic(self, *, chat_id, message_thread_id, name):
        self.operation_loops.append(asyncio.get_running_loop())
        self.group_renames.append((chat_id, message_thread_id, name))


def _runner(monkeypatch, loop, *, adapter=None, db=None, profile=None):
    import gateway.run as gateway_run

    runner = object.__new__(GatewayRunner)
    runner._gateway_loop = loop
    runner._session_db = db if db is not None else FakeSessionDB()
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter is not None else {}
    runner._profile_adapters = {}
    if profile is not None and adapter is not None:
        runner.adapters = {Platform.TELEGRAM: FakeAdapter()}
        runner._profile_adapters = {profile: {Platform.TELEGRAM: adapter}}
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    return runner


async def _call_in_worker(function, *, title=None):
    result_queue = queue.Queue(maxsize=1)
    worker_context = copy_context()

    def invoke():
        try:
            if title is None:
                result = worker_context.run(function)
            else:
                result = worker_context.run(function, title)
            result_queue.put((True, result))
        except BaseException as exc:
            result_queue.put((False, exc))

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    while result_queue.empty():
        # The restricted test sandbox does not reliably wake an asyncio
        # selector through its cross-thread self-pipe. This bounded tick keeps
        # the real gateway-loop callback moving without relying on asyncio's
        # default executor (whose shutdown has the same issue).
        await asyncio.sleep(0.001)
    succeeded, result = result_queue.get_nowait()
    if not succeeded:
        raise result
    return result


async def _call_with_context(function, *, title=None, **context):
    platform = context.pop("platform", "telegram")
    tokens = set_session_vars(platform=platform, **context)
    try:
        return await _call_in_worker(function, title=title)
    finally:
        clear_session_vars(tokens)


def _assert_error(result, code):
    assert result["success"] is False
    assert result["error"]["code"] == code


def test_public_service_contract_exposes_no_routing_overrides():
    service = _service()

    assert list(inspect.signature(service.get_current_telegram_topic_status).parameters) == []
    assert list(inspect.signature(service.create_current_telegram_topic).parameters) == ["title"]
    assert list(inspect.signature(service.rename_current_telegram_topic).parameters) == ["title"]


@pytest.mark.asyncio
async def test_status_never_uses_process_global_session_identity(monkeypatch):
    service = _service()
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "stale-chat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "dm")
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "stale-thread")
    monkeypatch.setenv("HERMES_SESSION_PROFILE", "stale-profile")

    result = await _call_in_worker(service.get_current_telegram_topic_status)

    _assert_error(result, "not_telegram_context")


@pytest.mark.asyncio
async def test_status_uses_only_context_identity_and_rejects_non_telegram(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    db = FakeSessionDB()
    _runner(monkeypatch, loop, adapter=FakeAdapter(), db=db, profile="ops")

    result = await _call_with_context(
        service.get_current_telegram_topic_status,
        chat_id="-10055",
        chat_type="supergroup",
        thread_id="19",
        user_id="77",
        message_id="88",
        session_key="agent:main:telegram:group:-10055:19",
        session_id="session-19",
        profile="ops",
    )

    assert result["success"] is True
    assert result["surface"] == "supergroup_forum_topic"
    assert result["context"] == {
        "platform": "telegram",
        "chat_id": "-10055",
        "chat_type": "supergroup",
        "thread_id": "19",
        "user_id": "77",
        "message_id": "88",
        "session_id": "session-19",
        "session_key": "agent:main:telegram:group:-10055:19",
        "profile": "ops",
    }
    assert result["routing"] == result["context"]
    assert db.reads == [("binding", "-10055", "19")]

    rejected = await _call_with_context(
        service.get_current_telegram_topic_status,
        platform="discord",
        chat_id="other",
    )
    _assert_error(rejected, "not_telegram_context")


@pytest.mark.asyncio
async def test_status_distinguishes_matching_and_mismatching_bindings(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    matching = FakeSessionDB(
        binding={
            "session_id": "session-19",
            "session_key": "agent:main:telegram:dm:55:19",
        },
        topic_mode=True,
    )
    _runner(monkeypatch, loop, adapter=FakeAdapter(), db=matching)

    matched = await _call_with_context(
        service.get_current_telegram_topic_status,
        chat_id="55",
        chat_type="dm",
        thread_id="19",
        user_id="77",
        session_key="agent:main:telegram:dm:55:19",
        session_id="session-19",
    )
    assert matched["surface"] == "private_topic"
    assert matched["binding_status"] == "matches_routing"
    assert matched["binding_matches_routing"] is True
    assert matched["binding_authority"] == "private_topic_binding"
    assert matched["binding_routing_effect"] == "native_private_topic_routing"
    assert matched["private_topic_mode_enabled"] is True

    matching.binding = {
        "session_id": "other-session",
        "session_key": "agent:main:telegram:dm:55:other",
    }
    mismatched = await _call_with_context(
        service.get_current_telegram_topic_status,
        chat_id="55",
        chat_type="dm",
        thread_id="19",
        user_id="77",
        session_key="agent:main:telegram:dm:55:19",
        session_id="session-19",
    )
    assert mismatched["binding_status"] == "mismatch"
    assert mismatched["binding_matches_routing"] is False


@pytest.mark.asyncio
async def test_supergroup_binding_is_metadata_only_and_topic_mode_is_not_applicable(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    db = FakeSessionDB(
        binding={"session_id": "source-session", "session_key": "source-key"},
        topic_mode=True,
    )
    _runner(monkeypatch, loop, adapter=FakeAdapter(), db=db)

    result = await _call_with_context(
        service.get_current_telegram_topic_status,
        chat_id="-10055",
        chat_type="group",
        thread_id="19",
        user_id="77",
        session_key="current-key",
        session_id="current-session",
    )

    assert result["surface"] == "supergroup_forum_topic"
    assert result["binding_status"] == "metadata_only_mismatch"
    assert result["binding_authority"] == "metadata_only"
    assert result["binding_routing_effect"] == "metadata_only"
    assert result["private_topic_mode_enabled"] is None
    assert ("topic_mode", "-10055", "77") not in db.reads


@pytest.mark.asyncio
async def test_plain_private_chat_reports_native_topic_mode_when_available(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    db = FakeSessionDB(topic_mode=True)
    _runner(monkeypatch, loop, adapter=FakeAdapter(), db=db)

    result = await _call_with_context(
        service.get_current_telegram_topic_status,
        chat_id="55",
        chat_type="private",
        user_id="77",
    )

    assert result["surface"] == "plain_chat"
    assert result["private_topic_mode_enabled"] is True
    assert result["binding"] is None
    assert result["binding_status"] == "missing"


@pytest.mark.asyncio
async def test_secondary_profile_selects_its_adapter_without_default_fallback(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    secondary = FakeAdapter()
    runner = _runner(
        monkeypatch,
        loop,
        adapter=secondary,
        profile="secondary",
    )
    default = runner.adapters[Platform.TELEGRAM]

    result = await _call_with_context(
        service.create_current_telegram_topic,
        title=" Secondary ",
        chat_id="55",
        chat_type="dm",
        profile="secondary",
    )

    assert result["success"] is True
    assert secondary.dm_creates == [(55, "Secondary")]
    assert default.dm_creates == []

    runner._profile_adapters = {}
    rejected = await _call_with_context(
        service.create_current_telegram_topic,
        title="No fallback",
        chat_id="55",
        chat_type="dm",
        profile="secondary",
    )
    _assert_error(rejected, "telegram_adapter_unavailable")
    assert default.dm_creates == []


@pytest.mark.asyncio
async def test_private_create_and_rename_use_live_gateway_loop_and_selected_adapter(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    adapter = FakeAdapter()
    db = FakeSessionDB()
    _runner(monkeypatch, loop, adapter=adapter, db=db)

    created = await _call_with_context(
        service.create_current_telegram_topic,
        title="  Project   Topic  ",
        chat_id="55",
        chat_type="dm",
        thread_id="19",
    )
    renamed = await _call_with_context(
        service.rename_current_telegram_topic,
        title="Renamed",
        chat_id="55",
        chat_type="private",
        thread_id="19",
    )

    assert created == {
        "success": True,
        "operation": "create_topic",
        "surface": "private_topic",
        "chat_id": "55",
        "thread_id": "221",
        "title": "Project Topic",
        "binding_created": False,
        "routing_effect": "topic_created_only",
    }
    assert renamed["success"] is True
    assert renamed["thread_id"] == "19"
    assert renamed["routing_effect"] == "topic_renamed_only"
    assert adapter.dm_creates == [(55, "Project Topic")]
    assert adapter.dm_renames == [("55", "19", "Renamed")]
    assert all(operation_loop is loop for operation_loop in adapter.operation_loops)
    assert db.writes == []


@pytest.mark.asyncio
async def test_supergroup_create_and_rename_use_bot_forum_api(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    adapter = FakeAdapter()
    _runner(monkeypatch, loop, adapter=adapter)

    created = await _call_with_context(
        service.create_current_telegram_topic,
        title="Forum Topic",
        chat_id="-10055",
        chat_type="supergroup",
        thread_id="19",
    )
    renamed = await _call_with_context(
        service.rename_current_telegram_topic,
        title="Forum Renamed",
        chat_id="-10055",
        chat_type="group",
        thread_id="19",
    )

    assert created["thread_id"] == "331"
    assert created["binding_created"] is False
    assert adapter.group_creates == [(-10055, "Forum Topic")]
    assert adapter.group_renames == [(-10055, 19, "Forum Renamed")]
    assert adapter.dm_creates == []
    assert adapter.dm_renames == []


@pytest.mark.asyncio
async def test_rename_without_authoritative_current_thread_fails_closed(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    adapter = FakeAdapter()
    _runner(monkeypatch, loop, adapter=adapter)

    result = await _call_with_context(
        service.rename_current_telegram_topic,
        title="Cannot target another topic",
        chat_id="-10055",
        chat_type="supergroup",
    )

    _assert_error(result, "missing_thread_id")
    assert adapter.group_renames == []


@pytest.mark.asyncio
async def test_private_rename_without_live_bot_fails_as_unsupported_capability(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    adapter = FakeAdapter()
    adapter._bot = None
    _runner(monkeypatch, loop, adapter=adapter)

    result = await _call_with_context(
        service.rename_current_telegram_topic,
        title="No false success",
        chat_id="55",
        chat_type="dm",
        thread_id="19",
    )

    _assert_error(result, "unsupported_capability")
    assert adapter.dm_renames == []


@pytest.mark.asyncio
async def test_unspecced_magicmock_adapter_cannot_fabricate_capabilities(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    adapter = MagicMock()
    _runner(monkeypatch, loop, adapter=adapter)

    created = await _call_with_context(
        service.create_current_telegram_topic,
        title="Synthetic create",
        chat_id="55",
        chat_type="dm",
    )
    renamed = await _call_with_context(
        service.rename_current_telegram_topic,
        title="Synthetic rename",
        chat_id="55",
        chat_type="dm",
        thread_id="19",
    )

    _assert_error(created, "unsupported_capability")
    _assert_error(renamed, "unsupported_capability")


@pytest.mark.asyncio
async def test_unspecced_asyncmock_bot_cannot_fabricate_forum_api(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    adapter = FakeAdapter()
    adapter._bot = AsyncMock()
    _runner(monkeypatch, loop, adapter=adapter)

    created = await _call_with_context(
        service.create_current_telegram_topic,
        title="Synthetic forum create",
        chat_id="-10055",
        chat_type="supergroup",
    )
    renamed = await _call_with_context(
        service.rename_current_telegram_topic,
        title="Synthetic forum rename",
        chat_id="-10055",
        chat_type="supergroup",
        thread_id="19",
    )

    _assert_error(created, "unsupported_capability")
    _assert_error(renamed, "unsupported_capability")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup", "operation", "expected_code"),
    [
        ("missing_runner", "status", "gateway_runner_unavailable"),
        ("missing_adapter", "create", "telegram_adapter_unavailable"),
        ("missing_capability", "create", "unsupported_capability"),
        ("missing_chat", "create", "missing_chat_id"),
        ("closed_loop", "rename", "gateway_loop_unavailable"),
    ],
)
async def test_missing_runtime_dependencies_fail_closed(
    monkeypatch,
    setup,
    operation,
    expected_code,
):
    service = _service()
    loop = asyncio.get_running_loop()
    adapter = FakeAdapter()
    runner = _runner(monkeypatch, loop, adapter=adapter)

    if setup == "missing_runner":
        import gateway.run as gateway_run

        monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: None)
    elif setup == "missing_adapter":
        runner.adapters = {}
    elif setup == "missing_capability":
        adapter._bot = None
        adapter._create_dm_topic = None
    elif setup == "closed_loop":
        closed = asyncio.new_event_loop()
        closed.close()
        runner._gateway_loop = closed

    function = {
        "status": service.get_current_telegram_topic_status,
        "create": service.create_current_telegram_topic,
        "rename": service.rename_current_telegram_topic,
    }[operation]
    result = await _call_with_context(
        function,
        title=None if operation == "status" else "Title",
        chat_id="" if setup == "missing_chat" else "55",
        chat_type="dm",
        thread_id="19",
    )
    _assert_error(result, expected_code)


@pytest.mark.asyncio
async def test_api_error_returns_structured_failure_without_binding_mutation(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    db = FakeSessionDB()
    adapter = FakeAdapter()
    adapter._bot.create_forum_topic = AsyncMock(side_effect=RuntimeError("telegram down"))
    _runner(monkeypatch, loop, adapter=adapter, db=db)

    result = await _call_with_context(
        service.create_current_telegram_topic,
        title="Fail safely",
        chat_id="-10055",
        chat_type="supergroup",
    )

    _assert_error(result, "telegram_api_error")
    assert "telegram down" not in result["error"]["message"]
    assert db.writes == []


@pytest.mark.asyncio
async def test_timeout_returns_structured_failure_and_cancels_gateway_work(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    cancelled = asyncio.Event()
    adapter = FakeAdapter()

    async def never_finishes(chat_id, name):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    adapter._create_dm_topic = never_finishes
    _runner(monkeypatch, loop, adapter=adapter)
    monkeypatch.setattr(service, "_GATEWAY_CALL_TIMEOUT_SECONDS", 0.01)

    result = await _call_with_context(
        service.create_current_telegram_topic,
        title="Timeout",
        chat_id="55",
        chat_type="dm",
    )

    _assert_error(result, "gateway_timeout")
    await asyncio.wait_for(cancelled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_shutdown_admission_boundary_rejects_late_loop_submission(
    monkeypatch,
    caplog,
):
    service = _service()
    loop = asyncio.get_running_loop()
    adapter = FakeAdapter()
    runner = _runner(monkeypatch, loop, adapter=adapter)
    runner._telegram_lifecycle_accepting = True
    runner._telegram_lifecycle_tasks = set()

    submission_paused = threading.Event()
    resume_submission = threading.Event()
    result_queue = queue.Queue(maxsize=1)
    original_call_soon_threadsafe = loop.call_soon_threadsafe
    original_work = service._create_on_gateway_loop
    coroutine_creations = []

    def observe_work(*args):
        coroutine_creations.append(args)
        return original_work(*args)

    def pause_lifecycle_submission(callback, *args, context=None):
        if getattr(callback, "__name__", "") == "_start_on_loop":
            submission_paused.set()
            assert resume_submission.wait(timeout=1)
        return original_call_soon_threadsafe(callback, *args, context=context)

    monkeypatch.setattr(service, "_create_on_gateway_loop", observe_work)
    monkeypatch.setattr(loop, "call_soon_threadsafe", pause_lifecycle_submission)

    tokens = set_session_vars(platform="telegram", chat_id="55", chat_type="dm")
    worker_context = copy_context()

    def invoke():
        result_queue.put(
            worker_context.run(service.create_current_telegram_topic, "Too late")
        )

    worker = threading.Thread(target=invoke, daemon=True)
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            worker.start()
            while not submission_paused.is_set():
                await asyncio.sleep(0.001)

            await service.cancel_gateway_telegram_lifecycle_tasks(runner)
            assert runner._telegram_lifecycle_accepting is False
            assert runner._telegram_lifecycle_tasks == set()

            resume_submission.set()
            while result_queue.empty():
                await asyncio.sleep(0.001)
            result = result_queue.get_nowait()
            worker.join(timeout=1)

            _assert_error(result, "gateway_runner_unavailable")
            assert coroutine_creations == []
            assert adapter.dm_creates == []
            assert runner._telegram_lifecycle_tasks == set()
            gc.collect()
            assert not any(
                "was never awaited" in str(item.message) for item in recorded
            )
            assert "exception calling callback" not in caplog.text.lower()
    finally:
        clear_session_vars(tokens)
        resume_submission.set()
        worker.join(timeout=1)


def test_timeout_stopped_loop_race_keeps_task_owned_until_gateway_cleanup(
    monkeypatch,
    caplog,
):
    service = _service()
    import gateway.run as gateway_run

    loop = asyncio.new_event_loop()
    runner = object.__new__(GatewayRunner)
    runner._gateway_loop = loop
    adapter = FakeAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    monkeypatch.setattr(service, "_GATEWAY_CALL_TIMEOUT_SECONDS", 0.03)

    operation_started = threading.Event()
    loop_stopped = threading.Event()
    cleanup_requested = threading.Event()
    loop_closed = threading.Event()
    observed_tasks = []

    async def stop_loop_while_pending(chat_id, name):
        operation_started.set()
        loop.call_soon(loop.stop)
        await asyncio.Event().wait()

    adapter._create_dm_topic = stop_loop_while_pending

    def loop_worker():
        asyncio.set_event_loop(loop)

        def pulse_restricted_selector():
            if not loop_stopped.is_set():
                loop.call_later(0.001, pulse_restricted_selector)

        loop.call_soon(pulse_restricted_selector)
        loop.run_forever()
        loop_stopped.set()
        cleanup_requested.wait(timeout=2)
        cleanup = getattr(
            service,
            "cancel_gateway_telegram_lifecycle_tasks",
            None,
        )
        if callable(cleanup):
            loop.run_until_complete(cleanup(runner))
        else:
            # RED cleanup only: prevent the pre-fix implementation from
            # leaking its real pending Task after the ownership assertion.
            pending = list(asyncio.all_tasks(loop))
            observed_tasks.extend(pending)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        loop.close()
        loop_closed.set()

    loop_thread = threading.Thread(target=loop_worker, daemon=True)
    loop_thread.start()
    tokens = set_session_vars(platform="telegram", chat_id="55", chat_type="dm")
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = service.create_current_telegram_topic("Race")
            assert operation_started.wait(timeout=1)
            assert loop_stopped.wait(timeout=1)
            _assert_error(result, "gateway_timeout")

            owned = getattr(runner, "_telegram_lifecycle_tasks", set())
            assert len(owned) == 1
            owned_task = next(iter(owned))
            assert owned_task.done() is False

            cleanup_requested.set()
            assert loop_closed.wait(timeout=2)
            loop_thread.join(timeout=1)

            assert owned_task.done() is True
            assert getattr(runner, "_telegram_lifecycle_tasks", set()) == set()
            gc.collect()
            assert not any(
                "was never awaited" in str(item.message) for item in recorded
            )
            assert "exception calling callback" not in caplog.text.lower()
    finally:
        clear_session_vars(tokens)
        cleanup_requested.set()
        loop_thread.join(timeout=2)
        if not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
        for task in observed_tasks:
            task._log_destroy_pending = False


def test_task_creation_shutdown_race_closes_unowned_coroutine(monkeypatch):
    service = _service()
    import gateway.run as gateway_run

    class RejectingLoop:
        submitted_coroutine = None

        def is_closed(self):
            return False

        def is_running(self):
            return True

        def call_soon_threadsafe(self, callback):
            callback()

        def create_task(self, coroutine):
            self.submitted_coroutine = coroutine
            raise RuntimeError("Event loop is closing")

    loop = RejectingLoop()
    runner = object.__new__(GatewayRunner)
    runner._gateway_loop = loop
    runner.adapters = {Platform.TELEGRAM: FakeAdapter()}
    runner._profile_adapters = {}
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)

    tokens = set_session_vars(platform="telegram", chat_id="55", chat_type="dm")
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = service.create_current_telegram_topic("Closing")
            coroutine = loop.submitted_coroutine
            assert coroutine is not None
            assert coroutine.cr_frame is None
            del coroutine
            gc.collect()
            assert not any(
                "was never awaited" in str(item.message) for item in recorded
            )
    finally:
        clear_session_vars(tokens)

    _assert_error(result, "gateway_operation_error")
