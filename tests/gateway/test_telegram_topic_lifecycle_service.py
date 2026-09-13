import asyncio
import importlib
import inspect
import queue
import threading
from contextvars import copy_context
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session_context import _UNSET, _VAR_MAP, clear_session_vars, set_session_vars


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
    runner._draining = False
    runner._telegram_lifecycle_accepting = True
    runner._telegram_lifecycle_tasks = set()
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
            result = worker_context.run(function) if title is None else worker_context.run(function, title)
            result_queue.put((True, result))
        except BaseException as exc:
            result_queue.put((False, exc))

    threading.Thread(target=invoke, daemon=True).start()
    while result_queue.empty():
        await asyncio.sleep(0.001)
    succeeded, result = result_queue.get_nowait()
    if not succeeded:
        raise result
    return result


async def _call_with_context(function, *, title=None, **context):
    tokens = set_session_vars(platform=context.pop("platform", "telegram"), **context)
    try:
        return await _call_in_worker(function, title=title)
    finally:
        clear_session_vars(tokens)


def _assert_error(result, code):
    assert result["success"] is False
    assert result["error"]["code"] == code


def test_public_service_contract_has_no_routing_or_message_overrides():
    service = _service()

    assert list(inspect.signature(service.get_current_telegram_topic_status).parameters) == []
    assert list(inspect.signature(service.create_current_telegram_topic).parameters) == ["title"]
    assert list(inspect.signature(service.rename_current_telegram_topic).parameters) == ["title"]


@pytest.mark.asyncio
async def test_status_does_not_use_process_global_identity(monkeypatch):
    service = _service()
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "stale-chat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "dm")

    result = await _call_in_worker(service.get_current_telegram_topic_status)

    _assert_error(result, "not_telegram_context")


@pytest.mark.asyncio
async def test_status_reports_private_binding_and_forum_metadata_semantics(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    db = FakeSessionDB(
        binding={"session_id": "session-19", "session_key": "private-key"},
        topic_mode=True,
    )
    _runner(monkeypatch, loop, adapter=FakeAdapter(), db=db)

    private = await _call_with_context(
        service.get_current_telegram_topic_status,
        chat_id="55",
        chat_type="dm",
        thread_id="19",
        user_id="77",
        session_key="private-key",
        session_id="session-19",
    )
    assert private["surface"] == "private_topic"
    assert private["binding_status"] == "matches_routing"
    assert private["binding_authority"] == "private_topic_binding"
    assert private["binding_routing_effect"] == "native_private_topic_routing"
    assert private["private_topic_mode_enabled"] is True

    db.binding = {"session_id": "old-session", "session_key": "old-key"}
    forum = await _call_with_context(
        service.get_current_telegram_topic_status,
        chat_id="-10055",
        chat_type="forum",
        thread_id="1",
        user_id="77",
        session_key="general-key",
        session_id="general-session",
    )
    assert forum["surface"] == "supergroup_forum_topic"
    assert forum["binding_status"] == "metadata_only_mismatch"
    assert forum["binding_authority"] == "metadata_only"
    assert forum["binding_routing_effect"] == "metadata_only"
    assert forum["private_topic_mode_enabled"] is None


@pytest.mark.asyncio
async def test_private_mutations_use_profile_adapter_and_gateway_loop(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    adapter = FakeAdapter()
    runner = _runner(monkeypatch, loop, adapter=adapter, profile="ops")
    default = runner.adapters[Platform.TELEGRAM]

    created = await _call_with_context(
        service.create_current_telegram_topic,
        title="  Project   Topic  ",
        chat_id="55",
        chat_type="dm",
        thread_id="19",
        profile="ops",
    )
    renamed = await _call_with_context(
        service.rename_current_telegram_topic,
        title="Renamed",
        chat_id="55",
        chat_type="private",
        thread_id="19",
        profile="ops",
    )

    assert created["thread_id"] == "221"
    assert created["routing_effect"] == "topic_created_only"
    assert renamed["routing_effect"] == "topic_renamed_only"
    assert adapter.dm_creates == [(55, "Project Topic")]
    assert adapter.dm_renames == [("55", "19", "Renamed")]
    assert default.dm_creates == default.dm_renames == []
    assert all(operation_loop is loop for operation_loop in adapter.operation_loops)

    runner._profile_adapters.clear()
    rejected = await _call_with_context(
        service.create_current_telegram_topic,
        title="No fallback",
        chat_id="55",
        chat_type="dm",
        profile="ops",
    )
    _assert_error(rejected, "telegram_adapter_unavailable")
    assert default.dm_creates == []


@pytest.mark.asyncio
async def test_forum_create_and_general_topic_rename_use_bot_api_only(monkeypatch):
    service = _service()
    loop = asyncio.get_running_loop()
    adapter = FakeAdapter()
    db = FakeSessionDB()
    _runner(monkeypatch, loop, adapter=adapter, db=db)

    created = await _call_with_context(
        service.create_current_telegram_topic,
        title="Forum Topic",
        chat_id="-10055",
        chat_type="supergroup",
        thread_id="1",
    )
    renamed = await _call_with_context(
        service.rename_current_telegram_topic,
        title="General Renamed",
        chat_id="-10055",
        chat_type="forum",
        thread_id="1",
    )

    assert created["thread_id"] == "331"
    assert created["binding_created"] is False
    assert renamed["thread_id"] == "1"
    assert adapter.group_creates == [(-10055, "Forum Topic")]
    assert adapter.group_renames == [(-10055, 1, "General Renamed")]
    assert adapter.dm_creates == adapter.dm_renames == []
    assert db.writes == []


@pytest.mark.asyncio
async def test_rename_without_current_thread_fails_closed(monkeypatch):
    service = _service()
    adapter = FakeAdapter()
    _runner(monkeypatch, asyncio.get_running_loop(), adapter=adapter)

    result = await _call_with_context(
        service.rename_current_telegram_topic,
        title="Cannot target another topic",
        chat_id="-10055",
        chat_type="forum",
    )

    _assert_error(result, "missing_thread_id")
    assert adapter.group_renames == []


@pytest.mark.asyncio
async def test_timeout_cancels_gateway_work(monkeypatch):
    service = _service()
    cancelled = asyncio.Event()
    adapter = FakeAdapter()

    async def never_finishes(chat_id, name):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    adapter._create_dm_topic = never_finishes
    _runner(monkeypatch, asyncio.get_running_loop(), adapter=adapter)
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
async def test_shutdown_cancels_and_reaps_owned_lifecycle_tasks():
    service = _service()
    runner = SimpleNamespace(
        _telegram_lifecycle_accepting=True,
        _telegram_lifecycle_tasks=set(),
    )
    cancelled = asyncio.Event()

    async def wait_forever():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(wait_forever())
    await asyncio.sleep(0)
    runner._telegram_lifecycle_tasks.add(task)
    await service.cancel_gateway_telegram_lifecycle_tasks(runner)

    assert runner._telegram_lifecycle_accepting is False
    assert runner._telegram_lifecycle_tasks == set()
    assert task.cancelled()
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_gateway_shutdown_cancels_lifecycle_before_adapter_teardown(monkeypatch):
    from gateway.run_shutdown import GatewayShutdownMixin

    service = _service()
    events = []

    async def cancel(_runner):
        events.append("lifecycle")

    async def finalize(_agents):
        return None

    async def teardown(_adapter, _platform, **_kwargs):
        events.append("adapter")

    monkeypatch.setattr(service, "cancel_gateway_telegram_lifecycle_tasks", cancel)
    runner = SimpleNamespace(
        _restart_requested=False,
        _restart_detached=False,
        _finalize_shutdown_agents=finalize,
        _agent_cache_lock=None,
        _agent_cache=None,
        _cancel_process_completion_batch_tasks=None,
        adapters={Platform.TELEGRAM: object()},
        _profile_adapters={},
        _bounded_adapter_teardown=teardown,
    )
    ctx = SimpleNamespace(active_agents={}, elapsed=lambda: 0.0)

    await GatewayShutdownMixin._stop_finalize_agents_and_adapters(runner, ctx)

    assert events == ["lifecycle", "adapter"]
