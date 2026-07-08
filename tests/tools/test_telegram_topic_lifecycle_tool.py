import json

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_state import SessionDB
from tools.telegram_topic_lifecycle_tool import (
    TELEGRAM_TOPIC_MANAGE_SCHEMA,
    TELEGRAM_TOPIC_STATUS_SCHEMA,
    _run_maybe_awaitable,
    telegram_topic_manage,
    telegram_topic_status,
)


class FakeTelegramLifecycleClient:
    def __init__(self):
        self.calls = []

    def create_topic(self, *, chat_id, title):
        self.calls.append(("create_topic", chat_id, title))
        return "222"

    def rename_topic(self, *, chat_id, thread_id, title):
        self.calls.append(("rename_topic", chat_id, thread_id, title))
        return True


def _decode(result):
    return json.loads(result) if isinstance(result, str) else result


def test_run_maybe_awaitable_preserves_original_coroutine_error():
    async def fails_once():
        raise RuntimeError("telegram api failed")

    with pytest.raises(RuntimeError, match="telegram api failed"):
        _run_maybe_awaitable(fails_once())


def test_tool_schemas_are_narrow_and_registered_to_one_toolset():
    from toolsets import resolve_toolset

    assert TELEGRAM_TOPIC_STATUS_SCHEMA["name"] == "telegram_topic_status"
    assert TELEGRAM_TOPIC_MANAGE_SCHEMA["name"] == "telegram_topic_manage"
    assert TELEGRAM_TOPIC_MANAGE_SCHEMA["parameters"]["properties"]["action"]["enum"] == [
        "create_topic",
        "rename_topic",
        "bind_session",
    ]
    assert {
        "telegram_topic_status",
        "telegram_topic_manage",
    }.issubset(set(resolve_toolset("hermes-telegram")))
    assert "telegram_pin_manage" not in set(resolve_toolset("hermes-telegram"))


def test_status_reports_current_context_and_binding_without_pin_lifecycle(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess-topic", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key="telegram:dm:208214988:17585",
        session_id="sess-topic",
        managed_mode="managed",
    )

    tokens = set_session_vars(
        platform="telegram",
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key="telegram:dm:208214988:17585",
        session_id="sess-topic",
    )
    try:
        result = _decode(telegram_topic_status({}, session_db=db))
    finally:
        clear_session_vars(tokens)

    assert result["success"] is True
    assert result["context"]["platform"] == "telegram"
    assert result["context"]["chat_id"] == "208214988"
    assert result["context"]["thread_id"] == "17585"
    assert result["surface"] == "private_topic"
    assert result["binding"]["session_id"] == "sess-topic"
    assert result["binding_status"] == "matches_routing"
    assert result["binding_authority"] == "private_topic_binding"
    assert "known_pins" not in result
    assert "known_pins_complete" not in result
    assert "known_pins_model" not in result


def test_status_reports_supergroup_forum_routing_separate_from_binding_metadata(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("source-session", source="telegram", user_id="208214988")
    db.create_session("current-topic-session", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="-1004313189330",
        thread_id="22",
        user_id="208214988",
        session_key="agent:main:telegram:group:-1004313189330:1",
        session_id="source-session",
        managed_mode="managed",
    )

    tokens = set_session_vars(
        platform="telegram",
        chat_id="-1004313189330",
        chat_type="group",
        thread_id="22",
        user_id="208214988",
        session_key="agent:main:telegram:group:-1004313189330:22",
        session_id="current-topic-session",
    )
    try:
        result = _decode(telegram_topic_status({}, session_db=db))
    finally:
        clear_session_vars(tokens)

    assert result["success"] is True
    assert result["surface"] == "supergroup_forum_topic"
    assert result["context"]["chat_type"] == "group"
    assert result["topic_mode_enabled"] is False
    assert result["routing"] == {
        "chat_id": "-1004313189330",
        "thread_id": "22",
        "session_id": "current-topic-session",
        "session_key": "agent:main:telegram:group:-1004313189330:22",
    }
    assert result["binding"]["session_id"] == "source-session"
    assert result["binding"]["session_key"] == "agent:main:telegram:group:-1004313189330:1"
    assert result["binding_status"] == "metadata_only_mismatch"
    assert result["binding_matches_routing"] is False
    assert result["binding_authority"] == "metadata_only"
    assert "known_pins" not in result
    assert "known_pins_complete" not in result
    assert "known_pins_model" not in result


def test_topic_manage_bind_session_requires_approval_metadata(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess-topic", source="telegram", user_id="208214988")
    tokens = set_session_vars(
        platform="telegram",
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key="telegram:dm:208214988:17585",
    )
    try:
        denied = _decode(
            telegram_topic_manage(
                {"action": "bind_session", "session_id": "sess-topic"},
                session_db=db,
            )
        )
        allowed = _decode(
            telegram_topic_manage(
                {
                    "action": "bind_session",
                    "session_id": "sess-topic",
                    "approval": {
                        "approved_by": "operator",
                        "approval_text_excerpt": "Bind this topic to the session",
                    },
                },
                session_db=db,
            )
        )
    finally:
        clear_session_vars(tokens)

    assert denied["success"] is False
    assert "approval" in denied["error"].lower()
    assert allowed["success"] is True
    assert allowed["binding"]["session_id"] == "sess-topic"
    assert allowed["surface"] == "private_topic"
    assert allowed["binding_authority"] == "private_topic_binding"
    assert allowed["routing_effect"] == "private_topic_binding"
    assert db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
    )["managed_mode"] == "managed"


def test_topic_manage_bind_session_in_supergroup_reports_metadata_only(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess-source", source="telegram", user_id="208214988")
    tokens = set_session_vars(
        platform="telegram",
        chat_id="-1004313189330",
        chat_type="group",
        thread_id="22",
        user_id="208214988",
        session_key="agent:main:telegram:group:-1004313189330:22",
    )
    try:
        bound = _decode(
            telegram_topic_manage(
                {
                    "action": "bind_session",
                    "session_id": "sess-source",
                    "session_key": "agent:main:telegram:group:-1004313189330:1",
                    "approval": {
                        "approved_by": "operator",
                        "approval_text_excerpt": "Record lifecycle metadata for this topic",
                    },
                },
                session_db=db,
            )
        )
    finally:
        clear_session_vars(tokens)

    assert bound["success"] is True
    assert bound["surface"] == "supergroup_forum_topic"
    assert bound["binding"]["session_id"] == "sess-source"
    assert bound["binding_authority"] == "metadata_only"
    assert bound["routing_effect"] == "metadata_only"
    assert "does not change gateway routing" in bound["warning"]


def test_topic_manage_create_and_rename_use_injected_client(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    client = FakeTelegramLifecycleClient()
    tokens = set_session_vars(
        platform="telegram",
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
    )
    approval = {
        "approved_by": "operator",
        "approval_text_excerpt": "Create and rename approved",
    }
    try:
        created = _decode(
            telegram_topic_manage(
                {
                    "action": "create_topic",
                    "title": "Project 0035",
                    "approval": approval,
                },
                session_db=db,
                telegram_client=client,
            )
        )
        renamed = _decode(
            telegram_topic_manage(
                {
                    "action": "rename_topic",
                    "thread_id": "222",
                    "title": "Project 0035 Updated",
                    "approval": approval,
                },
                session_db=db,
                telegram_client=client,
            )
        )
    finally:
        clear_session_vars(tokens)

    assert created["success"] is True
    assert created["thread_id"] == "222"
    assert created["binding_created"] is False
    assert created["routing_effect"] == "topic_created_only"
    assert db.get_telegram_topic_binding(chat_id="208214988", thread_id="222") is None
    assert renamed["success"] is True
    assert client.calls == [
        ("create_topic", "208214988", "Project 0035"),
        ("rename_topic", "208214988", "222", "Project 0035 Updated"),
    ]


def test_topic_manage_update_metadata_is_not_supported(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    tokens = set_session_vars(
        platform="telegram",
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_id="sess-topic",
    )
    try:
        result = _decode(
            telegram_topic_manage(
                {
                    "action": "update_metadata",
                    "message_id": "777",
                    "kind": "context",
                    "label": "Scope note",
                    "text_snapshot": "Scope: topic lifecycle MVP.",
                    "metadata": {"task": "0035", "state": "in_progress"},
                    "approval": {
                        "approved_by": "operator",
                        "approval_text_excerpt": "Record this metadata",
                    },
                },
                session_db=db,
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["success"] is False
    assert "unknown telegram_topic_manage action" in result["error"]
