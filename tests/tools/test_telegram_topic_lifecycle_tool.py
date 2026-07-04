import json

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_state import SessionDB
from tools.telegram_topic_lifecycle_tool import (
    TELEGRAM_PIN_MANAGE_SCHEMA,
    TELEGRAM_TOPIC_MANAGE_SCHEMA,
    TELEGRAM_TOPIC_STATUS_SCHEMA,
    telegram_pin_manage,
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

    def send_message(self, *, chat_id, thread_id, message):
        self.calls.append(("send_message", chat_id, thread_id, message))
        return "888"

    def pin_message(self, *, chat_id, message_id):
        self.calls.append(("pin_message", chat_id, message_id))
        return True

    def unpin_message(self, *, chat_id, message_id):
        self.calls.append(("unpin_message", chat_id, message_id))
        return True


def _decode(result):
    return json.loads(result) if isinstance(result, str) else result


def test_tool_schemas_are_narrow_and_registered_to_one_toolset():
    from toolsets import resolve_toolset

    assert TELEGRAM_TOPIC_STATUS_SCHEMA["name"] == "telegram_topic_status"
    assert TELEGRAM_TOPIC_MANAGE_SCHEMA["name"] == "telegram_topic_manage"
    assert TELEGRAM_PIN_MANAGE_SCHEMA["name"] == "telegram_pin_manage"
    assert TELEGRAM_TOPIC_MANAGE_SCHEMA["parameters"]["properties"]["action"]["enum"] == [
        "create_topic",
        "rename_topic",
        "bind_session",
        "update_metadata",
    ]
    assert TELEGRAM_PIN_MANAGE_SCHEMA["parameters"]["properties"]["action"]["enum"] == [
        "post_and_pin",
        "pin_existing",
        "unpin",
        "record_pin_metadata",
        "list_known_pins",
    ]
    assert {
        "telegram_topic_status",
        "telegram_topic_manage",
        "telegram_pin_manage",
    }.issubset(set(resolve_toolset("hermes-telegram")))


def test_status_reports_current_context_binding_and_known_markers(tmp_path):
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
    db.record_telegram_pinned_marker(
        chat_id="208214988",
        thread_id="17585",
        message_id="777",
        kind="topic_passport",
        label="Project bootstrap",
        metadata={"task": "0035"},
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
    assert result["binding"]["session_id"] == "sess-topic"
    assert result["known_pins"][0]["message_id"] == "777"
    assert result["known_pins_complete"] is False


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
    assert db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
    )["managed_mode"] == "managed"


def test_pin_manage_record_metadata_requires_approval_and_lists_known_pins(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    tokens = set_session_vars(
        platform="telegram",
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_id="sess-topic",
    )
    try:
        denied = _decode(
            telegram_pin_manage(
                {
                    "action": "record_pin_metadata",
                    "message_id": "777",
                    "kind": "topic_passport",
                    "label": "Project bootstrap",
                },
                session_db=db,
            )
        )
        allowed = _decode(
            telegram_pin_manage(
                {
                    "action": "record_pin_metadata",
                    "message_id": "777",
                    "kind": "topic_passport",
                    "label": "Project bootstrap",
                    "text_snapshot": "Purpose: keep project context recoverable.",
                    "metadata": {"task": "0035"},
                    "approval": {
                        "approved_by": "operator",
                        "approval_text_excerpt": "Record this pinned marker",
                    },
                },
                session_db=db,
            )
        )
        listed = _decode(
            telegram_pin_manage(
                {"action": "list_known_pins"},
                session_db=db,
            )
        )
    finally:
        clear_session_vars(tokens)

    assert denied["success"] is False
    assert "approval" in denied["error"].lower()
    assert allowed["success"] is True
    assert allowed["marker"]["message_id"] == "777"
    assert allowed["marker"]["metadata"] == {"task": "0035"}
    assert listed["success"] is True
    assert listed["known_pins"][0]["message_id"] == "777"
    assert listed["known_pins_complete"] is False


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
    assert renamed["success"] is True
    assert client.calls == [
        ("create_topic", "208214988", "Project 0035"),
        ("rename_topic", "208214988", "222", "Project 0035 Updated"),
    ]


def test_topic_manage_update_metadata_records_recoverable_marker(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    tokens = set_session_vars(
        platform="telegram",
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_id="sess-topic",
    )
    try:
        updated = _decode(
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

    assert updated["success"] is True
    assert updated["marker"]["message_id"] == "777"
    assert updated["marker"]["kind"] == "context"
    assert updated["marker"]["metadata"] == {"task": "0035", "state": "in_progress"}


def test_pin_manage_post_pin_existing_and_unpin_use_injected_client(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    client = FakeTelegramLifecycleClient()
    approval = {
        "approved_by": "operator",
        "approval_text_excerpt": "Pin marker approved",
    }
    tokens = set_session_vars(
        platform="telegram",
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_id="sess-topic",
    )
    try:
        posted = _decode(
            telegram_pin_manage(
                {
                    "action": "post_and_pin",
                    "message": "Purpose: project bootstrap",
                    "kind": "topic_passport",
                    "label": "Bootstrap",
                    "approval": approval,
                },
                session_db=db,
                telegram_client=client,
            )
        )
        pinned = _decode(
            telegram_pin_manage(
                {
                    "action": "pin_existing",
                    "message_id": "889",
                    "kind": "decision",
                    "label": "Decision",
                    "approval": approval,
                },
                session_db=db,
                telegram_client=client,
            )
        )
        unpinned = _decode(
            telegram_pin_manage(
                {
                    "action": "unpin",
                    "message_id": "889",
                    "approval": approval,
                },
                session_db=db,
                telegram_client=client,
            )
        )
    finally:
        clear_session_vars(tokens)

    assert posted["success"] is True
    assert posted["marker"]["message_id"] == "888"
    assert pinned["success"] is True
    assert pinned["marker"]["message_id"] == "889"
    assert unpinned["success"] is True
    assert unpinned["marker"]["state"] == "unpinned"
    assert client.calls == [
        ("send_message", "208214988", "17585", "Purpose: project bootstrap"),
        ("pin_message", "208214988", "888"),
        ("pin_message", "208214988", "889"),
        ("unpin_message", "208214988", "889"),
    ]
