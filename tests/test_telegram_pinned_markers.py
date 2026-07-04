import sqlite3

from hermes_state import SessionDB


def _tables(db: SessionDB) -> set[str]:
    rows = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row[0] for row in rows}


def test_pinned_marker_table_is_explicit_overlay(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")

    assert "telegram_pinned_markers" not in _tables(db)
    assert db.list_telegram_pinned_markers(chat_id="208214988") == []

    marker = db.record_telegram_pinned_marker(
        chat_id="208214988",
        thread_id="17585",
        message_id="777",
        kind="topic_passport",
        label="Project bootstrap",
        text_snapshot="Purpose: keep project context recoverable.",
        session_id="sess-topic",
        metadata={"repo": "hermes_infra", "task": "0035"},
        approved_by="operator",
        approval_ref="telegram:approval:42",
        approval_text_excerpt="Confirmed, create and pin it",
    )

    assert "telegram_pinned_markers" in _tables(db)
    assert marker["state"] == "pinned"
    assert marker["metadata"] == {"repo": "hermes_infra", "task": "0035"}
    assert marker["text_snapshot"] == "Purpose: keep project context recoverable."
    assert marker["approved_by"] == "operator"

    markers = db.list_telegram_pinned_markers(
        chat_id="208214988",
        thread_id="17585",
    )
    assert markers == [marker]


def test_pinned_marker_record_is_upsert_by_message_identity(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")

    first = db.record_telegram_pinned_marker(
        chat_id="208214988",
        thread_id="17585",
        message_id="777",
        kind="topic_passport",
        label="Old label",
        text_snapshot="old text",
    )
    second = db.record_telegram_pinned_marker(
        chat_id="208214988",
        thread_id="17585",
        message_id="777",
        kind="decision",
        label="New label",
        text_snapshot="new text",
        metadata={"status": "updated"},
    )

    markers = db.list_telegram_pinned_markers(
        chat_id="208214988",
        thread_id="17585",
    )

    assert len(markers) == 1
    assert second["id"] == first["id"]
    assert markers[0]["kind"] == "decision"
    assert markers[0]["label"] == "New label"
    assert markers[0]["metadata"] == {"status": "updated"}


def test_pinned_marker_state_update_keeps_recovery_history(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    marker = db.record_telegram_pinned_marker(
        chat_id="208214988",
        thread_id="17585",
        message_id="777",
        kind="context",
        label="Context note",
    )

    updated = db.update_telegram_pinned_marker_state(
        chat_id="208214988",
        message_id="777",
        state="unpinned",
    )

    assert updated["id"] == marker["id"]
    assert updated["state"] == "unpinned"
    assert db.list_telegram_pinned_markers(
        chat_id="208214988",
        thread_id="17585",
        state="pinned",
    ) == []
    assert db.list_telegram_pinned_markers(
        chat_id="208214988",
        thread_id="17585",
        include_unpinned=True,
    )[0]["state"] == "unpinned"


def test_pinned_marker_migration_does_not_require_topic_tables(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")

    db.record_telegram_pinned_marker(
        chat_id="208214988",
        thread_id="17585",
        message_id="777",
        kind="topic_passport",
        label="Standalone marker",
    )

    assert "telegram_pinned_markers" in _tables(db)
    assert "telegram_dm_topic_bindings" not in _tables(db)


def test_pinned_marker_methods_tolerate_missing_overlay_table(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")

    assert db.list_telegram_pinned_markers(chat_id="missing") == []
    try:
        db.update_telegram_pinned_marker_state(
            chat_id="missing",
            message_id="missing",
            state="unpinned",
        )
    except sqlite3.OperationalError as exc:
        raise AssertionError("missing overlay table should not leak sqlite errors") from exc
