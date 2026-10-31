"""Separate native presentation does not turn the execution source into UI text."""
import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.platforms.base import SendResult


class CaptureTelegram(TelegramAdapter):
    async def _send_exec_approval_prompt(self, prompt):
        self.prompts.append(prompt)
        return SendResult(success=True, message_id="synthetic")


@pytest.mark.asyncio
async def test_native_summary_is_plain_localized_and_preserves_request_choices(monkeypatch):
    from gateway.platforms import base_exec_approval
    monkeypatch.setattr(base_exec_approval, "approval_timeout_seconds", lambda: 300)
    adapter = object.__new__(CaptureTelegram)
    adapter.prompts = []
    code = "print('SOURCE_SENTINEL_9284')"
    await adapter.send_exec_approval("chat", code, "session", description="script",
        allow_session=False, allow_permanent=False,
        summary="План агента\nСоздать два предложения <unsafe>", summary_language="ru")
    (prompt,) = adapter.prompts
    assert prompt.command == code
    assert prompt.choices == ["once", "deny"]
    assert "Создать два предложения &lt;unsafe&gt;" in prompt.text
    assert "<pre>" not in prompt.text and "SOURCE_SENTINEL_9284" not in prompt.text
    assert "5 мин" in prompt.text and "Why it was flagged" not in prompt.text
    assert all(any("А" <= ch <= "я" for ch in label) for label, _, _ in prompt.actions)


@pytest.mark.asyncio
async def test_native_summary_respects_telegram_html_budget():
    adapter = object.__new__(CaptureTelegram)
    adapter.prompts = []
    await adapter.send_exec_approval("chat", "SOURCE_SENTINEL_9284", "session",
        summary="<&>" * 10000, summary_language="ru")
    text = adapter.prompts[0].text
    assert len(text) <= 4096 and "SOURCE_SENTINEL_9284" not in text
    assert "<pre>" not in text and "…" in text


def test_legacy_button_callback_does_not_receive_new_optional_kwargs():
    from gateway.platforms import base_exec_approval
    class Legacy:
        def send_exec_approval(self, chat_id, command, session_key, description=None,
                               *, metadata=None, allow_session=True, allow_permanent=True, smart_denied=False):
            pass
    data = {"command": "original source", "description": "rule", "summary": "Human plan", "summary_language": "ru"}
    command, description, flags, presentation = base_exec_approval.prepare_exec_approval_display(
        data, lambda value: value, Legacy())
    assert command == "original source" and description == "rule"
    assert presentation == {} and flags["allow_session"] is True


@pytest.mark.parametrize("buttons_work", [True, False])
def test_turn_runner_relays_native_presentation_without_replacing_source(monkeypatch, buttons_work):
    import asyncio
    from concurrent.futures import Future
    from types import SimpleNamespace
    from gateway.run_turn_runner import TurnRunner

    adapter = object.__new__(CaptureTelegram)
    adapter.prompts = []
    ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id="chat",
                          _status_thread_metadata={}, session_key="synthetic-session")
    runner = TurnRunner(None, ctx)
    monkeypatch.setattr(runner, "_close_native_stream_boundary", lambda _: None)
    monkeypatch.setattr(adapter, "pause_typing_for_chat", lambda _: None)
    fallback = []
    notices = []
    async def send(*args, **kwargs):
        fallback.append(args[1])
        return SendResult(success=True)
    async def buttons(prompt):
        adapter.prompts.append(prompt)
        return SendResult(success=buttons_work, message_id="synthetic")
    monkeypatch.setattr(adapter, "send", send)
    monkeypatch.setattr(adapter, "_send_exec_approval_prompt", buttons)
    monkeypatch.setattr("gateway.run_turn_runner_approval_settle.register_timeout_notice",
                        lambda *args, **kwargs: notices.append(kwargs))

    def schedule(coro, _):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    monkeypatch.setattr(runner, "_schedule", schedule)
    runner._approval_notify_sync({"command": "SOURCE_SENTINEL_9284", "description": "code",
        "summary": "План агента\nСоздать два файла", "summary_language": "ru",
        "allow_session": False, "allow_permanent": False})
    (prompt,) = adapter.prompts
    assert prompt.command == "SOURCE_SENTINEL_9284"
    assert prompt.summary == "План агента\nСоздать два файла"
    assert prompt.summary_language == "ru"
    assert "Создать два файла" in prompt.text and "SOURCE_SENTINEL_9284" not in prompt.text
    assert prompt.choices == ["once", "deny"]
    assert notices[0]["command"] == prompt.summary
    assert all("SOURCE_SENTINEL_9284" not in text for text in fallback)
    assert bool(fallback) is not buttons_work
