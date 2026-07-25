import asyncio
import os
from contextlib import suppress

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    get_session_env,
)


@pytest.fixture(autouse=True)
def _reset_contextvars():
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


def test_gateway_propagates_authoritative_chat_type_and_clears_stale_env(monkeypatch):
    runner = object.__new__(GatewayRunner)
    context = SessionContext(
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="supergroup",
        ),
        connected_platforms=[],
        home_channels={},
        session_id="session-supergroup",
    )
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "stale-private")

    tokens = runner._set_session_env(context)
    assert get_session_env("HERMES_SESSION_CHAT_TYPE") == "supergroup"
    assert get_session_env("HERMES_SESSION_ID") == "session-supergroup"

    runner._clear_session_env(tokens)
    assert get_session_env("HERMES_SESSION_CHAT_TYPE") == ""
    assert os.environ["HERMES_SESSION_CHAT_TYPE"] == "stale-private"


@pytest.mark.asyncio
async def test_chat_type_is_task_local_and_copied_into_gateway_executor():
    runner = object.__new__(GatewayRunner)
    observations = {}

    async def keep_restricted_selector_moving():
        while True:
            await asyncio.sleep(0.001)

    async def observe(name: str, chat_type: str, session_id: str):
        context = SessionContext(
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id=f"chat-{name}",
                chat_type=chat_type,
            ),
            connected_platforms=[],
            home_channels={},
            session_id=session_id,
        )
        tokens = runner._set_session_env(context)
        try:
            await asyncio.sleep(0)
            observations[name] = (
                get_session_env("HERMES_SESSION_CHAT_TYPE"),
                get_session_env("HERMES_SESSION_ID"),
                await runner._run_in_executor_with_context(
                    lambda: (
                        get_session_env("HERMES_SESSION_CHAT_TYPE"),
                        get_session_env("HERMES_SESSION_ID"),
                    )
                ),
            )
        finally:
            runner._clear_session_env(tokens)

    pulse = asyncio.create_task(keep_restricted_selector_moving())
    try:
        await asyncio.gather(
            observe("private", "dm", "session-private"),
            observe("forum", "supergroup", "session-forum"),
        )
    finally:
        runner._shutdown_executor()
        pulse.cancel()
        with suppress(asyncio.CancelledError):
            await pulse

    assert observations == {
        "private": ("dm", "session-private", ("dm", "session-private")),
        "forum": (
            "supergroup",
            "session-forum",
            ("supergroup", "session-forum"),
        ),
    }
