"""Tests for Google Workspace gws bridge and CLI wrapper."""

import importlib.util
import json
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


BRIDGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/gws_bridge.py"
)
API_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/google_api.py"
)


@pytest.fixture
def bridge_module(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    spec = importlib.util.spec_from_file_location("gws_bridge_test", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def api_module(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    spec = importlib.util.spec_from_file_location("gws_api_test", API_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # Ensure the gws CLI code path is taken even when the binary isn't
    # installed (CI).  Without this, calendar_list() falls through to the
    # Python SDK path which imports ``googleapiclient`` — not in deps.
    module._gws_binary = lambda: "/usr/bin/gws"
    # Bypass authentication check — no real token file in CI.
    module._ensure_authenticated = lambda: None
    return module


def _write_token(path: Path, *, token="ya29.test", expiry=None, **extra):
    data = {
        "token": token,
        "refresh_token": "1//refresh",
        "client_id": "123.apps.googleusercontent.com",
        "client_secret": "secret",
        "token_uri": "https://oauth2.googleapis.com/token",
        **extra,
    }
    if expiry is not None:
        data["expiry"] = expiry
    path.write_text(json.dumps(data))


def test_bridge_returns_valid_token(bridge_module, tmp_path):
    """Non-expired token is returned without refresh."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.valid", expiry=future)

    result = bridge_module.get_valid_token()
    assert result == "ya29.valid"










def test_bridge_main_injects_token_env(bridge_module, tmp_path):
    """main() sets GOOGLE_WORKSPACE_CLI_TOKEN in subprocess env."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.injected", expiry=future)

    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return MagicMock(returncode=0)

    with patch.object(sys, "argv", ["gws_bridge.py", "gmail", "+triage"]):
        with patch.object(subprocess, "run", side_effect=capture_run):
            with pytest.raises(SystemExit):
                bridge_module.main()

    assert captured["env"]["GOOGLE_WORKSPACE_CLI_TOKEN"] == "ya29.injected"
    assert captured["cmd"] == ["gws", "gmail", "+triage"]


def test_api_calendar_list_uses_events_list(api_module):
    """calendar_list calls _run_gws with events list + params."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout="{}", stderr="")

    args = api_module.argparse.Namespace(
        start="", end="", max=25, calendar="primary", func=api_module.calendar_list,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.calendar_list(args)

    cmd = captured["cmd"]
    # _gws_binary() returns "/usr/bin/gws", so cmd[0] is that binary
    assert cmd[0] == "/usr/bin/gws"
    assert "calendar" in cmd
    assert "events" in cmd
    assert "list" in cmd
    assert "--params" in cmd
    params = json.loads(cmd[cmd.index("--params") + 1])
    assert "timeMin" in params
    assert "timeMax" in params
    assert params["calendarId"] == "primary"


def test_tasks_add_uses_python_fallback(api_module, monkeypatch, capsys):
    captured = {}

    class Request:
        def __init__(self, payload):
            self.payload = payload

        def execute(self):
            return self.payload

    class TaskLists:
        def list(self, **kwargs):
            assert kwargs == {"maxResults": 10}
            return Request({"items": [{"id": "list-1", "title": "My Tasks"}]})

    class Tasks:
        def insert(self, **kwargs):
            captured.update(kwargs)
            return Request({
                "id": "task-1",
                "title": kwargs["body"]["title"],
                "due": kwargs["body"]["due"],
                "webViewLink": "https://tasks.google.test/task-1",
            })

    class Service:
        def tasklists(self):
            return TaskLists()

        def tasks(self):
            return Tasks()

    monkeypatch.setattr(api_module, "_gws_binary", lambda: None)
    monkeypatch.setattr(
        api_module,
        "build_service",
        lambda api, version: Service() if (api, version) == ("tasks", "v1") else None,
    )
    args = api_module.argparse.Namespace(
        title="Pay invoice",
        due="2026-08-28",
        tasklist="",
    )

    api_module.tasks_add(args)

    output = json.loads(capsys.readouterr().out)
    assert captured == {
        "tasklist": "list-1",
        "body": {"title": "Pay invoice", "due": "2026-08-28T00:00:00Z"},
    }
    assert output == {
        "status": "created",
        "tasklistId": "list-1",
        "tasklistTitle": "My Tasks",
        "id": "task-1",
        "title": "Pay invoice",
        "due": "2026-08-28T00:00:00Z",
        "webViewLink": "https://tasks.google.test/task-1",
    }


def test_tasks_update_uses_python_fallback(api_module, monkeypatch, capsys):
    captured = {}

    class Request:
        def __init__(self, payload):
            self.payload = payload

        def execute(self):
            return self.payload

    class TaskLists:
        def list(self, **kwargs):
            assert kwargs == {"maxResults": 10}
            return Request({"items": [{"id": "list-1", "title": "My Tasks"}]})

    class Tasks:
        def patch(self, **kwargs):
            captured.update(kwargs)
            return Request({
                "id": kwargs["task"],
                **kwargs["body"],
                "updated": "2026-09-02T10:00:00Z",
                "webViewLink": "https://tasks.google.test/task-1",
            })

    class Service:
        def tasklists(self):
            return TaskLists()

        def tasks(self):
            return Tasks()

    monkeypatch.setattr(api_module, "_gws_binary", lambda: None)
    monkeypatch.setattr(
        api_module,
        "build_service",
        lambda api, version: Service() if (api, version) == ("tasks", "v1") else None,
    )
    args = api_module.argparse.Namespace(
        task_id="task-1",
        tasklist="",
        title="Paid invoice",
        due="2026-09-02",
        status="completed",
        notes="Paid in full",
    )

    api_module.tasks_update(args)

    assert captured == {
        "tasklist": "list-1",
        "task": "task-1",
        "body": {
            "title": "Paid invoice",
            "due": "2026-09-02T00:00:00Z",
            "status": "completed",
            "notes": "Paid in full",
        },
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "updated",
        "tasklistId": "list-1",
        "tasklistTitle": "My Tasks",
        "id": "task-1",
        "title": "Paid invoice",
        "taskStatus": "completed",
        "due": "2026-09-02T00:00:00Z",
        "notes": "Paid in full",
        "updated": "2026-09-02T10:00:00Z",
        "webViewLink": "https://tasks.google.test/task-1",
    }


def test_tasks_update_uses_gws_with_only_supplied_fields(api_module, monkeypatch, capsys):
    calls = []

    def fake_run(parts, *, params=None, body=None):
        calls.append((parts, params, body))
        if parts == ["tasks", "tasklists", "list"]:
            return {"items": [{"id": "list-1", "title": "My Tasks"}]}
        return {
            "id": "task-1",
            "title": "Pay invoice",
            "status": body["status"],
            "notes": body["notes"],
            "updated": "2026-09-02T10:00:00Z",
            "webViewLink": "https://tasks.google.test/task-1",
        }

    monkeypatch.setattr(api_module, "_run_gws", fake_run)
    args = api_module.argparse.Namespace(
        task_id="task-1",
        tasklist="",
        title=None,
        due=None,
        status="completed",
        notes="Paid in full",
    )

    api_module.tasks_update(args)

    assert calls == [
        (["tasks", "tasklists", "list"], {"maxResults": 10}, None),
        (
            ["tasks", "tasks", "patch"],
            {"tasklist": "list-1", "task": "task-1"},
            {"status": "completed", "notes": "Paid in full"},
        ),
    ]
    assert json.loads(capsys.readouterr().out) == {
        "status": "updated",
        "tasklistId": "list-1",
        "tasklistTitle": "My Tasks",
        "id": "task-1",
        "title": "Pay invoice",
        "taskStatus": "completed",
        "due": "",
        "notes": "Paid in full",
        "updated": "2026-09-02T10:00:00Z",
        "webViewLink": "https://tasks.google.test/task-1",
    }


def test_tasks_update_rejects_empty_patch_before_backend(api_module, monkeypatch, capsys):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("backend must not run for an empty update")

    monkeypatch.setattr(api_module, "_run_gws", unexpected_run)
    args = api_module.argparse.Namespace(
        task_id="task-1",
        tasklist="",
        title=None,
        due=None,
        status=None,
        notes=None,
    )

    with pytest.raises(SystemExit) as exc:
        api_module.tasks_update(args)

    assert exc.value.code == 2
    assert capsys.readouterr().err == (
        "At least one of --title, --due, --status, or --notes is required.\n"
    )


def test_tasks_list_uses_gws_and_normalizes_output(api_module, monkeypatch, capsys):
    calls = []

    def fake_run(parts, *, params=None, body=None):
        calls.append((parts, params, body))
        if parts == ["tasks", "tasklists", "list"]:
            return {"items": [{"id": "list-1", "title": "My Tasks"}]}
        return {"items": [{
            "id": "task-1",
            "title": "Pay invoice",
            "status": "needsAction",
            "due": "2026-08-28T00:00:00Z",
            "updated": "2026-08-27T10:00:00Z",
            "webViewLink": "https://tasks.google.test/task-1",
        }]}

    monkeypatch.setattr(api_module, "_run_gws", fake_run)
    args = api_module.argparse.Namespace(tasklist="", max=20, show_completed=False)

    api_module.tasks_list(args)

    assert calls == [
        (["tasks", "tasklists", "list"], {"maxResults": 10}, None),
        (
            ["tasks", "tasks", "list"],
            {"tasklist": "list-1", "maxResults": 20, "showCompleted": False},
            None,
        ),
    ]
    assert json.loads(capsys.readouterr().out) == {
        "tasklistId": "list-1",
        "tasklistTitle": "My Tasks",
        "tasks": [{
            "id": "task-1",
            "title": "Pay invoice",
            "status": "needsAction",
            "due": "2026-08-28T00:00:00Z",
            "updated": "2026-08-27T10:00:00Z",
            "webViewLink": "https://tasks.google.test/task-1",
        }],
    }


def test_tasks_cli_exposes_list_add_and_update_commands(api_module, monkeypatch, capsys):
    result = subprocess.run(
        [sys.executable, str(API_PATH), "tasks", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "list" in result.stdout
    assert "add" in result.stdout
    assert "update" in result.stdout

    monkeypatch.setattr(
        sys,
        "argv",
        [str(API_PATH), "tasks", "update", "--help"],
    )
    with pytest.raises(SystemExit) as exc:
        api_module.main()

    assert exc.value.code == 0
    update_help = capsys.readouterr().out
    assert "--title" in update_help
    assert "--due" in update_help
    assert "--status" in update_help
    assert "--notes" in update_help












def test_api_get_credentials_refresh_persists_authorized_user_type(api_module, monkeypatch):
    token_path = api_module.TOKEN_PATH
    _write_token(token_path, token="ya29.old")

    class FakeCredentials:
        def __init__(self):
            self.expired = True
            self.refresh_token = "1//refresh"
            self.valid = True

        def refresh(self, request):
            self.expired = False

        def to_json(self):
            return json.dumps({
                "token": "ya29.refreshed",
                "refresh_token": "1//refresh",
                "client_id": "123.apps.googleusercontent.com",
                "client_secret": "secret",
                "token_uri": "https://oauth2.googleapis.com/token",
            })

    class FakeCredentialsModule:
        @staticmethod
        def from_authorized_user_file(filename, scopes):
            assert filename == str(token_path)
            assert scopes == api_module.SCOPES
            return FakeCredentials()

    google_module = types.ModuleType("google")
    oauth2_module = types.ModuleType("google.oauth2")
    credentials_module = types.ModuleType("google.oauth2.credentials")
    credentials_module.Credentials = FakeCredentialsModule
    transport_module = types.ModuleType("google.auth.transport")
    requests_module = types.ModuleType("google.auth.transport.requests")
    requests_module.Request = lambda: object()

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth2_module)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", credentials_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport", transport_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", requests_module)

    creds = api_module.get_credentials()

    saved = json.loads(token_path.read_text())
    assert isinstance(creds, FakeCredentials)
    assert saved["token"] == "ya29.refreshed"
    assert saved["type"] == "authorized_user"
