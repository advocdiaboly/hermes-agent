"""Security-floor tests for the Google Workspace runtime installer."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest


SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


@pytest.fixture()
def setup_module():
    spec = importlib.util.spec_from_file_location(
        "test_google_workspace_setup_module",
        SETUP_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stale_google_transitives_are_reported_missing(setup_module, monkeypatch):
    installed = {
        "google-api-python-client": "2.194.0",
        "google-auth": "2.55.0",
        "google-auth-oauthlib": "1.3.1",
        "google-auth-httplib2": "0.3.1",
        "httplib2": "0.31.2",
        "pyasn1": "0.6.3",
    }

    def fake_version(name):
        try:
            return installed[name]
        except KeyError:
            raise PackageNotFoundError(name) from None

    monkeypatch.setattr(setup_module, "_distribution_version", fake_version)

    assert setup_module._missing_required_packages() == [
        "google-auth==2.55.1",
        "httplib2==0.32.0",
        "pyasn1==0.6.4",
    ]


def test_installer_repairs_stale_transitives(setup_module, monkeypatch):
    states = iter(
        [
            [
                "google-auth==2.55.1",
                "httplib2==0.32.0",
                "pyasn1==0.6.4",
            ],
            [],
        ]
    )
    monkeypatch.setattr(
        setup_module,
        "_missing_required_packages",
        lambda: next(states),
    )
    calls = []
    monkeypatch.setattr(
        setup_module.subprocess,
        "check_call",
        lambda argv, **kwargs: calls.append(argv),
    )

    assert setup_module.install_deps() is True
    assert calls == [
        [
            setup_module.sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "google-auth==2.55.1",
            "httplib2==0.32.0",
            "pyasn1==0.6.4",
        ]
    ]


def test_resolve_scopes_limits_oauth_to_requested_services(setup_module):
    scopes, services = setup_module._resolve_scopes("email,tasks")

    assert services == ["email", "tasks"]
    assert scopes == [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/tasks",
    ]


def test_expected_scopes_follow_services_stored_with_token(setup_module):
    assert setup_module._expected_scopes_from_payload({"services": ["tasks"]}) == [
        "https://www.googleapis.com/auth/tasks"
    ]


def test_auth_url_json_persists_requested_scopes(
    setup_module, monkeypatch, tmp_path, capsys
):
    client_secret = tmp_path / "client.json"
    pending = tmp_path / "pending.json"
    last_url = tmp_path / "last-url.txt"
    client_secret.write_text("{}")
    monkeypatch.setattr(setup_module, "CLIENT_SECRET_PATH", client_secret)
    monkeypatch.setattr(setup_module, "PENDING_AUTH_PATH", pending)
    monkeypatch.setattr(setup_module, "LAST_AUTH_URL_PATH", last_url, raising=False)
    monkeypatch.setattr(setup_module, "_ensure_deps", lambda: None)

    class FakeFlow:
        code_verifier = "verifier"

        @classmethod
        def from_client_secrets_file(cls, path, **kwargs):
            assert path == str(client_secret)
            assert kwargs["scopes"] == ["https://www.googleapis.com/auth/tasks"]
            return cls()

        def authorization_url(self, **kwargs):
            assert kwargs == {"access_type": "offline", "prompt": "consent"}
            return "https://example.test/auth", "state"

    oauthlib = types.ModuleType("google_auth_oauthlib")
    flow_module = types.ModuleType("google_auth_oauthlib.flow")
    flow_module.Flow = FakeFlow
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", oauthlib)
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_module)

    setup_module.get_auth_url("tasks", "json")

    output = json.loads(capsys.readouterr().out)
    assert output["auth_url"] == "https://example.test/auth"
    assert output["services"] == ["tasks"]
    assert output["scopes"] == ["https://www.googleapis.com/auth/tasks"]
    assert json.loads(pending.read_text())["requested_scopes"] == output["scopes"]
    assert last_url.read_text() == output["auth_url"]


def test_main_forwards_services_and_format_to_auth_url(setup_module, monkeypatch):
    received = {}

    def capture(services, output_format):
        received.update(services=services, output_format=output_format)

    monkeypatch.setattr(setup_module, "get_auth_url", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        ["setup.py", "--auth-url", "--services", "tasks", "--format", "json"],
    )

    setup_module.main()

    assert received == {"services": "tasks", "output_format": "json"}


def test_auth_code_uses_pending_scopes_and_emits_json(
    setup_module, monkeypatch, tmp_path, capsys
):
    client_secret = tmp_path / "client.json"
    pending = tmp_path / "pending.json"
    token = tmp_path / "token.json"
    client_secret.write_text("{}")
    pending.write_text(json.dumps({
        "state": "state",
        "code_verifier": "verifier",
        "redirect_uri": setup_module.REDIRECT_URI,
        "requested_scopes": ["https://www.googleapis.com/auth/tasks"],
        "services": ["tasks"],
    }))
    monkeypatch.setattr(setup_module, "CLIENT_SECRET_PATH", client_secret)
    monkeypatch.setattr(setup_module, "PENDING_AUTH_PATH", pending)
    monkeypatch.setattr(setup_module, "TOKEN_PATH", token)
    monkeypatch.setattr(setup_module, "_ensure_deps", lambda: None)

    class FakeCredentials:
        granted_scopes = ["https://www.googleapis.com/auth/tasks"]

        def to_json(self):
            return json.dumps({
                "token": "access-token",
                "refresh_token": "refresh-token",
                "client_id": "client-id",
                "client_secret": "client-secret",
                "token_uri": "https://oauth2.googleapis.com/token",
            })

    class FakeFlow:
        credentials = FakeCredentials()

        @classmethod
        def from_client_secrets_file(cls, path, **kwargs):
            assert path == str(client_secret)
            assert kwargs["scopes"] == ["https://www.googleapis.com/auth/tasks"]
            assert kwargs["state"] == "state"
            assert kwargs["code_verifier"] == "verifier"
            return cls()

        def fetch_token(self, **kwargs):
            assert kwargs == {"code": "auth-code"}

    oauthlib = types.ModuleType("google_auth_oauthlib")
    flow_module = types.ModuleType("google_auth_oauthlib.flow")
    flow_module.Flow = FakeFlow
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", oauthlib)
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_module)

    setup_module.exchange_auth_code("auth-code", "json")

    output = json.loads(capsys.readouterr().out)
    saved = json.loads(token.read_text())
    assert output["status"] == "authenticated"
    assert output["services"] == ["tasks"]
    assert saved["services"] == ["tasks"]
    assert saved["scopes"] == ["https://www.googleapis.com/auth/tasks"]
    assert not pending.exists()


def test_failed_auth_code_json_returns_fresh_auth_url(
    setup_module, monkeypatch, tmp_path, capsys
):
    client_secret = tmp_path / "client.json"
    pending = tmp_path / "pending.json"
    last_url = tmp_path / "last-url.txt"
    client_secret.write_text("{}")
    pending.write_text(json.dumps({
        "state": "old-state",
        "code_verifier": "old-verifier",
        "redirect_uri": setup_module.REDIRECT_URI,
        "requested_scopes": ["https://www.googleapis.com/auth/tasks"],
        "services": ["tasks"],
    }))
    monkeypatch.setattr(setup_module, "CLIENT_SECRET_PATH", client_secret)
    monkeypatch.setattr(setup_module, "PENDING_AUTH_PATH", pending)
    monkeypatch.setattr(setup_module, "LAST_AUTH_URL_PATH", last_url)
    monkeypatch.setattr(setup_module, "_ensure_deps", lambda: None)

    class FakeFlow:
        code_verifier = "fresh-verifier"

        @classmethod
        def from_client_secrets_file(cls, path, **kwargs):
            assert path == str(client_secret)
            assert kwargs["scopes"] == ["https://www.googleapis.com/auth/tasks"]
            return cls()

        def fetch_token(self, **kwargs):
            raise RuntimeError("expired code")

        def authorization_url(self, **kwargs):
            return "https://example.test/fresh", "fresh-state"

    oauthlib = types.ModuleType("google_auth_oauthlib")
    flow_module = types.ModuleType("google_auth_oauthlib.flow")
    flow_module.Flow = FakeFlow
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", oauthlib)
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_module)

    with pytest.raises(SystemExit) as exc:
        setup_module.exchange_auth_code("expired", "json")

    assert exc.value.code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["error"] == "token_exchange_failed"
    assert output["fresh_auth_url"] == "https://example.test/fresh"
    assert json.loads(pending.read_text())["state"] == "fresh-state"
    assert last_url.read_text() == output["fresh_auth_url"]
