"""Declared plans are immutable request data and redacted before transport use."""
from uuid import UUID

from hermes_cli.approval_transport import ApprovalRequest, invoke_approval_transport


def _request(**extra):
    return ApprovalRequest.create(command="execute_code body", description="code",
        pattern_key="execute_code", pattern_keys=("execute_code",), session_key="s",
        surface="gateway", allow_session=False, allow_permanent=False, **extra)


def test_reason_is_bound_independently_of_the_random_request_id(monkeypatch):
    monkeypatch.setattr("hermes_cli.approval_transport.uuid.uuid4", lambda: UUID(int=1))
    a = _request(approval_reason="Create proposals only")
    b = _request(approval_reason="Overwrite primary documents")
    assert a.command == b.command and a.request_id == b.request_id
    assert a.digest != b.digest
    outcome = invoke_approval_transport(lambda req: b.respond("once"), a, timeout_seconds=1)
    assert outcome.choice == "deny" and outcome.failure is not None


def test_legacy_request_without_reason_keeps_its_original_decision_contract():
    request = _request()
    assert request.approval_reason is None
    result = invoke_approval_transport(lambda req: req.respond("once"), request, timeout_seconds=1)
    assert result.choice == "once" and result.failure is None


def test_transport_receives_redacted_reason_and_unchanged_source(monkeypatch):
    from types import SimpleNamespace
    from tools import approval_context, approval_prompt
    seen = []
    def present(request):
        seen.append(request)
        return request.respond("deny")
    monkeypatch.setattr(approval_context, "_get_approval_transport_config", lambda: ("fixture", None))
    manager = SimpleNamespace(get_approval_transport=lambda name: SimpleNamespace(present=present))
    monkeypatch.setattr(approval_prompt, "get_plugin_manager", lambda: manager)
    secret = "sk-proj-" + "Abc123" * 20
    result = approval_prompt._present_with_selected_transport(command="print('unchanged')", description="code",
        pattern_key="execute_code", pattern_keys=["execute_code"], session_key="s", surface="gateway",
        allow_session=False, allow_permanent=False,
        approval_reason="Inspect https://example.invalid/?access_token=private-value-123456789 " + secret)
    assert result["choice"] == "deny" and len(seen) == 1
    assert seen[0].command == "print('unchanged')"
    assert secret not in seen[0].approval_reason
    assert "private-value-123456789" not in seen[0].approval_reason


def test_reason_is_bounded_and_neutralizes_controls_and_typed_commands():
    from tools.approval_prompt import prepare_approval_reason
    text = "Plan\u202e\x00 /approve /deny /yolo /stop\n" + "x" * 2000
    cleaned = prepare_approval_reason(text)
    assert len(cleaned) <= 1001 and cleaned.endswith("…")
    assert "\u202e" not in cleaned and "\x00" not in cleaned
    assert all(cmd not in cleaned for cmd in ("/approve", "/deny", "/yolo", "/stop"))
    assert "\n" in cleaned
