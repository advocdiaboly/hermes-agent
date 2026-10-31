"""Equal source with different declared plans requires separate human decisions."""
import threading
from concurrent.futures import ThreadPoolExecutor

from tools import approval, approval_context
from tools.approval_gateway_wait import _await_gateway_decision


def test_distinct_plans_for_same_code_are_not_coalesced(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 10)
    session = "reason-distinct-plans"
    first = threading.Event()
    both = threading.Event()
    shown = []
    def notify(data):
        shown.append(dict(data))
        first.set()
        if len(shown) == 2:
            both.set()
    base = {"command": "for p in prepared: write_file(p, data[p])", "description": "code",
            "pattern_key": "execute_code", "pattern_keys": ["execute_code"]}
    a = {**base, "approval_reason": "Create proposals only", "summary": "Plan A"}
    b = {**base, "approval_reason": "Overwrite primary documents", "summary": "Plan B"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(_await_gateway_decision, session, notify, a)
        try:
            assert first.wait(2)
            two = pool.submit(_await_gateway_decision, session, notify, b)
            assert both.wait(2), "different plans were merged by source text"
            assert approval.resolve_gateway_approval(session, "deny") == 1
            assert approval.resolve_gateway_approval(session, "once") == 1
        finally:
            while approval.resolve_gateway_approval(session, "deny"):
                pass
        assert one.result(timeout=5)["choice"] == "deny"
        assert two.result(timeout=5)["choice"] == "once"
