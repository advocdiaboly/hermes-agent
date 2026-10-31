"""Shared wording for the exec-approval prompt every messaging surface renders.

The button card (``BasePlatformAdapter._format_exec_approval``) and the plain-text
``/approve`` fallback (``gateway.run._format_exec_approval_fallback``) must tell the user
the same three things: what Hermes wants to run, why it was flagged, and that silence means
the command does NOT run once ``approvals.timeout`` elapses. Keeping the text here means one
edit changes every platform; adapters only wrap these strings in their own markup.

The ``EA_*`` / ``APPROVAL_TIMED_OUT_NOTICE`` module constants are the English source of truth
(catalog keys ``gateway.exec_approval.*``); anything that renders the prompt calls the sibling
accessors (``ea_header_text()`` …) so the active ``display.language`` is honoured at render time.

No imports from ``gateway.platforms.base`` or ``gateway.run`` — both import this module.
"""

from __future__ import annotations

from typing import Any, Callable

from agent.i18n import t

# Bare strings; adapters add their own bold/HTML around them.
EA_HEADER_TEXT = "Hermes wants to run a command that needs your OK"
EA_REASON_LABEL_TEXT = "Why it was flagged"

# Timeout notice posted when nobody answered the prompt (``{window}`` = "5 minutes").
APPROVAL_TIMED_OUT_NOTICE = (
    "⌛ Approval timed out after {window} — the command was NOT run. "
    "Ask me to try again if you still want it, or raise approvals.timeout in config.yaml.")


def ea_header_text(lang: str | None = None) -> str:
    """Localized ``EA_HEADER_TEXT``."""
    return t("gateway.exec_approval.header", lang=lang)


def ea_reason_label_text() -> str:
    """Localized ``EA_REASON_LABEL_TEXT``."""
    return t("gateway.exec_approval.reason_label")


def ea_smart_deny_line_text(lang: str | None = None) -> str:
    """Localized smart-deny sentence (bare; adapters add their own line breaks/markup)."""
    return t("gateway.exec_approval.smart_deny_line", lang=lang)


def ea_default_reason_text() -> str:
    """Localized default "why it was flagged" reason when the approval carries none."""
    return t("gateway.exec_approval.default_reason")


def ea_action_labels(lang: str | None = None) -> dict[str, str]:
    """Localized ``choice -> button label`` for the four approval tiers."""
    return {
        "once": t("gateway.exec_approval.action_once", lang=lang),
        "session": t("gateway.exec_approval.action_session", lang=lang),
        "always": t("gateway.exec_approval.action_always", lang=lang),
        "deny": t("gateway.exec_approval.action_deny", lang=lang),
    }


def approval_timed_out_notice(window: str) -> str:
    """Localized ``APPROVAL_TIMED_OUT_NOTICE`` for an already-formatted ``window``."""
    return t("gateway.exec_approval.timed_out_notice", window=window)


def approval_timeout_seconds() -> int:
    """The configured ``approvals.timeout`` (default 300s); module attribute so tests can pin it."""
    from tools.approval_context import _get_approval_timeout
    return _get_approval_timeout()


def format_approval_window(seconds: int, *, lang: str | None = None) -> str:
    """Human wording for a timeout (300 → "5 minutes"); one formatter shared with the CLI notice and
    the tool result's ``user_summary`` — see ``tools.approval_context.format_approval_window``."""
    from tools.approval_context import format_approval_window as _shared
    return _shared(seconds, lang=lang)


def format_approval_deadline_line(timeout_s: int, *, lang: str | None = None) -> str:
    """The last line of every approval prompt: doing nothing is a safe no."""
    return t("gateway.exec_approval.deadline", lang=lang, window=format_approval_window(timeout_s, lang=lang))


def format_approval_timed_out_notice(timeout_s: int) -> str:
    return approval_timed_out_notice(format_approval_window(timeout_s))


def _fit_escaped_summary(text: str, budget: int, escape: Callable[[str], str]) -> str:
    """Fit before escaping, so HTML entities are never cut in half."""
    text = text[:max(0, budget + 1)]
    if len(escape(text)) <= budget:
        return escape(text)
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if len(escape(text[:mid])) <= max(0, budget - 1):
            low = mid
        else:
            high = mid - 1
    return escape(text[:low]) + "…"


def render_exec_approval(adapter: Any, command: str, description: str | None,
                         smart_denied: bool, *, summary: str | None = None,
                         language: str | None = None) -> str:
    """One rendering contract; presentation never substitutes the execution source."""
    if summary is not None:
        header = f"⚠️ {adapter._ea_escape(ea_header_text(language))}\n\n"
        tail = "\n\n" + adapter._ea_escape(format_approval_deadline_line(
            approval_timeout_seconds(), lang=language))
        if smart_denied:
            tail += "\n\n" + adapter._ea_escape(ea_smart_deny_line_text(language))
        budget = max(0, getattr(adapter, "MAX_MESSAGE_LENGTH", 4096) - len(header) - len(tail))
        return header + _fit_escaped_summary(summary, budget, adapter._ea_escape) + tail
    if description is None:
        description = ea_default_reason_text()
    if adapter._EA_REASON_BUDGET:
        description = adapter._ea_fit(str(description or ""), adapter._EA_REASON_BUDGET)
    preview = adapter._ea_fit(str(command or ""), adapter._exec_approval_cmd_budget(description, smart_denied))
    text = (f"{adapter._EA_HEADER}{adapter._EA_CODE_OPEN}{adapter._ea_escape(preview)}{adapter._EA_CODE_CLOSE}"
            f"{adapter._EA_REASON_LABEL}{adapter._ea_escape(description)}{adapter._ea_deadline_line()}")
    return text + adapter._EA_SMART_DENY_LINE if smart_denied else text


def prepare_exec_approval_display(data: dict, redact: Callable[[str], str], adapter: Any) -> tuple:
    """Keep command identity separate and pass new kwargs only to supporting adapters."""
    from agent.interrupt_compat import _accepts_keyword
    command = redact(data.get("command", ""))
    description = data.get("description") or ea_default_reason_text()
    flags = {k: data.get(k, default) for k, default in
             (("allow_permanent", True), ("allow_session", True), ("smart_denied", False))}
    presentation = {}
    send = getattr(adapter, "send_exec_approval", None)
    if isinstance(data.get("summary"), str) and send is not None:
        for key in ("summary", "summary_language"):
            value = data.get(key)
            if isinstance(value, str) and _accepts_keyword(send, key):
                presentation[key] = redact(value) if key == "summary" else value
    return command, description, flags, presentation
