---
title: "Declared execution plans"
description: "Optional approval presentation metadata and its authorization boundary"
---

# Declared execution plans

The `execute_code` model schema requires `approval_reason`: a concise plan in the
user's language. Describe the purpose, concrete intended targets, effects and
known limits using the current task and preparation results. Do not include
secrets or invent a diff. For a stateful cell such as `write_file(path, text)`,
name the prepared destination and intended overwrite behavior in the plan.

The Python API and old registry calls may omit the field. A supplied value must
be a non-empty string. This is presentation metadata, not authorization or proof
that the code matches the description. The existing guardian receives the code,
and all approval modes, offered choices and final host checks still apply.

## Transport contract

`ApprovalRequest.approval_reason` is optional and immutable. The host redacts it,
removes control/format characters and typed approval commands, and limits it to
1,000 characters plus an ellipsis before binding it into the request digest.
Requests without a reason retain their prior digest input shape.

`ApprovalRequest.supports_summary` advertises the additive native presentation
API. A transport may provide `summary` and `summary_language` in gateway approval
data while keeping `command` equal to the original redacted command. The gateway
passes these fields only to adapter callbacks that accept them. Pending requests
with distinct reasons or presentation text are not coalesced solely because the
source and pattern keys match.

`BasePlatformAdapter.send_exec_approval` accepts optional keyword-only `summary`
and `summary_language`. The native prompt renders escaped ordinary text within
the platform budget, localized host deadline and action labels, and the original
choice set. `ExecApprovalPrompt.command` retains the source for internal use.
Without a summary the existing command presentation remains available.

## Local Telegram pilot

The local approval-tldr pilot opts into this API only for Telegram `execute_code`.
A supplied reason becomes an explicitly labeled agent plan without another
summary-model request. Missing plans retain the bounded auxiliary summary path;
older Hermes versions retain the compact legacy presentation. Other plugin
surfaces keep their existing layouts.

Native button-send fallback and timeout handling use the displayed plan instead
of reintroducing the source. These paths retain the host's legacy layout. A
transport-level failure may still invoke the configured builtin transport and
its stock code view. Deployment and real Telegram acceptance are separate from
isolated source/test verification.
