---
name: telegram-topic-lifecycle
description: Use when managing Telegram private topics, supergroup forum topics, topic titles, or topic/session bindings in Hermes.
---

# Telegram Topic Lifecycle

Use this skill only in Telegram sessions where topic structure, topic naming, or topic/session bindings matter.

## Product Model

- Private-DM topics remain a supported personal-topic surface.
- Supergroup/forum topics are reliable when the operator explicitly chooses a group-based workflow; do not pressure migration between surfaces.
- In supergroup/forum topics, each Telegram topic is its own Hermes session by default.
- A binding row is lifecycle metadata unless `telegram_topic_status.binding_status` is `matches_routing`.
- `topic_mode_enabled=false` only means private-DM topic mode is off; it does not mean supergroup/forum topic support is unavailable.
- Telegram pinned-message memory is not a supported lifecycle primitive. Telegram Bot API and client behavior do not provide a consistent full pin/unpin model for this workflow.

## Core Rule

Discuss lifecycle choices in chat first; call tools only for operations chat cannot perform. Structural changes require explicit operator approval before any mutating tool call.

## When To Suggest A Topic

- Suggest a new topic for long-lived project work, repeated research, incidents, or context that should survive across sessions.
- Reuse an existing topic when the work clearly belongs to that topic's current session.
- Do not create duplicate topics for a naming preference alone.

## Naming

- Prefer Russian user-facing topic names.
- Use prefixes when they clarify intent: `[Проект]`, `[Ad-hoc]`, `[Исследование]`, `[Hermes]`.
- Keep established technical names in English when translating would reduce clarity.

## Tool Use

1. Call `telegram_topic_status` before proposing a structural action.
2. Read `surface`, `routing`, `binding_status`, and `binding_authority` before interpreting state.
3. Propose the exact title and operation in chat.
4. Wait for explicit operator approval.
5. For topic operations, call `telegram_topic_manage` with approval metadata.

`telegram_topic_manage(action=create_topic)` creates the Telegram topic and returns its `thread_id`; it does not bind or move the current Hermes session unless a separate approved operation explicitly reports that routing changed.

`telegram_topic_manage(action=bind_session)` in supergroup/forum topics records lifecycle metadata only unless the tool reports `routing_effect` other than `metadata_only`.

Approval metadata must include `approved_by` and `approval_text_excerpt`; include `approval_ref` when a message id or other reference is available.

## Do Not

- Do not silently create or rename Telegram topics.
- Do not claim Telegram operations succeeded unless the tool reports success.
- Do not treat a metadata-only binding mismatch as active routing.
- Do not recommend switching between private-DM topics and supergroup/forum topics unless the operator is explicitly discussing that change.
- Do not expose broad Telegram group administration through this workflow.
- Do not create, pin, unpin, track, or promise Telegram pinned messages as lifecycle memory.
