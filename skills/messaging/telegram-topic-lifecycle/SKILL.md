---
name: telegram-topic-lifecycle
description: Use when managing Telegram DM topics, topic titles, pinned bootstrap markers, topic/session bindings, or recoverable topic context in Hermes.
---

# Telegram Topic Lifecycle

Use this skill only in Telegram sessions where topic structure, topic naming, pinned markers, or topic/session bindings matter.

## Core Rule

Discuss lifecycle choices in chat first; call tools only for operations chat cannot perform. Structural changes require explicit operator approval before any mutating tool call.

## When To Suggest A Topic

- Suggest a new topic for long-lived project work, repeated research, incidents, or context that should survive across sessions.
- Reuse an existing topic when the work clearly belongs to that topic's pinned marker scope.
- Do not create duplicate topics for a naming preference alone.

## Naming

- Prefer Russian user-facing topic names.
- Use prefixes when they clarify intent: `[Проект]`, `[Ad-hoc]`, `[Исследование]`, `[Hermes]`.
- Keep established technical names in English when translating would reduce clarity.

## Pinned Markers

Pinned Telegram messages are local memory markers visible to both the operator and Hermes. A topic passport is the default bootstrap marker kind, not a special low-level operation.

Default topic passport content:

- purpose;
- scope and non-goals;
- key repos, links, and tasks;
- lifecycle state;
- routing hints for future sessions;
- operator expectations.

Allowed marker kinds include `topic_passport`, `decision`, `context`, `warning`, and `todo`.

## Tool Use

1. Call `telegram_topic_status` before proposing a structural action.
2. Propose the exact title, operation, and marker text in chat.
3. Wait for explicit operator approval.
4. For topic/session operations, call `telegram_topic_manage` with approval metadata.
5. For pinned markers, call `telegram_pin_manage` with approval metadata.
6. Treat `list_known_pins` as a Hermes-maintained index, not the complete Telegram pinned-message list.

Approval metadata must include `approved_by` and `approval_text_excerpt`; include `approval_ref` when a message id or other reference is available.

## Do Not

- Do not silently create or rename Telegram topics.
- Do not claim Telegram operations succeeded unless the tool reports success.
- Do not invent pinned-marker state.
- Do not expose broad Telegram group administration through this workflow.
- Do not ask the operator to manually create or pin bootstrap markers when the approved tool path is available.
