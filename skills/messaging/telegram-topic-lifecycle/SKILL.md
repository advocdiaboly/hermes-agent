---
name: telegram-topic-lifecycle
description: Inspect and manage the current Telegram topic through native gateway routing.
---

# Telegram Topic Lifecycle

Use this workflow for the authoritative current Telegram session:

1. Call `telegram_topic_status` and inspect the current surface and capability.
2. Present a proposal for either `create_topic` or `rename_topic`, including the proposed title.
3. Call `telegram_topic_manage`. It requests a fresh, real approval for that one operation.
4. Treat the returned result as the mutation outcome. A denial, cancellation, timeout, or approval error means no mutation occurred.

Native routing is authoritative: the gateway derives the current chat, topic, session, and profile. Profile isolation is preserved by that context.

Do not request or invent a routing override. This skill excludes binding mutation, global bot tokens, raw Bot API access, and `bind_session`.
