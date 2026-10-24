"""Preserve the operator's verified Sol route without widening other models."""
import pytest
from agent.native_compaction import resolve_native_compaction_capabilities

@pytest.mark.parametrize("model,provider,base_url,expected", [
    ("gpt-6-sol-900k", "openai-codex", "https://chatgpt.com/backend-api/codex", True),
    ("gpt-6-sol", "openai", "https://api.openai.com/v1", True),
    ("gpt-6.1-sol", "openai-codex", "https://chatgpt.com/backend-api/codex", True),
    ("gpt-6.1-sol-900k", "openai-codex", "https://chatgpt.com/backend-api/codex", True),
    ("gpt-6.1-sol", "openai", "https://api.openai.com/v1", True),
    ("gpt-6.1-sol-900k", "openrouter", "https://openrouter.ai/api/v1", False),
    ("gpt-6-sol", "openrouter", "https://openrouter.ai/api/v1", False),
    ("gpt-6-luna-900k", "openai-codex", "https://chatgpt.com/backend-api/codex", False),
])
def test_operator_sol_native_compaction_route(model, provider, base_url, expected):
    assert resolve_native_compaction_capabilities(
        model=model, provider=provider, base_url=base_url,
        is_codex_backend=provider == "openai-codex",
    )["native_compaction"] is expected
