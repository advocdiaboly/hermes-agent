from toolsets import TOOLSETS, resolve_toolset
from tools.registry import discover_builtin_tools, registry


_TOOLS = {"telegram_topic_status", "telegram_topic_manage"}


def test_registry_assigns_lifecycle_tools_only_to_dedicated_toolset():
    discover_builtin_tools()

    mapping = registry.get_tool_to_toolset_map()
    assert {name: mapping[name] for name in _TOOLS} == {
        "telegram_topic_status": "telegram_topic_lifecycle",
        "telegram_topic_manage": "telegram_topic_lifecycle",
    }


def test_only_telegram_platform_bundle_exposes_lifecycle_tools():
    assert TOOLSETS["telegram_topic_lifecycle"]["tools"] == sorted(_TOOLS)
    assert "telegram_topic_lifecycle" in TOOLSETS["hermes-telegram"]["includes"]

    assert _TOOLS <= set(resolve_toolset("hermes-telegram"))
    intentional_union_composites = {"hermes-gateway"}
    derived_union_composites = {
        name
        for name, definition in TOOLSETS.items()
        if name.startswith("hermes-")
        and any(include.startswith("hermes-") for include in definition["includes"])
    }
    assert derived_union_composites == intentional_union_composites
    other_platforms = {
        name
        for name in TOOLSETS
        if name.startswith("hermes-")
    } - {"hermes-telegram"} - intentional_union_composites
    assert other_platforms
    for platform in sorted(other_platforms):
        assert _TOOLS.isdisjoint(resolve_toolset(platform)), platform
