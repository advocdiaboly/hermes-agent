from pathlib import Path


SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"


def test_telegram_topic_lifecycle_skill_documents_safe_workflow_and_exclusions():
    description = (SKILLS_ROOT / "messaging" / "DESCRIPTION.md").read_text()
    skill = (
        SKILLS_ROOT / "messaging" / "telegram-topic-lifecycle" / "SKILL.md"
    ).read_text()
    normalized = skill.lower()

    assert "telegram-topic-lifecycle" in description
    for stage in ("status", "proposal", "approval", "mutation"):
        assert stage in normalized
    for concept in (
        "native routing",
        "profile isolation",
        "routing override",
        "binding mutation",
        "global bot token",
        "bind_session",
    ):
        assert concept in normalized
    assert "pinned" not in normalized
