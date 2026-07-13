from pathlib import Path
import subprocess


SKILL_ROOT = Path(".agents/skills/asr-production-delivery")


def test_project_asr_skill_tracks_the_model_runtime_incident_contract():
    skill = (SKILL_ROOT / "SKILL.md").read_text()
    contract = (SKILL_ROOT / "references/project-contract.md").read_text()
    gates = (SKILL_ROOT / "references/acceptance-gates.md").read_text()

    assert "Read `references/project-contract.md` for every task" in skill
    assert "Qwen/Qwen3-ASR-1.7B`" in contract
    assert "Qwen/Qwen3-ASR-1.7B-hf`" in contract
    assert "Qwen2TokenizerFast has no attribute audio_token" in contract
    assert "manifest verification proves" in contract
    assert "R08 real model warmup remains" in contract
    assert "invariant 13" in gates
    assert "manifest hash check is not model/runtime compatibility evidence" in gates

    tracked = set(
        subprocess.run(
            ["git", "ls-files", "--", str(SKILL_ROOT)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    expected = {
        str(SKILL_ROOT / "SKILL.md"),
        str(SKILL_ROOT / "agents/openai.yaml"),
        str(SKILL_ROOT / "references/acceptance-gates.md"),
        str(SKILL_ROOT / "references/agent-handoffs.md"),
        str(SKILL_ROOT / "references/project-contract.md"),
    }
    assert expected <= tracked
