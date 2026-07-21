import json
from pathlib import Path


def test_dockerfile_runs_two_uvicorn_workers():
    dockerfile = Path(__file__).parents[1] / "Dockerfile"
    cmd_line = next(
        line
        for line in reversed(dockerfile.read_text().splitlines())
        if line.startswith("CMD ")
    )
    command = json.loads(cmd_line.removeprefix("CMD "))

    assert command[:3] == ["python", "-m", "uvicorn"]
    assert command[3] == "app.main:app"
    assert command.count("--workers") == 1
    workers_index = command.index("--workers")
    assert command[workers_index + 1] == "2"
