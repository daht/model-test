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

    assert command == [
        "python",
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--workers",
        "2",
    ]
