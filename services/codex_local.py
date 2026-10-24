"""Shared local Codex runner; prompts travel through stdin in an isolated directory."""
import os
import json
import tomllib
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_EFFORT = "high"


def codex_models() -> list[str]:
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    models = [DEFAULT_MODEL]
    for filename, loader in (("config.toml", tomllib.loads), ("models_cache.json", json.loads)):
        try:
            data = loader((home / filename).read_text(encoding="utf-8"))
            candidates = ([data.get("model")] if filename == "config.toml" else
                          [m.get("slug") for m in data.get("models", [])
                           if isinstance(m, dict) and m.get("visibility") == "list"])
            models.extend(m for m in candidates if isinstance(m, str) and m and m not in models)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return models


def codex_path():
    """Local executable only; cloud sessions never offer Codex."""
    if os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true":
        return None
    candidates = [os.environ.get("CODEX_PATH"), shutil.which("codex"),
                  str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/OpenAI/Codex/bin/codex.exe")]
    return next((p for p in candidates if p and Path(p).is_file()), None)


def run_codex(prompt, timeout=300, model=None, effort=None) -> str:
    model = DEFAULT_MODEL if model is None else model
    effort = DEFAULT_EFFORT if effort is None else effort
    if model not in codex_models():
        raise ValueError("Unknown Codex model")
    if effort not in ("low", "medium", "high"):
        raise ValueError("Unsupported Codex reasoning effort")
    executable = codex_path()
    if not executable:
        raise RuntimeError("Codex unavailable: local executable missing or remote mode enabled")
    instruction = "Answer directly. Do not run any command or read any file; everything you need is below."
    with tempfile.TemporaryDirectory(prefix="er_codex_") as workdir:
        output = Path(workdir) / "summary.md"
        try:
            result = subprocess.run([executable, "exec", "-s", "read-only", "--skip-git-repo-check",
                                     "-C", workdir, "-c", 'web_search="disabled"',
                                     "-c", "features.shell_tool=false", "-m", model,
                                     "-c", f'model_reasoning_effort="{effort}"', "-o", str(output), "-"],
                                    input=instruction + "\n" + prompt, text=True, encoding="utf-8",
                                    capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Codex timed out after {timeout} seconds") from exc
        except OSError as exc:
            raise RuntimeError(f"Codex could not start: {exc}") from exc
        if result.returncode:
            raise RuntimeError(f"Codex failed with exit code {result.returncode}")
        note = output.read_text(encoding="utf-8").strip() if output.is_file() else ""
        if not note:
            raise RuntimeError("Codex returned an empty summary file")
        return note
