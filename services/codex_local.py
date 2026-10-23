"""Shared local Codex runner; prompts travel through stdin in an isolated directory."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def codex_path():
    """Local executable only; cloud sessions never offer Codex."""
    if os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true":
        return None
    candidates = [os.environ.get("CODEX_PATH"), shutil.which("codex"),
                  str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/OpenAI/Codex/bin/codex.exe")]
    return next((p for p in candidates if p and Path(p).is_file()), None)


def run_codex(prompt, timeout=300) -> str:
    executable = codex_path()
    if not executable:
        raise RuntimeError("Codex unavailable: local executable missing or remote mode enabled")
    instruction = "Answer directly. Do not run any command or read any file; everything you need is below."
    with tempfile.TemporaryDirectory(prefix="er_codex_") as workdir:
        output = Path(workdir) / "summary.md"
        try:
            result = subprocess.run([executable, "exec", "-s", "read-only", "--skip-git-repo-check",
                                     "-C", workdir, "-c", 'web_search="disabled"',
                                     "-c", "features.shell_tool=false", "-o", str(output), "-"],
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
