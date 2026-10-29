"""Run Antigravity with an isolated workspace and without app credentials."""
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

DEFAULT_MODEL = "gemini-4-argon-high"
DEFAULT_EFFORT = "high"
_FALLBACK_MODELS = (DEFAULT_MODEL, "gemini-4-argon-medium", "gemini-4-argon-low")
logger = logging.getLogger(__name__)


def agy_path():
    """Cloud sessions must not offer an executable from the server."""
    if os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true":
        return None
    candidates = [shutil.which("agy"), str(Path(os.environ.get("LOCALAPPDATA", "")) / "agy/bin/agy.exe")]
    return next((p for p in candidates if p and Path(p).is_file()), None)


def _environment():
    # Keep OS and CLI configuration needed for sign-in, but never forward app secrets.
    allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "HOME",
               "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "LANG", "LC_ALL"}
    return {k: v for k, v in os.environ.items() if k.upper() in allowed}


@lru_cache(maxsize=1)
def agy_models():
    executable = agy_path()
    if executable:
        try:
            with tempfile.TemporaryDirectory(prefix="agy_models_") as workdir:
                result = subprocess.run([executable, "models"], cwd=workdir, env=_environment(),
                                        capture_output=True, text=True, encoding="utf-8", timeout=30)
            if result.returncode == 0:
                models = re.findall(r"\b(?:gemini|claude|gpt)[a-zA-Z0-9._-]*[a-zA-Z0-9]\b", result.stdout)
                if models:
                    return list(dict.fromkeys([DEFAULT_MODEL, *models]))
        except (OSError, subprocess.TimeoutExpired):
            pass
    return list(_FALLBACK_MODELS)


def run_agy(prompt, timeout=300, model=None, effort=None, schema=None) -> str:
    from services.auth import require_signed_in
    require_signed_in("run Antigravity")
    executable = agy_path()
    if not executable:
        raise RuntimeError("Antigravity unavailable: local executable missing or remote mode enabled")
    model = model or DEFAULT_MODEL
    effort = effort or DEFAULT_EFFORT
    if model not in agy_models():
        raise ValueError("Unknown Antigravity model")
    if effort not in ("low", "medium", "high"):
        raise ValueError("Unsupported Antigravity reasoning effort")
    instruction = "Answer directly. Do not run commands, and do not read or write files. Everything you need is below."
    # The prompt goes in on stdin as one stream-json message: a release can be 30-40k characters, beyond the Windows
    # command-line limit (WinError 206) for "-p <prompt>". "-p=" means "read the turns from stdin".
    message = json.dumps({"event": "user", "message": {"role": "user", "content": instruction + "\n" + prompt}},
                         ensure_ascii=False)
    with tempfile.TemporaryDirectory(prefix="agy_workspace_") as workdir, tempfile.TemporaryDirectory(prefix="agy_schema_") as schema_dir:
        command = [executable, "-p=", "--input-format", "stream-json", "--mode", "plan", "--sandbox",
                   "--model", model, "--effort", effort, "--output-format", "stream-json",
                   "--print-timeout", f"{timeout}s"]
        if schema is not None:
            schema_path = Path(schema_dir) / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            command.extend(["--json-schema", str(schema_path)])
        try:
            result = subprocess.run(command, cwd=workdir, env=_environment(), text=True, encoding="utf-8",
                                    input=message + "\n", capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Antigravity timed out after {timeout} seconds") from exc
        except OSError as exc:
            raise RuntimeError(f"Antigravity could not start: {exc}") from exc
    if result.returncode:
        # agy reports the reason in its result event (quota, model, permission): show it, not only the exit code
        reason = None
        for line in reversed(result.stdout.splitlines()):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("event") == "result":
                reason = (event.get("result") or {}).get("error")
                break
        raise RuntimeError(f"Antigravity failed (exit code {result.returncode})" + (f": {reason}" if reason else ""))
    if not result.stdout.strip():
        raise RuntimeError("Antigravity returned empty output")
    try:
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        for event in events:
            if re.search(r"denied|permission", json.dumps(event), re.IGNORECASE):
                logger.warning("Antigravity reported a denied or permission-related action")
        response = events[-1]["result"]["response"]
        if not isinstance(response, str) or not response.strip():
            raise ValueError("empty response")
        if schema is not None:
            json.loads(response)
        return response.strip()
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise RuntimeError("Antigravity returned invalid JSON or an empty response") from exc
