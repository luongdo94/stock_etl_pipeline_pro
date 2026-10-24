"""Theme family switch (Theme A / iOS / One UI) for the app run on this computer only.

Streamlit reads the theme from the config when it builds each run's new-session message, so setting the `theme.*`
options and rerunning repaints the page. The options are SERVER-wide: on a shared deployment one user's click would
re-theme everybody, so the switch is offered only outside remote mode. Light / Dark stays the per-viewer choice in the
app menu (⋮ → Settings). A restart returns to .streamlit/config.toml.
"""
import os
import tomllib
from pathlib import Path

import streamlit as st
from streamlit import config

ROOT = Path(__file__).resolve().parent.parent
THEMES = {"Theme A": ROOT / ".streamlit" / "config.toml",
          "iOS": ROOT / ".streamlit" / "themes" / "ios.toml",
          "One UI": ROOT / ".streamlit" / "themes" / "oneui.toml"}
_applied = {"name": "Theme A"}            # server-wide, like the options themselves


def theme_options(path: Path) -> dict:
    """{"theme.x": v, "theme.light.x": v, …} from a config file's [theme] tables."""
    with open(path, "rb") as f:
        table = tomllib.load(f).get("theme", {})
    out = {}
    for key, value in table.items():
        if isinstance(value, dict):
            out.update({f"theme.{key}.{k}": v for k, v in value.items() if not isinstance(v, dict)})
        else:
            out[f"theme.{key}"] = value
    return out


def available() -> bool:
    return (os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() != "true"
            and sum(p.is_file() for p in THEMES.values()) > 1)


def apply(name: str) -> None:
    """Set every theme option any family uses: the chosen family's value, or unset (Streamlit's default)."""
    chosen = theme_options(THEMES[name])
    keys = set().union(*(theme_options(p) for p in THEMES.values() if p.is_file()))
    for key in keys:
        config.set_option(key, chosen.get(key), where_defined="theme switch")
    _applied["name"] = name


def render() -> None:
    st.sidebar.caption("Light / Dark: app menu ⋮ → Settings.")
    if not available():
        return
    names = [n for n, p in THEMES.items() if p.is_file()]
    choice = st.sidebar.selectbox("Theme (this computer only)", names, index=names.index(_applied["name"]),
                                  key="ui_theme_family",
                                  help="Changes the colours for everyone using this local app until it restarts.")
    if choice != _applied["name"]:
        apply(choice)
        st.rerun()
