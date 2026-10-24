"""Local theme-family switch: valid preview themes, every family option set or reset, hidden in remote mode."""
import tomllib

import pytest
from streamlit import config

from ui import theme_switch as ts


@pytest.mark.parametrize("name", ["iOS", "One UI"])
def test_preview_themes_use_only_known_options(name):
    known = set(config.get_config_options())
    opts = ts.theme_options(ts.THEMES[name])
    assert opts and set(opts) <= known
    assert {"theme.light.primaryColor", "theme.dark.primaryColor"} <= set(opts)


def test_apply_sets_the_family_and_unsets_options_it_lacks(monkeypatch):
    calls = {}
    monkeypatch.setattr(config, "set_option", lambda k, v, where_defined=None: calls.__setitem__(k, v))
    monkeypatch.setitem(ts._applied, "name", "Theme A")
    ts.apply("One UI")
    one = ts.theme_options(ts.THEMES["One UI"])
    assert all(calls[k] == v for k, v in one.items())
    assert calls["theme.font"] == one["theme.font"]
    ts.apply("Theme A")
    a = ts.theme_options(ts.THEMES["Theme A"])
    assert calls["theme.baseRadius"] is None and calls["theme.light.primaryColor"] == a["theme.light.primaryColor"]
    assert ts._applied["name"] == "Theme A"


def test_switch_is_hidden_in_remote_mode(monkeypatch):
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "true")
    assert ts.available() is False
    monkeypatch.setenv("SUPABASE_REMOTE_MODE", "false")
    assert ts.available() is True


def test_active_config_is_unchanged_theme_a():
    with open(ts.THEMES["Theme A"], "rb") as f:
        assert tomllib.load(f)["theme"]["light"]["primaryColor"] == "#1D4ED8"
