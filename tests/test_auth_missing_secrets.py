"""Missing secrets must show setup guidance, never crash or bypass authentication."""
import pytest
from streamlit.errors import StreamlitSecretNotFoundError

from services import auth


@pytest.fixture
def missing_secrets(monkeypatch):
    # Model the external secrets boundary: a missing file raises instead of returning an empty dict.
    class MissingSecrets:
        def get(self, key, default=None):
            raise StreamlitSecretNotFoundError("No secrets found.")

    monkeypatch.setattr(auth.st, "secrets", MissingSecrets())
    for name in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_KEY", "SUPABASE_SERVICE_ROLE_KEY",
                 "SUPABASE_SERVICE_KEY", "COOKIE_SECRET"):
        monkeypatch.delenv(name, raising=False)             # .env is loaded on import by other modules
    auth.get_supabase_client.clear()
    yield
    auth.get_supabase_client.clear()


@pytest.mark.parametrize("factory", [auth._new_auth_client, auth.get_supabase_client])
def test_missing_secrets_returns_no_client(missing_secrets, factory):
    assert factory() is None


def test_missing_cookie_secret_rejects_session(missing_secrets):
    assert auth._sign_session("u", "u@example.com") is None
    assert auth._verify_session("forged.signature") is None


def test_missing_secrets_explain_sign_in_without_stopping_the_app(missing_secrets, monkeypatch):
    """Browsing needs no account: a missing configuration is explained in the sign-in dialog, the page keeps running."""
    messages = []
    monkeypatch.setattr(auth.st, "error", messages.append)
    monkeypatch.setattr(auth.st, "stop", lambda: pytest.fail("the app must keep running for anonymous browsing"))
    auth._login_form(None)
    assert messages and "SUPABASE" in messages[0] and "secrets.toml" in messages[0]


def test_missing_secrets_does_not_enable_local_mode(missing_secrets, monkeypatch):
    monkeypatch.delenv("LOCAL_DEV_MODE", raising=False)
    monkeypatch.setattr(auth.st, "get_option", lambda _: "127.0.0.1")
    assert not auth.local_dev_mode()


def test_login_client_never_uses_service_role_key(monkeypatch):
    monkeypatch.setattr(auth.st, "secrets", {
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "service-only",
    })
    assert auth._new_auth_client() is None


@pytest.mark.parametrize("factory,key", [
    (auth._new_auth_client, "anon-key"),
    (auth.get_supabase_client, "service-key"),
])
def test_configured_clients_keep_their_key_roles(monkeypatch, factory, key):
    monkeypatch.setattr(auth.st, "secrets", {
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_ANON_KEY": "anon-key",
        "SUPABASE_SERVICE_ROLE_KEY": "service-key",
    })
    calls = []
    client = object()

    def create(url, token):
        calls.append((url, token))
        return client

    monkeypatch.setattr(auth, "create_client", create)
    auth.get_supabase_client.clear()
    try:
        assert factory() is client
        assert calls == [("https://example.supabase.co", key)]
    finally:
        auth.get_supabase_client.clear()


def test_environment_variables_configure_auth_when_secrets_lack_them(missing_secrets, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    assert auth._secret("SUPABASE_URL") == "https://example.supabase.co" and auth._secret("SUPABASE_ANON_KEY") == "anon"


@pytest.mark.anonymous
@pytest.mark.parametrize("state,allowed", [({"authenticated": True, "user_id": "u1"}, True),
                                           ({"authenticated": False, "user_id": None}, False),
                                           ({"authenticated": True, "user_id": None}, False)])
def test_can_use_asks_for_sign_in_only_when_there_is_no_account(monkeypatch, state, allowed):
    """Saving and the AI features need an account; the notice names what needs it, nothing is stopped."""
    shown = []

    class Col:
        def info(self, text):
            shown.append(text)

        def button(self, *a, **k):
            return False

    monkeypatch.setattr(auth.st, "session_state", dict(state))
    monkeypatch.setattr(auth.st, "columns", lambda spec: (Col(), Col()))
    monkeypatch.setattr(auth.st, "stop", lambda: pytest.fail("nothing is stopped"))
    assert auth.can_use("keep a watchlist", "wl") is allowed
    assert (shown == []) is allowed and all("Sign in to keep a watchlist" in s for s in shown)


@pytest.mark.anonymous
def test_the_paid_entry_points_refuse_a_call_without_an_account(monkeypatch):
    """A-05: the pages ask for an account; a call that still arrives anonymously is refused before any model is used."""
    from services import assistant, codex_local, earnings, research
    monkeypatch.setattr(auth.st, "session_state", {"authenticated": False, "user_id": None})
    calls = [lambda: codex_local.run_codex("x"),
             lambda: assistant.ask("x", None),
             lambda: research.answer("x", None, system="s"),
             lambda: research.news_digest("A", "A"),
             lambda: earnings.generate_er_sheet("x", "k")]
    for call in calls:
        with pytest.raises(PermissionError):
            call()


@pytest.mark.anonymous
def test_logout_without_a_cookie_still_signs_out(monkeypatch):
    """A-02: the cookie library deletes with `del` (KeyError when there is no cookie); the session must still end."""
    from types import SimpleNamespace

    class Rerun(Exception):
        pass

    class CookieManager:
        def delete(self, *a, **k):
            raise KeyError("hqi_session")

    state = {"authenticated": True, "user_id": "u1", "user_email": "u@x", "portfolio": {"A": 1}}
    sidebar = SimpleNamespace(markdown=lambda *a, **k: None, caption=lambda *a, **k: None,
                              button=lambda *a, **k: True)
    monkeypatch.setattr(auth, "local_dev_mode", lambda: False)
    monkeypatch.setattr(auth.st, "session_state", state)
    monkeypatch.setattr(auth.st, "sidebar", sidebar)
    monkeypatch.setattr(auth.time, "sleep", lambda s: None)
    monkeypatch.setattr(auth.st, "rerun", lambda: (_ for _ in ()).throw(Rerun()))
    with pytest.raises(Rerun):
        auth.render_user_profile(CookieManager())
    assert state == {"authenticated": False, "user_id": None, "user_email": None}     # nothing of the user remains
