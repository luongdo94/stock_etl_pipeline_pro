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


def test_missing_secrets_stops_login_with_setup_guidance(missing_secrets, monkeypatch):
    messages = []
    monkeypatch.setattr(auth.st, "error", messages.append)

    class Stopped(Exception):
        pass

    def stop():
        raise Stopped()

    monkeypatch.setattr(auth.st, "stop", stop)
    with pytest.raises(Stopped):
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
