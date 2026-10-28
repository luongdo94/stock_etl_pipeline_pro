"""
auth.py — Multi-tenant authentication gateway with robust cookie-based session persistence.

Uses `extra_streamlit_components.CookieManager` for reliable browser cookie access.
TTL: 7 days.
"""
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from supabase import create_client
import base64
import hashlib
import hmac
import os
import time
import json
from datetime import datetime, timedelta
from typing import Optional

_COOKIE_NAME = "hqi_session"
_COOKIE_TTL_DAYS = 7


def _secret(name):
    """Streamlit secrets first, then the environment (deployments that set environment variables, as the app read
    them before 2026-10-08). An unconfigured store must not crash authentication or enable local access."""
    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        value = None
    return value or os.environ.get(name) or None


@st.cache_resource
def get_supabase_client():
    """Shared data client (service role). Never call .auth.* on it: a sign-in would
    swap its Authorization header to that user's JWT for every session in the process."""
    url = _secret("SUPABASE_URL")
    key = _secret("SUPABASE_SERVICE_ROLE_KEY") or _secret("SUPABASE_SERVICE_KEY") or _secret("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def _new_auth_client():
    """Fresh, uncached client for sign-in / sign-up / reset so auth state stays per-request."""
    url = _secret("SUPABASE_URL")
    # Fail closed (audit F-21): sign-in / sign-up must never run with the service-role key, which bypasses RLS
    key = _secret("SUPABASE_ANON_KEY") or _secret("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


# ── Signed session cookie ────────────────────────────────────────────────────
# The cookie used to be plain JSON {"user_id": ...}; anyone could forge it and read or
# overwrite another user's watchlist/portfolio (queries run with the service-role key).
# It is now HMAC-SHA256 signed with COOKIE_SECRET and carries its own expiry.
def _cookie_secret() -> Optional[bytes]:
    secret = _secret("COOKIE_SECRET")
    return secret.encode() if secret else None


def _sign_session(user_id: str, user_email: str) -> Optional[str]:
    secret = _cookie_secret()
    if not secret:
        return None
    payload = json.dumps(
        {"user_id": user_id, "user_email": user_email,
         "exp": int(time.time()) + _COOKIE_TTL_DAYS * 86400},
        separators=(",", ":"),
    ).encode()
    body = base64.urlsafe_b64encode(payload).decode()
    sig = hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def _verify_session(token) -> Optional[dict]:
    secret = _cookie_secret()
    if not secret or not isinstance(token, str) or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    expected = hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig.encode(), expected.encode()):  # bytes: str form rejects non-ASCII input
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(body.encode()))
    except Exception:
        return None
    if not data.get("user_id") or data.get("exp", 0) < time.time():
        return None
    return data


def get_cookie_manager():
    if local_dev_mode():
        return None                                  # no login → no cookie round-trip
    import extra_streamlit_components as stx
    return stx.CookieManager(key="hqi_cookie_manager")


def _login_form(cm):
    """The Login / Signup widgets (shown in the sign-in dialog). Writes the cookie on success. Without a configured
    Supabase the reason is shown and the app keeps running for browsing — only saving and the AI features need it."""
    supabase = _new_auth_client()
    if not supabase:
        st.error("Sign-in is not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY "
                 "(or SUPABASE_KEY) to .streamlit/secrets.toml. For local development without "
                 "Supabase login, run ./scripts/windows/start_dashboard.ps1 from the project directory.")
        return

    with st.container():
        tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

        with tab_login:
            with st.container(border=True):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                if st.button("Sign In", type="primary", use_container_width=True):
                    if not email or not password:
                        st.error("Please enter Email and Password.")
                        return
                    try:
                        resp = supabase.auth.sign_in_with_password(
                            {"email": email, "password": password}
                        )
                        if resp.user:
                            # 1. Write the signed cookie (skipped when COOKIE_SECRET is unset:
                            #    the session then lasts only for this browser tab)
                            token = _sign_session(resp.user.id, resp.user.email)
                            if token and cm is not None:
                                expires = datetime.now() + timedelta(days=_COOKIE_TTL_DAYS)
                                cm.set(_COOKIE_NAME, token, expires_at=expires, key="set_login_cookie")

                            # 2. Assign state
                            st.session_state["authenticated"] = True
                            st.session_state["user_id"] = resp.user.id
                            st.session_state["user_email"] = resp.user.email

                            # 3. Brief sleep to allow the component to send the set-cookie command to browser BEFORE rerun
                            time.sleep(0.5)
                            st.rerun()
                    except Exception as e:
                        if "Invalid login credentials" in str(e):
                            st.error("Incorrect email or password.")
                        else:
                            st.error(f"System Error: {e}")

                with st.expander("Forgot Password?"):
                    reset_email = st.text_input("Enter your email address", key="reset_email")
                    if st.button("Send Reset Link", use_container_width=True):
                        if not reset_email:
                            st.error("Please enter your email.")
                        else:
                            try:
                                supabase.auth.reset_password_email(reset_email)
                                st.success("Password reset link sent! Check your inbox.")
                            except Exception as e:
                                st.error(f"Error sending reset link: {e}")

        with tab_signup:
            with st.container(border=True):
                new_email = st.text_input("Email", key="new_email")
                new_password = st.text_input(
                    "Password", type="password", help="Minimum 6 characters"
                )
                if st.button("Create Account", type="primary", use_container_width=True):
                    if not new_email or len(new_password) < 6:
                        st.error("Invalid email or password is less than 6 characters.")
                        return
                    try:
                        resp = supabase.auth.sign_up(
                            {"email": new_email, "password": new_password}
                        )
                        if resp.user:
                            st.success("Registration successful! You can now log in.")
                    except Exception as e:
                        if "User already registered" in str(e):
                            st.error("Email is already registered.")
                        else:
                            st.error(f"Initialization Error: {e}")


LOCAL_USER_ID = "00000000-0000-0000-0000-00000000dev1"
_LOCAL_HOSTS = (None, "", "localhost", "127.0.0.1", "::1")


def local_dev_mode() -> bool:
    """
    True only when LOCAL_DEV_MODE=1 is set in the environment AND the app is served from this machine and is not
    in cloud (remote warehouse) mode. A secret / config file cannot switch it on, and it can never be on for a
    deployed app: the login stays mandatory everywhere else.
    """
    import os
    if os.environ.get("LOCAL_DEV_MODE") != "1":
        return False
    if os.environ.get("SUPABASE_REMOTE_MODE", "false").lower() == "true":
        return False
    try:
        address = st.get_option("server.address")
    except Exception:
        address = None
    return address in _LOCAL_HOSTS


_COOKIE_MANAGER = None        # the page's cookie component (one per run), used by the sign-in dialog


def signed_in() -> bool:
    """A user with an account is signed in (local dev mode counts as signed in)."""
    return bool(st.session_state.get("authenticated") and st.session_state.get("user_id"))


def restore_session(cm=None):
    """Who is browsing. There is no login wall: a valid signed cookie restores the account; without one the visitor
    browses anonymously, and sign-in is asked for only where it is needed — to SAVE data (watchlist, portfolio, alert
    rules, trades) and for the AI features that run on the owner's language-model key (see can_use)."""
    import os
    global _COOKIE_MANAGER
    if os.environ.get("LOCAL_DEV_MODE") == "1" and not local_dev_mode():
        st.error("LOCAL_DEV_MODE is only allowed when the app is served from localhost and is not in cloud mode. "
                 "Unset it, or sign in normally.")
        st.stop()
    if local_dev_mode():
        st.session_state.update(authenticated=True, user_id=LOCAL_USER_ID, user_email="local@dev (no login)")
        return
    if cm is None:
        cm = get_cookie_manager()
    _COOKIE_MANAGER = cm

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["user_id"] = None
        st.session_state["user_email"] = None

    if st.session_state["authenticated"]:
        return

    # Check cookie
    # stx.CookieManager gets all cookies instantly after 0.1s spin up
    raw = cm.get(_COOKIE_NAME)

    if raw:
        data = _verify_session(raw)
        if data:
            st.session_state["authenticated"] = True
            st.session_state["user_id"] = data["user_id"]
            st.session_state["user_email"] = data.get("user_email", "")
            st.rerun()
            return

    # Give Streamlit 1 pass to establish the component connection if we have NO raw cookie
    if "cm_pass" not in st.session_state:
        st.session_state["cm_pass"] = 1
        with st.spinner(""):
            time.sleep(0.5)  # Let the component initialize
        st.rerun()  # Force a rerun to capture the cookies now that JS is loaded

    # no valid cookie: an anonymous visitor (nothing is stopped)


@st.dialog("Sign in")
def _sign_in_dialog():
    _login_form(_COOKIE_MANAGER)


def can_use(purpose: str, key: str) -> bool:
    """True when the visitor is signed in. Otherwise a one-line notice saying what needs an account, with a Sign in
    button that opens the sign-in dialog, and False — so a caller writes `if auth.can_use(...) and st.button(...)`."""
    if signed_in():
        return True
    c1, c2 = st.columns([4, 1])
    c1.info(f"Sign in to {purpose}. Everything else works without an account.")
    if c2.button("Sign in", key=f"signin_{key}", width="stretch"):
        _sign_in_dialog()
    return False


def render_user_profile(cm=None):
    if not signed_in():
        st.sidebar.caption("Browsing without an account. Sign in to save a watchlist, a portfolio and alerts, and to "
                           "use the AI features.")
        if st.sidebar.button("Sign in", key="sidebar_sign_in", use_container_width=True):
            _sign_in_dialog()
        return

    if local_dev_mode():
        st.sidebar.markdown("**Local dev mode**")
        st.sidebar.caption("No login. Watchlist, portfolio and alerts are saved to `warehouse/local_user/` on this machine.")
        return

    if cm is None:
        cm = get_cookie_manager()
    st.sidebar.markdown(f"**{st.session_state.get('user_email', 'User')}**")

    if st.sidebar.button("Logout", use_container_width=True):
        cm.delete(_COOKIE_NAME, key="del_login_cookie")
        # Everything the session holds belongs to this user (assistant history with portfolio documents, portfolio and
        # watchlist state, research answers): none of it may survive into the next sign-in (review REV-01).
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.session_state["authenticated"] = False
        st.session_state["user_id"] = None
        st.session_state["user_email"] = None
        time.sleep(0.5)
        st.rerun()
