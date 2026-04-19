"""
auth.py — Multi-tenant authentication gateway with robust cookie-based session persistence.

Uses `extra_streamlit_components.CookieManager` for reliable browser cookie access.
TTL: 7 days.
"""
import streamlit as st
from supabase import create_client
import time
import json
from datetime import datetime, timedelta

_COOKIE_NAME = "hqi_session"
_COOKIE_TTL_DAYS = 7


@st.cache_resource
def get_supabase_client():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def get_cookie_manager():
    import extra_streamlit_components as stx
    return stx.CookieManager(key="hqi_cookie_manager")


def _login_form(cm):
    """Render the Login / Signup UI. Writes cookie on success."""
    supabase = get_supabase_client()
    if not supabase:
        st.error("⚠️ System Error: SUPABASE_URL and SUPABASE_KEY missing in .streamlit/secrets.toml")
        st.stop()

    st.markdown("""
    <style>
    /* HIDE DEFAULT STREAMLIT ARTIFACTS - AGGRESSIVE MODE (Active on Login) */
    .stAppDeployButton, [data-testid="stAppDeployButton"] { visibility: hidden !important; display: none !important; }
    header[data-testid="stHeader"], [data-testid="stHeader"], header { visibility: hidden !important; display: none !important; }
    [data-testid="stToolbar"] { visibility: hidden !important; display: none !important; }
    [data-testid="stSidebarNav"] { visibility: hidden !important; display: none !important; }
    #stDecoration { visibility: hidden !important; display: none !important; }
    footer { visibility: hidden !important; display: none !important; }
    /* Push the UI up */
    .block-container { padding-top: 1rem !important; }

    .stApp {
        background: radial-gradient(circle at top right, #1a1c2c, #0d0e14);
        color: #e0e0e0;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .stTextInput input {
        background-color: rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        border: none !important;
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    [data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
    }
    .gradient-text {
        background: linear-gradient(90deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center;'><span class='gradient-text'>Honest Quant Intelligence</span></h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#8e95a3; font-size:1.1em;'>Secured Analytics Gateway</p><br>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔑 Log In", "📝 Sign Up"])

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
                            # 1. Write the cookie logically
                            payload = json.dumps({"user_id": resp.user.id, "user_email": resp.user.email})
                            expires = datetime.now() + timedelta(days=_COOKIE_TTL_DAYS)
                            cm.set(_COOKIE_NAME, payload, expires_at=expires, key="set_login_cookie")

                            # 2. Assign state
                            st.session_state["authenticated"] = True
                            st.session_state["user_id"] = resp.user.id
                            st.session_state["user_email"] = resp.user.email
                            
                            # 3. Brief sleep to allow the component to send the set-cookie command to browser BEFORE rerun
                            time.sleep(0.5)
                            st.rerun()
                    except Exception as e:
                        if "Invalid login credentials" in str(e):
                            st.error("❌ Incorrect email or password.")
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
                                st.success("✅ Password reset link sent! Check your inbox.")
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
                            st.success("✅ Registration successful! You can now log in.")
                    except Exception as e:
                        if "User already registered" in str(e):
                            st.error("⚠️ Email is already registered.")
                        else:
                            st.error(f"Initialization Error: {e}")


def require_auth(cm=None):
    """
    Authentication gateway.
    """
    if cm is None:
        cm = get_cookie_manager()
    
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
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            uid = data.get("user_id")
            if uid:
                st.session_state["authenticated"] = True
                st.session_state["user_id"] = uid
                st.session_state["user_email"] = data.get("user_email", "")
                st.rerun()
                return
        except Exception:
            pass

    # Give Streamlit 1 pass to establish the component connection if we have NO raw cookie
    if "cm_pass" not in st.session_state:
        st.session_state["cm_pass"] = 1
        with st.spinner(""):
            time.sleep(0.5)  # Let the component initialize
        st.rerun()  # Force a rerun to capture the cookies now that JS is loaded

    # If we reach here, user is truly not authenticated
    _login_form(cm)
    st.stop()


def render_user_profile(cm=None):
    if not st.session_state.get("authenticated"):
        return

    if cm is None:
        cm = get_cookie_manager()
    st.sidebar.markdown(f"👤 **{st.session_state.get('user_email', 'User')}**")

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        cm.delete(_COOKIE_NAME, key="del_login_cookie")
        try:
            get_supabase_client().auth.sign_out()
        except Exception:
            pass
        st.session_state["authenticated"] = False
        st.session_state["user_id"] = None
        st.session_state["user_email"] = None
        if "cm_pass" in st.session_state:
            del st.session_state["cm_pass"]
        time.sleep(0.5)
        st.rerun()

        # Best-effort Supabase sign out
        try:
            get_supabase_client().auth.sign_out()
        except Exception:
            pass

        st.rerun()
