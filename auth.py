import streamlit as st
from supabase import create_client, Client
import logging

@st.cache_resource
def get_supabase_client():
    """Initialize and cache the Supabase client."""
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

def login_form():
    """Render the authentication UI (Login/Signup)."""
    supabase = get_supabase_client()
    if not supabase:
        st.error("⚠️ System Error: SUPABASE_URL and SUPABASE_KEY missing in .streamlit/secrets.toml")
        st.stop()
        
    st.markdown("""
    <style>
    /* Global Background matching app.py */
    .stApp {
        background: radial-gradient(circle at top right, #1a1c2c, #0d0e14);
        color: #e0e0e0;
    }
    
    /* Center container styling */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Input fields */
    .stTextInput input {
        background-color: rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    
    /* Button primary styling */
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
    
    /* Gradient text for title */
    .gradient-text {
        background: linear-gradient(90deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'><span class='gradient-text'>Honest Quant Intelligence</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8e95a3; font-size: 1.1em;'>Secured Analytics Gateway</p><br>", unsafe_allow_html=True)
    
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
                        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        if response.user:
                            st.session_state["authenticated"] = True
                            st.session_state["user_id"] = response.user.id
                            st.session_state["user_email"] = response.user.email
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
                new_password = st.text_input("Password", type="password", help="Minimum 6 characters")
                if st.button("Create Account", type="primary", use_container_width=True):
                    if not new_email or len(new_password) < 6:
                        st.error("Invalid email or password is less than 6 characters.")
                        return
                    try:
                        response = supabase.auth.sign_up({"email": new_email, "password": new_password})
                        if response.user:
                            st.success("✅ Registration successful! You can now log in.")
                    except Exception as e:
                        if "User already registered" in str(e):
                            st.error("⚠️ Email is already registered.")
                        else:
                            st.error(f"Initialization Error: {e}")

def require_auth():
    """
    Authentication gateway. Blocks UI rendering if user is not authenticated.
    """
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["user_id"] = None
        
    if not st.session_state["authenticated"]:
        login_form()
        st.stop() # Halt execution of the rest of the Streamlit app
        
def render_user_profile():
    """Display user profile and logout button in sidebar."""
    if st.session_state.get("authenticated"):
        st.sidebar.markdown(f"👤 **{st.session_state.get('user_email', 'User')}**")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            try:
                supabase = get_supabase_client()
                supabase.auth.sign_out()
            except:
                pass
            st.session_state["authenticated"] = False
            st.session_state["user_id"] = None
            st.session_state["user_email"] = None
            st.rerun()
