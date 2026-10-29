"""Shared session choices, with separate widget keys for simultaneous surfaces."""
import streamlit as st
from services.agy_local import DEFAULT_MODEL, DEFAULT_EFFORT, agy_models


def agy_label(model, effort):
    return f"Antigravity (local): {model or DEFAULT_MODEL} / {effort or DEFAULT_EFFORT}"


def render_agy_options(scope="assistant"):
    def remember(shared, key):
        st.session_state[shared] = st.session_state[key]

    selected = []
    for col, kind, label, options, default in zip(st.columns(2), ("model", "effort"),
            ("Antigravity model", "Reasoning"), ([None, *(m for m in agy_models() if m != DEFAULT_MODEL)], [None, "low", "medium", "high"]),
            (DEFAULT_MODEL, DEFAULT_EFFORT)):
        shared, key = f"_agy_{kind}", f"agy_{scope}_{kind}"
        value = st.session_state.get(shared)
        st.session_state[key] = value if value in options else None
        with col:
            selected.append(st.selectbox(label, options, key=key,
                format_func=lambda v, d=default: f"{d} (default)" if v is None else v,
                on_change=remember, args=(shared, key)))
    return tuple(selected)
