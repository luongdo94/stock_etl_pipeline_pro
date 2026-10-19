"""Widgets whose value survives the runs in which they are not drawn.

Streamlit drops the state of a widget on a run that does not draw it (another group of the Stock Analysis page is open,
another tab). The last value is kept in a plain session key and passed back as the widget's initial value; the widget is
not given a value through `st.session_state[key] = ...` (that makes Streamlit warn on every run).
"""
import streamlit as st

KEPT = "_kept::"


def kept_key(key: str) -> str:
    return f"{KEPT}{key}"


def sticky(widget, label, default, key, **kw):
    """`widget(label, value=<last value or default>, key=key, **kw)` for number_input / text_input / text_area / toggle."""
    v = widget(label, value=st.session_state.get(kept_key(key), default), key=key, **kw)
    st.session_state[kept_key(key)] = v
    return v


def sticky_choice(label, options, default, key, **kw):
    """st.selectbox that remembers its choice (`default` is an option, not an index)."""
    options = list(options)
    cur = st.session_state.get(kept_key(key), default)
    v = st.selectbox(label, options, index=options.index(cur) if cur in options else (options.index(default) if default in options else None), key=key, **kw)
    st.session_state[kept_key(key)] = v
    return v


def forget(prefix: str) -> None:
    """Drop the remembered values (and the widget state) of every key that starts with `prefix`."""
    for k in [k for k in st.session_state if str(k).startswith(kept_key(prefix)) or str(k).startswith(prefix)]:
        del st.session_state[k]


def sticky_multi(label, options, default, key, **kw):
    """st.multiselect that remembers its choice; options that no longer exist (the snapshot changed) are dropped."""
    options = list(options)
    cur = [o for o in st.session_state.get(kept_key(key), default) if o in options]
    v = st.multiselect(label, options, default=cur, key=key, **kw)
    st.session_state[kept_key(key)] = v
    return v


def sticky_pills(label, options, default, key, **kw):
    """st.pills (single choice) that remembers its choice; never returns None (a deselected pill keeps the last one)."""
    options = list(options)
    cur = st.session_state.get(kept_key(key), default)
    cur = cur if cur in options else default
    v = st.pills(label, options, selection_mode="single", default=cur, key=key, **kw) or cur
    st.session_state[kept_key(key)] = v
    return v


def sticky_radio(label, options, default, key, **kw):
    """st.radio that remembers its choice (`default` is an option, not an index)."""
    options = list(options)
    cur = st.session_state.get(kept_key(key), default)
    v = st.radio(label, options, index=options.index(cur) if cur in options else options.index(default), key=key, **kw)
    st.session_state[kept_key(key)] = v
    return v


def sticky_segmented(label, options, default, key, **kw):
    """Single segmented control with the same persistence as sticky_pills."""
    options = list(options)
    cur = st.session_state.get(kept_key(key), default)
    cur = cur if cur in options else default
    v = st.segmented_control(label, options, default=cur, key=key, **kw) or cur
    st.session_state[kept_key(key)] = v
    return v
