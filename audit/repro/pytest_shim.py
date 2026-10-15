"""Minimal pytest stand-in (audit sandbox has no pytest and no network). Supports approx, raises,
mark.parametrize, mark.<any>, fixture (no-arg function fixtures), skip, monkeypatch."""
import math, contextlib, types
class _Approx:
    def __init__(s, v, rel=None, abs=None): s.v, s.rel, s.abs = v, rel, abs
    def __eq__(s, o):
        try:
            tol = max((s.rel if s.rel is not None else (1e-6 if s.abs is None else 0)) * abs(s.v), s.abs or (1e-12 if s.rel is None else 0))
            return abs(o - s.v) <= tol
        except TypeError: return o == s.v
    def __repr__(s): return f"approx({s.v})"
def approx(v, rel=None, abs=None): return _Approx(v, rel, abs)
@contextlib.contextmanager
def raises(exc, match=None):
    try: yield
    except exc: return
    raise AssertionError(f"DID NOT RAISE {exc}")
class Skipped(Exception): pass
def skip(msg=""): raise Skipped(msg)
def importorskip(name):
    try: return __import__(name)
    except ImportError: raise Skipped(name)
class _Mark:
    def parametrize(self, names, values, **k):
        def deco(f):
            f.__dict__.setdefault("_params", []).append((names, values)); return f
        return deco
    def __getattr__(self, n):
        def deco(*a, **k):
            if a and callable(a[0]) and len(a)==1 and not k: return a[0]
            return lambda f: f
        return deco
mark = _Mark()
def fixture(*a, **k):
    if a and callable(a[0]): a[0]._is_fixture=True; return a[0]
    def deco(f): f._is_fixture=True; return f
    return deco
class MonkeyPatch:
    def __init__(s): s._undo=[]
    def setattr(s, target, name, value=None, raising=True):
        if isinstance(target, str):
            mod, _, attr = target.rpartition("."); import importlib; target = importlib.import_module(mod); value=name; name=attr
        s._undo.append((target, name, getattr(target, name, None))); setattr(target, name, value)
    def setenv(s, k, v):
        import os; s._undo.append((os.environ, k, os.environ.get(k))); os.environ[k]=v
    def delenv(s, k, raising=True):
        import os; os.environ.pop(k, None)
    def undo(s):
        for t, n, v in reversed(s._undo):
            if t.__class__.__name__ == "_Environ":
                import os
                if v is None: os.environ.pop(n, None)
                else: os.environ[n]=v
            else: setattr(t, n, v)
