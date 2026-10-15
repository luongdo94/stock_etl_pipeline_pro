import sys, importlib, inspect, itertools, traceback
sys.path.insert(0, "/home/claude/shim"); sys.path.insert(0, ".")
import pytest
res = {"pass":0,"fail":0,"error":0,"skip":0}; fails=[]
for modname in sys.argv[1:]:
    try: m = importlib.import_module("tests."+modname)
    except Exception as e:
        res["error"]+=1; fails.append((modname, "IMPORT", repr(e)[:200])); continue
    fixtures = {n:f for n,f in vars(m).items() if getattr(f,"_is_fixture",False)}
    def calls(fn, owner=None):
        params = getattr(fn, "_params", [])
        combos=[{}]
        for names, values in params:
            names=[x.strip() for x in names.split(",")] if isinstance(names,str) else list(names)
            new=[]
            for c in combos:
                for v in values:
                    v = v if len(names)>1 else (v,)
                    new.append({**c, **dict(zip(names, v))})
            combos=new
        return combos
    items=[]
    for n,o in vars(m).items():
        if n.startswith("test_") and callable(o): items.append((n,o,None))
        if inspect.isclass(o) and n.startswith("Test"):
            for mn,mo in vars(o).items():
                if mn.startswith("test_"): items.append((f"{n}.{mn}", mo, o))
    for name, fn, cls in items:
        for kw in calls(fn):
            mp = pytest.MonkeyPatch()
            try:
                sig = inspect.signature(fn); args={}
                for p in sig.parameters:
                    if p=="self": continue
                    if p in kw: args[p]=kw[p]
                    elif p=="monkeypatch": args[p]=mp
                    elif p in fixtures: args[p]=fixtures[p]()
                    elif p=="tmp_path":
                        import tempfile, pathlib; args[p]=pathlib.Path(tempfile.mkdtemp())
                    else: raise pytest.Skipped(f"fixture {p}")
                if cls: fn(cls(), **args)
                else: fn(**args)
                res["pass"]+=1
            except pytest.Skipped: res["skip"]+=1
            except AssertionError as e:
                res["fail"]+=1; fails.append((modname, name+str(kw)[:60], (str(e) or traceback.format_exc().splitlines()[-3])[:300]))
            except Exception as e:
                res["error"]+=1; fails.append((modname, name+str(kw)[:60], repr(e)[:300]))
            finally: mp.undo()
print(res)
for f in fails: print("FAIL", f)
