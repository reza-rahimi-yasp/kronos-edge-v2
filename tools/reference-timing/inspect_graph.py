"""Print the Hardswish pattern, clone origins, and assert nodes of a decomposed .pt2 graph (CPU only; docs/reference-timing.md)."""
import sys, torch, collections
ep = torch.export.load(sys.argv[1] if len(sys.argv) > 1 else "/work/artifact/model.pt2")
g = ep.graph_module.graph
nodes = list(g.nodes)
def tname(n): return getattr(n.target, "__name__", None) or str(n.target) if isinstance(n, torch.fx.Node) else repr(n)
def desc(n): return f"{n.name}={tname(n)}(" + ", ".join(a.name if isinstance(a, torch.fx.Node) else repr(a) for a in n.args) + (f", **{n.kwargs}" if n.kwargs else "") + ")"
# first div node and its producer chain (hardswish pattern)
div = next(n for n in nodes if n.op == "call_function" and tname(n) == "div.Tensor")
print("== producer chain of first div:")
seen=set()
def chain(n, depth=0):
    if n in seen or depth>6: return
    seen.add(n); print("  "*depth + desc(n))
    for a in n.args:
        if isinstance(a, torch.fx.Node) and a.op=="call_function": chain(a, depth+1)
chain(div)
print("== clone nodes: kwargs histogram, input op histogram, consumer histogram, module stack sample")
clones=[n for n in nodes if n.op=="call_function" and tname(n)=="clone.default"]
print(" count", len(clones))
print(" kwargs", collections.Counter(str(n.kwargs) for n in clones))
print(" input", collections.Counter((n.args[0].op, tname(n.args[0]) if n.args[0].op=="call_function" else n.args[0].name[:12]) for n in clones))
print(" users", collections.Counter(tuple(sorted(tname(u) for u in n.users)) for n in clones))
for n in clones[:4]:
    st = n.meta.get("stack_trace", ""); ms = n.meta.get("nn_module_stack", {})
    print("  ", desc(n), "| users:", [desc(u)[:60] for u in n.users])
    print("     module_stack:", list(ms.values())[-2:] if ms else None)
    print("     stack_trace tail:", st.strip().splitlines()[-2:] if st else None)
print("== _assert_tensor_metadata sample:")
for n in [n for n in nodes if tname(n)=="_assert_tensor_metadata.default"][:2]:
    print("  ", desc(n)); st=n.meta.get("stack_trace",""); print("     ", st.strip().splitlines()[-2:] if st else None)
print("== placeholders:", sum(1 for n in nodes if n.op=="placeholder"), "user inputs:", [s.arg.name for s in ep.graph_signature.input_specs if str(s.kind).endswith("USER_INPUT")])
