"""Diagnostic timing harness behind docs/reference-timing.md. Runs inside a disposable container (see run_in.sh).

Subjects (what is timed):
  eager      Model() from model.py, as Kronos runs it (fp32, eval, on cuda).
  pt2        the decomposed ExportedProgram from model.pt2, run the way Gaia runs a reference: its graph_module
             called with the lifted parameters/buffers/constants and the user input as flat positional inputs.
  pt2module  ExportedProgram.module() (unlifted wrapper) called with the user input only.
  cand       Model() from an AgentKernel source (kernel.py); needs nvcc.
Factors: --sync (each|end), --flush (none|l2), --grad (no_grad|inference_mode|enabled), --mode (block|pair),
         --inputs (npz|seeded|randn), --weights (own|pt2|synth), --tf32, --cudnn-benchmark, warm-up and run counts.
Everything is reported as a per-run distribution. Nothing here is a product feature.
"""
import argparse
import copy
import importlib.util
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
from torch.export.graph_signature import InputKind


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flat_inputs(ep, user, device, synth=None):
    """Flat positional inputs of the exported graph, Gaia-style: real values for params/buffers/constants (or
    synthesized from their statistics when synth is a Generator), the user input last."""
    out, kinds = [], []
    users = list(user)
    for spec in ep.graph_signature.input_specs:
        k = spec.kind
        if k == InputKind.USER_INPUT:
            t = users.pop(0)
        elif k == InputKind.PARAMETER or (k == InputKind.BUFFER and spec.persistent):
            t = ep.state_dict[spec.target]
            if synth is not None and t.is_floating_point():
                s = torch.randn(t.shape, generator=synth, dtype=t.dtype) * (t.std() if t.numel() > 1 else 1) + t.mean()
                t = s.clamp(t.min(), t.max())
        elif k in (InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
            t = ep.constants[spec.target]
        else:
            raise RuntimeError(f"unsupported input kind {k}")
        out.append(t.detach().to(device))
        kinds.append(str(k).split(".")[-1])
    assert not users
    return out, kinds


def gaia_gm(ep):
    """Mirror exporter._to_serializable_gm: a plain GraphModule over a deep-copied, dead-code-eliminated graph."""
    g = copy.deepcopy(ep.graph_module.graph)
    g._tracer_cls = None
    g._tracer_extras = {}
    g.eliminate_dead_code()
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    gm.recompile()
    # Gaia's YaspIR.__post_init__ rewrites device kwargs baked in by the CPU trace to follow the input's device
    # (ir.normalize_device_kwargs). Use it when the yasp package is present; otherwise point them at cuda directly.
    baked = [n for n in gm.graph.nodes if n.op == "call_function" and isinstance(n.kwargs.get("device"), torch.device)]
    gm.graph_stats_device_kwargs = {"nodes": len(baked), "targets": sorted({str(n.target) for n in baked})}
    try:
        from yasp.compile.common.ir import normalize_device_kwargs
        normalize_device_kwargs(gm)
        gm.graph_stats_device_kwargs["normalized_by"] = "yasp.compile.common.ir.normalize_device_kwargs"
    except ImportError:
        for n in baked:
            n.kwargs = {**n.kwargs, "device": torch.device("cuda")}
        gm.recompile()
        gm.graph_stats_device_kwargs["normalized_by"] = "fallback: device kwarg set to cuda"
    return gm


def _t(n):
    return getattr(n.target, "__name__", None) or str(n.target)


def fuse_hardswish(gm):
    """Replace the decomposed pattern div(mul(x, clamp(clamp(add(x, 3), 0), None, 6)), 6) with aten.hardswish(x).
    This is the inverse of torch's hardswish decomposition; a diagnostic transform, not a product feature."""
    aten = torch.ops.aten
    g, fused = gm.graph, 0
    for div in list(g.nodes):
        if div.op != "call_function" or _t(div) != "div.Tensor" or len(div.args) != 2 or div.args[1] != 6:
            continue
        mul = div.args[0]
        if not (isinstance(mul, torch.fx.Node) and _t(mul) == "mul.Tensor" and len(mul.users) == 1):
            continue
        x, c2 = mul.args
        if not (isinstance(c2, torch.fx.Node) and _t(c2) == "clamp.default" and tuple(c2.args[1:]) == (None, 6) and len(c2.users) == 1):
            continue
        c1 = c2.args[0]
        if not (isinstance(c1, torch.fx.Node) and _t(c1) == "clamp.default" and tuple(c1.args[1:]) == (0,) and len(c1.users) == 1):
            continue
        add = c1.args[0]
        if not (isinstance(add, torch.fx.Node) and _t(add) == "add.Tensor" and add.args[0] is x and add.args[1] == 3 and len(add.users) == 1):
            continue
        with g.inserting_before(div):
            hs = g.call_function(aten.hardswish.default, (x,))
        div.replace_all_uses_with(hs)
        for n in (div, mul, c2, c1, add):
            g.erase_node(n)
        fused += 1
    g.lint(); gm.recompile()
    return fused


def drop_clones(gm):
    """Bypass clone nodes that carry no memory_format (the ones Dropout(p=0) leaves in an export); contiguous() clones stay."""
    g, dropped = gm.graph, 0
    for n in list(g.nodes):
        if n.op == "call_function" and _t(n) == "clone.default" and not n.kwargs:
            n.replace_all_uses_with(n.args[0])
            g.erase_node(n)
            dropped += 1
    g.lint(); gm.recompile()
    return dropped


def build_model(module):
    """Model() as the export does it: a fresh instance; re-seeded with the file's SEED when it defines one, so every
    instance in this process gets the weights the first (and the pt2) got."""
    if hasattr(module, "SEED"):
        torch.manual_seed(module.SEED)
    return module.Model()


def export_variants(module, user, device, wanted):
    """In-container torch.export of Model() on cuda (same no_grad/eval recipe as Gaia's exporter), plus decomposition variants."""
    model = build_model(module).eval().to(device)
    with torch.no_grad():
        ep = torch.export.export(model, tuple(t.clone() for t in user))
    out = {}
    if "exp_nodecomp" in wanted:
        out["exp_nodecomp"] = ep
    if "exp_decomp" in wanted:
        out["exp_decomp"] = ep.run_decompositions()
    if "exp_keephs" in wanted:
        table = torch.export.default_decompositions()
        removed = [op for op in list(table) if "hardswish" in str(op)]
        for op in removed:
            del table[op]
        out["exp_keephs"] = ep.run_decompositions(table)
        log("exp_keephs: removed from the decomposition table:", [str(o) for o in removed])
    return out


def graph_stats(gm):
    hist = {}
    for n in gm.graph.nodes:
        if n.op == "call_function":
            name = getattr(n.target, "__name__", None) or str(n.target)
            hist[name] = hist.get(name, 0) + 1
    return {"call_function": sum(hist.values()), "targets": dict(sorted(hist.items(), key=lambda kv: -kv[1]))}


def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0}


class Timer:
    def __init__(self, flush, sync):
        self.flush, self.sync = flush, sync
        self.cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda") if flush == "l2" else None

    def one(self, fn):
        if self.cache is not None:
            self.cache.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        if self.sync == "each":
            torch.cuda.synchronize()
        return s, e


def grad_ctx(kind):
    return {"no_grad": torch.no_grad, "inference_mode": torch.inference_mode, "enabled": torch.enable_grad}[kind]()


def profile(fn, iters, label):
    from torch.profiler import ProfilerActivity, profile as tprof
    fn(); torch.cuda.synchronize()
    with tprof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per = {}
    total_us, n = 0.0, 0
    for ev in p.events():
        dt = getattr(ev, "device_type", None)
        is_cuda = dt is not None and "CUDA" in str(dt)
        if not is_cuda:
            continue
        us = ev.time_range.elapsed_us()
        total_us += us; n += 1
        d = per.setdefault(ev.name, [0.0, 0])
        d[0] += us; d[1] += 1
    top = sorted(per.items(), key=lambda kv: -kv[1][0])[:40]
    return {"label": label, "iters": iters, "gpu_ms_per_iter": total_us / 1e3 / iters, "kernels_per_iter": n / iters,
            "top": [{"name": k[:120], "ms_per_iter": v[0] / 1e3 / iters, "count_per_iter": v[1] / iters} for k, v in top]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/work/model.py")
    ap.add_argument("--npz", default="/work/input.npz")
    ap.add_argument("--pt2", default="/work/artifact/model.pt2")
    ap.add_argument("--kernel", default="/work/artifact/kernel.py")
    ap.add_argument("--subjects", nargs="+", default=["eager", "pt2"])
    ap.add_argument("--inputs", choices=["npz", "seeded", "randn"], default="npz")
    ap.add_argument("--weights", choices=["own", "pt2", "synth"], default="own", help="eager: own=Model() init, pt2=load pt2 state_dict; pt2: synth=Gaia-like synthesized params")
    ap.add_argument("--grad", choices=["no_grad", "inference_mode", "enabled"], default="no_grad")
    ap.add_argument("--sync", choices=["each", "end"], default="end")
    ap.add_argument("--flush", choices=["none", "l2"], default="none")
    ap.add_argument("--mode", choices=["block", "pair"], default="block", help="block: each subject warm-up+runs contiguously (order alternates per trial); pair: subjects alternate call by call, AB/BA")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--tf32", choices=["asis", "on", "off"], default="asis")
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--profile", type=int, default=0, help="profile N iterations per subject instead of timing")
    ap.add_argument("--check", action="store_true", help="compare subject outputs with each other (max abs diff)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    if args.tf32 != "asis":
        on = args.tf32 == "on"
        torch.backends.cuda.matmul.allow_tf32 = on
        torch.backends.cudnn.allow_tf32 = on
    device = torch.device("cuda")
    env = {
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0],
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    try:
        import triton; env["triton"] = triton.__version__
    except Exception:
        env["triton"] = None
    try:
        import importlib.metadata as md; env["timm"] = md.version("timm")
    except Exception:
        env["timm"] = None
    log("env", env)

    # Inputs: pinned to input.npz unless asked otherwise.
    module = import_file(args.model, "kronos_user_model")
    get_inputs = getattr(module, "get_inputs", None) or getattr(module, "get_fwd_args")
    if args.inputs == "npz":
        with np.load(args.npz) as d:
            user = [torch.from_numpy(d[k]) for k in sorted(d.files)]
    elif args.inputs == "seeded":
        user = [t.detach() for t in get_inputs()]
    else:
        ref = [t for t in get_inputs()]
        user = [torch.randn(t.shape, dtype=t.dtype, generator=torch.Generator().manual_seed(1234)) for t in ref]
    user = [t.to(device) for t in user]
    log("inputs", args.inputs, [(tuple(t.shape), str(t.dtype)) for t in user])

    ep = None
    if any(s.split("+")[0] in ("pt2", "pt2module") for s in args.subjects) or args.weights == "pt2":
        ep = torch.export.load(args.pt2)

    subjects, outputs, info = {}, {}, {}
    wanted = {n.split("+")[0] for n in args.subjects if n.startswith("exp_")}
    exported = export_variants(module, user, device, wanted) if wanted else {}
    if "pt2" in {n.split("+")[0] for n in args.subjects} and ep is not None and "exp_decomp" in exported:
        info["exp_decomp_vs_pt2_targets_equal"] = graph_stats(gaia_gm(exported["exp_decomp"]))["targets"] == graph_stats(gaia_gm(ep))["targets"]
    for name in args.subjects:
        if name == "eager":
            model = build_model(module).eval()
            if ep is not None:
                sd = model.state_dict()
                diffs = {k: float((sd[k].float() - ep.state_dict[k].float()).abs().max()) for k in sd if k in ep.state_dict}
                missing = [k for k in sd if k not in ep.state_dict]
                info["eager_vs_pt2_weights"] = {"max_abs_diff": max(diffs.values()) if diffs else None, "compared": len(diffs), "missing_in_pt2": missing[:10]}
                log("eager vs pt2 weights", info["eager_vs_pt2_weights"])
                if args.weights == "pt2":
                    model.load_state_dict({k: v for k, v in ep.state_dict.items()}, strict=False)
            model = model.to(device)
            x = [t.clone() for t in user]
            subjects[name] = lambda m=model, x=x: m(*x)
        elif name.split("+")[0] in ("pt2", "exp_nodecomp", "exp_decomp", "exp_keephs"):
            base, *mods = name.split("+")
            src_ep = ep if base == "pt2" else exported[base]
            gm = gaia_gm(src_ep)
            if "fusehs" in mods:
                info[name + "_fused_hardswish"] = fuse_hardswish(gm)
            if "noclone" in mods:
                info[name + "_dropped_clones"] = drop_clones(gm)
            info[name + "_graph"] = graph_stats(gm)
            info[name + "_device_kwargs"] = gm.graph_stats_device_kwargs
            log(name, "graph: %d call_function nodes" % info[name + "_graph"]["call_function"], {k: v for k, v in info.items() if k.startswith(name + "_") and k.endswith(("hardswish", "clones"))})
            gen = torch.Generator().manual_seed(0) if args.weights == "synth" else None
            flat, kinds = flat_inputs(src_ep, [t.clone() for t in user], device, synth=gen)
            info[name + "_inputs"] = {k: kinds.count(k) for k in set(kinds)}
            gm = gm.to(device)
            subjects[name] = lambda g=gm, f=flat: g(*f)
        elif name == "pt2module":
            m = ep.module().to(device)
            x = [t.clone() for t in user]
            subjects[name] = lambda m=m, x=x: m(*x)
        elif name == "cand":
            os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_ext")
            kmod = import_file(args.kernel, "agent_kernel")
            km = kmod.Model().to(device)
            try:
                km.eval()
            except Exception:
                pass
            x = [t.clone() for t in user]
            subjects[name] = lambda m=km, x=x: m(*x)
        else:
            raise SystemExit(f"unknown subject {name}")

    result = {"label": args.label, "args": vars(args), "env": env, "info": info, "trials": []}

    with grad_ctx(args.grad):
        for n, fn in subjects.items():
            out = fn(); torch.cuda.synchronize()
            outputs[n] = out[0] if isinstance(out, (tuple, list)) else out
            log(n, "output", tuple(outputs[n].shape), str(outputs[n].dtype), "requires_grad", bool(outputs[n].requires_grad))
        if args.check and len(outputs) > 1:
            names = list(outputs)
            base = outputs[names[0]].float()
            result["check"] = {f"{names[0]}_vs_{m}": float((base - outputs[m].float()).abs().max()) for m in names[1:]}
            log("check", result["check"])

        if args.profile:
            result["profiles"] = [profile(fn, args.profile, n) for n, fn in subjects.items()]
            for p in result["profiles"]:
                log(f"profile {p['label']}: {p['gpu_ms_per_iter']:.3f} GPU ms/iter, {p['kernels_per_iter']:.0f} kernels/iter")
        else:
            timer = Timer(args.flush, args.sync)
            names = list(subjects)
            for trial in range(args.trials):
                order = names if trial % 2 == 0 else names[::-1]
                samples = {n: [] for n in names}
                t0 = time.perf_counter()
                if args.mode == "block":
                    for n in order:
                        fn = subjects[n]
                        for _ in range(args.warmup):
                            fn()
                        torch.cuda.synchronize()
                        evs = [timer.one(fn) for _ in range(args.runs)]
                        torch.cuda.synchronize()
                        samples[n] = [s.elapsed_time(e) for s, e in evs]
                else:
                    for _ in range(args.warmup):
                        for n in order:
                            subjects[n]()
                    torch.cuda.synchronize()
                    evs = {n: [] for n in names}
                    for i in range(args.runs):
                        for n in (order if i % 2 == 0 else order[::-1]):
                            evs[n].append(timer.one(subjects[n]))
                            if args.sync == "each":
                                torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    samples = {n: [s.elapsed_time(e) for s, e in evs[n]] for n in names}
                wall = time.perf_counter() - t0
                trial_res = {"trial": trial, "order": order, "wall_s": wall, "stats": {n: stats(v) for n, v in samples.items()}, "samples": samples}
                result["trials"].append(trial_res)
                log(f"trial {trial} order {order} wall {wall:.1f}s " + " | ".join(f"{n}: med {st['median']:.3f} mean {st['mean']:.3f} p95 {st['p95']:.3f}" for n, st in trial_res["stats"].items()))

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    log("wrote", args.out)


if __name__ == "__main__":
    main()
