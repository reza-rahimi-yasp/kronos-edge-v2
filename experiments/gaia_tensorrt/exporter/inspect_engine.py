"""Report the layouts TensorRT negotiated in an engine: per-layer formats, reformat layers, plugin boundaries.

Runs inside the dev container (TensorRT only, no GPU work beyond
deserialization). The engine must have been built with
ProfilingVerbosity.DETAILED (``build_hybrid_engine.py ... inspect``) for the
inspector to report tensor formats; an engine built at the default verbosity
yields layer names only, which is still reported. Writes <out json> with the
full inspector output and a summary: layer counts by type, every reformat
layer with the formats and sizes it converts, every plugin layer with the
formats of its inputs and outputs, and the formats TensorRT chose for the
native convolutions and matrix multiplies.

Usage: inspect_engine.py <engine> <plugin .so or -> <out json> [register symbol]
The register symbol defaults to the layout library's; the baseline library
registers through gaia_trt_register_plugins.
"""

import json
import sys
from collections import Counter
from pathlib import Path

PLUGIN_PREFIXES = ("dw_", "ep_", "q_", "att_")  # the schedule's plugin layer names (build_hybrid_engine.py)


def _format(tensor: dict) -> str:
    return tensor.get("Format", tensor.get("Format/Datatype", "?"))


def _volume(tensor: dict) -> int:
    v = 1
    for d in tensor.get("Dimensions", []):
        v *= int(d)
    return v


def summarize(layers: list) -> dict:
    detailed = all(isinstance(layer, dict) for layer in layers)
    if not detailed:
        names = [layer if isinstance(layer, str) else layer.get("Name", "?") for layer in layers]
        return {"detailed": False, "num_layers": len(names), "reformat_like_names": [n for n in names if "Tran" in n or "Reformat" in n or "reformat" in n],
                "names": names}
    by_type = Counter(layer.get("LayerType", "?") for layer in layers)
    reformats, plugins, native = [], [], []
    for layer in layers:
        kind, name = layer.get("LayerType", "?"), str(layer.get("Name", ""))
        ins = [{"name": t.get("Name"), "dims": t.get("Dimensions"), "format": _format(t)} for t in layer.get("Inputs", [])]
        outs = [{"name": t.get("Name"), "dims": t.get("Dimensions"), "format": _format(t)} for t in layer.get("Outputs", [])]
        entry = {"name": name, "type": kind, "inputs": ins, "outputs": outs, "tactic": layer.get("TacticName")}
        # TensorRT 11 names its layout changes __myl_Tran (transpose) and __myl_Move (copy into another layout).
        if kind == "Reformat" or layer.get("Origin") == "REFORMAT" or name.startswith(("__myl_Tran", "__myl_Move")):
            entry["elements"] = sum(_volume(t) for t in layer.get("Outputs", []))
            reformats.append(entry)
        elif kind in ("PluginV3", "custom_layer") and name.startswith(PLUGIN_PREFIXES):
            plugins.append(entry)
        elif kind in ("correlation", "gemm") or any(k in kind for k in ("Convolution", "MatrixMultiply")):
            native.append(entry)
    formats_at_plugins = Counter(f'{t["format"]}' for p in plugins for t in p["inputs"] + p["outputs"])
    return {
        "detailed": True, "num_layers": len(layers), "layers_by_type": dict(by_type),
        "reformat_layers": len(reformats), "reformat_elements_total": sum(r["elements"] for r in reformats),
        "reformats": sorted(reformats, key=lambda r: -r["elements"]),
        "plugin_layers": len(plugins), "formats_at_plugin_boundaries": dict(formats_at_plugins), "plugins": plugins,
        "native_conv_matmul": native,
    }


def main(engine_path: str, plugin_lib: str, out_path: str, symbol: str = "gaia_trt_layout_register_plugins") -> None:
    import tensorrt as trt

    import trt_build

    if plugin_lib != "-":
        trt_build.load_plugins(plugin_lib, symbol)
    log, _, _ = trt_build.new_builder()
    runtime = trt.Runtime(log)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise SystemExit("; ".join(log.errors) or "the engine does not deserialize")
    inspector = engine.create_engine_inspector()
    info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info.get("Layers", [])
    summary = summarize(layers)
    report = {"engine": engine_path, "tensorrt": trt.__version__, "summary": summary, "inspector": info}
    Path(out_path).write_text(json.dumps(report, indent=2))
    brief = {k: v for k, v in summary.items() if k not in ("reformats", "plugins", "native_conv_matmul", "names")}
    print(json.dumps(brief, indent=2))
    for r in summary.get("reformats", [])[:12]:
        print(f'{r["elements"]:>10d}  {r["name"][:60]:60s} ' + " | ".join(f'{t["format"]}' for t in r["inputs"]) + "  ->  " + " | ".join(f'{t["format"]}' for t in r["outputs"]))
    for p in summary.get("plugins", [])[:8]:
        print(f'plugin {p["name"][:40]:40s} in: ' + " | ".join(f'{t["format"]}' for t in p["inputs"]) + "  out: " + " | ".join(f'{t["format"]}' for t in p["outputs"]))


if __name__ == "__main__":
    main(*sys.argv[1:5])
