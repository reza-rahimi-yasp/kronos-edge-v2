"""The engine inspector summary reads TensorRT's detailed layer JSON: reformats, plugin boundaries, native formats."""

# The exporter modules are plain scripts, not a package: make them importable from this test alone.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "exporter"))

import inspect_engine

NHWC, LINEAR = "(NHWC) Channel major format", "(Linear) Row major linear format"


def _tensor(name, dims, fmt):
    return {"Name": name, "Dimensions": dims, "Datatype": "Float", "Format": fmt}


LAYERS = [
    {"Name": "__mye1_myl0_0", "LayerType": "custom_layer", "Inputs": [_tensor("l2_cache_policy_inputs_0", [8], LINEAR)], "Outputs": []},
    {"Name": "__myl_Move_myl0_1", "LayerType": "kgen", "Inputs": [_tensor("x", [4, 3, 8, 8], LINEAR)], "Outputs": [_tensor("x_nhwc", [4, 3, 8, 8], NHWC)]},
    {"Name": "conv_1_myl0_2", "LayerType": "correlation", "Inputs": [_tensor("x_nhwc", [4, 3, 8, 8], NHWC)], "Outputs": [_tensor("c1", [4, 16, 4, 4], NHWC)],
     "TacticName": "sm80_xmma_fprop_implicit_gemm_nhwckrsc_nhwc"},
    {"Name": "__myl_Tran_myl0_3", "LayerType": "kgen", "Inputs": [_tensor("c1", [4, 16, 4, 4], NHWC)], "Outputs": [_tensor("c1_lin", [4, 16, 4, 4], LINEAR)]},
    {"Name": "dw_2_myl0_4", "LayerType": "custom_layer", "Inputs": [_tensor("c1_lin", [4, 16, 4, 4], LINEAR)],
     "Outputs": [_tensor("d", [4, 16, 4, 4], LINEAR), _tensor("da", [4, 16, 4, 4], LINEAR)]},
    {"Name": "kv_36_myl0_65", "LayerType": "gemm", "Inputs": [_tensor("k", [64, 16, 16], LINEAR), _tensor("v", [64, 16, 17], LINEAR)], "Outputs": [_tensor("kv", [64, 16, 17], LINEAR)]},
    {"Name": "__myl_AddMul_myl0_9", "LayerType": "kgen", "Inputs": [_tensor("e", [4, 16, 4, 4], NHWC)], "Outputs": [_tensor("f", [4, 16, 4, 4], NHWC)]},
]


def test_summary_counts_reformats_and_reports_plugin_boundary_formats():
    summary = inspect_engine.summarize(LAYERS)
    assert summary["detailed"] is True and summary["num_layers"] == 7
    assert summary["reformat_layers"] == 2  # the input Move and the Tran before the plugin; the fused elementwise kgen is not one
    assert [r["name"] for r in summary["reformats"]] == ["__myl_Tran_myl0_3", "__myl_Move_myl0_1"]  # largest first
    assert summary["reformat_elements_total"] == 4 * 16 * 4 * 4 + 4 * 3 * 8 * 8
    assert summary["reformats"][0]["inputs"][0]["format"] == NHWC and summary["reformats"][0]["outputs"][0]["format"] == LINEAR
    assert summary["plugin_layers"] == 1  # the L2 cache policy custom layer is TensorRT's, not a plugin
    assert summary["plugins"][0]["name"] == "dw_2_myl0_4"
    assert summary["formats_at_plugin_boundaries"] == {LINEAR: 3}
    assert {n["type"] for n in summary["native_conv_matmul"]} == {"correlation", "gemm"}
    assert summary["native_conv_matmul"][0]["outputs"][0]["format"] == NHWC


def test_summary_of_a_default_verbosity_engine_lists_names_only():
    summary = inspect_engine.summarize(["conv_1_myl0_2", "__myl_Tran_myl0_3", "dw_2_myl0_4"])
    assert summary["detailed"] is False and summary["num_layers"] == 3
    assert summary["reformat_like_names"] == ["__myl_Tran_myl0_3"]
