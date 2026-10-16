// Public entry point of the Gaia-to-TensorRT plugin library.
//
// Four IPluginV3 plugins wrap the Gaia-generated CUDA kernels of the
// EfficientViT-B1 kernel.py (see gaia_device_code.cuh): the fused depthwise
// convolution, the BN/bias + hardswish + residual epilogue, and the two
// LiteMLA attention layout kernels. Everything else in the model is native
// TensorRT (convolutions and batched matrix multiplies).
//
// All plugins accept FP32 tensors in linear (row-major NCHW) layout only, with
// static shapes; anything else is rejected at build time (and again at
// deserialization) with a non-zero status and a message on stderr.
//
// Names, version, namespace, and field names must agree with
// exporter/plugin_spec.py; a CPU test checks this file's constants against it.
#pragma once

#include <cstdint>

namespace gaia_trt {

constexpr char const* kPluginNamespace = "gaia_effvit";
constexpr char const* kPluginVersion = "1";

constexpr char const* kDepthwiseName = "GaiaDepthwiseHswish";
constexpr char const* kEpilogueName = "GaiaEpilogue";
constexpr char const* kAttnPrepName = "GaiaAttnPrep";
constexpr char const* kAttnFinalName = "GaiaAttnFinal";

// Plugin fields (all INT32, length 1). Inputs follow the field values:
//   GaiaDepthwiseHswish: k, s, pre, act, emit, in_mode, out_mode
//       inputs: x[B,C,H,W], w[C,1,k,k], in-affine tensors, out-affine tensors
//       outputs: y[B,C,Ho,Wo] and, when emit=1, ya[B,C,H,W] (the pre-activated input tile)
//   GaiaEpilogue: mode, act, has_res
//       inputs: y[B,C,H,W], affine tensors, [res[B,C,H,W]]   output: out[B,C,H,W]
//   GaiaAttnPrep: (no fields)
//       inputs: qkv[B,48*Hh,H,W], agg[B,48*Hh,H,W]   outputs: q[BH,16,N], k[BH,16,N], vpad[BH,N,17]
//   GaiaAttnFinal: batch, height, width
//       inputs: r[BH,N,17]   output: out[batch, (BH/batch)*16, height, width]
// An affine "mode" is Gaia's: 2 = batch norm (gamma, beta, mean, invstd; four [C] tensors),
// 1 = bias (beta; one [C] tensor), 0 = none.
constexpr char const* kDepthwiseFields[] = {"k", "s", "pre", "act", "emit", "in_mode", "out_mode"};
constexpr char const* kEpilogueFields[] = {"mode", "act", "has_res"};
constexpr char const* kAttnFinalFields[] = {"batch", "height", "width"};

}  // namespace gaia_trt

// Registers the four creators with TensorRT's global plugin registry under
// kPluginNamespace. Idempotent; returns false if any registration failed.
extern "C" bool gaia_trt_register_plugins();
