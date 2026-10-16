// Public entry point of the layout-aware Gaia-to-TensorRT plugin library.
//
// Four IPluginV3 plugins wrap the Gaia-generated CUDA kernels of the
// EfficientViT-B1 kernel.py (see gaia_device_code.cuh): the fused depthwise
// convolution, the BN/bias + hardswish + residual epilogue, and the two
// LiteMLA attention layout kernels. Everything else in the model is native
// TensorRT (convolutions and batched matrix multiplies).
//
// All plugins take FP32 tensors with static shapes. Activation tensors (a
// plugin's 4-D input, its residual, its 4-D outputs) may be in the linear
// NCHW layout, served by Gaia's kernels, or in TensorRT's kHWC layout, served
// by the channel-last counterparts in gaia_hwc_kernels.cuh; within one
// plugin they share a layout. Weights, affine vectors and the 3-D attention
// operands between the matmuls are always linear. TensorRT picks the layout
// per plugin when it builds the engine; the plugin reads it from the tensor
// descriptors at run time. Anything else is rejected at build time (and
// again at deserialization) with a non-zero status and a message on stderr.
//
// This library is the layout experiment's: namespace gaia_effvit_layout and
// its own registration symbol, so that it and the baseline library
// (gaia_effvit, libgaia_trt_plugins.so) load into one process side by side.
// Names, version, namespace, and field names must agree with
// exporter/plugin_spec.py; a CPU test checks this file's constants against it.
#pragma once

#include <cstdint>

namespace gaia_trt {

constexpr char const* kPluginNamespace = "gaia_effvit_layout";
constexpr char const* kPluginVersion = "1";

constexpr char const* kDepthwiseName = "GaiaDepthwiseHswish";
constexpr char const* kEpilogueName = "GaiaEpilogue";
constexpr char const* kAttnPrepName = "GaiaAttnPrep";
constexpr char const* kAttnFinalName = "GaiaAttnFinal";

// Plugin fields (all INT32, length 1). Inputs follow the field values:
//   GaiaDepthwiseHswish: k, s, pre, act, emit, in_mode, out_mode, layout
//       inputs: x[B,C,H,W], w[C,1,k,k], in-affine tensors, out-affine tensors
//       outputs: y[B,C,Ho,Wo] and, when emit=1, ya[B,C,H,W] (the pre-activated input tile)
//   GaiaEpilogue: mode, act, has_res, layout
//       inputs: y[B,C,H,W], affine tensors, [res[B,C,H,W]]   output: out[B,C,H,W]
//   GaiaAttnPrep: layout
//       inputs: qkv[B,48*Hh,H,W], agg[B,48*Hh,H,W]   outputs: q[BH,16,N], k[BH,16,N], vpad[BH,N,17]
//   GaiaAttnFinal: batch, height, width, layout
//       inputs: r[BH,N,17]   output: out[batch, (BH/batch)*16, height, width]
// An affine "mode" is Gaia's: 2 = batch norm (gamma, beta, mean, invstd; four [C] tensors),
// 1 = bias (beta; one [C] tensor), 0 = none.
// "layout" says which layouts the plugin offers TensorRT for its activation tensors:
// 0 = linear or kHWC (TensorRT chooses by timing), 1 = linear only (the baseline's behaviour), 2 = kHWC only.
constexpr int32_t kLayoutAny = 0, kLayoutLinear = 1, kLayoutHwc = 2;
constexpr char const* kDepthwiseFields[] = {"k", "s", "pre", "act", "emit", "in_mode", "out_mode", "layout"};
constexpr char const* kEpilogueFields[] = {"mode", "act", "has_res", "layout"};
constexpr char const* kAttnPrepFields[] = {"layout"};
constexpr char const* kAttnFinalFields[] = {"batch", "height", "width", "layout"};

}  // namespace gaia_trt

// Registers the four creators with TensorRT's global plugin registry under
// kPluginNamespace. Idempotent; returns false if any registration failed.
// The only symbol the library exports (it is built with hidden visibility).
extern "C" __attribute__((visibility("default"))) bool gaia_trt_layout_register_plugins();
