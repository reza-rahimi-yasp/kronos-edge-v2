// Public entry point of the whole-model plugin library.
//
// One IPluginV3 plugin, GaiaEffVitWholeModel, runs the entire forward of the
// Gaia-generated kernel.py (EfficientViT-B1) for one static FP32 input: the
// four Gaia CUDA kernels exactly as generated (the epilogue in place), the
// convolutions through cuDNN's backend graph API and the batched matrix
// multiplies through cuBLAS, both driven the way torch 2.14.1 drives them for
// F.conv2d and torch.bmm. The plugin's fields carry the op program and every
// weight of the model; each execution context gets its own cuDNN and cuBLAS
// handles, execution plans, activation arena and workspaces.
//
// The namespace, name, version, field names and the registration symbol are
// distinct from the hybrid experiment's library (gaia_plugins.h), so both can
// be loaded in one process; a CPU test checks this header against
// exporter/whole_plugin_spec.py.
#pragma once

#include <cstdint>

namespace gaia_trt_whole {

constexpr char const* kPluginNamespace = "gaia_effvit_whole";
constexpr char const* kPluginVersion = "1";
constexpr char const* kPluginName = "GaiaEffVitWholeModel";

// Fields: program (INT32[n], exporter/whole_program.py), weights (FLOAT32[n_floats]),
// cudnn_tf32 (INT32[1]: allow TF32 engines for the convolutions, torch.backends.cudnn.allow_tf32),
// cublas_tf32 (INT32[1]: CUBLAS_TF32_TENSOR_OP_MATH for the matmuls, torch.backends.cuda.matmul.allow_tf32).
constexpr char const* kPluginFields[] = {"program", "weights", "cudnn_tf32", "cublas_tf32"};

// Program encoding (must equal exporter/whole_program.py).
constexpr int32_t kProgramMagic = 0x47574D31;
constexpr int32_t kProgramVersion = 1;
constexpr int32_t kHeaderWords = 10;
constexpr int32_t kTensorWords = 6;
constexpr int32_t kOpWords = 20;
constexpr int32_t kConv = 1, kDw = 2, kEpilogue = 3, kAttnPrep = 4, kBmm = 5, kAttnFinal = 6;

}  // namespace gaia_trt_whole

// Registers the creator with TensorRT's global plugin registry under
// kPluginNamespace. Idempotent; returns false if the registration failed.
extern "C" bool gaia_trt_whole_model_register_plugins();
