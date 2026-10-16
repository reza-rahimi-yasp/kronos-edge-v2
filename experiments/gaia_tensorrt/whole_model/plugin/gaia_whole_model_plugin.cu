// GaiaEffVitWholeModel: one IPluginV3 that runs the whole forward of the
// Gaia-generated kernel.py. See gaia_whole_model_plugin.h.
//
// Structure
//   Model      the plugin's fields, decoded and validated once (program, weights,
//              TF32 switches); shared by every clone through a shared_ptr.
//   Context    what one execution context owns: cuDNN and cuBLAS handles, the
//              weights on the device, the activation arena, the workspaces and
//              one cuDNN execution plan per distinct convolution. Created by
//              attachToContext (build and run time), freed with the clone.
//   enqueue    walks the program on TensorRT's stream: the Gaia kernels as
//              generated (the epilogue in place), cuDNN for the convolutions,
//              cuBLAS for the batched matmuls.
//
// What is mirrored from torch 2.14.1+cu130, measured with CUDNN_LOGLEVEL_DBG=3
// and CUBLAS_LOGINFO_DBG=1 on the unmodified kernel.py (the probe logs in the
// experiment's evidence):
//   F.conv2d  -> cuDNN backend graph API: FP32 NCHW tensors with 32-byte
//                alignment, compute type float, cross-correlation, alpha 1 beta 0,
//                engine heuristics CUDNN_HEUR_MODE_INSTANT then the fallback
//                list; configs with CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS are
//                skipped, and with CUDNN_NUMERICAL_NOTE_TENSOR_CORE when
//                torch.backends.cudnn.allow_tf32 is off, and engines with
//                CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION are skipped (torch
//                builds their plans and never executes them; this runtime has
//                no NVRTC); the first remaining config whose plan finalizes and
//                executes is kept, one plan per distinct convolution (shapes,
//                stride, padding, groups). ATen's hardcoded errata for this GPU
//                (engines 58 and 63 on compute capability 12.0 with cuBLAS < 13.6.1)
//                is applied too; torch first tries the top heuristic config and
//                then the full list, which selects the same config as one walk.
//   torch.bmm -> cublasSgemmStridedBatched on a handle with a 32 MiB workspace
//                and CUBLAS_DEFAULT_MATH (CUBLAS_TF32_TENSOR_OP_MATH when
//                torch.backends.cuda.matmul.allow_tf32 is on), the operand
//                roles swapped for row-major storage exactly as ATen does
//                (C^T = B^T A^T), the transposed q passed with CUBLAS_OP_T.
// Every interface method is noexcept; bodies that can throw are wrapped and
// report failure through the method's return convention with a reason on stderr.

#include "gaia_whole_model_plugin.h"

#include <NvInferRuntime.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace gaia_trt_whole {

// The Gaia device code, byte for byte (the hybrid experiment's extracted header), inside this
// library's namespace so its kernel symbols cannot collide with the hybrid plugin library's.
// <cuda_runtime.h> is already included above, so the header's own include is a no-op here.
#include "gaia_device_code.cuh"

using namespace nvinfer1;

namespace {

constexpr size_t kCublasWorkspaceBytes = 32 * 1024 * 1024;  // torch: cublasSetWorkspace(..., 33554432) on this GPU
constexpr int64_t kTensorAlignment = 32;                     // torch: CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT 32
constexpr size_t kBufferAlignment = 256;                     // arena buffers (cudaMalloc alignment)
constexpr int32_t kNone = -1;

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(std::string const& what) {
    throw Failure(what);
}

void cudaCheck(cudaError_t err, char const* what) {
    if (err != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(err));
}

void cudnnCheck(cudnnStatus_t st, char const* what) {
    if (st != CUDNN_STATUS_SUCCESS) fail(std::string(what) + ": " + cudnnGetErrorString(st));
}

void cublasCheck(cublasStatus_t st, char const* what) {
    if (st != CUBLAS_STATUS_SUCCESS) fail(std::string(what) + ": cuBLAS status " + std::to_string(static_cast<int>(st)));
}

bool verbose() {
    static bool on = std::getenv("GAIA_TRT_WHOLE_LOG_PLANS") != nullptr;
    return on;
}

size_t roundUp(size_t v, size_t a) {
    return (v + a - 1) / a * a;
}

// ---------------------------------------------------------------------------
// Model: the decoded program and the weights, validated once.

struct TensorInfo {
    int32_t ndim = 0;
    int64_t d[4] = {0, 0, 0, 0};
    int32_t alias = kNone;
    int64_t numel = 0;
};

struct Model {
    std::vector<int32_t> program;
    std::vector<float> weights;
    int32_t cudnnTf32 = 1;
    int32_t cublasTf32 = 0;
    int32_t nTensors = 0, nOps = 0, nFloats = 0, inputId = 0, outputId = 0;
    std::vector<TensorInfo> tensors;
    std::vector<int32_t const*> ops;  // kOpWords each, into program

    int32_t base(int32_t t) const {
        while (tensors[t].alias != kNone) t = tensors[t].alias;
        return t;
    }

    TensorInfo const& T(int32_t t) const { return tensors[t]; }

    void decode() {
        if (program.size() < (size_t) kHeaderWords) fail("program too short");
        if (program[0] != kProgramMagic || program[1] != kProgramVersion) fail("not a whole-model program");
        nTensors = program[2]; nOps = program[3]; nFloats = program[4]; inputId = program[5]; outputId = program[6];
        if (program[7] != kTensorWords || program[8] != kOpWords) fail("program record width mismatch");
        if (nTensors <= 0 || nOps <= 0 || nFloats <= 0) fail("empty program");
        size_t expected = (size_t) kHeaderWords + (size_t) nTensors * kTensorWords + (size_t) nOps * kOpWords;
        if (program.size() != expected) fail("program length does not match its header");
        if (weights.size() != (size_t) nFloats) fail("weights field has " + std::to_string(weights.size()) + " floats, program expects " + std::to_string(nFloats));
        if (cudnnTf32 != 0 && cudnnTf32 != 1) fail("cudnn_tf32 must be 0 or 1");
        if (cublasTf32 != 0 && cublasTf32 != 1) fail("cublas_tf32 must be 0 or 1");
        tensors.resize(nTensors);
        int32_t const* p = program.data() + kHeaderWords;
        for (int32_t i = 0; i < nTensors; ++i, p += kTensorWords) {
            TensorInfo& t = tensors[i];
            t.ndim = p[0];
            if (t.ndim < 1 || t.ndim > 4) fail("tensor rank must be 1..4");
            t.numel = 1;
            for (int32_t k = 0; k < t.ndim; ++k) {
                t.d[k] = p[1 + k];
                if (t.d[k] <= 0) fail("tensor has an empty dimension");
                t.numel *= t.d[k];
            }
            t.alias = p[5];
            if (t.alias != kNone && (t.alias < 0 || t.alias >= i)) fail("alias must name an earlier tensor");
            if (t.alias != kNone && tensors[t.alias].numel != t.numel) fail("alias size mismatch");
        }
        if (inputId < 0 || inputId >= nTensors || outputId < 0 || outputId >= nTensors) fail("input/output id out of range");
        if (tensors[inputId].alias != kNone || tensors[inputId].ndim != 4) fail("the input must be an unaliased 4-D tensor");
        if (base(outputId) == inputId) fail("the output cannot be the input buffer");
        ops.resize(nOps);
        for (int32_t i = 0; i < nOps; ++i, p += kOpWords) ops[i] = p;
        validateOps();
    }

    void ref(int32_t t, char const* what) const {
        if (t < 0 || t >= nTensors) fail(std::string("tensor reference out of range: ") + what);
    }

    void weight(int32_t off, int64_t count, char const* what) const {
        if (off < 0 || count <= 0 || (int64_t) off + count > nFloats || off % 16 != 0) fail(std::string("weight offset out of range: ") + what);
    }

    void affine(int32_t const* a, int64_t c, char const* what) const {  // mode, g, b, m, is
        int32_t mode = a[0];
        if (mode == 2) {
            for (int k = 1; k <= 4; ++k) weight(a[k], c, what);
        } else if (mode == 1) {
            weight(a[2], c, what);
            if (a[1] != kNone || a[3] != kNone || a[4] != kNone) fail(std::string("bias affine with extra vectors: ") + what);
        } else if (mode == 0) {
            for (int k = 1; k <= 4; ++k)
                if (a[k] != kNone) fail(std::string("affine mode 0 with vectors: ") + what);
        } else {
            fail(std::string("affine mode must be 0, 1 or 2: ") + what);
        }
    }

    static bool same4(TensorInfo const& a, TensorInfo const& b) {
        if (a.ndim != b.ndim) return false;
        for (int k = 0; k < a.ndim; ++k)
            if (a.d[k] != b.d[k]) return false;
        return true;
    }

    void expect(int32_t t, std::initializer_list<int64_t> dims, char const* what) const {
        TensorInfo const& ti = T(t);
        if (ti.ndim != (int32_t) dims.size()) fail(std::string("rank mismatch: ") + what);
        int k = 0;
        for (int64_t d : dims)
            if (ti.d[k++] != d) fail(std::string("shape mismatch: ") + what);
    }

    // Recompute every op's output shape from its inputs and require the program's recorded shapes;
    // check every weight offset. The exporter did the same without torch; this is the plugin's own gate.
    void validateOps() const {
        std::vector<bool> defined(nTensors, false);
        defined[inputId] = true;
        auto use = [&](int32_t t, char const* what) { ref(t, what); if (!defined[t]) fail(std::string("tensor used before definition: ") + what); };
        auto def = [&](int32_t t, char const* what) { ref(t, what); if (defined[t]) fail(std::string("tensor defined twice: ") + what); defined[t] = true; };
        for (int32_t i = 0; i < nOps; ++i) {
            int32_t const* o = ops[i];
            switch (o[0]) {
                case kConv: {
                    use(o[1], "conv x"); def(o[2], "conv out");
                    TensorInfo const& x = T(o[1]);
                    int64_t cout = o[4], cinG = o[5], k = o[6], s = o[7], pad = o[8], groups = o[9];
                    if (x.ndim != 4 || cout <= 0 || cinG <= 0 || k <= 0 || s <= 0 || pad < 0 || groups <= 0) fail("conv parameters");
                    if (cinG * groups != x.d[1] || cout % groups != 0) fail("conv channel/group mismatch");
                    weight(o[3], cout * cinG * k * k, "conv w");
                    int64_t ho = (x.d[2] + 2 * pad - k) / s + 1, wo = (x.d[3] + 2 * pad - k) / s + 1;
                    if (ho <= 0 || wo <= 0) fail("conv input smaller than the kernel");
                    expect(o[2], {x.d[0], cout, ho, wo}, "conv out");
                    if (T(o[2]).alias != kNone) fail("conv output must own its buffer");
                    break;
                }
                case kDw: {
                    use(o[1], "dw x"); def(o[2], "dw out");
                    TensorInfo const& x = T(o[1]);
                    int64_t k = o[4], s = o[5], pre = o[6], act = o[7], emit = o[8];
                    if (x.ndim != 4) fail("dw x must be 4-D");
                    bool config = (k == 3 && s == 1) || (k == 3 && s == 2) || (k == 5 && s == 1);
                    if (!config) fail("dw config (k, s) must be (3,1), (3,2) or (5,1)");
                    if ((pre != 0 && pre != 1) || (act != 0 && act != 1) || (emit != 0 && emit != 1)) fail("dw flags");
                    if (emit && s != 1) fail("emit requires stride 1");
                    int64_t C = x.d[1];
                    if (C > 65535 || x.d[0] > 65535) fail("dw: C and B must fit a CUDA grid dimension");
                    weight(o[3], C * k * k, "dw w");
                    affine(o + 10, C, "dw in affine");
                    affine(o + 15, C, "dw out affine");
                    int64_t P = k / 2, ho = (x.d[2] + 2 * P - k) / s + 1, wo = (x.d[3] + 2 * P - k) / s + 1;
                    if (ho <= 0 || wo <= 0) fail("dw input smaller than the kernel");
                    expect(o[2], {x.d[0], C, ho, wo}, "dw out");
                    if (T(o[2]).alias != kNone) fail("dw output must own its buffer");
                    if (emit) {
                        def(o[9], "dw out_pre");
                        if (!same4(T(o[9]), x) || T(o[9]).alias != kNone) fail("dw out_pre must match x");
                    } else if (o[9] != kNone) {
                        fail("dw out_pre without emit");
                    }
                    break;
                }
                case kEpilogue: {
                    use(o[1], "epilogue x"); def(o[2], "epilogue out");
                    TensorInfo const& x = T(o[1]);
                    if (x.ndim != 4) fail("epilogue x must be 4-D");
                    if (T(o[2]).alias != o[1]) fail("epilogue output must alias its input (in place)");
                    if (!same4(T(o[2]), x)) fail("epilogue out shape");
                    int64_t HW = x.d[2] * x.d[3];
                    if (HW % 4 != 0) fail("epilogue: H*W must be a multiple of 4");
                    if (x.numel / 4 >= (1LL << 31)) fail("epilogue: tensor too large");
                    affine(o + 3, x.d[1], "epilogue affine");
                    if (o[8] != 0 && o[8] != 1) fail("epilogue act");
                    if (o[9] != kNone) {
                        use(o[9], "epilogue res");
                        if (!same4(T(o[9]), x)) fail("epilogue residual shape");
                        if (base(o[9]) == base(o[1])) fail("epilogue residual aliases its tensor");
                    }
                    break;
                }
                case kAttnPrep: {
                    use(o[1], "attn_prep qkv"); use(o[2], "attn_prep agg");
                    TensorInfo const& a = T(o[1]);
                    if (a.ndim != 4 || !same4(a, T(o[2]))) fail("attn_prep: qkv and agg must be equal 4-D tensors");
                    if (a.d[1] % 48 != 0) fail("attn_prep: channels must be a multiple of 48");
                    int64_t BH = a.d[0] * 2 * (a.d[1] / 48), N = a.d[2] * a.d[3];
                    if (BH > 65535 || (N + 31) / 32 > 2147483647) fail("attn_prep: grid too large");
                    def(o[3], "q"); def(o[4], "k"); def(o[5], "vpad");
                    expect(o[3], {BH, 16, N}, "q"); expect(o[4], {BH, 16, N}, "k"); expect(o[5], {BH, N, 17}, "vpad");
                    for (int k = 3; k <= 5; ++k)
                        if (T(o[k]).alias != kNone) fail("attn_prep outputs must own their buffers");
                    break;
                }
                case kBmm: {
                    use(o[1], "bmm a"); use(o[2], "bmm b"); def(o[3], "bmm out");
                    TensorInfo const& a = T(o[1]);
                    TensorInfo const& b = T(o[2]);
                    if (a.ndim != 3 || b.ndim != 3 || a.d[0] != b.d[0]) fail("bmm operands must be 3-D with equal batch");
                    if (o[4] != 0 && o[4] != 1) fail("bmm transpose flag");
                    int64_t M = o[4] ? a.d[2] : a.d[1], K = o[4] ? a.d[1] : a.d[2];
                    if (K != b.d[1]) fail("bmm inner dimension mismatch");
                    expect(o[3], {a.d[0], M, b.d[2]}, "bmm out");
                    if (T(o[3]).alias != kNone) fail("bmm output must own its buffer");
                    for (int64_t v : {a.d[0], M, K, b.d[2], M * K, K * b.d[2], M * b.d[2]})
                        if (v > 2147483647) fail("bmm dimension does not fit cuBLAS int");
                    break;
                }
                case kAttnFinal: {
                    use(o[1], "attn_final r"); use(o[2], "attn_final ref"); def(o[3], "attn_final out");
                    TensorInfo const& r = T(o[1]);
                    TensorInfo const& refT = T(o[2]);
                    if (r.ndim != 3 || r.d[2] != 17 || refT.ndim != 4) fail("attn_final: r must be [B*heads, N, 17]");
                    int64_t BH = r.d[0], N = r.d[1], B = refT.d[0];
                    if (N != refT.d[2] * refT.d[3] || BH % B != 0 || BH > 65535) fail("attn_final: r does not match the reference shape");
                    expect(o[3], {B, (BH / B) * 16, refT.d[2], refT.d[3]}, "attn_final out");
                    if (T(o[3]).alias != kNone) fail("attn_final output must own its buffer");
                    break;
                }
                default:
                    fail("unknown op kind " + std::to_string(o[0]));
            }
        }
        for (int32_t t = 0; t < nTensors; ++t)
            if (!defined[t]) fail("tensor " + std::to_string(t) + " is never defined");
        if (!defined[outputId]) fail("the output is never produced");
    }
};

// ---------------------------------------------------------------------------
// cuDNN backend descriptors (RAII) and the convolution plan selection.

struct Descriptor {
    cudnnBackendDescriptor_t d = nullptr;
    Descriptor() = default;
    explicit Descriptor(cudnnBackendDescriptorType_t type) { cudnnCheck(cudnnBackendCreateDescriptor(type, &d), "cudnnBackendCreateDescriptor"); }
    Descriptor(Descriptor&& o) noexcept : d(o.d) { o.d = nullptr; }
    Descriptor& operator=(Descriptor&& o) noexcept { reset(); d = o.d; o.d = nullptr; return *this; }
    Descriptor(Descriptor const&) = delete;
    Descriptor& operator=(Descriptor const&) = delete;
    ~Descriptor() { reset(); }
    void reset() { if (d) cudnnBackendDestroyDescriptor(d); d = nullptr; }
    void set(cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type, int64_t count, void const* values, char const* what) {
        cudnnCheck(cudnnBackendSetAttribute(d, name, type, count, const_cast<void*>(values)), what);
    }
    void finalize(char const* what) { cudnnCheck(cudnnBackendFinalize(d), what); }
};

Descriptor tensorDesc(int64_t const dims[4], int64_t uid) {
    Descriptor t(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    int64_t strides[4] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    t.set(CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype, "tensor data type");
    t.set(CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims, "tensor dimensions");
    t.set(CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides, "tensor strides");
    t.set(CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid, "tensor uid");
    t.set(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &kTensorAlignment, "tensor alignment");
    t.finalize("tensor finalize");
    return t;
}

// ATen's hardcoded cuDNN errata (Conv_v8.cpp, plan_errata_exception, issue 188288): on compute
// capability 12.0 with a cuBLAS older than 13.6.1, cuDNN engines 58 and 63 may dispatch to a cuBLASLt
// kernel with wrong behaviour, so their plans are never executed. torch's log for kernel.py shows
// exactly that (engine 58 plans built and skipped); the plugin applies the same rule.
struct Errata {
    bool active = false;
    Errata() {
        int device = 0, major = 0, minor = 0, cublas = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) return;
        if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) return;
        if (cublasGetVersion(nullptr, &cublas) != CUBLAS_STATUS_SUCCESS) return;
        active = (major * 100 + minor * 10 == 1200) && cublas < 130601;
    }
    bool skips(int64_t engineIndex) const { return active && (engineIndex == 58 || engineIndex == 63); }
};

Errata const& errata() {
    static Errata e;
    return e;
}

struct ConvKey {
    int64_t x[4], w[4], stride, pad;
    bool operator<(ConvKey const& o) const {
        return std::tie(x[0], x[1], x[2], x[3], w[0], w[1], w[2], w[3], stride, pad) <
               std::tie(o.x[0], o.x[1], o.x[2], o.x[3], o.w[0], o.w[1], o.w[2], o.w[3], o.stride, o.pad);
    }
};

// One execution plan for one distinct convolution, with the descriptors it was built from.
struct ConvPlan {
    Descriptor x, w, y, conv, op, graph, cfg, plan;
    size_t workspace = 0;
    int64_t engineIndex = -1;
    std::string knobs;
    bool tensorCore = false;
};

enum { kUidX = 1, kUidW = 2, kUidY = 3 };

bool hasNote(Descriptor const& engine, cudnnBackendNumericalNote_t note) {
    cudnnBackendNumericalNote_t notes[CUDNN_NUMERICAL_NOTE_TYPE_COUNT];
    int64_t n = 0;
    cudnnCheck(cudnnBackendGetAttribute(engine.d, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE, CUDNN_NUMERICAL_NOTE_TYPE_COUNT, &n, notes), "engine numerical notes");
    for (int64_t i = 0; i < n; ++i)
        if (notes[i] == note) return true;
    return false;
}

bool hasBehavior(Descriptor const& engine, cudnnBackendBehaviorNote_t note) {
    cudnnBackendBehaviorNote_t notes[CUDNN_BEHAVIOR_NOTE_TYPE_COUNT];
    int64_t n = 0;
    cudnnCheck(cudnnBackendGetAttribute(engine.d, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE, CUDNN_BEHAVIOR_NOTE_TYPE_COUNT, &n, notes), "engine behavior notes");
    for (int64_t i = 0; i < n; ++i)
        if (notes[i] == note) return true;
    return false;
}

// "n:<numerical notes>/b:<behavior notes>" of an engine, for the plan log.
std::string noteText(Descriptor const& engine) {
    cudnnBackendNumericalNote_t num[CUDNN_NUMERICAL_NOTE_TYPE_COUNT];
    cudnnBackendBehaviorNote_t beh[CUDNN_BEHAVIOR_NOTE_TYPE_COUNT];
    int64_t nn = 0, nb = 0;
    cudnnCheck(cudnnBackendGetAttribute(engine.d, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE, CUDNN_NUMERICAL_NOTE_TYPE_COUNT, &nn, num), "engine numerical notes");
    cudnnCheck(cudnnBackendGetAttribute(engine.d, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE, CUDNN_BEHAVIOR_NOTE_TYPE_COUNT, &nb, beh), "engine behavior notes");
    std::string t = "n:";
    for (int64_t i = 0; i < nn; ++i) t += (i ? "," : "") + std::to_string(static_cast<int>(num[i]));
    t += "/b:";
    for (int64_t i = 0; i < nb; ++i) t += (i ? "," : "") + std::to_string(static_cast<int>(beh[i]));
    return t;
}

std::string knobText(Descriptor const& cfg) {
    constexpr int64_t kMax = CUDNN_KNOB_TYPE_COUNTS;
    std::vector<Descriptor> knobs;
    std::vector<cudnnBackendDescriptor_t> raw;
    for (int64_t i = 0; i < kMax; ++i) {
        knobs.emplace_back(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
        raw.push_back(knobs.back().d);
    }
    int64_t n = 0;
    cudnnCheck(cudnnBackendGetAttribute(cfg.d, CUDNN_ATTR_ENGINECFG_KNOB_CHOICES, CUDNN_TYPE_BACKEND_DESCRIPTOR, kMax, &n, raw.data()), "knob choices");
    std::string text;
    for (int64_t i = 0; i < n; ++i) {
        cudnnBackendKnobType_t type;
        int64_t value = 0, c = 0;
        cudnnCheck(cudnnBackendGetAttribute(raw[i], CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &c, &type), "knob type");
        cudnnCheck(cudnnBackendGetAttribute(raw[i], CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, &c, &value), "knob value");
        text += (i ? " " : "") + std::to_string(static_cast<int>(type)) + "=" + std::to_string(value);
    }
    return text;
}

// The engine configs a heuristics mode proposes for the graph, in its order.
std::vector<Descriptor> heuristicConfigs(Descriptor const& graph, cudnnBackendHeurMode_t mode) {
    Descriptor heur(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    heur.set(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph.d, "heuristics graph");
    heur.set(CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode, "heuristics mode");
    heur.finalize("heuristics finalize");
    int64_t count = 0;
    cudnnCheck(cudnnBackendGetAttribute(heur.d, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr), "heuristics result count");
    std::vector<Descriptor> configs;
    std::vector<cudnnBackendDescriptor_t> raw;
    for (int64_t i = 0; i < count; ++i) {
        configs.emplace_back(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        raw.push_back(configs.back().d);
    }
    int64_t got = 0;
    if (count > 0) cudnnCheck(cudnnBackendGetAttribute(heur.d, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, count, &got, raw.data()), "heuristics results");
    configs.resize(got);
    return configs;
}

// ---------------------------------------------------------------------------
// Context: the per-execution-context resources.

struct Context {
    std::shared_ptr<Model const> model;
    cudnnHandle_t cudnn = nullptr;
    cublasHandle_t cublas = nullptr;
    float* weights = nullptr;
    char* arena = nullptr;
    size_t arenaBytes = 0;
    void* cudnnWorkspace = nullptr;
    size_t cudnnWorkspaceBytes = 0;
    void* cublasWorkspace = nullptr;
    std::vector<int64_t> offsets;                 // per tensor base: byte offset in the arena, or -1 for the input/output
    std::map<ConvKey, std::unique_ptr<ConvPlan>> plans;
    struct ConvCall {                              // per conv op: its plan and the variant pack for its pointers
        ConvPlan* plan = nullptr;
        Descriptor varpack;
        void const* x = nullptr; void const* w = nullptr; void* y = nullptr; void* ws = nullptr;
    };
    std::vector<ConvCall> calls;                   // indexed by op

    explicit Context(std::shared_ptr<Model const> m) : model(std::move(m)) {
        try {
            setup();
        } catch (...) {
            release();
            throw;
        }
    }
    ~Context() { release(); }

    void release() noexcept {
        calls.clear();
        plans.clear();
        if (cublasWorkspace) cudaFree(cublasWorkspace);
        if (cudnnWorkspace) cudaFree(cudnnWorkspace);
        if (arena) cudaFree(arena);
        if (weights) cudaFree(weights);
        if (cublas) cublasDestroy(cublas);
        if (cudnn) cudnnDestroy(cudnn);
        cublasWorkspace = cudnnWorkspace = nullptr; arena = nullptr; weights = nullptr; cublas = nullptr; cudnn = nullptr;
    }

    // Arena: one buffer per tensor that owns storage, reused once its last reader has run
    // (a caching-allocator-like free list, decided once here).
    void planArena() {
        Model const& m = *model;
        offsets.assign(m.nTensors, -1);
        int32_t outputBase = m.base(m.outputId);
        std::vector<int32_t> lastUse(m.nTensors, -1);
        auto touch = [&](int32_t t, int32_t i) { if (t != kNone) { int32_t b = m.base(t); lastUse[b] = std::max(lastUse[b], i); } };
        for (int32_t i = 0; i < m.nOps; ++i) {
            int32_t const* o = m.ops[i];
            switch (o[0]) {
                case kConv: touch(o[1], i); touch(o[2], i); break;
                case kDw: touch(o[1], i); touch(o[2], i); touch(o[9], i); break;
                case kEpilogue: touch(o[1], i); touch(o[2], i); touch(o[9], i); break;
                case kAttnPrep: for (int k = 1; k <= 5; ++k) touch(o[k], i); break;
                case kBmm: for (int k = 1; k <= 3; ++k) touch(o[k], i); break;
                case kAttnFinal: for (int k = 1; k <= 3; ++k) touch(o[k], i); break;
            }
        }
        std::vector<std::pair<size_t, int64_t>> freeList;  // (bytes, offset)
        size_t cursor = 0;
        auto allocate = [&](int32_t t) {
            if (t == kNone || t == m.inputId || t == outputBase || m.tensors[t].alias != kNone) return;
            size_t bytes = roundUp((size_t) m.tensors[t].numel * sizeof(float), kBufferAlignment);
            auto best = freeList.end();
            for (auto it = freeList.begin(); it != freeList.end(); ++it)
                if (it->first >= bytes && (best == freeList.end() || it->first < best->first)) best = it;
            if (best != freeList.end()) {
                offsets[t] = best->second;
                freeList.erase(best);
            } else {
                offsets[t] = (int64_t) cursor;
                cursor += bytes;
            }
        };
        std::vector<size_t> sizes(m.nTensors, 0);
        for (int32_t t = 0; t < m.nTensors; ++t) sizes[t] = roundUp((size_t) m.tensors[t].numel * sizeof(float), kBufferAlignment);
        for (int32_t i = 0; i < m.nOps; ++i) {
            int32_t const* o = m.ops[i];
            switch (o[0]) {
                case kConv: allocate(o[2]); break;
                case kDw: allocate(o[2]); allocate(o[9]); break;
                case kEpilogue: break;  // in place
                case kAttnPrep: allocate(o[3]); allocate(o[4]); allocate(o[5]); break;
                case kBmm: allocate(o[3]); break;
                case kAttnFinal: allocate(o[3]); break;
            }
            for (int32_t t = 0; t < m.nTensors; ++t)
                if (lastUse[t] == i && offsets[t] >= 0) freeList.emplace_back(sizes[t], offsets[t]);
        }
        arenaBytes = cursor;
    }

    void setup() {
        Model const& m = *model;
        cudnnCheck(cudnnCreate(&cudnn), "cudnnCreate");
        cublasCheck(cublasCreate(&cublas), "cublasCreate");
        cudaCheck(cudaMalloc(&weights, m.weights.size() * sizeof(float)), "cudaMalloc weights");
        cudaCheck(cudaMemcpy(weights, m.weights.data(), m.weights.size() * sizeof(float), cudaMemcpyHostToDevice), "upload weights");
        planArena();
        cudaCheck(cudaMalloc(&arena, std::max<size_t>(arenaBytes, 256)), "cudaMalloc arena");
        cudaCheck(cudaMalloc(&cublasWorkspace, kCublasWorkspaceBytes), "cudaMalloc cuBLAS workspace");
        // Plan selection executes each candidate once (as ATen does on the real call), so the input and
        // output, which TensorRT provides only at enqueue, get scratch buffers for the trial runs.
        void* scratchIn = nullptr; void* scratchOut = nullptr;
        cudaCheck(cudaMalloc(&scratchIn, m.tensors[m.inputId].numel * sizeof(float)), "cudaMalloc scratch input");
        cudaCheck(cudaMalloc(&scratchOut, m.tensors[m.base(m.outputId)].numel * sizeof(float)), "cudaMalloc scratch output");
        try {
            calls.resize(m.nOps);
            for (int32_t i = 0; i < m.nOps; ++i) {
                int32_t const* o = m.ops[i];
                if (o[0] != kConv) continue;
                TensorInfo const& x = m.T(o[1]);
                TensorInfo const& y = m.T(o[2]);
                ConvKey key{{x.d[0], x.d[1], x.d[2], x.d[3]}, {o[4], o[5], o[6], o[6]}, o[7], o[8]};
                auto it = plans.find(key);
                if (it == plans.end()) {
                    auto plan = buildPlan(key, x, y, resolve(o[1], scratchIn, scratchOut), weights + o[3], resolve(o[2], scratchIn, scratchOut));
                    it = plans.emplace(key, std::move(plan)).first;
                }
                calls[i].plan = it->second.get();
            }
            cudaCheck(cudaDeviceSynchronize(), "plan trial runs");
        } catch (...) {
            cudaFree(scratchIn); cudaFree(scratchOut);
            throw;
        }
        cudaFree(scratchIn); cudaFree(scratchOut);
    }

    void* resolve(int32_t t, void* input, void* output) const {
        Model const& m = *model;
        int32_t b = m.base(t);
        if (b == m.inputId) return input;
        if (b == m.base(m.outputId)) return output;
        return arena + offsets[b];
    }

    void ensureWorkspace(size_t bytes) {
        if (bytes <= cudnnWorkspaceBytes) return;
        if (cudnnWorkspace) cudaFree(cudnnWorkspace);
        cudnnWorkspace = nullptr;
        cudaCheck(cudaMalloc(&cudnnWorkspace, bytes), "cudaMalloc cuDNN workspace");
        cudnnWorkspaceBytes = bytes;
    }

    Descriptor makeVarpack(void const* x, void const* w, void* y, void* ws) {
        Descriptor vp(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        void* ptrs[3] = {const_cast<void*>(x), const_cast<void*>(w), y};
        int64_t uids[3] = {kUidX, kUidW, kUidY};
        vp.set(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, ptrs, "variant pack pointers");
        vp.set(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids, "variant pack uids");
        vp.set(CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ws, "variant pack workspace");
        vp.finalize("variant pack finalize");
        return vp;
    }

    // ATen Conv_v8: operation graph, heuristics (INSTANT, then the fallback list), filtered by
    // numerical notes, first config whose plan finalizes and executes.
    std::unique_ptr<ConvPlan> buildPlan(ConvKey const& key, TensorInfo const& xi, TensorInfo const& yi, void* xptr, float const* wptr, void* yptr) {
        auto p = std::make_unique<ConvPlan>();
        int64_t wdims[4] = {key.w[0], key.w[1], key.w[2], key.w[3]};
        p->x = tensorDesc(xi.d, kUidX);
        p->w = tensorDesc(wdims, kUidW);
        p->y = tensorDesc(yi.d, kUidY);
        p->conv = Descriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        cudnnDataType_t comp = CUDNN_DATA_FLOAT;
        cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
        int64_t spatial = 2, dil[2] = {1, 1}, str[2] = {key.stride, key.stride}, pad[2] = {key.pad, key.pad};
        p->conv.set(CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp, "conv comp type");
        p->conv.set(CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode, "conv mode");
        p->conv.set(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial, "conv spatial dims");
        p->conv.set(CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dil, "conv dilations");
        p->conv.set(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, str, "conv strides");
        p->conv.set(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pad, "conv pre paddings");
        p->conv.set(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, pad, "conv post paddings");
        p->conv.finalize("conv finalize");
        p->op = Descriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
        float alpha = 1.f, beta = 0.f;
        p->op.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->x.d, "op x");
        p->op.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->w.d, "op w");
        p->op.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->y.d, "op y");
        p->op.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->conv.d, "op conv");
        p->op.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha, "op alpha");
        p->op.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta, "op beta");
        p->op.finalize("op finalize");
        p->graph = Descriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        p->graph.set(CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p->op.d, "graph ops");
        p->graph.set(CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnn, "graph handle");
        p->graph.finalize("graph finalize");

        std::string tried;
        for (cudnnBackendHeurMode_t heur : {CUDNN_HEUR_MODE_INSTANT, CUDNN_HEUR_MODE_FALLBACK}) {
            std::vector<Descriptor> configs = heuristicConfigs(p->graph, heur);
            for (Descriptor& cfg : configs) {
                Descriptor engine(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
                int64_t n = 0;
                cudnnCheck(cudnnBackendGetAttribute(cfg.d, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &n, &engine.d), "config engine");
                int64_t index = -1;
                cudnnCheck(cudnnBackendGetAttribute(engine.d, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &n, &index), "engine index");
                if (verbose()) tried += " [" + std::to_string(index) + " " + noteText(engine) + " {" + knobText(cfg) + "}]";
                if (hasNote(engine, CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS)) { tried += " " + std::to_string(index) + ":down-convert"; continue; }
                // Engines that need NVRTC: torch's log shows their plans built and never executed, and this
                // runtime ships no compiler, so they are skipped before the plan is built.
                if (hasBehavior(engine, CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION)) { tried += " " + std::to_string(index) + ":runtime-compiled"; continue; }
                if (errata().skips(index)) { tried += " " + std::to_string(index) + ":errata-188288"; continue; }
                bool tensorCore = hasNote(engine, CUDNN_NUMERICAL_NOTE_TENSOR_CORE);
                if (!model->cudnnTf32 && tensorCore) { tried += " " + std::to_string(index) + ":tensor-core"; continue; }
                Descriptor plan(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
                cudnnStatus_t st = cudnnBackendSetAttribute(plan.d, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfg.d);
                if (st == CUDNN_STATUS_SUCCESS) st = cudnnBackendFinalize(plan.d);
                if (st != CUDNN_STATUS_SUCCESS) { tried += " " + std::to_string(index) + ":" + cudnnGetErrorString(st); continue; }
                int64_t ws = 0;
                cudnnCheck(cudnnBackendGetAttribute(plan.d, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &n, &ws), "plan workspace size");
                ensureWorkspace((size_t) ws);
                Descriptor vp = makeVarpack(xptr, wptr, yptr, cudnnWorkspace);
                st = cudnnBackendExecute(cudnn, plan.d, vp.d);
                if (st != CUDNN_STATUS_SUCCESS) { tried += " " + std::to_string(index) + ":exec:" + cudnnGetErrorString(st); continue; }
                p->cfg = std::move(cfg);
                p->plan = std::move(plan);
                p->workspace = (size_t) ws;
                p->engineIndex = index;
                p->tensorCore = tensorCore;
                p->knobs = knobText(p->cfg);
                if (verbose())
                    std::fprintf(stderr, "[gaia_trt_whole] conv x=[%lld,%lld,%lld,%lld] w=[%lld,%lld,%lld,%lld] stride=%lld pad=%lld -> heur=%d engine=%lld knobs={%s} tensor_core=%d workspace=%lld skipped:%s\n",
                                 (long long) key.x[0], (long long) key.x[1], (long long) key.x[2], (long long) key.x[3], (long long) key.w[0], (long long) key.w[1], (long long) key.w[2], (long long) key.w[3],
                                 (long long) key.stride, (long long) key.pad, (int) heur, (long long) index, p->knobs.c_str(), (int) tensorCore, (long long) ws, tried.c_str());
                return p;
            }
        }
        fail("no cuDNN engine could run a convolution; tried:" + tried);
    }

    // Weight vector pointer for an affine word, or nullptr.
    float const* vec(int32_t off) const { return off == kNone ? nullptr : weights + off; }

    Aff affine(int32_t const* a) const {
        Aff r{vec(a[1]), vec(a[2]), vec(a[3]), vec(a[4]), 0};
        r.mode = r.g ? 2 : (r.b ? 1 : 0);
        return r;
    }

    void run(void const* input, void* output, cudaStream_t stream) {
        Model const& m = *model;
        cudnnCheck(cudnnSetStream(cudnn, stream), "cudnnSetStream");
        // ATen per call: set the stream (which resets the workspace), the 32 MiB workspace, the math mode.
        cublasCheck(cublasSetStream(cublas, stream), "cublasSetStream");
        cublasCheck(cublasSetWorkspace(cublas, cublasWorkspace, kCublasWorkspaceBytes), "cublasSetWorkspace");
        cublasCheck(cublasSetMathMode(cublas, m.cublasTf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
        auto ptr = [&](int32_t t) -> float* { return static_cast<float*>(resolve(t, const_cast<void*>(input), output)); };
        const float one = 1.f, zero = 0.f;
        for (int32_t i = 0; i < m.nOps; ++i) {
            int32_t const* o = m.ops[i];
            switch (o[0]) {
                case kConv: {
                    ConvCall& c = calls[i];
                    void const* x = ptr(o[1]); void const* w = weights + o[3]; void* y = ptr(o[2]);
                    if (c.varpack.d == nullptr || c.x != x || c.w != w || c.y != y || c.ws != cudnnWorkspace) {
                        c.varpack = makeVarpack(x, w, y, cudnnWorkspace);
                        c.x = x; c.w = w; c.y = y; c.ws = cudnnWorkspace;
                    }
                    cudnnCheck(cudnnBackendExecute(cudnn, c.plan->plan.d, c.varpack.d), "cudnnBackendExecute");
                    break;
                }
                case kDw: {  // kernel.py dw_forward
                    TensorInfo const& x = m.T(o[1]);
                    TensorInfo const& y = m.T(o[2]);
                    int B = (int) x.d[0], C = (int) x.d[1], H = (int) x.d[2], W = (int) x.d[3], Ho = (int) y.d[2], Wo = (int) y.d[3];
                    int k = o[4], s = o[5], pre = o[6], act = o[7], emit = o[8];
                    float const* xp = ptr(o[1]); float const* wp = weights + o[3]; float* yp = ptr(o[2]);
                    float* yap = emit ? ptr(o[9]) : nullptr;
                    Aff ai = affine(o + 10), ao = affine(o + 15);
                    int tilesX = (Wo + 31) / 32, tilesY = (Ho + 15) / 16;
                    dim3 grid(tilesX * tilesY, C, B), block(32, 4);
                    if (k == 3 && s == 1) dw_kernel<3, 1><<<grid, block, 0, stream>>>(xp, wp, ai, pre, ao, act, yp, yap, C, H, W, Ho, Wo, tilesX);
                    else if (k == 3 && s == 2) dw_kernel<3, 2><<<grid, block, 0, stream>>>(xp, wp, ai, pre, ao, act, yp, nullptr, C, H, W, Ho, Wo, tilesX);
                    else dw_kernel<5, 1><<<grid, block, 0, stream>>>(xp, wp, ai, pre, ao, act, yp, yap, C, H, W, Ho, Wo, tilesX);
                    break;
                }
                case kEpilogue: {  // kernel.py epilogue_, in place on the conv output
                    TensorInfo const& y = m.T(o[1]);
                    int C = (int) y.d[1];
                    int64_t HW = y.d[2] * y.d[3], total4 = y.numel / 4;
                    Aff a = affine(o + 3);
                    float4 const* res = o[9] == kNone ? nullptr : reinterpret_cast<float4 const*>(ptr(o[9]));
                    epilogue_kernel<<<(unsigned) ((total4 + 255) / 256), 256, 0, stream>>>(reinterpret_cast<float4*>(ptr(o[1])), res, a, o[8], C, (int) (HW / 4), (int) total4);
                    break;
                }
                case kAttnPrep: {  // kernel.py attn_prep
                    TensorInfo const& A = m.T(o[1]);
                    int B = (int) A.d[0], Hh = (int) (A.d[1] / 48), N = (int) (A.d[2] * A.d[3]);
                    int BH = B * 2 * Hh;
                    attn_prep_kernel<<<dim3((N + 31) / 32, BH), dim3(32, 8), 0, stream>>>(ptr(o[1]), ptr(o[2]), ptr(o[3]), ptr(o[4]), ptr(o[5]), Hh, N);
                    break;
                }
                case kBmm: {  // torch.bmm(a, b): ATen computes C^T = B^T A^T in cuBLAS's column-major terms
                    TensorInfo const& a = m.T(o[1]);
                    TensorInfo const& b = m.T(o[2]);
                    bool transA = o[4] != 0;
                    int batch = (int) a.d[0];
                    int M = (int) (transA ? a.d[2] : a.d[1]), K = (int) (transA ? a.d[1] : a.d[2]), Nn = (int) b.d[2];
                    // out[M, Nn] row-major: m = Nn, n = M, k = K; A operand is b ('N', lda = Nn); B operand is a:
                    // contiguous [M, K] -> 'N', ldb = K; a transposed view of [K, M] storage -> 'T', ldb = M.
                    cublasCheck(cublasSgemmStridedBatched(cublas, CUBLAS_OP_N, transA ? CUBLAS_OP_T : CUBLAS_OP_N, Nn, M, K, &one,
                                                          ptr(o[2]), Nn, (long long) K * Nn,
                                                          ptr(o[1]), transA ? M : K, (long long) M * K, &zero,
                                                          ptr(o[3]), Nn, (long long) M * Nn, batch), "cublasSgemmStridedBatched");
                    break;
                }
                case kAttnFinal: {  // kernel.py attn_final
                    TensorInfo const& r = m.T(o[1]);
                    int BH = (int) r.d[0], N = (int) r.d[1];
                    attn_final_kernel<<<dim3((N + 31) / 32, BH), 256, 0, stream>>>(ptr(o[1]), ptr(o[3]), N);
                    break;
                }
            }
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) fail(std::string("launch failed at op ") + std::to_string(i) + ": " + cudaGetErrorString(err));
        }
    }
};

// ---------------------------------------------------------------------------
// The plugin.

class WholeModelPlugin final : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    explicit WholeModelPlugin(PluginFieldCollection const* fc) {
        auto model = std::make_shared<Model>();
        bool seen[4] = {false, false, false, false};
        if (fc == nullptr) fail("no fields");
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            PluginField const& f = fc->fields[i];
            if (f.name == nullptr || f.data == nullptr || f.length <= 0) fail("empty field");
            if (std::strcmp(f.name, "program") == 0 && f.type == PluginFieldType::kINT32) {
                model->program.assign(static_cast<int32_t const*>(f.data), static_cast<int32_t const*>(f.data) + f.length);
                seen[0] = true;
            } else if (std::strcmp(f.name, "weights") == 0 && f.type == PluginFieldType::kFLOAT32) {
                model->weights.assign(static_cast<float const*>(f.data), static_cast<float const*>(f.data) + f.length);
                seen[1] = true;
            } else if (std::strcmp(f.name, "cudnn_tf32") == 0 && f.type == PluginFieldType::kINT32 && f.length == 1) {
                model->cudnnTf32 = *static_cast<int32_t const*>(f.data);
                seen[2] = true;
            } else if (std::strcmp(f.name, "cublas_tf32") == 0 && f.type == PluginFieldType::kINT32 && f.length == 1) {
                model->cublasTf32 = *static_cast<int32_t const*>(f.data);
                seen[3] = true;
            } else {
                fail(std::string("unexpected field ") + f.name);
            }
        }
        for (int i = 0; i < 4; ++i)
            if (!seen[i]) fail(std::string("missing field ") + kPluginFields[i]);
        model->decode();
        mModel = std::move(model);
    }

    WholeModelPlugin(WholeModelPlugin const& o) : mModel(o.mModel) {}  // never copies the context

    ~WholeModelPlugin() override = default;

    // IPluginV3
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
            case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
            case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
            case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    IPluginV3* clone() noexcept override {
        try { return new WholeModelPlugin(*this); } catch (...) { return nullptr; }
    }

    // IPluginV3OneCore
    AsciiChar const* getPluginName() const noexcept override { return kPluginName; }
    AsciiChar const* getPluginVersion() const noexcept override { return kPluginVersion; }
    AsciiChar const* getPluginNamespace() const noexcept override { return kPluginNamespace; }

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override {
        if (nbInputs != 1 || nbOutputs != 1) return reject("expects one input and one output");
        if (inputTypes[0] != DataType::kFLOAT) return reject("the input must be FP32");
        outputTypes[0] = DataType::kFLOAT;
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& eb) noexcept override {
        if (nbInputs != 1 || nbOutputs != 1) return reject("expects one input and one output");
        TensorInfo const& xi = mModel->T(mModel->inputId);
        DimsExprs const& x = inputs[0];
        if (x.nbDims != xi.ndim) return reject("input rank does not match the program");
        for (int32_t k = 0; k < x.nbDims; ++k)
            if (x.d[k] == nullptr || !x.d[k]->isConstant() || x.d[k]->getConstantValue() != xi.d[k]) return reject("the input shape must be the one the program was built for");
        TensorInfo const& yi = mModel->T(mModel->outputId);
        outputs[0].nbDims = yi.ndim;
        for (int32_t k = 0; k < yi.ndim; ++k) outputs[0].d[k] = eb.constant(yi.d[k]);
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override {
        if (pos < 0 || pos >= nbInputs + nbOutputs) return false;
        return inOut[pos].desc.type == DataType::kFLOAT && inOut[pos].desc.format == TensorFormat::kLINEAR;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override {
        if (nbInputs != 1 || nbOutputs != 1) return reject("expects one input and one output");
        if (!sameDims(in[0].min, in[0].max) || !sameDims(in[0].desc.dims, in[0].max)) return reject("dynamic input shapes are not supported");
        return checkShapes(in[0].desc.dims, out[0].desc.dims);
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        return 0;  // the context owns its arena and workspaces
    }

    // IPluginV3OneRuntime
    int32_t onShapeChange(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override {
        if (nbInputs != 1 || nbOutputs != 1) return reject("expects one input and one output");
        if (in[0].type != DataType::kFLOAT || in[0].format != TensorFormat::kLINEAR || out[0].type != DataType::kFLOAT || out[0].format != TensorFormat::kLINEAR)
            return reject("input and output must be FP32 linear at run time");
        return checkShapes(in[0].dims, out[0].dims);
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
        try {
            if (!mContext) return reject("enqueue on a plugin that is not attached to a context");
            if (checkShapes(inputDesc[0].dims, outputDesc[0].dims) != 0) return 1;
            mContext->run(inputs[0], outputs[0], stream);
            return 0;
        } catch (std::exception const& e) {
            return reject(e.what());
        }
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
        try {
            auto* attached = new WholeModelPlugin(*this);
            try {
                attached->mContext = std::make_unique<Context>(mModel);
            } catch (std::exception const& e) {
                delete attached;
                reject(e.what());
                return nullptr;
            }
            return attached;
        } catch (...) {
            return nullptr;
        }
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        try {
            mFields.clear();
            mFields.emplace_back("program", mModel->program.data(), PluginFieldType::kINT32, (int32_t) mModel->program.size());
            mFields.emplace_back("weights", mModel->weights.data(), PluginFieldType::kFLOAT32, (int32_t) mModel->weights.size());
            mFields.emplace_back("cudnn_tf32", &mModel->cudnnTf32, PluginFieldType::kINT32, 1);
            mFields.emplace_back("cublas_tf32", &mModel->cublasTf32, PluginFieldType::kINT32, 1);
            mCollection.nbFields = (int32_t) mFields.size();
            mCollection.fields = mFields.data();
            return &mCollection;
        } catch (...) {
            return nullptr;
        }
    }

private:
    static int reject(char const* why) noexcept {
        std::fprintf(stderr, "[gaia_trt_whole] %s rejected: %s\n", kPluginName, why);
        return 1;
    }

    static bool sameDims(Dims const& a, Dims const& b) noexcept {
        if (a.nbDims != b.nbDims) return false;
        for (int32_t i = 0; i < a.nbDims; ++i)
            if (a.d[i] != b.d[i]) return false;
        return true;
    }

    static bool equalsInfo(Dims const& d, TensorInfo const& t) noexcept {
        if (d.nbDims != t.ndim) return false;
        for (int32_t i = 0; i < d.nbDims; ++i)
            if (d.d[i] != t.d[i]) return false;
        return true;
    }

    int32_t checkShapes(Dims const& in, Dims const& out) const noexcept {
        if (!equalsInfo(in, mModel->T(mModel->inputId))) return reject("the input shape is not the one the program was built for");
        if (!equalsInfo(out, mModel->T(mModel->outputId))) return reject("output shape mismatch");
        return 0;
    }

    std::shared_ptr<Model const> mModel;
    std::unique_ptr<Context> mContext;  // only in clones returned by attachToContext
    std::vector<PluginField> mFields;
    PluginFieldCollection mCollection{};
};

class Creator final : public IPluginCreatorV3One {
public:
    Creator() {
        mFields.emplace_back(kPluginFields[0], nullptr, PluginFieldType::kINT32, 0);
        mFields.emplace_back(kPluginFields[1], nullptr, PluginFieldType::kFLOAT32, 0);
        mFields.emplace_back(kPluginFields[2], nullptr, PluginFieldType::kINT32, 1);
        mFields.emplace_back(kPluginFields[3], nullptr, PluginFieldType::kINT32, 1);
        mCollection.nbFields = (int32_t) mFields.size();
        mCollection.fields = mFields.data();
    }

    IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        try {
            return new WholeModelPlugin(fc);
        } catch (std::exception const& e) {
            std::fprintf(stderr, "[gaia_trt_whole] %s createPlugin failed: %s\n", kPluginName, e.what());
            return nullptr;
        } catch (...) {
            return nullptr;
        }
    }

    PluginFieldCollection const* getFieldNames() noexcept override { return &mCollection; }
    AsciiChar const* getPluginName() const noexcept override { return kPluginName; }
    AsciiChar const* getPluginVersion() const noexcept override { return kPluginVersion; }
    AsciiChar const* getPluginNamespace() const noexcept override { return kPluginNamespace; }

private:
    std::vector<PluginField> mFields;
    PluginFieldCollection mCollection{};
};

}  // namespace
}  // namespace gaia_trt_whole

extern "C" __attribute__((visibility("default"))) bool gaia_trt_whole_model_register_plugins() {
    using namespace gaia_trt_whole;
    static bool registered = false;
    if (registered) return true;
    try {
        static Creator creator;
        nvinfer1::IPluginRegistry* registry = getPluginRegistry();
        if (registry == nullptr) return false;
        registered = registry->registerCreator(creator, kPluginNamespace);
        return registered;
    } catch (...) {
        return false;
    }
}
