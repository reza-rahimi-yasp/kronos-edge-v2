// IPluginV3 wrappers around the Gaia-generated kernels and their channel-last
// counterparts. See gaia_plugins.h.
//
// Structure: GaiaPluginBase holds what the four plugins share (the three
// IPluginV3 capability interfaces, FP32/static-shape enforcement, the replay
// of each plugin's format rule at configure and shape-change time, INT32
// field storage that doubles as the serialization record, clone-based
// attachToContext). Each plugin adds its format rule (which of its tensors
// may be kHWC), its shape rule and its launch, which is the host logic of
// the corresponding kernel.py wrapper (dw_forward, epilogue_, attn_prep,
// attn_final) with the TensorRT buffers in place of torch tensors, choosing
// Gaia's kernel for linear tensors and the kernel of gaia_hwc_kernels.cuh
// for kHWC ones. The device code for the linear layout is Gaia's, included
// unmodified from gaia_device_code.cuh; the epilogue uses the out-of-place
// copy in gaia_adapted_epilogue.cuh.
//
// Every interface method is noexcept, so bodies that can throw (allocation,
// parsing) are wrapped and report failure through the method's return
// convention: nullptr, false, or a non-zero status.

#include "gaia_plugins.h"

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "gaia_device_code.cuh"
#include "gaia_adapted_epilogue.cuh"
#include "gaia_hwc_kernels.cuh"

namespace gaia_trt {

using namespace nvinfer1;

namespace {

int affineInputs(int32_t mode) noexcept {
    return mode == 2 ? 4 : mode == 1 ? 1 : mode == 0 ? 0 : -1;
}

// Build Gaia's Aff from consecutive FP32 input pointers, like mkaff() does from optional tensors.
Aff makeAff(int32_t mode, void const* const* inputs, int& pos) noexcept {
    Aff a{nullptr, nullptr, nullptr, nullptr, 0};
    if (mode == 2) {
        a.g = static_cast<float const*>(inputs[pos]);
        a.b = static_cast<float const*>(inputs[pos + 1]);
        a.m = static_cast<float const*>(inputs[pos + 2]);
        a.is = static_cast<float const*>(inputs[pos + 3]);
        pos += 4;
    } else if (mode == 1) {
        a.b = static_cast<float const*>(inputs[pos]);
        pos += 1;
    }
    a.mode = a.g ? 2 : (a.b ? 1 : 0);
    return a;
}

int64_t volume(Dims const& d) noexcept {
    int64_t v = 1;
    for (int32_t i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}

bool sameDims(Dims const& a, Dims const& b) noexcept {
    if (a.nbDims != b.nbDims) return false;
    for (int32_t i = 0; i < a.nbDims; ++i)
        if (a.d[i] != b.d[i]) return false;
    return true;
}

int reject(char const* plugin, char const* why) noexcept {
    std::fprintf(stderr, "[gaia_trt] %s rejected: %s\n", plugin, why);
    return 1;
}

// Format predicates for supportsFormatCombination. "Activation" tensors may be
// linear or kHWC; everything else must be linear.
bool fp32(DynamicPluginTensorDesc const& d) noexcept { return d.desc.type == DataType::kFLOAT; }
bool linear(DynamicPluginTensorDesc const& d) noexcept { return fp32(d) && d.desc.format == TensorFormat::kLINEAR; }
// An activation tensor may be linear or kHWC, as far as the plugin's "layout" field allows.
bool activation(DynamicPluginTensorDesc const& d, int32_t layout) noexcept {
    if (!fp32(d)) return false;
    if (d.desc.format == TensorFormat::kLINEAR) return layout != kLayoutHwc;
    if (d.desc.format == TensorFormat::kHWC) return layout != kLayoutLinear;
    return false;
}
bool sameFormat(DynamicPluginTensorDesc const& d, DynamicPluginTensorDesc const& ref) noexcept {
    return fp32(d) && d.desc.format == ref.desc.format;
}
bool isHwc(PluginTensorDesc const& d) noexcept { return d.format == TensorFormat::kHWC; }

// Diagnostic trace of the layout negotiation (GAIA_TRT_TRACE_FORMATS=1 in the environment): every
// supportsFormatCombination query with its answer, every configurePlugin, and every enqueue's layout.
bool traceFormats() noexcept {
    static int const on = [] { char const* v = std::getenv("GAIA_TRT_TRACE_FORMATS"); return (v != nullptr && v[0] == '1') ? 1 : 0; }();
    return on == 1;
}

char const* formatName(TensorFormat f) noexcept {
    switch (f) {
        case TensorFormat::kLINEAR: return "LINEAR";
        case TensorFormat::kHWC: return "HWC";
        case TensorFormat::kCHW32: return "CHW32";
        case TensorFormat::kDHWC: return "DHWC";
        default: return "other";
    }
}

void traceQuery(char const* plugin, int32_t pos, DynamicPluginTensorDesc const* inOut, bool answer) noexcept {
    if (!traceFormats()) return;
    std::fprintf(stderr, "[gaia_trt trace] %s supportsFormatCombination pos=%d type=%d format=%s -> %s\n", plugin, pos,
                 (int) inOut[pos].desc.type, formatName(inOut[pos].desc.format), answer ? "yes" : "no");
}

int afterLaunch(char const* plugin) noexcept {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[gaia_trt] %s launch failed: %s\n", plugin, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

// INT32 plugin fields: parsed from the creator's collection, kept as the
// serialization record. The PluginField entries point into mValues, so the
// collection is rebuilt whenever it is handed out and stays valid for the
// lifetime of the plugin that owns it (copies rebuild their own pointers).
class IntFields {
public:
    IntFields(char const* const* names, int count) : mNames(names, names + count), mValues(count, 0) {}

    void parse(PluginFieldCollection const* fc, char const* plugin) {
        if (fc == nullptr && !mNames.empty()) throw std::runtime_error(std::string(plugin) + ": no fields");
        for (size_t i = 0; i < mNames.size(); ++i) {
            bool found = false;
            for (int32_t f = 0; f < fc->nbFields; ++f) {
                PluginField const& field = fc->fields[f];
                if (field.name != nullptr && std::strcmp(field.name, mNames[i]) == 0) {
                    if (field.type != PluginFieldType::kINT32 || field.length != 1 || field.data == nullptr)
                        throw std::runtime_error(std::string(plugin) + ": field " + mNames[i] + " must be one INT32");
                    std::memcpy(&mValues[i], field.data, sizeof(int32_t));
                    found = true;
                }
            }
            if (!found) throw std::runtime_error(std::string(plugin) + ": missing field " + mNames[i]);
        }
    }

    int32_t get(char const* name) const {
        for (size_t i = 0; i < mNames.size(); ++i)
            if (std::strcmp(mNames[i], name) == 0) return mValues[i];
        throw std::runtime_error(std::string("unknown field ") + name);
    }

    PluginFieldCollection const* collection() {
        mFields.clear();
        for (size_t i = 0; i < mNames.size(); ++i)
            mFields.emplace_back(mNames[i], &mValues[i], PluginFieldType::kINT32, 1);
        mCollection.nbFields = static_cast<int32_t>(mFields.size());
        mCollection.fields = mFields.empty() ? nullptr : mFields.data();
        return &mCollection;
    }

private:
    std::vector<char const*> mNames;
    std::vector<int32_t> mValues;
    std::vector<PluginField> mFields;
    PluginFieldCollection mCollection{};
};

class GaiaPluginBase : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    GaiaPluginBase(char const* name, char const* const* fieldNames, int fieldCount, PluginFieldCollection const* fc)
        : mName(name), mFields(fieldNames, fieldCount) {
        mFields.parse(fc, name);
        mLayout = mFields.get("layout");
        if (mLayout != kLayoutAny && mLayout != kLayoutLinear && mLayout != kLayoutHwc) throw std::runtime_error("layout must be 0 (any), 1 (linear) or 2 (hwc)");
    }

    // IPluginV3
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
            case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
            case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
            case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    // IPluginV3OneCore
    AsciiChar const* getPluginName() const noexcept override { return mName; }
    AsciiChar const* getPluginVersion() const noexcept override { return kPluginVersion; }
    AsciiChar const* getPluginNamespace() const noexcept override { return kPluginNamespace; }

    // IPluginV3OneBuild
    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override {
        for (int32_t i = 0; i < nbInputs; ++i)
            if (inputTypes[i] != DataType::kFLOAT) return reject(mName, "inputs must be FP32");
        for (int32_t i = 0; i < nbOutputs; ++i) outputTypes[i] = DataType::kFLOAT;
        return 0;
    }

    // supportsFormatCombination is each plugin's own (which tensors may be kHWC).

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override {
        try {
            std::vector<DynamicPluginTensorDesc> all(in, in + nbInputs);
            all.insert(all.end(), out, out + nbOutputs);
            for (int32_t i = 0; i < nbInputs; ++i)
                if (!sameDims(in[i].min, in[i].max) || !sameDims(in[i].desc.dims, in[i].max))
                    return reject(mName, "dynamic input shapes are not supported");
            return checkAll(all, nbInputs, nbOutputs);
        } catch (std::exception const& e) {
            return reject(mName, e.what());
        }
    }

    // IPluginV3OneRuntime
    int32_t onShapeChange(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override {
        try {
            std::vector<DynamicPluginTensorDesc> all;
            for (int32_t i = 0; i < nbInputs + nbOutputs; ++i) {
                DynamicPluginTensorDesc d{};
                d.desc = i < nbInputs ? in[i] : out[i - nbInputs];
                d.min = d.max = d.opt = d.desc.dims;
                all.push_back(d);
            }
            return checkAll(all, nbInputs, nbOutputs);
        } catch (std::exception const& e) {
            return reject(mName, e.what());
        }
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
        // No per-context resources: the plugins own no device memory and no handles.
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        try {
            return mFields.collection();
        } catch (...) {
            return nullptr;
        }
    }

protected:
    // Shape rule shared by configurePlugin (build) and onShapeChange (run); 0 when the shapes are supported.
    // ``hwc`` says the activation tensors are in the kHWC layout (the format rule keeps them consistent).
    virtual int32_t checkShapes(std::vector<Dims> const& inputs, std::vector<Dims> const& outputs, bool hwc) const = 0;

    // Replays the plugin's own format rule over the whole combination, then applies the shape rule.
    int32_t checkAll(std::vector<DynamicPluginTensorDesc> const& all, int32_t nbInputs, int32_t nbOutputs) {
        bool hwc = false;
        for (int32_t pos = 0; pos < nbInputs + nbOutputs; ++pos) {
            if (!supportsFormatCombination(pos, all.data(), nbInputs, nbOutputs))
                return reject(mName, "unsupported type or layout combination (FP32 linear, or kHWC for activation tensors)");
            hwc |= all[pos].desc.format == TensorFormat::kHWC;
        }
        if (traceFormats()) std::fprintf(stderr, "[gaia_trt trace] %s configured: activation layout %s\n", mName, hwc ? "HWC" : "LINEAR");
        std::vector<Dims> ins, outs;
        for (int32_t i = 0; i < nbInputs; ++i) ins.push_back(all[i].desc.dims);
        for (int32_t i = 0; i < nbOutputs; ++i) outs.push_back(all[nbInputs + i].desc.dims);
        return checkShapes(ins, outs, hwc);
    }

    int32_t field(char const* name) const { return mFields.get(name); }

    // All dimensions constant, or the plugin cannot pin its launch to the measured shapes.
    static bool constantDims(DimsExprs const& e) noexcept {
        for (int32_t i = 0; i < e.nbDims; ++i)
            if (e.d[i] == nullptr || !e.d[i]->isConstant()) return false;
        return true;
    }

    char const* mName;
    IntFields mFields;
    int32_t mLayout = kLayoutAny;
};

// ---------------------------------------------------------------------------
// GaiaDepthwiseHswish: dw_kernel<K,S> (linear) or dw_kernel_hwc<K,S,CT> (kHWC). Host logic of dw_forward().
// Input 0 (x) may be kHWC; its outputs follow it; the weight and the affine vectors are linear.

class DepthwisePlugin final : public GaiaPluginBase {
public:
    explicit DepthwisePlugin(PluginFieldCollection const* fc)
        : GaiaPluginBase(kDepthwiseName, kDepthwiseFields, 8, fc) {
        mK = field("k"); mS = field("s"); mPre = field("pre"); mAct = field("act"); mEmit = field("emit");
        mInMode = field("in_mode"); mOutMode = field("out_mode");
        bool config = (mK == 3 && mS == 1) || (mK == 3 && mS == 2) || (mK == 5 && mS == 1);
        if (!config) throw std::runtime_error("unsupported dw config (k, s) must be (3,1), (3,2) or (5,1)");
        if (mEmit != 0 && mS != 1) throw std::runtime_error("emit requires stride 1");
        if (affineInputs(mInMode) < 0 || affineInputs(mOutMode) < 0) throw std::runtime_error("affine mode must be 0, 1 or 2");
        for (int32_t v : {mPre, mAct, mEmit})
            if (v != 0 && v != 1) throw std::runtime_error("pre, act and emit must be 0 or 1");
    }

    IPluginV3* clone() noexcept override {
        try { return new DepthwisePlugin(*this); } catch (...) { return nullptr; }
    }

    int32_t getNbOutputs() const noexcept override { return mEmit ? 2 : 1; }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override {
        if (pos < 0 || pos >= nbInputs + nbOutputs) return false;
        bool ok = pos == 0 ? activation(inOut[0], mLayout) : pos < nbInputs ? linear(inOut[pos]) : sameFormat(inOut[pos], inOut[0]);
        traceQuery(mName, pos, inOut, ok);
        return ok;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& eb) noexcept override {
        if (nbInputs != expectedInputs() || nbOutputs != getNbOutputs()) return reject(mName, "wrong number of inputs or outputs");
        DimsExprs const& x = inputs[0];
        if (x.nbDims != 4 || !constantDims(x)) return reject(mName, "x must be a static 4-D tensor");
        int64_t H = x.d[2]->getConstantValue(), W = x.d[3]->getConstantValue();
        int64_t P = mK / 2;
        int64_t Ho = (H + 2 * P - mK) / mS + 1, Wo = (W + 2 * P - mK) / mS + 1;
        if (Ho <= 0 || Wo <= 0) return reject(mName, "input smaller than the kernel");
        outputs[0].nbDims = 4;
        outputs[0].d[0] = x.d[0]; outputs[0].d[1] = x.d[1];
        outputs[0].d[2] = eb.constant(Ho); outputs[0].d[3] = eb.constant(Wo);
        if (mEmit) outputs[1] = x;
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
        Dims const& xd = inputDesc[0].dims;
        int B = (int) xd.d[0], C = (int) xd.d[1], H = (int) xd.d[2], W = (int) xd.d[3];
        int Ho = (int) outputDesc[0].dims.d[2], Wo = (int) outputDesc[0].dims.d[3];
        float const* xp = static_cast<float const*>(inputs[0]);
        float const* wp = static_cast<float const*>(inputs[1]);
        int pos = 2;
        Aff ai = makeAff(mInMode, inputs, pos);
        Aff ao = makeAff(mOutMode, inputs, pos);
        float* yp = static_cast<float*>(outputs[0]);
        float* yap = mEmit ? static_cast<float*>(outputs[1]) : nullptr;
        int p = mPre, a = mAct;
        if (traceFormats()) std::fprintf(stderr, "[gaia_trt trace] %s enqueue %s C=%d\n", mName, formatName(inputDesc[0].format), C);
        if (isHwc(inputDesc[0])) {
            // 256 threads: CT channels x TY rows; 32 channels unless C is not a multiple of 32 (the 16-channel stem).
            if (C % 32 == 0) launchHwc<32>(xp, wp, ai, p, ao, a, yp, yap, B, C, H, W, Ho, Wo, stream);
            else launchHwc<16>(xp, wp, ai, p, ao, a, yp, yap, B, C, H, W, Ho, Wo, stream);
            return afterLaunch(mName);
        }
        int tilesX = (Wo + 31) / 32, tilesY = (Ho + 15) / 16;
        dim3 grid(tilesX * tilesY, C, B), block(32, 4);
        if (mK == 3 && mS == 1) dw_kernel<3, 1><<<grid, block, 0, stream>>>(xp, wp, ai, p, ao, a, yp, yap, C, H, W, Ho, Wo, tilesX);
        else if (mK == 3 && mS == 2) dw_kernel<3, 2><<<grid, block, 0, stream>>>(xp, wp, ai, p, ao, a, yp, nullptr, C, H, W, Ho, Wo, tilesX);
        else if (mK == 5 && mS == 1) dw_kernel<5, 1><<<grid, block, 0, stream>>>(xp, wp, ai, p, ao, a, yp, yap, C, H, W, Ho, Wo, tilesX);
        else return reject(mName, "unsupported dw config");
        return afterLaunch(mName);
    }

protected:
    int32_t checkShapes(std::vector<Dims> const& in, std::vector<Dims> const& out, bool hwc) const override {
        if ((int) in.size() != expectedInputs()) return reject(mName, "wrong number of inputs");
        if ((int) out.size() != getNbOutputs()) return reject(mName, "wrong number of outputs");
        Dims const& x = in[0];
        if (x.nbDims != 4) return reject(mName, "x must be 4-D");
        int64_t B = x.d[0], C = x.d[1], H = x.d[2], W = x.d[3];
        if (B <= 0 || C <= 0 || H <= 0 || W <= 0) return reject(mName, "x has an empty dimension");
        if (C > 65535 || B > 65535) return reject(mName, "C and B must fit a CUDA grid dimension");
        if (volume(in[1]) != C * mK * mK) return reject(mName, "w must hold C*k*k weights");
        for (size_t i = 2; i < in.size(); ++i)
            if (volume(in[i]) != C) return reject(mName, "affine tensors must have C elements");
        int64_t P = mK / 2, Ho = (H + 2 * P - mK) / mS + 1, Wo = (W + 2 * P - mK) / mS + 1;
        Dims expect{4, {B, C, Ho, Wo}};
        if (!sameDims(out[0], expect)) return reject(mName, "output 0 shape mismatch");
        if (mEmit && !sameDims(out[1], x)) return reject(mName, "output 1 must match x");
        if (hwc) {
            int ct = (C % 32 == 0) ? 32 : 16, ty = 256 / ct, tw = (mS == 1) ? 8 : 4;
            if (((Wo + tw - 1) / tw) * ((Ho + ty - 1) / ty) > 2147483647 || (C + ct - 1) / ct > 65535) return reject(mName, "too many tiles");
        } else if (((int64_t) tiles(Ho, Wo)) > 2147483647) {
            return reject(mName, "too many tiles");
        }
        return 0;
    }

private:
    int expectedInputs() const noexcept { return 2 + affineInputs(mInMode) + affineInputs(mOutMode); }
    static int64_t tiles(int64_t Ho, int64_t Wo) noexcept { return ((Wo + 31) / 32) * ((Ho + 15) / 16); }

    template <int CT>
    void launchHwc(float const* xp, float const* wp, Aff ai, int p, Aff ao, int a, float* yp, float* yap,
                   int B, int C, int H, int W, int Ho, int Wo, cudaStream_t stream) const noexcept {
        constexpr int TY = 256 / CT;
        dim3 block(CT, TY);
        if (mS == 1) {
            constexpr int TW = 8;
            int tilesX = (Wo + TW - 1) / TW, tilesY = (Ho + TY - 1) / TY;
            dim3 grid(tilesX * tilesY, (C + CT - 1) / CT, B);
            if (mK == 3) dw_kernel_hwc<3, 1, CT><<<grid, block, 0, stream>>>(xp, wp, ai, p, ao, a, yp, yap, C, H, W, Ho, Wo, tilesX);
            else dw_kernel_hwc<5, 1, CT><<<grid, block, 0, stream>>>(xp, wp, ai, p, ao, a, yp, yap, C, H, W, Ho, Wo, tilesX);
        } else {
            constexpr int TW = 4;
            int tilesX = (Wo + TW - 1) / TW, tilesY = (Ho + TY - 1) / TY;
            dim3 grid(tilesX * tilesY, (C + CT - 1) / CT, B);
            dw_kernel_hwc<3, 2, CT><<<grid, block, 0, stream>>>(xp, wp, ai, p, ao, a, yp, nullptr, C, H, W, Ho, Wo, tilesX);
        }
    }

    int32_t mK, mS, mPre, mAct, mEmit, mInMode, mOutMode;
};

// ---------------------------------------------------------------------------
// GaiaEpilogue: epilogue_kernel_oop (linear) or epilogue_kernel_hwc (kHWC). Host logic of epilogue_(), out of place.
// Input 0 (y) may be kHWC when C % 4 == 0; the residual and the output follow it; the affine vectors are linear.

class EpiloguePlugin final : public GaiaPluginBase {
public:
    explicit EpiloguePlugin(PluginFieldCollection const* fc)
        : GaiaPluginBase(kEpilogueName, kEpilogueFields, 4, fc) {
        mMode = field("mode"); mAct = field("act"); mHasRes = field("has_res");
        if (affineInputs(mMode) < 0) throw std::runtime_error("mode must be 0, 1 or 2");
        if ((mAct != 0 && mAct != 1) || (mHasRes != 0 && mHasRes != 1)) throw std::runtime_error("act and has_res must be 0 or 1");
    }

    IPluginV3* clone() noexcept override {
        try { return new EpiloguePlugin(*this); } catch (...) { return nullptr; }
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override {
        if (pos < 0 || pos >= nbInputs + nbOutputs) return false;
        bool ok;
        if (pos == 0) {
            // four consecutive kHWC elements are four channels of one pixel: the float4 kernel needs C % 4 == 0
            ok = activation(inOut[0], mLayout)
                 && (inOut[0].desc.format != TensorFormat::kHWC || (inOut[0].desc.dims.nbDims == 4 && inOut[0].desc.dims.d[1] % 4 == 0));
        } else if (pos <= affineInputs(mMode)) {
            ok = linear(inOut[pos]);
        } else {
            ok = sameFormat(inOut[pos], inOut[0]);  // the residual (if any) and the output
        }
        traceQuery(mName, pos, inOut, ok);
        return ok;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder&) noexcept override {
        if (nbInputs != expectedInputs() || nbOutputs != 1) return reject(mName, "wrong number of inputs or outputs");
        if (inputs[0].nbDims != 4 || !constantDims(inputs[0])) return reject(mName, "y must be a static 4-D tensor");
        outputs[0] = inputs[0];
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
        Dims const& yd = inputDesc[0].dims;
        int C = (int) yd.d[1];
        int64_t HW = yd.d[2] * yd.d[3];
        int64_t total4 = volume(yd) / 4;
        int pos = 1;
        Aff a = makeAff(mMode, inputs, pos);
        float4 const* res = mHasRes ? static_cast<float4 const*>(inputs[pos]) : nullptr;
        if (isHwc(inputDesc[0]))
            epilogue_kernel_hwc<<<(unsigned) ((total4 + 255) / 256), 256, 0, stream>>>(
                static_cast<float4 const*>(inputs[0]), static_cast<float4*>(outputs[0]), res, a, mAct, C, (int) total4);
        else
            epilogue_kernel_oop<<<(unsigned) ((total4 + 255) / 256), 256, 0, stream>>>(
                static_cast<float4 const*>(inputs[0]), static_cast<float4*>(outputs[0]), res, a, mAct, C, (int) (HW / 4), (int) total4);
        return afterLaunch(mName);
    }

protected:
    int32_t checkShapes(std::vector<Dims> const& in, std::vector<Dims> const& out, bool hwc) const override {
        if ((int) in.size() != expectedInputs()) return reject(mName, "wrong number of inputs");
        if (out.size() != 1) return reject(mName, "wrong number of outputs");
        Dims const& y = in[0];
        if (y.nbDims != 4) return reject(mName, "y must be 4-D");
        int64_t C = y.d[1], HW = y.d[2] * y.d[3];
        if (hwc && C % 4 != 0) return reject(mName, "C must be a multiple of 4 in the kHWC layout");
        if (!hwc && HW % 4 != 0) return reject(mName, "H*W must be a multiple of 4");  // TORCH_CHECK(HW % 4 == 0)
        if (volume(y) / 4 >= (1LL << 31)) return reject(mName, "too large");             // TORCH_CHECK(total4 < 2^31)
        for (int i = 1; i <= affineInputs(mMode); ++i)
            if (volume(in[i]) != C) return reject(mName, "affine tensors must have C elements");
        if (mHasRes && !sameDims(in.back(), y)) return reject(mName, "res must match y");
        if (!sameDims(out[0], y)) return reject(mName, "output must match y");
        return 0;
    }

private:
    int expectedInputs() const noexcept { return 1 + affineInputs(mMode) + mHasRes; }

    int32_t mMode, mAct, mHasRes;
};

// ---------------------------------------------------------------------------
// GaiaAttnPrep: attn_prep_kernel (linear) or attn_prep_kernel_hwc (kHWC). Host logic of attn_prep().
// qkv and agg share a layout, linear or kHWC; the three matmul operands it produces are linear.

class AttnPrepPlugin final : public GaiaPluginBase {
public:
    explicit AttnPrepPlugin(PluginFieldCollection const* fc) : GaiaPluginBase(kAttnPrepName, kAttnPrepFields, 1, fc) {}

    IPluginV3* clone() noexcept override {
        try { return new AttnPrepPlugin(*this); } catch (...) { return nullptr; }
    }

    int32_t getNbOutputs() const noexcept override { return 3; }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override {
        if (pos < 0 || pos >= nbInputs + nbOutputs) return false;
        bool ok = pos == 0 ? activation(inOut[0], mLayout) : pos < nbInputs ? sameFormat(inOut[pos], inOut[0]) : linear(inOut[pos]);
        traceQuery(mName, pos, inOut, ok);
        return ok;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& eb) noexcept override {
        if (nbInputs != 2 || nbOutputs != 3) return reject(mName, "expects 2 inputs and 3 outputs");
        DimsExprs const& A = inputs[0];
        if (A.nbDims != 4 || !constantDims(A) || !constantDims(inputs[1]) || inputs[1].nbDims != 4) return reject(mName, "qkv and agg must be static 4-D tensors");
        int64_t B = A.d[0]->getConstantValue(), C = A.d[1]->getConstantValue();
        int64_t N = A.d[2]->getConstantValue() * A.d[3]->getConstantValue();
        if (C % 48 != 0) return reject(mName, "channels must be a multiple of 48 (q, k, v of 16 each per head)");
        int64_t BH = B * 2 * (C / 48);
        outputs[0].nbDims = 3; outputs[0].d[0] = eb.constant(BH); outputs[0].d[1] = eb.constant(16); outputs[0].d[2] = eb.constant(N);
        outputs[1] = outputs[0];
        outputs[2].nbDims = 3; outputs[2].d[0] = eb.constant(BH); outputs[2].d[1] = eb.constant(N); outputs[2].d[2] = eb.constant(17);
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
        Dims const& A = inputDesc[0].dims;
        int B = (int) A.d[0], Hh = (int) (A.d[1] / 48), N = (int) (A.d[2] * A.d[3]);
        int BH = B * 2 * Hh;
        if (isHwc(inputDesc[0]))
            attn_prep_kernel_hwc<<<dim3((N + 31) / 32, BH), dim3(32, 8), 0, stream>>>(
                static_cast<float const*>(inputs[0]), static_cast<float const*>(inputs[1]),
                static_cast<float*>(outputs[0]), static_cast<float*>(outputs[1]), static_cast<float*>(outputs[2]), Hh, N);
        else
            attn_prep_kernel<<<dim3((N + 31) / 32, BH), dim3(32, 8), 0, stream>>>(
                static_cast<float const*>(inputs[0]), static_cast<float const*>(inputs[1]),
                static_cast<float*>(outputs[0]), static_cast<float*>(outputs[1]), static_cast<float*>(outputs[2]), Hh, N);
        return afterLaunch(mName);
    }

protected:
    int32_t checkShapes(std::vector<Dims> const& in, std::vector<Dims> const& out, bool) const override {
        if (in.size() != 2 || out.size() != 3) return reject(mName, "expects 2 inputs and 3 outputs");
        Dims const& A = in[0];
        if (A.nbDims != 4 || !sameDims(A, in[1])) return reject(mName, "qkv and agg must be equal 4-D tensors");
        int64_t B = A.d[0], C = A.d[1], N = A.d[2] * A.d[3];
        if (C % 48 != 0 || C == 0) return reject(mName, "channels must be a multiple of 48");
        int64_t BH = B * 2 * (C / 48);
        if (BH > 65535 || (N + 31) / 32 > 2147483647) return reject(mName, "grid too large");
        Dims qk{3, {BH, 16, N}}, vp{3, {BH, N, 17}};
        if (!sameDims(out[0], qk) || !sameDims(out[1], qk) || !sameDims(out[2], vp)) return reject(mName, "output shape mismatch");
        return 0;
    }
};

// ---------------------------------------------------------------------------
// GaiaAttnFinal: attn_final_kernel (linear) or attn_final_kernel_hwc (kHWC). Host logic of attn_final().
// The matmul result r is linear; the 4-D output may be kHWC.

class AttnFinalPlugin final : public GaiaPluginBase {
public:
    explicit AttnFinalPlugin(PluginFieldCollection const* fc) : GaiaPluginBase(kAttnFinalName, kAttnFinalFields, 4, fc) {
        mBatch = field("batch"); mHeight = field("height"); mWidth = field("width");
        if (mBatch <= 0 || mHeight <= 0 || mWidth <= 0) throw std::runtime_error("batch, height and width must be positive");
    }

    IPluginV3* clone() noexcept override {
        try { return new AttnFinalPlugin(*this); } catch (...) { return nullptr; }
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override {
        if (pos < 0 || pos >= nbInputs + nbOutputs) return false;
        bool ok = pos < nbInputs ? linear(inOut[pos]) : activation(inOut[pos], mLayout);
        traceQuery(mName, pos, inOut, ok);
        return ok;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& eb) noexcept override {
        if (nbInputs != 1 || nbOutputs != 1) return reject(mName, "expects 1 input and 1 output");
        DimsExprs const& r = inputs[0];
        if (r.nbDims != 3 || !constantDims(r)) return reject(mName, "r must be a static 3-D tensor");
        int64_t BH = r.d[0]->getConstantValue(), N = r.d[1]->getConstantValue();
        if (r.d[2]->getConstantValue() != 17 || N != (int64_t) mHeight * mWidth || BH % mBatch != 0) return reject(mName, "r must be [B*heads, H*W, 17]");
        outputs[0].nbDims = 4;
        outputs[0].d[0] = eb.constant(mBatch); outputs[0].d[1] = eb.constant((BH / mBatch) * 16);
        outputs[0].d[2] = eb.constant(mHeight); outputs[0].d[3] = eb.constant(mWidth);
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
        Dims const& r = inputDesc[0].dims;
        int BH = (int) r.d[0], N = (int) r.d[1];
        if (isHwc(outputDesc[0]))
            attn_final_kernel_hwc<<<dim3((N + 31) / 32, BH), 256, 0, stream>>>(static_cast<float const*>(inputs[0]), static_cast<float*>(outputs[0]), N, BH / mBatch);
        else
            attn_final_kernel<<<dim3((N + 31) / 32, BH), 256, 0, stream>>>(static_cast<float const*>(inputs[0]), static_cast<float*>(outputs[0]), N);
        return afterLaunch(mName);
    }

protected:
    int32_t checkShapes(std::vector<Dims> const& in, std::vector<Dims> const& out, bool) const override {
        if (in.size() != 1 || out.size() != 1) return reject(mName, "expects 1 input and 1 output");
        Dims const& r = in[0];
        if (r.nbDims != 3 || r.d[2] != 17) return reject(mName, "r must be [B*heads, N, 17]");
        int64_t BH = r.d[0], N = r.d[1];
        if (N != (int64_t) mHeight * mWidth || BH % mBatch != 0 || BH > 65535) return reject(mName, "r does not match batch/height/width");
        Dims expect{4, {mBatch, (BH / mBatch) * 16, mHeight, mWidth}};
        if (!sameDims(out[0], expect)) return reject(mName, "output shape mismatch");
        return 0;
    }

private:
    int32_t mBatch, mHeight, mWidth;
};

// ---------------------------------------------------------------------------
// Creators. One per plugin; the field-name collection they advertise is the
// same list the plugin parses and serializes.

template <class Plugin>
class Creator final : public IPluginCreatorV3One {
public:
    Creator(char const* name, char const* const* fieldNames, int fieldCount) : mName(name) {
        for (int i = 0; i < fieldCount; ++i) mFields.emplace_back(fieldNames[i], nullptr, PluginFieldType::kINT32, 1);
        mCollection.nbFields = static_cast<int32_t>(mFields.size());
        mCollection.fields = mFields.empty() ? nullptr : mFields.data();
    }

    IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        // Build and runtime phases carry the same fields (getFieldsToSerialize returns what createPlugin parsed).
        try {
            return new Plugin(fc);
        } catch (std::exception const& e) {
            std::fprintf(stderr, "[gaia_trt] %s createPlugin failed: %s\n", mName, e.what());
            return nullptr;
        } catch (...) {
            return nullptr;
        }
    }

    PluginFieldCollection const* getFieldNames() noexcept override { return &mCollection; }
    AsciiChar const* getPluginName() const noexcept override { return mName; }
    AsciiChar const* getPluginVersion() const noexcept override { return kPluginVersion; }
    AsciiChar const* getPluginNamespace() const noexcept override { return kPluginNamespace; }

private:
    char const* mName;
    std::vector<PluginField> mFields;
    PluginFieldCollection mCollection{};
};

}  // namespace
}  // namespace gaia_trt

extern "C" bool gaia_trt_layout_register_plugins() {
    using namespace gaia_trt;
    static bool registered = false;
    if (registered) return true;
    try {
        static Creator<DepthwisePlugin> depthwise(kDepthwiseName, kDepthwiseFields, 8);
        static Creator<EpiloguePlugin> epilogue(kEpilogueName, kEpilogueFields, 4);
        static Creator<AttnPrepPlugin> attnPrep(kAttnPrepName, kAttnPrepFields, 1);
        static Creator<AttnFinalPlugin> attnFinal(kAttnFinalName, kAttnFinalFields, 4);
        nvinfer1::IPluginRegistry* registry = getPluginRegistry();
        if (registry == nullptr) return false;
        bool ok = true;
        ok &= registry->registerCreator(depthwise, kPluginNamespace);
        ok &= registry->registerCreator(epilogue, kPluginNamespace);
        ok &= registry->registerCreator(attnPrep, kPluginNamespace);
        ok &= registry->registerCreator(attnFinal, kPluginNamespace);
        registered = ok;
        return ok;
    } catch (...) {
        return false;
    }
}
