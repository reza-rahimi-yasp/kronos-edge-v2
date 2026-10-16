// Out-of-place variant of Gaia's epilogue_kernel, written for this experiment.
//
// The generated epilogue_kernel (gaia_device_code.cuh) updates its tensor in
// place: it reads y[i] and writes y[i]. A TensorRT plugin must not write its
// inputs, so this copy reads from `yin` and writes to `yout`. The arithmetic
// is the generated code unchanged; only the two memory accesses and the
// signature differ (tests/test_device_code.py checks that mechanically).
#pragma once
#include <cuda_runtime.h>
#include "gaia_device_code.cuh"

__global__ void epilogue_kernel_oop(const float4* __restrict__ yin, float4* __restrict__ yout, const float4* __restrict__ res, Aff a, int act,
                                int C, int HW4, int total4){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total4) return;
    int c = (i / HW4) % C;
    float4 v = yin[i];
    if (a.mode == 2){
        float g = a.g[c], b = a.b[c], m = a.m[c], is = a.is[c];
        v.x = bnf(v.x, g, b, m, is); v.y = bnf(v.y, g, b, m, is);
        v.z = bnf(v.z, g, b, m, is); v.w = bnf(v.w, g, b, m, is);
    } else if (a.mode == 1){
        float b = a.b[c];
        v.x = __fadd_rn(v.x, b); v.y = __fadd_rn(v.y, b); v.z = __fadd_rn(v.z, b); v.w = __fadd_rn(v.w, b);
    }
    if (act){ v.x = hsw(v.x); v.y = hsw(v.y); v.z = hsw(v.z); v.w = hsw(v.w); }
    if (res){
        float4 r = res[i];
        v.x = __fadd_rn(v.x, r.x); v.y = __fadd_rn(v.y, r.y); v.z = __fadd_rn(v.z, r.z); v.w = __fadd_rn(v.w, r.w);
    }
    yout[i] = v;
}
