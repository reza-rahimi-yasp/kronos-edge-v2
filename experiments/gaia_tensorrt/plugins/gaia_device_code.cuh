// Gaia-generated CUDA device code, extracted verbatim from kernel.py
// (sha256 980698e39f89ac6371c3d766e872f688c4645e1a2150499792dbad373527c032) by exporter/extract_gaia_device_code.py.
// Do not edit: the segments between the BEGIN/END markers are the generated
// code byte for byte; plugins/provenance.json records their hashes.
#pragma once
#include <cuda_runtime.h>

// BEGIN verbatim segment affine_helpers
struct Aff { const float* g; const float* b; const float* m; const float* is; int mode; };

// PyTorch computes div(tensor, cpu_scalar) as tensor * (1/scalar)
__device__ __forceinline__ float hsw(float x){
    constexpr float INV6 = 1.f / 6.f;
    float c = fminf(fmaxf(__fadd_rn(x, 3.f), 0.f), 6.f);
    return __fmul_rn(__fmul_rn(x, c), INV6);
}
// PyTorch batch_norm_transform_input_kernel: gamma * (x - mean) * invstd + beta
__device__ __forceinline__ float bnf(float x, float g, float b, float m, float is){
    return __fmaf_rn(__fmul_rn(g, __fsub_rn(x, m)), is, b);
}
// END verbatim segment affine_helpers

// BEGIN verbatim segment kernels
template<int K, int S>
__global__ void __launch_bounds__(128) dw_kernel(const float* __restrict__ x, const float* __restrict__ w, Aff ai, int pre,
                          Aff ao, int act, float* __restrict__ y, float* __restrict__ ya,
                          int C, int H, int W, int Ho, int Wo, int tilesX){
    constexpr int TW = 32, TH = 16, R = 4, P = K / 2;
    constexpr int IW = (TW - 1) * S + K, IH = (TH - 1) * S + K;
    constexpr int NR = (R - 1) * S + K;
    __shared__ float tile[IH][IW + 1];
    int c = blockIdx.y, b = blockIdx.z;
    int tx0 = (blockIdx.x % tilesX) * TW, ty0 = (blockIdx.x / tilesX) * TH;
    const float* xp = x + ((long long)b * C + c) * H * W;
    int ix0 = tx0 * S - P, iy0 = ty0 * S - P;
    int pm = pre ? ai.mode : 0;
    float gi = 1.f, bi = 0.f, mi = 0.f, isi = 1.f;
    if (pm == 2){ gi = ai.g[c]; bi = ai.b[c]; mi = ai.m[c]; isi = ai.is[c]; }
    else if (pm == 1){ bi = ai.b[c]; }
    for (int r = threadIdx.y; r < IH; r += 4){
        int ih = iy0 + r;
        bool rok = (ih >= 0 && ih < H);
        const float* rowp = xp + (long long)ih * W;
        for (int cc = threadIdx.x; cc < IW; cc += 32){
            int iw = ix0 + cc;
            float v = 0.f;
            if (rok && iw >= 0 && iw < W){
                v = __ldg(rowp + iw);
                if (pre){
                    if (pm == 2) v = bnf(v, gi, bi, mi, isi);
                    else if (pm == 1) v = __fadd_rn(v, bi);
                    v = hsw(v);
                }
            }
            tile[r][cc] = v;
        }
    }
    float wr[K * K];
#pragma unroll
    for (int i = 0; i < K * K; i++) wr[i] = w[c * K * K + i];
    float go = 1.f, bo = 0.f, mo = 0.f, iso = 1.f;
    if (ao.mode == 2){ go = ao.g[c]; bo = ao.b[c]; mo = ao.m[c]; iso = ao.is[c]; }
    else if (ao.mode == 1){ bo = ao.b[c]; }
    float init = (ao.mode == 1) ? bo : 0.f;
    __syncthreads();
    int oyl0 = threadIdx.y * R;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = init;
    const int colb = threadIdx.x * S;
#pragma unroll
    for (int j = 0; j < NR; j++){
        float row[K];
#pragma unroll
        for (int kw = 0; kw < K; kw++) row[kw] = tile[oyl0 * S + j][colb + kw];
#pragma unroll
        for (int r = 0; r < R; r++){
            const int kh = j - r * S;
            if (kh >= 0 && kh < K){
#pragma unroll
                for (int kw = 0; kw < K; kw++)
                    acc[r] = __fmaf_rn(wr[kh * K + kw], row[kw], acc[r]);
            }
        }
    }
    int ox = tx0 + threadIdx.x;
    long long plane = ((long long)b * C + c) * Ho * Wo;
    float* yp = y + plane;
    if (ox < Wo){
#pragma unroll
        for (int r = 0; r < R; r++){
            int oyl = oyl0 + r;
            int oy = ty0 + oyl;
            if (oy < Ho){
                float v = acc[r];
                if (ao.mode == 2) v = bnf(v, go, bo, mo, iso);
                if (act) v = hsw(v);
                yp[(long long)oy * Wo + ox] = v;
                if (S == 1 && ya) ya[plane + (long long)oy * Wo + ox] = tile[oyl + P][threadIdx.x + P];
            }
        }
    }
}

__global__ void epilogue_kernel(float4* __restrict__ y, const float4* __restrict__ res, Aff a, int act,
                                int C, int HW4, int total4){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total4) return;
    int c = (i / HW4) % C;
    float4 v = y[i];
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
    y[i] = v;
}

// q: [BH,16,N] relu, k: [BH,16,N] relu, vpad: [BH,N,17]
__global__ void attn_prep_kernel(const float* __restrict__ A, const float* __restrict__ Bt,
                                 float* __restrict__ q, float* __restrict__ k, float* __restrict__ vpad,
                                 int Hh, int N){
    __shared__ float sv[16][33];
    int bh = blockIdx.y;
    int heads = 2 * Hh;
    int b = bh / heads, h = bh % heads;
    const float* base = (h < Hh ? A : Bt) + ((long long)b * Hh * 48 + (long long)(h % Hh) * 48) * N;
    int n0 = blockIdx.x * 32;
    int n = n0 + threadIdx.x;
    bool ok = n < N;
    for (int c = threadIdx.y; c < 48; c += 8){
        float v = ok ? base[(long long)c * N + n] : 0.f;
        if (c < 16){
            if (ok) q[((long long)bh * 16 + c) * N + n] = fmaxf(v, 0.f);
        } else if (c < 32){
            if (ok) k[((long long)bh * 16 + (c - 16)) * N + n] = fmaxf(v, 0.f);
        } else {
            sv[c - 32][threadIdx.x] = v;
        }
    }
    __syncthreads();
    int tid = threadIdx.y * 32 + threadIdx.x;
    int cnt = min(32, N - n0) * 17;
    float* vp = vpad + ((long long)bh * N + n0) * 17;
    for (int j = tid; j < cnt; j += 256){
        int nl = j / 17, e = j % 17;
        vp[j] = (e == 16) ? 1.f : sv[e][nl];
    }
}

// r: [BH,N,17] -> out [BH*16, N] : r[n][e] / (r[n][16] + 1e-5)
__global__ void attn_final_kernel(const float* __restrict__ r, float* __restrict__ out, int N){
    __shared__ float sr[32 * 17];
    int bh = blockIdx.y;
    int n0 = blockIdx.x * 32;
    int cnt = min(32, N - n0) * 17;
    const float* rp = r + ((long long)bh * N + n0) * 17;
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) sr[j] = rp[j];
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * 32; idx += blockDim.x){
        int e = idx >> 5, nl = idx & 31;
        if (n0 + nl < N){
            float den = __fadd_rn(sr[nl * 17 + 16], 1e-5f);
            out[((long long)bh * 16 + e) * N + n0 + nl] = __fdiv_rn(sr[nl * 17 + e], den);
        }
    }
}
// END verbatim segment kernels
