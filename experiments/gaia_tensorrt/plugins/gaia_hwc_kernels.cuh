// Channel-last (TensorRT kHWC) variants of the Gaia kernels.
// These kernels were written for this experiment: they are not Gaia output and
// not verbatim generated code.
//
// TensorRT runs this model's FP32 convolutions in NHWC and had to transpose
// every tensor on its way into and out of the linear-only plugins. These
// kernels read and write the kHWC layout directly ([N][H][W][C] in memory
// for a tensor TensorRT describes as [N, C, H, W]). The arithmetic is the
// generated code's, unchanged: the same Aff, hsw and bnf helpers from
// gaia_device_code.cuh, the same rounding-controlled intrinsics, the same
// accumulation order (kh outer, kw inner, zero padding multiplied in), the
// same pre-activation rule (applied to in-bounds loads only). Only the
// memory indexing and the thread-to-element mapping differ, so that
// neighbouring lanes touch neighbouring channels of one pixel, which is the
// contiguous direction of this layout.
#pragma once
#include <cuda_runtime.h>
#include "gaia_device_code.cuh"

// Depthwise K x K, stride S, "same" padding, fused with the preceding affine +
// hardswish (pre) and the following affine (+ hardswish when act), on kHWC
// tensors. Counterpart of dw_kernel<K, S>.
//   x:  [B][H][W][C]      w: [C][K*K] (TensorRT linear, as for dw_kernel)
//   y:  [B][Ho][Wo][C]    ya: [B][H][W][C] when non-null and S == 1 (the
//                             pre-activated input, the stem's second output)
// Block: CT channels x TY rows of one output tile (TW columns per thread);
// a block loads its input window once into shared memory with the lanes
// strided along C, exactly like dw_kernel tiles one channel plane.
template <int K, int S, int CT>
__global__ void __launch_bounds__(256) dw_kernel_hwc(const float* __restrict__ x, const float* __restrict__ w, Aff ai, int pre,
                                                     Aff ao, int act, float* __restrict__ y, float* __restrict__ ya,
                                                     int C, int H, int W, int Ho, int Wo, int tilesX) {
    constexpr int TY = 256 / CT;            // output rows per block, one per y-lane
    constexpr int TW = (S == 1) ? 8 : 4;    // output columns per thread
    constexpr int P = K / 2;
    constexpr int IH = (TY - 1) * S + K, IW = (TW - 1) * S + K;
    __shared__ float tile[IH][IW][CT];
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * CT + tx;
    const int b = blockIdx.z, c0 = blockIdx.y * CT, c = c0 + tx;
    const int tx0 = (blockIdx.x % tilesX) * TW, ty0 = (blockIdx.x / tilesX) * TY;
    const int ix0 = tx0 * S - P, iy0 = ty0 * S - P;
    const bool cok = c < C;
    // Input affine of this lane's channel (every load iteration of a lane is the same channel: 256 % CT == 0).
    int pm = pre ? ai.mode : 0;
    float gi = 1.f, bi = 0.f, mi = 0.f, isi = 1.f;
    if (cok) {
        if (pm == 2) { gi = ai.g[c]; bi = ai.b[c]; mi = ai.m[c]; isi = ai.is[c]; }
        else if (pm == 1) { bi = ai.b[c]; }
    }
    const float* xb = x + (long long) b * H * W * C;
    for (int idx = tid; idx < IH * IW * CT; idx += 256) {
        int r = idx / (IW * CT), cc = (idx / CT) % IW;
        int ih = iy0 + r, iw = ix0 + cc;
        float v = 0.f;
        if (cok && ih >= 0 && ih < H && iw >= 0 && iw < W) {
            v = __ldg(xb + ((long long) ih * W + iw) * C + c);
            if (pre) {
                if (pm == 2) v = bnf(v, gi, bi, mi, isi);
                else if (pm == 1) v = __fadd_rn(v, bi);
                v = hsw(v);
            }
        }
        tile[r][cc][tx] = v;
    }
    __syncthreads();
    if (!cok) return;
    float wr[K * K];
#pragma unroll
    for (int i = 0; i < K * K; i++) wr[i] = w[c * K * K + i];
    float go = 1.f, bo = 0.f, mo = 0.f, iso = 1.f;
    if (ao.mode == 2) { go = ao.g[c]; bo = ao.b[c]; mo = ao.m[c]; iso = ao.is[c]; }
    else if (ao.mode == 1) { bo = ao.b[c]; }
    const float init = (ao.mode == 1) ? bo : 0.f;
    const int oy = ty0 + ty;
    if (oy >= Ho) return;
    float* yrow = y + ((long long) b * Ho + oy) * Wo * C;
    float* yarow = (S == 1 && ya) ? ya + ((long long) b * H + oy) * W * C : nullptr;
#pragma unroll
    for (int ox_l = 0; ox_l < TW; ox_l++) {
        int ox = tx0 + ox_l;
        if (ox < Wo) {
            float acc = init;
#pragma unroll
            for (int kh = 0; kh < K; kh++)
#pragma unroll
                for (int kw = 0; kw < K; kw++)
                    acc = __fmaf_rn(wr[kh * K + kw], tile[ty * S + kh][ox_l * S + kw][tx], acc);
            float v = acc;
            if (ao.mode == 2) v = bnf(v, go, bo, mo, iso);
            if (act) v = hsw(v);
            yrow[(long long) ox * C + c] = v;
            if (yarow) yarow[(long long) ox * C + c] = tile[ty * S + P][ox_l * S + P][tx];
        }
    }
}

// Affine (+ hardswish) (+ residual) on a kHWC tensor, out of place; four
// consecutive elements are four consecutive channels of one pixel, so each
// lane of a float4 carries its own channel's parameters. Requires C % 4 == 0.
// Counterpart of epilogue_kernel_oop (gaia_adapted_epilogue.cuh).
__global__ void epilogue_kernel_hwc(const float4* __restrict__ yin, float4* __restrict__ yout, const float4* __restrict__ res, Aff a, int act,
                                    int C, int total4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total4) return;
    int c = (int) (((long long) i * 4) % C);
    float4 v = yin[i];
    if (a.mode == 2) {
        v.x = bnf(v.x, a.g[c], a.b[c], a.m[c], a.is[c]);
        v.y = bnf(v.y, a.g[c + 1], a.b[c + 1], a.m[c + 1], a.is[c + 1]);
        v.z = bnf(v.z, a.g[c + 2], a.b[c + 2], a.m[c + 2], a.is[c + 2]);
        v.w = bnf(v.w, a.g[c + 3], a.b[c + 3], a.m[c + 3], a.is[c + 3]);
    } else if (a.mode == 1) {
        v.x = __fadd_rn(v.x, a.b[c]); v.y = __fadd_rn(v.y, a.b[c + 1]); v.z = __fadd_rn(v.z, a.b[c + 2]); v.w = __fadd_rn(v.w, a.b[c + 3]);
    }
    if (act) { v.x = hsw(v.x); v.y = hsw(v.y); v.z = hsw(v.z); v.w = hsw(v.w); }
    if (res) {
        float4 r = res[i];
        v.x = __fadd_rn(v.x, r.x); v.y = __fadd_rn(v.y, r.y); v.z = __fadd_rn(v.z, r.z); v.w = __fadd_rn(v.w, r.w);
    }
    yout[i] = v;
}

// LiteMLA prep from kHWC qkv/agg ([B][H][W][48*Hh]) to the linear q, k
// ([BH,16,N], relu) and vpad ([BH,N,17], ones in the last column) the batched
// matmuls consume. Counterpart of attn_prep_kernel: one block per (bh, 32
// pixels); the 48 channels of a pixel are contiguous in this layout, so the
// load runs along channels and the stores along pixels through shared memory.
__global__ void attn_prep_kernel_hwc(const float* __restrict__ A, const float* __restrict__ Bt,
                                     float* __restrict__ q, float* __restrict__ k, float* __restrict__ vpad,
                                     int Hh, int N) {
    __shared__ float s[48][33];
    int bh = blockIdx.y;
    int heads = 2 * Hh;
    int b = bh / heads, h = bh % heads;
    int C = 48 * Hh;
    const float* base = (h < Hh ? A : Bt) + (long long) b * N * C + (long long) (h % Hh) * 48;
    int n0 = blockIdx.x * 32;
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int idx = tid; idx < 32 * 48; idx += 256) {
        int nl = idx / 48, cc = idx % 48;
        int n = n0 + nl;
        s[cc][nl] = (n < N) ? base[(long long) n * C + cc] : 0.f;
    }
    __syncthreads();
    int n = n0 + threadIdx.x;
    if (n < N) {
        for (int cc = threadIdx.y; cc < 32; cc += 8) {
            float v = s[cc][threadIdx.x];
            if (cc < 16) q[((long long) bh * 16 + cc) * N + n] = fmaxf(v, 0.f);
            else k[((long long) bh * 16 + (cc - 16)) * N + n] = fmaxf(v, 0.f);
        }
    }
    int cnt = min(32, N - n0) * 17;
    float* vp = vpad + ((long long) bh * N + n0) * 17;
    for (int j = tid; j < cnt; j += 256) {
        int nl = j / 17, e = j % 17;
        vp[j] = (e == 16) ? 1.f : s[32 + e][nl];
    }
}

// r: [BH,N,17] linear -> out: [B][H][W][heads*16] (kHWC view of [B, heads*16, H, W]),
// out[b][n][hh*16+e] = r[bh][n][e] / (r[bh][n][16] + 1e-5). Counterpart of
// attn_final_kernel; lanes run along e so a pixel's 16 values of one head
// are written contiguously.
__global__ void attn_final_kernel_hwc(const float* __restrict__ r, float* __restrict__ out, int N, int heads) {
    __shared__ float sr[32 * 17];
    int bh = blockIdx.y;
    int b = bh / heads, hh = bh % heads;
    int n0 = blockIdx.x * 32;
    int cnt = min(32, N - n0) * 17;
    const float* rp = r + ((long long) bh * N + n0) * 17;
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) sr[j] = rp[j];
    __syncthreads();
    int Cout = heads * 16;
    for (int idx = threadIdx.x; idx < 16 * 32; idx += blockDim.x) {
        int nl = idx >> 4, e = idx & 15;
        if (n0 + nl < N) {
            float den = __fadd_rn(sr[nl * 17 + 16], 1e-5f);
            out[((long long) b * N + n0 + nl) * Cout + hh * 16 + e] = __fdiv_rn(sr[nl * 17 + e], den);
        }
    }
}
