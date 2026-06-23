// likelihood_scorer_kernels.cpp — part of the likelihood_scorer_unit.cpp umbrella.

#include "likelihood_scorer.h"
#line 98 "likelihood_scorer.cpp"
static bool _detect_avx512() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512dq")
        && __builtin_cpu_supports("avx512vl");
#else
    return false;
#endif
}
static const bool g_have_avx512 = _detect_avx512();

// One-time diagnostic at process startup: prints to stderr if SIMD_VERBOSE=1.
static const bool g_simd_diag_printed = []() {
    if (const char* v = std::getenv("SIMD_VERBOSE")) {
        if (v[0] && v[0] != '0') {
            std::fprintf(stderr,
                "[SIMD] avx512=%s (avx512f=%d avx512dq=%d avx512vl=%d)\n",
                g_have_avx512 ? "ENABLED" : "disabled",
#if defined(__GNUC__) || defined(__clang__)
                __builtin_cpu_supports("avx512f"),
                __builtin_cpu_supports("avx512dq"),
                __builtin_cpu_supports("avx512vl")
#else
                0, 0, 0
#endif
            );
        }
    }
    return true;
}();

// AVX-512 K=20 _p_apply: 8-wide vectors, 4-row inner unroll. Bit-identical to
// the AVX-2 kernel (same per-row accumulation order, doubled width).
__attribute__((target("avx512f,avx512vl,avx512dq")))
static void _p_apply_k20_avx512(const double* P, const double* clv,
                                double* out, int L) {
    constexpr int BLOCK = 16;
    constexpr int I_ROWS = 4;
    int Lblk = L - (L % BLOCK);
    for (int l0 = 0; l0 < Lblk; l0 += BLOCK) {
        for (int i0 = 0; i0 < 20; i0 += I_ROWS) {
            __m512d a00 = _mm512_setzero_pd(), a01 = _mm512_setzero_pd();
            __m512d a10 = _mm512_setzero_pd(), a11 = _mm512_setzero_pd();
            __m512d a20 = _mm512_setzero_pd(), a21 = _mm512_setzero_pd();
            __m512d a30 = _mm512_setzero_pd(), a31 = _mm512_setzero_pd();
            for (int j = 0; j < 20; j++) {
                const double* clv_j = clv + j * L + l0;
                __m512d c0 = _mm512_loadu_pd(clv_j +  0);
                __m512d c1 = _mm512_loadu_pd(clv_j +  8);
                __m512d Pi0 = _mm512_set1_pd(P[(i0 + 0) * 20 + j]);
                __m512d Pi1 = _mm512_set1_pd(P[(i0 + 1) * 20 + j]);
                __m512d Pi2 = _mm512_set1_pd(P[(i0 + 2) * 20 + j]);
                __m512d Pi3 = _mm512_set1_pd(P[(i0 + 3) * 20 + j]);
                a00 = _mm512_fmadd_pd(Pi0, c0, a00);
                a01 = _mm512_fmadd_pd(Pi0, c1, a01);
                a10 = _mm512_fmadd_pd(Pi1, c0, a10);
                a11 = _mm512_fmadd_pd(Pi1, c1, a11);
                a20 = _mm512_fmadd_pd(Pi2, c0, a20);
                a21 = _mm512_fmadd_pd(Pi2, c1, a21);
                a30 = _mm512_fmadd_pd(Pi3, c0, a30);
                a31 = _mm512_fmadd_pd(Pi3, c1, a31);
            }
            double* o0 = out + (i0 + 0) * L + l0;
            double* o1 = out + (i0 + 1) * L + l0;
            double* o2 = out + (i0 + 2) * L + l0;
            double* o3 = out + (i0 + 3) * L + l0;
            _mm512_storeu_pd(o0 + 0, a00); _mm512_storeu_pd(o0 + 8, a01);
            _mm512_storeu_pd(o1 + 0, a10); _mm512_storeu_pd(o1 + 8, a11);
            _mm512_storeu_pd(o2 + 0, a20); _mm512_storeu_pd(o2 + 8, a21);
            _mm512_storeu_pd(o3 + 0, a30); _mm512_storeu_pd(o3 + 8, a31);
        }
    }
    // Scalar tail for patterns L - Lblk
    for (int l = Lblk; l < L; l++) {
        for (int i = 0; i < 20; i++) {
            double s = 0.0;
            const double* Pi = P + i * 20;
            for (int j = 0; j < 20; j++) s += Pi[j] * clv[j * L + l];
            out[i * L + l] = s;
        }
    }
}

// AVX-512 kernel for K=20 _p_apply_mul (fused: out[i,l] = (sum_j P[i,j] * clv[j,l]) * mul[i,l]).
__attribute__((target("avx512f,avx512vl,avx512dq")))
static void _p_apply_mul_k20_avx512(const double* P, const double* clv,
                                     const double* mul, double* out, int L) {
    constexpr int BLOCK = 16;
    constexpr int I_ROWS = 4;
    int Lblk = L - (L % BLOCK);
    for (int l0 = 0; l0 < Lblk; l0 += BLOCK) {
        for (int i0 = 0; i0 < 20; i0 += I_ROWS) {
            __m512d a00 = _mm512_setzero_pd(), a01 = _mm512_setzero_pd();
            __m512d a10 = _mm512_setzero_pd(), a11 = _mm512_setzero_pd();
            __m512d a20 = _mm512_setzero_pd(), a21 = _mm512_setzero_pd();
            __m512d a30 = _mm512_setzero_pd(), a31 = _mm512_setzero_pd();
            for (int j = 0; j < 20; j++) {
                const double* clv_j = clv + j * L + l0;
                __m512d c0 = _mm512_loadu_pd(clv_j +  0);
                __m512d c1 = _mm512_loadu_pd(clv_j +  8);
                __m512d Pi0 = _mm512_set1_pd(P[(i0 + 0) * 20 + j]);
                __m512d Pi1 = _mm512_set1_pd(P[(i0 + 1) * 20 + j]);
                __m512d Pi2 = _mm512_set1_pd(P[(i0 + 2) * 20 + j]);
                __m512d Pi3 = _mm512_set1_pd(P[(i0 + 3) * 20 + j]);
                a00 = _mm512_fmadd_pd(Pi0, c0, a00);
                a01 = _mm512_fmadd_pd(Pi0, c1, a01);
                a10 = _mm512_fmadd_pd(Pi1, c0, a10);
                a11 = _mm512_fmadd_pd(Pi1, c1, a11);
                a20 = _mm512_fmadd_pd(Pi2, c0, a20);
                a21 = _mm512_fmadd_pd(Pi2, c1, a21);
                a30 = _mm512_fmadd_pd(Pi3, c0, a30);
                a31 = _mm512_fmadd_pd(Pi3, c1, a31);
            }
            const double* m0 = mul + (i0 + 0) * L + l0;
            const double* m1 = mul + (i0 + 1) * L + l0;
            const double* m2 = mul + (i0 + 2) * L + l0;
            const double* m3 = mul + (i0 + 3) * L + l0;
            double* o0 = out + (i0 + 0) * L + l0;
            double* o1 = out + (i0 + 1) * L + l0;
            double* o2 = out + (i0 + 2) * L + l0;
            double* o3 = out + (i0 + 3) * L + l0;
            _mm512_storeu_pd(o0 + 0, _mm512_mul_pd(a00, _mm512_loadu_pd(m0 + 0)));
            _mm512_storeu_pd(o0 + 8, _mm512_mul_pd(a01, _mm512_loadu_pd(m0 + 8)));
            _mm512_storeu_pd(o1 + 0, _mm512_mul_pd(a10, _mm512_loadu_pd(m1 + 0)));
            _mm512_storeu_pd(o1 + 8, _mm512_mul_pd(a11, _mm512_loadu_pd(m1 + 8)));
            _mm512_storeu_pd(o2 + 0, _mm512_mul_pd(a20, _mm512_loadu_pd(m2 + 0)));
            _mm512_storeu_pd(o2 + 8, _mm512_mul_pd(a21, _mm512_loadu_pd(m2 + 8)));
            _mm512_storeu_pd(o3 + 0, _mm512_mul_pd(a30, _mm512_loadu_pd(m3 + 0)));
            _mm512_storeu_pd(o3 + 8, _mm512_mul_pd(a31, _mm512_loadu_pd(m3 + 8)));
        }
    }
    // Scalar tail
    for (int l = Lblk; l < L; l++) {
        for (int i = 0; i < 20; i++) {
            double s = 0.0;
            const double* Pi = P + i * 20;
            for (int j = 0; j < 20; j++) s += Pi[j] * clv[j * L + l];
            out[i * L + l] = s * mul[i * L + l];
        }
    }
}

// AVX-512 fused-pair K=20: out[i,l] = (P1@clv1)[i,l] * (P2@clv2)[i,l], both
// projections accumulated then multiplied in one pass (no intermediate buffer).
__attribute__((target("avx512f,avx512vl,avx512dq")))
static void _p_apply_pair_k20_avx512(const double* P1, const double* clv1,
                                      const double* P2, const double* clv2,
                                      double* out, int L) {
    constexpr int BLOCK = 16;
    constexpr int I_ROWS = 4;
    int Lblk = L - (L % BLOCK);
    for (int l0 = 0; l0 < Lblk; l0 += BLOCK) {
        for (int i0 = 0; i0 < 20; i0 += I_ROWS) {
            __m512d a00 = _mm512_setzero_pd(), a01 = _mm512_setzero_pd();
            __m512d a10 = _mm512_setzero_pd(), a11 = _mm512_setzero_pd();
            __m512d a20 = _mm512_setzero_pd(), a21 = _mm512_setzero_pd();
            __m512d a30 = _mm512_setzero_pd(), a31 = _mm512_setzero_pd();
            __m512d b00 = _mm512_setzero_pd(), b01 = _mm512_setzero_pd();
            __m512d b10 = _mm512_setzero_pd(), b11 = _mm512_setzero_pd();
            __m512d b20 = _mm512_setzero_pd(), b21 = _mm512_setzero_pd();
            __m512d b30 = _mm512_setzero_pd(), b31 = _mm512_setzero_pd();
            for (int j = 0; j < 20; j++) {
                const double* c1j = clv1 + j * L + l0;
                const double* c2j = clv2 + j * L + l0;
                __m512d c1_0 = _mm512_loadu_pd(c1j +  0);
                __m512d c1_1 = _mm512_loadu_pd(c1j +  8);
                __m512d c2_0 = _mm512_loadu_pd(c2j +  0);
                __m512d c2_1 = _mm512_loadu_pd(c2j +  8);
                __m512d P1i0 = _mm512_set1_pd(P1[(i0 + 0) * 20 + j]);
                __m512d P1i1 = _mm512_set1_pd(P1[(i0 + 1) * 20 + j]);
                __m512d P1i2 = _mm512_set1_pd(P1[(i0 + 2) * 20 + j]);
                __m512d P1i3 = _mm512_set1_pd(P1[(i0 + 3) * 20 + j]);
                a00 = _mm512_fmadd_pd(P1i0, c1_0, a00);
                a01 = _mm512_fmadd_pd(P1i0, c1_1, a01);
                a10 = _mm512_fmadd_pd(P1i1, c1_0, a10);
                a11 = _mm512_fmadd_pd(P1i1, c1_1, a11);
                a20 = _mm512_fmadd_pd(P1i2, c1_0, a20);
                a21 = _mm512_fmadd_pd(P1i2, c1_1, a21);
                a30 = _mm512_fmadd_pd(P1i3, c1_0, a30);
                a31 = _mm512_fmadd_pd(P1i3, c1_1, a31);
                __m512d P2i0 = _mm512_set1_pd(P2[(i0 + 0) * 20 + j]);
                __m512d P2i1 = _mm512_set1_pd(P2[(i0 + 1) * 20 + j]);
                __m512d P2i2 = _mm512_set1_pd(P2[(i0 + 2) * 20 + j]);
                __m512d P2i3 = _mm512_set1_pd(P2[(i0 + 3) * 20 + j]);
                b00 = _mm512_fmadd_pd(P2i0, c2_0, b00);
                b01 = _mm512_fmadd_pd(P2i0, c2_1, b01);
                b10 = _mm512_fmadd_pd(P2i1, c2_0, b10);
                b11 = _mm512_fmadd_pd(P2i1, c2_1, b11);
                b20 = _mm512_fmadd_pd(P2i2, c2_0, b20);
                b21 = _mm512_fmadd_pd(P2i2, c2_1, b21);
                b30 = _mm512_fmadd_pd(P2i3, c2_0, b30);
                b31 = _mm512_fmadd_pd(P2i3, c2_1, b31);
            }
            double* o0 = out + (i0 + 0) * L + l0;
            double* o1 = out + (i0 + 1) * L + l0;
            double* o2 = out + (i0 + 2) * L + l0;
            double* o3 = out + (i0 + 3) * L + l0;
            _mm512_storeu_pd(o0 + 0, _mm512_mul_pd(a00, b00));
            _mm512_storeu_pd(o0 + 8, _mm512_mul_pd(a01, b01));
            _mm512_storeu_pd(o1 + 0, _mm512_mul_pd(a10, b10));
            _mm512_storeu_pd(o1 + 8, _mm512_mul_pd(a11, b11));
            _mm512_storeu_pd(o2 + 0, _mm512_mul_pd(a20, b20));
            _mm512_storeu_pd(o2 + 8, _mm512_mul_pd(a21, b21));
            _mm512_storeu_pd(o3 + 0, _mm512_mul_pd(a30, b30));
            _mm512_storeu_pd(o3 + 8, _mm512_mul_pd(a31, b31));
        }
    }
    // Scalar tail
    for (int l = Lblk; l < L; l++) {
        for (int i = 0; i < 20; i++) {
            double s1 = 0.0, s2 = 0.0;
            const double* Pi1 = P1 + i * 20;
            const double* Pi2 = P2 + i * 20;
            for (int j = 0; j < 20; j++) {
                s1 += Pi1[j] * clv1[j * L + l];
                s2 += Pi2[j] * clv2[j * L + l];
            }
            out[i * L + l] = s1 * s2;
        }
    }
}

// AVX-512 fused tip-internal K=20: out[i,l] = lookup_tip[i,codes[l]] *
// (P_int @ clv_int)[i,l], computed in registers (no intermediate buffer).
__attribute__((target("avx512f,avx512vl,avx512dq")))
static void _p_apply_tip_int_pair_k20_avx512(
    const double* lookup_tip, int n_tip_codes,
    const uint8_t* codes_tip,
    const double* P_int, const double* clv_int,
    double* out, int L)
{
    constexpr int BLOCK = 8;
    constexpr int I_ROWS = 4;
    int Lblk = L - (L % BLOCK);

    for (int l0 = 0; l0 < Lblk; l0 += BLOCK) {
        // 8 codes for this pattern block, shared across all i rows.
        __m128i u8_8 = _mm_loadl_epi64((const __m128i*)(codes_tip + l0));
        __m256i idx  = _mm256_cvtepu8_epi32(u8_8);

        for (int i0 = 0; i0 < 20; i0 += I_ROWS) {
            __m512d a0 = _mm512_setzero_pd();
            __m512d a1 = _mm512_setzero_pd();
            __m512d a2 = _mm512_setzero_pd();
            __m512d a3 = _mm512_setzero_pd();

            // Internal-side dgemv accumulation.
            for (int j = 0; j < 20; j++) {
                __m512d c = _mm512_loadu_pd(clv_int + j * L + l0);
                __m512d Pi0 = _mm512_set1_pd(P_int[(i0 + 0) * 20 + j]);
                __m512d Pi1 = _mm512_set1_pd(P_int[(i0 + 1) * 20 + j]);
                __m512d Pi2 = _mm512_set1_pd(P_int[(i0 + 2) * 20 + j]);
                __m512d Pi3 = _mm512_set1_pd(P_int[(i0 + 3) * 20 + j]);
                a0 = _mm512_fmadd_pd(Pi0, c, a0);
                a1 = _mm512_fmadd_pd(Pi1, c, a1);
                a2 = _mm512_fmadd_pd(Pi2, c, a2);
                a3 = _mm512_fmadd_pd(Pi3, c, a3);
            }

            // Tip-side gather: v[i_row, l] = lookup_tip[i, codes_tip[l]].
            __m512d v0 = _mm512_i32gather_pd(idx,
                lookup_tip + (i0 + 0) * n_tip_codes, 8);
            __m512d v1 = _mm512_i32gather_pd(idx,
                lookup_tip + (i0 + 1) * n_tip_codes, 8);
            __m512d v2 = _mm512_i32gather_pd(idx,
                lookup_tip + (i0 + 2) * n_tip_codes, 8);
            __m512d v3 = _mm512_i32gather_pd(idx,
                lookup_tip + (i0 + 3) * n_tip_codes, 8);

            _mm512_storeu_pd(out + (i0 + 0) * L + l0, _mm512_mul_pd(v0, a0));
            _mm512_storeu_pd(out + (i0 + 1) * L + l0, _mm512_mul_pd(v1, a1));
            _mm512_storeu_pd(out + (i0 + 2) * L + l0, _mm512_mul_pd(v2, a2));
            _mm512_storeu_pd(out + (i0 + 3) * L + l0, _mm512_mul_pd(v3, a3));
        }
    }

    // Scalar tail
    for (int l = Lblk; l < L; l++) {
        for (int i = 0; i < 20; i++) {
            double s = 0.0;
            const double* Pi = P_int + i * 20;
            for (int j = 0; j < 20; j++) s += Pi[j] * clv_int[j * L + l];
            out[i * L + l] = lookup_tip[i * n_tip_codes + codes_tip[l]] * s;
        }
    }
}

// AVX-512 fused tip-tip K=20: out[i,l] = lookup_1[i,codes_1[l]] *
// lookup_2[i,codes_2[l]] — two gathers, one multiply, one store (no j-loop).
__attribute__((target("avx512f,avx512vl,avx512dq")))
static void _p_apply_tip_tip_pair_k20_avx512(
    const double* lookup_1, int n_codes_1,
    const uint8_t* codes_1,
    const double* lookup_2, int n_codes_2,
    const uint8_t* codes_2,
    double* out, int L)
{
    constexpr int BLOCK  = 8;
    constexpr int I_ROWS = 4;
    int Lblk = L - (L % BLOCK);

    for (int l0 = 0; l0 < Lblk; l0 += BLOCK) {
        // 8 codes per side per pattern block, shared across all i rows.
        __m128i u8_1 = _mm_loadl_epi64((const __m128i*)(codes_1 + l0));
        __m128i u8_2 = _mm_loadl_epi64((const __m128i*)(codes_2 + l0));
        __m256i idx1 = _mm256_cvtepu8_epi32(u8_1);
        __m256i idx2 = _mm256_cvtepu8_epi32(u8_2);

        for (int i0 = 0; i0 < 20; i0 += I_ROWS) {
            // Gather side 1: a[i_row, l] = lookup_1[i, codes_1[l]].
            __m512d a0 = _mm512_i32gather_pd(idx1,
                lookup_1 + (i0 + 0) * n_codes_1, 8);
            __m512d a1 = _mm512_i32gather_pd(idx1,
                lookup_1 + (i0 + 1) * n_codes_1, 8);
            __m512d a2 = _mm512_i32gather_pd(idx1,
                lookup_1 + (i0 + 2) * n_codes_1, 8);
            __m512d a3 = _mm512_i32gather_pd(idx1,
                lookup_1 + (i0 + 3) * n_codes_1, 8);

            // Gather side 2.
            __m512d b0 = _mm512_i32gather_pd(idx2,
                lookup_2 + (i0 + 0) * n_codes_2, 8);
            __m512d b1 = _mm512_i32gather_pd(idx2,
                lookup_2 + (i0 + 1) * n_codes_2, 8);
            __m512d b2 = _mm512_i32gather_pd(idx2,
                lookup_2 + (i0 + 2) * n_codes_2, 8);
            __m512d b3 = _mm512_i32gather_pd(idx2,
                lookup_2 + (i0 + 3) * n_codes_2, 8);

            _mm512_storeu_pd(out + (i0 + 0) * L + l0, _mm512_mul_pd(a0, b0));
            _mm512_storeu_pd(out + (i0 + 1) * L + l0, _mm512_mul_pd(a1, b1));
            _mm512_storeu_pd(out + (i0 + 2) * L + l0, _mm512_mul_pd(a2, b2));
            _mm512_storeu_pd(out + (i0 + 3) * L + l0, _mm512_mul_pd(a3, b3));
        }
    }

    // Scalar tail
    for (int l = Lblk; l < L; l++) {
        for (int i = 0; i < 20; i++) {
            out[i * L + l] = lookup_1[i * n_codes_1 + codes_1[l]]
                           * lookup_2[i * n_codes_2 + codes_2[l]];
        }
    }
}

// ================================================================== //
// Core helpers                                                        //
// ================================================================== //

// Per-site CLV scaling (in-place).
// For each pattern l: if max over k of clv[k*L+l] < SCALE_THRESHOLD,
//   multiply clv[k*L+l] by SCALE_FACTOR for all k, add LOG_SCALE to log_scale[l].
void LikelihoodScorer::_scale_clv_inplace(double* clv, double* log_scale) const {
    int K = K_, L = L_;
#ifdef __AVX2__
    // Process patterns 4 at a time. For each block of 4 columns, gather the
    // max across K rows (horizontal reduction), then scale columns whose max
    // is below threshold.
    int L4 = L & ~3;
    __m256d vthresh = _mm256_set1_pd(SCALE_THRESHOLD);
    __m256d vfactor = _mm256_set1_pd(SCALE_FACTOR);
    __m256d vzero   = _mm256_setzero_pd();
    for (int l = 0; l < L4; l += 4) {
        __m256d vmax = _mm256_loadu_pd(clv + 0 * L + l);
        for (int k = 1; k < K; k++)
            vmax = _mm256_max_pd(vmax, _mm256_loadu_pd(clv + k * L + l));

        // mask: max > 0 AND max < threshold (per lane).
        __m256d gt0  = _mm256_cmp_pd(vmax, vzero,   _CMP_GT_OQ);
        __m256d ltth = _mm256_cmp_pd(vmax, vthresh, _CMP_LT_OQ);
        __m256d mask = _mm256_and_pd(gt0, ltth);
        int mask_bits = _mm256_movemask_pd(mask);
        if (mask_bits == 0) continue;

        // Scale columns where mask is set: multiply by factor, set log_scale.
        for (int k = 0; k < K; k++) {
            __m256d v = _mm256_loadu_pd(clv + k * L + l);
            __m256d scaled = _mm256_mul_pd(v, vfactor);
            __m256d picked = _mm256_blendv_pd(v, scaled, mask);
            _mm256_storeu_pd(clv + k * L + l, picked);
        }
        // Update log_scale per pattern.
        if (mask_bits & 1) log_scale[l + 0] += LOG_SCALE;
        if (mask_bits & 2) log_scale[l + 1] += LOG_SCALE;
        if (mask_bits & 4) log_scale[l + 2] += LOG_SCALE;
        if (mask_bits & 8) log_scale[l + 3] += LOG_SCALE;
    }
    // Tail: scalar for remaining < 4 patterns.
    for (int l = L4; l < L; l++) {
        double max_val = 0.0;
        for (int k = 0; k < K; k++) {
            double v = clv[k * L + l];
            if (v > max_val) max_val = v;
        }
        if (max_val > 0.0 && max_val < SCALE_THRESHOLD) {
            for (int k = 0; k < K; k++) clv[k * L + l] *= SCALE_FACTOR;
            log_scale[l] += LOG_SCALE;
        }
    }
#else
    for (int l = 0; l < L; l++) {
        double max_val = 0.0;
        for (int k = 0; k < K; k++) {
            double v = clv[k * L + l];
            if (v > max_val) max_val = v;
        }
        if (max_val > 0.0 && max_val < SCALE_THRESHOLD) {
            for (int k = 0; k < K; k++) clv[k * L + l] *= SCALE_FACTOR;
            log_scale[l] += LOG_SCALE;
        }
    }
#endif
}

// P @ clv -> out. P is K*K row-major, clv is K*L (states x patterns).
// out[i*L+l] = sum_j P[i*K+j] * clv[j*L+l]
// AVX2: vectorize over l (4 patterns at once). ~4x speedup on K=20.
void LikelihoodScorer::_p_apply(const double* P,
                                 const double* clv,
                                 double* out) const {
    PROF_SCOPE("_p_apply");
    int K = K_, L = L_;
#ifdef __AVX2__
    // Strip-mined K=20 kernel: BLOCK=16 patterns × I_ROWS=2 i-rows, sized to
    // fit the accumulators + CLV loads in 16 ymm without spilling.
    if (K == 20) {
        // Prefer AVX-512 on capable CPUs (well-predicted branch; same output).
        if (g_have_avx512) {
            _p_apply_k20_avx512(P, clv, out, L);
            return;
        }
        constexpr int BLOCK = 16;
        constexpr int I_ROWS = 2;
        int Lblk = L - (L % BLOCK);
        for (int l0 = 0; l0 < Lblk; l0 += BLOCK) {
            for (int i0 = 0; i0 < 20; i0 += I_ROWS) {
                __m256d a00 = _mm256_setzero_pd(), a01 = _mm256_setzero_pd();
                __m256d a02 = _mm256_setzero_pd(), a03 = _mm256_setzero_pd();
                __m256d a10 = _mm256_setzero_pd(), a11 = _mm256_setzero_pd();
                __m256d a12 = _mm256_setzero_pd(), a13 = _mm256_setzero_pd();
                for (int j = 0; j < 20; j++) {
                    const double* clv_j = clv + j * L + l0;
                    __m256d c0 = _mm256_loadu_pd(clv_j +  0);
                    __m256d c1 = _mm256_loadu_pd(clv_j +  4);
                    __m256d c2 = _mm256_loadu_pd(clv_j +  8);
                    __m256d c3 = _mm256_loadu_pd(clv_j + 12);
                    __m256d Pi0 = _mm256_set1_pd(P[(i0 + 0) * 20 + j]);
                    __m256d Pi1 = _mm256_set1_pd(P[(i0 + 1) * 20 + j]);
                    a00 = _mm256_fmadd_pd(Pi0, c0, a00);
                    a01 = _mm256_fmadd_pd(Pi0, c1, a01);
                    a02 = _mm256_fmadd_pd(Pi0, c2, a02);
                    a03 = _mm256_fmadd_pd(Pi0, c3, a03);
                    a10 = _mm256_fmadd_pd(Pi1, c0, a10);
                    a11 = _mm256_fmadd_pd(Pi1, c1, a11);
                    a12 = _mm256_fmadd_pd(Pi1, c2, a12);
                    a13 = _mm256_fmadd_pd(Pi1, c3, a13);
                }
                double* o0 = out + (i0 + 0) * L + l0;
                double* o1 = out + (i0 + 1) * L + l0;
                _mm256_storeu_pd(o0 +  0, a00);
                _mm256_storeu_pd(o0 +  4, a01);
                _mm256_storeu_pd(o0 +  8, a02);
                _mm256_storeu_pd(o0 + 12, a03);
                _mm256_storeu_pd(o1 +  0, a10);
                _mm256_storeu_pd(o1 +  4, a11);
                _mm256_storeu_pd(o1 +  8, a12);
                _mm256_storeu_pd(o1 + 12, a13);
            }
        }
        // Tail: 4-wide AVX for remaining patterns, scalar for last <4
        int L4 = Lblk + ((L - Lblk) & ~3);
        for (int l = Lblk; l < L4; l += 4) {
            for (int i = 0; i < 20; i++) {
                __m256d a = _mm256_setzero_pd();
                const double* Pi = P + i * 20;
                for (int j = 0; j < 20; j++)
                    a = _mm256_fmadd_pd(_mm256_set1_pd(Pi[j]),
                                        _mm256_loadu_pd(clv + j * L + l), a);
                _mm256_storeu_pd(out + i * L + l, a);
            }
        }
        for (int l = L4; l < L; l++) {
            for (int i = 0; i < 20; i++) {
                double s = 0.0;
                const double* Pi = P + i * 20;
                for (int j = 0; j < 20; j++) s += Pi[j] * clv[j * L + l];
                out[i * L + l] = s;
            }
        }
        return;
    }

    // General K (e.g. K=4 for DNA): 4-wide AVX, no strip-mining.
    int L4 = L & ~3;
    for (int i = 0; i < K; i++) {
        const double* Pi = P + i * K;
        double*       Oi = out + i * L;
        for (int l = 0; l < L4; l += 4) {
            __m256d acc = _mm256_setzero_pd();
            for (int j = 0; j < K; j++)
                acc = _mm256_fmadd_pd(_mm256_set1_pd(Pi[j]),
                                      _mm256_loadu_pd(clv + j * L + l), acc);
            _mm256_storeu_pd(Oi + l, acc);
        }
        for (int l = L4; l < L; l++) {
            double s = 0.0;
            for (int j = 0; j < K; j++) s += Pi[j] * clv[j * L + l];
            Oi[l] = s;
        }
    }
#else
    for (int i = 0; i < K; i++) {
        for (int l = 0; l < L; l++) {
            double s = 0.0;
            for (int j = 0; j < K; j++)
                s += P[i * K + j] * clv[j * L + l];
            out[i * L + l] = s;
        }
    }
#endif
}

void LikelihoodScorer::_clv_mul(const double* a, const double* b, double* out) const {
    int KL = K_ * L_;
#ifdef __AVX2__
    int kl4 = KL & ~3;
    for (int i = 0; i < kl4; i += 4)
        _mm256_storeu_pd(out + i,
            _mm256_mul_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
    for (int i = kl4; i < KL; i++) out[i] = a[i] * b[i];
#else
    for (int i = 0; i < KL; i++) out[i] = a[i] * b[i];
#endif
}

// Fused: out[i,l] = (P @ clv)[i,l] * mul[i,l]. Same kernel as _p_apply but the
// store multiplies by mul first. Bit-identical to _p_apply then _clv_mul.
void LikelihoodScorer::_p_apply_mul(const double* P,
                                     const double* clv,
                                     const double* mul,
                                     double* out) const {
    int K = K_, L = L_;
#ifdef __AVX2__
    if (K == 20) {
        if (g_have_avx512) {
            _p_apply_mul_k20_avx512(P, clv, mul, out, L);
            return;
        }
        constexpr int BLOCK = 16;
        constexpr int I_ROWS = 2;
        int Lblk = L - (L % BLOCK);
        for (int l0 = 0; l0 < Lblk; l0 += BLOCK) {
            for (int i0 = 0; i0 < 20; i0 += I_ROWS) {
                __m256d a00 = _mm256_setzero_pd(), a01 = _mm256_setzero_pd();
                __m256d a02 = _mm256_setzero_pd(), a03 = _mm256_setzero_pd();
                __m256d a10 = _mm256_setzero_pd(), a11 = _mm256_setzero_pd();
                __m256d a12 = _mm256_setzero_pd(), a13 = _mm256_setzero_pd();
                for (int j = 0; j < 20; j++) {
                    const double* clv_j = clv + j * L + l0;
                    __m256d c0 = _mm256_loadu_pd(clv_j +  0);
                    __m256d c1 = _mm256_loadu_pd(clv_j +  4);
                    __m256d c2 = _mm256_loadu_pd(clv_j +  8);
                    __m256d c3 = _mm256_loadu_pd(clv_j + 12);
                    __m256d Pi0 = _mm256_set1_pd(P[(i0 + 0) * 20 + j]);
                    __m256d Pi1 = _mm256_set1_pd(P[(i0 + 1) * 20 + j]);
                    a00 = _mm256_fmadd_pd(Pi0, c0, a00);
                    a01 = _mm256_fmadd_pd(Pi0, c1, a01);
                    a02 = _mm256_fmadd_pd(Pi0, c2, a02);
                    a03 = _mm256_fmadd_pd(Pi0, c3, a03);
                    a10 = _mm256_fmadd_pd(Pi1, c0, a10);
                    a11 = _mm256_fmadd_pd(Pi1, c1, a11);
                    a12 = _mm256_fmadd_pd(Pi1, c2, a12);
                    a13 = _mm256_fmadd_pd(Pi1, c3, a13);
                }
                const double* m0 = mul + (i0 + 0) * L + l0;
                const double* m1 = mul + (i0 + 1) * L + l0;
                double* o0 = out + (i0 + 0) * L + l0;
                double* o1 = out + (i0 + 1) * L + l0;
                _mm256_storeu_pd(o0 +  0, _mm256_mul_pd(a00, _mm256_loadu_pd(m0 +  0)));
                _mm256_storeu_pd(o0 +  4, _mm256_mul_pd(a01, _mm256_loadu_pd(m0 +  4)));
                _mm256_storeu_pd(o0 +  8, _mm256_mul_pd(a02, _mm256_loadu_pd(m0 +  8)));
                _mm256_storeu_pd(o0 + 12, _mm256_mul_pd(a03, _mm256_loadu_pd(m0 + 12)));
                _mm256_storeu_pd(o1 +  0, _mm256_mul_pd(a10, _mm256_loadu_pd(m1 +  0)));
                _mm256_storeu_pd(o1 +  4, _mm256_mul_pd(a11, _mm256_loadu_pd(m1 +  4)));
                _mm256_storeu_pd(o1 +  8, _mm256_mul_pd(a12, _mm256_loadu_pd(m1 +  8)));
                _mm256_storeu_pd(o1 + 12, _mm256_mul_pd(a13, _mm256_loadu_pd(m1 + 12)));
            }
        }
        // Tail: 4-wide AVX for remaining patterns, scalar for last <4
        int L4 = Lblk + ((L - Lblk) & ~3);
        for (int l = Lblk; l < L4; l += 4) {
            for (int i = 0; i < 20; i++) {
                __m256d a = _mm256_setzero_pd();
                const double* Pi = P + i * 20;
                for (int j = 0; j < 20; j++)
                    a = _mm256_fmadd_pd(_mm256_set1_pd(Pi[j]),
                                        _mm256_loadu_pd(clv + j * L + l), a);
                _mm256_storeu_pd(out + i * L + l,
                    _mm256_mul_pd(a, _mm256_loadu_pd(mul + i * L + l)));
            }
        }
        for (int l = L4; l < L; l++) {
            for (int i = 0; i < 20; i++) {
                double s = 0.0;
                const double* Pi = P + i * 20;
                for (int j = 0; j < 20; j++) s += Pi[j] * clv[j * L + l];
                out[i * L + l] = s * mul[i * L + l];
            }
        }
        return;
    }

    // General K (e.g. K=4 for DNA): 4-wide AVX, no strip-mining.
    int L4 = L & ~3;
    for (int i = 0; i < K; i++) {
        const double* Pi = P + i * K;
        const double* Mi = mul + i * L;
        double*       Oi = out + i * L;
        for (int l = 0; l < L4; l += 4) {
            __m256d acc = _mm256_setzero_pd();
            for (int j = 0; j < K; j++)
                acc = _mm256_fmadd_pd(_mm256_set1_pd(Pi[j]),
                                      _mm256_loadu_pd(clv + j * L + l), acc);
            _mm256_storeu_pd(Oi + l, _mm256_mul_pd(acc, _mm256_loadu_pd(Mi + l)));
        }
        for (int l = L4; l < L; l++) {
            double s = 0.0;
            for (int j = 0; j < K; j++) s += Pi[j] * clv[j * L + l];
            Oi[l] = s * Mi[l];
        }
    }
#else
    for (int i = 0; i < K; i++) {
        for (int l = 0; l < L; l++) {
            double s = 0.0;
            for (int j = 0; j < K; j++)
                s += P[i * K + j] * clv[j * L + l];
            out[i * L + l] = s * mul[i * L + l];
        }
    }
#endif
}

// ================================================================== //
// Fused pair kernel: out[i, l] = (P1 @ clv1)[i, l] * (P2 @ clv2)[i, l] //
// ================================================================== //
// Both projections accumulated in registers, multiplied, stored once — no
// intermediate KL buffer. Call sites: _clv internal-internal propagation;
// _build_sumtable ((Lpi @ A) * (R @ B)).
void LikelihoodScorer::_p_apply_pair(const double* P1, const double* clv1,
                                       const double* P2, const double* clv2,
                                       double* out) const {
    PROF_SCOPE("_p_apply_pair");
    int K = K_, L = L_;

#if defined(__AVX2__) && defined(__FMA__)
    if (K == 20) {
        if (g_have_avx512) {
            _p_apply_pair_k20_avx512(P1, clv1, P2, clv2, out, L);
            return;
        }
        // AVX2 K=20: BLOCK=8 patterns × I_ROWS=2 i-rows (fits 16 ymm).
        constexpr int BLOCK = 8;
        constexpr int I_ROWS = 2;
        int Lblk = L - (L % BLOCK);
        for (int l0 = 0; l0 < Lblk; l0 += BLOCK) {
            for (int i0 = 0; i0 < 20; i0 += I_ROWS) {
                __m256d a00 = _mm256_setzero_pd(), a01 = _mm256_setzero_pd();
                __m256d a10 = _mm256_setzero_pd(), a11 = _mm256_setzero_pd();
                __m256d b00 = _mm256_setzero_pd(), b01 = _mm256_setzero_pd();
                __m256d b10 = _mm256_setzero_pd(), b11 = _mm256_setzero_pd();

                for (int j = 0; j < 20; j++) {
                    __m256d c1_0 = _mm256_loadu_pd(clv1 + j * L + l0);
                    __m256d c1_1 = _mm256_loadu_pd(clv1 + j * L + l0 + 4);
                    __m256d c2_0 = _mm256_loadu_pd(clv2 + j * L + l0);
                    __m256d c2_1 = _mm256_loadu_pd(clv2 + j * L + l0 + 4);

                    __m256d P1i0 = _mm256_set1_pd(P1[(i0 + 0) * 20 + j]);
                    __m256d P1i1 = _mm256_set1_pd(P1[(i0 + 1) * 20 + j]);
                    a00 = _mm256_fmadd_pd(P1i0, c1_0, a00);
                    a01 = _mm256_fmadd_pd(P1i0, c1_1, a01);
                    a10 = _mm256_fmadd_pd(P1i1, c1_0, a10);
                    a11 = _mm256_fmadd_pd(P1i1, c1_1, a11);

                    __m256d P2i0 = _mm256_set1_pd(P2[(i0 + 0) * 20 + j]);
                    __m256d P2i1 = _mm256_set1_pd(P2[(i0 + 1) * 20 + j]);
                    b00 = _mm256_fmadd_pd(P2i0, c2_0, b00);
                    b01 = _mm256_fmadd_pd(P2i0, c2_1, b01);
                    b10 = _mm256_fmadd_pd(P2i1, c2_0, b10);
                    b11 = _mm256_fmadd_pd(P2i1, c2_1, b11);
                }
                double* o0 = out + (i0 + 0) * L + l0;
                double* o1 = out + (i0 + 1) * L + l0;
                _mm256_storeu_pd(o0 + 0, _mm256_mul_pd(a00, b00));
                _mm256_storeu_pd(o0 + 4, _mm256_mul_pd(a01, b01));
                _mm256_storeu_pd(o1 + 0, _mm256_mul_pd(a10, b10));
                _mm256_storeu_pd(o1 + 4, _mm256_mul_pd(a11, b11));
            }
        }
        // Tail for L not divisible by 8
        for (int l = Lblk; l < L; l++) {
            for (int i = 0; i < 20; i++) {
                double s1 = 0.0, s2 = 0.0;
                const double* Pi1 = P1 + i * 20;
                const double* Pi2 = P2 + i * 20;
                for (int j = 0; j < 20; j++) {
                    s1 += Pi1[j] * clv1[j * L + l];
                    s2 += Pi2[j] * clv2[j * L + l];
                }
                out[i * L + l] = s1 * s2;
            }
        }
        return;
    }
#endif

    // Scalar fallback for general K (e.g. K=4 DNA).
    for (int i = 0; i < K; i++) {
        const double* Pi1 = P1 + i * K;
        const double* Pi2 = P2 + i * K;
        double*       Oi  = out + i * L;
        for (int l = 0; l < L; l++) {
            double s1 = 0.0, s2 = 0.0;
            for (int j = 0; j < K; j++) {
                s1 += Pi1[j] * clv1[j * L + l];
                s2 += Pi2[j] * clv2[j * L + l];
            }
            Oi[l] = s1 * s2;
        }
    }
}

// ================================================================== //
// Tip-optimized P-application                                         //
// ================================================================== //
// A leaf's CLV is sparse 0/1, so P @ tip_clv reduces to a sum over the set
// bits of the leaf's bitmask. Precompute a lookup table indexed by character
// code: out[i,l] = sum_{k in mask[code[l]]} P[i,k], state-major like _p_apply.

// AVX-512 fast path for tip-kernel fill: gather 8 doubles per step using
// _mm512_i32gather_pd. Falls through to AVX-2 4-wide gather, then scalar.
__attribute__((target("avx512f,avx512vl,avx512dq")))
static void _tip_fill_k_avx512(const double* lookup_i,
                                const uint8_t* codes,
                                double* out_i, int L) {
    int l = 0;
    for (; l + 8 <= L; l += 8) {
        __m128i u8_8 = _mm_loadl_epi64((const __m128i*)(codes + l));
        __m256i idx  = _mm256_cvtepu8_epi32(u8_8);
        __m512d v    = _mm512_i32gather_pd(idx, lookup_i, 8);
        _mm512_storeu_pd(out_i + l, v);
    }
    for (; l + 4 <= L; l += 4) {
        int32_t four_codes;
        std::memcpy(&four_codes, codes + l, 4);
        __m128i u8_4 = _mm_cvtsi32_si128(four_codes);
        __m128i idx  = _mm_cvtepu8_epi32(u8_4);
        __m256d v    = _mm256_i32gather_pd(lookup_i, idx, 8);
        _mm256_storeu_pd(out_i + l, v);
    }
    for (; l < L; l++)
        out_i[l] = lookup_i[codes[l]];
}

// Same as _tip_fill_k_avx512, fused with elementwise multiply by mul_i.
__attribute__((target("avx512f,avx512vl,avx512dq")))
static void _tip_mul_fill_k_avx512(const double* lookup_i,
                                    const uint8_t* codes,
                                    const double* mul_i,
                                    double* out_i, int L) {
    int l = 0;
    for (; l + 8 <= L; l += 8) {
        __m128i u8_8 = _mm_loadl_epi64((const __m128i*)(codes + l));
        __m256i idx  = _mm256_cvtepu8_epi32(u8_8);
        __m512d v    = _mm512_i32gather_pd(idx, lookup_i, 8);
        __m512d m    = _mm512_loadu_pd(mul_i + l);
        _mm512_storeu_pd(out_i + l, _mm512_mul_pd(v, m));
    }
    for (; l + 4 <= L; l += 4) {
        int32_t four_codes;
        std::memcpy(&four_codes, codes + l, 4);
        __m128i u8_4 = _mm_cvtsi32_si128(four_codes);
        __m128i idx  = _mm_cvtepu8_epi32(u8_4);
        __m256d v    = _mm256_i32gather_pd(lookup_i, idx, 8);
        __m256d m    = _mm256_loadu_pd(mul_i + l);
        _mm256_storeu_pd(out_i + l, _mm256_mul_pd(v, m));
    }
    for (; l < L; l++)
        out_i[l] = lookup_i[codes[l]] * mul_i[l];
}

void LikelihoodScorer::_p_apply_tip(const double* P, double t, int taxon,
                                     double* out) const {
    PROF_SCOPE("_p_apply_tip");
    const int K = K_, L = L_;
    const uint8_t* codes = &tip_codes_[taxon * L];

    // Cache by (taxon, t): the lookup is fully determined by them (same t ->
    // same P; tip_mask_table_ fixed after _build_tip_clvs).
    auto& by_t = tip_lookup_cache_[taxon];
    auto it = by_t.find(t);
    const double* lookup;
    if (it != by_t.end()) {
        COUNTER_INC("tip.cache.hit");
        lookup = it->second.data();
    } else {
        COUNTER_INC("tip.cache.miss");
        std::vector<double>& cached = by_t[t];
        cached.assign(K * n_tip_codes_, 0.0);
        for (int c = 0; c < n_tip_codes_; c++) {
            uint32_t mask = tip_mask_table_[c];
            for (int i = 0; i < K; i++) {
                double sum = 0.0;
                const double* Pi = P + i * K;
                uint32_t m = mask;
                while (m) {
                    int k = __builtin_ctz(m);
                    sum += Pi[k];
                    m &= m - 1;  // clear lowest set bit
                }
                cached[i * n_tip_codes_ + c] = sum;
            }
        }
        lookup = cached.data();
    }

    // Fill out[i,l] = lookup[i, codes[l]] (AVX-512 8-wide gather, else AVX-2).
    if (g_have_avx512) {
        for (int i = 0; i < K; i++)
            _tip_fill_k_avx512(lookup + i * n_tip_codes_, codes, out + i * L, L);
    } else {
        for (int i = 0; i < K; i++) {
            const double* lookup_i = lookup + i * n_tip_codes_;
            double* out_i = out + i * L;
            int l = 0;
#if defined(__AVX2__)
            for (; l + 4 <= L; l += 4) {
                int32_t four_codes;
                std::memcpy(&four_codes, codes + l, 4);
                __m128i u8_4 = _mm_cvtsi32_si128(four_codes);
                __m128i idx  = _mm_cvtepu8_epi32(u8_4);
                __m256d v    = _mm256_i32gather_pd(lookup_i, idx, 8);
                _mm256_storeu_pd(out_i + l, v);
            }
#endif
            for (; l < L; l++)
                out_i[l] = lookup_i[codes[l]];
        }
    }
}

// Tip-optimized fused multiply: out[i, l] = lookup[i, codes[l]] * mul[i, l].
// Same lookup-table build/cache as _p_apply_tip.
void LikelihoodScorer::_p_apply_mul_tip(const double* P, double t, int taxon,
                                          const double* mul, double* out) const {
    PROF_SCOPE("_p_apply_mul_tip");
    const int K = K_, L = L_;
    const uint8_t* codes = &tip_codes_[taxon * L];

    auto& by_t = tip_lookup_cache_[taxon];
    auto it = by_t.find(t);
    const double* lookup;
    if (it != by_t.end()) {
        COUNTER_INC("tip.cache.hit");
        lookup = it->second.data();
    } else {
        COUNTER_INC("tip.cache.miss");
        std::vector<double>& cached = by_t[t];
        cached.assign(K * n_tip_codes_, 0.0);
        for (int c = 0; c < n_tip_codes_; c++) {
            uint32_t mask = tip_mask_table_[c];
            for (int i = 0; i < K; i++) {
                double sum = 0.0;
                const double* Pi = P + i * K;
                uint32_t m = mask;
                while (m) {
                    int k = __builtin_ctz(m);
                    sum += Pi[k];
                    m &= m - 1;
                }
                cached[i * n_tip_codes_ + c] = sum;
            }
        }
        lookup = cached.data();
    }

    if (g_have_avx512) {
        for (int i = 0; i < K; i++)
            _tip_mul_fill_k_avx512(lookup + i * n_tip_codes_, codes,
                                    mul + i * L, out + i * L, L);
    } else {
        for (int i = 0; i < K; i++) {
            const double* lookup_i = lookup + i * n_tip_codes_;
            const double* mul_i    = mul + i * L;
            double*       out_i    = out + i * L;
            int l = 0;
#if defined(__AVX2__)
            for (; l + 4 <= L; l += 4) {
                int32_t four_codes;
                std::memcpy(&four_codes, codes + l, 4);
                __m128i u8_4 = _mm_cvtsi32_si128(four_codes);
                __m128i idx  = _mm_cvtepu8_epi32(u8_4);
                __m256d v    = _mm256_i32gather_pd(lookup_i, idx, 8);
                __m256d m    = _mm256_loadu_pd(mul_i + l);
                _mm256_storeu_pd(out_i + l, _mm256_mul_pd(v, m));
            }
#endif
            for (; l < L; l++)
                out_i[l] = lookup_i[codes[l]] * mul_i[l];
        }
    }
}

// Fused tip-internal: out[i,l] = (P_tip @ tip_clv[taxon])[i,l] *
// (P_int @ clv_int)[i,l], in registers (no intermediate). Reuses the
// per-(taxon, t) lookup cache.
void LikelihoodScorer::_p_apply_tip_internal_pair(
    const double* P_tip, double t_tip, int taxon_tip,
    const double* P_int, const double* clv_int,
    double* out) const
{
    PROF_SCOPE("_p_apply_tip_internal_pair");
    const int K = K_, L = L_;
    const uint8_t* codes = &tip_codes_[taxon_tip * L];

    // Look up or build the K * n_tip_codes_ lookup table for this (taxon, t).
    auto& by_t = tip_lookup_cache_[taxon_tip];
    auto it = by_t.find(t_tip);
    const double* lookup;
    if (it != by_t.end()) {
        COUNTER_INC("tip.cache.hit");
        lookup = it->second.data();
    } else {
        COUNTER_INC("tip.cache.miss");
        std::vector<double>& cached = by_t[t_tip];
        cached.assign(K * n_tip_codes_, 0.0);
        for (int c = 0; c < n_tip_codes_; c++) {
            uint32_t mask = tip_mask_table_[c];
            for (int i = 0; i < K; i++) {
                double sum = 0.0;
                const double* Pi = P_tip + i * K;
                uint32_t m = mask;
                while (m) {
                    int k = __builtin_ctz(m);
                    sum += Pi[k];
                    m &= m - 1;
                }
                cached[i * n_tip_codes_ + c] = sum;
            }
        }
        lookup = cached.data();
    }

    // K=20 AVX-512 fast path.
    if (K == 20 && g_have_avx512) {
        _p_apply_tip_int_pair_k20_avx512(lookup, n_tip_codes_, codes,
                                          P_int, clv_int, out, L);
        return;
    }

    // Scalar fallback (K=4 DNA and any other K).
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < K; i++) {
            double s = 0.0;
            const double* Pi = P_int + i * K;
            for (int j = 0; j < K; j++) s += Pi[j] * clv_int[j * L + l];
            out[i * L + l] = lookup[i * n_tip_codes_ + codes[l]] * s;
        }
    }
}

// Fused tip-tip: out[i,l] = (P1 @ tip_clv[taxon1]) * (P2 @ tip_clv[taxon2]).
// Cache safety: if taxon1==taxon2 with t1!=t2, the 2nd lookup may rehash the
// per-taxon map, but unordered_map is node-based — the mapped vector doesn't
// move, so lookup_1 stays valid.
void LikelihoodScorer::_p_apply_tip_tip_pair(
    const double* P1, double t1, int taxon1,
    const double* P2, double t2, int taxon2,
    double* out) const
{
    PROF_SCOPE("_p_apply_tip_tip_pair");
    const int K = K_, L = L_;
    const uint8_t* codes_1 = &tip_codes_[taxon1 * L];
    const uint8_t* codes_2 = &tip_codes_[taxon2 * L];

    // Local lambda: fetch or build the K * n_tip_codes_ lookup for (taxon, t).
    auto get_lookup = [&](const double* P, double t, int taxon) -> const double* {
        auto& by_t = tip_lookup_cache_[taxon];
        auto it = by_t.find(t);
        if (it != by_t.end()) {
            COUNTER_INC("tip.cache.hit");
            return it->second.data();
        }
        COUNTER_INC("tip.cache.miss");
        std::vector<double>& cached = by_t[t];
        cached.assign(K * n_tip_codes_, 0.0);
        for (int c = 0; c < n_tip_codes_; c++) {
            uint32_t mask = tip_mask_table_[c];
            for (int i = 0; i < K; i++) {
                double sum = 0.0;
                const double* Pi = P + i * K;
                uint32_t m = mask;
                while (m) {
                    int k = __builtin_ctz(m);
                    sum += Pi[k];
                    m &= m - 1;
                }
                cached[i * n_tip_codes_ + c] = sum;
            }
        }
        return cached.data();
    };

    const double* lookup_1 = get_lookup(P1, t1, taxon1);
    const double* lookup_2 = get_lookup(P2, t2, taxon2);

    // K=20 AVX-512 fast path.
    if (K == 20 && g_have_avx512) {
        _p_apply_tip_tip_pair_k20_avx512(lookup_1, n_tip_codes_, codes_1,
                                          lookup_2, n_tip_codes_, codes_2,
                                          out, L);
        return;
    }

    // Scalar fallback (K=4 DNA and any future K).
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < K; i++) {
            out[i * L + l] = lookup_1[i * n_tip_codes_ + codes_1[l]]
                           * lookup_2[i * n_tip_codes_ + codes_2[l]];
        }
    }
}

// Tip-CLV lookup kernel is always enabled.
static bool _tip_kernel_enabled() { return true; }

// Sumtable BLO is always enabled (call sites still fall back automatically
// when the model has no eigendecomposition; see model_->eigenvalues() guards).
static bool _sumtable_enabled() { return true; }

void LikelihoodScorer::_p_for_edge(int u, int v, double* out) const {
    // p_arr_ holds all 3 slots per node, so any live tree edge is a cache hit;
    // the fallback handles transient non-edge (u,v) pairs (e.g. scratch scoring).
    int s = _slot_of(u, v);
    if (s >= 0) {
        std::memcpy(out, &p_arr_[(u * 3 + s) * K_ * K_],
                    K_ * K_ * sizeof(double));
        return;
    }
    model_->p_matrix(bl_[u * n_ + v], out);
}

// Thread-local scratch pool for _clv recursion: one indexed block per frame,
// grown on demand as recursion deepens.
namespace {
struct ClvScratch {
    std::vector<std::vector<double>> r1, r2, ls1, ls2, P1, P2, a1, a2;
    void ensure(int level, int KL, int L, int KK) {
        while ((int)r1.size() <= level) {
            r1.emplace_back(KL); r2.emplace_back(KL);
            ls1.emplace_back(L); ls2.emplace_back(L);
            P1.emplace_back(KK); P2.emplace_back(KK);
            a1.emplace_back(KL); a2.emplace_back(KL);
        }
        // Grow existing blocks if dimensions increased between scorer instances.
        if ((int)r1[level].size() < KL) {
            r1[level].assign(KL, 0.0); r2[level].assign(KL, 0.0);
            a1[level].assign(KL, 0.0); a2[level].assign(KL, 0.0);
        }
        if ((int)ls1[level].size() < L) {
            ls1[level].assign(L, 0.0); ls2[level].assign(L, 0.0);
        }
        if ((int)P1[level].size() < KK) {
            P1[level].assign(KK, 0.0); P2[level].assign(KK, 0.0);
        }
    }
};
thread_local ClvScratch g_clv_scratch;
thread_local int        g_clv_depth = 0;
}  // anon namespace