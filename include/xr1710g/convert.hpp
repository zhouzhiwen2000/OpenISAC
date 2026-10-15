#ifndef OPENISAC_XR1710G_CONVERT_HPP
#define OPENISAC_XR1710G_CONVERT_HPP

#include <complex>
#include <cstddef>
#include <cstdint>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace radio::xr1710g {
constexpr float kSampleScale = 1.0f / 8192.0f;

// The standard guarantees interleaved real/imaginary storage for complex<float>.
// Unaligned loads/stores support arbitrary recv() fragments and caller buffers.
inline void normalize(const int16_t* in, std::complex<float>* out, size_t n) {
    size_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 scale = _mm256_set1_ps(kSampleScale);
    for (; i + 4 <= n; i += 4) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 2*i));
        const __m256 x = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(raw));
        _mm256_storeu_ps(reinterpret_cast<float*>(out + i), _mm256_mul_ps(x, scale));
    }
#endif
    for (; i < n; ++i) out[i] = {in[2*i] * kSampleScale, in[2*i+1] * kSampleScale};
}

inline void rotate_portable(const int16_t* in, const std::complex<float>* oscillator,
                            std::complex<float>* out, size_t n, float phase_i, float phase_q) {
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const float oi = oscillator[i].real() * phase_i - oscillator[i].imag() * phase_q;
        const float oq = oscillator[i].real() * phase_q + oscillator[i].imag() * phase_i;
        out[i] = {(in[2*i] * oi - in[2*i+1] * oq) * kSampleScale,
                  (in[2*i] * oq + in[2*i+1] * oi) * kSampleScale};
    }
}

inline void rotate(const int16_t* in, const std::complex<float>* oscillator,
                   std::complex<float>* out, size_t n, float phase_i, float phase_q) {
    size_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    // Fold SC16 normalization into the block phase. Four IQ samples per vector;
    // fmaddsub subtracts the real cross term and adds the imaginary cross term.
    const __m256 pi = _mm256_set1_ps(phase_i * kSampleScale);
    const __m256 pq = _mm256_set1_ps(phase_q * kSampleScale);
    for (; i + 4 <= n; i += 4) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 2*i));
        const __m256 x = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(raw));
        const __m256 wave = _mm256_loadu_ps(reinterpret_cast<const float*>(oscillator + i));
        const __m256 osc = _mm256_fmaddsub_ps(wave, pi,
            _mm256_mul_ps(_mm256_permute_ps(wave, 0xb1), pq));
        const __m256 y = _mm256_fmaddsub_ps(_mm256_moveldup_ps(x), osc,
            _mm256_mul_ps(_mm256_movehdup_ps(x), _mm256_permute_ps(osc, 0xb1)));
        _mm256_storeu_ps(reinterpret_cast<float*>(out + i), y);
    }
#endif
    rotate_portable(in + 2*i, oscillator + i, out + i, n - i, phase_i, phase_q);
}
} // namespace radio::xr1710g

#endif
