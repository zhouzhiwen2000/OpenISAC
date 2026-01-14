#ifndef OFDM_SIGNAL_PROCESSING_HPP
#define OFDM_SIGNAL_PROCESSING_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <array>
#include <Common.hpp>

namespace OpenISAC {
namespace DSP {

    /**
     * @brief Generate Zadoff-Chu Sequence (Frequency Domain)
     * 
     * @param N Sequence length (FFT size)
     * @param root Root index (q)
     * @return AlignedVector vector containing the ZC sequence
     */
    inline AlignedVector generate_zc_sequence(int N, int root) {
        AlignedVector zc_seq(N);
        const int q = root;
        const int delta = (N & 1); // 0 for even N, 1 for odd N
        const double base = -M_PI * static_cast<double>(q) / static_cast<double>(N);

        #pragma omp simd
        for (int n = 0; n < N; ++n) {
            const double nd = static_cast<double>(n);
            const double arg = nd * (nd + static_cast<double>(delta));
            const double phase = base * arg;
            zc_seq[n] = std::polar(1.0f, static_cast<float>(phase));
        }
        return zc_seq;
    }

    /**
     * @brief Hamming Window Generator
     * 
     * @param size Window size
     * @return AlignedFloatVector containing Hamming window values
     */
    inline AlignedFloatVector generate_hamming_window(size_t size) {
        AlignedFloatVector window(size);
        for (size_t i = 0; i < size; ++i) {
            window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (size - 1));
        }
        return window;
    }

    /**
     * @brief Hanning Window Generator
     * 
     * @param size Window size
     * @return AlignedFloatVector containing Hanning window values
     */
    inline AlignedFloatVector generate_hanning_window(size_t size) {
        AlignedFloatVector window(size);
        for (size_t i = 0; i < size; ++i) {
            window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
        }
        return window;
    }

    /**
     * @brief QPSK Mapper
     * 
     * Efficient table-based QPSK mapping.
     */
    class QPSKMapper {
    public:
        QPSKMapper() = default;

        static constexpr float SQRT_2_INV = 0.7071067811865476f; // 1/sqrt(2)

        static inline std::complex<float> map(int x) {
            static const std::array<std::complex<float>, 4> table = {{
                { SQRT_2_INV,  SQRT_2_INV},  // 00 -> 0
                { SQRT_2_INV, -SQRT_2_INV},  // 01 -> 1
                {-SQRT_2_INV,  SQRT_2_INV},  // 10 -> 2
                {-SQRT_2_INV, -SQRT_2_INV}   // 11 -> 3
            }};
            return table[x & 3];
        }
    };

} // namespace DSP
} // namespace OpenISAC

#endif // OFDM_SIGNAL_PROCESSING_HPP
