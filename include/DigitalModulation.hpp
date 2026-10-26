#ifndef DIGITAL_MODULATION_HPP
#define DIGITAL_MODULATION_HPP

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

enum class ModulationScheme { Bpsk, Qpsk, Psk8, Qam16, Qam32, Qam64 };

/** @brief Parse a payload constellation name; an unknown name is an error. */
inline ModulationScheme parse_modulation_scheme(std::string name) {
    for (char& c : name) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (name == "BPSK") return ModulationScheme::Bpsk;
    if (name == "QPSK") return ModulationScheme::Qpsk;
    if (name == "8PSK") return ModulationScheme::Psk8;
    if (name == "16QAM") return ModulationScheme::Qam16;
    if (name == "32QAM") return ModulationScheme::Qam32;
    if (name == "64QAM") return ModulationScheme::Qam64;
    throw std::invalid_argument("modulation.scheme must be BPSK, QPSK, 8PSK, 16QAM, 32QAM or 64QAM");
}

/** @brief Unit-average-energy Gray constellations and max-log soft demapping.
 *
 * Labels are transmitted most-significant bit first. QAM alternates I and Q
 * label bits, beginning with I; 32QAM uses an 8-by-4 rectangular grid. QPSK
 * keeps the existing OpenISAC mapping exactly. Control symbols always use
 * QPSK, independently of the selected payload constellation.
 */
class DigitalModulation {
public:
    static constexpr float kInvSqrt2 = 0.7071067811865476f;

    explicit DigitalModulation(ModulationScheme scheme = ModulationScheme::Qpsk)
        : _scheme(scheme) {
        switch (scheme) {
        case ModulationScheme::Bpsk: _bits = 1; break;
        case ModulationScheme::Qpsk: _bits = 2; break;
        case ModulationScheme::Psk8: _bits = 3; break;
        case ModulationScheme::Qam16: _bits = 4; break;
        case ModulationScheme::Qam32: _bits = 5; break;
        case ModulationScheme::Qam64: _bits = 6; break;
        }
        _order = 1u << _bits;
        const unsigned ni = 1u << ((_bits + 1) / 2), nq = 1u << (_bits / 2);
        const float norm = 1.0f / std::sqrt(static_cast<float>(ni * ni + nq * nq - 2) / 3.0f);
        _points[0] = { kInvSqrt2,  kInvSqrt2};
        _points[1] = { kInvSqrt2, -kInvSqrt2};
        _points[2] = {-kInvSqrt2,  kInvSqrt2};
        _points[3] = {-kInvSqrt2, -kInvSqrt2};
        for (unsigned label = 0; label < _order; ++label) {
            if (scheme == ModulationScheme::Psk8) {
                constexpr double pi = 3.14159265358979323846;
                const double angle = gray_to_binary(label) * pi / 4.0;
                _points[4 + label] = {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle))};
            } else {
                unsigned gi = 0, gq = 0;
                for (unsigned bit = 0; bit < _bits; ++bit) {
                    const unsigned value = (label >> (_bits - bit - 1)) & 1u;
                    if ((bit & 1u) == 0) gi = (gi << 1) | value;
                    else gq = (gq << 1) | value;
                }
                _points[4 + label] = {pam(gi, ni) * norm, pam(gq, nq) * norm};
            }
        }
    }

    unsigned bits_per_symbol() const { return _bits; }
    unsigned order() const { return _order; }
    bool is_qpsk() const { return _scheme == ModulationScheme::Qpsk; }
    ModulationScheme scheme() const { return _scheme; }

    /** @brief Number of payload RE, including zero bits in the final partial RE. */
    size_t symbol_count(size_t bit_count) const {
        return bit_count / _bits + (bit_count % _bits != 0);
    }

    /** @brief Map an unsigned MSB-first bit label to a normalized constellation. */
    std::complex<float> modulate(unsigned label) const { return _points[4 + (label & (_order - 1))]; }

    /** @brief Look up a valid TX token from pack_bits() or the QPSK control writer. */
    std::complex<float> modulate_token(int token) const {
        return _points[token];
    }

    /** @brief Pack coded bits into payload tokens, optionally padding each LDPC block. */
    template<typename Bits, typename Symbols>
    void pack_bits(const Bits& bits, Symbols& symbols, size_t block_bits = 0) const {
        if (block_bits == 0) block_bits = std::max<size_t>(bits.size(), 1);
        symbols.resize((bits.size() / block_bits) * symbol_count(block_bits) +
                       symbol_count(bits.size() % block_bits));
        switch (_scheme) {
        case ModulationScheme::Bpsk: pack_blocks<1>(bits, symbols, block_bits); break;
        case ModulationScheme::Qpsk: pack_blocks<2>(bits, symbols, block_bits); break;
        case ModulationScheme::Psk8: pack_blocks<3>(bits, symbols, block_bits); break;
        case ModulationScheme::Qam16: pack_blocks<4>(bits, symbols, block_bits); break;
        case ModulationScheme::Qam32: pack_blocks<5>(bits, symbols, block_bits); break;
        case ModulationScheme::Qam64: pack_blocks<6>(bits, symbols, block_bits); break;
        }
    }

    /** @brief Nearest payload point, using angular/PAM decision thresholds. */
    std::complex<float> remodulate(std::complex<float> sample) const {
        switch (_scheme) {
        case ModulationScheme::Bpsk: return hard_point<1>(sample);
        case ModulationScheme::Qpsk: return hard_point<2>(sample);
        case ModulationScheme::Psk8: return hard_point<3>(sample);
        case ModulationScheme::Qam16: return hard_point<4>(sample);
        case ModulationScheme::Qam32: return hard_point<5>(sample);
        case ModulationScheme::Qam64: return hard_point<6>(sample);
        }
        return {};
    }

    /** @brief Hard-slice into a preallocated buffer that does not overlap the input. */
    void remodulate_samples(const std::complex<float>* input, std::complex<float>* output,
                            size_t count) const {
        switch (_scheme) {
        case ModulationScheme::Bpsk: remodulate_range<1>(input, output, count); break;
        case ModulationScheme::Qpsk: remodulate_range<2>(input, output, count); break;
        case ModulationScheme::Psk8: remodulate_range<3>(input, output, count); break;
        case ModulationScheme::Qam16: remodulate_range<4>(input, output, count); break;
        case ModulationScheme::Qam32: remodulate_range<5>(input, output, count); break;
        case ModulationScheme::Qam64: remodulate_range<6>(input, output, count); break;
        }
    }

    /** @brief Sum decision error over indexed RE, dispatching once per symbol.
     * @param mask 0 = QPSK control, 1 = payload, 2 = unknown; ignored for QPSK.
     */
    double decision_error_sum(const std::complex<float>* samples, const int* indices,
                              const uint8_t* mask, size_t count) const {
        switch (_scheme) {
        case ModulationScheme::Bpsk: return error_sum<1>(samples, indices, mask, count);
        case ModulationScheme::Qpsk: return error_sum<2>(samples, indices, mask, count);
        case ModulationScheme::Psk8: return error_sum<3>(samples, indices, mask, count);
        case ModulationScheme::Qam16: return error_sum<4>(samples, indices, mask, count);
        case ModulationScheme::Qam32: return error_sum<5>(samples, indices, mask, count);
        case ModulationScheme::Qam64: return error_sum<6>(samples, indices, mask, count);
        }
        return 0.0;
    }

    /** @brief Nearest point in the control/payload union for unknown frame regions. */
    std::complex<float> remodulate_mixed(std::complex<float> sample) const {
        return remodulate_mixed(sample, remodulate(sample));
    }

    /** @brief Resolve an unknown RE while reusing its already sliced payload point. */
    static std::complex<float> remodulate_mixed(std::complex<float> sample,
                                               std::complex<float> payload) {
        const std::complex<float> control{std::copysign(kInvSqrt2, sample.real()),
                                          std::copysign(kInvSqrt2, sample.imag())};
        return std::norm(sample - control) < std::norm(sample - payload) ? control : payload;
    }

    /** @brief Compute max-log LLRs; positive means zero, inv_noise = 1/E[|n|^2]. */
    void soft_llrs(std::complex<float> sample, float inv_noise, float* out) const {
        switch (_scheme) {
        case ModulationScheme::Bpsk: out[0] = 4.0f * sample.real() * inv_noise; return;
        case ModulationScheme::Qpsk:
            out[0] = sample.real() * (4.0f * kInvSqrt2 * inv_noise);
            out[1] = sample.imag() * (4.0f * kInvSqrt2 * inv_noise);
            return;
        case ModulationScheme::Psk8:
            scaled_llrs<3>(sample.real(), sample.imag(), 1.0f, out); break;
        case ModulationScheme::Qam16:
            scaled_llrs<4>(sample.real(), sample.imag(), normalization<4>(), out); break;
        case ModulationScheme::Qam32:
            scaled_llrs<5>(sample.real(), sample.imag(), normalization<5>(), out); break;
        case ModulationScheme::Qam64:
            scaled_llrs<6>(sample.real(), sample.imag(), normalization<6>(), out); break;
        }
        const float factor = 4.0f * kInvSqrt2 * inv_noise;
        for (unsigned bit = 0; bit < _bits; ++bit) out[bit] *= factor;
    }

    /** @brief Demap payload from scaled I/Q pairs, retaining the legacy QPSK LLR transport.
     * The multiplier includes Q16 scaling where applicable. Work is performed
     * only on packets whose QPSK mini-header has been recovered successfully.
     */
    template<typename Llr>
    void demap_scaled_pairs(const Llr* pairs, size_t bit_count, float iq_scale, Llr* output,
                            size_t block_bits = 0) const {
        switch (_scheme) {
        case ModulationScheme::Bpsk: demap_blocks<1>(pairs, bit_count, iq_scale, output, block_bits); break;
        case ModulationScheme::Qpsk: std::copy_n(pairs, bit_count, output); break;
        case ModulationScheme::Psk8: demap_blocks<3>(pairs, bit_count, iq_scale, output, block_bits); break;
        case ModulationScheme::Qam16: demap_blocks<4>(pairs, bit_count, iq_scale, output, block_bits); break;
        case ModulationScheme::Qam32: demap_blocks<5>(pairs, bit_count, iq_scale, output, block_bits); break;
        case ModulationScheme::Qam64: demap_blocks<6>(pairs, bit_count, iq_scale, output, block_bits); break;
        }
    }

private:
    template<unsigned Bits>
    static constexpr float normalization() {
        if constexpr (Bits == 4) return 0.31622776601683794f;
        if constexpr (Bits == 5) return 0.19611613513818404f;
        if constexpr (Bits == 6) return 0.1543033499620919f;
        return 1.0f;
    }

    template<bool Eight>
    [[gnu::always_inline]] static inline float slice_magnitude(float a, float norm) {
        float level = 1.0f + 2.0f * static_cast<float>(a >= 2.0f * norm);
        if constexpr (Eight)
            level += 2.0f * static_cast<float>(a >= 4.0f * norm) +
                     2.0f * static_cast<float>(a >= 6.0f * norm);
        return level * norm;
    }

    template<unsigned Bits>
    [[gnu::always_inline]] static inline std::complex<float> hard_point(std::complex<float> sample) {
        const float x = sample.real(), y = sample.imag();
        if constexpr (Bits == 1) {
            return {std::copysign(1.0f, x), 0.0f};
        } else if constexpr (Bits == 2) {
            return {std::copysign(kInvSqrt2, x), std::copysign(kInvSqrt2, y)};
        } else if constexpr (Bits == 3) {
            constexpr float tan_pi_8 = 0.41421356237309505f;
            const float ax = std::abs(x), ay = std::abs(y);
            const bool horizontal = ay <= tan_pi_8 * ax;
            const bool vertical = ax < tan_pi_8 * ay;
            return {std::copysign(horizontal ? 1.0f : (vertical ? 0.0f : kInvSqrt2), x),
                    std::copysign(horizontal ? 0.0f : (vertical ? 1.0f : kInvSqrt2), y)};
        } else {
            constexpr float norm = normalization<Bits>();
            return {std::copysign(slice_magnitude<(Bits >= 5)>(std::abs(x), norm), x),
                    std::copysign(slice_magnitude<(Bits == 6)>(std::abs(y), norm), y)};
        }
    }

    template<unsigned Bits>
    [[gnu::always_inline]] static inline float hard_error(float x, float y) {
        const float ax = std::abs(x), ay = std::abs(y);
        float di, dq;
        if constexpr (Bits == 1) {
            di = ax - 1.0f;
            dq = ay;
        } else if constexpr (Bits == 3) {
            constexpr float tan_pi_8 = 0.41421356237309505f;
            const bool horizontal = ay <= tan_pi_8 * ax;
            const bool vertical = ax < tan_pi_8 * ay;
            di = ax - (horizontal ? 1.0f : (vertical ? 0.0f : kInvSqrt2));
            dq = ay - (horizontal ? 0.0f : (vertical ? 1.0f : kInvSqrt2));
        } else {
            constexpr float norm = normalization<Bits>();
            di = ax - slice_magnitude<(Bits >= 5)>(ax, norm);
            dq = ay - slice_magnitude<(Bits == 6)>(ay, norm);
        }
        return di * di + dq * dq;
    }

    template<unsigned Bits>
    static void remodulate_range(const std::complex<float>* __restrict__ input,
                                 std::complex<float>* __restrict__ output, size_t count) {
        #pragma omp simd
        for (size_t i = 0; i < count; ++i) output[i] = hard_point<Bits>(input[i]);
    }

    template<unsigned Bits>
    static double error_sum(const std::complex<float>* __restrict__ samples,
                            const int* __restrict__ indices, const uint8_t* __restrict__ mask,
                            size_t count) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < count; ++i) {
            const float x = samples[indices[i]].real(), y = samples[indices[i]].imag();
            const float ci = std::abs(x) - kInvSqrt2;
            const float cq = std::abs(y) - kInvSqrt2;
            const float control_error = ci * ci + cq * cq;
            if constexpr (Bits == 2) {
                sum += static_cast<double>(control_error);
            } else {
                const float payload_error = hard_error<Bits>(x, y);
                sum += static_cast<double>(mask[i] == 0 ? control_error :
                    (mask[i] == 1 ? payload_error : std::min(control_error, payload_error)));
            }
        }
        return sum;
    }

    // Exact Gray-PAM max-log differences. unit is one constellation step in
    // the scaled I/Q domain. The common squared-distance term cancels out.
    [[gnu::always_inline]] static inline void pam4_llrs(
        float x, float unit, float gain, float& sign, float& inner) {
        const float a = std::abs(x), edge = a - 2.0f * unit;
        sign = gain * std::copysign(a + std::max(edge, 0.0f), x);
        inner = gain * edge;
    }

    [[gnu::always_inline]] static inline void pam8_llrs(
        float x, float unit, float gain, float& sign, float& middle, float& inner) {
        const float a = std::abs(x);
        const float e2 = a - 2.0f * unit, e4 = a - 4.0f * unit, e6 = a - 6.0f * unit;
        const float r6 = std::max(e6, 0.0f);
        sign = gain * std::copysign(a + std::max(e2, 0.0f) + std::max(e4, 0.0f) + r6, x);
        middle = gain * (e4 + r6 - std::max(-e2, 0.0f));
        inner = gain * (std::abs(e4) - 2.0f * unit);
    }

    template<unsigned Bits>
    [[gnu::always_inline]] static inline void scaled_llrs(float x, float y, float unit, float& b0, float& b1, float& b2, float& b3, float& b4, float& b5) {
        if constexpr (Bits == 1) {
            b0 = x / kInvSqrt2;
        } else if constexpr (Bits == 3) {
            // Equal-energy 8PSK: min squared distance becomes max projection.
            const float ax = std::abs(x), ay = std::abs(y);
            const float z0 = std::max(std::max(x, y), kInvSqrt2 * (y + ax));
            const float o0 = std::max(std::max(-x, -y), kInvSqrt2 * (ax - y));
            const float z1 = std::max(std::max(x, -y), kInvSqrt2 * (x + ay));
            const float o1 = std::max(std::max(-x, y), kInvSqrt2 * (ay - x));
            b0 = kInvSqrt2 * (z0 - o0);
            b1 = kInvSqrt2 * (z1 - o1);
            b2 = kInvSqrt2 * (std::max(ax, kInvSqrt2 * std::abs(x - y)) -
                                  std::max(ay, kInvSqrt2 * std::abs(x + y)));
        } else {
            constexpr float gain = normalization<Bits>() / kInvSqrt2;
            if constexpr (Bits == 4) {
                pam4_llrs(x, unit, gain, b0, b2);
                pam4_llrs(y, unit, gain, b1, b3);
            } else if constexpr (Bits == 5) {
                pam8_llrs(x, unit, gain, b0, b2, b4);
                pam4_llrs(y, unit, gain, b1, b3);
            } else {
                pam8_llrs(x, unit, gain, b0, b2, b4);
                pam8_llrs(y, unit, gain, b1, b3, b5);
            }
        }
    }

    template<unsigned Bits>
    [[gnu::always_inline]] static inline void scaled_llrs(float x, float y, float unit, float* out) {
        float b0, b1, b2, b3, b4, b5;
        scaled_llrs<Bits>(x, y, unit, b0, b1, b2, b3, b4, b5);
        out[0] = b0;
        if constexpr (Bits >= 3) { out[1] = b1; out[2] = b2; }
        if constexpr (Bits >= 4) out[3] = b3;
        if constexpr (Bits >= 5) out[4] = b4;
        if constexpr (Bits >= 6) out[5] = b5;
    }

    template<typename Llr>
    [[gnu::always_inline]] static inline Llr quantize(float value) {
        // Integer conversion already truncates: a signed half-step implements
        // rounding without a separate round-to-float instruction in the loop.
        if constexpr (std::is_integral_v<Llr>)
            return static_cast<Llr>(std::clamp(value, -32767.0f, 32767.0f) + std::copysign(0.5f, value));
        else return value;
    }

    template<unsigned Bits, typename Llr>
    static void demap_range(const Llr* pairs, size_t bit_count, float unit, Llr* output) {
        const size_t symbols = bit_count / Bits;
        #pragma omp simd
        for (size_t i = 0; i < symbols; ++i) {
            float b0, b1, b2, b3, b4, b5;
            scaled_llrs<Bits>(static_cast<float>(pairs[2 * i]), static_cast<float>(pairs[2 * i + 1]),
                              unit, b0, b1, b2, b3, b4, b5);
            output[i * Bits] = quantize<Llr>(b0);
            if constexpr (Bits >= 3) {
                output[i * Bits + 1] = quantize<Llr>(b1);
                output[i * Bits + 2] = quantize<Llr>(b2);
            }
            if constexpr (Bits >= 4) output[i * Bits + 3] = quantize<Llr>(b3);
            if constexpr (Bits >= 5) output[i * Bits + 4] = quantize<Llr>(b4);
            if constexpr (Bits >= 6) output[i * Bits + 5] = quantize<Llr>(b5);
        }
        // Only the final, partially occupied RE needs a bounds check (32QAM
        // has three useful bits in the last RE of each 1008-bit codeword).
        if (bit_count % Bits != 0) {
            float values[Bits];
            scaled_llrs<Bits>(static_cast<float>(pairs[2 * symbols]), static_cast<float>(pairs[2 * symbols + 1]), unit, values);
            for (size_t bit = 0; bit < bit_count % Bits; ++bit)
                output[symbols * Bits + bit] = quantize<Llr>(values[bit]);
        }
    }

    template<unsigned Bits, typename Llr>
    static void demap_blocks(const Llr* pairs, size_t bit_count, float iq_scale, Llr* output, size_t block_bits) {
        const float unit = normalization<Bits>() * iq_scale;
        if (block_bits == 0 || block_bits % Bits == 0) {
            demap_range<Bits>(pairs, bit_count, unit, output);
            return;
        }
        const size_t stride = 2 * ((block_bits + Bits - 1) / Bits);
        while (bit_count >= block_bits) {
            demap_range<Bits>(pairs, block_bits, unit, output);
            pairs += stride;
            output += block_bits;
            bit_count -= block_bits;
        }
        demap_range<Bits>(pairs, bit_count, unit, output);
    }

    template<unsigned Bits, typename Bit, typename Token>
    static void pack_range(const Bit* bits, size_t bit_count, Token* output) {
        constexpr unsigned base = Bits == 2 ? 0 : 4;
        const size_t symbols = bit_count / Bits;
        size_t first = 0;
        if constexpr (Bits == 5) {
            // Four 5-bit labels span five nibbles. Contract the contiguous
            // input first, so the expensive bit loads use a regular SIMD loop.
            constexpr size_t kBatchSymbols = 64;
            constexpr size_t kNibbles = kBatchSymbols * 5 / 4;
            alignas(64) unsigned nibbles[kNibbles];
            for (; first + kBatchSymbols <= symbols; first += kBatchSymbols) {
                const Bit* input = bits + first * 5;
                #pragma omp simd
                for (size_t j = 0; j < kNibbles; ++j)
                    nibbles[j] = ((static_cast<unsigned>(input[4 * j]) & 1u) << 3) |
                                 ((static_cast<unsigned>(input[4 * j + 1]) & 1u) << 2) |
                                 ((static_cast<unsigned>(input[4 * j + 2]) & 1u) << 1) |
                                  (static_cast<unsigned>(input[4 * j + 3]) & 1u);
                #pragma omp simd
                for (size_t j = 0; j < kBatchSymbols / 4; ++j) {
                    output[first + 4 * j] = 4 + ((nibbles[5 * j] << 1) | (nibbles[5 * j + 1] >> 3));
                    output[first + 4 * j + 1] = 4 + (((nibbles[5 * j + 1] & 7u) << 2) | (nibbles[5 * j + 2] >> 2));
                    output[first + 4 * j + 2] = 4 + (((nibbles[5 * j + 2] & 3u) << 3) | (nibbles[5 * j + 3] >> 1));
                    output[first + 4 * j + 3] = 4 + (((nibbles[5 * j + 3] & 1u) << 4) | nibbles[5 * j + 4]);
                }
            }
        } else if constexpr (Bits == 6) {
            // Two consecutive 3-bit groups form one 6-bit label. Both loops
            // have regular strides that the compiler can vectorize directly.
            constexpr size_t kBatchSymbols = 32;
            alignas(64) unsigned triples[kBatchSymbols * 2];
            for (; first + kBatchSymbols <= symbols; first += kBatchSymbols) {
                const Bit* input = bits + first * 6;
                #pragma omp simd
                for (size_t j = 0; j < kBatchSymbols * 2; ++j)
                    triples[j] = ((static_cast<unsigned>(input[3 * j]) & 1u) << 2) |
                                 ((static_cast<unsigned>(input[3 * j + 1]) & 1u) << 1) |
                                  (static_cast<unsigned>(input[3 * j + 2]) & 1u);
                #pragma omp simd
                for (size_t j = 0; j < kBatchSymbols; ++j)
                    output[first + j] = 4 + ((triples[2 * j] << 3) | triples[2 * j + 1]);
            }
        }
        #pragma omp simd
        for (size_t i = first; i < symbols; ++i) {
            unsigned label = static_cast<unsigned>(bits[i * Bits]) & 1u;
            if constexpr (Bits >= 2) label = (label << 1) | (static_cast<unsigned>(bits[i * Bits + 1]) & 1u);
            if constexpr (Bits >= 3) label = (label << 1) | (static_cast<unsigned>(bits[i * Bits + 2]) & 1u);
            if constexpr (Bits >= 4) label = (label << 1) | (static_cast<unsigned>(bits[i * Bits + 3]) & 1u);
            if constexpr (Bits >= 5) label = (label << 1) | (static_cast<unsigned>(bits[i * Bits + 4]) & 1u);
            if constexpr (Bits >= 6) label = (label << 1) | (static_cast<unsigned>(bits[i * Bits + 5]) & 1u);
            output[i] = static_cast<Token>(base + label);
        }
        if (bit_count % Bits != 0) {
            unsigned label = 0;
            for (unsigned bit = 0; bit < Bits; ++bit)
                label = (label << 1) | (bit < bit_count % Bits ?
                    (static_cast<unsigned>(bits[symbols * Bits + bit]) & 1u) : 0u);
            output[symbols] = static_cast<Token>(base + label);
        }
    }

    template<unsigned Bits, typename BitVector, typename TokenVector>
    static void pack_blocks(const BitVector& bits, TokenVector& symbols, size_t block_bits) {
        if (block_bits % Bits == 0) {
            pack_range<Bits>(bits.data(), bits.size(), symbols.data());
            return;
        }
        const size_t stride = (block_bits + Bits - 1) / Bits;
        size_t at = 0, symbol = 0;
        for (; bits.size() - at >= block_bits; at += block_bits, symbol += stride)
            pack_range<Bits>(bits.data() + at, block_bits, symbols.data() + symbol);
        if (at != bits.size())
            pack_range<Bits>(bits.data() + at, bits.size() - at, symbols.data() + symbol);
    }

    static unsigned gray_to_binary(unsigned gray) {
        unsigned value = gray;
        while ((gray >>= 1) != 0) value ^= gray;
        return value;
    }
    static float pam(unsigned gray, unsigned levels) {
        return static_cast<float>(static_cast<int>(levels) - 1 - 2 * static_cast<int>(gray_to_binary(gray)));
    }
    ModulationScheme _scheme;
    unsigned _bits = 2, _order = 4;
    std::array<std::complex<float>, 68> _points{};
};

#endif // DIGITAL_MODULATION_HPP
