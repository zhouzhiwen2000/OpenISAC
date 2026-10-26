#ifndef LDPC_RATE_MATCHING_HPP
#define LDPC_RATE_MATCHING_HPP

#include <cstddef>
#include <stdexcept>
#include <string>

/** @brief Payload rates for the LDPC(1008,504) mother code. */
enum class LdpcCodeRate { R12, R23, R34 };

/** @brief Parse the exact YAML code-rate notation. */
inline LdpcCodeRate parse_ldpc_code_rate(const std::string& value) {
    if (value == "1/2") return LdpcCodeRate::R12;
    if (value == "2/3") return LdpcCodeRate::R23;
    if (value == "3/4") return LdpcCodeRate::R34;
    throw std::invalid_argument("ldpc.code_rate must be 1/2, 2/3, or 3/4");
}

/** @brief Rate matching after the existing scrambler and 21-row interleaver.
 * Each 504-bit information block emits 1008, 756, or 672 bits.
 * Punctured positions have zero LLR before deinterleaving and descrambling.
 * The 1/2 path performs no copies or resizing.
 */
class LdpcRateMatching {
public:
    explicit constexpr LdpcRateMatching(LdpcCodeRate rate = LdpcCodeRate::R12)
        : _bits(rate == LdpcCodeRate::R23 ? 756 :
                rate == LdpcCodeRate::R34 ? 672 : 1008) {}

    /** @brief Transmitted bits per mother-code block, excluding RE padding. */
    constexpr size_t code_bits() const { return _bits; }

    /** @brief Select transmitted bits using a reusable, distinct scratch vector. */
    template<typename Vector>
    void match_inplace(Vector& bits, Vector& scratch) const {
        if (_bits == 1008) return;
        const size_t blocks = bits.size() / 1008;
        scratch.resize(blocks * _bits);
        for (size_t block = 0; block < blocks; ++block) {
            const auto* input = bits.data() + block * 1008;
            auto* output = scratch.data() + block * _bits;
            if (_bits == 756) {
                #pragma omp simd
                for (size_t i = 0; i < 756; ++i) output[i] = input[(i / 3) * 4 + i % 3];
            } else {
                #pragma omp simd
                for (size_t i = 0; i < 672; ++i) output[i] = input[(i / 2) * 3 + i % 2];
            }
        }
        bits.swap(scratch);
    }

    /** @brief Restore mother-code LLRs with zeros at punctured positions. */
    template<typename Vector>
    void recover_inplace(Vector& llrs, Vector& scratch) const {
        if (_bits == 1008) return;
        using Value = typename Vector::value_type;
        const size_t blocks = llrs.size() / _bits;
        scratch.resize(blocks * 1008);
        for (size_t block = 0; block < blocks; ++block) {
            const auto* input = llrs.data() + block * _bits;
            auto* output = scratch.data() + block * 1008;
            if (_bits == 756) {
                #pragma omp simd
                for (size_t i = 0; i < 252; ++i) {
                    output[4*i] = input[3*i];
                    output[4*i+1] = input[3*i+1];
                    output[4*i+2] = input[3*i+2];
                    output[4*i+3] = Value{};
                }
            } else {
                #pragma omp simd
                for (size_t i = 0; i < 336; ++i) {
                    output[3*i] = input[2*i];
                    output[3*i+1] = input[2*i+1];
                    output[3*i+2] = Value{};
                }
            }
        }
        llrs.swap(scratch);
    }

private:
    size_t _bits;
};

#endif // LDPC_RATE_MATCHING_HPP
