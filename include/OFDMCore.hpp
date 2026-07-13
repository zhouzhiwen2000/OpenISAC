#ifndef OFDM_CORE_HPP
#define OFDM_CORE_HPP

/**
 * @file OFDMCore.hpp
 * @brief Hardware-independent core computation classes for OFDM modulation/demodulation and sensing.
 * 
 * This header contains pure computation classes without any hardware interaction,
 * buffer management, or thread communication. All I/O operations remain in the Engine classes.
 */

#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <fftw3.h>
#include "Common.hpp"
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

/**
 * @brief Check if a float is NaN using bit manipulation.
 * Compatible with -ffast-math compiler flag.
 */
inline bool isNaN(float x) {
    uint32_t bits;
    static_assert(sizeof(float) == sizeof(uint32_t), "Unexpected float size");
    std::memcpy(&bits, &x, sizeof(float));
    constexpr uint32_t exponent_mask = 0x7F800000;
    constexpr uint32_t mantissa_mask = 0x007FFFFF;
    return ((bits & exponent_mask) == exponent_mask) && (bits & mantissa_mask);
}

/**
 * @brief Convert FFT bin index to shifted index.
 */
inline int fftshift_index(int original_index, int N) {
    return (original_index + N / 2) % N;
}

/**
 * @brief Calculate effective RX center frequency in Hz for UHD tuning.
 *
 * UHD RX path effective center is RF - DSP.
 */
inline double rx_effective_center_hz(double actual_rf_freq_hz, double actual_dsp_freq_hz) {
    return actual_rf_freq_hz - actual_dsp_freq_hz;
}

/**
 * @brief Calculate tune-chain systematic CFO in Hz.
 *
 * Assumes TX carrier is exactly at target_tx_center_freq_hz.
 */
inline double rx_tune_system_cfo_hz(double target_tx_center_freq_hz,
                                    double actual_rx_rf_freq_hz,
                                    double actual_rx_dsp_freq_hz) {
    return target_tx_center_freq_hz -
           rx_effective_center_hz(actual_rx_rf_freq_hz, actual_rx_dsp_freq_hz);
}

/**
 * @brief Predict delay correction in samples from observed CFO and elapsed frame time.
 */
inline int _predictive_delay_samples_from_cfo(const Config& cfg,
                                              int64_t source_frame_time_ns,
                                              double detected_freq_offset_hz,
                                              double actual_rx_rf_freq_hz,
                                              double actual_rx_dsp_freq_hz,
                                              int64_t now_ns) {
    if (!cfg.sync_tracking.predictive_delay || source_frame_time_ns < 0 ||
        !std::isfinite(detected_freq_offset_hz) ||
        cfg.rf_sampling.sample_rate <= 0.0 || cfg.samples_per_frame() == 0 ||
        std::abs(cfg.downlink.center_freq) <= 0.0) {
        return 0;
    }

    const double tune_system_cfo_hz = rx_tune_system_cfo_hz(
        cfg.downlink.center_freq,
        actual_rx_rf_freq_hz,
        actual_rx_dsp_freq_hz
    );
    const double clock_error_hz = detected_freq_offset_hz - tune_system_cfo_hz;
    const double clock_error_ratio = clock_error_hz / cfg.downlink.center_freq;
    const double predicted_samples_per_frame =
        clock_error_ratio * static_cast<double>(cfg.samples_per_frame());
    if (!std::isfinite(predicted_samples_per_frame) ||
        std::abs(predicted_samples_per_frame) < 1e-6) {
        return 0;
    }

    const double frame_duration_ns = frame_duration_from_cfg(cfg) * 1.0e9;
    if (!(frame_duration_ns > 0.0)) {
        return 0;
    }

    const double frame_gap_real =
        (static_cast<double>(now_ns) - static_cast<double>(source_frame_time_ns)) /
        frame_duration_ns;
    const int64_t frame_gap = std::max<int64_t>(
        1,
        static_cast<int64_t>(std::floor(frame_gap_real)) + 1
    );

    return static_cast<int>(std::llround(
        -predicted_samples_per_frame * static_cast<double>(frame_gap)));
}

/**
 * @brief Generate frequency-domain Zadoff-Chu sequence.
 */
inline AlignedVector generate_zc_freq(size_t fft_size, int zc_root) {
    AlignedVector zc_freq(fft_size);
    const int n = static_cast<int>(fft_size);
    const int q = zc_root;
    const int delta = (n & 1);
    const double base = -M_PI * static_cast<double>(q) / static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
        const double id = static_cast<double>(i);
        const double phase = base * id * (id + static_cast<double>(delta));
        zc_freq[static_cast<size_t>(i)] = std::polar(1.0f, static_cast<float>(phase));
    }
    return zc_freq;
}

inline int normalize_zc_root(int zc_root, size_t fft_size) {
    if (fft_size == 0) {
        return zc_root;
    }
    const int n = static_cast<int>(fft_size);
    int normalized = zc_root % n;
    if (normalized < 0) {
        normalized += n;
    }
    return normalized;
}

inline int select_known_sensing_pilot_zc_root(size_t fft_size, int sync_zc_root) {
    if (fft_size <= 1) {
        return sync_zc_root + 1;
    }

    const int n = static_cast<int>(fft_size);
    const int normalized_sync_root = normalize_zc_root(sync_zc_root, fft_size);
    for (int step = 1; step < n; ++step) {
        const int candidate = (normalized_sync_root + step) % n;
        if (candidate == 0 || candidate == normalized_sync_root) {
            continue;
        }
        if (std::gcd(candidate, n) == 1) {
            return candidate;
        }
    }

    for (int candidate = 1; candidate < n; ++candidate) {
        if (candidate != normalized_sync_root) {
            return candidate;
        }
    }

    return sync_zc_root + 1;
}

inline AlignedVector generate_sensing_pilot_freq(size_t fft_size, int sync_zc_root) {
    return generate_zc_freq(
        fft_size,
        select_known_sensing_pilot_zc_root(fft_size, sync_zc_root));
}

inline uint32_t splitmix32(uint32_t x) {
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

inline AlignedVector generate_cfo_training_freq(
    size_t fft_size,
    size_t period_samples,
    uint32_t seed = 0x43464F54u)
{
    if (fft_size == 0 || period_samples == 0 || period_samples >= fft_size ||
        (fft_size % period_samples) != 0) {
        throw std::runtime_error(
            "generate_cfo_training_freq requires period_samples to divide fft_size and be in [1, fft_size).");
    }

    AlignedVector symbol(fft_size, std::complex<float>(0.0f, 0.0f));
    const size_t comb_spacing = fft_size / period_samples;
    const float amp = std::sqrt(static_cast<float>(comb_spacing));
    for (size_t m = 0; m < period_samples; ++m) {
        const size_t k = m * comb_spacing;
        const uint32_t rnd = splitmix32(seed ^ static_cast<uint32_t>(m));
        const float re = (rnd & 1u) ? -M_SQRT1_2 : M_SQRT1_2;
        const float im = (rnd & 2u) ? -M_SQRT1_2 : M_SQRT1_2;
        symbol[k] = std::complex<float>(re * amp, im * amp);
    }
    return symbol;
}

/**
 * @brief Generate a deterministic BPSK pilot sequence for a frame symbol.
 *
 * The sequence is indexed by absolute OFDM symbol number and subcarrier, so TX and
 * RX can derive the same reference from YAML alone.
 */
inline AlignedVector generate_midframe_bpsk_pilot_freq(
    size_t fft_size,
    uint32_t seed,
    size_t symbol_index)
{
    AlignedVector pilot(fft_size);
    const uint32_t sym_mix = splitmix32(seed ^ static_cast<uint32_t>(symbol_index));
    for (size_t k = 0; k < fft_size; ++k) {
        const uint32_t rnd = splitmix32(sym_mix ^ static_cast<uint32_t>(k));
        pilot[k] = (rnd & 1u) ? std::complex<float>(-1.0f, 0.0f)
                              : std::complex<float>(1.0f, 0.0f);
    }
    return pilot;
}

inline void overlay_comb_pilot_re(
    AlignedVector& symbol,
    const std::vector<size_t>& pilot_positions,
    const AlignedVector& comb_pilot_seq)
{
    for (const size_t k : pilot_positions) {
        if (k < symbol.size() && k < comb_pilot_seq.size()) {
            symbol[k] = comb_pilot_seq[k];
        }
    }
}

/**
 * @brief Generate a mid-frame pilot while preserving comb-pilot RE.
 *
 * The non-comb-pilot subcarriers use deterministic BPSK, and the configured
 * comb pilot positions keep the regular known pilot sequence. This preserves
 * the normal pilot-phase tracking path on mid-frame pilot symbols.
 */
inline AlignedVector generate_midframe_bpsk_pilot_freq(
    size_t fft_size,
    uint32_t seed,
    size_t symbol_index,
    const std::vector<size_t>& pilot_positions,
    const AlignedVector& comb_pilot_seq)
{
    AlignedVector pilot = generate_midframe_bpsk_pilot_freq(fft_size, seed, symbol_index);
    overlay_comb_pilot_re(pilot, pilot_positions, comb_pilot_seq);
    return pilot;
}

struct LdpcMiniHeader {
    uint8_t version = 1;
    uint8_t flags = 0;
    uint16_t payload_len = 0;
    uint8_t payload_blocks = 0;
    uint16_t seq = 0;
};

/**
 * @brief Shared marker + mini-header framing for LDPC(1008,504) packets.
 */
class LdpcPacketFraming {
public:
    static constexpr size_t kMarkerSymbols = 64;
    static constexpr size_t kMarkerBits = kMarkerSymbols * 2;
    static constexpr size_t kMiniHeaderBits = 64;
    static constexpr size_t kMiniHeaderBchBits = 127;
    static constexpr size_t kMiniHeaderEncodedBits = 128;
    static constexpr size_t kMiniHeaderSymbols = kMiniHeaderEncodedBits / 2;
    static constexpr size_t kMiniHeaderBchT = 10;
    static constexpr size_t kControlSymbols = kMarkerSymbols + kMiniHeaderSymbols;
    static constexpr size_t kControlBits = kControlSymbols * 2;
    static constexpr size_t kLdpcInfoBytesPerBlock = 63;
    static constexpr size_t kLdpcCodeBitsPerBlock = 1008;
    static constexpr size_t kLdpcQpskSymbolsPerBlock = kLdpcCodeBitsPerBlock / 2;
    static constexpr uint8_t kVersion = 1;
    static constexpr uint8_t kFlags = 0;
    // Mini-header flag bits (4-bit field). Data frames use 0; ARQ feedback
    // frames set kFlagArqFeedback so receivers classify them by header, not by
    // sniffing payload bytes. Unknown flag bits are rejected on decode.
    static constexpr uint8_t kFlagArqFeedback = 0x01;
    static constexpr uint8_t kKnownFlagsMask = 0x01;
    static constexpr uint8_t kPayloadBlocksExtended = std::numeric_limits<uint8_t>::max();

    static bool flags_are_known(uint8_t flags) {
        return (flags & static_cast<uint8_t>(~kKnownFlagsMask)) == 0;
    }

    static bool is_arq_feedback_flags(uint8_t flags) {
        return (flags & kFlagArqFeedback) != 0;
    }

    static bool is_arq_feedback(const LdpcMiniHeader& header) {
        return is_arq_feedback_flags(header.flags);
    }
    static constexpr float kMarkerMetricThreshold = 0.50f;

    static size_t max_payload_bytes() {
        return std::numeric_limits<uint16_t>::max();
    }

    static bool payload_len_fits(size_t payload_len) {
        return payload_len <= max_payload_bytes();
    }

    static size_t payload_blocks_for_len(size_t payload_len) {
        if (!payload_len_fits(payload_len)) {
            throw std::runtime_error("LDPC mini-header payload length exceeds representable block count.");
        }
        return (payload_len + kLdpcInfoBytesPerBlock - 1) / kLdpcInfoBytesPerBlock;
    }

    static uint8_t payload_blocks_field_for_len(size_t payload_len) {
        const size_t blocks = payload_blocks_for_len(payload_len);
        return static_cast<uint8_t>(
            std::min<size_t>(blocks, kPayloadBlocksExtended));
    }

    static bool payload_blocks_field_matches(size_t payload_len, uint8_t field) {
        const size_t blocks = payload_blocks_for_len(payload_len);
        if (blocks >= kPayloadBlocksExtended) {
            return field == kPayloadBlocksExtended;
        }
        return field == static_cast<uint8_t>(blocks);
    }

    static size_t padded_payload_len(size_t payload_len) {
        return payload_blocks_for_len(payload_len) * kLdpcInfoBytesPerBlock;
    }

    static size_t packet_qpsk_symbols(size_t payload_blocks) {
        return kControlSymbols + payload_blocks * kLdpcQpskSymbolsPerBlock;
    }

    static int marker_symbol(size_t idx) {
        uint32_t x = 0x4f504953u ^ static_cast<uint32_t>(idx * 0x9e3779b9u);
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return static_cast<int>((x >> 5) & 0x3u);
    }

    template<typename Out>
    static void write_marker_qpsk(Out* out) {
        for (size_t i = 0; i < kMarkerSymbols; ++i) {
            out[i] = static_cast<Out>(marker_symbol(i));
        }
    }

    template<typename Out>
    static void write_mini_header_qpsk(const LdpcMiniHeader& header, Out* out) {
        const uint64_t word = pack_header(header);
        const auto code_bits = bch_encode_header(word);
        for (size_t sym = 0; sym < kMiniHeaderSymbols; ++sym) {
            const size_t bit_idx = sym * 2;
            const int b0 = (bit_idx < kMiniHeaderBchBits) ? code_bits[bit_idx] : 0;
            const int b1 = (bit_idx + 1 < kMiniHeaderBchBits) ? code_bits[bit_idx + 1] : 0;
            out[sym] = static_cast<Out>((b0 << 1) | b1);
        }
    }

    template<typename Out>
    static void write_control_qpsk(const LdpcMiniHeader& header, Out* out) {
        write_marker_qpsk(out);
        write_mini_header_qpsk(header, out + kMarkerSymbols);
    }

    template<typename Llr>
    static float marker_metric_from_llrs(const Llr* llrs) {
        double corr = 0.0;
        double energy = 0.0;
        for (size_t sym = 0; sym < kMarkerSymbols; ++sym) {
            const int qpsk = marker_symbol(sym);
            const int b0 = (qpsk >> 1) & 1;
            const int b1 = qpsk & 1;
            const float l0 = static_cast<float>(llrs[sym * 2]);
            const float l1 = static_cast<float>(llrs[sym * 2 + 1]);
            corr += (b0 ? -l0 : l0);
            corr += (b1 ? -l1 : l1);
            energy += std::abs(l0) + std::abs(l1);
        }
        if (energy <= 1.0e-9) {
            return 0.0f;
        }
        return static_cast<float>(corr / energy);
    }

    template<typename Llr>
    static bool detect_marker_llrs(const Llr* llrs, float* metric_out = nullptr) {
        const float metric = marker_metric_from_llrs(llrs);
        if (metric_out) {
            *metric_out = metric;
        }
        return metric >= kMarkerMetricThreshold;
    }

    template<typename Llr>
    static bool decode_mini_header_llrs(const Llr* llrs, LdpcMiniHeader& header_out) {
        std::array<uint8_t, kMiniHeaderBchBits> code_bits{};
        for (size_t bit = 0; bit < kMiniHeaderBchBits; ++bit) {
            code_bits[bit] = (static_cast<float>(llrs[bit]) < 0.0f) ? 1 : 0;
        }
        uint64_t word = 0;
        return bch_decode_header(code_bits, word) && unpack_header(word, header_out);
    }

    static uint64_t pack_header(const LdpcMiniHeader& header) {
        if (header.version != kVersion || !flags_are_known(header.flags)) {
            throw std::runtime_error("LDPC mini-header version/flags mismatch.");
        }
        if (!payload_len_fits(header.payload_len)) {
            throw std::runtime_error("LDPC mini-header payload length is too large.");
        }
        if (!payload_blocks_field_matches(header.payload_len, header.payload_blocks)) {
            throw std::runtime_error("LDPC mini-header payload block count mismatch.");
        }

        uint64_t prefix = 0;
        prefix |= (static_cast<uint64_t>(header.version & 0x0F) << 44);
        prefix |= (static_cast<uint64_t>(header.flags & 0x0F) << 40);
        prefix |= (static_cast<uint64_t>(header.payload_len) << 24);
        prefix |= (static_cast<uint64_t>(header.payload_blocks) << 16);
        prefix |= static_cast<uint64_t>(header.seq);

        uint8_t bytes[6] = {};
        for (int i = 0; i < 6; ++i) {
            bytes[i] = static_cast<uint8_t>((prefix >> (40 - i * 8)) & 0xFFu);
        }
        const uint16_t crc = crc16_ccitt(bytes, sizeof(bytes));
        return (prefix << 16) | static_cast<uint64_t>(crc);
    }

    static bool unpack_header(uint64_t word, LdpcMiniHeader& header_out) {
        uint8_t bytes[6] = {};
        const uint64_t prefix = word >> 16;
        for (int i = 0; i < 6; ++i) {
            bytes[i] = static_cast<uint8_t>((prefix >> (40 - i * 8)) & 0xFFu);
        }
        const uint16_t expected_crc = crc16_ccitt(bytes, sizeof(bytes));
        const uint16_t got_crc = static_cast<uint16_t>(word & 0xFFFFu);
        if (expected_crc != got_crc) {
            return false;
        }

        LdpcMiniHeader parsed;
        parsed.version = static_cast<uint8_t>((prefix >> 44) & 0x0Fu);
        parsed.flags = static_cast<uint8_t>((prefix >> 40) & 0x0Fu);
        parsed.payload_len = static_cast<uint16_t>((prefix >> 24) & 0xFFFFu);
        parsed.payload_blocks = static_cast<uint8_t>((prefix >> 16) & 0xFFu);
        parsed.seq = static_cast<uint16_t>(prefix & 0xFFFFu);

        if (parsed.version != kVersion || !flags_are_known(parsed.flags)) {
            return false;
        }
        if (!payload_len_fits(parsed.payload_len)) {
            return false;
        }
        if (!payload_blocks_field_matches(parsed.payload_len, parsed.payload_blocks)) {
            return false;
        }

        header_out = parsed;
        return true;
    }

private:
    struct BchTables {
        std::array<uint8_t, 127> exp{};
        std::array<int16_t, 128> log{};
        std::array<uint8_t, 64> generator{};
    };

    static int header_bit(uint64_t word, size_t bit_idx) {
        return static_cast<int>((word >> (63 - bit_idx)) & 0x1u);
    }

    static uint8_t gf_xtime(uint8_t value) {
        uint16_t shifted = static_cast<uint16_t>(value) << 1;
        if (shifted & 0x80u) {
            shifted ^= 0x83u; // x^7 + x + 1
        }
        return static_cast<uint8_t>(shifted & 0x7Fu);
    }

    static const BchTables& bch_tables() {
        static const BchTables tables = []() {
            BchTables t;
            t.log.fill(-1);

            uint8_t value = 1;
            for (int i = 0; i < 127; ++i) {
                t.exp[static_cast<size_t>(i)] = value;
                t.log[static_cast<size_t>(value)] = static_cast<int16_t>(i);
                value = gf_xtime(value);
            }
            if (value != 1) {
                throw std::runtime_error("BCH GF(2^7) primitive polynomial order check failed.");
            }

            auto gf_mul_local = [&t](uint8_t a, uint8_t b) -> uint8_t {
                if (a == 0 || b == 0) {
                    return 0;
                }
                const int sum = t.log[static_cast<size_t>(a)] + t.log[static_cast<size_t>(b)];
                return t.exp[static_cast<size_t>(sum % 127)];
            };

            std::array<uint8_t, 127> root_seen{};
            std::vector<int> root_exponents;
            root_exponents.reserve(63);
            for (int root = 1; root <= static_cast<int>(2 * kMiniHeaderBchT); ++root) {
                int exponent = root % 127;
                for (int i = 0; i < 7; ++i) {
                    if (!root_seen[static_cast<size_t>(exponent)]) {
                        root_seen[static_cast<size_t>(exponent)] = 1;
                        root_exponents.push_back(exponent);
                    }
                    exponent = (exponent * 2) % 127;
                }
            }
            std::sort(root_exponents.begin(), root_exponents.end());

            std::vector<uint8_t> poly(1, 1);
            for (int exponent : root_exponents) {
                const uint8_t root = t.exp[static_cast<size_t>(exponent)];
                std::vector<uint8_t> next(poly.size() + 1, 0);
                for (size_t i = 0; i < poly.size(); ++i) {
                    next[i] ^= gf_mul_local(poly[i], root);
                    next[i + 1] ^= poly[i];
                }
                poly = std::move(next);
            }

            if (poly.size() != t.generator.size()) {
                throw std::runtime_error("BCH(127,64) generator degree mismatch.");
            }
            for (size_t i = 0; i < poly.size(); ++i) {
                if (poly[i] != 0 && poly[i] != 1) {
                    throw std::runtime_error("BCH(127,64) generator is not binary.");
                }
                t.generator[i] = poly[i];
            }
            if (t.generator.back() != 1) {
                throw std::runtime_error("BCH(127,64) generator is not monic.");
            }
            return t;
        }();
        return tables;
    }

    static uint8_t gf_mul(uint8_t a, uint8_t b) {
        if (a == 0 || b == 0) {
            return 0;
        }
        const auto& t = bch_tables();
        const int sum = t.log[static_cast<size_t>(a)] + t.log[static_cast<size_t>(b)];
        return t.exp[static_cast<size_t>(sum % 127)];
    }

    static uint8_t gf_div(uint8_t a, uint8_t b) {
        if (a == 0) {
            return 0;
        }
        if (b == 0) {
            throw std::runtime_error("BCH GF divide by zero.");
        }
        const auto& t = bch_tables();
        int diff = t.log[static_cast<size_t>(a)] - t.log[static_cast<size_t>(b)];
        diff %= 127;
        if (diff < 0) {
            diff += 127;
        }
        return t.exp[static_cast<size_t>(diff)];
    }

    static std::array<uint8_t, kMiniHeaderBchBits> bch_encode_header(uint64_t word) {
        const auto& tables = bch_tables();
        std::array<uint8_t, kMiniHeaderBchBits> work{};
        std::array<uint8_t, kMiniHeaderBits> message{};
        for (size_t i = 0; i < kMiniHeaderBits; ++i) {
            message[i] = static_cast<uint8_t>(header_bit(word, i));
            work[63 + i] = message[i];
        }

        for (int pos = static_cast<int>(kMiniHeaderBchBits) - 1; pos >= 63; --pos) {
            if (work[static_cast<size_t>(pos)] == 0) {
                continue;
            }
            const int shift = pos - 63;
            for (size_t j = 0; j < tables.generator.size(); ++j) {
                work[static_cast<size_t>(shift) + j] ^= tables.generator[j];
            }
        }

        std::array<uint8_t, kMiniHeaderBchBits> code_bits{};
        for (size_t i = 0; i < 63; ++i) {
            code_bits[i] = work[i] & 1u;
        }
        for (size_t i = 0; i < kMiniHeaderBits; ++i) {
            code_bits[63 + i] = message[i];
        }
        return code_bits;
    }

    static std::array<uint8_t, 2 * kMiniHeaderBchT> bch_syndromes(
        const std::array<uint8_t, kMiniHeaderBchBits>& code_bits)
    {
        const auto& tables = bch_tables();
        std::array<uint8_t, 2 * kMiniHeaderBchT> syndromes{};
        for (size_t syndrome = 1; syndrome <= syndromes.size(); ++syndrome) {
            uint8_t accum = 0;
            for (size_t pos = 0; pos < kMiniHeaderBchBits; ++pos) {
                if (code_bits[pos]) {
                    accum ^= tables.exp[(syndrome * pos) % 127];
                }
            }
            syndromes[syndrome - 1] = accum;
        }
        return syndromes;
    }

    static bool bch_decode_header(
        std::array<uint8_t, kMiniHeaderBchBits> code_bits,
        uint64_t& word_out)
    {
        constexpr size_t kSyndromeCount = 2 * kMiniHeaderBchT;
        const auto& tables = bch_tables();
        auto syndromes = bch_syndromes(code_bits);

        bool has_error = false;
        for (uint8_t s : syndromes) {
            has_error = has_error || (s != 0);
        }
        if (has_error) {
            std::array<uint8_t, kSyndromeCount + 1> locator{};
            std::array<uint8_t, kSyndromeCount + 1> previous{};
            locator[0] = 1;
            previous[0] = 1;
            size_t degree = 0;
            size_t shift = 1;
            uint8_t scale = 1;

            for (size_t n = 0; n < kSyndromeCount; ++n) {
                uint8_t discrepancy = syndromes[n];
                for (size_t i = 1; i <= degree; ++i) {
                    if (locator[i] != 0 && syndromes[n - i] != 0) {
                        discrepancy ^= gf_mul(locator[i], syndromes[n - i]);
                    }
                }

                if (discrepancy == 0) {
                    ++shift;
                    continue;
                }

                const auto saved = locator;
                const uint8_t factor = gf_div(discrepancy, scale);
                for (size_t i = 0; i + shift < locator.size(); ++i) {
                    if (previous[i]) {
                        locator[i + shift] ^= gf_mul(factor, previous[i]);
                    }
                }

                if (2 * degree <= n) {
                    degree = n + 1 - degree;
                    previous = saved;
                    scale = discrepancy;
                    shift = 1;
                } else {
                    ++shift;
                }
            }

            if (degree > kMiniHeaderBchT) {
                return false;
            }

            std::vector<size_t> error_positions;
            error_positions.reserve(degree);
            for (size_t pos = 0; pos < kMiniHeaderBchBits; ++pos) {
                const uint8_t x = tables.exp[(127 - (pos % 127)) % 127];
                uint8_t value = locator[0];
                uint8_t x_power = 1;
                for (size_t i = 1; i <= degree; ++i) {
                    x_power = gf_mul(x_power, x);
                    if (locator[i]) {
                        value ^= gf_mul(locator[i], x_power);
                    }
                }
                if (value == 0) {
                    error_positions.push_back(pos);
                }
            }

            if (error_positions.size() != degree) {
                return false;
            }
            for (size_t pos : error_positions) {
                code_bits[pos] ^= 1u;
            }

            const auto corrected_syndromes = bch_syndromes(code_bits);
            for (uint8_t s : corrected_syndromes) {
                if (s != 0) {
                    return false;
                }
            }
        }

        uint64_t word = 0;
        for (size_t i = 0; i < kMiniHeaderBits; ++i) {
            word = (word << 1) | static_cast<uint64_t>(code_bits[63 + i] & 1u);
        }
        word_out = word;
        return true;
    }

    static uint16_t crc16_ccitt(const uint8_t* data, size_t len) {
        uint16_t crc = 0xFFFFu;
        for (size_t i = 0; i < len; ++i) {
            crc ^= static_cast<uint16_t>(data[i]) << 8;
            for (int bit = 0; bit < 8; ++bit) {
                crc = (crc & 0x8000u)
                    ? static_cast<uint16_t>((crc << 1) ^ 0x1021u)
                    : static_cast<uint16_t>(crc << 1);
            }
        }
        return crc;
    }
};


/**
 * @brief Manager for FFTW Wisdom.
 * 
 * Handles importing and exporting FFTW wisdom to/from a file.
 * This allows saving optimized FFT plans to disk to speed up subsequent initializations.
 */
class FFTWManager {
public:
    static void import_wisdom(const std::string& filename = "fftw_wisdom.dat") {
        if (FILE* f = std::fopen(filename.c_str(), "r")) {
            fftwf_import_wisdom_from_file(f);
            std::fclose(f);
            LOG_G_INFO() << "Imported FFTW wisdom from " << filename;
        } else {
            LOG_G_INFO() << "No existing FFTW wisdom found (will act as cold start).";
        }
    }

    static void export_wisdom(const std::string& filename = "fftw_wisdom.dat") {
        if (FILE* f = std::fopen(filename.c_str(), "w")) {
            fftwf_export_wisdom_to_file(f);
            std::fclose(f);
            LOG_G_INFO() << "Exported FFTW wisdom to " << filename;
        } else {
            LOG_G_ERROR() << "Failed to export FFTW wisdom to " << filename;
        }
    }
};

/**
 * @brief QPSK Scrambler/Descrambler.
 * 
 * Uses a Linear Feedback Shift Register (LFSR) to generate a pseudo-random sequence
 * for scrambling and descrambling bits. This helps in randomizing the data to avoid
 * long sequences of zeros or ones, which avoids high PAPRs in OFDM.
 */
class Scrambler {
public:
    Scrambler(size_t max_bits, uint8_t init = 0x5A)
        : scramble_seq_(max_bits)
    {
        uint8_t lfsr = init;
        for (size_t i = 0; i < max_bits; ++i) {
            scramble_seq_[i] = ((lfsr >> 7) ^ (lfsr >> 3) ^ (lfsr >> 2) ^ (lfsr >> 1)) & 1;
            lfsr = ((lfsr << 1) | scramble_seq_[i]) & 0xFF;
        }
    }

    // Scramble (in-place)
    template<typename Vec>
    void scramble(Vec& bits) const {
        size_t n = bits.size();
        size_t m = std::min(n, scramble_seq_.size());
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < m; ++i) {
            bits[i] ^= scramble_seq_[i];
        }
    }

    // Descramble (in-place)
    template<typename Vec>
    void descramble(Vec& bits) const {
        scramble(bits); // Same as scrambling
    }

    // Soft descramble (descramble LLR values)
    template<typename FloatVec>
    void soft_descramble(FloatVec& llr_values) const {
        size_t n = llr_values.size();
        size_t m = std::min(n, scramble_seq_.size());
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < m; ++i) {
            if (scramble_seq_[i] == 1) {
                llr_values[i] = -llr_values[i]; // Flip LLR sign if scramble bit is 1
            }
            // Keep LLR as is if scramble bit is 0
        }
    }

private:
    std::vector<uint8_t> scramble_seq_;
};

/**
 * @brief Fixed block bit interleaver for LDPC-coded QPSK payloads.
 *
 * The permutation is a row/column matrix transpose applied independently to
 * each coded block. For the default LDPC(1008,504) path, a 21 x 48 layout
 * keeps one dimension friendly to wide SIMD gathers on the CPU.
 */
class BitBlockInterleaver {
public:
    BitBlockInterleaver(size_t block_size, size_t rows)
        : _block_size(block_size),
          _rows(rows),
          _cols((rows > 0 && (block_size % rows) == 0) ? (block_size / rows) : 0),
          _interleave_map(block_size),
          _deinterleave_map(block_size)
    {
        if (_block_size == 0 || _rows == 0 || _cols == 0) {
            throw std::runtime_error("BitBlockInterleaver requires block_size divisible by rows.");
        }

        for (size_t row = 0; row < _rows; ++row) {
            for (size_t col = 0; col < _cols; ++col) {
                const size_t src = row * _cols + col;
                const size_t dst = col * _rows + row;
                _interleave_map[dst] = static_cast<uint16_t>(src);
                _deinterleave_map[src] = static_cast<uint16_t>(dst);
            }
        }
    }

    size_t block_size() const { return _block_size; }

    template<typename T, typename Alloc>
    void interleave_inplace(
        std::vector<T, Alloc>& values,
        std::vector<T, Alloc>& scratch
    ) const {
        apply_map_inplace(values, _interleave_map, scratch);
    }

    template<typename T, typename Alloc>
    void deinterleave_inplace(
        std::vector<T, Alloc>& values,
        std::vector<T, Alloc>& scratch
    ) const {
        apply_map_inplace(values, _deinterleave_map, scratch);
    }

private:
    template<typename T, typename Alloc>
    void apply_map_inplace(
        std::vector<T, Alloc>& values,
        const std::vector<uint16_t>& map,
        std::vector<T, Alloc>& scratch
    ) const {
        if (values.empty()) {
            return;
        }
        if ((values.size() % _block_size) != 0) {
            throw std::runtime_error("BitBlockInterleaver input size must be a multiple of block_size.");
        }

        scratch.resize(_block_size);
        T* const scratch_ptr = scratch.data();
        T* const values_ptr = values.data();
        const size_t block_count = values.size() / _block_size;
        for (size_t block = 0; block < block_count; ++block) {
            const T* const block_ptr = values_ptr + block * _block_size;
            // No omp simd: the map[i] indirection forces a gather; measured ~1.18x
            // slower than the scalar loop the compiler picks on its own.
            for (size_t i = 0; i < _block_size; ++i) {
                scratch_ptr[i] = block_ptr[map[i]];
            }
            std::memcpy(values_ptr + block * _block_size, scratch_ptr, _block_size * sizeof(T));
        }
    }

    size_t _block_size = 0;
    size_t _rows = 0;
    size_t _cols = 0;
    std::vector<uint16_t> _interleave_map;
    std::vector<uint16_t> _deinterleave_map;
};

/**
 * @brief QPSK Modulator/Demodulator.
 * 
 * Provides QPSK modulation and demodulation operations using a pre-computed
 * lookup table for maximum performance.
 */
class QPSKModulator {
public:
    static constexpr float SQRT_2_INV = 0.7071067811865476f;
    
    // Pre-computed QPSK constellation lookup table
    static constexpr std::array<float, 8> QPSK_TABLE_FLAT = {{
        SQRT_2_INV,  SQRT_2_INV,   // 00: (+, +)
        SQRT_2_INV, -SQRT_2_INV,   // 01: (+, -)
       -SQRT_2_INV,  SQRT_2_INV,   // 10: (-, +)
       -SQRT_2_INV, -SQRT_2_INV    // 11: (-, -)
    }};

    /**
     * @brief Map 2-bit symbol (0-3) to complex constellation point.
     */
    inline std::complex<float> modulate(int symbol) const {
        const int idx = (symbol & 3) * 2;
        return std::complex<float>(QPSK_TABLE_FLAT[idx], QPSK_TABLE_FLAT[idx + 1]);
    }

    /**
     * @brief Hard decision demodulation (constellation point to 0-3).
     */
    inline int demodulate(std::complex<float> symbol) const {
        // MSB from real part, LSB from imaginary part
        return ((symbol.real() < 0.0f) ? 2 : 0) | ((symbol.imag() < 0.0f) ? 1 : 0);
    }

    /**
     * @brief Remodulate equalized symbol using hard decision.
     * Returns the closest QPSK constellation point.
     */
    inline std::complex<float> remodulate(std::complex<float> equalized_symbol) const {
        return std::complex<float>(
            std::copysign(SQRT_2_INV, equalized_symbol.real()),
            std::copysign(SQRT_2_INV, equalized_symbol.imag())
        );
    }

    /**
     * @brief Remodulate entire symbol using hard decision QPSK mapping.
     * Replaces pilot positions with known pilot values.
     */
    static void remodulate_symbol(
        const AlignedVector& equalized_symbol,
        const AlignedVector& pilots,
        const std::vector<size_t>& pilot_positions,
        AlignedVector& output
    ) {
        const size_t fft_size = equalized_symbol.size();
        output.resize(fft_size);
        
        auto* __restrict__ out_ptr = output.data();
        const auto* __restrict__ sym_ptr = equalized_symbol.data();
        
        // SIMD-friendly hard decision QPSK remodulation
        #pragma omp simd simdlen(16)
        for (size_t j = 0; j < fft_size; ++j) {
            const float re = sym_ptr[j].real();
            const float im = sym_ptr[j].imag();
            out_ptr[j] = std::complex<float>(
                std::copysign(SQRT_2_INV, re),
                std::copysign(SQRT_2_INV, im)
            );
        }
        
        // Replace pilot positions with known pilots
        for (auto pilot : pilot_positions) {
            out_ptr[pilot] = pilots[pilot];
        }
    }
};


/**
 * @brief Zadoff-Chu Sequence Generator.
 * 
 * Generates Zadoff-Chu sequences for synchronization and pilot symbols.
 * ZC sequences have constant amplitude and zero autocorrelation.
 */
class ZadoffChuGenerator {
public:
    /**
     * @brief Generate ZC sequence of given length and root.
     * 
     * @param output Output buffer (will be resized to length)
     * @param length Sequence length (typically FFT size)
     * @param root ZC sequence root index
     */
    static void generate(AlignedVector& output, size_t length, int root) {
        output.resize(length);
        
        const int N = static_cast<int>(length);
        const int q = root;
        
        // delta: even N -> 0, odd N -> 1
        const int delta = (N & 1);
        
        // Pre-calculate constant coefficient
        const double base = -M_PI * static_cast<double>(q) / static_cast<double>(N);
        
        #pragma omp simd simdlen(16)
        for (int n = 0; n < N; ++n) {
            const double nd = static_cast<double>(n);
            const double arg = nd * (nd + static_cast<double>(delta));
            const double phase = base * arg;
            output[n] = std::polar(1.0f, static_cast<float>(phase));
        }
    }
};


/**
 * @brief Hamming Window Generator.
 * 
 * Generates Hamming windows for range and Doppler processing.
 */
class WindowGenerator {
public:
    /**
     * @brief Generate Hamming window of given size.
     * Formula: w(n) = 0.54 - 0.46*cos(2πn/(N-1))
     */
    static void generate_hamming(AlignedFloatVector& output, size_t length) {
        output.resize(length);
        const float factor = 2.0f * static_cast<float>(M_PI) / (length - 1);
        
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < length; ++i) {
            output[i] = 0.54f - 0.46f * std::cos(factor * i);
        }
    }
};


/**
 * @brief Channel Estimator.
 * 
 * Provides various channel estimation methods for OFDM receivers.
 * LMMSE estimation requires instance with internal FFT plans.
 * LS estimation and equalization are static (no FFT needed).
 */
class ChannelEstimator {
public:
    /**
     * @brief Construct a ChannelEstimator with internal FFT plans for LMMSE.
     * @param fft_size FFT size for the plans
     */
    explicit ChannelEstimator(size_t fft_size)
        : _fft_size(fft_size),
          _scratch_buf1(fft_size),
          _scratch_buf2(fft_size),
          _H_est_internal(fft_size)
    {
        _fft_plan = fftwf_plan_dft_1d(
            static_cast<int>(fft_size),
            reinterpret_cast<fftwf_complex*>(_scratch_buf1.data()),
            reinterpret_cast<fftwf_complex*>(_scratch_buf2.data()),
            FFTW_FORWARD, FFTW_MEASURE);
        
        _ifft_plan = fftwf_plan_dft_1d(
            static_cast<int>(fft_size),
            reinterpret_cast<fftwf_complex*>(_scratch_buf1.data()),
            reinterpret_cast<fftwf_complex*>(_scratch_buf2.data()),
            FFTW_BACKWARD, FFTW_MEASURE);
    }

    ~ChannelEstimator() {
        if (_fft_plan) fftwf_destroy_plan(_fft_plan);
        if (_ifft_plan) fftwf_destroy_plan(_ifft_plan);
    }

    // Non-copyable due to FFTW plans
    ChannelEstimator(const ChannelEstimator&) = delete;
    ChannelEstimator& operator=(const ChannelEstimator&) = delete;

    // Move constructible
    ChannelEstimator(ChannelEstimator&& other) noexcept
        : _fft_size(other._fft_size),
          _fft_plan(other._fft_plan),
          _ifft_plan(other._ifft_plan),
          _scratch_buf1(std::move(other._scratch_buf1)),
          _scratch_buf2(std::move(other._scratch_buf2)),
          _H_est_internal(std::move(other._H_est_internal))
    {
        other._fft_plan = nullptr;
        other._ifft_plan = nullptr;
    }

    /**
     * @brief Estimate SNR from time-domain channel impulse response.
     * Assumes channel energy is concentrated within CP length, and the rest is noise.
     * 
     * @param h_time Time-domain channel estimate (impulse response)
     * @param cp_length Cyclic prefix length (assumed delay spread boundary)
     * @return Estimated SNR (linear scale)
     */
    static float estimate_snr_from_impulse_response(
        const AlignedVector& h_time,
        size_t cp_length
    ) {
        const size_t fft_size = h_time.size();
        if (fft_size <= cp_length) return 10.0f; // Fallback

        double signal_energy = 0.0;
        double noise_energy = 0.0;

        #pragma omp simd simdlen(16) reduction(+:signal_energy)
        for (size_t i = 0; i < cp_length; ++i) {
            signal_energy += std::norm(h_time[i]);
        }
        
        #pragma omp simd simdlen(16) reduction(+:noise_energy)
        for (size_t i = cp_length; i < fft_size; ++i) {
            noise_energy += std::norm(h_time[i]);
        }

        double noise_power = noise_energy / (fft_size - cp_length);
        double signal_power = (signal_energy / cp_length) - noise_power;

        if (signal_power < 0.0) signal_power = 0.0;
        if (noise_power < 1e-10) noise_power = 1e-10;

        return static_cast<float>(signal_power / noise_power);
    }

    /**
     * @brief Estimate channel using LMMSE (DFT-based with Wiener smoothing).
     * Algorithm: H_ls -> IFFT -> Window/Denoise -> FFT -> H_lmmse
     * SNR is estimated dynamically from the Time-Domain Channel Impulse Response.
     * 
     * @param rx_symbol Received frequency domain symbol
     * @param tx_zc Known transmitted ZC sequence
     * @param H_est Output channel estimate (will be resized)
     * @param cp_length Cyclic prefix length (assumed max delay spread)
     */
    void estimate_from_sync_lmmse(
        const AlignedVector& rx_symbol,
        const AlignedVector& tx_zc,
        AlignedVector& H_est,
        size_t cp_length,
        float* corrected_snr_linear_out = nullptr
    ) {
        H_est.resize(_fft_size);

        // 1. LS Estimation into scratch_buf1
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _fft_size; ++i) {
            float rx_real = rx_symbol[i].real();
            float rx_imag = rx_symbol[i].imag();
            float tx_real = tx_zc[i].real();
            float tx_imag = tx_zc[i].imag();
            float denom = tx_real * tx_real + tx_imag * tx_imag;
            float inv_denom = 1.0f / denom;
            _scratch_buf1[i] = std::complex<float>(
                (rx_real * tx_real + rx_imag * tx_imag) * inv_denom,
                (rx_imag * tx_real - rx_real * tx_imag) * inv_denom
            );
        }

        // 2. IFFT: scratch_buf1 (H_ls) -> scratch_buf2 (h_ls)
        fftwf_execute_dft(_ifft_plan, 
            reinterpret_cast<fftwf_complex*>(_scratch_buf1.data()), 
            reinterpret_cast<fftwf_complex*>(_scratch_buf2.data()));

        // 3. SNR Estimation & Wiener Smoothing
        float n_float = static_cast<float>(_fft_size);
        // Normalize FFTW output (1/N) for SNR estimation
        float scale = 1.0f / n_float;
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _fft_size; ++i) {
            _scratch_buf2[i] *= scale;
        }

        float snr_est = estimate_snr_from_impulse_response(_scratch_buf2, cp_length);
        if (snr_est < 1e-4f) snr_est = 1e-4f;
        if (corrected_snr_linear_out != nullptr) {
            *corrected_snr_linear_out =
                corrected_impulse_snr_linear(snr_est, _fft_size, cp_length);
        }

        // Wiener coefficient: w = SNR / (SNR + 1)
        float w_pass = snr_est / (snr_est + 1.0f);
        
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _fft_size; ++i) {
            if (i < cp_length || i >= (_fft_size - cp_length)) {
                _scratch_buf2[i] *= w_pass;
            } else {
                _scratch_buf2[i] = {0.0f, 0.0f};
            }
        }

        // 4. FFT: scratch_buf2 (h_denoised) -> H_est (H_lmmse)
        fftwf_execute_dft(_fft_plan, 
            reinterpret_cast<fftwf_complex*>(_scratch_buf2.data()), 
            reinterpret_cast<fftwf_complex*>(H_est.data()));
    }

    /**
     * @brief Estimate channel from sync symbol using conjugate multiplication: H = Rx * conj(Tx)
     * Optimized for unit-magnitude sync sequences (e.g., ZC sequences where |Tx| = 1).
     * 
     * @param rx_symbol Received frequency domain symbol
     * @param tx_zc Known transmitted ZC sequence (unit magnitude)
     * @param H_est Output channel estimate (will be resized)
     */
    static void estimate_from_sync_ls(
        const AlignedVector& rx_symbol,
        const AlignedVector& tx_zc,
        AlignedVector& H_est
    ) {
        const size_t fft_size = rx_symbol.size();
        H_est.resize(fft_size);
        
        const auto* __restrict__ rx_ptr = rx_symbol.data();
        const auto* __restrict__ tx_ptr = tx_zc.data();
        auto* __restrict__ h_ptr = H_est.data();
        
        #pragma omp simd simdlen(16) aligned(rx_ptr, tx_ptr, h_ptr: 64)
        for (size_t i = 0; i < fft_size; ++i) {
            float rx_real = rx_ptr[i].real();
            float rx_imag = rx_ptr[i].imag();
            float tx_real = tx_ptr[i].real();
            float tx_imag = tx_ptr[i].imag();
            
            // Multiply by conjugate: (a+bi)*(c-di) = (ac+bd) + (bc-ad)i
            h_ptr[i] = std::complex<float>(
                rx_real * tx_real + rx_imag * tx_imag,
                rx_imag * tx_real - rx_real * tx_imag
            );
        }
    }

    /**
     * @brief Compute ZF equalizer inverse: H_inv = conj(H) / |H|^2 = 1/H
     */
    static void compute_zf_inverse(
        const AlignedVector& H_est,
        AlignedVector& H_inv,
        float mag_sq_floor = 1e-6f
    ) {
        const size_t fft_size = H_est.size();
        H_inv.resize(fft_size);
        const float floor_val = std::max(mag_sq_floor, 1e-12f);
        
        #pragma omp simd simdlen(16)
        for (size_t j = 0; j < fft_size; ++j) {
            const float h_real = H_est[j].real();
            const float h_imag = H_est[j].imag();
            const float mag_sq = std::max(h_real * h_real + h_imag * h_imag, floor_val);
            const float inv_mag_sq = 1.0f / mag_sq;
            H_inv[j] = std::complex<float>(h_real * inv_mag_sq, -h_imag * inv_mag_sq);
        }
    }

    /**
     * @brief Compute MMSE/regularized inverse: H_inv = conj(H) / (|H|^2 + noise_var).
     *
     * `noise_var` must be in the same received-frequency-domain scale as |H|^2
     * for unit-power transmitted constellation symbols.
     */
    static void compute_mmse_inverse(
        const AlignedVector& H_est,
        AlignedVector& H_inv,
        float noise_var,
        float mag_sq_floor = 1e-6f
    ) {
        const size_t fft_size = H_est.size();
        H_inv.resize(fft_size);
        const float regularization = std::max(noise_var, 0.0f);
        const float floor_val = std::max(mag_sq_floor, 1e-12f);

        #pragma omp simd simdlen(16)
        for (size_t j = 0; j < fft_size; ++j) {
            const float h_real = H_est[j].real();
            const float h_imag = H_est[j].imag();
            const float mag_sq = std::max(h_real * h_real + h_imag * h_imag, floor_val);
            const float inv_denom = 1.0f / (mag_sq + regularization);
            H_inv[j] = std::complex<float>(h_real * inv_denom, -h_imag * inv_denom);
        }
    }

    /**
     * @brief Equalize symbol with channel inverse and phase compensation.
     * symbol = symbol * H_inv * exp(-j*phase)
     */
    static void equalize_symbol(
        AlignedVector& symbol,
        const AlignedVector& H_inv,
        float phase_diff_CFO,
        float beta_rel,
        const std::vector<int>& subcarrier_indices
    ) {
        const size_t fft_size = symbol.size();
        if (fft_size == 0) return;

        // The derotation phase is phase(j) = phase_diff_CFO + beta_rel * idx(j).
        // For the standard FFT-shift layout idx(j) increments by exactly 1 within
        // each contiguous run, so the derotation phasor exp(-j*phase) follows a
        // geometric recurrence: phasor(j+1) = phasor(j) * step, step = exp(-j*beta_rel).
        // We exploit this with a *blocked* recurrence: a width-W lane vector is
        // advanced by step^W per block, so the per-element transcendentals (one
        // sin/cos pair per subcarrier in the old code, ~100K per frame) collapse to
        // W sincos per run. Each block stays vectorizable (no cross-lane dependency).
        constexpr size_t W = 16;
        const std::complex<float> step = std::polar(1.0f, -beta_rel);
        const std::complex<float> step_block = std::polar(1.0f, -beta_rel * static_cast<float>(W));

        auto* __restrict__ sym_ptr = symbol.data();
        const auto* __restrict__ hinv_ptr = H_inv.data();

        if (std::abs(phase_diff_CFO) <= 1e-12f && std::abs(beta_rel) <= 1e-12f) {
            #pragma omp simd simdlen(16)
            for (size_t j = 0; j < fft_size; ++j) {
                const float hinv_real = hinv_ptr[j].real();
                const float hinv_imag = hinv_ptr[j].imag();
                const float sym_real = sym_ptr[j].real();
                const float sym_imag = sym_ptr[j].imag();
                sym_ptr[j] = std::complex<float>(
                    sym_real * hinv_real - sym_imag * hinv_imag,
                    sym_real * hinv_imag + sym_imag * hinv_real);
            }
            return;
        }

        // Process a contiguous run [start, end) whose subcarrier index increments by
        // +1 each step, starting from idx0 = subcarrier_indices[start].
        auto process_run = [&](size_t start, size_t end, int idx0) {
            if (start >= end) return;
            const float theta0 = phase_diff_CFO + beta_rel * static_cast<float>(idx0);
            // Lane phasors: lane[w] = exp(-j*(theta0 + w*beta_rel)).
            std::complex<float> lane[W];
            for (size_t w = 0; w < W; ++w) {
                lane[w] = std::polar(1.0f, -(theta0 + beta_rel * static_cast<float>(w)));
            }
            size_t j = start;
            // Full vectorizable blocks of W.
            for (; j + W <= end; j += W) {
                #pragma omp simd simdlen(16)
                for (size_t w = 0; w < W; ++w) {
                    const size_t idx = j + w;
                    const float hinv_real = hinv_ptr[idx].real();
                    const float hinv_imag = hinv_ptr[idx].imag();
                    const float sym_real = sym_ptr[idx].real();
                    const float sym_imag = sym_ptr[idx].imag();
                    const float eq_re = sym_real * hinv_real - sym_imag * hinv_imag;
                    const float eq_im = sym_real * hinv_imag + sym_imag * hinv_real;
                    const float c = lane[w].real();
                    const float s = lane[w].imag(); // s already carries the -sin sign
                    sym_ptr[idx] = std::complex<float>(eq_re * c - eq_im * s,
                                                       eq_im * c + eq_re * s);
                }
                // Advance every lane by one full block.
                #pragma omp simd simdlen(16)
                for (size_t w = 0; w < W; ++w) {
                    lane[w] *= step_block;
                }
            }
            // Scalar tail (run length not a multiple of W).
            std::complex<float> phasor = lane[0];
            for (; j < end; ++j) {
                const float hinv_real = hinv_ptr[j].real();
                const float hinv_imag = hinv_ptr[j].imag();
                const float sym_real = sym_ptr[j].real();
                const float sym_imag = sym_ptr[j].imag();
                const float eq_re = sym_real * hinv_real - sym_imag * hinv_imag;
                const float eq_im = sym_real * hinv_imag + sym_imag * hinv_real;
                const float c = phasor.real();
                const float s = phasor.imag();
                sym_ptr[j] = std::complex<float>(eq_re * c - eq_im * s,
                                                 eq_im * c + eq_re * s);
                phasor *= step;
            }
        };

        // Split into contiguous +1 runs at the FFT-shift discontinuity. For the
        // standard layout this is [0, half) (idx 0..half-1) and [half, fft) (idx
        // -half..-1); deriving idx0 from subcarrier_indices keeps it correct for the
        // actual layout passed in.
        const size_t half = fft_size / 2;
        process_run(0, half, subcarrier_indices[0]);
        process_run(half, fft_size, subcarrier_indices[half]);
    }

private:
    size_t _fft_size;
    fftwf_plan _fft_plan = nullptr;
    fftwf_plan _ifft_plan = nullptr;
    AlignedVector _scratch_buf1;
    AlignedVector _scratch_buf2;
    AlignedVector _H_est_internal;
};

/**
 * @brief Extract a local-TX ZC symbol from an RX frame and estimate self-channel.
 *
 * The RX frame is treated as a circular frame-period window. `tx_symbol_start`
 * is the local transmitter's ZC symbol start relative to its own frame anchor.
 * `rx_frame_start_offset` is the RX window's first sample relative to that same
 * local TX anchor. Pass zero to sample the ideal local-TX frame position.
 */
class SelfZcChannelDebugEstimator {
public:
    SelfZcChannelDebugEstimator(size_t fft_size, size_t cp_length)
        : _fft_size(fft_size),
          _cp_length(cp_length),
          _fft_in(fft_size),
          _fft_out(fft_size) {
        _fft_plan = fftwf_plan_dft_1d(
            static_cast<int>(_fft_size),
            reinterpret_cast<fftwf_complex*>(_fft_in.data()),
            reinterpret_cast<fftwf_complex*>(_fft_out.data()),
            FFTW_FORWARD,
            FFTW_MEASURE);
    }

    ~SelfZcChannelDebugEstimator() {
        if (_fft_plan) {
            fftwf_destroy_plan(_fft_plan);
        }
    }

    SelfZcChannelDebugEstimator(const SelfZcChannelDebugEstimator&) = delete;
    SelfZcChannelDebugEstimator& operator=(const SelfZcChannelDebugEstimator&) = delete;

    bool estimate(
        const AlignedVector& rx_frame,
        size_t tx_symbol_start,
        int64_t rx_frame_start_offset,
        const AlignedVector& tx_zc,
        AlignedVector& h_est)
    {
        if (_fft_size == 0 || tx_zc.size() < _fft_size || rx_frame.size() < _fft_size) {
            return false;
        }
        const int64_t period = static_cast<int64_t>(rx_frame.size());
        const int64_t useful_start = static_cast<int64_t>(tx_symbol_start + _cp_length) -
                                     rx_frame_start_offset;
        int64_t start = useful_start % period;
        if (start < 0) {
            start += period;
        }

        size_t src = static_cast<size_t>(start);
        for (size_t i = 0; i < _fft_size; ++i) {
            _fft_in[i] = rx_frame[src];
            if (++src == rx_frame.size()) {
                src = 0;
            }
        }

        fftwf_execute(_fft_plan);
        _rx_freq.resize(_fft_size);
        const float scale = 1.0f / std::sqrt(static_cast<float>(_fft_size));
        for (size_t k = 0; k < _fft_size; ++k) {
            _rx_freq[k] = _fft_out[k] * scale;
        }
        ChannelEstimator::estimate_from_sync_ls(_rx_freq, tx_zc, h_est);
        return true;
    }

private:
    size_t _fft_size = 0;
    size_t _cp_length = 0;
    AlignedVector _fft_in;
    AlignedVector _fft_out;
    AlignedVector _rx_freq;
    fftwf_plan _fft_plan = nullptr;
};


/**
 * @brief Synchronization Processor.
 * 
 * Provides synchronization correlation and CFO estimation.
 * Maintains pre-allocated FFT plans and buffers for efficient reuse.
 */
class SyncProcessor {
public:
    struct SecSyncCoarseSyncResult {
        int coarse_symbol_start = 0;
        int coarse_useful_start = 0;
        float max_metric = 0.0f;
        double coarse_cfo_hz = 0.0;
        bool valid = false;
    };

    struct SecSyncRefineResult {
        struct AliasCandidate {
            int alias_index = 0;
            int max_pos = 0;
            float max_corr = 0.0f;
            float avg_corr = 0.0f;
            double cfo_hz = 0.0;
            bool valid = false;
        };

        int max_pos = 0;
        float max_corr = 0.0f;
        float avg_corr = 0.0f;
        double cfo_hz = 0.0;
        int alias_index = 0;
        bool valid = false;
        std::vector<AliasCandidate> alias_candidates;
    };

    struct CfoTrainingEstimate {
        double cfo_hz = 0.0;
        float metric = 0.0f;
        bool valid = false;
    };

    /**
     * @brief Construct a SyncProcessor with pre-allocated FFT plans and sync sequence.
     * @param data_len Expected data length (2 frames worth)
     * @param fft_size FFT size
     * @param cp_length Cyclic prefix length
     * @param zc_freq Pre-generated ZC sequence in frequency domain
     */
    explicit SyncProcessor(size_t data_len, size_t fft_size, size_t cp_length, const AlignedVector& zc_freq)
        : _symbol_len(fft_size + cp_length),
          _data_len(data_len),
          _fft_size(fft_size),
          _cp_length(cp_length)
    {
        // Calculate FFT size (next power of 2 for linear correlation)
        _fft_len = 1;
        const size_t min_len = data_len + _symbol_len - 1;
        while (_fft_len < min_len) _fft_len <<= 1;
        
        // Allocate buffers
        _x_padded.resize(_fft_len, {0.0f, 0.0f});
        _h_padded.resize(_fft_len, {0.0f, 0.0f});
        _X.resize(_fft_len);
        _H.resize(_fft_len);
        _corr_result.resize(_fft_len);
        
        // Generate sync sequence and prepare _h_padded using provided zc_freq
        prepare_sync_sequence(zc_freq);
        
        // Create FFT plans (use FFTW_MEASURE for optimal performance)
        _fft_x = fftwf_plan_dft_1d(
            static_cast<int>(_fft_len),
            reinterpret_cast<fftwf_complex*>(_x_padded.data()),
            reinterpret_cast<fftwf_complex*>(_X.data()),
            FFTW_FORWARD, FFTW_MEASURE);
        
        _fft_h = fftwf_plan_dft_1d(
            static_cast<int>(_fft_len),
            reinterpret_cast<fftwf_complex*>(_h_padded.data()),
            reinterpret_cast<fftwf_complex*>(_H.data()),
            FFTW_FORWARD, FFTW_MEASURE);
        
        _ifft_corr = fftwf_plan_dft_1d(
            static_cast<int>(_fft_len),
            reinterpret_cast<fftwf_complex*>(_X.data()),
            reinterpret_cast<fftwf_complex*>(_corr_result.data()),
            FFTW_BACKWARD, FFTW_MEASURE);
        
        // Pre-compute FFT of _h_padded
        fftwf_execute(_fft_h);
    }

    ~SyncProcessor() {
        if (_fft_x) fftwf_destroy_plan(_fft_x);
        if (_fft_h) fftwf_destroy_plan(_fft_h);
        if (_ifft_corr) fftwf_destroy_plan(_ifft_corr);
    }

    // Non-copyable due to FFTW plans
    SyncProcessor(const SyncProcessor&) = delete;
    SyncProcessor& operator=(const SyncProcessor&) = delete;

    // Move constructible
    SyncProcessor(SyncProcessor&& other) noexcept
        : _symbol_len(other._symbol_len),
          _data_len(other._data_len),
          _fft_len(other._fft_len),
          _fft_size(other._fft_size),
          _cp_length(other._cp_length),
          _fft_x(other._fft_x),
          _fft_h(other._fft_h),
          _ifft_corr(other._ifft_corr),
          _x_padded(std::move(other._x_padded)),
          _h_padded(std::move(other._h_padded)),
          _tx_sync_symbol(std::move(other._tx_sync_symbol)),
          _X(std::move(other._X)),
          _H(std::move(other._H)),
          _corr_result(std::move(other._corr_result))
    {
        other._fft_x = nullptr;
        other._fft_h = nullptr;
        other._ifft_corr = nullptr;
    }

    /**
     * @brief FFT-based sliding window correlation for sync detection.
     * Uses pre-allocated FFT plans for fast correlation computation.
     * 
     * @param sync_data Received data (2 frames worth)
     * @param max_pos Output: position of maximum correlation
     * @param max_corr Output: maximum correlation value
     * @param avg_corr Output: average correlation value
     */
    void find_sync_position(
        const AlignedVector& sync_data,
        int& max_pos,
        float& max_corr,
        float& avg_corr
    ) {
        find_sync_position_in_range(
            sync_data,
            0,
            sync_data.size() >= _symbol_len ? (sync_data.size() - _symbol_len) : 0,
            max_pos,
            max_corr,
            avg_corr);
    }

    void find_sync_position_in_range(
        const AlignedVector& sync_data,
        size_t search_start,
        size_t search_end,
        int& max_pos,
        float& max_corr,
        float& avg_corr
    ) {
        const size_t n_windows = sync_data.size() >= _symbol_len
            ? (sync_data.size() - _symbol_len + 1)
            : 0;
        if (n_windows == 0) {
            max_pos = 0;
            max_corr = 0.0f;
            avg_corr = 0.0f;
            return;
        }

        run_correlation(sync_data);

        const size_t begin = std::min(search_start, n_windows - 1);
        const size_t end = std::min(search_end, n_windows - 1);
        if (begin > end) {
            max_pos = 0;
            max_corr = 0.0f;
            avg_corr = 0.0f;
            return;
        }

        max_corr = 0.0f;
        avg_corr = 0.0f;
        max_pos = static_cast<int>(begin);
        const size_t sample_count = end - begin + 1;

        for (size_t i = begin; i <= end; ++i) {
            const float corr = std::norm(_corr_result[i + _symbol_len - 1]);
            if (corr > max_corr) {
                max_corr = corr;
                max_pos = static_cast<int>(i);
            }
            avg_corr += corr;
        }
        avg_corr /= static_cast<float>(sample_count);
    }

    static SecSyncCoarseSyncResult detect_sec_sync_symbol(
        const AlignedVector& data,
        size_t fft_size,
        size_t cp_length,
        double sample_rate
    ) {
        SecSyncCoarseSyncResult result;
        if (fft_size == 0 || sample_rate <= 0.0) {
            return result;
        }
        const size_t symbol_len = fft_size + cp_length;
        if (data.size() < 2 * symbol_len) {
            return result;
        }

        const size_t max_symbol_start = data.size() - 2 * symbol_len;
        std::complex<double> p_sum(0.0, 0.0);
        double r_sum = 0.0;
        const size_t useful_start0 = cp_length;
        const size_t next_useful_start0 = symbol_len + cp_length;
        for (size_t i = 0; i < fft_size; ++i) {
            const auto& a = data[useful_start0 + i];
            const auto& b = data[next_useful_start0 + i];
            p_sum += std::conj(std::complex<double>(a.real(), a.imag())) *
                     std::complex<double>(b.real(), b.imag());
            r_sum += std::norm(b);
        }

        std::complex<double> best_p = p_sum;
        double best_metric = 0.0;
        size_t best_start = 0;
        for (size_t symbol_start = 0; symbol_start <= max_symbol_start; ++symbol_start) {
            const double denom = std::max(r_sum * r_sum, 1e-12);
            const double metric = std::norm(p_sum) / denom;
            if (metric > best_metric) {
                best_metric = metric;
                best_start = symbol_start;
                best_p = p_sum;
            }
            if (symbol_start == max_symbol_start) {
                break;
            }

            const size_t old_useful = symbol_start + cp_length;
            const size_t old_useful_next = old_useful + symbol_len;
            const size_t new_useful = old_useful + fft_size;
            const size_t new_useful_next = old_useful_next + fft_size;
            const auto& old_a = data[old_useful];
            const auto& old_b = data[old_useful_next];
            const auto& new_a = data[new_useful];
            const auto& new_b = data[new_useful_next];
            p_sum -= std::conj(std::complex<double>(old_a.real(), old_a.imag())) *
                     std::complex<double>(old_b.real(), old_b.imag());
            p_sum += std::conj(std::complex<double>(new_a.real(), new_a.imag())) *
                     std::complex<double>(new_b.real(), new_b.imag());
            r_sum -= std::norm(old_b);
            r_sum += std::norm(new_b);
        }

        result.coarse_symbol_start = static_cast<int>(best_start);
        result.coarse_useful_start = static_cast<int>(best_start + cp_length);
        result.max_metric = static_cast<float>(best_metric);
        result.coarse_cfo_hz =
            std::arg(best_p) * sample_rate / (2.0 * M_PI * static_cast<double>(symbol_len));
        result.valid = std::isfinite(result.coarse_cfo_hz) && std::isfinite(best_metric);
        return result;
    }

    static void apply_cfo_compensation(
        const AlignedVector& input,
        double sample_rate,
        double cfo_hz,
        AlignedVector& output
    ) {
        output.resize(input.size());
        if (input.empty()) {
            return;
        }
        if (!(sample_rate > 0.0) || !std::isfinite(cfo_hz) || std::abs(cfo_hz) < 1e-12) {
            std::copy(input.begin(), input.end(), output.begin());
            return;
        }

        const double phase_step = -2.0 * M_PI * cfo_hz / sample_rate;
        for (size_t i = 0; i < input.size(); ++i) {
            const float phase = static_cast<float>(phase_step * static_cast<double>(i));
            output[i] = input[i] * std::polar(1.0f, phase);
        }
    }

    static CfoTrainingEstimate estimate_cfo_from_training_symbol(
        const AlignedVector& data,
        size_t cfo_symbol_start,
        size_t fft_size,
        size_t cp_length,
        size_t period_samples,
        double sample_rate)
    {
        CfoTrainingEstimate result;
        if (fft_size == 0 || period_samples == 0 || period_samples >= fft_size ||
            cp_length + fft_size == 0 || !(sample_rate > 0.0)) {
            return result;
        }
        const size_t symbol_len = fft_size + cp_length;
        if (cfo_symbol_start > data.size() ||
            data.size() - cfo_symbol_start < symbol_len ||
            fft_size <= period_samples) {
            return result;
        }

        const size_t useful_start = cfo_symbol_start + cp_length;
        const size_t pair_count = fft_size - period_samples;
        std::complex<double> p_sum(0.0, 0.0);
        double r0_sum = 0.0;
        double r1_sum = 0.0;
        for (size_t i = 0; i < pair_count; ++i) {
            const auto& a = data[useful_start + i];
            const auto& b = data[useful_start + i + period_samples];
            p_sum += std::conj(std::complex<double>(a.real(), a.imag())) *
                     std::complex<double>(b.real(), b.imag());
            r0_sum += std::norm(a);
            r1_sum += std::norm(b);
        }

        const double denom = std::max(r0_sum * r1_sum, 1e-12);
        result.metric = static_cast<float>(std::norm(p_sum) / denom);
        result.cfo_hz =
            std::arg(p_sum) * sample_rate /
            (2.0 * M_PI * static_cast<double>(period_samples));
        result.valid = std::isfinite(result.cfo_hz) &&
                       std::isfinite(result.metric) &&
                       r0_sum > 0.0 &&
                       r1_sum > 0.0;
        return result;
    }

    static bool select_alias_closest_to_cfo_reference(
        SecSyncRefineResult& result,
        double reference_cfo_hz)
    {
        if (!std::isfinite(reference_cfo_hz) || result.alias_candidates.empty()) {
            return false;
        }

        const SecSyncRefineResult::AliasCandidate* best_candidate = nullptr;
        double best_error = std::numeric_limits<double>::infinity();
        for (const auto& candidate : result.alias_candidates) {
            if (!candidate.valid || !std::isfinite(candidate.cfo_hz)) {
                continue;
            }
            const double err = std::abs(candidate.cfo_hz - reference_cfo_hz);
            if (err < best_error) {
                best_error = err;
                best_candidate = &candidate;
            }
        }

        if (best_candidate == nullptr) {
            return false;
        }

        result.max_pos = best_candidate->max_pos;
        result.max_corr = best_candidate->max_corr;
        result.avg_corr = best_candidate->avg_corr;
        result.cfo_hz = best_candidate->cfo_hz;
        result.alias_index = best_candidate->alias_index;
        result.valid = true;
        return true;
    }

    SecSyncRefineResult refine_sync_cfo_alias(
        const AlignedVector& sync_data,
        double modulo_cfo_hz,
        double alias_period_hz,
        double sample_rate,
        size_t search_start,
        size_t search_end,
        int max_alias_index,
        AlignedVector& cfo_compensated_buffer,
        bool collect_alias_candidates = false
    ) const {
        SecSyncRefineResult best;
        if (!std::isfinite(modulo_cfo_hz) || !(alias_period_hz > 0.0) ||
            !(sample_rate > 0.0) || _symbol_len == 0) {
            return best;
        }

        const int alias_span = std::max(0, max_alias_index);
        bool have_candidate = false;
        for (int alias = -alias_span; alias <= alias_span; ++alias) {
            const double candidate_cfo_hz =
                modulo_cfo_hz + static_cast<double>(alias) * alias_period_hz;
            apply_cfo_compensation(
                sync_data,
                sample_rate,
                candidate_cfo_hz,
                cfo_compensated_buffer);

            int trial_pos = 0;
            float trial_max_corr = 0.0f;
            float trial_avg_corr = 0.0f;
            find_sync_position_in_range_direct(
                cfo_compensated_buffer,
                search_start,
                search_end,
                trial_pos,
                trial_max_corr,
                trial_avg_corr);
            SecSyncRefineResult::AliasCandidate candidate;
            candidate.alias_index = alias;
            candidate.max_pos = trial_pos;
            candidate.max_corr = trial_max_corr;
            candidate.avg_corr = trial_avg_corr;
            candidate.cfo_hz = candidate_cfo_hz;
            candidate.valid = std::isfinite(trial_max_corr) &&
                              std::isfinite(trial_avg_corr) &&
                              trial_avg_corr > 0.0f;
            if (collect_alias_candidates) {
                best.alias_candidates.push_back(candidate);
            }
            if (!candidate.valid) {
                continue;
            }
            if (!have_candidate || trial_max_corr > best.max_corr) {
                best.max_pos = trial_pos;
                best.max_corr = trial_max_corr;
                best.avg_corr = trial_avg_corr;
                best.cfo_hz = candidate_cfo_hz;
                best.alias_index = alias;
                best.valid = true;
                have_candidate = true;
            }
        }
        return best;
    }

    SecSyncRefineResult refine_sec_sync_with_alias_search(
        const AlignedVector& sync_data,
        const SecSyncCoarseSyncResult& coarse,
        double sample_rate,
        size_t search_start,
        size_t search_end,
        int max_alias_index,
        AlignedVector& cfo_compensated_buffer,
        bool collect_alias_candidates = false
    ) const {
        if (!coarse.valid || _symbol_len == 0) {
            return SecSyncRefineResult{};
        }
        return refine_sync_cfo_alias(
            sync_data,
            coarse.coarse_cfo_hz,
            sample_rate / static_cast<double>(_symbol_len),
            sample_rate,
            search_start,
            search_end,
            max_alias_index,
            cfo_compensated_buffer,
            collect_alias_candidates);
    }

    /**
     * @brief Estimate coarse CFO using CP correlation.
     * Correlates CP samples with corresponding tail samples.
     * 
     * @param data Input time-domain samples
     * @param start_pos Starting position in the buffer
     * @param available_symbols Number of symbols to process
     * @param symbol_len Symbol length (fft_size + cp_length)
     * @param cp_length Cyclic prefix length
     * @param fft_size FFT size
     * @return Phase difference (in radians)
     */
    static double estimate_cfo_phase(
        const AlignedVector& data,
        size_t start_pos,
        size_t available_symbols,
        size_t symbol_len,
        size_t cp_length,
        size_t fft_size
    ) {
        double total_real = 0.0;
        double total_imag = 0.0;
        
        for (size_t sym = 0; sym < available_symbols; ++sym) {
            const size_t pos = start_pos + sym * symbol_len;
            
            double sym_real = 0.0;
            double sym_imag = 0.0;
            
            #pragma omp simd reduction(+:sym_real, sym_imag)
            for (size_t i = 0; i < cp_length; ++i) {
                const auto& cp_sample = data[pos + i];
                const auto& tail_sample = data[pos + i + fft_size];
                
                // conj(cp_sample) * tail_sample
                sym_real += cp_sample.real() * tail_sample.real() + 
                            cp_sample.imag() * tail_sample.imag();
                sym_imag += cp_sample.real() * tail_sample.imag() - 
                            cp_sample.imag() * tail_sample.real();
            }
            
            total_real += sym_real;
            total_imag += sym_imag;
        }
        
        return std::atan2(total_imag, total_real);
    }

    /**
     * @brief Convert phase difference to CFO in Hz.
     */
    static double phase_to_cfo(double phase_diff, double sample_rate, size_t fft_size) {
        const double T_symbol = static_cast<double>(fft_size) / sample_rate;
        return phase_diff / (2.0 * M_PI * T_symbol);
    }

private:
    void find_sync_position_in_range_direct(
        const AlignedVector& sync_data,
        size_t search_start,
        size_t search_end,
        int& max_pos,
        float& max_corr,
        float& avg_corr
    ) const {
        const size_t n_windows = sync_data.size() >= _symbol_len
            ? (sync_data.size() - _symbol_len + 1)
            : 0;
        if (n_windows == 0 || _tx_sync_symbol.size() != _symbol_len) {
            max_pos = 0;
            max_corr = 0.0f;
            avg_corr = 0.0f;
            return;
        }

        const size_t begin = std::min(search_start, n_windows - 1);
        const size_t end = std::min(search_end, n_windows - 1);
        if (begin > end) {
            max_pos = 0;
            max_corr = 0.0f;
            avg_corr = 0.0f;
            return;
        }

        max_corr = 0.0f;
        avg_corr = 0.0f;
        max_pos = static_cast<int>(begin);
        const size_t sample_count = end - begin + 1;

        for (size_t pos = begin; pos <= end; ++pos) {
            double sum_real = 0.0;
            double sum_imag = 0.0;
            for (size_t i = 0; i < _symbol_len; ++i) {
                const auto& x = sync_data[pos + i];
                const auto& s = _tx_sync_symbol[i];
                sum_real += x.real() * s.real() + x.imag() * s.imag();
                sum_imag += x.imag() * s.real() - x.real() * s.imag();
            }
            const float corr = static_cast<float>(sum_real * sum_real + sum_imag * sum_imag);
            if (corr > max_corr) {
                max_corr = corr;
                max_pos = static_cast<int>(pos);
            }
            avg_corr += corr;
        }
        avg_corr /= static_cast<float>(sample_count);
    }

    void run_correlation(const AlignedVector& sync_data) {
        std::fill(_x_padded.begin(), _x_padded.end(), std::complex<float>(0.0f, 0.0f));
        std::copy(sync_data.begin(), sync_data.end(), _x_padded.begin());

        fftwf_execute(_fft_x);

        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _fft_len; ++i) {
            _X[i] *= _H[i];
        }

        fftwf_execute(_ifft_corr);

        const float norm_factor = 1.0f / static_cast<float>(_fft_len);
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _fft_len; ++i) {
            _corr_result[i] *= norm_factor;
        }
    }

    /**
     * @brief Generate sync sequence and prepare _h_padded for correlation.
     * Uses provided ZC sequence in frequency domain.
     */
    void prepare_sync_sequence(AlignedVector zc_freq) {
        const int N = static_cast<int>(_fft_size);
        
        // Execute IFFT to get time-domain sync symbol
        AlignedVector ifft_out(N);
        fftwf_plan plan = fftwf_plan_dft_1d(
            N,
            reinterpret_cast<fftwf_complex*>(zc_freq.data()),
            reinterpret_cast<fftwf_complex*>(ifft_out.data()),
            FFTW_BACKWARD, FFTW_ESTIMATE
        );
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        
        // Add cyclic prefix to create tx_sync_symbol
        _tx_sync_symbol.resize(_symbol_len);
        if (_cp_length > 0) {
            std::copy(ifft_out.end() - _cp_length, ifft_out.end(), _tx_sync_symbol.begin());
        }
        std::copy(ifft_out.begin(), ifft_out.end(), _tx_sync_symbol.begin() + _cp_length);
        
        // Prepare reversed and conjugated sync sequence in _h_padded
        // corr[i] = sum_j rx[i+j]*conj(sync[j]) = conv(rx, conj(sync_reversed))
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _symbol_len; ++i) {
            _h_padded[i] = std::conj(_tx_sync_symbol[_symbol_len - 1 - i]);
        }
    }

    size_t _symbol_len;
    size_t _data_len;
    size_t _fft_len;
    size_t _fft_size;
    size_t _cp_length;
    
    // FFT plans
    fftwf_plan _fft_x = nullptr;
    fftwf_plan _fft_h = nullptr;
    fftwf_plan _ifft_corr = nullptr;
    
    // Work buffers
    AlignedVector _x_padded;
    AlignedVector _h_padded;  // Pre-computed: reversed + conjugated sync sequence
    AlignedVector _tx_sync_symbol;
    AlignedVector _X;
    AlignedVector _H;         // Pre-computed: FFT of _h_padded
    AlignedVector _corr_result;
};


/**
 * @brief QPSK LLR Processor.
 * 
 * Provides noise variance estimation and LLR calculation for QPSK modulation.
 */
class QPSK_LLR {
public:
    /**
     * @brief Estimate noise variance from pilot errors.
     * 
     * @param symbols Equalized symbols
     * @param pilot_positions Indices of pilot subcarriers
     * @param zc_freq Known ZC sequence in frequency domain
     * @return Estimated noise variance
     */
    static double estimate_noise_variance(
        const std::vector<AlignedVector>& symbols,
        const std::vector<size_t>& pilot_positions,
        const AlignedVector& zc_freq
    ) {
        double err_power_acc = 0.0;
        size_t err_count = 0;
        
        for (const auto& sym : symbols) {
            for (auto p : pilot_positions) {
                if (p < sym.size()) {
                    std::complex<float> y_eq = sym[p];
                    std::complex<float> x_ref = zc_freq[p];
                    auto e = y_eq - x_ref;
                    err_power_acc += std::norm(e);
                    err_count++;
                }
            }
        }
        
        if (err_count > 8) {
            double noise_var = err_power_acc / err_count;
            return std::max(noise_var, 1e-6);
        }
        return 0.5;  // Default value
    }

    /**
     * @brief Calculate LLR scaling factor from noise variance.
     * For QPSK Gray mapping: LLR_scale = 4 / noise_var
     */
    static double compute_llr_scale(double noise_var) {
        double sigma2_dim = noise_var / 2.0;
        return 2.0 / sigma2_dim;  // = 4 / noise_var
    }

    /**
     * @brief Compute LLR values for QPSK symbols.
     */
    static void compute_llr(
        const std::vector<AlignedVector>& symbols,
        const std::vector<size_t>& data_indices,
        float llr_scale,
        AlignedFloatVector& llr_output
    ) {
        const size_t num_data_sc = data_indices.size();
        const size_t total_llr = symbols.size() * num_data_sc * 2;
        llr_output.resize(total_llr);
        
        float* __restrict__ llr_ptr = llr_output.data();
        
        for (size_t sym_idx = 0; sym_idx < symbols.size(); ++sym_idx) {
            const auto* __restrict__ sym_ptr = symbols[sym_idx].data();
            float* __restrict__ out_ptr = llr_ptr + sym_idx * num_data_sc * 2;
            
            // No omp simd: sym_ptr[data_indices[i]] is a gather; scalar matches it.
            for (size_t i = 0; i < num_data_sc; ++i) {
                const size_t k = data_indices[i];
                out_ptr[i * 2]     = sym_ptr[k].real() * llr_scale;
                out_ptr[i * 2 + 1] = sym_ptr[k].imag() * llr_scale;
            }
        }
    }

    /**
     * @brief Compute LLR values directly to pre-allocated output buffer.
     */
    static void compute_llr_to_buffer(
        const std::vector<AlignedVector>& symbols,
        const std::vector<size_t>& data_indices,
        float llr_scale,
        float* __restrict__ llr_output
    ) {
        const size_t num_data_sc = data_indices.size();
        
        for (size_t sym_idx = 0; sym_idx < symbols.size(); ++sym_idx) {
            const auto* __restrict__ sym_ptr = symbols[sym_idx].data();
            float* __restrict__ out_ptr = llr_output + sym_idx * num_data_sc * 2;
            
            // No omp simd: sym_ptr[data_indices[i]] is a gather; scalar matches it.
            for (size_t i = 0; i < num_data_sc; ++i) {
                const size_t k = data_indices[i];
                out_ptr[i * 2]     = sym_ptr[k].real() * llr_scale;
                out_ptr[i * 2 + 1] = sym_ptr[k].imag() * llr_scale;
            }
        }
    }
};


/**
 * @brief Frequency Offset Estimator.
 * 
 * Estimates CFO and SFO from pilot phase differences.
 */
class FrequencyOffsetEstimator {
public:
    /**
     * @brief Estimate phase difference between consecutive symbols at each pilot.
     * Returns (beta, alpha) where:
     * - beta: phase slope vs subcarrier (related to SFO)
     * - alpha: constant phase offset (related to CFO)
     * 
     * @param symbols Equalized frequency domain symbols
     * @param pilot_positions Indices of pilot subcarriers
     * @param fft_size FFT size
     * @param sync_pos Sync symbol position (to skip)
     * @param pilot_indices Output: actual frequency indices of pilots
     * @param avg_phase_diff Output: average phase difference at each pilot
     * @param weights Output: weights for regression
     * @param actual_symbol_indices Optional actual frame symbol index for each
     *        entry in symbols. When present, only physically adjacent symbol
     *        pairs are accumulated.
     * @param skip_actual_symbol_mask Optional per-actual-symbol mask. When
     *        present, pairs touching masked symbols are ignored.
     * @return true if at least one valid symbol pair and pilot were used.
     */
    static bool compute_pilot_phase_diff(
        const std::vector<AlignedVector>& symbols,
        const std::vector<size_t>& pilot_positions,
        size_t fft_size,
        size_t sync_pos,
        std::vector<int>& pilot_indices,
        std::vector<float>& avg_phase_diff,
        std::vector<float>& weights,
        const std::vector<int>* actual_symbol_indices = nullptr,
        const std::vector<uint8_t>* skip_actual_symbol_mask = nullptr
    ) {
        const size_t num_pilots = pilot_positions.size();
        pilot_indices.resize(num_pilots);
        avg_phase_diff.resize(num_pilots);
        weights.assign(num_pilots, 0.0f);

        if (symbols.empty() || num_pilots == 0) {
            return false;
        }
        if (actual_symbol_indices != nullptr &&
            actual_symbol_indices->size() != symbols.size()) {
            return false;
        }

        const bool use_actual_symbols = (actual_symbol_indices != nullptr);
        const size_t first_pair =
            use_actual_symbols ? 0 : std::min(sync_pos, symbols.size());
        auto pair_allowed = [&](size_t current_idx, size_t next_idx) {
            if (!use_actual_symbols) {
                return true;
            }
            const int current_actual = (*actual_symbol_indices)[current_idx];
            const int next_actual = (*actual_symbol_indices)[next_idx];
            if (current_actual < 0 || next_actual != current_actual + 1) {
                return false;
            }
            if (skip_actual_symbol_mask == nullptr) {
                return true;
            }
            const size_t current_sym = static_cast<size_t>(current_actual);
            const size_t next_sym = static_cast<size_t>(next_actual);
            if (current_sym >= skip_actual_symbol_mask->size() ||
                next_sym >= skip_actual_symbol_mask->size()) {
                return false;
            }
            return (*skip_actual_symbol_mask)[current_sym] == 0 &&
                   (*skip_actual_symbol_mask)[next_sym] == 0;
        };

        size_t pair_count = 0;
        for (size_t i = first_pair; i + 1 < symbols.size(); ++i) {
            if (pair_allowed(i, i + 1)) {
                ++pair_count;
            }
        }
        if (pair_count == 0) {
            return false;
        }

        bool used_pilot = false;
        
        for (size_t j = 0; j < num_pilots; ++j) {
            auto pilot_index = pilot_positions[j];
            std::complex<double> next_current_sum(0.0, 0.0);
            if (pilot_index >= fft_size) {
                pilot_indices[j] = 0;
                avg_phase_diff[j] = 0.0f;
                continue;
            }
            used_pilot = true;

            // NOTE: no omp simd here — the body reduces into a complex<double>
            // accumulator via conj()*mul, which the vectorizer cannot lane-split
            // anyway. The annotation was a no-op and is intentionally omitted.
            for (size_t i = first_pair; i + 1 < symbols.size(); ++i) {
                if (!pair_allowed(i, i + 1)) {
                    continue;
                }
                std::complex<float> current_pilot = symbols[i][pilot_index];
                std::complex<float> next_pilot = symbols[i+1][pilot_index];
                next_current_sum += std::conj(current_pilot) * next_pilot;
            }
            
            avg_phase_diff[j] = static_cast<float>(std::arg(next_current_sum));
            
            // Convert to actual frequency index (-fft_size/2 to fft_size/2-1)
            int freq_index = static_cast<int>(pilot_index);
            if (freq_index >= static_cast<int>(fft_size)/2) {
                freq_index -= static_cast<int>(fft_size);
            }
            pilot_indices[j] = freq_index;
            weights[j] = static_cast<float>(std::norm(next_current_sum));
        }
        return used_pilot;
    }

    /**
     * @brief Convert alpha (phase per symbol) to CFO in Hz.
     */
    static float alpha_to_cfo(float alpha, size_t fft_size, size_t cp_length, double sample_rate) {
        const float T = static_cast<float>(fft_size + cp_length) / static_cast<float>(sample_rate);
        return alpha / (2.0f * static_cast<float>(M_PI) * T);
    }
};

/**
 * @brief Adaptive Kalman filter for CFO error (ppm) tracking.
 *
 * State model:
 * x = [error_ppm, drift_ppm_per_frame]^T
 * x(k+1) = F x(k) + Gw*w(k) + Grw*r(k)
 * z(k) = H x(k) + v(k), where z is raw error_ppm from CFO estimate
 *
 * The filter adapts q_wf/q_rw/R online using innovation ACF and a regularized
 * least-squares update on the linear model r_nu = Phi * theta.
 */
class AdaptiveCFOAKF {
public:
    struct Params {
        bool enable = true;
        size_t bootstrap_frames = 64;
        size_t innovation_window = 64;
        size_t max_lag = 4;
        size_t adapt_interval = 64;
        double gate_sigma = 3.0;
        double tikhonov_lambda = 1e-3;
        double update_smooth = 0.2;
        double q_wf_min = 1e-10;
        double q_wf_max = 1e2;
        double q_rw_min = 1e-12;
        double q_rw_max = 1e1;
        double r_min = 1e-8;
        double r_max = 1e3;
    };

    struct UpdateResult {
        double raw_error_ppm = 0.0;
        double filtered_error_ppm = 0.0;
        double innovation = 0.0;
        double innovation_variance = 1.0;
        double q_wf = 0.0;
        double q_rw = 0.0;
        double r = 1.0;
        bool initialized = false;
        bool gated = false;
        bool adapted = false;
    };

    explicit AdaptiveCFOAKF(const Params& params, double frame_duration_s = 1.0)
        : params_(sanitize_params(params)),
          dt_s_(std::max(frame_duration_s, 1e-9)) {
        reset();
    }

    void configure(const Params& params, double frame_duration_s) {
        params_ = sanitize_params(params);
        dt_s_ = std::max(frame_duration_s, 1e-9);
        reset();
    }

    // Inject externally applied OCXO control action (delta ppm).
    void notify_control_action(double delta_ppm) {
        if (!params_.enable || !std::isfinite(delta_ppm)) {
            return;
        }
        if (std::abs(delta_ppm) <= kControlDeltaEpsPpm) {
            return;
        }
        pending_control_delta_ppm_ += delta_ppm;
        adapt_freeze_countdown_ = std::max(adapt_freeze_countdown_, kAdaptFreezeFramesAfterControl);
    }

    void reset() {
        initialized_ = false;
        bootstrap_samples_.clear();
        innovations_.clear();
        frame_counter_ = 0;
        accepted_update_count_ = 0;
        pending_control_delta_ppm_ = 0.0;
        adapt_freeze_countdown_ = 0;

        x0_ = 0.0;
        x1_ = 0.0;
        p00_ = 1.0;
        p01_ = 0.0;
        p10_ = 0.0;
        p11_ = 1.0;

        q_wf_ = clamp(1e-4, params_.q_wf_min, params_.q_wf_max);
        q_rw_ = clamp(1e-5, params_.q_rw_min, params_.q_rw_max);
        r_ = clamp(1e-2, params_.r_min, params_.r_max);

        last_innovation_ = 0.0;
        last_s_ = std::max(r_, 1e-9);
        last_k0_ = 0.0;
        last_k1_ = 0.0;
        has_valid_gain_ = false;
    }

    UpdateResult update(double raw_error_ppm) {
        UpdateResult out;
        out.raw_error_ppm = std::isfinite(raw_error_ppm) ? raw_error_ppm : 0.0;
        out.q_wf = q_wf_;
        out.q_rw = q_rw_;
        out.r = r_;

        if (!params_.enable) {
            pending_control_delta_ppm_ = 0.0;
            adapt_freeze_countdown_ = 0;
            out.filtered_error_ppm = out.raw_error_ppm;
            out.innovation = 0.0;
            out.innovation_variance = 1.0;
            out.initialized = false;
            return out;
        }

        if (!initialized_) {
            bootstrap_samples_.push_back(out.raw_error_ppm);
            if (bootstrap_samples_.size() >= params_.bootstrap_frames) {
                initialize_from_bootstrap();
            } else if (bootstrap_samples_.size() > params_.bootstrap_frames) {
                bootstrap_samples_.pop_front();
            }
            out.filtered_error_ppm = initialized_ ? x0_ : bootstrap_filtered_error();
            out.innovation = 0.0;
            out.innovation_variance = std::max(r_, 1e-9);
            out.initialized = initialized_;
            out.q_wf = q_wf_;
            out.q_rw = q_rw_;
            out.r = r_;
            return out;
        }

        frame_counter_++;
        const bool freeze_active = (adapt_freeze_countdown_ > 0);
        if (adapt_freeze_countdown_ > 0) {
            --adapt_freeze_countdown_;
        }

        const double control_delta_ppm = pending_control_delta_ppm_;
        pending_control_delta_ppm_ = 0.0;
        const double control_comp_ppm = -kControlToErrorGain * control_delta_ppm;

        // Keep covariance symmetric in floating-point math.
        const double p01_sym = 0.5 * (p01_ + p10_);
        p01_ = p01_sym;
        p10_ = p01_sym;

        const double x0_pred = x0_ + dt_s_ * x1_ + control_comp_ppm;
        const double x1_pred = x1_;

        const double p00_pred = p00_ + dt_s_ * (p10_ + p01_) + dt_s_ * dt_s_ * p11_ + q_wf_;
        const double p01_pred = p01_ + dt_s_ * p11_;
        const double p10_pred = p10_ + dt_s_ * p11_;
        const double p11_pred = p11_ + q_rw_;

        const double s = std::max(p00_pred + r_, 1e-9);
        const double innovation = out.raw_error_ppm - x0_pred;
        const double gate_sigma_sq = params_.gate_sigma * params_.gate_sigma;
        const double nis = innovation * innovation / s;
        const bool gated = nis > gate_sigma_sq;

        out.innovation = innovation;
        out.innovation_variance = s;
        out.gated = gated;

        if (gated) {
            x0_ = x0_pred;
            x1_ = x1_pred;
            p00_ = p00_pred;
            p01_ = p01_pred;
            p10_ = p10_pred;
            p11_ = p11_pred;
        } else {
            const double k0 = p00_pred / s;
            const double k1 = p10_pred / s;

            x0_ = x0_pred + k0 * innovation;
            x1_ = x1_pred + k1 * innovation;

            // Joseph-form covariance update for numerical robustness.
            const double a = 1.0 - k0;
            const double b = -k1;

            const double m00 = a * p00_pred;
            const double m01 = a * p01_pred;
            const double m10 = b * p00_pred + p10_pred;
            const double m11 = b * p01_pred + p11_pred;

            p00_ = m00 * a + k0 * k0 * r_;
            p01_ = m00 * b + m01 + k0 * k1 * r_;
            p10_ = m10 * a + k1 * k0 * r_;
            p11_ = m10 * b + m11 + k1 * k1 * r_;

            const double p_off = 0.5 * (p01_ + p10_);
            p01_ = p_off;
            p10_ = p_off;

            last_k0_ = k0;
            last_k1_ = k1;
            has_valid_gain_ = true;
            accepted_update_count_++;

            if (!freeze_active) {
                innovations_.push_back(innovation);
                while (innovations_.size() > params_.innovation_window) {
                    innovations_.pop_front();
                }
            }
        }

        bool adapted = false;
        if (!freeze_active &&
            has_valid_gain_ &&
            (frame_counter_ % params_.adapt_interval == 0) &&
            innovations_.size() >= std::max(params_.innovation_window, params_.max_lag + size_t(2))) {
            adapted = adapt_qr_from_innovation_acf();
        }

        out.filtered_error_ppm = x0_;
        out.q_wf = q_wf_;
        out.q_rw = q_rw_;
        out.r = r_;
        out.initialized = initialized_;
        out.adapted = adapted;
        last_innovation_ = innovation;
        last_s_ = s;
        return out;
    }

    double filtered_error_ppm() const { return x0_; }
    double q_wf() const { return q_wf_; }
    double q_rw() const { return q_rw_; }
    double r() const { return r_; }
    double innovation() const { return last_innovation_; }
    double innovation_variance() const { return last_s_; }
    bool initialized() const { return initialized_; }

private:
    Params params_{};
    double dt_s_ = 1.0;
    bool initialized_ = false;
    bool has_valid_gain_ = false;

    std::deque<double> bootstrap_samples_;
    std::deque<double> innovations_;
    size_t frame_counter_ = 0;
    size_t accepted_update_count_ = 0;
    double pending_control_delta_ppm_ = 0.0;
    size_t adapt_freeze_countdown_ = 0;

    // State x=[error_ppm, drift_ppm_per_frame]
    double x0_ = 0.0;
    double x1_ = 0.0;

    // Covariance P
    double p00_ = 1.0;
    double p01_ = 0.0;
    double p10_ = 0.0;
    double p11_ = 1.0;

    // Adaptive noise coefficients
    double q_wf_ = 1e-4;
    double q_rw_ = 1e-5;
    double r_ = 1e-2;

    double last_innovation_ = 0.0;
    double last_s_ = 1.0;
    double last_k0_ = 0.0;
    double last_k1_ = 0.0;

    static constexpr double kControlToErrorGain = 1.0;
    static constexpr double kControlDeltaEpsPpm = 1e-12;
    static constexpr size_t kAdaptFreezeFramesAfterControl = 16;

    static double clamp(double v, double lo, double hi) {
        return std::max(lo, std::min(hi, v));
    }

    static Params sanitize_params(const Params& in) {
        Params p = in;
        p.bootstrap_frames = std::max<size_t>(p.bootstrap_frames, 8);
        p.innovation_window = std::max<size_t>(p.innovation_window, 8);
        p.max_lag = std::max<size_t>(p.max_lag, 2);
        p.max_lag = std::min(p.max_lag, p.innovation_window - 1);
        p.adapt_interval = std::max<size_t>(p.adapt_interval, 1);
        p.gate_sigma = std::max(p.gate_sigma, 1.0);
        p.tikhonov_lambda = std::max(p.tikhonov_lambda, 0.0);
        p.update_smooth = clamp(p.update_smooth, 0.0, 1.0);
        if (p.q_wf_min > p.q_wf_max) std::swap(p.q_wf_min, p.q_wf_max);
        if (p.q_rw_min > p.q_rw_max) std::swap(p.q_rw_min, p.q_rw_max);
        if (p.r_min > p.r_max) std::swap(p.r_min, p.r_max);
        p.q_wf_min = std::max(p.q_wf_min, 0.0);
        p.q_rw_min = std::max(p.q_rw_min, 0.0);
        p.r_min = std::max(p.r_min, 1e-12);
        return p;
    }

    double bootstrap_filtered_error() const {
        if (bootstrap_samples_.empty()) return 0.0;
        const double sum = std::accumulate(bootstrap_samples_.begin(), bootstrap_samples_.end(), 0.0);
        return sum / static_cast<double>(bootstrap_samples_.size());
    }

    static double median_of_sorted(const std::vector<double>& sorted) {
        if (sorted.empty()) return 0.0;
        const size_t n = sorted.size();
        if ((n & 1U) != 0U) return sorted[n / 2];
        return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    }

    static double robust_variance(const std::deque<double>& samples) {
        if (samples.empty()) return 1e-2;
        std::vector<double> v(samples.begin(), samples.end());
        std::sort(v.begin(), v.end());
        const double med = median_of_sorted(v);
        std::vector<double> abs_dev(v.size(), 0.0);
        for (size_t i = 0; i < v.size(); ++i) {
            abs_dev[i] = std::abs(v[i] - med);
        }
        std::sort(abs_dev.begin(), abs_dev.end());
        const double mad = median_of_sorted(abs_dev);
        double sigma = 1.4826 * mad;
        if (!(sigma > 1e-8) || !std::isfinite(sigma)) {
            const double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
            double var = 0.0;
            for (double x : v) {
                const double d = x - mean;
                var += d * d;
            }
            var /= static_cast<double>(std::max<size_t>(1, v.size() - 1));
            sigma = std::sqrt(std::max(var, 1e-8));
        }
        return std::max(sigma * sigma, 1e-8);
    }

    void initialize_from_bootstrap() {
        std::vector<double> sorted(bootstrap_samples_.begin(), bootstrap_samples_.end());
        std::sort(sorted.begin(), sorted.end());
        const double med = median_of_sorted(sorted);
        const double var = robust_variance(bootstrap_samples_);

        x0_ = med;
        x1_ = 0.0;
        p00_ = std::max(var, 1e-6);
        p01_ = 0.0;
        p10_ = 0.0;
        p11_ = std::max(var / std::max(dt_s_ * dt_s_, 1e-6), 1e-6);

        q_wf_ = clamp(var * 0.05, params_.q_wf_min, params_.q_wf_max);
        q_rw_ = clamp(var * 0.005, params_.q_rw_min, params_.q_rw_max);
        r_ = clamp(var * 0.5, params_.r_min, params_.r_max);

        initialized_ = true;
        frame_counter_ = 0;
        accepted_update_count_ = 0;
        pending_control_delta_ppm_ = 0.0;
        adapt_freeze_countdown_ = 0;
        innovations_.clear();
        last_innovation_ = 0.0;
        last_s_ = std::max(r_, 1e-9);
        last_k0_ = 0.0;
        last_k1_ = 0.0;
        has_valid_gain_ = false;
    }

    static bool solve_discrete_lyapunov_2x2(
        const std::array<double, 4>& a,
        const std::array<double, 4>& q,
        std::array<double, 4>& p
    ) {
        p = {0.0, 0.0, 0.0, 0.0};
        constexpr int kMaxIter = 200;
        constexpr double kTol = 1e-12;

        for (int iter = 0; iter < kMaxIter; ++iter) {
            const double ap00 = a[0] * p[0] + a[1] * p[2];
            const double ap01 = a[0] * p[1] + a[1] * p[3];
            const double ap10 = a[2] * p[0] + a[3] * p[2];
            const double ap11 = a[2] * p[1] + a[3] * p[3];

            std::array<double, 4> pn{};
            pn[0] = ap00 * a[0] + ap01 * a[1] + q[0];
            pn[1] = ap00 * a[2] + ap01 * a[3] + q[1];
            pn[2] = ap10 * a[0] + ap11 * a[1] + q[2];
            pn[3] = ap10 * a[2] + ap11 * a[3] + q[3];

            double max_diff = 0.0;
            for (size_t i = 0; i < 4; ++i) {
                if (!std::isfinite(pn[i])) return false;
                max_diff = std::max(max_diff, std::abs(pn[i] - p[i]));
            }
            p = pn;
            if (max_diff < kTol) break;
        }
        const double sym = 0.5 * (p[1] + p[2]);
        p[1] = sym;
        p[2] = sym;
        return true;
    }

    static bool solve_3x3(std::array<double, 9> a, std::array<double, 3> b, std::array<double, 3>& x) {
        for (int i = 0; i < 3; ++i) {
            int pivot = i;
            double max_abs = std::abs(a[i * 3 + i]);
            for (int r = i + 1; r < 3; ++r) {
                const double v = std::abs(a[r * 3 + i]);
                if (v > max_abs) {
                    max_abs = v;
                    pivot = r;
                }
            }
            if (max_abs < 1e-14) return false;
            if (pivot != i) {
                for (int c = i; c < 3; ++c) std::swap(a[i * 3 + c], a[pivot * 3 + c]);
                std::swap(b[i], b[pivot]);
            }
            const double diag = a[i * 3 + i];
            for (int c = i; c < 3; ++c) a[i * 3 + c] /= diag;
            b[i] /= diag;

            for (int r = i + 1; r < 3; ++r) {
                const double f = a[r * 3 + i];
                for (int c = i; c < 3; ++c) a[r * 3 + c] -= f * a[i * 3 + c];
                b[r] -= f * b[i];
            }
        }
        for (int i = 2; i >= 0; --i) {
            double acc = b[i];
            for (int c = i + 1; c < 3; ++c) acc -= a[i * 3 + c] * x[c];
            x[i] = acc;
        }
        return std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
    }

    std::array<double, 3> innovation_basis_coeffs(int basis_idx, size_t lag_max) const {
        // H = [1, 0], F = [[1, dt], [0, 1]]
        const double c0 = 1.0;
        const double c1 = dt_s_;

        // A = I - K*H and Acl = A*F
        const double a00 = 1.0 - last_k0_;
        const double a01 = 0.0;
        const double a10 = -last_k1_;
        const double a11 = 1.0;

        std::array<double, 4> acl{
            a00,
            a00 * dt_s_ + a01,
            a10,
            a10 * dt_s_ + a11
        };

        // B columns: process/measurement noises mapped into posterior error dynamics.
        const std::array<double, 2> b_wf{a00, a10};      // A * [1, 0]^T
        const std::array<double, 2> b_rw{a01, a11};      // A * [0, 1]^T
        const std::array<double, 2> b_v{-last_k0_, -last_k1_}; // -K * v

        // D columns in innovation equation nu = C x_prev + D u.
        const double d_wf = 1.0;
        const double d_rw = 0.0;
        const double d_v = 1.0;

        std::array<double, 2> b{};
        double d = 0.0;
        if (basis_idx == 0) {
            b = b_wf;
            d = d_wf;
        } else if (basis_idx == 1) {
            b = b_rw;
            d = d_rw;
        } else {
            b = b_v;
            d = d_v;
        }

        std::array<double, 4> q{
            b[0] * b[0], b[0] * b[1],
            b[1] * b[0], b[1] * b[1]
        };
        std::array<double, 4> p{};
        if (!solve_discrete_lyapunov_2x2(acl, q, p)) {
            return {0.0, 0.0, 0.0};
        }

        // g = E[x_k y_k] = Acl*P*C^T + b*d
        const double pc0 = p[0] * c0 + p[1] * c1;
        const double pc1 = p[2] * c0 + p[3] * c1;
        std::array<double, 2> g{
            acl[0] * pc0 + acl[1] * pc1 + b[0] * d,
            acl[2] * pc0 + acl[3] * pc1 + b[1] * d
        };

        const double r0 = c0 * (p[0] * c0 + p[1] * c1) + c1 * (p[2] * c0 + p[3] * c1) + d * d;

        std::array<double, 3> coeff_head{r0, 0.0, 0.0};
        if (lag_max == 0) return coeff_head;

        // Return only the first 3 entries used by the caller (actual lag loop computes full vector).
        coeff_head[1] = c0 * g[0] + c1 * g[1];
        if (lag_max >= 2) {
            const std::array<double, 2> ag{
                acl[0] * g[0] + acl[1] * g[1],
                acl[2] * g[0] + acl[3] * g[1]
            };
            coeff_head[2] = c0 * ag[0] + c1 * ag[1];
        }
        return coeff_head;
    }

    void build_phi_matrix(size_t lag_max, std::vector<std::array<double, 3>>& phi) const {
        phi.assign(lag_max + 1, {0.0, 0.0, 0.0});

        // H = [1, 0], F = [[1, dt], [0, 1]]
        const double c0 = 1.0;
        const double c1 = dt_s_;

        const double a00 = 1.0 - last_k0_;
        const double a01 = 0.0;
        const double a10 = -last_k1_;
        const double a11 = 1.0;
        std::array<double, 4> acl{
            a00,
            a00 * dt_s_ + a01,
            a10,
            a10 * dt_s_ + a11
        };

        const std::array<std::array<double, 2>, 3> b_cols{{
            {a00, a10},             // wf
            {a01, a11},             // rw
            {-last_k0_, -last_k1_}  // v (measurement noise)
        }};
        const std::array<double, 3> d_cols{1.0, 0.0, 1.0};

        for (int col = 0; col < 3; ++col) {
            const auto& b = b_cols[col];
            const double d = d_cols[col];
            std::array<double, 4> q{
                b[0] * b[0], b[0] * b[1],
                b[1] * b[0], b[1] * b[1]
            };
            std::array<double, 4> p{};
            if (!solve_discrete_lyapunov_2x2(acl, q, p)) {
                continue;
            }

            const double pc0 = p[0] * c0 + p[1] * c1;
            const double pc1 = p[2] * c0 + p[3] * c1;
            std::array<double, 2> g{
                acl[0] * pc0 + acl[1] * pc1 + b[0] * d,
                acl[2] * pc0 + acl[3] * pc1 + b[1] * d
            };

            phi[0][col] = c0 * (p[0] * c0 + p[1] * c1) + c1 * (p[2] * c0 + p[3] * c1) + d * d;
            if (lag_max == 0) continue;

            std::array<double, 2> a_pow_g = g;
            for (size_t lag = 1; lag <= lag_max; ++lag) {
                if (lag > 1) {
                    const std::array<double, 2> next{
                        acl[0] * a_pow_g[0] + acl[1] * a_pow_g[1],
                        acl[2] * a_pow_g[0] + acl[3] * a_pow_g[1]
                    };
                    a_pow_g = next;
                }
                phi[lag][col] = c0 * a_pow_g[0] + c1 * a_pow_g[1];
            }
        }
    }

    bool adapt_qr_from_innovation_acf() {
        const size_t n = innovations_.size();
        if (n < 8) return false;
        const size_t lag_max = std::min(params_.max_lag, n - 1);
        if (lag_max < 2) return false;

        std::vector<double> v(innovations_.begin(), innovations_.end());
        std::vector<double> r_hat(lag_max + 1, 0.0);
        for (size_t lag = 0; lag <= lag_max; ++lag) {
            double acc = 0.0;
            const size_t count = n - lag;
            for (size_t i = lag; i < n; ++i) {
                acc += v[i] * v[i - lag];
            }
            r_hat[lag] = acc / static_cast<double>(count);
        }

        std::vector<std::array<double, 3>> phi;
        build_phi_matrix(lag_max, phi);
        if (phi.empty()) return false;

        std::array<double, 9> normal{
            params_.tikhonov_lambda, 0.0, 0.0,
            0.0, params_.tikhonov_lambda, 0.0,
            0.0, 0.0, params_.tikhonov_lambda
        };
        std::array<double, 3> rhs{
            params_.tikhonov_lambda * q_wf_,
            params_.tikhonov_lambda * q_rw_,
            params_.tikhonov_lambda * r_
        };

        for (size_t lag = 0; lag <= lag_max; ++lag) {
            const double w = 1.0 / static_cast<double>(lag + 1);
            const auto& row = phi[lag];
            for (int i = 0; i < 3; ++i) {
                rhs[i] += w * row[i] * r_hat[lag];
                for (int j = 0; j < 3; ++j) {
                    normal[i * 3 + j] += w * row[i] * row[j];
                }
            }
        }

        std::array<double, 3> theta{0.0, 0.0, 0.0};
        if (!solve_3x3(normal, rhs, theta)) return false;

        double q_wf_new = clamp(theta[0], params_.q_wf_min, params_.q_wf_max);
        double q_rw_new = clamp(theta[1], params_.q_rw_min, params_.q_rw_max);
        double r_new = clamp(theta[2], params_.r_min, params_.r_max);

        const double g = params_.update_smooth;
        q_wf_ = clamp((1.0 - g) * q_wf_ + g * q_wf_new, params_.q_wf_min, params_.q_wf_max);
        q_rw_ = clamp((1.0 - g) * q_rw_ + g * q_rw_new, params_.q_rw_min, params_.q_rw_max);
        r_ = clamp((1.0 - g) * r_ + g * r_new, params_.r_min, params_.r_max);
        return true;
    }
};
/**
 * @brief Delay Spectrum Processor.
 * 
 * Computes delay (time-domain) spectrum from channel estimate.
 * Manages its own IFFT plan and buffers internally.
 */
class DelayProcessor {
public:
    /**
     * @brief Construct a DelayProcessor with internal IFFT plan.
     * @param fft_size FFT size for the plan
     */
    explicit DelayProcessor(size_t fft_size)
        : _fft_size(fft_size),
          _ifft_in(fft_size),
          _ifft_out(fft_size)
    {
        _ifft_plan = fftwf_plan_dft_1d(
            static_cast<int>(fft_size),
            reinterpret_cast<fftwf_complex*>(_ifft_in.data()),
            reinterpret_cast<fftwf_complex*>(_ifft_out.data()),
            FFTW_BACKWARD, FFTW_MEASURE);
    }

    ~DelayProcessor() {
        if (_ifft_plan) fftwf_destroy_plan(_ifft_plan);
    }

    // Non-copyable due to FFTW plans
    DelayProcessor(const DelayProcessor&) = delete;
    DelayProcessor& operator=(const DelayProcessor&) = delete;

    // Move constructible
    DelayProcessor(DelayProcessor&& other) noexcept
        : _fft_size(other._fft_size),
          _ifft_plan(other._ifft_plan),
          _ifft_in(std::move(other._ifft_in)),
          _ifft_out(std::move(other._ifft_out))
    {
        other._ifft_plan = nullptr;
    }

    /**
     * @brief Compute delay spectrum from channel estimate.
     * Performs FFT shift + IFFT + scaling.
     * 
     * @param H_est Channel estimate (frequency domain)
     * @param delay_spectrum Output delay spectrum
     */
    void compute_delay_spectrum(
        const AlignedVector& H_est,
        AlignedVector& delay_spectrum
    ) {
        // FFT shift (swap halves)
        const size_t half = _fft_size / 2;
        
        #pragma omp simd
        for (size_t i = 0; i < half; ++i) {
            _ifft_in[i] = H_est[i + half];
            _ifft_in[i + half] = H_est[i];
        }
        
        // Execute IFFT
        fftwf_execute(_ifft_plan);
        
        // Scale and copy to output
        const float scale = 1.0f / std::sqrt(static_cast<float>(_fft_size));
        delay_spectrum.resize(_fft_size);
        
        #pragma omp simd
        for (size_t i = 0; i < _fft_size; ++i) {
            delay_spectrum[i] = _ifft_out[i] * scale;
        }
    }

    /**
     * @brief Find peak in delay spectrum and compute statistics.
     * 
     * @param delay_spectrum Input delay spectrum
     * @param max_index Output: index of maximum magnitude
     * @param max_mag Output: maximum magnitude value
     * @param avg_mag Output: average magnitude
     * @param cp_length Cyclic prefix length (search range = CP length on both sides)
     */
    static void find_peak(
        const AlignedVector& delay_spectrum,
        size_t& max_index,
        float& max_mag,
        float& avg_mag,
        size_t cp_length = 0
    ) {
        const size_t fft_size = delay_spectrum.size();
        max_index = 0;
        max_mag = 0.0f;
        avg_mag = 0.0f;
        
        // If cp_length is specified, search only within CP range
        // Range: [0, cp_length) and [fft_size - cp_length, fft_size)
        if (cp_length > 0 && cp_length < fft_size) {
            size_t count = 0;

            if (cp_length * 2 >= fft_size) {
                for (size_t i = 0; i < fft_size; ++i) {
                    const float mag = std::abs(delay_spectrum[i]);
                    if (mag > max_mag) {
                        max_mag = mag;
                        max_index = i;
                    }
                    avg_mag += mag;
                    count++;
                }
            } else {
                for (size_t i = 0; i < cp_length; ++i) {
                    const float mag = std::abs(delay_spectrum[i]);
                    if (mag > max_mag) {
                        max_mag = mag;
                        max_index = i;
                    }
                    avg_mag += mag;
                    count++;
                }

                for (size_t i = fft_size - cp_length; i < fft_size; ++i) {
                    const float mag = std::abs(delay_spectrum[i]);
                    if (mag > max_mag) {
                        max_mag = mag;
                        max_index = i;
                    }
                    avg_mag += mag;
                    count++;
                }
            }

            avg_mag /= static_cast<float>(count);
        } else {
            // Original behavior: search entire range
            for (size_t i = 0; i < fft_size; ++i) {
                const float mag = std::abs(delay_spectrum[i]);
                if (mag > max_mag) {
                    max_mag = mag;
                    max_index = i;
                }
                avg_mag += mag;
            }
            avg_mag /= static_cast<float>(fft_size);
        }
    }

    /**
     * @brief Adjust delay index to signed value.
     * Maps indices >= fft_size/2 to negative values.
     */
    static int adjust_delay_index(size_t max_index, size_t fft_size) {
        int adjusted = static_cast<int>(max_index);
        const int half_fft = static_cast<int>(fft_size) / 2;
        if (adjusted >= half_fft) {
            adjusted -= static_cast<int>(fft_size);
        }
        return adjusted;
    }

    /**
     * @brief Quinn's Algorithm for Fractional Delay Estimation.
     * Refines the peak position in the delay spectrum to sub-sample precision.
     * 
     * @param spectrum Delay spectrum (complex values)
     * @param max_index Index of the maximum magnitude
     * @return Fractional delay estimate (-0.5 to 0.5 samples)
     */
    static float estimate_fractional_delay(const AlignedVector& spectrum, size_t max_index) {
        const size_t N = spectrum.size();
        if (N < 3) return 0.0f;
        
        const auto& d0 = spectrum[max_index];
        const auto& d_prev = spectrum[(max_index == 0) ? (N - 1) : (max_index - 1)];
        const auto& d_next = spectrum[(max_index == N - 1) ? 0 : (max_index + 1)];
        const auto magnitude = std::abs(d0);
        
        constexpr float EPSILON = 1e-10f;
        if (magnitude < EPSILON) {
            return 0.0f;
        }
        
        float alpha1 = 0.0f, alpha2 = 0.0f;
        {
            const float denom = std::real(std::conj(d0) * d0);
            const float num1 = std::real(std::conj(d_prev) * d0);
            const float num2 = std::real(std::conj(d_next) * d0);
            
            if (denom > EPSILON) {
                alpha1 = num1 / denom;
                alpha2 = num2 / denom;
            } else {
                alpha1 = std::abs(d_prev) / (magnitude + EPSILON);
                alpha2 = std::abs(d_next) / (magnitude + EPSILON);
            }
            
            // Limit ratio range
            alpha1 = std::max(-0.9999f, std::min(0.9999f, alpha1));
            alpha2 = std::max(-0.9999f, std::min(0.9999f, alpha2));
        }
        
        const float delta1 = alpha1 / (1.0f - alpha1);
        const float delta2 = -alpha2 / (1.0f - alpha2);
        
        // Fast NaN check compatible with -ffast-math.
        if (isNaN(delta1) || isNaN(delta2)) {
            return 0.0f;
        }
        
        const float abs1 = std::abs(delta1);
        const float abs2 = std::abs(delta2);
        
        if (abs1 > 2.0f && abs2 > 2.0f) {
            return 0.5f;
        } else if (abs1 > 2.0f) {
            return delta2;
        } else if (abs2 > 2.0f) {
            return delta1;
        } else {
            return (delta1 > 0.0f && delta2 > 0.0f) ? delta2 : delta1;
        }
    }

private:
    size_t _fft_size;
    fftwf_plan _ifft_plan = nullptr;
    AlignedVector _ifft_in;
    AlignedVector _ifft_out;
};

/**
 * @brief Moving Target Indication (MTI) Filter.
 * 
 * Implements an 8-stage IIR filter for clutter suppression in sensing applications.
 * Uses AVX/OpenMP SIMD for efficient processing of subcarriers.
 */
class MTIFilter {
public:
    MTIFilter(size_t range_fft_size = 1024) {
        resize(range_fft_size);
    }

    void resize(size_t range_fft_size) {
        _range_fft_size = range_fft_size;
        // State is kept as separate real/imag planes (SoA): with std::complex AoS
        // the per-element interleave defeats SIMD and the cascade is compute-bound.
        // Planar state + planar scratch lets the inner loop vectorize cleanly.
        _state_re.assign(8 * 2 * _range_fft_size, 0.0f);
        _state_im.assign(8 * 2 * _range_fft_size, 0.0f);
    }

    void reset() {
        std::fill(_state_re.begin(), _state_re.end(), 0.0f);
        std::fill(_state_im.begin(), _state_im.end(), 0.0f);
    }

    /**
     * @brief Apply MTI filter to the buffer.
     * 
     * @param buffer Input/Output buffer (Channel Response)
     * @param N_proc Number of start subcarriers to process (e.g., fft_size)
     * @param num_symbols Number of symbols in the buffer
     */
    void apply(AlignedVector& buffer, size_t N_proc, size_t num_symbols) {
        static const float SOS[8][6] = {
            {0.993542f, -1.987084f, 0.993542f, 1.000000f, -1.993112f, 0.993951f},
            {0.981889f, -1.963778f, 0.981889f, 1.000000f, -1.981389f, 0.982224f},
            {0.971190f, -1.942380f, 0.971190f, 1.000000f, -1.970564f, 0.971395f},
            {0.961860f, -1.923721f, 0.961860f, 1.000000f, -1.961077f, 0.961903f},
            {0.954245f, -1.908491f, 0.954245f, 1.000000f, -1.953298f, 0.954121f},
            {0.948614f, -1.897228f, 0.948614f, 1.000000f, -1.947526f, 0.948346f},
            {0.945158f, -1.890316f, 0.945158f, 1.000000f, -1.943975f, 0.944795f},
            {0.971593f, -0.971593f, 0.000000f, 1.000000f, -0.971389f, 0.000000f}
        };

        const size_t N_alloc = _range_fft_size;
        const size_t scratch_needed = num_symbols * N_alloc;
        if (_buf_re.size() < scratch_needed) {
            _buf_re.resize(scratch_needed);
            _buf_im.resize(scratch_needed);
        }

        // Deinterleave the processed region of each symbol row into real/imag planes.
        for (size_t i = 0; i < num_symbols; ++i) {
            const std::complex<float>* __restrict__ sd = buffer.data() + i * N_alloc;
            float* __restrict__ re = _buf_re.data() + i * N_alloc;
            float* __restrict__ im = _buf_im.data() + i * N_alloc;
            #pragma omp simd
            for (size_t col = 0; col < N_proc; ++col) {
                re[col] = sd[col].real();
                im[col] = sd[col].imag();
            }
        }

        // 8-stage cascade on planar data (stage -> symbol -> col); the recurrence
        // lives on the symbol axis, so each stage's state plane stays resident
        // across the symbol loop.
        for (int stage = 0; stage < 8; ++stage) {
            const float b0 = SOS[stage][0];
            const float b1 = SOS[stage][1];
            const float b2 = SOS[stage][2];
            const float a1 = SOS[stage][4];
            const float a2 = SOS[stage][5];

            float* __restrict__ s0_re = &_state_re[stage * 2 * N_alloc];
            float* __restrict__ s0_im = &_state_im[stage * 2 * N_alloc];
            float* __restrict__ s1_re = &_state_re[stage * 2 * N_alloc + N_alloc];
            float* __restrict__ s1_im = &_state_im[stage * 2 * N_alloc + N_alloc];

            for (size_t i = 0; i < num_symbols; ++i) {
                float* __restrict__ re = _buf_re.data() + i * N_alloc;
                float* __restrict__ im = _buf_im.data() + i * N_alloc;

                #pragma omp simd
                for (size_t col = 0; col < N_proc; ++col) {
                    const float xr = re[col];
                    const float xi = im[col];
                    const float p0r = s0_re[col];
                    const float p0i = s0_im[col];
                    const float p1r = s1_re[col];
                    const float p1i = s1_im[col];

                    const float yr = xr * b0 + p0r;
                    const float yi = xi * b0 + p0i;
                    s0_re[col] = xr * b1 - yr * a1 + p1r;
                    s0_im[col] = xi * b1 - yi * a1 + p1i;
                    s1_re[col] = xr * b2 - yr * a2;
                    s1_im[col] = xi * b2 - yi * a2;

                    re[col] = yr;
                    im[col] = yi;
                }
            }
        }

        // Interleave the filtered planes back into the buffer (processed region only).
        for (size_t i = 0; i < num_symbols; ++i) {
            std::complex<float>* __restrict__ sd = buffer.data() + i * N_alloc;
            const float* __restrict__ re = _buf_re.data() + i * N_alloc;
            const float* __restrict__ im = _buf_im.data() + i * N_alloc;
            #pragma omp simd
            for (size_t col = 0; col < N_proc; ++col) {
                sd[col] = std::complex<float>(re[col], im[col]);
            }
        }
    }

private:
    AlignedFloatVector _state_re;   // 8 stages x 2 taps x range_fft_size
    AlignedFloatVector _state_im;
    AlignedFloatVector _buf_re;     // deinterleave scratch (grows lazily)
    AlignedFloatVector _buf_im;
    size_t _range_fft_size;
};

/**
 * @brief FFT-based Zoom FFT / CZT evaluator for contiguous DFT-bin segments.
 *
 * This computes a contiguous subset of a length-@p fft_length DFT or inverse DFT
 * without materializing the full transform. Internally it uses the Bluestein /
 * chirp-z identity and reuses FFTW plans across calls.
 */
class ZoomFFTProcessor {
public:
    enum class Direction {
        Forward,
        Backward
    };

    struct Params {
        size_t input_size = 0;
        size_t fft_length = 0;
        size_t output_start_bin = 0;
        size_t output_size = 0;
        Direction direction = Direction::Forward;
    };

    explicit ZoomFFTProcessor(const Params& params)
        : _params(params)
    {
        if (_params.input_size == 0 || _params.fft_length == 0 || _params.output_size == 0) {
            throw std::runtime_error("ZoomFFTProcessor requires non-zero input/output/FFT sizes.");
        }
        if (_params.input_size > _params.fft_length) {
            throw std::runtime_error("ZoomFFTProcessor input_size exceeds fft_length.");
        }
        if (_params.output_start_bin >= _params.fft_length ||
            _params.output_size > _params.fft_length ||
            _params.output_start_bin + _params.output_size > _params.fft_length) {
            throw std::runtime_error("ZoomFFTProcessor output bin range exceeds fft_length.");
        }

        _conv_size = next_power_of_two(_params.input_size + _params.output_size - 1);
        _time_buffer.assign(_conv_size, std::complex<float>(0.0f, 0.0f));
        _freq_buffer.assign(_conv_size, std::complex<float>(0.0f, 0.0f));
        _kernel_fft.assign(_conv_size, std::complex<float>(0.0f, 0.0f));
        _input_chirp.resize(_params.input_size);
        _output_chirp.resize(_params.output_size);

        _forward_plan = fftwf_plan_dft_1d(
            static_cast<int>(_conv_size),
            reinterpret_cast<fftwf_complex*>(_time_buffer.data()),
            reinterpret_cast<fftwf_complex*>(_freq_buffer.data()),
            FFTW_FORWARD,
            FFTW_MEASURE);
        _inverse_plan = fftwf_plan_dft_1d(
            static_cast<int>(_conv_size),
            reinterpret_cast<fftwf_complex*>(_freq_buffer.data()),
            reinterpret_cast<fftwf_complex*>(_time_buffer.data()),
            FFTW_BACKWARD,
            FFTW_MEASURE);

        initialize_kernel();
    }

    ~ZoomFFTProcessor() {
        if (_forward_plan) {
            fftwf_destroy_plan(_forward_plan);
            _forward_plan = nullptr;
        }
        if (_inverse_plan) {
            fftwf_destroy_plan(_inverse_plan);
            _inverse_plan = nullptr;
        }
    }

    ZoomFFTProcessor(const ZoomFFTProcessor&) = delete;
    ZoomFFTProcessor& operator=(const ZoomFFTProcessor&) = delete;

    ZoomFFTProcessor(ZoomFFTProcessor&& other) noexcept
        : _params(other._params),
          _conv_size(other._conv_size),
          _forward_plan(other._forward_plan),
          _inverse_plan(other._inverse_plan),
          _time_buffer(std::move(other._time_buffer)),
          _freq_buffer(std::move(other._freq_buffer)),
          _kernel_fft(std::move(other._kernel_fft)),
          _input_chirp(std::move(other._input_chirp)),
          _output_chirp(std::move(other._output_chirp))
    {
        other._forward_plan = nullptr;
        other._inverse_plan = nullptr;
        other._conv_size = 0;
    }

    /**
     * @brief Evaluate the configured transform segment for a strided input vector.
     *
     * @param input Pointer to the first complex input sample
     * @param input_stride Stride, in complex samples, between consecutive inputs
     * @param output Pointer to the first complex output sample
     * @param output_stride Stride, in complex samples, between consecutive outputs
     */
    void execute(
        const std::complex<float>* input,
        size_t input_stride,
        std::complex<float>* output,
        size_t output_stride = 1)
    {
        if (input == nullptr || output == nullptr) {
            return;
        }

        std::fill(_time_buffer.begin(), _time_buffer.end(), std::complex<float>(0.0f, 0.0f));
        for (size_t n = 0; n < _params.input_size; ++n) {
            _time_buffer[n] = input[n * input_stride] * _input_chirp[n];
        }

        fftwf_execute(_forward_plan);

        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _conv_size; ++i) {
            _freq_buffer[i] *= _kernel_fft[i];
        }

        fftwf_execute(_inverse_plan);

        const float scale = 1.0f / static_cast<float>(_conv_size);
        const size_t output_offset = _params.input_size - 1;
        for (size_t k = 0; k < _params.output_size; ++k) {
            output[k * output_stride] =
                _time_buffer[output_offset + k] * _output_chirp[k] * scale;
        }
    }

    const Params& params() const { return _params; }

private:
    static size_t next_power_of_two(size_t value) {
        size_t out = 1;
        while (out < value) {
            out <<= 1;
        }
        return out;
    }

    void initialize_kernel() {
        const double sign = (_params.direction == Direction::Forward) ? -1.0 : 1.0;
        const double fft_length = static_cast<double>(_params.fft_length);
        const double start_bin = static_cast<double>(_params.output_start_bin);
        const double two_pi_over_n = (2.0 * M_PI) / fft_length;
        const double pi_over_n = M_PI / fft_length;

        for (size_t n = 0; n < _params.input_size; ++n) {
            const double nd = static_cast<double>(n);
            const double phase =
                sign * (two_pi_over_n * start_bin * nd + pi_over_n * nd * nd);
            _input_chirp[n] = std::polar(1.0f, static_cast<float>(phase));
        }

        for (size_t k = 0; k < _params.output_size; ++k) {
            const double kd = static_cast<double>(k);
            const double phase = sign * pi_over_n * kd * kd;
            _output_chirp[k] = std::polar(1.0f, static_cast<float>(phase));
        }

        std::fill(_time_buffer.begin(), _time_buffer.end(), std::complex<float>(0.0f, 0.0f));
        for (int64_t diff = -static_cast<int64_t>(_params.input_size) + 1;
             diff <= static_cast<int64_t>(_params.output_size) - 1;
             ++diff) {
            const size_t idx = static_cast<size_t>(
                diff + static_cast<int64_t>(_params.input_size) - 1);
            const double delta = static_cast<double>(diff);
            const double phase = -sign * pi_over_n * delta * delta;
            _time_buffer[idx] = std::polar(1.0f, static_cast<float>(phase));
        }

        fftwf_execute(_forward_plan);
        std::copy(_freq_buffer.begin(), _freq_buffer.end(), _kernel_fft.begin());
    }

    Params _params{};
    size_t _conv_size = 0;
    fftwf_plan _forward_plan = nullptr;
    fftwf_plan _inverse_plan = nullptr;
    AlignedVector _time_buffer;
    AlignedVector _freq_buffer;
    AlignedVector _kernel_fft;
    AlignedVector _input_chirp;
    AlignedVector _output_chirp;
};


/**
 * @brief Core Sensing Processing Operations.
 * 
 */
class SensingProcessor {
public:
    struct Params {
        size_t fft_size;
        size_t range_fft_size;
        size_t doppler_fft_size;
        size_t sensing_symbol_num;
    };

private:
    Params _params;
    fftwf_plan _range_ifft_plan = nullptr;
    fftwf_plan _doppler_fft_plan = nullptr;
    AlignedVector _channel_buffer;
    MTIFilter _mti_filter;

public:
    explicit SensingProcessor(const Params& params)
        : _params(params),
          _channel_buffer(params.range_fft_size * params.doppler_fft_size, std::complex<float>(0.0f, 0.0f)),
          _mti_filter(params.range_fft_size)
    {
        // Create batch Range IFFT plan
        const int fft_size_int = static_cast<int>(params.range_fft_size);
        const int doppler_fft_size_int = static_cast<int>(params.doppler_fft_size);
        
        _range_ifft_plan = fftwf_plan_many_dft(
            1,                         // rank
            &fft_size_int,             // n (FFT size)
            doppler_fft_size_int,      // howmany (number of symbols)
            reinterpret_cast<fftwf_complex*>(_channel_buffer.data()),
            nullptr,                   // inembed (contiguous)
            1,                         // istride
            fft_size_int,              // idist (distance between FFTs)
            reinterpret_cast<fftwf_complex*>(_channel_buffer.data()),
            nullptr,                   // onembed
            1,                         // ostride
            fft_size_int,              // odist
            FFTW_BACKWARD,             // sign (IFFT)
            FFTW_MEASURE
        );
        
        // Create Doppler FFT plan
        _doppler_fft_plan = fftwf_plan_many_dft(
            1,                         // rank
            &doppler_fft_size_int,     // n (number of symbols)
            fft_size_int,              // howmany (number of subcarriers)
            reinterpret_cast<fftwf_complex*>(_channel_buffer.data()),
            nullptr,                   // inembed
            fft_size_int,              // istride (stride between symbols)
            1,                         // idist (distance between subcarriers)
            reinterpret_cast<fftwf_complex*>(_channel_buffer.data()),
            nullptr,                   // onembed
            fft_size_int,              // ostride
            1,                         // odist
            FFTW_FORWARD,              // sign (FFT)
            FFTW_MEASURE
        );
    }

    ~SensingProcessor() {
        if (_range_ifft_plan) fftwf_destroy_plan(_range_ifft_plan);
        if (_doppler_fft_plan) fftwf_destroy_plan(_doppler_fft_plan);
    }

    // Non-copyable due to FFTW plans
    SensingProcessor(const SensingProcessor&) = delete;
    SensingProcessor& operator=(const SensingProcessor&) = delete;

    // Move constructible
    SensingProcessor(SensingProcessor&& other) noexcept
        : _params(other._params),
          _range_ifft_plan(other._range_ifft_plan),
          _doppler_fft_plan(other._doppler_fft_plan),
          _channel_buffer(std::move(other._channel_buffer))
    {
        other._range_ifft_plan = nullptr;
        other._doppler_fft_plan = nullptr;
    }

    /**
     * @brief Get the internal channel buffer for external data input.
     */
    AlignedVector& channel_buffer() { return _channel_buffer; }
    const AlignedVector& channel_buffer() const { return _channel_buffer; }

    /**
     * @brief Execute Range IFFT on internal channel buffer.
     */
    void execute_range_ifft() {
        fftwf_execute(_range_ifft_plan);
    }

    /**
     * @brief Execute Doppler FFT on internal channel buffer.
     */
    void execute_doppler_fft() {
        fftwf_execute(_doppler_fft_plan);
    }

    /**
     * @brief Clear the channel buffer to zeros.
     */
    void clear_channel_buffer() {
        _channel_buffer.assign(_params.range_fft_size * _params.doppler_fft_size, 
                               std::complex<float>(0.0f, 0.0f));
    }

    /**
     * @brief Copy frequency-domain symbols directly to internal channel buffer.
     * Used by UE where symbols are already in frequency domain.
     * 
     * @param rx_symbols Vector of frequency-domain symbols to copy
     */
    void copy_symbols_to_buffer(const std::vector<AlignedVector>& rx_symbols, size_t symbol_count = 0) {
        size_t num_symbols = symbol_count > 0 ? symbol_count : rx_symbols.size();
        num_symbols = std::min(num_symbols, std::min(rx_symbols.size(), _params.sensing_symbol_num));
        for (size_t i = 0; i < num_symbols; ++i) {
            auto* dest = _channel_buffer.data() + i * _params.range_fft_size;
            std::copy(rx_symbols[i].begin(), rx_symbols[i].end(), dest);
        }
    }

    /**
     * @brief Copy a single FFT output to the internal channel buffer at specified index.
     * Used by BS where FFT is executed externally per symbol.
     * 
     * @param symbol_idx Index of the symbol in the buffer
     * @param fft_output FFT output data to copy
     */
    void copy_fft_result_to_buffer(size_t symbol_idx, const AlignedVector& fft_output) {
        if (symbol_idx >= _params.doppler_fft_size) return;
        auto* dest = _channel_buffer.data() + symbol_idx * _params.range_fft_size;
        std::copy(fft_output.begin(), fft_output.end(), dest);
    }

    /**
     * @brief Initialize Hamming windows for range and Doppler processing.
     */
    void init_windows(AlignedFloatVector& range_window, AlignedFloatVector& doppler_window) {
        WindowGenerator::generate_hamming(range_window, _params.fft_size);
        WindowGenerator::generate_hamming(doppler_window, _params.sensing_symbol_num);
    }

    /**
     * @brief Apply MTI filter to the internal channel buffer.
     * 
     * @param enabled Whether to enable MTI filtering. If false, does nothing.
     */
    void apply_mti(bool enabled, size_t symbol_count = 0) {
        if (enabled) {
            const size_t num_symbols =
                (symbol_count > 0) ? std::min(symbol_count, _params.sensing_symbol_num) : _params.sensing_symbol_num;
            _mti_filter.apply(_channel_buffer, _params.fft_size, num_symbols);
        }
    }

    /**
     * @brief Channel estimation with conjugate multiplication and in-place FFT shift.
     * For mono-static sensing where TX symbols have unit magnitude.
     * Operates on internal channel buffer.
     */
    void channel_estimate_with_shift(const std::vector<AlignedVector>& tx_symbols, size_t symbol_count = 0) {
        const size_t fft_size = _params.fft_size;
        const size_t half_size = fft_size / 2;

        size_t num_symbols = symbol_count > 0 ? symbol_count : tx_symbols.size();
        num_symbols = std::min(num_symbols, std::min(tx_symbols.size(), _params.sensing_symbol_num));
        for (size_t i = 0; i < num_symbols; ++i) {
            auto* __restrict__ ch_data = _channel_buffer.data() + i * _params.range_fft_size;
            const auto* __restrict__ tx_data = tx_symbols[i].data();
            
            // Combined: channel estimation with multiplication + FFT shift
            #pragma omp simd simdlen(16) aligned(ch_data, tx_data: 64)
            for (size_t k = 0; k < half_size; ++k) {
                // First half
                float ch_real = ch_data[k].real();
                float ch_imag = ch_data[k].imag();
                float tx_real = tx_data[k].real();
                float tx_imag = tx_data[k].imag();
                
                float est_real = ch_real * tx_real + ch_imag * tx_imag;
                float est_imag = ch_imag * tx_real - ch_real * tx_imag;
                
                // Second half
                float ch2_real = ch_data[k + half_size].real();
                float ch2_imag = ch_data[k + half_size].imag();
                float tx2_real = tx_data[k + half_size].real();
                float tx2_imag = tx_data[k + half_size].imag();
                
                // FFT shift: swap halves
                ch_data[k] = std::complex<float>(
                    ch2_real * tx2_real + ch2_imag * tx2_imag,
                    ch2_imag * tx2_real - ch2_real * tx2_imag
                );
                ch_data[k + half_size] = std::complex<float>(est_real, est_imag);
            }
        }
    }

    /**
     * @brief Channel estimation with division.
     * @deprecated Not used - remodulated QPSK symbols have unit magnitude, use channel_estimate_with_shift instead.
     */
    void channel_estimate_with_division(
        AlignedVector& channel_buffer,
        const std::vector<AlignedVector>& tx_symbols,
        size_t fft_size
    ) {
        for (size_t i = 0; i < _params.sensing_symbol_num; ++i) {
            auto* __restrict__ ch_data = channel_buffer.data() + i * _params.range_fft_size;
            const auto* __restrict__ tx_data = tx_symbols[i].data();
            
            #pragma omp simd simdlen(16) aligned(ch_data, tx_data: 64)
            for (size_t k = 0; k < fft_size; ++k) {
                float rx_real = ch_data[k].real();
                float rx_imag = ch_data[k].imag();
                float tx_real = tx_data[k].real();
                float tx_imag = tx_data[k].imag();
                float denom = tx_real * tx_real + tx_imag * tx_imag;
                float inv_denom = 1.0f / denom;
                ch_data[k] = std::complex<float>(
                    (rx_real * tx_real + rx_imag * tx_imag) * inv_denom,
                    (rx_imag * tx_real - rx_real * tx_imag) * inv_denom
                );
            }
        }
    }

    /**
     * @brief In-place FFT shift for each symbol in buffer.
     */
    void fft_shift_symbols(AlignedVector& buffer, size_t fft_size) {
        const size_t half_size = fft_size / 2;
        
        for (size_t i = 0; i < _params.sensing_symbol_num; ++i) {
            auto* symbol_data = buffer.data() + i * _params.range_fft_size;
            
            #pragma omp simd simdlen(16) aligned(symbol_data: 64)
            for (size_t j = 0; j < half_size; ++j) {
                std::complex<float> temp = symbol_data[j];
                symbol_data[j] = symbol_data[j + half_size];
                symbol_data[j + half_size] = temp;
            }
        }
    }

    /**
     * @brief Apply range and Doppler windows to channel buffer.
     */
    void apply_windows(
        AlignedVector& buffer,
        const AlignedFloatVector& range_window,
        const AlignedFloatVector& doppler_window,
        size_t symbol_count = 0
    ) {
        const size_t num_symbols =
            (symbol_count > 0) ? std::min(symbol_count, _params.sensing_symbol_num) : _params.sensing_symbol_num;
        // Fuse range and Doppler windows into a single contiguous per-symbol pass.
        // The Doppler weight is constant across a symbol row (broadcast scalar), so
        // symbol_data[j] *= range_window[j] * doppler_window[i] yields the same product
        // as the previous two passes while avoiding the strided Doppler access (which
        // walked the buffer with stride range_fft_size, i.e. a gather) and halving the
        // buffer traffic.
        for (size_t i = 0; i < num_symbols; ++i) {
            auto* __restrict__ symbol_data = buffer.data() + i * _params.range_fft_size;
            const float dw = doppler_window[i];
            #pragma omp simd simdlen(16) aligned(symbol_data: 64)
            for (size_t j = 0; j < _params.fft_size; ++j) {
                symbol_data[j] *= range_window[j] * dw;
            }
        }
    }

    const Params& params() const { return _params; }
};

class MicroDopplerState {
public:
    static constexpr size_t kBufferLength = 5000;
    static constexpr size_t kWindowLength = 256;
    static constexpr size_t kOverlap = 192;
    static constexpr size_t kStep = kWindowLength - kOverlap;
    static constexpr size_t kFftSize = 256;

    explicit MicroDopplerState(size_t buffer_length = kBufferLength)
        : _buffer_length(buffer_length),
          _ring(buffer_length, std::complex<float>(0.0f, 0.0f)),
          _window(kWindowLength),
          _fft_in(kFftSize),
          _fft_out(kFftSize)
    {
        WindowGenerator::generate_hamming(_window, kWindowLength);
        _fft_plan = fftwf_plan_dft_1d(
            static_cast<int>(kFftSize),
            reinterpret_cast<fftwf_complex*>(_fft_in.data()),
            reinterpret_cast<fftwf_complex*>(_fft_out.data()),
            FFTW_FORWARD,
            FFTW_MEASURE);
    }

    ~MicroDopplerState() {
        if (_fft_plan != nullptr) {
            fftwf_destroy_plan(_fft_plan);
            _fft_plan = nullptr;
        }
    }

    MicroDopplerState(const MicroDopplerState&) = delete;
    MicroDopplerState& operator=(const MicroDopplerState&) = delete;

    MicroDopplerState(MicroDopplerState&& other) noexcept
        : _buffer_length(other._buffer_length),
          _ring(std::move(other._ring)),
          _write_pos(other._write_pos),
          _size(other._size),
          _window(std::move(other._window)),
          _fft_in(std::move(other._fft_in)),
          _fft_out(std::move(other._fft_out)),
          _signal_scratch(std::move(other._signal_scratch)),
          _fft_plan(other._fft_plan)
    {
        other._fft_plan = nullptr;
        other._write_pos = 0;
        other._size = 0;
    }

    void clear() {
        _write_pos = 0;
        _size = 0;
        std::fill(_ring.begin(), _ring.end(), std::complex<float>(0.0f, 0.0f));
    }

    size_t size() const { return _size; }

    void append_samples(const std::complex<float>* samples, size_t count) {
        if (samples == nullptr || count == 0 || _ring.empty()) {
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            _ring[_write_pos] = samples[i];
            _write_pos = (_write_pos + 1) % _buffer_length;
            if (_size < _buffer_length) {
                ++_size;
            }
        }
    }

    bool compute_spectrum(
        std::vector<float>& out_spectrum,
        uint32_t& out_rows,
        uint32_t& out_cols,
        std::array<float, 4>& out_extent)
    {
        out_spectrum.clear();
        out_rows = 0;
        out_cols = 0;
        out_extent = {0.0f, 0.0f, 0.0f, 0.0f};
        if (_size < kWindowLength || _fft_plan == nullptr) {
            return false;
        }

        _signal_scratch.resize(_size);
        const size_t start =
            (_size == _buffer_length) ? _write_pos : 0;
        for (size_t i = 0; i < _size; ++i) {
            _signal_scratch[i] = _ring[(start + i) % _buffer_length];
        }

        const size_t n_frames = 1 + (_size - kWindowLength) / kStep;
        if (n_frames == 0) {
            return false;
        }

        out_rows = static_cast<uint32_t>(kFftSize);
        out_cols = static_cast<uint32_t>(n_frames);
        out_spectrum.assign(kFftSize * n_frames, 0.0f);

        const size_t half_fft = kFftSize / 2;
        for (size_t frame_idx = 0; frame_idx < n_frames; ++frame_idx) {
            const size_t start_idx = frame_idx * kStep;
            std::fill(_fft_in.begin(), _fft_in.end(), std::complex<float>(0.0f, 0.0f));
            #pragma omp simd simdlen(16)
            for (size_t i = 0; i < kWindowLength; ++i) {
                _fft_in[i] = _signal_scratch[start_idx + i] * _window[i];
            }
            fftwf_execute(_fft_plan);

            for (size_t row = 0; row < kFftSize; ++row) {
                const size_t shifted_row = (row + half_fft) % kFftSize;
                out_spectrum[row * n_frames + frame_idx] =
                    20.0f * std::log10(std::abs(_fft_out[shifted_row]) + 1e-12f);
            }
        }

        const float x0 = 0.0f;
        const float x1 = (n_frames > 1)
            ? static_cast<float>((n_frames - 1) * kStep)
            : 1.0f;
        const float y0 = -0.5f;
        const float y1 = (static_cast<float>(half_fft) - 1.0f) /
            static_cast<float>(kFftSize);
        out_extent = {x0, x1, y0, y1};
        return true;
    }

private:
    size_t _buffer_length = kBufferLength;
    std::vector<std::complex<float>> _ring;
    size_t _write_pos = 0;
    size_t _size = 0;
    AlignedFloatVector _window;
    AlignedVector _fft_in;
    AlignedVector _fft_out;
    std::vector<std::complex<float>> _signal_scratch;
    fftwf_plan _fft_plan = nullptr;
};

inline void compute_shifted_magnitude_db(
    const std::complex<float>* rd_data,
    size_t rows,
    size_t cols,
    AlignedFloatVector& out_db)
{
    out_db.resize(rows * cols);
    if (rd_data == nullptr || rows == 0 || cols == 0) {
        return;
    }
    const size_t half_rows = rows / 2;
    for (size_t row = 0; row < rows; ++row) {
        const size_t shifted_row = (row + half_rows) % rows;
        const auto* __restrict__ src = rd_data + shifted_row * cols;
        auto* __restrict__ dst = out_db.data() + row * cols;
        #pragma omp simd simdlen(16)
        for (size_t col = 0; col < cols; ++col) {
            const float re = src[col].real();
            const float im = src[col].imag();
            dst[col] = 10.0f * std::log10(re * re + im * im + 1e-24f);
        }
    }
}

inline void extract_range_bin_trace(
    const std::complex<float>* range_time_rows,
    size_t num_symbols,
    size_t stride,
    size_t range_bin,
    AlignedVector& out_trace)
{
    out_trace.resize(num_symbols);
    if (range_time_rows == nullptr || stride == 0) {
        return;
    }
    // No omp simd: strided (stride=range_fft_size) gather; scalar is equal-or-faster.
    for (size_t row = 0; row < num_symbols; ++row) {
        out_trace[row] = range_time_rows[row * stride + range_bin];
    }
}

inline void run_ca_cfar_2d_full(
    const AlignedFloatVector& rd_db,
    size_t rows,
    size_t cols,
    const SensingCfarParams& params,
    std::vector<SensingDetectionPoint>& out_points,
    uint32_t& raw_hits,
    uint32_t& shown_hits,
    SensingCfarStats& out_stats)
{
    out_points.clear();
    raw_hits = 0;
    shown_hits = 0;
    out_stats = SensingCfarStats{};
    out_stats.power_min_db = params.min_power_db;

    if (!params.enabled || rows == 0 || cols == 0) {
        return;
    }

    const int td = std::max(0, params.train_doppler);
    const int tr = std::max(0, params.train_range);
    const int gd = std::max(0, params.guard_doppler);
    const int gr = std::max(0, params.guard_range);
    const int outer_h = td + gd;
    const int outer_w = tr + gr;
    if (rows <= static_cast<size_t>(2 * outer_h) ||
        cols <= static_cast<size_t>(2 * outer_w)) {
        return;
    }

    const double alpha = std::max(
        1e-12,
        std::pow(10.0, static_cast<double>(params.alpha_db) / 10.0));
    const double eps = 1e-12;
    std::vector<double> power(rows * cols, 0.0);
    #pragma omp simd simdlen(16)
    for (size_t idx = 0; idx < rd_db.size(); ++idx) {
        power[idx] = std::pow(10.0, static_cast<double>(rd_db[idx]) / 10.0);
    }

    std::vector<double> integral((rows + 1) * (cols + 1), 0.0);
    for (size_t row = 0; row < rows; ++row) {
        double row_accum = 0.0;
        const size_t row_offset = row * cols;
        for (size_t col = 0; col < cols; ++col) {
            row_accum += power[row_offset + col];
            integral[(row + 1) * (cols + 1) + (col + 1)] =
                integral[row * (cols + 1) + (col + 1)] + row_accum;
        }
    }

    auto rect_sum = [&](int top, int left, int bottom, int right) -> double {
        const size_t top_u = static_cast<size_t>(top);
        const size_t left_u = static_cast<size_t>(left);
        const size_t bottom_u = static_cast<size_t>(bottom + 1);
        const size_t right_u = static_cast<size_t>(right + 1);
        return integral[bottom_u * (cols + 1) + right_u]
             - integral[top_u * (cols + 1) + right_u]
             - integral[bottom_u * (cols + 1) + left_u]
             + integral[top_u * (cols + 1) + left_u];
    };

    const int outer_cells = (2 * outer_h + 1) * (2 * outer_w + 1);
    const int guard_cells = (2 * gd + 1) * (2 * gr + 1);
    const int training_cells = std::max(1, outer_cells - guard_cells);
    const int center_row = static_cast<int>(rows / 2);

    double noise_min = std::numeric_limits<double>::infinity();
    double noise_max = 0.0;
    double thresh_min = std::numeric_limits<double>::infinity();
    double thresh_max = 0.0;
    uint32_t invalid_cells = 0;
    uint32_t nonfinite_cells = 0;
    uint32_t nonpositive_cells = 0;

    struct RankedPoint {
        float value_db = 0.0f;
        SensingDetectionPoint point;
    };
    std::vector<RankedPoint> ranked_points;

    for (int row = outer_h; row < static_cast<int>(rows) - outer_h; ++row) {
        for (int col = outer_w; col < static_cast<int>(cols) - outer_w; ++col) {
            if (params.min_range_bin > 0 && col < params.min_range_bin) {
                continue;
            }
            if (params.dc_exclusion_bins > 0 &&
                std::abs(row - center_row) <= params.dc_exclusion_bins) {
                continue;
            }

            const double outer_sum = rect_sum(
                row - outer_h,
                col - outer_w,
                row + outer_h,
                col + outer_w);
            const double guard_sum = rect_sum(
                row - gd,
                col - gr,
                row + gd,
                col + gr);
            const double noise_mean = (outer_sum - guard_sum) /
                static_cast<double>(training_cells);
            const double cut_power = power[static_cast<size_t>(row) * cols + static_cast<size_t>(col)];
            const bool finite = std::isfinite(noise_mean) && std::isfinite(cut_power);
            const bool positive = noise_mean > eps;
            if (!finite) {
                ++nonfinite_cells;
                ++invalid_cells;
                continue;
            }
            if (!positive) {
                ++nonpositive_cells;
                ++invalid_cells;
                continue;
            }

            const double threshold = alpha * noise_mean;
            noise_min = std::min(noise_min, noise_mean);
            noise_max = std::max(noise_max, noise_mean);
            thresh_min = std::min(thresh_min, threshold);
            thresh_max = std::max(thresh_max, threshold);

            const float value_db =
                rd_db[static_cast<size_t>(row) * cols + static_cast<size_t>(col)];
            if (value_db < params.min_power_db || cut_power <= threshold) {
                continue;
            }

            RankedPoint ranked{};
            ranked.value_db = value_db;
            ranked.point.doppler_idx = row;
            ranked.point.range_idx = col;
            ranked_points.push_back(ranked);
        }
    }

    raw_hits = static_cast<uint32_t>(ranked_points.size());
    std::sort(
        ranked_points.begin(),
        ranked_points.end(),
        [](const RankedPoint& lhs, const RankedPoint& rhs) {
            return lhs.value_db > rhs.value_db;
        });
    if (ranked_points.size() > static_cast<size_t>(std::max(1, params.max_points))) {
        ranked_points.resize(static_cast<size_t>(std::max(1, params.max_points)));
    }

    shown_hits = static_cast<uint32_t>(ranked_points.size());
    out_points.reserve(ranked_points.size());
    for (const auto& ranked : ranked_points) {
        out_points.push_back(ranked.point);
    }

    if (!std::isfinite(noise_min)) {
        noise_min = 0.0;
    }
    if (!std::isfinite(thresh_min)) {
        thresh_min = 0.0;
    }
    out_stats.noise_min = static_cast<float>(noise_min);
    out_stats.noise_max = static_cast<float>(noise_max);
    out_stats.thresh_min = static_cast<float>(thresh_min);
    out_stats.thresh_max = static_cast<float>(thresh_max);
    out_stats.invalid_cells = invalid_cells;
    out_stats.nonfinite_cells = nonfinite_cells;
    out_stats.nonpositive_cells = nonpositive_cells;
    out_stats.power_min_db = params.min_power_db;
}

inline void run_os_cfar_2d_full(
    const AlignedFloatVector& rd_db,
    size_t rows,
    size_t cols,
    const SensingCfarParams& params,
    std::vector<SensingDetectionPoint>& out_points,
    uint32_t& raw_hits,
    uint32_t& shown_hits,
    SensingCfarStats& out_stats)
{
    out_points.clear();
    raw_hits = 0;
    shown_hits = 0;
    out_stats = SensingCfarStats{};
    out_stats.power_min_db = params.min_power_db;

    if (!params.enabled || rows == 0 || cols == 0) {
        return;
    }

    const int td = std::max(0, params.train_doppler);
    const int tr = std::max(0, params.train_range);
    const int gd = std::max(0, params.guard_doppler);
    const int gr = std::max(0, params.guard_range);
    const int outer_h = td + gd;
    const int outer_w = tr + gr;
    if (rows <= static_cast<size_t>(2 * outer_h) ||
        cols <= static_cast<size_t>(2 * outer_w)) {
        return;
    }

    const double alpha = std::max(
        1e-12,
        std::pow(10.0, static_cast<double>(params.alpha_db) / 10.0));
    const double eps = 1e-12;
    std::vector<double> power(rows * cols, 0.0);
    #pragma omp simd simdlen(16)
    for (size_t idx = 0; idx < rd_db.size(); ++idx) {
        power[idx] = std::pow(10.0, static_cast<double>(rd_db[idx]) / 10.0);
    }

    struct WindowOffset {
        int row = 0;
        int col = 0;
    };
    std::vector<WindowOffset> training_offsets;
    training_offsets.reserve(static_cast<size_t>((2 * outer_h + 1) * (2 * outer_w + 1)));
    for (int row = -outer_h; row <= outer_h; ++row) {
        for (int col = -outer_w; col <= outer_w; ++col) {
            if (std::abs(row) <= gd && std::abs(col) <= gr) {
                continue;
            }
            training_offsets.push_back(WindowOffset{row, col});
        }
    }
    const int training_cells = static_cast<int>(training_offsets.size());
    if (training_cells <= 0) {
        return;
    }

    const double clamped_rank = std::clamp(static_cast<double>(params.os_rank_percent), 0.0, 100.0);
    const size_t rank_index = static_cast<size_t>(std::llround(
        (clamped_rank / 100.0) * static_cast<double>(training_cells - 1)));
    const int center_row = static_cast<int>(rows / 2);

    struct CandidatePoint {
        float value_db = 0.0f;
        int row = 0;
        int col = 0;
    };
    std::vector<CandidatePoint> candidates;
    candidates.reserve((rows - static_cast<size_t>(2 * outer_h)) *
                       (cols - static_cast<size_t>(2 * outer_w)));
    for (int row = outer_h; row < static_cast<int>(rows) - outer_h; ++row) {
        if (params.dc_exclusion_bins > 0 &&
            std::abs(row - center_row) <= params.dc_exclusion_bins) {
            continue;
        }
        for (int col = outer_w; col < static_cast<int>(cols) - outer_w; ++col) {
            if (params.min_range_bin > 0 && col < params.min_range_bin) {
                continue;
            }
            candidates.push_back(CandidatePoint{
                rd_db[static_cast<size_t>(row) * cols + static_cast<size_t>(col)],
                row,
                col
            });
        }
    }
    if (candidates.empty()) {
        return;
    }

    const size_t candidate_limit = std::min<size_t>(
        candidates.size(),
        std::max<size_t>(
            static_cast<size_t>(std::max(1, params.max_points)) * static_cast<size_t>(32),
            static_cast<size_t>(1024)));
    auto candidate_desc = [](const CandidatePoint& lhs, const CandidatePoint& rhs) {
        return lhs.value_db > rhs.value_db;
    };
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + candidate_limit,
        candidates.end(),
        candidate_desc);
    candidates.resize(candidate_limit);

    const int active_rows = static_cast<int>(rows) - 2 * outer_h;
    const int active_cols = static_cast<int>(cols) - 2 * outer_w;
    std::vector<uint8_t> suppressed(static_cast<size_t>(active_rows) * static_cast<size_t>(active_cols), 0);
    const int suppress_d = std::max(0, params.os_suppress_doppler);
    const int suppress_r = std::max(0, params.os_suppress_range);
    double noise_min = std::numeric_limits<double>::infinity();
    double noise_max = 0.0;
    double thresh_min = std::numeric_limits<double>::infinity();
    double thresh_max = 0.0;
    uint32_t invalid_cells = 0;
    uint32_t nonfinite_cells = 0;
    uint32_t nonpositive_cells = 0;

    struct RankedPoint {
        float value_db = 0.0f;
        SensingDetectionPoint point;
    };
    std::vector<RankedPoint> ranked_points;
    ranked_points.reserve(candidate_limit);
    std::vector<double> training_values;
    training_values.reserve(static_cast<size_t>(training_cells));

    for (const auto& candidate : candidates) {
        const int local_row = candidate.row - outer_h;
        const int local_col = candidate.col - outer_w;
        if (suppressed[static_cast<size_t>(local_row) * static_cast<size_t>(active_cols) +
                       static_cast<size_t>(local_col)] != 0) {
            continue;
        }

        const float cut_db = candidate.value_db;
        if (!std::isfinite(cut_db) || cut_db < params.min_power_db) {
            continue;
        }

        training_values.clear();
        for (const auto& offset : training_offsets) {
            const double value = power[
                static_cast<size_t>(candidate.row + offset.row) * cols +
                static_cast<size_t>(candidate.col + offset.col)];
            if (std::isfinite(value)) {
                training_values.push_back(value);
            }
        }
        if (training_values.empty()) {
            ++nonfinite_cells;
            ++invalid_cells;
            continue;
        }

        const size_t local_rank_index = std::min(rank_index, training_values.size() - 1);
        std::nth_element(
            training_values.begin(),
            training_values.begin() + static_cast<std::ptrdiff_t>(local_rank_index),
            training_values.end());
        const double noise_order = training_values[local_rank_index];
        if (!std::isfinite(noise_order)) {
            ++nonfinite_cells;
            ++invalid_cells;
            continue;
        }
        if (noise_order <= eps) {
            ++nonpositive_cells;
            ++invalid_cells;
            continue;
        }

        const double threshold = alpha * noise_order;
        noise_min = std::min(noise_min, noise_order);
        noise_max = std::max(noise_max, noise_order);
        thresh_min = std::min(thresh_min, threshold);
        thresh_max = std::max(thresh_max, threshold);

        const double cut_power = power[
            static_cast<size_t>(candidate.row) * cols + static_cast<size_t>(candidate.col)];
        if (cut_power <= threshold) {
            continue;
        }

        RankedPoint ranked{};
        ranked.value_db = cut_db;
        ranked.point.doppler_idx = candidate.row;
        ranked.point.range_idx = candidate.col;
        ranked_points.push_back(ranked);

        const int lo_r = std::max(0, local_row - suppress_d);
        const int hi_r = std::min(active_rows, local_row + suppress_d + 1);
        const int lo_c = std::max(0, local_col - suppress_r);
        const int hi_c = std::min(active_cols, local_col + suppress_r + 1);
        for (int row = lo_r; row < hi_r; ++row) {
            auto* suppress_row = suppressed.data() +
                static_cast<size_t>(row) * static_cast<size_t>(active_cols);
            std::fill(
                suppress_row + static_cast<size_t>(lo_c),
                suppress_row + static_cast<size_t>(hi_c),
                static_cast<uint8_t>(1));
        }
    }

    raw_hits = static_cast<uint32_t>(ranked_points.size());
    if (ranked_points.size() > static_cast<size_t>(std::max(1, params.max_points))) {
        ranked_points.resize(static_cast<size_t>(std::max(1, params.max_points)));
    }

    shown_hits = static_cast<uint32_t>(ranked_points.size());
    out_points.reserve(ranked_points.size());
    for (const auto& ranked : ranked_points) {
        out_points.push_back(ranked.point);
    }

    if (!std::isfinite(noise_min)) {
        noise_min = 0.0;
    }
    if (!std::isfinite(thresh_min)) {
        thresh_min = 0.0;
    }
    out_stats.noise_min = static_cast<float>(noise_min);
    out_stats.noise_max = static_cast<float>(noise_max);
    out_stats.thresh_min = static_cast<float>(thresh_min);
    out_stats.thresh_max = static_cast<float>(thresh_max);
    out_stats.invalid_cells = invalid_cells;
    out_stats.nonfinite_cells = nonfinite_cells;
    out_stats.nonpositive_cells = nonpositive_cells;
    out_stats.power_min_db = params.min_power_db;
}

inline std::vector<SensingCluster> cluster_detected_targets(
    const std::vector<SensingDetectionPoint>& points,
    const AlignedFloatVector& rd_db,
    size_t rows,
    size_t cols,
    int doppler_gap = 2,
    int range_gap = 2)
{
    if (points.empty() || rd_db.empty() || rows == 0 || cols == 0) {
        return {};
    }

    std::vector<uint8_t> visited(points.size(), 0);
    std::vector<SensingCluster> clusters;
    clusters.reserve(points.size());

    auto rd_value = [&](int row, int col) -> float {
        if (row < 0 || col < 0 ||
            row >= static_cast<int>(rows) ||
            col >= static_cast<int>(cols)) {
            return -std::numeric_limits<float>::infinity();
        }
        return rd_db[static_cast<size_t>(row) * cols + static_cast<size_t>(col)];
    };

    for (size_t seed_idx = 0; seed_idx < points.size(); ++seed_idx) {
        if (visited[seed_idx]) {
            continue;
        }

        std::vector<size_t> queue;
        std::vector<size_t> cluster_indices;
        queue.push_back(seed_idx);
        visited[seed_idx] = 1;
        while (!queue.empty()) {
            const size_t cur_idx = queue.back();
            queue.pop_back();
            cluster_indices.push_back(cur_idx);
            const int cur_d = points[cur_idx].doppler_idx;
            const int cur_r = points[cur_idx].range_idx;

            for (size_t nbr_idx = 0; nbr_idx < points.size(); ++nbr_idx) {
                if (visited[nbr_idx]) {
                    continue;
                }
                if (std::abs(points[nbr_idx].doppler_idx - cur_d) <= doppler_gap &&
                    std::abs(points[nbr_idx].range_idx - cur_r) <= range_gap) {
                    visited[nbr_idx] = 1;
                    queue.push_back(nbr_idx);
                }
            }
        }

        if (cluster_indices.empty()) {
            continue;
        }

        size_t peak_local_idx = 0;
        float peak_strength = -std::numeric_limits<float>::infinity();
        double weight_sum = 0.0;
        double centroid_d = 0.0;
        double centroid_r = 0.0;
        for (size_t i = 0; i < cluster_indices.size(); ++i) {
            const auto& point = points[cluster_indices[i]];
            const float value_db = rd_value(point.doppler_idx, point.range_idx);
            if (value_db > peak_strength) {
                peak_strength = value_db;
                peak_local_idx = i;
            }
            const double weight = std::pow(10.0, static_cast<double>(value_db) / 20.0);
            weight_sum += weight;
            centroid_d += static_cast<double>(point.doppler_idx) * weight;
            centroid_r += static_cast<double>(point.range_idx) * weight;
        }

        SensingCluster cluster{};
        const auto& peak_point = points[cluster_indices[peak_local_idx]];
        cluster.peak_doppler_idx = peak_point.doppler_idx;
        cluster.peak_range_idx = peak_point.range_idx;
        cluster.peak_strength_db = peak_strength;
        cluster.cluster_size = static_cast<uint32_t>(cluster_indices.size());
        if (weight_sum > 0.0) {
            cluster.centroid_doppler_idx = static_cast<float>(centroid_d / weight_sum);
            cluster.centroid_range_idx = static_cast<float>(centroid_r / weight_sum);
        } else {
            cluster.centroid_doppler_idx = static_cast<float>(peak_point.doppler_idx);
            cluster.centroid_range_idx = static_cast<float>(peak_point.range_idx);
        }
        clusters.push_back(cluster);
    }

    std::sort(
        clusters.begin(),
        clusters.end(),
        [](const SensingCluster& lhs, const SensingCluster& rhs) {
            return lhs.peak_strength_db > rhs.peak_strength_db;
        });
    return clusters;
}


// ============== Pure Signal Processing Functions ==============

/**
 * @brief Weighted Linear Regression.
 * 
 * Calculates the slope (beta) and intercept (alpha) of a line that best fits the
 * weighted input data points using the least squares method. 
 * Used for estimating SFO/SIO and CFO effects where some pilot subcarriers might have higher SNR (weight).
 * 
 * @return std::pair<float, float> {slope (beta), intercept (alpha)}
 */
template <typename T>
std::pair<float, float> weightedlinearRegression(const std::vector<T>& x_values,
                                         const std::vector<float>& y_values,
                                         const std::vector<float>& weights) {
    if (x_values.size() != y_values.size() || 
        x_values.size() != weights.size() || 
        x_values.empty()) {
        return std::make_pair(0.0f, 0.0f);
    }
    float sum_w = 0.0f, sum_wx = 0.0f, sum_wy = 0.0f;
    float sum_wxx = 0.0f, sum_wxy = 0.0f;
    const int N = x_values.size();

    for (int i = 0; i < N; ++i) {
        const float w = weights[i];
        const float x = static_cast<float>(x_values[i]);
        const float y = y_values[i];
        sum_w += w; sum_wx += w * x; sum_wy += w * y;
        sum_wxx += w * x * x; sum_wxy += w * x * y;
    }

    float beta = 0.0f, alpha = 0.0f;
    float denom = sum_w * sum_wxx - sum_wx * sum_wx;
    if (std::abs(denom) > 1e-10f) {
        beta = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
        alpha = (sum_wy - beta * sum_wx) / sum_w;
    }
    return std::make_pair(beta, alpha);
}

/**
 * @brief Standard Linear Regression.
 * 
 * Calculates the slope (beta) and intercept (alpha) for unweighted data.
 * Used for estimating SFO/SIO across frames using timing offset estimates.
 * 
 * @return std::pair<float, float> {slope (beta), intercept (alpha)}
 */
template <typename T>
std::pair<float, float> linearRegression(const std::vector<T>& x_values,
                                         const std::vector<float>& y_values) {
    if (x_values.size() != y_values.size() || x_values.empty()) {
        return std::make_pair(0.0f, 0.0f);
    }
    float sum_x = 0.0f, sum_y = 0.0f, sum_xx = 0.0f, sum_xy = 0.0f;
    const int N = x_values.size();

    for (int i = 0; i < N; ++i) {
        const float x = static_cast<float>(x_values[i]);
        const float y = y_values[i];
        sum_x += x; sum_y += y; sum_xx += x * x; sum_xy += x * y;
    }

    float beta = 0.0f, alpha = 0.0f;
    float denom = N * sum_xx - sum_x * sum_x;
    if (std::abs(denom) > 1e-10f) {
        beta = (N * sum_xy - sum_x * sum_y) / denom;
        alpha = (sum_y - beta * sum_x) / N;
    }
    return std::make_pair(beta, alpha);
}

/**
 * @brief Finite Impulse Response (FIR) Filter.
 * 
 * Implements a standard FIR filter with a circular buffer for efficiency.
 */
class FIRFilter {
private:
    std::vector<float> coeffs;
    std::vector<float> buffer;
    size_t order;
    size_t index;

public:
    FIRFilter(const std::vector<float>& coefficients) 
        : coeffs(coefficients), 
          buffer(coefficients.size(), 0.0f),
          order(coefficients.size()),
          index(0) {}

    float process(float input) {
        buffer[index] = input;
        index = (index + 1) % order;

        float output = 0.0f;
        size_t i = index;
        for (size_t j = 0; j < order; j++) {
            output += coeffs[j] * buffer[i];
            i = (i + 1) % order;
        }
        return output;
    }
    
    void warm_up(float value, size_t samples = 50) {
        for (size_t i = 0; i < samples; i++) {
            process(value);
        }
    }
};

/**
 * @brief Phase Unwrapping Function.
 * 
 * Unwraps the phase values in a vector to eliminate 2*pi jumps.
 * Essential for accurate frequency offset estimation from phase differences.
 */
inline void unwrap(std::vector<float>& phase) {
    if (phase.size() > 1) {
        std::vector<float> diffs(phase.size());
        
        #pragma omp simd simdlen(16)
        for (size_t i = 1; i < phase.size(); ++i) {
            float d = phase[i] - phase[i - 1];
            float k = std::round(d / (2 * (float)M_PI));
            d -= k * 2 * (float)M_PI;
            diffs[i] = d;
        }

        for (size_t i = 1; i < phase.size(); ++i) {
            phase[i] = phase[i - 1] + diffs[i];
        }
    }
}

// ============== End of Pure Signal Processing Functions ==============

/**
 * @brief Sampling Frequency Offset (SFO) / Sampling Interval Offset (SIO) Estimator.
 * 
 * Estimates the SFO/SIO by tracking the drift of timing offsets over time.
 * Uses a estimation window of timing offset measurements and performs 
 * linear regression to determine the rate of change (SIO). Also 
 * incorporates a control loop to adjust synchronization.
 */
class SFOEstimator {
public:
    explicit SFOEstimator(size_t window_size) 
        : _window_size(window_size),
          _delay_offsets(window_size, 0.0f),
          _delay_offsets_indices(window_size) {
        std::iota(_delay_offsets_indices.begin(), _delay_offsets_indices.end(), 0);
        reset();
    }

    void reset() {
        _count = 0;
        _cumulative_delay_offset = 0.0f;
        _sfo_per_frame = 0.0f;
        std::fill(_delay_offsets.begin(), _delay_offsets.end(), 0.0f);
    }

    // Update delay offset estimation and calculate SFO/SIO
    void update(float delay_offset_reading, float Alignment) {
        if (!_first_delay_offset_reading) {
            if (Alignment != 0.0f) {
                _cumulative_delay_offset += Alignment;
            }
            _delay_offsets[_count] = delay_offset_reading + _cumulative_delay_offset;
            if (++_count >= _window_size) {
                _sfo_per_frame = linearRegression(_delay_offsets_indices, _delay_offsets).first;
                _count = 0;
                _cumulative_delay_offset = 0.0f;
            }
            if (std::abs(_sfo_per_frame) > 1.0f) {
                _sfo_per_frame = 0.0f;
            }
            _cumulative_sensing_delay_offset += _sfo_per_frame;
            _cumulative_sensing_delay_offset -= Alignment;
            auto err = delay_offset_reading - _cumulative_sensing_delay_offset;
            if (std::abs(err) > 0.1f) {
                _err_large_count++;
                if (_err_large_count > 100) {
                    _pd = 1e-2;
                }
            } else {
                _err_large_count = 0;
                _pd = 1e-5;
            }
            _cumulative_sensing_delay_offset += _pd * err;
        }
        if (_first_delay_offset_reading) {
            _count++;
            if (_count >= 10) {
                _first_delay_offset_reading = false;
                _count = 0;
            }
        }
    }

    float get_sfo_per_frame() const { return _sfo_per_frame; }
    float get_sensing_delay_offset() const { return _cumulative_sensing_delay_offset; }

private:
    size_t _window_size;
    size_t _count = 0;
    size_t _err_large_count = 0;
    float _pd = 1e-5;
    
    std::vector<float> _delay_offsets;
    std::vector<int> _delay_offsets_indices;
    
    float _cumulative_delay_offset = 0.0f;
    bool _first_delay_offset_reading = true;
    float _sfo_per_frame = 0.0f;
    float _cumulative_sensing_delay_offset = 0.0f;
};

// ============================================================================
// ARQ Link-Layer Retransmission Helpers
// ============================================================================

/**
 * @brief Modulo-65536 sequence comparison helpers.
 *
 * The mini-header seq is 16-bit. Window logic must stay below half the
 * sequence space (32768).
 */
inline int16_t arq_seq_diff(uint16_t a, uint16_t b) {
    return static_cast<int16_t>(a - b);
}

inline bool arq_seq_leq(uint16_t a, uint16_t b) {
    return arq_seq_diff(a, b) <= 0;
}

/**
 * @brief ARQ feedback packet identification and payload.
 *
 * Feedback packets are carried as regular LDPC payloads over the air link.
 * They begin with the 4-byte magic "ARQ1" followed by:
 *   [4] direction (uint8_t): 0=downlink, 1=uplink
 *   [5..6] ack_base (uint16_t LE): all seq < ack_base in modulo order are ACKed
 *   [7..14] ack_bitmap (uint64_t LE): bit i = 1 => (ack_base + i) is ACKed
 * Total feedback payload: 15 bytes.
 */
struct ArqFeedback {
    static constexpr size_t kMagicLen = 4;
    static constexpr char kMagic[kMagicLen + 1] = "ARQ1";
    static constexpr size_t kPayloadSize = 15; // 4 magic + 1 dir + 2 base + 8 bitmap

    uint8_t direction = 0;     // 0=downlink, 1=uplink
    uint16_t ack_base = 0;
    uint64_t ack_bitmap = 0;

    /** Pack into a byte buffer suitable for LDPC encoding. */
    void pack(std::vector<uint8_t>& out) const {
        out.resize(kPayloadSize);
        std::memcpy(out.data(), kMagic, kMagicLen);
        out[4] = direction;
        out[5] = static_cast<uint8_t>(ack_base & 0xFFu);
        out[6] = static_cast<uint8_t>((ack_base >> 8) & 0xFFu);
        for (int i = 0; i < 8; ++i) {
            out[7 + i] = static_cast<uint8_t>((ack_bitmap >> (i * 8)) & 0xFFu);
        }
    }

    /** Try to unpack from decoded payload bytes. Returns true on success. */
    static bool try_unpack(const uint8_t* data, size_t len, ArqFeedback& out) {
        if (len < kPayloadSize) return false;
        if (std::memcmp(data, kMagic, kMagicLen) != 0) return false;
        out.direction = data[4];
        if (out.direction > 1) return false;
        out.ack_base = static_cast<uint16_t>(data[5]) | (static_cast<uint16_t>(data[6]) << 8);
        out.ack_bitmap = 0;
        for (int i = 0; i < 8; ++i) {
            out.ack_bitmap |= static_cast<uint64_t>(data[7 + i]) << (i * 8);
        }
        return true;
    }

    /** Quick check: does this payload look like an ARQ feedback packet? */
    static bool is_feedback(const uint8_t* data, size_t len) {
        ArqFeedback ignored;
        return try_unpack(data, len, ignored);
    }
};

/**
 * @brief ARQ transmit-side window entry.
 *
 * Stores the raw payload and optionally the pre-encoded QPSK symbols for
 * cheap retransmission.
 */
struct ArqTxEntry {
    uint16_t seq = 0;
    int64_t last_tx_time_ms = 0;
    int retry_count = 0;
    std::vector<uint8_t> raw_payload;
    // Pre-encoded QPSK symbols (control + payload), stored for cheap retransmit.
    // When non-empty, retransmission can skip LDPC encode.
    AlignedIntVector encoded_qpsk;
};

/**
 * @brief Bounded ARQ transmit window.
 *
 * Tracks outstanding unacked packets. Prioritizes retransmissions over new
 * packets. Releases entries on ACK. Retransmits entries whose RTO has expired.
 */
class ArqTxWindow {
public:
    ArqTxWindow() = default;

    void configure(const NetworkOutputConfig& net) {
        _window_size = static_cast<uint16_t>(
            std::max(1, std::min<int>(net.arq_window_packets, 64)));
        _rto_ms = net.arq_retransmit_timeout_ms > 0
            ? net.arq_retransmit_timeout_ms : 10;
        _max_retries = net.arq_max_retries;
        _direction = 0; // set by caller
    }

    void set_direction(uint8_t dir) { _direction = dir; }

    /** Insert a new packet. Returns false if window is full (backpressure). */
    bool try_insert(uint16_t seq, const uint8_t* payload, size_t len,
                    int64_t now_ms) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!has_room_unlocked()) return false;
        ArqTxEntry entry;
        entry.seq = seq;
        entry.last_tx_time_ms = now_ms;
        entry.retry_count = 0;
        entry.raw_payload.assign(payload, payload + len);
        _entries.emplace(seq, std::move(entry));
        return true;
    }

    /** Insert with pre-encoded QPSK for cheap retransmission. */
    bool try_insert_encoded(uint16_t seq, const uint8_t* payload, size_t len,
                            AlignedIntVector encoded_qpsk, int64_t now_ms) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!has_room_unlocked()) return false;
        ArqTxEntry entry;
        entry.seq = seq;
        entry.last_tx_time_ms = now_ms;
        entry.retry_count = 0;
        entry.raw_payload.assign(payload, payload + len);
        entry.encoded_qpsk = std::move(encoded_qpsk);
        _entries.emplace(seq, std::move(entry));
        return true;
    }

    /**
     * Process an ACK feedback. Releases all acknowledged entries.
     * Returns number of entries released.
     */
    size_t process_ack(const ArqFeedback& ack, int64_t now_ms) {
        (void)now_ms;
        std::lock_guard<std::mutex> lock(_mutex);
        if (ack.direction != _direction) {
            return 0;
        }
        size_t released = 0;
        // Release all seqs < ack_base (cumulative)
        auto it = _entries.begin();
        while (it != _entries.end()) {
            if (arq_seq_diff(it->first, ack.ack_base) < 0) {
                it = _entries.erase(it);
                ++released;
            } else {
                ++it;
            }
        }
        // Release selective bitmap entries
        const uint16_t base = ack.ack_base;
        for (int i = 0; i < 64; ++i) {
            if (ack.ack_bitmap & (static_cast<uint64_t>(1) << i)) {
                uint16_t acked_seq = static_cast<uint16_t>(base + i);
                auto found = _entries.find(acked_seq);
                if (found != _entries.end()) {
                    _entries.erase(found);
                    ++released;
                }
            }
        }
        return released;
    }

    /**
     * Collect seqs that need retransmission (RTO expired).
     * Returns at most max_count seq numbers.
     */
    void get_retransmit(std::vector<uint16_t>& out, int64_t now_ms,
                        size_t max_count = 16) const {
        std::lock_guard<std::mutex> lock(_mutex);
        out.clear();
        for (const auto& [seq, entry] : _entries) {
            if (out.size() >= max_count) break;
            if ((now_ms - entry.last_tx_time_ms) >= _rto_ms) {
                if (_max_retries <= 0 || entry.retry_count < _max_retries) {
                    out.push_back(seq);
                }
            }
        }
    }

    /** Mark a seq as just transmitted (update timestamp, bump retry count). */
    void mark_transmitted(uint16_t seq, int64_t now_ms) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _entries.find(seq);
        if (it != _entries.end()) {
            it->second.last_tx_time_ms = now_ms;
            it->second.retry_count++;
        }
    }

    bool get_entry_copy(uint16_t seq, ArqTxEntry& out) const {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _entries.find(seq);
        if (it == _entries.end()) return false;
        out = it->second;
        return true;
    }

    bool has_room() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return has_room_unlocked();
    }

    size_t outstanding_count() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _entries.size();
    }

    bool has_entry(uint16_t seq) const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _entries.find(seq) != _entries.end();
    }

    bool has_room_unlocked() const {
        return _entries.size() < static_cast<size_t>(_window_size);
    }

    uint16_t window_size() const { return _window_size; }
    int rto_ms() const { return _rto_ms; }

    /** Drop entries that exceeded max_retries (if nonzero). Returns count. */
    size_t drop_abandoned(int64_t now_ms) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_max_retries <= 0) return 0;
        size_t dropped = 0;
        auto it = _entries.begin();
        while (it != _entries.end()) {
            if (it->second.retry_count >= _max_retries &&
                (now_ms - it->second.last_tx_time_ms) >= _rto_ms) {
                it = _entries.erase(it);
                ++dropped;
            } else {
                ++it;
            }
        }
        return dropped;
    }

    /** Check if any retransmissions are pending. */
    bool has_retransmit_pending(int64_t now_ms) const {
        std::lock_guard<std::mutex> lock(_mutex);
        for (const auto& [seq, entry] : _entries) {
            if ((now_ms - entry.last_tx_time_ms) >= _rto_ms) {
                if (_max_retries <= 0 || entry.retry_count < _max_retries) {
                    return true;
                }
            }
        }
        return false;
    }

private:
    uint16_t _window_size = 256;
    int _rto_ms = 10;
    int _max_retries = 0;
    uint8_t _direction = 0;
    mutable std::mutex _mutex;
    std::unordered_map<uint16_t, ArqTxEntry> _entries;
};

/**
 * @brief Bounded ARQ receive window with duplicate suppression and
 * optional ordered delivery.
 *
 * Tracks received sequence numbers, suppresses duplicates, and generates
 * cumulative + selective ACK feedback.
 */
class ArqRxWindow {
public:
    ArqRxWindow() = default;

    void configure(const NetworkOutputConfig& net) {
        _window_size = static_cast<uint16_t>(
            std::max(1, std::min<int>(net.arq_window_packets, 64)));
        _ordered = net.arq_ordered_delivery;
        _feedback_interval_ms = net.arq_feedback_interval_ms;
        _max_reorder_buf = static_cast<size_t>(_window_size);
        _got_any = false;
        _expected_seq = 0;
        _ack_base = 0;
        _ack_bitmap = 0;
        _reorder_buffer.clear();
    }

    void set_direction(uint8_t dir) { _feedback_dir = dir; }

    /**
     * Process a received data packet.
     * Returns true if this is a new (non-duplicate) packet accepted.
     * Returns false if duplicate or outside window.
     *
     * For ordered delivery: the payload is buffered until it can be
     * delivered contiguously.
     * For unordered delivery: the caller should forward the payload
     * immediately when this returns true.
     */
    bool process_received(uint16_t seq, const uint8_t* payload, size_t len) {
        if (!_got_any) {
            _got_any = true;
        }

        const int16_t bit_idx = arq_seq_diff(seq, _ack_base);
        if (bit_idx < 0) {
            _dup_count++;
            return false;
        }
        if (bit_idx >= static_cast<int16_t>(_window_size) || bit_idx >= 64) {
            return false;
        }
        const uint64_t bit = static_cast<uint64_t>(1) << bit_idx;
        if (_ack_bitmap & bit) {
            _dup_count++;
            return false;
        }
        _ack_bitmap |= bit;
        _advance_ack_base();

        _accepted_count++;

        if (_ordered) {
            // Buffer for ordered delivery
            _reorder_buffer.emplace(seq, std::vector<uint8_t>(payload, payload + len));
            if (_reorder_buffer.size() > _max_reorder_buf) {
                _reorder_buffer.erase(_reorder_buffer.begin());
            }
        }

        return true;
    }

    /**
     * For ordered delivery: collect packets that can be delivered
     * contiguously from _expected_seq.
     * Returns payloads in order.
     */
    void get_deliverable(std::vector<std::vector<uint8_t>>& out) {
        out.clear();
        if (!_ordered) return;
        while (true) {
            auto it = _reorder_buffer.find(_expected_seq);
            if (it == _reorder_buffer.end()) break;
            out.push_back(std::move(it->second));
            _reorder_buffer.erase(it);
            _expected_seq = static_cast<uint16_t>(_expected_seq + 1);
        }
    }

    /**
     * Generate an ACK feedback packet for the current receive state.
     */
    ArqFeedback generate_ack() const {
        ArqFeedback ack;
        ack.direction = _feedback_dir;
        ack.ack_base = _ack_base;
        ack.ack_bitmap = _ack_bitmap;
        return ack;
    }

    /**
     * Check if enough time has elapsed since the last ACK was sent.
     */
    bool should_send_ack(int64_t now_ms) const {
        if (!_got_any) return false;
        if (_last_ack_time_ms == 0) return true;
        return (now_ms - _last_ack_time_ms) >= _feedback_interval_ms;
    }

    void mark_ack_sent(int64_t now_ms) {
        _last_ack_time_ms = now_ms;
    }

    bool got_any() const { return _got_any; }
    uint16_t expected_seq() const { return _expected_seq; }
    uint64_t dup_count() const { return _dup_count; }
    uint64_t accepted_count() const { return _accepted_count; }

private:
    void _advance_ack_base() {
        while (_ack_bitmap & 0x1u) {
            _ack_bitmap >>= 1;
            _ack_base = static_cast<uint16_t>(_ack_base + 1);
        }
    }

    bool _got_any = false;
    bool _ordered = false;
    uint16_t _expected_seq = 0;
    uint16_t _ack_base = 0;
    uint64_t _ack_bitmap = 0;
    uint16_t _window_size = 256;
    size_t _max_reorder_buf = 256;
    int _feedback_interval_ms = 2;
    uint8_t _feedback_dir = 0;
    int64_t _last_ack_time_ms = 0;
    uint64_t _dup_count = 0;
    uint64_t _accepted_count = 0;
    std::map<uint16_t, std::vector<uint8_t>> _reorder_buffer;
};

// Convenience: current time in milliseconds for ARQ timestamps
inline int64_t arq_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}



#endif // OFDM_CORE_HPP
