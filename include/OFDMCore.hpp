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

struct NormalCfoDealiasResult {
    bool valid = false;
    double modulo_hz = 0.0;
    double dealiased_hz = 0.0;
    int64_t alias_index = 0;
};

/** Resolve adjacent-symbol CFO onto the branch nearest the current tune offset. */
inline NormalCfoDealiasResult dealias_normal_cfo_hz(
    double modulo_cfo_hz,
    double alias_period_hz,
    double reference_cfo_hz)
{
    NormalCfoDealiasResult out;
    out.modulo_hz = modulo_cfo_hz;
    if (!std::isfinite(modulo_cfo_hz) ||
        !std::isfinite(alias_period_hz) || alias_period_hz <= 0.0 ||
        !std::isfinite(reference_cfo_hz)) {
        return out;
    }
    const double branch =
        std::round((reference_cfo_hz - modulo_cfo_hz) / alias_period_hz);
    if (branch < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
        branch > static_cast<double>(std::numeric_limits<int64_t>::max())) {
        return out;
    }
    out.alias_index = static_cast<int64_t>(branch);
    out.dealiased_hz =
        modulo_cfo_hz + branch * alias_period_hz;
    out.valid = std::isfinite(out.dealiased_hz);
    return out;
}

/**
 * @brief Track residual CFO across acquisition, phase wrapping and RX retunes.
 *
 * The tune offset alone is not a residual-CFO reference: an external LO or
 * oscillator error can be much larger than the adjacent-symbol alias period.
 * Anchor the reference to a full acquisition estimate, then translate it by
 * the actual tune-chain change. Only averaged, valid observations advance it.
 * Owned by the demod/control thread; no allocation or additional IQ processing.
 */
class NormalCfoTracker {
public:
    /** @brief Invalidate the old acquisition before searching for a new frame. */
    void reset() { _valid = false; }

    /** @brief Anchor full acquisition or averaged CFO to its capture tune. */
    void acquire(double cfo_hz, double tune_cfo_hz) {
        _valid = std::isfinite(cfo_hz) && std::isfinite(tune_cfo_hz);
        _reference_hz = cfo_hz;
        _reference_tune_hz = tune_cfo_hz;
    }

    /**
     * @brief Resolve fresh modulo CFO without letting single frames move the anchor.
     * @param observation_valid False for stale or invalid demod results; these
     * leave the reference unchanged and return an invalid result.
     */
    NormalCfoDealiasResult resolve(double modulo_hz, double alias_period_hz,
                                 double tune_cfo_hz, bool observation_valid) const {
        if (!_valid || !observation_valid || !std::isfinite(tune_cfo_hz)) {
            return {};
        }
        const double expected_hz =
            _reference_hz + (tune_cfo_hz - _reference_tune_hz);
        return dealias_normal_cfo_hz(modulo_hz, alias_period_hz, expected_hz);
    }

private:
    bool _valid = false;
    double _reference_hz = 0.0;
    double _reference_tune_hz = 0.0;
};

/** Require a short continuous run before de-aliased normal CFO drives OCXO. */
class NormalCfoQualityGate {
public:
    explicit NormalCfoQualityGate(
        size_t min_consecutive = 3,
        double max_step_fraction = 0.25)
        : _min_consecutive(std::max<size_t>(min_consecutive, 1)),
          _max_step_fraction(std::max(max_step_fraction, 0.0)) {}

    void reset() {
        _has_previous = false;
        _previous_hz = 0.0;
        _consecutive = 0;
    }

    bool update(const NormalCfoDealiasResult& estimate,
                double alias_period_hz,
                bool upstream_valid = true) {
        if (!upstream_valid || !estimate.valid ||
            !std::isfinite(estimate.dealiased_hz) ||
            !std::isfinite(alias_period_hz) || alias_period_hz <= 0.0) {
            reset();
            return false;
        }
        const double max_step_hz = _max_step_fraction * alias_period_hz;
        if (!_has_previous ||
            std::abs(estimate.dealiased_hz - _previous_hz) > max_step_hz) {
            _consecutive = 1;
        } else {
            ++_consecutive;
        }
        _previous_hz = estimate.dealiased_hz;
        _has_previous = true;
        return _consecutive >= _min_consecutive;
    }

private:
    size_t _min_consecutive = 3;
    double _max_step_fraction = 0.25;
    bool _has_previous = false;
    double _previous_hz = 0.0;
    size_t _consecutive = 0;
};

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
    uint16_t seq = 0;  ///< ARQ / packet sequence (data path). Not air-frame index.
    // Tier-1 Air-Frame Identity SIB (two BCH words after one marker):
    //   word0: version4|flags4|epoch_lo16|seq_lo24|CRC16  (flag = AirFrameIdentity)
    //   word1: version4|flags4|epoch_hi16|seq_hi8|pad8|CRC16 (flag = AirFrameIdentityHi)
    // Combined: epoch32 (OTA canonical) + frame_seq32.
    uint32_t timeline_epoch = 0;  ///< OTA epoch32 (matches AirFrameIdentity.timeline_epoch)
    uint32_t air_frame_seq = 0;   ///< Full 32-bit air frame within epoch
};

struct ErtmTimingPayloadView {
    uint32_t fft_size = 0;
    double sample_rate = 0.0;
    int32_t duti_samples = 0;              ///< DUTI applied on UL measure frame M
    uint32_t seq = 0;                      ///< Diagnostic / de-dup only (not air frame)
    uint32_t timeline_epoch = 0;           ///< Canonical OTA epoch32 of measure frame M
    uint64_t ul_measure_frame_seq = 0;     ///< Absolute air-frame index M at BS UL-RX
    bool bs_pps_anchor_valid = false;       ///< BS frame-0/PPS anchor is usable
    int64_t bs_last_pps_ticks = -1;         ///< Raw BS time_last_pps() in BS clock ticks
    int64_t bs_t0_to_pps_ticks = 0;        ///< wrap_1s(B_0 - P_B), in BS clock ticks
    int64_t bs_frame_ticks = 0;             ///< BS air-frame period in clock ticks
    double bs_tick_rate = 0.0;              ///< BS clock tick rate in ticks/s
    AlignedVector channel_freq;
};

struct ErtmPpsPhaseObservation {
    bool valid = false;
    double r_bs_samples = 0.0;
    double r_ue_samples = 0.0;
    double phase_error_samples = 0.0; ///< wrap_1s(P_U - P_B), target zero
    int64_t bs_last_pps_ticks = -1;
    int64_t bs_t0_to_pps_ticks = -1;
    int64_t ue_last_pps_ns = -1;
    double bs_tick_rate = 0.0;
    double sample_rate = 0.0;
};

inline uint64_t add_mod_u64(uint64_t lhs, uint64_t rhs, uint64_t modulus) {
    if (modulus == 0) {
        return 0;
    }
    lhs %= modulus;
    rhs %= modulus;
    return lhs >= modulus - rhs
        ? lhs - (modulus - rhs)
        : lhs + rhs;
}

inline uint64_t mul_mod_u64(uint64_t lhs, uint64_t rhs, uint64_t modulus) {
    if (modulus == 0) {
        return 0;
    }
    uint64_t result = 0;
    lhs %= modulus;
    while (rhs != 0) {
        if ((rhs & 1u) != 0u) {
            result = add_mod_u64(result, lhs, modulus);
        }
        rhs >>= 1u;
        if (rhs != 0) {
            lhs = add_mod_u64(lhs, lhs, modulus);
        }
    }
    return result;
}

inline uint64_t signed_mod_i64(int64_t value, uint64_t modulus) {
    if (modulus == 0) {
        return 0;
    }
    if (value >= 0) {
        return static_cast<uint64_t>(value) % modulus;
    }
    const uint64_t magnitude = static_cast<uint64_t>(-(value + 1)) + 1u;
    const uint64_t remainder = magnitude % modulus;
    return remainder == 0 ? 0 : modulus - remainder;
}

/**
 * @brief Wrap a sample-domain interval into [-0.5 s, 0.5 s).
 */
inline double wrap_pps_phase_samples(double samples, double sample_rate) {
    if (!(sample_rate > 0.0) || !std::isfinite(sample_rate) ||
        !std::isfinite(samples)) {
        return 0.0;
    }
    double wrapped = std::fmod(samples + 0.5 * sample_rate, sample_rate);
    if (wrapped < 0.0) {
        wrapped += sample_rate;
    }
    return wrapped - 0.5 * sample_rate;
}

/**
 * @brief Compute the UE-minus-BS PPS phase from one same-frame observation.
 *
 * For downlink frame N:
 *   P_U - P_B = wrap_1s(tau_DL_RF - TO_UE + r_B(N) - r_U(N)).
 * The BS anchor carried by each ERTM5 report is refreshed from time_last_pps()
 * before that report is packed and is only used to reconstruct r_B(N). This
 * function never reanchors either radio timeline or TO state.
 */
inline ErtmPpsPhaseObservation compute_ertm_pps_phase(
    const ErtmTimingPayloadView& report,
    uint64_t dl_frame_seq,
    int64_t ue_frame_boundary_ns,
    int64_t ue_last_pps_ns,
    double tau_dl_rf_samples,
    double to_ue_samples)
{
    ErtmPpsPhaseObservation out;
    if (!report.bs_pps_anchor_valid || report.bs_frame_ticks <= 0 ||
        !(report.bs_tick_rate > 0.0) || !std::isfinite(report.bs_tick_rate) ||
        !(report.sample_rate > 0.0) || !std::isfinite(report.sample_rate) ||
        ue_frame_boundary_ns < 0 || ue_last_pps_ns < 0 ||
        !std::isfinite(tau_dl_rf_samples) || !std::isfinite(to_ue_samples)) {
        return out;
    }

    if (report.bs_tick_rate >
        static_cast<double>(std::numeric_limits<int64_t>::max())) {
        return out;
    }
    const int64_t rounded_tick_rate =
        static_cast<int64_t>(std::llround(report.bs_tick_rate));
    if (rounded_tick_rate <= 0 ||
        std::abs(report.bs_tick_rate - static_cast<double>(rounded_tick_rate)) >
            0.5) {
        return out;
    }
    const uint64_t second_ticks = static_cast<uint64_t>(rounded_tick_rate);
    const uint64_t frame_phase_ticks = mul_mod_u64(
        dl_frame_seq,
        static_cast<uint64_t>(report.bs_frame_ticks) % second_ticks,
        second_ticks);
    const uint64_t r_bs_ticks = add_mod_u64(
        signed_mod_i64(report.bs_t0_to_pps_ticks, second_ticks),
        frame_phase_ticks,
        second_ticks);
    constexpr uint64_t kNanosecondsPerSecond = 1'000'000'000u;
    const uint64_t ue_boundary_phase_ns =
        signed_mod_i64(ue_frame_boundary_ns, kNanosecondsPerSecond);
    const uint64_t ue_pps_phase_ns =
        signed_mod_i64(ue_last_pps_ns, kNanosecondsPerSecond);
    const uint64_t r_ue_ns = add_mod_u64(
        ue_boundary_phase_ns,
        ue_pps_phase_ns == 0 ? 0 : kNanosecondsPerSecond - ue_pps_phase_ns,
        kNanosecondsPerSecond);

    out.r_bs_samples = static_cast<double>(
        static_cast<long double>(r_bs_ticks) *
        static_cast<long double>(report.sample_rate) /
        static_cast<long double>(second_ticks));
    out.r_ue_samples = static_cast<double>(
        static_cast<long double>(r_ue_ns) *
        static_cast<long double>(report.sample_rate) /
        static_cast<long double>(kNanosecondsPerSecond));
    out.bs_last_pps_ticks = report.bs_last_pps_ticks;
    out.bs_t0_to_pps_ticks = report.bs_t0_to_pps_ticks;
    out.ue_last_pps_ns = ue_last_pps_ns;
    out.bs_tick_rate = report.bs_tick_rate;
    out.sample_rate = report.sample_rate;
    out.phase_error_samples = wrap_pps_phase_samples(
        tau_dl_rf_samples - to_ue_samples +
            out.r_bs_samples - out.r_ue_samples,
        report.sample_rate);
    out.valid = true;
    return out;
}

/**
 * @brief eRTM UL channel report wire format (ERTM5).
 *
 * Layout (little-endian):
 *   magic[5]='ERTM5'
 *   fft_size u32, sample_rate f64, duti_samples i32, seq u32,
 *   timeline_epoch u32, ul_measure_frame_seq u64,
 *   pps_anchor_flags u32, bs_last_pps_ticks i64, bs_t0_to_pps_ticks i64,
 *   bs_frame_ticks i64, bs_tick_rate f64,
 *   channel_freq[fft_size] as interleaved f32 re/im.
 *
 * ERTM2/3/4 are rejected: ERTM2 lacks absolute frame identity, ERTM3 lacks
 * the BS PPS/frame-epoch anchor, and ERTM4 lacks the raw BS PPS timestamp
 * required to diagnose per-second PPS-anchor discontinuities.
 */
class ErtmTimingPayload {
public:
    static constexpr uint8_t kMagic[5] = {'E', 'R', 'T', 'M', '5'};
    static constexpr uint8_t kMagicV2[5] = {'E', 'R', 'T', 'M', '2'};
    static constexpr uint8_t kMagicV3[5] = {'E', 'R', 'T', 'M', '3'};
    static constexpr uint8_t kMagicV4[5] = {'E', 'R', 'T', 'M', '4'};
    static constexpr uint32_t kPpsAnchorValid = 0x01u;
    // Base ERTM3 header (37) + PPS flags/raw/t0/frame/rate (36).
    static constexpr size_t kHeaderBytes =
        5 + 4 + 8 + 4 + 4 + 4 + 8 + 4 + 8 + 8 + 8 + 8;

    static bool looks_like(const uint8_t* data, size_t len) {
        return data != nullptr && len >= sizeof(kMagic) &&
               std::memcmp(data, kMagic, sizeof(kMagic)) == 0;
    }

    static bool looks_like_v2(const uint8_t* data, size_t len) {
        return data != nullptr && len >= sizeof(kMagicV2) &&
               std::memcmp(data, kMagicV2, sizeof(kMagicV2)) == 0;
    }

    static bool looks_like_v3(const uint8_t* data, size_t len) {
        return data != nullptr && len >= sizeof(kMagicV3) &&
               std::memcmp(data, kMagicV3, sizeof(kMagicV3)) == 0;
    }

    static bool looks_like_v4(const uint8_t* data, size_t len) {
        return data != nullptr && len >= sizeof(kMagicV4) &&
               std::memcmp(data, kMagicV4, sizeof(kMagicV4)) == 0;
    }

    static bool pack(uint32_t seq,
                     uint32_t fft_size,
                     double sample_rate,
                     int32_t duti_samples,
                     uint32_t timeline_epoch,
                     uint64_t ul_measure_frame_seq,
                     int64_t bs_last_pps_ticks,
                     int64_t bs_t0_to_pps_ticks,
                     int64_t bs_frame_ticks,
                     double bs_tick_rate,
                     const AlignedVector& channel_freq,
                     std::vector<uint8_t>& out,
                     std::string* error = nullptr) {
        auto set_error = [error](const std::string& msg) {
            if (error) *error = msg;
        };
        if (fft_size == 0 || channel_freq.size() != fft_size) {
            set_error("channel frequency response size does not match fft_size");
            return false;
        }
        if (!(sample_rate > 0.0) || !std::isfinite(sample_rate)) {
            set_error("sample_rate must be finite and positive");
            return false;
        }
        if (bs_last_pps_ticks < 0 || bs_t0_to_pps_ticks < 0 || bs_frame_ticks <= 0 ||
            !(bs_tick_rate > 0.0) || !std::isfinite(bs_tick_rate)) {
            set_error(
                "BS PPS ticks must be non-negative and frame/tick rates "
                "must be finite and positive");
            return false;
        }
        const size_t total = kHeaderBytes + static_cast<size_t>(fft_size) * sizeof(float) * 2;
        if (total > std::numeric_limits<uint16_t>::max()) {
            set_error("eRTM payload exceeds LDPC mini-header payload limit");
            return false;
        }

        out.clear();
        out.reserve(total);
        out.insert(out.end(), std::begin(kMagic), std::end(kMagic));
        write_u32(out, fft_size);
        write_double(out, sample_rate);
        write_i32(out, duti_samples);
        write_u32(out, seq);
        write_u32(out, timeline_epoch);
        write_u64(out, ul_measure_frame_seq);
        write_u32(out, kPpsAnchorValid);
        write_i64(out, bs_last_pps_ticks);
        write_i64(out, bs_t0_to_pps_ticks);
        write_i64(out, bs_frame_ticks);
        write_double(out, bs_tick_rate);
        for (const auto& v : channel_freq) {
            write_float(out, v.real());
            write_float(out, v.imag());
        }
        return true;
    }

    /**
     * @brief Parse ERTM5 scalar header fields without touching @p out.channel_freq.
     *
     * Used to select a reusable cache slot before channel unpack so the slot's
     * existing channel buffer capacity can be preserved.
     */
    static bool peek(const uint8_t* data,
                     size_t len,
                     uint32_t expected_fft_size,
                     ErtmTimingPayloadView& out,
                     std::string* error = nullptr) {
        auto set_error = [error](const std::string& msg) {
            if (error) *error = msg;
        };
        if ((looks_like_v2(data, len) || looks_like_v3(data, len) ||
             looks_like_v4(data, len)) &&
            !looks_like(data, len)) {
            set_error("rejected legacy ERTM2/3/4 payload (raw PPS diagnostics required)");
            return false;
        }
        if (!looks_like(data, len)) {
            set_error("missing ERTM5 magic");
            return false;
        }
        if (len < kHeaderBytes) {
            set_error("payload shorter than ERTM5 header");
            return false;
        }

        size_t off = sizeof(kMagic);
        const uint32_t fft_size = read_u32(data, off);
        off += 4;
        const double sample_rate = read_double(data, off);
        off += 8;
        const int32_t duti_samples = read_i32(data, off);
        off += 4;
        const uint32_t seq = read_u32(data, off);
        off += 4;
        const uint32_t timeline_epoch = read_u32(data, off);
        off += 4;
        const uint64_t ul_measure_frame_seq = read_u64(data, off);
        off += 8;
        const uint32_t pps_anchor_flags = read_u32(data, off);
        off += 4;
        const int64_t bs_last_pps_ticks = read_i64(data, off);
        off += 8;
        const int64_t bs_t0_to_pps_ticks = read_i64(data, off);
        off += 8;
        const int64_t bs_frame_ticks = read_i64(data, off);
        off += 8;
        const double bs_tick_rate = read_double(data, off);
        off += 8;

        if (fft_size == 0 || fft_size != expected_fft_size) {
            set_error("ERTM5 fft_size mismatch");
            return false;
        }
        if (!(sample_rate > 0.0) || !std::isfinite(sample_rate)) {
            set_error("ERTM5 sample_rate must be finite and positive");
            return false;
        }
        if ((pps_anchor_flags & kPpsAnchorValid) == 0u ||
            bs_last_pps_ticks < 0 || bs_t0_to_pps_ticks < 0 || bs_frame_ticks <= 0 ||
            !(bs_tick_rate > 0.0) || !std::isfinite(bs_tick_rate)) {
            set_error("ERTM5 BS PPS/frame anchor is invalid");
            return false;
        }
        const size_t expected_len = kHeaderBytes + static_cast<size_t>(fft_size) * sizeof(float) * 2;
        if (len != expected_len) {
            set_error("ERTM5 payload length mismatch");
            return false;
        }

        out.fft_size = fft_size;
        out.sample_rate = sample_rate;
        out.duti_samples = duti_samples;
        out.seq = seq;
        out.timeline_epoch = timeline_epoch;
        out.ul_measure_frame_seq = ul_measure_frame_seq;
        out.bs_pps_anchor_valid = true;
        out.bs_last_pps_ticks = bs_last_pps_ticks;
        out.bs_t0_to_pps_ticks = bs_t0_to_pps_ticks;
        out.bs_frame_ticks = bs_frame_ticks;
        out.bs_tick_rate = bs_tick_rate;
        return true;
    }

    /**
     * @brief Unpack ERTM5 into @p out, reusing @p out.channel_freq capacity when size matches.
     */
    static bool unpack(const uint8_t* data,
                       size_t len,
                       uint32_t expected_fft_size,
                       ErtmTimingPayloadView& out,
                       std::string* error = nullptr) {
        if (!peek(data, len, expected_fft_size, out, error)) {
            return false;
        }
        // Capacity-preserving resize: no allocation when already sized to fft_size.
        if (out.channel_freq.size() != out.fft_size) {
            out.channel_freq.resize(out.fft_size);
        }
        size_t off = kHeaderBytes;
        for (uint32_t i = 0; i < out.fft_size; ++i) {
            const float re = read_float(data, off);
            off += 4;
            const float im = read_float(data, off);
            off += 4;
            out.channel_freq[i] = std::complex<float>(re, im);
        }
        return true;
    }

private:
    static void write_u32(std::vector<uint8_t>& out, uint32_t value) {
        out.push_back(static_cast<uint8_t>(value & 0xFFu));
        out.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
        out.push_back(static_cast<uint8_t>((value >> 16) & 0xFFu));
        out.push_back(static_cast<uint8_t>((value >> 24) & 0xFFu));
    }

    static void write_u64(std::vector<uint8_t>& out, uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            out.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFFu));
        }
    }

    static void write_i32(std::vector<uint8_t>& out, int32_t value) {
        write_u32(out, static_cast<uint32_t>(value));
    }

    static void write_i64(std::vector<uint8_t>& out, int64_t value) {
        write_u64(out, static_cast<uint64_t>(value));
    }

    static void write_float(std::vector<uint8_t>& out, float value) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "Unexpected float size");
        std::memcpy(&bits, &value, sizeof(bits));
        write_u32(out, bits);
    }

    static void write_double(std::vector<uint8_t>& out, double value) {
        uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "Unexpected double size");
        std::memcpy(&bits, &value, sizeof(bits));
        for (int i = 0; i < 8; ++i) {
            out.push_back(static_cast<uint8_t>((bits >> (i * 8)) & 0xFFu));
        }
    }

    static uint32_t read_u32(const uint8_t* data, size_t off) {
        return static_cast<uint32_t>(data[off]) |
               (static_cast<uint32_t>(data[off + 1]) << 8) |
               (static_cast<uint32_t>(data[off + 2]) << 16) |
               (static_cast<uint32_t>(data[off + 3]) << 24);
    }

    static uint64_t read_u64(const uint8_t* data, size_t off) {
        uint64_t bits = 0;
        for (int i = 0; i < 8; ++i) {
            bits |= static_cast<uint64_t>(data[off + static_cast<size_t>(i)]) << (i * 8);
        }
        return bits;
    }

    static int32_t read_i32(const uint8_t* data, size_t off) {
        return static_cast<int32_t>(read_u32(data, off));
    }

    static int64_t read_i64(const uint8_t* data, size_t off) {
        return static_cast<int64_t>(read_u64(data, off));
    }

    static float read_float(const uint8_t* data, size_t off) {
        const uint32_t bits = read_u32(data, off);
        float value = 0.0f;
        static_assert(sizeof(bits) == sizeof(value), "Unexpected float size");
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    static double read_double(const uint8_t* data, size_t off) {
        uint64_t bits = 0;
        for (int i = 0; i < 8; ++i) {
            bits |= static_cast<uint64_t>(data[off + static_cast<size_t>(i)]) << (i * 8);
        }
        double value = 0.0;
        static_assert(sizeof(bits) == sizeof(value), "Unexpected double size");
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
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
    // The mini-header payload length is 16 bits; cover even the largest UDP datagram.
    static constexpr size_t kMaxPayloadCodeBits = ((65535 + 63 - 1) / 63) * 1008;
    /** @brief Preserve the legacy QPSK scrambler span; new modes cover full packets. */
    static size_t scrambler_bits(const DigitalModulation& modulation) {
        return modulation.is_qpsk() ? 201600 : kMaxPayloadCodeBits;
    }
    static constexpr size_t kLdpcQpskSymbolsPerBlock = kLdpcCodeBitsPerBlock / 2;
    static constexpr uint8_t kVersion = 1;
    static constexpr uint8_t kFlags = 0;
    // Mini-header flag bits (4-bit field). Data frames use 0; control/internal
    // frames set a dedicated flag so receivers classify them by header, not by
    // sniffing payload bytes. Unknown flag bits are rejected on decode.
    static constexpr uint8_t kFlagArqFeedback = 0x01;
    static constexpr uint8_t kFlagErtmTiming = 0x02;
    /// Tier-1 Air-Frame Identity SIB word0 (BCH mini-header only, no LDPC payload).
    static constexpr uint8_t kFlagAirFrameIdentity = 0x04;
    /// Second BCH word of the expanded Identity SIB (seq high bits).
    static constexpr uint8_t kFlagAirFrameIdentityHi = 0x08;
    static constexpr uint8_t kKnownFlagsMask = 0x0F;
    /// Marker + two mini-headers (epoch32 + full frame_seq32).
    static constexpr size_t kIdentityMiniHeaderWords = 2;
    static constexpr size_t kIdentityControlSymbols =
        kMarkerSymbols + kIdentityMiniHeaderWords * kMiniHeaderSymbols;
    static constexpr size_t kIdentityControlBits = kIdentityControlSymbols * 2;
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

    static bool is_ertm_timing_flags(uint8_t flags) {
        return (flags & kFlagErtmTiming) != 0;
    }

    static bool is_ertm_timing(const LdpcMiniHeader& header) {
        return is_ertm_timing_flags(header.flags);
    }

    static bool is_air_frame_identity_flags(uint8_t flags) {
        // Word0 only — word1 uses kFlagAirFrameIdentityHi alone.
        return flags == kFlagAirFrameIdentity;
    }

    static bool is_air_frame_identity(const LdpcMiniHeader& header) {
        return is_air_frame_identity_flags(header.flags);
    }

    static bool is_air_frame_identity_hi_flags(uint8_t flags) {
        return flags == kFlagAirFrameIdentityHi;
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

    static size_t packet_qpsk_symbols(size_t payload_blocks) {
        return kControlSymbols + payload_blocks * kLdpcQpskSymbolsPerBlock;
    }

    /** @brief QPSK control RE plus packed payload RE for the selected modulation. */
    static size_t packet_symbols(size_t payload_blocks, const DigitalModulation& modulation,
                                 size_t transmitted_bits = kLdpcCodeBitsPerBlock) {
        return kControlSymbols + payload_blocks * modulation.symbol_count(transmitted_bits);
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
        // Full expanded Identity SIB must use write_identity_control_qpsk.
        // Per-word helpers support lo/hi for identity flag variants.
        write_header_word_qpsk(pack_header(header), out);
    }

    template<typename Out>
    static void write_control_qpsk(const LdpcMiniHeader& header, Out* out) {
        write_marker_qpsk(out);
        write_mini_header_qpsk(header, out + kMarkerSymbols);
    }

    /**
     * @brief Encode expanded Tier-1 Identity SIB: marker + 2× BCH mini-headers.
     *
     * word0: version|Identity|epoch_lo16|seq_lo24|CRC16
     * word1: version|IdentityHi|epoch_hi16|seq_hi8|pad8|CRC16
     * Combined: epoch32 | frame_seq32 (canonical OTA identity).
     */
    template<typename Out>
    static void write_identity_control_qpsk(
        uint32_t timeline_epoch32,
        uint32_t air_frame_seq32,
        Out* out)
    {
        write_marker_qpsk(out);
        const uint16_t epoch_lo = static_cast<uint16_t>(timeline_epoch32 & 0xFFFFu);
        const uint16_t epoch_hi = static_cast<uint16_t>((timeline_epoch32 >> 16) & 0xFFFFu);
        const uint32_t seq_lo = air_frame_seq32 & 0x00FFFFFFu;
        const uint32_t seq_hi = (air_frame_seq32 >> 24) & 0xFFu;
        write_header_word_qpsk(
            pack_identity_header_lo(epoch_lo, seq_lo),
            out + kMarkerSymbols);
        write_header_word_qpsk(
            pack_identity_header_hi(epoch_hi, seq_hi),
            out + kMarkerSymbols + kMiniHeaderSymbols);
    }

    /**
     * @brief Decode expanded Identity SIB control block (marker already checked).
     * @param header_llrs LLRs starting at first mini-header bit (after marker bits)
     */
    template<typename Llr>
    static bool decode_identity_control_llrs(const Llr* header_llrs, LdpcMiniHeader& header_out) {
        LdpcMiniHeader lo{};
        LdpcMiniHeader hi{};
        if (!decode_mini_header_llrs(header_llrs, lo)) {
            return false;
        }
        if (!is_air_frame_identity(lo)) {
            return false;
        }
        if (!decode_mini_header_llrs(
                header_llrs + static_cast<std::ptrdiff_t>(kMiniHeaderEncodedBits),
                hi)) {
            return false;
        }
        if (!is_air_frame_identity_hi_flags(hi.flags) || hi.version != kVersion) {
            return false;
        }
        // lo.timeline_epoch = epoch_lo16, hi.timeline_epoch = epoch_hi16
        header_out = lo;
        header_out.timeline_epoch =
            (lo.timeline_epoch & 0xFFFFu) |
            ((hi.timeline_epoch & 0xFFFFu) << 16);
        header_out.air_frame_seq =
            (lo.air_frame_seq & 0x00FFFFFFu) |
            ((hi.air_frame_seq & 0xFFu) << 24);
        return true;
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
        if (is_air_frame_identity(header)) {
            return pack_identity_header_lo(
                static_cast<uint16_t>(header.timeline_epoch & 0xFFFFu),
                header.air_frame_seq & 0x00FFFFFFu);
        }
        if (is_air_frame_identity_hi_flags(header.flags)) {
            return pack_identity_header_hi(
                static_cast<uint16_t>((header.timeline_epoch >> 16) & 0xFFFFu),
                header.air_frame_seq & 0xFFu);
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

        return prefix_with_crc(prefix);
    }

    /**
     * @brief Identity word0: version|Identity|epoch_lo16|seq_lo24|CRC16.
     */
    static uint64_t pack_identity_header_lo(uint16_t epoch_lo16, uint32_t seq_lo24) {
        uint64_t prefix = 0;
        prefix |= (static_cast<uint64_t>(kVersion & 0x0F) << 44);
        prefix |= (static_cast<uint64_t>(kFlagAirFrameIdentity & 0x0F) << 40);
        prefix |= (static_cast<uint64_t>(epoch_lo16) << 24);
        prefix |= (static_cast<uint64_t>(seq_lo24) & 0x00FFFFFFull);
        return prefix_with_crc(prefix);
    }

    /**
     * @brief Identity word1: version|IdentityHi|epoch_hi16|seq_hi8|pad8|CRC16.
     * Uses former pad bits to carry epoch high 16 → full epoch32 without more QPSK.
     */
    static uint64_t pack_identity_header_hi(uint16_t epoch_hi16, uint32_t seq_hi8) {
        uint64_t prefix = 0;
        prefix |= (static_cast<uint64_t>(kVersion & 0x0F) << 44);
        prefix |= (static_cast<uint64_t>(kFlagAirFrameIdentityHi & 0x0F) << 40);
        prefix |= (static_cast<uint64_t>(epoch_hi16) << 24);
        prefix |= (static_cast<uint64_t>(seq_hi8 & 0xFFu) << 16);
        // bits [15:8] reserved pad; [7:0] unused in prefix before CRC
        return prefix_with_crc(prefix);
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

        if (parsed.version != kVersion || !flags_are_known(parsed.flags)) {
            return false;
        }

        if (is_air_frame_identity_flags(parsed.flags) ||
            is_air_frame_identity_hi_flags(parsed.flags)) {
            parsed.payload_len = 0;
            parsed.payload_blocks = 0;
            parsed.seq = 0;
            // For lo: epoch_lo16; for hi: epoch_hi16 (combined in decode_identity).
            parsed.timeline_epoch = static_cast<uint32_t>((prefix >> 24) & 0xFFFFu);
            if (is_air_frame_identity_hi_flags(parsed.flags)) {
                // seq_hi lives in bits [23:16] of the 48-bit prefix.
                parsed.air_frame_seq = static_cast<uint32_t>((prefix >> 16) & 0xFFu);
            } else {
                parsed.air_frame_seq = static_cast<uint32_t>(prefix & 0x00FFFFFFu);
            }
            header_out = parsed;
            return true;
        }

        parsed.payload_len = static_cast<uint16_t>((prefix >> 24) & 0xFFFFu);
        parsed.payload_blocks = static_cast<uint8_t>((prefix >> 16) & 0xFFu);
        parsed.seq = static_cast<uint16_t>(prefix & 0xFFFFu);
        parsed.timeline_epoch = 0;
        parsed.air_frame_seq = 0;

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
    static uint64_t prefix_with_crc(uint64_t prefix_48) {
        uint8_t bytes[6] = {};
        for (int i = 0; i < 6; ++i) {
            bytes[i] = static_cast<uint8_t>((prefix_48 >> (40 - i * 8)) & 0xFFu);
        }
        const uint16_t crc = crc16_ccitt(bytes, sizeof(bytes));
        return (prefix_48 << 16) | static_cast<uint64_t>(crc);
    }

    template<typename Out>
    static void write_header_word_qpsk(uint64_t word, Out* out) {
        const auto code_bits = bch_encode_header(word);
        for (size_t sym = 0; sym < kMiniHeaderSymbols; ++sym) {
            const size_t bit_idx = sym * 2;
            const int b0 = (bit_idx < kMiniHeaderBchBits) ? code_bits[bit_idx] : 0;
            const int b1 = (bit_idx + 1 < kMiniHeaderBchBits) ? code_bits[bit_idx + 1] : 0;
            out[sym] = static_cast<Out>((b0 << 1) | b1);
        }
    }

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
 * @brief Classify data RE for decision-directed sensing without LDPC decoding.
 * @param sample Accessor returning equalized IQ at a flattened payload-RE index.
 * @param mask Preallocated output: 0 = QPSK control, 1 = payload, 2 = unknown.
 * A valid BCH header identifies each payload extent, including final mapping
 * padding. Corrupted headers leave the remainder unknown rather than guessing
 * packet lengths. Known idle fill uses the configured payload constellation.
 */
template<typename SampleAt>
inline void classify_payload_modulation(size_t count, SampleAt sample,
                                        const DigitalModulation& modulation,
                                        uint8_t* mask, size_t transmitted_bits = 1008) {
    if (count == 0) return;
    std::fill_n(mask, count, uint8_t{2});
    std::array<float, 2 * LdpcPacketFraming::kIdentityControlSymbols> control{};
    size_t offset = 0;
    while (count - offset >= LdpcPacketFraming::kControlSymbols) {
        const size_t available = std::min(count - offset, LdpcPacketFraming::kIdentityControlSymbols);
        for (size_t i = 0; i < available; ++i) {
            const auto value = sample(offset + i);
            control[2 * i] = value.real();
            control[2 * i + 1] = value.imag();
        }
        if (!LdpcPacketFraming::detect_marker_llrs(control.data())) {
            std::fill(mask + offset, mask + count, uint8_t{1});
            return;
        }
        LdpcMiniHeader header{};
        if (available == LdpcPacketFraming::kIdentityControlSymbols &&
            LdpcPacketFraming::decode_identity_control_llrs(
                control.data() + LdpcPacketFraming::kMarkerBits, header)) {
            std::fill_n(mask + offset, LdpcPacketFraming::kIdentityControlSymbols, uint8_t{0});
            offset += LdpcPacketFraming::kIdentityControlSymbols;
            continue;
        }
        if (!LdpcPacketFraming::decode_mini_header_llrs(
                control.data() + LdpcPacketFraming::kMarkerBits, header) ||
            LdpcPacketFraming::is_air_frame_identity_hi_flags(header.flags)) return;
        const size_t size = LdpcPacketFraming::packet_symbols(
            LdpcPacketFraming::payload_blocks_for_len(header.payload_len), modulation, transmitted_bits);
        if (size > count - offset) return;
        std::fill_n(mask + offset, LdpcPacketFraming::kControlSymbols, uint8_t{0});
        std::fill(mask + offset + LdpcPacketFraming::kControlSymbols, mask + offset + size, uint8_t{1});
        offset += size;
    }
    std::fill(mask + offset, mask + count, uint8_t{1});
}

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
            LOG_G_INFO_M(Config) << "Imported FFTW wisdom from " << filename;
        } else {
            LOG_G_INFO_M(Config) << "No existing FFTW wisdom found (will act as cold start).";
        }
    }

    static void export_wisdom(const std::string& filename = "fftw_wisdom.dat") {
        if (FILE* f = std::fopen(filename.c_str(), "w")) {
            fftwf_export_wisdom_to_file(f);
            std::fclose(f);
            LOG_G_INFO_M(Config) << "Exported FFTW wisdom to " << filename;
        } else {
            LOG_G_ERROR_M(Config) << "Failed to export FFTW wisdom to " << filename;
        }
    }
};

/**
 * @brief Payload bit scrambler/descrambler.
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
 * @brief Fixed block bit interleaver for LDPC-coded payloads.
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

    /** @brief Correlation quality and completeness of the configured sync preamble. */
    struct PreambleValidation {
        bool valid = false;
        float repeated_zc_metric = 0.0f;
        float cfo_training_metric = 0.0f;
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

    // Diagnostic: copy the per-window matched-filter output produced by the most
    // recent find_sync_position[_in_range]() call into `out`, one complex value
    // per candidate ZC-symbol start position i in [0, n_windows). Callers can
    // take std::norm() for a power spectrum. `data_size` is the length of the
    // buffer that was passed to the correlation call.
    void copy_last_correlation(size_t data_size, AlignedVector& out) const {
        const size_t n_windows = data_size >= _symbol_len
            ? (data_size - _symbol_len + 1)
            : 0;
        out.resize(n_windows);
        for (size_t i = 0; i < n_windows; ++i) {
            out[i] = _corr_result[i + _symbol_len - 1];
        }
    }

    /**
     * @brief Copy a contiguous slice of the last matched-filter correlation.
     *
     * Same indexing as copy_last_correlation(): sample i is the complex
     * correlation at candidate ZC start i in [0, n_windows).
     *
     * @param data_size Length of the buffer used for the last correlation
     * @param start First window index to copy
     * @param len Number of windows to copy (clamped to available length)
     * @param out Destination; resized to the actual number of copied samples
     * @return Actual start index used after clamping
     */
    size_t copy_last_correlation_slice(
        size_t data_size,
        size_t start,
        size_t len,
        AlignedVector& out) const
    {
        const size_t n_windows = data_size >= _symbol_len
            ? (data_size - _symbol_len + 1)
            : 0;
        if (n_windows == 0 || len == 0) {
            out.clear();
            return 0;
        }
        const size_t clamped_start = std::min(start, n_windows - 1);
        const size_t clamped_len = std::min(len, n_windows - clamped_start);
        out.resize(clamped_len);
        for (size_t i = 0; i < clamped_len; ++i) {
            out[i] = _corr_result[clamped_start + i + _symbol_len - 1];
        }
        return clamped_start;
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

    /**
     * @brief Validate the configured fields around a main ZC candidate.
     * @param data Time-domain samples, including cyclic prefixes.
     * @param main_zc_start Candidate main-ZC cyclic-prefix start.
     * @param fft_size Useful samples per OFDM symbol.
     * @param cp_length Cyclic-prefix samples per symbol.
     * @param require_repeated_zc Require a matching ZC immediately before the candidate.
     * @param require_cfo_training Require the periodic field immediately after the candidate.
     * @param period_samples CFO-field repetition period.
     * @param sample_rate Sample rate in samples per second.
     * @param min_metric Minimum normalized correlation for each enabled field.
     * @return Field metrics and whether the complete configured preamble matches.
     */
    static PreambleValidation validate_sync_preamble(
        const AlignedVector& data, size_t main_zc_start,
        size_t fft_size, size_t cp_length,
        bool require_repeated_zc, bool require_cfo_training,
        size_t period_samples, double sample_rate, float min_metric = 0.10f)
    {
        PreambleValidation result;
        if (fft_size == 0 || cp_length > std::numeric_limits<size_t>::max() - fft_size ||
            !std::isfinite(min_metric) || min_metric < 0.0f || min_metric >= 1.0f) {
            return result;
        }
        const size_t symbol_len = fft_size + cp_length;
        if (main_zc_start > data.size() || data.size() - main_zc_start < symbol_len) {
            return result;
        }
        if (require_repeated_zc) {
            if (main_zc_start < symbol_len) {
                return result;
            }
            const size_t first = main_zc_start - symbol_len + cp_length;
            const size_t second = main_zc_start + cp_length;
            double corr_re = 0.0, corr_im = 0.0, energy0 = 0.0, energy1 = 0.0;
            #pragma omp simd reduction(+:corr_re,corr_im,energy0,energy1)
            for (size_t i = 0; i < fft_size; ++i) {
                const double ar = data[first + i].real(), ai = data[first + i].imag();
                const double br = data[second + i].real(), bi = data[second + i].imag();
                corr_re += ar * br + ai * bi;
                corr_im += ar * bi - ai * br;
                energy0 += ar * ar + ai * ai;
                energy1 += br * br + bi * bi;
            }
            if (!(energy0 > 0.0) || !(energy1 > 0.0)) {
                return result;
            }
            result.repeated_zc_metric = static_cast<float>(
                (corr_re * corr_re + corr_im * corr_im) / (energy0 * energy1));
            if (!std::isfinite(result.repeated_zc_metric) ||
                result.repeated_zc_metric <= min_metric) {
                return result;
            }
        }
        if (require_cfo_training) {
            const auto training = estimate_cfo_from_training_symbol(
                data, main_zc_start + symbol_len, fft_size, cp_length,
                period_samples, sample_rate);
            result.cfo_training_metric = training.metric;
            if (!training.valid || !std::isfinite(training.metric) || training.metric <= min_metric) {
                return result;
            }
        }
        result.valid = true;
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
 * @brief Light-weight EWMA filter for residual CFO error (ppm).
 *
 * Replaces the former adaptive Kalman filter. Used only as a pre-filter before
 * OCXODiscipliner (Mode A Coarse/FLL CFO observation).
 *
 *   filtered <- (1 - alpha) * filtered + alpha * raw
 *
 * Control actions (OCXO ppm steps) are injected so the filter does not treat
 * self-induced residual steps as fresh measurement.
 */
class CfoErrorEwma {
public:
    struct Params {
        bool enable = true;
        double alpha = 0.02;  // 0..1; higher = faster tracking / less smoothing
    };

    struct UpdateResult {
        double raw_error_ppm = 0.0;
        double filtered_error_ppm = 0.0;
        bool initialized = false;
    };

    CfoErrorEwma() { configure(Params{}); }

    explicit CfoErrorEwma(const Params& params) { configure(params); }

    void configure(const Params& params) {
        params_ = sanitize(params);
        reset();
    }

    void reset() {
        initialized_ = false;
        filtered_ = 0.0;
        pending_control_delta_ppm_ = 0.0;
    }

    /**
     * @brief Inject an applied OCXO command step (ppm of the UE clock).
     *
     * The Mode A plant has negative residual gain: a positive OCXO command step
     * reduces a positive residual by the same ppm amount. Shift the filter
     * prediction by the resulting residual step so the controller does not
     * re-see its own action as a fresh disturbance.
     */
    void notify_control_action(double applied_delta_ppm) {
        if (!params_.enable || !std::isfinite(applied_delta_ppm)) {
            return;
        }
        if (std::abs(applied_delta_ppm) <= 1e-15) {
            return;
        }
        pending_control_delta_ppm_ += applied_delta_ppm;
    }

    UpdateResult update(double raw_error_ppm) {
        UpdateResult out;
        out.raw_error_ppm = std::isfinite(raw_error_ppm) ? raw_error_ppm : 0.0;

        if (!params_.enable) {
            pending_control_delta_ppm_ = 0.0;
            out.filtered_error_ppm = out.raw_error_ppm;
            out.initialized = false;
            return out;
        }

        // Apply pending plant steps before the measurement update. Residual
        // change is the negative of the applied actuator command.
        if (initialized_ && std::abs(pending_control_delta_ppm_) > 1e-15) {
            filtered_ -= pending_control_delta_ppm_;
        }
        pending_control_delta_ppm_ = 0.0;

        if (!initialized_) {
            filtered_ = out.raw_error_ppm;
            initialized_ = true;
            out.filtered_error_ppm = filtered_;
            out.initialized = true;
            return out;
        }

        const double a = params_.alpha;
        filtered_ = (1.0 - a) * filtered_ + a * out.raw_error_ppm;
        out.filtered_error_ppm = filtered_;
        out.initialized = true;
        return out;
    }

    double filtered_error_ppm() const { return filtered_; }
    bool initialized() const { return initialized_; }

private:
    Params params_{};
    bool initialized_ = false;
    double filtered_ = 0.0;
    double pending_control_delta_ppm_ = 0.0;

    static Params sanitize(const Params& in) {
        Params p = in;
        if (!(p.alpha > 0.0) || !std::isfinite(p.alpha)) {
            p.alpha = 0.02;
        }
        p.alpha = std::min(std::max(p.alpha, 1e-6), 1.0);
        return p;
    }
};

/**
 * @brief Staged OCXO frequency/phase disciplining loop.
 *
 * Mode A drives Coarse/FLL from residual CFO (same sign as the plant-oriented
 * residual: +OCXO reduces +residual). Mode B estimates raw clock SFO from the
 * Alignment-continuous eRTM TO_clock slope, then controls with the negated
 * slope so UE-local OCXO is negative-feedback on the bench; debug still
 * reports the raw slope in to_slope_ppm. Legacy Phase/Auto lock TO_clock with
 * error (setpoint - to_clock); explicit PpsPhase locks measured UE-minus-BS
 * PPS phase to zero after an optional one-shot PPSOUT placement.
 */
class OCXODiscipliner {
public:
    /** Maximum sync-loss interval that retains the active loop stage. */
    static constexpr double kReacquisitionModeHoldS = 30.0;
    /** Minimum continuous fault interval before an automatic stage downgrade. */
    static constexpr double kDegradationHoldS = 10.0;

    enum class ModePolicy {
        Auto,
        Freq,
        Phase,
        PpsPhase
    };

    enum class ActiveMode {
        Off,
        FreqAlign,
        PhaseLock
    };

    enum class FrequencySource {
        None,
        Cfo,
        ToSlope
    };

    enum class FllFastControlMode {
        Pi,
        HalfStep
    };

    enum class Stage {
        Coarse,
        FllFast,
        FllSlow,
        Phase,
        Freeze
    };

    enum class TransitionReason {
        None,
        Reset,
        CoarseComplete,
        CoarseOvershoot,
        FllStable,
        FllUnstable,
        PhaseReady,
        PhaseSfoConfirmed,
        ToStale,
        ToJump,
        FrequencySourceChanged,
        ControlFrozen
    };

    enum class ReacquisitionAction {
        Waiting,
        Preserved,
        Reset
    };

    struct ReacquisitionResult {
        ReacquisitionAction action = ReacquisitionAction::Preserved;
        Stage previous_stage = Stage::Freeze;
        double elapsed_s = 0.0;
    };

    enum class SetpointMode {
        FreezeOnLock,
        Zero,
        Fixed
    };

    struct Params {
        ModePolicy policy = ModePolicy::Auto;
        double samples_per_frame = 115200.0;
        double max_abs_u_ppm = 0.45;

        bool coarse_enable = true;
        double coarse_gain = 1.0;
        double coarse_max_abs_ppm = 0.2;
        double coarse_obs_window_s = 1.0;
        double coarse_control_period_s = 0.5;
        double coarse_max_slew_ppm_per_s = 0.01;

        double fll_fast_obs_window_s = 1.0;
        double fll_fast_control_period_s = 0.5;
        FllFastControlMode fll_fast_control_mode = FllFastControlMode::Pi;
        double fll_fast_kp = 0.5;
        double fll_fast_ki = 0.03;
        double fll_fast_max_slew_ppm_per_s = 0.005;
        double fll_slow_obs_window_s = 10.0;
        double fll_slow_control_period_s = 2.0;
        double fll_slow_kp = 0.1;
        double fll_slow_ki = 0.007;
        double fll_slow_deadzone_ppm = 5.0e-5;
        double fll_slow_max_slew_ppm_per_s = 0.005;
        double fll_slow_enter_abs_ppm = 0.0005;
        double fll_fast_enter_abs_ppm = 0.01;
        double fll_switch_hold_s = 30.0;

        double phase_control_period_s = 5.0;
        double phase_jump_gate_samples = 0.5;
        uint64_t phase_max_report_age_frames = 256;
        SetpointMode phase_setpoint_mode = SetpointMode::FreezeOnLock;
        double phase_setpoint_fixed_samples = 0.0;
        double phase_kp = 2.0e-4;
        double phase_ki = 1.0e-6;
        double phase_max_slew_ppm_per_s = 5.0e-5;
        double phase_enter_sfo_abs_ppm = 2.0e-4;
        double phase_enter_hold_s = 30.0;
        double phase_exit_sfo_abs_ppm = 0.01;
        double phase_exit_hold_s = kDegradationHoldS;
    };

    struct Observation {
        bool cfo_valid = false;
        double cfo_ppm = 0.0;
        bool to_valid = false;
        double to_ue_samples = 0.0;
        int32_t to_alignment_samples = 0;
        uint64_t to_timeline_epoch = 0;
        uint64_t to_frame_seq = 0;
        uint64_t to_sync_generation = 0;
        uint64_t to_report_age_frames = 0;
        bool to_is_new = false;
        bool pps_phase_valid = false;
        double pps_phase_error_samples = 0.0; ///< P_U-P_B, target zero
        bool phase_control_ready = true; ///< false while one-shot AP is pending
        bool allow_control = true;
    };

    struct UpdateResult {
        bool request_control = false;
        double desired_delta_ppm = 0.0;
        double applied_delta_ppm = 0.0;
        double command_ppm = 0.0;
        double cfo_error_ppm = 0.0;
        double frequency_error_ppm = 0.0; ///< Plant-oriented FLL error; Mode B is -to_slope_ppm
        FrequencySource frequency_source = FrequencySource::None;
        double to_raw = 0.0;
        double to_clock = 0.0;
        double to_err = 0.0; ///< Active Phase error: TO setpoint or PPS phase
        double pps_phase_error_samples = 0.0;
        double to_slope_ppm = 0.0; ///< Raw TO_clock regression slope before plant-sign mapping
        bool to_slope_valid = false;
        double alignment_feedforward_samples = 0.0;
        double coarse_target_ppm = 0.0;
        double coarse_outstanding_ppm = 0.0;
        bool to_jump_rejected = false;
        bool stage_changed = false;
        ActiveMode mode = ActiveMode::Off;
        Stage stage = Stage::Freeze;
        TransitionReason transition_reason = TransitionReason::None;
    };

    OCXODiscipliner() { configure(Params{}); }

    explicit OCXODiscipliner(const Params& params) { configure(params); }

    void configure(const Params& params) {
        params_ = sanitize_params(params);
        reset_tracking_at_command(0.0);
    }

    /**
     * @brief Reset observations and loop stages while retaining the command.
     */
    void reset() {
        reset_tracking_at_command(u_ppm_);
    }

    /**
     * @brief Reset observations and loop stages while retaining the DAC command.
     *
     * Long synchronization reacquisition restarts estimation, not the physical
     * OCXO operating point. The retained command is clamped to the configured
     * actuator range and becomes the initial integrator/output state.
     */
    void reset_tracking_at_command(double retained_command_ppm) {
        stage_ = params_.coarse_enable ? Stage::Coarse : Stage::FllFast;
        u_ppm_ = std::isfinite(retained_command_ppm)
            ? clamp(retained_command_ppm,
                    -params_.max_abs_u_ppm,
                    params_.max_abs_u_ppm)
            : 0.0;
        prev_control_error_ = 0.0;
        has_prev_error_ = false;
        stable_time_s_ = 0.0;
        phase_exit_time_s_ = 0.0;
        degradation_pending_.fill(false);
        degradation_started_.fill({});
        pending_frequency_source_ = FrequencySource::None;
        last_control_time_ = {};
        has_control_time_ = false;
        clear_observation_window();
        coarse_target_valid_ = false;
        coarse_target_ppm_ = 0.0;
        last_applied_delta_ppm_ = 0.0;
        frequency_source_ = FrequencySource::None;
        frozen_ = false;
        freeze_started_ = {};
        pending_to_jump_ = false;
        pending_to_discontinuity_ = false;
        reacquisition_hold_active_ = false;
        reacquisition_started_ = {};
        reset_to_anchor(true);
        refresh_debug_state();
    }

    void reanchor_to(bool allow_state_changes = true) {
        if (reacquisition_hold_active_) {
            reset_to_anchor(false);
            pending_to_discontinuity_ = false;
            refresh_debug_state();
            return;
        }
        reset_to_anchor(allow_state_changes);
    }

    /** @brief Freeze loop time and retain the active stage during sync search. */
    void begin_reacquisition_hold() {
        begin_reacquisition_hold_at(std::chrono::steady_clock::now());
    }

    /** @brief Deterministic-time overload for reacquisition tests. */
    void begin_reacquisition_hold_at(
        std::chrono::steady_clock::time_point now)
    {
        if (!reacquisition_hold_active_) {
            reacquisition_hold_active_ = true;
            reacquisition_started_ = now;
        }
        enter_freeze(now);
        reset_to_anchor(false);
        pending_to_discontinuity_ = false;
        refresh_debug_state();
    }

    /**
     * @brief Decide whether recovered tracking may retain its previous stage.
     *
     * Phase remains frozen until the new timeline establishes a valid TO/PPS
     * observation. A hold longer than @p preserve_mode_s resets loop stages
     * while retaining the current DAC command.
     */
    ReacquisitionResult prepare_reacquisition_resume(
        const Observation& obs,
        double preserve_mode_s)
    {
        return prepare_reacquisition_resume_at(
            obs, preserve_mode_s, std::chrono::steady_clock::now());
    }

    /** @brief Deterministic-time overload for reacquisition tests. */
    ReacquisitionResult prepare_reacquisition_resume_at(
        const Observation& obs,
        double preserve_mode_s,
        std::chrono::steady_clock::time_point now)
    {
        ReacquisitionResult result;
        result.previous_stage = stage_;
        if (!reacquisition_hold_active_) {
            return result;
        }

        result.elapsed_s = std::max(
            0.0,
            std::chrono::duration<double>(now - reacquisition_started_).count());
        if (result.elapsed_s > std::max(preserve_mode_s, 0.0)) {
            const double retained_command_ppm = u_ppm_;
            reset_tracking_at_command(retained_command_ppm);
            result.action = ReacquisitionAction::Reset;
            return result;
        }

        // A Phase loop cannot resume until the new timeline has established
        // its TO/PPS anchor. update_at(allow_control=false) may ingest that
        // anchor while the original stage remains frozen.
        if (stage_ == Stage::Phase &&
            (!phase_observation_is_fresh(obs) ||
             (!uses_pps_phase() && !to_setpoint_valid_))) {
            result.action = ReacquisitionAction::Waiting;
            return result;
        }

        reacquisition_hold_active_ = false;
        reacquisition_started_ = {};
        result.action = ReacquisitionAction::Preserved;
        return result;
    }

    /** @return True while a synchronization reacquisition hold is pending. */
    bool reacquisition_hold_active() const {
        return reacquisition_hold_active_;
    }

    double command_ppm() const { return u_ppm_; }
    Stage stage() const { return stage_; }

    void commit_applied_delta(double applied_delta_ppm) {
        if (!std::isfinite(applied_delta_ppm)) {
            applied_delta_ppm = 0.0;
        }
        const double before = u_ppm_;
        u_ppm_ = clamp(u_ppm_ + applied_delta_ppm,
                       -params_.max_abs_u_ppm,
                       params_.max_abs_u_ppm);
        const double committed = u_ppm_ - before;
        last_applied_delta_ppm_ = committed;
        refresh_debug_state();
    }

    UpdateResult debug_state() const { return debug_state_; }

    static int stage_code(Stage s) {
        switch (s) {
            case Stage::Coarse: return 0;
            case Stage::FllFast: return 1;
            case Stage::FllSlow: return 2;
            case Stage::Phase: return 3;
            case Stage::Freeze: return 4;
        }
        return 4;
    }

    static const char* mode_policy_name(ModePolicy p) {
        switch (p) {
            case ModePolicy::Auto: return "auto";
            case ModePolicy::Freq: return "freq";
            case ModePolicy::Phase: return "phase";
            case ModePolicy::PpsPhase: return "pps_phase";
        }
        return "auto";
    }

    static const char* active_mode_name(ActiveMode m) {
        switch (m) {
            case ActiveMode::Off: return "off";
            case ActiveMode::FreqAlign: return "freq";
            case ActiveMode::PhaseLock: return "phase";
        }
        return "off";
    }

    static const char* frequency_source_name(FrequencySource source) {
        switch (source) {
            case FrequencySource::None: return "none";
            case FrequencySource::Cfo: return "cfo";
            case FrequencySource::ToSlope: return "to_slope";
        }
        return "none";
    }

    static const char* stage_name(Stage s) {
        switch (s) {
            case Stage::Coarse: return "coarse";
            case Stage::FllFast: return "fll_fast";
            case Stage::FllSlow: return "fll_slow";
            case Stage::Phase: return "phase";
            case Stage::Freeze: return "freeze";
        }
        return "freeze";
    }

    static const char* transition_reason_name(TransitionReason reason) {
        switch (reason) {
            case TransitionReason::None: return "none";
            case TransitionReason::Reset: return "reset";
            case TransitionReason::CoarseComplete: return "coarse_complete";
            case TransitionReason::CoarseOvershoot: return "coarse_overshoot";
            case TransitionReason::FllStable: return "fll_stable";
            case TransitionReason::FllUnstable: return "fll_unstable";
            case TransitionReason::PhaseReady: return "phase_ready";
            case TransitionReason::PhaseSfoConfirmed: return "phase_sfo_confirmed";
            case TransitionReason::ToStale: return "to_stale";
            case TransitionReason::ToJump: return "to_jump";
            case TransitionReason::FrequencySourceChanged: return "frequency_source_changed";
            case TransitionReason::ControlFrozen: return "control_frozen";
        }
        return "none";
    }

    UpdateResult update(const Observation& obs) {
        return update_at(obs, std::chrono::steady_clock::now());
    }

    UpdateResult update_at(
        const Observation& obs,
        std::chrono::steady_clock::time_point now)
    {
        UpdateResult out;
        out.command_ppm = u_ppm_;
        out.stage = stage_;
        out.cfo_error_ppm =
            obs.cfo_valid && std::isfinite(obs.cfo_ppm) ? obs.cfo_ppm : 0.0;
        out.to_raw = std::isfinite(obs.to_ue_samples) ? obs.to_ue_samples : 0.0;
        pps_phase_valid_ =
            obs.pps_phase_valid && std::isfinite(obs.pps_phase_error_samples);
        if (pps_phase_valid_) {
            pps_phase_error_samples_ = obs.pps_phase_error_samples;
        }
        if (obs.to_valid && obs.to_is_new) {
            ingest_new_to_report(obs, now, out);
        }
        populate_to_debug(out);

        if (!obs.allow_control) {
            enter_freeze(now);
            out.mode = ActiveMode::Off;
            out.stage = Stage::Freeze;
            out.transition_reason = TransitionReason::ControlFrozen;
            debug_state_ = out;
            return out;
        }
        leave_freeze(now);

        if (apply_pending_to_event(now, out)) {
            out.mode = stage_ == Stage::Phase
                ? ActiveMode::PhaseLock
                : ActiveMode::FreqAlign;
            out.stage = stage_;
            out.command_ppm = u_ppm_;
            populate_to_debug(out);
            debug_state_ = out;
            return out;
        }

        const bool phase_observation_stale =
            stage_ == Stage::Phase &&
            (!phase_observation_is_fresh(obs) ||
             (!uses_pps_phase() && !to_setpoint_valid_));
        if (phase_observation_stale) {
            if (degradation_confirmed(TransitionReason::ToStale, now)) {
                transition_to(Stage::FllFast, TransitionReason::ToStale, out);
            }
            out.mode = stage_ == Stage::Phase
                ? ActiveMode::PhaseLock
                : ActiveMode::FreqAlign;
            out.stage = stage_;
            out.command_ppm = u_ppm_;
            populate_to_debug(out);
            debug_state_ = out;
            return out;
        }
        clear_degradation_candidate(TransitionReason::ToStale);
        if (to_slope_valid_) {
            clear_degradation_candidate(TransitionReason::ToJump);
        }
        if (prepare_frequency_source(obs, now, out)) {
            out.mode = stage_ == Stage::Phase
                ? ActiveMode::PhaseLock
                : ActiveMode::FreqAlign;
            out.stage = stage_;
            out.command_ppm = u_ppm_;
            populate_to_debug(out);
            debug_state_ = out;
            return out;
        }

        switch (stage_) {
            case Stage::Coarse:
                out = update_coarse(obs, now, out);
                break;
            case Stage::FllFast:
            case Stage::FllSlow:
                out = update_fll(obs, now, out);
                break;
            case Stage::Phase:
                out = update_phase(obs, now, out);
                break;
            case Stage::Freeze:
                transition_to(Stage::FllFast, TransitionReason::ControlFrozen, out);
                break;
        }
        out.mode = stage_ == Stage::Phase
            ? ActiveMode::PhaseLock
            : ActiveMode::FreqAlign;
        out.stage = stage_;
        out.command_ppm = u_ppm_;
        populate_to_debug(out);
        debug_state_ = out;
        return out;
    }

private:
    Params params_{};
    Stage stage_ = Stage::Coarse;
    double u_ppm_ = 0.0;
    double prev_control_error_ = 0.0;
    bool has_prev_error_ = false;
    double stable_time_s_ = 0.0;
    double phase_exit_time_s_ = 0.0;
    std::chrono::steady_clock::time_point last_control_time_{};
    bool has_control_time_ = false;

    double obs_sum_ = 0.0;
    size_t obs_count_ = 0;
    std::chrono::steady_clock::time_point obs_window_start_{};
    bool has_obs_window_ = false;

    bool coarse_target_valid_ = false;
    double coarse_target_ppm_ = 0.0;

    bool to_anchor_valid_ = false;
    double to_last_raw_ = 0.0;
    int32_t to_last_alignment_ = 0;
    uint64_t to_last_timeline_epoch_ = 0;
    uint64_t to_last_frame_seq_ = 0;
    uint64_t to_last_sync_generation_ = 0;
    double to_clock_ = 0.0;
    double to_setpoint_ = 0.0;
    bool to_setpoint_valid_ = false;
    double pps_phase_error_samples_ = 0.0;
    bool pps_phase_valid_ = false;
    double to_slope_ppm_ = 0.0;
    bool to_slope_valid_ = false;
    double alignment_feedforward_samples_ = 0.0;
    double last_applied_delta_ppm_ = 0.0;

    struct ToSlopePoint {
        uint64_t frame_seq = 0;
        double to_clock = 0.0;
        std::chrono::steady_clock::time_point observed_at{};
    };
    std::deque<ToSlopePoint> to_slope_points_;
    FrequencySource frequency_source_ = FrequencySource::None;
    bool frozen_ = false;
    std::chrono::steady_clock::time_point freeze_started_{};
    bool pending_to_jump_ = false;
    bool pending_to_discontinuity_ = false;
    static constexpr size_t kTransitionReasonCount =
        static_cast<size_t>(TransitionReason::ControlFrozen) + 1;
    std::array<bool, kTransitionReasonCount> degradation_pending_{};
    std::array<std::chrono::steady_clock::time_point,
               kTransitionReasonCount> degradation_started_{};
    FrequencySource pending_frequency_source_ = FrequencySource::None;
    bool reacquisition_hold_active_ = false;
    std::chrono::steady_clock::time_point reacquisition_started_{};
    UpdateResult debug_state_{};

    static double clamp(double v, double lo, double hi) {
        return std::max(lo, std::min(hi, v));
    }

    static Params sanitize_params(const Params& in) {
        Params p = in;
        p.samples_per_frame = std::max(p.samples_per_frame, 1.0);
        p.max_abs_u_ppm = std::max(p.max_abs_u_ppm, 0.0);
        p.coarse_gain = std::max(p.coarse_gain, 0.0);
        p.coarse_max_abs_ppm = std::max(p.coarse_max_abs_ppm, 0.0);
        p.coarse_obs_window_s = std::max(p.coarse_obs_window_s, 0.0);
        p.coarse_control_period_s = std::max(p.coarse_control_period_s, 0.01);
        p.coarse_max_slew_ppm_per_s = std::max(p.coarse_max_slew_ppm_per_s, 0.0);
        p.fll_fast_obs_window_s = std::max(p.fll_fast_obs_window_s, 0.0);
        p.fll_fast_control_period_s = std::max(p.fll_fast_control_period_s, 0.01);
        p.fll_fast_kp = std::max(p.fll_fast_kp, 0.0);
        p.fll_fast_ki = std::max(p.fll_fast_ki, 0.0);
        p.fll_fast_max_slew_ppm_per_s = std::max(p.fll_fast_max_slew_ppm_per_s, 0.0);
        p.fll_slow_obs_window_s = std::max(p.fll_slow_obs_window_s, 0.0);
        p.fll_slow_control_period_s = std::max(p.fll_slow_control_period_s, 0.01);
        p.fll_slow_kp = std::max(p.fll_slow_kp, 0.0);
        p.fll_slow_ki = std::max(p.fll_slow_ki, 0.0);
        p.fll_slow_deadzone_ppm = std::max(p.fll_slow_deadzone_ppm, 0.0);
        p.fll_slow_max_slew_ppm_per_s = std::max(p.fll_slow_max_slew_ppm_per_s, 0.0);
        p.fll_slow_enter_abs_ppm = std::max(p.fll_slow_enter_abs_ppm, 0.0);
        p.fll_fast_enter_abs_ppm =
            std::max(p.fll_fast_enter_abs_ppm, p.fll_slow_enter_abs_ppm);
        p.fll_switch_hold_s = std::max(p.fll_switch_hold_s, 0.0);
        p.phase_control_period_s = std::max(p.phase_control_period_s, 0.1);
        p.phase_jump_gate_samples = std::max(p.phase_jump_gate_samples, 0.0);
        p.phase_kp = std::max(p.phase_kp, 0.0);
        p.phase_ki = std::max(p.phase_ki, 0.0);
        p.phase_max_slew_ppm_per_s = std::max(p.phase_max_slew_ppm_per_s, 0.0);
        p.phase_enter_sfo_abs_ppm = std::max(p.phase_enter_sfo_abs_ppm, 0.0);
        p.phase_enter_hold_s = std::max(p.phase_enter_hold_s, 0.0);
        p.phase_exit_sfo_abs_ppm =
            std::max(p.phase_exit_sfo_abs_ppm, p.phase_enter_sfo_abs_ppm);
        p.phase_exit_hold_s = std::max(p.phase_exit_hold_s, kDegradationHoldS);
        return p;
    }

    bool valid_cfo(const Observation& obs) const {
        return obs.cfo_valid && std::isfinite(obs.cfo_ppm);
    }

    bool to_is_fresh(const Observation& obs) const {
        return params_.policy != ModePolicy::Freq &&
               obs.to_valid && to_anchor_valid_ &&
               obs.to_report_age_frames <= params_.phase_max_report_age_frames &&
               std::isfinite(obs.to_ue_samples);
    }

    bool uses_pps_phase() const {
        return params_.policy == ModePolicy::PpsPhase;
    }

    bool phase_observation_is_fresh(const Observation& obs) const {
        if (!to_is_fresh(obs)) {
            return false;
        }
        return !uses_pps_phase() ||
            (obs.pps_phase_valid && std::isfinite(obs.pps_phase_error_samples));
    }

    bool mode_b_requested(const Observation& obs) const {
        if (params_.policy == ModePolicy::Phase ||
            params_.policy == ModePolicy::PpsPhase) {
            return true;
        }
        if (params_.policy == ModePolicy::Freq) {
            return false;
        }
        return to_is_fresh(obs);
    }

    FrequencySource selected_frequency_source(const Observation& obs) const {
        return mode_b_requested(obs) ? FrequencySource::ToSlope : FrequencySource::Cfo;
    }

    /**
     * @brief Plant-oriented Mode B frequency error from TO_clock slope.
     *
     * Raw regression slope is retained in to_slope_ppm_ for debug. Bench with
     * the OCXO on the local UE clock showed Mode B FLL as positive feedback when
     * the loop used +to_slope as the control error (Mode A CFO residual was
     * already negative-feedback). Negate once for Coarse/FLL so +OCXO ppm
     * reduces the controlled error (same plant-gain convention as Mode A).
     */
    double mode_b_frequency_error_ppm() const {
        return -to_slope_ppm_;
    }

    /**
     * @brief Plant-oriented legacy Phase error from Alignment-continuous TO.
     *
     * Uses the opposite of (to_clock - setpoint) so the velocity-form Phase PI
     * matches the Mode B FLL sign on the same TO_clock state. PPS Phase keeps
     * its own error (and ocxo_pps_initial_ap_polarity) unchanged.
     */
    double mode_b_to_phase_error_samples() const {
        if (!to_setpoint_valid_) {
            return 0.0;
        }
        return to_setpoint_ - to_clock_;
    }

    bool frequency_observation(const Observation& obs, double& error_ppm) const {
        if (frequency_source_ == FrequencySource::ToSlope) {
            if (!obs.to_is_new || !to_is_fresh(obs) || !to_slope_valid_) {
                return false;
            }
            error_ppm = mode_b_frequency_error_ppm();
            return true;
        }
        if (frequency_source_ == FrequencySource::Cfo && valid_cfo(obs)) {
            error_ppm = obs.cfo_ppm;
            return true;
        }
        return false;
    }

    double to_slope_window_s() const {
        switch (stage_) {
            case Stage::Coarse:
                return params_.coarse_obs_window_s;
            case Stage::FllFast:
                return params_.fll_fast_obs_window_s;
            case Stage::FllSlow:
            case Stage::Phase:
                return params_.fll_slow_obs_window_s;
            case Stage::Freeze:
                return params_.fll_fast_obs_window_s;
        }
        return params_.fll_fast_obs_window_s;
    }

    double max_to_slope_window_s() const {
        return std::max({params_.coarse_obs_window_s,
                         params_.fll_fast_obs_window_s,
                         params_.fll_slow_obs_window_s});
    }

    void clear_to_slope_history() {
        to_slope_points_.clear();
        to_slope_ppm_ = 0.0;
        to_slope_valid_ = false;
    }

    void update_to_slope_estimate(std::chrono::steady_clock::time_point now) {
        const double max_window_s = max_to_slope_window_s();
        while (to_slope_points_.size() > 2) {
            const double second_age_s = std::chrono::duration<double>(
                now - to_slope_points_[1].observed_at).count();
            if (second_age_s <= max_window_s) {
                break;
            }
            to_slope_points_.pop_front();
        }

        if (to_slope_points_.size() < 2) {
            to_slope_ppm_ = 0.0;
            to_slope_valid_ = false;
            return;
        }

        const double required_window_s = to_slope_window_s();
        size_t first = 0;
        if (required_window_s <= 0.0) {
            first = to_slope_points_.size() - 2;
        } else {
            while (first + 1 < to_slope_points_.size()) {
                const double next_age_s = std::chrono::duration<double>(
                    now - to_slope_points_[first + 1].observed_at).count();
                if (next_age_s < required_window_s) {
                    break;
                }
                ++first;
            }
            const double coverage_s = std::chrono::duration<double>(
                to_slope_points_.back().observed_at -
                to_slope_points_[first].observed_at).count();
            if (coverage_s + 1e-12 < required_window_s) {
                to_slope_ppm_ = 0.0;
                to_slope_valid_ = false;
                return;
            }
        }

        const uint64_t base_frame = to_slope_points_[first].frame_seq;
        double mean_x = 0.0;
        double mean_y = 0.0;
        size_t count = 0;
        for (size_t i = first; i < to_slope_points_.size(); ++i) {
            const double x = static_cast<double>(
                to_slope_points_[i].frame_seq - base_frame) *
                params_.samples_per_frame;
            mean_x += x;
            mean_y += to_slope_points_[i].to_clock;
            ++count;
        }
        mean_x /= static_cast<double>(count);
        mean_y /= static_cast<double>(count);

        double covariance = 0.0;
        double variance = 0.0;
        for (size_t i = first; i < to_slope_points_.size(); ++i) {
            const double x = static_cast<double>(
                to_slope_points_[i].frame_seq - base_frame) *
                params_.samples_per_frame;
            const double dx = x - mean_x;
            covariance += dx * (to_slope_points_[i].to_clock - mean_y);
            variance += dx * dx;
        }
        to_slope_ppm_ = variance > 0.0
            ? covariance / variance * 1.0e6
            : 0.0;
        to_slope_valid_ = variance > 0.0 && std::isfinite(to_slope_ppm_);
        if (!to_slope_valid_) {
            to_slope_ppm_ = 0.0;
        }
    }

    void append_to_slope_point(uint64_t frame_seq,
                               double to_clock,
                               std::chrono::steady_clock::time_point now) {
        to_slope_points_.push_back(ToSlopePoint{frame_seq, to_clock, now});
        update_to_slope_estimate(now);
    }

    bool prepare_frequency_source(const Observation& obs,
                                  std::chrono::steady_clock::time_point now,
                                  UpdateResult& out) {
        const FrequencySource selected = selected_frequency_source(obs);
        if (frequency_source_ == FrequencySource::None) {
            frequency_source_ = selected;
        } else if (frequency_source_ != selected) {
            if (stage_ == Stage::FllSlow || stage_ == Stage::Phase) {
                if (pending_frequency_source_ != selected) {
                    clear_degradation_candidate(
                        TransitionReason::FrequencySourceChanged);
                    pending_frequency_source_ = selected;
                }
                if (!degradation_confirmed(
                        TransitionReason::FrequencySourceChanged, now)) {
                    out.frequency_source = frequency_source_;
                    return false;
                }
            }
            frequency_source_ = selected;
            pending_frequency_source_ = FrequencySource::None;
            reset_control_history();
            if (coarse_target_valid_) {
                coarse_target_valid_ = false;
                coarse_target_ppm_ = u_ppm_;
                transition_to(
                    Stage::FllFast,
                    TransitionReason::FrequencySourceChanged,
                    out);
            } else if (stage_ == Stage::FllSlow || stage_ == Stage::Phase) {
                transition_to(
                    Stage::FllFast,
                    TransitionReason::FrequencySourceChanged,
                    out);
            } else {
                out.transition_reason = TransitionReason::FrequencySourceChanged;
            }
        } else {
            pending_frequency_source_ = FrequencySource::None;
            clear_degradation_candidate(
                TransitionReason::FrequencySourceChanged);
        }
        out.frequency_source = frequency_source_;
        return out.stage_changed;
    }

    void clear_observation_window() {
        obs_sum_ = 0.0;
        obs_count_ = 0;
        obs_window_start_ = {};
        has_obs_window_ = false;
    }

    void reset_control_history() {
        clear_observation_window();
        has_prev_error_ = false;
        has_control_time_ = false;
        stable_time_s_ = 0.0;
        phase_exit_time_s_ = 0.0;
    }

    bool degradation_confirmed(
        TransitionReason reason,
        std::chrono::steady_clock::time_point now)
    {
        const size_t index = static_cast<size_t>(reason);
        if (index >= kTransitionReasonCount) {
            return false;
        }
        if (!degradation_pending_[index]) {
            degradation_pending_[index] = true;
            degradation_started_[index] = now;
            return false;
        }
        return std::chrono::duration<double>(
                   now - degradation_started_[index]).count() >=
               kDegradationHoldS;
    }

    void clear_degradation_candidate(TransitionReason reason) {
        const size_t index = static_cast<size_t>(reason);
        if (index >= kTransitionReasonCount) {
            return;
        }
        degradation_pending_[index] = false;
        degradation_started_[index] = {};
    }

    void clear_degradation_candidate() {
        degradation_pending_.fill(false);
        degradation_started_.fill({});
        pending_frequency_source_ = FrequencySource::None;
    }

    void enter_freeze(std::chrono::steady_clock::time_point now) {
        if (!frozen_) {
            frozen_ = true;
            freeze_started_ = now;
        }
    }

    void leave_freeze(std::chrono::steady_clock::time_point now) {
        if (!frozen_) {
            return;
        }
        const auto frozen_duration = now - freeze_started_;
        if (has_control_time_) {
            last_control_time_ += frozen_duration;
        }
        if (has_obs_window_) {
            obs_window_start_ += frozen_duration;
        }
        for (size_t i = 0; i < kTransitionReasonCount; ++i) {
            if (degradation_pending_[i]) {
                degradation_started_[i] += frozen_duration;
            }
        }
        frozen_ = false;
        freeze_started_ = {};
    }

    void cancel_coarse_target() {
        coarse_target_valid_ = false;
        coarse_target_ppm_ = u_ppm_;
    }

    bool apply_pending_to_event(
        std::chrono::steady_clock::time_point now,
        UpdateResult& out)
    {
        if (!pending_to_jump_ && !pending_to_discontinuity_) {
            return false;
        }
        const bool jump = pending_to_jump_;
        pending_to_jump_ = false;
        pending_to_discontinuity_ = false;
        if (jump) {
            // Force a fresh absolute anchor so zero/fixed setpoints cannot hide
            // the rejected offset indefinitely.
            clear_to_slope_history();
            to_anchor_valid_ = false;
            out.to_jump_rejected = true;
        }
        if (!uses_pps_phase() &&
            params_.phase_setpoint_mode == SetpointMode::FreezeOnLock) {
            to_setpoint_valid_ = false;
        }
        const TransitionReason reason = jump
            ? TransitionReason::ToJump
            : TransitionReason::ToStale;
        if ((stage_ == Stage::FllSlow || stage_ == Stage::Phase) &&
            !degradation_confirmed(reason, now)) {
            return true;
        }
        if (stage_ == Stage::FllFast) {
            reset_control_history();
            out.transition_reason = reason;
        } else {
            transition_to(Stage::FllFast, reason, out);
        }
        return true;
    }

    void reset_to_anchor(bool allow_state_changes) {
        to_anchor_valid_ = false;
        to_last_raw_ = 0.0;
        to_last_alignment_ = 0;
        to_last_timeline_epoch_ = 0;
        to_last_frame_seq_ = 0;
        to_last_sync_generation_ = 0;
        to_clock_ = 0.0;
        to_setpoint_ = 0.0;
        to_setpoint_valid_ = false;
        pps_phase_error_samples_ = 0.0;
        pps_phase_valid_ = false;
        clear_to_slope_history();
        alignment_feedforward_samples_ = 0.0;
        if (!allow_state_changes) {
            pending_to_discontinuity_ = true;
            return;
        }
        if (!uses_pps_phase() &&
            params_.phase_setpoint_mode == SetpointMode::FreezeOnLock) {
            to_setpoint_valid_ = false;
        }
        if (stage_ == Stage::Phase || stage_ == Stage::FllSlow) {
            pending_to_discontinuity_ = true;
        }
    }

    void refresh_debug_state() {
        debug_state_.command_ppm = u_ppm_;
        debug_state_.applied_delta_ppm = last_applied_delta_ppm_;
        debug_state_.coarse_target_ppm = coarse_target_ppm_;
        debug_state_.coarse_outstanding_ppm =
            coarse_target_valid_ ? coarse_target_ppm_ - u_ppm_ : 0.0;
        debug_state_.to_clock = to_clock_;
        debug_state_.to_err = uses_pps_phase()
            ? (pps_phase_valid_ ? pps_phase_error_samples_ : 0.0)
            : mode_b_to_phase_error_samples();
        debug_state_.pps_phase_error_samples =
            pps_phase_valid_ ? pps_phase_error_samples_ : 0.0;
        debug_state_.to_slope_ppm = to_slope_ppm_;
        debug_state_.to_slope_valid = to_slope_valid_;
        debug_state_.alignment_feedforward_samples = alignment_feedforward_samples_;
        debug_state_.stage = stage_;
    }

    void populate_to_debug(UpdateResult& out) const {
        out.frequency_source = frequency_source_;
        out.to_clock = to_clock_;
        out.to_err = uses_pps_phase()
            ? (pps_phase_valid_ ? pps_phase_error_samples_ : 0.0)
            : mode_b_to_phase_error_samples();
        out.pps_phase_error_samples =
            pps_phase_valid_ ? pps_phase_error_samples_ : 0.0;
        out.to_slope_ppm = to_slope_ppm_;
        out.to_slope_valid = to_slope_valid_;
        out.alignment_feedforward_samples = alignment_feedforward_samples_;
        out.coarse_target_ppm = coarse_target_ppm_;
        out.coarse_outstanding_ppm =
            coarse_target_valid_ ? coarse_target_ppm_ - u_ppm_ : 0.0;
        out.applied_delta_ppm = last_applied_delta_ppm_;
    }

    void transition_to(Stage next, TransitionReason reason, UpdateResult& out) {
        if (stage_ == next) {
            return;
        }
        if (stage_ == Stage::Coarse && next != Stage::Coarse) {
            cancel_coarse_target();
        }
        stage_ = next;
        out.stage_changed = true;
        out.transition_reason = reason;
        clear_degradation_candidate();
        reset_control_history();
        if (next == Stage::Phase) {
            if (!uses_pps_phase()) {
                if (params_.phase_setpoint_mode == SetpointMode::Zero) {
                    to_setpoint_ = 0.0;
                } else if (params_.phase_setpoint_mode == SetpointMode::Fixed) {
                    to_setpoint_ = params_.phase_setpoint_fixed_samples;
                } else {
                    to_setpoint_ = to_clock_;
                }
                to_setpoint_valid_ = true;
            }
            // Phase uses velocity-form PI around the inherited FLL command.
            // Seed at zero so fixed/zero TO setpoints and post-AP PPS errors
            // receive the normal proportional response.
            prev_control_error_ = 0.0;
            has_prev_error_ = true;
        }
    }

    bool collect_window(double value,
                        double required_window_s,
                        std::chrono::steady_clock::time_point now,
                        double& average,
                        double& elapsed_s) {
        if (!std::isfinite(value)) {
            return false;
        }
        if (!has_obs_window_) {
            has_obs_window_ = true;
            obs_window_start_ = now;
        }
        obs_sum_ += value;
        ++obs_count_;
        elapsed_s = std::chrono::duration<double>(now - obs_window_start_).count();
        if (elapsed_s < required_window_s || obs_count_ == 0) {
            return false;
        }
        average = obs_sum_ / static_cast<double>(obs_count_);
        return true;
    }

    void request_delta(double desired_delta_ppm, UpdateResult& out) {
        const double room_lo = -params_.max_abs_u_ppm - u_ppm_;
        const double room_hi = params_.max_abs_u_ppm - u_ppm_;
        desired_delta_ppm = clamp(desired_delta_ppm, room_lo, room_hi);
        out.desired_delta_ppm = desired_delta_ppm;
        out.request_control = std::abs(desired_delta_ppm) > 1e-15;
    }

    double velocity_pi_delta(double error_value,
                             double kp,
                             double ki,
                             double deadzone_value,
                             double max_slew_ppm_per_s,
                             double dt) {
        if (!std::isfinite(error_value) || dt <= 0.0) {
            return 0.0;
        }
        double e = error_value;
        if (std::abs(e) < deadzone_value) {
            e = 0.0;
        }
        double de = 0.0;
        if (has_prev_error_) {
            de = e - prev_control_error_;
        }
        double delta = (kp * de + ki * dt * e);
        const double max_step = max_slew_ppm_per_s * dt;
        delta = clamp(delta, -max_step, max_step);
        prev_control_error_ = e;
        has_prev_error_ = true;
        return delta;
    }

    UpdateResult update_coarse(const Observation& obs,
                               std::chrono::steady_clock::time_point now,
                               UpdateResult out) {
        double frequency_error_ppm = 0.0;
        if (!frequency_observation(obs, frequency_error_ppm)) {
            return out;
        }
        out.frequency_error_ppm = frequency_error_ppm;
        out.frequency_source = frequency_source_;
        if (!coarse_target_valid_) {
            double average = frequency_error_ppm;
            if (frequency_source_ != FrequencySource::ToSlope) {
                double elapsed_s = 0.0;
                if (!collect_window(frequency_error_ppm, params_.coarse_obs_window_s,
                                    now, average, elapsed_s)) {
                    return out;
                }
            }
            const double relative_target = clamp(
                params_.coarse_gain * average,
                -params_.coarse_max_abs_ppm,
                params_.coarse_max_abs_ppm);
            coarse_target_ppm_ = clamp(
                u_ppm_ + relative_target,
                -params_.max_abs_u_ppm,
                params_.max_abs_u_ppm);
            coarse_target_valid_ = true;
            clear_observation_window();
            last_control_time_ = now;
            has_control_time_ = true;
        }

        const double outstanding = coarse_target_ppm_ - u_ppm_;
        if (std::abs(outstanding) <= 1e-12) {
            transition_to(Stage::FllFast, TransitionReason::CoarseComplete, out);
            return out;
        }
        if (std::abs(frequency_error_ppm) <= 1e-12 ||
            outstanding * frequency_error_ppm < 0.0) {
            transition_to(Stage::FllFast, TransitionReason::CoarseOvershoot, out);
            return out;
        }

        const double dt = has_control_time_
            ? std::max(std::chrono::duration<double>(now - last_control_time_).count(),
                       params_.coarse_control_period_s)
            : params_.coarse_control_period_s;
        if (has_control_time_ &&
            std::chrono::duration<double>(now - last_control_time_).count() <
                params_.coarse_control_period_s) {
            return out;
        }
        request_delta(
            clamp(outstanding,
                  -params_.coarse_max_slew_ppm_per_s * dt,
                  params_.coarse_max_slew_ppm_per_s * dt),
            out);
        last_control_time_ = now;
        has_control_time_ = true;
        return out;
    }

    UpdateResult update_fll(const Observation& obs,
                            std::chrono::steady_clock::time_point now,
                            UpdateResult out) {
        double frequency_error_ppm = 0.0;
        if (!frequency_observation(obs, frequency_error_ppm)) {
            return out;
        }
        out.frequency_error_ppm = frequency_error_ppm;
        out.frequency_source = frequency_source_;
        const bool fast = stage_ == Stage::FllFast;
        const double obs_window = fast
            ? params_.fll_fast_obs_window_s
            : params_.fll_slow_obs_window_s;
        const double control_period = fast
            ? params_.fll_fast_control_period_s
            : params_.fll_slow_control_period_s;
        const bool regression_slope =
            frequency_source_ == FrequencySource::ToSlope;
        double average = frequency_error_ppm;
        double window_s = 0.0;
        if (!regression_slope) {
            if (!collect_window(frequency_error_ppm, obs_window, now, average, window_s)) {
                return out;
            }
        }
        double dt = has_control_time_
            ? std::chrono::duration<double>(now - last_control_time_).count()
            : (regression_slope ? control_period
                                : std::max(window_s, control_period));
        const bool fll_unstable =
            !fast &&
            std::abs(average) >= params_.fll_fast_enter_abs_ppm;
        if (!fast && !fll_unstable) {
            // Recovery cancels the downgrade dwell immediately, even when the
            // normal FLL control period has not elapsed yet.
            clear_degradation_candidate(TransitionReason::FllUnstable);
        }
        if (fll_unstable &&
            degradation_confirmed(TransitionReason::FllUnstable, now)) {
            transition_to(Stage::FllFast, TransitionReason::FllUnstable, out);
            return out;
        }
        if (has_control_time_ && dt < control_period) {
            return out;
        }
        if (!regression_slope) {
            clear_observation_window();
        }
        dt = std::max(dt, control_period);
        out.frequency_error_ppm = average;
        out.frequency_source = frequency_source_;

        if (fast) {
            if (std::abs(average) <= params_.fll_slow_enter_abs_ppm) {
                stable_time_s_ += dt;
                if (stable_time_s_ >= params_.fll_switch_hold_s) {
                    transition_to(Stage::FllSlow, TransitionReason::FllStable, out);
                    return out;
                }
            } else {
                stable_time_s_ = 0.0;
            }
        }

        if (stage_ == Stage::FllSlow && phase_observation_is_fresh(obs) &&
            frequency_source_ == FrequencySource::ToSlope &&
            to_slope_valid_ &&
            std::abs(average) <= params_.phase_enter_sfo_abs_ppm) {
            stable_time_s_ += dt;
            if (stable_time_s_ >= params_.phase_enter_hold_s) {
                transition_to(Stage::Phase, TransitionReason::PhaseReady, out);
                return out;
            }
        } else if (stage_ == Stage::FllSlow) {
            stable_time_s_ = 0.0;
        }

        const bool now_fast = stage_ == Stage::FllFast;
        double delta = 0.0;
        if (now_fast &&
            params_.fll_fast_control_mode == FllFastControlMode::HalfStep) {
            const double max_step =
                params_.fll_fast_max_slew_ppm_per_s * dt;
            delta = clamp(
                0.5 * average,
                -max_step,
                max_step);
        } else {
            delta = velocity_pi_delta(
                average,
                now_fast ? params_.fll_fast_kp : params_.fll_slow_kp,
                now_fast ? params_.fll_fast_ki : params_.fll_slow_ki,
                now_fast ? 0.0 : params_.fll_slow_deadzone_ppm,
                now_fast ? params_.fll_fast_max_slew_ppm_per_s
                         : params_.fll_slow_max_slew_ppm_per_s,
                dt);
        }
        request_delta(delta, out);
        last_control_time_ = now;
        has_control_time_ = true;
        return out;
    }

    void anchor_to(const Observation& obs,
                   std::chrono::steady_clock::time_point now) {
        to_anchor_valid_ = true;
        to_last_raw_ = obs.to_ue_samples;
        to_last_alignment_ = obs.to_alignment_samples;
        to_last_timeline_epoch_ = obs.to_timeline_epoch;
        to_last_frame_seq_ = obs.to_frame_seq;
        to_last_sync_generation_ = obs.to_sync_generation;
        to_clock_ = obs.to_ue_samples;
        clear_to_slope_history();
        append_to_slope_point(obs.to_frame_seq, to_clock_, now);
        alignment_feedforward_samples_ = 0.0;
    }

    void ingest_new_to_report(const Observation& obs,
                              std::chrono::steady_clock::time_point now,
                              UpdateResult& out) {
        if (!std::isfinite(obs.to_ue_samples) ||
            obs.to_timeline_epoch == 0) {
            return;
        }
        const bool same_timeline =
            to_anchor_valid_ &&
            obs.to_timeline_epoch == to_last_timeline_epoch_ &&
            obs.to_sync_generation == to_last_sync_generation_ &&
            obs.to_frame_seq > to_last_frame_seq_;
        if (!same_timeline) {
            const bool replaced_existing_anchor = to_anchor_valid_;
            anchor_to(obs, now);
            if (replaced_existing_anchor) {
                pending_to_discontinuity_ = true;
            }
            if (!uses_pps_phase()) {
                if (params_.phase_setpoint_mode == SetpointMode::FreezeOnLock) {
                    if (reacquisition_hold_active_ && stage_ == Stage::Phase) {
                        to_setpoint_ = to_clock_;
                        to_setpoint_valid_ = true;
                    }
                } else {
                    to_setpoint_ = params_.phase_setpoint_mode == SetpointMode::Zero
                        ? 0.0
                        : params_.phase_setpoint_fixed_samples;
                    to_setpoint_valid_ = true;
                }
            }
            return;
        }

        const double raw_delta = obs.to_ue_samples - to_last_raw_;
        const double alignment_delta =
            static_cast<double>(obs.to_alignment_samples) -
            static_cast<double>(to_last_alignment_);
        const double clock_delta = raw_delta - alignment_delta;
        alignment_feedforward_samples_ = -alignment_delta;
        to_last_raw_ = obs.to_ue_samples;
        to_last_alignment_ = obs.to_alignment_samples;
        to_last_frame_seq_ = obs.to_frame_seq;
        if (std::abs(clock_delta) > params_.phase_jump_gate_samples) {
            out.to_jump_rejected = true;
            clear_to_slope_history();
            pending_to_jump_ = true;
            return;
        }

        to_clock_ += clock_delta;
        append_to_slope_point(obs.to_frame_seq, to_clock_, now);
    }

    UpdateResult update_phase(const Observation& obs,
                              std::chrono::steady_clock::time_point now,
                              UpdateResult out) {
        if (!phase_observation_is_fresh(obs) ||
            (!uses_pps_phase() && !to_setpoint_valid_)) {
            return out;
        }
        out.frequency_source = FrequencySource::ToSlope;
        out.frequency_error_ppm = to_slope_valid_
            ? mode_b_frequency_error_ppm()
            : 0.0;
        if (uses_pps_phase()) {
            pps_phase_error_samples_ = obs.pps_phase_error_samples;
            pps_phase_valid_ = true;
        }
        if (uses_pps_phase() && !obs.phase_control_ready) {
            // AP is a one-shot coarse phase placement. Stay in Phase/PLL while
            // it is queued/settling, but do not ask SF to chase the pre-AP error.
            return out;
        }
        double control_dt = has_control_time_
            ? std::chrono::duration<double>(now - last_control_time_).count()
            : params_.phase_control_period_s;
        if (has_control_time_ && control_dt < params_.phase_control_period_s) {
            return out;
        }
        control_dt = std::max(control_dt, params_.phase_control_period_s);

        const bool exit_confirmed =
            to_slope_valid_ &&
            std::abs(to_slope_ppm_) >= params_.phase_exit_sfo_abs_ppm;
        phase_exit_time_s_ = exit_confirmed
            ? phase_exit_time_s_ + control_dt
            : 0.0;
        if (phase_exit_time_s_ >= params_.phase_exit_hold_s) {
            transition_to(
                Stage::FllFast,
                TransitionReason::PhaseSfoConfirmed,
                out);
            return out;
        }

        const double phase_error = uses_pps_phase()
            ? obs.pps_phase_error_samples
            : mode_b_to_phase_error_samples();
        const double delta = velocity_pi_delta(
            phase_error,
            params_.phase_kp,
            params_.phase_ki,
            0.0,
            params_.phase_max_slew_ppm_per_s,
            control_dt);
        request_delta(delta, out);
        last_control_time_ = now;
        has_control_time_ = true;
        return out;
    }
};

/**
 * @brief One-shot coarse PPSOUT placement gate for each Phase entry.
 *
 * AP changes only the external PPSDO output phase. It does not modify USRP
 * time, the air-frame timeline, eRTM cache, TO_clock, or the PLL command. Each
 * FllSlow-to-Phase entry rearms one AP transaction unless the residual is
 * already inside the deadzone and the previous hardware transaction is in a
 * terminal state. Once AP succeeds and a post-adjustment PPS observation
 * arrives, the gate stays done for that Phase residence and the normal Phase SF
 * loop remains active. Missing post-AP observations use bounded retries based
 * on a fresh residual phase.
 */
class OCXOPpsInitialAligner {
public:
    enum class CommandStatus {
        Idle,
        Queued,
        InFlight,
        Succeeded,
        Failed
    };

    enum class QueueResult {
        Queued,
        Busy,
        Failed
    };

    enum class State {
        Armed,
        WaitingForCommand,
        Settling,
        Done,
        Failed
    };

    struct Params {
        bool enable = true;
        double max_abs_adjust_ns = 500.0e6;
        double settle_s = 2.0;
        double command_timeout_s = 3.0;
        size_t max_retries = 1;
        double deadzone_ns = 10.0;
        int command_polarity = 1;
    };

    struct Observation {
        bool phase_stage = false;
        bool phase_valid = false;
        double phase_error_samples = 0.0;
        double sample_rate = 0.0;
        uint64_t publication_generation = 0;
        CommandStatus command_status = CommandStatus::Idle;
    };

    struct UpdateResult {
        bool request_adjust = false;
        bool request_hardware_rearm = false;
        bool phase_entry_rearm = false;
        int64_t adjustment_ns = 0;
        size_t retry_count = 0;
        bool phase_control_ready = false;
        bool state_changed = false;
        State state = State::Armed;
    };

    OCXOPpsInitialAligner() { configure(Params{}); }
    explicit OCXOPpsInitialAligner(const Params& params) { configure(params); }

    void configure(const Params& params) {
        params_ = params;
        params_.max_abs_adjust_ns = std::clamp(
            params_.max_abs_adjust_ns, 0.0, 500.0e6);
        params_.settle_s = std::max(params_.settle_s, 0.0);
        params_.command_timeout_s = std::max(params_.command_timeout_s, 0.1);
        params_.deadzone_ns = std::max(params_.deadzone_ns, 0.0);
        params_.command_polarity = params_.command_polarity < 0 ? -1 : 1;
        reset();
    }

    void reset() {
        arm_cycle();
        phase_stage_active_ = false;
    }

    State state() const { return state_; }
    bool phase_control_ready() const {
        return state_ == State::Done || state_ == State::Failed;
    }

    bool notify_command_queue_result(QueueResult result) {
        const State before = state_;
        if (state_ == State::WaitingForCommand) {
            if (result == QueueResult::Busy) {
                state_ = State::Armed;
                request_started_ = {};
            } else if (result == QueueResult::Failed) {
                state_ = State::Failed;
            }
        }
        return state_ != before;
    }

    static int state_code(State state) {
        switch (state) {
            case State::Armed: return 0;
            case State::WaitingForCommand: return 1;
            case State::Settling: return 2;
            case State::Done: return 3;
            case State::Failed: return 4;
        }
        return 4;
    }

    static const char* state_name(State state) {
        switch (state) {
            case State::Armed: return "armed";
            case State::WaitingForCommand: return "waiting_for_command";
            case State::Settling: return "settling";
            case State::Done: return "done";
            case State::Failed: return "failed";
        }
        return "failed";
    }

    UpdateResult update(const Observation& obs) {
        return update_at(obs, std::chrono::steady_clock::now());
    }

    UpdateResult update_at(
        const Observation& obs,
        std::chrono::steady_clock::time_point now)
    {
        UpdateResult out;
        const State before = state_;
        const bool phase_entry = obs.phase_stage && !phase_stage_active_;
        phase_stage_active_ = obs.phase_stage;
        const bool phase_observation_valid =
            obs.phase_valid && std::isfinite(obs.phase_error_samples) &&
            obs.sample_rate > 0.0 && std::isfinite(obs.sample_rate);
        const double phase_error_ns = phase_observation_valid
            ? obs.phase_error_samples * 1.0e9 / obs.sample_rate
            : 0.0;
        const bool command_terminal =
            obs.command_status == CommandStatus::Idle ||
            obs.command_status == CommandStatus::Succeeded ||
            obs.command_status == CommandStatus::Failed;
        const bool aligned_terminal_reentry =
            phase_entry && params_.enable && phase_observation_valid &&
            command_terminal &&
            std::abs(phase_error_ns) <= params_.deadzone_ns;
        if (phase_entry) {
            arm_cycle();
            if (aligned_terminal_reentry) {
                state_ = State::Done;
            } else {
                out.request_hardware_rearm = params_.enable;
                out.phase_entry_rearm = params_.enable;
            }
        }
        out.retry_count = retry_count_;
        if (state_ == State::Armed && obs.phase_stage &&
            phase_observation_valid &&
            obs.command_status == CommandStatus::Idle &&
            (retry_count_ == 0 ||
             obs.publication_generation > retry_after_publication_generation_)) {
            if (std::abs(phase_error_ns) <= params_.deadzone_ns) {
                state_ = State::Done;
                out.state_changed = state_ != before;
                out.state = state_;
                out.phase_control_ready = true;
                return out;
            }
            const double requested_ns = std::clamp(
                static_cast<double>(params_.command_polarity) *
                    phase_error_ns,
                -params_.max_abs_adjust_ns,
                params_.max_abs_adjust_ns);
            out.adjustment_ns = static_cast<int64_t>(std::llround(requested_ns));
            out.request_adjust = true;
            request_publication_generation_ = obs.publication_generation;
            request_started_ = now;
            state_ = State::WaitingForCommand;
        } else if (state_ == State::WaitingForCommand) {
            if (obs.command_status == CommandStatus::Succeeded) {
                settle_started_ = now;
                state_ = State::Settling;
            } else if (obs.command_status == CommandStatus::Failed) {
                // Do not deadlock Phase if AP transport fails. The PLL may still
                // converge through SF, while diagnostics preserve the failure.
                state_ = State::Failed;
            } else if (std::chrono::duration<double>(
                           now - request_started_).count() >=
                       params_.command_timeout_s + 0.5) {
                // The serial worker owns the configured wall-clock deadline.
                // This is only a guard for a missing terminal status update.
                state_ = State::Failed;
            }
        } else if (state_ == State::Settling) {
            const double elapsed_s = std::chrono::duration<double>(
                now - settle_started_).count();
            if (elapsed_s >= params_.settle_s && obs.phase_valid &&
                obs.publication_generation > request_publication_generation_) {
                state_ = State::Done;
            } else if (elapsed_s >=
                       params_.settle_s + params_.command_timeout_s) {
                if (retry_count_ < params_.max_retries) {
                    ++retry_count_;
                    retry_after_publication_generation_ =
                        request_publication_generation_;
                    state_ = State::Armed;
                    out.request_hardware_rearm = true;
                } else {
                    // Bounded fallback: do not block SF forever after all
                    // post-AP observation retries have been exhausted.
                    state_ = State::Failed;
                }
            }
        }

        out.state_changed = state_ != before;
        out.state = state_;
        out.retry_count = retry_count_;
        out.phase_control_ready = phase_control_ready();
        return out;
    }

private:
    void arm_cycle() {
        state_ = params_.enable ? State::Armed : State::Done;
        request_publication_generation_ = 0;
        retry_after_publication_generation_ = 0;
        retry_count_ = 0;
        request_started_ = {};
        settle_started_ = {};
    }
    Params params_{};
    State state_ = State::Armed;
    bool phase_stage_active_ = false;
    uint64_t request_publication_generation_ = 0;
    uint64_t retry_after_publication_generation_ = 0;
    size_t retry_count_ = 0;
    std::chrono::steady_clock::time_point request_started_{};
    std::chrono::steady_clock::time_point settle_started_{};
};

inline OCXODiscipliner::Params make_ocxo_discipliner_params(const Config& cfg) {
    OCXODiscipliner::Params p;
    const std::string& mode = cfg.sync_tracking.ocxo_mode;
    if (mode == kOCXOModeFreq) {
        p.policy = OCXODiscipliner::ModePolicy::Freq;
    } else if (mode == kOCXOModePpsPhase) {
        p.policy = OCXODiscipliner::ModePolicy::PpsPhase;
    } else if (mode == kOCXOModePhase) {
        p.policy = OCXODiscipliner::ModePolicy::Phase;
    } else {
        p.policy = OCXODiscipliner::ModePolicy::Auto;
    }
    p.samples_per_frame = static_cast<double>(cfg.samples_per_frame());
    p.max_abs_u_ppm = cfg.sync_tracking.ocxo_max_abs_ppm;
    p.coarse_enable = cfg.sync_tracking.ocxo_coarse_enable;
    p.coarse_gain = cfg.sync_tracking.ocxo_coarse_gain;
    p.coarse_max_abs_ppm = cfg.sync_tracking.ocxo_coarse_max_abs_ppm;
    p.coarse_obs_window_s = cfg.sync_tracking.ocxo_coarse_obs_window_s;
    p.coarse_control_period_s = cfg.sync_tracking.ocxo_coarse_control_period_s;
    p.coarse_max_slew_ppm_per_s = cfg.sync_tracking.ocxo_coarse_max_slew_ppm_per_s;
    p.fll_fast_obs_window_s = cfg.sync_tracking.ocxo_fll_fast_obs_window_s;
    p.fll_fast_control_period_s = cfg.sync_tracking.ocxo_fll_fast_control_period_s;
    p.fll_fast_control_mode =
        cfg.sync_tracking.ocxo_fll_fast_control_mode ==
                kOCXOFllFastControlHalfStep
            ? OCXODiscipliner::FllFastControlMode::HalfStep
            : OCXODiscipliner::FllFastControlMode::Pi;
    p.fll_fast_kp = cfg.sync_tracking.ocxo_fll_fast_kp;
    p.fll_fast_ki = cfg.sync_tracking.ocxo_fll_fast_ki;
    p.fll_fast_max_slew_ppm_per_s = cfg.sync_tracking.ocxo_fll_fast_max_slew_ppm_per_s;
    p.fll_slow_obs_window_s = cfg.sync_tracking.ocxo_fll_slow_obs_window_s;
    p.fll_slow_control_period_s = cfg.sync_tracking.ocxo_fll_slow_control_period_s;
    p.fll_slow_kp = cfg.sync_tracking.ocxo_fll_slow_kp;
    p.fll_slow_ki = cfg.sync_tracking.ocxo_fll_slow_ki;
    p.fll_slow_deadzone_ppm = cfg.sync_tracking.ocxo_fll_slow_deadzone_ppm;
    p.fll_slow_max_slew_ppm_per_s = cfg.sync_tracking.ocxo_fll_slow_max_slew_ppm_per_s;
    p.fll_slow_enter_abs_ppm = cfg.sync_tracking.ocxo_fll_slow_enter_abs_ppm;
    p.fll_fast_enter_abs_ppm = cfg.sync_tracking.ocxo_fll_fast_enter_abs_ppm;
    p.fll_switch_hold_s = cfg.sync_tracking.ocxo_fll_switch_hold_s;
    p.phase_control_period_s = cfg.sync_tracking.ocxo_phase_control_period_s;
    p.phase_jump_gate_samples = cfg.sync_tracking.ocxo_phase_jump_gate_samples;
    p.phase_max_report_age_frames = cfg.sync_tracking.ocxo_phase_max_report_age_frames;
    const std::string& setpoint_mode = cfg.sync_tracking.ocxo_phase_setpoint_mode;
    if (setpoint_mode == kOCXOPhaseSetpointZero) {
        p.phase_setpoint_mode = OCXODiscipliner::SetpointMode::Zero;
    } else if (setpoint_mode == kOCXOPhaseSetpointFixed) {
        p.phase_setpoint_mode = OCXODiscipliner::SetpointMode::Fixed;
    } else {
        p.phase_setpoint_mode = OCXODiscipliner::SetpointMode::FreezeOnLock;
    }
    p.phase_setpoint_fixed_samples =
        cfg.sync_tracking.ocxo_phase_setpoint_fixed_samples;
    p.phase_kp = cfg.sync_tracking.ocxo_phase_kp;
    p.phase_ki = cfg.sync_tracking.ocxo_phase_ki;
    p.phase_max_slew_ppm_per_s = cfg.sync_tracking.ocxo_phase_max_slew_ppm_per_s;
    p.phase_enter_sfo_abs_ppm = cfg.sync_tracking.ocxo_phase_enter_sfo_abs_ppm;
    p.phase_enter_hold_s = cfg.sync_tracking.ocxo_phase_enter_hold_s;
    p.phase_exit_sfo_abs_ppm = cfg.sync_tracking.ocxo_phase_exit_sfo_abs_ppm;
    p.phase_exit_hold_s = cfg.sync_tracking.ocxo_phase_exit_hold_s;
    return p;
}

inline OCXOPpsInitialAligner::Params make_ocxo_pps_initial_aligner_params(
    const Config& cfg)
{
    OCXOPpsInitialAligner::Params p;
    p.enable = cfg.sync_tracking.ocxo_pps_initial_ap_enable;
    p.max_abs_adjust_ns = cfg.sync_tracking.ocxo_pps_initial_ap_max_abs_ns;
    p.settle_s = cfg.sync_tracking.ocxo_pps_initial_ap_settle_s;
    p.command_timeout_s = cfg.sync_tracking.ocxo_pps_initial_ap_command_timeout_s;
    p.max_retries = cfg.sync_tracking.ocxo_pps_initial_ap_max_retries;
    p.deadzone_ns = cfg.sync_tracking.ocxo_pps_initial_ap_deadzone_ns;
    p.command_polarity = cfg.sync_tracking.ocxo_pps_initial_ap_polarity;
    return p;
}

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
        
        float estimate = 0.0f;
        if (abs1 > 2.0f && abs2 > 2.0f) {
            estimate = 0.5f;
        } else if (abs1 > 2.0f) {
            estimate = delta2;
        } else if (abs2 > 2.0f) {
            estimate = delta1;
        } else {
            estimate = (delta1 > 0.0f && delta2 > 0.0f) ? delta2 : delta1;
        }
        // Quinn's estimate is meaningful only within the selected FFT bin.
        // Low-quality/noise peaks can otherwise produce values outside the
        // documented half-bin interval and contaminate SFO tracking.
        return std::max(-0.5f, std::min(0.5f, estimate));
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
 * @brief Remove residual common phase from CPU dense bistatic CSI before MTI.
 * Requires a stable LoS path and no relative motion between TX and RX.
 * Doppler is measured relative to that path; echo targets may still move.
 */
class SensingPhaseReference {
public:
    /** @brief Preallocate Hamming weights and per-row unit rotations. */
    SensingPhaseReference(size_t bins, size_t max_rows)
        : _weights(bins), _reference_weights(bins), _rotations(max_rows) {
        for (size_t k = 0; k < bins; ++k) {
            _weights[k] = bins > 1
                ? 0.54f - 0.46f * std::cos(2.0 * M_PI * k / (bins - 1)) : 1.0f;
            _weight_sum += _weights[k];
        }
        set_user_delay_offset(0.0f);
    }

    /** @brief Follow a known display shift when measuring the LoS reference.
     * The channel output retains the ramp, so ALGN still moves displayed delay.
     * @return False for non-finite shifts or an empty reference geometry.
     */
    bool set_user_delay_offset(float samples) {
        if (!finite(samples) || _weights.empty()) return false;
        const double delay = std::remainder(double(samples), double(_weights.size()));
        for (size_t k = 0; k < _weights.size(); ++k) {
            const double bin = double(k) - double(_weights.size() / 2);
            const double phase = -2.0 * M_PI * bin * delay / _weights.size();
            _reference_weights[k] = _weights[k] * std::polar(1.0f, static_cast<float>(phase));
        }
        return true;
    }

    /** @brief Validate the whole batch, then apply phase-only correction in place.
     * No allocation; a rejected batch is unchanged. row_stride includes padding.
     * @param weakest_coherence Minimum reference coherence inspected in this batch.
     * @return True when every row has a finite, sufficiently coherent reference.
     */
    bool apply(std::complex<float>* data, size_t rows, size_t row_stride,
               float min_coherence, float& weakest_coherence) {
        weakest_coherence = 0.0f;
        if (!data || rows == 0 || rows > _rotations.size() || _weights.empty() ||
            row_stride < _weights.size() || !finite(min_coherence) ||
            min_coherence <= 0.0f || min_coherence > 1.0f) return false;
        weakest_coherence = 1.0f;
        for (size_t m = 0; m < rows; ++m) {
            const auto* row = data + m * row_stride;
            double re = 0.0, im = 0.0, power = 0.0;
            for (size_t k = 0; k < _weights.size(); ++k) {
                if (!finite(row[k].real()) || !finite(row[k].imag())) {
                    weakest_coherence = 0.0f;
                    return false;
                }
                const double r = row[k].real(), i = row[k].imag(), w = _weights[k];
                const double wr = _reference_weights[k].real(), wi = _reference_weights[k].imag();
                re += wr * r - wi * i;
                im += wr * i + wi * r;
                power += w * (r * r + i * i);
            }
            const double reference_power = re * re + im * im;
            const float coherence = power > 1e-20 * _weight_sum
                ? static_cast<float>(reference_power / (_weight_sum * power)) : 0.0f;
            weakest_coherence = std::min(weakest_coherence, coherence);
            if (coherence < min_coherence) return false;
            const double scale = 1.0 / std::sqrt(reference_power);
            _rotations[m] = {static_cast<float>(re * scale), static_cast<float>(-im * scale)};
        }
        for (size_t m = 0; m < rows; ++m) {
            auto* row = data + m * row_stride;
            const float cr = _rotations[m].real(), ci = _rotations[m].imag();
            #pragma omp simd
            for (size_t k = 0; k < _weights.size(); ++k) {
                const float re = row[k].real(), im = row[k].imag();
                row[k] = {re * cr - im * ci, re * ci + im * cr};
            }
        }
        return true;
    }

private:
    // The CPU UE uses -ffast-math; std::isfinite may be optimized away.
    static bool finite(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return (bits & 0x7f800000u) != 0x7f800000u;
    }
    AlignedFloatVector _weights;
    AlignedVector _reference_weights;
    AlignedVector _rotations;
    double _weight_sum = 0.0;
};

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

    /** @brief Clear channel and MTI history when the input timeline restarts. */
    void reset_history() {
        clear_channel_buffer();
        _mti_filter.reset();
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
            if (_params.range_fft_size > _params.fft_size) {
                std::fill(dest + _params.fft_size, dest + _params.range_fft_size,
                          std::complex<float>(0.0f, 0.0f));
            }
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
        if (_params.range_fft_size > _params.fft_size) {
            std::fill(dest + _params.fft_size, dest + _params.range_fft_size,
                      std::complex<float>(0.0f, 0.0f));
        }
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
            if (_params.range_fft_size > fft_size) {
                std::fill(ch_data + fft_size, ch_data + _params.range_fft_size,
                          std::complex<float>(0.0f, 0.0f));
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
        _err_large_count = 0;
        _pd = 1e-5f;
        _cumulative_delay_offset = 0.0f;
        _cumulative_sensing_delay_offset = 0.0f;
        _sfo_per_frame = 0.0f;
        _has_sfo_estimate = false;
        _first_delay_offset_reading = true;
        std::fill(_delay_offsets.begin(), _delay_offsets.end(), 0.0f);
    }

    // Update delay offset estimation and calculate SFO/SIO
    void update(float delay_offset_reading, float Alignment) {
        if (!_first_delay_offset_reading) {
            apply_alignment_without_observation(Alignment);
            _delay_offsets[_count] = delay_offset_reading + _cumulative_delay_offset;
            if (++_count >= _window_size) {
                _sfo_per_frame = linearRegression(_delay_offsets_indices, _delay_offsets).first;
                _count = 0;
                _cumulative_delay_offset = 0.0f;
                _has_sfo_estimate = true;
            }
            if (std::abs(_sfo_per_frame) > 1.0f) {
                _sfo_per_frame = 0.0f;
            }
            _cumulative_sensing_delay_offset += _sfo_per_frame;
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

    /**
     * @brief Preserve delay-coordinate continuity when an RX alignment commits
     *        on a frame whose delay peak is not trustworthy.
     *
     * The alignment receipt must still update the accumulated coordinate, but
     * the invalid delay observation itself must not enter the SFO regression.
     */
    void apply_alignment_without_observation(float alignment) {
        if (_first_delay_offset_reading || alignment == 0.0f) {
            return;
        }
        _cumulative_delay_offset += alignment;
        _cumulative_sensing_delay_offset -= alignment;
    }

    float get_sfo_per_frame() const { return _sfo_per_frame; }
    float get_sensing_delay_offset() const { return _cumulative_sensing_delay_offset; }
    /** True after the first full regression window has produced an SFO estimate. */
    bool has_estimate() const { return _has_sfo_estimate; }

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
    bool _has_sfo_estimate = false;
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
        std::lock_guard<std::mutex> lock(_mutex);
        _window_size = static_cast<uint16_t>(
            std::max(1, std::min<int>(net.arq_window_packets, kMaxArqWindowPackets)));
        _rto_ms = net.arq_retransmit_timeout_ms > 0
            ? net.arq_retransmit_timeout_ms : 100;
        _max_retries = net.arq_max_retries;
        // direction left unchanged; use reset() or set_direction()
    }

    void set_direction(uint8_t dir) {
        std::lock_guard<std::mutex> lock(_mutex);
        _direction = dir;
    }

    /** Drop all outstanding entries (e.g. BS timeline epoch change). */
    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _entries.clear();
    }

    /**
     * @brief Atomically clear + reconfigure (safe across producer/consumer threads).
     */
    void reset(const NetworkOutputConfig& net, uint8_t direction) {
        std::lock_guard<std::mutex> lock(_mutex);
        _entries.clear();
        _window_size = static_cast<uint16_t>(
            std::max(1, std::min<int>(net.arq_window_packets, kMaxArqWindowPackets)));
        _rto_ms = net.arq_retransmit_timeout_ms > 0
            ? net.arq_retransmit_timeout_ms : 100;
        _max_retries = net.arq_max_retries;
        _direction = direction;
    }

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
            std::max(1, std::min<int>(net.arq_window_packets, kMaxArqWindowPackets)));
        _ordered = net.arq_ordered_delivery;
        _feedback_interval_ms = net.arq_feedback_interval_ms;
        _max_reorder_buf = static_cast<size_t>(_window_size);
        _got_any = false;
        _expected_seq = 0;
        _ack_base = 0;
        _ack_bitmap = 0;
        _skip_count = 0;
        _outside_window_drop_count = 0;
        _reorder_drop_count = 0;
        _skip_buffer_drop_count = 0;
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
            _expected_seq = seq;
            _ack_base = seq;
            _ack_bitmap = 0;
        }

        const int16_t bit_idx = arq_seq_diff(seq, _ack_base);
        if (bit_idx < 0) {
            _dup_count++;
            return false;
        }
        if (bit_idx >= static_cast<int16_t>(_window_size) || bit_idx >= 64) {
            _skip_to_include(seq);
        }
        const int16_t adjusted_bit_idx = arq_seq_diff(seq, _ack_base);
        if (adjusted_bit_idx < 0) {
            _dup_count++;
            return false;
        }
        if (adjusted_bit_idx >= static_cast<int16_t>(_window_size) ||
            adjusted_bit_idx >= 64) {
            _outside_window_drop_count++;
            return false;
        }
        const uint64_t bit = static_cast<uint64_t>(1) << adjusted_bit_idx;
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
                _reorder_drop_count++;
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

    uint16_t expected_seq() const { return _expected_seq; }
    uint16_t ack_base() const { return _ack_base; }
    uint64_t ack_bitmap() const { return _ack_bitmap; }
    uint64_t dup_count() const { return _dup_count; }
    uint64_t accepted_count() const { return _accepted_count; }
    uint64_t skip_count() const { return _skip_count; }
    uint64_t outside_window_drop_count() const { return _outside_window_drop_count; }
    uint64_t reorder_drop_count() const { return _reorder_drop_count; }
    uint64_t skip_buffer_drop_count() const { return _skip_buffer_drop_count; }

private:
    void _skip_to_include(uint16_t seq) {
        const uint16_t ack_span = static_cast<uint16_t>(
            std::min<size_t>(static_cast<size_t>(_window_size), 64u));
        const uint16_t new_ack_base = static_cast<uint16_t>(
            seq - static_cast<uint16_t>(ack_span - 1));
        if (arq_seq_diff(new_ack_base, _ack_base) <= 0) {
            return;
        }

        uint64_t new_bitmap = 0;
        for (int i = 0; i < 64; ++i) {
            if (!(_ack_bitmap & (static_cast<uint64_t>(1) << i))) {
                continue;
            }
            const uint16_t old_seq = static_cast<uint16_t>(_ack_base + i);
            const int16_t new_idx = arq_seq_diff(old_seq, new_ack_base);
            if (new_idx >= 0 && new_idx < static_cast<int16_t>(ack_span)) {
                new_bitmap |= static_cast<uint64_t>(1) << new_idx;
            }
        }

        _ack_base = new_ack_base;
        _ack_bitmap = new_bitmap;
        if (arq_seq_diff(_expected_seq, _ack_base) < 0) {
            _expected_seq = _ack_base;
        }
        for (auto it = _reorder_buffer.begin(); it != _reorder_buffer.end(); ) {
            if (arq_seq_diff(it->first, _ack_base) < 0) {
                it = _reorder_buffer.erase(it);
                _skip_buffer_drop_count++;
            } else {
                ++it;
            }
        }
        _skip_count++;
    }

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
    uint64_t _skip_count = 0;
    uint64_t _outside_window_drop_count = 0;
    uint64_t _reorder_drop_count = 0;
    uint64_t _skip_buffer_drop_count = 0;
    std::map<uint16_t, std::vector<uint8_t>> _reorder_buffer;
};

// Convenience: current time in milliseconds for ARQ timestamps
inline int64_t arq_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}



#endif // OFDM_CORE_HPP
