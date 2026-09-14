#ifndef AIR_FRAME_CLOCK_HPP
#define AIR_FRAME_CLOCK_HPP

/**
 * @file AirFrameClock.hpp
 * @brief BS-USRP-time absolute air-frame numbering.
 *
 * Frame index is defined by the BS radio timeline within a timeline epoch:
 *   frame_seq(t) = floor((t_ticks - t0_ticks) / frame_ticks)
 *
 * UE derives frame_seq from a (BS seq, UE boundary time) anchor and the
 * current frame boundary — never from an async free-run counter reset by the
 * bit-processing thread.
 */

#include <atomic>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <random>

#include "RadioTypes.hpp"

/**
 * @brief Absolute air-frame identity shared across BS/UE modules.
 *
 * @c timeline_epoch is the **canonical OTA epoch** (32-bit on the wire).
 * BS may keep a full 64-bit boot id only inside AirFrameClock; that full id is
 * never stored here so BS and UE field values match for the same air frame.
 */
struct AirFrameIdentity {
    uint64_t timeline_epoch = 0;         ///< Canonical OTA epoch (epoch32)
    uint64_t frame_seq = 0;              ///< Absolute air-frame index
    uint64_t frame_start_symbol = 0;
    int64_t scheduled_time_ns = -1;      ///< BS planned TX start
    int64_t ue_frame_start_time_ns = -1; ///< UE boundary
    int64_t ue_frame_mid_time_ns = -1;   ///< UE mid-frame (robust stamp)
    bool valid = false;
    /// BS mod generation; TX drops frames whose generation != current.
    uint64_t mod_generation = 0;

    void clear() {
        timeline_epoch = 0;
        frame_seq = 0;
        frame_start_symbol = 0;
        scheduled_time_ns = -1;
        ue_frame_start_time_ns = -1;
        ue_frame_mid_time_ns = -1;
        valid = false;
        mod_generation = 0;
    }
};

/**
 * @brief Pending Identity-SIB / timeline event for owner-thread application.
 *
 * Bit-processing publishes; RX or demod owner consumes (no cross-thread
 * mutation of SFO/ARQ from the bit thread).
 */
struct AirFrameSibEvent {
    uint64_t sync_generation = 0;
    uint64_t timeline_epoch = 0;
    uint64_t frame_seq = 0;
    int64_t ue_boundary_time_ns = -1;
    int64_t ue_mid_time_ns = -1;
    bool epoch_changed = false;
};

/**
 * @brief UE timeline: anchor-based absolute frame index + boundary/mid times.
 *
 * SIB establishes (bs_seq, ue_boundary_ns). Every RX stamp derives
 *   seq = anchor_seq + round((boundary - anchor_boundary) / T_frame)
 * so decode pipeline delay cannot lag the free-run counter.
 */
class UeAirFrameTimeline {
public:
    void configure(size_t samples_per_frame, size_t num_symbols, double sample_rate) {
        std::lock_guard<std::mutex> lock(_mu);
        _samples_per_frame = samples_per_frame;
        _num_symbols = num_symbols;
        _sample_rate = sample_rate;
        _frame_duration_ns = samples_to_ns_unlocked(
            static_cast<int64_t>(samples_per_frame));
        _half_frame_samples = static_cast<int64_t>(samples_per_frame / 2);
        _half_frame_duration_ns = samples_to_ns_unlocked(_half_frame_samples);
    }

    static int64_t samples_to_ns(int64_t samples, double sample_rate) {
        if (!(sample_rate > 0.0)) {
            return 0;
        }
        return static_cast<int64_t>(std::llround(
            static_cast<double>(samples) * 1.0e9 / sample_rate));
    }

    static int64_t boundary_time_ns(
        int64_t raw_first_sample_time_ns,
        int64_t alignment_sample_adjust,
        double sample_rate)
    {
        if (raw_first_sample_time_ns < 0) {
            return -1;
        }
        return raw_first_sample_time_ns +
               samples_to_ns(alignment_sample_adjust, sample_rate);
    }

    static int64_t mid_time_from_boundary_ns(
        int64_t boundary_ns,
        int64_t half_frame_duration_ns)
    {
        if (boundary_ns < 0) {
            return -1;
        }
        return boundary_ns + half_frame_duration_ns;
    }

    int64_t samples_to_ns(int64_t samples) const {
        std::lock_guard<std::mutex> lock(_mu);
        return samples_to_ns_unlocked(samples);
    }

    int64_t half_frame_duration_ns() const {
        std::lock_guard<std::mutex> lock(_mu);
        return _half_frame_duration_ns;
    }

    int64_t frame_duration_ns() const {
        std::lock_guard<std::mutex> lock(_mu);
        return _frame_duration_ns;
    }

    /**
     * @brief Stamp one committed RX frame from its UE boundary time.
     * When anchored, derives absolute seq from time delta (SFO-tolerant via
     * alignment-tracked boundaries).
     */
    AirFrameIdentity stamp_rx_boundary(int64_t ue_boundary_time_ns) {
        std::lock_guard<std::mutex> lock(_mu);
        _last_ue_frame_start_time_ns = ue_boundary_time_ns;
        const int64_t mid_ns = mid_time_from_boundary_ns(
            ue_boundary_time_ns, _half_frame_duration_ns);
        AirFrameIdentity out;
        out.ue_frame_start_time_ns = ue_boundary_time_ns;
        out.ue_frame_mid_time_ns = mid_ns;
        if (!_anchor_valid || _frame_duration_ns <= 0 || ue_boundary_time_ns < 0) {
            return out;
        }

        const int64_t delta_ns = ue_boundary_time_ns - _anchor_boundary_ns;
        // Nearest-frame index relative to the SIB anchor (handles pipeline lag
        // and occasional missed RX stamps without async counter reset).
        const long double frames_ld =
            static_cast<long double>(delta_ns) /
            static_cast<long double>(_frame_duration_ns);
        const int64_t frames = static_cast<int64_t>(std::llround(frames_ld));
        int64_t seq_i = static_cast<int64_t>(_anchor_seq) + frames;
        if (seq_i < 0) {
            seq_i = 0;
        }
        out.timeline_epoch = _anchor_epoch;
        out.frame_seq = static_cast<uint64_t>(seq_i);
        out.frame_start_symbol =
            (_num_symbols == 0) ? 0 : (out.frame_seq * static_cast<uint64_t>(_num_symbols));
        out.valid = true;
        _last_derived = out;
        return out;
    }

    /**
     * @brief Establish or refresh the (seq, boundary) anchor from Identity SIB.
     *
     * @param frame_seq  Absolute (or modulo-extended) BS frame index.
     * @return true if OTA epoch changed or large same-epoch seq reset (protocol reset).
     */
    bool apply_identity_sib_anchor(
        uint64_t timeline_epoch,
        uint64_t frame_seq,
        int64_t ue_boundary_time_ns = -1,
        int64_t ue_mid_time_ns = -1)
    {
        std::lock_guard<std::mutex> lock(_mu);
        const bool had = _anchor_valid;
        const uint64_t prev_epoch = _anchor_epoch;
        const uint64_t prev_seq = _anchor_seq;
        const bool epoch_changed = had && (prev_epoch != timeline_epoch);

        // Modulo-extend first, then judge abnormal backward jumps on the
        // unwrapped value (so 0xffffffff → 0 is wrap, not timeline reset).
        uint64_t abs_seq = frame_seq;
        if (had && !epoch_changed) {
            abs_seq = extend_mod_seq(prev_seq, frame_seq, /*mod_bits=*/32);
        }
        const bool seq_reset =
            had && !epoch_changed && (abs_seq + 64ull < prev_seq);

        const int64_t boundary =
            (ue_boundary_time_ns >= 0) ? ue_boundary_time_ns : _last_ue_frame_start_time_ns;
        const int64_t mid = (ue_mid_time_ns >= 0)
            ? ue_mid_time_ns
            : mid_time_from_boundary_ns(boundary, _half_frame_duration_ns);

        _anchor_epoch = timeline_epoch;
        _anchor_seq = abs_seq;
        _anchor_boundary_ns = boundary;
        _anchor_valid = (boundary >= 0);
        _last_ue_frame_start_time_ns = boundary;
        _last_derived.timeline_epoch = timeline_epoch;
        _last_derived.frame_seq = abs_seq;
        _last_derived.frame_start_symbol =
            (_num_symbols == 0) ? 0 : (abs_seq * static_cast<uint64_t>(_num_symbols));
        _last_derived.ue_frame_start_time_ns = boundary;
        _last_derived.ue_frame_mid_time_ns = mid;
        _last_derived.valid = _anchor_valid;
        return epoch_changed || seq_reset;
    }

    /**
     * @brief Lift a modular OTA sequence into a continuous uint64 near @p prev_abs.
     */
    static uint64_t extend_mod_seq(uint64_t prev_abs, uint64_t observed_mod, unsigned mod_bits) {
        if (mod_bits == 0 || mod_bits >= 64) {
            return observed_mod;
        }
        const uint64_t mod = uint64_t{1} << mod_bits;
        const uint64_t mask = mod - 1;
        const uint64_t prev_lo = prev_abs & mask;
        const uint64_t obs = observed_mod & mask;
        int64_t delta = static_cast<int64_t>(obs) - static_cast<int64_t>(prev_lo);
        const int64_t half = static_cast<int64_t>(mod >> 1);
        if (delta > half) {
            delta -= static_cast<int64_t>(mod);
        } else if (delta < -half) {
            delta += static_cast<int64_t>(mod);
        }
        const int64_t out = static_cast<int64_t>(prev_abs) + delta;
        return out < 0 ? obs : static_cast<uint64_t>(out);
    }

    AirFrameIdentity snapshot() const {
        std::lock_guard<std::mutex> lock(_mu);
        return _last_derived;
    }

    bool locked() const {
        std::lock_guard<std::mutex> lock(_mu);
        return _anchor_valid;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(_mu);
        _anchor_valid = false;
        _anchor_epoch = 0;
        _anchor_seq = 0;
        _anchor_boundary_ns = -1;
        _last_derived.clear();
        _last_ue_frame_start_time_ns = -1;
    }

private:
    int64_t samples_to_ns_unlocked(int64_t samples) const {
        return samples_to_ns(samples, _sample_rate);
    }

    mutable std::mutex _mu;
    size_t _samples_per_frame = 0;
    size_t _num_symbols = 0;
    double _sample_rate = 0.0;
    int64_t _frame_duration_ns = 0;
    int64_t _half_frame_samples = 0;
    int64_t _half_frame_duration_ns = 0;

    bool _anchor_valid = false;
    uint64_t _anchor_epoch = 0;
    uint64_t _anchor_seq = 0;
    int64_t _anchor_boundary_ns = -1;

    AirFrameIdentity _last_derived;
    int64_t _last_ue_frame_start_time_ns = -1;
};

/**
 * @brief Thread-safe air-frame clock for BS.
 *
 * TX is time authority. Mod claims sequential slots tagged with mod_generation.
 * On underflow restart, bump generation and force_next_seq so stale waveforms
 * cannot re-enter the air.
 */
class AirFrameClock {
public:
    void configure(
        size_t samples_per_frame,
        size_t num_symbols,
        double sample_rate,
        double tick_rate)
    {
        _samples_per_frame = samples_per_frame;
        _num_symbols = num_symbols;
        _sample_rate = sample_rate;
        _tick_rate = (tick_rate > 0.0) ? tick_rate : sample_rate;
        _frame_ticks = compute_frame_ticks(samples_per_frame, sample_rate, _tick_rate);
    }

    size_t samples_per_frame() const { return _samples_per_frame; }
    size_t num_symbols() const { return _num_symbols; }
    double sample_rate() const { return _sample_rate; }
    double tick_rate() const { return _tick_rate; }
    int64_t frame_ticks() const { return _frame_ticks; }
    int64_t t0_ticks() const { return _t0_ticks.load(std::memory_order_acquire); }

    uint64_t timeline_epoch() const {
        return _timeline_epoch.load(std::memory_order_acquire);
    }

    uint64_t mod_generation() const {
        return _mod_generation.load(std::memory_order_acquire);
    }

    uint64_t next_seq() const {
        return _next_seq.load(std::memory_order_acquire);
    }

    /**
     * @brief New boot/session epoch from high-entropy id; seq reset to 0.
     * OTA Identity SIB carries the low 32 bits of this value.
     *
     * Holds @_claim_mu so a concurrent claim_next / force_next_seq cannot
     * interleave with the t0/next_seq reset (safe for future runtime calls).
     */
    uint64_t start_new_session_epoch(const radio::TimeSpec& start_time = radio::TimeSpec(0.0)) {
        std::lock_guard<std::mutex> lock(_claim_mu);
        const uint64_t epoch = make_boot_epoch();
        _timeline_epoch.store(epoch, std::memory_order_release);
        _t0_ticks.store(start_time.to_ticks(_tick_rate), std::memory_order_release);
        _next_seq.store(0, std::memory_order_release);
        return epoch;
    }

    /**
     * @brief Keep claim cursor; rebind t0 to @p start_time.
     * Holds @_claim_mu with other claim-cursor mutators.
     */
    void reanchor_keep_seq(const radio::TimeSpec& start_time) {
        std::lock_guard<std::mutex> lock(_claim_mu);
        _t0_ticks.store(start_time.to_ticks(_tick_rate), std::memory_order_release);
    }

    /**
     * @brief Ensure a non-zero session epoch exists (random boot id).
     */
    uint64_t ensure_epoch_started() {
        uint64_t epoch = _timeline_epoch.load(std::memory_order_acquire);
        if (epoch == 0) {
            epoch = make_boot_epoch();
            uint64_t expected = 0;
            if (!_timeline_epoch.compare_exchange_strong(
                    expected, epoch, std::memory_order_acq_rel)) {
                epoch = expected;
            }
        }
        return epoch;
    }

    AirFrameIdentity make_identity(uint64_t frame_seq, uint64_t gen) const {
        AirFrameIdentity id;
        // Canonical field = OTA epoch32 (not the full 64-bit boot id).
        id.timeline_epoch = ota_epoch32(timeline_epoch());
        id.frame_seq = frame_seq;
        id.frame_start_symbol =
            (_num_symbols == 0) ? 0 : (frame_seq * static_cast<uint64_t>(_num_symbols));
        const int64_t ticks = ticks_from_seq(frame_seq);
        if (_tick_rate > 0.0 && ticks >= 0) {
            id.scheduled_time_ns = static_cast<int64_t>(std::llround(
                radio::TimeSpec::from_ticks(ticks, _tick_rate).get_real_secs() * 1e9));
        }
        id.mod_generation = gen;
        id.valid = (id.timeline_epoch != 0);
        return id;
    }

    /**
     * @brief Claim next (generation, seq) as one consistent snapshot.
     */
    AirFrameIdentity claim_next() {
        std::lock_guard<std::mutex> lock(_claim_mu);
        ensure_epoch_started();
        const uint64_t gen = _mod_generation.load(std::memory_order_relaxed);
        const uint64_t seq = _next_seq.fetch_add(1, std::memory_order_relaxed);
        return make_identity(seq, gen);
    }

    void force_next_seq(uint64_t seq) {
        std::lock_guard<std::mutex> lock(_claim_mu);
        _next_seq.store(seq, std::memory_order_relaxed);
    }

    /**
     * @brief Atomically move the claim cursor beyond an already elapsed slot.
     *
     * Unlike a separate next_seq()/force_next_seq() pair, the comparison and
     * update share @_claim_mu with claim_next(), so a concurrent modulation
     * claim can never be overwritten by the TX blank-frame recovery path.
     */
    uint64_t advance_next_seq_past(uint64_t elapsed_seq) {
        std::lock_guard<std::mutex> lock(_claim_mu);
        uint64_t cursor = _next_seq.load(std::memory_order_relaxed);
        if (cursor <= elapsed_seq && elapsed_seq != UINT64_MAX) {
            cursor = elapsed_seq + 1;
            _next_seq.store(cursor, std::memory_order_relaxed);
        }
        return cursor;
    }

    /**
     * @brief Cancel in-flight modulated frames and jump claim cursor.
     * @return New mod_generation that TX/mod must tag/check.
     */
    uint64_t cancel_and_force_next_seq(uint64_t seq) {
        std::lock_guard<std::mutex> lock(_claim_mu);
        const uint64_t gen =
            _mod_generation.fetch_add(1, std::memory_order_relaxed) + 1;
        _next_seq.store(seq, std::memory_order_relaxed);
        return gen;
    }

    uint64_t seq_from_ticks(int64_t ticks) const {
        const int64_t t0 = _t0_ticks.load(std::memory_order_acquire);
        const int64_t ft = _frame_ticks;
        if (ft <= 0) {
            return 0;
        }
        if (ticks < t0) {
            return 0;
        }
        return static_cast<uint64_t>((ticks - t0) / ft);
    }

    int64_t ticks_from_seq(uint64_t seq) const {
        const int64_t t0 = _t0_ticks.load(std::memory_order_acquire);
        return t0 + static_cast<int64_t>(seq) * _frame_ticks;
    }

    /**
     * @brief Canonical OTA epoch (32-bit). Matches AirFrameIdentity.timeline_epoch
     * on both BS and UE. Full boot id remains only in AirFrameClock::timeline_epoch().
     */
    static uint32_t ota_epoch32(uint64_t full_boot_id) {
        return static_cast<uint32_t>(full_boot_id & 0xFFFFFFFFu);
    }

    /** OTA frame_seq (32-bit). */
    static uint32_t ota_frame_seq32(uint64_t frame_seq) {
        return static_cast<uint32_t>(frame_seq & 0xFFFFFFFFu);
    }

    static int64_t compute_frame_ticks(
        size_t samples_per_frame,
        double sample_rate,
        double tick_rate)
    {
        if (samples_per_frame == 0 || !(sample_rate > 0.0) || !(tick_rate > 0.0)) {
            return 0;
        }
        const double exact =
            static_cast<double>(samples_per_frame) * tick_rate / sample_rate;
        return static_cast<int64_t>(std::llround(exact));
    }

    /**
     * @brief Payload RE available for user/ARQ packets after Identity SIB.
     */
    static size_t effective_payload_re_count(size_t payload_re_count, size_t sib_qpsk_symbols) {
        if (payload_re_count <= sib_qpsk_symbols) {
            return 0;
        }
        return payload_re_count - sib_qpsk_symbols;
    }

private:
    static uint64_t make_boot_epoch() {
        // Distinguish process restarts: mix random bits with coarse wall time.
        std::random_device rd;
        const uint64_t r =
            (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        uint64_t epoch = r ^ static_cast<uint64_t>(now);
        // epoch32=0 is the explicit OTA unlock marker. A non-zero 64-bit value
        // with zero low word would otherwise disable every generated identity.
        if (static_cast<uint32_t>(epoch) == 0u) {
            epoch |= 1u;
        }
        return epoch;
    }

    size_t _samples_per_frame = 0;
    size_t _num_symbols = 0;
    double _sample_rate = 0.0;
    double _tick_rate = 0.0;
    int64_t _frame_ticks = 0;
    std::mutex _claim_mu; ///< Serializes claim_next / force / cancel snapshots
    std::atomic<int64_t> _t0_ticks{0};
    std::atomic<uint64_t> _timeline_epoch{0};
    std::atomic<uint64_t> _next_seq{0};
    std::atomic<uint64_t> _mod_generation{1};
};

#endif // AIR_FRAME_CLOCK_HPP
