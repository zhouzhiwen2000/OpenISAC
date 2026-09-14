#ifndef AIR_FRAME_TX_GATE_HPP
#define AIR_FRAME_TX_GATE_HPP

/**
 * @file AirFrameTxGate.hpp
 * @brief Backend-independent absolute-frame claim and timed-TX gate state.
 */

#include <atomic>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <mutex>

#include "AirFrameClock.hpp"
#include "RadioTypes.hpp"

/**
 * @brief Shared CPU/CUDA host-side air-frame claim and hard-gate controller.
 *
 * The class owns only timeline metadata. IQ buffers, modulation and radio I/O
 * remain in the backend-specific transmitter. A stamp update, claim-cursor
 * change and final gate validation are serialized by one mutex. The RT
 * pre-select path intentionally derives an advisory expected identity from
 * lock-free stamp/timing seqlock snapshots, so it does not contend with RX
 * stamp publication or producer claims. That advisory read may straddle a
 * generation change; only validate_for_send(), under @c _claim_mutex, is allowed
 * to authorize transmission.
 */
class AirFrameTxGate {
public:
    enum class Decision {
        Match,
        CancelledGeneration,
        InvalidIdentity,
        EpochMismatch,
        Stale,
        Future,
    };

    struct StampSnapshot {
        uint64_t timeline_epoch = 0;
        uint64_t frame_seq = 0;
        uint64_t frame_start_symbol = 0;
        int64_t ue_frame_start_time_ns = -1;
        int64_t ue_frame_mid_time_ns = -1;
        bool valid = false;
    };

    struct StampUpdate {
        bool valid = false;
        bool epoch_changed = false;
        bool validity_changed = false;
        bool generation_bumped = false;
        uint64_t generation = 1;
        uint64_t next_claim_seq = 0;
    };

    struct Claim {
        AirFrameIdentity identity;
        uint64_t generation = 0;
    };

    struct GateSnapshot {
        Decision decision = Decision::InvalidIdentity;
        AirFrameIdentity expected;
        uint64_t generation = 0;
    };

    AirFrameTxGate(size_t period_samples, size_t num_symbols)
        : _period_samples(period_samples), _num_symbols(num_symbols) {}

    AirFrameTxGate(const AirFrameTxGate&) = delete;
    AirFrameTxGate& operator=(const AirFrameTxGate&) = delete;

    void configure_timing(bool use_timed_tx, double tick_rate, double tx_sample_rate) {
        std::lock_guard<std::mutex> lock(_claim_mutex);
        _timing_version.fetch_add(1, std::memory_order_acq_rel);
        _use_timed_tx.store(use_timed_tx, std::memory_order_relaxed);
        _tick_rate.store(tick_rate, std::memory_order_relaxed);
        _tx_sample_rate.store(tx_sample_rate, std::memory_order_relaxed);
        _timing_version.fetch_add(1, std::memory_order_release);
    }

    StampUpdate store_stamp(const AirFrameIdentity& id) {
        const bool new_valid = id.valid && id.timeline_epoch != 0;
        std::lock_guard<std::mutex> claim_lock(_claim_mutex);
        publish_stamp_unlocked(id, new_valid);

        StampUpdate update;
        update.valid = new_valid;
        update.epoch_changed =
            new_valid && _claim_valid && _claim_epoch != id.timeline_epoch;
        update.validity_changed = new_valid != _claim_valid;
        update.generation_bumped = update.epoch_changed || update.validity_changed;
        if (update.generation_bumped) {
            update.generation =
                _generation.fetch_add(1, std::memory_order_release) + 1;
            _claim_valid = new_valid;
            _claim_epoch = new_valid ? id.timeline_epoch : 0;
            // A completed RX stamp S arrives after period S is already past.
            _next_claim_seq = new_valid ? (id.frame_seq + 1) : 0;
        } else {
            update.generation = _generation.load(std::memory_order_relaxed);
        }
        update.next_claim_seq = _next_claim_seq;
        return update;
    }

    StampSnapshot stamp() const {
        for (;;) {
            const uint64_t before = _stamp_version.load(std::memory_order_acquire);
            if ((before & 1u) != 0u) {
                continue;
            }
            StampSnapshot snapshot;
            snapshot.timeline_epoch = _stamp_epoch.load(std::memory_order_relaxed);
            snapshot.frame_seq = _stamp_seq.load(std::memory_order_relaxed);
            snapshot.frame_start_symbol =
                _stamp_start_symbol.load(std::memory_order_relaxed);
            snapshot.ue_frame_start_time_ns =
                _stamp_start_time_ns.load(std::memory_order_relaxed);
            snapshot.ue_frame_mid_time_ns =
                _stamp_mid_time_ns.load(std::memory_order_relaxed);
            snapshot.valid = _stamp_valid.load(std::memory_order_relaxed);
            const uint64_t after = _stamp_version.load(std::memory_order_acquire);
            if (before == after) {
                return snapshot;
            }
        }
    }

    uint64_t generation() const {
        return _generation.load(std::memory_order_acquire);
    }

    Claim claim_period() {
        std::lock_guard<std::mutex> lock(_claim_mutex);
        Claim claim;
        claim.generation = _generation.load(std::memory_order_relaxed);
        claim.identity.mod_generation = claim.generation;
        if (!_claim_valid) {
            return claim;
        }
        claim.identity.timeline_epoch = _claim_epoch;
        claim.identity.frame_seq = _next_claim_seq++;
        claim.identity.frame_start_symbol =
            claim.identity.frame_seq * static_cast<uint64_t>(_num_symbols);
        claim.identity.valid = true;
        return claim;
    }

    AirFrameIdentity identity_for_grid_ticks(
        long long nominal_grid_ticks,
        int32_t rx_alignment_shift_samples) const
    {
        return identity_for_grid_ticks_impl(
            nominal_grid_ticks, rx_alignment_shift_samples);
    }

    AirFrameIdentity cancel_and_reanchor(
        long long nominal_grid_ticks,
        int32_t rx_alignment_shift_samples)
    {
        std::lock_guard<std::mutex> lock(_claim_mutex);
        const AirFrameIdentity identity = identity_for_grid_ticks_impl(
            nominal_grid_ticks, rx_alignment_shift_samples);
        _generation.fetch_add(1, std::memory_order_acq_rel);
        _claim_valid = identity.valid;
        _claim_epoch = identity.valid ? identity.timeline_epoch : 0;
        _next_claim_seq = identity.valid ? identity.frame_seq : 0;
        return identity;
    }

    void advance_claim_past(const AirFrameIdentity& identity, uint64_t generation) {
        if (!identity.valid) {
            return;
        }
        std::lock_guard<std::mutex> lock(_claim_mutex);
        if (_generation.load(std::memory_order_relaxed) != generation ||
            !_claim_valid || _claim_epoch != identity.timeline_epoch) {
            return;
        }
        if (_next_claim_seq <= identity.frame_seq) {
            _next_claim_seq = identity.frame_seq + 1;
        }
    }

    GateSnapshot validate_for_send(
        const AirFrameIdentity& queued,
        uint64_t queued_generation,
        long long nominal_grid_ticks,
        int32_t rx_alignment_shift_samples,
        bool require_timed_identity) const
    {
        std::lock_guard<std::mutex> lock(_claim_mutex);
        GateSnapshot snapshot;
        snapshot.generation = _generation.load(std::memory_order_acquire);
        snapshot.expected = require_timed_identity
            ? identity_for_grid_ticks_impl(
                  nominal_grid_ticks, rx_alignment_shift_samples)
            : AirFrameIdentity{};
        snapshot.decision = classify(
            queued,
            queued_generation,
            snapshot.expected,
            snapshot.generation,
            require_timed_identity);
        return snapshot;
    }

    static Decision classify(
        const AirFrameIdentity& queued,
        uint64_t queued_generation,
        const AirFrameIdentity& expected,
        uint64_t live_generation,
        bool require_identity)
    {
        if (queued_generation != live_generation ||
            (queued.mod_generation != 0 &&
             queued.mod_generation != live_generation)) {
            return Decision::CancelledGeneration;
        }
        if (!require_identity) {
            return Decision::Match;
        }
        if (!expected.valid) {
            return queued.valid ? Decision::InvalidIdentity : Decision::Match;
        }
        if (!queued.valid) {
            return Decision::InvalidIdentity;
        }
        if (queued.timeline_epoch != expected.timeline_epoch) {
            return Decision::EpochMismatch;
        }
        if (queued.frame_seq < expected.frame_seq) {
            return Decision::Stale;
        }
        if (queued.frame_seq > expected.frame_seq) {
            return Decision::Future;
        }
        return Decision::Match;
    }

    static const char* decision_name(Decision decision) {
        switch (decision) {
        case Decision::Match: return "match";
        case Decision::CancelledGeneration: return "cancelled_generation";
        case Decision::InvalidIdentity: return "invalid_identity";
        case Decision::EpochMismatch: return "epoch_mismatch";
        case Decision::Stale: return "stale";
        case Decision::Future: return "future";
        }
        return "unknown";
    }

private:
    struct TimingSnapshot {
        bool use_timed_tx = false;
        double tick_rate = 0.0;
        double tx_sample_rate = 0.0;
    };

    void publish_stamp_unlocked(const AirFrameIdentity& id, bool valid) {
        _stamp_version.fetch_add(1, std::memory_order_acq_rel);
        _stamp_seq.store(id.frame_seq, std::memory_order_relaxed);
        _stamp_epoch.store(id.timeline_epoch, std::memory_order_relaxed);
        _stamp_start_symbol.store(id.frame_start_symbol, std::memory_order_relaxed);
        _stamp_start_time_ns.store(id.ue_frame_start_time_ns, std::memory_order_relaxed);
        _stamp_mid_time_ns.store(
            id.ue_frame_mid_time_ns >= 0
                ? id.ue_frame_mid_time_ns
                : id.ue_frame_start_time_ns,
            std::memory_order_relaxed);
        _stamp_valid.store(valid, std::memory_order_relaxed);
        _stamp_version.fetch_add(1, std::memory_order_release);
    }

    TimingSnapshot timing() const {
        for (;;) {
            const uint64_t before =
                _timing_version.load(std::memory_order_acquire);
            if ((before & 1u) != 0u) {
                continue;
            }
            TimingSnapshot snapshot;
            snapshot.use_timed_tx =
                _use_timed_tx.load(std::memory_order_relaxed);
            snapshot.tick_rate = _tick_rate.load(std::memory_order_relaxed);
            snapshot.tx_sample_rate =
                _tx_sample_rate.load(std::memory_order_relaxed);
            const uint64_t after =
                _timing_version.load(std::memory_order_acquire);
            if (before == after) {
                return snapshot;
            }
        }
    }

    static long long samples_to_ticks(
        int64_t samples,
        const TimingSnapshot& timing)
    {
        if (!timing.use_timed_tx || !(timing.tick_rate > 0.0) ||
            !(timing.tx_sample_rate > 0.0)) {
            return samples;
        }
        return static_cast<long long>(std::llround(
            static_cast<long double>(samples) *
            static_cast<long double>(timing.tick_rate) /
            static_cast<long double>(timing.tx_sample_rate)));
    }

    AirFrameIdentity identity_for_grid_ticks_impl(
        long long nominal_grid_ticks,
        int32_t rx_alignment_shift_samples) const
    {
        AirFrameIdentity identity;
        const TimingSnapshot timing_snapshot = timing();
        if (!timing_snapshot.use_timed_tx ||
            !(timing_snapshot.tick_rate > 0.0) ||
            !(timing_snapshot.tx_sample_rate > 0.0)) {
            return identity;
        }
        const StampSnapshot snapshot = stamp();
        if (!snapshot.valid || snapshot.timeline_epoch == 0 ||
            snapshot.ue_frame_start_time_ns < 0) {
            return identity;
        }

        const long long identity_ticks = nominal_grid_ticks -
            samples_to_ticks(
                static_cast<int64_t>(rx_alignment_shift_samples),
                timing_snapshot);
        const radio::TimeSpec grid_time =
            radio::TimeSpec::from_ticks(identity_ticks, timing_snapshot.tick_rate);
        const int64_t grid_time_ns =
            grid_time.get_full_secs() * 1'000'000'000ll +
            static_cast<int64_t>(std::llround(
                grid_time.get_frac_secs() * 1.0e9));
        const long double frame_duration_ns =
            static_cast<long double>(_period_samples) * 1.0e9L /
            static_cast<long double>(timing_snapshot.tx_sample_rate);
        if (!(frame_duration_ns > 0.0L)) {
            return identity;
        }

        const long double frame_delta =
            static_cast<long double>(
                grid_time_ns - snapshot.ue_frame_start_time_ns) /
            frame_duration_ns;
        const int64_t rounded_delta =
            static_cast<int64_t>(std::llround(frame_delta));
        uint64_t target_seq = 0;
        if (rounded_delta >= 0) {
            target_seq =
                snapshot.frame_seq + static_cast<uint64_t>(rounded_delta);
        } else {
            const uint64_t backwards = static_cast<uint64_t>(-rounded_delta);
            target_seq = backwards > snapshot.frame_seq
                ? 0
                : snapshot.frame_seq - backwards;
        }

        identity.timeline_epoch = snapshot.timeline_epoch;
        identity.frame_seq = target_seq;
        identity.frame_start_symbol =
            target_seq * static_cast<uint64_t>(_num_symbols);
        identity.scheduled_time_ns = grid_time_ns;
        identity.ue_frame_start_time_ns = grid_time_ns;
        identity.ue_frame_mid_time_ns = grid_time_ns +
            static_cast<int64_t>(std::llround(frame_duration_ns / 2.0L));
        identity.valid = true;
        return identity;
    }

    const size_t _period_samples;
    const size_t _num_symbols;
    std::atomic<bool> _use_timed_tx{false};
    std::atomic<double> _tick_rate{0.0};
    std::atomic<double> _tx_sample_rate{0.0};
    std::atomic<uint64_t> _timing_version{0};

    mutable std::mutex _claim_mutex;
    bool _claim_valid = false;
    uint64_t _claim_epoch = 0;
    uint64_t _next_claim_seq = 0;
    std::atomic<uint64_t> _generation{1};

    std::atomic<bool> _stamp_valid{false};
    std::atomic<uint64_t> _stamp_seq{0};
    std::atomic<uint64_t> _stamp_epoch{0};
    std::atomic<uint64_t> _stamp_start_symbol{0};
    std::atomic<int64_t> _stamp_start_time_ns{-1};
    std::atomic<int64_t> _stamp_mid_time_ns{-1};
    std::atomic<uint64_t> _stamp_version{0};
};

#endif // AIR_FRAME_TX_GATE_HPP
