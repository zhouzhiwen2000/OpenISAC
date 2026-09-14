#ifndef TX_BURST_RECOVERY_HPP
#define TX_BURST_RECOVERY_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>

enum class TxRestartReason : uint32_t {
    None = 0,
    Underflow = 1,
    TimeError = 2,
};

class TxAsyncRestartGate {
public:
    static constexpr int64_t kDefaultCooldownNs = 200'000'000;

    explicit TxAsyncRestartGate(int64_t cooldown_ns = kDefaultCooldownNs)
        : _cooldown_ns(std::max<int64_t>(0, cooldown_ns)) {}

    void reset() noexcept {
        std::lock_guard<std::mutex> lock(_mutex);
        _last_accepted_ns = -1;
        _active_restart_time_ns = -1;
        _accepted_event_floor_ns = -1;
        _pending_mask = 0;
        _deferred_mask = 0;
        _coalesced_count = 0;
        _recovery_active = false;
        _pending_hint.store(false, std::memory_order_release);
    }

    bool request(TxRestartReason reason) noexcept {
        return request_impl(reason, steady_now_ns(), false, -1);
    }

    bool request_at(TxRestartReason reason, int64_t now_ns) noexcept {
        return request_impl(reason, now_ns, false, -1);
    }

    bool request_event(TxRestartReason reason, int64_t event_time_ns) noexcept {
        return request_event_at(reason, steady_now_ns(), event_time_ns);
    }

    bool request_event_at(
        TxRestartReason reason,
        int64_t now_ns,
        int64_t event_time_ns) noexcept
    {
        return request_impl(reason, now_ns, true, event_time_ns);
    }

    TxRestartReason consume() noexcept {
        if (!_pending_hint.load(std::memory_order_acquire)) {
            return TxRestartReason::None;
        }
        std::lock_guard<std::mutex> lock(_mutex);
        if (_recovery_active || _pending_mask == 0) {
            return TxRestartReason::None;
        }
        const uint32_t mask = _pending_mask;
        _pending_mask = 0;
        _pending_hint.store(false, std::memory_order_release);
        _deferred_mask = 0;
        _active_restart_time_ns = -1;
        _recovery_active = true;
        return reason_from_mask(mask);
    }

    void set_recovery_target_time_ns(int64_t restart_time_ns) noexcept {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_recovery_active) {
            _active_restart_time_ns = restart_time_ns;
        }
    }

    void finish_recovery(int64_t restart_time_ns) noexcept {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_recovery_active) {
            return;
        }
        _accepted_event_floor_ns = std::max(
            _accepted_event_floor_ns, restart_time_ns);
        _pending_mask |= _deferred_mask;
        if (_pending_mask != 0) {
            _pending_hint.store(true, std::memory_order_release);
        }
        _deferred_mask = 0;
        _active_restart_time_ns = -1;
        _recovery_active = false;
    }

    uint64_t consume_coalesced_count() noexcept {
        std::lock_guard<std::mutex> lock(_mutex);
        const uint64_t count = _coalesced_count;
        _coalesced_count = 0;
        return count;
    }

private:
    bool request_impl(
        TxRestartReason reason,
        int64_t now_ns,
        bool has_event_time,
        int64_t event_time_ns) noexcept
    {
        const uint32_t mask = reason_mask(reason);
        if (mask == 0) {
            return false;
        }

        std::lock_guard<std::mutex> lock(_mutex);
        if (has_event_time && _accepted_event_floor_ns >= 0 &&
            event_time_ns < _accepted_event_floor_ns) {
            ++_coalesced_count;
            return false;
        }

        if (_recovery_active) {
            // Events timestamped before the new timed SOB belong to the burst
            // currently being retired. A failure at/after the new anchor is
            // deferred until this recovery transaction has completed.
            if (has_event_time && _active_restart_time_ns >= 0 &&
                event_time_ns >= _active_restart_time_ns) {
                _deferred_mask |= mask;
            }
            ++_coalesced_count;
            return false;
        }

        // A blocked TX send may leave a request pending beyond the cooldown.
        // Keep it as one recovery until the TX thread consumes it.
        if (_pending_mask != 0) {
            ++_coalesced_count;
            return false;
        }

        if (_last_accepted_ns >= 0 &&
            (now_ns < _last_accepted_ns ||
             now_ns - _last_accepted_ns < _cooldown_ns)) {
            ++_coalesced_count;
            return false;
        }

        _last_accepted_ns = now_ns;
        _pending_mask |= mask;
        _pending_hint.store(true, std::memory_order_release);
        return true;
    }

    static int64_t steady_now_ns() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static constexpr uint32_t reason_mask(TxRestartReason reason) noexcept {
        return static_cast<uint32_t>(reason);
    }

    static constexpr TxRestartReason reason_from_mask(uint32_t mask) noexcept {
        if ((mask & reason_mask(TxRestartReason::TimeError)) != 0) {
            return TxRestartReason::TimeError;
        }
        if ((mask & reason_mask(TxRestartReason::Underflow)) != 0) {
            return TxRestartReason::Underflow;
        }
        return TxRestartReason::None;
    }

    const int64_t _cooldown_ns;
    std::mutex _mutex;
    int64_t _last_accepted_ns{-1};
    int64_t _active_restart_time_ns{-1};
    int64_t _accepted_event_floor_ns{-1};
    uint32_t _pending_mask{0};
    uint32_t _deferred_mask{0};
    uint64_t _coalesced_count{0};
    bool _recovery_active{false};
    std::atomic<bool> _pending_hint{false};
};

inline long long serialized_tx_restart_target_ticks(
    long long now_ticks,
    long long restart_lead_ticks,
    long long last_scheduled_burst_ticks) noexcept
{
    const auto saturating_add = [](long long lhs, long long rhs) {
        if (rhs > 0 && lhs > std::numeric_limits<long long>::max() - rhs) {
            return std::numeric_limits<long long>::max();
        }
        if (rhs < 0 && lhs < std::numeric_limits<long long>::min() - rhs) {
            return std::numeric_limits<long long>::min();
        }
        return lhs + rhs;
    };

    const long long lead_ticks = std::max<long long>(0, restart_lead_ticks);
    const long long from_now = saturating_add(now_ticks, lead_ticks);
    const long long after_previous =
        saturating_add(last_scheduled_burst_ticks, lead_ticks);
    return std::max(from_now, after_previous);
}

class TxHotPathProfiler {
public:
    struct SendToken {
        int64_t start_ns{0};
    };

    explicit TxHotPathProfiler(int64_t frame_duration_ns)
        : TxHotPathProfiler(frame_duration_ns, steady_now_ns()) {}

    TxHotPathProfiler(int64_t frame_duration_ns, int64_t start_ns)
        : _frame_duration_ns(std::max<int64_t>(0, frame_duration_ns))
    {
        reset_interval_at(start_ns, true);
    }

    void observe_queue(size_t depth) noexcept {
        if (!_queue_observations) {
            _queue_min = depth;
            _queue_max = depth;
        } else {
            _queue_min = std::min(_queue_min, depth);
            _queue_max = std::max(_queue_max, depth);
        }
        _queue_total += depth;
        ++_queue_observations;
    }

    void note_gate_drop() noexcept {
        ++_gate_drops;
    }

    SendToken begin_send() const noexcept {
        return begin_send_at(steady_now_ns());
    }

    SendToken begin_send_at(int64_t start_ns) const noexcept {
        return SendToken{start_ns};
    }

    void complete_send(
        SendToken token,
        bool modulated,
        bool held_future,
        size_t requested_samples,
        size_t sent_samples,
        uint64_t air_seq) noexcept
    {
        complete_send_at(
            token,
            modulated,
            held_future,
            requested_samples,
            sent_samples,
            air_seq,
            steady_now_ns());
    }

    void complete_send_at(
        SendToken token,
        bool modulated,
        bool held_future,
        size_t requested_samples,
        size_t sent_samples,
        uint64_t air_seq,
        int64_t end_ns) noexcept
    {
        const int64_t send_ns = nonnegative_delta(end_ns, token.start_ns);
        const uint8_t kind = modulated ? kModulated : (held_future ? kBlankFuture : kBlankEmpty);

        ++_frames;
        _requested_samples += requested_samples;
        _sent_samples += sent_samples;
        _send_total_ns += static_cast<uint64_t>(send_ns);
        if (send_ns >= _max_send_ns) {
            _max_send_ns = send_ns;
            _max_send_seq = air_seq;
            _max_send_kind = kind;
            _max_send_requested = requested_samples;
            _max_send_sent = sent_samples;
        }

        if (_last_send_end_ns > 0) {
            const int64_t gap_ns = nonnegative_delta(token.start_ns, _last_send_end_ns);
            const int64_t submit_ns = nonnegative_delta(end_ns, _last_send_end_ns);
            _gap_total_ns += static_cast<uint64_t>(gap_ns);
            _max_gap_ns = std::max(_max_gap_ns, gap_ns);
            _submit_total_ns += static_cast<uint64_t>(submit_ns);
            ++_submit_intervals;
            if (submit_ns >= _max_submit_ns) {
                _max_submit_ns = submit_ns;
                _max_submit_seq = air_seq;
            }
            if (_frame_duration_ns > 0 && submit_ns > _frame_duration_ns) {
                ++_over_budget_intervals;
            }
        }
        _last_send_end_ns = end_ns;

        if (sent_samples < requested_samples) {
            ++_short_sends;
            _short_samples += requested_samples - sent_samples;
        }

        if (modulated) {
            ++_modulated_frames;
            _consecutive_blank = 0;
        } else {
            if (held_future) {
                ++_blank_future_frames;
            } else {
                ++_blank_empty_frames;
            }
            ++_consecutive_blank;
            _max_consecutive_blank = std::max(_max_consecutive_blank, _consecutive_blank);
        }
    }

    bool periodic_report_due() const noexcept {
        return _frames != 0 && (_frames % 256u) == 0u &&
               nonnegative_delta(_last_send_end_ns, _interval_start_ns) >= 1'000'000'000;
    }

    std::string report_and_reset(
        const char* trigger,
        size_t current_queue_depth,
        uint64_t next_air_seq)
    {
        return report_and_reset_at(
            trigger,
            current_queue_depth,
            next_air_seq,
            steady_now_ns());
    }

    std::string report_and_reset_at(
        const char* trigger,
        size_t current_queue_depth,
        uint64_t next_air_seq,
        int64_t now_ns)
    {
        const int64_t elapsed_ns = std::max<int64_t>(
            1, nonnegative_delta(now_ns, _interval_start_ns));
        const double elapsed_ms = static_cast<double>(elapsed_ns) / 1.0e6;
        const double send_avg_us = _frames
            ? static_cast<double>(_send_total_ns) / static_cast<double>(_frames) / 1.0e3
            : 0.0;
        const double send_excluding_max_avg_us = _frames > 1
            ? static_cast<double>(_send_total_ns - static_cast<uint64_t>(_max_send_ns)) /
                  static_cast<double>(_frames - 1) / 1.0e3
            : 0.0;
        const double submit_avg_us = _submit_intervals
            ? static_cast<double>(_submit_total_ns) /
                  static_cast<double>(_submit_intervals) / 1.0e3
            : 0.0;
        const double submit_excluding_max_avg_us = _submit_intervals > 1
            ? static_cast<double>(_submit_total_ns - static_cast<uint64_t>(_max_submit_ns)) /
                  static_cast<double>(_submit_intervals - 1) / 1.0e3
            : 0.0;
        const double gap_avg_us = _submit_intervals
            ? static_cast<double>(_gap_total_ns) /
                  static_cast<double>(_submit_intervals) / 1.0e3
            : 0.0;
        const double queue_avg = _queue_observations
            ? static_cast<double>(_queue_total) / static_cast<double>(_queue_observations)
            : 0.0;
        const double sent_msps =
            static_cast<double>(_sent_samples) * 1.0e3 / static_cast<double>(elapsed_ns);
        const double last_send_age_us = _last_send_end_ns > 0
            ? static_cast<double>(nonnegative_delta(now_ns, _last_send_end_ns)) / 1.0e3
            : 0.0;

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << "[TX profile] trigger=" << (trigger ? trigger : "periodic")
            << " elapsed_ms=" << elapsed_ms
            << " next_seq=" << next_air_seq
            << " frames=" << _frames
            << " data=" << _modulated_frames
            << " blank_empty=" << _blank_empty_frames
            << " blank_future=" << _blank_future_frames
            << " blank_run_max=" << _max_consecutive_blank
            << " gate_drops=" << _gate_drops
            << " samples=" << _sent_samples << "/" << _requested_samples
            << " sent_msps=" << sent_msps
            << " short_calls=" << _short_sends
            << " short_samples=" << _short_samples
            << " send_us(avg/excl_max/max)=" << send_avg_us << "/"
            << send_excluding_max_avg_us << "/"
            << static_cast<double>(_max_send_ns) / 1.0e3
            << "@" << _max_send_seq << ":" << kind_name(_max_send_kind)
            << "(" << _max_send_sent << "/" << _max_send_requested << ")"
            << " submit_us(avg/excl_max/max)=" << submit_avg_us << "/"
            << submit_excluding_max_avg_us << "/"
            << static_cast<double>(_max_submit_ns) / 1.0e3
            << "@" << _max_submit_seq
            << " intervals=" << _submit_intervals
            << " budget_us=" << static_cast<double>(_frame_duration_ns) / 1.0e3
            << " gap_us(avg/max)=" << gap_avg_us << "/"
            << static_cast<double>(_max_gap_ns) / 1.0e3
            << " over_budget=" << _over_budget_intervals
            << " queue(min/avg/max/current)="
            << (_queue_observations ? _queue_min : current_queue_depth) << "/"
            << queue_avg << "/"
            << (_queue_observations ? _queue_max : current_queue_depth) << "/"
            << current_queue_depth
            << " last_send_age_us=" << last_send_age_us;

        reset_interval_at(now_ns, false);
        return oss.str();
    }

    void reset_interval(bool reset_send_history = true) noexcept {
        reset_interval_at(steady_now_ns(), reset_send_history);
    }

    void reset_interval_at(int64_t start_ns, bool reset_send_history = true) noexcept {
        _interval_start_ns = start_ns;
        _frames = 0;
        _modulated_frames = 0;
        _blank_empty_frames = 0;
        _blank_future_frames = 0;
        _gate_drops = 0;
        _requested_samples = 0;
        _sent_samples = 0;
        _short_sends = 0;
        _short_samples = 0;
        _send_total_ns = 0;
        _max_send_ns = 0;
        _max_send_seq = 0;
        _max_send_kind = kUnknown;
        _max_send_requested = 0;
        _max_send_sent = 0;
        _submit_total_ns = 0;
        _submit_intervals = 0;
        _max_submit_ns = 0;
        _max_submit_seq = 0;
        _gap_total_ns = 0;
        _max_gap_ns = 0;
        _over_budget_intervals = 0;
        _queue_total = 0;
        _queue_observations = 0;
        _queue_min = 0;
        _queue_max = 0;
        _max_consecutive_blank = _consecutive_blank;
        if (reset_send_history) {
            _last_send_end_ns = 0;
            _consecutive_blank = 0;
            _max_consecutive_blank = 0;
        }
    }

private:
    static constexpr uint8_t kUnknown = 0;
    static constexpr uint8_t kModulated = 1;
    static constexpr uint8_t kBlankEmpty = 2;
    static constexpr uint8_t kBlankFuture = 3;

    static int64_t steady_now_ns() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static int64_t nonnegative_delta(int64_t later, int64_t earlier) noexcept {
        return later >= earlier ? later - earlier : 0;
    }

    static const char* kind_name(uint8_t kind) noexcept {
        switch (kind) {
        case kModulated:
            return "data";
        case kBlankEmpty:
            return "blank-empty";
        case kBlankFuture:
            return "blank-future";
        default:
            return "none";
        }
    }

    const int64_t _frame_duration_ns;
    int64_t _interval_start_ns{0};
    int64_t _last_send_end_ns{0};
    uint64_t _frames{0};
    uint64_t _modulated_frames{0};
    uint64_t _blank_empty_frames{0};
    uint64_t _blank_future_frames{0};
    uint64_t _gate_drops{0};
    uint64_t _requested_samples{0};
    uint64_t _sent_samples{0};
    uint64_t _short_sends{0};
    uint64_t _short_samples{0};
    uint64_t _send_total_ns{0};
    int64_t _max_send_ns{0};
    uint64_t _max_send_seq{0};
    uint8_t _max_send_kind{kUnknown};
    size_t _max_send_requested{0};
    size_t _max_send_sent{0};
    uint64_t _submit_total_ns{0};
    uint64_t _submit_intervals{0};
    int64_t _max_submit_ns{0};
    uint64_t _max_submit_seq{0};
    uint64_t _gap_total_ns{0};
    int64_t _max_gap_ns{0};
    uint64_t _over_budget_intervals{0};
    uint64_t _queue_total{0};
    uint64_t _queue_observations{0};
    size_t _queue_min{0};
    size_t _queue_max{0};
    uint64_t _consecutive_blank{0};
    uint64_t _max_consecutive_blank{0};
};

#endif // TX_BURST_RECOVERY_HPP
