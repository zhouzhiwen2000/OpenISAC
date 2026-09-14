#ifndef TIMED_STREAM_RESTART_HPP
#define TIMED_STREAM_RESTART_HPP

#include <atomic>
#include <cstdint>

enum class TimedStreamRestartStatus : uint8_t {
    Pending,
    Armed,
    Failed,
};

class TimedStreamRestartAck {
public:
    uint64_t begin_request() noexcept {
        const uint64_t id = _next_id.fetch_add(1, std::memory_order_relaxed);
        _failed_id.store(0, std::memory_order_relaxed);
        return id;
    }

    void mark_armed(uint64_t id) noexcept {
        if (id != 0) {
            _armed_id.store(id, std::memory_order_release);
        }
    }

    void mark_failed(uint64_t id) noexcept {
        if (id != 0) {
            _failed_id.store(id, std::memory_order_release);
        }
    }

    TimedStreamRestartStatus status(uint64_t id) const noexcept {
        if (id == 0) {
            return TimedStreamRestartStatus::Armed;
        }
        if (_failed_id.load(std::memory_order_acquire) == id) {
            return TimedStreamRestartStatus::Failed;
        }
        if (_armed_id.load(std::memory_order_acquire) == id) {
            return TimedStreamRestartStatus::Armed;
        }
        return TimedStreamRestartStatus::Pending;
    }

private:
    std::atomic<uint64_t> _next_id{1};
    std::atomic<uint64_t> _armed_id{0};
    std::atomic<uint64_t> _failed_id{0};
};

#endif // TIMED_STREAM_RESTART_HPP
