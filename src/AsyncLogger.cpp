#include "AsyncLogger.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

namespace async_logger {
namespace {

constexpr double kDefaultInfoHz = 50.0;
constexpr double kDefaultWarnHz = 100.0;
constexpr double kDefaultErrorHz = 200.0;

enum class QueueId : size_t {
    GError = 0,
    RTError,
    GWarn,
    RTWarn,
    GInfo,
    RTInfo,
    Count,
};

constexpr std::array<QueueId, 6> kSchedule = {
    QueueId::GError,
    QueueId::RTError,
    QueueId::GWarn,
    QueueId::RTWarn,
    QueueId::GInfo,
    QueueId::RTInfo,
};

struct LogMessage {
    LogLevel level;
    std::string text;
};

struct QueueState {
    explicit QueueState(size_t cap = 0) : capacity(cap) {}

    std::mutex mutex;
    std::condition_variable cv_not_full;
    std::deque<LogMessage> queue;
    size_t capacity = 0;
};

struct TokenBucketState {
    double tokens = 0.0;
    std::chrono::steady_clock::time_point last_refill = std::chrono::steady_clock::now();
    uint64_t suppressed = 0;
    bool initialized = false;
};

thread_local LoggerThreadMode g_thread_mode = LoggerThreadMode::NonRealtime;

} // namespace

struct AsyncLogger::Impl {
    std::array<QueueState, static_cast<size_t>(QueueId::Count)> queues = {
        QueueState(8192), // G_ERROR
        QueueState(8192), // RT_ERROR
        QueueState(4096), // G_WARN
        QueueState(4096), // RT_WARN
        QueueState(4096), // G_INFO
        QueueState(4096), // RT_INFO
    };

    std::mutex wake_mutex;
    std::condition_variable wake_cv;

    std::atomic<bool> accepting{false};
    std::atomic<bool> started{false};
    std::thread worker;

    std::atomic<uint64_t> rt_dropped_lock{0};
    std::atomic<uint64_t> rt_dropped_queue{0};
    std::atomic<uint64_t> rt_dropped_rate{0};
    std::atomic<uint64_t> guaranteed_dropped{0};
    std::atomic<uint64_t> guaranteed_in_rt_violation{0};
    std::atomic<uint64_t> emitted{0};
    std::atomic<uint64_t> enqueued_rt{0};
    std::atomic<uint64_t> enqueued_guaranteed{0};

    QueueState& queue_for_realtime(LogLevel level) {
        switch (level) {
            case LogLevel::Error: return queues[static_cast<size_t>(QueueId::RTError)];
            case LogLevel::Warn:  return queues[static_cast<size_t>(QueueId::RTWarn)];
            case LogLevel::Info:  return queues[static_cast<size_t>(QueueId::RTInfo)];
        }
        return queues[static_cast<size_t>(QueueId::RTInfo)];
    }

    QueueState& queue_for_guaranteed(LogLevel level) {
        switch (level) {
            case LogLevel::Error: return queues[static_cast<size_t>(QueueId::GError)];
            case LogLevel::Warn:  return queues[static_cast<size_t>(QueueId::GWarn)];
            case LogLevel::Info:  return queues[static_cast<size_t>(QueueId::GInfo)];
        }
        return queues[static_cast<size_t>(QueueId::GInfo)];
    }

    static bool is_guaranteed_queue(QueueId id) {
        return id == QueueId::GError || id == QueueId::GWarn || id == QueueId::GInfo;
    }

    static double rate_for_level(LogLevel level, int hz_override) {
        if (hz_override > 0) {
            return static_cast<double>(hz_override);
        }
        switch (level) {
            case LogLevel::Info: return kDefaultInfoHz;
            case LogLevel::Warn: return kDefaultWarnHz;
            case LogLevel::Error: return kDefaultErrorHz;
        }
        return kDefaultInfoHz;
    }

    bool has_any_message() {
        for (size_t i = 0; i < static_cast<size_t>(QueueId::Count); ++i) {
            auto& q = queues[i];
            std::lock_guard<std::mutex> lock(q.mutex);
            if (!q.queue.empty()) {
                return true;
            }
        }
        return false;
    }

    bool try_pop_from_queue(QueueId id, LogMessage& out) {
        auto& q = queues[static_cast<size_t>(id)];
        bool notify_not_full = false;
        {
            std::lock_guard<std::mutex> lock(q.mutex);
            if (q.queue.empty()) {
                return false;
            }
            out = std::move(q.queue.front());
            q.queue.pop_front();
            notify_not_full = is_guaranteed_queue(id);
        }
        if (notify_not_full) {
            q.cv_not_full.notify_one();
        }
        return true;
    }

    void emit_line(const LogMessage& msg) {
        std::ostream& os = (msg.level == LogLevel::Info) ? std::cout : std::cerr;
        os << msg.text << '\n';
        os.flush();
        emitted.fetch_add(1, std::memory_order_relaxed);
    }

    void worker_loop() {
        size_t next_idx = 0;

        while (accepting.load(std::memory_order_acquire) || has_any_message()) {
            LogMessage msg;
            bool popped = false;

            for (size_t i = 0; i < kSchedule.size(); ++i) {
                const size_t idx = (next_idx + i) % kSchedule.size();
                if (try_pop_from_queue(kSchedule[idx], msg)) {
                    next_idx = (idx + 1) % kSchedule.size();
                    popped = true;
                    break;
                }
            }

            if (popped) {
                emit_line(msg);
                continue;
            }

            std::unique_lock<std::mutex> lock(wake_mutex);
            wake_cv.wait_for(lock, std::chrono::milliseconds(20), [this]() {
                return !accepting.load(std::memory_order_acquire) || has_any_message();
            });
        }
    }

    bool should_emit_realtime(LogLevel level, uint32_t callsite_id, int hz_override, uint64_t& suppressed_count) {
        using Clock = std::chrono::steady_clock;

        struct ThreadRateState {
            std::unordered_map<uint32_t, TokenBucketState> per_callsite;
        };
        thread_local ThreadRateState tls_state;

        const double rate_hz = rate_for_level(level, hz_override);
        const double burst = std::max(1.0, rate_hz);
        const auto now = Clock::now();

        TokenBucketState& state = tls_state.per_callsite[callsite_id];
        if (!state.initialized) {
            state.initialized = true;
            state.tokens = burst;
            state.last_refill = now;
            state.suppressed = 0;
        } else {
            const double dt = std::chrono::duration<double>(now - state.last_refill).count();
            state.last_refill = now;
            state.tokens = std::min(burst, state.tokens + dt * rate_hz);
        }

        if (state.tokens >= 1.0) {
            state.tokens -= 1.0;
            suppressed_count = state.suppressed;
            state.suppressed = 0;
            return true;
        }

        state.suppressed += 1;
        rt_dropped_rate.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    void submit_realtime(LogLevel level, std::string&& msg) {
        if (!accepting.load(std::memory_order_acquire)) {
            rt_dropped_queue.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        QueueState& q = queue_for_realtime(level);
        std::unique_lock<std::mutex> lock(q.mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            rt_dropped_lock.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        if (q.queue.size() >= q.capacity) {
            q.queue.pop_front();
            rt_dropped_queue.fetch_add(1, std::memory_order_relaxed);
        }

        q.queue.push_back(LogMessage{level, std::move(msg)});
        enqueued_rt.fetch_add(1, std::memory_order_relaxed);

        lock.unlock();
        wake_cv.notify_one();
    }

    void submit_guaranteed(LogLevel level, std::string&& msg) {
        if (!accepting.load(std::memory_order_acquire)) {
            guaranteed_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        QueueState& q = queue_for_guaranteed(level);
        std::unique_lock<std::mutex> lock(q.mutex);
        q.cv_not_full.wait(lock, [this, &q]() {
            return !accepting.load(std::memory_order_acquire) || q.queue.size() < q.capacity;
        });

        if (!accepting.load(std::memory_order_acquire)) {
            guaranteed_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        q.queue.push_back(LogMessage{level, std::move(msg)});
        enqueued_guaranteed.fetch_add(1, std::memory_order_relaxed);

        lock.unlock();
        wake_cv.notify_one();
    }

    LoggerStats snapshot_stats() const {
        LoggerStats s;
        s.rt_dropped_lock = rt_dropped_lock.load(std::memory_order_relaxed);
        s.rt_dropped_queue = rt_dropped_queue.load(std::memory_order_relaxed);
        s.rt_dropped_rate = rt_dropped_rate.load(std::memory_order_relaxed);
        s.guaranteed_dropped = guaranteed_dropped.load(std::memory_order_relaxed);
        s.guaranteed_in_rt_violation = guaranteed_in_rt_violation.load(std::memory_order_relaxed);
        s.emitted = emitted.load(std::memory_order_relaxed);
        s.enqueued_rt = enqueued_rt.load(std::memory_order_relaxed);
        s.enqueued_guaranteed = enqueued_guaranteed.load(std::memory_order_relaxed);
        return s;
    }
};

AsyncLogger::AsyncLogger() : _impl(new Impl()) {}

AsyncLogger::~AsyncLogger() {
    shutdown();
    delete _impl;
    _impl = nullptr;
}

AsyncLogger& AsyncLogger::instance() {
    static AsyncLogger logger;
    return logger;
}

void AsyncLogger::start() {
    bool expected = false;
    if (!_impl->started.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
    }

    _impl->accepting.store(true, std::memory_order_release);
    _impl->worker = std::thread([this]() { _impl->worker_loop(); });
}

void AsyncLogger::shutdown() {
    bool expected = true;
    if (!_impl->started.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
        return;
    }

    _impl->accepting.store(false, std::memory_order_release);

    for (size_t i = 0; i < static_cast<size_t>(QueueId::Count); ++i) {
        _impl->queues[i].cv_not_full.notify_all();
    }
    _impl->wake_cv.notify_all();

    if (_impl->worker.joinable()) {
        _impl->worker.join();
    }
}

void AsyncLogger::set_thread_mode(LoggerThreadMode mode) {
    g_thread_mode = mode;
}

LoggerThreadMode AsyncLogger::thread_mode() const {
    return g_thread_mode;
}

bool AsyncLogger::should_emit_realtime(LogLevel level, uint32_t callsite_id, int hz_override, uint64_t& suppressed_count) {
    return _impl->should_emit_realtime(level, callsite_id, hz_override, suppressed_count);
}

void AsyncLogger::submit_realtime(LogLevel level, std::string&& msg) {
    _impl->submit_realtime(level, std::move(msg));
}

void AsyncLogger::submit_guaranteed(LogLevel level, std::string&& msg) {
    _impl->submit_guaranteed(level, std::move(msg));
}

void AsyncLogger::note_guaranteed_in_rt_violation() {
    _impl->guaranteed_in_rt_violation.fetch_add(1, std::memory_order_relaxed);
}

LoggerStats AsyncLogger::snapshot_stats() {
    return instance()._impl->snapshot_stats();
}

} // namespace async_logger
