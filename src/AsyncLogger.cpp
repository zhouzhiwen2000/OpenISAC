#include "AsyncLogger.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cctype>
#include <cstring>
#include <ctime>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

namespace async_logger {
namespace {

constexpr double kDefaultDebugHz = 10.0;
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
    GDebug,
    RTDebug,
    Count,
};

constexpr std::array<QueueId, 8> kSchedule = {
    QueueId::GError,
    QueueId::RTError,
    QueueId::GWarn,
    QueueId::RTWarn,
    QueueId::GInfo,
    QueueId::RTInfo,
    QueueId::GDebug,
    QueueId::RTDebug,
};

struct LogMessage {
    LogLevel level;
    LogModule mod;
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

// Parent == self means "no parent" for Root; other modules point at their parent.
// Paths ending in "_profiling" are performance-timer modules: when left as
// inherit they stay OFF (do not inherit parent verbosity). Enable explicitly,
// e.g. logging.modules.demod_profiling: info
constexpr LogModuleInfo kModuleTable[] = {
    {LogModule::Root, LogModule::Root, "", "root"},
    {LogModule::Config, LogModule::Root, "config", "Config / YAML"},
    {LogModule::Radio, LogModule::Root, "radio", "Radio"},
    {LogModule::RadioUhd, LogModule::Radio, "radio.uhd", "UHD / USRP"},
    {LogModule::RadioSim, LogModule::Radio, "radio.sim", "Sim backend"},
    {LogModule::Sync, LogModule::Root, "sync", "Sync / CFO / align (debug)"},
    {LogModule::Demod, LogModule::Root, "demod", "Demodulator (debug)"},
    {LogModule::DemodLdpc, LogModule::Demod, "demod.ldpc", "LDPC decode (debug)"},
    {LogModule::DemodLdpcDetail, LogModule::DemodLdpc, "demod.ldpc.detail", "LDPC decode detail"},
    {LogModule::DemodLlr, LogModule::Demod, "demod.llr", "LLR / demap (debug)"},
    {LogModule::DemodSnr, LogModule::Demod, "demod.snr", "SNR diagnostics (debug)"},
    {LogModule::DemodProfiling, LogModule::Demod, "demod_profiling", "Demod load / latency (perf)"},
    {LogModule::DemodEqProfiling, LogModule::Demod, "demod_eq_profiling", "Equalizer breakdown (perf)"},
    {LogModule::Mod, LogModule::Root, "mod", "Modulator (debug)"},
    {LogModule::ModLdpc, LogModule::Mod, "mod.ldpc", "LDPC encode (debug)"},
    {LogModule::ModLdpcDetail, LogModule::ModLdpc, "mod.ldpc.detail", "LDPC encode detail"},
    {LogModule::ModProfiling, LogModule::Mod, "mod_profiling", "Mod load / latency (perf)"},
    {LogModule::ModLdpcProfiling, LogModule::ModLdpc, "mod_ldpc_profiling", "LDPC encode timing (perf)"},
    {LogModule::UlRx, LogModule::Root, "ul_rx", "Uplink RX (debug)"},
    {LogModule::UlTx, LogModule::Root, "ul_tx", "Uplink TX (debug)"},
    {LogModule::Arq, LogModule::Root, "arq", "ARQ (debug)"},
    {LogModule::ArqDl, LogModule::Arq, "arq.dl", "Downlink ARQ (debug)"},
    {LogModule::ArqUl, LogModule::Arq, "arq.ul", "Uplink ARQ (debug)"},
    {LogModule::Ertm, LogModule::Root, "ertm", "eRTM (debug)"},
    {LogModule::Sensing, LogModule::Root, "sensing", "Sensing (debug)"},
    {LogModule::SensingAggregate, LogModule::Sensing, "sensing.aggregate", "Sensing aggregate (debug)"},
    {LogModule::SensingProfiling, LogModule::Sensing, "sensing_profiling", "Sensing processing timers (perf)"},
    {LogModule::Recovery, LogModule::Root, "recovery", "Recovery / restart (debug)"},
    {LogModule::Agc, LogModule::Root, "agc", "AGC (debug)"},
    {LogModule::Ocxo, LogModule::Root, "ocxo", "OCXO / clock (debug)"},
    {LogModule::ChannelSim, LogModule::Root, "channelsim", "Channel simulator"},
    {LogModule::UdpEgressProfiling, LogModule::Root, "udp_egress_profiling", "UDP pacer stats (perf)"},
    {LogModule::Cuda, LogModule::Root, "cuda", "CUDA (debug)"},
    {LogModule::CudaDemod, LogModule::Cuda, "cuda.demod", "CUDA demod (debug)"},
    {LogModule::CudaDemodLdpc, LogModule::CudaDemod, "cuda.demod.ldpc", "CUDA LDPC decode (debug)"},
    {LogModule::CudaDemodProfiling, LogModule::CudaDemod, "cuda.demod_profiling", "CUDA demod load (perf)"},
    {LogModule::CudaDemodEqProfiling, LogModule::CudaDemod, "cuda.demod_eq_profiling", "CUDA EQ breakdown (perf)"},
    {LogModule::CudaMod, LogModule::Cuda, "cuda.mod", "CUDA mod (debug)"},
    {LogModule::CudaModLdpc, LogModule::CudaMod, "cuda.mod.ldpc", "CUDA LDPC encode (debug)"},
    {LogModule::CudaModProfiling, LogModule::CudaMod, "cuda.mod_profiling", "CUDA mod load (perf)"},
    {LogModule::CudaModLdpcProfiling, LogModule::CudaModLdpc, "cuda.mod_ldpc_profiling", "CUDA LDPC encode timing (perf)"},
    {LogModule::CudaSensing, LogModule::Cuda, "cuda.sensing", "CUDA sensing (debug)"},
    {LogModule::CudaSensingProfiling, LogModule::CudaSensing, "cuda.sensing_profiling", "CUDA sensing timers (perf)"},
    {LogModule::CudaLdpc, LogModule::Cuda, "cuda.ldpc", "CUDA LDPC shared (debug)"},
    {LogModule::CudaCfo, LogModule::Cuda, "cuda.cfo", "CUDA CFO (debug)"},
    {LogModule::CudaRecovery, LogModule::Cuda, "cuda.recovery", "CUDA recovery (debug)"},
};

bool path_is_profiling_module(const char* path) {
    if (!path || !*path) return false;
    const char* suf = "_profiling";
    const size_t n = std::char_traits<char>::length(path);
    const size_t sn = std::char_traits<char>::length(suf);
    return n >= sn && std::strcmp(path + (n - sn), suf) == 0;
}

static_assert(sizeof(kModuleTable) / sizeof(kModuleTable[0]) ==
                  static_cast<size_t>(LogModule::Count),
              "kModuleTable must list every LogModule");

std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

const char* level_tag(LogLevel level) {
    switch (level) {
        case LogLevel::Off: return "OFF  ";
        case LogLevel::Error: return "ERROR";
        case LogLevel::Warn: return "WARN ";
        case LogLevel::Info: return "INFO ";
        case LogLevel::Debug: return "DEBUG";
    }
    return "INFO ";
}

} // namespace

const LogModuleInfo* log_module_table(size_t& count) {
    count = static_cast<size_t>(LogModule::Count);
    return kModuleTable;
}

const LogModuleInfo& log_module_info(LogModule mod) {
    const size_t idx = static_cast<size_t>(mod);
    if (idx >= static_cast<size_t>(LogModule::Count)) {
        return kModuleTable[0];
    }
    return kModuleTable[idx];
}

const char* log_module_path(LogModule mod) {
    return log_module_info(mod).path;
}

const char* log_level_name(LogLevel level) {
    switch (level) {
        case LogLevel::Off: return "off";
        case LogLevel::Error: return "error";
        case LogLevel::Warn: return "warn";
        case LogLevel::Info: return "info";
        case LogLevel::Debug: return "debug";
    }
    return "info";
}

bool parse_log_level(const std::string& text, LogLevel& out) {
    const std::string t = to_lower_ascii(text);
    if (t == "off" || t == "none" || t == "silent") {
        out = LogLevel::Off;
        return true;
    }
    if (t == "error" || t == "err" || t == "e") {
        out = LogLevel::Error;
        return true;
    }
    if (t == "warn" || t == "warning" || t == "w") {
        out = LogLevel::Warn;
        return true;
    }
    if (t == "info" || t == "information" || t == "i") {
        out = LogLevel::Info;
        return true;
    }
    if (t == "debug" || t == "dbg" || t == "d" || t == "verbose") {
        out = LogLevel::Debug;
        return true;
    }
    if (t == "inherit" || t == "default" || t.empty()) {
        // Caller may treat inherit specially; still parse as Off+flag elsewhere.
        return false;
    }
    return false;
}

LogModule find_log_module_by_path(const std::string& path) {
    if (path.empty() || path == "root" || path == "*") {
        return LogModule::Root;
    }
    for (size_t i = 0; i < static_cast<size_t>(LogModule::Count); ++i) {
        if (path == kModuleTable[i].path) {
            return kModuleTable[i].id;
        }
    }
    return LogModule::Count; // not found
}

struct AsyncLogger::Impl {
    std::array<QueueState, static_cast<size_t>(QueueId::Count)> queues = {
        QueueState(8192), // G_ERROR
        QueueState(8192), // RT_ERROR
        QueueState(4096), // G_WARN
        QueueState(4096), // RT_WARN
        QueueState(4096), // G_INFO
        QueueState(4096), // RT_INFO
        QueueState(2048), // G_DEBUG
        QueueState(2048), // RT_DEBUG
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
    std::atomic<uint64_t> filtered{0};
    std::atomic<uint64_t> emitted{0};
    std::atomic<uint64_t> enqueued_rt{0};
    std::atomic<uint64_t> enqueued_guaranteed{0};

    // Filter state (written under config_mutex, read lock-free).
    mutable std::mutex config_mutex;
    LoggerRuntimeConfig config_snapshot{};
    std::atomic<uint8_t> default_level{static_cast<uint8_t>(LogLevel::Warn)};
    std::atomic<bool> timestamps{false};
    std::atomic<bool> force_error{true};
    // Per-module configured level; kLogLevelInherit means inherit parent.
    std::array<std::atomic<uint8_t>, static_cast<size_t>(LogModule::Count)> configured_level{};
    // Precomputed effective verbosity ceiling per module.
    std::array<std::atomic<uint8_t>, static_cast<size_t>(LogModule::Count)> effective_level{};

    Impl() {
        for (size_t i = 0; i < static_cast<size_t>(LogModule::Count); ++i) {
            configured_level[i].store(kLogLevelInherit, std::memory_order_relaxed);
            effective_level[i].store(static_cast<uint8_t>(LogLevel::Warn), std::memory_order_relaxed);
        }
    }

    void recompute_effective_levels_unlocked() {
        const uint8_t def = default_level.load(std::memory_order_relaxed);
        // Modules are ordered so parents appear before children in kModuleTable.
        for (size_t i = 0; i < static_cast<size_t>(LogModule::Count); ++i) {
            const LogModuleInfo& info = kModuleTable[i];
            const uint8_t cfg = configured_level[i].load(std::memory_order_relaxed);
            uint8_t eff = def;
            if (cfg != kLogLevelInherit) {
                eff = cfg;
            } else if (path_is_profiling_module(info.path)) {
                // Performance modules stay off until set explicitly (do not
                // inherit parent "info" which would enable expensive timers).
                eff = static_cast<uint8_t>(LogLevel::Off);
            } else if (info.parent != info.id) {
                eff = effective_level[static_cast<size_t>(info.parent)].load(std::memory_order_relaxed);
            } else {
                eff = def;
            }
            effective_level[i].store(eff, std::memory_order_relaxed);
        }
    }

    void configure(const LoggerRuntimeConfig& cfg) {
        std::lock_guard<std::mutex> lock(config_mutex);
        config_snapshot = cfg;
        default_level.store(static_cast<uint8_t>(cfg.default_level), std::memory_order_relaxed);
        timestamps.store(cfg.timestamps, std::memory_order_relaxed);
        force_error.store(cfg.force_error, std::memory_order_relaxed);

        for (size_t i = 0; i < static_cast<size_t>(LogModule::Count); ++i) {
            configured_level[i].store(kLogLevelInherit, std::memory_order_relaxed);
        }
        for (const auto& kv : cfg.module_levels) {
            const LogModule mod = find_log_module_by_path(kv.first);
            if (mod == LogModule::Count) {
                continue;
            }
            configured_level[static_cast<size_t>(mod)].store(
                static_cast<uint8_t>(kv.second), std::memory_order_relaxed);
        }
        recompute_effective_levels_unlocked();
    }

    LoggerRuntimeConfig snapshot_config() const {
        std::lock_guard<std::mutex> lock(config_mutex);
        return config_snapshot;
    }

    bool should_log(LogModule mod, LogLevel level) const {
        if (level == LogLevel::Error && force_error.load(std::memory_order_relaxed)) {
            return true;
        }
        const size_t idx = static_cast<size_t>(mod);
        const uint8_t eff = (idx < static_cast<size_t>(LogModule::Count))
            ? effective_level[idx].load(std::memory_order_relaxed)
            : default_level.load(std::memory_order_relaxed);
        if (eff == static_cast<uint8_t>(LogLevel::Off)) {
            return false;
        }
        return static_cast<uint8_t>(level) <= eff;
    }

    QueueState& queue_for_realtime(LogLevel level) {
        switch (level) {
            case LogLevel::Error: return queues[static_cast<size_t>(QueueId::RTError)];
            case LogLevel::Warn:  return queues[static_cast<size_t>(QueueId::RTWarn)];
            case LogLevel::Info:  return queues[static_cast<size_t>(QueueId::RTInfo)];
            case LogLevel::Debug:
            case LogLevel::Off:   return queues[static_cast<size_t>(QueueId::RTDebug)];
        }
        return queues[static_cast<size_t>(QueueId::RTInfo)];
    }

    QueueState& queue_for_guaranteed(LogLevel level) {
        switch (level) {
            case LogLevel::Error: return queues[static_cast<size_t>(QueueId::GError)];
            case LogLevel::Warn:  return queues[static_cast<size_t>(QueueId::GWarn)];
            case LogLevel::Info:  return queues[static_cast<size_t>(QueueId::GInfo)];
            case LogLevel::Debug:
            case LogLevel::Off:   return queues[static_cast<size_t>(QueueId::GDebug)];
        }
        return queues[static_cast<size_t>(QueueId::GInfo)];
    }

    static bool is_guaranteed_queue(QueueId id) {
        return id == QueueId::GError || id == QueueId::GWarn || id == QueueId::GInfo ||
               id == QueueId::GDebug;
    }

    static double rate_for_level(LogLevel level, int hz_override) {
        if (hz_override > 0) {
            return static_cast<double>(hz_override);
        }
        switch (level) {
            case LogLevel::Debug: return kDefaultDebugHz;
            case LogLevel::Info: return kDefaultInfoHz;
            case LogLevel::Warn: return kDefaultWarnHz;
            case LogLevel::Error: return kDefaultErrorHz;
            case LogLevel::Off: return kDefaultDebugHz;
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

    void format_prefix(std::ostream& os, LogLevel level, LogModule mod) const {
        if (timestamps.load(std::memory_order_relaxed)) {
            using clock = std::chrono::system_clock;
            const auto now = clock::now();
            const auto tt = clock::to_time_t(now);
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                                now.time_since_epoch())
                                .count() %
                            1000000;
            std::tm tm_buf{};
#if defined(_WIN32)
            localtime_s(&tm_buf, &tt);
#else
            localtime_r(&tt, &tm_buf);
#endif
            os << std::put_time(&tm_buf, "%H:%M:%S") << '.'
               << std::setw(6) << std::setfill('0') << us << std::setfill(' ') << ' ';
        }
        os << '[' << level_tag(level) << ']';
        const char* path = log_module_path(mod);
        if (path && path[0] != '\0') {
            os << " [" << path << ']';
        }
        os << ' ';
    }

    void emit_line(const LogMessage& msg) {
        std::ostream& os = (msg.level == LogLevel::Info || msg.level == LogLevel::Debug)
            ? std::cout
            : std::cerr;
        format_prefix(os, msg.level, msg.mod);
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

    void submit_realtime(LogLevel level, LogModule mod, std::string&& msg) {
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

        q.queue.push_back(LogMessage{level, mod, std::move(msg)});
        enqueued_rt.fetch_add(1, std::memory_order_relaxed);

        lock.unlock();
        wake_cv.notify_one();
    }

    void submit_guaranteed(LogLevel level, LogModule mod, std::string&& msg) {
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

        q.queue.push_back(LogMessage{level, mod, std::move(msg)});
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
        s.filtered = filtered.load(std::memory_order_relaxed);
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

void AsyncLogger::configure(const LoggerRuntimeConfig& cfg) {
    _impl->configure(cfg);
}

LoggerRuntimeConfig AsyncLogger::snapshot_config() const {
    return _impl->snapshot_config();
}

bool AsyncLogger::should_log(LogModule mod, LogLevel level) const {
    return _impl->should_log(mod, level);
}

bool AsyncLogger::should_emit_realtime(LogLevel level, uint32_t callsite_id, int hz_override, uint64_t& suppressed_count) {
    return _impl->should_emit_realtime(level, callsite_id, hz_override, suppressed_count);
}

void AsyncLogger::submit_realtime(LogLevel level, LogModule mod, std::string&& msg) {
    _impl->submit_realtime(level, mod, std::move(msg));
}

void AsyncLogger::submit_guaranteed(LogLevel level, LogModule mod, std::string&& msg) {
    _impl->submit_guaranteed(level, mod, std::move(msg));
}

void AsyncLogger::note_guaranteed_in_rt_violation() {
    _impl->guaranteed_in_rt_violation.fetch_add(1, std::memory_order_relaxed);
}

void AsyncLogger::note_filtered() {
    _impl->filtered.fetch_add(1, std::memory_order_relaxed);
}

LoggerStats AsyncLogger::snapshot_stats() {
    return instance()._impl->snapshot_stats();
}

} // namespace async_logger
