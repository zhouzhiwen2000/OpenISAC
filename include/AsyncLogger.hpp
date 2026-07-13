#ifndef ASYNC_LOGGER_HPP
#define ASYNC_LOGGER_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace async_logger {

/**
 * @brief Log severity. Higher numeric values are more verbose.
 *
 * Emit rule: message level L is printed when L <= effective module level,
 * except Error which always prints when force_error is enabled.
 * Off is configuration-only (silence a module tree).
 */
enum class LogLevel : uint8_t {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
};

/** @brief Sentinel for "inherit parent / default" in module configuration. */
constexpr uint8_t kLogLevelInherit = 0xFF;

enum class LoggerPath : uint8_t {
    Realtime = 0,
    Guaranteed = 1,
};

enum class LoggerThreadMode : uint8_t {
    NonRealtime = 0,
    Realtime = 1,
};

/**
 * @brief Hierarchical log modules (compile-time IDs for RT-safe filtering).
 *
 * Paths use dotted notation, e.g. demod.ldpc.detail. Unlisted call sites use Root.
 */
enum class LogModule : uint16_t {
    Root = 0,
    Config,
    Radio,
    RadioUhd,
    RadioSim,
    Sync,
    Demod,
    DemodLdpc,
    DemodLdpcDetail,
    DemodLlr,
    DemodSnr,            // demod.snr — SNR diagnostics (debug, not timers)
    DemodProfiling,      // demod_profiling — demod load / latency reports
    DemodEqProfiling,    // demod_eq_profiling — equalizer stage breakdown
    Mod,
    ModLdpc,
    ModLdpcDetail,
    ModProfiling,        // mod_profiling — modulation load / latency
    ModLdpcProfiling,    // mod_ldpc_profiling — LDPC encode timing
    UlRx,
    UlTx,
    Arq,
    ArqDl,
    ArqUl,
    Ertm,
    Sensing,
    SensingAggregate,
    SensingProfiling,    // sensing_profiling — sensing processing timers
    Recovery,
    Agc,
    Ocxo,
    ChannelSim,
    UdpEgressProfiling,  // udp_egress_profiling — UDP pacer throughput stats
    Cuda,
    CudaDemod,
    CudaDemodLdpc,
    CudaDemodProfiling,
    CudaDemodEqProfiling,
    CudaMod,
    CudaModLdpc,
    CudaModProfiling,
    CudaModLdpcProfiling,
    CudaSensing,
    CudaSensingProfiling,
    CudaLdpc,
    CudaCfo,
    CudaRecovery,
    Count
};

struct LogModuleInfo {
    LogModule id;
    LogModule parent;   // parent == id means no parent (root of registry tree)
    const char* path;   // e.g. "demod.ldpc"; Root uses ""
    const char* title;  // short UI label
};

struct LoggerStats {
    uint64_t rt_dropped_lock = 0;
    uint64_t rt_dropped_queue = 0;
    uint64_t rt_dropped_rate = 0;
    uint64_t guaranteed_dropped = 0;
    uint64_t guaranteed_in_rt_violation = 0;
    uint64_t filtered = 0;
    uint64_t emitted = 0;
    uint64_t enqueued_rt = 0;
    uint64_t enqueued_guaranteed = 0;
};

/**
 * @brief Runtime filter configuration applied via configure().
 *
 * module_levels maps dotted path -> level name (or "inherit" / "off").
 * Paths not listed inherit their parent; Root inherits default_level.
 */
struct LoggerRuntimeConfig {
    LogLevel default_level = LogLevel::Warn;
    bool timestamps = false;
    bool force_error = true;
    std::unordered_map<std::string, LogLevel> module_levels;
};

namespace detail {
constexpr uint32_t fnv1a_32(const char* s, uint32_t hash = 2166136261u) {
    return (*s == '\0')
        ? hash
        : fnv1a_32(s + 1, (hash ^ static_cast<uint8_t>(*s)) * 16777619u);
}

constexpr uint32_t mix_callsite(uint32_t file_hash, uint32_t line) {
    return (file_hash ^ (line + 0x9e3779b9u + (file_hash << 6) + (file_hash >> 2)));
}

constexpr uint32_t callsite_id(const char* file, uint32_t line) {
    return mix_callsite(fnv1a_32(file), line);
}
} // namespace detail

/** @brief Full module registry (path, parent, title). */
const LogModuleInfo* log_module_table(size_t& count);
const LogModuleInfo& log_module_info(LogModule mod);
const char* log_module_path(LogModule mod);
const char* log_level_name(LogLevel level);
bool parse_log_level(const std::string& text, LogLevel& out);
LogModule find_log_module_by_path(const std::string& path);

class AsyncLogger {
public:
    static AsyncLogger& instance();

    void start();
    void shutdown();

    void set_thread_mode(LoggerThreadMode mode);
    LoggerThreadMode thread_mode() const;

    /** @brief Apply hierarchical filter config (lock-free reads after return). */
    void configure(const LoggerRuntimeConfig& cfg);
    LoggerRuntimeConfig snapshot_config() const;

    bool should_log(LogModule mod, LogLevel level) const;
    bool should_emit_realtime(LogLevel level, uint32_t callsite_id, int hz_override, uint64_t& suppressed_count);
    void submit_realtime(LogLevel level, LogModule mod, std::string&& msg);
    void submit_guaranteed(LogLevel level, LogModule mod, std::string&& msg);

    void note_guaranteed_in_rt_violation();
    void note_filtered();
    static LoggerStats snapshot_stats();

private:
    AsyncLogger();
    ~AsyncLogger();

    AsyncLogger(const AsyncLogger&) = delete;
    AsyncLogger& operator=(const AsyncLogger&) = delete;

    struct Impl;
    Impl* _impl;
};

class LoggerThreadModeGuard {
public:
    explicit LoggerThreadModeGuard(LoggerThreadMode mode)
        : _prev(AsyncLogger::instance().thread_mode())
    {
        AsyncLogger::instance().set_thread_mode(mode);
    }

    ~LoggerThreadModeGuard() {
        AsyncLogger::instance().set_thread_mode(_prev);
    }

private:
    LoggerThreadMode _prev;
};

class AsyncLoggerGuard {
public:
    AsyncLoggerGuard() {
        AsyncLogger::instance().start();
    }

    ~AsyncLoggerGuard() {
        AsyncLogger::instance().shutdown();
    }
};

class LogLine {
public:
    LogLine(LoggerPath path, LogLevel level, LogModule mod, uint32_t callsite_id, int hz_override = 0)
        : _path(path),
          _level(level),
          _mod(mod),
          _callsite_id(callsite_id)
    {
        auto& logger = AsyncLogger::instance();
        if (!logger.should_log(_mod, _level)) {
            _enabled = false;
            logger.note_filtered();
            return;
        }

        if (_path == LoggerPath::Realtime) {
            _enabled = logger.should_emit_realtime(_level, _callsite_id, hz_override, _suppressed_count);
            return;
        }

        if (logger.thread_mode() == LoggerThreadMode::Realtime) {
            logger.note_guaranteed_in_rt_violation();
            _downgraded_to_rt = true;
            _enabled = logger.should_emit_realtime(_level, _callsite_id, hz_override, _suppressed_count);
        }
    }

    ~LogLine() {
        if (!_enabled) return;

        std::string msg = _stream.str();
        if (msg.empty()) return;

        if (_suppressed_count > 0) {
            msg = "[rl_drop=" + std::to_string(_suppressed_count) + "] " + msg;
        }

        auto& logger = AsyncLogger::instance();
        if (_path == LoggerPath::Realtime || _downgraded_to_rt) {
            logger.submit_realtime(_level, _mod, std::move(msg));
        } else {
            logger.submit_guaranteed(_level, _mod, std::move(msg));
        }
    }

    template<typename T>
    LogLine& operator<<(const T& value) {
        if (_enabled) {
            _stream << value;
        }
        return *this;
    }

    LogLine& operator<<(std::ostream& (*manip)(std::ostream&)) {
        if (_enabled) {
            manip(_stream);
        }
        return *this;
    }

    LogLine& operator<<(std::ios& (*manip)(std::ios&)) {
        if (_enabled) {
            manip(_stream);
        }
        return *this;
    }

    LogLine& operator<<(std::ios_base& (*manip)(std::ios_base&)) {
        if (_enabled) {
            manip(_stream);
        }
        return *this;
    }

private:
    LoggerPath _path;
    LogLevel _level;
    LogModule _mod;
    uint32_t _callsite_id;
    uint64_t _suppressed_count = 0;
    bool _enabled = true;
    bool _downgraded_to_rt = false;
    std::ostringstream _stream;
};

} // namespace async_logger

#define ASYNC_LOG_CALLSITE_ID (::async_logger::detail::callsite_id(__FILE__, static_cast<uint32_t>(__LINE__)))

// Module-aware macros. Prefer these for new / retagged call sites.
// Example: LOG_G_INFO_M(DemodLdpc) << "blocks=" << n;
#define LOG_G_M(mod, level) \
    ::async_logger::LogLine(::async_logger::LoggerPath::Guaranteed, (level), \
                            ::async_logger::LogModule::mod, ASYNC_LOG_CALLSITE_ID)
#define LOG_RT_M(mod, level) \
    ::async_logger::LogLine(::async_logger::LoggerPath::Realtime, (level), \
                            ::async_logger::LogModule::mod, ASYNC_LOG_CALLSITE_ID)

#define LOG_G_DEBUG_M(mod) LOG_G_M(mod, ::async_logger::LogLevel::Debug)
#define LOG_G_INFO_M(mod)  LOG_G_M(mod, ::async_logger::LogLevel::Info)
#define LOG_G_WARN_M(mod)  LOG_G_M(mod, ::async_logger::LogLevel::Warn)
#define LOG_G_ERROR_M(mod) LOG_G_M(mod, ::async_logger::LogLevel::Error)

#define LOG_RT_DEBUG_M(mod) LOG_RT_M(mod, ::async_logger::LogLevel::Debug)
#define LOG_RT_INFO_M(mod)  LOG_RT_M(mod, ::async_logger::LogLevel::Info)
#define LOG_RT_WARN_M(mod)  LOG_RT_M(mod, ::async_logger::LogLevel::Warn)
#define LOG_RT_ERROR_M(mod) LOG_RT_M(mod, ::async_logger::LogLevel::Error)
#define LOG_RT_WARN_HZ_M(mod, hz) \
    ::async_logger::LogLine(::async_logger::LoggerPath::Realtime, ::async_logger::LogLevel::Warn, \
                            ::async_logger::LogModule::mod, ASYNC_LOG_CALLSITE_ID, (hz))
#define LOG_RT_INFO_HZ_M(mod, hz) \
    ::async_logger::LogLine(::async_logger::LoggerPath::Realtime, ::async_logger::LogLevel::Info, \
                            ::async_logger::LogModule::mod, ASYNC_LOG_CALLSITE_ID, (hz))
#define LOG_RT_DEBUG_HZ_M(mod, hz) \
    ::async_logger::LogLine(::async_logger::LoggerPath::Realtime, ::async_logger::LogLevel::Debug, \
                            ::async_logger::LogModule::mod, ASYNC_LOG_CALLSITE_ID, (hz))

// Legacy macros — Root module (still subject to default_level / force_error).
#define LOG_RT_DEBUG() LOG_RT_DEBUG_M(Root)
#define LOG_RT_INFO()  LOG_RT_INFO_M(Root)
#define LOG_RT_WARN()  LOG_RT_WARN_M(Root)
#define LOG_RT_ERROR() LOG_RT_ERROR_M(Root)
#define LOG_RT_WARN_HZ(hz) LOG_RT_WARN_HZ_M(Root, hz)

#define LOG_G_DEBUG() LOG_G_DEBUG_M(Root)
#define LOG_G_INFO()  LOG_G_INFO_M(Root)
#define LOG_G_WARN()  LOG_G_WARN_M(Root)
#define LOG_G_ERROR() LOG_G_ERROR_M(Root)

/**
 * @brief Whether a module is enabled at the given severity.
 *
 * Use for gating expensive work (timers, breakdowns). Text logs already filter
 * via LOG_*_M. Performance modules use paths ending in `_profiling` and do
 * **not** inherit parent verbosity (stay off until set explicitly).
 */
#define LOG_MOD_ON(mod) \
    (::async_logger::AsyncLogger::instance().should_log( \
        ::async_logger::LogModule::mod, ::async_logger::LogLevel::Info))
#define LOG_MOD_DEBUG_ON(mod) \
    (::async_logger::AsyncLogger::instance().should_log( \
        ::async_logger::LogModule::mod, ::async_logger::LogLevel::Debug))

#endif // ASYNC_LOGGER_HPP
