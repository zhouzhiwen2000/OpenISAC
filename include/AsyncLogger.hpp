#ifndef ASYNC_LOGGER_HPP
#define ASYNC_LOGGER_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <sstream>
#include <string>

namespace async_logger {

enum class LogLevel : uint8_t {
    Info = 0,
    Warn = 1,
    Error = 2,
};

enum class LoggerPath : uint8_t {
    Realtime = 0,
    Guaranteed = 1,
};

enum class LoggerThreadMode : uint8_t {
    NonRealtime = 0,
    Realtime = 1,
};

struct LoggerStats {
    uint64_t rt_dropped_lock = 0;
    uint64_t rt_dropped_queue = 0;
    uint64_t rt_dropped_rate = 0;
    uint64_t guaranteed_dropped = 0;
    uint64_t guaranteed_in_rt_violation = 0;
    uint64_t emitted = 0;
    uint64_t enqueued_rt = 0;
    uint64_t enqueued_guaranteed = 0;
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

class AsyncLogger {
public:
    static AsyncLogger& instance();

    void start();
    void shutdown();

    void set_thread_mode(LoggerThreadMode mode);
    LoggerThreadMode thread_mode() const;

    bool should_emit_realtime(LogLevel level, uint32_t callsite_id, int hz_override, uint64_t& suppressed_count);
    void submit_realtime(LogLevel level, std::string&& msg);
    void submit_guaranteed(LogLevel level, std::string&& msg);

    void note_guaranteed_in_rt_violation();
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
    LogLine(LoggerPath path, LogLevel level, uint32_t callsite_id, int hz_override = 0)
        : _path(path),
          _level(level),
          _callsite_id(callsite_id)
    {
        auto& logger = AsyncLogger::instance();

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
            logger.submit_realtime(_level, std::move(msg));
        } else {
            logger.submit_guaranteed(_level, std::move(msg));
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
    uint32_t _callsite_id;
    uint64_t _suppressed_count = 0;
    bool _enabled = true;
    bool _downgraded_to_rt = false;
    std::ostringstream _stream;
};

} // namespace async_logger

#define ASYNC_LOG_CALLSITE_ID (::async_logger::detail::callsite_id(__FILE__, static_cast<uint32_t>(__LINE__)))

#define LOG_RT_INFO() ::async_logger::LogLine(::async_logger::LoggerPath::Realtime, ::async_logger::LogLevel::Info, ASYNC_LOG_CALLSITE_ID)
#define LOG_RT_WARN() ::async_logger::LogLine(::async_logger::LoggerPath::Realtime, ::async_logger::LogLevel::Warn, ASYNC_LOG_CALLSITE_ID)
#define LOG_RT_ERROR() ::async_logger::LogLine(::async_logger::LoggerPath::Realtime, ::async_logger::LogLevel::Error, ASYNC_LOG_CALLSITE_ID)
#define LOG_RT_WARN_HZ(hz) ::async_logger::LogLine(::async_logger::LoggerPath::Realtime, ::async_logger::LogLevel::Warn, ASYNC_LOG_CALLSITE_ID, (hz))

#define LOG_G_INFO() ::async_logger::LogLine(::async_logger::LoggerPath::Guaranteed, ::async_logger::LogLevel::Info, ASYNC_LOG_CALLSITE_ID)
#define LOG_G_WARN() ::async_logger::LogLine(::async_logger::LoggerPath::Guaranteed, ::async_logger::LogLevel::Warn, ASYNC_LOG_CALLSITE_ID)
#define LOG_G_ERROR() ::async_logger::LogLine(::async_logger::LoggerPath::Guaranteed, ::async_logger::LogLevel::Error, ASYNC_LOG_CALLSITE_ID)

#endif // ASYNC_LOGGER_HPP
