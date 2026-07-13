#ifndef RADIO_TYPES_HPP
#define RADIO_TYPES_HPP

// Native radio HAL value types.
//
// These are the backend-independent vocabulary the BS/UE/sensing/uplink engines
// speak. They deliberately mirror the subset of UHD value types the project used
// to depend on (time_spec_t, tx/rx_metadata_t, stream_cmd_t, tune_request_t/...),
// but carry ZERO dependency on UHD. The UHD backend translates radio:: <-> uhd::
// at its boundary; the simulation backend (and any future backend) uses these
// types directly and never includes a UHD header.
//
// Header-only and free of any heavy include so it is cheap to pull into hot-path
// translation units and safe to include from CUDA (.cu/.cuh) sources.

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace radio {

// Host-side sample type for every stream (matches the UHD "fc32" CPU format and
// sim_shm::sample_t).
using sample_t = std::complex<float>;

// ---------------------------------------------------------------------------
// TimeSpec — sample-precise time, mirroring uhd::time_spec_t semantics.
//
// Stored as a full-seconds integer plus a fractional-seconds double so that
// sample-accurate frame timing survives over long runtimes (a single double of
// seconds would lose sub-sample precision after many hours at a high tick rate).
// from_ticks/to_ticks reproduce UHD's exact algorithm so values round-trip
// identically across the sim shared-memory boundary and the UHD backend.
// ---------------------------------------------------------------------------
class TimeSpec {
public:
    TimeSpec() = default;

    // Construct from a real number of seconds.
    explicit TimeSpec(double secs) : _full_secs(0), _frac_secs(secs) { _normalize(); }

    // Construct from explicit full + fractional seconds (used by the UHD backend
    // translation, which preserves UHD's internal representation exactly).
    TimeSpec(int64_t full_secs, double frac_secs)
        : _full_secs(full_secs), _frac_secs(frac_secs) {
        _normalize();
    }

    // Build a time from a sample/tick count at a given tick rate.
    static TimeSpec from_ticks(long long ticks, double tick_rate) {
        const long long rate_i = static_cast<long long>(tick_rate);
        const double rate_f = tick_rate - static_cast<double>(rate_i);
        if (rate_i == 0) {
            return TimeSpec(0, static_cast<double>(ticks) / tick_rate);
        }
        const long long secs_full = ticks / rate_i;
        const long long ticks_error = ticks - (secs_full * rate_i);
        const double ticks_frac = static_cast<double>(ticks_error) -
                                  static_cast<double>(secs_full) * rate_f;
        return TimeSpec(static_cast<int64_t>(secs_full), ticks_frac / tick_rate);
    }

    // Convert this time to an integer sample/tick count at a given tick rate.
    long long to_ticks(double tick_rate) const {
        const long long rate_i = static_cast<long long>(tick_rate);
        const double rate_f = tick_rate - static_cast<double>(rate_i);
        const long long ticks_full = _full_secs * rate_i;
        const double ticks_error = static_cast<double>(_full_secs) * rate_f;
        const double ticks_frac = _frac_secs * tick_rate;
        return ticks_full + static_cast<long long>(std::llround(ticks_error + ticks_frac));
    }

    double get_real_secs() const { return static_cast<double>(_full_secs) + _frac_secs; }
    int64_t get_full_secs() const { return _full_secs; }
    double get_frac_secs() const { return _frac_secs; }

    TimeSpec& operator+=(const TimeSpec& rhs) {
        _full_secs += rhs._full_secs;
        _frac_secs += rhs._frac_secs;
        _normalize();
        return *this;
    }
    TimeSpec& operator-=(const TimeSpec& rhs) {
        _full_secs -= rhs._full_secs;
        _frac_secs -= rhs._frac_secs;
        _normalize();
        return *this;
    }

private:
    void _normalize() {
        const double frac_int = std::floor(_frac_secs);
        _full_secs += static_cast<int64_t>(frac_int);
        _frac_secs -= frac_int;  // now in [0, 1)
    }

    int64_t _full_secs = 0;
    double _frac_secs = 0.0;
};

inline TimeSpec operator+(TimeSpec lhs, const TimeSpec& rhs) { lhs += rhs; return lhs; }
inline TimeSpec operator-(TimeSpec lhs, const TimeSpec& rhs) { lhs -= rhs; return lhs; }
inline bool operator<(const TimeSpec& a, const TimeSpec& b) {
    return a.get_real_secs() < b.get_real_secs();
}
inline bool operator>(const TimeSpec& a, const TimeSpec& b) { return b < a; }
inline bool operator<=(const TimeSpec& a, const TimeSpec& b) { return !(b < a); }
inline bool operator>=(const TimeSpec& a, const TimeSpec& b) { return !(a < b); }

// ---------------------------------------------------------------------------
// Stream metadata
// ---------------------------------------------------------------------------

struct TxMetadata {
    bool start_of_burst = false;
    bool end_of_burst = false;
    bool has_time_spec = false;
    TimeSpec time_spec{};
};

enum class RxError {
    None,
    Timeout,
    LateCommand,
    BrokenChain,
    Overflow,
    Alignment,
    BadPacket,
};

struct RxMetadata {
    bool has_time_spec = false;
    TimeSpec time_spec{};
    bool start_of_burst = false;
    bool end_of_burst = false;
    bool more_fragments = false;
    size_t fragment_offset = 0;
    bool out_of_sequence = false;
    RxError error_code = RxError::None;

    void reset() { *this = RxMetadata{}; }

    const char* strerror() const {
        switch (error_code) {
            case RxError::None: return "no error";
            case RxError::Timeout: return "timeout";
            case RxError::LateCommand: return "late command";
            case RxError::BrokenChain: return "broken chain";
            case RxError::Overflow: return "overflow";
            case RxError::Alignment: return "alignment";
            case RxError::BadPacket: return "bad packet";
        }
        return "unknown";
    }
};

enum class AsyncEvent {
    BurstAck,
    Underflow,
    UnderflowInPacket,
    SeqError,
    SeqErrorInBurst,
    TimeError,
    UserPayload,
    Unknown,
};

struct AsyncMetadata {
    AsyncEvent event_code = AsyncEvent::Unknown;
    uint32_t channel = 0;
    bool has_time_spec = false;
    TimeSpec time_spec{};
};

// ---------------------------------------------------------------------------
// Stream commands
// ---------------------------------------------------------------------------

enum class StreamMode {
    StartContinuous,
    StopContinuous,
    NumSampsAndDone,
    NumSampsAndMore,
};

struct StreamCmd {
    StreamMode mode = StreamMode::StartContinuous;
    bool stream_now = true;
    TimeSpec time_spec{};
    size_t num_samps = 0;

    StreamCmd() = default;
    explicit StreamCmd(StreamMode m) : mode(m) {}
};

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

enum class TunePolicy { None, Auto, Manual };

struct TuneRequest {
    double target_freq = 0.0;
    double rf_freq = 0.0;
    double dsp_freq = 0.0;
    TunePolicy rf_freq_policy = TunePolicy::Auto;
    TunePolicy dsp_freq_policy = TunePolicy::Auto;

    TuneRequest() = default;
    explicit TuneRequest(double target) : target_freq(target) {}
};

struct TuneResult {
    double target_rf_freq = 0.0;
    double actual_rf_freq = 0.0;
    double target_dsp_freq = 0.0;
    double actual_dsp_freq = 0.0;
};

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

struct GainRange {
    double start = 0.0;
    double stop = 0.0;
    double step = 0.0;
};

// Stream construction arguments (mirrors uhd::stream_args_t for the subset used).
// `args` carries free-form key/value hints; the sim backend reads "sim_suffix" to
// pick the shared-memory ring name, which the UHD backend ignores.
struct StreamArgs {
    std::string cpu_format = "fc32";
    std::string wire_format;
    std::vector<size_t> channels{0};
    std::map<std::string, std::string> args;

    StreamArgs() = default;
    explicit StreamArgs(std::string wire) : wire_format(std::move(wire)) {}
    StreamArgs(std::string cpu, std::string wire)
        : cpu_format(std::move(cpu)), wire_format(std::move(wire)) {}
};

}  // namespace radio

#endif  // RADIO_TYPES_HPP
