#include "Xr1710gBackend.hpp"
#include "xr1710g/w1700k_stream.h"
#include "xr1710g/convert.hpp"
#include "xr1710g/pll_control.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

namespace radio {
namespace {
constexpr size_t kSlice = W1700K_STREAM_SLICE;
constexpr double kTwoPi = 6.283185307179586476925286766559;
using Clock = std::chrono::steady_clock;

void channel_zero(size_t chan) {
    if (chan != 0) throw std::invalid_argument("XR1710G supports RX channel 0 only");
}

bool valid_timeout(double seconds) {
    return std::isfinite(seconds) && seconds >= 0.0 && seconds <= 3600.0;
}

// Own the mapping and cache its validated layout. Do not use the ABI's open()
// helper: it does not validate offsets, power-of-two capacity or overflow.
class Mapping {
public:
    ~Mapping() { if (_header) munmap(_header, _bytes); }
    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;

    bool open(const std::string& name) {
        const int fd = shm_open(name.c_str(), O_RDONLY | O_CLOEXEC, 0);
        if (fd < 0) {
            if (errno == ENOENT) return false;
            throw std::runtime_error("XR1710G shm_open " + name + ": " + std::strerror(errno));
        }
        struct stat st{};
        if (fstat(fd, &st) != 0) {
            close(fd);
            throw std::runtime_error("XR1710G fstat failed");
        }
        // shm_open publishes the name before the producer sizes/initializes it.
        if (st.st_size == 0) { close(fd); return false; }
        if (st.st_size < W1700K_STREAM_HEADER_BYTES) {
            close(fd);
            throw std::runtime_error("XR1710G shared memory has a truncated header");
        }
        void* p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("XR1710G mmap failed");
        auto* h = static_cast<w1700k_stream_header*>(p);
        const uint64_t bytes = static_cast<uint64_t>(st.st_size);
        const uint32_t magic = w1700k_stream_load(&h->magic);
        if (magic == 0) { munmap(p, static_cast<size_t>(bytes)); return false; }
        const uint64_t n = h->ring_slices;
        const bool valid = magic == W1700K_STREAM_MAGIC &&
            h->version == W1700K_STREAM_VERSION &&
            h->header_bytes == W1700K_STREAM_HEADER_BYTES && h->slice_samples == kSlice &&
            h->total_bytes == bytes && n && !(n & (n - 1)) &&
            h->tags_offset >= W1700K_STREAM_HEADER_BYTES && h->tags_offset % 8 == 0 &&
            h->tags_offset <= bytes && n <= (bytes - h->tags_offset) / sizeof(uint64_t) &&
            h->samples_offset >= h->tags_offset + n * sizeof(uint64_t) &&
            h->samples_offset % alignof(int16_t) == 0 && h->samples_offset <= bytes &&
            n <= (bytes - h->samples_offset) / (kSlice * 2 * sizeof(int16_t));
        if (!valid) {
            munmap(p, static_cast<size_t>(bytes));
            throw std::runtime_error("XR1710G shared memory is not a valid w1700k-stream ABI v1 ring");
        }
        if (_header) munmap(_header, _bytes);
        _header = h;
        _bytes = static_cast<size_t>(bytes);
        _inode = static_cast<uint64_t>(st.st_ino);
        _capacity = n;
        _tags = reinterpret_cast<const uint64_t*>(static_cast<const char*>(p) + h->tags_offset);
        _samples = reinterpret_cast<const int16_t*>(static_cast<const char*>(p) + h->samples_offset);
        return true;
    }

    bool alive() const {
        if (!_header) return false;
        // Read heartbeat before the clock: a concurrent heartbeat update must
        // not appear to be in the future and underflow unsigned subtraction.
        const auto heartbeat = w1700k_stream_load(&_header->heartbeat_ns);
        const auto now = w1700k_stream_now_ns();
        return heartbeat && heartbeat <= now && now - heartbeat < 1000000000ull;
    }
    uint64_t inode() const { return _inode; }
    uint64_t capacity() const { return _capacity; }
    uint64_t rate() const { return w1700k_stream_load(&_header->radio_sample_rate); }
    uint64_t tag(uint64_t index) const {
        return w1700k_stream_load(_tags + (index & (_capacity - 1)));
    }
    void copy(uint64_t index, int16_t* dest) const {
        std::memcpy(dest, _samples + (index & (_capacity - 1)) * kSlice * 2,
                    kSlice * 2 * sizeof(int16_t));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
    }
    bool position(uint64_t& run, uint64_t& newest) const {
        // A crashed writer can leave its seqlock odd. Bound retries so recv()
        // still observes its timeout and the UE can terminate normally.
        for (int attempt = 0; attempt != 32; ++attempt) {
            const uint64_t before = w1700k_stream_load(&_header->position_seq);
            if (before & 1) continue;
            run = w1700k_stream_load(&_header->run);
            newest = w1700k_stream_load(&_header->newest);
            if (before == w1700k_stream_load(&_header->position_seq)) return true;
        }
        return false;
    }

private:
    w1700k_stream_header* _header = nullptr;
    size_t _bytes = 0;
    uint64_t _inode = 0, _capacity = 0;
    const uint64_t* _tags = nullptr;
    const int16_t* _samples = nullptr;
};

struct State {
    explicit State(Xr1710gConfig config) : cfg(std::move(config)) {}
    const Xr1710gConfig cfg;
    std::atomic<double> correction_hz{0.0};
    std::atomic<int64_t> ticks{0};
    std::atomic<bool> alive{false};
    std::atomic<bool> stream_created{false};
    std::unique_ptr<xr1710g::PllControl> pll;
};

class Xr1710gRxStream final : public IRxStream {
public:
    explicit Xr1710gRxStream(std::shared_ptr<State> state) : _state(std::move(state)) {
        const auto deadline = Clock::now() + std::chrono::duration<double>(_state->cfg.attach_timeout_s);
        do {
            auto candidate = std::make_unique<Mapping>();
            if (candidate->open(_state->cfg.shm_name) && candidate->alive()) {
                check_rate(*candidate);
                _map = std::move(candidate);
                _state->alive.store(true);
                return;
            }
            if (Clock::now() >= deadline) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } while (true);
        throw std::runtime_error("XR1710G: no live w1700k-stream at " + _state->cfg.shm_name +
                                 "; start the host receiver and RF stream first");
    }
    ~Xr1710gRxStream() override { _state->stream_created.store(false); }

    size_t num_channels() const override { return 1; }
    size_t max_num_samps() const override { return kSlice; }

    void issue_stream_cmd(const StreamCmd& cmd) override {
        if (cmd.mode == StreamMode::StopContinuous) {
            _started = false;
            _cached = kSlice;
            return;
        }
        if (cmd.mode != StreamMode::StartContinuous || !cmd.stream_now) {
            throw std::invalid_argument("XR1710G supports immediate continuous RX only; no timed/finite commands");
        }
        _started = true;
        _cached = kSlice;
        // Each start follows the current frontier, leaving other consumers and
        // the producer's restart_requests/control words untouched.
        _follow_latest = !_map->position(_run, _next);
        _first = true;
    }

    size_t recv(sample_t* buff, size_t nsamps, RxMetadata& md, double timeout,
                bool one_packet) override {
        md.reset();
        if (!valid_timeout(timeout)) throw std::invalid_argument("XR1710G invalid receive timeout");
        if (nsamps && !buff) throw std::invalid_argument("XR1710G null receive buffer");
        if (!nsamps) return 0;
        if (!_started) { md.error_code = RxError::Timeout; return 0; }
        const auto deadline = Clock::now() + std::chrono::duration<double>(timeout);
        size_t done = 0;
        while (done < nsamps) {
            if (_cached == kSlice) {
                const RxError error = next_slice();
                if (error != RxError::None && error != RxError::Timeout) {
                    // Do not return a buffer spanning a discontinuity. Existing
                    // UE recovery drops this read and reacquires synchronization.
                    md.reset();
                    md.error_code = error;
                    md.out_of_sequence = true;
                    return 0;
                }
                if (error == RxError::Timeout) {
                    if (Clock::now() >= deadline) {
                        md.error_code = RxError::Timeout;
                        return done;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                    continue;
                }
            }
            if (!done) {
                md.has_time_spec = true;
                md.time_spec = TimeSpec::from_ticks(_ticks, _state->cfg.sample_rate);
                md.fragment_offset = _cached;
                md.start_of_burst = _first;
                _first = false;
            }
            const size_t n = std::min(nsamps - done, kSlice - _cached);
            convert(buff + done, n);
            _cached += n;
            _ticks += static_cast<int64_t>(n);
            _state->ticks.store(_ticks, std::memory_order_relaxed);
            done += n;
            md.more_fragments = _cached < kSlice;
            if (one_packet) break;
        }
        return done;
    }

private:
    void check_rate(const Mapping& map) const {
        const uint64_t rate = map.rate();
        if (rate && std::abs(static_cast<double>(rate) - _state->cfg.sample_rate) > 0.5) {
            throw std::runtime_error("XR1710G source sample rate disagrees with rf_sampling.sample_rate");
        }
    }

    RxError gap(RxError error, uint64_t slices = 1) {
        _cached = kSlice;
        _ticks += static_cast<int64_t>(slices * kSlice);
        _state->ticks.store(_ticks, std::memory_order_relaxed);
        _phase = 0.0;
        _first = true;
        return error;
    }

    RxError next_slice() {
        if (!_map->alive()) {
            _state->alive.store(false, std::memory_order_relaxed);
            // A producer restart unlinks/recreates its shm object. Reopen only
            // when the old heartbeat expires; never write to either object.
            if (Clock::now() >= _next_reopen) {
                _next_reopen = Clock::now() + std::chrono::milliseconds(100);
                auto candidate = std::make_unique<Mapping>();
                if (candidate->open(_state->cfg.shm_name) && candidate->alive() &&
                    candidate->inode() != _map->inode()) {
                    check_rate(*candidate);
                    _map = std::move(candidate);
                    _run = 0;
                    _follow_latest = true;
                    _dead_reported = false;
                    _state->alive.store(true, std::memory_order_relaxed);
                    return gap(RxError::BrokenChain);
                }
            }
            if (!_dead_reported) {
                _dead_reported = true;
                return gap(RxError::BrokenChain);
            }
            return RxError::Timeout;
        }
        _state->alive.store(true, std::memory_order_relaxed);
        if (_dead_reported) {
            _dead_reported = false;
            _follow_latest = true;
            return gap(RxError::BrokenChain);
        }
        const uint64_t rate = _map->rate();
        if (rate && std::abs(static_cast<double>(rate) - _state->cfg.sample_rate) > 0.5) {
            _follow_latest = true;
            if (_rate_bad) return RxError::Timeout;
            _rate_bad = true;
            return gap(RxError::BadPacket);
        }
        if (_rate_bad) {
            _rate_bad = false;
            return gap(RxError::BrokenChain);
        }
        uint64_t run = 0, newest = 0;
        if (!_map->position(run, newest)) return RxError::Timeout;
        if (run >= (1ull << 21) || newest >= (1ull << 40)) return gap(RxError::BadPacket);
        if (_follow_latest) {
            _run = run;
            _next = newest;
            _follow_latest = false;
            _first = true;
        }
        if (!run) return RxError::Timeout;
        if (run != _run) {
            const bool had_run = _run != 0;
            _run = run;
            _next = had_run ? newest : 0;
            if (had_run) return gap(RxError::BrokenChain);
        }
        if (newest < _next) {
            _next = newest;
            return gap(RxError::BrokenChain);
        }
        if (_next == newest) return RxError::Timeout;
        if (newest - _next > _map->capacity()) {
            const uint64_t skipped = newest - _next;
            _next = newest;
            return gap(RxError::Overflow, skipped);
        }
        const uint64_t tag = _map->tag(_next);
        const uint64_t expected = w1700k_stream_tag(run, _next);
        if ((tag & ~W1700K_STREAM_TAG_KIND) != expected) {
            _next = newest;
            return gap(RxError::Overflow);
        }
        ++_next;
        if (tag & W1700K_STREAM_TAG_LOST) return gap(RxError::Overflow);
        if (tag & W1700K_STREAM_TAG_EVENT) return RxError::Timeout; // no samples/time in events
        _map->copy(_next - 1, _raw.data());
        if (_map->tag(_next - 1) != tag) {
            _next = newest;
            return gap(RxError::Overflow);
        }
        _cached = 0;
        return RxError::None;
    }

    void convert(sample_t* out, size_t n) {
        const int16_t* in = _raw.data() + _cached * 2;
        if (_state->pll) {
            xr1710g::normalize(in, out, n);
            return;
        }
        const double hz = _state->correction_hz.load(std::memory_order_relaxed);
        if (hz == 0.0 && _phase == 0.0) {
            xr1710g::normalize(in, out, n);
            return;
        }
        const double step = -kTwoPi * hz / _state->cfg.sample_rate;
        if (!_osc_valid || hz != _osc_hz) {
            for (size_t i = 0; i < kSlice; ++i)
                _osc[i] = sample_t(static_cast<float>(std::cos(step*i)), static_cast<float>(std::sin(step*i)));
            _osc_hz = hz;
            _osc_valid = true;
        }
        const float phase_i = static_cast<float>(std::cos(_phase));
        const float phase_q = static_cast<float>(std::sin(_phase));
        xr1710g::rotate(in, _osc.data(), out, n, phase_i, phase_q);
        _phase = std::remainder(_phase + step * n, kTwoPi);
    }

    std::shared_ptr<State> _state;
    std::unique_ptr<Mapping> _map;
    std::array<int16_t, kSlice * 2> _raw{};
    std::array<sample_t, kSlice> _osc{};
    size_t _cached = kSlice;
    uint64_t _run = 0, _next = 0;
    int64_t _ticks = 0;
    double _phase = 0.0, _osc_hz = 0.0;
    bool _osc_valid = false;
    bool _started = false, _follow_latest = true, _first = true, _dead_reported = false;
    bool _rate_bad = false;
    Clock::time_point _next_reopen{};
};

class Xr1710gDevice final : public IDevice {
public:
    explicit Xr1710gDevice(const DeviceConfig& cfg) : _state(std::make_shared<State>(cfg.xr1710g)) {
        validate_xr1710g_config(cfg.xr1710g);
        set_clock_source(cfg.clock_source);
        set_time_source(cfg.time_source);
        if (!cfg.xr1710g.pll_command.empty())
            _state->pll = std::make_unique<xr1710g::PllControl>(
                cfg.xr1710g.pll_command, cfg.xr1710g.center_freq, cfg.xr1710g.sample_rate);
    }
    bool supports(Capability cap) const override {
        return cap == Capability::RfDspTune || cap == Capability::StreamRestart;
    }
    void set_clock_source(const std::string& source) override {
        if (!source.empty() && source != "internal")
            throw std::invalid_argument("XR1710G has no external reference/PPS support; use internal clock/time sources");
    }
    void set_time_source(const std::string& source) override { set_clock_source(source); }
    TimeSpec time_now() const override {
        return TimeSpec::from_ticks(_state->ticks.load(std::memory_order_relaxed), _state->cfg.sample_rate);
    }
    double master_clock_rate() const override { return _state->cfg.sample_rate; }
    void set_rx_rate(double rate) override {
        if (!std::isfinite(rate) || std::abs(rate - _state->cfg.sample_rate) > 0.5)
            throw std::invalid_argument("XR1710G rate is set by its external controller; no resampling is performed");
    }
    double get_tx_rate(size_t) const override { throw std::logic_error("XR1710G is receive-only"); }
    double get_rx_rate(size_t chan) const override { channel_zero(chan); return _state->cfg.sample_rate; }
    size_t get_tx_num_channels() const override { return 0; }
    size_t get_rx_num_channels() const override { return 1; }
    void set_rx_bandwidth(double bw, size_t chan) override {
        channel_zero(chan);
        if (!std::isfinite(bw) || bw <= 0.0 || bw > _state->cfg.sample_rate)
            throw std::invalid_argument("XR1710G bandwidth must be positive and no larger than the source sample rate");
    }
    TuneResult set_tx_freq(const TuneRequest&, size_t) override { throw std::logic_error("XR1710G is receive-only"); }
    TuneResult set_rx_freq(const TuneRequest& req, size_t chan) override {
        channel_zero(chan);
        const double rf = _state->cfg.center_freq;
        if (req.rf_freq_policy == TunePolicy::Manual &&
            (!std::isfinite(req.rf_freq) || std::abs(req.rf_freq - rf) > 0.5))
            throw std::invalid_argument("XR1710G cannot retune RF; configure the external controller and restart UE");
        // UHD RX uses effective center = RF - DSP. The existing UE adds
        // -measured_CFO to its manual DSP request. Our physical correction
        // raises the LO (or mixes IQ down), so it has the opposite sign.
        const double dsp = req.dsp_freq_policy == TunePolicy::Manual ? req.dsp_freq :
            (req.dsp_freq_policy == TunePolicy::None ? -rx_freq_correction() : rf - req.target_freq);
        set_rx_freq_correction(-dsp);
        return TuneResult{req.target_freq, rf, dsp, -rx_freq_correction()};
    }
    void set_rx_freq_correction(double hz) override {
        if (!std::isfinite(hz) || std::abs(hz) >= _state->cfg.sample_rate / 2)
            throw std::invalid_argument("XR1710G frequency correction must be inside source Nyquist bandwidth");
        const double actual = _state->pll ? _state->pll->set(hz) : hz;
        _state->correction_hz.store(actual, std::memory_order_relaxed);
    }
    double rx_freq_correction() const override { return _state->correction_hz.load(std::memory_order_relaxed); }
    ITxStreamPtr get_tx_stream(const StreamArgs&) override { throw std::logic_error("XR1710G is receive-only; disable uplink"); }
    IRxStreamPtr get_rx_stream(const StreamArgs& args) override {
        if (args.channels != std::vector<size_t>{0} || args.cpu_format != "fc32" ||
            (!args.wire_format.empty() && args.wire_format != "sc16"))
            throw std::invalid_argument("XR1710G requires one channel [0], fc32 CPU format and sc16 wire format");
        if (_state->stream_created.exchange(true))
            throw std::logic_error("XR1710G device already has an RX stream");
        try { return std::make_shared<Xr1710gRxStream>(_state); }
        catch (...) { _state->stream_created.store(false); throw; }
    }
    bool running() const override { return _state->alive.load(std::memory_order_relaxed); }
private:
    std::shared_ptr<State> _state;
};
} // namespace

void validate_xr1710g_config(const Xr1710gConfig& cfg) {
    if (cfg.shm_name.size() < 2 || cfg.shm_name.front() != '/' || cfg.shm_name.find('/', 1) != std::string::npos)
        throw std::invalid_argument("radio.xr1710g_shm must be a POSIX shm name such as /xr1710g-live");
    if (cfg.sample_rate != 40e6 && cfg.sample_rate != 80e6 && cfg.sample_rate != 160e6 && cfg.sample_rate != 320e6)
        throw std::invalid_argument("XR1710G requires 40, 80, 160 or 320 MS/s; match the BS and external source");
    if (!std::isfinite(cfg.center_freq) || cfg.center_freq <= 0.0)
        throw std::invalid_argument("radio.xr1710g_center_freq must specify the source's actual RF center in Hz");
    if (!valid_timeout(cfg.attach_timeout_s))
        throw std::invalid_argument("radio.xr1710g_attach_timeout_s must be finite and within [0, 3600]");
}

IDevicePtr make_xr1710g_device(const DeviceConfig& cfg) {
    return std::make_shared<Xr1710gDevice>(cfg);
}
} // namespace radio
