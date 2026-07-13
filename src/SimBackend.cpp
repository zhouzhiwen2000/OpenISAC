#include "SimBackend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

#include "AsyncLogger.hpp"

namespace radio {

using sim_shm::sample_t;  // std::complex<float>, identical to radio::sample_t

namespace {
// Pick the shared-memory ring suffix for a stream. Carried in StreamArgs so the
// generic get_tx_stream/get_rx_stream signature stays backend-neutral.
std::string ring_suffix(const StreamArgs& args, const char* fallback) {
    auto it = args.args.find("sim_suffix");
    if (it != args.args.end() && !it->second.empty()) return it->second;
    return fallback;
}
}  // namespace

// ---------------------------------------------------------------------------
// SimTxStream
// ---------------------------------------------------------------------------

SimTxStream::SimTxStream(std::shared_ptr<sim_shm::ShmRing> ring,
                         std::shared_ptr<sim_shm::ShmControl> ctrl, size_t max_samps)
    : _ring(std::move(ring)),
      _ctrl(std::move(ctrl)),
      _max_samps(max_samps),
      _zero_buffer(std::max<size_t>(1, std::min<size_t>(max_samps, 65536)), sample_t(0.0f, 0.0f)) {}

size_t SimTxStream::_write_gap(uint64_t sample_count, double timeout) {
    size_t written_total = 0;
    std::atomic<int>* running = _ctrl->valid() ? &_ctrl->block()->running : nullptr;
    while (sample_count > 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(
            sample_count, static_cast<uint64_t>(_zero_buffer.size())));
        const size_t written = _ring->push_block(
            _zero_buffer.data(), chunk, running, timeout);
        written_total += written;
        sample_count -= written;
        if (written < chunk) break;
    }
    return written_total;
}

size_t SimTxStream::send(const sample_t* buff, size_t nsamps,
                         const TxMetadata& metadata, double timeout) {
    // A zero-length EOB terminates the logical continuous burst. The next SOB
    // may establish a later absolute start, represented as a sparse-time gap in
    // front of the next samples rather than a second unrelated ring clock.
    if (nsamps == 0 || buff == nullptr) {
        if (metadata.end_of_burst) _in_burst = false;
        return 0;
    }

    const double tick_rate = _ctrl->tick_rate() > 0.0 ? _ctrl->tick_rate() : 1.0;
    const bool starts_burst = metadata.start_of_burst || !_in_burst;
    if (!_ring->timeline_origin_is_set()) {
        const int64_t requested_start = metadata.has_time_spec
            ? metadata.time_spec.to_ticks(tick_rate)
            : static_cast<int64_t>(_ctrl->clock_sample_index());
        if (requested_start < 0 || !_ring->set_timeline_origin(requested_start)) {
            return 0;
        }
    } else if (starts_burst && metadata.has_time_spec) {
        const int64_t requested_start = metadata.time_spec.to_ticks(tick_rate);
        const int64_t produced_until = _ring->absolute_produced();
        if (requested_start < produced_until) {
            // A real timed transmitter reports a late/time error here. The sim
            // has no async-event channel, so reject this submission as a short
            // send and let the existing engine recovery path handle it.
            return 0;
        }
        const uint64_t gap = static_cast<uint64_t>(requested_start - produced_until);
        if (gap > 0 && _write_gap(gap, timeout) != gap) {
            return 0;
        }
    }

    _in_burst = true;
    std::atomic<int>* running = _ctrl->valid() ? &_ctrl->block()->running : nullptr;
    return _ring->push_block(buff, nsamps, running, timeout);
}

bool SimTxStream::recv_async_msg(AsyncMetadata& /*metadata*/, double timeout) {
    // No async TX events in simulation. Sleep for the requested timeout so the
    // async-event thread does not busy-spin, then report "no message".
    if (timeout > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(timeout));
    }
    return false;
}

// ---------------------------------------------------------------------------
// SimRxStream
// ---------------------------------------------------------------------------

SimRxStream::SimRxStream(std::shared_ptr<sim_shm::ShmRing> ring,
                         std::shared_ptr<sim_shm::ShmControl> ctrl, size_t max_samps)
    : _ring(std::move(ring)), _ctrl(std::move(ctrl)), _max_samps(max_samps) {}

bool SimRxStream::_apply_pending_start(
    std::atomic<int>* running,
    double timeout,
    RxError& start_error)
{
    int64_t requested_start = 0;
    {
        std::lock_guard<std::mutex> lock(_stream_mutex);
        if (!_streaming) return false;
        if (!_start_pending) return true;
        requested_start = _requested_start_sample;
        _start_pending = false;
    }

    if (!_ring->wait_for_timeline_origin(running, timeout)) {
        std::lock_guard<std::mutex> lock(_stream_mutex);
        if (_streaming) _start_pending = true;
        return false;
    }

    const int64_t timeline_origin = _ring->timeline_origin();
    if (timeline_origin != sim_shm::kShmTimelineUnset && requested_start < timeline_origin) {
        // The RX process may start before the BS establishes the air timeline.
        // There are no samples before the producer's first timed TX, so align
        // the initial receive command to that authoritative origin.
        requested_start = timeline_origin;
    }

    const int64_t current = _ring->absolute_consumed();
    if (current == sim_shm::kShmTimelineUnset) {
        std::lock_guard<std::mutex> lock(_stream_mutex);
        if (_streaming) _start_pending = true;
        return false;
    }
    if (requested_start < current) {
        start_error = RxError::LateCommand;
        return true;
    }

    const uint64_t skip = static_cast<uint64_t>(requested_start - current);
    if (skip == 0) return true;
    if (skip > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        start_error = RxError::LateCommand;
        return true;
    }
    const size_t skipped = _ring->skip_block(
        static_cast<size_t>(skip), running, timeout);
    if (skipped != static_cast<size_t>(skip)) {
        std::lock_guard<std::mutex> lock(_stream_mutex);
        if (_streaming) _start_pending = true;
        return false;
    }
    return true;
}

size_t SimRxStream::recv(sample_t* buff, size_t nsamps, RxMetadata& metadata,
                         double timeout, bool /*one_packet*/) {
    std::atomic<int>* running = _ctrl->valid() ? &_ctrl->block()->running : nullptr;

    metadata.reset();
    RxError start_error = RxError::None;
    if (!_apply_pending_start(running, timeout, start_error)) {
        metadata.error_code = RxError::Timeout;
        return 0;
    }

    // Absolute shared sample clock: timestamp of the first sample about to be
    // returned after applying any timed-start seek.
    const int64_t first_sample = _ring->absolute_consumed();
    const double tick_rate = _ctrl->tick_rate() > 0.0 ? _ctrl->tick_rate() : 1.0;

    // Honor the timeout (like a real USRP) so the caller's loop can re-check its
    // own running flag and shut down even while the hub is still alive.
    const size_t got = _ring->pop_block(buff, nsamps, running, timeout);

    metadata.has_time_spec = true;
    metadata.time_spec = TimeSpec::from_ticks(first_sample, tick_rate);
    metadata.error_code = start_error != RxError::None
        ? start_error
        : ((got == nsamps) ? RxError::None : RxError::Timeout);
    return got;
}

void SimRxStream::issue_stream_cmd(const StreamCmd& cmd) {
    std::lock_guard<std::mutex> lock(_stream_mutex);
    if (cmd.mode == StreamMode::StopContinuous) {
        _streaming = false;
        _start_pending = false;
        return;
    }

    const double tick_rate = _ctrl->tick_rate() > 0.0 ? _ctrl->tick_rate() : 1.0;
    _requested_start_sample = cmd.stream_now
        ? static_cast<int64_t>(_ctrl->clock_sample_index())
        : cmd.time_spec.to_ticks(tick_rate);
    _streaming = true;
    _start_pending = true;
}

// ---------------------------------------------------------------------------
// SimDevice
// ---------------------------------------------------------------------------

SimDevice::SimDevice(const DeviceConfig& cfg)
    : _session(cfg.sim_session),
      _tick_rate(cfg.sim_tick_rate),
      _center_freq(cfg.sim_center_freq) {
    _ctrl = std::make_shared<sim_shm::ShmControl>();
    const std::string ctrl_name = sim_shm::make_shm_name(_session, "ctrl");
    if (!_ctrl->open(ctrl_name)) {
        throw std::runtime_error(
            "SimDevice: failed to connect to ChannelSimulator session '" + _session +
            "'. Start ChannelSimulator first.");
    }
    // Prefer the hub's authoritative tick rate when available.
    if (_ctrl->tick_rate() > 0.0) _tick_rate = _ctrl->tick_rate();

    if (cfg.sim_predictive_delay && _center_freq > 0.0) {
        const double configured_cfo_hz = _ctrl->configured_cfo_hz();
        const double configured_sro_ppm = _ctrl->configured_sample_rate_offset_ppm();
        const double cfo_implied_sro_ppm = configured_cfo_hz / _center_freq * 1.0e6;
        constexpr double kClockConsistencyTolerancePpm = 0.01;
        if (std::isfinite(cfo_implied_sro_ppm) && std::isfinite(configured_sro_ppm) &&
            std::abs(cfo_implied_sro_ppm - configured_sro_ppm) >
                kClockConsistencyTolerancePpm) {
            LOG_G_WARN_M(Config)
                << "[sim] sync_tracking.predictive_delay=true, but simulation.cfo_hz="
                << configured_cfo_hz << " Hz implies a shared-oscillator clock offset of "
                << cfo_implied_sro_ppm << " ppm at center_freq=" << _center_freq
                << " Hz, while simulation.sample_rate_offset_ppm=" << configured_sro_ppm
                << ". The simulator models CFO and SRO independently; predictive delay may "
                   "apply a false timing correction unless these values are consistent.";
        }
    }
}

bool SimDevice::supports(Capability cap) const {
    // Timed TX/RX starts are modeled on the shared absolute sample clock. The
    // simulator still has no real RF front end, async TX event path, or full
    // stream-restart/error machinery.
    return cap == Capability::TimedTx || cap == Capability::RfDspTune;
}

TimeSpec SimDevice::time_now() const {
    const double tick_rate = (_ctrl && _ctrl->tick_rate() > 0.0) ? _ctrl->tick_rate() : 1.0;
    const uint64_t idx = _ctrl ? _ctrl->clock_sample_index() : 0;
    return TimeSpec::from_ticks(static_cast<long long>(idx), tick_rate);
}

TuneResult SimDevice::set_tx_freq(const TuneRequest& req, size_t /*chan*/) {
    const bool manual_retune =
        req.rf_freq_policy == TunePolicy::Manual &&
        req.dsp_freq_policy == TunePolicy::Manual;
    const double dsp = (req.dsp_freq_policy == TunePolicy::Manual) ? req.dsp_freq : 0.0;
    if (manual_retune && _ctrl) {
        // Share the logical emitted-carrier shift, not UHD's TX DSP sign. The UE
        // constructs manual requests with target_freq = rf_freq + correction.
        _ctrl->set_uplink_tx_freq_correction_hz(req.target_freq - req.rf_freq);
    }
    TuneResult r;
    r.target_rf_freq = req.target_freq != 0.0 ? req.target_freq : _center_freq;
    r.actual_rf_freq =
        (req.rf_freq_policy == TunePolicy::Manual) ? req.rf_freq : r.target_rf_freq;
    r.target_dsp_freq = req.dsp_freq;
    r.actual_dsp_freq = dsp;
    return r;
}

TuneResult SimDevice::set_rx_freq(const TuneRequest& req, size_t /*chan*/) {
    // Only an explicit manual retune (the comm CFO-tracking path) writes the
    // hub's receiver-side correction. A plain center-frequency tune leaves it
    // untouched so it cannot race another stream's correction.
    const double dsp = (req.dsp_freq_policy == TunePolicy::Manual) ? req.dsp_freq : 0.0;
    if (req.dsp_freq_policy == TunePolicy::Manual && _ctrl) {
        _ctrl->set_comm_rx_freq_correction_hz(dsp);
    }
    TuneResult r;
    r.target_rf_freq = req.target_freq != 0.0 ? req.target_freq : _center_freq;
    r.actual_rf_freq = _center_freq;
    r.target_dsp_freq = req.dsp_freq;
    r.actual_dsp_freq = dsp;
    return r;
}

void SimDevice::set_rx_freq_correction(double hz) {
    if (_ctrl) _ctrl->set_comm_rx_freq_correction_hz(hz);
}

double SimDevice::rx_freq_correction() const {
    return _ctrl ? _ctrl->comm_rx_freq_correction_hz() : 0.0;
}

ITxStreamPtr SimDevice::get_tx_stream(const StreamArgs& args) {
    auto ring = std::make_shared<sim_shm::ShmRing>();
    const std::string name = sim_shm::make_shm_name(_session, ring_suffix(args, "tx"));
    std::atomic<int>* running = _ctrl->valid() ? &_ctrl->block()->running : nullptr;
    if (!ring->open(name, running)) {
        throw std::runtime_error("SimDevice: failed to open TX ring " + name +
                                 " (is ChannelSimulator running?)");
    }
    return std::make_shared<SimTxStream>(ring, _ctrl, /*max_samps*/ ring->capacity());
}

IRxStreamPtr SimDevice::get_rx_stream(const StreamArgs& args) {
    auto ring = std::make_shared<sim_shm::ShmRing>();
    const std::string name = sim_shm::make_shm_name(_session, ring_suffix(args, "rx"));
    std::atomic<int>* running = _ctrl->valid() ? &_ctrl->block()->running : nullptr;
    if (!ring->open(name, running)) {
        throw std::runtime_error("SimDevice: failed to open RX ring " + name +
                                 " (is ChannelSimulator running?)");
    }
    return std::make_shared<SimRxStream>(ring, _ctrl, /*max_samps*/ ring->capacity());
}

bool SimDevice::running() const {
    return _ctrl && _ctrl->running() != 0;
}

IDevicePtr make_sim_device(const DeviceConfig& cfg) {
    return std::make_shared<SimDevice>(cfg);
}

}  // namespace radio
