#include "SimBackend.hpp"

#include <chrono>
#include <stdexcept>
#include <thread>

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
    : _ring(std::move(ring)), _ctrl(std::move(ctrl)), _max_samps(max_samps) {}

size_t SimTxStream::send(const sample_t* buff, size_t nsamps,
                         const TxMetadata& /*metadata*/, double timeout) {
    // Single-channel sim: timed-burst metadata is ignored; the stream is treated
    // as continuous (RX sync recovers framing from the waveform itself). A zero
    // sample count (end-of-burst marker) is a no-op. The timeout is honored so a
    // stalled hub does not block the caller forever (clean shutdown).
    if (nsamps == 0 || buff == nullptr) return 0;
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

size_t SimRxStream::recv(sample_t* buff, size_t nsamps, RxMetadata& metadata,
                         double timeout, bool /*one_packet*/) {
    std::atomic<int>* running = _ctrl->valid() ? &_ctrl->block()->running : nullptr;

    // Per-stream sample clock: timestamp of the first sample about to be returned.
    const uint64_t first_sample = _ring->consumed();
    const double tick_rate = _ctrl->tick_rate() > 0.0 ? _ctrl->tick_rate() : 1.0;

    // Honor the timeout (like a real USRP) so the caller's loop can re-check its
    // own running flag and shut down even while the hub is still alive.
    const size_t got = _ring->pop_block(buff, nsamps, running, timeout);

    metadata.reset();
    metadata.has_time_spec = true;
    metadata.time_spec = TimeSpec::from_ticks(static_cast<long long>(first_sample), tick_rate);
    metadata.error_code = (got == nsamps) ? RxError::None : RxError::Timeout;
    return got;
}

void SimRxStream::issue_stream_cmd(const StreamCmd& /*cmd*/) {
    // The hub streams continuously; start/stop commands are no-ops in simulation.
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
}

bool SimDevice::supports(Capability cap) const {
    // The simulator has no real RF front end, no timed-burst engine, no async TX
    // event path, and no stream restart. It does model manual RF/DSP retuning so
    // the UE can feed both downlink RX and uplink TX corrections back to the hub.
    return cap == Capability::RfDspTune;
}

TimeSpec SimDevice::time_now() const {
    const double tick_rate = (_ctrl && _ctrl->tick_rate() > 0.0) ? _ctrl->tick_rate() : 1.0;
    const uint64_t idx = _ctrl ? _ctrl->sample_index() : 0;
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
