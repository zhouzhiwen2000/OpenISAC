#include "SimStreamer.hpp"

#include <chrono>
#include <thread>

using sim_shm::sample_t;

// ---------------------------------------------------------------------------
// SimTxStreamer
// ---------------------------------------------------------------------------

SimTxStreamer::SimTxStreamer(std::shared_ptr<sim_shm::ShmRing> ring,
                             std::shared_ptr<sim_shm::ShmControl> ctrl,
                             size_t max_samps)
    : _ring(std::move(ring)), _ctrl(std::move(ctrl)), _max_samps(max_samps) {}

size_t SimTxStreamer::send(const buffs_type& buffs,
                           size_t nsamps_per_buff,
                           const uhd::tx_metadata_t& /*metadata*/,
                           double timeout) {
    // Single-channel sim: timed-burst metadata is ignored; the stream is treated
    // as continuous (RX sync recovers framing from the waveform itself). The timeout
    // is honored so a stalled hub does not block the caller forever (clean shutdown).
    const sample_t* data = reinterpret_cast<const sample_t*>(buffs[0]);
    std::atomic<int>* running = _ctrl->valid() ? &_ctrl->block()->running : nullptr;
    return _ring->push_block(data, nsamps_per_buff, running, timeout);
}

bool SimTxStreamer::recv_async_msg(uhd::async_metadata_t& /*async_metadata*/, double timeout) {
    // No async TX events in simulation. Sleep for the requested timeout so the
    // modulator's async-event thread does not busy-spin, then report "no message".
    if (timeout > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(timeout));
    }
    return false;
}

// ---------------------------------------------------------------------------
// SimRxStreamer
// ---------------------------------------------------------------------------

SimRxStreamer::SimRxStreamer(std::shared_ptr<sim_shm::ShmRing> ring,
                             std::shared_ptr<sim_shm::ShmControl> ctrl,
                             size_t max_samps)
    : _ring(std::move(ring)), _ctrl(std::move(ctrl)), _max_samps(max_samps) {}

size_t SimRxStreamer::recv(const buffs_type& buffs,
                           size_t nsamps_per_buff,
                           uhd::rx_metadata_t& metadata,
                           double timeout,
                           bool /*one_packet*/) {
    sample_t* out = reinterpret_cast<sample_t*>(buffs[0]);
    std::atomic<int>* running = _ctrl->valid() ? &_ctrl->block()->running : nullptr;

    // Per-stream sample clock: timestamp of the first sample about to be returned.
    const uint64_t first_sample = _ring->consumed();
    const double tick_rate = _ctrl->tick_rate() > 0.0 ? _ctrl->tick_rate() : 1.0;

    // Honor the timeout (like a real USRP) so the caller's loop can re-check its own
    // running flag and shut down even while the hub is still alive.
    const size_t got = _ring->pop_block(out, nsamps_per_buff, running, timeout);

    metadata.reset();
    metadata.has_time_spec = true;
    metadata.time_spec = uhd::time_spec_t::from_ticks(static_cast<long long>(first_sample), tick_rate);
    metadata.start_of_burst = false;
    metadata.end_of_burst = false;
    metadata.more_fragments = false;
    metadata.fragment_offset = 0;
    metadata.out_of_sequence = false;
    metadata.error_code = (got == nsamps_per_buff)
                              ? uhd::rx_metadata_t::ERROR_CODE_NONE
                              : uhd::rx_metadata_t::ERROR_CODE_TIMEOUT;
    return got;
}

void SimRxStreamer::issue_stream_cmd(const uhd::stream_cmd_t& /*stream_cmd*/) {
    // The hub streams continuously; start/stop commands are no-ops in simulation.
}

#ifdef UHD_HAS_STREAMER_ACTION_HOOKS
void SimTxStreamer::post_output_action(
    const std::shared_ptr<uhd::rfnoc::action_info>& /*action*/, size_t /*port*/) {
    // Simulation streamers do not expose RFNoC edges; actions are ignored.
}

void SimRxStreamer::post_input_action(
    const std::shared_ptr<uhd::rfnoc::action_info>& /*action*/, size_t /*port*/) {
    // Simulation streamers do not expose RFNoC edges; actions are ignored.
}
#endif

// ---------------------------------------------------------------------------
// SimRadio
// ---------------------------------------------------------------------------

bool SimRadio::connect(const SimConfig& sim_cfg) {
    _cfg = sim_cfg;
    _ctrl = std::make_shared<sim_shm::ShmControl>();
    const std::string ctrl_name = sim_shm::make_shm_name(_cfg.session, "ctrl");
    return _ctrl->open(ctrl_name);
}

uhd::tx_streamer::sptr SimRadio::make_tx_streamer(size_t max_samps) {
    auto ring = std::make_shared<sim_shm::ShmRing>();
    const std::string name = sim_shm::make_shm_name(_cfg.session, "tx");
    std::atomic<int>* running = _ctrl->valid() ? &_ctrl->block()->running : nullptr;
    if (!ring->open(name, running)) {
        throw std::runtime_error("SimRadio: failed to open TX ring " + name +
                                 " (is ChannelSimulator running?)");
    }
    return std::make_shared<SimTxStreamer>(ring, _ctrl, max_samps);
}

uhd::rx_streamer::sptr SimRadio::make_rx_streamer(const std::string& suffix, size_t max_samps) {
    auto ring = std::make_shared<sim_shm::ShmRing>();
    const std::string name = sim_shm::make_shm_name(_cfg.session, suffix);
    std::atomic<int>* running = _ctrl->valid() ? &_ctrl->block()->running : nullptr;
    if (!ring->open(name, running)) {
        throw std::runtime_error("SimRadio: failed to open RX ring " + name +
                                 " (is ChannelSimulator running?)");
    }
    return std::make_shared<SimRxStreamer>(ring, _ctrl, max_samps);
}

uhd::time_spec_t SimRadio::time_now() const {
    const double tick_rate = (_ctrl && _ctrl->tick_rate() > 0.0) ? _ctrl->tick_rate() : 1.0;
    const uint64_t idx = _ctrl ? _ctrl->sample_index() : 0;
    return uhd::time_spec_t::from_ticks(static_cast<long long>(idx), tick_rate);
}

void SimRadio::set_comm_rx_freq_correction_hz(double value_hz) {
    if (_ctrl) {
        _ctrl->set_comm_rx_freq_correction_hz(value_hz);
    }
}

double SimRadio::comm_rx_freq_correction_hz() const {
    return _ctrl ? _ctrl->comm_rx_freq_correction_hz() : 0.0;
}

bool SimRadio::running() const {
    return _ctrl && _ctrl->running() != 0;
}
