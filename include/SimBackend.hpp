#ifndef SIM_BACKEND_HPP
#define SIM_BACKEND_HPP

// Simulation radio backend — implements the radio:: HAL over the shared-memory
// channel simulator (ShmRing / ShmControl). Contains ZERO dependency on UHD:
// it speaks only radio:: native types and POSIX shared memory.
//
// The simulator is a peer backend behind radio::IDevice, selected by
// make_device() when DeviceConfig::backend == "sim".

#include <memory>
#include <string>

#include "RadioBackend.hpp"
#include "ShmRing.hpp"

namespace radio {

// Build a simulation device connected to the ChannelSimulator hub identified by
// cfg.sim_session. Blocks until the hub's control block is available. Throws on
// timeout. Caching/identity is handled by make_device(); this always constructs.
IDevicePtr make_sim_device(const DeviceConfig& cfg);

// TX stream: producer into a named shared-memory ring. Timed-burst metadata is
// ignored (the stream is continuous; RX recovers framing from the waveform).
class SimTxStream : public ITxStream {
public:
    SimTxStream(std::shared_ptr<sim_shm::ShmRing> ring,
                std::shared_ptr<sim_shm::ShmControl> ctrl, size_t max_samps);

    size_t num_channels() const override { return 1; }
    size_t max_num_samps() const override { return _max_samps; }
    size_t send(const sample_t* buff, size_t nsamps,
                const TxMetadata& metadata, double timeout) override;
    bool recv_async_msg(AsyncMetadata& metadata, double timeout) override;

private:
    std::shared_ptr<sim_shm::ShmRing> _ring;
    std::shared_ptr<sim_shm::ShmControl> _ctrl;
    size_t _max_samps;
};

// RX stream: consumer of a named shared-memory ring. Synthesizes RX metadata
// (timestamp) from the per-stream sample clock and the hub tick rate.
class SimRxStream : public IRxStream {
public:
    SimRxStream(std::shared_ptr<sim_shm::ShmRing> ring,
                std::shared_ptr<sim_shm::ShmControl> ctrl, size_t max_samps);

    size_t num_channels() const override { return 1; }
    size_t max_num_samps() const override { return _max_samps; }
    size_t recv(sample_t* buff, size_t nsamps, RxMetadata& metadata,
                double timeout, bool one_packet) override;
    void issue_stream_cmd(const StreamCmd& cmd) override;

private:
    std::shared_ptr<sim_shm::ShmRing> _ring;
    std::shared_ptr<sim_shm::ShmControl> _ctrl;
    size_t _max_samps;
};

// Simulation device. One per hub session; opens the hub control block and
// manufactures streamers attached to the shared rings named "<session>.<suffix>"
// (the suffix is carried in StreamArgs.args["sim_suffix"]).
class SimDevice : public IDevice {
public:
    // Open the hub control block for cfg.sim_session. Throws on timeout.
    explicit SimDevice(const DeviceConfig& cfg);

    bool supports(Capability cap) const override;

    TimeSpec time_now() const override;
    double master_clock_rate() const override { return _tick_rate; }
    double get_tx_rate(size_t /*chan*/ = 0) const override { return _tick_rate; }
    double get_rx_rate(size_t /*chan*/ = 0) const override { return _tick_rate; }

    TuneResult set_tx_freq(const TuneRequest& req, size_t chan = 0) override;
    TuneResult set_rx_freq(const TuneRequest& req, size_t chan = 0) override;
    void set_rx_freq_correction(double hz) override;
    double rx_freq_correction() const override;

    ITxStreamPtr get_tx_stream(const StreamArgs& args) override;
    IRxStreamPtr get_rx_stream(const StreamArgs& args) override;

    bool running() const override;

private:
    std::string _session;
    double _tick_rate = 0.0;
    double _center_freq = 0.0;
    std::shared_ptr<sim_shm::ShmControl> _ctrl;
};

}  // namespace radio

#endif  // SIM_BACKEND_HPP
