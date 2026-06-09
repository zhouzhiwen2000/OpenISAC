#ifndef SIM_STREAMER_HPP
#define SIM_STREAMER_HPP

// Simulated UHD streamers backed by the shared-memory channel simulator.
//
// SimTxStreamer / SimRxStreamer implement the abstract uhd::tx_streamer /
// uhd::rx_streamer interfaces, so the Modulator/Demodulator engines keep their
// existing uhd::*_streamer::sptr members and their send()/recv() hot loops are
// unchanged. Only the device-creation paths construct these instead of calling
// uhd::usrp::multi_usrp::make.
//
// SimRadio is the client-side handle each engine holds: it opens the hub's
// control block and manufactures streamers attached to the shared rings.

#include <memory>
#include <string>

#include <uhd/stream.hpp>
#include <uhd/types/time_spec.hpp>
#ifdef UHD_HAS_STREAMER_ACTION_HOOKS
#include <uhd/rfnoc/actions.hpp>
#endif

#include "Common.hpp"
#include "ShmRing.hpp"

class SimTxStreamer : public uhd::tx_streamer {
public:
    SimTxStreamer(std::shared_ptr<sim_shm::ShmRing> ring,
                  std::shared_ptr<sim_shm::ShmControl> ctrl,
                  size_t max_samps);

    size_t get_num_channels() const override { return 1; }
    size_t get_max_num_samps() const override { return _max_samps; }
    size_t send(const buffs_type& buffs,
                size_t nsamps_per_buff,
                const uhd::tx_metadata_t& metadata,
                double timeout) override;
    bool recv_async_msg(uhd::async_metadata_t& async_metadata, double timeout) override;
#ifdef UHD_HAS_STREAMER_ACTION_HOOKS
    void post_output_action(const std::shared_ptr<uhd::rfnoc::action_info>& action,
                            size_t port) override;
#endif

private:
    std::shared_ptr<sim_shm::ShmRing> _ring;
    std::shared_ptr<sim_shm::ShmControl> _ctrl;
    size_t _max_samps;
};

class SimRxStreamer : public uhd::rx_streamer {
public:
    SimRxStreamer(std::shared_ptr<sim_shm::ShmRing> ring,
                  std::shared_ptr<sim_shm::ShmControl> ctrl,
                  size_t max_samps);

    size_t get_num_channels() const override { return 1; }
    size_t get_max_num_samps() const override { return _max_samps; }
    size_t recv(const buffs_type& buffs,
                size_t nsamps_per_buff,
                uhd::rx_metadata_t& metadata,
                double timeout,
                bool one_packet) override;
    void issue_stream_cmd(const uhd::stream_cmd_t& stream_cmd) override;
#ifdef UHD_HAS_STREAMER_ACTION_HOOKS
    void post_input_action(const std::shared_ptr<uhd::rfnoc::action_info>& action,
                           size_t port) override;
#endif

private:
    std::shared_ptr<sim_shm::ShmRing> _ring;
    std::shared_ptr<sim_shm::ShmControl> _ctrl;
    size_t _max_samps;
};

// Client-side handle to the channel simulator. One per engine process.
class SimRadio {
public:
    // Open the hub's control block for this session. Blocks until the hub is up.
    // Returns false on timeout.
    bool connect(const SimConfig& sim_cfg);

    // Manufacture a TX streamer (producer into "<session>.tx").
    uhd::tx_streamer::sptr make_tx_streamer(size_t max_samps);

    // Manufacture an RX streamer (consumer of "<session>.<suffix>").
    uhd::rx_streamer::sptr make_rx_streamer(const std::string& suffix, size_t max_samps);

    // Current simulated time from the hub's sample clock.
    uhd::time_spec_t time_now() const;

    bool running() const;

private:
    SimConfig _cfg;
    std::shared_ptr<sim_shm::ShmControl> _ctrl;
};

#endif // SIM_STREAMER_HPP
