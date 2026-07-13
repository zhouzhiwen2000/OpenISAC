#ifndef RADIO_BACKEND_HPP
#define RADIO_BACKEND_HPP

// Radio hardware abstraction layer (HAL).
//
// IDevice / ITxStream / IRxStream are the backend-independent interfaces the
// engines drive. Two backends implement them today:
//   - UhdBackend  (include/UhdBackend.hpp, src/UhdBackend.cpp) — real USRP via UHD.
//   - SimBackend  (include/SimBackend.hpp, src/SimBackend.cpp) — shared-memory
//                  channel simulator, ZERO dependency on UHD.
// A third backend is a new IDevice/I*Stream implementation plus one branch in
// make_device(). Engines never name a concrete backend or a uhd:: symbol; they
// hold IDevicePtr / I*StreamPtr and query supports(Capability) where behavior
// differs between backends.

#include <memory>
#include <string>

#include "RadioTypes.hpp"

namespace radio {

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

class ITxStream {
public:
    virtual ~ITxStream() = default;

    virtual size_t num_channels() const = 0;
    virtual size_t max_num_samps() const = 0;

    // Single-channel hot path. `buff` may be null when `nsamps == 0` (e.g. an
    // end-of-burst marker).
    virtual size_t send(const sample_t* buff, size_t nsamps,
                        const TxMetadata& metadata, double timeout) = 0;

    // Multi-channel overload; defaults to the single-channel path (channel 0).
    virtual size_t send(const sample_t* const* buffs, size_t nsamps,
                        const TxMetadata& metadata, double timeout) {
        return send(buffs[0], nsamps, metadata, timeout);
    }

    virtual bool recv_async_msg(AsyncMetadata& metadata, double timeout) = 0;
};

class IRxStream {
public:
    virtual ~IRxStream() = default;

    virtual size_t num_channels() const = 0;
    virtual size_t max_num_samps() const = 0;

    // Single-channel hot path. Default timeout mirrors uhd::rx_streamer::recv.
    virtual size_t recv(sample_t* buff, size_t nsamps, RxMetadata& metadata,
                        double timeout = 0.1, bool one_packet = false) = 0;

    // Multi-channel overload; defaults to the single-channel path (channel 0).
    virtual size_t recv(sample_t* const* buffs, size_t nsamps, RxMetadata& metadata,
                        double timeout, bool one_packet = false) {
        return recv(buffs[0], nsamps, metadata, timeout, one_packet);
    }

    virtual void issue_stream_cmd(const StreamCmd& cmd) = 0;
};

using ITxStreamPtr = std::shared_ptr<ITxStream>;
using IRxStreamPtr = std::shared_ptr<IRxStream>;

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

// Backend feature flags. Engines branch on these instead of testing the concrete
// backend, so every "is this the simulator?" check becomes a capability query.
enum class Capability {
    TimedTx,        // honors tx_metadata time_spec / timed bursts (real radio)
    FreeRunningClock, // device time advances independently of streamed samples
    AsyncTxEvents,  // produces TX async messages (underflow / seq error / ...)
    StreamRestart,  // RX stream can be stopped + timed-restarted
    HardwareGain,   // set_*_gain / get_*_gain_range act on real hardware
    RfDspTune,      // supports manual RF/DSP split retuning
    PpsTimeSync,    // supports PPS-based multi-device time alignment
};

class IDevice {
public:
    virtual ~IDevice() = default;

    virtual bool supports(Capability cap) const = 0;

    // Reference / timing sources.
    virtual void set_clock_source(const std::string& /*source*/) {}
    virtual void set_time_source(const std::string& /*source*/) {}
    virtual TimeSpec time_now() const = 0;
    virtual void set_time_now(const TimeSpec& /*t*/) {}
    virtual void set_time_next_pps(const TimeSpec& /*t*/) {}
    virtual TimeSpec time_last_pps() const { return TimeSpec(0.0); }
    virtual double master_clock_rate() const = 0;

    // Rates / bandwidth.
    virtual void set_tx_rate(double /*rate*/) {}
    virtual void set_rx_rate(double /*rate*/) {}
    virtual double get_tx_rate(size_t chan = 0) const = 0;
    virtual double get_rx_rate(size_t chan = 0) const { return get_tx_rate(chan); }
    virtual void set_tx_bandwidth(double /*bw*/, size_t /*chan*/ = 0) {}
    virtual void set_rx_bandwidth(double /*bw*/, size_t /*chan*/ = 0) {}
    virtual size_t get_tx_num_channels() const { return 1; }
    virtual size_t get_rx_num_channels() const { return 1; }

    // Gain. No-ops on backends without HardwareGain.
    virtual void set_tx_gain(double /*gain*/, size_t /*chan*/ = 0) {}
    virtual void set_rx_gain(double /*gain*/, size_t /*chan*/ = 0) {}
    virtual GainRange get_tx_gain_range(size_t /*chan*/ = 0) const { return GainRange{}; }
    virtual GainRange get_rx_gain_range(size_t /*chan*/ = 0) const { return GainRange{}; }
    virtual void set_rx_antenna(const std::string& /*ant*/, size_t /*chan*/ = 0) {}

    // Tuning.
    virtual TuneResult set_tx_freq(const TuneRequest& req, size_t chan = 0) = 0;
    virtual TuneResult set_rx_freq(const TuneRequest& req, size_t chan = 0) = 0;
    // Receiver-side comm frequency correction used by the simulator's DSP path.
    // No-op (and 0) on real hardware, which retunes via set_rx_freq instead.
    virtual void set_rx_freq_correction(double /*hz*/) {}
    virtual double rx_freq_correction() const { return 0.0; }

    // Streamers.
    virtual ITxStreamPtr get_tx_stream(const StreamArgs& args) = 0;
    virtual IRxStreamPtr get_rx_stream(const StreamArgs& args) = 0;

    // Liveness — real hardware is always "running"; the simulator reflects the
    // hub's running flag so engines can detect a paused/stopped hub.
    virtual bool running() const { return true; }
};

using IDevicePtr = std::shared_ptr<IDevice>;

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

// Everything a backend needs to materialize one device. Callers (BS/UE/sensing)
// populate the fields relevant to the selected backend; the rest are ignored.
struct DeviceConfig {
    std::string backend = "uhd";  // "uhd" | "sim"

    // --- UHD ---
    std::string device_args;      // multi_usrp::make() args (also the cache key)
    std::string clock_source;     // applied if non-empty
    std::string time_source;      // applied if non-empty

    // --- SIM ---
    std::string sim_session;      // ChannelSimulator session name
    double sim_tick_rate = 0.0;   // sample rate (Hz) reported as the radio clock
    double sim_center_freq = 0.0; // center freq used to synthesize tune results
    bool sim_predictive_delay = false; // validate simulator CFO/SRO consistency for UE prediction
};

// Create (or return a shared, cached) device for the given config. Devices are
// deduplicated per process by backend + key (device_args for UHD, session for
// SIM) so a TX device shared with uplink-RX / sensing-RX is the same object
// (pointer identity preserved). Throws on backend errors / clock-source conflict.
IDevicePtr make_device(const DeviceConfig& cfg);

// RT-thread priority helper (wraps uhd::set_thread_priority_safe). Declared here
// so engines stay UHD-free; defined in src/UhdBackend.cpp, linked into every target.
// Defaults mirror uhd::set_thread_priority_safe (priority 0.5, realtime on) so a
// bare set_thread_priority() preserves the previous RT-scheduling behavior.
void set_thread_priority(float priority = 0.5f, bool realtime = true);

}  // namespace radio

#endif  // RADIO_BACKEND_HPP
