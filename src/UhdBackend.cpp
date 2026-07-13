#include "UhdBackend.hpp"
#include "SimBackend.hpp"

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/stream.hpp>
#include <uhd/types/metadata.hpp>
#include <uhd/types/stream_cmd.hpp>
#include <uhd/types/time_spec.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/types/tune_result.hpp>
#include <uhd/utils/thread.hpp>

namespace radio {

namespace {

// ---- radio:: <-> uhd:: value-type translation -----------------------------

inline uhd::time_spec_t to_uhd(const TimeSpec& t) {
    return uhd::time_spec_t(static_cast<time_t>(t.get_full_secs()), t.get_frac_secs());
}
inline TimeSpec from_uhd(const uhd::time_spec_t& t) {
    return TimeSpec(static_cast<int64_t>(t.get_full_secs()), t.get_frac_secs());
}

inline uhd::tx_metadata_t to_uhd(const TxMetadata& m) {
    uhd::tx_metadata_t um;
    um.start_of_burst = m.start_of_burst;
    um.end_of_burst = m.end_of_burst;
    um.has_time_spec = m.has_time_spec;
    um.time_spec = to_uhd(m.time_spec);
    return um;
}

inline RxError rx_error_from_uhd(uhd::rx_metadata_t::error_code_t code) {
    switch (code) {
        case uhd::rx_metadata_t::ERROR_CODE_NONE: return RxError::None;
        case uhd::rx_metadata_t::ERROR_CODE_TIMEOUT: return RxError::Timeout;
        case uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND: return RxError::LateCommand;
        case uhd::rx_metadata_t::ERROR_CODE_BROKEN_CHAIN: return RxError::BrokenChain;
        case uhd::rx_metadata_t::ERROR_CODE_OVERFLOW: return RxError::Overflow;
        case uhd::rx_metadata_t::ERROR_CODE_ALIGNMENT: return RxError::Alignment;
        case uhd::rx_metadata_t::ERROR_CODE_BAD_PACKET: return RxError::BadPacket;
        default: return RxError::BadPacket;
    }
}

inline void fill_rx_metadata(const uhd::rx_metadata_t& um, RxMetadata& m) {
    m.has_time_spec = um.has_time_spec;
    m.time_spec = from_uhd(um.time_spec);
    m.start_of_burst = um.start_of_burst;
    m.end_of_burst = um.end_of_burst;
    m.more_fragments = um.more_fragments;
    m.fragment_offset = um.fragment_offset;
    m.out_of_sequence = um.out_of_sequence;
    m.error_code = rx_error_from_uhd(um.error_code);
}

inline AsyncEvent async_event_from_uhd(uhd::async_metadata_t::event_code_t code) {
    switch (code) {
        case uhd::async_metadata_t::EVENT_CODE_BURST_ACK: return AsyncEvent::BurstAck;
        case uhd::async_metadata_t::EVENT_CODE_UNDERFLOW: return AsyncEvent::Underflow;
        case uhd::async_metadata_t::EVENT_CODE_UNDERFLOW_IN_PACKET: return AsyncEvent::UnderflowInPacket;
        case uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR: return AsyncEvent::SeqError;
        case uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR_IN_BURST: return AsyncEvent::SeqErrorInBurst;
        case uhd::async_metadata_t::EVENT_CODE_TIME_ERROR: return AsyncEvent::TimeError;
        case uhd::async_metadata_t::EVENT_CODE_USER_PAYLOAD: return AsyncEvent::UserPayload;
        default: return AsyncEvent::Unknown;
    }
}

inline uhd::stream_cmd_t::stream_mode_t to_uhd(StreamMode mode) {
    switch (mode) {
        case StreamMode::StartContinuous: return uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS;
        case StreamMode::StopContinuous: return uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
        case StreamMode::NumSampsAndDone: return uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE;
        case StreamMode::NumSampsAndMore: return uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_MORE;
    }
    return uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS;
}

inline uhd::tune_request_t::policy_t to_uhd(TunePolicy p) {
    switch (p) {
        case TunePolicy::None: return uhd::tune_request_t::POLICY_NONE;
        case TunePolicy::Auto: return uhd::tune_request_t::POLICY_AUTO;
        case TunePolicy::Manual: return uhd::tune_request_t::POLICY_MANUAL;
    }
    return uhd::tune_request_t::POLICY_AUTO;
}

inline uhd::tune_request_t to_uhd(const TuneRequest& req) {
    uhd::tune_request_t ur(req.target_freq);
    ur.rf_freq = req.rf_freq;
    ur.dsp_freq = req.dsp_freq;
    ur.rf_freq_policy = to_uhd(req.rf_freq_policy);
    ur.dsp_freq_policy = to_uhd(req.dsp_freq_policy);
    return ur;
}

inline TuneResult from_uhd(const uhd::tune_result_t& ur) {
    TuneResult r;
    r.target_rf_freq = ur.target_rf_freq;
    r.actual_rf_freq = ur.actual_rf_freq;
    r.target_dsp_freq = ur.target_dsp_freq;
    r.actual_dsp_freq = ur.actual_dsp_freq;
    return r;
}

inline uhd::stream_args_t to_uhd(const StreamArgs& args) {
    uhd::stream_args_t ua(args.cpu_format, args.wire_format);
    ua.channels = args.channels;
    for (const auto& kv : args.args) {
        if (kv.first == "sim_suffix") continue;  // sim-only hint; not a UHD arg
        ua.args[kv.first] = kv.second;
    }
    return ua;
}

// ---- Stream wrappers ------------------------------------------------------

class UhdTxStream : public ITxStream {
public:
    explicit UhdTxStream(uhd::tx_streamer::sptr s) : _s(std::move(s)) {}

    size_t num_channels() const override { return _s->get_num_channels(); }
    size_t max_num_samps() const override { return _s->get_max_num_samps(); }

    size_t send(const sample_t* buff, size_t nsamps, const TxMetadata& md,
                double timeout) override {
        static const sample_t dummy{0.0f, 0.0f};
        const void* b = buff ? static_cast<const void*>(buff) : static_cast<const void*>(&dummy);
        return _s->send(b, nsamps, to_uhd(md), timeout);
    }

    bool recv_async_msg(AsyncMetadata& md, double timeout) override {
        uhd::async_metadata_t um;
        if (!_s->recv_async_msg(um, timeout)) return false;
        md.event_code = async_event_from_uhd(um.event_code);
        md.channel = um.channel;
        md.has_time_spec = um.has_time_spec;
        md.time_spec = from_uhd(um.time_spec);
        return true;
    }

private:
    uhd::tx_streamer::sptr _s;
};

class UhdRxStream : public IRxStream {
public:
    explicit UhdRxStream(uhd::rx_streamer::sptr s) : _s(std::move(s)) {}

    size_t num_channels() const override { return _s->get_num_channels(); }
    size_t max_num_samps() const override { return _s->get_max_num_samps(); }

    size_t recv(sample_t* buff, size_t nsamps, RxMetadata& md, double timeout,
                bool one_packet) override {
        uhd::rx_metadata_t um;
        void* b = buff;
        const size_t got = _s->recv(b, nsamps, um, timeout, one_packet);
        fill_rx_metadata(um, md);
        return got;
    }

    void issue_stream_cmd(const StreamCmd& cmd) override {
        uhd::stream_cmd_t uc(to_uhd(cmd.mode));
        uc.stream_now = cmd.stream_now;
        uc.time_spec = to_uhd(cmd.time_spec);
        uc.num_samps = cmd.num_samps;
        _s->issue_stream_cmd(uc);
    }

private:
    uhd::rx_streamer::sptr _s;
};

// ---- Device wrapper -------------------------------------------------------

class UhdDevice : public IDevice {
public:
    explicit UhdDevice(const DeviceConfig& cfg) {
        _usrp = uhd::usrp::multi_usrp::make(cfg.device_args);
        if (!cfg.clock_source.empty()) _usrp->set_clock_source(cfg.clock_source);
        if (!cfg.time_source.empty()) _usrp->set_time_source(cfg.time_source);
    }

    bool supports(Capability cap) const override {
        (void)cap;
        return true;  // a real USRP supports every modeled capability
    }

    void set_clock_source(const std::string& s) override { _usrp->set_clock_source(s); }
    void set_time_source(const std::string& s) override { _usrp->set_time_source(s); }
    TimeSpec time_now() const override { return from_uhd(_usrp->get_time_now()); }
    void set_time_now(const TimeSpec& t) override { _usrp->set_time_now(to_uhd(t)); }
    void set_time_next_pps(const TimeSpec& t) override { _usrp->set_time_next_pps(to_uhd(t)); }
    TimeSpec time_last_pps() const override { return from_uhd(_usrp->get_time_last_pps()); }
    double master_clock_rate() const override { return _usrp->get_master_clock_rate(); }

    void set_tx_rate(double rate) override { _usrp->set_tx_rate(rate); }
    void set_rx_rate(double rate) override { _usrp->set_rx_rate(rate); }
    double get_tx_rate(size_t chan) const override { return _usrp->get_tx_rate(chan); }
    double get_rx_rate(size_t chan) const override { return _usrp->get_rx_rate(chan); }
    void set_tx_bandwidth(double bw, size_t chan) override { _usrp->set_tx_bandwidth(bw, chan); }
    void set_rx_bandwidth(double bw, size_t chan) override { _usrp->set_rx_bandwidth(bw, chan); }
    size_t get_tx_num_channels() const override { return _usrp->get_tx_num_channels(); }
    size_t get_rx_num_channels() const override { return _usrp->get_rx_num_channels(); }

    void set_tx_gain(double gain, size_t chan) override { _usrp->set_tx_gain(gain, chan); }
    void set_rx_gain(double gain, size_t chan) override { _usrp->set_rx_gain(gain, chan); }
    GainRange get_tx_gain_range(size_t chan) const override {
        const auto r = _usrp->get_tx_gain_range(chan);
        return GainRange{r.start(), r.stop(), r.step()};
    }
    GainRange get_rx_gain_range(size_t chan) const override {
        const auto r = _usrp->get_rx_gain_range(chan);
        return GainRange{r.start(), r.stop(), r.step()};
    }
    void set_rx_antenna(const std::string& ant, size_t chan) override {
        _usrp->set_rx_antenna(ant, chan);
    }

    TuneResult set_tx_freq(const TuneRequest& req, size_t chan) override {
        return from_uhd(_usrp->set_tx_freq(to_uhd(req), chan));
    }
    TuneResult set_rx_freq(const TuneRequest& req, size_t chan) override {
        return from_uhd(_usrp->set_rx_freq(to_uhd(req), chan));
    }

    ITxStreamPtr get_tx_stream(const StreamArgs& args) override {
        return std::make_shared<UhdTxStream>(_usrp->get_tx_stream(to_uhd(args)));
    }
    IRxStreamPtr get_rx_stream(const StreamArgs& args) override {
        return std::make_shared<UhdRxStream>(_usrp->get_rx_stream(to_uhd(args)));
    }

private:
    uhd::usrp::multi_usrp::sptr _usrp;
};

// ---- Per-process device registry ------------------------------------------
//
// Deduplicates devices by (backend, key) so a USRP shared across roles maps to
// one IDevice (pointer identity preserved for "shared device" checks). For UHD,
// a clock/time-source conflict on a shared device is a hard error, matching the
// previous sensing-channel behavior.

struct RegistryEntry {
    IDevicePtr device;
    std::string clock_source;
    std::string time_source;
};

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}
std::map<std::string, RegistryEntry>& registry() {
    static std::map<std::string, RegistryEntry> r;
    return r;
}

}  // namespace

IDevicePtr make_device(const DeviceConfig& cfg) {
    const bool is_sim = (cfg.backend == "sim");
    const std::string key = cfg.backend + "|" + (is_sim ? cfg.sim_session : cfg.device_args);

    std::lock_guard<std::mutex> lock(registry_mutex());
    auto& reg = registry();
    auto it = reg.find(key);
    if (it != reg.end()) {
        if (!is_sim) {
            const auto& e = it->second;
            if ((!cfg.clock_source.empty() && cfg.clock_source != e.clock_source) ||
                (!cfg.time_source.empty() && cfg.time_source != e.time_source)) {
                throw std::runtime_error(
                    "Clock/time source conflict on shared device (" + cfg.device_args +
                    "): existing clock='" + e.clock_source + "', requested clock='" +
                    cfg.clock_source + "', existing time='" + e.time_source +
                    "', requested time='" + cfg.time_source +
                    "'. Use distinct device_args if different REF/PPS sources are required.");
            }
        }
        return it->second.device;
    }

    IDevicePtr dev = is_sim ? make_sim_device(cfg)
                            : std::static_pointer_cast<IDevice>(std::make_shared<UhdDevice>(cfg));
    reg.emplace(key, RegistryEntry{dev, cfg.clock_source, cfg.time_source});
    return dev;
}

void set_thread_priority(float priority, bool realtime) {
    // Matches the previous direct uhd::set_thread_priority_safe() calls. UHD
    // catches and swallows the failure-to-set exception internally.
    uhd::set_thread_priority_safe(priority, realtime);
}

}  // namespace radio
