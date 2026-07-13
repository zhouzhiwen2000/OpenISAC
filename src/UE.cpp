#include <uhd/utils/thread.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <complex>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <fftw3.h>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <filesystem>
#include <Common.hpp>
#include "LDPCCodec.hpp"
#include "UplinkTxEngine.hpp"
#include "OFDMCore.hpp"
#include "SensingChannel.hpp"
#include "SimStreamer.hpp"

namespace {
inline int64_t host_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

const char* tx_async_event_code_to_string(uhd::async_metadata_t::event_code_t event_code)
{
    switch (event_code) {
    case uhd::async_metadata_t::EVENT_CODE_BURST_ACK:
        return "BURST_ACK";
    case uhd::async_metadata_t::EVENT_CODE_UNDERFLOW:
        return "UNDERFLOW";
    case uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR:
        return "SEQ_ERROR";
    case uhd::async_metadata_t::EVENT_CODE_TIME_ERROR:
        return "TIME_ERROR";
    case uhd::async_metadata_t::EVENT_CODE_UNDERFLOW_IN_PACKET:
        return "UNDERFLOW_IN_PACKET";
    case uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR_IN_BURST:
        return "SEQ_ERROR_IN_BURST";
    case uhd::async_metadata_t::EVENT_CODE_USER_PAYLOAD:
        return "USER_PAYLOAD";
    default:
        return "UNKNOWN";
    }
}

const char* rx_error_code_to_string(uhd::rx_metadata_t::error_code_t error_code)
{
    switch (error_code) {
    case uhd::rx_metadata_t::ERROR_CODE_NONE:
        return "NONE";
    case uhd::rx_metadata_t::ERROR_CODE_TIMEOUT:
        return "TIMEOUT";
    case uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND:
        return "LATE_COMMAND";
    case uhd::rx_metadata_t::ERROR_CODE_BROKEN_CHAIN:
        return "BROKEN_CHAIN";
    case uhd::rx_metadata_t::ERROR_CODE_OVERFLOW:
        return "OVERFLOW";
    case uhd::rx_metadata_t::ERROR_CODE_ALIGNMENT:
        return "ALIGNMENT";
    case uhd::rx_metadata_t::ERROR_CODE_BAD_PACKET:
        return "BAD_PACKET";
    default:
        return "UNKNOWN";
    }
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n") == std::string::npos) {
        return value;
    }
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

bool append_csv_row(
    const std::string& path,
    const std::vector<std::string>& header,
    const std::vector<std::string>& row)
{
    if (path.empty() || header.size() != row.size()) {
        return false;
    }
    try {
        const std::filesystem::path csv_path(path);
        if (csv_path.has_parent_path()) {
            std::filesystem::create_directories(csv_path.parent_path());
        }
        const bool write_header = !std::filesystem::exists(csv_path);
        std::ofstream out(path, std::ios::app);
        if (!out) {
            return false;
        }
        auto write_line = [&out](const std::vector<std::string>& cols) {
            for (size_t i = 0; i < cols.size(); ++i) {
                if (i > 0) {
                    out << ',';
                }
                out << csv_escape(cols[i]);
            }
            out << '\n';
        };
        if (write_header) {
            write_line(header);
        }
        write_line(row);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

double power_to_db(double value) {
    return 10.0 * std::log10(std::max(value, 1e-30));
}

double peak_ratio_db(double peak, double avg) {
    return (avg > 0.0) ? power_to_db(peak / avg) : -300.0;
}

int sync_cfo_alias_span_from_range(double range_hz, double alias_period_hz) {
    if (!(range_hz > 0.0) || !(alias_period_hz > 0.0) ||
        !std::isfinite(range_hz) || !std::isfinite(alias_period_hz)) {
        return 0;
    }
    return static_cast<int>(std::ceil(range_hz / alias_period_hz));
}

std::string format_sync_alias_candidates(
    const char* label,
    const SyncProcessor::SecSyncRefineResult& result)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << label << " alias candidates:";

    if (result.alias_candidates.empty()) {
        oss << "\n  []";
        return oss.str();
    }

    for (const auto& candidate : result.alias_candidates) {
        oss << "\n  "
            << (result.valid && candidate.alias_index == result.alias_index ? "*" : " ")
            << "k=" << candidate.alias_index
            << " peak=" << power_to_db(candidate.max_corr) << "dB";
        if (!candidate.valid) {
            oss << " invalid";
        }
    }
    return oss.str();
}

std::vector<AlignedVector> make_midframe_pilot_sequences(
    const Config& cfg,
    const DataResourceGridLayout& layout)
{
    std::vector<AlignedVector> sequences;
    sequences.reserve(layout.midframe_pilot_symbols.size());
    const AlignedVector comb_pilot_seq = generate_zc_freq(cfg.ofdm.fft_size, cfg.ofdm.zc_root);
    for (const int sym : layout.midframe_pilot_symbols) {
        sequences.push_back(generate_midframe_bpsk_pilot_freq(
            cfg.ofdm.fft_size,
            cfg.ofdm.midframe_pilot_seed,
            static_cast<size_t>(sym),
            cfg.ofdm.pilot_positions,
            comb_pilot_seq));
    }
    return sequences;
}

AlignedVector make_cfo_training_sequence(const Config& cfg)
{
    if (!cfo_training_sequence_enabled(cfg)) {
        return {};
    }
    return generate_cfo_training_freq(
        cfg.ofdm.fft_size,
        cfg.ofdm.cfo_training_period_samples);
}
} // namespace

/**
 * @brief OFDM Receiver Engine.
 * 
 * Main class for OFDM Demodulation and ISAC Sensing Receiver.
 * Integrates UHD for USRP control, FFTW for signal processing, and aff3ct for LDPC decoding.
 * Implements a multi-threaded architecture:
 * - rx_thread: Receives raw samples from USRP.
 * - process_thread: Performs OFDM FFT, Channel Estimation, Equalization.
 * - sensing_thread: Accumulated channel response for further analysis.
 * - bit_processing_thread: Soft Demodulation, LDPC Decoding, and UDP output.
 */
class UEEngine {
public:
    explicit UEEngine(const Config& cfg)
        : cfg_(cfg),
          _measurement_enabled(measurement_mode_enabled(cfg)),
          _duplex_layout(build_duplex_frame_layout(cfg)),
          _data_resource_layout(build_data_resource_grid_layout(cfg)),
          zc_freq_(generate_zc_freq(cfg.ofdm.fft_size, cfg.ofdm.zc_root)),
          _cfo_training_seq(make_cfo_training_sequence(cfg)),
          _midframe_pilot_seqs(make_midframe_pilot_sequences(cfg, _data_resource_layout)),
          _sensing_pilot_zc_root(select_known_sensing_pilot_zc_root(cfg.ofdm.fft_size, cfg.ofdm.zc_root)),
          sensing_pilot_freq_(generate_sensing_pilot_freq(cfg.ofdm.fft_size, cfg.ofdm.zc_root)),
          _sync_scratch_buffer(cfg.sync_samples()),
          frame_queue_(cfg.ofdm.frame_queue_size),
          sync_queue_(cfg.sync_tracking.sync_queue_size, [&cfg]() {
              SyncBatch batch;
              batch.data.resize(cfg.sync_samples());
              batch.usrp_time_ns = -1;
              return batch;
          }),
          _channel_estimator(cfg.ofdm.fft_size),
          _delay_processor(cfg.ofdm.fft_size),
          _sync_processor(cfg.sync_samples(), cfg.ofdm.fft_size, cfg.ofdm.cp_length, zc_freq_),
          _uplink_self_zc_freq(generate_zc_freq(cfg.ofdm.fft_size, make_uplink_config(cfg).ofdm.zc_root)),
          _uplink_self_debug_estimator(cfg.ofdm.fft_size, cfg.ofdm.cp_length),
          _control_handler(cfg.network_output.bi_sensing_ip, cfg.network_output.control_port),
          channel_sender_(2, [this](const auto& data) {
              channel_pub_->send_container(data);
          }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly),
          pdf_sender_(2, [this](const auto& data) {
              pdf_pub_->send_container(data);
          }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly),
          constellation_sender_(10, [this](const auto& data) {
              constellation_pub_->send_container(data);
          }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly),
          uplink_self_channel_sender_(2, [this](const auto& data) {
              uplink_self_channel_pub_->send_container(data);
          }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly),
          uplink_self_pdf_sender_(2, [this](const auto& data) {
              uplink_self_pdf_pub_->send_container(data);
          }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly),
          // Initialize object pools for memory reuse
          _rx_frame_pool(32, [&cfg]() {
              RxFrame frame;
              frame.frame_data.resize(cfg.samples_per_frame());
              frame.Alignment = 0;
              return frame;
          }),
          _llr_pool(32, [&cfg]() {
              const DataResourceGridLayout layout = build_data_resource_grid_layout(cfg);
              return AlignedFloatVector(layout.payload_re_count * 2);
          }),
          _llr_pool_i16(32, [&cfg]() {
              const DataResourceGridLayout layout = build_data_resource_grid_layout(cfg);
              return LDPCCodec::AlignedShortVector(layout.payload_re_count * 2);
          }),
          _sensing_frame_pool(32, [&cfg]() {
              SensingFrame frame;
              frame.rx_symbols.resize(cfg.ofdm.num_symbols);
              frame.tx_symbols.resize(cfg.ofdm.num_symbols);
              for (size_t i = 0; i < cfg.ofdm.num_symbols; ++i) {
                  frame.rx_symbols[i].resize(cfg.ofdm.fft_size);
                  frame.tx_symbols[i].resize(cfg.ofdm.fft_size);
              }
              frame.CFO = 0.0f;
              frame.SFO = 0.0f;
              frame.delay_offset = 0.0f;
              return frame;
          }),
          _rx_agc(cfg),
          _sync_search_gain_sweep(cfg),
          _reset_hold_frames(reset_hold_frames_from_cfg(cfg)),
          _akf(make_akf_params(cfg), frame_duration_from_cfg(cfg)) {
        _bit_interleaver = std::make_unique<BitBlockInterleaver>(_ldpc_decoder.get_N(), 21);
        _measurement_active_epoch_id.store(0, std::memory_order_relaxed);
        if (_measurement_enabled && !cfg_.measurement.measurement_output_dir.empty()) {
            _measurement_summary_path = cfg_.measurement.measurement_output_dir + "/ue_measurement_summary.csv";
        }
        _build_cfo_symbol_skip_mask();
        _build_compact_payload_indices();
        LOG_G_INFO() << "Payload resource grid: " << _data_resource_layout.payload_re_count
                     << " payload RE out of " << _data_resource_layout.non_pilot_re_count
                     << " non-sync/non-pilot RE per frame"
                     << (cfg_.resource_preview.data_resource_blocks_configured ? " (configured blocks)." : " (legacy full-grid mode).");
        if (_data_resource_layout.sensing_pilot_re_count > 0) {
            LOG_G_INFO() << "Sensing-pilot sequence uses alternate ZC root "
                         << _sensing_pilot_zc_root
                         << " (sync root=" << cfg_.ofdm.zc_root << ").";
        }

        init_usrp();
        init_filter();
        prepare_fftw();
        init_zmq_publishers();
        if (cfg_.sync_tracking.hardware_sync) {
            _hw_sync = std::make_unique<HardwareSyncController>(cfg_.sync_tracking.hardware_sync_tty);
            _hw_sync->configure_ocxo_pi(
                cfg_.sync_tracking.ocxo_pi_kp_fast,
                cfg_.sync_tracking.ocxo_pi_ki_fast,
                cfg_.sync_tracking.ocxo_pi_kp_slow,
                cfg_.sync_tracking.ocxo_pi_ki_slow,
                cfg_.sync_tracking.ocxo_pi_switch_abs_error_ppm,
                cfg_.sync_tracking.ocxo_pi_switch_hold_s,
                cfg_.sync_tracking.ocxo_pi_max_step_fast_ppm,
                cfg_.sync_tracking.ocxo_pi_max_step_slow_ppm
            );
        }
        // Initialize sensing processing
        init_sensing();
        // Initialize data processing
        init_data_processing();
        _register_commands();
    }

    ~UEEngine() {
        stop();
        
        
        fftwf_destroy_plan(fft_plan_);
        // Note: _channel_estimator, _delay_processor
        // manage their own FFT plans and clean them up in their destructors
    }

    void start() {
        _control_handler.start();
        _ue_timing_advance.store(cfg_.uplink.ue_timing_advance, std::memory_order_relaxed);
        log_duplex_summary(cfg_, "UE");

        uhd::time_spec_t stream_start_time(0.0);
        if (_uplink_tx && !_sim_radio && usrp_) {
            stream_start_time = _next_timed_stream_start();
            _uplink_tx->set_timed_tx(
                stream_start_time,
                usrp_->get_master_clock_rate(),
                usrp_->get_tx_rate(cfg_.uplink.tx_channel));
            LOG_G_INFO() << "[UE] timed RX/UL-TX stream start at "
                         << stream_start_time.get_real_secs()
                         << " s on the shared radio clock";
        }

        running_.store(true);
        rx_thread_ = std::thread(&UEEngine::rx_proc, this, stream_start_time);
        if (_uplink_tx) {
            _uplink_tx->start();
            if (!_sim_radio) {
                _tx_async_exit_requested.store(false, std::memory_order_relaxed);
                _tx_async_thread = std::thread(&UEEngine::_tx_async_event_proc, this);
            }
        }
        process_thread_ = std::thread(&UEEngine::process_proc, this);
        
        // Start all senders
        channel_sender_.start();
        pdf_sender_.start();
        constellation_sender_.start();
        if (uplink_self_channel_debug_enabled(cfg_) && _uplink_tx) {
            uplink_self_channel_sender_.start();
            uplink_self_pdf_sender_.start();
        }
        
        // Start data processing thread
        _bit_processing_running.store(true);
        _bit_processing_thread = std::thread(&UEEngine::bit_processing_proc, this);
    }

    void stop() {
        running_.store(false);
        _tx_async_exit_requested.store(true, std::memory_order_relaxed);
        if (_tx_async_thread.joinable()) _tx_async_thread.join();
        if (_uplink_tx) {
            _uplink_tx->stop();
        }
        if (rx_thread_.joinable()) rx_thread_.join();
        if (process_thread_.joinable()) process_thread_.join();
        sensing_running_.store(false);
        if (sensing_thread_.joinable()) sensing_thread_.join();
        if (_bistatic_sensing_channel) {
            _bistatic_sensing_channel->stop_bistatic();
        }
        
        // Stop data processing thread
        _bit_processing_running.store(false);
        if (_bit_processing_thread.joinable()) _bit_processing_thread.join();
        if (_measurement_enabled) {
            _switch_measurement_epoch(0);
        }
        
        // Stop all senders
        channel_sender_.stop();
        pdf_sender_.stop();
        constellation_sender_.stop();
        uplink_self_channel_sender_.stop();
        uplink_self_pdf_sender_.stop();
        _control_handler.stop();
    }

private:
private:
    static LDPCCodec::LDPCConfig ldpc_cfg_from_config(const Config& cfg) {
        LDPCCodec::LDPCConfig c = make_ldpc_5041008_cfg();
        c.fixed_point = cfg.ldpc.fixed_point;
        return c;
    }

    static inline int16_t sat16_llr(float v) {
        long q = std::lroundf(v);
        if (q > 32767) q = 32767;
        else if (q < -32767) q = -32767;
        return static_cast<int16_t>(q);
    }

    static AdaptiveCFOAKF::Params make_akf_params(const Config& cfg) {
        AdaptiveCFOAKF::Params p;
        p.enable = cfg.sync_tracking.akf_enable;
        p.bootstrap_frames = cfg.sync_tracking.akf_bootstrap_frames;
        p.innovation_window = cfg.sync_tracking.akf_innovation_window;
        p.max_lag = cfg.sync_tracking.akf_max_lag;
        p.adapt_interval = cfg.sync_tracking.akf_adapt_interval;
        p.gate_sigma = cfg.sync_tracking.akf_gate_sigma;
        p.tikhonov_lambda = cfg.sync_tracking.akf_tikhonov_lambda;
        p.update_smooth = cfg.sync_tracking.akf_update_smooth;
        p.q_wf_min = cfg.sync_tracking.akf_q_wf_min;
        p.q_wf_max = cfg.sync_tracking.akf_q_wf_max;
        p.q_rw_min = cfg.sync_tracking.akf_q_rw_min;
        p.q_rw_max = cfg.sync_tracking.akf_q_rw_max;
        p.r_min = cfg.sync_tracking.akf_r_min;
        p.r_max = cfg.sync_tracking.akf_r_max;
        return p;
    }

    enum class RxState { SYNC_SEARCH, ALIGNMENT, NORMAL };

    Config cfg_;
    const bool _measurement_enabled;
    const DuplexFrameLayout _duplex_layout;
    const DataResourceGridLayout _data_resource_layout;
    std::vector<uint8_t> _cfo_symbol_skip_mask;
    std::vector<int> _payload_subcarrier_indices_flat;
    uhd::usrp::multi_usrp::sptr usrp_;
    uhd::rx_streamer::sptr rx_stream_;
    std::unique_ptr<SimRadio> _sim_radio;  // non-null when radio_backend == "sim"
    std::unique_ptr<UplinkTxEngine> _uplink_tx;  // non-null when duplex uplink enabled
    std::thread _tx_async_thread;
    std::atomic<bool> _tx_async_exit_requested{false};
    std::atomic<bool> _stream_restart_requested{false};
    std::atomic<uint64_t> _stream_restart_count{0};
    std::atomic<uint64_t> _rx_overflow_count{0};
    std::atomic<uint64_t> _tx_async_error_count{0};
    std::atomic<uint64_t> _handled_ul_tx_error_count{0};
    std::mutex _uplink_tx_gain_mutex;
    bool _uplink_tx_gain_range_initialized = false;
    double _uplink_tx_gain_min_db = 0.0;
    double _uplink_tx_gain_max_db = 0.0;
    double _uplink_tx_gain_restore_db = 0.0;
    std::atomic<bool> _uplink_tx_gain_muted{false};

    // Current radio time: real USRP clock, or the simulator's shared sample clock.
    uhd::time_spec_t radio_time_now() const {
        return _sim_radio ? _sim_radio->time_now() : usrp_->get_time_now();
    }

    uhd::time_spec_t _next_timed_stream_start() const {
        if (!usrp_) {
            return uhd::time_spec_t(0.0);
        }
        constexpr double kStartLeadTimeSec = 1.0;
        const double scheduled_start_s =
            std::ceil(usrp_->get_time_now().get_real_secs() + kStartLeadTimeSec);
        return uhd::time_spec_t(scheduled_start_s);
    }

    void _request_stream_restart(const char* reason) {
        if (_sim_radio || !_uplink_tx || !usrp_) {
            return;
        }
        _stream_restart_requested.store(true, std::memory_order_release);
        _stream_restart_count.fetch_add(1, std::memory_order_relaxed);
        LOG_G_WARN() << "[UE] requested shared RX/UL-TX stream restart: " << reason;
    }

    void _reset_receive_state_after_stream_restart() {
        _sync_generation.fetch_add(1, std::memory_order_acq_rel);
        _sync_in_progress = false;
        _delay_adjustment_count = 0;
        _last_delay_index_err = 0;
        _reset_count = 0;
        sync_offset_.store(0, std::memory_order_relaxed);
        discard_samples_.store(0, std::memory_order_relaxed);
        _reset_uplink_tx_rx_alignment_shift();
        _set_uplink_waveform_enabled(false);
        sfo_estimator.reset();
        state_ = RxState::SYNC_SEARCH;
    }

    void _tx_async_event_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        while (!_tx_async_exit_requested.load(std::memory_order_relaxed)) {
            if (!_uplink_tx || !usrp_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            uhd::async_metadata_t async_md;
            bool got_event = false;
            try {
                got_event = _uplink_tx->tx_stream()->recv_async_msg(async_md, 0.1);
            } catch (const std::exception& e) {
                if (!_tx_async_exit_requested.load(std::memory_order_relaxed)) {
                    LOG_G_WARN() << "[UL-TX Async] recv_async_msg failed: " << e.what();
                }
                continue;
            }
            if (!got_event) {
                continue;
            }

            auto log_event = [&](auto&& log_line) {
                log_line << "[UL-TX Async] " << tx_async_event_code_to_string(async_md.event_code)
                         << " (code=0x" << std::hex << static_cast<int>(async_md.event_code) << std::dec
                         << ", channel=" << async_md.channel;
                if (async_md.has_time_spec) {
                    log_line << ", event_time=" << std::fixed << std::setprecision(6)
                             << async_md.time_spec.get_real_secs() << " s" << std::defaultfloat;
                }
                log_line << ")";
            };

            switch (async_md.event_code) {
            case uhd::async_metadata_t::EVENT_CODE_BURST_ACK:
                log_event(LOG_G_INFO());
                break;
            case uhd::async_metadata_t::EVENT_CODE_UNDERFLOW:
            case uhd::async_metadata_t::EVENT_CODE_UNDERFLOW_IN_PACKET:
                _tx_async_error_count.fetch_add(1, std::memory_order_relaxed);
                log_event(LOG_G_WARN());
                _request_stream_restart("UL-TX underflow");
                break;
            case uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR:
            case uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR_IN_BURST:
            case uhd::async_metadata_t::EVENT_CODE_TIME_ERROR:
                _tx_async_error_count.fetch_add(1, std::memory_order_relaxed);
                log_event(LOG_G_ERROR());
                _request_stream_restart("UL-TX async timing/sequence error");
                break;
            default:
                log_event(LOG_G_INFO());
                break;
            }
        }
    }

    bool _handle_rx_metadata_error(
        const uhd::rx_metadata_t& md,
        const char* context,
        bool* restart_current_read = nullptr)
    {
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_NONE ||
            md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
            return false;
        }

        if (restart_current_read) {
            *restart_current_read = true;
        }

        LOG_RT_WARN() << "[UE RX] " << context << " metadata error "
                      << rx_error_code_to_string(md.error_code) << ": "
                      << md.strerror();

        switch (md.error_code) {
        case uhd::rx_metadata_t::ERROR_CODE_OVERFLOW:
            _rx_overflow_count.fetch_add(1, std::memory_order_relaxed);
            _request_stream_restart("RX overflow");
            break;
        case uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND:
        case uhd::rx_metadata_t::ERROR_CODE_BROKEN_CHAIN:
        case uhd::rx_metadata_t::ERROR_CODE_ALIGNMENT:
        case uhd::rx_metadata_t::ERROR_CODE_BAD_PACKET:
            _request_stream_restart("RX metadata error");
            break;
        default:
            break;
        }
        return true;
    }

    AlignedVector zc_freq_;
    AlignedVector _cfo_training_seq;
    std::vector<AlignedVector> _midframe_pilot_seqs;
    int _sensing_pilot_zc_root = 0;
    AlignedVector sensing_pilot_freq_;
    AlignedVector tx_sync_symbol_; 
    AlignedVector _sync_scratch_buffer;
    AlignedVector _sync_cfo_compensated_buffer;
    std::atomic<int> sync_offset_{0};
    std::atomic<bool> sync_done_{false};
    std::atomic<RxState> state_{RxState::SYNC_SEARCH};
    std::atomic<int> discard_samples_{0};
    std::atomic<uint64_t> _sync_generation{0};
    SPSCRingBuffer<RxFrame> frame_queue_;

    SPSCRingBuffer<SyncBatch> sync_queue_;

    AlignedVector fft_input_;
    AlignedVector fft_output_;
    fftwf_plan fft_plan_;

    // Persistent per-frame symbol storage (reused across frames to avoid
    // per-symbol heap allocation in the hot path). Sized once in prepare_fftw().
    std::vector<AlignedVector> _symbols_buf;     // data symbols (excludes sync)
    AlignedVector _sync_symbol_freq_buf;         // sync symbol, freq domain
    std::vector<AlignedVector> _midframe_symbol_freq_bufs; // full-band pilot symbols

    // Persistent scratch for the per-frame demod path. These were previously
    // re-declared (and thus heap-allocated) inside process_ofdm_frame every
    // frame; hoisting them to members keeps capacity across frames and removes
    // per-frame malloc/free, which is what introduces timing jitter on isolated
    // cores. process_ofdm_frame runs only on the single process thread, so plain
    // members (no synchronization) are safe.
    AlignedVector _h_est_buf;                    // channel estimate H_est
    std::vector<int> _pilot_indices_buf;         // pilot subcarrier indices
    std::vector<float> _avg_phase_diff_buf;      // per-pilot avg phase diff
    std::vector<float> _weights_buf;             // per-pilot regression weights
    std::vector<int> _tracking_pilot_indices_buf; // per-symbol pilot tracking indices
    std::vector<float> _tracking_phase_buf;      // per-symbol pilot residual phase
    std::vector<float> _tracking_weights_buf;    // per-symbol pilot residual weights
    AlignedVector _tracking_h_inv_buf;           // per-symbol tracked inverse
    AlignedVector _midframe_tmp_h_buf;           // LS estimate from one full-band pilot
    AlignedVector _midframe_interp_h_buf;        // time-interpolated channel estimate
    std::vector<AlignedVector> _channel_anchor_h_bufs;
    std::vector<int> _channel_anchor_symbols_buf;
    AlignedVector _delay_spectrum_buf;           // delay (time-domain) spectrum

    // Core computation instances (manage their own FFT plans)
    ChannelEstimator _channel_estimator;
    DelayProcessor _delay_processor;
    SyncProcessor _sync_processor;
    AlignedVector _uplink_self_zc_freq;
    SelfZcChannelDebugEstimator _uplink_self_debug_estimator;
    AlignedVector _uplink_self_h_est;
    AlignedVector _uplink_self_delay_spectrum;
    uint64_t _uplink_self_debug_frame_counter = 0;
    
    std::unique_ptr<FIRFilter> freq_offset_filter_;
    uhd::tune_result_t current_rx_tune_;
    uhd::tune_result_t current_ul_tx_tune_;
    bool tune_initialized_ = false;
    std::atomic<bool> running_{false};
    std::thread rx_thread_, process_thread_;

    using AlignedAlloc = AlignedAllocator<std::complex<float>, 64>;

    // ZeroMQ publishers for frontend debug streams
    std::unique_ptr<ZmqByteSender> channel_pub_;
    std::unique_ptr<ZmqByteSender> pdf_pub_;
    std::unique_ptr<ZmqByteSender> constellation_pub_;
    std::unique_ptr<ZmqByteSender> uplink_self_channel_pub_;
    std::unique_ptr<ZmqByteSender> uplink_self_pdf_pub_;
    std::unique_ptr<VofaPlusDebugSender> vofa_debug_sender_;
    // Control handler
    ControlCommandHandler _control_handler;
    // Data sender management
    DataSender<std::complex<float>, AlignedAlloc> channel_sender_;
    DataSender<std::complex<float>, AlignedAlloc> pdf_sender_;
    DataSender<std::complex<float>, AlignedAlloc> constellation_sender_;
    DataSender<std::complex<float>, AlignedAlloc> uplink_self_channel_sender_;
    DataSender<std::complex<float>, AlignedAlloc> uplink_self_pdf_sender_;
    uint32_t _reset_count = 0;
    uint32_t _constellation_frame_counter = 0;
    
    // Sensing related variables
    SPSCRingBuffer<SensingFrame> sensing_queue_{4};
    std::thread sensing_thread_;
    std::atomic<bool> sensing_running_{false};
    SharedSensingRuntime _shared_sensing_cfg;
    std::mutex _shared_sensing_cfg_mutex;
    std::unique_ptr<SensingChannel> _bistatic_sensing_channel;
    std::atomic<uint64_t> _next_bistatic_frame_start_symbol{0};

    double _freq_offset_sum = 0.0f; // Average frequency offset
    size_t _freq_offset_count = 0; // Sensing symbol count
    double _avg_freq_offset = 0.0;
    std::vector<int> _actual_subcarrier_indices;
    SFOEstimator sfo_estimator{1000};
    bool _sync_in_progress = false; // Flag to indicate if sync is in progress in process_proc
    int _last_delay_index_err = 0; // Last adjusted index
    uint32_t _delay_adjustment_count = 0;
    std::atomic<float> _user_delay_offset = 0.0f;
    std::atomic<int32_t> _ue_timing_advance{0};  // Timing Advance (samples)
    std::atomic<int64_t> _last_tadv_ns{0};       // rate-limit guard for TADV
    std::atomic<int32_t> _uplink_tx_rx_alignment_shift{0};

    void _reset_uplink_tx_rx_alignment_shift() {
        _uplink_tx_rx_alignment_shift.store(0, std::memory_order_relaxed);
        if (_uplink_tx) {
            _uplink_tx->rx_alignment_shift().store(0, std::memory_order_relaxed);
        }
    }

    // Keep the UL-TX frame anchor tied to the currently aligned UE RX/downlink
    // frame anchor. UplinkTxEngine applies this target by shortening or
    // lengthening the continuous TX stream.
    void _apply_uplink_tx_rx_alignment_delta(int32_t rx_alignment_delta_samples) {
        if (rx_alignment_delta_samples == 0) {
            return;
        }

        const int64_t tx_delta64 = std::clamp<int64_t>(
            -static_cast<int64_t>(rx_alignment_delta_samples),
            static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
            static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
        const int32_t tx_alignment_delta_samples = static_cast<int32_t>(tx_delta64);
        int32_t current = _uplink_tx_rx_alignment_shift.load(std::memory_order_relaxed);
        int32_t next = current;
        do {
            const int64_t next64 = std::clamp<int64_t>(
                static_cast<int64_t>(current) + static_cast<int64_t>(tx_alignment_delta_samples),
                static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
                static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
            next = static_cast<int32_t>(next64);
        } while (!_uplink_tx_rx_alignment_shift.compare_exchange_weak(
            current,
            next,
            std::memory_order_relaxed,
            std::memory_order_relaxed));

        if (_uplink_tx) {
            _uplink_tx->rx_alignment_shift().store(next, std::memory_order_relaxed);
        }
        if (cfg_.should_profile("uplink")) {
            LOG_RT_WARN_HZ(2) << "[UL-TX] RX alignment delta=" << rx_alignment_delta_samples
                              << " samples -> TX timing delta=" << tx_alignment_delta_samples
                              << " samples, cumulative RX-alignment target=" << next;
        }
    }

    void _schedule_receive_alignment(int32_t alignment_samples) {
        discard_samples_.store(static_cast<int>(alignment_samples), std::memory_order_relaxed);
        _apply_uplink_tx_rx_alignment_delta(alignment_samples);
        _restore_uplink_tx_gain_after_sync();
        state_ = RxState::ALIGNMENT;
    }

    void _set_uplink_waveform_enabled(bool enabled) {
        if (!_uplink_tx) {
            return;
        }
        if (enabled) {
            _restore_uplink_tx_gain_after_sync();
        }
        if (_uplink_tx->set_waveform_enabled(enabled) && cfg_.should_profile("uplink")) {
            LOG_RT_INFO() << "[UL-TX] "
                          << (enabled ? "enabled" : "muted")
                          << " uplink waveform transmission";
        }
    }

    void _mute_uplink_tx_gain_for_sync_search() {
        if (!_uplink_tx || _sim_radio || !usrp_ || !_uplink_tx_gain_range_initialized) {
            return;
        }
        std::lock_guard<std::mutex> lock(_uplink_tx_gain_mutex);
        try {
            usrp_->set_tx_gain(_uplink_tx_gain_min_db, cfg_.uplink.tx_channel);
            _uplink_tx_gain_muted.store(true, std::memory_order_release);
        } catch (const std::exception& e) {
            LOG_RT_WARN() << "[UL-TX] failed to mute TX gain during sync search: " << e.what();
        }
    }

    void _restore_uplink_tx_gain_after_sync() {
        if (!_uplink_tx || _sim_radio || !usrp_ || !_uplink_tx_gain_range_initialized) {
            return;
        }
        std::lock_guard<std::mutex> lock(_uplink_tx_gain_mutex);
        try {
            usrp_->set_tx_gain(_uplink_tx_gain_restore_db, cfg_.uplink.tx_channel);
            _uplink_tx_gain_muted.store(false, std::memory_order_release);
        } catch (const std::exception& e) {
            LOG_RT_WARN() << "[UL-TX] failed to restore TX gain after sync: " << e.what();
        }
    }
    std::unique_ptr<HardwareSyncController> _hw_sync;

    // Data processing related member variables
    struct LlrFrame {
        AlignedFloatVector llr;
        uint64_t generation = 0;
        int64_t rx_enqueue_time_ns = 0;
        int64_t process_dequeue_time_ns = 0;
        int64_t demod_done_time_ns = 0;
    };
    // int16 (Q16) variant used when cfg_.ldpc.fixed_point is set. The demapper
    // quantizes LLRs directly into this so the whole pipeline runs int16.
    struct LlrFrameI16 {
        LDPCCodec::AlignedShortVector llr;
        uint64_t generation = 0;
        int64_t rx_enqueue_time_ns = 0;
        int64_t process_dequeue_time_ns = 0;
        int64_t demod_done_time_ns = 0;
    };
    struct LatencyAccumulator {
        std::atomic<int64_t> rx_queue_total_ns{0};
        std::atomic<int64_t> demod_total_ns{0};
        std::atomic<int64_t> bit_total_ns{0};
        std::atomic<int64_t> e2e_total_ns{0};
        std::atomic<int> count{0};
    };
    struct LatencySnapshot {
        int64_t rx_queue_total_ns{0};
        int64_t demod_total_ns{0};
        int64_t bit_total_ns{0};
        int64_t e2e_total_ns{0};
        int count{0};
    };
    SPSCRingBuffer<LlrFrame> _data_llr_buffer{128};  // LLR data buffer (float path)
    SPSCRingBuffer<LlrFrameI16> _data_llr_buffer_i16{128};  // LLR data buffer (int16 path)
    std::thread _bit_processing_thread;
    std::atomic<bool> _bit_processing_running{false};
    LatencyAccumulator _latency_accumulator;

    const bool _ldpc_fixed_point{cfg_.ldpc.fixed_point};
    LDPCCodec _ldpc_decoder{ldpc_cfg_from_config(cfg_)};
    std::unique_ptr<BitBlockInterleaver> _bit_interleaver;
    LDPCCodec::AlignedFloatVector _deinterleaver_llr_scratch;
    LDPCCodec::AlignedShortVector _deinterleaver_llr_scratch_i16;
    Scrambler _descrambler{201600, 0x5A};
    std::unique_ptr<UdpSender> _udp_output_sender;

    // ARQ: downlink receive window for duplicate suppression + ordered delivery
    ArqRxWindow _dl_arq_rx;
    bool _arq_enabled{false};

    // Noise/LLR estimation related
    double _noise_var{0.5};              // Complex noise power E[|n|^2] initial value (assume 0.25 per dimension)
    double _llr_scale{2.0};              // LLR scaling factor (updated based on noise variance)
    double _snr_linear{1.0};             // Es/N0 Linear value
    double _snr_db{0.0};                 // Es/N0 dB
    double _llr_snr_linear_filtered{1.0};
    bool _llr_snr_filter_initialized{false};
    
    // Object pools for memory reuse (eliminates per-frame memory allocations)
    ObjectPool<RxFrame> _rx_frame_pool;           // Pool for RX frame buffers
    ObjectPool<AlignedFloatVector> _llr_pool;     // Pool for LLR data (float path)
    ObjectPool<LDPCCodec::AlignedShortVector> _llr_pool_i16;  // Pool for LLR data (int16 path)
    ObjectPool<SensingFrame> _sensing_frame_pool; // Pool for sensing frames

    // Core computation classes (hardware-independent)
    HardwareRxAgc _rx_agc;
    SyncSearchRxGainSweep _sync_search_gain_sweep;
    uint32_t _reset_hold_frames = 1;
    double _rx_gain_min_db = 0.0;
    double _rx_gain_max_db = 0.0;
    AdaptiveCFOAKF _akf;
    size_t _ocxo_update_counter = 0;
    DemodControlTimeGates _control_time_gates;
    std::string _measurement_summary_path;
    std::atomic<uint32_t> _measurement_active_epoch_id{0};
    std::atomic<uint64_t> _measurement_successful_packets{0};
    std::atomic<uint64_t> _measurement_compared_bits{0};
    std::atomic<uint64_t> _measurement_bit_errors{0};
    std::atomic<uint64_t> _measurement_frame_count{0};
    std::atomic<int64_t> _measurement_snr_db_sum_milli{0};
    std::atomic<int64_t> _measurement_evm_rms_sum_micro{0};
    std::atomic<int64_t> _measurement_evm_db_sum_milli{0};
    std::atomic<int64_t> _measurement_evm_first_db_sum_milli{0};
    std::atomic<int64_t> _measurement_evm_last_db_sum_milli{0};
    std::atomic<int64_t> _measurement_evm_max_db_sum_milli{0};
    std::atomic<int64_t> _measurement_evm_slope_db_sum_milli{0};
    std::atomic<int32_t> _measurement_epoch_tx_gain_x10{std::numeric_limits<int32_t>::min()};

    LatencySnapshot _take_latency_snapshot_and_reset() {
        LatencySnapshot snapshot;
        snapshot.count = _latency_accumulator.count.exchange(0, std::memory_order_acq_rel);
        snapshot.rx_queue_total_ns = _latency_accumulator.rx_queue_total_ns.exchange(0, std::memory_order_acq_rel);
        snapshot.demod_total_ns = _latency_accumulator.demod_total_ns.exchange(0, std::memory_order_acq_rel);
        snapshot.bit_total_ns = _latency_accumulator.bit_total_ns.exchange(0, std::memory_order_acq_rel);
        snapshot.e2e_total_ns = _latency_accumulator.e2e_total_ns.exchange(0, std::memory_order_acq_rel);
        return snapshot;
    }

    struct FrameEvmStats {
        double rms = std::numeric_limits<double>::quiet_NaN();
        double db = std::numeric_limits<double>::quiet_NaN();
        double first_db = std::numeric_limits<double>::quiet_NaN();
        double last_db = std::numeric_limits<double>::quiet_NaN();
        double max_db = std::numeric_limits<double>::quiet_NaN();
        double slope_db_per_symbol = std::numeric_limits<double>::quiet_NaN();

        bool valid() const {
            return std::isfinite(rms) && rms >= 0.0 &&
                   std::isfinite(db) &&
                   std::isfinite(first_db) &&
                   std::isfinite(last_db) &&
                   std::isfinite(max_db) &&
                   std::isfinite(slope_db_per_symbol);
        }
    };

    void _record_measurement_frame(const FrameEvmStats& evm) {
        if (!_measurement_enabled) {
            return;
        }
        const uint32_t epoch_id = _measurement_active_epoch_id.load(std::memory_order_relaxed);
        if (epoch_id == 0 || !evm.valid()) {
            return;
        }
        const double snr_db = _snr_db;
        _measurement_frame_count.fetch_add(1, std::memory_order_relaxed);
        _measurement_snr_db_sum_milli.fetch_add(
            static_cast<int64_t>(std::llround(snr_db * 1000.0)), std::memory_order_relaxed);
        _measurement_evm_rms_sum_micro.fetch_add(
            static_cast<int64_t>(std::llround(evm.rms * 1.0e6)), std::memory_order_relaxed);
        _measurement_evm_db_sum_milli.fetch_add(
            static_cast<int64_t>(std::llround(evm.db * 1000.0)), std::memory_order_relaxed);
        _measurement_evm_first_db_sum_milli.fetch_add(
            static_cast<int64_t>(std::llround(evm.first_db * 1000.0)), std::memory_order_relaxed);
        _measurement_evm_last_db_sum_milli.fetch_add(
            static_cast<int64_t>(std::llround(evm.last_db * 1000.0)), std::memory_order_relaxed);
        _measurement_evm_max_db_sum_milli.fetch_add(
            static_cast<int64_t>(std::llround(evm.max_db * 1000.0)), std::memory_order_relaxed);
        _measurement_evm_slope_db_sum_milli.fetch_add(
            static_cast<int64_t>(std::llround(evm.slope_db_per_symbol * 1000.0)),
            std::memory_order_relaxed);
    }

    double _update_llr_snr_filter(double raw_snr_linear) {
        constexpr double kLlrSnrEwmaAlpha = 0.2;
        const double clamped_raw = std::max(raw_snr_linear, 1e-6);
        if (!_llr_snr_filter_initialized) {
            _llr_snr_linear_filtered = clamped_raw;
            _llr_snr_filter_initialized = true;
        } else {
            _llr_snr_linear_filtered =
                (1.0 - kLlrSnrEwmaAlpha) * _llr_snr_linear_filtered +
                kLlrSnrEwmaAlpha * clamped_raw;
        }
        return _llr_snr_linear_filtered;
    }

    void _build_compact_payload_indices() {
        _payload_subcarrier_indices_flat.clear();
        _payload_subcarrier_indices_flat.reserve(_data_resource_layout.payload_re_count);

        for (size_t sym_idx = 0; sym_idx < _data_resource_layout.data_symbol_count; ++sym_idx) {
            const size_t non_pilot_base = _data_resource_layout.non_pilot_offsets[sym_idx];
            for (size_t di = 0; di < _data_resource_layout.num_non_pilot_subcarriers; ++di) {
                if (_data_resource_layout.payload_rank[non_pilot_base + di] < 0) {
                    continue;
                }
                const int k = _data_resource_layout.non_pilot_subcarrier_indices[di];
                _payload_subcarrier_indices_flat.push_back(k);
            }
            if (_payload_subcarrier_indices_flat.size() != _data_resource_layout.payload_offsets[sym_idx + 1]) {
                throw std::runtime_error("Failed to compact payload subcarrier indices for CPU RX.");
            }
        }
    }

    void _build_cfo_symbol_skip_mask() {
        _cfo_symbol_skip_mask.assign(cfg_.ofdm.num_symbols, 0);
        if (_duplex_layout.mode != DuplexMode::TDD ||
            !_duplex_layout.uplink_enabled ||
            cfg_.ofdm.num_symbols == 0) {
            return;
        }

        for (size_t sym = 0; sym < cfg_.ofdm.num_symbols; ++sym) {
            if (_duplex_layout.is_uplink(sym) || _duplex_layout.is_guard(sym)) {
                _cfo_symbol_skip_mask[sym] = 1;
            }
        }
        for (size_t sym = 0; sym < cfg_.ofdm.num_symbols; ++sym) {
            if (_duplex_layout.is_uplink(sym) || _duplex_layout.is_guard(sym)) {
                continue;
            }
            const size_t prev = (sym == 0) ? (cfg_.ofdm.num_symbols - 1) : (sym - 1);
            const size_t next = (sym + 1 == cfg_.ofdm.num_symbols) ? 0 : (sym + 1);
            if (_duplex_layout.is_uplink(prev) || _duplex_layout.is_guard(prev) ||
                _duplex_layout.is_uplink(next) || _duplex_layout.is_guard(next)) {
                _cfo_symbol_skip_mask[sym] = 1;
            }
        }
    }

    FrameEvmStats _compute_frame_evm_stats(const std::vector<AlignedVector>& symbols) const
    {
        FrameEvmStats stats;
        if (symbols.empty() || _payload_subcarrier_indices_flat.empty()) {
            return stats;
        }
        double err_power_acc = 0.0;
        uint64_t err_count = 0;
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_xx = 0.0;
        double sum_xy = 0.0;
        uint64_t per_symbol_count = 0;
        for (size_t sym_idx = 0; sym_idx < symbols.size(); ++sym_idx) {
            const auto& symbol = symbols[sym_idx];
            const auto* __restrict__ sym_ptr = symbol.data();
            const size_t payload_begin = _data_resource_layout.payload_offsets[sym_idx];
            const size_t payload_end = _data_resource_layout.payload_offsets[sym_idx + 1];
            double sym_err_power_acc = 0.0;
            uint64_t sym_err_count = 0;
            // No omp simd: sym_ptr[_payload_subcarrier_indices_flat[idx]] is a gather;
            // scalar matches the vectorized version here.
            for (size_t idx = payload_begin; idx < payload_end; ++idx) {
                const size_t k = static_cast<size_t>(_payload_subcarrier_indices_flat[idx]);
                const float ref_re = std::copysign(QPSKModulator::SQRT_2_INV, sym_ptr[k].real());
                const float ref_im = std::copysign(QPSKModulator::SQRT_2_INV, sym_ptr[k].imag());
                const float err_re = sym_ptr[k].real() - ref_re;
                const float err_im = sym_ptr[k].imag() - ref_im;
                const double err_power = static_cast<double>(err_re * err_re + err_im * err_im);
                sym_err_power_acc += err_power;
                sym_err_count += 1;
            }
            for (size_t i = 0; i < cfg_.ofdm.pilot_positions.size(); ++i) {
                const size_t k = cfg_.ofdm.pilot_positions[i];
                if (k >= symbol.size()) {
                    continue;
                }
                const float err_re = sym_ptr[k].real() - zc_freq_[k].real();
                const float err_im = sym_ptr[k].imag() - zc_freq_[k].imag();
                const double err_power = static_cast<double>(err_re * err_re + err_im * err_im);
                sym_err_power_acc += err_power;
                sym_err_count += 1;
            }
            if (sym_err_count == 0) {
                continue;
            }
            err_power_acc += sym_err_power_acc;
            err_count += sym_err_count;

            const double sym_rms = std::sqrt(sym_err_power_acc / static_cast<double>(sym_err_count));
            const double sym_db = 20.0 * std::log10(std::max(sym_rms, 1e-12));
            if (per_symbol_count == 0) {
                stats.first_db = sym_db;
                stats.max_db = sym_db;
            } else {
                stats.max_db = std::max(stats.max_db, sym_db);
            }
            stats.last_db = sym_db;
            const double x = static_cast<double>(sym_idx);
            sum_x += x;
            sum_y += sym_db;
            sum_xx += x * x;
            sum_xy += x * sym_db;
            ++per_symbol_count;
        }
        if (err_count == 0 || per_symbol_count == 0) {
            return stats;
        }
        stats.rms = std::sqrt(err_power_acc / static_cast<double>(err_count));
        stats.db = 20.0 * std::log10(std::max(stats.rms, 1e-12));
        stats.slope_db_per_symbol = 0.0;
        if (per_symbol_count > 1) {
            const double n = static_cast<double>(per_symbol_count);
            const double denom = n * sum_xx - sum_x * sum_x;
            if (std::abs(denom) > 1e-12) {
                stats.slope_db_per_symbol = (n * sum_xy - sum_x * sum_y) / denom;
            }
        }
        return stats;
    }

    void _compute_channel_inverse(
        const AlignedVector& H_est,
        AlignedVector& H_inv,
        float equalizer_noise_var
    ) const {
        const float mag_floor = static_cast<float>(cfg_.downlink.equalizer.equalizer_mag_floor);
        if (cfg_.downlink.equalizer.equalizer_mode == kEqualizerModeMmse) {
            ChannelEstimator::compute_mmse_inverse(
                H_est,
                H_inv,
                equalizer_noise_var,
                mag_floor);
        } else {
            ChannelEstimator::compute_zf_inverse(H_est, H_inv, mag_floor);
        }
    }

    double _estimate_equalizer_noise_var_from_pilots(
        const std::vector<AlignedVector>& symbols,
        const AlignedVector& H_est,
        float global_alpha,
        float global_beta,
        double fallback_noise_var
    ) {
        if (symbols.empty() || cfg_.ofdm.pilot_positions.empty() ||
            _data_resource_layout.data_symbol_to_actual_symbol.size() < symbols.size()) {
            return fallback_noise_var;
        }

        const float mag_floor = static_cast<float>(cfg_.downlink.equalizer.equalizer_mag_floor);
        double err_acc = 0.0;
        size_t err_count = 0;
        for (size_t sym_idx = 0; sym_idx < symbols.size(); ++sym_idx) {
            const int actual_symbol = _data_resource_layout.data_symbol_to_actual_symbol[sym_idx];
            const AlignedVector& symbol_H_base =
                _interpolated_channel_for_symbol(actual_symbol, H_est);
            const bool using_midframe_channel = (&symbol_H_base != &H_est);
            const int relative_symbol_index = actual_symbol - static_cast<int>(cfg_.ofdm.sync_pos);
            const float phase_alpha = using_midframe_channel ? 0.0f : (global_alpha * relative_symbol_index);
            const float phase_beta = using_midframe_channel ? 0.0f : (global_beta * relative_symbol_index);

            const auto& symbol = symbols[sym_idx];
            for (const size_t k : cfg_.ofdm.pilot_positions) {
                if (k >= symbol.size() || k >= symbol_H_base.size() || k >= zc_freq_.size()) {
                    continue;
                }
                const std::complex<float> h = symbol_H_base[k];
                const float h_mag_sq = std::norm(h);
                if (!std::isfinite(h_mag_sq) || h_mag_sq <= mag_floor) {
                    continue;
                }
                const float phase = phase_alpha + phase_beta *
                    static_cast<float>(_actual_subcarrier_indices[k]);
                const std::complex<float> phase_rot(std::cos(phase), std::sin(phase));
                const std::complex<float> pred = h * zc_freq_[k] * phase_rot;
                const std::complex<float> err = symbol[k] - pred;
                err_acc += static_cast<double>(std::norm(err));
                ++err_count;
            }
        }

        if (err_count <= 8) {
            return fallback_noise_var;
        }
        return std::min(std::max(err_acc / static_cast<double>(err_count), 1e-6), 1e6);
    }

    double _average_channel_power(const AlignedVector& H_est) const {
        const double floor_val = std::max(cfg_.downlink.equalizer.equalizer_mag_floor, 1e-12);
        double acc = 0.0;
        size_t count = 0;
        if (!cfg_.ofdm.pilot_positions.empty()) {
            for (const size_t k : cfg_.ofdm.pilot_positions) {
                if (k >= H_est.size()) {
                    continue;
                }
                const double mag_sq = std::norm(H_est[k]);
                if (std::isfinite(mag_sq) && mag_sq > floor_val) {
                    acc += mag_sq;
                    ++count;
                }
            }
        }
        if (count == 0) {
            for (const auto& h : H_est) {
                const double mag_sq = std::norm(h);
                if (std::isfinite(mag_sq) && mag_sq > floor_val) {
                    acc += mag_sq;
                    ++count;
                }
            }
        }
        return (count > 0) ? std::max(acc / static_cast<double>(count), floor_val) : floor_val;
    }

    void _estimate_midframe_pilot_ls(
        const AlignedVector& rx_symbol,
        const AlignedVector& tx_pilot,
        AlignedVector& H_est_out
    ) {
        const size_t n = std::min(rx_symbol.size(), tx_pilot.size());
        H_est_out.resize(n);
        const auto* __restrict__ rx_ptr = rx_symbol.data();
        const auto* __restrict__ tx_ptr = tx_pilot.data();
        auto* __restrict__ h_ptr = H_est_out.data();

        #pragma omp simd simdlen(16)
        for (size_t j = 0; j < n; ++j) {
            const float rx_re = rx_ptr[j].real();
            const float rx_im = rx_ptr[j].imag();
            const float tx_re = tx_ptr[j].real();
            const float tx_im = tx_ptr[j].imag();
            h_ptr[j] = std::complex<float>(
                rx_re * tx_re + rx_im * tx_im,
                rx_im * tx_re - rx_re * tx_im);
        }
    }

    const AlignedVector& _interpolated_channel_for_symbol(int actual_symbol, const AlignedVector& fallback_H)
    {
        if (_channel_anchor_symbols_buf.size() < 2) {
            return fallback_H;
        }
        const auto upper = std::upper_bound(
            _channel_anchor_symbols_buf.begin(),
            _channel_anchor_symbols_buf.end(),
            actual_symbol);
        if (upper == _channel_anchor_symbols_buf.begin()) {
            return fallback_H;
        }
        if (upper == _channel_anchor_symbols_buf.end()) {
            return _channel_anchor_h_bufs.back();
        }
        const size_t hi = static_cast<size_t>(
            std::distance(_channel_anchor_symbols_buf.begin(), upper));
        const size_t lo = hi - 1;
        const int x0 = _channel_anchor_symbols_buf[lo];
        const int x1 = _channel_anchor_symbols_buf[hi];
        const float denom = static_cast<float>(x1 - x0);
        const float t = (denom > 0.0f)
            ? (static_cast<float>(actual_symbol - x0) / denom)
            : 0.0f;
        const auto& h0 = _channel_anchor_h_bufs[lo];
        const auto& h1 = _channel_anchor_h_bufs[hi];
        _midframe_interp_h_buf.resize(h0.size());
        #pragma omp simd simdlen(16)
        for (size_t j = 0; j < h0.size(); ++j) {
            _midframe_interp_h_buf[j] = h0[j] * (1.0f - t) + h1[j] * t;
        }
        return _midframe_interp_h_buf;
    }

    bool _fit_symbol_pilot_phase(
        const AlignedVector& symbol,
        const AlignedVector& H_base,
        float& beta_out,
        float& alpha_out
    ) {
        _tracking_pilot_indices_buf.clear();
        _tracking_phase_buf.clear();
        _tracking_weights_buf.clear();

        const float denom_floor = static_cast<float>(cfg_.downlink.equalizer.equalizer_mag_floor);
        const float min_weight = static_cast<float>(cfg_.downlink.equalizer.channel_tracking_min_pilot_snr);
        for (const size_t k : cfg_.ofdm.pilot_positions) {
            if (k >= symbol.size() || k >= H_base.size() || k >= zc_freq_.size()) {
                continue;
            }
            const std::complex<float> denom = H_base[k] * zc_freq_[k];
            const float denom_power = std::norm(denom);
            if (!std::isfinite(denom_power) || denom_power <= denom_floor) {
                continue;
            }

            const std::complex<float> residual = symbol[k] * std::conj(denom) / denom_power;
            const float residual_power = std::norm(residual);
            if (!std::isfinite(residual.real()) ||
                !std::isfinite(residual.imag()) ||
                !std::isfinite(residual_power) ||
                residual_power <= min_weight) {
                continue;
            }

            _tracking_pilot_indices_buf.push_back(_actual_subcarrier_indices[k]);
            _tracking_phase_buf.push_back(std::arg(residual));
            _tracking_weights_buf.push_back(std::max(denom_power, min_weight));
        }

        if (_tracking_pilot_indices_buf.size() < 2) {
            return false;
        }

        unwrap(_tracking_phase_buf);
        auto [beta, alpha] = weightedlinearRegression(
            _tracking_pilot_indices_buf,
            _tracking_phase_buf,
            _tracking_weights_buf);
        if (!std::isfinite(alpha) || !std::isfinite(beta)) {
            return false;
        }
        beta_out = beta;
        alpha_out = alpha;
        return true;
    }

    void _finalize_measurement_epoch(uint32_t epoch_id) {
        if (!_measurement_enabled || epoch_id == 0) {
            return;
        }
        const uint64_t successful_packets =
            _measurement_successful_packets.exchange(0, std::memory_order_acq_rel);
        const uint64_t compared_bits =
            _measurement_compared_bits.exchange(0, std::memory_order_acq_rel);
        const uint64_t bit_errors =
            _measurement_bit_errors.exchange(0, std::memory_order_acq_rel);
        const uint64_t frame_count =
            _measurement_frame_count.exchange(0, std::memory_order_acq_rel);
        const int64_t snr_db_sum_milli =
            _measurement_snr_db_sum_milli.exchange(0, std::memory_order_acq_rel);
        const int64_t evm_rms_sum_micro =
            _measurement_evm_rms_sum_micro.exchange(0, std::memory_order_acq_rel);
        const int64_t evm_db_sum_milli =
            _measurement_evm_db_sum_milli.exchange(0, std::memory_order_acq_rel);
        const int64_t evm_first_db_sum_milli =
            _measurement_evm_first_db_sum_milli.exchange(0, std::memory_order_acq_rel);
        const int64_t evm_last_db_sum_milli =
            _measurement_evm_last_db_sum_milli.exchange(0, std::memory_order_acq_rel);
        const int64_t evm_max_db_sum_milli =
            _measurement_evm_max_db_sum_milli.exchange(0, std::memory_order_acq_rel);
        const int64_t evm_slope_db_sum_milli =
            _measurement_evm_slope_db_sum_milli.exchange(0, std::memory_order_acq_rel);
        const int32_t tx_gain_x10 =
            _measurement_epoch_tx_gain_x10.exchange(std::numeric_limits<int32_t>::min(),
                                                    std::memory_order_acq_rel);

        const uint64_t packets_expected = cfg_.measurement.measurement_packets_per_point;
        const uint64_t packets_failed =
            (successful_packets >= packets_expected) ? 0 : (packets_expected - successful_packets);
        const double ber_decoded = (compared_bits > 0)
            ? static_cast<double>(bit_errors) / static_cast<double>(compared_bits)
            : std::numeric_limits<double>::quiet_NaN();
        const double bler = (packets_expected > 0)
            ? static_cast<double>(packets_failed) / static_cast<double>(packets_expected)
            : std::numeric_limits<double>::quiet_NaN();
        const double snr_mean = (frame_count > 0)
            ? (static_cast<double>(snr_db_sum_milli) / 1000.0 / static_cast<double>(frame_count))
            : std::numeric_limits<double>::quiet_NaN();
        const double evm_rms_mean = (frame_count > 0)
            ? (static_cast<double>(evm_rms_sum_micro) / 1.0e6 / static_cast<double>(frame_count))
            : std::numeric_limits<double>::quiet_NaN();
        const double evm_db_mean = (frame_count > 0)
            ? (static_cast<double>(evm_db_sum_milli) / 1000.0 / static_cast<double>(frame_count))
            : std::numeric_limits<double>::quiet_NaN();
        const double evm_first_db_mean = (frame_count > 0)
            ? (static_cast<double>(evm_first_db_sum_milli) / 1000.0 / static_cast<double>(frame_count))
            : std::numeric_limits<double>::quiet_NaN();
        const double evm_last_db_mean = (frame_count > 0)
            ? (static_cast<double>(evm_last_db_sum_milli) / 1000.0 / static_cast<double>(frame_count))
            : std::numeric_limits<double>::quiet_NaN();
        const double evm_max_db_mean = (frame_count > 0)
            ? (static_cast<double>(evm_max_db_sum_milli) / 1000.0 / static_cast<double>(frame_count))
            : std::numeric_limits<double>::quiet_NaN();
        const double evm_slope_db_per_symbol_mean = (frame_count > 0)
            ? (static_cast<double>(evm_slope_db_sum_milli) / 1000.0 / static_cast<double>(frame_count))
            : std::numeric_limits<double>::quiet_NaN();
        const double tx_gain_db = (tx_gain_x10 == std::numeric_limits<int32_t>::min())
            ? std::numeric_limits<double>::quiet_NaN()
            : static_cast<double>(tx_gain_x10) / 10.0;

        if (_measurement_summary_path.empty()) {
            return;
        }
        const std::vector<std::string> header{
            "run_id",
            "epoch_id",
            "tx_gain_db",
            "packets_expected",
            "packets_successful",
            "packets_failed",
            "compared_bits",
            "bit_errors",
            "ber_decoded",
            "bler",
            "frame_count",
            "estimated_snr_db_mean",
            "evm_rms_mean",
            "evm_db_mean",
            "evm_first_db_mean",
            "evm_last_db_mean",
            "evm_max_db_mean",
            "evm_slope_db_per_symbol_mean",
        };
        const std::vector<std::string> row{
            cfg_.measurement.measurement_run_id,
            std::to_string(epoch_id),
            std::to_string(tx_gain_db),
            std::to_string(packets_expected),
            std::to_string(successful_packets),
            std::to_string(packets_failed),
            std::to_string(compared_bits),
            std::to_string(bit_errors),
            std::to_string(ber_decoded),
            std::to_string(bler),
            std::to_string(frame_count),
            std::to_string(snr_mean),
            std::to_string(evm_rms_mean),
            std::to_string(evm_db_mean),
            std::to_string(evm_first_db_mean),
            std::to_string(evm_last_db_mean),
            std::to_string(evm_max_db_mean),
            std::to_string(evm_slope_db_per_symbol_mean),
        };
        if (!append_csv_row(_measurement_summary_path, header, row)) {
            LOG_G_WARN() << "Failed to append UE measurement row to "
                         << _measurement_summary_path;
        }
    }

    void _switch_measurement_epoch(uint32_t next_epoch_id) {
        if (!_measurement_enabled) {
            return;
        }
        const uint32_t previous_epoch =
            _measurement_active_epoch_id.exchange(0, std::memory_order_acq_rel);
        _finalize_measurement_epoch(previous_epoch);
        if (next_epoch_id > 0) {
            _measurement_active_epoch_id.store(next_epoch_id, std::memory_order_release);
        }
    }
    
    void init_filter() {
        const std::vector<float> filter_coeffs = {
            0.25f, 0.25f, 0.25f, 0.25f, 0.25f
        };
        
        freq_offset_filter_ = std::make_unique<FIRFilter>(filter_coeffs);
        freq_offset_filter_->warm_up(0.0f, filter_coeffs.size() * 2);
    }

    void init_usrp() {
        if (radio_is_sim(cfg_)) {
            init_sim_radio();
            return;
        }
        // Use device arguments from configuration
        usrp_ = uhd::usrp::multi_usrp::make(cfg_.usrp_device.device_args);
        usrp_->set_clock_source(cfg_.clock_time.clock_source);
        usrp_->set_rx_rate(cfg_.rf_sampling.sample_rate);
        usrp_->set_rx_bandwidth(cfg_.rf_sampling.bandwidth, cfg_.downlink.rx_channel);
        current_rx_tune_ = usrp_->set_rx_freq(uhd::tune_request_t(cfg_.downlink.center_freq), cfg_.downlink.rx_channel);
        tune_initialized_ = true;
        LOG_G_INFO() << "Actual RX RF Freq: " << format_freq_hz(current_rx_tune_.actual_rf_freq)
                     << " Hz, DSP: " << format_freq_hz(current_rx_tune_.actual_dsp_freq)
                     << " Hz";
        const auto gain_range = usrp_->get_rx_gain_range(cfg_.downlink.rx_channel);
        _rx_gain_min_db = gain_range.start();
        _rx_gain_max_db = gain_range.stop();
        const double initial_rx_gain_db = std::clamp(cfg_.rf_sampling.rx_gain, _rx_gain_min_db, _rx_gain_max_db);
        if (initial_rx_gain_db != cfg_.rf_sampling.rx_gain) {
            LOG_G_WARN() << "Configured rx_gain=" << cfg_.rf_sampling.rx_gain
                         << " dB is outside hardware range ["
                         << _rx_gain_min_db << ", " << _rx_gain_max_db
                         << "] dB. Clamping to " << initial_rx_gain_db << " dB.";
        }
        usrp_->set_rx_gain(initial_rx_gain_db, cfg_.downlink.rx_channel);
        _rx_agc.initialize(initial_rx_gain_db, _rx_gain_min_db, _rx_gain_max_db);
        _sync_search_gain_sweep.initialize(initial_rx_gain_db, _rx_gain_min_db, _rx_gain_max_db);
        LOG_G_INFO() << "RX gain range: [" << _rx_gain_min_db << ", " << _rx_gain_max_db
                     << "] dB, initial gain: " << initial_rx_gain_db << " dB";

        uhd::stream_args_t args("fc32", cfg_.downlink.rx_wire_format);
        args.args["block_id"] = "radio";
        args.channels = {cfg_.downlink.rx_channel};
        rx_stream_ = usrp_->get_rx_stream(args);

        if (uplink_enabled(cfg_)) {
            // Full-duplex uplink TX on the UE device. TDD: same carrier as RX.
            // FDD: the uplink carrier (duplex.ul_center_freq).
            const double ul_freq = (cfg_.uplink.duplex.mode == DuplexMode::FDD &&
                                    cfg_.uplink.duplex.ul_center_freq > 0.0)
                ? cfg_.uplink.duplex.ul_center_freq : cfg_.downlink.center_freq;
            usrp_->set_tx_rate(cfg_.rf_sampling.sample_rate);
            current_ul_tx_tune_ = usrp_->set_tx_freq(uhd::tune_request_t(ul_freq), cfg_.uplink.tx_channel);
            const auto tx_gain_range = usrp_->get_tx_gain_range(cfg_.uplink.tx_channel);
            _uplink_tx_gain_min_db = tx_gain_range.start();
            _uplink_tx_gain_max_db = tx_gain_range.stop();
            _uplink_tx_gain_restore_db = std::clamp(
                cfg_.uplink.tx_gain,
                _uplink_tx_gain_min_db,
                _uplink_tx_gain_max_db);
            _uplink_tx_gain_range_initialized = true;
            if (_uplink_tx_gain_restore_db != cfg_.uplink.tx_gain) {
                LOG_G_WARN() << "Configured tx_gain=" << cfg_.uplink.tx_gain
                             << " dB is outside hardware range ["
                             << _uplink_tx_gain_min_db << ", "
                             << _uplink_tx_gain_max_db
                             << "] dB. Clamping to "
                             << _uplink_tx_gain_restore_db << " dB.";
            }
            usrp_->set_tx_gain(_uplink_tx_gain_restore_db, cfg_.uplink.tx_channel);
            usrp_->set_tx_bandwidth(cfg_.rf_sampling.bandwidth, cfg_.uplink.tx_channel);
            uhd::stream_args_t tx_args("fc32", cfg_.uplink.wire_format_tx);
            tx_args.channels = {cfg_.uplink.tx_channel};
            _uplink_tx = std::make_unique<UplinkTxEngine>(cfg_);
            _uplink_tx->set_tx_stream(usrp_->get_tx_stream(tx_args));
            _uplink_tx->timing_advance().store(cfg_.uplink.ue_timing_advance, std::memory_order_relaxed);
            if (cfg_.should_profile("uplink")) {
                LOG_G_INFO() << "[UL-TX] uplink transmit enabled on TX ch " << cfg_.uplink.tx_channel
                             << " @ " << format_freq_hz(ul_freq) << " Hz, "
                             << _uplink_tx->uplink_config().ofdm.num_symbols << " UL symbols/frame, "
                             << "zc_root=" << _uplink_tx->uplink_config().ofdm.zc_root;
            }
        }
    }

    // Channel-simulator backend: attach to the hub's "rx.comm" ring instead of a USRP.
    void init_sim_radio() {
        _sim_radio = std::make_unique<SimRadio>();
        if (!_sim_radio->connect(cfg_.simulation)) {
            throw std::runtime_error("UE: failed to connect to ChannelSimulator session '" +
                                     cfg_.simulation.session + "'. Start ChannelSimulator first.");
        }
        rx_stream_ = _sim_radio->make_rx_streamer("rx.comm", cfg_.samples_per_frame());
        // Present a perfect tune so CFO/predictive-delay math sees zero tuning error.
        current_rx_tune_.target_rf_freq = cfg_.downlink.center_freq;
        current_rx_tune_.actual_rf_freq = cfg_.downlink.center_freq;
        current_rx_tune_.target_dsp_freq = 0.0;
        current_rx_tune_.actual_dsp_freq = 0.0;
        tune_initialized_ = true;
        _sim_radio->set_comm_rx_freq_correction_hz(0.0);
        // No hardware gain in simulation; expose a benign range so AGC/sweep stay inert.
        _rx_gain_min_db = 0.0;
        _rx_gain_max_db = 0.0;
        _rx_agc.initialize(0.0, _rx_gain_min_db, _rx_gain_max_db);
        _sync_search_gain_sweep.initialize(0.0, _rx_gain_min_db, _rx_gain_max_db);
        LOG_G_INFO() << "RX radio backend: SIMULATION (session='" << cfg_.simulation.session
                     << "', no USRP).";

        if (uplink_enabled(cfg_)) {
            _uplink_tx = std::make_unique<UplinkTxEngine>(cfg_);
            _uplink_tx->set_tx_stream(
                _sim_radio->make_tx_streamer("ul.tx", cfg_.samples_per_frame()));
            _uplink_tx->timing_advance().store(cfg_.uplink.ue_timing_advance, std::memory_order_relaxed);
            // sim: continuous send paced by shm backpressure (no timed scheduling).
            if (cfg_.should_profile("uplink")) {
                LOG_G_INFO() << "[UL-TX] uplink transmit enabled (sim ul.tx), "
                             << _uplink_tx->uplink_config().ofdm.num_symbols << " UL symbols/frame, "
                             << "zc_root=" << _uplink_tx->uplink_config().ofdm.zc_root;
            }
        }
    }

    void _schedule_shared_sensing_update(std::function<void(SharedSensingRuntime&)> updater) {
        if (!_bistatic_sensing_channel) {
            return;
        }
        SharedSensingRuntime snapshot;
        {
            std::lock_guard<std::mutex> lock(_shared_sensing_cfg_mutex);
            updater(_shared_sensing_cfg);
            _shared_sensing_cfg.generation++;
            const uint64_t next_symbol = _next_bistatic_frame_start_symbol.load(std::memory_order_relaxed);
            if (cfg_.ofdm.num_symbols == 0) {
                _shared_sensing_cfg.apply_symbol_index = next_symbol;
            } else {
                const uint64_t frame_len = static_cast<uint64_t>(cfg_.ofdm.num_symbols);
                const uint64_t boundary = ((next_symbol + frame_len - 1) / frame_len) * frame_len;
                _shared_sensing_cfg.apply_symbol_index = boundary;
            }
            snapshot = _shared_sensing_cfg;
        }
        _bistatic_sensing_channel->apply_shared_cfg(snapshot);
    }

    void _register_commands() {
        const bool compact_mask_mode = sensing_output_mode_is_compact_mask(cfg_);
        const bool compact_mask_fft_controls_supported =
            compact_mask_runtime_fft_controls_supported(cfg_);
        const std::string compact_mask_reason =
            compact_mask_runtime_fft_controls_reason(cfg_);
        const bool backend_processing_mode =
            backend_sensing_processing_supported(cfg_);

        _control_handler.register_command("SKIP", [this, compact_mask_mode, compact_mask_fft_controls_supported, compact_mask_reason, backend_processing_mode](int32_t value) {
            if (backend_processing_mode) {
                LOG_G_INFO() << "Ignoring SKIP command in backend sensing processing mode";
                return;
            }
            if (compact_mask_mode && !compact_mask_fft_controls_supported) {
                LOG_G_INFO() << "Ignoring SKIP command in compact_mask sensing mode: "
                             << (compact_mask_reason.empty() ? "mask is not local-DD compatible" : compact_mask_reason);
                return;
            }
            const bool new_skip = (value != 0);
            _schedule_shared_sensing_update([new_skip](SharedSensingRuntime& cfg) {
                cfg.skip_sensing_fft = new_skip;
            });
            LOG_G_INFO() << "Received skip sensing FFT command: " << value;
        });

        _control_handler.register_command("STRD", [this, compact_mask_mode](int32_t value) {
            if (compact_mask_mode) {
                LOG_G_INFO() << "Ignoring STRD command in compact_mask sensing mode: stride is defined by sensing_mask_blocks";
                return;
            }
            const size_t stride = value <= 0 ? 1 : static_cast<size_t>(value);
            const std::string stride_error = dense_sensing_stride_cfo_training_error(
                cfg_,
                stride,
                "Runtime STRD command");
            if (!stride_error.empty()) {
                LOG_G_WARN() << "Ignoring STRD command: " << stride_error;
                return;
            }
            _schedule_shared_sensing_update([stride](SharedSensingRuntime& cfg) {
                cfg.sensing_symbol_stride = stride;
            });
            LOG_G_INFO() << "Set sensing stride to: " << stride;
        });

        // Register alignment command
        _control_handler.register_command("ALGN", [this](int32_t value) {
            const int64_t max_adjust = static_cast<int64_t>(cfg_.samples_per_frame());
            const int64_t clamped_value = std::clamp<int64_t>(
                static_cast<int64_t>(value),
                -max_adjust,
                max_adjust
            );
            const int32_t adjusted_value = static_cast<int32_t>(clamped_value);
            _user_delay_offset = _user_delay_offset - static_cast<float>(adjusted_value);
            LOG_G_INFO() << "Received alignment command: " << adjusted_value << " samples";
        });

        // Timing Advance — pulls the UE uplink TX window earlier on the shared
        // clock so it lands aligned at the BS. Runtime-adjustable.
        _control_handler.register_command("TADV", [this](int32_t value) {
            // Rate-limit so rapid repeated commands don't thrash the framing.
            const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (now - _last_tadv_ns.load(std::memory_order_relaxed) < 50'000'000) {
                LOG_G_WARN() << "TADV rate-limited (<50ms since last); ignored " << value;
                return;
            }
            _last_tadv_ns.store(now, std::memory_order_relaxed);
            _ue_timing_advance.store(value, std::memory_order_relaxed);
            if (_uplink_tx) {
                _uplink_tx->timing_advance().store(value, std::memory_order_relaxed);
            }
            LOG_G_INFO() << "TADV (Timing Advance) set to " << value << " samples";
        });
        _control_handler.register_request("TADV", [this](
            int32_t, const ControlCommandHandler::ControlPeer& peer) {
            _control_handler.send_control_status(
                peer, "TADV", _ue_timing_advance.load(std::memory_order_relaxed));
        });

        _control_handler.register_command("MTI ", [this, compact_mask_mode, compact_mask_fft_controls_supported, compact_mask_reason](int32_t value) {
            if (compact_mask_mode && !compact_mask_fft_controls_supported) {
                LOG_G_INFO() << "Ignoring MTI command in compact_mask sensing mode: "
                             << (compact_mask_reason.empty() ? "mask is not local-DD compatible" : compact_mask_reason);
                return;
            }
            const bool enable = (value != 0);
            _schedule_shared_sensing_update([enable](SharedSensingRuntime& cfg) {
                cfg.enable_mti = enable;
            });
            LOG_G_INFO() << "MTI " << (enable ? "Enabled" : "Disabled");
        });

        _control_handler.register_command("CFEN", [this](int32_t value) {
            const bool enabled = (value != 0);
            _schedule_shared_sensing_update([enabled](SharedSensingRuntime& cfg) {
                cfg.cfar_enabled = enabled;
            });
        });
        _control_handler.register_command("CFTD", [this](int32_t value) {
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.cfar_train_doppler = std::max(0, value);
            });
        });
        _control_handler.register_command("CFTR", [this](int32_t value) {
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.cfar_train_range = std::max(0, value);
            });
        });
        _control_handler.register_command("CFGD", [this](int32_t value) {
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.cfar_guard_doppler = std::max(0, value);
            });
        });
        _control_handler.register_command("CFGR", [this](int32_t value) {
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.cfar_guard_range = std::max(0, value);
            });
        });
        _control_handler.register_command("CFAL", [this](int32_t value) {
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.cfar_alpha_db = static_cast<float>(value) / 100.0f;
            });
        });
        _control_handler.register_command("CFMR", [this](int32_t value) {
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.cfar_min_range_bin = std::max(0, value);
            });
        });
        _control_handler.register_command("CFDC", [this](int32_t value) {
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.cfar_dc_exclusion_bins = std::max(0, value);
            });
        });
        _control_handler.register_command("CFMP", [this](int32_t value) {
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.cfar_min_power_db = static_cast<float>(value) / 100.0f;
            });
        });
        _control_handler.register_command("CFRK", [this](int32_t value) {
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.cfar_os_rank_percent =
                    static_cast<float>(std::clamp(value, 0, 10000)) / 100.0f;
            });
        });
        _control_handler.register_command("CFSD", [this](int32_t value) {
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.cfar_os_suppress_doppler = std::max(0, value);
            });
        });
        _control_handler.register_command("CFSR", [this](int32_t value) {
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.cfar_os_suppress_range = std::max(0, value);
            });
        });
        _control_handler.register_command("MDEN", [this](int32_t value) {
            const bool enabled = (value != 0);
            _schedule_shared_sensing_update([enabled](SharedSensingRuntime& cfg) {
                cfg.micro_doppler_enabled = enabled;
            });
        });
        _control_handler.register_command("MDRB", [this](int32_t value) {
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.micro_doppler_range_bin = std::max(0, value);
            });
        });

        _control_handler.register_command("MRST", [this](int32_t value) {
            if (!_measurement_enabled) {
                LOG_G_WARN() << "MRST ignored: measurement mode disabled";
                return;
            }
            if (value <= 0) {
                LOG_G_WARN() << "MRST ignored: invalid epoch id " << value;
                return;
            }
            _switch_measurement_epoch(static_cast<uint32_t>(value));
            LOG_G_INFO() << "Measurement epoch reset to " << value;
        });

        _control_handler.register_command("CALB", [this](int32_t value) {
            if (!_bistatic_sensing_channel) {
                LOG_G_WARN() << "CALB ignored: bistatic sensing channel is not initialized";
                return;
            }
            const size_t target_symbols = value <= 0 ? 0u : static_cast<size_t>(value);
            _bistatic_sensing_channel->request_system_response_calibration(target_symbols);
        });

        _control_handler.register_request("PARM", [this](int32_t, const ControlCommandHandler::ControlPeer& peer) {
            if (!cfg_.sensing.bi_enabled) {
                return;
            }
            SharedSensingRuntime snapshot;
            {
                std::lock_guard<std::mutex> lock(_shared_sensing_cfg_mutex);
                snapshot = _shared_sensing_cfg;
            }
            const SensingViewerParamsPacket packet =
                make_sensing_viewer_params_packet(
                    cfg_,
                    snapshot.skip_sensing_fft,
                    snapshot.enable_mti,
                    snapshot.cfar_os_rank_percent,
                    snapshot.cfar_os_suppress_doppler,
                    snapshot.cfar_os_suppress_range,
                    true,
                    1u, 0x1u, false, 0.0);
            _control_handler.send_sensing_viewer_params(peer, packet);
        });
    }

    void init_sensing() {
        _actual_subcarrier_indices.resize(cfg_.ofdm.fft_size);
        const size_t half_fft = static_cast<int>(cfg_.ofdm.fft_size) / 2;
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < cfg_.ofdm.fft_size; ++i) {
            _actual_subcarrier_indices[i] = (i >= half_fft) ?
                (static_cast<int>(i) - cfg_.ofdm.fft_size) :
                static_cast<int>(i);
        }

        if (!cfg_.sensing.bi_enabled) {
            LOG_G_INFO() << "Bistatic sensing disabled by config.";
            return;
        }

        const CompactSensingMaskAnalysis compact_mask_analysis = analyze_compact_sensing_mask(cfg_);
        _shared_sensing_cfg.sensing_symbol_stride =
            compact_mask_analysis.local_delay_doppler_supported
                ? compact_mask_analysis.implicit_symbol_stride
                : cfg_.sensing.symbol_stride;
        _shared_sensing_cfg.enable_mti = true;
        _shared_sensing_cfg.skip_sensing_fft = true;
        _shared_sensing_cfg.generation = 1;
        _shared_sensing_cfg.apply_symbol_index = 0;
        _next_bistatic_frame_start_symbol.store(0, std::memory_order_relaxed);

        SensingRxChannelConfig bistatic_cfg;
        bistatic_cfg.enable_system_delay_estimation = false;
        bistatic_cfg.enable_sensing_output = cfg_.network_output.bi_sensing_output_enabled;
        _bistatic_sensing_channel = std::make_unique<SensingChannel>(
            cfg_,
            bistatic_cfg,
            SensingChannel::SensingRole::Bistatic,
            cfg_.network_output.bi_sensing_ip,
            cfg_.network_output.bi_sensing_port,
            0,
            running_,
            nullptr,
            nullptr,
            nullptr,
            [this](const std::string& ip, int port) {
                if (!_control_handler.send_heartbeat_to_last_peer()) {
                    _control_handler.send_heartbeat(ip, port);
                }
            },
            [](size_t) {
                return std::nullopt;
            }
        );
        _bistatic_sensing_channel->apply_shared_cfg(_shared_sensing_cfg);
        _bistatic_sensing_channel->start_bistatic();

        sensing_running_.store(true);
        sensing_thread_ = std::thread(&UEEngine::sensing_process_proc, this);
    }

    void init_data_processing() {
        // Initialize UDP output sender
        _udp_output_sender = std::make_unique<UdpSender>(
            cfg_.network_output.udp_output_ip,
            static_cast<uint16_t>(cfg_.network_output.udp_output_port),
            cfg_.network_output.udp_egress_pacer);

        // ARQ: configure DL RX window if enabled
        _arq_enabled = cfg_.network_output.arq_enabled;
        if (_arq_enabled) {
            _dl_arq_rx.configure(cfg_.network_output);
            _dl_arq_rx.set_direction(0); // downlink direction
            LOG_G_INFO() << "[UE ARQ] DL RX enabled: window="
                         << cfg_.network_output.arq_window_packets
                         << " ordered=" << cfg_.network_output.arq_ordered_delivery;
        }
    }

    void adjust_rx_freq(double detected_offset, bool reset = false) {
        // Create new tune request
        uhd::tune_request_t new_tune_req;
        new_tune_req.target_freq = current_rx_tune_.actual_rf_freq + detected_offset;
        new_tune_req.rf_freq = current_rx_tune_.actual_rf_freq; // Keep LO unchanged
        new_tune_req.dsp_freq = current_rx_tune_.actual_dsp_freq + detected_offset; // Update DSP only
        new_tune_req.rf_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
        new_tune_req.dsp_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
        if (reset) {
            // Reset tune request
            new_tune_req.target_freq = cfg_.downlink.center_freq;
            new_tune_req.dsp_freq = 0.0; // Reset DSP frequency
        }
        // Apply new tune (update DSP only, fast and does not affect LO).
        if (_sim_radio) {
            current_rx_tune_.target_rf_freq = new_tune_req.target_freq;
            current_rx_tune_.actual_rf_freq = cfg_.downlink.center_freq;
            current_rx_tune_.target_dsp_freq = new_tune_req.dsp_freq;
            current_rx_tune_.actual_dsp_freq = new_tune_req.dsp_freq;
            _sim_radio->set_comm_rx_freq_correction_hz(current_rx_tune_.actual_dsp_freq);
        } else {
            current_rx_tune_ = usrp_->set_rx_freq(new_tune_req, cfg_.downlink.rx_channel);
            _retune_uplink_tx_from_rx_correction(current_rx_tune_.actual_dsp_freq);
        }
    }

    void _retune_uplink_tx_from_rx_correction(double rx_dsp_correction_hz) {
        if (!_uplink_tx || _sim_radio || !usrp_) {
            return;
        }
        const double ul_base_freq =
            (cfg_.uplink.duplex.mode == DuplexMode::FDD && cfg_.uplink.duplex.ul_center_freq > 0.0)
                ? cfg_.uplink.duplex.ul_center_freq
                : cfg_.downlink.center_freq;
        double tx_target_correction_hz = rx_dsp_correction_hz;
        if (cfg_.uplink.duplex.mode == DuplexMode::FDD && cfg_.downlink.center_freq > 0.0 && ul_base_freq > 0.0) {
            tx_target_correction_hz *= ul_base_freq / cfg_.downlink.center_freq;
        }

        // UHD applies DSP tune signs differently: RX target = RF + DSP, TX target = RF - DSP.
        const double tx_dsp_correction_hz = -tx_target_correction_hz;
        uhd::tune_request_t tx_tune_req;
        tx_tune_req.target_freq = ul_base_freq + tx_target_correction_hz;
        tx_tune_req.rf_freq = ul_base_freq;
        tx_tune_req.dsp_freq = tx_dsp_correction_hz;
        tx_tune_req.rf_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
        tx_tune_req.dsp_freq_policy = uhd::tune_request_t::POLICY_MANUAL;
        current_ul_tx_tune_ = usrp_->set_tx_freq(tx_tune_req, cfg_.uplink.tx_channel);
        if (cfg_.should_profile("uplink")) {
            LOG_RT_INFO() << "[UL-TX] adjusted TX frequency with RX CFO correction: base="
                          << format_freq_hz(ul_base_freq)
                          << " Hz, target correction=" << format_freq_hz(tx_target_correction_hz)
                          << " Hz, requested TX DSP=" << format_freq_hz(tx_dsp_correction_hz)
                          << " Hz, actual RF=" << format_freq_hz(current_ul_tx_tune_.actual_rf_freq)
                          << " Hz, DSP=" << format_freq_hz(current_ul_tx_tune_.actual_dsp_freq)
                          << " Hz";
        }
    }
    void prepare_fftw() {
        fft_input_.resize(cfg_.ofdm.fft_size);
        fft_output_.resize(cfg_.ofdm.fft_size);
        fft_plan_ = fftwf_plan_dft_1d(cfg_.ofdm.fft_size,
            reinterpret_cast<fftwf_complex*>(fft_input_.data()),
            reinterpret_cast<fftwf_complex*>(fft_output_.data()),
            FFTW_FORWARD, FFTW_MEASURE);

        // Pre-size persistent symbol buffers (data symbols exclude sync and
        // full-band mid-frame pilot symbols).
        const size_t data_symbol_count = _data_resource_layout.data_symbol_count;
        _symbols_buf.resize(data_symbol_count);
        for (auto& s : _symbols_buf) s.resize(cfg_.ofdm.fft_size);
        _sync_symbol_freq_buf.resize(cfg_.ofdm.fft_size);
        _midframe_symbol_freq_bufs.resize(_data_resource_layout.midframe_pilot_symbol_count);
        for (auto& s : _midframe_symbol_freq_bufs) s.resize(cfg_.ofdm.fft_size);
        _midframe_tmp_h_buf.resize(cfg_.ofdm.fft_size);
        _midframe_interp_h_buf.resize(cfg_.ofdm.fft_size);
        _channel_anchor_h_bufs.reserve(1 + _data_resource_layout.midframe_pilot_symbol_count);
        _channel_anchor_symbols_buf.reserve(1 + _data_resource_layout.midframe_pilot_symbol_count);
        _tracking_h_inv_buf.resize(cfg_.ofdm.fft_size);
        _tracking_pilot_indices_buf.reserve(cfg_.ofdm.pilot_positions.size());
        _tracking_phase_buf.reserve(cfg_.ofdm.pilot_positions.size());
        _tracking_weights_buf.reserve(cfg_.ofdm.pilot_positions.size());

        // Note: ChannelEstimator and DelayProcessor now manage their own FFT plans internally
    }

    void init_zmq_publishers() {
        channel_pub_ = std::make_unique<ZmqByteSender>(cfg_.network_output.channel_ip, cfg_.network_output.channel_port);
        pdf_pub_ = std::make_unique<ZmqByteSender>(cfg_.network_output.pdf_ip, cfg_.network_output.pdf_port);
        constellation_pub_ = std::make_unique<ZmqByteSender>(cfg_.network_output.constellation_ip, cfg_.network_output.constellation_port);
        if (uplink_self_channel_debug_enabled(cfg_)) {
            uplink_self_channel_pub_ = std::make_unique<ZmqByteSender>(
                cfg_.network_output.uplink_self_channel_ip,
                static_cast<uint16_t>(cfg_.network_output.uplink_self_channel_port));
            uplink_self_pdf_pub_ = std::make_unique<ZmqByteSender>(
                cfg_.network_output.uplink_self_pdf_ip,
                static_cast<uint16_t>(cfg_.network_output.uplink_self_pdf_port));
            LOG_G_INFO() << "[UL-TX] self-channel debug streams: channel="
                         << cfg_.network_output.uplink_self_channel_ip << ':' << cfg_.network_output.uplink_self_channel_port
                         << ", pdf=" << cfg_.network_output.uplink_self_pdf_ip << ':'
                         << cfg_.network_output.uplink_self_pdf_port;
        }
        vofa_debug_sender_ = std::make_unique<VofaPlusDebugSender>(cfg_.network_output.vofa_debug_ip, cfg_.network_output.vofa_debug_port, 64);
    }

    void _clear_frame_queue() {
        RxFrame frame;
        while (frame_queue_.try_pop(frame)) {
            _rx_frame_pool.release(std::move(frame));
        }
    }

    void _clear_sensing_queue() {
        SensingFrame frame;
        while (sensing_queue_.try_pop(frame)) {
            _sensing_frame_pool.release(std::move(frame));
        }
    }

    void _clear_llr_queue() {
        LlrFrame frame_llr;
        while (_data_llr_buffer.try_pop(frame_llr)) {
            _llr_pool.release(std::move(frame_llr.llr));
        }
        LlrFrameI16 frame_llr_i16;
        while (_data_llr_buffer_i16.try_pop(frame_llr_i16)) {
            _llr_pool_i16.release(std::move(frame_llr_i16.llr));
        }
    }

    void _enter_sync_search_state() {
        _sync_generation.fetch_add(1, std::memory_order_acq_rel);
        _sync_in_progress = false;
        _delay_adjustment_count = 0;
        _last_delay_index_err = 0;
        _set_uplink_waveform_enabled(false);
        _llr_snr_linear_filtered = 1.0;
        _llr_snr_filter_initialized = false;
        const bool log_agc = cfg_.should_profile("agc");
        double search_gain_db = 0.0;
        const bool reset_search_gain = _sync_search_gain_sweep.reset_to_default(
            [this](double gain_db) {
                if (!_sim_radio) usrp_->set_rx_gain(gain_db, cfg_.downlink.rx_channel);
            },
            [this](double gain_db) {
                _rx_agc.sync_to_gain(gain_db);
            },
            &search_gain_db
        );
        if (reset_search_gain && log_agc) {
            LOG_RT_INFO() << "Search RX AGC reset gain to default: " << search_gain_db << " dB";
        }
        _clear_frame_queue();
        sync_queue_.clear();
        _clear_sensing_queue();
        _clear_llr_queue();
        state_ = RxState::SYNC_SEARCH;
    }

    /**
     * @brief Rx Streamer Thread Function.
     * 
     * Continuous loop that receives baseband samples from the USRP.
     * Implements a state machine (SYNC_SEARCH -> ALIGNMENT -> NORMAL) to handle
     * frame synchronization and alignment before normal reception.
     */
    void rx_proc(uhd::time_spec_t stream_start_time) {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        uhd::set_thread_priority_safe(1.0, true);
        bind_current_thread_from_downlink_hint(cfg_, 0);
        uhd::rx_metadata_t md;
        auto issue_start = [&](const uhd::time_spec_t& start_time) {
            uhd::stream_cmd_t cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
            const bool timed_start = start_time.get_real_secs() > 0.0;
            cmd.stream_now = !timed_start;
            if (timed_start) {
                cmd.time_spec = start_time;
                LOG_G_INFO() << "[UE] timed RX stream start at "
                             << start_time.get_real_secs()
                             << " s, shared with UL-TX";
            } else {
                LOG_G_INFO() << "[UE] immediate RX stream start";
            }
            rx_stream_->issue_stream_cmd(cmd);
        };
        auto issue_stop = [&]() {
            try {
                rx_stream_->issue_stream_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
            } catch (const std::exception& e) {
                LOG_RT_WARN() << "[UE RX] failed to stop stream before restart: " << e.what();
            }
        };

        issue_start(stream_start_time);

        while (running_.load()) {
            const bool tx_requested_restart =
                _uplink_tx &&
                (_uplink_tx->tx_error_count().load(std::memory_order_relaxed) !=
                 _handled_ul_tx_error_count.load(std::memory_order_relaxed));
            if (_stream_restart_requested.exchange(false, std::memory_order_acq_rel) ||
                tx_requested_restart) {
                if (_uplink_tx) {
                    _handled_ul_tx_error_count.store(
                        _uplink_tx->tx_error_count().load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
                }
                issue_stop();
                stream_start_time = _next_timed_stream_start();
                if (_uplink_tx && stream_start_time.get_real_secs() > 0.0) {
                    _uplink_tx->reschedule_timed_tx(stream_start_time);
                }
                _reset_receive_state_after_stream_restart();
                issue_start(stream_start_time);
                LOG_RT_WARN() << "[UE] shared RX/UL-TX restart at "
                              << stream_start_time.get_real_secs() << " s";
                continue;
            }

            switch (state_.load()) {
            case RxState::SYNC_SEARCH:
                handle_sync_search(md);
                break;
            case RxState::ALIGNMENT:
                handle_alignment(md);
                break;
            case RxState::NORMAL:
                handle_normal_rx(md);
                break;
            }
        }

        issue_stop();
    }

    /**
     * @brief Handle Synchronization Search State.
     * 
     * Reads a blocks of samples and pushes them to the sync queue for correlation.
     * Used to find the start of a frame.
     */
    void handle_sync_search(uhd::rx_metadata_t& md) {
        SyncBatch* sync_batch = sync_queue_.producer_slot();
        AlignedVector& target_buffer =
            (sync_batch != nullptr) ? sync_batch->data : _sync_scratch_buffer;

        int64_t first_time_ns = -1;
        size_t received = 0;
        while (received < cfg_.sync_samples() && running_.load()) {
            const size_t got = rx_stream_->recv(
                &target_buffer[received],
                cfg_.sync_samples() - received,
                md
            );
            bool restart_read = false;
            if (_handle_rx_metadata_error(md, "sync search", &restart_read) && restart_read) {
                break;
            }
            if (received == 0 && got > 0) {
                first_time_ns = metadata_time_to_ns(md);
            }
            received += got;
        }

        if (sync_batch == nullptr || received < cfg_.sync_samples()) {
            return;
        }
        sync_batch->usrp_time_ns = first_time_ns;
        sync_queue_.producer_commit();
    }

    /**
     * @brief Handle Frame Alignment State.
     * 
     * Once sync is found, this state aligns the sample stream to the frame boundary
     * by discarding a specific number of samples (discard_samples_). This is also 
     * used for timing adjustments during normal operation.
     */
    void handle_alignment(uhd::rx_metadata_t& md) {
        const bool do_latency_profile =
            cfg_.should_profile("demodulation") && cfg_.should_profile("latency");
        const int alignment_samples = discard_samples_.load(std::memory_order_relaxed);
        const size_t frame_samples = cfg_.samples_per_frame();
        const size_t negative_shift = (alignment_samples < 0)
            ? std::min<size_t>(static_cast<size_t>(-alignment_samples), frame_samples)
            : 0;
        const size_t positive_shift = (alignment_samples > 0)
            ? static_cast<size_t>(alignment_samples)
            : 0;
        const size_t total_read = frame_samples + positive_shift - negative_shift;
        AlignedVector temp_buf(total_read);
        size_t received = 0;
        int64_t frame_time_ns = -1;
        while (received < total_read && running_.load()) {
            const size_t got = rx_stream_->recv(&temp_buf[received], total_read - received, md);
            bool restart_read = false;
            if (_handle_rx_metadata_error(md, "alignment", &restart_read) && restart_read) {
                break;
            }
            if (received == 0 && got > 0) {
                frame_time_ns = metadata_time_to_ns(md);
            }
            received += got;
        }
        if (received < total_read) {
            return;
        }
        // Acquire pre-allocated RX frame from pool
        RxFrame frame = _rx_frame_pool.acquire();
        frame.Alignment = alignment_samples;
        frame.usrp_time_ns = frame_time_ns;
        frame.host_enqueue_time_ns = do_latency_profile ? host_now_ns() : 0;
        frame.generation = _sync_generation.load(std::memory_order_acquire);
        if (positive_shift > 0) {
            std::copy(temp_buf.begin() + positive_shift,
                    temp_buf.begin() + positive_shift + frame_samples,
                    frame.frame_data.begin());
        } else if (negative_shift > 0) {
            std::fill(frame.frame_data.begin(), frame.frame_data.end(),
                      std::complex<float>(0.0f, 0.0f));
            std::copy(temp_buf.begin(),
                    temp_buf.end(),
                    frame.frame_data.begin() + negative_shift);
        } else {
            std::copy(temp_buf.begin(),
                    temp_buf.begin() + frame_samples,
                    frame.frame_data.begin());
        }

        if (!frame_queue_.try_push(std::move(frame))) {
            LOG_RT_WARN_HZ(5) << "RX frame queue full during alignment, dropping newest frame";
            _rx_frame_pool.release(std::move(frame));
        }
//        LOG_G_INFO() << "Alignment done, moving "<< discard_samples_<< " samples" << std::endl;
//        LOG_G_INFO() <<  discard_samples_<< std::endl;
        _set_uplink_waveform_enabled(true);
        state_ = RxState::NORMAL;
    }

    /**
     * @brief Handle Normal Reception State.
     * 
     * Reads complete frames of aligned samples and pushes them to the frame queue
     * for processing.
     */
    void handle_normal_rx(uhd::rx_metadata_t& md) {
        const bool do_latency_profile =
            cfg_.should_profile("demodulation") && cfg_.should_profile("latency");
        // Acquire pre-allocated RX frame from pool
        RxFrame frame = _rx_frame_pool.acquire();
        frame.Alignment = 0;
        frame.usrp_time_ns = -1;
        frame.host_enqueue_time_ns = 0;
        frame.generation = _sync_generation.load(std::memory_order_acquire);
        size_t received = 0;

        while (received < cfg_.samples_per_frame() && running_.load()) {
            const size_t got = rx_stream_->recv(
                &frame.frame_data[received],
                cfg_.samples_per_frame() - received,
                md
            );
            bool restart_read = false;
            if (_handle_rx_metadata_error(md, "normal", &restart_read) && restart_read) {
                break;
            }
            if (received == 0 && got > 0) {
                frame.usrp_time_ns = metadata_time_to_ns(md);
            }
            received += got;
        }

        if (received == cfg_.samples_per_frame()) {
            if (do_latency_profile) {
                frame.host_enqueue_time_ns = host_now_ns();
            }
            if (!frame_queue_.try_push(std::move(frame))) {
                LOG_RT_WARN_HZ(5) << "RX frame queue full, dropping newest frame";
                _rx_frame_pool.release(std::move(frame));
            }
        } else {
            _rx_frame_pool.release(std::move(frame));
        }
    }

    /**
     * @brief Process Synchronization Data.
     * 
     * Performs cross-correlation with the local Zadoff-Chu sequence to detect frame start.
     * Also estimates coarse Carrier Frequency Offset (CFO) using Cyclic Prefix (CP) correlation.
     */
    void process_sync_data(const AlignedVector& sync_data, int64_t sync_time_ns) {
        const size_t symbol_len = cfg_.ofdm.fft_size + cfg_.ofdm.cp_length;
        const bool allow_freq_adjust = _control_time_gates.allow_freq_adjust(sync_time_ns);
        bool issued_freq_adjust = false;
        bool issued_alignment = false;

        constexpr float kZcEnergyThreshold = 100.0f;
        constexpr float kSecSyncMetricThreshold = 0.10f;
        constexpr float kCfoTrainingMetricThreshold = 0.10f;
        const size_t local_zc_search_radius = 2 * cfg_.ofdm.cp_length;
        const bool log_sync_profile = cfg_.should_profile("sync");
        const bool collect_alias_candidates =
            log_sync_profile || cfo_training_sequence_enabled(cfg_);

        int max_pos = 0;
        float max_corr = 0.0f;
        float avg_corr = 0.0f;
        double initial_cfo_hz = 0.0;
        bool used_sec_sync_symbol = false;
        bool sync_found = false;
        auto estimate_cfo_training = [&](int sync_pos_samples) {
            SyncProcessor::CfoTrainingEstimate estimate;
            if (!cfo_training_sequence_enabled(cfg_) || sync_pos_samples < 0) {
                return estimate;
            }
            const size_t cfo_symbol_start =
                static_cast<size_t>(sync_pos_samples) + symbol_len;
            estimate = SyncProcessor::estimate_cfo_from_training_symbol(
                sync_data,
                cfo_symbol_start,
                cfg_.ofdm.fft_size,
                cfg_.ofdm.cp_length,
                cfg_.ofdm.cfo_training_period_samples,
                cfg_.rf_sampling.sample_rate);
            if (log_sync_profile) {
                LOG_RT_INFO() << "CFO training field estimate: metric="
                              << estimate.metric
                              << ", cfo=" << estimate.cfo_hz << " Hz"
                              << ", symbol_start=" << cfo_symbol_start;
            }
            return estimate;
        };

        if (cfg_.ofdm.enable_sec_sync_symbol) {
            const auto sec_result = SyncProcessor::detect_sec_sync_symbol(
                sync_data,
                cfg_.ofdm.fft_size,
                cfg_.ofdm.cp_length,
                cfg_.rf_sampling.sample_rate);
            if (sec_result.valid && sec_result.max_metric > kSecSyncMetricThreshold) {
                const int expected_zc_start = sec_result.coarse_symbol_start +
                    static_cast<int>(symbol_len);
                const int max_search_start = sync_data.size() >= symbol_len
                    ? static_cast<int>(sync_data.size() - symbol_len)
                    : 0;
                const size_t search_start = static_cast<size_t>(std::max(
                    0,
                    expected_zc_start - static_cast<int>(local_zc_search_radius)));
                const size_t search_end = static_cast<size_t>(std::max(
                    0,
                    std::min(
                        max_search_start,
                        expected_zc_start + static_cast<int>(local_zc_search_radius))));
                const double sec_alias_period_hz =
                    cfg_.rf_sampling.sample_rate / static_cast<double>(symbol_len);
                const int sec_alias_span = sync_cfo_alias_span_from_range(
                    cfg_.sync_tracking.sync_cfo_alias_search_range_hz,
                    sec_alias_period_hz);

                auto refine_result = _sync_processor.refine_sec_sync_with_alias_search(
                    sync_data,
                    sec_result,
                    cfg_.rf_sampling.sample_rate,
                    search_start,
                    search_end,
                    sec_alias_span,
                    _sync_cfo_compensated_buffer,
                    collect_alias_candidates);
                const auto cfo_training_est = estimate_cfo_training(refine_result.max_pos);
                const bool cfo_training_alias_selected =
                    cfo_training_est.valid &&
                    cfo_training_est.metric > kCfoTrainingMetricThreshold &&
                    SyncProcessor::select_alias_closest_to_cfo_reference(
                        refine_result,
                        cfo_training_est.cfo_hz);
                if (log_sync_profile) {
                    LOG_RT_INFO() << format_sync_alias_candidates(
                        "Second sync ZC refine",
                        refine_result);
                }
                max_pos = refine_result.max_pos;
                max_corr = refine_result.max_corr;
                avg_corr = refine_result.avg_corr;
                if (avg_corr > 0.0f && (max_corr / avg_corr) > kZcEnergyThreshold) {
                    used_sec_sync_symbol = true;
                    sync_found = true;
                    initial_cfo_hz = refine_result.cfo_hz;
                    SyncProcessor::apply_cfo_compensation(
                        sync_data,
                        cfg_.rf_sampling.sample_rate,
                        initial_cfo_hz,
                        _sync_cfo_compensated_buffer);
                    LOG_RT_INFO() << "Second sync symbol coarse metric: " << sec_result.max_metric
                                  << ", coarse symbol start: " << sec_result.coarse_symbol_start
                                  << ", modulo CFO: " << sec_result.coarse_cfo_hz
                                  << " Hz, alias index: " << refine_result.alias_index
                                  << ", refined CFO: " << initial_cfo_hz << " Hz"
                                  << (cfo_training_alias_selected ? " (CFO field deambiguated)" : "");
                } else {
                    LOG_RT_WARN() << "Second sync symbol coarse sync detected but local ZC refine failed."
                                  << " metric=" << sec_result.max_metric
                                  << " local_peak_ratio=" << (avg_corr > 0.0f ? (max_corr / avg_corr) : 0.0f)
                                  << ". Falling back to legacy ZC sync search.";
                }
            }
        }

        if (!sync_found) {
            _sync_processor.find_sync_position(sync_data, max_pos, max_corr, avg_corr);
            sync_found = (avg_corr > 0.0f) && ((max_corr / avg_corr) > kZcEnergyThreshold);
        }

        if (sync_found) {
            _restore_uplink_tx_gain_after_sync();
            _sync_search_gain_sweep.note_sync_found();
            LOG_RT_INFO() << "Sync found at pos: " << max_pos
                          << " with peak/avg: " << peak_ratio_db(max_corr, avg_corr)
                          << " dB (peak=" << power_to_db(max_corr)
                          << " dB, avg=" << power_to_db(avg_corr)
                          << " dB) Threshold: " << power_to_db(kZcEnergyThreshold)
                          << " dB";
            int symbol_offset = max_pos % symbol_len;
            // Calculate available symbol count
            const size_t available_symbols = std::min(
                static_cast<size_t>(cfg_.ofdm.num_symbols*2),
                (sync_data.size() - symbol_offset) / symbol_len
            );
            
            // Use SyncProcessor for CFO estimation
            if (available_symbols > 0) {
                const AlignedVector& cfo_est_data =
                    used_sec_sync_symbol ? _sync_cfo_compensated_buffer : sync_data;
                double phase_diff = SyncProcessor::estimate_cfo_phase(
                    cfo_est_data, symbol_offset, available_symbols,
                    symbol_len, cfg_.ofdm.cp_length, cfg_.ofdm.fft_size
                );
                const double residual_cfo_hz =
                    SyncProcessor::phase_to_cfo(phase_diff, cfg_.rf_sampling.sample_rate, cfg_.ofdm.fft_size);
                double cfo = used_sec_sync_symbol
                    ? (initial_cfo_hz + residual_cfo_hz)
                    : residual_cfo_hz;
                if (!used_sec_sync_symbol) {
                    const int max_search_start = sync_data.size() >= symbol_len
                        ? static_cast<int>(sync_data.size() - symbol_len)
                        : 0;
                    const size_t cp_alias_search_start = static_cast<size_t>(std::max(
                        0,
                        max_pos - static_cast<int>(local_zc_search_radius)));
                    const size_t cp_alias_search_end = static_cast<size_t>(std::max(
                        0,
                        std::min(
                            max_search_start,
                            max_pos + static_cast<int>(local_zc_search_radius))));
                    const double cp_alias_period_hz =
                        cfg_.rf_sampling.sample_rate / static_cast<double>(cfg_.ofdm.fft_size);
                    const int cp_alias_span = sync_cfo_alias_span_from_range(
                        cfg_.sync_tracking.sync_cfo_alias_search_range_hz,
                        cp_alias_period_hz);
                    auto cp_refine = _sync_processor.refine_sync_cfo_alias(
                        sync_data,
                        residual_cfo_hz,
                        cp_alias_period_hz,
                        cfg_.rf_sampling.sample_rate,
                        cp_alias_search_start,
                        cp_alias_search_end,
                        cp_alias_span,
                        _sync_cfo_compensated_buffer,
                        collect_alias_candidates);
                    const auto cfo_training_est = estimate_cfo_training(cp_refine.max_pos);
                    const bool cfo_training_alias_selected =
                        cfo_training_est.valid &&
                        cfo_training_est.metric > kCfoTrainingMetricThreshold &&
                        SyncProcessor::select_alias_closest_to_cfo_reference(
                            cp_refine,
                            cfo_training_est.cfo_hz);
                    if (log_sync_profile) {
                        LOG_RT_INFO() << format_sync_alias_candidates(
                            "CP CFO ZC refine",
                            cp_refine);
                    }
                    if (cp_refine.valid && cp_refine.avg_corr > 0.0f &&
                        (cp_refine.max_corr / cp_refine.avg_corr) > kZcEnergyThreshold) {
                        cfo = cp_refine.cfo_hz;
                        max_pos = cp_refine.max_pos;
                        LOG_RT_INFO() << "CP CFO alias refine: modulo CFO=" << residual_cfo_hz
                                      << " Hz, alias index=" << cp_refine.alias_index
                                      << ", refined CFO=" << cfo << " Hz"
                                      << (cfo_training_alias_selected ? " (CFO field deambiguated)" : "");
                    }
                }
                
                LOG_RT_INFO() << "CFO estimate: " << cfo << " Hz (using " << available_symbols
                              << " symbols)";

                int predictive_delay_samples = 0;
                if (cfg_.sync_tracking.predictive_delay) {
                    const int frame_start_offset_samples =
                        max_pos - static_cast<int>(cfg_.ofdm.sync_pos * symbol_len);
                    const int64_t detected_frame_time_ns =
                        sync_time_ns +
                        static_cast<int64_t>(std::llround(
                            static_cast<double>(frame_start_offset_samples) * 1.0e9 / cfg_.rf_sampling.sample_rate));
                    predictive_delay_samples =
                        _predictive_delay_samples_from_cfo(
                            cfg_,
                            detected_frame_time_ns,
                            cfo,
                            current_rx_tune_.actual_rf_freq,
                            current_rx_tune_.actual_dsp_freq,
                            time_spec_to_ns(radio_time_now()));
                }
                
                // Perform initial CFO correction
                if (cfg_.sync_tracking.software_sync && allow_freq_adjust){
                    adjust_rx_freq(-cfo, false);
                    issued_freq_adjust = true;
                }

                // Record time offset. A fresh sync-search acquisition must always
                // schedule alignment; otherwise RX/TX windows can remain at the
                // old/reset position even though sync detection succeeded.
                sync_offset_ =
                    (max_pos - cfg_.ofdm.sync_pos * symbol_len - cfg_.sync_tracking.desired_peak_pos +
                     predictive_delay_samples);
                if (sync_offset_ > 0) {
                    sync_offset_ = sync_offset_ % cfg_.samples_per_frame();
                }
                _clear_frame_queue();
                _schedule_receive_alignment(static_cast<int32_t>(sync_offset_));
                issued_alignment = true;
            } else {
                LOG_RT_WARN() << "No valid symbols for CFO estimation";
            }
            if (issued_freq_adjust) {
                _control_time_gates.mark_freq_adjust_now(radio_time_now());
            }
            if (issued_alignment) {
                _control_time_gates.mark_alignment_now(radio_time_now());
            }
        } else {
            const size_t frame_samples = cfg_.samples_per_frame();
            const size_t search_equivalent_frames =
                (frame_samples > 0) ? std::max<size_t>(1, sync_data.size() / frame_samples) : 1;
            double search_gain_db = 0.0;
            const bool stepped_search_gain = _sync_search_gain_sweep.on_search_miss(
                search_equivalent_frames,
                [this](double gain_db) {
                    if (!_sim_radio) usrp_->set_rx_gain(gain_db, cfg_.downlink.rx_channel);
                },
                [this](double gain_db) {
                    _rx_agc.sync_to_gain(gain_db);
                },
                &search_gain_db
            );
            if (stepped_search_gain && cfg_.should_profile("agc")) {
                LOG_RT_INFO() << "Search RX AGC stepped gain to " << search_gain_db
                              << " dB after sync miss";
            }
        }
    }

    void process_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        uhd::set_thread_priority_safe(1, true);
        bind_current_thread_from_downlink_hint(cfg_, 1);
        
        using Clock = std::chrono::high_resolution_clock;
        Clock::time_point frame_start, frame_end;
        double total_processing_time = 0.0;
        int frame_count = 0;
        constexpr int REPORT_INTERVAL = 434;
        SPSCBackoff sync_backoff;
        const bool do_latency_profile =
            cfg_.should_profile("demodulation") && cfg_.should_profile("latency");

        while (running_.load()) {
            if (state_ == RxState::SYNC_SEARCH) {
                SyncBatch* sync_batch = sync_queue_.consumer_slot();
                if (sync_batch == nullptr) {
                    sync_backoff.pause();
                    continue;
                }
                sync_backoff.reset();
                process_sync_data(sync_batch->data, sync_batch->usrp_time_ns);
                sync_queue_.consumer_pop();
            } else {
                RxFrame frame = wait_for_frame();
                if (frame.frame_data.empty()) {
                    if (!running_.load()) {
                        break;
                    }
                    continue;
                }
                frame_start = Clock::now();
                const int64_t frame_dequeue_time_ns = do_latency_profile ? host_now_ns() : 0;
                process_ofdm_frame(frame, frame_dequeue_time_ns);
                // Return frame to pool for reuse
                _rx_frame_pool.release(std::move(frame));
                frame_end = Clock::now();
                double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
                total_processing_time += frame_time;
                frame_count++;

                if (frame_count >= REPORT_INTERVAL) {
                    double avg_time = total_processing_time / frame_count;
                    double frame_duration = cfg_.samples_per_frame() / cfg_.rf_sampling.sample_rate * 1000.0;
                    double load = avg_time / frame_duration;
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2)
                        << "Average processing time: " << avg_time
                        << " ms, Load: " << load * 100.0 << "%; "
                        << "Actual RX RF Freq: " << format_freq_hz(current_rx_tune_.actual_rf_freq)
                        << " Hz, DSP: " << format_freq_hz(current_rx_tune_.actual_dsp_freq)
                        << " Hz; Average CFO: " << _avg_freq_offset << " Hz";
                    if (do_latency_profile) {
                        const LatencySnapshot latency = _take_latency_snapshot_and_reset();
                        oss << "\n";
                        if (latency.count > 0) {
                            const double n = static_cast<double>(latency.count);
                            oss << "\n---------- Latency (avg per valid frame, ms) ----------\n"
                                << "RX frame queue wait:         " << (latency.rx_queue_total_ns / n) * 1e-6 << " ms\n"
                                << "Dequeue + FFT/EQ/LLR queue:  " << (latency.demod_total_ns / n) * 1e-6 << " ms\n"
                                << "Bit queue + LDPC/UDP out:    " << (latency.bit_total_ns / n) * 1e-6 << " ms\n"
                                << "TOTAL E2E (excl. RX wait):   " << (latency.e2e_total_ns / n) * 1e-6 << " ms\n"
                                << "Latency sample count:        " << latency.count << "\n";
                        } else {
                            oss << "\n---------- Latency (avg per valid frame, ms) ----------\n"
                                << "No valid latency samples in this interval.\n";
                        }
                    }
                    LOG_RT_INFO() << oss.str();
                    total_processing_time = 0.0;
                    frame_count = 0;
                }
            }
        }
    }

    void _publish_uplink_self_channel_debug(const RxFrame& frame) {
        if (!uplink_self_channel_debug_enabled(cfg_) || !_uplink_tx) {
            return;
        }

        const Config ul_cfg = _uplink_tx->uplink_config();
        if (ul_cfg.ofdm.num_symbols == 0) {
            return;
        }
        const size_t sym_len = cfg_.ofdm.fft_size + cfg_.ofdm.cp_length;
        const size_t local_tx_zc_start =
            _duplex_layout.ul_sample_offset(cfg_) + ul_cfg.ofdm.sync_pos * sym_len;

        if (!_uplink_self_debug_estimator.estimate(
                frame.frame_data,
                local_tx_zc_start,
                0,
                _uplink_self_zc_freq,
                _uplink_self_h_est)) {
            return;
        }

        uplink_self_channel_sender_.add_data(
            AlignedVector(_uplink_self_h_est.begin(), _uplink_self_h_est.end()));

        _delay_processor.compute_delay_spectrum(_uplink_self_h_est, _uplink_self_delay_spectrum);
        uplink_self_pdf_sender_.add_data(
            AlignedVector(_uplink_self_delay_spectrum.begin(), _uplink_self_delay_spectrum.end()));

        if ((_uplink_self_debug_frame_counter++ % 4096) == 0) {
            LOG_G_INFO() << "[UL-TX] self-channel debug: local UE UL ZC offset="
                         << local_tx_zc_start
                         << ", rx_frame_start_offset=0"
                         << ", frame_period=" << frame.frame_data.size();
        }
    }

    RxFrame wait_for_frame() {
        SPSCBackoff frame_backoff;
        while (running_.load()) {
            if (!running_.load()) {
                return RxFrame{};
            }
            if (state_.load() == RxState::SYNC_SEARCH) {
                return RxFrame{};
            }

            RxFrame frame;
            if (!frame_queue_.try_pop(frame)) {
                frame_backoff.pause();
                continue;
            }
            frame_backoff.reset();
            if (frame.generation != _sync_generation.load(std::memory_order_acquire)) {
                _rx_frame_pool.release(std::move(frame));
                continue;
            }
            return frame;
        }
        return RxFrame{};
    }

    /**
     * @brief Main OFDM Frame Processing Pipeline.
     * 
     * Steps:
     * 1. FFT: Convert time-domain samples to frequency domain.
     * 2. Channel Estimation: Use pilot symbols to estimate channel response.
     * 3. SFO/CFO Estimation: Refine offset estimates.
     * 4. Equalization: Compensate for channel effects (Zero-Forcing).
     * 5. Sensing: Extract Micro-Doppler signature and valid range/doppler data.
     * 6. LLR Calculation: Compute Log-Likelihood Ratios for soft decoding.
     */
    void process_ofdm_frame(const RxFrame& frame, int64_t frame_dequeue_time_ns) {
        // ============== Profiling variables ==============
        using ProfileClock = std::chrono::high_resolution_clock;
        static double prof_fft_total = 0.0;
        static double prof_channel_est_total = 0.0;
        static double prof_cfo_sfo_est_total = 0.0;
        static double prof_equalization_total = 0.0;
        static double prof_eq_base_inv_total = 0.0;
        static double prof_eq_channel_select_total = 0.0;
        static double prof_eq_symbol_inv_total = 0.0;
        static double prof_eq_pilot_phase_total = 0.0;
        static double prof_eq_apply_total = 0.0;
        static uint64_t prof_eq_data_symbols_total = 0;
        static uint64_t prof_eq_midframe_channel_symbols_total = 0;
        static uint64_t prof_eq_symbol_inv_count_total = 0;
        static uint64_t prof_eq_pilot_phase_attempt_total = 0;
        static uint64_t prof_eq_pilot_phase_success_total = 0;
        static double prof_noise_est_total = 0.0;
        static double prof_remodulate_total = 0.0;
        static double prof_delay_spectrum_total = 0.0;
        static double prof_timing_sync_total = 0.0;
        static double prof_sensing_queue_total = 0.0;
        static double prof_udp_send_total = 0.0;
        static double prof_llr_total = 0.0;
        static int prof_frame_count = 0;
        constexpr int PROF_REPORT_INTERVAL = 434;
        const bool do_latency_profile =
            cfg_.should_profile("demodulation") && cfg_.should_profile("latency");
        // Dedicated switch for the per-symbol equalization channel breakdown
        // (Eq base H_inv / channel select / symbol H_inv / pilot phase / apply).
        // Kept separate from "latency" so the heavy per-symbol timers can be
        // toggled independently of end-to-end latency instrumentation.
        const bool do_eq_breakdown =
            cfg_.should_profile("demodulation") && cfg_.should_profile("breakdown_eq");

        auto prof_step_start = ProfileClock::now();
        auto prof_step_end = prof_step_start;
        // =================================================
        
        const bool sensing_enabled = static_cast<bool>(_bistatic_sensing_channel);
        SensingFrame sense_frame;
        if (sensing_enabled) {
            // Acquire the pooled frame only when bistatic sensing is enabled.
            sense_frame = _sensing_frame_pool.acquire();
            sense_frame.rx_symbols.resize(cfg_.ofdm.num_symbols);
            sense_frame.tx_symbols.resize(cfg_.ofdm.num_symbols);
        }

        // Reuse persistent buffers instead of allocating one AlignedVector per
        // symbol every frame. _symbols_buf holds the cfg_.ofdm.num_symbols-1 data
        // symbols; the sync symbol lands in _sync_symbol_freq_buf.
        std::vector<AlignedVector>& symbols = _symbols_buf;
        AlignedVector& sync_symbol_freq = _sync_symbol_freq_buf;
        const size_t scale_n = cfg_.ofdm.fft_size;
        const float scale = 1.0f / sqrtf(static_cast<float>(scale_n));
        size_t pos = 0;
        prof_step_start = ProfileClock::now();
        // Per-symbol FFT is intentionally kept (vs. a frame-wide
        // fftwf_plan_many_dft or new-array execute straight from frame_data):
        // each fft_size-point transform stays L1/L2 resident (copy -> in-cache
        // FFT -> fused scale/scatter). A frame-wide batch adds ~3x the memory
        // traffic, and skipping the input copy changed nothing -- both measured
        // no faster in sim, so this stage is FFT-compute + scatter bound, not
        // copy bound.
        for (size_t i = 0; i < cfg_.ofdm.num_symbols; ++i) {
            const bool is_sync = (i == cfg_.ofdm.sync_pos);
            const int data_idx_int = is_sync ? -1 : _data_resource_layout.actual_symbol_to_data_symbol[i];
            std::complex<float>* __restrict__ dst = nullptr;
            if (is_sync) {
                dst = sync_symbol_freq.data();
            } else if (data_idx_int >= 0) {
                dst = symbols[static_cast<size_t>(data_idx_int)].data();
            } else if (i < _data_resource_layout.midframe_pilot_symbol_to_rank.size()) {
                const int pilot_rank = _data_resource_layout.midframe_pilot_symbol_to_rank[i];
                if (pilot_rank >= 0) {
                    dst = _midframe_symbol_freq_bufs[static_cast<size_t>(pilot_rank)].data();
                }
            }

            std::copy(frame.frame_data.begin() + pos + cfg_.ofdm.cp_length,
                     frame.frame_data.begin() + pos + cfg_.ofdm.cp_length + cfg_.ofdm.fft_size,
                     fft_input_.begin());

            fftwf_execute(fft_plan_);

            const std::complex<float>* __restrict__ src = fft_output_.data();
            if (sensing_enabled) {
                // Fuse the rx_symbols copy into the scale pass: write the scaled
                // sample to both the working slot and the sensing buffer in one
                // sweep, avoiding a separate full-symbol copy afterwards.
                std::complex<float>* __restrict__ rx = sense_frame.rx_symbols[i].data();
                #pragma omp simd
                for (size_t j = 0; j < scale_n; ++j) {
                    const std::complex<float> v = src[j] * scale;
                    if (dst != nullptr) {
                        dst[j] = v;
                    }
                    rx[j] = v;
                }
            } else if (dst != nullptr) {
                #pragma omp simd
                for (size_t j = 0; j < scale_n; ++j) {
                    dst[j] = src[j] * scale;
                }
            }
            pos += cfg_.ofdm.fft_size + cfg_.ofdm.cp_length;
        }
        prof_step_end = ProfileClock::now();
        prof_fft_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        // Calculate initial channel response H_est using ChannelEstimator
        prof_step_start = ProfileClock::now();
        AlignedVector& H_est = _h_est_buf;  // persistent; estimate_from_sync_lmmse resizes it
        float corrected_impulse_snr_linear_est = 1.0f;
        _channel_estimator.estimate_from_sync_lmmse(
            sync_symbol_freq,
            zc_freq_,
            H_est,
            cfg_.ofdm.cp_length,
            &corrected_impulse_snr_linear_est);
        //_channel_estimator.estimate_from_sync_ls(sync_symbol_freq, zc_freq_, H_est);

        _channel_anchor_symbols_buf.clear();
        _channel_anchor_h_bufs.clear();
        _channel_anchor_symbols_buf.push_back(static_cast<int>(cfg_.ofdm.sync_pos));
        _channel_anchor_h_bufs.push_back(H_est);
        if (!_midframe_pilot_seqs.empty()) {
            for (size_t p = 0; p < _data_resource_layout.midframe_pilot_symbols.size(); ++p) {
                const int actual_sym = _data_resource_layout.midframe_pilot_symbols[p];
                if (actual_sym < 0 || p >= _midframe_pilot_seqs.size() ||
                    p >= _midframe_symbol_freq_bufs.size()) {
                    continue;
                }
                _estimate_midframe_pilot_ls(
                    _midframe_symbol_freq_bufs[p],
                    _midframe_pilot_seqs[p],
                    _midframe_tmp_h_buf);
                const auto insert_at = std::lower_bound(
                    _channel_anchor_symbols_buf.begin(),
                    _channel_anchor_symbols_buf.end(),
                    actual_sym);
                const size_t offset = static_cast<size_t>(
                    std::distance(_channel_anchor_symbols_buf.begin(), insert_at));
                _channel_anchor_symbols_buf.insert(insert_at, actual_sym);
                _channel_anchor_h_bufs.insert(
                    _channel_anchor_h_bufs.begin() + static_cast<std::ptrdiff_t>(offset),
                    _midframe_tmp_h_buf);
            }
        }

        prof_step_end = ProfileClock::now();
        prof_channel_est_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        // Estimate frequency offset using FrequencyOffsetEstimator
        prof_step_start = ProfileClock::now();
        std::vector<int>& pilot_indices = _pilot_indices_buf;          // persistent scratch
        std::vector<float>& avg_phase_diff = _avg_phase_diff_buf;      // persistent scratch
        std::vector<float>& weights = _weights_buf;                    // persistent scratch
        bool cfo_sfo_estimate_valid = FrequencyOffsetEstimator::compute_pilot_phase_diff(
            symbols,
            cfg_.ofdm.pilot_positions,
            cfg_.ofdm.fft_size,
            cfg_.ofdm.sync_pos,
            pilot_indices,
            avg_phase_diff,
            weights,
            &_data_resource_layout.data_symbol_to_actual_symbol,
            &_cfo_symbol_skip_mask);
        
        // Phase unwrapping with SIMD optimization
        float beta = 0.0f;
        float alpha = 0.0f;
        if (cfo_sfo_estimate_valid) {
            unwrap(avg_phase_diff);
            std::tie(beta, alpha) = weightedlinearRegression(pilot_indices, avg_phase_diff, weights);
            cfo_sfo_estimate_valid = std::isfinite(beta) && std::isfinite(alpha);
        }
        prof_step_end = ProfileClock::now();
        prof_cfo_sfo_est_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        // Convert alpha to CFO
        float detected_freq_offset = FrequencyOffsetEstimator::alpha_to_cfo(
            alpha, cfg_.ofdm.fft_size, cfg_.ofdm.cp_length, cfg_.rf_sampling.sample_rate
        );
        const double tune_system_cfo_hz = rx_tune_system_cfo_hz(
            cfg_.downlink.center_freq,
            current_rx_tune_.actual_rf_freq,
            current_rx_tune_.actual_dsp_freq
        );
        const double clock_error_hz =
            static_cast<double>(detected_freq_offset) - tune_system_cfo_hz;
        const double raw_error_ppm =
            (std::abs(cfg_.downlink.center_freq) > 0.0)
                ? (clock_error_hz / cfg_.downlink.center_freq * 1e6)
                : 0.0;
        bool vofa_debug_valid = false;
        float vofa_raw_error_ppm = 0.0f;
        float vofa_filtered_error_ppm = 0.0f;
        const bool allow_reset = _control_time_gates.allow_reset(frame.usrp_time_ns);
        const bool allow_alignment = _control_time_gates.allow_alignment(frame.usrp_time_ns);
        const bool allow_freq_adjust = _control_time_gates.allow_freq_adjust(frame.usrp_time_ns);
        const bool allow_rx_gain_adjust = _control_time_gates.allow_rx_gain_adjust(frame.usrp_time_ns);
        const bool log_agc = cfg_.should_profile("agc");
        bool issued_alignment = false;
        bool issued_freq_adjust = false;
        bool issued_rx_gain_adjust = false;

        if (cfg_.sync_tracking.hardware_sync && cfo_sfo_estimate_valid) {
            const auto akf_result = _akf.update(raw_error_ppm);
            const double control_error_ppm = cfg_.sync_tracking.akf_enable
                ? akf_result.filtered_error_ppm
                : raw_error_ppm;
            vofa_debug_valid = true;
            vofa_raw_error_ppm = static_cast<float>(raw_error_ppm);
            vofa_filtered_error_ppm = static_cast<float>(akf_result.filtered_error_ppm);
            ++_ocxo_update_counter;
            if (_ocxo_update_counter >= 434) {
                _ocxo_update_counter = 0;
                const double applied_delta_ppm = _hw_sync->update_ocxo_pi_with_error_ppm(control_error_ppm);
                _akf.notify_control_action(applied_delta_ppm);
            }
        }

        // Frequency offset correction
        if (cfo_sfo_estimate_valid) {
            _freq_offset_sum += detected_freq_offset;
            _freq_offset_count++;
            if (_freq_offset_count>=434) {
                _avg_freq_offset = _freq_offset_sum / _freq_offset_count;
                _freq_offset_sum = 0.0f;
                _freq_offset_count = 0;
                if(abs(_avg_freq_offset) > 2.0f)
                {
                    if (cfg_.sync_tracking.software_sync && allow_freq_adjust){
                        LOG_RT_INFO() << "Adjusting RX frequency by: " << _avg_freq_offset << " Hz";
                        adjust_rx_freq(-_avg_freq_offset, false);
                        issued_freq_adjust = true;
                    }
                }
            }
        } else {
            LOG_RT_WARN_HZ(2) << "Skipping CFO/SFO update: no clean adjacent downlink pilot-symbol pairs";
        }
        if (issued_freq_adjust) {
            _control_time_gates.mark_freq_adjust_now(radio_time_now());
        }
        if (vofa_debug_valid && vofa_debug_sender_) {
            const std::array<float, 3> channels{
                vofa_raw_error_ppm,
                vofa_filtered_error_ppm,
                static_cast<float>(_avg_freq_offset)
            };
            vofa_debug_sender_->send_channels(channels);
        }

        prof_step_start = ProfileClock::now();
        double prof_eq_base_inv_us = 0.0;
        double prof_eq_channel_select_us = 0.0;
        double prof_eq_symbol_inv_us = 0.0;
        double prof_eq_pilot_phase_us = 0.0;
        double prof_eq_apply_us = 0.0;
        uint64_t prof_eq_data_symbols = 0;
        uint64_t prof_eq_midframe_channel_symbols = 0;
        uint64_t prof_eq_symbol_inv_count = 0;
        uint64_t prof_eq_pilot_phase_attempt = 0;
        uint64_t prof_eq_pilot_phase_success = 0;
        const double equalized_noise_var = noise_variance_from_snr_linear(
            std::max<double>(corrected_impulse_snr_linear_est, 1e-6));
        const double equalizer_noise_var_fallback =
            equalized_noise_var * _average_channel_power(H_est);
        const double equalizer_noise_var_est = (cfg_.downlink.equalizer.equalizer_mode == kEqualizerModeMmse)
            ? _estimate_equalizer_noise_var_from_pilots(
                symbols,
                H_est,
                alpha,
                beta,
                equalizer_noise_var_fallback)
            : 0.0;
        const float equalizer_noise_var = (cfg_.downlink.equalizer.equalizer_mode == kEqualizerModeMmse)
            ? static_cast<float>(equalizer_noise_var_est)
            : 0.0f;

        // Fine-grained equalization sub-timers add several ProfileClock::now()
        // calls per data symbol (hundreds per frame) to this hot loop. Gate them
        // behind the breakdown_eq switch so production runs (profiling off) pay
        // nothing; the coarse per-stage Equalization timer above still always runs.
        const auto eq_tick = [do_eq_breakdown]() {
            return do_eq_breakdown ? ProfileClock::now() : ProfileClock::time_point{};
        };
        const auto eq_accum = [do_eq_breakdown](double& acc,
                                                   ProfileClock::time_point t0,
                                                   ProfileClock::time_point t1) {
            if (do_eq_breakdown) {
                acc += std::chrono::duration<double, std::micro>(t1 - t0).count();
            }
        };

        // Pre-compute a fallback channel inverse. Mid-frame full-band pilots
        // provide additional time anchors and may replace this per symbol.
        static thread_local AlignedVector H_inv;
        static thread_local AlignedVector last_anchor_H_inv;
        bool last_anchor_H_inv_valid = false;
        auto prof_eq_sub_start = eq_tick();
        _compute_channel_inverse(H_est, H_inv, equalizer_noise_var);
        auto prof_eq_sub_end = eq_tick();
        eq_accum(prof_eq_base_inv_us, prof_eq_sub_start, prof_eq_sub_end);
        const bool use_symbol_tracking =
            cfg_.downlink.equalizer.channel_tracking_mode != kChannelTrackingModeOff &&
            !cfg_.ofdm.pilot_positions.empty();

        for (size_t i = 0; i < symbols.size(); ++i) {
            ++prof_eq_data_symbols;
            auto& symbol = symbols[i];
            const int actual_symbol = _data_resource_layout.data_symbol_to_actual_symbol[i];
            prof_eq_sub_start = eq_tick();
            const AlignedVector& symbol_H_base =
                _interpolated_channel_for_symbol(actual_symbol, H_est);
            prof_eq_sub_end = eq_tick();
            eq_accum(prof_eq_channel_select_us, prof_eq_sub_start, prof_eq_sub_end);

            const int relative_symbol_index = actual_symbol - static_cast<int>(cfg_.ofdm.sync_pos);
            const bool using_midframe_channel = (&symbol_H_base != &H_est);
            if (using_midframe_channel) {
                ++prof_eq_midframe_channel_symbols;
            }
            float phase_diff_CFO = using_midframe_channel ? 0.0f : (alpha * relative_symbol_index);
            float beta_rel = using_midframe_channel ? 0.0f : (beta * relative_symbol_index);
            const AlignedVector* symbol_H_inv = &H_inv;
            const bool using_last_anchor_channel =
                using_midframe_channel &&
                !_channel_anchor_h_bufs.empty() &&
                (&symbol_H_base == &_channel_anchor_h_bufs.back());
            if (&symbol_H_base != &H_est) {
                if (using_last_anchor_channel) {
                    if (!last_anchor_H_inv_valid) {
                        ++prof_eq_symbol_inv_count;
                        prof_eq_sub_start = eq_tick();
                        _compute_channel_inverse(symbol_H_base, last_anchor_H_inv, equalizer_noise_var);
                        prof_eq_sub_end = eq_tick();
                        eq_accum(prof_eq_symbol_inv_us, prof_eq_sub_start, prof_eq_sub_end);
                        last_anchor_H_inv_valid = true;
                    }
                    symbol_H_inv = &last_anchor_H_inv;
                } else {
                    ++prof_eq_symbol_inv_count;
                    prof_eq_sub_start = eq_tick();
                    _compute_channel_inverse(symbol_H_base, _tracking_h_inv_buf, equalizer_noise_var);
                    prof_eq_sub_end = eq_tick();
                    eq_accum(prof_eq_symbol_inv_us, prof_eq_sub_start, prof_eq_sub_end);
                    symbol_H_inv = &_tracking_h_inv_buf;
                }
            }

            if (use_symbol_tracking) {
                float tracked_beta = 0.0f;
                float tracked_alpha = 0.0f;
                ++prof_eq_pilot_phase_attempt;
                prof_eq_sub_start = eq_tick();
                if (_fit_symbol_pilot_phase(symbol, symbol_H_base, tracked_beta, tracked_alpha)) {
                    ++prof_eq_pilot_phase_success;
                    phase_diff_CFO = tracked_alpha;
                    beta_rel = tracked_beta;
                }
                prof_eq_sub_end = eq_tick();
                eq_accum(prof_eq_pilot_phase_us, prof_eq_sub_start, prof_eq_sub_end);
            }

            prof_eq_sub_start = eq_tick();
            ChannelEstimator::equalize_symbol(
                symbol,
                *symbol_H_inv,
                phase_diff_CFO,
                beta_rel,
                _actual_subcarrier_indices);
            prof_eq_sub_end = eq_tick();
            eq_accum(prof_eq_apply_us, prof_eq_sub_start, prof_eq_sub_end);
        }
        prof_step_end = ProfileClock::now();
        prof_equalization_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        prof_eq_base_inv_total += prof_eq_base_inv_us;
        prof_eq_channel_select_total += prof_eq_channel_select_us;
        prof_eq_symbol_inv_total += prof_eq_symbol_inv_us;
        prof_eq_pilot_phase_total += prof_eq_pilot_phase_us;
        prof_eq_apply_total += prof_eq_apply_us;
        prof_eq_data_symbols_total += prof_eq_data_symbols;
        prof_eq_midframe_channel_symbols_total += prof_eq_midframe_channel_symbols;
        prof_eq_symbol_inv_count_total += prof_eq_symbol_inv_count;
        prof_eq_pilot_phase_attempt_total += prof_eq_pilot_phase_attempt;
        prof_eq_pilot_phase_success_total += prof_eq_pilot_phase_success;

        // ================= SNR / LLR Scale Estimation =================
        prof_step_start = ProfileClock::now();
        _snr_linear = std::max<double>(corrected_impulse_snr_linear_est, 1e-6);
        _snr_db = 10.0 * std::log10(_snr_linear);
        const double llr_snr_linear = _update_llr_snr_filter(_snr_linear);
        _noise_var = std::max(noise_variance_from_snr_linear(llr_snr_linear), 1e-6);
        _llr_scale = 4.0 / _noise_var;
        prof_step_end = ProfileClock::now();
        prof_noise_est_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        // =====================================================================

        // Remodulate frequency domain symbols using QPSKModulator
        prof_step_start = ProfileClock::now();
        
        if (sensing_enabled) {
            for (size_t i = 0; i < cfg_.ofdm.num_symbols; ++i) {
                // Reserved sync symbols always use the original ZC sequence.
                if (is_zc_sync_symbol(cfg_, i)) {
                    sense_frame.tx_symbols[i] = zc_freq_;
                } else if (is_cfo_training_symbol(cfg_, i)) {
                    sense_frame.tx_symbols[i] = _cfo_training_seq;
                } else if (i < _data_resource_layout.midframe_pilot_symbol_to_rank.size() &&
                           _data_resource_layout.midframe_pilot_symbol_to_rank[i] >= 0) {
                    const size_t pilot_rank = static_cast<size_t>(
                        _data_resource_layout.midframe_pilot_symbol_to_rank[i]);
                    if (pilot_rank < _midframe_pilot_seqs.size()) {
                        sense_frame.tx_symbols[i] = _midframe_pilot_seqs[pilot_rank];
                    }
                } else {
                    // Data symbol remodulation is only needed for bistatic sensing.
                    const int symbol_idx_int = _data_resource_layout.actual_symbol_to_data_symbol[i];
                    if (symbol_idx_int < 0) {
                        continue;
                    }
                    const size_t symbol_idx = static_cast<size_t>(symbol_idx_int);
                    QPSKModulator::remodulate_symbol(
                        symbols[symbol_idx],
                        zc_freq_,
                        cfg_.ofdm.pilot_positions,
                        sense_frame.tx_symbols[i]
                    );
                    const size_t non_pilot_base = _data_resource_layout.non_pilot_offsets[symbol_idx];
                    for (size_t di = 0; di < _data_resource_layout.num_non_pilot_subcarriers; ++di) {
                        const size_t flat_idx = non_pilot_base + di;
                        if (_data_resource_layout.sensing_pilot_mask[flat_idx] == 0) {
                            continue;
                        }
                        const size_t k = static_cast<size_t>(_data_resource_layout.non_pilot_subcarrier_indices[di]);
                        sense_frame.tx_symbols[i][k] = sensing_pilot_freq_[k];
                    }
                }
            }
        }
        prof_step_end = ProfileClock::now();
        prof_remodulate_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        // Compute delay spectrum using DelayProcessor
        prof_step_start = ProfileClock::now();
        AlignedVector& delay_spectrum = _delay_spectrum_buf;  // persistent; resized inside
        _delay_processor.compute_delay_spectrum(H_est, delay_spectrum);
        
        // Find peak in delay spectrum (search within CP range)
        size_t max_index = 0;
        float max_mag = 0.0f;
        float average_mag = 0.0f;
        DelayProcessor::find_peak(delay_spectrum, max_index, max_mag, average_mag, cfg_.ofdm.cp_length);
        
        // Adjust delay index to signed value
        int adjusted_index = DelayProcessor::adjust_delay_index(max_index, cfg_.ofdm.fft_size);

        float fractional_delay = DelayProcessor::estimate_fractional_delay(delay_spectrum, max_index);
        float delay_offset_reading = adjusted_index + fractional_delay;
        prof_step_end = ProfileClock::now();
        prof_delay_spectrum_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        prof_step_start = ProfileClock::now();
        sfo_estimator.update(delay_offset_reading, frame.Alignment);
        auto _sfo_per_frame = sfo_estimator.get_sfo_per_frame();

        // auto sfo_via_estimator = _sfo_per_frame/((cfg_.ofdm.fft_size+cfg_.ofdm.cp_length)*cfg_.ofdm.num_symbols*cfg_.rf_sampling.sample_rate);
        // auto sfo_via_beta = -beta/((cfg_.ofdm.fft_size+cfg_.ofdm.cp_length)*cfg_.rf_sampling.sample_rate/cfg_.ofdm.fft_size*2*M_PI);
        // LOG_G_INFO() << "SFO via estimator: " << sfo_via_estimator << std::endl;
        // LOG_G_INFO() << "SFO via beta: " << sfo_via_beta << std::endl;
        // LOG_G_INFO() << "ratio:" << sfo_via_beta/sfo_via_estimator <<std::endl;
        if (_sfo_per_frame != 0.0f) {
            //LOG_G_INFO() << "SFO Frame: " << _sfo_per_frame << std::endl;
        }
        const size_t sync_symbol_len = cfg_.ofdm.fft_size + cfg_.ofdm.cp_length;
        const size_t sync_symbol_offset = cfg_.ofdm.sync_pos * sync_symbol_len;
        const std::complex<float>* sync_symbol_td = nullptr;
        size_t sync_symbol_td_count = 0;
        if (sync_symbol_offset + sync_symbol_len <= frame.frame_data.size()) {
            sync_symbol_td = frame.frame_data.data() + sync_symbol_offset;
            sync_symbol_td_count = sync_symbol_len;
        }
        auto delay_offset = sfo_estimator.get_sensing_delay_offset();
        int predictive_delay_samples = 0;
        if (cfg_.sync_tracking.predictive_delay) {
            predictive_delay_samples =
                _predictive_delay_samples_from_cfo(
                    cfg_,
                    frame.usrp_time_ns,
                    detected_freq_offset,
                    current_rx_tune_.actual_rf_freq,
                    current_rx_tune_.actual_dsp_freq,
                    time_spec_to_ns(radio_time_now()));
        }
        int delay_index_err =
            adjusted_index - cfg_.sync_tracking.desired_peak_pos + predictive_delay_samples;
        if (allow_rx_gain_adjust) {
            RxAgcAdjustment agc_adjustment;
            issued_rx_gain_adjust = _rx_agc.maybe_apply_from_delay_peak(
                max_mag,
                average_mag,
                sync_symbol_td,
                sync_symbol_td_count,
                frame.usrp_time_ns,
                _control_time_gates,
                [this](double gain_db) {
                    if (!_sim_radio) usrp_->set_rx_gain(gain_db, cfg_.downlink.rx_channel);
                },
                &agc_adjustment
            );
            if (issued_rx_gain_adjust && log_agc) {
                LOG_RT_INFO() << "RX AGC adjusted gain to " << agc_adjustment.next_gain_db
                              << " dB (delta=" << agc_adjustment.delta_db
                              << " dB, delay_peak=" << agc_adjustment.observed_peak
                              << ", delay_peak_db=" << agc_adjustment.observed_peak_db
                              << ", filtered_peak_db=" << agc_adjustment.filtered_peak_db
                              << ", peak_ratio=" << agc_adjustment.peak_ratio
                              << ", max_sync_component=" << agc_adjustment.max_sync_sample_component
                              << ", near_fs_count=" << agc_adjustment.near_full_scale_count
                              << ", hard_fs_count=" << agc_adjustment.hard_full_scale_count
                              << ", saturation=" << agc_adjustment.saturation_detected << ")";
            }
        }
        if((max_mag/average_mag < 20.0f || (abs(delay_index_err) > cfg_.sync_tracking.delay_adjust_step+5)) &&
            (cfg_.sync_tracking.software_sync || cfg_.sync_tracking.hardware_sync) && allow_reset) {
            _reset_count++;
            if (_reset_count >= _reset_hold_frames) {
                _reset_count = 0;
                LOG_RT_WARN() << "No valid delay found, resetting state.";
                adjust_rx_freq(0.0, true); // Reset frequency
                sfo_estimator.reset(); // Reset SFO estimator
                _akf.reset();
                _ocxo_update_counter = 0;
                if (vofa_debug_sender_) {
                    vofa_debug_sender_->reset_counter();
                }
                if (cfg_.sync_tracking.hardware_sync) {
                    _hw_sync->reset_frequency_control(); // Reset hardware frequency control
                    _hw_sync->reset_ocxo_pi_state();
                    LOG_RT_INFO() << "OCXO PI state reset to fast stage after sync reset.";
                }
                _enter_sync_search_state();
                _mute_uplink_tx_gain_for_sync_search();
                _control_time_gates.mark_reset_now(radio_time_now());
                return;
            }
        }
        else
        {
            _reset_count = 0;
        }
        if(allow_alignment && _sync_in_progress && frame.Alignment != 0)// Sync command has been executed
        {
            _sync_in_progress = false;
        }
        if(abs(delay_index_err) >= cfg_.sync_tracking.delay_adjust_step &&
            abs(delay_index_err) < cfg_.ofdm.cp_length  &&
            ( cfg_.sync_tracking.software_sync || cfg_.sync_tracking.hardware_sync ) &&
            !_sync_in_progress &&
            allow_alignment)
        {   
            if (_delay_adjustment_count++ >= 1) {
                _delay_adjustment_count = 0;
                // Residual timing trim while already locked: keep UL waveform enabled.
                _schedule_receive_alignment(static_cast<int32_t>(delay_index_err)); // Send sync command. May have delay due to FIFO
                _sync_in_progress = true;
                issued_alignment = true;
            }
            if (delay_index_err * _last_delay_index_err < 0){
                 // If symbol delay direction changes, it indicates jitter, reset counter
                _delay_adjustment_count = 0;
            }
        }
        else
        {
            _delay_adjustment_count = 0;
        }
        _last_delay_index_err = delay_index_err;
        prof_step_end = ProfileClock::now();
        prof_timing_sync_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        if (issued_alignment) {
            _control_time_gates.mark_alignment_now(radio_time_now());
        }
        if (issued_rx_gain_adjust) {
            _control_time_gates.mark_rx_gain_adjust_now(radio_time_now());
        }

        prof_step_start = ProfileClock::now();
        if (sensing_enabled) {
            sense_frame.CFO = alpha;
            if (_sfo_per_frame != 0.0f) {
                sense_frame.SFO = -_sfo_per_frame * (2 * M_PI) / (cfg_.ofdm.fft_size * cfg_.ofdm.num_symbols);
            } else {
                sense_frame.SFO = beta;
            }
            sense_frame.delay_offset = delay_offset + _user_delay_offset;
            sense_frame.generation = frame.generation;

            if (!spsc_wait_push(sensing_queue_, std::move(sense_frame), [this]() {
                    return !sensing_running_.load(std::memory_order_acquire);
                })) {
                _sensing_frame_pool.release(std::move(sense_frame));
            }
        }
        prof_step_end = ProfileClock::now();
        prof_sensing_queue_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        // Channel response data
        prof_step_start = ProfileClock::now();
        channel_sender_.add_data(std::move(H_est));

        // PDF data
        pdf_sender_.add_data(std::move(delay_spectrum));

        // Constellation data. The sender throttles the display to 50ms
        // (LatestOnly), while frames arrive far faster, so copying the full
        // last symbol every frame is wasted work. Gate the 8KB copy to every
        // Nth frame — still well above the display refresh rate.
        constexpr uint32_t kConstellationStride = 8;
        if ((_constellation_frame_counter++ % kConstellationStride) == 0) {
            const size_t last_idx = symbols.size() - 1;
            auto constellation_copy = symbols[last_idx];
            constellation_sender_.add_data(std::move(constellation_copy));
        }
        _publish_uplink_self_channel_debug(frame);
        prof_step_end = ProfileClock::now();
        prof_udp_send_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        // Extract selected payload RE to generate compact LLR.
        prof_step_start = ProfileClock::now();
        float scale_llr = std::min(static_cast<float>(_llr_scale * M_SQRT1_2), 500.0f);

        if (_measurement_enabled &&
            _measurement_active_epoch_id.load(std::memory_order_relaxed) != 0) {
            const FrameEvmStats evm = _compute_frame_evm_stats(symbols);
            _record_measurement_frame(evm);
        }

        if (_data_resource_layout.payload_re_count > 0) {
            if (_ldpc_fixed_point) {
                // Fused pow2 quantization: write int16 LLRs directly (no extra pass).
                const float scale_llr_q =
                    scale_llr * static_cast<float>(cfg_.ldpc.fixed_point_scale);
                LDPCCodec::AlignedShortVector frame_llr = _llr_pool_i16.acquire();
                int16_t* __restrict__ llr_ptr = frame_llr.data();
                for (size_t sym_idx = 0; sym_idx < symbols.size(); ++sym_idx) {
                    const auto* __restrict__ sym_ptr = symbols[sym_idx].data();
                    const size_t payload_begin = _data_resource_layout.payload_offsets[sym_idx];
                    const size_t payload_end = _data_resource_layout.payload_offsets[sym_idx + 1];
                    size_t llr_offset = payload_begin * 2;
                    for (size_t idx = payload_begin; idx < payload_end; ++idx) {
                        const size_t k = static_cast<size_t>(_payload_subcarrier_indices_flat[idx]);
                        llr_ptr[llr_offset++] = sat16_llr(sym_ptr[k].real() * scale_llr_q);
                        llr_ptr[llr_offset++] = sat16_llr(sym_ptr[k].imag() * scale_llr_q);
                    }
                }
                const int64_t demod_done_time_ns = do_latency_profile ? host_now_ns() : 0;
                if (!spsc_wait_push(_data_llr_buffer_i16, LlrFrameI16{
                        std::move(frame_llr),
                        frame.generation,
                        frame.host_enqueue_time_ns,
                        frame_dequeue_time_ns,
                        demod_done_time_ns,
                    }, [this]() {
                        return !_bit_processing_running.load(std::memory_order_acquire);
                    })) {
                    _llr_pool_i16.release(std::move(frame_llr));
                }
            } else {
                AlignedFloatVector frame_llr = _llr_pool.acquire();
                float* __restrict__ llr_ptr = frame_llr.data();
                for (size_t sym_idx = 0; sym_idx < symbols.size(); ++sym_idx) {
                    const auto* __restrict__ sym_ptr = symbols[sym_idx].data();
                    const size_t payload_begin = _data_resource_layout.payload_offsets[sym_idx];
                    const size_t payload_end = _data_resource_layout.payload_offsets[sym_idx + 1];
                    size_t llr_offset = payload_begin * 2;
                    for (size_t idx = payload_begin; idx < payload_end; ++idx) {
                        const size_t k = static_cast<size_t>(_payload_subcarrier_indices_flat[idx]);
                        llr_ptr[llr_offset++] = sym_ptr[k].real() * scale_llr;
                        llr_ptr[llr_offset++] = sym_ptr[k].imag() * scale_llr;
                    }
                }

                // Put LLR data into circular buffer
                const int64_t demod_done_time_ns = do_latency_profile ? host_now_ns() : 0;
                if (!spsc_wait_push(_data_llr_buffer, LlrFrame{
                        std::move(frame_llr),
                        frame.generation,
                        frame.host_enqueue_time_ns,
                        frame_dequeue_time_ns,
                        demod_done_time_ns,
                    }, [this]() {
                        return !_bit_processing_running.load(std::memory_order_acquire);
                    })) {
                    _llr_pool.release(std::move(frame_llr));
                }
            }
        }
        prof_step_end = ProfileClock::now();
        prof_llr_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        // ============== Profiling report ==============
        prof_frame_count++;
        if (prof_frame_count >= PROF_REPORT_INTERVAL && cfg_.should_profile("demodulation")) {
            double total = prof_fft_total + prof_channel_est_total + prof_cfo_sfo_est_total + 
                          prof_equalization_total + prof_noise_est_total + prof_remodulate_total + 
                          prof_delay_spectrum_total + prof_timing_sync_total + prof_sensing_queue_total + 
                          prof_udp_send_total + prof_llr_total;
            const double avg_eq_symbols =
                static_cast<double>(prof_eq_data_symbols_total) / static_cast<double>(prof_frame_count);
            const double avg_eq_midframe_symbols =
                static_cast<double>(prof_eq_midframe_channel_symbols_total) / static_cast<double>(prof_frame_count);
            const double avg_eq_symbol_inv_count =
                static_cast<double>(prof_eq_symbol_inv_count_total) / static_cast<double>(prof_frame_count);
            const double avg_eq_pilot_phase_attempt =
                static_cast<double>(prof_eq_pilot_phase_attempt_total) / static_cast<double>(prof_frame_count);
            const double avg_eq_pilot_phase_success =
                static_cast<double>(prof_eq_pilot_phase_success_total) / static_cast<double>(prof_frame_count);
            std::ostringstream oss;
            oss << "\n========== process_ofdm_frame Profiling (avg per frame, us) ==========\n"
                << "FFT (all symbols):    " << prof_fft_total / prof_frame_count << " us\n"
                << "Channel Estimation:   " << prof_channel_est_total / prof_frame_count << " us\n"
                << "CFO/SFO Estimation:   " << prof_cfo_sfo_est_total / prof_frame_count << " us\n"
                << "Equalization:         " << prof_equalization_total / prof_frame_count << " us\n";
            // The Eq sub-breakdown is only timed when breakdown_eq is enabled;
            // skip it otherwise (the per-symbol sub-timers were gated off, so the
            // buckets would just read 0 us).
            if (do_eq_breakdown) {
                oss << "  Eq base H_inv:      " << prof_eq_base_inv_total / prof_frame_count << " us\n"
                    << "  Eq channel select:  " << prof_eq_channel_select_total / prof_frame_count << " us"
                    << " (" << avg_eq_midframe_symbols << "/" << avg_eq_symbols << " midframe-H symbols)\n"
                    << "  Eq symbol H_inv:    " << prof_eq_symbol_inv_total / prof_frame_count << " us"
                    << " (" << avg_eq_symbol_inv_count << " calls/frame)\n"
                    << "  Eq pilot phase fit: " << prof_eq_pilot_phase_total / prof_frame_count << " us"
                    << " (" << avg_eq_pilot_phase_success << "/" << avg_eq_pilot_phase_attempt << " ok/frame)\n"
                    << "  Eq apply:           " << prof_eq_apply_total / prof_frame_count << " us\n";
            }
            oss << "Noise Estimation:     " << prof_noise_est_total / prof_frame_count << " us\n"
                << "Remodulation:         " << prof_remodulate_total / prof_frame_count << " us\n"
                << "Delay Spectrum:       " << prof_delay_spectrum_total / prof_frame_count << " us\n"
                << "Timing Sync:          " << prof_timing_sync_total / prof_frame_count << " us\n"
                << "Sensing Queue:        " << prof_sensing_queue_total / prof_frame_count << " us\n"
                << "UDP Send:             " << prof_udp_send_total / prof_frame_count << " us\n"
                << "LLR Calculation:      " << prof_llr_total / prof_frame_count << " us\n"
                << "TOTAL:                " << total / prof_frame_count << " us\n"
                << "======================================================================\n";
            LOG_RT_INFO() << oss.str();
            
            // Reset counters
            prof_fft_total = 0.0;
            prof_channel_est_total = 0.0;
            prof_cfo_sfo_est_total = 0.0;
            prof_equalization_total = 0.0;
            prof_eq_base_inv_total = 0.0;
            prof_eq_channel_select_total = 0.0;
            prof_eq_symbol_inv_total = 0.0;
            prof_eq_pilot_phase_total = 0.0;
            prof_eq_apply_total = 0.0;
            prof_eq_data_symbols_total = 0;
            prof_eq_midframe_channel_symbols_total = 0;
            prof_eq_symbol_inv_count_total = 0;
            prof_eq_pilot_phase_attempt_total = 0;
            prof_eq_pilot_phase_success_total = 0;
            prof_noise_est_total = 0.0;
            prof_remodulate_total = 0.0;
            prof_delay_spectrum_total = 0.0;
            prof_timing_sync_total = 0.0;
            prof_sensing_queue_total = 0.0;
            prof_udp_send_total = 0.0;
            prof_llr_total = 0.0;
            prof_frame_count = 0;
        }
    }

    /**
     * @brief Sensing Processing Thread.
     * 
     * Forwards demodulated bistatic sensing frames to SensingChannel.
     */
    void sensing_process_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        uhd::set_thread_priority_safe(1);
        bind_current_thread_from_downlink_hint(cfg_, 2);
        SPSCBackoff sensing_backoff;
        while (sensing_running_.load()) {
            SensingFrame frame;
            if (!sensing_queue_.try_pop(frame)) {
                sensing_backoff.pause();
                continue;
            }
            sensing_backoff.reset();
            if (frame.generation != _sync_generation.load(std::memory_order_acquire)) {
                _sensing_frame_pool.release(std::move(frame));
                continue;
            }
            if (_bistatic_sensing_channel) {
                const uint64_t frame_symbol_count = static_cast<uint64_t>(frame.rx_symbols.size());
                const uint64_t frame_start_symbol = _next_bistatic_frame_start_symbol.fetch_add(
                    frame_symbol_count,
                    std::memory_order_relaxed
                );
                _bistatic_sensing_channel->process_bistatic_frame(frame, frame_start_symbol);
            }
            _sensing_frame_pool.release(std::move(frame));
        }
    }
    
    /**
     * @brief Bit Processing Thread.
     * 
     * Handles the "Back-end" of the communication receiver:
     * 1. Soft Descrambling.
     * 2. LDPC Soft Decoding.
     * 3. Payload Extraction and Validation (Length check, CRC).
     * 4. UDP Output of decoded user data.
     */
    // Parse and decode all LDPC packets in one demodulated LLR frame. Templated
    // on the LLR sample type so the float and int16 (Q16) paths share one body;
    // the decode_frame() overload is selected by the payload vector type.
    template <typename LlrVec, typename ScratchVec>
    void process_payload_llr(const LlrVec& llr, ScratchVec& deint_scratch,
                             int64_t rx_enqueue_time_ns, int64_t process_dequeue_time_ns,
                             int64_t demod_done_time_ns, bool do_latency_profile,
                             std::vector<uint8_t>& expected_measurement_payload) {
        const size_t bits_per_block = _ldpc_decoder.get_N();
        const size_t bytes_per_ldpc_block = (_ldpc_decoder.get_K() + 7) / 8;
        if (bits_per_block != LdpcPacketFraming::kLdpcCodeBitsPerBlock ||
            bytes_per_ldpc_block != LdpcPacketFraming::kLdpcInfoBytesPerBlock) {
            LOG_G_WARN() << "[Demod] LDPC codec dimensions do not match unified framing.";
            return;
        }

        size_t symbol_offset = 0;
        bool latency_recorded = false;
        while ((symbol_offset + LdpcPacketFraming::kControlSymbols) * 2 <= llr.size()) {
            const size_t control_llr_offset = symbol_offset * 2;
            float marker_metric = 0.0f;
            if (!LdpcPacketFraming::detect_marker_llrs(
                    llr.data() + control_llr_offset, &marker_metric)) {
                break;
            }

            LdpcMiniHeader mini_header;
            if (!LdpcPacketFraming::decode_mini_header_llrs(
                    llr.data() + control_llr_offset + LdpcPacketFraming::kMarkerBits,
                    mini_header)) {
                LOG_G_WARN() << "[Demod] Mini-header CRC/version check failed; stop parsing frame.";
                break;
            }

            const size_t payload_blocks =
                LdpcPacketFraming::payload_blocks_for_len(mini_header.payload_len);
            const size_t required_llr = payload_blocks * bits_per_block;
            const size_t payload_llr_offset =
                control_llr_offset + LdpcPacketFraming::kControlBits;
            const size_t next_symbol_offset =
                symbol_offset + LdpcPacketFraming::packet_qpsk_symbols(payload_blocks);

            if (payload_llr_offset + required_llr > llr.size()) {
                LOG_G_WARN() << "[Demod] Mini-header requested payload beyond frame: blocks="
                             << payload_blocks << ", seq=" << mini_header.seq;
                break;
            }

            if (payload_blocks == 0) {
                symbol_offset = next_symbol_offset;
                continue;
            }

            LlrVec payload_llr(required_llr);
            std::copy(llr.begin() + payload_llr_offset,
                      llr.begin() + payload_llr_offset + required_llr,
                      payload_llr.begin());

            _bit_interleaver->deinterleave_inplace(payload_llr, deint_scratch);
            _descrambler.soft_descramble(payload_llr);

            LDPCCodec::AlignedByteVector decoded_payload;
            try {
                _ldpc_decoder.decode_frame(payload_llr, decoded_payload);
                if (decoded_payload.size() < mini_header.payload_len) {
                    LOG_G_WARN() << "[Demod] Decoded payload shorter than mini-header length.";
                    symbol_offset = next_symbol_offset;
                    continue;
                }

                std::vector<uint8_t> udp_data(
                    decoded_payload.begin(),
                    decoded_payload.begin() + mini_header.payload_len);

                bool handled_measurement_payload = false;
                if (_measurement_enabled) {
                    MeasurementPayloadMetadata meta;
                    if (parse_measurement_payload(udp_data.data(), udp_data.size(), meta)) {
                        handled_measurement_payload = true;
                        if (meta.epoch_id != 0) {
                            const uint32_t active_epoch =
                                _measurement_active_epoch_id.load(std::memory_order_relaxed);
                            if (active_epoch != 0 && meta.epoch_id == active_epoch) {
                                if (build_measurement_payload(expected_measurement_payload, meta)) {
                                    const auto compare = compare_measurement_payload(
                                        expected_measurement_payload,
                                        udp_data.data(),
                                        udp_data.size());
                                    _measurement_compared_bits.fetch_add(
                                        compare.compared_bits, std::memory_order_relaxed);
                                    _measurement_bit_errors.fetch_add(
                                        compare.bit_errors, std::memory_order_relaxed);
                                    if (compare.exact_match) {
                                        _measurement_successful_packets.fetch_add(
                                            1, std::memory_order_relaxed);
                                    }
                                    _measurement_epoch_tx_gain_x10.store(
                                        meta.tx_gain_x10, std::memory_order_relaxed);
                                } else {
                                    LOG_G_WARN() << "[Demod] Failed to rebuild expected measurement payload"
                                                 << " for epoch " << meta.epoch_id
                                                 << " seq " << meta.seq_in_epoch;
                                }
                            }
                        }
                    }
                }

                if (!handled_measurement_payload) {
                    // ARQ: classify feedback by mini-header flag, not payload sniffing
                    bool arq_consumed = false;
                    if (_arq_enabled) {
                        if (LdpcPacketFraming::is_arq_feedback(mini_header)) {
                            // This is an ARQ ACK from the BS for our UL packets
                            ArqFeedback ack;
                            if (ArqFeedback::try_unpack(udp_data.data(), udp_data.size(), ack)) {
                                if (_uplink_tx && _uplink_tx->arq_enabled()) {
                                    _uplink_tx->arq_tx_window().process_ack(ack, arq_now_ms());
                                }
                            }
                            arq_consumed = true; // never forward feedback to user UDP
                        } else {
                            // Data packet; process through DL RX ARQ window.
                            const bool accepted = _dl_arq_rx.process_received(
                                mini_header.seq, udp_data.data(), udp_data.size());
                            if (!accepted) {
                                arq_consumed = true; // duplicate, suppress
                            } else if (!cfg_.network_output.arq_ordered_delivery) {
                                // Unordered: forward immediately (fall through to send)
                            } else {
                                // Ordered: buffer, deliver later
                                arq_consumed = true;
                            }

                            // Generate ACK feedback and inject into UL TX
                            const int64_t now = arq_now_ms();
                            if (_dl_arq_rx.should_send_ack(now) && _uplink_tx) {
                                ArqFeedback fb = _dl_arq_rx.generate_ack();
                                _uplink_tx->inject_arq_feedback(fb);
                                _dl_arq_rx.mark_ack_sent(now);
                            }
                        }
                    }

                    if (!arq_consumed) {
                        _udp_output_sender->send(udp_data.data(), udp_data.size());
                    }

                    // For ordered delivery, flush any contiguous deliverable packets
                    if (_arq_enabled && cfg_.network_output.arq_ordered_delivery) {
                        std::vector<std::vector<uint8_t>> deliverable;
                        _dl_arq_rx.get_deliverable(deliverable);
                        for (auto& pkt : deliverable) {
                            _udp_output_sender->send(pkt.data(), pkt.size());
                        }
                    }
                    if (!latency_recorded &&
                        do_latency_profile &&
                        rx_enqueue_time_ns > 0 &&
                        process_dequeue_time_ns > 0 &&
                        demod_done_time_ns > 0) {
                        const int64_t udp_done_time_ns = host_now_ns();
                        _latency_accumulator.rx_queue_total_ns.fetch_add(
                            process_dequeue_time_ns - rx_enqueue_time_ns,
                            std::memory_order_relaxed);
                        _latency_accumulator.demod_total_ns.fetch_add(
                            demod_done_time_ns - process_dequeue_time_ns,
                            std::memory_order_relaxed);
                        _latency_accumulator.bit_total_ns.fetch_add(
                            udp_done_time_ns - demod_done_time_ns,
                            std::memory_order_relaxed);
                        _latency_accumulator.e2e_total_ns.fetch_add(
                            udp_done_time_ns - process_dequeue_time_ns,
                            std::memory_order_relaxed);
                        _latency_accumulator.count.fetch_add(1, std::memory_order_relaxed);
                        latency_recorded = true;
                    }
                }

            } catch (const std::exception& e) {
                LOG_G_WARN() << "[Demod] Payload LDPC decode failed: " << e.what();
            }
            symbol_offset = next_symbol_offset;
        }
    }

    void bit_processing_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        uhd::set_thread_priority_safe();
        bind_current_thread_from_downlink_hint(cfg_, 3);
        SPSCBackoff llr_backoff;
        const bool do_latency_profile =
            cfg_.should_profile("demodulation") && cfg_.should_profile("latency");
        const bool log_snr = cfg_.should_profile("snr");
        std::vector<uint8_t> expected_measurement_payload;
        size_t snr_print_counter = 0;

        auto maybe_log_snr = [&]() {
            if (log_snr && ((snr_print_counter++ & 0x3F) == 0)) {
                const double llr_snr_db = 10.0 * std::log10(std::max(1.0 / _noise_var, 1e-6));
                LOG_G_INFO() << "[LLR] SNR(dB): " << _snr_db
                             << " llr_snr(dB): " << llr_snr_db
                             << " noise_var: " << _noise_var
                             << " llr_scale: " << _llr_scale;
            }
        };

        while (_bit_processing_running.load()) {
            if (_ldpc_fixed_point) {
                LlrFrameI16 frame_llr;
                if (!_data_llr_buffer_i16.try_pop(frame_llr)) {
                    llr_backoff.pause();
                    continue;
                }
                llr_backoff.reset();
                if (frame_llr.llr.empty()) continue;
                if (frame_llr.generation != _sync_generation.load(std::memory_order_acquire)) {
                    _llr_pool_i16.release(std::move(frame_llr.llr));
                    continue;
                }
                maybe_log_snr();
                process_payload_llr(frame_llr.llr, _deinterleaver_llr_scratch_i16,
                                    frame_llr.rx_enqueue_time_ns, frame_llr.process_dequeue_time_ns,
                                    frame_llr.demod_done_time_ns, do_latency_profile,
                                    expected_measurement_payload);
                _llr_pool_i16.release(std::move(frame_llr.llr));
            } else {
                LlrFrame frame_llr;
                if (!_data_llr_buffer.try_pop(frame_llr)) {
                    llr_backoff.pause();
                    continue;
                }
                llr_backoff.reset();
                if (frame_llr.llr.empty()) continue;
                if (frame_llr.generation != _sync_generation.load(std::memory_order_acquire)) {
                    _llr_pool.release(std::move(frame_llr.llr));
                    continue;
                }
                maybe_log_snr();
                process_payload_llr(frame_llr.llr, _deinterleaver_llr_scratch,
                                    frame_llr.rx_enqueue_time_ns, frame_llr.process_dequeue_time_ns,
                                    frame_llr.demod_done_time_ns, do_latency_profile,
                                    expected_measurement_payload);
                _llr_pool.release(std::move(frame_llr.llr));
            }
        }
    }
};

std::atomic<bool> stop_signal(false);

void signal_handler(int) {
    stop_signal.store(true);
}

int UHD_SAFE_MAIN(int argc, char*[]) {
    async_logger::AsyncLoggerGuard async_logger_guard;
    std::signal(SIGINT, &signal_handler);
    const std::string default_config_file = "UE.yaml";
    Config cfg = make_default_ue_config();

    if (argc > 1) {
        LOG_G_ERROR() << "CLI parameters are no longer supported. Please configure UE via "
                      << default_config_file << ".";
        return 1;
    }

    if (!path_exists(default_config_file)) {
        LOG_G_ERROR() << "Config file '" << default_config_file
                      << "' not found. Copy a sample file from the repository config directory, "
                      << "such as 'UE_X310.yaml' or 'UE_B210.yaml', to '" << default_config_file
                      << "' and edit it before starting UE.";
        return 1;
    }

    if (!load_ue_config_from_yaml(cfg, default_config_file)) {
        return 1;
    }

    LOG_G_INFO() << "Loaded config from: " << default_config_file;
    finalize_ue_network_defaults(cfg);
    log_ue_sync_mode(cfg);
    log_ue_agc_mode(cfg);
    uhd::set_thread_priority_safe(1, true);
    // Use last available core for main thread
    bind_current_thread_to_main_core(cfg);
    
    // Load FFTW wisdom
    FFTWManager::import_wisdom();

    UEEngine receiver(cfg);
    receiver.start();
    
    while (!stop_signal.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    receiver.stop();
    
    // Save FFTW wisdom
    FFTWManager::export_wisdom();
    
    return 0;
}
