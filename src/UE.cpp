#include <uhd/utils/safe_main.hpp>  // UHD_SAFE_MAIN entry macro only
#include <algorithm>
#include <array>
#include <complex>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <fftw3.h>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <cstdint>
#include <cstring>
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
#include "ErtmAbsoluteFrameState.hpp"
#include "PpsTimeAlignment.hpp"
#include "SensingChannel.hpp"
#include "RadioBackend.hpp"

namespace {
inline int64_t host_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

const char* tx_async_event_code_to_string(radio::AsyncEvent event_code)
{
    switch (event_code) {
    case radio::AsyncEvent::BurstAck:
        return "BURST_ACK";
    case radio::AsyncEvent::Underflow:
        return "UNDERFLOW";
    case radio::AsyncEvent::SeqError:
        return "SEQ_ERROR";
    case radio::AsyncEvent::TimeError:
        return "TIME_ERROR";
    case radio::AsyncEvent::UnderflowInPacket:
        return "UNDERFLOW_IN_PACKET";
    case radio::AsyncEvent::SeqErrorInBurst:
        return "SEQ_ERROR_IN_BURST";
    case radio::AsyncEvent::UserPayload:
        return "USER_PAYLOAD";
    default:
        return "UNKNOWN";
    }
}

const char* rx_error_code_to_string(radio::RxError error_code)
{
    switch (error_code) {
    case radio::RxError::None:
        return "NONE";
    case radio::RxError::Timeout:
        return "TIMEOUT";
    case radio::RxError::LateCommand:
        return "LATE_COMMAND";
    case radio::RxError::BrokenChain:
        return "BROKEN_CHAIN";
    case radio::RxError::Overflow:
        return "OVERFLOW";
    case radio::RxError::Alignment:
        return "ALIGNMENT";
    case radio::RxError::BadPacket:
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
              batch.generation = 0;
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
          }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly,
             "UE channel debug sender"),
          pdf_sender_(2, [this](const auto& data) {
              pdf_pub_->send_container(data);
          }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly,
             "UE delay-spectrum debug sender"),
          constellation_sender_(10, [this](const auto& data) {
              constellation_pub_->send_container(data);
          }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly,
             "UE constellation debug sender"),
          uplink_self_channel_sender_(2, [this](const auto& data) {
              uplink_self_channel_pub_->send_container(data);
          }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly,
             "UE uplink self-channel debug sender"),
          uplink_self_pdf_sender_(2, [this](const auto& data) {
              uplink_self_pdf_pub_->send_container(data);
          }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly,
             "UE uplink self-spectrum debug sender"),
          uplink_self_scan_sender_(2, [this](const auto& data) {
              if (!uplink_self_scan_pub_ || data.empty()) {
                  return;
              }
              uplink_self_scan_pub_->send(data.data(), data.size());
          }, std::chrono::milliseconds(50), DataSender<uint8_t>::DeliveryMode::LatestOnly,
             "UE uplink self-scan debug sender"),
          ertm_debug_sender_(2, [this](const auto& frames) {
              if (!ertm_debug_pub_ || frames.empty()) {
                  return;
              }
              const auto parts = make_ertm_debug_msg_parts(frames.back());
              ertm_debug_pub_->send_frame(parts);
          }, std::chrono::milliseconds(50),
             DataSender<ErtmDebugFrameData>::DeliveryMode::LatestOnly,
             "UE eRTM debug sender"),
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
              frame.alignment_samples = 0;
              frame.ertm_rx_alignment_shift_samples = 0;
              frame.force_ertm_remeasure = false;
              if (cfg.uplink.ertm_to_enable) {
                  frame.ertm_downlink_channel_freq.resize(cfg.ofdm.fft_size);
              }
              return frame;
          }),
          _rx_agc(cfg),
          _sync_search_gain_sweep(cfg),
          _reset_hold_frames(reset_hold_frames_from_cfg(cfg)),
          _cfo_ewma(make_cfo_ewma_params(cfg)) {
        _init_cpu_ldpc_workers();
        _bit_interleaver = std::make_unique<BitBlockInterleaver>(
            _cpu_ldpc_contexts.front()->decoder.get_N(), 21);
        _measurement_active_epoch_id.store(0, std::memory_order_relaxed);
        if (_measurement_enabled && !cfg_.measurement.measurement_output_dir.empty()) {
            _measurement_summary_path = cfg_.measurement.measurement_output_dir + "/ue_measurement_summary.csv";
        }
        _build_cfo_symbol_skip_mask();
        _build_compact_payload_indices();
        LOG_G_INFO_M(Config) << "Payload resource grid: " << _data_resource_layout.payload_re_count
                     << " payload RE out of " << _data_resource_layout.non_pilot_re_count
                     << " non-sync/non-pilot RE per frame"
                     << (cfg_.resource_preview.data_resource_blocks_configured ? " (configured blocks)." : " (legacy full-grid mode).");
        if (_data_resource_layout.sensing_pilot_re_count > 0) {
            LOG_G_INFO_M(Sensing) << "Sensing-pilot sequence uses alternate ZC root "
                         << _sensing_pilot_zc_root
                         << " (sync root=" << cfg_.ofdm.zc_root << ").";
        }

        init_radio();
        _configure_ue_air_timeline();
        init_filter();
        prepare_fftw();
        _init_cpu_demod_workers();
        init_zmq_publishers();
        if (cfg_.sync_tracking.hardware_sync) {
            const double loop_max_slew_ppm_per_s = std::max({
                cfg_.sync_tracking.ocxo_coarse_max_slew_ppm_per_s,
                cfg_.sync_tracking.ocxo_fll_fast_max_slew_ppm_per_s,
                cfg_.sync_tracking.ocxo_fll_slow_max_slew_ppm_per_s,
                cfg_.sync_tracking.ocxo_phase_max_slew_ppm_per_s});
            if (cfg_.sync_tracking.ocxo_actuator_max_slew_ppm_per_s + 1e-15 <
                loop_max_slew_ppm_per_s) {
                LOG_G_WARN_M(Ocxo)
                    << "OCXO actuator slew is below a loop slew; desired steps will be clipped"
                    << " actuator=" << cfg_.sync_tracking.ocxo_actuator_max_slew_ppm_per_s
                    << " loop_max=" << loop_max_slew_ppm_per_s;
            }
            _hw_sync = std::make_unique<HardwareSyncController>(cfg_.sync_tracking.hardware_sync_tty);
            _hw_sync->configure_plant(
                cfg_.sync_tracking.ocxo_dac_ppm_per_lsb,
                cfg_.sync_tracking.ocxo_dac_zero_word,
                cfg_.sync_tracking.ocxo_max_abs_ppm,
                cfg_.sync_tracking.ocxo_actuator_max_slew_ppm_per_s,
                cfg_.sync_tracking.ocxo_dac_state_file);
            _hw_sync->configure_pps_phase_adjust_timeout(
                cfg_.sync_tracking.ocxo_pps_initial_ap_command_timeout_s);
            _ocxo_discipliner = std::make_unique<OCXODiscipliner>(make_ocxo_params(cfg_));
            _ocxo_discipliner->reset_tracking_at_command(
                _hw_sync->current_offset_ppm());
            _ocxo_pps_initial_aligner = std::make_unique<OCXOPpsInitialAligner>(
                make_ocxo_pps_initial_aligner_params(cfg_));
            LOG_G_INFO_M(Ocxo) << "OCXO staged discipliner enabled"
                         << " (mode=" << cfg_.sync_tracking.ocxo_mode
                         << ", fast_control="
                         << cfg_.sync_tracking.ocxo_fll_fast_control_mode
                         << ", fast_period_s="
                         << cfg_.sync_tracking.ocxo_fll_fast_control_period_s
                         << ", phase_entry_ap="
                         << (cfg_.sync_tracking.ocxo_mode == kOCXOModePpsPhase
                                 ? (cfg_.sync_tracking.ocxo_pps_initial_ap_enable
                                        ? "enabled" : "disabled")
                                 : "inactive")
                         << ", ap_polarity="
                         << (cfg_.sync_tracking.ocxo_pps_initial_ap_polarity < 0 ? -1 : 1)
                         << ")";
        }
        // Initialize sensing processing
        init_sensing();
        // Initialize data processing
        init_data_processing();
        _prefill_pools_for_topology();
        _register_commands();
    }

    ~UEEngine() {
        stop();
        
        
        if (_ertm_os_ifft_plan) fftwf_destroy_plan(_ertm_os_ifft_plan);
        if (_ertm_corr_fft_a_plan) fftwf_destroy_plan(_ertm_corr_fft_a_plan);
        if (_ertm_corr_fft_b_plan) fftwf_destroy_plan(_ertm_corr_fft_b_plan);
        if (_ertm_corr_ifft_plan) fftwf_destroy_plan(_ertm_corr_ifft_plan);
        // Note: _channel_estimator, _delay_processor
        // manage their own FFT plans and clean them up in their destructors
    }

    void _configure_ue_air_timeline() {
        double rx_rate = cfg_.rf_sampling.sample_rate;
        if (dev_) {
            try {
                const double actual = dev_->get_rx_rate(cfg_.downlink.rx_channel);
                if (actual > 0.0) {
                    rx_rate = actual;
                }
            } catch (...) {
            }
        }
        _ue_rx_sample_rate = rx_rate;
        _ue_air_timeline.configure(
            cfg_.samples_per_frame(),
            cfg_.ofdm.num_symbols,
            _ue_rx_sample_rate);
        LOG_G_INFO_M(AirFrame) << "[AirFrame] UE timeline configured: samples_per_frame="
                     << cfg_.samples_per_frame()
                     << ", sample_rate=" << _ue_rx_sample_rate
                     << (std::abs(_ue_rx_sample_rate - cfg_.rf_sampling.sample_rate) > 1.0
                             ? " (coerced from config)"
                             : "");
    }

    double _rx_rate_hz() const {
        return (_ue_rx_sample_rate > 0.0) ? _ue_rx_sample_rate : cfg_.rf_sampling.sample_rate;
    }

    /**
     * @brief Stamp absolute air-frame identity + UE-USRP frame-start on an RX frame.
     *
     * @param raw_first_ns  Metadata time of the first sample of the RX read
     * @param alignment_sample_adjust  Samples from raw first to committed boundary
     *        (alignment frame: discard amount; normal: 0)
     */
    void _stamp_rx_air_frame(RxFrame& frame, int64_t raw_first_ns, int64_t alignment_sample_adjust) {
        // Apply any SIB anchors published by the bit thread before deriving seq.
        _drain_sib_events_on_rx();

        // Always use the same rate as timeline configure (actual UHD rate).
        const double rate = _rx_rate_hz();
        const int64_t boundary_ns = UeAirFrameTimeline::boundary_time_ns(
            raw_first_ns, alignment_sample_adjust, rate);
        // Store the committed boundary as usrp_time_ns so downstream control
        // gates and eRTM share the same frame-start reference.
        if (boundary_ns >= 0) {
            frame.usrp_time_ns = boundary_ns;
        }
        // Derive absolute seq from (SIB seq, SIB boundary) + time delta — never
        // from an async free-run counter reset by the bit-processing thread.
        frame.identity = _ue_air_timeline.stamp_rx_boundary(boundary_ns);
        _ue_air_frame_mid_time_ns.store(
            frame.identity.ue_frame_mid_time_ns, std::memory_order_relaxed);
        _ue_air_frame_start_time_ns.store(
            frame.identity.ue_frame_start_time_ns, std::memory_order_relaxed);
        if (frame.identity.valid) {
            // Single-writer snapshot for diagnostics (RX thread only).
            _air_identity_rx_snapshot = frame.identity;
            _next_bistatic_frame_start_symbol.store(
                frame.identity.frame_start_symbol, std::memory_order_relaxed);
            const uint64_t n =
                _air_identity_stamp_count.fetch_add(1, std::memory_order_relaxed) + 1;
            // Heartbeat: first 8 stamps, then every 512 frames.
            if (n <= 8 || (n % 512) == 0) {
                LOG_RT_INFO_M(AirFrame)
                    << "[AirFrame] RX stamp"
                    << ": n=" << n
                    << ", epoch=0x" << std::hex << frame.identity.timeline_epoch
                    << std::dec
                    << ", seq=" << frame.identity.frame_seq
                    << ", boundary_ns=" << frame.identity.ue_frame_start_time_ns
                    << ", mid_ns=" << frame.identity.ue_frame_mid_time_ns
                    << ", raw_first_ns=" << raw_first_ns
                    << ", align_adj=" << alignment_sample_adjust
                    << ", locks=" << _air_identity_lock_count.load(std::memory_order_relaxed)
                    << ", sfo_resets="
                    << _air_timeline_reset_count.load(std::memory_order_relaxed);
            }
        }
        if (_uplink_tx) {
            _uplink_tx->store_air_frame_stamp(frame.identity);
        }
    }

    /**
     * @brief RX-owned: drain Identity-SIB events from bit path and install anchors.
     * Protocol resets (ARQ/SFO) are requested via flags for the process thread.
     */
    void _drain_sib_events_on_rx() {
        AirFrameSibEvent ev;
        while (_sib_event_queue.try_pop(ev)) {
            const uint64_t live_gen =
                _sync_generation.load(std::memory_order_acquire);
            if (ev.sync_generation != live_gen) {
                const uint64_t n =
                    _air_sib_stale_drop_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (n <= 16 || (n % 64) == 0) {
                    LOG_RT_WARN_M(AirFrame)
                        << "[AirFrame] drop stale SIB event"
                        << ": event_gen=" << ev.sync_generation
                        << ", live_gen=" << live_gen
                        << ", epoch=0x" << std::hex << ev.timeline_epoch << std::dec
                        << ", seq=" << ev.frame_seq
                        << ", drop_count=" << n;
                }
                continue; // stale across hard resync
            }
            const bool apply_epoch_changed = _ue_air_timeline.apply_identity_sib_anchor(
                ev.timeline_epoch,
                ev.frame_seq,
                ev.ue_boundary_time_ns,
                ev.ue_mid_time_ns);
            // Sticky bit-thread flag survives lost/stale first SIB events: after
            // last_seen advances, later SIBs have epoch_changed=0, but SFO must
            // still run once on the first successful post-reboot anchor install.
            const bool sticky_epoch_sfo =
                _ota_epoch_sfo_needed.load(std::memory_order_acquire);
            const bool need_sfo_reset =
                apply_epoch_changed || ev.epoch_changed || sticky_epoch_sfo;
            const uint64_t locks =
                _air_identity_lock_count.fetch_add(1, std::memory_order_relaxed) + 1;
            // Heartbeat on anchor install (first 16 + every 256).
            if (locks <= 16 || (locks % 256) == 0 || need_sfo_reset) {
                LOG_RT_INFO_M(AirFrame)
                    << "[AirFrame] SIB anchor installed on RX"
                    << ": locks=" << locks
                    << ", epoch=0x" << std::hex << ev.timeline_epoch << std::dec
                    << ", seq=" << ev.frame_seq
                    << ", boundary_ns=" << ev.ue_boundary_time_ns
                    << ", mid_ns=" << ev.ue_mid_time_ns
                    << ", gen=" << ev.sync_generation
                    << ", apply_epoch_changed=" << apply_epoch_changed
                    << ", event_epoch_changed=" << ev.epoch_changed
                    << ", sticky_sfo=" << sticky_epoch_sfo;
            }
            // Only RX (after successful anchor install) requests SFO/sync reset.
            // Tag with *live* generation so process applies under the current
            // sync epoch even if the SIB event was published slightly earlier.
            if (need_sfo_reset) {
                _ota_epoch_sfo_needed.store(false, std::memory_order_release);
                _pending_sfo_reset_generation.store(
                    live_gen, std::memory_order_release);
                // Belt-and-suspenders: if bit-thread clear was lost or this is a
                // sticky install after a missed event, re-request eRTM wipe.
                if (apply_epoch_changed || ev.epoch_changed) {
                    _request_ertm_report_clear("ota_epoch_change_rx_anchor");
                }
                LOG_G_WARN_M(AirFrame) << "[AirFrame] BS timeline epoch change on RX"
                             << ": epoch=" << ev.timeline_epoch
                             << ", seq=" << ev.frame_seq
                             << ", live_gen=" << live_gen
                             << ", event_gen=" << ev.sync_generation
                             << ", apply_epoch_changed=" << apply_epoch_changed
                             << ", event_epoch_changed=" << ev.epoch_changed
                             << ", sticky_sfo=" << sticky_epoch_sfo;
            }
        }
    }

    void start() {
        _ue_timing_advance.store(cfg_.uplink.ue_timing_advance, std::memory_order_relaxed);
        log_duplex_summary(cfg_, "UE");

        radio::TimeSpec stream_start_time(0.0);
        if (_uplink_tx && dev_->supports(radio::Capability::TimedTx)) {
            stream_start_time = _next_timed_stream_start();
            const double tx_rate = dev_->get_tx_rate(cfg_.uplink.tx_channel);
            const double rx_rate = _rx_rate_hz();
            if (std::abs(tx_rate - rx_rate) > 1.0) {
                throw std::runtime_error(
                    "UE timed RX/UL-TX requires identical actual sample rates "
                    "for exact identity mapping: rx_rate=" +
                    std::to_string(rx_rate) + ", tx_rate=" +
                    std::to_string(tx_rate) + ", samples_per_period=" +
                    std::to_string(cfg_.samples_per_frame()));
            }
            _uplink_tx->set_timed_tx(
                stream_start_time,
                dev_->master_clock_rate(),
                tx_rate);
            LOG_G_INFO_M(Radio) << "[UE] timed RX/UL-TX stream start at "
                         << stream_start_time.get_real_secs()
                         << " s on the shared radio clock";
        }

        _control_handler.start();
        running_.store(true);
        rx_thread_ = std::thread(&UEEngine::rx_proc, this, stream_start_time);
        if (_uplink_tx) {
            _uplink_tx->start();
            if (dev_->supports(radio::Capability::AsyncTxEvents)) {
                _tx_async_exit_requested.store(false, std::memory_order_relaxed);
                _tx_async_thread = std::thread(&UEEngine::_tx_async_event_proc, this);
            }
        }
        // Start data processing thread
        _bit_processing_running.store(true);
        _bit_processing_thread = std::thread(&UEEngine::bit_processing_proc, this);

        process_thread_ = std::thread(&UEEngine::process_proc, this);
        
        // Start all senders
        channel_sender_.start();
        pdf_sender_.start();
        constellation_sender_.start();
        if (uplink_self_channel_debug_enabled(cfg_) && _uplink_tx) {
            uplink_self_channel_sender_.start();
            uplink_self_pdf_sender_.start();
            if (uplink_self_scan_spectrum_enabled(cfg_)) {
                uplink_self_scan_sender_.start();
            }
        }
        if (cfg_.uplink.ertm_debug_output_enabled) {
            ertm_debug_sender_.start();
        }
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
        uplink_self_scan_sender_.stop();
        ertm_debug_sender_.stop();
        _control_handler.stop();
    }

private:
private:
    static LDPCCodec::LDPCConfig ldpc_cfg_from_config(const Config& cfg) {
        return make_ldpc_5041008_cfg(cfg.ldpc);
    }

    static inline int16_t sat16_llr(float v) {
        long q = std::lroundf(v);
        if (q > 32767) q = 32767;
        else if (q < -32767) q = -32767;
        return static_cast<int16_t>(q);
    }

    static CfoErrorEwma::Params make_cfo_ewma_params(const Config& cfg) {
        CfoErrorEwma::Params p;
        p.enable = cfg.sync_tracking.cfo_ewma_enable;
        p.alpha = cfg.sync_tracking.cfo_ewma_alpha;
        return p;
    }

    static OCXODiscipliner::Params make_ocxo_params(const Config& cfg) {
        return make_ocxo_discipliner_params(cfg);
    }

    enum class RxState { SYNC_SEARCH, ALIGNMENT, NORMAL };

    Config cfg_;
    const bool _measurement_enabled;
    const DuplexFrameLayout _duplex_layout;
    const DataResourceGridLayout _data_resource_layout;
    std::vector<uint8_t> _cfo_symbol_skip_mask;
    std::vector<int> _payload_subcarrier_indices_flat;
    radio::IDevicePtr dev_;        // RX (+ uplink TX) radio device, any backend
    radio::IRxStreamPtr rx_stream_;
    std::unique_ptr<UplinkTxEngine> _uplink_tx;  // non-null when duplex uplink enabled
    std::thread _tx_async_thread;
    std::atomic<bool> _tx_async_exit_requested{false};
    std::atomic<bool> _stream_restart_requested{false};
    std::atomic<uint64_t> _stream_restart_count{0};
    std::atomic<uint64_t> _stream_restart_pending_epoch{0};
    // Last timed stream-start (seconds) handed out by _next_timed_stream_start().
    // Used to keep consecutive restart start times strictly increasing so two
    // restarts inside the same lead window can't collide on the same timestamp.
    double _last_scheduled_stream_start_s = -1.0;
    // Host-time of the last restart request triggered by a TX async
    // timing/sequence error, to rate-limit the storm a single stale/colliding
    // burst produces (thousands of TIME_ERRORs microseconds apart).
    std::atomic<int64_t> _last_async_restart_request_ns{-1};
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
    radio::TimeSpec radio_time_now() const {
        return dev_->time_now();
    }

    radio::TimeSpec _next_timed_stream_start() {
        if (!dev_->supports(radio::Capability::TimedTx)) {
            return radio::TimeSpec(0.0);
        }
        constexpr double kStartLeadTimeSec = 1.0;
        double scheduled_start_s =
            std::ceil(dev_->time_now().get_real_secs() + kStartLeadTimeSec);
        // Serialize restart start times. The start is quantized to whole
        // seconds, so several restarts landing in the same lead window (e.g.
        // back-to-back underflows) would otherwise all pick the same timestamp;
        // the later burst then collides with the earlier one on the radio and
        // the USRP rejects it with a TIME_ERROR, which triggers yet another
        // restart -- a self-sustaining storm. Force each start strictly after
        // the previous one so restarts serialize cleanly instead.
        if (_last_scheduled_stream_start_s > 0.0 &&
            scheduled_start_s <= _last_scheduled_stream_start_s) {
            scheduled_start_s = _last_scheduled_stream_start_s + kStartLeadTimeSec;
        }
        _last_scheduled_stream_start_s = scheduled_start_s;
        return radio::TimeSpec(scheduled_start_s);
    }

    void _request_stream_restart(const char* reason) {
        if (!dev_->supports(radio::Capability::StreamRestart) || !_uplink_tx) {
            if (LOG_MOD_ON(Recovery)) {
                LOG_G_WARN_M(Recovery) << "[UE recovery] restart_request_ignored reason=" << reason
                             << ", stream_restart_supported="
                             << dev_->supports(radio::Capability::StreamRestart)
                             << ", has_uplink_tx=" << static_cast<bool>(_uplink_tx);
            }
            return;
        }
        const uint64_t restart_count =
            _stream_restart_count.fetch_add(1, std::memory_order_relaxed) + 1;
        _stream_restart_pending_epoch.store(restart_count, std::memory_order_release);
        _stream_restart_requested.store(true, std::memory_order_release);
        LOG_G_WARN_M(Recovery) << "[UE] requested shared RX/UL-TX stream restart: " << reason;
        if (LOG_MOD_ON(Recovery)) {
            LOG_G_WARN_M(Recovery) << "[UE recovery] restart_requested reason=" << reason
                         << ", restart_epoch=" << restart_count
                         << ", state=" << static_cast<int>(state_.load(std::memory_order_relaxed))
                         << ", sync_generation=" << _sync_generation.load(std::memory_order_relaxed)
                         << ", pending_alignment_id="
                         << _pending_alignment_id.load(std::memory_order_relaxed)
                         << ", discard_samples=" << discard_samples_.load(std::memory_order_relaxed)
                         << ", sync_offset=" << sync_offset_.load(std::memory_order_relaxed)
                         << ", ul_tx_rx_shift="
                         << _uplink_tx_rx_alignment_shift.load(std::memory_order_relaxed)
                         << ", ul_tx_errors="
                         << (_uplink_tx ? _uplink_tx->tx_error_count().load(std::memory_order_relaxed) : 0)
                         << ", tx_async_errors="
                         << _tx_async_error_count.load(std::memory_order_relaxed)
                         << ", rx_overflows="
                         << _rx_overflow_count.load(std::memory_order_relaxed);
        }
    }

    void _reset_receive_state_after_stream_restart(uint64_t restart_epoch) {
        // RX-thread safe portion only. process-owned tracking (latch, SFO,
        // OCXO age) is reset via _process_tracking_reset_req on the process
        // thread — writing those fields here is a data race.
        const uint64_t next_generation =
            _sync_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
        _recovery_queue_clear_requested.store(true, std::memory_order_release);
        sync_offset_.store(0, std::memory_order_relaxed);
        discard_samples_.store(0, std::memory_order_relaxed);
        _reset_uplink_tx_rx_alignment_shift(restart_epoch);
        _pending_alignment_id.store(0, std::memory_order_relaxed);
        _alignment_applied_id.store(0, std::memory_order_relaxed);
        _alignment_applied_samples.store(0, std::memory_order_relaxed);
        _alignment_applied_time_ns.store(-1, std::memory_order_relaxed);
        _pending_alignment_restart_epoch.store(restart_epoch, std::memory_order_relaxed);
        _set_uplink_waveform_enabled(false);
        _process_tracking_reset_req.fetch_add(1, std::memory_order_acq_rel);
        _clear_air_frame_identity("stream_restart");
        state_ = RxState::SYNC_SEARCH;
        if (LOG_MOD_ON(Recovery)) {
            LOG_G_WARN_M(Recovery) << "[UE recovery] reset_receive_state generation="
                         << next_generation
                         << ", restart_epoch=" << restart_epoch
                         << ", ertm_measurement_available="
                         << _ertm_state.snapshot().available
                         << ", ul_tx_rx_shift="
                         << _uplink_tx_rx_alignment_shift.load(std::memory_order_relaxed)
                         << ", waveform_enabled="
                         << (_uplink_tx ? _uplink_tx->waveform_enabled() : false);
        }
    }

    void _tx_async_event_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        while (!_tx_async_exit_requested.load(std::memory_order_relaxed)) {
            if (!_uplink_tx || !dev_->supports(radio::Capability::AsyncTxEvents)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            radio::AsyncMetadata async_md;
            bool got_event = false;
            try {
                got_event = _uplink_tx->tx_stream()->recv_async_msg(async_md, 0.1);
            } catch (const std::exception& e) {
                if (!_tx_async_exit_requested.load(std::memory_order_relaxed)) {
                    LOG_G_WARN_M(UlTx) << "[UL-TX Async] recv_async_msg failed: " << e.what();
                }
                continue;
            }
            if (!got_event) {
                continue;
            }

            auto log_event = [&](auto&& log_line) {
                log_line << "[UL-TX Async] " << tx_async_event_code_to_string(async_md.event_code)
                         << " (channel=" << async_md.channel;
                if (async_md.has_time_spec) {
                    log_line << ", event_time=" << std::fixed << std::setprecision(6)
                             << async_md.time_spec.get_real_secs() << " s" << std::defaultfloat;
                }
                log_line << ")";
            };

            switch (async_md.event_code) {
            case radio::AsyncEvent::BurstAck:
                log_event(LOG_G_INFO_M(UlTx));
                break;
            case radio::AsyncEvent::Underflow:
            case radio::AsyncEvent::UnderflowInPacket:
                _tx_async_error_count.fetch_add(1, std::memory_order_relaxed);
                log_event(LOG_G_WARN_M(UlTx));
                _request_stream_restart("UL-TX underflow");
                break;
            case radio::AsyncEvent::SeqError:
            case radio::AsyncEvent::SeqErrorInBurst:
            case radio::AsyncEvent::TimeError:
                _tx_async_error_count.fetch_add(1, std::memory_order_relaxed);
                log_event(LOG_G_ERROR_M(UlTx));
                {
                    // Coalesce async timing/sequence errors: a single stale or
                    // colliding burst emits thousands of these microseconds
                    // apart, and each restart tears down a burst that emits yet
                    // more. Rate-limit restart requests from this source so one
                    // burst-worth of errors triggers one recovery instead of a
                    // self-sustaining storm. Underflow/overflow paths are
                    // unaffected.
                    constexpr int64_t kAsyncRestartCooldownNs = 200'000'000;  // 200 ms
                    const int64_t now_ns = host_now_ns();
                    const int64_t last = _last_async_restart_request_ns.load(
                        std::memory_order_relaxed);
                    if (last < 0 || now_ns - last >= kAsyncRestartCooldownNs) {
                        _last_async_restart_request_ns.store(
                            now_ns, std::memory_order_relaxed);
                        _request_stream_restart("UL-TX async timing/sequence error");
                    }
                }
                break;
            default:
                log_event(LOG_G_INFO_M(UlTx));
                break;
            }
        }
    }

    bool _handle_rx_metadata_error(
        const radio::RxMetadata& md,
        const char* context,
        bool* restart_current_read = nullptr)
    {
        if (md.error_code == radio::RxError::None ||
            md.error_code == radio::RxError::Timeout) {
            return false;
        }

        if (restart_current_read) {
            *restart_current_read = true;
        }

        LOG_RT_WARN_M(Sync) << "[UE RX] " << context << " metadata error "
                      << rx_error_code_to_string(md.error_code) << ": "
                      << md.strerror();

        switch (md.error_code) {
        case radio::RxError::Overflow:
            _rx_overflow_count.fetch_add(1, std::memory_order_relaxed);
            _request_stream_restart("RX overflow");
            break;
        case radio::RxError::LateCommand:
        case radio::RxError::BrokenChain:
        case radio::RxError::Alignment:
        case radio::RxError::BadPacket:
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
    std::atomic<bool> _recovery_queue_clear_requested{false};
    std::atomic<int64_t> _rx_stream_start_time_ns{-1};
    std::atomic<uint64_t> _rx_stream_start_restart_epoch{0};
    std::atomic<uint32_t> _rx_boundary_log_remaining{0};
    std::atomic<int64_t> _last_rx_boundary_offset_samples{
        std::numeric_limits<int64_t>::min()};
    SPSCRingBuffer<RxFrame> frame_queue_;

    SPSCRingBuffer<SyncBatch> sync_queue_;

    size_t _ertm_delay_oversample_factor = 1;
    size_t _ertm_oversampled_fft_size = 0;
    AlignedVector _ertm_os_ifft_in;
    AlignedVector _ertm_os_ifft_out;
    fftwf_plan _ertm_os_ifft_plan = nullptr;
    AlignedVector _ertm_corr_a;
    AlignedVector _ertm_corr_b;
    AlignedVector _ertm_corr_a_freq;
    AlignedVector _ertm_corr_b_freq;
    AlignedVector _ertm_corr_out;
    // Consumer-owned, pre-sized once in prepare_fftw() — no per-frame heap growth.
    AlignedVector _ertm_bs_uplink_delay;
    AlignedVector _ertm_local_delay;
    AlignedVector _ertm_corrected_uplink_delay;
    AlignedVector _ertm_corrected_downlink_delay;
    std::vector<float> _ertm_correlation_spectrum;
    fftwf_plan _ertm_corr_fft_a_plan = nullptr;
    fftwf_plan _ertm_corr_fft_b_plan = nullptr;
    fftwf_plan _ertm_corr_ifft_plan = nullptr;

    // Per-frame demod scratch lives in CpuDemodWorkerContext (one per demod
    // worker); per-frame outputs live in the CpuDemodResult ring slots.
    ErtmAbsoluteFrameState _ertm_state;
    bool _ertm_legacy_payload_warned = false;

    // Core computation instances (manage their own FFT plans)
    ChannelEstimator _channel_estimator;
    DelayProcessor _delay_processor;
    SyncProcessor _sync_processor;
    AlignedVector _uplink_self_zc_freq;
    SelfZcChannelDebugEstimator _uplink_self_debug_estimator;
    AlignedVector _uplink_self_h_est;
    AlignedVector _uplink_self_delay_spectrum;
    uint64_t _uplink_self_debug_frame_counter = 0;
    // Diagnostic: full-frame matched filter for the UE's own uplink ZC. Unlike
    // _uplink_self_debug_estimator (which reads a FIXED window and silently
    // returns garbage when the TX/RX self-loopback is misaligned), this scans
    // the whole RX frame and reports where the self-ZC actually landed, so an
    // arbitrarily large TX-vs-RX window offset is still measurable. Only built
    // when self-channel debug is enabled; only run behind profiling + a stride.
    // The ZMQ wire payload is a peak-centered correlation slice + fixed header
    // (see _publish_uplink_self_scan_frame), not the full spectrum.
    std::unique_ptr<SyncProcessor> _uplink_self_scan_processor;
    uint64_t _uplink_self_scan_frame_counter = 0;
    AlignedVector _uplink_self_scan_slice;
    
    std::unique_ptr<FIRFilter> freq_offset_filter_;
    radio::TuneResult current_rx_tune_;
    radio::TuneResult current_ul_tx_tune_;
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
    std::unique_ptr<ZmqByteSender> uplink_self_scan_pub_;
    std::unique_ptr<ZmqByteSender> ertm_debug_pub_;
    std::unique_ptr<OcxoDebugSender> ocxo_debug_sender_;
    // Control handler
    ControlCommandHandler _control_handler;
    // Data sender management
    DataSender<std::complex<float>, AlignedAlloc> channel_sender_;
    DataSender<std::complex<float>, AlignedAlloc> pdf_sender_;
    DataSender<std::complex<float>, AlignedAlloc> constellation_sender_;
    DataSender<std::complex<float>, AlignedAlloc> uplink_self_channel_sender_;
    DataSender<std::complex<float>, AlignedAlloc> uplink_self_pdf_sender_;
    DataSender<uint8_t> uplink_self_scan_sender_;
    DataSender<ErtmDebugFrameData> ertm_debug_sender_;
    uint32_t _reset_count = 0;
    uint32_t _preamble_bad_count = 0; // process-thread-owned
    // Debug publish stride: constellation / channel / PDF snapshots are copied
    // out of the demod path once every N frames (senders pace at 50 ms anyway).
    static constexpr uint32_t kDebugPublishStride = 8;
    uint32_t _constellation_frame_counter = 0;
    
    // Sensing related variables
    // Parallel demod workers can complete a burst while sensing publishes a batch.
    // Pool sizing below includes this bounded queue's actual capacity.
    SPSCRingBuffer<SensingFrame> sensing_queue_{16};
    std::thread sensing_thread_;
    std::atomic<bool> sensing_running_{false};
    SharedSensingRuntime _shared_sensing_cfg;
    std::mutex _shared_sensing_cfg_mutex;
    std::unique_ptr<SensingChannel> _bistatic_sensing_channel;
    std::atomic<uint64_t> _next_bistatic_frame_start_symbol{0};
    // Owned by the demod-result collector.  If the aligned sensing frame is
    // dropped before enqueue, carry the mandatory eRTM refresh to the next
    // frame that actually reaches the sensing consumer.
    bool _ertm_remeasure_deferred_after_sensing_drop = false;

    double _freq_offset_sum = 0.0f; // Average frequency offset
    size_t _freq_offset_count = 0; // Sensing symbol count
    double _avg_freq_offset = 0.0;
    std::vector<int> _actual_subcarrier_indices;
    SFOEstimator sfo_estimator{1000};
    bool _sync_in_progress = false; // Flag to indicate if sync is in progress in process_proc
    uint32_t _sync_in_progress_hold_frames = 0; // Frames spent waiting for Alignment feedback
    // Alignment id process_proc is waiting on while _sync_in_progress is set.
    // process-thread only; RX stream-restart requests clear via
    // _process_tracking_reset_req (never written from RX).
    uint64_t _sync_in_progress_expect_id = 0;
    int _last_delay_index_err = 0; // Last adjusted index
    uint32_t _delay_adjustment_count = 0;
    std::atomic<float> _user_delay_offset = 0.0f;
    std::atomic<int32_t> _ue_timing_advance{0};  // Timing Advance (samples)
    std::atomic<int64_t> _last_tadv_ns{0};       // rate-limit guard for TADV
    // Target mirrored into UL-TX as soon as Alignment is scheduled.
    std::atomic<int32_t> _uplink_tx_rx_alignment_shift{0};
    // Alignment actually committed to received DL frame coordinates.
    std::atomic<int32_t> _committed_uplink_tx_rx_alignment_shift{0};
    std::atomic<uint64_t> _alignment_schedule_count{0};
    std::atomic<uint64_t> _pending_alignment_id{0};
    std::atomic<uint64_t> _pending_alignment_restart_epoch{0};
    std::atomic<uint64_t> _pending_alignment_generation{0};
    // RX thread publishes the id / samples / time of the last successfully
    // applied alignment. process_proc uses this to:
    //  - clear _sync_in_progress when the one-shot Alignment frame is dropped
    //  - inject Alignment into SFO / force eRTM remeasure on the first
    //    post-align frame that reaches process (full Alignment semantics).
    std::atomic<uint64_t> _alignment_applied_id{0};
    std::atomic<int32_t> _alignment_applied_samples{0};
    std::atomic<int64_t> _alignment_applied_time_ns{-1};
    // process-thread only: last applied_id whose SFO/eRTM side effects ran.
    uint64_t _alignment_applied_consumed_id = 0;
    // RX requests process-owned tracking reset (latch, SFO, OCXO age).
    std::atomic<uint64_t> _process_tracking_reset_req{0};
    uint64_t _process_tracking_reset_applied = 0;
    // Sensing (or any non-process eRTM consumer) requests OCXO TO reanchor.
    std::atomic<uint64_t> _ocxo_to_reanchor_req{0};
    uint64_t _ocxo_to_reanchor_applied = 0;
    // True while the pending alignment came from a fresh SYNC_SEARCH acquisition,
    // so handle_alignment() re-anchors the UL-TX window absolutely to the newly
    // committed RX frame's grid boundary phase (see _schedule_receive_alignment).
    std::atomic<bool> _pending_alignment_fresh{false};

    void _reset_uplink_tx_rx_alignment_shift(uint64_t restart_epoch = 0) {
        _uplink_tx_rx_alignment_shift.store(0, std::memory_order_relaxed);
        _committed_uplink_tx_rx_alignment_shift.store(0, std::memory_order_release);
        if (_uplink_tx) {
            _uplink_tx->reset_rx_alignment_shift(restart_epoch);
        }
    }

    int32_t _canonical_uplink_tx_alignment_delta(int32_t rx_alignment_delta_samples) const {
        const int64_t frame_samples = static_cast<int64_t>(cfg_.samples_per_frame());
        if (frame_samples <= 0) {
            return rx_alignment_delta_samples;
        }
        int64_t delta = static_cast<int64_t>(rx_alignment_delta_samples) % frame_samples;
        const int64_t half_frame = frame_samples / 2;
        if (delta > half_frame) {
            delta -= frame_samples;
        } else if (delta < -half_frame) {
            delta += frame_samples;
        }
        return static_cast<int32_t>(std::clamp<int64_t>(
            delta,
            static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
            static_cast<int64_t>(std::numeric_limits<int32_t>::max())));
    }

    // Keep the UL-TX frame anchor tied to the currently aligned UE RX/downlink
    // frame anchor modulo one frame. RX may need to consume almost a whole frame
    // during reacquire; UL-TX only needs the equivalent window-phase residual.
    void _apply_uplink_tx_rx_alignment_delta(
        int32_t rx_alignment_delta_samples,
        uint64_t alignment_id,
        uint64_t restart_epoch)
    {
        const int32_t ul_tx_rx_alignment_delta_samples =
            _canonical_uplink_tx_alignment_delta(rx_alignment_delta_samples);
        if (ul_tx_rx_alignment_delta_samples == 0) {
            if (_uplink_tx) {
                _uplink_tx->store_rx_alignment_shift(
                    _uplink_tx_rx_alignment_shift.load(std::memory_order_relaxed),
                    alignment_id,
                    restart_epoch);
            }
            return;
        }

        const int64_t tx_delta64 = std::clamp<int64_t>(
            -static_cast<int64_t>(ul_tx_rx_alignment_delta_samples),
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
            _uplink_tx->store_rx_alignment_shift(next, alignment_id, restart_epoch);
        }
        if (LOG_MOD_ON(Recovery)) {
            LOG_RT_WARN_M(UlTx) << "[UL-TX] RX alignment delta=" << rx_alignment_delta_samples
                          << " samples, ul_tx_effective_rx_delta="
                          << ul_tx_rx_alignment_delta_samples
                          << " samples, alignment_id=" << alignment_id
                          << ", restart_epoch=" << restart_epoch
                          << " -> TX timing delta=" << tx_alignment_delta_samples
                          << " samples, cumulative RX-alignment target=" << next;
        } else if (LOG_MOD_ON(UlTx)) {
            LOG_RT_WARN_HZ_M(UlTx, 2) << "[UL-TX] RX alignment delta=" << rx_alignment_delta_samples
                              << " samples -> TX timing delta=" << tx_alignment_delta_samples
                              << " samples, cumulative RX-alignment target=" << next;
        }
    }

    // Fresh-acquisition absolute reconstruction of the UL-TX window shift.
    // grid_phase_samples is the RX frame boundary offset modulo one frame,
    // measured against the timed RX stream anchor (sync-read timestamp +
    // discard), so it is independent of where the sync search happened to start
    // reading. The relative-delta path instead folds the sync-read-relative
    // discard amount, which equals the true grid phase only when the sync read
    // starts on a frame boundary (nearest_boundary_error==0). After a stream
    // restart that read is not grid-aligned, so the leftover misalignment leaks
    // straight into the TX window and breaks the self-loopback even though the
    // downlink RX (which only discards to its own boundary) stays fine.
    // Anchoring the TX shift to -grid_phase keeps the UE's own uplink ZC landing
    // at the uplink window inside its own RX frame across restarts.
    void _reconstruct_uplink_tx_rx_alignment_absolute(
        int64_t grid_phase_samples,
        uint64_t alignment_id,
        uint64_t restart_epoch)
    {
        const int64_t frame_samples = static_cast<int64_t>(cfg_.samples_per_frame());
        int64_t phase = grid_phase_samples;
        if (frame_samples > 0) {
            phase = _positive_mod_i64(grid_phase_samples, frame_samples);
        }
        // Fold to the nearest-zero representative so the physical TX shift stays
        // within +/- half a frame (the shorter insert/skip direction).
        const int32_t canonical_phase =
            _canonical_uplink_tx_alignment_delta(static_cast<int32_t>(phase));
        const int32_t shift = -canonical_phase;
        _uplink_tx_rx_alignment_shift.store(shift, std::memory_order_relaxed);
        if (_uplink_tx) {
            _uplink_tx->store_rx_alignment_shift(shift, alignment_id, restart_epoch);
        }
        if (LOG_MOD_ON(Recovery)) {
            LOG_RT_WARN_M(UlTx) << "[UL-TX] absolute reacquire: grid_phase="
                          << grid_phase_samples
                          << " samples, canonical_phase=" << canonical_phase
                          << " -> ul_tx_rx_shift=" << shift
                          << ", alignment_id=" << alignment_id
                          << ", restart_epoch=" << restart_epoch;
        }
    }

    void _schedule_receive_alignment(int32_t alignment_samples) {
        const RxState state_before = state_.load(std::memory_order_relaxed);
        const uint64_t alignment_id =
            _alignment_schedule_count.fetch_add(1, std::memory_order_relaxed) + 1;
        const uint64_t sync_generation =
            _sync_generation.load(std::memory_order_relaxed);
        const uint64_t restart_epoch =
            _stream_restart_count.load(std::memory_order_relaxed);
        const bool restart_pending =
            _stream_restart_requested.load(std::memory_order_acquire);
        const int32_t before_shift =
            _uplink_tx_rx_alignment_shift.load(std::memory_order_relaxed);
        _pending_alignment_id.store(alignment_id, std::memory_order_relaxed);
        _pending_alignment_restart_epoch.store(restart_epoch, std::memory_order_relaxed);
        _pending_alignment_generation.store(sync_generation, std::memory_order_relaxed);
        discard_samples_.store(static_cast<int>(alignment_samples), std::memory_order_relaxed);
        // Fresh acquisition (SYNC_SEARCH -> ALIGNMENT): defer the UL-TX window
        // phase to handle_alignment(), which rebuilds it absolutely from the
        // committed RX frame's own grid boundary phase (authoritative: that
        // frame's RX timestamp + applied discard). The sync-read timestamp does
        // not give the boundary phase reliably. Continuous-tracking corrections
        // keep the relative-delta accumulation here.
        const bool fresh_acquisition = (state_before == RxState::SYNC_SEARCH);
        _pending_alignment_fresh.store(fresh_acquisition, std::memory_order_relaxed);
        if (!fresh_acquisition) {
            _apply_uplink_tx_rx_alignment_delta(
                alignment_samples, alignment_id, restart_epoch);
        }
        if (LOG_MOD_ON(Recovery)) {
            LOG_RT_WARN_M(Recovery) << "[UE recovery] schedule_receive_alignment alignment_id="
                          << alignment_id
                          << ", alignment="
                          << alignment_samples
                          << ", previous_ul_tx_rx_shift=" << before_shift
                          << ", next_ul_tx_rx_shift="
                          << _uplink_tx_rx_alignment_shift.load(std::memory_order_relaxed)
                          << ", state_before="
                          << static_cast<int>(state_before)
                          << ", sync_generation=" << sync_generation
                          << ", restart_epoch=" << restart_epoch
                          << ", restart_pending=" << restart_pending;
        }
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
        if (_uplink_tx->set_waveform_enabled(enabled) && LOG_MOD_ON(UlTx)) {
            LOG_RT_INFO_M(UlTx) << "[UL-TX] "
                          << (enabled ? "enabled" : "muted")
                          << " uplink waveform transmission";
        }
    }

    void _mute_uplink_tx_gain_for_sync_search() {
        if (!_uplink_tx || !dev_->supports(radio::Capability::HardwareGain) ||
            !_uplink_tx_gain_range_initialized) {
            return;
        }
        std::lock_guard<std::mutex> lock(_uplink_tx_gain_mutex);
        try {
            dev_->set_tx_gain(_uplink_tx_gain_min_db, cfg_.uplink.tx_channel);
            _uplink_tx_gain_muted.store(true, std::memory_order_release);
        } catch (const std::exception& e) {
            LOG_RT_WARN_M(UlTx) << "[UL-TX] failed to mute TX gain during sync search: " << e.what();
        }
    }

    void _restore_uplink_tx_gain_after_sync() {
        if (!_uplink_tx || !dev_->supports(radio::Capability::HardwareGain) ||
            !_uplink_tx_gain_range_initialized) {
            return;
        }
        std::lock_guard<std::mutex> lock(_uplink_tx_gain_mutex);
        try {
            dev_->set_tx_gain(_uplink_tx_gain_restore_db, cfg_.uplink.tx_channel);
            _uplink_tx_gain_muted.store(false, std::memory_order_release);
        } catch (const std::exception& e) {
            LOG_RT_WARN_M(UlTx) << "[UL-TX] failed to restore TX gain after sync: " << e.what();
        }
    }
    std::unique_ptr<HardwareSyncController> _hw_sync;
    std::unique_ptr<OCXODiscipliner> _ocxo_discipliner;
    std::unique_ptr<OCXOPpsInitialAligner> _ocxo_pps_initial_aligner;

    // Data processing related member variables
    struct LlrFrame {
        AlignedFloatVector llr;
        uint64_t generation = 0;
        int64_t rx_enqueue_time_ns = 0;
        int64_t process_dequeue_time_ns = 0;
        int64_t demod_done_time_ns = 0;
        AirFrameIdentity identity;
    };
    // int16 (Q16) variant used when cfg_.ldpc.fixed_point is set. The demapper
    // quantizes LLRs directly into this so the whole pipeline runs int16.
    struct LlrFrameI16 {
        LDPCCodec::AlignedShortVector llr;
        uint64_t generation = 0;
        int64_t rx_enqueue_time_ns = 0;
        int64_t process_dequeue_time_ns = 0;
        int64_t demod_done_time_ns = 0;
        AirFrameIdentity identity;
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

    struct CpuDemodProfile {
        double fft_total = 0.0;
        double channel_est_total = 0.0;
        double cfo_sfo_est_total = 0.0;
        double equalization_total = 0.0;
        double eq_base_inv_total = 0.0;
        double eq_channel_select_total = 0.0;
        double eq_symbol_inv_total = 0.0;
        double eq_pilot_phase_total = 0.0;
        double eq_apply_total = 0.0;
        uint64_t eq_data_symbols_total = 0;
        uint64_t eq_midframe_channel_symbols_total = 0;
        uint64_t eq_symbol_inv_count_total = 0;
        uint64_t eq_pilot_phase_attempt_total = 0;
        uint64_t eq_pilot_phase_success_total = 0;
        double noise_est_total = 0.0;
        double remodulate_total = 0.0;
        double delay_spectrum_total = 0.0;
        double timing_sync_total = 0.0;
        double sensing_queue_total = 0.0;
        double udp_send_total = 0.0;
        double llr_total = 0.0;
        int frame_count = 0;

        void reset() {
            *this = CpuDemodProfile{};
        }
    };

    struct CpuDemodWorkerContext {
        AlignedVector fft_input;
        AlignedVector fft_output;
        fftwf_plan fft_plan = nullptr;
        std::vector<AlignedVector> symbols;
        AlignedVector sync_symbol_freq;
        std::vector<AlignedVector> midframe_symbol_freqs;
        std::vector<int> pilot_indices;
        std::vector<float> avg_phase_diff;
        std::vector<float> weights;
        std::vector<int> tracking_pilot_indices;
        std::vector<float> tracking_phase;
        std::vector<float> tracking_weights;
        AlignedVector tracking_h_inv;
        AlignedVector midframe_interp_h;
        // Channel anchors: the symbol set is fixed by the resource layout
        // (sync + valid full-band mid-frame pilots), so the sorted symbol list
        // and per-anchor source are precomputed once in
        // _init_cpu_demod_workers() and only the H buffers are refilled
        // in place each frame (no per-frame allocation).
        std::vector<AlignedVector> channel_anchor_h;
        std::vector<int> channel_anchor_symbols;
        std::vector<int> channel_anchor_source; // -1 = sync H_est, else midframe pilot rank
        AlignedVector h_inv;
        AlignedVector last_anchor_h_inv;
        ChannelEstimator channel_estimator;
        DelayProcessor delay_processor;

        explicit CpuDemodWorkerContext(const Config& cfg)
            : channel_estimator(cfg.ofdm.fft_size),
              delay_processor(cfg.ofdm.fft_size)
        {
            fft_input.resize(cfg.ofdm.fft_size);
            fft_output.resize(cfg.ofdm.fft_size);
            fft_plan = fftwf_plan_dft_1d(
                cfg.ofdm.fft_size,
                reinterpret_cast<fftwf_complex*>(fft_input.data()),
                reinterpret_cast<fftwf_complex*>(fft_output.data()),
                FFTW_FORWARD,
                FFTW_MEASURE);
            const DataResourceGridLayout layout = build_data_resource_grid_layout(cfg);
            symbols.resize(layout.data_symbol_count);
            for (auto& s : symbols) {
                s.resize(cfg.ofdm.fft_size);
            }
            sync_symbol_freq.resize(cfg.ofdm.fft_size);
            midframe_symbol_freqs.resize(layout.midframe_pilot_symbol_count);
            for (auto& s : midframe_symbol_freqs) {
                s.resize(cfg.ofdm.fft_size);
            }
            tracking_h_inv.resize(cfg.ofdm.fft_size);
            midframe_interp_h.resize(cfg.ofdm.fft_size);
            h_inv.resize(cfg.ofdm.fft_size);
            last_anchor_h_inv.resize(cfg.ofdm.fft_size);
            tracking_pilot_indices.reserve(cfg.ofdm.pilot_positions.size());
            tracking_phase.reserve(cfg.ofdm.pilot_positions.size());
            tracking_weights.reserve(cfg.ofdm.pilot_positions.size());
        }

        ~CpuDemodWorkerContext() {
            if (fft_plan) {
                fftwf_destroy_plan(fft_plan);
            }
        }

        CpuDemodWorkerContext(const CpuDemodWorkerContext&) = delete;
        CpuDemodWorkerContext& operator=(const CpuDemodWorkerContext&) = delete;
    };

    // Result lives inside the SPSC ring slot: the worker fills it in place via
    // producer_slot()/producer_commit(), the collector reads it in place via
    // consumer_slot()/consumer_pop(). h_est / delay_spectrum /
    // constellation_symbol are pre-sized by the slot factory and refilled each
    // frame; pooled objects (frame, sense_frame, llr) move through.
    struct CpuDemodResult {
        RxFrame frame;
        uint64_t generation = 0;
        bool dropped = false;   // stale-generation fast-drop token
        bool debug_frame = false; // strided debug-publish frame
        int64_t frame_dequeue_time_ns = 0;
        bool cfo_sfo_estimate_valid = false;
        float alpha = 0.0f;
        float beta = 0.0f;
        float detected_freq_offset = 0.0f;
        float corrected_impulse_snr_linear_est = 1.0f;
        size_t delay_max_index = 0;
        float delay_max_mag = 0.0f;
        float delay_average_mag = 0.0f;
        int adjusted_delay_index = 0;
        float fractional_delay = 0.0f;
        bool evm_valid = false;
        FrameEvmStats evm{};
        AlignedVector h_est;
        AlignedVector delay_spectrum;
        bool has_constellation = false;
        AlignedVector constellation_symbol;
        SensingFrame sense_frame;
        bool has_sensing = false;
        bool has_llr_float = false;
        bool has_llr_i16 = false;
        AlignedFloatVector llr;
        LDPCCodec::AlignedShortVector llr_i16;
        CpuDemodProfile profile;
    };

    struct CpuDemodTask {
        RxFrame frame;
        int64_t frame_dequeue_time_ns = 0;
        float llr_scale_snapshot = 2.0f;
        bool want_debug_copies = false;
    };

    struct CpuDemodSlot {
        // Depth 2 double-buffers each worker: it can start the next frame while
        // the collector is still consuming the previous result, hiding the
        // control-stage latency that a depth-1 slot exposes as worker idle time.
        static constexpr size_t kPipelineDepth = 2;

        SPSCRingBuffer<CpuDemodTask> task_queue;
        SPSCRingBuffer<CpuDemodResult> result_queue;
        std::thread thread;
        size_t pending = 0; // control-thread-owned in-flight count (<= kPipelineDepth)

        explicit CpuDemodSlot(const Config& cfg)
            : task_queue(kPipelineDepth),
              result_queue(kPipelineDepth, [&cfg]() {
                  CpuDemodResult r;
                  r.h_est.resize(cfg.ofdm.fft_size);
                  r.delay_spectrum.resize(cfg.ofdm.fft_size);
                  r.constellation_symbol.resize(cfg.ofdm.fft_size);
                  return r;
              }) {}
    };

    // ---- Parallel LDPC decode stage (same ordered round-robin pattern as the
    // demod stage): the bit-processing thread dispatches LLR frames to N
    // decode workers and collects decoded packets in frame order; the stateful
    // packet handling (eRTM / measurement / ARQ / UDP / latency) stays serial
    // on the collector.

    struct CpuLdpcPacketRef {
        LdpcMiniHeader mini_header{};
        size_t payload_offset = 0; // into CpuLdpcResult::payload_bytes
        size_t payload_len = 0;
    };

    struct CpuLdpcResult {
        uint64_t generation = 0;
        bool dropped = false;
        int64_t rx_enqueue_time_ns = 0;
        int64_t process_dequeue_time_ns = 0;
        int64_t demod_done_time_ns = 0;
        AirFrameIdentity identity;
        std::vector<CpuLdpcPacketRef> packets;
        std::vector<uint8_t> payload_bytes; // flat decoded-payload storage
    };

    struct CpuLdpcTask {
        AlignedFloatVector llr;                    // float path (pooled)
        LDPCCodec::AlignedShortVector llr_i16;     // int16 path (pooled)
        bool is_i16 = false;
        uint64_t generation = 0;
        int64_t rx_enqueue_time_ns = 0;
        int64_t process_dequeue_time_ns = 0;
        int64_t demod_done_time_ns = 0;
        AirFrameIdentity identity;
    };

    struct CpuLdpcWorkerContext {
        LDPCCodec decoder; // stateful aff3ct instance: strictly one per worker
        LDPCCodec::AlignedFloatVector deint_scratch;
        LDPCCodec::AlignedShortVector deint_scratch_i16;
        AlignedFloatVector payload_llr;
        LDPCCodec::AlignedShortVector payload_llr_i16;
        LDPCCodec::AlignedByteVector decoded_payload;

        explicit CpuLdpcWorkerContext(const LDPCCodec::LDPCConfig& ldpc_cfg)
            : decoder(ldpc_cfg) {}

        CpuLdpcWorkerContext(const CpuLdpcWorkerContext&) = delete;
        CpuLdpcWorkerContext& operator=(const CpuLdpcWorkerContext&) = delete;
    };

    struct CpuLdpcSlot {
        static constexpr size_t kPipelineDepth = 2;

        SPSCRingBuffer<CpuLdpcTask> task_queue;
        SPSCRingBuffer<CpuLdpcResult> result_queue;
        std::thread thread;
        size_t pending = 0; // collector-thread-owned in-flight count

        CpuLdpcSlot()
            : task_queue(kPipelineDepth),
              result_queue(kPipelineDepth, []() {
                  CpuLdpcResult r;
                  r.packets.reserve(16);
                  r.payload_bytes.reserve(16384);
                  return r;
              }) {}
    };

    // LLR hand-off queues (float / int16 path). Deliberately deep (128): LLR
    // frames feed the non-realtime decode stage, so buffering bursty payload
    // here is cheap latency-wise, and the pool prefill below covers the full
    // depth (~100 MB on the float path at full-grid payloads — accepted).
    SPSCRingBuffer<LlrFrame> _data_llr_buffer{128};
    SPSCRingBuffer<LlrFrameI16> _data_llr_buffer_i16{128};
    std::thread _bit_processing_thread;
    std::atomic<bool> _bit_processing_running{false};
    LatencyAccumulator _latency_accumulator;

    const bool _ldpc_fixed_point{cfg_.ldpc.fixed_point};
    // Per-worker LDPCCodec instances live in _cpu_ldpc_contexts (aff3ct
    // decoders are stateful and not shareable); the interleaver/scrambler are
    // const after construction and shared by all workers.
    std::unique_ptr<BitBlockInterleaver> _bit_interleaver;
    Scrambler _descrambler{201600, 0x5A};
    std::unique_ptr<UdpSender> _udp_output_sender;

    // ARQ: downlink receive window for duplicate suppression + ordered delivery
    ArqRxWindow _dl_arq_rx;
    bool _arq_enabled{false};
    UeAirFrameTimeline _ue_air_timeline; ///< Absolute seq + UE-USRP times (mutex)
    AirFrameIdentity _air_identity_rx_snapshot; ///< RX-thread-only diagnostic snapshot
    SPSCRingBuffer<AirFrameSibEvent> _sib_event_queue{32};
    /// Non-zero = process should reset SFO/sync if still equal to live generation.
    std::atomic<uint64_t> _pending_sfo_reset_generation{0};
    /// Set by bit thread on OTA epoch change; cleared by RX when SFO is requested.
    /// Survives a lost first SIB event after last_seen already advanced.
    std::atomic<bool> _ota_epoch_sfo_needed{false};
    /// Survives RF/sync clear — used to detect BS restart after SYNC_SEARCH.
    /// Stores canonical OTA epoch32 (same as AirFrameIdentity.timeline_epoch).
    std::atomic<uint32_t> _last_seen_ota_epoch{0};
    std::atomic<bool> _has_seen_ota_epoch{false};
    double _ue_rx_sample_rate = 0.0; ///< Actual UHD RX rate (may differ from YAML)
    std::atomic<int64_t> _ue_air_frame_start_time_ns{-1}; ///< Last boundary (alignment geometry)
    std::atomic<int64_t> _ue_air_frame_mid_time_ns{-1};   ///< Last mid-frame (robust stamp)
    std::atomic<uint64_t> _air_identity_lock_count{0};
    std::atomic<uint64_t> _air_identity_stamp_count{0};
    std::atomic<uint64_t> _air_sib_publish_count{0};
    std::atomic<uint64_t> _air_sib_stale_drop_count{0};
    std::atomic<uint64_t> _air_timeline_reset_count{0};
    std::atomic<uint64_t> _arq_profile_last_log_ms{0};
    std::atomic<uint64_t> _arq_dl_data_seen{0};
    std::atomic<uint64_t> _arq_dl_data_accepted{0};
    std::atomic<uint64_t> _arq_dl_data_duplicates{0};
    std::atomic<uint64_t> _arq_dl_feedback_seen{0};
    std::atomic<uint64_t> _arq_dl_feedback_valid{0};
    std::atomic<uint64_t> _arq_dl_feedback_invalid{0};
    std::atomic<uint64_t> _arq_dl_feedback_unhandled{0};
    std::atomic<uint64_t> _arq_dl_ack_generated{0};
    std::atomic<uint64_t> _arq_dl_ack_injected{0};
    std::atomic<uint64_t> _arq_dl_ack_no_uplink_tx{0};
    std::atomic<uint64_t> _arq_dl_ack_inject_failed{0};

    bool _compute_ertm_oversampled_delay_spectrum(
        const AlignedVector& channel_freq,
        double shift_samples,
        AlignedVector& delay_spectrum)
    {
        const size_t n = cfg_.ofdm.fft_size;
        const size_t os_n = _ertm_oversampled_fft_size;
        if (n == 0 || os_n == 0 || channel_freq.size() != n || !_ertm_os_ifft_plan) {
            return false;
        }

        std::fill(_ertm_os_ifft_in.begin(), _ertm_os_ifft_in.end(), std::complex<float>(0.0f, 0.0f));
        const size_t half = n / 2;
        static constexpr double kTwoPi = 6.283185307179586476925286766559;
        const double inv_n = 1.0 / static_cast<double>(n);
        const bool apply_shift = std::isfinite(shift_samples) && shift_samples != 0.0;
        for (size_t i = 0; i < half; ++i) {
            auto rotate = [&](std::complex<float> value, int64_t signed_k) {
                if (!apply_shift) {
                    return value;
                }
                const double phase = -kTwoPi * static_cast<double>(signed_k) * shift_samples * inv_n;
                const std::complex<float> rot(
                    static_cast<float>(std::cos(phase)),
                    static_cast<float>(std::sin(phase)));
                return value * rot;
            };
            // Convert FFTW-native H_est into natural signed-frequency order
            // [-N/2, ..., N/2-1], then leave all oversampling bins after N as zero.
            _ertm_os_ifft_in[i] = rotate(
                channel_freq[i + half],
                static_cast<int64_t>(i) - static_cast<int64_t>(half));
            _ertm_os_ifft_in[i + half] = rotate(channel_freq[i], static_cast<int64_t>(i));
        }

        fftwf_execute(_ertm_os_ifft_plan);

        const float scale = 1.0f / std::sqrt(static_cast<float>(n));
        // Prefer capacity-preserving resize when the caller pre-sized the buffer.
        if (delay_spectrum.size() != os_n) {
            delay_spectrum.resize(os_n);
        }
        for (size_t i = 0; i < os_n; ++i) {
            delay_spectrum[i] = _ertm_os_ifft_out[i] * scale;
        }
        return true;
    }

    static double _ertm_centroid3_delta(double y_left, double y_center, double y_right) {
        const double min_y = std::min(y_left, std::min(y_center, y_right));
        double w_left = std::max(0.0, y_left - min_y);
        double w_center = std::max(0.0, y_center - min_y);
        double w_right = std::max(0.0, y_right - min_y);
        double total = w_left + w_center + w_right;
        if (!(total > 0.0) || !std::isfinite(total)) {
            w_left = std::max(0.0, y_left);
            w_center = std::max(0.0, y_center);
            w_right = std::max(0.0, y_right);
            total = w_left + w_center + w_right;
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            return 0.0;
        }
        const double delta = (w_right - w_left) / total;
        if (!std::isfinite(delta)) {
            return 0.0;
        }
        return std::max(-0.5, std::min(0.5, delta));
    }

    static double _ertm_parabolic_delta(double y_left, double y_center, double y_right) {
        const double curvature = y_left - 2.0 * y_center + y_right;
        if (!std::isfinite(curvature) || curvature >= -1e-15) {
            return 0.0;
        }
        const double delta = 0.5 * (y_left - y_right) / curvature;
        if (!std::isfinite(delta)) {
            return 0.0;
        }
        return std::max(-0.5, std::min(0.5, delta));
    }

    bool _estimate_ertm_delay_shift(
        const AlignedVector& bs_uplink_delay,
        const AlignedVector& ue_downlink_delay,
        const std::string& timing_metric,
        int64_t& signed_shift_bins,
        double& metric_out,
        std::vector<float>* correlation_out = nullptr,
        size_t* peak_index_out = nullptr,
        double* fractional_delta_bins_out = nullptr
    ) {
        const size_t n = bs_uplink_delay.size();
        if (n == 0 || ue_downlink_delay.size() != n ||
            n != _ertm_oversampled_fft_size ||
            !_ertm_corr_fft_a_plan || !_ertm_corr_fft_b_plan || !_ertm_corr_ifft_plan) {
            return false;
        }
        if (correlation_out) {
            if (correlation_out->size() != n) {
                correlation_out->assign(n, 0.0f);
            } else {
                std::fill(correlation_out->begin(), correlation_out->end(), 0.0f);
            }
        }
        const bool maximum_likelihood =
            timing_metric == kErtmTimingMetricMaximumLikelihood;

        double bs_energy = 0.0;
        double ue_energy = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const auto bs_value = bs_uplink_delay[i];
            const auto ue_value = ue_downlink_delay[i];
            const float bs_power = std::norm(bs_value);
            const float ue_power = std::norm(ue_value);
            if (maximum_likelihood) {
                // By the correlation theorem this complex delay-profile
                // correlation is exactly the oversampled frequency-domain ML
                // metric IFFT{H_BS * conj(H_UE)} for white endpoint noise.
                _ertm_corr_a[i] = bs_value;
                _ertm_corr_b[i] = ue_value;
            } else {
                _ertm_corr_a[i] = std::complex<float>(std::sqrt(bs_power), 0.0f);
                _ertm_corr_b[i] = std::complex<float>(std::sqrt(ue_power), 0.0f);
            }
            bs_energy += static_cast<double>(bs_power);
            ue_energy += static_cast<double>(ue_power);
        }
        const double denom = std::sqrt(bs_energy * ue_energy);
        if (!(denom > 0.0) || !std::isfinite(denom)) {
            return false;
        }

        fftwf_execute(_ertm_corr_fft_a_plan);
        fftwf_execute(_ertm_corr_fft_b_plan);
        for (size_t i = 0; i < n; ++i) {
            _ertm_corr_a_freq[i] = _ertm_corr_a_freq[i] * std::conj(_ertm_corr_b_freq[i]);
        }
        fftwf_execute(_ertm_corr_ifft_plan);

        auto metric_at = [&](size_t shift) {
            const auto corr = _ertm_corr_out[shift];
            const double raw_metric = maximum_likelihood
                ? static_cast<double>(std::abs(corr))
                : static_cast<double>(corr.real());
            return raw_metric / (static_cast<double>(n) * denom);
        };

        size_t best_shift = 0;
        double best_metric = -std::numeric_limits<double>::infinity();
        for (size_t shift = 0; shift < n; ++shift) {
            const double metric = metric_at(shift);
            if (correlation_out) {
                (*correlation_out)[shift] = static_cast<float>(metric);
            }
            if (metric > best_metric) {
                best_metric = metric;
                best_shift = shift;
            }
        }

        double fractional_delta_bins = 0.0;
        if (n >= 3) {
            const double y_left = metric_at((best_shift + n - 1) % n);
            const double y_center = metric_at(best_shift);
            const double y_right = metric_at((best_shift + 1) % n);
            fractional_delta_bins = maximum_likelihood
                ? _ertm_parabolic_delta(y_left, y_center, y_right)
                : _ertm_centroid3_delta(y_left, y_center, y_right);
        }
        if (fractional_delta_bins_out) {
            *fractional_delta_bins_out = fractional_delta_bins;
        }
        if (peak_index_out) {
            *peak_index_out = best_shift;
        }
        signed_shift_bins = (best_shift < (n + 1) / 2)
            ? static_cast<int64_t>(best_shift)
            : static_cast<int64_t>(best_shift) - static_cast<int64_t>(n);
        metric_out = best_metric;
        return true;
    }

    static void _append_u32_le(std::vector<uint8_t>& out, uint32_t value) {
        out.push_back(static_cast<uint8_t>(value & 0xFFu));
        out.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
        out.push_back(static_cast<uint8_t>((value >> 16) & 0xFFu));
        out.push_back(static_cast<uint8_t>((value >> 24) & 0xFFu));
    }

    static void _append_i32_le(std::vector<uint8_t>& out, int32_t value) {
        _append_u32_le(out, static_cast<uint32_t>(value));
    }

    static void _append_u64_le(std::vector<uint8_t>& out, uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            out.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFFu));
        }
    }

    static void _append_i64_le(std::vector<uint8_t>& out, int64_t value) {
        _append_u64_le(out, static_cast<uint64_t>(value));
    }

    static void _append_double_le(std::vector<uint8_t>& out, double value) {
        uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "Unexpected double size");
        std::memcpy(&bits, &value, sizeof(bits));
        _append_u64_le(out, bits);
    }

    size_t _ertm_debug_window_bins(size_t full_bins) const {
        if (full_bins == 0) {
            return 0;
        }
        size_t window = std::max(
            cfg_.ofdm.fft_size,
            2 * cfg_.ofdm.cp_length * _ertm_delay_oversample_factor);
        window = std::min(window, full_bins);
        if ((window & 1u) != 0u && window > 1) {
            --window;
        }
        return std::max<size_t>(window, 1);
    }

    static void _copy_center_delay_window(
        const AlignedVector& input,
        size_t window_bins,
        AlignedVector& output)
    {
        const size_t n = input.size();
        if (window_bins == 0 || window_bins >= n) {
            output = input;
            return;
        }
        const size_t positive_bins = window_bins / 2;
        const size_t negative_bins = window_bins - positive_bins;
        output.resize(window_bins);
        std::copy(input.begin(), input.begin() + positive_bins, output.begin());
        std::copy(input.end() - negative_bins, input.end(), output.begin() + positive_bins);
    }

    static void _copy_center_delay_window(
        const std::vector<float>& input,
        size_t window_bins,
        std::vector<float>& output)
    {
        const size_t n = input.size();
        if (window_bins == 0 || window_bins >= n) {
            output = input;
            return;
        }
        const size_t positive_bins = window_bins / 2;
        const size_t negative_bins = window_bins - positive_bins;
        output.resize(window_bins);
        std::copy(input.begin(), input.begin() + positive_bins, output.begin());
        std::copy(input.end() - negative_bins, input.end(), output.begin() + positive_bins);
    }

    void _publish_ertm_debug_frame(
        const ErtmTimingPayloadView& bs_payload,
        const AlignedVector& bs_uplink_delay,
        const AlignedVector& ue_downlink_delay,
        const std::vector<float>& correlation,
        const AlignedVector& corrected_uplink_delay,
        const AlignedVector& corrected_downlink_delay,
        size_t peak_index,
        int64_t signed_shift_bins,
        double metric,
        int32_t tadv_samples,
        double rf_delay_samples,
        double tau_c_samples,
        double to_bs_ue_samples,
        double to_ue_samples,
        double to_bs_samples)
    {
        if (!cfg_.uplink.ertm_debug_output_enabled || !ertm_debug_pub_) {
            return;
        }
        const size_t n = bs_uplink_delay.size();
        if (n == 0 || ue_downlink_delay.size() != n || correlation.size() != n ||
            corrected_uplink_delay.size() != n || corrected_downlink_delay.size() != n) {
            return;
        }

        const size_t wire_bins = _ertm_debug_window_bins(n);
        ErtmDebugFrameData frame;
        _copy_center_delay_window(bs_uplink_delay, wire_bins, frame.uplink_delay);
        _copy_center_delay_window(ue_downlink_delay, wire_bins, frame.downlink_delay);
        _copy_center_delay_window(
            corrected_uplink_delay, wire_bins, frame.corrected_uplink_delay);
        _copy_center_delay_window(
            corrected_downlink_delay, wire_bins, frame.corrected_downlink_delay);
        _copy_center_delay_window(correlation, wire_bins, frame.correlation);

        frame.header.reserve(112);
        static constexpr uint8_t kMagic[8] = {'E', 'R', 'T', 'M', 'D', 'B', 'G', '1'};
        frame.header.insert(frame.header.end(), kMagic, kMagic + sizeof(kMagic));
        _append_u32_le(frame.header, 3);  // format version
        _append_u32_le(frame.header, static_cast<uint32_t>(wire_bins));
        _append_u32_le(frame.header, static_cast<uint32_t>(cfg_.ofdm.fft_size));
        _append_u32_le(frame.header, static_cast<uint32_t>(_ertm_delay_oversample_factor));
        _append_u32_le(frame.header, bs_payload.seq);
        _append_u32_le(frame.header, static_cast<uint32_t>(peak_index));
        _append_i64_le(frame.header, signed_shift_bins);
        _append_i32_le(frame.header, bs_payload.duti_samples);
        _append_i32_le(frame.header, tadv_samples);
        _append_double_le(frame.header, bs_payload.sample_rate);
        _append_double_le(frame.header, metric);
        _append_double_le(frame.header, rf_delay_samples);
        _append_double_le(frame.header, tau_c_samples);
        _append_double_le(frame.header, to_bs_ue_samples);
        _append_double_le(frame.header, to_ue_samples);
        _append_double_le(frame.header, to_bs_samples);

        DataSender<ErtmDebugFrameData>::DataPackage package;
        package.emplace_back(std::move(frame));
        ertm_debug_sender_.add_data(std::move(package));
    }

    // Fast-path classifier, called inline from the LDPC decode/bit-processing
    // thread. Only cheap checks happen here; sensing consumes the queued report
    // and pairs it with the downlink channel from its current frame.
    bool _handle_ertm_payload(std::vector<uint8_t>& payload, uint8_t flags) {
        if (!LdpcPacketFraming::is_ertm_timing_flags(flags)) {
            return false;
        }
        if (!cfg_.uplink.ertm_to_enable) {
            LOG_RT_WARN_HZ_M(Ertm, 1)
                << "[eRTM] TO measurement received while eRTM is disabled; dropping report";
            return true;
        }
        if (!_ertm_state.enqueue(std::move(payload))) {
            static std::atomic<uint64_t> drop_count{0};
            const uint64_t dropped = drop_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (dropped <= 20 || (dropped % 100) == 0) {
                LOG_G_WARN_M(Ertm) << "[eRTM] TO payload queue full; dropping report"
                             << ": dropped=" << dropped;
            }
        }
        return true;
    }

    bool _send_udp_output_packet(const uint8_t* data, size_t size, const char* source) {
        try {
            _udp_output_sender->send(data, size);
            return true;
        } catch (const std::exception& e) {
            LOG_G_WARN_M(DemodLdpc) << "[UE] UDP output failed; dropping decoded packet"
                                 << ": source=" << source
                                 << ", payload_bytes=" << size
                                 << ", error=" << e.what();
            return false;
        }
    }

    void _apply_uplink_arq_feedback(const ArqFeedback& ack, const char* source) {
        if (ack.direction != 1) {
            const uint64_t dropped =
                _arq_dl_feedback_invalid.fetch_add(1, std::memory_order_relaxed) + 1;
            if (dropped <= 20 || (dropped % 100) == 0) {
                LOG_G_WARN_M(ArqUl) << "[UE ARQ UL] ACK direction mismatch; dropping feedback"
                              << ": source=" << source
                              << ", direction=" << static_cast<int>(ack.direction)
                              << ", expected=1, dropped=" << dropped;
            }
            return;
        }
        if (!_uplink_tx || !_uplink_tx->arq_enabled()) {
            const uint64_t dropped =
                _arq_dl_feedback_unhandled.fetch_add(1, std::memory_order_relaxed) + 1;
            if (dropped <= 20 || (dropped % 100) == 0) {
                LOG_G_WARN_M(ArqUl) << "[UE ARQ UL] no active uplink ARQ TX; dropping ACK feedback"
                              << ": source=" << source
                              << ", dropped=" << dropped;
            }
            return;
        }
        _uplink_tx->arq_tx_window().process_ack(ack, arq_now_ms());
    }

    /**
     * @brief Request eRTM invalidation from any thread (RX/bit/process).
     *
     * Immediately marks TO unavailable and bumps a clear generation. Does NOT
     * touch the SPSC report queue or the consumer-owned report cache — the
     * actual eRTM consumer applies the wipe via @c _apply_pending_ertm_clear.
     */
    void _request_ertm_report_clear(const char* reason) {
        // Finish invalidating state(M) before publishing the new queue
        // generation. Any report stamped with the new generation therefore
        // cannot be followed by this clear wiping its timing history.
        if (_uplink_tx) {
            _uplink_tx->clear_ul_tx_timing_history();
        }
        const uint64_t gen = _ertm_state.request_clear();
        if (LOG_MOD_ON(Ertm)) {
            LOG_G_INFO_M(Ertm) << "[eRTM] clear requested"
                         << ": reason=" << reason
                         << ", clear_gen=" << gen
                         << " (TO invalidated; consumer will filter queue and clear cache)";
        }
    }

    /** Consumer-only: wipe cache when a clear request is pending. */
    bool _apply_pending_ertm_clear() {
        const ErtmAbsoluteFrameState::ClearResult clear =
            _ertm_state.apply_pending_clear();
        if (!clear.applied) {
            return false;
        }
        // Do not touch process-owned OCXO fields here (may run on sensing).
        _request_ocxo_to_reanchor();
        if (LOG_MOD_ON(Ertm)) {
            LOG_G_INFO_M(Ertm) << "[eRTM] consumer applied clear"
                         << ": clear_gen=" << clear.generation
                         << ", invalidated_cache=" << clear.invalidated_reports
                         << " (queued reports filtered by generation)";
        }
        return true;
    }

    /**
     * @brief Consistent TO publish snapshot from shared eRTM state.
     *
     * samples and pub_gen must be observed together; separate relaxed loads can
     * pair generation G with sample of G+1 and re-ingest the same sample as a
     * "new" report with a near-zero dTO slope.
     */
    using ErtmToPublishSnapshot = ErtmAbsoluteFrameState::PublishSnapshot;

    ErtmToPublishSnapshot _load_ertm_to_publish_snapshot() const {
        return _ertm_state.snapshot();
    }

    /** Any thread: ask process to reanchor OCXO Mode B TO state. */
    void _request_ocxo_to_reanchor() {
        _ocxo_to_reanchor_req.fetch_add(1, std::memory_order_acq_rel);
    }

    /** process-thread only: apply pending OCXO TO reanchor requests. */
    void _apply_pending_ocxo_to_reanchor(bool allow_state_changes) {
        const uint64_t req = _ocxo_to_reanchor_req.load(std::memory_order_acquire);
        if (req == _ocxo_to_reanchor_applied) {
            return;
        }
        _ocxo_to_pub_known = false;
        _ocxo_to_last_pub_gen = 0;
        _ocxo_to_age_frames = 0;
        if (_ocxo_discipliner) {
            _ocxo_discipliner->reanchor_to(allow_state_changes);
        }
        _ocxo_to_reanchor_applied = req;
    }

    /**
     * @brief process-thread only: apply RX-requested tracking reset.
     *
     * Stream restart runs on the RX thread and must not write latch / SFO /
     * OCXO age fields. It only bumps @_process_tracking_reset_req.
     */
    void _apply_pending_process_tracking_reset() {
        const uint64_t req =
            _process_tracking_reset_req.load(std::memory_order_acquire);
        if (req == _process_tracking_reset_applied) {
            return;
        }
        _sync_in_progress = false;
        _sync_in_progress_hold_frames = 0;
        _sync_in_progress_expect_id = 0;
        _delay_adjustment_count = 0;
        _last_delay_index_err = 0;
        _reset_count = 0;
        _preamble_bad_count = 0;
        _alignment_applied_consumed_id = 0;
        sfo_estimator.reset();
        _ocxo_to_pub_known = false;
        _ocxo_to_last_pub_gen = 0;
        _ocxo_to_age_frames = 0;
        if (_ocxo_discipliner) {
            _ocxo_discipliner->reanchor_to();
        }
        _ocxo_to_reanchor_applied =
            _ocxo_to_reanchor_req.load(std::memory_order_acquire);
        _process_tracking_reset_applied = req;
        if (LOG_MOD_ON(Recovery)) {
            LOG_G_WARN_M(Recovery)
                << "[UE recovery] process tracking reset applied req=" << req;
        }
    }

    /**
     * @brief process-thread only: resolve Alignment for SFO / eRTM / latch.
     *
     * When the one-shot Alignment frame was queue-dropped, inject applied
     * samples on the first process frame whose usrp_time is at/after the
     * applied RX timestamp so SFO and eRTM still see the window jump.
     */
    int32_t _effective_alignment_for_frame(
        int32_t frame_alignment,
        int64_t frame_usrp_time_ns,
        bool* from_sidechannel_out = nullptr)
    {
        if (from_sidechannel_out) {
            *from_sidechannel_out = false;
        }
        int32_t effective = frame_alignment;
        const uint64_t applied_id =
            _alignment_applied_id.load(std::memory_order_acquire);
        if (applied_id == 0 || applied_id == _alignment_applied_consumed_id) {
            return effective;
        }
        if (frame_alignment != 0) {
            // Frame already carries the one-shot receipt.
            _alignment_applied_consumed_id = applied_id;
            return effective;
        }
        const int64_t applied_time_ns =
            _alignment_applied_time_ns.load(std::memory_order_relaxed);
        const bool time_ok =
            (applied_time_ns < 0) ||
            (frame_usrp_time_ns < 0) ||
            (frame_usrp_time_ns >= applied_time_ns);
        if (!time_ok) {
            // Still processing pre-alignment pipeline frames.
            return 0;
        }
        effective =
            _alignment_applied_samples.load(std::memory_order_relaxed);
        _alignment_applied_consumed_id = applied_id;
        if (from_sidechannel_out) {
            *from_sidechannel_out = (effective != 0);
        }
        return effective;
    }

    bool _alignment_latch_receipt_ready(int32_t frame_alignment) const {
        if (frame_alignment != 0) {
            return true;
        }
        if (_sync_in_progress_expect_id == 0) {
            return false;
        }
        const uint64_t applied_id =
            _alignment_applied_id.load(std::memory_order_acquire);
        return applied_id == _sync_in_progress_expect_id;
    }

    using ErtmCachedReport = ErtmAbsoluteFrameState::CachedReport;

    bool _ingest_ertm_raw_payload(
        std::vector<uint8_t>&& payload,
        const AirFrameIdentity& dl_identity)
    {
        const ErtmAbsoluteFrameState::IngestResult result = _ertm_state.ingest(
            std::move(payload),
            dl_identity,
            static_cast<uint32_t>(cfg_.ofdm.fft_size));
        switch (result.status) {
        case ErtmAbsoluteFrameState::IngestStatus::Accepted:
            return true;
        case ErtmAbsoluteFrameState::IngestStatus::LegacyPayload:
            if (!_ertm_legacy_payload_warned) {
                _ertm_legacy_payload_warned = true;
                LOG_G_WARN_M(Ertm)
                    << "[eRTM] rejected legacy ERTM2/3/4 payload; "
                       "raw BS PPS diagnostics required";
            }
            break;
        case ErtmAbsoluteFrameState::IngestStatus::InvalidPayload:
            LOG_G_WARN_M(Ertm)
                << "[eRTM] invalid TO payload; dropping report: " << result.error;
            break;
        case ErtmAbsoluteFrameState::IngestStatus::MissingEpoch:
            LOG_G_WARN_M(Ertm)
                << "[eRTM] ERTM5 missing timeline epoch; dropping report"
                << ": seq=" << result.payload_seq;
            break;
        case ErtmAbsoluteFrameState::IngestStatus::Empty:
        case ErtmAbsoluteFrameState::IngestStatus::CacheRejected:
            break;
        }
        return false;
    }

    void _process_ertm_report(
        const ErtmTimingPayloadView& bs_payload,
        const UlTxTimingStateEntry& state_m,
        const AlignedVector& local_channel_freq,
        const AirFrameIdentity& dl_identity,
        int32_t a_n_samples,
        uint64_t sync_generation)
    {
        uint64_t compute_clear_gen = 0;
        if (!_ertm_state.begin_compute(compute_clear_gen)) {
            return;
        }
        if (local_channel_freq.size() != cfg_.ofdm.fft_size) {
            LOG_RT_WARN_HZ_M(Ertm, 1)
                << "[eRTM] current downlink frame has an invalid channel estimate size";
            return;
        }
        if (!dl_identity.valid || dl_identity.timeline_epoch == 0) {
            LOG_RT_WARN_HZ_M(Ertm, 1)
                << "[eRTM] current DL air-frame identity is invalid; keeping previous TO";
            return;
        }
        const uint32_t n_epoch =
            AirFrameClock::ota_epoch32(dl_identity.timeline_epoch);
        const uint64_t n_frame = dl_identity.frame_seq;
        if (bs_payload.timeline_epoch != n_epoch) {
            LOG_G_WARN_M(Ertm) << "[eRTM] epoch mismatch; keeping previous TO"
                         << ": report_epoch=0x" << std::hex << bs_payload.timeline_epoch
                         << ", dl_epoch=0x" << n_epoch << std::dec
                         << ", M=" << bs_payload.ul_measure_frame_seq
                         << ", N=" << n_frame;
            return;
        }
        if (bs_payload.ul_measure_frame_seq > n_frame) {
            // Future relative to current N — caller should not select these.
            return;
        }

        const int32_t a_m = state_m.rx_alignment_shift_samples;
        const int32_t tadv_m = state_m.tadv_samples;
        const int32_t a_n = a_n_samples;
        // Positive UL-TX alignment advances physical TX, which decreases the BS
        // observed UL delay. Transform H_UL[M] into N's Alignment coordinate.
        const int32_t alignment_delta = a_n - a_m;
        const double ul_delay_shift = -static_cast<double>(alignment_delta);

        if (!_compute_ertm_oversampled_delay_spectrum(
                bs_payload.channel_freq, ul_delay_shift, _ertm_bs_uplink_delay) ||
            !_compute_ertm_oversampled_delay_spectrum(
                local_channel_freq, 0.0, _ertm_local_delay)) {
            LOG_G_WARN_M(Ertm) << "[eRTM] failed to compute oversampled delay spectra; keeping previous TO";
            return;
        }

        int64_t signed_shift_bins = 0;
        double metric = 0.0;
        size_t peak_index = 0;
        double fractional_delta_bins = 0.0;
        const bool publish_debug = cfg_.uplink.ertm_debug_output_enabled;
        if (!_estimate_ertm_delay_shift(
                _ertm_bs_uplink_delay,
                _ertm_local_delay,
                cfg_.uplink.ertm_timing_metric,
                signed_shift_bins,
                metric,
                publish_debug ? &_ertm_correlation_spectrum : nullptr,
                &peak_index,
                &fractional_delta_bins)) {
            LOG_G_WARN_M(Ertm) << "[eRTM] failed to evaluate timing metric; keeping previous TO"
                         << ": metric=" << cfg_.uplink.ertm_timing_metric;
            return;
        }

        // tau_c uses DUTI_M / TADV_M paired with H_UL[M], not live control values.
        const double rf_delay_samples =
            cfg_.uplink.ertm_dl_rf_delay_samples + cfg_.uplink.ertm_ul_rf_delay_samples;
        const double tau_c_samples =
            rf_delay_samples -
            static_cast<double>(bs_payload.duti_samples) -
            static_cast<double>(tadv_m);
        const double shift_frac_os_bins =
            static_cast<double>(signed_shift_bins) + fractional_delta_bins;
        const double to_bs_ue_samples =
            shift_frac_os_bins / static_cast<double>(_ertm_delay_oversample_factor);
        const double to_ue_samples = 0.5 * (tau_c_samples - to_bs_ue_samples);
        const double to_bs_samples = 0.5 * (tau_c_samples + to_bs_ue_samples);
        const int64_t report_age_frames =
            static_cast<int64_t>(n_frame) -
            static_cast<int64_t>(bs_payload.ul_measure_frame_seq);
        ErtmPpsPhaseObservation pps_phase;
        if (cfg_.sync_tracking.ocxo_mode == kOCXOModePpsPhase) {
            try {
                const radio::TimeSpec ue_pps = dev_->time_last_pps();
                const int64_t ue_pps_ns =
                    ue_pps.get_full_secs() * 1'000'000'000LL +
                    static_cast<int64_t>(
                        std::llround(ue_pps.get_frac_secs() * 1.0e9));
                pps_phase = compute_ertm_pps_phase(
                    bs_payload,
                    n_frame,
                    dl_identity.ue_frame_start_time_ns,
                    ue_pps_ns,
                    cfg_.uplink.ertm_dl_rf_delay_samples,
                    to_ue_samples);
            } catch (const std::exception& e) {
                LOG_RT_WARN_HZ_M(Ertm, 1)
                    << "[eRTM] failed to read UE PPS timestamp; PPS phase unavailable: "
                    << e.what();
            }
        }
        if (std::isfinite(to_ue_samples)) {
            const ErtmAbsoluteFrameState::PublishResult publish =
                _ertm_state.publish(
                    compute_clear_gen,
                    to_ue_samples,
                    bs_payload.timeline_epoch,
                    bs_payload.ul_measure_frame_seq,
                    dl_identity,
                    a_n_samples,
                    sync_generation,
                    pps_phase.valid,
                    pps_phase.phase_error_samples,
                    pps_phase.r_bs_samples,
                    pps_phase.r_ue_samples,
                    pps_phase.bs_last_pps_ticks,
                    pps_phase.bs_t0_to_pps_ticks,
                    pps_phase.ue_last_pps_ns,
                    pps_phase.bs_tick_rate,
                    pps_phase.sample_rate);
            if (!publish.accepted) {
                LOG_G_WARN_M(Ertm)
                    << "[eRTM] discarding TO publish after clear request"
                    << ": clear_req=" << publish.request_generation
                    << ", clear_applied=" << publish.applied_generation
                    << ", seq=" << bs_payload.seq;
                return;
            }
            if (publish.had_previous) {
                const double to_ue_diff =
                    to_ue_samples - publish.previous_to_ue_samples;
                if (std::abs(to_ue_diff) > 0.1 && LOG_MOD_ON(Ertm)) {
                    LOG_G_INFO_M(Ertm) << "[eRTM] TO_UE diff"
                                 << " seq=" << bs_payload.seq
                                 << ", previous_to_ue="
                                 << publish.previous_to_ue_samples
                                 << " samples, current_to_ue=" << to_ue_samples
                                 << " samples, to_ue_diff=" << to_ue_diff
                                 << " samples";
                }
            }
        }

        if (LOG_MOD_ON(Ertm)) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3)
                << "[eRTM] TO(samples) seq=" << bs_payload.seq
                << ", epoch=0x" << std::hex << bs_payload.timeline_epoch << std::dec
                << ", M=" << bs_payload.ul_measure_frame_seq
                << ", N=" << n_frame
                << ", age_frames=" << report_age_frames
                << ", oversample=" << _ertm_delay_oversample_factor
                << ", timing_metric=" << cfg_.uplink.ertm_timing_metric
                << ", peak_refinement="
                << (cfg_.uplink.ertm_timing_metric == kErtmTimingMetricMaximumLikelihood
                        ? "parabolic3" : "centroid3")
                << ", shift_os_bins=" << signed_shift_bins
                << ", shift_fractional_delta_os_bins=" << fractional_delta_bins
                << ", shift_frac_os_bins=" << shift_frac_os_bins
                << ", shift_samples=" << to_bs_ue_samples
                << ", peak_index=" << peak_index
                << ", rf_delay_samples=" << rf_delay_samples
                << ", metric=" << std::setprecision(6) << metric << std::setprecision(3)
                << ", DUTI_M=" << bs_payload.duti_samples
                << ", TADV_M=" << tadv_m
                << ", A_M=" << a_m
                << ", A_N=" << a_n
                << ", ul_delay_shift=" << ul_delay_shift
                << ", tau_c_samples=" << tau_c_samples
                << ", TO_BS_UE_samples=" << to_bs_ue_samples
                << ", TO_UE_samples=" << to_ue_samples
                << ", TO_BS_samples=" << to_bs_samples;
            if (cfg_.sync_tracking.ocxo_mode == kOCXOModePpsPhase) {
                oss << ", r_B_samples=" << pps_phase.r_bs_samples
                    << ", r_U_samples=" << pps_phase.r_ue_samples
                    << ", bs_last_pps_ticks=" << pps_phase.bs_last_pps_ticks
                    << ", bs_t0_to_pps_ticks=" << pps_phase.bs_t0_to_pps_ticks
                    << ", ue_last_pps_ns=" << pps_phase.ue_last_pps_ns
                    << ", PPS_UE_minus_BS_samples=" << pps_phase.phase_error_samples
                    << ", PPS_phase_valid=" << (pps_phase.valid ? "true" : "false");
            }
            oss << ", rf_delay_samples_cfg=(" << cfg_.uplink.ertm_dl_rf_delay_samples
                << "+" << cfg_.uplink.ertm_ul_rf_delay_samples << ")";
            LOG_G_INFO_M(Ertm) << oss.str();
        }
        if (publish_debug) {
            // Delay-spectrum shifting uses the opposite sign of the effective
            // timing offset (SensingChannel applies the inverse phase directly).
            const double correction_to_ue_samples = -to_ue_samples;
            const double correction_to_bs_samples = -to_bs_samples;
            if (!_compute_ertm_oversampled_delay_spectrum(
                    bs_payload.channel_freq,
                    ul_delay_shift + correction_to_bs_samples,
                    _ertm_corrected_uplink_delay) ||
                !_compute_ertm_oversampled_delay_spectrum(
                    local_channel_freq,
                    correction_to_ue_samples,
                    _ertm_corrected_downlink_delay)) {
                LOG_G_WARN_M(Ertm) << "[eRTM] failed to compute TO-corrected debug spectra; dropping debug frame";
                return;
            }
            _publish_ertm_debug_frame(
                bs_payload,
                _ertm_bs_uplink_delay,
                _ertm_local_delay,
                _ertm_correlation_spectrum,
                _ertm_corrected_uplink_delay,
                _ertm_corrected_downlink_delay,
                peak_index,
                signed_shift_bins,
                metric,
                tadv_m,
                rf_delay_samples,
                tau_c_samples,
                to_bs_ue_samples,
                to_ue_samples,
                to_bs_samples);
        }
    }

    /**
     * @brief Drop published TO without wiping report cache (consumer-only).
     *
     * Used when Alignment force-remeasure cannot form a usable absolute pair:
     * keeping the previous TO would apply an A_N-stale absolute delay. Sensing
     * falls back to los_tracking until a fresh pair succeeds.
     */
    void _invalidate_ertm_to_available(const char* reason) {
        // May run on sensing thread — only touch shared eRTM atomics here.
        // OCXO reanchor is requested for the process thread.
        _ertm_state.invalidate_to();
        _request_ocxo_to_reanchor();
        LOG_RT_WARN_HZ_M(Ertm, 2)
            << "[eRTM] TO marked unavailable"
            << ": reason=" << reason
            << " (sensing falls back until a fresh absolute pair succeeds)";
    }

    void _update_ertm_measurement_for_downlink_frame(
        const AlignedVector& local_channel_freq,
        bool force_remeasure,
        int alignment_samples,
        const AirFrameIdentity& dl_identity,
        int32_t a_n_samples,
        uint64_t sync_generation)
    {
        if (!cfg_.uplink.ertm_to_enable) {
            return;
        }

        // Sole consumer of the SPSC queue and report cache.
        _apply_pending_ertm_clear();

        bool received_new_report = false;
        size_t queued_reports = 0;
        size_t stale_generation_reports = 0;
        ErtmAbsoluteFrameState::QueuedPayload queued;
        while (_ertm_state.try_pop(queued)) {
            // A clear may race the drain. Apply it before classifying this item;
            // reports stamped after the clear survive, older ones are discarded.
            if (_apply_pending_ertm_clear()) {
                received_new_report = false;
            }
            if (queued.clear_generation !=
                _ertm_state.applied_clear_generation()) {
                ++stale_generation_reports;
                ++queued_reports;
                continue;
            }
            if (_ingest_ertm_raw_payload(std::move(queued.raw), dl_identity)) {
                received_new_report = true;
            }
            ++queued_reports;
        }
        // If clear landed after the final pop/ingest, wipe the stale cache before
        // report selection. A post-clear item remains queued for the next frame.
        if (_apply_pending_ertm_clear()) {
            received_new_report = false;
        }
        if (queued_reports > 1 && LOG_MOD_ON(Ertm)) {
            LOG_G_INFO_M(Ertm) << "[eRTM] drained queued BS UL reports"
                         << ": count=" << queued_reports
                         << ", stale_generation=" << stale_generation_reports
                         << ", cache_size=" << _ertm_state.cache_size();
        }

        if (!dl_identity.valid || dl_identity.timeline_epoch == 0) {
            if (force_remeasure || received_new_report) {
                LOG_RT_WARN_HZ_M(Ertm, 1)
                    << "[eRTM] DL identity invalid; not publishing TO"
                    << ": force_remeasure=" << force_remeasure
                    << ", received_new_report=" << received_new_report;
            }
            // Alignment remeasure cannot form a new absolute pair without N's
            // identity; do not keep applying a TO from the previous coordinate.
            if (force_remeasure) {
                _invalidate_ertm_to_available("force_remeasure_invalid_dl_identity");
            }
            return;
        }

        const ErtmAbsoluteFrameState::StateMSelection selection =
            _ertm_state.select_with_state_m(
                dl_identity,
                [this](uint32_t epoch, uint64_t frame, UlTxTimingStateEntry& state) {
                    return _uplink_tx &&
                        _uplink_tx->query_ul_tx_timing_state(epoch, frame, state);
                },
                [&dl_identity](const ErtmTimingPayloadView& report) {
                    LOG_G_WARN_M(Ertm)
                        << "[eRTM] UL-TX timing history miss; invalidating report and trying older M"
                        << ": epoch=0x" << std::hex << report.timeline_epoch
                        << std::dec
                        << ", M=" << report.ul_measure_frame_seq
                        << ", N=" << dl_identity.frame_seq;
                });
        ErtmCachedReport* selected = selection.report;
        const UlTxTimingStateEntry& state_m = selection.state_m;
        const size_t future_count = selection.future_count;
        const size_t history_miss_count = selection.history_miss_count;
        if (selected == nullptr) {
            if (force_remeasure) {
                LOG_RT_WARN_HZ_M(Ertm, 1)
                    << "[eRTM] Alignment requires a TO remeasurement, but no usable BS UL report is cached"
                    << " (future_pending=" << future_count
                    << ", history_miss=" << history_miss_count << ")";
                // Previous TO is A_N-stale after Alignment; fall back until a
                // fresh absolute pair is available.
                _invalidate_ertm_to_available("force_remeasure_no_usable_pair");
            }
            return;
        }
        // Auto-activate a previously-future report once N >= M, even without a
        // new queue push or Alignment force remeasure.
        const bool unused_ready_report = _ertm_state.is_unused(*selected);
        if (!received_new_report && !force_remeasure && !unused_ready_report) {
            return;
        }

        if (force_remeasure && LOG_MOD_ON(Ertm)) {
            LOG_G_INFO_M(Ertm) << "[eRTM] forced TO remeasurement after Alignment"
                         << " alignment=" << alignment_samples << " samples"
                         << ", A_N=" << a_n_samples
                         << ", deferred_after_sensing_drop="
                         << (alignment_samples == 0 ? "true" : "false")
                         << ", M=" << selected->view.ul_measure_frame_seq
                         << ", N=" << dl_identity.frame_seq
                         << ", future_pending=" << future_count;
        } else if (unused_ready_report && !received_new_report && LOG_MOD_ON(Ertm)) {
            LOG_G_INFO_M(Ertm) << "[eRTM] activating previously-future UL report"
                         << ": M=" << selected->view.ul_measure_frame_seq
                         << ", N=" << dl_identity.frame_seq
                         << ", future_pending=" << future_count;
        }
        _process_ertm_report(
            selected->view,
            state_m,
            local_channel_freq,
            dl_identity,
            a_n_samples,
            sync_generation);
    }

    float _select_sensing_delay_offset(float tracking_delay_offset, float user_delay_offset) {
        float base_delay_offset = tracking_delay_offset;
        if (cfg_.sensing.delay_correction_mode == kSensingDelayCorrectionModeErtmAbsolute) {
            const ErtmToPublishSnapshot snap = _load_ertm_to_publish_snapshot();
            const bool use_ertm =
                snap.available &&
                snap.clear_gen == snap.request_gen &&
                std::isfinite(snap.to_ue_samples);
            if (use_ertm) {
                // SensingChannel::delay_offset is a frequency-domain inverse
                // phase correction, so positive TO_UE removes positive UE delay.
                base_delay_offset = static_cast<float>(snap.to_ue_samples);
            } else {
                LOG_RT_WARN_HZ_M(Ertm, 1) << "[eRTM] sensing.sensing_delay_correction_mode=ertm_absolute "
                    << "but no valid TO_UE estimate is available yet; using los_tracking delay";
            }
        }
        return base_delay_offset + user_delay_offset;
    }

    void _log_arq_profile_if_due(const char* reason, int64_t now_ms) {
        if (!LOG_MOD_ON(Arq)) {
            return;
        }
        const uint64_t now = static_cast<uint64_t>(std::max<int64_t>(now_ms, 0));
        uint64_t last = _arq_profile_last_log_ms.load(std::memory_order_relaxed);
        if (last != 0 && now - last < 1000) {
            return;
        }
        if (!_arq_profile_last_log_ms.compare_exchange_strong(
                last, now, std::memory_order_relaxed, std::memory_order_relaxed)) {
            return;
        }

        const bool uplink_tx_present = static_cast<bool>(_uplink_tx);
        bool feedback_carrier_enabled = false;
        uint64_t tx_feedback_injected = 0;
        uint64_t tx_feedback_dropped_disabled = 0;
        uint64_t tx_feedback_drained = 0;
        size_t tx_feedback_pending = 0;
        if (_uplink_tx) {
            feedback_carrier_enabled = _uplink_tx->arq_feedback_enabled();
            tx_feedback_injected = _uplink_tx->arq_feedback_injected_count();
            tx_feedback_dropped_disabled =
                _uplink_tx->arq_feedback_dropped_disabled_count();
            tx_feedback_drained = _uplink_tx->arq_feedback_drained_count();
            tx_feedback_pending = _uplink_tx->arq_feedback_pending_count();
        }

        std::ostringstream oss;
        oss << "[UE ARQ PROFILE] reason=" << reason
            << "\n  config: dl_arq_enabled=" << (_arq_enabled ? "yes" : "no")
            << " uplink_tx_present=" << (uplink_tx_present ? "yes" : "no")
            << " feedback_carrier=" << (feedback_carrier_enabled ? "yes" : "no")
            << "\n  dl_rx: data_seen=" << _arq_dl_data_seen.load(std::memory_order_relaxed)
            << " accepted=" << _arq_dl_data_accepted.load(std::memory_order_relaxed)
            << " duplicates="
            << _arq_dl_data_duplicates.load(std::memory_order_relaxed)
            << " ack_base=" << _dl_arq_rx.ack_base()
            << " ack_bitmap=0x" << std::hex << _dl_arq_rx.ack_bitmap() << std::dec
            << " rx_accepted_total=" << _dl_arq_rx.accepted_count()
            << " rx_dup_total=" << _dl_arq_rx.dup_count()
            << " rx_skip_total=" << _dl_arq_rx.skip_count()
            << "\n  dl_feedback: seen="
            << _arq_dl_feedback_seen.load(std::memory_order_relaxed)
            << " valid="
            << _arq_dl_feedback_valid.load(std::memory_order_relaxed)
            << " invalid="
            << _arq_dl_feedback_invalid.load(std::memory_order_relaxed)
            << " unhandled="
            << _arq_dl_feedback_unhandled.load(std::memory_order_relaxed)
            << "\n  ack_tx: generated="
            << _arq_dl_ack_generated.load(std::memory_order_relaxed)
            << " injected="
            << _arq_dl_ack_injected.load(std::memory_order_relaxed)
            << " no_uplink_tx="
            << _arq_dl_ack_no_uplink_tx.load(std::memory_order_relaxed)
            << " inject_failed="
            << _arq_dl_ack_inject_failed.load(std::memory_order_relaxed)
            << "\n  uplink_feedback_queue: injected=" << tx_feedback_injected
            << " drained=" << tx_feedback_drained
            << " pending=" << tx_feedback_pending
            << " dropped_disabled=" << tx_feedback_dropped_disabled;
        LOG_G_INFO_M(Arq) << oss.str();
    }

    // Noise/LLR estimation related
    double _noise_var{0.5};              // Complex noise power E[|n|^2] initial value (assume 0.25 per dimension)
    double _llr_scale{2.0};              // LLR scaling factor (updated based on noise variance)
    std::atomic<float> _llr_scale_snapshot{2.0f};
    double _snr_linear{1.0};             // Es/N0 Linear value
    double _snr_db{0.0};                 // Es/N0 dB
    double _llr_snr_linear_filtered{1.0};
    bool _llr_snr_filter_initialized{false};
    
    // Object pools for memory reuse (eliminates per-frame memory allocations)
    ObjectPool<RxFrame> _rx_frame_pool;           // Pool for RX frame buffers
    ObjectPool<AlignedFloatVector> _llr_pool;     // Pool for LLR data (float path)
    ObjectPool<LDPCCodec::AlignedShortVector> _llr_pool_i16;  // Pool for LLR data (int16 path)
    ObjectPool<SensingFrame> _sensing_frame_pool; // Pool for sensing frames
    std::vector<std::unique_ptr<CpuDemodSlot>> _cpu_demod_slots;
    std::vector<std::unique_ptr<CpuDemodWorkerContext>> _cpu_demod_contexts;
    std::vector<std::unique_ptr<CpuLdpcSlot>> _cpu_ldpc_slots;
    std::vector<std::unique_ptr<CpuLdpcWorkerContext>> _cpu_ldpc_contexts;

    // Core computation classes (hardware-independent)
    HardwareRxAgc _rx_agc;
    SyncSearchRxGainSweep _sync_search_gain_sweep;
    uint32_t _reset_hold_frames = 1;
    double _rx_gain_min_db = 0.0;
    double _rx_gain_max_db = 0.0;
    CfoErrorEwma _cfo_ewma;
    NormalCfoQualityGate _ocxo_cfo_quality_gate;
    size_t _ocxo_log_counter = 0;
    // eRTM TO age tracking for OCXO Mode B (frames since last publication gen).
    bool _ocxo_to_pub_known = false;
    uint64_t _ocxo_to_last_pub_gen = 0;
    uint64_t _ocxo_to_age_frames = 0;
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
        _cfo_symbol_skip_mask = build_cfo_tracking_skip_mask(cfg_, _duplex_layout);
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
        CpuDemodWorkerContext& ctx,
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
                _interpolated_channel_for_symbol(ctx, actual_symbol, H_est);
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

    const AlignedVector& _interpolated_channel_for_symbol(
        CpuDemodWorkerContext& ctx,
        int actual_symbol,
        const AlignedVector& fallback_H)
    {
        if (ctx.channel_anchor_symbols.size() < 2) {
            return fallback_H;
        }
        const auto upper = std::upper_bound(
            ctx.channel_anchor_symbols.begin(),
            ctx.channel_anchor_symbols.end(),
            actual_symbol);
        if (upper == ctx.channel_anchor_symbols.begin()) {
            return fallback_H;
        }
        if (upper == ctx.channel_anchor_symbols.end()) {
            return ctx.channel_anchor_h.back();
        }
        const size_t hi = static_cast<size_t>(
            std::distance(ctx.channel_anchor_symbols.begin(), upper));
        const size_t lo = hi - 1;
        const int x0 = ctx.channel_anchor_symbols[lo];
        const int x1 = ctx.channel_anchor_symbols[hi];
        const float denom = static_cast<float>(x1 - x0);
        const float t = (denom > 0.0f)
            ? (static_cast<float>(actual_symbol - x0) / denom)
            : 0.0f;
        const auto& h0 = ctx.channel_anchor_h[lo];
        const auto& h1 = ctx.channel_anchor_h[hi];
        ctx.midframe_interp_h.resize(h0.size());
        #pragma omp simd simdlen(16)
        for (size_t j = 0; j < h0.size(); ++j) {
            ctx.midframe_interp_h[j] = h0[j] * (1.0f - t) + h1[j] * t;
        }
        return ctx.midframe_interp_h;
    }

    bool _fit_symbol_pilot_phase(
        CpuDemodWorkerContext& ctx,
        const AlignedVector& symbol,
        const AlignedVector& H_base,
        float& beta_out,
        float& alpha_out
    ) {
        ctx.tracking_pilot_indices.clear();
        ctx.tracking_phase.clear();
        ctx.tracking_weights.clear();

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

            ctx.tracking_pilot_indices.push_back(_actual_subcarrier_indices[k]);
            ctx.tracking_phase.push_back(std::arg(residual));
            ctx.tracking_weights.push_back(std::max(denom_power, min_weight));
        }

        if (ctx.tracking_pilot_indices.size() < 2) {
            return false;
        }

        unwrap(ctx.tracking_phase);
        auto [beta, alpha] = weightedlinearRegression(
            ctx.tracking_pilot_indices,
            ctx.tracking_phase,
            ctx.tracking_weights);
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
            LOG_G_WARN_M(Config) << "Failed to append UE measurement row to "
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

    radio::IDevicePtr _make_device() {
        radio::DeviceConfig dcfg;
        dcfg.backend = cfg_.radio.radio_backend;
        dcfg.device_args = cfg_.usrp_device.device_args;
        dcfg.clock_source = cfg_.clock_time.clock_source;
        dcfg.time_source = resolve_ue_time_source(cfg_);
        dcfg.sim_session = cfg_.simulation.session;
        dcfg.sim_tick_rate = cfg_.rf_sampling.sample_rate;
        dcfg.sim_center_freq = cfg_.downlink.center_freq;
        dcfg.sim_predictive_delay = cfg_.sync_tracking.predictive_delay;
        dcfg.xr1710g = cfg_.radio.xr1710g;
        return radio::make_device(dcfg);
    }

    // Backend-independent radio init. The real radio tunes/gains a USRP; the
    // simulator attaches RX (and uplink TX) to the hub's shared-memory rings.
    // Hardware-only behavior is gated on IDevice capability queries.
    void init_radio() {
        const bool is_sim = radio_is_sim(cfg_);
        validate_xr1710g_ue_config(cfg_);
        dev_ = _make_device();
        dev_->set_rx_rate(cfg_.rf_sampling.sample_rate);
        dev_->set_rx_bandwidth(cfg_.rf_sampling.bandwidth, cfg_.downlink.rx_channel);
        dev_->set_rx_freq_correction(0.0);
        current_rx_tune_ = dev_->set_rx_freq(radio::TuneRequest(cfg_.downlink.center_freq), cfg_.downlink.rx_channel);
        tune_initialized_ = true;
        LOG_G_INFO_M(Radio) << "Actual RX RF Freq: " << format_freq_hz(current_rx_tune_.actual_rf_freq)
                     << " Hz, DSP: " << format_freq_hz(current_rx_tune_.actual_dsp_freq)
                     << " Hz";

        if (dev_->supports(radio::Capability::HardwareGain)) {
            const radio::GainRange gain_range = dev_->get_rx_gain_range(cfg_.downlink.rx_channel);
            _rx_gain_min_db = gain_range.start;
            _rx_gain_max_db = gain_range.stop;
            const double initial_rx_gain_db = std::clamp(cfg_.rf_sampling.rx_gain, _rx_gain_min_db, _rx_gain_max_db);
            if (initial_rx_gain_db != cfg_.rf_sampling.rx_gain) {
                LOG_G_WARN_M(Radio) << "Configured rx_gain=" << cfg_.rf_sampling.rx_gain
                             << " dB is outside hardware range ["
                             << _rx_gain_min_db << ", " << _rx_gain_max_db
                             << "] dB. Clamping to " << initial_rx_gain_db << " dB.";
            }
            dev_->set_rx_gain(initial_rx_gain_db, cfg_.downlink.rx_channel);
            _rx_agc.initialize(initial_rx_gain_db, _rx_gain_min_db, _rx_gain_max_db);
            _sync_search_gain_sweep.initialize(initial_rx_gain_db, _rx_gain_min_db, _rx_gain_max_db);
            LOG_G_INFO_M(Radio) << "RX gain range: [" << _rx_gain_min_db << ", " << _rx_gain_max_db
                         << "] dB, initial gain: " << initial_rx_gain_db << " dB";
        } else {
            // No hardware gain (simulation): benign zero range so AGC/sweep stay inert.
            _rx_gain_min_db = 0.0;
            _rx_gain_max_db = 0.0;
            _rx_agc.initialize(0.0, _rx_gain_min_db, _rx_gain_max_db);
            _sync_search_gain_sweep.initialize(0.0, _rx_gain_min_db, _rx_gain_max_db);
        }

        radio::StreamArgs args("fc32", cfg_.downlink.rx_wire_format);
        args.args["block_id"] = "radio";
        args.args["sim_suffix"] = "rx.comm";
        args.channels = {cfg_.downlink.rx_channel};
        rx_stream_ = dev_->get_rx_stream(args);

        if (is_sim) {
            LOG_G_INFO_M(Radio) << "RX radio backend: SIMULATION (session='" << cfg_.simulation.session
                         << "', no USRP).";
        }

        if (cfg_.radio.radio_backend == "xr1710g") {
            LOG_G_INFO_M(Radio) << "RX radio backend: XR1710G, shm='"
                << cfg_.radio.xr1710g.shm_name << "', source="
                << cfg_.radio.xr1710g.center_freq << " Hz, "
                << cfg_.rf_sampling.sample_rate << " samples/s; local sample timestamps";
        }

        if (uplink_enabled(cfg_)) {
            _init_uplink_tx();
        }
        const std::string time_source = resolve_ue_time_source(cfg_);
        if (dev_->supports(radio::Capability::PpsTimeSync) &&
            !is_sim && radio::clock_source_enables_pps_alignment(
                cfg_.clock_time.clock_source)) {
            LOG_G_INFO_M(Radio)
                << "Waiting for PPS with clock_source='"
                << cfg_.clock_time.clock_source
                << "', time_source='" << time_source
                << "', then aligning UE USRP timekeeper...";
            const auto alignment = radio::align_device_times_to_next_pps(
                {{"UE", dev_, time_source}});
            const auto& snapshot = alignment.devices.front();
            LOG_G_INFO_M(Radio)
                << "Aligned UE USRP timekeeper to PPS edge"
                << ": time_last_pps_ticks=" << snapshot.last_pps_ticks
                << ", time_now_ticks=" << snapshot.now_ticks
                << ", tick_rate=" << snapshot.tick_rate;
        }
    }

    void _init_uplink_tx() {
        const bool is_sim = radio_is_sim(cfg_);
        // Full-duplex uplink TX on the UE device. TDD: same carrier as RX.
        // FDD: the uplink carrier (duplex.ul_center_freq).
        const double ul_freq = (cfg_.uplink.duplex.mode == DuplexMode::FDD &&
                                cfg_.uplink.duplex.ul_center_freq > 0.0)
            ? cfg_.uplink.duplex.ul_center_freq : cfg_.downlink.center_freq;
        dev_->set_tx_rate(cfg_.rf_sampling.sample_rate);
        current_ul_tx_tune_ = dev_->set_tx_freq(radio::TuneRequest(ul_freq), cfg_.uplink.tx_channel);
        if (dev_->supports(radio::Capability::HardwareGain)) {
            const radio::GainRange tx_gain_range = dev_->get_tx_gain_range(cfg_.uplink.tx_channel);
            _uplink_tx_gain_min_db = tx_gain_range.start;
            _uplink_tx_gain_max_db = tx_gain_range.stop;
            _uplink_tx_gain_restore_db = std::clamp(
                cfg_.uplink.tx_gain,
                _uplink_tx_gain_min_db,
                _uplink_tx_gain_max_db);
            _uplink_tx_gain_range_initialized = true;
            if (_uplink_tx_gain_restore_db != cfg_.uplink.tx_gain) {
                LOG_G_WARN_M(Radio) << "Configured tx_gain=" << cfg_.uplink.tx_gain
                             << " dB is outside hardware range ["
                             << _uplink_tx_gain_min_db << ", "
                             << _uplink_tx_gain_max_db
                             << "] dB. Clamping to "
                             << _uplink_tx_gain_restore_db << " dB.";
            }
            dev_->set_tx_gain(_uplink_tx_gain_restore_db, cfg_.uplink.tx_channel);
        }
        dev_->set_tx_bandwidth(cfg_.rf_sampling.bandwidth, cfg_.uplink.tx_channel);

        radio::StreamArgs tx_args("fc32", cfg_.uplink.wire_format_tx);
        tx_args.args["sim_suffix"] = "ul.tx";
        tx_args.channels = {cfg_.uplink.tx_channel};
        _uplink_tx = std::make_unique<UplinkTxEngine>(cfg_);
        _uplink_tx->set_tx_stream(dev_->get_tx_stream(tx_args));
        _uplink_tx->timing_advance().store(cfg_.uplink.ue_timing_advance, std::memory_order_relaxed);
        if (LOG_MOD_ON(UlTx)) {
            LOG_G_INFO_M(UlTx) << "[UL-TX] uplink transmit enabled" << (is_sim ? " (sim ul.tx)" : "")
                         << " on TX ch " << cfg_.uplink.tx_channel
                         << " @ " << format_freq_hz(ul_freq) << " Hz, "
                         << _uplink_tx->uplink_config().ofdm.num_symbols << " UL symbols/frame, "
                         << "zc_root=" << _uplink_tx->uplink_config().ofdm.zc_root;
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
                LOG_G_INFO_M(Sensing) << "Ignoring SKIP command in backend sensing processing mode";
                return;
            }
            if (compact_mask_mode && !compact_mask_fft_controls_supported) {
                LOG_G_INFO_M(Sensing) << "Ignoring SKIP command in compact_mask sensing mode: "
                             << (compact_mask_reason.empty() ? "mask is not local-DD compatible" : compact_mask_reason);
                return;
            }
            const bool new_skip = (value != 0);
            _schedule_shared_sensing_update([new_skip](SharedSensingRuntime& cfg) {
                cfg.skip_sensing_fft = new_skip;
            });
            LOG_G_INFO_M(Sensing) << "Received skip sensing FFT command: " << value;
        });

        _control_handler.register_command("FDEC", [this](int32_t value) {
            if (value < 1 || value > 1000) {
                LOG_G_WARN() << "Ignoring FDEC: expected 1..1000";
                return;
            }
            _schedule_shared_sensing_update([value](SharedSensingRuntime& cfg) {
                cfg.frame_decimation = static_cast<uint32_t>(value);
            });
            LOG_G_INFO() << "Sensing frame decimation requested: " << value;
        });

        _control_handler.register_command("STRD", [this, compact_mask_mode](int32_t value) {
            if (compact_mask_mode) {
                LOG_G_INFO_M(Sensing) << "Ignoring STRD command in compact_mask sensing mode: stride is defined by sensing_mask_blocks";
                return;
            }
            const size_t stride = value <= 0 ? 1 : static_cast<size_t>(value);
            const std::string stride_error = dense_sensing_stride_cfo_training_error(
                cfg_,
                stride,
                "Runtime STRD command");
            if (!stride_error.empty()) {
                LOG_G_WARN_M(Sensing) << "Ignoring STRD command: " << stride_error;
                return;
            }
            _schedule_shared_sensing_update([stride](SharedSensingRuntime& cfg) {
                cfg.sensing_symbol_stride = stride;
            });
            LOG_G_INFO_M(Sensing) << "Set sensing stride to: " << stride;
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
            LOG_G_INFO_M(UlTx) << "Received alignment command: " << adjusted_value << " samples";
        });

        // Timing Advance — pulls the UE uplink TX window earlier on the shared
        // clock so it lands aligned at the BS. Runtime-adjustable.
        _control_handler.register_command("TADV", [this](int32_t value) {
            // Rate-limit so rapid repeated commands don't thrash the framing.
            const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (now - _last_tadv_ns.load(std::memory_order_relaxed) < 50'000'000) {
                LOG_G_WARN_M(UlTx) << "TADV rate-limited (<50ms since last); ignored " << value;
                return;
            }
            _last_tadv_ns.store(now, std::memory_order_relaxed);
            _ue_timing_advance.store(value, std::memory_order_relaxed);
            if (_uplink_tx) {
                _uplink_tx->timing_advance().store(value, std::memory_order_relaxed);
            }
            LOG_G_INFO_M(UlTx) << "TADV (Timing Advance) set to " << value << " samples";
        });
        _control_handler.register_request("TADV", [this](
            int32_t, const ControlCommandHandler::ControlPeer& peer) {
            _control_handler.send_control_status(
                peer, "TADV", _ue_timing_advance.load(std::memory_order_relaxed));
        });

        _control_handler.register_command("MTI ", [this, compact_mask_mode, compact_mask_fft_controls_supported, compact_mask_reason](int32_t value) {
            if (compact_mask_mode && !compact_mask_fft_controls_supported) {
                LOG_G_INFO_M(Sensing) << "Ignoring MTI command in compact_mask sensing mode: "
                             << (compact_mask_reason.empty() ? "mask is not local-DD compatible" : compact_mask_reason);
                return;
            }
            const bool enable = (value != 0);
            _schedule_shared_sensing_update([enable](SharedSensingRuntime& cfg) {
                cfg.enable_mti = enable;
            });
            LOG_G_INFO_M(Sensing) << "MTI " << (enable ? "Enabled" : "Disabled");
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
                LOG_G_WARN_M(Config) << "MRST ignored: measurement mode disabled";
                return;
            }
            if (value <= 0) {
                LOG_G_WARN_M(Config) << "MRST ignored: invalid epoch id " << value;
                return;
            }
            _switch_measurement_epoch(static_cast<uint32_t>(value));
            LOG_G_INFO_M(Config) << "Measurement epoch reset to " << value;
        });

        _control_handler.register_command("CALB", [this](int32_t value) {
            if (!_bistatic_sensing_channel) {
                LOG_G_WARN_M(Sensing) << "CALB ignored: bistatic sensing channel is not initialized";
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
                    1u, 0x1u, false, 0.0,
                    snapshot.sensing_symbol_stride, snapshot.frame_decimation);
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
            LOG_G_INFO_M(Sensing) << "Bistatic sensing disabled by config.";
            return;
        }

        const CompactSensingMaskAnalysis compact_mask_analysis = analyze_compact_sensing_mask(cfg_);
        _shared_sensing_cfg.frame_decimation = cfg_.sensing.frame_decimation;
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
            [this](const std::string&, int) {
                if (!_control_handler.send_heartbeat_to_last_peer()) {
                    _control_handler.send_heartbeat();
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
            cfg_.network_output.udp_egress_pacer,
            LOG_MOD_ON(UdpEgressProfiling));

        // ARQ: configure DL RX window if enabled
        _arq_enabled = cfg_.network_output.arq_enabled;
        if (_arq_enabled) {
            _dl_arq_rx.configure(cfg_.network_output);
            _dl_arq_rx.set_direction(0); // downlink direction
            LOG_G_INFO_M(ArqDl) << "[UE ARQ] DL RX enabled: window="
                         << cfg_.network_output.arq_window_packets
                         << " ordered=" << cfg_.network_output.arq_ordered_delivery;
        }
    }

    void adjust_rx_freq(double detected_offset, bool reset = false) {
        // Create new tune request
        radio::TuneRequest new_tune_req;
        new_tune_req.target_freq = current_rx_tune_.actual_rf_freq + detected_offset;
        new_tune_req.rf_freq = current_rx_tune_.actual_rf_freq; // Keep LO unchanged
        new_tune_req.dsp_freq = current_rx_tune_.actual_dsp_freq + detected_offset; // Update DSP only
        new_tune_req.rf_freq_policy = radio::TunePolicy::Manual;
        new_tune_req.dsp_freq_policy = radio::TunePolicy::Manual;
        if (reset) {
            // Reset tune request
            new_tune_req.target_freq = cfg_.downlink.center_freq;
            new_tune_req.dsp_freq = 0.0; // Reset DSP frequency
        }
        // Apply new tune (update DSP only, fast and does not affect LO). On the
        // simulator backend set_rx_freq applies the comm correction internally and
        // returns a synthetic tune result; on real hardware it retunes the DSP.
        current_rx_tune_ = dev_->set_rx_freq(new_tune_req, cfg_.downlink.rx_channel);
        if (dev_->supports(radio::Capability::RfDspTune)) {
            _retune_uplink_tx_from_rx_correction(current_rx_tune_.actual_dsp_freq);
        }
    }

    void _retune_uplink_tx_from_rx_correction(double rx_dsp_correction_hz) {
        if (!_uplink_tx || !dev_->supports(radio::Capability::RfDspTune)) {
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
        radio::TuneRequest tx_tune_req;
        tx_tune_req.target_freq = ul_base_freq + tx_target_correction_hz;
        tx_tune_req.rf_freq = ul_base_freq;
        tx_tune_req.dsp_freq = tx_dsp_correction_hz;
        tx_tune_req.rf_freq_policy = radio::TunePolicy::Manual;
        tx_tune_req.dsp_freq_policy = radio::TunePolicy::Manual;
        current_ul_tx_tune_ = dev_->set_tx_freq(tx_tune_req, cfg_.uplink.tx_channel);
        if (LOG_MOD_ON(UlTx)) {
            LOG_RT_INFO_M(UlTx) << "[UL-TX] adjusted TX frequency with RX CFO correction: base="
                          << format_freq_hz(ul_base_freq)
                          << " Hz, target correction=" << format_freq_hz(tx_target_correction_hz)
                          << " Hz, requested TX DSP=" << format_freq_hz(tx_dsp_correction_hz)
                          << " Hz, actual RF=" << format_freq_hz(current_ul_tx_tune_.actual_rf_freq)
                          << " Hz, DSP=" << format_freq_hz(current_ul_tx_tune_.actual_dsp_freq)
                          << " Hz";
        }
    }
    // eRTM plans only; the per-worker demod FFT plans are created in
    // _init_cpu_demod_workers(). Both run in the constructor because FFTW
    // planning is not thread-safe.
    void prepare_fftw() {
        if (cfg_.uplink.ertm_to_enable) {
            _ertm_delay_oversample_factor = std::max<size_t>(1, cfg_.uplink.ertm_delay_oversample_factor);
            if (cfg_.ofdm.fft_size > std::numeric_limits<size_t>::max() / _ertm_delay_oversample_factor) {
                throw std::runtime_error("eRTM oversampled FFT size overflows size_t");
            }
            _ertm_oversampled_fft_size = cfg_.ofdm.fft_size * _ertm_delay_oversample_factor;
            if (_ertm_oversampled_fft_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("eRTM oversampled FFT size exceeds FFTW int plan limit");
            }
            _ertm_os_ifft_in.resize(_ertm_oversampled_fft_size);
            _ertm_os_ifft_out.resize(_ertm_oversampled_fft_size);
            _ertm_os_ifft_plan = fftwf_plan_dft_1d(
                static_cast<int>(_ertm_oversampled_fft_size),
                reinterpret_cast<fftwf_complex*>(_ertm_os_ifft_in.data()),
                reinterpret_cast<fftwf_complex*>(_ertm_os_ifft_out.data()),
                FFTW_BACKWARD,
                FFTW_MEASURE);

            _ertm_corr_a.resize(_ertm_oversampled_fft_size);
            _ertm_corr_b.resize(_ertm_oversampled_fft_size);
            _ertm_corr_a_freq.resize(_ertm_oversampled_fft_size);
            _ertm_corr_b_freq.resize(_ertm_oversampled_fft_size);
            _ertm_corr_out.resize(_ertm_oversampled_fft_size);
            // Per-compute delay/correlation outputs — sized once, reused forever.
            _ertm_bs_uplink_delay.resize(_ertm_oversampled_fft_size);
            _ertm_local_delay.resize(_ertm_oversampled_fft_size);
            _ertm_corrected_uplink_delay.resize(_ertm_oversampled_fft_size);
            _ertm_corrected_downlink_delay.resize(_ertm_oversampled_fft_size);
            _ertm_correlation_spectrum.assign(_ertm_oversampled_fft_size, 0.0f);
            // Pre-size shared report-cache channel buffers for unpack reuse.
            _ertm_state.prepare_channel_buffers(cfg_.ofdm.fft_size);
            _ertm_corr_fft_a_plan = fftwf_plan_dft_1d(
                static_cast<int>(_ertm_oversampled_fft_size),
                reinterpret_cast<fftwf_complex*>(_ertm_corr_a.data()),
                reinterpret_cast<fftwf_complex*>(_ertm_corr_a_freq.data()),
                FFTW_FORWARD,
                FFTW_MEASURE);
            _ertm_corr_fft_b_plan = fftwf_plan_dft_1d(
                static_cast<int>(_ertm_oversampled_fft_size),
                reinterpret_cast<fftwf_complex*>(_ertm_corr_b.data()),
                reinterpret_cast<fftwf_complex*>(_ertm_corr_b_freq.data()),
                FFTW_FORWARD,
                FFTW_MEASURE);
            _ertm_corr_ifft_plan = fftwf_plan_dft_1d(
                static_cast<int>(_ertm_oversampled_fft_size),
                reinterpret_cast<fftwf_complex*>(_ertm_corr_a_freq.data()),
                reinterpret_cast<fftwf_complex*>(_ertm_corr_out.data()),
                FFTW_BACKWARD,
                FFTW_MEASURE);
        }
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
            LOG_G_INFO_M(UlTx) << "[UL-TX] self-channel debug streams: channel="
                         << cfg_.network_output.uplink_self_channel_ip << ':' << cfg_.network_output.uplink_self_channel_port
                         << ", pdf=" << cfg_.network_output.uplink_self_pdf_ip << ':'
                         << cfg_.network_output.uplink_self_pdf_port;
            // Full-frame self-ZC matched filter (diagnostic; see member comment).
            _uplink_self_scan_processor = std::make_unique<SyncProcessor>(
                cfg_.samples_per_frame(),
                cfg_.ofdm.fft_size,
                cfg_.ofdm.cp_length,
                _uplink_self_zc_freq);
            if (uplink_self_scan_spectrum_enabled(cfg_)) {
                uplink_self_scan_pub_ = std::make_unique<ZmqByteSender>(
                    cfg_.network_output.uplink_self_scan_ip,
                    static_cast<uint16_t>(cfg_.network_output.uplink_self_scan_port));
                LOG_G_INFO_M(UlTx) << "[UL-TX] self-scan spectrum stream: "
                             << cfg_.network_output.uplink_self_scan_ip << ':'
                             << cfg_.network_output.uplink_self_scan_port;
            }
        }
        if (cfg_.uplink.ertm_debug_output_enabled) {
            ertm_debug_pub_ = std::make_unique<ZmqByteSender>(
                cfg_.network_output.ertm_debug_ip,
                static_cast<uint16_t>(cfg_.network_output.ertm_debug_port));
            if (LOG_MOD_ON(Ertm)) {
                LOG_G_INFO_M(Ertm) << "[eRTM] debug stream: "
                             << cfg_.network_output.ertm_debug_ip << ':'
                             << cfg_.network_output.ertm_debug_port;
            }
        }
        if (cfg_.sync_tracking.hardware_sync) {
            ocxo_debug_sender_ = std::make_unique<OcxoDebugSender>(
                cfg_.network_output.ocxo_debug_ip,
                static_cast<uint16_t>(cfg_.network_output.ocxo_debug_port),
                64);
            LOG_G_INFO_M(Ocxo)
                << "OCXO ZeroMQ debug stream: bind="
                << cfg_.network_output.ocxo_debug_ip << ':'
                << cfg_.network_output.ocxo_debug_port
                << ", protocol=OCXODBG3, channels=16, pps_diagnostics=exact";
        }
    }

    size_t _clear_frame_queue(const char* reason, bool warn = true) {
        size_t dropped = 0;
        RxFrame frame;
        while (frame_queue_.try_pop(frame)) {
            _rx_frame_pool.release(std::move(frame));
            ++dropped;
        }
        if (warn && dropped > 0) {
            LOG_RT_WARN_M(Recovery) << "[UE recovery] dropping " << dropped
                          << " queued RX frames: reason=" << reason;
        }
        return dropped;
    }

    void _clear_recovery_queues_from_process_thread() {
        const size_t dropped_frames = _clear_frame_queue("stream recovery");
        const size_t dropped_sync = sync_queue_.size();
        sync_queue_.clear();
        if (dropped_sync > 0) {
            LOG_RT_WARN_M(Recovery) << "[UE recovery] dropping " << dropped_sync
                          << " queued sync batches: generation="
                          << _sync_generation.load(std::memory_order_acquire);
        }
        if (dropped_frames > 0 || dropped_sync > 0) {
            LOG_RT_WARN_M(Recovery) << "[UE recovery] queue clear complete: rx_frames="
                          << dropped_frames << ", sync_batches=" << dropped_sync;
        }
    }

    static int64_t _positive_mod_i64(int64_t value, int64_t mod) {
        if (mod <= 0) {
            return 0;
        }
        int64_t out = value % mod;
        if (out < 0) {
            out += mod;
        }
        return out;
    }

    void _log_rx_timestamp_boundary(
        const char* context,
        int64_t timestamp_ns,
        int64_t sample_adjust,
        bool update_previous_delta)
    {
        if (!LOG_MOD_ON(Recovery)) {
            return;
        }
        const int64_t start_ns =
            _rx_stream_start_time_ns.load(std::memory_order_acquire);
        const int64_t frame_samples =
            static_cast<int64_t>(cfg_.samples_per_frame());
        const double rate = _rx_rate_hz();
        if (timestamp_ns < 0 || start_ns < 0 || frame_samples <= 0 ||
            !(rate > 0.0)) {
            LOG_RT_WARN_M(Recovery) << "[UE recovery] rx_timestamp_boundary context=" << context
                          << ", valid=0"
                          << ", timestamp_ns=" << timestamp_ns
                          << ", stream_start_ns=" << start_ns
                          << ", sample_adjust=" << sample_adjust
                          << ", frame_samples=" << frame_samples;
            return;
        }

        // Same actual RX rate used by timeline configure / stamp boundary.
        const double raw_offset =
            static_cast<double>(timestamp_ns - start_ns) *
            rate * 1.0e-9;
        const int64_t raw_offset_samples =
            static_cast<int64_t>(std::llround(raw_offset));
        const int64_t adjusted_offset_samples = raw_offset_samples + sample_adjust;
        const int64_t phase_samples =
            _positive_mod_i64(adjusted_offset_samples, frame_samples);
        int64_t nearest_boundary_error = phase_samples;
        if (nearest_boundary_error > frame_samples / 2) {
            nearest_boundary_error -= frame_samples;
        }

        int64_t delta_from_previous = std::numeric_limits<int64_t>::min();
        if (update_previous_delta) {
            const int64_t previous =
                _last_rx_boundary_offset_samples.exchange(
                    adjusted_offset_samples,
                    std::memory_order_acq_rel);
            if (previous != std::numeric_limits<int64_t>::min()) {
                delta_from_previous = adjusted_offset_samples - previous;
            }
        }

        LOG_RT_WARN_M(Recovery) << "[UE recovery] rx_timestamp_boundary context=" << context
                      << ", valid=1"
                      << ", restart_epoch="
                      << _rx_stream_start_restart_epoch.load(std::memory_order_acquire)
                      << ", generation="
                      << _sync_generation.load(std::memory_order_acquire)
                      << ", timestamp_ns=" << timestamp_ns
                      << ", stream_start_ns=" << start_ns
                      << ", raw_offset_samples=" << raw_offset_samples
                      << ", sample_adjust=" << sample_adjust
                      << ", adjusted_offset_samples=" << adjusted_offset_samples
                      << ", frame_index_floor="
                      << (adjusted_offset_samples / frame_samples)
                      << ", phase_samples=" << phase_samples
                      << ", nearest_boundary_error=" << nearest_boundary_error
                      << ", delta_from_previous="
                      << (delta_from_previous == std::numeric_limits<int64_t>::min()
                              ? 0
                              : delta_from_previous)
                      << ", has_previous="
                      << (delta_from_previous == std::numeric_limits<int64_t>::min()
                              ? 0
                              : 1);
    }

    void _reset_for_reacquisition() {
        _reset_count = 0;
        _preamble_bad_count = 0;
        adjust_rx_freq(0.0, true);
        sfo_estimator.reset();
        _cfo_ewma.reset();
        _ocxo_cfo_quality_gate.reset();
        _ocxo_log_counter = 0;
        _ocxo_to_pub_known = false;
        _ocxo_to_last_pub_gen = 0;
        _ocxo_to_age_frames = 0;
        if (ocxo_debug_sender_) {
            ocxo_debug_sender_->reset_counter();
        }
        if (cfg_.sync_tracking.hardware_sync) {
            if (_hw_sync) {
                if (_ocxo_discipliner) {
                    _ocxo_discipliner->begin_reacquisition_hold();
                }
                _hw_sync->rebase_frequency_control_slew();
            }
            LOG_RT_INFO_M(Ocxo)
                << "OCXO reacquisition hold started after sync reset"
                << " stage="
                << (_ocxo_discipliner
                        ? OCXODiscipliner::stage_name(
                              _ocxo_discipliner->stage())
                        : "inactive")
                << " preserve_s="
                << OCXODiscipliner::kReacquisitionModeHoldS
                << " ppm=" << (_hw_sync ? _hw_sync->current_offset_ppm() : 0.0)
                << " word=" << (_hw_sync ? _hw_sync->current_word() : 0);
        }
        _enter_sync_search_state();
        _mute_uplink_tx_gain_for_sync_search();
        _control_time_gates.mark_reset_now(radio_time_now());
    }

    // Process-thread-only validation before any demodulation or tracking.
    bool _accept_normal_preamble(const RxFrame& frame) {
        if (!(cfg_.sync_tracking.software_sync || cfg_.sync_tracking.hardware_sync)) {
            return true;
        }
        const size_t symbol_len = cfg_.ofdm.fft_size + cfg_.ofdm.cp_length;
        const auto preamble = SyncProcessor::validate_sync_preamble(
            frame.frame_data, cfg_.ofdm.sync_pos * symbol_len,
            cfg_.ofdm.fft_size, cfg_.ofdm.cp_length,
            cfg_.ofdm.enable_sec_sync_symbol, cfo_training_sequence_enabled(cfg_),
            cfg_.ofdm.cfo_training_period_samples, cfg_.rf_sampling.sample_rate);
        if (preamble.valid) {
            _preamble_bad_count = 0;
            return true;
        }
        if (_preamble_bad_count < _reset_hold_frames) {
            ++_preamble_bad_count;
        }
        LOG_RT_WARN_HZ_M(Sync, 2)
            << "[sync] Dropping frame with mismatched preamble"
            << ": count=" << _preamble_bad_count << "/" << _reset_hold_frames
            << ", repeated_zc_metric=" << preamble.repeated_zc_metric
            << ", cfo_training_metric=" << preamble.cfo_training_metric;
        if (_preamble_bad_count >= _reset_hold_frames &&
            _control_time_gates.allow_reset(frame.usrp_time_ns)) {
            LOG_RT_WARN_M(Sync) << "[sync] RESET Persistent preamble mismatch";
            _reset_for_reacquisition();
        }
        return false;
    }

    void _enter_sync_search_state() {
        _preamble_bad_count = 0;
        _sync_generation.fetch_add(1, std::memory_order_acq_rel);
        _sync_in_progress = false;
        _sync_in_progress_hold_frames = 0;
        _sync_in_progress_expect_id = 0;
        _delay_adjustment_count = 0;
        _last_delay_index_err = 0;
        _set_uplink_waveform_enabled(false);
        _clear_air_frame_identity("sync_search");
        _llr_snr_linear_filtered = 1.0;
        _llr_snr_filter_initialized = false;
        const bool log_agc = LOG_MOD_ON(Agc);
        double search_gain_db = 0.0;
        const bool reset_search_gain = _sync_search_gain_sweep.reset_to_default(
            [this](double gain_db) {
                dev_->set_rx_gain(gain_db, cfg_.downlink.rx_channel);
            },
            [this](double gain_db) {
                _rx_agc.sync_to_gain(gain_db);
            },
            &search_gain_db
        );
        if (reset_search_gain && log_agc) {
            LOG_RT_INFO_M(Agc) << "Search RX AGC reset gain to default: " << search_gain_db << " dB";
        }
        // Only the queues consumed by THIS thread may be drained here:
        // frame_queue_ and sync_queue_ (process_proc is their sole consumer).
        // sensing_queue_ and the LLR queues belong to the sensing and
        // bit-processing consumers; popping them from here would make a second
        // concurrent consumer on an SPSC ring (double-pop, moved-from frames
        // polluting the pools). Their consumers already drop frames whose
        // generation predates the bump above, so stale entries drain naturally
        // within a few frames.
        _clear_frame_queue("enter sync search");
        const size_t dropped_sync = sync_queue_.size();
        sync_queue_.clear();
        if (dropped_sync > 0) {
            LOG_RT_WARN_M(Recovery) << "[UE recovery] entering sync search dropped "
                          << dropped_sync << " queued sync batches";
        }
        state_ = RxState::SYNC_SEARCH;
    }

    /**
     * @brief Rx Streamer Thread Function.
     * 
     * Continuous loop that receives baseband samples from the USRP.
     * Implements a state machine (SYNC_SEARCH -> ALIGNMENT -> NORMAL) to handle
     * frame synchronization and alignment before normal reception.
     */
    void rx_proc(radio::TimeSpec stream_start_time) {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        radio::set_thread_priority(1.0, true);
        bind_current_thread_from_ue_downlink_role(cfg_, 0);
        radio::RxMetadata md;
        auto issue_start = [&](const radio::TimeSpec& start_time) {
            radio::StreamCmd cmd(radio::StreamMode::StartContinuous);
            const bool timed_start = start_time.get_real_secs() > 0.0;
            cmd.stream_now = !timed_start;
            if (timed_start) {
                cmd.time_spec = start_time;
                const int64_t start_time_ns = time_spec_to_ns(start_time);
                const uint64_t restart_epoch =
                    _stream_restart_count.load(std::memory_order_acquire);
                _rx_stream_start_time_ns.store(start_time_ns, std::memory_order_release);
                _rx_stream_start_restart_epoch.store(restart_epoch, std::memory_order_release);
                _rx_boundary_log_remaining.store(16, std::memory_order_release);
                _last_rx_boundary_offset_samples.store(
                    std::numeric_limits<int64_t>::min(),
                    std::memory_order_release);
                LOG_G_INFO_M(Radio) << "[UE] timed RX stream start at "
                             << start_time.get_real_secs()
                             << " s, shared with UL-TX";
                if (LOG_MOD_ON(Recovery)) {
                    LOG_RT_WARN_M(Recovery) << "[UE recovery] rx_stream_anchor start_time="
                                  << start_time.get_real_secs()
                                  << ", start_time_ns=" << start_time_ns
                                  << ", restart_epoch=" << restart_epoch
                                  << ", generation="
                                  << _sync_generation.load(std::memory_order_acquire)
                                  << ", frame_samples=" << cfg_.samples_per_frame();
                }
            } else {
                _rx_stream_start_time_ns.store(-1, std::memory_order_release);
                _rx_stream_start_restart_epoch.store(
                    _stream_restart_count.load(std::memory_order_acquire),
                    std::memory_order_release);
                LOG_G_INFO_M(Radio) << "[UE] immediate RX stream start";
            }
            rx_stream_->issue_stream_cmd(cmd);
        };
        auto issue_stop = [&]() {
            try {
                rx_stream_->issue_stream_cmd(radio::StreamCmd(radio::StreamMode::StopContinuous));
            } catch (const std::exception& e) {
                LOG_RT_WARN_M(Sync) << "[UE RX] failed to stop stream before restart: " << e.what();
            }
        };

        issue_start(stream_start_time);

        while (running_.load()) {
            const bool tx_requested_restart =
                _uplink_tx &&
                (_uplink_tx->tx_error_count().load(std::memory_order_relaxed) !=
                 _handled_ul_tx_error_count.load(std::memory_order_relaxed));
            const bool stream_restart_requested =
                _stream_restart_requested.exchange(false, std::memory_order_acq_rel);
            if (stream_restart_requested || tx_requested_restart) {
                uint64_t restart_epoch = 0;
                if (stream_restart_requested) {
                    restart_epoch =
                        _stream_restart_pending_epoch.exchange(0, std::memory_order_acq_rel);
                }
                if (restart_epoch == 0) {
                    restart_epoch =
                        _stream_restart_count.fetch_add(1, std::memory_order_relaxed) + 1;
                }
                if (LOG_MOD_ON(Recovery)) {
                    LOG_RT_WARN_M(Recovery) << "[UE recovery] shared_restart_begin restart_epoch="
                                  << restart_epoch
                                  << ", stream_restart_requested="
                                  << stream_restart_requested
                                  << ", tx_requested_restart=" << tx_requested_restart
                                  << ", state_before="
                                  << static_cast<int>(state_.load(std::memory_order_relaxed))
                                  << ", sync_generation_before="
                                  << _sync_generation.load(std::memory_order_relaxed)
                                  << ", pending_alignment_id="
                                  << _pending_alignment_id.load(std::memory_order_relaxed)
                                  << ", ul_tx_rx_shift_before="
                                  << _uplink_tx_rx_alignment_shift.load(std::memory_order_relaxed);
                }
                if (_uplink_tx) {
                    _handled_ul_tx_error_count.store(
                        _uplink_tx->tx_error_count().load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
                }
                issue_stop();
                stream_start_time = _next_timed_stream_start();
                _reset_receive_state_after_stream_restart(restart_epoch);
                if (_uplink_tx && stream_start_time.get_real_secs() > 0.0) {
                    _uplink_tx->reschedule_timed_tx(stream_start_time, restart_epoch);
                    const bool tx_restart_submitted =
                        _uplink_tx->wait_for_restart_submitted(restart_epoch);
                    if (!tx_restart_submitted) {
                        LOG_RT_WARN_M(Recovery) << "[UE recovery] timed UL-TX restart was not submitted "
                                      << "before RX restart; restart_epoch=" << restart_epoch
                                      << ", start_time=" << stream_start_time.get_real_secs();
                    } else if (LOG_MOD_ON(Recovery)) {
                        LOG_RT_WARN_M(Recovery) << "[UE recovery] timed UL-TX restart submitted before RX start"
                                      << ", restart_epoch=" << restart_epoch
                                      << ", start_time=" << stream_start_time.get_real_secs();
                    }
                }
                issue_start(stream_start_time);
                LOG_RT_WARN_M(Recovery) << "[UE] shared RX/UL-TX restart at "
                              << stream_start_time.get_real_secs() << " s";
                if (LOG_MOD_ON(Recovery)) {
                    LOG_RT_WARN_M(Recovery) << "[UE recovery] shared_restart_applied start_time="
                                  << stream_start_time.get_real_secs()
                                  << ", restart_epoch=" << restart_epoch
                                  << ", stream_restart_requested="
                                  << stream_restart_requested
                                  << ", tx_requested_restart=" << tx_requested_restart
                                  << ", handled_ul_tx_errors="
                                  << _handled_ul_tx_error_count.load(std::memory_order_relaxed)
                                  << ", stream_restart_count="
                                  << _stream_restart_count.load(std::memory_order_relaxed)
                                  << ", next_state="
                                  << static_cast<int>(state_.load(std::memory_order_relaxed))
                                  << ", sync_generation="
                                  << _sync_generation.load(std::memory_order_relaxed);
                }
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
    void handle_sync_search(radio::RxMetadata& md) {
        SyncBatch* sync_batch = sync_queue_.producer_slot();
        AlignedVector& target_buffer =
            (sync_batch != nullptr) ? sync_batch->data : _sync_scratch_buffer;
        const uint64_t batch_generation =
            _sync_generation.load(std::memory_order_acquire);

        int64_t first_time_ns = -1;
        size_t received = rx_stream_->recv(
            target_buffer.data(), cfg_.sync_samples(), md);
        bool restart_read = false;
        if (_handle_rx_metadata_error(md, "sync search", &restart_read) && restart_read) {
            received = 0;
        }
        if (received > 0) {
            first_time_ns = metadata_time_to_ns(md);
        }

        if (sync_batch == nullptr) {
            if (running_.load(std::memory_order_relaxed)) {
                LOG_RT_WARN_HZ_M(Sync, 5) << "UE sync queue full; dropping newly received sync batch"
                                << ": received_samples=" << received;
            }
            return;
        }
        if (received < cfg_.sync_samples()) {
            if (received > 0 && running_.load(std::memory_order_relaxed)) {
                LOG_RT_WARN_HZ_M(Sync, 5) << "UE sync read ended early; dropping partial sync batch"
                                << ": received_samples=" << received
                                << ", expected_samples=" << cfg_.sync_samples();
            }
            return;
        }
        if (batch_generation != _sync_generation.load(std::memory_order_acquire) ||
            _stream_restart_requested.load(std::memory_order_acquire)) {
            LOG_RT_WARN_HZ_M(Recovery, 5)
                << "UE sync generation changed before enqueue; dropping completed sync batch"
                << ": batch_generation=" << batch_generation
                << ", current_generation="
                << _sync_generation.load(std::memory_order_relaxed);
            return;
        }
        sync_batch->usrp_time_ns = first_time_ns;
        sync_batch->generation = batch_generation;
        sync_queue_.producer_commit();
    }

    /**
     * @brief Handle Frame Alignment State.
     * 
     * Once sync is found, this state aligns the sample stream to the frame boundary
     * by discarding a specific number of samples (discard_samples_). This is also 
     * used for timing adjustments during normal operation.
     */
    void handle_alignment(radio::RxMetadata& md) {
        const bool do_latency_profile =
            LOG_MOD_ON(DemodProfiling);
        const int alignment_samples = discard_samples_.load(std::memory_order_relaxed);
        const uint64_t alignment_id =
            _pending_alignment_id.load(std::memory_order_relaxed);
        const uint64_t alignment_restart_epoch =
            _pending_alignment_restart_epoch.load(std::memory_order_relaxed);
        const uint64_t alignment_generation =
            _pending_alignment_generation.load(std::memory_order_relaxed);
        const size_t frame_samples = cfg_.samples_per_frame();
        const size_t negative_shift = (alignment_samples < 0)
            ? std::min<size_t>(static_cast<size_t>(-alignment_samples), frame_samples)
            : 0;
        const size_t positive_shift = (alignment_samples > 0)
            ? static_cast<size_t>(alignment_samples)
            : 0;
        const size_t total_read = frame_samples + positive_shift - negative_shift;
        AlignedVector temp_buf(total_read);
        int64_t frame_time_ns = -1;
        size_t received = rx_stream_->recv(temp_buf.data(), total_read, md);
        bool restart_read = false;
        if (_handle_rx_metadata_error(md, "alignment", &restart_read) && restart_read) {
            received = 0;
        }
        if (received > 0) {
            frame_time_ns = metadata_time_to_ns(md);
        }
        if (received < total_read) {
            if (received > 0 && running_.load(std::memory_order_relaxed)) {
                LOG_RT_WARN_M(Recovery) << "[UE recovery] alignment_rx_abort alignment_id="
                              << alignment_id
                              << ", alignment=" << alignment_samples
                              << ", restart_epoch=" << alignment_restart_epoch
                              << ", scheduled_generation=" << alignment_generation
                              << ", current_generation="
                              << _sync_generation.load(std::memory_order_acquire)
                              << ", received=" << received
                              << "/" << total_read
                              << ", restart_pending="
                              << _stream_restart_requested.load(std::memory_order_acquire);
            }
            return;
        }
        // Acquire pre-allocated RX frame from pool
        RxFrame frame = _rx_frame_pool.acquire();
        frame.Alignment = alignment_samples;
        frame.host_enqueue_time_ns = do_latency_profile ? host_now_ns() : 0;
        frame.generation = _sync_generation.load(std::memory_order_acquire);
        frame.ertm_rx_alignment_shift_samples = 0;
        // Alignment moves the committed frame start by alignment_samples relative
        // to the raw first-sample metadata time (same convention as boundary logs).
        _stamp_rx_air_frame(frame, frame_time_ns, static_cast<int64_t>(alignment_samples));
        if (positive_shift > 0) {
            LOG_RT_INFO_M(Recovery)
                << "UE alignment is dropping " << positive_shift
                << " leading RX samples to advance the frame boundary"
                << ": alignment_id=" << alignment_id;
        } else if (negative_shift > 0) {
            LOG_RT_INFO_M(Recovery)
                << "UE alignment is padding " << negative_shift
                << " leading samples because the aligned RX frame has no data for them"
                << ": alignment_id=" << alignment_id;
        }
        // Fresh acquisition: re-anchor the UL-TX window to this committed RX
        // frame's boundary phase on the timed grid. Uses the same reference as
        // the alignment_output_frame_start boundary log (this frame's own RX
        // timestamp plus the applied discard), which is the authoritative
        // boundary position -- the sync-read timestamp is not. This keeps the
        // UE's own uplink ZC landing at the uplink window inside its own RX
        // frame across restarts, instead of drifting by the sync read's grid
        // misalignment.
        if (_pending_alignment_fresh.exchange(false, std::memory_order_relaxed) &&
            frame_time_ns >= 0) {
            const int64_t start_ns =
                _rx_stream_start_time_ns.load(std::memory_order_acquire);
            const int64_t fs = static_cast<int64_t>(frame_samples);
            const double rate = _rx_rate_hz();
            if (start_ns >= 0 && fs > 0 && rate > 0.0) {
                // Use actual (possibly coerced) RX rate — must match timeline.
                const int64_t raw_offset_samples =
                    static_cast<int64_t>(std::llround(
                        static_cast<double>(frame_time_ns - start_ns) *
                        rate * 1.0e-9));
                const int64_t boundary_offset_samples =
                    raw_offset_samples + static_cast<int64_t>(alignment_samples);
                const int64_t grid_phase_samples =
                    _positive_mod_i64(boundary_offset_samples, fs);
                _reconstruct_uplink_tx_rx_alignment_absolute(
                    grid_phase_samples, alignment_id, alignment_restart_epoch);
            }
        }
        // The alignment read above has now committed the new RX coordinate.
        const int32_t committed_alignment =
            _uplink_tx_rx_alignment_shift.load(std::memory_order_relaxed);
        _committed_uplink_tx_rx_alignment_shift.store(
            committed_alignment, std::memory_order_release);
        frame.ertm_rx_alignment_shift_samples = committed_alignment;
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

        // Publish applied receipt before queue push so process_proc can clear
        // the latch and inject SFO/eRTM Alignment even if this frame is dropped.
        _alignment_applied_samples.store(alignment_samples, std::memory_order_relaxed);
        _alignment_applied_time_ns.store(frame_time_ns, std::memory_order_relaxed);
        _alignment_applied_id.store(alignment_id, std::memory_order_release);

        if (!frame_queue_.try_push(std::move(frame))) {
            // Sample IQ for this boundary may be lost, but the latch side-channel
            // above already recorded the applied alignment_id.
            if (running_.load(std::memory_order_relaxed)) {
                LOG_RT_WARN_HZ_M(Sync, 5)
                    << "[delay_track] alignment_frame_drop queue_full alignment="
                    << alignment_samples
                    << " alignment_id=" << alignment_id
                    << " depth=" << frame_queue_.size()
                    << "/" << frame_queue_.capacity()
                    << " (latch clears via applied side-channel; IQ frame lost)";
            }
            _rx_frame_pool.release(std::move(frame));
        }
        if (LOG_MOD_ON(Recovery)) {
            LOG_RT_WARN_M(Recovery) << "[UE recovery] alignment_rx_commit alignment_id="
                          << alignment_id
                          << ", alignment=" << alignment_samples
                          << ", restart_epoch=" << alignment_restart_epoch
                          << ", scheduled_generation=" << alignment_generation
                          << ", frame_generation="
                          << _sync_generation.load(std::memory_order_acquire)
                          << ", frame_time_ns=" << frame_time_ns
                          << ", total_read=" << total_read
                          << ", positive_shift=" << positive_shift
                          << ", negative_shift=" << negative_shift
                          << ", restart_pending="
                          << _stream_restart_requested.load(std::memory_order_acquire);
            _log_rx_timestamp_boundary(
                "alignment_raw_first",
                frame_time_ns,
                0,
                false);
            _log_rx_timestamp_boundary(
                "alignment_output_frame_start",
                frame_time_ns,
                static_cast<int64_t>(alignment_samples),
                true);
        }
//        LOG_G_INFO_M(Root) << "Alignment done, moving "<< discard_samples_<< " samples" << std::endl;
//        LOG_G_INFO_M(Root) <<  discard_samples_<< std::endl;
        _set_uplink_waveform_enabled(true);
        state_ = RxState::NORMAL;
    }

    /**
     * @brief Handle Normal Reception State.
     * 
     * Reads complete frames of aligned samples and pushes them to the frame queue
     * for processing.
     */
    void handle_normal_rx(radio::RxMetadata& md) {
        const bool do_latency_profile =
            LOG_MOD_ON(DemodProfiling);
        // Acquire pre-allocated RX frame from pool
        RxFrame frame = _rx_frame_pool.acquire();
        frame.Alignment = 0;
        frame.usrp_time_ns = -1;
        frame.identity.clear();
        frame.host_enqueue_time_ns = 0;
        frame.generation = _sync_generation.load(std::memory_order_acquire);
        frame.ertm_rx_alignment_shift_samples = 0;
        int64_t raw_first_ns = -1;
        size_t received = rx_stream_->recv(
            frame.frame_data.data(), cfg_.samples_per_frame(), md);
        bool restart_read = false;
        if (_handle_rx_metadata_error(md, "normal", &restart_read) && restart_read) {
            received = 0;
        }
        if (received > 0) {
            raw_first_ns = metadata_time_to_ns(md);
        }

        if (received == cfg_.samples_per_frame()) {
            // NORMAL: stream is already on the committed grid; raw first sample
            // is the air-frame boundary (alignment_sample_adjust = 0).
            _stamp_rx_air_frame(frame, raw_first_ns, 0);
            // Freeze cumulative UL-TX Alignment with this RX frame so process/
            // sensing never pair h_est of frame N with a later schedule update.
            frame.ertm_rx_alignment_shift_samples =
                _committed_uplink_tx_rx_alignment_shift.load(std::memory_order_acquire);
            uint32_t remaining =
                _rx_boundary_log_remaining.load(std::memory_order_acquire);
            while (remaining > 0 &&
                   !_rx_boundary_log_remaining.compare_exchange_weak(
                       remaining,
                       remaining - 1,
                       std::memory_order_acq_rel,
                       std::memory_order_acquire)) {
            }
            if (remaining > 0) {
                _log_rx_timestamp_boundary(
                    "normal_frame_start",
                    frame.usrp_time_ns,
                    0,
                    true);
            }
            if (do_latency_profile) {
                frame.host_enqueue_time_ns = host_now_ns();
            }
            if (!frame_queue_.try_push(std::move(frame))) {
                if (running_.load(std::memory_order_relaxed)) {
                    LOG_RT_WARN_HZ(5) << "RX frame queue full, dropping newest frame";
                }
                _rx_frame_pool.release(std::move(frame));
            }
        } else {
            if (received > 0 && running_.load(std::memory_order_relaxed)) {
                LOG_RT_WARN_HZ_M(Recovery, 5)
                    << "UE normal RX read ended early; dropping partial frame"
                    << ": received_samples=" << received
                    << ", expected_samples=" << cfg_.samples_per_frame();
            }
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
        const bool log_sync_profile = LOG_MOD_ON(Sync);
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
                LOG_RT_INFO_M(Sync) << "CFO training field estimate: metric="
                              << estimate.metric
                              << ", cfo=" << estimate.cfo_hz << " Hz"
                              << ", symbol_start=" << cfo_symbol_start;
            }
            return estimate;
        };

        auto validate_candidate = [&](int position) {
            return position >= 0 && SyncProcessor::validate_sync_preamble(
                sync_data, static_cast<size_t>(position),
                cfg_.ofdm.fft_size, cfg_.ofdm.cp_length,
                cfg_.ofdm.enable_sec_sync_symbol, cfo_training_sequence_enabled(cfg_),
                cfg_.ofdm.cfo_training_period_samples, cfg_.rf_sampling.sample_rate).valid;
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
                    LOG_RT_INFO_M(Sync) << format_sync_alias_candidates(
                        "Second sync ZC refine",
                        refine_result);
                }
                max_pos = refine_result.max_pos;
                max_corr = refine_result.max_corr;
                avg_corr = refine_result.avg_corr;
                if (refine_result.valid && avg_corr > 0.0f &&
                    (max_corr / avg_corr) > kZcEnergyThreshold && validate_candidate(max_pos)) {
                    used_sec_sync_symbol = true;
                    sync_found = true;
                    initial_cfo_hz = refine_result.cfo_hz;
                    SyncProcessor::apply_cfo_compensation(
                        sync_data,
                        cfg_.rf_sampling.sample_rate,
                        initial_cfo_hz,
                        _sync_cfo_compensated_buffer);
                    LOG_RT_INFO_M(Sync) << "Second sync symbol coarse metric: " << sec_result.max_metric
                                  << ", coarse symbol start: " << sec_result.coarse_symbol_start
                                  << ", modulo CFO: " << sec_result.coarse_cfo_hz
                                  << " Hz, alias index: " << refine_result.alias_index
                                  << ", refined CFO: " << initial_cfo_hz << " Hz"
                                  << (cfo_training_alias_selected ? " (CFO field deambiguated)" : "");
                } else {
                    LOG_RT_WARN_M(Sync) << "Second sync symbol coarse sync detected but local ZC refine failed."
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
                    const double cp_alias_period_hz =
                        cfg_.rf_sampling.sample_rate / static_cast<double>(cfg_.ofdm.fft_size);
                    const int cp_alias_span = sync_cfo_alias_span_from_range(
                        cfg_.sync_tracking.sync_cfo_alias_search_range_hz,
                        cp_alias_period_hz);
                    const int first_candidate = max_pos;
                    bool accepted = false;
                    // A global peak may be the first of the two identical ZCs.
                    // Check its following symbol as a main-ZC candidate too.
                    for (int candidate = 0;
                         candidate <= (cfg_.ofdm.enable_sec_sync_symbol ? 1 : 0);
                         ++candidate) {
                        const int center = first_candidate + candidate * static_cast<int>(symbol_len);
                        if (center > max_search_start) {
                            continue;
                        }
                        const size_t search_start = static_cast<size_t>(std::max(
                            0, center - static_cast<int>(local_zc_search_radius)));
                        const size_t search_end = static_cast<size_t>(std::min(
                            max_search_start, center + static_cast<int>(local_zc_search_radius)));
                        auto cp_refine = _sync_processor.refine_sync_cfo_alias(
                            sync_data, residual_cfo_hz, cp_alias_period_hz,
                            cfg_.rf_sampling.sample_rate, search_start, search_end,
                            cp_alias_span, _sync_cfo_compensated_buffer, collect_alias_candidates);
                        const auto cfo_training_est = estimate_cfo_training(cp_refine.max_pos);
                        const bool cfo_training_alias_selected =
                            cfo_training_est.valid &&
                            cfo_training_est.metric > kCfoTrainingMetricThreshold &&
                            SyncProcessor::select_alias_closest_to_cfo_reference(
                                cp_refine, cfo_training_est.cfo_hz);
                        if (log_sync_profile) {
                            LOG_RT_INFO_M(Sync) << format_sync_alias_candidates(
                                "CP CFO ZC refine", cp_refine);
                        }
                        if (cp_refine.valid && cp_refine.avg_corr > 0.0f &&
                            (cp_refine.max_corr / cp_refine.avg_corr) > kZcEnergyThreshold &&
                            validate_candidate(cp_refine.max_pos)) {
                            cfo = cp_refine.cfo_hz;
                            max_pos = cp_refine.max_pos;
                            max_corr = cp_refine.max_corr;
                            avg_corr = cp_refine.avg_corr;
                            accepted = true;
                            LOG_RT_INFO_M(Sync) << "CP CFO alias refine: modulo CFO=" << residual_cfo_hz
                                          << " Hz, alias index=" << cp_refine.alias_index
                                          << ", refined CFO=" << cfo << " Hz"
                                          << (cfo_training_alias_selected ? " (CFO field deambiguated)" : "");
                            break;
                        }
                    }
                    // ZC-only configurations retain the original coarse-lock
                    // behavior if the optional CFO refinement cannot improve it.
                    if (!accepted && (cfg_.ofdm.enable_sec_sync_symbol ||
                                      cfo_training_sequence_enabled(cfg_))) {
                        LOG_RT_WARN_HZ_M(Sync, 2)
                            << "[sync] Rejected ZC candidate: configured preamble incomplete or mismatched";
                        return;
                    }
                }

                _restore_uplink_tx_gain_after_sync();
                _sync_search_gain_sweep.note_sync_found();
                LOG_RT_INFO_M(Sync) << "Sync found at pos: " << max_pos
                                  << " with peak/avg: " << peak_ratio_db(max_corr, avg_corr)
                                  << " dB (peak=" << power_to_db(max_corr)
                                  << " dB, avg=" << power_to_db(avg_corr)
                                  << " dB) Threshold: " << power_to_db(kZcEnergyThreshold)
                                  << " dB";

                LOG_RT_INFO_M(Sync) << "CFO estimate: " << cfo << " Hz (using " << available_symbols
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
                if (LOG_MOD_ON(Recovery)) {
                    LOG_RT_WARN_M(Recovery) << "[UE recovery] sync_alignment_candidate max_pos="
                                  << max_pos
                                  << ", sync_pos_samples="
                                  << (cfg_.ofdm.sync_pos * symbol_len)
                                  << ", desired_peak_pos="
                                  << cfg_.sync_tracking.desired_peak_pos
                                  << ", predictive_delay_samples="
                                  << predictive_delay_samples
                                  << ", scheduled_alignment="
                                  << sync_offset_.load(std::memory_order_relaxed)
                                  << ", frame_samples=" << cfg_.samples_per_frame()
                                  << ", ul_tx_rx_shift_before="
                                  << _uplink_tx_rx_alignment_shift.load(std::memory_order_relaxed)
                                  << ", generation="
                                  << _sync_generation.load(std::memory_order_relaxed);
                }
                _clear_frame_queue("fresh alignment scheduled");
                _schedule_receive_alignment(static_cast<int32_t>(sync_offset_));
                issued_alignment = true;
            } else {
                LOG_RT_WARN_M(Sync) << "No valid symbols for CFO estimation";
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
                    dev_->set_rx_gain(gain_db, cfg_.downlink.rx_channel);
                },
                [this](double gain_db) {
                    _rx_agc.sync_to_gain(gain_db);
                },
                &search_gain_db
            );
            if (stepped_search_gain && LOG_MOD_ON(Agc)) {
                LOG_RT_INFO_M(Agc) << "Search RX AGC stepped gain to " << search_gain_db
                              << " dB after sync miss";
            }
        }
    }

    void process_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        radio::set_thread_priority(1, true);
        bind_current_thread_from_ue_downlink_role(cfg_, 1);

        using Clock = std::chrono::high_resolution_clock;
        double total_collect_wall_ms = 0.0;
        double total_launch_wait_ms = 0.0;
        double total_worker_dsp_ms = 0.0;
        size_t frame_count = 0;
        constexpr size_t REPORT_INTERVAL = 434;
        size_t launch_slot_idx = 0;
        size_t collect_slot_idx = 0;
        // process_proc has frame queues / pipeline slots as cushion — cooperative
        // backoff is acceptable (only USRP sample ingest is hard-RT).
        SPSCBackoff sync_backoff;
        const bool do_latency_profile =
            LOG_MOD_ON(DemodProfiling);
        const bool do_eq_breakdown =
            LOG_MOD_ON(DemodEqProfiling);
        CpuDemodProfile demod_profile;

        _start_cpu_demod_workers();

        auto log_process_load = [&]() {
            if (frame_count < REPORT_INTERVAL) {
                return;
            }
            if (!LOG_MOD_ON(DemodProfiling)) {
                total_collect_wall_ms = 0.0;
                total_launch_wait_ms = 0.0;
                total_worker_dsp_ms = 0.0;
                frame_count = 0;
                return;
            }
            const double n = static_cast<double>(frame_count);
            const double avg_collect = total_collect_wall_ms / n;
            const double avg_launch_wait = total_launch_wait_ms / n;
            const double avg_worker_dsp = total_worker_dsp_ms / n;
            const double frame_duration = cfg_.samples_per_frame() / cfg_.rf_sampling.sample_rate * 1000.0;
            const double worker_count = static_cast<double>(std::max<size_t>(1, _cpu_demod_slots.size()));
            // Control-thread load counts only the serial collect stage; the
            // parallel DSP load is reported separately as worker utilization
            // (avg worker busy time per frame vs. N workers' frame budget).
            const double load = frame_duration > 0.0 ? (avg_collect / frame_duration) : 0.0;
            const double worker_util = frame_duration > 0.0
                ? (avg_worker_dsp / (frame_duration * worker_count))
                : 0.0;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2)
                << "[UE CPU demod] Collect wall: " << avg_collect
                << " ms, launch wait: " << avg_launch_wait
                << " ms, control-thread load: " << load * 100.0
                << "%, worker DSP: " << avg_worker_dsp
                << " ms/frame, worker utilization: " << worker_util * 100.0
                << "% (" << _cpu_demod_slots.size() << " workers); "
                << "Actual RX RF Freq: " << format_freq_hz(current_rx_tune_.actual_rf_freq)
                << " Hz, DSP: " << format_freq_hz(current_rx_tune_.actual_dsp_freq)
                << " Hz; Average CFO: " << _avg_freq_offset << " Hz";
            if (do_latency_profile) {
                const LatencySnapshot latency = _take_latency_snapshot_and_reset();
                oss << "\n";
                if (latency.count > 0) {
                    const double ln = static_cast<double>(latency.count);
                    oss << "\n---------- Latency (avg per valid frame, ms) ----------\n"
                        << "RX frame queue wait:         " << (latency.rx_queue_total_ns / ln) * 1e-6 << " ms\n"
                        << "Dequeue + FFT/EQ/LLR queue:  " << (latency.demod_total_ns / ln) * 1e-6 << " ms\n"
                        << "Bit queue + LDPC/UDP out:    " << (latency.bit_total_ns / ln) * 1e-6 << " ms\n"
                        << "TOTAL E2E (excl. RX wait):   " << (latency.e2e_total_ns / ln) * 1e-6 << " ms\n"
                        << "Latency sample count:        " << latency.count << "\n";
                } else {
                    oss << "\n---------- Latency (avg per valid frame, ms) ----------\n"
                        << "No valid latency samples in this interval.\n";
                }
            }
            LOG_RT_INFO_M(DemodProfiling) << oss.str();
            total_collect_wall_ms = 0.0;
            total_launch_wait_ms = 0.0;
            total_worker_dsp_ms = 0.0;
            frame_count = 0;
        };

        auto collect_ready_slots = [&](bool blocking) {
            // SFO/sync-tracking ownership lives here — apply BS timeline resets.
            _apply_bs_air_timeline_reset_on_process_thread();
            bool collected = false;
            while (!_cpu_demod_slots.empty()) {
                auto& slot = *_cpu_demod_slots[collect_slot_idx];
                if (slot.pending == 0) {
                    break;
                }
                CpuDemodResult* result = slot.result_queue.consumer_slot();
                if (result == nullptr) {
                    if (!blocking) {
                        break;
                    }
                    SPSCBackoff result_backoff;
                    while (running_.load(std::memory_order_acquire) &&
                           (result = slot.result_queue.consumer_slot()) == nullptr) {
                        result_backoff.pause();
                    }
                    if (result == nullptr) {
                        break; // shutting down
                    }
                }
                const auto collect_start = Clock::now();
                // Profile fields are purely worker-side at this point (the
                // control stage adds its own times inside _collect below).
                total_worker_dsp_ms += 1e-3 * (
                    result->profile.fft_total + result->profile.channel_est_total +
                    result->profile.cfo_sfo_est_total + result->profile.equalization_total +
                    result->profile.remodulate_total + result->profile.delay_spectrum_total +
                    result->profile.llr_total);
                const bool still_normal = _collect_cpu_demod_result(*result, demod_profile);
                slot.result_queue.consumer_pop();
                --slot.pending;
                total_collect_wall_ms += std::chrono::duration<double, std::milli>(
                    Clock::now() - collect_start).count();
                ++frame_count;
                collected = true;
                _report_cpu_demod_profile_if_needed(demod_profile, do_eq_breakdown);
                log_process_load();
                collect_slot_idx = (collect_slot_idx + 1) % _cpu_demod_slots.size();
                if (!still_normal || !running_.load(std::memory_order_acquire) ||
                    state_.load(std::memory_order_acquire) != RxState::NORMAL) {
                    break;
                }
                blocking = false;
            }
            return collected;
        };

        while (running_.load()) {
            collect_ready_slots(false);
            _apply_pending_process_tracking_reset();
            if (_recovery_queue_clear_requested.exchange(false, std::memory_order_acq_rel)) {
                _clear_recovery_queues_from_process_thread();
            }
            if (state_ == RxState::SYNC_SEARCH) {
                SyncBatch* sync_batch = sync_queue_.consumer_slot();
                if (sync_batch == nullptr) {
                    sync_backoff.pause();
                    continue;
                }
                sync_backoff.reset();
                if (sync_batch->generation !=
                    _sync_generation.load(std::memory_order_acquire)) {
                    LOG_RT_WARN_M(Recovery) << "[UE recovery] dropping stale sync batch"
                                  << ": batch_generation=" << sync_batch->generation
                                  << ", current_generation="
                                  << _sync_generation.load(std::memory_order_acquire);
                    sync_queue_.consumer_pop();
                    continue;
                }
                process_sync_data(sync_batch->data, sync_batch->usrp_time_ns);
                sync_queue_.consumer_pop();
            } else {
                if (_cpu_demod_slots.empty()) {
                    continue;
                }
                auto& launch_slot = *_cpu_demod_slots[launch_slot_idx];
                if (launch_slot.pending >= CpuDemodSlot::kPipelineDepth) {
                    const auto wait_start = Clock::now();
                    collect_ready_slots(true);
                    total_launch_wait_ms += std::chrono::duration<double, std::milli>(
                        Clock::now() - wait_start).count();
                    if (!running_.load(std::memory_order_acquire)) {
                        break;
                    }
                    if (launch_slot.pending >= CpuDemodSlot::kPipelineDepth) {
                        continue;
                    }
                }
                // Wait for the next RX frame, draining finished results in the
                // meantime so in-flight frames are not stranded when the RX
                // stream stalls. Queue depth cushions jitter — backoff is fine.
                RxFrame frame;
                bool have_frame = false;
                SPSCBackoff frame_backoff;
                while (running_.load(std::memory_order_acquire) &&
                       state_.load(std::memory_order_acquire) == RxState::NORMAL) {
                    if (frame_queue_.try_pop(frame)) {
                        if (frame.generation !=
                            _sync_generation.load(std::memory_order_acquire)) {
                            LOG_RT_WARN_HZ_M(Recovery, 5)
                                << "[UE recovery] dropping stale RX frame before demod"
                                << ": frame_generation=" << frame.generation
                                << ", current_generation="
                                << _sync_generation.load(std::memory_order_relaxed);
                            _rx_frame_pool.release(std::move(frame));
                            continue;
                        }
                        have_frame = true;
                        break;
                    }
                    if (collect_ready_slots(false)) {
                        frame_backoff.reset();
                        continue;
                    }
                    frame_backoff.pause();
                }
                if (!have_frame) {
                    if (!running_.load()) {
                        break;
                    }
                    continue;
                }
                if (!_accept_normal_preamble(frame)) {
                    _rx_frame_pool.release(std::move(frame));
                    continue;
                }
                const int64_t frame_dequeue_time_ns = do_latency_profile ? host_now_ns() : 0;
                const float llr_scale_snapshot =
                    _llr_scale_snapshot.load(std::memory_order_acquire);
                CpuDemodTask task{
                    std::move(frame),
                    frame_dequeue_time_ns,
                    llr_scale_snapshot,
                    (_constellation_frame_counter++ % kDebugPublishStride) == 0,
                };
                if (spsc_wait_push(launch_slot.task_queue, std::move(task), [this]() {
                        return !running_.load(std::memory_order_acquire);
                    })) {
                    ++launch_slot.pending;
                    launch_slot_idx = (launch_slot_idx + 1) % _cpu_demod_slots.size();
                } else {
                    if (!task.frame.frame_data.empty()) {
                        _rx_frame_pool.release(std::move(task.frame));
                    }
                }
                collect_ready_slots(false);
            }
        }
        while (!_cpu_demod_slots.empty() && _cpu_demod_slots[collect_slot_idx]->pending > 0) {
            collect_ready_slots(true);
            if (!running_.load(std::memory_order_acquire)) {
                break;
            }
        }
        _stop_cpu_demod_workers();
        _clear_frame_queue("demod pipeline shutdown", false);
        sync_queue_.clear();
        _report_cpu_demod_profile_if_needed(demod_profile, do_eq_breakdown);
        log_process_load();
    }

    // Self-scan ZMQ wire format (single message):
    //   [64-byte LE header "ULSCSCAN1"][complex64 * slice_len]
    // Viewer is pure display: peak-centered slice + metadata; no client-side
    // slicing. Header fields are little-endian.
    void _publish_uplink_self_scan_frame(
        const RxFrame& frame,
        int scan_pos,
        float scan_peak,
        float scan_avg,
        size_t expected_zc_pos,
        int32_t offset_samples,
        int32_t ul_tx_rx_shift,
        int32_t timing_advance)
    {
        if (!uplink_self_scan_pub_ || !_uplink_self_scan_processor) {
            return;
        }
        const size_t frame_samples = frame.frame_data.size();
        if (frame_samples == 0) {
            return;
        }
        const size_t symbol_len = cfg_.ofdm.fft_size + cfg_.ofdm.cp_length;
        const size_t corr_len = frame_samples >= symbol_len
            ? (frame_samples - symbol_len + 1)
            : 0;
        if (corr_len == 0) {
            return;
        }

        // Default (0): one OFDM symbol including CP, matching the sync-symbol
        // / correlation window length used by the self-ZC scan.
        size_t want_slice = cfg_.uplink.debug_self_scan_slice_samples;
        if (want_slice == 0) {
            want_slice = symbol_len;
        }
        want_slice = std::max<size_t>(1, want_slice);
        const size_t slice_len = std::min(want_slice, corr_len);
        const size_t peak_u = static_cast<size_t>(
            std::max(0, std::min(scan_pos, static_cast<int>(corr_len - 1))));
        size_t slice_start = (peak_u > slice_len / 2) ? (peak_u - slice_len / 2) : 0;
        if (slice_start + slice_len > corr_len) {
            slice_start = corr_len - slice_len;
        }

        const size_t actual_start = _uplink_self_scan_processor->copy_last_correlation_slice(
            frame_samples, slice_start, slice_len, _uplink_self_scan_slice);
        if (_uplink_self_scan_slice.empty()) {
            return;
        }

        static constexpr size_t kHeaderBytes = 64;
        static constexpr uint8_t kMagic[8] = {
            'U', 'L', 'S', 'C', 'S', 'C', 'A', 'N'};
        std::vector<uint8_t> packet(
            kHeaderBytes +
            _uplink_self_scan_slice.size() * sizeof(std::complex<float>));
        uint8_t* hdr = packet.data();
        std::memcpy(hdr, kMagic, sizeof(kMagic));
        size_t o = sizeof(kMagic);
        auto put_u32 = [&](uint32_t v) {
            hdr[o++] = static_cast<uint8_t>(v & 0xFFu);
            hdr[o++] = static_cast<uint8_t>((v >> 8) & 0xFFu);
            hdr[o++] = static_cast<uint8_t>((v >> 16) & 0xFFu);
            hdr[o++] = static_cast<uint8_t>((v >> 24) & 0xFFu);
        };
        auto put_i32 = [&](int32_t v) { put_u32(static_cast<uint32_t>(v)); };
        auto put_f32 = [&](float v) {
            uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(v), "Unexpected float size");
            std::memcpy(&bits, &v, sizeof(bits));
            put_u32(bits);
        };
        auto put_u64 = [&](uint64_t v) {
            for (int i = 0; i < 8; ++i) {
                hdr[o++] = static_cast<uint8_t>((v >> (i * 8)) & 0xFFu);
            }
        };

        put_u32(1);  // version
        put_u32(static_cast<uint32_t>(frame_samples));
        put_u32(static_cast<uint32_t>(corr_len));
        put_u32(static_cast<uint32_t>(actual_start));
        put_u32(static_cast<uint32_t>(_uplink_self_scan_slice.size()));
        put_u32(static_cast<uint32_t>(peak_u));
        put_u32(static_cast<uint32_t>(expected_zc_pos));
        put_i32(offset_samples);
        put_f32(scan_peak);
        put_f32(scan_avg);
        put_i32(ul_tx_rx_shift);
        put_i32(timing_advance);
        put_u64(frame.generation);
        // Pad remainder of the fixed 64-byte header.
        while (o < kHeaderBytes) {
            hdr[o++] = 0;
        }

        std::memcpy(
            packet.data() + kHeaderBytes,
            _uplink_self_scan_slice.data(),
            _uplink_self_scan_slice.size() * sizeof(std::complex<float>));
        uplink_self_scan_sender_.add_data(std::move(packet));
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

        // Diagnostic scan: find the UE's own uplink ZC anywhere in the RX frame
        // and report its displacement from the assumed window. This exposes a
        // large post-underflow/post-resync TX-vs-RX offset that the fixed-window
        // estimator below cannot (it would silently emit a flat, peakless
        // h_est). Gated behind ue_recovery/uplink profiling + a stride so a
        // normal run pays nothing.
        constexpr uint64_t kUplinkSelfScanStride = 128;
        const bool scan_log =
            (LOG_MOD_ON(Recovery) || LOG_MOD_ON(UlTx));
        const bool scan_publish =
            uplink_self_scan_spectrum_enabled(cfg_) && uplink_self_scan_pub_;
        if (_uplink_self_scan_processor && (scan_log || scan_publish) &&
            (_uplink_self_scan_frame_counter++ % kUplinkSelfScanStride) == 0 &&
            frame.frame_data.size() >= sym_len) {
            int scan_pos = 0;
            float scan_peak = 0.0f;
            float scan_avg = 0.0f;
            _uplink_self_scan_processor->find_sync_position(
                frame.frame_data, scan_pos, scan_peak, scan_avg);
            const int64_t frame_samples = static_cast<int64_t>(frame.frame_data.size());
            int64_t offset = static_cast<int64_t>(scan_pos) -
                             static_cast<int64_t>(local_tx_zc_start);
            offset %= frame_samples;  // fold to (-frame/2, frame/2] to show lead/lag sign
            if (offset > frame_samples / 2) offset -= frame_samples;
            else if (offset < -frame_samples / 2) offset += frame_samples;
            const float peak_avg_db = (scan_avg > 0.0f)
                ? 10.0f * std::log10(scan_peak / scan_avg)
                : 0.0f;
            const int32_t ul_tx_rx_shift =
                _uplink_tx_rx_alignment_shift.load(std::memory_order_relaxed);
            const int32_t timing_advance =
                _uplink_tx->timing_advance().load(std::memory_order_relaxed);
            if (scan_log) {
                LOG_G_WARN_M(UlTx) << "[UL-TX self-scan] measured_zc_pos=" << scan_pos
                             << ", expected_zc_pos=" << local_tx_zc_start
                             << ", offset=" << offset
                             << " samples, peak/avg=" << peak_avg_db
                             << " dB, rx_alignment=" << frame.Alignment
                             << ", ul_tx_rx_shift=" << ul_tx_rx_shift
                             << ", timing_advance=" << timing_advance
                             << ", target_shift=" << (ul_tx_rx_shift + timing_advance)
                             << ", generation=" << frame.generation;
            }
            if (scan_publish) {
                _publish_uplink_self_scan_frame(
                    frame,
                    scan_pos,
                    scan_peak,
                    scan_avg,
                    local_tx_zc_start,
                    static_cast<int32_t>(offset),
                    ul_tx_rx_shift,
                    timing_advance);
            }
        }

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
            LOG_G_INFO_M(UlTx) << "[UL-TX] self-channel debug: local UE UL ZC offset="
                         << local_tx_zc_start
                         << ", rx_frame_start_offset=0"
                         << ", frame_period=" << frame.frame_data.size();
        }
    }

    void _run_cpu_demod_task(CpuDemodWorkerContext& ctx, CpuDemodTask&& task, CpuDemodResult& result) {
        using ProfileClock = std::chrono::high_resolution_clock;
        result.frame = std::move(task.frame);
        result.generation = result.frame.generation;
        result.frame_dequeue_time_ns = task.frame_dequeue_time_ns;
        result.debug_frame = task.want_debug_copies;
        result.dropped = false;
        result.evm_valid = false;
        result.has_constellation = false;
        result.cfo_sfo_estimate_valid = false;
        result.alpha = 0.0f;
        result.beta = 0.0f;
        result.profile.reset();
        result.profile.frame_count = 1;

        // Stale-generation fast drop: a resync already invalidated this frame.
        // Skip the DSP but still emit the result token so the per-slot pending
        // accounting stays gapless; the collector releases the frame.
        if (result.generation != _sync_generation.load(std::memory_order_acquire)) {
            LOG_RT_WARN_HZ_M(Recovery, 5)
                << "[UE recovery] dropping stale demod task before DSP"
                << ": frame_generation=" << result.generation
                << ", current_generation="
                << _sync_generation.load(std::memory_order_relaxed);
            result.dropped = true;
            return;
        }

        const bool sensing_enabled = static_cast<bool>(_bistatic_sensing_channel);
        if (sensing_enabled) {
            result.sense_frame = _sensing_frame_pool.acquire();
            result.sense_frame.rx_symbols.resize(cfg_.ofdm.num_symbols);
            result.sense_frame.tx_symbols.resize(cfg_.ofdm.num_symbols);
            result.has_sensing = true;
        }

        auto prof_step_start = ProfileClock::now();
        auto prof_step_end = prof_step_start;
        const size_t scale_n = cfg_.ofdm.fft_size;
        const float scale = 1.0f / sqrtf(static_cast<float>(scale_n));
        size_t pos = 0;
        for (size_t i = 0; i < cfg_.ofdm.num_symbols; ++i) {
            const bool is_sync = (i == cfg_.ofdm.sync_pos);
            const int data_idx_int = is_sync ? -1 : _data_resource_layout.actual_symbol_to_data_symbol[i];
            std::complex<float>* __restrict__ dst = nullptr;
            if (is_sync) {
                dst = ctx.sync_symbol_freq.data();
            } else if (data_idx_int >= 0) {
                dst = ctx.symbols[static_cast<size_t>(data_idx_int)].data();
            } else if (i < _data_resource_layout.midframe_pilot_symbol_to_rank.size()) {
                const int pilot_rank = _data_resource_layout.midframe_pilot_symbol_to_rank[i];
                if (pilot_rank >= 0) {
                    dst = ctx.midframe_symbol_freqs[static_cast<size_t>(pilot_rank)].data();
                }
            }

            std::copy(result.frame.frame_data.begin() + pos + cfg_.ofdm.cp_length,
                      result.frame.frame_data.begin() + pos + cfg_.ofdm.cp_length + cfg_.ofdm.fft_size,
                      ctx.fft_input.begin());
            fftwf_execute(ctx.fft_plan);
            const std::complex<float>* __restrict__ src = ctx.fft_output.data();
            if (sensing_enabled) {
                std::complex<float>* __restrict__ rx = result.sense_frame.rx_symbols[i].data();
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
        result.profile.fft_total += std::chrono::duration<double, std::micro>(
            prof_step_end - prof_step_start).count();

        prof_step_start = ProfileClock::now();
        ctx.channel_estimator.estimate_from_sync_lmmse(
            ctx.sync_symbol_freq,
            zc_freq_,
            result.h_est,
            cfg_.ofdm.cp_length,
            &result.corrected_impulse_snr_linear_est);

        // Refill the fixed, pre-sized channel anchors in place (anchor symbol
        // order is precomputed in _init_cpu_demod_workers; no per-frame
        // allocation).
        for (size_t a = 0; a < ctx.channel_anchor_symbols.size(); ++a) {
            const int source = ctx.channel_anchor_source[a];
            if (source < 0) {
                std::copy(result.h_est.begin(), result.h_est.end(),
                          ctx.channel_anchor_h[a].begin());
            } else {
                _estimate_midframe_pilot_ls(
                    ctx.midframe_symbol_freqs[static_cast<size_t>(source)],
                    _midframe_pilot_seqs[static_cast<size_t>(source)],
                    ctx.channel_anchor_h[a]);
            }
        }
        prof_step_end = ProfileClock::now();
        result.profile.channel_est_total += std::chrono::duration<double, std::micro>(
            prof_step_end - prof_step_start).count();

        prof_step_start = ProfileClock::now();
        result.cfo_sfo_estimate_valid = FrequencyOffsetEstimator::compute_pilot_phase_diff(
            ctx.symbols,
            cfg_.ofdm.pilot_positions,
            cfg_.ofdm.fft_size,
            cfg_.ofdm.sync_pos,
            ctx.pilot_indices,
            ctx.avg_phase_diff,
            ctx.weights,
            &_data_resource_layout.data_symbol_to_actual_symbol,
            &_cfo_symbol_skip_mask);
        if (result.cfo_sfo_estimate_valid) {
            unwrap(ctx.avg_phase_diff);
            std::tie(result.beta, result.alpha) =
                weightedlinearRegression(ctx.pilot_indices, ctx.avg_phase_diff, ctx.weights);
            result.cfo_sfo_estimate_valid =
                std::isfinite(result.beta) && std::isfinite(result.alpha);
        }
        result.detected_freq_offset = FrequencyOffsetEstimator::alpha_to_cfo(
            result.alpha, cfg_.ofdm.fft_size, cfg_.ofdm.cp_length, cfg_.rf_sampling.sample_rate);
        prof_step_end = ProfileClock::now();
        result.profile.cfo_sfo_est_total += std::chrono::duration<double, std::micro>(
            prof_step_end - prof_step_start).count();

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
        const bool do_eq_breakdown =
            LOG_MOD_ON(DemodEqProfiling);
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
        const double equalized_noise_var = noise_variance_from_snr_linear(
            std::max<double>(result.corrected_impulse_snr_linear_est, 1e-6));
        const double equalizer_noise_var_fallback =
            equalized_noise_var * _average_channel_power(result.h_est);
        const double equalizer_noise_var_est = (cfg_.downlink.equalizer.equalizer_mode == kEqualizerModeMmse)
            ? _estimate_equalizer_noise_var_from_pilots(
                ctx,
                ctx.symbols,
                result.h_est,
                result.alpha,
                result.beta,
                equalizer_noise_var_fallback)
            : 0.0;
        const float equalizer_noise_var = (cfg_.downlink.equalizer.equalizer_mode == kEqualizerModeMmse)
            ? static_cast<float>(equalizer_noise_var_est)
            : 0.0f;

        bool last_anchor_H_inv_valid = false;
        auto prof_eq_sub_start = eq_tick();
        _compute_channel_inverse(result.h_est, ctx.h_inv, equalizer_noise_var);
        auto prof_eq_sub_end = eq_tick();
        eq_accum(prof_eq_base_inv_us, prof_eq_sub_start, prof_eq_sub_end);
        const bool use_symbol_tracking =
            cfg_.downlink.equalizer.channel_tracking_mode != kChannelTrackingModeOff &&
            !cfg_.ofdm.pilot_positions.empty();
        for (size_t i = 0; i < ctx.symbols.size(); ++i) {
            ++prof_eq_data_symbols;
            auto& symbol = ctx.symbols[i];
            const int actual_symbol = _data_resource_layout.data_symbol_to_actual_symbol[i];
            prof_eq_sub_start = eq_tick();
            const AlignedVector& symbol_H_base =
                _interpolated_channel_for_symbol(ctx, actual_symbol, result.h_est);
            prof_eq_sub_end = eq_tick();
            eq_accum(prof_eq_channel_select_us, prof_eq_sub_start, prof_eq_sub_end);

            const int relative_symbol_index = actual_symbol - static_cast<int>(cfg_.ofdm.sync_pos);
            const bool using_midframe_channel = (&symbol_H_base != &result.h_est);
            if (using_midframe_channel) {
                ++prof_eq_midframe_channel_symbols;
            }
            float phase_diff_CFO = using_midframe_channel ? 0.0f : (result.alpha * relative_symbol_index);
            float beta_rel = using_midframe_channel ? 0.0f : (result.beta * relative_symbol_index);
            const AlignedVector* symbol_H_inv = &ctx.h_inv;
            const bool using_last_anchor_channel =
                using_midframe_channel &&
                !ctx.channel_anchor_h.empty() &&
                (&symbol_H_base == &ctx.channel_anchor_h.back());
            if (&symbol_H_base != &result.h_est) {
                if (using_last_anchor_channel) {
                    if (!last_anchor_H_inv_valid) {
                        ++prof_eq_symbol_inv_count;
                        prof_eq_sub_start = eq_tick();
                        _compute_channel_inverse(symbol_H_base, ctx.last_anchor_h_inv, equalizer_noise_var);
                        prof_eq_sub_end = eq_tick();
                        eq_accum(prof_eq_symbol_inv_us, prof_eq_sub_start, prof_eq_sub_end);
                        last_anchor_H_inv_valid = true;
                    }
                    symbol_H_inv = &ctx.last_anchor_h_inv;
                } else {
                    ++prof_eq_symbol_inv_count;
                    prof_eq_sub_start = eq_tick();
                    _compute_channel_inverse(symbol_H_base, ctx.tracking_h_inv, equalizer_noise_var);
                    prof_eq_sub_end = eq_tick();
                    eq_accum(prof_eq_symbol_inv_us, prof_eq_sub_start, prof_eq_sub_end);
                    symbol_H_inv = &ctx.tracking_h_inv;
                }
            }
            if (use_symbol_tracking) {
                float tracked_beta = 0.0f;
                float tracked_alpha = 0.0f;
                ++prof_eq_pilot_phase_attempt;
                prof_eq_sub_start = eq_tick();
                if (_fit_symbol_pilot_phase(ctx, symbol, symbol_H_base, tracked_beta, tracked_alpha)) {
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
        result.profile.equalization_total += std::chrono::duration<double, std::micro>(
            prof_step_end - prof_step_start).count();
        result.profile.eq_base_inv_total += prof_eq_base_inv_us;
        result.profile.eq_channel_select_total += prof_eq_channel_select_us;
        result.profile.eq_symbol_inv_total += prof_eq_symbol_inv_us;
        result.profile.eq_pilot_phase_total += prof_eq_pilot_phase_us;
        result.profile.eq_apply_total += prof_eq_apply_us;
        result.profile.eq_data_symbols_total += prof_eq_data_symbols;
        result.profile.eq_midframe_channel_symbols_total += prof_eq_midframe_channel_symbols;
        result.profile.eq_symbol_inv_count_total += prof_eq_symbol_inv_count;
        result.profile.eq_pilot_phase_attempt_total += prof_eq_pilot_phase_attempt;
        result.profile.eq_pilot_phase_success_total += prof_eq_pilot_phase_success;

        prof_step_start = ProfileClock::now();
        if (sensing_enabled) {
            for (size_t i = 0; i < cfg_.ofdm.num_symbols; ++i) {
                if (is_zc_sync_symbol(cfg_, i)) {
                    result.sense_frame.tx_symbols[i] = zc_freq_;
                } else if (is_cfo_training_symbol(cfg_, i)) {
                    result.sense_frame.tx_symbols[i] = _cfo_training_seq;
                } else if (i < _data_resource_layout.midframe_pilot_symbol_to_rank.size() &&
                           _data_resource_layout.midframe_pilot_symbol_to_rank[i] >= 0) {
                    const size_t pilot_rank = static_cast<size_t>(
                        _data_resource_layout.midframe_pilot_symbol_to_rank[i]);
                    if (pilot_rank < _midframe_pilot_seqs.size()) {
                        result.sense_frame.tx_symbols[i] = _midframe_pilot_seqs[pilot_rank];
                    }
                } else {
                    const int symbol_idx_int = _data_resource_layout.actual_symbol_to_data_symbol[i];
                    if (symbol_idx_int < 0) {
                        continue;
                    }
                    const size_t symbol_idx = static_cast<size_t>(symbol_idx_int);
                    QPSKModulator::remodulate_symbol(
                        ctx.symbols[symbol_idx],
                        zc_freq_,
                        cfg_.ofdm.pilot_positions,
                        result.sense_frame.tx_symbols[i]);
                    const size_t non_pilot_base = _data_resource_layout.non_pilot_offsets[symbol_idx];
                    for (size_t di = 0; di < _data_resource_layout.num_non_pilot_subcarriers; ++di) {
                        const size_t flat_idx = non_pilot_base + di;
                        if (_data_resource_layout.sensing_pilot_mask[flat_idx] == 0) {
                            continue;
                        }
                        const size_t k = static_cast<size_t>(_data_resource_layout.non_pilot_subcarrier_indices[di]);
                        result.sense_frame.tx_symbols[i][k] = sensing_pilot_freq_[k];
                    }
                }
            }
        }
        prof_step_end = ProfileClock::now();
        result.profile.remodulate_total += std::chrono::duration<double, std::micro>(
            prof_step_end - prof_step_start).count();

        prof_step_start = ProfileClock::now();
        ctx.delay_processor.compute_delay_spectrum(result.h_est, result.delay_spectrum);
        DelayProcessor::find_peak(
            result.delay_spectrum,
            result.delay_max_index,
            result.delay_max_mag,
            result.delay_average_mag,
            cfg_.ofdm.cp_length);
        result.adjusted_delay_index =
            DelayProcessor::adjust_delay_index(result.delay_max_index, cfg_.ofdm.fft_size);
        result.fractional_delay =
            DelayProcessor::estimate_fractional_delay(result.delay_spectrum, result.delay_max_index);
        prof_step_end = ProfileClock::now();
        result.profile.delay_spectrum_total += std::chrono::duration<double, std::micro>(
            prof_step_end - prof_step_start).count();

        prof_step_start = ProfileClock::now();
        const float scale_llr = std::min(task.llr_scale_snapshot * static_cast<float>(M_SQRT1_2), 500.0f);
        if (_measurement_enabled &&
            _measurement_active_epoch_id.load(std::memory_order_relaxed) != 0) {
            // Stats only; recording happens on the control thread right after
            // it updates the matching per-frame SNR (avoids the cross-thread
            // read of non-atomic _snr_db and keeps SNR/EVM pairing per-frame).
            result.evm = _compute_frame_evm_stats(ctx.symbols);
            result.evm_valid = true;
        }
        if (_data_resource_layout.payload_re_count > 0) {
            if (_ldpc_fixed_point) {
                const float scale_llr_q =
                    scale_llr * static_cast<float>(cfg_.ldpc.fixed_point_scale);
                result.llr_i16 = _llr_pool_i16.acquire();
                int16_t* __restrict__ llr_ptr = result.llr_i16.data();
                for (size_t sym_idx = 0; sym_idx < ctx.symbols.size(); ++sym_idx) {
                    const auto* __restrict__ sym_ptr = ctx.symbols[sym_idx].data();
                    const size_t payload_begin = _data_resource_layout.payload_offsets[sym_idx];
                    const size_t payload_end = _data_resource_layout.payload_offsets[sym_idx + 1];
                    size_t llr_offset = payload_begin * 2;
                    for (size_t idx = payload_begin; idx < payload_end; ++idx) {
                        const size_t k = static_cast<size_t>(_payload_subcarrier_indices_flat[idx]);
                        llr_ptr[llr_offset++] = sat16_llr(sym_ptr[k].real() * scale_llr_q);
                        llr_ptr[llr_offset++] = sat16_llr(sym_ptr[k].imag() * scale_llr_q);
                    }
                }
                result.has_llr_i16 = true;
            } else {
                result.llr = _llr_pool.acquire();
                float* __restrict__ llr_ptr = result.llr.data();
                for (size_t sym_idx = 0; sym_idx < ctx.symbols.size(); ++sym_idx) {
                    const auto* __restrict__ sym_ptr = ctx.symbols[sym_idx].data();
                    const size_t payload_begin = _data_resource_layout.payload_offsets[sym_idx];
                    const size_t payload_end = _data_resource_layout.payload_offsets[sym_idx + 1];
                    size_t llr_offset = payload_begin * 2;
                    for (size_t idx = payload_begin; idx < payload_end; ++idx) {
                        const size_t k = static_cast<size_t>(_payload_subcarrier_indices_flat[idx]);
                        llr_ptr[llr_offset++] = sym_ptr[k].real() * scale_llr;
                        llr_ptr[llr_offset++] = sym_ptr[k].imag() * scale_llr;
                    }
                }
                result.has_llr_float = true;
            }
        }
        prof_step_end = ProfileClock::now();
        result.profile.llr_total += std::chrono::duration<double, std::micro>(
            prof_step_end - prof_step_start).count();

        if (task.want_debug_copies && !ctx.symbols.empty()) {
            const AlignedVector& last_symbol = ctx.symbols.back();
            std::copy(last_symbol.begin(), last_symbol.end(),
                      result.constellation_symbol.begin());
            result.has_constellation = true;
        }
    }

    // Returns pooled objects held by a result to their pools. The result's
    // pre-sized slot buffers (h_est / delay_spectrum / constellation_symbol)
    // are left untouched so the ring slot keeps its capacity.
    void _release_cpu_demod_result(CpuDemodResult& result) {
        if (!result.frame.frame_data.empty()) {
            _rx_frame_pool.release(std::move(result.frame));
        }
        if (result.has_sensing) {
            _sensing_frame_pool.release(std::move(result.sense_frame));
            result.has_sensing = false;
        }
        if (result.has_llr_float) {
            _llr_pool.release(std::move(result.llr));
            result.has_llr_float = false;
        }
        if (result.has_llr_i16) {
            _llr_pool_i16.release(std::move(result.llr_i16));
            result.has_llr_i16 = false;
        }
    }

    void _merge_cpu_demod_profile(CpuDemodProfile& acc, const CpuDemodProfile& src) {
        acc.fft_total += src.fft_total;
        acc.channel_est_total += src.channel_est_total;
        acc.cfo_sfo_est_total += src.cfo_sfo_est_total;
        acc.equalization_total += src.equalization_total;
        acc.eq_base_inv_total += src.eq_base_inv_total;
        acc.eq_channel_select_total += src.eq_channel_select_total;
        acc.eq_symbol_inv_total += src.eq_symbol_inv_total;
        acc.eq_pilot_phase_total += src.eq_pilot_phase_total;
        acc.eq_apply_total += src.eq_apply_total;
        acc.eq_data_symbols_total += src.eq_data_symbols_total;
        acc.eq_midframe_channel_symbols_total += src.eq_midframe_channel_symbols_total;
        acc.eq_symbol_inv_count_total += src.eq_symbol_inv_count_total;
        acc.eq_pilot_phase_attempt_total += src.eq_pilot_phase_attempt_total;
        acc.eq_pilot_phase_success_total += src.eq_pilot_phase_success_total;
        acc.noise_est_total += src.noise_est_total;
        acc.remodulate_total += src.remodulate_total;
        acc.delay_spectrum_total += src.delay_spectrum_total;
        acc.timing_sync_total += src.timing_sync_total;
        acc.sensing_queue_total += src.sensing_queue_total;
        acc.udp_send_total += src.udp_send_total;
        acc.llr_total += src.llr_total;
        acc.frame_count += src.frame_count;
    }

    bool _collect_cpu_demod_result(CpuDemodResult& result, CpuDemodProfile& prof_acc) {
        using ProfileClock = std::chrono::high_resolution_clock;
        if (result.dropped) {
            _release_cpu_demod_result(result);
            return true;
        }
        if (result.generation != _sync_generation.load(std::memory_order_acquire)) {
            LOG_RT_WARN_HZ_M(Recovery, 5)
                << "[UE recovery] dropping completed demod result after generation change"
                << ": frame_generation=" << result.generation
                << ", current_generation="
                << _sync_generation.load(std::memory_order_relaxed);
            _release_cpu_demod_result(result);
            return true;
        }

        auto prof_step_start = ProfileClock::now();
        auto prof_step_end = prof_step_start;
        const bool do_latency_profile =
            LOG_MOD_ON(DemodProfiling);
        const RxFrame& frame = result.frame;
        const bool allow_freq_adjust = _control_time_gates.allow_freq_adjust(frame.usrp_time_ns);
        bool issued_freq_adjust = false;
        bool cfo_observation_valid = result.cfo_sfo_estimate_valid;
        if (cfo_observation_valid && !allow_freq_adjust) {
            cfo_observation_valid = false;
        }
        const double tune_system_cfo_hz = rx_tune_system_cfo_hz(
            cfg_.downlink.center_freq,
            current_rx_tune_.actual_rf_freq,
            current_rx_tune_.actual_dsp_freq);
        const double normal_cfo_alias_period_hz =
            cfg_.rf_sampling.sample_rate /
            static_cast<double>(cfg_.ofdm.fft_size + cfg_.ofdm.cp_length);
        const auto normal_cfo = dealias_normal_cfo_hz(
            static_cast<double>(result.detected_freq_offset),
            normal_cfo_alias_period_hz,
            tune_system_cfo_hz);
        cfo_observation_valid = cfo_observation_valid && normal_cfo.valid;
        const bool ocxo_cfo_quality_valid = _ocxo_cfo_quality_gate.update(
            normal_cfo,
            normal_cfo_alias_period_hz,
            cfo_observation_valid);
        const double detected_cfo_hz = normal_cfo.valid
            ? normal_cfo.dealiased_hz
            : static_cast<double>(result.detected_freq_offset);
        const double clock_error_hz =
            detected_cfo_hz - tune_system_cfo_hz;
        // Mode A residual for OCXO: plant-oriented (∂e/∂u = -1 with local UE
        // OCXO). Do not invert here — Mode A closed-loop polarity is correct.
        // Mode B TO_clock slope polarity is handled inside OCXODiscipliner.
        const double raw_error_ppm =
            (std::abs(cfg_.downlink.center_freq) > 0.0)
                ? (clock_error_hz / cfg_.downlink.center_freq * 1e6)
                : 0.0;
        double control_error_ppm = raw_error_ppm;
        if (cfg_.sync_tracking.hardware_sync && ocxo_cfo_quality_valid) {
            const auto ewma_result = _cfo_ewma.update(raw_error_ppm);
            control_error_ppm = cfg_.sync_tracking.cfo_ewma_enable
                ? ewma_result.filtered_error_ppm
                : raw_error_ppm;
        }
        if (cfo_observation_valid) {
            _freq_offset_sum += detected_cfo_hz;
            _freq_offset_count++;
            if (_freq_offset_count >= 434) {
                _avg_freq_offset = _freq_offset_sum / _freq_offset_count;
                _freq_offset_sum = 0.0;
                _freq_offset_count = 0;
                if (std::abs(_avg_freq_offset) > 2.0f &&
                    cfg_.sync_tracking.software_sync && allow_freq_adjust) {
                    LOG_RT_INFO_M(Sync) << "Adjusting RX frequency by: " << _avg_freq_offset << " Hz";
                    adjust_rx_freq(-_avg_freq_offset, false);
                    issued_freq_adjust = true;
                }
            }
        } else if (!result.cfo_sfo_estimate_valid) {
            LOG_RT_WARN_HZ(2) << "Skipping CFO/SFO update: no clean adjacent downlink pilot-symbol pairs";
        }
        if (issued_freq_adjust) {
            _control_time_gates.mark_freq_adjust_now(radio_time_now());
        }
        prof_step_end = ProfileClock::now();
        result.profile.cfo_sfo_est_total += std::chrono::duration<double, std::micro>(
            prof_step_end - prof_step_start).count();

        prof_step_start = ProfileClock::now();
        _snr_linear = std::max<double>(result.corrected_impulse_snr_linear_est, 1e-6);
        _snr_db = 10.0 * std::log10(_snr_linear);
        const double llr_snr_linear = _update_llr_snr_filter(_snr_linear);
        _noise_var = std::max(noise_variance_from_snr_linear(llr_snr_linear), 1e-6);
        _llr_scale = 4.0 / _noise_var;
        _llr_scale_snapshot.store(
            static_cast<float>(_llr_scale),
            std::memory_order_release);
        if (result.evm_valid) {
            // Recorded here (not in the worker) so the EVM pairs with the SNR
            // just updated for this same frame and _snr_db stays
            // control-thread-private.
            _record_measurement_frame(result.evm);
            result.evm_valid = false;
        }
        prof_step_end = ProfileClock::now();
        result.profile.noise_est_total += std::chrono::duration<double, std::micro>(
            prof_step_end - prof_step_start).count();

        prof_step_start = ProfileClock::now();
        _apply_pending_process_tracking_reset();
        bool alignment_from_sidechannel = false;
        const int32_t effective_alignment = _effective_alignment_for_frame(
            frame.Alignment,
            frame.usrp_time_ns,
            &alignment_from_sidechannel);
        if (alignment_from_sidechannel && LOG_MOD_ON(Sync)) {
            LOG_RT_INFO_M(Sync)
                << "[delay_track] alignment_sidechannel_inject"
                << " samples=" << effective_alignment
                << " applied_id=" << _alignment_applied_consumed_id
                << " frame_time_ns=" << frame.usrp_time_ns
                << " (queue-dropped Alignment frame; full SFO/eRTM semantics)";
        }
        const float peak_ratio =
            (result.delay_average_mag > 0.0f)
                ? (result.delay_max_mag / result.delay_average_mag)
                : 0.0f;
        const bool bad_peak =
            (result.delay_average_mag > 0.0f) && (peak_ratio < 20.0f);
        const bool delay_peak_valid =
            (result.delay_average_mag > 0.0f) && !bad_peak;
        if (delay_peak_valid) {
            sfo_estimator.update(
                result.adjusted_delay_index + result.fractional_delay,
                static_cast<float>(effective_alignment));
        } else {
            sfo_estimator.apply_alignment_without_observation(
                static_cast<float>(effective_alignment));
        }
        const auto sfo_per_frame = sfo_estimator.get_sfo_per_frame();
        const float delay_offset = sfo_estimator.get_sensing_delay_offset();

        // OFDM delay-track SFO stays diagnostic here; Mode B control uses the
        // Alignment-continuous eRTM TO_clock slope inside OCXODiscipliner.
        if (cfg_.sync_tracking.hardware_sync && _hw_sync && _ocxo_discipliner &&
            _ocxo_pps_initial_aligner) {
            _apply_pending_ocxo_to_reanchor(allow_freq_adjust);
            OCXODiscipliner::Observation ocxo_obs;
            ocxo_obs.allow_control = allow_freq_adjust;
            ocxo_obs.cfo_valid = ocxo_cfo_quality_valid;
            ocxo_obs.cfo_ppm = control_error_ppm;
            const double spf = static_cast<double>(cfg_.samples_per_frame());
            const double ofdm_sfo_ppm = (spf > 0.0)
                ? (static_cast<double>(sfo_per_frame) / spf * 1.0e6)
                : 0.0;
            ErtmToPublishSnapshot to_snap;
            bool to_ok = false;
            if (cfg_.uplink.ertm_to_enable) {
                // Atomic same-frame TO/Alignment/identity snapshot.
                to_snap = _load_ertm_to_publish_snapshot();
                to_ok =
                    to_snap.available &&
                    to_snap.clear_gen == to_snap.request_gen &&
                    std::isfinite(to_snap.to_ue_samples);
                if (to_ok) {
                    if (!_ocxo_to_pub_known ||
                        to_snap.pub_gen != _ocxo_to_last_pub_gen) {
                        _ocxo_to_pub_known = true;
                        _ocxo_to_last_pub_gen = to_snap.pub_gen;
                        _ocxo_to_age_frames = 0;
                        ocxo_obs.to_is_new = true;
                    } else {
                        ++_ocxo_to_age_frames;
                        ocxo_obs.to_is_new = false;
                    }
                    ocxo_obs.to_valid = true;
                    ocxo_obs.to_ue_samples = to_snap.to_ue_samples;
                    ocxo_obs.to_alignment_samples =
                        to_snap.rx_alignment_shift_samples;
                    ocxo_obs.to_timeline_epoch =
                        to_snap.dl_identity.timeline_epoch;
                    ocxo_obs.to_frame_seq = to_snap.dl_identity.frame_seq;
                    ocxo_obs.to_sync_generation = to_snap.sync_generation;
                    ocxo_obs.to_report_age_frames = _ocxo_to_age_frames;
                    ocxo_obs.pps_phase_valid =
                        to_snap.pps_phase_available &&
                        std::isfinite(to_snap.pps_phase_error_samples);
                    ocxo_obs.pps_phase_error_samples =
                        to_snap.pps_phase_error_samples;
                } else {
                    _ocxo_to_pub_known = false;
                    _ocxo_to_age_frames = 0;
                }
            } else {
                _ocxo_to_pub_known = false;
                _ocxo_to_age_frames = 0;
            }

            const auto reacquisition =
                _ocxo_discipliner->prepare_reacquisition_resume(
                    ocxo_obs,
                    OCXODiscipliner::kReacquisitionModeHoldS);
            if (reacquisition.action ==
                OCXODiscipliner::ReacquisitionAction::Waiting) {
                ocxo_obs.allow_control = false;
            } else if (reacquisition.action ==
                       OCXODiscipliner::ReacquisitionAction::Reset) {
                _hw_sync->rebase_frequency_control_slew();
                if (cfg_.sync_tracking.ocxo_mode == kOCXOModePpsPhase) {
                    _hw_sync->rearm_pps_phase_adjust();
                    _ocxo_pps_initial_aligner->reset();
                }
                LOG_G_WARN_M(Ocxo)
                    << "OCXO reacquisition exceeded mode-preserve window; tracking reset"
                    << " previous_stage="
                    << OCXODiscipliner::stage_name(reacquisition.previous_stage)
                    << " elapsed_s=" << reacquisition.elapsed_s
                    << " preserve_s="
                    << OCXODiscipliner::kReacquisitionModeHoldS
                    << " retained_ppm=" << _hw_sync->current_offset_ppm();
            } else if (reacquisition.elapsed_s > 0.0) {
                LOG_G_INFO_M(Ocxo)
                    << "OCXO reacquisition resumed with retained tracking mode"
                    << " stage="
                    << OCXODiscipliner::stage_name(reacquisition.previous_stage)
                    << " elapsed_s=" << reacquisition.elapsed_s
                    << " preserve_s="
                    << OCXODiscipliner::kReacquisitionModeHoldS;
            }

            const bool pps_phase_mode =
                cfg_.sync_tracking.ocxo_mode == kOCXOModePpsPhase;
            OCXOPpsInitialAligner::UpdateResult align_out;
            align_out.state = _ocxo_pps_initial_aligner->state();
            align_out.phase_control_ready = true;
            bool align_state_changed = false;
            if (pps_phase_mode) {
                const auto map_ap_status = [](
                    HardwareSyncController::PpsPhaseAdjustStatus status) {
                    using HwStatus = HardwareSyncController::PpsPhaseAdjustStatus;
                    using AlignStatus = OCXOPpsInitialAligner::CommandStatus;
                    switch (status) {
                        case HwStatus::Idle: return AlignStatus::Idle;
                        case HwStatus::Queued: return AlignStatus::Queued;
                        case HwStatus::InFlight: return AlignStatus::InFlight;
                        case HwStatus::Succeeded: return AlignStatus::Succeeded;
                        case HwStatus::Failed: return AlignStatus::Failed;
                    }
                    return AlignStatus::Failed;
                };
                OCXOPpsInitialAligner::Observation align_obs;
                align_obs.phase_stage =
                    _ocxo_discipliner->stage() == OCXODiscipliner::Stage::Phase;
                align_obs.phase_valid = ocxo_obs.pps_phase_valid;
                align_obs.phase_error_samples = ocxo_obs.pps_phase_error_samples;
                align_obs.sample_rate = cfg_.rf_sampling.sample_rate;
                align_obs.publication_generation = to_ok ? to_snap.pub_gen : 0;
                align_obs.command_status = map_ap_status(
                    _hw_sync->pps_phase_adjust_status());
                align_out = _ocxo_pps_initial_aligner->update(align_obs);
                align_state_changed = align_out.state_changed;
                if (align_out.request_hardware_rearm) {
                    _hw_sync->rearm_pps_phase_adjust();
                    if (align_out.phase_entry_rearm) {
                        LOG_G_INFO_M(Ocxo)
                            << "OCXO PPS AP rearmed for FllSlow-to-Phase entry";
                    } else {
                        LOG_G_WARN_M(Ocxo)
                            << "OCXO PPS AP settling timed out; waiting for "
                               "a fresh PPS publication before bounded retry"
                            << ": retry=" << align_out.retry_count;
                    }
                }
                if (align_out.request_adjust) {
                    using HwQueueResult =
                        HardwareSyncController::PpsPhaseAdjustQueueResult;
                    const HwQueueResult queue_result =
                        _hw_sync->request_pps_phase_adjust_ns(
                        align_out.adjustment_ns);
                    const auto align_queue_result =
                        queue_result == HwQueueResult::Queued
                            ? OCXOPpsInitialAligner::QueueResult::Queued
                            : OCXOPpsInitialAligner::QueueResult::Busy;
                    align_state_changed |=
                        _ocxo_pps_initial_aligner->notify_command_queue_result(
                            align_queue_result);
                    align_out.state = _ocxo_pps_initial_aligner->state();
                    align_out.phase_control_ready =
                        _ocxo_pps_initial_aligner->phase_control_ready();
                    LOG_G_INFO_M(Ocxo)
                        << "OCXO phase-entry PPS AP request"
                        << ": pps_error_samples=" << ocxo_obs.pps_phase_error_samples
                        << ", adjustment_ns=" << align_out.adjustment_ns
                        << ", queue_result="
                        << (queue_result == HwQueueResult::Queued ? "queued" : "busy")
                        << ", retry=" << align_out.retry_count
                        << " (AP changes PPSOUT only; USRP/TO anchors unchanged)";
                }
                ocxo_obs.phase_control_ready = align_out.phase_control_ready;
            }
            if (align_state_changed) {
                if (align_out.state == OCXOPpsInitialAligner::State::Failed) {
                    LOG_G_WARN_M(Ocxo)
                        << "OCXO phase-entry PPS AP failed; Phase SF tracking is released"
                        << ": pps_error_samples=" << ocxo_obs.pps_phase_error_samples
                        << ", retry=" << align_out.retry_count
                        << ". A large residual may remain and the slew-limited "
                           "Phase PI may not remove it quickly.";
                } else {
                    LOG_G_INFO_M(Ocxo) << "OCXO phase-entry PPS AP state="
                                 << OCXOPpsInitialAligner::state_name(align_out.state);
                }
            }
            const auto ocxo_out = _ocxo_discipliner->update(ocxo_obs);
            double applied_delta_ppm = 0.0;
            if (ocxo_out.request_control) {
                applied_delta_ppm =
                    _hw_sync->apply_delta_ppm(ocxo_out.desired_delta_ppm);
                _ocxo_discipliner->commit_applied_delta(applied_delta_ppm);
                _cfo_ewma.notify_control_action(applied_delta_ppm);
            }
            const auto ocxo_debug = _ocxo_discipliner->debug_state();
            if (ocxo_debug_sender_) {
                const double alignment_event_samples = ocxo_obs.to_is_new
                    ? ocxo_debug.alignment_feedforward_samples
                    : 0.0;
                const ocxo_debug_protocol::Channels channels{
                    static_cast<float>(normal_cfo.modulo_hz),
                    static_cast<float>(detected_cfo_hz),
                    static_cast<float>(raw_error_ppm),
                    static_cast<float>(control_error_ppm),
                    static_cast<float>(ofdm_sfo_ppm),
                    static_cast<float>(ocxo_debug.to_raw),
                    static_cast<float>(ocxo_debug.to_clock),
                    static_cast<float>(ocxo_debug.to_err),
                    static_cast<float>(ocxo_debug.to_slope_ppm),
                    static_cast<float>(ocxo_obs.to_alignment_samples),
                    static_cast<float>(alignment_event_samples),
                    static_cast<float>(ocxo_out.desired_delta_ppm),
                    static_cast<float>(applied_delta_ppm),
                    static_cast<float>(ocxo_debug.command_ppm),
                    static_cast<float>(ocxo_debug.coarse_outstanding_ppm),
                    static_cast<float>(OCXODiscipliner::stage_code(ocxo_debug.stage))
                };
                ocxo_debug_protocol::Diagnostics diagnostics;
                if (to_ok && to_snap.pps_phase_available) {
                    diagnostics.r_bs_samples = to_snap.r_bs_samples;
                    diagnostics.r_ue_samples = to_snap.r_ue_samples;
                    diagnostics.bs_last_pps_ticks = to_snap.bs_last_pps_ticks;
                    diagnostics.bs_t0_to_pps_ticks = to_snap.bs_t0_to_pps_ticks;
                    diagnostics.ue_last_pps_ns = to_snap.ue_last_pps_ns;
                    diagnostics.bs_tick_rate = to_snap.bs_tick_rate;
                    diagnostics.sample_rate = to_snap.sample_rate;
                }
                const bool force_debug_event =
                    ocxo_out.request_control ||
                    ocxo_out.transition_reason !=
                        OCXODiscipliner::TransitionReason::None ||
                    ocxo_out.to_jump_rejected ||
                    align_state_changed ||
                    std::abs(alignment_event_samples) > 1e-15;
                ocxo_debug_sender_->send_channels(
                    channels, diagnostics, force_debug_event);
            }
            ++_ocxo_log_counter;
            if (ocxo_out.transition_reason != OCXODiscipliner::TransitionReason::None ||
                ocxo_out.to_jump_rejected ||
                align_state_changed ||
                _ocxo_log_counter % 256 == 0) {
                LOG_RT_INFO_M(Ocxo)
                    << "OCXO loop: mode=" << OCXODiscipliner::active_mode_name(ocxo_debug.mode)
                    << " stage=" << OCXODiscipliner::stage_name(ocxo_debug.stage)
                    << " reason=" << OCXODiscipliner::transition_reason_name(ocxo_out.transition_reason)
                    << " cfo_mod_hz=" << normal_cfo.modulo_hz
                    << " cfo_hz=" << detected_cfo_hz
                    << " cfo_ppm=" << control_error_ppm
                    << " ofdm_sfo_ppm=" << ofdm_sfo_ppm
                    << " freq_source="
                    << OCXODiscipliner::frequency_source_name(ocxo_debug.frequency_source)
                    << " freq_error_ppm=" << ocxo_debug.frequency_error_ppm
                    << " cmd_ppm=" << ocxo_debug.command_ppm
                    << " desired_ppm=" << ocxo_out.desired_delta_ppm
                    << " applied_ppm=" << applied_delta_ppm
                    << " coarse_left_ppm=" << ocxo_debug.coarse_outstanding_ppm
                    << " to_raw=" << ocxo_debug.to_raw
                    << " to_clock=" << ocxo_debug.to_clock
                    << " phase_error=" << ocxo_debug.to_err
                    << " phase_entry_ap_state="
                    << (pps_phase_mode
                            ? OCXOPpsInitialAligner::state_name(align_out.state)
                            : "inactive")
                    << " to_slope_ppm=" << ocxo_debug.to_slope_ppm
                    << " align_ff=" << ocxo_debug.alignment_feedforward_samples;
            }
        }
        const size_t sync_symbol_len = cfg_.ofdm.fft_size + cfg_.ofdm.cp_length;
        const size_t sync_symbol_offset = cfg_.ofdm.sync_pos * sync_symbol_len;
        const std::complex<float>* sync_symbol_td = nullptr;
        size_t sync_symbol_td_count = 0;
        if (sync_symbol_offset + sync_symbol_len <= frame.frame_data.size()) {
            sync_symbol_td = frame.frame_data.data() + sync_symbol_offset;
            sync_symbol_td_count = sync_symbol_len;
        }

        int predictive_delay_samples = 0;
        if (cfg_.sync_tracking.predictive_delay) {
            predictive_delay_samples =
                _predictive_delay_samples_from_cfo(
                    cfg_,
                    frame.usrp_time_ns,
                    result.detected_freq_offset,
                    current_rx_tune_.actual_rf_freq,
                    current_rx_tune_.actual_dsp_freq,
                    time_spec_to_ns(radio_time_now()));
        }
        const int delay_index_err =
            result.adjusted_delay_index - cfg_.sync_tracking.desired_peak_pos + predictive_delay_samples;
        const bool allow_reset = _control_time_gates.allow_reset(frame.usrp_time_ns);
        const bool allow_alignment = _control_time_gates.allow_alignment(frame.usrp_time_ns);
        const bool allow_rx_gain_adjust = _control_time_gates.allow_rx_gain_adjust(frame.usrp_time_ns);
        const bool log_agc = LOG_MOD_ON(Agc);
        const bool log_sync = LOG_MOD_ON(Sync);
        const bool log_sync_debug = LOG_MOD_DEBUG_ON(Sync);
        const int abs_delay_err = std::abs(delay_index_err);
        const int delay_adjust_step = cfg_.sync_tracking.delay_adjust_step;
        const int delay_reset_err_thresh = delay_adjust_step + 5;
        const int cp_length = static_cast<int>(cfg_.ofdm.cp_length);
        const bool large_delay_err = abs_delay_err > delay_reset_err_thresh;
        const bool sync_enabled =
            cfg_.sync_tracking.software_sync || cfg_.sync_tracking.hardware_sync;
        const bool err_in_align_window =
            abs_delay_err >= delay_adjust_step && abs_delay_err < cp_length;
        const bool err_beyond_cp = abs_delay_err >= cp_length;
        bool issued_alignment = false;
        bool issued_rx_gain_adjust = false;
        if (allow_rx_gain_adjust) {
            RxAgcAdjustment agc_adjustment;
            issued_rx_gain_adjust = _rx_agc.maybe_apply_from_delay_peak(
                result.delay_max_mag,
                result.delay_average_mag,
                sync_symbol_td,
                sync_symbol_td_count,
                frame.usrp_time_ns,
                _control_time_gates,
                [this](double gain_db) {
                    dev_->set_rx_gain(gain_db, cfg_.downlink.rx_channel);
                },
                &agc_adjustment);
            if (issued_rx_gain_adjust && log_agc) {
                LOG_RT_INFO_M(Agc) << "RX AGC adjusted gain to " << agc_adjustment.next_gain_db
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
        if ((result.delay_average_mag > 0.0f) &&
            (bad_peak || large_delay_err) &&
            sync_enabled && allow_reset) {
            _reset_count++;
            if (log_sync &&
                (_reset_count == 1 ||
                 _reset_count == _reset_hold_frames ||
                 (_reset_count % 32u) == 0u)) {
                LOG_RT_INFO_M(Sync)
                    << "[delay_track] reset_pending count=" << _reset_count
                    << "/" << _reset_hold_frames
                    << " reasons="
                    << (bad_peak ? "bad_peak," : "")
                    << (large_delay_err ? "large_err," : "")
                    << " peak_ratio=" << peak_ratio
                    << " delay_err=" << delay_index_err
                    << " adj_idx=" << result.adjusted_delay_index
                    << " desired=" << cfg_.sync_tracking.desired_peak_pos
                    << " predictive=" << predictive_delay_samples
                    << " frac_delay=" << result.fractional_delay
                    << " align_win=" << (err_in_align_window ? 1 : 0)
                    << " beyond_cp=" << (err_beyond_cp ? 1 : 0)
                    << " sync_in_progress=" << (_sync_in_progress ? 1 : 0)
                    << " hold_frames=" << _sync_in_progress_hold_frames
                    << " allow_align=" << (allow_alignment ? 1 : 0)
                    << " confirm=" << _delay_adjustment_count
                    << " frame_align=" << frame.Alignment
                    << " sfo_per_frame=" << sfo_per_frame;
            }
            if (_reset_count >= _reset_hold_frames) {
                _reset_count = 0;
                LOG_RT_WARN_M(Sync)
                    << "[delay_track] RESET No valid delay found"
                    << " reasons="
                    << (bad_peak ? "bad_peak," : "")
                    << (large_delay_err ? "large_err," : "")
                    << " peak_ratio=" << peak_ratio
                    << " delay_err=" << delay_index_err
                    << " thresh=" << delay_reset_err_thresh
                    << " adj_idx=" << result.adjusted_delay_index
                    << " desired=" << cfg_.sync_tracking.desired_peak_pos
                    << " predictive=" << predictive_delay_samples
                    << " frac_delay=" << result.fractional_delay
                    << " align_win=" << (err_in_align_window ? 1 : 0)
                    << " beyond_cp=" << (err_beyond_cp ? 1 : 0)
                    << " sync_in_progress=" << (_sync_in_progress ? 1 : 0)
                    << " hold_frames=" << _sync_in_progress_hold_frames
                    << " allow_align=" << (allow_alignment ? 1 : 0)
                    << " confirm=" << _delay_adjustment_count
                    << " frame_align=" << frame.Alignment
                    << " sfo_per_frame=" << sfo_per_frame
                    << " max_mag=" << result.delay_max_mag
                    << " avg_mag=" << result.delay_average_mag;
                _reset_for_reacquisition();
                _release_cpu_demod_result(result);
                return false;
            }
        } else {
            _reset_count = 0;
        }
        // Latch clear: frame receipt and/or applied side-channel. SFO/eRTM
        // already used effective_alignment above for full semantics.
        if (_sync_in_progress &&
            _alignment_latch_receipt_ready(frame.Alignment)) {
            const uint64_t applied_id =
                _alignment_applied_id.load(std::memory_order_acquire);
            const int32_t applied_samples =
                _alignment_applied_samples.load(std::memory_order_relaxed);
            const bool receipt_frame = (frame.Alignment != 0);
            const bool receipt_sidechannel =
                (_sync_in_progress_expect_id != 0) &&
                (applied_id == _sync_in_progress_expect_id);
            _sync_in_progress = false;
            _sync_in_progress_hold_frames = 0;
            _sync_in_progress_expect_id = 0;
            if (log_sync) {
                LOG_RT_INFO_M(Sync)
                    << "[delay_track] alignment_feedback_clear"
                    << " frame_align=" << frame.Alignment
                    << " effective_align=" << effective_alignment
                    << " applied_id=" << applied_id
                    << " applied_samples=" << applied_samples
                    << " via="
                    << (receipt_frame
                            ? (receipt_sidechannel ? "frame+sidechannel" : "frame")
                            : "sidechannel")
                    << " delay_err=" << delay_index_err
                    << " adj_idx=" << result.adjusted_delay_index
                    << " peak_ratio=" << peak_ratio
                    << " predictive=" << predictive_delay_samples
                    << " allow_align=" << (allow_alignment ? 1 : 0)
                    << (allow_alignment ? "" : " (cleared despite allow_align=0)");
            }
        }
        if (_sync_in_progress) {
            ++_sync_in_progress_hold_frames;
            if (_sync_in_progress_hold_frames == 8u ||
                _sync_in_progress_hold_frames == 32u ||
                (_sync_in_progress_hold_frames % 64u) == 0u) {
                const uint64_t applied_id =
                    _alignment_applied_id.load(std::memory_order_acquire);
                LOG_RT_WARN_M(Sync)
                    << "[delay_track] sync_in_progress_held"
                    << " hold_frames=" << _sync_in_progress_hold_frames
                    << " delay_err=" << delay_index_err
                    << " frame_align=" << frame.Alignment
                    << " effective_align=" << effective_alignment
                    << " expect_id=" << _sync_in_progress_expect_id
                    << " applied_id=" << applied_id
                    << " allow_align=" << (allow_alignment ? 1 : 0)
                    << " peak_ratio=" << peak_ratio
                    << " (no alignment receipt yet; small align blocked)";
            }
        } else {
            _sync_in_progress_hold_frames = 0;
        }
        if (err_beyond_cp && sync_enabled) {
            LOG_RT_WARN_HZ_M(Sync, 2)
                << "[delay_track] err_beyond_cp delay_err=" << delay_index_err
                << " cp=" << cp_length
                << " adj_idx=" << result.adjusted_delay_index
                << " peak_ratio=" << peak_ratio
                << " (tracking alignment disabled; only hard reset can recover)";
        }
        if (delay_peak_valid && err_in_align_window && sync_enabled) {
            if (!_sync_in_progress && allow_alignment) {
                const uint32_t confirm_before = _delay_adjustment_count;
                if (_delay_adjustment_count++ >= 1) {
                    _delay_adjustment_count = 0;
                    _schedule_receive_alignment(static_cast<int32_t>(delay_index_err));
                    _sync_in_progress = true;
                    _sync_in_progress_hold_frames = 0;
                    _sync_in_progress_expect_id =
                        _pending_alignment_id.load(std::memory_order_relaxed);
                    issued_alignment = true;
                    if (log_sync) {
                        LOG_RT_INFO_M(Sync)
                            << "[delay_track] schedule_alignment"
                            << " samples=" << delay_index_err
                            << " alignment_id=" << _sync_in_progress_expect_id
                            << " adj_idx=" << result.adjusted_delay_index
                            << " desired=" << cfg_.sync_tracking.desired_peak_pos
                            << " predictive=" << predictive_delay_samples
                            << " frac_delay=" << result.fractional_delay
                            << " peak_ratio=" << peak_ratio
                            << " confirm_was=" << confirm_before;
                    }
                } else if (log_sync) {
                    LOG_RT_INFO_HZ_M(Sync, 10)
                        << "[delay_track] confirm_wait"
                        << " delay_err=" << delay_index_err
                        << " confirm=" << _delay_adjustment_count
                        << " need=2"
                        << " adj_idx=" << result.adjusted_delay_index
                        << " peak_ratio=" << peak_ratio;
                }
                if (delay_index_err * _last_delay_index_err < 0) {
                    if (log_sync && _delay_adjustment_count != 0) {
                        LOG_RT_INFO_HZ_M(Sync, 5)
                            << "[delay_track] confirm_reset sign_flip"
                            << " prev_err=" << _last_delay_index_err
                            << " err=" << delay_index_err;
                    }
                    _delay_adjustment_count = 0;
                }
            } else {
                _delay_adjustment_count = 0;
                if (log_sync) {
                    LOG_RT_INFO_HZ_M(Sync, 5)
                        << "[delay_track] align_blocked"
                        << " delay_err=" << delay_index_err
                        << " reasons="
                        << (_sync_in_progress ? "sync_in_progress," : "")
                        << (!allow_alignment ? "allow_align=0," : "")
                        << " hold_frames=" << _sync_in_progress_hold_frames
                        << " frame_align=" << frame.Alignment
                        << " peak_ratio=" << peak_ratio
                        << " adj_idx=" << result.adjusted_delay_index;
                }
            }
        } else {
            if (log_sync && err_in_align_window && sync_enabled && !delay_peak_valid) {
                LOG_RT_INFO_HZ_M(Sync, 5)
                    << "[delay_track] align_blocked"
                    << " delay_err=" << delay_index_err
                    << " reasons=invalid_peak,"
                    << " frame_align=" << frame.Alignment
                    << " peak_ratio=" << peak_ratio
                    << " max_mag=" << result.delay_max_mag
                    << " avg_mag=" << result.delay_average_mag
                    << " adj_idx=" << result.adjusted_delay_index;
            } else if (log_sync &&
                abs_delay_err > 0 &&
                abs_delay_err < delay_adjust_step &&
                sync_enabled) {
                LOG_RT_INFO_HZ_M(Sync, 2)
                    << "[delay_track] dead_zone"
                    << " delay_err=" << delay_index_err
                    << " step=" << delay_adjust_step
                    << " adj_idx=" << result.adjusted_delay_index
                    << " peak_ratio=" << peak_ratio
                    << " (no sample alignment below step)";
            }
            _delay_adjustment_count = 0;
        }
        if (log_sync_debug &&
            (abs_delay_err > 0 || _sync_in_progress || _reset_count > 0)) {
            LOG_RT_DEBUG_HZ_M(Sync, 5)
                << "[delay_track] heartbeat"
                << " delay_err=" << delay_index_err
                << " adj_idx=" << result.adjusted_delay_index
                << " desired=" << cfg_.sync_tracking.desired_peak_pos
                << " predictive=" << predictive_delay_samples
                << " frac_delay=" << result.fractional_delay
                << " peak_ratio=" << peak_ratio
                << " bad_peak=" << (bad_peak ? 1 : 0)
                << " large_err=" << (large_delay_err ? 1 : 0)
                << " align_win=" << (err_in_align_window ? 1 : 0)
                << " beyond_cp=" << (err_beyond_cp ? 1 : 0)
                << " sync_in_progress=" << (_sync_in_progress ? 1 : 0)
                << " hold_frames=" << _sync_in_progress_hold_frames
                << " allow_align=" << (allow_alignment ? 1 : 0)
                << " allow_reset=" << (allow_reset ? 1 : 0)
                << " confirm=" << _delay_adjustment_count
                << " reset_count=" << _reset_count
                << " frame_align=" << frame.Alignment
                << " sfo_per_frame=" << sfo_per_frame;
        }
        _last_delay_index_err = delay_peak_valid ? delay_index_err : 0;
        if (issued_alignment) {
            _control_time_gates.mark_alignment_now(radio_time_now());
        }
        if (issued_rx_gain_adjust) {
            _control_time_gates.mark_rx_gain_adjust_now(radio_time_now());
        }
        prof_step_end = ProfileClock::now();
        result.profile.timing_sync_total += std::chrono::duration<double, std::micro>(
            prof_step_end - prof_step_start).count();

        prof_step_start = ProfileClock::now();
        // Without the bistatic sensing pipeline, keep eRTM/debug functionality
        // by consuming reports here.  With sensing enabled, that thread is the
        // sole consumer of both the report queue and the eRTM FFT plans.
        // A_N comes from the RxFrame freeze at RX enqueue, not the live atomic
        // (which process may already have updated via schedule_receive_alignment).
        const int32_t a_n_samples = frame.ertm_rx_alignment_shift_samples;
        const bool force_ertm_from_align = (effective_alignment != 0);
        if (!_bistatic_sensing_channel) {
            _update_ertm_measurement_for_downlink_frame(
                result.h_est,
                force_ertm_from_align,
                effective_alignment,
                frame.identity,
                a_n_samples,
                frame.generation);
        }
        if (result.has_sensing) {
            result.sense_frame.CFO = result.alpha;
            if (sfo_per_frame != 0.0f) {
                result.sense_frame.SFO =
                    -sfo_per_frame * (2 * M_PI) / (cfg_.ofdm.fft_size * cfg_.ofdm.num_symbols);
            } else {
                result.sense_frame.SFO = result.beta;
            }
            result.sense_frame.delay_offset = delay_offset;
            result.sense_frame.alignment_samples = effective_alignment;
            result.sense_frame.identity = frame.identity;
            result.sense_frame.ertm_rx_alignment_shift_samples = a_n_samples;
            result.sense_frame.force_ertm_remeasure =
                force_ertm_from_align ||
                _ertm_remeasure_deferred_after_sensing_drop;
            if (cfg_.uplink.ertm_to_enable) {
                if (result.sense_frame.ertm_downlink_channel_freq.size() !=
                    result.h_est.size()) {
                    result.sense_frame.ertm_downlink_channel_freq.resize(
                        result.h_est.size());
                }
                std::copy(
                    result.h_est.begin(),
                    result.h_est.end(),
                    result.sense_frame.ertm_downlink_channel_freq.begin());
            }
            result.sense_frame.generation = frame.generation;
            const bool force_ertm_remeasure =
                result.sense_frame.force_ertm_remeasure;
            if (sensing_queue_.try_push(std::move(result.sense_frame))) {
                if (force_ertm_remeasure) {
                    _ertm_remeasure_deferred_after_sensing_drop = false;
                }
            } else {
                if (force_ertm_remeasure) {
                    _ertm_remeasure_deferred_after_sensing_drop = true;
                }
                if (running_.load(std::memory_order_relaxed)) {
                    LOG_RT_WARN_HZ(5) << "Bistatic sensing queue full; dropping newest sensing frame";
                }
                _sensing_frame_pool.release(std::move(result.sense_frame));
            }
            result.has_sensing = false;
        }
        prof_step_end = ProfileClock::now();
        result.profile.sensing_queue_total += std::chrono::duration<double, std::micro>(
            prof_step_end - prof_step_start).count();

        prof_step_start = ProfileClock::now();
        if (result.debug_frame) {
            // Strided debug publishing (stride chosen at dispatch): the senders
            // are LatestOnly with a 50 ms pace, so one snapshot every
            // kDebugPublishStride frames stays well above the display refresh
            // rate. Copy rather than move so the ring slot's pre-sized buffers
            // keep their capacity.
            channel_sender_.add_data(AlignedVector(result.h_est));
            pdf_sender_.add_data(AlignedVector(result.delay_spectrum));
            if (result.has_constellation) {
                constellation_sender_.add_data(AlignedVector(result.constellation_symbol));
                result.has_constellation = false;
            }
        }
        _publish_uplink_self_channel_debug(frame);
        prof_step_end = ProfileClock::now();
        result.profile.udp_send_total += std::chrono::duration<double, std::micro>(
            prof_step_end - prof_step_start).count();

        const int64_t demod_done_time_ns = do_latency_profile ? host_now_ns() : 0;
        if (result.has_llr_i16) {
            if (spsc_wait_push(_data_llr_buffer_i16, LlrFrameI16{
                    std::move(result.llr_i16),
                    frame.generation,
                    frame.host_enqueue_time_ns,
                    result.frame_dequeue_time_ns,
                    demod_done_time_ns,
                    frame.identity,
                }, [this]() {
                    return !_bit_processing_running.load(std::memory_order_acquire);
                })) {
                result.has_llr_i16 = false;
            }
        } else if (result.has_llr_float) {
            if (spsc_wait_push(_data_llr_buffer, LlrFrame{
                    std::move(result.llr),
                    frame.generation,
                    frame.host_enqueue_time_ns,
                    result.frame_dequeue_time_ns,
                    demod_done_time_ns,
                    frame.identity,
                }, [this]() {
                    return !_bit_processing_running.load(std::memory_order_acquire);
                })) {
                result.has_llr_float = false;
            }
        }

        _log_arq_profile_if_due("periodic", arq_now_ms());
        _merge_cpu_demod_profile(prof_acc, result.profile);
        // Guard on empty(): a failed push above (bit thread stopping) leaves
        // the vector moved-from, and releasing that would pollute the pool
        // with unsized buffers.
        if (result.has_llr_i16) {
            if (!result.llr_i16.empty()) {
                _llr_pool_i16.release(std::move(result.llr_i16));
            }
            result.has_llr_i16 = false;
        }
        if (result.has_llr_float) {
            if (!result.llr.empty()) {
                _llr_pool.release(std::move(result.llr));
            }
            result.has_llr_float = false;
        }
        _rx_frame_pool.release(std::move(result.frame));
        return true;
    }

    size_t _cpu_demod_worker_count() const {
        if (!cfg_.cpu_cores.demod_worker_cpu_cores.empty()) {
            return cfg_.cpu_cores.demod_worker_cpu_cores.size();
        }
        return 1;
    }

    void _cpu_demod_worker_proc(size_t worker_idx) {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        radio::set_thread_priority(1, true);
        bind_current_thread_from_ue_demod_worker(cfg_, worker_idx);

        CpuDemodWorkerContext& ctx = *_cpu_demod_contexts[worker_idx];
        auto& slot = *_cpu_demod_slots[worker_idx];
        // Pipeline depth cushions workers — cooperative backoff is acceptable.
        SPSCBackoff backoff;
        while (running_.load(std::memory_order_acquire)) {
            CpuDemodTask task;
            if (!slot.task_queue.try_pop(task)) {
                backoff.pause();
                continue;
            }
            backoff.reset();
            if (task.frame.frame_data.empty()) {
                continue;
            }
            // Fill the result directly in the ring slot: the slot's buffers are
            // pre-sized by the CpuDemodSlot factory, so the steady-state worker
            // makes no heap allocations. The dispatcher never over-commits a
            // slot (pending <= kPipelineDepth == ring capacity), so a free
            // producer slot is guaranteed; the wait below is shutdown-only
            // belt and braces.
            CpuDemodResult* result = slot.result_queue.producer_slot();
            while (result == nullptr) {
                if (!running_.load(std::memory_order_acquire)) {
                    _rx_frame_pool.release(std::move(task.frame));
                    return;
                }
                backoff.pause();
                result = slot.result_queue.producer_slot();
            }
            backoff.reset();
            _run_cpu_demod_task(ctx, std::move(task), *result);
            slot.result_queue.producer_commit();
        }
    }

    // Builds worker slots and contexts. Runs in the UEEngine constructor,
    // before any thread exists: FFTW planning is not thread-safe, so every
    // per-worker plan must be created in this single-threaded phase (this also
    // keeps the N x FFTW_MEASURE cost out of the streaming startup path).
    void _init_cpu_demod_workers() {
        const size_t worker_count = std::max<size_t>(1, _cpu_demod_worker_count());
        _cpu_demod_slots.reserve(worker_count);
        _cpu_demod_contexts.reserve(worker_count);
        for (size_t i = 0; i < worker_count; ++i) {
            _cpu_demod_slots.push_back(std::make_unique<CpuDemodSlot>(cfg_));
            _cpu_demod_contexts.push_back(std::make_unique<CpuDemodWorkerContext>(cfg_));
            auto& ctx = *_cpu_demod_contexts.back();
            // Precompute the fixed channel-anchor layout (sync + valid
            // full-band mid-frame pilots, sorted by symbol index) so the
            // per-frame path only refills pre-sized H buffers.
            ctx.channel_anchor_symbols.push_back(static_cast<int>(cfg_.ofdm.sync_pos));
            ctx.channel_anchor_source.push_back(-1);
            if (!_midframe_pilot_seqs.empty()) {
                for (size_t p = 0; p < _data_resource_layout.midframe_pilot_symbols.size(); ++p) {
                    const int actual_sym = _data_resource_layout.midframe_pilot_symbols[p];
                    if (actual_sym < 0 || p >= _midframe_pilot_seqs.size() ||
                        p >= ctx.midframe_symbol_freqs.size()) {
                        continue;
                    }
                    const auto insert_at = std::lower_bound(
                        ctx.channel_anchor_symbols.begin(),
                        ctx.channel_anchor_symbols.end(),
                        actual_sym);
                    const size_t offset = static_cast<size_t>(
                        std::distance(ctx.channel_anchor_symbols.begin(), insert_at));
                    ctx.channel_anchor_symbols.insert(insert_at, actual_sym);
                    ctx.channel_anchor_source.insert(
                        ctx.channel_anchor_source.begin() + static_cast<std::ptrdiff_t>(offset),
                        static_cast<int>(p));
                }
            }
            ctx.channel_anchor_h.resize(ctx.channel_anchor_symbols.size());
            for (auto& h : ctx.channel_anchor_h) {
                h.resize(cfg_.ofdm.fft_size);
            }
        }
    }

    void _start_cpu_demod_workers() {
        for (size_t i = 0; i < _cpu_demod_slots.size(); ++i) {
            _cpu_demod_slots[i]->thread = std::thread(&UEEngine::_cpu_demod_worker_proc, this, i);
        }
        if (cfg_.cpu_cores.demod_worker_cpu_cores.empty()) {
            LOG_G_WARN_M(Demod) << "[UE CPU demod] cpu_cores.demod_worker_cpu_cores is empty; "
                         << "running 1 unbound demod worker. Configure dedicated cores "
                         << "for stable real-time performance.";
        }
        LOG_G_INFO_M(Demod) << "[UE CPU demod] started " << _cpu_demod_slots.size()
                     << " OFDM/LLR worker thread(s), pipeline depth "
                     << CpuDemodSlot::kPipelineDepth;
    }

    void _stop_cpu_demod_workers() {
        for (auto& slot : _cpu_demod_slots) {
            if (slot && slot->thread.joinable()) {
                slot->thread.join();
            }
        }
        for (auto& slot : _cpu_demod_slots) {
            if (!slot) {
                continue;
            }
            CpuDemodTask task;
            while (slot->task_queue.try_pop(task)) {
                if (!task.frame.frame_data.empty()) {
                    _rx_frame_pool.release(std::move(task.frame));
                }
            }
            CpuDemodResult* result = nullptr;
            while ((result = slot->result_queue.consumer_slot()) != nullptr) {
                _release_cpu_demod_result(*result);
                slot->result_queue.consumer_pop();
            }
            slot->pending = 0;
        }
        // Slots/contexts stay allocated; the destructor frees them (FFTW plan
        // destruction shares the planner's single-threaded requirement).
    }

    void _report_cpu_demod_profile_if_needed(CpuDemodProfile& prof, bool do_eq_breakdown) {
        constexpr int PROF_REPORT_INTERVAL = 434;
        if (prof.frame_count < PROF_REPORT_INTERVAL || !LOG_MOD_ON(DemodProfiling)) {
            return;
        }
        const double total = prof.fft_total + prof.channel_est_total + prof.cfo_sfo_est_total +
            prof.equalization_total + prof.noise_est_total + prof.remodulate_total +
            prof.delay_spectrum_total + prof.timing_sync_total + prof.sensing_queue_total +
            prof.udp_send_total + prof.llr_total;
        const double n = static_cast<double>(prof.frame_count);
        const double avg_eq_symbols = static_cast<double>(prof.eq_data_symbols_total) / n;
        const double avg_eq_midframe_symbols =
            static_cast<double>(prof.eq_midframe_channel_symbols_total) / n;
        const double avg_eq_symbol_inv_count =
            static_cast<double>(prof.eq_symbol_inv_count_total) / n;
        const double avg_eq_pilot_phase_attempt =
            static_cast<double>(prof.eq_pilot_phase_attempt_total) / n;
        const double avg_eq_pilot_phase_success =
            static_cast<double>(prof.eq_pilot_phase_success_total) / n;
        std::ostringstream oss;
        oss << "\n========== CPU demod worker profiling (avg per frame, us) ==========\n"
            << "FFT (all symbols):    " << prof.fft_total / n << " us\n"
            << "Channel Estimation:   " << prof.channel_est_total / n << " us\n"
            << "CFO/SFO + control:    " << prof.cfo_sfo_est_total / n << " us\n"
            << "Equalization:         " << prof.equalization_total / n << " us\n";
        if (do_eq_breakdown) {
            oss << "  Eq base H_inv:      " << prof.eq_base_inv_total / n << " us\n"
                << "  Eq channel select:  " << prof.eq_channel_select_total / n << " us"
                << " (" << avg_eq_midframe_symbols << "/" << avg_eq_symbols << " midframe-H symbols)\n"
                << "  Eq symbol H_inv:    " << prof.eq_symbol_inv_total / n << " us"
                << " (" << avg_eq_symbol_inv_count << " calls/frame)\n"
                << "  Eq pilot phase fit: " << prof.eq_pilot_phase_total / n << " us"
                << " (" << avg_eq_pilot_phase_success << "/" << avg_eq_pilot_phase_attempt << " ok/frame)\n"
                << "  Eq apply:           " << prof.eq_apply_total / n << " us\n";
        }
        oss << "Noise Estimation:     " << prof.noise_est_total / n << " us\n"
            << "Remodulation:         " << prof.remodulate_total / n << " us\n"
            << "Delay Spectrum:       " << prof.delay_spectrum_total / n << " us\n"
            << "Timing Sync:          " << prof.timing_sync_total / n << " us\n"
            << "Sensing Queue:        " << prof.sensing_queue_total / n << " us\n"
            << "Debug Publish:        " << prof.udp_send_total / n << " us\n"
            << "LLR Calculation:      " << prof.llr_total / n << " us\n"
            << "TOTAL:                " << total / n << " us\n"
            << "====================================================================\n";
        LOG_RT_INFO_M(DemodProfiling) << oss.str();
        prof.reset();
    }

    /**
     * @brief Sensing Processing Thread.
     * 
     * Forwards demodulated bistatic sensing frames to SensingChannel.
     */
    void sensing_process_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        radio::set_thread_priority(1);
        bind_current_thread_from_sensing_hint(cfg_, 0);
        // sensing_queue_ cushions DSP; not a sample-ingest deadline path.
        SPSCBackoff sensing_backoff;
        while (sensing_running_.load()) {
            SensingFrame frame;
            if (!sensing_queue_.try_pop(frame)) {
                sensing_backoff.pause();
                continue;
            }
            sensing_backoff.reset();
            if (frame.generation != _sync_generation.load(std::memory_order_acquire)) {
                LOG_RT_WARN_HZ_M(Recovery, 5)
                    << "[UE recovery] dropping stale sensing frame"
                    << ": frame_generation=" << frame.generation
                    << ", current_generation="
                    << _sync_generation.load(std::memory_order_relaxed);
                _sensing_frame_pool.release(std::move(frame));
                continue;
            }
            if (_bistatic_sensing_channel) {
                _update_ertm_measurement_for_downlink_frame(
                    frame.ertm_downlink_channel_freq,
                    frame.force_ertm_remeasure,
                    frame.alignment_samples,
                    frame.identity,
                    frame.ertm_rx_alignment_shift_samples,
                    frame.generation);
                // One snapshot ties the reference coordinate to the applied display shift.
                frame.user_delay_offset = _user_delay_offset.load(std::memory_order_relaxed);
                frame.delay_offset = _select_sensing_delay_offset(frame.delay_offset, frame.user_delay_offset);
                const uint64_t frame_symbol_count = static_cast<uint64_t>(frame.rx_symbols.size());
                // Prefer absolute air-frame identity when locked; otherwise fall
                // back to the local monotonic symbol counter.
                uint64_t frame_start_symbol = 0;
                if (frame.identity.valid) {
                    frame_start_symbol = frame.identity.frame_start_symbol;
                    _next_bistatic_frame_start_symbol.store(
                        frame_start_symbol + frame_symbol_count,
                        std::memory_order_relaxed);
                } else {
                    frame_start_symbol = _next_bistatic_frame_start_symbol.fetch_add(
                        frame_symbol_count,
                        std::memory_order_relaxed);
                }
                _bistatic_sensing_channel->process_bistatic_frame(frame, frame_start_symbol);
            }
            _sensing_frame_pool.release(std::move(frame));
        }
        SensingFrame frame;
        while (sensing_queue_.try_pop(frame)) {
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
    // Parallel half of the old process_payload_llr(): marker/mini-header
    // parsing, deinterleave, descramble and LDPC decode into a flat decoded
    // packet list. Runs on a decode worker with a per-worker LDPCCodec; the
    // interleaver and scrambler are const and shared. Templated on the LLR
    // sample type so the float and int16 (Q16) paths share one body.
    template <typename LlrVec, typename ScratchVec>
    void _decode_llr_frame(const LlrVec& llr,
                           ScratchVec& deint_scratch,
                           LlrVec& payload_scratch,
                           CpuLdpcWorkerContext& ctx,
                           CpuLdpcResult& result) {
        const size_t bits_per_block = ctx.decoder.get_N();
        const size_t bytes_per_ldpc_block = (ctx.decoder.get_K() + 7) / 8;
        if (bits_per_block != LdpcPacketFraming::kLdpcCodeBitsPerBlock ||
            bytes_per_ldpc_block != LdpcPacketFraming::kLdpcInfoBytesPerBlock) {
            LOG_G_WARN_M(Demod) << "[Demod] LDPC codec dimensions do not match unified framing.";
            return;
        }

        size_t symbol_offset = 0;
        while ((symbol_offset + LdpcPacketFraming::kControlSymbols) * 2 <= llr.size()) {
            const size_t control_llr_offset = symbol_offset * 2;
            float marker_metric = 0.0f;
            if (!LdpcPacketFraming::detect_marker_llrs(
                    llr.data() + control_llr_offset, &marker_metric)) {
                break;
            }

            // Expanded Identity SIB: marker + 2 BCH words (may exceed one
            // standard control header). Prefer identity decode when enough LLRs.
            if ((symbol_offset + LdpcPacketFraming::kIdentityControlSymbols) * 2 <=
                llr.size()) {
                LdpcMiniHeader identity_hdr{};
                if (LdpcPacketFraming::decode_identity_control_llrs(
                        llr.data() + control_llr_offset + LdpcPacketFraming::kMarkerBits,
                        identity_hdr)) {
                    CpuLdpcPacketRef ref;
                    ref.mini_header = identity_hdr;
                    ref.payload_offset = 0;
                    ref.payload_len = 0;
                    result.packets.push_back(ref);
                    symbol_offset += LdpcPacketFraming::kIdentityControlSymbols;
                    continue;
                }
            }

            LdpcMiniHeader mini_header;
            if (!LdpcPacketFraming::decode_mini_header_llrs(
                    llr.data() + control_llr_offset + LdpcPacketFraming::kMarkerBits,
                    mini_header)) {
                LOG_G_WARN_M(Demod) << "[Demod] Mini-header CRC/version check failed; dropping remainder of frame.";
                break;
            }
            // Standalone IdentityHi without Identity word0 is invalid.
            if (LdpcPacketFraming::is_air_frame_identity_hi_flags(mini_header.flags)) {
                LOG_G_WARN_M(Demod) << "[Demod] orphan IdentityHi mini-header; dropping remainder.";
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
                LOG_G_WARN_M(Demod) << "[Demod] Mini-header requested payload beyond frame: blocks="
                             << payload_blocks << ", seq=" << mini_header.seq;
                break;
            }

            if (payload_blocks == 0) {
                // Non-identity control-only packets (if any).
                symbol_offset = next_symbol_offset;
                continue;
            }

            payload_scratch.resize(required_llr);
            std::copy(llr.begin() + payload_llr_offset,
                      llr.begin() + payload_llr_offset + required_llr,
                      payload_scratch.begin());

            _bit_interleaver->deinterleave_inplace(payload_scratch, deint_scratch);
            _descrambler.soft_descramble(payload_scratch);

            try {
                ctx.decoder.decode_frame(payload_scratch, ctx.decoded_payload);
                if (ctx.decoded_payload.size() < mini_header.payload_len) {
                    LOG_G_WARN_M(Demod) << "[Demod] Decoded payload shorter than mini-header length.";
                } else {
                    CpuLdpcPacketRef ref;
                    ref.mini_header = mini_header;
                    ref.payload_offset = result.payload_bytes.size();
                    ref.payload_len = mini_header.payload_len;
                    result.payload_bytes.insert(
                        result.payload_bytes.end(),
                        ctx.decoded_payload.begin(),
                        ctx.decoded_payload.begin() +
                            static_cast<std::ptrdiff_t>(mini_header.payload_len));
                    result.packets.push_back(ref);
                }
            } catch (const std::exception& e) {
                LOG_G_WARN_M(Demod) << "[Demod] Payload LDPC decode failed: " << e.what();
            }
            symbol_offset = next_symbol_offset;
        }
    }

    // Serial half: stateful packet handling in frame order on the
    // bit-processing collector thread (eRTM classification, measurement
    // comparison, ARQ window, UDP output, latency accounting).
    /**
     * @brief Bit-thread only: publish SIB anchor for RX to install.
     * @param result_generation  Must be the LLR/result generation (not wall-clock current).
     * @param epoch_changed  True when last_seen OTA epoch differs (survives RF clear).
     *        RX needs this flag: after SYNC_SEARCH the local anchor is cleared, so
     *        apply_identity_sib_anchor() cannot detect a BS restart by itself.
     * @return true if the event was accepted into the queue under a live generation.
     */
    bool _publish_identity_sib_event(
        uint64_t result_generation,
        uint32_t ota_epoch32,
        uint32_t ota_frame_seq32,
        int64_t ue_boundary_time_ns = -1,
        int64_t ue_mid_time_ns = -1,
        bool epoch_changed = false)
    {
        const uint64_t live_gen = _sync_generation.load(std::memory_order_acquire);
        if (result_generation != live_gen) {
            return false;
        }
        AirFrameSibEvent ev;
        ev.sync_generation = result_generation;
        ev.timeline_epoch = static_cast<uint64_t>(ota_epoch32);
        ev.frame_seq = static_cast<uint64_t>(ota_frame_seq32);
        ev.ue_boundary_time_ns = ue_boundary_time_ns;
        ev.ue_mid_time_ns = ue_mid_time_ns;
        // Carry bit-thread OTA epoch-change detection across RF/sync clear.
        // Must not hard-code false: that drops SFO reset after BS-only restart.
        ev.epoch_changed = epoch_changed;
        if (!_sib_event_queue.try_push(ev)) {
            LOG_RT_WARN_HZ_M(AirFrame, 2)
                << "[AirFrame] SIB event queue full; dropping identity update"
                << " epoch32=0x" << std::hex << ota_epoch32 << std::dec
                << " seq32=" << ota_frame_seq32
                << " epoch_changed=" << epoch_changed;
            return false;
        }
        return true;
    }

    void _clear_air_frame_identity(const char* reason) {
        const AirFrameIdentity snap = _ue_air_timeline.snapshot();
        if (snap.valid || LOG_MOD_ON(AirFrame)) {
            LOG_RT_WARN_M(AirFrame) << "[AirFrame] identity cleared"
                          << ": reason=" << reason
                          << ", had_valid=" << snap.valid
                          << ", epoch=" << snap.timeline_epoch
                          << ", seq=" << snap.frame_seq
                          << ", ue_frame_start_ns=" << snap.ue_frame_start_time_ns
                          << ", ue_frame_mid_ns=" << snap.ue_frame_mid_time_ns
                          << ", last_seen_ota_epoch=0x" << std::hex
                          << _last_seen_ota_epoch.load(std::memory_order_relaxed)
                          << std::dec
                          << ", has_seen_ota_epoch="
                          << _has_seen_ota_epoch.load(std::memory_order_relaxed);
        }
        // RF/sync anchor clear only — last_seen_ota_epoch survives for BS restart
        // detection after a period of sync loss.
        _ue_air_timeline.clear();
        _air_identity_rx_snapshot.clear();
        if (_uplink_tx) {
            // Invalidate queued and producer-held UL identities immediately;
            // no subsequent RX frame is guaranteed to publish an invalid stamp
            // while the receiver is in SYNC_SEARCH.
            _uplink_tx->store_air_frame_stamp(AirFrameIdentity{});
        }
        // Stale ERTM5 reports and TO/PPS phase belong to the previous timeline.
        // Do not SPSC-pop from this thread — request consumer-side wipe.
        _request_ertm_report_clear(reason);
        AirFrameSibEvent drop;
        while (_sib_event_queue.try_pop(drop)) {
        }
        _pending_sfo_reset_generation.store(0, std::memory_order_release);
        _ue_air_frame_start_time_ns.store(-1, std::memory_order_relaxed);
        _ue_air_frame_mid_time_ns.store(-1, std::memory_order_relaxed);
    }

    /**
     * @brief Bit-thread: rebuild ARQ windows after BS epoch change.
     * @return true if still generation-live after the reset (caller may commit last_seen).
     */
    bool _reset_arq_on_timeline_change(uint64_t result_generation) {
        if (result_generation !=
            _sync_generation.load(std::memory_order_acquire)) {
            return false;
        }
        if (_arq_enabled) {
            _dl_arq_rx.configure(cfg_.network_output);
            _dl_arq_rx.set_direction(0);
        }
        if (_uplink_tx && _uplink_tx->arq_enabled()) {
            _uplink_tx->arq_tx_window().reset(cfg_.uplink_arq, /*direction=*/1);
        }
        return result_generation ==
               _sync_generation.load(std::memory_order_acquire);
    }

    /**
     * @brief Demod/process-thread owner: rebuild SFO/sync tracking after BS epoch change.
     */
    void _apply_bs_air_timeline_reset_on_process_thread() {
        // Only apply if pending generation still matches live sync generation.
        uint64_t pending_gen =
            _pending_sfo_reset_generation.load(std::memory_order_acquire);
        if (pending_gen == 0) {
            return;
        }
        const uint64_t live =
            _sync_generation.load(std::memory_order_acquire);
        if (pending_gen != live) {
            // Stale request from a previous generation — drop it.
            _pending_sfo_reset_generation.compare_exchange_strong(
                pending_gen, 0, std::memory_order_acq_rel);
            return;
        }
        if (!_pending_sfo_reset_generation.compare_exchange_strong(
                pending_gen, 0, std::memory_order_acq_rel)) {
            return; // another consumer took it
        }
        LOG_G_WARN_M(AirFrame) << "[AirFrame] applying BS timeline SFO/sync reset"
                     << " on process thread gen=" << live;
        sfo_estimator.reset();
        _sync_in_progress = false;
        _sync_in_progress_hold_frames = 0;
        _sync_in_progress_expect_id = 0;
        _delay_adjustment_count = 0;
        _last_delay_index_err = 0;
        _llr_snr_linear_filtered = 1.0;
        _llr_snr_filter_initialized = false;
        _next_bistatic_frame_start_symbol.store(0, std::memory_order_relaxed);
        _air_timeline_reset_count.fetch_add(1, std::memory_order_relaxed);
    }

    void _process_decoded_packets(const CpuLdpcResult& result,
                                  bool do_latency_profile,
                                  std::vector<uint8_t>& expected_measurement_payload) {
        const int64_t rx_enqueue_time_ns = result.rx_enqueue_time_ns;
        const int64_t process_dequeue_time_ns = result.process_dequeue_time_ns;
        const int64_t demod_done_time_ns = result.demod_done_time_ns;
        bool latency_recorded = false;

        // Publish Identity SIB; commit last_seen only after generation-safe
        // protocol actions so a resync cannot mark an epoch "seen" without reset.
        for (const CpuLdpcPacketRef& pkt : result.packets) {
            if (LdpcPacketFraming::is_air_frame_identity(pkt.mini_header)) {
                if (result.generation !=
                    _sync_generation.load(std::memory_order_acquire)) {
                    break;
                }
                const uint32_t ota_epoch = pkt.mini_header.timeline_epoch;
                const uint32_t ota_seq = pkt.mini_header.air_frame_seq;
                // Survives sync-clear: detect BS restart after SYNC_SEARCH.
                const bool had_seen =
                    _has_seen_ota_epoch.load(std::memory_order_acquire);
                const uint32_t prev_ota =
                    _last_seen_ota_epoch.load(std::memory_order_relaxed);
                const bool epoch_changed = had_seen && (prev_ota != ota_epoch);

                const bool published = _publish_identity_sib_event(
                    result.generation,
                    ota_epoch,
                    ota_seq,
                    result.identity.ue_frame_start_time_ns,
                    result.identity.ue_frame_mid_time_ns,
                    epoch_changed);
                if (!published) {
                    break; // stale or queue full — do not advance last_seen
                }
                const uint64_t pub_n =
                    _air_sib_publish_count.fetch_add(1, std::memory_order_relaxed) + 1;
                // Decode-path heartbeat (default Info): first 16 + every 512.
                if (pub_n <= 16 || (pub_n % 512) == 0 || epoch_changed) {
                    LOG_G_INFO_M(AirFrame)
                        << "[AirFrame] Identity SIB decoded (bit)"
                        << ": n=" << pub_n
                        << ", ota_epoch=0x" << std::hex << ota_epoch
                        << ", prev_seen=0x" << prev_ota << std::dec
                        << ", ota_seq=" << ota_seq
                        << ", epoch_changed=" << epoch_changed
                        << ", gen=" << result.generation
                        << ", rx_boundary_ns=" << result.identity.ue_frame_start_time_ns
                        << ", rx_mid_ns=" << result.identity.ue_frame_mid_time_ns;
                }

                if (epoch_changed) {
                    // Invalidate eRTM first (atomic TO + clear request) so sensing
                    // cannot keep using the previous epoch while ARQ/SFO catch up.
                    _request_ertm_report_clear("ota_epoch_change");
                    // Bit thread only resets ARQ. SFO is requested by RX after
                    // the next successful anchor install (sticky flag below).
                    if (!_reset_arq_on_timeline_change(result.generation)) {
                        LOG_G_WARN_M(AirFrame)
                            << "[AirFrame] epoch-change ARQ reset aborted (generation)"
                            << ": prev=0x" << std::hex << prev_ota
                            << ", new=0x" << ota_epoch << std::dec
                            << " (eRTM already cleared)";
                        break;
                    }
                    // Sticky: even if this SIB event is lost/stale before RX
                    // drains it, the next successful install still requests SFO.
                    _ota_epoch_sfo_needed.store(true, std::memory_order_release);
                    LOG_G_WARN_M(AirFrame)
                        << "[AirFrame] OTA epoch change ARQ reset (bit thread)"
                        << ": prev=0x" << std::hex << prev_ota
                        << ", new=0x" << ota_epoch << std::dec
                        << ", seq=" << ota_seq
                        << ", sfo_sticky=1"
                        << ", ertm_clear_requested=1";
                }

                // Commit last_seen only after publish (+ optional reset) succeeded
                // under a live generation.
                if (result.generation ==
                    _sync_generation.load(std::memory_order_acquire)) {
                    _last_seen_ota_epoch.store(ota_epoch, std::memory_order_release);
                    _has_seen_ota_epoch.store(true, std::memory_order_release);
                }
                break;
            }
        }

        for (const CpuLdpcPacketRef& pkt : result.packets) {
            const LdpcMiniHeader& mini_header = pkt.mini_header;
            if (LdpcPacketFraming::is_air_frame_identity(mini_header)) {
                continue; // already consumed; never forward to app/ARQ
            }
            const uint8_t* payload = result.payload_bytes.data() + pkt.payload_offset;
            std::vector<uint8_t> udp_data(payload, payload + pkt.payload_len);

            {
                if (_handle_ertm_payload(udp_data, mini_header.flags)) {
                    continue;
                }

                bool handled_measurement_payload = false;
                if (_measurement_enabled) {
                    MeasurementPayloadMetadata meta;
                    if (parse_measurement_payload(udp_data.data(), udp_data.size(), meta)) {
                        handled_measurement_payload = true;
                        const uint32_t active_epoch =
                            _measurement_active_epoch_id.load(std::memory_order_relaxed);
                        if (meta.epoch_id != 0 &&
                            active_epoch != 0 &&
                            meta.epoch_id == active_epoch) {
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
                                LOG_G_WARN_M(Demod) << "[Demod] Failed to rebuild expected measurement payload; dropping packet"
                                             << " for epoch " << meta.epoch_id
                                             << " seq " << meta.seq_in_epoch;
                            }
                        } else {
                            LOG_RT_WARN_HZ_M(Demod, 5)
                                << "[Demod] measurement epoch is inactive or stale; dropping packet"
                                << ": packet_epoch=" << meta.epoch_id
                                << ", active_epoch=" << active_epoch
                                << ", seq=" << meta.seq_in_epoch;
                        }
                    }
                }

                if (!handled_measurement_payload) {
                    // ARQ: classify feedback by mini-header flag, not payload sniffing
                    bool arq_consumed = false;
                    if (_arq_enabled) {
                        if (LdpcPacketFraming::is_arq_feedback(mini_header)) {
                            // This is an ARQ ACK from the BS for our UL packets
                            _arq_dl_feedback_seen.fetch_add(1, std::memory_order_relaxed);
                            ArqFeedback ack;
                            if (ArqFeedback::try_unpack(udp_data.data(), udp_data.size(), ack)) {
                                _arq_dl_feedback_valid.fetch_add(1, std::memory_order_relaxed);
                                _apply_uplink_arq_feedback(
                                    ack, "decoded downlink payload");
                            } else {
                                const uint64_t dropped =
                                    _arq_dl_feedback_invalid.fetch_add(
                                        1, std::memory_order_relaxed) + 1;
                                if (dropped <= 20 || (dropped % 100) == 0) {
                                    LOG_G_WARN_M(ArqDl) << "[UE ARQ DL] invalid ACK feedback; dropping packet"
                                                 << ": payload_bytes=" << udp_data.size()
                                                 << ", dropped=" << dropped;
                                }
                            }
                            _log_arq_profile_if_due("dl_feedback", arq_now_ms());
                            arq_consumed = true; // never forward feedback to user UDP
                        } else {
                            // Data packet; process through DL RX ARQ window.
                            _arq_dl_data_seen.fetch_add(1, std::memory_order_relaxed);
                            const uint64_t skip_before = _dl_arq_rx.skip_count();
                            const uint64_t outside_before =
                                _dl_arq_rx.outside_window_drop_count();
                            const uint64_t reorder_before = _dl_arq_rx.reorder_drop_count();
                            const uint64_t skip_buffer_before =
                                _dl_arq_rx.skip_buffer_drop_count();
                            const bool accepted = _dl_arq_rx.process_received(
                                mini_header.seq, udp_data.data(), udp_data.size());
                            if (_dl_arq_rx.skip_count() != skip_before) {
                                LOG_G_WARN_M(ArqDl) << "[UE ARQ DL] advanced receive window past missing packets"
                                             << ": seq=" << mini_header.seq
                                             << ", skip_count=" << _dl_arq_rx.skip_count();
                            }
                            if (_dl_arq_rx.outside_window_drop_count() != outside_before) {
                                LOG_G_WARN_M(ArqDl) << "[UE ARQ DL] packet remained outside receive window; dropping"
                                             << ": seq=" << mini_header.seq
                                             << ", dropped="
                                             << _dl_arq_rx.outside_window_drop_count();
                            }
                            if (_dl_arq_rx.reorder_drop_count() != reorder_before ||
                                _dl_arq_rx.skip_buffer_drop_count() != skip_buffer_before) {
                                LOG_G_WARN_M(ArqDl) << "[UE ARQ DL] ordered-delivery buffer discarded packets"
                                             << ": capacity_drops="
                                             << _dl_arq_rx.reorder_drop_count()
                                             << ", window_skip_drops="
                                             << _dl_arq_rx.skip_buffer_drop_count();
                            }
                            if (!accepted) {
                                _arq_dl_data_duplicates.fetch_add(1, std::memory_order_relaxed);
                                arq_consumed = true; // duplicate, suppress
                            } else if (!cfg_.network_output.arq_ordered_delivery) {
                                _arq_dl_data_accepted.fetch_add(1, std::memory_order_relaxed);
                                // Unordered: forward immediately (fall through to send)
                            } else {
                                _arq_dl_data_accepted.fetch_add(1, std::memory_order_relaxed);
                                // Ordered: buffer, deliver later
                                arq_consumed = true;
                            }

                            // Generate ACK feedback and inject into UL TX
                            const int64_t now = arq_now_ms();
                            if (_dl_arq_rx.should_send_ack(now)) {
                                ArqFeedback fb = _dl_arq_rx.generate_ack();
                                _arq_dl_ack_generated.fetch_add(1, std::memory_order_relaxed);
                                if (!_uplink_tx) {
                                    const uint64_t dropped =
                                        _arq_dl_ack_no_uplink_tx.fetch_add(
                                            1, std::memory_order_relaxed) + 1;
                                    if (dropped <= 20 || (dropped % 100) == 0) {
                                        LOG_G_WARN_M(Arq) << "[UE ARQ] no uplink TX; dropping ACK feedback"
                                                     << ": dropped=" << dropped;
                                    }
                                    _log_arq_profile_if_due("ack_no_uplink_tx", now);
                                } else if (_uplink_tx->inject_arq_feedback(fb)) {
                                    _arq_dl_ack_injected.fetch_add(1, std::memory_order_relaxed);
                                } else {
                                    const uint64_t dropped =
                                        _arq_dl_ack_inject_failed.fetch_add(
                                            1, std::memory_order_relaxed) + 1;
                                    if (dropped <= 20 || (dropped % 100) == 0) {
                                        LOG_G_WARN_M(Arq) << "[UE ARQ] failed to enqueue ACK feedback; dropping it"
                                                     << ": dropped=" << dropped;
                                    }
                                    _log_arq_profile_if_due("ack_inject_failed", now);
                                }
                                _dl_arq_rx.mark_ack_sent(now);
                            }
                            _log_arq_profile_if_due(accepted ? "dl_data" : "dl_duplicate", now);
                        }
                    }

                    if (!arq_consumed) {
                        _send_udp_output_packet(
                            udp_data.data(), udp_data.size(), "decoded downlink payload");
                    }

                    // For ordered delivery, flush any contiguous deliverable packets
                    if (_arq_enabled && cfg_.network_output.arq_ordered_delivery) {
                        std::vector<std::vector<uint8_t>> deliverable;
                        _dl_arq_rx.get_deliverable(deliverable);
                        for (auto& pkt : deliverable) {
                            _send_udp_output_packet(
                                pkt.data(), pkt.size(), "ordered ARQ downlink payload");
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

            }
        }
    }

    size_t _cpu_ldpc_worker_count() const {
        if (!cfg_.cpu_cores.ldpc_worker_cpu_cores.empty()) {
            return cfg_.cpu_cores.ldpc_worker_cpu_cores.size();
        }
        return 1;
    }

    // Runs in the UEEngine constructor: LDPCCodec construction is heavy
    // (aff3ct factory + .alist loads) and must never happen on the RT path.
    void _init_cpu_ldpc_workers() {
        const size_t worker_count = std::max<size_t>(1, _cpu_ldpc_worker_count());
        _cpu_ldpc_slots.reserve(worker_count);
        _cpu_ldpc_contexts.reserve(worker_count);
        for (size_t i = 0; i < worker_count; ++i) {
            _cpu_ldpc_slots.push_back(std::make_unique<CpuLdpcSlot>());
            _cpu_ldpc_contexts.push_back(
                std::make_unique<CpuLdpcWorkerContext>(ldpc_cfg_from_config(cfg_)));
        }
    }

    // Sizes the hot-path object pools for the worst-case in-flight count of
    // the configured worker topology so ObjectPool::acquire() never hits its
    // allocating fallback inside an RT worker. release() is unbounded, so a
    // backlogged run would grow to this footprint anyway; prefilling only
    // moves those allocations to startup. Runs at the end of the constructor
    // (after worker init and init_sensing, whose outcomes set the targets).
    void _prefill_pools_for_topology() {
        const size_t demod_inflight =
            _cpu_demod_slots.size() * CpuDemodSlot::kPipelineDepth;
        const size_t ldpc_inflight =
            _cpu_ldpc_slots.size() * CpuLdpcSlot::kPipelineDepth;
        constexpr size_t kPoolMargin = 2;

        const auto top_up = [](auto& pool, size_t target, const char* name) {
            const size_t available = pool.available();
            if (target > available) {
                pool.prefill(target - available);
                LOG_G_INFO_M(Config) << "[UE pools] prefilled " << name << " to "
                             << target << " buffers";
            }
        };

        // frame_queue_ + one task and one result per pipeline stage per worker
        // + the frame in the dispatcher's hand.
        top_up(_rx_frame_pool,
               cfg_.ofdm.frame_queue_size + demod_inflight + 1 + kPoolMargin,
               "rx_frame_pool");

        // LLR buffers: hand-off queue + demod results in flight + LDPC tasks
        // in flight + the one being dispatched. Only the active sample-type
        // pool sees traffic.
        const size_t llr_target =
            _data_llr_buffer.capacity() + demod_inflight + ldpc_inflight + 1 + kPoolMargin;
        if (_ldpc_fixed_point) {
            top_up(_llr_pool_i16, llr_target, "llr_pool_i16");
        } else {
            top_up(_llr_pool, llr_target, "llr_pool");
        }

        // Sensing frames flow only when bistatic sensing is enabled.
        if (_bistatic_sensing_channel) {
            top_up(_sensing_frame_pool,
                   sensing_queue_.capacity() + demod_inflight + 1 + kPoolMargin,
                   "sensing_frame_pool");
        }
    }

    void _release_cpu_ldpc_task(CpuLdpcTask& task) {
        if (task.is_i16) {
            if (!task.llr_i16.empty()) {
                _llr_pool_i16.release(std::move(task.llr_i16));
            }
        } else if (!task.llr.empty()) {
            _llr_pool.release(std::move(task.llr));
        }
    }

    void _cpu_ldpc_worker_proc(size_t worker_idx) {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        radio::set_thread_priority();
        bind_current_thread_from_ue_ldpc_worker(cfg_, worker_idx);

        CpuLdpcWorkerContext& ctx = *_cpu_ldpc_contexts[worker_idx];
        auto& slot = *_cpu_ldpc_slots[worker_idx];
        SPSCBackoff backoff;
        while (_bit_processing_running.load(std::memory_order_acquire)) {
            CpuLdpcTask task;
            if (!slot.task_queue.try_pop(task)) {
                backoff.pause();
                continue;
            }
            backoff.reset();
            // Decode in place into the result ring slot (buffers keep their
            // capacity across frames). The dispatcher never over-commits a
            // slot, so a free producer slot is guaranteed; the wait below is
            // shutdown-only belt and braces.
            CpuLdpcResult* result = slot.result_queue.producer_slot();
            while (result == nullptr) {
                if (!_bit_processing_running.load(std::memory_order_acquire)) {
                    _release_cpu_ldpc_task(task);
                    return;
                }
                backoff.pause();
                result = slot.result_queue.producer_slot();
            }
            backoff.reset();
            result->generation = task.generation;
            result->dropped = false;
            result->rx_enqueue_time_ns = task.rx_enqueue_time_ns;
            result->process_dequeue_time_ns = task.process_dequeue_time_ns;
            result->demod_done_time_ns = task.demod_done_time_ns;
            result->identity = task.identity;
            result->packets.clear();
            result->payload_bytes.clear();
            if (task.generation != _sync_generation.load(std::memory_order_acquire)) {
                LOG_RT_WARN_HZ_M(Recovery, 5)
                    << "[UE recovery] dropping stale LDPC task before decode"
                    << ": frame_generation=" << task.generation
                    << ", current_generation="
                    << _sync_generation.load(std::memory_order_relaxed);
                result->dropped = true; // stale after resync; token keeps order
            } else if (task.is_i16) {
                _decode_llr_frame(task.llr_i16, ctx.deint_scratch_i16,
                                  ctx.payload_llr_i16, ctx, *result);
            } else {
                _decode_llr_frame(task.llr, ctx.deint_scratch,
                                  ctx.payload_llr, ctx, *result);
            }
            _release_cpu_ldpc_task(task);
            slot.result_queue.producer_commit();
        }
    }

    void _start_cpu_ldpc_workers() {
        for (size_t i = 0; i < _cpu_ldpc_slots.size(); ++i) {
            _cpu_ldpc_slots[i]->thread = std::thread(&UEEngine::_cpu_ldpc_worker_proc, this, i);
        }
        if (cfg_.cpu_cores.ldpc_worker_cpu_cores.empty()) {
            LOG_G_INFO_M(DemodLdpc) << "[UE CPU LDPC] cpu_cores.ldpc_worker_cpu_cores is empty; "
                         << "running 1 unbound LDPC decode worker.";
        }
        LOG_G_INFO_M(DemodLdpc) << "[UE CPU LDPC] started " << _cpu_ldpc_slots.size()
                     << " LDPC decode worker thread(s), pipeline depth "
                     << CpuLdpcSlot::kPipelineDepth;
    }

    void _stop_cpu_ldpc_workers() {
        for (auto& slot : _cpu_ldpc_slots) {
            if (slot && slot->thread.joinable()) {
                slot->thread.join();
            }
        }
        for (auto& slot : _cpu_ldpc_slots) {
            if (!slot) {
                continue;
            }
            CpuLdpcTask task;
            while (slot->task_queue.try_pop(task)) {
                _release_cpu_ldpc_task(task);
            }
            while (slot->result_queue.consumer_slot() != nullptr) {
                slot->result_queue.consumer_pop();
            }
            slot->pending = 0;
        }
    }

    void bit_processing_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        radio::set_thread_priority();
        bind_current_thread_from_ue_downlink_role(cfg_, 2);
        SPSCBackoff llr_backoff;
        const bool do_latency_profile =
            LOG_MOD_ON(DemodProfiling);
        const bool log_snr = LOG_MOD_ON(DemodSnr);
        std::vector<uint8_t> expected_measurement_payload;
        size_t snr_print_counter = 0;

        auto maybe_log_snr = [&]() {
            if (log_snr && ((snr_print_counter++ & 0x3F) == 0)) {
                const double llr_snr_db = 10.0 * std::log10(std::max(1.0 / _noise_var, 1e-6));
                LOG_G_INFO_M(DemodLlr) << "[LLR] SNR(dB): " << _snr_db
                             << " llr_snr(dB): " << llr_snr_db
                             << " noise_var: " << _noise_var
                             << " llr_scale: " << _llr_scale;
            }
        };

        size_t launch_slot_idx = 0;
        size_t collect_slot_idx = 0;
        _start_cpu_ldpc_workers();

        auto collect_ready_ldpc = [&](bool blocking) {
            bool collected = false;
            while (!_cpu_ldpc_slots.empty()) {
                auto& slot = *_cpu_ldpc_slots[collect_slot_idx];
                if (slot.pending == 0) {
                    break;
                }
                CpuLdpcResult* result = slot.result_queue.consumer_slot();
                if (result == nullptr) {
                    if (!blocking) {
                        break;
                    }
                    SPSCBackoff result_backoff;
                    while (_bit_processing_running.load(std::memory_order_acquire) &&
                           (result = slot.result_queue.consumer_slot()) == nullptr) {
                        result_backoff.pause();
                    }
                    if (result == nullptr) {
                        break; // shutting down
                    }
                }
                if (!result->dropped &&
                    result->generation == _sync_generation.load(std::memory_order_acquire)) {
                    maybe_log_snr();
                    _process_decoded_packets(*result, do_latency_profile,
                                             expected_measurement_payload);
                } else if (!result->dropped) {
                    LOG_RT_WARN_HZ_M(Recovery, 5)
                        << "[UE recovery] dropping decoded LDPC result after generation change"
                        << ": frame_generation=" << result->generation
                        << ", current_generation="
                        << _sync_generation.load(std::memory_order_relaxed);
                }
                slot.result_queue.consumer_pop();
                --slot.pending;
                collect_slot_idx = (collect_slot_idx + 1) % _cpu_ldpc_slots.size();
                collected = true;
                blocking = false;
            }
            return collected;
        };

        auto dispatch_ldpc_task = [&](CpuLdpcTask&& task) {
            auto& launch_slot = *_cpu_ldpc_slots[launch_slot_idx];
            if (spsc_wait_push(launch_slot.task_queue, std::move(task), [this]() {
                    return !_bit_processing_running.load(std::memory_order_acquire);
                })) {
                ++launch_slot.pending;
                launch_slot_idx = (launch_slot_idx + 1) % _cpu_ldpc_slots.size();
            } else {
                _release_cpu_ldpc_task(task);
            }
        };

        while (_bit_processing_running.load()) {
            collect_ready_ldpc(false);
            if (_cpu_ldpc_slots.empty()) {
                continue;
            }
            auto& launch_slot = *_cpu_ldpc_slots[launch_slot_idx];
            if (launch_slot.pending >= CpuLdpcSlot::kPipelineDepth) {
                collect_ready_ldpc(true);
                if (!_bit_processing_running.load(std::memory_order_acquire)) {
                    break;
                }
                if (launch_slot.pending >= CpuLdpcSlot::kPipelineDepth) {
                    continue;
                }
            }
            if (_ldpc_fixed_point) {
                LlrFrameI16 frame_llr;
                if (!_data_llr_buffer_i16.try_pop(frame_llr)) {
                    if (!collect_ready_ldpc(false)) {
                        llr_backoff.pause();
                    }
                    continue;
                }
                llr_backoff.reset();
                if (frame_llr.llr.empty()) continue;
                if (frame_llr.generation != _sync_generation.load(std::memory_order_acquire)) {
                    LOG_RT_WARN_HZ_M(Recovery, 5)
                        << "[UE recovery] dropping stale fixed-point LLR frame"
                        << ": frame_generation=" << frame_llr.generation
                        << ", current_generation="
                        << _sync_generation.load(std::memory_order_relaxed);
                    _llr_pool_i16.release(std::move(frame_llr.llr));
                    continue;
                }
                CpuLdpcTask task;
                task.llr_i16 = std::move(frame_llr.llr);
                task.is_i16 = true;
                task.generation = frame_llr.generation;
                task.rx_enqueue_time_ns = frame_llr.rx_enqueue_time_ns;
                task.process_dequeue_time_ns = frame_llr.process_dequeue_time_ns;
                task.demod_done_time_ns = frame_llr.demod_done_time_ns;
                task.identity = frame_llr.identity;
                dispatch_ldpc_task(std::move(task));
            } else {
                LlrFrame frame_llr;
                if (!_data_llr_buffer.try_pop(frame_llr)) {
                    if (!collect_ready_ldpc(false)) {
                        llr_backoff.pause();
                    }
                    continue;
                }
                llr_backoff.reset();
                if (frame_llr.llr.empty()) continue;
                if (frame_llr.generation != _sync_generation.load(std::memory_order_acquire)) {
                    LOG_RT_WARN_HZ_M(Recovery, 5)
                        << "[UE recovery] dropping stale floating-point LLR frame"
                        << ": frame_generation=" << frame_llr.generation
                        << ", current_generation="
                        << _sync_generation.load(std::memory_order_relaxed);
                    _llr_pool.release(std::move(frame_llr.llr));
                    continue;
                }
                CpuLdpcTask task;
                task.llr = std::move(frame_llr.llr);
                task.is_i16 = false;
                task.generation = frame_llr.generation;
                task.rx_enqueue_time_ns = frame_llr.rx_enqueue_time_ns;
                task.process_dequeue_time_ns = frame_llr.process_dequeue_time_ns;
                task.demod_done_time_ns = frame_llr.demod_done_time_ns;
                task.identity = frame_llr.identity;
                dispatch_ldpc_task(std::move(task));
            }
        }
        while (!_cpu_ldpc_slots.empty() && _cpu_ldpc_slots[collect_slot_idx]->pending > 0) {
            collect_ready_ldpc(true);
            if (!_bit_processing_running.load(std::memory_order_acquire)) {
                break;
            }
        }
        _stop_cpu_ldpc_workers();
        LlrFrameI16 pending_i16;
        while (_data_llr_buffer_i16.try_pop(pending_i16)) {
            if (!pending_i16.llr.empty()) {
                _llr_pool_i16.release(std::move(pending_i16.llr));
            }
        }
        LlrFrame pending_float;
        while (_data_llr_buffer.try_pop(pending_float)) {
            if (!pending_float.llr.empty()) {
                _llr_pool.release(std::move(pending_float.llr));
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
        LOG_G_ERROR_M(Config) << "CLI parameters are no longer supported. Please configure UE via "
                      << default_config_file << ".";
        return 1;
    }

    if (!path_exists(default_config_file)) {
        LOG_G_ERROR_M(Config) << "Config file '" << default_config_file
                      << "' not found. Copy a sample file from the repository config directory, "
                      << "such as 'UE_X310.yaml' or 'UE_B210.yaml', to '" << default_config_file
                      << "' and edit it before starting UE.";
        return 1;
    }

    if (!load_ue_config_from_yaml(cfg, default_config_file)) {
        return 1;
    }
    apply_logging_config(cfg.logging);

    LOG_G_INFO_M(Config) << "Loaded config from: " << default_config_file;
    finalize_ue_network_defaults(cfg);
    log_ue_sync_mode(cfg);
    log_ue_agc_mode(cfg);

    // Load FFTW wisdom
    FFTWManager::import_wisdom();

    UEEngine receiver(cfg);
    receiver.start();

    // Keep construction on the process-wide affinity and SCHED_OTHER policy so
    // UHD/ZMQ helpers do not inherit the main thread's dedicated CPU.
    bind_current_thread_to_main_core(cfg);
    
    while (!stop_signal.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    receiver.stop();
    
    // Save FFTW wisdom
    FFTWManager::export_wisdom();
    
    return 0;
}
