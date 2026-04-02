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
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <functional>
#include <memory>
#include <utility>
#include <filesystem>
#include <Common.hpp>
#include "LDPCCodec.hpp"
#include "OFDMCore.hpp"
#include "SensingChannel.hpp"

namespace {
inline int64_t host_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
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
class OFDMRxEngine {
public:
    explicit OFDMRxEngine(const Config& cfg) 
        : cfg_(cfg),
          _measurement_enabled(measurement_mode_enabled(cfg)),
          _data_resource_layout(build_data_resource_grid_layout(cfg)),
          zc_freq_(generate_zc_freq(cfg.fft_size, cfg.zc_root)),
          _sync_scratch_buffer(cfg.sync_samples()),
          frame_queue_(cfg.frame_queue_size),
          sync_queue_(cfg.sync_queue_size, [&cfg]() {
              SyncBatch batch;
              batch.data.resize(cfg.sync_samples());
              batch.usrp_time_ns = -1;
              return batch;
          }),
          _channel_estimator(cfg.fft_size),
          _delay_processor(cfg.fft_size),
          _sync_processor(cfg.sync_samples(), cfg.fft_size, cfg.cp_length, zc_freq_),
          _control_handler(cfg.control_port),
          channel_sender_(2, [this](const auto& data) { 
              channel_udp_->send_container(data); 
          }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly),
          pdf_sender_(2, [this](const auto& data) { 
              pdf_udp_->send_container(data); 
          }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly),
          constellation_sender_(10, [this](const auto& data) { 
              constellation_udp_->send_container(data); 
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
          _sensing_frame_pool(32, [&cfg]() {
              SensingFrame frame;
              frame.rx_symbols.resize(cfg.num_symbols);
              frame.tx_symbols.resize(cfg.num_symbols);
              for (size_t i = 0; i < cfg.num_symbols; ++i) {
                  frame.rx_symbols[i].resize(cfg.fft_size);
                  frame.tx_symbols[i].resize(cfg.fft_size);
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
        _measurement_active_epoch_id.store(0, std::memory_order_relaxed);
        if (_measurement_enabled && !cfg_.measurement_output_dir.empty()) {
            _measurement_summary_path = cfg_.measurement_output_dir + "/demodulator_measurement_summary.csv";
        }
        _build_compact_payload_indices();
        LOG_G_INFO() << "Payload resource grid: " << _data_resource_layout.payload_re_count
                     << " payload RE out of " << _data_resource_layout.non_pilot_re_count
                     << " non-sync/non-pilot RE per frame"
                     << (cfg_.data_resource_blocks_configured ? " (configured blocks)." : " (legacy full-grid mode).");

        init_usrp();
        init_filter();
        prepare_fftw();
        init_udp();
        if (cfg_.hardware_sync) {
            _hw_sync = std::make_unique<HardwareSyncController>(cfg_.hardware_sync_tty);
            _hw_sync->configure_ocxo_pi(
                cfg_.ocxo_pi_kp_fast,
                cfg_.ocxo_pi_ki_fast,
                cfg_.ocxo_pi_kp_slow,
                cfg_.ocxo_pi_ki_slow,
                cfg_.ocxo_pi_switch_abs_error_ppm,
                cfg_.ocxo_pi_switch_hold_s,
                cfg_.ocxo_pi_max_step_fast_ppm,
                cfg_.ocxo_pi_max_step_slow_ppm
            );
        }
        // Initialize sensing processing
        init_sensing();
        // Initialize data processing
        init_data_processing();
        _register_commands();
    }

    ~OFDMRxEngine() {
        stop();
        
        
        fftwf_destroy_plan(fft_plan_);
        // Note: _channel_estimator, _delay_processor
        // manage their own FFT plans and clean them up in their destructors
    }

    void start() {
        _control_handler.start();
        running_.store(true);
        rx_thread_ = std::thread(&OFDMRxEngine::rx_proc, this);
        process_thread_ = std::thread(&OFDMRxEngine::process_proc, this);
        
        // Start all senders
        channel_sender_.start();
        pdf_sender_.start();
        constellation_sender_.start();
        
        // Start data processing thread
        _bit_processing_running.store(true);
        _bit_processing_thread = std::thread(&OFDMRxEngine::bit_processing_proc, this);
    }

    void stop() {
        running_.store(false);
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
        _control_handler.stop();
    }

private:
private:
    static AdaptiveCFOAKF::Params make_akf_params(const Config& cfg) {
        AdaptiveCFOAKF::Params p;
        p.enable = cfg.akf_enable;
        p.bootstrap_frames = cfg.akf_bootstrap_frames;
        p.innovation_window = cfg.akf_innovation_window;
        p.max_lag = cfg.akf_max_lag;
        p.adapt_interval = cfg.akf_adapt_interval;
        p.gate_sigma = cfg.akf_gate_sigma;
        p.tikhonov_lambda = cfg.akf_tikhonov_lambda;
        p.update_smooth = cfg.akf_update_smooth;
        p.q_wf_min = cfg.akf_q_wf_min;
        p.q_wf_max = cfg.akf_q_wf_max;
        p.q_rw_min = cfg.akf_q_rw_min;
        p.q_rw_max = cfg.akf_q_rw_max;
        p.r_min = cfg.akf_r_min;
        p.r_max = cfg.akf_r_max;
        return p;
    }

    enum class RxState { SYNC_SEARCH, ALIGNMENT, NORMAL };

    Config cfg_;
    const bool _measurement_enabled;
    const DataResourceGridLayout _data_resource_layout;
    std::vector<int> _payload_subcarrier_indices_flat;
    uhd::usrp::multi_usrp::sptr usrp_;
    uhd::rx_streamer::sptr rx_stream_;
    
    AlignedVector zc_freq_;
    AlignedVector tx_sync_symbol_; 
    AlignedVector _sync_scratch_buffer;
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
    
    // Core computation instances (manage their own FFT plans)
    ChannelEstimator _channel_estimator;
    DelayProcessor _delay_processor;
    SyncProcessor _sync_processor;
    
    std::unique_ptr<FIRFilter> freq_offset_filter_;
    uhd::tune_result_t current_rx_tune_;
    bool tune_initialized_ = false;
    std::atomic<bool> running_{false};
    std::thread rx_thread_, process_thread_;

    using AlignedAlloc = AlignedAllocator<std::complex<float>, 64>;

    // Various UDP senders
    std::unique_ptr<UdpSender> channel_udp_;
    std::unique_ptr<UdpSender> pdf_udp_;
    std::unique_ptr<UdpSender> constellation_udp_;
    std::unique_ptr<VofaPlusDebugSender> vofa_debug_sender_;
    // Control handler
    ControlCommandHandler _control_handler;
    // Data sender management
    DataSender<std::complex<float>, AlignedAlloc> channel_sender_;
    DataSender<std::complex<float>, AlignedAlloc> pdf_sender_;
    DataSender<std::complex<float>, AlignedAlloc> constellation_sender_;
    uint32_t _reset_count = 0;
    
    // Sensing related variables
    SPSCRingBuffer<SensingFrame> sensing_queue_{4};
    std::thread sensing_thread_;
    std::atomic<bool> sensing_running_{false};
    SharedSensingRuntime _shared_sensing_cfg;
    std::mutex _shared_sensing_cfg_mutex;
    std::unique_ptr<SensingChannel> _bistatic_sensing_channel;
    std::atomic<uint64_t> _next_bistatic_frame_start_symbol{0};

    bool _saved = false; // Whether data has been saved
    double _freq_offset_sum = 0.0f; // Average frequency offset
    size_t _freq_offset_count = 0; // Sensing symbol count
    double _avg_freq_offset = 0.0;
    std::vector<int> _actual_subcarrier_indices;
    SFOEstimator sfo_estimator{1000};
    bool _sync_in_progress = false; // Flag to indicate if sync is in progress in process_proc
    int _last_delay_index_err = 0; // Last adjusted index
    uint32_t _delay_adjustment_count = 0;
    std::atomic<float> _user_delay_offset = 0.0f;
    std::unique_ptr<HardwareSyncController> _hw_sync;

    // Data processing related member variables
    struct LlrFrame {
        AlignedFloatVector llr;
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
    SPSCRingBuffer<LlrFrame> _data_llr_buffer{128};  // LLR data buffer
    std::thread _bit_processing_thread;
    std::atomic<bool> _bit_processing_running{false};
    LatencyAccumulator _latency_accumulator;
    
    LDPCCodec _ldpc_decoder{make_ldpc_5041008_cfg()};
    Scrambler _descrambler{201600, 0x5A};
    std::unique_ptr<UdpSender> _udp_output_sender;

    // Noise/LLR estimation related
    double _noise_var{0.5};              // Complex noise power E[|n|^2] initial value (assume 0.25 per dimension)
    double _llr_scale{2.0};              // LLR scaling factor (updated based on noise variance)
    double _snr_linear{1.0};             // Es/N0 Linear value
    double _snr_db{0.0};                 // Es/N0 dB
    double _llr_snr_linear_filtered{1.0};
    bool _llr_snr_filter_initialized{false};
    
    // Object pools for memory reuse (eliminates per-frame memory allocations)
    ObjectPool<RxFrame> _rx_frame_pool;           // Pool for RX frame buffers
    ObjectPool<AlignedFloatVector> _llr_pool;     // Pool for LLR data
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

    void _record_measurement_frame(double evm_rms) {
        if (!_measurement_enabled) {
            return;
        }
        const uint32_t epoch_id = _measurement_active_epoch_id.load(std::memory_order_relaxed);
        if (epoch_id == 0 || !std::isfinite(evm_rms) || evm_rms < 0.0) {
            return;
        }
        const double snr_db = _snr_db;
        const double evm_db = 20.0 * std::log10(std::max(evm_rms, 1e-12));
        _measurement_frame_count.fetch_add(1, std::memory_order_relaxed);
        _measurement_snr_db_sum_milli.fetch_add(
            static_cast<int64_t>(std::llround(snr_db * 1000.0)), std::memory_order_relaxed);
        _measurement_evm_rms_sum_micro.fetch_add(
            static_cast<int64_t>(std::llround(evm_rms * 1.0e6)), std::memory_order_relaxed);
        _measurement_evm_db_sum_milli.fetch_add(
            static_cast<int64_t>(std::llround(evm_db * 1000.0)), std::memory_order_relaxed);
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

    double _compute_frame_evm_rms(const std::vector<AlignedVector>& symbols) const
    {
        if (symbols.empty() || _payload_subcarrier_indices_flat.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const size_t evm_symbol_count = std::min<size_t>(3, symbols.size());
        double err_power_acc = 0.0;
        uint64_t err_count = 0;
        for (size_t sym_idx = 0; sym_idx < evm_symbol_count; ++sym_idx) {
            const auto& symbol = symbols[sym_idx];
            const auto* __restrict__ sym_ptr = symbol.data();
            const size_t payload_begin = _data_resource_layout.payload_offsets[sym_idx];
            const size_t payload_end = _data_resource_layout.payload_offsets[sym_idx + 1];
            #pragma omp simd reduction(+:err_power_acc, err_count)
            for (size_t idx = payload_begin; idx < payload_end; ++idx) {
                const size_t k = static_cast<size_t>(_payload_subcarrier_indices_flat[idx]);
                const float ref_re = std::copysign(QPSKModulator::SQRT_2_INV, sym_ptr[k].real());
                const float ref_im = std::copysign(QPSKModulator::SQRT_2_INV, sym_ptr[k].imag());
                const float err_re = sym_ptr[k].real() - ref_re;
                const float err_im = sym_ptr[k].imag() - ref_im;
                err_power_acc += static_cast<double>(err_re * err_re + err_im * err_im);
                err_count += 1;
            }
            for (size_t i = 0; i < cfg_.pilot_positions.size(); ++i) {
                const size_t k = cfg_.pilot_positions[i];
                if (k >= symbol.size()) {
                    continue;
                }
                const float err_re = sym_ptr[k].real() - zc_freq_[k].real();
                const float err_im = sym_ptr[k].imag() - zc_freq_[k].imag();
                err_power_acc += static_cast<double>(err_re * err_re + err_im * err_im);
                err_count += 1;
            }
        }
        if (err_count == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::sqrt(err_power_acc / static_cast<double>(err_count));
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
        const int32_t tx_gain_x10 =
            _measurement_epoch_tx_gain_x10.exchange(std::numeric_limits<int32_t>::min(),
                                                    std::memory_order_acq_rel);

        const uint64_t packets_expected = cfg_.measurement_packets_per_point;
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
        };
        const std::vector<std::string> row{
            cfg_.measurement_run_id,
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
        };
        if (!append_csv_row(_measurement_summary_path, header, row)) {
            LOG_G_WARN() << "Failed to append demodulator measurement row to "
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
        // Use device arguments from configuration
        usrp_ = uhd::usrp::multi_usrp::make(cfg_.device_args);
        usrp_->set_clock_source(cfg_.clocksource);
        usrp_->set_rx_rate(cfg_.sample_rate);
        usrp_->set_rx_bandwidth(cfg_.bandwidth, cfg_.rx_channel);
        current_rx_tune_ = usrp_->set_rx_freq(uhd::tune_request_t(cfg_.center_freq), cfg_.rx_channel);
        tune_initialized_ = true;
        LOG_G_INFO() << "Actual RX RF Freq: " << format_freq_hz(current_rx_tune_.actual_rf_freq)
                     << " Hz, DSP: " << format_freq_hz(current_rx_tune_.actual_dsp_freq)
                     << " Hz";
        const auto gain_range = usrp_->get_rx_gain_range(cfg_.rx_channel);
        _rx_gain_min_db = gain_range.start();
        _rx_gain_max_db = gain_range.stop();
        const double initial_rx_gain_db = std::clamp(cfg_.rx_gain, _rx_gain_min_db, _rx_gain_max_db);
        if (initial_rx_gain_db != cfg_.rx_gain) {
            LOG_G_WARN() << "Configured rx_gain=" << cfg_.rx_gain
                         << " dB is outside hardware range ["
                         << _rx_gain_min_db << ", " << _rx_gain_max_db
                         << "] dB. Clamping to " << initial_rx_gain_db << " dB.";
        }
        usrp_->set_rx_gain(initial_rx_gain_db, cfg_.rx_channel);
        _rx_agc.initialize(initial_rx_gain_db, _rx_gain_min_db, _rx_gain_max_db);
        _sync_search_gain_sweep.initialize(initial_rx_gain_db, _rx_gain_min_db, _rx_gain_max_db);
        LOG_G_INFO() << "RX gain range: [" << _rx_gain_min_db << ", " << _rx_gain_max_db
                     << "] dB, initial gain: " << initial_rx_gain_db << " dB";

        uhd::stream_args_t args("fc32", cfg_.wire_format_rx);
        args.args["block_id"] = "radio";
        args.channels = {cfg_.rx_channel};
        rx_stream_ = usrp_->get_rx_stream(args);
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
            if (cfg_.num_symbols == 0) {
                _shared_sensing_cfg.apply_symbol_index = next_symbol;
            } else {
                const uint64_t frame_len = static_cast<uint64_t>(cfg_.num_symbols);
                const uint64_t boundary = ((next_symbol + frame_len - 1) / frame_len) * frame_len;
                _shared_sensing_cfg.apply_symbol_index = boundary;
            }
            snapshot = _shared_sensing_cfg;
        }
        _bistatic_sensing_channel->apply_shared_cfg(snapshot);
    }

    void _register_commands() {
        _control_handler.register_command("SKIP", [this](int32_t value) {
            const bool new_skip = (value != 0);
            _schedule_shared_sensing_update([new_skip](SharedSensingRuntime& cfg) {
                cfg.skip_sensing_fft = new_skip;
            });
            LOG_G_INFO() << "Received skip sensing FFT command: " << value;
        });

        _control_handler.register_command("STRD", [this](int32_t value) {
            const size_t stride = value <= 0 ? 1 : static_cast<size_t>(value);
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

        _control_handler.register_command("MTI ", [this](int32_t value) {
            const bool enable = (value != 0);
            _schedule_shared_sensing_update([enable](SharedSensingRuntime& cfg) {
                cfg.enable_mti = enable;
            });
            LOG_G_INFO() << "MTI " << (enable ? "Enabled" : "Disabled");
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
    }

    void init_sensing() {
        _actual_subcarrier_indices.resize(cfg_.fft_size);
        const size_t half_fft = static_cast<int>(cfg_.fft_size) / 2;
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < cfg_.fft_size; ++i) {
            _actual_subcarrier_indices[i] = (i >= half_fft) ?
                (static_cast<int>(i) - cfg_.fft_size) :
                static_cast<int>(i);
        }

        if (!cfg_.enable_bi_sensing) {
            LOG_G_INFO() << "Bistatic sensing disabled by config.";
            return;
        }

        _shared_sensing_cfg.sensing_symbol_stride = cfg_.sensing_symbol_stride;
        _shared_sensing_cfg.enable_mti = true;
        _shared_sensing_cfg.skip_sensing_fft = true;
        _shared_sensing_cfg.generation = 1;
        _shared_sensing_cfg.apply_symbol_index = 0;
        _next_bistatic_frame_start_symbol.store(0, std::memory_order_relaxed);

        SensingRxChannelConfig bistatic_cfg;
        bistatic_cfg.sensing_ip = cfg_.bi_sensing_ip;
        bistatic_cfg.sensing_port = cfg_.bi_sensing_port;
        bistatic_cfg.enable_system_delay_estimation = false;
        _bistatic_sensing_channel = std::make_unique<SensingChannel>(
            cfg_,
            bistatic_cfg,
            0,
            running_,
            [this](const std::string& ip, int port) {
                _control_handler.send_heartbeat(ip, port);
            },
            [this](size_t hint) {
                return core_from_hint(cfg_, hint);
            }
        );
        _bistatic_sensing_channel->apply_shared_cfg(_shared_sensing_cfg);
        _bistatic_sensing_channel->start_bistatic();

        sensing_running_.store(true);
        sensing_thread_ = std::thread(&OFDMRxEngine::sensing_process_proc, this);
    }

    void init_data_processing() {
        // Initialize UDP output sender
        _udp_output_sender = std::make_unique<UdpSender>(cfg_.udp_output_ip, static_cast<uint16_t>(cfg_.udp_output_port));
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
            new_tune_req.target_freq = cfg_.center_freq;
            new_tune_req.dsp_freq = 0.0; // Reset DSP frequency
        }
        // Apply new tune (update DSP only, fast and does not affect LO)
        current_rx_tune_ = usrp_->set_rx_freq(new_tune_req, cfg_.rx_channel);

    }

    void prepare_fftw() {
        fft_input_.resize(cfg_.fft_size);
        fft_output_.resize(cfg_.fft_size);
        fft_plan_ = fftwf_plan_dft_1d(cfg_.fft_size,
            reinterpret_cast<fftwf_complex*>(fft_input_.data()),
            reinterpret_cast<fftwf_complex*>(fft_output_.data()),
            FFTW_FORWARD, FFTW_MEASURE);

        // Note: ChannelEstimator and DelayProcessor now manage their own FFT plans internally
    }

    void init_udp() {
        // Use IP and port from configuration
        channel_udp_ = std::make_unique<UdpSender>(cfg_.channel_ip, cfg_.channel_port);
        pdf_udp_ = std::make_unique<UdpSender>(cfg_.pdf_ip, cfg_.pdf_port);
        constellation_udp_ = std::make_unique<UdpSender>(cfg_.constellation_ip, cfg_.constellation_port);
        vofa_debug_sender_ = std::make_unique<VofaPlusDebugSender>(cfg_.vofa_debug_ip, cfg_.vofa_debug_port, 64);
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
    }

    void _enter_sync_search_state() {
        _sync_generation.fetch_add(1, std::memory_order_acq_rel);
        _sync_in_progress = false;
        _delay_adjustment_count = 0;
        _last_delay_index_err = 0;
        _llr_snr_linear_filtered = 1.0;
        _llr_snr_filter_initialized = false;
        const bool log_agc = cfg_.should_profile("agc");
        double search_gain_db = 0.0;
        const bool reset_search_gain = _sync_search_gain_sweep.reset_to_default(
            [this](double gain_db) {
                usrp_->set_rx_gain(gain_db, cfg_.rx_channel);
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
    void rx_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        uhd::set_thread_priority_safe(1.0, true);
        // Assign to available core (index 0)
        size_t core_idx = 0 % cfg_.available_cores.size();
        std::vector<size_t> cpu_list = {cfg_.available_cores[core_idx]};
        uhd::set_thread_affinity(cpu_list);
        uhd::rx_metadata_t md;
        rx_stream_->issue_stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);

        while (running_.load()) {
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

        rx_stream_->issue_stream_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
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
            if (received == 0 && got > 0) {
                first_time_ns = metadata_time_to_ns(md);
            }
            received += got;
        }

        if (sync_batch == nullptr) {
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
        const size_t total_read = cfg_.samples_per_frame() + discard_samples_;
        AlignedVector temp_buf(total_read);
        size_t received = 0;
        int64_t frame_time_ns = -1;
        while (received < total_read && running_.load()) {
            const size_t got = rx_stream_->recv(&temp_buf[received], total_read - received, md);
            if (received == 0 && got > 0) {
                frame_time_ns = metadata_time_to_ns(md);
            }
            received += got;
        }
        // Acquire pre-allocated RX frame from pool
        RxFrame frame = _rx_frame_pool.acquire();
        frame.Alignment = discard_samples_;
        frame.usrp_time_ns = frame_time_ns;
        frame.host_enqueue_time_ns = do_latency_profile ? host_now_ns() : 0;
        frame.generation = _sync_generation.load(std::memory_order_acquire);
        if(discard_samples_ > 0)
        {
            std::copy(temp_buf.begin() + discard_samples_,
                    temp_buf.begin() + discard_samples_ + cfg_.samples_per_frame(),
                    frame.frame_data.begin());
        }
        else
        {
            std::copy(temp_buf.begin(),
                    temp_buf.begin() + discard_samples_+cfg_.samples_per_frame(),
                    frame.frame_data.begin() - discard_samples_);
        }

        if (!frame_queue_.try_push(std::move(frame))) {
            LOG_RT_WARN_HZ(5) << "RX frame queue full during alignment, dropping newest frame";
            _rx_frame_pool.release(std::move(frame));
        }
//        LOG_G_INFO() << "Alignment done, moving "<< discard_samples_<< " samples" << std::endl;
//        LOG_G_INFO() <<  discard_samples_<< std::endl;
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
            if (received == 0 && got > 0) {
                frame.usrp_time_ns = metadata_time_to_ns(md);
            }
            received += got;
        }

        if (received > 0) {
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
        const size_t symbol_len = cfg_.fft_size + cfg_.cp_length;
        const bool allow_freq_adjust = _control_time_gates.allow_freq_adjust(sync_time_ns);
        const bool allow_alignment = _control_time_gates.allow_alignment(sync_time_ns);
        bool issued_freq_adjust = false;
        bool issued_alignment = false;
        
        // Use SyncProcessor for sync detection
        int max_pos = 0;
        float max_corr = 0.0f;
        float avg_corr = 0.0f;
        _sync_processor.find_sync_position(sync_data, max_pos, max_corr, avg_corr);
        
        const float energy_threshold = 100.0f;
        if (max_corr/avg_corr > energy_threshold) {
            _sync_search_gain_sweep.note_sync_found();
            LOG_RT_INFO() << "Sync found at pos: " << max_pos
                          << " with value: " << max_corr
                          << " Threshold: " << energy_threshold;
            int symbol_offset = max_pos % symbol_len;
            // Calculate available symbol count
            const size_t available_symbols = std::min(
                static_cast<size_t>(cfg_.num_symbols*2),
                (sync_data.size() - symbol_offset) / symbol_len
            );
            
            // Use SyncProcessor for CFO estimation
            if (available_symbols > 0) {
                double phase_diff = SyncProcessor::estimate_cfo_phase(
                    sync_data, symbol_offset, available_symbols,
                    symbol_len, cfg_.cp_length, cfg_.fft_size
                );
                double cfo = SyncProcessor::phase_to_cfo(phase_diff, cfg_.sample_rate, cfg_.fft_size);
                
                LOG_RT_INFO() << "CFO estimate: " << cfo << " Hz (using " << available_symbols
                              << " symbols)";
                
                // Perform initial CFO correction
                if (cfg_.software_sync && allow_freq_adjust){
                    adjust_rx_freq(-cfo, false);
                    issued_freq_adjust = true;
                }
            } else {
                LOG_RT_WARN() << "No valid symbols for CFO estimation";
            }
            
            // Record time offset
            if (allow_alignment) {
                sync_offset_ = (max_pos - cfg_.sync_pos * symbol_len);
                if (sync_offset_ > 0) {
                    sync_offset_ = sync_offset_ % cfg_.samples_per_frame();
                }
                _clear_frame_queue();
                discard_samples_.store(sync_offset_);
                state_ = RxState::ALIGNMENT;
                issued_alignment = true;
            }
            if (issued_freq_adjust) {
                _control_time_gates.mark_freq_adjust_now(usrp_->get_time_now());
            }
            if (issued_alignment) {
                _control_time_gates.mark_alignment_now(usrp_->get_time_now());
            }
        } else {
            const size_t frame_samples = cfg_.samples_per_frame();
            const size_t search_equivalent_frames =
                (frame_samples > 0) ? std::max<size_t>(1, sync_data.size() / frame_samples) : 1;
            double search_gain_db = 0.0;
            const bool stepped_search_gain = _sync_search_gain_sweep.on_search_miss(
                search_equivalent_frames,
                [this](double gain_db) {
                    usrp_->set_rx_gain(gain_db, cfg_.rx_channel);
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

    void save_data(const AlignedVector& data, const std::string& filename) {
        if (!_saved) {
            std::ofstream data_file(filename, std::ios::binary);
            data_file.write(reinterpret_cast<const char*>(data.data()),
                data.size() * sizeof(std::complex<float>));
            data_file.close();
            _saved = true;
        }
    }

    void process_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        uhd::set_thread_priority_safe(1, true);
        // Assign to available core (index 1)
        size_t core_idx = 1 % cfg_.available_cores.size();
        std::vector<size_t> cpu_list = {cfg_.available_cores[core_idx]};
        uhd::set_thread_affinity(cpu_list);
        
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
                    double frame_duration = cfg_.samples_per_frame() / cfg_.sample_rate * 1000.0;
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
        
        auto prof_step_start = ProfileClock::now();
        auto prof_step_end = prof_step_start;
        // =================================================
        
        const bool sensing_enabled = static_cast<bool>(_bistatic_sensing_channel);
        SensingFrame sense_frame;
        if (sensing_enabled) {
            // Acquire the pooled frame only when bistatic sensing is enabled.
            sense_frame = _sensing_frame_pool.acquire();
            sense_frame.rx_symbols.resize(cfg_.num_symbols);
            sense_frame.tx_symbols.resize(cfg_.num_symbols);
        }
        
        std::vector<AlignedVector> symbols;
        symbols.reserve(cfg_.num_symbols-1);
        // Pre-create pilot boolean mask
        static std::vector<uint8_t> pilot_mask;
        if (pilot_mask.size() != cfg_.fft_size) {
            pilot_mask.assign(cfg_.fft_size, 0);
            for (auto p : cfg_.pilot_positions) if (p < pilot_mask.size()) pilot_mask[p] = 1;
        }
        AlignedVector sync_symbol_freq;
        size_t pos = 0;
        //save_data(frame, "rx_frame.bin");
        prof_step_start = ProfileClock::now();
        for (size_t i = 0; i < cfg_.num_symbols; ++i) {
            AlignedVector symbol(cfg_.fft_size);
            std::copy(frame.frame_data.begin() + pos + cfg_.cp_length,
                     frame.frame_data.begin() + pos + cfg_.cp_length + cfg_.fft_size,
                     fft_input_.begin());
            
            fftwf_execute(fft_plan_);
            
            const float scale = 1.0f / sqrtf(cfg_.fft_size);
            #pragma omp simd
            for (size_t j = 0; j < cfg_.fft_size; ++j) {
                symbol[j] = fft_output_[j] * scale;
            }
            if (sensing_enabled) {
                // Write directly to sense_frame.rx_symbols to avoid later copy.
                sense_frame.rx_symbols[i] = symbol;
            }
            
            if (i == cfg_.sync_pos) {
                sync_symbol_freq = std::move(symbol);
            }
            else {
                symbols.push_back(std::move(symbol));
            }
            pos += cfg_.fft_size + cfg_.cp_length;
        }
        prof_step_end = ProfileClock::now();
        prof_fft_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        // Calculate initial channel response H_est using ChannelEstimator
        prof_step_start = ProfileClock::now();
        AlignedVector H_est;
        float corrected_impulse_snr_linear_est = 1.0f;
        _channel_estimator.estimate_from_sync_lmmse(
            sync_symbol_freq,
            zc_freq_,
            H_est,
            cfg_.cp_length,
            &corrected_impulse_snr_linear_est);
        //_channel_estimator.estimate_from_sync_ls(sync_symbol_freq, zc_freq_, H_est);

        prof_step_end = ProfileClock::now();
        prof_channel_est_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        // Estimate frequency offset using FrequencyOffsetEstimator
        prof_step_start = ProfileClock::now();
        std::vector<int> pilot_indices;
        std::vector<float> avg_phase_diff;
        std::vector<float> weights;
        FrequencyOffsetEstimator::compute_pilot_phase_diff(
            symbols, cfg_.pilot_positions, cfg_.fft_size, cfg_.sync_pos,
            pilot_indices, avg_phase_diff, weights
        );
        
        // Phase unwrapping with SIMD optimization
        unwrap(avg_phase_diff);
        auto [beta, alpha] = weightedlinearRegression(pilot_indices, avg_phase_diff, weights);
        prof_step_end = ProfileClock::now();
        prof_cfo_sfo_est_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        // Convert alpha to CFO
        float detected_freq_offset = FrequencyOffsetEstimator::alpha_to_cfo(
            alpha, cfg_.fft_size, cfg_.cp_length, cfg_.sample_rate
        );
        const double tune_system_cfo_hz = rx_tune_system_cfo_hz(
            cfg_.center_freq,
            current_rx_tune_.actual_rf_freq,
            current_rx_tune_.actual_dsp_freq
        );
        const double clock_error_hz =
            static_cast<double>(detected_freq_offset) - tune_system_cfo_hz;
        const double raw_error_ppm =
            (std::abs(cfg_.center_freq) > 0.0)
                ? (clock_error_hz / cfg_.center_freq * 1e6)
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

        if (cfg_.hardware_sync) {
            const auto akf_result = _akf.update(raw_error_ppm);
            const double control_error_ppm = cfg_.akf_enable
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
        {
            _freq_offset_sum += detected_freq_offset;
            _freq_offset_count++;
            if (_freq_offset_count>=434) {
                _avg_freq_offset = _freq_offset_sum / _freq_offset_count;
                _freq_offset_sum = 0.0f;
                _freq_offset_count = 0;
                if(abs(_avg_freq_offset) > 2.0f)
                {
                    if (cfg_.software_sync && allow_freq_adjust){
                        LOG_RT_INFO() << "Adjusting RX frequency by: " << _avg_freq_offset << " Hz";
                        adjust_rx_freq(-_avg_freq_offset, false);
                        issued_freq_adjust = true;
                    }
                }
            }
        }
        if (issued_freq_adjust) {
            _control_time_gates.mark_freq_adjust_now(usrp_->get_time_now());
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
        // Pre-compute H_est inverse (ZF equalizer base) using ChannelEstimator
        static thread_local AlignedVector H_inv;
        ChannelEstimator::compute_zf_inverse(H_est, H_inv);
        
        for (size_t i = 0; i < symbols.size(); ++i) {
            auto& symbol = symbols[i];
            
            const int relative_symbol_index = (i < cfg_.sync_pos) ? 
                (static_cast<int>(i) - static_cast<int>(cfg_.sync_pos)) : 
                (static_cast<int>(i) + 1 - static_cast<int>(cfg_.sync_pos));
            const float phase_diff_CFO = alpha * relative_symbol_index;
            const float beta_rel = beta * relative_symbol_index;

            // Equalization using ChannelEstimator
            ChannelEstimator::equalize_symbol(symbol, H_inv, phase_diff_CFO, beta_rel, _actual_subcarrier_indices);
        }
        prof_step_end = ProfileClock::now();
        prof_equalization_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        // ================= SNR / LLR Scale Estimation (Corrected impulse-response SNR) =================
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
            for (size_t i = 0; i < cfg_.num_symbols; ++i) {
                if (i == cfg_.sync_pos) {
                    // Sync symbol uses original ZC sequence.
                    sense_frame.tx_symbols[i] = zc_freq_;
                } else {
                    // Data symbol remodulation is only needed for bistatic sensing.
                    const size_t symbol_idx = (i < cfg_.sync_pos) ? i : i - 1;
                    QPSKModulator::remodulate_symbol(
                        symbols[symbol_idx],
                        zc_freq_,
                        cfg_.pilot_positions,
                        sense_frame.tx_symbols[i]
                    );
                }
            }
        }
        prof_step_end = ProfileClock::now();
        prof_remodulate_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        // Compute delay spectrum using DelayProcessor
        prof_step_start = ProfileClock::now();
        AlignedVector delay_spectrum;
        _delay_processor.compute_delay_spectrum(H_est, delay_spectrum);
        
        // Find peak in delay spectrum (search within CP range)
        size_t max_index = 0;
        float max_mag = 0.0f;
        float average_mag = 0.0f;
        DelayProcessor::find_peak(delay_spectrum, max_index, max_mag, average_mag, cfg_.cp_length);
        
        // Adjust delay index to signed value
        int adjusted_index = DelayProcessor::adjust_delay_index(max_index, cfg_.fft_size);

        float fractional_delay = DelayProcessor::estimate_fractional_delay(delay_spectrum, max_index);
        float delay_offset_reading = adjusted_index + fractional_delay;
        prof_step_end = ProfileClock::now();
        prof_delay_spectrum_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        prof_step_start = ProfileClock::now();
        sfo_estimator.update(delay_offset_reading, frame.Alignment);
        auto _sfo_per_frame = sfo_estimator.get_sfo_per_frame();

        // auto sfo_via_estimator = _sfo_per_frame/((cfg_.fft_size+cfg_.cp_length)*cfg_.num_symbols*cfg_.sample_rate);
        // auto sfo_via_beta = -beta/((cfg_.fft_size+cfg_.cp_length)*cfg_.sample_rate/cfg_.fft_size*2*M_PI);
        // LOG_G_INFO() << "SFO via estimator: " << sfo_via_estimator << std::endl;
        // LOG_G_INFO() << "SFO via beta: " << sfo_via_beta << std::endl;
        // LOG_G_INFO() << "ratio:" << sfo_via_beta/sfo_via_estimator <<std::endl;
        if (_sfo_per_frame != 0.0f) {
            //LOG_G_INFO() << "SFO Frame: " << _sfo_per_frame << std::endl;
        }
        const size_t sync_symbol_len = cfg_.fft_size + cfg_.cp_length;
        const size_t sync_symbol_offset = cfg_.sync_pos * sync_symbol_len;
        const std::complex<float>* sync_symbol_td = nullptr;
        size_t sync_symbol_td_count = 0;
        if (sync_symbol_offset + sync_symbol_len <= frame.frame_data.size()) {
            sync_symbol_td = frame.frame_data.data() + sync_symbol_offset;
            sync_symbol_td_count = sync_symbol_len;
        }
        auto delay_offset = sfo_estimator.get_sensing_delay_offset();
        int delay_index_err = adjusted_index - cfg_.desired_peak_pos;
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
                    usrp_->set_rx_gain(gain_db, cfg_.rx_channel);
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
        if((max_mag/average_mag < 20.0f || (abs(delay_index_err) > cfg_.delay_adjust_step+5)) &&
            (cfg_.software_sync || cfg_.hardware_sync) && allow_reset) {
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
                if (cfg_.hardware_sync) {
                    _hw_sync->reset_frequency_control(); // Reset hardware frequency control
                    _hw_sync->reset_ocxo_pi_state();
                    LOG_RT_INFO() << "OCXO PI state reset to fast stage after sync reset.";
                }
                _enter_sync_search_state();
                _control_time_gates.mark_reset_now(usrp_->get_time_now());
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
        if(abs(delay_index_err) >= cfg_.delay_adjust_step &&
            abs(delay_index_err) < cfg_.cp_length  &&
            abs(detected_freq_offset) < 100.0f &&
            ( cfg_.software_sync || cfg_.hardware_sync ) &&
            !_sync_in_progress &&
            allow_alignment)
        {   
            if (_delay_adjustment_count++ >= 1) {
                _delay_adjustment_count = 0;
                discard_samples_.store(delay_index_err);
                state_ = RxState::ALIGNMENT; // Send sync command. May have delay due to FIFO
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
            _control_time_gates.mark_alignment_now(usrp_->get_time_now());
        }
        if (issued_rx_gain_adjust) {
            _control_time_gates.mark_rx_gain_adjust_now(usrp_->get_time_now());
        }

        prof_step_start = ProfileClock::now();
        if (sensing_enabled) {
            sense_frame.CFO = alpha;
            if (_sfo_per_frame != 0.0f) {
                sense_frame.SFO = -_sfo_per_frame * (2 * M_PI) / (cfg_.fft_size * cfg_.num_symbols);
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

        // Constellation data
        const size_t last_idx = symbols.size() - 1;
        auto constellation_copy = symbols[last_idx];
        constellation_sender_.add_data(std::move(constellation_copy));
        prof_step_end = ProfileClock::now();
        prof_udp_send_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        // Extract selected payload RE to generate compact LLR.
        prof_step_start = ProfileClock::now();
        float scale_llr = std::min(static_cast<float>(_llr_scale * M_SQRT1_2), 500.0f);

        if (_measurement_enabled &&
            _measurement_active_epoch_id.load(std::memory_order_relaxed) != 0) {
            const double evm_rms = _compute_frame_evm_rms(symbols);
            _record_measurement_frame(evm_rms);
        }

        if (_data_resource_layout.payload_re_count > 0) {
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
        prof_step_end = ProfileClock::now();
        prof_llr_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        // ============== Profiling report ==============
        prof_frame_count++;
        if (prof_frame_count >= PROF_REPORT_INTERVAL && cfg_.should_profile("demodulation")) {
            double total = prof_fft_total + prof_channel_est_total + prof_cfo_sfo_est_total + 
                          prof_equalization_total + prof_noise_est_total + prof_remodulate_total + 
                          prof_delay_spectrum_total + prof_timing_sync_total + prof_sensing_queue_total + 
                          prof_udp_send_total + prof_llr_total;
            std::ostringstream oss;
            oss << "\n========== process_ofdm_frame Profiling (avg per frame, us) ==========\n"
                << "FFT (all symbols):    " << prof_fft_total / prof_frame_count << " us\n"
                << "Channel Estimation:   " << prof_channel_est_total / prof_frame_count << " us\n"
                << "CFO/SFO Estimation:   " << prof_cfo_sfo_est_total / prof_frame_count << " us\n"
                << "Equalization:         " << prof_equalization_total / prof_frame_count << " us\n"
                << "Noise Estimation:     " << prof_noise_est_total / prof_frame_count << " us\n"
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
        // Assign to available core (index 2)
        size_t core_idx = 2 % cfg_.available_cores.size();
        std::vector<size_t> cpu_list = {cfg_.available_cores[core_idx]};
        uhd::set_thread_affinity(cpu_list);
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
    void bit_processing_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        uhd::set_thread_priority_safe();
        // Assign to available core (index 3)
        size_t core_idx = 3 % cfg_.available_cores.size();
        std::vector<size_t> cpu_list = {cfg_.available_cores[core_idx]};
        uhd::set_thread_affinity(cpu_list);
        SPSCBackoff llr_backoff;
        const bool do_latency_profile =
            cfg_.should_profile("demodulation") && cfg_.should_profile("latency");
        const bool log_snr = cfg_.should_profile("snr");
        std::vector<uint8_t> expected_measurement_payload;
        size_t snr_print_counter = 0;
        
        while (_bit_processing_running.load()) {
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

            if (log_snr && ((snr_print_counter++ & 0x3F) == 0)) {
                const double llr_snr_db = 10.0 * std::log10(std::max(1.0 / _noise_var, 1e-6));
                LOG_G_INFO() << "[LLR] SNR(dB): " << _snr_db
                             << " llr_snr(dB): " << llr_snr_db
                             << " noise_var: " << _noise_var
                             << " llr_scale: " << _llr_scale;
            }
            
            // Process frame data, block by block according to decoder N (bits)
            size_t bits_per_block = _ldpc_decoder.get_N();
            size_t llr_offset = 0;
            bool latency_recorded = false;
            while (llr_offset + bits_per_block <= frame_llr.llr.size()) {
                // 1. Extract LLR of current block (corresponding to one LDPC code block)
                LDPCCodec::AlignedFloatVector header_llr(bits_per_block);
                std::copy(frame_llr.llr.begin() + llr_offset,
                         frame_llr.llr.begin() + llr_offset + bits_per_block,
                         header_llr.begin());
                
                // 2. Soft descramble LDPC header
                _descrambler.soft_descramble(header_llr);
                
                // 3. LDPC soft decode
                LDPCCodec::AlignedByteVector decoded_header;
                bool header_decoded = false;
                try {
                    _ldpc_decoder.decode_frame(header_llr, decoded_header);
                    header_decoded = true;
                } catch (const std::exception& e) {
                    // LDPC decode failed, skip this frame
                    break;
                }
                
                // 4. Detect header: dynamic header_len (based on decoder K) and verify repetition of payload_len in first and second half
                if (header_decoded) {
                    // Use K/N from decoder to calculate bytes/bits
                    size_t K_bits_local = _ldpc_decoder.get_K();
                    size_t bytes_per_ldpc_block = (K_bits_local + 7) / 8;
                    size_t header_len = (K_bits_local + 7) / 8; // Number of bytes

                    // Split into two halves, ensure second half has even bytes, extra byte goes to first half
                    size_t half1 = (header_len + 1) / 2;
                    size_t half2 = header_len - half1;
                    if ((half2 % 2) != 0 && half2 > 0) { half1 += 1; half2 -= 1; }

                    // Verify first half is 1,2,3... (byte-wise, wrapped to 8 bits)
                    bool header_match = true;
                    for (size_t i = 0; i < half1; ++i) {
                        if (static_cast<uint8_t>(decoded_header[i]) != static_cast<uint8_t>((i + 1) & 0xFF)) { header_match = false; break; }
                    }
                    if (!header_match) {
                        // Mismatch, skip this frame
                        break;
                    }

                    size_t second_start = half1;
                    uint16_t payload_len = (static_cast<uint16_t>(static_cast<uint8_t>(decoded_header[second_start])) << 8) |
                                           static_cast<uint16_t>(static_cast<uint8_t>(decoded_header[second_start + 1]));

                    bool all_equal = true;
                    for (size_t k = 1; k < half2/2; ++k) {
                        size_t idx = second_start + k*2;
                        uint16_t v = (static_cast<uint16_t>(static_cast<uint8_t>(decoded_header[idx])) << 8) |
                                     static_cast<uint16_t>(static_cast<uint8_t>(decoded_header[idx+1]));
                        if (v != payload_len) { all_equal = false; break; }
                    }
                    if (!all_equal) {
                        LOG_G_WARN() << "[Demod] Warning: payload_len inconsistency in repetition area";
                        llr_offset += bits_per_block;
                        continue;
                    }

                    // Calculate padding and required LLR count (using bytes_per_ldpc_block and decoder N)
                    size_t padded_len = ((payload_len + bytes_per_ldpc_block - 1) / bytes_per_ldpc_block) * bytes_per_ldpc_block;
                    size_t num_blocks = padded_len / bytes_per_ldpc_block;
                    size_t required_llr = num_blocks * bits_per_block;

                    llr_offset += bits_per_block; // Move past header LLR (one LDPC block)
                    
                    // 7. Check if there is enough LLR data
                    if (llr_offset + required_llr <= frame_llr.llr.size()) {
                        // 8. Extract payload LLR
                        LDPCCodec::AlignedFloatVector payload_llr(required_llr);
                        std::copy(frame_llr.llr.begin() + llr_offset,
                                 frame_llr.llr.begin() + llr_offset + required_llr,
                                 payload_llr.begin());
                        
                        // 9. Soft descramble payload data
                        _descrambler.soft_descramble(payload_llr);
                        
                        // 10. LDPC soft decode payload data
                        LDPCCodec::AlignedByteVector decoded_payload;
                        try {
                            _ldpc_decoder.decode_frame(payload_llr, decoded_payload);
                            
                            // 11. Send decoded UDP data directly (padding removed)
                            std::vector<uint8_t> udp_data(decoded_payload.begin(), decoded_payload.begin() + payload_len);

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
                                _udp_output_sender->send(udp_data.data(), udp_data.size());
                                if (!latency_recorded &&
                                    do_latency_profile &&
                                    frame_llr.rx_enqueue_time_ns > 0 &&
                                    frame_llr.process_dequeue_time_ns > 0 &&
                                    frame_llr.demod_done_time_ns > 0) {
                                    const int64_t udp_done_time_ns = host_now_ns();
                                    _latency_accumulator.rx_queue_total_ns.fetch_add(
                                        frame_llr.process_dequeue_time_ns - frame_llr.rx_enqueue_time_ns,
                                        std::memory_order_relaxed);
                                    _latency_accumulator.demod_total_ns.fetch_add(
                                        frame_llr.demod_done_time_ns - frame_llr.process_dequeue_time_ns,
                                        std::memory_order_relaxed);
                                    _latency_accumulator.bit_total_ns.fetch_add(
                                        udp_done_time_ns - frame_llr.demod_done_time_ns,
                                        std::memory_order_relaxed);
                                    _latency_accumulator.e2e_total_ns.fetch_add(
                                        udp_done_time_ns - frame_llr.process_dequeue_time_ns,
                                        std::memory_order_relaxed);
                                    _latency_accumulator.count.fetch_add(1, std::memory_order_relaxed);
                                    latency_recorded = true;
                                }
                            }
                            // LOG_G_INFO() << "[Demod] Successfully reconstructed and sent UDP packet, size: " << udp_data.size() << " bytes" << std::endl;

                            llr_offset += required_llr; // Move past payload data
                            
                        } catch (const std::exception& e) {
                            LOG_G_WARN() << "[Demod] Payload LDPC decode failed: " << e.what();
                            llr_offset += required_llr; // Skip this data chunk
                        }
                    } else {
                        // Insufficient data, exit loop
                        break;
                    }
                    }
            }
            // Return LLR buffer to pool for reuse
            _llr_pool.release(std::move(frame_llr.llr));
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
    const std::string default_config_file = "Demodulator.yaml";
    Config cfg = make_default_demodulator_config();

    if (argc > 1) {
        LOG_G_ERROR() << "CLI parameters are no longer supported. Please configure OFDMDemodulator via "
                      << default_config_file << ".";
        return 1;
    }

    if (!path_exists(default_config_file)) {
        LOG_G_ERROR() << "Config file '" << default_config_file
                      << "' not found. Copy a sample file from the repository config directory, "
                      << "such as 'Demodulator_X310.yaml' or 'Demodulator_B210.yaml', to '" << default_config_file
                      << "' and edit it before starting OFDMDemodulator.";
        return 1;
    }

    if (!load_demodulator_config_from_yaml(cfg, default_config_file)) {
        return 1;
    }

    LOG_G_INFO() << "Loaded config from: " << default_config_file;
    finalize_demodulator_network_defaults(cfg);
    log_demodulator_sync_mode(cfg);
    log_demodulator_agc_mode(cfg);
    uhd::set_thread_priority_safe(1, true);
    // Use last available core for main thread
    if (!cfg.available_cores.empty()) {
        std::vector<size_t> cpu_list = {cfg.available_cores.back()};
        uhd::set_thread_affinity(cpu_list);
    }
    
    // Load FFTW wisdom
    FFTWManager::import_wisdom();

    OFDMRxEngine receiver(cfg);
    receiver.start();
    
    while (!stop_signal.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    receiver.stop();
    
    // Save FFTW wisdom
    FFTWManager::export_wisdom();
    
    return 0;
}
