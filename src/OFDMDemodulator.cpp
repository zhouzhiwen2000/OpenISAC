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
#include <Common.hpp>
#include "OFDMCore.hpp"
#include "SensingChannel.hpp"

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
              const size_t data_subcarriers = cfg.fft_size - cfg.pilot_positions.size();
              const size_t llr_size = (cfg.num_symbols - 1) * data_subcarriers * 2;
              return AlignedFloatVector(llr_size);
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
    };
    SPSCRingBuffer<LlrFrame> _data_llr_buffer{128};  // LLR data buffer
    std::thread _bit_processing_thread;
    std::atomic<bool> _bit_processing_running{false};
    
    LDPCCodec _ldpc_decoder{make_ldpc_5041008_cfg()};
    Scrambler _descrambler{201600, 0x5A};
    std::unique_ptr<UdpSender> _udp_output_sender;

    // Noise/LLR estimation related
    double _noise_var{0.5};              // Complex noise power E[|n|^2] initial value (assume 0.25 per dimension)
    double _llr_scale{2.0};              // LLR scaling factor (updated based on noise variance)
    double _snr_linear{1.0};             // Es/N0 Linear value
    double _snr_db{0.0};                 // Es/N0 dB
    
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
        _register_commands();
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
        // Acquire pre-allocated RX frame from pool
        RxFrame frame = _rx_frame_pool.acquire();
        frame.Alignment = 0;
        frame.usrp_time_ns = -1;
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
                process_ofdm_frame(frame);
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
    void process_ofdm_frame(const RxFrame& frame) {
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
        _channel_estimator.estimate_from_sync_lmmse(sync_symbol_freq, zc_freq_, H_est, cfg_.cp_length);
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

        // ================= Noise Variance Estimation (Based on equalized pilot error) =================
        prof_step_start = ProfileClock::now();
        // Theoretical constellation points of pilots after equalization: corresponding positions in zc_freq_ (amplitude ~1). Error = y_eq_pilot - x_pilot
        // Estimate complex noise power E[|n|^2] = mean(|e|^2). Under BPSK/QPSK, LLR = 2*Re{y*x*}/sigma2 (Gray mapping splits per bit)
        if (!cfg_.pilot_positions.empty()) {
            double err_power_acc = 0.0;
            size_t err_count = 0;
            for (const auto &sym : symbols) {
                for (auto p : cfg_.pilot_positions) {
                    if (p < sym.size()) {
                        std::complex<float> y_eq = sym[p];
                        std::complex<float> x_ref = zc_freq_[p];
                        auto e = y_eq - x_ref; // Error
                        err_power_acc += std::norm(e);
                        err_count++;
                    }
                }
            }
            if (err_count > 8) { // Sufficient samples
                _noise_var = err_power_acc / err_count; // Complex noise power
                if (_noise_var < 1e-6) _noise_var = 1e-6;
                // QPSK energy per symbol ~1 (normalized), Es/N0 = 1 / noise_var
                _snr_linear = 1.0 / _noise_var;
                _snr_db = 10.0 * std::log10(_snr_linear);
                // LLR scaling: For QPSK Gray, bit LLR ≈ 2 * component / (sigma^2), where sigma^2 = noise variance per dimension = noise_var/2
                double sigma2_dim = _noise_var / 2.0;
                _llr_scale = 2.0 / sigma2_dim; // = 4 / noise_var
            }
        }
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
        
        // Extract data symbols to generate LLR using QPSK_LLR
        prof_step_start = ProfileClock::now();
        const size_t data_subcarriers_per_symbol = cfg_.fft_size - cfg_.pilot_positions.size();
        // Acquire pre-allocated LLR buffer from pool
        AlignedFloatVector frame_llr = _llr_pool.acquire();
        float scale_llr = std::min(static_cast<float>(_llr_scale * M_SQRT1_2), 500.0f);
        
        // Build data subcarrier index list (once, cached)
        static std::vector<size_t> data_indices;
        if (data_indices.empty() || data_indices.size() != data_subcarriers_per_symbol) {
            data_indices.clear();
            data_indices.reserve(data_subcarriers_per_symbol);
            for (size_t k = 0; k < cfg_.fft_size; ++k) {
                if (!pilot_mask[k]) {
                    data_indices.push_back(k);
                }
            }
        }
        
        // Compute LLR using QPSK_LLR
        QPSK_LLR::compute_llr_to_buffer(symbols, data_indices, scale_llr, frame_llr.data());
        
        // Put LLR data into circular buffer
        if (!spsc_wait_push(_data_llr_buffer, LlrFrame{std::move(frame_llr), frame.generation}, [this]() {
                return !_bit_processing_running.load(std::memory_order_acquire);
            })) {
            _llr_pool.release(std::move(frame_llr));
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

            // Debug: Periodically print current SNR
            // static size_t snr_print_counter = 0;
            // if ((snr_print_counter++ & 0x3F) == 0) { // Every 64 frames
            //     LOG_G_INFO() << "[LLR] SNR(dB): " << _snr_db << " noise_var: " << _noise_var << " llr_scale: " << _llr_scale << std::endl;
            // }
            
            // Process frame data, block by block according to decoder N (bits)
            size_t bits_per_block = _ldpc_decoder.get_N();
            size_t llr_offset = 0;
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
                            
                            // 12. Send UDP data
                            _udp_output_sender->send(udp_data.data(), udp_data.size());
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

    if (!std::filesystem::exists(default_config_file)) {
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
