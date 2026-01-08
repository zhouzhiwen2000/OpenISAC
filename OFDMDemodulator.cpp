#include <uhd/utils/thread.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <boost/circular_buffer.hpp>
#include <complex>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <immintrin.h>
#include <fftw3.h>
#include <csignal>
#include <fstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <functional>
#include <memory>
#include <utility>
#include <Common.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

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
          _accumulated_rx_symbols(),
          _accumulated_tx_symbols(),
          _sensing_sender(cfg.bi_sensing_ip, cfg.bi_sensing_port, cfg.fft_size * cfg.num_symbols),
          _control_handler(cfg.control_port),
          channel_sender_(2, [this](const auto& data) { 
              channel_udp_->send_container(data); 
          }, std::chrono::milliseconds(50)),
          pdf_sender_(2, [this](const auto& data) { 
              pdf_udp_->send_container(data); 
          }, std::chrono::milliseconds(50)),
          constellation_sender_(10, [this](const auto& data) { 
              constellation_udp_->send_container(data); 
          }, std::chrono::milliseconds(50)) {
        init_usrp();
        init_filter();
        prepare_tx_sync_sequence();
        prepare_fftw();
        init_udp();
        if (cfg_.hardware_sync) {
            _hw_sync = std::make_unique<HardwareSyncController>(cfg_.hardware_sync_tty);
        }
        // Initialize sensing processing
        init_sensing();
        // Initialize data processing
        init_data_processing();
    }

    ~OFDMRxEngine() {
        stop();
        fftwf_destroy_plan(fft_plan_);
        fftwf_destroy_plan(ifft_plan_);
        fftwf_destroy_plan(_range_ifft_plan);
        fftwf_destroy_plan(_doppler_fft_plan);
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
        sensing_cv_.notify_all();
        if (sensing_thread_.joinable()) sensing_thread_.join();
        
        // Stop data processing thread
        _bit_processing_running.store(false);
        _data_llr_cv.notify_all();
        if (_bit_processing_thread.joinable()) _bit_processing_thread.join();
        
        // Stop all senders
        channel_sender_.stop();
        pdf_sender_.stop();
        constellation_sender_.stop();
        _control_handler.stop();
    }

private:
    enum class RxState { SYNC_SEARCH, ALIGNMENT, NORMAL };

    Config cfg_;
    uhd::usrp::multi_usrp::sptr usrp_;
    uhd::rx_streamer::sptr rx_stream_;
    
    AlignedVector zc_freq_;
    AlignedVector tx_sync_symbol_; 
    std::atomic<int> sync_offset_{0};
    std::atomic<bool> sync_done_{false};
    std::atomic<RxState> state_{RxState::SYNC_SEARCH};
    std::atomic<int> discard_samples_{0};
    
    boost::circular_buffer<RxFrame> frame_queue_{8};
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    boost::circular_buffer<AlignedVector> sync_queue_{8};
    std::mutex sync_queue_mutex_;
    std::condition_variable sync_queue_cv_;
    std::vector<AlignedVector> _accumulated_rx_symbols;
    std::vector<AlignedVector> _accumulated_tx_symbols;

    AlignedVector fft_input_;
    AlignedVector fft_output_;
    fftwf_plan fft_plan_;
    
    AlignedVector ifft_input_;
    AlignedVector ifft_output_;
    fftwf_plan ifft_plan_;
    
    std::unique_ptr<FIRFilter> freq_offset_filter_;
    uhd::tune_result_t current_rx_tune_;
    bool tune_initialized_ = false;
    AlignedFloatVector sync_real_;
    AlignedFloatVector sync_imag_;
    std::atomic<bool> running_{false};
    std::thread rx_thread_, process_thread_;

    using AlignedAlloc = AlignedAllocator<std::complex<float>, 32>;

    // Various UDP senders
    std::unique_ptr<UdpSender> channel_udp_;
    std::unique_ptr<UdpSender> pdf_udp_;
    std::unique_ptr<UdpSender> constellation_udp_;
    std::unique_ptr<UdpSender> freq_offset_udp_;
    SensingDataSender _sensing_sender;
    // Control handler
    ControlCommandHandler _control_handler;
    // Data sender management
    DataSender<std::complex<float>, AlignedAlloc> channel_sender_;
    DataSender<std::complex<float>, AlignedAlloc> pdf_sender_;
    DataSender<std::complex<float>, AlignedAlloc> constellation_sender_;
    uint32_t _reset_count = 0;
    
    // Sensing related variables
    boost::circular_buffer<SensingFrame> sensing_queue_{4};
    std::mutex sensing_mutex_;
    std::condition_variable sensing_cv_;
    std::thread sensing_thread_;
    std::atomic<bool> sensing_running_{false};

    // FFTW Plans
    AlignedVector _channel_response_buffer;
    AlignedVector _sensing_fft_in;
    AlignedVector _sensing_fft_out;
    fftwf_plan _range_ifft_plan;
    fftwf_plan _doppler_fft_plan;
    
    AlignedFloatVector _range_window;    // Range window
    AlignedFloatVector _doppler_window;  // Doppler window
    std::atomic<bool> skip_sensing_fft{true};
    std::vector<bool> is_pilot;
    size_t _global_symbol_index = 0; // Global symbol index (for sensing processing)
    std::atomic<size_t> sensing_sybmol_stride{20};
    size_t shadow_sensing_symbol_stride=20; // Shadow sensing symbol stride

    bool _saved = false; // Whether data has been saved
    std::chrono::steady_clock::time_point _next_send_time = std::chrono::steady_clock::now();
    double _freq_offset_sum = 0.0f; // Average frequency offset
    size_t _freq_offset_count = 0; // Sensing symbol count
    double _avg_freq_offset = 0.0;
    std::vector<int> _actual_subcarrier_indices;
    std::vector<float> _subcarrier_phases_unit_delay;
    SFOEstimator sfo_estimator{1000};
    bool _sync_in_progress = false; // Flag to indicate if sync is in progress in process_proc
    int _last_adjusted_index = 0; // Last adjusted index
    uint32_t _delay_adjustment_count = 0;
    std::atomic<float> _user_delay_offset = 0.0f;
    std::unique_ptr<HardwareSyncController> _hw_sync;
    
    // MTI Filter
    MTIFilter _mti_filter;
    std::atomic<bool> enable_mti{true};

    // Data processing related member variables
    boost::circular_buffer<AlignedFloatVector> _data_llr_buffer{128};  // LLR data buffer
    std::mutex _data_llr_mutex;
    std::condition_variable _data_llr_cv;
    std::thread _bit_processing_thread;
    std::atomic<bool> _bit_processing_running{false};
    
    LDPCCodec _ldpc_decoder{LDPCCodec::LDPCConfig()};
    Scrambler _descrambler{201600, 0x5A};
    std::unique_ptr<UdpSender> _udp_output_sender;

    // Noise/LLR estimation related
    double _noise_var{0.5};              // Complex noise power E[|n|^2] initial value (assume 0.25 per dimension)
    double _llr_scale{2.0};              // LLR scaling factor (updated based on noise variance)
    double _snr_linear{1.0};             // Es/N0 Linear value
    double _snr_db{0.0};                 // Es/N0 dB
    
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
        usrp_->set_rx_rate(cfg_.sample_rate);
        usrp_->set_rx_bandwidth(cfg_.bandwidth, cfg_.rx_channel);
        current_rx_tune_ = usrp_->set_rx_freq(uhd::tune_request_t(cfg_.center_freq), cfg_.rx_channel);
        tune_initialized_ = true;
        std::cout << "Actual RX Freq: " << current_rx_tune_.actual_rf_freq / 1e6 
              << " MHz, DSP: " << current_rx_tune_.actual_dsp_freq
              << " Hz" << std::endl;
        usrp_->set_rx_gain(cfg_.rx_gain, cfg_.rx_channel);
        usrp_->set_clock_source(cfg_.clocksource);

        uhd::stream_args_t args("fc32", cfg_.wire_format_tx);
        args.channels = {cfg_.rx_channel};
        rx_stream_ = usrp_->get_rx_stream(args);
    }

    void _register_commands() {
        // Register skip sensing FFT command
        _control_handler.register_command("SKIP", [this](int32_t value) {
            skip_sensing_fft.store(value);
            std::cout << "Received skip sensing FFT command: " << value << std::endl;
        });
        
        // Register stride setting command
        _control_handler.register_command("STRD", [this](int32_t value) {
            sensing_sybmol_stride.store(value);
            std::cout << "Set sensing stride to: " << value << std::endl;
        });

        // Register alignment command
        _control_handler.register_command("ALGN", [this](int32_t value) {
            int32_t adjusted_value = value;
            if (adjusted_value < -1000) adjusted_value = -1000;
            if (adjusted_value > 1000) adjusted_value = 1000;
            _user_delay_offset = _user_delay_offset - static_cast<float>(adjusted_value);
            std::cout << "Received alignment command: " << adjusted_value << " samples" << std::endl;
        });

        // Register MTI command
        _control_handler.register_command("MTI ", [this](int32_t value) {
            enable_mti.store(value != 0);
            std::cout << "MTI " << (value ? "Enabled" : "Disabled") << std::endl;
        });
    }

    void init_sensing() {
        is_pilot.resize(cfg_.fft_size, false);
        for (auto pos : cfg_.pilot_positions) {
            if (pos < cfg_.fft_size) {
                is_pilot[pos] = true;
            }
        }
        _actual_subcarrier_indices.resize(cfg_.fft_size);
        _subcarrier_phases_unit_delay.resize(cfg_.fft_size);
        const size_t half_fft = static_cast<int>(cfg_.fft_size) / 2;
        #pragma omp simd
        for (size_t i = 0; i < cfg_.fft_size; ++i) {
            // Calculate actual frequency index of subcarriers (negative to positive)
            // Range: [-fft_size/2, fft_size/2 - 1]
            _actual_subcarrier_indices[i] = (i >= half_fft) ? 
                (static_cast<int>(i) - cfg_.fft_size) : 
                i;
            _subcarrier_phases_unit_delay[i] = 
                -2 * M_PI * _actual_subcarrier_indices[i] / cfg_.fft_size;
        }
        // Allocate memory
        _channel_response_buffer.resize(cfg_.range_fft_size * cfg_.doppler_fft_size);
        _mti_filter.resize(cfg_.range_fft_size);
        
        const int fft_size_int = static_cast<int>(cfg_.range_fft_size);
        const int doppler_fft_size_int = static_cast<int>(cfg_.doppler_fft_size);
        
        _range_ifft_plan = fftwf_plan_many_dft(
            1,                         // rank
            &fft_size_int,             // n (FFT size)
            doppler_fft_size_int,           // howmany (number of symbols)
            reinterpret_cast<fftwf_complex*>(_channel_response_buffer.data()),
            nullptr,                   // inembed (contiguous)
            1,                         // istride
            fft_size_int,              // idist (distance between FFTs)
            reinterpret_cast<fftwf_complex*>(_channel_response_buffer.data()),
            nullptr,                   // onembed
            1,                         // ostride
            fft_size_int,              // odist
            FFTW_BACKWARD,             // sign (IFFT)
            FFTW_MEASURE
        );
        
        // Create Doppler FFT plan (for sensing)
        _doppler_fft_plan = fftwf_plan_many_dft(
            1,                         // rank
            &doppler_fft_size_int,          // n (number of symbols)
            fft_size_int,              // howmany (number of subcarriers)
            reinterpret_cast<fftwf_complex*>(_channel_response_buffer.data()),
            nullptr,                   // inembed
            fft_size_int,              // istride (stride between symbols)
            1,                         // idist (distance between subcarriers)
            reinterpret_cast<fftwf_complex*>(_channel_response_buffer.data()),
            nullptr,                   // onembed
            fft_size_int,              // ostride
            1,                         // odist
            FFTW_FORWARD,              // sign (FFT)
            FFTW_MEASURE
        );
        
        // Initialize window functions
        _range_window.resize(cfg_.fft_size);
        _doppler_window.resize(cfg_.num_symbols);
        
        // Hanning window - Range dimension
        #pragma omp simd
        for (size_t i = 0; i < cfg_.fft_size; ++i) {
            _range_window[i] = 0.5f * (1 - std::cos(2 * M_PI * i / (cfg_.fft_size - 1)));
        }
        
        // Hanning window - Doppler dimension
        #pragma omp simd
        for (size_t i = 0; i < cfg_.num_symbols; ++i) {
            _doppler_window[i] = 0.5f * (1 - std::cos(2 * M_PI * i / (cfg_.num_symbols - 1)));
        }
        // Sensing processing thread
        sensing_running_.store(true);
        sensing_thread_ = std::thread(&OFDMRxEngine::sensing_process_proc, this);
        _accumulated_rx_symbols.reserve(cfg_.sensing_symbol_num);
        _accumulated_tx_symbols.reserve(cfg_.sensing_symbol_num);
        // UDP data sender initialization
        _sensing_sender.start();
        // Register control command handler
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

    void prepare_tx_sync_sequence() {
        // Generate frequency domain ZC sequence (consistent with modulator)
        zc_freq_.resize(cfg_.fft_size);
        const int N = cfg_.fft_size;
        const int q = cfg_.zc_root;
        const int delta = (N & 1);  // even N -> 0, odd N -> 1
        const float norm = 1.0f;

        // Pre-compute phase coefficients, use double to avoid precision loss
        const double base = -M_PI * static_cast<double>(q) / static_cast<double>(N);

        #pragma omp simd
        for (int n = 0; n < N; ++n) {
            const double nd = static_cast<double>(n);
            const double arg = nd * (nd + static_cast<double>(delta)); // n*(n+delta)
            const double phase = base * arg;                           // -π*q/N * n*(n+δ)
            zc_freq_[n] = std::polar(norm, static_cast<float>(phase)); // unit-modulus complex
        }

        // Execute IFFT to generate time-domain sync symbol
        AlignedVector ifft_out(N);
        fftwf_plan plan = fftwf_plan_dft_1d(
            N,
            reinterpret_cast<fftwf_complex*>(zc_freq_.data()),
            reinterpret_cast<fftwf_complex*>(ifft_out.data()),
            FFTW_BACKWARD, FFTW_ESTIMATE
        );
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);

        // Add cyclic prefix
        tx_sync_symbol_.resize(N + cfg_.cp_length);
        if (cfg_.cp_length > 0) {
            std::copy(ifft_out.end() - cfg_.cp_length, ifft_out.end(), tx_sync_symbol_.begin());
        }
        std::copy(ifft_out.begin(), ifft_out.end(), tx_sync_symbol_.begin() + cfg_.cp_length);

        const size_t sync_symbol_len = static_cast<size_t>(N) + static_cast<size_t>(cfg_.cp_length);
        sync_real_.resize(sync_symbol_len);
        sync_imag_.resize(sync_symbol_len);

        #pragma omp simd
        for (size_t i = 0; i < sync_symbol_len; ++i) {
            sync_real_[i] = tx_sync_symbol_[i].real();
            sync_imag_[i] = tx_sync_symbol_[i].imag();
        }
    }

    void prepare_fftw() {
        fft_input_.resize(cfg_.fft_size);
        fft_output_.resize(cfg_.fft_size);
        fft_plan_ = fftwf_plan_dft_1d(cfg_.fft_size,
            reinterpret_cast<fftwf_complex*>(fft_input_.data()),
            reinterpret_cast<fftwf_complex*>(fft_output_.data()),
            FFTW_FORWARD, FFTW_MEASURE);

        ifft_input_.resize(cfg_.fft_size);
        ifft_output_.resize(cfg_.fft_size);
        ifft_plan_ = fftwf_plan_dft_1d(cfg_.fft_size,
            reinterpret_cast<fftwf_complex*>(ifft_input_.data()),
            reinterpret_cast<fftwf_complex*>(ifft_output_.data()),
            FFTW_BACKWARD, FFTW_MEASURE);
    }

    void init_udp() {
        // Use IP and port from configuration
        channel_udp_ = std::make_unique<UdpSender>(cfg_.channel_ip, cfg_.channel_port);
        pdf_udp_ = std::make_unique<UdpSender>(cfg_.pdf_ip, cfg_.pdf_port);
        constellation_udp_ = std::make_unique<UdpSender>(cfg_.constellation_ip, cfg_.constellation_port);
        freq_offset_udp_ = std::make_unique<UdpSender>(cfg_.freq_offset_ip, cfg_.freq_offset_port);
    }

    /**
     * @brief Rx Streamer Thread Function.
     * 
     * Continuous loop that receives baseband samples from the USRP.
     * Implements a state machine (SYNC_SEARCH -> ALIGNMENT -> NORMAL) to handle
     * frame synchronization and alignment before normal reception.
     */
    void rx_proc() {
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
        AlignedVector sync_data(cfg_.sync_samples());
        size_t received = 0;
        while (received < cfg_.sync_samples() && running_.load()) {
            received += rx_stream_->recv(&sync_data[received], cfg_.sync_samples() - received, md);
        }
        {
            std::lock_guard<std::mutex> lock(sync_queue_mutex_);
            if(!sync_queue_.full()) {
                sync_queue_.push_back(std::move(sync_data));
            }
        }
        sync_queue_cv_.notify_one();
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
        while (received < total_read && running_.load()) {
            received += rx_stream_->recv(&temp_buf[received], total_read - received, md);
        }
        RxFrame frame;
        frame.frame_data.resize(cfg_.samples_per_frame());
        frame.Alignment = discard_samples_;
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

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            frame_queue_.push_back(std::move(frame));
        }
        queue_cv_.notify_one();
//        std::cout << "Alignment done, moving "<< discard_samples_<< " samples" << std::endl;
//        std::cout <<  discard_samples_<< std::endl;
        state_ = RxState::NORMAL;
    }

    /**
     * @brief Handle Normal Reception State.
     * 
     * Reads complete frames of aligned samples and pushes them to the frame queue
     * for processing.
     */
    void handle_normal_rx(uhd::rx_metadata_t& md) {
        RxFrame frame;
        frame.frame_data.resize(cfg_.samples_per_frame());
        frame.Alignment = 0;
        size_t received = 0;

        while (received < cfg_.samples_per_frame() && running_.load()) {
            received += rx_stream_->recv(&frame.frame_data[received], cfg_.samples_per_frame() - received, md);
        }

        if (received > 0) {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            frame_queue_.push_back(std::move(frame));
            queue_cv_.notify_one();
        }
    }

    /**
     * @brief Process Synchronization Data.
     * 
     * Performs cross-correlation with the local Zadoff-Chu sequence to detect frame start.
     * Also estimates coarse Carrier Frequency Offset (CFO) using Cyclic Prefix (CP) correlation.
     */
    void process_sync_data(const AlignedVector& sync_data) {
        const size_t symbol_len = cfg_.fft_size + cfg_.cp_length;
        size_t n_windows = sync_data.size() - symbol_len + 1;
        float max_corr = 0.0f;
        float average_corr = 0.0f;
        int max_pos = 0;
        
        // Sliding window to calculate sync symbol correlation
        for (size_t i = 0; i < n_windows; ++i) {
            float sum_real = 0.0f, sum_imag = 0.0f;

            #pragma omp simd reduction(+:sum_real, sum_imag)
            for (size_t j = 0; j < symbol_len; ++j) {
                const float rx_real = sync_data[i + j].real();
                const float rx_imag = sync_data[i + j].imag();

                sum_real += rx_real * sync_real_[j] + rx_imag * sync_imag_[j];
                sum_imag += rx_imag * sync_real_[j] - rx_real * sync_imag_[j];
            }

            const float corr = sum_real * sum_real + sum_imag * sum_imag;

            if (corr > max_corr) {
                max_corr = corr;
                max_pos = i;
            }
            average_corr += corr;
        }
        average_corr /= n_windows;
        
        const float energy_threshold = 100.0f;
        if (max_corr/average_corr > energy_threshold) {
            std::cout << "Sync found at pos: " << max_pos 
                    << " with value: " << max_corr 
                    << " Threshold: " << energy_threshold << std::endl;
            int symbol_offset = max_pos % symbol_len;
            // Calculate available symbol count
            const size_t available_symbols = std::min(
                static_cast<size_t>(cfg_.num_symbols*2),
                (sync_data.size() - symbol_offset) / symbol_len
            );
            
            // Accumulate CP correlation of all symbols using complex numbers
            double total_real = 0.0;
            double total_imag = 0.0;
            const size_t cp_length = cfg_.cp_length;
            const size_t fft_size = cfg_.fft_size;
            int valid_symbol_count = 0;
            
            // Iterate through all available symbols
            for (size_t sym = 0; sym < available_symbols; ++sym) {
                const size_t start_pos = symbol_offset + sym * symbol_len;
                
                // Calculate real and imaginary parts separately
                double sym_real = 0.0;
                double sym_imag = 0.0;
                
                // Calculate CP correlation for this symbol
                #pragma omp simd reduction(+:sym_real, sym_imag)
                for (size_t i = 0; i < cp_length; ++i) {
                    const auto& cp_sample = sync_data[start_pos + i];
                    const auto& tail_sample = sync_data[start_pos + i + fft_size];
                    
                    // Manually calculate conj(cp_sample) * tail_sample
                    sym_real += cp_sample.real() * tail_sample.real() + 
                                cp_sample.imag() * tail_sample.imag();
                    sym_imag += cp_sample.real() * tail_sample.imag() - 
                                cp_sample.imag() * tail_sample.real();
                }
                
                // Accumulate to total sum
                total_real += sym_real;
                total_imag += sym_imag;
                valid_symbol_count++;
            }
            
            std::complex<double> total_sum(total_real, total_imag);
            
            if (valid_symbol_count > 0) {
                // Calculate overall phase difference
                const double phase_diff = std::arg(total_sum);
                
                // Calculate CFO: Δf = phase_diff / (2π * T_symbol)
                const double Ts = 1.0 / cfg_.sample_rate;          // Sampling period
                const double T_symbol = fft_size * Ts;             // Effective symbol duration
                const double cfo = phase_diff / (2.0 * M_PI * T_symbol); // CFO estimate (Hz)
                
                std::cout << "CFO estimate: " << cfo << " Hz (using " << valid_symbol_count 
                        << " symbols)" << std::endl;
                
                // Perform initial CFO correction
                if (cfg_.software_sync){
                    adjust_rx_freq(-cfo, false);
                }
            } else {
                std::cout << "Warning: No valid symbols for CFO estimation" << std::endl;
            }
            
            // Record time offset
            sync_offset_ = (max_pos - cfg_.sync_pos * symbol_len);
            if (sync_offset_ > 0) {
                sync_offset_ = sync_offset_ % cfg_.samples_per_frame();
            }
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                frame_queue_.clear();
            }
            discard_samples_.store(sync_offset_);
            state_ = RxState::ALIGNMENT;
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

        while (running_.load()) {
            if (state_ == RxState::SYNC_SEARCH) {
                AlignedVector sync_data;
                {
                    std::unique_lock<std::mutex> lock(sync_queue_mutex_);
                    sync_queue_cv_.wait(lock, [&]{ return !sync_queue_.empty() || !running_.load();});
                    if (!running_.load()) break;
                    sync_data = std::move(sync_queue_.front());
                    sync_queue_.pop_front();
                }
                process_sync_data(sync_data);
            } else {
                RxFrame frame = wait_for_frame();
                frame_start = Clock::now();
                process_ofdm_frame(frame);
                frame_end = Clock::now();
                double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
                total_processing_time += frame_time;
                frame_count++;

                if (frame_count >= REPORT_INTERVAL) {
                    double avg_time = total_processing_time / frame_count;
                    double frame_duration = cfg_.samples_per_frame() / cfg_.sample_rate * 1000.0;
                    double load = avg_time / frame_duration;
                    std::cout << "Average processing time: " << avg_time 
                              << " ms, Load: " << load * 100.0 << "%" << std::endl;
                    std::cout << "Actual RX Freq: " << current_rx_tune_.actual_rf_freq / 1e6 
                              << " MHz, DSP: " << current_rx_tune_.actual_dsp_freq
                              << " Hz" << std::endl;
                    std::cout << "Average CFO: " << _avg_freq_offset << " Hz" << std::endl;
                    total_processing_time = 0.0;
                    frame_count = 0;
                }
            }
        }
    }

    RxFrame wait_for_frame() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [&]{ return !frame_queue_.empty() || !running_.load(); });
        RxFrame frame = std::move(frame_queue_.front());
        frame_queue_.pop_front();
        return frame;
    }

    void fft_shift(const AlignedVector& data, AlignedVector& shifted) {
        const size_t n = data.size();
        const size_t half = n / 2;
        shifted.resize(n);
        
        #pragma omp simd
        for (size_t i = 0; i < half; ++i) {
            shifted[i] = data[i + half];
            shifted[i + half] = data[i];
        }
    }

    int fftshift_index(int original_index, int N) {
        return (original_index + N/2) % N;
    }

    // Use bit manipulation to detect NaN (compatible with -ffast-math)
    inline bool isNaN(float x) {
        uint32_t bits;
        static_assert(sizeof(float) == sizeof(uint32_t), "Unexpected float size");
        memcpy(&bits, &x, sizeof(float));
        constexpr uint32_t exponent_mask = 0x7F800000;  // Exponent mask
        constexpr uint32_t mantissa_mask = 0x007FFFFF;   // Mantissa mask
        return ((bits & exponent_mask) == exponent_mask) && (bits & mantissa_mask);
    }

    // Quinn's algorithm for fractional delay estimation
    /**
     * @brief Quinn's Algorithm for Fractional Delay Estimation.
     * 
     * Refines the peak position in the delay spectrum to sub-sample precision.
     * Used for accurate timing offset estimation.
     */
    float estimateFractionalDelay(const AlignedVector& spectrum, size_t max_index) {
        const size_t N = spectrum.size();
        if (N < 3) return 0.0f;
        const auto& d0 = spectrum[max_index];
        const auto& d_prev = spectrum[(max_index == 0) ? (N - 1) : (max_index - 1)];
        const auto& d_next = spectrum[(max_index == N - 1) ? 0 : (max_index + 1)];
        const auto magnitude = std::abs(d0);
        constexpr float EPSILON = 1e-10f;
        if (magnitude < EPSILON) {
            return 0.0f;
        }
        float alpha1 = 0.0f, alpha2 = 0.0f;
        {
            const float denom = std::real(std::conj(d0) * d0);
            const float num1 = std::real(std::conj(d_prev) * d0);
            const float num2 = std::real(std::conj(d_next) * d0);
            
            // Bounded calculation
            if (denom > EPSILON) {
                alpha1 = num1 / denom;
                alpha2 = num2 / denom;
            } else {
                // Use amplitude ratio for small denominator
                alpha1 = std::abs(d_prev) / (magnitude + EPSILON);
                alpha2 = std::abs(d_next) / (magnitude + EPSILON);
                std::cout << "Warning: Small denominator in delay estimation, using amplitude ratio." << std::endl;
            }
            
            // Limit ratio range [-0.99, 0.99] to avoid division close to 1 issues
            alpha1 = std::max(-0.9999f, std::min(0.9999f, alpha1));
            alpha2 = std::max(-0.9999f, std::min(0.9999f, alpha2));
        }
        
        // Calculate delta using stable formula
        const float delta1 = alpha1 / (1.0f - alpha1);
        const float delta2 = -alpha2 / (1.0f - alpha2);
        
        // Use custom function to detect NaN (compatible with -ffast-math)
        if (isNaN(delta1) || isNaN(delta2)) {
            return 0.0f;
        }
        
        // Improved selection logic
        const float abs1 = std::abs(delta1);
        const float abs2 = std::abs(delta2);
        
        // Selection logic
        if (abs1 > 2.0f && abs2 > 2.0f) {
            return 0.5f;
        } else if (abs1 > 2.0f) {
            return delta2;
        } else if (abs2 > 2.0f) {
            return delta1;
        } else {
            if(delta1 > 0.0f && delta2 > 0.0f) {
                return delta2;
            } else {
                return delta1;
            }
        }
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
        
        std::vector<AlignedVector> symbols;
        std::vector<AlignedVector> rx_symbols_raw;
        symbols.reserve(cfg_.num_symbols-1);
        rx_symbols_raw.reserve(cfg_.num_symbols);
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
            rx_symbols_raw.push_back(symbol);  // Save raw received symbols for sensing
            
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

        // Calculate initial channel response H_est
        prof_step_start = ProfileClock::now();
        AlignedVector H_est(cfg_.fft_size);
        #pragma omp simd
        for (size_t i = 0; i < cfg_.fft_size; ++i) {
            H_est[i] = sync_symbol_freq[i] / zc_freq_[i];
        }
        prof_step_end = ProfileClock::now();
        prof_channel_est_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        // Estimate frequency offset
        prof_step_start = ProfileClock::now();
        const float T = (cfg_.fft_size + cfg_.cp_length) / cfg_.sample_rate;
        std::vector<int> pilot_indices(cfg_.pilot_positions.size());
        std::vector<float> weights(cfg_.pilot_positions.size(), 0.0f);
        std::vector<float> avg_phase_diff(cfg_.pilot_positions.size(), 0.0f);
        for (size_t j = 0; j < cfg_.pilot_positions.size(); j++) {
            auto pilot_index = cfg_.pilot_positions[j];
            std::complex<double> next_current_sum(0.0f, 0.0f);
             #pragma omp simd
             for (size_t i = cfg_.sync_pos; i < symbols.size() - 1; i++) {
                std::complex<float> current_pilot = symbols[i][pilot_index];
                std::complex<float> next_pilot = symbols[i+1][pilot_index];
                next_current_sum += std::conj(current_pilot) * (next_pilot);
            }
            avg_phase_diff[j] = std::arg(next_current_sum);
            int freq_index = static_cast<int>(pilot_index);
            if (freq_index >= static_cast<int>(cfg_.fft_size)/2) {
                freq_index -= cfg_.fft_size;
            }
            pilot_indices[j]=freq_index;
            weights[j] = std::norm(next_current_sum);
        }
        
        // Phase unwrapping with SIMD optimization
        unwrap(avg_phase_diff);
        auto [beta, alpha] = weightedlinearRegression(pilot_indices, avg_phase_diff, weights);// Calculate fixed phase difference alpha between OFDM symbols caused by CFO, and phase slope beta varying with subcarriers caused by SFO
        prof_step_end = ProfileClock::now();
        prof_cfo_sfo_est_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        // Process each data symbol (add phase compensation)
        float detected_freq_offset = alpha / (2 * M_PI * T);
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
                    if (cfg_.software_sync){
                        std::cout << "Adjusting RX frequency by: " << _avg_freq_offset << " Hz" << std::endl;
                        adjust_rx_freq(-_avg_freq_offset, false);
                    }
                }
                if (cfg_.hardware_sync) {
                        double ppm = _avg_freq_offset / cfg_.center_freq * 1e6; // Calculate PPM
                        double ppm_adjusted;
                        if (abs(_avg_freq_offset) > 1.0){
                            ppm_adjusted = ppm * cfg_.ppm_adjust_factor * 10; // Increase adjustment rate if offset is large
                        } else {
                            ppm_adjusted = ppm * cfg_.ppm_adjust_factor; // Normal adjustment rate
                        }
                        std::cout << "Adjusting OCXO by: " << ppm << " ppm" << std::endl;
                        _hw_sync->set_frequency_control_ppm_relative(ppm_adjusted);
                }
            }
        }

        prof_step_start = ProfileClock::now();
        // Pre-compute H_est inverse (ZF equalizer base) - done once per frame
        static thread_local AlignedVector H_inv;
        if (H_inv.size() != cfg_.fft_size) {
            H_inv.resize(cfg_.fft_size);
        }
        
        #pragma omp simd
        for (size_t j = 0; j < cfg_.fft_size; ++j) {
            const auto& h = H_est[j];
            const float mag_sq = std::norm(h);
            const float inv_mag_sq = 1.0f / mag_sq;
            // H_inv = conj(H_est) / |H_est|^2 = 1/H_est
            H_inv[j] = std::complex<float>(h.real() * inv_mag_sq, -h.imag() * inv_mag_sq);
        }
        
        for (size_t i = 0; i < symbols.size(); ++i) {
            auto& symbol = symbols[i];
            
            const int relative_symbol_index = (i < cfg_.sync_pos) ? 
                (static_cast<int>(i) - static_cast<int>(cfg_.sync_pos)) : 
                (static_cast<int>(i) + 1 - static_cast<int>(cfg_.sync_pos));
            const float phase_diff_CFO = alpha * relative_symbol_index;
            const float beta_rel = beta * relative_symbol_index;

            // Equalization: symbol = symbol * H_inv * exp(-j*phase)
            // Phase rotation: exp(-j*phase) = cos(phase) - j*sin(phase)
            #pragma omp simd
            for (size_t j = 0; j < cfg_.fft_size; ++j) {
                const float phase = beta_rel * _actual_subcarrier_indices[j] + phase_diff_CFO;
                // First multiply by H_inv
                const auto& hinv = H_inv[j];
                const auto& sym = symbol[j];
                const float eq_re = sym.real() * hinv.real() - sym.imag() * hinv.imag();
                const float eq_im = sym.real() * hinv.imag() + sym.imag() * hinv.real();
                // Then rotate by -phase (derotate)
                const float c = std::cos(phase);
                const float s = std::sin(phase);
                symbol[j] = std::complex<float>(eq_re * c + eq_im * s, eq_im * c - eq_re * s);
            }
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

        // Remodulate frequency domain symbols (as TX frequency domain symbol estimation)
        prof_step_start = ProfileClock::now();
        std::vector<AlignedVector> tx_symbols_est(cfg_.num_symbols);
        
        // Pre-compute QPSK constellation points
        constexpr float QPSK_POS = static_cast<float>(M_SQRT1_2);
        constexpr float QPSK_NEG = -static_cast<float>(M_SQRT1_2);

        for (size_t i = 0; i < cfg_.num_symbols; ++i) {
            if (i == cfg_.sync_pos) {
                // Sync symbol uses original ZC sequence
                tx_symbols_est[i] = zc_freq_;
            } else {
                // Data symbol - remodulate based on demodulation results
                tx_symbols_est[i].resize(cfg_.fft_size);
                auto* __restrict__ mod_ptr = tx_symbols_est[i].data();
                const size_t symbol_idx = (i < cfg_.sync_pos) ? i : i - 1;
                const auto* __restrict__ sym_ptr = symbols[symbol_idx].data();
                const auto* __restrict__ zc_ptr = zc_freq_.data();
                
                // SIMD-friendly hard decision without branches
                #pragma omp simd
                for (size_t j = 0; j < cfg_.fft_size; ++j) {
                    const float re = sym_ptr[j].real();
                    const float im = sym_ptr[j].imag();
                    // Use copysign for branchless QPSK mapping
                    mod_ptr[j] = std::complex<float>(
                        std::copysign(QPSK_POS, re),
                        std::copysign(QPSK_POS, im)
                    );
                }

                // Replace pilot positions with known pilots
                for (auto pilot : cfg_.pilot_positions) {
                    mod_ptr[pilot] = zc_ptr[pilot];
                }
            }
        }
        prof_step_end = ProfileClock::now();
        prof_remodulate_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        prof_step_start = ProfileClock::now();
        AlignedVector delay_spectrum(cfg_.fft_size);
        fft_shift(H_est, ifft_input_);
        fftwf_execute(ifft_plan_);
        
        const float scale = 1.0f / sqrtf(cfg_.fft_size);
        #pragma omp simd
        for (size_t i = 0; i < cfg_.fft_size; ++i) {
            delay_spectrum[i] = ifft_output_[i] * scale;
        }

        size_t max_index = 0;
        float max_mag = 0.0f;
        float average_mag = 0.0f;
        for (size_t i = 0; i < cfg_.fft_size; ++i) {
            const float mag = std::abs(delay_spectrum[i]);
            if (mag > max_mag) {
                max_mag = mag;
                max_index = i;
            }
            average_mag += mag;
        }
        average_mag /= cfg_.fft_size;
        // Index adjustment (map 512-1023 to -512~-1)
        int adjusted_index = static_cast<int>(max_index);
        const int half_fft = cfg_.fft_size / 2;
        if (adjusted_index >= half_fft) {
            adjusted_index -= cfg_.fft_size;
        }
        // adjusted_index -= 2*cfg_.delay_adjust_step; // Extra offset of 4 samples to display the LoS more clearly

        float fractional_delay = estimateFractionalDelay(delay_spectrum, max_index);
        if (std::isnan(fractional_delay)) {
            fractional_delay = 0.0f;
        }
        float delay_offset_reading = adjusted_index + fractional_delay;
        prof_step_end = ProfileClock::now();
        prof_delay_spectrum_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        prof_step_start = ProfileClock::now();
        sfo_estimator.update(delay_offset_reading, frame.Alignment);
        auto _sfo_per_frame = sfo_estimator.get_sfo_per_frame();

        // auto sfo_via_estimator = _sfo_per_frame/((cfg_.fft_size+cfg_.cp_length)*cfg_.num_symbols*cfg_.sample_rate);
        // auto sfo_via_beta = -beta/((cfg_.fft_size+cfg_.cp_length)*cfg_.sample_rate/cfg_.fft_size*2*M_PI);
        // std::cout << "SFO via estimator: " << sfo_via_estimator << std::endl;
        // std::cout << "SFO via beta: " << sfo_via_beta << std::endl;
        // std::cout << "ratio:" << sfo_via_beta/sfo_via_estimator <<std::endl;
        if (_sfo_per_frame != 0.0f) {
            //std::cout << "SFO Frame: " << _sfo_per_frame << std::endl;
        }
        auto delay_offset = sfo_estimator.get_sensing_delay_offset();
        // Create delay offset data packet
        size_t packet_size = sizeof(delay_offset) + sizeof(delay_offset_reading) + 4;
        uint8_t* freq_packet = new uint8_t[packet_size];
        memcpy(freq_packet, &delay_offset, sizeof(float));
        memcpy(freq_packet + sizeof(delay_offset), &delay_offset_reading, sizeof(float));
        size_t tail_start = sizeof(delay_offset) + sizeof(delay_offset_reading);
        freq_packet[tail_start]     = 0x00;
        freq_packet[tail_start + 1] = 0x00;
        freq_packet[tail_start + 2] = 0x80;
        freq_packet[tail_start + 3] = 0x7f;
        freq_offset_udp_->send(freq_packet, packet_size);
        delete[] freq_packet;
        if((max_mag/average_mag < 20.0f || (abs(adjusted_index) > cfg_.delay_adjust_step+5))&&(cfg_.software_sync || cfg_.hardware_sync)) {
            _reset_count++;
            if (_reset_count >= 217) { // Approx 0.5 seconds
                _reset_count = 0;
                std::cout << "No valid delay found, resetting state." << std::endl;
                adjust_rx_freq(0.0, true); // Reset frequency
                sfo_estimator.reset(); // Reset SFO estimator
                if (cfg_.hardware_sync) {
                    _hw_sync->reset_frequency_control(); // Reset hardware frequency control
                }
                {
                    std::unique_lock<std::mutex> lock(sync_queue_mutex_);
                    sync_queue_.clear();
                }
                state_ = RxState::SYNC_SEARCH;
                return;
            }
        }
        else
        {
            _reset_count = 0;
        }
        if(_sync_in_progress && frame.Alignment != 0)// Sync command has been executed
        {
            _sync_in_progress = false;
        }
        if(abs(adjusted_index) >= cfg_.delay_adjust_step && abs(adjusted_index) < cfg_.cp_length  && abs(detected_freq_offset) < 100.0f &&( cfg_.software_sync || cfg_.hardware_sync )&& !_sync_in_progress)
        {   
            if (_delay_adjustment_count++ >= 1) {
                _delay_adjustment_count = 0;
                discard_samples_.store(adjusted_index);
                state_ = RxState::ALIGNMENT; // Send sync command. May have delay due to FIFO
                _sync_in_progress = true;
            }
            if (adjusted_index * _last_adjusted_index < 0){
                 // If symbol delay direction changes, it indicates jitter, reset counter
                _delay_adjustment_count = 0;
            }
        }
        else
        {
            _delay_adjustment_count = 0;
        }
        _last_adjusted_index = adjusted_index;
        prof_step_end = ProfileClock::now();
        prof_timing_sync_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        // Add sensing data to queue
        prof_step_start = ProfileClock::now();
        SensingFrame sense_frame;
        sense_frame.rx_symbols = std::move(rx_symbols_raw);
        sense_frame.tx_symbols = std::move(tx_symbols_est);
        sense_frame.CFO = alpha;
        if (_sfo_per_frame != 0.0f) {
            sense_frame.SFO = -_sfo_per_frame * (2 * M_PI) / (cfg_.fft_size * cfg_.num_symbols); // Use it if more accurate SFO estimation is available
        }
        else
        {
            sense_frame.SFO = beta;
        }
        sense_frame.delay_offset = delay_offset + _user_delay_offset;

        {
            std::lock_guard<std::mutex> lock(sensing_mutex_);
            if (!sensing_queue_.full()) {
                sensing_queue_.push_back(std::move(sense_frame));
            }
        }
        sensing_cv_.notify_one();
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
        
        // Extract data symbols to generate LLR
        prof_step_start = ProfileClock::now();
        const size_t data_subcarriers_per_symbol = cfg_.fft_size - cfg_.pilot_positions.size();
        const size_t total_llr_count = symbols.size() * data_subcarriers_per_symbol * 2;
        AlignedFloatVector frame_llr(total_llr_count);
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
        
        float* __restrict__ llr_ptr = frame_llr.data();
        const size_t num_data_sc = data_indices.size();
        
        for (size_t sym_idx = 0; sym_idx < symbols.size(); ++sym_idx) {
            const auto* __restrict__ sym_ptr = symbols[sym_idx].data();
            float* __restrict__ out_ptr = llr_ptr + sym_idx * num_data_sc * 2;
            
            #pragma omp simd
            for (size_t i = 0; i < num_data_sc; ++i) {
                const size_t k = data_indices[i];
                out_ptr[i * 2]     = sym_ptr[k].real() * scale_llr;
                out_ptr[i * 2 + 1] = sym_ptr[k].imag() * scale_llr;
            }
        }
        
        // Put LLR data into circular buffer
        {
            std::unique_lock<std::mutex> lock(_data_llr_mutex);
            if (!_data_llr_buffer.full()) {
                _data_llr_buffer.push_back(std::move(frame_llr));
                _data_llr_cv.notify_one();
            }
        }
        prof_step_end = ProfileClock::now();
        prof_llr_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        // ============== Profiling report ==============
        prof_frame_count++;
        if (prof_frame_count >= PROF_REPORT_INTERVAL && cfg_.enable_profiling) {
            std::cout << "\n========== process_ofdm_frame Profiling (avg per frame, us) ==========" << std::endl;
            std::cout << "FFT (all symbols):    " << prof_fft_total / prof_frame_count << " us" << std::endl;
            std::cout << "Channel Estimation:   " << prof_channel_est_total / prof_frame_count << " us" << std::endl;
            std::cout << "CFO/SFO Estimation:   " << prof_cfo_sfo_est_total / prof_frame_count << " us" << std::endl;
            std::cout << "Equalization:         " << prof_equalization_total / prof_frame_count << " us" << std::endl;
            std::cout << "Noise Estimation:     " << prof_noise_est_total / prof_frame_count << " us" << std::endl;
            std::cout << "Remodulation:         " << prof_remodulate_total / prof_frame_count << " us" << std::endl;
            std::cout << "Delay Spectrum:       " << prof_delay_spectrum_total / prof_frame_count << " us" << std::endl;
            std::cout << "Timing Sync:          " << prof_timing_sync_total / prof_frame_count << " us" << std::endl;
            std::cout << "Sensing Queue:        " << prof_sensing_queue_total / prof_frame_count << " us" << std::endl;
            std::cout << "UDP Send:             " << prof_udp_send_total / prof_frame_count << " us" << std::endl;
            std::cout << "LLR Calculation:      " << prof_llr_total / prof_frame_count << " us" << std::endl;
            double total = prof_fft_total + prof_channel_est_total + prof_cfo_sfo_est_total + 
                          prof_equalization_total + prof_noise_est_total + prof_remodulate_total + 
                          prof_delay_spectrum_total + prof_timing_sync_total + prof_sensing_queue_total + 
                          prof_udp_send_total + prof_llr_total;
            std::cout << "TOTAL:                " << total / prof_frame_count << " us" << std::endl;
            std::cout << "======================================================================\n" << std::endl;
            
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

    void demodulate_symbol(const AlignedVector& symbol) {
        std::vector<uint8_t> bits;
        bits.reserve(cfg_.fft_size * 2);
        
        for (const auto& s : symbol) {
            bits.push_back((s.real() > 0) ? 1 : 0);
            bits.push_back((s.imag() > 0) ? 1 : 0);
        }
    }
    /**
     * @brief Sensing Processing Thread.
     * 
     * Accumulates sensing frames (rx/tx symbols) over time (Coherent Processing Interval).
     * Performs Range-Doppler processing (2D FFT) to generate radar maps.
     * Sends processed channel response data via UDP for visualization.
     */
    void sensing_process_proc() {
        uhd::set_thread_priority_safe(1);
        // Assign to available core (index 2)
        size_t core_idx = 2 % cfg_.available_cores.size();
        std::vector<size_t> cpu_list = {cfg_.available_cores[core_idx]};
        uhd::set_thread_affinity(cpu_list);
        while (sensing_running_.load()) {
            bool has_frame = false;
            SensingFrame frame;
            {
                std::unique_lock<std::mutex> lock(sensing_mutex_);
                sensing_cv_.wait_for(lock, std::chrono::milliseconds(10), 
                    [&]{ return !sensing_queue_.empty() || !sensing_running_.load(); });
                if (!sensing_queue_.empty()) {
                    frame = std::move(sensing_queue_.front());
                    sensing_queue_.pop_front();
                    has_frame = true;
                }
            }
            if (has_frame) {
                const size_t symbols_in_this_frame = frame.rx_symbols.size();
                auto alpha = frame.CFO;
                auto beta = frame.SFO;
                auto delay_offset = frame.delay_offset;      
                while (_global_symbol_index < symbols_in_this_frame) {
                    const size_t symbol_idx = _global_symbol_index;
                    // Compensate phase for sensing symbols
                    auto& rx_symbol = frame.rx_symbols[symbol_idx];
                    int relative_symbol_index = symbol_idx- cfg_.sync_pos;
                    auto phase_diff_CFO = alpha*relative_symbol_index;
                    #pragma omp simd
                    for (size_t j = 0; j < rx_symbol.size(); ++j) {
                        auto phase_diff_SFO = beta * _actual_subcarrier_indices[j] * relative_symbol_index;
                        float phase_diff_delay = _subcarrier_phases_unit_delay[j] * delay_offset;
                        auto phase_diff_total = phase_diff_delay + phase_diff_SFO + phase_diff_CFO;
                        auto phase_diff = std::polar(1.0f, - phase_diff_total);
                        rx_symbol[j] = rx_symbol[j] * phase_diff;
                    }
                    // Add current symbol to accumulation buffer
                    _accumulated_rx_symbols.push_back(std::move(rx_symbol));
                    _accumulated_tx_symbols.push_back(std::move(frame.tx_symbols[symbol_idx]));
                    _global_symbol_index += shadow_sensing_symbol_stride;
                    
                    // Check if processing threshold is reached
                    if (_accumulated_tx_symbols.size() >= cfg_.sensing_symbol_num) {
                        // Construct sensing frame
                        SensingFrame sensing_frame;
                        sensing_frame.rx_symbols = std::move(_accumulated_rx_symbols);
                        sensing_frame.tx_symbols = std::move(_accumulated_tx_symbols);
                        
                        // Reset accumulation buffer
                        _accumulated_rx_symbols.clear();
                        _accumulated_tx_symbols.clear();
                        
                        // Execute sensing process
                        _sensing_process(sensing_frame);
                        shadow_sensing_symbol_stride = sensing_sybmol_stride.load();
                    }
                }
                // Frame processing complete, reset global index (subtract symbols in this frame)
                _global_symbol_index -= symbols_in_this_frame;
            }
        }
    }

    // Sensing processing function
    void _sensing_process(const SensingFrame& frame) {
        // Send heartbeat for NAT traversal (once per second)
        static auto next_hb_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (now >= next_hb_time) {
            _control_handler.send_heartbeat(cfg_.bi_sensing_ip, cfg_.bi_sensing_port);
            next_hb_time = now + std::chrono::seconds(1);
        }

        if (std::chrono::steady_clock::now() >= _next_send_time) {
            // 1. Perform OFDM demodulation on received frame
            for (size_t i = 0; i < cfg_.sensing_symbol_num; ++i) {
                // Copy to channel response buffer
                auto* dest = _channel_response_buffer.data() + i * cfg_.range_fft_size;
                std::copy(frame.rx_symbols[i].begin(),
                          frame.rx_symbols[i].end(),
                          dest);
            }
            
            // 2. Channel estimation (Frequency domain division)
            #pragma omp simd
            for (size_t i = 0; i < cfg_.sensing_symbol_num; ++i) {
                for (size_t k = 0; k < cfg_.fft_size; ++k) {
                    size_t idx = i * cfg_.range_fft_size + k;
                    _channel_response_buffer[idx] /= frame.tx_symbols[i][k];
                }
            }

            for (size_t i = 0; i < cfg_.sensing_symbol_num; ++i) {
                auto* symbol_data = _channel_response_buffer.data() + i * cfg_.range_fft_size;
                const size_t half_size = cfg_.fft_size / 2;
                
                // In-place FFT shift - swap front and back halves
                #pragma omp simd
                for (size_t j = 0; j < half_size; ++j) {
                    std::swap(symbol_data[j], symbol_data[j + half_size]);
                }
            }

            if (enable_mti.load()) {
                _mti_filter.apply(_channel_response_buffer, cfg_.fft_size, cfg_.sensing_symbol_num);
            }

            if (!skip_sensing_fft.load()) {

                for (size_t i = 0; i < cfg_.sensing_symbol_num; ++i) {
                    auto* symbol_data = _channel_response_buffer.data() + i * cfg_.range_fft_size;
                    #pragma omp simd
                    for (size_t j = 0; j < cfg_.fft_size; ++j) {
                        // Apply range window (Frequency domain windowing)
                        symbol_data[j] *= _range_window[j];
                    }
                }

                for (size_t bin = 0; bin < cfg_.fft_size; ++bin) {
                    #pragma omp simd
                    for (size_t i = 0; i < cfg_.sensing_symbol_num; ++i) {
                        size_t idx = i * cfg_.range_fft_size + bin;
                        // Apply Doppler window (Time domain windowing)
                        _channel_response_buffer[idx] *= _doppler_window[i];
                    }
                }

                // 3. Batch process Range dimension (Process all symbols together)
                fftwf_execute(_range_ifft_plan);  // Range IFFT (Frequency to Time domain)
                
                // 4. Batch process Doppler dimension (Process all subcarriers together)
                fftwf_execute(_doppler_fft_plan); // Doppler FFT (Time to Frequency domain)
            }

            // 5. Send processing results
            _sensing_sender.push_data(_channel_response_buffer);
            if(cfg_.range_fft_size != cfg_.fft_size || cfg_.doppler_fft_size != cfg_.fft_size) {
                _channel_response_buffer.assign(cfg_.range_fft_size * cfg_.doppler_fft_size, std::complex<float>(0.0f, 0.0f));
            }
            if(!skip_sensing_fft.load()) {
                _next_send_time = _next_send_time + std::chrono::milliseconds(33);
            } else {
                _next_send_time = std::chrono::steady_clock::now();
            }
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
        uhd::set_thread_priority_safe();
        // Assign to available core (index 3)
        size_t core_idx = 3 % cfg_.available_cores.size();
        std::vector<size_t> cpu_list = {cfg_.available_cores[core_idx]};
        uhd::set_thread_affinity(cpu_list);
        
        while (_bit_processing_running.load()) {
            AlignedFloatVector frame_llr;
            
            // Get LLR data from circular buffer
            {
                std::unique_lock<std::mutex> lock(_data_llr_mutex);
                _data_llr_cv.wait_for(lock, std::chrono::milliseconds(100),
                    [this] { return !_data_llr_buffer.empty() || !_bit_processing_running.load(); });
                
                if (!_bit_processing_running.load()) break;
                
                if (!_data_llr_buffer.empty()) {
                    frame_llr = std::move(_data_llr_buffer.front());
                    _data_llr_buffer.pop_front();
                }
            }
            
            if (frame_llr.empty()) continue;

            // Debug: Periodically print current SNR
            // static size_t snr_print_counter = 0;
            // if ((snr_print_counter++ & 0x3F) == 0) { // Every 64 frames
            //     std::cout << "[LLR] SNR(dB): " << _snr_db << " noise_var: " << _noise_var << " llr_scale: " << _llr_scale << std::endl;
            // }
            
            // Process frame data, block by block according to decoder N (bits)
            size_t bits_per_block = _ldpc_decoder.get_N();
            size_t llr_offset = 0;
            while (llr_offset + bits_per_block <= frame_llr.size()) {
                // 1. Extract LLR of current block (corresponding to one LDPC code block)
                LDPCCodec::AlignedFloatVector header_llr(bits_per_block);
                std::copy(frame_llr.begin() + llr_offset,
                         frame_llr.begin() + llr_offset + bits_per_block,
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
                        std::cout << "[Demod] Warning: payload_len inconsistency in repetition area" << std::endl;
                        llr_offset += bits_per_block;
                        continue;
                    }

                    // Calculate padding and required LLR count (using bytes_per_ldpc_block and decoder N)
                    size_t padded_len = ((payload_len + bytes_per_ldpc_block - 1) / bytes_per_ldpc_block) * bytes_per_ldpc_block;
                    size_t num_blocks = padded_len / bytes_per_ldpc_block;
                    size_t required_llr = num_blocks * bits_per_block;

                    llr_offset += bits_per_block; // Move past header LLR (one LDPC block)
                    
                    // 7. Check if there is enough LLR data
                    if (llr_offset + required_llr <= frame_llr.size()) {
                        // 8. Extract payload LLR
                        LDPCCodec::AlignedFloatVector payload_llr(required_llr);
                        std::copy(frame_llr.begin() + llr_offset,
                                 frame_llr.begin() + llr_offset + required_llr,
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
                            // std::cout << "[Demod] Successfully reconstructed and sent UDP packet, size: " << udp_data.size() << " bytes" << std::endl;

                            llr_offset += required_llr; // Move past payload data
                            
                        } catch (const std::exception& e) {
                            std::cerr << "[Demod] Payload LDPC decode failed: " << e.what() << std::endl;
                            llr_offset += required_llr; // Skip this data chunk
                        }
                    } else {
                        // Insufficient data, exit loop
                        break;
                    }
                    }
            }
        }
    }
};

std::atomic<bool> stop_signal(false);

void signal_handler(int) {
    stop_signal.store(true);
}

int UHD_SAFE_MAIN(int argc, char* argv[]) {
    std::signal(SIGINT, &signal_handler);
    Config cfg;
    // Set default values
    cfg.fft_size = 1024;
    cfg.cp_length = 128;
    cfg.center_freq = 2.4e9;
    cfg.pilot_positions = {571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451};
    cfg.range_fft_size = 1024;
    cfg.doppler_fft_size = 100;
    cfg.num_symbols = 100;
    cfg.sensing_symbol_num = 100;
    cfg.sync_pos = 1;
    cfg.sample_rate = 50e6;
    cfg.rx_gain = 50.0;
    cfg.zc_root = 29;
    cfg.device_args = ""; // Default device arguments
    cfg.bi_sensing_ip = "";
    cfg.bi_sensing_port = 8889;
    cfg.control_port = 9999;
    cfg.channel_ip = "";
    cfg.channel_port = 12348;
    cfg.pdf_ip = "";
    cfg.pdf_port = 12349;
    cfg.constellation_ip = "";
    cfg.constellation_port = 12346;
    cfg.freq_offset_ip = "";
    cfg.freq_offset_port = 12347;
    cfg.udp_output_ip = "";
    cfg.software_sync = true; // Enable software sync

    // Parse command line arguments
    std::string default_ip = "127.0.0.1";
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("default-ip", po::value<std::string>(&default_ip)->default_value("127.0.0.1"), "Default IP for all services")
        ("args", po::value<std::string>(&cfg.device_args)->default_value("num_recv_frames=512, num_send_frames=512, send_frame_size=11520, recv_frame_size=11520"), "USRP device arguments")
        ("fft-size", po::value<size_t>(&cfg.fft_size)->default_value(1024), "FFT size")
        ("cp-length", po::value<size_t>(&cfg.cp_length)->default_value(128), "CP length")
        ("center-freq", po::value<double>(&cfg.center_freq)->default_value(2.4e9), "Center frequency")
        ("sample-rate", po::value<double>(&cfg.sample_rate)->default_value(50e6), "Sample rate")
        ("bandwidth", po::value<double>(&cfg.bandwidth)->default_value(50e6), "Bandwidth")
        ("rx-gain", po::value<double>(&cfg.rx_gain)->default_value(60), "RX gain")
        ("rx-channel", po::value<size_t>(&cfg.rx_channel)->default_value(0), "RX channel")
        ("sync-pos", po::value<size_t>(&cfg.sync_pos)->default_value(1), "Sync position")
        ("sensing-ip", po::value<std::string>(&cfg.bi_sensing_ip), "Sensing data IP")
        ("sensing-port", po::value<int>(&cfg.bi_sensing_port)->default_value(8889), "Sensing data port")
        ("control-port", po::value<int>(&cfg.control_port)->default_value(9999), "Control command port")
        ("channel-ip", po::value<std::string>(&cfg.channel_ip), "Channel data IP")
        ("channel-port", po::value<int>(&cfg.channel_port)->default_value(12348), "Channel data port")
        ("pdf-ip", po::value<std::string>(&cfg.pdf_ip), "PDF data IP")
        ("pdf-port", po::value<int>(&cfg.pdf_port)->default_value(12349), "PDF data port")
        ("constellation-ip", po::value<std::string>(&cfg.constellation_ip), "Constellation data IP")
        ("constellation-port", po::value<int>(&cfg.constellation_port)->default_value(12346), "Constellation data port")
        ("freq-offset-ip", po::value<std::string>(&cfg.freq_offset_ip), "Frequency offset data IP")
        ("freq-offset-port", po::value<int>(&cfg.freq_offset_port)->default_value(12347), "Frequency offset data port")
        ("udp-output-ip", po::value<std::string>(&cfg.udp_output_ip), "UDP output IP for decoded payloads")
        ("udp-output-port", po::value<int>(&cfg.udp_output_port)->default_value(50001), "UDP output port for decoded payloads")
        ("zc-root", po::value<int>(&cfg.zc_root)->default_value(29), "ZC root sequence")
        ("num-symbols", po::value<size_t>(&cfg.num_symbols)->default_value(100), "Number of symbols per frame")
        ("sensing-symbol-num", po::value<size_t>(&cfg.sensing_symbol_num)->default_value(100), "Number of symbols for sensing")
        ("clock-source", po::value<std::string>(&cfg.clocksource)->default_value("external"), "Clock source (internal or external)")
        ("software-sync", po::value<bool>(&cfg.software_sync)->default_value(true), "Enable software synchronization")
        ("hardware-sync", po::value<bool>(&cfg.hardware_sync)->default_value(false), "Enable hardware synchronization")
        ("hardware-sync-tty", po::value<std::string>(&cfg.hardware_sync_tty)->default_value("/dev/ttyUSB0"), "Hardware sync TTY device")
        ("wire-format-rx", po::value<std::string>(&cfg.wire_format_rx)->default_value("sc16"), "RX wire format (sc8 or sc16)")
        ("profiling", po::value<bool>(&cfg.enable_profiling)->default_value(false), "Enable detailed profiling output")
        ("cpu-cores", po::value<std::string>(), "Comma-separated list of CPU cores to use (e.g., 0,1,2,3,4,5,6)");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Sync sample rate and bandwidth defaults
    if (vm.count("sample-rate") && !vm.count("bandwidth")) {
        cfg.bandwidth = cfg.sample_rate;
        std::cout << "Bandwidth not specified, using sample rate: " << cfg.bandwidth / 1e6 << " MHz" << std::endl;
    } else if (vm.count("bandwidth") && !vm.count("sample-rate")) {
        cfg.sample_rate = cfg.bandwidth;
        std::cout << "Sample rate not specified, using bandwidth: " << cfg.sample_rate / 1e6 << " MHz" << std::endl;
    }

    if (cfg.bi_sensing_ip.empty()) cfg.bi_sensing_ip = default_ip;
    if (cfg.channel_ip.empty()) cfg.channel_ip = default_ip;
    if (cfg.pdf_ip.empty()) cfg.pdf_ip = default_ip;
    if (cfg.constellation_ip.empty()) cfg.constellation_ip = default_ip;
    if (cfg.freq_offset_ip.empty()) cfg.freq_offset_ip = default_ip;
    if (cfg.udp_output_ip.empty()) cfg.udp_output_ip = default_ip;

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    if (vm.count("cpu-cores")) {
        std::string cores_str = vm["cpu-cores"].as<std::string>();
        cfg.available_cores.clear();
        std::stringstream ss(cores_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                cfg.available_cores.push_back(std::stoul(item));
            } catch (...) {
                std::cerr << "Invalid core index: " << item << std::endl;
            }
        }
        if (cfg.available_cores.empty()) {
            std::cerr << "No valid cores specified, using default." << std::endl;
            cfg.available_cores = {0, 1, 2, 3, 4, 5};
        }
    }

    if (cfg.hardware_sync) {
        cfg.software_sync = false; // Disable software sync when hardware sync is enabled
        std::cout << "Hardware sync enabled. Software sync will be disabled." << std::endl;
    }
    uhd::set_thread_priority_safe(1, true);
    // Use last available core for main thread
    if (!cfg.available_cores.empty()) {
        std::vector<size_t> cpu_list = {cfg.available_cores.back()};
        uhd::set_thread_affinity(cpu_list);
    }
    
    OFDMRxEngine receiver(cfg);
    receiver.start();
    
    while (!stop_signal.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    receiver.stop();
    return 0;
}