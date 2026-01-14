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
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <deque>
#include <OFDMDemodulatorCore.hpp>
#include <SensingCore.hpp>

using namespace OpenISAC;
using namespace OpenISAC::Core;

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
          _data_llr_buffer(8),
          _accumulated_rx_symbols(),
          _accumulated_tx_symbols(),
          _blank_frame(cfg.samples_per_frame(), 0.0f),
          channel_sender_(16, [this](const std::vector<std::complex<float>, AlignedAlloc>& d){ if(channel_udp_) channel_udp_->send(reinterpret_cast<const char*>(d.data()), d.size()*sizeof(std::complex<float>)); }, std::chrono::milliseconds(10)),
          pdf_sender_(16, [this](const std::vector<std::complex<float>, AlignedAlloc>& d){ if(pdf_udp_) pdf_udp_->send(reinterpret_cast<const char*>(d.data()), d.size()*sizeof(std::complex<float>)); }, std::chrono::milliseconds(10)),
          constellation_sender_(16, [this](const std::vector<std::complex<float>, AlignedAlloc>& d){ if(constellation_udp_) constellation_udp_->send(reinterpret_cast<const char*>(d.data()), d.size()*sizeof(std::complex<float>)); }, std::chrono::milliseconds(10)),
          freq_offset_udp_(new UdpSender(cfg.mono_sensing_ip, cfg.mono_sensing_port + 3)),
          _control_handler(9998),
          // Initialize object pools
          _frame_pool(32, [&cfg]() {
              return AlignedVector(cfg.samples_per_frame());
          }),
          _rx_frame_pool(32, [&cfg]() {
               return AlignedVector(cfg.samples_per_frame());
          }),
          _llr_pool(32, [&cfg]() {
              // Estimate LLR buffer size: Symbols * DataSC * 2
              size_t data_sc = cfg.fft_size - cfg.pilot_positions.size();
              return AlignedFloatVector(cfg.num_symbols * data_sc * 2); 
          }),
          _sensing_frame_pool(16, []() {
               return SensingFrame();
          }),
          _data_llr_pool(16, []() {
              return AlignedFloatVector();
          })
    {
        // Initialize Cores
        _init_cores();
        
        // Initialize UDP senders
        // (Moved from init_udp)
        channel_udp_ = std::make_unique<UdpSender>(cfg_.channel_ip, cfg_.channel_port);
        pdf_udp_ = std::make_unique<UdpSender>(cfg_.pdf_ip, cfg_.pdf_port);
        constellation_udp_ = std::make_unique<UdpSender>(cfg_.constellation_ip, cfg_.constellation_port);
        // freq_offset_udp_ is already initialized in initializer list

        // Initialize USRP
        init_usrp();

        _actual_subcarrier_indices.resize(cfg_.fft_size);
        for(size_t i=0; i<cfg_.fft_size; ++i) {
             if (i < cfg_.fft_size/2) _actual_subcarrier_indices[i] = (int)i;
             else _actual_subcarrier_indices[i] = (int)i - (int)cfg_.fft_size;
        }

        const float freq_resolution = cfg_.sample_rate / cfg_.fft_size;
        _subcarrier_phases_unit_delay.resize(cfg_.fft_size);
        for(size_t i=0; i<cfg_.fft_size; ++i) {
             _subcarrier_phases_unit_delay[i] = -2.0f * M_PI * _actual_subcarrier_indices[i] * freq_resolution;
        }
        
        if (cfg.hardware_sync) {
             try {
                 _hw_sync = std::make_unique<HardwareSyncController>(cfg_.hardware_sync_tty);
             } catch(...) {
                 std::cerr << "Failed to init hardware sync" << std::endl;
             }
        }
        
        // Initialize data processing
        init_data_processing();
    }

    ~OFDMRxEngine() {
        stop();
        
        // Save wisdom before destroying plans
        // FFTW plans are now managed by cores, no direct destruction here.
        
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

    
    std::unique_ptr<FIRFilter> freq_offset_filter_;
    uhd::tune_result_t current_rx_tune_;
    bool tune_initialized_ = false;
    AlignedFloatVector sync_real_;
    AlignedFloatVector sync_imag_;
    std::atomic<bool> running_{false};
    std::thread rx_thread_, process_thread_;

    using AlignedAlloc = AlignedAllocator<std::complex<float>, 64>;



    void init_filter() {
        const std::vector<float> filter_coeffs = {
            0.25f, 0.25f, 0.25f, 0.25f, 0.25f
        };
        
        freq_offset_filter_ = std::make_unique<FIRFilter>(filter_coeffs);
        freq_offset_filter_->warm_up(0.0f, filter_coeffs.size() * 2);
    }

    void _init_cores() {
        // Demodulator Core
        Core::OFDMDemodulatorCore::Params demod_params;
        demod_params.fft_size = cfg_.fft_size;
        demod_params.cp_length = cfg_.cp_length;
        demod_params.num_symbols = cfg_.num_symbols;
        demod_params.sync_pos = cfg_.sync_pos;
        demod_params.zc_root = cfg_.zc_root;
        demod_params.sample_rate = cfg_.sample_rate;
        demod_params.center_freq = cfg_.center_freq;
        demod_params.pilot_positions = cfg_.pilot_positions;
        demod_params.sync_samples = cfg_.sync_samples();
        _demod_core = std::make_unique<Core::OFDMDemodulatorCore>(demod_params);

        // Sensing Core
        Core::SensingCore::Params sensing_params;
        sensing_params.fft_size = cfg_.fft_size;
        sensing_params.cp_length = cfg_.cp_length;
        sensing_params.sensing_symbol_num = cfg_.sensing_symbol_num;
        sensing_params.range_fft_size = cfg_.range_fft_size;
        sensing_params.doppler_fft_size = cfg_.doppler_fft_size;
        sensing_params.frame_count_for_process = cfg_.sensing_symbol_num; // Assuming 1 frame unit? No, frame_count is buffer depth
        // Wait, SensingCore accumulates symbols or frames.
        // process takes vector<rx_symbols>.
        // Check SensingCore Params logic.
        // It processes a block of `sensing_symbol_num` symbols.
        // So params seem correct.
        _sensing_core = std::make_unique<Core::SensingCore>(sensing_params);
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

    void init_data_processing() {
        // Only if additional initialization is needed
    }
    
    void adjust_rx_freq(double offset, bool abs_set = false) {
        if (abs_set) {
            current_rx_tune_.actual_dsp_freq = offset;
        } else {
            current_rx_tune_.actual_dsp_freq += offset;
        }
        
        uhd::tune_request_t tune_req(current_rx_tune_.actual_rf_freq, current_rx_tune_.actual_dsp_freq);
        tune_req.args = uhd::device_addr_t("mode_n=integer");
        usrp_->set_rx_freq(tune_req);
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
        // Acquire pre-allocated sync data buffer from pool
        AlignedVector sync_data = _sync_data_pool.acquire();
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
        // Acquire pre-allocated RX frame from pool
        RxFrame frame = _rx_frame_pool.acquire();
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
        // Acquire pre-allocated RX frame from pool
        RxFrame frame = _rx_frame_pool.acquire();
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
                // Return sync data to pool for reuse
                _sync_data_pool.release(std::move(sync_data));
            } else {
                RxFrame frame = wait_for_frame();
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
     * @brief Process Synchronization Data.
     */
    void process_sync_data(const AlignedVector& sync_data) {
        using ProfileClock = std::chrono::high_resolution_clock;
        static double prof_sync_total = 0.0;
        static int prof_sync_count = 0;
        constexpr int PROF_REPORT_INTERVAL = 43; 

        auto prof_start = ProfileClock::now();

        auto result = _demod_core->find_sync(sync_data);
        
        prof_sync_total += std::chrono::duration<double, std::micro>(ProfileClock::now() - prof_start).count();
        prof_sync_count++;

        if (prof_sync_count >= PROF_REPORT_INTERVAL && cfg_.should_profile("sync")) {
             std::cout << "[Sync] Search Avg Time: " << prof_sync_total / prof_sync_count << " us" << std::endl;
             prof_sync_total = 0.0;
             prof_sync_count = 0;
        }

        if (result.found) {
             std::cout << "Sync Found! Offset: " << result.offset << ", Correlation: " << result.correlation << std::endl;
             if (cfg_.software_sync) {
                 adjust_rx_freq(-result.cfo, false);
             }
             // Transition to Alignment state
             {
                 std::lock_guard<std::mutex> lock(sync_queue_mutex_);
                 sync_queue_.clear(); 
             }
             _delay_adjustment_count = 0;
             state_ = RxState::ALIGNMENT;
        }
    }

    // @brief Main OFDM Frame Processing Pipeline.
    // 
    // Steps:
    // 1. FFT: Convert time-domain samples to frequency domain.
    // 2. Channel Estimation: Use pilot symbols to estimate channel response.
    // 3. SFO/CFO Estimation: Refine offset estimates.
    // 4. Equalization: Compensate for channel effects (Zero-Forcing).
    // 5. Sensing: Extract Micro-Doppler signature and valid range/doppler data.
    // 6. LLR Calculation: Compute Log-Likelihood Ratios for soft decoding.
    void process_ofdm_frame(const RxFrame& frame) {
        // ============== Profiling variables ==============
        using ProfileClock = std::chrono::high_resolution_clock;
        static double prof_demod_core_total = 0.0;
        static double prof_loop_maint_total = 0.0;
        static double prof_sensing_queue_total = 0.0;
        static double prof_udp_send_total = 0.0;
        static double prof_llr_total = 0.0;
        static int prof_frame_count = 0;
        constexpr int PROF_REPORT_INTERVAL = 434;
        
        auto prof_step_start = ProfileClock::now();
        auto prof_step_end = prof_step_start;
        // =================================================
        
        // 1. Core Demodulation
        prof_step_start = ProfileClock::now();
        
        Core::OFDMDemodulatorCore::DemodResult result;
        // Pre-allocate result vectors if needed or let core resize (core resizes)
        
        // Process Frame
        // Note: frame.frame_data is Time Domain Frame
        _demod_core->process_frame(frame.frame_data, result);
        
        prof_step_end = ProfileClock::now();
        prof_demod_core_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        // 2. Loop Maintenance (SFO/Delay Tracking) & Sensing Data Prep
        prof_step_start = ProfileClock::now();
        
        // Use result.channel_est for delay profile
        AlignedVector delay_spectrum(cfg_.fft_size);
        _demod_core->compute_delay_profile(result.channel_est, delay_spectrum);
        
        // Find Peak in delay spectrum
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
        
        int adjusted_index = static_cast<int>(max_index);
        const int half_fft = cfg_.fft_size / 2;
        if (adjusted_index >= half_fft) adjusted_index -= cfg_.fft_size;

        float fractional_delay = estimateFractionalDelay(delay_spectrum, max_index);
        if (std::isnan(fractional_delay)) fractional_delay = 0.0f;
        float delay_offset_reading = adjusted_index + fractional_delay;

        sfo_estimator.update(delay_offset_reading, frame.Alignment);
        auto _sfo_per_frame = sfo_estimator.get_sfo_per_frame();
        auto delay_offset = sfo_estimator.get_sensing_delay_offset();
        
        // Send freq offset / delay packet
        size_t packet_size = sizeof(delay_offset) + sizeof(delay_offset_reading) + 4;
        std::vector<uint8_t> freq_packet(packet_size);
        memcpy(freq_packet.data(), &delay_offset, sizeof(float));
        memcpy(freq_packet.data() + sizeof(delay_offset), &delay_offset_reading, sizeof(float));
        freq_packet[sizeof(float)*2] = 0x00;
        freq_packet[sizeof(float)*2+1] = 0x00;
        freq_packet[sizeof(float)*2+2] = 0x80;
        freq_packet[sizeof(float)*2+3] = 0x7f;
        freq_offset_udp_->send(freq_packet.data(), packet_size);

        // Sync State Maintenance
        if((max_mag/average_mag < 20.0f || (abs(adjusted_index) > cfg_.delay_adjust_step+5))&&(cfg_.software_sync || cfg_.hardware_sync)) {
            _reset_count++;
            if (_reset_count >= 217) {
                _reset_count = 0;
                std::cout << "No valid delay found, resetting state." << std::endl;
                adjust_rx_freq(0.0, true);
                sfo_estimator.reset();
                if (cfg_.hardware_sync && _hw_sync) _hw_sync->reset_frequency_control();
                {
                    std::unique_lock<std::mutex> lock(sync_queue_mutex_);
                    sync_queue_.clear();
                }
                state_ = RxState::SYNC_SEARCH;
                return;
            }
        } else {
            _reset_count = 0;
        }
        
        if(_sync_in_progress && frame.Alignment != 0) _sync_in_progress = false;
        
        if(abs(adjusted_index) >= cfg_.delay_adjust_step && abs(adjusted_index) < cfg_.cp_length  && ( cfg_.software_sync || cfg_.hardware_sync )&& !_sync_in_progress) {   
            if (_delay_adjustment_count++ >= 1) {
                _delay_adjustment_count = 0;
                discard_samples_.store(adjusted_index);
                state_ = RxState::ALIGNMENT; 
                _sync_in_progress = true;
            }
            if (adjusted_index * _last_adjusted_index < 0) _delay_adjustment_count = 0;
        } else {
            _delay_adjustment_count = 0;
        }
        _last_adjusted_index = adjusted_index;
        
        // Frequency Offset Correction Logic (using result.cfo_est)
        // Original code used alpha (CFO) and beta (SFO)
        {
            float detected_freq_offset = result.cfo_est / (2 * M_PI * ((cfg_.fft_size + cfg_.cp_length) / cfg_.sample_rate));
             _freq_offset_sum += detected_freq_offset;
            _freq_offset_count++;
             if (_freq_offset_count>=434) {
                 _avg_freq_offset = _freq_offset_sum / _freq_offset_count;
                 _freq_offset_sum = 0.0f;
                 _freq_offset_count = 0;
                 if(abs(_avg_freq_offset) > 2.0f) {
                     if (cfg_.software_sync){
                         std::cout << "Adjusting RX frequency by: " << _avg_freq_offset << " Hz" << std::endl;
                         adjust_rx_freq(-_avg_freq_offset, false);
                     }
                 }
                 if (cfg_.hardware_sync && _hw_sync) {
                         double ppm = _avg_freq_offset / cfg_.center_freq * 1e6;
                         double ppm_adjusted = (abs(_avg_freq_offset) > 1.0) ? ppm * cfg_.ppm_adjust_factor * 10 : ppm * cfg_.ppm_adjust_factor;
                         std::cout << "Adjusting OCXO by: " << ppm << " ppm" << std::endl;
                         _hw_sync->set_frequency_control_ppm_relative(ppm_adjusted);
                 }
             }
        }

        prof_step_end = ProfileClock::now();
        prof_loop_maint_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        // 3. Sensing Data Queue
        prof_step_start = ProfileClock::now();
        SensingFrame sense_frame = _sensing_frame_pool.acquire();
        sense_frame.rx_symbols = result.rx_symbols; // Copy/Move rx symbols
        // We need tx_symbols for sensing accumulation?
        // Original code "remodulated" rx symbols to get estimated tx symbols.
        // It used simple hard decision on demodulated symbols.
        // I need to implement "remodulate" or use `OFDMModulatorCore` to generate expected symbols?
        // Original code used: if sync, ZC; if data, QPSK hard decision.
        sense_frame.tx_symbols.resize(cfg_.num_symbols);
        // We MUST populate sense_frame.tx_symbols.
        // Implementation of Remodulation (Hard Decision)
        // ...
        // Re-implementing simplified remodulation here as it's specific to "Blind" sensing (or Comm-based sensing)
        // Core doesn't have "Remodulate".
        // Using `result.equalized_symbols` would be better for hard decision than `rx_symbols`.
        
        static const AlignedVector zc = DSP::generate_zc_sequence(cfg_.fft_size, cfg_.zc_root);
        constexpr float ONE_OVER_SQRT2 = 0.70710678f;
        
        for(size_t i=0; i<cfg_.num_symbols; ++i) {
             if (i == cfg_.sync_pos) {
                 sense_frame.tx_symbols[i] = zc;
             } else {
                 sense_frame.tx_symbols[i].resize(cfg_.fft_size);
                 // Need equalized symbols for decision
                 const auto& eq_sym = result.equalized_symbols[i];
                 auto* tx_ptr = sense_frame.tx_symbols[i].data();
                 const auto* eq_ptr = eq_sym.data();
                 const auto* zc_ptr = zc.data();
                 
                  #pragma omp simd
                 for(size_t k=0; k<cfg_.fft_size; ++k) {
                     // Branchless QPSK Hard Decision
                     float re = eq_ptr[k].real();
                     float im = eq_ptr[k].imag();
                     tx_ptr[k] = std::complex<float>(
                         std::copysign(ONE_OVER_SQRT2, re),
                         std::copysign(ONE_OVER_SQRT2, im)
                     );
                 }
                 // Restore Pilots
                 for(auto p : cfg_.pilot_positions) if(p < cfg_.fft_size) tx_ptr[p] = zc_ptr[p];
             }
        }

        sense_frame.CFO = result.cfo_est;
        sense_frame.SFO = (_sfo_per_frame != 0.0f) ? 
            -_sfo_per_frame * (2 * M_PI) / (cfg_.fft_size * cfg_.num_symbols) : 
            result.sfo_est;
        sense_frame.delay_offset = delay_offset + _user_delay_offset;

        {
            std::lock_guard<std::mutex> lock(sensing_mutex_);
            if (!sensing_queue_.full()) {
                sensing_queue_.push_back(std::move(sense_frame));
            } else {
                _sensing_frame_pool.release(std::move(sense_frame));
            }
        }
        sensing_cv_.notify_one();
        prof_step_end = ProfileClock::now();
        prof_sensing_queue_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        
        // 4. UDP Sending
        prof_step_start = ProfileClock::now();
        channel_sender_.add_data(std::move(result.channel_est));
        pdf_sender_.add_data(std::move(delay_spectrum));
        if(!result.equalized_symbols.empty()) {
             constellation_sender_.add_data(result.equalized_symbols.back()); // Send last symbol
        }
        prof_step_end = ProfileClock::now();
        prof_udp_send_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        // 5. LLR Queue
        prof_step_start = ProfileClock::now();
        {
            std::unique_lock<std::mutex> lock(_data_llr_mutex);
            if (!_data_llr_buffer.full()) {
                _data_llr_buffer.push_back(std::move(result.llr));
                _data_llr_cv.notify_one();
            }
        }
        prof_step_end = ProfileClock::now();
        prof_llr_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        // Report
        prof_frame_count++;
        if (prof_frame_count >= PROF_REPORT_INTERVAL && cfg_.should_profile("demodulation")) {
            std::cout << "\n========== process_ofdm_frame Profiling (avg per frame, us) ==========" << std::endl;
            std::cout << "Core Demodulation:    " << prof_demod_core_total / prof_frame_count << " us" << std::endl;
            std::cout << "Loop Maintenance:     " << prof_loop_maint_total / prof_frame_count << " us" << std::endl;
            std::cout << "Sensing Queue Prep:   " << prof_sensing_queue_total / prof_frame_count << " us" << std::endl;
            std::cout << "UDP Send:             " << prof_udp_send_total / prof_frame_count << " us" << std::endl;
            std::cout << "LLR Queue:            " << prof_llr_total / prof_frame_count << " us" << std::endl;
            double total = prof_demod_core_total + prof_loop_maint_total + prof_sensing_queue_total + prof_udp_send_total + prof_llr_total;
            std::cout << "TOTAL:                " << total / prof_frame_count << " us" << std::endl;
            std::cout << "======================================================================\n" << std::endl;
            prof_demod_core_total = 0.0;
            prof_loop_maint_total = 0.0;
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
                    #pragma omp simd simdlen(16)
                    for (size_t j = 0; j < rx_symbol.size(); ++j) {
                        auto phase_diff_SFO = beta * _actual_subcarrier_indices[j] * relative_symbol_index;
                        float phase_diff_delay = _subcarrier_phases_unit_delay[j] * delay_offset;
                        auto phase_diff_total = phase_diff_delay + phase_diff_SFO + phase_diff_CFO;
                        auto phase_diff = std::polar(1.0f, - phase_diff_total);
                        rx_symbol[j] = rx_symbol[j] * phase_diff;
                    }
                    // Add current symbol to accumulation buffer
                    _accumulated_rx_symbols.push_back(rx_symbol);
                    _accumulated_tx_symbols.push_back(frame.tx_symbols[symbol_idx]);
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
                _sensing_frame_pool.release(std::move(frame));
            }
        }
    }


    // Sensing processing function
    // Sensing processing function
    void _sensing_process(const SensingFrame& frame) {
        // Send heartbeat for NAT traversal (once per second)
        static auto next_hb_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (now >= next_hb_time) {
            _control_handler.send_heartbeat(cfg_.bi_sensing_ip, cfg_.bi_sensing_port);
            next_hb_time = now + std::chrono::seconds(1);
        }

        // Profiling
        using ProfileClock = std::chrono::high_resolution_clock;
        static double prof_sensing_core_total = 0.0;
        static int prof_sensing_count = 0;
        constexpr int PROF_REPORT_INTERVAL = 434;

        if (std::chrono::steady_clock::now() >= _next_send_time && !skip_sensing_fft.load()) {
            
            auto prof_start = ProfileClock::now();
            
            // Core Process (FFT, Channel Est, MTI, Windowing, IFFT/FFT 2D)
            const auto& result_buffer = _sensing_core->process(frame.rx_symbols, frame.tx_symbols);
            
            prof_sensing_core_total += std::chrono::duration<double, std::micro>(ProfileClock::now() - prof_start).count();
            prof_sensing_count++;
            
            if (prof_sensing_count >= PROF_REPORT_INTERVAL && cfg_.should_profile("sensing")) {
                std::cout << "[Sensing] Core Process: " << prof_sensing_core_total / prof_sensing_count << " us" << std::endl;
                prof_sensing_core_total = 0.0;
                prof_sensing_count = 0;
            }

            // Send
            _sensing_sender.push_data(result_buffer);
            
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
            // Return LLR buffer to pool for reuse
            _llr_pool.release(std::move(frame_llr));
        }
    }
    // Core members
    std::unique_ptr<Core::OFDMDemodulatorCore> _demod_core;
    std::unique_ptr<Core::SensingCore> _sensing_core;
    
    // USRP members
    uhd::usrp::multi_usrp::sptr usrp_;
    uhd::rx_streamer::sptr rx_stream_;
    uhd::tune_result_t current_rx_tune_;
    
    // External utilities
    SFOEstimator sfo_estimator;
    LDPCCodec _ldpc_decoder;
    Scrambler _descrambler;
    
    // Senders
    std::unique_ptr<UdpSender> channel_udp_;
    std::unique_ptr<UdpSender> pdf_udp_;
    std::unique_ptr<UdpSender> constellation_udp_;
    std::unique_ptr<UdpSender> freq_offset_udp_;
    std::unique_ptr<UdpSender> _udp_output_sender;

    // Buffered Data Senders
    DataSender<std::complex<float>, AlignedAlloc> channel_sender_;
    DataSender<std::complex<float>, AlignedAlloc> pdf_sender_;
    DataSender<std::complex<float>, AlignedAlloc> constellation_sender_;
    
    // Sync controller
    std::unique_ptr<HardwareSyncController> _hw_sync;
    
    // State
    std::atomic<bool> _sync_in_progress{false};
    int _delay_adjustment_count{0};
    int _last_adjusted_index{0};
    int _reset_count{0};
    double _freq_offset_sum{0.0};
    int _freq_offset_count{0};
    double _avg_freq_offset{0.0};
    
    // Pre-calculated tables
    std::vector<int> _actual_subcarrier_indices;
    std::vector<float> _subcarrier_phases_unit_delay;
    
    // Control variables
    std::atomic<bool> skip_sensing_fft{false};
    std::atomic<bool> enable_mti{true};
    std::atomic<bool> sensing_running_{false};
    std::atomic<size_t> sensing_sybmol_stride{20};
    size_t shadow_sensing_symbol_stride{20};

    // Thread control
    std::atomic<bool> running_{false};
    std::thread rx_thread_;
    std::thread processing_thread_;
    std::thread sensing_thread_;
    std::atomic<bool> _bit_processing_running{false};
    std::thread _bit_processing_thread;
    
    // Buffers and Queues
    boost::circular_buffer<AlignedFloatVector> _data_llr_buffer;
    AlignedVector _mti_filter; // Likely obsolete if Core handles it? No, keep if used locally in old logic which is being replaced.
                               // Wait, I am replacing _sensing_process, so _mti_filter member is likely obsolete.
                               // But let's check. Yes, SensingCore handles MTI.
    std::vector<AlignedVector> _accumulated_rx_symbols;
    std::vector<AlignedVector> _accumulated_tx_symbols; // Frequency domain, but accumulation logic is in thread.

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    boost::circular_buffer<RxFrame> frame_queue_{16};

    std::mutex sync_queue_mutex_;
    std::condition_variable sync_queue_cv_;
    boost::circular_buffer<AlignedVector> sync_queue_{16};
    
    std::mutex sensing_mutex_;
    std::condition_variable sensing_cv_;
    boost::circular_buffer<SensingFrame> sensing_queue_{16};
    
    std::mutex _data_llr_mutex;
    std::condition_variable _data_llr_cv;
    
    // Pools
    ObjectPool<AlignedVector> _frame_pool;
    ObjectPool<AlignedVector> _rx_frame_pool;
    ObjectPool<AlignedVector> _sync_data_pool{32, [this](){ return AlignedVector(cfg_.sync_samples()); }};
    ObjectPool<AlignedFloatVector> _llr_pool;
    ObjectPool<SensingFrame> _sensing_frame_pool;
    ObjectPool<AlignedFloatVector> _data_llr_pool;

    ControlCommandHandler _control_handler;
    SensingDataSender _sensing_sender{cfg_.bi_sensing_ip, cfg_.bi_sensing_port, cfg_.fft_size * cfg_.num_symbols};
    
    AlignedVector _blank_frame;
    AlignedVector _channel_response_buffer; // Used for sensing sending
    std::chrono::steady_clock::time_point _next_send_time;
};

std::atomic<bool> stop_signal(false);

void signal_handler(int) {
    stop_signal.store(true);
}

int UHD_SAFE_MAIN(int argc, char* argv[]) {
    std::signal(SIGINT, &signal_handler);
    
    // Default config file path
    const std::string default_config_file = "Demodulator.yaml";
    
    // Helper function: Save config to YAML file
    auto save_config_to_yaml = [](const Config& cfg, const std::string& filepath) {
        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "fft_size" << YAML::Value << cfg.fft_size;
        out << YAML::Key << "cp_length" << YAML::Value << cfg.cp_length;
        out << YAML::Key << "sync_pos" << YAML::Value << cfg.sync_pos;
        out << YAML::Key << "sample_rate" << YAML::Value << cfg.sample_rate;
        out << YAML::Key << "bandwidth" << YAML::Value << cfg.bandwidth;
        out << YAML::Key << "center_freq" << YAML::Value << cfg.center_freq;
        out << YAML::Key << "rx_gain" << YAML::Value << cfg.rx_gain;
        out << YAML::Key << "rx_channel" << YAML::Value << cfg.rx_channel;
        out << YAML::Key << "zc_root" << YAML::Value << cfg.zc_root;
        out << YAML::Key << "num_symbols" << YAML::Value << cfg.num_symbols;
        out << YAML::Key << "sensing_symbol_num" << YAML::Value << cfg.sensing_symbol_num;
        out << YAML::Key << "range_fft_size" << YAML::Value << cfg.range_fft_size;
        out << YAML::Key << "doppler_fft_size" << YAML::Value << cfg.doppler_fft_size;
        out << YAML::Key << "device_args" << YAML::Value << cfg.device_args;
        out << YAML::Key << "clock_source" << YAML::Value << cfg.clocksource;
        out << YAML::Key << "wire_format_rx" << YAML::Value << cfg.wire_format_rx;
        out << YAML::Key << "sensing_ip" << YAML::Value << cfg.bi_sensing_ip;
        out << YAML::Key << "sensing_port" << YAML::Value << cfg.bi_sensing_port;
        out << YAML::Key << "control_port" << YAML::Value << cfg.control_port;
        out << YAML::Key << "channel_ip" << YAML::Value << cfg.channel_ip;
        out << YAML::Key << "channel_port" << YAML::Value << cfg.channel_port;
        out << YAML::Key << "pdf_ip" << YAML::Value << cfg.pdf_ip;
        out << YAML::Key << "pdf_port" << YAML::Value << cfg.pdf_port;
        out << YAML::Key << "constellation_ip" << YAML::Value << cfg.constellation_ip;
        out << YAML::Key << "constellation_port" << YAML::Value << cfg.constellation_port;
        out << YAML::Key << "freq_offset_ip" << YAML::Value << cfg.freq_offset_ip;
        out << YAML::Key << "freq_offset_port" << YAML::Value << cfg.freq_offset_port;
        out << YAML::Key << "udp_output_ip" << YAML::Value << cfg.udp_output_ip;
        out << YAML::Key << "udp_output_port" << YAML::Value << cfg.udp_output_port;
        out << YAML::Key << "software_sync" << YAML::Value << cfg.software_sync;
        out << YAML::Key << "hardware_sync" << YAML::Value << cfg.hardware_sync;
        out << YAML::Key << "hardware_sync_tty" << YAML::Value << cfg.hardware_sync_tty;
        out << YAML::Key << "profiling_modules" << YAML::Value << cfg.profiling_modules;
        out << YAML::Key << "default_ip" << YAML::Value << cfg.default_ip;
        out << YAML::Key << "pilot_positions" << YAML::Value << YAML::Flow << cfg.pilot_positions;
        out << YAML::Key << "cpu_cores" << YAML::Value << YAML::Flow << cfg.available_cores;
        out << YAML::EndMap;
        
        std::ofstream fout(filepath);
        if (!fout) {
            std::cerr << "Error: Cannot write to config file: " << filepath << std::endl;
            return false;
        }
        fout << out.c_str();
        fout.close();
        return true;
    };
    
    // Helper function: Load config from YAML file
    auto load_config_from_yaml = [](Config& cfg, const std::string& filepath) {
        if (!fs::exists(filepath)) {
            return false;
        }
        try {
            YAML::Node config = YAML::LoadFile(filepath);
            if (config["fft_size"]) cfg.fft_size = config["fft_size"].as<size_t>();
            if (config["cp_length"]) cfg.cp_length = config["cp_length"].as<size_t>();
            if (config["sync_pos"]) cfg.sync_pos = config["sync_pos"].as<size_t>();
            if (config["sample_rate"]) cfg.sample_rate = config["sample_rate"].as<double>();
            if (config["bandwidth"]) cfg.bandwidth = config["bandwidth"].as<double>();
            if (config["center_freq"]) cfg.center_freq = config["center_freq"].as<double>();
            if (config["rx_gain"]) cfg.rx_gain = config["rx_gain"].as<double>();
            if (config["rx_channel"]) cfg.rx_channel = config["rx_channel"].as<size_t>();
            if (config["zc_root"]) cfg.zc_root = config["zc_root"].as<int>();
            if (config["num_symbols"]) cfg.num_symbols = config["num_symbols"].as<size_t>();
            if (config["sensing_symbol_num"]) cfg.sensing_symbol_num = config["sensing_symbol_num"].as<size_t>();
            if (config["range_fft_size"]) cfg.range_fft_size = config["range_fft_size"].as<size_t>();
            if (config["doppler_fft_size"]) cfg.doppler_fft_size = config["doppler_fft_size"].as<size_t>();
            if (config["device_args"]) cfg.device_args = config["device_args"].as<std::string>();
            if (config["clock_source"]) cfg.clocksource = config["clock_source"].as<std::string>();
            if (config["wire_format_rx"]) cfg.wire_format_rx = config["wire_format_rx"].as<std::string>();
            if (config["sensing_ip"]) cfg.bi_sensing_ip = config["sensing_ip"].as<std::string>();
            if (config["sensing_port"]) cfg.bi_sensing_port = config["sensing_port"].as<int>();
            if (config["control_port"]) cfg.control_port = config["control_port"].as<int>();
            if (config["channel_ip"]) cfg.channel_ip = config["channel_ip"].as<std::string>();
            if (config["channel_port"]) cfg.channel_port = config["channel_port"].as<int>();
            if (config["pdf_ip"]) cfg.pdf_ip = config["pdf_ip"].as<std::string>();
            if (config["pdf_port"]) cfg.pdf_port = config["pdf_port"].as<int>();
            if (config["constellation_ip"]) cfg.constellation_ip = config["constellation_ip"].as<std::string>();
            if (config["constellation_port"]) cfg.constellation_port = config["constellation_port"].as<int>();
            if (config["freq_offset_ip"]) cfg.freq_offset_ip = config["freq_offset_ip"].as<std::string>();
            if (config["freq_offset_port"]) cfg.freq_offset_port = config["freq_offset_port"].as<int>();
            if (config["udp_output_ip"]) cfg.udp_output_ip = config["udp_output_ip"].as<std::string>();
            if (config["udp_output_port"]) cfg.udp_output_port = config["udp_output_port"].as<int>();
            if (config["software_sync"]) cfg.software_sync = config["software_sync"].as<bool>();
            if (config["hardware_sync"]) cfg.hardware_sync = config["hardware_sync"].as<bool>();
            if (config["hardware_sync_tty"]) cfg.hardware_sync_tty = config["hardware_sync_tty"].as<std::string>();
            if (config["profiling_modules"]) cfg.profiling_modules = config["profiling_modules"].as<std::string>();
            if (config["default_ip"]) cfg.default_ip = config["default_ip"].as<std::string>();
            if (config["pilot_positions"]) cfg.pilot_positions = config["pilot_positions"].as<std::vector<size_t>>();
            if (config["cpu_cores"]) cfg.available_cores = config["cpu_cores"].as<std::vector<size_t>>();
            return true;
        } catch (const YAML::Exception& e) {
            std::cerr << "Error parsing YAML config: " << e.what() << std::endl;
            return false;
        }
    };
    
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
    cfg.bandwidth = 50e6;
    cfg.rx_gain = 50.0;
    cfg.zc_root = 29;
    cfg.device_args = "";
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
    cfg.software_sync = true;

    // Parse command line arguments
    std::string config_file = default_config_file;
    std::string save_config = "";
    
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("config,c", po::value<std::string>(&config_file)->default_value(default_config_file), "Config file path (default: Demodulator.yaml)")
        ("save-config,s", po::value<std::string>(&save_config)->implicit_value(""), "Save current config to file and exit (optionally specify filename)")
        ("default-ip", po::value<std::string>(&cfg.default_ip), "Default IP for all services (default: 127.0.0.1)")
        ("args", po::value<std::string>(&cfg.device_args), "USRP device arguments")
        ("fft-size", po::value<size_t>(&cfg.fft_size), "FFT size (default: 1024)")
        ("cp-length", po::value<size_t>(&cfg.cp_length), "CP length (default: 128)")
        ("center-freq", po::value<double>(&cfg.center_freq), "Center frequency (default: 2.4e9)")
        ("sample-rate", po::value<double>(&cfg.sample_rate), "Sample rate (default: 50e6)")
        ("bandwidth", po::value<double>(&cfg.bandwidth), "Bandwidth (default: 50e6)")
        ("rx-gain", po::value<double>(&cfg.rx_gain), "RX gain (default: 60)")
        ("rx-channel", po::value<size_t>(&cfg.rx_channel), "RX channel (default: 0)")
        ("sync-pos", po::value<size_t>(&cfg.sync_pos), "Sync position (default: 1)")
        ("sensing-ip", po::value<std::string>(&cfg.bi_sensing_ip), "Sensing data IP")
        ("sensing-port", po::value<int>(&cfg.bi_sensing_port), "Sensing data port (default: 8889)")
        ("control-port", po::value<int>(&cfg.control_port), "Control command port (default: 9999)")
        ("channel-ip", po::value<std::string>(&cfg.channel_ip), "Channel data IP")
        ("channel-port", po::value<int>(&cfg.channel_port), "Channel data port (default: 12348)")
        ("pdf-ip", po::value<std::string>(&cfg.pdf_ip), "PDF data IP")
        ("pdf-port", po::value<int>(&cfg.pdf_port), "PDF data port (default: 12349)")
        ("constellation-ip", po::value<std::string>(&cfg.constellation_ip), "Constellation data IP")
        ("constellation-port", po::value<int>(&cfg.constellation_port), "Constellation data port (default: 12346)")
        ("freq-offset-ip", po::value<std::string>(&cfg.freq_offset_ip), "Frequency offset data IP")
        ("freq-offset-port", po::value<int>(&cfg.freq_offset_port), "Frequency offset data port (default: 12347)")
        ("udp-output-ip", po::value<std::string>(&cfg.udp_output_ip), "UDP output IP for decoded payloads")
        ("udp-output-port", po::value<int>(&cfg.udp_output_port), "UDP output port (default: 50001)")
        ("zc-root", po::value<int>(&cfg.zc_root), "ZC root sequence (default: 29)")
        ("num-symbols", po::value<size_t>(&cfg.num_symbols), "Number of symbols per frame (default: 100)")
        ("sensing-symbol-num", po::value<size_t>(&cfg.sensing_symbol_num), "Number of symbols for sensing (default: 100)")
        ("clock-source", po::value<std::string>(&cfg.clocksource), "Clock source (default: external)")
        ("software-sync", po::value<bool>(&cfg.software_sync), "Enable software synchronization (default: true)")
        ("hardware-sync", po::value<bool>(&cfg.hardware_sync), "Enable hardware synchronization (default: false)")
        ("hardware-sync-tty", po::value<std::string>(&cfg.hardware_sync_tty), "Hardware sync TTY device (default: /dev/ttyUSB0)")
        ("wire-format-rx", po::value<std::string>(&cfg.wire_format_rx), "RX wire format (default: sc16)")
        ("profiling-modules", po::value<std::string>(&cfg.profiling_modules), "Comma-separated modules to profile")
        ("cpu-cores", po::value<std::string>(), "Comma-separated list of CPU cores");
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    // Load config from YAML file (if exists), then CLI args override
    if (fs::exists(config_file)) {
        if (load_config_from_yaml(cfg, config_file)) {
            std::cout << "Loaded config from: " << config_file << std::endl;
        }
    } else if (config_file == default_config_file) {
        // Auto-create default config file with current defaults
        if (save_config_to_yaml(cfg, config_file)) {
            std::cout << "Config file '" << config_file << "' not found. Created with default values." << std::endl;
        }
    }
    
    // Re-parse CLI to override YAML values (only update options explicitly provided in CLI)
    vm.clear(); // Clear to only contain CLI-specified options
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Handle --save-config option
    if (vm.count("save-config")) {
        std::string output_file = config_file; // Default to config_file
        if (!save_config.empty()) {
            output_file = save_config; // Use custom filename if provided
        }
        if (save_config_to_yaml(cfg, output_file)) {
            std::cout << "Config saved to: " << output_file << std::endl;
        }
        return 0;
    }

    // Sync sample rate and bandwidth defaults
    if (vm.count("sample-rate") && !vm.count("bandwidth")) {
        cfg.bandwidth = cfg.sample_rate;
        std::cout << "Bandwidth not specified, using sample rate: " << cfg.bandwidth / 1e6 << " MHz" << std::endl;
    } else if (vm.count("bandwidth") && !vm.count("sample-rate")) {
        cfg.sample_rate = cfg.bandwidth;
        std::cout << "Sample rate not specified, using bandwidth: " << cfg.sample_rate / 1e6 << " MHz" << std::endl;
    }

    if (cfg.bi_sensing_ip.empty()) cfg.bi_sensing_ip = cfg.default_ip;
    if (cfg.channel_ip.empty()) cfg.channel_ip = cfg.default_ip;
    if (cfg.pdf_ip.empty()) cfg.pdf_ip = cfg.default_ip;
    if (cfg.constellation_ip.empty()) cfg.constellation_ip = cfg.default_ip;
    if (cfg.freq_offset_ip.empty()) cfg.freq_offset_ip = cfg.default_ip;
    if (cfg.udp_output_ip.empty()) cfg.udp_output_ip = cfg.default_ip;

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
        cfg.software_sync = false;
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