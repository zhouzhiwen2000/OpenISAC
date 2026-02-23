#include <uhd/utils/thread.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <uhd/types/tune_request.hpp>
#include <fftw3.h>
#include <iostream>
#include <vector>
#include <complex>
#include <atomic>
#include <csignal>
#include <random>
#include <omp.h>
#include <fstream>
#include <boost/circular_buffer.hpp>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <Common.hpp>
#include "OFDMCore.hpp"
#include <functional>
#include <unordered_map>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace po = boost::program_options;
namespace fs = std::filesystem;

// Type aliases are now defined in Common.hpp

/**
 * @brief OFDM ISAC (Integrated Sensing and Communication) Transmitter Engine.
 * 
 * Main class for OFDM Modulation and Sensing Transmission.
 * Implements a multi-threaded architecture:
 * - mod_thread: Generates OFDM symbols (IFFT) from data and pilots.
 * - tx_thread: Streams generated time-domain frames to USRP.
 * - rx_thread: Receives self-reflected signals (for sensing).
 * - sensing_thread: Processes received signals for further analysis.
 * - data_thread: Ingests user UDP data and performs LDPC encoding.
 */
class OFDMISACEngine {
public:
    OFDMISACEngine(const Config& cfg) 
      : _cfg(cfg),
        _gen(std::random_device{}()),        
        _dist(0, 3),                         
        _circular_buffer(cfg.tx_frame_buffer_size),
        _symbols_buffer(cfg.tx_symbols_buffer_size),
        _rx_frame_buffer(cfg.rx_frame_buffer_size),            
        _data_packet_buffer(32),                // Initialize data packet buffer
        _accumulated_rx_symbols(),
        _accumulated_tx_symbols(),
        _fft_in(cfg.fft_size),               
        _fft_out(cfg.fft_size),
        _demod_fft_in(cfg.fft_size),
        _demod_fft_out(cfg.fft_size),
        _blank_frame(cfg.samples_per_frame(), 0.0f),
        tx_data_file("tx_frame.bin", std::ios::binary),
        _sensing_sender(cfg.mono_sensing_ip, cfg.mono_sensing_port),
        _control_handler(9999),  // Initialize control handler
        // Initialize object pools for memory reuse
        _frame_pool(32, [&cfg]() {
            return AlignedVector(cfg.samples_per_frame());
        }),
        _symbols_pool(32, [&cfg]() {
            SymbolVector sv;
            sv.resize(cfg.num_symbols);
            for (size_t i = 0; i < cfg.num_symbols; ++i) {
                sv[i].resize(cfg.fft_size);
            }
            return sv;
        }),
        _rx_frame_pool(32, [&cfg]() {
            return AlignedVector(cfg.samples_per_frame());
        }),
        _data_packet_pool(64, []() {
            return AlignedIntVector();
        }),
        _sensing_core({cfg.fft_size, cfg.range_fft_size, cfg.doppler_fft_size, cfg.sensing_symbol_num})
    {

        // Initialize FFTW resources
        _init_fftw();
        
        // Prepare ZC sequence
        _prepare_zc_sequence();
        
        // Initialize USRP
        _init_usrp();
        
        // Pre-calculate symbol positions
        _precalc_positions();
        
        // Pre-generate data
        size_t total_data_samples = (cfg.num_symbols - 1) * (cfg.fft_size- cfg.pilot_positions.size());
        _pregen_data.resize(total_data_samples);
        for (auto& x : _pregen_data) {
            x = _dist(_gen);
        }
        // UDP data sender initialization
        _sensing_sender.start();

        // Register control command handler
        _register_commands();

        // Initialize Hamming windows
        _init_hamming_windows();

        _accumulated_rx_symbols.reserve(_cfg.sensing_symbol_num);
        _accumulated_tx_symbols.reserve(_cfg.sensing_symbol_num);
        
        // Initialize MTI filter

    }

    ~OFDMISACEngine() {
        stop();
        
        
        fftwf_destroy_plan(_ifft_plan);
        fftwf_destroy_plan(_demod_fft_plan);
        // Note: _sensing_core manages its own FFT plans and cleans them up in its destructor
    }

    void start() {
        _control_handler.start();
        _running.store(true);
        _mod_thread = std::thread(&OFDMISACEngine::_modulation_proc, this); 
        _rx_thread = std::thread(&OFDMISACEngine::_rx_proc, this);         
        _sensing_thread = std::thread(&OFDMISACEngine::_sensing_proc, this);
        _tx_thread = std::thread(&OFDMISACEngine::_tx_proc, this);
        _data_thread = std::thread(&OFDMISACEngine::_data_ingest_proc, this);

    }

    void stop() {
        _running.store(false);
        _control_handler.stop();
        {
            _buffer_not_full.notify_all();    
            _buffer_not_empty.notify_all();   
            _symbols_not_empty.notify_all();  
            _symbols_not_full.notify_all();   
            _rx_not_empty.notify_all();       
            _rx_not_full.notify_all();        
        }
        if (_mod_thread.joinable()) _mod_thread.join();
        if (_tx_thread.joinable()) _tx_thread.join();
        if (_rx_thread.joinable()) _rx_thread.join();
        if (_sensing_thread.joinable()) _sensing_thread.join();
        if (_data_thread.joinable()) _data_thread.join();

    }

private:
    static LDPCCodec::LDPCConfig make_ldpc_5041008_cfg() {
        LDPCCodec::LDPCConfig c;
        c.h_matrix_path = "../LDPC_504_1008.alist";
        c.g_matrix_path = "../LDPC_504_1008_G_fromH.alist";
        c.decoder_iterations = 6;
        c.n_frames = 16;
        c.enc_type = "LDPC_H";
        c.enc_g_method = "IDENTITY";
        c.dec_type = "BP_HORIZONTAL_LAYERED";
        c.dec_implem = "NMS";
        c.dec_simd = "INTER";
        c.use_custom_encoder = true;
        return c;
    }

    Config _cfg;
    uhd::usrp::multi_usrp::sptr _usrp;   
    uhd::tx_streamer::sptr _tx_stream;   
    uhd::rx_streamer::sptr _rx_stream;   
    
    // Random number generator
    std::mt19937 _gen;
    std::uniform_int_distribution<> _dist;
    
    // Ring buffer system
    boost::circular_buffer<AlignedVector> _circular_buffer; 
    boost::circular_buffer<SymbolVector> _symbols_buffer;   
    boost::circular_buffer<AlignedVector> _rx_frame_buffer; 
    boost::circular_buffer<AlignedIntVector> _data_packet_buffer; // Data packet buffer
    std::vector<AlignedVector> _accumulated_rx_symbols;   // Accumulated RX symbols (Time domain)
    std::vector<AlignedVector> _accumulated_tx_symbols;     // Accumulated TX symbols (Frequency domain)
    // Synchronization primitives
    std::mutex _buffer_mutex;
    std::mutex _symbols_mutex;
    std::mutex _rx_mutex;     
    std::mutex _data_mutex;                               // Data packet buffer mutex
    std::condition_variable _buffer_not_full;                 
    std::condition_variable _buffer_not_empty;                
    std::condition_variable _symbols_not_empty;               
    std::condition_variable _symbols_not_full;                
    std::condition_variable _rx_not_empty;                    
    std::condition_variable _rx_not_full;                     
    std::condition_variable _data_not_empty;              // Data packet buffer condition variable (not empty)
    std::condition_variable _data_not_full;               // Data packet buffer condition variable (not full)
    
    // FFTW resources
    AlignedVector _fft_in;               // IFFT input
    AlignedVector _fft_out;              // IFFT output
    fftwf_plan _ifft_plan;               // IFFT plan
    
    AlignedVector _demod_fft_in;         // Range FFT input
    AlignedVector _demod_fft_out;        // Range FFT output
    fftwf_plan _demod_fft_plan = nullptr; // Demodulation FFT plan
    // Note: Range IFFT and Doppler FFT plans are now managed by SensingProcessor internally
    
    // ZC序列
    AlignedVector _zc_seq;               
    const AlignedVector _blank_frame;

    // Pre-calculated symbol positions
    std::vector<size_t> _symbol_positions; 
    
    // Thread control
    std::atomic<bool> _running{false};
    std::thread _mod_thread;             // Modulation thread
    std::thread _tx_thread;              // TX thread
    std::thread _rx_thread;              // RX thread
    std::thread _sensing_thread;         // Sensing processing thread
    
    uhd::time_spec_t _start_time{0.0}; 

    // Pre-generated data and file
    std::ofstream tx_data_file;
    std::atomic<bool> data_saved{false};
    AlignedIntVector _pregen_data;
    // Data ingest/encoding related
    std::thread _data_thread;

    SensingDataSender _sensing_sender;
    enum class RxState { ALIGNMENT, NORMAL };
    std::atomic<RxState> _rx_state{RxState::ALIGNMENT};
    std::atomic<int> _discard_samples{63};
    
    // Control handler
    ControlCommandHandler _control_handler;

    std::atomic<bool> skip_sensing_fft{true};
    std::atomic<bool> enable_mti{true}; // Enable MTI filter by default
    std::atomic<size_t> sensing_sybmol_stride{20};
    size_t shadow_sensing_symbol_stride=20; // Shadow sensing symbol stride
    size_t sensing_symbol_count=0;
    size_t sensing_symbol_saved_count=0;
    // Hamming windows (float, not complex)
    AlignedFloatVector _range_window;    // Range processing window (Frequency domain)
    AlignedFloatVector _doppler_window;  // Doppler processing window (Time domain)
    size_t _global_symbol_index = 0; // Global symbol index (for sensing processing)
    
    // LDPC encoding and scrambling
    LDPCCodec _ldpc{make_ldpc_5041008_cfg()};
    Scrambler scrambler{201600, 0x5A};
    
    // MTI Filter

    
    // UDP socket
    int _udp_sock{-1};
    struct sockaddr_in _udp_addr{};
    
    // Object pools for memory reuse (eliminates per-frame memory allocations)
    ObjectPool<AlignedVector> _frame_pool;      // Pool for TX frame buffers
    ObjectPool<SymbolVector> _symbols_pool;     // Pool for symbol vectors (frequency domain)
    ObjectPool<AlignedVector> _rx_frame_pool;   // Pool for RX frame buffers
    ObjectPool<AlignedIntVector> _data_packet_pool; // Pool for data packet buffers

    // Core computation classes (hardware-independent)
    SensingProcessor _sensing_core;
    
    void _register_commands() {

        // Register alignment command
        _control_handler.register_command("ALGN", [this](int32_t value) {
            int32_t adjusted_value = value;
            if (adjusted_value < -1000) adjusted_value = -1000;
            if (adjusted_value > 1000) adjusted_value = 1000;
            
            _discard_samples.store(adjusted_value);
            _rx_state.store(RxState::ALIGNMENT);
            std::cout << "Received alignment command: " << adjusted_value << " samples" << std::endl;
        });
        
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

        // Register MTI control command
        _control_handler.register_command("MTI ", [this](int32_t value) {
            enable_mti.store(value != 0);
            std::cout << "Received MTI command: " << (value ? "Enable" : "Disable") << std::endl;
        });
    }

    void _init_usrp() {
        // Use device arguments from configuration
        _usrp = uhd::usrp::multi_usrp::make(_cfg.device_args);
        
        // Configure RF parameters
        _usrp->set_tx_rate(_cfg.sample_rate);
        _usrp->set_rx_rate(_cfg.sample_rate);
        
        uhd::tune_request_t tune_req(_cfg.center_freq);
        _usrp->set_tx_freq(tune_req, _cfg.tx_channel);
        _usrp->set_rx_freq(tune_req, _cfg.rx_channel);
        
        _usrp->set_tx_gain(_cfg.tx_gain, _cfg.tx_channel);
        _usrp->set_rx_gain(_cfg.rx_gain, _cfg.rx_channel);
        _usrp->set_tx_bandwidth(_cfg.bandwidth, _cfg.tx_channel);
        _usrp->set_rx_bandwidth(_cfg.bandwidth, _cfg.rx_channel);
        _usrp->set_clock_source(_cfg.clocksource);
        _discard_samples.store(_cfg.system_delay);
        // Configure TX stream
        uhd::stream_args_t tx_stream_args("fc32", _cfg.wire_format_tx);
        tx_stream_args.args["block_id"] = "radio";
        tx_stream_args.channels = {_cfg.tx_channel};
        _tx_stream = _usrp->get_tx_stream(tx_stream_args);
        
        // Configure RX stream
        uhd::stream_args_t rx_stream_args("fc32", _cfg.wire_format_rx);
        rx_stream_args.args["block_id"] = "radio";
        rx_stream_args.channels = {_cfg.rx_channel};
        _rx_stream = _usrp->get_rx_stream(rx_stream_args);
        
        // Time synchronization
        _usrp->set_time_now(uhd::time_spec_t(0.0));
        _start_time = uhd::time_spec_t(1.0); // Start after 1 second
    }

    void _init_fftw() {
        // fftwf_plan_with_nthreads(static_cast<int>(_cfg.fft_threads));
            
        // Create IFFT plan (for transmission)
        _ifft_plan = fftwf_plan_dft_1d(
            static_cast<int>(_cfg.fft_size),
            reinterpret_cast<fftwf_complex*>(_fft_in.data()),
            reinterpret_cast<fftwf_complex*>(_fft_out.data()),
            FFTW_BACKWARD,
            FFTW_MEASURE
        );
        
        // Create demodulation FFT plan (for sensing)
        _demod_fft_plan = fftwf_plan_dft_1d(
            static_cast<int>(_cfg.fft_size),
            reinterpret_cast<fftwf_complex*>(_demod_fft_in.data()),
            reinterpret_cast<fftwf_complex*>(_demod_fft_out.data()),
            FFTW_FORWARD,
            FFTW_MEASURE
        );
        
        // Note: SensingProcessor now manages its own Range IFFT and Doppler FFT plans internally
        

    }



    // Initialize Hamming windows using WindowGenerator
    void _init_hamming_windows() {
        // Range processing window (frequency domain, applied to each subcarrier)
        _range_window.resize(_cfg.fft_size);
        WindowGenerator::generate_hamming(_range_window, _cfg.fft_size);
        
        // Doppler processing window (Time domain, applied to each symbol)
        _doppler_window.resize(_cfg.sensing_symbol_num);
        WindowGenerator::generate_hamming(_doppler_window, _cfg.sensing_symbol_num);
    }

    // Prepare ZC sequence using ZadoffChuGenerator
    void _prepare_zc_sequence() {
        ZadoffChuGenerator::generate(_zc_seq, _cfg.fft_size, _cfg.zc_root);
    }

    void _precalc_positions() {
        _symbol_positions.reserve(_cfg.num_symbols);
        for (size_t i = 0; i < _cfg.num_symbols; ++i) {
            _symbol_positions.push_back(i * (_cfg.fft_size + _cfg.cp_length));
        }
    }

    // Use QPSKModulator from OFDMCore.hpp for QPSK modulation
    QPSKModulator _qpsk_modulator;

    /**
     * @brief Data Ingest and Encoding Thread.
     * 
     * Listens on a UDP port for user data.
     * On packet receipt:
     * 1. Constructs a protocol header.
     * 2. Performs LDPC encoding on the header and payload.
     * 3. Scrambles the encoded bits.
     * 4. Maps bits to QPSK symbols.
     * 5. Pushes ready-to-modulate packets to the data buffer.
     */
    void _data_ingest_proc() {
        uhd::set_thread_priority_safe();
        // Assign to available core (index 4)
        size_t core_idx = 4 % _cfg.available_cores.size();
        std::vector<size_t> cpu_list = {_cfg.available_cores[core_idx]};
        uhd::set_thread_affinity(cpu_list);
        
        // ============== Profiling variables ==============
        using ProfileClock = std::chrono::high_resolution_clock;
        static double prof_header_encode_total = 0.0;
        static double prof_payload_encode_total = 0.0;
        static double prof_enqueue_total = 0.0;
        static int prof_packet_count = 0;
        constexpr int PROF_REPORT_INTERVAL = 100;
        auto prof_step_start = ProfileClock::now();
        auto prof_step_end = prof_step_start;
        // =================================================
        
        // Initialize UDP
        _udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (_udp_sock < 0) {
            std::cerr << "UDP socket create failed" << std::endl; return; }
        int enable=1; setsockopt(_udp_sock,SOL_SOCKET,SO_REUSEADDR,&enable,sizeof(enable));
        memset(&_udp_addr,0,sizeof(_udp_addr));
        _udp_addr.sin_family = AF_INET;
        _udp_addr.sin_port = htons(static_cast<uint16_t>(_cfg.modulator_udp_port));
        if (_cfg.modulator_udp_ip == "0.0.0.0") {
            _udp_addr.sin_addr.s_addr = INADDR_ANY;
        } else {
            if (inet_pton(AF_INET, _cfg.modulator_udp_ip.c_str(), &_udp_addr.sin_addr) != 1) {
                std::cerr << "Invalid modulator UDP bind IP: " << _cfg.modulator_udp_ip << std::endl;
                close(_udp_sock); _udp_sock = -1; return;
            }
        }
        if (bind(_udp_sock,(sockaddr*)&_udp_addr,sizeof(_udp_addr))<0) {
            std::cerr << "UDP bind failed" << std::endl; close(_udp_sock); _udp_sock=-1; return; }
        // Non-blocking
        fcntl(_udp_sock,F_SETFL, O_NONBLOCK);

        std::vector<uint8_t> udp_buf(25200); // Max packet size: 1008*2*100/8 bytes per frame
        while (_running.load()) {
            ssize_t recv_len = recv(_udp_sock, udp_buf.data(), udp_buf.size(), 0);
            if (recv_len <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            size_t payload_len = static_cast<size_t>(recv_len); // Raw UDP payload bytes
            // Each LDPC input block represents information bits in bytes: bytes_per_ldpc_block = ceil(K/8)
            size_t K_bits_local = _ldpc.get_K();
            size_t bytes_per_ldpc_block = (K_bits_local + 7) / 8; // Information bytes per LDPC block
            size_t padded_len = ((payload_len + bytes_per_ldpc_block-1) / bytes_per_ldpc_block) * bytes_per_ldpc_block; // Payload bytes after zero padding
            // size_t num_blocks = padded_len / bytes_per_ldpc_block; // Number of LDPC blocks (unused)
            // Construct LDPC header: Determine header bytes dynamically based on LDPC K (K/8), split header into two halves.
            // First half fills with counts starting from 1 (byte-wise), second half repeats payload_len (truncated to 16 bits) in big-endian 2-byte chunks.
            // header_len already calculated based on K_bits_local (info bit bytes)
            size_t header_len = K_bits_local ? (K_bits_local + 7) / 8 : 0; // Bytes
            LDPCCodec::AlignedByteVector header_bytes(header_len, 0x00);
            // Split into two halves: Prioritize extra bytes to first half, ensure second half has even byte count
            // Initially put extra byte to first half
            size_t half1 = (header_len + 1) / 2; // First half gets extra byte priority
            size_t half2 = header_len - half1;
            // If second half is odd and adjustable, move one byte to first half to ensure second half is even
            if ((half2 % 2) != 0 && half2 > 0) {
                // Move one byte from second half to first half
                half1 += 1;
                half2 -= 1;
            }

            // First half: count from 1 (byte), wraps around to low 8 bits if exceeding 255
            for (size_t i = 0; i < half1; ++i) {
                header_bytes[i] = static_cast<uint8_t>((i + 1) & 0xFF);
            }

            // Second half: Repeat payload_len (write 2 bytes each time, big-endian). If second half bytes is odd, last byte writes half of low byte.
            uint16_t payload16 = payload_len > 0xFFFF ? 0xFFFF : static_cast<uint16_t>(payload_len & 0xFFFF);
            for (size_t i = 0; i < half2; ++i) {
                size_t idx = half1 + i;
                // Even pos writes high byte, odd pos writes low byte (relative to start of second half)
                if ((i % 2) == 0) {
                    header_bytes[idx] = static_cast<uint8_t>((payload16 >> 8) & 0xFF);
                } else {
                    header_bytes[idx] = static_cast<uint8_t>(payload16 & 0xFF);
                }
            }

            // LDPC encode header
            prof_step_start = ProfileClock::now();
            LDPCCodec::AlignedIntVector header_coded_bits;
            _ldpc.encode_frame(header_bytes, header_coded_bits);
            
            // Scramble header
            scrambler.scramble(header_coded_bits);
            
            // Convert to QPSK symbols
            LDPCCodec::AlignedIntVector header_qpsk_ints;
            LDPCCodec::pack_bits_qpsk(header_coded_bits, header_qpsk_ints);
            prof_step_end = ProfileClock::now();
            double header_encode_time = std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
            
            // Process payload data
            // Construct: Pure UDP data + padding(0x00) to make length a multiple of bytes_per_ldpc_block
            LDPCCodec::AlignedByteVector input_bytes(padded_len, 0x00);
            // Direct copy UDP payload
            std::memcpy(input_bytes.data(), udp_buf.data(), std::min<size_t>(recv_len, udp_buf.size()));
            // Remaining padding stays 0x00


            // LDPC encode whole packet (may contain multiple LDPC blocks)
            prof_step_start = ProfileClock::now();
            LDPCCodec::AlignedIntVector encoded_bits_all;
            _ldpc.encode_frame(input_bytes, encoded_bits_all); // bits: (#blocks*bytes_per_ldpc_block*8)

            // Batch scramble using QPSKScrambler
            scrambler.scramble(encoded_bits_all);

            // Convert to QPSK symbol integers
            LDPCCodec::AlignedIntVector qpsk_ints_all;
            LDPCCodec::pack_bits_qpsk(encoded_bits_all, qpsk_ints_all); // (#blocks*bytes_per_ldpc_block*8) ints
            prof_step_end = ProfileClock::now();
            double payload_encode_time = std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

            // Construct final output: LDPC encoded header + encoded payload data
            AlignedIntVector packet = _data_packet_pool.acquire();
            packet.clear();
            packet.reserve(header_qpsk_ints.size() + qpsk_ints_all.size());
            packet.insert(packet.end(), header_qpsk_ints.begin(), header_qpsk_ints.end());
            packet.insert(packet.end(), qpsk_ints_all.begin(), qpsk_ints_all.end());
            // Print debug info
            // std::cout << "Packet constructed with size: " << packet.size() << std::endl;
            // Enqueue
            prof_step_start = ProfileClock::now();
            {
                std::unique_lock<std::mutex> lock(_data_mutex);
                _data_not_full.wait(lock,[this]{return !_running.load() || !_data_packet_buffer.full();});
                if (!_running.load()) break;
                _data_packet_buffer.push_back(std::move(packet));
                _data_not_empty.notify_one();
            }

            prof_step_end = ProfileClock::now();
            double enqueue_time = std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
            
            // ============== Profiling accumulation ==============
            prof_header_encode_total += header_encode_time;
            prof_payload_encode_total += payload_encode_time;
            prof_enqueue_total += enqueue_time;
            prof_packet_count++;
            
            if (prof_packet_count >= PROF_REPORT_INTERVAL && _cfg.should_profile("data_ingest")) {
                std::cout << "\n========== _data_ingest_proc Profiling (avg per packet, us) ==========" << std::endl;
                std::cout << "Header Encode:        " << prof_header_encode_total / prof_packet_count << " us" << std::endl;
                std::cout << "Payload Encode:       " << prof_payload_encode_total / prof_packet_count << " us" << std::endl;
                std::cout << "Enqueue:              " << prof_enqueue_total / prof_packet_count << " us" << std::endl;
                double total = prof_header_encode_total + prof_payload_encode_total + prof_enqueue_total;
                std::cout << "TOTAL:                " << total / prof_packet_count << " us" << std::endl;
                std::cout << "======================================================================\n" << std::endl;
                
                // Reset counters
                prof_header_encode_total = 0.0;
                prof_payload_encode_total = 0.0;
                prof_enqueue_total = 0.0;
                prof_packet_count = 0;
            }
        }
        if (_udp_sock>=0) close(_udp_sock);
    }
    /**
     * @brief Sensing Processing Thread (Modulator Side).
     * 
     * Processes self-received signals for mono-static sensing.
     * Performs channel estimation and Range-Doppler processing.
     */
    void _sensing_process(const SensingFrame& frame) {
        // ============== Profiling variables ==============
        using ProfileClock = std::chrono::high_resolution_clock;
        static double prof_fft_total = 0.0;
        static double prof_channel_est_total = 0.0;
        static double prof_mti_total = 0.0;
        static double prof_window_total = 0.0;
        static double prof_range_doppler_total = 0.0;
        static double prof_send_total = 0.0;
        static int prof_frame_count = 0;
        constexpr int PROF_REPORT_INTERVAL = 100;
        auto prof_step_start = ProfileClock::now();
        auto prof_step_end = prof_step_start;
        // =================================================
        
        // Send heartbeat for NAT traversal (once per second)
        static auto next_hb_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (now >= next_hb_time) {
            _control_handler.send_heartbeat(_cfg.mono_sensing_ip, _cfg.mono_sensing_port);
            next_hb_time = now + std::chrono::seconds(1);
        }

        static auto next_send_time = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() >= next_send_time) {
            // 1. Perform OFDM demodulation on received frame and copy to internal buffer
            prof_step_start = ProfileClock::now();
            for (size_t i = 0; i < _cfg.sensing_symbol_num; ++i) {
                std::copy(frame.rx_symbols[i].begin(), frame.rx_symbols[i].end(), _demod_fft_in.begin());
                // Execute FFT
                fftwf_execute(_demod_fft_plan);
                // Copy FFT result to SensingProcessor's internal buffer
                _sensing_core.copy_fft_result_to_buffer(i, _demod_fft_out);
            }
            prof_step_end = ProfileClock::now();
            prof_fft_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
            
            // Get reference to channel buffer for subsequent operations
            auto& channel_buf = _sensing_core.channel_buffer();
            
            // 2. Channel estimation (Frequency domain division) + FFT shift
            // Using SensingProcessor for combined channel estimation and FFT shift
            prof_step_start = ProfileClock::now();
            _sensing_core.channel_estimate_with_shift(frame.tx_symbols);
            prof_step_end = ProfileClock::now();
            prof_channel_est_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

            // Apply MTI Filter (Moving Target Indication)
            // Applied after channel estimation and FFT shift, before windowing and IFFT
            prof_step_start = ProfileClock::now();
            _sensing_core.apply_mti(enable_mti.load());
            prof_step_end = ProfileClock::now();
            prof_mti_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

            if (!skip_sensing_fft.load()) {
                // Apply range and Doppler windows using SensingProcessor
                prof_step_start = ProfileClock::now();
                _sensing_core.apply_windows(channel_buf, _range_window, _doppler_window);
                prof_step_end = ProfileClock::now();
                prof_window_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

                // Range-Doppler processing using SensingProcessor's internal FFT plans
                prof_step_start = ProfileClock::now();
                _sensing_core.execute_range_ifft();
                _sensing_core.execute_doppler_fft();
                prof_step_end = ProfileClock::now();
                prof_range_doppler_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
            }

            // 5. Send processing results
            prof_step_start = ProfileClock::now();
            _sensing_sender.push_data(channel_buf);
            if(_cfg.range_fft_size != _cfg.fft_size || _cfg.doppler_fft_size != _cfg.fft_size) {
                _sensing_core.clear_channel_buffer();
            }
            prof_step_end = ProfileClock::now();
            prof_send_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
            
            // ============== Profiling report ==============
            prof_frame_count++;
            if (prof_frame_count >= PROF_REPORT_INTERVAL && _cfg.should_profile("sensing_process")) {
                std::cout << "\n========== _sensing_process Profiling (avg per frame, us) ==========" << std::endl;
                std::cout << "FFT Demodulation:     " << prof_fft_total / prof_frame_count << " us" << std::endl;
                std::cout << "Channel Estimation:   " << prof_channel_est_total / prof_frame_count << " us" << std::endl;
                std::cout << "MTI Filter:           " << prof_mti_total / prof_frame_count << " us" << std::endl;
                std::cout << "Windowing:            " << prof_window_total / prof_frame_count << " us" << std::endl;
                std::cout << "Range-Doppler FFT:    " << prof_range_doppler_total / prof_frame_count << " us" << std::endl;
                std::cout << "Send Data:            " << prof_send_total / prof_frame_count << " us" << std::endl;
                double total = prof_fft_total + prof_channel_est_total + prof_mti_total + prof_window_total + prof_range_doppler_total + prof_send_total;
                std::cout << "TOTAL:                " << total / prof_frame_count << " us" << std::endl;
                std::cout << "====================================================================\n" << std::endl;
                
                // Reset counters
                prof_fft_total = 0.0;
                prof_channel_est_total = 0.0;
                prof_mti_total = 0.0;
                prof_window_total = 0.0;
                prof_range_doppler_total = 0.0;
                prof_send_total = 0.0;
                prof_frame_count = 0;
            }
            
            next_send_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
        }
        
    }

    /**
     * @brief Modulation Thread.
     * 
     * Continuous loop that:
     * 1. Fetches data packets from the data buffer.
     * 2. Maps bits to QPSK symbols.
     * 3. Inserts pilots and ZC sequences (Sync).
     * 4. Performs IFFT to generate time-domain OFDM symbols.
     * 5. Adds Cyclic Prefix (CP).
     * 6. Pushes generated frames to the circular buffer for transmission.
     */
    void _modulation_proc() {
        uhd::set_thread_priority_safe();
        // Assign to available core (index 2)
        size_t core_idx = 2 % _cfg.available_cores.size();
        std::vector<size_t> cpu_list = {_cfg.available_cores[core_idx]};
        uhd::set_thread_affinity(cpu_list);
        // Frame processing time statistics
        using Clock = std::chrono::high_resolution_clock;
        Clock::time_point frame_start, frame_end;
        double total_processing_time = 0.0;
        int frame_count = 0;
        constexpr int REPORT_INTERVAL = 434;  // Report average time every 434 frames (approx 1s)

        // ============== Profiling variables ==============
        using ProfileClock = std::chrono::high_resolution_clock;
        static double prof_data_fetch_total = 0.0;
        static double prof_symbol_gen_total = 0.0;
        static double prof_ifft_total = 0.0;
        static double prof_cp_write_total = 0.0;
        static int prof_frame_count = 0;
        constexpr int PROF_REPORT_INTERVAL = 434;
        static uint64_t oversize_head_warn_count = 0;
        
        auto prof_step_start = ProfileClock::now();
        auto prof_step_end = prof_step_start;
        // =================================================

        const float scale = 1.0f / std::sqrt(_cfg.fft_size)/4; // Output digital signal power is 1/16

        // Pre-calculate pilot and data subcarrier indices to reduce branches inside loop
        std::vector<char> is_pilot(_cfg.fft_size, 0);
        for (auto pos : _cfg.pilot_positions) if (pos < _cfg.fft_size) is_pilot[pos] = 1;
        std::vector<int> data_subcarriers; data_subcarriers.reserve(_cfg.fft_size - _cfg.pilot_positions.size());
        for (size_t k=0;k<_cfg.fft_size;++k) if (!is_pilot[k]) data_subcarriers.push_back((int)k);
        const size_t max_data_syms_per_frame = (_cfg.num_symbols > 0)
            ? (_cfg.num_symbols - 1) * data_subcarriers.size()
            : 0;

        while (_running.load()) {
            frame_start = Clock::now(); // Record frame start time

            // Acquire pre-allocated objects from pools (zero heap allocations)
            AlignedVector current_frame = _frame_pool.acquire();
            SymbolVector current_symbols = _symbols_pool.acquire();
            
            size_t frame_data_index = 0; // Entire frame data index
            // Pool of real data symbols available for this frame
            prof_step_start = ProfileClock::now();
            AlignedIntVector data_pool;
            data_pool.reserve(max_data_syms_per_frame);
            {
                std::unique_lock<std::mutex> lock(_data_mutex);
                // Pull packets only when the *whole* packet fits in current frame room.
                while (data_pool.size() < max_data_syms_per_frame && !_data_packet_buffer.empty()) {
                    const size_t room = max_data_syms_per_frame - data_pool.size();
                    const size_t pkt_size = _data_packet_buffer.front().size();
                    if (pkt_size > room) {
                        // Current frame has no room for the next complete packet: stop pulling.
                        // Leave remaining subcarriers blank/pregen in this frame.
                        if (pkt_size > max_data_syms_per_frame) {
                            // Keep packet in queue (strict "do not fetch if it does not fit").
                            // Warn periodically because this can stall queue progress.
                            oversize_head_warn_count++;
                            if (oversize_head_warn_count <= 20 || (oversize_head_warn_count % 100) == 0) {
                                std::cerr << "[LDPC] Queue head packet exceeds per-frame capacity, not fetched: qpsk_syms="
                                          << pkt_size << ", max_qpsk_per_frame=" << max_data_syms_per_frame
                                          << ", warn_count=" << oversize_head_warn_count << std::endl;
                            }
                        }
                        break;
                    }
                    AlignedIntVector pkt = std::move(_data_packet_buffer.front());
                    _data_packet_buffer.pop_front();
                    data_pool.insert(data_pool.end(), pkt.begin(), pkt.end());
                    _data_packet_pool.release(std::move(pkt));
                }
                _data_not_full.notify_all();
            }

            size_t data_pool_pos = 0;
            prof_step_end = ProfileClock::now();
            prof_data_fetch_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

            double symbol_gen_time = 0.0;
            double ifft_time = 0.0;
            double cp_write_time = 0.0;

            // Objects from pool are already correctly sized - no need to resize

            for (size_t i = 0; i < _cfg.num_symbols; ++i) {
                const size_t pos = _symbol_positions[i];
                auto* buf_ptr = current_frame.data() + pos;

                prof_step_start = ProfileClock::now();
                if (i == _cfg.sync_pos) {
                    // Sync symbol: Use ZC sequence for the whole symbol
                    std::memcpy(_fft_in.data(), _zc_seq.data(), _cfg.fft_size * sizeof(std::complex<float>));
                } else {
                    // Non-sync symbol: Use real data symbols from data_pool, fallback to pre-generated data if exhausted
                    // Fill all pilots first (vectorization friendly)
                    for (size_t idx = 0; idx < _cfg.pilot_positions.size(); ++idx) {
                        size_t k = _cfg.pilot_positions[idx];
                        _fft_in[k] = _zc_seq[k];
                    }
                    // Fill data subcarriers - use lookup table for QPSK
                    const size_t ds_count = data_subcarriers.size();
                    const int* __restrict__ ds_ptr = data_subcarriers.data();
                    auto* __restrict__ fft_ptr = _fft_in.data();
                    const auto* __restrict__ pool_ptr = data_pool.data();
                    const auto* __restrict__ pregen_ptr = _pregen_data.data();
                    const size_t pool_size = data_pool.size();
                    
                    #pragma omp simd
                    for (size_t di = 0; di < ds_count; ++di) {
                        const int k = ds_ptr[di];
                        const int sym = (data_pool_pos + di < pool_size) ? 
                            pool_ptr[data_pool_pos + di] : pregen_ptr[frame_data_index + di];
                        // Use QPSKModulator static lookup table for SIMD compatibility
                        const int idx = (sym & 3) * 2;
                        fft_ptr[k] = std::complex<float>(
                            QPSKModulator::QPSK_TABLE_FLAT[idx],
                            QPSKModulator::QPSK_TABLE_FLAT[idx + 1]
                        );
                    }
                    // Update indices after loop
                    const size_t used_from_pool = std::min(ds_count, pool_size > data_pool_pos ? pool_size - data_pool_pos : 0);
                    data_pool_pos += used_from_pool;
                    frame_data_index += ds_count - used_from_pool;
                }
                
                // Save frequency domain data of current symbol (for sensing) - direct memcpy
                std::memcpy(current_symbols[i].data(), _fft_in.data(), _cfg.fft_size * sizeof(std::complex<float>));
                prof_step_end = ProfileClock::now();
                symbol_gen_time += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

                // IFFT transform
                prof_step_start = ProfileClock::now();
                fftwf_execute(_ifft_plan);
                prof_step_end = ProfileClock::now();
                ifft_time += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

                // Add Cyclic Prefix (CP)
                prof_step_start = ProfileClock::now();
                #pragma omp simd
                for (size_t j = 0; j < _cfg.cp_length; ++j) {
                    buf_ptr[j] = _fft_out[_cfg.fft_size - _cfg.cp_length + j] * scale;
                }
                
                // Write symbol body
                #pragma omp simd
                for (size_t j = 0; j < _cfg.fft_size; ++j) {
                    buf_ptr[_cfg.cp_length + j] = _fft_out[j] * scale;
                }
                prof_step_end = ProfileClock::now();
                cp_write_time += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
            }

            prof_symbol_gen_total += symbol_gen_time;
            prof_ifft_total += ifft_time;
            prof_cp_write_total += cp_write_time;

            frame_end = Clock::now();// Record frame end time
            double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
            total_processing_time += frame_time;
            frame_count++;

            if (frame_count >= REPORT_INTERVAL) {
                double avg_time = total_processing_time / frame_count;
                double frame_duration = _cfg.samples_per_frame() / _cfg.sample_rate * 1000.0; // Convert to ms
                double load = avg_time / frame_duration;
                std::cout << "Average processing time: " << avg_time 
                            << " ms, Load: " << load * 100.0 << "%" << std::endl;
                // Reset statistics
                total_processing_time = 0.0;
                frame_count = 0;
            }
            {
                std::unique_lock<std::mutex> lock(_buffer_mutex);
                _buffer_not_full.wait(lock, [this]() {
                    return !_running.load() || !_circular_buffer.full();
                });

                if (!_running.load()) break;

                _circular_buffer.push_back(std::move(current_frame));
                _buffer_not_empty.notify_one();
            }
            {
                std::unique_lock<std::mutex> lock(_symbols_mutex);
                if(_symbols_buffer.full())
                {
                    _symbols_not_full.wait(lock, [this]() {
                        return !_running.load() || !_symbols_buffer.full();
                    });
                }


                if (!_running.load()) break;

                _symbols_buffer.push_back(std::move(current_symbols));
                _symbols_not_empty.notify_one();
            }
            
            // ============== Profiling report ==============
            prof_frame_count++;
            if (prof_frame_count >= PROF_REPORT_INTERVAL && _cfg.should_profile("modulation")) {
                std::cout << "\n========== _modulation_proc Profiling (avg per frame, us) ==========" << std::endl;
                std::cout << "Data Fetch:           " << prof_data_fetch_total / prof_frame_count << " us" << std::endl;
                std::cout << "Symbol Generation:    " << prof_symbol_gen_total / prof_frame_count << " us" << std::endl;
                std::cout << "IFFT (all symbols):   " << prof_ifft_total / prof_frame_count << " us" << std::endl;
                std::cout << "CP & Write:           " << prof_cp_write_total / prof_frame_count << " us" << std::endl;
                double total = prof_data_fetch_total + prof_symbol_gen_total + prof_ifft_total + prof_cp_write_total;
                std::cout << "TOTAL:                " << total / prof_frame_count << " us" << std::endl;
                std::cout << "===================================================================\n" << std::endl;
                
                // Reset counters
                prof_data_fetch_total = 0.0;
                prof_symbol_gen_total = 0.0;
                prof_ifft_total = 0.0;
                prof_cp_write_total = 0.0;
                prof_frame_count = 0;
            }
        }
    }

    /**
     * @brief Tx Streamer Thread.
     * 
     * Continuous loop that sends frames from the circular buffer to the USRP.
     * Handles underflow by sending blank frames if no data is available.
     * Ensures continuous transmission for stable sensing and communication.
     */
    void _tx_proc() {
        uhd::set_thread_priority_safe(1.0f);
        // Assign to available core (index 0)
        size_t core_idx = 0 % _cfg.available_cores.size();
        std::vector<size_t> cpu_list = {_cfg.available_cores[core_idx]};
        uhd::set_thread_affinity(cpu_list);
        
        uhd::tx_metadata_t md;
        md.start_of_burst = true;
        md.end_of_burst = false;
        md.has_time_spec = true;
        md.time_spec = _start_time;
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // wait for prefill
        while (_running.load(std::memory_order_relaxed)) {
            AlignedVector frame_to_send;
            bool has_frame = false;
            {
                std::unique_lock<std::mutex> lock(_buffer_mutex);
                if (_buffer_not_empty.wait_for(lock, std::chrono::microseconds(0), 
                    [this]() { return !_running.load() || !_circular_buffer.empty(); })) 
                {
                    if (!_running.load()) break;
                    if (!_circular_buffer.empty()) {
                        frame_to_send = std::move(_circular_buffer.front());
                        _circular_buffer.pop_front();
                        has_frame = true;
                        _buffer_not_full.notify_one();
                    }
                }
            }

            if (has_frame) {
                size_t sent = _tx_stream->send(frame_to_send.data(), frame_to_send.size(), md, 2.0);
                if (sent < frame_to_send.size()) {
                    std::cerr << "TX Underflow: " << (frame_to_send.size() - sent) << " samples\n";
                }
                // Return frame to pool for reuse (instead of destroying)
                _frame_pool.release(std::move(frame_to_send));
            } else {
                _tx_stream->send(_blank_frame.data(), _blank_frame.size(), md);
                std::cerr.put('B');
            }
            //md.time_spec += uhd::time_spec_t(_cfg.samples_per_frame() / _cfg.sample_rate);
            md.start_of_burst = false;
            md.has_time_spec = false;
        }
        
        md.end_of_burst = true;
        _tx_stream->send("", 0, md);
    }

    /**
     * @brief Rx (Sensing) Streamer Thread.
     * 
     * Continuous loop that receives the self-transmitted signal (echo) from the USRP.
     * Used for mono-static sensing.
     * Implements state machine for alignment (system latency removal) and normal reception.
     */
    void _rx_proc() {
        uhd::set_thread_priority_safe();
        // Assign to available core (index 1)
        size_t core_idx = 1 % _cfg.available_cores.size();
        std::vector<size_t> cpu_list = {_cfg.available_cores[core_idx]};
        uhd::set_thread_affinity(cpu_list);
        
        uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
        stream_cmd.stream_now = false;
        stream_cmd.time_spec = _start_time;
        _rx_stream->issue_stream_cmd(stream_cmd);
        
        while (_running.load()) {
            switch (_rx_state.load()) {
            case RxState::ALIGNMENT:
                _handle_alignment();
                break;
            case RxState::NORMAL:
                _handle_normal_rx();
                break;
            }
        }
        
        _rx_stream->issue_stream_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
    }
    
    /**
     * @brief Alignment State Handler.
     * 
     * Discards a set number of samples to align the receive stream with the transmit stream.
     * This compensates for system latency (hardware delay + USRP pipeline delay).
     */
    void _handle_alignment() {
        const size_t total_read = _cfg.samples_per_frame() + _discard_samples;
        AlignedVector temp_buf(total_read);
        uhd::rx_metadata_t md;
        size_t received = 0;
        
        // Read total samples (one frame + discarded samples)
        while (received < total_read && _running.load()) {
            size_t num_rx = _rx_stream->recv(
                temp_buf.data() + received, 
                total_read - received, 
                md, 
                1.0,  // Shorten timeout
                false
            );
            
            if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
                if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
                    std::cerr << "RX alignment error: " << md.strerror() << std::endl;
                }
                continue;
            }
            
            received += num_rx;
        }
        
        if (!_running.load()) return;
        
        // Discard first _discard_samples samples, take one frame data
        // Acquire from pool instead of allocating new
        AlignedVector aligned_frame = _rx_frame_pool.acquire();
        if(_discard_samples > 0)
        {
            std::copy(
                temp_buf.begin() + _discard_samples,
                temp_buf.begin() + _discard_samples + _cfg.samples_per_frame(),
                aligned_frame.begin()
            );
        }
        else
        {
            std::copy(temp_buf.begin(),
                temp_buf.begin() + _discard_samples + _cfg.samples_per_frame(),
                aligned_frame.begin() - _discard_samples);
        }
        
        // Put into receive buffer
        {
            std::unique_lock<std::mutex> lock(_rx_mutex);
            _rx_not_full.wait(lock, [this]() {
                return !_running.load() || !_rx_frame_buffer.full();
            });

            if (!_running.load()) return;

            _rx_frame_buffer.push_back(std::move(aligned_frame));
            _rx_not_empty.notify_one();
        }
        
        // Switch to normal receive mode
        _rx_state.store(RxState::NORMAL);
        std::cout << "Frame aligned. Discarded " << _discard_samples << " samples." << std::endl;
    }
    
    // Normal receive state handling
    /**
     * @brief Normal Reception State Handler.
     * 
     * Receives aligned frames from the RX stream and pushes them to the circular buffer
     * for sensing processing.
     */
    void _handle_normal_rx() {
        // Acquire pre-allocated RX frame from pool (zero heap allocation)
        AlignedVector rx_frame = _rx_frame_pool.acquire();
        uhd::rx_metadata_t md;
        size_t received = 0;
        
        // Read one frame data
        while (received < _cfg.samples_per_frame() && _running.load()) {
            size_t num_rx = _rx_stream->recv(
                rx_frame.data() + received, 
                _cfg.samples_per_frame() - received, 
                md, 
                2.0, 
                false
            );
            
            if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
                if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
                    std::cerr << "RX error: " << md.strerror() << std::endl;
                }
                continue;
            }
            
            received += num_rx;
        }
        
        if (!_running.load()) return;
        
        // Put into receive buffer
        {
            std::unique_lock<std::mutex> lock(_rx_mutex);
            _rx_not_full.wait(lock, [this]() {
                return !_running.load() || !_rx_frame_buffer.full();
            });

            if (!_running.load()) return;

            _rx_frame_buffer.push_back(std::move(rx_frame));
            _rx_not_empty.notify_one();
        }
    }

    // Sensing processing thread
    void _sensing_proc() {
        uhd::set_thread_priority_safe();
        // Assign to available core (index 3)
        size_t core_idx = 3 % _cfg.available_cores.size();
        std::vector<size_t> cpu_list = {_cfg.available_cores[core_idx]};
        uhd::set_thread_affinity(cpu_list);
        
        // ============== Profiling variables ==============
        using ProfileClock = std::chrono::high_resolution_clock;
        static double prof_frame_fetch_total = 0.0;
        static double prof_symbol_extract_total = 0.0;
        static double prof_sensing_process_total = 0.0;
        static int prof_frame_count = 0;
        constexpr int PROF_REPORT_INTERVAL = 100;
        auto prof_step_start = ProfileClock::now();
        auto prof_step_end = prof_step_start;
        // =================================================
        
        while (_running.load()) {
            SymbolVector tx_symbols;
            AlignedVector rx_frame_data;
            bool has_frame = false;
            prof_step_start = ProfileClock::now();
            {
                while (!has_frame && _running.load()) {
                    std::unique_lock<std::mutex> rx_lock(_rx_mutex);
                    _rx_not_empty.wait_for(rx_lock, std::chrono::milliseconds(10), [this]() {
                        return !_running.load() || !_rx_frame_buffer.empty();
                    });

                    if (!_running.load()) break;

                    if (!_rx_frame_buffer.empty()) {
                        std::unique_lock<std::mutex> symbols_lock(_symbols_mutex);
                        auto condition = [this]() { return !_symbols_buffer.empty(); };
                        if (_symbols_not_empty.wait_for(symbols_lock, 
                                                    std::chrono::milliseconds(0), 
                                                    condition)) 
                        {
                            // Both buffers have data
                            rx_frame_data = std::move(_rx_frame_buffer.front());
                            _rx_frame_buffer.pop_front();
                            _rx_not_full.notify_one();
                            tx_symbols = std::move(_symbols_buffer.front());
                            _symbols_buffer.pop_front();
                            _symbols_not_full.notify_one();
                            has_frame = true;
                        }
                    }
                }

                if (!has_frame) continue;
            }
            prof_step_end = ProfileClock::now();
            prof_frame_fetch_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
            
            if (has_frame) {
                prof_step_start = ProfileClock::now();
                const size_t symbols_in_this_frame = tx_symbols.size();                
                // Process all symbols in this frame matching global index
                while (_global_symbol_index < symbols_in_this_frame) {
                    const size_t symbol_idx = _global_symbol_index;
                    
                    // Extract current symbol from received frame (remove CP)
                    AlignedVector rx_symbol(_cfg.fft_size);
                    size_t symbol_start = symbol_idx * (_cfg.fft_size + _cfg.cp_length) + _cfg.cp_length;
                    std::copy(
                        rx_frame_data.begin() + symbol_start,
                        rx_frame_data.begin() + symbol_start+_cfg.fft_size,
                        rx_symbol.begin()
                    );
                    
                    // Add current symbol to accumulation buffer
                    _accumulated_rx_symbols.push_back(std::move(rx_symbol));
                    _accumulated_tx_symbols.push_back(tx_symbols[symbol_idx]);

                    _global_symbol_index += shadow_sensing_symbol_stride;
                    
                    // Check if processing threshold is reached
                    if (_accumulated_tx_symbols.size() >= _cfg.sensing_symbol_num) {
                        prof_step_end = ProfileClock::now();
                        prof_symbol_extract_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
                        
                        // Construct sensing frame
                        SensingFrame sensing_frame;
                        sensing_frame.rx_symbols = std::move(_accumulated_rx_symbols);
                        sensing_frame.tx_symbols = std::move(_accumulated_tx_symbols);
                        
                        // Reset accumulation buffer
                        _accumulated_rx_symbols.clear();
                        _accumulated_tx_symbols.clear();
                        
                        // Execute sensing process
                        auto sensing_start = ProfileClock::now();
                        _sensing_process(sensing_frame);
                        auto sensing_end = ProfileClock::now();
                        prof_sensing_process_total += std::chrono::duration<double, std::micro>(sensing_end - sensing_start).count();
                        shadow_sensing_symbol_stride = sensing_sybmol_stride.load();
                        
                        // ============== Profiling report ==============
                        prof_frame_count++;
                        if (prof_frame_count >= PROF_REPORT_INTERVAL && _cfg.should_profile("sensing_proc")) {
                            std::cout << "\n========== _sensing_proc Profiling (avg per sensing frame, us) ==========" << std::endl;
                            std::cout << "Frame Fetch:          " << prof_frame_fetch_total / prof_frame_count << " us" << std::endl;
                            std::cout << "Symbol Extract:       " << prof_symbol_extract_total / prof_frame_count << " us" << std::endl;
                            std::cout << "Sensing Process:      " << prof_sensing_process_total / prof_frame_count << " us" << std::endl;
                            double total = prof_frame_fetch_total + prof_symbol_extract_total + prof_sensing_process_total;
                            std::cout << "TOTAL:                " << total / prof_frame_count << " us" << std::endl;
                            std::cout << "==========================================================================\n" << std::endl;
                            
                            // Reset counters
                            prof_frame_fetch_total = 0.0;
                            prof_symbol_extract_total = 0.0;
                            prof_sensing_process_total = 0.0;
                            prof_frame_count = 0;
                        }
                        
                        prof_step_start = ProfileClock::now(); // Reset for next symbol extraction
                    }
                }
                // Frame processing complete, reset global index (subtract symbols in this frame)
                _global_symbol_index -= symbols_in_this_frame;
                

                // Return objects to pools for reuse (instead of destroying)
                _symbols_pool.release(std::move(tx_symbols));
                _rx_frame_pool.release(std::move(rx_frame_data));
            }
        }
    }
};

// --- Global Signal Handling ---
std::atomic<bool> stop_signal(false);
void signal_handler(int) { stop_signal.store(true); }

int UHD_SAFE_MAIN(int argc, char *argv[]) {
    std::signal(SIGINT, &signal_handler);
    uhd::set_thread_priority_safe();
    
    // Default config file path
    const std::string default_config_file = "Modulator.yaml";
    
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
        out << YAML::Key << "tx_gain" << YAML::Value << cfg.tx_gain;
        out << YAML::Key << "rx_gain" << YAML::Value << cfg.rx_gain;
        out << YAML::Key << "rx_channel" << YAML::Value << cfg.rx_channel;
        out << YAML::Key << "zc_root" << YAML::Value << cfg.zc_root;
        out << YAML::Key << "num_symbols" << YAML::Value << cfg.num_symbols;
        out << YAML::Key << "tx_frame_buffer_size" << YAML::Value << cfg.tx_frame_buffer_size;
        out << YAML::Key << "tx_symbols_buffer_size" << YAML::Value << cfg.tx_symbols_buffer_size;
        out << YAML::Key << "rx_frame_buffer_size" << YAML::Value << cfg.rx_frame_buffer_size;
        out << YAML::Key << "device_args" << YAML::Value << cfg.device_args;
        out << YAML::Key << "clock_source" << YAML::Value << cfg.clocksource;
        out << YAML::Key << "system_delay" << YAML::Value << cfg.system_delay;
        out << YAML::Key << "wire_format_tx" << YAML::Value << cfg.wire_format_tx;
        out << YAML::Key << "wire_format_rx" << YAML::Value << cfg.wire_format_rx;
        out << YAML::Key << "mod_udp_ip" << YAML::Value << cfg.modulator_udp_ip;
        out << YAML::Key << "mod_udp_port" << YAML::Value << cfg.modulator_udp_port;
        out << YAML::Key << "sensing_ip" << YAML::Value << cfg.mono_sensing_ip;
        out << YAML::Key << "sensing_port" << YAML::Value << cfg.mono_sensing_port;
        out << YAML::Key << "default_ip" << YAML::Value << cfg.default_ip;
        out << YAML::Key << "profiling_modules" << YAML::Value << cfg.profiling_modules;
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
            if (config["tx_gain"]) cfg.tx_gain = config["tx_gain"].as<double>();
            if (config["rx_gain"]) cfg.rx_gain = config["rx_gain"].as<double>();
            if (config["rx_channel"]) cfg.rx_channel = config["rx_channel"].as<size_t>();
            if (config["zc_root"]) cfg.zc_root = config["zc_root"].as<int>();
            if (config["num_symbols"]) cfg.num_symbols = config["num_symbols"].as<size_t>();
            if (config["tx_frame_buffer_size"]) cfg.tx_frame_buffer_size = config["tx_frame_buffer_size"].as<size_t>();
            if (config["tx_symbols_buffer_size"]) cfg.tx_symbols_buffer_size = config["tx_symbols_buffer_size"].as<size_t>();
            if (config["rx_frame_buffer_size"]) cfg.rx_frame_buffer_size = config["rx_frame_buffer_size"].as<size_t>();
            if (config["device_args"]) cfg.device_args = config["device_args"].as<std::string>();
            if (config["clock_source"]) cfg.clocksource = config["clock_source"].as<std::string>();
            if (config["system_delay"]) cfg.system_delay = config["system_delay"].as<int32_t>();
            if (config["wire_format_tx"]) cfg.wire_format_tx = config["wire_format_tx"].as<std::string>();
            if (config["wire_format_rx"]) cfg.wire_format_rx = config["wire_format_rx"].as<std::string>();
            if (config["mod_udp_ip"]) cfg.modulator_udp_ip = config["mod_udp_ip"].as<std::string>();
            if (config["mod_udp_port"]) cfg.modulator_udp_port = config["mod_udp_port"].as<int>();
            if (config["sensing_ip"]) cfg.mono_sensing_ip = config["sensing_ip"].as<std::string>();
            if (config["sensing_port"]) cfg.mono_sensing_port = config["sensing_port"].as<int>();
            if (config["default_ip"]) cfg.default_ip = config["default_ip"].as<std::string>();
            if (config["profiling_modules"]) cfg.profiling_modules = config["profiling_modules"].as<std::string>();
            if (config["pilot_positions"]) cfg.pilot_positions = config["pilot_positions"].as<std::vector<size_t>>();
            if (config["cpu_cores"]) cfg.available_cores = config["cpu_cores"].as<std::vector<size_t>>();
            return true;
        } catch (const YAML::Exception& e) {
            std::cerr << "Error parsing YAML config: " << e.what() << std::endl;
            return false;
        }
    };
    
    // Configure default parameters
    Config cfg;
    cfg.fft_size = 1024;
    cfg.cp_length = 128;
    cfg.sync_pos = 1;
    cfg.sample_rate = 50e6;
    cfg.bandwidth = 50e6;
    cfg.center_freq = 2.4e9;
    cfg.tx_gain = 30.0;
    cfg.rx_gain = 30.0;
    cfg.rx_channel = 1;
    cfg.zc_root = 29;
    cfg.pilot_positions = {571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451};
    cfg.num_symbols = 100;
    cfg.mono_sensing_ip = "";

    // Add command line argument parsing
    std::string config_file = default_config_file;
    std::string save_config = "";
    
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("config,c", po::value<std::string>(&config_file)->default_value(default_config_file), "Config file path (default: Modulator.yaml)")
        ("save-config,s", po::value<std::string>(&save_config)->implicit_value(""), "Save current config to file and exit (optionally specify filename)")
        ("default-ip", po::value<std::string>(&cfg.default_ip), "Default IP for all services (default: 127.0.0.1)")
        ("args", po::value<std::string>(&cfg.device_args), "USRP device arguments")
        ("fft-size", po::value<size_t>(&cfg.fft_size), "FFT size (default: 1024)")
        ("cp-length", po::value<size_t>(&cfg.cp_length), "CP length (default: 128)")
        ("sync-pos", po::value<size_t>(&cfg.sync_pos), "Sync position (default: 1)")
        ("sample-rate", po::value<double>(&cfg.sample_rate), "Sample rate (default: 50e6)")
        ("bandwidth", po::value<double>(&cfg.bandwidth), "Bandwidth (default: 50e6)")
        ("center-freq", po::value<double>(&cfg.center_freq), "Center frequency (default: 2.4e9)")
        ("tx-gain", po::value<double>(&cfg.tx_gain), "TX gain (default: 20)")
        ("rx-gain", po::value<double>(&cfg.rx_gain), "RX gain (default: 30)")
        ("rx-channel", po::value<size_t>(&cfg.rx_channel), "RX channel (default: 1)")
        ("zc-root", po::value<int>(&cfg.zc_root), "ZC root (default: 29)")
        ("num-symbols", po::value<size_t>(&cfg.num_symbols), "Number of symbols per frame (default: 100)")
        ("tx-frame-buffer-size", po::value<size_t>(&cfg.tx_frame_buffer_size), "TX frame buffer size (default: 8)")
        ("tx-symbols-buffer-size", po::value<size_t>(&cfg.tx_symbols_buffer_size), "TX symbols buffer size (default: 8)")
        ("rx-frame-buffer-size", po::value<size_t>(&cfg.rx_frame_buffer_size), "RX frame buffer size (default: 8)")
        ("clock-source", po::value<std::string>(&cfg.clocksource), "Clock source (default: external)")
        ("system-delay",po::value<int32_t>(&cfg.system_delay), "System delay in samples (default: 63)")
        ("wire-format-tx", po::value<std::string>(&cfg.wire_format_tx), "TX wire format (default: sc16)")
        ("wire-format-rx", po::value<std::string>(&cfg.wire_format_rx), "RX wire format (default: sc16)")
        ("mod-udp-ip", po::value<std::string>(&cfg.modulator_udp_ip), "Modulator UDP bind IP (default: 0.0.0.0)")
        ("mod-udp-port", po::value<int>(&cfg.modulator_udp_port), "Modulator UDP bind port (default: 50000)")
        ("sensing-ip", po::value<std::string>(&cfg.mono_sensing_ip), "Sensing destination IP")
        ("sensing-port", po::value<int>(&cfg.mono_sensing_port), "Sensing destination port (default: 8888)")
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

    auto clamp_buffer_size = [](size_t& value, const char* name) {
        constexpr size_t kDefaultBufferSize = 8;
        if (value == 0) {
            std::cerr << "Warning: " << name
                      << " is 0; falling back to default value "
                      << kDefaultBufferSize << "." << std::endl;
            value = kDefaultBufferSize;
        }
    };
    clamp_buffer_size(cfg.tx_frame_buffer_size, "tx_frame_buffer_size");
    clamp_buffer_size(cfg.tx_symbols_buffer_size, "tx_symbols_buffer_size");
    clamp_buffer_size(cfg.rx_frame_buffer_size, "rx_frame_buffer_size");

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

    if (cfg.mono_sensing_ip.empty()) cfg.mono_sensing_ip = cfg.default_ip;

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

    // Set main thread affinity (use the last available core)
    if (!cfg.available_cores.empty()) {
        std::vector<size_t> cpu_list = {cfg.available_cores.back()};
        uhd::set_thread_affinity(cpu_list);
    }

    // Load FFTW wisdom
    FFTWManager::import_wisdom();

    // Create and start engine
    OFDMISACEngine isac_engine(cfg);
    isac_engine.start();

    // Main loop
    while (!stop_signal.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Stop processing
    isac_engine.stop();
    
    // Save FFTW wisdom
    FFTWManager::export_wisdom();
    std::cout << "\nTransmission and sensing stopped.\n";
    return 0;
}
