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
#include <immintrin.h>
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
#include <functional>
#include <unordered_map>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

using AlignedVector = std::vector<std::complex<float>, AlignedAllocator<std::complex<float>>>;
using AlignedIntVector = std::vector<int, AlignedAllocator<int>>;
using SymbolVector = std::vector<AlignedVector>; 

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
        _circular_buffer(8),
        _symbols_buffer(8),
        _rx_frame_buffer(8),            
        _data_packet_buffer(32),                // Initialize data packet buffer
        _accumulated_rx_symbols(),
        _accumulated_tx_symbols(),
        _fft_in(cfg.fft_size),               
        _fft_out(cfg.fft_size),
        _demod_fft_in(cfg.fft_size),
        _demod_fft_out(cfg.fft_size),
        _channel_response_buffer(cfg.range_fft_size * cfg.doppler_fft_size, std::complex<float>(0.0f, 0.0f)),
        _blank_frame(cfg.samples_per_frame(), 0.0f),
        tx_data_file("tx_frame.bin", std::ios::binary),
        _sensing_sender(cfg.mono_sensing_ip, cfg.mono_sensing_port, cfg.fft_size * cfg.num_symbols),
        _control_handler(9999)  // Initialize control handler
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
        _mti_filter.resize(_cfg.range_fft_size);
    }

    ~OFDMISACEngine() {
        stop();
        fftwf_destroy_plan(_ifft_plan);
        fftwf_destroy_plan(_range_ifft_plan);
        fftwf_destroy_plan(_doppler_fft_plan);
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
    AlignedVector _channel_response_buffer;  // Channel response buffer
    fftwf_plan _demod_fft_plan = nullptr; // Demodulation FFT plan
    fftwf_plan _range_ifft_plan = nullptr;   // Batch Range IFFT plan
    fftwf_plan _doppler_fft_plan = nullptr;   // Doppler FFT plan
    
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
    // Hamming windows
    AlignedVector _range_window;    // Range processing window (Frequency domain)
    AlignedVector _doppler_window;  // Doppler processing window (Time domain)
    size_t _global_symbol_index = 0; // Global symbol index (for sensing processing)
    
    // LDPC encoding and scrambling
    LDPCCodec _ldpc{LDPCCodec::LDPCConfig()};
    Scrambler scrambler{201600, 0x5A};
    
    // MTI Filter
    MTIFilter _mti_filter;

    
    // UDP socket
    int _udp_sock{-1};
    struct sockaddr_in _udp_addr{};
    
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
            FFTW_MEASURE | FFTW_PATIENT
        );
        
        // Create demodulation FFT plan (for sensing)
        _demod_fft_plan = fftwf_plan_dft_1d(
            static_cast<int>(_cfg.fft_size),
            reinterpret_cast<fftwf_complex*>(_demod_fft_in.data()),
            reinterpret_cast<fftwf_complex*>(_demod_fft_out.data()),
            FFTW_FORWARD,
            FFTW_MEASURE
        );
        
        // Create batch Range IFFT plan (for sensing)
        _channel_response_buffer.resize(_cfg.range_fft_size * _cfg.doppler_fft_size, std::complex<float>(0.0f, 0.0f));
        
        // Convert size_t to int
        const int fft_size_int = static_cast<int>(_cfg.range_fft_size);
        const int doppler_fft_size_int = static_cast<int>(_cfg.doppler_fft_size);
        
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
    }

    // Initialize Hamming windows
    void _init_hamming_windows() {
        // Range processing window (frequency domain, applied to each subcarrier)
        _range_window.resize(_cfg.fft_size);
        for (size_t i = 0; i < _cfg.fft_size; ++i) {
            // Hamming window formula: w(n) = 0.54 - 0.46*cos(2πn/(N-1))
            _range_window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (_cfg.fft_size - 1));
        }
        
        // Doppler processing window (Time domain, applied to each symbol)
        _doppler_window.resize(_cfg.sensing_symbol_num);
        for (size_t i = 0; i < _cfg.sensing_symbol_num; ++i) {
            _doppler_window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (_cfg.sensing_symbol_num - 1));
        }
    }

    void _prepare_zc_sequence() {
        _zc_seq.resize(_cfg.fft_size);

        const int N = _cfg.fft_size;
        const int q = _cfg.zc_root;

        // delta: even N -> 0, odd N -> 1
        const int delta = (N & 1);

        // Pre-calculate constant coefficients, use double for more stable phase calculation
        const double base = -M_PI * static_cast<double>(q) / static_cast<double>(N);

        #pragma omp simd
        for (int n = 0; n < N; ++n) {
            const double nd  = static_cast<double>(n);
            const double arg = nd * (nd + static_cast<double>(delta)); // n*(n+delta)
            const double phase = base * arg;                           // -π*q/N * n*(n+δ)
            _zc_seq[n] = std::polar(1.0f, static_cast<float>(phase));  // unit-modulus complex
        }
    }

    void _precalc_positions() {
        _symbol_positions.reserve(_cfg.num_symbols);
        for (size_t i = 0; i < _cfg.num_symbols; ++i) {
            _symbol_positions.push_back(i * (_cfg.fft_size + _cfg.cp_length));
        }
    }

    inline std::complex<float> _qpsk_mod(int x) const {
        x &= 3;
        const float sqrt_2_inv = 1/std::sqrt(2.0f);
        return {
            (x & 2) ? -sqrt_2_inv : sqrt_2_inv, // Real
            (x & 1) ? -sqrt_2_inv : sqrt_2_inv // Imaginary
        };
    }

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
            LDPCCodec::AlignedIntVector header_coded_bits;
            _ldpc.encode_frame(header_bytes, header_coded_bits);
            
            // Scramble header
            scrambler.scramble(header_coded_bits);
            
            // Convert to QPSK symbols
            LDPCCodec::AlignedIntVector header_qpsk_ints;
            _pack_bits_qpsk(header_coded_bits, header_qpsk_ints);
            
            // Process payload data
            // Construct: Pure UDP data + padding(0x00) to make length a multiple of bytes_per_ldpc_block
            LDPCCodec::AlignedByteVector input_bytes(padded_len, 0x00);
            // Direct copy UDP payload
            std::memcpy(input_bytes.data(), udp_buf.data(), std::min<size_t>(recv_len, udp_buf.size()));
            // Remaining padding stays 0x00


            // LDPC encode whole packet (may contain multiple LDPC blocks)
            LDPCCodec::AlignedIntVector encoded_bits_all;
            _ldpc.encode_frame(input_bytes, encoded_bits_all); // bits: (#blocks*bytes_per_ldpc_block*8)

            // Batch scramble using QPSKScrambler
            scrambler.scramble(encoded_bits_all);

            // Convert to QPSK symbol integers
            LDPCCodec::AlignedIntVector qpsk_ints_all;
            _pack_bits_qpsk(encoded_bits_all, qpsk_ints_all); // (#blocks*bytes_per_ldpc_block*8) ints

            // Construct final output: LDPC encoded header + encoded payload data
            AlignedIntVector packet;
            packet.reserve(header_qpsk_ints.size() + qpsk_ints_all.size());
            packet.insert(packet.end(), header_qpsk_ints.begin(), header_qpsk_ints.end());
            packet.insert(packet.end(), qpsk_ints_all.begin(), qpsk_ints_all.end());
            // Print debug info
            // std::cout << "Packet constructed with size: " << packet.size() << std::endl;
            // Enqueue
            {
                std::unique_lock<std::mutex> lock(_data_mutex);
                _data_not_full.wait(lock,[this]{return !_running.load() || !_data_packet_buffer.full();});
                if (!_running.load()) break;
                _data_packet_buffer.push_back(packet);
                _data_not_empty.notify_one();
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
        // Send heartbeat for NAT traversal (once per second)
        static auto next_hb_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (now >= next_hb_time) {
            _control_handler.send_heartbeat(_cfg.mono_sensing_ip, _cfg.mono_sensing_port);
            next_hb_time = now + std::chrono::seconds(1);
        }

        static auto next_send_time = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() >= next_send_time) {
            // 1. Perform OFDM demodulation on received frame
            std::vector<AlignedVector> rx_symbols;
            rx_symbols.reserve(_cfg.sensing_symbol_num);
            for (size_t i = 0; i < _cfg.sensing_symbol_num; ++i) {
                std::copy(frame.rx_symbols[i].begin(), frame.rx_symbols[i].end(), _demod_fft_in.begin());
                // Execute FFT
                fftwf_execute(_demod_fft_plan);
                // Copy to channel response buffer
                auto* dest = _channel_response_buffer.data() + i * _cfg.range_fft_size;
                std::copy(_demod_fft_out.begin(), _demod_fft_out.end(), dest);
            }
            
            // 2. Channel estimation (Frequency domain division)
            #pragma omp simd
            for (size_t i = 0; i < _cfg.sensing_symbol_num; ++i) {
                for (size_t k = 0; k < _cfg.fft_size; ++k) {
                    size_t idx = i * _cfg.range_fft_size + k;
                    _channel_response_buffer[idx] /= frame.tx_symbols[i][k];
                }
            }

            for (size_t i = 0; i < _cfg.sensing_symbol_num; ++i) {
                auto* symbol_data = _channel_response_buffer.data() + i * _cfg.range_fft_size;
                const size_t half_size = _cfg.fft_size / 2;
                
                // In-place FFT shift - swap front and back halves
                #pragma omp simd
                for (size_t j = 0; j < half_size; ++j) {
                    std::swap(symbol_data[j], symbol_data[j + half_size]);
                }
            }

            // Apply MTI Filter (Moving Target Indication)
            // Applied after channel estimation and FFT shift, before windowing and IFFT
            if (enable_mti.load()) {
                _mti_filter.apply(_channel_response_buffer, _cfg.fft_size, _cfg.sensing_symbol_num);
            }

            if (!skip_sensing_fft.load()) {

                for (size_t i = 0; i < _cfg.sensing_symbol_num; ++i) {
                    auto* symbol_data = _channel_response_buffer.data() + i * _cfg.range_fft_size;
                    #pragma omp simd
                    for (size_t j = 0; j < _cfg.fft_size; ++j) {
                        // Apply range window (Frequency domain windowing)
                        symbol_data[j] *= _range_window[j];
                    }
                }

                for (size_t bin = 0; bin < _cfg.fft_size; ++bin) {
                    #pragma omp simd
                    for (size_t i = 0; i < _cfg.sensing_symbol_num; ++i) {
                        size_t idx = i * _cfg.range_fft_size + bin;
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
            if(_cfg.range_fft_size != _cfg.fft_size || _cfg.doppler_fft_size != _cfg.fft_size) {
                _channel_response_buffer.assign(_cfg.range_fft_size * _cfg.doppler_fft_size, std::complex<float>(0.0f, 0.0f));
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

        const float scale = 1.0f / std::sqrt(_cfg.fft_size)/4; // Output digital signal power is 1/16

        // Pre-calculate pilot and data subcarrier indices to reduce branches inside loop
        std::vector<char> is_pilot(_cfg.fft_size, 0);
        for (auto pos : _cfg.pilot_positions) if (pos < _cfg.fft_size) is_pilot[pos] = 1;
        std::vector<int> data_subcarriers; data_subcarriers.reserve(_cfg.fft_size - _cfg.pilot_positions.size());
        for (size_t k=0;k<_cfg.fft_size;++k) if (!is_pilot[k]) data_subcarriers.push_back((int)k);

        while (_running.load()) {
            frame_start = Clock::now(); // Record frame start time

            AlignedVector current_frame(_cfg.samples_per_frame());
            SymbolVector current_symbols;  
            current_symbols.reserve(_cfg.num_symbols);
            
            size_t frame_data_index = 0; // Entire frame data index
            // Pool of real data symbols available for this frame
            AlignedIntVector data_pool;
            {
                std::unique_lock<std::mutex> lock(_data_mutex);
                size_t total = 0;
                for (auto &pkt : _data_packet_buffer) total += pkt.size();
                data_pool.reserve(total);

                while (!_data_packet_buffer.empty()) {
                    const auto &pkt = _data_packet_buffer.front();
                    data_pool.insert(data_pool.end(), pkt.begin(), pkt.end());
                    _data_packet_buffer.pop_front();
                }
                _data_not_full.notify_all();
            }
            size_t data_pool_pos = 0;

            for (size_t i = 0; i < _cfg.num_symbols; ++i) {
                const size_t pos = _symbol_positions[i];
                auto* buf_ptr = current_frame.data() + pos;

                if (i == _cfg.sync_pos) {
                    // Sync symbol: Use ZC sequence for the whole symbol
                    std::copy(_zc_seq.begin(), _zc_seq.end(), _fft_in.begin());
                } else {
                    // Non-sync symbol: Use real data symbols from data_pool, fallback to pre-generated data if exhausted
                    // Fill all pilots first (vectorization friendly)
                    #pragma omp simd
                    for (size_t idx=0; idx<_cfg.pilot_positions.size(); ++idx) {
                        size_t k = _cfg.pilot_positions[idx];
                        if (k < _cfg.fft_size) _fft_in[k] = _zc_seq[k];
                    }
                    // Fill data subcarriers
                    const size_t ds_count = data_subcarriers.size();
                    #pragma omp simd
                    for (size_t di=0; di<ds_count; ++di) {
                        int k = data_subcarriers[di];
                        int sym = (data_pool_pos < data_pool.size()) ? data_pool[data_pool_pos++] : _pregen_data[frame_data_index++];
                        _fft_in[k] = _qpsk_mod(sym);
                    }
                }
                
                // Save frequency domain data of current symbol (for sensing)
                current_symbols.push_back(_fft_in);

                // IFFT transform
                fftwf_execute(_ifft_plan);

                // Add Cyclic Prefix (CP)
                #pragma omp simd
                for (size_t j = 0; j < _cfg.cp_length; ++j) {
                    buf_ptr[j] = _fft_out[_cfg.fft_size - _cfg.cp_length + j] * scale;
                }
                
                // Write symbol body
                #pragma omp simd
                for (size_t j = 0; j < _cfg.fft_size; ++j) {
                    buf_ptr[_cfg.cp_length + j] = _fft_out[j] * scale;
                }
            }
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
            } else {
                _tx_stream->send(_blank_frame.data(), _blank_frame.size(), md);
                // std::cout << "No frame to send, sending blank frame.\n";
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
        AlignedVector aligned_frame(_cfg.samples_per_frame());
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
        AlignedVector rx_frame(_cfg.samples_per_frame());
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
        
        while (_running.load()) {
            SymbolVector tx_symbols;
            AlignedVector rx_frame_data;
            bool has_frame = false;
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
            
            if (has_frame) {
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
                    _accumulated_tx_symbols.push_back(std::move(tx_symbols[symbol_idx]));
                    _global_symbol_index += shadow_sensing_symbol_stride;
                    
                    // Check if processing threshold is reached
                    if (_accumulated_tx_symbols.size() >= _cfg.sensing_symbol_num) {
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
};

// --- Global Signal Handling ---
std::atomic<bool> stop_signal(false);
void signal_handler(int) { stop_signal.store(true); }

int UHD_SAFE_MAIN(int argc, char *argv[]) {
    std::signal(SIGINT, &signal_handler);
    uhd::set_thread_priority_safe();
    
    // Configure parameters
    Config cfg;
    cfg.fft_size = 1024;
    cfg.cp_length = 128;
    cfg.sync_pos = 1;
    cfg.sample_rate = 50e6;
    cfg.center_freq = 2.4e9;
    cfg.tx_gain = 30.0;
    cfg.rx_gain = 30.0;
    cfg.rx_channel = 1;
    cfg.zc_root = 29;
    cfg.pilot_positions = {571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451};
    cfg.num_symbols = 100; // Symbols per frame
    cfg.mono_sensing_ip = "";

    // Add command line argument parsing
    std::string default_ip = "127.0.0.1";
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("default-ip", po::value<std::string>(&default_ip)->default_value("127.0.0.1"), "Default IP for all services")
        ("args", po::value<std::string>(&cfg.device_args)->default_value("addr=192.168.40.2, master_clock_rate=200e6, num_recv_frames=512, num_send_frames=512"), "USRP device arguments")
        ("fft-size", po::value<size_t>(&cfg.fft_size)->default_value(1024), "FFT size")
        ("cp-length", po::value<size_t>(&cfg.cp_length)->default_value(128), "CP length")
        ("sync-pos", po::value<size_t>(&cfg.sync_pos)->default_value(1), "Sync position")
        ("sample-rate", po::value<double>(&cfg.sample_rate)->default_value(50e6), "Sample rate")
        ("bandwidth", po::value<double>(&cfg.bandwidth)->default_value(50e6), "Bandwidth")
        ("center-freq", po::value<double>(&cfg.center_freq)->default_value(2.4e9), "Center frequency")
        ("tx-gain", po::value<double>(&cfg.tx_gain)->default_value(20), "TX gain")
        ("rx-gain", po::value<double>(&cfg.rx_gain)->default_value(30), "RX gain")
        ("rx-channel", po::value<size_t>(&cfg.rx_channel)->default_value(1), "RX channel")
        ("zc-root", po::value<int>(&cfg.zc_root)->default_value(29), "ZC root")
        ("num-symbols", po::value<size_t>(&cfg.num_symbols)->default_value(100), "Number of symbols per frame")
        ("clock-source", po::value<std::string>(&cfg.clocksource)->default_value("external"), "Clock source (internal or external)")
        ("system-delay",po::value<int32_t>(&cfg.system_delay)->default_value(63), "System delay in samples (for alignment)")
        ("wire-format-tx", po::value<std::string>(&cfg.wire_format_tx)->default_value("sc16"), "TX wire format (sc8 or sc16)")
        ("wire-format-rx", po::value<std::string>(&cfg.wire_format_rx)->default_value("sc16"), "RX wire format (sc8 or sc16)")
        ("mod-udp-ip", po::value<std::string>(&cfg.modulator_udp_ip)->default_value("0.0.0.0"), "Modulator UDP bind IP for incoming payloads")
        ("mod-udp-port", po::value<int>(&cfg.modulator_udp_port)->default_value(50000), "Modulator UDP bind port for incoming payloads")
        ("sensing-ip", po::value<std::string>(&cfg.mono_sensing_ip), "Sensing destination IP")
        ("sensing-port", po::value<int>(&cfg.mono_sensing_port)->default_value(8888), "Sensing destination port")
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

    if (cfg.mono_sensing_ip.empty()) cfg.mono_sensing_ip = default_ip;

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

    // Set main thread affinity (use the last available core)
    if (!cfg.available_cores.empty()) {
        std::vector<size_t> cpu_list = {cfg.available_cores.back()};
        uhd::set_thread_affinity(cpu_list);
    }

    // Create and start engine
    OFDMISACEngine isac_engine(cfg);
    isac_engine.start();

    // Main loop
    while (!stop_signal.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Stop processing
    isac_engine.stop();
    std::cout << "\nTransmission and sensing stopped.\n";
    return 0;
}