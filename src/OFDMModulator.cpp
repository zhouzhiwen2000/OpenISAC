#include <uhd/utils/thread.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <uhd/types/tune_request.hpp>
#include <fftw3.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <atomic>
#include <csignal>
#include <random>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <deque>
#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <sys/mman.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <Common.hpp>
#include "OFDMCore.hpp"
#include "SensingChannel.hpp"
#include <functional>
#include <unordered_map>
#include <memory>

// Type aliases are now defined in Common.hpp

struct QueuedTxFrame {
    AlignedVector samples;
    std::shared_ptr<const SymbolVector> symbols;
    int64_t ingest_time_ns{0};
    int64_t encoded_time_ns{0};
    int64_t mod_done_time_ns{0};
};

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
    explicit OFDMISACEngine(const Config& cfg)
      : _cfg(cfg),
        _gen(std::random_device{}()),
        _dist(0, 3),
        _fft_in(cfg.fft_size),
        _fft_out(cfg.fft_size),
        _blank_frame(cfg.samples_per_frame(), 0.0f),
        tx_data_file("tx_frame.bin", std::ios::binary),
        _control_handler(cfg.control_port),
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
        _data_packet_pool(64, []() {
            return AlignedIntVector();
        }),
        _circular_buffer(cfg.tx_circular_buffer_size),
        _data_packet_buffer(cfg.data_packet_buffer_size),
        _data_packet_ingest_ts(cfg.data_packet_buffer_size)
    {
        _shared_sensing_cfg.sensing_symbol_stride = cfg.sensing_symbol_stride;
        _shared_sensing_cfg.enable_mti = true;
        _shared_sensing_cfg.skip_sensing_fft = true;
        _shared_sensing_cfg.generation = 1;
        _shared_sensing_cfg.apply_symbol_index = 0;

        _build_sensing_channels();
        _init_fftw();
        _prepare_zc_sequence();
        _init_usrp();
        _precalc_positions();

        size_t total_data_samples = (cfg.num_symbols - 1) * (cfg.fft_size - cfg.pilot_positions.size());
        _pregen_data.resize(total_data_samples);
        for (auto& x : _pregen_data) {
            x = _dist(_gen);
        }

        _register_commands();
    }

    ~OFDMISACEngine() {
        stop();
        if (_ifft_plan != nullptr) {
            fftwf_destroy_plan(_ifft_plan);
            _ifft_plan = nullptr;
        }
    }

    void start() {
        _control_handler.start();
        _running.store(true);
        _next_frame_start_symbol.store(0, std::memory_order_relaxed);
        _next_tx_frame_seq.store(0, std::memory_order_relaxed);

        _mod_thread = std::thread(&OFDMISACEngine::_modulation_proc, this);
        _data_thread = std::thread(&OFDMISACEngine::_data_ingest_proc, this);

        // Let modulation prefill as much as possible (prefer full queue) before scheduling TX/RX start.
        bool prefilled_full = false;
        const auto prefill_deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(1200);
        SPSCBackoff prefill_backoff;
        while (_running.load(std::memory_order_relaxed)) {
            prefilled_full = _circular_buffer.full();
            if (prefilled_full || std::chrono::steady_clock::now() >= prefill_deadline) {
                break;
            }
            prefill_backoff.pause();
        }
        const size_t prefilled_frames = _circular_buffer.size();
        const size_t prefill_capacity = _circular_buffer.capacity();
        LOG_G_INFO() << "[Start] Modulation prefilled "
                  << prefilled_frames << "/" << prefill_capacity
                  << " frames.";
        if (!prefilled_full && prefilled_frames < prefill_capacity) {
            LOG_G_WARN() << "[Start] modulation prefill timeout before buffer became full, "
                         << "TX may underflow at startup.";
        }

        if (_tx_usrp) {
            const double now_s = _tx_usrp->get_time_now().get_real_secs();
            const double scheduled_start_s = std::ceil(now_s) + 1.0;
            _start_time = uhd::time_spec_t(scheduled_start_s);
            LOG_G_INFO() << std::fixed << std::setprecision(6)
                      << "[Start] Scheduled unified TX/RX start_time="
                      << scheduled_start_s << " s after prefill."
                      << std::defaultfloat;
        }

        _tx_thread = std::thread(&OFDMISACEngine::_tx_proc, this);

        for (auto& ch : _sensing_channels) {
            ch->start(_start_time);
        }
    }

    void stop() {
        if (!_running.exchange(false)) {
            _control_handler.stop();
            return;
        }

        _control_handler.stop();

        for (auto& ch : _sensing_channels) {
            ch->stop();
        }

        if (_mod_thread.joinable()) _mod_thread.join();
        if (_tx_thread.joinable()) _tx_thread.join();
        if (_data_thread.joinable()) _data_thread.join();

        for (auto& ch : _sensing_channels) {
            ch->join();
        }
    }

private:
    Config _cfg;
    uhd::usrp::multi_usrp::sptr _tx_usrp;
    uhd::tx_streamer::sptr _tx_stream;
    size_t _tx_chunk_samps = 0;

    // Random number generator
    std::mt19937 _gen;
    std::uniform_int_distribution<> _dist;
    
    // FFTW resources
    AlignedVector _fft_in;               // IFFT input
    AlignedVector _fft_out;              // IFFT output
    fftwf_plan _ifft_plan = nullptr;     // IFFT plan
    
    // Zadoff-Chu sequence
    AlignedVector _zc_seq;               
    const AlignedVector _blank_frame;

    // Pre-calculated symbol positions
    std::vector<size_t> _symbol_positions; 
    
    // Thread control
    std::atomic<bool> _running{false};
    std::thread _mod_thread;             // Modulation thread
    std::thread _tx_thread;              // TX thread
    
    uhd::time_spec_t _start_time{0.0}; 

    // Pre-generated data and file
    std::ofstream tx_data_file;
    std::atomic<bool> data_saved{false};
    AlignedIntVector _pregen_data;
    // Data ingest/encoding related
    std::thread _data_thread;
    
    // Control handler
    ControlCommandHandler _control_handler;
    std::atomic<int64_t> _align_target_channel{-1}; // -1 means all channels
    SharedSensingRuntime _shared_sensing_cfg;
    std::mutex _shared_sensing_cfg_mutex;
    
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
    ObjectPool<AlignedIntVector> _data_packet_pool; // Pool for data packet buffers
    // Ring buffers must be destroyed before the pools because queued items can
    // hold shared_ptr deleters that return objects back into these pools.
    SPSCRingBuffer<QueuedTxFrame> _circular_buffer;
    SPSCRingBuffer<AlignedIntVector> _data_packet_buffer; // Data packet buffer
    struct PktTimestamps { int64_t ingest_ns{0}; int64_t encoded_ns{0}; };
    struct LatencyAccumulator {
        std::atomic<int64_t> ldpc_total_ns{0};
        std::atomic<int64_t> mod_total_ns{0};
        std::atomic<int64_t> tx_wait_total_ns{0};
        std::atomic<int64_t> e2e_total_ns{0};
        std::atomic<int> count{0};
    };
    struct LatencySnapshot {
        int64_t ldpc_total_ns{0};
        int64_t mod_total_ns{0};
        int64_t tx_wait_total_ns{0};
        int64_t e2e_total_ns{0};
        int count{0};
    };
    SPSCRingBuffer<PktTimestamps> _data_packet_ingest_ts; // Timestamps paired with _data_packet_buffer
    LatencyAccumulator _latency_accumulator;
    std::vector<std::unique_ptr<SensingChannel>> _sensing_channels;
    std::atomic<uint64_t> _next_frame_start_symbol{0};
    std::atomic<uint64_t> _next_tx_frame_seq{0};

    LatencySnapshot _take_latency_snapshot_and_reset() {
        LatencySnapshot snapshot;
        snapshot.count = _latency_accumulator.count.exchange(0, std::memory_order_acq_rel);
        snapshot.ldpc_total_ns = _latency_accumulator.ldpc_total_ns.exchange(0, std::memory_order_acq_rel);
        snapshot.mod_total_ns = _latency_accumulator.mod_total_ns.exchange(0, std::memory_order_acq_rel);
        snapshot.tx_wait_total_ns = _latency_accumulator.tx_wait_total_ns.exchange(0, std::memory_order_acq_rel);
        snapshot.e2e_total_ns = _latency_accumulator.e2e_total_ns.exchange(0, std::memory_order_acq_rel);
        return snapshot;
    }

    std::shared_ptr<SymbolVector> _acquire_symbol_frame() {
        auto* raw = new SymbolVector(_symbols_pool.acquire());
        return std::shared_ptr<SymbolVector>(raw, [this](SymbolVector* p) {
            _symbols_pool.release(std::move(*p));
            delete p;
        });
    }

    void _build_sensing_channels() {
        _sensing_channels.clear();
        _sensing_channels.reserve(_cfg.sensing_rx_channels.size());
        for (uint32_t i = 0; i < _cfg.sensing_rx_channels.size(); ++i) {
            auto channel = std::make_unique<SensingChannel>(
                _cfg,
                _cfg.sensing_rx_channels[i],
                i,
                _running,
                [this](const std::string& ip, int port) {
                    _control_handler.send_heartbeat(ip, port);
                },
                [this](size_t hint) {
                    return core_from_hint(_cfg, hint);
                }
            );
            channel->apply_shared_cfg(_shared_sensing_cfg);
            _sensing_channels.push_back(std::move(channel));
        }
    }

    void _schedule_shared_sensing_update(std::function<void(SharedSensingRuntime&)> updater) {
        SharedSensingRuntime snapshot;
        {
            std::lock_guard<std::mutex> lock(_shared_sensing_cfg_mutex);
            updater(_shared_sensing_cfg);
            _shared_sensing_cfg.generation++;
            const uint64_t next_symbol = _next_frame_start_symbol.load(std::memory_order_relaxed);
            if (_cfg.num_symbols == 0) {
                _shared_sensing_cfg.apply_symbol_index = next_symbol;
            } else {
                const uint64_t frame_len = static_cast<uint64_t>(_cfg.num_symbols);
                const uint64_t boundary = ((next_symbol + frame_len - 1) / frame_len) * frame_len;
                _shared_sensing_cfg.apply_symbol_index = boundary;
            }
            snapshot = _shared_sensing_cfg;
        }
        for (auto& ch : _sensing_channels) {
            ch->apply_shared_cfg(snapshot);
        }
    }

    void _set_alignment_for_channel(uint32_t ch_id, int32_t value) {
        if (ch_id >= _sensing_channels.size()) {
            LOG_G_WARN() << "Invalid sensing channel id for ALGN: " << ch_id;
            return;
        }
        _sensing_channels[ch_id]->set_alignment(value);
    }

    void _set_rx_gain_for_channel(uint32_t ch_id, double gain_db) {
        if (ch_id >= _sensing_channels.size()) {
            LOG_G_WARN() << "Invalid sensing channel id for RXGN: " << ch_id;
            return;
        }
        _sensing_channels[ch_id]->set_rx_gain(gain_db, nullptr);
    }

    void _register_commands() {
        _control_handler.register_command("ALCH", [this](int32_t value) {
            if (value < 0) {
                _align_target_channel.store(-1);
                LOG_G_INFO() << "ALCH set to ALL channels";
                return;
            }
            _align_target_channel.store(static_cast<int64_t>(value));
            LOG_G_INFO() << "ALCH set to channel " << value;
        });

        _control_handler.register_command("ALGN", [this](int32_t value) {
            const int64_t max_adjust = static_cast<int64_t>(_cfg.samples_per_frame());
            const int64_t clamped_value = std::clamp<int64_t>(
                static_cast<int64_t>(value),
                -max_adjust,
                max_adjust
            );
            const int32_t adjusted_value = static_cast<int32_t>(clamped_value);
            const int64_t target = _align_target_channel.load();
            if (target < 0) {
                for (uint32_t i = 0; i < _sensing_channels.size(); ++i) {
                    _set_alignment_for_channel(i, adjusted_value);
                }
                return;
            }
            _set_alignment_for_channel(static_cast<uint32_t>(target), adjusted_value);
        });

        _control_handler.register_command("SKIP", [this](int32_t value) {
            const bool new_skip = (value != 0);
            _schedule_shared_sensing_update([new_skip](SharedSensingRuntime& cfg) {
                cfg.skip_sensing_fft = new_skip;
            });
            LOG_G_INFO() << "Received SKIP command: " << (new_skip ? 1 : 0);
        });

        _control_handler.register_command("STRD", [this](int32_t value) {
            size_t stride = value <= 0 ? 1 : static_cast<size_t>(value);
            _schedule_shared_sensing_update([stride](SharedSensingRuntime& cfg) {
                cfg.sensing_symbol_stride = stride;
            });
            LOG_G_INFO() << "Received STRD command: " << stride;
        });

        _control_handler.register_command("MTI ", [this](int32_t value) {
            const bool new_mti = (value != 0);
            _schedule_shared_sensing_update([new_mti](SharedSensingRuntime& cfg) {
                cfg.enable_mti = new_mti;
            });
            LOG_G_INFO() << "Received MTI command: " << (new_mti ? "Enable" : "Disable");
        });

        _control_handler.register_command("TXGN", [this](int32_t value) {
            if (!_tx_usrp) {
                LOG_G_WARN() << "TXGN ignored: USRP not initialized";
                return;
            }
            const double requested_gain = static_cast<double>(value) / 10.0;
            const auto gain_range = _tx_usrp->get_tx_gain_range(_cfg.tx_channel);
            const double clamped_gain = std::clamp(requested_gain, gain_range.start(), gain_range.stop());
            _cfg.tx_gain = clamped_gain;
            _tx_usrp->set_tx_gain(clamped_gain, _cfg.tx_channel);
            LOG_G_INFO() << "Received TXGN command: " << requested_gain
                         << " dB (applied " << clamped_gain << " dB)";
        });

        _control_handler.register_command("RXGN", [this](int32_t value) {
            if (_sensing_channels.empty()) {
                LOG_G_WARN() << "RXGN ignored: no sensing RX channels configured";
                return;
            }

            const double requested_gain = static_cast<double>(value) / 10.0;
            const int64_t target = _align_target_channel.load();

            auto apply_one = [&](uint32_t ch_id) {
                if (ch_id >= _sensing_channels.size()) {
                    LOG_G_WARN() << "Invalid target channel for RXGN: " << ch_id;
                    return;
                }
                _set_rx_gain_for_channel(ch_id, requested_gain);
            };

            if (target < 0) {
                for (uint32_t i = 0; i < _sensing_channels.size(); ++i) {
                    apply_one(i);
                }
                return;
            }

            apply_one(static_cast<uint32_t>(target));
        });
    }

    void _init_usrp() {
        const std::string tx_device_args = _cfg.tx_device_args.empty() ? _cfg.device_args : _cfg.tx_device_args;
        const std::string tx_clock_source = _cfg.tx_clock_source.empty() ? _cfg.clocksource : _cfg.tx_clock_source;
        const std::string tx_time_source = _cfg.tx_time_source.empty() ?
            (_cfg.timesource.empty() ? tx_clock_source : _cfg.timesource) :
            _cfg.tx_time_source;

        _tx_usrp = uhd::usrp::multi_usrp::make(tx_device_args);
        _tx_usrp->set_clock_source(tx_clock_source);
        _tx_usrp->set_time_source(tx_time_source);
        _tx_usrp->set_tx_rate(_cfg.sample_rate);

        const size_t usrp_tx_channels = _tx_usrp->get_tx_num_channels();
        if (_cfg.tx_channel >= usrp_tx_channels) {
            throw std::runtime_error(
                "Configured TX channel out of range: " +
                std::to_string(_cfg.tx_channel) +
                " (USRP supports " + std::to_string(usrp_tx_channels) + " TX channels)");
        }

        const uhd::tune_request_t tune_req(_cfg.center_freq);
        const uhd::tune_result_t tx_tune = _tx_usrp->set_tx_freq(tune_req, _cfg.tx_channel);
        LOG_G_INFO() << "Actual TX RF Freq: " << format_freq_hz(tx_tune.actual_rf_freq)
                     << " Hz, DSP: " << format_freq_hz(tx_tune.actual_dsp_freq)
                     << " Hz";
        _tx_usrp->set_tx_gain(_cfg.tx_gain, _cfg.tx_channel);
        _tx_usrp->set_tx_bandwidth(_cfg.bandwidth, _cfg.tx_channel);

        uhd::stream_args_t tx_stream_args("fc32", _cfg.wire_format_tx);
        tx_stream_args.args["block_id"] = "radio";
        tx_stream_args.channels = {_cfg.tx_channel};
        _tx_stream = _tx_usrp->get_tx_stream(tx_stream_args);
        _tx_chunk_samps = _tx_stream->get_max_num_samps();
        if (_tx_chunk_samps == 0) {
            _tx_chunk_samps = _cfg.samples_per_frame();
            LOG_G_WARN() << "TX streamer reported max_num_samps=0, falling back to frame-sized chunks: "
                         << _tx_chunk_samps;
        } else {
            LOG_G_INFO() << "TX streamer chunk size: " << _tx_chunk_samps << " samples";
        }

        SensingChannel::initialize_rx_hardware_and_sync(
            _cfg,
            tune_req,
            tx_device_args,
            tx_clock_source,
            tx_time_source,
            _tx_usrp,
            _sensing_channels
        );
    }

    size_t _send_frame_chunked(
        const std::complex<float>* data,
        size_t total_samps,
        uhd::tx_metadata_t md)
    {
        if (total_samps == 0) {
            return 0;
        }

        const size_t chunk_samps = _tx_chunk_samps ? _tx_chunk_samps : total_samps;
        size_t total_sent = 0;
        while (total_sent < total_samps) {
            const size_t samps_this_call = std::min(chunk_samps, total_samps - total_sent);
            const size_t sent = _tx_stream->send(data + total_sent, samps_this_call, md, 2.0);
            total_sent += sent;

            md.start_of_burst = false;
            md.has_time_spec = false;

            if (sent == 0) {
                break;
            }
        }

        return total_sent;
    }

    void _init_fftw() {
        _ifft_plan = fftwf_plan_dft_1d(
            static_cast<int>(_cfg.fft_size),
            reinterpret_cast<fftwf_complex*>(_fft_in.data()),
            reinterpret_cast<fftwf_complex*>(_fft_out.data()),
            FFTW_BACKWARD,
            FFTW_MEASURE
        );
    }

    // Prepare ZC sequence using ZadoffChuGenerator
    void _prepare_zc_sequence() {
        _zc_seq = generate_zc_freq(_cfg.fft_size, _cfg.zc_root);
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
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        uhd::set_thread_priority_safe(0.2f, true);
        std::vector<size_t> cpu_list = {core_from_hint(_cfg, 2)};
        uhd::set_thread_affinity(cpu_list);
        prefault_thread_stack();
        const bool do_latency_profile =
            _cfg.should_profile("modulation") && _cfg.should_profile("latency");
        
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
            LOG_G_ERROR() << "UDP socket create failed";
            return;
        }
        int enable=1; setsockopt(_udp_sock,SOL_SOCKET,SO_REUSEADDR,&enable,sizeof(enable));
        memset(&_udp_addr,0,sizeof(_udp_addr));
        _udp_addr.sin_family = AF_INET;
        _udp_addr.sin_port = htons(static_cast<uint16_t>(_cfg.udp_input_port));
        if (_cfg.udp_input_ip == "0.0.0.0") {
            _udp_addr.sin_addr.s_addr = INADDR_ANY;
        } else {
            if (inet_pton(AF_INET, _cfg.udp_input_ip.c_str(), &_udp_addr.sin_addr) != 1) {
                LOG_G_ERROR() << "Invalid modulator UDP bind IP: " << _cfg.udp_input_ip;
                close(_udp_sock); _udp_sock = -1; return;
            }
        }
        if (bind(_udp_sock,(sockaddr*)&_udp_addr,sizeof(_udp_addr))<0) {
            LOG_G_ERROR() << "UDP bind failed";
            close(_udp_sock); _udp_sock=-1; return;
        }
        // Non-blocking
        fcntl(_udp_sock,F_SETFL, O_NONBLOCK);

        std::vector<uint8_t> udp_buf(25200); // Max packet size: 1008*2*100/8 bytes per frame
        while (_running.load()) {
            ssize_t recv_len = recv(_udp_sock, udp_buf.data(), udp_buf.size(), 0);
            if (recv_len <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            const int64_t pkt_ingest_ns = do_latency_profile
                ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()).count()
                : 0;

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
            // LOG_G_INFO() << "Packet constructed with size: " << packet.size() << std::endl;
            // Enqueue
            prof_step_start = ProfileClock::now();
            bool enqueued = false;
            SPSCBackoff enqueue_backoff;
            while (_running.load(std::memory_order_relaxed)) {
                if (_data_packet_buffer.try_push(std::move(packet))) {
                    enqueue_backoff.reset();
                    enqueued = true;
                    if (do_latency_profile) {
                        const int64_t encoded_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                        _data_packet_ingest_ts.try_push(PktTimestamps{pkt_ingest_ns, encoded_ns});
                    }
                    break;
                }
                enqueue_backoff.pause();
            }
            if (!enqueued) {
                _data_packet_pool.release(std::move(packet));
                break;
            }

            prof_step_end = ProfileClock::now();
            double enqueue_time = std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
            
            // ============== Profiling accumulation ==============
            prof_header_encode_total += header_encode_time;
            prof_payload_encode_total += payload_encode_time;
            prof_enqueue_total += enqueue_time;
            prof_packet_count++;
            
            if (prof_packet_count >= PROF_REPORT_INTERVAL && _cfg.should_profile("data_ingest")) {
                double total = prof_header_encode_total + prof_payload_encode_total + prof_enqueue_total;
                std::ostringstream oss;
                oss << "\n========== _data_ingest_proc Profiling (avg per packet, us) ==========\n"
                    << "Header Encode:        " << prof_header_encode_total / prof_packet_count << " us\n"
                    << "Payload Encode:       " << prof_payload_encode_total / prof_packet_count << " us\n"
                    << "Enqueue:              " << prof_enqueue_total / prof_packet_count << " us\n"
                    << "TOTAL:                " << total / prof_packet_count << " us\n"
                    << "======================================================================\n";
                LOG_G_INFO() << oss.str();
                
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
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        uhd::set_thread_priority_safe(0.6f, true);
        std::vector<size_t> cpu_list = {core_from_hint(_cfg, 1)};
        uhd::set_thread_affinity(cpu_list);
        prefault_thread_stack();
        const bool do_latency_profile =
            _cfg.should_profile("modulation") && _cfg.should_profile("latency");
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
            auto current_symbols = _acquire_symbol_frame();
            SymbolVector& current_symbols_ref = *current_symbols;
            
            size_t frame_data_index = 0; // Entire frame data index
            // Pool of real data symbols available for this frame
            prof_step_start = ProfileClock::now();
            AlignedIntVector data_pool;
            data_pool.reserve(max_data_syms_per_frame);
            int64_t frame_ingest_ns = 0;
            int64_t frame_encoded_ns = 0;
            // Pull packets only when the *whole* packet fits in current frame room.
            while (data_pool.size() < max_data_syms_per_frame) {
                AlignedIntVector* pkt_slot = _data_packet_buffer.consumer_slot();
                if (pkt_slot == nullptr) {
                    break;
                }
                const size_t room = max_data_syms_per_frame - data_pool.size();
                const size_t pkt_size = pkt_slot->size();
                if (pkt_size > room) {
                    // Current frame has no room for the next complete packet: stop pulling.
                    // Leave remaining subcarriers blank/pregen in this frame.
                    if (pkt_size > max_data_syms_per_frame) {
                        // Keep packet in queue (strict "do not fetch if it does not fit").
                        // Warn periodically because this can stall queue progress.
                        oversize_head_warn_count++;
                        if (oversize_head_warn_count <= 20 || (oversize_head_warn_count % 100) == 0) {
                            LOG_RT_WARN() << "[LDPC] Queue head packet exceeds per-frame capacity, not fetched: qpsk_syms="
                                          << pkt_size << ", max_qpsk_per_frame=" << max_data_syms_per_frame
                                          << ", warn_count=" << oversize_head_warn_count;
                        }
                    }
                    break;
                }
                AlignedIntVector pkt = std::move(*pkt_slot);
                _data_packet_buffer.consumer_pop();
                if (do_latency_profile) {
                    PktTimestamps pts{};
                    _data_packet_ingest_ts.try_pop(pts);
                    if (frame_ingest_ns == 0) {
                        frame_ingest_ns = pts.ingest_ns;
                        frame_encoded_ns = pts.encoded_ns;
                    }
                }
                data_pool.insert(data_pool.end(), pkt.begin(), pkt.end());
                _data_packet_pool.release(std::move(pkt));
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
                std::memcpy(current_symbols_ref[i].data(), _fft_in.data(), _cfg.fft_size * sizeof(std::complex<float>));
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
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2)
                    << "Average processing time: " << avg_time
                    << " ms, Load: " << load * 100.0 << "%";
                LOG_RT_INFO() << oss.str();
                // Reset statistics
                total_processing_time = 0.0;
                frame_count = 0;
            }
            {
                QueuedTxFrame queued_frame;
                SPSCBackoff queue_backoff;
                bool queued = false;
                queued_frame.samples = std::move(current_frame);
                queued_frame.symbols = current_symbols;
                queued_frame.ingest_time_ns = frame_ingest_ns;
                queued_frame.encoded_time_ns = frame_encoded_ns;
                queued_frame.mod_done_time_ns = do_latency_profile
                    ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch()).count()
                    : 0;
                while (_running.load(std::memory_order_relaxed)) {
                    if (_circular_buffer.try_push(std::move(queued_frame))) {
                        queue_backoff.reset();
                        queued = true;
                        break;
                    }
                    queue_backoff.pause();
                }
                if (!queued) {
                    break;
                }
            }
            
            // ============== Profiling report ==============
            prof_frame_count++;
            if (prof_frame_count >= PROF_REPORT_INTERVAL && _cfg.should_profile("modulation")) {
                double total = prof_data_fetch_total + prof_symbol_gen_total + prof_ifft_total + prof_cp_write_total;
                std::ostringstream oss;
                oss << "\n========== _modulation_proc Profiling (avg per frame, us) ==========\n"
                    << "Data Fetch:           " << prof_data_fetch_total / prof_frame_count << " us\n"
                    << "Symbol Generation:    " << prof_symbol_gen_total / prof_frame_count << " us\n"
                    << "IFFT (all symbols):   " << prof_ifft_total / prof_frame_count << " us\n"
                    << "CP & Write:           " << prof_cp_write_total / prof_frame_count << " us\n"
                    << "TOTAL:                " << total / prof_frame_count << " us\n";
                if (do_latency_profile) {
                    const LatencySnapshot latency = _take_latency_snapshot_and_reset();
                    if (latency.count > 0) {
                        const double n = static_cast<double>(latency.count);
                        oss << "\n---------- Latency (avg per valid frame, ms) ----------\n"
                            << "LDPC encode + ingest queue:   " << (latency.ldpc_total_ns / n) * 1e-6 << " ms\n"
                            << "Dequeue + IFFT/CP + mod queue:" << (latency.mod_total_ns / n) * 1e-6 << " ms\n"
                            << "TX circular buffer wait:      " << (latency.tx_wait_total_ns / n) * 1e-6 << " ms\n"
                            << "TOTAL E2E (excl. TX wait):    " << (latency.e2e_total_ns / n) * 1e-6 << " ms\n"
                            << "Latency sample count:         " << latency.count << "\n";
                    } else {
                        oss << "\n---------- Latency (avg per valid frame, ms) ----------\n"
                            << "No valid latency samples in this interval.\n";
                    }
                }
                oss << "===================================================================\n";
                LOG_RT_INFO() << oss.str();
                
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
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        uhd::set_thread_priority_safe(1.0f, true);
        std::vector<size_t> cpu_list = {core_from_hint(_cfg, 0)};
        uhd::set_thread_affinity(cpu_list);
        prefault_thread_stack();
        const bool do_latency_profile =
            _cfg.should_profile("modulation") && _cfg.should_profile("latency");
        
        uhd::tx_metadata_t md;
        md.start_of_burst = true;
        md.end_of_burst = false;
        md.has_time_spec = true;
        md.time_spec = _start_time;
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // wait for prefill

        while (_running.load(std::memory_order_relaxed)) {
            QueuedTxFrame frame_to_send;
            const bool has_frame = _circular_buffer.try_pop(frame_to_send);
            const uint64_t air_frame_seq = _next_tx_frame_seq.load(std::memory_order_relaxed);
            const uint64_t frame_start_symbol = _next_frame_start_symbol.load(std::memory_order_relaxed);

            if (has_frame) {
                if (do_latency_profile && frame_to_send.ingest_time_ns != 0) {
                    const int64_t t4_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                    const int64_t t1 = frame_to_send.ingest_time_ns;
                    const int64_t t2 = frame_to_send.encoded_time_ns;
                    const int64_t t3 = frame_to_send.mod_done_time_ns;
                    _latency_accumulator.ldpc_total_ns.fetch_add(t2 - t1, std::memory_order_relaxed);
                    _latency_accumulator.mod_total_ns.fetch_add(t3 - t2, std::memory_order_relaxed);
                    _latency_accumulator.tx_wait_total_ns.fetch_add(t4_ns - t3, std::memory_order_relaxed);
                    _latency_accumulator.e2e_total_ns.fetch_add(t3 - t1, std::memory_order_relaxed);
                    _latency_accumulator.count.fetch_add(1, std::memory_order_relaxed);
                }
                const size_t sent = _send_frame_chunked(
                    frame_to_send.samples.data(),
                    frame_to_send.samples.size(),
                    md);
                if (sent < frame_to_send.samples.size()) {
                    LOG_RT_WARN() << "TX Underflow: "
                                  << (frame_to_send.samples.size() - sent) << " samples";
                } else if (frame_to_send.symbols) {
                    for (auto& ch : _sensing_channels) {
                        ch->enqueue_tx_symbols(frame_to_send.symbols, frame_start_symbol, air_frame_seq);
                    }
                }
                // Return frame to pool for reuse (instead of destroying)
                _frame_pool.release(std::move(frame_to_send.samples));
            } else {
                _send_frame_chunked(_blank_frame.data(), _blank_frame.size(), md);
                LOG_RT_WARN_HZ(5) << "TX blank frame injected";
            }
            _next_tx_frame_seq.fetch_add(1, std::memory_order_relaxed);
            _next_frame_start_symbol.fetch_add(
                static_cast<uint64_t>(_cfg.num_symbols),
                std::memory_order_relaxed);
            //md.time_spec += uhd::time_spec_t(_cfg.samples_per_frame() / _cfg.sample_rate);
            md.start_of_burst = false;
            md.has_time_spec = false;
        }
        
        md.end_of_burst = true;
        _tx_stream->send("", 0, md);
    }

};

// --- Global Signal Handling ---
std::atomic<bool> stop_signal(false);
void signal_handler(int) { stop_signal.store(true); }

int UHD_SAFE_MAIN(int argc, char *[]) {
    async_logger::AsyncLoggerGuard async_logger_guard;
    std::signal(SIGINT, &signal_handler);
    uhd::set_thread_priority_safe();
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
        LOG_G_INFO() << "Locked current and future process memory with mlockall().";
    } else {
        LOG_G_WARN() << "mlockall() failed: " << std::strerror(errno);
    }
    const std::string default_config_file = "Modulator.yaml";
    Config cfg = make_default_modulator_config();

    if (argc > 1) {
        LOG_G_ERROR() << "CLI parameters are no longer supported. Please configure OFDMModulator via "
                      << default_config_file << ".";
        return 1;
    }

    if (!std::filesystem::exists(default_config_file)) {
        LOG_G_ERROR() << "Config file '" << default_config_file
                      << "' not found. Copy a sample file from the repository config directory, "
                      << "such as 'Modulator_X310.yaml' or 'Modulator_B210.yaml', to '" << default_config_file
                      << "' and edit it before starting OFDMModulator.";
        return 1;
    }

    if (!load_modulator_config_from_yaml(cfg, default_config_file)) {
        return 1;
    }

    LOG_G_INFO() << "Loaded config from: " << default_config_file;
    normalize_modulator_sensing_channels(cfg);

    // Set main thread affinity to the last configured core.
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
    LOG_G_INFO() << "\nTransmission and sensing stopped.\n";
    return 0;
}
