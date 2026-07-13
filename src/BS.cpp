#include <uhd/utils/safe_main.hpp>  // UHD_SAFE_MAIN entry macro only
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
#include "LDPCCodec.hpp"
#include "UplinkRxEngine.hpp"
#include "OFDMCore.hpp"
#include "SensingChannel.hpp"
#include "RadioBackend.hpp"
#include <functional>
#include <unordered_map>
#include <memory>
#include <filesystem>

// Type aliases are now defined in Common.hpp

namespace {

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

std::string csv_escape(const std::string& value)
{
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

struct QueuedTxFrame {
    AlignedVector samples;
    std::shared_ptr<const SymbolVector> symbols;
    int64_t ingest_time_ns{0};    // T1: UDP packet received by _udp_recv_proc
    int64_t encoded_time_ns{0};   // T2: packet pushed into _data_packet_buffer (after LDPC encode)
    int64_t mod_done_time_ns{0};  // T3: frame pushed into _circular_buffer (after IFFT/CP)
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
 * - ldpc_encode_thread: Converts payload packets into LDPC/QPSK symbols.
 * - udp_recv_thread: Receives user UDP payloads for downlink transmission.
 */
class BSEngine {
public:
    explicit BSEngine(const Config& cfg)
      : _cfg(cfg),
        _gen(std::random_device{}()),
        _dist(0, 3),
        _fft_in(cfg.ofdm.fft_size),
        _fft_out(cfg.ofdm.fft_size),
        _blank_frame(cfg.samples_per_frame(), 0.0f),
        _duplex_layout(build_duplex_frame_layout(cfg)),
        _data_resource_layout(build_data_resource_grid_layout(cfg)),
        _control_handler(cfg.network_output.mono_sensing_ip, cfg.network_output.control_port),
        _measurement_enabled(measurement_mode_enabled(cfg)),
        _frame_pool(32, [&cfg]() {
            return AlignedVector(cfg.samples_per_frame());
        }),
        _symbols_pool(32, [&cfg]() {
            SymbolVector sv;
            sv.resize(cfg.ofdm.num_symbols);
            for (size_t i = 0; i < cfg.ofdm.num_symbols; ++i) {
                sv[i].resize(cfg.ofdm.fft_size);
            }
            return sv;
        }),
        _data_packet_pool(64, []() {
            return AlignedIntVector();
        }),
        _circular_buffer(cfg.downlink_pipeline.tx_circular_buffer_size),
        _data_packet_buffer(cfg.downlink_pipeline.data_packet_buffer_size),
        _data_packet_ingest_ts(cfg.downlink_pipeline.data_packet_buffer_size),
        _uplink_channel_sender(2, [this](const auto& data) {
            _uplink_channel_pub->send_container(data);
        }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly),
        _uplink_pdf_sender(2, [this](const auto& data) {
            _uplink_pdf_pub->send_container(data);
        }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly),
        _uplink_constellation_sender(10, [this](const auto& data) {
            _uplink_constellation_pub->send_container(data);
        }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly),
        _uplink_self_channel_sender(2, [this](const auto& data) {
            _uplink_self_channel_pub->send_container(data);
        }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly),
        _uplink_self_pdf_sender(2, [this](const auto& data) {
            _uplink_self_pdf_pub->send_container(data);
        }, std::chrono::milliseconds(50), DataSender<std::complex<float>, AlignedAlloc>::DeliveryMode::LatestOnly)
    {
        _measurement_tx_gain_x10.store(
            static_cast<int32_t>(std::llround(cfg.downlink.tx_gain * 10.0)),
            std::memory_order_relaxed);
        _bit_interleaver = std::make_unique<BitBlockInterleaver>(_ldpc.get_N(), 21);
        if (_measurement_enabled && !_cfg.measurement.measurement_output_dir.empty()) {
            _measurement_summary_path =
                (_cfg.measurement.measurement_output_dir.empty()
                    ? std::string()
                    : (_cfg.measurement.measurement_output_dir + "/bs_measurement_summary.csv"));
        }
        const CompactSensingMaskAnalysis compact_mask_analysis = analyze_compact_sensing_mask(_cfg);
        _shared_sensing_cfg.sensing_symbol_stride =
            compact_mask_analysis.local_delay_doppler_supported
                ? compact_mask_analysis.implicit_symbol_stride
                : cfg.sensing.symbol_stride;
        _shared_sensing_cfg.enable_mti = true;
        _shared_sensing_cfg.skip_sensing_fft = true;
        _shared_sensing_cfg.generation = 1;
        _shared_sensing_cfg.apply_symbol_index = 0;

        _build_sensing_channels();
        _init_fftw();
        _prepare_zc_sequence();
        _init_radio();
        _precalc_positions();

        const size_t total_data_samples = _data_resource_layout.non_pilot_re_count;
        _pregen_symbols.resize(total_data_samples);
        for (auto& x : _pregen_symbols) {
            x = _qpsk_symbol_from_int(_dist(_gen));
        }
        _build_symbol_templates();
        LOG_G_INFO() << "Payload resource grid: " << _data_resource_layout.payload_re_count
                     << " payload RE out of " << _data_resource_layout.non_pilot_re_count
                     << " non-sync/non-pilot RE per frame, "
                     << _data_resource_layout.sensing_pilot_re_count << " sensing-pilot RE"
                     << (_cfg.resource_preview.data_resource_blocks_configured ? " (configured blocks)." : " (legacy full-grid mode).");
        if (_data_resource_layout.sensing_pilot_re_count > 0) {
            LOG_G_INFO() << "Sensing-pilot sequence uses alternate ZC root "
                         << _sensing_pilot_zc_root
                         << " (sync root=" << _cfg.ofdm.zc_root << ").";
        }
        if (!_measurement_enabled && _data_resource_layout.payload_re_count == 0) {
            LOG_G_WARN() << "Configured payload resource grid selects 0 RE. Incoming UDP payloads will be dropped.";
        }

        _register_commands();
    }

    ~BSEngine() {
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

        // Downlink ARQ: BS owns the DL TX window. The uplink path only carries
        // UE ACK feedback until a separate uplink ARQ namespace is added.
        _arq_enabled = _cfg.network_output.arq_enabled;
        if (_arq_enabled) {
            _dl_arq_tx.configure(_cfg.network_output);
            _dl_arq_tx.set_direction(0); // downlink
            LOG_G_INFO() << "[BS ARQ] enabled: window=" << _cfg.network_output.arq_window_packets
                         << " rto=" << _cfg.network_output.arq_retransmit_timeout_ms << "ms"
                         << " max_retries=" << _cfg.network_output.arq_max_retries;
        }
        if (_cfg.uplink_arq.arq_enabled) {
            _ul_arq_rx.configure(_cfg.uplink_arq);
            _ul_arq_rx.set_direction(1); // uplink
            LOG_G_INFO() << "[BS UL ARQ] enabled: window=" << _cfg.uplink_arq.arq_window_packets
                         << " ordered=" << _cfg.uplink_arq.arq_ordered_delivery
                         << " feedback_interval=" << _cfg.uplink_arq.arq_feedback_interval_ms
                         << "ms";
        }
        // Wire ARQ intercept on uplink RX if available
        if ((_arq_enabled || _cfg.uplink_arq.arq_enabled) && _uplink_rx) {
            _uplink_rx->set_arq_payload_intercept(
                [this](const uint8_t* data, size_t len, uint16_t seq, uint8_t flags) -> bool {
                    return _handle_arq_uplink_payload(data, len, seq, flags);
                });
        }

        _mod_thread = std::thread(&BSEngine::_modulation_proc, this);
        _ldpc_encode_thread = std::thread(&BSEngine::_ldpc_encode_proc, this);
        if (!_measurement_enabled) {
            _udp_recv_thread = std::thread(&BSEngine::_udp_recv_proc, this);
        }

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

        if (_tx_dev->supports(radio::Capability::TimedTx)) {
            constexpr double kStartLeadTimeSec = 1.0;
            const double now_s = _tx_dev->time_now().get_real_secs();
            const double scheduled_start_s = std::ceil(now_s + kStartLeadTimeSec);
            _start_time = radio::TimeSpec(scheduled_start_s);
            LOG_G_INFO() << std::fixed << std::setprecision(6)
                      << "[Start] Scheduled unified TX/RX start_time="
                      << scheduled_start_s << " s after prefill."
                      << std::defaultfloat;
        }

        _tx_async_exit_requested.store(false, std::memory_order_relaxed);
        _tx_underflow_restart_requested.store(false, std::memory_order_relaxed);
        _tx_time_error_restart_requested.store(false, std::memory_order_relaxed);
        _tx_time_error_count.store(0, std::memory_order_relaxed);
        _tx_thread = std::thread(&BSEngine::_tx_proc, this);
        if (_tx_stream) {
            _tx_async_thread = std::thread(&BSEngine::_tx_async_event_proc, this);
        }

        if (_aggregated_sensing_sender) {
            _aggregated_sensing_sender->start();
        }
        for (auto& ch : _sensing_channels) {
            ch->start(_start_time);
        }

        _bs_dl_ul_timing_diff.store(_cfg.uplink.bs_dl_ul_timing_diff, std::memory_order_relaxed);
        log_duplex_summary(_cfg, "BS");
        if (_duplex_layout.mode == DuplexMode::TDD && _duplex_layout.uplink_enabled) {
            LOG_G_INFO() << "[BS] TDD downlink blanking symbols ["
                         << _duplex_layout.ul_start << ","
                         << (_duplex_layout.ul_start + _duplex_layout.ul_count)
                         << ") for guard/uplink.";
        } else if (_duplex_layout.mode == DuplexMode::FDD && _duplex_layout.uplink_enabled) {
            LOG_G_INFO() << "[BS] FDD downlink remains active over the full frame.";
        }
        if (_uplink_rx) {
            _uplink_channel_sender.start();
            _uplink_pdf_sender.start();
            _uplink_constellation_sender.start();
            if (uplink_self_channel_debug_enabled(_cfg)) {
                _uplink_self_channel_sender.start();
                _uplink_self_pdf_sender.start();
            }
            _uplink_rx->start(_start_time);
        }
    }

    void stop() {
        if (!_running.exchange(false)) {
            _control_handler.stop();
            return;
        }

        _control_handler.stop();

        if (_uplink_rx) {
            _uplink_rx->stop();
            _uplink_channel_sender.stop();
            _uplink_pdf_sender.stop();
            _uplink_constellation_sender.stop();
            _uplink_self_channel_sender.stop();
            _uplink_self_pdf_sender.stop();
        }

        for (auto& ch : _sensing_channels) {
            ch->stop();
        }
        const int udp_sock = _udp_sock.load(std::memory_order_acquire);
        if (udp_sock >= 0) {
            shutdown(udp_sock, SHUT_RDWR);
        }

        if (_mod_thread.joinable()) _mod_thread.join();
        if (_tx_thread.joinable()) _tx_thread.join();
        if (_udp_recv_thread.joinable()) _udp_recv_thread.join();
        if (_ldpc_encode_thread.joinable()) _ldpc_encode_thread.join();
        _tx_async_exit_requested.store(true, std::memory_order_relaxed);
        if (_tx_async_thread.joinable()) _tx_async_thread.join();

        for (auto& ch : _sensing_channels) {
            ch->join();
        }
        if (_aggregated_sensing_sender) {
            _aggregated_sensing_sender->stop();
        }
    }

private:
    Config _cfg;
    radio::IDevicePtr _tx_dev;        // TX (downlink) radio device, any backend
    radio::IDevicePtr _uplink_rx_dev; // dedicated uplink RX device; null when sharing _tx_dev
    radio::ITxStreamPtr _tx_stream;
    std::unique_ptr<UplinkRxEngine> _uplink_rx;  // non-null when duplex uplink enabled
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
    AlignedVector _cfo_training_seq;
    std::vector<AlignedVector> _midframe_pilot_seqs;
    AlignedVector _sensing_pilot_seq;
    int _sensing_pilot_zc_root = 0;
    const AlignedVector _blank_frame;
    const DuplexFrameLayout _duplex_layout;
    const DataResourceGridLayout _data_resource_layout;
    SymbolVector _symbol_templates;
    std::vector<int> _payload_subcarrier_indices_flat;

    // Pre-calculated symbol positions
    std::vector<size_t> _symbol_positions; 
    
    // Thread control
    std::atomic<bool> _running{false};
    std::thread _mod_thread;             // Modulation thread
    std::thread _tx_thread;              // TX thread
    std::thread _tx_async_thread;        // TX async event monitor
    std::atomic<bool> _tx_async_exit_requested{false};
    std::atomic<bool> _tx_underflow_restart_requested{false};
    std::atomic<bool> _tx_time_error_restart_requested{false};
    std::atomic<uint64_t> _tx_time_error_count{0};
    
    radio::TimeSpec _start_time{0.0};

    // Pre-generated data
    AlignedVector _pregen_symbols;
    // Payload encoding related
    std::thread _ldpc_encode_thread;
    
    // Control handler
    ControlCommandHandler _control_handler;
    std::atomic<int64_t> _align_target_channel{-1}; // -1 means all channels
    std::atomic<int32_t> _bs_dl_ul_timing_diff{0};  // DL/UL timing difference (samples)
    std::atomic<int64_t> _last_duti_ns{0};          // rate-limit guard for DUTI
    SharedSensingRuntime _shared_sensing_cfg;
    std::mutex _shared_sensing_cfg_mutex;
    const bool _measurement_enabled;
    std::atomic<uint32_t> _measurement_requested_epoch{0};
    std::atomic<int32_t> _measurement_tx_gain_x10{0};
    std::string _measurement_summary_path;
    
    // LDPC encoding and scrambling
    LDPCCodec _ldpc{make_ldpc_5041008_cfg()};
    std::unique_ptr<BitBlockInterleaver> _bit_interleaver;
    LDPCCodec::AlignedIntVector _interleaver_bits_scratch;
    Scrambler scrambler{201600, 0x5A};
    
    // MTI Filter

    
    // UDP socket
    std::atomic<int> _udp_sock{-1};
    struct sockaddr_in _udp_addr{};
    
    // Object pools for memory reuse (eliminates per-frame memory allocations)
    ObjectPool<AlignedVector> _frame_pool;      // Pool for TX frame buffers
    ObjectPool<SymbolVector> _symbols_pool;     // Pool for symbol vectors (frequency domain)
    ObjectPool<AlignedIntVector> _data_packet_pool; // Pool for data packet buffers
    // Ring buffers must be destroyed before the pools because queued items can
    // hold shared_ptr deleters that return objects back into these pools.
    SPSCRingBuffer<QueuedTxFrame> _circular_buffer;
    SPSCRingBuffer<AlignedIntVector> _data_packet_buffer; // Data packet buffer
    struct RawUdpPacket {
        std::vector<uint8_t> bytes;
        int64_t ingest_ns{0};
    };
    SPSCRingBuffer<RawUdpPacket> _raw_udp_buffer{256}; // UDP recv -> LDPC encode intermediate buffer
    std::thread _udp_recv_thread; // UDP recv thread (separate from LDPC encode)
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
    using AlignedAlloc = AlignedAllocator<std::complex<float>, 64>;
    std::unique_ptr<ZmqByteSender> _uplink_channel_pub;
    std::unique_ptr<ZmqByteSender> _uplink_pdf_pub;
    std::unique_ptr<ZmqByteSender> _uplink_constellation_pub;
    std::unique_ptr<ZmqByteSender> _uplink_self_channel_pub;
    std::unique_ptr<ZmqByteSender> _uplink_self_pdf_pub;
    DataSender<std::complex<float>, AlignedAlloc> _uplink_channel_sender;
    DataSender<std::complex<float>, AlignedAlloc> _uplink_pdf_sender;
    DataSender<std::complex<float>, AlignedAlloc> _uplink_constellation_sender;
    DataSender<std::complex<float>, AlignedAlloc> _uplink_self_channel_sender;
    DataSender<std::complex<float>, AlignedAlloc> _uplink_self_pdf_sender;
    LatencyAccumulator _latency_accumulator;
    std::vector<std::unique_ptr<SensingChannel>> _sensing_channels;
    std::shared_ptr<AggregatedSensingDataSender> _aggregated_sensing_sender;
    std::shared_ptr<std::atomic<uint64_t>> _shared_batch_reset_symbol =
        std::make_shared<std::atomic<uint64_t>>(0);
    std::atomic<uint64_t> _next_frame_start_symbol{0};
    std::atomic<uint64_t> _next_tx_frame_seq{0};
    std::atomic<uint32_t> _ldpc_packet_seq{0};
    SeqlockedChannelEstimate _ertm_latest_ul_channel;
    uint64_t _ertm_last_injected_frame = std::numeric_limits<uint64_t>::max();
    std::atomic<uint32_t> _ertm_payload_seq{0};

    // ARQ link-layer retransmission state
    bool _arq_enabled{false};
    ArqTxWindow _dl_arq_tx;     // DL TX window (BS->UE)
    ArqRxWindow _ul_arq_rx;     // UL RX window (UE->BS)
    std::mutex _arq_feedback_mutex;
    std::deque<std::vector<uint8_t>> _arq_pending_feedback; // feedback payloads to inject into DL TX
    std::atomic<uint16_t> _arq_feedback_seq{0}; // dedicated seq space for feedback frames
    std::atomic<uint64_t> _arq_profile_last_log_ms{0};
    std::atomic<uint64_t> _arq_dl_acks_received{0};
    std::atomic<uint64_t> _arq_dl_ack_released_total{0};
    std::atomic<uint64_t> _arq_dl_invalid_feedback{0};
    std::atomic<uint64_t> _arq_dl_ack_direction_mismatch{0};
    std::atomic<uint64_t> _arq_ul_intercept_payloads{0};
    std::atomic<uint64_t> _arq_ul_intercept_feedback_flags{0};
    std::atomic<uint64_t> _arq_ul_intercept_data_flags{0};
    std::atomic<uint64_t> _arq_ul_data_accepted{0};
    std::atomic<uint64_t> _arq_ul_data_duplicates{0};
    std::atomic<uint64_t> _arq_dl_packets_tracked{0};
    std::atomic<uint64_t> _arq_dl_track_failures{0};
    std::atomic<uint64_t> _arq_dl_retransmit_enqueued{0};
    std::atomic<uint64_t> _arq_dl_feedback_enqueued{0};
    std::atomic<uint64_t> _arq_dl_feedback_transmitted{0};
    std::atomic<uint64_t> _arq_dl_window_drops{0};
    std::atomic<uint64_t> _arq_dl_window_stalls{0};
    std::atomic<uint64_t> _arq_raw_udp_drops{0};
    std::atomic<uint64_t> _arq_encoded_queue_full_events{0};
    std::atomic<uint64_t> _arq_profile_frames{0};
    std::atomic<uint64_t> _arq_profile_payload_re_used{0};
    std::atomic<uint64_t> _arq_profile_packets_pulled{0};
    std::atomic<uint64_t> _arq_profile_last_tracked{0};
    std::atomic<uint64_t> _arq_profile_last_ack_released{0};
    std::atomic<uint64_t> _arq_profile_last_retx_enqueued{0};
    std::atomic<uint64_t> _arq_profile_last_raw_udp_drops{0};
    std::atomic<uint64_t> _arq_profile_last_encoded_full{0};
    std::atomic<uint64_t> _arq_profile_last_frames{0};
    std::atomic<uint64_t> _arq_profile_last_payload_re_used{0};
    std::atomic<uint64_t> _arq_profile_last_packets_pulled{0};
    std::atomic<uint16_t> _arq_dl_last_ack_base{0};
    std::atomic<uint64_t> _arq_dl_last_ack_bitmap{0};
    std::atomic<uint64_t> _arq_dl_last_ack_released{0};

    struct EncodePacketProfile {
        double header_encode_us{0.0};
        double payload_encode_us{0.0};
        double enqueue_us{0.0};
    };

    int64_t _estimate_arq_queue_delay_ms(size_t queued_packets_ahead,
                                         size_t packet_qpsk_symbols) const {
        if (_data_resource_layout.payload_re_count == 0 ||
            _cfg.rf_sampling.sample_rate <= 0.0) {
            return 0;
        }
        const size_t packet_symbols = std::max<size_t>(
            packet_qpsk_symbols, LdpcPacketFraming::kControlSymbols);
        const size_t packets_per_frame = std::max<size_t>(
            1, _data_resource_layout.payload_re_count / packet_symbols);
        const size_t frames_until_tx = (queued_packets_ahead / packets_per_frame) + 1;
        const double frame_ms =
            _cfg.samples_per_frame() / _cfg.rf_sampling.sample_rate * 1000.0;
        return static_cast<int64_t>(std::ceil(frame_ms * frames_until_tx));
    }

    void _log_arq_profile_if_due(const char* reason, int64_t now_ms) {
        if (!_arq_enabled || !_cfg.should_profile("arq")) {
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

        size_t pending_feedback = 0;
        {
            std::lock_guard<std::mutex> lock(_arq_feedback_mutex);
            pending_feedback = _arq_pending_feedback.size();
        }

        std::vector<uint16_t> retransmit_due;
        _dl_arq_tx.get_retransmit(retransmit_due, now_ms, 64);
        const size_t outstanding = _dl_arq_tx.outstanding_count();
        const uint16_t window = _dl_arq_tx.window_size();
        const size_t raw_size = _raw_udp_buffer.size();
        const size_t raw_cap = _raw_udp_buffer.capacity();
        const size_t encoded_size = _data_packet_buffer.size();
        const size_t encoded_cap = _data_packet_buffer.capacity();
        const uint64_t tracked_total =
            _arq_dl_packets_tracked.load(std::memory_order_relaxed);
        const uint64_t ack_released_total =
            _arq_dl_ack_released_total.load(std::memory_order_relaxed);
        const uint64_t retx_total =
            _arq_dl_retransmit_enqueued.load(std::memory_order_relaxed);
        const uint64_t raw_udp_drops_total =
            _arq_raw_udp_drops.load(std::memory_order_relaxed);
        const uint64_t encoded_full_total =
            _arq_encoded_queue_full_events.load(std::memory_order_relaxed);
        const uint64_t frames_total =
            _arq_profile_frames.load(std::memory_order_relaxed);
        const uint64_t payload_re_used_total =
            _arq_profile_payload_re_used.load(std::memory_order_relaxed);
        const uint64_t packets_pulled_total =
            _arq_profile_packets_pulled.load(std::memory_order_relaxed);

        const uint64_t tracked_delta =
            tracked_total - _arq_profile_last_tracked.exchange(
                tracked_total, std::memory_order_relaxed);
        const uint64_t ack_released_delta =
            ack_released_total - _arq_profile_last_ack_released.exchange(
                ack_released_total, std::memory_order_relaxed);
        const uint64_t retx_delta =
            retx_total - _arq_profile_last_retx_enqueued.exchange(
                retx_total, std::memory_order_relaxed);
        const uint64_t raw_udp_drop_delta =
            raw_udp_drops_total - _arq_profile_last_raw_udp_drops.exchange(
                raw_udp_drops_total, std::memory_order_relaxed);
        const uint64_t encoded_full_delta =
            encoded_full_total - _arq_profile_last_encoded_full.exchange(
                encoded_full_total, std::memory_order_relaxed);
        const uint64_t frames_delta =
            frames_total - _arq_profile_last_frames.exchange(
                frames_total, std::memory_order_relaxed);
        const uint64_t payload_re_used_delta =
            payload_re_used_total - _arq_profile_last_payload_re_used.exchange(
                payload_re_used_total, std::memory_order_relaxed);
        const uint64_t packets_pulled_delta =
            packets_pulled_total - _arq_profile_last_packets_pulled.exchange(
                packets_pulled_total, std::memory_order_relaxed);
        const double frame_payload_fill_ratio =
            (frames_delta > 0 && _data_resource_layout.payload_re_count > 0)
                ? static_cast<double>(payload_re_used_delta) /
                      (static_cast<double>(frames_delta) *
                       static_cast<double>(_data_resource_layout.payload_re_count))
                : 0.0;
        const double packets_per_frame =
            frames_delta > 0
                ? static_cast<double>(packets_pulled_delta) /
                      static_cast<double>(frames_delta)
                : 0.0;

        std::ostringstream oss;
        oss << "[BS ARQ PROFILE] reason=" << reason
            << "\n  dl: outstanding=" << outstanding << "/" << window
            << " room=" << (_dl_arq_tx.has_room() ? "yes" : "no")
            << " retx_due=" << retransmit_due.size()
            << " tracked=" << tracked_total
            << " track_fail=" << _arq_dl_track_failures.load(std::memory_order_relaxed)
            << " retx_enqueued=" << retx_total
            << "\n  queues: raw_udp=" << raw_size << "/" << raw_cap
            << " encoded=" << encoded_size << "/" << encoded_cap
            << " pending_ul_ack_feedback=" << pending_feedback
            << " raw_udp_drops=" << raw_udp_drops_total
            << " encoded_full=" << encoded_full_total
            << "\n  frame: payload_fill=" << std::fixed << std::setprecision(3)
            << frame_payload_fill_ratio
            << " packets_per_frame=" << packets_per_frame << std::defaultfloat
            << "\n  dl_ack: rx=" << _arq_dl_acks_received.load(std::memory_order_relaxed)
            << " released_total=" << ack_released_total
            << " last_base=" << _arq_dl_last_ack_base.load(std::memory_order_relaxed)
            << " last_released=" << _arq_dl_last_ack_released.load(std::memory_order_relaxed)
            << " last_bitmap=0x" << std::hex
            << _arq_dl_last_ack_bitmap.load(std::memory_order_relaxed) << std::dec
            << " invalid=" << _arq_dl_invalid_feedback.load(std::memory_order_relaxed)
            << " direction_bad="
            << _arq_dl_ack_direction_mismatch.load(std::memory_order_relaxed)
            << "\n  ul_intercept: payloads="
            << _arq_ul_intercept_payloads.load(std::memory_order_relaxed)
            << " feedback_flags="
            << _arq_ul_intercept_feedback_flags.load(std::memory_order_relaxed)
            << " data_flags="
            << _arq_ul_intercept_data_flags.load(std::memory_order_relaxed)
            << "\n  ul_arq_rx: accepted="
            << _arq_ul_data_accepted.load(std::memory_order_relaxed)
            << " duplicates="
            << _arq_ul_data_duplicates.load(std::memory_order_relaxed)
            << " ack_base=" << _ul_arq_rx.ack_base()
            << " ack_bitmap=0x" << std::hex << _ul_arq_rx.ack_bitmap()
            << std::dec
            << " feedback_enqueued=" << _arq_dl_feedback_enqueued.load(std::memory_order_relaxed)
            << " feedback_tx="
            << _arq_dl_feedback_transmitted.load(std::memory_order_relaxed)
            << "\n  deltas: tracked=" << tracked_delta
            << " ack_released=" << ack_released_delta
            << " retx=" << retx_delta
            << " raw_udp_drop=" << raw_udp_drop_delta
            << " encoded_full=" << encoded_full_delta
            << "\n  stalls: window_drops="
            << _arq_dl_window_drops.load(std::memory_order_relaxed)
            << " window_stalls="
            << _arq_dl_window_stalls.load(std::memory_order_relaxed);
        LOG_G_INFO() << oss.str();
    }

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
        _aggregated_sensing_sender.reset();
        if (radio_is_sim(_cfg) && !_cfg.simulation.enable_sensing_rx) {
            _sensing_channels.clear();
            LOG_G_INFO() << "[Sensing] disabled in simulation (simulation.enable_sensing_rx=false).";
            return;
        }
        std::vector<uint32_t> aggregate_channel_ids;
        aggregate_channel_ids.reserve(_cfg.sensing.rx_channels.size());
        for (uint32_t i = 0; i < _cfg.sensing.rx_channels.size(); ++i) {
            const auto& ch_cfg = _cfg.sensing.rx_channels[i];
            if (!ch_cfg.enable_sensing_output || ch_cfg.enable_system_delay_estimation) {
                continue;
            }
            aggregate_channel_ids.push_back(i);
        }
        if (!aggregate_channel_ids.empty()) {
            _aggregated_sensing_sender = std::make_shared<AggregatedSensingDataSender>(
                _cfg.network_output.mono_sensing_ip,
                _cfg.network_output.mono_sensing_port,
                std::move(aggregate_channel_ids),
                true,
                8,
                _cfg.sensing.on_wire_format);
            LOG_G_INFO() << "[Sensing Aggregate] enabled for "
                         << _aggregated_sensing_sender->channel_count()
                         << " channels -> " << _cfg.network_output.mono_sensing_ip
                         << ':' << _cfg.network_output.mono_sensing_port;
        }

        _sensing_channels.clear();
        _sensing_channels.reserve(_cfg.sensing.rx_channels.size());
        for (uint32_t i = 0; i < _cfg.sensing.rx_channels.size(); ++i) {
            auto channel = std::make_unique<SensingChannel>(
                _cfg,
                _cfg.sensing.rx_channels[i],
                SensingChannel::SensingRole::Monostatic,
                _cfg.network_output.mono_sensing_ip,
                _cfg.network_output.mono_sensing_port,
                i,
                _running,
                _aggregated_sensing_sender,
                _shared_batch_reset_symbol,
                [this]() {
                    const uint64_t frame_len = (_cfg.ofdm.num_symbols == 0) ? 1u : static_cast<uint64_t>(_cfg.ofdm.num_symbols);
                    constexpr uint64_t kResetLeadFrames = 4;
                    const uint64_t reset_symbol =
                        _next_frame_start_symbol.load(std::memory_order_relaxed) +
                        kResetLeadFrames * frame_len;
                    uint64_t current = _shared_batch_reset_symbol->load(std::memory_order_relaxed);
                    while (current < reset_symbol &&
                           !_shared_batch_reset_symbol->compare_exchange_weak(
                               current, reset_symbol, std::memory_order_relaxed)) {
                    }
                    LOG_G_WARN() << "[Sensing Aggregate] scheduled batch reset at symbol "
                                 << _shared_batch_reset_symbol->load(std::memory_order_relaxed);
                },
                [this](const std::string&, int) {
                    if (_aggregated_sensing_sender) {
                        if (_control_handler.send_heartbeat_to_last_peer()) {
                            return;
                        }
                        _control_handler.send_heartbeat();
                        return;
                    }
                    _control_handler.send_heartbeat();
                },
                [](size_t) {
                    return std::nullopt;
                }
            );
            channel->apply_shared_cfg(_shared_sensing_cfg);
            _sensing_channels.push_back(std::move(channel));
        }
    }

    void _init_uplink_debug_publishers() {
        _uplink_channel_pub = std::make_unique<ZmqByteSender>(
            _cfg.network_output.uplink_channel_ip, static_cast<uint16_t>(_cfg.network_output.uplink_channel_port));
        _uplink_pdf_pub = std::make_unique<ZmqByteSender>(
            _cfg.network_output.uplink_pdf_ip, static_cast<uint16_t>(_cfg.network_output.uplink_pdf_port));
        _uplink_constellation_pub = std::make_unique<ZmqByteSender>(
            _cfg.network_output.uplink_constellation_ip, static_cast<uint16_t>(_cfg.network_output.uplink_constellation_port));
        if (uplink_self_channel_debug_enabled(_cfg)) {
            _uplink_self_channel_pub = std::make_unique<ZmqByteSender>(
                _cfg.network_output.uplink_self_channel_ip, static_cast<uint16_t>(_cfg.network_output.uplink_self_channel_port));
            _uplink_self_pdf_pub = std::make_unique<ZmqByteSender>(
                _cfg.network_output.uplink_self_pdf_ip, static_cast<uint16_t>(_cfg.network_output.uplink_self_pdf_port));
        }
        LOG_G_INFO() << "[UL-RX] debug ZMQ streams: channel="
                     << _cfg.network_output.uplink_channel_ip << ':' << _cfg.network_output.uplink_channel_port
                     << ", pdf=" << _cfg.network_output.uplink_pdf_ip << ':' << _cfg.network_output.uplink_pdf_port
                     << ", constellation=" << _cfg.network_output.uplink_constellation_ip << ':'
                     << _cfg.network_output.uplink_constellation_port;
        if (uplink_self_channel_debug_enabled(_cfg)) {
            LOG_G_INFO() << "[UL-RX] self-channel debug streams: channel="
                         << _cfg.network_output.uplink_self_channel_ip << ':' << _cfg.network_output.uplink_self_channel_port
                         << ", pdf=" << _cfg.network_output.uplink_self_pdf_ip << ':'
                         << _cfg.network_output.uplink_self_pdf_port;
        }
    }

    void _attach_uplink_debug_sinks() {
        if (!_uplink_rx) return;
        const bool self_debug_enabled = uplink_self_channel_debug_enabled(_cfg);
        if (!_uplink_channel_pub || !_uplink_pdf_pub || !_uplink_constellation_pub ||
            (self_debug_enabled &&
             (!_uplink_self_channel_pub || !_uplink_self_pdf_pub))) {
            _init_uplink_debug_publishers();
        }
        auto channel_sink = [this](AlignedVector&& data) {
            _uplink_channel_sender.add_data(std::move(data));
        };
        auto pdf_sink = [this](AlignedVector&& data) {
            _uplink_pdf_sender.add_data(std::move(data));
        };
        auto constellation_sink = [this](AlignedVector&& data) {
            _uplink_constellation_sender.add_data(std::move(data));
        };
        if (self_debug_enabled) {
            _uplink_rx->set_debug_sinks(
                channel_sink,
                pdf_sink,
                constellation_sink,
                [this](AlignedVector&& data) {
                    _uplink_self_channel_sender.add_data(std::move(data));
                },
                [this](AlignedVector&& data) {
                    _uplink_self_pdf_sender.add_data(std::move(data));
                });
            return;
        }
        _uplink_rx->set_debug_sinks(channel_sink, pdf_sink, constellation_sink);
    }

    void _store_ertm_uplink_channel_freq(const AlignedVector& channel_freq) {
        if (!_cfg.uplink.ertm_to_enable || channel_freq.size() != _cfg.ofdm.fft_size) {
            return;
        }
        _ertm_latest_ul_channel.store(channel_freq);
    }

    bool _try_inject_ertm_payload() {
        if (!_cfg.uplink.ertm_to_enable || !_uplink_rx || !_cfg.uplink.enabled) {
            return false;
        }
        const uint64_t current_frame = _next_tx_frame_seq.load(std::memory_order_relaxed);
        const uint64_t interval =
            std::max<uint64_t>(1, static_cast<uint64_t>(_cfg.uplink.ertm_report_interval_frames));
        if (_ertm_last_injected_frame != std::numeric_limits<uint64_t>::max() &&
            current_frame < _ertm_last_injected_frame + interval) {
            return false;
        }

        AlignedVector channel_freq;
        if (!_ertm_latest_ul_channel.load(channel_freq)) {
            return false;
        }

        const uint32_t seq = _ertm_payload_seq.fetch_add(1, std::memory_order_relaxed);
        const int32_t duti = _bs_dl_ul_timing_diff.load(std::memory_order_relaxed);
        std::vector<uint8_t> payload;
        std::string pack_error;
        if (!ErtmTimingPayload::pack(
                seq,
                static_cast<uint32_t>(_cfg.ofdm.fft_size),
                _cfg.rf_sampling.sample_rate,
                duti,
                channel_freq,
                payload,
                &pack_error)) {
            if (_cfg.should_profile("ertm")) {
                LOG_G_WARN() << "[eRTM] failed to pack TO payload: " << pack_error;
            }
            _ertm_last_injected_frame = current_frame;
            return false;
        }

        const size_t payload_blocks = LdpcPacketFraming::payload_blocks_for_len(payload.size());
        const size_t packet_qpsk_symbols = LdpcPacketFraming::packet_qpsk_symbols(payload_blocks);
        if (packet_qpsk_symbols > _data_resource_layout.payload_re_count) {
            if (_cfg.should_profile("ertm")) {
                LOG_G_WARN() << "[eRTM] TO payload does not fit in one downlink frame: payload_bytes="
                             << payload.size()
                             << ", qpsk_syms=" << packet_qpsk_symbols
                             << ", frame_capacity=" << _data_resource_layout.payload_re_count;
            }
            _ertm_last_injected_frame = current_frame;
            return false;
        }

        EncodePacketProfile profile;
        const bool enqueued = _encode_and_enqueue_payload(
            payload.data(), payload.size(), 0, false, profile, false, false,
            LdpcPacketFraming::kFlagErtmTiming);
        if (enqueued) {
            _ertm_last_injected_frame = current_frame;
            if (_cfg.should_profile("ertm") && (seq == 0 || (seq % 64u) == 0u)) {
                LOG_G_INFO() << "[eRTM] injected TO payload seq=" << seq
                             << ", duti_samples=" << duti
                             << ", fft_size=" << _cfg.ofdm.fft_size
                             << ", payload_bytes=" << payload.size();
            }
        }
        return enqueued;
    }

    void _schedule_shared_sensing_update(std::function<void(SharedSensingRuntime&)> updater) {
        auto same_runtime_values = [](const SharedSensingRuntime& lhs, const SharedSensingRuntime& rhs) {
            return lhs.sensing_symbol_stride == rhs.sensing_symbol_stride &&
                   lhs.enable_mti == rhs.enable_mti &&
                   lhs.skip_sensing_fft == rhs.skip_sensing_fft &&
                   lhs.cfar_enabled == rhs.cfar_enabled &&
                   lhs.cfar_train_doppler == rhs.cfar_train_doppler &&
                   lhs.cfar_train_range == rhs.cfar_train_range &&
                   lhs.cfar_guard_doppler == rhs.cfar_guard_doppler &&
                   lhs.cfar_guard_range == rhs.cfar_guard_range &&
                   lhs.cfar_alpha_db == rhs.cfar_alpha_db &&
                   lhs.cfar_min_range_bin == rhs.cfar_min_range_bin &&
                   lhs.cfar_dc_exclusion_bins == rhs.cfar_dc_exclusion_bins &&
                   lhs.cfar_min_power_db == rhs.cfar_min_power_db &&
                   lhs.cfar_os_rank_percent == rhs.cfar_os_rank_percent &&
                   lhs.cfar_os_suppress_doppler == rhs.cfar_os_suppress_doppler &&
                   lhs.cfar_os_suppress_range == rhs.cfar_os_suppress_range &&
                   lhs.micro_doppler_enabled == rhs.micro_doppler_enabled &&
                   lhs.micro_doppler_range_bin == rhs.micro_doppler_range_bin;
        };
        SharedSensingRuntime snapshot;
        uint64_t next_symbol = 0;
        bool duplicate = false;
        {
            std::lock_guard<std::mutex> lock(_shared_sensing_cfg_mutex);
            SharedSensingRuntime updated = _shared_sensing_cfg;
            updater(updated);
            if (same_runtime_values(updated, _shared_sensing_cfg)) {
                snapshot = _shared_sensing_cfg;
                duplicate = true;
            } else {
                _shared_sensing_cfg = updated;
                _shared_sensing_cfg.generation++;
                next_symbol = _next_frame_start_symbol.load(std::memory_order_relaxed);
                if (_cfg.ofdm.num_symbols == 0) {
                    _shared_sensing_cfg.apply_symbol_index = next_symbol;
                } else {
                    const uint64_t frame_len = static_cast<uint64_t>(_cfg.ofdm.num_symbols);
                    const uint64_t boundary = ((next_symbol + frame_len - 1) / frame_len) * frame_len;
                    _shared_sensing_cfg.apply_symbol_index = boundary;
                }
                snapshot = _shared_sensing_cfg;
            }
        }
        if (duplicate) {
            return;
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

    void _set_target_alignment_for_channel(uint32_t ch_id, int32_t value) {
        if (ch_id >= _sensing_channels.size()) {
            LOG_G_WARN() << "Invalid sensing channel id for target ALGN: " << ch_id;
            return;
        }
        _sensing_channels[ch_id]->set_target_alignment(value);
    }

    void _set_rx_gain_for_channel(uint32_t ch_id, double gain_db) {
        if (ch_id >= _sensing_channels.size()) {
            LOG_G_WARN() << "Invalid sensing channel id for RXGN: " << ch_id;
            return;
        }
        _sensing_channels[ch_id]->set_rx_gain(gain_db, nullptr);
    }

    SharedSensingRuntime _viewer_params_snapshot() {
        std::lock_guard<std::mutex> lock(_shared_sensing_cfg_mutex);
        return _shared_sensing_cfg;
    }

    SensingViewerParamsPacket _make_viewer_params_packet() {
        const SharedSensingRuntime snapshot = _viewer_params_snapshot();
        const bool aggregated_stream = static_cast<bool>(_aggregated_sensing_sender);
        const uint32_t stream_channel_count = aggregated_stream
            ? _aggregated_sensing_sender->channel_count()
            : 1u;
        const uint32_t stream_channel_mask = aggregated_stream
            ? _aggregated_sensing_sender->channel_mask()
            : 0x1u;
        // In sim mode supply the physical antenna spacing so the viewer can auto-set AoA.
        const double antenna_spacing_m = radio_is_sim(_cfg)
            ? (_cfg.simulation.array_spacing_m > 0.0
                   ? _cfg.simulation.array_spacing_m
                   : (_cfg.simulation.array_spacing_lambda * 3e8 /
                      (_cfg.downlink.center_freq > 0.0 ? _cfg.downlink.center_freq : 1.0)))
            : 0.0;
        const SensingViewerParamsPacket packet = make_sensing_viewer_params_packet(
            _cfg,
            snapshot.skip_sensing_fft,
            snapshot.enable_mti,
            snapshot.cfar_os_rank_percent,
            snapshot.cfar_os_suppress_doppler,
            snapshot.cfar_os_suppress_range,
            false,
            stream_channel_count,
            stream_channel_mask,
            aggregated_stream,
            antenna_spacing_m);
        return packet;
    }

    void _send_viewer_params() {
        const SensingViewerParamsPacket packet = _make_viewer_params_packet();
        _control_handler.send_sensing_viewer_params(
            _cfg.network_output.mono_sensing_ip,
            _cfg.network_output.mono_sensing_port,
            packet);
    }

    void _send_viewer_params(const ControlCommandHandler::ControlPeer& peer) {
        const SensingViewerParamsPacket packet = _make_viewer_params_packet();
        _control_handler.send_sensing_viewer_params(peer, packet);
    }

    void _register_commands() {
        const bool compact_mask_mode = sensing_output_mode_is_compact_mask(_cfg);
        const bool compact_mask_fft_controls_supported =
            compact_mask_runtime_fft_controls_supported(_cfg);
        const std::string compact_mask_reason =
            compact_mask_runtime_fft_controls_reason(_cfg);
        const bool backend_processing_mode =
            backend_sensing_processing_supported(_cfg);

        _control_handler.register_command("ALCH", [this](int32_t value) {
            if (value < 0) {
                _align_target_channel.store(-1);
                LOG_G_INFO() << "ALCH set to ALL channels";
                return;
            }
            _align_target_channel.store(static_cast<int64_t>(value));
            LOG_G_INFO() << "ALCH set to channel " << value;
        });

        // DL/UL boundary timing difference (samples) for the BS.
        // uplink RX window, relative to the TX frame anchor. Runtime-adjustable.
        _control_handler.register_command("DUTI", [this](int32_t value) {
            // Rate-limit so rapid repeated commands don't thrash the framing.
            const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (now - _last_duti_ns.load(std::memory_order_relaxed) < 50'000'000) {
                LOG_G_WARN() << "DUTI rate-limited (<50ms since last); ignored " << value;
                return;
            }
            _last_duti_ns.store(now, std::memory_order_relaxed);
            _bs_dl_ul_timing_diff.store(value, std::memory_order_relaxed);
            if (_uplink_rx) {
                _uplink_rx->dl_ul_timing_diff().store(value, std::memory_order_relaxed);
            }
            LOG_G_INFO() << "DUTI (DL/UL timing difference) set to "
                         << value << " samples";
        });
        _control_handler.register_request("DUTI", [this](
            int32_t, const ControlCommandHandler::ControlPeer& peer) {
            _control_handler.send_control_status(
                peer, "DUTI", _bs_dl_ul_timing_diff.load(std::memory_order_relaxed));
        });

        _control_handler.register_command("ALGN", [this](int32_t value) {
            const int64_t max_adjust = static_cast<int64_t>(_cfg.samples_per_frame());
            const int64_t target = _align_target_channel.load();
            if (target < 0) {
                for (uint32_t i = 0; i < _sensing_channels.size(); ++i) {
                    const int64_t next_target = std::clamp<int64_t>(
                        static_cast<int64_t>(_sensing_channels[i]->target_alignment()) +
                            static_cast<int64_t>(value),
                        -max_adjust,
                        max_adjust);
                    _set_target_alignment_for_channel(i, static_cast<int32_t>(next_target));
                }
                return;
            }
            if (static_cast<uint32_t>(target) >= _sensing_channels.size()) {
                LOG_G_WARN() << "Invalid sensing channel id for target ALGN delta: " << target;
                return;
            }
            const int64_t next_target = std::clamp<int64_t>(
                static_cast<int64_t>(_sensing_channels[static_cast<uint32_t>(target)]->target_alignment()) +
                    static_cast<int64_t>(value),
                -max_adjust,
                max_adjust);
            _set_target_alignment_for_channel(
                static_cast<uint32_t>(target),
                static_cast<int32_t>(next_target));
        });

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
            LOG_G_INFO() << "Received SKIP command: " << (new_skip ? 1 : 0);
        });

        _control_handler.register_command("STRD", [this, compact_mask_mode](int32_t value) {
            if (compact_mask_mode) {
                LOG_G_INFO() << "Ignoring STRD command in compact_mask sensing mode: stride is defined by sensing_mask_blocks";
                return;
            }
            size_t stride = value <= 0 ? 1 : static_cast<size_t>(value);
            const std::string stride_error = dense_sensing_stride_cfo_training_error(
                _cfg,
                stride,
                "Runtime STRD command");
            if (!stride_error.empty()) {
                LOG_G_WARN() << "Ignoring STRD command: " << stride_error;
                return;
            }
            _schedule_shared_sensing_update([stride](SharedSensingRuntime& cfg) {
                cfg.sensing_symbol_stride = stride;
            });
            LOG_G_INFO() << "Received STRD command: " << stride;
        });

        _control_handler.register_command("MTI ", [this, compact_mask_mode, compact_mask_fft_controls_supported, compact_mask_reason](int32_t value) {
            if (compact_mask_mode && !compact_mask_fft_controls_supported) {
                LOG_G_INFO() << "Ignoring MTI command in compact_mask sensing mode: "
                             << (compact_mask_reason.empty() ? "mask is not local-DD compatible" : compact_mask_reason);
                return;
            }
            const bool new_mti = (value != 0);
            _schedule_shared_sensing_update([new_mti](SharedSensingRuntime& cfg) {
                cfg.enable_mti = new_mti;
            });
            LOG_G_INFO() << "Received MTI command: " << (new_mti ? "Enable" : "Disable");
        });

        _control_handler.register_command("CFEN", [this](int32_t value) {
            const bool enabled = (value != 0);
            _schedule_shared_sensing_update([enabled](SharedSensingRuntime& cfg) {
                cfg.cfar_enabled = enabled;
            });
            LOG_G_INFO() << "Received CFEN command: " << (enabled ? 1 : 0);
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

        _control_handler.register_command("TXGN", [this](int32_t value) {
            if (!_tx_dev || !_tx_dev->supports(radio::Capability::HardwareGain)) {
                LOG_G_WARN() << "TXGN ignored: hardware gain not supported on this backend";
                return;
            }
            const double requested_gain = static_cast<double>(value) / 10.0;
            const radio::GainRange gain_range = _tx_dev->get_tx_gain_range(_cfg.downlink.tx_channel);
            const double clamped_gain = std::clamp(requested_gain, gain_range.start, gain_range.stop);
            _cfg.downlink.tx_gain = clamped_gain;
            _tx_dev->set_tx_gain(clamped_gain, _cfg.downlink.tx_channel);
            _measurement_tx_gain_x10.store(
                static_cast<int32_t>(std::llround(clamped_gain * 10.0)),
                std::memory_order_relaxed);
            LOG_G_INFO() << "Received TXGN command: " << requested_gain
                         << " dB (applied " << clamped_gain << " dB)";
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
            _measurement_requested_epoch.store(
                static_cast<uint32_t>(value), std::memory_order_release);
            LOG_G_INFO() << "Received MRST command: epoch=" << value;
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

        _control_handler.register_command("CALB", [this](int32_t value) {
            if (_sensing_channels.empty()) {
                LOG_G_WARN() << "CALB ignored: no sensing RX channels configured";
                return;
            }

            const size_t target_symbols = value <= 0 ? 0u : static_cast<size_t>(value);
            const int64_t target = _align_target_channel.load();

            auto request_one = [&](uint32_t ch_id) {
                if (ch_id >= _sensing_channels.size()) {
                    LOG_G_WARN() << "Invalid target channel for CALB: " << ch_id;
                    return;
                }
                _sensing_channels[ch_id]->request_system_response_calibration(target_symbols);
            };

            if (target < 0) {
                for (uint32_t i = 0; i < _sensing_channels.size(); ++i) {
                    request_one(i);
                }
                return;
            }

            request_one(static_cast<uint32_t>(target));
        });

        _control_handler.register_request("PARM", [this](int32_t, const ControlCommandHandler::ControlPeer& peer) {
            _send_viewer_params(peer);
        });
    }

    radio::IDevicePtr _make_device(const std::string& device_args,
                                   const std::string& clock_source,
                                   const std::string& time_source) {
        radio::DeviceConfig dcfg;
        dcfg.backend = _cfg.radio.radio_backend;
        dcfg.device_args = device_args;
        dcfg.clock_source = clock_source;
        dcfg.time_source = time_source;
        dcfg.sim_session = _cfg.simulation.session;
        dcfg.sim_tick_rate = _cfg.rf_sampling.sample_rate;
        dcfg.sim_center_freq = _cfg.downlink.center_freq;
        return radio::make_device(dcfg);
    }

    // Backend-independent radio init. The real radio tunes/gains a USRP and runs
    // PPS sync; the simulator attaches TX + sensing RX streams to the hub's
    // shared-memory rings. Branches collapse onto IDevice capability queries.
    void _init_radio() {
        const bool is_sim = radio_is_sim(_cfg);
        const std::string tx_device_args = _cfg.downlink.tx_device_args.empty() ? _cfg.usrp_device.device_args : _cfg.downlink.tx_device_args;
        const std::string tx_clock_source = _cfg.downlink.tx_clock_source.empty() ? _cfg.clock_time.clock_source : _cfg.downlink.tx_clock_source;
        const std::string tx_time_source = _cfg.downlink.tx_time_source.empty() ?
            (_cfg.clock_time.time_source.empty() ? tx_clock_source : _cfg.clock_time.time_source) :
            _cfg.downlink.tx_time_source;

        _tx_dev = _make_device(tx_device_args, tx_clock_source, tx_time_source);
        _tx_dev->set_tx_rate(_cfg.rf_sampling.sample_rate);

        if (!is_sim) {
            const size_t usrp_tx_channels = _tx_dev->get_tx_num_channels();
            if (_cfg.downlink.tx_channel >= usrp_tx_channels) {
                throw std::runtime_error(
                    "Configured TX channel out of range: " +
                    std::to_string(_cfg.downlink.tx_channel) +
                    " (USRP supports " + std::to_string(usrp_tx_channels) + " TX channels)");
            }
        }

        const radio::TuneRequest tune_req(_cfg.downlink.center_freq);
        const radio::TuneResult tx_tune = _tx_dev->set_tx_freq(tune_req, _cfg.downlink.tx_channel);
        LOG_G_INFO() << "Actual TX RF Freq: " << format_freq_hz(tx_tune.actual_rf_freq)
                     << " Hz, DSP: " << format_freq_hz(tx_tune.actual_dsp_freq)
                     << " Hz";
        _tx_dev->set_tx_gain(_cfg.downlink.tx_gain, _cfg.downlink.tx_channel);
        _tx_dev->set_tx_bandwidth(_cfg.rf_sampling.bandwidth, _cfg.downlink.tx_channel);

        radio::StreamArgs tx_stream_args("fc32", _cfg.downlink.wire_format_tx);
        tx_stream_args.args["block_id"] = "radio";
        tx_stream_args.args["sim_suffix"] = "tx";
        tx_stream_args.channels = {_cfg.downlink.tx_channel};
        _tx_stream = _tx_dev->get_tx_stream(tx_stream_args);
        _tx_chunk_samps = is_sim ? _cfg.samples_per_frame() : _tx_stream->max_num_samps();
        if (_tx_chunk_samps == 0) {
            _tx_chunk_samps = _cfg.samples_per_frame();
            LOG_G_WARN() << "TX streamer reported max_num_samps=0, falling back to frame-sized chunks: "
                         << _tx_chunk_samps;
        } else if (!is_sim) {
            LOG_G_INFO() << "TX streamer chunk size: " << _tx_chunk_samps << " samples";
        }
        if (is_sim) {
            LOG_G_INFO() << "TX radio backend: SIMULATION (session='" << _cfg.simulation.session
                         << "', no USRP). Sensing channels: " << _sensing_channels.size();
        }

        SensingChannel::initialize_rx_and_sync(
            _cfg, tune_req, _tx_dev, tx_device_args, _sensing_channels);

        if (uplink_enabled(_cfg)) {
            _init_uplink_rx(tx_device_args, tx_clock_source, tx_time_source);
        }
    }

    void _init_uplink_rx(const std::string& tx_device_args,
                         const std::string& tx_clock_source,
                         const std::string& tx_time_source) {
        const bool is_sim = radio_is_sim(_cfg);
        radio::IDevicePtr ul_rx_dev;
        bool shared_tx_device = true;
        if (is_sim) {
            ul_rx_dev = _tx_dev;
        } else {
            // Resolve the uplink RX device. Empty overrides reuse the shared TX
            // device (and its clock/time sources); a non-empty uplink.rx_device_args
            // selects a dedicated USRP, mirroring the sensing RX device scheme.
            const std::string ul_rx_device_args = _cfg.uplink.rx_device_args.empty()
                ? tx_device_args : _cfg.uplink.rx_device_args;
            const std::string ul_rx_clock = _cfg.uplink.rx_clock_source.empty()
                ? tx_clock_source : _cfg.uplink.rx_clock_source;
            const std::string ul_rx_time = _cfg.uplink.rx_time_source.empty()
                ? tx_time_source : _cfg.uplink.rx_time_source;
            if (ul_rx_device_args == tx_device_args) {
                if (ul_rx_clock != tx_clock_source || ul_rx_time != tx_time_source) {
                    throw std::runtime_error(
                        "Uplink RX clock/time source conflicts with the shared TX device. "
                        "Set uplink.rx_device_args to select a separate uplink RX USRP.");
                }
                ul_rx_dev = _tx_dev;
            } else {
                _uplink_rx_dev = _make_device(ul_rx_device_args, ul_rx_clock, ul_rx_time);
                ul_rx_dev = _uplink_rx_dev;
                shared_tx_device = false;
            }
        }

        // Full-duplex uplink RX on the BS device. TDD: same carrier as TX.
        // FDD: the uplink carrier (duplex.ul_center_freq).
        const double ul_freq = (_cfg.uplink.duplex.mode == DuplexMode::FDD &&
                                _cfg.uplink.duplex.ul_center_freq > 0.0)
            ? _cfg.uplink.duplex.ul_center_freq : _cfg.downlink.center_freq;
        const size_t ul_rx_ch = _cfg.uplink.rx_channel;
        if (!is_sim) {
            const size_t usrp_rx_channels = ul_rx_dev->get_rx_num_channels();
            if (ul_rx_ch >= usrp_rx_channels) {
                throw std::runtime_error(
                    "Configured uplink_rx_channel out of range: " +
                    std::to_string(ul_rx_ch) +
                    " (USRP supports " + std::to_string(usrp_rx_channels) + " RX channels)");
            }
        }
        ul_rx_dev->set_rx_rate(_cfg.rf_sampling.sample_rate);
        ul_rx_dev->set_rx_freq(radio::TuneRequest(ul_freq), ul_rx_ch);
        ul_rx_dev->set_rx_gain(_cfg.uplink.rx_gain, ul_rx_ch);
        ul_rx_dev->set_rx_bandwidth(_cfg.rf_sampling.bandwidth, ul_rx_ch);

        radio::StreamArgs ul_rx_args("fc32", _cfg.uplink.rx_wire_format);
        ul_rx_args.args["sim_suffix"] = "rx.ul";
        ul_rx_args.channels = {ul_rx_ch};
        _uplink_rx = std::make_unique<UplinkRxEngine>(_cfg);
        _uplink_rx->set_rx_stream(ul_rx_dev->get_rx_stream(ul_rx_args));
        _uplink_rx->dl_ul_timing_diff().store(_cfg.uplink.bs_dl_ul_timing_diff,
                                              std::memory_order_relaxed);
        _uplink_rx->set_bs_frame_provider([this]() {
            const uint64_t s = _next_frame_start_symbol.load(std::memory_order_relaxed);
            return _cfg.ofdm.num_symbols ? (s / _cfg.ofdm.num_symbols) : s;
        });
        _attach_uplink_debug_sinks();
        if (_cfg.uplink.ertm_to_enable) {
            _ertm_latest_ul_channel.resize(_cfg.ofdm.fft_size);
            _uplink_rx->set_latest_channel_estimate_sink([this](AlignedVector&& data) {
                _store_ertm_uplink_channel_freq(data);
            });
        }
        if (ul_rx_dev->supports(radio::Capability::HardwareGain)) {
            const radio::GainRange gr = ul_rx_dev->get_rx_gain_range(ul_rx_ch);
            _uplink_rx->configure_agc(
                _cfg.uplink.rx_gain, gr.start, gr.stop,
                [ul_rx_dev, ul_rx_ch](double g) { ul_rx_dev->set_rx_gain(g, ul_rx_ch); });
            if (_cfg.rf_sampling.rx_agc_enable && _cfg.should_profile("agc")) {
                LOG_G_INFO() << "[UL-RX] RX AGC enabled. initial_gain_db=" << _cfg.uplink.rx_gain
                             << ", gain_range=[" << gr.start << ", " << gr.stop << "]"
                             << ", low_threshold_db=" << _cfg.rf_sampling.rx_agc_low_threshold_db
                             << ", high_threshold_db=" << _cfg.rf_sampling.rx_agc_high_threshold_db
                             << ", max_step_db=" << _cfg.rf_sampling.rx_agc_max_step_db
                             << ", update_frames=" << _cfg.rf_sampling.rx_agc_update_frames;
            }
        } else if (_cfg.rf_sampling.rx_agc_enable) {
            LOG_G_WARN() << "[UL-RX] RX AGC requested but this radio backend has no hardware gain control; "
                         << "using fixed uplink RX gain " << _cfg.uplink.rx_gain << " dB";
        }
        LOG_G_INFO() << "[UL-RX] uplink receive enabled on RX ch " << ul_rx_ch
                     << " @ " << format_freq_hz(ul_freq) << " Hz on "
                     << (is_sim ? "simulation" : (shared_tx_device ? "shared TX device" : "dedicated device")) << ", "
                     << _uplink_rx->uplink_config().ofdm.num_symbols << " UL symbols/frame, "
                     << "zc_root=" << _uplink_rx->uplink_config().ofdm.zc_root;
    }

    size_t _send_frame_chunked(
        const std::complex<float>* data,
        size_t total_samps,
        radio::TxMetadata md)
    {
        if (total_samps == 0) {
            return 0;
        }

        // In simulation the hub paces TX via shared-memory backpressure: when a
        // consumer (e.g. the UE on the comm ring) is not yet attached or is
        // catching up, send() returns short. That is a clean pause, NOT a radio
        // underflow — treat it as such and keep retrying so the air-frame sequence
        // stays locked to the samples actually placed on the air (otherwise the
        // monostatic sensing RX/TX seq pairing drifts apart).
        const bool is_sim = radio_is_sim(_cfg);

        const size_t chunk_samps = _tx_chunk_samps ? _tx_chunk_samps : total_samps;
        size_t total_sent = 0;
        while (total_sent < total_samps) {
            const size_t samps_this_call = std::min(chunk_samps, total_samps - total_sent);
            const size_t sent = _tx_stream->send(data + total_sent, samps_this_call, md, 2.0);
            total_sent += sent;

            md.start_of_burst = false;
            md.has_time_spec = false;

            if (sent == 0) {
                if (is_sim && _running.load(std::memory_order_relaxed) &&
                    _tx_dev->running()) {
                    // Hub is alive but paused (backpressure); wait for it to drain.
                    continue;
                }
                break;
            }
        }

        return total_sent;
    }

    void _tx_async_event_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);

        while (!_tx_async_exit_requested.load(std::memory_order_relaxed)) {
            if (!_tx_stream) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            radio::AsyncMetadata async_md;
            bool got_event = false;
            try {
                got_event = _tx_stream->recv_async_msg(async_md, 0.1);
            } catch (const std::exception& e) {
                if (!_tx_async_exit_requested.load(std::memory_order_relaxed)) {
                    LOG_G_WARN() << "[TX Async] recv_async_msg failed: " << e.what();
                }
                continue;
            }

            if (!got_event) {
                continue;
            }

            auto log_event = [&](auto&& log_line) {
                log_line << "[TX Async] " << tx_async_event_code_to_string(async_md.event_code)
                         << " (channel=" << async_md.channel;
                if (async_md.has_time_spec) {
                    log_line << ", event_time=" << std::fixed << std::setprecision(6)
                             << async_md.time_spec.get_real_secs() << " s" << std::defaultfloat;
                }
                log_line << ")";
            };

            switch (async_md.event_code) {
            case radio::AsyncEvent::BurstAck:
                log_event(LOG_G_INFO());
                break;
            case radio::AsyncEvent::Underflow:
            case radio::AsyncEvent::UnderflowInPacket:
                _tx_underflow_restart_requested.store(true, std::memory_order_relaxed);
                log_event(LOG_G_WARN());
                break;
            case radio::AsyncEvent::SeqError:
            case radio::AsyncEvent::SeqErrorInBurst:
            case radio::AsyncEvent::TimeError:
                if (async_md.event_code == radio::AsyncEvent::TimeError) {
                    _tx_time_error_restart_requested.store(true, std::memory_order_relaxed);
                    _tx_time_error_count.fetch_add(1, std::memory_order_relaxed);
                }
                log_event(LOG_G_ERROR());
                break;
            default:
                log_event(LOG_G_INFO());
                break;
            }
        }
    }

    void _init_fftw() {
        _ifft_plan = fftwf_plan_dft_1d(
            static_cast<int>(_cfg.ofdm.fft_size),
            reinterpret_cast<fftwf_complex*>(_fft_in.data()),
            reinterpret_cast<fftwf_complex*>(_fft_out.data()),
            FFTW_BACKWARD,
            FFTW_MEASURE
        );
    }

    // Prepare ZC sequence using ZadoffChuGenerator
    void _prepare_zc_sequence() {
        _zc_seq = generate_zc_freq(_cfg.ofdm.fft_size, _cfg.ofdm.zc_root);
        _sensing_pilot_zc_root =
            select_known_sensing_pilot_zc_root(_cfg.ofdm.fft_size, _cfg.ofdm.zc_root);
        _sensing_pilot_seq = generate_zc_freq(_cfg.ofdm.fft_size, _sensing_pilot_zc_root);
        if (cfo_training_sequence_enabled(_cfg)) {
            _cfo_training_seq = generate_cfo_training_freq(
                _cfg.ofdm.fft_size,
                _cfg.ofdm.cfo_training_period_samples);
        } else {
            _cfo_training_seq.clear();
        }
        _midframe_pilot_seqs.clear();
        _midframe_pilot_seqs.reserve(_data_resource_layout.midframe_pilot_symbols.size());
        for (const int sym : _data_resource_layout.midframe_pilot_symbols) {
            _midframe_pilot_seqs.push_back(generate_midframe_bpsk_pilot_freq(
                _cfg.ofdm.fft_size,
                _cfg.ofdm.midframe_pilot_seed,
                static_cast<size_t>(sym),
                _cfg.ofdm.pilot_positions,
                _zc_seq));
        }
    }

    static std::complex<float> _qpsk_symbol_from_int(int sym) {
        const int idx = (sym & 3) * 2;
        return std::complex<float>(
            QPSKModulator::QPSK_TABLE_FLAT[idx],
            QPSKModulator::QPSK_TABLE_FLAT[idx + 1]
        );
    }

    void _build_symbol_templates() {
        _symbol_templates.resize(_cfg.ofdm.num_symbols);
        for (size_t sym = 0; sym < _cfg.ofdm.num_symbols; ++sym) {
            _symbol_templates[sym].resize(_cfg.ofdm.fft_size);
        }

        _payload_subcarrier_indices_flat.clear();
        _payload_subcarrier_indices_flat.reserve(_data_resource_layout.payload_re_count);

        size_t pregen_offset = 0;
        for (size_t sym = 0; sym < _cfg.ofdm.num_symbols; ++sym) {
            auto& template_symbol = _symbol_templates[sym];
            // Reserved sync symbols always keep the dedicated sync ZC sequence.
            if (is_zc_sync_symbol(_cfg, sym)) {
                std::memcpy(template_symbol.data(), _zc_seq.data(),
                            _cfg.ofdm.fft_size * sizeof(std::complex<float>));
                continue;
            }
            if (is_cfo_training_symbol(_cfg, sym)) {
                std::memcpy(template_symbol.data(), _cfo_training_seq.data(),
                            _cfg.ofdm.fft_size * sizeof(std::complex<float>));
                continue;
            }
            const int midframe_pilot_rank =
                (sym < _data_resource_layout.midframe_pilot_symbol_to_rank.size())
                    ? _data_resource_layout.midframe_pilot_symbol_to_rank[sym]
                    : -1;
            if (midframe_pilot_rank >= 0) {
                const auto& pilot_seq =
                    _midframe_pilot_seqs[static_cast<size_t>(midframe_pilot_rank)];
                std::memcpy(template_symbol.data(), pilot_seq.data(),
                            _cfg.ofdm.fft_size * sizeof(std::complex<float>));
                continue;
            }

            const int data_symbol_idx_int =
                (sym < _data_resource_layout.actual_symbol_to_data_symbol.size())
                    ? _data_resource_layout.actual_symbol_to_data_symbol[sym]
                    : -1;
            if (data_symbol_idx_int < 0) {
                continue;
            }
            const size_t data_symbol_idx = static_cast<size_t>(data_symbol_idx_int);
            const size_t non_pilot_base = _data_resource_layout.non_pilot_offsets[data_symbol_idx];
            for (size_t di = 0; di < _data_resource_layout.num_non_pilot_subcarriers; ++di) {
                const size_t k = static_cast<size_t>(_data_resource_layout.non_pilot_subcarrier_indices[di]);
                const size_t flat_idx = non_pilot_base + di;
                template_symbol[k] = _data_resource_layout.sensing_pilot_mask[flat_idx] != 0
                    ? _sensing_pilot_seq[k]
                    : _pregen_symbols[pregen_offset + di];
                const int payload_rank = _data_resource_layout.payload_rank[non_pilot_base + di];
                if (payload_rank >= 0) {
                    _payload_subcarrier_indices_flat.push_back(static_cast<int>(k));
                }
            }
            // Pilots are written after sensing-pilot RE so pilots keep highest priority
            // within non-sync symbols.
            for (size_t idx = 0; idx < _cfg.ofdm.pilot_positions.size(); ++idx) {
                const size_t k = _cfg.ofdm.pilot_positions[idx];
                if (k < _cfg.ofdm.fft_size) {
                    template_symbol[k] = _zc_seq[k];
                }
            }
            pregen_offset += _data_resource_layout.num_non_pilot_subcarriers;
        }

        if (pregen_offset != _data_resource_layout.non_pilot_re_count ||
            _payload_subcarrier_indices_flat.size() != _data_resource_layout.payload_re_count) {
            throw std::runtime_error("Failed to build symbol templates for data_resource_layout.");
        }
    }

    void _precalc_positions() {
        _symbol_positions.reserve(_cfg.ofdm.num_symbols);
        for (size_t i = 0; i < _cfg.ofdm.num_symbols; ++i) {
            _symbol_positions.push_back(i * (_cfg.ofdm.fft_size + _cfg.ofdm.cp_length));
        }
    }

    // Use QPSKModulator from OFDMCore.hpp for QPSK modulation
    QPSKModulator _qpsk_modulator;

    void _append_measurement_epoch_row(uint32_t epoch_id, double tx_gain_db, uint32_t packets_sent) {
        if (_measurement_summary_path.empty()) {
            return;
        }
        const std::vector<std::string> header{
            "run_id",
            "epoch_id",
            "tx_gain_db",
            "packets_sent",
            "payload_bytes",
            "prbs_seed",
        };
        const std::vector<std::string> row{
            _cfg.measurement.measurement_run_id,
            std::to_string(epoch_id),
            std::to_string(tx_gain_db),
            std::to_string(packets_sent),
            std::to_string(_cfg.measurement.measurement_payload_bytes),
            std::to_string(_cfg.measurement.measurement_prbs_seed),
        };
        if (!append_csv_row(_measurement_summary_path, header, row)) {
            LOG_G_WARN() << "Failed to append measurement epoch row to "
                         << _measurement_summary_path;
        }
    }

    bool _encode_and_enqueue_payload(
        const uint8_t* payload_data,
        size_t payload_len,
        int64_t pkt_ingest_ns,
        bool do_latency_profile,
        EncodePacketProfile& profile,
        bool track_arq = true,
        bool is_feedback = false,
        uint8_t packet_flags = LdpcPacketFraming::kFlags
    ) {
        using ProfileClock = std::chrono::high_resolution_clock;
        auto prof_step_start = ProfileClock::now();
        auto prof_step_end = prof_step_start;

        const size_t K_bits_local = _ldpc.get_K();
        const size_t bytes_per_ldpc_block = (K_bits_local + 7) / 8;
        if (bytes_per_ldpc_block != LdpcPacketFraming::kLdpcInfoBytesPerBlock ||
            _ldpc.get_N() != LdpcPacketFraming::kLdpcCodeBitsPerBlock) {
            throw std::runtime_error("CPU LDPC codec dimensions do not match unified framing.");
        }
        if (!LdpcPacketFraming::payload_len_fits(payload_len)) {
            static std::atomic<uint64_t> dropped_oversize_count{0};
            const uint64_t dropped = dropped_oversize_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (dropped <= 20 || (dropped % 100) == 0) {
                LOG_G_WARN() << "Dropping UDP payload because mini-header supports at most "
                             << LdpcPacketFraming::max_payload_bytes()
                             << " bytes per packet: payload_bytes=" << payload_len
                             << ", dropped_packets=" << dropped;
            }
            return true;
        }

        const size_t payload_blocks = LdpcPacketFraming::payload_blocks_for_len(payload_len);
        const size_t padded_len = payload_blocks * bytes_per_ldpc_block;
        const size_t packet_qpsk_symbols =
            LdpcPacketFraming::packet_qpsk_symbols(payload_blocks);
        if (packet_qpsk_symbols > _data_resource_layout.payload_re_count) {
            static std::atomic<uint64_t> dropped_capacity_count{0};
            const uint64_t dropped = dropped_capacity_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (dropped <= 20 || (dropped % 100) == 0) {
                LOG_G_WARN() << "Dropping UDP payload because one unified LDPC packet exceeds payload RE capacity: "
                             << "payload_bytes=" << payload_len
                             << ", qpsk_syms=" << packet_qpsk_symbols
                             << ", frame_capacity=" << _data_resource_layout.payload_re_count
                             << ", dropped_packets=" << dropped;
            }
            return true;
        }

        AlignedIntVector packet = _data_packet_pool.acquire();
        packet.clear();
        packet.resize(LdpcPacketFraming::kControlSymbols);
        const uint16_t frame_seq = is_feedback
            ? static_cast<uint16_t>(_arq_feedback_seq.fetch_add(1, std::memory_order_relaxed) & 0xFFFFu)
            : static_cast<uint16_t>(_ldpc_packet_seq.fetch_add(1, std::memory_order_relaxed) & 0xFFFFu);
        const uint8_t header_flags = static_cast<uint8_t>(
            is_feedback ? LdpcPacketFraming::kFlagArqFeedback : packet_flags);
        const LdpcMiniHeader mini_header{
            LdpcPacketFraming::kVersion,
            header_flags,
            static_cast<uint16_t>(payload_len),
            LdpcPacketFraming::payload_blocks_field_for_len(payload_len),
            frame_seq,
        };
        LdpcPacketFraming::write_control_qpsk(mini_header, packet.data());
        prof_step_end = ProfileClock::now();
        profile.header_encode_us =
            std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        LDPCCodec::AlignedByteVector input_bytes(padded_len, 0x00);
        if (payload_len > 0 && payload_data != nullptr) {
            std::memcpy(input_bytes.data(), payload_data, payload_len);
        }

        prof_step_start = ProfileClock::now();
        LDPCCodec::AlignedIntVector encoded_bits_all;
        LDPCCodec::AlignedIntVector qpsk_ints_all;
        if (payload_blocks > 0) {
            _ldpc.encode_frame(input_bytes, encoded_bits_all);
            scrambler.scramble(encoded_bits_all);
            _bit_interleaver->interleave_inplace(encoded_bits_all, _interleaver_bits_scratch);
            LDPCCodec::pack_bits_qpsk(encoded_bits_all, qpsk_ints_all);
        }
        prof_step_end = ProfileClock::now();
        profile.payload_encode_us =
            std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        packet.reserve(packet_qpsk_symbols);
        packet.insert(packet.end(), qpsk_ints_all.begin(), qpsk_ints_all.end());
        if (packet.size() != packet_qpsk_symbols) {
            _data_packet_pool.release(std::move(packet));
            throw std::runtime_error("Unified LDPC packet symbol count mismatch.");
        }

        const bool should_track_arq = _arq_enabled && track_arq && !is_feedback;
        const uint16_t arq_seq = mini_header.seq;
        AlignedIntVector arq_copy;
        if (should_track_arq) {
            arq_copy = packet; // copy before moving into the encoded queue
        }

        prof_step_start = ProfileClock::now();
        bool enqueued = false;
        bool counted_full_wait = false;
        size_t queued_packets_ahead = 0;
        SPSCBackoff enqueue_backoff;
        while (_running.load(std::memory_order_relaxed)) {
            queued_packets_ahead = _data_packet_buffer.size();
            if (_data_packet_buffer.try_push(std::move(packet))) {
                enqueued = true;
                if (do_latency_profile) {
                    const int64_t encoded_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                    _data_packet_ingest_ts.try_push(PktTimestamps{pkt_ingest_ns, encoded_ns});
                }
                break;
            }
            if (_arq_enabled) {
                if (!counted_full_wait) {
                    _arq_encoded_queue_full_events.fetch_add(1, std::memory_order_relaxed);
                    counted_full_wait = true;
                }
                _log_arq_profile_if_due("encoded_queue_full", arq_now_ms());
            }
            enqueue_backoff.pause();
        }
        if (!enqueued) {
            _data_packet_pool.release(std::move(packet));
            return false;
        }
        // ARQ: only track packets after they are accepted by the encoded queue.
        // The timeout starts at the estimated air time, not at LDPC encode time;
        // otherwise a deep encoded queue triggers retransmission before first TX.
        if (should_track_arq) {
            const int64_t now = arq_now_ms();
            const int64_t tx_due_ms =
                now + _estimate_arq_queue_delay_ms(queued_packets_ahead, packet_qpsk_symbols);
            if (_dl_arq_tx.try_insert_encoded(arq_seq, payload_data, payload_len,
                                              std::move(arq_copy), tx_due_ms)) {
                _arq_dl_packets_tracked.fetch_add(1, std::memory_order_relaxed);
            } else {
                _arq_dl_track_failures.fetch_add(1, std::memory_order_relaxed);
                _log_arq_profile_if_due("track_failed", now);
            }
        }
        prof_step_end = ProfileClock::now();
        profile.enqueue_us =
            std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        return true;
    }

    // ARQ: handle a decoded uplink payload. Returns true if consumed (feedback).
    // Called from UplinkRxEngine's decode thread via the intercept callback.
    // Feedback frames are identified by the mini-header flag, not payload sniffing.
    bool _handle_arq_uplink_payload(const uint8_t* data, size_t len, uint16_t seq, uint8_t flags) {
        _arq_ul_intercept_payloads.fetch_add(1, std::memory_order_relaxed);
        if (LdpcPacketFraming::is_arq_feedback_flags(flags)) {
            _arq_ul_intercept_feedback_flags.fetch_add(1, std::memory_order_relaxed);
            // This is an ARQ ACK from the UE for our DL packets. Feedback frames
            // carry their own seq space and are never tracked in the UL RX window.
            ArqFeedback ack;
            if (ArqFeedback::try_unpack(data, len, ack)) {
                const int64_t now = arq_now_ms();
                if (ack.direction != 0) {
                    _arq_dl_ack_direction_mismatch.fetch_add(1, std::memory_order_relaxed);
                    _log_arq_profile_if_due("dl_ack_direction_bad", now);
                }
                const size_t released = _dl_arq_tx.process_ack(ack, now);
                _arq_dl_acks_received.fetch_add(1, std::memory_order_relaxed);
                _arq_dl_ack_released_total.fetch_add(released, std::memory_order_relaxed);
                _arq_dl_last_ack_base.store(ack.ack_base, std::memory_order_relaxed);
                _arq_dl_last_ack_bitmap.store(ack.ack_bitmap, std::memory_order_relaxed);
                _arq_dl_last_ack_released.store(released, std::memory_order_relaxed);
                _log_arq_profile_if_due("dl_ack", now);
            } else {
                _arq_dl_invalid_feedback.fetch_add(1, std::memory_order_relaxed);
                _log_arq_profile_if_due("invalid_feedback", arq_now_ms());
            }
            return true; // consumed; do not forward to UDP
        }
        _arq_ul_intercept_data_flags.fetch_add(1, std::memory_order_relaxed);
        if (_cfg.uplink_arq.arq_enabled) {
            const bool accepted = _ul_arq_rx.process_received(seq, data, len);
            if (accepted) {
                _arq_ul_data_accepted.fetch_add(1, std::memory_order_relaxed);
            } else {
                _arq_ul_data_duplicates.fetch_add(1, std::memory_order_relaxed);
            }
            const int64_t now = arq_now_ms();
            if (_ul_arq_rx.should_send_ack(now)) {
                ArqFeedback ack = _ul_arq_rx.generate_ack();
                std::vector<uint8_t> fb_payload;
                ack.pack(fb_payload);
                {
                    std::lock_guard<std::mutex> lock(_arq_feedback_mutex);
                    _arq_pending_feedback.push_back(std::move(fb_payload));
                }
                _arq_dl_feedback_enqueued.fetch_add(1, std::memory_order_relaxed);
                _ul_arq_rx.mark_ack_sent(now);
                _log_arq_profile_if_due("ul_ack_feedback_enqueued", now);
            }
            if (!accepted) {
                return true; // duplicate, suppress
            }
        }
        return false; // not consumed; let normal UDP output proceed
    }

    /**
     * @brief LDPC Encoding Thread.
     * 
     * Consumes already-received payload packets from the UDP receive thread.
     * 1. Constructs a protocol header.
     * 2. Performs LDPC encoding on the header and payload.
     * 3. Scrambles and interleaves the encoded bits.
     * 4. Maps interleaved bits to QPSK symbols.
     * 5. Pushes ready-to-modulate packets to the data buffer.
     */
    void _ldpc_encode_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        radio::set_thread_priority(0.2f, true);
        bind_current_thread_from_downlink_hint(_cfg, 2);
        prefault_thread_stack();
        const bool do_latency_profile =
            _cfg.should_profile("modulation") && _cfg.should_profile("latency");

        static double prof_header_encode_total = 0.0;
        static double prof_payload_encode_total = 0.0;
        static double prof_enqueue_total = 0.0;
        static int prof_packet_count = 0;
        constexpr int PROF_REPORT_INTERVAL = 100;

        auto accumulate_profile = [&](const EncodePacketProfile& profile) {
            prof_header_encode_total += profile.header_encode_us;
            prof_payload_encode_total += profile.payload_encode_us;
            prof_enqueue_total += profile.enqueue_us;
            ++prof_packet_count;
            if (prof_packet_count >= PROF_REPORT_INTERVAL &&
                _cfg.should_profile("ldpc_encode")) {
                const double total =
                    prof_header_encode_total + prof_payload_encode_total + prof_enqueue_total;
                std::ostringstream oss;
                oss << "\n========== _ldpc_encode_proc Profiling (avg per packet, us) ==========\n"
                    << "Header Encode:        " << prof_header_encode_total / prof_packet_count << " us\n"
                    << "Payload Encode:       " << prof_payload_encode_total / prof_packet_count << " us\n"
                    << "Enqueue:              " << prof_enqueue_total / prof_packet_count << " us\n"
                    << "TOTAL:                " << total / prof_packet_count << " us\n"
                    << "======================================================================\n";
                LOG_G_INFO() << oss.str();
                prof_header_encode_total = 0.0;
                prof_payload_encode_total = 0.0;
                prof_enqueue_total = 0.0;
                prof_packet_count = 0;
            }
        };

        if (_measurement_enabled) {
            std::vector<uint8_t> payload;
            payload.reserve(_cfg.measurement.measurement_payload_bytes);
            uint32_t active_epoch_id = 0;
            uint32_t seq_in_epoch = 0;
            uint32_t packets_sent_in_epoch = 0;

            while (_running.load(std::memory_order_relaxed)) {
                const uint32_t requested_epoch = _measurement_requested_epoch.exchange(
                    0, std::memory_order_acq_rel);
                if (requested_epoch > 0) {
                    if (active_epoch_id > 0 && packets_sent_in_epoch < _cfg.measurement.measurement_packets_per_point) {
                        LOG_G_WARN() << "Interrupting measurement epoch " << active_epoch_id
                                     << " after " << packets_sent_in_epoch << " packets.";
                    }
                    active_epoch_id = requested_epoch;
                    seq_in_epoch = 0;
                    packets_sent_in_epoch = 0;
                }

                if (active_epoch_id == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                MeasurementPayloadMetadata meta;
                meta.epoch_id = active_epoch_id;
                meta.seq_in_epoch = seq_in_epoch;
                meta.packets_per_point = _cfg.measurement.measurement_packets_per_point;
                meta.payload_bytes = static_cast<uint32_t>(_cfg.measurement.measurement_payload_bytes);
                meta.prbs_seed = _cfg.measurement.measurement_prbs_seed;
                meta.tx_gain_x10 = _measurement_tx_gain_x10.load(std::memory_order_relaxed);
                if (!build_measurement_payload(payload, meta)) {
                    LOG_G_ERROR() << "Failed to build measurement payload.";
                    break;
                }

                const int64_t pkt_ingest_ns = do_latency_profile
                    ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch()).count()
                    : 0;
                EncodePacketProfile profile;
                if (!_encode_and_enqueue_payload(
                        payload.data(), payload.size(), pkt_ingest_ns, do_latency_profile, profile)) {
                    break;
                }
                accumulate_profile(profile);

                if (active_epoch_id > 0) {
                    ++seq_in_epoch;
                    ++packets_sent_in_epoch;
                    if (packets_sent_in_epoch >= _cfg.measurement.measurement_packets_per_point) {
                        const double tx_gain_db =
                            static_cast<double>(_measurement_tx_gain_x10.load(std::memory_order_relaxed)) / 10.0;
                        _append_measurement_epoch_row(active_epoch_id, tx_gain_db, packets_sent_in_epoch);
                        active_epoch_id = 0;
                        seq_in_epoch = 0;
                        packets_sent_in_epoch = 0;
                    }
                }
            }

            if (active_epoch_id > 0 && packets_sent_in_epoch > 0) {
                const double tx_gain_db =
                    static_cast<double>(_measurement_tx_gain_x10.load(std::memory_order_relaxed)) / 10.0;
                _append_measurement_epoch_row(active_epoch_id, tx_gain_db, packets_sent_in_epoch);
            }
            return;
        }

        _udp_encode_proc(do_latency_profile, accumulate_profile);
    }

    void _udp_recv_proc() {
        bind_current_thread_from_downlink_hint(_cfg, 3);
        const int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) {
            LOG_G_ERROR() << "UDP socket create failed";
            return;
        }
        _udp_sock.store(sock, std::memory_order_release);

        auto close_sock = [&]() {
            int expected = sock;
            if (_udp_sock.compare_exchange_strong(expected, -1, std::memory_order_acq_rel)) {
                close(sock);
            }
        };

        int enable = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
        constexpr int rcvbuf_size = 4 * 1024 * 1024;
        setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf_size, sizeof(rcvbuf_size));
        memset(&_udp_addr, 0, sizeof(_udp_addr));
        _udp_addr.sin_family = AF_INET;
        _udp_addr.sin_port = htons(static_cast<uint16_t>(_cfg.network_output.udp_input_port));
        if (_cfg.network_output.udp_input_ip == "0.0.0.0") {
            _udp_addr.sin_addr.s_addr = INADDR_ANY;
        } else if (inet_pton(AF_INET, _cfg.network_output.udp_input_ip.c_str(), &_udp_addr.sin_addr) != 1) {
            LOG_G_ERROR() << "Invalid BS UDP bind IP: " << _cfg.network_output.udp_input_ip;
            close_sock();
            return;
        }
        if (bind(sock, (sockaddr*)&_udp_addr, sizeof(_udp_addr)) < 0) {
            LOG_G_ERROR() << "UDP bind failed";
            close_sock();
            return;
        }
        fcntl(sock, F_SETFL, O_NONBLOCK);

        std::vector<uint8_t> udp_buf(25200);
        uint64_t dropped_payload_warn_count = 0;
        while (_running.load(std::memory_order_relaxed)) {
            const ssize_t recv_len = recv(sock, udp_buf.data(), udp_buf.size(), 0);
            if (recv_len <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if (_data_resource_layout.payload_re_count == 0) {
                ++dropped_payload_warn_count;
                if (dropped_payload_warn_count <= 20 || (dropped_payload_warn_count % 100) == 0) {
                    LOG_G_WARN() << "Dropping UDP payload because data_resource_blocks select 0 payload RE per frame. "
                                 << "dropped_packets=" << dropped_payload_warn_count;
                }
                continue;
            }
            RawUdpPacket* pkt = _raw_udp_buffer.producer_slot();
            if (pkt == nullptr) {
                static std::atomic<uint64_t> drop_count{0};
                const uint64_t dc = drop_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (_arq_enabled) {
                    _arq_raw_udp_drops.fetch_add(1, std::memory_order_relaxed);
                    _log_arq_profile_if_due("raw_udp_full", arq_now_ms());
                }
                if (dc <= 20 || (dc % 100) == 0) {
                    LOG_G_WARN() << "UDP recv intermediate buffer full, dropping packet: dropped=" << dc;
                }
                continue;
            }
            pkt->bytes.resize(static_cast<size_t>(recv_len));
            std::memcpy(pkt->bytes.data(), udp_buf.data(), static_cast<size_t>(recv_len));
            pkt->ingest_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            _raw_udp_buffer.producer_commit();
        }
        close_sock();
    }

    template <typename AccumulateProfile>
    void _udp_encode_proc(bool do_latency_profile, AccumulateProfile& accumulate_profile) {
        std::vector<uint16_t> retransmit_seqs;
        while (_running.load(std::memory_order_relaxed)) {
            // ARQ: inject pending feedback packets with highest priority
            if (_arq_enabled || _cfg.uplink_arq.arq_enabled) {
                std::deque<std::vector<uint8_t>> pending_fb;
                {
                    std::lock_guard<std::mutex> lock(_arq_feedback_mutex);
                    pending_fb.swap(_arq_pending_feedback);
                }
                for (auto& fb : pending_fb) {
                    EncodePacketProfile profile;
                    _encode_and_enqueue_payload(fb.data(), fb.size(), 0, false, profile, false, true);
                    _arq_dl_feedback_transmitted.fetch_add(1, std::memory_order_relaxed);
                }
            }

            if (_arq_enabled) {
                // ARQ: handle DL retransmissions (priority over new data)
                const int64_t now = arq_now_ms();
                _dl_arq_tx.drop_abandoned(now);
                _dl_arq_tx.get_retransmit(retransmit_seqs, now);
                for (uint16_t seq : retransmit_seqs) {
                    const size_t encoded_cap = _data_packet_buffer.capacity();
                    const size_t encoded_size = _data_packet_buffer.size();
                    const size_t retx_high_water =
                        std::max<size_t>(1, (encoded_cap * 3) / 4);
                    if (encoded_cap > 0 && encoded_size >= retx_high_water) {
                        _log_arq_profile_if_due("retx_deferred_queue_high", now);
                        break;
                    }

                    ArqTxEntry entry;
                    if (!_dl_arq_tx.get_entry_copy(seq, entry)) {
                        continue;
                    }
                    if (!entry.encoded_qpsk.empty()) {
                        // Cheap retransmit: push pre-encoded QPSK directly
                        AlignedIntVector retrans(entry.encoded_qpsk);
                        const size_t queued_packets_ahead = _data_packet_buffer.size();
                        if (!_data_packet_buffer.try_push(std::move(retrans))) {
                            _arq_encoded_queue_full_events.fetch_add(1, std::memory_order_relaxed);
                            _log_arq_profile_if_due("retx_encoded_queue_full", arq_now_ms());
                            break;
                        }
                        const int64_t estimated_tx_ms =
                            now + _estimate_arq_queue_delay_ms(
                                queued_packets_ahead, entry.encoded_qpsk.size());
                        _arq_dl_retransmit_enqueued.fetch_add(1, std::memory_order_relaxed);
                        _dl_arq_tx.mark_transmitted(seq, estimated_tx_ms);
                    } else {
                        const size_t encoded_cap = _data_packet_buffer.capacity();
                        const size_t encoded_size = _data_packet_buffer.size();
                        const size_t retx_high_water =
                            std::max<size_t>(1, (encoded_cap * 3) / 4);
                        if (encoded_cap > 0 && encoded_size >= retx_high_water) {
                            _log_arq_profile_if_due("retx_deferred_queue_high", now);
                            break;
                        }
                        EncodePacketProfile profile;
                        _encode_and_enqueue_payload(
                            entry.raw_payload.data(), entry.raw_payload.size(),
                            0, false, profile, false);
                        const size_t payload_blocks =
                            LdpcPacketFraming::payload_blocks_for_len(entry.raw_payload.size());
                        const size_t packet_qpsk_symbols =
                            LdpcPacketFraming::packet_qpsk_symbols(payload_blocks);
                        const int64_t estimated_tx_ms =
                            now + _estimate_arq_queue_delay_ms(encoded_size, packet_qpsk_symbols);
                        _arq_dl_retransmit_enqueued.fetch_add(1, std::memory_order_relaxed);
                        _dl_arq_tx.mark_transmitted(seq, estimated_tx_ms);
                    }
                }
                _log_arq_profile_if_due("periodic", arq_now_ms());
            }

            const bool injected_ertm_payload = _try_inject_ertm_payload();

            RawUdpPacket* pkt = nullptr;
            // Multiplex internal payload sources and UDP without blocking on the
            // UDP queue. eRTM reports must keep flowing even when no user data is
            // arriving on the downlink UDP input.
            if (!_arq_enabled || !_dl_arq_tx.has_retransmit_pending(arq_now_ms())) {
                pkt = _raw_udp_buffer.consumer_slot();
            }
            if (!_running.load(std::memory_order_relaxed)) break;
            if (injected_ertm_payload && pkt == nullptr) {
                continue;
            }
            if (pkt == nullptr) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            // ARQ: check window room before encoding new data
            if (_arq_enabled && !_dl_arq_tx.has_room()) {
                const uint64_t stalls =
                    _arq_dl_window_stalls.fetch_add(1, std::memory_order_relaxed) + 1;
                if (stalls <= 20 || (stalls % 100) == 0) {
                    LOG_G_WARN() << "[BS ARQ DL] TX window full, pausing UDP intake: stalls="
                                 << stalls;
                }
                _log_arq_profile_if_due("window_full_stall", arq_now_ms());
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            EncodePacketProfile profile;
            if (!_encode_and_enqueue_payload(
                    pkt->bytes.data(), pkt->bytes.size(),
                    pkt->ingest_ns, do_latency_profile, profile)) {
                _raw_udp_buffer.consumer_pop();
                break;
            }
            _raw_udp_buffer.consumer_pop();
            accumulate_profile(profile);
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
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        radio::set_thread_priority(0.6f, true);
        bind_current_thread_from_downlink_hint(_cfg, 1);
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

        const float scale = 1.0f / std::sqrt(_cfg.ofdm.fft_size)/4; // Output digital signal power is 1/16
        const size_t max_data_syms_per_frame = _data_resource_layout.payload_re_count;
        const size_t measurement_packet_limit_per_frame =
            (_measurement_enabled && _cfg.measurement.measurement_max_packets_per_frame > 0)
                ? _cfg.measurement.measurement_max_packets_per_frame
                : std::numeric_limits<size_t>::max();
        AlignedVector modulated_data_pool;
        modulated_data_pool.reserve(max_data_syms_per_frame);

        while (_running.load()) {
            frame_start = Clock::now(); // Record frame start time

            // Acquire pre-allocated objects from pools (zero heap allocations)
            AlignedVector current_frame = _frame_pool.acquire();
            auto current_symbols = _acquire_symbol_frame();
            SymbolVector& current_symbols_ref = *current_symbols;

            // Pool of real data symbols available for this frame
            prof_step_start = ProfileClock::now();
            AlignedIntVector data_pool;
            data_pool.reserve(max_data_syms_per_frame);
            int64_t frame_ingest_ns = 0;  // T1: UDP recv time of first real packet
            int64_t frame_encoded_ns = 0; // T2: LDPC encode done time of first real packet
            size_t packets_pulled = 0;
            // Pull packets only when the *whole* packet fits in current frame room.
            while (data_pool.size() < max_data_syms_per_frame) {
                if (packets_pulled >= measurement_packet_limit_per_frame) {
                    break;
                }
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
                // Consume paired timestamps; use first packet's timestamps as frame reference
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
                ++packets_pulled;
            }
            prof_step_end = ProfileClock::now();
            prof_data_fetch_total += std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
            if (_arq_enabled && _cfg.should_profile("arq")) {
                _arq_profile_frames.fetch_add(1, std::memory_order_relaxed);
                _arq_profile_payload_re_used.fetch_add(data_pool.size(), std::memory_order_relaxed);
                _arq_profile_packets_pulled.fetch_add(packets_pulled, std::memory_order_relaxed);
            }

            modulated_data_pool.resize(data_pool.size());
            auto* __restrict__ mod_ptr = modulated_data_pool.data();
            const auto* __restrict__ data_pool_ptr = data_pool.data();
            #pragma omp simd
            for (size_t payload_idx = 0; payload_idx < data_pool.size(); ++payload_idx) {
                mod_ptr[payload_idx] = _qpsk_symbol_from_int(data_pool_ptr[payload_idx]);
            }

            double symbol_gen_time = 0.0;
            double ifft_time = 0.0;
            double cp_write_time = 0.0;

            // Objects from pool are already correctly sized - no need to resize

            for (size_t i = 0; i < _cfg.ofdm.num_symbols; ++i) {
                const size_t pos = _symbol_positions[i];
                auto* buf_ptr = current_frame.data() + pos;
                if (!_duplex_layout.is_downlink(i)) {
                    std::fill_n(buf_ptr, _cfg.ofdm.fft_size + _cfg.ofdm.cp_length,
                                std::complex<float>(0.0f, 0.0f));
                    std::fill(current_symbols_ref[i].begin(), current_symbols_ref[i].end(),
                              std::complex<float>(0.0f, 0.0f));
                    continue;
                }

                prof_step_start = ProfileClock::now();
                std::memcpy(_fft_in.data(), _symbol_templates[i].data(),
                            _cfg.ofdm.fft_size * sizeof(std::complex<float>));
                if (!is_reserved_sync_symbol(_cfg, i)) {
                    const int data_symbol_idx_int = _data_resource_layout.actual_symbol_to_data_symbol[i];
                    if (data_symbol_idx_int >= 0) {
                        const size_t data_symbol_idx = static_cast<size_t>(data_symbol_idx_int);
                        const size_t payload_begin = _data_resource_layout.payload_offsets[data_symbol_idx];
                        const size_t payload_end = _data_resource_layout.payload_offsets[data_symbol_idx + 1];
                        const size_t payload_count = payload_end - payload_begin;
                        const size_t payload_available = (payload_begin < data_pool.size())
                            ? std::min(payload_count, data_pool.size() - payload_begin)
                            : 0;
                        auto* __restrict__ fft_ptr = _fft_in.data();
                        const auto* __restrict__ payload_ptr = modulated_data_pool.data();
                        const int* __restrict__ payload_sc_ptr =
                            _payload_subcarrier_indices_flat.data() + payload_begin;

                        for (size_t payload_idx = 0; payload_idx < payload_available; ++payload_idx) {
                            const int k = payload_sc_ptr[payload_idx];
                            fft_ptr[k] = payload_ptr[payload_begin + payload_idx];
                        }
                    }
                }
                
                // Save frequency domain data of current symbol (for sensing) - direct memcpy
                std::memcpy(current_symbols_ref[i].data(), _fft_in.data(), _cfg.ofdm.fft_size * sizeof(std::complex<float>));
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
                for (size_t j = 0; j < _cfg.ofdm.cp_length; ++j) {
                    buf_ptr[j] = _fft_out[_cfg.ofdm.fft_size - _cfg.ofdm.cp_length + j] * scale;
                }
                
                // Write symbol body
                #pragma omp simd
                for (size_t j = 0; j < _cfg.ofdm.fft_size; ++j) {
                    buf_ptr[_cfg.ofdm.cp_length + j] = _fft_out[j] * scale;
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
                double frame_duration = _cfg.samples_per_frame() / _cfg.rf_sampling.sample_rate * 1000.0; // Convert to ms
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
        radio::set_thread_priority(1.0f, true);
        bind_current_thread_from_downlink_hint(_cfg, 0);
        prefault_thread_stack();
        const bool do_latency_profile =
            _cfg.should_profile("modulation") && _cfg.should_profile("latency");
        
        radio::TxMetadata md;
        md.start_of_burst = true;
        md.end_of_burst = false;
        const double tick_rate = _tx_dev->master_clock_rate();
        double tx_sample_rate = _cfg.rf_sampling.sample_rate;
        {
            const double actual_tx_rate = _tx_dev->get_tx_rate(_cfg.downlink.tx_channel);
            if (actual_tx_rate > 0.0) {
                tx_sample_rate = actual_tx_rate;
            }
        }
        const double exact_frame_ticks =
            static_cast<double>(_cfg.samples_per_frame()) * tick_rate / tx_sample_rate;
        const long long frame_ticks = std::llround(exact_frame_ticks);
        const double frame_tick_error = std::abs(exact_frame_ticks - static_cast<double>(frame_ticks));
        if (frame_tick_error > 1e-6) {
            LOG_G_WARN() << std::fixed << std::setprecision(6)
                         << "[TX] Frame duration quantized to " << frame_ticks
                         << " ticks at tick_rate=" << tick_rate
                         << " Hz using tx_rate=" << tx_sample_rate
                         << " Hz (error=" << frame_tick_error << " ticks)"
                         << std::defaultfloat;
        }
        long long next_frame_ticks = _start_time.to_ticks(tick_rate);
        uint64_t handled_time_error_count = _tx_time_error_count.load(std::memory_order_relaxed);
        bool next_frame_starts_burst = true;

        if (_tx_dev->supports(radio::Capability::TimedTx)) {
            constexpr double kTxSubmitLeadSec = 0.25;
            while (_running.load(std::memory_order_relaxed)) {
                const radio::TimeSpec next_frame_time =
                    radio::TimeSpec::from_ticks(next_frame_ticks, tick_rate);
                const double wait_s =
                    (next_frame_time - _tx_dev->time_now()).get_real_secs() - kTxSubmitLeadSec;
                if (wait_s <= 0.0) break;
                std::this_thread::sleep_for(std::chrono::duration<double>(std::min(wait_s, 0.05)));
            }
        }

        auto drop_one_queued_frame = [&]() -> bool {
            QueuedTxFrame dropped_frame;
            if (!_circular_buffer.try_pop(dropped_frame)) {
                return false;
            }
            _frame_pool.release(std::move(dropped_frame.samples));
            return true;
        };

        auto restart_tx_burst = [&](const char* reason) {
            if (!_tx_dev->supports(radio::Capability::TimedTx) || !_tx_stream ||
                next_frame_ticks <= 0 || frame_ticks <= 0) {
                return;
            }

            radio::TxMetadata eob_md;
            eob_md.start_of_burst = false;
            eob_md.end_of_burst = true;
            eob_md.has_time_spec = false;
            try {
                _tx_stream->send(static_cast<const std::complex<float>*>(nullptr), 0, eob_md, 0.1);
            } catch (const std::exception& e) {
                LOG_RT_WARN() << "[TX] Failed to terminate burst after " << reason
                              << ": " << e.what();
            }

            constexpr double kRestartLeadSec = 1.0;
            const long long restart_lead_ticks = std::max<long long>(
                frame_ticks,
                static_cast<long long>(std::ceil(kRestartLeadSec * tick_rate)));
            const long long now_ticks = _tx_dev->time_now().to_ticks(tick_rate);
            const long long target_ticks = now_ticks + restart_lead_ticks;
            uint64_t frames_to_skip = 0;
            if (next_frame_ticks < target_ticks) {
                const long long late_ticks = target_ticks - next_frame_ticks;
                frames_to_skip = static_cast<uint64_t>((late_ticks + frame_ticks - 1) / frame_ticks);
            }

            size_t dropped_frames = 0;
            for (uint64_t i = 0; i < frames_to_skip; ++i) {
                if (drop_one_queued_frame()) {
                    ++dropped_frames;
                }
            }

            next_frame_ticks += static_cast<long long>(frames_to_skip) * frame_ticks;
            _next_tx_frame_seq.fetch_add(frames_to_skip, std::memory_order_relaxed);
            _next_frame_start_symbol.fetch_add(
                frames_to_skip * static_cast<uint64_t>(_cfg.ofdm.num_symbols),
                std::memory_order_relaxed);
            handled_time_error_count = _tx_time_error_count.load(std::memory_order_relaxed);
            next_frame_starts_burst = true;

            // The TX frame anchor jumped; the uplink RX (anchored to the shared
            // radio clock on real hardware) must re-lock to the new framing.
            if (_uplink_rx) {
                _uplink_rx->request_reacquire(
                    radio::TimeSpec::from_ticks(next_frame_ticks, tick_rate));
            }

            LOG_RT_WARN() << std::fixed << std::setprecision(6)
                          << "[TX] " << reason << " restart scheduled at "
                          << radio::TimeSpec::from_ticks(next_frame_ticks, tick_rate).get_real_secs()
                          << " s, skipped " << frames_to_skip
                          << " frame slots and dropped " << dropped_frames
                          << " queued frames" << std::defaultfloat;
        };

        while (_running.load(std::memory_order_relaxed)) {
            if (_tx_underflow_restart_requested.exchange(false, std::memory_order_relaxed)) {
                restart_tx_burst("UNDERFLOW");
            }

            if (_tx_time_error_restart_requested.exchange(false, std::memory_order_relaxed)) {
                handled_time_error_count = _tx_time_error_count.load(std::memory_order_relaxed);
                restart_tx_burst("TIME_ERROR");
            }

            const uint64_t observed_time_error_count =
                _tx_time_error_count.load(std::memory_order_relaxed);
            if (observed_time_error_count != handled_time_error_count) {
                handled_time_error_count = observed_time_error_count;
            }

            QueuedTxFrame frame_to_send;
            const bool has_frame = _circular_buffer.try_pop(frame_to_send);
            const uint64_t air_frame_seq = _next_tx_frame_seq.load(std::memory_order_relaxed);
            const uint64_t frame_start_symbol = _next_frame_start_symbol.load(std::memory_order_relaxed);
            md.start_of_burst = next_frame_starts_burst;
            md.end_of_burst = false;
            md.has_time_spec = true;
            md.time_spec = radio::TimeSpec::from_ticks(next_frame_ticks, tick_rate);

            if (has_frame) {
                // Measure per-stage latency
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
                    // In sim mode a short send after the hub stops is a clean shutdown,
                    // not a radio underflow — skip the underflow restart to avoid
                    // spurious log noise and frame-seq skips during teardown.
                    const bool backend_shutting_down = !_tx_dev->running();
                    if (!backend_shutting_down) {
                        _tx_underflow_restart_requested.store(true, std::memory_order_relaxed);
                        LOG_RT_WARN() << "TX Underflow: "
                                      << (frame_to_send.samples.size() - sent) << " samples";
                    }
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
                static_cast<uint64_t>(_cfg.ofdm.num_symbols),
                std::memory_order_relaxed);
            next_frame_ticks += frame_ticks;
            next_frame_starts_burst = false;
        }
        
        md.end_of_burst = true;
        _tx_stream->send(static_cast<const std::complex<float>*>(nullptr), 0, md, 0.1);
    }

};

// --- Global Signal Handling ---
std::atomic<bool> stop_signal(false);
void signal_handler(int) { stop_signal.store(true); }

int UHD_SAFE_MAIN(int argc, char *[]) {
    async_logger::AsyncLoggerGuard async_logger_guard;
    std::signal(SIGINT, &signal_handler);
    radio::set_thread_priority();
#if defined(__linux__)
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
        LOG_G_INFO() << "Locked current and future process memory with mlockall().";
    } else {
        LOG_G_WARN() << "mlockall() failed: " << std::strerror(errno);
    }
#else
    LOG_G_INFO() << "Skipping mlockall(): unsupported on this platform.";
#endif
    const std::string default_config_file = "BS.yaml";
    Config cfg = make_default_bs_config();

    if (argc > 1) {
        LOG_G_ERROR() << "CLI parameters are no longer supported. Please configure BS via "
                      << default_config_file << ".";
        return 1;
    }

    if (!path_exists(default_config_file)) {
        LOG_G_ERROR() << "Config file '" << default_config_file
                      << "' not found. Copy a sample file from the repository config directory, "
                      << "such as 'BS_X310.yaml' or 'BS_B210.yaml', to '" << default_config_file
                      << "' and edit it before starting BS.";
        return 1;
    }

    if (!load_bs_config_from_yaml(cfg, default_config_file)) {
        return 1;
    }

    LOG_G_INFO() << "Loaded config from: " << default_config_file;
    normalize_bs_sensing_channels(cfg);

    // Set main thread affinity to the last configured core.
    bind_current_thread_to_main_core(cfg);

    // Load FFTW wisdom
    FFTWManager::import_wisdom();

    // Create and start engine
    BSEngine isac_engine(cfg);
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
