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
#include "LDPCCodec.hpp"
#include "OFDMCore.hpp"
#include "SensingChannel.hpp"
#include <functional>
#include <unordered_map>
#include <memory>
#include <filesystem>

// Type aliases are now defined in Common.hpp

namespace {

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
    int64_t ingest_time_ns{0};    // T1: UDP packet received by _data_ingest_proc
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
        _data_resource_layout(build_data_resource_grid_layout(cfg)),
        _control_handler(cfg.control_port),
        _measurement_enabled(measurement_mode_enabled(cfg)),
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
        _measurement_tx_gain_x10.store(
            static_cast<int32_t>(std::llround(cfg.tx_gain * 10.0)),
            std::memory_order_relaxed);
        _bit_interleaver = std::make_unique<BitBlockInterleaver>(_ldpc.get_N(), 21);
        if (_measurement_enabled && !_cfg.measurement_output_dir.empty()) {
            _measurement_summary_path =
                (_cfg.measurement_output_dir.empty()
                    ? std::string()
                    : (_cfg.measurement_output_dir + "/modulator_measurement_summary.csv"));
        }
        const CompactSensingMaskAnalysis compact_mask_analysis = analyze_compact_sensing_mask(_cfg);
        _shared_sensing_cfg.sensing_symbol_stride =
            compact_mask_analysis.local_delay_doppler_supported
                ? compact_mask_analysis.implicit_symbol_stride
                : cfg.sensing_symbol_stride;
        _shared_sensing_cfg.enable_mti = true;
        _shared_sensing_cfg.skip_sensing_fft = true;
        _shared_sensing_cfg.generation = 1;
        _shared_sensing_cfg.apply_symbol_index = 0;

        _build_sensing_channels();
        _init_fftw();
        _prepare_zc_sequence();
        _init_usrp();
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
                     << (_cfg.data_resource_blocks_configured ? " (configured blocks)." : " (legacy full-grid mode).");
        if (_data_resource_layout.sensing_pilot_re_count > 0) {
            LOG_G_INFO() << "Sensing-pilot sequence uses alternate ZC root "
                         << _sensing_pilot_zc_root
                         << " (sync root=" << _cfg.zc_root << ").";
        }
        if (!_measurement_enabled && _data_resource_layout.payload_re_count == 0) {
            LOG_G_WARN() << "Configured payload resource grid selects 0 RE. Incoming UDP payloads will be dropped.";
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
            constexpr double kStartLeadTimeSec = 1.0;
            const double now_s = _tx_usrp->get_time_now().get_real_secs();
            const double scheduled_start_s = std::ceil(now_s + kStartLeadTimeSec);
            _start_time = uhd::time_spec_t(scheduled_start_s);
            LOG_G_INFO() << std::fixed << std::setprecision(6)
                      << "[Start] Scheduled unified TX/RX start_time="
                      << scheduled_start_s << " s after prefill."
                      << std::defaultfloat;
        }

        _tx_async_exit_requested.store(false, std::memory_order_relaxed);
        _tx_underflow_restart_requested.store(false, std::memory_order_relaxed);
        _tx_time_error_restart_requested.store(false, std::memory_order_relaxed);
        _tx_time_error_count.store(0, std::memory_order_relaxed);
        _tx_thread = std::thread(&OFDMISACEngine::_tx_proc, this);
        if (_tx_stream) {
            _tx_async_thread = std::thread(&OFDMISACEngine::_tx_async_event_proc, this);
        }

        if (_aggregated_sensing_sender) {
            _aggregated_sensing_sender->start();
        }
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
    AlignedVector _sensing_pilot_seq;
    int _sensing_pilot_zc_root = 0;
    const AlignedVector _blank_frame;
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
    
    uhd::time_spec_t _start_time{0.0}; 

    // Pre-generated data
    AlignedVector _pregen_symbols;
    // Data ingest/encoding related
    std::thread _data_thread;
    
    // Control handler
    ControlCommandHandler _control_handler;
    std::atomic<int64_t> _align_target_channel{-1}; // -1 means all channels
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
    std::shared_ptr<AggregatedSensingDataSender> _aggregated_sensing_sender;
    std::shared_ptr<std::atomic<uint64_t>> _shared_batch_reset_symbol =
        std::make_shared<std::atomic<uint64_t>>(0);
    std::atomic<uint64_t> _next_frame_start_symbol{0};
    std::atomic<uint64_t> _next_tx_frame_seq{0};

    struct EncodePacketProfile {
        double header_encode_us{0.0};
        double payload_encode_us{0.0};
        double enqueue_us{0.0};
    };

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
        std::vector<uint32_t> aggregate_channel_ids;
        aggregate_channel_ids.reserve(_cfg.sensing_rx_channels.size());
        for (uint32_t i = 0; i < _cfg.sensing_rx_channels.size(); ++i) {
            const auto& ch_cfg = _cfg.sensing_rx_channels[i];
            if (!ch_cfg.enable_sensing_output || ch_cfg.enable_system_delay_estimation) {
                continue;
            }
            aggregate_channel_ids.push_back(i);
        }
        if (!aggregate_channel_ids.empty()) {
            _aggregated_sensing_sender = std::make_shared<AggregatedSensingDataSender>(
                _cfg.mono_sensing_ip,
                _cfg.mono_sensing_port,
                std::move(aggregate_channel_ids),
                true);
            LOG_G_INFO() << "[Sensing Aggregate] enabled for "
                         << _aggregated_sensing_sender->channel_count()
                         << " channels -> " << _cfg.mono_sensing_ip
                         << ':' << _cfg.mono_sensing_port;
        }

        _sensing_channels.clear();
        _sensing_channels.reserve(_cfg.sensing_rx_channels.size());
        for (uint32_t i = 0; i < _cfg.sensing_rx_channels.size(); ++i) {
            auto channel = std::make_unique<SensingChannel>(
                _cfg,
                _cfg.sensing_rx_channels[i],
                _cfg.mono_sensing_ip,
                _cfg.mono_sensing_port,
                i,
                _running,
                _aggregated_sensing_sender,
                _shared_batch_reset_symbol,
                [this]() {
                    const uint64_t frame_len = (_cfg.num_symbols == 0) ? 1u : static_cast<uint64_t>(_cfg.num_symbols);
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
                [this](const std::string& ip, int port) {
                    if (_aggregated_sensing_sender) {
                        _control_handler.send_heartbeat(_cfg.mono_sensing_ip, _cfg.mono_sensing_port);
                        return;
                    }
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
        auto same_runtime_values = [](const SharedSensingRuntime& lhs, const SharedSensingRuntime& rhs) {
            return lhs.sensing_symbol_stride == rhs.sensing_symbol_stride &&
                   lhs.enable_mti == rhs.enable_mti &&
                   lhs.skip_sensing_fft == rhs.skip_sensing_fft;
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
                if (_cfg.num_symbols == 0) {
                    _shared_sensing_cfg.apply_symbol_index = next_symbol;
                } else {
                    const uint64_t frame_len = static_cast<uint64_t>(_cfg.num_symbols);
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

    void _send_viewer_params() {
        const SharedSensingRuntime snapshot = _viewer_params_snapshot();
        const bool aggregated_stream = static_cast<bool>(_aggregated_sensing_sender);
        const uint32_t stream_channel_count = aggregated_stream
            ? _aggregated_sensing_sender->channel_count()
            : 1u;
        const uint32_t stream_channel_mask = aggregated_stream
            ? _aggregated_sensing_sender->channel_mask()
            : 0x1u;
        const SensingViewerParamsPacket packet = make_sensing_viewer_params_packet(
            _cfg,
            snapshot.skip_sensing_fft,
            snapshot.enable_mti,
            false,
            stream_channel_count,
            stream_channel_mask,
            aggregated_stream);
        if (aggregated_stream) {
            _control_handler.send_sensing_viewer_params(
                _cfg.mono_sensing_ip,
                _cfg.mono_sensing_port,
                packet);
            return;
        }
        _control_handler.send_sensing_viewer_params(
            _cfg.mono_sensing_ip,
            _cfg.mono_sensing_port,
            packet);
    }

    void _register_commands() {
        const bool compact_mask_mode = sensing_output_mode_is_compact_mask(_cfg);
        const bool compact_mask_fft_controls_supported =
            compact_mask_runtime_fft_controls_supported(_cfg);
        const std::string compact_mask_reason =
            compact_mask_runtime_fft_controls_reason(_cfg);

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

        _control_handler.register_command("SKIP", [this, compact_mask_mode, compact_mask_fft_controls_supported, compact_mask_reason](int32_t value) {
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

        _control_handler.register_request("PARM", [this](int32_t) {
            _send_viewer_params();
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

    void _tx_async_event_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);

        while (!_tx_async_exit_requested.load(std::memory_order_relaxed)) {
            if (!_tx_stream) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            uhd::async_metadata_t async_md;
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
                _tx_underflow_restart_requested.store(true, std::memory_order_relaxed);
                log_event(LOG_G_WARN());
                break;
            case uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR:
            case uhd::async_metadata_t::EVENT_CODE_SEQ_ERROR_IN_BURST:
            case uhd::async_metadata_t::EVENT_CODE_TIME_ERROR:
                if (async_md.event_code == uhd::async_metadata_t::EVENT_CODE_TIME_ERROR) {
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
        _sensing_pilot_zc_root =
            select_known_sensing_pilot_zc_root(_cfg.fft_size, _cfg.zc_root);
        _sensing_pilot_seq = generate_zc_freq(_cfg.fft_size, _sensing_pilot_zc_root);
    }

    static std::complex<float> _qpsk_symbol_from_int(int sym) {
        const int idx = (sym & 3) * 2;
        return std::complex<float>(
            QPSKModulator::QPSK_TABLE_FLAT[idx],
            QPSKModulator::QPSK_TABLE_FLAT[idx + 1]
        );
    }

    void _build_symbol_templates() {
        _symbol_templates.resize(_cfg.num_symbols);
        for (size_t sym = 0; sym < _cfg.num_symbols; ++sym) {
            _symbol_templates[sym].resize(_cfg.fft_size);
        }

        _payload_subcarrier_indices_flat.clear();
        _payload_subcarrier_indices_flat.reserve(_data_resource_layout.payload_re_count);

        size_t data_symbol_idx = 0;
        size_t pregen_offset = 0;
        for (size_t sym = 0; sym < _cfg.num_symbols; ++sym) {
            auto& template_symbol = _symbol_templates[sym];
            // The sync symbol always keeps the dedicated sync ZC sequence.
            if (sym == _cfg.sync_pos) {
                std::memcpy(template_symbol.data(), _zc_seq.data(),
                            _cfg.fft_size * sizeof(std::complex<float>));
                continue;
            }

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
            for (size_t idx = 0; idx < _cfg.pilot_positions.size(); ++idx) {
                const size_t k = _cfg.pilot_positions[idx];
                if (k < _cfg.fft_size) {
                    template_symbol[k] = _zc_seq[k];
                }
            }
            pregen_offset += _data_resource_layout.num_non_pilot_subcarriers;
            ++data_symbol_idx;
        }

        if (data_symbol_idx != _data_resource_layout.data_symbol_count ||
            pregen_offset != _data_resource_layout.non_pilot_re_count ||
            _payload_subcarrier_indices_flat.size() != _data_resource_layout.payload_re_count) {
            throw std::runtime_error("Failed to build symbol templates for data_resource_layout.");
        }
    }

    void _precalc_positions() {
        _symbol_positions.reserve(_cfg.num_symbols);
        for (size_t i = 0; i < _cfg.num_symbols; ++i) {
            _symbol_positions.push_back(i * (_cfg.fft_size + _cfg.cp_length));
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
            _cfg.measurement_run_id,
            std::to_string(epoch_id),
            std::to_string(tx_gain_db),
            std::to_string(packets_sent),
            std::to_string(_cfg.measurement_payload_bytes),
            std::to_string(_cfg.measurement_prbs_seed),
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
        EncodePacketProfile& profile
    ) {
        using ProfileClock = std::chrono::high_resolution_clock;
        auto prof_step_start = ProfileClock::now();
        auto prof_step_end = prof_step_start;

        const size_t K_bits_local = _ldpc.get_K();
        const size_t bytes_per_ldpc_block = (K_bits_local + 7) / 8;
        const size_t padded_len =
            ((payload_len + bytes_per_ldpc_block - 1) / bytes_per_ldpc_block) * bytes_per_ldpc_block;
        const size_t header_len = K_bits_local ? (K_bits_local + 7) / 8 : 0;

        LDPCCodec::AlignedByteVector header_bytes(header_len, 0x00);
        size_t half1 = (header_len + 1) / 2;
        size_t half2 = header_len - half1;
        if ((half2 % 2) != 0 && half2 > 0) {
            half1 += 1;
            half2 -= 1;
        }
        for (size_t i = 0; i < half1; ++i) {
            header_bytes[i] = static_cast<uint8_t>((i + 1) & 0xFF);
        }
        const uint16_t payload16 =
            payload_len > 0xFFFF ? 0xFFFF : static_cast<uint16_t>(payload_len & 0xFFFF);
        for (size_t i = 0; i < half2; ++i) {
            const size_t idx = half1 + i;
            header_bytes[idx] = ((i % 2) == 0)
                ? static_cast<uint8_t>((payload16 >> 8) & 0xFF)
                : static_cast<uint8_t>(payload16 & 0xFF);
        }

        LDPCCodec::AlignedIntVector header_coded_bits;
        _ldpc.encode_frame(header_bytes, header_coded_bits);
        scrambler.scramble(header_coded_bits);
        _bit_interleaver->interleave_inplace(header_coded_bits, _interleaver_bits_scratch);
        LDPCCodec::AlignedIntVector header_qpsk_ints;
        LDPCCodec::pack_bits_qpsk(header_coded_bits, header_qpsk_ints);
        prof_step_end = ProfileClock::now();
        profile.header_encode_us =
            std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        LDPCCodec::AlignedByteVector input_bytes(padded_len, 0x00);
        if (payload_len > 0 && payload_data != nullptr) {
            std::memcpy(input_bytes.data(), payload_data, payload_len);
        }

        prof_step_start = ProfileClock::now();
        LDPCCodec::AlignedIntVector encoded_bits_all;
        _ldpc.encode_frame(input_bytes, encoded_bits_all);
        scrambler.scramble(encoded_bits_all);
        _bit_interleaver->interleave_inplace(encoded_bits_all, _interleaver_bits_scratch);
        LDPCCodec::AlignedIntVector qpsk_ints_all;
        LDPCCodec::pack_bits_qpsk(encoded_bits_all, qpsk_ints_all);
        prof_step_end = ProfileClock::now();
        profile.payload_encode_us =
            std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();

        AlignedIntVector packet = _data_packet_pool.acquire();
        packet.clear();
        packet.reserve(header_qpsk_ints.size() + qpsk_ints_all.size());
        packet.insert(packet.end(), header_qpsk_ints.begin(), header_qpsk_ints.end());
        packet.insert(packet.end(), qpsk_ints_all.begin(), qpsk_ints_all.end());

        prof_step_start = ProfileClock::now();
        bool enqueued = false;
        SPSCBackoff enqueue_backoff;
        while (_running.load(std::memory_order_relaxed)) {
            if (_data_packet_buffer.try_push(std::move(packet))) {
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
            return false;
        }
        prof_step_end = ProfileClock::now();
        profile.enqueue_us =
            std::chrono::duration<double, std::micro>(prof_step_end - prof_step_start).count();
        return true;
    }

    /**
     * @brief Data Ingest and Encoding Thread.
     * 
     * Listens on a UDP port for user data.
     * On packet receipt:
     * 1. Constructs a protocol header.
     * 2. Performs LDPC encoding on the header and payload.
     * 3. Scrambles and interleaves the encoded bits.
     * 4. Maps interleaved bits to QPSK symbols.
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
            if (prof_packet_count >= PROF_REPORT_INTERVAL && _cfg.should_profile("data_ingest")) {
                const double total =
                    prof_header_encode_total + prof_payload_encode_total + prof_enqueue_total;
                std::ostringstream oss;
                oss << "\n========== _data_ingest_proc Profiling (avg per packet, us) ==========\n"
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
            payload.reserve(_cfg.measurement_payload_bytes);
            uint32_t active_epoch_id = 0;
            uint32_t seq_in_epoch = 0;
            uint32_t packets_sent_in_epoch = 0;

            while (_running.load(std::memory_order_relaxed)) {
                const uint32_t requested_epoch = _measurement_requested_epoch.exchange(
                    0, std::memory_order_acq_rel);
                if (requested_epoch > 0) {
                    if (active_epoch_id > 0 && packets_sent_in_epoch < _cfg.measurement_packets_per_point) {
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
                meta.packets_per_point = _cfg.measurement_packets_per_point;
                meta.payload_bytes = static_cast<uint32_t>(_cfg.measurement_payload_bytes);
                meta.prbs_seed = _cfg.measurement_prbs_seed;
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
                    if (packets_sent_in_epoch >= _cfg.measurement_packets_per_point) {
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

        _udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (_udp_sock < 0) {
            LOG_G_ERROR() << "UDP socket create failed";
            return;
        }
        int enable = 1;
        setsockopt(_udp_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
        memset(&_udp_addr, 0, sizeof(_udp_addr));
        _udp_addr.sin_family = AF_INET;
        _udp_addr.sin_port = htons(static_cast<uint16_t>(_cfg.udp_input_port));
        if (_cfg.udp_input_ip == "0.0.0.0") {
            _udp_addr.sin_addr.s_addr = INADDR_ANY;
        } else if (inet_pton(AF_INET, _cfg.udp_input_ip.c_str(), &_udp_addr.sin_addr) != 1) {
            LOG_G_ERROR() << "Invalid modulator UDP bind IP: " << _cfg.udp_input_ip;
            close(_udp_sock);
            _udp_sock = -1;
            return;
        }
        if (bind(_udp_sock, (sockaddr*)&_udp_addr, sizeof(_udp_addr)) < 0) {
            LOG_G_ERROR() << "UDP bind failed";
            close(_udp_sock);
            _udp_sock = -1;
            return;
        }
        fcntl(_udp_sock, F_SETFL, O_NONBLOCK);

        std::vector<uint8_t> udp_buf(25200);
        uint64_t dropped_payload_warn_count = 0;
        while (_running.load(std::memory_order_relaxed)) {
            const ssize_t recv_len = recv(_udp_sock, udp_buf.data(), udp_buf.size(), 0);
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
            const int64_t pkt_ingest_ns = do_latency_profile
                ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()).count()
                : 0;
            EncodePacketProfile profile;
            if (!_encode_and_enqueue_payload(
                    udp_buf.data(),
                    static_cast<size_t>(recv_len),
                    pkt_ingest_ns,
                    do_latency_profile,
                    profile)) {
                break;
            }
            accumulate_profile(profile);
        }
        if (_udp_sock >= 0) {
            close(_udp_sock);
            _udp_sock = -1;
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
        const size_t max_data_syms_per_frame = _data_resource_layout.payload_re_count;
        const size_t measurement_packet_limit_per_frame =
            (_measurement_enabled && _cfg.measurement_max_packets_per_frame > 0)
                ? _cfg.measurement_max_packets_per_frame
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

            for (size_t i = 0; i < _cfg.num_symbols; ++i) {
                const size_t pos = _symbol_positions[i];
                auto* buf_ptr = current_frame.data() + pos;

                prof_step_start = ProfileClock::now();
                std::memcpy(_fft_in.data(), _symbol_templates[i].data(),
                            _cfg.fft_size * sizeof(std::complex<float>));
                if (i != _cfg.sync_pos) {
                    const int data_symbol_idx_int = _data_resource_layout.actual_symbol_to_data_symbol[i];
                    if (data_symbol_idx_int < 0) {
                        throw std::runtime_error("Invalid payload resource layout for non-sync symbol.");
                    }
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
        const double tick_rate = _tx_usrp ? _tx_usrp->get_master_clock_rate() : _cfg.sample_rate;
        const double exact_frame_ticks =
            static_cast<double>(_cfg.samples_per_frame()) * tick_rate / _cfg.sample_rate;
        const long long frame_ticks = std::llround(exact_frame_ticks);
        const double frame_tick_error = std::abs(exact_frame_ticks - static_cast<double>(frame_ticks));
        if (frame_tick_error > 1e-6) {
            LOG_G_WARN() << std::fixed << std::setprecision(6)
                         << "[TX] Frame duration quantized to " << frame_ticks
                         << " ticks at tick_rate=" << tick_rate
                         << " Hz (error=" << frame_tick_error << " ticks)"
                         << std::defaultfloat;
        }
        long long next_frame_ticks = _start_time.to_ticks(tick_rate);
        uint64_t handled_time_error_count = _tx_time_error_count.load(std::memory_order_relaxed);
        bool next_frame_starts_burst = true;

        if (_tx_usrp) {
            constexpr double kTxSubmitLeadSec = 0.25;
            while (_running.load(std::memory_order_relaxed)) {
                const uhd::time_spec_t next_frame_time =
                    uhd::time_spec_t::from_ticks(next_frame_ticks, tick_rate);
                const double wait_s =
                    (next_frame_time - _tx_usrp->get_time_now()).get_real_secs() - kTxSubmitLeadSec;
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
            if (!_tx_usrp || !_tx_stream || next_frame_ticks <= 0 || frame_ticks <= 0) {
                return;
            }

            uhd::tx_metadata_t eob_md;
            eob_md.start_of_burst = false;
            eob_md.end_of_burst = true;
            eob_md.has_time_spec = false;
            try {
                _tx_stream->send("", 0, eob_md);
            } catch (const std::exception& e) {
                LOG_RT_WARN() << "[TX] Failed to terminate burst after " << reason
                              << ": " << e.what();
            }

            constexpr double kRestartLeadSec = 1.0;
            const long long restart_lead_ticks = std::max<long long>(
                frame_ticks,
                static_cast<long long>(std::ceil(kRestartLeadSec * tick_rate)));
            const long long now_ticks = _tx_usrp->get_time_now().to_ticks(tick_rate);
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
                frames_to_skip * static_cast<uint64_t>(_cfg.num_symbols),
                std::memory_order_relaxed);
            handled_time_error_count = _tx_time_error_count.load(std::memory_order_relaxed);
            next_frame_starts_burst = true;

            LOG_RT_WARN() << std::fixed << std::setprecision(6)
                          << "[TX] " << reason << " restart scheduled at "
                          << uhd::time_spec_t::from_ticks(next_frame_ticks, tick_rate).get_real_secs()
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
            md.time_spec = uhd::time_spec_t::from_ticks(next_frame_ticks, tick_rate);

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
                    _tx_underflow_restart_requested.store(true, std::memory_order_relaxed);
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
            next_frame_ticks += frame_ticks;
            next_frame_starts_burst = false;
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
#if defined(__linux__)
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
        LOG_G_INFO() << "Locked current and future process memory with mlockall().";
    } else {
        LOG_G_WARN() << "mlockall() failed: " << std::strerror(errno);
    }
#else
    LOG_G_INFO() << "Skipping mlockall(): unsupported on this platform.";
#endif
    const std::string default_config_file = "Modulator.yaml";
    Config cfg = make_default_modulator_config();

    if (argc > 1) {
        LOG_G_ERROR() << "CLI parameters are no longer supported. Please configure OFDMModulator via "
                      << default_config_file << ".";
        return 1;
    }

    if (!path_exists(default_config_file)) {
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
