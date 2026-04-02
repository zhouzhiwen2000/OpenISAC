#include "SensingChannel.hpp"

#include <uhd/utils/thread.hpp>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <limits>
#include <unordered_map>

namespace {

inline bool should_profile_sensing(const Config& cfg) {
    const bool sensing_enabled =
        cfg.should_profile("sensing") ||
        cfg.should_profile("sensing_proc") ||
        cfg.should_profile("sensing_process");
    return sensing_enabled && cfg.should_profile("latency");
}

int64_t round_divide_nearest_away_from_zero(int64_t numerator, int64_t denominator) {
    if (denominator <= 0) {
        return 0;
    }

    const uint64_t denom_u = static_cast<uint64_t>(denominator);
    const uint64_t half_u = denom_u / 2;
    const uint64_t magnitude_u = (numerator < 0)
        ? static_cast<uint64_t>(-(numerator + 1)) + 1u
        : static_cast<uint64_t>(numerator);
    const uint64_t rounded_u = (magnitude_u + half_u) / denom_u;

    if (rounded_u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return (numerator < 0)
            ? std::numeric_limits<int64_t>::min()
            : std::numeric_limits<int64_t>::max();
    }

    const int64_t rounded = static_cast<int64_t>(rounded_u);
    return (numerator < 0) ? -rounded : rounded;
}

uint64_t compute_rx_frame_seq_from_time(
    const uhd::time_spec_t& raw_start_time,
    const uhd::time_spec_t& stream_start_time,
    int32_t discard_samples,
    int32_t target_alignment_samples,
    const Config& cfg
) {
    if (cfg.sample_rate <= 0.0 || cfg.samples_per_frame() == 0) {
        return 0;
    }

    const uhd::time_spec_t aligned_start_time =
        raw_start_time + uhd::time_spec_t(static_cast<double>(discard_samples) / cfg.sample_rate);
    const double delta_samples =
        (aligned_start_time - stream_start_time).get_real_secs() * cfg.sample_rate;
    const int64_t rounded_samples = static_cast<int64_t>(std::llround(delta_samples));
    const int64_t frame_samples = static_cast<int64_t>(cfg.samples_per_frame());
    if (frame_samples <= 0) {
        return 0;
    }

    const int64_t pairing_samples =
        rounded_samples - static_cast<int64_t>(target_alignment_samples);
    // Only round once when converting UHD time to integer samples; keep frame indexing in integer math.
    const int64_t rounded_frame_seq =
        round_divide_nearest_away_from_zero(pairing_samples, frame_samples);
    return rounded_frame_seq > 0 ? static_cast<uint64_t>(rounded_frame_seq) : 0;
}

int32_t normalize_alignment_samples(int64_t samples, int64_t frame_samples) {
    if (frame_samples <= 0) {
        return 0;
    }
    int64_t normalized = samples % frame_samples;
    if (normalized > frame_samples / 2) {
        normalized -= frame_samples;
    } else if (normalized < -(frame_samples / 2)) {
        normalized += frame_samples;
    }
    normalized = std::clamp<int64_t>(
        normalized,
        static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
        static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
    return static_cast<int32_t>(normalized);
}

int32_t compute_rx_frame_boundary_error_samples(
    const uhd::time_spec_t& raw_start_time,
    const uhd::time_spec_t& stream_start_time,
    int32_t discard_samples,
    const Config& cfg
) {
    if (cfg.sample_rate <= 0.0 || cfg.samples_per_frame() == 0) {
        return 0;
    }

    const uhd::time_spec_t aligned_start_time =
        raw_start_time + uhd::time_spec_t(static_cast<double>(discard_samples) / cfg.sample_rate);
    const double delta_samples =
        (aligned_start_time - stream_start_time).get_real_secs() * cfg.sample_rate;
    const int64_t rounded_samples = static_cast<int64_t>(std::llround(delta_samples));
    const int64_t frame_samples = static_cast<int64_t>(cfg.samples_per_frame());
    if (frame_samples <= 0) {
        return 0;
    }

    const int64_t nearest_frame_seq =
        round_divide_nearest_away_from_zero(rounded_samples, frame_samples);
    const int64_t residual = rounded_samples - nearest_frame_seq * frame_samples;
    return normalize_alignment_samples(residual, frame_samples);
}

constexpr uint64_t kSystemDelayEstimationFrameInterval = 434;

}

class SensingChannel::PairedFrameQueue {
public:
    PairedFrameQueue(size_t rx_capacity, size_t tx_capacity, uint32_t logical_id, size_t frame_samples)
      : _rx_q(rx_capacity),
        _tx_q(tx_capacity),
        _logical_id(logical_id),
        _frame_samples(frame_samples) {}

    ~PairedFrameQueue() {
        close();
    }

    void set_rx_recycler(std::function<void(AlignedVector&&)> recycler) {
        _recycle_rx = std::move(recycler);
    }

    bool push_rx(RxSymbolsFrame&& frame) {
        if (_closed.load(std::memory_order_relaxed)) {
            return false;
        }
        return _rx_q.try_push(std::move(frame));
    }

    bool push_tx(TxSymbolsFrame&& frame) {
        if (_closed.load(std::memory_order_relaxed)) {
            return false;
        }
        return _tx_q.try_push(std::move(frame));
    }

    bool pop_pair(AlignedVector& rx_out, TxSymbolsFrame& tx_out) {
        RxSymbolsFrame rx_item;
        TxSymbolsFrame tx_item;
        bool have_rx = false;
        bool have_tx = false;

        while (true) {
            if (!have_rx) {
                if (!_pop_next_rx(rx_item)) {
                    return false;
                }
                have_rx = true;
            }

            if (!have_tx) {
                if (!_pop_next_tx(tx_item)) {
                    _recycle_one(std::move(rx_item.samples));
                    return false;
                }
                have_tx = true;
            }

            if (rx_item.frame_seq == tx_item.frame_seq) {
                rx_out = std::move(rx_item.samples);
                tx_out = std::move(tx_item);
                return true;
            }

            if (rx_item.frame_seq < tx_item.frame_seq) {
                _log_drop_rx(rx_item.frame_seq, tx_item.frame_seq);
                _recycle_one(std::move(rx_item.samples));
                have_rx = false;
                continue;
            }

            _log_drop_tx(rx_item.frame_seq, tx_item.frame_seq);
            have_tx = false;
        }
    }

    void close() {
        _closed.store(true, std::memory_order_relaxed);
    }

    PairedFrameQueue(const PairedFrameQueue&) = delete;
    PairedFrameQueue& operator=(const PairedFrameQueue&) = delete;
    PairedFrameQueue(PairedFrameQueue&&) = delete;
    PairedFrameQueue& operator=(PairedFrameQueue&&) = delete;

private:
    bool _pop_next_rx(RxSymbolsFrame& out) {
        SPSCBackoff backoff;
        while (true) {
            if (_rx_q.try_pop(out)) {
                backoff.reset();
                return true;
            }
            if (_closed.load(std::memory_order_relaxed)) {
                return false;
            }
            backoff.pause();
        }
    }

    bool _pop_next_tx(TxSymbolsFrame& out) {
        SPSCBackoff backoff;
        while (true) {
            if (_tx_q.try_pop(out)) {
                backoff.reset();
                return true;
            }
            if (_closed.load(std::memory_order_relaxed)) {
                return false;
            }
            backoff.pause();
        }
    }

    void _recycle_one(AlignedVector&& frame) {
        if (_recycle_rx) {
            _recycle_rx(std::move(frame));
        }
    }

    void _log_drop_rx(uint64_t rx_seq, uint64_t tx_seq) {
        _rx_drop_count++;
        const uint64_t gap_frames = tx_seq - rx_seq;
        const int64_t signed_gap = static_cast<int64_t>(tx_seq) - static_cast<int64_t>(rx_seq);
        _track_gap(signed_gap);
        if (_rx_drop_count <= 20 || (_rx_drop_count % 100) == 0 || gap_frames >= 2) {
            LOG_RT_WARN() << "[Sensing CH " << _logical_id
                          << "] drop RX frame due to seq mismatch: rx_seq="
                          << rx_seq << ", tx_seq=" << tx_seq
                          << ", gap_frames=" << gap_frames
                          << ", gap_samples=" << (gap_frames * _frame_samples)
                          << ", drop_count=" << _rx_drop_count;
        }
    }

    void _log_drop_tx(uint64_t rx_seq, uint64_t tx_seq) {
        _tx_drop_count++;
        const uint64_t gap_frames = rx_seq - tx_seq;
        const int64_t signed_gap = static_cast<int64_t>(tx_seq) - static_cast<int64_t>(rx_seq);
        _track_gap(signed_gap);
        if (_tx_drop_count <= 20 || (_tx_drop_count % 100) == 0 || gap_frames >= 2) {
            LOG_RT_WARN() << "[Sensing CH " << _logical_id
                          << "] drop TX frame due to seq mismatch: rx_seq="
                          << rx_seq << ", tx_seq=" << tx_seq
                          << ", gap_frames=" << gap_frames
                          << ", gap_samples=" << (gap_frames * _frame_samples)
                          << ", drop_count=" << _tx_drop_count;
        }
    }

    void _track_gap(int64_t signed_gap) {
        if (signed_gap == _last_signed_gap) {
            _same_gap_streak++;
        } else {
            _last_signed_gap = signed_gap;
            _same_gap_streak = 1;
        }

        const uint64_t abs_gap = static_cast<uint64_t>(std::llabs(signed_gap));
        if (abs_gap == 0) return;
        if (_same_gap_streak == 8 || (_same_gap_streak % 200) == 0) {
            LOG_RT_WARN() << "[Sensing CH " << _logical_id
                          << "] stable TX/RX seq gap detected: signed_gap_frames="
                          << signed_gap
                          << ", abs_gap_samples=" << (abs_gap * _frame_samples)
                          << ", streak=" << _same_gap_streak
                          << " (possible fixed integer-frame system latency)";
        }
    }

    SPSCRingBuffer<RxSymbolsFrame> _rx_q;
    SPSCRingBuffer<TxSymbolsFrame> _tx_q;
    std::atomic<bool> _closed{false};
    std::function<void(AlignedVector&&)> _recycle_rx;
    uint32_t _logical_id = 0;
    size_t _frame_samples = 0;
    uint64_t _rx_drop_count = 0;
    uint64_t _tx_drop_count = 0;
    int64_t _last_signed_gap = std::numeric_limits<int64_t>::min();
    uint64_t _same_gap_streak = 0;
};

SensingChannel::RxIoContext::RxIoContext(const Config& cfg, const SensingRxChannelConfig& c, uint32_t logical_id_)
  : logical_id(logical_id_),
    channel_cfg(c),
    rx_frame_pool(32, [&cfg]() {
        return AlignedVector(cfg.samples_per_frame());
    }),
    paired_queue(std::make_unique<PairedFrameQueue>(
        cfg.paired_frame_queue_size,
        cfg.paired_frame_queue_size,
        logical_id_,
        cfg.samples_per_frame()
    )),
    target_alignment(c.alignment),
    discard_samples(c.alignment) {}

SensingChannel::RxIoContext::~RxIoContext() = default;

SensingChannel::SensingComputeContext::SensingComputeContext(const Config& cfg, const SensingRxChannelConfig& c)
  : demod_fft_in(cfg.fft_size),
    demod_fft_out(cfg.fft_size),
    sensing_core({cfg.fft_size, cfg.range_fft_size, cfg.doppler_fft_size, cfg.sensing_symbol_num}),
    sensing_sender(c.sensing_ip, c.sensing_port, c.enable_sensing_output),
    next_hb_time(std::chrono::steady_clock::now()) {
    accumulated_rx_symbols.reserve(cfg.sensing_symbol_num);
    accumulated_tx_symbols.reserve(cfg.sensing_symbol_num);
    range_window.resize(cfg.fft_size);
    WindowGenerator::generate_hamming(range_window, cfg.fft_size);
    doppler_window.resize(cfg.sensing_symbol_num);
    WindowGenerator::generate_hamming(doppler_window, cfg.sensing_symbol_num);
    actual_subcarrier_indices.resize(cfg.fft_size);
    subcarrier_phases_unit_delay.resize(cfg.fft_size);
    const size_t half_fft = cfg.fft_size / 2;
    #pragma omp simd simdlen(16)
    for (size_t i = 0; i < cfg.fft_size; ++i) {
        const int k = (i >= half_fft) ? (static_cast<int>(i) - static_cast<int>(cfg.fft_size)) : static_cast<int>(i);
        actual_subcarrier_indices[i] = k;
        subcarrier_phases_unit_delay[i] = -2.0f * static_cast<float>(M_PI) * static_cast<float>(k) /
            static_cast<float>(cfg.fft_size);
    }

    demod_fft_plan = fftwf_plan_dft_1d(
        static_cast<int>(cfg.fft_size),
        reinterpret_cast<fftwf_complex*>(demod_fft_in.data()),
        reinterpret_cast<fftwf_complex*>(demod_fft_out.data()),
        FFTW_FORWARD,
        FFTW_MEASURE
    );

    delay_estimation_enabled = c.enable_system_delay_estimation;
    sensing_pipeline_disabled_by_mode = delay_estimation_enabled;
    if (delay_estimation_enabled) {
        if (cfg.num_symbols == 0 || cfg.sync_pos >= cfg.num_symbols) {
            LOG_G_WARN() << "[Sensing] system delay estimation disabled due to invalid sync config: "
                      << "num_symbols=" << cfg.num_symbols
                      << ", sync_pos=" << cfg.sync_pos;
            delay_estimation_enabled = false;
            sensing_pipeline_disabled_by_mode = false;
        } else {
            AlignedVector zc_freq(cfg.fft_size);
            ZadoffChuGenerator::generate(zc_freq, cfg.fft_size, cfg.zc_root);
            system_delay_sync = std::make_unique<SyncProcessor>(
                cfg.samples_per_frame(),
                cfg.fft_size,
                cfg.cp_length,
                zc_freq
            );
            system_delay_symbol_len = cfg.fft_size + cfg.cp_length;
            system_delay_expected_sync_pos = cfg.sync_pos * system_delay_symbol_len;
        }
    }
}

SensingChannel::SensingComputeContext::~SensingComputeContext() {
    if (demod_fft_plan != nullptr) {
        fftwf_destroy_plan(demod_fft_plan);
        demod_fft_plan = nullptr;
    }
}

SensingChannel::SensingChannel(
    const Config& cfg,
    const SensingRxChannelConfig& channel_cfg,
    uint32_t logical_id,
    std::atomic<bool>& running_ref,
    HeartbeatCallback heartbeat_sender,
    CoreResolver core_resolver
)
  : _cfg(cfg),
    _running_ref(running_ref),
    _heartbeat_sender(std::move(heartbeat_sender)),
    _core_resolver(std::move(core_resolver)),
    _rx_io(cfg, channel_cfg, logical_id),
    _compute(cfg, channel_cfg) {
    _rx_io.paired_queue->set_rx_recycler([this](AlignedVector&& frame) {
        this->_rx_io.rx_frame_pool.release(std::move(frame));
    });

    if (_compute.delay_estimation_enabled && _compute.sensing_pipeline_disabled_by_mode) {
        LOG_G_INFO() << "[Sensing CH " << _rx_io.logical_id
                  << "] enable_system_delay_estimation=1, sensing pipeline disabled; "
                  << "system delay estimation runs every "
                  << kSystemDelayEstimationFrameInterval << " frames."
                  ;
    } else if (_rx_io.channel_cfg.enable_system_delay_estimation && !_compute.delay_estimation_enabled) {
        LOG_G_WARN() << "[Sensing CH " << _rx_io.logical_id
                  << "] enable_system_delay_estimation requested but disabled due to invalid config."
                  ;
    }
}

SensingChannel::~SensingChannel() {
    stop();
    join();
}

void SensingChannel::start(const uhd::time_spec_t& start_time) {
    _stop_requested.store(false, std::memory_order_relaxed);
    _compute.bistatic_active = false;
    _compute.next_delay_estimation_frame_seq = 0;
    _rx_io.stream_start_time = start_time;
    _rx_io.next_rx_frame_seq = 0;
    _compute.sensing_sender.start();
    _rx_io.rx_thread = std::thread(&SensingChannel::_rx_loop, this, start_time);
    _compute.sensing_thread = std::thread(&SensingChannel::_sensing_loop, this);
}

void SensingChannel::stop() {
    const bool already_stopped = _stop_requested.exchange(true, std::memory_order_relaxed);
    if (already_stopped) return;
    _compute.bistatic_active = false;
    if (_rx_io.paired_queue) {
        _rx_io.paired_queue->close();
    }
}

void SensingChannel::join() {
    if (_rx_io.rx_thread.joinable()) _rx_io.rx_thread.join();
    if (_compute.sensing_thread.joinable()) _compute.sensing_thread.join();
    _compute.sensing_sender.stop();
}

void SensingChannel::start_bistatic() {
    _compute.bistatic_active = true;
    _compute.next_hb_time = std::chrono::steady_clock::now();
    _compute.sensing_sender.start();
}

void SensingChannel::stop_bistatic() {
    _compute.bistatic_active = false;
    _compute.sensing_sender.stop();
}

bool SensingChannel::enqueue_tx_symbols(
    const std::shared_ptr<const SymbolVector>& symbols,
    uint64_t frame_start_symbol_index,
    uint64_t frame_seq
) {
    TxSymbolsFrame tx_frame;
    tx_frame.symbols = symbols;
    tx_frame.frame_start_symbol_index = frame_start_symbol_index;
    tx_frame.frame_seq = frame_seq;
    const bool pushed = _rx_io.paired_queue->push_tx(std::move(tx_frame));
    if (!pushed) {
        LOG_RT_WARN_HZ(5) << "[Sensing CH " << _rx_io.logical_id
                          << "] paired TX queue full, dropping newest TX sensing frame";
    }
    return pushed;
}

void SensingChannel::process_bistatic_frame(const SensingFrame& frame, uint64_t frame_start_symbol_index) {
    if (!_compute.bistatic_active) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    _send_heartbeat_if_due(now);
    if (_compute.sensing_pipeline_disabled_by_mode) {
        return;
    }

    const size_t symbols_in_frame = std::min(frame.rx_symbols.size(), frame.tx_symbols.size());
    if (symbols_in_frame == 0) {
        return;
    }

    const uint64_t frame_start = frame_start_symbol_index;
    const uint64_t frame_end = frame_start + static_cast<uint64_t>(symbols_in_frame);
    if (_compute.next_symbol_to_sample < frame_start) {
        _compute.next_symbol_to_sample = frame_start;
    }

    while (_compute.next_symbol_to_sample < frame_end) {
        _apply_shared_sensing_if_due(_compute.next_symbol_to_sample);
        if (_compute.next_symbol_to_sample < frame_start) {
            _compute.next_symbol_to_sample = frame_start;
        }
        if (_compute.next_symbol_to_sample >= frame_end) break;

        const auto gather_start = std::chrono::steady_clock::now();
        const size_t symbol_idx = static_cast<size_t>(_compute.next_symbol_to_sample - frame_start);
        if (symbol_idx >= symbols_in_frame) {
            break;
        }

        AlignedVector rx_symbol = frame.rx_symbols[symbol_idx];
        const int relative_symbol_index = static_cast<int>(symbol_idx) - static_cast<int>(_cfg.sync_pos);
        const float phase_diff_cfo = frame.CFO * static_cast<float>(relative_symbol_index);
        const size_t phase_bins = std::min(rx_symbol.size(), _compute.actual_subcarrier_indices.size());
        #pragma omp simd simdlen(16)
        for (size_t j = 0; j < phase_bins; ++j) {
            const float phase_diff_sfo =
                frame.SFO * static_cast<float>(_compute.actual_subcarrier_indices[j]) *
                static_cast<float>(relative_symbol_index);
            const float phase_diff_delay = _compute.subcarrier_phases_unit_delay[j] * frame.delay_offset;
            const float phase_diff_total = phase_diff_delay + phase_diff_sfo + phase_diff_cfo;
            const std::complex<float> phase = std::polar(1.0f, -phase_diff_total);
            rx_symbol[j] *= phase;
        }

        if (!_compute.batch_has_first_symbol) {
            _compute.current_batch_first_symbol = _compute.next_symbol_to_sample;
            _compute.batch_has_first_symbol = true;
        }

        _compute.accumulated_rx_symbols.push_back(std::move(rx_symbol));
        _compute.accumulated_tx_symbols.push_back(frame.tx_symbols[symbol_idx]);
        _compute.next_symbol_to_sample += _compute.active_stride;
        _compute.pending_batch_gather_us += std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - gather_start).count();

        if (_compute.accumulated_tx_symbols.size() >= _cfg.sensing_symbol_num) {
            SensingFrame sensing_frame;
            sensing_frame.rx_symbols = std::move(_compute.accumulated_rx_symbols);
            sensing_frame.tx_symbols = std::move(_compute.accumulated_tx_symbols);
            _compute.accumulated_rx_symbols.clear();
            _compute.accumulated_tx_symbols.clear();

            const uint64_t first_symbol = _compute.current_batch_first_symbol;
            _compute.batch_has_first_symbol = false;
            const double gather_us = _compute.pending_batch_gather_us;
            _compute.pending_batch_gather_us = 0.0;
            _sensing_process_freq(sensing_frame, first_symbol, gather_us);
        }
    }
}

void SensingChannel::set_target_alignment(int32_t samples) {
    const int64_t frame_samples = static_cast<int64_t>(_cfg.samples_per_frame());
    if (frame_samples > 0 && samples != 0 && (samples % frame_samples) == 0) {
        LOG_G_WARN() << "[Sensing CH " << _rx_io.logical_id
                  << "] target ALGN equals an integer number of frames: samples="
                  << samples << ", frame_samples=" << frame_samples
                  << ", frame_shift=" << (samples / frame_samples);
    }
    _rx_io.target_alignment.store(samples);
    LOG_G_INFO() << "Set target ALGN for channel " << _rx_io.logical_id
              << ": " << samples << " samples";
}

void SensingChannel::set_alignment(int32_t samples) {
    const int64_t frame_samples = static_cast<int64_t>(_cfg.samples_per_frame());
    if (frame_samples > 0 && samples != 0 && (samples % frame_samples) == 0) {
        LOG_G_WARN() << "[Sensing CH " << _rx_io.logical_id
                  << "] ALGN equals an integer number of frames: samples="
                  << samples << ", frame_samples=" << frame_samples
                  << ", frame_shift=" << (samples / frame_samples);
    }
    _rx_io.discard_samples.store(samples);
    _rx_io.rx_state.store(RxState::ALIGNMENT);
    LOG_G_INFO() << "Set ALGN for channel " << _rx_io.logical_id
              << ": " << samples << " samples";
}

bool SensingChannel::set_rx_gain(double requested_gain_db, double* applied_gain_db) {
    if (!_rx_io.rx_usrp) {
        LOG_G_WARN() << "RXGN ignored for channel " << _rx_io.logical_id
                     << ": RX USRP not initialized";
        return false;
    }

    const auto gain_range = _rx_io.rx_usrp->get_rx_gain_range(_rx_io.channel_cfg.usrp_channel);
    const double clamped_gain = std::clamp(requested_gain_db, gain_range.start(), gain_range.stop());
    _rx_io.channel_cfg.rx_gain = clamped_gain;
    _rx_io.rx_usrp->set_rx_gain(clamped_gain, _rx_io.channel_cfg.usrp_channel);

    if (applied_gain_db != nullptr) {
        *applied_gain_db = clamped_gain;
    }
    LOG_G_INFO() << "Set RXGN for channel " << _rx_io.logical_id
              << " (USRP ch " << _rx_io.channel_cfg.usrp_channel
              << "): " << clamped_gain << " dB";
    return true;
}

void SensingChannel::apply_shared_cfg(const SharedSensingRuntime& snapshot) {
    std::lock_guard<std::mutex> lock(_shared_cfg_mutex);
    _pending_shared_cfg = snapshot;
    _has_pending_shared_cfg = true;
}

uint32_t SensingChannel::logical_id() const {
    return _rx_io.logical_id;
}

int32_t SensingChannel::target_alignment() const {
    return _rx_io.target_alignment.load(std::memory_order_relaxed);
}

const SensingRxChannelConfig& SensingChannel::channel_cfg() const {
    return _rx_io.channel_cfg;
}

void SensingChannel::initialize_rx_hardware_and_sync(
    const Config& cfg,
    const uhd::tune_request_t& tune_req,
    const std::string& tx_device_args,
    const std::string& tx_clock_source,
    const std::string& tx_time_source,
    const uhd::usrp::multi_usrp::sptr& tx_usrp,
    std::vector<std::unique_ptr<SensingChannel>>& channels
) {
    struct CachedUsrpContext {
        uhd::usrp::multi_usrp::sptr usrp;
        std::string clock_source;
        std::string time_source;
    };
    std::unordered_map<std::string, CachedUsrpContext> rx_usrp_cache;
    rx_usrp_cache.emplace(tx_device_args, CachedUsrpContext{tx_usrp, tx_clock_source, tx_time_source});

    auto resolve_rx_device_args = [&cfg](const SensingRxChannelConfig& ch_cfg) -> std::string {
        if (!ch_cfg.device_args.empty()) return ch_cfg.device_args;
        if (!cfg.rx_device_args.empty()) return cfg.rx_device_args;
        return cfg.device_args;
    };

    auto resolve_rx_clock_source = [&cfg](const SensingRxChannelConfig& ch_cfg) -> std::string {
        if (!ch_cfg.clock_source.empty()) return ch_cfg.clock_source;
        if (!cfg.rx_clock_source.empty()) return cfg.rx_clock_source;
        return cfg.clocksource;
    };

    auto resolve_rx_time_source = [&cfg](const SensingRxChannelConfig& ch_cfg, const std::string& resolved_clock_source) -> std::string {
        if (!ch_cfg.time_source.empty()) return ch_cfg.time_source;
        if (!cfg.rx_time_source.empty()) return cfg.rx_time_source;
        if (!cfg.timesource.empty()) return cfg.timesource;
        return resolved_clock_source;
    };

    auto resolve_rx_wire_format = [&cfg](const SensingRxChannelConfig& ch_cfg) -> std::string {
        if (!ch_cfg.wire_format_rx.empty()) return ch_cfg.wire_format_rx;
        return cfg.wire_format_rx;
    };

    auto get_or_create_rx_usrp = [&](const std::string& rx_args, const std::string& rx_clock_source, const std::string& rx_time_source) -> uhd::usrp::multi_usrp::sptr {
        auto it = rx_usrp_cache.find(rx_args);
        if (it != rx_usrp_cache.end()) {
            if (it->second.clock_source != rx_clock_source || it->second.time_source != rx_time_source) {
                throw std::runtime_error(
                    "Clock/time source conflict on shared RX USRP (" + rx_args + "): existing clock='" +
                    it->second.clock_source + "', requested clock='" + rx_clock_source +
                    "', existing time='" + it->second.time_source + "', requested time='" + rx_time_source +
                    "'. Use distinct device_args if different REF/PPS sources are required.");
            }
            return it->second.usrp;
        }
        auto usrp = uhd::usrp::multi_usrp::make(rx_args);
        usrp->set_clock_source(rx_clock_source);
        usrp->set_time_source(rx_time_source);
        usrp->set_rx_rate(cfg.sample_rate);
        rx_usrp_cache.emplace(rx_args, CachedUsrpContext{usrp, rx_clock_source, rx_time_source});
        return usrp;
    };

    for (auto& ch : channels) {
        auto& io = ch->_rx_io;
        const std::string rx_device_args = resolve_rx_device_args(io.channel_cfg);
        const std::string rx_clock_source = resolve_rx_clock_source(io.channel_cfg);
        const std::string rx_time_source = resolve_rx_time_source(io.channel_cfg, rx_clock_source);
        const std::string rx_wire_format = resolve_rx_wire_format(io.channel_cfg);
        io.rx_usrp = get_or_create_rx_usrp(rx_device_args, rx_clock_source, rx_time_source);

        if (!io.rx_usrp) {
            throw std::runtime_error("Failed to initialize RX USRP for sensing channel " + std::to_string(io.logical_id));
        }

        const size_t usrp_rx_channels = io.rx_usrp->get_rx_num_channels();
        if (io.channel_cfg.usrp_channel >= usrp_rx_channels) {
            throw std::runtime_error(
                "Configured sensing RX channel out of range: " +
                std::to_string(io.channel_cfg.usrp_channel) +
                " (USRP supports " + std::to_string(usrp_rx_channels) + " RX channels)");
        }

        io.rx_usrp->set_rx_rate(cfg.sample_rate);
        io.rx_usrp->set_rx_freq(tune_req, io.channel_cfg.usrp_channel);
        io.rx_usrp->set_rx_gain(io.channel_cfg.rx_gain, io.channel_cfg.usrp_channel);
        io.rx_usrp->set_rx_bandwidth(cfg.bandwidth, io.channel_cfg.usrp_channel);
        if (!io.channel_cfg.rx_antenna.empty()) {
            io.rx_usrp->set_rx_antenna(io.channel_cfg.rx_antenna, io.channel_cfg.usrp_channel);
        }

        uhd::stream_args_t rx_stream_args("fc32", rx_wire_format);
        rx_stream_args.args["block_id"] = "radio";
        rx_stream_args.channels = {io.channel_cfg.usrp_channel};
        io.rx_stream = io.rx_usrp->get_rx_stream(rx_stream_args);
    }

    struct SyncUsrpEntry {
        std::string args;
        uhd::usrp::multi_usrp::sptr usrp;
    };
    struct SyncSnapshot {
        double now_s = 0.0;
        double last_pps_s = 0.0;
    };
    std::vector<SyncUsrpEntry> sync_usrps;
    sync_usrps.reserve(rx_usrp_cache.size());
    for (const auto& entry : rx_usrp_cache) {
        if (entry.second.usrp) {
            sync_usrps.push_back(SyncUsrpEntry{entry.first, entry.second.usrp});
        }
    }

    auto collect_sync_snapshot = [&]() {
        std::vector<SyncSnapshot> snapshots;
        snapshots.reserve(sync_usrps.size());
        for (const auto& entry : sync_usrps) {
            SyncSnapshot s;
            s.now_s = entry.usrp->get_time_now().get_real_secs();
            s.last_pps_s = entry.usrp->get_time_last_pps().get_real_secs();
            snapshots.push_back(s);
        }
        return snapshots;
    };

    auto print_last_pps_status = [&](const std::string& title, const std::vector<SyncSnapshot>& snapshots) {
        if (sync_usrps.empty()) return;
        LOG_G_INFO() << title;
        for (size_t i = 0; i < sync_usrps.size(); ++i) {
            const auto& entry = sync_usrps[i];
            const double pps_s = snapshots[i].last_pps_s;
            LOG_G_INFO() << std::fixed << std::setprecision(9)
                      << "  [USRP " << i << "] args='" << entry.args
                      << "', time_last_pps=" << pps_s << " s"
                      << std::defaultfloat;
        }
    };

    auto latest_sync_snapshot = collect_sync_snapshot();
    if (sync_usrps.size() > 1) {
        try {
            for (const auto& entry : sync_usrps) {
                entry.usrp->set_time_next_pps(uhd::time_spec_t(0.0));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1100));
            latest_sync_snapshot = collect_sync_snapshot();
            LOG_G_INFO() << "Time synchronized across " << sync_usrps.size()
                         << " USRPs using next PPS.";
            print_last_pps_status("Post-sync USRP PPS status:", latest_sync_snapshot);

            bool all_zero_last_pps = true;
            for (const auto& s : latest_sync_snapshot) {
                if (std::abs(s.last_pps_s) > 1e-9) {
                    all_zero_last_pps = false;
                    break;
                }
            }
            if (all_zero_last_pps) {
                LOG_G_INFO() << "PPS verification success: all USRPs report time_last_pps == 0.";
            } else {
                LOG_G_WARN() << "Warning: PPS verification failed: not all USRPs report time_last_pps == 0."
                             ;
            }
        } catch (const std::exception& e) {
            LOG_G_WARN() << "PPS time sync failed (" << e.what()
                         << "), fallback to set_time_now per device.";
            for (const auto& entry : sync_usrps) {
                entry.usrp->set_time_now(uhd::time_spec_t(0.0));
            }
            latest_sync_snapshot = collect_sync_snapshot();
            print_last_pps_status("Fallback set_time_now USRP PPS status:", latest_sync_snapshot);
        }
    } else if (!sync_usrps.empty()) {
        sync_usrps.front().usrp->set_time_now(uhd::time_spec_t(0.0));
        latest_sync_snapshot = collect_sync_snapshot();
        print_last_pps_status("Single-USRP PPS status:", latest_sync_snapshot);
    }

    return;
}

size_t SensingChannel::_resolve_core(size_t hint) const {
    if (_core_resolver) return _core_resolver(hint);
    return 0;
}

void SensingChannel::_rx_loop(const uhd::time_spec_t& start_time) {
    async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
    uhd::set_thread_priority_safe(1.0f, true);
    std::vector<size_t> cpu_list = {_resolve_core(3 + static_cast<size_t>(_rx_io.logical_id) * 2)};
    uhd::set_thread_affinity(cpu_list);
    prefault_thread_stack();

    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    stream_cmd.stream_now = false;
    stream_cmd.time_spec = start_time;
    _rx_io.stream_start_time = start_time;
    _rx_io.rx_stream->issue_stream_cmd(stream_cmd);

    while (_running_ref.load(std::memory_order_relaxed)) {
        switch (_rx_io.rx_state.load()) {
        case RxState::ALIGNMENT:
            _handle_alignment();
            break;
        case RxState::NORMAL:
            _handle_normal_rx();
            break;
        }
    }

    _rx_io.rx_stream->issue_stream_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
}

void SensingChannel::_handle_alignment() {
    const int32_t discard = _rx_io.discard_samples.load();
    const int64_t frame_samples = static_cast<int64_t>(_cfg.samples_per_frame());
    if (frame_samples > 0 && discard != 0 && (discard % frame_samples) == 0) {
        LOG_RT_WARN() << "[Sensing CH " << _rx_io.logical_id
                      << "] applying ALGN with integer-frame shift: discard="
                      << discard << ", frame_samples=" << frame_samples
                      << ", frame_shift=" << (discard / frame_samples);
    }
    const int64_t total_read_signed =
        static_cast<int64_t>(_cfg.samples_per_frame()) + static_cast<int64_t>(discard);
    if (total_read_signed <= 0) {
        LOG_RT_WARN() << "[Sensing CH " << _rx_io.logical_id
                      << "] alignment total_read <= 0 (discard=" << discard
                      << ", frame_samples=" << _cfg.samples_per_frame()
                      << "). This can manifest as integer-frame TX/RX offset.";
    }
    const size_t total_read = static_cast<size_t>(total_read_signed);
    AlignedVector temp_buf(total_read);
    uhd::rx_metadata_t md;
    size_t received = 0;
    bool have_first_time = false;
    uhd::time_spec_t first_time_spec(0.0);

    while (received < total_read && _running_ref.load(std::memory_order_relaxed)) {
        size_t num_rx = _rx_io.rx_stream->recv(
            temp_buf.data() + received,
            total_read - received,
            md,
            1.0,
            false
        );

        if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
            if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
                LOG_RT_WARN() << "RX alignment error: " << md.strerror();
                received = 0;
                have_first_time = false;
            }
            continue;
        }

        if (!have_first_time && num_rx > 0 && md.has_time_spec) {
            first_time_spec = md.time_spec;
            have_first_time = true;
        }
        received += num_rx;
    }

    if (!_running_ref.load(std::memory_order_relaxed)) return;

    AlignedVector aligned_frame = _rx_io.rx_frame_pool.acquire();
    if (discard >= 0) {
        const size_t positive_discard = static_cast<size_t>(discard);
        std::copy(
            temp_buf.begin() + positive_discard,
            temp_buf.begin() + positive_discard + _cfg.samples_per_frame(),
            aligned_frame.begin()
        );
    } else {
        const size_t shift = static_cast<size_t>(-discard);
        std::fill(aligned_frame.begin(), aligned_frame.end(), std::complex<float>(0.0f, 0.0f));
        std::copy(
            temp_buf.begin(),
            temp_buf.begin() + total_read,
            aligned_frame.begin() + shift
        );
    }

    const uint64_t frame_seq = have_first_time
        ? compute_rx_frame_seq_from_time(
            first_time_spec,
            _rx_io.stream_start_time,
            discard,
            _rx_io.target_alignment.load(std::memory_order_relaxed),
            _cfg)
        : _rx_io.next_rx_frame_seq;
    RxSymbolsFrame rx_item;
    rx_item.samples = std::move(aligned_frame);
    rx_item.frame_seq = frame_seq;
    if (!_rx_io.paired_queue->push_rx(std::move(rx_item))) {
        LOG_RT_WARN_HZ(5) << "[Sensing CH " << _rx_io.logical_id
                          << "] paired RX queue full during alignment, dropping newest RX frame";
        _rx_io.next_rx_frame_seq = frame_seq + 1;
        _rx_io.rx_frame_pool.release(std::move(rx_item.samples));
        return;
    }
    _rx_io.next_rx_frame_seq = frame_seq + 1;

    _rx_io.rx_state.store(RxState::NORMAL);
    LOG_RT_INFO() << "[Sensing CH " << _rx_io.logical_id << "] aligned. ALGN="
                  << discard << " samples.";
}

void SensingChannel::_handle_normal_rx() {
    AlignedVector rx_frame = _rx_io.rx_frame_pool.acquire();
    uhd::rx_metadata_t md;
    size_t received = 0;
    bool have_first_time = false;
    uhd::time_spec_t first_time_spec(0.0);

    while (received < _cfg.samples_per_frame() && _running_ref.load(std::memory_order_relaxed)) {
        size_t num_rx = _rx_io.rx_stream->recv(
            rx_frame.data() + received,
            _cfg.samples_per_frame() - received,
            md,
            2.0,
            false
        );

        if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
            if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
                LOG_RT_WARN() << "RX error: " << md.strerror();
                received = 0;
                have_first_time = false;
            }
            continue;
        }

        if (!have_first_time && num_rx > 0 && md.has_time_spec) {
            first_time_spec = md.time_spec;
            have_first_time = true;
        }
        received += num_rx;
    }

    if (!_running_ref.load(std::memory_order_relaxed)) return;

    if (have_first_time) {
        const int64_t frame_samples = static_cast<int64_t>(_cfg.samples_per_frame());
        const int32_t desired_alignment = _rx_io.target_alignment.load(std::memory_order_relaxed);
        const int32_t current_alignment = _rx_io.discard_samples.load(std::memory_order_relaxed);
        const int32_t frame_offset = compute_rx_frame_boundary_error_samples(
            first_time_spec,
            _rx_io.stream_start_time,
            0,
            _cfg);
        const int32_t correction = normalize_alignment_samples(
            static_cast<int64_t>(desired_alignment) - static_cast<int64_t>(frame_offset),
            frame_samples);
        if (correction != 0) {
            LOG_RT_WARN_HZ(5) << "[Sensing CH " << _rx_io.logical_id
                              << "] RX frame boundary mismatch: offset=" << frame_offset
                              << ", target_ALGN=" << desired_alignment
                              << ", last_ALGN=" << current_alignment
                              << ", correction=" << correction;
            set_alignment(correction);
            _rx_io.rx_frame_pool.release(std::move(rx_frame));
            return;
        }
    }

    const uint64_t frame_seq = have_first_time
        ? compute_rx_frame_seq_from_time(
            first_time_spec,
            _rx_io.stream_start_time,
            0,
            _rx_io.target_alignment.load(std::memory_order_relaxed),
            _cfg)
        : _rx_io.next_rx_frame_seq;
    RxSymbolsFrame rx_item;
    rx_item.samples = std::move(rx_frame);
    rx_item.frame_seq = frame_seq;
    if (!_rx_io.paired_queue->push_rx(std::move(rx_item))) {
        LOG_RT_WARN_HZ(5) << "[Sensing CH " << _rx_io.logical_id
                          << "] paired RX queue full, dropping newest RX frame";
        _rx_io.next_rx_frame_seq = frame_seq + 1;
        _rx_io.rx_frame_pool.release(std::move(rx_item.samples));
        return;
    }
    _rx_io.next_rx_frame_seq = frame_seq + 1;
}

void SensingChannel::_sensing_loop() {
    async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
    uhd::set_thread_priority_safe(0.6f, true);
    std::vector<size_t> cpu_list = {_resolve_core(4 + static_cast<size_t>(_rx_io.logical_id) * 2)};
    uhd::set_thread_affinity(cpu_list);
    prefault_thread_stack();

    while (_running_ref.load(std::memory_order_relaxed)) {
        AlignedVector rx_frame_data;
        TxSymbolsFrame tx_frame;
        if (!_rx_io.paired_queue->pop_pair(rx_frame_data, tx_frame)) {
            break;
        }

        const auto now = std::chrono::steady_clock::now();
        _send_heartbeat_if_due(now);
        if (_compute.delay_estimation_enabled &&
            tx_frame.frame_seq >= _compute.next_delay_estimation_frame_seq) {
            _estimate_system_delay(rx_frame_data, tx_frame.frame_seq);
        }
        if (_compute.sensing_pipeline_disabled_by_mode) {
            _rx_io.rx_frame_pool.release(std::move(rx_frame_data));
            continue;
        }

        const auto& tx_symbols = *tx_frame.symbols;
        const uint64_t frame_start = tx_frame.frame_start_symbol_index;
        const uint64_t frame_end = frame_start + static_cast<uint64_t>(tx_symbols.size());
        if (_compute.next_symbol_to_sample < frame_start) {
            _compute.next_symbol_to_sample = frame_start;
        }

        while (_compute.next_symbol_to_sample < frame_end) {
            _apply_shared_sensing_if_due(_compute.next_symbol_to_sample);
            if (_compute.next_symbol_to_sample < frame_start) {
                _compute.next_symbol_to_sample = frame_start;
            }
            if (_compute.next_symbol_to_sample >= frame_end) break;

            const auto gather_start = std::chrono::steady_clock::now();
            const size_t symbol_idx = static_cast<size_t>(_compute.next_symbol_to_sample - frame_start);
            AlignedVector rx_symbol(_cfg.fft_size);
            const size_t symbol_start = symbol_idx * (_cfg.fft_size + _cfg.cp_length) + _cfg.cp_length;
            std::copy(
                rx_frame_data.begin() + symbol_start,
                rx_frame_data.begin() + symbol_start + _cfg.fft_size,
                rx_symbol.begin()
            );

            if (!_compute.batch_has_first_symbol) {
                _compute.current_batch_first_symbol = _compute.next_symbol_to_sample;
                _compute.batch_has_first_symbol = true;
            }

            _compute.accumulated_rx_symbols.push_back(std::move(rx_symbol));
            _compute.accumulated_tx_symbols.push_back(tx_symbols[symbol_idx]);
            _compute.next_symbol_to_sample += _compute.active_stride;
            _compute.pending_batch_gather_us += std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - gather_start).count();

            if (_compute.accumulated_tx_symbols.size() >= _cfg.sensing_symbol_num) {
                SensingFrame sensing_frame;
                sensing_frame.rx_symbols = std::move(_compute.accumulated_rx_symbols);
                sensing_frame.tx_symbols = std::move(_compute.accumulated_tx_symbols);
                _compute.accumulated_rx_symbols.clear();
                _compute.accumulated_tx_symbols.clear();

                const uint64_t first_symbol = _compute.current_batch_first_symbol;
                _compute.batch_has_first_symbol = false;
                const double gather_us = _compute.pending_batch_gather_us;
                _compute.pending_batch_gather_us = 0.0;
                _sensing_process(sensing_frame, first_symbol, gather_us);
            }
        }

        _rx_io.rx_frame_pool.release(std::move(rx_frame_data));
    }
}

void SensingChannel::_send_heartbeat_if_due(const std::chrono::steady_clock::time_point& now) {
    if (!_rx_io.channel_cfg.enable_sensing_output) {
        return;
    }
    if (now < _compute.next_hb_time) {
        return;
    }
    if (_heartbeat_sender) {
        _heartbeat_sender(_rx_io.channel_cfg.sensing_ip, _rx_io.channel_cfg.sensing_port);
    }
    _compute.next_hb_time = now + std::chrono::seconds(1);
}

void SensingChannel::_estimate_system_delay(const AlignedVector& rx_frame_data, uint64_t frame_seq) {
    if (!_compute.delay_estimation_enabled || !_compute.system_delay_sync) {
        return;
    }

    int max_pos = 0;
    float max_corr = 0.0f;
    float avg_corr = 0.0f;
    _compute.system_delay_sync->find_sync_position(rx_frame_data, max_pos, max_corr, avg_corr);

    const int64_t frame_samples = static_cast<int64_t>(_cfg.samples_per_frame());
    if (frame_samples <= 0) {
        return;
    }
    const int64_t raw_delay = static_cast<int64_t>(max_pos) -
        static_cast<int64_t>(_compute.system_delay_expected_sync_pos);
    int64_t delay_samples = raw_delay % frame_samples;
    if (delay_samples < 0) {
        delay_samples += frame_samples;
    }
    if (frame_samples > 1) {
        const int64_t half_frame = frame_samples / 2;
        if (delay_samples >= half_frame) {
            delay_samples -= frame_samples;
        }
    }

    const double delay_us = (_cfg.sample_rate > 0.0)
        ? (static_cast<double>(delay_samples) * 1e6 / _cfg.sample_rate)
        : 0.0;
    const float corr_ratio = (avg_corr > 0.0f) ? (max_corr / avg_corr) : 0.0f;
    const int32_t alignment_now = _rx_io.discard_samples.load(std::memory_order_relaxed);
    const int64_t suggested_alignment_raw = static_cast<int64_t>(alignment_now) + delay_samples;

    _compute.next_delay_estimation_frame_seq = frame_seq + kSystemDelayEstimationFrameInterval;

    LOG_RT_INFO() << "[SYSDLY CH " << _rx_io.logical_id
                  << "] frame_seq=" << frame_seq
                  << ", delay=" << delay_samples << " samp (" << delay_us << " us)"
                  << ", peak=" << max_pos
                  << ", expected=" << _compute.system_delay_expected_sync_pos
                  << ", corr_ratio=" << corr_ratio
                  << ", alignment_now=" << alignment_now
                  << ", alignment_suggest=" << suggested_alignment_raw
                  << ", next_frame_seq=" << _compute.next_delay_estimation_frame_seq
                  << ", interval=" << kSystemDelayEstimationFrameInterval;
}

void SensingChannel::_sensing_process(const SensingFrame& frame, uint64_t first_symbol_index, double gather_us) {
    using ProfileClock = std::chrono::steady_clock;
    auto& channel_buf = _compute.sensing_core.channel_buffer();
    const auto& sensing_params = _compute.sensing_core.params();
    const size_t range_stride = sensing_params.range_fft_size;
    const size_t doppler_slots = sensing_params.doppler_fft_size;

    const size_t symbol_count = std::min(frame.rx_symbols.size(), _cfg.sensing_symbol_num);
    const int plan_input_alignment = fftwf_alignment_of(
        reinterpret_cast<float*>(_compute.demod_fft_in.data())
    );
    const int plan_output_alignment = fftwf_alignment_of(
        reinterpret_cast<float*>(_compute.demod_fft_out.data())
    );
    const auto prep_start = ProfileClock::now();
    for (size_t i = 0; i < symbol_count; ++i) {
        if (i >= doppler_slots) {
            continue;
        }

        auto* fft_in = reinterpret_cast<fftwf_complex*>(
            const_cast<std::complex<float>*>(frame.rx_symbols[i].data())
        );
        auto* slot_out = reinterpret_cast<fftwf_complex*>(
            channel_buf.data() + i * range_stride
        );

        const int current_input_alignment = fftwf_alignment_of(
            reinterpret_cast<float*>(fft_in)
        );
        const int current_output_alignment = fftwf_alignment_of(
            reinterpret_cast<float*>(slot_out)
        );

        if (current_input_alignment == plan_input_alignment &&
            current_output_alignment == plan_output_alignment) {
            fftwf_execute_dft(_compute.demod_fft_plan, fft_in, slot_out);
        } else {
            std::copy(frame.rx_symbols[i].begin(), frame.rx_symbols[i].end(), _compute.demod_fft_in.begin());
            fftwf_execute(_compute.demod_fft_plan);
            _compute.sensing_core.copy_fft_result_to_buffer(i, _compute.demod_fft_out);
        }
    }

    const double prep_us = std::chrono::duration<double, std::micro>(
        ProfileClock::now() - prep_start).count();
    _sensing_process_finalize(frame.tx_symbols, first_symbol_index, gather_us, prep_us);
}

void SensingChannel::_sensing_process_freq(const SensingFrame& frame, uint64_t first_symbol_index, double gather_us) {
    using ProfileClock = std::chrono::steady_clock;
    const auto prep_start = ProfileClock::now();
    _compute.sensing_core.copy_symbols_to_buffer(frame.rx_symbols);
    const double prep_us = std::chrono::duration<double, std::micro>(
        ProfileClock::now() - prep_start).count();
    _sensing_process_finalize(frame.tx_symbols, first_symbol_index, gather_us, prep_us);
}

void SensingChannel::_sensing_process_finalize(
    const std::vector<AlignedVector>& tx_symbols,
    uint64_t first_symbol_index,
    double gather_us,
    double prep_us
) {
    using ProfileClock = std::chrono::steady_clock;
    const bool do_prof = should_profile_sensing(_cfg);
    auto& channel_buf = _compute.sensing_core.channel_buffer();
    const auto chest_start = ProfileClock::now();
    _compute.sensing_core.channel_estimate_with_shift(tx_symbols);
    const double chest_shift_us = std::chrono::duration<double, std::micro>(
        ProfileClock::now() - chest_start).count();
    const auto mti_start = ProfileClock::now();
    _compute.sensing_core.apply_mti(_compute.active_enable_mti);
    const double mti_us = std::chrono::duration<double, std::micro>(
        ProfileClock::now() - mti_start).count();

    double windows_fft_us = 0.0;
    if (!_compute.active_skip_sensing_fft) {
        const auto fft_start = ProfileClock::now();
        _compute.sensing_core.apply_windows(channel_buf, _compute.range_window, _compute.doppler_window);
        _compute.sensing_core.execute_range_ifft();
        _compute.sensing_core.execute_doppler_fft();
        windows_fft_us = std::chrono::duration<double, std::micro>(
            ProfileClock::now() - fft_start).count();
    }

    const auto send_start = ProfileClock::now();
    _compute.sensing_sender.push_data(channel_buf, first_symbol_index);
    const double send_us = std::chrono::duration<double, std::micro>(
        ProfileClock::now() - send_start).count();
    if (_cfg.range_fft_size != _cfg.fft_size || _cfg.doppler_fft_size != _cfg.fft_size) {
        _compute.sensing_core.clear_channel_buffer();
    }

    if (do_prof) {
        _compute.prof_gather_total_us += gather_us;
        _compute.prof_prep_total_us += prep_us;
        _compute.prof_chest_shift_total_us += chest_shift_us;
        _compute.prof_mti_total_us += mti_us;
        _compute.prof_windows_fft_total_us += windows_fft_us;
        _compute.prof_send_total_us += send_us;
        _compute.prof_batch_count++;

        constexpr uint64_t PROF_REPORT_INTERVAL = 64;
        if (_compute.prof_batch_count >= PROF_REPORT_INTERVAL) {
            const double n = static_cast<double>(_compute.prof_batch_count);
            const double total_latency_us =
                _compute.prof_prep_total_us +
                _compute.prof_chest_shift_total_us +
                _compute.prof_mti_total_us +
                _compute.prof_windows_fft_total_us;
            std::ostringstream oss;
            oss << "\n========== Sensing CH " << _rx_io.logical_id
                << " Profiling (avg per batch, us) ==========\n"
                << "Batch gather:            " << _compute.prof_gather_total_us / n << " us\n"
                << "RX symbol prep:          " << _compute.prof_prep_total_us / n << " us\n"
                << "ChEst + Shift:           " << _compute.prof_chest_shift_total_us / n << " us\n"
                << "MTI:                     " << _compute.prof_mti_total_us / n << " us\n"
                << "Windows+IFFT+DopFFT:     " << _compute.prof_windows_fft_total_us / n << " us\n"
                << "Send queue push:         " << _compute.prof_send_total_us / n << " us\n"
                << "TOTAL LATENCY (excl. gather/send): " << total_latency_us / n << " us\n"
                << "Profile batch count:     " << _compute.prof_batch_count << "\n"
                << "========================================================\n";
            LOG_RT_INFO() << oss.str();
            _compute.prof_gather_total_us = 0.0;
            _compute.prof_prep_total_us = 0.0;
            _compute.prof_chest_shift_total_us = 0.0;
            _compute.prof_mti_total_us = 0.0;
            _compute.prof_windows_fft_total_us = 0.0;
            _compute.prof_send_total_us = 0.0;
            _compute.prof_batch_count = 0;
        }
    }
}

void SensingChannel::_apply_shared_sensing_if_due(uint64_t symbol_index) {
    SharedSensingRuntime snapshot;
    bool should_apply = false;
    {
        std::lock_guard<std::mutex> lock(_shared_cfg_mutex);
        if (_has_pending_shared_cfg && _pending_shared_cfg.generation > _compute.applied_generation) {
            snapshot = _pending_shared_cfg;
            if (symbol_index >= _pending_shared_cfg.apply_symbol_index) {
                should_apply = true;
            }
        }
    }
    if (!should_apply) {
        return;
    }

    _compute.active_stride = std::max<size_t>(1, snapshot.sensing_symbol_stride);
    _compute.active_enable_mti = snapshot.enable_mti;
    _compute.active_skip_sensing_fft = snapshot.skip_sensing_fft;
    _compute.applied_generation = snapshot.generation;

    if (_compute.next_symbol_to_sample < snapshot.apply_symbol_index) {
        _compute.next_symbol_to_sample = snapshot.apply_symbol_index;
    }

    LOG_RT_INFO() << "[Sensing CH " << _rx_io.logical_id
                  << "] applied shared params at OFDM symbol "
                  << snapshot.apply_symbol_index
                  << " (stride=" << _compute.active_stride
                  << ", MTI=" << (_compute.active_enable_mti ? 1 : 0)
                  << ", SKIP=" << (_compute.active_skip_sensing_fft ? 1 : 0)
                  << ")";
}
