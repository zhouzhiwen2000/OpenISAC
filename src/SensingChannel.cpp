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
    return cfg.should_profile("sensing") ||
           cfg.should_profile("sensing_proc");
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

int64_t round_long_double_to_int64(long double value) {
    if (!std::isfinite(value)) {
        return 0;
    }
    const long double max_i64 = static_cast<long double>(std::numeric_limits<int64_t>::max());
    const long double min_i64 = static_cast<long double>(std::numeric_limits<int64_t>::min());
    if (value >= max_i64) {
        return std::numeric_limits<int64_t>::max();
    }
    if (value <= min_i64) {
        return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(std::llround(value));
}

int64_t compute_elapsed_samples_from_ticks(
    const uhd::time_spec_t& raw_start_time,
    const uhd::time_spec_t& stream_start_time,
    double sample_rate,
    double tick_rate)
{
    if (sample_rate <= 0.0 || tick_rate <= 0.0) {
        return 0;
    }

    const int64_t raw_ticks = raw_start_time.to_ticks(tick_rate);
    const int64_t stream_start_ticks = stream_start_time.to_ticks(tick_rate);
    const int64_t delta_ticks = raw_ticks - stream_start_ticks;
    const long double elapsed_samples =
        static_cast<long double>(delta_ticks) * static_cast<long double>(sample_rate) /
        static_cast<long double>(tick_rate);
    return round_long_double_to_int64(elapsed_samples);
}

uint64_t compute_rx_frame_seq_from_time(
    const uhd::time_spec_t& raw_start_time,
    const uhd::time_spec_t& stream_start_time,
    int32_t discard_samples,
    int32_t target_alignment_samples,
    const Config& cfg,
    double sample_rate,
    double tick_rate
) {
    if (sample_rate <= 0.0 || tick_rate <= 0.0 || cfg.samples_per_frame() == 0) {
        return 0;
    }

    const int64_t rounded_samples =
        compute_elapsed_samples_from_ticks(raw_start_time, stream_start_time, sample_rate, tick_rate) +
        static_cast<int64_t>(discard_samples);
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
    const Config& cfg,
    double sample_rate,
    double tick_rate
) {
    if (sample_rate <= 0.0 || tick_rate <= 0.0 || cfg.samples_per_frame() == 0) {
        return 0;
    }

    const int64_t rounded_samples =
        compute_elapsed_samples_from_ticks(raw_start_time, stream_start_time, sample_rate, tick_rate) +
        static_cast<int64_t>(discard_samples);
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

size_t sensing_symbol_capacity(const Config& cfg) {
    const CompactSensingMaskAnalysis analysis = analyze_compact_sensing_mask(cfg);
    if (!analysis.local_delay_doppler_supported) {
        return cfg.sensing_symbol_num;
    }
    return std::max(cfg.sensing_symbol_num, analysis.selected_symbol_count);
}

void copy_backend_range_view_input(
    const AlignedVector& src,
    size_t src_rows,
    size_t src_stride,
    size_t fft_rows,
    size_t view_cols,
    AlignedVector& dst)
{
    if (dst.size() != fft_rows * view_cols) {
        dst.assign(fft_rows * view_cols, std::complex<float>(0.0f, 0.0f));
    } else {
        std::fill(dst.begin(), dst.end(), std::complex<float>(0.0f, 0.0f));
    }

    const size_t rows_to_copy = std::min(src_rows, fft_rows);
    for (size_t row = 0; row < rows_to_copy; ++row) {
        const auto* src_row = src.data() + row * src_stride;
        auto* dst_row = dst.data() + row * view_cols;
        std::copy_n(src_row, view_cols, dst_row);
    }
}

void crop_doppler_view_output(
    const std::complex<float>* src,
    size_t full_rows,
    size_t cols,
    size_t output_rows,
    AlignedVector& dst)
{
    if (src == nullptr || full_rows == 0 || cols == 0 || output_rows == 0) {
        dst.clear();
        return;
    }

    if (output_rows >= full_rows) {
        dst.assign(src, src + full_rows * cols);
        return;
    }

    if (dst.size() != output_rows * cols) {
        dst.assign(output_rows * cols, std::complex<float>(0.0f, 0.0f));
    }

    const size_t shifted_start = (full_rows / 2) - (output_rows / 2);
    const size_t ifftshift_offset = output_rows / 2;
    for (size_t row = 0; row < output_rows; ++row) {
        const size_t shifted_row = (row + ifftshift_offset) % output_rows;
        const size_t src_row = (shifted_start + shifted_row + full_rows / 2) % full_rows;
        const auto* src_row_ptr = src + src_row * cols;
        auto* dst_row = dst.data() + row * cols;
        std::copy_n(src_row_ptr, cols, dst_row);
    }
}

float hamming_coherent_gain(size_t length)
{
    if (length <= 1) {
        return 1.0f;
    }
    return 0.54f - (0.46f / static_cast<float>(length));
}

float sensing_rd_amplitude_scale(size_t active_rows, size_t active_cols)
{
    if (active_rows == 0 || active_cols == 0) {
        return 1.0f;
    }
    const float periodogram_norm =
        std::sqrt(static_cast<float>(active_rows) * static_cast<float>(active_cols));
    const float processing_gain_norm =
        std::sqrt(static_cast<float>(active_rows) * static_cast<float>(active_cols));
    const float window_coherent_gain =
        hamming_coherent_gain(active_cols) * hamming_coherent_gain(active_rows);
    // Divide by sqrt(M*K): standard periodogram normalization so the noise power stays unchanged.
    // Divide by another sqrt(M*K): remove the coherent processing gain from integrating M symbols and K subcarriers.
    // Finally divide by G_r * G_d: compensate the coherent gain introduced by the range and Doppler windows.
    const float total_gain =
        periodogram_norm * processing_gain_norm * window_coherent_gain;
    if (!(total_gain > 0.0f) || !std::isfinite(total_gain)) {
        return 1.0f;
    }
    return 1.0f / total_gain;
}

void scale_complex_buffer_inplace(
    std::complex<float>* data,
    size_t complex_count,
    float scale)
{
    if (data == nullptr || complex_count == 0 || scale == 1.0f) {
        return;
    }

    auto* scalar = reinterpret_cast<float*>(data);
    const size_t scalar_count = complex_count * 2;
    #pragma omp simd simdlen(16)
    for (size_t i = 0; i < scalar_count; ++i) {
        scalar[i] *= scale;
    }
}

size_t doppler_zoom_head_bin_count(size_t output_rows)
{
    return (output_rows + 1) / 2;
}

size_t doppler_zoom_tail_bin_count(size_t output_rows)
{
    return output_rows / 2;
}

bool backend_range_zoom_active(
    const SensingChannel::BackendZoomContext& zoom,
    bool micro_doppler_enabled,
    size_t micro_doppler_range_bin)
{
    if (!zoom.has_range_plan()) {
        return false;
    }
    if (!micro_doppler_enabled) {
        return true;
    }
    return micro_doppler_range_bin < zoom.range_view_bins;
}

void execute_range_zoom_rows(
    ZoomFFTProcessor& plan,
    const std::complex<float>* src,
    size_t src_rows,
    size_t src_stride,
    size_t fft_rows,
    AlignedVector& dst)
{
    const size_t view_cols = plan.params().output_size;
    if (dst.size() != fft_rows * view_cols) {
        dst.assign(fft_rows * view_cols, std::complex<float>(0.0f, 0.0f));
    } else {
        std::fill(dst.begin(), dst.end(), std::complex<float>(0.0f, 0.0f));
    }

    const size_t rows_to_process = std::min(src_rows, fft_rows);
    for (size_t row = 0; row < rows_to_process; ++row) {
        plan.execute(
            src + row * src_stride,
            1,
            dst.data() + row * view_cols,
            1);
    }
}

void execute_doppler_zoom_columns(
    SensingChannel::BackendZoomContext& zoom,
    const std::complex<float>* src,
    size_t cols,
    AlignedVector& dst)
{
    const size_t output_rows = zoom.doppler_view_bins;
    if (dst.size() != output_rows * cols) {
        dst.assign(output_rows * cols, std::complex<float>(0.0f, 0.0f));
    } else {
        std::fill(dst.begin(), dst.end(), std::complex<float>(0.0f, 0.0f));
    }

    for (size_t col = 0; col < cols; ++col) {
        if (zoom.doppler_head_bins > 0 && zoom.doppler_head_plan) {
            zoom.doppler_head_plan->execute(
                src + col,
                cols,
                dst.data() + col,
                cols);
        }
        if (zoom.doppler_tail_bins > 0 && zoom.doppler_tail_plan) {
            zoom.doppler_tail_plan->execute(
                src + col,
                cols,
                dst.data() + zoom.doppler_head_bins * cols + col,
                cols);
        }
    }
}

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

SensingChannel::SensingComputeContext::SensingComputeContext(
    const Config& cfg,
    const SensingRxChannelConfig& c,
    uint32_t logical_id,
    std::shared_ptr<AggregatedSensingDataSender> aggregated_sender,
    const std::string& output_ip,
    int output_port)
  : demod_fft_in(cfg.fft_size),
    demod_fft_out(cfg.fft_size),
    sensing_core({cfg.fft_size, std::max(cfg.range_fft_size, cfg.fft_size), cfg.doppler_fft_size, sensing_symbol_capacity(cfg)}),
    sensing_sender(
        output_ip,
        output_port,
        c.enable_sensing_output,
        logical_id,
        std::move(aggregated_sender),
        cfg.sensing_on_wire_format),
    next_hb_time(std::chrono::steady_clock::now()) {
    accumulated_rx_symbols.reserve(cfg.sensing_symbol_num);
    accumulated_tx_symbols.reserve(cfg.sensing_symbol_num);
    if (sensing_output_mode_is_compact_mask(cfg)) {
        sensing_mask_layout = build_sensing_mask_layout(cfg);
        compact_mask_analysis = analyze_compact_sensing_mask(cfg);
        compact_mask_local_delay_doppler_supported = compact_mask_analysis.local_delay_doppler_supported;
        compact_channel_output.resize(sensing_mask_layout.total_re_count);
        compact_selected_tx_symbols.reserve(cfg.sensing_symbol_num);
        compact_shifted_subcarrier_indices.reserve(compact_mask_analysis.common_subcarrier_count);
        for (size_t sc_idx = 0; sc_idx < compact_mask_analysis.common_subcarrier_count; ++sc_idx) {
            compact_shifted_subcarrier_indices.push_back(
                static_cast<int>(raw_fft_bin_to_shifted_index(sc_idx, compact_mask_analysis.common_subcarrier_count)));
        }
        if (compact_mask_local_delay_doppler_supported) {
            compact_sensing_core = std::make_unique<SensingProcessor>(SensingProcessor::Params{
                compact_mask_analysis.common_subcarrier_count,
                cfg.range_fft_size,
                cfg.doppler_fft_size,
                cfg.sensing_symbol_num,
            });
        }
    }
    range_window.resize(cfg.fft_size);
    WindowGenerator::generate_hamming(range_window, cfg.fft_size);
    doppler_window.resize(cfg.sensing_symbol_num);
    WindowGenerator::generate_hamming(doppler_window, cfg.sensing_symbol_num);
    cfar_params.max_points = 256;
    backend_view_range_bins = resolved_sensing_view_range_bins(cfg);
    backend_view_doppler_bins = resolved_sensing_view_doppler_bins(cfg);
    const bool backend_processing = backend_sensing_processing_supported(cfg);
    backend_range_view_limited =
        backend_processing &&
        (backend_view_range_bins != sensing_core.params().range_fft_size);
    backend_doppler_view_limited =
        backend_processing &&
        (backend_view_doppler_bins != sensing_core.params().doppler_fft_size);
    backend_zoom.range_enabled = backend_range_view_limited;
    backend_zoom.doppler_enabled = backend_doppler_view_limited;
    backend_zoom.range_view_bins = backend_view_range_bins;
    backend_zoom.doppler_view_bins = backend_view_doppler_bins;
    backend_zoom.doppler_head_bins = doppler_zoom_head_bin_count(backend_view_doppler_bins);
    backend_zoom.doppler_tail_bins = doppler_zoom_tail_bin_count(backend_view_doppler_bins);
    if (backend_zoom.range_enabled) {
        backend_zoom.range_plan = std::make_unique<ZoomFFTProcessor>(ZoomFFTProcessor::Params{
            sensing_core.params().fft_size,
            sensing_core.params().range_fft_size,
            0,
            backend_view_range_bins,
            ZoomFFTProcessor::Direction::Backward,
        });
    }
    if (backend_zoom.doppler_enabled) {
        if (backend_zoom.doppler_head_bins > 0) {
            backend_zoom.doppler_head_plan = std::make_unique<ZoomFFTProcessor>(ZoomFFTProcessor::Params{
                sensing_core.params().doppler_fft_size,
                sensing_core.params().doppler_fft_size,
                0,
                backend_zoom.doppler_head_bins,
                ZoomFFTProcessor::Direction::Forward,
            });
        }
        if (backend_zoom.doppler_tail_bins > 0) {
            backend_zoom.doppler_tail_plan = std::make_unique<ZoomFFTProcessor>(ZoomFFTProcessor::Params{
                sensing_core.params().doppler_fft_size,
                sensing_core.params().doppler_fft_size,
                sensing_core.params().doppler_fft_size - backend_zoom.doppler_tail_bins,
                backend_zoom.doppler_tail_bins,
                ZoomFFTProcessor::Direction::Forward,
            });
        }
    }
    if (backend_range_view_limited) {
        backend_view_buffer.assign(
            backend_view_range_bins * sensing_core.params().doppler_fft_size,
            std::complex<float>(0.0f, 0.0f));
        const int backend_view_rows = static_cast<int>(sensing_core.params().doppler_fft_size);
        const int backend_view_cols = static_cast<int>(backend_view_range_bins);
        backend_view_doppler_fft_plan = fftwf_plan_many_dft(
            1,
            &backend_view_rows,
            backend_view_cols,
            reinterpret_cast<fftwf_complex*>(backend_view_buffer.data()),
            nullptr,
            backend_view_cols,
            1,
            reinterpret_cast<fftwf_complex*>(backend_view_buffer.data()),
            nullptr,
            backend_view_cols,
            1,
            FFTW_FORWARD,
            FFTW_MEASURE);
    }
    if (backend_doppler_view_limited) {
        backend_output_buffer.assign(
            backend_view_range_bins * backend_view_doppler_bins,
            std::complex<float>(0.0f, 0.0f));
    }
    if (compact_mask_local_delay_doppler_supported) {
        compact_range_window.resize(compact_mask_analysis.common_subcarrier_count);
        WindowGenerator::generate_hamming(
            compact_range_window,
            compact_mask_analysis.common_subcarrier_count);
        compact_doppler_window.resize(cfg.sensing_symbol_num);
        WindowGenerator::generate_hamming(compact_doppler_window, cfg.sensing_symbol_num);
        compact_backend_zoom.range_enabled = backend_processing &&
            (backend_view_range_bins != cfg.range_fft_size);
        compact_backend_zoom.doppler_enabled = backend_processing &&
            (backend_view_doppler_bins != cfg.doppler_fft_size);
        compact_backend_zoom.range_view_bins = backend_view_range_bins;
        compact_backend_zoom.doppler_view_bins = backend_view_doppler_bins;
        compact_backend_zoom.doppler_head_bins = doppler_zoom_head_bin_count(backend_view_doppler_bins);
        compact_backend_zoom.doppler_tail_bins = doppler_zoom_tail_bin_count(backend_view_doppler_bins);
        if (compact_backend_zoom.range_enabled) {
            compact_backend_zoom.range_plan = std::make_unique<ZoomFFTProcessor>(ZoomFFTProcessor::Params{
                compact_mask_analysis.common_subcarrier_count,
                cfg.range_fft_size,
                0,
                backend_view_range_bins,
                ZoomFFTProcessor::Direction::Backward,
            });
        }
        if (compact_backend_zoom.doppler_enabled) {
            if (compact_backend_zoom.doppler_head_bins > 0) {
                compact_backend_zoom.doppler_head_plan = std::make_unique<ZoomFFTProcessor>(ZoomFFTProcessor::Params{
                    cfg.doppler_fft_size,
                    cfg.doppler_fft_size,
                    0,
                    compact_backend_zoom.doppler_head_bins,
                    ZoomFFTProcessor::Direction::Forward,
                });
            }
            if (compact_backend_zoom.doppler_tail_bins > 0) {
                compact_backend_zoom.doppler_tail_plan = std::make_unique<ZoomFFTProcessor>(ZoomFFTProcessor::Params{
                    cfg.doppler_fft_size,
                    cfg.doppler_fft_size,
                    cfg.doppler_fft_size - compact_backend_zoom.doppler_tail_bins,
                    compact_backend_zoom.doppler_tail_bins,
                    ZoomFFTProcessor::Direction::Forward,
                });
            }
        }
    }
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
    if (backend_view_doppler_fft_plan != nullptr) {
        fftwf_destroy_plan(backend_view_doppler_fft_plan);
        backend_view_doppler_fft_plan = nullptr;
    }
    if (demod_fft_plan != nullptr) {
        fftwf_destroy_plan(demod_fft_plan);
        demod_fft_plan = nullptr;
    }
}

SensingChannel::SensingChannel(
    const Config& cfg,
    const SensingRxChannelConfig& channel_cfg,
    const std::string& output_ip,
    int output_port,
    uint32_t logical_id,
    std::atomic<bool>& running_ref,
    std::shared_ptr<AggregatedSensingDataSender> aggregated_sender,
    std::shared_ptr<std::atomic<uint64_t>> batch_reset_symbol,
    BatchResetRequester batch_reset_requester,
    HeartbeatCallback heartbeat_sender,
    CoreResolver core_resolver
)
  : _cfg(cfg),
    _running_ref(running_ref),
    _output_ip(output_ip),
    _output_port(output_port),
    _shared_batch_reset_symbol(std::move(batch_reset_symbol)),
    _batch_reset_requester(std::move(batch_reset_requester)),
    _heartbeat_sender(std::move(heartbeat_sender)),
    _core_resolver(std::move(core_resolver)),
    _rx_io(cfg, channel_cfg, logical_id),
    _compute(cfg, channel_cfg, logical_id, std::move(aggregated_sender), _output_ip, _output_port) {
    _rx_io.paired_queue->set_rx_recycler([this](AlignedVector&& frame) {
        this->_rx_io.rx_frame_pool.release(std::move(frame));
    });

    if (sensing_output_mode_is_compact_mask(_cfg)) {
        if (_compute.compact_mask_local_delay_doppler_supported) {
            LOG_G_INFO() << "[Sensing CH " << _rx_io.logical_id
                         << "] compact_mask regular sampling enables MTI/local Delay-Doppler"
                         << " (symbols=" << _compute.compact_mask_analysis.selected_symbol_count
                         << ", stride=" << _compute.compact_mask_analysis.implicit_symbol_stride
                         << ", subcarriers=" << _compute.compact_mask_analysis.common_subcarrier_count
                         << ")";
        } else {
            const std::string reason = _compute.compact_mask_analysis.effective_reason();
            LOG_G_INFO() << "[Sensing CH " << _rx_io.logical_id
                         << "] compact_mask stays in raw compact-only mode: "
                         << (reason.empty() ? "mask is not regular-sampling compatible" : reason);
        }
    }

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
    _apply_batch_reset_if_due(frame_start_symbol_index);
    const auto now = std::chrono::steady_clock::now();
    _send_heartbeat_if_due(now);
    if (_compute.sensing_pipeline_disabled_by_mode) {
        return;
    }

    const size_t symbols_in_frame = std::min(frame.rx_symbols.size(), frame.tx_symbols.size());
    if (symbols_in_frame == 0) {
        return;
    }
    if (sensing_output_mode_is_compact_mask(_cfg)) {
        _apply_shared_sensing_if_due(frame_start_symbol_index);
        if (_compute.compact_mask_local_delay_doppler_supported) {
            _process_regular_compact_bistatic_frame(frame, frame_start_symbol_index);
        } else {
            _process_compact_bistatic_frame(frame, frame_start_symbol_index);
        }
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
    _request_shared_batch_reset();
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

void SensingChannel::_request_shared_batch_reset() {
    if (_batch_reset_requester) {
        _batch_reset_requester();
    }
}

void SensingChannel::_apply_batch_reset_if_due(uint64_t frame_start_symbol_index) {
    if (!_shared_batch_reset_symbol) {
        return;
    }
    const uint64_t reset_symbol = _shared_batch_reset_symbol->load(std::memory_order_relaxed);
    if (reset_symbol == 0 || reset_symbol == _applied_batch_reset_symbol) {
        return;
    }
    if (frame_start_symbol_index < reset_symbol) {
        return;
    }

    _compute.accumulated_rx_symbols.clear();
    _compute.accumulated_tx_symbols.clear();
    _compute.compact_selected_tx_symbols.clear();
    _compute.pending_batch_gather_us = 0.0;
    _compute.batch_has_first_symbol = false;
    _compute.current_batch_first_symbol = frame_start_symbol_index;
    _compute.next_symbol_to_sample = frame_start_symbol_index;
    if (backend_sensing_processing_supported(_cfg)) {
        _compute.micro_doppler_state.clear();
    }
    _compute.sensing_core.clear_channel_buffer();
    if (_compute.compact_sensing_core) {
        _compute.compact_sensing_core->clear_channel_buffer();
    }

    _applied_batch_reset_symbol = reset_symbol;
    LOG_G_WARN() << "[Sensing CH " << _rx_io.logical_id
                 << "] applied shared batch reset at frame_start_symbol="
                 << frame_start_symbol_index;
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
        const double actual_rx_rate = io.rx_usrp->get_rx_rate(io.channel_cfg.usrp_channel);
        io.rx_sample_rate = (actual_rx_rate > 0.0) ? actual_rx_rate : cfg.sample_rate;
        const double actual_tick_rate = io.rx_usrp->get_master_clock_rate();
        io.rx_tick_rate = (actual_tick_rate > 0.0) ? actual_tick_rate : io.rx_sample_rate;
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

std::optional<size_t> SensingChannel::_resolve_core(size_t hint) const {
    if (_core_resolver) return _core_resolver(hint);
    return std::nullopt;
}

void SensingChannel::_rx_loop(const uhd::time_spec_t& start_time) {
    async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
    uhd::set_thread_priority_safe(1.0f, true);
    bind_current_thread_to_core(_resolve_core(3 + static_cast<size_t>(_rx_io.logical_id) * 2));
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
            _cfg,
            _rx_io.rx_sample_rate > 0.0 ? _rx_io.rx_sample_rate : _cfg.sample_rate,
            _rx_io.rx_tick_rate > 0.0 ? _rx_io.rx_tick_rate :
                (_rx_io.rx_sample_rate > 0.0 ? _rx_io.rx_sample_rate : _cfg.sample_rate))
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
            _cfg,
            _rx_io.rx_sample_rate > 0.0 ? _rx_io.rx_sample_rate : _cfg.sample_rate,
            _rx_io.rx_tick_rate > 0.0 ? _rx_io.rx_tick_rate :
                (_rx_io.rx_sample_rate > 0.0 ? _rx_io.rx_sample_rate : _cfg.sample_rate));
        const int32_t correction = normalize_alignment_samples(
            static_cast<int64_t>(desired_alignment) - static_cast<int64_t>(frame_offset),
            frame_samples);
        if (correction != 0) {
            LOG_RT_WARN_HZ(5) << "[Sensing CH " << _rx_io.logical_id
                              << "] RX frame boundary mismatch: offset=" << frame_offset
                              << ", target_ALGN=" << desired_alignment
                              << ", last_ALGN=" << current_alignment
                              << ", correction=" << correction;
            // Keep the current frame paired; apply the corrected ALGN starting from the next frame.
            set_alignment(correction);
        }
    }

    const uint64_t frame_seq = have_first_time
        ? compute_rx_frame_seq_from_time(
            first_time_spec,
            _rx_io.stream_start_time,
            0,
            _rx_io.target_alignment.load(std::memory_order_relaxed),
            _cfg,
            _rx_io.rx_sample_rate > 0.0 ? _rx_io.rx_sample_rate : _cfg.sample_rate,
            _rx_io.rx_tick_rate > 0.0 ? _rx_io.rx_tick_rate :
                (_rx_io.rx_sample_rate > 0.0 ? _rx_io.rx_sample_rate : _cfg.sample_rate))
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
    bind_current_thread_to_core(_resolve_core(4 + static_cast<size_t>(_rx_io.logical_id) * 2));
    prefault_thread_stack();

    while (_running_ref.load(std::memory_order_relaxed)) {
        AlignedVector rx_frame_data;
        TxSymbolsFrame tx_frame;
        if (!_rx_io.paired_queue->pop_pair(rx_frame_data, tx_frame)) {
            break;
        }

        _apply_batch_reset_if_due(tx_frame.frame_start_symbol_index);

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
        if (sensing_output_mode_is_compact_mask(_cfg)) {
            _apply_shared_sensing_if_due(tx_frame.frame_start_symbol_index);
            if (_compute.compact_mask_local_delay_doppler_supported) {
                _process_regular_compact_monostatic_frame(rx_frame_data, tx_frame);
            } else {
                _process_compact_monostatic_frame(rx_frame_data, tx_frame);
            }
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
        _heartbeat_sender(_output_ip, _output_port);
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

    const double rx_sample_rate =
        (_rx_io.rx_sample_rate > 0.0) ? _rx_io.rx_sample_rate : _cfg.sample_rate;
    const double delay_us = (rx_sample_rate > 0.0)
        ? (static_cast<double>(delay_samples) * 1e6 / rx_sample_rate)
        : 0.0;
    const float corr_ratio = (avg_corr > 0.0f) ? (max_corr / avg_corr) : 0.0f;
    const int32_t target_alignment = _rx_io.target_alignment.load(std::memory_order_relaxed);
    const int64_t suggested_alignment_raw =
        static_cast<int64_t>(target_alignment) + delay_samples;

    _compute.next_delay_estimation_frame_seq = frame_seq + kSystemDelayEstimationFrameInterval;

    LOG_RT_INFO() << "[SYSDLY CH " << _rx_io.logical_id
                  << "] frame_seq=" << frame_seq
                  << ", delay=" << delay_samples << " samp (" << delay_us << " us)"
                  << ", peak=" << max_pos
                  << ", expected=" << _compute.system_delay_expected_sync_pos
                  << ", corr_ratio=" << corr_ratio
                  << ", target_alignment=" << target_alignment
                  << ", alignment_suggest=" << suggested_alignment_raw
                  << ", next_frame_seq=" << _compute.next_delay_estimation_frame_seq
                  << ", interval=" << kSystemDelayEstimationFrameInterval;
}

void SensingChannel::_process_compact_monostatic_frame(
    const AlignedVector& rx_frame_data,
    const TxSymbolsFrame& tx_frame)
{
    const auto& layout = _compute.sensing_mask_layout;
    if (layout.empty() || tx_frame.symbols == nullptr) {
        return;
    }

    const auto& tx_symbols = *tx_frame.symbols;
    auto& compact_output = _compute.compact_channel_output;
    for (size_t row = 0; row < layout.selected_symbols.size(); ++row) {
        const size_t symbol_idx = static_cast<size_t>(layout.selected_symbols[row]);
        if (symbol_idx >= tx_symbols.size()) {
            continue;
        }
        const size_t symbol_start = symbol_idx * (_cfg.fft_size + _cfg.cp_length) + _cfg.cp_length;
        std::copy(
            rx_frame_data.begin() + symbol_start,
            rx_frame_data.begin() + symbol_start + _cfg.fft_size,
            _compute.demod_fft_in.begin());
        fftwf_execute(_compute.demod_fft_plan);

        const auto& tx_symbol = tx_symbols[symbol_idx];
        const size_t begin = layout.selected_symbol_offsets[row];
        const size_t end = layout.selected_symbol_offsets[row + 1];
        for (size_t idx = begin; idx < end; ++idx) {
            const size_t sc = static_cast<size_t>(layout.flat_subcarrier_indices[idx]);
            compact_output[idx] = _compute.demod_fft_out[sc] * std::conj(tx_symbol[sc]);
        }
    }

    _compute.sensing_sender.push_compact_data(
        compact_output,
        layout.mask_hash,
        tx_frame.frame_start_symbol_index);
}

void SensingChannel::_process_compact_bistatic_frame(
    const SensingFrame& frame,
    uint64_t frame_start_symbol_index)
{
    const auto& layout = _compute.sensing_mask_layout;
    if (layout.empty()) {
        return;
    }

    auto& compact_output = _compute.compact_channel_output;
    const size_t symbols_in_frame = std::min(frame.rx_symbols.size(), frame.tx_symbols.size());
    for (size_t row = 0; row < layout.selected_symbols.size(); ++row) {
        const size_t symbol_idx = static_cast<size_t>(layout.selected_symbols[row]);
        if (symbol_idx >= symbols_in_frame) {
            continue;
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

        const auto& tx_symbol = frame.tx_symbols[symbol_idx];
        const size_t begin = layout.selected_symbol_offsets[row];
        const size_t end = layout.selected_symbol_offsets[row + 1];
        for (size_t idx = begin; idx < end; ++idx) {
            const size_t sc = static_cast<size_t>(layout.flat_subcarrier_indices[idx]);
            compact_output[idx] = rx_symbol[sc] * std::conj(tx_symbol[sc]);
        }
    }

    _compute.sensing_sender.push_compact_data(
        compact_output,
        layout.mask_hash,
        frame_start_symbol_index);
}

void SensingChannel::_process_regular_compact_monostatic_frame(
    const AlignedVector& rx_frame_data,
    const TxSymbolsFrame& tx_frame)
{
    const auto& analysis = _compute.compact_mask_analysis;
    if (!analysis.local_delay_doppler_supported || tx_frame.symbols == nullptr) {
        _process_compact_monostatic_frame(rx_frame_data, tx_frame);
        return;
    }

    const auto& tx_symbols = *tx_frame.symbols;
    if (analysis.selected_symbols.empty()) {
        return;
    }

    if (!_compute.compact_sensing_core) {
        return;
    }

    auto& compact_core = *_compute.compact_sensing_core;
    auto& channel_buf = compact_core.channel_buffer();
    const size_t range_stride = compact_core.params().range_fft_size;

    if (_compute.compact_selected_tx_symbols.empty()) {
        compact_core.clear_channel_buffer();
    }

    for (size_t selected_row = 0; selected_row < analysis.selected_symbols.size(); ++selected_row) {
        const size_t symbol_idx = static_cast<size_t>(analysis.selected_symbols[selected_row]);
        if (symbol_idx >= tx_symbols.size()) {
            return;
        }
        if (_compute.compact_selected_tx_symbols.size() >= _cfg.sensing_symbol_num) {
            _process_regular_compact_buffer(
                _compute.compact_selected_tx_symbols,
                _compute.current_batch_first_symbol);
            _compute.compact_selected_tx_symbols.clear();
            _compute.batch_has_first_symbol = false;
            compact_core.clear_channel_buffer();
        }

        const size_t batch_row = _compute.compact_selected_tx_symbols.size();
        if (!_compute.batch_has_first_symbol) {
            _compute.current_batch_first_symbol =
                tx_frame.frame_start_symbol_index + static_cast<uint64_t>(symbol_idx);
            _compute.batch_has_first_symbol = true;
        }

        const size_t symbol_start = symbol_idx * (_cfg.fft_size + _cfg.cp_length) + _cfg.cp_length;
        if (symbol_start + _cfg.fft_size > rx_frame_data.size()) {
            return;
        }
        std::copy(
            rx_frame_data.begin() + static_cast<std::ptrdiff_t>(symbol_start),
            rx_frame_data.begin() + static_cast<std::ptrdiff_t>(symbol_start + _cfg.fft_size),
            _compute.demod_fft_in.begin());
        fftwf_execute(_compute.demod_fft_plan);

        auto* row_out = channel_buf.data() + batch_row * range_stride;
        AlignedVector compact_tx_symbol(analysis.common_subcarrier_count);
        for (size_t sub_idx = 0; sub_idx < analysis.common_subcarrier_count; ++sub_idx) {
            const size_t sc_idx = static_cast<size_t>(analysis.common_subcarrier_indices[sub_idx]);
            row_out[sub_idx] = _compute.demod_fft_out[sc_idx];
            compact_tx_symbol[sub_idx] = tx_symbols[symbol_idx][sc_idx];
        }
        _compute.compact_selected_tx_symbols.push_back(std::move(compact_tx_symbol));
        if (_compute.compact_selected_tx_symbols.size() >= _cfg.sensing_symbol_num) {
            _process_regular_compact_buffer(
                _compute.compact_selected_tx_symbols,
                _compute.current_batch_first_symbol);
            _compute.compact_selected_tx_symbols.clear();
            _compute.batch_has_first_symbol = false;
            compact_core.clear_channel_buffer();
        }
    }
}

void SensingChannel::_process_regular_compact_bistatic_frame(
    const SensingFrame& frame,
    uint64_t frame_start_symbol_index)
{
    const auto& analysis = _compute.compact_mask_analysis;
    if (!analysis.local_delay_doppler_supported || analysis.selected_symbols.empty()) {
        _process_compact_bistatic_frame(frame, frame_start_symbol_index);
        return;
    }

    const size_t symbols_in_frame = std::min(frame.rx_symbols.size(), frame.tx_symbols.size());
    if (symbols_in_frame == 0) {
        return;
    }

    if (!_compute.compact_sensing_core) {
        return;
    }

    auto& compact_core = *_compute.compact_sensing_core;
    auto& channel_buf = compact_core.channel_buffer();
    const size_t range_stride = compact_core.params().range_fft_size;

    if (_compute.compact_selected_tx_symbols.empty()) {
        compact_core.clear_channel_buffer();
    }

    for (size_t selected_row = 0; selected_row < analysis.selected_symbols.size(); ++selected_row) {
        const size_t symbol_idx = static_cast<size_t>(analysis.selected_symbols[selected_row]);
        if (symbol_idx >= symbols_in_frame) {
            return;
        }
        if (_compute.compact_selected_tx_symbols.size() >= _cfg.sensing_symbol_num) {
            _process_regular_compact_buffer(
                _compute.compact_selected_tx_symbols,
                _compute.current_batch_first_symbol);
            _compute.compact_selected_tx_symbols.clear();
            _compute.batch_has_first_symbol = false;
            compact_core.clear_channel_buffer();
        }

        const size_t batch_row = _compute.compact_selected_tx_symbols.size();
        if (!_compute.batch_has_first_symbol) {
            _compute.current_batch_first_symbol =
                frame_start_symbol_index + static_cast<uint64_t>(symbol_idx);
            _compute.batch_has_first_symbol = true;
        }

        const int relative_symbol_index = static_cast<int>(symbol_idx) - static_cast<int>(_cfg.sync_pos);
        const float phase_diff_cfo = frame.CFO * static_cast<float>(relative_symbol_index);
        auto* row_out = channel_buf.data() + batch_row * range_stride;
        AlignedVector compact_tx_symbol(analysis.common_subcarrier_count);
        for (size_t sub_idx = 0; sub_idx < analysis.common_subcarrier_count; ++sub_idx) {
            const size_t sc_idx = static_cast<size_t>(analysis.common_subcarrier_indices[sub_idx]);
            const float phase_diff_sfo =
                frame.SFO * static_cast<float>(_compute.actual_subcarrier_indices[sc_idx]) *
                static_cast<float>(relative_symbol_index);
            const float phase_diff_delay =
                _compute.subcarrier_phases_unit_delay[sc_idx] * frame.delay_offset;
            const float phase_diff_total = phase_diff_delay + phase_diff_sfo + phase_diff_cfo;
            const std::complex<float> phase = std::polar(1.0f, -phase_diff_total);
            row_out[sub_idx] = frame.rx_symbols[symbol_idx][sc_idx] * phase;
            compact_tx_symbol[sub_idx] = frame.tx_symbols[symbol_idx][sc_idx];
        }
        _compute.compact_selected_tx_symbols.push_back(std::move(compact_tx_symbol));
        if (_compute.compact_selected_tx_symbols.size() >= _cfg.sensing_symbol_num) {
            _process_regular_compact_buffer(
                _compute.compact_selected_tx_symbols,
                _compute.current_batch_first_symbol);
            _compute.compact_selected_tx_symbols.clear();
            _compute.batch_has_first_symbol = false;
            compact_core.clear_channel_buffer();
        }
    }
}

void SensingChannel::_process_regular_compact_buffer(
    const std::vector<AlignedVector>& tx_symbols,
    uint64_t frame_start_symbol_index)
{
    const auto& analysis = _compute.compact_mask_analysis;
    if (!analysis.local_delay_doppler_supported || tx_symbols.empty()) {
        return;
    }

    if (!_compute.compact_sensing_core) {
        return;
    }

    auto& compact_core = *_compute.compact_sensing_core;
    auto& channel_buf = compact_core.channel_buffer();
    const size_t symbol_count = tx_symbols.size();
    const size_t range_stride = compact_core.params().range_fft_size;
    const bool backend_processing = backend_sensing_processing_supported(_cfg);

    compact_core.channel_estimate_with_shift(tx_symbols, symbol_count);
    compact_core.apply_mti(_compute.active_enable_mti, symbol_count);

    if (_compute.active_skip_sensing_fft) {
        AlignedVector compact_output(symbol_count * analysis.common_subcarrier_count);
        for (size_t row = 0; row < symbol_count; ++row) {
            for (size_t sub_idx = 0; sub_idx < analysis.common_subcarrier_count; ++sub_idx) {
                const size_t compact_idx = row * analysis.common_subcarrier_count + sub_idx;
                const size_t shifted_sc = static_cast<size_t>(_compute.compact_shifted_subcarrier_indices[sub_idx]);
                compact_output[compact_idx] = channel_buf[row * range_stride + shifted_sc];
            }
        }
        _compute.sensing_sender.push_compact_data(
            compact_output,
            _compute.sensing_mask_layout.mask_hash,
            frame_start_symbol_index);
        return;
    }

    compact_core.apply_windows(
        channel_buf,
        _compute.compact_range_window,
        _compute.compact_doppler_window,
        symbol_count);
    const bool range_zoom_enabled = backend_processing &&
        backend_range_zoom_active(
            _compute.compact_backend_zoom,
            _compute.micro_doppler_enabled,
            _compute.micro_doppler_range_bin);
    const bool doppler_zoom_enabled = backend_processing &&
        _compute.compact_backend_zoom.has_doppler_plan();
    const size_t fft_rows = compact_core.params().doppler_fft_size;
    const std::complex<float>* doppler_input = nullptr;
    size_t doppler_input_cols = range_stride;
    AlignedVector* post_doppler_buf = &channel_buf;

    if (range_zoom_enabled) {
        execute_range_zoom_rows(
            *_compute.compact_backend_zoom.range_plan,
            channel_buf.data(),
            symbol_count,
            range_stride,
            fft_rows,
            _compute.compact_backend_zoom.range_output_buffer);
        doppler_input = _compute.compact_backend_zoom.range_output_buffer.data();
        doppler_input_cols = _compute.compact_backend_zoom.range_view_bins;
    } else {
        compact_core.execute_range_ifft();
        doppler_input = channel_buf.data();
        if (backend_processing && _compute.backend_range_view_limited) {
            copy_backend_range_view_input(
                channel_buf,
                symbol_count,
                range_stride,
                fft_rows,
                _compute.backend_view_range_bins,
                _compute.backend_view_buffer);
            doppler_input = _compute.backend_view_buffer.data();
            doppler_input_cols = _compute.backend_view_range_bins;
        }
    }

    if (backend_processing &&
        _compute.micro_doppler_enabled &&
        doppler_input_cols > 0) {
        const size_t selected_range_bin = std::min(
            _compute.micro_doppler_range_bin,
            doppler_input_cols - 1);
        extract_range_bin_trace(
            doppler_input,
            symbol_count,
            doppler_input_cols,
            selected_range_bin,
            _compute.micro_doppler_trace);
        _compute.micro_doppler_state.append_samples(
            _compute.micro_doppler_trace.data(),
            _compute.micro_doppler_trace.size());
    }

    if (doppler_zoom_enabled) {
        execute_doppler_zoom_columns(
            _compute.compact_backend_zoom,
            doppler_input,
            doppler_input_cols,
            _compute.backend_output_buffer);
        post_doppler_buf = &_compute.backend_output_buffer;
    } else if (backend_processing && _compute.backend_range_view_limited) {
        if (doppler_input == _compute.backend_view_buffer.data()) {
            fftwf_execute(_compute.backend_view_doppler_fft_plan);
            post_doppler_buf = &_compute.backend_view_buffer;
        } else {
            fftwf_execute_dft(
                _compute.backend_view_doppler_fft_plan,
                reinterpret_cast<fftwf_complex*>(const_cast<std::complex<float>*>(doppler_input)),
                reinterpret_cast<fftwf_complex*>(const_cast<std::complex<float>*>(doppler_input)));
            post_doppler_buf = &_compute.compact_backend_zoom.range_output_buffer;
        }
    } else {
        compact_core.execute_doppler_fft();
        post_doppler_buf = &channel_buf;
    }

    if (backend_processing) {
        AlignedVector* output_buf = nullptr;
        const size_t output_cols = doppler_input_cols;
        if (doppler_zoom_enabled) {
            output_buf = &_compute.backend_output_buffer;
        } else if (_compute.backend_doppler_view_limited) {
            const auto* fft_output = post_doppler_buf->data();
            crop_doppler_view_output(
                fft_output,
                fft_rows,
                output_cols,
                _compute.backend_view_doppler_bins,
                _compute.backend_output_buffer);
            output_buf = &_compute.backend_output_buffer;
        } else {
            output_buf = post_doppler_buf;
        }
        const size_t output_rows = (doppler_zoom_enabled || _compute.backend_doppler_view_limited)
            ? _compute.backend_view_doppler_bins
            : fft_rows;
        const float amplitude_scale = sensing_rd_amplitude_scale(
            symbol_count,
            compact_core.params().fft_size);
        scale_complex_buffer_inplace(output_buf->data(), output_buf->size(), amplitude_scale);
        _compute.metadata_bytes = _build_backend_metadata(
            output_buf->data(),
            output_rows,
            output_cols,
            frame_start_symbol_index);
        _compute.sensing_sender.push_data(
            *output_buf,
            frame_start_symbol_index,
            std::move(_compute.metadata_bytes));
    } else {
        _compute.sensing_sender.push_data(channel_buf, frame_start_symbol_index);
    }
    compact_core.clear_channel_buffer();
}

void SensingChannel::_sensing_process(const SensingFrame& frame, uint64_t first_symbol_index, double gather_us) {
    using ProfileClock = std::chrono::steady_clock;
    auto& channel_buf = _compute.sensing_core.channel_buffer();
    const auto& sensing_params = _compute.sensing_core.params();
    const size_t range_stride = sensing_params.range_fft_size;
    const size_t doppler_slots = sensing_params.doppler_fft_size;

    const size_t symbol_count = std::min(frame.rx_symbols.size(), _cfg.sensing_symbol_num);
    if (symbol_count < sensing_params.sensing_symbol_num) {
        _compute.sensing_core.clear_channel_buffer();
    }
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
    _sensing_process_finalize(frame.tx_symbols, first_symbol_index, gather_us, prep_us, symbol_count);
}

void SensingChannel::_sensing_process_freq(const SensingFrame& frame, uint64_t first_symbol_index, double gather_us) {
    using ProfileClock = std::chrono::steady_clock;
    const size_t symbol_count = std::min(frame.rx_symbols.size(), _cfg.sensing_symbol_num);
    if (symbol_count < _compute.sensing_core.params().sensing_symbol_num) {
        _compute.sensing_core.clear_channel_buffer();
    }
    const auto prep_start = ProfileClock::now();
    _compute.sensing_core.copy_symbols_to_buffer(frame.rx_symbols, symbol_count);
    const double prep_us = std::chrono::duration<double, std::micro>(
        ProfileClock::now() - prep_start).count();
    _sensing_process_finalize(frame.tx_symbols, first_symbol_index, gather_us, prep_us, symbol_count);
}

void SensingChannel::_sensing_process_finalize(
    const std::vector<AlignedVector>& tx_symbols,
    uint64_t first_symbol_index,
    double gather_us,
    double prep_us,
    size_t symbol_count
) {
    using ProfileClock = std::chrono::steady_clock;
    const bool do_prof = should_profile_sensing(_cfg);
    const bool backend_processing = backend_sensing_processing_supported(_cfg);
    auto& channel_buf = _compute.sensing_core.channel_buffer();
    const auto chest_start = ProfileClock::now();
    _compute.sensing_core.channel_estimate_with_shift(tx_symbols, symbol_count);
    const double chest_shift_us = std::chrono::duration<double, std::micro>(
        ProfileClock::now() - chest_start).count();
    const auto mti_start = ProfileClock::now();
    _compute.sensing_core.apply_mti(_compute.active_enable_mti, symbol_count);
    const double mti_us = std::chrono::duration<double, std::micro>(
        ProfileClock::now() - mti_start).count();

    double windows_fft_us = 0.0;
    const size_t fft_rows = _compute.sensing_core.params().doppler_fft_size;
    const bool range_zoom_enabled = backend_processing &&
        backend_range_zoom_active(
            _compute.backend_zoom,
            _compute.micro_doppler_enabled,
            _compute.micro_doppler_range_bin);
    const bool doppler_zoom_enabled = backend_processing &&
        _compute.backend_zoom.has_doppler_plan();
    const std::complex<float>* doppler_input = channel_buf.data();
    size_t doppler_input_cols = _compute.sensing_core.params().range_fft_size;
    AlignedVector* post_doppler_buf = &channel_buf;
    if (!_compute.active_skip_sensing_fft) {
        const auto fft_start = ProfileClock::now();
        _compute.sensing_core.apply_windows(
            channel_buf,
            _compute.range_window,
            _compute.doppler_window,
            symbol_count);
        if (range_zoom_enabled) {
            execute_range_zoom_rows(
                *_compute.backend_zoom.range_plan,
                channel_buf.data(),
                symbol_count,
                _compute.sensing_core.params().range_fft_size,
                fft_rows,
                _compute.backend_zoom.range_output_buffer);
            doppler_input = _compute.backend_zoom.range_output_buffer.data();
            doppler_input_cols = _compute.backend_zoom.range_view_bins;
        } else {
            _compute.sensing_core.execute_range_ifft();
            doppler_input = channel_buf.data();
            doppler_input_cols = _compute.sensing_core.params().range_fft_size;
            if (backend_processing && _compute.backend_range_view_limited) {
                copy_backend_range_view_input(
                    channel_buf,
                    symbol_count,
                    _compute.sensing_core.params().range_fft_size,
                    fft_rows,
                    _compute.backend_view_range_bins,
                    _compute.backend_view_buffer);
                doppler_input = _compute.backend_view_buffer.data();
                doppler_input_cols = _compute.backend_view_range_bins;
            }
        }

        if (backend_processing &&
            _compute.micro_doppler_enabled &&
            doppler_input_cols > 0) {
            const size_t selected_range_bin = std::min(
                _compute.micro_doppler_range_bin,
                doppler_input_cols - 1);
            extract_range_bin_trace(
                doppler_input,
                symbol_count,
                doppler_input_cols,
                selected_range_bin,
                _compute.micro_doppler_trace);
            _compute.micro_doppler_state.append_samples(
                _compute.micro_doppler_trace.data(),
                _compute.micro_doppler_trace.size());
        }

        if (doppler_zoom_enabled) {
            execute_doppler_zoom_columns(
                _compute.backend_zoom,
                doppler_input,
                doppler_input_cols,
                _compute.backend_output_buffer);
            post_doppler_buf = &_compute.backend_output_buffer;
        } else if (backend_processing && _compute.backend_range_view_limited) {
            if (doppler_input == _compute.backend_view_buffer.data()) {
                fftwf_execute(_compute.backend_view_doppler_fft_plan);
                post_doppler_buf = &_compute.backend_view_buffer;
            } else {
                fftwf_execute_dft(
                    _compute.backend_view_doppler_fft_plan,
                    reinterpret_cast<fftwf_complex*>(const_cast<std::complex<float>*>(doppler_input)),
                    reinterpret_cast<fftwf_complex*>(const_cast<std::complex<float>*>(doppler_input)));
                post_doppler_buf = &_compute.backend_zoom.range_output_buffer;
            }
        } else {
            _compute.sensing_core.execute_doppler_fft();
            post_doppler_buf = &channel_buf;
        }
        windows_fft_us = std::chrono::duration<double, std::micro>(
            ProfileClock::now() - fft_start).count();
    }

    const auto send_start = ProfileClock::now();
    if (backend_processing) {
        AlignedVector* output_buf = nullptr;
        const size_t output_cols = doppler_input_cols;
        if (doppler_zoom_enabled) {
            output_buf = &_compute.backend_output_buffer;
        } else if (_compute.backend_doppler_view_limited) {
            const auto* fft_output = post_doppler_buf->data();
            crop_doppler_view_output(
                fft_output,
                fft_rows,
                output_cols,
                _compute.backend_view_doppler_bins,
                _compute.backend_output_buffer);
            output_buf = &_compute.backend_output_buffer;
        } else {
            output_buf = post_doppler_buf;
        }
        const size_t output_rows = (doppler_zoom_enabled || _compute.backend_doppler_view_limited)
            ? _compute.backend_view_doppler_bins
            : fft_rows;
        const float amplitude_scale = sensing_rd_amplitude_scale(
            symbol_count,
            _compute.sensing_core.params().fft_size);
        scale_complex_buffer_inplace(output_buf->data(), output_buf->size(), amplitude_scale);
        _compute.metadata_bytes = _build_backend_metadata(
            output_buf->data(),
            output_rows,
            output_cols,
            first_symbol_index);
        _compute.sensing_sender.push_data(
            *output_buf,
            first_symbol_index,
            std::move(_compute.metadata_bytes));
    } else {
        _compute.sensing_sender.push_data(channel_buf, first_symbol_index);
    }
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

void SensingChannel::_prepare_backend_processing_state(
    SharedSensingRuntime& snapshot,
    bool& clear_micro_doppler)
{
    clear_micro_doppler = false;
    _compute.cfar_params.enabled = snapshot.cfar_enabled;
    _compute.cfar_params.train_doppler = snapshot.cfar_train_doppler;
    _compute.cfar_params.train_range = snapshot.cfar_train_range;
    _compute.cfar_params.guard_doppler = snapshot.cfar_guard_doppler;
    _compute.cfar_params.guard_range = snapshot.cfar_guard_range;
    _compute.cfar_params.alpha_db = snapshot.cfar_alpha_db;
    _compute.cfar_params.min_range_bin = snapshot.cfar_min_range_bin;
    _compute.cfar_params.dc_exclusion_bins = snapshot.cfar_dc_exclusion_bins;
    _compute.cfar_params.min_power_db = snapshot.cfar_min_power_db;
    _compute.cfar_params.os_rank_percent = snapshot.cfar_os_rank_percent;
    _compute.cfar_params.os_suppress_doppler = snapshot.cfar_os_suppress_doppler;
    _compute.cfar_params.os_suppress_range = snapshot.cfar_os_suppress_range;

    if (_compute.micro_doppler_enabled != snapshot.micro_doppler_enabled) {
        _compute.micro_doppler_enabled = snapshot.micro_doppler_enabled;
        clear_micro_doppler = true;
    }
    const size_t next_range_bin = static_cast<size_t>(std::max(0, snapshot.micro_doppler_range_bin));
    if (_compute.micro_doppler_range_bin != next_range_bin) {
        _compute.micro_doppler_range_bin = next_range_bin;
        clear_micro_doppler = true;
    }
}

std::vector<uint8_t> SensingChannel::_build_backend_metadata(
    const std::complex<float>* rd_data,
    size_t rows,
    size_t cols,
    uint64_t frame_start_symbol_index)
{
    if (!backend_sensing_processing_supported(_cfg) ||
        rd_data == nullptr ||
        rows == 0 ||
        cols == 0) {
        return {};
    }

    SensingMetadata metadata{};
    metadata.cfar_enabled = _compute.cfar_params.enabled;
    metadata.micro_doppler_enabled = _compute.micro_doppler_enabled;
    compute_shifted_magnitude_db(rd_data, rows, cols, _compute.rd_magnitude_db);
    run_os_cfar_2d_full(
        _compute.rd_magnitude_db,
        rows,
        cols,
        _compute.cfar_params,
        metadata.cfar_points,
        metadata.cfar_hits,
        metadata.cfar_shown_hits,
        metadata.cfar_stats);
    if (!metadata.cfar_points.empty()) {
        metadata.target_clusters = cluster_detected_targets(
            metadata.cfar_points,
            _compute.rd_magnitude_db,
            rows,
            cols,
            std::max(0, _compute.cfar_params.os_suppress_doppler) + 1,
            std::max(0, _compute.cfar_params.os_suppress_range) + 1);
    }

    if (_compute.micro_doppler_enabled) {
        _compute.micro_doppler_state.compute_spectrum(
            metadata.micro_doppler_spectrum,
            metadata.micro_doppler_rows,
            metadata.micro_doppler_cols,
            metadata.micro_doppler_extent);
    }
    return serialize_sensing_metadata(metadata, frame_start_symbol_index);
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
    _compute.active_skip_sensing_fft = backend_sensing_processing_supported(_cfg)
        ? false
        : snapshot.skip_sensing_fft;
    bool clear_micro_doppler = false;
    _prepare_backend_processing_state(snapshot, clear_micro_doppler);
    if (clear_micro_doppler) {
        _compute.micro_doppler_state.clear();
    }
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
                  << ", CFAR=" << (_compute.cfar_params.enabled ? 1 : 0)
                  << ", OS%=" << _compute.cfar_params.os_rank_percent
                  << ", OSS=" << _compute.cfar_params.os_suppress_doppler
                  << "/" << _compute.cfar_params.os_suppress_range
                  << ", MD=" << (_compute.micro_doppler_enabled ? 1 : 0)
                  << ")";
}
