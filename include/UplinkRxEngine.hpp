#ifndef UPLINK_RX_ENGINE_HPP
#define UPLINK_RX_ENGINE_HPP

// UplinkRxEngine — focused uplink (UE->BS) OFDM receive + decode engine.
//
// Matched RX half of the duplex uplink. It reads the BS RX stream one downlink
// frame period at a time (anchored to the shared radio clock, offset by the
// passive DL/UL timing-difference setting (bs_dl_ul_timing_diff) — there is NO active sync/timing
// loop), extracts the uplink symbol window, and decodes it as a self-contained
// compact OFDM frame (make_uplink_config): per-symbol FFT, ZC-based channel
// estimation + equalization, QPSK LLRs, deinterleave/descramble/LDPC decode,
// LdpcPacketFraming parse, and UDP output. It also runs AGC on the uplink RX.
//
// Reuses only OFDMCore.hpp primitives (ChannelEstimator, QPSK_LLR,
// LdpcPacketFraming, Scrambler, BitBlockInterleaver) + LDPCCodec + the
// HardwareRxAgc helper. No sensing.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

#include <fftw3.h>
#include "RadioBackend.hpp"

#include "Common.hpp"
#include "OFDMCore.hpp"
#include "LDPCCodec.hpp"

class UplinkRxEngine {
private:
    using DebugVectorCallback = std::function<void(AlignedVector&&)>;

    struct UplinkRxFrame {
        AlignedVector samples;
        uint64_t generation = 0;
    };

    struct UplinkLlrFrame {
        LDPCCodec::AlignedFloatVector llr;
        uint64_t frame_index = 0;
        uint64_t generation = 0;
        double noise_var = 0.0;
    };

    // int16 (Q16) variant used when _link_cfg.ldpc.fixed_point is set.
    struct UplinkLlrFrameI16 {
        LDPCCodec::AlignedShortVector llr;
        uint64_t frame_index = 0;
        uint64_t generation = 0;
        double noise_var = 0.0;
    };

    static LDPCCodec::LDPCConfig _ldpc_cfg(const Config& c) {
        LDPCCodec::LDPCConfig cfg = make_ldpc_5041008_cfg();
        cfg.fixed_point = c.ldpc.fixed_point;
        return cfg;
    }

    static inline int16_t sat16_llr(float v) {
        long q = std::lroundf(v);
        if (q > 32767) q = 32767;
        else if (q < -32767) q = -32767;
        return static_cast<int16_t>(q);
    }

public:
    explicit UplinkRxEngine(const Config& link_cfg)
        : _link_cfg(link_cfg),
          _cfg(make_uplink_config(link_cfg)),
          _duplex(build_duplex_frame_layout(link_cfg)),
          _layout(build_data_resource_grid_layout(_cfg)),
          _zc_freq(generate_zc_freq(_cfg.ofdm.fft_size, _cfg.ofdm.zc_root)),
          _link_zc_freq(generate_zc_freq(link_cfg.ofdm.fft_size, link_cfg.ofdm.zc_root)),
          _self_debug_estimator(link_cfg.ofdm.fft_size, link_cfg.ofdm.cp_length),
          _fft_in(_cfg.ofdm.fft_size),
          _fft_out(_cfg.ofdm.fft_size),
          _ce_in(_cfg.ofdm.fft_size),
          _ce_out(_cfg.ofdm.fft_size),
          _period_samples(link_cfg.samples_per_frame()),
          _window_offset(_duplex.ul_sample_offset(link_cfg)),
          _window_samples(_duplex.ul_sample_count(link_cfg)),
          _link_sample_rate(link_cfg.rf_sampling.sample_rate > 0.0 ? link_cfg.rf_sampling.sample_rate : 1.0),
          _rx_agc(link_cfg) {
        if (_cfg.ofdm.num_symbols < 2) {
            throw std::runtime_error("UplinkRxEngine: uplink needs >= 2 OFDM symbols.");
        }
        _bit_interleaver = std::make_unique<BitBlockInterleaver>(_ldpc.get_N(), 21);
        _build_payload_indices();
        _build_tracking_tables();
        _fft_plan = fftwf_plan_dft_1d(
            static_cast<int>(_cfg.ofdm.fft_size),
            reinterpret_cast<fftwf_complex*>(_fft_in.data()),
            reinterpret_cast<fftwf_complex*>(_fft_out.data()),
            FFTW_FORWARD, FFTW_MEASURE);
        _ce_ifft_plan = fftwf_plan_dft_1d(
            static_cast<int>(_cfg.ofdm.fft_size),
            reinterpret_cast<fftwf_complex*>(_ce_in.data()),
            reinterpret_cast<fftwf_complex*>(_ce_out.data()),
            FFTW_BACKWARD, FFTW_MEASURE);
        _ce_fft_plan = fftwf_plan_dft_1d(
            static_cast<int>(_cfg.ofdm.fft_size),
            reinterpret_cast<fftwf_complex*>(_ce_out.data()),
            reinterpret_cast<fftwf_complex*>(_ce_in.data()),
            FFTW_FORWARD, FFTW_MEASURE);
        _udp_out = std::make_unique<UdpSender>(
            link_cfg.network_output.ul_udp_output_ip,
            static_cast<uint16_t>(link_cfg.network_output.ul_udp_output_port),
            link_cfg.network_output.udp_egress_pacer);
        _rx_frame_queue.reset(4, [this]() {
            UplinkRxFrame frame;
            frame.samples.resize(_period_samples);
            return frame;
        });
        if (_ldpc_fixed_point) {
            _llr_queue_i16.reset(8, [this]() {
                UplinkLlrFrameI16 frame;
                frame.llr.resize(_layout.payload_re_count * 2);
                return frame;
            });
        } else {
            _llr_queue.reset(8, [this]() {
                UplinkLlrFrame frame;
                frame.llr.resize(_layout.payload_re_count * 2);
                return frame;
            });
        }
        _data_symbols.assign(_layout.data_symbol_count, AlignedVector(_cfg.ofdm.fft_size));
    }

    ~UplinkRxEngine() {
        stop();
        if (_fft_plan) fftwf_destroy_plan(_fft_plan);
        if (_ce_ifft_plan) fftwf_destroy_plan(_ce_ifft_plan);
        if (_ce_fft_plan) fftwf_destroy_plan(_ce_fft_plan);
    }

    UplinkRxEngine(const UplinkRxEngine&) = delete;
    UplinkRxEngine& operator=(const UplinkRxEngine&) = delete;

    void set_rx_stream(radio::IRxStreamPtr stream) { _rx_stream = std::move(stream); }

    // AGC plumbing: host supplies a gain setter and the HW gain limits. If unset,
    // AGC is a no-op even when rx_agc_enable is true.
    void configure_agc(double initial_gain_db, double min_gain_db, double max_gain_db,
                       std::function<void(double)> apply_gain) {
        _rx_agc.initialize(initial_gain_db, min_gain_db, max_gain_db);
        _apply_gain = std::move(apply_gain);
    }

    // Passive DL/UL timing difference (samples), runtime-adjustable.
    std::atomic<int32_t>& dl_ul_timing_diff() { return _dl_ul_timing_diff; }
    const Config& uplink_config() const { return _cfg; }

    using ArqPayloadIntercept = std::function<bool(const uint8_t*, size_t, uint16_t, uint8_t)>;

    // ARQ: set payload intercept callback for decoded uplink packets.
    // The callback receives (payload, len, seq, flags). Return true to consume
    // the packet (suppress UDP output); return false to let normal UDP output
    // proceed. Feedback frames are identified by the mini-header flags.
    void set_arq_payload_intercept(ArqPayloadIntercept fn) {
        _arq_payload_intercept = std::move(fn);
    }

    void request_reacquire(const radio::TimeSpec& start_time) {
        {
            std::lock_guard<std::mutex> lock(_restart_mutex);
            _pending_restart_time = start_time;
        }
        _restart_requested.store(true, std::memory_order_release);
        LOG_G_WARN() << "[UL-RX] requested shared restart at "
                     << start_time.get_real_secs()
                     << " s, aligned to BS TX restart";
    }

    void request_reacquire() {
        {
            std::lock_guard<std::mutex> lock(_restart_mutex);
            _pending_restart_time = radio::TimeSpec(0.0);
        }
        _restart_requested.store(true, std::memory_order_release);
        LOG_G_WARN() << "[UL-RX] requested restart without timed anchor";
    }

    // Provider for the BS TX frame index, used by the duplex-invariant health log
    // to compare the BS TX frame anchor against the uplink-RX frame index.
    void set_bs_frame_provider(std::function<uint64_t()> fn) { _bs_frame_fn = std::move(fn); }

    void set_debug_sinks(
        DebugVectorCallback channel_sink,
        DebugVectorCallback delay_sink,
        DebugVectorCallback constellation_sink,
        DebugVectorCallback self_channel_sink = DebugVectorCallback{},
        DebugVectorCallback self_delay_sink = DebugVectorCallback{})
    {
        _channel_debug_sink = std::move(channel_sink);
        _delay_debug_sink = std::move(delay_sink);
        _constellation_debug_sink = std::move(constellation_sink);
        _self_channel_debug_sink = std::move(self_channel_sink);
        _self_delay_debug_sink = std::move(self_delay_sink);
    }

    void set_latest_delay_spectrum_sink(DebugVectorCallback sink) {
        _latest_delay_spectrum_sink = std::move(sink);
    }

    void set_latest_channel_estimate_sink(DebugVectorCallback sink) {
        _latest_channel_estimate_sink = std::move(sink);
    }

    void start(const radio::TimeSpec& start_time = radio::TimeSpec(0.0)) {
        if (!_rx_stream) throw std::runtime_error("UplinkRxEngine::start without an RX stream.");
        _running.store(true);
        _rx_thread = std::thread(&UplinkRxEngine::_rx_ingest_proc, this, start_time);
        _signal_thread = std::thread(&UplinkRxEngine::_signal_proc, this);
        _decode_thread = std::thread(&UplinkRxEngine::_decode_proc, this);
    }

    void stop() {
        _running.store(false);
        if (_rx_thread.joinable()) _rx_thread.join();
        if (_signal_thread.joinable()) _signal_thread.join();
        if (_decode_thread.joinable()) _decode_thread.join();
    }

    void _build_payload_indices() {
        _payload_subcarrier_indices_flat.clear();
        _payload_subcarrier_indices_flat.reserve(_layout.payload_re_count);
        size_t data_symbol_idx = 0;
        for (size_t sym = 0; sym < _cfg.ofdm.num_symbols; ++sym) {
            if (is_zc_sync_symbol(_cfg, sym)) continue;
            const size_t base = _layout.non_pilot_offsets[data_symbol_idx];
            for (size_t di = 0; di < _layout.num_non_pilot_subcarriers; ++di) {
                const size_t k = static_cast<size_t>(_layout.non_pilot_subcarrier_indices[di]);
                if (_layout.payload_rank[base + di] >= 0) {
                    _payload_subcarrier_indices_flat.push_back(static_cast<int>(k));
                }
            }
            ++data_symbol_idx;
        }
    }

    // Build the constant frequency-index table the per-frame equalizer needs:
    // FFT-buffer position -> signed physical frequency index (standard fft-shift
    // layout), required by equalize_symbol's per-subcarrier derotation. The
    // data-slot -> actual-symbol mapping reuses _layout.data_symbol_to_actual_symbol
    // (the same authoritative source that sizes _data_symbols).
    void _build_tracking_tables() {
        _track_enabled = (_cfg.uplink.equalizer.channel_tracking_mode != kChannelTrackingModeOff) &&
                         !_cfg.ofdm.pilot_positions.empty();
        _use_mmse = (_cfg.uplink.equalizer.equalizer_mode == kEqualizerModeMmse);

        _actual_subcarrier_indices.resize(_cfg.ofdm.fft_size);
        const int half = static_cast<int>(_cfg.ofdm.fft_size / 2);
        for (size_t i = 0; i < _cfg.ofdm.fft_size; ++i) {
            _actual_subcarrier_indices[i] = (static_cast<int>(i) >= half)
                ? static_cast<int>(i) - static_cast<int>(_cfg.ofdm.fft_size)
                : static_cast<int>(i);
        }
    }

    // Received-domain noise variance for MMSE equalization. Predicts comb
    // pilots from H_est plus alpha/beta, accumulates residual power, and falls
    // back to the sync impulse SNR if too few pilots are usable. The returned
    // value is in the same units as |H|^2.
    float _estimate_mmse_noise_var(float alpha, float beta) {
        const size_t fft_size = _cfg.ofdm.fft_size;
        const float floor_val = std::max(
            static_cast<float>(_cfg.uplink.equalizer.equalizer_mag_floor), 1e-12f);

        double err_acc = 0.0;
        size_t err_count = 0;
        for (size_t sym = 0; sym < _data_symbols.size(); ++sym) {
            const int actual_sym = _layout.data_symbol_to_actual_symbol[sym];
            const int rel = actual_sym - static_cast<int>(_cfg.ofdm.sync_pos);
            const float alpha_rel = alpha * static_cast<float>(rel);
            const float beta_rel = beta * static_cast<float>(rel);
            const auto& data_symbol = _data_symbols[sym];
            for (size_t p = 0; p < _cfg.ofdm.pilot_positions.size(); ++p) {
                const size_t k = _cfg.ofdm.pilot_positions[p];
                if (k >= fft_size) continue;
                const float phase = alpha_rel + beta_rel * static_cast<float>(_actual_subcarrier_indices[k]);
                const std::complex<float> pred =
                    (_h_est[k] * _zc_freq[k]) * std::polar(1.0f, phase);
                const float h_mag_sq = std::norm(_h_est[k]);
                if (!std::isfinite(h_mag_sq) || h_mag_sq <= floor_val) {
                    continue;
                }
                err_acc += std::norm(data_symbol[k] - pred);
                ++err_count;
            }
        }

        double h_power_acc = 0.0;
        size_t h_power_count = 0;
        auto accumulate_h_power = [&](size_t k) {
            if (k >= fft_size) return;
            const float mag_sq = std::norm(_h_est[k]);
            if (std::isfinite(mag_sq) && mag_sq > floor_val) {
                h_power_acc += mag_sq;
                ++h_power_count;
            }
        };
        for (const size_t k : _cfg.ofdm.pilot_positions) {
            accumulate_h_power(k);
        }
        if (h_power_count == 0) {
            for (size_t k = 0; k < fft_size; ++k) {
                accumulate_h_power(k);
            }
        }
        const double avg_h_power = h_power_count > 0
            ? std::max(h_power_acc / static_cast<double>(h_power_count), static_cast<double>(floor_val))
            : static_cast<double>(floor_val);
        double noise_var = avg_h_power / static_cast<double>(std::max(_sync_corrected_snr_linear, 1e-6f));
        if (err_count > 8) {
            noise_var = err_acc / static_cast<double>(err_count);
        }
        return static_cast<float>(std::min(std::max(noise_var, 1e-6), 1e6));
    }

    static int64_t _host_now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    // Read exactly `count` samples from the RX stream into out (blocking).
    // If `first_idx` is non-null, it receives the absolute sample index (shared
    // radio clock) of the first returned sample.
    size_t _read_exact(
        std::complex<float>* out,
        size_t count,
        int64_t* first_idx = nullptr,
        const radio::TimeSpec* stream_start_time = nullptr)
    {
        size_t got = 0;
        radio::RxMetadata md;
        bool got_idx = false;
        while (got < count && _running.load(std::memory_order_relaxed)) {
            const size_t n = _rx_stream->recv(out + got, count - got, md, 1.0, false);
            if (md.error_code != radio::RxError::None &&
                md.error_code != radio::RxError::Timeout) {
                _rx_error_count.fetch_add(1, std::memory_order_relaxed);
                LOG_RT_WARN() << "[UL-RX] RX metadata error: " << md.strerror();
                if (md.has_time_spec && stream_start_time != nullptr &&
                    stream_start_time->get_real_secs() > 0.0) {
                    _request_restart_at(
                        _next_frame_boundary_after(md.time_spec, *stream_start_time),
                        "RX metadata error");
                } else {
                    {
                        std::lock_guard<std::mutex> lock(_restart_mutex);
                        _pending_restart_time = radio::TimeSpec(0.0);
                    }
                    _restart_requested.store(true, std::memory_order_release);
                }
                break;
            }
            if (first_idx && !got_idx && n > 0) {
                *first_idx = md.time_spec.to_ticks(_tick_rate_for_idx());
                got_idx = true;
            }
            got += n;
        }
        return got;
    }

    // The sim RX timestamps each sample with the shared sample clock (tick_rate ==
    // sample_rate), so to_ticks(sample_rate) recovers the absolute sample index.
    double _tick_rate_for_idx() const { return _link_sample_rate; }

    int64_t _normalize_frame_offset(int64_t samples) const {
        const int64_t period = static_cast<int64_t>(_period_samples);
        if (period <= 0) return 0;
        samples %= period;
        if (samples < 0) samples += period;
        return samples;
    }

    radio::TimeSpec _next_frame_boundary_after(
        const radio::TimeSpec& event_time,
        const radio::TimeSpec& stream_start_time) const
    {
        const double tick_rate = _tick_rate_for_idx();
        const int64_t period_ticks = static_cast<int64_t>(_period_samples);
        if (tick_rate <= 0.0 || period_ticks <= 0) {
            return radio::TimeSpec(0.0);
        }
        const int64_t start_ticks = stream_start_time.to_ticks(tick_rate);
        const int64_t event_ticks = event_time.to_ticks(tick_rate);
        const int64_t lead_ticks = static_cast<int64_t>(std::ceil(tick_rate));
        const int64_t min_target = event_ticks + std::max<int64_t>(period_ticks, lead_ticks);
        int64_t frames_ahead = 0;
        if (min_target > start_ticks) {
            frames_ahead = (min_target - start_ticks + period_ticks - 1) / period_ticks;
        }
        return radio::TimeSpec::from_ticks(
            start_ticks + frames_ahead * period_ticks,
            tick_rate);
    }

    void _request_restart_at(const radio::TimeSpec& start_time, const char* reason) {
        {
            std::lock_guard<std::mutex> lock(_restart_mutex);
            _pending_restart_time = start_time;
        }
        _restart_requested.store(true, std::memory_order_release);
        LOG_RT_WARN() << "[UL-RX] requested restart after " << reason
                      << " at " << start_time.get_real_secs() << " s";
    }

    void _rx_ingest_proc(radio::TimeSpec start_time) {
        radio::set_thread_priority();
        bind_current_thread_from_uplink_hint(_link_cfg, 0);
        auto issue_start = [&](const radio::TimeSpec& stream_start_time) {
            if (!_rx_stream) return;
            radio::StreamCmd cmd(radio::StreamMode::StartContinuous);
            const bool timed_start = stream_start_time.get_real_secs() > 0.0;
            cmd.stream_now = !timed_start;
            if (timed_start) {
                cmd.time_spec = stream_start_time;
                LOG_G_INFO() << "[UL-RX] timed RX stream start at "
                             << stream_start_time.get_real_secs()
                             << " s, aligned to BS TX frame anchor";
            } else {
                LOG_G_INFO() << "[UL-RX] immediate RX stream start (no timed anchor)";
            }
            _rx_stream->issue_stream_cmd(cmd);
        };
        auto issue_stop = [&]() {
            if (!_rx_stream) return;
            try {
                _rx_stream->issue_stream_cmd(radio::StreamCmd(radio::StreamMode::StopContinuous));
            } catch (const std::exception& e) {
                LOG_RT_WARN() << "[UL-RX] failed to stop RX stream before restart: " << e.what();
            }
        };

        std::vector<std::complex<float>> skip_buf;
        auto discard = [&](int64_t n) {
            if (n <= 0) return;
            skip_buf.resize(static_cast<size_t>(n));
            _read_exact(skip_buf.data(), static_cast<size_t>(n), nullptr, &start_time);
        };

        auto start_stream_and_align = [&](const char* reason) {
            issue_start(start_time);
            const uint64_t generation =
                _stream_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
            const int64_t diff = _normalize_frame_offset(
                _dl_ul_timing_diff.load(std::memory_order_relaxed));
            discard(diff);
            LOG_RT_WARN() << "[UL-RX] stream generation " << generation
                          << " aligned after " << reason
                          << " with DUTI=" << diff << " samples";
            return diff;
        };

        auto apply_pending_restart = [&](int64_t& applied_diff) {
            if (!_restart_requested.exchange(false, std::memory_order_acq_rel)) {
                return false;
            }
            radio::TimeSpec restart_time(0.0);
            {
                std::lock_guard<std::mutex> lock(_restart_mutex);
                restart_time = _pending_restart_time;
                _pending_restart_time = radio::TimeSpec(0.0);
            }
            issue_stop();
            start_time = restart_time;
            applied_diff = start_stream_and_align("restart");
            return true;
        };

        // Deterministic frame anchoring — NO uplink-ZC correlation search.
        int64_t applied_diff = start_stream_and_align("initial start");

        while (_running.load(std::memory_order_relaxed)) {
            if (apply_pending_restart(applied_diff)) {
                continue;
            }
            // Runtime DUTI adjustment (operator-driven residual trim). Only positive
            // increments are realizable in a streaming reader (we cannot un-read);
            // a decrease is applied by skipping a near-full frame period so the net
            // boundary shift equals the requested delta (mod period).
            const int64_t target = _normalize_frame_offset(
                _dl_ul_timing_diff.load(std::memory_order_relaxed));
            if (target != applied_diff) {
                int64_t delta = target - applied_diff;
                while (delta < 0) delta += static_cast<int64_t>(_period_samples);
                discard(delta);
                applied_diff = target;
            }

            UplinkRxFrame* frame = nullptr;
            SPSCBackoff queue_backoff;
            bool restarted = false;
            while (_running.load(std::memory_order_relaxed)) {
                frame = _rx_frame_queue.producer_slot();
                if (frame != nullptr) break;
                if (apply_pending_restart(applied_diff)) {
                    restarted = true;
                    break;
                }
                queue_backoff.pause();
            }
            if (restarted) {
                continue;
            }
            if (frame == nullptr) {
                break;
            }
            const uint64_t frame_generation =
                _stream_generation.load(std::memory_order_acquire);
            const size_t got = _read_exact(
                frame->samples.data(),
                _period_samples,
                nullptr,
                &start_time);
            if (got != _period_samples ||
                _restart_requested.load(std::memory_order_acquire)) {
                if (!_running.load(std::memory_order_relaxed)) break;
                continue;
            }
            frame->generation = frame_generation;
            _rx_frame_queue.producer_commit();
        }

        issue_stop();
    }

    void _signal_proc() {
        radio::set_thread_priority();
        bind_current_thread_from_uplink_hint(_link_cfg, 1);
        SPSCBackoff backoff;
        while (_running.load(std::memory_order_relaxed) || !_rx_frame_queue.empty()) {
            UplinkRxFrame* frame = _rx_frame_queue.consumer_slot();
            if (frame == nullptr) {
                backoff.pause();
                continue;
            }
            backoff.reset();
            const uint64_t generation = frame->generation;
            if (generation == _stream_generation.load(std::memory_order_acquire)) {
                _process_frame(frame->samples, generation);
            }
            _rx_frame_queue.consumer_pop();
        }
    }

    void _decode_proc() {
        radio::set_thread_priority();
        bind_current_thread_from_uplink_hint(_link_cfg, 2);
        SPSCBackoff backoff;
        if (_ldpc_fixed_point) {
            while (_running.load(std::memory_order_relaxed) || !_llr_queue_i16.empty()) {
                UplinkLlrFrameI16* frame = _llr_queue_i16.consumer_slot();
                if (frame == nullptr) {
                    backoff.pause();
                    continue;
                }
                backoff.reset();
                if (frame->generation == _stream_generation.load(std::memory_order_acquire)) {
                    _decode_llr_stream(frame->llr, _deint_scratch_i16, _payload_llr_scratch_i16);
                }
                _llr_queue_i16.consumer_pop();
            }
        } else {
            while (_running.load(std::memory_order_relaxed) || !_llr_queue.empty()) {
                UplinkLlrFrame* frame = _llr_queue.consumer_slot();
                if (frame == nullptr) {
                    backoff.pause();
                    continue;
                }
                backoff.reset();
                if (frame->generation == _stream_generation.load(std::memory_order_acquire)) {
                    _decode_llr_stream(frame->llr, _deint_scratch, _payload_llr_scratch);
                }
                _llr_queue.consumer_pop();
            }
        }
    }

    void _process_frame(const AlignedVector& frame, uint64_t generation) {
        const std::complex<float>* win = frame.data() + _window_offset;
        const size_t sym_len = _cfg.ofdm.fft_size + _cfg.ofdm.cp_length;

        // 1) Channel estimate from the uplink sync symbol. The derived uplink
        //    config sets sync_pos relative to the uplink window, avoiding a
        //    hard-coded symbol-0 assumption.
        const size_t sync_pos = _cfg.ofdm.sync_pos;
        _fft_symbol(win + sync_pos * sym_len + _cfg.ofdm.cp_length, _sync_freq_buf);
        _estimate_sync_channel_ls_wiener(_sync_freq_buf, _zc_freq, _h_est);

        // 2) FFT every data symbol (scaled, pre-equalization). The frequency-domain
        //    symbols are produced before equalization so the pilot-phase tracker can
        //    measure the residual phase ramp from adjacent-symbol pilot products.
        size_t data_idx = 0;
        for (size_t sym = 0; sym < _cfg.ofdm.num_symbols; ++sym) {
            if (is_zc_sync_symbol(_cfg, sym)) continue;
            _fft_symbol(win + sym * sym_len + _cfg.ofdm.cp_length, _data_symbols[data_idx]);
            ++data_idx;
        }

        // 3) Channel tracking. TX and RX run on independent LO/sample clocks, so a
        //    residual CFO and sample-clock offset (SFO) accumulate across the data
        //    symbols relative to the single sync-symbol channel estimate. Estimate
        //    the per-symbol phase ramp from comb pilots: alpha = constant (CFO),
        //    beta = frequency slope (SFO). Honors channel_tracking_mode.
        float alpha = 0.0f, beta = 0.0f;
        if (_track_enabled) {
            const bool est_valid = FrequencyOffsetEstimator::compute_pilot_phase_diff(
                _data_symbols, _cfg.ofdm.pilot_positions, _cfg.ofdm.fft_size, _cfg.ofdm.sync_pos,
                _pilot_indices_buf, _avg_phase_diff_buf, _weights_buf,
                &_layout.data_symbol_to_actual_symbol, nullptr);
            if (est_valid) {
                unwrap(_avg_phase_diff_buf);
                std::tie(beta, alpha) = weightedlinearRegression(
                    _pilot_indices_buf, _avg_phase_diff_buf, _weights_buf);
                if (!std::isfinite(alpha) || !std::isfinite(beta)) {
                    alpha = 0.0f;
                    beta = 0.0f;
                }
            }
        }

        // 4) Equalizer inverse. Honors equalizer_mode (ZF vs MMSE); MMSE uses
        //    pilot-residual noise variance with sync-SNR fallback.
        if (_use_mmse) {
            ChannelEstimator::compute_mmse_inverse(
                _h_est, _h_inv, _estimate_mmse_noise_var(alpha, beta),
                static_cast<float>(_cfg.uplink.equalizer.equalizer_mag_floor));
        } else {
            ChannelEstimator::compute_zf_inverse(
                _h_est, _h_inv, static_cast<float>(_cfg.uplink.equalizer.equalizer_mag_floor));
        }

        // 5) Equalize + derotate each data symbol in place. Reuses the SIMD
        //    equalize_symbol; the derotation removes the accumulated alpha*n (CFO)
        //    and beta*n*idx (SFO) for the n-th symbol after the sync anchor. When
        //    tracking is off, alpha=beta=0 and this is plain ZF/MMSE equalization.
        for (size_t i = 0; i < _data_symbols.size(); ++i) {
            const int rel = _layout.data_symbol_to_actual_symbol[i] -
                            static_cast<int>(_cfg.ofdm.sync_pos);
            ChannelEstimator::equalize_symbol(
                _data_symbols[i], _h_inv,
                alpha * static_cast<float>(rel),
                beta * static_cast<float>(rel),
                _actual_subcarrier_indices);
        }

        // 6) Noise variance from equalized pilots -> LLR scale.
        const double noise_var = QPSK_LLR::estimate_noise_variance(
            _data_symbols, _cfg.ofdm.pilot_positions, _zc_freq);

        // Periodic alignment-health / drift indicator: with a locked frame the
        // sync-symbol channel estimate stays a clean impulse (|H| flat). A rising
        // pilot noise variance signals drift — re-acquisition would be needed.
        if ((_frame_count++ % 4096) == 0) {
            // Duplex invariant: UL-RX frame index should track the BS TX frame
            // anchor at a fixed relative offset; a growing gap signals drift.
            std::ostringstream inv;
            if (_bs_frame_fn) {
                const uint64_t bs_frame = _bs_frame_fn();
                inv << ", bs_tx_frame=" << bs_frame
                    << ", ul_rx-bs_tx_gap=" << (static_cast<int64_t>(_frame_count) -
                                                static_cast<int64_t>(bs_frame));
            }
            LOG_G_INFO() << "[UL-RX] health: ul_rx_frame=" << _frame_count
                         << ", pilot_noise_var=" << noise_var
                         << ", decoded=" << _decoded_count.load(std::memory_order_relaxed)
                         << inv.str();
        }
        const float scale_llr = std::min(
            static_cast<float>((4.0 / std::max(noise_var, 1e-6)) * M_SQRT1_2), 500.0f);

        // 7) Build the matched payload-RE LLR stream (symbol-major, flat order).
        if (_layout.payload_re_count > 0 && _ldpc_fixed_point) {
            // Fused pow2 quantization: write int16 LLRs directly (no extra pass).
            const float scale_llr_q =
                scale_llr * static_cast<float>(_link_cfg.ldpc.fixed_point_scale);
            UplinkLlrFrameI16* llr_frame = nullptr;
            SPSCBackoff llr_backoff;
            while (_running.load(std::memory_order_relaxed)) {
                llr_frame = _llr_queue_i16.producer_slot();
                if (llr_frame != nullptr) break;
                llr_backoff.pause();
            }
            if (llr_frame == nullptr) {
                return;
            }
            llr_frame->frame_index = _frame_count;
            llr_frame->generation = generation;
            llr_frame->noise_var = noise_var;
            llr_frame->llr.resize(_layout.payload_re_count * 2);
            int16_t* __restrict__ llr = llr_frame->llr.data();
            for (size_t s = 0; s < _data_symbols.size(); ++s) {
                const auto* __restrict__ sym_ptr = _data_symbols[s].data();
                const size_t begin = _layout.payload_offsets[s];
                const size_t end = _layout.payload_offsets[s + 1];
                size_t off = begin * 2;
                for (size_t idx = begin; idx < end; ++idx) {
                    const size_t k = static_cast<size_t>(_payload_subcarrier_indices_flat[idx]);
                    llr[off++] = sat16_llr(sym_ptr[k].real() * scale_llr_q);
                    llr[off++] = sat16_llr(sym_ptr[k].imag() * scale_llr_q);
                }
            }
            _llr_queue_i16.producer_commit();
        } else if (_layout.payload_re_count > 0) {
            UplinkLlrFrame* llr_frame = nullptr;
            SPSCBackoff llr_backoff;
            while (_running.load(std::memory_order_relaxed)) {
                llr_frame = _llr_queue.producer_slot();
                if (llr_frame != nullptr) break;
                llr_backoff.pause();
            }
            if (llr_frame == nullptr) {
                return;
            }
            llr_frame->frame_index = _frame_count;
            llr_frame->generation = generation;
            llr_frame->noise_var = noise_var;
            llr_frame->llr.resize(_layout.payload_re_count * 2);
            float* __restrict__ llr = llr_frame->llr.data();
            for (size_t s = 0; s < _data_symbols.size(); ++s) {
                const auto* __restrict__ sym_ptr = _data_symbols[s].data();
                const size_t begin = _layout.payload_offsets[s];
                const size_t end = _layout.payload_offsets[s + 1];
                size_t off = begin * 2;
                for (size_t idx = begin; idx < end; ++idx) {
                    const size_t k = static_cast<size_t>(_payload_subcarrier_indices_flat[idx]);
                    llr[off++] = sym_ptr[k].real() * scale_llr;
                    llr[off++] = sym_ptr[k].imag() * scale_llr;
                }
            }
            _llr_queue.producer_commit();
        }

        _update_self_channel_debug(frame);
        _publish_debug_streams();

        // 8) AGC from the channel impulse-response delay peak.
        _run_agc(frame);
    }

    void _fft_symbol(const std::complex<float>* time_in, AlignedVector& freq_out) {
        std::memcpy(_fft_in.data(), time_in, _cfg.ofdm.fft_size * sizeof(std::complex<float>));
        fftwf_execute(_fft_plan);
        freq_out.resize(_cfg.ofdm.fft_size);
        const float s = 1.0f / std::sqrt(static_cast<float>(_cfg.ofdm.fft_size));
        const auto* __restrict__ src = _fft_out.data();
        auto* __restrict__ dst = freq_out.data();
        #pragma omp simd simdlen(16)
        for (size_t k = 0; k < _cfg.ofdm.fft_size; ++k) dst[k] = src[k] * s;
    }

    // Parse LdpcPacketFraming packets out of the frame LLR stream and decode.
    // Templated on the LLR sample type so the float and int16 (Q16) paths share
    // one body; decode_frame() is overload-resolved by the payload vector type.
    template <typename LlrVec, typename ScratchVec>
    void _decode_llr_stream(const LlrVec& frame_llr, ScratchVec& deint_scratch,
                            LlrVec& payload_scratch) {
        const size_t bits_per_block = _ldpc.get_N();
        const size_t bytes_per_block = (_ldpc.get_K() + 7) / 8;
        if (bits_per_block != LdpcPacketFraming::kLdpcCodeBitsPerBlock ||
            bytes_per_block != LdpcPacketFraming::kLdpcInfoBytesPerBlock) {
            return;
        }
        size_t symbol_offset = 0;
        while ((symbol_offset + LdpcPacketFraming::kControlSymbols) * 2 <= frame_llr.size()) {
            const size_t control_off = symbol_offset * 2;
            const bool mk = LdpcPacketFraming::detect_marker_llrs(frame_llr.data() + control_off);
            if (!mk) break;
            LdpcMiniHeader hdr;
            if (!LdpcPacketFraming::decode_mini_header_llrs(
                    frame_llr.data() + control_off + LdpcPacketFraming::kMarkerBits, hdr)) {
                break;
            }
            const size_t payload_blocks = LdpcPacketFraming::payload_blocks_for_len(hdr.payload_len);
            const size_t required = payload_blocks * bits_per_block;
            const size_t payload_off = control_off + LdpcPacketFraming::kControlBits;
            const size_t next_off = symbol_offset + LdpcPacketFraming::packet_qpsk_symbols(payload_blocks);
            if (payload_off + required > frame_llr.size()) break;
            if (payload_blocks == 0) { symbol_offset = next_off; continue; }

            // Reuse scratch buffers across packets (decode thread, but still on
            // the hot per-frame path) — std::copy fully overwrites payload_llr.
            LlrVec& payload_llr = payload_scratch;
            payload_llr.resize(required);
            std::copy(frame_llr.begin() + payload_off,
                      frame_llr.begin() + payload_off + required, payload_llr.begin());
            _bit_interleaver->deinterleave_inplace(payload_llr, deint_scratch);
            _descrambler.soft_descramble(payload_llr);
            LDPCCodec::AlignedByteVector& decoded = _decoded_scratch;
            try {
                _ldpc.decode_frame(payload_llr, decoded);
            } catch (const std::exception& e) {
                LOG_RT_WARN_HZ(1) << "[UL-RX] LDPC decode failed: " << e.what();
                symbol_offset = next_off;
                continue;
            }
            if (decoded.size() >= hdr.payload_len) {
                const auto* payload_ptr = reinterpret_cast<const uint8_t*>(decoded.data());
                // ARQ intercept: if callback set and returns true, skip UDP output
                if (!_arq_payload_intercept ||
                    !_arq_payload_intercept(payload_ptr, hdr.payload_len, hdr.seq, hdr.flags)) {
                    _udp_out->send(payload_ptr, hdr.payload_len);
                }
                _decoded_count.fetch_add(1, std::memory_order_relaxed);
            }
            symbol_offset = next_off;
        }
    }

    void _estimate_sync_channel_ls_wiener(
        const AlignedVector& rx_symbol,
        const AlignedVector& tx_zc,
        AlignedVector& h_est)
    {
        const size_t fft_size = _cfg.ofdm.fft_size;
        const size_t cp_length = _cfg.ofdm.cp_length;
        h_est.resize(fft_size);

        // LS estimate into _ce_in. This mirrors ls_channel_estimate_kernel:
        // complex divide by the known ZC, with a denominator guard.
        const auto* __restrict__ rx = rx_symbol.data();
        const auto* __restrict__ tx = tx_zc.data();
        auto* __restrict__ h_ls = _ce_in.data();
        #pragma omp simd simdlen(16)
        for (size_t k = 0; k < fft_size; ++k) {
            const float rx_re = rx[k].real();
            const float rx_im = rx[k].imag();
            const float tx_re = tx[k].real();
            const float tx_im = tx[k].imag();
            float denom = tx_re * tx_re + tx_im * tx_im;
            if (denom < 1e-12f) denom = 1e-12f;
            const float inv_denom = 1.0f / denom;
            h_ls[k] = std::complex<float>(
                (rx_re * tx_re + rx_im * tx_im) * inv_denom,
                (rx_im * tx_re - rx_re * tx_im) * inv_denom);
        }

        // IFFT to impulse response and normalize by 1/N, matching cuFFT inverse
        // followed by normalize_ifft_kernel.
        fftwf_execute(_ce_ifft_plan);
        const float inv_n = 1.0f / static_cast<float>(fft_size);
        auto* __restrict__ h_time = _ce_out.data();
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < fft_size; ++i) {
            h_time[i] *= inv_n;
        }

        double sig_energy = 0.0;
        double noise_energy = 0.0;
        double cpu_sig_energy = 0.0;
        double cpu_noise_energy = 0.0;
        #pragma omp simd simdlen(16) reduction(+:sig_energy,noise_energy,cpu_sig_energy,cpu_noise_energy)
        for (size_t i = 0; i < fft_size; ++i) {
            const double power = static_cast<double>(std::norm(h_time[i]));
            if (i < cp_length || i > fft_size - cp_length) {
                sig_energy += power;
            } else {
                noise_energy += power;
            }
            if (i < cp_length) {
                cpu_sig_energy += power;
            } else {
                cpu_noise_energy += power;
            }
        }
        const float snr = (noise_energy > 1e-12)
            ? static_cast<float>(sig_energy / noise_energy)
            : 1000.0f;
        const float wiener = snr / (snr + 1.0f);
        _sync_corrected_snr_linear = 1.0f;
        if (fft_size > cp_length && cp_length > 0) {
            double noise_power = cpu_noise_energy / static_cast<double>(fft_size - cp_length);
            if (noise_power < 1e-10) noise_power = 1e-10;
            double signal_power = cpu_sig_energy / static_cast<double>(cp_length) - noise_power;
            if (signal_power < 0.0) signal_power = 0.0;
            const double raw_snr = signal_power / noise_power;
            _sync_corrected_snr_linear = static_cast<float>(
                raw_snr * static_cast<double>(cp_length) / static_cast<double>(fft_size));
        }
        if (_sync_corrected_snr_linear < 1e-6f) {
            _sync_corrected_snr_linear = 1e-6f;
        }

        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < fft_size; ++i) {
            if (i < cp_length || i > fft_size - cp_length) {
                h_time[i] *= wiener;
            } else {
                h_time[i] = {0.0f, 0.0f};
            }
        }

        fftwf_execute(_ce_fft_plan);
        std::memcpy(h_est.data(), _ce_in.data(), fft_size * sizeof(std::complex<float>));
    }

    void _run_agc(const AlignedVector& frame) {
        if (!_rx_agc.enabled() || !_apply_gain) return;
        // Delay spectrum (impulse response) from the channel estimate.
        const size_t half = _cfg.ofdm.fft_size / 2;
        for (size_t i = 0; i < half; ++i) {
            _ce_in[i] = _h_est[i + half];
            _ce_in[i + half] = _h_est[i];
        }
        fftwf_execute(_ce_ifft_plan);
        _delay_spectrum.resize(_cfg.ofdm.fft_size);
        const float scale = 1.0f / std::sqrt(static_cast<float>(_cfg.ofdm.fft_size));
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _cfg.ofdm.fft_size; ++i) {
            _delay_spectrum[i] = _ce_out[i] * scale;
        }
        size_t peak_index = 0;
        float peak = 0.0f;
        float avg = 0.0f;
        DelayProcessor::find_peak(_delay_spectrum, peak_index, peak, avg, _cfg.ofdm.cp_length);
        const std::complex<float>* sync_time = frame.data() + _window_offset + _cfg.ofdm.cp_length;
        RxAgcAdjustment agc_adjustment;
        const bool adjusted = _rx_agc.maybe_apply_from_delay_peak(
            peak, avg, sync_time, _cfg.ofdm.fft_size, _host_now_ns(), _agc_gates,
            [this](double g) { if (_apply_gain) _apply_gain(g); },
            &agc_adjustment);
        if (adjusted && _link_cfg.should_profile("agc")) {
            LOG_RT_INFO() << "[UL-RX] RX AGC adjusted gain to " << agc_adjustment.next_gain_db
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

    void _update_self_channel_debug(const AlignedVector& frame) {
        if (!uplink_self_channel_debug_enabled(_link_cfg) ||
            (!_self_channel_debug_sink && !_self_delay_debug_sink)) {
            return;
        }
        const size_t link_sym_len = _link_cfg.ofdm.fft_size + _link_cfg.ofdm.cp_length;
        const size_t local_tx_zc_start = _link_cfg.ofdm.sync_pos * link_sym_len;
        if (!_self_debug_estimator.estimate(
                frame,
                local_tx_zc_start,
                0,
                _link_zc_freq,
                _self_h_est)) {
            _self_h_est.clear();
            _self_delay_spectrum.clear();
            return;
        }
        if ((_self_debug_frame_counter++ % 4096) == 0) {
            LOG_G_INFO() << "[UL-RX] self-channel debug: local BS ZC offset="
                         << local_tx_zc_start
                         << ", rx_frame_start_offset=0"
                         << ", frame_period=" << frame.size();
        }
    }

    void _publish_debug_streams() {
        if (_channel_debug_sink) {
            _channel_debug_sink(AlignedVector(_h_est.begin(), _h_est.end()));
        }
        if (_latest_channel_estimate_sink) {
            _latest_channel_estimate_sink(AlignedVector(_h_est.begin(), _h_est.end()));
        }

        if (_self_channel_debug_sink && !_self_h_est.empty()) {
            _self_channel_debug_sink(AlignedVector(_self_h_est.begin(), _self_h_est.end()));
        }

        const bool need_delay = static_cast<bool>(_delay_debug_sink) ||
                                static_cast<bool>(_latest_delay_spectrum_sink);
        if (need_delay) {
            const size_t half = _cfg.ofdm.fft_size / 2;
            for (size_t i = 0; i < half; ++i) {
                _ce_in[i] = _h_est[i + half];
                _ce_in[i + half] = _h_est[i];
            }
            fftwf_execute(_ce_ifft_plan);
            _delay_spectrum.resize(_cfg.ofdm.fft_size);
            const float scale = 1.0f / std::sqrt(static_cast<float>(_cfg.ofdm.fft_size));
            for (size_t i = 0; i < _cfg.ofdm.fft_size; ++i) {
                _delay_spectrum[i] = _ce_out[i] * scale;
            }
            if (_delay_debug_sink) {
                _delay_debug_sink(AlignedVector(_delay_spectrum.begin(), _delay_spectrum.end()));
            }
            if (_latest_delay_spectrum_sink) {
                _latest_delay_spectrum_sink(AlignedVector(_delay_spectrum.begin(), _delay_spectrum.end()));
            }
        }

        const bool need_self_delay = static_cast<bool>(_self_delay_debug_sink) && !_self_h_est.empty();
        if (need_self_delay) {
            const size_t half = _link_cfg.ofdm.fft_size / 2;
            for (size_t i = 0; i < half; ++i) {
                _ce_in[i] = _self_h_est[i + half];
                _ce_in[i + half] = _self_h_est[i];
            }
            fftwf_execute(_ce_ifft_plan);
            _self_delay_spectrum.resize(_link_cfg.ofdm.fft_size);
            const float scale = 1.0f / std::sqrt(static_cast<float>(_link_cfg.ofdm.fft_size));
            for (size_t i = 0; i < _link_cfg.ofdm.fft_size; ++i) {
                _self_delay_spectrum[i] = _ce_out[i] * scale;
            }
            _self_delay_debug_sink(AlignedVector(_self_delay_spectrum.begin(), _self_delay_spectrum.end()));
        }

        constexpr uint32_t kConstellationStride = 8;
        if (_constellation_debug_sink && !_data_symbols.empty() &&
            ((_constellation_frame_counter++ % kConstellationStride) == 0)) {
            const auto& symbol = _data_symbols.back();
            _constellation_debug_sink(AlignedVector(symbol.begin(), symbol.end()));
        }
    }

    const Config _link_cfg;
    const Config _cfg;
    const DuplexFrameLayout _duplex;
    const DataResourceGridLayout _layout;
    AlignedVector _zc_freq;
    AlignedVector _link_zc_freq;
    SelfZcChannelDebugEstimator _self_debug_estimator;

    AlignedVector _fft_in, _fft_out;
    fftwf_plan _fft_plan = nullptr;
    AlignedVector _ce_in, _ce_out;       // channel-estimate IFFT for AGC/debug delay peak
    AlignedVector _delay_spectrum;
    AlignedVector _self_delay_spectrum;
    fftwf_plan _ce_ifft_plan = nullptr;
    fftwf_plan _ce_fft_plan = nullptr;

    AlignedVector _sync_freq_buf, _h_est, _h_inv;
    AlignedVector _self_h_est;
    std::vector<AlignedVector> _data_symbols;
    std::vector<int> _payload_subcarrier_indices_flat;
    float _sync_corrected_snr_linear = 1.0f;

    // Channel-tracking / equalizer-mode state (honors channel_tracking_mode and
    // equalizer_mode on the derived uplink config).
    bool _track_enabled = false;
    bool _use_mmse = false;
    std::vector<int> _actual_subcarrier_indices;   // fft pos -> signed freq index
    std::vector<int> _pilot_indices_buf;           // compute_pilot_phase_diff scratch
    std::vector<float> _avg_phase_diff_buf;        // compute_pilot_phase_diff scratch
    std::vector<float> _weights_buf;               // compute_pilot_phase_diff scratch

    const size_t _period_samples;
    const size_t _window_offset;
    const size_t _window_samples;
    const double _link_sample_rate;
    SPSCRingBuffer<UplinkRxFrame> _rx_frame_queue;
    SPSCRingBuffer<UplinkLlrFrame> _llr_queue;
    SPSCRingBuffer<UplinkLlrFrameI16> _llr_queue_i16;     // used when _ldpc_fixed_point

    const bool _ldpc_fixed_point{_link_cfg.ldpc.fixed_point};
    LDPCCodec _ldpc{_ldpc_cfg(_link_cfg)};
    std::unique_ptr<BitBlockInterleaver> _bit_interleaver;
    LDPCCodec::AlignedFloatVector _deint_scratch;
    LDPCCodec::AlignedFloatVector _payload_llr_scratch;   // reused per-packet LDPC input
    LDPCCodec::AlignedShortVector _deint_scratch_i16;     // int16 path scratch
    LDPCCodec::AlignedShortVector _payload_llr_scratch_i16; // int16 path per-packet input
    LDPCCodec::AlignedByteVector _decoded_scratch;        // reused per-packet LDPC output
    Scrambler _descrambler{201600, 0x5A};
    std::unique_ptr<UdpSender> _udp_out;

    HardwareRxAgc _rx_agc;
    DemodControlTimeGates _agc_gates;
    std::function<void(double)> _apply_gain;
    DebugVectorCallback _channel_debug_sink;
    DebugVectorCallback _delay_debug_sink;
    DebugVectorCallback _constellation_debug_sink;
    DebugVectorCallback _self_channel_debug_sink;
    DebugVectorCallback _self_delay_debug_sink;
    DebugVectorCallback _latest_delay_spectrum_sink;
    DebugVectorCallback _latest_channel_estimate_sink;

    ArqPayloadIntercept _arq_payload_intercept;

    radio::IRxStreamPtr _rx_stream;
    std::atomic<int32_t> _dl_ul_timing_diff{0};
    std::mutex _restart_mutex;
    radio::TimeSpec _pending_restart_time{0.0};
    std::atomic<bool> _restart_requested{false};
    std::atomic<uint64_t> _stream_generation{0};
    std::atomic<uint64_t> _rx_error_count{0};
    std::function<uint64_t()> _bs_frame_fn;
    std::atomic<uint64_t> _decoded_count{0};
    uint64_t _frame_count = 0;
    uint64_t _self_debug_frame_counter = 0;
    uint32_t _constellation_frame_counter = 0;
    std::atomic<bool> _running{false};
    std::thread _rx_thread;
    std::thread _signal_thread;
    std::thread _decode_thread;
};

#endif // UPLINK_RX_ENGINE_HPP
