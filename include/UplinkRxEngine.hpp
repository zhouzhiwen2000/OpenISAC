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
#include <uhd/stream.hpp>
#include <uhd/types/time_spec.hpp>

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
        _udp_out = std::make_unique<UdpSender>(
            link_cfg.network_output.ul_udp_output_ip, static_cast<uint16_t>(link_cfg.network_output.ul_udp_output_port));
        _rx_frame_queue.reset(4, [this]() {
            UplinkRxFrame frame;
            frame.samples.resize(_period_samples);
            return frame;
        });
        _llr_queue.reset(8, [this]() {
            UplinkLlrFrame frame;
            frame.llr.resize(_layout.payload_re_count * 2);
            return frame;
        });
        _data_symbols.assign(_layout.data_symbol_count, AlignedVector(_cfg.ofdm.fft_size));
    }

    ~UplinkRxEngine() {
        stop();
        if (_fft_plan) fftwf_destroy_plan(_fft_plan);
        if (_ce_ifft_plan) fftwf_destroy_plan(_ce_ifft_plan);
    }

    UplinkRxEngine(const UplinkRxEngine&) = delete;
    UplinkRxEngine& operator=(const UplinkRxEngine&) = delete;

    void set_rx_stream(uhd::rx_streamer::sptr stream) { _rx_stream = std::move(stream); }

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

    void request_reacquire(const uhd::time_spec_t& start_time) {
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
            _pending_restart_time = uhd::time_spec_t(0.0);
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

    void start(const uhd::time_spec_t& start_time = uhd::time_spec_t(0.0)) {
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

    // Received-domain noise variance for MMSE equalization. The sync channel
    // impulse response concentrates signal energy within the CP; energy beyond it
    // is noise (estimate_snr_from_impulse_response). Scaling the resulting
    // signal-normalized noise variance by the mean channel power |H|^2 puts it in
    // the same units as compute_mmse_inverse's |H|^2 + noise_var denominator.
    float _estimate_mmse_noise_var() {
        std::memcpy(_ce_in.data(), _h_est.data(), _cfg.ofdm.fft_size * sizeof(std::complex<float>));
        fftwf_execute(_ce_ifft_plan);
        const float snr = ChannelEstimator::estimate_snr_from_impulse_response(
            _ce_out, _cfg.ofdm.cp_length);
        const double norm_nv = noise_variance_from_snr_linear(std::max(snr, 1e-6f));

        const auto* __restrict__ h = _h_est.data();
        double avg_power = 0.0;
        #pragma omp simd simdlen(16) reduction(+:avg_power)
        for (size_t k = 0; k < _cfg.ofdm.fft_size; ++k) {
            avg_power += h[k].real() * h[k].real() + h[k].imag() * h[k].imag();
        }
        avg_power /= static_cast<double>(_cfg.ofdm.fft_size);
        return static_cast<float>(norm_nv * avg_power);
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
        const uhd::time_spec_t* stream_start_time = nullptr)
    {
        size_t got = 0;
        uhd::rx_metadata_t md;
        bool got_idx = false;
        while (got < count && _running.load(std::memory_order_relaxed)) {
            const size_t n = _rx_stream->recv(out + got, count - got, md, 1.0, false);
            if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE &&
                md.error_code != uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
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
                        _pending_restart_time = uhd::time_spec_t(0.0);
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

    uhd::time_spec_t _next_frame_boundary_after(
        const uhd::time_spec_t& event_time,
        const uhd::time_spec_t& stream_start_time) const
    {
        const double tick_rate = _tick_rate_for_idx();
        const int64_t period_ticks = static_cast<int64_t>(_period_samples);
        if (tick_rate <= 0.0 || period_ticks <= 0) {
            return uhd::time_spec_t(0.0);
        }
        const int64_t start_ticks = stream_start_time.to_ticks(tick_rate);
        const int64_t event_ticks = event_time.to_ticks(tick_rate);
        const int64_t lead_ticks = static_cast<int64_t>(std::ceil(tick_rate));
        const int64_t min_target = event_ticks + std::max<int64_t>(period_ticks, lead_ticks);
        int64_t frames_ahead = 0;
        if (min_target > start_ticks) {
            frames_ahead = (min_target - start_ticks + period_ticks - 1) / period_ticks;
        }
        return uhd::time_spec_t::from_ticks(
            start_ticks + frames_ahead * period_ticks,
            tick_rate);
    }

    void _request_restart_at(const uhd::time_spec_t& start_time, const char* reason) {
        {
            std::lock_guard<std::mutex> lock(_restart_mutex);
            _pending_restart_time = start_time;
        }
        _restart_requested.store(true, std::memory_order_release);
        LOG_RT_WARN() << "[UL-RX] requested restart after " << reason
                      << " at " << start_time.get_real_secs() << " s";
    }

    void _rx_ingest_proc(uhd::time_spec_t start_time) {
        uhd::set_thread_priority_safe();
        bind_current_thread_from_uplink_hint(_link_cfg, 0);
        auto issue_start = [&](const uhd::time_spec_t& stream_start_time) {
            if (!_rx_stream) return;
            uhd::stream_cmd_t cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
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
                _rx_stream->issue_stream_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
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
            uhd::time_spec_t restart_time(0.0);
            {
                std::lock_guard<std::mutex> lock(_restart_mutex);
                restart_time = _pending_restart_time;
                _pending_restart_time = uhd::time_spec_t(0.0);
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
        uhd::set_thread_priority_safe();
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
        uhd::set_thread_priority_safe();
        bind_current_thread_from_uplink_hint(_link_cfg, 2);
        SPSCBackoff backoff;
        while (_running.load(std::memory_order_relaxed) || !_llr_queue.empty()) {
            UplinkLlrFrame* frame = _llr_queue.consumer_slot();
            if (frame == nullptr) {
                backoff.pause();
                continue;
            }
            backoff.reset();
            if (frame->generation == _stream_generation.load(std::memory_order_acquire)) {
                _decode_llr_stream(frame->llr);
            }
            _llr_queue.consumer_pop();
        }
    }

    void _process_frame(const AlignedVector& frame, uint64_t generation) {
        const std::complex<float>* win = frame.data() + _window_offset;
        const size_t sym_len = _cfg.ofdm.fft_size + _cfg.ofdm.cp_length;

        // 1) Channel estimate from the sync symbol (symbol 0 of the UL frame).
        _fft_symbol(win + /*sym0*/ 0 * sym_len + _cfg.ofdm.cp_length, _sync_freq_buf);
        ChannelEstimator::estimate_from_sync_ls(_sync_freq_buf, _zc_freq, _h_est);

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

        // 4) Equalizer inverse. Honors equalizer_mode (ZF vs MMSE); MMSE uses a
        //    received-domain noise variance from the sync impulse response.
        if (_use_mmse) {
            ChannelEstimator::compute_mmse_inverse(
                _h_est, _h_inv, _estimate_mmse_noise_var(),
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
        if (_layout.payload_re_count > 0) {
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
    void _decode_llr_stream(const LDPCCodec::AlignedFloatVector& frame_llr) {
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
            LDPCCodec::AlignedFloatVector& payload_llr = _payload_llr_scratch;
            payload_llr.resize(required);
            std::copy(frame_llr.begin() + payload_off,
                      frame_llr.begin() + payload_off + required, payload_llr.begin());
            _bit_interleaver->deinterleave_inplace(payload_llr, _deint_scratch);
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
                _udp_out->send(reinterpret_cast<const uint8_t*>(decoded.data()), hdr.payload_len);
                const uint64_t decoded_count = _decoded_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if ((decoded_count & 0xFFu) == 1) {
                    LOG_G_INFO() << "[UL-RX] decoded uplink packets: " << decoded_count
                                 << " (latest seq=" << hdr.seq << ", len=" << hdr.payload_len << ")";
                }
            }
            symbol_offset = next_off;
        }
    }

    void _run_agc(const AlignedVector& frame) {
        if (!_rx_agc.enabled() || !_apply_gain) return;
        // Delay spectrum (impulse response) from the channel estimate.
        std::memcpy(_ce_in.data(), _h_est.data(), _cfg.ofdm.fft_size * sizeof(std::complex<float>));
        fftwf_execute(_ce_ifft_plan);
        float peak = 0.0f;
        double sum = 0.0;
        const auto* __restrict__ ce = _ce_out.data();
        #pragma omp simd simdlen(16) reduction(max:peak) reduction(+:sum)
        for (size_t i = 0; i < _cfg.ofdm.fft_size; ++i) {
            const float re = ce[i].real();
            const float im = ce[i].imag();
            const float m = std::sqrt(re * re + im * im);
            peak = std::max(peak, m);
            sum += m;
        }
        const float avg = static_cast<float>(sum / static_cast<double>(_cfg.ofdm.fft_size));
        const std::complex<float>* sync_time = frame.data() + _window_offset + _cfg.ofdm.cp_length;
        _rx_agc.maybe_apply_from_delay_peak(
            peak, avg, sync_time, _cfg.ofdm.fft_size, _host_now_ns(), _agc_gates,
            [this](double g) { if (_apply_gain) _apply_gain(g); });
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

        if (_self_channel_debug_sink && !_self_h_est.empty()) {
            _self_channel_debug_sink(AlignedVector(_self_h_est.begin(), _self_h_est.end()));
        }

        const bool need_delay = static_cast<bool>(_delay_debug_sink);
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
            _delay_debug_sink(AlignedVector(_delay_spectrum.begin(), _delay_spectrum.end()));
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

    AlignedVector _sync_freq_buf, _h_est, _h_inv;
    AlignedVector _self_h_est;
    std::vector<AlignedVector> _data_symbols;
    std::vector<int> _payload_subcarrier_indices_flat;

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

    LDPCCodec _ldpc{make_ldpc_5041008_cfg()};
    std::unique_ptr<BitBlockInterleaver> _bit_interleaver;
    LDPCCodec::AlignedFloatVector _deint_scratch;
    LDPCCodec::AlignedFloatVector _payload_llr_scratch;   // reused per-packet LDPC input
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

    uhd::rx_streamer::sptr _rx_stream;
    std::atomic<int32_t> _dl_ul_timing_diff{0};
    std::mutex _restart_mutex;
    uhd::time_spec_t _pending_restart_time{0.0};
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
