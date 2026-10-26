#ifndef UPLINK_TX_ENGINE_HPP
#define UPLINK_TX_ENGINE_HPP

// UplinkTxEngine — focused uplink (UE->BS) OFDM transmit engine.
//
// This is the matched TX half of the duplex uplink. It builds a self-contained
// compact OFDM frame (ZC sync at symbol 0, comb pilots, LDPC-coded payload) using
// the uplink config derived by make_uplink_config(), then places that frame at
// the uplink symbol window of the downlink frame period and streams it. The UE
// can gate the whole uplink waveform to zero until downlink synchronization is locked.
//
// It deliberately reuses only the pure DSP primitives from OFDMCore.hpp
// (QPSKModulator, generate_zc_freq, LdpcPacketFraming, Scrambler,
// BitBlockInterleaver) + LDPCCodec — no sensing / AGC / OCXO / control coupling.
// The uplink runs no active clock/timing loop: it rides the shared radio clock,
// with alignment provided passively by the host (timing advance / frame shift).
//
// Payload source: a UDP input socket (mirrors the BS udp_input_* path).

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <fftw3.h>
#include "RadioBackend.hpp"

#include "AirFrameTxGate.hpp"
#include "Common.hpp"
#include "OFDMCore.hpp"
#include "LDPCCodec.hpp"

class UplinkTxEngine {
private:
    struct QueuedUlPeriod {
        AlignedVector samples;
        AirFrameIdentity identity;
        uint64_t generation = 0;
    };

public:
    explicit UplinkTxEngine(const Config& link_cfg)
        : _link_cfg(link_cfg),
          _cfg(make_uplink_config(link_cfg)),
          _duplex(build_duplex_frame_layout(link_cfg)),
          _layout(build_data_resource_grid_layout(_cfg)),
          _fft_in(_cfg.ofdm.fft_size),
          _fft_out(_cfg.ofdm.fft_size),
          _period_samples(link_cfg.samples_per_frame()),
          _air_frame_gate(_period_samples, link_cfg.ofdm.num_symbols),
          _window_offset(_duplex.ul_sample_offset(link_cfg)),
          _window_samples(_duplex.ul_sample_count(link_cfg)) {
        if (_cfg.ofdm.num_symbols < 2) {
            throw std::runtime_error("UplinkTxEngine: uplink needs >= 2 OFDM symbols (1 sync + data).");
        }
        if (_window_samples != _cfg.samples_per_frame()) {
            throw std::runtime_error("UplinkTxEngine: uplink window samples != uplink frame samples.");
        }
        if (_layout.payload_re_count < LdpcPacketFraming::kIdentityControlSymbols) {
            throw std::runtime_error(
                "UplinkTxEngine: payload_re_count (" +
                std::to_string(_layout.payload_re_count) +
                ") < kIdentityControlSymbols (" +
                std::to_string(LdpcPacketFraming::kIdentityControlSymbols) +
                "); increase uplink data resources so the Identity SIB fits");
        }
        _zc_seq = generate_zc_freq(_cfg.ofdm.fft_size, _cfg.ofdm.zc_root);
        _bit_interleaver = std::make_unique<BitBlockInterleaver>(_ldpc.get_N(), 21);
        // Downlink ARQ uses the uplink only as an ACK feedback carrier. Uplink
        // user-data ARQ will get its own config namespace when added.
        _arq_feedback_enabled = link_cfg.network_output.arq_enabled;
        _arq_enabled = link_cfg.uplink_arq.arq_enabled;
        if (_arq_enabled) {
            _ul_arq_tx.configure(link_cfg.uplink_arq);
            _ul_arq_tx.set_direction(1); // uplink direction
        }
        _build_symbol_templates();
        _ifft_plan = fftwf_plan_dft_1d(
            static_cast<int>(_cfg.ofdm.fft_size),
            reinterpret_cast<fftwf_complex*>(_fft_in.data()),
            reinterpret_cast<fftwf_complex*>(_fft_out.data()),
            FFTW_BACKWARD, FFTW_MEASURE);
        // Deeper queue than the historical depth-2 design: absorb modulation
        // jitter so the TX thread can keep the USRP continuously fed (mirrors
        // BS _circular_buffer prefill philosophy).
        _period_queue.reset(kPeriodQueueDepth, [this]() {
            QueuedUlPeriod period;
            period.samples.resize(_period_samples);
            return period;
        });
        _tx_frame_buffer.assign(_period_samples * 2, std::complex<float>(0.0f, 0.0f));
        _blank_period_frame.assign(_period_samples, std::complex<float>(0.0f, 0.0f));
    }

    ~UplinkTxEngine() {
        stop();
        if (_ifft_plan) {
            fftwf_destroy_plan(_ifft_plan);
            _ifft_plan = nullptr;
        }
    }

    UplinkTxEngine(const UplinkTxEngine&) = delete;
    UplinkTxEngine& operator=(const UplinkTxEngine&) = delete;

    void set_tx_stream(radio::ITxStreamPtr stream) { _tx_stream = std::move(stream); }
    radio::ITxStreamPtr tx_stream() const { return _tx_stream; }

    // Schedule transmission on the radio clock, including the simulator's
    // absolute sample clock. Simulator pacing still comes from its shm ring.
    void set_timed_tx(radio::TimeSpec start_time, double tick_rate, double tx_sample_rate) {
        std::lock_guard<std::mutex> lock(_timing_mutex);
        _start_time = start_time;
        _tick_rate = tick_rate;
        _tx_sample_rate = tx_sample_rate > 0.0 ? tx_sample_rate : _link_cfg.rf_sampling.sample_rate;
        _use_timed_tx = (tick_rate > 0.0);
        _air_frame_gate.configure_timing(_use_timed_tx, _tick_rate, _tx_sample_rate);
        _scheduled_restart_epoch.store(0, std::memory_order_relaxed);
        _restart_requested.store(false, std::memory_order_release);
    }

    void reschedule_timed_tx(radio::TimeSpec start_time, uint64_t restart_epoch = 0) {
        {
            std::lock_guard<std::mutex> lock(_timing_mutex);
            _start_time = start_time;
        }
        _scheduled_restart_epoch.store(restart_epoch, std::memory_order_release);
        _restart_requested.store(true, std::memory_order_release);
    }

    void request_restart() {
        _restart_requested.store(true, std::memory_order_release);
    }

    bool wait_for_restart_submitted(
        uint64_t restart_epoch,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(750)) const
    {
        if (restart_epoch == 0) {
            return true;
        }
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (_running.load(std::memory_order_relaxed) &&
               std::chrono::steady_clock::now() < deadline) {
            if (_submitted_restart_epoch.load(std::memory_order_acquire) >= restart_epoch) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        return _submitted_restart_epoch.load(std::memory_order_acquire) >= restart_epoch;
    }

    // Runtime-adjustable Timing Advance (samples). Pulls the uplink TX window
    // earlier on the shared clock so it lands aligned at the BS.
    std::atomic<int32_t>& timing_advance() { return _timing_advance; }
    // Relative shift tracking the UE RX alignment (Phase 5).
    std::atomic<int32_t>& rx_alignment_shift() { return _rx_alignment_shift; }
    std::atomic<uint64_t>& tx_error_count() { return _tx_error_count; }

    void store_rx_alignment_shift(
        int32_t shift_samples,
        uint64_t alignment_id,
        uint64_t restart_epoch)
    {
        _rx_alignment_update_id.store(alignment_id, std::memory_order_relaxed);
        _rx_alignment_restart_epoch.store(restart_epoch, std::memory_order_relaxed);
        _rx_alignment_shift.store(shift_samples, std::memory_order_release);
    }

    void reset_rx_alignment_shift(uint64_t restart_epoch = 0) {
        store_rx_alignment_shift(0, 0, restart_epoch);
    }

    /**
     * @brief Query UL-TX timing state effective at absolute air frame @p frame_seq.
     *
     * Returns the newest entry in the same OTA epoch with
     * @c effective_from_frame <= frame_seq. Fails when history does not cover
     * the requested frame (including ring overwrite of still-referenced states).
     */
    bool query_ul_tx_timing_state(
        uint32_t timeline_epoch,
        uint64_t frame_seq,
        UlTxTimingStateEntry& out) const
    {
        if (timeline_epoch == 0) {
            return false;
        }
        // Plain POD history entries are not atomic. Hold the same mutex used by
        // publish/clear so concurrent readers never race ordinary stores (a
        // version/seqlock around non-atomic fields is still a C++ data race).
        std::lock_guard<std::mutex> lock(_timing_history_mutex);
        if (_timing_history_count == 0) {
            return false;
        }
        const size_t available = std::min(_timing_history_count, kUlTxTimingHistoryCapacity);
        for (size_t i = 0; i < available; ++i) {
            const size_t idx =
                (_timing_history_count - 1u - i) % kUlTxTimingHistoryCapacity;
            const UlTxTimingStateEntry& entry = _timing_history[idx];
            if (entry.timeline_epoch != timeline_epoch) {
                continue;
            }
            if (entry.effective_from_frame <= frame_seq) {
                out = entry;
                return true;
            }
        }
        return false;
    }

    /** Drop all published UL-TX timing history (epoch switch / stream restart). */
    void clear_ul_tx_timing_history() {
        std::lock_guard<std::mutex> lock(_timing_history_mutex);
        // Count is the publication boundary; old POD slots are unreachable once
        // it becomes zero and are overwritten by later publishes.
        _timing_history_count = 0;
        _timing_history_has_last = false;
        _timing_history_last = UlTxTimingStateEntry{};
        _timing_history_fast_valid.store(false, std::memory_order_release);
    }

    /**
     * @brief Publish the latest DL air-frame stamp (absolute seq + UE-USRP
     * frame-start time) so UL-TX / duplex processing can correlate with the
     * concurrent downlink frame.
     */
    void store_air_frame_stamp(const AirFrameIdentity& id) {
        const AirFrameTxGate::StampUpdate update =
            _air_frame_gate.store_stamp(id);
        if (update.generation_bumped) {
            LOG_RT_WARN_M(AirFrame)
                << "[AirFrame][UL-TX] period_generation bump on stamp"
                << ": gen=" << update.generation
                << ", valid=" << update.valid
                << ", epoch_changed=" << update.epoch_changed
                << ", validity_changed=" << update.validity_changed
                << ", epoch=0x" << std::hex
                << (update.valid ? id.timeline_epoch : 0) << std::dec
                << ", stamp_seq=" << (update.valid ? id.frame_seq : 0)
                << ", next_claim_seq=" << update.next_claim_seq;
        } else if (update.valid) {
            const uint64_t n =
                _stamp_update_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n <= 8 || (n % 512) == 0) {
                LOG_RT_INFO_M(AirFrame)
                    << "[AirFrame][UL-TX] stamp update (no gen bump)"
                    << ": n=" << n
                    << ", epoch=0x" << std::hex << id.timeline_epoch << std::dec
                    << ", stamp_seq=" << id.frame_seq
                    << ", next_claim_seq=" << update.next_claim_seq
                    << ", period_gen=" << update.generation;
            }
        }
        // Same-epoch lag never invalidates the whole queue. TX drops stale
        // periods and advances the claim cursor after a blank slot.
    }

    bool air_frame_valid() const {
        return _air_frame_gate.stamp().valid;
    }
    uint64_t air_frame_seq() const {
        return _air_frame_gate.stamp().frame_seq;
    }
    uint64_t air_frame_epoch() const {
        return _air_frame_gate.stamp().timeline_epoch;
    }
    int64_t ue_air_frame_start_time_ns() const {
        return _air_frame_gate.stamp().ue_frame_start_time_ns;
    }
    /** Preferred robust UE-USRP stamp (mid-frame). */
    int64_t ue_air_frame_mid_time_ns() const {
        return _air_frame_gate.stamp().ue_frame_mid_time_ns;
    }

    bool set_waveform_enabled(bool enabled) {
        return _waveform_enabled.exchange(enabled, std::memory_order_acq_rel) != enabled;
    }
    bool waveform_enabled() const {
        return _waveform_enabled.load(std::memory_order_acquire);
    }

    const Config& uplink_config() const { return _cfg; }

    // ARQ: access the uplink TX ARQ window for ACK processing from the DL RX side.
    ArqTxWindow& arq_tx_window() { return _ul_arq_tx; }
    bool arq_enabled() const { return _arq_enabled; }
    bool arq_feedback_enabled() const { return _arq_feedback_enabled; }
    uint64_t arq_feedback_injected_count() const {
        return _arq_feedback_injected.load(std::memory_order_relaxed);
    }
    uint64_t arq_feedback_dropped_disabled_count() const {
        return _arq_feedback_dropped_disabled.load(std::memory_order_relaxed);
    }
    uint64_t arq_feedback_drained_count() const {
        return _arq_feedback_drained.load(std::memory_order_relaxed);
    }
    size_t arq_feedback_pending_count() const {
        std::lock_guard<std::mutex> lock(_arq_feedback_mutex);
        return _arq_pending_feedback.size();
    }

    // ARQ: inject a feedback packet for transmission. Called from the downlink
    // decode thread, so we must NOT touch the SPSC _packet_buffer here (the
    // uplink encode thread is its sole producer). Instead queue the feedback
    // payload and let _ldpc_encode_proc drain it, mirroring the BS-side pattern.
    bool inject_arq_feedback(const ArqFeedback& feedback) {
        if (!_arq_feedback_enabled) {
            const uint64_t dropped =
                _arq_feedback_dropped_disabled.fetch_add(1, std::memory_order_relaxed) + 1;
            if (dropped <= 20 || (dropped % 100) == 0) {
                LOG_G_WARN_M(ArqUl) << "[UL-TX ARQ] feedback disabled; dropping ACK"
                             << ": dropped=" << dropped;
            }
            return false;
        }
        std::vector<uint8_t> fb_payload;
        feedback.pack(fb_payload);
        std::lock_guard<std::mutex> lock(_arq_feedback_mutex);
        _arq_pending_feedback.push_back(std::move(fb_payload));
        _arq_feedback_injected.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void start() {
        if (!_tx_stream) {
            throw std::runtime_error("UplinkTxEngine::start without a TX stream.");
        }
        _waveform_enabled.store(false, std::memory_order_release);
        _running.store(true);
        // Prefill the entire period queue before TX starts so the first timed
        // bursts are not starved while modulation warms up.
        const size_t prefilled = _prefill_period_queue();
        LOG_G_INFO_M(UlTx) << "[UL-TX] period queue prefilled "
                           << prefilled << "/" << _period_queue.capacity()
                           << " frames before stream start";
        if (prefilled < _period_queue.capacity()) {
            LOG_G_WARN_M(UlTx) << "[UL-TX] period queue not fully prefilled; "
                               << "TX may inject blank periods until modulation catches up";
        }
        _ldpc_encode_thread = std::thread(&UplinkTxEngine::_ldpc_encode_proc, this);
        _udp_recv_thread = std::thread(&UplinkTxEngine::_udp_recv_proc, this);
        _mod_thread = std::thread(&UplinkTxEngine::_mod_proc, this);
        _tx_thread = std::thread(&UplinkTxEngine::_tx_proc, this);
    }

    void stop() {
        _running.store(false);
        const int udp_sock = _udp_sock.load(std::memory_order_acquire);
        if (udp_sock >= 0) {
            ::shutdown(udp_sock, SHUT_RDWR);
        }
        if (_udp_recv_thread.joinable()) _udp_recv_thread.join();
        if (_ldpc_encode_thread.joinable()) _ldpc_encode_thread.join();
        if (_mod_thread.joinable()) _mod_thread.join();
        if (_tx_thread.joinable()) _tx_thread.join();

        {
            std::lock_guard<std::mutex> lock(_arq_feedback_mutex);
            _arq_pending_feedback.clear();
        }
        _raw_udp_buffer.clear();
        _packet_buffer.clear();
        _period_queue.clear();
    }

private:
    // ---- One-time setup: build per-symbol frequency templates + payload map ----
    static int _template_payload_qpsk(size_t payload_rank) {
        return static_cast<int>(
            splitmix32(0x554C544Du ^ static_cast<uint32_t>(payload_rank)) & 0x3u);
    }

    void _build_symbol_templates() {
        _symbol_templates.assign(_cfg.ofdm.num_symbols, AlignedVector(_cfg.ofdm.fft_size, {0.0f, 0.0f}));
        _payload_subcarrier_indices_flat.clear();
        _payload_subcarrier_indices_flat.reserve(_layout.payload_re_count);

        size_t data_symbol_idx = 0;
        for (size_t sym = 0; sym < _cfg.ofdm.num_symbols; ++sym) {
            AlignedVector& tmpl = _symbol_templates[sym];
            if (is_zc_sync_symbol(_cfg, sym)) {
                std::memcpy(tmpl.data(), _zc_seq.data(), _cfg.ofdm.fft_size * sizeof(std::complex<float>));
                continue;
            }
            const size_t base = _layout.non_pilot_offsets[data_symbol_idx];
            for (size_t di = 0; di < _layout.num_non_pilot_subcarriers; ++di) {
                const size_t k = static_cast<size_t>(_layout.non_pilot_subcarrier_indices[di]);
                const int payload_rank = _layout.payload_rank[base + di];
                if (payload_rank >= 0) {
                    // Payload RE: prefilled like downlink, then overwritten by
                    // per-frame UDP payload symbols where available.
                    _payload_subcarrier_indices_flat.push_back(static_cast<int>(k));
                    if (_link_cfg.uplink.idle_waveform == kUplinkIdleWaveformRandomQpsk) {
                        tmpl[k] = _qpsk_from_int(_template_payload_qpsk(static_cast<size_t>(payload_rank)));
                    }
                } else {
                    // deterministic unit-magnitude filler (RX ignores these REs).
                    tmpl[k] = _zc_seq[k];
                }
            }
            for (const size_t k : _cfg.ofdm.pilot_positions) {
                if (k < _cfg.ofdm.fft_size) tmpl[k] = _zc_seq[k];
            }
            ++data_symbol_idx;
        }
        if (_payload_subcarrier_indices_flat.size() != _layout.payload_re_count) {
            throw std::runtime_error("UplinkTxEngine: payload RE flatten mismatch.");
        }
    }

    std::complex<float> _qpsk_from_int(int sym) const {
        return _payload_modulation.modulate_token(sym);
    }

    size_t _effective_user_payload_re() const {
        return AirFrameClock::effective_payload_re_count(
            _layout.payload_re_count,
            LdpcPacketFraming::kIdentityControlSymbols);
    }

    /**
     * @brief Snapshot DL air-frame stamp and write Tier-1 Identity SIB at the
     * head of the UL payload (frozen into this period's samples).
     * @return QPSK symbols written (0 if capacity insufficient).
     */
    size_t _inject_identity_sib(
        AlignedIntVector& data_pool,
        const AirFrameIdentity& identity)
    {
        const size_t sib = LdpcPacketFraming::kIdentityControlSymbols;
        const size_t begin = data_pool.size();
        data_pool.resize(begin + sib);
        // Always embed a structured SIB so BS can double-check closed loop.
        // When unlock/invalid, epoch=0 and seq=0 is an explicit "no lock" marker.
        const uint32_t ota_epoch = identity.valid
            ? AirFrameClock::ota_epoch32(identity.timeline_epoch)
            : 0u;
        const uint32_t ota_seq = identity.valid
            ? AirFrameClock::ota_frame_seq32(identity.frame_seq)
            : 0u;
        LdpcPacketFraming::write_identity_control_qpsk(
            ota_epoch, ota_seq, data_pool.data() + begin);
        if (identity.valid) {
            _identity_sib_injected.fetch_add(1, std::memory_order_relaxed);
        } else {
            _identity_sib_unlocked.fetch_add(1, std::memory_order_relaxed);
        }
        return sib;
    }

    void _append_idle_symbols(AlignedIntVector& data_pool) {
        // Fill remaining RE after Identity SIB (already at head when present).
        const size_t room = (_layout.payload_re_count > data_pool.size())
            ? (_layout.payload_re_count - data_pool.size())
            : 0;
        if (room < LdpcPacketFraming::kControlSymbols) {
            // Not enough room for a control packet; pad random QPSK if configured.
            if (_link_cfg.uplink.idle_waveform == kUplinkIdleWaveformRandomQpsk) {
                const uint32_t idle_seq =
                    _idle_seq.fetch_add(1, std::memory_order_relaxed);
                while (data_pool.size() < _layout.payload_re_count) {
                    const uint32_t rnd = splitmix32(
                        0x554C4944u ^ idle_seq ^
                        static_cast<uint32_t>(data_pool.size()));
                    data_pool.push_back(static_cast<int>(rnd & 0x3u));
                }
            }
            return;
        }
        const uint32_t idle_seq = _idle_seq.fetch_add(1, std::memory_order_relaxed);
        const LdpcMiniHeader hdr{
            LdpcPacketFraming::kVersion,
            LdpcPacketFraming::kFlags,
            0,
            0,
            static_cast<uint16_t>(idle_seq & 0xFFFFu),
        };
        const size_t begin = data_pool.size();
        data_pool.resize(begin + LdpcPacketFraming::kControlSymbols);
        LdpcPacketFraming::write_control_qpsk(hdr, data_pool.data() + begin);

        if (_link_cfg.uplink.idle_waveform != kUplinkIdleWaveformRandomQpsk) {
            return;
        }
        data_pool.reserve(_layout.payload_re_count);
        while (data_pool.size() < _layout.payload_re_count) {
            const uint32_t rnd = splitmix32(0x554C4944u ^ idle_seq ^ static_cast<uint32_t>(data_pool.size()));
            data_pool.push_back(static_cast<int>(rnd & 0x3u));
        }
    }

    // ---- LDPC encode: raw UDP packet queue -> encoded packet queue ----
    void _ldpc_encode_proc() {
        bind_current_thread_from_uplink_hint(_link_cfg, 0);
        std::vector<uint16_t> retransmit_seqs;
        std::deque<std::vector<uint8_t>> pending_fb;
        while (_running.load(std::memory_order_relaxed)) {
            // ARQ: drain pending feedback (queued by the DL decode thread) and
            // encode it here on the single producer thread. Highest priority.
            if (_arq_feedback_enabled) {
                {
                    std::lock_guard<std::mutex> lock(_arq_feedback_mutex);
                    pending_fb.swap(_arq_pending_feedback);
                }
                for (auto& fb : pending_fb) {
                    if (_encode_and_enqueue_payload(fb.data(), fb.size(), false, true)) {
                        _arq_feedback_drained.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                pending_fb.clear();
            }

            // ARQ: check for retransmissions first (priority over new data)
            if (_arq_enabled) {
                const int64_t now = arq_now_ms();
                const size_t abandoned = _ul_arq_tx.drop_abandoned(now);
                if (abandoned > 0) {
                    LOG_G_WARN_M(ArqUl) << "[UL-TX ARQ] max retries exceeded; dropping "
                                 << abandoned << " unacknowledged packets";
                }
                _ul_arq_tx.get_retransmit(retransmit_seqs, now);
                for (uint16_t seq : retransmit_seqs) {
                    ArqTxEntry entry;
                    if (!_ul_arq_tx.get_entry_copy(seq, entry)) {
                        continue;
                    }
                    if (!entry.encoded_qpsk.empty()) {
                        // Cheap retransmit: push pre-encoded symbol tokens directly
                        AlignedIntVector retrans(entry.encoded_qpsk);
                        SPSCBackoff bo;
                        bool enqueued = false;
                        while (_running.load(std::memory_order_relaxed)) {
                            if (_packet_buffer.try_push(std::move(retrans))) {
                                enqueued = true;
                                break;
                            }
                            bo.pause();
                        }
                        if (!enqueued) {
                            break;
                        }
                        _ul_arq_tx.mark_transmitted(seq, now);
                    } else {
                        // Re-encode from raw payload
                        if (!_encode_and_enqueue_payload(
                                entry.raw_payload.data(), entry.raw_payload.size(), false)) {
                            LOG_G_WARN_M(ArqUl) << "[UL-TX ARQ] retransmission was not queued"
                                         << ": seq=" << seq;
                            break;
                        }
                        _ul_arq_tx.mark_transmitted(seq, now);
                    }
                }
            }

            RawUdpPacket* pkt = nullptr;
            // Only read new data if no pending retransmissions (backpressure)
            if (!_arq_enabled || !_ul_arq_tx.has_retransmit_pending(arq_now_ms())) {
                while (_running.load(std::memory_order_relaxed)) {
                    pkt = _raw_udp_buffer.consumer_slot();
                    if (pkt != nullptr) break;
                    if (_arq_enabled) break;
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
            }
            if (!_running.load(std::memory_order_relaxed)) break;
            if (pkt == nullptr) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            // ARQ: check window room before encoding new data
            if (_arq_enabled && !_ul_arq_tx.has_room()) {
                // Window full, drop new packet with backpressure
                static std::atomic<uint64_t> arq_drop_count{0};
                const uint64_t dc = arq_drop_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (dc <= 20 || (dc % 100) == 0) {
                    LOG_G_WARN_M(ArqUl) << "[UL-TX ARQ] TX window full, dropping new packet: dropped=" << dc;
                }
                _raw_udp_buffer.consumer_pop();
                continue;
            }
            _encode_and_enqueue_payload(pkt->bytes.data(), pkt->bytes.size());
            _raw_udp_buffer.consumer_pop();
        }
    }

    void _udp_recv_proc() {
        bind_current_thread_from_uplink_hint(_link_cfg, 3);
        const int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) {
            LOG_G_ERROR_M(UlTx) << "[UL-TX] UDP socket create failed";
            return;
        }
        _udp_sock.store(sock, std::memory_order_release);

        auto close_sock = [&]() {
            int expected = sock;
            if (_udp_sock.compare_exchange_strong(expected, -1, std::memory_order_acq_rel)) {
                ::close(sock);
            }
        };

        int enable = 1;
        ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
        constexpr int rcvbuf_size = 4 * 1024 * 1024;
        ::setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf_size, sizeof(rcvbuf_size));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(_link_cfg.network_output.ul_udp_input_port));
        if (_link_cfg.network_output.ul_udp_input_ip == "0.0.0.0") {
            addr.sin_addr.s_addr = INADDR_ANY;
        } else if (inet_pton(AF_INET, _link_cfg.network_output.ul_udp_input_ip.c_str(), &addr.sin_addr) != 1) {
            LOG_G_ERROR_M(UlTx) << "[UL-TX] Invalid ul_udp_input_ip: " << _link_cfg.network_output.ul_udp_input_ip;
            close_sock();
            return;
        }
        if (::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            LOG_G_ERROR_M(UlTx) << "[UL-TX] UDP bind failed on " << _link_cfg.network_output.ul_udp_input_ip << ":"
                          << _link_cfg.network_output.ul_udp_input_port;
            close_sock();
            return;
        }
        ::fcntl(sock, F_SETFL, O_NONBLOCK);
        const bool overflow_reporting = enable_udp_rx_overflow_reporting(sock);
        if (!overflow_reporting) {
            LOG_G_WARN_M(UlTx) << "[UL-TX] UDP input could not enable kernel overflow reporting; "
                          << "socket-buffer drops may be unobservable";
        }
        if (LOG_MOD_ON(UlTx)) {
            LOG_G_INFO_M(UlTx) << "[UL-TX] uplink payload UDP input on " << _link_cfg.network_output.ul_udp_input_ip << ":"
                         << _link_cfg.network_output.ul_udp_input_port;
        }

        std::vector<uint8_t> buf(65536);
        uint32_t last_kernel_drop_count = 0;
        while (_running.load(std::memory_order_relaxed)) {
            const UdpDatagramReceiveResult received =
                receive_udp_datagram(sock, buf.data(), buf.size());
            if (received.bytes < 0) {
                if (!_running.load(std::memory_order_relaxed)) break;
                if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                    LOG_G_WARN_M(UlTx) << "[UL-TX] UDP input receive failed: "
                                  << std::strerror(errno);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            if (received.has_kernel_drop_count) {
                const uint32_t dropped = received.kernel_drop_count - last_kernel_drop_count;
                last_kernel_drop_count = received.kernel_drop_count;
                if (dropped > 0) {
                    LOG_G_WARN_M(UlTx) << "[UL-TX] UDP socket receive buffer overflow; kernel dropped "
                                  << dropped << " datagrams"
                                  << ": cumulative=" << received.kernel_drop_count;
                }
            }
            if (received.truncated) {
                LOG_G_WARN_M(UlTx) << "[UL-TX] UDP input datagram exceeded receive buffer; dropping truncated packet"
                              << ": copied_bytes=" << received.bytes
                              << ", buffer_bytes=" << buf.size();
                continue;
            }
            const ssize_t n = received.bytes;
            if (n == 0) {
                LOG_RT_WARN_HZ_M(UlTx, 1)
                    << "[UL-TX] UDP input received an empty datagram; dropping packet";
                continue;
            }
            RawUdpPacket* pkt = _raw_udp_buffer.producer_slot();
            if (pkt == nullptr) {
                static std::atomic<uint64_t> drop_count{0};
                const uint64_t dc = drop_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (dc <= 20 || (dc % 100) == 0) {
                    LOG_G_WARN_M(UlTx) << "[UL-TX] UDP recv intermediate buffer full, dropping packet: dropped=" << dc;
                }
                continue;
            }
            pkt->bytes.resize(static_cast<size_t>(n));
            std::memcpy(pkt->bytes.data(), buf.data(), static_cast<size_t>(n));
            _raw_udp_buffer.producer_commit();
        }
        close_sock();
    }

    bool _encode_and_enqueue_payload(const uint8_t* data, size_t len, bool track_arq = true,
                                     bool is_feedback = false) {
        const size_t bytes_per_block = (_ldpc.get_K() + 7) / 8;
        if (bytes_per_block != LdpcPacketFraming::kLdpcInfoBytesPerBlock ||
            _ldpc.get_N() != LdpcPacketFraming::kLdpcCodeBitsPerBlock) {
            throw std::runtime_error("UplinkTxEngine: LDPC dimensions do not match framing.");
        }
        if (!LdpcPacketFraming::payload_len_fits(len)) {
            static std::atomic<uint64_t> oversize_drop_count{0};
            const uint64_t dropped =
                oversize_drop_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (dropped <= 20 || (dropped % 100) == 0) {
                LOG_G_WARN_M(UlTx) << "[UL-TX] payload exceeds mini-header limit; dropping"
                            << ": payload_bytes=" << len << ", dropped=" << dropped;
            }
            return false;
        }
        const size_t payload_blocks = LdpcPacketFraming::payload_blocks_for_len(len);
        const size_t packet_qpsk = LdpcPacketFraming::packet_symbols(payload_blocks, _payload_modulation, _payload_rate.code_bits());
        // Capacity after per-period Identity SIB reservation (mirrors BS DL).
        const size_t effective_re = _effective_user_payload_re();
        if (packet_qpsk > effective_re) {
            LOG_RT_WARN_HZ_M(UlTx, 2) << "[UL-TX] dropping payload: " << packet_qpsk
                              << " qpsk syms > effective capacity " << effective_re
                              << " (raw=" << _layout.payload_re_count
                              << ", identity_sib="
                              << LdpcPacketFraming::kIdentityControlSymbols << ")";
            return false;
        }

        AlignedIntVector packet;
        packet.resize(LdpcPacketFraming::kControlSymbols);
        const uint16_t frame_seq = is_feedback
            ? static_cast<uint16_t>(_arq_feedback_seq.fetch_add(1, std::memory_order_relaxed) & 0xFFFFu)
            : static_cast<uint16_t>(_packet_seq.fetch_add(1, std::memory_order_relaxed) & 0xFFFFu);
        const LdpcMiniHeader hdr{
            LdpcPacketFraming::kVersion,
            static_cast<uint8_t>(is_feedback ? LdpcPacketFraming::kFlagArqFeedback
                                             : LdpcPacketFraming::kFlags),
            static_cast<uint16_t>(len),
            LdpcPacketFraming::payload_blocks_field_for_len(len),
            frame_seq,
        };
        LdpcPacketFraming::write_control_qpsk(hdr, packet.data());

        if (payload_blocks > 0) {
            LDPCCodec::AlignedByteVector input_bytes(payload_blocks * bytes_per_block, 0);
            std::memcpy(input_bytes.data(), data, len);
            LDPCCodec::AlignedIntVector encoded_bits;
            LDPCCodec::AlignedIntVector qpsk_ints;
            _ldpc.encode_frame(input_bytes, encoded_bits);
            _scrambler.scramble(encoded_bits);
            _bit_interleaver->interleave_inplace(encoded_bits, _interleaver_scratch);
            _payload_rate.match_inplace(encoded_bits, _interleaver_scratch);
            _payload_modulation.pack_bits(encoded_bits, qpsk_ints, _payload_rate.code_bits());
            packet.insert(packet.end(), qpsk_ints.begin(), qpsk_ints.end());
        }
        if (packet.size() != packet_qpsk) {
            throw std::runtime_error("UplinkTxEngine: packet symbol count mismatch.");
        }
        // ARQ: store encoded packet in TX window for potential retransmission.
        // Feedback frames are never tracked as data.
        if (_arq_enabled && track_arq && !is_feedback) {
            const int64_t now = arq_now_ms();
            AlignedIntVector arq_copy(packet); // copy before move
            if (!_ul_arq_tx.try_insert_encoded(
                    hdr.seq, data, len, std::move(arq_copy), now)) {
                LOG_G_WARN_M(ArqUl) << "[UL-TX ARQ] failed to track packet in TX window"
                             << ": seq=" << hdr.seq
                             << "; packet will not be retransmittable";
            }
        }
        bool enqueued = false;
        SPSCBackoff backoff;
        while (_running.load(std::memory_order_relaxed)) {
            if (_packet_buffer.try_push(std::move(packet))) {
                enqueued = true;
                break;
            }
            backoff.pause();
        }
        if (!enqueued && !packet.empty()) {
            return false;
        }
        return enqueued;
    }

    // ---- Build one uplink frame's IQ into period_buffer at the UL window ----
    void _build_period_buffer(QueuedUlPeriod& period) {
        const AirFrameTxGate::Claim claim = _air_frame_gate.claim_period();
        period.identity = claim.identity;
        period.generation = claim.generation;
        {
            const uint64_t n =
                _period_claim_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n <= 8 || (n % 512) == 0) {
                LOG_RT_INFO_M(AirFrame)
                    << "[AirFrame][UL-TX] claim period"
                    << ": n=" << n
                    << ", valid=" << period.identity.valid
                    << ", epoch=0x" << std::hex << period.identity.timeline_epoch
                    << std::dec
                    << ", seq=" << period.identity.frame_seq
                    << ", period_gen=" << period.generation;
            }
        }
        AlignedVector& period_buffer = period.samples;
        // The period buffers come from the pool zero-initialized and the only
        // region ever written is the UL window [_window_offset, +_window_samples),
        // which the symbol loop below overwrites in full every frame (the ctor
        // asserts _window_samples == uplink samples_per_frame() == num_symbols *
        // sym_len, and the loop writes exactly that many contiguous samples). So
        // the rest of the DL-period buffer stays zero permanently — no per-frame
        // fill over the (large) full period is needed. If a future layout leaves
        // gaps inside the UL window, re-add a window-scoped zeroing here.

        AlignedIntVector& data_pool = _data_pool;
        data_pool.clear();
        data_pool.reserve(_layout.payload_re_count);

        // Freeze the claimed target air slot into this period's samples. The TX
        // hard gate verifies the same identity and generation before sending.
        const size_t sib_syms = _inject_identity_sib(data_pool, period.identity);
        const size_t max_re = _layout.payload_re_count;
        const size_t effective_user_re = (max_re > sib_syms) ? (max_re - sib_syms) : 0;

        // Pull payload packets into the remaining capacity after Identity SIB.
        while (data_pool.size() < max_re) {
            AlignedIntVector* slot = _packet_buffer.consumer_slot();
            if (slot == nullptr) break;
            const size_t room = max_re - data_pool.size();
            if (slot->size() > room) {
                if (slot->size() > effective_user_re) {
                    LOG_RT_WARN_M(UlTx) << "[UL-TX] encoded packet exceeds effective frame capacity; "
                                  << "dropping queue head: qpsk_syms=" << slot->size()
                                  << ", effective_user_re=" << effective_user_re
                                  << ", raw=" << max_re;
                    _packet_buffer.consumer_pop();  // oversized: drop to avoid stall
                }
                break;
            }
            data_pool.insert(data_pool.end(), slot->begin(), slot->end());
            _packet_buffer.consumer_pop();
        }
        // Only SIB (or empty): pad with idle control / random QPSK.
        if (data_pool.size() <= sib_syms) {
            _append_idle_symbols(data_pool);
        }

        _mod_pool.resize(data_pool.size());
        for (size_t i = 0; i < data_pool.size(); ++i) _mod_pool[i] = _qpsk_from_int(data_pool[i]);

        const float scale = 1.0f / std::sqrt(static_cast<float>(_cfg.ofdm.fft_size)) / 4.0f;
        const size_t sym_len = _cfg.ofdm.fft_size + _cfg.ofdm.cp_length;
        size_t data_symbol_idx = 0;
        for (size_t sym = 0; sym < _cfg.ofdm.num_symbols; ++sym) {
            std::memcpy(_fft_in.data(), _symbol_templates[sym].data(),
                        _cfg.ofdm.fft_size * sizeof(std::complex<float>));
            if (!is_zc_sync_symbol(_cfg, sym)) {
                const size_t begin = _layout.payload_offsets[data_symbol_idx];
                const size_t end = _layout.payload_offsets[data_symbol_idx + 1];
                const size_t avail = (begin < _mod_pool.size())
                    ? std::min(end - begin, _mod_pool.size() - begin) : 0;
                for (size_t p = 0; p < avail; ++p) {
                    _fft_in[static_cast<size_t>(_payload_subcarrier_indices_flat[begin + p])] =
                        _mod_pool[begin + p];
                }
                ++data_symbol_idx;
            }
            fftwf_execute(_ifft_plan);
            std::complex<float>* __restrict__ dst =
                period_buffer.data() + _window_offset + sym * sym_len;
            const auto* __restrict__ src = _fft_out.data();
            // Cyclic prefix (tail of the IFFT body) then the symbol body, scaled.
            #pragma omp simd
            for (size_t j = 0; j < _cfg.ofdm.cp_length; ++j) {
                dst[j] = src[_cfg.ofdm.fft_size - _cfg.ofdm.cp_length + j] * scale;
            }
            #pragma omp simd
            for (size_t j = 0; j < _cfg.ofdm.fft_size; ++j) {
                dst[_cfg.ofdm.cp_length + j] = src[j] * scale;
            }
        }
    }

    /**
     * @brief Prefill period queue with fully-built frames (single-threaded producer).
     * @return Number of frames placed into the queue.
     */
    size_t _prefill_period_queue() {
        size_t filled = 0;
        while (QueuedUlPeriod* slot = _period_queue.producer_slot()) {
            _build_period_buffer(*slot);
            _period_queue.producer_commit();
            ++filled;
        }
        return filled;
    }

    void _mod_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        // Below TX priority so IFFT work cannot starve USRP send (mirrors BS).
        radio::set_thread_priority(0.6f, true);
        if (!bind_current_thread_from_uplink_hint(_link_cfg, 1)) {
            LOG_G_WARN_M(UlTx) << "[UL-TX] modulation thread unbound "
                               << "(set uplink_cpu_cores[1] for a dedicated mod core)";
        }
        prefault_thread_stack();
        // period_queue (depth 8) cushions modulation; cooperative backoff OK.
        // Hard-RT is _tx_proc sample send only.
        SPSCBackoff backoff;
        while (_running.load(std::memory_order_relaxed)) {
            QueuedUlPeriod* frame = _period_queue.producer_slot();
            if (frame == nullptr) {
                backoff.pause();
                continue;
            }
            _build_period_buffer(*frame);
            _period_queue.producer_commit();
            backoff.reset();
        }
    }

    void _tx_proc() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::Realtime);
        // Highest RT priority among UL threads — keep the radio continuously fed.
        radio::set_thread_priority(1.0f, true);
        if (!bind_current_thread_from_uplink_hint(_link_cfg, 2)) {
            LOG_G_WARN_M(UlTx) << "[UL-TX] TX send thread unbound "
                               << "(set uplink_cpu_cores[2] to a dedicated isolated core)";
        }
        prefault_thread_stack();
        radio::TxMetadata md;
        md.start_of_burst = true;
        md.end_of_burst = false;

        long long period_ticks = 0;
        long long next_ticks = 0;
        int64_t applied_shift_samples = 0;
        bool first = true;
        uint64_t tx_period_seq = 0;
        uint64_t active_restart_epoch = 0;
        uint32_t recovery_log_periods_remaining = 0;
        int64_t last_logged_target_shift = std::numeric_limits<int64_t>::min();
        uint64_t last_logged_alignment_id = 0;
        if (_use_timed_tx) {
            const double exact = static_cast<double>(_period_samples) * _tick_rate / _tx_sample_rate;
            period_ticks = std::llround(exact);
            std::lock_guard<std::mutex> lock(_timing_mutex);
            next_ticks = _start_time.to_ticks(_tick_rate);
        }

        while (_running.load(std::memory_order_relaxed)) {
            if (_use_timed_tx && _restart_requested.exchange(false, std::memory_order_acq_rel)) {
                radio::TxMetadata eob_md;
                eob_md.start_of_burst = false;
                eob_md.end_of_burst = true;
                eob_md.has_time_spec = false;
                try {
                    _tx_stream->send(&_eob_sample, 0, eob_md, 0.1);
                } catch (const std::exception& e) {
                    if (LOG_MOD_ON(UlTx)) {
                        LOG_RT_WARN_M(UlTx) << "[UL-TX] failed to terminate burst before restart: " << e.what();
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(_timing_mutex);
                    next_ticks = _start_time.to_ticks(_tick_rate);
                }
                applied_shift_samples = 0;
                first = true;
                // Restart invalidates prior absolute-frame timing history.
                clear_ul_tx_timing_history();
                // Generation cancellation also invalidates a producer-held slot
                // that has not reached producer_commit() yet. Queue clear alone
                // cannot cancel that in-flight modulation work.
                const AirFrameIdentity restart_identity =
                    _air_frame_gate.cancel_and_reanchor(
                        next_ticks,
                        _rx_alignment_shift.load(std::memory_order_acquire));
                const size_t dropped_periods = _period_queue.size();
                _period_queue.clear();
                if (dropped_periods > 0) {
                    LOG_RT_WARN_M(UlTx) << "[UL-TX] timed restart dropped "
                                  << dropped_periods << " queued transmit periods";
                }
                active_restart_epoch =
                    _scheduled_restart_epoch.load(std::memory_order_acquire);
                recovery_log_periods_remaining = 8;
                if (LOG_MOD_ON(UlTx)) {
                    LOG_RT_WARN_M(UlTx) << "[UL-TX] timed TX restart scheduled at "
                                  << radio::TimeSpec::from_ticks(next_ticks, _tick_rate).get_real_secs()
                                  << " s, target_epoch=" << restart_identity.timeline_epoch
                                  << ", target_seq=" << restart_identity.frame_seq
                                  << ", identity_valid=" << restart_identity.valid
                                  << ", period_generation="
                                  << _air_frame_gate.generation();
                }
                if (LOG_MOD_ON(Recovery)) {
                    LOG_RT_WARN_M(Recovery) << "[UL-TX recovery] restart_seen restart_epoch="
                                  << active_restart_epoch
                                  << ", next_tx_time="
                                  << radio::TimeSpec::from_ticks(next_ticks, _tick_rate).get_real_secs()
                                  << ", rx_alignment_shift="
                                  << _rx_alignment_shift.load(std::memory_order_acquire)
                                  << ", rx_alignment_id="
                                  << _rx_alignment_update_id.load(std::memory_order_relaxed)
                                  << ", rx_alignment_restart_epoch="
                                  << _rx_alignment_restart_epoch.load(std::memory_order_relaxed)
                                  << ", waveform_enabled="
                                  << _waveform_enabled.load(std::memory_order_acquire);
                }
            }

            uint64_t gate_generation = _air_frame_gate.generation();
            AirFrameIdentity expected_identity = _use_timed_tx
                ? _air_frame_gate.identity_for_grid_ticks(
                      next_ticks,
                      _rx_alignment_shift.load(std::memory_order_acquire))
                : AirFrameIdentity{};
            QueuedUlPeriod* selected_period = nullptr;
            bool held_future_period = false;

            // Hard gate the modulated identity against this nominal TX grid
            // slot. Timing advance shifts the physical samples, but does not
            // change which absolute air-frame period owns the UL window.
            while (_running.load(std::memory_order_relaxed)) {
                QueuedUlPeriod* candidate = _period_queue.consumer_slot();
                if (candidate == nullptr) {
                    break;
                }

                const uint64_t live_generation = _air_frame_gate.generation();
                if (live_generation != gate_generation) {
                    gate_generation = live_generation;
                    expected_identity = _use_timed_tx
                        ? _air_frame_gate.identity_for_grid_ticks(
                              next_ticks,
                              _rx_alignment_shift.load(std::memory_order_acquire))
                        : AirFrameIdentity{};
                    continue;
                }
                const AirFrameTxGate::Decision decision =
                    AirFrameTxGate::classify(
                        candidate->identity,
                        candidate->generation,
                        expected_identity,
                        gate_generation,
                        _use_timed_tx);
                if (decision == AirFrameTxGate::Decision::CancelledGeneration) {
                    LOG_RT_WARN_HZ_M(AirFrame, 20)
                        << "[AirFrame][UL-TX] hard-gate mismatch: cancelled generation"
                        << ": queued_generation=" << candidate->generation
                        << ", live_generation=" << gate_generation
                        << ", queued_epoch=0x" << std::hex
                        << candidate->identity.timeline_epoch << std::dec
                        << ", queued_seq=" << candidate->identity.frame_seq;
                    _period_queue.consumer_pop();
                    continue;
                }

                if (decision == AirFrameTxGate::Decision::InvalidIdentity ||
                    decision == AirFrameTxGate::Decision::EpochMismatch ||
                    decision == AirFrameTxGate::Decision::Stale) {
                    LOG_RT_WARN_HZ_M(AirFrame, 20)
                        << "[AirFrame][UL-TX] hard-gate mismatch: stale/epoch"
                        << ": queued_epoch=0x" << std::hex
                        << candidate->identity.timeline_epoch
                        << ", expected_epoch=0x" << expected_identity.timeline_epoch
                        << std::dec
                        << ", queued_seq=" << candidate->identity.frame_seq
                        << ", expected_seq=" << expected_identity.frame_seq
                        << ", generation=" << gate_generation;
                    _period_queue.consumer_pop();
                    continue;
                }
                if (decision == AirFrameTxGate::Decision::Future) {
                    held_future_period = true;
                    break;
                }

                selected_period = candidate;
                break;
            }

            if (held_future_period) {
                const uint64_t n =
                    _gate_hold_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (n <= 16 || (n % 256) == 0) {
                    QueuedUlPeriod* head = _period_queue.consumer_slot();
                    LOG_RT_INFO_M(AirFrame)
                        << "[AirFrame][UL-TX] hard-gate HOLD future"
                        << ": n=" << n
                        << ", expected_epoch=0x" << std::hex
                        << expected_identity.timeline_epoch << std::dec
                        << ", expected_seq=" << expected_identity.frame_seq
                        << ", queued_seq="
                        << (head ? head->identity.frame_seq : 0)
                        << ", generation=" << gate_generation;
                }
            }

            // Final send commit is serialized with stamp/generation updates.
            // Keep a rejected queue head in place; the next slot iteration will
            // either discard it as stale or match it against the refreshed grid.
            if (selected_period != nullptr) {
                const AirFrameTxGate::GateSnapshot commit =
                    _air_frame_gate.validate_for_send(
                        selected_period->identity,
                        selected_period->generation,
                        next_ticks,
                        _rx_alignment_shift.load(std::memory_order_acquire),
                        _use_timed_tx);
                gate_generation = commit.generation;
                expected_identity = commit.expected;
                if (commit.decision != AirFrameTxGate::Decision::Match) {
                    LOG_RT_WARN_HZ_M(AirFrame, 20)
                        << "[AirFrame][UL-TX] hard-gate mismatch: final-commit reject"
                        << ": decision="
                        << AirFrameTxGate::decision_name(commit.decision)
                        << ", queued_seq=" << selected_period->identity.frame_seq
                        << ", expected_seq=" << commit.expected.frame_seq
                        << ", queued_gen=" << selected_period->generation
                        << ", live_gen=" << commit.generation;
                    selected_period = nullptr;
                }
            }
            const bool has_frame = (selected_period != nullptr);
            const AlignedVector& period_src = has_frame
                ? selected_period->samples
                : _blank_period_frame;
            if (has_frame) {
                const uint64_t n =
                    _gate_match_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (n <= 16 || (n % 512) == 0) {
                    LOG_RT_INFO_M(AirFrame)
                        << "[AirFrame][UL-TX] hard-gate MATCH send"
                        << ": n=" << n
                        << ", epoch=0x" << std::hex
                        << selected_period->identity.timeline_epoch << std::dec
                        << ", seq=" << selected_period->identity.frame_seq
                        << ", period_gen=" << selected_period->generation
                        << ", expected_seq=" << expected_identity.frame_seq
                        << ", matches=" << n
                        << ", holds="
                        << _gate_hold_count.load(std::memory_order_relaxed)
                        << ", blanks="
                        << _gate_blank_count.load(std::memory_order_relaxed)
                        << ", sib_injected="
                        << _identity_sib_injected.load(std::memory_order_relaxed);
                }
            } else {
                const uint64_t n =
                    _gate_blank_count.fetch_add(1, std::memory_order_relaxed) + 1;
                // Abnormal under steady lock — WARN every event (rate-capped for RT).
                LOG_RT_WARN_HZ_M(AirFrame, 50)
                    << "[AirFrame][UL-TX] hard-gate BLANK"
                    << ": n=" << n
                    << (held_future_period ? " (holding future)" : " (empty/stale/mismatch)")
                    << ", expected_valid=" << expected_identity.valid
                    << ", expected_epoch=0x" << std::hex
                    << expected_identity.timeline_epoch << std::dec
                    << ", expected_seq=" << expected_identity.frame_seq
                    << ", generation=" << gate_generation
                    << ", matches="
                    << _gate_match_count.load(std::memory_order_relaxed)
                    << ", holds="
                    << _gate_hold_count.load(std::memory_order_relaxed);
            }

            const int32_t timing_advance_samples =
                _timing_advance.load(std::memory_order_relaxed);
            const int32_t rx_alignment_shift_samples =
                _rx_alignment_shift.load(std::memory_order_acquire);
            const uint64_t rx_alignment_id =
                _rx_alignment_update_id.load(std::memory_order_relaxed);
            const uint64_t rx_alignment_restart_epoch =
                _rx_alignment_restart_epoch.load(std::memory_order_relaxed);
            const int64_t target_shift_samples =
                static_cast<int64_t>(timing_advance_samples) +
                static_cast<int64_t>(rx_alignment_shift_samples);
            const int64_t applied_shift_before = applied_shift_samples;
            const uint64_t period_seq = ++tx_period_seq;

            // Absolute-frame UL-TX timing history for eRTM stale-UL compensation.
            // Timed path: hard-gate expected identity. Untimed path: the
            // selected period's stamped identity (no TX grid tick mapping).
            const AirFrameIdentity& timing_identity =
                (_use_timed_tx || !has_frame)
                    ? expected_identity
                    : selected_period->identity;
            if (timing_identity.valid && timing_identity.timeline_epoch != 0) {
                _maybe_publish_ul_tx_timing_state(
                    static_cast<uint32_t>(
                        AirFrameClock::ota_epoch32(timing_identity.timeline_epoch)),
                    timing_identity.frame_seq,
                    gate_generation,
                    timing_advance_samples,
                    rx_alignment_shift_samples,
                    /*continuous_stream=*/!first,
                    /*boundary_shift_pending=*/
                        !first && target_shift_samples != applied_shift_before);
            }

            if (_use_timed_tx) {
                if (first) {
                    // A new burst can use timed metadata for the initial alignment.
                    // Later periods omit timed metadata; sample-count changes
                    // move the real boundary within the continuous stream.
                    applied_shift_samples = target_shift_samples;
                }
                md.has_time_spec = first;
                md.start_of_burst = first;
                md.end_of_burst = false;
                if (md.has_time_spec) {
                    md.time_spec = radio::TimeSpec::from_ticks(
                        next_ticks - _samples_to_ticks(target_shift_samples),
                        _tick_rate);
                }
                next_ticks += period_ticks;
            } else {
                md.has_time_spec = false;
                md.start_of_burst = first;
            }

            const SendResult result = _send_period_with_stream_shift(
                period_src,
                md,
                target_shift_samples,
                applied_shift_samples);
            if (LOG_MOD_ON(Recovery)) {
                const bool log_period =
                    first ||
                    recovery_log_periods_remaining > 0 ||
                    target_shift_samples != last_logged_target_shift ||
                    rx_alignment_id != last_logged_alignment_id ||
                    !result.complete ||
                    !has_frame;
                if (log_period) {
                    LOG_RT_WARN_M(Recovery) << "[UL-TX recovery] period period_seq=" << period_seq
                                  << ", restart_epoch=" << active_restart_epoch
                                  << ", first=" << first
                                  << ", blank=" << !has_frame
                                  << ", expected_identity_valid=" << expected_identity.valid
                                  << ", expected_epoch=" << expected_identity.timeline_epoch
                                  << ", expected_seq=" << expected_identity.frame_seq
                                  << ", queued_epoch="
                                  << (has_frame ? selected_period->identity.timeline_epoch : 0)
                                  << ", queued_seq="
                                  << (has_frame ? selected_period->identity.frame_seq : 0)
                                  << ", period_generation=" << gate_generation
                                  << ", md_time="
                                  << (md.has_time_spec ? md.time_spec.get_real_secs() : -1.0)
                                  << ", target_shift=" << target_shift_samples
                                  << ", applied_before=" << applied_shift_before
                                  << ", applied_after=" << applied_shift_samples
                                  << ", timing_advance=" << timing_advance_samples
                                  << ", rx_alignment_shift=" << rx_alignment_shift_samples
                                  << ", rx_alignment_id=" << rx_alignment_id
                                  << ", rx_alignment_restart_epoch="
                                  << rx_alignment_restart_epoch
                                  << ", sent=" << result.sent_samples
                                  << "/" << result.expected_samples
                                  << ", complete=" << result.complete
                                  << ", waveform_enabled="
                                  << _waveform_enabled.load(std::memory_order_acquire);
                    last_logged_target_shift = target_shift_samples;
                    last_logged_alignment_id = rx_alignment_id;
                }
                if (recovery_log_periods_remaining > 0) {
                    --recovery_log_periods_remaining;
                }
            }
            if (!result.complete && _running.load(std::memory_order_relaxed)) {
                _tx_error_count.fetch_add(1, std::memory_order_relaxed);
                LOG_RT_WARN_M(UlTx) << "[UL-TX] short send; dropping "
                              << (result.expected_samples - result.sent_samples)
                              << " unsent samples"
                              << (has_frame ? "" : " (blank period)")
                              << " due to underflow";
            }
            if (first && active_restart_epoch != 0 && result.sent_samples > 0) {
                _submitted_restart_epoch.store(active_restart_epoch, std::memory_order_release);
            }
            if (result.sent_samples > 0) {
                first = false;
            }
            if (has_frame) {
                _period_queue.consumer_pop();
            } else if (_use_timed_tx && expected_identity.valid &&
                       _air_frame_gate.generation() == gate_generation) {
                // Queue starvation must not leave the producer claim cursor
                // permanently behind the air timeline.
                _air_frame_gate.advance_claim_past(
                    expected_identity, gate_generation);
            }
        }
        radio::TxMetadata eob_md;
        eob_md.start_of_burst = false;
        eob_md.end_of_burst = true;
        eob_md.has_time_spec = false;
        _tx_stream->send(&_eob_sample, 0, eob_md, 0.1);
    }

    struct SendResult {
        size_t sent_samples = 0;
        size_t expected_samples = 0;
        bool complete = true;
    };

    int64_t _target_shift_samples() const {
        return static_cast<int64_t>(_timing_advance.load(std::memory_order_relaxed)) +
               static_cast<int64_t>(_rx_alignment_shift.load(std::memory_order_relaxed));
    }

    void _maybe_publish_ul_tx_timing_state(
        uint32_t timeline_epoch,
        uint64_t current_frame_seq,
        uint64_t period_generation,
        int32_t tadv_samples,
        int32_t rx_alignment_shift_samples,
        bool continuous_stream,
        bool boundary_shift_pending)
    {
        if (timeline_epoch == 0) {
            return;
        }
        // Stable periods are the common case. Avoid putting std::mutex on every
        // highest-priority TX iteration; the locked path rechecks before publish.
        if (_timing_history_fast_valid.load(std::memory_order_acquire) &&
            _timing_history_fast_epoch.load(std::memory_order_relaxed) == timeline_epoch &&
            _timing_history_fast_tadv.load(std::memory_order_relaxed) == tadv_samples &&
            _timing_history_fast_alignment.load(std::memory_order_relaxed) ==
                rx_alignment_shift_samples) {
            return;
        }
        bool should_log_publish = false;
        size_t log_n = 0;
        uint64_t log_effective_from = 0;
        bool log_continuous_shift_change = false;
        {
            std::lock_guard<std::mutex> lock(_timing_history_mutex);
            const bool component_changed =
                !_timing_history_has_last ||
                _timing_history_last.timeline_epoch != timeline_epoch ||
                _timing_history_last.tadv_samples != tadv_samples ||
                _timing_history_last.rx_alignment_shift_samples !=
                    rx_alignment_shift_samples;
            if (!component_changed) {
                _timing_history_fast_epoch.store(
                    timeline_epoch, std::memory_order_relaxed);
                _timing_history_fast_tadv.store(tadv_samples, std::memory_order_relaxed);
                _timing_history_fast_alignment.store(
                    rx_alignment_shift_samples, std::memory_order_relaxed);
                _timing_history_fast_valid.store(true, std::memory_order_release);
                return;
            }
            // First timed burst applies to the current frame. In a continuous burst,
            // aggregate shift changes move the next boundary; component changes whose
            // TADV/Alignment deltas cancel apply to the current frame.
            const bool continuous_shift_change =
                continuous_stream && boundary_shift_pending;
            const uint64_t effective_from_frame =
                ul_tx_timing_effective_from_frame(
                    current_frame_seq, continuous_stream, boundary_shift_pending);

            const size_t idx = _timing_history_count % kUlTxTimingHistoryCapacity;
            UlTxTimingStateEntry& slot = _timing_history[idx];
            slot.timeline_epoch = timeline_epoch;
            slot.effective_from_frame = effective_from_frame;
            slot.period_generation = period_generation;
            slot.tadv_samples = tadv_samples;
            slot.rx_alignment_shift_samples = rx_alignment_shift_samples;
            ++_timing_history_count;
            _timing_history_last = slot;
            _timing_history_has_last = true;
            _timing_history_fast_epoch.store(timeline_epoch, std::memory_order_relaxed);
            _timing_history_fast_tadv.store(tadv_samples, std::memory_order_relaxed);
            _timing_history_fast_alignment.store(
                rx_alignment_shift_samples, std::memory_order_relaxed);
            _timing_history_fast_valid.store(true, std::memory_order_release);

            // Snapshot log fields under the lock; emit after unlock so the
            // highest-priority TX path never holds this mutex across logging.
            log_n = _timing_history_count;
            log_effective_from = effective_from_frame;
            log_continuous_shift_change = continuous_shift_change;
            should_log_publish =
                LOG_MOD_ON(Ertm) && (log_n <= 16 || (log_n % 64) == 0);
        }
        if (should_log_publish) {
            LOG_G_INFO_M(Ertm)
                << "[eRTM] UL-TX timing state publish"
                << ": n=" << log_n
                << ", epoch=0x" << std::hex << timeline_epoch << std::dec
                << ", effective_from_frame=" << log_effective_from
                << ", tadv=" << tadv_samples
                << ", rx_alignment=" << rx_alignment_shift_samples
                << ", period_gen=" << period_generation
                << ", continuous_shift_change=" << log_continuous_shift_change
                << ", boundary_shift_pending=" << boundary_shift_pending;
        }
    }

    long long _samples_to_ticks(int64_t samples) const {
        if (!_use_timed_tx || _tick_rate <= 0.0 || _tx_sample_rate <= 0.0) {
            return samples;
        }
        return static_cast<long long>(std::llround(
            static_cast<double>(samples) * _tick_rate / _tx_sample_rate));
    }

    size_t _send_samples_once(
        const std::complex<float>* data,
        size_t sample_count,
        const radio::TxMetadata& first_md)
    {
        if (sample_count == 0) {
            return 0;
        }
        // UHD handles transport-packet fragmentation internally. Do not retry a
        // short return here: a second application call would no longer be the
        // same one-frame transaction.
        if (!radio_is_sim(_cfg)) {
            return _tx_stream->send(data, sample_count, first_md, 0.1);
        }

        // A timed simulator burst can fill the ring before its scheduled start.
        // Retry backpressure without dropping samples or restarting RX/TX.
        size_t sent_total = 0;
        radio::TxMetadata md = first_md;
        SPSCBusySpin spin;
        while (sent_total < sample_count && _running.load(std::memory_order_relaxed)) {
            const size_t sent = _tx_stream->send(
                data + sent_total, sample_count - sent_total, md, 0.1);
            if (sent == 0) {
                spin.pause();
                if (spin.spins() >= 10'000'000u) break;
                continue;
            }
            spin.reset();
            sent_total += sent;
            md.has_time_spec = false;
            md.start_of_burst = false;
        }
        return sent_total;
    }

    SendResult _send_period_with_stream_shift(
        const AlignedVector& frame,
        const radio::TxMetadata& first_md,
        int64_t target_shift_samples,
        int64_t& applied_shift_samples)
    {
        SendResult result{};
        result.expected_samples = _period_samples;
        result.complete = true;

        const int64_t delta = target_shift_samples - applied_shift_samples;
        const bool waveform_enabled =
            _waveform_enabled.load(std::memory_order_acquire);
        const std::complex<float>* send_data = waveform_enabled
            ? frame.data()
            : _blank_period_frame.data();
        size_t send_samples = frame.size();

        if (delta < 0) {
            const size_t insert_samples = static_cast<size_t>(-delta);
            result.expected_samples += insert_samples;
            if (insert_samples > 0) {
                LOG_RT_INFO_HZ_M(UlTx, 2) << "[UL-TX] delaying stream by lengthening this period by "
                                  << insert_samples << " samples (target_shift="
                                  << target_shift_samples << ", applied_shift="
                                  << target_shift_samples << ")";
            }
            if (frame.size() + insert_samples > _tx_frame_buffer.size()) {
                result.complete = false;
                return result;
            }
            if (waveform_enabled) {
                std::copy(frame.begin(), frame.end(), _tx_frame_buffer.begin());
            } else {
                std::fill_n(
                    _tx_frame_buffer.begin(), frame.size(), std::complex<float>(0.0f, 0.0f));
            }
            std::fill_n(
                _tx_frame_buffer.begin() + static_cast<std::ptrdiff_t>(frame.size()),
                insert_samples,
                std::complex<float>(0.0f, 0.0f));
            send_data = _tx_frame_buffer.data();
            send_samples = frame.size() + insert_samples;
        } else if (delta > 0) {
            const size_t skip_samples = std::min<size_t>(
                static_cast<size_t>(delta),
                frame.size());
            result.expected_samples -= skip_samples;
            if (skip_samples > 0) {
                LOG_RT_INFO_HZ_M(UlTx, 2) << "[UL-TX] advancing stream by dropping "
                                  << skip_samples << " samples from this period (target_shift="
                                  << target_shift_samples << ", applied_shift="
                                  << (applied_shift_samples + static_cast<int64_t>(skip_samples))
                                  << ")";
            }
            if (skip_samples >= frame.size()) {
                applied_shift_samples += static_cast<int64_t>(skip_samples);
                return result;
            }
            send_samples = frame.size() - skip_samples;
        }

        result.sent_samples = _send_samples_once(send_data, send_samples, first_md);
        result.complete = (result.sent_samples == send_samples);
        if (result.complete && delta != 0) {
            applied_shift_samples = target_shift_samples;
        }
        return result;
    }

    const Config _link_cfg;
    const Config _cfg;                       // derived uplink (sub)frame config
    const DigitalModulation _payload_modulation{_cfg.modulation.scheme};
    const LdpcRateMatching _payload_rate{_cfg.ldpc.code_rate};
    const DuplexFrameLayout _duplex;
    const DataResourceGridLayout _layout;

    AlignedVector _zc_seq;
    SymbolVector _symbol_templates;
    std::vector<int> _payload_subcarrier_indices_flat;

    AlignedVector _fft_in;
    AlignedVector _fft_out;
    fftwf_plan _ifft_plan = nullptr;
    AlignedVector _mod_pool;
    AlignedIntVector _data_pool;   // reused per-frame payload symbol scratch

    // Match BS-style multi-frame TX cushion (was 2; too shallow under load).
    static constexpr size_t kPeriodQueueDepth = 8;

    const size_t _period_samples;
    AirFrameTxGate _air_frame_gate;
    const size_t _window_offset;
    const size_t _window_samples;
    SPSCRingBuffer<QueuedUlPeriod> _period_queue;
    std::complex<float> _eob_sample{0.0f, 0.0f};
    AlignedVector _tx_frame_buffer;  // frame plus up to one frame of timing padding
    AlignedVector _blank_period_frame;  // zeros; injected when period queue is empty

    LDPCCodec _ldpc{make_ldpc_5041008_cfg()};
    std::unique_ptr<BitBlockInterleaver> _bit_interleaver;
    LDPCCodec::AlignedIntVector _interleaver_scratch;
    Scrambler _scrambler{LdpcPacketFraming::scrambler_bits(_payload_modulation), 0x5A};
    std::atomic<uint32_t> _packet_seq{0};
    std::atomic<uint32_t> _idle_seq{0};
    SPSCRingBuffer<AlignedIntVector> _packet_buffer{256};

    // ARQ transmit window for uplink direction
    ArqTxWindow _ul_arq_tx;
    bool _arq_enabled{false};
    bool _arq_feedback_enabled{false};
    // Dedicated seq space for feedback frames so they never collide with data seqs.
    std::atomic<uint32_t> _arq_feedback_seq{0};
    std::atomic<uint64_t> _arq_feedback_injected{0};
    std::atomic<uint64_t> _arq_feedback_dropped_disabled{0};
    std::atomic<uint64_t> _arq_feedback_drained{0};
    // Feedback queued by the DL decode thread, drained by the encode thread to
    // preserve the single-producer contract of _packet_buffer.
    mutable std::mutex _arq_feedback_mutex;
    std::deque<std::vector<uint8_t>> _arq_pending_feedback;

    struct RawUdpPacket {
        std::vector<uint8_t> bytes;
    };
    SPSCRingBuffer<RawUdpPacket> _raw_udp_buffer{256};
    std::thread _udp_recv_thread;

    radio::ITxStreamPtr _tx_stream;
    bool _use_timed_tx = false;
    std::mutex _timing_mutex;
    radio::TimeSpec _start_time{0.0};
    double _tick_rate = 0.0;
    double _tx_sample_rate = 0.0;
    std::atomic<int32_t> _timing_advance{0};
    std::atomic<int32_t> _rx_alignment_shift{0};
    std::atomic<uint64_t> _rx_alignment_update_id{0};
    std::atomic<uint64_t> _rx_alignment_restart_epoch{0};

    // Absolute-frame indexed UL-TX timing change history (eRTM compensation).
    // Capacity covers engineering-scale TADV/Alignment change counts; if an
    // old entry is overwritten while still referenced by a cached ERTM5, the
    // UE must skip that report rather than guess current state.
    // POD history access is serialized by _timing_history_mutex. The TX-only
    // atomic mirror skips the lock while the published components are unchanged.
    static constexpr size_t kUlTxTimingHistoryCapacity = 64;
    UlTxTimingStateEntry _timing_history[kUlTxTimingHistoryCapacity]{};
    size_t _timing_history_count = 0;
    mutable std::mutex _timing_history_mutex;
    bool _timing_history_has_last = false;
    UlTxTimingStateEntry _timing_history_last{};
    std::atomic<bool> _timing_history_fast_valid{false};
    std::atomic<uint32_t> _timing_history_fast_epoch{0};
    std::atomic<int32_t> _timing_history_fast_tadv{0};
    std::atomic<int32_t> _timing_history_fast_alignment{0};
    std::atomic<uint64_t> _identity_sib_injected{0};  ///< Periods with valid DL stamp
    std::atomic<uint64_t> _identity_sib_unlocked{0};  ///< Periods with zero/invalid stamp
    std::atomic<uint64_t> _stamp_update_count{0};
    std::atomic<uint64_t> _period_claim_count{0};
    std::atomic<uint64_t> _gate_match_count{0};
    std::atomic<uint64_t> _gate_hold_count{0};
    std::atomic<uint64_t> _gate_blank_count{0};
    std::atomic<uint64_t> _scheduled_restart_epoch{0};
    std::atomic<uint64_t> _submitted_restart_epoch{0};
    std::atomic<bool> _waveform_enabled{false};
    std::atomic<bool> _restart_requested{false};
    std::atomic<uint64_t> _tx_error_count{0};

    std::atomic<bool> _running{false};
    std::thread _ldpc_encode_thread;
    std::thread _mod_thread;
    std::thread _tx_thread;
    std::atomic<int> _udp_sock{-1};
};

#endif // UPLINK_TX_ENGINE_HPP
