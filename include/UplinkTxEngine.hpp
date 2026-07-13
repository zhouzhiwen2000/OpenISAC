#ifndef UPLINK_TX_ENGINE_HPP
#define UPLINK_TX_ENGINE_HPP

// UplinkTxEngine — focused uplink (UE->BS) OFDM transmit engine.
//
// This is the matched TX half of the duplex uplink. It builds a self-contained
// compact OFDM frame (ZC sync at symbol 0, comb pilots, LDPC/QPSK payload) using
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
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <fftw3.h>
#include <uhd/stream.hpp>
#include <uhd/types/time_spec.hpp>

#include "Common.hpp"
#include "OFDMCore.hpp"
#include "LDPCCodec.hpp"

class UplinkTxEngine {
public:
    explicit UplinkTxEngine(const Config& link_cfg)
        : _link_cfg(link_cfg),
          _cfg(make_uplink_config(link_cfg)),
          _duplex(build_duplex_frame_layout(link_cfg)),
          _layout(build_data_resource_grid_layout(_cfg)),
          _fft_in(_cfg.fft_size),
          _fft_out(_cfg.fft_size),
          _period_samples(link_cfg.samples_per_frame()),
          _window_offset(_duplex.ul_sample_offset(link_cfg)),
          _window_samples(_duplex.ul_sample_count(link_cfg)) {
        if (_cfg.num_symbols < 2) {
            throw std::runtime_error("UplinkTxEngine: uplink needs >= 2 OFDM symbols (1 sync + data).");
        }
        if (_window_samples != _cfg.samples_per_frame()) {
            throw std::runtime_error("UplinkTxEngine: uplink window samples != uplink frame samples.");
        }
        _zc_seq = generate_zc_freq(_cfg.fft_size, _cfg.zc_root);
        _bit_interleaver = std::make_unique<BitBlockInterleaver>(_ldpc.get_N(), 21);
        _build_symbol_templates();
        _ifft_plan = fftwf_plan_dft_1d(
            static_cast<int>(_cfg.fft_size),
            reinterpret_cast<fftwf_complex*>(_fft_in.data()),
            reinterpret_cast<fftwf_complex*>(_fft_out.data()),
            FFTW_BACKWARD, FFTW_MEASURE);
        _period_queue.reset(2, [this]() {
            AlignedVector frame;
            frame.resize(_period_samples);
            return frame;
        });
        _timing_pad_buffer.assign(_period_samples, std::complex<float>(0.0f, 0.0f));
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

    void set_tx_stream(uhd::tx_streamer::sptr stream) { _tx_stream = std::move(stream); }
    uhd::tx_streamer::sptr tx_stream() const { return _tx_stream; }

    // Enable timed (real-USRP) transmission scheduled on the radio clock. When
    // not enabled (sim), frames are streamed back-to-back, paced by the shm ring.
    void set_timed_tx(uhd::time_spec_t start_time, double tick_rate, double tx_sample_rate) {
        std::lock_guard<std::mutex> lock(_timing_mutex);
        _start_time = start_time;
        _tick_rate = tick_rate;
        _tx_sample_rate = tx_sample_rate > 0.0 ? tx_sample_rate : _link_cfg.sample_rate;
        _use_timed_tx = (tick_rate > 0.0);
        _restart_requested.store(false, std::memory_order_release);
    }

    void reschedule_timed_tx(uhd::time_spec_t start_time) {
        {
            std::lock_guard<std::mutex> lock(_timing_mutex);
            _start_time = start_time;
        }
        _restart_requested.store(true, std::memory_order_release);
    }

    void request_restart() {
        _restart_requested.store(true, std::memory_order_release);
    }

    // Runtime-adjustable Timing Advance (samples). Pulls the uplink TX window
    // earlier on the shared clock so it lands aligned at the BS.
    std::atomic<int32_t>& timing_advance() { return _timing_advance; }
    // Relative shift tracking the UE RX alignment (Phase 5).
    std::atomic<int32_t>& rx_alignment_shift() { return _rx_alignment_shift; }
    std::atomic<uint64_t>& tx_error_count() { return _tx_error_count; }

    bool set_waveform_enabled(bool enabled) {
        return _waveform_enabled.exchange(enabled, std::memory_order_acq_rel) != enabled;
    }
    bool waveform_enabled() const {
        return _waveform_enabled.load(std::memory_order_acquire);
    }

    const Config& uplink_config() const { return _cfg; }

    void start() {
        if (!_tx_stream) {
            throw std::runtime_error("UplinkTxEngine::start without a TX stream.");
        }
        _waveform_enabled.store(false, std::memory_order_release);
        _running.store(true);
        if (AlignedVector* first_frame = _period_queue.producer_slot()) {
            _build_period_buffer(*first_frame);
            _period_queue.producer_commit();
        }
        _ingest_thread = std::thread(&UplinkTxEngine::_udp_ingest_proc, this);
        _mod_thread = std::thread(&UplinkTxEngine::_mod_proc, this);
        _tx_thread = std::thread(&UplinkTxEngine::_tx_proc, this);
    }

    void stop() {
        _running.store(false);
        if (_udp_sock >= 0) {
            ::shutdown(_udp_sock, SHUT_RDWR);
        }
        if (_ingest_thread.joinable()) _ingest_thread.join();
        if (_mod_thread.joinable()) _mod_thread.join();
        if (_tx_thread.joinable()) _tx_thread.join();
        if (_udp_sock >= 0) {
            ::close(_udp_sock);
            _udp_sock = -1;
        }
    }

private:
    // ---- One-time setup: build per-symbol frequency templates + payload map ----
    void _build_symbol_templates() {
        _symbol_templates.assign(_cfg.num_symbols, AlignedVector(_cfg.fft_size, {0.0f, 0.0f}));
        _payload_subcarrier_indices_flat.clear();
        _payload_subcarrier_indices_flat.reserve(_layout.payload_re_count);

        size_t data_symbol_idx = 0;
        for (size_t sym = 0; sym < _cfg.num_symbols; ++sym) {
            AlignedVector& tmpl = _symbol_templates[sym];
            if (is_zc_sync_symbol(_cfg, sym)) {
                std::memcpy(tmpl.data(), _zc_seq.data(), _cfg.fft_size * sizeof(std::complex<float>));
                continue;
            }
            const size_t base = _layout.non_pilot_offsets[data_symbol_idx];
            for (size_t di = 0; di < _layout.num_non_pilot_subcarriers; ++di) {
                const size_t k = static_cast<size_t>(_layout.non_pilot_subcarrier_indices[di]);
                const int payload_rank = _layout.payload_rank[base + di];
                if (payload_rank >= 0) {
                    // payload RE: filled per-frame; record the subcarrier.
                    _payload_subcarrier_indices_flat.push_back(static_cast<int>(k));
                } else {
                    // deterministic unit-magnitude filler (RX ignores these REs).
                    tmpl[k] = _zc_seq[k];
                }
            }
            for (const size_t k : _cfg.pilot_positions) {
                if (k < _cfg.fft_size) tmpl[k] = _zc_seq[k];
            }
            ++data_symbol_idx;
        }
        if (_payload_subcarrier_indices_flat.size() != _layout.payload_re_count) {
            throw std::runtime_error("UplinkTxEngine: payload RE flatten mismatch.");
        }
    }

    static std::complex<float> _qpsk_from_int(int sym) {
        const int idx = (sym & 3) * 2;
        return std::complex<float>(QPSKModulator::QPSK_TABLE_FLAT[idx],
                                   QPSKModulator::QPSK_TABLE_FLAT[idx + 1]);
    }

    void _append_idle_symbols(AlignedIntVector& data_pool) {
        if (_layout.payload_re_count < LdpcPacketFraming::kControlSymbols) {
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
        data_pool.resize(LdpcPacketFraming::kControlSymbols);
        LdpcPacketFraming::write_control_qpsk(hdr, data_pool.data());

        if (_link_cfg.uplink_idle_waveform != kUplinkIdleWaveformRandomQpsk) {
            return;
        }
        data_pool.reserve(_layout.payload_re_count);
        while (data_pool.size() < _layout.payload_re_count) {
            const uint32_t rnd = splitmix32(0x554C4944u ^ idle_seq ^ static_cast<uint32_t>(data_pool.size()));
            data_pool.push_back(static_cast<int>(rnd & 0x3u));
        }
    }

    // ---- UDP payload ingest: recv -> LDPC encode/frame -> packet queue ----
    void _udp_ingest_proc() {
        bind_current_thread_from_uplink_hint(_link_cfg, 0);
        _udp_sock = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (_udp_sock < 0) {
            LOG_G_ERROR() << "[UL-TX] UDP socket create failed";
            return;
        }
        int enable = 1;
        ::setsockopt(_udp_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(_link_cfg.ul_udp_input_port));
        if (_link_cfg.ul_udp_input_ip == "0.0.0.0") {
            addr.sin_addr.s_addr = INADDR_ANY;
        } else if (inet_pton(AF_INET, _link_cfg.ul_udp_input_ip.c_str(), &addr.sin_addr) != 1) {
            LOG_G_ERROR() << "[UL-TX] Invalid ul_udp_input_ip: " << _link_cfg.ul_udp_input_ip;
            return;
        }
        if (::bind(_udp_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            LOG_G_ERROR() << "[UL-TX] UDP bind failed on " << _link_cfg.ul_udp_input_ip << ":"
                          << _link_cfg.ul_udp_input_port;
            return;
        }
        if (_link_cfg.should_profile("uplink")) {
            LOG_G_INFO() << "[UL-TX] uplink payload UDP input on " << _link_cfg.ul_udp_input_ip << ":"
                         << _link_cfg.ul_udp_input_port;
        }

        std::vector<uint8_t> buf(65536);
        while (_running.load(std::memory_order_relaxed)) {
            const ssize_t n = ::recvfrom(_udp_sock, buf.data(), buf.size(), 0, nullptr, nullptr);
            if (n <= 0) {
                if (!_running.load(std::memory_order_relaxed)) break;
                continue;
            }
            _encode_and_enqueue_payload(buf.data(), static_cast<size_t>(n));
        }
    }

    void _encode_and_enqueue_payload(const uint8_t* data, size_t len) {
        const size_t bytes_per_block = (_ldpc.get_K() + 7) / 8;
        if (bytes_per_block != LdpcPacketFraming::kLdpcInfoBytesPerBlock ||
            _ldpc.get_N() != LdpcPacketFraming::kLdpcCodeBitsPerBlock) {
            throw std::runtime_error("UplinkTxEngine: LDPC dimensions do not match framing.");
        }
        if (!LdpcPacketFraming::payload_len_fits(len)) return;
        const size_t payload_blocks = LdpcPacketFraming::payload_blocks_for_len(len);
        const size_t packet_qpsk = LdpcPacketFraming::packet_qpsk_symbols(payload_blocks);
        if (packet_qpsk > _layout.payload_re_count) {
            if (_link_cfg.should_profile("uplink")) {
                LOG_RT_WARN_HZ(2) << "[UL-TX] dropping payload: " << packet_qpsk
                                  << " qpsk syms > capacity " << _layout.payload_re_count;
            }
            return;
        }

        AlignedIntVector packet;
        packet.resize(LdpcPacketFraming::kControlSymbols);
        const LdpcMiniHeader hdr{
            LdpcPacketFraming::kVersion,
            LdpcPacketFraming::kFlags,
            static_cast<uint16_t>(len),
            LdpcPacketFraming::payload_blocks_field_for_len(len),
            static_cast<uint16_t>(_packet_seq.fetch_add(1, std::memory_order_relaxed) & 0xFFFFu),
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
            LDPCCodec::pack_bits_qpsk(encoded_bits, qpsk_ints);
            packet.insert(packet.end(), qpsk_ints.begin(), qpsk_ints.end());
        }
        if (packet.size() != packet_qpsk) {
            throw std::runtime_error("UplinkTxEngine: packet symbol count mismatch.");
        }
        SPSCBackoff backoff;
        while (_running.load(std::memory_order_relaxed)) {
            if (_packet_buffer.try_push(std::move(packet))) break;
            backoff.pause();
        }
    }

    // ---- Build one uplink frame's IQ into period_buffer at the UL window ----
    void _build_period_buffer(AlignedVector& period_buffer) {
        // The period buffers come from the pool zero-initialized and the only
        // region ever written is the UL window [_window_offset, +_window_samples),
        // which the symbol loop below overwrites in full every frame (the ctor
        // asserts _window_samples == uplink samples_per_frame() == num_symbols *
        // sym_len, and the loop writes exactly that many contiguous samples). So
        // the rest of the DL-period buffer stays zero permanently — no per-frame
        // fill over the (large) full period is needed. If a future layout leaves
        // gaps inside the UL window, re-add a window-scoped zeroing here.

        // Pull payload packets up to one frame's payload capacity.
        AlignedIntVector& data_pool = _data_pool;
        data_pool.clear();
        data_pool.reserve(_layout.payload_re_count);
        while (data_pool.size() < _layout.payload_re_count) {
            AlignedIntVector* slot = _packet_buffer.consumer_slot();
            if (slot == nullptr) break;
            const size_t room = _layout.payload_re_count - data_pool.size();
            if (slot->size() > room) {
                if (slot->size() > _layout.payload_re_count) {
                    _packet_buffer.consumer_pop();  // oversized: drop to avoid stall
                }
                break;
            }
            data_pool.insert(data_pool.end(), slot->begin(), slot->end());
            _packet_buffer.consumer_pop();
        }
        if (data_pool.empty()) {
            _append_idle_symbols(data_pool);
        }

        _mod_pool.resize(data_pool.size());
        for (size_t i = 0; i < data_pool.size(); ++i) _mod_pool[i] = _qpsk_from_int(data_pool[i]);

        const float scale = 1.0f / std::sqrt(static_cast<float>(_cfg.fft_size)) / 4.0f;
        const size_t sym_len = _cfg.fft_size + _cfg.cp_length;
        size_t data_symbol_idx = 0;
        for (size_t sym = 0; sym < _cfg.num_symbols; ++sym) {
            std::memcpy(_fft_in.data(), _symbol_templates[sym].data(),
                        _cfg.fft_size * sizeof(std::complex<float>));
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
            for (size_t j = 0; j < _cfg.cp_length; ++j) {
                dst[j] = src[_cfg.fft_size - _cfg.cp_length + j] * scale;
            }
            #pragma omp simd
            for (size_t j = 0; j < _cfg.fft_size; ++j) {
                dst[_cfg.cp_length + j] = src[j] * scale;
            }
        }
    }

    void _mod_proc() {
        uhd::set_thread_priority_safe();
        bind_current_thread_from_uplink_hint(_link_cfg, 1);
        SPSCBackoff backoff;
        while (_running.load(std::memory_order_relaxed)) {
            AlignedVector* frame = _period_queue.producer_slot();
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
        uhd::set_thread_priority_safe();
        bind_current_thread_from_uplink_hint(_link_cfg, 2);
        uhd::tx_metadata_t md;
        md.start_of_burst = true;
        md.end_of_burst = false;

        long long period_ticks = 0;
        long long next_ticks = 0;
        int64_t applied_shift_samples = 0;
        bool first = true;
        if (_use_timed_tx) {
            const double exact = static_cast<double>(_period_samples) * _tick_rate / _tx_sample_rate;
            period_ticks = std::llround(exact);
            std::lock_guard<std::mutex> lock(_timing_mutex);
            next_ticks = _start_time.to_ticks(_tick_rate);
        }

        SPSCBackoff frame_backoff;
        while (_running.load(std::memory_order_relaxed)) {
            if (_use_timed_tx && _restart_requested.exchange(false, std::memory_order_acq_rel)) {
                uhd::tx_metadata_t eob_md;
                eob_md.start_of_burst = false;
                eob_md.end_of_burst = true;
                eob_md.has_time_spec = false;
                try {
                    _tx_stream->send(&_eob_sample, 0, eob_md, 0.1);
                } catch (const std::exception& e) {
                    if (_link_cfg.should_profile("uplink")) {
                        LOG_RT_WARN() << "[UL-TX] failed to terminate burst before restart: " << e.what();
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(_timing_mutex);
                    next_ticks = _start_time.to_ticks(_tick_rate);
                }
                applied_shift_samples = 0;
                first = true;
                _period_queue.clear();
                if (_link_cfg.should_profile("uplink")) {
                    LOG_RT_WARN() << "[UL-TX] timed TX restart scheduled at "
                                  << uhd::time_spec_t::from_ticks(next_ticks, _tick_rate).get_real_secs()
                                  << " s";
                }
            }

            AlignedVector* frame = _period_queue.consumer_slot();
            if (frame == nullptr) {
                frame_backoff.pause();
                continue;
            }
            frame_backoff.reset();

            const int64_t target_shift_samples = _target_shift_samples();
            if (_use_timed_tx) {
                if (first) {
                    // A new burst can use timed metadata for the initial alignment.
                    // Later in the same continuous burst, metadata follows the
                    // target and the sample count change moves the real boundary.
                    applied_shift_samples = target_shift_samples;
                }
                md.has_time_spec = true;
                md.start_of_burst = first;
                md.end_of_burst = false;
                md.time_spec = uhd::time_spec_t::from_ticks(
                    next_ticks - _samples_to_ticks(target_shift_samples),
                    _tick_rate);
                next_ticks += period_ticks;
            } else {
                md.has_time_spec = false;
                md.start_of_burst = first;
            }

            const SendResult result = _send_period_with_stream_shift(
                *frame,
                md,
                target_shift_samples,
                applied_shift_samples);
            if (!result.complete && _running.load(std::memory_order_relaxed)) {
                _tx_error_count.fetch_add(1, std::memory_order_relaxed);
                if (_link_cfg.should_profile("uplink")) {
                    LOG_RT_WARN() << "[UL-TX] short send/underflow: "
                                  << (result.expected_samples - result.sent_samples)
                                  << " samples not sent";
                }
            }
            if (result.sent_samples > 0) {
                first = false;
            }
            _period_queue.consumer_pop();
        }
        uhd::tx_metadata_t eob_md;
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

    long long _samples_to_ticks(int64_t samples) const {
        if (!_use_timed_tx || _tick_rate <= 0.0 || _tx_sample_rate <= 0.0) {
            return samples;
        }
        return static_cast<long long>(std::llround(
            static_cast<double>(samples) * _tick_rate / _tx_sample_rate));
    }

    size_t _send_samples(
        const std::complex<float>* data,
        size_t sample_count,
        const uhd::tx_metadata_t& first_md)
    {
        size_t sent_total = 0;
        uhd::tx_metadata_t md = first_md;
        SPSCBackoff backoff;
        while (sent_total < sample_count && _running.load(std::memory_order_relaxed)) {
            const size_t sent = _tx_stream->send(
                data + sent_total,
                sample_count - sent_total,
                md,
                1.0);
            if (sent == 0) {
                backoff.pause();
                continue;
            }
            backoff.reset();
            sent_total += sent;
            md.has_time_spec = false;
            md.start_of_burst = false;
        }
        return sent_total;
    }

    size_t _send_zeros(size_t sample_count, const uhd::tx_metadata_t& first_md) {
        size_t sent_total = 0;
        uhd::tx_metadata_t md = first_md;
        while (sent_total < sample_count && _running.load(std::memory_order_relaxed)) {
            const size_t chunk = std::min(_timing_pad_buffer.size(), sample_count - sent_total);
            const size_t sent = _send_samples(_timing_pad_buffer.data(), chunk, md);
            sent_total += sent;
            md.has_time_spec = false;
            md.start_of_burst = false;
            if (sent < chunk) {
                break;
            }
        }
        return sent_total;
    }

    size_t _send_period_samples(
        const AlignedVector& frame,
        size_t sample_offset,
        size_t sample_count,
        const uhd::tx_metadata_t& first_md)
    {
        if (sample_offset >= frame.size() || sample_count == 0) {
            return 0;
        }
        const size_t frame_end = sample_count >= (frame.size() - sample_offset)
            ? frame.size()
            : (sample_offset + sample_count);
        const size_t bounded_count = frame_end - sample_offset;
        if (_waveform_enabled.load(std::memory_order_acquire)) {
            return _send_samples(frame.data() + sample_offset, bounded_count, first_md);
        }
        return _send_zeros(bounded_count, first_md);
    }

    SendResult _send_period_with_stream_shift(
        const AlignedVector& frame,
        const uhd::tx_metadata_t& first_md,
        int64_t target_shift_samples,
        int64_t& applied_shift_samples)
    {
        SendResult result{};
        result.expected_samples = _period_samples;
        result.complete = true;

        int64_t delta = target_shift_samples - applied_shift_samples;
        uhd::tx_metadata_t frame_md = first_md;

        if (delta < 0) {
            const size_t insert_samples = static_cast<size_t>(-delta);
            result.expected_samples += insert_samples;
            if (insert_samples > 0) {
                if (_link_cfg.should_profile("uplink")) {
                    LOG_RT_WARN_HZ(2) << "[UL-TX] delaying stream by lengthening this period by "
                                      << insert_samples << " samples (target_shift="
                                      << target_shift_samples << ", applied_shift="
                                      << target_shift_samples << ")";
                }
            }
            const size_t sent = _send_period_samples(frame, 0, frame.size(), frame_md);
            result.sent_samples += sent;
            if (sent < frame.size()) {
                result.complete = false;
                return result;
            }
            frame_md.has_time_spec = false;
            frame_md.start_of_burst = false;
            const size_t inserted = _send_zeros(insert_samples, frame_md);
            result.sent_samples += inserted;
            if (inserted < insert_samples) {
                result.complete = false;
                return result;
            }
            applied_shift_samples = target_shift_samples;
        } else if (delta > 0) {
            const size_t skip_samples = std::min<size_t>(
                static_cast<size_t>(delta),
                frame.size());
            result.expected_samples -= skip_samples;
            if (skip_samples > 0) {
                if (_link_cfg.should_profile("uplink")) {
                    LOG_RT_WARN_HZ(2) << "[UL-TX] advancing stream by shortening this period by "
                                      << skip_samples << " samples (target_shift="
                                      << target_shift_samples << ", applied_shift="
                                      << (applied_shift_samples + static_cast<int64_t>(skip_samples))
                                      << ")";
                }
            }
            if (skip_samples >= frame.size()) {
                applied_shift_samples += static_cast<int64_t>(skip_samples);
                return result;
            }
            const size_t sent = _send_period_samples(
                frame,
                0,
                frame.size() - skip_samples,
                frame_md);
            result.sent_samples += sent;
            result.complete = (sent == frame.size() - skip_samples);
            if (result.complete) {
                applied_shift_samples += static_cast<int64_t>(skip_samples);
            }
            return result;
        }

        const size_t sent = _send_period_samples(frame, 0, frame.size(), frame_md);
        result.sent_samples += sent;
        result.complete = (sent == frame.size());
        return result;
    }

    const Config _link_cfg;
    const Config _cfg;                       // derived uplink (sub)frame config
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

    const size_t _period_samples;
    const size_t _window_offset;
    const size_t _window_samples;
    SPSCRingBuffer<AlignedVector> _period_queue;
    std::complex<float> _eob_sample{0.0f, 0.0f};
    AlignedVector _timing_pad_buffer;

    LDPCCodec _ldpc{make_ldpc_5041008_cfg()};
    std::unique_ptr<BitBlockInterleaver> _bit_interleaver;
    LDPCCodec::AlignedIntVector _interleaver_scratch;
    Scrambler _scrambler{201600, 0x5A};
    std::atomic<uint32_t> _packet_seq{0};
    std::atomic<uint32_t> _idle_seq{0};
    SPSCRingBuffer<AlignedIntVector> _packet_buffer{32};

    uhd::tx_streamer::sptr _tx_stream;
    bool _use_timed_tx = false;
    std::mutex _timing_mutex;
    uhd::time_spec_t _start_time{0.0};
    double _tick_rate = 0.0;
    double _tx_sample_rate = 0.0;
    std::atomic<int32_t> _timing_advance{0};
    std::atomic<int32_t> _rx_alignment_shift{0};
    std::atomic<bool> _waveform_enabled{false};
    std::atomic<bool> _restart_requested{false};
    std::atomic<uint64_t> _tx_error_count{0};

    std::atomic<bool> _running{false};
    std::thread _ingest_thread;
    std::thread _mod_thread;
    std::thread _tx_thread;
    int _udp_sock = -1;
};

#endif // UPLINK_TX_ENGINE_HPP
