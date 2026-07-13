#ifndef UPLINK_TX_ENGINE_HPP
#define UPLINK_TX_ENGINE_HPP

// UplinkTxEngine — focused uplink (UE->BS) OFDM transmit engine.
//
// This is the matched TX half of the duplex uplink. It builds a self-contained
// compact OFDM frame (ZC sync at symbol 0, comb pilots, LDPC/QPSK payload) using
// the uplink config derived by make_uplink_config(), then places that frame at
// the uplink symbol window of the downlink frame period and streams it.
//
// It deliberately reuses only the pure DSP primitives from OFDMCore.hpp
// (QPSKModulator, generate_zc_freq, LdpcPacketFraming, Scrambler,
// BitBlockInterleaver) + LDPCCodec — no sensing / AGC / OCXO / control coupling.
// The uplink runs no active clock/timing loop: it rides the shared radio clock,
// with alignment provided passively by the host (timing advance / frame shift).
//
// Payload source: a UDP input socket (mirrors the BS udp_input_* path).

#include <atomic>
#include <cstring>
#include <memory>
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

    // Enable timed (real-USRP) transmission scheduled on the radio clock. When
    // not enabled (sim), frames are streamed back-to-back, paced by the shm ring.
    void set_timed_tx(uhd::time_spec_t start_time, double tick_rate, double tx_sample_rate) {
        _start_time = start_time;
        _tick_rate = tick_rate;
        _tx_sample_rate = tx_sample_rate > 0.0 ? tx_sample_rate : _link_cfg.sample_rate;
        _use_timed_tx = (tick_rate > 0.0);
    }

    // Runtime-adjustable Timing Advance (samples). Pulls the uplink TX window
    // earlier on the shared clock so it lands aligned at the BS.
    std::atomic<int32_t>& timing_advance() { return _timing_advance; }
    // Relative shift tracking the UE RX alignment (Phase 5).
    std::atomic<int32_t>& rx_alignment_shift() { return _rx_alignment_shift; }

    const Config& uplink_config() const { return _cfg; }

    void start() {
        if (!_tx_stream) {
            throw std::runtime_error("UplinkTxEngine::start without a TX stream.");
        }
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
        LOG_G_INFO() << "[UL-TX] uplink payload UDP input on " << _link_cfg.ul_udp_input_ip << ":"
                     << _link_cfg.ul_udp_input_port;

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
            LOG_RT_WARN_HZ(2) << "[UL-TX] dropping payload: " << packet_qpsk
                              << " qpsk syms > capacity " << _layout.payload_re_count;
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
        std::fill(period_buffer.begin(), period_buffer.end(), std::complex<float>(0.0f, 0.0f));

        // Pull payload packets up to one frame's payload capacity.
        AlignedIntVector data_pool;
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
            std::complex<float>* dst = period_buffer.data() + _window_offset + sym * sym_len;
            for (size_t j = 0; j < _cfg.cp_length; ++j) {
                dst[j] = _fft_out[_cfg.fft_size - _cfg.cp_length + j] * scale;
            }
            for (size_t j = 0; j < _cfg.fft_size; ++j) {
                dst[_cfg.cp_length + j] = _fft_out[j] * scale;
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
        if (_use_timed_tx) {
            const double exact = static_cast<double>(_period_samples) * _tick_rate / _tx_sample_rate;
            period_ticks = std::llround(exact);
            next_ticks = _start_time.to_ticks(_tick_rate);
        }

        bool first = true;
        SPSCBackoff frame_backoff;
        while (_running.load(std::memory_order_relaxed)) {
            AlignedVector* frame = _period_queue.consumer_slot();
            if (frame == nullptr) {
                frame_backoff.pause();
                continue;
            }
            frame_backoff.reset();

            if (_use_timed_tx) {
                // Timing advance + RX-alignment shift pull the uplink window
                // earlier/later on the shared clock (passive alignment).
                const long long shift =
                    static_cast<long long>(_timing_advance.load(std::memory_order_relaxed)) +
                    static_cast<long long>(_rx_alignment_shift.load(std::memory_order_relaxed));
                md.has_time_spec = true;
                md.start_of_burst = first;
                md.end_of_burst = false;
                md.time_spec = uhd::time_spec_t::from_ticks(next_ticks - shift, _tick_rate);
                next_ticks += period_ticks;
                first = false;
            } else {
                md.has_time_spec = false;
                md.start_of_burst = false;
            }

            _send_period(*frame, md);
            _period_queue.consumer_pop();
        }
        md.end_of_burst = true;
        _tx_stream->send(&_eob_sample, 0, md, 0.1);
    }

    void _send_period(const AlignedVector& frame, const uhd::tx_metadata_t& first_md) {
        size_t sent_total = 0;
        uhd::tx_metadata_t md = first_md;
        SPSCBackoff backoff;
        while (sent_total < _period_samples && _running.load(std::memory_order_relaxed)) {
            const size_t sent = _tx_stream->send(
                frame.data() + sent_total,
                _period_samples - sent_total,
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

    const size_t _period_samples;
    const size_t _window_offset;
    const size_t _window_samples;
    SPSCRingBuffer<AlignedVector> _period_queue;
    std::complex<float> _eob_sample{0.0f, 0.0f};

    LDPCCodec _ldpc{make_ldpc_5041008_cfg()};
    std::unique_ptr<BitBlockInterleaver> _bit_interleaver;
    LDPCCodec::AlignedIntVector _interleaver_scratch;
    Scrambler _scrambler{201600, 0x5A};
    std::atomic<uint32_t> _packet_seq{0};
    SPSCRingBuffer<AlignedIntVector> _packet_buffer{32};

    uhd::tx_streamer::sptr _tx_stream;
    bool _use_timed_tx = false;
    uhd::time_spec_t _start_time{0.0};
    double _tick_rate = 0.0;
    double _tx_sample_rate = 0.0;
    std::atomic<int32_t> _timing_advance{0};
    std::atomic<int32_t> _rx_alignment_shift{0};

    std::atomic<bool> _running{false};
    std::thread _ingest_thread;
    std::thread _mod_thread;
    std::thread _tx_thread;
    int _udp_sock = -1;
};

#endif // UPLINK_TX_ENGINE_HPP
