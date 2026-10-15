#include "Xr1710gBackend.hpp"
#include "xr1710g/w1700k_stream.h"
#include "xr1710g/convert.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cstring>
#include <fstream>

namespace {
constexpr size_t N = W1700K_STREAM_SLICE;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
template<class F> void rejects(F f) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    require(threw, "expected rejection");
}

struct Producer {
    std::string name;
    w1700k_stream_header* h = nullptr;
    size_t bytes = 4096 + 8*8 + 8*N*4;
    bool unlink_on_exit = true;
    explicit Producer(std::string supplied = {}) {
        static unsigned sequence = 0;
        name = supplied.empty() ? "/xr1710g-test-" + std::to_string(getpid()) + "-" + std::to_string(++sequence) : supplied;
        int fd = shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        require(fd >= 0, "create fixture");
        require(ftruncate(fd, bytes) == 0, "size fixture");
        h = static_cast<w1700k_stream_header*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        close(fd);
        require(h != MAP_FAILED, "map fixture");
        std::memset(h, 0, bytes);
        h->magic = W1700K_STREAM_MAGIC; h->version = 1; h->header_bytes = 4096;
        h->slice_samples = N; h->ring_slices = 8;
        h->tags_offset = 4096; h->samples_offset = 4096+64; h->total_bytes = bytes;
        h->radio_sample_rate = 40000000; h->run = 1;
        heartbeat();
    }
    ~Producer() { munmap(h, bytes); if (unlink_on_exit) shm_unlink(name.c_str()); }
    void heartbeat() { w1700k_stream_store(&h->heartbeat_ns, w1700k_stream_now_ns()); }
    void position(uint64_t run, uint64_t newest) {
        const uint64_t seq = h->position_seq;
        w1700k_stream_store(&h->position_seq, seq+1);
        w1700k_stream_store(&h->run, run);
        w1700k_stream_store(&h->newest, newest);
        w1700k_stream_store(&h->position_seq, seq+2);
    }
    void put(uint64_t index, uint64_t kind = 0, double tone_hz = 0.0) {
        uint64_t* tag = w1700k_stream_tags(h) + (index & 7);
        w1700k_stream_store(tag, 0);
        int16_t* samples = w1700k_stream_slice(h, index);
        for (size_t i = 0; i < N; ++i) {
            const double phase = 6.2831853071795864769 * tone_hz * (index*N+i) / 40e6;
            samples[2*i] = tone_hz ? static_cast<int16_t>(std::lround(4096*std::cos(phase))) : -8192;
            samples[2*i+1] = tone_hz ? static_cast<int16_t>(std::lround(4096*std::sin(phase))) : 8191;
        }
        w1700k_stream_store(tag, w1700k_stream_tag(h->run, index) | kind);
        position(h->run, index+1);
    }
    radio::DeviceConfig config() const {
        radio::DeviceConfig cfg;
        cfg.backend = "xr1710g";
        cfg.xr1710g = {name, 40e6, 2442e6, 0.0};
        return cfg;
    }
};
struct Receiver {
    radio::IDevicePtr dev;
    radio::IRxStreamPtr rx;
    explicit Receiver(const Producer& p) {
        dev = radio::make_xr1710g_device(p.config());
        rx = dev->get_rx_stream(radio::StreamArgs("fc32", "sc16"));
        rx->issue_stream_cmd(radio::StreamCmd(radio::StreamMode::StartContinuous));
    }
};

void samples_events_and_fragment_time() {
    Producer p; Receiver r(p);
    p.put(0); p.put(1, W1700K_STREAM_TAG_EVENT); p.put(2, W1700K_STREAM_TAG_FLAT);
    w1700k_stream_slice(p.h, 2)[0] = 32767; // corrected samples retain int16 headroom
    std::vector<radio::sample_t> out(2*N);
    radio::RxMetadata md;
    require(r.rx->recv(out.data(), 7, md, 0.01) == 7, "first fragment");
    require(md.has_time_spec && md.time_spec.to_ticks(40e6) == 0 && md.more_fragments && md.start_of_burst, "initial metadata");
    require(out[0] == radio::sample_t(-1.0f, 8191.0f/8192), "signed 14-bit normalization");
    require(r.rx->recv(out.data(), 2*N-7, md, 0.01) == 2*N-7, "cross-slice read skips events");
    require(md.time_spec.to_ticks(40e6) == 7 && md.fragment_offset == 7 && !md.more_fragments, "fragment time");
    require(out[N-7].real() == 32767.0f/8192, "corrected sample headroom is not clipped");
    require(r.dev->time_now().to_ticks(40e6) == 2*N, "events consume no sample time");
    p.put(3);
    require(r.rx->recv(out.data(), 2*N, md, 0.01, true) == N, "one_packet stops at slice boundary");
    require(md.error_code == radio::RxError::None, "one packet success");
    require(r.rx->recv(out.data(), 1, md, 0.001) == 0 && md.error_code == radio::RxError::Timeout, "bounded no-data timeout");
    require(p.h->restart_requests == 0 && p.h->flatten == 0, "consumer never writes control");
}

void loss_overrun_and_run_change() {
    Producer p; Receiver r(p);
    radio::RxMetadata md; std::array<radio::sample_t, N*2> out{};
    p.put(0); p.put(1, W1700K_STREAM_TAG_LOST); p.put(2);
    require(r.rx->recv(out.data(), 2*N, md, .01) == 0 && md.error_code == radio::RxError::Overflow && md.out_of_sequence, "drop entire read on loss");
    require(r.rx->recv(out.data(), N, md, .01) == N && md.time_spec.to_ticks(40e6) == 2*N, "resume after loss with gap");
    for (size_t i = 3; i < 13; ++i) p.put(i);
    require(r.rx->recv(out.data(), N, md, .01) == 0 && md.error_code == radio::RxError::Overflow, "ring overrun");
    p.put(13);
    require(r.rx->recv(out.data(), N, md, .01) == N, "resume at live frontier");
    p.position(2, 0);
    require(r.rx->recv(out.data(), N, md, .01) == 0 && md.error_code == radio::RxError::BrokenChain, "run change");
    p.put(0);
    require(r.rx->recv(out.data(), N, md, .01) == N, "new run continues");
    const auto before = md.time_spec.to_ticks(40e6);
    r.rx->issue_stream_cmd(radio::StreamCmd(radio::StreamMode::StopContinuous));
    p.put(1);
    require(r.rx->recv(out.data(), N, md, .001) == 0, "stopped consumer");
    r.rx->issue_stream_cmd(radio::StreamCmd(radio::StreamMode::StartContinuous));
    p.put(2);
    require(r.rx->recv(out.data(), N, md, .01) == N && md.time_spec.to_ticks(40e6) > before, "consumer restart preserves local time");
}

void software_cfo() {
    for (bool manual : {false, true}) for (double hz : {2.5e6, -2.5e6}) {
        Producer p; Receiver r(p);
        radio::TuneRequest tune(2442e6 + hz);
        if (manual) {
            tune.rf_freq_policy = radio::TunePolicy::Manual;
            tune.rf_freq = 2442e6;
            tune.dsp_freq_policy = radio::TunePolicy::Manual;
            tune.dsp_freq = -hz; // Existing UE sends -measured_CFO, as UHD requires.
        }
        const auto result = r.dev->set_rx_freq(tune);
        require(result.actual_rf_freq == 2442e6 && result.actual_dsp_freq == -hz, "UHD RX RF-minus-DSP tune result");
        require(r.dev->rx_freq_correction() == hz, "physical correction opposes UHD RX DSP sign");
        tune.dsp_freq_policy = radio::TunePolicy::None;
        require(r.dev->set_rx_freq(tune).actual_dsp_freq == -hz, "DSP None retains correction sign");
        p.put(0, 0, hz); p.put(1, 0, hz);
        std::vector<radio::sample_t> out(2*N); radio::RxMetadata md;
        require(r.rx->recv(out.data(), 13, md, .01) == 13, "CFO first fragment");
        require(r.rx->recv(out.data()+13, 2*N-13, md, .01) == 2*N-13, "CFO remainder");
        for (auto sample : out) require(std::abs(sample-radio::sample_t(.5f, 0.f)) < .0002f, "CFO sign and phase continuity");
        tune.rf_freq_policy = radio::TunePolicy::Manual;
        tune.rf_freq = 2442e6; tune.dsp_freq_policy = radio::TunePolicy::Manual; tune.dsp_freq = 0;
        r.dev->set_rx_freq(tune);
        require(r.dev->rx_freq_correction() == 0, "CFO reset");
    }
}

void pll_cfo_preserves_samples() {
    const std::string helper = "/tmp/xr1710g-pll-test-" + std::to_string(getpid()) + ".py";
    struct Remove { std::string p; ~Remove() { unlink(p.c_str()); } } remove{helper};
    std::ofstream file(helper);
    file << R"PY(import sys
step = 14.30511474609375
for line in sys.stdin:
    fields = line.split(); op, seq = fields[:2]
    if op == 'OPEN':
        print('OK', seq, 0, step, flush=True)
    elif op == 'SET':
        if len(sys.argv) > 1:
            print('OK', int(seq)-1, 0, step, flush=True)
        else:
            print('OK', seq, round(float(fields[2])/step)*step, step, flush=True)
)PY";
    file.close();
    Producer p;
    auto cfg = p.config(); cfg.xr1710g.pll_command = "python3 " + helper;
    auto dev = radio::make_xr1710g_device(cfg);
    auto rx = dev->get_rx_stream(radio::StreamArgs());
    rx->issue_stream_cmd(radio::StreamCmd(radio::StreamMode::StartContinuous));
    p.put(0); p.put(1);
    std::array<radio::sample_t, N*2> samples{}; radio::RxMetadata md;
    require(rx->recv(samples.data(), 7, md, .01) == 7, "PLL first fragment");
    radio::TuneRequest req(2442e6 + 100);
    req.rf_freq = 2442e6; req.rf_freq_policy = radio::TunePolicy::Manual;
    req.dsp_freq = 100; req.dsp_freq_policy = radio::TunePolicy::Manual;
    const auto first = dev->set_rx_freq(req);
    require(first.actual_rf_freq == 2442e6 && std::abs(first.actual_dsp_freq - 100.13580322265625) < 1e-9,
            "PLL tune reports applied offset with existing logical RF/DSP semantics");
    require(dev->rx_freq_correction() == -first.actual_dsp_freq,
            "PLL physical LO shift opposes UHD RX DSP sign");
    require(rx->recv(samples.data(), N*2-7, md, .01) == N*2-7 && md.error_code == radio::RxError::None,
            "PLL update does not discard cached or queued IQ");
    require(md.time_spec.to_ticks(40e6) == 7 && !md.start_of_burst, "PLL update preserves sample timeline");
    for (size_t i = 0; i < N*2-7; ++i)
        require(samples[i] == radio::sample_t(-1.0f, 8191.0f/8192), "PLL mode never also applies software NCO");
    const auto again = dev->set_rx_freq(req);
    require(again.actual_dsp_freq == first.actual_dsp_freq, "PLL correction is total, not incremental");
    dev->set_rx_freq_correction(-100);
    require(dev->rx_freq_correction() == -first.actual_dsp_freq, "negative PLL correction");
    dev->set_rx_freq_correction(0);
    require(dev->rx_freq_correction() == 0, "PLL zero correction");
    rejects([&] { dev->set_rx_freq_correction(100001); });
    rejects([&] { dev->set_rx_freq_correction(std::numeric_limits<double>::infinity()); });
    require(dev->rx_freq_correction() == 0, "invalid PLL request leaves acknowledged state unchanged");
    rx.reset(); dev.reset();
    cfg.xr1710g.pll_command += " stale";
    auto stale = radio::make_xr1710g_device(cfg);
    rejects([&] { stale->set_rx_freq_correction(100); });
    require(stale->rx_freq_correction() == 0, "stale acknowledgement never takes effect");
    rejects([&] { stale->set_rx_freq_correction(0); });
}

void conversion_matches_double_reference() {
    // Exercise unaligned input/output, int16 headroom, SIMD tails and arbitrary
    // initial phase; the reference uses double complex multiplication.
    std::mt19937 rng(1710);
    std::uniform_int_distribution<int> iq(-32768, 32767);
    std::array<int16_t, 2*(N+4)> input{};
    std::array<radio::sample_t, N+4> wave{}, out{};
    for (auto& x : input) x = static_cast<int16_t>(iq(rng));
    input[2] = -32768; input[3] = 32767;
    for (double step : {0.0, 0.000123, -0.17, 3.13, -3.13}) {
        for (size_t i = 0; i < wave.size(); ++i)
            wave[i] = radio::sample_t(std::cos(step*i), std::sin(step*i));
        for (size_t n : {size_t(0), size_t(1), size_t(3), size_t(4), size_t(7), size_t(13), N-1, N}) {
            const float pi = std::cos(1.234), pq = std::sin(1.234);
            out.fill({123, 456});
            radio::xr1710g::rotate(input.data()+2, wave.data()+1, out.data()+1, n, pi, pq);
            require(out[0] == radio::sample_t(123,456) && out[n+1] == radio::sample_t(123,456), "SIMD output bounds");
            for (size_t i = 0; i < n; ++i) {
                const auto expected = std::complex<double>(input[2*i+2], input[2*i+3]) *
                    std::complex<double>(wave[i+1]) * std::complex<double>(pi,pq) / 8192.0;
                require(std::abs(std::complex<double>(out[i+1])-expected) < 1.5e-6, "SIMD matches double reference");
            }
            radio::xr1710g::normalize(input.data()+2, out.data()+1, n);
            for (size_t i = 0; i < n; ++i)
                require(out[i+1] == radio::sample_t(input[2*i+2]/8192.0f,input[2*i+3]/8192.0f), "normalization exact");
        }
    }
}

void service_restart_and_stuck_writer() {
    Producer p; Receiver r(p); radio::RxMetadata md; radio::sample_t out;
    w1700k_stream_store(&p.h->heartbeat_ns, 0);
    require(r.rx->recv(&out, 1, md, .001) == 0 && md.error_code == radio::RxError::BrokenChain, "stale heartbeat");
    require(r.rx->recv(&out, 1, md, .001) == 0 && md.error_code == radio::RxError::Timeout, "stale service does not spin errors");
    shm_unlink(p.name.c_str()); p.unlink_on_exit = false;
    Producer replacement(p.name);
    std::this_thread::sleep_for(std::chrono::milliseconds(110));
    require(r.rx->recv(&out, 1, md, .01) == 0 && md.error_code == radio::RxError::BrokenChain, "replacement inode");
    r.rx->issue_stream_cmd(radio::StreamCmd(radio::StreamMode::StartContinuous));
    replacement.put(0);
    require(r.rx->recv(&out, 1, md, .01) == 1, "new producer samples");
    r.rx->issue_stream_cmd(radio::StreamCmd(radio::StreamMode::StopContinuous));
    replacement.h->position_seq |= 1;
    r.rx->issue_stream_cmd(radio::StreamCmd(radio::StreamMode::StartContinuous));
    const auto begin = std::chrono::steady_clock::now();
    require(r.rx->recv(&out, 1, md, .01) == 0 && md.error_code == radio::RxError::Timeout, "odd seqlock timeout");
    require(std::chrono::steady_clock::now() - begin < std::chrono::seconds(1), "stuck writer bounded");
}

void concurrent_heartbeat_does_not_report_restart() {
    Producer p; Receiver r(p);
    std::atomic<bool> stop{false};
    std::thread heartbeat([&] { while (!stop.load()) p.heartbeat(); });
    bool failed = false;
    for (int i = 0; i < 50000; ++i) {
        radio::sample_t sample; radio::RxMetadata md;
        r.rx->recv(&sample, 1, md, 0.0);
        if (md.error_code != radio::RxError::Timeout) { failed = true; break; }
    }
    stop.store(true);
    heartbeat.join();
    require(!failed, "concurrent heartbeat must not look like service death");
}

void producer_initialization_is_awaited() {
    Producer p;
    w1700k_stream_store(&p.h->magic, 0);
    auto cfg = p.config();
    cfg.xr1710g.attach_timeout_s = .5;
    std::thread initialize([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        p.heartbeat();
        w1700k_stream_store(&p.h->magic, W1700K_STREAM_MAGIC);
    });
    auto dev = radio::make_xr1710g_device(cfg);
    radio::IRxStreamPtr rx;
    try { rx = dev->get_rx_stream(radio::StreamArgs()); }
    catch (...) { initialize.join(); throw; }
    initialize.join();
    require(bool(rx), "attach waits for producer magic publication");
}

void configuration_and_layout_rejections() {
    Producer p; auto cfg = p.config();
    auto invalid = cfg; invalid.xr1710g.sample_rate = 50e6;
    rejects([&] { radio::make_xr1710g_device(invalid); });
    invalid = cfg; invalid.xr1710g.center_freq = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { radio::make_xr1710g_device(invalid); });
    invalid = cfg; invalid.clock_source = "external";
    rejects([&] { radio::make_xr1710g_device(invalid); });
    invalid = cfg; invalid.xr1710g.shm_name = "/dev/shm/bad";
    rejects([&] { radio::make_xr1710g_device(invalid); });
    invalid = cfg; invalid.xr1710g.shm_name = "/xr1710g-missing-" + std::to_string(getpid());
    rejects([&] { auto d = radio::make_xr1710g_device(invalid); d->get_rx_stream(radio::StreamArgs()); });
    for (auto mutate : std::vector<std::function<void()>>{
            [&] { p.h->version = 2; }, [&] { p.h->ring_slices = 3; },
            [&] { p.h->tags_offset = UINT64_MAX; }, [&] { p.h->samples_offset = p.bytes; },
            [&] { p.h->header_bytes = 32; }, [&] { p.h->samples_offset = 4096; }}) {
        const auto saved = *p.h;
        mutate();
        rejects([&] { Receiver r(p); });
        *p.h = saved;
    }
    p.h->radio_sample_rate = 80e6;
    rejects([&] { Receiver r(p); });
    p.h->radio_sample_rate = 40e6;
    Receiver r(p);
    rejects([&] { r.dev->get_tx_stream(radio::StreamArgs()); });
    rejects([&] { r.dev->get_tx_rate(); });
    rejects([&] { r.dev->set_rx_rate(50e6); });
    rejects([&] { r.dev->set_rx_freq_correction(20e6); });
    rejects([&] { r.dev->set_rx_freq_correction(std::numeric_limits<double>::quiet_NaN()); });
    rejects([&] { r.dev->get_rx_rate(1); });
    radio::StreamCmd timed; timed.stream_now = false;
    rejects([&] { r.rx->issue_stream_cmd(timed); });
    rejects([&] { r.rx->issue_stream_cmd(radio::StreamCmd(radio::StreamMode::NumSampsAndDone)); });
    radio::StreamArgs multi; multi.channels = {0, 1};
    rejects([&] { r.dev->get_rx_stream(multi); });
    require(!r.dev->supports(radio::Capability::PpsTimeSync) && !r.dev->supports(radio::Capability::TimedTx) &&
            !r.dev->supports(radio::Capability::HardwareGain) && r.dev->get_tx_num_channels() == 0, "capabilities are honest");
    radio::sample_t out; radio::RxMetadata md;
    p.h->radio_sample_rate = 80e6;
    require(r.rx->recv(&out, 1, md, .01) == 0 && md.error_code == radio::RxError::BadPacket, "live rate change");
    require(r.rx->recv(&out, 1, md, .001) == 0 && md.error_code == radio::RxError::Timeout, "hold incompatible rate");
    p.h->radio_sample_rate = 40e6;
    require(r.rx->recv(&out, 1, md, .01) == 0 && md.error_code == radio::RxError::BrokenChain, "rate restoration discontinuity");
}
} // namespace

int main() {
    try {
        samples_events_and_fragment_time();
        loss_overrun_and_run_change();
        software_cfo();
        pll_cfo_preserves_samples();
        conversion_matches_double_reference();
        service_restart_and_stuck_writer();
        concurrent_heartbeat_does_not_report_restart();
        producer_initialization_is_awaited();
        configuration_and_layout_rejections();
        std::cout << "XR1710G backend: samples, timestamps, gaps, CFO, restart and validation passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
