// Test fixture: CPU BS (SIM TX) -> quantized XR1710G ABI. No RF/device access.
#include "ShmRing.hpp"
#include "xr1710g/w1700k_stream.h"
#include <array>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>

static volatile sig_atomic_t stopped = 0;
static void stop(int) { stopped = 1; }
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    const std::string session = argv[1], shm = argv[2], directory = argv[3];
    constexpr size_t slice = 2048, capacity = 4096;
    constexpr double rate = 40e6, cfo = 5000.0;
    const size_t bytes = 4096 + capacity*8 + capacity*slice*4;
    sim_shm::ShmControl ctrl;
    sim_shm::ShmRing tx;
    ctrl.create(sim_shm::make_shm_name(session, "ctrl"), rate, 0, cfo, 0);
    tx.create(sim_shm::make_shm_name(session, "tx"), 1<<20);
    int fd = shm_open(shm.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0 || ftruncate(fd, bytes)) return 2;
    auto* h = static_cast<w1700k_stream_header*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);
    if (h == MAP_FAILED) return 3;
    std::memset(h, 0, bytes);
    h->magic = W1700K_STREAM_MAGIC; h->version = 1; h->header_bytes = 4096;
    h->slice_samples = slice; h->ring_slices = capacity; h->tags_offset = 4096;
    h->samples_offset = 4096+capacity*8; h->total_bytes = bytes; h->radio_sample_rate = rate;
    h->run = 1;
    std::signal(SIGINT, stop); std::signal(SIGTERM, stop);
    std::array<sim_shm::sample_t, slice> input{}, oscillator{};
    const double step = 6.2831853071795864769*cfo/rate;
    for (size_t i = 0; i < slice; ++i) oscillator[i] = {float(std::cos(step*i)), float(std::sin(step*i))};
    uint64_t index = 0, total = 0;
    double phase = 0;
    auto next = std::chrono::steady_clock::now();
    while (!stopped) {
        w1700k_stream_store(&h->heartbeat_ns, w1700k_stream_now_ns());
        const auto n = tx.pop_block(input.data(), slice, &ctrl.block()->running, .02);
        if (n != slice) continue;
        uint64_t kind = 0;
        if ((total/slice) % 64 == 0) {
            for (const auto& command : {"loss", "restart"}) {
                const auto path = std::filesystem::path(directory)/command;
                if (!std::filesystem::exists(path)) continue;
                std::filesystem::remove(path);
                if (std::string(command) == "loss") kind = W1700K_STREAM_TAG_LOST;
                else {
                    w1700k_stream_store(&h->position_seq, h->position_seq+1);
                    w1700k_stream_store(&h->run, h->run+1);
                    w1700k_stream_store(&h->newest, 0);
                    w1700k_stream_store(&h->position_seq, h->position_seq+1);
                    index = 0;
                }
                std::ofstream(path.string()+".done") << total << '\n';
            }
        }
        auto* tag = w1700k_stream_tags(h) + (index & (capacity-1));
        w1700k_stream_store(tag, 0);
        auto* samples = w1700k_stream_slice(h, index);
        const sim_shm::sample_t ph(float(std::cos(phase)), float(std::sin(phase)));
        for (size_t i = 0; i < slice; ++i) {
            const auto v = input[i] * oscillator[i] * ph * 4096.0f;
            samples[2*i] = static_cast<int16_t>(std::lround(std::clamp(v.real(), -8192.f, 8191.f)));
            samples[2*i+1] = static_cast<int16_t>(std::lround(std::clamp(v.imag(), -8192.f, 8191.f)));
        }
        phase = std::remainder(phase + step*slice, 6.2831853071795864769);
        w1700k_stream_store(tag, w1700k_stream_tag(h->run, index) | kind);
        w1700k_stream_store(&h->position_seq, h->position_seq+1);
        w1700k_stream_store(&h->newest, ++index);
        w1700k_stream_store(&h->position_seq, h->position_seq+1);
        total += slice;
        ctrl.set_sample_index(static_cast<uint64_t>(tx.timeline_origin()) + total);
        // This functional test is paced at 10 MS/s wall time. Metadata and
        // waveform geometry are 40 MS/s; it is not a throughput benchmark.
        next += std::chrono::nanoseconds(204800);
        std::this_thread::sleep_until(next);
        if (std::chrono::steady_clock::now() - next > std::chrono::milliseconds(10))
            next = std::chrono::steady_clock::now();
    }
    ctrl.stop(); ctrl.unlink(); tx.unlink();
    munmap(h, bytes); shm_unlink(shm.c_str());
    std::cout << "bridged_samples=" << total << '\n';
}
