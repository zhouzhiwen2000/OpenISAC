#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "DigitalModulation.hpp"

// Keep the packet-level call boundary fixed across before/after builds. This
// prevents benchmark-loop inlining decisions from changing the DSP being timed.
template<typename Llr>
__attribute__((noinline)) void demap_packet(const DigitalModulation& modem,
    const Llr* input, size_t bits, float scale, Llr* output) {
    modem.demap_scaled_pairs(input, bits, scale, output, 1008);
}

__attribute__((noinline)) void pack_packet(const DigitalModulation& modem,
    const std::vector<int>& bits, std::vector<int>& tokens) {
    modem.pack_bits(bits, tokens, 1008);
}

template<typename Function>
double measure_us(Function function) {
    using Clock = std::chrono::steady_clock;
    for (int i = 0; i < 32; ++i) function();
    auto start = Clock::now();
    for (int i = 0; i < 32; ++i) function();
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count() / 32;
    const int iterations = std::clamp(static_cast<int>(0.02 / std::max(seconds, 1e-9)), 32, 200000);
    std::vector<double> times;
    for (int repeat = 0; repeat < 7; ++repeat) {
        start = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            function();
            std::atomic_signal_fence(std::memory_order_seq_cst);
        }
        times.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / iterations);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main() {
    constexpr size_t code_bits = 21 * 1008; // A 1316-byte UDP payload, rate 1/2.
    double checksum = 0;
    std::cout << "mode,pack_us,demap_float_us,demap_q16_us\n";
    for (const char* name : {"BPSK", "QPSK", "8PSK", "16QAM", "32QAM", "64QAM"}) {
        const DigitalModulation modem(parse_modulation_scheme(name));
        std::mt19937 random(984);
        std::normal_distribution<float> noise(0, 0.025f);
        std::vector<int> bits(code_bits), tokens;
        for (int& bit : bits) bit = random() & 1;
        modem.pack_bits(bits, tokens, 1008);
        std::vector<float> fp_pairs(tokens.size() * 2), fp_output(code_bits);
        std::vector<int16_t> q_pairs(tokens.size() * 2), q_output(code_bits);
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto point = modem.modulate_token(tokens[i]) + std::complex<float>{noise(random), noise(random)};
            fp_pairs[i * 2] = point.real() * 4096;
            fp_pairs[i * 2 + 1] = point.imag() * 4096;
            q_pairs[i * 2] = static_cast<int16_t>(std::round(fp_pairs[i * 2]));
            q_pairs[i * 2 + 1] = static_cast<int16_t>(std::round(fp_pairs[i * 2 + 1]));
        }
        const double pack = measure_us([&] { pack_packet(modem, bits, tokens); });
        const double fp = measure_us([&] {
            demap_packet(modem, fp_pairs.data(), code_bits, 4096, fp_output.data());
        });
        const double q16 = measure_us([&] {
            demap_packet(modem, q_pairs.data(), code_bits, 4096, q_output.data());
        });
        checksum += tokens[13] + fp_output[13] + q_output[13];
        std::cout << name << ',' << std::fixed << std::setprecision(3)
                  << pack << ',' << fp << ',' << q16 << '\n';
    }
    std::cerr << "checksum=" << checksum << '\n';
}
