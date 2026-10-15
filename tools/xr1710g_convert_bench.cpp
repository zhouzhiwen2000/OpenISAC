#include "xr1710g/convert.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>

namespace {
constexpr size_t N = 2048;
using Sample = std::complex<float>;
using Kernel = void (*)(const int16_t*, const Sample*, Sample*, size_t, float, float);
// A call boundary prevents elimination of identical successive blocks. Each
// variant pays the same call overhead; use native Release builds on the host.
__attribute__((noinline)) void portable(const int16_t* in, const Sample* wave, Sample* out,
                                      size_t n, float pi, float pq) {
    radio::xr1710g::rotate_portable(in, wave, out, n, pi, pq);
}
__attribute__((noinline)) void simd(const int16_t* in, const Sample* wave, Sample* out,
                                  size_t n, float pi, float pq) {
    radio::xr1710g::rotate(in, wave, out, n, pi, pq);
}
__attribute__((noinline)) void raw(const int16_t* in, const Sample*, Sample* out,
                                 size_t n, float, float) {
    radio::xr1710g::normalize(in, out, n);
}
void bench(const char* name, Kernel kernel) {
    std::array<int16_t, 2*N> in{};
    std::array<Sample, N> wave{}, out{};
    for (size_t i = 0; i < N; ++i) {
        in[2*i] = static_cast<int16_t>(i*13); in[2*i+1] = static_cast<int16_t>(-int(i*7));
        wave[i] = {float(std::cos(-0.001*i)), float(std::sin(-0.001*i))};
    }
    constexpr size_t blocks = 500000;
    const auto start = std::chrono::steady_clock::now();
    for (size_t k = 0; k < blocks; ++k) kernel(in.data(),wave.data(),out.data(),N,0.6f,0.8f);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << "{\"kernel\":\"" << name << "\",\"samples\":" << blocks*N
              << ",\"seconds\":" << seconds << ",\"samples_per_second\":" << blocks*N/seconds
              << ",\"check\":" << out[53].real() << "}\n";
}
}

int main() {
    bench("raw", raw);
    bench("portable_nco", portable);
    bench("simd_nco", simd);
}
