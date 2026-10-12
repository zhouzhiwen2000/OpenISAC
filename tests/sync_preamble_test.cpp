#include "OFDMCore.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
constexpr size_t kFft = 1024, kCp = 128, kSymbol = kFft + kCp;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

AlignedVector time_symbol(AlignedVector frequency) {
    AlignedVector useful(kFft), symbol(kSymbol);
    auto plan = fftwf_plan_dft_1d(kFft,
        reinterpret_cast<fftwf_complex*>(frequency.data()),
        reinterpret_cast<fftwf_complex*>(useful.data()), FFTW_BACKWARD, FFTW_ESTIMATE);
    require(plan != nullptr, "FFTW plan failed");
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    for (size_t i = 0; i < kSymbol; ++i)
        symbol[i] = useful[(i + kFft - kCp) % kFft] / std::sqrt(float(kFft));
    return symbol;
}

SyncProcessor::PreambleValidation validate(const AlignedVector& data, size_t primary,
                                          bool repeated = true, bool training = true) {
    return SyncProcessor::validate_sync_preamble(
        data, primary, kFft, kCp, repeated, training, 16, 100e6);
}
} // namespace

int main() {
    try {
        const auto zc = time_symbol(generate_zc_freq(kFft, 29));
        const auto training = time_symbol(generate_cfo_training_freq(kFft, 16));
        AlignedVector complete(8 * kSymbol, {0, 0});
        const size_t first = 2 * kSymbol, primary = first + kSymbol;
        std::copy(zc.begin(), zc.end(), complete.begin() + first);
        std::copy(zc.begin(), zc.end(), complete.begin() + primary);
        std::copy(training.begin(), training.end(), complete.begin() + primary + kSymbol);
        require(validate(complete, primary).valid, "complete preamble rejected");
        require(!validate(complete, first).valid, "duplicate ZC accepted as main ZC");
        require(!validate(complete, first, true, false).valid,
                "duplicate ZC accepted with training disabled");
        require(!validate(complete, first, false, true).valid,
                "ZC mistaken for CFO field when repeated-ZC checking disabled");
        require(validate(complete, first, false, false).valid, "legacy ZC-only configuration rejected");
        require(validate(complete, primary, true, false).valid, "repeated-ZC-only configuration rejected");
        require(validate(complete, primary, false, true).valid, "CFO-field-only configuration rejected");

        for (size_t tail : {size_t(0), size_t(64), kSymbol, kSymbol + 64, 2 * kSymbol - 1}) {
            AlignedVector partial(complete.begin(), complete.begin() + primary + tail);
            require(!validate(partial, first).valid, "partial preamble accepted at duplicate ZC");
            require(!validate(partial, primary).valid, "partial preamble accepted before complete training");
        }
        AlignedVector noisy = complete;
        std::mt19937 generator(42);
        std::normal_distribution<float> noise(0.0f, .01f);
        for (size_t i = 0; i < noisy.size(); ++i)
            noisy[i] = noisy[i] * std::polar(1.0f, float(2 * M_PI * 300000.0 * i / 100e6)) +
                       std::complex<float>(noise(generator), noise(generator));
        require(validate(noisy, primary).valid, "complete preamble with noise/CFO rejected");
        require(!validate(noisy, first).valid, "duplicate ZC with noise/CFO accepted");

        // RX tracking leaves the physical delay inside the CP. The check must
        // tolerate that offset rather than resetting a correctly aligned link.
        for (int shift : {-40, -20, 0, 20, 40}) {
            AlignedVector delayed(noisy.size(), {0, 0});
            for (size_t i = 0; i < noisy.size(); ++i) {
                const auto index = static_cast<int64_t>(i) + shift;
                if (index >= 0 && index < static_cast<int64_t>(delayed.size()))
                    delayed[static_cast<size_t>(index)] = noisy[i];
            }
            require(validate(delayed, primary).valid, "normal CP timing offset rejected");
            require(!validate(delayed, primary - kSymbol).valid, "one-symbol frame slip accepted");
        }
        require(!validate(AlignedVector(8 * kSymbol, {0, 0}), primary).valid, "silence accepted");

        constexpr int kRepeats = 10000;
        int accepted = 0;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < kRepeats; ++i) accepted += validate(noisy, primary).valid;
        const double microseconds = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - begin).count() / kRepeats;
        require(accepted == kRepeats, "repeated validation was unstable");
        std::cout << "Sync preamble regression passed; validation_us=" << microseconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
