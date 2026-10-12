// Standalone single-thread CPU LDPC decoder precision comparison:
//   <int,float>  (float32, current production decoder)
//   <short,short>        (Q16 fixed-point)
//   <signed char,signed char> (Q8 fixed-point)
// Same (1008,504) H and horizontal-layered decoding, with configurable
// maximum iterations, NMS factor and throughput Es/N0. Reports BER/FER and
// median single-thread information-bit throughput (decode only).
//
// This file is additive and does not touch the production float path or the
// CUDA FP16 decoder. Fixed-point input is produced by a plain pow2 quantizer
// of the float LLRs (q = sat(round(llr * 2^fp))).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <aff3ct.hpp>

using namespace aff3ct;

namespace {
constexpr int K = 504;
constexpr int N = 1008;
const std::string HPATH = "../LDPC_504_1008.alist";
volatile uint64_t quantized_checksum = 0;

bool finite_value(double value) {
    // Also works when the caller builds with -ffast-math.
    uint64_t bits = 0;
    static_assert(sizeof bits == sizeof value);
    std::memcpy(&bits, &value, sizeof bits);
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

int parse_int(const char* value) {
    size_t end = 0;
    const int parsed = std::stoi(value, &end);
    if (value[end] != '\0') throw std::invalid_argument("invalid integer argument");
    return parsed;
}

double parse_double(const char* value) {
    size_t end = 0;
    const double parsed = std::stod(value, &end);
    if (value[end] != '\0' || !finite_value(parsed))
        throw std::invalid_argument("expected a finite numeric argument");
    return parsed;
}

double median_seconds(std::vector<double> elapsed) {
    std::sort(elapsed.begin(), elapsed.end());
    const size_t middle = elapsed.size() / 2;
    return elapsed.size() % 2 ? elapsed[middle] :
        (elapsed[middle - 1] + elapsed[middle]) / 2.0;
}

void print_usage(const char* program) {
    std::printf("Usage: %s [frames=8192] [q16_scale=64] [q8_scale=4] "
                "[iterations=12] [nms=0.75] [timing_esn0_db=2] [reps=5]\n"
                "Run from build/. Frames must be a positive multiple of 64; "
                "NMS must be a multiple of 0.125 in (0,1] for Q16.\n"
                "Rates count 504 information bits per codeword. "
                "Es/N0 uses energy per coded BPSK bit; Eb/N0 = Es/N0 + 3.01 dB.\n",
                program);
}

template <typename B, typename Q>
struct Codec {
    std::unique_ptr<factory::Codec_LDPC> fac;
    std::unique_ptr<tools::Codec_LDPC<B, Q>> codec;
};

template <typename B, typename Q>
Codec<B, Q> make_codec(int n_frames, int iterations, double nms,
                       const std::string& implem = "NMS") {
    std::vector<std::string> args = {
        "prog", "--enc-type", "LDPC_H", "--enc-g-method", "IDENTITY",
        "--dec-type", "BP_HORIZONTAL_LAYERED", "--dec-implem", implem,
        "--dec-ite", std::to_string(iterations), "--dec-synd-depth", "1",
        "--dec-simd", "INTER", "--dec-h-path", HPATH};
    if (implem == "NMS") {
        args.insert(args.end(), {"--dec-norm", std::to_string(nms)});
    }
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());

    Codec<B, Q> c;
    c.fac = std::make_unique<factory::Codec_LDPC>();
    std::vector<factory::Factory*> pl{c.fac.get()};
    tools::Command_parser cp(static_cast<int>(argv.size()), argv.data(), pl, true);
    c.codec.reset(c.fac->template build<B, Q>());
    c.codec->set_n_frames(n_frames);
    c.codec->get_decoder_siho().reset();
    return c;
}

double snr_to_sigma(double snr_db) {
    const double snr_lin = std::pow(10.0, snr_db / 10.0);
    return std::sqrt(1.0 / (2.0 * snr_lin)); // rate-1/2, Es=1 per coded bit
}

// Decode `total` independent frames already laid out [total][N], in groups of
// n_frames. Correctness is checked separately from repeated timing passes.
template <typename B, typename Q>
struct Result { double ber; double fer; double mbps; double best_mbps; long frame_errors; };

template <typename B, typename Q>
Result<B, Q> decode_all(Codec<B, Q>& cf, const std::vector<Q>& llr,
                        const std::vector<int>& info, int total, int n_frames,
                        int reps, bool time_it) {
    auto& dec = cf.codec->get_decoder_siho();
    const int groups = total / n_frames;
    std::vector<B> out(static_cast<size_t>(total) * K);

    // correctness pass (single)
    for (int g = 0; g < groups; ++g) {
        dec.decode_siho(llr.data() + static_cast<size_t>(g) * n_frames * N,
                        out.data() + static_cast<size_t>(g) * n_frames * K, -1, false);
    }
    long bit_err = 0, frame_err = 0;
    for (int f = 0; f < total; ++f) {
        bool ferr = false;
        for (int k = 0; k < K; ++k) {
            const int got = static_cast<int>(out[static_cast<size_t>(f) * K + k]) & 1;
            if (got != info[static_cast<size_t>(f) * K + k]) { ++bit_err; ferr = true; }
        }
        if (ferr) ++frame_err;
    }
    double mbps = -1.0, best_mbps = -1.0;
    if (time_it) {
        std::vector<double> elapsed;
        elapsed.reserve(reps);
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int g = 0; g < groups; ++g) {
                dec.decode_siho(llr.data() + static_cast<size_t>(g) * n_frames * N,
                                out.data() + static_cast<size_t>(g) * n_frames * K, -1, false);
            }
            const auto t1 = std::chrono::steady_clock::now();
            elapsed.push_back(std::chrono::duration<double>(t1 - t0).count());
        }
        mbps = static_cast<double>(total) * K / median_seconds(elapsed) / 1e6;
        best_mbps = static_cast<double>(total) * K /
            *std::min_element(elapsed.begin(), elapsed.end()) / 1e6;
    }
    return {static_cast<double>(bit_err) / (static_cast<double>(total) * K),
            static_cast<double>(frame_err) / total, mbps, best_mbps, frame_err};
}

// Branchless pow2 quantizer; `f` is float-clamp friendly so the compiler can
// vectorize the whole loop under AVX-512.
template <typename Q>
inline void quantize_into(const float* in, Q* out, size_t n, float scale, float sat) {
    for (size_t i = 0; i < n; ++i) {
        float v = std::nearbyintf(in[i] * scale);
        v = v > sat ? sat : (v < -sat ? -sat : v);
        out[i] = static_cast<Q>(v);
    }
}

template <typename Q>
std::vector<Q> quantize(const std::vector<float>& llr, double scale, int sat) {
    std::vector<Q> q(llr.size());
    quantize_into<Q>(llr.data(), q.data(), llr.size(),
                     static_cast<float>(scale), static_cast<float>(sat));
    return q;
}

// info-bit-rate-equivalent throughput of the quantization step alone.
template <typename Q>
double time_quant_mbps(const std::vector<float>& llr, double scale, int sat,
                       int reps, int total) {
    std::vector<Q> out(llr.size());
    std::vector<double> elapsed;
    elapsed.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        quantize_into<Q>(llr.data(), out.data(), llr.size(),
                         static_cast<float>(scale), static_cast<float>(sat));
        const auto t1 = std::chrono::steady_clock::now();
        elapsed.push_back(std::chrono::duration<double>(t1 - t0).count());
        // Consume all output outside the timed region to prevent dead stores.
        uint64_t checksum = 0;
        for (Q value : out) checksum += static_cast<uint64_t>(value);
        quantized_checksum ^= checksum;
    }
    return static_cast<double>(total) * K / median_seconds(elapsed) / 1e6;
}
} // namespace

int main(int argc, char** argv) try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return 0;
    }
    if (argc > 8) throw std::invalid_argument("too many arguments");
    const int TOTAL = (argc > 1) ? parse_int(argv[1]) : 8192;
    const double q16_scale = (argc > 2) ? parse_double(argv[2]) : 64.0;
    const double q8_scale = (argc > 3) ? parse_double(argv[3]) : 4.0;
    const int iterations = (argc > 4) ? parse_int(argv[4]) : 12;
    const double nms = (argc > 5) ? parse_double(argv[5]) : 0.75;
    const double thr_snr = (argc > 6) ? parse_double(argv[6]) : 2.0;
    const int reps = (argc > 7) ? parse_int(argv[7]) : 5;
    if (TOTAL <= 0 || TOTAL % 64 != 0)
        throw std::invalid_argument("frames must be a positive multiple of 64");
    if (iterations <= 0 || reps <= 0)
        throw std::invalid_argument("iterations and repetitions must be positive");
    if (q16_scale <= 0 || q8_scale <= 0 ||
        q16_scale > std::numeric_limits<float>::max() ||
        q8_scale > std::numeric_limits<float>::max())
        throw std::invalid_argument("quantization scales must be positive finite floats");
    if (nms <= 0 || nms > 1 || nms * 8 != std::round(nms * 8))
        throw std::invalid_argument("Q16 NMS must be a multiple of 0.125 in (0,1]");
    if (thr_snr < -100 || thr_snr > 100)
        throw std::invalid_argument("timing Es/N0 must be in [-100,100] dB");

    auto cf32 = make_codec<int, float>(16, iterations, nms);
    auto cf16 = make_codec<short, short>(32, iterations, nms);
    auto cf8 = make_codec<signed char, signed char>(64, iterations, nms, "MS");

    auto& enc = cf32.codec->get_encoder();

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> bit(0, 1);

    // Generate TOTAL frames: random info -> codeword bits (via float codec encoder).
    std::vector<int> info(static_cast<size_t>(TOTAL) * K);
    std::vector<int> cw(static_cast<size_t>(TOTAL) * N);
    for (int g = 0; g + 16 <= TOTAL; g += 16) {
        for (int j = 0; j < 16 * K; ++j) info[static_cast<size_t>(g) * K + j] = bit(rng);
        enc.encode(info.data() + static_cast<size_t>(g) * K,
                   cw.data() + static_cast<size_t>(g) * N, -1, false);
    }

    std::printf("CPU LDPC(1008,504) precision comparison  TOTAL=%d  q16_scale=%.1f q8_scale=%.1f\n",
                TOTAL, q16_scale, q8_scale);
    std::printf("iterations=%d nms=%.3f syndrome_early_stop=on reps=%d "
                "statistic=median seed=12345 (Q8 uses MS without normalization)\n",
                iterations, nms, reps);
    std::printf("\n== BER / FER vs Es/N0 (coded BPSK-bit energy) ==\n");
    std::printf("%6s | %-22s | %-22s | %-22s\n", "Es/N0", "float32/NMS", "Q16/NMS", "Q8/MS");

    Result<int, float> r32_thr{}; Result<short, short> r16_thr{}; Result<signed char, signed char> r8_thr{};
    double q16_quant_mbps = -1.0, q8_quant_mbps = -1.0;

    std::vector<double> snrs{1.0, 1.5, 2.0, 2.5, 3.0, 4.0, thr_snr};
    std::sort(snrs.begin(), snrs.end());
    snrs.erase(std::unique(snrs.begin(), snrs.end()), snrs.end());
    for (double snr : snrs) {
        const double sigma = snr_to_sigma(snr);
        const double inv_s2 = 1.0 / (sigma * sigma);
        std::normal_distribution<float> noise(0.0f, static_cast<float>(sigma));
        std::vector<float> llr(static_cast<size_t>(TOTAL) * N);
        for (size_t i = 0; i < llr.size(); ++i) {
            const float x = (cw[i] & 1) ? -1.0f : 1.0f; // bit0->+1
            const float y = x + noise(rng);
            llr[i] = static_cast<float>(2.0 * y * inv_s2);
        }
        auto llr16 = quantize<short>(llr, q16_scale, 32767);
        auto llr8  = quantize<signed char>(llr, q8_scale, 127);

        const bool timed = std::abs(snr - thr_snr) < 1e-9;
        auto r32 = decode_all<int, float>(cf32, llr, info, TOTAL, 16, reps, timed);
        auto r16 = decode_all<short, short>(cf16, llr16, info, TOTAL, 32, reps, timed);
        auto r8  = decode_all<signed char, signed char>(cf8, llr8, info, TOTAL, 64, reps, timed);
        if (timed) {
            r32_thr = r32; r16_thr = r16; r8_thr = r8;
            q16_quant_mbps = time_quant_mbps<short>(llr, q16_scale, 32767, reps, TOTAL);
            q8_quant_mbps = time_quant_mbps<signed char>(llr, q8_scale, 127, reps, TOTAL);
        }

        char b32[64], b16[64], b8[64];
        std::snprintf(b32, sizeof b32, "BER=%.2e FER=%.2e", r32.ber, r32.fer);
        std::snprintf(b16, sizeof b16, "BER=%.2e FER=%.2e", r16.ber, r16.fer);
        std::snprintf(b8,  sizeof b8,  "BER=%.2e FER=%.2e", r8.ber, r8.fer);
        std::printf("%6.2f | %-22s | %-22s | %-22s\n", snr, b32, b16, b8);
    }

    auto combined = [](double dec, double quant) {
        return (dec > 0 && quant > 0) ? 1.0 / (1.0 / dec + 1.0 / quant) : dec;
    };
    const double q16_eff = combined(r16_thr.mbps, q16_quant_mbps);
    const double q8_eff = combined(r8_thr.mbps, q8_quant_mbps);

    std::printf("\n== single-thread median information-bit throughput @ Es/N0=%.2f dB ==\n", thr_snr);
    std::printf("%-14s %-12s %-12s %-14s %-9s %-9s\n",
                "precision", "decode_Mbps", "quant_Mbps", "dec+quant_Mbps", "vs_flt", "qnt_ovh");
    std::printf("%-14s %-12.1f %-12s %-14.1f %-9.2f %-9s\n",
                "float32/NMS", r32_thr.mbps, "-", r32_thr.mbps, 1.0, "-");
    std::printf("%-14s %-12.1f %-12.1f %-14.1f %-9.2f %-8.1f%%\n",
                "Q16/NMS", r16_thr.mbps, q16_quant_mbps, q16_eff,
                q16_eff / r32_thr.mbps, 100.0 * (r16_thr.mbps - q16_eff) / r16_thr.mbps);
    std::printf("%-14s %-12.1f %-12.1f %-14.1f %-9.2f %-8.1f%%\n",
                "Q8/MS", r8_thr.mbps, q8_quant_mbps, q8_eff,
                q8_eff / r32_thr.mbps, 100.0 * (r8_thr.mbps - q8_eff) / r8_thr.mbps);
    std::printf("\n(quant_Mbps = info-bit-rate-equivalent of the float->int LLR quantization alone;\n"
                " dec+quant is a serial estimate: 1/(1/dec + 1/quant). Codeword-bit rates\n"
                " are twice these rates. This does not measure SDR/OFDM or UDP goodput.)\n");
    auto report = [&](const char* precision, const auto& result, double quant, double serial) {
        std::printf("RESULT precision=%s iterations=%d nms=%.3f esn0_db=%.2f "
                    "frames=%d reps=%d info_mbps_median=%.6f info_mbps_best=%.6f "
                    "quant_info_mbps_median=%.6f serial_info_mbps_estimate=%.6f "
                    "ber=%.9g fer=%.9g frame_errors=%ld\n",
                    precision, iterations, nms, thr_snr, TOTAL, reps, result.mbps,
                    result.best_mbps, quant, serial, result.ber, result.fer, result.frame_errors);
    };
    report("float32/NMS", r32_thr, -1.0, r32_thr.mbps);
    report("Q16/NMS", r16_thr, q16_quant_mbps, q16_eff);
    // Q8/MS has no NMS normalization factor; keep it in the human-readable table.
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "LDPCFixedBench: %s\n", error.what());
    print_usage(argv[0]);
    return 1;
}
