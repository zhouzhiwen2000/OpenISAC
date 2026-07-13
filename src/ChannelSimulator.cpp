// ChannelSimulator — the simulated "air" for running UHD_OFDM without USRP.
//
// Reads the same YAML config as BS. Creates the shared-memory rings
// and control block for the session, then continuously:
//   1. drains transmit samples from the BS TX ring,
//   2. applies the channel model for each receive antenna:
//        - sensing RX channels k: superposition of point targets with delay
//          (range), Doppler (velocity), complex gain, and steering vector (angle),
//        - communication RX: a tapped-delay-line multipath channel + CFO,
//      plus a constant timing offset and per-channel AWGN,
//   3. publishes each antenna's samples to its RX ring and advances the clock.
//
// Steering vectors default to a uniform linear array a_k(theta) = exp(j 2pi d k sin theta)
// but can be overridden by an explicit [num_targets x num_channels] complex<float> file.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "AsyncLogger.hpp"
#include "Common.hpp"
#include "ShmRing.hpp"

using sim_shm::sample_t;
using cf = std::complex<float>;

namespace {

constexpr double kSpeedOfLight = 299792458.0;
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr float kTwoPiF = 6.283185307179586f;

// xoshiro256** — small, fast, high-quality PRNG used as the AWGN uniform source.
struct Xoshiro {
    uint64_t s[4];
    explicit Xoshiro(uint64_t seed) {
        // splitmix64 to spread the seed across the 256-bit state.
        for (int i = 0; i < 4; ++i) {
            seed += 0x9E3779B97F4A7C15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            s[i] = z ^ (z >> 31);
        }
    }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        const uint64_t r = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }
    inline float uniform01() {  // (0,1], 24-bit mantissa, never exactly 0 (safe for log)
        return ((next() >> 40) + 1u) * (1.0f / 16777216.0f);
    }
};

// Batched complex AWGN via Box-Muller. Uniforms are drawn into a scratch buffer
// first so the transcendental transform loop vectorizes (libmvec sqrt/log/sincos),
// which is ~5x faster than std::mt19937 + std::normal_distribution for I/Q noise.
struct NoiseGen {
    static constexpr size_t kBlock = 2048;
    Xoshiro rng;
    std::vector<float> u1, u2, re, im;
    explicit NoiseGen(uint64_t seed)
        : rng(seed), u1(kBlock), u2(kBlock), re(kBlock), im(kBlock) {}

    // dst[0..M) += complex AWGN, each of I/Q ~ N(0, sigma^2).
    void add(std::complex<float>* dst, size_t M, float sigma) {
        for (size_t off = 0; off < M; off += kBlock) {
            const size_t m = std::min(kBlock, M - off);
            for (size_t i = 0; i < m; ++i) { u1[i] = rng.uniform01(); u2[i] = rng.uniform01(); }
            for (size_t i = 0; i < m; ++i) {
                const float r = sigma * std::sqrt(-2.0f * std::log(u1[i]));
                const float ang = kTwoPiF * u2[i];
                re[i] = r * std::cos(ang);
                im[i] = r * std::sin(ang);
            }
            std::complex<float>* d = dst + off;
            for (size_t i = 0; i < m; ++i) d[i] += std::complex<float>(re[i], im[i]);
        }
    }
};

std::atomic<bool> g_stop{false};
// Pointer to the shared "running" flag so the signal handler can break the hub's
// blocking shared-memory ring operations on Ctrl-C.
std::atomic<std::atomic<int>*> g_run_flag{nullptr};
void handle_signal(int) {
    g_stop.store(true);
    std::atomic<int>* rf = g_run_flag.load();
    if (rf) rf->store(0);
}

double db_to_linear_amplitude(double db) { return std::pow(10.0, db / 20.0); }

double sinc(double x) {
    if (std::abs(x) < 1e-12) {
        return 1.0;
    }
    const double pix = (0.5 * kTwoPi) * x;
    return std::sin(pix) / pix;
}

double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    const double y = 0.25 * x * x;
    for (int k = 1; k < 30; ++k) {
        term *= y / static_cast<double>(k * k);
        sum += term;
        if (term < sum * 1e-14) {
            break;
        }
    }
    return sum;
}

class PolyphaseSincResampler {
public:
    explicit PolyphaseSincResampler(double source_step, double sample_rate_ratio)
        : _source_step(source_step)
    {
        _buffer.assign(kCenter, sample_t(0.0f, 0.0f));
        _read_pos = static_cast<double>(kCenter);

        constexpr double beta = 8.0;
        const double i0_beta = bessel_i0(beta);
        const double cutoff = std::min(1.0, std::max(0.0, sample_rate_ratio));
        std::vector<double> tap(kTaps);
        for (size_t phase = 0; phase < kPhases; ++phase) {
            const double frac = static_cast<double>(phase) / static_cast<double>(kPhases);
            double sum = 0.0;
            for (size_t k = 0; k < kTaps; ++k) {
                const double rel = frac + static_cast<double>(kCenter) - static_cast<double>(k);
                const double pos = (2.0 * static_cast<double>(k)) /
                    static_cast<double>(kTaps - 1) - 1.0;
                const double window =
                    bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - pos * pos))) / i0_beta;
                const double coeff = cutoff * sinc(cutoff * rel) * window;
                tap[k] = coeff;
                sum += coeff;
            }
            const double inv_sum = (std::abs(sum) > 1e-30) ? (1.0 / sum) : 1.0;
            // Each tap coefficient is stored twice (real+imag lanes) to match the
            // interleaved [re,im,re,im,...] layout of `samples`, so filter_aos below
            // is a single elementwise multiply-accumulate with no per-tap unpacking.
            for (size_t k = 0; k < kTaps; ++k) {
                const float v = static_cast<float>(tap[k] * inv_sum);
                _coeffs[phase * kTaps * 2 + 2 * k] = v;
                _coeffs[phase * kTaps * 2 + 2 * k + 1] = v;
            }
        }
    }

    void append(const sample_t* data, size_t count) {
        _buffer.insert(_buffer.end(), data, data + count);
    }

    void produce(std::vector<sample_t>& out) {
        out.clear();
        out.reserve(static_cast<size_t>(
            std::max(0.0, static_cast<double>(_buffer.size()) / std::max(_source_step, 1e-12))));
        while (true) {
            auto base = static_cast<size_t>(_read_pos);
            const double frac = _read_pos - static_cast<double>(base);
            size_t phase = static_cast<size_t>(std::floor(frac * static_cast<double>(kPhases) + 0.5));
            if (phase >= kPhases) {
                phase = 0;
                ++base;
            }
            if (base + kRight >= _buffer.size()) {
                break;
            }
            out.push_back(filter_aos(&_buffer[base - kCenter], &_coeffs[phase * kTaps * 2]));
            _read_pos += _source_step;
        }

        const auto base = static_cast<size_t>(_read_pos);
        const size_t drop = base > kCenter ? (base - kCenter) : 0;
        if (drop > 0) {
            _buffer.erase(_buffer.begin(), _buffer.begin() + static_cast<std::ptrdiff_t>(drop));
            _read_pos -= static_cast<double>(drop);
        }
    }

private:
    static constexpr size_t kTaps = 32;
    static constexpr size_t kPhases = 1024;
    static constexpr size_t kCenter = 15;
    static constexpr size_t kRight = kTaps - kCenter - 1;
    // Width of the local accumulator the dot product reduces into. This is a plain
    // array, not an intrinsic register width: #pragma omp simd auto-vectorizes the
    // inner loop to whatever the target ISA offers (AVX2/AVX-512/NEON/...) under
    // -march=native, so this only needs to divide 2*kTaps evenly, not match a
    // specific instruction set.
    static constexpr size_t kSimdWidth = 8;
    static_assert((2 * kTaps) % kSimdWidth == 0, "kSimdWidth must divide 2*kTaps");

    // `coeffs` points at kTaps pairs of [re,im] where re==im==the tap weight
    // (see the constructor), matching the interleaved layout of `samples` so the
    // multiply-accumulate below is a straight elementwise pass with no per-tap
    // scalar coefficient assembly.
    static sample_t filter_aos(const sample_t* samples, const float* coeffs) {
        const float* sample_f = reinterpret_cast<const float*>(samples);
        alignas(64) float acc[kSimdWidth] = {};
        for (size_t k = 0; k < 2 * kTaps; k += kSimdWidth) {
            #pragma omp simd
            for (size_t lane = 0; lane < kSimdWidth; ++lane) {
                acc[lane] += sample_f[k + lane] * coeffs[k + lane];
            }
        }
        float re = 0.0f, im = 0.0f;
        for (size_t lane = 0; lane < kSimdWidth; lane += 2) {
            re += acc[lane];
            im += acc[lane + 1];
        }
        return sample_t(re, im);
    }

    double _source_step = 1.0;
    double _read_pos = 0.0;
    std::vector<sample_t> _buffer;
    std::vector<float> _coeffs = std::vector<float>(kPhases * kTaps * 2, 0.0f);
};

constexpr int32_t kSnrControlDisabled = std::numeric_limits<int32_t>::min();

int32_t snr_db_to_command_value(double snr_db) {
    if (!std::isfinite(snr_db)) {
        return kSnrControlDisabled;
    }
    const double clamped = std::clamp(snr_db, -200.0, 200.0);
    return static_cast<int32_t>(std::llround(clamped * 100.0));
}

double command_value_to_snr_db(int32_t value) {
    return static_cast<double>(value) / 100.0;
}

float scale_clean_signal_to_snr(
    sample_t* data,
    size_t count,
    double target_snr_db,
    double noise_power)
{
    if (data == nullptr || count == 0 || noise_power <= 0.0 ||
        !std::isfinite(target_snr_db)) {
        return 1.0f;
    }

    double signal_power = 0.0;
    for (size_t i = 0; i < count; ++i) {
        signal_power += static_cast<double>(std::norm(data[i]));
    }
    signal_power /= static_cast<double>(count);
    if (signal_power <= 1e-30 || !std::isfinite(signal_power)) {
        return 1.0f;
    }

    const double snr_linear = std::pow(10.0, target_snr_db / 10.0);
    const double target_signal_power = noise_power * snr_linear;
    const float scale = static_cast<float>(std::sqrt(target_signal_power / signal_power));
    if (!std::isfinite(scale)) {
        return 1.0f;
    }

    #pragma omp simd
    for (size_t i = 0; i < count; ++i) {
        data[i] *= scale;
    }
    return scale;
}

// Precomputed per-target parameters.
struct Target {
    cf gain;                       // linear complex amplitude (magnitude from gain_db)
    long delay_samples;            // round-trip delay + timing offset, in samples
    cf doppler_step;               // per-sample phase increment exp(j 2pi fd / fs)
    cf doppler_phasor{1.0f, 0.0f}; // running phasor (continuous over time)
    std::vector<cf> steering;      // per-channel steering weight a_k
};

struct Tap {
    cf gain;
    long delay_samples;            // tap delay + timing offset
};

// Load an optional [num_targets x num_channels] complex<float> steering override.
// Returns true and fills `out` (row-major) when the file is present and well-sized.
bool load_steering_override(const std::string& path, size_t num_targets,
                            size_t num_channels, std::vector<cf>& out) {
    if (path.empty()) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        LOG_G_WARN_M(ChannelSim) << "[ChannelSim] steering_override_file '" << path
                     << "' not found; falling back to ULA model.";
        return false;
    }
    const size_t expected = num_targets * num_channels;
    out.resize(expected);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(expected * sizeof(cf)));
    if (static_cast<size_t>(f.gcount()) != expected * sizeof(cf)) {
        LOG_G_WARN_M(ChannelSim) << "[ChannelSim] steering_override_file '" << path << "' has "
                     << f.gcount() << " bytes, expected " << expected * sizeof(cf)
                     << "; falling back to ULA model.";
        out.clear();
        return false;
    }
    LOG_G_INFO_M(ChannelSim) << "[ChannelSim] Loaded steering override (" << num_targets << " targets x "
                 << num_channels << " channels) from " << path;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    async_logger::AsyncLoggerGuard async_logger_guard;
    std::signal(SIGINT, &handle_signal);
    std::signal(SIGTERM, &handle_signal);

    const std::string config_file = (argc > 1) ? argv[1] : "BS.yaml";
    Config cfg = make_default_bs_config();
    if (!path_exists(config_file)) {
        LOG_G_ERROR_M(ChannelSim) << "[ChannelSim] Config file '" << config_file << "' not found.";
        return 1;
    }
    if (!load_bs_config_from_yaml(cfg, config_file)) {
        LOG_G_ERROR_M(ChannelSim) << "[ChannelSim] Failed to load config from '" << config_file << "'.";
        return 1;
    }
    apply_logging_config(cfg.logging);
    normalize_bs_sensing_channels(cfg);

    if (!radio_is_sim(cfg)) {
        LOG_G_WARN_M(ChannelSim) << "[ChannelSim] radio_backend is not 'sim' in " << config_file
                     << "; running the simulator anyway.";
    }

    const SimConfig& sim = cfg.simulation;
    const double fs = cfg.rf_sampling.sample_rate;
    const double sample_rate_offset_ppm = sim.sample_rate_offset_ppm;
    if (!std::isfinite(sample_rate_offset_ppm)) {
        LOG_G_ERROR_M(ChannelSim) << "[ChannelSim] sample_rate_offset_ppm must be finite.";
        return 1;
    }
    const double ue_to_bs_sample_rate_ratio = 1.0 + sample_rate_offset_ppm * 1e-6;
    if (ue_to_bs_sample_rate_ratio < 0.9 || ue_to_bs_sample_rate_ratio > 1.1) {
        LOG_G_ERROR_M(ChannelSim) << "[ChannelSim] sample_rate_offset_ppm=" << sample_rate_offset_ppm
                      << " is outside the supported +/-100000 ppm range.";
        return 1;
    }
    const bool sro_on = std::abs(sample_rate_offset_ppm) > 1e-12;
    const double lambda = kSpeedOfLight / cfg.downlink.center_freq;
    // Selectively enable receive paths. Disabling a path means the hub neither
    // creates nor produces into its ring, so that path's consumer need not run
    // (and the hub will not block waiting for it).
    const bool enable_comm = sim.enable_comm_rx;
    const bool enable_uplink = sim.enable_uplink;
    const size_t num_channels =
        sim.enable_sensing_rx ? cfg.sensing.rx_channels.size() : 0; // sensing antennas
    // Electrical ULA spacing d/lambda used in the steering phase. Derive it from the
    // PHYSICAL element spacing and the carrier so the simulated angles track
    // center_freq exactly like a real array (the viewers invert phase->angle using the
    // same physical spacing). Fall back to the legacy frequency-independent
    // wavelength spacing only when array_spacing_m is disabled (<= 0).
    const double spacing = (sim.array_spacing_m > 0.0)
        ? sim.array_spacing_m * cfg.downlink.center_freq / kSpeedOfLight
        : sim.array_spacing_lambda;

    LOG_G_INFO_M(ChannelSim) << "[ChannelSim] session=" << sim.session << " fs=" << fs
                 << " Hz, center=" << cfg.downlink.center_freq
                 << " Hz, sample_rate_offset_ppm=" << sample_rate_offset_ppm
                 << " (UE/BS ratio=" << ue_to_bs_sample_rate_ratio << ")"
                 << ", comm_rx=" << (enable_comm ? "on" : "off")
                 << ", sensing_channels=" << num_channels
                 << ", pacing=" << (sim.pacing_enabled ? "on" : "off")
                 << ", ULA spacing=" << spacing << " lambda ("
                 << (sim.array_spacing_m > 0.0
                         ? std::to_string(sim.array_spacing_m * 1e3) + " mm @ center_freq"
                         : std::string("legacy array_spacing_lambda"))
                 << ")"
                 << ", targets=" << sim.targets.size()
                 << ", bistatic_targets="
                 << (sim.bistatic_targets.empty() ? sim.targets.size() : sim.bistatic_targets.size())
                 << (sim.bistatic_targets.empty() ? " (reusing targets)" : " (independent)");

    // --- Build target tables ---
    // The bistatic (comm) channel uses its own target list when provided, otherwise it
    // reuses the monostatic `targets` so a single scene drives both views by default.
    const std::vector<SimTarget>& bistatic_src =
        sim.bistatic_targets.empty() ? sim.targets : sim.bistatic_targets;

    std::vector<cf> steering_override;
    const bool has_override =
        load_steering_override(sim.steering_override_file, sim.targets.size(), num_channels, steering_override);

    long max_delay = 0;
    // Build a target table. `with_steering` adds per-antenna steering vectors (monostatic);
    // the bistatic comm channel is a single antenna and needs none.
    auto build_target_table = [&](const std::vector<SimTarget>& list, bool with_steering) {
        std::vector<Target> out;
        out.reserve(list.size());
        for (size_t t = 0; t < list.size(); ++t) {
            const SimTarget& s = list[t];
            Target tg;
            tg.gain = cf(static_cast<float>(db_to_linear_amplitude(s.gain_db)), 0.0f);
            const double tau = 2.0 * s.range_m / kSpeedOfLight;
            tg.delay_samples = std::lround(tau * fs) + sim.timing_offset_samples;
            if (tg.delay_samples < 0) tg.delay_samples = 0;
            const double fd = 2.0 * s.velocity_mps / lambda;
            const double dphi = kTwoPi * fd / fs;
            tg.doppler_step = cf(static_cast<float>(std::cos(dphi)), static_cast<float>(std::sin(dphi)));
            if (with_steering) {
                tg.steering.resize(num_channels);
                for (size_t k = 0; k < num_channels; ++k) {
                    if (has_override) {
                        tg.steering[k] = steering_override[t * num_channels + k];
                    } else {
                        const double ph = kTwoPi * spacing * static_cast<double>(k) *
                                          std::sin(s.angle_deg * kTwoPi / 360.0);
                        tg.steering[k] = cf(static_cast<float>(std::cos(ph)), static_cast<float>(std::sin(ph)));
                    }
                }
            }
            max_delay = std::max(max_delay, tg.delay_samples);
            out.push_back(std::move(tg));
        }
        return out;
    };

    std::vector<Target> targets = build_target_table(sim.targets, true);            // monostatic
    std::vector<Target> bistatic_targets = build_target_table(bistatic_src, false); // downlink bistatic / comm

    // --- Build comm multipath taps (default: single unit tap) ---
    std::vector<Tap> taps;
    if (sim.comm_multipath_taps.empty()) {
        taps.push_back(Tap{cf(1.0f, 0.0f), static_cast<long>(sim.timing_offset_samples)});
    } else {
        for (const auto& tp : sim.comm_multipath_taps) {
            const double amp = db_to_linear_amplitude(tp.gain_db);
            const double ph = tp.phase_deg * kTwoPi / 360.0;
            Tap tap;
            tap.gain = cf(static_cast<float>(amp * std::cos(ph)), static_cast<float>(amp * std::sin(ph)));
            tap.delay_samples = tp.delay_samples + sim.timing_offset_samples;
            if (tap.delay_samples < 0) tap.delay_samples = 0;
            taps.push_back(tap);
        }
    }
    for (const auto& tap : taps) max_delay = std::max(max_delay, tap.delay_samples);

    const size_t L = static_cast<size_t>(max_delay) + 1; // history length

    // --- Build reciprocal uplink (UE->BS) communication channel ---
    // The uplink uses the same static taps and bistatic scatterer scene as the
    // downlink communication path. Per-link timing (timing advance / DL-UL timing
    // difference) is handled by the engines, not here.
    std::vector<Tap> ul_taps;
    std::vector<Target> uplink_bistatic_targets;
    long ul_max_delay = 0;
    if (enable_uplink) {
        ul_taps = taps;
        uplink_bistatic_targets = build_target_table(bistatic_src, false);
        for (const auto& tap : ul_taps) ul_max_delay = std::max(ul_max_delay, tap.delay_samples);
        for (const auto& tg : uplink_bistatic_targets) ul_max_delay = std::max(ul_max_delay, tg.delay_samples);
    }
    const size_t ul_L = static_cast<size_t>(ul_max_delay) + 1; // uplink history length

    // --- Noise setup ---
    const bool noise_on = sim.noise_power_dbfs > -200.0;
    const double noise_power = noise_on ? std::pow(10.0, sim.noise_power_dbfs / 10.0) : 0.0;
    const double noise_sigma = noise_on ? std::sqrt(noise_power / 2.0) : 0.0;
    // A single shared noise stream feeds every output. Generation is serial: the
    // noise kernel is vectorized and so cheap per chunk that an OpenMP fork per chunk
    // measured as a net loss even at 32 channels, so per-channel streams would buy
    // nothing but complexity here.
    NoiseGen noise_gen(0xC0FFEE);
    std::atomic<int32_t> target_snr_centidb{
        sim.snr_control_enable ? snr_db_to_command_value(sim.target_snr_db) : kSnrControlDisabled};
    if (sim.snr_control_enable && !noise_on) {
        LOG_G_WARN_M(ChannelSim) << "[ChannelSim] target SNR control requested, but noise_power_dbfs <= -200 disables AWGN.";
    }

    // --- Comm CFO ---
    // `simulation.cfo_hz` is the injected transmitter/receiver carrier mismatch.
    // The UE writes its simulated RX frequency correction into the shared
    // control block, so the comm path rotates with the residual CFO after retuning.
    cf downlink_cfo_phasor(1.0f, 0.0f);
    const double comm_rx_sample_rate = fs * ue_to_bs_sample_rate_ratio;
    double last_logged_rx_freq_correction_hz = 0.0;
    bool have_logged_rx_freq_correction = false;

    // The same BS/UE oscillator mismatch appears with the opposite sign on the
    // reciprocal UE->BS link. In FDD, the mismatch in Hz scales with carrier
    // frequency, matching the UE's uplink TX correction mapping.
    const double uplink_center_freq =
        (cfg.uplink.duplex.mode == DuplexMode::FDD &&
         cfg.uplink.duplex.ul_center_freq > 0.0)
            ? cfg.uplink.duplex.ul_center_freq
            : cfg.downlink.center_freq;
    const double uplink_cfo_scale =
        (cfg.downlink.center_freq > 0.0 && uplink_center_freq > 0.0)
            ? uplink_center_freq / cfg.downlink.center_freq
            : 1.0;
    const double initial_uplink_cfo_hz = -sim.cfo_hz * uplink_cfo_scale;
    cf uplink_cfo_phasor(1.0f, 0.0f);
    double last_logged_uplink_tx_correction_hz = 0.0;
    bool have_logged_uplink_tx_correction = false;

    // --- Create shared-memory segments (hub is the creator) ---
    sim_shm::ShmControl ctrl;
    ctrl.create(
        sim_shm::make_shm_name(sim.session, "ctrl"),
        fs,
        static_cast<uint32_t>(num_channels),
        sim.cfo_hz,
        sample_rate_offset_ppm);
    std::atomic<int>* running = &ctrl.block()->running;
    g_run_flag.store(running); // let the signal handler break blocking ring ops

    sim_shm::ShmRing tx_ring;
    tx_ring.create(sim_shm::make_shm_name(sim.session, "tx"), sim.ring_capacity_samples);

    std::vector<std::unique_ptr<sim_shm::ShmRing>> sens_rings;
    for (size_t k = 0; k < num_channels; ++k) {
        auto r = std::make_unique<sim_shm::ShmRing>();
        r->create(sim_shm::make_shm_name(sim.session, "rx.sens" + std::to_string(k)), sim.ring_capacity_samples);
        sens_rings.push_back(std::move(r));
    }
    sim_shm::ShmRing comm_ring;
    if (enable_comm) {
        comm_ring.create(sim_shm::make_shm_name(sim.session, "rx.comm"), sim.ring_capacity_samples);
    }
    // Uplink transport: the UE produces into "ul.tx"; the hub applies the uplink
    // channel and publishes into "rx.ul" for the BS uplink RX to consume.
    sim_shm::ShmRing ul_tx_ring;
    sim_shm::ShmRing ul_rx_ring;
    if (enable_uplink) {
        ul_tx_ring.create(sim_shm::make_shm_name(sim.session, "ul.tx"), sim.ring_capacity_samples);
        ul_rx_ring.create(sim_shm::make_shm_name(sim.session, "rx.ul"), sim.ring_capacity_samples);
        LOG_G_INFO_M(ChannelSim) << "[ChannelSim] uplink enabled (UE ul.tx -> hub -> BS rx.ul), reciprocal comm channel, taps="
                     << ul_taps.size()
                     << ", bistatic_targets=" << uplink_bistatic_targets.size()
                     << ", initial CFO=" << initial_uplink_cfo_hz
                     << " Hz (opposite/scaled from downlink)";
    }

    std::unique_ptr<ControlCommandHandler> control_handler;
    if (sim.control_port > 0) {
        control_handler = std::make_unique<ControlCommandHandler>("0.0.0.0", sim.control_port);
        control_handler->register_command("SNR ", [&target_snr_centidb, noise_on](int32_t value) {
            if (value == kSnrControlDisabled) {
                target_snr_centidb.store(kSnrControlDisabled, std::memory_order_release);
                LOG_G_INFO_M(ChannelSim) << "[ChannelSim] target SNR scaling disabled";
                return;
            }
            const int32_t clamped = std::clamp<int32_t>(value, -20000, 20000);
            target_snr_centidb.store(clamped, std::memory_order_release);
            LOG_G_INFO_M(ChannelSim) << "[ChannelSim] target SNR set to "
                         << command_value_to_snr_db(clamped) << " dB"
                         << (noise_on ? "" : " (AWGN is disabled)");
        });
        control_handler->register_request("SNR ", [&target_snr_centidb, &control_handler](
            int32_t,
            const ControlCommandHandler::ControlPeer& peer)
        {
            control_handler->send_control_status(
                peer,
                "SNR ",
                target_snr_centidb.load(std::memory_order_acquire));
        });
        control_handler->start();
        LOG_G_INFO_M(ChannelSim) << "[ChannelSim] ZMQ control ready on port " << sim.control_port
                     << " (command SNR = dB*100; value INT32_MIN disables scaling)";
    } else {
        LOG_G_INFO_M(ChannelSim) << "[ChannelSim] ZMQ control disabled (simulation.control_port <= 0)";
    }

    LOG_G_INFO_M(ChannelSim) << "[ChannelSim] Shared memory ready. Waiting for transmit samples... (Ctrl-C to stop)";

    // --- Processing buffers ---
    const size_t max_chunk = std::max<size_t>(cfg.samples_per_frame(), 4096);
    const size_t max_ul_input_chunk = static_cast<size_t>(
        std::ceil(static_cast<double>(max_chunk) * std::max(1.0, ue_to_bs_sample_rate_ratio))) + 64;
    std::vector<sample_t> in_chunk(max_chunk);
    std::vector<sample_t> ext(L + max_chunk);                 // [history | new chunk]
    std::fill(ext.begin(), ext.begin() + L, sample_t(0.0f, 0.0f));
    std::vector<std::vector<sample_t>> out_sens(num_channels, std::vector<sample_t>(max_chunk));
    std::vector<sample_t> out_comm(max_chunk);
    std::vector<sample_t> comm_sro_out;
    PolyphaseSincResampler comm_sro_resampler(
        1.0 / ue_to_bs_sample_rate_ratio,
        ue_to_bs_sample_rate_ratio);
    // Uplink processing buffers: [history | new UE-clock chunk] for the TDL, plus
    // staging/output buffers. A BS-clock chunk of M samples spans about M*ratio
    // UE samples, then the reciprocal resampler publishes BS-clock rx.ul samples.
    std::vector<sample_t> ul_in(max_ul_input_chunk);
    std::vector<sample_t> ul_out(max_ul_input_chunk);
    std::vector<sample_t> ul_sro_out;
    std::vector<sample_t> ul_ext(ul_L + max_ul_input_chunk);
    if (enable_uplink) std::fill(ul_ext.begin(), ul_ext.begin() + ul_L, sample_t(0.0f, 0.0f));
    PolyphaseSincResampler ul_sro_resampler(
        ue_to_bs_sample_rate_ratio,
        1.0 / ue_to_bs_sample_rate_ratio);
    // Per-target precomputed base signal (gain * delayed x * Doppler) for the
    // monostatic path. Computing this once per target isolates the serial Doppler
    // recurrence so the per-channel accumulation below stays contiguous,
    // vectorizable, and parallelizable. Laid out flat as [target][sample].
    std::vector<cf> target_base(targets.size() * max_chunk);

    uint64_t total_produced = 0;
    auto last_log = std::chrono::steady_clock::now();
    auto next_release_time = last_log;
    int32_t last_logged_snr_centidb = std::numeric_limits<int32_t>::max();

    while (!g_stop.load() && running->load(std::memory_order_acquire) != 0) {
        if (!tx_ring.wait_for_timeline_origin(running, -1.0)) break;
        const uint64_t tx_read_before = tx_ring.consumed();
        const size_t M = tx_ring.pop_upto(in_chunk.data(), max_chunk, running);
        if (M == 0) break; // running cleared
        const int64_t tx_origin = tx_ring.timeline_origin();
        if (tx_origin == sim_shm::kShmTimelineUnset || tx_origin < 0 ||
            tx_read_before > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            tx_origin > std::numeric_limits<int64_t>::max() - static_cast<int64_t>(tx_read_before)) {
            LOG_G_ERROR_M(ChannelSim) << "[ChannelSim] invalid downlink TX timeline origin/read position";
            break;
        }
        const int64_t chunk_start_sample =
            tx_origin + static_cast<int64_t>(tx_read_before);
        ctrl.set_sample_index(static_cast<uint64_t>(chunk_start_sample));

        // Every simulated RX stream uses the same absolute origin as the first
        // downlink air sample. Consumers may seek forward from this point using
        // timed stream commands, exactly like device-clock anchored UHD RX.
        for (auto& r : sens_rings) {
            if (!r->timeline_origin_is_set() &&
                !r->set_timeline_origin(chunk_start_sample)) {
                LOG_G_ERROR_M(ChannelSim) << "[ChannelSim] failed to establish sensing RX timeline";
                running->store(0, std::memory_order_release);
                break;
            }
        }
        if (enable_comm && !comm_ring.timeline_origin_is_set() &&
            !comm_ring.set_timeline_origin(chunk_start_sample)) {
            LOG_G_ERROR_M(ChannelSim) << "[ChannelSim] failed to establish communication RX timeline";
            break;
        }
        if (enable_uplink && !ul_rx_ring.timeline_origin_is_set() &&
            !ul_rx_ring.set_timeline_origin(chunk_start_sample)) {
            LOG_G_ERROR_M(ChannelSim) << "[ChannelSim] failed to establish uplink RX timeline";
            break;
        }

        // Build extended buffer: history (L) followed by the new chunk (M).
        std::copy(in_chunk.begin(), in_chunk.begin() + M, ext.begin() + L);

        // Parallelize the monostatic work across channels only when the per-chunk
        // workload is large enough to amortize the OpenMP fork overhead. Below this
        // threshold the serial path is faster (the precompute+reorder already gives
        // a ~2x SIMD win on its own); forking threads per chunk would only slow it.
        const bool par_channels =
            num_channels * targets.size() * M >= (1u << 18); // ~262k complex MACs

        // --- Zero the monostatic sensing outputs ---
        #pragma omp parallel for schedule(static) if(par_channels)
        for (size_t k = 0; k < num_channels; ++k) {
            std::fill_n(out_sens[k].data(), M, sample_t(0.0f, 0.0f));
        }

        // --- Communication RX direct path: tapped-delay-line (LoS + static multipath) ---
        // This is the strong, decodable component the comm receiver synchronizes to.
        if (enable_comm) {
            for (size_t n = 0; n < M; ++n) {
                cf acc(0.0f, 0.0f);
                for (const auto& tap : taps) {
                    const size_t d = static_cast<size_t>(tap.delay_samples);
                    acc += tap.gain * ext[L + n - d];
                }
                out_comm[n] = acc;
            }
        }

        // --- Monostatic target reflections: moving scatterers with delay + Doppler + steering ---
        if (num_channels > 0 && !targets.empty()) {
            // Phase 1 (serial): precompute each target's delayed, Doppler-shifted
            // base signal once. The Doppler phasor recurrence is inherently serial;
            // pulling it out here keeps the accumulation below vectorizable.
            for (size_t t = 0; t < targets.size(); ++t) {
                Target& tg = targets[t];
                cf phasor = tg.doppler_phasor;
                const size_t d = static_cast<size_t>(tg.delay_samples);
                cf* b = target_base.data() + t * max_chunk;
                for (size_t n = 0; n < M; ++n) {
                    b[n] = tg.gain * ext[L + n - d] * phasor; // gain * x[n-d] * doppler
                    phasor *= tg.doppler_step;
                }
                // Keep the phasor continuous across chunks; renormalize to avoid drift.
                const float mag = std::abs(phasor);
                if (mag > 1e-6f) phasor /= mag;
                tg.doppler_phasor = phasor;
            }
            // Phase 2 (parallel over channels): accumulate each target's steered
            // contribution with a contiguous, vectorizable inner loop. Channels are
            // independent, so there is no cross-thread aliasing on out_sens.
            #pragma omp parallel for schedule(static) if(par_channels)
            for (size_t k = 0; k < num_channels; ++k) {
                sample_t* dst = out_sens[k].data();
                for (size_t t = 0; t < targets.size(); ++t) {
                    const cf s = targets[t].steering[k];
                    const cf* b = target_base.data() + t * max_chunk;
                    for (size_t n = 0; n < M; ++n) dst[n] += s * b[n];
                }
            }
        }

        // --- Bistatic target reflections onto the comm channel (single antenna, no steering) ---
        if (enable_comm) {
            for (auto& tg : bistatic_targets) {
                cf phasor = tg.doppler_phasor;
                const size_t d = static_cast<size_t>(tg.delay_samples);
                for (size_t n = 0; n < M; ++n) {
                    out_comm[n] += tg.gain * ext[L + n - d] * phasor; // gain * x[n-d] * doppler
                    phasor *= tg.doppler_step;
                }
                const float mag = std::abs(phasor);
                if (mag > 1e-6f) phasor /= mag;
                tg.doppler_phasor = phasor;
            }
        }

        size_t comm_count = M;
        if (enable_comm && sro_on) {
            comm_sro_resampler.append(out_comm.data(), M);
            comm_sro_resampler.produce(comm_sro_out);
            comm_count = comm_sro_out.size();
            if (out_comm.size() < comm_count) {
                out_comm.resize(comm_count);
            }
            std::copy(comm_sro_out.begin(), comm_sro_out.end(), out_comm.begin());
        }

        // --- Apply RX carrier frequency offset to the whole comm signal ---
        if (enable_comm) {
            const double rx_freq_correction_hz = ctrl.comm_rx_freq_correction_hz();
            const double residual_cfo_hz = sim.cfo_hz + rx_freq_correction_hz;
            if (!have_logged_rx_freq_correction ||
                std::abs(rx_freq_correction_hz - last_logged_rx_freq_correction_hz) > 1e-3) {
                LOG_G_INFO_M(ChannelSim) << "[ChannelSim] comm RX frequency correction="
                             << rx_freq_correction_hz << " Hz, residual CFO="
                             << residual_cfo_hz << " Hz";
                last_logged_rx_freq_correction_hz = rx_freq_correction_hz;
                have_logged_rx_freq_correction = true;
            }
            const double cfo_dphi =
                (comm_rx_sample_rate > 0.0) ? (kTwoPi * residual_cfo_hz / comm_rx_sample_rate) : 0.0;
            const cf cfo_step(
                static_cast<float>(std::cos(cfo_dphi)),
                static_cast<float>(std::sin(cfo_dphi)));
            for (size_t n = 0; n < comm_count; ++n) {
                out_comm[n] *= downlink_cfo_phasor;
                downlink_cfo_phasor *= cfo_step;
            }
            const float mag = std::abs(downlink_cfo_phasor);
            if (mag > 1e-6f) downlink_cfo_phasor /= mag;
        }

        // --- Target SNR scaling ---
        // The clean simulated signal is scaled before AWGN, so online SNR changes
        // leave the noise floor fixed and adjust only the effective signal level.
        const int32_t snr_centidb = target_snr_centidb.load(std::memory_order_acquire);
        if (snr_centidb != last_logged_snr_centidb) {
            if (snr_centidb == kSnrControlDisabled) {
                LOG_G_INFO_M(ChannelSim) << "[ChannelSim] target SNR scaling is off";
            } else {
                LOG_G_INFO_M(ChannelSim) << "[ChannelSim] applying target SNR "
                             << command_value_to_snr_db(snr_centidb) << " dB";
            }
            last_logged_snr_centidb = snr_centidb;
        }
        if (noise_on && snr_centidb != kSnrControlDisabled) {
            const double target_snr_db = command_value_to_snr_db(snr_centidb);
            for (size_t k = 0; k < num_channels; ++k) {
                (void)scale_clean_signal_to_snr(out_sens[k].data(), M, target_snr_db, noise_power);
            }
            if (enable_comm) {
                (void)scale_clean_signal_to_snr(out_comm.data(), comm_count, target_snr_db, noise_power);
            }
        }

        // --- AWGN ---
        if (noise_on) {
            const float sigma = static_cast<float>(noise_sigma);
            for (size_t k = 0; k < num_channels; ++k) {
                noise_gen.add(out_sens[k].data(), M, sigma);
            }
            if (enable_comm) {
                noise_gen.add(out_comm.data(), comm_count, sigma);
            }
        }

        if (sim.pacing_enabled && fs > 0.0) {
            const auto now = std::chrono::steady_clock::now();
            if (next_release_time < now) {
                next_release_time = now;
            }
            next_release_time += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(static_cast<double>(M) / fs));
            if (next_release_time > now) {
                std::this_thread::sleep_until(next_release_time);
            }
        }

        // --- Publish ---
        for (size_t k = 0; k < num_channels; ++k) {
            sens_rings[k]->push_block(out_sens[k].data(), M, running);
        }
        if (enable_comm) {
            comm_ring.push_block(out_comm.data(), comm_count, running);
        }

        // --- Uplink (UE->BS): drain UE TX for this BS-clock chunk, apply TDL + AWGN ---
        // When SRO is enabled, the UE sample stream advances at UE/BS ratio relative
        // to the BS clock, then the reciprocal resampler publishes rx.ul on the BS
        // clock. The output timeline never pauses: before/after available UE TX
        // samples the BS receives silence plus configured AWGN, matching a running
        // radio clock instead of giving rx.ul an unrelated ring-local epoch.
        if (enable_uplink) {
            const size_t desired_ul_input = sro_on
                ? std::max<size_t>(
                      1,
                      static_cast<size_t>(std::ceil(static_cast<double>(M) * ue_to_bs_sample_rate_ratio)))
                : M;
            std::fill_n(ul_in.data(), desired_ul_input, sample_t(0.0f, 0.0f));

            // Map this BS-clock interval to the corresponding UE-stream sample
            // offset. A future UE timed start leaves a zero prefix; a late reader
            // discards stale samples so the two streams cannot silently acquire
            // independent time origins.
            size_t ul_insert_offset = 0;
            bool ul_aligned_for_read = false;
            if (ul_tx_ring.timeline_origin_is_set()) {
                const int64_t ul_origin = ul_tx_ring.timeline_origin();
                const long double relative_ue_samples =
                    (static_cast<long double>(chunk_start_sample) -
                     static_cast<long double>(ul_origin)) *
                    static_cast<long double>(ue_to_bs_sample_rate_ratio);
                if (relative_ue_samples < 0.0L) {
                    const long double prefix = std::clamp<long double>(
                        -relative_ue_samples,
                        0.0L,
                        static_cast<long double>(desired_ul_input));
                    ul_insert_offset = static_cast<size_t>(std::llround(prefix));
                    ul_aligned_for_read = true;
                } else {
                    const uint64_t target = static_cast<uint64_t>(std::llround(
                        std::min<long double>(
                            relative_ue_samples,
                            static_cast<long double>(std::numeric_limits<int64_t>::max()))));
                    const uint64_t consumed = ul_tx_ring.consumed();
                    if (consumed <= target) {
                        const uint64_t stale = target - consumed;
                        const size_t stale_bounded = static_cast<size_t>(std::min<uint64_t>(
                            stale, static_cast<uint64_t>(std::numeric_limits<size_t>::max())));
                        const size_t skipped = ul_tx_ring.skip_block(
                            stale_bounded, running, 0.0);
                        ul_aligned_for_read = skipped == stale;
                    } else {
                        // Rounding at a non-zero SRO can put the consumer one
                        // sample ahead of the ideal mapping. Keep the stream
                        // continuous rather than attempting to un-read.
                        ul_aligned_for_read = true;
                    }
                }
            }

            if (ul_aligned_for_read && ul_insert_offset < desired_ul_input) {
                (void)ul_tx_ring.pop_block(
                    ul_in.data() + ul_insert_offset,
                    desired_ul_input - ul_insert_offset,
                    running,
                    0.0);
            }

            std::copy(
                ul_in.begin(), ul_in.begin() + desired_ul_input, ul_ext.begin() + ul_L);
            for (size_t n = 0; n < desired_ul_input; ++n) {
                cf acc(0.0f, 0.0f);
                for (const auto& tap : ul_taps) {
                    const size_t d = static_cast<size_t>(tap.delay_samples);
                    acc += tap.gain * ul_ext[ul_L + n - d];
                }
                ul_out[n] = acc;
            }
            for (auto& tg : uplink_bistatic_targets) {
                cf phasor = tg.doppler_phasor;
                const size_t d = static_cast<size_t>(tg.delay_samples);
                for (size_t n = 0; n < desired_ul_input; ++n) {
                    ul_out[n] += tg.gain * ul_ext[ul_L + n - d] * phasor;
                    phasor *= tg.doppler_step;
                }
                const float mag = std::abs(phasor);
                if (mag > 1e-6f) phasor /= mag;
                tg.doppler_phasor = phasor;
            }
            size_t ul_count = desired_ul_input;
            if (sro_on) {
                ul_sro_resampler.append(ul_out.data(), desired_ul_input);
                ul_sro_resampler.produce(ul_sro_out);
                ul_count = ul_sro_out.size();
                if (ul_out.size() < ul_count) {
                    ul_out.resize(ul_count);
                }
                std::copy(ul_sro_out.begin(), ul_sro_out.end(), ul_out.begin());
            }
            // Apply reciprocal-link residual CFO on the BS-clock output. The
            // UE publishes its logical TX carrier correction through the sim
            // control block whenever downlink tracking retunes the uplink TX.
            const double uplink_tx_correction_hz = ctrl.uplink_tx_freq_correction_hz();
            const double residual_uplink_cfo_hz =
                initial_uplink_cfo_hz - uplink_tx_correction_hz;
            if (!have_logged_uplink_tx_correction ||
                std::abs(uplink_tx_correction_hz - last_logged_uplink_tx_correction_hz) > 1e-3) {
                LOG_G_INFO_M(ChannelSim) << "[ChannelSim] uplink TX frequency correction="
                             << uplink_tx_correction_hz << " Hz, residual CFO="
                             << residual_uplink_cfo_hz << " Hz";
                last_logged_uplink_tx_correction_hz = uplink_tx_correction_hz;
                have_logged_uplink_tx_correction = true;
            }
            const double uplink_cfo_dphi =
                (fs > 0.0) ? (kTwoPi * residual_uplink_cfo_hz / fs) : 0.0;
            const cf uplink_cfo_step(
                static_cast<float>(std::cos(uplink_cfo_dphi)),
                static_cast<float>(std::sin(uplink_cfo_dphi)));
            for (size_t n = 0; n < ul_count; ++n) {
                ul_out[n] *= uplink_cfo_phasor;
                uplink_cfo_phasor *= uplink_cfo_step;
            }
            const float uplink_cfo_mag = std::abs(uplink_cfo_phasor);
            if (uplink_cfo_mag > 1e-6f) uplink_cfo_phasor /= uplink_cfo_mag;
            // NOTE: no per-chunk SNR-control scaling on the uplink — the uplink
            // frame is zero-padded within the DL period, so chunk-wise power
            // normalization would vary the signal level across the frame and
            // break the BS equalization. Uplink SNR = tap gain vs AWGN.
            if (noise_on && ul_count > 0) {
                noise_gen.add(ul_out.data(), ul_count, static_cast<float>(noise_sigma));
            }
            if (ul_count > 0) {
                ul_rx_ring.push_block(ul_out.data(), ul_count, running);
            }
            // Slide uplink history (last ul_L samples of the extended buffer).
            std::copy(
                ul_ext.begin() + desired_ul_input,
                ul_ext.begin() + desired_ul_input + ul_L,
                ul_ext.begin());
        }

        ctrl.set_sample_index(static_cast<uint64_t>(chunk_start_sample) + M);
        total_produced += M;

        // Slide history: last L samples of ext become the new history.
        std::copy(ext.begin() + M, ext.begin() + M + L, ext.begin());

        const auto now = std::chrono::steady_clock::now();
        if (now - last_log > std::chrono::seconds(2)) {
            LOG_G_INFO_M(ChannelSim) << "[ChannelSim] produced " << total_produced << " samples ("
                         << static_cast<double>(total_produced) / fs << " s of air time)";
            last_log = now;
        }
    }

    LOG_G_INFO_M(ChannelSim) << "[ChannelSim] Stopping. Cleaning up shared memory.";
    if (control_handler) {
        control_handler->stop();
    }
    ctrl.stop();
    // Give clients a moment to observe the stop flag before unlinking.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    tx_ring.unlink();
    for (auto& r : sens_rings) r->unlink();
    comm_ring.unlink();
    if (enable_uplink) {
        ul_tx_ring.unlink();
        ul_rx_ring.unlink();
    }
    ctrl.unlink();
    return 0;
}
