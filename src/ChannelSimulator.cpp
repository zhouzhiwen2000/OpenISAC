// ChannelSimulator — the simulated "air" for running UHD_OFDM without USRP.
//
// Reads the same YAML config as the Modulator. Creates the shared-memory rings
// and control block for the session, then continuously:
//   1. drains transmit samples from the modulator's TX ring,
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
        LOG_G_WARN() << "[ChannelSim] steering_override_file '" << path
                     << "' not found; falling back to ULA model.";
        return false;
    }
    const size_t expected = num_targets * num_channels;
    out.resize(expected);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(expected * sizeof(cf)));
    if (static_cast<size_t>(f.gcount()) != expected * sizeof(cf)) {
        LOG_G_WARN() << "[ChannelSim] steering_override_file '" << path << "' has "
                     << f.gcount() << " bytes, expected " << expected * sizeof(cf)
                     << "; falling back to ULA model.";
        out.clear();
        return false;
    }
    LOG_G_INFO() << "[ChannelSim] Loaded steering override (" << num_targets << " targets x "
                 << num_channels << " channels) from " << path;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    async_logger::AsyncLoggerGuard async_logger_guard;
    std::signal(SIGINT, &handle_signal);
    std::signal(SIGTERM, &handle_signal);

    const std::string config_file = (argc > 1) ? argv[1] : "Modulator.yaml";
    Config cfg = make_default_modulator_config();
    if (!path_exists(config_file)) {
        LOG_G_ERROR() << "[ChannelSim] Config file '" << config_file << "' not found.";
        return 1;
    }
    if (!load_modulator_config_from_yaml(cfg, config_file)) {
        LOG_G_ERROR() << "[ChannelSim] Failed to load config from '" << config_file << "'.";
        return 1;
    }
    normalize_modulator_sensing_channels(cfg);

    if (!radio_is_sim(cfg)) {
        LOG_G_WARN() << "[ChannelSim] radio_backend is not 'sim' in " << config_file
                     << "; running the simulator anyway.";
    }

    const SimConfig& sim = cfg.simulation;
    const double fs = cfg.sample_rate;
    const double lambda = kSpeedOfLight / cfg.center_freq;
    // Selectively enable receive paths. Disabling a path means the hub neither
    // creates nor produces into its ring, so that path's consumer need not run
    // (and the hub will not block waiting for it).
    const bool enable_comm = sim.enable_comm_rx;
    const size_t num_channels =
        sim.enable_sensing_rx ? cfg.sensing_rx_channels.size() : 0; // sensing antennas
    // Electrical ULA spacing d/lambda used in the steering phase. Derive it from the
    // PHYSICAL element spacing and the carrier so the simulated angles track
    // center_freq exactly like a real array (the viewers invert phase->angle using the
    // same physical spacing). Fall back to the legacy frequency-independent
    // wavelength spacing only when array_spacing_m is disabled (<= 0).
    const double spacing = (sim.array_spacing_m > 0.0)
        ? sim.array_spacing_m * cfg.center_freq / kSpeedOfLight
        : sim.array_spacing_lambda;

    LOG_G_INFO() << "[ChannelSim] session=" << sim.session << " fs=" << fs
                 << " Hz, center=" << cfg.center_freq
                 << " Hz, comm_rx=" << (enable_comm ? "on" : "off")
                 << ", sensing_channels=" << num_channels
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
    std::vector<Target> bistatic_targets = build_target_table(bistatic_src, false); // bistatic / comm

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

    // --- Noise setup ---
    const bool noise_on = sim.noise_power_dbfs > -200.0;
    const double noise_sigma = noise_on ? std::sqrt(std::pow(10.0, sim.noise_power_dbfs / 10.0) / 2.0) : 0.0;
    // A single shared noise stream feeds every output. Generation is serial: the
    // noise kernel is vectorized and so cheap per chunk that an OpenMP fork per chunk
    // measured as a net loss even at 32 channels, so per-channel streams would buy
    // nothing but complexity here.
    NoiseGen noise_gen(0xC0FFEE);

    // --- Comm CFO ---
    const double cfo_dphi = kTwoPi * sim.cfo_hz / fs;
    cf cfo_step(static_cast<float>(std::cos(cfo_dphi)), static_cast<float>(std::sin(cfo_dphi)));
    cf cfo_phasor(1.0f, 0.0f);

    // --- Create shared-memory segments (hub is the creator) ---
    sim_shm::ShmControl ctrl;
    ctrl.create(sim_shm::make_shm_name(sim.session, "ctrl"), fs, static_cast<uint32_t>(num_channels));
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

    LOG_G_INFO() << "[ChannelSim] Shared memory ready. Waiting for transmit samples... (Ctrl-C to stop)";

    // --- Processing buffers ---
    const size_t max_chunk = std::max<size_t>(cfg.samples_per_frame(), 4096);
    std::vector<sample_t> in_chunk(max_chunk);
    std::vector<sample_t> ext(L + max_chunk);                 // [history | new chunk]
    std::fill(ext.begin(), ext.begin() + L, sample_t(0.0f, 0.0f));
    std::vector<std::vector<sample_t>> out_sens(num_channels, std::vector<sample_t>(max_chunk));
    std::vector<sample_t> out_comm(max_chunk);
    // Per-target precomputed base signal (gain * delayed x * Doppler) for the
    // monostatic path. Computing this once per target isolates the serial Doppler
    // recurrence so the per-channel accumulation below stays contiguous,
    // vectorizable, and parallelizable. Laid out flat as [target][sample].
    std::vector<cf> target_base(targets.size() * max_chunk);

    uint64_t total_produced = 0;
    auto last_log = std::chrono::steady_clock::now();

    while (!g_stop.load() && running->load(std::memory_order_acquire) != 0) {
        const size_t M = tx_ring.pop_upto(in_chunk.data(), max_chunk, running);
        if (M == 0) break; // running cleared

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

        // --- Apply RX carrier frequency offset to the whole comm signal ---
        if (enable_comm) {
            for (size_t n = 0; n < M; ++n) {
                out_comm[n] *= cfo_phasor;
                cfo_phasor *= cfo_step;
            }
            const float mag = std::abs(cfo_phasor);
            if (mag > 1e-6f) cfo_phasor /= mag;
        }

        // --- AWGN ---
        if (noise_on) {
            const float sigma = static_cast<float>(noise_sigma);
            for (size_t k = 0; k < num_channels; ++k) {
                noise_gen.add(out_sens[k].data(), M, sigma);
            }
            if (enable_comm) {
                noise_gen.add(out_comm.data(), M, sigma);
            }
        }

        // --- Publish ---
        for (size_t k = 0; k < num_channels; ++k) {
            sens_rings[k]->push_block(out_sens[k].data(), M, running);
        }
        if (enable_comm) {
            comm_ring.push_block(out_comm.data(), M, running);
        }
        ctrl.advance(M);
        total_produced += M;

        // Slide history: last L samples of ext become the new history.
        std::copy(ext.begin() + M, ext.begin() + M + L, ext.begin());

        const auto now = std::chrono::steady_clock::now();
        if (now - last_log > std::chrono::seconds(2)) {
            LOG_G_INFO() << "[ChannelSim] produced " << total_produced << " samples ("
                         << static_cast<double>(total_produced) / fs << " s of air time)";
            last_log = now;
        }
    }

    LOG_G_INFO() << "[ChannelSim] Stopping. Cleaning up shared memory.";
    ctrl.stop();
    // Give clients a moment to observe the stop flag before unlinking.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    tx_ring.unlink();
    for (auto& r : sens_rings) r->unlink();
    comm_ring.unlink();
    ctrl.unlink();
    return 0;
}
