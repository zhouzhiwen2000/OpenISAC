#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdint>
#include <vector>
#include <algorithm>
#include <alloca.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <complex>
#include <functional>
#include <array>
#include <deque>
#include <numeric>
#include <memory>
#include <map>
#include <set>
#include <utility>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iterator>
#include <optional>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <string>
#include <stdexcept>
#include <limits>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fftw3.h>
#include <boost/circular_buffer.hpp>
#include <functional>
#include <unordered_map>
#include "RadioTypes.hpp"
#include <uhd/types/metadata.hpp>
#include <uhd/utils/thread.hpp>
#include <uhd/types/time_spec.hpp>
#include <fcntl.h>
#include <termios.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <yaml-cpp/yaml.h>
#include "AsyncLogger.hpp"
#include "ZmqTransport.hpp"

/**
 * @brief STL-compliant Aligned Memory Allocator.
 * 
 * Ensures that allocated memory is aligned to specific boundaries (default 64 bytes).
 * This is critical for SIMD operations (AVX2/AVX-512) and other low-level optimizations.
 */
template <typename T, size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = size_t;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    bool operator==(const AlignedAllocator&) const { return true; }
    bool operator!=(const AlignedAllocator& other) const { return !(*this == other); }

    pointer allocate(size_type n) {
        if (n > (std::numeric_limits<size_type>::max() / sizeof(value_type))) {
            throw std::bad_alloc();
        }
        size_type bytes = n * sizeof(value_type);
        size_type aligned_bytes = (bytes + Alignment - 1) & ~(Alignment - 1);
        void* ptr = std::aligned_alloc(Alignment, aligned_bytes);
        if (!ptr) throw std::bad_alloc();
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) { std::free(p); }
};

// Type Definitions (64-byte alignment for AVX-512 optimization)
using AlignedVector = std::vector<std::complex<float>, AlignedAllocator<std::complex<float>, 64>>;
using AlignedFloatVector = std::vector<float, AlignedAllocator<float, 64>>;
using AlignedIntVector = std::vector<int, AlignedAllocator<int, 64>>;
using SymbolVector = std::vector<AlignedVector>;

inline constexpr const char* kEqualizerModeZf = "zf";
inline constexpr const char* kEqualizerModeMmse = "mmse";
inline constexpr const char* kChannelTrackingModeOff = "disabled";
inline constexpr const char* kChannelTrackingModePilotPhase = "pilot_phase";
inline constexpr const char* kSensingDelayCorrectionModeLosTracking = "los_tracking";
inline constexpr const char* kSensingDelayCorrectionModeErtmAbsolute = "ertm_absolute";
inline constexpr const char* kErtmTimingMetricDelayMagnitude = "delay_magnitude";
inline constexpr const char* kErtmTimingMetricMaximumLikelihood = "maximum_likelihood";

/**
 * @brief Zero-copy view over frequency-domain TX symbols for sensing.
 *
 * Supports two storage backends:
 *   1. Flat contiguous buffer (CUDA pinned memory path):
 *      Symbols are stored as [num_symbols × fft_size] complex<float> in a
 *      single contiguous allocation.  operator[] extracts a copy of the
 *      requested symbol on demand — only the ~5 symbols per frame actually
 *      sampled by the sensing channel are ever copied.
 *   2. SymbolVector (CPU path):
 *      Wraps an existing shared_ptr<const SymbolVector> with zero overhead.
 *
 * The class is cheap to copy (shared_ptr semantics) and keeps the
 * underlying memory alive until the last reference is released.
 */
class TxSymbolView {
public:
    TxSymbolView() = default;

    /** Construct from flat contiguous buffer (CUDA pinned memory path). */
    TxSymbolView(std::shared_ptr<const void> owner,
                 const std::complex<float>* flat_data,
                 size_t num_symbols, size_t fft_size)
        : _owner(std::move(owner))
        , _flat_data(flat_data)
        , _num_symbols(num_symbols)
        , _fft_size(fft_size)
    {}

    /** Construct from existing SymbolVector (CPU path — zero overhead). */
    explicit TxSymbolView(std::shared_ptr<const SymbolVector> sv)
        : _sv(std::move(sv))
        , _num_symbols(_sv ? _sv->size() : 0)
        , _fft_size((_sv && !_sv->empty()) ? (*_sv)[0].size() : 0)
    {}

    /** Number of OFDM symbols in this frame. */
    size_t size() const { return _num_symbols; }

    /** Extract symbol @p i as AlignedVector (copies fft_size complex<float>). */
    AlignedVector operator[](size_t i) const {
        if (_sv) {
            // CPU path — direct reference, copied by caller via push_back
            return (*_sv)[i];
        }
        // Flat buffer path — construct AlignedVector from contiguous slice
        const auto* src = _flat_data + i * _fft_size;
        return AlignedVector(src, src + _fft_size);
    }

private:
    std::shared_ptr<const void> _owner;            // Prevents deallocation (pinned mem / pool)
    const std::complex<float>* _flat_data = nullptr;
    std::shared_ptr<const SymbolVector> _sv;       // CPU path (nullptr for flat buffer path)
    size_t _num_symbols = 0;
    size_t _fft_size = 0;
};

/**
 * @brief Per sensing RX channel configuration.
 *
 * Used by BS for multi-channel monostatic sensing reception.
 */
struct SensingRxChannelConfig {
    uint32_t usrp_channel = 0;          // USRP RX channel index
    std::string device_args = "";       // Optional per-channel USRP args
    std::string clock_source = "";      // Optional per-channel USRP clock/REF source override
    std::string time_source = "";       // Optional per-channel USRP time/PPS source override
    std::string wire_format = "";       // Optional per-channel RX wire format override
    double rx_gain = 0.0;               // RX gain for this channel
    int32_t alignment = 63;             // Per-channel alignment offset (samples)
    std::string rx_antenna = "";        // RX antenna for this channel (e.g., TX/RX, RX1)
    bool enable_system_delay_estimation = false; // Enable per-channel system delay estimation mode
    bool enable_sensing_output = true;  // Enable ZMQ output for this sensing channel
    int rx_cpu_core = -1;               // Dedicated SensingChannel::_rx_loop core; -1 = no explicit binding
    int processing_cpu_core = -1;       // Dedicated SensingChannel::_sensing_loop core; -1 = no explicit binding
};

/**
 * @brief A single point scatterer / target used by the channel simulator.
 *
 * The simulated RX signal for each antenna is the superposition over targets of
 * the transmit waveform delayed by tau = 2*range_m/c, frequency-shifted by the
 * Doppler fd = 2*velocity_mps/lambda, scaled by the linear gain, and weighted by
 * the array steering vector for angle_deg.
 */
struct SimTarget {
    double range_m = 0.0;       // One-way range in meters (round-trip delay = 2*range/c)
    double velocity_mps = 0.0;  // Radial velocity in m/s (positive = approaching)
    double gain_db = 0.0;       // Complex amplitude magnitude in dB
    double angle_deg = 0.0;     // Angle of arrival in degrees (used for steering vector)
};

/**
 * @brief One tap of the simulated communication-link tapped-delay-line channel.
 */
struct SimMultipathTap {
    int delay_samples = 0;      // Integer sample delay
    double gain_db = 0.0;       // Tap magnitude in dB
    double phase_deg = 0.0;     // Tap phase in degrees
};

/**
 * @brief Channel simulator parameters (used when radio_backend == "sim").
 *
 * Shared by the BS, UE, and the standalone ChannelSimulator hub.
 * The `session` string namespaces the POSIX shared-memory segments so all three
 * processes attach to the same simulated "air".
 */
struct SimConfig {
    std::string session = "oisac_sim";        // shm namespace shared by all processes
    bool enable_comm_rx = true;                // produce the communication RX channel (run with UE)
    bool enable_sensing_rx = true;             // produce the monostatic sensing RX channels (one per antenna)
    bool enable_uplink = false;                // route the UE->BS uplink stream (UE TX -> BS uplink RX)
    bool pacing_enabled = true;                // pace the simulator to wall-clock sample time
    double noise_power_dbfs = -100.0;          // AWGN power per RX channel (dBFS); very low = effectively off
    bool snr_control_enable = false;           // Enable target-SNR scaling of clean signal before AWGN
    double target_snr_db = 40.0;               // Target SNR when snr_control_enable is true
    int control_port = 10002;                  // ZMQ ROUTER port for ChannelSimulator runtime controls
    double cfo_hz = 0.0;                        // Initial BS->UE CFO; reciprocal UL is opposite/scaled and TX-retuned (Hz)
    double sample_rate_offset_ppm = 0.0;        // UE sample clock offset relative to the BS clock (ppm)
    int timing_offset_samples = 0;             // Constant integer sample delay injected on RX
    // Physical ULA element spacing in meters. The steering vector's electrical spacing
    // (d/lambda) is derived from this and `center_freq`, so the simulated angles track
    // the carrier exactly like a real array would (the sensing viewers invert the
    // measured phase slope back to an angle using this same PHYSICAL spacing). The
    // default 42.83 mm equals lambda/2 at 3.5 GHz and matches ANTENNA_SPACING_M in the
    // viewers; with it the recovered angle is correct at any center_freq.
    double array_spacing_m = 0.04283;
    // Legacy: ULA spacing directly in wavelengths (frequency-independent). Only used
    // when `array_spacing_m` is set to <= 0; prefer `array_spacing_m` for correct
    // angle recovery across center_freq.
    double array_spacing_lambda = 0.5;         // ULA element spacing in wavelengths
    std::vector<SimMultipathTap> comm_multipath_taps; // Comm-link TDL taps (empty => single unit tap)
    std::vector<SimTarget> targets;            // Point scatterers for monostatic sensing
    std::vector<SimTarget> bistatic_targets;   // Scatterers for the bistatic (comm) channel; empty => reuse `targets`
    std::string steering_override_file = "";   // Optional [num_targets x num_rx_channels] complex<float> matrix
    size_t ring_capacity_samples = 1u << 22;   // Per-stream shm ring capacity (complex<float> samples)
};

/**
 * @brief Duplexing scheme between the downlink (BS->UE) and uplink (UE->BS).
 *
 * TDD: uplink shares the downlink carrier and is time-multiplexed into a
 *      contiguous range of OFDM symbols within each frame, separated from the
 *      downlink by an optional guard interval (both measured in OFDM symbols).
 * FDD: uplink occupies a separate carrier (`ul_center_freq`) and transmits a
 *      continuous full-frame uplink. TDD ignores `ul_center_freq` and uses the
 *      downlink carrier.
 */
enum class DuplexMode : uint8_t {
    TDD = 0,
    FDD = 1,
};

inline constexpr const char* kDuplexModeTdd = "tdd";
inline constexpr const char* kDuplexModeFdd = "fdd";
inline constexpr const char* kUplinkIdleWaveformZero = "zero";
inline constexpr const char* kUplinkIdleWaveformRandomQpsk = "random_qpsk";

inline const char* duplex_mode_to_string(DuplexMode mode) {
    switch (mode) {
    case DuplexMode::FDD:
        return kDuplexModeFdd;
    case DuplexMode::TDD:
    default:
        return kDuplexModeTdd;
    }
}

inline bool parse_duplex_mode_string(const std::string& raw_mode, DuplexMode& out_mode) {
    std::string mode;
    mode.reserve(raw_mode.size());
    for (char ch : raw_mode) {
        mode.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (mode == kDuplexModeTdd) {
        out_mode = DuplexMode::TDD;
        return true;
    }
    if (mode == kDuplexModeFdd) {
        out_mode = DuplexMode::FDD;
        return true;
    }
    return false;
}

inline std::string normalize_uplink_idle_waveform_string(std::string waveform) {
    std::transform(waveform.begin(), waveform.end(), waveform.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (waveform == kUplinkIdleWaveformZero ||
        waveform == kUplinkIdleWaveformRandomQpsk) {
        return waveform;
    }
    throw std::runtime_error(
        "Invalid uplink.idle_waveform='" + waveform +
        "'. Expected 'zero' or 'random_qpsk'.");
}

inline std::string normalize_channel_tracking_mode_string(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (mode.empty()) {
        return kChannelTrackingModePilotPhase;
    }
    if (mode == "off" || mode == "false" || mode == "0" || mode == "no") {
        return kChannelTrackingModeOff;
    }
    if (mode == kChannelTrackingModeOff || mode == kChannelTrackingModePilotPhase) {
        return mode;
    }
    return kChannelTrackingModePilotPhase;
}

inline std::string normalize_sensing_delay_correction_mode_string(const std::string& mode) {
    if (mode == kSensingDelayCorrectionModeLosTracking ||
        mode == kSensingDelayCorrectionModeErtmAbsolute) {
        return mode;
    }
    throw std::runtime_error(
        "Invalid sensing.sensing_delay_correction_mode='" + mode +
        "'. Supported values: los_tracking, ertm_absolute.");
}

inline std::string normalize_ertm_timing_metric_string(std::string metric) {
    std::transform(metric.begin(), metric.end(), metric.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (metric == kErtmTimingMetricDelayMagnitude ||
        metric == kErtmTimingMetricMaximumLikelihood) {
        return metric;
    }
    throw std::runtime_error(
        "Invalid uplink.ertm_timing_metric='" + metric +
        "'. Supported values: delay_magnitude, maximum_likelihood.");
}

/**
 * @brief Duplexing configuration shared by the BS and UE.
 *
 * The uplink reuses the identical OFDM numerology and frame structure as the
 * downlink (same fft_size/cp_length/sync_pos/pilots), so the uplink RX can reuse
 * the downlink demod pipeline verbatim. In TDD the uplink occupies symbols
 * [ul_symbol_start, ul_symbol_start + ul_symbol_count); the first
 * `ul_guard_symbols` of that range are a blanked DL->UL guard (may be 0).
 */
struct DuplexConfig {
    DuplexMode mode = DuplexMode::TDD;
    size_t ul_symbol_start = 90;    // TDD: first uplink OFDM symbol within a frame
    size_t ul_symbol_count = 10;    // TDD: number of uplink symbols (0 => uplink disabled)
    size_t ul_guard_symbols = 1;    // TDD: guard symbols at the DL->UL boundary (within the UL range)
    double ul_center_freq = 2.5e9;  // FDD: uplink carrier frequency in Hz; TDD ignored
};

/**
 * @brief Rectangular payload resource block in the OFDM time-frequency grid.
 *
 * Coordinates use absolute 0-based frame indices:
 * - symbol_start / symbol_count refer to OFDM symbols within one frame
 * - subcarrier_start / subcarrier_count refer to FFT-bin indices
 */
enum class DataResourceBlockKind : uint8_t {
    Payload = 0,
    SensingPilot = 1,
};

inline constexpr const char* kDataResourceBlockKindPayload = "payload";
inline constexpr const char* kDataResourceBlockKindSensingPilot = "sensing_pilot";

inline const char* data_resource_block_kind_to_string(DataResourceBlockKind kind) {
    switch (kind) {
    case DataResourceBlockKind::Payload:
        return kDataResourceBlockKindPayload;
    case DataResourceBlockKind::SensingPilot:
        return kDataResourceBlockKindSensingPilot;
    default:
        return kDataResourceBlockKindPayload;
    }
}

inline bool parse_data_resource_block_kind_string(
    const std::string& raw_kind,
    DataResourceBlockKind& out_kind)
{
    std::string kind;
    kind.reserve(raw_kind.size());
    for (char ch : raw_kind) {
        if (ch == '-' || ch == ' ') {
            kind.push_back('_');
        } else {
            kind.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    if (kind.empty() || kind == kDataResourceBlockKindPayload) {
        out_kind = DataResourceBlockKind::Payload;
        return true;
    }
    if (kind == kDataResourceBlockKindSensingPilot) {
        out_kind = DataResourceBlockKind::SensingPilot;
        return true;
    }
    return false;
}

struct DataResourceBlock {
    DataResourceBlockKind kind = DataResourceBlockKind::Payload;
    size_t symbol_start = 0;
    size_t symbol_count = 0;
    size_t subcarrier_start = 0;
    size_t subcarrier_count = 0;
};

inline constexpr const char* kSensingOutputModeDense = "dense";
inline constexpr const char* kSensingOutputModeCompactMask = "compact_mask";
inline constexpr const char* kSensingOnWireFormatCf32 = "cf32";
inline constexpr const char* kSensingOnWireFormatCf16 = "cf16";
inline constexpr const char* kCudaLdpcDecoderBackendGpu = "gpu";
inline constexpr const char* kCudaLdpcDecoderBackendCpu = "cpu";

enum class SensingOnWireFormat : uint32_t {
    ComplexFloat32 = 0,
    ComplexFloat16 = 1,
};

/**
 * @brief Shared payload-resource layout derived from Config.
 *
 * The flattened non-pilot order is:
 *   1. OFDM symbols in ascending absolute frame order, skipping reserved sync
 *      symbols and mid-frame pilots
 *   2. Within each data symbol, non-pilot subcarriers in ascending order
 *
 * payload_mask / payload_rank are indexed in that flattened order.
 */
struct DataResourceGridLayout {
    size_t num_symbols = 0;
    size_t fft_size = 0;
    size_t sync_pos = 0;
    size_t data_symbol_count = 0;
    size_t midframe_pilot_symbol_count = 0;
    size_t num_non_pilot_subcarriers = 0;
    size_t non_pilot_re_count = 0;
    size_t payload_re_count = 0;
    size_t sensing_pilot_re_count = 0;

    std::vector<int> midframe_pilot_symbols;
    std::vector<uint8_t> midframe_pilot_symbol_mask;
    std::vector<int> midframe_pilot_symbol_to_rank;
    std::vector<int> data_symbol_to_actual_symbol;
    std::vector<int> actual_symbol_to_data_symbol;
    std::vector<int> non_pilot_subcarrier_indices;
    std::vector<int> subcarrier_to_non_pilot_index;
    std::vector<uint8_t> pilot_mask;
    std::vector<uint8_t> payload_mask;
    std::vector<uint8_t> sensing_pilot_mask;
    std::vector<int> payload_rank;
    std::vector<size_t> payload_offsets;
    std::vector<size_t> non_pilot_offsets;

    size_t flat_non_pilot_index(size_t data_symbol_idx, size_t non_pilot_idx) const {
        return non_pilot_offsets[data_symbol_idx] + non_pilot_idx;
    }
};

/**
 * @brief Selected sensing RE layout for compact-mask sensing output.
 *
 * RE are stored in deterministic wire order:
 *   1. OFDM symbol index ascending
 *   2. FFT-bin index ascending within each symbol
 */
struct SensingMaskLayout {
    size_t num_symbols = 0;
    size_t fft_size = 0;
    size_t total_re_count = 0;
    uint32_t mask_hash = 0;

    std::vector<int> selected_symbols;
    std::vector<int> symbol_to_selected_rank;
    std::vector<size_t> selected_symbol_offsets;
    std::vector<int> flat_subcarrier_indices;

    bool empty() const { return total_re_count == 0; }
};

/**
 * @brief Structural analysis of a compact sensing mask.
 *
 * regular_subsampling_compatible means the compact mask selects:
 *   1. The same subcarrier set for every selected OFDM symbol.
 *   2. OFDM symbols that are equally spaced on the frame ring, including wrap-around.
 *
 * local_delay_doppler_supported is a stricter runtime gate used by the sensing
 * pipelines. It additionally requires the selected symbol count to fit inside
 * doppler_fft_size so the local slow-time FFT path can reuse the configured
 * Doppler dimension safely.
 */
struct CompactSensingMaskAnalysis {
    bool regular_subsampling_compatible = false;
    bool local_delay_doppler_supported = false;
    size_t selected_symbol_count = 0;
    size_t common_subcarrier_count = 0;
    size_t implicit_symbol_stride = 0;
    std::vector<int> selected_symbols;
    std::vector<int> common_subcarrier_indices;
    std::string incompatibility_reason;
    std::string runtime_restriction_reason;

    std::string effective_reason() const {
        if (!runtime_restriction_reason.empty()) {
            return runtime_restriction_reason;
        }
        return incompatibility_reason;
    }
};

struct SensingCfarParams {
    bool enabled = false;
    int train_doppler = 20;
    int train_range = 20;
    int guard_doppler = 10;
    int guard_range = 10;
    float alpha_db = 16.989700f; // 10 * log10(50)
    int min_range_bin = 0;
    int dc_exclusion_bins = 0;
    float min_power_db = 0.0f;
    float os_rank_percent = 75.0f;
    int os_suppress_doppler = 2;
    int os_suppress_range = 2;
    int max_points = 256;
};

struct SensingCfarStats {
    float noise_min = 0.0f;
    float noise_max = 0.0f;
    float thresh_min = 0.0f;
    float thresh_max = 0.0f;
    float power_min_db = 0.0f;
    uint32_t invalid_cells = 0;
    uint32_t nonfinite_cells = 0;
    uint32_t nonpositive_cells = 0;
};

struct SensingDetectionPoint {
    int32_t doppler_idx = 0;
    int32_t range_idx = 0;
};

struct SensingCluster {
    int32_t peak_doppler_idx = 0;
    int32_t peak_range_idx = 0;
    float peak_strength_db = 0.0f;
    uint32_t cluster_size = 0;
    float centroid_doppler_idx = 0.0f;
    float centroid_range_idx = 0.0f;
};

struct SensingMetadata {
    bool cfar_enabled = false;
    bool micro_doppler_enabled = false;
    uint32_t cfar_hits = 0;
    uint32_t cfar_shown_hits = 0;
    SensingCfarStats cfar_stats;
    std::vector<SensingDetectionPoint> cfar_points;
    std::vector<SensingCluster> target_clusters;
    std::vector<float> micro_doppler_spectrum;
    uint32_t micro_doppler_rows = 0;
    uint32_t micro_doppler_cols = 0;
    std::array<float, 4> micro_doppler_extent{{0.0f, 0.0f, 0.0f, 0.0f}};

    bool empty() const {
        return !cfar_enabled &&
               !micro_doppler_enabled &&
               cfar_points.empty() &&
               target_clusters.empty() &&
               micro_doppler_spectrum.empty() &&
               cfar_hits == 0 &&
               cfar_shown_hits == 0;
    }
};

/**
 * @brief Thread-safe Object Pool for Memory Reuse.
 * 
 * Provides a pool of pre-allocated objects that can be acquired and released
 * to avoid repeated heap allocations. Uses mutex-based synchronization for
 * thread safety. Objects are created using a factory function that allows
 * custom initialization (e.g., pre-sized vectors).
 * 
 * Usage pattern:
 * - Producer thread: acquire() -> use object -> move to consumer
 * - Consumer thread: use object -> release() to return to pool
 * 
 * If the pool is empty when acquire() is called, a new object is created
 * using the factory function (graceful degradation).
 * 
 * @tparam T Type of objects to pool (must be movable)
 */
template<typename T>
class ObjectPool {
public:
    using FactoryFunc = std::function<T()>;
    
    /**
     * @brief Construct an object pool with initial capacity.
     * 
     * @param initial_size Number of objects to pre-allocate
     * @param factory Function that creates a new properly-initialized object
     */
    ObjectPool(size_t initial_size, FactoryFunc factory, size_t max_available = 0)
        : _factory(std::move(factory)), _max_available(max_available)
    {
        _pool.reserve(initial_size);
        for (size_t i = 0; i < initial_size; ++i) {
            _pool.push_back(_factory());
        }
    }
    
    /**
     * @brief Acquire an object from the pool.
     * 
     * If the pool is empty, creates a new object using the factory function.
     * The returned object is moved out of the pool.
     * 
     * @return T An object ready for use
     */
    T acquire() {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_pool.empty()) {
            // Pool exhausted, create new object
            return _factory();
        }
        T obj = std::move(_pool.back());
        _pool.pop_back();
        return obj;
    }
    
    /**
     * @brief Return an object to the pool for reuse.
     * 
     * The object is moved into the pool. Caller should not use the
     * object after calling release().
     * 
     * @param obj Object to return to the pool
     */
    void release(T&& obj) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_max_available > 0 && _pool.size() >= _max_available) {
            return;
        }
        _pool.push_back(std::move(obj));
    }
    
    /**
     * @brief Get the number of available objects in the pool.
     * 
     * @return size_t Number of objects currently in the pool
     */
    size_t available() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _pool.size();
    }
    
    /**
     * @brief Pre-allocate additional objects into the pool.
     * 
     * @param count Number of additional objects to create
     */
    void prefill(size_t count) {
        std::lock_guard<std::mutex> lock(_mutex);
        _pool.reserve(_pool.size() + count);
        for (size_t i = 0; i < count; ++i) {
            _pool.push_back(_factory());
        }
    }

private:
    FactoryFunc _factory;
    size_t _max_available = 0;
    std::vector<T> _pool;
    mutable std::mutex _mutex;
};

/**
 * @brief Single-producer single-consumer fixed-slot ring buffer.
 *
 * Storage is pre-allocated at construction/reset time and then reused without
 * further allocation. The producer and consumer communicate only through the
 * atomic head/tail indices, so queue access itself is lock-free.
 *
 * @tparam T Slot type stored in the ring.
 */
template<typename T>
class SPSCRingBuffer {
public:
    using FactoryFunc = std::function<T()>;
    using value_type = T;

    SPSCRingBuffer() = default;

    explicit SPSCRingBuffer(size_t capacity)
    {
        reset(capacity, []() { return T{}; });
    }

    SPSCRingBuffer(size_t capacity, FactoryFunc factory)
    {
        reset(capacity, std::move(factory));
    }

    SPSCRingBuffer(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;

    template<typename Factory>
    void reset(size_t capacity, Factory&& factory) {
        _slots.clear();
        _slots.reserve(capacity);
        for (size_t i = 0; i < capacity; ++i) {
            _slots.push_back(factory());
        }
        _head.store(0, std::memory_order_relaxed);
        _tail.store(0, std::memory_order_relaxed);
    }

    bool empty() const {
        return _head.load(std::memory_order_acquire) ==
               _tail.load(std::memory_order_acquire);
    }

    bool full() const {
        if (_slots.empty()) {
            return true;
        }
        const size_t head = _head.load(std::memory_order_acquire);
        const size_t tail = _tail.load(std::memory_order_acquire);
        return (head - tail) >= _slots.size();
    }

    size_t size() const {
        const size_t head = _head.load(std::memory_order_acquire);
        const size_t tail = _tail.load(std::memory_order_acquire);
        return head - tail;
    }

    size_t capacity() const {
        return _slots.size();
    }

    T* producer_slot() {
        if (_slots.empty()) {
            return nullptr;
        }
        const size_t head = _head.load(std::memory_order_relaxed);
        const size_t tail = _tail.load(std::memory_order_acquire);
        if ((head - tail) >= _slots.size()) {
            return nullptr;
        }
        return &_slots[head % _slots.size()];
    }

    void producer_commit() {
        const size_t head = _head.load(std::memory_order_relaxed);
        _head.store(head + 1, std::memory_order_release);
    }

    template<typename U>
    bool try_push(U&& value) {
        T* slot = producer_slot();
        if (slot == nullptr) {
            return false;
        }
        *slot = std::forward<U>(value);
        producer_commit();
        return true;
    }

    T* consumer_slot() {
        if (_slots.empty()) {
            return nullptr;
        }
        const size_t tail = _tail.load(std::memory_order_relaxed);
        const size_t head = _head.load(std::memory_order_acquire);
        if (tail == head) {
            return nullptr;
        }
        return &_slots[tail % _slots.size()];
    }

    void consumer_pop() {
        const size_t tail = _tail.load(std::memory_order_relaxed);
        _tail.store(tail + 1, std::memory_order_release);
    }

    bool try_pop(T& out) {
        T* slot = consumer_slot();
        if (slot == nullptr) {
            return false;
        }
        out = std::move(*slot);
        consumer_pop();
        return true;
    }

    void clear() {
        _tail.store(_head.load(std::memory_order_acquire), std::memory_order_release);
    }

private:
    std::vector<T> _slots;
    std::atomic<size_t> _head{0};
    std::atomic<size_t> _tail{0};
};

/**
 * @brief Lock-free "latest value" publisher for a fixed-size complex buffer.
 *
 * One writer thread calls store() (e.g. once per received frame); any number
 * of reader threads call load() to snapshot the most recent value. The complex
 * samples are stored as atomic 64-bit slots so readers never race ordinary
 * memory while the seqlock detects overlapping snapshots. resize() must be
 * called once (before any store()/load()) and never again.
 */
class SeqlockedChannelEstimate {
public:
    void resize(size_t n) {
        _size = n;
        _buf = std::make_unique<std::atomic<uint64_t>[]>(n);
        const uint64_t zero = pack_complex(std::complex<float>(0.0f, 0.0f));
        for (size_t i = 0; i < n; ++i) {
            _buf[i].store(zero, std::memory_order_relaxed);
        }
        _seq.store(0, std::memory_order_relaxed);
    }

    // Single-writer only. `value.size()` must equal the size passed to resize().
    void store(const AlignedVector& value) {
        if (!_buf || value.size() != _size) {
            return;
        }
        const uint32_t s = _seq.load(std::memory_order_relaxed);
        _seq.store(s + 1, std::memory_order_release);
        for (size_t i = 0; i < _size; ++i) {
            _buf[i].store(pack_complex(value[i]), std::memory_order_relaxed);
        }
        _seq.store(s + 2, std::memory_order_release);
    }

    // Safe for any number of concurrent readers. Returns false (leaving `out`
    // untouched) if nothing has been published yet.
    bool load(AlignedVector& out) const {
        for (;;) {
            const uint32_t s1 = _seq.load(std::memory_order_acquire);
            if (s1 & 1u) {
                continue;  // writer in progress; spin
            }
            if (s1 == 0) {
                return false;  // never published
            }
            if (!_buf) {
                return false;
            }
            if (out.size() != _size) {
                out.resize(_size);
            }
            for (size_t i = 0; i < _size; ++i) {
                out[i] = unpack_complex(_buf[i].load(std::memory_order_relaxed));
            }
            const uint32_t s2 = _seq.load(std::memory_order_acquire);
            if (s1 == s2) {
                return true;
            }
            // A store overlapped the copy; retry.
        }
    }

private:
    static uint64_t pack_complex(const std::complex<float>& value) {
        uint32_t re_bits = 0;
        uint32_t im_bits = 0;
        const float re = value.real();
        const float im = value.imag();
        std::memcpy(&re_bits, &re, sizeof(re_bits));
        std::memcpy(&im_bits, &im, sizeof(im_bits));
        return static_cast<uint64_t>(re_bits) |
               (static_cast<uint64_t>(im_bits) << 32);
    }

    static std::complex<float> unpack_complex(uint64_t bits) {
        const uint32_t re_bits = static_cast<uint32_t>(bits & 0xffffffffu);
        const uint32_t im_bits = static_cast<uint32_t>(bits >> 32);
        float re = 0.0f;
        float im = 0.0f;
        std::memcpy(&re, &re_bits, sizeof(re));
        std::memcpy(&im, &im_bits, sizeof(im));
        return std::complex<float>(re, im);
    }

    std::atomic<uint32_t> _seq{0};
    size_t _size = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> _buf;
};

/**
 * @brief Relax the CPU without leaving the core (hard real-time safe).
 *
 * Prefer this over yield/sleep on TX/RX sample paths: once a SCHED_FIFO thread
 * sleeps or yields, wake latency is unbounded under load.
 */
inline void rt_cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    // compiler barrier only
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

/**
 * @brief Busy-spin helper for hard real-time SPSC polling / short-send retry.
 *
 * Never yields or sleeps — the thread keeps the CPU. Use on USRP TX/RX threads
 * that must meet timed deadlines. Prefer @ref SPSCBackoff on best-effort
 * workers (UDP, LDPC, etc.) where burning a core is undesirable.
 */
class SPSCBusySpin {
public:
    void reset() {
        _spins = 0;
    }

    void pause() {
        ++_spins;
        rt_cpu_relax();
    }

    uint32_t spins() const { return _spins; }

private:
    uint32_t _spins = 0;
};

/**
 * @brief Cooperative backoff helper for lock-free SPSC queue polling.
 *
 * Starts with a few scheduler yields, then falls back to short sleeps if the
 * queue stays empty/full. This avoids mutex blocking on the hot path while
 * keeping idle CPU burn bounded.
 *
 * @warning Not for hard real-time sample TX/RX threads — yield/sleep can
 *          postpone the next wake indefinitely. Use @ref SPSCBusySpin there.
 */
class SPSCBackoff {
public:
    void reset() {
        _attempts = 0;
    }

    void pause() {
        if (_attempts < 16) {
            ++_attempts;
            std::this_thread::yield();
            return;
        }
        if (_attempts < 64) {
            ++_attempts;
            std::this_thread::sleep_for(std::chrono::microseconds(25));
            return;
        }
        ++_attempts;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

private:
    uint32_t _attempts = 0;
};

template<typename Queue, typename U, typename StopPredicate>
bool spsc_wait_push(Queue& queue, U&& value, StopPredicate&& should_stop) {
    SPSCBackoff backoff;
    while (true) {
        if (queue.try_push(std::forward<U>(value))) {
            backoff.reset();
            return true;
        }
        if (should_stop()) {
            return false;
        }
        backoff.pause();
    }
}

template<typename Queue, typename T, typename StopPredicate>
bool spsc_wait_pop(Queue& queue, T& out, StopPredicate&& should_stop) {
    SPSCBackoff backoff;
    while (true) {
        if (queue.try_pop(out)) {
            backoff.reset();
            return true;
        }
        if (should_stop()) {
            return false;
        }
        backoff.pause();
    }
}

struct OfdmFrameConfig {
    size_t fft_size = 1024;
    size_t cp_length = 128;            // Cyclic prefix length
    size_t num_symbols = 100;          // Number of symbols per frame
    size_t sensing_symbol_num = 100;   // Number of sensing symbols
    size_t frame_queue_size = 8;       // Capacity of demod RX frame queue
    size_t sync_pos = 1;               // Synchronization symbol position
    bool enable_sec_sync_symbol = false; // Reserve sync_pos-1 for the duplicate second sync symbol
    bool enable_cfo_training_sequence = false; // Reserve sync_pos+1 for a repeated CFO training symbol
    size_t cfo_training_period_samples = 16; // Repetition period of the CFO training symbol in samples
    int zc_root = 29;                  // Zadoff-Chu sequence root index
    std::vector<size_t> pilot_positions = {571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451};
    std::vector<size_t> midframe_pilot_symbols; // Absolute mid-frame pilot symbols; comb pilot RE are preserved
    uint32_t midframe_pilot_seed = 0x4D46504Cu; // Deterministic BPSK pilot seed ("MFPL")
};

struct CudaConfig {
    size_t cuda_mod_pipeline_slots = 2;   // Number of CUDA mod pipeline slots
    size_t cuda_demod_pipeline_slots = 3; // Number of CUDA demod pipeline slots
    std::string cuda_ldpc_decoder_backend = kCudaLdpcDecoderBackendGpu; // CUDA demod LDPC backend: gpu/cpu
    size_t cuda_ldpc_worker_buffers = 3;  // Number of CUDA LDPC async worker batch buffers
    size_t cuda_ldpc_cross_frame_flush_frames = 2; // Max frames to accumulate before CUDA LDPC batch decode
    double cuda_ldpc_cross_frame_flush_us = 1000.0; // Max CUDA LDPC cross-frame batch wait time
};

struct RfSamplingConfig {
    double sample_rate = 50e6;         // Sample rate
    double bandwidth = 50e6;           // Bandwidth
    double rx_gain = 0.0;              // UE downlink RX gain
    bool rx_agc_enable = false;        // Enable hardware RX AGC via USRP gain control
    double rx_agc_low_threshold_db = 11.0; // Increase gain when delay-spectrum peak is below this threshold
    double rx_agc_high_threshold_db = 13.0; // Decrease gain when delay-spectrum peak is above this threshold
    double rx_agc_max_step_db = 3.0;   // Maximum gain change per AGC update
    size_t rx_agc_update_frames = 4;   // Frame interval between AGC updates
};

struct UsrpDeviceConfig {
    std::string device_args = "";
};

struct ClockTimeConfig {
    std::string clock_source = "internal"; // Clock source
    std::string time_source = "";         // Time source; empty means follow clock_source
};

struct SyncTrackingConfig {
    size_t sync_queue_size = 8;        // Capacity of demod sync-search queue
    double sync_cfo_alias_search_range_hz = 800000.0; // Max absolute CFO span covered by sync alias search
    int delay_adjust_step = 2;         // Delay adjustment step
    double reset_hold_s = 0.5;         // Time window of persistent invalid delay before forcing a hard reset
    int desired_peak_pos = 20;         // Desired delay peak position to include non-causal components
    bool predictive_delay = true;      // Enable CFO-based predictive delay compensation during alignment/tracking
    bool software_sync = true;         // Software synchronization flag
    bool hardware_sync = false;        // Hardware synchronization flag
    std::string hardware_sync_tty = "/dev/ttyUSB0"; // Hardware sync TTY device
    double ocxo_pi_kp_fast = 30.0;    // Fast-stage PI proportional gain Kp
    double ocxo_pi_ki_fast = 1.0;     // Fast-stage PI integral gain Ki
    double ocxo_pi_kp_slow = 30.0;    // Slow-stage PI proportional gain Kp
    double ocxo_pi_ki_slow = 0.05;    // Slow-stage PI integral gain Ki
    double ocxo_pi_switch_abs_error_ppm = 0.0002; // Switch threshold: absolute error_ppm
    double ocxo_pi_switch_hold_s = 60.0;    // Required stable duration below threshold (seconds)
    double ocxo_pi_max_step_fast_ppm = 0.01; // Fast-stage per-update OCXO adjustment clamp (ppm)
    double ocxo_pi_max_step_slow_ppm = 0.01; // Slow-stage per-update OCXO adjustment clamp (ppm)
    bool akf_enable = true;           // Enable adaptive Kalman filter for hardware sync error_ppm
    size_t akf_bootstrap_frames = 64; // Cold-start frame count before KF update starts
    size_t akf_innovation_window = 64; // Innovation history length used for ACF/LS adaptation
    size_t akf_max_lag = 4;           // Maximum innovation autocorrelation lag
    size_t akf_adapt_interval = 64;   // Frames between Q/R adaptation updates
    double akf_gate_sigma = 3.0;      // Innovation gating threshold in sigma units
    double akf_tikhonov_lambda = 1e-3; // Tikhonov regularization for LS adaptation
    double akf_update_smooth = 0.2;   // Smoothing factor for adapted Q/R updates
    double akf_q_wf_min = 1e-10;      // Lower bound of white-frequency-noise coefficient
    double akf_q_wf_max = 1e2;        // Upper bound of white-frequency-noise coefficient
    double akf_q_rw_min = 1e-12;      // Lower bound of random-walk-frequency-noise coefficient
    double akf_q_rw_max = 1e1;        // Upper bound of random-walk-frequency-noise coefficient
    double akf_r_min = 1e-8;          // Lower bound of observation noise variance R
    double akf_r_max = 1e3;           // Upper bound of observation noise variance R

};

struct EqualizerConfig {
    std::string equalizer_mode = kEqualizerModeMmse; // zf or mmse
    std::string channel_tracking_mode = kChannelTrackingModePilotPhase; // disabled or pilot_phase
    double equalizer_mag_floor = 1e-6; // Lower bound for |H|^2 in channel inversion
    double channel_tracking_min_pilot_snr = 1e-4; // Minimum pilot residual weight before falling back
};

struct DownlinkConfig {
    double center_freq = 2.4e9;        // Downlink RF center frequency
    double tx_gain = 0.0;              // BS downlink TX gain
    uint32_t tx_channel = 0;           // BS downlink TX channel index
    std::string tx_device_args = "";
    std::string tx_clock_source = "";
    std::string tx_time_source = "";
    std::string wire_format_tx = "sc16";
    size_t rx_channel = 0;             // UE downlink RX channel index
    std::string rx_wire_format = "sc16"; // UE downlink RX wire format
    EqualizerConfig equalizer;
};

struct DownlinkPipelineConfig {
    size_t tx_circular_buffer_size = 8;    // Capacity of BS frame circular buffer
    size_t data_packet_buffer_size = 256;  // Capacity of BS encoded packet buffer
};

struct UplinkConfig {
    bool enabled = false;               // UE->BS uplink master switch
    DuplexConfig duplex;                // Duplexing (TDD/FDD) + uplink frame structure
    int32_t bs_dl_ul_timing_diff = 63; // BS: uplink-RX window offset relative to the TX frame anchor (samples)
    int32_t ue_timing_advance = 63;    // UE: uplink-TX window advance relative to the RX frame anchor (samples)
    std::string idle_waveform = kUplinkIdleWaveformRandomQpsk; // UE idle UL payload RE: zero or random_qpsk
    bool debug_self_channel = false; // Estimate local-TX leakage channel from RX windows for DUTI/TADV debug
    bool debug_self_scan_spectrum = false; // Publish a peak-centered self-ZC matched-filter slice + metadata (needs debug_self_channel)
    size_t debug_self_scan_slice_samples = 0; // Correlation samples around the peak; 0 = one OFDM symbol (fft_size + cp_length)
    bool ertm_to_enable = false;      // Enable eRTM timing-offset estimation payloads/logs
    std::string ertm_timing_metric = kErtmTimingMetricDelayMagnitude; // Differential-TO metric
    size_t ertm_delay_oversample_factor = 10; // UE eRTM delay-spectrum IFFT oversampling factor
    bool ertm_debug_output_enabled = false; // Publish UE-side eRTM debug spectra over ZMQ
    double ertm_dl_rf_delay_samples = 0.0;  // Calibrated downlink RF-chain delay in samples
    double ertm_ul_rf_delay_samples = 0.0;  // Calibrated uplink RF-chain delay in samples
    size_t ertm_report_interval_frames = 32; // BS eRTM payload cadence in TX frames
    double rx_gain = 0.0;              // BS uplink RX gain
    size_t rx_channel = 0;             // BS uplink RX channel index
    std::string rx_wire_format = "sc16"; // BS uplink RX wire format
    std::string rx_device_args = "";   // BS uplink RX device args override (empty = shared TX device)
    std::string rx_clock_source = "";  // BS uplink RX clock source override (empty = TX clock source)
    std::string rx_time_source = "";   // BS uplink RX time source override (empty = TX time source)
    double tx_gain = 0.0;              // UE uplink TX gain
    uint32_t tx_channel = 0;           // UE uplink TX channel index
    std::string wire_format_tx = "sc16"; // UE uplink TX wire format
    EqualizerConfig equalizer;         // BS uplink equalizer/tracking
};

struct SensingConfig {
    size_t range_fft_size = 1024;      // Range FFT size
    size_t doppler_fft_size = 100;     // Doppler FFT size
    size_t view_range_bins = 0;        // Backend RD view width (0 = full range_fft_size)
    size_t view_doppler_bins = 0;      // Backend RD view height (0 = full doppler_fft_size)
    std::string output_mode = kSensingOutputModeDense;
    SensingOnWireFormat on_wire_format = SensingOnWireFormat::ComplexFloat32;
    bool backend_processing_enabled = false;
    std::vector<DataResourceBlock> mask_blocks;
    std::string rx_device_args = "";
    std::string rx_clock_source = "";
    std::string rx_time_source = "";
    std::string rx_wire_format = "sc16";  // BS sensing RX default wire format
    uint32_t rx_channel_count = 1; // Number of sensing RX channels
    std::vector<SensingRxChannelConfig> rx_channels; // Per-channel sensing RX config
    size_t symbol_stride = 20;     // Default sensing STRD applied at startup
    size_t paired_frame_queue_size = 16;   // Keep headroom above the default TX frame queue
    bool bi_enabled = true;
    std::string delay_correction_mode = kSensingDelayCorrectionModeLosTracking; // los_tracking or ertm_absolute
};

struct RadioConfig {
    std::string radio_backend = "uhd";  // Radio I/O backend: "uhd" (real USRP) or "sim" (channel simulator)
};

struct UdpEgressPacerConfig {
    bool enabled = false;             // Enable queued/paced UDP egress for decoded payload streams
    double target_mbps = 0.0;         // <=0: auto-estimate from enqueue rate; >0: fixed payload Mbps
    size_t queue_packets = 10240;     // Max queued UDP datagrams before dropping oldest
    double max_delay_ms = 0.0;        // Drop queued datagrams older than this; <=0 disables age drop
};

// ARQ sequence comparisons use signed 16-bit distance. The active window must
// remain below half of the uint16_t sequence space to keep ordering unambiguous.
inline constexpr int kMaxArqWindowPackets = std::numeric_limits<int16_t>::max();

struct NetworkOutputConfig {
    std::string default_out_ip = "127.0.0.1";
    bool mono_sensing_output_enabled = true;
    std::string mono_sensing_ip = "0.0.0.0";
    int mono_sensing_port = 8888;
    bool bi_sensing_output_enabled = true;
    std::string bi_sensing_ip = "0.0.0.0";
    int bi_sensing_port = 8889;
    int control_port = 9999;
    std::string uplink_channel_ip = "0.0.0.0";
    int uplink_channel_port = 12358;
    std::string uplink_pdf_ip = "0.0.0.0";
    int uplink_pdf_port = 12359;
    std::string uplink_constellation_ip = "0.0.0.0";
    int uplink_constellation_port = 12356;
    std::string uplink_self_channel_ip = "0.0.0.0";
    int uplink_self_channel_port = 12360;
    std::string uplink_self_pdf_ip = "0.0.0.0";
    int uplink_self_pdf_port = 12361;
    std::string uplink_self_scan_ip = "0.0.0.0";
    int uplink_self_scan_port = 12352;
    std::string channel_ip = "0.0.0.0";
    int channel_port = 12348;
    std::string pdf_ip = "0.0.0.0";
    int pdf_port = 12349;
    std::string constellation_ip = "0.0.0.0";
    int constellation_port = 12346;
    std::string vofa_debug_ip = "127.0.0.1";
    int vofa_debug_port = 12347;
    std::string ertm_debug_ip = "0.0.0.0";
    int ertm_debug_port = 12362;
    std::string udp_output_ip = "127.0.0.1";
    int udp_output_port = 50001;
    std::string udp_input_ip = "0.0.0.0"; // bind address
    int udp_input_port = 50000;
    std::string ul_udp_input_ip = "0.0.0.0"; // UE uplink payload UDP input
    int ul_udp_input_port = 50002;
    std::string ul_udp_output_ip = "127.0.0.1"; // BS decoded uplink UDP output
    int ul_udp_output_port = 50003;
    UdpEgressPacerConfig udp_egress_pacer;

    // ARQ link-layer retransmission
    bool arq_enabled = false;
    bool arq_ordered_delivery = false;
    int arq_window_packets = 256;        // Default 256; protocol range [1, 32767]
    int arq_ack_bitmap_bits = 64;        // fixed 64 for first pass
    int arq_retransmit_timeout_ms = 100; // RTO in ms; 0 => default RTO
    int arq_max_retries = 5;             // 0 = unlimited within window
    int arq_feedback_interval_ms = 10;   // min interval between ACK feedback packets
};

struct CpuCoresConfig {
    std::vector<int> downlink_cpu_cores; // BS: TX/mod/LDPC-encode/UDP-recv; UE: RX/process/bit-processing
    std::vector<int> demod_worker_cpu_cores; // UE CPU demod worker cores; empty = one unbound worker
    std::vector<int> ldpc_worker_cpu_cores;  // UE CPU LDPC decode worker cores; empty = one unbound worker
    std::vector<int> sensing_cpu_cores;  // Dedicated sensing processing cores; empty = no explicit sensing binding
    std::vector<int> uplink_cpu_cores;   // Dedicated uplink thread cores; empty = no explicit uplink binding
    int main_cpu_core = -1;              // Main-thread affinity; -1 = no explicit binding
};

/**
 * @brief Hierarchical text-log filter (see plans/active/hierarchical_logging.md).
 *
 * YAML:
 * @code
 * logging:
 *   default_level: warn
 *   timestamps: false
 *   force_error: true
 *   modules:
 *     demod.ldpc: debug
 *     cuda.demod: warn
 * @endcode
 *
 * Module paths match async_logger::LogModule registry (demod, cuda.ldpc, ...).
 * Unlisted modules inherit their parent; root inherits default_level.
 * Error is always printed when force_error is true.
 */
struct LoggingConfig {
    std::string default_level = "warn";  // off|error|warn|info|debug
    bool timestamps = false;
    bool force_error = true;             // unrecoverable Error always emits
    // path -> level name (off|error|warn|info|debug). Missing paths inherit.
    std::map<std::string, std::string> modules;
};

struct MeasurementConfig {
    bool measurement_enable = false;
    std::string measurement_mode = "";
    std::string measurement_run_id = "";
    std::string measurement_output_dir = "";
    size_t measurement_payload_bytes = 1024;
    uint32_t measurement_prbs_seed = 0x5A;
    uint32_t measurement_packets_per_point = 1;
    size_t measurement_max_packets_per_frame = 1; // 0 = unlimited
};

struct ResourcePreviewConfig {
    bool data_resource_blocks_configured = false;
    std::vector<DataResourceBlock> data_resource_blocks;
    size_t payload_re_count = 0;
    size_t non_pilot_re_count = 0;
};

/**
 * @brief Typed runtime configuration.
 *
 * Mirrors the sectioned YAML layout. YAML is parsed once into this structure;
 * hot paths use typed fields instead of querying YAML nodes.
 */
// CPU LDPC decode precision options. Applies to both UE downlink RX and BS
// uplink RX. Default reproduces the float32 decode path exactly.
struct LdpcDecodeConfig {
    bool fixed_point = false;     // use int16 (Q16) layered-NMS decoder instead of float32
    int fixed_point_scale = 16;   // pow2 multiplier applied at the demapper before int16 saturation
};

struct Config {
    RadioConfig radio;
    SimConfig simulation;               // Channel simulator parameters (used when radio.radio_backend == "sim")
    OfdmFrameConfig ofdm;
    CudaConfig cuda;
    SensingConfig sensing;
    RfSamplingConfig rf_sampling;
    UsrpDeviceConfig usrp_device;
    ClockTimeConfig clock_time;
    DownlinkConfig downlink;
    DownlinkPipelineConfig downlink_pipeline;
    UplinkConfig uplink;
    SyncTrackingConfig sync_tracking;
    MeasurementConfig measurement;
    NetworkOutputConfig network_output;
    NetworkOutputConfig uplink_arq;
    CpuCoresConfig cpu_cores;
    LoggingConfig logging;
    ResourcePreviewConfig resource_preview;
    LdpcDecodeConfig ldpc;

    // Calculate total samples per frame
    size_t samples_per_frame() const { 
        return ofdm.num_symbols * (ofdm.fft_size + ofdm.cp_length);
    }
    
    // Calculate synchronization samples
    size_t sync_samples() const { 
        return 2 * samples_per_frame(); 
    }
};

inline bool uplink_self_channel_debug_enabled(const Config& cfg) {
    return cfg.uplink.debug_self_channel && cfg.uplink.duplex.mode != DuplexMode::FDD;
}

// The full-frame self-ZC scan spectrum is an add-on to self-channel debug: it
// reuses the same RX frames but runs a whole-frame matched filter to locate the
// UE's own uplink ZC, so it requires self-channel debug to be enabled too.
inline bool uplink_self_scan_spectrum_enabled(const Config& cfg) {
    return cfg.uplink.debug_self_scan_spectrum && uplink_self_channel_debug_enabled(cfg);
}

inline bool sec_sync_symbol_enabled(const Config& cfg) {
    return cfg.ofdm.enable_sec_sync_symbol;
}

inline bool cfo_training_sequence_enabled(const Config& cfg) {
    return cfg.ofdm.enable_cfo_training_sequence;
}

inline size_t cfo_training_symbol_index(const Config& cfg) {
    return cfg.ofdm.sync_pos + 1;
}

inline bool is_sec_sync_symbol(const Config& cfg, size_t symbol_idx) {
    return sec_sync_symbol_enabled(cfg) &&
           cfg.ofdm.sync_pos > 0 &&
           symbol_idx + 1 == cfg.ofdm.sync_pos;
}

inline bool is_main_sync_symbol(const Config& cfg, size_t symbol_idx) {
    return symbol_idx == cfg.ofdm.sync_pos;
}

inline bool is_zc_sync_symbol(const Config& cfg, size_t symbol_idx) {
    return is_main_sync_symbol(cfg, symbol_idx) || is_sec_sync_symbol(cfg, symbol_idx);
}

inline bool is_cfo_training_symbol(const Config& cfg, size_t symbol_idx) {
    return cfo_training_sequence_enabled(cfg) &&
           cfg.ofdm.sync_pos < std::numeric_limits<size_t>::max() &&
           symbol_idx == cfo_training_symbol_index(cfg);
}

inline bool is_reserved_sync_symbol(const Config& cfg, size_t symbol_idx) {
    return is_zc_sync_symbol(cfg, symbol_idx) || is_cfo_training_symbol(cfg, symbol_idx);
}

inline const char* reserved_sync_symbol_label(const Config& cfg, size_t symbol_idx) {
    if (is_main_sync_symbol(cfg, symbol_idx)) {
        return "main ZC sync symbol";
    }
    if (is_sec_sync_symbol(cfg, symbol_idx)) {
        return "second sync symbol";
    }
    if (is_cfo_training_symbol(cfg, symbol_idx)) {
        return "CFO training field";
    }
    return "reserved sync/training symbol";
}

inline size_t reserved_sync_symbol_count(const Config& cfg) {
    size_t count = 1;
    if (sec_sync_symbol_enabled(cfg)) {
        ++count;
    }
    if (cfo_training_sequence_enabled(cfg)) {
        ++count;
    }
    return count;
}

/**
 * @brief Per-frame partition of OFDM symbols into downlink / uplink / guard.
 *
 * Built identically on the BS and UE from the shared DuplexConfig so both ends
 * agree on which symbols carry uplink. In TDD the uplink occupies a contiguous
 * symbol range whose leading `ul_guard` symbols are a blanked DL->UL guard. In
 * FDD every symbol is simultaneously downlink (on the DL carrier) and uplink
 * (on the UL carrier), with no guard.
 */
struct DuplexFrameLayout {
    DuplexMode mode = DuplexMode::TDD;
    size_t num_symbols = 0;
    size_t ul_start = 0;     // TDD: first symbol of the uplink range
    size_t ul_count = 0;     // TDD: uplink range length (includes guard)
    size_t ul_guard = 0;     // TDD: leading guard symbols within the uplink range
    bool uplink_enabled = false;
    std::vector<uint8_t> symbol_is_uplink; // TDD per-symbol UL-data mask (size num_symbols)
    std::vector<uint8_t> symbol_is_guard;  // TDD per-symbol guard mask (size num_symbols)
    std::string warning;     // non-empty if the configured range was clamped/invalid

    bool is_uplink(size_t s) const {
        if (mode == DuplexMode::FDD) return uplink_enabled;
        return s < symbol_is_uplink.size() && symbol_is_uplink[s] != 0;
    }
    bool is_guard(size_t s) const {
        if (mode == DuplexMode::FDD) return false;
        return s < symbol_is_guard.size() && symbol_is_guard[s] != 0;
    }
    bool is_downlink(size_t s) const {
        if (mode == DuplexMode::FDD) return true;       // DL carrier always active
        return !is_uplink(s) && !is_guard(s);
    }
    // Start sample (within a frame) of the contiguous uplink data block (TDD).
    size_t ul_sample_offset(const Config& cfg) const {
        if (mode == DuplexMode::FDD) return 0;
        return (ul_start + ul_guard) * (cfg.ofdm.fft_size + cfg.ofdm.cp_length);
    }
    // Number of samples in the uplink data block.
    size_t ul_sample_count(const Config& cfg) const {
        if (mode == DuplexMode::FDD) return cfg.samples_per_frame();
        const size_t ul_data_syms = (ul_count > ul_guard) ? (ul_count - ul_guard) : 0;
        return ul_data_syms * (cfg.ofdm.fft_size + cfg.ofdm.cp_length);
    }
};

inline DuplexFrameLayout build_duplex_frame_layout(const Config& cfg) {
    DuplexFrameLayout layout;
    layout.mode = cfg.uplink.duplex.mode;
    layout.num_symbols = cfg.ofdm.num_symbols;
    if (!cfg.uplink.enabled) {
        layout.uplink_enabled = false;
        layout.symbol_is_uplink.assign(cfg.ofdm.num_symbols, 0);
        layout.symbol_is_guard.assign(cfg.ofdm.num_symbols, 0);
        return layout;
    }

    if (cfg.uplink.duplex.mode == DuplexMode::FDD) {
        // FDD: continuous uplink on a separate carrier; no symbol gating/guard.
        layout.uplink_enabled = true;
        layout.ul_start = 0;
        layout.ul_count = cfg.ofdm.num_symbols;
        layout.ul_guard = 0;
        return layout;
    }

    // TDD: validate and clamp the configured uplink symbol range.
    size_t ul_start = cfg.uplink.duplex.ul_symbol_start;
    size_t ul_count = cfg.uplink.duplex.ul_symbol_count;
    size_t ul_guard = cfg.uplink.duplex.ul_guard_symbols;

    if (ul_count == 0) {
        layout.uplink_enabled = false;   // uplink disabled
        layout.symbol_is_uplink.assign(cfg.ofdm.num_symbols, 0);
        layout.symbol_is_guard.assign(cfg.ofdm.num_symbols, 0);
        return layout;
    }

    if (ul_start >= cfg.ofdm.num_symbols) {
        layout.warning = "uplink symbol_start beyond frame; uplink disabled";
        layout.uplink_enabled = false;
        layout.symbol_is_uplink.assign(cfg.ofdm.num_symbols, 0);
        layout.symbol_is_guard.assign(cfg.ofdm.num_symbols, 0);
        return layout;
    }
    if (ul_start + ul_count > cfg.ofdm.num_symbols) {
        layout.warning = "uplink range exceeds frame; clamped to num_symbols";
        ul_count = cfg.ofdm.num_symbols - ul_start;
    }
    if (ul_guard > ul_count) {
        layout.warning = "uplink guard exceeds uplink range; clamped";
        ul_guard = ul_count;
    }

    layout.ul_start = ul_start;
    layout.ul_count = ul_count;
    layout.ul_guard = ul_guard;
    layout.uplink_enabled = (ul_count > ul_guard);

    layout.symbol_is_uplink.assign(cfg.ofdm.num_symbols, 0);
    layout.symbol_is_guard.assign(cfg.ofdm.num_symbols, 0);
    for (size_t s = ul_start; s < ul_start + ul_count; ++s) {
        if (s < ul_start + ul_guard) {
            layout.symbol_is_guard[s] = 1;          // leading guard symbols
        } else {
            if (is_reserved_sync_symbol(cfg, s)) {
                throw std::runtime_error(
                    "TDD uplink data range overlaps reserved DL " +
                    std::string(reserved_sync_symbol_label(cfg, s)) +
                    " at symbol " + std::to_string(s) + '.');
            }
            layout.symbol_is_uplink[s] = 1;         // uplink data symbols
        }
    }
    return layout;
}

inline std::vector<uint8_t> build_cfo_tracking_skip_mask(
    const Config& cfg,
    const DuplexFrameLayout& duplex_layout)
{
    std::vector<uint8_t> skip_mask(cfg.ofdm.num_symbols, 0);
    if (duplex_layout.mode != DuplexMode::TDD ||
        !duplex_layout.uplink_enabled ||
        cfg.ofdm.num_symbols == 0) {
        return skip_mask;
    }

    for (size_t sym = 0; sym < cfg.ofdm.num_symbols; ++sym) {
        if (duplex_layout.is_uplink(sym) || duplex_layout.is_guard(sym)) {
            skip_mask[sym] = 1;
        }
    }
    for (size_t sym = 0; sym < cfg.ofdm.num_symbols; ++sym) {
        if (duplex_layout.is_uplink(sym) || duplex_layout.is_guard(sym)) {
            continue;
        }
        const size_t prev = (sym == 0) ? (cfg.ofdm.num_symbols - 1) : (sym - 1);
        const size_t next = (sym + 1 == cfg.ofdm.num_symbols) ? 0 : (sym + 1);
        if (duplex_layout.is_uplink(prev) || duplex_layout.is_guard(prev) ||
            duplex_layout.is_uplink(next) || duplex_layout.is_guard(next)) {
            skip_mask[sym] = 1;
        }
    }
    return skip_mask;
}

inline int normalize_config_zc_root(int zc_root, size_t fft_size) {
    if (fft_size == 0) {
        return zc_root;
    }
    const int n = static_cast<int>(fft_size);
    int normalized = zc_root % n;
    if (normalized < 0) {
        normalized += n;
    }
    return normalized;
}

inline int select_distinct_zc_root(
    size_t fft_size,
    int base_root,
    const std::vector<int>& additional_avoid_roots = {})
{
    if (fft_size <= 1) {
        return base_root + 1;
    }

    const int n = static_cast<int>(fft_size);
    std::vector<int> avoid_roots;
    avoid_roots.reserve(additional_avoid_roots.size() + 1);
    avoid_roots.push_back(normalize_config_zc_root(base_root, fft_size));
    for (const int root : additional_avoid_roots) {
        avoid_roots.push_back(normalize_config_zc_root(root, fft_size));
    }

    auto is_avoided = [&](int candidate) {
        const int normalized = normalize_config_zc_root(candidate, fft_size);
        return std::find(avoid_roots.begin(), avoid_roots.end(), normalized) != avoid_roots.end();
    };

    const int normalized_base_root = normalize_config_zc_root(base_root, fft_size);
    for (int step = 1; step < n; ++step) {
        const int candidate = (normalized_base_root + step) % n;
        if (candidate == 0 || is_avoided(candidate)) {
            continue;
        }
        if (std::gcd(candidate, n) == 1) {
            return candidate;
        }
    }

    for (int candidate = 1; candidate < n; ++candidate) {
        if (!is_avoided(candidate)) {
            return candidate;
        }
    }

    return base_root + 1;
}

// Derive the self-contained uplink OFDM (sub)frame configuration from the link
// config. The uplink is its own compact frame (ZC sync at symbol 0, comb pilots,
// payload on the remaining REs) that occupies the uplink symbol window of the DL
// frame period. It reuses the DL numerology (fft_size/cp_length/pilots), chooses
// a distinct uplink ZC root, and drops DL-only features the uplink does not use
// (sec-sync, CFO training, midframe pilots, sensing pilots, custom data-resource
// blocks).
//
// TDD: num_symbols = ul_symbol_count - ul_guard_symbols (the data-bearing part of
//      the uplink window). FDD: num_symbols = cfg.ofdm.num_symbols (continuous uplink).
inline Config make_uplink_config(const Config& cfg) {
    Config ul = cfg;
    const DuplexFrameLayout dl = build_duplex_frame_layout(cfg);
    const size_t ul_syms = !cfg.uplink.enabled
        ? 0
        : ((cfg.uplink.duplex.mode == DuplexMode::FDD)
            ? cfg.ofdm.num_symbols
            : ((dl.ul_count > dl.ul_guard) ? (dl.ul_count - dl.ul_guard) : 0));
    ul.ofdm.num_symbols = ul_syms;
    ul.ofdm.sync_pos = 0;
    const int sensing_pilot_root = select_distinct_zc_root(cfg.ofdm.fft_size, cfg.ofdm.zc_root);
    ul.ofdm.zc_root = select_distinct_zc_root(
        cfg.ofdm.fft_size,
        cfg.ofdm.zc_root,
        std::vector<int>{sensing_pilot_root});
    ul.ofdm.enable_sec_sync_symbol = false;
    ul.ofdm.enable_cfo_training_sequence = false;
    ul.ofdm.midframe_pilot_symbols.clear();
    ul.sensing.mask_blocks.clear();
    ul.resource_preview.data_resource_blocks.clear();
    ul.resource_preview.data_resource_blocks_configured = false;
    // The uplink does not run sensing; keep sensing_symbol_num consistent.
    ul.ofdm.sensing_symbol_num = ul_syms;
    return ul;
}

// True when the uplink carries at least one data-bearing symbol (a ZC sync plus
// at least one data symbol). Uses make_uplink_config()'s derived num_symbols.
inline bool uplink_enabled(const Config& cfg) {
    return make_uplink_config(cfg).ofdm.num_symbols >= 2;
}

// One-line startup summary of the duplex frame partition. `role` is "BS" or "UE".
inline void log_duplex_summary(const Config& cfg, const char* role) {
    const DuplexFrameLayout dl = build_duplex_frame_layout(cfg);
    if (!uplink_enabled(cfg)) {
        LOG_G_INFO_M(Radio) << "[" << role << "] duplex: uplink DISABLED (downlink-only); "
                     << "num_symbols=" << cfg.ofdm.num_symbols;
        return;
    }
    const Config ul = make_uplink_config(cfg);
    std::ostringstream oss;
    oss << "[" << role << "] duplex partition: mode="
        << duplex_mode_to_string(cfg.uplink.duplex.mode)
        << ", frame_symbols=" << cfg.ofdm.num_symbols;
    if (cfg.uplink.duplex.mode == DuplexMode::TDD) {
        const size_t dl_syms = cfg.ofdm.num_symbols - dl.ul_count;
        oss << ", DL=" << dl_syms << ", guard=" << dl.ul_guard
            << ", UL=" << ul.ofdm.num_symbols
            << " (UL symbols [" << (dl.ul_start + dl.ul_guard) << ","
            << (dl.ul_start + dl.ul_count) << "))";
    } else {
        oss << ", UL=" << ul.ofdm.num_symbols << " (continuous), ul_center_freq="
            << cfg.uplink.duplex.ul_center_freq << " Hz";
    }
    LOG_G_INFO_M(Config) << oss.str();
}

inline bool validate_cfo_training_period(const Config& cfg, std::string* error = nullptr) {
    if (!cfo_training_sequence_enabled(cfg)) {
        return true;
    }
    if (cfg.ofdm.cfo_training_period_samples == 0) {
        if (error) *error = "cfo_training_period_samples must be greater than 0.";
        return false;
    }
    if (cfg.ofdm.cfo_training_period_samples >= cfg.ofdm.fft_size) {
        if (error) {
            *error = "cfo_training_period_samples must be smaller than fft_size.";
        }
        return false;
    }
    if ((cfg.ofdm.fft_size % cfg.ofdm.cfo_training_period_samples) != 0) {
        if (error) {
            *error = "cfo_training_period_samples must divide fft_size so the CFO field repeats exactly.";
        }
        return false;
    }
    return true;
}

inline bool has_reserved_sync_symbol_overlap(const Config& cfg) {
    if (!cfo_training_sequence_enabled(cfg)) {
        return false;
    }
    const size_t cfo_sym = cfo_training_symbol_index(cfg);
    return cfo_sym == cfg.ofdm.sync_pos || is_sec_sync_symbol(cfg, cfo_sym);
}

inline bool dense_sensing_stride_hits_symbol(
    const Config& cfg,
    size_t stride,
    size_t symbol_idx)
{
    if (cfg.ofdm.num_symbols == 0 || stride == 0 || symbol_idx >= cfg.ofdm.num_symbols) {
        return false;
    }
    return (symbol_idx % std::gcd(stride, cfg.ofdm.num_symbols)) == 0;
}

inline std::string dense_sensing_stride_cfo_training_error(
    const Config& cfg,
    size_t stride,
    const std::string& context)
{
    if (cfg.ofdm.num_symbols == 0 || stride == 0 || !cfo_training_sequence_enabled(cfg)) {
        return {};
    }
    const size_t sym = cfo_training_symbol_index(cfg);
    if (sym < cfg.ofdm.num_symbols && dense_sensing_stride_hits_symbol(cfg, stride, sym)) {
        return context + " selects symbol " + std::to_string(sym) + ", which is the CFO training field. "
               "CFO training fields are not valid sensing symbols; choose a sensing_symbol_stride "
               "that does not sample sync_pos+1.";
    }
    return {};
}

inline void validate_dense_sensing_stride(
    const Config& cfg,
    size_t stride,
    const std::string& context)
{
    const std::string error = dense_sensing_stride_cfo_training_error(cfg, stride, context);
    if (!error.empty()) {
        throw std::runtime_error(error);
    }
}

inline std::string normalize_sensing_output_mode_string(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (mode.empty()) {
        return kSensingOutputModeDense;
    }
    if (mode == kSensingOutputModeDense || mode == kSensingOutputModeCompactMask) {
        return mode;
    }
    throw std::runtime_error(
        "Invalid sensing.output_mode='" + mode +
        "'. Supported values: dense, compact_mask.");
}

inline bool sensing_output_mode_is_compact_mask(const Config& cfg) {
    return cfg.sensing.output_mode == kSensingOutputModeCompactMask;
}

inline const char* sensing_on_wire_format_to_string(SensingOnWireFormat format) {
    switch (format) {
    case SensingOnWireFormat::ComplexFloat16:
        return kSensingOnWireFormatCf16;
    case SensingOnWireFormat::ComplexFloat32:
    default:
        return kSensingOnWireFormatCf32;
    }
}

inline std::string normalize_sensing_on_wire_format_string(std::string format) {
    std::transform(format.begin(), format.end(), format.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (format.empty()) {
        return kSensingOnWireFormatCf32;
    }
    if (format == kSensingOnWireFormatCf32 || format == kSensingOnWireFormatCf16) {
        return format;
    }
    throw std::runtime_error(
        "Invalid sensing_on_wire_format='" + format +
        "'. Supported values: cf32, cf16.");
}

inline SensingOnWireFormat parse_sensing_on_wire_format_string(const std::string& raw_format) {
    const std::string format = normalize_sensing_on_wire_format_string(raw_format);
    if (format == kSensingOnWireFormatCf16) {
        return SensingOnWireFormat::ComplexFloat16;
    }
    return SensingOnWireFormat::ComplexFloat32;
}

inline std::string normalize_cuda_ldpc_decoder_backend_string(std::string backend) {
    std::transform(backend.begin(), backend.end(), backend.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (backend.empty()) {
        return kCudaLdpcDecoderBackendGpu;
    }
    if (backend == kCudaLdpcDecoderBackendGpu || backend == kCudaLdpcDecoderBackendCpu) {
        return backend;
    }
    throw std::runtime_error(
        "Invalid cuda_ldpc_decoder_backend='" + backend +
        "'. Supported values: gpu, cpu.");
}

inline bool cuda_ldpc_decoder_backend_is_cpu(const Config& cfg) {
    return cfg.cuda.cuda_ldpc_decoder_backend == kCudaLdpcDecoderBackendCpu;
}

inline bool cuda_ldpc_decoder_backend_is_gpu(const Config& cfg) {
    return cfg.cuda.cuda_ldpc_decoder_backend == kCudaLdpcDecoderBackendGpu;
}

inline size_t sensing_on_wire_complex_bytes(SensingOnWireFormat format) {
    switch (format) {
    case SensingOnWireFormat::ComplexFloat16:
        return sizeof(uint16_t) * 2u;
    case SensingOnWireFormat::ComplexFloat32:
    default:
        return sizeof(std::complex<float>);
    }
}

inline uint16_t float32_to_half_ieee_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t exponent = (bits >> 23) & 0xFFu;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent == 0xFFu) {
        if (mantissa != 0) {
            return static_cast<uint16_t>(sign | 0x7E00u);
        }
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    if (exponent == 0) {
        return static_cast<uint16_t>(sign);
    }

    int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
        uint32_t rounded = mantissa >> shift;
        const uint32_t round_bit = 1u << (shift - 1);
        if ((mantissa & round_bit) &&
            (((mantissa & (round_bit - 1u)) != 0u) || (rounded & 0x1u))) {
            ++rounded;
        }
        return static_cast<uint16_t>(sign | rounded);
    }

    mantissa += 0x1000u;
    if (mantissa & 0x800000u) {
        mantissa = 0;
        ++half_exponent;
        if (half_exponent >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
    }

    return static_cast<uint16_t>(
        sign |
        (static_cast<uint32_t>(half_exponent) << 10) |
        (mantissa >> 13));
}

inline size_t raw_fft_bin_to_shifted_index(size_t subcarrier_index, size_t fft_size) {
    const size_t half_fft = fft_size / 2;
    return (subcarrier_index < half_fft) ? (subcarrier_index + half_fft) : (subcarrier_index - half_fft);
}

inline uint32_t sensing_mask_hash_from_layout(
    size_t num_symbols,
    size_t fft_size,
    const std::vector<int>& selected_symbols,
    const std::vector<size_t>& selected_symbol_offsets,
    const std::vector<int>& flat_subcarrier_indices)
{
    uint32_t hash = 2166136261u; // FNV-1a
    auto mix_u32 = [&hash](uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<uint8_t>((value >> shift) & 0xFFu);
            hash *= 16777619u;
        }
    };

    mix_u32(static_cast<uint32_t>(num_symbols));
    mix_u32(static_cast<uint32_t>(fft_size));
    for (size_t row = 0; row < selected_symbols.size(); ++row) {
        mix_u32(static_cast<uint32_t>(selected_symbols[row]));
        const size_t begin = selected_symbol_offsets[row];
        const size_t end = selected_symbol_offsets[row + 1];
        for (size_t idx = begin; idx < end; ++idx) {
            mix_u32(static_cast<uint32_t>(flat_subcarrier_indices[idx]));
        }
    }
    return hash;
}

inline SensingMaskLayout build_sensing_mask_layout(const Config& cfg) {
    if (cfg.ofdm.num_symbols == 0) {
        throw std::runtime_error("num_symbols=0 is invalid for sensing mask layout.");
    }
    if (cfg.ofdm.fft_size == 0) {
        throw std::runtime_error("fft_size=0 is invalid for sensing mask layout.");
    }

    SensingMaskLayout layout;
    layout.num_symbols = cfg.ofdm.num_symbols;
    layout.fft_size = cfg.ofdm.fft_size;
    layout.symbol_to_selected_rank.assign(cfg.ofdm.num_symbols, -1);

    std::vector<uint8_t> mask(cfg.ofdm.num_symbols * cfg.ofdm.fft_size, 0);
    auto flat_index = [&cfg](size_t symbol_index, size_t subcarrier_index) {
        return symbol_index * cfg.ofdm.fft_size + subcarrier_index;
    };

    for (size_t block_idx = 0; block_idx < cfg.sensing.mask_blocks.size(); ++block_idx) {
        const auto& block = cfg.sensing.mask_blocks[block_idx];
        if (block.symbol_count == 0) {
            throw std::runtime_error(
                "mask_blocks[" + std::to_string(block_idx) +
                "].symbol_count must be greater than 0.");
        }
        if (block.subcarrier_count == 0) {
            throw std::runtime_error(
                "mask_blocks[" + std::to_string(block_idx) +
                "].subcarrier_count must be greater than 0.");
        }
        if (block.symbol_start >= cfg.ofdm.num_symbols ||
            block.symbol_start + block.symbol_count > cfg.ofdm.num_symbols) {
            throw std::runtime_error(
                "mask_blocks[" + std::to_string(block_idx) +
                "] exceeds the configured symbol range.");
        }
        if (block.subcarrier_start >= cfg.ofdm.fft_size ||
            block.subcarrier_start + block.subcarrier_count > cfg.ofdm.fft_size) {
            throw std::runtime_error(
                "mask_blocks[" + std::to_string(block_idx) +
                "] exceeds the configured subcarrier range.");
        }

        for (size_t sym = block.symbol_start; sym < block.symbol_start + block.symbol_count; ++sym) {
            if (is_cfo_training_symbol(cfg, sym)) {
                throw std::runtime_error(
                    "mask_blocks[" + std::to_string(block_idx) +
                    "] selects symbol " + std::to_string(sym) +
                    ", which is the CFO training field. CFO training fields are not valid sensing symbols.");
            }
            for (size_t sc = block.subcarrier_start; sc < block.subcarrier_start + block.subcarrier_count; ++sc) {
                mask[flat_index(sym, sc)] = 1;
            }
        }
    }

    layout.selected_symbol_offsets.push_back(0);
    for (size_t sym = 0; sym < cfg.ofdm.num_symbols; ++sym) {
        size_t row_count = 0;
        for (size_t sc = 0; sc < cfg.ofdm.fft_size; ++sc) {
            if (mask[flat_index(sym, sc)] == 0) {
                continue;
            }
            if (layout.symbol_to_selected_rank[sym] < 0) {
                layout.symbol_to_selected_rank[sym] =
                    static_cast<int>(layout.selected_symbols.size());
                layout.selected_symbols.push_back(static_cast<int>(sym));
            }
            layout.flat_subcarrier_indices.push_back(static_cast<int>(sc));
            ++row_count;
        }
        if (row_count > 0) {
            layout.selected_symbol_offsets.push_back(layout.flat_subcarrier_indices.size());
        }
    }

    layout.total_re_count = layout.flat_subcarrier_indices.size();
    if (layout.total_re_count == 0) {
        throw std::runtime_error(
            "compact_mask mode requires sensing.mask_blocks to select at least one RE.");
    }
    layout.mask_hash = sensing_mask_hash_from_layout(
        layout.num_symbols,
        layout.fft_size,
        layout.selected_symbols,
        layout.selected_symbol_offsets,
        layout.flat_subcarrier_indices);
    return layout;
}

inline CompactSensingMaskAnalysis analyze_compact_sensing_mask(const Config& cfg) {
    CompactSensingMaskAnalysis analysis;
    if (!sensing_output_mode_is_compact_mask(cfg)) {
        return analysis;
    }

    const SensingMaskLayout layout = build_sensing_mask_layout(cfg);
    analysis.selected_symbols = layout.selected_symbols;
    analysis.selected_symbol_count = layout.selected_symbols.size();
    if (layout.empty() || layout.selected_symbols.empty()) {
        analysis.incompatibility_reason = "compact_mask does not select any sensing RE.";
        return analysis;
    }
    if (cfg.ofdm.num_symbols == 0) {
        analysis.incompatibility_reason = "num_symbols=0 is invalid for compact_mask analysis.";
        return analysis;
    }

    const size_t first_begin = layout.selected_symbol_offsets[0];
    const size_t first_end = layout.selected_symbol_offsets[1];
    analysis.common_subcarrier_indices.assign(
        layout.flat_subcarrier_indices.begin() + static_cast<std::ptrdiff_t>(first_begin),
        layout.flat_subcarrier_indices.begin() + static_cast<std::ptrdiff_t>(first_end));
    analysis.common_subcarrier_count = analysis.common_subcarrier_indices.size();
    if (analysis.common_subcarrier_count == 0) {
        analysis.incompatibility_reason = "compact_mask selected symbols must include at least one subcarrier.";
        return analysis;
    }

    for (size_t row = 1; row < layout.selected_symbols.size(); ++row) {
        const size_t begin = layout.selected_symbol_offsets[row];
        const size_t end = layout.selected_symbol_offsets[row + 1];
        const size_t count = end - begin;
        if (count != analysis.common_subcarrier_count) {
            analysis.incompatibility_reason =
                "Selected subcarrier count is not identical across compact_mask symbols.";
            return analysis;
        }
        for (size_t idx = 0; idx < count; ++idx) {
            if (layout.flat_subcarrier_indices[begin + idx] != analysis.common_subcarrier_indices[idx]) {
                analysis.incompatibility_reason =
                    "Selected subcarrier set is not identical across compact_mask symbols.";
                return analysis;
            }
        }
    }

    size_t expected_gap = cfg.ofdm.num_symbols;
    if (analysis.selected_symbol_count > 1) {
        const int first_gap = layout.selected_symbols[1] - layout.selected_symbols[0];
        if (first_gap <= 0) {
            analysis.incompatibility_reason =
                "compact_mask selected symbols must be strictly increasing.";
            return analysis;
        }
        expected_gap = static_cast<size_t>(first_gap);
    }
    if (expected_gap == 0) {
        analysis.incompatibility_reason =
            "compact_mask selected symbols must have a positive wrap-around stride.";
        return analysis;
    }

    for (size_t row = 1; row < layout.selected_symbols.size(); ++row) {
        const size_t gap = static_cast<size_t>(layout.selected_symbols[row] - layout.selected_symbols[row - 1]);
        if (gap != expected_gap) {
            analysis.incompatibility_reason =
                "compact_mask selected symbols are not equally spaced across the frame.";
            return analysis;
        }
    }

    const size_t wrap_gap = static_cast<size_t>(
        static_cast<int>(cfg.ofdm.num_symbols) +
        layout.selected_symbols.front() -
        layout.selected_symbols.back());
    if (wrap_gap != expected_gap) {
        analysis.incompatibility_reason =
            "compact_mask selected symbols are not equally spaced when wrap-around is included.";
        return analysis;
    }

    analysis.regular_subsampling_compatible = true;
    analysis.implicit_symbol_stride = expected_gap;

    if (cfg.sensing.range_fft_size < analysis.common_subcarrier_count) {
        analysis.runtime_restriction_reason =
            "compact_mask regular sampling selects " + std::to_string(analysis.common_subcarrier_count) +
            " subcarriers, which exceeds range_fft_size=" + std::to_string(cfg.sensing.range_fft_size) + '.';
        return analysis;
    }
    if (analysis.selected_symbol_count > cfg.sensing.doppler_fft_size) {
        analysis.runtime_restriction_reason =
            "compact_mask regular sampling selects " + std::to_string(analysis.selected_symbol_count) +
            " symbols, which exceeds doppler_fft_size=" + std::to_string(cfg.sensing.doppler_fft_size) + '.';
        return analysis;
    }

    analysis.local_delay_doppler_supported = true;
    return analysis;
}

inline bool compact_mask_runtime_fft_controls_supported(const Config& cfg) {
    return analyze_compact_sensing_mask(cfg).local_delay_doppler_supported;
}

inline std::string compact_mask_runtime_fft_controls_reason(const Config& cfg) {
    return analyze_compact_sensing_mask(cfg).effective_reason();
}

inline bool backend_sensing_processing_supported(const Config& cfg) {
    if (!cfg.sensing.backend_processing_enabled) {
        return false;
    }
    if (!sensing_output_mode_is_compact_mask(cfg)) {
        return true;
    }
    return analyze_compact_sensing_mask(cfg).local_delay_doppler_supported;
}

inline std::string backend_sensing_processing_reason(const Config& cfg) {
    if (!cfg.sensing.backend_processing_enabled) {
        return "backend sensing processing disabled";
    }
    if (!sensing_output_mode_is_compact_mask(cfg)) {
        return {};
    }
    return analyze_compact_sensing_mask(cfg).effective_reason();
}

inline size_t required_sensing_range_bin_count(const Config& cfg) {
    if (!sensing_output_mode_is_compact_mask(cfg)) {
        return std::max<size_t>(cfg.ofdm.fft_size, 1);
    }
    const CompactSensingMaskAnalysis analysis = analyze_compact_sensing_mask(cfg);
    if (analysis.regular_subsampling_compatible) {
        return std::max<size_t>(analysis.common_subcarrier_count, 1);
    }
    return 0;
}

inline size_t required_sensing_doppler_symbol_count(const Config& cfg) {
    size_t required = std::max<size_t>(cfg.ofdm.sensing_symbol_num, 1);
    if (!sensing_output_mode_is_compact_mask(cfg)) {
        return required;
    }
    const CompactSensingMaskAnalysis analysis = analyze_compact_sensing_mask(cfg);
    if (analysis.regular_subsampling_compatible) {
        required = std::max(required, analysis.selected_symbol_count);
    }
    return required;
}

inline size_t resolved_sensing_view_range_bins(const Config& cfg) {
    const size_t full_range_bins = std::max<size_t>(cfg.sensing.range_fft_size, 1);
    if (cfg.sensing.view_range_bins == 0) {
        return full_range_bins;
    }
    return std::clamp(cfg.sensing.view_range_bins, size_t{1}, full_range_bins);
}

inline size_t resolved_sensing_view_doppler_bins(const Config& cfg) {
    const size_t full_doppler_bins = std::max<size_t>(cfg.sensing.doppler_fft_size, 1);
    const size_t min_doppler_bins = std::max<size_t>(required_sensing_doppler_symbol_count(cfg), 1);
    if (cfg.sensing.view_doppler_bins == 0) {
        return full_doppler_bins;
    }
    return std::clamp(cfg.sensing.view_doppler_bins, min_doppler_bins, full_doppler_bins);
}

inline void normalize_sensing_fft_sizes(Config& cfg, const char* context_name) {
    const size_t required_range = required_sensing_range_bin_count(cfg);
    if (cfg.sensing.range_fft_size == 0) {
        const size_t fallback_range = (required_range > 0) ? required_range : std::max<size_t>(cfg.ofdm.fft_size, 1);
        LOG_G_WARN_M(Sensing) << "range_fft_size is unset or 0. Defaulting to " << fallback_range << '.';
        cfg.sensing.range_fft_size = fallback_range;
    }
    if (required_range > 0 && cfg.sensing.range_fft_size < required_range) {
        LOG_G_WARN_M(Sensing) << "range_fft_size=" << cfg.sensing.range_fft_size
                     << " is smaller than the required sensing subcarrier count=" << required_range
                     << " for " << context_name
                     << ". Expanding range_fft_size to keep delay FFT buffers consistent.";
        cfg.sensing.range_fft_size = required_range;
    }
    if (cfg.sensing.doppler_fft_size == 0) {
        LOG_G_WARN_M(Sensing) << "doppler_fft_size=0 is invalid. Clamping to 1.";
        cfg.sensing.doppler_fft_size = 1;
    }
    if (cfg.ofdm.sensing_symbol_num == 0) {
        LOG_G_WARN_M(Sensing) << "sensing_symbol_num=0 is invalid. Clamping to 1.";
        cfg.ofdm.sensing_symbol_num = 1;
    }

    const size_t required_doppler = required_sensing_doppler_symbol_count(cfg);
    if (cfg.sensing.doppler_fft_size < required_doppler) {
        LOG_G_WARN_M(Sensing) << "doppler_fft_size=" << cfg.sensing.doppler_fft_size
                     << " is smaller than the required sensing symbol count=" << required_doppler
                     << " for " << context_name
                     << ". Expanding doppler_fft_size to keep sensing buffers consistent.";
        cfg.sensing.doppler_fft_size = required_doppler;
    }
}

inline void normalize_sensing_view_bins(Config& cfg, const char* context_name) {
    if (cfg.sensing.view_range_bins != 0 && cfg.sensing.view_range_bins > cfg.sensing.range_fft_size) {
        LOG_G_WARN_M(Sensing) << context_name
                     << " sensing_view_range_bins=" << cfg.sensing.view_range_bins
                     << " exceeds range_fft_size=" << cfg.sensing.range_fft_size
                     << ". Clamping backend view width to the configured range FFT size.";
        cfg.sensing.view_range_bins = cfg.sensing.range_fft_size;
    }

    if (cfg.sensing.view_doppler_bins != 0) {
        const size_t min_doppler_bins = std::max<size_t>(required_sensing_doppler_symbol_count(cfg), 1);
        if (cfg.sensing.view_doppler_bins < min_doppler_bins) {
            LOG_G_WARN_M(Sensing) << context_name
                         << " sensing_view_doppler_bins=" << cfg.sensing.view_doppler_bins
                         << " is smaller than the required slow-time symbol count="
                         << min_doppler_bins
                         << ". Clamping backend view height to preserve the full sensing aperture.";
            cfg.sensing.view_doppler_bins = min_doppler_bins;
        }
        if (cfg.sensing.view_doppler_bins > cfg.sensing.doppler_fft_size) {
            LOG_G_WARN_M(Sensing) << context_name
                         << " sensing_view_doppler_bins=" << cfg.sensing.view_doppler_bins
                         << " exceeds doppler_fft_size=" << cfg.sensing.doppler_fft_size
                         << ". Clamping backend view height to the configured Doppler FFT size.";
            cfg.sensing.view_doppler_bins = cfg.sensing.doppler_fft_size;
        }
    }
}

inline DataResourceGridLayout build_data_resource_grid_layout(
    const Config& cfg,
    bool log_warnings = false)
{
    if (cfg.ofdm.num_symbols == 0) {
        throw std::runtime_error("num_symbols=0 is invalid for data-resource layout.");
    }
    if (cfg.ofdm.sync_pos >= cfg.ofdm.num_symbols) {
        throw std::runtime_error(
            "sync_pos=" + std::to_string(cfg.ofdm.sync_pos) +
            " is out of range for num_symbols=" + std::to_string(cfg.ofdm.num_symbols) + '.');
    }
    if (sec_sync_symbol_enabled(cfg)) {
        if (cfg.ofdm.num_symbols < 2) {
            throw std::runtime_error("enable_sec_sync_symbol requires num_symbols >= 2.");
        }
        if (cfg.ofdm.sync_pos == 0) {
            throw std::runtime_error(
                "enable_sec_sync_symbol requires sync_pos >= 1 so sync_pos-1 can hold the second sync symbol.");
        }
    }
    if (cfo_training_sequence_enabled(cfg)) {
        if (cfg.ofdm.sync_pos + 1 >= cfg.ofdm.num_symbols) {
            throw std::runtime_error(
                "enable_cfo_training_sequence requires sync_pos+1 to be inside the frame.");
        }
        std::string cfo_period_error;
        if (!validate_cfo_training_period(cfg, &cfo_period_error)) {
            throw std::runtime_error(cfo_period_error);
        }
        if (has_reserved_sync_symbol_overlap(cfg)) {
            throw std::runtime_error(
                "enable_cfo_training_sequence overlaps another reserved sync symbol.");
        }
    }

    DataResourceGridLayout layout;
    layout.num_symbols = cfg.ofdm.num_symbols;
    layout.fft_size = cfg.ofdm.fft_size;
    layout.sync_pos = cfg.ofdm.sync_pos;
    const DuplexFrameLayout duplex_layout = build_duplex_frame_layout(cfg);
    layout.midframe_pilot_symbol_mask.assign(cfg.ofdm.num_symbols, 0);
    layout.midframe_pilot_symbol_to_rank.assign(cfg.ofdm.num_symbols, -1);
    for (auto sym : cfg.ofdm.midframe_pilot_symbols) {
        if (sym >= cfg.ofdm.num_symbols) {
            if (log_warnings) {
                LOG_G_WARN_M(Config) << "Ignoring midframe_pilot_symbols entry " << sym
                             << " outside num_symbols=" << cfg.ofdm.num_symbols << '.';
            }
            continue;
        }
        if (!duplex_layout.is_downlink(sym)) {
            if (log_warnings) {
                LOG_G_WARN_M(Config) << "Ignoring midframe_pilot_symbols entry " << sym
                             << " because it falls inside a TDD uplink/guard symbol.";
            }
            continue;
        }
        if (is_reserved_sync_symbol(cfg, sym)) {
            if (log_warnings) {
                LOG_G_WARN_M(Config) << "Ignoring midframe_pilot_symbols entry " << sym
                             << " because it overlaps a reserved sync symbol.";
            }
            continue;
        }
        if (layout.midframe_pilot_symbol_mask[sym] != 0) {
            continue;
        }
        layout.midframe_pilot_symbol_to_rank[sym] =
            static_cast<int>(layout.midframe_pilot_symbols.size());
        layout.midframe_pilot_symbols.push_back(static_cast<int>(sym));
        layout.midframe_pilot_symbol_mask[sym] = 1;
    }
    layout.midframe_pilot_symbol_count = layout.midframe_pilot_symbols.size();
    size_t downlink_symbol_count = 0;
    size_t reserved_downlink_symbol_count = 0;
    for (size_t sym = 0; sym < cfg.ofdm.num_symbols; ++sym) {
        if (!duplex_layout.is_downlink(sym)) {
            continue;
        }
        ++downlink_symbol_count;
        if (is_reserved_sync_symbol(cfg, sym)) {
            ++reserved_downlink_symbol_count;
        }
    }
    if (reserved_downlink_symbol_count + layout.midframe_pilot_symbol_count >
        downlink_symbol_count) {
        throw std::runtime_error(
            "Reserved sync/training symbols plus midframe_pilot_symbols exceed downlink symbols.");
    }

    layout.pilot_mask.assign(cfg.ofdm.fft_size, 0);
    for (auto pos : cfg.ofdm.pilot_positions) {
        if (pos < cfg.ofdm.fft_size) {
            layout.pilot_mask[pos] = 1;
        }
    }

    layout.subcarrier_to_non_pilot_index.assign(cfg.ofdm.fft_size, -1);
    layout.non_pilot_subcarrier_indices.reserve(cfg.ofdm.fft_size);
    for (size_t k = 0; k < cfg.ofdm.fft_size; ++k) {
        if (layout.pilot_mask[k] != 0) {
            continue;
        }
        layout.subcarrier_to_non_pilot_index[k] =
            static_cast<int>(layout.non_pilot_subcarrier_indices.size());
        layout.non_pilot_subcarrier_indices.push_back(static_cast<int>(k));
    }
    layout.num_non_pilot_subcarriers = layout.non_pilot_subcarrier_indices.size();

    layout.data_symbol_to_actual_symbol.reserve(downlink_symbol_count);
    layout.actual_symbol_to_data_symbol.assign(cfg.ofdm.num_symbols, -1);
    for (size_t sym = 0; sym < cfg.ofdm.num_symbols; ++sym) {
        if (!duplex_layout.is_downlink(sym) ||
            is_reserved_sync_symbol(cfg, sym) ||
            layout.midframe_pilot_symbol_mask[sym] != 0) {
            continue;
        }
        layout.actual_symbol_to_data_symbol[sym] =
            static_cast<int>(layout.data_symbol_to_actual_symbol.size());
        layout.data_symbol_to_actual_symbol.push_back(static_cast<int>(sym));
    }
    layout.data_symbol_count = layout.data_symbol_to_actual_symbol.size();
    layout.non_pilot_re_count = layout.data_symbol_count * layout.num_non_pilot_subcarriers;

    layout.payload_mask.assign(layout.non_pilot_re_count, cfg.resource_preview.data_resource_blocks_configured ? 0 : 1);
    layout.sensing_pilot_mask.assign(layout.non_pilot_re_count, 0);
    layout.payload_rank.assign(layout.non_pilot_re_count, -1);
    layout.non_pilot_offsets.resize(layout.data_symbol_count + 1, 0);
    layout.payload_offsets.resize(layout.data_symbol_count + 1, 0);
    for (size_t data_sym = 0; data_sym <= layout.data_symbol_count; ++data_sym) {
        layout.non_pilot_offsets[data_sym] = data_sym * layout.num_non_pilot_subcarriers;
    }

    size_t stripped_reserved_symbol_re = 0;
    size_t stripped_non_downlink_symbol_re = 0;
    size_t stripped_pilot_re = 0;
    size_t payload_sensing_pilot_overlap_re = 0;
    if (cfg.resource_preview.data_resource_blocks_configured) {
        for (size_t block_idx = 0; block_idx < cfg.resource_preview.data_resource_blocks.size(); ++block_idx) {
            const auto& block = cfg.resource_preview.data_resource_blocks[block_idx];
            if (block.symbol_count == 0) {
                throw std::runtime_error(
                    "data_resource_blocks[" + std::to_string(block_idx) +
                    "].symbol_count must be greater than 0.");
            }
            if (block.subcarrier_count == 0) {
                throw std::runtime_error(
                    "data_resource_blocks[" + std::to_string(block_idx) +
                    "].subcarrier_count must be greater than 0.");
            }
            if (block.symbol_start >= cfg.ofdm.num_symbols ||
                block.symbol_start + block.symbol_count > cfg.ofdm.num_symbols) {
                throw std::runtime_error(
                    "data_resource_blocks[" + std::to_string(block_idx) +
                    "] exceeds the configured symbol range.");
            }
            if (block.subcarrier_start >= cfg.ofdm.fft_size ||
                block.subcarrier_start + block.subcarrier_count > cfg.ofdm.fft_size) {
                throw std::runtime_error(
                    "data_resource_blocks[" + std::to_string(block_idx) +
                    "] exceeds the configured subcarrier range.");
            }

            for (size_t sym = block.symbol_start; sym < block.symbol_start + block.symbol_count; ++sym) {
                if (!duplex_layout.is_downlink(sym)) {
                    stripped_non_downlink_symbol_re += block.subcarrier_count;
                    continue;
                }
                // Reserved sync and mid-frame pilot symbols keep their dedicated
                // content, so resource blocks never claim RE from those symbols.
                if (is_reserved_sync_symbol(cfg, sym) ||
                    layout.midframe_pilot_symbol_mask[sym] != 0) {
                    if (block.kind == DataResourceBlockKind::SensingPilot &&
                        is_cfo_training_symbol(cfg, sym)) {
                        throw std::runtime_error(
                            "data_resource_blocks[" + std::to_string(block_idx) +
                            "] selects symbol " + std::to_string(sym) +
                            " as sensing_pilot, but that symbol is the CFO training field. "
                            "CFO training fields are not valid sensing symbols.");
                    }
                    if (block.kind == DataResourceBlockKind::Payload) {
                        stripped_reserved_symbol_re += block.subcarrier_count;
                    }
                    continue;
                }
                const int data_sym = layout.actual_symbol_to_data_symbol[sym];
                if (data_sym < 0) {
                    throw std::runtime_error("Invalid internal data-symbol mapping while building layout.");
                }
                const size_t base = layout.non_pilot_offsets[static_cast<size_t>(data_sym)];
                for (size_t sc = block.subcarrier_start;
                     sc < block.subcarrier_start + block.subcarrier_count;
                     ++sc) {
                    // Pilot RE always keep the known pilot sequence, regardless of block kind.
                    if (layout.pilot_mask[sc] != 0) {
                        ++stripped_pilot_re;
                        continue;
                    }
                    const int non_pilot_idx = layout.subcarrier_to_non_pilot_index[sc];
                    if (non_pilot_idx < 0) {
                        continue;
                    }
                    const size_t flat_idx = base + static_cast<size_t>(non_pilot_idx);
                    if (block.kind == DataResourceBlockKind::SensingPilot) {
                        layout.sensing_pilot_mask[flat_idx] = 1;
                    } else {
                        layout.payload_mask[flat_idx] = 1;
                    }
                }
            }
        }
    }

    for (size_t flat_idx = 0; flat_idx < layout.non_pilot_re_count; ++flat_idx) {
        if (layout.sensing_pilot_mask[flat_idx] == 0) {
            continue;
        }
        if (layout.payload_mask[flat_idx] != 0) {
            ++payload_sensing_pilot_overlap_re;
            layout.payload_mask[flat_idx] = 0;
        }
        ++layout.sensing_pilot_re_count;
    }

    size_t payload_rank = 0;
    for (size_t data_sym = 0; data_sym < layout.data_symbol_count; ++data_sym) {
        const size_t base = layout.non_pilot_offsets[data_sym];
        for (size_t sc_idx = 0; sc_idx < layout.num_non_pilot_subcarriers; ++sc_idx) {
            const size_t flat_idx = base + sc_idx;
            if (layout.payload_mask[flat_idx] == 0) {
                continue;
            }
            layout.payload_rank[flat_idx] = static_cast<int>(payload_rank++);
        }
        layout.payload_offsets[data_sym + 1] = payload_rank;
    }
    layout.payload_re_count = payload_rank;

    if (log_warnings && cfg.resource_preview.data_resource_blocks_configured &&
        (stripped_reserved_symbol_re > 0 ||
         stripped_non_downlink_symbol_re > 0 ||
         stripped_pilot_re > 0)) {
        LOG_G_WARN_M(Config) << "data_resource_blocks overlap stripped " << stripped_reserved_symbol_re
                     << " sync/mid-frame-pilot symbol RE, "
                     << stripped_non_downlink_symbol_re
                     << " TDD uplink/guard symbol RE, and " << stripped_pilot_re
                     << " pilot RE. reserved sync symbols, pilot_positions, and midframe_pilot_symbols take precedence.";
    }
    if (log_warnings && payload_sensing_pilot_overlap_re > 0) {
        LOG_G_WARN_M(Sensing) << "data_resource_blocks contain " << payload_sensing_pilot_overlap_re
                     << " RE selected as both payload and sensing_pilot. sensing_pilot takes precedence.";
    }

    return layout;
}

inline void finalize_data_resource_grid_config(Config& cfg, const char* role_name) {
    const DataResourceGridLayout layout = build_data_resource_grid_layout(cfg, true);
    cfg.resource_preview.payload_re_count = layout.payload_re_count;
    cfg.resource_preview.non_pilot_re_count = layout.non_pilot_re_count;
    if (cfg.measurement.measurement_enable && cfg.resource_preview.payload_re_count == 0) {
        throw std::runtime_error(
            std::string(role_name) +
            " measurement_enable requires at least one payload RE. "
            "Omit data_resource_blocks for full payload coverage or configure a non-empty payload grid.");
    }
}

inline void finalize_sensing_mask_config(Config& cfg, const char* role_name) {
    cfg.sensing.output_mode = normalize_sensing_output_mode_string(cfg.sensing.output_mode);
    if (!sensing_output_mode_is_compact_mask(cfg)) {
        validate_dense_sensing_stride(
            cfg,
            cfg.sensing.symbol_stride,
            std::string(role_name) + " dense sensing stride");
        if (cfg.sensing.backend_processing_enabled && !backend_sensing_processing_supported(cfg)) {
            LOG_G_WARN_M(Sensing) << role_name
                         << " requested sensing.backend_processing_enabled=1, but the current sensing mode "
                         << "cannot provide dense backend RD output. Falling back to viewer-local processing.";
            cfg.sensing.backend_processing_enabled = false;
        }
        return;
    }
    const SensingMaskLayout layout = build_sensing_mask_layout(cfg);
    if (layout.empty()) {
        throw std::runtime_error(
            std::string(role_name) +
            " compact_mask mode requires a non-empty sensing.mask_blocks selection.");
    }
    if (cfg.sensing.backend_processing_enabled && !backend_sensing_processing_supported(cfg)) {
        LOG_G_WARN_M(Sensing) << role_name
                     << " requested sensing.backend_processing_enabled=1 in compact_mask mode, but the mask "
                     << "is not regular local-DD compatible: "
                     << backend_sensing_processing_reason(cfg)
                     << ". Falling back to viewer-local processing.";
        cfg.sensing.backend_processing_enabled = false;
    }
}

struct MeasurementPayloadMetadata {
    uint32_t epoch_id = 0;
    uint32_t seq_in_epoch = 0;
    uint32_t packets_per_point = 0;
    uint32_t payload_bytes = 0;
    uint32_t prbs_seed = 0;
    int32_t tx_gain_x10 = 0;
};

struct MeasurementCompareResult {
    uint64_t compared_bits = 0;
    uint64_t bit_errors = 0;
    bool exact_match = false;
};

inline constexpr size_t measurement_payload_header_size() {
    return 32;
}

inline bool measurement_mode_enabled(const Config& cfg) {
    return cfg.measurement.measurement_enable && cfg.measurement.measurement_mode == "internal_prbs";
}

/**
 * @brief True when the radio I/O backend is the channel simulator (no USRP).
 */
inline bool radio_is_sim(const Config& cfg) {
    return cfg.radio.radio_backend == "sim";
}

/**
 * @brief Emit the radio_backend + simulation block to a YAML emitter.
 *
 * Shared by the BS and UE config writers so the two stay in sync.
 */
inline void emit_simulation_config(YAML::Emitter& out, const Config& cfg) {
    out << YAML::Key << "radio_backend" << YAML::Value << cfg.radio.radio_backend;
    out << YAML::Key << "simulation" << YAML::Value << YAML::BeginMap;
    const SimConfig& sim = cfg.simulation;
    out << YAML::Key << "session" << YAML::Value << sim.session;
    out << YAML::Key << "enable_comm_rx" << YAML::Value << sim.enable_comm_rx;
    out << YAML::Key << "enable_sensing_rx" << YAML::Value << sim.enable_sensing_rx;
    out << YAML::Key << "enable_uplink" << YAML::Value << sim.enable_uplink;
    out << YAML::Key << "pacing_enabled" << YAML::Value << sim.pacing_enabled;
    out << YAML::Key << "noise_power_dbfs" << YAML::Value << sim.noise_power_dbfs;
    out << YAML::Key << "snr_control_enable" << YAML::Value << sim.snr_control_enable;
    out << YAML::Key << "target_snr_db" << YAML::Value << sim.target_snr_db;
    out << YAML::Key << "control_port" << YAML::Value << sim.control_port;
    out << YAML::Key << "cfo_hz" << YAML::Value << sim.cfo_hz;
    out << YAML::Key << "sample_rate_offset_ppm" << YAML::Value << sim.sample_rate_offset_ppm;
    out << YAML::Key << "timing_offset_samples" << YAML::Value << sim.timing_offset_samples;
    out << YAML::Key << "array_spacing_m" << YAML::Value << sim.array_spacing_m;
    out << YAML::Key << "array_spacing_lambda" << YAML::Value << sim.array_spacing_lambda;
    out << YAML::Key << "ring_capacity_samples" << YAML::Value << sim.ring_capacity_samples;
    out << YAML::Key << "steering_override_file" << YAML::Value << sim.steering_override_file;
    out << YAML::Key << "comm_multipath_taps" << YAML::Value << YAML::BeginSeq;
    for (const auto& tap : sim.comm_multipath_taps) {
        out << YAML::BeginMap;
        out << YAML::Key << "delay_samples" << YAML::Value << tap.delay_samples;
        out << YAML::Key << "gain_db" << YAML::Value << tap.gain_db;
        out << YAML::Key << "phase_deg" << YAML::Value << tap.phase_deg;
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;
    auto emit_target_seq = [&out](const std::vector<SimTarget>& list) {
        out << YAML::BeginSeq;
        for (const auto& tgt : list) {
            out << YAML::BeginMap;
            out << YAML::Key << "range_m" << YAML::Value << tgt.range_m;
            out << YAML::Key << "velocity_mps" << YAML::Value << tgt.velocity_mps;
            out << YAML::Key << "gain_db" << YAML::Value << tgt.gain_db;
            out << YAML::Key << "angle_deg" << YAML::Value << tgt.angle_deg;
            out << YAML::EndMap;
        }
        out << YAML::EndSeq;
    };
    out << YAML::Key << "targets" << YAML::Value;
    emit_target_seq(sim.targets);
    out << YAML::Key << "bistatic_targets" << YAML::Value;
    emit_target_seq(sim.bistatic_targets);
    out << YAML::EndMap;
}

/**
 * @brief Parse the radio_backend + simulation block from a YAML node.
 *
 * Shared by the BS and UE config loaders. Missing keys keep their
 * struct defaults so existing (hardware) configs are unaffected.
 */
inline void load_simulation_config(const YAML::Node& config, Config& cfg) {
    if (config["radio"] && config["radio"].IsMap() && config["radio"]["radio_backend"]) {
        cfg.radio.radio_backend = config["radio"]["radio_backend"].as<std::string>();
    }
    if (config["simulation"] && config["simulation"].IsMap()) {
        const YAML::Node& sim_node = config["simulation"];
        SimConfig& sim = cfg.simulation;
        if (sim_node["session"]) sim.session = sim_node["session"].as<std::string>();
        if (sim_node["enable_comm_rx"]) sim.enable_comm_rx = sim_node["enable_comm_rx"].as<bool>();
        if (sim_node["enable_sensing_rx"]) sim.enable_sensing_rx = sim_node["enable_sensing_rx"].as<bool>();
        if (sim_node["enable_uplink"]) sim.enable_uplink = sim_node["enable_uplink"].as<bool>();
        if (sim_node["pacing_enabled"]) sim.pacing_enabled = sim_node["pacing_enabled"].as<bool>();
        if (sim_node["noise_power_dbfs"]) sim.noise_power_dbfs = sim_node["noise_power_dbfs"].as<double>();
        if (sim_node["snr_control_enable"]) sim.snr_control_enable = sim_node["snr_control_enable"].as<bool>();
        if (sim_node["target_snr_db"]) sim.target_snr_db = sim_node["target_snr_db"].as<double>();
        if (sim_node["control_port"]) sim.control_port = sim_node["control_port"].as<int>();
        if (sim_node["cfo_hz"]) sim.cfo_hz = sim_node["cfo_hz"].as<double>();
        if (sim_node["sample_rate_offset_ppm"]) sim.sample_rate_offset_ppm = sim_node["sample_rate_offset_ppm"].as<double>();
        if (sim_node["timing_offset_samples"]) sim.timing_offset_samples = sim_node["timing_offset_samples"].as<int>();
        if (sim_node["array_spacing_m"]) sim.array_spacing_m = sim_node["array_spacing_m"].as<double>();
        if (sim_node["array_spacing_lambda"]) sim.array_spacing_lambda = sim_node["array_spacing_lambda"].as<double>();
        if (sim_node["ring_capacity_samples"]) sim.ring_capacity_samples = sim_node["ring_capacity_samples"].as<size_t>();
        if (sim_node["steering_override_file"]) sim.steering_override_file = sim_node["steering_override_file"].as<std::string>();
        auto load_tap_seq = [](const YAML::Node& seq, std::vector<SimMultipathTap>& out_list) {
            out_list.clear();
            for (const auto& node : seq) {
                SimMultipathTap tap;
                if (node["delay_samples"]) tap.delay_samples = node["delay_samples"].as<int>();
                if (node["gain_db"]) tap.gain_db = node["gain_db"].as<double>();
                if (node["phase_deg"]) tap.phase_deg = node["phase_deg"].as<double>();
                out_list.push_back(tap);
            }
        };
        if (sim_node["comm_multipath_taps"] && sim_node["comm_multipath_taps"].IsSequence()) {
            load_tap_seq(sim_node["comm_multipath_taps"], sim.comm_multipath_taps);
        }
        auto load_target_seq = [](const YAML::Node& seq, std::vector<SimTarget>& out_list) {
            out_list.clear();
            for (const auto& node : seq) {
                SimTarget tgt;
                if (node["range_m"]) tgt.range_m = node["range_m"].as<double>();
                if (node["velocity_mps"]) tgt.velocity_mps = node["velocity_mps"].as<double>();
                if (node["gain_db"]) tgt.gain_db = node["gain_db"].as<double>();
                if (node["angle_deg"]) tgt.angle_deg = node["angle_deg"].as<double>();
                out_list.push_back(tgt);
            }
        };
        if (sim_node["targets"] && sim_node["targets"].IsSequence()) {
            load_target_seq(sim_node["targets"], sim.targets);
        }
        if (sim_node["bistatic_targets"] && sim_node["bistatic_targets"].IsSequence()) {
            load_target_seq(sim_node["bistatic_targets"], sim.bistatic_targets);
        }
    }
}

// Parse the duplexing block. Shared by the BS and UE config loaders so both
// agree on the DL/UL/guard symbol partition. Schema:
//   uplink:
//     enabled: true | false
//     duplex_mode: tdd | fdd
//     symbol_start: <size_t>     # TDD: first uplink OFDM symbol
//     symbol_count: <size_t>     # TDD: uplink symbol count (0 => uplink off)
//     guard_symbols: <size_t>    # TDD: guard symbols at the DL->UL boundary
//     center_freq: <double>      # FDD only: uplink carrier (Hz)
//   bs_dl_ul_timing_diff: <int>  # BS startup default (samples)
//   ue_timing_advance: <int>     # UE startup default (samples)
inline void load_duplex_config(const YAML::Node& config, Config& cfg) {
    const YAML::Node ul = config["uplink"] && config["uplink"].IsMap()
        ? config["uplink"]
        : YAML::Node();
    if (ul["enabled"]) {
        cfg.uplink.enabled = ul["enabled"].as<bool>();
    }
    if (ul["duplex_mode"]) {
        const std::string raw = ul["duplex_mode"].as<std::string>();
        DuplexMode mode = cfg.uplink.duplex.mode;
        if (parse_duplex_mode_string(raw, mode)) {
            cfg.uplink.duplex.mode = mode;
        }
    }
    if (ul) {
        if (ul["symbol_start"]) cfg.uplink.duplex.ul_symbol_start = ul["symbol_start"].as<size_t>();
        if (ul["symbol_count"]) cfg.uplink.duplex.ul_symbol_count = ul["symbol_count"].as<size_t>();
        if (ul["guard_symbols"]) cfg.uplink.duplex.ul_guard_symbols = ul["guard_symbols"].as<size_t>();
        if (cfg.uplink.duplex.mode == DuplexMode::FDD && ul["center_freq"]) {
            cfg.uplink.duplex.ul_center_freq = ul["center_freq"].as<double>();
        }
        if (ul["debug_self_channel"]) cfg.uplink.debug_self_channel = ul["debug_self_channel"].as<bool>();
        if (ul["debug_self_scan_spectrum"]) cfg.uplink.debug_self_scan_spectrum = ul["debug_self_scan_spectrum"].as<bool>();
        if (ul["debug_self_scan_slice_samples"]) {
            // 0 keeps the runtime default: one OFDM symbol (fft + CP).
            cfg.uplink.debug_self_scan_slice_samples =
                ul["debug_self_scan_slice_samples"].as<size_t>();
        }
        if (ul["ertm_to_enable"]) cfg.uplink.ertm_to_enable = ul["ertm_to_enable"].as<bool>();
        if (ul["ertm_timing_metric"]) {
            cfg.uplink.ertm_timing_metric = normalize_ertm_timing_metric_string(
                ul["ertm_timing_metric"].as<std::string>());
        }
        if (ul["ertm_debug_output_enabled"]) {
            cfg.uplink.ertm_debug_output_enabled = ul["ertm_debug_output_enabled"].as<bool>();
        }
        if (ul["ertm_dl_rf_delay_samples"]) {
            cfg.uplink.ertm_dl_rf_delay_samples = ul["ertm_dl_rf_delay_samples"].as<double>();
        }
        if (ul["ertm_ul_rf_delay_samples"]) {
            cfg.uplink.ertm_ul_rf_delay_samples = ul["ertm_ul_rf_delay_samples"].as<double>();
        }
        if (ul["ertm_report_interval_frames"]) {
            cfg.uplink.ertm_report_interval_frames = std::max<size_t>(
                1, ul["ertm_report_interval_frames"].as<size_t>());
        }
    }
    if (cfg.uplink.duplex.mode != DuplexMode::FDD) {
        cfg.uplink.duplex.ul_center_freq = 0.0;
    }
    if (ul["bs_dl_ul_timing_diff"]) {
        cfg.uplink.bs_dl_ul_timing_diff = ul["bs_dl_ul_timing_diff"].as<int32_t>();
    }
    if (ul["ue_timing_advance"]) {
        cfg.uplink.ue_timing_advance = ul["ue_timing_advance"].as<int32_t>();
    }
    if (ul["idle_waveform"]) {
        cfg.uplink.idle_waveform = normalize_uplink_idle_waveform_string(
            ul["idle_waveform"].as<std::string>());
    }
    const YAML::Node net = config["network_output"] && config["network_output"].IsMap()
        ? config["network_output"]
        : YAML::Node();
    if (net["self_channel_ip"]) {
        cfg.network_output.uplink_self_channel_ip = net["self_channel_ip"].as<std::string>();
    }
    if (net["self_channel_port"]) {
        cfg.network_output.uplink_self_channel_port = net["self_channel_port"].as<int>();
    }
    if (net["self_pdf_ip"]) {
        cfg.network_output.uplink_self_pdf_ip = net["self_pdf_ip"].as<std::string>();
    }
    if (net["self_pdf_port"]) {
        cfg.network_output.uplink_self_pdf_port = net["self_pdf_port"].as<int>();
    }
    if (net["self_scan_ip"]) {
        cfg.network_output.uplink_self_scan_ip = net["self_scan_ip"].as<std::string>();
    }
    if (net["self_scan_port"]) {
        cfg.network_output.uplink_self_scan_port = net["self_scan_port"].as<int>();
    }
}

inline uint32_t measurement_effective_prbs_seed(
    uint32_t base_seed,
    uint32_t epoch_id,
    uint32_t seq_in_epoch
) {
    uint32_t state = base_seed ^ 0x9E3779B9u;
    state ^= epoch_id * 0x85EBCA6Bu;
    state ^= seq_in_epoch * 0xC2B2AE35u;
    if (state == 0) {
        state = 0xA341316Cu;
    }
    return state;
}

inline uint32_t measurement_prbs_next(uint32_t state) {
    if (state == 0) {
        state = 0xA341316Cu;
    }
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state ? state : 0xA341316Cu;
}

inline void measurement_store_u32_le(uint8_t* dst, uint32_t value) {
    dst[0] = static_cast<uint8_t>(value & 0xFFu);
    dst[1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    dst[2] = static_cast<uint8_t>((value >> 16) & 0xFFu);
    dst[3] = static_cast<uint8_t>((value >> 24) & 0xFFu);
}

inline uint32_t measurement_load_u32_le(const uint8_t* src) {
    return static_cast<uint32_t>(src[0]) |
           (static_cast<uint32_t>(src[1]) << 8) |
           (static_cast<uint32_t>(src[2]) << 16) |
           (static_cast<uint32_t>(src[3]) << 24);
}

inline bool build_measurement_payload(
    std::vector<uint8_t>& payload,
    const MeasurementPayloadMetadata& meta
) {
    constexpr size_t kHeaderSize = measurement_payload_header_size();
    constexpr uint8_t kVersion = 1;
    if (meta.payload_bytes < kHeaderSize) {
        return false;
    }

    payload.resize(meta.payload_bytes, 0);
    payload[0] = 'M';
    payload[1] = 'S';
    payload[2] = 'M';
    payload[3] = 'T';
    payload[4] = kVersion;
    payload[5] = 0;
    payload[6] = 0;
    payload[7] = 0;
    measurement_store_u32_le(payload.data() + 8, meta.epoch_id);
    measurement_store_u32_le(payload.data() + 12, meta.seq_in_epoch);
    measurement_store_u32_le(payload.data() + 16, meta.packets_per_point);
    measurement_store_u32_le(payload.data() + 20, meta.payload_bytes);
    measurement_store_u32_le(payload.data() + 24, meta.prbs_seed);
    measurement_store_u32_le(payload.data() + 28, static_cast<uint32_t>(meta.tx_gain_x10));

    uint32_t prbs_state = measurement_effective_prbs_seed(
        meta.prbs_seed, meta.epoch_id, meta.seq_in_epoch);
    for (size_t i = kHeaderSize; i < payload.size(); ++i) {
        prbs_state = measurement_prbs_next(prbs_state);
        payload[i] = static_cast<uint8_t>(prbs_state & 0xFFu);
    }
    return true;
}

inline bool parse_measurement_payload(
    const uint8_t* payload,
    size_t payload_size,
    MeasurementPayloadMetadata& meta
) {
    constexpr size_t kHeaderSize = measurement_payload_header_size();
    constexpr uint8_t kVersion = 1;
    if (payload == nullptr || payload_size < kHeaderSize) {
        return false;
    }
    if (payload[0] != 'M' || payload[1] != 'S' || payload[2] != 'M' || payload[3] != 'T') {
        return false;
    }
    if (payload[4] != kVersion) {
        return false;
    }

    meta.epoch_id = measurement_load_u32_le(payload + 8);
    meta.seq_in_epoch = measurement_load_u32_le(payload + 12);
    meta.packets_per_point = measurement_load_u32_le(payload + 16);
    meta.payload_bytes = measurement_load_u32_le(payload + 20);
    meta.prbs_seed = measurement_load_u32_le(payload + 24);
    meta.tx_gain_x10 = static_cast<int32_t>(measurement_load_u32_le(payload + 28));
    if (meta.payload_bytes != payload_size || meta.payload_bytes < kHeaderSize) {
        return false;
    }
    return true;
}

inline MeasurementCompareResult compare_measurement_payload(
    const std::vector<uint8_t>& expected,
    const uint8_t* actual,
    size_t payload_size
) {
    MeasurementCompareResult result;
    if (actual == nullptr || expected.size() != payload_size) {
        return result;
    }
    result.compared_bits = static_cast<uint64_t>(payload_size) * 8u;
    uint64_t bit_errors = 0;
    for (size_t i = 0; i < payload_size; ++i) {
        const uint8_t diff = static_cast<uint8_t>(expected[i] ^ actual[i]);
        bit_errors += static_cast<uint64_t>(__builtin_popcount(static_cast<unsigned int>(diff)));
    }
    result.bit_errors = bit_errors;
    result.exact_match = (bit_errors == 0);
    return result;
}

inline float corrected_impulse_snr_linear(float raw_impulse_snr_linear, size_t fft_size, size_t cp_length) {
    if (fft_size == 0 || cp_length == 0) {
        return raw_impulse_snr_linear;
    }
    return raw_impulse_snr_linear * static_cast<float>(cp_length) / static_cast<float>(fft_size);
}

inline double noise_variance_from_snr_linear(double snr_linear) {
    return (snr_linear > 1e-12) ? (1.0 / snr_linear) : 1e12;
}

inline double frame_duration_from_cfg(const Config& cfg) {
    if (cfg.rf_sampling.sample_rate <= 0.0) return 1.0;
    return static_cast<double>(cfg.samples_per_frame()) / cfg.rf_sampling.sample_rate;
}

inline uint32_t reset_hold_frames_from_cfg(const Config& cfg) {
    const double frame_duration_s = frame_duration_from_cfg(cfg);
    if (frame_duration_s <= 0.0) return 1;
    const double hold_frames = std::ceil(cfg.sync_tracking.reset_hold_s / frame_duration_s);
    return static_cast<uint32_t>(std::max(1.0, hold_frames));
}

inline std::string format_freq_hz(double value_hz) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(5) << value_hz;
    return oss.str();
}

inline bool path_exists(const std::string& path) {
    return !path.empty() && ::access(path.c_str(), F_OK) == 0;
}

inline std::optional<size_t> configured_core_to_optional(int configured_core) {
    if (configured_core < 0) {
        return std::nullopt;
    }
    return static_cast<size_t>(configured_core);
}

inline std::optional<size_t> core_from_list_hint(const std::vector<int>& cores, size_t hint) {
    if (cores.empty()) {
        return std::nullopt;
    }
    return configured_core_to_optional(cores[hint % cores.size()]);
}

inline std::optional<size_t> core_from_list_index(const std::vector<int>& cores, size_t index) {
    if (index >= cores.size()) {
        return std::nullopt;
    }
    return configured_core_to_optional(cores[index]);
}

inline std::optional<size_t> downlink_core_from_hint(const Config& cfg, size_t hint) {
    return core_from_list_hint(cfg.cpu_cores.downlink_cpu_cores, hint);
}

inline std::optional<size_t> ue_downlink_core_from_role(const Config& cfg, size_t role) {
    return core_from_list_index(cfg.cpu_cores.downlink_cpu_cores, role);
}

inline std::optional<size_t> sensing_core_from_hint(const Config& cfg, size_t hint) {
    return core_from_list_hint(cfg.cpu_cores.sensing_cpu_cores, hint);
}

inline std::optional<size_t> uplink_core_from_hint(const Config& cfg, size_t hint) {
    return core_from_list_hint(cfg.cpu_cores.uplink_cpu_cores, hint);
}

inline std::optional<size_t> main_thread_core(const Config& cfg) {
    return configured_core_to_optional(cfg.cpu_cores.main_cpu_core);
}

inline bool bind_current_thread_to_core(const std::optional<size_t>& core) {
    if (!core.has_value()) {
        return false;
    }
    std::vector<size_t> cpu_list = {*core};
    uhd::set_thread_affinity(cpu_list);
    return true;
}

inline bool bind_current_thread_from_downlink_hint(const Config& cfg, size_t hint) {
    return bind_current_thread_to_core(downlink_core_from_hint(cfg, hint));
}

inline bool bind_current_thread_from_ue_downlink_role(const Config& cfg, size_t role) {
    return bind_current_thread_to_core(ue_downlink_core_from_role(cfg, role));
}

inline bool bind_current_thread_from_ue_demod_worker(const Config& cfg, size_t worker_idx) {
    if (!cfg.cpu_cores.demod_worker_cpu_cores.empty()) {
        return bind_current_thread_to_core(core_from_list_index(cfg.cpu_cores.demod_worker_cpu_cores, worker_idx));
    }
    // No dedicated worker cores configured: leave the worker unbound. Pinning it
    // to the process_proc core would place two RT threads (spinning collector +
    // saturated DSP worker) on one core, which is strictly worse than letting
    // the scheduler separate them.
    return false;
}

inline bool bind_current_thread_from_ue_ldpc_worker(const Config& cfg, size_t worker_idx) {
    if (!cfg.cpu_cores.ldpc_worker_cpu_cores.empty()) {
        return bind_current_thread_to_core(core_from_list_index(cfg.cpu_cores.ldpc_worker_cpu_cores, worker_idx));
    }
    // Same rationale as the demod workers: unbound beats sharing the
    // bit_processing collector core.
    return false;
}

inline bool bind_current_thread_from_sensing_hint(const Config& cfg, size_t hint) {
    return bind_current_thread_to_core(sensing_core_from_hint(cfg, hint));
}

inline bool bind_current_thread_from_uplink_hint(const Config& cfg, size_t hint) {
    return bind_current_thread_to_core(uplink_core_from_hint(cfg, hint));
}

inline bool bind_current_thread_to_main_core(const Config& cfg) {
    return bind_current_thread_to_core(main_thread_core(cfg));
}

inline void prefault_thread_stack(size_t bytes = 512 * 1024) {
    constexpr size_t page_size = 4096;
    volatile char* stack = static_cast<volatile char*>(alloca(bytes));
    for (size_t i = 0; i < bytes; i += page_size) {
        stack[i] = 0;
    }
    if (bytes > 0) {
        stack[bytes - 1] = 0;
    }
}

/**
 * @brief Convert Config.logging into AsyncLogger runtime filter and apply it.
 *
 * Safe to call after YAML load (and again if config is hot-reloaded later).
 * Unknown module paths are ignored with a warning.
 */
inline void apply_logging_config(const LoggingConfig& logging) {
    async_logger::LoggerRuntimeConfig rt;
    async_logger::LogLevel default_level = async_logger::LogLevel::Warn;
    if (!async_logger::parse_log_level(logging.default_level, default_level)) {
        LOG_G_WARN_M(Config) << "logging.default_level='" << logging.default_level
                             << "' is invalid; using warn.";
        default_level = async_logger::LogLevel::Warn;
    }
    rt.default_level = default_level;
    rt.timestamps = logging.timestamps;
    rt.force_error = logging.force_error;

    for (const auto& kv : logging.modules) {
        const std::string& path = kv.first;
        const std::string& level_text = kv.second;
        std::string lower = level_text;
        for (char& c : lower) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (lower == "inherit" || lower == "default") {
            continue; // leave as inherit
        }
        async_logger::LogLevel level = async_logger::LogLevel::Info;
        if (!async_logger::parse_log_level(level_text, level)) {
            LOG_G_WARN_M(Config) << "logging.modules['" << path << "']='" << level_text
                                 << "' is invalid; ignoring.";
            continue;
        }
        if (async_logger::find_log_module_by_path(path) == async_logger::LogModule::Count) {
            LOG_G_WARN_M(Config) << "logging.modules path '" << path
                                 << "' is not a known log module; ignoring.";
            continue;
        }
        rt.module_levels[path] = level;
    }

    async_logger::AsyncLogger::instance().configure(rt);
}

namespace config_detail {
inline YAML::Node section_node(const YAML::Node& config, const char* key) {
    const YAML::Node section = config[key];
    return (section && section.IsMap()) ? section : YAML::Node();
}

inline bool reject_top_level_config_values(const YAML::Node& config, const char* context_name) {
    if (!config || !config.IsMap()) {
        LOG_G_ERROR_M(Config) << context_name << " root must be a YAML mapping of section maps.";
        return false;
    }
    for (auto it = config.begin(); it != config.end(); ++it) {
        const std::string key = it->first.as<std::string>();
        const YAML::Node value = it->second;
        if (!value.IsMap()) {
            LOG_G_ERROR_M(Config) << context_name << " top-level key '" << key
                          << "' is no longer supported. Move settings under their section.";
            return false;
        }
    }
    return true;
}

template <typename T>
inline void load_value(const YAML::Node& section, const char* key, T& value) {
    if (section && section[key]) {
        value = section[key].as<T>();
    }
}

inline void load_logging_config(const YAML::Node& config, LoggingConfig& logging) {
    const YAML::Node section = section_node(config, "logging");
    if (!section) {
        return;
    }
    load_value(section, "default_level", logging.default_level);
    load_value(section, "timestamps", logging.timestamps);
    load_value(section, "force_error", logging.force_error);
    if (section["modules"] && section["modules"].IsMap()) {
        logging.modules.clear();
        for (auto it = section["modules"].begin(); it != section["modules"].end(); ++it) {
            if (!it->first || !it->second || !it->second.IsScalar()) {
                continue;
            }
            logging.modules[it->first.as<std::string>()] = it->second.as<std::string>();
        }
    }
}

inline bool yaml_node_is_blank_scalar(const YAML::Node& node) {
    if (!node || !node.IsScalar()) return false;
    const std::string raw = node.as<std::string>();
    return raw.find_first_not_of(" \t\r\n") == std::string::npos;
}

inline void load_optional_int_yaml(const YAML::Node& node, int& value) {
    if (!node || yaml_node_is_blank_scalar(node)) return;
    value = node.as<int>();
}

inline void emit_resource_blocks_yaml(
    YAML::Emitter& out,
    const char* key,
    const std::vector<DataResourceBlock>& blocks)
{
    out << YAML::Key << key << YAML::Value << YAML::BeginSeq;
    for (const auto& block : blocks) {
        out << YAML::BeginMap;
        if (std::strcmp(key, "data_resource_blocks") == 0 &&
            block.kind != DataResourceBlockKind::Payload) {
            out << YAML::Key << "kind" << YAML::Value
                << data_resource_block_kind_to_string(block.kind);
        }
        out << YAML::Key << "symbol_start" << YAML::Value << block.symbol_start;
        out << YAML::Key << "symbol_count" << YAML::Value << block.symbol_count;
        out << YAML::Key << "subcarrier_start" << YAML::Value << block.subcarrier_start;
        out << YAML::Key << "subcarrier_count" << YAML::Value << block.subcarrier_count;
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;
}

inline void emit_data_resource_blocks_yaml(YAML::Emitter& out, const Config& cfg) {
    if (!cfg.resource_preview.data_resource_blocks_configured) {
        return;
    }
    emit_resource_blocks_yaml(out, "data_resource_blocks", cfg.resource_preview.data_resource_blocks);
}

inline void emit_sensing_mask_blocks_yaml(YAML::Emitter& out, const Config& cfg) {
    if (cfg.sensing.mask_blocks.empty()) {
        return;
    }
    emit_resource_blocks_yaml(out, "mask_blocks", cfg.sensing.mask_blocks);
}

inline bool load_resource_blocks_from_yaml(
    std::vector<DataResourceBlock>& out_blocks,
    const YAML::Node& config,
    const char* key,
    const char* context_name,
    bool* key_present = nullptr)
{
    const YAML::Node blocks = config[key];
    if (!blocks) {
        if (key_present != nullptr) {
            *key_present = false;
        }
        out_blocks.clear();
        return true;
    }
    if (!blocks.IsSequence()) {
        LOG_G_ERROR_M(Config) << context_name << " key '" << key << "' must be a YAML sequence.";
        return false;
    }
    if (key_present != nullptr) {
        *key_present = true;
    }

    out_blocks.clear();
    out_blocks.reserve(blocks.size());
    for (size_t idx = 0; idx < blocks.size(); ++idx) {
        const YAML::Node& node = blocks[idx];
        if (!node.IsMap()) {
            LOG_G_ERROR_M(Config) << context_name << ' ' << key << "[" << idx
                          << "] must be a YAML map.";
            return false;
        }
        if (!node["symbol_start"] || !node["symbol_count"] ||
            !node["subcarrier_start"] || !node["subcarrier_count"]) {
            LOG_G_ERROR_M(Config) << context_name << ' ' << key << "[" << idx
                          << "] must define symbol_start, symbol_count, subcarrier_start, and subcarrier_count.";
            return false;
        }

        DataResourceBlock block;
        if (std::strcmp(key, "data_resource_blocks") == 0 && node["kind"]) {
            const std::string raw_kind = node["kind"].as<std::string>();
            if (!parse_data_resource_block_kind_string(raw_kind, block.kind)) {
                LOG_G_ERROR_M(Sensing) << context_name << ' ' << key << "[" << idx
                              << "] kind must be '" << kDataResourceBlockKindPayload
                              << "' or '" << kDataResourceBlockKindSensingPilot << "'.";
                return false;
            }
        }
        block.symbol_start = node["symbol_start"].as<size_t>();
        block.symbol_count = node["symbol_count"].as<size_t>();
        block.subcarrier_start = node["subcarrier_start"].as<size_t>();
        block.subcarrier_count = node["subcarrier_count"].as<size_t>();
        out_blocks.push_back(block);
    }
    return true;
}

inline bool load_data_resource_blocks_from_yaml(Config& cfg, const YAML::Node& config, const char* context_name) {
    bool key_present = false;
    if (!load_resource_blocks_from_yaml(
            cfg.resource_preview.data_resource_blocks,
            config,
            "data_resource_blocks",
            context_name,
            &key_present)) {
        return false;
    }
    cfg.resource_preview.data_resource_blocks_configured = key_present;
    return true;
}

inline bool load_sensing_mask_blocks_from_yaml(Config& cfg, const YAML::Node& config, const char* context_name) {
    return load_resource_blocks_from_yaml(
        cfg.sensing.mask_blocks,
        config,
        "mask_blocks",
        context_name);
}
} // namespace config_detail

inline void normalize_equalizer_config(EqualizerConfig& equalizer) {
    if (equalizer.equalizer_mode != kEqualizerModeZf &&
        equalizer.equalizer_mode != kEqualizerModeMmse) {
        LOG_G_WARN_M(Config) << "Unsupported equalizer_mode='" << equalizer.equalizer_mode
                     << "'. Falling back to '" << kEqualizerModeMmse << "'.";
        equalizer.equalizer_mode = kEqualizerModeMmse;
    }
    equalizer.channel_tracking_mode =
        normalize_channel_tracking_mode_string(equalizer.channel_tracking_mode);
    if (equalizer.equalizer_mag_floor <= 0.0 ||
        !std::isfinite(equalizer.equalizer_mag_floor)) {
        LOG_G_WARN_M(Config) << "equalizer_mag_floor is invalid. Falling back to 1e-6.";
        equalizer.equalizer_mag_floor = 1e-6;
    }
    if (equalizer.channel_tracking_min_pilot_snr <= 0.0 ||
        !std::isfinite(equalizer.channel_tracking_min_pilot_snr)) {
        LOG_G_WARN_M(Config) << "channel_tracking_min_pilot_snr is invalid. Falling back to 1e-4.";
        equalizer.channel_tracking_min_pilot_snr = 1e-4;
    }
}

inline Config make_default_bs_config() {
    Config cfg;
    cfg.ofdm.fft_size = 1024;
    cfg.ofdm.cp_length = 128;
    cfg.ofdm.sync_pos = 1;
    cfg.ofdm.enable_sec_sync_symbol = false;
    cfg.ofdm.enable_cfo_training_sequence = false;
    cfg.ofdm.cfo_training_period_samples = 16;
    cfg.rf_sampling.sample_rate = 50e6;
    cfg.rf_sampling.bandwidth = 50e6;
    cfg.downlink.center_freq = 2.4e9;
    cfg.downlink.tx_gain = 30.0;
    cfg.downlink.tx_channel = 0;
    cfg.ofdm.zc_root = 29;
    cfg.ofdm.pilot_positions = {571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451};
    cfg.ofdm.midframe_pilot_symbols = {};
    cfg.ofdm.midframe_pilot_seed = 0x4D46504Cu;
    cfg.ofdm.num_symbols = 100;
    cfg.sensing.output_mode = kSensingOutputModeDense;
    cfg.sensing.on_wire_format = SensingOnWireFormat::ComplexFloat32;
    cfg.sensing.backend_processing_enabled = false;
    cfg.cuda.cuda_mod_pipeline_slots = 2;
    cfg.network_output.mono_sensing_output_enabled = true;
    cfg.network_output.mono_sensing_ip = "";
    cfg.network_output.mono_sensing_port = 8888;
    cfg.network_output.control_port = 9999;
    cfg.network_output.uplink_channel_ip = "0.0.0.0";
    cfg.network_output.uplink_channel_port = 12358;
    cfg.network_output.uplink_pdf_ip = "0.0.0.0";
    cfg.network_output.uplink_pdf_port = 12359;
    cfg.network_output.uplink_constellation_ip = "0.0.0.0";
    cfg.network_output.uplink_constellation_port = 12356;
    cfg.network_output.uplink_self_channel_ip = "0.0.0.0";
    cfg.network_output.uplink_self_channel_port = 12360;
    cfg.network_output.uplink_self_pdf_ip = "0.0.0.0";
    cfg.network_output.uplink_self_pdf_port = 12361;
    cfg.sensing.rx_channel_count = 1;
    cfg.sensing.symbol_stride = 20;
    cfg.downlink_pipeline.tx_circular_buffer_size = 8;
    cfg.downlink_pipeline.data_packet_buffer_size = 256;
    cfg.sensing.paired_frame_queue_size = 16;
    cfg.network_output.udp_input_ip = "0.0.0.0";
    cfg.network_output.udp_input_port = 50000;
    cfg.radio.radio_backend = "uhd";
    cfg.simulation = SimConfig{};
    cfg.measurement.measurement_enable = false;
    cfg.measurement.measurement_mode = "";
    cfg.measurement.measurement_run_id = "";
    cfg.measurement.measurement_output_dir = "";
    cfg.measurement.measurement_payload_bytes = 1024;
    cfg.measurement.measurement_prbs_seed = 0x5A;
    cfg.measurement.measurement_packets_per_point = 1;
    cfg.measurement.measurement_max_packets_per_frame = 1;
    return cfg;
}

inline void normalize_udp_egress_pacer_config(UdpEgressPacerConfig& pacer) {
    if (pacer.target_mbps < 0.0) {
        pacer.target_mbps = 0.0;
    }
    if (pacer.queue_packets == 0) {
        LOG_G_WARN_M(Config) << "udp_egress_pacer_queue_packets=0 is invalid. Clamping to 1.";
        pacer.queue_packets = 1;
    }
    if (pacer.max_delay_ms < 0.0) {
        LOG_G_WARN_M(Config) << "udp_egress_pacer_max_delay_ms<0 is invalid. Clamping to 0 ms.";
        pacer.max_delay_ms = 0.0;
    }
}

inline void normalize_arq_config(NetworkOutputConfig& net) {
    if (net.arq_window_packets < 1) {
        LOG_G_WARN_M(Arq) << "arq_window_packets < 1, clamping to 1.";
        net.arq_window_packets = 1;
    }
    if (net.arq_window_packets > kMaxArqWindowPackets) {
        LOG_G_WARN_M(Arq) << "arq_window_packets exceeds the 16-bit sequence half-space; clamping to "
                     << kMaxArqWindowPackets << ".";
        net.arq_window_packets = kMaxArqWindowPackets;
    }
    if (net.arq_ack_bitmap_bits != 64) {
        LOG_G_WARN_M(Arq) << "arq_ack_bitmap_bits must be 64 for first pass; overriding.";
        net.arq_ack_bitmap_bits = 64;
    }
    if (net.arq_retransmit_timeout_ms < 0) {
        net.arq_retransmit_timeout_ms = 0;
    }
    if (net.arq_max_retries < 0) {
        net.arq_max_retries = 0;
    }
    if (net.arq_feedback_interval_ms < 0) {
        net.arq_feedback_interval_ms = 0;
    }
}

inline void load_bs_downlink_arq_config(const YAML::Node& downlink, NetworkOutputConfig& net) {
    config_detail::load_value(downlink, "arq_enabled", net.arq_enabled);
    config_detail::load_value(downlink, "arq_window_packets", net.arq_window_packets);
    config_detail::load_value(downlink, "arq_retransmit_timeout_ms", net.arq_retransmit_timeout_ms);
    config_detail::load_value(downlink, "arq_max_retries", net.arq_max_retries);
}

inline void load_ue_downlink_arq_config(const YAML::Node& downlink, NetworkOutputConfig& net) {
    config_detail::load_value(downlink, "arq_enabled", net.arq_enabled);
    config_detail::load_value(downlink, "arq_ordered_delivery", net.arq_ordered_delivery);
    config_detail::load_value(downlink, "arq_window_packets", net.arq_window_packets);
    config_detail::load_value(downlink, "arq_feedback_interval_ms", net.arq_feedback_interval_ms);
}

inline void load_bs_uplink_arq_config(const YAML::Node& uplink, NetworkOutputConfig& net) {
    config_detail::load_value(uplink, "arq_enabled", net.arq_enabled);
    config_detail::load_value(uplink, "arq_ordered_delivery", net.arq_ordered_delivery);
    config_detail::load_value(uplink, "arq_window_packets", net.arq_window_packets);
    config_detail::load_value(uplink, "arq_feedback_interval_ms", net.arq_feedback_interval_ms);
}

inline void load_ue_uplink_arq_config(const YAML::Node& uplink, NetworkOutputConfig& net) {
    config_detail::load_value(uplink, "arq_enabled", net.arq_enabled);
    config_detail::load_value(uplink, "arq_window_packets", net.arq_window_packets);
    config_detail::load_value(uplink, "arq_retransmit_timeout_ms", net.arq_retransmit_timeout_ms);
    config_detail::load_value(uplink, "arq_max_retries", net.arq_max_retries);
}

inline bool load_bs_config_from_yaml(Config& cfg, const std::string& filepath) {
    if (!path_exists(filepath)) {
        return false;
    }
    try {
        YAML::Node config = YAML::LoadFile(filepath);
        if (!config_detail::reject_top_level_config_values(config, "BS config")) {
            return false;
        }
        const YAML::Node ofdm = config_detail::section_node(config, "ofdm_frame");
        const YAML::Node cuda = config_detail::section_node(config, "cuda");
        const YAML::Node sensing = config_detail::section_node(config, "sensing");
        const YAML::Node rf = config_detail::section_node(config, "rf_sampling");
        const YAML::Node usrp = config_detail::section_node(config, "usrp_device");
        const YAML::Node clock = config_detail::section_node(config, "clock_time");
        const YAML::Node downlink = config_detail::section_node(config, "downlink");
        const YAML::Node downlink_pipeline = config_detail::section_node(config, "downlink_pipeline");
        const YAML::Node uplink = config_detail::section_node(config, "uplink");
        const YAML::Node measurement = config_detail::section_node(config, "measurement");
        const YAML::Node network = config_detail::section_node(config, "network_output");
        const YAML::Node cpu = config_detail::section_node(config, "cpu_cores");
        const YAML::Node runtime = config_detail::section_node(config, "runtime");
        const YAML::Node resource_preview = config_detail::section_node(config, "resource_preview");
        const YAML::Node ldpc = config_detail::section_node(config, "ldpc");

        config_detail::load_value(ldpc, "fixed_point", cfg.ldpc.fixed_point);
        config_detail::load_value(ldpc, "fixed_point_scale", cfg.ldpc.fixed_point_scale);

        config_detail::load_value(ofdm, "fft_size", cfg.ofdm.fft_size);
        config_detail::load_value(ofdm, "cp_length", cfg.ofdm.cp_length);
        config_detail::load_value(ofdm, "sync_pos", cfg.ofdm.sync_pos);
        config_detail::load_value(ofdm, "enable_sec_sync_symbol", cfg.ofdm.enable_sec_sync_symbol);
        config_detail::load_value(
            ofdm, "enable_cfo_training_sequence", cfg.ofdm.enable_cfo_training_sequence);
        config_detail::load_value(
            ofdm, "cfo_training_period_samples", cfg.ofdm.cfo_training_period_samples);
        config_detail::load_value(ofdm, "zc_root", cfg.ofdm.zc_root);
        config_detail::load_value(ofdm, "num_symbols", cfg.ofdm.num_symbols);
        config_detail::load_value(ofdm, "sensing_symbol_num", cfg.ofdm.sensing_symbol_num);
        if (ofdm["pilot_positions"]) {
            cfg.ofdm.pilot_positions = ofdm["pilot_positions"].as<std::vector<size_t>>();
        }
        if (ofdm["midframe_pilot_symbols"]) {
            cfg.ofdm.midframe_pilot_symbols =
                ofdm["midframe_pilot_symbols"].as<std::vector<size_t>>();
        }
        config_detail::load_value(ofdm, "midframe_pilot_seed", cfg.ofdm.midframe_pilot_seed);

        config_detail::load_value(cuda, "cuda_mod_pipeline_slots", cfg.cuda.cuda_mod_pipeline_slots);

        config_detail::load_value(sensing, "range_fft_size", cfg.sensing.range_fft_size);
        config_detail::load_value(sensing, "doppler_fft_size", cfg.sensing.doppler_fft_size);
        config_detail::load_value(sensing, "view_range_bins", cfg.sensing.view_range_bins);
        config_detail::load_value(
            sensing, "view_doppler_bins", cfg.sensing.view_doppler_bins);
        config_detail::load_value(sensing, "output_mode", cfg.sensing.output_mode);
        if (sensing["on_wire_format"]) {
            cfg.sensing.on_wire_format = parse_sensing_on_wire_format_string(
                sensing["on_wire_format"].as<std::string>());
        }
        config_detail::load_value(
            sensing, "backend_processing_enabled", cfg.sensing.backend_processing_enabled);
        config_detail::load_value(sensing, "rx_device_args", cfg.sensing.rx_device_args);
        config_detail::load_value(sensing, "rx_clock_source", cfg.sensing.rx_clock_source);
        config_detail::load_value(sensing, "rx_time_source", cfg.sensing.rx_time_source);
        config_detail::load_value(sensing, "rx_wire_format", cfg.sensing.rx_wire_format);
        config_detail::load_value(sensing, "symbol_stride", cfg.sensing.symbol_stride);
        config_detail::load_value(sensing, "paired_frame_queue_size", cfg.sensing.paired_frame_queue_size);
        const bool has_sensing_count_key = static_cast<bool>(sensing["rx_channel_count"]);
        if (has_sensing_count_key) {
            cfg.sensing.rx_channel_count = sensing["rx_channel_count"].as<uint32_t>();
        }
        if (sensing["rx_channels"] && sensing["rx_channels"].IsSequence()) {
            cfg.sensing.rx_channels.clear();
            for (const auto& node : sensing["rx_channels"]) {
                SensingRxChannelConfig ch;
                if (node["usrp_channel"]) ch.usrp_channel = node["usrp_channel"].as<uint32_t>();
                if (node["device_args"]) ch.device_args = node["device_args"].as<std::string>();
                if (node["clock_source"]) ch.clock_source = node["clock_source"].as<std::string>();
                if (node["time_source"]) ch.time_source = node["time_source"].as<std::string>();
                if (node["wire_format"]) ch.wire_format = node["wire_format"].as<std::string>();
                if (node["rx_gain"]) ch.rx_gain = node["rx_gain"].as<double>();
                if (node["alignment"]) ch.alignment = node["alignment"].as<int32_t>();
                if (node["rx_antenna"]) ch.rx_antenna = node["rx_antenna"].as<std::string>();
                if (node["enable_system_delay_estimation"]) {
                    ch.enable_system_delay_estimation =
                        node["enable_system_delay_estimation"].as<bool>();
                }
                if (node["enable_sensing_output"]) {
                    ch.enable_sensing_output = node["enable_sensing_output"].as<bool>();
                }
                config_detail::load_optional_int_yaml(node["rx_cpu_core"], ch.rx_cpu_core);
                config_detail::load_optional_int_yaml(
                    node["processing_cpu_core"], ch.processing_cpu_core);
                cfg.sensing.rx_channels.push_back(ch);
            }
            if (!has_sensing_count_key) {
                cfg.sensing.rx_channel_count =
                    static_cast<uint32_t>(cfg.sensing.rx_channels.size());
            }
        }

        config_detail::load_value(rf, "sample_rate", cfg.rf_sampling.sample_rate);
        config_detail::load_value(rf, "bandwidth", cfg.rf_sampling.bandwidth);

        config_detail::load_value(usrp, "device_args", cfg.usrp_device.device_args);
        config_detail::load_value(clock, "clock_source", cfg.clock_time.clock_source);
        config_detail::load_value(clock, "time_source", cfg.clock_time.time_source);

        config_detail::load_value(downlink, "center_freq", cfg.downlink.center_freq);
        config_detail::load_value(downlink, "tx_gain", cfg.downlink.tx_gain);
        config_detail::load_value(downlink, "tx_channel", cfg.downlink.tx_channel);
        config_detail::load_value(downlink, "tx_device_args", cfg.downlink.tx_device_args);
        config_detail::load_value(downlink, "tx_clock_source", cfg.downlink.tx_clock_source);
        config_detail::load_value(downlink, "tx_time_source", cfg.downlink.tx_time_source);
        config_detail::load_value(downlink, "wire_format_tx", cfg.downlink.wire_format_tx);
        config_detail::load_value(
            downlink_pipeline, "tx_circular_buffer_size", cfg.downlink_pipeline.tx_circular_buffer_size);
        config_detail::load_value(
            downlink_pipeline, "data_packet_buffer_size", cfg.downlink_pipeline.data_packet_buffer_size);

        config_detail::load_value(uplink, "rx_gain", cfg.uplink.rx_gain);
        config_detail::load_value(uplink, "rx_channel", cfg.uplink.rx_channel);
        config_detail::load_value(uplink, "rx_wire_format", cfg.uplink.rx_wire_format);
        config_detail::load_value(uplink, "rx_device_args", cfg.uplink.rx_device_args);
        config_detail::load_value(uplink, "rx_clock_source", cfg.uplink.rx_clock_source);
        config_detail::load_value(uplink, "rx_time_source", cfg.uplink.rx_time_source);
        config_detail::load_value(uplink, "rx_agc_enable", cfg.rf_sampling.rx_agc_enable);
        config_detail::load_value(uplink, "rx_agc_low_threshold_db", cfg.rf_sampling.rx_agc_low_threshold_db);
        config_detail::load_value(uplink, "rx_agc_high_threshold_db", cfg.rf_sampling.rx_agc_high_threshold_db);
        config_detail::load_value(uplink, "rx_agc_max_step_db", cfg.rf_sampling.rx_agc_max_step_db);
        config_detail::load_value(uplink, "rx_agc_update_frames", cfg.rf_sampling.rx_agc_update_frames);
        config_detail::load_value(uplink, "equalizer_mode", cfg.uplink.equalizer.equalizer_mode);
        if (uplink["channel_tracking_mode"]) {
            cfg.uplink.equalizer.channel_tracking_mode = normalize_channel_tracking_mode_string(
                uplink["channel_tracking_mode"].as<std::string>());
        }
        config_detail::load_value(uplink, "equalizer_mag_floor", cfg.uplink.equalizer.equalizer_mag_floor);
        config_detail::load_value(
            uplink, "channel_tracking_min_pilot_snr", cfg.uplink.equalizer.channel_tracking_min_pilot_snr);

        load_simulation_config(config, cfg);
        load_duplex_config(config, cfg);

        config_detail::load_value(measurement, "measurement_enable", cfg.measurement.measurement_enable);
        config_detail::load_value(measurement, "measurement_mode", cfg.measurement.measurement_mode);
        config_detail::load_value(measurement, "measurement_run_id", cfg.measurement.measurement_run_id);
        config_detail::load_value(measurement, "measurement_output_dir", cfg.measurement.measurement_output_dir);
        config_detail::load_value(
            measurement, "measurement_payload_bytes", cfg.measurement.measurement_payload_bytes);
        config_detail::load_value(measurement, "measurement_prbs_seed", cfg.measurement.measurement_prbs_seed);
        config_detail::load_value(
            measurement, "measurement_packets_per_point", cfg.measurement.measurement_packets_per_point);
        config_detail::load_value(
            measurement, "measurement_max_packets_per_frame", cfg.measurement.measurement_max_packets_per_frame);

        config_detail::load_value(network, "udp_input_ip", cfg.network_output.udp_input_ip);
        config_detail::load_value(network, "udp_input_port", cfg.network_output.udp_input_port);
        config_detail::load_value(network, "udp_output_ip", cfg.network_output.ul_udp_output_ip);
        config_detail::load_value(network, "udp_output_port", cfg.network_output.ul_udp_output_port);
        config_detail::load_value(
            network, "udp_egress_pacer_enabled", cfg.network_output.udp_egress_pacer.enabled);
        config_detail::load_value(
            network, "udp_egress_pacer_target_mbps", cfg.network_output.udp_egress_pacer.target_mbps);
        config_detail::load_value(
            network, "udp_egress_pacer_queue_packets", cfg.network_output.udp_egress_pacer.queue_packets);
        config_detail::load_value(
            network, "udp_egress_pacer_max_delay_ms", cfg.network_output.udp_egress_pacer.max_delay_ms);
        config_detail::load_value(
            network, "mono_sensing_output_enabled", cfg.network_output.mono_sensing_output_enabled);
        config_detail::load_value(network, "mono_sensing_ip", cfg.network_output.mono_sensing_ip);
        config_detail::load_value(network, "mono_sensing_port", cfg.network_output.mono_sensing_port);
        config_detail::load_value(network, "uplink_channel_ip", cfg.network_output.uplink_channel_ip);
        config_detail::load_value(network, "uplink_channel_port", cfg.network_output.uplink_channel_port);
        config_detail::load_value(network, "uplink_pdf_ip", cfg.network_output.uplink_pdf_ip);
        config_detail::load_value(network, "uplink_pdf_port", cfg.network_output.uplink_pdf_port);
        config_detail::load_value(
            network, "uplink_constellation_ip", cfg.network_output.uplink_constellation_ip);
        config_detail::load_value(
            network, "uplink_constellation_port", cfg.network_output.uplink_constellation_port);
        config_detail::load_value(network, "ertm_debug_ip", cfg.network_output.ertm_debug_ip);
        config_detail::load_value(network, "ertm_debug_port", cfg.network_output.ertm_debug_port);
        config_detail::load_value(network, "control_port", cfg.network_output.control_port);

        load_bs_downlink_arq_config(downlink, cfg.network_output);
        normalize_arq_config(cfg.network_output);
        load_bs_uplink_arq_config(uplink, cfg.uplink_arq);
        normalize_arq_config(cfg.uplink_arq);

        config_detail::load_value(cpu, "downlink_cpu_cores", cfg.cpu_cores.downlink_cpu_cores);
        config_detail::load_value(cpu, "uplink_cpu_cores", cfg.cpu_cores.uplink_cpu_cores);
        config_detail::load_value(cpu, "main_cpu_core", cfg.cpu_cores.main_cpu_core);
        config_detail::load_logging_config(config, cfg.logging);

        if (!config_detail::load_data_resource_blocks_from_yaml(
                cfg, resource_preview, "BS config")) {
            return false;
        }
        if (!config_detail::load_sensing_mask_blocks_from_yaml(
                cfg, resource_preview, "BS config")) {
            return false;
        }
        normalize_udp_egress_pacer_config(cfg.network_output.udp_egress_pacer);
        return true;
    } catch (const YAML::Exception& e) {
        LOG_G_ERROR_M(Config) << "Error parsing YAML config: " << e.what();
        return false;
    }
}

inline void normalize_bs_sensing_channels(Config& cfg) {
    normalize_sensing_fft_sizes(cfg, "BS sensing");
    normalize_sensing_view_bins(cfg, "BS sensing");
    if (cfg.cuda.cuda_mod_pipeline_slots == 0) {
        LOG_G_WARN_M(Cuda) << "cuda_mod_pipeline_slots=0 is invalid. Clamping to 1.";
        cfg.cuda.cuda_mod_pipeline_slots = 1;
    }
    if (cfg.downlink_pipeline.tx_circular_buffer_size == 0) {
        LOG_G_WARN_M(Config) << "tx_circular_buffer_size=0 is invalid. Clamping to 1.";
        cfg.downlink_pipeline.tx_circular_buffer_size = 1;
    }
    if (cfg.downlink_pipeline.data_packet_buffer_size == 0) {
        LOG_G_WARN_M(Sensing) << "data_packet_buffer_size=0 is invalid. Clamping to 1.";
        cfg.downlink_pipeline.data_packet_buffer_size = 1;
    }
    if (cfg.sensing.paired_frame_queue_size == 0) {
        LOG_G_WARN_M(Sensing) << "paired_frame_queue_size=0 is invalid. Clamping to 1.";
        cfg.sensing.paired_frame_queue_size = 1;
    }
    if (cfg.sensing.symbol_stride == 0) {
        LOG_G_WARN_M(Sensing) << "sensing_symbol_stride=0 is invalid. Clamping to 1.";
        cfg.sensing.symbol_stride = 1;
    }
    if (cfg.measurement.measurement_enable) {
        if (cfg.measurement.measurement_mode.empty()) {
            cfg.measurement.measurement_mode = "internal_prbs";
        }
        if (!measurement_mode_enabled(cfg)) {
            LOG_G_WARN_M(Config) << "Unsupported BS measurement_mode='"
                         << cfg.measurement.measurement_mode << "'. Disabling measurement mode.";
            cfg.measurement.measurement_enable = false;
        }
        if (cfg.measurement.measurement_payload_bytes < measurement_payload_header_size()) {
            LOG_G_WARN_M(Config) << "measurement_payload_bytes=" << cfg.measurement.measurement_payload_bytes
                         << " is smaller than the measurement header. Clamping to "
                         << measurement_payload_header_size() << '.';
            cfg.measurement.measurement_payload_bytes = measurement_payload_header_size();
        }
        if (cfg.measurement.measurement_packets_per_point == 0) {
            LOG_G_WARN_M(Config) << "measurement_packets_per_point=0 is invalid. Clamping to 1.";
            cfg.measurement.measurement_packets_per_point = 1;
        }
    }
    std::sort(cfg.ofdm.midframe_pilot_symbols.begin(), cfg.ofdm.midframe_pilot_symbols.end());
    cfg.ofdm.midframe_pilot_symbols.erase(
        std::unique(cfg.ofdm.midframe_pilot_symbols.begin(), cfg.ofdm.midframe_pilot_symbols.end()),
        cfg.ofdm.midframe_pilot_symbols.end());
    normalize_equalizer_config(cfg.uplink.equalizer);
    finalize_data_resource_grid_config(cfg, "BS");
    finalize_sensing_mask_config(cfg, "BS");

    auto make_default_ch0 = [&cfg]() {
        SensingRxChannelConfig ch;
        ch.usrp_channel = 0;
        ch.rx_gain = 30.0;
        ch.alignment = SensingRxChannelConfig{}.alignment;
        ch.clock_source = "";
        ch.time_source = "";
        ch.wire_format = "";
        ch.enable_system_delay_estimation = false;
        ch.enable_sensing_output = cfg.network_output.mono_sensing_output_enabled;
        return ch;
    };
    if (cfg.sensing.rx_channels.empty() && cfg.sensing.rx_channel_count > 0) {
        cfg.sensing.rx_channels.push_back(make_default_ch0());
        for (uint32_t i = 1; i < cfg.sensing.rx_channel_count; ++i) {
            auto ch = make_default_ch0();
            ch.usrp_channel = i;
            cfg.sensing.rx_channels.push_back(ch);
        }
    }

    if (cfg.sensing.rx_channel_count == 0) {
        cfg.sensing.rx_channels.clear();
    } else if (cfg.sensing.rx_channels.size() > cfg.sensing.rx_channel_count) {
        cfg.sensing.rx_channels.resize(cfg.sensing.rx_channel_count);
    } else if (cfg.sensing.rx_channels.size() < cfg.sensing.rx_channel_count) {
        const auto base = cfg.sensing.rx_channels.empty() ? make_default_ch0() : cfg.sensing.rx_channels.front();
        for (size_t i = cfg.sensing.rx_channels.size(); i < cfg.sensing.rx_channel_count; ++i) {
            auto ch = base;
            ch.usrp_channel = static_cast<uint32_t>(base.usrp_channel + i);
            cfg.sensing.rx_channels.push_back(ch);
        }
    }

    if (cfg.network_output.mono_sensing_ip.empty()) {
        cfg.network_output.mono_sensing_ip = "0.0.0.0";
    }
    if (cfg.network_output.uplink_channel_ip.empty()) {
        cfg.network_output.uplink_channel_ip = "0.0.0.0";
    }
    if (cfg.network_output.uplink_pdf_ip.empty()) {
        cfg.network_output.uplink_pdf_ip = "0.0.0.0";
    }
    if (cfg.network_output.uplink_constellation_ip.empty()) {
        cfg.network_output.uplink_constellation_ip = "0.0.0.0";
    }
    if (cfg.network_output.uplink_self_channel_ip.empty()) {
        cfg.network_output.uplink_self_channel_ip = "0.0.0.0";
    }
    if (cfg.network_output.uplink_self_pdf_ip.empty()) {
        cfg.network_output.uplink_self_pdf_ip = "0.0.0.0";
    }
    if (cfg.network_output.ertm_debug_ip.empty()) {
        cfg.network_output.ertm_debug_ip = "0.0.0.0";
    }
    if (cfg.network_output.mono_sensing_port <= 0) {
        LOG_G_WARN_M(Sensing) << "mono_sensing_port=" << cfg.network_output.mono_sensing_port
                     << " is invalid. Falling back to 8888.";
        cfg.network_output.mono_sensing_port = 8888;
    }
    if (cfg.network_output.ertm_debug_port <= 0 || cfg.network_output.ertm_debug_port > 65535) {
        LOG_G_WARN_M(Ertm) << "ertm_debug_port=" << cfg.network_output.ertm_debug_port
                     << " is invalid. Falling back to 12362.";
        cfg.network_output.ertm_debug_port = 12362;
    }

    cfg.sensing.rx_channel_count = static_cast<uint32_t>(cfg.sensing.rx_channels.size());
}

inline Config make_default_ue_config() {
    Config cfg;
    cfg.ofdm.fft_size = 1024;
    cfg.ofdm.cp_length = 128;
    cfg.ofdm.enable_sec_sync_symbol = false;
    cfg.ofdm.enable_cfo_training_sequence = false;
    cfg.ofdm.cfo_training_period_samples = 16;
    cfg.downlink.center_freq = 2.4e9;
    cfg.ofdm.pilot_positions = {571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451};
    cfg.ofdm.midframe_pilot_symbols = {};
    cfg.ofdm.midframe_pilot_seed = 0x4D46504Cu;
    cfg.sensing.range_fft_size = 1024;
    cfg.sensing.doppler_fft_size = 100;
    cfg.ofdm.num_symbols = 100;
    cfg.ofdm.sensing_symbol_num = 100;
    cfg.cuda.cuda_demod_pipeline_slots = 3;
    cfg.cuda.cuda_ldpc_decoder_backend = kCudaLdpcDecoderBackendGpu;
    cfg.cuda.cuda_ldpc_worker_buffers = 3;
    cfg.cuda.cuda_ldpc_cross_frame_flush_frames = 2;
    cfg.cuda.cuda_ldpc_cross_frame_flush_us = 1000.0;
    cfg.ofdm.frame_queue_size = 8;
    cfg.sync_tracking.sync_queue_size = 8;
    cfg.ofdm.sync_pos = 1;
    cfg.sync_tracking.sync_cfo_alias_search_range_hz = 800000.0;
    cfg.sync_tracking.reset_hold_s = 0.5;
    cfg.rf_sampling.sample_rate = 50e6;
    cfg.rf_sampling.bandwidth = 50e6;
    cfg.rf_sampling.rx_gain = 50.0;
    cfg.rf_sampling.rx_agc_enable = false;
    cfg.rf_sampling.rx_agc_low_threshold_db = 11.0;
    cfg.rf_sampling.rx_agc_high_threshold_db = 13.0;
    cfg.rf_sampling.rx_agc_max_step_db = 3.0;
    cfg.rf_sampling.rx_agc_update_frames = 4;
    cfg.ofdm.zc_root = 29;
    cfg.usrp_device.device_args = "";
    cfg.sensing.bi_enabled = true;
    cfg.network_output.bi_sensing_output_enabled = true;
    cfg.network_output.bi_sensing_ip = "";
    cfg.network_output.bi_sensing_port = 8889;
    cfg.network_output.control_port = 10000;
    cfg.network_output.channel_ip = "0.0.0.0";
    cfg.network_output.channel_port = 12348;
    cfg.network_output.pdf_ip = "0.0.0.0";
    cfg.network_output.pdf_port = 12349;
    cfg.network_output.uplink_self_channel_ip = "0.0.0.0";
    cfg.network_output.uplink_self_channel_port = 12350;
    cfg.network_output.uplink_self_pdf_ip = "0.0.0.0";
    cfg.network_output.uplink_self_pdf_port = 12351;
    cfg.network_output.constellation_ip = "0.0.0.0";
    cfg.network_output.constellation_port = 12346;
    cfg.network_output.vofa_debug_ip = "";
    cfg.network_output.vofa_debug_port = 12347;
    cfg.network_output.udp_output_ip = "";
    cfg.sync_tracking.software_sync = true;
    cfg.sync_tracking.predictive_delay = true;
    cfg.sync_tracking.akf_enable = true;
    cfg.sync_tracking.akf_bootstrap_frames = 64;
    cfg.sync_tracking.akf_innovation_window = 64;
    cfg.sync_tracking.akf_max_lag = 4;
    cfg.sync_tracking.akf_adapt_interval = 64;
    cfg.sync_tracking.akf_gate_sigma = 3.0;
    cfg.sync_tracking.akf_tikhonov_lambda = 1e-3;
    cfg.sync_tracking.akf_update_smooth = 0.2;
    cfg.sync_tracking.akf_q_wf_min = 1e-10;
    cfg.sync_tracking.akf_q_wf_max = 1e2;
    cfg.sync_tracking.akf_q_rw_min = 1e-12;
    cfg.sync_tracking.akf_q_rw_max = 1e1;
    cfg.sync_tracking.akf_r_min = 1e-8;
    cfg.sync_tracking.akf_r_max = 1e3;
    cfg.downlink.equalizer.equalizer_mode = kEqualizerModeMmse;
    cfg.downlink.equalizer.channel_tracking_mode = kChannelTrackingModePilotPhase;
    cfg.downlink.equalizer.equalizer_mag_floor = 1e-6;
    cfg.downlink.equalizer.channel_tracking_min_pilot_snr = 1e-4;
    cfg.sensing.output_mode = kSensingOutputModeDense;
    cfg.sensing.on_wire_format = SensingOnWireFormat::ComplexFloat32;
    cfg.sensing.backend_processing_enabled = false;
    cfg.sensing.symbol_stride = 20;
    cfg.radio.radio_backend = "uhd";
    cfg.simulation = SimConfig{};
    cfg.measurement.measurement_enable = false;
    cfg.measurement.measurement_mode = "";
    cfg.measurement.measurement_run_id = "";
    cfg.measurement.measurement_output_dir = "";
    cfg.measurement.measurement_payload_bytes = 1024;
    cfg.measurement.measurement_prbs_seed = 0x5A;
    cfg.measurement.measurement_packets_per_point = 1;
    cfg.measurement.measurement_max_packets_per_frame = 1;
    return cfg;
}

inline bool load_ue_config_from_yaml(Config& cfg, const std::string& filepath) {
    if (!path_exists(filepath)) {
        return false;
    }
    try {
        YAML::Node config = YAML::LoadFile(filepath);
        if (!config_detail::reject_top_level_config_values(config, "UE config")) {
            return false;
        }
        const YAML::Node ofdm = config_detail::section_node(config, "ofdm_frame");
        const YAML::Node cuda = config_detail::section_node(config, "cuda");
        const YAML::Node sensing = config_detail::section_node(config, "sensing");
        const YAML::Node rf = config_detail::section_node(config, "rf_sampling");
        const YAML::Node usrp = config_detail::section_node(config, "usrp_device");
        const YAML::Node downlink = config_detail::section_node(config, "downlink");
        const YAML::Node uplink = config_detail::section_node(config, "uplink");
        const YAML::Node sync = config_detail::section_node(config, "sync_tracking");
        const YAML::Node measurement = config_detail::section_node(config, "measurement");
        const YAML::Node network = config_detail::section_node(config, "network_output");
        const YAML::Node cpu = config_detail::section_node(config, "cpu_cores");
        const YAML::Node runtime = config_detail::section_node(config, "runtime");
        const YAML::Node resource_preview = config_detail::section_node(config, "resource_preview");
        const YAML::Node ldpc = config_detail::section_node(config, "ldpc");

        config_detail::load_value(ldpc, "fixed_point", cfg.ldpc.fixed_point);
        config_detail::load_value(ldpc, "fixed_point_scale", cfg.ldpc.fixed_point_scale);

        config_detail::load_value(ofdm, "fft_size", cfg.ofdm.fft_size);
        config_detail::load_value(ofdm, "cp_length", cfg.ofdm.cp_length);
        config_detail::load_value(ofdm, "sync_pos", cfg.ofdm.sync_pos);
        config_detail::load_value(ofdm, "enable_sec_sync_symbol", cfg.ofdm.enable_sec_sync_symbol);
        config_detail::load_value(
            ofdm, "enable_cfo_training_sequence", cfg.ofdm.enable_cfo_training_sequence);
        config_detail::load_value(
            ofdm, "cfo_training_period_samples", cfg.ofdm.cfo_training_period_samples);
        config_detail::load_value(ofdm, "num_symbols", cfg.ofdm.num_symbols);
        config_detail::load_value(ofdm, "sensing_symbol_num", cfg.ofdm.sensing_symbol_num);
        config_detail::load_value(ofdm, "frame_queue_size", cfg.ofdm.frame_queue_size);
        config_detail::load_value(ofdm, "zc_root", cfg.ofdm.zc_root);
        if (ofdm["pilot_positions"]) {
            cfg.ofdm.pilot_positions = ofdm["pilot_positions"].as<std::vector<size_t>>();
        }
        if (ofdm["midframe_pilot_symbols"]) {
            cfg.ofdm.midframe_pilot_symbols =
                ofdm["midframe_pilot_symbols"].as<std::vector<size_t>>();
        }
        config_detail::load_value(ofdm, "midframe_pilot_seed", cfg.ofdm.midframe_pilot_seed);

        config_detail::load_value(cuda, "cuda_demod_pipeline_slots", cfg.cuda.cuda_demod_pipeline_slots);
        if (cuda["cuda_ldpc_decoder_backend"]) {
            cfg.cuda.cuda_ldpc_decoder_backend = normalize_cuda_ldpc_decoder_backend_string(
                cuda["cuda_ldpc_decoder_backend"].as<std::string>());
        }
        config_detail::load_value(cuda, "cuda_ldpc_worker_buffers", cfg.cuda.cuda_ldpc_worker_buffers);
        config_detail::load_value(
            cuda, "cuda_ldpc_cross_frame_flush_frames", cfg.cuda.cuda_ldpc_cross_frame_flush_frames);
        config_detail::load_value(
            cuda, "cuda_ldpc_cross_frame_flush_us", cfg.cuda.cuda_ldpc_cross_frame_flush_us);

        config_detail::load_value(sensing, "bi_enabled", cfg.sensing.bi_enabled);
        config_detail::load_value(sensing, "range_fft_size", cfg.sensing.range_fft_size);
        config_detail::load_value(sensing, "doppler_fft_size", cfg.sensing.doppler_fft_size);
        config_detail::load_value(sensing, "output_mode", cfg.sensing.output_mode);
        if (sensing["sensing_delay_correction_mode"]) {
            cfg.sensing.delay_correction_mode =
                normalize_sensing_delay_correction_mode_string(
                    sensing["sensing_delay_correction_mode"].as<std::string>());
        }
        if (sensing["on_wire_format"]) {
            cfg.sensing.on_wire_format = parse_sensing_on_wire_format_string(
                sensing["on_wire_format"].as<std::string>());
        }
        config_detail::load_value(
            sensing, "backend_processing_enabled", cfg.sensing.backend_processing_enabled);
        config_detail::load_value(sensing, "view_range_bins", cfg.sensing.view_range_bins);
        config_detail::load_value(
            sensing, "view_doppler_bins", cfg.sensing.view_doppler_bins);
        config_detail::load_value(sensing, "symbol_stride", cfg.sensing.symbol_stride);

        config_detail::load_value(rf, "sample_rate", cfg.rf_sampling.sample_rate);
        config_detail::load_value(rf, "bandwidth", cfg.rf_sampling.bandwidth);
        config_detail::load_value(rf, "rx_gain", cfg.rf_sampling.rx_gain);
        config_detail::load_value(rf, "rx_agc_enable", cfg.rf_sampling.rx_agc_enable);
        config_detail::load_value(rf, "rx_agc_low_threshold_db", cfg.rf_sampling.rx_agc_low_threshold_db);
        config_detail::load_value(rf, "rx_agc_high_threshold_db", cfg.rf_sampling.rx_agc_high_threshold_db);
        config_detail::load_value(rf, "rx_agc_max_step_db", cfg.rf_sampling.rx_agc_max_step_db);
        config_detail::load_value(rf, "rx_agc_update_frames", cfg.rf_sampling.rx_agc_update_frames);

        config_detail::load_value(usrp, "device_args", cfg.usrp_device.device_args);
        config_detail::load_value(usrp, "clock_source", cfg.clock_time.clock_source);

        config_detail::load_value(downlink, "center_freq", cfg.downlink.center_freq);
        config_detail::load_value(downlink, "rx_wire_format", cfg.downlink.rx_wire_format);
        config_detail::load_value(downlink, "rx_channel", cfg.downlink.rx_channel);
        config_detail::load_value(downlink, "equalizer_mode", cfg.downlink.equalizer.equalizer_mode);
        if (downlink["channel_tracking_mode"]) {
            cfg.downlink.equalizer.channel_tracking_mode = normalize_channel_tracking_mode_string(
                downlink["channel_tracking_mode"].as<std::string>());
        }
        config_detail::load_value(downlink, "equalizer_mag_floor", cfg.downlink.equalizer.equalizer_mag_floor);
        config_detail::load_value(
            downlink, "channel_tracking_min_pilot_snr", cfg.downlink.equalizer.channel_tracking_min_pilot_snr);

        config_detail::load_value(uplink, "tx_gain", cfg.uplink.tx_gain);
        config_detail::load_value(uplink, "tx_channel", cfg.uplink.tx_channel);
        config_detail::load_value(uplink, "wire_format_tx", cfg.uplink.wire_format_tx);

        config_detail::load_value(sync, "sync_queue_size", cfg.sync_tracking.sync_queue_size);
        config_detail::load_value(
            sync, "sync_cfo_alias_search_range_hz", cfg.sync_tracking.sync_cfo_alias_search_range_hz);
        config_detail::load_value(sync, "reset_hold_s", cfg.sync_tracking.reset_hold_s);
        config_detail::load_value(sync, "software_sync", cfg.sync_tracking.software_sync);
        config_detail::load_value(sync, "predictive_delay", cfg.sync_tracking.predictive_delay);
        config_detail::load_value(sync, "hardware_sync", cfg.sync_tracking.hardware_sync);
        config_detail::load_value(sync, "hardware_sync_tty", cfg.sync_tracking.hardware_sync_tty);
        config_detail::load_value(sync, "ocxo_pi_kp_fast", cfg.sync_tracking.ocxo_pi_kp_fast);
        config_detail::load_value(sync, "ocxo_pi_ki_fast", cfg.sync_tracking.ocxo_pi_ki_fast);
        config_detail::load_value(sync, "ocxo_pi_kp_slow", cfg.sync_tracking.ocxo_pi_kp_slow);
        config_detail::load_value(sync, "ocxo_pi_ki_slow", cfg.sync_tracking.ocxo_pi_ki_slow);
        config_detail::load_value(
            sync, "ocxo_pi_switch_abs_error_ppm", cfg.sync_tracking.ocxo_pi_switch_abs_error_ppm);
        config_detail::load_value(
            sync, "ocxo_pi_switch_hold_s", cfg.sync_tracking.ocxo_pi_switch_hold_s);
        config_detail::load_value(sync, "ocxo_pi_max_step_fast_ppm", cfg.sync_tracking.ocxo_pi_max_step_fast_ppm);
        config_detail::load_value(sync, "ocxo_pi_max_step_slow_ppm", cfg.sync_tracking.ocxo_pi_max_step_slow_ppm);
        if (sync["ocxo_pi_max_step_ppm"]) {
            const auto max_step = sync["ocxo_pi_max_step_ppm"].as<double>();
            cfg.sync_tracking.ocxo_pi_max_step_fast_ppm = max_step;
            cfg.sync_tracking.ocxo_pi_max_step_slow_ppm = max_step;
        }
        config_detail::load_value(sync, "akf_enable", cfg.sync_tracking.akf_enable);
        config_detail::load_value(sync, "akf_bootstrap_frames", cfg.sync_tracking.akf_bootstrap_frames);
        config_detail::load_value(sync, "akf_innovation_window", cfg.sync_tracking.akf_innovation_window);
        config_detail::load_value(sync, "akf_max_lag", cfg.sync_tracking.akf_max_lag);
        config_detail::load_value(sync, "akf_adapt_interval", cfg.sync_tracking.akf_adapt_interval);
        config_detail::load_value(sync, "akf_gate_sigma", cfg.sync_tracking.akf_gate_sigma);
        config_detail::load_value(sync, "akf_tikhonov_lambda", cfg.sync_tracking.akf_tikhonov_lambda);
        config_detail::load_value(sync, "akf_update_smooth", cfg.sync_tracking.akf_update_smooth);
        config_detail::load_value(sync, "akf_q_wf_min", cfg.sync_tracking.akf_q_wf_min);
        config_detail::load_value(sync, "akf_q_wf_max", cfg.sync_tracking.akf_q_wf_max);
        config_detail::load_value(sync, "akf_q_rw_min", cfg.sync_tracking.akf_q_rw_min);
        config_detail::load_value(sync, "akf_q_rw_max", cfg.sync_tracking.akf_q_rw_max);
        config_detail::load_value(sync, "akf_r_min", cfg.sync_tracking.akf_r_min);
        config_detail::load_value(sync, "akf_r_max", cfg.sync_tracking.akf_r_max);
        config_detail::load_value(sync, "desired_peak_pos", cfg.sync_tracking.desired_peak_pos);

        config_detail::load_value(measurement, "measurement_enable", cfg.measurement.measurement_enable);
        config_detail::load_value(measurement, "measurement_mode", cfg.measurement.measurement_mode);
        config_detail::load_value(measurement, "measurement_run_id", cfg.measurement.measurement_run_id);
        config_detail::load_value(measurement, "measurement_output_dir", cfg.measurement.measurement_output_dir);
        config_detail::load_value(
            measurement, "measurement_payload_bytes", cfg.measurement.measurement_payload_bytes);
        config_detail::load_value(measurement, "measurement_prbs_seed", cfg.measurement.measurement_prbs_seed);
        config_detail::load_value(
            measurement, "measurement_packets_per_point", cfg.measurement.measurement_packets_per_point);
        config_detail::load_value(
            measurement, "measurement_max_packets_per_frame", cfg.measurement.measurement_max_packets_per_frame);

        config_detail::load_value(network, "bi_sensing_output_enabled", cfg.network_output.bi_sensing_output_enabled);
        config_detail::load_value(network, "bi_sensing_ip", cfg.network_output.bi_sensing_ip);
        config_detail::load_value(network, "bi_sensing_port", cfg.network_output.bi_sensing_port);
        config_detail::load_value(network, "control_port", cfg.network_output.control_port);
        config_detail::load_value(network, "channel_ip", cfg.network_output.channel_ip);
        config_detail::load_value(network, "channel_port", cfg.network_output.channel_port);
        config_detail::load_value(network, "pdf_ip", cfg.network_output.pdf_ip);
        config_detail::load_value(network, "pdf_port", cfg.network_output.pdf_port);
        config_detail::load_value(network, "constellation_ip", cfg.network_output.constellation_ip);
        config_detail::load_value(network, "constellation_port", cfg.network_output.constellation_port);
        config_detail::load_value(network, "udp_input_ip", cfg.network_output.ul_udp_input_ip);
        config_detail::load_value(network, "udp_input_port", cfg.network_output.ul_udp_input_port);
        config_detail::load_value(network, "udp_output_ip", cfg.network_output.udp_output_ip);
        config_detail::load_value(network, "udp_output_port", cfg.network_output.udp_output_port);
        config_detail::load_value(
            network, "udp_egress_pacer_enabled", cfg.network_output.udp_egress_pacer.enabled);
        config_detail::load_value(
            network, "udp_egress_pacer_target_mbps", cfg.network_output.udp_egress_pacer.target_mbps);
        config_detail::load_value(
            network, "udp_egress_pacer_queue_packets", cfg.network_output.udp_egress_pacer.queue_packets);
        config_detail::load_value(
            network, "udp_egress_pacer_max_delay_ms", cfg.network_output.udp_egress_pacer.max_delay_ms);
        config_detail::load_value(network, "self_channel_ip", cfg.network_output.uplink_self_channel_ip);
        config_detail::load_value(network, "self_channel_port", cfg.network_output.uplink_self_channel_port);
        config_detail::load_value(network, "self_pdf_ip", cfg.network_output.uplink_self_pdf_ip);
        config_detail::load_value(network, "self_pdf_port", cfg.network_output.uplink_self_pdf_port);
        config_detail::load_value(network, "self_scan_ip", cfg.network_output.uplink_self_scan_ip);
        config_detail::load_value(network, "self_scan_port", cfg.network_output.uplink_self_scan_port);
        config_detail::load_value(network, "ertm_debug_ip", cfg.network_output.ertm_debug_ip);
        config_detail::load_value(network, "ertm_debug_port", cfg.network_output.ertm_debug_port);

        load_ue_downlink_arq_config(downlink, cfg.network_output);
        normalize_arq_config(cfg.network_output);
        load_ue_uplink_arq_config(uplink, cfg.uplink_arq);
        normalize_arq_config(cfg.uplink_arq);

        config_detail::load_value(runtime, "default_out_ip", cfg.network_output.default_out_ip);
        config_detail::load_value(runtime, "vofa_debug_ip", cfg.network_output.vofa_debug_ip);
        config_detail::load_value(runtime, "vofa_debug_port", cfg.network_output.vofa_debug_port);
        config_detail::load_logging_config(config, cfg.logging);

        load_simulation_config(config, cfg);
        load_duplex_config(config, cfg);
        if (uplink["ertm_delay_oversample_factor"]) {
            const int64_t factor = uplink["ertm_delay_oversample_factor"].as<int64_t>();
            cfg.uplink.ertm_delay_oversample_factor = static_cast<size_t>(
                std::min<int64_t>(128, std::max<int64_t>(1, factor)));
        }
        if (!cfg.uplink.ertm_to_enable &&
            cfg.sensing.delay_correction_mode == kSensingDelayCorrectionModeErtmAbsolute) {
            throw std::runtime_error(
                "sensing.sensing_delay_correction_mode=ertm_absolute requires uplink.ertm_to_enable=true");
        }

        config_detail::load_value(cpu, "downlink_cpu_cores", cfg.cpu_cores.downlink_cpu_cores);
        config_detail::load_value(cpu, "demod_worker_cpu_cores", cfg.cpu_cores.demod_worker_cpu_cores);
        config_detail::load_value(cpu, "ldpc_worker_cpu_cores", cfg.cpu_cores.ldpc_worker_cpu_cores);
        config_detail::load_value(cpu, "sensing_cpu_cores", cfg.cpu_cores.sensing_cpu_cores);
        config_detail::load_value(cpu, "uplink_cpu_cores", cfg.cpu_cores.uplink_cpu_cores);
        config_detail::load_value(cpu, "main_cpu_core", cfg.cpu_cores.main_cpu_core);
        if (cfg.cpu_cores.downlink_cpu_cores.size() > 3 &&
            cfg.cpu_cores.demod_worker_cpu_cores.empty()) {
            cfg.cpu_cores.demod_worker_cpu_cores.assign(
                cfg.cpu_cores.downlink_cpu_cores.begin() + 3,
                cfg.cpu_cores.downlink_cpu_cores.end());
            cfg.cpu_cores.downlink_cpu_cores.resize(3);
        }
        if (cfg.cpu_cores.downlink_cpu_cores.size() > 3) {
            throw std::runtime_error(
                "UE downlink_cpu_cores now accepts [rx_proc, process_proc, bit_processing_proc]. "
                "Move OFDM demod workers to cpu_cores.demod_worker_cpu_cores and sensing_process_proc "
                "to cpu_cores.sensing_cpu_cores.");
        }

        if (!config_detail::load_data_resource_blocks_from_yaml(
                cfg, resource_preview, "UE config")) {
            return false;
        }
        if (!config_detail::load_sensing_mask_blocks_from_yaml(
                cfg, resource_preview, "UE config")) {
            return false;
        }
        normalize_udp_egress_pacer_config(cfg.network_output.udp_egress_pacer);
        return true;
    } catch (const YAML::Exception& e) {
        LOG_G_ERROR_M(Config) << "Error parsing YAML config: " << e.what();
        return false;
    }
}

inline void finalize_ue_network_defaults(Config& cfg) {
    normalize_sensing_fft_sizes(cfg, "deBS sensing");
    normalize_sensing_view_bins(cfg, "deBS sensing");
    if (cfg.cuda.cuda_demod_pipeline_slots == 0) {
        LOG_G_WARN_M(Cuda) << "cuda_demod_pipeline_slots=0 is invalid. Clamping to 1.";
        cfg.cuda.cuda_demod_pipeline_slots = 1;
    }
    cfg.cuda.cuda_ldpc_decoder_backend =
        normalize_cuda_ldpc_decoder_backend_string(cfg.cuda.cuda_ldpc_decoder_backend);
    if (cfg.cuda.cuda_ldpc_worker_buffers < 2) {
        LOG_G_WARN_M(Cuda) << "cuda_ldpc_worker_buffers<2 is invalid. Clamping to 2.";
        cfg.cuda.cuda_ldpc_worker_buffers = 2;
    }
    if (cfg.cuda.cuda_ldpc_cross_frame_flush_frames == 0) {
        LOG_G_WARN_M(Cuda) << "cuda_ldpc_cross_frame_flush_frames=0 is invalid. Clamping to 1.";
        cfg.cuda.cuda_ldpc_cross_frame_flush_frames = 1;
    }
    if (cfg.cuda.cuda_ldpc_cross_frame_flush_us < 0.0) {
        LOG_G_WARN_M(Cuda) << "cuda_ldpc_cross_frame_flush_us<0 is invalid. Clamping to 0 us.";
        cfg.cuda.cuda_ldpc_cross_frame_flush_us = 0.0;
    }
    if (cfg.ofdm.frame_queue_size == 0) {
        LOG_G_WARN_M(Config) << "frame_queue_size=0 is invalid. Clamping to 1.";
        cfg.ofdm.frame_queue_size = 1;
    }
    if (cfg.sync_tracking.sync_queue_size == 0) {
        LOG_G_WARN_M(Config) << "sync_queue_size=0 is invalid. Clamping to 1.";
        cfg.sync_tracking.sync_queue_size = 1;
    }
    if (cfg.sync_tracking.sync_cfo_alias_search_range_hz < 0.0 ||
        !std::isfinite(cfg.sync_tracking.sync_cfo_alias_search_range_hz)) {
        LOG_G_WARN_M(Config) << "sync_cfo_alias_search_range_hz is invalid. Clamping to 0 Hz.";
        cfg.sync_tracking.sync_cfo_alias_search_range_hz = 0.0;
    }
    if (cfg.sync_tracking.reset_hold_s <= 0.0) {
        LOG_G_WARN_M(Sensing) << "reset_hold_s<=0 is invalid. Clamping to 0.5 s.";
        cfg.sync_tracking.reset_hold_s = 0.5;
    }
    if (cfg.sensing.symbol_stride == 0) {
        LOG_G_WARN_M(Sensing) << "sensing_symbol_stride=0 is invalid. Clamping to 1.";
        cfg.sensing.symbol_stride = 1;
    }
    if (cfg.rf_sampling.rx_agc_update_frames == 0) {
        LOG_G_WARN_M(Config) << "rx_agc_update_frames=0 is invalid. Clamping to 1.";
        cfg.rf_sampling.rx_agc_update_frames = 1;
    }
    if (cfg.rf_sampling.rx_agc_max_step_db <= 0.0) {
        LOG_G_WARN_M(Config) << "rx_agc_max_step_db<=0 is invalid. Clamping to 1 dB.";
        cfg.rf_sampling.rx_agc_max_step_db = 1.0;
    }
    if (cfg.rf_sampling.rx_agc_low_threshold_db >= cfg.rf_sampling.rx_agc_high_threshold_db) {
        LOG_G_WARN_M(Config) << "rx_agc_low_threshold_db>=rx_agc_high_threshold_db is invalid. Resetting to 11/13 dB.";
        cfg.rf_sampling.rx_agc_low_threshold_db = 11.0;
        cfg.rf_sampling.rx_agc_high_threshold_db = 13.0;
    }
    normalize_equalizer_config(cfg.downlink.equalizer);
    {
        std::sort(cfg.ofdm.midframe_pilot_symbols.begin(), cfg.ofdm.midframe_pilot_symbols.end());
        cfg.ofdm.midframe_pilot_symbols.erase(
            std::unique(cfg.ofdm.midframe_pilot_symbols.begin(), cfg.ofdm.midframe_pilot_symbols.end()),
            cfg.ofdm.midframe_pilot_symbols.end());
    }
    if (cfg.measurement.measurement_enable) {
        if (cfg.measurement.measurement_mode.empty()) {
            cfg.measurement.measurement_mode = "internal_prbs";
        }
        if (!measurement_mode_enabled(cfg)) {
            LOG_G_WARN_M(Config) << "Unsupported UE measurement_mode='"
                         << cfg.measurement.measurement_mode << "'. Disabling measurement mode.";
            cfg.measurement.measurement_enable = false;
        }
        if (cfg.measurement.measurement_payload_bytes < measurement_payload_header_size()) {
            LOG_G_WARN_M(Config) << "measurement_payload_bytes=" << cfg.measurement.measurement_payload_bytes
                         << " is smaller than the measurement header. Clamping to "
                         << measurement_payload_header_size() << '.';
            cfg.measurement.measurement_payload_bytes = measurement_payload_header_size();
        }
        if (cfg.measurement.measurement_packets_per_point == 0) {
            LOG_G_WARN_M(Config) << "measurement_packets_per_point=0 is invalid. Clamping to 1.";
            cfg.measurement.measurement_packets_per_point = 1;
        }
    }
    finalize_data_resource_grid_config(cfg, "UE");
    finalize_sensing_mask_config(cfg, "UE");
    if (cfg.network_output.bi_sensing_ip.empty()) cfg.network_output.bi_sensing_ip = "0.0.0.0";
    if (cfg.network_output.channel_ip.empty()) cfg.network_output.channel_ip = "0.0.0.0";
    if (cfg.network_output.pdf_ip.empty()) cfg.network_output.pdf_ip = "0.0.0.0";
    if (cfg.network_output.constellation_ip.empty()) cfg.network_output.constellation_ip = "0.0.0.0";
    if (cfg.network_output.uplink_self_channel_ip.empty()) cfg.network_output.uplink_self_channel_ip = "0.0.0.0";
    if (cfg.network_output.uplink_self_pdf_ip.empty()) cfg.network_output.uplink_self_pdf_ip = "0.0.0.0";
    if (cfg.network_output.ertm_debug_ip.empty()) cfg.network_output.ertm_debug_ip = "0.0.0.0";
    if (cfg.network_output.ertm_debug_port <= 0 || cfg.network_output.ertm_debug_port > 65535) {
        LOG_G_WARN_M(Ertm) << "ertm_debug_port=" << cfg.network_output.ertm_debug_port
                     << " is invalid. Falling back to 12362.";
        cfg.network_output.ertm_debug_port = 12362;
    }
    if (cfg.network_output.vofa_debug_ip.empty()) cfg.network_output.vofa_debug_ip = cfg.network_output.default_out_ip;
    if (cfg.network_output.udp_output_ip.empty()) cfg.network_output.udp_output_ip = cfg.network_output.default_out_ip;
    normalize_udp_egress_pacer_config(cfg.network_output.udp_egress_pacer);
}

inline void log_ue_sync_mode(const Config& cfg) {
    if (cfg.sync_tracking.hardware_sync && cfg.sync_tracking.software_sync) {
        LOG_G_INFO_M(Config) << "Both software_sync and hardware_sync are enabled.";
    } else if (cfg.sync_tracking.hardware_sync) {
        LOG_G_INFO_M(Config) << "Hardware sync enabled.";
    } else if (cfg.sync_tracking.software_sync) {
        LOG_G_INFO_M(Config) << "Software sync enabled.";
    } else {
        LOG_G_WARN_M(Config) << "Both software_sync and hardware_sync are disabled.";
    }
    LOG_G_INFO_M(Config) << "Predictive delay compensation "
                 << (cfg.sync_tracking.predictive_delay ? "enabled." : "disabled.");
}

inline void log_ue_agc_mode(const Config& cfg) {
    if (!cfg.rf_sampling.rx_agc_enable) {
        LOG_G_INFO_M(Agc) << "RX AGC disabled. Using fixed RX gain: " << cfg.rf_sampling.rx_gain << " dB";
        return;
    }
    LOG_G_INFO_M(Agc) << "RX AGC enabled. low_threshold_db=" << cfg.rf_sampling.rx_agc_low_threshold_db
                 << ", high_threshold_db=" << cfg.rf_sampling.rx_agc_high_threshold_db
                 << ", max_step_db=" << cfg.rf_sampling.rx_agc_max_step_db
                 << ", update_frames=" << cfg.rf_sampling.rx_agc_update_frames;
}

/**
 * @brief Synchronization search batch with source USRP timestamp.
 */
struct SyncBatch {
    AlignedVector data;
    int64_t usrp_time_ns = -1;
    uint64_t generation = 0;
};

inline int64_t time_spec_to_ns(const radio::TimeSpec& time_spec) {
    return static_cast<int64_t>(std::llround(time_spec.get_real_secs() * 1e9));
}

inline int64_t time_spec_to_ns(const uhd::time_spec_t& time_spec) {
    return static_cast<int64_t>(std::llround(time_spec.get_real_secs() * 1e9));
}

inline int64_t metadata_time_to_ns(const radio::RxMetadata& md) {
    if (!md.has_time_spec) {
        return -1;
    }
    return time_spec_to_ns(md.time_spec);
}

inline int64_t metadata_time_to_ns(const uhd::rx_metadata_t& md) {
    if (!md.has_time_spec) {
        return -1;
    }
    return time_spec_to_ns(md.time_spec);
}

/**
 * @brief Received Data Frame Structure.
 * 
 * Contains the raw time-domain samples of a received OFDM frame and alignment information.
 */
struct RxFrame {
    AlignedVector frame_data;  // Received symbol collection
    int Alignment;
    int64_t usrp_time_ns = -1; // USRP timestamp of the frame start sample
    int64_t host_enqueue_time_ns = 0; // Host timestamp after full-frame RX, before queue push
    uint64_t generation = 0;   // Sync generation used to drop stale frames after a hard resync
};

/**
 * @brief Per-command timestamp gates for demod control actions.
 *
 * Each control action type keeps an independent "last executed" USRP timestamp.
 * A frame is allowed to issue a control action only if its source timestamp is
 * newer than the last execution timestamp of that same action type.
 */
class DemodControlTimeGates {
public:
    bool allow_reset(int64_t source_time_ns) const {
        return allow_for(_last_reset_exec_time_ns, source_time_ns);
    }

    bool allow_alignment(int64_t source_time_ns) const {
        return allow_for(_last_alignment_exec_time_ns, source_time_ns);
    }

    bool allow_freq_adjust(int64_t source_time_ns) const {
        return allow_for(_last_freq_adjust_exec_time_ns, source_time_ns);
    }

    bool allow_rx_gain_adjust(int64_t source_time_ns) const {
        return allow_for(_last_rx_gain_exec_time_ns, source_time_ns);
    }

    void mark_reset(int64_t exec_time_ns) {
        mark(_last_reset_exec_time_ns, exec_time_ns);
    }

    void mark_alignment(int64_t exec_time_ns) {
        mark(_last_alignment_exec_time_ns, exec_time_ns);
    }

    void mark_freq_adjust(int64_t exec_time_ns) {
        mark(_last_freq_adjust_exec_time_ns, exec_time_ns);
    }

    void mark_rx_gain_adjust(int64_t exec_time_ns) {
        mark(_last_rx_gain_exec_time_ns, exec_time_ns);
    }

    void mark_reset_now(const radio::TimeSpec& now) {
        mark_reset(time_spec_to_ns(now));
    }

    void mark_reset_now(const uhd::time_spec_t& now) {
        mark_reset(time_spec_to_ns(now));
    }

    void mark_alignment_now(const radio::TimeSpec& now) {
        mark_alignment(time_spec_to_ns(now));
    }

    void mark_alignment_now(const uhd::time_spec_t& now) {
        mark_alignment(time_spec_to_ns(now));
    }

    void mark_freq_adjust_now(const radio::TimeSpec& now) {
        mark_freq_adjust(time_spec_to_ns(now));
    }

    void mark_freq_adjust_now(const uhd::time_spec_t& now) {
        mark_freq_adjust(time_spec_to_ns(now));
    }

    void mark_rx_gain_adjust_now(const radio::TimeSpec& now) {
        mark_rx_gain_adjust(time_spec_to_ns(now));
    }

    void mark_rx_gain_adjust_now(const uhd::time_spec_t& now) {
        mark_rx_gain_adjust(time_spec_to_ns(now));
    }

private:
    static bool allow_for(const std::atomic<int64_t>& last_exec_time_ns, int64_t source_time_ns) {
        if (source_time_ns < 0) {
            return true;
        }
        const int64_t last_exec_ns = last_exec_time_ns.load(std::memory_order_acquire);
        return last_exec_ns < 0 || source_time_ns > last_exec_ns;
    }

    static void mark(std::atomic<int64_t>& last_exec_time_ns, int64_t exec_time_ns) {
        last_exec_time_ns.store(exec_time_ns, std::memory_order_release);
    }

    std::atomic<int64_t> _last_reset_exec_time_ns{-1};
    std::atomic<int64_t> _last_alignment_exec_time_ns{-1};
    std::atomic<int64_t> _last_freq_adjust_exec_time_ns{-1};
    std::atomic<int64_t> _last_rx_gain_exec_time_ns{-1};
};

struct RxAgcAdjustment {
    double next_gain_db = 0.0;
    double delta_db = 0.0;
    double observed_peak = 0.0;
    double observed_peak_db = -120.0;
    double filtered_peak_db = -120.0;
    double peak_ratio = 0.0;
    double max_sync_sample_component = 0.0;
    size_t near_full_scale_count = 0;
    size_t hard_full_scale_count = 0;
    bool saturation_detected = false;
};

class HardwareRxAgc {
public:
    explicit HardwareRxAgc(const Config& cfg)
        : _enabled(cfg.rf_sampling.rx_agc_enable),
          _low_threshold_db(cfg.rf_sampling.rx_agc_low_threshold_db),
          _high_threshold_db(cfg.rf_sampling.rx_agc_high_threshold_db),
          _max_step_db(cfg.rf_sampling.rx_agc_max_step_db),
          _update_frames(std::max<size_t>(1, cfg.rf_sampling.rx_agc_update_frames)) {}

    void initialize(double initial_gain_db, double hw_min_gain_db, double hw_max_gain_db) {
        _min_gain_db = std::min(hw_min_gain_db, hw_max_gain_db);
        _max_gain_db = std::max(hw_min_gain_db, hw_max_gain_db);
        _current_gain_db = std::clamp(initial_gain_db, _min_gain_db, _max_gain_db);
        _frames_since_update = 0;
        _filtered_peak_db = -120.0;
        _has_filtered_peak = false;
        _last_error_direction = 0;
        _stable_error_count = 0;
        _saturation_rise_holdoff = 0;
    }

    bool enabled() const {
        return _enabled;
    }

    double current_gain_db() const {
        return _current_gain_db;
    }

    void sync_to_gain(double gain_db) {
        _current_gain_db = std::clamp(gain_db, _min_gain_db, _max_gain_db);
        _frames_since_update = 0;
        _filtered_peak_db = -120.0;
        _has_filtered_peak = false;
        _last_error_direction = 0;
        _stable_error_count = 0;
        _saturation_rise_holdoff = 0;
    }

    template <typename ApplyGainFn>
    bool maybe_apply_from_delay_peak(
        float peak_mag,
        float average_mag,
        const std::complex<float>* sync_symbol_samples,
        size_t sync_symbol_sample_count,
        int64_t source_time_ns,
        const DemodControlTimeGates& time_gates,
        ApplyGainFn&& apply_gain,
        RxAgcAdjustment* applied_adjustment = nullptr
    ) {
        if (!_enabled || !time_gates.allow_rx_gain_adjust(source_time_ns)) {
            return false;
        }
        if (!std::isfinite(peak_mag) || peak_mag <= 0.0f) {
            return false;
        }

        RxAgcAdjustment adjustment;
        analyze_sync_symbol_saturation(sync_symbol_samples, sync_symbol_sample_count, adjustment);

        if (!adjustment.saturation_detected &&
            (!std::isfinite(average_mag) || average_mag <= 0.0f)) {
            return false;
        }

        if (!adjustment.saturation_detected) {
            const double peak_ratio = static_cast<double>(peak_mag) / static_cast<double>(average_mag);
            constexpr double kMinAgcPeakRatio = 8.0;
            if (peak_ratio < kMinAgcPeakRatio) {
                return false;
            }
            adjustment.peak_ratio = peak_ratio;
        }

        if (!observe_delay_peak(static_cast<double>(peak_mag), adjustment)) {
            return false;
        }
        apply_gain(adjustment.next_gain_db);
        if (applied_adjustment != nullptr) {
            *applied_adjustment = adjustment;
        }
        return true;
    }

private:
    bool observe_delay_peak(double peak_mag, RxAgcAdjustment& adjustment) {
        if (!_enabled || !std::isfinite(peak_mag) || peak_mag <= 0.0) {
            return false;
        }
        if (++_frames_since_update < _update_frames) {
            return false;
        }
        _frames_since_update = 0;

        constexpr double kMinPeak = 1e-12;
        const double peak_clamped = std::max(peak_mag, kMinPeak);
        adjustment.observed_peak = peak_clamped;
        adjustment.observed_peak_db = 20.0 * std::log10(peak_clamped);
        if (!_has_filtered_peak) {
            _filtered_peak_db = adjustment.observed_peak_db;
            _has_filtered_peak = true;
        } else {
            constexpr double kPeakDbEmaAlpha = 0.25;
            _filtered_peak_db =
                (1.0 - kPeakDbEmaAlpha) * _filtered_peak_db + kPeakDbEmaAlpha * adjustment.observed_peak_db;
        }
        adjustment.filtered_peak_db = _filtered_peak_db;

        double delta_db = 0.0;
        if (adjustment.saturation_detected) {
            delta_db = -_max_step_db;
            _saturation_rise_holdoff = kSaturationRiseHoldoffUpdates;
        } else if (adjustment.filtered_peak_db < _low_threshold_db) {
            delta_db = _low_threshold_db - adjustment.filtered_peak_db;
        } else if (adjustment.filtered_peak_db > _high_threshold_db) {
            delta_db = _high_threshold_db - adjustment.filtered_peak_db;
        } else {
            _last_error_direction = 0;
            _stable_error_count = 0;
            if (_saturation_rise_holdoff > 0) {
                --_saturation_rise_holdoff;
            }
            return false;
        }

        if (!adjustment.saturation_detected && delta_db > 0.0 && _saturation_rise_holdoff > 0) {
            --_saturation_rise_holdoff;
            return false;
        }

        constexpr size_t kRequiredStableErrorCount = 2;
        const int error_direction = (delta_db > 0.0) ? 1 : -1;
        if (!adjustment.saturation_detected && error_direction != _last_error_direction) {
            _last_error_direction = error_direction;
            _stable_error_count = 1;
            return false;
        }
        if (!adjustment.saturation_detected) {
            ++_stable_error_count;
        } else {
            _last_error_direction = error_direction;
            _stable_error_count = kRequiredStableErrorCount;
        }
        if (_stable_error_count < kRequiredStableErrorCount) {
            return false;
        }
        _stable_error_count = 0;

        delta_db = std::clamp(delta_db, -_max_step_db, _max_step_db);
        const double next_gain_db = std::clamp(_current_gain_db + delta_db, _min_gain_db, _max_gain_db);
        if (std::abs(next_gain_db - _current_gain_db) < 0.25) {
            return false;
        }

        adjustment.delta_db = next_gain_db - _current_gain_db;
        adjustment.next_gain_db = next_gain_db;
        _current_gain_db = next_gain_db;
        return true;
    }

    static void analyze_sync_symbol_saturation(
        const std::complex<float>* samples,
        size_t count,
        RxAgcAdjustment& adjustment
    ) {
        if (samples == nullptr || count == 0) {
            return;
        }

        constexpr float kNearFullScaleThreshold = 0.65f;
        constexpr float kHardFullScaleThreshold = 0.7f;
        const size_t near_count_limit = std::max<size_t>(8, count / 100);

        uint64_t near_count = 0;
        uint64_t hard_count = 0;
        float max_component = 0.0f;

        #pragma omp simd simdlen(16) reduction(max:max_component) reduction(+:near_count, hard_count)
        for (size_t i = 0; i < count; ++i) {
            const float i_abs = std::fabs(samples[i].real());
            const float q_abs = std::fabs(samples[i].imag());
            const float component_abs = (i_abs > q_abs) ? i_abs : q_abs;
            max_component = (component_abs > max_component) ? component_abs : max_component;
            near_count += static_cast<uint64_t>(component_abs >= kNearFullScaleThreshold);
            hard_count += static_cast<uint64_t>(component_abs >= kHardFullScaleThreshold);
        }

        adjustment.max_sync_sample_component = static_cast<double>(max_component);
        adjustment.near_full_scale_count = static_cast<size_t>(near_count);
        adjustment.hard_full_scale_count = static_cast<size_t>(hard_count);
        adjustment.saturation_detected = (hard_count > 0) || (near_count >= near_count_limit);
    }

    bool _enabled = false;
    double _low_threshold_db = 11.0;
    double _high_threshold_db = 13.0;
    double _max_step_db = 3.0;
    size_t _update_frames = 1;
    size_t _frames_since_update = 0;
    double _min_gain_db = 0.0;
    double _max_gain_db = 0.0;
    double _current_gain_db = 0.0;
    double _filtered_peak_db = -120.0;
    bool _has_filtered_peak = false;
    int _last_error_direction = 0;
    size_t _stable_error_count = 0;
    static constexpr size_t kSaturationRiseHoldoffUpdates = 4;
    size_t _saturation_rise_holdoff = 0;
};

class SyncSearchRxGainSweep {
public:
    explicit SyncSearchRxGainSweep(const Config& cfg)
        : _enabled(cfg.rf_sampling.rx_agc_enable),
          _default_gain_db(cfg.rf_sampling.rx_gain) {}

    void initialize(double default_gain_db, double min_gain_db, double max_gain_db) {
        _min_gain_db = std::min(min_gain_db, max_gain_db);
        _max_gain_db = std::max(min_gain_db, max_gain_db);
        _default_gain_db = std::clamp(default_gain_db, _min_gain_db, _max_gain_db);
        _current_gain_db = _default_gain_db;
        _frames_without_sync = 0;
    }

    bool enabled() const {
        return _enabled;
    }

    template <typename ApplyGainFn, typename SyncAgcFn>
    bool reset_to_default(ApplyGainFn&& apply_gain, SyncAgcFn&& sync_agc, double* applied_gain_db = nullptr) {
        if (!_enabled) {
            return false;
        }
        _frames_without_sync = 0;
        const double next_gain_db = std::clamp(_default_gain_db, _min_gain_db, _max_gain_db);
        const bool changed = std::abs(next_gain_db - _current_gain_db) >= 0.25;
        _current_gain_db = next_gain_db;
        sync_agc(_current_gain_db);
        if (!changed) {
            return false;
        }
        apply_gain(_current_gain_db);
        if (applied_gain_db != nullptr) {
            *applied_gain_db = _current_gain_db;
        }
        return true;
    }

    void note_sync_found() {
        _frames_without_sync = 0;
    }

    template <typename ApplyGainFn, typename SyncAgcFn>
    bool on_search_miss(
        size_t equivalent_frames,
        ApplyGainFn&& apply_gain,
        SyncAgcFn&& sync_agc,
        double* applied_gain_db = nullptr
    ) {
        if (!_enabled || equivalent_frames == 0) {
            return false;
        }
        _frames_without_sync += equivalent_frames;
        if (_frames_without_sync < kFramesPerStep) {
            return false;
        }
        _frames_without_sync %= kFramesPerStep;

        const double next_gain_db = (_current_gain_db >= _max_gain_db - 1e-6)
            ? _min_gain_db
            : std::min(_current_gain_db + kGainStepDb, _max_gain_db);
        if (std::abs(next_gain_db - _current_gain_db) < 0.25) {
            return false;
        }

        _current_gain_db = next_gain_db;
        sync_agc(_current_gain_db);
        apply_gain(_current_gain_db);
        if (applied_gain_db != nullptr) {
            *applied_gain_db = _current_gain_db;
        }
        return true;
    }

private:
    static constexpr size_t kFramesPerStep = 10;
    static constexpr double kGainStepDb = 1.0;

    bool _enabled = false;
    double _default_gain_db = 0.0;
    double _current_gain_db = 0.0;
    double _min_gain_db = 0.0;
    double _max_gain_db = 0.0;
    size_t _frames_without_sync = 0;
};

/**
 * @brief Sensing Frame Structure.
 * 
 * Holds processed frequency-domain symbols for sensing applications (Radar/ISAC).
 * Includes both received symbols and (estimated) transmitted symbols for channel estimation.
 */
struct SensingFrame {
    std::vector<AlignedVector> rx_symbols;  // Received symbol collection
    std::vector<AlignedVector> tx_symbols;  // Corresponding transmitted symbol collection
    // The local downlink estimate produced from the same received frame.  eRTM
    // consumes it on the sensing thread so the TO measurement and the sensing
    // correction share one downlink-frame coordinate system.
    AlignedVector ertm_downlink_channel_freq;
    float CFO;
    float SFO;
    float delay_offset;
    int alignment_samples = 0;
    bool force_ertm_remeasure = false;
    uint64_t generation = 0;
};

inline uint64_t host_to_network_u64(uint64_t value) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap64(value);
#else
    return value;
#endif
}

#pragma pack(push, 1)
struct CompactSensingFrameHeader {
    uint32_t magic_version = 0;
    uint32_t mask_hash = 0;
    uint32_t re_count = 0;
    uint64_t frame_start_symbol_index = 0;
};
#pragma pack(pop)

static_assert(sizeof(CompactSensingFrameHeader) == 20, "CompactSensingFrameHeader size mismatch");
inline constexpr uint32_t kCompactSensingMagicVersion = 0x43534D31u; // "CSM1"

#pragma pack(push, 1)
struct AggregatedSensingFrameHeader {
    uint32_t magic_version = 0;
    uint32_t channel_count = 0;
    uint32_t channel_payload_bytes = 0;
    uint32_t channel_mask = 0;
    uint64_t frame_start_symbol_index = 0;
};
#pragma pack(pop)

static_assert(sizeof(AggregatedSensingFrameHeader) == 24, "AggregatedSensingFrameHeader size mismatch");
inline constexpr uint32_t kAggregatedSensingMagicVersion = 0x41534731u; // "ASG1"

#pragma pack(push, 1)
struct SensingMetadataWireHeader {
    char magic[4];
    uint32_t total_bytes = 0;
    uint32_t flags = 0;
    uint32_t cfar_point_count = 0;
    uint32_t cluster_count = 0;
    uint32_t md_rows = 0;
    uint32_t md_cols = 0;
    uint32_t cfar_hits = 0;
    uint32_t cfar_shown_hits = 0;
    uint32_t invalid_cells = 0;
    uint32_t nonfinite_cells = 0;
    uint32_t nonpositive_cells = 0;
    float noise_min = 0.0f;
    float noise_max = 0.0f;
    float thresh_min = 0.0f;
    float thresh_max = 0.0f;
    float power_min_db = 0.0f;
    float md_t0 = 0.0f;
    float md_t1 = 0.0f;
    float md_f0 = 0.0f;
    float md_f1 = 0.0f;
    uint64_t frame_start_symbol_index = 0;
};
#pragma pack(pop)

static_assert(sizeof(SensingMetadataWireHeader) == 92, "SensingMetadataWireHeader size mismatch");

#pragma pack(push, 1)
struct SensingClusterWire {
    int32_t peak_doppler_idx = 0;
    int32_t peak_range_idx = 0;
    float peak_strength_db = 0.0f;
    uint32_t cluster_size = 0;
    float centroid_doppler_idx = 0.0f;
    float centroid_range_idx = 0.0f;
};
#pragma pack(pop)

static_assert(sizeof(SensingClusterWire) == 24, "SensingClusterWire size mismatch");

#pragma pack(push, 1)
struct AggregatedSensingMetadataHeader {
    char magic[4];
    uint32_t channel_count = 0;
    uint32_t channel_mask = 0;
    uint32_t reserved = 0;
    uint64_t frame_start_symbol_index = 0;
};
#pragma pack(pop)

static_assert(sizeof(AggregatedSensingMetadataHeader) == 24, "AggregatedSensingMetadataHeader size mismatch");
inline constexpr char kSensingMetadataMagic[4] = {'S', 'M', 'D', '1'};
inline constexpr char kAggregatedSensingMetadataMagic[4] = {'A', 'S', 'M', '1'};

enum class SensingViewerFrameFormat : uint32_t {
    DenseChannelBuffer = 0,
    CompactRaw = 1,
    DenseRangeDoppler = 2,
    CompactSparse = 3
};

enum class SensingViewerWireDataFormat : uint32_t {
    ComplexFloat32 = 0,
    ComplexFloat16 = 1,
};

inline constexpr uint32_t kSensingMetadataFlagCfarEnabled = 1u << 0;
inline constexpr uint32_t kSensingMetadataFlagMicroDopplerEnabled = 1u << 1;

inline constexpr uint32_t kSensingViewerParamsVersion = 7u;
inline constexpr uint32_t kSensingViewerFlagCompactMask = 1u << 0;
inline constexpr uint32_t kSensingViewerFlagCompactLocalDelayDoppler = 1u << 1;
inline constexpr uint32_t kSensingViewerFlagSkipSensingFft = 1u << 2;
inline constexpr uint32_t kSensingViewerFlagEnableMti = 1u << 3;
inline constexpr uint32_t kSensingViewerFlagBistatic = 1u << 4;
inline constexpr uint32_t kSensingViewerFlagAggregatedStream = 1u << 5;
inline constexpr uint32_t kSensingViewerFlagBackendProcessing = 1u << 6;
inline constexpr uint32_t kSensingViewerFlagMetadataSidecar = 1u << 7;

#pragma pack(push, 1)
struct SensingViewerParamsPacket {
    char header[4];
    char command[4];
    uint32_t version = 0;
    uint32_t flags = 0;
    uint32_t frame_format = 0;
    uint32_t wire_rows = 0;
    uint32_t wire_cols = 0;
    uint32_t active_rows = 0;
    uint32_t active_cols = 0;
    uint32_t frame_symbol_period = 0;
    uint32_t range_fft_size = 0;
    uint32_t doppler_fft_size = 0;
    uint32_t compact_mask_hash = 0;
    uint32_t wire_data_format = 0;
    uint32_t stream_channel_count = 0;
    uint32_t stream_channel_mask = 0;
    uint32_t os_cfar_rank_percent_x100 = 0;
    uint32_t os_cfar_suppress_doppler = 0;
    uint32_t os_cfar_suppress_range = 0;
    // V6/V7 additions — zero in older packets; viewers treat 0 as "not provided".
    // V7 stores center_freq / 100 so uint32 covers RF carriers well above mmWave.
    uint32_t center_freq_hz_div100 = 0; // center_freq / 100 (units: 100 Hz; uint32 covers up to ~429 GHz)
    uint32_t sample_rate_hz_div100 = 0; // sample_rate / 100 (units: 100 Hz; uint32 covers up to ~429 GHz)
    uint32_t antenna_spacing_um = 0;    // physical ULA element spacing in micrometres (0 = not provided)
};
#pragma pack(pop)

static_assert(sizeof(SensingViewerParamsPacket) == 88, "SensingViewerParamsPacket size mismatch");

inline std::vector<uint8_t> serialize_sensing_metadata(
    const SensingMetadata& metadata,
    uint64_t frame_start_symbol_index)
{
    const size_t cfar_point_bytes =
        metadata.cfar_points.size() * sizeof(SensingDetectionPoint);
    const size_t cluster_bytes =
        metadata.target_clusters.size() * sizeof(SensingClusterWire);
    const size_t md_bytes =
        metadata.micro_doppler_spectrum.size() * sizeof(float);
    const size_t total_bytes =
        sizeof(SensingMetadataWireHeader) + cfar_point_bytes + cluster_bytes + md_bytes;

    std::vector<uint8_t> bytes(total_bytes, 0);
    SensingMetadataWireHeader header{};
    std::memcpy(header.magic, kSensingMetadataMagic, sizeof(header.magic));
    header.total_bytes = static_cast<uint32_t>(total_bytes);
    if (metadata.cfar_enabled) {
        header.flags |= kSensingMetadataFlagCfarEnabled;
    }
    if (metadata.micro_doppler_enabled) {
        header.flags |= kSensingMetadataFlagMicroDopplerEnabled;
    }
    header.cfar_point_count = static_cast<uint32_t>(metadata.cfar_points.size());
    header.cluster_count = static_cast<uint32_t>(metadata.target_clusters.size());
    header.md_rows = metadata.micro_doppler_rows;
    header.md_cols = metadata.micro_doppler_cols;
    header.cfar_hits = metadata.cfar_hits;
    header.cfar_shown_hits = metadata.cfar_shown_hits;
    header.invalid_cells = metadata.cfar_stats.invalid_cells;
    header.nonfinite_cells = metadata.cfar_stats.nonfinite_cells;
    header.nonpositive_cells = metadata.cfar_stats.nonpositive_cells;
    header.noise_min = metadata.cfar_stats.noise_min;
    header.noise_max = metadata.cfar_stats.noise_max;
    header.thresh_min = metadata.cfar_stats.thresh_min;
    header.thresh_max = metadata.cfar_stats.thresh_max;
    header.power_min_db = metadata.cfar_stats.power_min_db;
    header.md_t0 = metadata.micro_doppler_extent[0];
    header.md_t1 = metadata.micro_doppler_extent[1];
    header.md_f0 = metadata.micro_doppler_extent[2];
    header.md_f1 = metadata.micro_doppler_extent[3];
    header.frame_start_symbol_index = frame_start_symbol_index;

    std::memcpy(bytes.data(), &header, sizeof(header));

    size_t offset = sizeof(header);
    if (!metadata.cfar_points.empty()) {
        std::memcpy(
            bytes.data() + static_cast<std::ptrdiff_t>(offset),
            metadata.cfar_points.data(),
            cfar_point_bytes);
        offset += cfar_point_bytes;
    }
    if (!metadata.target_clusters.empty()) {
        for (size_t i = 0; i < metadata.target_clusters.size(); ++i) {
            SensingClusterWire cluster_wire{};
            cluster_wire.peak_doppler_idx = metadata.target_clusters[i].peak_doppler_idx;
            cluster_wire.peak_range_idx = metadata.target_clusters[i].peak_range_idx;
            cluster_wire.peak_strength_db = metadata.target_clusters[i].peak_strength_db;
            cluster_wire.cluster_size = metadata.target_clusters[i].cluster_size;
            cluster_wire.centroid_doppler_idx = metadata.target_clusters[i].centroid_doppler_idx;
            cluster_wire.centroid_range_idx = metadata.target_clusters[i].centroid_range_idx;
            std::memcpy(
                bytes.data() + static_cast<std::ptrdiff_t>(offset + i * sizeof(SensingClusterWire)),
                &cluster_wire,
                sizeof(SensingClusterWire));
        }
        offset += cluster_bytes;
    }
    if (!metadata.micro_doppler_spectrum.empty()) {
        std::memcpy(
            bytes.data() + static_cast<std::ptrdiff_t>(offset),
            metadata.micro_doppler_spectrum.data(),
            md_bytes);
    }
    return bytes;
}

inline uint32_t sensing_viewer_active_flags(
    const Config& cfg,
    bool skip_sensing_fft,
    bool enable_mti,
    bool bistatic = false)
{
    uint32_t flags = 0;
    const CompactSensingMaskAnalysis analysis = analyze_compact_sensing_mask(cfg);
    const bool backend_processing = backend_sensing_processing_supported(cfg);
    if (sensing_output_mode_is_compact_mask(cfg)) {
        flags |= kSensingViewerFlagCompactMask;
    }
    if (analysis.local_delay_doppler_supported) {
        flags |= kSensingViewerFlagCompactLocalDelayDoppler;
    }
    if (skip_sensing_fft && !backend_processing) {
        flags |= kSensingViewerFlagSkipSensingFft;
    }
    if (enable_mti) {
        flags |= kSensingViewerFlagEnableMti;
    }
    if (bistatic) {
        flags |= kSensingViewerFlagBistatic;
    }
    if (backend_processing) {
        flags |= kSensingViewerFlagBackendProcessing;
        flags |= kSensingViewerFlagMetadataSidecar;
    }
    return flags;
}

inline SensingViewerFrameFormat sensing_viewer_frame_format(
    const Config& cfg,
    bool skip_sensing_fft)
{
    if (backend_sensing_processing_supported(cfg)) {
        return SensingViewerFrameFormat::DenseRangeDoppler;
    }
    if (!sensing_output_mode_is_compact_mask(cfg)) {
        return skip_sensing_fft
            ? SensingViewerFrameFormat::DenseChannelBuffer
            : SensingViewerFrameFormat::DenseRangeDoppler;
    }

    const CompactSensingMaskAnalysis analysis = analyze_compact_sensing_mask(cfg);
    if (!analysis.local_delay_doppler_supported) {
        return SensingViewerFrameFormat::CompactSparse;
    }
    return skip_sensing_fft
        ? SensingViewerFrameFormat::CompactRaw
        : SensingViewerFrameFormat::DenseRangeDoppler;
}

inline size_t sensing_viewer_wire_rows(
    const Config& cfg,
    bool skip_sensing_fft)
{
    if (backend_sensing_processing_supported(cfg)) {
        return resolved_sensing_view_doppler_bins(cfg);
    }
    const CompactSensingMaskAnalysis analysis = analyze_compact_sensing_mask(cfg);
    switch (sensing_viewer_frame_format(cfg, skip_sensing_fft)) {
    case SensingViewerFrameFormat::DenseChannelBuffer:
    case SensingViewerFrameFormat::DenseRangeDoppler:
        return cfg.sensing.doppler_fft_size;
    case SensingViewerFrameFormat::CompactRaw:
        return cfg.ofdm.sensing_symbol_num;
    case SensingViewerFrameFormat::CompactSparse:
    default:
        return 0;
    }
}

inline size_t sensing_viewer_wire_cols(
    const Config& cfg,
    bool skip_sensing_fft)
{
    if (backend_sensing_processing_supported(cfg)) {
        return resolved_sensing_view_range_bins(cfg);
    }
    const CompactSensingMaskAnalysis analysis = analyze_compact_sensing_mask(cfg);
    switch (sensing_viewer_frame_format(cfg, skip_sensing_fft)) {
    case SensingViewerFrameFormat::DenseChannelBuffer:
    case SensingViewerFrameFormat::DenseRangeDoppler:
        return cfg.sensing.range_fft_size;
    case SensingViewerFrameFormat::CompactRaw:
        return analysis.common_subcarrier_count;
    case SensingViewerFrameFormat::CompactSparse:
    default:
        return 0;
    }
}

inline size_t sensing_viewer_active_rows(
    const Config& cfg,
    bool skip_sensing_fft)
{
    if (backend_sensing_processing_supported(cfg)) {
        return resolved_sensing_view_doppler_bins(cfg);
    }
    const CompactSensingMaskAnalysis analysis = analyze_compact_sensing_mask(cfg);
    switch (sensing_viewer_frame_format(cfg, skip_sensing_fft)) {
    case SensingViewerFrameFormat::DenseChannelBuffer:
        return std::min(cfg.ofdm.sensing_symbol_num, cfg.sensing.doppler_fft_size);
    case SensingViewerFrameFormat::CompactRaw:
        return cfg.ofdm.sensing_symbol_num;
    case SensingViewerFrameFormat::DenseRangeDoppler:
        return cfg.sensing.doppler_fft_size;
    case SensingViewerFrameFormat::CompactSparse:
    default:
        return 0;
    }
}

inline size_t sensing_viewer_active_cols(
    const Config& cfg,
    bool skip_sensing_fft)
{
    if (backend_sensing_processing_supported(cfg)) {
        return resolved_sensing_view_range_bins(cfg);
    }
    const CompactSensingMaskAnalysis analysis = analyze_compact_sensing_mask(cfg);
    switch (sensing_viewer_frame_format(cfg, skip_sensing_fft)) {
    case SensingViewerFrameFormat::DenseChannelBuffer:
        return std::min(cfg.ofdm.fft_size, cfg.sensing.range_fft_size);
    case SensingViewerFrameFormat::CompactRaw:
        return analysis.common_subcarrier_count;
    case SensingViewerFrameFormat::DenseRangeDoppler:
        return cfg.sensing.range_fft_size;
    case SensingViewerFrameFormat::CompactSparse:
    default:
        return 0;
    }
}

inline uint32_t sensing_viewer_mask_hash(const Config& cfg)
{
    if (!sensing_output_mode_is_compact_mask(cfg)) {
        return 0;
    }
    return build_sensing_mask_layout(cfg).mask_hash;
}

inline SensingViewerWireDataFormat sensing_viewer_wire_data_format(const Config& cfg)
{
    switch (cfg.sensing.on_wire_format) {
    case SensingOnWireFormat::ComplexFloat16:
        return SensingViewerWireDataFormat::ComplexFloat16;
    case SensingOnWireFormat::ComplexFloat32:
    default:
        return SensingViewerWireDataFormat::ComplexFloat32;
    }
}

inline SensingViewerParamsPacket make_sensing_viewer_params_packet(
    const Config& cfg,
    bool skip_sensing_fft,
    bool enable_mti,
    float os_cfar_rank_percent = 75.0f,
    int os_cfar_suppress_doppler = 2,
    int os_cfar_suppress_range = 2,
    bool bistatic = false,
    uint32_t stream_channel_count = 1,
    uint32_t stream_channel_mask = 0x1u,
    bool aggregated_stream = false,
    double antenna_spacing_m = 0.0)
{
    SensingViewerParamsPacket packet{};
    std::memcpy(packet.header, "CTRL", 4);
    std::memcpy(packet.command, "PARM", 4);
    packet.version = htonl(kSensingViewerParamsVersion);
    uint32_t flags = sensing_viewer_active_flags(cfg, skip_sensing_fft, enable_mti, bistatic);
    if (aggregated_stream) {
        flags |= kSensingViewerFlagAggregatedStream;
    }
    packet.flags = htonl(flags);
    packet.frame_format = htonl(static_cast<uint32_t>(sensing_viewer_frame_format(cfg, skip_sensing_fft)));
    packet.wire_rows = htonl(static_cast<uint32_t>(sensing_viewer_wire_rows(cfg, skip_sensing_fft)));
    packet.wire_cols = htonl(static_cast<uint32_t>(sensing_viewer_wire_cols(cfg, skip_sensing_fft)));
    packet.active_rows = htonl(static_cast<uint32_t>(sensing_viewer_active_rows(cfg, skip_sensing_fft)));
    packet.active_cols = htonl(static_cast<uint32_t>(sensing_viewer_active_cols(cfg, skip_sensing_fft)));
    packet.frame_symbol_period = htonl(static_cast<uint32_t>(cfg.ofdm.num_symbols));
    packet.range_fft_size = htonl(static_cast<uint32_t>(cfg.sensing.range_fft_size));
    packet.doppler_fft_size = htonl(static_cast<uint32_t>(cfg.sensing.doppler_fft_size));
    packet.compact_mask_hash = htonl(sensing_viewer_mask_hash(cfg));
    packet.wire_data_format = htonl(static_cast<uint32_t>(sensing_viewer_wire_data_format(cfg)));
    packet.stream_channel_count = htonl(stream_channel_count);
    packet.stream_channel_mask = htonl(stream_channel_mask);
    const uint32_t rank_percent_x100 = static_cast<uint32_t>(std::llround(
        std::clamp(static_cast<double>(os_cfar_rank_percent), 0.0, 100.0) * 100.0));
    packet.os_cfar_rank_percent_x100 = htonl(rank_percent_x100);
    packet.os_cfar_suppress_doppler = htonl(static_cast<uint32_t>(std::max(0, os_cfar_suppress_doppler)));
    packet.os_cfar_suppress_range = htonl(static_cast<uint32_t>(std::max(0, os_cfar_suppress_range)));
    // V7: physical radio parameters so the viewer can auto-configure axes and AoA.
    if (cfg.downlink.center_freq > 0.0) {
        packet.center_freq_hz_div100 = htonl(static_cast<uint32_t>(
            std::llround(std::clamp(cfg.downlink.center_freq / 100.0, 0.0, 4294967295.0))));
    }
    if (cfg.rf_sampling.sample_rate > 0.0) {
        packet.sample_rate_hz_div100 = htonl(static_cast<uint32_t>(
            std::llround(std::clamp(cfg.rf_sampling.sample_rate / 100.0, 0.0, 4294967295.0))));
    }
    if (antenna_spacing_m > 0.0) {
        packet.antenna_spacing_um = htonl(static_cast<uint32_t>(
            std::llround(std::clamp(antenna_spacing_m * 1e6, 0.0, 4294967295.0))));
    }
    return packet;
}

struct UdpDatagramReceiveResult {
    ssize_t bytes = -1;
    bool truncated = false;
    bool has_kernel_drop_count = false;
    uint32_t kernel_drop_count = 0;
};

inline bool enable_udp_rx_overflow_reporting(int sockfd) {
#ifdef SO_RXQ_OVFL
    const int enabled = 1;
    return setsockopt(
               sockfd, SOL_SOCKET, SO_RXQ_OVFL,
               &enabled, sizeof(enabled)) == 0;
#else
    (void)sockfd;
    return false;
#endif
}

/**
 * @brief Receive one UDP datagram and expose kernel/truncation loss metadata.
 *
 * SO_RXQ_OVFL reports the cumulative number of datagrams dropped by the local
 * socket receive queue. MSG_TRUNC identifies an application buffer that was
 * too small for the datagram. The caller owns delta tracking and warning logs.
 */
inline UdpDatagramReceiveResult receive_udp_datagram(
    int sockfd, void* buffer, size_t buffer_size)
{
    UdpDatagramReceiveResult result;
    iovec iov{};
    iov.iov_base = buffer;
    iov.iov_len = buffer_size;

    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(uint32_t))]{};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    result.bytes = recvmsg(sockfd, &msg, 0);
    if (result.bytes < 0) {
        return result;
    }
    result.truncated = (msg.msg_flags & MSG_TRUNC) != 0;
#ifdef SO_RXQ_OVFL
    for (cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
         cmsg != nullptr;
         cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SO_RXQ_OVFL &&
            cmsg->cmsg_len >= CMSG_LEN(sizeof(uint32_t))) {
            std::memcpy(
                &result.kernel_drop_count,
                CMSG_DATA(cmsg),
                sizeof(result.kernel_drop_count));
            result.has_kernel_drop_count = true;
            break;
        }
    }
#endif
    return result;
}

/**
 * @brief Base Class for UDP Senders.
 * 
 * Provides common functionality for creating UDP sockets and sending raw data.
 * Handles socket creation, address configuration, and basic error checking.
 */
class UdpBaseSender {
protected:
    int sockfd_ = -1;
    struct sockaddr_in dest_addr_;

public:
    UdpBaseSender(const std::string& ip, uint16_t port) {
        // Create UDP socket
        sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0) {
            throw std::runtime_error("Failed to create UDP socket: " + std::string(strerror(errno)));
        }

        // Set destination address
        memset(&dest_addr_, 0, sizeof(dest_addr_));
        dest_addr_.sin_family = AF_INET;
        dest_addr_.sin_port = htons(port);

        // Convert and validate IP address
        if (inet_pton(AF_INET, ip.c_str(), &dest_addr_.sin_addr) <= 0) {
            close(sockfd_);
            sockfd_ = -1;
            throw std::runtime_error("Invalid IP address: " + ip);
        }

        // Increase send buffer size to avoid dropped packets
        int sendbuff = 4 * 1024 * 1024; // 4MB
        if (setsockopt(sockfd_, SOL_SOCKET, SO_SNDBUF, &sendbuff, sizeof(sendbuff)) < 0) {
            LOG_G_WARN_M(Config) << "Warning: Failed to set send buffer size";
        }
    }

    virtual ~UdpBaseSender() {
        if (sockfd_ != -1) {
            close(sockfd_);
            sockfd_ = -1;
        }
    }

    // Basic send method (usable by derived classes)
    template <typename T>
    void send_raw(const T* data, size_t size_bytes) {
        ssize_t bytes_sent = sendto(sockfd_, data, size_bytes, 0,
                                  reinterpret_cast<struct sockaddr*>(&dest_addr_),
                                  sizeof(dest_addr_));

        // Complete error checking and handling
        if (bytes_sent < 0) {
            throw std::runtime_error("UDP send failed: " + std::string(strerror(errno)));
        } else if (static_cast<size_t>(bytes_sent) != size_bytes) {
            throw std::runtime_error("UDP send partial: " + std::to_string(bytes_sent) +
                                     " of " + std::to_string(size_bytes) + " bytes sent");
        }
    }
};

/**
 * @brief Simple UDP Data Sender.
 *
 * A general-purpose UDP sender that inherits from UdpBaseSender.
 * Provides template methods to send raw data or standard containers (e.g., std::vector).
 */
class UdpSender : public UdpBaseSender {
public:
    UdpSender(const std::string& ip, uint16_t port) : UdpBaseSender(ip, port) {}

    UdpSender(
        const std::string& ip,
        uint16_t port,
        const UdpEgressPacerConfig& pacer_cfg,
        bool pacer_stats_enabled = false)
        : UdpBaseSender(ip, port)
        , pacer_cfg_(pacer_cfg)
        , pacer_stats_enabled_(pacer_stats_enabled) {
        if (pacer_cfg_.enabled) {
            pacer_running_ = true;
            pacer_thread_ = std::thread(&UdpSender::pacer_loop, this);
            LOG_G_INFO_M(Config) << "UDP egress pacer enabled: target_mbps="
                         << pacer_cfg_.target_mbps
                         << " (0=auto), queue_packets="
                         << pacer_cfg_.queue_packets
                         << ", max_delay_ms="
                         << pacer_cfg_.max_delay_ms;
        }
    }

    ~UdpSender() override {
        stop_pacer();
    }

    UdpSender(const UdpSender&) = delete;
    UdpSender& operator=(const UdpSender&) = delete;

    template <typename T>
    void send(const T* data, size_t size_bytes) {
        if (!pacer_cfg_.enabled) {
            send_raw(data, size_bytes);
            return;
        }
        enqueue_paced(data, size_bytes);
    }

    template <typename Container>
    void send_container(const Container& data) {
        send(data.data(), data.size() * sizeof(typename Container::value_type));
    }

private:
    using SteadyClock = std::chrono::steady_clock;

    struct PacerPacket {
        std::vector<uint8_t> bytes;
        SteadyClock::time_point enqueue_time;
    };

    UdpEgressPacerConfig pacer_cfg_;
    bool pacer_stats_enabled_ = false;
    std::mutex pacer_mutex_;
    std::condition_variable pacer_cv_;
    std::deque<PacerPacket> pacer_queue_;
    std::thread pacer_thread_;
    bool pacer_running_ = false;
    uint64_t pacer_dropped_oldest_ = 0;
    uint64_t pacer_dropped_stale_ = 0;
    uint64_t pacer_enqueued_packets_ = 0;
    uint64_t pacer_enqueued_bytes_ = 0;
    uint64_t pacer_sent_packets_ = 0;
    uint64_t pacer_sent_bytes_ = 0;
    uint64_t pacer_send_failed_ = 0;
    size_t pacer_max_queue_depth_ = 0;
    SteadyClock::time_point next_send_time_{};
    SteadyClock::time_point auto_rate_interval_start_{};
    SteadyClock::time_point pacer_stats_last_log_time_{};
    size_t auto_rate_interval_bytes_ = 0;
    double auto_rate_bytes_per_s_ = 0.0;
    uint64_t pacer_stats_last_enqueued_bytes_ = 0;
    uint64_t pacer_stats_last_sent_bytes_ = 0;
    uint64_t pacer_stats_last_dropped_oldest_ = 0;
    uint64_t pacer_stats_last_dropped_stale_ = 0;
    uint64_t pacer_stats_last_send_failed_ = 0;
    static constexpr double kAutoRateBootstrapMbps = 8.0;
    static constexpr double kPacerStatsIntervalS = 1.0;

    static double mbps_to_bytes_per_s(double mbps) {
        return (mbps * 1000.0 * 1000.0) / 8.0;
    }

    static double bytes_per_s_to_mbps(double bytes_per_s) {
        return (bytes_per_s * 8.0) / (1000.0 * 1000.0);
    }

    double current_rate_bytes_per_s_locked() const {
        if (pacer_cfg_.target_mbps > 0.0) {
            return mbps_to_bytes_per_s(pacer_cfg_.target_mbps);
        }
        if (auto_rate_bytes_per_s_ > 0.0) {
            return auto_rate_bytes_per_s_;
        }
        return mbps_to_bytes_per_s(kAutoRateBootstrapMbps);
    }

    void update_auto_rate_locked(size_t size_bytes, SteadyClock::time_point now) {
        if (pacer_cfg_.target_mbps > 0.0) {
            return;
        }
        if (auto_rate_interval_start_ == SteadyClock::time_point{}) {
            auto_rate_interval_start_ = now;
        }
        auto_rate_interval_bytes_ += size_bytes;

        const double elapsed_s =
            std::chrono::duration<double>(now - auto_rate_interval_start_).count();
        if (elapsed_s < 0.2) {
            return;
        }

        const double interval_rate = static_cast<double>(auto_rate_interval_bytes_) / elapsed_s;
        if (interval_rate > 0.0) {
            if (auto_rate_bytes_per_s_ <= 0.0) {
                auto_rate_bytes_per_s_ = interval_rate;
            } else {
                auto_rate_bytes_per_s_ = 0.85 * auto_rate_bytes_per_s_ + 0.15 * interval_rate;
            }
        }
        auto_rate_interval_start_ = now;
        auto_rate_interval_bytes_ = 0;
    }

    template <typename T>
    void enqueue_paced(const T* data, size_t size_bytes) {
        if (size_bytes == 0) {
            return;
        }
        PacerPacket packet;
        packet.bytes.resize(size_bytes);
        std::memcpy(packet.bytes.data(), data, size_bytes);
        packet.enqueue_time = SteadyClock::now();

        {
            std::lock_guard<std::mutex> lock(pacer_mutex_);
            update_auto_rate_locked(size_bytes, packet.enqueue_time);
            while (pacer_queue_.size() >= pacer_cfg_.queue_packets) {
                pacer_queue_.pop_front();
                ++pacer_dropped_oldest_;
                if (pacer_dropped_oldest_ <= 5 || (pacer_dropped_oldest_ % 100) == 0) {
                    LOG_G_WARN_M(Config) << "UDP egress pacer dropped oldest packet: queue full, dropped="
                                 << pacer_dropped_oldest_;
                }
            }
            pacer_queue_.push_back(std::move(packet));
            ++pacer_enqueued_packets_;
            pacer_enqueued_bytes_ += size_bytes;
            pacer_max_queue_depth_ = std::max(pacer_max_queue_depth_, pacer_queue_.size());
        }
        pacer_cv_.notify_one();
    }

    void stop_pacer() {
        if (!pacer_cfg_.enabled) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(pacer_mutex_);
            pacer_running_ = false;
            pacer_queue_.clear();
        }
        pacer_cv_.notify_one();
        if (pacer_thread_.joinable()) {
            pacer_thread_.join();
        }
    }

    bool wait_until_send_time(SteadyClock::time_point send_time) {
        std::unique_lock<std::mutex> lock(pacer_mutex_);
        return !pacer_cv_.wait_until(lock, send_time, [this]() {
            return !pacer_running_;
        });
    }

    bool packet_is_stale(const PacerPacket& packet, SteadyClock::time_point now) const {
        if (pacer_cfg_.max_delay_ms <= 0.0) {
            return false;
        }
        const double age_ms =
            std::chrono::duration<double, std::milli>(now - packet.enqueue_time).count();
        return age_ms > pacer_cfg_.max_delay_ms;
    }

    void log_pacer_stats(SteadyClock::time_point now) {
        if (!pacer_stats_enabled_) {
            return;
        }
        if (pacer_stats_last_log_time_ == SteadyClock::time_point{}) {
            std::lock_guard<std::mutex> lock(pacer_mutex_);
            pacer_stats_last_log_time_ = now;
            pacer_stats_last_enqueued_bytes_ = pacer_enqueued_bytes_;
            pacer_stats_last_sent_bytes_ = pacer_sent_bytes_;
            pacer_stats_last_dropped_oldest_ = pacer_dropped_oldest_;
            pacer_stats_last_dropped_stale_ = pacer_dropped_stale_;
            pacer_stats_last_send_failed_ = pacer_send_failed_;
            return;
        }

        const double elapsed_s =
            std::chrono::duration<double>(now - pacer_stats_last_log_time_).count();
        if (elapsed_s < kPacerStatsIntervalS) {
            return;
        }

        uint64_t enqueued_packets = 0;
        uint64_t enqueued_bytes = 0;
        uint64_t dropped_oldest = 0;
        size_t queue_depth = 0;
        size_t max_queue_depth = 0;
        double auto_rate_bytes_per_s = 0.0;
        {
            std::lock_guard<std::mutex> lock(pacer_mutex_);
            enqueued_packets = pacer_enqueued_packets_;
            enqueued_bytes = pacer_enqueued_bytes_;
            dropped_oldest = pacer_dropped_oldest_;
            queue_depth = pacer_queue_.size();
            max_queue_depth = pacer_max_queue_depth_;
            auto_rate_bytes_per_s = auto_rate_bytes_per_s_;
        }

        const uint64_t interval_enqueued_bytes = enqueued_bytes - pacer_stats_last_enqueued_bytes_;
        const uint64_t interval_sent_bytes = pacer_sent_bytes_ - pacer_stats_last_sent_bytes_;
        const uint64_t interval_dropped_oldest = dropped_oldest - pacer_stats_last_dropped_oldest_;
        const uint64_t interval_dropped_stale = pacer_dropped_stale_ - pacer_stats_last_dropped_stale_;
        const uint64_t interval_send_failed = pacer_send_failed_ - pacer_stats_last_send_failed_;
        const double enqueue_mbps =
            bytes_per_s_to_mbps(static_cast<double>(interval_enqueued_bytes) / elapsed_s);
        const double send_mbps =
            bytes_per_s_to_mbps(static_cast<double>(interval_sent_bytes) / elapsed_s);
        const double effective_rate_mbps = bytes_per_s_to_mbps(
            pacer_cfg_.target_mbps > 0.0
                ? mbps_to_bytes_per_s(pacer_cfg_.target_mbps)
                : (auto_rate_bytes_per_s > 0.0
                       ? auto_rate_bytes_per_s
                       : mbps_to_bytes_per_s(kAutoRateBootstrapMbps)));

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3)
            << "UDP egress pacer stats: mode="
            << (pacer_cfg_.target_mbps > 0.0 ? "fixed" : "auto")
            << ", effective_rate_mbps=" << effective_rate_mbps;
        if (pacer_cfg_.target_mbps <= 0.0) {
            oss << ", auto_rate_mbps="
                << bytes_per_s_to_mbps(auto_rate_bytes_per_s);
        }
        oss << ", enqueue_mbps=" << enqueue_mbps
            << ", send_mbps=" << send_mbps
            << ", queue=" << queue_depth << "/" << pacer_cfg_.queue_packets
            << ", max_queue=" << max_queue_depth
            << ", enqueued_pkts=" << enqueued_packets
            << ", sent_pkts=" << pacer_sent_packets_
            << ", dropped_oldest=" << dropped_oldest;
        if (interval_dropped_oldest > 0) {
            oss << " (+" << interval_dropped_oldest << ")";
        }
        oss << ", dropped_stale=" << pacer_dropped_stale_;
        if (interval_dropped_stale > 0) {
            oss << " (+" << interval_dropped_stale << ")";
        }
        oss << ", send_failed=" << pacer_send_failed_;
        if (interval_send_failed > 0) {
            oss << " (+" << interval_send_failed << ")";
        }
        LOG_G_INFO_M(UdpEgressProfiling) << oss.str();

        pacer_stats_last_log_time_ = now;
        pacer_stats_last_enqueued_bytes_ = enqueued_bytes;
        pacer_stats_last_sent_bytes_ = pacer_sent_bytes_;
        pacer_stats_last_dropped_oldest_ = dropped_oldest;
        pacer_stats_last_dropped_stale_ = pacer_dropped_stale_;
        pacer_stats_last_send_failed_ = pacer_send_failed_;
    }

    void pacer_loop() {
        while (true) {
            PacerPacket packet;
            double rate_bytes_per_s = 0.0;
            {
                std::unique_lock<std::mutex> lock(pacer_mutex_);
                pacer_cv_.wait(lock, [this]() {
                    return !pacer_running_ || !pacer_queue_.empty();
                });
                if (!pacer_running_) {
                    break;
                }
                packet = std::move(pacer_queue_.front());
                pacer_queue_.pop_front();
                rate_bytes_per_s = current_rate_bytes_per_s_locked();
            }

            const auto now = SteadyClock::now();
            if (packet_is_stale(packet, now)) {
                ++pacer_dropped_stale_;
                if (pacer_dropped_stale_ <= 5 || (pacer_dropped_stale_ % 100) == 0) {
                    LOG_G_WARN_M(Config) << "UDP egress pacer dropped stale packet: dropped="
                                 << pacer_dropped_stale_;
                }
                log_pacer_stats(now);
                continue;
            }

            if (next_send_time_ == SteadyClock::time_point{} || now > next_send_time_) {
                next_send_time_ = now;
            }
            const auto send_time = next_send_time_;
            if (rate_bytes_per_s > 0.0) {
                const auto spacing = std::chrono::duration<double>(
                    static_cast<double>(packet.bytes.size()) / rate_bytes_per_s);
                next_send_time_ += std::chrono::duration_cast<SteadyClock::duration>(spacing);
            }

            if (send_time > now && !wait_until_send_time(send_time)) {
                break;
            }

            try {
                send_raw(packet.bytes.data(), packet.bytes.size());
                ++pacer_sent_packets_;
                pacer_sent_bytes_ += packet.bytes.size();
            } catch (const std::exception& e) {
                ++pacer_send_failed_;
                LOG_G_WARN_M(Config) << "UDP egress pacer send failed: " << e.what();
            }
            log_pacer_stats(SteadyClock::now());
        }
    }
};

/**
 * @brief ZeroMQ PUB sender for Backend->Frontend visualization streams.
 *
 * Backend binds a TCP PUB endpoint and frontend scripts connect with SUB.
 * This is intentionally separate from UdpSender because it does not use UDP.
 */
class ZmqPubSender {
protected:
    std::shared_ptr<zmq_transport::SharedPubSocket> pub_;
    std::atomic<uint64_t> _send_drop_count{0};

    void _warn_send_drop(const char* kind, size_t part_count, size_t size_bytes) {
        const uint64_t dropped =
            _send_drop_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (dropped <= 20 || (dropped % 100) == 0) {
            LOG_G_WARN_M(Config) << "ZeroMQ PUB " << kind
                         << " failed; dropping outbound frame: parts=" << part_count
                         << ", bytes=" << size_bytes
                         << ", dropped=" << dropped;
        }
    }

public:
    ZmqPubSender(const std::string& ip, uint16_t port) {
        pub_ = zmq_transport::bind_pub(zmq_transport::make_tcp_endpoint(ip, port));
    }

    virtual ~ZmqPubSender() = default;

    template <typename T>
    bool send_raw(const T* data, size_t size_bytes) {
        if (!pub_->send_single(static_cast<const void*>(data), size_bytes)) {
            _warn_send_drop("single-part send", 1, size_bytes);
            return false;
        }
        return true;
    }

    bool send_frame(const std::vector<zmq_transport::MsgPart>& parts) {
        if (!pub_->send_multipart(parts)) {
            size_t size_bytes = 0;
            for (const auto& part : parts) {
                size_bytes += part.size;
            }
            _warn_send_drop("multipart send", parts.size(), size_bytes);
            return false;
        }
        return true;
    }
};

/**
 * @brief ZeroMQ PUB sender for byte-array debug streams.
 */
class ZmqByteSender : public ZmqPubSender {
public:
    ZmqByteSender(const std::string& ip, uint16_t port) : ZmqPubSender(ip, port) {}

    template <typename T>
    void send(const T* data, size_t size_bytes) {
        send_raw(data, size_bytes);
    }

    template <typename Container>
    void send_container(const Container& data) {
        send(data.data(), data.size() * sizeof(typename Container::value_type));
    }
};

/**
 * @brief Owned eRTM debug frame handed from DSP to the asynchronous sender.
 *
 * Keeping the multipart buffers alive in one movable object lets the producer
 * enqueue without calling the ZeroMQ socket.  The sender thread only builds
 * byte views after it owns the frame.
 */
struct ErtmDebugFrameData {
    std::vector<uint8_t> header;
    AlignedVector uplink_delay;
    AlignedVector downlink_delay;
    std::vector<float> correlation;
    AlignedVector corrected_uplink_delay;
    AlignedVector corrected_downlink_delay;
};

inline std::vector<zmq_transport::MsgPart> make_ertm_debug_msg_parts(
    const ErtmDebugFrameData& frame)
{
    return {
        {frame.header.data(), frame.header.size()},
        {frame.uplink_delay.data(), frame.uplink_delay.size() * sizeof(std::complex<float>)},
        {frame.downlink_delay.data(), frame.downlink_delay.size() * sizeof(std::complex<float>)},
        {frame.correlation.data(), frame.correlation.size() * sizeof(float)},
        {frame.corrected_uplink_delay.data(),
         frame.corrected_uplink_delay.size() * sizeof(std::complex<float>)},
        {frame.corrected_downlink_delay.data(),
         frame.corrected_downlink_delay.size() * sizeof(std::complex<float>)},
    };
}

/**
 * @brief VOFA+ FireWater debug sender.
 *
 * Sends N float channels in little-endian with FireWater tail:
 * [ch0, ch1, ...] + 0x00 0x00 0x80 0x7F
 */
class VofaPlusDebugSender {
public:
    VofaPlusDebugSender(const std::string& ip, uint16_t port, size_t interval_frames = 64)
        : udp_sender_(ip, port),
          interval_frames_(std::max<size_t>(interval_frames, 1)) {}

    void send_channels(const float* channels, size_t channel_count) {
        if (channel_count == 0) {
            return;
        }
        if (channels == nullptr) {
            const uint64_t dropped = ++send_drop_count_;
            LOG_RT_WARN_M(Demod) << "VOFA+ debug frame has a null payload; dropping frame"
                          << ": channels=" << channel_count
                          << ", dropped=" << dropped;
            return;
        }
        if (channel_count > kMaxChannels) {
            const uint64_t dropped = ++send_drop_count_;
            LOG_RT_WARN_M(Demod) << "VOFA+ debug frame has too many channels; dropping frame"
                          << ": channels=" << channel_count
                          << ", max_channels=" << kMaxChannels
                          << ", dropped=" << dropped;
            return;
        }
        if (++frame_counter_ < interval_frames_) {
            return;
        }
        frame_counter_ = 0;

        const size_t payload_bytes = channel_count * sizeof(float);
        std::memcpy(packet_buffer_.data(), channels, payload_bytes);
        std::memcpy(packet_buffer_.data() + payload_bytes, kFireWaterTail.data(), kFireWaterTail.size());
        try {
            udp_sender_.send(packet_buffer_.data(), payload_bytes + kFireWaterTail.size());
        } catch (const std::exception& e) {
            const uint64_t dropped = ++send_drop_count_;
            if (dropped <= 20 || (dropped % 100) == 0) {
                LOG_RT_WARN_M(Demod) << "VOFA+ UDP debug send failed; dropping frame"
                              << ": channels=" << channel_count
                              << ", dropped=" << dropped
                              << ", error=" << e.what();
            }
        }
    }

    template <size_t N>
    void send_channels(const std::array<float, N>& channels) {
        send_channels(channels.data(), N);
    }

    void reset_counter() {
        frame_counter_ = 0;
    }

private:
    static constexpr size_t kMaxChannels = 16;
    static constexpr std::array<uint8_t, 4> kFireWaterTail{{0x00, 0x00, 0x80, 0x7f}};

    UdpSender udp_sender_;
    size_t interval_frames_ = 64;
    size_t frame_counter_ = 0;
    uint64_t send_drop_count_ = 0;
    std::array<uint8_t, kMaxChannels * sizeof(float) + kFireWaterTail.size()> packet_buffer_{};
};

/**
 * @brief Lock-free SPSC ZeroMQ PUB sender for sensing data.
 *
 * Specialized sender for high-throughput sensing data. A single producer thread
 * enqueues payloads into a bounded lock-free ring, and a background consumer
 * thread serializes frames into ZMQ messages asynchronously.
 */
class SensingDataSender : public ZmqPubSender {
public:
    enum class PayloadFormat {
        Dense,
        CompactMask
    };

    struct FrameData {
        AlignedVector data;
        std::shared_ptr<const void> owner;
        const std::complex<float>* external_data = nullptr;
        size_t external_size = 0;
        uint64_t first_symbol_index = 0; // OFDM symbol index before sparse sampling
        PayloadFormat format = PayloadFormat::Dense;
        uint32_t mask_hash = 0;
        std::vector<uint8_t> metadata;
    };

    SensingDataSender(
        const std::string& ip,
        int port,
        bool enabled = true,
        SensingOnWireFormat wire_data_format = SensingOnWireFormat::ComplexFloat32)
        : ZmqPubSender(ip, port),
          _enabled(enabled),
          _wire_data_format(wire_data_format),
          _data_queue(64) {}

    ~SensingDataSender() {
        stop();
    }

    void start() {
        if (!_enabled) return;
        if (_running.load()) return;
        _running.store(true);
        _send_thread = std::thread(&SensingDataSender::run, this);
    }

    void stop() {
        if (!_running.exchange(false)) return;
        if (_send_thread.joinable()) {
            _send_thread.join();
        }
        _data_queue.clear();
    }

    void push_data(const AlignedVector& data) {
        push_data(data, 0);
    }

    void push_data(const AlignedVector& data, uint64_t first_symbol_index) {
        push_data(data, first_symbol_index, {});
    }

    void push_data(
        const AlignedVector& data,
        uint64_t first_symbol_index,
        std::vector<uint8_t> metadata)
    {
        if (!_accepting_frames()) return;

        FrameData frame_data;
        frame_data.data = data;
        frame_data.first_symbol_index = first_symbol_index;
        frame_data.metadata = std::move(metadata);
        _queue_frame(std::move(frame_data));
    }

    void push_data(AlignedVector&& data) {
        push_data(std::move(data), 0);
    }

    void push_data(AlignedVector&& data, uint64_t first_symbol_index) {
        push_data(std::move(data), first_symbol_index, {});
    }

    void push_data(
        AlignedVector&& data,
        uint64_t first_symbol_index,
        std::vector<uint8_t> metadata)
    {
        if (!_accepting_frames()) return;

        FrameData frame_data;
        frame_data.data = std::move(data);
        frame_data.first_symbol_index = first_symbol_index;
        frame_data.metadata = std::move(metadata);
        _queue_frame(std::move(frame_data));
    }

    void push_compact_data(
        const AlignedVector& data,
        uint32_t mask_hash,
        uint64_t frame_start_symbol_index)
    {
        if (!_accepting_frames()) return;

        FrameData frame_data;
        frame_data.data = data;
        frame_data.first_symbol_index = frame_start_symbol_index;
        frame_data.format = PayloadFormat::CompactMask;
        frame_data.mask_hash = mask_hash;
        _queue_frame(std::move(frame_data));
    }

    void push_compact_data(
        AlignedVector&& data,
        uint32_t mask_hash,
        uint64_t frame_start_symbol_index)
    {
        if (!_accepting_frames()) return;

        FrameData frame_data;
        frame_data.data = std::move(data);
        frame_data.first_symbol_index = frame_start_symbol_index;
        frame_data.format = PayloadFormat::CompactMask;
        frame_data.mask_hash = mask_hash;
        _queue_frame(std::move(frame_data));
    }

    void push_external(
        std::shared_ptr<const void> owner,
        const std::complex<float>* data,
        size_t count,
        uint64_t first_symbol_index
    ) {
        push_external(std::move(owner), data, count, first_symbol_index, {});
    }

    void push_external(
        std::shared_ptr<const void> owner,
        const std::complex<float>* data,
        size_t count,
        uint64_t first_symbol_index,
        std::vector<uint8_t> metadata
    ) {
        if (!_accepting_frames()) return;

        FrameData frame_data;
        frame_data.owner = std::move(owner);
        frame_data.external_data = data;
        frame_data.external_size = count;
        frame_data.first_symbol_index = first_symbol_index;
        frame_data.metadata = std::move(metadata);
        _queue_frame(std::move(frame_data));
    }

    void push_compact_external(
        std::shared_ptr<const void> owner,
        const std::complex<float>* data,
        size_t count,
        uint32_t mask_hash,
        uint64_t frame_start_symbol_index
    ) {
        if (!_accepting_frames()) return;

        FrameData frame_data;
        frame_data.owner = std::move(owner);
        frame_data.external_data = data;
        frame_data.external_size = count;
        frame_data.first_symbol_index = frame_start_symbol_index;
        frame_data.format = PayloadFormat::CompactMask;
        frame_data.mask_hash = mask_hash;
        _queue_frame(std::move(frame_data));
    }

private:
    bool _enabled = true;
    SensingOnWireFormat _wire_data_format = SensingOnWireFormat::ComplexFloat32;

    bool _accepting_frames() {
        return _enabled && _running.load(std::memory_order_acquire);
    }

    void _queue_frame(FrameData&& frame_data) {
        if (_data_queue.try_push(std::move(frame_data))) {
            return;
        }
        if (!_running.load(std::memory_order_acquire)) {
            return;
        }
        const uint64_t dropped =
            _queue_full_drop_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (dropped <= 20 || (dropped % 100) == 0) {
            LOG_RT_WARN_M(Sensing) << "Sensing sender queue full; dropping newest frame"
                          << ": queue=" << _data_queue.size() << "/" << _data_queue.capacity()
                          << ", dropped=" << dropped;
        }
    }

    void run() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        uhd::set_thread_priority_safe();

        while (_running.load(std::memory_order_acquire)) {
            FrameData frame_data;

            if (!spsc_wait_pop(_data_queue, frame_data, [this]() {
                    return !_running.load(std::memory_order_acquire);
                })) {
                break;
            }

            _send_frame(frame_data);
        }
    }

    // Emit one frame as a single ZMQ (multipart) message:
    //   part 0 = data payload (may be empty)
    //   part 1 = metadata sidecar (only when present)
    // The previous 60 KB UDP chunking and the !III reassembly header are gone:
    // ZMQ delivers arbitrarily large multipart messages atomically.
    void _send_frame(const FrameData& frame_data) {
        const bool has_inline = !frame_data.data.empty();
        const bool has_external = frame_data.external_data != nullptr && frame_data.external_size > 0;

        std::vector<uint8_t> data_payload;
        if (has_inline || has_external) {
            if (frame_data.format == PayloadFormat::CompactMask) {
                data_payload = _build_compact_payload(frame_data);
            } else {
                data_payload = _build_dense_payload(frame_data);
            }
        }

        const bool has_metadata = !frame_data.metadata.empty();
        if (data_payload.empty() && !has_metadata) {
            return;
        }

        std::vector<zmq_transport::MsgPart> parts;
        parts.push_back({data_payload.data(), data_payload.size()});
        if (has_metadata) {
            parts.push_back({frame_data.metadata.data(), frame_data.metadata.size()});
        }
        send_frame(parts);
    }

    static const std::complex<float>* _frame_data_ptr(const FrameData& fd) {
        if (fd.external_data != nullptr && fd.external_size > 0) {
            return fd.external_data;
        }
        return fd.data.data();
    }

    static size_t _frame_data_size(const FrameData& fd) {
        if (fd.external_data != nullptr && fd.external_size > 0) {
            return fd.external_size;
        }
        return fd.data.size();
    }

    static uint16_t _float32_to_half_bits(float value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));

        const uint32_t sign = (bits >> 16) & 0x8000u;
        uint32_t exponent = (bits >> 23) & 0xFFu;
        uint32_t mantissa = bits & 0x7FFFFFu;

        if (exponent == 0xFFu) {
            if (mantissa != 0) {
                return static_cast<uint16_t>(sign | 0x7E00u);
            }
            return static_cast<uint16_t>(sign | 0x7C00u);
        }

        if (exponent == 0) {
            return static_cast<uint16_t>(sign);
        }

        int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
        if (half_exponent >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        if (half_exponent <= 0) {
            if (half_exponent < -10) {
                return static_cast<uint16_t>(sign);
            }
            mantissa |= 0x800000u;
            const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
            uint32_t rounded = mantissa >> shift;
            const uint32_t round_bit = 1u << (shift - 1);
            if ((mantissa & round_bit) &&
                (((mantissa & (round_bit - 1u)) != 0u) || (rounded & 0x1u))) {
                ++rounded;
            }
            return static_cast<uint16_t>(sign | rounded);
        }

        mantissa += 0x1000u;
        if (mantissa & 0x800000u) {
            mantissa = 0;
            ++half_exponent;
            if (half_exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00u);
            }
        }

        return static_cast<uint16_t>(
            sign |
            (static_cast<uint32_t>(half_exponent) << 10) |
            (mantissa >> 13));
    }

    static void _append_cf16_payload(
        std::vector<uint8_t>& out,
        const std::complex<float>* data_ptr,
        size_t data_size)
    {
        const size_t start = out.size();
        out.resize(start + data_size * sensing_on_wire_complex_bytes(SensingOnWireFormat::ComplexFloat16));
        auto* dst = out.data() + static_cast<std::ptrdiff_t>(start);
        for (size_t i = 0; i < data_size; ++i) {
            const uint16_t re_bits = _float32_to_half_bits(data_ptr[i].real());
            const uint16_t im_bits = _float32_to_half_bits(data_ptr[i].imag());
            std::memcpy(dst + i * 4, &re_bits, sizeof(re_bits));
            std::memcpy(dst + i * 4 + sizeof(re_bits), &im_bits, sizeof(im_bits));
        }
    }

    void _append_payload_bytes(
        std::vector<uint8_t>& out,
        const std::complex<float>* data_ptr,
        size_t data_size) const
    {
        if (data_ptr == nullptr || data_size == 0) {
            return;
        }
        if (_wire_data_format == SensingOnWireFormat::ComplexFloat16) {
            _append_cf16_payload(out, data_ptr, data_size);
            return;
        }
        const auto* payload_bytes = reinterpret_cast<const uint8_t*>(data_ptr);
        out.insert(
            out.end(),
            payload_bytes,
            payload_bytes + data_size * sizeof(std::complex<float>));
    }

    // Dense payload: raw on-wire complex samples, no header (the viewer reads
    // the geometry from the params packet).
    std::vector<uint8_t> _build_dense_payload(const FrameData& frame_data) const {
        const std::complex<float>* data_ptr = _frame_data_ptr(frame_data);
        const size_t data_size = _frame_data_size(frame_data);
        std::vector<uint8_t> encoded_payload;
        if (data_ptr == nullptr || data_size == 0) {
            return encoded_payload;
        }
        encoded_payload.reserve(data_size * sensing_on_wire_complex_bytes(_wire_data_format));
        _append_payload_bytes(encoded_payload, data_ptr, data_size);
        return encoded_payload;
    }

    // Compact payload: CompactSensingFrameHeader followed by on-wire samples.
    std::vector<uint8_t> _build_compact_payload(const FrameData& frame_data) const {
        const std::complex<float>* data_ptr = _frame_data_ptr(frame_data);
        const size_t data_size = _frame_data_size(frame_data);
        std::vector<uint8_t> encoded_payload;
        if (data_ptr == nullptr || data_size == 0) {
            return encoded_payload;
        }
        encoded_payload.reserve(
            sizeof(CompactSensingFrameHeader) +
            data_size * sensing_on_wire_complex_bytes(_wire_data_format));

        CompactSensingFrameHeader compact_header{};
        compact_header.magic_version = htonl(kCompactSensingMagicVersion);
        compact_header.mask_hash = htonl(frame_data.mask_hash);
        compact_header.re_count = htonl(static_cast<uint32_t>(data_size));
        compact_header.frame_start_symbol_index =
            host_to_network_u64(frame_data.first_symbol_index);

        const auto* compact_header_bytes = reinterpret_cast<const uint8_t*>(&compact_header);
        encoded_payload.insert(
            encoded_payload.end(),
            compact_header_bytes,
            compact_header_bytes + sizeof(CompactSensingFrameHeader));
        _append_payload_bytes(encoded_payload, data_ptr, data_size);
        return encoded_payload;
    }

    std::atomic<bool> _running{false};
    std::atomic<uint64_t> _queue_full_drop_count{0};
    SPSCRingBuffer<FrameData> _data_queue;
    std::thread _send_thread;
};

/**
 * @brief Aggregates per-channel sensing frames and emits a single ZeroMQ stream.
 *
 * Each channel owns one SPSC queue into this sender. The background thread
 * waits for matching `first_symbol_index` values across all enabled channels,
 * serializes the per-channel payloads in logical-channel order, and sends a
 * single aggregated ZMQ message stream to the viewer.
 */
class AggregatedSensingDataSender : public ZmqPubSender {
public:
    using FrameData = SensingDataSender::FrameData;

    AggregatedSensingDataSender(
        const std::string& ip,
        int port,
        std::vector<uint32_t> channel_ids,
        bool enabled = true,
        size_t per_channel_queue_size = 8,
        SensingOnWireFormat wire_data_format = SensingOnWireFormat::ComplexFloat32)
        : ZmqPubSender(ip, static_cast<uint16_t>(port)),
          _enabled(enabled && !channel_ids.empty()),
          _wire_data_format(wire_data_format),
          _channel_ids(std::move(channel_ids)),
          _channel_queues(_channel_ids.size())
    {
        uint32_t max_channel_id = 0;
        for (uint32_t channel_id : _channel_ids) {
            max_channel_id = std::max(max_channel_id, channel_id);
            if (channel_id < 32) {
                _channel_mask |= (1u << channel_id);
            }
        }
        _channel_index_by_id.assign(static_cast<size_t>(max_channel_id) + 1, -1);
        for (size_t i = 0; i < _channel_ids.size(); ++i) {
            _channel_queues[i].reset(per_channel_queue_size, []() {
                return FrameData{};
            });
            _channel_index_by_id[_channel_ids[i]] = static_cast<int32_t>(i);
        }
    }

    ~AggregatedSensingDataSender() {
        stop();
    }

    void start() {
        if (!_enabled) return;
        if (_running.exchange(true, std::memory_order_acq_rel)) return;
        _send_thread = std::thread(&AggregatedSensingDataSender::run, this);
    }

    void stop() {
        if (!_running.exchange(false, std::memory_order_acq_rel)) return;
        if (_send_thread.joinable()) {
            _send_thread.join();
        }
        for (auto& queue : _channel_queues) {
            queue.clear();
        }
    }

    uint32_t channel_count() const {
        return static_cast<uint32_t>(_channel_ids.size());
    }

    uint32_t channel_mask() const {
        return _channel_mask;
    }

    bool has_channel(uint32_t channel_id) const {
        return _channel_index(channel_id) >= 0;
    }

    void push_data(uint32_t channel_id, const AlignedVector& data, uint64_t first_symbol_index) {
        push_data(channel_id, data, first_symbol_index, {});
    }

    void push_data(
        uint32_t channel_id,
        const AlignedVector& data,
        uint64_t first_symbol_index,
        std::vector<uint8_t> metadata)
    {
        if (!_accepting_frames()) return;
        FrameData frame_data;
        frame_data.data = data;
        frame_data.first_symbol_index = first_symbol_index;
        frame_data.metadata = std::move(metadata);
        _queue_frame(channel_id, std::move(frame_data));
    }

    void push_data(uint32_t channel_id, AlignedVector&& data, uint64_t first_symbol_index) {
        push_data(channel_id, std::move(data), first_symbol_index, {});
    }

    void push_data(
        uint32_t channel_id,
        AlignedVector&& data,
        uint64_t first_symbol_index,
        std::vector<uint8_t> metadata)
    {
        if (!_accepting_frames()) return;
        FrameData frame_data;
        frame_data.data = std::move(data);
        frame_data.first_symbol_index = first_symbol_index;
        frame_data.metadata = std::move(metadata);
        _queue_frame(channel_id, std::move(frame_data));
    }

    void push_compact_data(
        uint32_t channel_id,
        const AlignedVector& data,
        uint32_t mask_hash,
        uint64_t frame_start_symbol_index)
    {
        if (!_accepting_frames()) return;
        FrameData frame_data;
        frame_data.data = data;
        frame_data.first_symbol_index = frame_start_symbol_index;
        frame_data.format = SensingDataSender::PayloadFormat::CompactMask;
        frame_data.mask_hash = mask_hash;
        _queue_frame(channel_id, std::move(frame_data));
    }

    void push_compact_data(
        uint32_t channel_id,
        AlignedVector&& data,
        uint32_t mask_hash,
        uint64_t frame_start_symbol_index)
    {
        if (!_accepting_frames()) return;
        FrameData frame_data;
        frame_data.data = std::move(data);
        frame_data.first_symbol_index = frame_start_symbol_index;
        frame_data.format = SensingDataSender::PayloadFormat::CompactMask;
        frame_data.mask_hash = mask_hash;
        _queue_frame(channel_id, std::move(frame_data));
    }

    void push_external(
        uint32_t channel_id,
        std::shared_ptr<const void> owner,
        const std::complex<float>* data,
        size_t count,
        uint64_t first_symbol_index)
    {
        push_external(channel_id, std::move(owner), data, count, first_symbol_index, {});
    }

    void push_external(
        uint32_t channel_id,
        std::shared_ptr<const void> owner,
        const std::complex<float>* data,
        size_t count,
        uint64_t first_symbol_index,
        std::vector<uint8_t> metadata)
    {
        if (!_accepting_frames()) return;
        FrameData frame_data;
        frame_data.owner = std::move(owner);
        frame_data.external_data = data;
        frame_data.external_size = count;
        frame_data.first_symbol_index = first_symbol_index;
        frame_data.metadata = std::move(metadata);
        _queue_frame(channel_id, std::move(frame_data));
    }

    void push_compact_external(
        uint32_t channel_id,
        std::shared_ptr<const void> owner,
        const std::complex<float>* data,
        size_t count,
        uint32_t mask_hash,
        uint64_t frame_start_symbol_index)
    {
        if (!_accepting_frames()) return;
        FrameData frame_data;
        frame_data.owner = std::move(owner);
        frame_data.external_data = data;
        frame_data.external_size = count;
        frame_data.first_symbol_index = frame_start_symbol_index;
        frame_data.format = SensingDataSender::PayloadFormat::CompactMask;
        frame_data.mask_hash = mask_hash;
        _queue_frame(channel_id, std::move(frame_data));
    }

private:
    struct PendingAggregateFrame {
        std::vector<FrameData> frames;
        std::vector<uint8_t> present;
        size_t received_channels = 0;
        PendingAggregateFrame() = default;
        explicit PendingAggregateFrame(size_t channel_count)
            : frames(channel_count),
              present(channel_count, 0) {}
    };

    static constexpr size_t kMaxPendingAggregateFrames = 4;

    bool _accepting_frames() {
        return _enabled && _running.load(std::memory_order_acquire);
    }

    int32_t _channel_index(uint32_t channel_id) const {
        if (channel_id >= _channel_index_by_id.size()) {
            return -1;
        }
        return _channel_index_by_id[channel_id];
    }

    void _queue_frame(uint32_t channel_id, FrameData&& frame_data) {
        const int32_t channel_index = _channel_index(channel_id);
        if (channel_index < 0) {
            LOG_G_WARN_M(Sensing) << "Ignoring aggregated sensing frame for unknown channel id=" << channel_id;
            return;
        }
        auto& queue = _channel_queues[static_cast<size_t>(channel_index)];
        if (queue.try_push(std::move(frame_data))) {
            return;
        }
        if (!_running.load(std::memory_order_acquire)) {
            return;
        }
        const uint64_t dropped =
            _queue_full_drop_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (dropped <= 20 || (dropped % 100) == 0) {
            LOG_RT_WARN_M(SensingAggregate)
                << "Aggregated sensing channel queue full; dropping newest frame"
                << ": channel_id=" << channel_id
                << ", queue=" << queue.size() << "/" << queue.capacity()
                << ", dropped=" << dropped;
        }
    }

    static size_t _payload_complex_count(const FrameData& frame_data) {
        if (frame_data.external_data != nullptr && frame_data.external_size > 0) {
            return frame_data.external_size;
        }
        return frame_data.data.size();
    }

    static const std::complex<float>* _payload_data_ptr(const FrameData& frame_data) {
        if (frame_data.external_data != nullptr && frame_data.external_size > 0) {
            return frame_data.external_data;
        }
        return frame_data.data.data();
    }

    size_t _wire_payload_bytes(const FrameData& frame_data) const {
        const size_t complex_bytes =
            _payload_complex_count(frame_data) * sensing_on_wire_complex_bytes(_wire_data_format);
        if (frame_data.format == SensingDataSender::PayloadFormat::CompactMask) {
            return sizeof(CompactSensingFrameHeader) + complex_bytes;
        }
        return complex_bytes;
    }

    void _append_wire_payload(std::vector<uint8_t>& packet, const FrameData& frame_data) const {
        const auto* payload_data = _payload_data_ptr(frame_data);
        const size_t payload_complex_count = _payload_complex_count(frame_data);
        if (frame_data.format == SensingDataSender::PayloadFormat::CompactMask) {
            CompactSensingFrameHeader compact_header{};
            compact_header.magic_version = htonl(kCompactSensingMagicVersion);
            compact_header.mask_hash = htonl(frame_data.mask_hash);
            compact_header.re_count = htonl(static_cast<uint32_t>(payload_complex_count));
            compact_header.frame_start_symbol_index = host_to_network_u64(frame_data.first_symbol_index);
            const auto* header_bytes = reinterpret_cast<const uint8_t*>(&compact_header);
            packet.insert(packet.end(), header_bytes, header_bytes + sizeof(compact_header));
        }
        if (_wire_data_format == SensingOnWireFormat::ComplexFloat16) {
            const size_t start = packet.size();
            packet.resize(start + payload_complex_count * sensing_on_wire_complex_bytes(_wire_data_format));
            auto* dst = packet.data() + static_cast<std::ptrdiff_t>(start);
            for (size_t i = 0; i < payload_complex_count; ++i) {
                const uint16_t re_bits = float32_to_half_ieee_bits(payload_data[i].real());
                const uint16_t im_bits = float32_to_half_ieee_bits(payload_data[i].imag());
                std::memcpy(dst + i * 4, &re_bits, sizeof(re_bits));
                std::memcpy(dst + i * 4 + sizeof(re_bits), &im_bits, sizeof(im_bits));
            }
            return;
        }
        const auto* raw_bytes = reinterpret_cast<const uint8_t*>(payload_data);
        packet.insert(
            packet.end(),
            raw_bytes,
            raw_bytes + payload_complex_count * sizeof(std::complex<float>));
    }

    void _drop_oldest_pending_frame() {
        if (_pending_frames.empty()) {
            return;
        }
        const auto it = _pending_frames.begin();
        LOG_RT_WARN_HZ_M(SensingAggregate, 5) << "[Sensing Aggregate] dropping incomplete frame start_symbol="
                          << it->first << " after collecting "
                          << it->second.received_channels << "/" << _channel_ids.size()
                          << " channels";
        _pending_frames.erase(it);
    }

    void _send_pending_frame(uint64_t frame_start_symbol_index, const PendingAggregateFrame& pending) {
        if (pending.frames.empty()) {
            return;
        }

        const size_t channel_count = _channel_ids.size();
        const size_t channel_payload_bytes = _wire_payload_bytes(pending.frames.front());
        std::vector<uint8_t> payload;
        payload.reserve(sizeof(AggregatedSensingFrameHeader) + channel_count * channel_payload_bytes);

        AggregatedSensingFrameHeader aggregate_header{};
        aggregate_header.magic_version = htonl(kAggregatedSensingMagicVersion);
        aggregate_header.channel_count = htonl(static_cast<uint32_t>(channel_count));
        aggregate_header.channel_payload_bytes = htonl(static_cast<uint32_t>(channel_payload_bytes));
        aggregate_header.channel_mask = htonl(_channel_mask);
        aggregate_header.frame_start_symbol_index = host_to_network_u64(frame_start_symbol_index);
        const auto* header_bytes = reinterpret_cast<const uint8_t*>(&aggregate_header);
        payload.insert(payload.end(), header_bytes, header_bytes + sizeof(aggregate_header));

        for (size_t i = 0; i < channel_count; ++i) {
            const size_t payload_bytes = _wire_payload_bytes(pending.frames[i]);
            if (payload_bytes != channel_payload_bytes) {
                LOG_G_WARN_M(SensingAggregate) << "[Sensing Aggregate] channel payload size mismatch for frame "
                             << frame_start_symbol_index << ": expected " << channel_payload_bytes
                             << " bytes, got " << payload_bytes << " bytes on channel "
                             << _channel_ids[i];
                return;
            }
            _append_wire_payload(payload, pending.frames[i]);
        }

        // Build the optional metadata sidecar (only when every channel
        // provided metadata). It travels as part 1 of the same multipart
        // message; the previous 0x80000000 chunk-header flag is gone.
        std::vector<uint8_t> metadata_payload;
        bool have_all_metadata = true;
        bool any_metadata = false;
        for (size_t i = 0; i < channel_count; ++i) {
            const bool has_metadata = !pending.frames[i].metadata.empty();
            any_metadata = any_metadata || has_metadata;
            have_all_metadata = have_all_metadata && has_metadata;
        }
        if (any_metadata && !have_all_metadata) {
            LOG_G_WARN_M(SensingAggregate) << "[Sensing Aggregate] dropping metadata sidecar for frame "
                         << frame_start_symbol_index
                         << " because not all channels provided metadata";
        } else if (have_all_metadata) {
            metadata_payload.reserve(sizeof(AggregatedSensingMetadataHeader));
            AggregatedSensingMetadataHeader metadata_header{};
            std::memcpy(
                metadata_header.magic,
                kAggregatedSensingMetadataMagic,
                sizeof(metadata_header.magic));
            metadata_header.channel_count = static_cast<uint32_t>(channel_count);
            metadata_header.channel_mask = _channel_mask;
            metadata_header.reserved = 0;
            metadata_header.frame_start_symbol_index = frame_start_symbol_index;
            const auto* metadata_header_bytes =
                reinterpret_cast<const uint8_t*>(&metadata_header);
            metadata_payload.insert(
                metadata_payload.end(),
                metadata_header_bytes,
                metadata_header_bytes + sizeof(metadata_header));

            for (size_t i = 0; i < channel_count; ++i) {
                metadata_payload.insert(
                    metadata_payload.end(),
                    pending.frames[i].metadata.begin(),
                    pending.frames[i].metadata.end());
            }
        }

        std::vector<zmq_transport::MsgPart> parts;
        parts.push_back({payload.data(), payload.size()});
        if (!metadata_payload.empty()) {
            parts.push_back({metadata_payload.data(), metadata_payload.size()});
        }
        send_frame(parts);
    }

    void run() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        uhd::set_thread_priority_safe();
        SPSCBackoff backoff;

        while (_running.load(std::memory_order_acquire)) {
            bool did_work = false;
            for (size_t channel_index = 0; channel_index < _channel_queues.size(); ++channel_index) {
                FrameData frame_data;
                if (!_channel_queues[channel_index].try_pop(frame_data)) {
                    continue;
                }

                did_work = true;
                const uint64_t frame_start_symbol_index = frame_data.first_symbol_index;
                auto [it, inserted] = _pending_frames.emplace(
                    frame_start_symbol_index,
                    PendingAggregateFrame(_channel_queues.size()));
                auto& pending = it->second;
                if (!pending.present[channel_index]) {
                    pending.present[channel_index] = 1;
                    pending.received_channels++;
                } else {
                    const uint64_t dropped =
                        _duplicate_channel_drop_count.fetch_add(
                            1, std::memory_order_relaxed) + 1;
                    LOG_G_WARN_M(SensingAggregate)
                        << "Aggregated sensing received a duplicate channel frame; replacing older data"
                        << ": channel_id=" << _channel_ids[channel_index]
                        << ", frame_start_symbol_index=" << frame_start_symbol_index
                        << ", dropped=" << dropped;
                }
                pending.frames[channel_index] = std::move(frame_data);

                if (pending.received_channels == _channel_queues.size()) {
                    _send_pending_frame(frame_start_symbol_index, pending);
                    _pending_frames.erase(it);
                }
            }

            while (_pending_frames.size() > kMaxPendingAggregateFrames) {
                _drop_oldest_pending_frame();
            }

            if (!did_work) {
                backoff.pause();
            } else {
                backoff.reset();
            }
        }

        _pending_frames.clear();
    }

    bool _enabled = true;
    SensingOnWireFormat _wire_data_format = SensingOnWireFormat::ComplexFloat32;
    std::vector<uint32_t> _channel_ids;
    std::vector<int32_t> _channel_index_by_id;
    uint32_t _channel_mask = 0;
    std::vector<SPSCRingBuffer<FrameData>> _channel_queues;
    std::map<uint64_t, PendingAggregateFrame> _pending_frames;
    std::atomic<bool> _running{false};
    std::atomic<uint64_t> _queue_full_drop_count{0};
    std::atomic<uint64_t> _duplicate_channel_drop_count{0};
    std::thread _send_thread;
};

class SensingOutputDispatcher {
public:
    SensingOutputDispatcher(
        const std::string& ip,
        int port,
        bool enabled,
        uint32_t channel_id = 0,
        std::shared_ptr<AggregatedSensingDataSender> aggregated_sender = nullptr,
        SensingOnWireFormat wire_data_format = SensingOnWireFormat::ComplexFloat32)
        : _enabled(enabled),
          _channel_id(channel_id),
          _aggregated_sender(std::move(aggregated_sender))
    {
        if (_enabled && _aggregated_sender == nullptr) {
            _direct_sender = std::make_unique<SensingDataSender>(ip, port, enabled, wire_data_format);
        }
    }

    void start() {
        if (_direct_sender) {
            _direct_sender->start();
        }
    }

    void stop() {
        if (_direct_sender) {
            _direct_sender->stop();
        }
    }

    void push_data(const AlignedVector& data, uint64_t first_symbol_index) {
        push_data(data, first_symbol_index, {});
    }

    void push_data(
        const AlignedVector& data,
        uint64_t first_symbol_index,
        std::vector<uint8_t> metadata)
    {
        if (!_enabled) return;
        if (_aggregated_sender) {
            _aggregated_sender->push_data(
                _channel_id,
                data,
                first_symbol_index,
                std::move(metadata));
            return;
        }
        _direct_sender->push_data(data, first_symbol_index, std::move(metadata));
    }

    void push_data(AlignedVector&& data, uint64_t first_symbol_index) {
        push_data(std::move(data), first_symbol_index, {});
    }

    void push_data(
        AlignedVector&& data,
        uint64_t first_symbol_index,
        std::vector<uint8_t> metadata)
    {
        if (!_enabled) return;
        if (_aggregated_sender) {
            _aggregated_sender->push_data(
                _channel_id,
                std::move(data),
                first_symbol_index,
                std::move(metadata));
            return;
        }
        _direct_sender->push_data(std::move(data), first_symbol_index, std::move(metadata));
    }

    void push_compact_data(
        const AlignedVector& data,
        uint32_t mask_hash,
        uint64_t frame_start_symbol_index)
    {
        if (!_enabled) return;
        if (_aggregated_sender) {
            _aggregated_sender->push_compact_data(_channel_id, data, mask_hash, frame_start_symbol_index);
            return;
        }
        _direct_sender->push_compact_data(data, mask_hash, frame_start_symbol_index);
    }

    void push_compact_data(
        AlignedVector&& data,
        uint32_t mask_hash,
        uint64_t frame_start_symbol_index)
    {
        if (!_enabled) return;
        if (_aggregated_sender) {
            _aggregated_sender->push_compact_data(
                _channel_id,
                std::move(data),
                mask_hash,
                frame_start_symbol_index);
            return;
        }
        _direct_sender->push_compact_data(std::move(data), mask_hash, frame_start_symbol_index);
    }

    void push_external(
        std::shared_ptr<const void> owner,
        const std::complex<float>* data,
        size_t count,
        uint64_t first_symbol_index)
    {
        push_external(std::move(owner), data, count, first_symbol_index, {});
    }

    void push_external(
        std::shared_ptr<const void> owner,
        const std::complex<float>* data,
        size_t count,
        uint64_t first_symbol_index,
        std::vector<uint8_t> metadata)
    {
        if (!_enabled) return;
        if (_aggregated_sender) {
            _aggregated_sender->push_external(
                _channel_id,
                std::move(owner),
                data,
                count,
                first_symbol_index,
                std::move(metadata));
            return;
        }
        _direct_sender->push_external(
            std::move(owner),
            data,
            count,
            first_symbol_index,
            std::move(metadata));
    }

    void push_compact_external(
        std::shared_ptr<const void> owner,
        const std::complex<float>* data,
        size_t count,
        uint32_t mask_hash,
        uint64_t frame_start_symbol_index)
    {
        if (!_enabled) return;
        if (_aggregated_sender) {
            _aggregated_sender->push_compact_external(
                _channel_id,
                std::move(owner),
                data,
                count,
                mask_hash,
                frame_start_symbol_index);
            return;
        }
        _direct_sender->push_compact_external(
            std::move(owner),
            data,
            count,
            mask_hash,
            frame_start_symbol_index);
    }

private:
    bool _enabled = true;
    uint32_t _channel_id = 0;
    std::shared_ptr<AggregatedSensingDataSender> _aggregated_sender;
    std::unique_ptr<SensingDataSender> _direct_sender;
};

/**
 * @brief Generic Asynchronous Data Sender Manager.
 * 
 * A template class that manages a queue of data packages and sends them at a fixed interval
 * using a background thread. Useful for sending periodic status updates or telemetry.
 */
template <typename DataType, typename Allocator = std::allocator<DataType>>
class DataSender {
public:
    using DataPackage = std::vector<DataType, Allocator>;
    using SendFunction = std::function<void(const DataPackage&)>;
    enum class DeliveryMode {
        QueuedBlocking,
        LatestOnly
    };

    DataSender(
        size_t queue_size,
        SendFunction send_func,
        std::chrono::milliseconds interval,
        DeliveryMode mode = DeliveryMode::QueuedBlocking,
        std::string name = "DataSender"
    )
        : queue_(queue_size),
          send_func_(std::move(send_func)),
          send_interval_(interval),
          delivery_mode_(mode),
          name_(std::move(name)),
          running_(false) {}

    ~DataSender() {
        stop();
    }

    void add_data(DataPackage&& data) {
        if (!running_.load(std::memory_order_acquire)) return;
        if (delivery_mode_ == DeliveryMode::LatestOnly) {
            // Display/telemetry path: never block the producer. If the queue is
            // full, drop the newest update and let the sender drain to the most
            // recent queued snapshot on its next scheduled wake-up.
            if (!queue_.try_push(std::move(data))) {
                if (!running_.load(std::memory_order_acquire)) {
                    return;
                }
                const uint64_t dropped =
                    queue_full_drop_count_.fetch_add(1, std::memory_order_relaxed) + 1;
                if (dropped <= 20 || (dropped % 100) == 0) {
                    LOG_RT_DEBUG_M(Config) << name_
                                  << " queue full; dropping newest outbound update"
                                  << ": queue=" << queue_.size() << "/" << queue_.capacity()
                                  << ", dropped=" << dropped;
                }
            }
            return;
        }
        spsc_wait_push(queue_, std::move(data), [this]() {
            return !running_.load(std::memory_order_acquire);
        });
    }

    void start() {
        if (running_) return;
        running_.store(true);
        thread_ = std::thread(&DataSender::sender_thread, this);
    }

    void stop() {
        if (!running_) return;
        running_.store(false);
        if (thread_.joinable()) thread_.join();
        queue_.clear();
    }

private:
    void sender_thread() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        auto next_send = std::chrono::steady_clock::now();

        while (running_.load()) {
            if (delivery_mode_ == DeliveryMode::LatestOnly) {
                std::this_thread::sleep_until(next_send);
                next_send += send_interval_;

                DataPackage latest;
                if (!queue_.try_pop(latest)) {
                    continue;
                }

                DataPackage newer;
                size_t superseded = 0;
                while (queue_.try_pop(newer)) {
                    latest = std::move(newer);
                    ++superseded;
                }
                if (superseded > 0) {
                    const uint64_t dropped =
                        superseded_drop_count_.fetch_add(
                            superseded, std::memory_order_relaxed) + superseded;
                    if (dropped <= 20 || (dropped % 100) < superseded) {
                        LOG_G_DEBUG_M(Config) << name_
                                     << " superseded queued updates with the latest frame"
                                     << ": dropped=" << dropped;
                    }
                }

                try {
                    send_func_(latest);
                } catch (const std::exception& e) {
                    LOG_G_WARN_M(Config) << name_
                                 << " send failed; dropping outbound update: " << e.what();
                }
                continue;
            }

            DataPackage data;

            if (!spsc_wait_pop(queue_, data, [this]() {
                    return !running_.load(std::memory_order_acquire);
                })) {
                break;
            }

            // Send data if available
            try {
                send_func_(data);
            } catch (const std::exception& e) {
                LOG_G_WARN_M(Config) << name_
                             << " send failed; dropping outbound update: " << e.what();
            }

            // Update next send time
            next_send += send_interval_;
            const auto now = std::chrono::steady_clock::now();
            if (next_send < now) {
                next_send = now;
            }
            std::this_thread::sleep_until(next_send);
        }
    }

    SPSCRingBuffer<DataPackage> queue_;
    SendFunction send_func_;
    std::chrono::milliseconds send_interval_;
    DeliveryMode delivery_mode_;
    std::string name_;
    std::thread thread_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> queue_full_drop_count_{0};
    std::atomic<uint64_t> superseded_drop_count_{0};
};

/**
 * @brief UDP-based Control Command Handler.
 * 
 * Listens for UDP control commands on a dedicated thread and executes registered callbacks.
 * Supports a custom command protocol with a header, command ID, and integer parameter.
 * Used for runtime configuration changes (e.g., gain, frequency, alignment).
 */
// Bidirectional control/params channel. Migrated from raw UDP to a ZeroMQ
// ROUTER socket: the backend binds ROUTER, each frontend connects a DEALER.
// A peer is identified by its DEALER routing id (ControlPeer) instead of a
// sockaddr_in. The ROUTER socket is owned by the single poll thread; replies
// posted from other threads are queued and flushed by that thread.
class ControlCommandHandler {
public:
    using ControlPeer = zmq_transport::PeerId;
    using Callback = std::function<void(int32_t value)>;
    using RequestCallback = std::function<void(int32_t value, const ControlPeer& peer)>;

    // Command structure definition
    #pragma pack(push, 1)
    struct ControlCommand {
        char header[4]; // "CMD "
        char command[4]; // Command ID
        int32_t value;  // Parameter value
    };
    #pragma pack(pop)

    static_assert(sizeof(ControlCommand) == 12, "ControlCommand size mismatch");

    ControlCommandHandler(int port)
        : ControlCommandHandler("0.0.0.0", port) {}

    ControlCommandHandler(const std::string& bind_ip, int port) : _port(port), _running(false) {
        _router = std::make_unique<zmq_transport::ControlRouter>(
            zmq_transport::make_tcp_endpoint(bind_ip, _port));
    }

    ~ControlCommandHandler() {
        stop();
    }

    void start() {
        if (_running) return;
        _running = true;
        _thread = std::thread(&ControlCommandHandler::_run, this);
    }

    void stop() {
        if (!_running) return;
        _running = false;
        if (_thread.joinable()) {
            _thread.join();
        }
    }

    // Register command handler
    void register_command(const std::string& command, Callback callback) {
        std::lock_guard<std::mutex> lock(_mutex);
        _handlers[command] = std::move(callback);
    }

    void register_request(const std::string& command, Callback callback) {
        std::lock_guard<std::mutex> lock(_mutex);
        _request_handlers[command] =
            [callback = std::move(callback)](int32_t value, const ControlPeer&) {
                callback(value);
            };
    }

    void register_request(const std::string& command, RequestCallback callback) {
        std::lock_guard<std::mutex> lock(_mutex);
        _request_handlers[command] = std::move(callback);
    }

    // Send a readiness/heartbeat ("RDY ") packet. With ZMQ the TCP link is
    // persistent (no NAT mapping to keep alive), but the RDY signal still tells
    // connected viewers the backend is up.
    void send_heartbeat() {
        broadcast_heartbeat();
    }

    void broadcast_heartbeat() {
        const ControlCommand hb = _make_heartbeat();
        _broadcast(&hb, sizeof(hb));
    }

    bool send_heartbeat_to_last_peer() {
        ControlPeer peer;
        if (!_get_last_control_peer(peer)) {
            return false;
        }
        const ControlCommand hb = _make_heartbeat();
        _send_to(peer, &hb, sizeof(hb));
        return true;
    }

    void send_sensing_viewer_params(
        const std::string& /*dest_ip*/,
        int /*dest_port*/,
        const SensingViewerParamsPacket& packet)
    {
        _broadcast(&packet, sizeof(packet));
    }

    void send_sensing_viewer_params(
        const ControlPeer& peer,
        const SensingViewerParamsPacket& packet)
    {
        _send_to(peer, &packet, sizeof(packet));
    }

    bool send_sensing_viewer_params_to_last_peer(const SensingViewerParamsPacket& packet) {
        ControlPeer peer;
        if (!_get_last_control_peer(peer)) {
            return false;
        }
        send_sensing_viewer_params(peer, packet);
        return true;
    }

    void send_control_status(
        const ControlPeer& peer,
        const std::string& command,
        int32_t value)
    {
        if (command.size() != 4) {
            LOG_G_WARN_M(Config) << "Control status command id must be exactly 4 bytes";
            return;
        }
        ControlCommand reply;
        std::memcpy(reply.header, "CTRL", 4);
        std::memcpy(reply.command, command.data(), 4);
        reply.value = htonl(value);
        _send_to(peer, &reply, sizeof(reply));
    }

private:
    bool _get_last_control_peer(ControlPeer& peer) const {
        std::lock_guard<std::mutex> lock(_mutex);
        const auto now = std::chrono::steady_clock::now();
        _prune_peers_locked(now);
        if (!_has_last_control_peer) {
            return false;
        }
        peer = _last_control_peer;
        return true;
    }

    ControlCommand _make_heartbeat() const {
        ControlCommand hb;
        std::memcpy(hb.header, "CTRL", 4);
        std::memcpy(hb.command, "RDY ", 4);
        hb.value = 0;
        return hb;
    }

    // Queue an outbound message to every recently active control peer.
    void _broadcast(const void* data, size_t size) {
        std::vector<ControlPeer> peers;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            const auto now = std::chrono::steady_clock::now();
            _prune_peers_locked(now);
            peers.reserve(_peers.size());
            for (const auto& item : _peers) {
                peers.push_back(item.first);
            }
        }
        for (const auto& peer : peers) {
            if (!_router->post_send(peer, data, size)) {
                const uint64_t dropped =
                    _control_outbound_drop_count.fetch_add(1, std::memory_order_relaxed) + 1;
                LOG_G_WARN_M(Config) << "Control outbound queue evicted or rejected a reply"
                             << ": dropped=" << dropped;
            }
        }
    }

    void _send_to(const ControlPeer& peer, const void* data, size_t size) {
        if (!_router->post_send(peer, data, size)) {
            const uint64_t dropped =
                _control_outbound_drop_count.fetch_add(1, std::memory_order_relaxed) + 1;
            LOG_G_WARN_M(Config) << "Control outbound queue evicted or rejected a reply"
                         << ": dropped=" << dropped;
        }
    }

    void _run() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        ControlPeer identity;
        std::vector<uint8_t> payload;

        while (_running) {
            if (_router->recv(identity, payload)) {
                _dispatch(identity, payload);
            }
            // Flush replies/heartbeats queued by other threads (the ROUTER
            // socket is only ever touched on this thread).
            const size_t failed = _router->flush();
            if (_running.load(std::memory_order_acquire) && failed > 0) {
                const uint64_t dropped =
                    _control_send_drop_count.fetch_add(failed, std::memory_order_relaxed) + failed;
                LOG_G_WARN_M(Config) << "Control ROUTER failed to deliver " << failed
                             << " queued replies: dropped=" << dropped;
            }
        }
        _router->flush();
    }

    void _dispatch(const ControlPeer& identity, const std::vector<uint8_t>& payload) {
        if (payload.size() != sizeof(ControlCommand)) {
            LOG_G_WARN_M(Config) << "Received malformed control command (" << payload.size() << " bytes)";
            return;
        }
        ControlCommand cmd;
        std::memcpy(&cmd, payload.data(), sizeof(cmd));

        const std::string header_str(cmd.header, 4);
        const int32_t value = ntohl(cmd.value);  // Network to host byte order
        const std::string command_str(cmd.command, 4);

        {
            std::lock_guard<std::mutex> lock(_mutex);
            _record_peer_locked(identity, std::chrono::steady_clock::now());
        }

        if (header_str == "CMD ") {
            Callback callback;
            {
                std::lock_guard<std::mutex> lock(_mutex);
                auto it = _handlers.find(command_str);
                if (it != _handlers.end()) {
                    callback = it->second;
                }
            }
            if (!callback) {
                LOG_G_WARN_M(Config) << "Unknown command: " << command_str;
                return;
            }
            try {
                callback(value);
            } catch (const std::exception& e) {
                LOG_G_WARN_M(Config) << "Error processing command '" << command_str
                             << "': " << e.what();
            }
        } else if (header_str == "REQ ") {
            RequestCallback callback;
            {
                std::lock_guard<std::mutex> lock(_mutex);
                auto it = _request_handlers.find(command_str);
                if (it != _request_handlers.end()) {
                    callback = it->second;
                }
            }
            if (!callback) {
                LOG_G_WARN_M(Config) << "Unknown command: " << command_str;
                return;
            }
            try {
                callback(value, identity);
            } catch (const std::exception& e) {
                LOG_G_WARN_M(Config) << "Error processing command '" << command_str
                             << "': " << e.what();
            }
        } else {
            LOG_G_WARN_M(Config) << "Invalid command header received";
        }
    }

    void _record_peer_locked(
        const ControlPeer& identity,
        std::chrono::steady_clock::time_point now)
    {
        if (identity.empty()) {
            return;
        }
        _peers[identity] = now;
        _last_control_peer = identity;
        _has_last_control_peer = true;
        _prune_peers_locked(now);
    }

    void _prune_peers_locked(std::chrono::steady_clock::time_point now) const {
        for (auto it = _peers.begin(); it != _peers.end();) {
            if (now - it->second > kControlPeerTtl) {
                it = _peers.erase(it);
            } else {
                ++it;
            }
        }

        while (_peers.size() > kMaxControlPeers) {
            auto oldest = _peers.begin();
            for (auto it = std::next(_peers.begin()); it != _peers.end(); ++it) {
                if (it->second < oldest->second) {
                    oldest = it;
                }
            }
            _peers.erase(oldest);
        }

        if (_has_last_control_peer && _peers.find(_last_control_peer) == _peers.end()) {
            _has_last_control_peer = false;
            if (!_peers.empty()) {
                auto newest = _peers.begin();
                for (auto it = std::next(_peers.begin()); it != _peers.end(); ++it) {
                    if (it->second > newest->second) {
                        newest = it;
                    }
                }
                _last_control_peer = newest->first;
                _has_last_control_peer = true;
            }
        }
    }

    int _port;
    std::unique_ptr<zmq_transport::ControlRouter> _router;
    std::atomic<bool> _running{false};
    std::thread _thread;
    std::atomic<uint64_t> _control_outbound_drop_count{0};
    std::atomic<uint64_t> _control_send_drop_count{0};
    mutable std::mutex _mutex;
    std::unordered_map<std::string, Callback> _handlers;
    std::unordered_map<std::string, RequestCallback> _request_handlers;
    static constexpr size_t kMaxControlPeers = 64;
    static constexpr std::chrono::seconds kControlPeerTtl{30};
    mutable bool _has_last_control_peer = false;
    mutable ControlPeer _last_control_peer;
    mutable std::unordered_map<ControlPeer, std::chrono::steady_clock::time_point> _peers;
};

/**
 * @brief Hardware Synchronization Controller.
 * 
 * Controls an external hardware oscillator (e.g., OCXO) via a serial interface.
 * Allows fine-tuning of the frequency control word (DAC value) to correct for
 * frequency offsets (PPM) based on software estimation.
 */
class HardwareSyncController {
public:
    explicit HardwareSyncController(const std::string& device_path)
        : serial_fd_(-1), 
          worker_thread_(nullptr),
          terminate_flag_(false)
    {
        // Open serial port in non-blocking mode
        serial_fd_ = open(device_path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (serial_fd_ == -1) {
            LOG_G_ERROR_M(Config) << "ERROR: Failed to open serial device: " << strerror(errno);
            throw std::runtime_error("Serial device open failed");
        }
        
        // Configure serial port
        termios options;
        tcgetattr(serial_fd_, &options);
        cfsetispeed(&options, B57600);
        cfsetospeed(&options, B57600);
        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~PARENB;
        options.c_cflag &= ~CSTOPB;
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        options.c_oflag &= ~OPOST;
        options.c_cc[VMIN] = 0;     // Non-blocking read
        options.c_cc[VTIME] = 0;    // Return immediately
        
        if (tcsetattr(serial_fd_, TCSANOW, &options) != 0) {
            LOG_G_ERROR_M(Config) << "ERROR: Failed to set serial attributes: " << strerror(errno);
            close(serial_fd_);
            throw std::runtime_error("Serial configuration failed");
        }
        
        // Start worker thread
        worker_thread_ = new std::thread(&HardwareSyncController::worker_thread_func, this);
        hw_freq_word = MID_DAC_VALUE;
        set_frequency_control_word(hw_freq_word);
    }

    ~HardwareSyncController() {
        terminate_flag_ = true;
        
        // Wake up worker thread
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            condition_.notify_all();
        }
        
        if (worker_thread_ && worker_thread_->joinable()) {
            worker_thread_->join();
            delete worker_thread_;
        }
        
        if (serial_fd_ != -1) close(serial_fd_);
    }

    // Non-blocking frequency setting interface (no return)
    void set_frequency_control_word(int32_t control_value) {
        hw_freq_word = control_value;
        control_value = std::clamp(hw_freq_word, 0, MAX_DAC_VALUE);
        // Construct command string
        std::ostringstream oss;
        oss << "SF" << std::setw(7) << std::setfill('0') << control_value;
        std::string payload = oss.str();
        
        // Calculate checksum
        char checksum = 0;
        for (char c : payload) {
            checksum ^= c;
        }
        
        oss.str("");
        oss << payload << '*' << std::setw(3) << std::setfill('0') 
             << static_cast<int>(checksum) << '\n';
        
        // Add to send queue
        std::lock_guard<std::mutex> lock(queue_mutex_);
        command_queue_.push(oss.str());
        condition_.notify_one();
    }

    /**
    * Get current frequency control word
    * @return Current 20-bit DAC control word
    */
    int32_t get_frequency_control_word() const {
        return hw_freq_word;
    }

    /**
    * Convert ppm value to DAC control word
    * 
    * @param ppm Frequency offset in ppm (parts per million), should be within ±0.4 ppm
    * @return 20-bit DAC control word
    */
    int32_t ppm_to_control_word(double ppm) {
        double ratio = ppm * 1e-6;
        double fraction = ratio / (2.0 * FREQ_RANGE);  // Normalize to [-0.5, 0.5]
        double dac_value = MID_DAC_VALUE + fraction * MAX_DAC_VALUE;
        int32_t value = static_cast<int32_t>(std::round(dac_value));
        return std::clamp(value, 0, MAX_DAC_VALUE);
    }

    /**
    * Convert relative ppm value to DAC control word change
    * 
    * @param ppm Frequency offset in ppm
    * @return 20-bit DAC control word
    */
    int32_t ppm_to_control_word_relative(double delta_ppm) {
        double ratio = delta_ppm * 1e-6;
        double fraction = ratio / (2.0 * FREQ_RANGE);  // Normalize to [-0.5, 0.5]
        double dac_value = fraction * MAX_DAC_VALUE;
        int32_t value = static_cast<int32_t>(std::round(dac_value));
        return value;
    }

    /**
    * Convert DAC control word to ppm value
    * 
    * @param control_word 20-bit DAC control word
    * @return Frequency offset in ppm
    */
    double control_word_to_ppm(int32_t control_word) {
        int32_t clamped_word = std::clamp(control_word, 0, MAX_DAC_VALUE);
        double fraction = (static_cast<double>(clamped_word) - MID_DAC_VALUE) / MAX_DAC_VALUE;
        double ratio = fraction * (2.0 * FREQ_RANGE);
        return ratio * 1e6;
    }

    void set_frequency_control_ppm(double ppm) {
        int32_t control_word = ppm_to_control_word(ppm);
        set_frequency_control_word(control_word);
    }

    void configure_ocxo_pi(double kp_fast,
                           double ki_fast,
                           double kp_slow,
                           double ki_slow,
                           double switch_abs_error_ppm,
                           double switch_hold_s,
                           double max_step_fast_ppm,
                           double max_step_slow_ppm) {
        ocxo_pi_kp_fast_ = std::max(kp_fast, 0.0);
        ocxo_pi_ki_fast_ = std::max(ki_fast, 0.0);
        ocxo_pi_kp_slow_ = std::max(kp_slow, 0.0);
        ocxo_pi_ki_slow_ = std::max(ki_slow, 0.0);
        ocxo_pi_switch_abs_error_ppm_ = std::max(switch_abs_error_ppm, 0.0);
        ocxo_pi_switch_hold_s_ = std::max(switch_hold_s, 0.0);
        ocxo_pi_max_step_fast_ppm_ = std::max(max_step_fast_ppm, 0.0);
        ocxo_pi_max_step_slow_ppm_ = std::max(max_step_slow_ppm, 0.0);
        reset_ocxo_pi_state();
        LOG_G_INFO_M(Ocxo) << "Configured OCXO PI: Kp_fast=" << ocxo_pi_kp_fast_
                     << ", Ki_fast=" << ocxo_pi_ki_fast_
                     << ", Kp_slow=" << ocxo_pi_kp_slow_
                     << ", Ki_slow=" << ocxo_pi_ki_slow_
                     << ", switch_abs_error_ppm=" << ocxo_pi_switch_abs_error_ppm_
                     << ", hold=" << ocxo_pi_switch_hold_s_
                     << " s, max_step_fast_ppm=" << ocxo_pi_max_step_fast_ppm_
                     << ", max_step_slow_ppm=" << ocxo_pi_max_step_slow_ppm_;
    }

    double update_ocxo_pi_with_error_ppm(double error_ppm) {
        const auto now = std::chrono::steady_clock::now();
        if (!ocxo_pi_initialized_) {
            ocxo_pi_initialized_ = true;
            ocxo_pi_prev_error_ppm_ = error_ppm;
            ocxo_pi_stable_time_s_ = 0.0;
            ocxo_pi_last_update_time_ = now;
            const bool fast_stage = ocxo_pi_fast_mode_;
            const double kp = fast_stage ? ocxo_pi_kp_fast_ : ocxo_pi_kp_slow_;
            const double ki = fast_stage ? ocxo_pi_ki_fast_ : ocxo_pi_ki_slow_;
            LOG_RT_INFO_M(Ocxo) << "OCXO PI initialized (stage="
                          << (fast_stage ? "fast" : "slow")
                          << ", Kp=" << kp
                          << ", Ki=" << ki << ")";
            return 0.0;
        }

        double dt = std::chrono::duration<double>(now - ocxo_pi_last_update_time_).count();
        if (dt < 0.0) {
            dt = 0.0;
        }

        if (std::abs(error_ppm) < ocxo_pi_switch_abs_error_ppm_) {
            ocxo_pi_stable_time_s_ += dt;
        } else {
            ocxo_pi_stable_time_s_ = 0.0;
        }

        if (ocxo_pi_fast_mode_ && ocxo_pi_stable_time_s_ >= ocxo_pi_switch_hold_s_) {
            ocxo_pi_fast_mode_ = false;
            LOG_RT_INFO_M(Ocxo) << "OCXO PI switched to slow stage (Kp="
                          << ocxo_pi_kp_slow_ << ", Ki=" << ocxo_pi_ki_slow_ << ")";
        }

        const bool fast_stage = ocxo_pi_fast_mode_;
        const double kp = fast_stage ? ocxo_pi_kp_fast_ : ocxo_pi_kp_slow_;
        const double ki = fast_stage ? ocxo_pi_ki_fast_ : ocxo_pi_ki_slow_;
        const double max_step_ppm = fast_stage ? ocxo_pi_max_step_fast_ppm_ : ocxo_pi_max_step_slow_ppm_;
        double delta_ppm =
            kp * (error_ppm - ocxo_pi_prev_error_ppm_) + ki * dt * error_ppm;
        delta_ppm = std::clamp(delta_ppm, -max_step_ppm, max_step_ppm);

        ++ocxo_pi_adjust_log_counter_;
        if (ocxo_pi_adjust_log_counter_ % OCXO_PI_ADJUST_LOG_INTERVAL == 0) {
            LOG_RT_INFO_M(Ocxo) << "OCXO PI adjust: error_ppm=" << error_ppm
                          << ", delta_ppm=" << delta_ppm
                          << ", stage=" << (fast_stage ? "fast" : "slow")
                          << ", Kp=" << kp
                          << ", Ki=" << ki
                          << ", max_step=" << max_step_ppm
                          << ", stable=" << ocxo_pi_stable_time_s_ << " s";
        }
        set_frequency_control_ppm_relative(delta_ppm);
        ocxo_pi_prev_error_ppm_ = error_ppm;
        ocxo_pi_last_update_time_ = now;
        return delta_ppm;
    }

    void set_frequency_control_ppm_relative(double delta_ppm, double extra_ppm = 0) {
        int32_t control_word = ppm_to_control_word_relative(delta_ppm);
        hw_freq_word += control_word;
        set_frequency_control_word(hw_freq_word + extra_ppm);
    }

    void reset_ocxo_pi_state() {
        ocxo_pi_fast_mode_ = true;
        ocxo_pi_initialized_ = false;
        ocxo_pi_prev_error_ppm_ = 0.0;
        ocxo_pi_stable_time_s_ = 0.0;
        ocxo_pi_adjust_log_counter_ = 0;
        ocxo_pi_last_update_time_ = std::chrono::steady_clock::time_point{};
    }

    void reset_frequency_control() {
        hw_freq_word = MID_DAC_VALUE;
        set_frequency_control_word(hw_freq_word);
    }

private:
    int serial_fd_;
    std::thread* worker_thread_;
    std::atomic<bool> terminate_flag_;
    
    // Command queue
    std::queue<std::string> command_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    // Frequency adjustment range ±4.0e-7 (proportional), equivalent to ±0.4 ppm
    static constexpr double FREQ_RANGE = 4.0e-7;

    // DAC is 20-bit, control word range from 0 to (2^20 - 1)
    static constexpr int32_t MAX_DAC_VALUE = (1 << 20) - 1;  // 1048575
    static constexpr int32_t MID_DAC_VALUE = MAX_DAC_VALUE / 2;  // 524288
    int32_t hw_freq_word = 0;

    // OCXO PI state and parameters.
    bool ocxo_pi_fast_mode_ = true;
    bool ocxo_pi_initialized_ = false;
    double ocxo_pi_prev_error_ppm_ = 0.0;
    double ocxo_pi_stable_time_s_ = 0.0;
    std::chrono::steady_clock::time_point ocxo_pi_last_update_time_;
    double ocxo_pi_kp_fast_ = 30.0;
    double ocxo_pi_ki_fast_ = 1.0;
    double ocxo_pi_kp_slow_ = 30.0;
    double ocxo_pi_ki_slow_ = 0.05;
    double ocxo_pi_switch_abs_error_ppm_ = 0.0;
    double ocxo_pi_switch_hold_s_ = 60.0;
    double ocxo_pi_max_step_fast_ppm_ = 0.01;
    double ocxo_pi_max_step_slow_ppm_ = 0.01;
    static constexpr size_t OCXO_PI_ADJUST_LOG_INTERVAL = 256;
    size_t ocxo_pi_adjust_log_counter_ = 0;

    // Worker thread function - handles serial communication
    void worker_thread_func() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        while (!terminate_flag_) {
            std::string command;
            
            // Wait for command in queue
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this]{
                    return !command_queue_.empty() || terminate_flag_;
                });
                
                if (terminate_flag_) break;
                
                if (!command_queue_.empty()) {
                    command = std::move(command_queue_.front());
                    command_queue_.pop();
                }
            }
            
            // Send command and handle response
            if (!command.empty()) {
                send_command_and_handle_response(command);
            }
        }
    }

    // Send command and handle response (executed in background thread)
    void send_command_and_handle_response(const std::string& command) {
        // 1. Send command (with retry)
        ssize_t bytes_written = 0;
        int retry_count = 0;
        const char* data = command.c_str();
        size_t remaining = command.size();
        
        while (remaining > 0 && retry_count < 3) {
            ssize_t result = write(serial_fd_, data + bytes_written, remaining);
            
            if (result > 0) {
                bytes_written += result;
                remaining -= result;
            } else if (result == 0) {
                // Write timeout? Retry later
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                retry_count++;
            } else {
                if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    LOG_G_ERROR_M(Config) << "ERROR: Serial write error: " << strerror(errno)
                                  << " [command: " << command << "]";
                    return;
                }
                // Wait for buffer availability
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        
        if (remaining > 0) {
            LOG_G_ERROR_M(Config) << "ERROR: Failed to send full command after 3 attempts: "
                          << command;
            return;
        }
        
        // 2. Wait for response
        char response[32] = {0};
        int total_read = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        while (true) {
            // Timeout check (500ms)
            if (std::chrono::steady_clock::now() - start_time > std::chrono::milliseconds(500)) {
                LOG_G_ERROR_M(Config) << "ERROR: Response timeout for command: " << command;
                tcflush(serial_fd_, TCIFLUSH);
                return;
            }
            
            // Attempt to read response
            ssize_t n = read(serial_fd_, response + total_read, sizeof(response) - 1 - total_read);
            if (n > 0) {
                total_read += n;
                response[total_read] = '\0';
                
                // Check if full response received
                if (strchr(response, '\n') != nullptr) {
                    break;
                }
            } else if (n == 0) {
                // No data, wait briefly
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } else {
                if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    LOG_G_ERROR_M(Config) << "ERROR: Serial read error: " << strerror(errno);
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        
        // 3. Handle response
        if (strncmp(response, "OK", 2) == 0) {
            // Success response - silent handling
        } 
        else if (strncmp(response, "BAD DATA", 8) == 0) {
            LOG_G_ERROR_M(Config) << "ERROR: Device rejected command - BAD DATA: " << command;
        }
        else if (strncmp(response, "BAD CMD", 7) == 0) {
            LOG_G_ERROR_M(Config) << "ERROR: Device rejected command - BAD CMD: " << command;
        }
        else if (strncmp(response, "TIMEOUT", 7) == 0) {
            LOG_G_ERROR_M(Config) << "ERROR: Device operation timed out for command: " << command;
        }
        else {
            LOG_G_ERROR_M(Config) << "ERROR: Unexpected device response: " << response
                          << " for command: " << command;
        }
    }
};

/**
 * @brief Get the directory of the current executable.
 */
inline std::string get_executable_dir() {
    char result[4096];
    ssize_t count = readlink("/proc/self/exe", result, 4096);
    if (count != -1) {
        std::string path(result, count);
        size_t last_slash_idx = path.rfind('/');
        if (std::string::npos != last_slash_idx) {
            return path.substr(0, last_slash_idx);
        }
    }
    return ".";
}

#endif // COMMON_HPP
