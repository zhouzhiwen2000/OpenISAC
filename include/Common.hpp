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
#include <uhd/types/metadata.hpp>
#include <uhd/types/time_spec.hpp>
#include <uhd/utils/thread.hpp>
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
    std::vector<SimMultipathTap> uplink_multipath_taps; // Uplink TDL taps (empty => reuse comm_multipath_taps)
    double noise_power_dbfs = -100.0;          // AWGN power per RX channel (dBFS); very low = effectively off
    bool snr_control_enable = false;           // Enable target-SNR scaling of clean signal before AWGN
    double target_snr_db = 40.0;               // Target SNR when snr_control_enable is true
    int control_port = 10002;                  // ZMQ ROUTER port for ChannelSimulator runtime controls
    double cfo_hz = 0.0;                        // Initial carrier offset before simulated RX correction (Hz)
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
 * FDD: uplink occupies a separate carrier (`ul_center_freq`) and is transmitted
 *      continuously, simultaneously with the downlink. Uplink OFDM-symbol
 *      boundaries remain time-aligned to the downlink frame grid.
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
        "Invalid uplink_idle_waveform='" + waveform +
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
    double ul_center_freq = 0.0;    // FDD: uplink carrier frequency in Hz
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
    ObjectPool(size_t initial_size, FactoryFunc factory)
        : _factory(std::move(factory))
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
 * @brief Cooperative backoff helper for lock-free SPSC queue polling.
 *
 * Starts with a few scheduler yields, then falls back to short sleeps if the
 * queue stays empty/full. This avoids mutex blocking on the hot path while
 * keeping idle CPU burn bounded.
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

/**
 * @brief System Configuration Structure.
 * 
 * Holds all configurable parameters for the OFDM system, including FFT size,
 * cyclic prefix length, frequency settings, gain, and network configurations.
 */
struct Config {
    size_t fft_size = 1024;
    size_t range_fft_size = 1024;      // Range FFT size
    size_t doppler_fft_size = 100;     // Doppler FFT size
    size_t sensing_view_range_bins = 0;   // Backend RD view width (0 = full range_fft_size)
    size_t sensing_view_doppler_bins = 0; // Backend RD view height (0 = full doppler_fft_size)
    size_t cp_length = 128;            // Cyclic prefix length
    size_t num_symbols = 100;          // Number of symbols per frame
    size_t sensing_symbol_num = 100;   // Number of sensing symbols
    size_t cuda_mod_pipeline_slots = 2;   // Number of CUDA mod pipeline slots
    size_t cuda_demod_pipeline_slots = 3; // Number of CUDA demod pipeline slots
    size_t frame_queue_size = 8;       // Capacity of demod RX frame queue
    size_t sync_queue_size = 8;        // Capacity of demod sync-search queue
    size_t sync_pos = 1;               // Synchronization symbol position
    bool enable_sec_sync_symbol = false; // Reserve sync_pos-1 for the duplicate second sync symbol
    bool enable_cfo_training_sequence = false; // Reserve sync_pos+1 for a repeated CFO training symbol
    size_t cfo_training_period_samples = 16; // Repetition period of the CFO training symbol in samples
    double sync_cfo_alias_search_range_hz = 800000.0; // Max absolute CFO span covered by sync alias search
    int delay_adjust_step = 2;         // Delay adjustment step
    double reset_hold_s = 0.5;         // Time window of persistent invalid delay before forcing a hard reset
    int desired_peak_pos = 20;         // Desired delay peak position to include non-causal components
    bool predictive_delay = true;      // Enable CFO-based predictive delay compensation during alignment/tracking
    double sample_rate = 50e6;         // Sample rate
    double bandwidth = 50e6;           // Bandwidth
    double center_freq = 2.4e9;        // Center frequency

    double tx_gain = 0.0;              // TX gain
    double rx_gain = 0.0;              // RX gain (Channel 1)
    bool rx_agc_enable = false;        // Enable hardware RX AGC via USRP gain control
    double rx_agc_low_threshold_db = 11.0; // Increase gain when delay-spectrum peak is below this threshold
    double rx_agc_high_threshold_db = 13.0; // Decrease gain when delay-spectrum peak is above this threshold
    double rx_agc_max_step_db = 3.0;   // Maximum gain change per AGC update
    size_t rx_agc_update_frames = 4;   // Frame interval between AGC updates
    uint32_t tx_channel = 0;           // TX channel index
    size_t rx_channel = 0;             // RX channel index
    int zc_root = 29;                  // Zadoff-Chu sequence root index
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

    std::vector<size_t> pilot_positions = {571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451};
    std::vector<size_t> midframe_pilot_symbols; // Absolute mid-frame pilot symbols; comb pilot RE are preserved
    uint32_t midframe_pilot_seed = 0x4D46504Cu; // Deterministic BPSK pilot seed ("MFPL")
    std::string equalizer_mode = kEqualizerModeMmse; // zf or mmse
    std::string channel_tracking_mode = kChannelTrackingModePilotPhase; // disabled or pilot_phase
    double equalizer_mag_floor = 1e-6; // Lower bound for |H|^2 in channel inversion
    double channel_tracking_min_pilot_snr = 1e-4; // Minimum pilot residual weight before falling back
    bool data_resource_blocks_configured = false;
    std::vector<DataResourceBlock> data_resource_blocks;
    std::string sensing_output_mode = kSensingOutputModeDense;
    SensingOnWireFormat sensing_on_wire_format = SensingOnWireFormat::ComplexFloat32;
    bool enable_backend_sensing_processing = false;
    std::vector<DataResourceBlock> sensing_mask_blocks;
    size_t payload_re_count = 0;
    size_t non_pilot_re_count = 0;
    std::string device_args = "";
    std::string tx_device_args = "";
    std::string rx_device_args = "";
    std::string radio_backend = "uhd";  // Radio I/O backend: "uhd" (real USRP) or "sim" (channel simulator)
    SimConfig simulation;               // Channel simulator parameters (used when radio_backend == "sim")
    bool enable_uplink = false;         // Top-level UE->BS uplink master switch
    DuplexConfig duplex;                // Duplexing (TDD/FDD) + uplink frame structure
    // Runtime-adjustable DL/UL boundary timing knobs (samples). The BS knob is
    // the DL/UL timing difference; the UE knob is the Timing Advance. Each side
    // adjusts its own value independently at runtime; stored here as the startup
    // default, mirrored into atomics by the engines.
    int32_t bs_dl_ul_timing_diff = 63; // BS: uplink-RX window offset relative to the TX frame anchor (samples)
    int32_t ue_timing_advance = 63;    // UE: uplink-TX window advance relative to the RX frame anchor (samples)
    std::string uplink_idle_waveform = kUplinkIdleWaveformRandomQpsk; // UE idle UL payload RE: zero or random_qpsk
    // UE uplink payload UDP input (mirrors udp_input_* on the BS downlink).
    std::string ul_udp_input_ip = "0.0.0.0";
    int ul_udp_input_port = 50002;
    // BS uplink decoded-payload UDP output (mirrors udp_output_* on the UE downlink).
    std::string ul_udp_output_ip = "127.0.0.1";
    int ul_udp_output_port = 50003;
    std::string default_out_ip = "127.0.0.1";
    bool mono_sensing_output_enabled = true;
    std::string mono_sensing_ip = "0.0.0.0";
    int mono_sensing_port = 8888;
    bool enable_bi_sensing = true;
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
    std::string channel_ip = "0.0.0.0";
    int channel_port = 12348;
    std::string pdf_ip = "0.0.0.0";
    int pdf_port = 12349;
    std::string constellation_ip = "0.0.0.0";
    int constellation_port = 12346;
    std::string vofa_debug_ip = "127.0.0.1";
    int vofa_debug_port = 12347;
    // UDP output for decoded payloads
    std::string udp_output_ip = "127.0.0.1";
    int udp_output_port = 50001;
    // BS UDP input (for incoming payloads to modulate)
    std::string udp_input_ip = "0.0.0.0"; // bind address
    int udp_input_port = 50000;
    std::string clocksource = "internal"; // Clock source
    std::string timesource = "";          // Time source; empty means follow clocksource
    std::string tx_clock_source = "";     // TX clock source override
    std::string tx_time_source = "";      // TX time source override
    std::string rx_clock_source = "";     // Default RX clock source override
    std::string rx_time_source = "";      // Default RX time source override
    std::string wire_format_tx = "sc16";
    std::string uplink_rx_wire_format = "sc16";   // BS uplink RX wire format
    std::string sensing_rx_wire_format = "sc16";  // BS sensing RX default wire format
    std::string downlink_rx_wire_format = "sc16"; // UE downlink RX wire format
    uint32_t sensing_rx_channel_count = 1; // Number of sensing RX channels
    std::vector<SensingRxChannelConfig> sensing_rx_channels; // Per-channel sensing RX config
    size_t sensing_symbol_stride = 20;     // Default sensing STRD applied at startup
    size_t tx_circular_buffer_size = 8;    // Capacity of BS frame circular buffer
    size_t data_packet_buffer_size = 32;   // Capacity of BS encoded packet buffer
    size_t paired_frame_queue_size = 16;   // Keep headroom above the default TX frame queue
    std::vector<int> downlink_cpu_cores; // BS: TX/mod/data-ingest; UE: RX/process/sensing/bit-processing
    std::vector<int> uplink_cpu_cores;   // Dedicated uplink thread cores; empty = no explicit uplink binding
    int main_cpu_core = -1;              // Main-thread affinity; -1 = no explicit binding
    std::string profiling_modules = "";  // Comma-separated list of modules to profile: modulation, latency, sensing_proc, data_ingest, demodulation, agc, align, snr, uplink, or "all"
    bool measurement_enable = false;
    std::string measurement_mode = "";
    std::string measurement_run_id = "";
    std::string measurement_output_dir = "";
    size_t measurement_payload_bytes = 1024;
    uint32_t measurement_prbs_seed = 0x5A;
    uint32_t measurement_packets_per_point = 1;
    size_t measurement_max_packets_per_frame = 1; // 0 = unlimited
    
    // Check if a specific module should be profiled
    bool should_profile(const std::string& module) const {
        if (profiling_modules.empty()) return false;
        if (profiling_modules == "all") return true;
        size_t pos = 0;
        while (pos < profiling_modules.size()) {
            while (pos < profiling_modules.size() &&
                   (profiling_modules[pos] == ',' ||
                    std::isspace(static_cast<unsigned char>(profiling_modules[pos])))) {
                ++pos;
            }
            const size_t token_start = pos;
            while (pos < profiling_modules.size() && profiling_modules[pos] != ',') {
                ++pos;
            }
            size_t token_end = pos;
            while (token_end > token_start &&
                   std::isspace(static_cast<unsigned char>(profiling_modules[token_end - 1]))) {
                --token_end;
            }
            const size_t token_len = token_end - token_start;
            if (token_len == module.size() &&
                profiling_modules.compare(token_start, token_len, module) == 0) {
                return true;
            }
        }
        return false;
    }

    // Calculate total samples per frame
    size_t samples_per_frame() const { 
        return num_symbols * (fft_size + cp_length); 
    }
    
    // Calculate synchronization samples
    size_t sync_samples() const { 
        return 2 * samples_per_frame(); 
    }
};

inline bool sec_sync_symbol_enabled(const Config& cfg) {
    return cfg.enable_sec_sync_symbol;
}

inline bool cfo_training_sequence_enabled(const Config& cfg) {
    return cfg.enable_cfo_training_sequence;
}

inline size_t cfo_training_symbol_index(const Config& cfg) {
    return cfg.sync_pos + 1;
}

inline bool is_sec_sync_symbol(const Config& cfg, size_t symbol_idx) {
    return sec_sync_symbol_enabled(cfg) &&
           cfg.sync_pos > 0 &&
           symbol_idx + 1 == cfg.sync_pos;
}

inline bool is_main_sync_symbol(const Config& cfg, size_t symbol_idx) {
    return symbol_idx == cfg.sync_pos;
}

inline bool is_zc_sync_symbol(const Config& cfg, size_t symbol_idx) {
    return is_main_sync_symbol(cfg, symbol_idx) || is_sec_sync_symbol(cfg, symbol_idx);
}

inline bool is_cfo_training_symbol(const Config& cfg, size_t symbol_idx) {
    return cfo_training_sequence_enabled(cfg) &&
           cfg.sync_pos < std::numeric_limits<size_t>::max() &&
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
        return (ul_start + ul_guard) * (cfg.fft_size + cfg.cp_length);
    }
    // Number of samples in the uplink data block.
    size_t ul_sample_count(const Config& cfg) const {
        if (mode == DuplexMode::FDD) return cfg.samples_per_frame();
        const size_t ul_data_syms = (ul_count > ul_guard) ? (ul_count - ul_guard) : 0;
        return ul_data_syms * (cfg.fft_size + cfg.cp_length);
    }
};

inline DuplexFrameLayout build_duplex_frame_layout(const Config& cfg) {
    DuplexFrameLayout layout;
    layout.mode = cfg.duplex.mode;
    layout.num_symbols = cfg.num_symbols;
    if (!cfg.enable_uplink) {
        layout.uplink_enabled = false;
        layout.symbol_is_uplink.assign(cfg.num_symbols, 0);
        layout.symbol_is_guard.assign(cfg.num_symbols, 0);
        return layout;
    }

    if (cfg.duplex.mode == DuplexMode::FDD) {
        // FDD: continuous uplink on a separate carrier; no symbol gating/guard.
        layout.uplink_enabled = true;
        layout.ul_start = 0;
        layout.ul_count = cfg.num_symbols;
        layout.ul_guard = 0;
        return layout;
    }

    // TDD: validate and clamp the configured uplink symbol range.
    size_t ul_start = cfg.duplex.ul_symbol_start;
    size_t ul_count = cfg.duplex.ul_symbol_count;
    size_t ul_guard = cfg.duplex.ul_guard_symbols;

    if (ul_count == 0) {
        layout.uplink_enabled = false;   // uplink disabled
        layout.symbol_is_uplink.assign(cfg.num_symbols, 0);
        layout.symbol_is_guard.assign(cfg.num_symbols, 0);
        return layout;
    }

    if (ul_start >= cfg.num_symbols) {
        layout.warning = "uplink symbol_start beyond frame; uplink disabled";
        layout.uplink_enabled = false;
        layout.symbol_is_uplink.assign(cfg.num_symbols, 0);
        layout.symbol_is_guard.assign(cfg.num_symbols, 0);
        return layout;
    }
    if (ul_start + ul_count > cfg.num_symbols) {
        layout.warning = "uplink range exceeds frame; clamped to num_symbols";
        ul_count = cfg.num_symbols - ul_start;
    }
    if (ul_guard > ul_count) {
        layout.warning = "uplink guard exceeds uplink range; clamped";
        ul_guard = ul_count;
    }

    layout.ul_start = ul_start;
    layout.ul_count = ul_count;
    layout.ul_guard = ul_guard;
    layout.uplink_enabled = (ul_count > ul_guard);

    layout.symbol_is_uplink.assign(cfg.num_symbols, 0);
    layout.symbol_is_guard.assign(cfg.num_symbols, 0);
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
//      the uplink window). FDD: num_symbols = cfg.num_symbols (continuous uplink).
inline Config make_uplink_config(const Config& cfg) {
    Config ul = cfg;
    const DuplexFrameLayout dl = build_duplex_frame_layout(cfg);
    const size_t ul_syms = !cfg.enable_uplink
        ? 0
        : ((cfg.duplex.mode == DuplexMode::FDD)
            ? cfg.num_symbols
            : ((dl.ul_count > dl.ul_guard) ? (dl.ul_count - dl.ul_guard) : 0));
    ul.num_symbols = ul_syms;
    ul.sync_pos = 0;
    const int sensing_pilot_root = select_distinct_zc_root(cfg.fft_size, cfg.zc_root);
    ul.zc_root = select_distinct_zc_root(
        cfg.fft_size,
        cfg.zc_root,
        std::vector<int>{sensing_pilot_root});
    ul.enable_sec_sync_symbol = false;
    ul.enable_cfo_training_sequence = false;
    ul.midframe_pilot_symbols.clear();
    ul.sensing_mask_blocks.clear();
    ul.data_resource_blocks.clear();
    ul.data_resource_blocks_configured = false;
    // The uplink does not run sensing; keep sensing_symbol_num consistent.
    ul.sensing_symbol_num = ul_syms;
    return ul;
}

// True when the uplink carries at least one data-bearing symbol (a ZC sync plus
// at least one data symbol). Uses make_uplink_config()'s derived num_symbols.
inline bool uplink_enabled(const Config& cfg) {
    return make_uplink_config(cfg).num_symbols >= 2;
}

// One-line startup summary of the duplex frame partition. `role` is "BS" or "UE".
inline void log_duplex_summary(const Config& cfg, const char* role) {
    const DuplexFrameLayout dl = build_duplex_frame_layout(cfg);
    if (!uplink_enabled(cfg)) {
        LOG_G_INFO() << "[" << role << "] duplex: uplink DISABLED (downlink-only); "
                     << "num_symbols=" << cfg.num_symbols;
        return;
    }
    const Config ul = make_uplink_config(cfg);
    std::ostringstream oss;
    oss << "[" << role << "] duplex partition: mode="
        << duplex_mode_to_string(cfg.duplex.mode)
        << ", frame_symbols=" << cfg.num_symbols;
    if (cfg.duplex.mode == DuplexMode::TDD) {
        const size_t dl_syms = cfg.num_symbols - dl.ul_count;
        oss << ", DL=" << dl_syms << ", guard=" << dl.ul_guard
            << ", UL=" << ul.num_symbols
            << " (UL symbols [" << (dl.ul_start + dl.ul_guard) << ","
            << (dl.ul_start + dl.ul_count) << "))";
    } else {
        oss << ", UL=" << ul.num_symbols << " (continuous), ul_center_freq="
            << cfg.duplex.ul_center_freq << " Hz";
    }
    LOG_G_INFO() << oss.str();
}

inline bool validate_cfo_training_period(const Config& cfg, std::string* error = nullptr) {
    if (!cfo_training_sequence_enabled(cfg)) {
        return true;
    }
    if (cfg.cfo_training_period_samples == 0) {
        if (error) *error = "cfo_training_period_samples must be greater than 0.";
        return false;
    }
    if (cfg.cfo_training_period_samples >= cfg.fft_size) {
        if (error) {
            *error = "cfo_training_period_samples must be smaller than fft_size.";
        }
        return false;
    }
    if ((cfg.fft_size % cfg.cfo_training_period_samples) != 0) {
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
    return cfo_sym == cfg.sync_pos || is_sec_sync_symbol(cfg, cfo_sym);
}

inline bool dense_sensing_stride_hits_symbol(
    const Config& cfg,
    size_t stride,
    size_t symbol_idx)
{
    if (cfg.num_symbols == 0 || stride == 0 || symbol_idx >= cfg.num_symbols) {
        return false;
    }
    return (symbol_idx % std::gcd(stride, cfg.num_symbols)) == 0;
}

inline std::string dense_sensing_stride_cfo_training_error(
    const Config& cfg,
    size_t stride,
    const std::string& context)
{
    if (cfg.num_symbols == 0 || stride == 0 || !cfo_training_sequence_enabled(cfg)) {
        return {};
    }
    const size_t sym = cfo_training_symbol_index(cfg);
    if (sym < cfg.num_symbols && dense_sensing_stride_hits_symbol(cfg, stride, sym)) {
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
        "Invalid sensing_output_mode='" + mode +
        "'. Supported values: dense, compact_mask.");
}

inline bool sensing_output_mode_is_compact_mask(const Config& cfg) {
    return cfg.sensing_output_mode == kSensingOutputModeCompactMask;
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
    if (cfg.num_symbols == 0) {
        throw std::runtime_error("num_symbols=0 is invalid for sensing mask layout.");
    }
    if (cfg.fft_size == 0) {
        throw std::runtime_error("fft_size=0 is invalid for sensing mask layout.");
    }

    SensingMaskLayout layout;
    layout.num_symbols = cfg.num_symbols;
    layout.fft_size = cfg.fft_size;
    layout.symbol_to_selected_rank.assign(cfg.num_symbols, -1);

    std::vector<uint8_t> mask(cfg.num_symbols * cfg.fft_size, 0);
    auto flat_index = [&cfg](size_t symbol_index, size_t subcarrier_index) {
        return symbol_index * cfg.fft_size + subcarrier_index;
    };

    for (size_t block_idx = 0; block_idx < cfg.sensing_mask_blocks.size(); ++block_idx) {
        const auto& block = cfg.sensing_mask_blocks[block_idx];
        if (block.symbol_count == 0) {
            throw std::runtime_error(
                "sensing_mask_blocks[" + std::to_string(block_idx) +
                "].symbol_count must be greater than 0.");
        }
        if (block.subcarrier_count == 0) {
            throw std::runtime_error(
                "sensing_mask_blocks[" + std::to_string(block_idx) +
                "].subcarrier_count must be greater than 0.");
        }
        if (block.symbol_start >= cfg.num_symbols ||
            block.symbol_start + block.symbol_count > cfg.num_symbols) {
            throw std::runtime_error(
                "sensing_mask_blocks[" + std::to_string(block_idx) +
                "] exceeds the configured symbol range.");
        }
        if (block.subcarrier_start >= cfg.fft_size ||
            block.subcarrier_start + block.subcarrier_count > cfg.fft_size) {
            throw std::runtime_error(
                "sensing_mask_blocks[" + std::to_string(block_idx) +
                "] exceeds the configured subcarrier range.");
        }

        for (size_t sym = block.symbol_start; sym < block.symbol_start + block.symbol_count; ++sym) {
            if (is_cfo_training_symbol(cfg, sym)) {
                throw std::runtime_error(
                    "sensing_mask_blocks[" + std::to_string(block_idx) +
                    "] selects symbol " + std::to_string(sym) +
                    ", which is the CFO training field. CFO training fields are not valid sensing symbols.");
            }
            for (size_t sc = block.subcarrier_start; sc < block.subcarrier_start + block.subcarrier_count; ++sc) {
                mask[flat_index(sym, sc)] = 1;
            }
        }
    }

    layout.selected_symbol_offsets.push_back(0);
    for (size_t sym = 0; sym < cfg.num_symbols; ++sym) {
        size_t row_count = 0;
        for (size_t sc = 0; sc < cfg.fft_size; ++sc) {
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
            "compact_mask mode requires sensing_mask_blocks to select at least one RE.");
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
    if (cfg.num_symbols == 0) {
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

    size_t expected_gap = cfg.num_symbols;
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
        static_cast<int>(cfg.num_symbols) +
        layout.selected_symbols.front() -
        layout.selected_symbols.back());
    if (wrap_gap != expected_gap) {
        analysis.incompatibility_reason =
            "compact_mask selected symbols are not equally spaced when wrap-around is included.";
        return analysis;
    }

    analysis.regular_subsampling_compatible = true;
    analysis.implicit_symbol_stride = expected_gap;

    if (cfg.range_fft_size < analysis.common_subcarrier_count) {
        analysis.runtime_restriction_reason =
            "compact_mask regular sampling selects " + std::to_string(analysis.common_subcarrier_count) +
            " subcarriers, which exceeds range_fft_size=" + std::to_string(cfg.range_fft_size) + '.';
        return analysis;
    }
    if (analysis.selected_symbol_count > cfg.doppler_fft_size) {
        analysis.runtime_restriction_reason =
            "compact_mask regular sampling selects " + std::to_string(analysis.selected_symbol_count) +
            " symbols, which exceeds doppler_fft_size=" + std::to_string(cfg.doppler_fft_size) + '.';
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
    if (!cfg.enable_backend_sensing_processing) {
        return false;
    }
    if (!sensing_output_mode_is_compact_mask(cfg)) {
        return true;
    }
    return analyze_compact_sensing_mask(cfg).local_delay_doppler_supported;
}

inline std::string backend_sensing_processing_reason(const Config& cfg) {
    if (!cfg.enable_backend_sensing_processing) {
        return "backend sensing processing disabled";
    }
    if (!sensing_output_mode_is_compact_mask(cfg)) {
        return {};
    }
    return analyze_compact_sensing_mask(cfg).effective_reason();
}

inline size_t required_sensing_range_bin_count(const Config& cfg) {
    if (!sensing_output_mode_is_compact_mask(cfg)) {
        return std::max<size_t>(cfg.fft_size, 1);
    }
    const CompactSensingMaskAnalysis analysis = analyze_compact_sensing_mask(cfg);
    if (analysis.regular_subsampling_compatible) {
        return std::max<size_t>(analysis.common_subcarrier_count, 1);
    }
    return 0;
}

inline size_t required_sensing_doppler_symbol_count(const Config& cfg) {
    size_t required = std::max<size_t>(cfg.sensing_symbol_num, 1);
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
    const size_t full_range_bins = std::max<size_t>(cfg.range_fft_size, 1);
    if (cfg.sensing_view_range_bins == 0) {
        return full_range_bins;
    }
    return std::clamp(cfg.sensing_view_range_bins, size_t{1}, full_range_bins);
}

inline size_t resolved_sensing_view_doppler_bins(const Config& cfg) {
    const size_t full_doppler_bins = std::max<size_t>(cfg.doppler_fft_size, 1);
    const size_t min_doppler_bins = std::max<size_t>(required_sensing_doppler_symbol_count(cfg), 1);
    if (cfg.sensing_view_doppler_bins == 0) {
        return full_doppler_bins;
    }
    return std::clamp(cfg.sensing_view_doppler_bins, min_doppler_bins, full_doppler_bins);
}

inline void normalize_sensing_fft_sizes(Config& cfg, const char* context_name) {
    const size_t required_range = required_sensing_range_bin_count(cfg);
    if (cfg.range_fft_size == 0) {
        const size_t fallback_range = (required_range > 0) ? required_range : std::max<size_t>(cfg.fft_size, 1);
        LOG_G_WARN() << "range_fft_size is unset or 0. Defaulting to " << fallback_range << '.';
        cfg.range_fft_size = fallback_range;
    }
    if (required_range > 0 && cfg.range_fft_size < required_range) {
        LOG_G_WARN() << "range_fft_size=" << cfg.range_fft_size
                     << " is smaller than the required sensing subcarrier count=" << required_range
                     << " for " << context_name
                     << ". Expanding range_fft_size to keep delay FFT buffers consistent.";
        cfg.range_fft_size = required_range;
    }
    if (cfg.doppler_fft_size == 0) {
        LOG_G_WARN() << "doppler_fft_size=0 is invalid. Clamping to 1.";
        cfg.doppler_fft_size = 1;
    }
    if (cfg.sensing_symbol_num == 0) {
        LOG_G_WARN() << "sensing_symbol_num=0 is invalid. Clamping to 1.";
        cfg.sensing_symbol_num = 1;
    }

    const size_t required_doppler = required_sensing_doppler_symbol_count(cfg);
    if (cfg.doppler_fft_size < required_doppler) {
        LOG_G_WARN() << "doppler_fft_size=" << cfg.doppler_fft_size
                     << " is smaller than the required sensing symbol count=" << required_doppler
                     << " for " << context_name
                     << ". Expanding doppler_fft_size to keep sensing buffers consistent.";
        cfg.doppler_fft_size = required_doppler;
    }
}

inline void normalize_sensing_view_bins(Config& cfg, const char* context_name) {
    if (cfg.sensing_view_range_bins != 0 && cfg.sensing_view_range_bins > cfg.range_fft_size) {
        LOG_G_WARN() << context_name
                     << " sensing_view_range_bins=" << cfg.sensing_view_range_bins
                     << " exceeds range_fft_size=" << cfg.range_fft_size
                     << ". Clamping backend view width to the configured range FFT size.";
        cfg.sensing_view_range_bins = cfg.range_fft_size;
    }

    if (cfg.sensing_view_doppler_bins != 0) {
        const size_t min_doppler_bins = std::max<size_t>(required_sensing_doppler_symbol_count(cfg), 1);
        if (cfg.sensing_view_doppler_bins < min_doppler_bins) {
            LOG_G_WARN() << context_name
                         << " sensing_view_doppler_bins=" << cfg.sensing_view_doppler_bins
                         << " is smaller than the required slow-time symbol count="
                         << min_doppler_bins
                         << ". Clamping backend view height to preserve the full sensing aperture.";
            cfg.sensing_view_doppler_bins = min_doppler_bins;
        }
        if (cfg.sensing_view_doppler_bins > cfg.doppler_fft_size) {
            LOG_G_WARN() << context_name
                         << " sensing_view_doppler_bins=" << cfg.sensing_view_doppler_bins
                         << " exceeds doppler_fft_size=" << cfg.doppler_fft_size
                         << ". Clamping backend view height to the configured Doppler FFT size.";
            cfg.sensing_view_doppler_bins = cfg.doppler_fft_size;
        }
    }
}

inline DataResourceGridLayout build_data_resource_grid_layout(
    const Config& cfg,
    bool log_warnings = false)
{
    if (cfg.num_symbols == 0) {
        throw std::runtime_error("num_symbols=0 is invalid for data-resource layout.");
    }
    if (cfg.sync_pos >= cfg.num_symbols) {
        throw std::runtime_error(
            "sync_pos=" + std::to_string(cfg.sync_pos) +
            " is out of range for num_symbols=" + std::to_string(cfg.num_symbols) + '.');
    }
    if (sec_sync_symbol_enabled(cfg)) {
        if (cfg.num_symbols < 2) {
            throw std::runtime_error("enable_sec_sync_symbol requires num_symbols >= 2.");
        }
        if (cfg.sync_pos == 0) {
            throw std::runtime_error(
                "enable_sec_sync_symbol requires sync_pos >= 1 so sync_pos-1 can hold the second sync symbol.");
        }
    }
    if (cfo_training_sequence_enabled(cfg)) {
        if (cfg.sync_pos + 1 >= cfg.num_symbols) {
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
    layout.num_symbols = cfg.num_symbols;
    layout.fft_size = cfg.fft_size;
    layout.sync_pos = cfg.sync_pos;
    const DuplexFrameLayout duplex_layout = build_duplex_frame_layout(cfg);
    layout.midframe_pilot_symbol_mask.assign(cfg.num_symbols, 0);
    layout.midframe_pilot_symbol_to_rank.assign(cfg.num_symbols, -1);
    for (auto sym : cfg.midframe_pilot_symbols) {
        if (sym >= cfg.num_symbols) {
            if (log_warnings) {
                LOG_G_WARN() << "Ignoring midframe_pilot_symbols entry " << sym
                             << " outside num_symbols=" << cfg.num_symbols << '.';
            }
            continue;
        }
        if (!duplex_layout.is_downlink(sym)) {
            if (log_warnings) {
                LOG_G_WARN() << "Ignoring midframe_pilot_symbols entry " << sym
                             << " because it falls inside a TDD uplink/guard symbol.";
            }
            continue;
        }
        if (is_reserved_sync_symbol(cfg, sym)) {
            if (log_warnings) {
                LOG_G_WARN() << "Ignoring midframe_pilot_symbols entry " << sym
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
    for (size_t sym = 0; sym < cfg.num_symbols; ++sym) {
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

    layout.pilot_mask.assign(cfg.fft_size, 0);
    for (auto pos : cfg.pilot_positions) {
        if (pos < cfg.fft_size) {
            layout.pilot_mask[pos] = 1;
        }
    }

    layout.subcarrier_to_non_pilot_index.assign(cfg.fft_size, -1);
    layout.non_pilot_subcarrier_indices.reserve(cfg.fft_size);
    for (size_t k = 0; k < cfg.fft_size; ++k) {
        if (layout.pilot_mask[k] != 0) {
            continue;
        }
        layout.subcarrier_to_non_pilot_index[k] =
            static_cast<int>(layout.non_pilot_subcarrier_indices.size());
        layout.non_pilot_subcarrier_indices.push_back(static_cast<int>(k));
    }
    layout.num_non_pilot_subcarriers = layout.non_pilot_subcarrier_indices.size();

    layout.data_symbol_to_actual_symbol.reserve(downlink_symbol_count);
    layout.actual_symbol_to_data_symbol.assign(cfg.num_symbols, -1);
    for (size_t sym = 0; sym < cfg.num_symbols; ++sym) {
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

    layout.payload_mask.assign(layout.non_pilot_re_count, cfg.data_resource_blocks_configured ? 0 : 1);
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
    if (cfg.data_resource_blocks_configured) {
        for (size_t block_idx = 0; block_idx < cfg.data_resource_blocks.size(); ++block_idx) {
            const auto& block = cfg.data_resource_blocks[block_idx];
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
            if (block.symbol_start >= cfg.num_symbols ||
                block.symbol_start + block.symbol_count > cfg.num_symbols) {
                throw std::runtime_error(
                    "data_resource_blocks[" + std::to_string(block_idx) +
                    "] exceeds the configured symbol range.");
            }
            if (block.subcarrier_start >= cfg.fft_size ||
                block.subcarrier_start + block.subcarrier_count > cfg.fft_size) {
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

    if (log_warnings && cfg.data_resource_blocks_configured &&
        (stripped_reserved_symbol_re > 0 ||
         stripped_non_downlink_symbol_re > 0 ||
         stripped_pilot_re > 0)) {
        LOG_G_WARN() << "data_resource_blocks overlap stripped " << stripped_reserved_symbol_re
                     << " sync/mid-frame-pilot symbol RE, "
                     << stripped_non_downlink_symbol_re
                     << " TDD uplink/guard symbol RE, and " << stripped_pilot_re
                     << " pilot RE. reserved sync symbols, pilot_positions, and midframe_pilot_symbols take precedence.";
    }
    if (log_warnings && payload_sensing_pilot_overlap_re > 0) {
        LOG_G_WARN() << "data_resource_blocks contain " << payload_sensing_pilot_overlap_re
                     << " RE selected as both payload and sensing_pilot. sensing_pilot takes precedence.";
    }

    return layout;
}

inline void finalize_data_resource_grid_config(Config& cfg, const char* role_name) {
    const DataResourceGridLayout layout = build_data_resource_grid_layout(cfg, true);
    cfg.payload_re_count = layout.payload_re_count;
    cfg.non_pilot_re_count = layout.non_pilot_re_count;
    if (cfg.measurement_enable && cfg.payload_re_count == 0) {
        throw std::runtime_error(
            std::string(role_name) +
            " measurement_enable requires at least one payload RE. "
            "Omit data_resource_blocks for full payload coverage or configure a non-empty payload grid.");
    }
}

inline void finalize_sensing_mask_config(Config& cfg, const char* role_name) {
    cfg.sensing_output_mode = normalize_sensing_output_mode_string(cfg.sensing_output_mode);
    if (!sensing_output_mode_is_compact_mask(cfg)) {
        validate_dense_sensing_stride(
            cfg,
            cfg.sensing_symbol_stride,
            std::string(role_name) + " dense sensing stride");
        if (cfg.enable_backend_sensing_processing && !backend_sensing_processing_supported(cfg)) {
            LOG_G_WARN() << role_name
                         << " requested enable_backend_sensing_processing=1, but the current sensing mode "
                         << "cannot provide dense backend RD output. Falling back to viewer-local processing.";
            cfg.enable_backend_sensing_processing = false;
        }
        return;
    }
    const SensingMaskLayout layout = build_sensing_mask_layout(cfg);
    if (layout.empty()) {
        throw std::runtime_error(
            std::string(role_name) +
            " compact_mask mode requires a non-empty sensing_mask_blocks selection.");
    }
    if (cfg.enable_backend_sensing_processing && !backend_sensing_processing_supported(cfg)) {
        LOG_G_WARN() << role_name
                     << " requested enable_backend_sensing_processing=1 in compact_mask mode, but the mask "
                     << "is not regular local-DD compatible: "
                     << backend_sensing_processing_reason(cfg)
                     << ". Falling back to viewer-local processing.";
        cfg.enable_backend_sensing_processing = false;
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
    return cfg.measurement_enable && cfg.measurement_mode == "internal_prbs";
}

/**
 * @brief True when the radio I/O backend is the channel simulator (no USRP).
 */
inline bool radio_is_sim(const Config& cfg) {
    return cfg.radio_backend == "sim";
}

/**
 * @brief Emit the radio_backend + simulation block to a YAML emitter.
 *
 * Shared by the BS and UE config writers so the two stay in sync.
 */
inline void emit_simulation_config(YAML::Emitter& out, const Config& cfg) {
    out << YAML::Key << "radio_backend" << YAML::Value << cfg.radio_backend;
    out << YAML::Key << "simulation" << YAML::Value << YAML::BeginMap;
    const SimConfig& sim = cfg.simulation;
    out << YAML::Key << "session" << YAML::Value << sim.session;
    out << YAML::Key << "enable_comm_rx" << YAML::Value << sim.enable_comm_rx;
    out << YAML::Key << "enable_sensing_rx" << YAML::Value << sim.enable_sensing_rx;
    out << YAML::Key << "enable_uplink" << YAML::Value << sim.enable_uplink;
    out << YAML::Key << "noise_power_dbfs" << YAML::Value << sim.noise_power_dbfs;
    out << YAML::Key << "snr_control_enable" << YAML::Value << sim.snr_control_enable;
    out << YAML::Key << "target_snr_db" << YAML::Value << sim.target_snr_db;
    out << YAML::Key << "control_port" << YAML::Value << sim.control_port;
    out << YAML::Key << "cfo_hz" << YAML::Value << sim.cfo_hz;
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

inline void expose_section_config_keys(
    YAML::Node& config,
    const char* section_key,
    std::initializer_list<const char*> keys) {
    YAML::Node section = config[section_key];
    if (!section || !section.IsMap()) {
        return;
    }
    for (const char* key : keys) {
        if (!config[key] && section[key]) {
            config[key] = section[key];
        }
    }
}

inline void expose_editor_sectioned_config(YAML::Node& config) {
    expose_section_config_keys(config, "radio", {
        "radio_backend",
    });
    expose_section_config_keys(config, "ofdm_frame", {
        "fft_size", "cp_length", "sync_pos", "enable_sec_sync_symbol",
        "enable_cfo_training_sequence", "cfo_training_period_samples",
        "num_symbols", "sensing_symbol_num", "frame_queue_size", "zc_root",
        "pilot_positions", "midframe_pilot_symbols", "midframe_pilot_seed",
    });
    expose_section_config_keys(config, "cuda", {
        "cuda_mod_pipeline_slots", "cuda_demod_pipeline_slots",
        "cuda_ldpc_decoder_backend", "cuda_ldpc_worker_buffers",
        "cuda_ldpc_cross_frame_flush_frames", "cuda_ldpc_cross_frame_flush_us",
    });
    expose_section_config_keys(config, "sensing", {
        "enable_bi_sensing", "sensing_on_wire_format", "range_fft_size",
        "doppler_fft_size", "sensing_output_mode",
        "enable_backend_sensing_processing", "sensing_view_range_bins",
        "sensing_view_doppler_bins", "rx_device_args", "rx_clock_source",
        "rx_time_source", "sensing_rx_wire_format", "sensing_rx_channel_count",
        "sensing_rx_channels", "paired_frame_queue_size",
    });
    expose_section_config_keys(config, "rf_sampling", {
        "sample_rate", "bandwidth", "center_freq", "rx_gain", "rx_agc_enable",
        "rx_agc_low_threshold_db", "rx_agc_high_threshold_db",
        "rx_agc_max_step_db", "rx_agc_update_frames",
    });
    expose_section_config_keys(config, "usrp_device", {
        "device_args", "clock_source",
    });
    expose_section_config_keys(config, "clock_time", {
        "clock_source", "time_source",
    });
    expose_section_config_keys(config, "downlink", {
        "tx_gain", "tx_channel", "tx_device_args", "tx_clock_source",
        "tx_time_source", "wire_format_tx", "rx_channel",
        "downlink_rx_wire_format", "equalizer_mode", "channel_tracking_mode",
        "equalizer_mag_floor", "channel_tracking_min_pilot_snr",
        "downlink_cpu_cores",
    });
    expose_section_config_keys(config, "downlink_pipeline", {
        "tx_circular_buffer_size", "data_packet_buffer_size",
        "downlink_cpu_cores",
    });
    expose_section_config_keys(config, "uplink", {
        "enable_uplink", "duplex_mode", "uplink_idle_waveform",
        "bs_dl_ul_timing_diff", "ue_timing_advance", "equalizer_mode",
        "channel_tracking_mode", "equalizer_mag_floor",
        "channel_tracking_min_pilot_snr", "rx_gain", "uplink_rx_channel",
        "uplink_rx_device_args", "uplink_rx_wire_format",
        "uplink_rx_clock_source", "uplink_rx_time_source", "tx_gain",
        "tx_channel", "wire_format_tx", "uplink_cpu_cores",
    });
    expose_section_config_keys(config, "sync_tracking", {
        "sync_queue_size", "sync_cfo_alias_search_range_hz", "reset_hold_s",
        "software_sync", "predictive_delay", "hardware_sync",
        "hardware_sync_tty", "ocxo_pi_kp_fast", "ocxo_pi_ki_fast",
        "ocxo_pi_kp_slow", "ocxo_pi_ki_slow",
        "ocxo_pi_switch_abs_error_ppm", "ocxo_pi_switch_hold_s",
        "ocxo_pi_max_step_fast_ppm", "ocxo_pi_max_step_slow_ppm",
        "akf_enable", "akf_bootstrap_frames", "akf_innovation_window",
        "akf_max_lag", "akf_adapt_interval", "akf_gate_sigma",
        "akf_tikhonov_lambda", "akf_update_smooth", "akf_q_wf_min",
        "akf_q_wf_max", "akf_q_rw_min", "akf_q_rw_max", "akf_r_min",
        "akf_r_max", "desired_peak_pos", "sensing_symbol_stride",
    });
    expose_section_config_keys(config, "measurement", {
        "measurement_enable", "measurement_mode", "measurement_run_id",
        "measurement_output_dir", "measurement_payload_bytes",
        "measurement_prbs_seed", "measurement_packets_per_point",
        "measurement_max_packets_per_frame",
    });
    expose_section_config_keys(config, "network_output", {
        "udp_input_ip", "udp_input_port", "mono_sensing_ip",
        "mono_sensing_port", "mono_sensing_output_enabled",
        "uplink_channel_ip", "uplink_channel_port", "uplink_pdf_ip",
        "uplink_pdf_port", "uplink_constellation_ip",
        "uplink_constellation_port", "bi_sensing_output_enabled",
        "bi_sensing_ip", "bi_sensing_port", "control_port", "channel_ip",
        "channel_port", "pdf_ip", "pdf_port", "constellation_ip",
        "constellation_port", "udp_output_ip", "udp_output_port",
    });
    expose_section_config_keys(config, "runtime", {
        "default_out_ip", "vofa_debug_ip", "vofa_debug_port",
        "main_cpu_core", "profiling_modules",
    });
    expose_section_config_keys(config, "resource_preview", {
        "data_resource_blocks", "sensing_mask_blocks",
    });
}

/**
 * @brief Parse the radio_backend + simulation block from a YAML node.
 *
 * Shared by the BS and UE config loaders. Missing keys keep their
 * struct defaults so existing (hardware) configs are unaffected.
 */
inline void load_simulation_config(const YAML::Node& config, Config& cfg) {
    if (config["radio_backend"]) cfg.radio_backend = config["radio_backend"].as<std::string>();
    if (config["simulation"] && config["simulation"].IsMap()) {
        const YAML::Node& sim_node = config["simulation"];
        SimConfig& sim = cfg.simulation;
        if (sim_node["session"]) sim.session = sim_node["session"].as<std::string>();
        if (sim_node["enable_comm_rx"]) sim.enable_comm_rx = sim_node["enable_comm_rx"].as<bool>();
        if (sim_node["enable_sensing_rx"]) sim.enable_sensing_rx = sim_node["enable_sensing_rx"].as<bool>();
        if (sim_node["enable_uplink"]) sim.enable_uplink = sim_node["enable_uplink"].as<bool>();
        if (sim_node["noise_power_dbfs"]) sim.noise_power_dbfs = sim_node["noise_power_dbfs"].as<double>();
        if (sim_node["snr_control_enable"]) sim.snr_control_enable = sim_node["snr_control_enable"].as<bool>();
        if (sim_node["target_snr_db"]) sim.target_snr_db = sim_node["target_snr_db"].as<double>();
        if (sim_node["control_port"]) sim.control_port = sim_node["control_port"].as<int>();
        if (sim_node["cfo_hz"]) sim.cfo_hz = sim_node["cfo_hz"].as<double>();
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
        if (sim_node["uplink_multipath_taps"] && sim_node["uplink_multipath_taps"].IsSequence()) {
            load_tap_seq(sim_node["uplink_multipath_taps"], sim.uplink_multipath_taps);
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
//   duplex_mode: tdd | fdd
//   uplink:
//     symbol_start: <size_t>     # TDD: first uplink OFDM symbol
//     symbol_count: <size_t>     # TDD: uplink symbol count (0 => uplink off)
//     guard_symbols: <size_t>    # TDD: guard symbols at the DL->UL boundary
//     center_freq: <double>      # FDD: uplink carrier (Hz)
//   bs_dl_ul_timing_diff: <int>  # BS startup default (samples)
//   ue_timing_advance: <int>     # UE startup default (samples)
inline void load_duplex_config(const YAML::Node& config, Config& cfg) {
    if (config["enable_uplink"]) {
        cfg.enable_uplink = config["enable_uplink"].as<bool>();
    }
    if (config["duplex_mode"]) {
        const std::string raw = config["duplex_mode"].as<std::string>();
        DuplexMode mode = cfg.duplex.mode;
        if (parse_duplex_mode_string(raw, mode)) {
            cfg.duplex.mode = mode;
        }
    }
    if (config["uplink"] && config["uplink"].IsMap()) {
        const YAML::Node& ul = config["uplink"];
        if (ul["symbol_start"]) cfg.duplex.ul_symbol_start = ul["symbol_start"].as<size_t>();
        if (ul["symbol_count"]) cfg.duplex.ul_symbol_count = ul["symbol_count"].as<size_t>();
        if (ul["guard_symbols"]) cfg.duplex.ul_guard_symbols = ul["guard_symbols"].as<size_t>();
        if (ul["center_freq"]) cfg.duplex.ul_center_freq = ul["center_freq"].as<double>();
        if (ul["udp_input_ip"]) cfg.ul_udp_input_ip = ul["udp_input_ip"].as<std::string>();
        if (ul["udp_input_port"]) cfg.ul_udp_input_port = ul["udp_input_port"].as<int>();
        if (ul["udp_output_ip"]) cfg.ul_udp_output_ip = ul["udp_output_ip"].as<std::string>();
        if (ul["udp_output_port"]) cfg.ul_udp_output_port = ul["udp_output_port"].as<int>();
    }
    if (config["bs_dl_ul_timing_diff"]) {
        cfg.bs_dl_ul_timing_diff = config["bs_dl_ul_timing_diff"].as<int32_t>();
    }
    if (config["ue_timing_advance"]) {
        cfg.ue_timing_advance = config["ue_timing_advance"].as<int32_t>();
    }
    if (config["uplink_idle_waveform"]) {
        cfg.uplink_idle_waveform = normalize_uplink_idle_waveform_string(
            config["uplink_idle_waveform"].as<std::string>());
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
    if (cfg.sample_rate <= 0.0) return 1.0;
    return static_cast<double>(cfg.samples_per_frame()) / cfg.sample_rate;
}

inline uint32_t reset_hold_frames_from_cfg(const Config& cfg) {
    const double frame_duration_s = frame_duration_from_cfg(cfg);
    if (frame_duration_s <= 0.0) return 1;
    const double hold_frames = std::ceil(cfg.reset_hold_s / frame_duration_s);
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

inline std::optional<size_t> downlink_core_from_hint(const Config& cfg, size_t hint) {
    return core_from_list_hint(cfg.downlink_cpu_cores, hint);
}

inline std::optional<size_t> uplink_core_from_hint(const Config& cfg, size_t hint) {
    return core_from_list_hint(cfg.uplink_cpu_cores, hint);
}

inline std::optional<size_t> main_thread_core(const Config& cfg) {
    return configured_core_to_optional(cfg.main_cpu_core);
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

namespace config_detail {
inline bool reject_legacy_key(const YAML::Node& config, const char* old_key, const char* new_key) {
    if (config[old_key]) {
        LOG_G_ERROR() << "Legacy YAML key '" << old_key
                      << "' is no longer supported. Use '" << new_key << "' instead.";
        return false;
    }
    return true;
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
    if (!cfg.data_resource_blocks_configured) {
        return;
    }
    emit_resource_blocks_yaml(out, "data_resource_blocks", cfg.data_resource_blocks);
}

inline void emit_sensing_mask_blocks_yaml(YAML::Emitter& out, const Config& cfg) {
    if (cfg.sensing_mask_blocks.empty()) {
        return;
    }
    emit_resource_blocks_yaml(out, "sensing_mask_blocks", cfg.sensing_mask_blocks);
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
        LOG_G_ERROR() << context_name << " key '" << key << "' must be a YAML sequence.";
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
            LOG_G_ERROR() << context_name << ' ' << key << "[" << idx
                          << "] must be a YAML map.";
            return false;
        }
        if (!node["symbol_start"] || !node["symbol_count"] ||
            !node["subcarrier_start"] || !node["subcarrier_count"]) {
            LOG_G_ERROR() << context_name << ' ' << key << "[" << idx
                          << "] must define symbol_start, symbol_count, subcarrier_start, and subcarrier_count.";
            return false;
        }

        DataResourceBlock block;
        if (std::strcmp(key, "data_resource_blocks") == 0 && node["kind"]) {
            const std::string raw_kind = node["kind"].as<std::string>();
            if (!parse_data_resource_block_kind_string(raw_kind, block.kind)) {
                LOG_G_ERROR() << context_name << ' ' << key << "[" << idx
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
            cfg.data_resource_blocks,
            config,
            "data_resource_blocks",
            context_name,
            &key_present)) {
        return false;
    }
    cfg.data_resource_blocks_configured = key_present;
    return true;
}

inline bool load_sensing_mask_blocks_from_yaml(Config& cfg, const YAML::Node& config, const char* context_name) {
    return load_resource_blocks_from_yaml(
        cfg.sensing_mask_blocks,
        config,
        "sensing_mask_blocks",
        context_name);
}
} // namespace config_detail

inline Config make_default_bs_config() {
    Config cfg;
    cfg.fft_size = 1024;
    cfg.cp_length = 128;
    cfg.sync_pos = 1;
    cfg.enable_sec_sync_symbol = false;
    cfg.enable_cfo_training_sequence = false;
    cfg.cfo_training_period_samples = 16;
    cfg.sample_rate = 50e6;
    cfg.bandwidth = 50e6;
    cfg.center_freq = 2.4e9;
    cfg.tx_gain = 30.0;
    cfg.tx_channel = 0;
    cfg.zc_root = 29;
    cfg.pilot_positions = {571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451};
    cfg.midframe_pilot_symbols = {};
    cfg.midframe_pilot_seed = 0x4D46504Cu;
    cfg.num_symbols = 100;
    cfg.sensing_output_mode = kSensingOutputModeDense;
    cfg.sensing_on_wire_format = SensingOnWireFormat::ComplexFloat32;
    cfg.enable_backend_sensing_processing = false;
    cfg.cuda_mod_pipeline_slots = 2;
    cfg.mono_sensing_output_enabled = true;
    cfg.mono_sensing_ip = "";
    cfg.mono_sensing_port = 8888;
    cfg.control_port = 9999;
    cfg.uplink_channel_ip = "0.0.0.0";
    cfg.uplink_channel_port = 12358;
    cfg.uplink_pdf_ip = "0.0.0.0";
    cfg.uplink_pdf_port = 12359;
    cfg.uplink_constellation_ip = "0.0.0.0";
    cfg.uplink_constellation_port = 12356;
    cfg.sensing_rx_channel_count = 1;
    cfg.sensing_symbol_stride = 20;
    cfg.tx_circular_buffer_size = 8;
    cfg.data_packet_buffer_size = 32;
    cfg.paired_frame_queue_size = 16;
    cfg.udp_input_ip = "0.0.0.0";
    cfg.udp_input_port = 50000;
    cfg.radio_backend = "uhd";
    cfg.simulation = SimConfig{};
    cfg.measurement_enable = false;
    cfg.measurement_mode = "";
    cfg.measurement_run_id = "";
    cfg.measurement_output_dir = "";
    cfg.measurement_payload_bytes = 1024;
    cfg.measurement_prbs_seed = 0x5A;
    cfg.measurement_packets_per_point = 1;
    cfg.measurement_max_packets_per_frame = 1;
    return cfg;
}

inline bool save_bs_config_to_yaml(const Config& cfg, const std::string& filepath) {
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "fft_size" << YAML::Value << cfg.fft_size;
    out << YAML::Key << "range_fft_size" << YAML::Value << cfg.range_fft_size;
    out << YAML::Key << "doppler_fft_size" << YAML::Value << cfg.doppler_fft_size;
    out << YAML::Key << "sensing_view_range_bins" << YAML::Value << cfg.sensing_view_range_bins;
    out << YAML::Key << "sensing_view_doppler_bins" << YAML::Value << cfg.sensing_view_doppler_bins;
    out << YAML::Key << "cp_length" << YAML::Value << cfg.cp_length;
    out << YAML::Key << "sync_pos" << YAML::Value << cfg.sync_pos;
    out << YAML::Key << "enable_sec_sync_symbol" << YAML::Value << cfg.enable_sec_sync_symbol;
    out << YAML::Key << "enable_cfo_training_sequence" << YAML::Value
        << cfg.enable_cfo_training_sequence;
    out << YAML::Key << "cfo_training_period_samples" << YAML::Value
        << cfg.cfo_training_period_samples;
    out << YAML::Key << "sample_rate" << YAML::Value << cfg.sample_rate;
    out << YAML::Key << "bandwidth" << YAML::Value << cfg.bandwidth;
    out << YAML::Key << "center_freq" << YAML::Value << cfg.center_freq;
    out << YAML::Key << "tx_gain" << YAML::Value << cfg.tx_gain;
    out << YAML::Key << "tx_channel" << YAML::Value << cfg.tx_channel;
    out << YAML::Key << "rx_gain" << YAML::Value << cfg.rx_gain;
    out << YAML::Key << "rx_channel" << YAML::Value << cfg.rx_channel;
    out << YAML::Key << "zc_root" << YAML::Value << cfg.zc_root;
    out << YAML::Key << "num_symbols" << YAML::Value << cfg.num_symbols;
    out << YAML::Key << "sensing_symbol_num" << YAML::Value << cfg.sensing_symbol_num;
    out << YAML::Key << "sensing_output_mode" << YAML::Value << cfg.sensing_output_mode;
    out << YAML::Key << "sensing_on_wire_format" << YAML::Value
        << sensing_on_wire_format_to_string(cfg.sensing_on_wire_format);
    out << YAML::Key << "enable_backend_sensing_processing" << YAML::Value
        << cfg.enable_backend_sensing_processing;
    out << YAML::Key << "cuda_mod_pipeline_slots" << YAML::Value << cfg.cuda_mod_pipeline_slots;
    out << YAML::Key << "device_args" << YAML::Value << cfg.device_args;
    out << YAML::Key << "tx_device_args" << YAML::Value << cfg.tx_device_args;
    out << YAML::Key << "rx_device_args" << YAML::Value << cfg.rx_device_args;
    out << YAML::Key << "clock_source" << YAML::Value << cfg.clocksource;
    out << YAML::Key << "time_source" << YAML::Value << cfg.timesource;
    out << YAML::Key << "tx_clock_source" << YAML::Value << cfg.tx_clock_source;
    out << YAML::Key << "tx_time_source" << YAML::Value << cfg.tx_time_source;
    out << YAML::Key << "rx_clock_source" << YAML::Value << cfg.rx_clock_source;
    out << YAML::Key << "rx_time_source" << YAML::Value << cfg.rx_time_source;
    out << YAML::Key << "control_port" << YAML::Value << cfg.control_port;
    out << YAML::Key << "wire_format_tx" << YAML::Value << cfg.wire_format_tx;
    out << YAML::Key << "uplink_rx_wire_format" << YAML::Value << cfg.uplink_rx_wire_format;
    out << YAML::Key << "sensing_rx_wire_format" << YAML::Value << cfg.sensing_rx_wire_format;
    out << YAML::Key << "udp_input_ip" << YAML::Value << cfg.udp_input_ip;
    out << YAML::Key << "udp_input_port" << YAML::Value << cfg.udp_input_port;
    out << YAML::Key << "mono_sensing_output_enabled" << YAML::Value << cfg.mono_sensing_output_enabled;
    out << YAML::Key << "mono_sensing_ip" << YAML::Value << cfg.mono_sensing_ip;
    out << YAML::Key << "mono_sensing_port" << YAML::Value << cfg.mono_sensing_port;
    out << YAML::Key << "uplink_channel_ip" << YAML::Value << cfg.uplink_channel_ip;
    out << YAML::Key << "uplink_channel_port" << YAML::Value << cfg.uplink_channel_port;
    out << YAML::Key << "uplink_pdf_ip" << YAML::Value << cfg.uplink_pdf_ip;
    out << YAML::Key << "uplink_pdf_port" << YAML::Value << cfg.uplink_pdf_port;
    out << YAML::Key << "uplink_constellation_ip" << YAML::Value << cfg.uplink_constellation_ip;
    out << YAML::Key << "uplink_constellation_port" << YAML::Value << cfg.uplink_constellation_port;
    out << YAML::Key << "sensing_symbol_stride" << YAML::Value << cfg.sensing_symbol_stride;
    out << YAML::Key << "sensing_rx_channel_count" << YAML::Value << cfg.sensing_rx_channel_count;
    out << YAML::Key << "sensing_rx_channels" << YAML::Value << YAML::BeginSeq;
    for (const auto& ch : cfg.sensing_rx_channels) {
        out << YAML::BeginMap;
        out << YAML::Key << "usrp_channel" << YAML::Value << ch.usrp_channel;
        out << YAML::Key << "device_args" << YAML::Value << ch.device_args;
        out << YAML::Key << "clock_source" << YAML::Value << ch.clock_source;
        out << YAML::Key << "time_source" << YAML::Value << ch.time_source;
        out << YAML::Key << "wire_format" << YAML::Value << ch.wire_format;
        out << YAML::Key << "rx_gain" << YAML::Value << ch.rx_gain;
        out << YAML::Key << "alignment" << YAML::Value << ch.alignment;
        out << YAML::Key << "rx_antenna" << YAML::Value << ch.rx_antenna;
        out << YAML::Key << "enable_system_delay_estimation" << YAML::Value << ch.enable_system_delay_estimation;
        out << YAML::Key << "enable_sensing_output" << YAML::Value << ch.enable_sensing_output;
        out << YAML::Key << "rx_cpu_core" << YAML::Value << ch.rx_cpu_core;
        out << YAML::Key << "processing_cpu_core" << YAML::Value << ch.processing_cpu_core;
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;
    emit_simulation_config(out, cfg);
    out << YAML::Key << "enable_uplink" << YAML::Value << cfg.enable_uplink;
    out << YAML::Key << "measurement_enable" << YAML::Value << cfg.measurement_enable;
    out << YAML::Key << "measurement_mode" << YAML::Value << cfg.measurement_mode;
    out << YAML::Key << "measurement_run_id" << YAML::Value << cfg.measurement_run_id;
    out << YAML::Key << "measurement_output_dir" << YAML::Value << cfg.measurement_output_dir;
    out << YAML::Key << "measurement_payload_bytes" << YAML::Value << cfg.measurement_payload_bytes;
    out << YAML::Key << "measurement_prbs_seed" << YAML::Value << cfg.measurement_prbs_seed;
    out << YAML::Key << "measurement_packets_per_point" << YAML::Value << cfg.measurement_packets_per_point;
    out << YAML::Key << "measurement_max_packets_per_frame" << YAML::Value << cfg.measurement_max_packets_per_frame;
    out << YAML::Key << "tx_circular_buffer_size" << YAML::Value << cfg.tx_circular_buffer_size;
    out << YAML::Key << "data_packet_buffer_size" << YAML::Value << cfg.data_packet_buffer_size;
    out << YAML::Key << "paired_frame_queue_size" << YAML::Value << cfg.paired_frame_queue_size;
    out << YAML::Key << "downlink_cpu_cores" << YAML::Value << YAML::Flow << cfg.downlink_cpu_cores;
    out << YAML::Key << "uplink_cpu_cores" << YAML::Value << YAML::Flow << cfg.uplink_cpu_cores;
    out << YAML::Key << "main_cpu_core" << YAML::Value << cfg.main_cpu_core;
    out << YAML::Key << "profiling_modules" << YAML::Value << cfg.profiling_modules;
    out << YAML::Key << "pilot_positions" << YAML::Value << YAML::Flow << cfg.pilot_positions;
    out << YAML::Key << "midframe_pilot_symbols" << YAML::Value << YAML::Flow
        << cfg.midframe_pilot_symbols;
    out << YAML::Key << "midframe_pilot_seed" << YAML::Value << cfg.midframe_pilot_seed;
    config_detail::emit_data_resource_blocks_yaml(out, cfg);
    config_detail::emit_sensing_mask_blocks_yaml(out, cfg);
    out << YAML::EndMap;

    std::ofstream fout(filepath);
    if (!fout) {
        LOG_G_ERROR() << "Error: Cannot write to config file: " << filepath;
        return false;
    }
    fout << out.c_str();
    fout.close();
    return true;
}

inline bool load_bs_config_from_yaml(Config& cfg, const std::string& filepath) {
    if (!path_exists(filepath)) {
        return false;
    }
    try {
        YAML::Node config = YAML::LoadFile(filepath);
        expose_editor_sectioned_config(config);
        const bool has_range_fft_size = static_cast<bool>(config["range_fft_size"]);
        if (!config_detail::reject_legacy_key(config, "mod_udp_ip", "udp_input_ip")) return false;
        if (!config_detail::reject_legacy_key(config, "mod_udp_port", "udp_input_port")) return false;
        if (!config_detail::reject_legacy_key(config, "sensing_ip", "mono_sensing_ip")) return false;
        if (!config_detail::reject_legacy_key(config, "sensing_port", "mono_sensing_port")) return false;

        if (config["fft_size"]) cfg.fft_size = config["fft_size"].as<size_t>();
        if (has_range_fft_size) cfg.range_fft_size = config["range_fft_size"].as<size_t>();
        if (config["doppler_fft_size"]) cfg.doppler_fft_size = config["doppler_fft_size"].as<size_t>();
        if (config["sensing_view_range_bins"]) {
            cfg.sensing_view_range_bins = config["sensing_view_range_bins"].as<size_t>();
        }
        if (config["sensing_view_doppler_bins"]) {
            cfg.sensing_view_doppler_bins = config["sensing_view_doppler_bins"].as<size_t>();
        }
        if (config["cp_length"]) cfg.cp_length = config["cp_length"].as<size_t>();
        if (config["sync_pos"]) cfg.sync_pos = config["sync_pos"].as<size_t>();
        if (config["enable_sec_sync_symbol"]) {
            cfg.enable_sec_sync_symbol = config["enable_sec_sync_symbol"].as<bool>();
        }
        if (config["enable_cfo_training_sequence"]) {
            cfg.enable_cfo_training_sequence =
                config["enable_cfo_training_sequence"].as<bool>();
        }
        if (config["cfo_training_period_samples"]) {
            cfg.cfo_training_period_samples =
                config["cfo_training_period_samples"].as<size_t>();
        }
        if (config["sample_rate"]) cfg.sample_rate = config["sample_rate"].as<double>();
        if (config["bandwidth"]) cfg.bandwidth = config["bandwidth"].as<double>();
        if (config["center_freq"]) cfg.center_freq = config["center_freq"].as<double>();
        if (config["tx_gain"]) cfg.tx_gain = config["tx_gain"].as<double>();
        if (config["tx_channel"]) cfg.tx_channel = config["tx_channel"].as<uint32_t>();
        if (config["rx_gain"]) cfg.rx_gain = config["rx_gain"].as<double>();
        if (config["rx_channel"]) cfg.rx_channel = config["rx_channel"].as<size_t>();
        if (config["zc_root"]) cfg.zc_root = config["zc_root"].as<int>();
        if (config["num_symbols"]) cfg.num_symbols = config["num_symbols"].as<size_t>();
        if (config["sensing_symbol_num"]) cfg.sensing_symbol_num = config["sensing_symbol_num"].as<size_t>();
        if (config["sensing_output_mode"]) {
            cfg.sensing_output_mode = config["sensing_output_mode"].as<std::string>();
        }
        if (config["sensing_on_wire_format"]) {
            cfg.sensing_on_wire_format = parse_sensing_on_wire_format_string(
                config["sensing_on_wire_format"].as<std::string>());
        }
        if (config["enable_backend_sensing_processing"]) {
            cfg.enable_backend_sensing_processing =
                config["enable_backend_sensing_processing"].as<bool>();
        }
        if (config["cuda_mod_pipeline_slots"]) {
            cfg.cuda_mod_pipeline_slots = config["cuda_mod_pipeline_slots"].as<size_t>();
        }
        if (config["device_args"]) cfg.device_args = config["device_args"].as<std::string>();
        if (config["tx_device_args"]) cfg.tx_device_args = config["tx_device_args"].as<std::string>();
        if (config["rx_device_args"]) cfg.rx_device_args = config["rx_device_args"].as<std::string>();
        if (config["clock_source"]) cfg.clocksource = config["clock_source"].as<std::string>();
        if (config["time_source"]) cfg.timesource = config["time_source"].as<std::string>();
        if (config["tx_clock_source"]) cfg.tx_clock_source = config["tx_clock_source"].as<std::string>();
        if (config["tx_time_source"]) cfg.tx_time_source = config["tx_time_source"].as<std::string>();
        if (config["rx_clock_source"]) cfg.rx_clock_source = config["rx_clock_source"].as<std::string>();
        if (config["rx_time_source"]) cfg.rx_time_source = config["rx_time_source"].as<std::string>();
        if (config["control_port"]) cfg.control_port = config["control_port"].as<int>();
        if (config["wire_format_tx"]) cfg.wire_format_tx = config["wire_format_tx"].as<std::string>();
        if (config["uplink_rx_wire_format"]) {
            cfg.uplink_rx_wire_format = config["uplink_rx_wire_format"].as<std::string>();
        }
        if (config["sensing_rx_wire_format"]) {
            cfg.sensing_rx_wire_format = config["sensing_rx_wire_format"].as<std::string>();
        }
        if (config["udp_input_ip"]) cfg.udp_input_ip = config["udp_input_ip"].as<std::string>();
        if (config["udp_input_port"]) cfg.udp_input_port = config["udp_input_port"].as<int>();
        if (config["mono_sensing_output_enabled"]) {
            cfg.mono_sensing_output_enabled = config["mono_sensing_output_enabled"].as<bool>();
        }
        if (config["mono_sensing_ip"]) cfg.mono_sensing_ip = config["mono_sensing_ip"].as<std::string>();
        if (config["mono_sensing_port"]) cfg.mono_sensing_port = config["mono_sensing_port"].as<int>();
        if (config["uplink_channel_ip"]) cfg.uplink_channel_ip = config["uplink_channel_ip"].as<std::string>();
        if (config["uplink_channel_port"]) cfg.uplink_channel_port = config["uplink_channel_port"].as<int>();
        if (config["uplink_pdf_ip"]) cfg.uplink_pdf_ip = config["uplink_pdf_ip"].as<std::string>();
        if (config["uplink_pdf_port"]) cfg.uplink_pdf_port = config["uplink_pdf_port"].as<int>();
        if (config["uplink_constellation_ip"]) {
            cfg.uplink_constellation_ip = config["uplink_constellation_ip"].as<std::string>();
        }
        if (config["uplink_constellation_port"]) {
            cfg.uplink_constellation_port = config["uplink_constellation_port"].as<int>();
        }
        if (config["sensing_symbol_stride"]) {
            cfg.sensing_symbol_stride = config["sensing_symbol_stride"].as<size_t>();
        }
        if (config["tx_circular_buffer_size"]) cfg.tx_circular_buffer_size = config["tx_circular_buffer_size"].as<size_t>();
        if (config["data_packet_buffer_size"]) cfg.data_packet_buffer_size = config["data_packet_buffer_size"].as<size_t>();
        if (config["paired_frame_queue_size"]) cfg.paired_frame_queue_size = config["paired_frame_queue_size"].as<size_t>();
        const bool has_sensing_count_key = static_cast<bool>(config["sensing_rx_channel_count"]);
        if (has_sensing_count_key) {
            cfg.sensing_rx_channel_count = config["sensing_rx_channel_count"].as<uint32_t>();
        }
        if (config["sensing_rx_channels"] && config["sensing_rx_channels"].IsSequence()) {
            cfg.sensing_rx_channels.clear();
            for (const auto& node : config["sensing_rx_channels"]) {
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
                    ch.enable_system_delay_estimation = node["enable_system_delay_estimation"].as<bool>();
                }
                if (node["enable_sensing_output"]) {
                    ch.enable_sensing_output = node["enable_sensing_output"].as<bool>();
                }
                config_detail::load_optional_int_yaml(node["rx_cpu_core"], ch.rx_cpu_core);
                config_detail::load_optional_int_yaml(
                    node["processing_cpu_core"], ch.processing_cpu_core);
                cfg.sensing_rx_channels.push_back(ch);
            }
            if (!has_sensing_count_key) {
                cfg.sensing_rx_channel_count = static_cast<uint32_t>(cfg.sensing_rx_channels.size());
            }
        }
        load_simulation_config(config, cfg);
        load_duplex_config(config, cfg);
        if (config["measurement_enable"]) cfg.measurement_enable = config["measurement_enable"].as<bool>();
        if (config["measurement_mode"]) cfg.measurement_mode = config["measurement_mode"].as<std::string>();
        if (config["measurement_run_id"]) cfg.measurement_run_id = config["measurement_run_id"].as<std::string>();
        if (config["measurement_output_dir"]) cfg.measurement_output_dir = config["measurement_output_dir"].as<std::string>();
        if (config["measurement_payload_bytes"]) cfg.measurement_payload_bytes = config["measurement_payload_bytes"].as<size_t>();
        if (config["measurement_prbs_seed"]) cfg.measurement_prbs_seed = config["measurement_prbs_seed"].as<uint32_t>();
        if (config["measurement_packets_per_point"]) cfg.measurement_packets_per_point = config["measurement_packets_per_point"].as<uint32_t>();
        if (config["measurement_max_packets_per_frame"]) {
            cfg.measurement_max_packets_per_frame = config["measurement_max_packets_per_frame"].as<size_t>();
        }
        if (config["profiling_modules"]) cfg.profiling_modules = config["profiling_modules"].as<std::string>();
        if (config["pilot_positions"]) cfg.pilot_positions = config["pilot_positions"].as<std::vector<size_t>>();
        if (config["midframe_pilot_symbols"]) {
            cfg.midframe_pilot_symbols = config["midframe_pilot_symbols"].as<std::vector<size_t>>();
        }
        if (config["midframe_pilot_seed"]) {
            cfg.midframe_pilot_seed = config["midframe_pilot_seed"].as<uint32_t>();
        }
        if (!config_detail::load_data_resource_blocks_from_yaml(cfg, config, "BS config")) {
            return false;
        }
        if (!config_detail::load_sensing_mask_blocks_from_yaml(cfg, config, "BS config")) {
            return false;
        }
        if (config["downlink_cpu_cores"]) {
            cfg.downlink_cpu_cores = config["downlink_cpu_cores"].as<std::vector<int>>();
        }
        if (config["uplink_cpu_cores"]) {
            cfg.uplink_cpu_cores = config["uplink_cpu_cores"].as<std::vector<int>>();
        }
        if (config["main_cpu_core"]) cfg.main_cpu_core = config["main_cpu_core"].as<int>();
        return true;
    } catch (const YAML::Exception& e) {
        LOG_G_ERROR() << "Error parsing YAML config: " << e.what();
        return false;
    }
}

inline void normalize_bs_sensing_channels(Config& cfg) {
    normalize_sensing_fft_sizes(cfg, "BS sensing");
    normalize_sensing_view_bins(cfg, "BS sensing");
    if (cfg.cuda_mod_pipeline_slots == 0) {
        LOG_G_WARN() << "cuda_mod_pipeline_slots=0 is invalid. Clamping to 1.";
        cfg.cuda_mod_pipeline_slots = 1;
    }
    if (cfg.tx_circular_buffer_size == 0) {
        LOG_G_WARN() << "tx_circular_buffer_size=0 is invalid. Clamping to 1.";
        cfg.tx_circular_buffer_size = 1;
    }
    if (cfg.data_packet_buffer_size == 0) {
        LOG_G_WARN() << "data_packet_buffer_size=0 is invalid. Clamping to 1.";
        cfg.data_packet_buffer_size = 1;
    }
    if (cfg.paired_frame_queue_size == 0) {
        LOG_G_WARN() << "paired_frame_queue_size=0 is invalid. Clamping to 1.";
        cfg.paired_frame_queue_size = 1;
    }
    if (cfg.sensing_symbol_stride == 0) {
        LOG_G_WARN() << "sensing_symbol_stride=0 is invalid. Clamping to 1.";
        cfg.sensing_symbol_stride = 1;
    }
    if (cfg.measurement_enable) {
        if (cfg.measurement_mode.empty()) {
            cfg.measurement_mode = "internal_prbs";
        }
        if (!measurement_mode_enabled(cfg)) {
            LOG_G_WARN() << "Unsupported BS measurement_mode='"
                         << cfg.measurement_mode << "'. Disabling measurement mode.";
            cfg.measurement_enable = false;
        }
        if (cfg.measurement_payload_bytes < measurement_payload_header_size()) {
            LOG_G_WARN() << "measurement_payload_bytes=" << cfg.measurement_payload_bytes
                         << " is smaller than the measurement header. Clamping to "
                         << measurement_payload_header_size() << '.';
            cfg.measurement_payload_bytes = measurement_payload_header_size();
        }
        if (cfg.measurement_packets_per_point == 0) {
            LOG_G_WARN() << "measurement_packets_per_point=0 is invalid. Clamping to 1.";
            cfg.measurement_packets_per_point = 1;
        }
    }
    std::sort(cfg.midframe_pilot_symbols.begin(), cfg.midframe_pilot_symbols.end());
    cfg.midframe_pilot_symbols.erase(
        std::unique(cfg.midframe_pilot_symbols.begin(), cfg.midframe_pilot_symbols.end()),
        cfg.midframe_pilot_symbols.end());
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
        ch.enable_sensing_output = cfg.mono_sensing_output_enabled;
        return ch;
    };
    if (cfg.sensing_rx_channels.empty() && cfg.sensing_rx_channel_count > 0) {
        cfg.sensing_rx_channels.push_back(make_default_ch0());
        for (uint32_t i = 1; i < cfg.sensing_rx_channel_count; ++i) {
            auto ch = make_default_ch0();
            ch.usrp_channel = i;
            cfg.sensing_rx_channels.push_back(ch);
        }
    }

    if (cfg.sensing_rx_channel_count == 0) {
        cfg.sensing_rx_channels.clear();
    } else if (cfg.sensing_rx_channels.size() > cfg.sensing_rx_channel_count) {
        cfg.sensing_rx_channels.resize(cfg.sensing_rx_channel_count);
    } else if (cfg.sensing_rx_channels.size() < cfg.sensing_rx_channel_count) {
        const auto base = cfg.sensing_rx_channels.empty() ? make_default_ch0() : cfg.sensing_rx_channels.front();
        for (size_t i = cfg.sensing_rx_channels.size(); i < cfg.sensing_rx_channel_count; ++i) {
            auto ch = base;
            ch.usrp_channel = static_cast<uint32_t>(base.usrp_channel + i);
            cfg.sensing_rx_channels.push_back(ch);
        }
    }

    if (cfg.mono_sensing_ip.empty()) {
        cfg.mono_sensing_ip = "0.0.0.0";
    }
    if (cfg.uplink_channel_ip.empty()) {
        cfg.uplink_channel_ip = "0.0.0.0";
    }
    if (cfg.uplink_pdf_ip.empty()) {
        cfg.uplink_pdf_ip = "0.0.0.0";
    }
    if (cfg.uplink_constellation_ip.empty()) {
        cfg.uplink_constellation_ip = "0.0.0.0";
    }
    if (cfg.mono_sensing_port <= 0) {
        LOG_G_WARN() << "mono_sensing_port=" << cfg.mono_sensing_port
                     << " is invalid. Falling back to 8888.";
        cfg.mono_sensing_port = 8888;
    }

    cfg.sensing_rx_channel_count = static_cast<uint32_t>(cfg.sensing_rx_channels.size());
}

inline Config make_default_ue_config() {
    Config cfg;
    cfg.fft_size = 1024;
    cfg.cp_length = 128;
    cfg.enable_sec_sync_symbol = false;
    cfg.enable_cfo_training_sequence = false;
    cfg.cfo_training_period_samples = 16;
    cfg.center_freq = 2.4e9;
    cfg.pilot_positions = {571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451};
    cfg.midframe_pilot_symbols = {};
    cfg.midframe_pilot_seed = 0x4D46504Cu;
    cfg.range_fft_size = 1024;
    cfg.doppler_fft_size = 100;
    cfg.num_symbols = 100;
    cfg.sensing_symbol_num = 100;
    cfg.cuda_demod_pipeline_slots = 3;
    cfg.frame_queue_size = 8;
    cfg.sync_queue_size = 8;
    cfg.sync_pos = 1;
    cfg.sync_cfo_alias_search_range_hz = 800000.0;
    cfg.reset_hold_s = 0.5;
    cfg.sample_rate = 50e6;
    cfg.bandwidth = 50e6;
    cfg.rx_gain = 50.0;
    cfg.rx_agc_enable = false;
    cfg.rx_agc_low_threshold_db = 11.0;
    cfg.rx_agc_high_threshold_db = 13.0;
    cfg.rx_agc_max_step_db = 3.0;
    cfg.rx_agc_update_frames = 4;
    cfg.zc_root = 29;
    cfg.device_args = "";
    cfg.enable_bi_sensing = true;
    cfg.bi_sensing_output_enabled = true;
    cfg.bi_sensing_ip = "";
    cfg.bi_sensing_port = 8889;
    cfg.control_port = 10000;
    cfg.channel_ip = "0.0.0.0";
    cfg.channel_port = 12348;
    cfg.pdf_ip = "0.0.0.0";
    cfg.pdf_port = 12349;
    cfg.constellation_ip = "0.0.0.0";
    cfg.constellation_port = 12346;
    cfg.vofa_debug_ip = "";
    cfg.vofa_debug_port = 12347;
    cfg.udp_output_ip = "";
    cfg.software_sync = true;
    cfg.predictive_delay = true;
    cfg.akf_enable = true;
    cfg.akf_bootstrap_frames = 64;
    cfg.akf_innovation_window = 64;
    cfg.akf_max_lag = 4;
    cfg.akf_adapt_interval = 64;
    cfg.akf_gate_sigma = 3.0;
    cfg.akf_tikhonov_lambda = 1e-3;
    cfg.akf_update_smooth = 0.2;
    cfg.akf_q_wf_min = 1e-10;
    cfg.akf_q_wf_max = 1e2;
    cfg.akf_q_rw_min = 1e-12;
    cfg.akf_q_rw_max = 1e1;
    cfg.akf_r_min = 1e-8;
    cfg.akf_r_max = 1e3;
    cfg.equalizer_mode = kEqualizerModeMmse;
    cfg.channel_tracking_mode = kChannelTrackingModePilotPhase;
    cfg.equalizer_mag_floor = 1e-6;
    cfg.channel_tracking_min_pilot_snr = 1e-4;
    cfg.sensing_output_mode = kSensingOutputModeDense;
    cfg.sensing_on_wire_format = SensingOnWireFormat::ComplexFloat32;
    cfg.enable_backend_sensing_processing = false;
    cfg.sensing_symbol_stride = 20;
    cfg.radio_backend = "uhd";
    cfg.simulation = SimConfig{};
    cfg.measurement_enable = false;
    cfg.measurement_mode = "";
    cfg.measurement_run_id = "";
    cfg.measurement_output_dir = "";
    cfg.measurement_payload_bytes = 1024;
    cfg.measurement_prbs_seed = 0x5A;
    cfg.measurement_packets_per_point = 1;
    cfg.measurement_max_packets_per_frame = 1;
    return cfg;
}

inline bool save_ue_config_to_yaml(const Config& cfg, const std::string& filepath) {
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "fft_size" << YAML::Value << cfg.fft_size;
    out << YAML::Key << "cp_length" << YAML::Value << cfg.cp_length;
    out << YAML::Key << "sync_pos" << YAML::Value << cfg.sync_pos;
    out << YAML::Key << "enable_sec_sync_symbol" << YAML::Value << cfg.enable_sec_sync_symbol;
    out << YAML::Key << "enable_cfo_training_sequence" << YAML::Value
        << cfg.enable_cfo_training_sequence;
    out << YAML::Key << "cfo_training_period_samples" << YAML::Value
        << cfg.cfo_training_period_samples;
    out << YAML::Key << "sample_rate" << YAML::Value << cfg.sample_rate;
    out << YAML::Key << "bandwidth" << YAML::Value << cfg.bandwidth;
    out << YAML::Key << "center_freq" << YAML::Value << cfg.center_freq;
    out << YAML::Key << "tx_gain" << YAML::Value << cfg.tx_gain;
    out << YAML::Key << "tx_channel" << YAML::Value << cfg.tx_channel;
    out << YAML::Key << "rx_gain" << YAML::Value << cfg.rx_gain;
    out << YAML::Key << "rx_agc_enable" << YAML::Value << cfg.rx_agc_enable;
    out << YAML::Key << "rx_agc_low_threshold_db" << YAML::Value << cfg.rx_agc_low_threshold_db;
    out << YAML::Key << "rx_agc_high_threshold_db" << YAML::Value << cfg.rx_agc_high_threshold_db;
    out << YAML::Key << "rx_agc_max_step_db" << YAML::Value << cfg.rx_agc_max_step_db;
    out << YAML::Key << "rx_agc_update_frames" << YAML::Value << cfg.rx_agc_update_frames;
    out << YAML::Key << "rx_channel" << YAML::Value << cfg.rx_channel;
    out << YAML::Key << "zc_root" << YAML::Value << cfg.zc_root;
    out << YAML::Key << "num_symbols" << YAML::Value << cfg.num_symbols;
    out << YAML::Key << "sensing_symbol_num" << YAML::Value << cfg.sensing_symbol_num;
    out << YAML::Key << "sensing_output_mode" << YAML::Value << cfg.sensing_output_mode;
    out << YAML::Key << "sensing_on_wire_format" << YAML::Value
        << sensing_on_wire_format_to_string(cfg.sensing_on_wire_format);
    out << YAML::Key << "cuda_demod_pipeline_slots" << YAML::Value << cfg.cuda_demod_pipeline_slots;
    out << YAML::Key << "frame_queue_size" << YAML::Value << cfg.frame_queue_size;
    out << YAML::Key << "sync_queue_size" << YAML::Value << cfg.sync_queue_size;
    out << YAML::Key << "sync_cfo_alias_search_range_hz" << YAML::Value
        << cfg.sync_cfo_alias_search_range_hz;
    out << YAML::Key << "reset_hold_s" << YAML::Value << cfg.reset_hold_s;
    out << YAML::Key << "range_fft_size" << YAML::Value << cfg.range_fft_size;
    out << YAML::Key << "doppler_fft_size" << YAML::Value << cfg.doppler_fft_size;
    out << YAML::Key << "sensing_view_range_bins" << YAML::Value << cfg.sensing_view_range_bins;
    out << YAML::Key << "sensing_view_doppler_bins" << YAML::Value << cfg.sensing_view_doppler_bins;
    out << YAML::Key << "device_args" << YAML::Value << cfg.device_args;
    out << YAML::Key << "clock_source" << YAML::Value << cfg.clocksource;
    out << YAML::Key << "wire_format_tx" << YAML::Value << cfg.wire_format_tx;
    out << YAML::Key << "downlink_rx_wire_format" << YAML::Value << cfg.downlink_rx_wire_format;
    out << YAML::Key << "enable_bi_sensing" << YAML::Value << cfg.enable_bi_sensing;
    out << YAML::Key << "bi_sensing_output_enabled" << YAML::Value << cfg.bi_sensing_output_enabled;
    out << YAML::Key << "bi_sensing_ip" << YAML::Value << cfg.bi_sensing_ip;
    out << YAML::Key << "bi_sensing_port" << YAML::Value << cfg.bi_sensing_port;
    out << YAML::Key << "control_port" << YAML::Value << cfg.control_port;
    out << YAML::Key << "channel_ip" << YAML::Value << cfg.channel_ip;
    out << YAML::Key << "channel_port" << YAML::Value << cfg.channel_port;
    out << YAML::Key << "pdf_ip" << YAML::Value << cfg.pdf_ip;
    out << YAML::Key << "pdf_port" << YAML::Value << cfg.pdf_port;
    out << YAML::Key << "constellation_ip" << YAML::Value << cfg.constellation_ip;
    out << YAML::Key << "constellation_port" << YAML::Value << cfg.constellation_port;
    out << YAML::Key << "vofa_debug_ip" << YAML::Value << cfg.vofa_debug_ip;
    out << YAML::Key << "vofa_debug_port" << YAML::Value << cfg.vofa_debug_port;
    out << YAML::Key << "udp_output_ip" << YAML::Value << cfg.udp_output_ip;
    out << YAML::Key << "udp_output_port" << YAML::Value << cfg.udp_output_port;
    out << YAML::Key << "measurement_enable" << YAML::Value << cfg.measurement_enable;
    out << YAML::Key << "measurement_mode" << YAML::Value << cfg.measurement_mode;
    out << YAML::Key << "measurement_run_id" << YAML::Value << cfg.measurement_run_id;
    out << YAML::Key << "measurement_output_dir" << YAML::Value << cfg.measurement_output_dir;
    out << YAML::Key << "measurement_payload_bytes" << YAML::Value << cfg.measurement_payload_bytes;
    out << YAML::Key << "measurement_prbs_seed" << YAML::Value << cfg.measurement_prbs_seed;
    out << YAML::Key << "measurement_packets_per_point" << YAML::Value << cfg.measurement_packets_per_point;
    out << YAML::Key << "measurement_max_packets_per_frame" << YAML::Value << cfg.measurement_max_packets_per_frame;
    out << YAML::Key << "software_sync" << YAML::Value << cfg.software_sync;
    out << YAML::Key << "predictive_delay" << YAML::Value << cfg.predictive_delay;
    out << YAML::Key << "hardware_sync" << YAML::Value << cfg.hardware_sync;
    out << YAML::Key << "hardware_sync_tty" << YAML::Value << cfg.hardware_sync_tty;
    out << YAML::Key << "downlink_cpu_cores" << YAML::Value << YAML::Flow << cfg.downlink_cpu_cores;
    out << YAML::Key << "uplink_cpu_cores" << YAML::Value << YAML::Flow << cfg.uplink_cpu_cores;
    out << YAML::Key << "main_cpu_core" << YAML::Value << cfg.main_cpu_core;
    out << YAML::Key << "profiling_modules" << YAML::Value << cfg.profiling_modules;
    out << YAML::Key << "default_out_ip" << YAML::Value << cfg.default_out_ip;
    emit_simulation_config(out, cfg);
    out << YAML::Key << "enable_uplink" << YAML::Value << cfg.enable_uplink;
    out << YAML::Key << "uplink_idle_waveform" << YAML::Value << cfg.uplink_idle_waveform;
    out << YAML::Key << "ocxo_pi_kp_fast" << YAML::Value << cfg.ocxo_pi_kp_fast;
    out << YAML::Key << "ocxo_pi_ki_fast" << YAML::Value << cfg.ocxo_pi_ki_fast;
    out << YAML::Key << "ocxo_pi_kp_slow" << YAML::Value << cfg.ocxo_pi_kp_slow;
    out << YAML::Key << "ocxo_pi_ki_slow" << YAML::Value << cfg.ocxo_pi_ki_slow;
    out << YAML::Key << "ocxo_pi_switch_abs_error_ppm" << YAML::Value << cfg.ocxo_pi_switch_abs_error_ppm;
    out << YAML::Key << "ocxo_pi_switch_hold_s" << YAML::Value << cfg.ocxo_pi_switch_hold_s;
    out << YAML::Key << "ocxo_pi_max_step_fast_ppm" << YAML::Value << cfg.ocxo_pi_max_step_fast_ppm;
    out << YAML::Key << "ocxo_pi_max_step_slow_ppm" << YAML::Value << cfg.ocxo_pi_max_step_slow_ppm;
    out << YAML::Key << "akf_enable" << YAML::Value << cfg.akf_enable;
    out << YAML::Key << "akf_bootstrap_frames" << YAML::Value << cfg.akf_bootstrap_frames;
    out << YAML::Key << "akf_innovation_window" << YAML::Value << cfg.akf_innovation_window;
    out << YAML::Key << "akf_max_lag" << YAML::Value << cfg.akf_max_lag;
    out << YAML::Key << "akf_adapt_interval" << YAML::Value << cfg.akf_adapt_interval;
    out << YAML::Key << "akf_gate_sigma" << YAML::Value << cfg.akf_gate_sigma;
    out << YAML::Key << "akf_tikhonov_lambda" << YAML::Value << cfg.akf_tikhonov_lambda;
    out << YAML::Key << "akf_update_smooth" << YAML::Value << cfg.akf_update_smooth;
    out << YAML::Key << "akf_q_wf_min" << YAML::Value << cfg.akf_q_wf_min;
    out << YAML::Key << "akf_q_wf_max" << YAML::Value << cfg.akf_q_wf_max;
    out << YAML::Key << "akf_q_rw_min" << YAML::Value << cfg.akf_q_rw_min;
    out << YAML::Key << "akf_q_rw_max" << YAML::Value << cfg.akf_q_rw_max;
    out << YAML::Key << "akf_r_min" << YAML::Value << cfg.akf_r_min;
    out << YAML::Key << "akf_r_max" << YAML::Value << cfg.akf_r_max;
    out << YAML::Key << "desired_peak_pos" << YAML::Value << cfg.desired_peak_pos;
    out << YAML::Key << "sensing_symbol_stride" << YAML::Value << cfg.sensing_symbol_stride;
    out << YAML::Key << "enable_backend_sensing_processing" << YAML::Value
        << cfg.enable_backend_sensing_processing;
    out << YAML::Key << "pilot_positions" << YAML::Value << YAML::Flow << cfg.pilot_positions;
    out << YAML::Key << "midframe_pilot_symbols" << YAML::Value << YAML::Flow
        << cfg.midframe_pilot_symbols;
    out << YAML::Key << "midframe_pilot_seed" << YAML::Value << cfg.midframe_pilot_seed;
    out << YAML::Key << "equalizer_mode" << YAML::Value << cfg.equalizer_mode;
    out << YAML::Key << "channel_tracking_mode" << YAML::Value << cfg.channel_tracking_mode;
    out << YAML::Key << "equalizer_mag_floor" << YAML::Value << cfg.equalizer_mag_floor;
    out << YAML::Key << "channel_tracking_min_pilot_snr" << YAML::Value
        << cfg.channel_tracking_min_pilot_snr;
    config_detail::emit_data_resource_blocks_yaml(out, cfg);
    config_detail::emit_sensing_mask_blocks_yaml(out, cfg);
    out << YAML::EndMap;

    std::ofstream fout(filepath);
    if (!fout) {
        LOG_G_ERROR() << "Error: Cannot write to config file: " << filepath;
        return false;
    }
    fout << out.c_str();
    fout.close();
    return true;
}

inline bool load_ue_config_from_yaml(Config& cfg, const std::string& filepath) {
    if (!path_exists(filepath)) {
        return false;
    }
    try {
        YAML::Node config = YAML::LoadFile(filepath);
        expose_editor_sectioned_config(config);
        if (!config_detail::reject_legacy_key(config, "sensing_ip", "bi_sensing_ip")) return false;
        if (!config_detail::reject_legacy_key(config, "sensing_port", "bi_sensing_port")) return false;

        if (config["fft_size"]) cfg.fft_size = config["fft_size"].as<size_t>();
        if (config["cp_length"]) cfg.cp_length = config["cp_length"].as<size_t>();
        if (config["sync_pos"]) cfg.sync_pos = config["sync_pos"].as<size_t>();
        if (config["enable_sec_sync_symbol"]) {
            cfg.enable_sec_sync_symbol = config["enable_sec_sync_symbol"].as<bool>();
        }
        if (config["enable_cfo_training_sequence"]) {
            cfg.enable_cfo_training_sequence =
                config["enable_cfo_training_sequence"].as<bool>();
        }
        if (config["cfo_training_period_samples"]) {
            cfg.cfo_training_period_samples =
                config["cfo_training_period_samples"].as<size_t>();
        }
        if (config["sample_rate"]) cfg.sample_rate = config["sample_rate"].as<double>();
        if (config["bandwidth"]) cfg.bandwidth = config["bandwidth"].as<double>();
        if (config["center_freq"]) cfg.center_freq = config["center_freq"].as<double>();
        if (config["tx_gain"]) cfg.tx_gain = config["tx_gain"].as<double>();
        if (config["tx_channel"]) cfg.tx_channel = config["tx_channel"].as<uint32_t>();
        if (config["rx_gain"]) cfg.rx_gain = config["rx_gain"].as<double>();
        if (config["rx_agc_enable"]) cfg.rx_agc_enable = config["rx_agc_enable"].as<bool>();
        if (config["rx_agc_low_threshold_db"]) {
            cfg.rx_agc_low_threshold_db = config["rx_agc_low_threshold_db"].as<double>();
        }
        if (config["rx_agc_high_threshold_db"]) {
            cfg.rx_agc_high_threshold_db = config["rx_agc_high_threshold_db"].as<double>();
        }
        if (config["rx_agc_max_step_db"]) {
            cfg.rx_agc_max_step_db = config["rx_agc_max_step_db"].as<double>();
        }
        if (config["rx_agc_update_frames"]) {
            cfg.rx_agc_update_frames = config["rx_agc_update_frames"].as<size_t>();
        }
        if (config["rx_channel"]) cfg.rx_channel = config["rx_channel"].as<size_t>();
        if (config["zc_root"]) cfg.zc_root = config["zc_root"].as<int>();
        if (config["num_symbols"]) cfg.num_symbols = config["num_symbols"].as<size_t>();
        if (config["sensing_symbol_num"]) cfg.sensing_symbol_num = config["sensing_symbol_num"].as<size_t>();
        if (config["sensing_output_mode"]) {
            cfg.sensing_output_mode = config["sensing_output_mode"].as<std::string>();
        }
        if (config["sensing_on_wire_format"]) {
            cfg.sensing_on_wire_format = parse_sensing_on_wire_format_string(
                config["sensing_on_wire_format"].as<std::string>());
        }
        if (config["enable_backend_sensing_processing"]) {
            cfg.enable_backend_sensing_processing =
                config["enable_backend_sensing_processing"].as<bool>();
        }
        if (config["cuda_demod_pipeline_slots"]) {
            cfg.cuda_demod_pipeline_slots = config["cuda_demod_pipeline_slots"].as<size_t>();
        }
        if (config["frame_queue_size"]) cfg.frame_queue_size = config["frame_queue_size"].as<size_t>();
        if (config["sync_queue_size"]) cfg.sync_queue_size = config["sync_queue_size"].as<size_t>();
        if (config["sync_cfo_alias_search_range_hz"]) {
            cfg.sync_cfo_alias_search_range_hz =
                config["sync_cfo_alias_search_range_hz"].as<double>();
        }
        if (config["reset_hold_s"]) cfg.reset_hold_s = config["reset_hold_s"].as<double>();
        if (config["range_fft_size"]) cfg.range_fft_size = config["range_fft_size"].as<size_t>();
        if (config["doppler_fft_size"]) cfg.doppler_fft_size = config["doppler_fft_size"].as<size_t>();
        if (config["sensing_view_range_bins"]) {
            cfg.sensing_view_range_bins = config["sensing_view_range_bins"].as<size_t>();
        }
        if (config["sensing_view_doppler_bins"]) {
            cfg.sensing_view_doppler_bins = config["sensing_view_doppler_bins"].as<size_t>();
        }
        if (config["device_args"]) cfg.device_args = config["device_args"].as<std::string>();
        if (config["clock_source"]) cfg.clocksource = config["clock_source"].as<std::string>();
        if (config["wire_format_tx"]) cfg.wire_format_tx = config["wire_format_tx"].as<std::string>();
        if (config["downlink_rx_wire_format"]) {
            cfg.downlink_rx_wire_format = config["downlink_rx_wire_format"].as<std::string>();
        }
        if (config["enable_bi_sensing"]) cfg.enable_bi_sensing = config["enable_bi_sensing"].as<bool>();
        if (config["bi_sensing_output_enabled"]) {
            cfg.bi_sensing_output_enabled = config["bi_sensing_output_enabled"].as<bool>();
        }
        if (config["bi_sensing_ip"]) cfg.bi_sensing_ip = config["bi_sensing_ip"].as<std::string>();
        if (config["bi_sensing_port"]) cfg.bi_sensing_port = config["bi_sensing_port"].as<int>();
        if (config["control_port"]) cfg.control_port = config["control_port"].as<int>();
        if (config["channel_ip"]) cfg.channel_ip = config["channel_ip"].as<std::string>();
        if (config["channel_port"]) cfg.channel_port = config["channel_port"].as<int>();
        if (config["pdf_ip"]) cfg.pdf_ip = config["pdf_ip"].as<std::string>();
        if (config["pdf_port"]) cfg.pdf_port = config["pdf_port"].as<int>();
        if (config["constellation_ip"]) cfg.constellation_ip = config["constellation_ip"].as<std::string>();
        if (config["constellation_port"]) cfg.constellation_port = config["constellation_port"].as<int>();
        if (config["vofa_debug_ip"]) cfg.vofa_debug_ip = config["vofa_debug_ip"].as<std::string>();
        if (config["vofa_debug_port"]) cfg.vofa_debug_port = config["vofa_debug_port"].as<int>();
        if (config["udp_output_ip"]) cfg.udp_output_ip = config["udp_output_ip"].as<std::string>();
        if (config["udp_output_port"]) cfg.udp_output_port = config["udp_output_port"].as<int>();
        if (config["measurement_enable"]) cfg.measurement_enable = config["measurement_enable"].as<bool>();
        if (config["measurement_mode"]) cfg.measurement_mode = config["measurement_mode"].as<std::string>();
        if (config["measurement_run_id"]) cfg.measurement_run_id = config["measurement_run_id"].as<std::string>();
        if (config["measurement_output_dir"]) cfg.measurement_output_dir = config["measurement_output_dir"].as<std::string>();
        if (config["measurement_payload_bytes"]) cfg.measurement_payload_bytes = config["measurement_payload_bytes"].as<size_t>();
        if (config["measurement_prbs_seed"]) cfg.measurement_prbs_seed = config["measurement_prbs_seed"].as<uint32_t>();
        if (config["measurement_packets_per_point"]) cfg.measurement_packets_per_point = config["measurement_packets_per_point"].as<uint32_t>();
        if (config["measurement_max_packets_per_frame"]) {
            cfg.measurement_max_packets_per_frame = config["measurement_max_packets_per_frame"].as<size_t>();
        }
        if (config["software_sync"]) cfg.software_sync = config["software_sync"].as<bool>();
        if (config["predictive_delay"]) cfg.predictive_delay = config["predictive_delay"].as<bool>();
        if (config["hardware_sync"]) cfg.hardware_sync = config["hardware_sync"].as<bool>();
        if (config["hardware_sync_tty"]) cfg.hardware_sync_tty = config["hardware_sync_tty"].as<std::string>();
        if (config["profiling_modules"]) cfg.profiling_modules = config["profiling_modules"].as<std::string>();
        if (config["default_out_ip"]) {
            cfg.default_out_ip = config["default_out_ip"].as<std::string>();
        }
        load_simulation_config(config, cfg);
        load_duplex_config(config, cfg);
        if (config["ocxo_pi_kp_fast"]) cfg.ocxo_pi_kp_fast = config["ocxo_pi_kp_fast"].as<double>();
        if (config["ocxo_pi_ki_fast"]) cfg.ocxo_pi_ki_fast = config["ocxo_pi_ki_fast"].as<double>();
        if (config["ocxo_pi_kp_slow"]) cfg.ocxo_pi_kp_slow = config["ocxo_pi_kp_slow"].as<double>();
        if (config["ocxo_pi_ki_slow"]) cfg.ocxo_pi_ki_slow = config["ocxo_pi_ki_slow"].as<double>();
        if (config["ocxo_pi_switch_abs_error_ppm"]) cfg.ocxo_pi_switch_abs_error_ppm = config["ocxo_pi_switch_abs_error_ppm"].as<double>();
        if (config["ocxo_pi_switch_hold_s"]) cfg.ocxo_pi_switch_hold_s = config["ocxo_pi_switch_hold_s"].as<double>();
        if (config["ocxo_pi_max_step_fast_ppm"]) cfg.ocxo_pi_max_step_fast_ppm = config["ocxo_pi_max_step_fast_ppm"].as<double>();
        if (config["ocxo_pi_max_step_slow_ppm"]) cfg.ocxo_pi_max_step_slow_ppm = config["ocxo_pi_max_step_slow_ppm"].as<double>();
        if (config["ocxo_pi_max_step_ppm"]) {
            const auto max_step = config["ocxo_pi_max_step_ppm"].as<double>();
            cfg.ocxo_pi_max_step_fast_ppm = max_step;
            cfg.ocxo_pi_max_step_slow_ppm = max_step;
        }
        if (config["akf_enable"]) cfg.akf_enable = config["akf_enable"].as<bool>();
        if (config["akf_bootstrap_frames"]) cfg.akf_bootstrap_frames = config["akf_bootstrap_frames"].as<size_t>();
        if (config["akf_innovation_window"]) cfg.akf_innovation_window = config["akf_innovation_window"].as<size_t>();
        if (config["akf_max_lag"]) cfg.akf_max_lag = config["akf_max_lag"].as<size_t>();
        if (config["akf_adapt_interval"]) cfg.akf_adapt_interval = config["akf_adapt_interval"].as<size_t>();
        if (config["akf_gate_sigma"]) cfg.akf_gate_sigma = config["akf_gate_sigma"].as<double>();
        if (config["akf_tikhonov_lambda"]) cfg.akf_tikhonov_lambda = config["akf_tikhonov_lambda"].as<double>();
        if (config["akf_update_smooth"]) cfg.akf_update_smooth = config["akf_update_smooth"].as<double>();
        if (config["akf_q_wf_min"]) cfg.akf_q_wf_min = config["akf_q_wf_min"].as<double>();
        if (config["akf_q_wf_max"]) cfg.akf_q_wf_max = config["akf_q_wf_max"].as<double>();
        if (config["akf_q_rw_min"]) cfg.akf_q_rw_min = config["akf_q_rw_min"].as<double>();
        if (config["akf_q_rw_max"]) cfg.akf_q_rw_max = config["akf_q_rw_max"].as<double>();
        if (config["akf_r_min"]) cfg.akf_r_min = config["akf_r_min"].as<double>();
        if (config["akf_r_max"]) cfg.akf_r_max = config["akf_r_max"].as<double>();
        if (config["desired_peak_pos"]) cfg.desired_peak_pos = config["desired_peak_pos"].as<int>();
        if (config["sensing_symbol_stride"]) {
            cfg.sensing_symbol_stride = config["sensing_symbol_stride"].as<size_t>();
        }
        if (config["pilot_positions"]) cfg.pilot_positions = config["pilot_positions"].as<std::vector<size_t>>();
        if (config["midframe_pilot_symbols"]) {
            cfg.midframe_pilot_symbols = config["midframe_pilot_symbols"].as<std::vector<size_t>>();
        }
        if (config["midframe_pilot_seed"]) {
            cfg.midframe_pilot_seed = config["midframe_pilot_seed"].as<uint32_t>();
        }
        if (config["equalizer_mode"]) cfg.equalizer_mode = config["equalizer_mode"].as<std::string>();
        if (config["channel_tracking_mode"]) {
            cfg.channel_tracking_mode = normalize_channel_tracking_mode_string(
                config["channel_tracking_mode"].as<std::string>());
        }
        if (config["equalizer_mag_floor"]) {
            cfg.equalizer_mag_floor = config["equalizer_mag_floor"].as<double>();
        }
        if (config["channel_tracking_min_pilot_snr"]) {
            cfg.channel_tracking_min_pilot_snr =
                config["channel_tracking_min_pilot_snr"].as<double>();
        }
        if (!config_detail::load_data_resource_blocks_from_yaml(cfg, config, "UE config")) {
            return false;
        }
        if (!config_detail::load_sensing_mask_blocks_from_yaml(cfg, config, "UE config")) {
            return false;
        }
        if (config["downlink_cpu_cores"]) {
            cfg.downlink_cpu_cores = config["downlink_cpu_cores"].as<std::vector<int>>();
        }
        if (config["uplink_cpu_cores"]) {
            cfg.uplink_cpu_cores = config["uplink_cpu_cores"].as<std::vector<int>>();
        }
        if (config["main_cpu_core"]) cfg.main_cpu_core = config["main_cpu_core"].as<int>();
        return true;
    } catch (const YAML::Exception& e) {
        LOG_G_ERROR() << "Error parsing YAML config: " << e.what();
        return false;
    }
}

inline void finalize_ue_network_defaults(Config& cfg) {
    normalize_sensing_fft_sizes(cfg, "deBS sensing");
    normalize_sensing_view_bins(cfg, "deBS sensing");
    if (cfg.cuda_demod_pipeline_slots == 0) {
        LOG_G_WARN() << "cuda_demod_pipeline_slots=0 is invalid. Clamping to 1.";
        cfg.cuda_demod_pipeline_slots = 1;
    }
    if (cfg.frame_queue_size == 0) {
        LOG_G_WARN() << "frame_queue_size=0 is invalid. Clamping to 1.";
        cfg.frame_queue_size = 1;
    }
    if (cfg.sync_queue_size == 0) {
        LOG_G_WARN() << "sync_queue_size=0 is invalid. Clamping to 1.";
        cfg.sync_queue_size = 1;
    }
    if (cfg.sync_cfo_alias_search_range_hz < 0.0 ||
        !std::isfinite(cfg.sync_cfo_alias_search_range_hz)) {
        LOG_G_WARN() << "sync_cfo_alias_search_range_hz is invalid. Clamping to 0 Hz.";
        cfg.sync_cfo_alias_search_range_hz = 0.0;
    }
    if (cfg.reset_hold_s <= 0.0) {
        LOG_G_WARN() << "reset_hold_s<=0 is invalid. Clamping to 0.5 s.";
        cfg.reset_hold_s = 0.5;
    }
    if (cfg.sensing_symbol_stride == 0) {
        LOG_G_WARN() << "sensing_symbol_stride=0 is invalid. Clamping to 1.";
        cfg.sensing_symbol_stride = 1;
    }
    if (cfg.rx_agc_update_frames == 0) {
        LOG_G_WARN() << "rx_agc_update_frames=0 is invalid. Clamping to 1.";
        cfg.rx_agc_update_frames = 1;
    }
    if (cfg.rx_agc_max_step_db <= 0.0) {
        LOG_G_WARN() << "rx_agc_max_step_db<=0 is invalid. Clamping to 1 dB.";
        cfg.rx_agc_max_step_db = 1.0;
    }
    if (cfg.rx_agc_low_threshold_db >= cfg.rx_agc_high_threshold_db) {
        LOG_G_WARN() << "rx_agc_low_threshold_db>=rx_agc_high_threshold_db is invalid. Resetting to 11/13 dB.";
        cfg.rx_agc_low_threshold_db = 11.0;
        cfg.rx_agc_high_threshold_db = 13.0;
    }
    if (cfg.equalizer_mode != kEqualizerModeZf &&
        cfg.equalizer_mode != kEqualizerModeMmse) {
        LOG_G_WARN() << "Unsupported equalizer_mode='" << cfg.equalizer_mode
                     << "'. Falling back to '" << kEqualizerModeMmse << "'.";
        cfg.equalizer_mode = kEqualizerModeMmse;
    }
    cfg.channel_tracking_mode = normalize_channel_tracking_mode_string(cfg.channel_tracking_mode);
    if (cfg.equalizer_mag_floor <= 0.0 || !std::isfinite(cfg.equalizer_mag_floor)) {
        LOG_G_WARN() << "equalizer_mag_floor is invalid. Falling back to 1e-6.";
        cfg.equalizer_mag_floor = 1e-6;
    }
    if (cfg.channel_tracking_min_pilot_snr <= 0.0 ||
        !std::isfinite(cfg.channel_tracking_min_pilot_snr)) {
        LOG_G_WARN() << "channel_tracking_min_pilot_snr is invalid. Falling back to 1e-4.";
        cfg.channel_tracking_min_pilot_snr = 1e-4;
    }
    {
        std::sort(cfg.midframe_pilot_symbols.begin(), cfg.midframe_pilot_symbols.end());
        cfg.midframe_pilot_symbols.erase(
            std::unique(cfg.midframe_pilot_symbols.begin(), cfg.midframe_pilot_symbols.end()),
            cfg.midframe_pilot_symbols.end());
    }
    if (cfg.measurement_enable) {
        if (cfg.measurement_mode.empty()) {
            cfg.measurement_mode = "internal_prbs";
        }
        if (!measurement_mode_enabled(cfg)) {
            LOG_G_WARN() << "Unsupported UE measurement_mode='"
                         << cfg.measurement_mode << "'. Disabling measurement mode.";
            cfg.measurement_enable = false;
        }
        if (cfg.measurement_payload_bytes < measurement_payload_header_size()) {
            LOG_G_WARN() << "measurement_payload_bytes=" << cfg.measurement_payload_bytes
                         << " is smaller than the measurement header. Clamping to "
                         << measurement_payload_header_size() << '.';
            cfg.measurement_payload_bytes = measurement_payload_header_size();
        }
        if (cfg.measurement_packets_per_point == 0) {
            LOG_G_WARN() << "measurement_packets_per_point=0 is invalid. Clamping to 1.";
            cfg.measurement_packets_per_point = 1;
        }
    }
    finalize_data_resource_grid_config(cfg, "UE");
    finalize_sensing_mask_config(cfg, "UE");
    if (cfg.bi_sensing_ip.empty()) cfg.bi_sensing_ip = "0.0.0.0";
    if (cfg.channel_ip.empty()) cfg.channel_ip = "0.0.0.0";
    if (cfg.pdf_ip.empty()) cfg.pdf_ip = "0.0.0.0";
    if (cfg.constellation_ip.empty()) cfg.constellation_ip = "0.0.0.0";
    if (cfg.vofa_debug_ip.empty()) cfg.vofa_debug_ip = cfg.default_out_ip;
    if (cfg.udp_output_ip.empty()) cfg.udp_output_ip = cfg.default_out_ip;
}

inline void log_ue_sync_mode(const Config& cfg) {
    if (cfg.hardware_sync && cfg.software_sync) {
        LOG_G_INFO() << "Both software_sync and hardware_sync are enabled.";
    } else if (cfg.hardware_sync) {
        LOG_G_INFO() << "Hardware sync enabled.";
    } else if (cfg.software_sync) {
        LOG_G_INFO() << "Software sync enabled.";
    } else {
        LOG_G_WARN() << "Both software_sync and hardware_sync are disabled.";
    }
    LOG_G_INFO() << "Predictive delay compensation "
                 << (cfg.predictive_delay ? "enabled." : "disabled.");
}

inline void log_ue_agc_mode(const Config& cfg) {
    if (!cfg.rx_agc_enable) {
        LOG_G_INFO() << "RX AGC disabled. Using fixed RX gain: " << cfg.rx_gain << " dB";
        return;
    }
    LOG_G_INFO() << "RX AGC enabled. low_threshold_db=" << cfg.rx_agc_low_threshold_db
                 << ", high_threshold_db=" << cfg.rx_agc_high_threshold_db
                 << ", max_step_db=" << cfg.rx_agc_max_step_db
                 << ", update_frames=" << cfg.rx_agc_update_frames;
}

/**
 * @brief Synchronization search batch with source USRP timestamp.
 */
struct SyncBatch {
    AlignedVector data;
    int64_t usrp_time_ns = -1;
};

inline int64_t time_spec_to_ns(const uhd::time_spec_t& time_spec) {
    return static_cast<int64_t>(std::llround(time_spec.get_real_secs() * 1e9));
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

    void mark_reset_now(const uhd::time_spec_t& now) {
        mark_reset(time_spec_to_ns(now));
    }

    void mark_alignment_now(const uhd::time_spec_t& now) {
        mark_alignment(time_spec_to_ns(now));
    }

    void mark_freq_adjust_now(const uhd::time_spec_t& now) {
        mark_freq_adjust(time_spec_to_ns(now));
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
        : _enabled(cfg.rx_agc_enable),
          _low_threshold_db(cfg.rx_agc_low_threshold_db),
          _high_threshold_db(cfg.rx_agc_high_threshold_db),
          _max_step_db(cfg.rx_agc_max_step_db),
          _update_frames(std::max<size_t>(1, cfg.rx_agc_update_frames)) {}

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
        : _enabled(cfg.rx_agc_enable),
          _default_gain_db(cfg.rx_gain) {}

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
    float CFO;
    float SFO;
    float delay_offset;
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
        return cfg.doppler_fft_size;
    case SensingViewerFrameFormat::CompactRaw:
        return cfg.sensing_symbol_num;
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
        return cfg.range_fft_size;
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
        return std::min(cfg.sensing_symbol_num, cfg.doppler_fft_size);
    case SensingViewerFrameFormat::CompactRaw:
        return cfg.sensing_symbol_num;
    case SensingViewerFrameFormat::DenseRangeDoppler:
        return cfg.doppler_fft_size;
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
        return std::min(cfg.fft_size, cfg.range_fft_size);
    case SensingViewerFrameFormat::CompactRaw:
        return analysis.common_subcarrier_count;
    case SensingViewerFrameFormat::DenseRangeDoppler:
        return cfg.range_fft_size;
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
    switch (cfg.sensing_on_wire_format) {
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
    packet.frame_symbol_period = htonl(static_cast<uint32_t>(cfg.num_symbols));
    packet.range_fft_size = htonl(static_cast<uint32_t>(cfg.range_fft_size));
    packet.doppler_fft_size = htonl(static_cast<uint32_t>(cfg.doppler_fft_size));
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
    if (cfg.center_freq > 0.0) {
        packet.center_freq_hz_div100 = htonl(static_cast<uint32_t>(
            std::llround(std::clamp(cfg.center_freq / 100.0, 0.0, 4294967295.0))));
    }
    if (cfg.sample_rate > 0.0) {
        packet.sample_rate_hz_div100 = htonl(static_cast<uint32_t>(
            std::llround(std::clamp(cfg.sample_rate / 100.0, 0.0, 4294967295.0))));
    }
    if (antenna_spacing_m > 0.0) {
        packet.antenna_spacing_um = htonl(static_cast<uint32_t>(
            std::llround(std::clamp(antenna_spacing_m * 1e6, 0.0, 4294967295.0))));
    }
    return packet;
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
            LOG_G_WARN() << "Warning: Failed to set send buffer size";
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
 * @brief ZeroMQ PUB sender for Backend->Frontend visualization streams.
 *
 * Backend binds a TCP PUB endpoint and frontend scripts connect with SUB.
 * This is intentionally separate from UdpSender because it does not use UDP.
 */
class ZmqPubSender {
protected:
    std::shared_ptr<zmq_transport::SharedPubSocket> pub_;

public:
    ZmqPubSender(const std::string& ip, uint16_t port) {
        pub_ = zmq_transport::bind_pub(zmq_transport::make_tcp_endpoint(ip, port));
    }

    virtual ~ZmqPubSender() = default;

    template <typename T>
    void send_raw(const T* data, size_t size_bytes) {
        pub_->send_single(static_cast<const void*>(data), size_bytes);
    }

    void send_frame(const std::vector<zmq_transport::MsgPart>& parts) {
        pub_->send_multipart(parts);
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
        if (channels == nullptr || channel_count == 0 || channel_count > kMaxChannels) {
            return;
        }
        if (++frame_counter_ < interval_frames_) {
            return;
        }
        frame_counter_ = 0;

        const size_t payload_bytes = channel_count * sizeof(float);
        std::memcpy(packet_buffer_.data(), channels, payload_bytes);
        std::memcpy(packet_buffer_.data() + payload_bytes, kFireWaterTail.data(), kFireWaterTail.size());
        udp_sender_.send(packet_buffer_.data(), payload_bytes + kFireWaterTail.size());
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
        if (!_enabled) return;
        if (!_running.load()) return;

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
        if (!_enabled) return;
        if (!_running.load()) return;

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
        if (!_enabled) return;
        if (!_running.load()) return;

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
        if (!_enabled) return;
        if (!_running.load()) return;

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
        if (!_enabled) return;
        if (!_running.load()) return;

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
        if (!_enabled) return;
        if (!_running.load()) return;

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

    void _queue_frame(FrameData&& frame_data) {
        spsc_wait_push(_data_queue, std::move(frame_data), [this]() {
            return !_running.load(std::memory_order_acquire);
        });
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
        if (!_enabled || !_running.load(std::memory_order_acquire)) return;
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
        if (!_enabled || !_running.load(std::memory_order_acquire)) return;
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
        if (!_enabled || !_running.load(std::memory_order_acquire)) return;
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
        if (!_enabled || !_running.load(std::memory_order_acquire)) return;
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
        if (!_enabled || !_running.load(std::memory_order_acquire)) return;
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
        if (!_enabled || !_running.load(std::memory_order_acquire)) return;
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

    int32_t _channel_index(uint32_t channel_id) const {
        if (channel_id >= _channel_index_by_id.size()) {
            return -1;
        }
        return _channel_index_by_id[channel_id];
    }

    void _queue_frame(uint32_t channel_id, FrameData&& frame_data) {
        const int32_t channel_index = _channel_index(channel_id);
        if (channel_index < 0) {
            LOG_G_WARN() << "Ignoring aggregated sensing frame for unknown channel id=" << channel_id;
            return;
        }
        auto& queue = _channel_queues[static_cast<size_t>(channel_index)];
        spsc_wait_push(queue, std::move(frame_data), [this]() {
            return !_running.load(std::memory_order_acquire);
        });
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
        LOG_RT_WARN_HZ(5) << "[Sensing Aggregate] dropping incomplete frame start_symbol="
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
                LOG_G_WARN() << "[Sensing Aggregate] channel payload size mismatch for frame "
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
            LOG_G_WARN() << "[Sensing Aggregate] dropping metadata sidecar for frame "
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
        DeliveryMode mode = DeliveryMode::QueuedBlocking
    )
        : queue_(queue_size),
          send_func_(std::move(send_func)),
          send_interval_(interval),
          delivery_mode_(mode),
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
            queue_.try_push(std::move(data));
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
                while (queue_.try_pop(newer)) {
                    latest = std::move(newer);
                }

                try {
                    send_func_(latest);
                } catch (const std::exception& e) {
                    LOG_G_WARN() << "DataSender error: " << e.what();
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
                LOG_G_WARN() << "DataSender error: " << e.what();
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
    std::thread thread_;
    std::atomic<bool> running_;
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
    // connected viewers the backend is up. The ip/port arguments are retained
    // for source compatibility but ignored: the message is broadcast to recent
    // control peers (ROUTER can only address known DEALER ids).
    void send_heartbeat(const std::string& /*dest_ip*/, int /*dest_port*/) {
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
            LOG_G_WARN() << "Control status command id must be exactly 4 bytes";
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
            _router->post_send(peer, data, size);
        }
    }

    void _send_to(const ControlPeer& peer, const void* data, size_t size) {
        _router->post_send(peer, data, size);
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
            _router->flush();
        }
    }

    void _dispatch(const ControlPeer& identity, const std::vector<uint8_t>& payload) {
        if (payload.size() != sizeof(ControlCommand)) {
            LOG_G_WARN() << "Received malformed control command (" << payload.size() << " bytes)";
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
                LOG_G_WARN() << "Unknown command: " << command_str;
                return;
            }
            try {
                callback(value);
            } catch (const std::exception& e) {
                LOG_G_WARN() << "Error processing command '" << command_str
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
                LOG_G_WARN() << "Unknown command: " << command_str;
                return;
            }
            try {
                callback(value, identity);
            } catch (const std::exception& e) {
                LOG_G_WARN() << "Error processing command '" << command_str
                             << "': " << e.what();
            }
        } else {
            LOG_G_WARN() << "Invalid command header received";
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
            LOG_G_ERROR() << "ERROR: Failed to open serial device: " << strerror(errno);
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
            LOG_G_ERROR() << "ERROR: Failed to set serial attributes: " << strerror(errno);
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
        LOG_G_INFO() << "Configured OCXO PI: Kp_fast=" << ocxo_pi_kp_fast_
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
            LOG_RT_INFO() << "OCXO PI initialized (stage="
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
            LOG_RT_INFO() << "OCXO PI switched to slow stage (Kp="
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
            LOG_RT_INFO() << "OCXO PI adjust: error_ppm=" << error_ppm
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
                    LOG_G_ERROR() << "ERROR: Serial write error: " << strerror(errno) 
                                  << " [command: " << command << "]";
                    return;
                }
                // Wait for buffer availability
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        
        if (remaining > 0) {
            LOG_G_ERROR() << "ERROR: Failed to send full command after 3 attempts: " 
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
                LOG_G_ERROR() << "ERROR: Response timeout for command: " << command;
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
                    LOG_G_ERROR() << "ERROR: Serial read error: " << strerror(errno);
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
            LOG_G_ERROR() << "ERROR: Device rejected command - BAD DATA: " << command;
        }
        else if (strncmp(response, "BAD CMD", 7) == 0) {
            LOG_G_ERROR() << "ERROR: Device rejected command - BAD CMD: " << command;
        }
        else if (strncmp(response, "TIMEOUT", 7) == 0) {
            LOG_G_ERROR() << "ERROR: Device operation timed out for command: " << command;
        }
        else {
            LOG_G_ERROR() << "ERROR: Unexpected device response: " << response 
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
