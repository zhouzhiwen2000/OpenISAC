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
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <cctype>
#include <cstring>
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
 * Used by OFDMModulator for multi-channel monostatic sensing reception.
 */
struct SensingRxChannelConfig {
    uint32_t usrp_channel = 0;          // USRP RX channel index
    std::string device_args = "";       // Optional per-channel USRP args
    std::string clock_source = "";      // Optional per-channel USRP clock/REF source override
    std::string time_source = "";       // Optional per-channel USRP time/PPS source override
    std::string wire_format_rx = "";    // Optional per-channel RX wire format override
    double rx_gain = 0.0;               // RX gain for this channel
    int32_t alignment = 63;             // Per-channel alignment offset (samples)
    std::string rx_antenna = "";        // RX antenna for this channel (e.g., TX/RX, RX1)
    bool enable_system_delay_estimation = false; // Enable per-channel system delay estimation mode
    bool enable_sensing_output = true;  // Enable UDP output for this sensing channel
    std::string sensing_ip = "127.0.0.1";
    int sensing_port = 8888;
};

/**
 * @brief Rectangular payload resource block in the OFDM time-frequency grid.
 *
 * Coordinates use absolute 0-based frame indices:
 * - symbol_start / symbol_count refer to OFDM symbols within one frame
 * - subcarrier_start / subcarrier_count refer to FFT-bin indices
 */
struct DataResourceBlock {
    size_t symbol_start = 0;
    size_t symbol_count = 0;
    size_t subcarrier_start = 0;
    size_t subcarrier_count = 0;
};

/**
 * @brief Shared payload-resource layout derived from Config.
 *
 * The flattened non-pilot order is:
 *   1. OFDM symbols in ascending absolute frame order, skipping sync_pos
 *   2. Within each data symbol, non-pilot subcarriers in ascending order
 *
 * payload_mask / payload_rank are indexed in that flattened order.
 */
struct DataResourceGridLayout {
    size_t num_symbols = 0;
    size_t fft_size = 0;
    size_t sync_pos = 0;
    size_t data_symbol_count = 0;
    size_t num_non_pilot_subcarriers = 0;
    size_t non_pilot_re_count = 0;
    size_t payload_re_count = 0;

    std::vector<int> data_symbol_to_actual_symbol;
    std::vector<int> actual_symbol_to_data_symbol;
    std::vector<int> non_pilot_subcarrier_indices;
    std::vector<int> subcarrier_to_non_pilot_index;
    std::vector<uint8_t> pilot_mask;
    std::vector<uint8_t> payload_mask;
    std::vector<int> payload_rank;
    std::vector<size_t> payload_offsets;
    std::vector<size_t> non_pilot_offsets;

    size_t flat_non_pilot_index(size_t data_symbol_idx, size_t non_pilot_idx) const {
        return non_pilot_offsets[data_symbol_idx] + non_pilot_idx;
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
    size_t cp_length = 128;            // Cyclic prefix length
    size_t num_symbols = 100;          // Number of symbols per frame
    size_t sensing_symbol_num = 100;   // Number of sensing symbols
    size_t cuda_mod_pipeline_slots = 2;   // Number of CUDA mod pipeline slots
    size_t cuda_demod_pipeline_slots = 3; // Number of CUDA demod pipeline slots
    size_t frame_queue_size = 8;       // Capacity of demod RX frame queue
    size_t sync_queue_size = 8;        // Capacity of demod sync-search queue
    size_t sync_pos = 1;               // Synchronization symbol position
    int delay_adjust_step = 2;         // Delay adjustment step
    double reset_hold_s = 0.5;         // Time window of persistent invalid delay before forcing a hard reset
    int desired_peak_pos = 20;         // Desired delay peak position to include non-causal components
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
    bool data_resource_blocks_configured = false;
    std::vector<DataResourceBlock> data_resource_blocks;
    size_t payload_re_count = 0;
    size_t non_pilot_re_count = 0;
    std::string device_args = "";
    std::string tx_device_args = "";
    std::string rx_device_args = "";
    std::string default_ip = "127.0.0.1";
    bool mono_sensing_output_enabled = true;
    std::string mono_sensing_ip = "127.0.0.1";
    int mono_sensing_port = 8888;
    bool enable_bi_sensing = true;
    std::string bi_sensing_ip = "127.0.0.1";
    int bi_sensing_port = 8889;
    int control_port = 9999;
    std::string channel_ip = "127.0.0.1";
    int channel_port = 12348;
    std::string pdf_ip = "127.0.0.1";
    int pdf_port = 12349;
    std::string constellation_ip = "127.0.0.1";
    int constellation_port = 12346;
    std::string vofa_debug_ip = "127.0.0.1";
    int vofa_debug_port = 12347;
    // UDP output for decoded payloads
    std::string udp_output_ip = "127.0.0.1";
    int udp_output_port = 50001;
    // Modulator UDP input (for incoming payloads to modulate)
    std::string udp_input_ip = "0.0.0.0"; // bind address
    int udp_input_port = 50000;
    std::string clocksource = "internal"; // Clock source
    std::string timesource = "";          // Time source; empty means follow clocksource
    std::string tx_clock_source = "";     // TX clock source override
    std::string tx_time_source = "";      // TX time source override
    std::string rx_clock_source = "";     // Default RX clock source override
    std::string rx_time_source = "";      // Default RX time source override
    std::string wire_format_tx = "sc16";
    std::string wire_format_rx = "sc16";
    uint32_t sensing_rx_channel_count = 1; // Number of sensing RX channels
    std::vector<SensingRxChannelConfig> sensing_rx_channels; // Per-channel sensing RX config
    size_t sensing_symbol_stride = 20;     // Default sensing STRD applied at startup
    size_t tx_circular_buffer_size = 8;    // Capacity of modulator frame circular buffer
    size_t data_packet_buffer_size = 32;   // Capacity of modulator encoded packet buffer
    size_t paired_frame_queue_size = 8;    // Capacity of per-channel RX/TX pairing queues
    std::vector<size_t> available_cores = {0, 1, 2, 3, 4, 5};
    std::string profiling_modules = "";  // Comma-separated list of modules to profile: modulation, latency, sensing, sensing_proc, sensing_process, data_ingest, demodulation, agc, align, snr, or "all"
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

    DataResourceGridLayout layout;
    layout.num_symbols = cfg.num_symbols;
    layout.fft_size = cfg.fft_size;
    layout.sync_pos = cfg.sync_pos;
    layout.data_symbol_count = cfg.num_symbols - 1;

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
    layout.non_pilot_re_count = layout.data_symbol_count * layout.num_non_pilot_subcarriers;

    layout.data_symbol_to_actual_symbol.reserve(layout.data_symbol_count);
    layout.actual_symbol_to_data_symbol.assign(cfg.num_symbols, -1);
    for (size_t sym = 0; sym < cfg.num_symbols; ++sym) {
        if (sym == cfg.sync_pos) {
            continue;
        }
        layout.actual_symbol_to_data_symbol[sym] =
            static_cast<int>(layout.data_symbol_to_actual_symbol.size());
        layout.data_symbol_to_actual_symbol.push_back(static_cast<int>(sym));
    }
    if (layout.data_symbol_to_actual_symbol.size() != layout.data_symbol_count) {
        throw std::runtime_error("Failed to derive data-symbol index mapping from sync_pos.");
    }

    layout.payload_mask.assign(layout.non_pilot_re_count, cfg.data_resource_blocks_configured ? 0 : 1);
    layout.payload_rank.assign(layout.non_pilot_re_count, -1);
    layout.non_pilot_offsets.resize(layout.data_symbol_count + 1, 0);
    layout.payload_offsets.resize(layout.data_symbol_count + 1, 0);
    for (size_t data_sym = 0; data_sym <= layout.data_symbol_count; ++data_sym) {
        layout.non_pilot_offsets[data_sym] = data_sym * layout.num_non_pilot_subcarriers;
    }

    size_t stripped_sync_re = 0;
    size_t stripped_pilot_re = 0;
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
                if (sym == cfg.sync_pos) {
                    stripped_sync_re += block.subcarrier_count;
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
                    if (layout.pilot_mask[sc] != 0) {
                        ++stripped_pilot_re;
                        continue;
                    }
                    const int non_pilot_idx = layout.subcarrier_to_non_pilot_index[sc];
                    if (non_pilot_idx < 0) {
                        continue;
                    }
                    layout.payload_mask[base + static_cast<size_t>(non_pilot_idx)] = 1;
                }
            }
        }
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
        (stripped_sync_re > 0 || stripped_pilot_re > 0)) {
        LOG_G_WARN() << "data_resource_blocks overlap stripped " << stripped_sync_re
                     << " sync RE and " << stripped_pilot_re
                     << " pilot RE. sync_pos and pilot_positions take precedence.";
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

inline size_t core_from_hint(const Config& cfg, size_t hint) {
    if (cfg.available_cores.empty()) {
        return 0;
    }
    return cfg.available_cores[hint % cfg.available_cores.size()];
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

inline void emit_data_resource_blocks_yaml(YAML::Emitter& out, const Config& cfg) {
    if (!cfg.data_resource_blocks_configured) {
        return;
    }

    out << YAML::Key << "data_resource_blocks" << YAML::Value << YAML::BeginSeq;
    for (const auto& block : cfg.data_resource_blocks) {
        out << YAML::BeginMap;
        out << YAML::Key << "symbol_start" << YAML::Value << block.symbol_start;
        out << YAML::Key << "symbol_count" << YAML::Value << block.symbol_count;
        out << YAML::Key << "subcarrier_start" << YAML::Value << block.subcarrier_start;
        out << YAML::Key << "subcarrier_count" << YAML::Value << block.subcarrier_count;
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;
}

inline bool load_data_resource_blocks_from_yaml(Config& cfg, const YAML::Node& config, const char* context_name) {
    const YAML::Node blocks = config["data_resource_blocks"];
    if (!blocks) {
        cfg.data_resource_blocks_configured = false;
        cfg.data_resource_blocks.clear();
        return true;
    }
    if (!blocks.IsSequence()) {
        LOG_G_ERROR() << context_name << " key 'data_resource_blocks' must be a YAML sequence.";
        return false;
    }

    cfg.data_resource_blocks_configured = true;
    cfg.data_resource_blocks.clear();
    cfg.data_resource_blocks.reserve(blocks.size());
    for (size_t idx = 0; idx < blocks.size(); ++idx) {
        const YAML::Node& node = blocks[idx];
        if (!node.IsMap()) {
            LOG_G_ERROR() << context_name << " data_resource_blocks[" << idx
                          << "] must be a YAML map.";
            return false;
        }
        if (!node["symbol_start"] || !node["symbol_count"] ||
            !node["subcarrier_start"] || !node["subcarrier_count"]) {
            LOG_G_ERROR() << context_name << " data_resource_blocks[" << idx
                          << "] must define symbol_start, symbol_count, subcarrier_start, and subcarrier_count.";
            return false;
        }

        DataResourceBlock block;
        block.symbol_start = node["symbol_start"].as<size_t>();
        block.symbol_count = node["symbol_count"].as<size_t>();
        block.subcarrier_start = node["subcarrier_start"].as<size_t>();
        block.subcarrier_count = node["subcarrier_count"].as<size_t>();
        cfg.data_resource_blocks.push_back(block);
    }
    return true;
}
} // namespace config_detail

inline Config make_default_modulator_config() {
    Config cfg;
    cfg.fft_size = 1024;
    cfg.cp_length = 128;
    cfg.sync_pos = 1;
    cfg.sample_rate = 50e6;
    cfg.bandwidth = 50e6;
    cfg.center_freq = 2.4e9;
    cfg.tx_gain = 30.0;
    cfg.tx_channel = 0;
    cfg.zc_root = 29;
    cfg.pilot_positions = {571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451};
    cfg.num_symbols = 100;
    cfg.cuda_mod_pipeline_slots = 2;
    cfg.mono_sensing_output_enabled = true;
    cfg.mono_sensing_ip = "";
    cfg.mono_sensing_port = 8888;
    cfg.control_port = 9999;
    cfg.sensing_rx_channel_count = 1;
    cfg.sensing_symbol_stride = 20;
    cfg.tx_circular_buffer_size = 8;
    cfg.data_packet_buffer_size = 32;
    cfg.paired_frame_queue_size = 8;
    cfg.udp_input_ip = "0.0.0.0";
    cfg.udp_input_port = 50000;
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

inline bool save_modulator_config_to_yaml(const Config& cfg, const std::string& filepath) {
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "fft_size" << YAML::Value << cfg.fft_size;
    out << YAML::Key << "range_fft_size" << YAML::Value << cfg.range_fft_size;
    out << YAML::Key << "doppler_fft_size" << YAML::Value << cfg.doppler_fft_size;
    out << YAML::Key << "cp_length" << YAML::Value << cfg.cp_length;
    out << YAML::Key << "sync_pos" << YAML::Value << cfg.sync_pos;
    out << YAML::Key << "sample_rate" << YAML::Value << cfg.sample_rate;
    out << YAML::Key << "bandwidth" << YAML::Value << cfg.bandwidth;
    out << YAML::Key << "center_freq" << YAML::Value << cfg.center_freq;
    out << YAML::Key << "tx_gain" << YAML::Value << cfg.tx_gain;
    out << YAML::Key << "tx_channel" << YAML::Value << cfg.tx_channel;
    out << YAML::Key << "zc_root" << YAML::Value << cfg.zc_root;
    out << YAML::Key << "num_symbols" << YAML::Value << cfg.num_symbols;
    out << YAML::Key << "sensing_symbol_num" << YAML::Value << cfg.sensing_symbol_num;
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
    out << YAML::Key << "wire_format_rx" << YAML::Value << cfg.wire_format_rx;
    out << YAML::Key << "udp_input_ip" << YAML::Value << cfg.udp_input_ip;
    out << YAML::Key << "udp_input_port" << YAML::Value << cfg.udp_input_port;
    out << YAML::Key << "mono_sensing_output_enabled" << YAML::Value << cfg.mono_sensing_output_enabled;
    out << YAML::Key << "mono_sensing_ip" << YAML::Value << cfg.mono_sensing_ip;
    out << YAML::Key << "mono_sensing_port" << YAML::Value << cfg.mono_sensing_port;
    out << YAML::Key << "sensing_symbol_stride" << YAML::Value << cfg.sensing_symbol_stride;
    out << YAML::Key << "sensing_rx_channel_count" << YAML::Value << cfg.sensing_rx_channel_count;
    out << YAML::Key << "sensing_rx_channels" << YAML::Value << YAML::BeginSeq;
    for (const auto& ch : cfg.sensing_rx_channels) {
        out << YAML::BeginMap;
        out << YAML::Key << "usrp_channel" << YAML::Value << ch.usrp_channel;
        out << YAML::Key << "device_args" << YAML::Value << ch.device_args;
        out << YAML::Key << "clock_source" << YAML::Value << ch.clock_source;
        out << YAML::Key << "time_source" << YAML::Value << ch.time_source;
        out << YAML::Key << "wire_format_rx" << YAML::Value << ch.wire_format_rx;
        out << YAML::Key << "rx_gain" << YAML::Value << ch.rx_gain;
        out << YAML::Key << "alignment" << YAML::Value << ch.alignment;
        out << YAML::Key << "rx_antenna" << YAML::Value << ch.rx_antenna;
        out << YAML::Key << "enable_system_delay_estimation" << YAML::Value << ch.enable_system_delay_estimation;
        out << YAML::Key << "enable_sensing_output" << YAML::Value << ch.enable_sensing_output;
        out << YAML::Key << "sensing_ip" << YAML::Value << ch.sensing_ip;
        out << YAML::Key << "sensing_port" << YAML::Value << ch.sensing_port;
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;
    out << YAML::Key << "default_ip" << YAML::Value << cfg.default_ip;
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
    out << YAML::Key << "profiling_modules" << YAML::Value << cfg.profiling_modules;
    out << YAML::Key << "pilot_positions" << YAML::Value << YAML::Flow << cfg.pilot_positions;
    config_detail::emit_data_resource_blocks_yaml(out, cfg);
    out << YAML::Key << "cpu_cores" << YAML::Value << YAML::Flow << cfg.available_cores;
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

inline bool load_modulator_config_from_yaml(Config& cfg, const std::string& filepath) {
    if (!path_exists(filepath)) {
        return false;
    }
    try {
        YAML::Node config = YAML::LoadFile(filepath);
        const bool has_range_fft_size = static_cast<bool>(config["range_fft_size"]);
        if (!config_detail::reject_legacy_key(config, "mod_udp_ip", "udp_input_ip")) return false;
        if (!config_detail::reject_legacy_key(config, "mod_udp_port", "udp_input_port")) return false;
        if (!config_detail::reject_legacy_key(config, "sensing_ip", "mono_sensing_ip")) return false;
        if (!config_detail::reject_legacy_key(config, "sensing_port", "mono_sensing_port")) return false;

        if (config["fft_size"]) cfg.fft_size = config["fft_size"].as<size_t>();
        if (has_range_fft_size) cfg.range_fft_size = config["range_fft_size"].as<size_t>();
        if (config["doppler_fft_size"]) cfg.doppler_fft_size = config["doppler_fft_size"].as<size_t>();
        if (config["cp_length"]) cfg.cp_length = config["cp_length"].as<size_t>();
        if (config["sync_pos"]) cfg.sync_pos = config["sync_pos"].as<size_t>();
        if (config["sample_rate"]) cfg.sample_rate = config["sample_rate"].as<double>();
        if (config["bandwidth"]) cfg.bandwidth = config["bandwidth"].as<double>();
        if (config["center_freq"]) cfg.center_freq = config["center_freq"].as<double>();
        if (config["tx_gain"]) cfg.tx_gain = config["tx_gain"].as<double>();
        if (config["tx_channel"]) cfg.tx_channel = config["tx_channel"].as<uint32_t>();
        if (config["zc_root"]) cfg.zc_root = config["zc_root"].as<int>();
        if (config["num_symbols"]) cfg.num_symbols = config["num_symbols"].as<size_t>();
        if (config["sensing_symbol_num"]) cfg.sensing_symbol_num = config["sensing_symbol_num"].as<size_t>();
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
        if (config["wire_format_rx"]) cfg.wire_format_rx = config["wire_format_rx"].as<std::string>();
        if (config["udp_input_ip"]) cfg.udp_input_ip = config["udp_input_ip"].as<std::string>();
        if (config["udp_input_port"]) cfg.udp_input_port = config["udp_input_port"].as<int>();
        if (config["mono_sensing_output_enabled"]) {
            cfg.mono_sensing_output_enabled = config["mono_sensing_output_enabled"].as<bool>();
        }
        if (config["mono_sensing_ip"]) cfg.mono_sensing_ip = config["mono_sensing_ip"].as<std::string>();
        if (config["mono_sensing_port"]) cfg.mono_sensing_port = config["mono_sensing_port"].as<int>();
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
                if (node["wire_format_rx"]) ch.wire_format_rx = node["wire_format_rx"].as<std::string>();
                if (node["rx_gain"]) ch.rx_gain = node["rx_gain"].as<double>();
                if (node["alignment"]) ch.alignment = node["alignment"].as<int32_t>();
                if (node["rx_antenna"]) ch.rx_antenna = node["rx_antenna"].as<std::string>();
                if (node["enable_system_delay_estimation"]) {
                    ch.enable_system_delay_estimation = node["enable_system_delay_estimation"].as<bool>();
                }
                if (node["enable_sensing_output"]) {
                    ch.enable_sensing_output = node["enable_sensing_output"].as<bool>();
                }
                if (node["sensing_ip"]) ch.sensing_ip = node["sensing_ip"].as<std::string>();
                if (node["sensing_port"]) ch.sensing_port = node["sensing_port"].as<int>();
                cfg.sensing_rx_channels.push_back(ch);
            }
            if (!has_sensing_count_key) {
                cfg.sensing_rx_channel_count = static_cast<uint32_t>(cfg.sensing_rx_channels.size());
            }
        }
        if (config["default_ip"]) cfg.default_ip = config["default_ip"].as<std::string>();
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
        if (!config_detail::load_data_resource_blocks_from_yaml(cfg, config, "Modulator config")) {
            return false;
        }
        if (config["cpu_cores"]) cfg.available_cores = config["cpu_cores"].as<std::vector<size_t>>();
        return true;
    } catch (const YAML::Exception& e) {
        LOG_G_ERROR() << "Error parsing YAML config: " << e.what();
        return false;
    }
}

inline void normalize_modulator_sensing_channels(Config& cfg) {
    if (cfg.default_ip.empty()) {
        cfg.default_ip = "127.0.0.1";
    }
    if (cfg.range_fft_size == 0) {
        LOG_G_WARN() << "range_fft_size is unset or 0. Defaulting to fft_size=" << cfg.fft_size << '.';
        cfg.range_fft_size = cfg.fft_size;
    }
    if (cfg.doppler_fft_size == 0) {
        LOG_G_WARN() << "doppler_fft_size=0 is invalid. Clamping to 1.";
        cfg.doppler_fft_size = 1;
    }
    if (cfg.sensing_symbol_num == 0) {
        LOG_G_WARN() << "sensing_symbol_num=0 is invalid. Clamping to 1.";
        cfg.sensing_symbol_num = 1;
    }
    if (cfg.doppler_fft_size < cfg.sensing_symbol_num) {
        LOG_G_WARN() << "doppler_fft_size=" << cfg.doppler_fft_size
                     << " is smaller than sensing_symbol_num=" << cfg.sensing_symbol_num
                     << ". Expanding doppler_fft_size to sensing_symbol_num to keep sensing buffers consistent.";
        cfg.doppler_fft_size = cfg.sensing_symbol_num;
    }
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
            LOG_G_WARN() << "Unsupported modulator measurement_mode='"
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
    finalize_data_resource_grid_config(cfg, "Modulator");

    auto make_default_ch0 = [&cfg]() {
        SensingRxChannelConfig ch;
        ch.usrp_channel = 0;
        ch.rx_gain = 30.0;
        ch.alignment = SensingRxChannelConfig{}.alignment;
        ch.clock_source = "";
        ch.time_source = "";
        ch.wire_format_rx = "";
        ch.enable_system_delay_estimation = false;
        ch.enable_sensing_output = cfg.mono_sensing_output_enabled;
        ch.sensing_ip = cfg.mono_sensing_ip.empty() ? cfg.default_ip : cfg.mono_sensing_ip;
        ch.sensing_port = cfg.mono_sensing_port;
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

    for (auto& ch : cfg.sensing_rx_channels) {
        if (ch.sensing_ip.empty()) ch.sensing_ip = cfg.default_ip;
    }

    cfg.sensing_rx_channel_count = static_cast<uint32_t>(cfg.sensing_rx_channels.size());
    if (cfg.sensing_rx_channel_count > 0) {
        const auto& ch0 = cfg.sensing_rx_channels.front();
        cfg.mono_sensing_ip = ch0.sensing_ip;
        cfg.mono_sensing_port = ch0.sensing_port;
    }
}

inline Config make_default_demodulator_config() {
    Config cfg;
    cfg.fft_size = 1024;
    cfg.cp_length = 128;
    cfg.center_freq = 2.4e9;
    cfg.pilot_positions = {571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451};
    cfg.range_fft_size = 1024;
    cfg.doppler_fft_size = 100;
    cfg.num_symbols = 100;
    cfg.sensing_symbol_num = 100;
    cfg.cuda_demod_pipeline_slots = 3;
    cfg.frame_queue_size = 8;
    cfg.sync_queue_size = 8;
    cfg.sync_pos = 1;
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
    cfg.bi_sensing_ip = "";
    cfg.bi_sensing_port = 8889;
    cfg.control_port = 9999;
    cfg.channel_ip = "";
    cfg.channel_port = 12348;
    cfg.pdf_ip = "";
    cfg.pdf_port = 12349;
    cfg.constellation_ip = "";
    cfg.constellation_port = 12346;
    cfg.vofa_debug_ip = "";
    cfg.vofa_debug_port = 12347;
    cfg.udp_output_ip = "";
    cfg.software_sync = true;
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
    cfg.sensing_symbol_stride = 20;
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

inline bool save_demodulator_config_to_yaml(const Config& cfg, const std::string& filepath) {
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "fft_size" << YAML::Value << cfg.fft_size;
    out << YAML::Key << "cp_length" << YAML::Value << cfg.cp_length;
    out << YAML::Key << "sync_pos" << YAML::Value << cfg.sync_pos;
    out << YAML::Key << "sample_rate" << YAML::Value << cfg.sample_rate;
    out << YAML::Key << "bandwidth" << YAML::Value << cfg.bandwidth;
    out << YAML::Key << "center_freq" << YAML::Value << cfg.center_freq;
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
    out << YAML::Key << "frame_queue_size" << YAML::Value << cfg.frame_queue_size;
    out << YAML::Key << "sync_queue_size" << YAML::Value << cfg.sync_queue_size;
    out << YAML::Key << "reset_hold_s" << YAML::Value << cfg.reset_hold_s;
    out << YAML::Key << "range_fft_size" << YAML::Value << cfg.range_fft_size;
    out << YAML::Key << "doppler_fft_size" << YAML::Value << cfg.doppler_fft_size;
    out << YAML::Key << "device_args" << YAML::Value << cfg.device_args;
    out << YAML::Key << "clock_source" << YAML::Value << cfg.clocksource;
    out << YAML::Key << "wire_format_rx" << YAML::Value << cfg.wire_format_rx;
    out << YAML::Key << "enable_bi_sensing" << YAML::Value << cfg.enable_bi_sensing;
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
    out << YAML::Key << "hardware_sync" << YAML::Value << cfg.hardware_sync;
    out << YAML::Key << "hardware_sync_tty" << YAML::Value << cfg.hardware_sync_tty;
    out << YAML::Key << "profiling_modules" << YAML::Value << cfg.profiling_modules;
    out << YAML::Key << "default_ip" << YAML::Value << cfg.default_ip;
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
    out << YAML::Key << "pilot_positions" << YAML::Value << YAML::Flow << cfg.pilot_positions;
    config_detail::emit_data_resource_blocks_yaml(out, cfg);
    out << YAML::Key << "cpu_cores" << YAML::Value << YAML::Flow << cfg.available_cores;
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

inline bool load_demodulator_config_from_yaml(Config& cfg, const std::string& filepath) {
    if (!path_exists(filepath)) {
        return false;
    }
    try {
        YAML::Node config = YAML::LoadFile(filepath);
        if (!config_detail::reject_legacy_key(config, "sensing_ip", "bi_sensing_ip")) return false;
        if (!config_detail::reject_legacy_key(config, "sensing_port", "bi_sensing_port")) return false;

        if (config["fft_size"]) cfg.fft_size = config["fft_size"].as<size_t>();
        if (config["cp_length"]) cfg.cp_length = config["cp_length"].as<size_t>();
        if (config["sync_pos"]) cfg.sync_pos = config["sync_pos"].as<size_t>();
        if (config["sample_rate"]) cfg.sample_rate = config["sample_rate"].as<double>();
        if (config["bandwidth"]) cfg.bandwidth = config["bandwidth"].as<double>();
        if (config["center_freq"]) cfg.center_freq = config["center_freq"].as<double>();
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
        if (config["cuda_demod_pipeline_slots"]) {
            cfg.cuda_demod_pipeline_slots = config["cuda_demod_pipeline_slots"].as<size_t>();
        }
        if (config["frame_queue_size"]) cfg.frame_queue_size = config["frame_queue_size"].as<size_t>();
        if (config["sync_queue_size"]) cfg.sync_queue_size = config["sync_queue_size"].as<size_t>();
        if (config["reset_hold_s"]) cfg.reset_hold_s = config["reset_hold_s"].as<double>();
        if (config["range_fft_size"]) cfg.range_fft_size = config["range_fft_size"].as<size_t>();
        if (config["doppler_fft_size"]) cfg.doppler_fft_size = config["doppler_fft_size"].as<size_t>();
        if (config["device_args"]) cfg.device_args = config["device_args"].as<std::string>();
        if (config["clock_source"]) cfg.clocksource = config["clock_source"].as<std::string>();
        if (config["wire_format_rx"]) cfg.wire_format_rx = config["wire_format_rx"].as<std::string>();
        if (config["enable_bi_sensing"]) cfg.enable_bi_sensing = config["enable_bi_sensing"].as<bool>();
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
        if (config["hardware_sync"]) cfg.hardware_sync = config["hardware_sync"].as<bool>();
        if (config["hardware_sync_tty"]) cfg.hardware_sync_tty = config["hardware_sync_tty"].as<std::string>();
        if (config["profiling_modules"]) cfg.profiling_modules = config["profiling_modules"].as<std::string>();
        if (config["default_ip"]) cfg.default_ip = config["default_ip"].as<std::string>();
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
        if (!config_detail::load_data_resource_blocks_from_yaml(cfg, config, "Demodulator config")) {
            return false;
        }
        if (config["cpu_cores"]) cfg.available_cores = config["cpu_cores"].as<std::vector<size_t>>();
        return true;
    } catch (const YAML::Exception& e) {
        LOG_G_ERROR() << "Error parsing YAML config: " << e.what();
        return false;
    }
}

inline void finalize_demodulator_network_defaults(Config& cfg) {
    if (cfg.range_fft_size == 0) {
        LOG_G_WARN() << "range_fft_size is unset or 0. Defaulting to fft_size=" << cfg.fft_size << '.';
        cfg.range_fft_size = cfg.fft_size;
    }
    if (cfg.doppler_fft_size == 0) {
        LOG_G_WARN() << "doppler_fft_size=0 is invalid. Clamping to 1.";
        cfg.doppler_fft_size = 1;
    }
    if (cfg.sensing_symbol_num == 0) {
        LOG_G_WARN() << "sensing_symbol_num=0 is invalid. Clamping to 1.";
        cfg.sensing_symbol_num = 1;
    }
    if (cfg.doppler_fft_size < cfg.sensing_symbol_num) {
        LOG_G_WARN() << "doppler_fft_size=" << cfg.doppler_fft_size
                     << " is smaller than sensing_symbol_num=" << cfg.sensing_symbol_num
                     << ". Expanding doppler_fft_size to sensing_symbol_num to keep sensing buffers consistent.";
        cfg.doppler_fft_size = cfg.sensing_symbol_num;
    }
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
    if (cfg.measurement_enable) {
        if (cfg.measurement_mode.empty()) {
            cfg.measurement_mode = "internal_prbs";
        }
        if (!measurement_mode_enabled(cfg)) {
            LOG_G_WARN() << "Unsupported demodulator measurement_mode='"
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
    finalize_data_resource_grid_config(cfg, "Demodulator");
    if (cfg.bi_sensing_ip.empty()) cfg.bi_sensing_ip = cfg.default_ip;
    if (cfg.channel_ip.empty()) cfg.channel_ip = cfg.default_ip;
    if (cfg.pdf_ip.empty()) cfg.pdf_ip = cfg.default_ip;
    if (cfg.constellation_ip.empty()) cfg.constellation_ip = cfg.default_ip;
    if (cfg.vofa_debug_ip.empty()) cfg.vofa_debug_ip = cfg.default_ip;
    if (cfg.udp_output_ip.empty()) cfg.udp_output_ip = cfg.default_ip;
}

inline void log_demodulator_sync_mode(const Config& cfg) {
    if (cfg.hardware_sync && cfg.software_sync) {
        LOG_G_INFO() << "Both software_sync and hardware_sync are enabled.";
    } else if (cfg.hardware_sync) {
        LOG_G_INFO() << "Hardware sync enabled.";
    } else if (cfg.software_sync) {
        LOG_G_INFO() << "Software sync enabled.";
    } else {
        LOG_G_WARN() << "Both software_sync and hardware_sync are disabled.";
    }
}

inline void log_demodulator_agc_mode(const Config& cfg) {
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
 * @brief Lock-free SPSC UDP Sender for sensing data.
 *
 * Specialized sender for high-throughput sensing data. A single producer thread
 * enqueues payloads into a bounded lock-free ring, and a background consumer
 * thread handles UDP packetization/transmission asynchronously.
 */
class SensingDataSender : public UdpBaseSender {
public:
    struct FrameData {
        AlignedVector data;
        std::shared_ptr<const void> owner;
        const std::complex<float>* external_data = nullptr;
        size_t external_size = 0;
        uint64_t first_symbol_index = 0; // OFDM symbol index before sparse sampling
    };

    SensingDataSender(const std::string& ip, int port, bool enabled = true) 
        : UdpBaseSender(ip, port),
          _enabled(enabled),
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
        if (!_enabled) return;
        if (!_running.load()) return;

        FrameData frame_data;
        frame_data.data = data;
        frame_data.first_symbol_index = first_symbol_index;
        spsc_wait_push(_data_queue, std::move(frame_data), [this]() {
            return !_running.load(std::memory_order_acquire);
        });
    }

    void push_data(AlignedVector&& data) {
        push_data(std::move(data), 0);
    }

    void push_data(AlignedVector&& data, uint64_t first_symbol_index) {
        if (!_enabled) return;
        if (!_running.load()) return;

        FrameData frame_data;
        frame_data.data = std::move(data);
        frame_data.first_symbol_index = first_symbol_index;
        spsc_wait_push(_data_queue, std::move(frame_data), [this]() {
            return !_running.load(std::memory_order_acquire);
        });
    }

    void push_external(
        std::shared_ptr<const void> owner,
        const std::complex<float>* data,
        size_t count,
        uint64_t first_symbol_index
    ) {
        if (!_enabled) return;
        if (!_running.load()) return;

        FrameData frame_data;
        frame_data.owner = std::move(owner);
        frame_data.external_data = data;
        frame_data.external_size = count;
        frame_data.first_symbol_index = first_symbol_index;
        spsc_wait_push(_data_queue, std::move(frame_data), [this]() {
            return !_running.load(std::memory_order_acquire);
        });
    }

private:
    bool _enabled = true;
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

            // Send data if available
            const bool has_inline = !frame_data.data.empty();
            const bool has_external = frame_data.external_data != nullptr && frame_data.external_size > 0;
            if (has_inline || has_external) {
                send_data_with_original_format(frame_data);
            }
        }
    }

    void send_data_with_original_format(const AlignedVector& data, uint64_t first_symbol_index) {
        send_data_with_original_format_impl(data.data(), data.size(), first_symbol_index);
    }

    void send_data_with_original_format(const FrameData& frame_data) {
        if (frame_data.external_data != nullptr && frame_data.external_size > 0) {
            send_data_with_original_format_impl(
                frame_data.external_data,
                frame_data.external_size,
                frame_data.first_symbol_index
            );
            return;
        }
        send_data_with_original_format_impl(
            frame_data.data.data(),
            frame_data.data.size(),
            frame_data.first_symbol_index
        );
    }

    void send_data_with_original_format_impl(
        const std::complex<float>* data_ptr,
        size_t data_size,
        uint64_t first_symbol_index
    ) {
        const size_t chunk_size = 60000;
        size_t total_chunks = (data_size * sizeof(std::complex<float>) + chunk_size - 1) / chunk_size;
        // Packet Header: [Frame ID | Total Chunks | Current Chunk Index]
        // Reuse frame_id field to carry first OFDM symbol index (lower 32 bits)
        const uint32_t frame_id = static_cast<uint32_t>(first_symbol_index & 0xFFFFFFFFu);
        
        for (size_t i = 0; i < total_chunks; i++) {
            // Prepare packet header
            uint32_t header[3] = {
                htonl(frame_id),
                htonl(static_cast<uint32_t>(total_chunks)),
                htonl(static_cast<uint32_t>(i))
            };
            // Calculate current chunk size
            size_t offset = i * chunk_size;
            size_t remaining = data_size * sizeof(std::complex<float>) - offset;
            size_t current_chunk_size = (remaining < chunk_size) ? remaining : chunk_size;
            
            // Construct complete packet
            std::vector<uint8_t> packet(sizeof(header) + current_chunk_size);
            memcpy(packet.data(), header, sizeof(header));
            memcpy(packet.data() + sizeof(header),
                  reinterpret_cast<const uint8_t*>(data_ptr) + offset,
                  current_chunk_size);
            
            try {
                send_raw(packet.data(), packet.size());
            } catch (const std::exception& e) {
                LOG_G_WARN() << "UDP send error: " << e.what();
            }
        }
    }
    
    std::atomic<bool> _running{false};
    SPSCRingBuffer<FrameData> _data_queue;
    std::thread _send_thread;
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
class ControlCommandHandler {
public:
    using Callback = std::function<void(int32_t value)>;
    
    // Command structure definition
    #pragma pack(push, 1)
    struct ControlCommand {
        char header[4]; // "CMD "
        char command[4]; // Command ID
        int32_t value;  // Parameter value
    };
    #pragma pack(pop)
    
    static_assert(sizeof(ControlCommand) == 12, "ControlCommand size mismatch");

    ControlCommandHandler(int port) : _port(port), _running(false) {
        // Create and bind socket
        _create_socket();
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
        close(_socket);
        if (_thread.joinable()) {
            _thread.join();
        }
    }

    // Register command handler
    void register_command(const std::string& command, Callback callback) {
        std::lock_guard<std::mutex> lock(_mutex);
        _handlers[command] = std::move(callback);
    }

    // Send heartbeat to the sensing receiver to maintain NAT mapping
    void send_heartbeat(const std::string& dest_ip, int dest_port) {
        if (_socket < 0) return;

        struct sockaddr_in dest_addr;
        memset(&dest_addr, 0, sizeof(dest_addr));
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(dest_port);
        if (inet_pton(AF_INET, dest_ip.c_str(), &dest_addr.sin_addr) <= 0) {
            return;
        }

        // Send a "CTRL_RDY" packet (Header + "RDY " + 0)
        ControlCommand hb;
        memcpy(hb.header, "CTRL", 4);
        memcpy(hb.command, "RDY ", 4);
        hb.value = 0; // Value doesn't matter

        sendto(_socket, &hb, sizeof(hb), 0, (struct sockaddr*)&dest_addr, sizeof(dest_addr));
    }

private:
    void _create_socket() {
        _socket = socket(AF_INET, SOCK_DGRAM, 0);
        if (_socket < 0) {
            throw std::runtime_error("Failed to create control socket: " + std::string(strerror(errno)));
        }
        
        // Set socket options
        int opt = 1;
        setsockopt(_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        // Bind to port
        struct sockaddr_in serv_addr;
        memset(&serv_addr, 0, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr.s_addr = INADDR_ANY;
        serv_addr.sin_port = htons(_port);
        
        if (bind(_socket, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            throw std::runtime_error("Control socket bind failed: " + std::string(strerror(errno)));
        }
        
        // Set timeout options
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 10000; // 10ms timeout
        setsockopt(_socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }

    void _run() {
        async_logger::LoggerThreadModeGuard log_mode_guard(async_logger::LoggerThreadMode::NonRealtime);
        ControlCommand cmd;
        struct sockaddr_in cli_addr;
        socklen_t cli_len = sizeof(cli_addr);
        
        while (_running) {
            ssize_t n = recvfrom(_socket, &cmd, sizeof(cmd), 0,
                                (struct sockaddr*)&cli_addr, &cli_len);
            
            if (n <= 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue; // Normal timeout
                } else if (errno != 0) {
                    LOG_G_WARN() << "Control recv error: " << strerror(errno);
                    continue;
                }
            }
            
            if (n != sizeof(cmd)) {
                LOG_G_WARN() << "Received partial command (" << n << " bytes)";
                continue;
            }
            
            // Verify frame header
            if (std::string(cmd.header, 4) != "CMD ") {
                LOG_G_WARN() << "Invalid command header received";
                continue;
            }
            
            int32_t value = ntohl(cmd.value); // Network to host byte order
            std::string command_str(cmd.command, 4);
            
            // Find and execute handler
            {
                std::lock_guard<std::mutex> lock(_mutex);
                auto it = _handlers.find(command_str);
                if (it != _handlers.end()) {
                    try {
                        it->second(value);
                    } catch (const std::exception& e) {
                        LOG_G_WARN() << "Error processing command '" << command_str 
                                     << "': " << e.what();
                    }
                } else {
                    LOG_G_WARN() << "Unknown command: " << command_str;
                }
            }
        }
    }
    
    int _port;
    int _socket = -1;
    std::atomic<bool> _running{false};
    std::thread _thread;
    std::mutex _mutex;
    std::unordered_map<std::string, Callback> _handlers;
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
    HardwareSyncController(const std::string& device_path) 
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
