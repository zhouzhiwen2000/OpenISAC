#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief QPSK Scrambler/Descrambler.
 * 
 * Uses a Linear Feedback Shift Register (LFSR) to generate a pseudo-random sequence
 * for scrambling and descrambling bits. This helps in randomizing the data to avoid
 * long sequences of zeros or ones, which avoids high PAPRs in OFDM.
 */
class Scrambler {
public:
    Scrambler(size_t max_bits, uint8_t init = 0x5A)
        : scramble_seq_(max_bits)
    {
        uint8_t lfsr = init;
        for (size_t i = 0; i < max_bits; ++i) {
            scramble_seq_[i] = ((lfsr >> 7) ^ (lfsr >> 3) ^ (lfsr >> 2) ^ (lfsr >> 1)) & 1;
            lfsr = ((lfsr << 1) | scramble_seq_[i]) & 0xFF;
        }
    }

    // Scramble (in-place)
    template<typename Vec>
    void scramble(Vec& bits) const {
        size_t n = bits.size();
        size_t m = std::min(n, scramble_seq_.size());
        #pragma omp simd
        for (size_t i = 0; i < m; ++i) {
            bits[i] ^= scramble_seq_[i];
        }
    }

    // Descramble (in-place)
    template<typename Vec>
    void descramble(Vec& bits) const {
        scramble(bits); // Same as scrambling
    }

    // Soft descramble (descramble LLR values)
    template<typename FloatVec>
    void soft_descramble(FloatVec& llr_values) const {
        size_t n = llr_values.size();
        size_t m = std::min(n, scramble_seq_.size());
        #pragma omp simd
        for (size_t i = 0; i < m; ++i) {
            if (scramble_seq_[i] == 1) {
                llr_values[i] = -llr_values[i]; // Flip LLR sign if scramble bit is 1
            }
            // Keep LLR as is if scramble bit is 0
        }
    }

private:
    std::vector<uint8_t> scramble_seq_;
};
#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <complex>
#include <functional>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <cstdlib>
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
#include <uhd/utils/thread.hpp>
#include <aff3ct.hpp>
#include <fcntl.h>
#include <termios.h>
#include <iomanip>
#include <sstream>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cstdint>


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
    size_t sync_pos = 1;               // Synchronization symbol position
    int delay_adjust_step = 2;         // Delay adjustment step
    
    double sample_rate = 50e6;         // Sample rate
    double bandwidth = 50e6;           // Bandwidth
    double center_freq = 2.4e9;        // Center frequency

    double tx_gain = 0.0;              // TX gain
    double rx_gain = 0.0;              // RX gain (Channel 1)
    double rx_gain2 = 0.0;             // RX gain (Channel 2)
    uint32_t tx_channel = 0;           // TX channel index
    size_t rx_channel = 0;             // RX channel index
    size_t rx_channel2 = 1;            // Second RX channel index
    int zc_root = 29;                  // Zadoff-Chu sequence root index
    bool software_sync = true;         // Software synchronization flag
    bool hardware_sync = false;        // Hardware synchronization flag
    std::string hardware_sync_tty = "/dev/ttyUSB0"; // Hardware sync TTY device
    double ppm_adjust_factor = 0.01;
    double delay_ppm_adjust_factor = 0.01;

    std::vector<size_t> pilot_positions = {571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451};
    std::string device_args = "";
    std::string default_ip = "127.0.0.1";
    std::string mono_sensing_ip = "127.0.0.1";
    int mono_sensing_port = 8888;
    std::string mono_sensing_ip2 = "127.0.0.1";
    int mono_sensing_port2 = 8890;
    std::string bi_sensing_ip = "127.0.0.1";
    int bi_sensing_port = 8889;
    int control_port = 9999;
    std::string channel_ip = "127.0.0.1";
    int channel_port = 12348;
    std::string pdf_ip = "127.0.0.1";
    int pdf_port = 12349;
    std::string constellation_ip = "127.0.0.1";
    int constellation_port = 12346;
    std::string freq_offset_ip = "127.0.0.1";
    int freq_offset_port = 12347;
    // UDP output for decoded payloads
    std::string udp_output_ip = "127.0.0.1";
    int udp_output_port = 50001;
    // Modulator UDP input (for incoming payloads to modulate)
    std::string modulator_udp_ip = "0.0.0.0"; // bind address
    int modulator_udp_port = 50000;
    std::string clocksource = "internal"; // Clock source
    int system_delay = 63; // System delay (for alignment)
    std::string wire_format_tx = "sc16";
    std::string wire_format_rx = "sc16";
    std::vector<size_t> available_cores = {0, 1, 2, 3, 4, 5};
    std::string profiling_modules = "";  // Comma-separated list of modules to profile: modulation, sensing_proc, sensing_process, data_ingest, demodulation, or "all"
    
    // Check if a specific module should be profiled
    bool should_profile(const std::string& module) const {
        if (profiling_modules.empty()) return false;
        if (profiling_modules == "all") return true;
        return profiling_modules.find(module) != std::string::npos;
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
/**
 * @brief Received Data Frame Structure.
 * 
 * Contains the raw time-domain samples of a received OFDM frame and alignment information.
 */
struct RxFrame {
    AlignedVector frame_data;  // Received symbol collection
    int Alignment;
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
            std::cerr << "Warning: Failed to set send buffer size" << std::endl;
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
 * @brief Thread-safe UDP Sender for Sensing Data.
 * 
 * Specialized sender for high-throughput sensing data.
 * Features a circular buffer and a background thread to handle data transmission asynchronously,
 * minimizing impact on the main processing loop. Handles data packetization and fragmentation.
 */
class SensingDataSender : public UdpBaseSender {
public:
    SensingDataSender(const std::string& ip, int port, int buffer_size) 
        : UdpBaseSender(ip, port),
          _buffer_size(buffer_size) 
    {
        // Initialize data queue (capacity 64)
        _data_queue = std::make_unique<boost::circular_buffer<AlignedVector>>(64);
    }

    ~SensingDataSender() {
        stop();
    }

    void start() {
        if (_running.load()) return;
        _running.store(true);
        _send_thread = std::thread(&SensingDataSender::run, this);
    }

    void stop() {
        if (!_running.exchange(false)) return;
        _cond.notify_all();
        if (_send_thread.joinable()) {
            _send_thread.join();
        }
    }

    void push_data(const AlignedVector& data) {
        if (!_running.load()) return;
        
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_data_queue->full()) {
                // Drop oldest data when queue is full
                _data_queue->pop_front();
            }
            _data_queue->push_back(data);
        }
        _cond.notify_one();
    }

private:
    void run() {
        uhd::set_thread_priority_safe();
        
        while (_running.load(std::memory_order_acquire)) {
            AlignedVector data;
            bool has_data = false;
            
            {
                std::unique_lock<std::mutex> lock(_mutex);
                if (_cond.wait_for(lock, std::chrono::milliseconds(100), 
                                   [this]{ 
                                       return !_running.load(std::memory_order_relaxed) || 
                                              !_data_queue->empty(); 
                                   })) 
                {
                    // Check if should exit
                    if (!_running.load(std::memory_order_relaxed)) break;
                    
                    // Get data from queue
                    if (!_data_queue->empty()) {
                        data = std::move(_data_queue->front());
                        _data_queue->pop_front();
                        has_data = true;
                    }
                }
            }
            
            // Send data if available
            if (has_data && !data.empty()) {
                send_data_with_original_format(data);
            }
        }
    }

    void send_data_with_original_format(const AlignedVector& data) {
        const size_t chunk_size = 60000;
        size_t total_chunks = (data.size() * sizeof(std::complex<float>) + chunk_size - 1) / chunk_size;
        // Packet Header: [Frame ID | Total Chunks | Current Chunk Index]
        static uint32_t frame_id = 0;
        frame_id++;
        
        for (size_t i = 0; i < total_chunks; i++) {
            // Prepare packet header
            uint32_t header[3] = {
                htonl(frame_id),
                htonl(static_cast<uint32_t>(total_chunks)),
                htonl(static_cast<uint32_t>(i))
            };
            // Calculate current chunk size
            size_t offset = i * chunk_size;
            size_t remaining = data.size() * sizeof(std::complex<float>) - offset;
            size_t current_chunk_size = (remaining < chunk_size) ? remaining : chunk_size;
            
            // Construct complete packet
            std::vector<uint8_t> packet(sizeof(header) + current_chunk_size);
            memcpy(packet.data(), header, sizeof(header));
            memcpy(packet.data() + sizeof(header),
                  reinterpret_cast<const uint8_t*>(data.data()) + offset,
                  current_chunk_size);
            
            try {
                send_raw(packet.data(), packet.size());
            } catch (const std::exception& e) {
                std::cerr << "UDP send error: " << e.what() << std::endl;
            }
        }
    }
    
    int _buffer_size;
    std::atomic<bool> _running{false};
    std::unique_ptr<boost::circular_buffer<AlignedVector>> _data_queue;
    std::mutex _mutex;
    std::condition_variable _cond;
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
    using QueueType = boost::circular_buffer<DataPackage>;
    using SendFunction = std::function<void(const DataPackage&)>;

    DataSender(size_t queue_size, SendFunction send_func, std::chrono::milliseconds interval)
        : queue_(queue_size),
          send_func_(std::move(send_func)),
          send_interval_(interval),
          running_(false) {}

    ~DataSender() {
        stop();
    }

    void add_data(DataPackage&& data) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_.push_back(std::move(data));
        }
        queue_cv_.notify_one();
    }

    void start() {
        if (running_) return;
        running_.store(true);
        thread_ = std::thread(&DataSender::sender_thread, this);
    }

    void stop() {
        if (!running_) return;
        running_.store(false);
        queue_cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    void sender_thread() {
        auto next_send = std::chrono::steady_clock::now();
        
        while (running_.load()) {
            DataPackage data;
            bool has_data = false;
            
            // Wait for data or timeout
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (queue_cv_.wait_until(lock, next_send, 
                    [&]{ return !queue_.empty() || !running_.load(); })) {
                    if (!running_.load()) break;
                    
                    if (!queue_.empty()) {
                        data = std::move(queue_.front());
                        queue_.pop_front();
                        has_data = true;
                    }
                }
            }
            
            // Send data if available
            if (has_data) {
                try {
                    send_func_(data);
                } catch (const std::exception& e) {
                    std::cerr << "DataSender error: " << e.what() << std::endl;
                }
            }
            
            // Update next send time
            next_send += send_interval_;
            std::this_thread::sleep_until(next_send);
        }
    }

    QueueType queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    SendFunction send_func_;
    std::chrono::milliseconds send_interval_;
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
                    std::cerr << "Control recv error: " << strerror(errno) << std::endl;
                    continue;
                }
            }
            
            if (n != sizeof(cmd)) {
                std::cerr << "Received partial command (" << n << " bytes)" << std::endl;
                continue;
            }
            
            // Verify frame header
            if (std::string(cmd.header, 4) != "CMD ") {
                std::cerr << "Invalid command header received" << std::endl;
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
                        std::cerr << "Error processing command '" << command_str 
                                  << "': " << e.what() << std::endl;
                    }
                } else {
                    std::cerr << "Unknown command: " << command_str << std::endl;
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
 * @brief Weighted Linear Regression.
 * 
 * Calculates the slope (beta) and intercept (alpha) of a line that best fits the
 * weighted input data points using the least squares method. 
 * Used for estimating SFO/SIO and CFO effects where some pilot subcarriers might have higher SNR (weight).
 * 
 * @return std::pair<float, float> {slope (beta), intercept (alpha)}
 */
template <typename T>
std::pair<float, float> weightedlinearRegression(const std::vector<T>& x_values,
                                         const std::vector<float>& y_values,
                                         const std::vector<float>& weights) {
    // Validate input data size and weights match
    if (x_values.size() != y_values.size() || 
        x_values.size() != weights.size() || 
        x_values.empty()) {
        return std::make_pair(0.0f, 0.0f);
    }
    // Initialize weighted sum variables
    float sum_w = 0.0f;
    float sum_wx = 0.0f;
    float sum_wy = 0.0f;
    float sum_wxx = 0.0f;
    float sum_wxy = 0.0f;
    const int N = x_values.size();

    // Calculate weighted sums
    for (int i = 0; i < N; ++i) {
        const float w = weights[i];
        const float x = static_cast<float>(x_values[i]);
        const float y = y_values[i];
        
        sum_w += w;
        sum_wx += w * x;
        sum_wy += w * y;
        sum_wxx += w * x * x;
        sum_wxy += w * x * y;
    }

    // Calculate weighted least squares slope and intercept
    float beta = 0.0f, alpha = 0.0f;
    float denom = sum_w * sum_wxx - sum_wx * sum_wx;
    
    if (std::abs(denom) > 1e-10f) {
        beta = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
        alpha = (sum_wy - beta * sum_wx) / sum_w;
    }

    return std::make_pair(beta, alpha);
}

/**
 * @brief Standard Linear Regression.
 * 
 * Calculates the slope (beta) and intercept (alpha) for unweighted data.
 * Used for estimating SFO/SIO across frames using timing offset estimates where all 
 * measurements are treated equally.
 * 
 * @return std::pair<float, float> {slope (beta), intercept (alpha)}
 */
template <typename T>
std::pair<float, float> linearRegression(const std::vector<T>& x_values,
                                         const std::vector<float>& y_values) {
    // Validate input data size size matches and not empty
    if (x_values.size() != y_values.size() || x_values.empty()) {
        return std::make_pair(0.0f, 0.0f);
    }
    // Initialize sum variables
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_xx = 0.0f;
    float sum_xy = 0.0f;
    const int N = x_values.size();

    // Calculate sums
    for (int i = 0; i < N; ++i) {
        const float x = static_cast<float>(x_values[i]);
        const float y = y_values[i];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    // Calculate least squares slope and intercept
    float beta = 0.0f, alpha = 0.0f;
    float denom = N * sum_xx - sum_x * sum_x;  // Denominator = n*Σx² - (Σx)²
    if (std::abs(denom) > 1e-10f) {
        beta = (N * sum_xy - sum_x * sum_y) / denom;
        alpha = (sum_y - beta * sum_x) / N;
    }

    return std::make_pair(beta, alpha);
}

/**
 * @brief Finite Impulse Response (FIR) Filter.
 * 
 * Implements a standard FIR filter with a circular buffer for efficiency.
 * Used for signal smoothing or filtering.
 */
class FIRFilter {
private:
    std::vector<float> coeffs;  // Filter coefficients
    std::vector<float> buffer;  // Data buffer
    size_t order;               // Filter order
    size_t index;               // Current write index

public:
    FIRFilter(const std::vector<float>& coefficients) 
        : coeffs(coefficients), 
          buffer(coefficients.size(), 0.0f),
          order(coefficients.size()),
          index(0) {}

    float process(float input) {
        buffer[index] = input;
        index = (index + 1) % order;  // Update index

        float output = 0.0f;
        size_t i = index;  // Start from oldest data point

        for (size_t j = 0; j < order; j++) {
            output += coeffs[j] * buffer[i];
            i = (i + 1) % order;  // Move to next newer sample
        }

        return output;
    }
    
    // Warm up the buffer to avoid initial transients
    void warm_up(float value, size_t samples = 50) {
        for (size_t i = 0; i < samples; i++) {
            process(value);
        }
    }
};

/**
 * @brief Sampling Frequency Offset (SFO) / Sampling Interval Offset (SIO) Estimator.
 * 
 * Estimates the SFO/SIO by tracking the drift of timing offsets over time.
 * Uses a estimation window of timing offset measurements and performs 
 * linear regression to determine the rate of change (SIO). Also 
 * incorporates a control loop to adjust synchronization.
 */
class SFOEstimator {
public:
    explicit SFOEstimator(size_t window_size) 
        : _window_size(window_size),
          _delay_offsets(window_size, 0.0f),
          _delay_offsets_indices(window_size) {
        // Initialize index array
        std::iota(_delay_offsets_indices.begin(), _delay_offsets_indices.end(), 0);
        reset();
    }

    void reset() {
        _count = 0;
        _cumulative_delay_offset = 0.0f;
        _sfo_per_frame = 0.0f;
        std::fill(_delay_offsets.begin(), _delay_offsets.end(), 0.0f);
    }

    // Update delay offset estimation and calculate SFO/SIO
    void update(float delay_offset_reading, float Alignment) {

        if (!_first_delay_offset_reading) {
            if (Alignment != 0.0f){
                //std::cout << "Alignment: " << Alignment << std::endl;
                _cumulative_delay_offset += Alignment;
            }
            _delay_offsets[_count] = delay_offset_reading + _cumulative_delay_offset;
            // Perform linear regression when window is full
            if (++_count >= _window_size) {
                _sfo_per_frame = linearRegression(_delay_offsets_indices, _delay_offsets).first;
                _count = 0;
                //std::cout << "SFO per frame: " << _sfo_per_frame << std::endl;
                _cumulative_delay_offset = 0.0f;
            }
            if (abs(_sfo_per_frame)>1.0f) {
                _sfo_per_frame = 0.0f;
            }
            _cumulative_sensing_delay_offset += _sfo_per_frame;
            _cumulative_sensing_delay_offset -= Alignment;
            auto err = delay_offset_reading - _cumulative_sensing_delay_offset;
            if (abs(err) > 0.1f) {
                _err_large_count++;
                if (_err_large_count > 100) {
                    _pd = 1e-2; // Increase proportional gain
                }
            } else {
                _err_large_count = 0;
                _pd = 1e-5; // Restore default proportional gain
            }
            _cumulative_sensing_delay_offset += _pd * err;
        }
        if (_first_delay_offset_reading){
            _count ++;
            if (_count >= 10) {
                _first_delay_offset_reading = false;
                _count = 0;
            }
        }

    }

    float get_sfo_per_frame() const { return _sfo_per_frame; }
    float get_sensing_delay_offset() const { return _cumulative_sensing_delay_offset; }

private:
    size_t _window_size;      // Regression window size
    size_t _count = 0;        // Count of readings in current window
    size_t _err_large_count = 0; // Large error counter
    float _pd = 1e-5;         // Proportional gain, used to adjust sensing delay offset accumulation
    
    std::vector<float> _delay_offsets;
    std::vector<int> _delay_offsets_indices;
    
    float _cumulative_delay_offset = 0.0f;
    bool _first_delay_offset_reading = true;
    float _sfo_per_frame = 0.0f;  // Delay offset per frame due to SFO
    float _cumulative_sensing_delay_offset = 0.0f; // Accumulated sensing delay offset
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
            std::cerr << "ERROR: Failed to open serial device: " << strerror(errno) << std::endl;
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
            std::cerr << "ERROR: Failed to set serial attributes: " << strerror(errno) << std::endl;
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

    void set_frequency_control_ppm_relative(double delta_ppm, double extra_ppm = 0) {
        int32_t control_word = ppm_to_control_word_relative(delta_ppm);
        hw_freq_word += control_word;
        set_frequency_control_word(hw_freq_word + extra_ppm);
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

    // Worker thread function - handles serial communication
    void worker_thread_func() {
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
                    std::cerr << "ERROR: Serial write error: " << strerror(errno) 
                              << " [command: " << command << "]" << std::endl;
                    return;
                }
                // Wait for buffer availability
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        
        if (remaining > 0) {
            std::cerr << "ERROR: Failed to send full command after 3 attempts: " 
                      << command << std::endl;
            return;
        }
        
        // 2. Wait for response
        char response[32] = {0};
        int total_read = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        while (true) {
            // Timeout check (500ms)
            if (std::chrono::steady_clock::now() - start_time > std::chrono::milliseconds(500)) {
                std::cerr << "ERROR: Response timeout for command: " << command << std::endl;
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
                    std::cerr << "ERROR: Serial read error: " << strerror(errno) << std::endl;
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
            std::cerr << "ERROR: Device rejected command - BAD DATA: " << command << std::endl;
        }
        else if (strncmp(response, "BAD CMD", 7) == 0) {
            std::cerr << "ERROR: Device rejected command - BAD CMD: " << command << std::endl;
        }
        else if (strncmp(response, "TIMEOUT", 7) == 0) {
            std::cerr << "ERROR: Device operation timed out for command: " << command << std::endl;
        }
        else {
            std::cerr << "ERROR: Unexpected device response: " << response 
                      << " for command: " << command << std::endl;
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

/**
 * @brief LDPC Codec Wrapper (using aff3ct library).
 * 
 * Provides a simplified interface for LDPC encoding and decoding using the aff3ct library.
 * Handles bit packing/unpacking and interfacing with the aff3ct encoder/decoder objects.
 */
class LDPCCodec {
public:
    using AlignedByteVector = mipp::vector<int8_t>;
    using AlignedIntVector = mipp::vector<int32_t>;
    using AlignedFloatVector = mipp::vector<float>;
    struct LDPCConfig {
        std::string h_matrix_path = "../LDPC_504_1008.alist";
        int decoder_iterations = 10;
        size_t n_frames = 1;
    };

    LDPCCodec(const LDPCConfig& config) : cfg(config) {
        // Resolve relative path against executable directory
        if (!cfg.h_matrix_path.empty() && cfg.h_matrix_path[0] != '/') {
            cfg.h_matrix_path = get_executable_dir() + "/" + cfg.h_matrix_path;
        }
        _init_aff3ct_params();
        codec = std::unique_ptr<aff3ct::tools::Codec_LDPC<int, float>>(codec_factory->build<int, float>());
        codec->set_n_frames(cfg.n_frames);
    }

    void encode_frame(const AlignedByteVector& input, AlignedIntVector& encoded_bits) {
        auto& encoder = codec->get_encoder();
        // Unpack input data
        AlignedIntVector unpacked_bits(input.size() * 8);
        _unpack_bits(input, unpacked_bits);
        size_t N = encoder.get_N();
        size_t K = encoder.get_K();
        size_t n_frames_per_wave = encoder.get_n_frames_per_wave();
        size_t encoded_bits_length = unpacked_bits.size()*N/K;
        encoded_bits.resize(encoded_bits_length);
        for (size_t i = 0; i < unpacked_bits.size()/K; i+=n_frames_per_wave) {
            //encoder.encode(unpacked_bits.data()+i*K, encoded_bits.data()+i*N,i,false);
            encoder.encode(unpacked_bits.data()+i*K, encoded_bits.data()+i*N,-1,false);
        }
    }

    // Decode entire frame
    void decode_frame(const AlignedFloatVector& llr_input, AlignedByteVector& decoded_bytes) {
        auto decoder = codec->get_decoder_siho();
        size_t N = decoder.get_N();
        size_t K = decoder.get_K();
        size_t n_frames_per_wave = decoder.get_n_frames_per_wave();
        AlignedIntVector decoded_bits(llr_input.size()* K/N);
        for (size_t i = 0; i < llr_input.size()/N; i+=n_frames_per_wave) {
            //decoder.decode_siho(llr_input.data()+i*N, decoded_bits.data()+i*K,i,false);
            decoder.decode_siho(llr_input.data()+i*N, decoded_bits.data()+i*K,-1,false);
        }
        decoded_bytes.resize(decoded_bits.size() / 8);
        _pack_bits(decoded_bits, decoded_bytes);
    }
    
    size_t get_K() const {
        return codec->get_encoder().get_K();
    }
    size_t get_N() const {
        return codec->get_encoder().get_N();
    }

private:
    LDPCConfig cfg;
    std::unique_ptr<aff3ct::factory::Codec_LDPC> codec_factory;
    std::unique_ptr<aff3ct::tools::Codec_LDPC<int, float>> codec;
    void _init_aff3ct_params() {
        // Use basic code configuration (504, 1008)
        std::vector<std::string> args = {
            "LDPCEncoder",
            "--enc-type", "LDPC_H",
            "--enc-g-method", "IDENTITY",
            "--dec-type", "BP_FLOODING", // BP_FLOODING
            "--dec-implem", "GALA", //OMS
            "--dec-ite", std::to_string(cfg.decoder_iterations),
            "--dec-synd-depth", "1",
            "--dec-h-path", cfg.h_matrix_path,
//            "--dec-simd", "INTER",
//            "--enc-g-path", "../PEGReg504x1008_Gen.alist",
        };

        codec_factory = std::make_unique<aff3ct::factory::Codec_LDPC>();
        
        std::vector<char*> argv;
        for (auto& arg : args)
            argv.push_back(&arg[0]);
        int argc = static_cast<int>(argv.size());
        
        std::vector<aff3ct::factory::Factory*> params_list;
        params_list.push_back(codec_factory.get());
        
        aff3ct::tools::Command_parser cp(argc, argv.data(), params_list, true);
        aff3ct::tools::Header::print_parameters(params_list);
    }

    void _unpack_bits(const AlignedByteVector& input_data, AlignedIntVector& unpacked_bits) {
        const int input_bytes = input_data.size();
        #pragma omp simd
        for (int byte_idx = 0; byte_idx < input_bytes; ++byte_idx) {
            uint8_t byte = input_data[byte_idx];
            unpacked_bits[byte_idx * 8 + 0] = (byte >> 7) & 1;
            unpacked_bits[byte_idx * 8 + 1] = (byte >> 6) & 1;
            unpacked_bits[byte_idx * 8 + 2] = (byte >> 5) & 1;
            unpacked_bits[byte_idx * 8 + 3] = (byte >> 4) & 1;
            unpacked_bits[byte_idx * 8 + 4] = (byte >> 3) & 1;
            unpacked_bits[byte_idx * 8 + 5] = (byte >> 2) & 1;
            unpacked_bits[byte_idx * 8 + 6] = (byte >> 1) & 1;
            unpacked_bits[byte_idx * 8 + 7] = (byte >> 0) & 1;
        }
    }

    void _pack_bits(const AlignedIntVector& bits, AlignedByteVector& output_data) {
        const int output_bytes = output_data.size();
        #pragma omp simd
        for (int byte_idx = 0; byte_idx < output_bytes; ++byte_idx) {
            uint8_t byte = 0;
            byte |= (bits[byte_idx * 8 + 0] & 1) << 7;
            byte |= (bits[byte_idx * 8 + 1] & 1) << 6;
            byte |= (bits[byte_idx * 8 + 2] & 1) << 5;
            byte |= (bits[byte_idx * 8 + 3] & 1) << 4;
            byte |= (bits[byte_idx * 8 + 4] & 1) << 3;
            byte |= (bits[byte_idx * 8 + 5] & 1) << 2;
            byte |= (bits[byte_idx * 8 + 6] & 1) << 1;
            byte |= (bits[byte_idx * 8 + 7] & 1) << 0;
            output_data[byte_idx] = byte;
        }
    }
};

/**
 * @brief QC-LDPC Bit Packing for QPSK.
 * 
 * Packs a vector of bits into QPSK symbols (0-3).
 * Each pair of bits corresponds to one QPSK symbol.
 * Optimized with SIMD for performance.
 */
inline void _pack_bits_qpsk(const LDPCCodec::AlignedIntVector &bits, LDPCCodec::AlignedIntVector &qpsk_ints) {
    const size_t bit_count = bits.size();
    const size_t symbol_count = (bit_count + 1) / 2; // Pad with 0 if odd
    qpsk_ints.resize(symbol_count);
    const size_t even_pairs = bit_count / 2; // Number of complete pairs
    // Main loop: Process 2 bits -> 1 QPSK symbol each time
    #pragma omp simd
    for (size_t k = 0; k < even_pairs; ++k) {
        int b0 = bits[2*k] & 1;
        int b1 = bits[2*k + 1] & 1;
        qpsk_ints[k] = (b0 << 1) | b1;
    }
    // Trailing bit if odd (Rarely triggered, not in SIMD loop)
    if (bit_count & 1) {
        qpsk_ints[even_pairs] = (bits[bit_count - 1] & 1) << 1;
    }
}

/**
 * @brief Phase Unwrapping Function.
 * 
 * Unwraps the phase values in a vector to eliminate 2*pi jumps.
 * Essential for accurate frequency offset estimation from phase differences.
 * Uses SIMD-friendly implementation.
 */
inline void unwrap(std::vector<float>& phase) {
    if (phase.size() > 1) {
        std::vector<float> diffs(phase.size());
        
        // 1. Calculate and wrap differences (SIMD-friendly)
        #pragma omp simd
        for (size_t i = 1; i < phase.size(); ++i) {
            float d = phase[i] - phase[i - 1];
            // Robust wrapping using round for SIMD compatibility
            float k = std::round(d / (2 * (float)M_PI));
            d -= k * 2 * (float)M_PI;
            diffs[i] = d;
        }

        // 2. Integrate unwrapped phase (Sequential)
        for (size_t i = 1; i < phase.size(); ++i) {
            phase[i] = phase[i - 1] + diffs[i];
        }
    }
}

/**
 * @brief Moving Target Indication (MTI) Filter.
 * 
 * Implements an 8-stage IIR filter for clutter suppression in sensing applications.
 * Uses AVX/OpenMP SIMD for efficient processing of subcarriers.
 */
class MTIFilter {
public:
    MTIFilter(size_t range_fft_size = 1024) {
        resize(range_fft_size);
    }

    void resize(size_t range_fft_size) {
        _range_fft_size = range_fft_size;
        // 8 stages, 2 states per stage (s0, s1), x range_fft_size
        _state.resize(8 * 2 * _range_fft_size, std::complex<float>(0.0f, 0.0f));
        reset();
    }

    void reset() {
        std::fill(_state.begin(), _state.end(), std::complex<float>(0.0f, 0.0f));
    }

    /**
     * @brief Apply MTI filter to the buffer.
     * 
     * @param buffer Input/Output buffer (Channel Response)
     * @param N_proc Number of start subcarriers to process (e.g., fft_size). Data beyond this (zero-padding) is skipped.
     * @param num_symbols Number of symbols in the buffer.
     */
    void apply(AlignedVector& buffer, size_t N_proc, size_t num_symbols) {
        static const float SOS[8][6] = {
            {0.993542f, -1.987084f, 0.993542f, 1.000000f, -1.993112f, 0.993951f},
            {0.981889f, -1.963778f, 0.981889f, 1.000000f, -1.981389f, 0.982224f},
            {0.971190f, -1.942380f, 0.971190f, 1.000000f, -1.970564f, 0.971395f},
            {0.961860f, -1.923721f, 0.961860f, 1.000000f, -1.961077f, 0.961903f},
            {0.954245f, -1.908491f, 0.954245f, 1.000000f, -1.953298f, 0.954121f},
            {0.948614f, -1.897228f, 0.948614f, 1.000000f, -1.947526f, 0.948346f},
            {0.945158f, -1.890316f, 0.945158f, 1.000000f, -1.943975f, 0.944795f},
            {0.971593f, -0.971593f, 0.000000f, 1.000000f, -0.971389f, 0.000000f}
        };

        const size_t N_alloc = _range_fft_size; // Stride

        for (int stage = 0; stage < 8; ++stage) {
            const float b0 = SOS[stage][0];
            const float b1 = SOS[stage][1];
            const float b2 = SOS[stage][2];
            const float a1 = SOS[stage][4]; 
            const float a2 = SOS[stage][5];

            std::complex<float>* s0_arr = &_state[stage * 2 * N_alloc]; 
            std::complex<float>* s1_arr = &_state[stage * 2 * N_alloc + N_alloc];

            for (size_t i = 0; i < num_symbols; ++i) {
                std::complex<float>* symbol_data = buffer.data() + i * N_alloc;

                #pragma omp simd
                for (size_t col = 0; col < N_proc; ++col) {
                    std::complex<float> x = symbol_data[col];
                    std::complex<float> s0 = s0_arr[col];
                    std::complex<float> s1 = s1_arr[col];

                    std::complex<float> y = x * b0 + s0;
                    std::complex<float> new_s0 = x * b1 - y * a1 + s1;
                    std::complex<float> new_s1 = x * b2 - y * a2;

                    symbol_data[col] = y;
                    s0_arr[col] = new_s0;
                    s1_arr[col] = new_s1;
                }
            }
        }
    }

private:
    AlignedVector _state;
    size_t _range_fft_size;
};

#endif // COMMON_HPP