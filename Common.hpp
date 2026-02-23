#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdint>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

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
#include <filesystem>
#include "LDPC5041008SIMD.hpp"

// ============== Common Utility Functions ==============

/**
 * @brief Check if a float is NaN using bit manipulation.
 * Compatible with -ffast-math compiler flag.
 */
inline bool isNaN(float x) {
    uint32_t bits;
    static_assert(sizeof(float) == sizeof(uint32_t), "Unexpected float size");
    std::memcpy(&bits, &x, sizeof(float));
    constexpr uint32_t exponent_mask = 0x7F800000;
    constexpr uint32_t mantissa_mask = 0x007FFFFF;
    return ((bits & exponent_mask) == exponent_mask) && (bits & mantissa_mask);
}

/**
 * @brief Convert FFT bin index to shifted index.
 */
inline int fftshift_index(int original_index, int N) {
    return (original_index + N/2) % N;
}


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
    size_t tx_frame_buffer_size = 8;   // TX frame ring buffer capacity
    size_t tx_symbols_buffer_size = 8; // TX symbols ring buffer capacity
    size_t rx_frame_buffer_size = 8;   // RX frame ring buffer capacity
    size_t cp_length = 128;            // Cyclic prefix length
    size_t num_symbols = 100;          // Number of symbols per frame
    size_t sensing_symbol_num = 100;   // Number of sensing symbols
    size_t sync_pos = 1;               // Synchronization symbol position
    int delay_adjust_step = 2;         // Delay adjustment step
    int desired_peak_pos = 20;         // Desired delay peak position to include non-causal components
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
    double ppm_adjust_factor = 0.05;

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
    SensingDataSender(const std::string& ip, int port) 
        : UdpBaseSender(ip, port)
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
 * @brief LDPC Codec Wrapper (custom LDPC5041008 encoder + AFF3CT decoder).
 * 
 * Provides a simplified interface for LDPC encoding/decoding.
 * Uses LDPC5041008SIMD for encoding and AFF3CT for decoding.
 */
class LDPCCodec {
public:
    using AlignedByteVector = mipp::vector<int8_t>;
    using AlignedIntVector = mipp::vector<int32_t>;
    using AlignedFloatVector = mipp::vector<float>;
    struct LDPCConfig {
        std::string h_matrix_path = "../LDPC_504_1008.alist";
        std::string g_matrix_path = "../LDPC_504_1008_G_fromH.alist";
        int decoder_iterations = 6;
        size_t n_frames = 16;
        std::string enc_type = "LDPC_H";
        std::string enc_g_method = "IDENTITY";
        std::string dec_type = "BP_HORIZONTAL_LAYERED";
        std::string dec_implem = "NMS";
        std::string dec_simd = "INTER";
        bool use_custom_encoder = true;
    };

    LDPCCodec(const LDPCConfig& config) : cfg(config) {
        namespace fs = std::filesystem;
        auto path_exists = [](const std::string& p) {
            std::error_code ec;
            return fs::exists(fs::path(p), ec);
        };

        // Resolve relative path against executable directory
        if (!cfg.h_matrix_path.empty() && cfg.h_matrix_path[0] != '/') {
            cfg.h_matrix_path = get_executable_dir() + "/" + cfg.h_matrix_path;
        }
        if (!cfg.g_matrix_path.empty() && cfg.g_matrix_path[0] != '/') {
            cfg.g_matrix_path = get_executable_dir() + "/" + cfg.g_matrix_path;
        }
        if (!cfg.g_matrix_path.empty() && !path_exists(cfg.g_matrix_path)) {
            const fs::path g_parent = fs::path(cfg.g_matrix_path).parent_path();
            const std::string alt_g_1 = (g_parent / "LDPC_504_1008G.alist").string();
            const std::string alt_g_2 = (g_parent / "PEGReg504x1008_Gen.alist").string();
            if (path_exists(alt_g_1)) {
                cfg.g_matrix_path = alt_g_1;
            } else if (path_exists(alt_g_2)) {
                cfg.g_matrix_path = alt_g_2;
            }
        }
        if (cfg.use_custom_encoder) {
            custom_encoder = std::make_unique<LDPC5041008SIMD>(cfg.h_matrix_path, cfg.g_matrix_path);
        }
        _init_aff3ct_params();
        codec = std::unique_ptr<aff3ct::tools::Codec_LDPC<int, float>>(codec_factory->build<int, float>());
        codec->set_n_frames(cfg.n_frames);
    }

    void encode_frame(const AlignedByteVector& input, AlignedIntVector& encoded_bits) {
        if (input.empty()) {
            encoded_bits.clear();
            return;
        }

        if (custom_encoder) {
            custom_encoder->encode_bytes(input, encoded_bits);
            return;
        }

        auto& encoder = codec->get_encoder();
        // Unpack input data
        AlignedIntVector unpacked_bits(input.size() * 8);
        _unpack_bits(input, unpacked_bits);
        const size_t N = encoder.get_N();
        const size_t K = encoder.get_K();
        // frame_id=-1 processes get_n_frames() frames per call, NOT n_frames_per_wave
        const size_t batch = encoder.get_n_frames();
        const size_t total_frames = unpacked_bits.size() / K;

        encoded_bits.resize(total_frames * N);

        size_t i = 0;
        for (; i + batch <= total_frames; i += batch) {
            encoder.encode(unpacked_bits.data() + i * K, encoded_bits.data() + i * N, -1, false);
        }

        const size_t remaining = total_frames - i;
        if (remaining > 0) {
            AlignedIntVector tmp_in(batch * K, 0);
            AlignedIntVector tmp_out(batch * N, 0);
            std::memcpy(tmp_in.data(), unpacked_bits.data() + i * K, remaining * K * sizeof(int32_t));
            encoder.encode(tmp_in.data(), tmp_out.data(), -1, false);
            std::memcpy(encoded_bits.data() + i * N, tmp_out.data(), remaining * N * sizeof(int32_t));
        }
    }

    // Decode entire frame
    void decode_frame(const AlignedFloatVector& llr_input, AlignedByteVector& decoded_bytes) {
        auto& decoder = codec->get_decoder_siho();
        const size_t N = decoder.get_N();
        const size_t K = decoder.get_K();
        // frame_id=-1 processes get_n_frames() frames per call, NOT n_frames_per_wave
        const size_t batch = decoder.get_n_frames();
        const size_t total_frames = llr_input.size() / N;

        AlignedIntVector decoded_bits(total_frames * K, 0);

        size_t i = 0;
        for (; i + batch <= total_frames; i += batch) {
            decoder.decode_siho(llr_input.data() + i * N, decoded_bits.data() + i * K, -1, false);
        }

        const size_t remaining = total_frames - i;
        if (remaining > 0) {
            AlignedFloatVector tmp_in(batch * N, 0.0f);
            AlignedIntVector tmp_out(batch * K, 0);
            std::memcpy(tmp_in.data(), llr_input.data() + i * N, remaining * N * sizeof(float));
            decoder.decode_siho(tmp_in.data(), tmp_out.data(), -1, false);
            std::memcpy(decoded_bits.data() + i * K, tmp_out.data(), remaining * K * sizeof(int32_t));
        }

        decoded_bytes.resize(decoded_bits.size() / 8);
        _pack_bits(decoded_bits, decoded_bytes);
    }
    
    size_t get_K() const {
        if (custom_encoder) return static_cast<size_t>(LDPC5041008SIMD::K);
        return codec->get_encoder().get_K();
    }
    size_t get_N() const {
        if (custom_encoder) return static_cast<size_t>(LDPC5041008SIMD::N);
        return codec->get_encoder().get_N();
    }

    /**
     * @brief Pack bits into QPSK symbols (0-3).
     * Each pair of bits corresponds to one QPSK symbol.
     */
    static void pack_bits_qpsk(const AlignedIntVector &bits, AlignedIntVector &qpsk_ints) {
        const size_t bit_count = bits.size();
        const size_t symbol_count = (bit_count + 1) / 2;
        qpsk_ints.resize(symbol_count);
        const size_t even_pairs = bit_count / 2;
        #pragma omp simd
        for (size_t k = 0; k < even_pairs; ++k) {
            int b0 = bits[2*k] & 1;
            int b1 = bits[2*k + 1] & 1;
            qpsk_ints[k] = (b0 << 1) | b1;
        }
        if (bit_count & 1) {
            qpsk_ints[even_pairs] = (bits[bit_count - 1] & 1) << 1;
        }
    }

private:
    LDPCConfig cfg;
    std::unique_ptr<aff3ct::factory::Codec_LDPC> codec_factory;
    std::unique_ptr<aff3ct::tools::Codec_LDPC<int, float>> codec;
    std::unique_ptr<LDPC5041008SIMD> custom_encoder;
    void _init_aff3ct_params() {
        std::vector<std::string> args = {
            "LDPCEncoder",
            "--enc-type", cfg.enc_type,
            "--enc-g-method", cfg.enc_g_method,
            "--dec-type", cfg.dec_type,
            "--dec-implem", cfg.dec_implem,
            "--dec-ite", std::to_string(cfg.decoder_iterations),
            "--dec-synd-depth", "1",
            "--dec-h-path", cfg.h_matrix_path,
        };
        if (!cfg.dec_simd.empty()) {
            args.push_back("--dec-simd");
            args.push_back(cfg.dec_simd);
        }
        // Only pass G-matrix to AFF3CT when it handles encoding itself.
        // When use_custom_encoder is true, LDPC5041008SIMD owns encoding and
        // the G-matrix format may be incompatible with AFF3CT, causing
        // internal corruption (double free).
        if (!cfg.use_custom_encoder && !cfg.g_matrix_path.empty()) {
            args.push_back("--enc-g-path");
            args.push_back(cfg.g_matrix_path);
        }

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

#endif // COMMON_HPP
