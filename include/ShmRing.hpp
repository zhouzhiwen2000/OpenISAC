#ifndef SHM_RING_HPP
#define SHM_RING_HPP

// POSIX shared-memory transport for the channel simulator.
//
// Provides:
//   - ShmRing      : a cross-process single-producer/single-consumer ring buffer
//                    of std::complex<float> samples (used for one TX or RX stream).
//   - SimControlBlock / ShmControl : a small shared block holding the simulated
//                    sample clock and a global running flag.
//
// All segments are named "/<session>.<suffix>" so the ChannelSimulator hub and the
// BS/UE clients attach to the same simulated "air". The hub creates
// the segments; clients open them (polling until the creator has published the magic).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sim_shm {

using sample_t = std::complex<float>;

inline std::string make_shm_name(const std::string& session, const std::string& suffix) {
    std::string sanitized = session + "." + suffix;
    for (char& c : sanitized) {
        if (c == '/') c = '_';
    }
    return "/" + sanitized;
}

// ---------------------------------------------------------------------------
// Ring buffer
// ---------------------------------------------------------------------------

struct ShmRingHeader {
    std::atomic<uint64_t> magic;       // 0 until fully initialized by the creator
    uint64_t capacity;                 // number of sample_t slots
    char pad0[48];                     // keep head/tail on separate cache lines
    std::atomic<uint64_t> write_pos;   // total samples ever written (producer owns)
    char pad1[56];
    std::atomic<uint64_t> read_pos;    // total samples ever read (consumer owns)
    char pad2[56];
};

static constexpr uint64_t kShmRingMagic = 0x4F49534143524E47ull; // "OISACRNG"

class ShmRing {
public:
    ShmRing() = default;
    ~ShmRing() { close(); }

    ShmRing(const ShmRing&) = delete;
    ShmRing& operator=(const ShmRing&) = delete;

    // Create (or re-create) a ring. Truncates and re-initializes the segment.
    // Any pre-existing segment with the same name is unlinked first so stale segments
    // left by a previous run (even one owned by root) do not cause ftruncate to fail.
    void create(const std::string& name, uint64_t capacity) {
        _name = name;
        ::shm_unlink(name.c_str()); // ignore error — segment may not exist
        const size_t bytes = sizeof(ShmRingHeader) + capacity * sizeof(sample_t);
        int fd = ::shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0) throw std::runtime_error("shm_open(create) failed for " + name);
        if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
            ::close(fd);
            throw std::runtime_error("ftruncate failed for " + name);
        }
        _map(fd, bytes);
        ::close(fd);
        _header->capacity = capacity;
        _header->write_pos.store(0, std::memory_order_relaxed);
        _header->read_pos.store(0, std::memory_order_relaxed);
        _capacity = capacity;
        _data = reinterpret_cast<sample_t*>(reinterpret_cast<char*>(_addr) + sizeof(ShmRingHeader));
        // Publish last so openers see a consistent header.
        _header->magic.store(kShmRingMagic, std::memory_order_release);
        _creator = true;
    }

    // Open an existing ring created by the hub. Blocks until the segment exists and
    // its magic has been published, or until `running` is cleared / timeout elapses.
    bool open(const std::string& name, std::atomic<int>* running = nullptr,
              double timeout_s = 30.0) {
        _name = name;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                  std::chrono::duration<double>(timeout_s));
        int fd = -1;
        while (true) {
            fd = ::shm_open(name.c_str(), O_RDWR, 0666);
            if (fd >= 0) break;
            if (running && running->load(std::memory_order_acquire) == 0) return false;
            if (std::chrono::steady_clock::now() > deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        struct stat st {};
        if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(ShmRingHeader))) {
            ::close(fd);
            return false;
        }
        _map(fd, static_cast<size_t>(st.st_size));
        ::close(fd);
        // Wait for the creator to finish initialization.
        while (_header->magic.load(std::memory_order_acquire) != kShmRingMagic) {
            if (running && running->load(std::memory_order_acquire) == 0) { close(); return false; }
            if (std::chrono::steady_clock::now() > deadline) { close(); return false; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        _capacity = _header->capacity;
        _data = reinterpret_cast<sample_t*>(reinterpret_cast<char*>(_addr) + sizeof(ShmRingHeader));
        return true;
    }

    bool valid() const { return _addr != nullptr; }
    uint64_t capacity() const { return _capacity; }

    uint64_t available() const {
        return _header->write_pos.load(std::memory_order_acquire) -
               _header->read_pos.load(std::memory_order_acquire);
    }

    // Total samples consumed so far (the per-stream sample clock for RX timestamps).
    uint64_t consumed() const { return _header->read_pos.load(std::memory_order_acquire); }
    // Total samples produced so far.
    uint64_t produced() const { return _header->write_pos.load(std::memory_order_acquire); }

    // Producer: write `count` samples, blocking until space is available.
    // Returns the number written (== count unless `running` was cleared or `timeout_s`
    // elapsed). A negative `timeout_s` blocks indefinitely.
    size_t push_block(const sample_t* data, size_t count, std::atomic<int>* running,
                      double timeout_s = -1.0) {
        size_t written = 0;
        Backoff backoff;
        const bool has_deadline = timeout_s >= 0.0;
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(has_deadline ? timeout_s : 0.0));
        while (written < count) {
            const uint64_t w = _header->write_pos.load(std::memory_order_relaxed);
            const uint64_t r = _header->read_pos.load(std::memory_order_acquire);
            uint64_t free_slots = _capacity - (w - r);
            if (free_slots == 0) {
                if (running && running->load(std::memory_order_acquire) == 0) break;
                if (has_deadline && std::chrono::steady_clock::now() >= deadline) break;
                backoff.pause();
                continue;
            }
            backoff.reset();
            size_t chunk = static_cast<size_t>(free_slots);
            if (chunk > count - written) chunk = count - written;
            const size_t idx = static_cast<size_t>(w % _capacity);
            const size_t first = std::min<size_t>(chunk, _capacity - idx);
            std::memcpy(_data + idx, data + written, first * sizeof(sample_t));
            if (chunk > first) {
                std::memcpy(_data, data + written + first, (chunk - first) * sizeof(sample_t));
            }
            _header->write_pos.store(w + chunk, std::memory_order_release);
            written += chunk;
        }
        return written;
    }

    // Consumer: read `count` samples, blocking until available.
    // Returns the number read (== count unless `running` was cleared or `timeout_s`
    // elapsed). A negative `timeout_s` blocks indefinitely.
    size_t pop_block(sample_t* out, size_t count, std::atomic<int>* running,
                     double timeout_s = -1.0) {
        size_t read = 0;
        Backoff backoff;
        const bool has_deadline = timeout_s >= 0.0;
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(has_deadline ? timeout_s : 0.0));
        while (read < count) {
            const uint64_t r = _header->read_pos.load(std::memory_order_relaxed);
            const uint64_t w = _header->write_pos.load(std::memory_order_acquire);
            uint64_t avail = w - r;
            if (avail == 0) {
                if (running && running->load(std::memory_order_acquire) == 0) break;
                if (has_deadline && std::chrono::steady_clock::now() >= deadline) break;
                backoff.pause();
                continue;
            }
            backoff.reset();
            size_t chunk = static_cast<size_t>(avail);
            if (chunk > count - read) chunk = count - read;
            const size_t idx = static_cast<size_t>(r % _capacity);
            const size_t first = std::min<size_t>(chunk, _capacity - idx);
            std::memcpy(out + read, _data + idx, first * sizeof(sample_t));
            if (chunk > first) {
                std::memcpy(out + read + first, _data, (chunk - first) * sizeof(sample_t));
            }
            _header->read_pos.store(r + chunk, std::memory_order_release);
            read += chunk;
        }
        return read;
    }

    // Consumer: block until at least one sample is available, then read up to
    // `max` samples. Returns the number read (0 only if `running` was cleared).
    size_t pop_upto(sample_t* out, size_t max, std::atomic<int>* running) {
        Backoff backoff;
        while (true) {
            const uint64_t r = _header->read_pos.load(std::memory_order_relaxed);
            const uint64_t w = _header->write_pos.load(std::memory_order_acquire);
            const uint64_t avail = w - r;
            if (avail == 0) {
                if (running && running->load(std::memory_order_acquire) == 0) return 0;
                backoff.pause();
                continue;
            }
            size_t chunk = static_cast<size_t>(avail);
            if (chunk > max) chunk = max;
            const size_t idx = static_cast<size_t>(r % _capacity);
            const size_t first = std::min<size_t>(chunk, _capacity - idx);
            std::memcpy(out, _data + idx, first * sizeof(sample_t));
            if (chunk > first) {
                std::memcpy(out + first, _data, (chunk - first) * sizeof(sample_t));
            }
            _header->read_pos.store(r + chunk, std::memory_order_release);
            return chunk;
        }
    }

    void unlink() {
        if (!_name.empty()) ::shm_unlink(_name.c_str());
    }

    void close() {
        if (_addr) {
            ::munmap(_addr, _bytes);
            _addr = nullptr;
            _header = nullptr;
            _data = nullptr;
        }
    }

private:
    struct Backoff {
        int spins = 0;
        void reset() { spins = 0; }
        void pause() {
            if (spins < 64) {
                ++spins;
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
    };

    void _map(int fd, size_t bytes) {
        void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed for " + _name);
        _addr = p;
        _bytes = bytes;
        _header = reinterpret_cast<ShmRingHeader*>(p);
    }

    std::string _name;
    void* _addr = nullptr;
    size_t _bytes = 0;
    ShmRingHeader* _header = nullptr;
    sample_t* _data = nullptr;
    uint64_t _capacity = 0;
    bool _creator = false;
};

// ---------------------------------------------------------------------------
// Control block (shared sample clock + running flag)
// ---------------------------------------------------------------------------

struct SimControlBlock {
    std::atomic<uint64_t> magic;
    std::atomic<int> running;                 // 1 = simulation active, 0 = shutting down
    std::atomic<uint64_t> global_sample_index; // hub's sample clock (RX timeline)
    double tick_rate;                          // sample rate (Hz) for time conversions
    uint32_t num_sensing_channels;
    std::atomic<int64_t> comm_rx_freq_correction_millihz; // Receiver-side comm RX tune correction
    char pad[16];
};

static constexpr uint64_t kShmCtrlMagic = 0x4F49534143435452ull; // "OISACCTR"

class ShmControl {
public:
    ShmControl() = default;
    ~ShmControl() { close(); }

    ShmControl(const ShmControl&) = delete;
    ShmControl& operator=(const ShmControl&) = delete;

    void create(const std::string& name, double tick_rate, uint32_t num_sensing_channels) {
        _name = name;
        ::shm_unlink(name.c_str()); // remove stale segment from a previous run if present
        int fd = ::shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0) throw std::runtime_error("shm_open(create) failed for " + name);
        if (::ftruncate(fd, sizeof(SimControlBlock)) != 0) {
            ::close(fd);
            throw std::runtime_error("ftruncate failed for " + name);
        }
        _map(fd);
        ::close(fd);
        _blk->running.store(1, std::memory_order_relaxed);
        _blk->global_sample_index.store(0, std::memory_order_relaxed);
        _blk->tick_rate = tick_rate;
        _blk->num_sensing_channels = num_sensing_channels;
        _blk->comm_rx_freq_correction_millihz.store(0, std::memory_order_relaxed);
        _blk->magic.store(kShmCtrlMagic, std::memory_order_release);
    }

    bool open(const std::string& name, double timeout_s = 30.0) {
        _name = name;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                  std::chrono::duration<double>(timeout_s));
        int fd = -1;
        while ((fd = ::shm_open(name.c_str(), O_RDWR, 0666)) < 0) {
            if (std::chrono::steady_clock::now() > deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        _map(fd);
        ::close(fd);
        while (_blk->magic.load(std::memory_order_acquire) != kShmCtrlMagic) {
            if (std::chrono::steady_clock::now() > deadline) { close(); return false; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return true;
    }

    bool valid() const { return _blk != nullptr; }
    SimControlBlock* block() { return _blk; }

    int running() const { return _blk ? _blk->running.load(std::memory_order_acquire) : 0; }
    void stop() { if (_blk) _blk->running.store(0, std::memory_order_release); }
    uint64_t sample_index() const {
        return _blk ? _blk->global_sample_index.load(std::memory_order_acquire) : 0;
    }
    void advance(uint64_t n) {
        if (_blk) _blk->global_sample_index.fetch_add(n, std::memory_order_release);
    }
    double tick_rate() const { return _blk ? _blk->tick_rate : 0.0; }
    void set_comm_rx_freq_correction_hz(double value_hz) {
        if (!_blk || !std::isfinite(value_hz)) return;
        const double scaled = value_hz * 1000.0;
        const double lo = static_cast<double>(std::numeric_limits<int64_t>::min());
        const double hi = static_cast<double>(std::numeric_limits<int64_t>::max());
        const auto quantized = static_cast<int64_t>(std::llround(std::clamp(scaled, lo, hi)));
        _blk->comm_rx_freq_correction_millihz.store(quantized, std::memory_order_release);
    }
    double comm_rx_freq_correction_hz() const {
        if (!_blk) return 0.0;
        return static_cast<double>(
            _blk->comm_rx_freq_correction_millihz.load(std::memory_order_acquire)) / 1000.0;
    }

    void unlink() { if (!_name.empty()) ::shm_unlink(_name.c_str()); }

    void close() {
        if (_blk) {
            ::munmap(_blk, sizeof(SimControlBlock));
            _blk = nullptr;
        }
    }

private:
    void _map(int fd) {
        void* p = ::mmap(nullptr, sizeof(SimControlBlock), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed for " + _name);
        _blk = reinterpret_cast<SimControlBlock*>(p);
    }

    std::string _name;
    SimControlBlock* _blk = nullptr;
};

} // namespace sim_shm

#endif // SHM_RING_HPP
