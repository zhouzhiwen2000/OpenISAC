#ifndef UDP_LINK_TEST_HPP
#define UDP_LINK_TEST_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace udp_link_test {

constexpr size_t kHeaderBytes = 32;
constexpr size_t kCopies = 3;
constexpr size_t kPrefixBytes = kHeaderBytes * kCopies;
constexpr uint64_t kMagic = 0x3130545354435349ULL;

inline void put_le(uint8_t* out, uint64_t value, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) out[i] = uint8_t(value >> (8 * i));
}
inline uint64_t get_le(const uint8_t* in, size_t bytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < bytes; ++i) value |= uint64_t(in[i]) << (8 * i);
    return value;
}
inline uint32_t crc32(const uint8_t* data, size_t size) {
    static const auto table = [] {
        std::array<uint32_t, 256> values{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) c = (c >> 1) ^ (0xedb88320U & (0U - (c & 1U)));
            values[i] = c;
        }
        return values;
    }();
    uint32_t c = ~0U;
    for (size_t i = 0; i < size; ++i) c = table[(c ^ data[i]) & 255] ^ (c >> 8);
    return ~c;
}
inline uint64_t pattern_word(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Header {
    uint64_t session = 0;
    uint64_t sequence = 0;
    uint32_t packet_bytes = 0;
};

/** @brief Fill an entire known UDP datagram, including three protected headers. */
inline void make_packet(uint8_t* out, size_t size, uint64_t session, uint64_t sequence) {
    if (size < kPrefixBytes + 8 || size > 65507) throw std::invalid_argument("Invalid packet size");
    put_le(out, kMagic, 8);
    put_le(out + 8, session, 8);
    put_le(out + 16, sequence, 8);
    put_le(out + 24, size, 4);
    put_le(out + 28, crc32(out, 28), 4);
    for (size_t i = 1; i < kCopies; ++i) std::memcpy(out + i * kHeaderBytes, out, kHeaderBytes);
    uint64_t state = session ^ (sequence * 0xd6e8feb86659fd93ULL);
    for (size_t i = kPrefixBytes; i < size; i += 8) {
        put_le(out + i, pattern_word(state), std::min<size_t>(8, size - i));
    }
}

/** @brief Recover metadata from a valid header copy or bitwise majority repair. */
inline bool read_header(const uint8_t* data, size_t size, Header& header) {
    if (size < kPrefixBytes + 8 || size > 65507) return false;
    std::array<uint8_t, kHeaderBytes> repaired{};
    for (size_t i = 0; i < kHeaderBytes; ++i) {
        const uint8_t a = data[i], b = data[i + kHeaderBytes], c = data[i + 2 * kHeaderBytes];
        repaired[i] = (a & b) | (a & c) | (b & c);
    }
    for (size_t copy = 0; copy <= kCopies; ++copy) {
        const uint8_t* p = copy == kCopies ? repaired.data() : data + copy * kHeaderBytes;
        if (get_le(p, 8) != kMagic || get_le(p + 28, 4) != crc32(p, 28) ||
            get_le(p + 24, 4) != size || get_le(p + 16, 8) == UINT64_MAX) continue;
        header = {get_le(p + 8, 8), get_le(p + 16, 8), uint32_t(size)};
        return true;
    }
    return false;
}

/** @brief Count differing bits without rejecting payloads that contain errors. */
inline uint64_t bit_errors(const uint8_t* received, const uint8_t* expected, size_t size) {
    uint64_t errors = 0;
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        uint64_t a, b;
        std::memcpy(&a, received + i, 8);
        std::memcpy(&b, expected + i, 8);
        errors += __builtin_popcountll(a ^ b);
    }
    for (; i < size; ++i) errors += __builtin_popcount(unsigned(received[i] ^ expected[i]));
    return errors;
}

/**
 * @brief Bounded sequence accounting with duplicate, reorder and late recovery.
 * Confirm sender counts only after a reorder grace interval. Very old packets
 * outside the retained window are reported separately instead of double-counted.
 */
class SequenceStats {
public:
    explicit SequenceStats(size_t capacity = 262144)
        : _seen(capacity, UINT64_MAX) {
        if (!capacity) throw std::invalid_argument("Sequence window must be nonempty");
    }
    /** @brief Record one identifiable packet; return true only for a new sequence. */
    bool record(uint64_t sequence) {
        if (!_started) {
            _started = true;
            _loss_begin = confirmed_sent = sequence;
        }
        const uint64_t horizon = std::max(highest, confirmed_sent);
        if (horizon > _seen.size() && sequence < horizon - _seen.size()) {
            ++too_old;
            return false;
        }
        const uint64_t previous_highest = highest;
        highest = std::max(highest, sequence + 1);
        if (highest > _seen.size() && highest - _seen.size() > confirmed_sent)
            confirm(highest - _seen.size());
        uint64_t& slot = _seen[sequence % _seen.size()];
        if (slot == sequence) { ++duplicates; return false; }
        slot = sequence;
        ++unique;
        if (sequence + 1 < previous_highest) ++reordered;
        if (sequence < confirmed_sent) {
            ++late;
            if (sequence >= _loss_begin) ++confirmed_received;
        }
        return true;
    }
    /** @brief Confirm an aged sender count, including loss during a full blackout. */
    void confirm(uint64_t sent) {
        if (!_started || sent <= confirmed_sent) return;
        if (sent - confirmed_sent <= _seen.size()) {
            for (uint64_t i = confirmed_sent; i < sent; ++i)
                confirmed_received += (_seen[i % _seen.size()] == i);
        } else {
            for (uint64_t i : _seen)
                confirmed_received += (i != UINT64_MAX && i >= confirmed_sent && i < sent);
        }
        confirmed_sent = sent;
    }
    /** @brief Clear loss accounting while retaining duplicate and receive history. */
    void reset_loss(uint64_t sender_count = 0) {
        if (!_started) return;
        _loss_begin = confirmed_sent = std::max({highest, confirmed_sent, sender_count});
        confirmed_received = 0;
    }
    /** @brief Whether a packet from this session has started loss accounting. */
    bool started() const { return _started; }
    /** @brief Confirmed packet count since the first RX packet or latest reset. */
    uint64_t expected() const { return confirmed_sent - _loss_begin; }
    /** @brief Missing packets in the current loss interval. */
    uint64_t lost() const { return expected() - confirmed_received; }
    uint64_t highest = 0, unique = 0, duplicates = 0, reordered = 0, late = 0, too_old = 0;
    uint64_t confirmed_sent = 0, confirmed_received = 0;
private:
    std::vector<uint64_t> _seen;
    uint64_t _loss_begin = 0;
    bool _started = false;
};

} // namespace udp_link_test
#endif // UDP_LINK_TEST_HPP
