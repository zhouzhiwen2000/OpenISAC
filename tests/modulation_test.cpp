#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "LDPCCodec.hpp"
#include "OFDMCore.hpp"

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
constexpr const char* names[] = {"BPSK", "QPSK", "8PSK", "16QAM", "32QAM", "64QAM"};

void check_constellation(const DigitalModulation& modem) {
    double energy = 0;
    for (unsigned label = 0; label < modem.order(); ++label) {
        const auto point = modem.modulate(label);
        energy += std::norm(point);
        require(std::abs(modem.remodulate(point) - point) < 1e-6f, "hard decision is not a constellation point");
        float llrs[6];
        modem.soft_llrs(point, 7.0f, llrs);
        for (unsigned bit = 0; bit < modem.bits_per_symbol(); ++bit)
            require((llrs[bit] < 0) == bool((label >> (modem.bits_per_symbol() - bit - 1)) & 1), "LLR bit order/sign mismatch");
    }
    require(std::abs(energy / modem.order() - 1.0) < 1e-6, "average symbol energy is not one");
    if (modem.is_qpsk()) {
        LDPCCodec::AlignedIntVector bits(1008), legacy, current;
        for (size_t index = 0; index < bits.size(); ++index) bits[index] = (index * 7 + index / 3) & 1;
        LDPCCodec::pack_bits_qpsk(bits, legacy);
        modem.pack_bits(bits, current, 1008);
        require(current == legacy, "default QPSK bit packing changed");
        for (int label = 0; label < 4; ++label)
            require(modem.modulate_token(label) == QPSKModulator{}.modulate(label),
                    "legacy QPSK constellation changed");
    }
    // Independent full-constellation distance reference validates optimized PAM
    // demapping away from constellation centers and across all decision edges.
    std::mt19937 generator(491);
    std::uniform_real_distribution<float> uniform(-2, 2);
    for (unsigned trial = 0; trial < 1000; ++trial) {
        const std::complex<float> sample{uniform(generator), uniform(generator)};
        float llrs[6];
        modem.soft_llrs(sample, 3.25f, llrs);
        for (unsigned bit = 0; bit < modem.bits_per_symbol(); ++bit) {
            float distance[2] = {1e9f, 1e9f};
            for (unsigned label = 0; label < modem.order(); ++label) {
                const unsigned value = (label >> (modem.bits_per_symbol() - bit - 1)) & 1;
                distance[value] = std::min(distance[value], std::norm(sample - modem.modulate(label)));
            }
            const float reference = (distance[1] - distance[0]) * 3.25f;
            require(std::abs(llrs[bit] - reference) < 3e-5f * std::max(1.0f, std::abs(reference)), "max-log LLR differs from exhaustive distances");
        }
    }
    // Every nearest grid/ring neighbour must differ by exactly one label bit.
    float minimum = 1e9f;
    for (unsigned i = 0; i < modem.order(); ++i)
        for (unsigned j = i + 1; j < modem.order(); ++j)
            minimum = std::min(minimum, std::norm(modem.modulate(i) - modem.modulate(j)));
    for (unsigned i = 0; i < modem.order(); ++i)
        for (unsigned j = i + 1; j < modem.order(); ++j)
            if (std::abs(std::norm(modem.modulate(i) - modem.modulate(j)) - minimum) < 1e-6f)
                require(__builtin_popcount(i ^ j) == 1, "nearest-neighbour labels are not Gray coded");
}

// Independent exhaustive-distance oracle, evaluated in double precision so
// Q16 rounding and saturation can be checked even far outside the constellation.
double reference_llr(const DigitalModulation& modem, double x, double y,
                     unsigned bit, double scale) {
    double distance[2] = {1e100, 1e100};
    for (unsigned label = 0; label < modem.order(); ++label) {
        const auto p = modem.modulate(label);
        const double dx = x / scale - p.real(), dy = y / scale - p.imag();
        const unsigned value = (label >> (modem.bits_per_symbol() - bit - 1)) & 1;
        distance[value] = std::min(distance[value], dx * dx + dy * dy);
    }
    return (distance[1] - distance[0]) * scale / (4 * DigitalModulation::kInvSqrt2);
}

void check_hard_reference(const DigitalModulation& modem) {
    std::mt19937 random(177);
    std::uniform_real_distribution<float> uniform(-4, 4);
    std::vector<std::complex<float>> samples(4099);
    for (auto& value : samples) value = {uniform(random), uniform(random)};
    samples.insert(samples.end(), {{0, 0}, {-0.0f, -0.0f}, {1000, -1000}});
    // Midpoints and their adjacent floats exercise decision ties independently
    // of the threshold formulas used by the implementation.
    for (unsigned a = 0; a < modem.order(); ++a) {
        for (unsigned b = a + 1; b < modem.order(); ++b) {
            const auto mid = (modem.modulate(a) + modem.modulate(b)) * 0.5f;
            samples.push_back(mid);
            samples.emplace_back(std::nextafter(mid.real(), -10.0f), mid.imag());
            samples.emplace_back(std::nextafter(mid.real(), 10.0f), mid.imag());
        }
    }
    std::vector<std::complex<float>> references(samples.size());
    modem.remodulate_samples(samples.data(), references.data(), samples.size());
    std::vector<double> payload_error(samples.size()), control_error(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        double nearest = 1e100;
        double on_constellation = 1e100;
        for (unsigned label = 0; label < modem.order(); ++label) {
            const auto point = modem.modulate(label);
            nearest = std::min(nearest, double(std::norm(samples[i] - point)));
            on_constellation = std::min(on_constellation, double(std::norm(references[i] - point)));
            require(std::abs(modem.modulate_token((modem.is_qpsk() ? 0 : 4) + label) - point) < 1e-6f,
                    "TX token table changed the payload mapping");
        }
        require(on_constellation < 1e-10, "batch slicer returned a point outside the constellation");
        const double tolerance = 2e-6 * std::max(1.0, nearest);
        require(std::abs(double(std::norm(samples[i] - references[i])) - nearest) <= tolerance,
                "batch hard decision differs from exhaustive distance");
        require(std::abs(double(std::norm(samples[i] - modem.remodulate(samples[i]))) - nearest) <= tolerance,
                "scalar hard decision differs from exhaustive distance");
        double control = 1e100;
        for (int label = 0; label < 4; ++label) {
            const auto point = QPSKModulator{}.modulate(label);
            control = std::min(control, double(std::norm(samples[i] - point)));
            require(modem.modulate_token(label) == point, "TX token table changed QPSK control");
        }
        const auto mixed = DigitalModulation::remodulate_mixed(samples[i], references[i]);
        const double union_error = std::min(nearest, control);
        require(std::abs(double(std::norm(samples[i] - mixed)) - union_error) <=
                    2e-6 * std::max(1.0, union_error), "unknown-region reference is not nearest");
        payload_error[i] = nearest;
        control_error[i] = control;
    }
    std::vector<int> indices(samples.size());
    std::vector<uint8_t> mask(samples.size());
    double expected = 0.0;
    for (size_t i = 0; i < samples.size(); ++i) {
        const size_t k = (i * 31) % samples.size();
        indices[i] = static_cast<int>(k);
        mask[i] = static_cast<uint8_t>(i % 3);
        expected += modem.is_qpsk() || mask[i] == 0 ? control_error[k] :
            (mask[i] == 1 ? payload_error[k] : std::min(control_error[k], payload_error[k]));
    }
    // QPSK does not allocate a modulation mask in the receiver.
    const double actual = modem.decision_error_sum(samples.data(), indices.data(),
        modem.is_qpsk() ? nullptr : mask.data(), indices.size());
    require(std::abs(actual - expected) <= 2e-6 * std::max(1.0, expected),
            "indexed EVM differs from exhaustive constellation distances");
    require(modem.decision_error_sum(samples.data(), indices.data(), nullptr, 0) == 0.0,
            "empty EVM range is not zero");
}

void check_boundaries(const DigitalModulation& modem) {
    std::mt19937 random(71);
    std::uniform_real_distribution<float> sample(-2, 2);
    const unsigned k = modem.bits_per_symbol();
    for (size_t block : {size_t(0), size_t(7), size_t(1008)}) {
        for (size_t count : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(4), size_t(5), size_t(6),
                             size_t(7), size_t(31), size_t(39), size_t(40), size_t(41),
                             size_t(47), size_t(48), size_t(49), size_t(79), size_t(80), size_t(81),
                             size_t(95), size_t(96), size_t(97), size_t(191), size_t(192), size_t(193),
                             size_t(319), size_t(320), size_t(321), size_t(639), size_t(640), size_t(641),
                             size_t(1007), size_t(1008), size_t(1009), size_t(21168)}) {
            const size_t block_size = block == 0 ? std::max<size_t>(count, 1) : block;
            std::vector<int> bits(count), tokens, expected_tokens;
            for (auto& bit : bits) bit = random() & 1;
            for (size_t at = 0; at < count; at += block_size) {
                const size_t end = std::min(at + block_size, count);
                for (size_t symbol = at; symbol < end; symbol += k) {
                    unsigned label = 0;
                    for (unsigned bit = 0; bit < k; ++bit)
                        label = (label << 1) | (symbol + bit < end ? bits[symbol + bit] : 0);
                    expected_tokens.push_back((modem.is_qpsk() ? 0 : 4) + label);
                }
            }
            modem.pack_bits(bits, tokens, block);
            require(tokens == expected_tokens, "specialized packing changed padding or bit order");
            // QPSK keeps its historical contiguous LLR copy, independent of block.
            const size_t transport_symbols = modem.is_qpsk() ? modem.symbol_count(count) : tokens.size();
            for (float scale : {0.3f, 17.0f, 4096.0f, 8192.0f}) {
                std::vector<float> fp_pairs(2 * transport_symbols), fp(count + 2, 12345);
                std::vector<int16_t> q_pairs(2 * transport_symbols), q(count + 2, 12345);
                for (size_t j = 0; j < fp_pairs.size(); ++j) {
                    fp_pairs[j] = sample(random) * scale;
                    q_pairs[j] = j % 13 == 0 ? static_cast<int16_t>(random() & 65535) :
                        static_cast<int16_t>(std::round(fp_pairs[j]));
                }
                // Offset outputs deliberately to exercise unaligned SIMD stores.
                modem.demap_scaled_pairs(fp_pairs.data(), count, scale, fp.data() + 1, block);
                modem.demap_scaled_pairs(q_pairs.data(), count, scale, q.data() + 1, block);
                require(fp.front() == 12345 && fp.back() == 12345 && q.front() == 12345 && q.back() == 12345,
                        "soft demapper wrote beyond the payload");
                for (size_t j = 0; j < count; ++j) {
                    if (modem.is_qpsk()) {
                        require(fp[j + 1] == fp_pairs[j] && q[j + 1] == q_pairs[j], "QPSK LLR copy changed");
                        continue;
                    }
                    const size_t symbol = (j / block_size) * modem.symbol_count(block_size) + (j % block_size) / k;
                    const unsigned bit = (j % block_size) % k;
                    const double rf = reference_llr(modem, fp_pairs[2 * symbol], fp_pairs[2 * symbol + 1], bit, scale);
                    const double rq = reference_llr(modem, q_pairs[2 * symbol], q_pairs[2 * symbol + 1], bit, scale);
                    require(std::abs(fp[j + 1] - rf) <= 5e-6 * (scale + std::abs(rf)), "SIMD float LLR mismatch");
                    const double quantized = std::round(std::clamp(rq, -32767.0, 32767.0));
                    // Reassociated float operations may cross a half-integer by
                    // a few ulps. The allowed error is one Q16 least-significant bit.
                    require(std::abs(q[j + 1] - quantized) <= 1, "SIMD Q16 LLR mismatch");
                }
            }
        }
    }
}

void check_packet(const DigitalModulation& modem, LDPCCodec& codec, bool fixed) {
    // Include block and constellation boundaries, a normal UDP size, and the
    // 32QAM case where 1008 coded bits do not divide by five.
    for (size_t length : {size_t(1), size_t(63), size_t(64), size_t(1316), size_t(12601), size_t(65507)}) {
        const size_t blocks = LdpcPacketFraming::payload_blocks_for_len(length);
        LDPCCodec::AlignedByteVector bytes(blocks * 63), decoded;
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<int8_t>((i * 37 + length) & 255);
        LDPCCodec::AlignedIntVector encoded, scratch, payload, packet;
        codec.encode_frame(bytes, encoded);
        Scrambler scrambler(LdpcPacketFraming::scrambler_bits(modem), 0x5A);
        BitBlockInterleaver interleaver(1008, 21);
        scrambler.scramble(encoded);
        interleaver.interleave_inplace(encoded, scratch);
        modem.pack_bits(encoded, payload, 1008);
        require(payload.size() == blocks * modem.symbol_count(1008), "payload padding length mismatch");
        const LdpcMiniHeader header{LdpcPacketFraming::kVersion, 0, static_cast<uint16_t>(length),
            LdpcPacketFraming::payload_blocks_field_for_len(length), 97};
        packet.resize(LdpcPacketFraming::kIdentityControlSymbols + LdpcPacketFraming::kControlSymbols);
        LdpcPacketFraming::write_identity_control_qpsk(0x12345678, 0x87654321, packet.data());
        LdpcPacketFraming::write_control_qpsk(header, packet.data() + LdpcPacketFraming::kIdentityControlSymbols);
        packet.insert(packet.end(), payload.begin(), payload.end());
        const size_t payload_begin = LdpcPacketFraming::kIdentityControlSymbols + LdpcPacketFraming::kControlSymbols;
        require(packet.size() - LdpcPacketFraming::kIdentityControlSymbols ==
                LdpcPacketFraming::packet_symbols(blocks, modem), "packet RE budget mismatch");
        std::vector<std::complex<float>> iq(packet.size());
        std::mt19937 random(152 + static_cast<unsigned>(length));
        std::normal_distribution<float> noise(0, .006f);
        for (size_t i = 0; i < packet.size(); ++i)
            iq[i] = modem.modulate_token(packet[i]) + std::complex<float>(noise(random), noise(random));
        std::vector<uint8_t> mask(packet.size());
        classify_payload_modulation(iq.size(), [&](size_t i) { return iq[i]; }, modem, mask.data());
        for (size_t i = 0; i < mask.size(); ++i)
            require(mask[i] == (i < payload_begin ? 0 : 1), "sensing selected the wrong control/payload constellation");
        // Check the actual two-value transport used by both CPU RX pipelines.
        const float scale = fixed ? 4096.0f : 128.0f;
        if (fixed) {
            LDPCCodec::AlignedShortVector pairs(iq.size() * 2), llrs(encoded.size()), deint;
            for (size_t i = 0; i < iq.size(); ++i) {
                pairs[2 * i] = static_cast<int16_t>(std::round(iq[i].real() * scale));
                pairs[2 * i + 1] = static_cast<int16_t>(std::round(iq[i].imag() * scale));
            }
            modem.demap_scaled_pairs(pairs.data() + payload_begin * 2, encoded.size(), scale, llrs.data(), 1008);
            interleaver.deinterleave_inplace(llrs, deint);
            scrambler.soft_descramble(llrs);
            codec.decode_frame(llrs, decoded);
        } else {
            LDPCCodec::AlignedFloatVector pairs(iq.size() * 2), llrs(encoded.size()), deint;
            for (size_t i = 0; i < iq.size(); ++i) {
                pairs[2 * i] = iq[i].real() * scale;
                pairs[2 * i + 1] = iq[i].imag() * scale;
            }
            modem.demap_scaled_pairs(pairs.data() + payload_begin * 2, encoded.size(), scale, llrs.data(), 1008);
            interleaver.deinterleave_inplace(llrs, deint);
            scrambler.soft_descramble(llrs);
            codec.decode_frame(llrs, decoded);
        }
        require(bytes == decoded, "LDPC round trip changed payload bytes");
    }
}

void check_yaml() {
    const auto path = std::filesystem::temp_directory_path() / ("openisac-modulation-" + std::to_string(getpid()) + ".yaml");
    for (const char* name : names) {
        { std::ofstream output(path); output << "modulation:\n  scheme: " << name << '\n'; }
        Config bs = make_default_bs_config(), ue = make_default_ue_config();
        require(load_bs_config_from_yaml(bs, path.string()), "BS modulation YAML rejected");
        require(load_ue_config_from_yaml(ue, path.string()), "UE modulation YAML rejected");
        require(bs.modulation.scheme == parse_modulation_scheme(name) && ue.modulation.scheme == bs.modulation.scheme,
                "BS/UE did not load the same constellation");
    }
    { std::ofstream output(path); output << "modulation:\n  scheme: typo\n"; }
    Config cfg;
    require(!load_bs_config_from_yaml(cfg, path.string()), "BS accepted an unknown constellation");
    require(!load_ue_config_from_yaml(cfg, path.string()), "UE accepted an unknown constellation");
    std::filesystem::remove(path);
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "supply repository root");
        for (const char* name : names) {
            DigitalModulation modem(parse_modulation_scheme(name));
            check_constellation(modem);
            check_hard_reference(modem);
            check_boundaries(modem);
            for (bool fixed : {false, true}) {
                LdpcDecodeConfig decode;
                decode.fixed_point = fixed;
                auto config = make_ldpc_5041008_cfg(decode);
                config.h_matrix_path = std::string(argv[1]) + "/LDPC_504_1008.alist";
                config.g_matrix_path = std::string(argv[1]) + "/LDPC_504_1008G.alist";
                LDPCCodec codec(config);
                check_packet(modem, codec, fixed);
            }
            std::cout << name << ": constellation, soft demapping, sensing classification and float/Q16 LDPC passed\n";
        }
        check_yaml();
        std::cout << "Modulation checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
