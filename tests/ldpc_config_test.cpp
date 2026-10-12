#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "LDPCCodec.hpp"
#include "LDPC5041008SIMD.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct TemporaryConfig {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("openisac-ldpc-config-" + std::to_string(getpid()) + ".yaml");
    ~TemporaryConfig() { std::filesystem::remove(path); }
    void write(const std::string& value) const { std::ofstream(path) << value; }
};

LDPCCodec::LDPCConfig codec_config(const LdpcDecodeConfig& decode, const std::string& root) {
    auto cfg = make_ldpc_5041008_cfg(decode);
    cfg.h_matrix_path = root + "/LDPC_504_1008.alist";
    cfg.g_matrix_path = root + "/LDPC_504_1008G.alist";
    return cfg;
}

size_t frame_errors(const LDPCCodec::AlignedByteVector& expected,
                    const LDPCCodec::AlignedByteVector& actual) {
    require(actual.size() == expected.size(), "decoded payload length changed");
    size_t errors = 0;
    for (size_t begin = 0; begin < expected.size(); begin += 63) {
        if (!std::equal(expected.begin() + begin, expected.begin() + begin + 63,
                        actual.begin() + begin)) {
            ++errors;
        }
    }
    return errors;
}

size_t decode_errors(const LdpcDecodeConfig& config, const std::string& root,
                     const LDPCCodec::AlignedFloatVector& llrs,
                     const LDPCCodec::AlignedByteVector& expected) {
    LDPCCodec codec(codec_config(config, root));
    LDPCCodec::AlignedByteVector decoded;
    if (config.fixed_point) {
        LDPCCodec::AlignedShortVector quantized(llrs.size());
        for (size_t i = 0; i < llrs.size(); ++i) {
            const long value = std::lround(llrs[i] * config.fixed_point_scale);
            quantized[i] = static_cast<int16_t>(std::clamp(value, -32767L, 32767L));
        }
        codec.decode_frame(quantized, decoded);
    } else {
        codec.decode_frame(llrs, decoded);
    }
    return frame_errors(expected, decoded);
}

void test_yaml() {
    TemporaryConfig file;
    for (const auto loader : {load_bs_config_from_yaml, load_ue_config_from_yaml}) {
        file.write("{}\n");
        Config legacy;
        require(loader(legacy, file.path.string()), "legacy YAML failed to load");
        require(legacy.ldpc.decoder_iterations == 6 &&
                legacy.ldpc.nms_normalization_factor == 1.0f, "legacy defaults changed");

        file.write("ldpc:\n  decoder_iterations: 12\n  nms_normalization_factor: 0.75\n"
                   "  fixed_point: true\n  fixed_point_scale: 32\n");
        Config tuned;
        require(loader(tuned, file.path.string()), "configured YAML failed to load");
        require(tuned.ldpc.decoder_iterations == 12 &&
                tuned.ldpc.nms_normalization_factor == 0.75f && tuned.ldpc.fixed_point,
                "CPU LDPC YAML settings were not loaded");

        for (const std::string invalid : {
                "decoder_iterations: 0", "decoder_iterations: -1",
                "nms_normalization_factor: 0", "nms_normalization_factor: 1.1",
                "nms_normalization_factor: .nan", "nms_normalization_factor: .inf",
                "fixed_point: true\n  nms_normalization_factor: 0.8"}) {
            file.write("ldpc:\n  " + invalid + "\n");
            Config rejected;
            require(!loader(rejected, file.path.string()), "unsupported LDPC YAML was accepted");
        }
    }
}

void test_decoder(const std::string& root) {
    constexpr size_t kFrames = 256;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> byte(0, 255);
    LDPCCodec::AlignedByteVector info(kFrames * 63);
    for (auto& value : info) value = static_cast<int8_t>(byte(rng));
    LdpcDecodeConfig defaults;
    LDPCCodec encoder(codec_config(defaults, root));
    LDPCCodec::AlignedIntVector encoded;
    encoder.encode_frame(info, encoded);
    require(encoded.size() == kFrames * 1008, "unexpected LDPC codeword geometry");
    LDPC5041008SIMD syndrome(root + "/LDPC_504_1008.alist", root + "/LDPC_504_1008G.alist");
    const LDPC5041008SIMD::IntVec words(encoded.begin(), encoded.end());
    const int invalid_codewords = syndrome.check_syndrome_frames(words);
    std::cout << "Encoded codewords with nonzero syndrome=" << invalid_codewords << '\n';
    require(invalid_codewords == 0, "encoder generated invalid codewords");
    LDPCCodec::AlignedFloatVector clean(encoded.size());
    for (size_t i = 0; i < encoded.size(); ++i) clean[i] = encoded[i] ? -10.0f : 10.0f;
    const size_t default_clean_errors = decode_errors(defaults, root, clean, info);
    std::cout << "Clean codewords: default=" << default_clean_errors << '\n';
    require(default_clean_errors == 0, "default decode broke clean payloads");

    LdpcDecodeConfig tuned;
    tuned.decoder_iterations = 12;
    tuned.nms_normalization_factor = 0.75f;
    tuned.fixed_point_scale = 32;
    auto iteration_clean = defaults;
    iteration_clean.decoder_iterations = 12;
    auto factor_clean = defaults;
    factor_clean.nms_normalization_factor = 0.75f;
    const size_t iteration_clean_errors = decode_errors(iteration_clean, root, clean, info);
    const size_t factor_clean_errors = decode_errors(factor_clean, root, clean, info);
    const size_t tuned_clean_errors = decode_errors(tuned, root, clean, info);
    std::cout << "Clean codewords: iterations=" << iteration_clean_errors
              << " factor=" << factor_clean_errors << " combined=" << tuned_clean_errors << '\n';
    require(iteration_clean_errors == 0, "iteration tuning broke clean payloads");
    require(factor_clean_errors == 0, "NMS tuning broke clean payloads");
    require(tuned_clean_errors == 0, "tuned decode broke clean payloads");
    tuned.fixed_point = true;
    require(decode_errors(tuned, root, clean, info) == 0, "configured Q16 broke clean payloads");
    tuned.fixed_point = false;

    const float sigma = std::sqrt(1.0f / (2.0f * std::pow(10.0f, -0.5f / 10.0f)));
    std::normal_distribution<float> noise(0.0f, sigma);
    LDPCCodec::AlignedFloatVector noisy(encoded.size());
    for (size_t i = 0; i < encoded.size(); ++i) {
        const float signal = encoded[i] ? -1.0f : 1.0f;
        noisy[i] = 2.0f * (signal + noise(rng)) / (sigma * sigma);
    }
    const size_t baseline = decode_errors(defaults, root, noisy, info);
    auto iterations_only = defaults;
    iterations_only.decoder_iterations = 12;
    auto factor_only = defaults;
    factor_only.nms_normalization_factor = 0.875f;
    const size_t iteration_errors = decode_errors(iterations_only, root, noisy, info);
    const size_t factor_errors = decode_errors(factor_only, root, noisy, info);
    const size_t tuned_errors = decode_errors(tuned, root, noisy, info);
    std::cout << "Noisy codewords: default=" << baseline << " iterations=" << iteration_errors
              << " factor=" << factor_errors << " combined=" << tuned_errors << '\n';
    require(iteration_errors < baseline, "iteration setting did not improve the noisy fixture");
    require(factor_errors < baseline, "NMS setting did not improve the noisy fixture");
    require(tuned_errors < baseline, "combined settings did not reach the decoder");
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "pass the source directory containing the LDPC matrices");
        test_yaml();
        test_decoder(argv[1]);
        std::cout << "CPU LDPC configuration and decoding checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
