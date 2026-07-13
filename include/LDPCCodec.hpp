#ifndef LDPCCODEC_HPP
#define LDPCCODEC_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Common.hpp"

/**
 * @brief LDPC Codec Wrapper (custom LDPC5041008 encoder + AFF3CT decoder).
 *
 * Keeps AFF3CT/MIPP implementation details out of CUDA translation units.
 */
class LDPCCodec {
public:
    using AlignedByteVector = std::vector<int8_t, AlignedAllocator<int8_t, 64>>;
    using AlignedIntVector = std::vector<int32_t, AlignedAllocator<int32_t, 64>>;
    using AlignedFloatVector = std::vector<float, AlignedAllocator<float, 64>>;
    using AlignedShortVector = std::vector<int16_t, AlignedAllocator<int16_t, 64>>;

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
        // When true, additionally build an int16 (Q16) layered-NMS decoder so the
        // AlignedShortVector decode_frame overload can be used. The float decoder
        // is always built; the float decode path is unaffected.
        bool fixed_point = false;
    };

    explicit LDPCCodec(const LDPCConfig& config);
    ~LDPCCodec();
    LDPCCodec(const LDPCCodec&) = delete;
    LDPCCodec& operator=(const LDPCCodec&) = delete;
    LDPCCodec(LDPCCodec&&) = delete;
    LDPCCodec& operator=(LDPCCodec&&) = delete;

    void encode_frame(const AlignedByteVector& input, AlignedIntVector& encoded_bits);
    void decode_frame(const AlignedFloatVector& llr_input, AlignedByteVector& decoded_bytes);
    // int16 (Q16) decode path. Requires LDPCConfig::fixed_point = true at
    // construction. Input LLRs are quantized int16 (pow2 scaled at the demapper).
    void decode_frame(const AlignedShortVector& llr_input, AlignedByteVector& decoded_bytes);

    size_t get_K() const;
    size_t get_N() const;

    /**
     * @brief Pack bits into QPSK symbols (0-3).
     *
     * Each pair of bits corresponds to one QPSK symbol.
     */
    static void pack_bits_qpsk(const AlignedIntVector& bits, AlignedIntVector& qpsk_ints);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    static void unpack_bits(const AlignedByteVector& input_data, AlignedIntVector& unpacked_bits);
    static void pack_bits(const AlignedIntVector& bits, AlignedByteVector& output_data);
};

/**
 * @brief Build the default LDPC(1008,504) codec configuration.
 */
LDPCCodec::LDPCConfig make_ldpc_5041008_cfg();

#endif // LDPCCODEC_HPP
