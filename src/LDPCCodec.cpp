#include "LDPCCodec.hpp"

#include <aff3ct.hpp>

#include "LDPC5041008SIMD.hpp"

struct LDPCCodec::Impl {
    LDPCConfig cfg;
    std::unique_ptr<aff3ct::factory::Codec_LDPC> codec_factory;
    std::unique_ptr<aff3ct::tools::Codec_LDPC<int, float>> codec;
    // Optional int16 (Q16) decoder, built only when cfg.fixed_point is set.
    // AFF3CT NMS supports 16-bit fixed point; INTER SIMD packs 32 frames/AVX-512
    // register at int16 (vs 16 at float), so its n_frames is doubled.
    std::unique_ptr<aff3ct::tools::Codec_LDPC<short, short>> codec_q16;
    std::unique_ptr<LDPC5041008SIMD> custom_encoder;
    LDPC5041008SIMD::ByteVec custom_input_bytes;
    LDPC5041008SIMD::IntVec custom_encoded_bits;
    AlignedIntVector unpacked_bits;
    AlignedIntVector tmp_in;
    AlignedIntVector tmp_out;

    explicit Impl(const LDPCConfig& config)
        : cfg(config) {
        auto parent_dir = [](const std::string& p) {
            const size_t slash = p.rfind('/');
            if (slash == std::string::npos) {
                return std::string(".");
            }
            if (slash == 0) {
                return std::string("/");
            }
            return p.substr(0, slash);
        };

        if (!cfg.h_matrix_path.empty() && cfg.h_matrix_path[0] != '/') {
            cfg.h_matrix_path = get_executable_dir() + "/" + cfg.h_matrix_path;
        }
        if (!cfg.g_matrix_path.empty() && cfg.g_matrix_path[0] != '/') {
            cfg.g_matrix_path = get_executable_dir() + "/" + cfg.g_matrix_path;
        }
        if (!cfg.g_matrix_path.empty() && !path_exists(cfg.g_matrix_path)) {
            const std::string g_parent = parent_dir(cfg.g_matrix_path);
            const std::string alt_g_1 = g_parent + "/LDPC_504_1008G.alist";
            const std::string alt_g_2 = g_parent + "/PEGReg504x1008_Gen.alist";
            if (path_exists(alt_g_1)) {
                cfg.g_matrix_path = alt_g_1;
            } else if (path_exists(alt_g_2)) {
                cfg.g_matrix_path = alt_g_2;
            }
        }

        if (cfg.use_custom_encoder) {
            custom_encoder = std::make_unique<LDPC5041008SIMD>(cfg.h_matrix_path, cfg.g_matrix_path);
        }

        init_aff3ct_params();
        codec = std::unique_ptr<aff3ct::tools::Codec_LDPC<int, float>>(codec_factory->build<int, float>());
        codec->set_n_frames(cfg.n_frames);

        if (cfg.fixed_point) {
            codec_q16 = std::unique_ptr<aff3ct::tools::Codec_LDPC<short, short>>(
                codec_factory->build<short, short>());
            codec_q16->set_n_frames(cfg.n_frames * 2);
        }
    }

    void init_aff3ct_params() {
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
        if (!cfg.use_custom_encoder && !cfg.g_matrix_path.empty()) {
            args.push_back("--enc-g-path");
            args.push_back(cfg.g_matrix_path);
        }

        codec_factory = std::make_unique<aff3ct::factory::Codec_LDPC>();

        std::vector<char*> argv;
        argv.reserve(args.size());
        for (auto& arg : args) {
            argv.push_back(arg.data());
        }
        int argc = static_cast<int>(argv.size());

        std::vector<aff3ct::factory::Factory*> params_list;
        params_list.push_back(codec_factory.get());

        aff3ct::tools::Command_parser cp(argc, argv.data(), params_list, true);
        aff3ct::tools::Header::print_parameters(params_list);
    }
};

LDPCCodec::LDPCCodec(const LDPCConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

LDPCCodec::~LDPCCodec() = default;

void LDPCCodec::encode_frame(const AlignedByteVector& input, AlignedIntVector& encoded_bits) {
    if (input.empty()) {
        encoded_bits.clear();
        return;
    }

    if (impl_->custom_encoder) {
        impl_->custom_input_bytes.resize(input.size());
        std::copy(input.begin(), input.end(), impl_->custom_input_bytes.begin());

        impl_->custom_encoder->encode_bytes(
            impl_->custom_input_bytes,
            impl_->custom_encoded_bits);

        encoded_bits.resize(impl_->custom_encoded_bits.size());
        std::copy(
            impl_->custom_encoded_bits.begin(),
            impl_->custom_encoded_bits.end(),
            encoded_bits.begin());
        return;
    }

    auto& encoder = impl_->codec->get_encoder();
    impl_->unpacked_bits.resize(input.size() * 8);
    unpack_bits(input, impl_->unpacked_bits);
    const size_t N = encoder.get_N();
    const size_t K = encoder.get_K();
    const size_t batch = encoder.get_n_frames();
    const size_t total_frames = impl_->unpacked_bits.size() / K;

    encoded_bits.resize(total_frames * N);

    size_t i = 0;
    for (; i + batch <= total_frames; i += batch) {
        encoder.encode(
            impl_->unpacked_bits.data() + i * K,
            encoded_bits.data() + i * N,
            -1,
            false);
    }

    const size_t remaining = total_frames - i;
    if (remaining > 0) {
        impl_->tmp_in.assign(batch * K, 0);
        impl_->tmp_out.assign(batch * N, 0);
        std::memcpy(
            impl_->tmp_in.data(),
            impl_->unpacked_bits.data() + i * K,
            remaining * K * sizeof(int32_t));
        encoder.encode(impl_->tmp_in.data(), impl_->tmp_out.data(), -1, false);
        std::memcpy(
            encoded_bits.data() + i * N,
            impl_->tmp_out.data(),
            remaining * N * sizeof(int32_t));
    }
}

void LDPCCodec::decode_frame(const AlignedFloatVector& llr_input, AlignedByteVector& decoded_bytes) {
    auto& decoder = impl_->codec->get_decoder_siho();
    const size_t N = decoder.get_N();
    const size_t K = decoder.get_K();
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
    pack_bits(decoded_bits, decoded_bytes);
}

void LDPCCodec::decode_frame(const AlignedShortVector& llr_input, AlignedByteVector& decoded_bytes) {
    if (!impl_->codec_q16) {
        throw std::runtime_error(
            "LDPCCodec::decode_frame(int16) requires LDPCConfig::fixed_point = true");
    }
    auto& decoder = impl_->codec_q16->get_decoder_siho();
    const size_t N = decoder.get_N();
    const size_t K = decoder.get_K();
    const size_t batch = decoder.get_n_frames();
    const size_t total_frames = llr_input.size() / N;

    // decode_siho writes short hard bits; collect them, then pack to bytes via
    // an int view (pack_bits takes AlignedIntVector).
    std::vector<short> decoded_bits(total_frames * K, 0);

    size_t i = 0;
    for (; i + batch <= total_frames; i += batch) {
        decoder.decode_siho(llr_input.data() + i * N, decoded_bits.data() + i * K, -1, false);
    }

    const size_t remaining = total_frames - i;
    if (remaining > 0) {
        AlignedShortVector tmp_in(batch * N, 0);
        std::vector<short> tmp_out(batch * K, 0);
        std::memcpy(tmp_in.data(), llr_input.data() + i * N, remaining * N * sizeof(int16_t));
        decoder.decode_siho(tmp_in.data(), tmp_out.data(), -1, false);
        std::memcpy(decoded_bits.data() + i * K, tmp_out.data(), remaining * K * sizeof(short));
    }

    AlignedIntVector decoded_bits_int(decoded_bits.begin(), decoded_bits.end());
    decoded_bytes.resize(decoded_bits_int.size() / 8);
    pack_bits(decoded_bits_int, decoded_bytes);
}

size_t LDPCCodec::get_K() const {
    if (impl_->custom_encoder) {
        return static_cast<size_t>(LDPC5041008SIMD::K);
    }
    return impl_->codec->get_encoder().get_K();
}

size_t LDPCCodec::get_N() const {
    if (impl_->custom_encoder) {
        return static_cast<size_t>(LDPC5041008SIMD::N);
    }
    return impl_->codec->get_encoder().get_N();
}

void LDPCCodec::pack_bits_qpsk(const AlignedIntVector& bits, AlignedIntVector& qpsk_ints) {
    const size_t bit_count = bits.size();
    const size_t symbol_count = (bit_count + 1) / 2;
    qpsk_ints.resize(symbol_count);
    const size_t even_pairs = bit_count / 2;
    #pragma omp simd
    for (size_t k = 0; k < even_pairs; ++k) {
        int b0 = bits[2 * k] & 1;
        int b1 = bits[2 * k + 1] & 1;
        qpsk_ints[k] = (b0 << 1) | b1;
    }
    if (bit_count & 1) {
        qpsk_ints[even_pairs] = (bits[bit_count - 1] & 1) << 1;
    }
}

void LDPCCodec::unpack_bits(const AlignedByteVector& input_data, AlignedIntVector& unpacked_bits) {
    const int input_bytes = static_cast<int>(input_data.size());
    #pragma omp simd
    for (int byte_idx = 0; byte_idx < input_bytes; ++byte_idx) {
        uint8_t byte = static_cast<uint8_t>(input_data[byte_idx]);
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

void LDPCCodec::pack_bits(const AlignedIntVector& bits, AlignedByteVector& output_data) {
    const int output_bytes = static_cast<int>(output_data.size());
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
        output_data[byte_idx] = static_cast<int8_t>(byte);
    }
}

LDPCCodec::LDPCConfig make_ldpc_5041008_cfg() {
    LDPCCodec::LDPCConfig c;
    c.h_matrix_path = "../LDPC_504_1008.alist";
    c.g_matrix_path = "../LDPC_504_1008_G_fromH.alist";
    c.decoder_iterations = 6;
    c.n_frames = 16;
    c.enc_type = "LDPC_H";
    c.enc_g_method = "IDENTITY";
    c.dec_type = "BP_HORIZONTAL_LAYERED";
    c.dec_implem = "NMS";
    c.dec_simd = "INTER";
    c.use_custom_encoder = true;
    return c;
}
