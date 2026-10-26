#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include "LDPCCodec.hpp"
#include "OFDMCore.hpp"

namespace {
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
constexpr const char* rates[] = {"1/2", "2/3", "3/4"};
constexpr size_t lengths[] = {1008, 756, 672};
constexpr const char* modes[] = {"BPSK", "QPSK", "8PSK", "16QAM", "32QAM", "64QAM"};
void yaml_config() {
    const auto path = std::filesystem::temp_directory_path() /
        ("openisac-rate-" + std::to_string(getpid()) + ".yaml");
    struct Remove { std::filesystem::path p; ~Remove() { std::filesystem::remove(p); } } remove{path};
    for (auto loader : {load_bs_config_from_yaml, load_ue_config_from_yaml}) {
        std::ofstream(path) << "{}\n";
        Config legacy;
        require(loader(legacy, path.string()) && legacy.ldpc.code_rate == LdpcCodeRate::R12,
                "default code rate changed");
        for (size_t i=0; i<std::size(rates); ++i) {
            std::ofstream(path) << "ldpc:\n  code_rate: '" << rates[i] << "'\n";
            Config cfg;
            require(loader(cfg, path.string()), "code rate YAML failed");
            require(LdpcRateMatching(cfg.ldpc.code_rate).code_bits() == lengths[i], "rate length mismatch");
        }
        for (const char* invalid : {"0", "1", "0.75", "1/3", "4/3", "bogus"}) {
            std::ofstream(path) << "ldpc:\n  code_rate: '" << invalid << "'\n";
            Config cfg;
            require(!loader(cfg, path.string()), "unsupported rate accepted");
        }
    }
}
void round_trips(const std::string& root) {
    auto cfg=make_ldpc_5041008_cfg();cfg.fixed_point=true;
    cfg.h_matrix_path=root+"/LDPC_504_1008.alist";
    cfg.g_matrix_path=root+"/LDPC_504_1008G.alist";
    LDPCCodec codec(cfg);
    std::mt19937 random(5941);
    std::normal_distribution<float> noise(0.0f,0.004f);
    // Cross a full float and Q16 SIMD batch and exercise a partial final batch.
    constexpr size_t blocks=65;
    LDPCCodec::AlignedByteVector input(63*blocks);
    for(auto& value:input)value=static_cast<int8_t>(random());
    LDPCCodec::AlignedIntVector encoded,interleave_scratch;
    codec.encode_frame(input,encoded);
    BitBlockInterleaver interleaver(1008,21);
    for(size_t r=0;r<std::size(rates);++r) {
        const LdpcRateMatching rate(parse_ldpc_code_rate(rates[r]));
        for(const auto* name:modes) {
            const DigitalModulation modem(parse_modulation_scheme(name));
            Scrambler scrambler(LdpcPacketFraming::scrambler_bits(modem),0x5A);
            auto transmitted=encoded;
            scrambler.scramble(transmitted);
            interleaver.interleave_inplace(transmitted,interleave_scratch);
            const auto before=transmitted;
            rate.match_inplace(transmitted,interleave_scratch);
            require(transmitted.size()==blocks*lengths[r],"wrong transmitted length");
            if(r==0)require(transmitted==before,"legacy rate bitstream changed");
            LDPCCodec::AlignedIntVector symbols;
            modem.pack_bits(transmitted,symbols,rate.code_bits());
            require(symbols.size()==blocks*modem.symbol_count(lengths[r]),"block RE padding mismatch");
            const LdpcMiniHeader header{LdpcPacketFraming::kVersion,0,
                static_cast<uint16_t>(input.size()),LdpcPacketFraming::payload_blocks_field_for_len(input.size()),123};
            LDPCCodec::AlignedIntVector frame(LdpcPacketFraming::kControlSymbols);
            LdpcPacketFraming::write_control_qpsk(header,frame.data());
            frame.insert(frame.end(),symbols.begin(),symbols.end());
            std::vector<uint8_t> mask(frame.size());
            classify_payload_modulation(frame.size(),[&](size_t i){return modem.modulate_token(frame[i]);},
                                        modem,mask.data(),rate.code_bits());
            for(size_t i=0;i<mask.size();++i)
                require(mask[i]==(i<LdpcPacketFraming::kControlSymbols?0:1),"sensing payload classification mismatch");
            LDPCCodec::AlignedFloatVector pairs(symbols.size()*2),llrs(transmitted.size()),scratch;
            constexpr float scale=96.0f;
            for(size_t i=0;i<symbols.size();++i) {
                const auto point=modem.modulate_token(symbols[i]);
                pairs[2*i]=(point.real()+noise(random))*scale;
                pairs[2*i+1]=(point.imag()+noise(random))*scale;
            }
            modem.demap_scaled_pairs(pairs.data(),llrs.size(),scale,llrs.data(),rate.code_bits());
            LDPCCodec::AlignedShortVector q16(llrs.size()),q_scratch;
            for(size_t i=0;i<llrs.size();++i)q16[i]=static_cast<int16_t>(std::clamp(std::lround(llrs[i]*16.0f),-32767L,32767L));
            rate.recover_inplace(llrs,scratch);
            rate.recover_inplace(q16,q_scratch);
            interleaver.deinterleave_inplace(llrs,scratch);
            interleaver.deinterleave_inplace(q16,q_scratch);
            scrambler.soft_descramble(llrs);scrambler.soft_descramble(q16);
            LDPCCodec::AlignedByteVector decoded;
            codec.decode_frame(llrs,decoded);require(decoded==input,"float payload mismatch");
            codec.decode_frame(q16,decoded);if(decoded!=input) throw std::runtime_error(std::string(rates[r])+" "+name+": Q16 payload mismatch");
            require(LdpcPacketFraming::packet_symbols(blocks,modem,rate.code_bits())==
                LdpcPacketFraming::kControlSymbols+symbols.size(),"packet extent mismatch");
        }
        std::cout<<rates[r]<<": six constellations, float/Q16 and partial SIMD batches passed\n";
    }
}
}
int main(int argc,char** argv) {
    try { require(argc==2,"usage: LDPCRateTest SOURCE_ROOT");yaml_config();round_trips(argv[1]); }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
