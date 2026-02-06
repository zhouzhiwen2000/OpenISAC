#ifndef LDPC5041008SIMD_HPP
#define LDPC5041008SIMD_HPP

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <mipp.h>

class LDPC5041008SIMD {
public:
    using ByteVec = mipp::vector<int8_t>;
    using IntVec = mipp::vector<int32_t>;

    static constexpr int K = 504;
    static constexpr int N = 1008;
    static constexpr int M = 504;
    static constexpr int K_BYTES = K / 8;

    LDPC5041008SIMD(const std::string& h_alist_path,
                    const std::string& g_alist_path);

    // info_bits.size() must be a multiple of K, output is resized to frames * N.
    void encode_bits(const IntVec& info_bits, IntVec& codeword_bits) const;

    // info_bytes.size() must be a multiple of K_BYTES, output is resized to frames * N.
    void encode_bytes(const ByteVec& info_bytes, IntVec& codeword_bits) const;

    // Checks only the first frame (first N bits).
    bool check_syndrome(const IntVec& codeword_bits, int* weight_out = nullptr) const;

    // Checks all frames in codeword_bits (size must be multiple of N).
    // Returns the number of frames that fail syndrome.
    int check_syndrome_frames(const IntVec& codeword_bits, IntVec* weights_out = nullptr) const;

    static int simd_width();

private:
    struct Alist {
        int n = 0; // columns
        int m = 0; // rows
        std::vector<int> col_deg;
        std::vector<int> row_deg;
        std::vector<std::vector<int>> col_adj; // per column, 0-based row indices
        std::vector<std::vector<int>> row_adj; // per row, 0-based column indices
    };

    static constexpr int CODEWORD_WORDS = (N + 63) / 64;

    Alist h_alist_;
    Alist g_alist_;

    // Packed generator matrix as K rows, each row is N bits (CODEWORD_WORDS words).
    std::vector<uint64_t> g_info_words_;
    // Packed parity-check matrix as M rows, each row is N bits (CODEWORD_WORDS words).
    std::vector<uint64_t> h_row_words_;

    static Alist read_alist(const std::string& path);
    static bool next_data_line(std::ifstream& in, std::string& line);

    void build_generator_bitmasks();
    void build_parity_check_bitmasks();

    void encode_frame_bits(const int32_t* info_bits, int32_t* codeword_bits) const;
    static void unpack_info_bytes(const int8_t* info_bytes, int32_t* info_bits);
    static void pack_codeword_bits_to_words(const int32_t* codeword_bits, uint64_t* codeword_words);

    int check_frame_syndrome(const int32_t* codeword_bits) const;
};

#endif
