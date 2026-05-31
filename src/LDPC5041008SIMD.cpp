#include "LDPC5041008SIMD.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace {

static inline std::string trim_copy(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return std::string();
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

} // namespace

LDPC5041008SIMD::LDPC5041008SIMD(const std::string& h_alist_path,
                                 const std::string& g_alist_path)
    : h_alist_(read_alist(h_alist_path))
    , g_alist_(read_alist(g_alist_path)) {
    build_generator_bitmasks();
    build_parity_check_bitmasks();
}

int LDPC5041008SIMD::simd_width() {
    return mipp::N<int32_t>();
}

bool LDPC5041008SIMD::next_data_line(std::ifstream& in, std::string& line) {
    while (std::getline(in, line)) {
        const auto hash_pos = line.find('#');
        if (hash_pos != std::string::npos) line.erase(hash_pos);
        line = trim_copy(line);
        if (!line.empty()) return true;
    }
    return false;
}

LDPC5041008SIMD::Alist LDPC5041008SIMD::read_alist(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open alist: " + path);
    }

    Alist a;
    std::string line;

    if (!next_data_line(in, line)) {
        throw std::runtime_error("Invalid alist (empty): " + path);
    }
    {
        std::istringstream iss(line);
        if (!(iss >> a.n >> a.m) || a.n <= 0 || a.m <= 0) {
            throw std::runtime_error("Invalid alist header dims: " + path);
        }
    }

    // max degrees line (can be ignored but must exist)
    if (!next_data_line(in, line)) {
        throw std::runtime_error("Invalid alist (missing max degree line): " + path);
    }

    a.col_deg.assign(static_cast<size_t>(a.n), 0);
    a.row_deg.assign(static_cast<size_t>(a.m), 0);

    size_t idx = 0;
    while (idx < static_cast<size_t>(a.n) && next_data_line(in, line)) {
        std::istringstream iss(line);
        while (idx < static_cast<size_t>(a.n) && (iss >> a.col_deg[idx])) {
            if (a.col_deg[idx] < 0) {
                throw std::runtime_error("Invalid negative col degree in alist: " + path);
            }
            idx++;
        }
    }
    if (idx != static_cast<size_t>(a.n)) {
        throw std::runtime_error("Invalid alist col degree section: " + path);
    }

    idx = 0;
    while (idx < static_cast<size_t>(a.m) && next_data_line(in, line)) {
        std::istringstream iss(line);
        while (idx < static_cast<size_t>(a.m) && (iss >> a.row_deg[idx])) {
            if (a.row_deg[idx] < 0) {
                throw std::runtime_error("Invalid negative row degree in alist: " + path);
            }
            idx++;
        }
    }
    if (idx != static_cast<size_t>(a.m)) {
        throw std::runtime_error("Invalid alist row degree section: " + path);
    }

    a.col_adj.assign(static_cast<size_t>(a.n), {});
    a.row_adj.assign(static_cast<size_t>(a.m), {});

    for (int col = 0; col < a.n; ++col) {
        if (!next_data_line(in, line)) {
            throw std::runtime_error("Invalid alist column adjacency section: " + path);
        }

        std::istringstream iss(line);
        for (int k = 0; k < a.col_deg[static_cast<size_t>(col)]; ++k) {
            int r = 0;
            if (!(iss >> r)) break;
            if (r <= 0) continue;
            const int rr = r - 1;
            if (rr >= a.m) {
                throw std::runtime_error("Out-of-range col adjacency in alist: " + path);
            }
            a.col_adj[static_cast<size_t>(col)].push_back(rr);
        }
    }

    for (int row = 0; row < a.m; ++row) {
        if (!next_data_line(in, line)) {
            throw std::runtime_error("Invalid alist row adjacency section: " + path);
        }

        std::istringstream iss(line);
        for (int k = 0; k < a.row_deg[static_cast<size_t>(row)]; ++k) {
            int c = 0;
            if (!(iss >> c)) break;
            if (c <= 0) continue;
            const int cc = c - 1;
            if (cc >= a.n) {
                throw std::runtime_error("Out-of-range row adjacency in alist: " + path);
            }
            a.row_adj[static_cast<size_t>(row)].push_back(cc);
        }
    }

    return a;
}

void LDPC5041008SIMD::build_generator_bitmasks() {
    g_info_words_.assign(static_cast<size_t>(K * CODEWORD_WORDS), 0ull);

    if (g_alist_.n == N && g_alist_.m == K) {
        // Rows are information-bit rows, columns are codeword bits.
        for (int info_idx = 0; info_idx < K; ++info_idx) {
            const auto& cols = g_alist_.row_adj[static_cast<size_t>(info_idx)];
            for (const int cw_idx : cols) {
                const int w = cw_idx >> 6;
                const int b = cw_idx & 63;
                g_info_words_[static_cast<size_t>(info_idx * CODEWORD_WORDS + w)] |= (1ull << b);
            }
        }
        return;
    }

    if (g_alist_.n == K && g_alist_.m == N) {
        // Transposed orientation: rows are codeword bits, columns are information bits.
        for (int cw_idx = 0; cw_idx < N; ++cw_idx) {
            const auto& info_cols = g_alist_.row_adj[static_cast<size_t>(cw_idx)];
            for (const int info_idx : info_cols) {
                const int w = cw_idx >> 6;
                const int b = cw_idx & 63;
                g_info_words_[static_cast<size_t>(info_idx * CODEWORD_WORDS + w)] |= (1ull << b);
            }
        }
        return;
    }

    std::stringstream ss;
    ss << "G alist dims mismatch for LDPC_504_1008 (got n=" << g_alist_.n << ", m=" << g_alist_.m << ")";
    throw std::runtime_error(ss.str());
}

void LDPC5041008SIMD::build_parity_check_bitmasks() {
    h_row_words_.assign(static_cast<size_t>(M * CODEWORD_WORDS), 0ull);

    if (h_alist_.n == N && h_alist_.m == M) {
        for (int row = 0; row < M; ++row) {
            const auto& cols = h_alist_.row_adj[static_cast<size_t>(row)];
            for (const int cw_idx : cols) {
                const int w = cw_idx >> 6;
                const int b = cw_idx & 63;
                h_row_words_[static_cast<size_t>(row * CODEWORD_WORDS + w)] |= (1ull << b);
            }
        }
        return;
    }

    if (h_alist_.n == M && h_alist_.m == N) {
        // Transposed orientation: columns correspond to parity-check rows.
        for (int row = 0; row < M; ++row) {
            const auto& cols = h_alist_.col_adj[static_cast<size_t>(row)];
            for (const int cw_idx : cols) {
                const int w = cw_idx >> 6;
                const int b = cw_idx & 63;
                h_row_words_[static_cast<size_t>(row * CODEWORD_WORDS + w)] |= (1ull << b);
            }
        }
        return;
    }

    std::stringstream ss;
    ss << "H alist dims mismatch for LDPC_504_1008 (got n=" << h_alist_.n << ", m=" << h_alist_.m << ")";
    throw std::runtime_error(ss.str());
}

void LDPC5041008SIMD::unpack_info_bytes(const int8_t* info_bytes, int32_t* info_bits) {
    #pragma omp simd
    for (int i = 0; i < K; ++i) {
        const uint8_t v = static_cast<uint8_t>(info_bytes[i >> 3]);
        info_bits[i] = static_cast<int32_t>((v >> (7 - (i & 7))) & 1u);
    }
}

void LDPC5041008SIMD::encode_frame_bits(const int32_t* info_bits, int32_t* codeword_bits) const {
    std::array<uint64_t, CODEWORD_WORDS> cw_words{};

    #pragma omp simd
    for (int w = 0; w < CODEWORD_WORDS; ++w) {
        cw_words[static_cast<size_t>(w)] = 0ull;
    }

    for (int info_idx = 0; info_idx < K; ++info_idx) {
        const uint64_t mask = static_cast<uint64_t>(-static_cast<int64_t>(info_bits[info_idx] & 1));
        const uint64_t* row = &g_info_words_[static_cast<size_t>(info_idx * CODEWORD_WORDS)];

        #pragma omp simd
        for (int w = 0; w < CODEWORD_WORDS; ++w) {
            cw_words[static_cast<size_t>(w)] ^= (row[w] & mask);
        }
    }

    #pragma omp simd
    for (int i = 0; i < N; ++i) {
        codeword_bits[i] = static_cast<int32_t>((cw_words[static_cast<size_t>(i >> 6)] >> (i & 63)) & 1ull);
    }
}

void LDPC5041008SIMD::encode_bits(const IntVec& info_bits, IntVec& codeword_bits) const {
    if (info_bits.empty() || (info_bits.size() % static_cast<size_t>(K)) != 0) {
        throw std::runtime_error("encode_bits: info_bits size must be a non-zero multiple of K=504");
    }

    const size_t frames = info_bits.size() / static_cast<size_t>(K);
    codeword_bits.resize(frames * static_cast<size_t>(N));

    for (size_t f = 0; f < frames; ++f) {
        const int32_t* in = info_bits.data() + f * static_cast<size_t>(K);
        int32_t* out = codeword_bits.data() + f * static_cast<size_t>(N);
        encode_frame_bits(in, out);
    }
}

void LDPC5041008SIMD::encode_bytes(const ByteVec& info_bytes, IntVec& codeword_bits) const {
    if (info_bytes.empty() || (info_bytes.size() % static_cast<size_t>(K_BYTES)) != 0) {
        throw std::runtime_error("encode_bytes: info_bytes size must be a non-zero multiple of K_BYTES=63");
    }

    const size_t frames = info_bytes.size() / static_cast<size_t>(K_BYTES);
    codeword_bits.resize(frames * static_cast<size_t>(N));

    std::array<int32_t, K> info_bits{};
    for (size_t f = 0; f < frames; ++f) {
        const int8_t* in = info_bytes.data() + f * static_cast<size_t>(K_BYTES);
        int32_t* out = codeword_bits.data() + f * static_cast<size_t>(N);
        unpack_info_bytes(in, info_bits.data());
        encode_frame_bits(info_bits.data(), out);
    }
}

void LDPC5041008SIMD::pack_codeword_bits_to_words(const int32_t* codeword_bits, uint64_t* codeword_words) {
    for (int w = 0; w < CODEWORD_WORDS; ++w) {
        uint64_t word = 0ull;
        const int base = w << 6;

        #pragma omp simd reduction(|:word)
        for (int b = 0; b < 64; ++b) {
            const int idx = base + b;
            if (idx < N) {
                word |= static_cast<uint64_t>(codeword_bits[idx] & 1) << b;
            }
        }

        codeword_words[w] = word;
    }
}

int LDPC5041008SIMD::check_frame_syndrome(const int32_t* codeword_bits) const {
    std::array<uint64_t, CODEWORD_WORDS> cw_words{};
    pack_codeword_bits_to_words(codeword_bits, cw_words.data());

    int unsatisfied = 0;
    for (int row = 0; row < M; ++row) {
        const uint64_t* h = &h_row_words_[static_cast<size_t>(row * CODEWORD_WORDS)];
        int parity = 0;

        #pragma omp simd reduction(^:parity)
        for (int w = 0; w < CODEWORD_WORDS; ++w) {
            const uint64_t v = cw_words[static_cast<size_t>(w)] & h[w];
            parity ^= static_cast<int>(__builtin_popcountll(v) & 1u);
        }

        unsatisfied += (parity & 1);
    }

    return unsatisfied;
}

bool LDPC5041008SIMD::check_syndrome(const IntVec& codeword_bits, int* weight_out) const {
    if (codeword_bits.size() < static_cast<size_t>(N)) {
        if (weight_out) *weight_out = -1;
        return false;
    }

    const int w = check_frame_syndrome(codeword_bits.data());
    if (weight_out) *weight_out = w;
    return w == 0;
}

int LDPC5041008SIMD::check_syndrome_frames(const IntVec& codeword_bits, IntVec* weights_out) const {
    if (codeword_bits.empty() || (codeword_bits.size() % static_cast<size_t>(N)) != 0) {
        throw std::runtime_error("check_syndrome_frames: codeword_bits size must be a non-zero multiple of N=1008");
    }

    const size_t frames = codeword_bits.size() / static_cast<size_t>(N);
    if (weights_out) weights_out->assign(frames, 0);

    int bad_frames = 0;
    for (size_t f = 0; f < frames; ++f) {
        const int w = check_frame_syndrome(codeword_bits.data() + f * static_cast<size_t>(N));
        if (weights_out) (*weights_out)[f] = w;
        if (w != 0) ++bad_frames;
    }

    return bad_frames;
}
