#ifndef OFDM_CORE_HPP
#define OFDM_CORE_HPP

/**
 * @file OFDMCore.hpp
 * @brief Hardware-independent core computation classes for OFDM modulation/demodulation and sensing.
 * 
 * This header contains pure computation classes without any hardware interaction,
 * buffer management, or thread communication. All I/O operations remain in the Engine classes.
 */

#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <fftw3.h>
#include "Common.hpp"
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif


/**
 * @brief Manager for FFTW Wisdom.
 * 
 * Handles importing and exporting FFTW wisdom to/from a file.
 * This allows saving optimized FFT plans to disk to speed up subsequent initializations.
 */
class FFTWManager {
public:
    static void import_wisdom(const std::string& filename = "fftw_wisdom.dat") {
        if (FILE* f = std::fopen(filename.c_str(), "r")) {
            fftwf_import_wisdom_from_file(f);
            std::fclose(f);
            std::cout << "Imported FFTW wisdom from " << filename << std::endl;
        } else {
            std::cout << "No existing FFTW wisdom found (will act as cold start)." << std::endl;
        }
    }

    static void export_wisdom(const std::string& filename = "fftw_wisdom.dat") {
        if (FILE* f = std::fopen(filename.c_str(), "w")) {
            fftwf_export_wisdom_to_file(f);
            std::fclose(f);
            std::cout << "Exported FFTW wisdom to " << filename << std::endl;
        } else {
            std::cerr << "Failed to export FFTW wisdom to " << filename << std::endl;
        }
    }
};

/**
 * @brief QPSK Scrambler/Descrambler.
 * 
 * Uses a Linear Feedback Shift Register (LFSR) to generate a pseudo-random sequence
 * for scrambling and descrambling bits. This helps in randomizing the data to avoid
 * long sequences of zeros or ones, which avoids high PAPRs in OFDM.
 */
class Scrambler {
public:
    Scrambler(size_t max_bits, uint8_t init = 0x5A)
        : scramble_seq_(max_bits)
    {
        uint8_t lfsr = init;
        for (size_t i = 0; i < max_bits; ++i) {
            scramble_seq_[i] = ((lfsr >> 7) ^ (lfsr >> 3) ^ (lfsr >> 2) ^ (lfsr >> 1)) & 1;
            lfsr = ((lfsr << 1) | scramble_seq_[i]) & 0xFF;
        }
    }

    // Scramble (in-place)
    template<typename Vec>
    void scramble(Vec& bits) const {
        size_t n = bits.size();
        size_t m = std::min(n, scramble_seq_.size());
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < m; ++i) {
            bits[i] ^= scramble_seq_[i];
        }
    }

    // Descramble (in-place)
    template<typename Vec>
    void descramble(Vec& bits) const {
        scramble(bits); // Same as scrambling
    }

    // Soft descramble (descramble LLR values)
    template<typename FloatVec>
    void soft_descramble(FloatVec& llr_values) const {
        size_t n = llr_values.size();
        size_t m = std::min(n, scramble_seq_.size());
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < m; ++i) {
            if (scramble_seq_[i] == 1) {
                llr_values[i] = -llr_values[i]; // Flip LLR sign if scramble bit is 1
            }
            // Keep LLR as is if scramble bit is 0
        }
    }

private:
    std::vector<uint8_t> scramble_seq_;
};

/**
 * @brief QPSK Modulator/Demodulator.
 * 
 * Provides QPSK modulation and demodulation operations using a pre-computed
 * lookup table for maximum performance.
 */
class QPSKModulator {
public:
    static constexpr float SQRT_2_INV = 0.7071067811865476f;
    
    // Pre-computed QPSK constellation lookup table
    static constexpr std::array<float, 8> QPSK_TABLE_FLAT = {{
        SQRT_2_INV,  SQRT_2_INV,   // 00: (+, +)
        SQRT_2_INV, -SQRT_2_INV,   // 01: (+, -)
       -SQRT_2_INV,  SQRT_2_INV,   // 10: (-, +)
       -SQRT_2_INV, -SQRT_2_INV    // 11: (-, -)
    }};

    /**
     * @brief Map 2-bit symbol (0-3) to complex constellation point.
     */
    inline std::complex<float> modulate(int symbol) const {
        const int idx = (symbol & 3) * 2;
        return std::complex<float>(QPSK_TABLE_FLAT[idx], QPSK_TABLE_FLAT[idx + 1]);
    }

    /**
     * @brief Hard decision demodulation (constellation point to 0-3).
     */
    inline int demodulate(std::complex<float> symbol) const {
        // MSB from real part, LSB from imaginary part
        return ((symbol.real() < 0.0f) ? 2 : 0) | ((symbol.imag() < 0.0f) ? 1 : 0);
    }

    /**
     * @brief Remodulate equalized symbol using hard decision.
     * Returns the closest QPSK constellation point.
     */
    inline std::complex<float> remodulate(std::complex<float> equalized_symbol) const {
        return std::complex<float>(
            std::copysign(SQRT_2_INV, equalized_symbol.real()),
            std::copysign(SQRT_2_INV, equalized_symbol.imag())
        );
    }

    /**
     * @brief Remodulate entire symbol using hard decision QPSK mapping.
     * Replaces pilot positions with known pilot values.
     */
    static void remodulate_symbol(
        const AlignedVector& equalized_symbol,
        const AlignedVector& pilots,
        const std::vector<size_t>& pilot_positions,
        AlignedVector& output
    ) {
        const size_t fft_size = equalized_symbol.size();
        output.resize(fft_size);
        
        auto* __restrict__ out_ptr = output.data();
        const auto* __restrict__ sym_ptr = equalized_symbol.data();
        
        // SIMD-friendly hard decision QPSK remodulation
        #pragma omp simd simdlen(16)
        for (size_t j = 0; j < fft_size; ++j) {
            const float re = sym_ptr[j].real();
            const float im = sym_ptr[j].imag();
            out_ptr[j] = std::complex<float>(
                std::copysign(SQRT_2_INV, re),
                std::copysign(SQRT_2_INV, im)
            );
        }
        
        // Replace pilot positions with known pilots
        for (auto pilot : pilot_positions) {
            out_ptr[pilot] = pilots[pilot];
        }
    }
};


/**
 * @brief Zadoff-Chu Sequence Generator.
 * 
 * Generates Zadoff-Chu sequences for synchronization and pilot symbols.
 * ZC sequences have constant amplitude and zero autocorrelation.
 */
class ZadoffChuGenerator {
public:
    /**
     * @brief Generate ZC sequence of given length and root.
     * 
     * @param output Output buffer (will be resized to length)
     * @param length Sequence length (typically FFT size)
     * @param root ZC sequence root index
     */
    static void generate(AlignedVector& output, size_t length, int root) {
        output.resize(length);
        
        const int N = static_cast<int>(length);
        const int q = root;
        
        // delta: even N -> 0, odd N -> 1
        const int delta = (N & 1);
        
        // Pre-calculate constant coefficient
        const double base = -M_PI * static_cast<double>(q) / static_cast<double>(N);
        
        #pragma omp simd simdlen(16)
        for (int n = 0; n < N; ++n) {
            const double nd = static_cast<double>(n);
            const double arg = nd * (nd + static_cast<double>(delta));
            const double phase = base * arg;
            output[n] = std::polar(1.0f, static_cast<float>(phase));
        }
    }
};


/**
 * @brief Hamming Window Generator.
 * 
 * Generates Hamming windows for range and Doppler processing.
 */
class WindowGenerator {
public:
    /**
     * @brief Generate Hamming window of given size.
     * Formula: w(n) = 0.54 - 0.46*cos(2πn/(N-1))
     */
    static void generate_hamming(AlignedFloatVector& output, size_t length) {
        output.resize(length);
        const float factor = 2.0f * static_cast<float>(M_PI) / (length - 1);
        
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < length; ++i) {
            output[i] = 0.54f - 0.46f * std::cos(factor * i);
        }
    }
};


/**
 * @brief Channel Estimator.
 * 
 * Provides various channel estimation methods for OFDM receivers.
 * LMMSE estimation requires instance with internal FFT plans.
 * LS estimation and equalization are static (no FFT needed).
 */
class ChannelEstimator {
public:
    /**
     * @brief Construct a ChannelEstimator with internal FFT plans for LMMSE.
     * @param fft_size FFT size for the plans
     */
    explicit ChannelEstimator(size_t fft_size)
        : _fft_size(fft_size),
          _scratch_buf1(fft_size),
          _scratch_buf2(fft_size),
          _H_est_internal(fft_size)
    {
        _fft_plan = fftwf_plan_dft_1d(
            static_cast<int>(fft_size),
            reinterpret_cast<fftwf_complex*>(_scratch_buf1.data()),
            reinterpret_cast<fftwf_complex*>(_scratch_buf2.data()),
            FFTW_FORWARD, FFTW_MEASURE);
        
        _ifft_plan = fftwf_plan_dft_1d(
            static_cast<int>(fft_size),
            reinterpret_cast<fftwf_complex*>(_scratch_buf1.data()),
            reinterpret_cast<fftwf_complex*>(_scratch_buf2.data()),
            FFTW_BACKWARD, FFTW_MEASURE);
    }

    ~ChannelEstimator() {
        if (_fft_plan) fftwf_destroy_plan(_fft_plan);
        if (_ifft_plan) fftwf_destroy_plan(_ifft_plan);
    }

    // Non-copyable due to FFTW plans
    ChannelEstimator(const ChannelEstimator&) = delete;
    ChannelEstimator& operator=(const ChannelEstimator&) = delete;

    // Move constructible
    ChannelEstimator(ChannelEstimator&& other) noexcept
        : _fft_size(other._fft_size),
          _fft_plan(other._fft_plan),
          _ifft_plan(other._ifft_plan),
          _scratch_buf1(std::move(other._scratch_buf1)),
          _scratch_buf2(std::move(other._scratch_buf2)),
          _H_est_internal(std::move(other._H_est_internal))
    {
        other._fft_plan = nullptr;
        other._ifft_plan = nullptr;
    }

    /**
     * @brief Estimate SNR from time-domain channel impulse response.
     * Assumes channel energy is concentrated within CP length, and the rest is noise.
     * 
     * @param h_time Time-domain channel estimate (impulse response)
     * @param cp_length Cyclic prefix length (assumed delay spread boundary)
     * @return Estimated SNR (linear scale)
     */
    static float estimate_snr_from_impulse_response(
        const AlignedVector& h_time,
        size_t cp_length
    ) {
        const size_t fft_size = h_time.size();
        if (fft_size <= cp_length) return 10.0f; // Fallback

        double signal_energy = 0.0;
        double noise_energy = 0.0;

        #pragma omp simd simdlen(16) reduction(+:signal_energy)
        for (size_t i = 0; i < cp_length; ++i) {
            signal_energy += std::norm(h_time[i]);
        }
        
        #pragma omp simd simdlen(16) reduction(+:noise_energy)
        for (size_t i = cp_length; i < fft_size; ++i) {
            noise_energy += std::norm(h_time[i]);
        }

        double noise_power = noise_energy / (fft_size - cp_length);
        double signal_power = (signal_energy / cp_length) - noise_power;

        if (signal_power < 0.0) signal_power = 0.0;
        if (noise_power < 1e-10) noise_power = 1e-10;

        return static_cast<float>(signal_power / noise_power);
    }

    /**
     * @brief Estimate channel using LMMSE (DFT-based with Wiener smoothing).
     * Algorithm: H_ls -> IFFT -> Window/Denoise -> FFT -> H_lmmse
     * SNR is estimated dynamically from the Time-Domain Channel Impulse Response.
     * 
     * @param rx_symbol Received frequency domain symbol
     * @param tx_zc Known transmitted ZC sequence
     * @param H_est Output channel estimate (will be resized)
     * @param cp_length Cyclic prefix length (assumed max delay spread)
     */
    void estimate_from_sync_lmmse(
        const AlignedVector& rx_symbol,
        const AlignedVector& tx_zc,
        AlignedVector& H_est,
        size_t cp_length
    ) {
        H_est.resize(_fft_size);

        // 1. LS Estimation into scratch_buf1
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _fft_size; ++i) {
            float rx_real = rx_symbol[i].real();
            float rx_imag = rx_symbol[i].imag();
            float tx_real = tx_zc[i].real();
            float tx_imag = tx_zc[i].imag();
            float denom = tx_real * tx_real + tx_imag * tx_imag;
            float inv_denom = 1.0f / denom;
            _scratch_buf1[i] = std::complex<float>(
                (rx_real * tx_real + rx_imag * tx_imag) * inv_denom,
                (rx_imag * tx_real - rx_real * tx_imag) * inv_denom
            );
        }

        // 2. IFFT: scratch_buf1 (H_ls) -> scratch_buf2 (h_ls)
        fftwf_execute_dft(_ifft_plan, 
            reinterpret_cast<fftwf_complex*>(_scratch_buf1.data()), 
            reinterpret_cast<fftwf_complex*>(_scratch_buf2.data()));

        // 3. SNR Estimation & Wiener Smoothing
        float n_float = static_cast<float>(_fft_size);
        // Normalize FFTW output (1/N) for SNR estimation
        float scale = 1.0f / n_float;
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _fft_size; ++i) {
            _scratch_buf2[i] *= scale;
        }

        float snr_est = estimate_snr_from_impulse_response(_scratch_buf2, cp_length);
        if (snr_est < 1e-4f) snr_est = 1e-4f;

        // Wiener coefficient: w = SNR / (SNR + 1)
        float w_pass = snr_est / (snr_est + 1.0f);
        
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _fft_size; ++i) {
            if (i < cp_length) {
                _scratch_buf2[i] *= w_pass;
            } else {
                _scratch_buf2[i] = {0.0f, 0.0f};
            }
        }

        // 4. FFT: scratch_buf2 (h_denoised) -> H_est (H_lmmse)
        fftwf_execute_dft(_fft_plan, 
            reinterpret_cast<fftwf_complex*>(_scratch_buf2.data()), 
            reinterpret_cast<fftwf_complex*>(H_est.data()));
    }

    /**
     * @brief Estimate channel from sync symbol using conjugate multiplication: H = Rx * conj(Tx)
     * Optimized for unit-magnitude sync sequences (e.g., ZC sequences where |Tx| = 1).
     * 
     * @param rx_symbol Received frequency domain symbol
     * @param tx_zc Known transmitted ZC sequence (unit magnitude)
     * @param H_est Output channel estimate (will be resized)
     */
    static void estimate_from_sync_ls(
        const AlignedVector& rx_symbol,
        const AlignedVector& tx_zc,
        AlignedVector& H_est
    ) {
        const size_t fft_size = rx_symbol.size();
        H_est.resize(fft_size);
        
        const auto* __restrict__ rx_ptr = rx_symbol.data();
        const auto* __restrict__ tx_ptr = tx_zc.data();
        auto* __restrict__ h_ptr = H_est.data();
        
        #pragma omp simd simdlen(16) aligned(rx_ptr, tx_ptr, h_ptr: 64)
        for (size_t i = 0; i < fft_size; ++i) {
            float rx_real = rx_ptr[i].real();
            float rx_imag = rx_ptr[i].imag();
            float tx_real = tx_ptr[i].real();
            float tx_imag = tx_ptr[i].imag();
            
            // Multiply by conjugate: (a+bi)*(c-di) = (ac+bd) + (bc-ad)i
            h_ptr[i] = std::complex<float>(
                rx_real * tx_real + rx_imag * tx_imag,
                rx_imag * tx_real - rx_real * tx_imag
            );
        }
    }

    /**
     * @brief Compute ZF equalizer inverse: H_inv = conj(H) / |H|^2 = 1/H
     */
    static void compute_zf_inverse(
        const AlignedVector& H_est,
        AlignedVector& H_inv
    ) {
        const size_t fft_size = H_est.size();
        H_inv.resize(fft_size);
        
        #pragma omp simd simdlen(16)
        for (size_t j = 0; j < fft_size; ++j) {
            const float h_real = H_est[j].real();
            const float h_imag = H_est[j].imag();
            const float mag_sq = h_real * h_real + h_imag * h_imag;
            const float inv_mag_sq = 1.0f / mag_sq;
            H_inv[j] = std::complex<float>(h_real * inv_mag_sq, -h_imag * inv_mag_sq);
        }
    }

    /**
     * @brief Equalize symbol with channel inverse and phase compensation.
     * symbol = symbol * H_inv * exp(-j*phase)
     */
    static void equalize_symbol(
        AlignedVector& symbol,
        const AlignedVector& H_inv,
        float phase_diff_CFO,
        float beta_rel,
        const std::vector<int>& subcarrier_indices
    ) {
        const size_t fft_size = symbol.size();
        
        #pragma omp simd simdlen(16)
        for (size_t j = 0; j < fft_size; ++j) {
            const float phase = beta_rel * subcarrier_indices[j] + phase_diff_CFO;
            
            // First multiply by H_inv (complex multiplication)
            const float hinv_real = H_inv[j].real();
            const float hinv_imag = H_inv[j].imag();
            const float sym_real = symbol[j].real();
            const float sym_imag = symbol[j].imag();
            const float eq_re = sym_real * hinv_real - sym_imag * hinv_imag;
            const float eq_im = sym_real * hinv_imag + sym_imag * hinv_real;
            
            // Then rotate by -phase (derotate)
            const float c = std::cos(phase);
            const float s = std::sin(phase);
            symbol[j] = std::complex<float>(eq_re * c + eq_im * s, eq_im * c - eq_re * s);
        }
    }

private:
    size_t _fft_size;
    fftwf_plan _fft_plan = nullptr;
    fftwf_plan _ifft_plan = nullptr;
    AlignedVector _scratch_buf1;
    AlignedVector _scratch_buf2;
    AlignedVector _H_est_internal;
};


/**
 * @brief Synchronization Processor.
 * 
 * Provides synchronization correlation and CFO estimation.
 * Maintains pre-allocated FFT plans and buffers for efficient reuse.
 */
class SyncProcessor {
public:
    /**
     * @brief Construct a SyncProcessor with pre-allocated FFT plans and sync sequence.
     * @param data_len Expected data length (2 frames worth)
     * @param fft_size FFT size
     * @param cp_length Cyclic prefix length
     * @param zc_freq Pre-generated ZC sequence in frequency domain
     */
    explicit SyncProcessor(size_t data_len, size_t fft_size, size_t cp_length, const AlignedVector& zc_freq)
        : _symbol_len(fft_size + cp_length),
          _data_len(data_len),
          _fft_size(fft_size),
          _cp_length(cp_length)
    {
        // Calculate FFT size (next power of 2 for linear correlation)
        _fft_len = 1;
        const size_t min_len = data_len + _symbol_len - 1;
        while (_fft_len < min_len) _fft_len <<= 1;
        
        // Allocate buffers
        _x_padded.resize(_fft_len, {0.0f, 0.0f});
        _h_padded.resize(_fft_len, {0.0f, 0.0f});
        _X.resize(_fft_len);
        _H.resize(_fft_len);
        _corr_result.resize(_fft_len);
        
        // Generate sync sequence and prepare _h_padded using provided zc_freq
        prepare_sync_sequence(zc_freq);
        
        // Create FFT plans (use FFTW_MEASURE for optimal performance)
        _fft_x = fftwf_plan_dft_1d(
            static_cast<int>(_fft_len),
            reinterpret_cast<fftwf_complex*>(_x_padded.data()),
            reinterpret_cast<fftwf_complex*>(_X.data()),
            FFTW_FORWARD, FFTW_MEASURE);
        
        _fft_h = fftwf_plan_dft_1d(
            static_cast<int>(_fft_len),
            reinterpret_cast<fftwf_complex*>(_h_padded.data()),
            reinterpret_cast<fftwf_complex*>(_H.data()),
            FFTW_FORWARD, FFTW_MEASURE);
        
        _ifft_corr = fftwf_plan_dft_1d(
            static_cast<int>(_fft_len),
            reinterpret_cast<fftwf_complex*>(_X.data()),
            reinterpret_cast<fftwf_complex*>(_corr_result.data()),
            FFTW_BACKWARD, FFTW_MEASURE);
        
        // Pre-compute FFT of _h_padded
        fftwf_execute(_fft_h);
    }

    ~SyncProcessor() {
        if (_fft_x) fftwf_destroy_plan(_fft_x);
        if (_fft_h) fftwf_destroy_plan(_fft_h);
        if (_ifft_corr) fftwf_destroy_plan(_ifft_corr);
    }

    // Non-copyable due to FFTW plans
    SyncProcessor(const SyncProcessor&) = delete;
    SyncProcessor& operator=(const SyncProcessor&) = delete;

    // Move constructible
    SyncProcessor(SyncProcessor&& other) noexcept
        : _symbol_len(other._symbol_len),
          _data_len(other._data_len),
          _fft_len(other._fft_len),
          _fft_size(other._fft_size),
          _cp_length(other._cp_length),
          _fft_x(other._fft_x),
          _fft_h(other._fft_h),
          _ifft_corr(other._ifft_corr),
          _x_padded(std::move(other._x_padded)),
          _h_padded(std::move(other._h_padded)),
          _X(std::move(other._X)),
          _H(std::move(other._H)),
          _corr_result(std::move(other._corr_result))
    {
        other._fft_x = nullptr;
        other._fft_h = nullptr;
        other._ifft_corr = nullptr;
    }

    /**
     * @brief FFT-based sliding window correlation for sync detection.
     * Uses pre-allocated FFT plans for fast correlation computation.
     * 
     * @param sync_data Received data (2 frames worth)
     * @param max_pos Output: position of maximum correlation
     * @param max_corr Output: maximum correlation value
     * @param avg_corr Output: average correlation value
     */
    void find_sync_position(
        const AlignedVector& sync_data,
        int& max_pos,
        float& max_corr,
        float& avg_corr
    ) {
        const size_t n_windows = sync_data.size() - _symbol_len + 1;
        
        // Clear x_padded buffer
        std::fill(_x_padded.begin(), _x_padded.end(), std::complex<float>(0.0f, 0.0f));
        
        // Copy received data
        std::copy(sync_data.begin(), sync_data.end(), _x_padded.begin());
        
        // Execute forward FFT on received data (_H is pre-computed)
        fftwf_execute(_fft_x);
        
        // Frequency domain multiplication: X = X .* H
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _fft_len; ++i) {
            _X[i] *= _H[i];
        }
        
        // Execute inverse FFT
        fftwf_execute(_ifft_corr);
        
        // Normalize IFFT output (FFTW doesn't do this automatically)
        const float norm_factor = 1.0f / static_cast<float>(_fft_len);
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _fft_len; ++i) {
            _corr_result[i] *= norm_factor;
        }
        
        // Find maximum correlation in valid range
        // Note: convolution output index needs adjustment for correlation
        // corr[n] corresponds to _corr_result[n + _symbol_len - 1]
        max_corr = 0.0f;
        avg_corr = 0.0f;
        max_pos = 0;
        
        for (size_t i = 0; i < n_windows; ++i) {
            const float corr = std::norm(_corr_result[i + _symbol_len - 1]);
            
            if (corr > max_corr) {
                max_corr = corr;
                max_pos = static_cast<int>(i);
            }
            avg_corr += corr;
        }
        avg_corr /= n_windows;
    }

    /**
     * @brief Estimate coarse CFO using CP correlation.
     * Correlates CP samples with corresponding tail samples.
     * 
     * @param data Input time-domain samples
     * @param start_pos Starting position in the buffer
     * @param available_symbols Number of symbols to process
     * @param symbol_len Symbol length (fft_size + cp_length)
     * @param cp_length Cyclic prefix length
     * @param fft_size FFT size
     * @return Phase difference (in radians)
     */
    static double estimate_cfo_phase(
        const AlignedVector& data,
        size_t start_pos,
        size_t available_symbols,
        size_t symbol_len,
        size_t cp_length,
        size_t fft_size
    ) {
        double total_real = 0.0;
        double total_imag = 0.0;
        
        for (size_t sym = 0; sym < available_symbols; ++sym) {
            const size_t pos = start_pos + sym * symbol_len;
            
            double sym_real = 0.0;
            double sym_imag = 0.0;
            
            #pragma omp simd reduction(+:sym_real, sym_imag)
            for (size_t i = 0; i < cp_length; ++i) {
                const auto& cp_sample = data[pos + i];
                const auto& tail_sample = data[pos + i + fft_size];
                
                // conj(cp_sample) * tail_sample
                sym_real += cp_sample.real() * tail_sample.real() + 
                            cp_sample.imag() * tail_sample.imag();
                sym_imag += cp_sample.real() * tail_sample.imag() - 
                            cp_sample.imag() * tail_sample.real();
            }
            
            total_real += sym_real;
            total_imag += sym_imag;
        }
        
        return std::atan2(total_imag, total_real);
    }

    /**
     * @brief Convert phase difference to CFO in Hz.
     */
    static double phase_to_cfo(double phase_diff, double sample_rate, size_t fft_size) {
        const double T_symbol = static_cast<double>(fft_size) / sample_rate;
        return phase_diff / (2.0 * M_PI * T_symbol);
    }

private:
    /**
     * @brief Generate sync sequence and prepare _h_padded for correlation.
     * Uses provided ZC sequence in frequency domain.
     */
    void prepare_sync_sequence(AlignedVector zc_freq) {
        const int N = static_cast<int>(_fft_size);
        
        // Execute IFFT to get time-domain sync symbol
        AlignedVector ifft_out(N);
        fftwf_plan plan = fftwf_plan_dft_1d(
            N,
            reinterpret_cast<fftwf_complex*>(zc_freq.data()),
            reinterpret_cast<fftwf_complex*>(ifft_out.data()),
            FFTW_BACKWARD, FFTW_ESTIMATE
        );
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        
        // Add cyclic prefix to create tx_sync_symbol
        AlignedVector tx_sync_symbol(_symbol_len);
        if (_cp_length > 0) {
            std::copy(ifft_out.end() - _cp_length, ifft_out.end(), tx_sync_symbol.begin());
        }
        std::copy(ifft_out.begin(), ifft_out.end(), tx_sync_symbol.begin() + _cp_length);
        
        // Prepare reversed and conjugated sync sequence in _h_padded
        // corr[i] = sum_j rx[i+j]*conj(sync[j]) = conv(rx, conj(sync_reversed))
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < _symbol_len; ++i) {
            _h_padded[i] = std::conj(tx_sync_symbol[_symbol_len - 1 - i]);
        }
    }

    size_t _symbol_len;
    size_t _data_len;
    size_t _fft_len;
    size_t _fft_size;
    size_t _cp_length;
    
    // FFT plans
    fftwf_plan _fft_x = nullptr;
    fftwf_plan _fft_h = nullptr;
    fftwf_plan _ifft_corr = nullptr;
    
    // Work buffers
    AlignedVector _x_padded;
    AlignedVector _h_padded;  // Pre-computed: reversed + conjugated sync sequence
    AlignedVector _X;
    AlignedVector _H;         // Pre-computed: FFT of _h_padded
    AlignedVector _corr_result;
};


/**
 * @brief QPSK LLR Processor.
 * 
 * Provides noise variance estimation and LLR calculation for QPSK modulation.
 */
class QPSK_LLR {
public:
    /**
     * @brief Estimate noise variance from pilot errors.
     * 
     * @param symbols Equalized symbols
     * @param pilot_positions Indices of pilot subcarriers
     * @param zc_freq Known ZC sequence in frequency domain
     * @return Estimated noise variance
     */
    static double estimate_noise_variance(
        const std::vector<AlignedVector>& symbols,
        const std::vector<size_t>& pilot_positions,
        const AlignedVector& zc_freq
    ) {
        double err_power_acc = 0.0;
        size_t err_count = 0;
        
        for (const auto& sym : symbols) {
            for (auto p : pilot_positions) {
                if (p < sym.size()) {
                    std::complex<float> y_eq = sym[p];
                    std::complex<float> x_ref = zc_freq[p];
                    auto e = y_eq - x_ref;
                    err_power_acc += std::norm(e);
                    err_count++;
                }
            }
        }
        
        if (err_count > 8) {
            double noise_var = err_power_acc / err_count;
            return std::max(noise_var, 1e-6);
        }
        return 0.5;  // Default value
    }

    /**
     * @brief Calculate LLR scaling factor from noise variance.
     * For QPSK Gray mapping: LLR_scale = 4 / noise_var
     */
    static double compute_llr_scale(double noise_var) {
        double sigma2_dim = noise_var / 2.0;
        return 2.0 / sigma2_dim;  // = 4 / noise_var
    }

    /**
     * @brief Compute LLR values for QPSK symbols.
     */
    static void compute_llr(
        const std::vector<AlignedVector>& symbols,
        const std::vector<size_t>& data_indices,
        float llr_scale,
        AlignedFloatVector& llr_output
    ) {
        const size_t num_data_sc = data_indices.size();
        const size_t total_llr = symbols.size() * num_data_sc * 2;
        llr_output.resize(total_llr);
        
        float* __restrict__ llr_ptr = llr_output.data();
        
        for (size_t sym_idx = 0; sym_idx < symbols.size(); ++sym_idx) {
            const auto* __restrict__ sym_ptr = symbols[sym_idx].data();
            float* __restrict__ out_ptr = llr_ptr + sym_idx * num_data_sc * 2;
            
            #pragma omp simd simdlen(16)
            for (size_t i = 0; i < num_data_sc; ++i) {
                const size_t k = data_indices[i];
                out_ptr[i * 2]     = sym_ptr[k].real() * llr_scale;
                out_ptr[i * 2 + 1] = sym_ptr[k].imag() * llr_scale;
            }
        }
    }

    /**
     * @brief Compute LLR values directly to pre-allocated output buffer.
     */
    static void compute_llr_to_buffer(
        const std::vector<AlignedVector>& symbols,
        const std::vector<size_t>& data_indices,
        float llr_scale,
        float* __restrict__ llr_output
    ) {
        const size_t num_data_sc = data_indices.size();
        
        for (size_t sym_idx = 0; sym_idx < symbols.size(); ++sym_idx) {
            const auto* __restrict__ sym_ptr = symbols[sym_idx].data();
            float* __restrict__ out_ptr = llr_output + sym_idx * num_data_sc * 2;
            
            #pragma omp simd simdlen(16)
            for (size_t i = 0; i < num_data_sc; ++i) {
                const size_t k = data_indices[i];
                out_ptr[i * 2]     = sym_ptr[k].real() * llr_scale;
                out_ptr[i * 2 + 1] = sym_ptr[k].imag() * llr_scale;
            }
        }
    }
};


/**
 * @brief Frequency Offset Estimator.
 * 
 * Estimates CFO and SFO from pilot phase differences.
 */
class FrequencyOffsetEstimator {
public:
    /**
     * @brief Estimate phase difference between consecutive symbols at each pilot.
     * Returns (beta, alpha) where:
     * - beta: phase slope vs subcarrier (related to SFO)
     * - alpha: constant phase offset (related to CFO)
     * 
     * @param symbols Equalized frequency domain symbols
     * @param pilot_positions Indices of pilot subcarriers
     * @param fft_size FFT size
     * @param sync_pos Sync symbol position (to skip)
     * @param pilot_indices Output: actual frequency indices of pilots
     * @param avg_phase_diff Output: average phase difference at each pilot
     * @param weights Output: weights for regression
     */
    static void compute_pilot_phase_diff(
        const std::vector<AlignedVector>& symbols,
        const std::vector<size_t>& pilot_positions,
        size_t fft_size,
        size_t sync_pos,
        std::vector<int>& pilot_indices,
        std::vector<float>& avg_phase_diff,
        std::vector<float>& weights
    ) {
        const size_t num_pilots = pilot_positions.size();
        pilot_indices.resize(num_pilots);
        avg_phase_diff.resize(num_pilots);
        weights.resize(num_pilots, 0.0f);
        
        for (size_t j = 0; j < num_pilots; ++j) {
            auto pilot_index = pilot_positions[j];
            std::complex<double> next_current_sum(0.0, 0.0);
            
            #pragma omp simd
            for (size_t i = sync_pos; i < symbols.size() - 1; ++i) {
                std::complex<float> current_pilot = symbols[i][pilot_index];
                std::complex<float> next_pilot = symbols[i+1][pilot_index];
                next_current_sum += std::conj(current_pilot) * next_pilot;
            }
            
            avg_phase_diff[j] = static_cast<float>(std::arg(next_current_sum));
            
            // Convert to actual frequency index (-fft_size/2 to fft_size/2-1)
            int freq_index = static_cast<int>(pilot_index);
            if (freq_index >= static_cast<int>(fft_size)/2) {
                freq_index -= static_cast<int>(fft_size);
            }
            pilot_indices[j] = freq_index;
            weights[j] = static_cast<float>(std::norm(next_current_sum));
        }
    }

    /**
     * @brief Convert alpha (phase per symbol) to CFO in Hz.
     */
    static float alpha_to_cfo(float alpha, size_t fft_size, size_t cp_length, double sample_rate) {
        const float T = static_cast<float>(fft_size + cp_length) / static_cast<float>(sample_rate);
        return alpha / (2.0f * static_cast<float>(M_PI) * T);
    }
};


/**
 * @brief Delay Spectrum Processor.
 * 
 * Computes delay (time-domain) spectrum from channel estimate.
 * Manages its own IFFT plan and buffers internally.
 */
class DelayProcessor {
public:
    /**
     * @brief Construct a DelayProcessor with internal IFFT plan.
     * @param fft_size FFT size for the plan
     */
    explicit DelayProcessor(size_t fft_size)
        : _fft_size(fft_size),
          _ifft_in(fft_size),
          _ifft_out(fft_size)
    {
        _ifft_plan = fftwf_plan_dft_1d(
            static_cast<int>(fft_size),
            reinterpret_cast<fftwf_complex*>(_ifft_in.data()),
            reinterpret_cast<fftwf_complex*>(_ifft_out.data()),
            FFTW_BACKWARD, FFTW_MEASURE);
    }

    ~DelayProcessor() {
        if (_ifft_plan) fftwf_destroy_plan(_ifft_plan);
    }

    // Non-copyable due to FFTW plans
    DelayProcessor(const DelayProcessor&) = delete;
    DelayProcessor& operator=(const DelayProcessor&) = delete;

    // Move constructible
    DelayProcessor(DelayProcessor&& other) noexcept
        : _fft_size(other._fft_size),
          _ifft_plan(other._ifft_plan),
          _ifft_in(std::move(other._ifft_in)),
          _ifft_out(std::move(other._ifft_out))
    {
        other._ifft_plan = nullptr;
    }

    /**
     * @brief Compute delay spectrum from channel estimate.
     * Performs FFT shift + IFFT + scaling.
     * 
     * @param H_est Channel estimate (frequency domain)
     * @param delay_spectrum Output delay spectrum
     */
    void compute_delay_spectrum(
        const AlignedVector& H_est,
        AlignedVector& delay_spectrum
    ) {
        // FFT shift (swap halves)
        const size_t half = _fft_size / 2;
        
        #pragma omp simd
        for (size_t i = 0; i < half; ++i) {
            _ifft_in[i] = H_est[i + half];
            _ifft_in[i + half] = H_est[i];
        }
        
        // Execute IFFT
        fftwf_execute(_ifft_plan);
        
        // Scale and copy to output
        const float scale = 1.0f / std::sqrt(static_cast<float>(_fft_size));
        delay_spectrum.resize(_fft_size);
        
        #pragma omp simd
        for (size_t i = 0; i < _fft_size; ++i) {
            delay_spectrum[i] = _ifft_out[i] * scale;
        }
    }

    /**
     * @brief Find peak in delay spectrum and compute statistics.
     * 
     * @param delay_spectrum Input delay spectrum
     * @param max_index Output: index of maximum magnitude
     * @param max_mag Output: maximum magnitude value
     * @param avg_mag Output: average magnitude
     * @param cp_length Cyclic prefix length (search range = CP length on both sides)
     */
    static void find_peak(
        const AlignedVector& delay_spectrum,
        size_t& max_index,
        float& max_mag,
        float& avg_mag,
        size_t cp_length = 0
    ) {
        const size_t fft_size = delay_spectrum.size();
        max_index = 0;
        max_mag = 0.0f;
        avg_mag = 0.0f;
        
        // If cp_length is specified, search only within CP range
        // Range: [0, cp_length) and [fft_size - cp_length, fft_size)
        if (cp_length > 0 && cp_length < fft_size) {
            size_t count = 0;
            
            // Search in [0, cp_length)
            for (size_t i = 0; i < cp_length; ++i) {
                const float mag = std::abs(delay_spectrum[i]);
                if (mag > max_mag) {
                    max_mag = mag;
                    max_index = i;
                }
                avg_mag += mag;
                count++;
            }
            avg_mag /= static_cast<float>(count);
        } else {
            // Original behavior: search entire range
            for (size_t i = 0; i < fft_size; ++i) {
                const float mag = std::abs(delay_spectrum[i]);
                if (mag > max_mag) {
                    max_mag = mag;
                    max_index = i;
                }
                avg_mag += mag;
            }
            avg_mag /= static_cast<float>(fft_size);
        }
    }

    /**
     * @brief Adjust delay index to signed value.
     * Maps indices >= fft_size/2 to negative values.
     */
    static int adjust_delay_index(size_t max_index, size_t fft_size) {
        int adjusted = static_cast<int>(max_index);
        const int half_fft = static_cast<int>(fft_size) / 2;
        if (adjusted >= half_fft) {
            adjusted -= static_cast<int>(fft_size);
        }
        return adjusted;
    }

    /**
     * @brief Quinn's Algorithm for Fractional Delay Estimation.
     * Refines the peak position in the delay spectrum to sub-sample precision.
     * 
     * @param spectrum Delay spectrum (complex values)
     * @param max_index Index of the maximum magnitude
     * @return Fractional delay estimate (-0.5 to 0.5 samples)
     */
    static float estimate_fractional_delay(const AlignedVector& spectrum, size_t max_index) {
        const size_t N = spectrum.size();
        if (N < 3) return 0.0f;
        
        const auto& d0 = spectrum[max_index];
        const auto& d_prev = spectrum[(max_index == 0) ? (N - 1) : (max_index - 1)];
        const auto& d_next = spectrum[(max_index == N - 1) ? 0 : (max_index + 1)];
        const auto magnitude = std::abs(d0);
        
        constexpr float EPSILON = 1e-10f;
        if (magnitude < EPSILON) {
            return 0.0f;
        }
        
        float alpha1 = 0.0f, alpha2 = 0.0f;
        {
            const float denom = std::real(std::conj(d0) * d0);
            const float num1 = std::real(std::conj(d_prev) * d0);
            const float num2 = std::real(std::conj(d_next) * d0);
            
            if (denom > EPSILON) {
                alpha1 = num1 / denom;
                alpha2 = num2 / denom;
            } else {
                alpha1 = std::abs(d_prev) / (magnitude + EPSILON);
                alpha2 = std::abs(d_next) / (magnitude + EPSILON);
            }
            
            // Limit ratio range
            alpha1 = std::max(-0.9999f, std::min(0.9999f, alpha1));
            alpha2 = std::max(-0.9999f, std::min(0.9999f, alpha2));
        }
        
        const float delta1 = alpha1 / (1.0f - alpha1);
        const float delta2 = -alpha2 / (1.0f - alpha2);
        
        // Use global isNaN from Common.hpp
        if (isNaN(delta1) || isNaN(delta2)) {
            return 0.0f;
        }
        
        const float abs1 = std::abs(delta1);
        const float abs2 = std::abs(delta2);
        
        if (abs1 > 2.0f && abs2 > 2.0f) {
            return 0.5f;
        } else if (abs1 > 2.0f) {
            return delta2;
        } else if (abs2 > 2.0f) {
            return delta1;
        } else {
            return (delta1 > 0.0f && delta2 > 0.0f) ? delta2 : delta1;
        }
    }

private:
    size_t _fft_size;
    fftwf_plan _ifft_plan = nullptr;
    AlignedVector _ifft_in;
    AlignedVector _ifft_out;
};

/**
 * @brief Moving Target Indication (MTI) Filter.
 * 
 * Implements an 8-stage IIR filter for clutter suppression in sensing applications.
 * Uses AVX/OpenMP SIMD for efficient processing of subcarriers.
 */
class MTIFilter {
public:
    MTIFilter(size_t range_fft_size = 1024) {
        resize(range_fft_size);
    }

    void resize(size_t range_fft_size) {
        _range_fft_size = range_fft_size;
        _state.resize(8 * 2 * _range_fft_size, std::complex<float>(0.0f, 0.0f));
        reset();
    }

    void reset() {
        std::fill(_state.begin(), _state.end(), std::complex<float>(0.0f, 0.0f));
    }

    /**
     * @brief Apply MTI filter to the buffer.
     * 
     * @param buffer Input/Output buffer (Channel Response)
     * @param N_proc Number of start subcarriers to process (e.g., fft_size)
     * @param num_symbols Number of symbols in the buffer
     */
    void apply(AlignedVector& buffer, size_t N_proc, size_t num_symbols) {
        static const float SOS[8][6] = {
            {0.993542f, -1.987084f, 0.993542f, 1.000000f, -1.993112f, 0.993951f},
            {0.981889f, -1.963778f, 0.981889f, 1.000000f, -1.981389f, 0.982224f},
            {0.971190f, -1.942380f, 0.971190f, 1.000000f, -1.970564f, 0.971395f},
            {0.961860f, -1.923721f, 0.961860f, 1.000000f, -1.961077f, 0.961903f},
            {0.954245f, -1.908491f, 0.954245f, 1.000000f, -1.953298f, 0.954121f},
            {0.948614f, -1.897228f, 0.948614f, 1.000000f, -1.947526f, 0.948346f},
            {0.945158f, -1.890316f, 0.945158f, 1.000000f, -1.943975f, 0.944795f},
            {0.971593f, -0.971593f, 0.000000f, 1.000000f, -0.971389f, 0.000000f}
        };

        const size_t N_alloc = _range_fft_size;

        for (int stage = 0; stage < 8; ++stage) {
            const float b0 = SOS[stage][0];
            const float b1 = SOS[stage][1];
            const float b2 = SOS[stage][2];
            const float a1 = SOS[stage][4]; 
            const float a2 = SOS[stage][5];

            std::complex<float>* s0_arr = &_state[stage * 2 * N_alloc]; 
            std::complex<float>* s1_arr = &_state[stage * 2 * N_alloc + N_alloc];

            for (size_t i = 0; i < num_symbols; ++i) {
                std::complex<float>* symbol_data = buffer.data() + i * N_alloc;

                #pragma omp simd
                for (size_t col = 0; col < N_proc; ++col) {
                    std::complex<float> x = symbol_data[col];
                    std::complex<float> s0 = s0_arr[col];
                    std::complex<float> s1 = s1_arr[col];

                    std::complex<float> y = x * b0 + s0;
                    std::complex<float> new_s0 = x * b1 - y * a1 + s1;
                    std::complex<float> new_s1 = x * b2 - y * a2;

                    symbol_data[col] = y;
                    s0_arr[col] = new_s0;
                    s1_arr[col] = new_s1;
                }
            }
        }
    }

private:
    AlignedVector _state;
    size_t _range_fft_size;
};


/**
 * @brief Core Sensing Processing Operations.
 * 
 */
class SensingProcessor {
public:
    struct Params {
        size_t fft_size;
        size_t range_fft_size;
        size_t doppler_fft_size;
        size_t sensing_symbol_num;
    };

private:
    Params _params;
    fftwf_plan _range_ifft_plan = nullptr;
    fftwf_plan _doppler_fft_plan = nullptr;
    AlignedVector _channel_buffer;
    MTIFilter _mti_filter;

public:
    explicit SensingProcessor(const Params& params)
        : _params(params),
          _channel_buffer(params.range_fft_size * params.doppler_fft_size, std::complex<float>(0.0f, 0.0f)),
          _mti_filter(params.range_fft_size)
    {
        // Create batch Range IFFT plan
        const int fft_size_int = static_cast<int>(params.range_fft_size);
        const int doppler_fft_size_int = static_cast<int>(params.doppler_fft_size);
        
        _range_ifft_plan = fftwf_plan_many_dft(
            1,                         // rank
            &fft_size_int,             // n (FFT size)
            doppler_fft_size_int,      // howmany (number of symbols)
            reinterpret_cast<fftwf_complex*>(_channel_buffer.data()),
            nullptr,                   // inembed (contiguous)
            1,                         // istride
            fft_size_int,              // idist (distance between FFTs)
            reinterpret_cast<fftwf_complex*>(_channel_buffer.data()),
            nullptr,                   // onembed
            1,                         // ostride
            fft_size_int,              // odist
            FFTW_BACKWARD,             // sign (IFFT)
            FFTW_MEASURE
        );
        
        // Create Doppler FFT plan
        _doppler_fft_plan = fftwf_plan_many_dft(
            1,                         // rank
            &doppler_fft_size_int,     // n (number of symbols)
            fft_size_int,              // howmany (number of subcarriers)
            reinterpret_cast<fftwf_complex*>(_channel_buffer.data()),
            nullptr,                   // inembed
            fft_size_int,              // istride (stride between symbols)
            1,                         // idist (distance between subcarriers)
            reinterpret_cast<fftwf_complex*>(_channel_buffer.data()),
            nullptr,                   // onembed
            fft_size_int,              // ostride
            1,                         // odist
            FFTW_FORWARD,              // sign (FFT)
            FFTW_MEASURE
        );
    }

    ~SensingProcessor() {
        if (_range_ifft_plan) fftwf_destroy_plan(_range_ifft_plan);
        if (_doppler_fft_plan) fftwf_destroy_plan(_doppler_fft_plan);
    }

    // Non-copyable due to FFTW plans
    SensingProcessor(const SensingProcessor&) = delete;
    SensingProcessor& operator=(const SensingProcessor&) = delete;

    // Move constructible
    SensingProcessor(SensingProcessor&& other) noexcept
        : _params(other._params),
          _range_ifft_plan(other._range_ifft_plan),
          _doppler_fft_plan(other._doppler_fft_plan),
          _channel_buffer(std::move(other._channel_buffer))
    {
        other._range_ifft_plan = nullptr;
        other._doppler_fft_plan = nullptr;
    }

    /**
     * @brief Get the internal channel buffer for external data input.
     */
    AlignedVector& channel_buffer() { return _channel_buffer; }
    const AlignedVector& channel_buffer() const { return _channel_buffer; }

    /**
     * @brief Execute Range IFFT on internal channel buffer.
     */
    void execute_range_ifft() {
        fftwf_execute(_range_ifft_plan);
    }

    /**
     * @brief Execute Doppler FFT on internal channel buffer.
     */
    void execute_doppler_fft() {
        fftwf_execute(_doppler_fft_plan);
    }

    /**
     * @brief Clear the channel buffer to zeros.
     */
    void clear_channel_buffer() {
        _channel_buffer.assign(_params.range_fft_size * _params.doppler_fft_size, 
                               std::complex<float>(0.0f, 0.0f));
    }

    /**
     * @brief Copy frequency-domain symbols directly to internal channel buffer.
     * Used by OFDMDemodulator where symbols are already in frequency domain.
     * 
     * @param rx_symbols Vector of frequency-domain symbols to copy
     */
    void copy_symbols_to_buffer(const std::vector<AlignedVector>& rx_symbols) {
        const size_t num_symbols = std::min(rx_symbols.size(), _params.sensing_symbol_num);
        for (size_t i = 0; i < num_symbols; ++i) {
            auto* dest = _channel_buffer.data() + i * _params.range_fft_size;
            std::copy(rx_symbols[i].begin(), rx_symbols[i].end(), dest);
        }
    }

    /**
     * @brief Copy a single FFT output to the internal channel buffer at specified index.
     * Used by OFDMModulator where FFT is executed externally per symbol.
     * 
     * @param symbol_idx Index of the symbol in the buffer
     * @param fft_output FFT output data to copy
     */
    void copy_fft_result_to_buffer(size_t symbol_idx, const AlignedVector& fft_output) {
        if (symbol_idx >= _params.doppler_fft_size) return;
        auto* dest = _channel_buffer.data() + symbol_idx * _params.range_fft_size;
        std::copy(fft_output.begin(), fft_output.end(), dest);
    }

    /**
     * @brief Initialize Hamming windows for range and Doppler processing.
     */
    void init_windows(AlignedFloatVector& range_window, AlignedFloatVector& doppler_window) {
        WindowGenerator::generate_hamming(range_window, _params.fft_size);
        WindowGenerator::generate_hamming(doppler_window, _params.sensing_symbol_num);
    }

    /**
     * @brief Apply MTI filter to the internal channel buffer.
     * 
     * @param enabled Whether to enable MTI filtering. If false, does nothing.
     */
    void apply_mti(bool enabled) {
        if (enabled) {
            _mti_filter.apply(_channel_buffer, _params.fft_size, _params.sensing_symbol_num);
        }
    }

    /**
     * @brief Channel estimation with conjugate multiplication and in-place FFT shift.
     * For mono-static sensing where TX symbols have unit magnitude.
     * Operates on internal channel buffer.
     */
    void channel_estimate_with_shift(const std::vector<AlignedVector>& tx_symbols) {
        const size_t fft_size = _params.fft_size;
        const size_t half_size = fft_size / 2;
        
        for (size_t i = 0; i < _params.sensing_symbol_num; ++i) {
            auto* __restrict__ ch_data = _channel_buffer.data() + i * _params.range_fft_size;
            const auto* __restrict__ tx_data = tx_symbols[i].data();
            
            // Combined: channel estimation with multiplication + FFT shift
            #pragma omp simd simdlen(16) aligned(ch_data, tx_data: 64)
            for (size_t k = 0; k < half_size; ++k) {
                // First half
                float ch_real = ch_data[k].real();
                float ch_imag = ch_data[k].imag();
                float tx_real = tx_data[k].real();
                float tx_imag = tx_data[k].imag();
                
                float est_real = ch_real * tx_real + ch_imag * tx_imag;
                float est_imag = ch_imag * tx_real - ch_real * tx_imag;
                
                // Second half
                float ch2_real = ch_data[k + half_size].real();
                float ch2_imag = ch_data[k + half_size].imag();
                float tx2_real = tx_data[k + half_size].real();
                float tx2_imag = tx_data[k + half_size].imag();
                
                // FFT shift: swap halves
                ch_data[k] = std::complex<float>(
                    ch2_real * tx2_real + ch2_imag * tx2_imag,
                    ch2_imag * tx2_real - ch2_real * tx2_imag
                );
                ch_data[k + half_size] = std::complex<float>(est_real, est_imag);
            }
        }
    }

    /**
     * @brief Channel estimation with division.
     * @deprecated Not used - remodulated QPSK symbols have unit magnitude, use channel_estimate_with_shift instead.
     */
    void channel_estimate_with_division(
        AlignedVector& channel_buffer,
        const std::vector<AlignedVector>& tx_symbols,
        size_t fft_size
    ) {
        for (size_t i = 0; i < _params.sensing_symbol_num; ++i) {
            auto* __restrict__ ch_data = channel_buffer.data() + i * _params.range_fft_size;
            const auto* __restrict__ tx_data = tx_symbols[i].data();
            
            #pragma omp simd simdlen(16) aligned(ch_data, tx_data: 64)
            for (size_t k = 0; k < fft_size; ++k) {
                float rx_real = ch_data[k].real();
                float rx_imag = ch_data[k].imag();
                float tx_real = tx_data[k].real();
                float tx_imag = tx_data[k].imag();
                float denom = tx_real * tx_real + tx_imag * tx_imag;
                float inv_denom = 1.0f / denom;
                ch_data[k] = std::complex<float>(
                    (rx_real * tx_real + rx_imag * tx_imag) * inv_denom,
                    (rx_imag * tx_real - rx_real * tx_imag) * inv_denom
                );
            }
        }
    }

    /**
     * @brief In-place FFT shift for each symbol in buffer.
     */
    void fft_shift_symbols(AlignedVector& buffer, size_t fft_size) {
        const size_t half_size = fft_size / 2;
        
        for (size_t i = 0; i < _params.sensing_symbol_num; ++i) {
            auto* symbol_data = buffer.data() + i * _params.range_fft_size;
            
            #pragma omp simd simdlen(16) aligned(symbol_data: 64)
            for (size_t j = 0; j < half_size; ++j) {
                std::complex<float> temp = symbol_data[j];
                symbol_data[j] = symbol_data[j + half_size];
                symbol_data[j + half_size] = temp;
            }
        }
    }

    /**
     * @brief Apply range and Doppler windows to channel buffer.
     */
    void apply_windows(
        AlignedVector& buffer,
        const AlignedFloatVector& range_window,
        const AlignedFloatVector& doppler_window
    ) {
        // Apply range window (per symbol)
        for (size_t i = 0; i < _params.sensing_symbol_num; ++i) {
            auto* symbol_data = buffer.data() + i * _params.range_fft_size;
            #pragma omp simd simdlen(16) aligned(symbol_data: 64)
            for (size_t j = 0; j < _params.fft_size; ++j) {
                symbol_data[j] *= range_window[j];
            }
        }

        // Apply Doppler window (across symbols for each bin)
        for (size_t bin = 0; bin < _params.fft_size; ++bin) {
            #pragma omp simd simdlen(16)
            for (size_t i = 0; i < _params.sensing_symbol_num; ++i) {
                size_t idx = i * _params.range_fft_size + bin;
                buffer[idx] *= doppler_window[i];
            }
        }
    }

    /**
     * @brief Compensate phase for sensing symbols (CFO/SFO/delay).
     */
    void compensate_phase(
        std::vector<AlignedVector>& rx_symbols,
        float CFO,
        float SFO,
        float delay_offset,
        const std::vector<float>& subcarrier_phases_unit_delay,
        const std::vector<float>& subcarrier_indices,
        size_t sync_pos
    ) {
        for (size_t symbol_idx = 0; symbol_idx < rx_symbols.size(); ++symbol_idx) {
            auto& rx_symbol = rx_symbols[symbol_idx];
            int relative_symbol_index = static_cast<int>(symbol_idx) - static_cast<int>(sync_pos);
            float phase_diff_CFO = CFO * relative_symbol_index;
            
            #pragma omp simd simdlen(16)
            for (size_t j = 0; j < rx_symbol.size(); ++j) {
                float phase_diff_SFO = SFO * subcarrier_indices[j] * relative_symbol_index;
                float phase_diff_delay = subcarrier_phases_unit_delay[j] * delay_offset;
                float phase_diff_total = phase_diff_delay + phase_diff_SFO + phase_diff_CFO;
                auto phase_diff = std::polar(1.0f, -phase_diff_total);
                rx_symbol[j] = rx_symbol[j] * phase_diff;
            }
        }
    }

    const Params& params() const { return _params; }
};


// ============== Pure Signal Processing Functions ==============

/**
 * @brief Weighted Linear Regression.
 * 
 * Calculates the slope (beta) and intercept (alpha) of a line that best fits the
 * weighted input data points using the least squares method. 
 * Used for estimating SFO/SIO and CFO effects where some pilot subcarriers might have higher SNR (weight).
 * 
 * @return std::pair<float, float> {slope (beta), intercept (alpha)}
 */
template <typename T>
std::pair<float, float> weightedlinearRegression(const std::vector<T>& x_values,
                                         const std::vector<float>& y_values,
                                         const std::vector<float>& weights) {
    if (x_values.size() != y_values.size() || 
        x_values.size() != weights.size() || 
        x_values.empty()) {
        return std::make_pair(0.0f, 0.0f);
    }
    float sum_w = 0.0f, sum_wx = 0.0f, sum_wy = 0.0f;
    float sum_wxx = 0.0f, sum_wxy = 0.0f;
    const int N = x_values.size();

    for (int i = 0; i < N; ++i) {
        const float w = weights[i];
        const float x = static_cast<float>(x_values[i]);
        const float y = y_values[i];
        sum_w += w; sum_wx += w * x; sum_wy += w * y;
        sum_wxx += w * x * x; sum_wxy += w * x * y;
    }

    float beta = 0.0f, alpha = 0.0f;
    float denom = sum_w * sum_wxx - sum_wx * sum_wx;
    if (std::abs(denom) > 1e-10f) {
        beta = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
        alpha = (sum_wy - beta * sum_wx) / sum_w;
    }
    return std::make_pair(beta, alpha);
}

/**
 * @brief Standard Linear Regression.
 * 
 * Calculates the slope (beta) and intercept (alpha) for unweighted data.
 * Used for estimating SFO/SIO across frames using timing offset estimates.
 * 
 * @return std::pair<float, float> {slope (beta), intercept (alpha)}
 */
template <typename T>
std::pair<float, float> linearRegression(const std::vector<T>& x_values,
                                         const std::vector<float>& y_values) {
    if (x_values.size() != y_values.size() || x_values.empty()) {
        return std::make_pair(0.0f, 0.0f);
    }
    float sum_x = 0.0f, sum_y = 0.0f, sum_xx = 0.0f, sum_xy = 0.0f;
    const int N = x_values.size();

    for (int i = 0; i < N; ++i) {
        const float x = static_cast<float>(x_values[i]);
        const float y = y_values[i];
        sum_x += x; sum_y += y; sum_xx += x * x; sum_xy += x * y;
    }

    float beta = 0.0f, alpha = 0.0f;
    float denom = N * sum_xx - sum_x * sum_x;
    if (std::abs(denom) > 1e-10f) {
        beta = (N * sum_xy - sum_x * sum_y) / denom;
        alpha = (sum_y - beta * sum_x) / N;
    }
    return std::make_pair(beta, alpha);
}

/**
 * @brief Finite Impulse Response (FIR) Filter.
 * 
 * Implements a standard FIR filter with a circular buffer for efficiency.
 */
class FIRFilter {
private:
    std::vector<float> coeffs;
    std::vector<float> buffer;
    size_t order;
    size_t index;

public:
    FIRFilter(const std::vector<float>& coefficients) 
        : coeffs(coefficients), 
          buffer(coefficients.size(), 0.0f),
          order(coefficients.size()),
          index(0) {}

    float process(float input) {
        buffer[index] = input;
        index = (index + 1) % order;

        float output = 0.0f;
        size_t i = index;
        for (size_t j = 0; j < order; j++) {
            output += coeffs[j] * buffer[i];
            i = (i + 1) % order;
        }
        return output;
    }
    
    void warm_up(float value, size_t samples = 50) {
        for (size_t i = 0; i < samples; i++) {
            process(value);
        }
    }
};

/**
 * @brief Phase Unwrapping Function.
 * 
 * Unwraps the phase values in a vector to eliminate 2*pi jumps.
 * Essential for accurate frequency offset estimation from phase differences.
 */
inline void unwrap(std::vector<float>& phase) {
    if (phase.size() > 1) {
        std::vector<float> diffs(phase.size());
        
        #pragma omp simd simdlen(16)
        for (size_t i = 1; i < phase.size(); ++i) {
            float d = phase[i] - phase[i - 1];
            float k = std::round(d / (2 * (float)M_PI));
            d -= k * 2 * (float)M_PI;
            diffs[i] = d;
        }

        for (size_t i = 1; i < phase.size(); ++i) {
            phase[i] = phase[i - 1] + diffs[i];
        }
    }
}

// ============== End of Pure Signal Processing Functions ==============

/**
 * @brief Sampling Frequency Offset (SFO) / Sampling Interval Offset (SIO) Estimator.
 * 
 * Estimates the SFO/SIO by tracking the drift of timing offsets over time.
 * Uses a estimation window of timing offset measurements and performs 
 * linear regression to determine the rate of change (SIO). Also 
 * incorporates a control loop to adjust synchronization.
 */
class SFOEstimator {
public:
    explicit SFOEstimator(size_t window_size) 
        : _window_size(window_size),
          _delay_offsets(window_size, 0.0f),
          _delay_offsets_indices(window_size) {
        std::iota(_delay_offsets_indices.begin(), _delay_offsets_indices.end(), 0);
        reset();
    }

    void reset() {
        _count = 0;
        _cumulative_delay_offset = 0.0f;
        _sfo_per_frame = 0.0f;
        std::fill(_delay_offsets.begin(), _delay_offsets.end(), 0.0f);
    }

    // Update delay offset estimation and calculate SFO/SIO
    void update(float delay_offset_reading, float Alignment) {
        if (!_first_delay_offset_reading) {
            if (Alignment != 0.0f) {
                _cumulative_delay_offset += Alignment;
            }
            _delay_offsets[_count] = delay_offset_reading + _cumulative_delay_offset;
            if (++_count >= _window_size) {
                _sfo_per_frame = linearRegression(_delay_offsets_indices, _delay_offsets).first;
                _count = 0;
                _cumulative_delay_offset = 0.0f;
            }
            if (std::abs(_sfo_per_frame) > 1.0f) {
                _sfo_per_frame = 0.0f;
            }
            _cumulative_sensing_delay_offset += _sfo_per_frame;
            _cumulative_sensing_delay_offset -= Alignment;
            auto err = delay_offset_reading - _cumulative_sensing_delay_offset;
            if (std::abs(err) > 0.1f) {
                _err_large_count++;
                if (_err_large_count > 100) {
                    _pd = 1e-2;
                }
            } else {
                _err_large_count = 0;
                _pd = 1e-5;
            }
            _cumulative_sensing_delay_offset += _pd * err;
        }
        if (_first_delay_offset_reading) {
            _count++;
            if (_count >= 10) {
                _first_delay_offset_reading = false;
                _count = 0;
            }
        }
    }

    float get_sfo_per_frame() const { return _sfo_per_frame; }
    float get_sensing_delay_offset() const { return _cumulative_sensing_delay_offset; }

private:
    size_t _window_size;
    size_t _count = 0;
    size_t _err_large_count = 0;
    float _pd = 1e-5;
    
    std::vector<float> _delay_offsets;
    std::vector<int> _delay_offsets_indices;
    
    float _cumulative_delay_offset = 0.0f;
    bool _first_delay_offset_reading = true;
    float _sfo_per_frame = 0.0f;
    float _cumulative_sensing_delay_offset = 0.0f;
};




#endif // OFDM_CORE_HPP
