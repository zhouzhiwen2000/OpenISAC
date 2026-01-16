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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif


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
        #pragma omp simd
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
        #pragma omp simd
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
        #pragma omp simd
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
        
        #pragma omp simd
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
        
        #pragma omp simd
        for (size_t i = 0; i < length; ++i) {
            output[i] = 0.54f - 0.46f * std::cos(factor * i);
        }
    }
};


/**
 * @brief Channel Estimator.
 * 
 * Provides various channel estimation methods for OFDM receivers.
 */
class ChannelEstimator {
public:
    /**
     * @brief Estimate channel from sync symbol using division: H = Rx / Tx
     * 
     * @param rx_symbol Received frequency domain symbol
     * @param tx_zc Known transmitted ZC sequence
     * @param H_est Output channel estimate (will be resized)
     */
    static void estimate_from_sync(
        const AlignedVector& rx_symbol,
        const AlignedVector& tx_zc,
        AlignedVector& H_est
    ) {
        const size_t fft_size = rx_symbol.size();
        H_est.resize(fft_size);
        
        #pragma omp simd simdlen(16)
        for (size_t i = 0; i < fft_size; ++i) {
            float rx_real = rx_symbol[i].real();
            float rx_imag = rx_symbol[i].imag();
            float tx_real = tx_zc[i].real();
            float tx_imag = tx_zc[i].imag();
            float denom = tx_real * tx_real + tx_imag * tx_imag;
            float inv_denom = 1.0f / denom;
            H_est[i] = std::complex<float>(
                (rx_real * tx_real + rx_imag * tx_imag) * inv_denom,
                (rx_imag * tx_real - rx_real * tx_imag) * inv_denom
            );
        }
    }

    /**
     * @brief Estimate channel using multiplication (for unit magnitude TX): H = Rx * conj(Tx)
     * This is faster when TX symbols have unit magnitude (e.g., QPSK, ZC).
     */
    static void estimate_with_conjugate_multiply(
        const std::complex<float>* __restrict__ rx_data,
        const std::complex<float>* __restrict__ tx_data,
        std::complex<float>* __restrict__ h_est,
        size_t length
    ) {
        #pragma omp simd simdlen(16) aligned(rx_data, tx_data, h_est: 64)
        for (size_t k = 0; k < length; ++k) {
            float rx_real = rx_data[k].real();
            float rx_imag = rx_data[k].imag();
            float tx_real = tx_data[k].real();
            float tx_imag = tx_data[k].imag();
            
            // Multiply by conjugate: (a+bi)*(c-di) = (ac+bd) + (bc-ad)i
            h_est[k] = std::complex<float>(
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
};


/**
 * @brief Synchronization Processor.
 * 
 * Provides synchronization correlation and CFO estimation.
 */
class SyncProcessor {
public:
    /**
     * @brief Sliding window correlation for sync detection.
     * Correlates received signal with known sync sequence.
     * 
     * @param sync_data Received data (2 frames worth)
     * @param sync_real Real part of local sync sequence (CP + symbol)
     * @param sync_imag Imag part of local sync sequence (CP + symbol)
     * @param symbol_len Length of one OFDM symbol (fft_size + cp_length)
     * @param max_pos Output: position of maximum correlation
     * @param max_corr Output: maximum correlation value
     * @param avg_corr Output: average correlation value
     */
    static void find_sync_position(
        const AlignedVector& sync_data,
        const AlignedFloatVector& sync_real,
        const AlignedFloatVector& sync_imag,
        size_t symbol_len,
        int& max_pos,
        float& max_corr,
        float& avg_corr
    ) {
        const size_t n_windows = sync_data.size() - symbol_len + 1;
        max_corr = 0.0f;
        avg_corr = 0.0f;
        max_pos = 0;
        
        for (size_t i = 0; i < n_windows; ++i) {
            float sum_real = 0.0f, sum_imag = 0.0f;
            
            #pragma omp simd reduction(+:sum_real, sum_imag)
            for (size_t j = 0; j < symbol_len; ++j) {
                const float rx_real = sync_data[i + j].real();
                const float rx_imag = sync_data[i + j].imag();
                
                sum_real += rx_real * sync_real[j] + rx_imag * sync_imag[j];
                sum_imag += rx_imag * sync_real[j] - rx_real * sync_imag[j];
            }
            
            const float corr = sum_real * sum_real + sum_imag * sum_imag;
            
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
};


/**
 * @brief Noise and LLR Processor.
 * 
 * Provides noise variance estimation and LLR calculation.
 */
class NoiseEstimator {
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
 */
class DelayProcessor {
public:
    /**
     * @brief Compute delay spectrum from channel estimate.
     * Performs FFT shift + IFFT + scaling.
     * 
     * @param H_est Channel estimate (frequency domain)
     * @param ifft_in IFFT input buffer (workspace)
     * @param ifft_out IFFT output buffer (workspace)
     * @param ifft_plan External FFTW plan
     * @param delay_spectrum Output delay spectrum
     */
    static void compute_delay_spectrum(
        const AlignedVector& H_est,
        AlignedVector& ifft_in,
        AlignedVector& ifft_out,
        fftwf_plan ifft_plan,
        AlignedVector& delay_spectrum
    ) {
        const size_t fft_size = H_est.size();
        
        // FFT shift (swap halves)
        const size_t half = fft_size / 2;
        ifft_in.resize(fft_size);
        
        #pragma omp simd
        for (size_t i = 0; i < half; ++i) {
            ifft_in[i] = H_est[i + half];
            ifft_in[i + half] = H_est[i];
        }
        
        // Execute IFFT
        fftwf_execute(ifft_plan);
        
        // Scale and copy to output
        const float scale = 1.0f / std::sqrt(static_cast<float>(fft_size));
        delay_spectrum.resize(fft_size);
        
        #pragma omp simd
        for (size_t i = 0; i < fft_size; ++i) {
            delay_spectrum[i] = ifft_out[i] * scale;
        }
    }

    /**
     * @brief Find peak in delay spectrum and compute statistics.
     * 
     * @param delay_spectrum Input delay spectrum
     * @param max_index Output: index of maximum magnitude
     * @param max_mag Output: maximum magnitude value
     * @param avg_mag Output: average magnitude
     */
    static void find_peak(
        const AlignedVector& delay_spectrum,
        size_t& max_index,
        float& max_mag,
        float& avg_mag
    ) {
        const size_t fft_size = delay_spectrum.size();
        max_index = 0;
        max_mag = 0.0f;
        avg_mag = 0.0f;
        
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
};


/**
 * @brief Core Sensing Processing Operations.
 * 
 * Contains pure computation functions for ISAC sensing without any I/O.
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

public:
    explicit SensingProcessor(const Params& params)
        : _params(params) {}

    /**
     * @brief Initialize Hamming windows for range and Doppler processing.
     */
    void init_windows(AlignedFloatVector& range_window, AlignedFloatVector& doppler_window) {
        WindowGenerator::generate_hamming(range_window, _params.fft_size);
        WindowGenerator::generate_hamming(doppler_window, _params.sensing_symbol_num);
    }

    /**
     * @brief Channel estimation with conjugate multiplication and in-place FFT shift.
     * For mono-static sensing where TX symbols have unit magnitude.
     */
    void channel_estimate_with_shift(
        AlignedVector& channel_buffer,
        const std::vector<AlignedVector>& tx_symbols,
        size_t fft_size
    ) {
        const size_t half_size = fft_size / 2;
        
        for (size_t i = 0; i < _params.sensing_symbol_num; ++i) {
            auto* __restrict__ ch_data = channel_buffer.data() + i * _params.range_fft_size;
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
     * @brief Channel estimation with division (for bi-static sensing).
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
        
        #pragma omp simd
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

#endif // OFDM_CORE_HPP
