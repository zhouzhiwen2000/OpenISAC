#ifndef OFDM_DEMODULATOR_CORE_HPP
#define OFDM_DEMODULATOR_CORE_HPP

#include <vector>
#include <complex>
#include <fftw3.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <Common.hpp>
#include <OFDMSignalProcessing.hpp>

namespace OpenISAC {
namespace Core {

    class OFDMDemodulatorCore {
    public:
        struct Params {
            size_t fft_size;
            size_t cp_length;
            size_t num_symbols;
            size_t sync_pos;
            size_t zc_root;
            double sample_rate;
            double center_freq;
            std::vector<size_t> pilot_positions;
            size_t sync_samples; // Window size for sync search
        };

        struct SyncResult {
            bool found;
            int offset;
            float correlation;
            double cfo;
        };

        struct DemodResult {
            std::vector<AlignedVector> rx_symbols;      // Raw Frequency Domain
            std::vector<AlignedVector> equalized_symbols;
            AlignedVector channel_est;
            AlignedFloatVector llr;
            float cfo_est;
            float sfo_est;
            float noise_var;
        };

        OFDMDemodulatorCore(const Params& params) : _params(params) {
            _init_memory();
            _init_fftw();
            _init_resources();
        }

        ~OFDMDemodulatorCore() {
            if (_fft_plan) fftwf_destroy_plan(_fft_plan);
            if (_ifft_plan) fftwf_destroy_plan(_ifft_plan);
        }

        /**
         * @brief Perform synchronization search on raw samples
         */
        SyncResult find_sync(const AlignedVector& samples) {
             const size_t symbol_len = _params.fft_size + _params.cp_length;
             // Ensure sufficient data
             if (samples.size() < symbol_len) return {false, 0, 0.0f, 0.0f};

             size_t n_windows = samples.size() - symbol_len + 1;
             float max_corr = 0.0f;
             float avg_corr = 0.0f;
             int max_pos = 0;

             // Prepare local sync ref (Time Domain)
             // We need this pre-calculated. _tx_sync_time
             
             // Sliding window correlation
             for (size_t i = 0; i < n_windows; ++i) {
                 float sum_real = 0.0f, sum_imag = 0.0f;
                 #pragma omp simd reduction(+:sum_real, sum_imag)
                 for (size_t j = 0; j < symbol_len; ++j) {
                     float rx_r = samples[i+j].real();
                     float rx_i = samples[i+j].imag();
                     float ref_r = _tx_sync_time[j].real();
                     float ref_i = _tx_sync_time[j].imag();
                     // Conjugate correlation: sum(rx * conj(ref))
                     sum_real += rx_r * ref_r + rx_i * ref_i;
                     sum_imag += rx_i * ref_r - rx_r * ref_i;
                 }
                 float corr = sum_real*sum_real + sum_imag*sum_imag;
                 if (corr > max_corr) { max_corr = corr; max_pos = i; }
                 avg_corr += corr;
             }
             avg_corr /= n_windows;

             // Threshold check
             if (max_corr / avg_corr > 50.0f) {
                 // Calculate CFO using CP correlation
                 double cfo = _estimate_cfo_cp(samples, max_pos);
                 return {true, max_pos, max_corr, cfo};
             }

             return {false, 0, 0.0f, 0.0f};
        }

        /**
         * @brief Demodulate a full aligned frame
         */
        void process_frame(const AlignedVector& time_domain_frame, 
                           DemodResult& result) 
        {
            // 1. FFT
            _fft_process(time_domain_frame, result.rx_symbols);

            // 2. Channel Estimation (using Sync)
            _channel_estimation(result.rx_symbols, result.channel_est);

            // 3. CFO/SFO Estimation (using Pilots)
            auto [alpha, beta] = _cfo_sfo_estimation(result.rx_symbols);
            result.cfo_est = alpha;
            result.sfo_est = beta;

            // 4. Equalization & Phase Correction
            _equalize(result.rx_symbols, result.channel_est, alpha, beta, result.equalized_symbols);

            // 5. Noise Estimation
            result.noise_var = _estimate_noise(result.equalized_symbols);

            // 6. LLR Calculation
            _calculate_llr(result.equalized_symbols, result.noise_var, result.llr);
        }

        /**
         * @brief Compute Delay Spectrum (IFFT of Channel Estimate)
         */
        void compute_delay_profile(const AlignedVector& channel_est, AlignedVector& delay_profile) {
            if (delay_profile.size() != _params.fft_size) delay_profile.resize(_params.fft_size);
            
            // Perform IFFT
            // Copy input to output buffer (if inplace) or use separate buffer?
            // FFTW plans use specific pointers. 
            // Reuse _fft_in/_fft_out buffers to save memory?
            // channel_est -> _fft_in --IFFT--> _fft_out -> delay_profile
            std::copy(channel_est.begin(), channel_est.end(), _fft_in.begin());
            
            fftwf_execute(_ifft_plan);

            const float scale = 1.0f / sqrtf(_params.fft_size);
            #pragma omp simd
            for(size_t i=0; i<_params.fft_size; ++i) {
                delay_profile[i] = _fft_out[i] * scale;
            }
        }

    private:
        Params _params;
        AlignedVector _fft_in;
        AlignedVector _fft_out;
        fftwf_plan _fft_plan = nullptr;
        fftwf_plan _ifft_plan = nullptr;

        AlignedVector _zc_freq;
        AlignedVector _tx_sync_time; // Time domain sync symbol with CP
        
        AlignedVector _H_inv; // Cache for equalizer

        std::vector<int> _actual_subcarrier_indices;
        
        void _init_memory() {
            _fft_in.resize(_params.fft_size);
            _fft_out.resize(_params.fft_size);
            _H_inv.resize(_params.fft_size);
            
            _actual_subcarrier_indices.resize(_params.fft_size);
            for(size_t i=0; i<_params.fft_size; ++i) {
                if (i < _params.fft_size/2) _actual_subcarrier_indices[i] = (int)i;
                else _actual_subcarrier_indices[i] = (int)i - (int)_params.fft_size;
            }
        }

        void _init_fftw() {
            if (_fft_plan) fftwf_destroy_plan(_fft_plan);
            _fft_plan = fftwf_plan_dft_1d(
                static_cast<int>(_params.fft_size),
                reinterpret_cast<fftwf_complex*>(_fft_in.data()),
                reinterpret_cast<fftwf_complex*>(_fft_out.data()),
                FFTW_FORWARD,
                FFTW_MEASURE
            );
            
            _ifft_plan = fftwf_plan_dft_1d(
                static_cast<int>(_params.fft_size),
                reinterpret_cast<fftwf_complex*>(_fft_in.data()),
                reinterpret_cast<fftwf_complex*>(_fft_out.data()),
                FFTW_BACKWARD,
                FFTW_MEASURE
            );
        }

        void _init_resources() {
            // 1. ZC Freq
            _zc_freq = DSP::generate_zc_sequence(_params.fft_size, _params.zc_root);

            // 2. Tx Sync Time (IFFT(ZC) + CP)
            AlignedVector ifft_out(_params.fft_size);
            fftwf_plan p = fftwf_plan_dft_1d(_params.fft_size,
                 reinterpret_cast<fftwf_complex*>(_zc_freq.data()),
                 reinterpret_cast<fftwf_complex*>(ifft_out.data()),
                 FFTW_BACKWARD, FFTW_ESTIMATE);
            fftwf_execute(p);
            fftwf_destroy_plan(p);
            
            const float scale = 1.0f / sqrtf(_params.fft_size);
            for(auto& val : ifft_out) val *= scale; 

            _tx_sync_time.resize(_params.fft_size + _params.cp_length);
            // Append CP
            if (_params.cp_length > 0)
                 std::copy(ifft_out.end() - _params.cp_length, ifft_out.end(), _tx_sync_time.begin());
            std::copy(ifft_out.begin(), ifft_out.end(), _tx_sync_time.begin() + _params.cp_length);
        }

        double _estimate_cfo_cp(const AlignedVector& samples, int start_pos) {
            // Simple CP correlation
            // angle(sum(x[i] * conj(x[i+N])))
            size_t cp_start = start_pos; // Start of CP
            size_t sym_start = start_pos + _params.cp_length; // Start of Symbol
            // But we compare CP (end of symbol) with CP (start of symbol)
            // Wait, standard CP CFO matches CP with end of symbol. 
            // x[i] is CP, x[i+N] is end of symbol.
            // Loop over CP length
            std::complex<float> sum(0,0);
            for(size_t i=0; i<_params.cp_length; ++i) {
                if (start_pos + i + _params.fft_size < samples.size()) {
                    sum += samples[start_pos + i] * std::conj(samples[start_pos + i + _params.fft_size]);
                }
            }
             // angle = -2*pi*delta_f*T_sym?
            // phase = 2*pi*cfo*N/N = 2*pi*cfo
            return -std::arg(sum) / (2.0 * M_PI); // Normalized CFO
        }

        void _fft_process(const AlignedVector& time_domain_frame, std::vector<AlignedVector>& rx_symbols) {
            if (rx_symbols.size() != _params.num_symbols) 
                rx_symbols.resize(_params.num_symbols, AlignedVector(_params.fft_size));

            size_t pos = 0;
            const float scale = 1.0f / sqrtf(_params.fft_size);

            for (size_t i = 0; i < _params.num_symbols; ++i) {
                // Remove CP and copy to FFT input
                if (pos + _params.cp_length + _params.fft_size > time_domain_frame.size()) break;
                
                std::copy(time_domain_frame.begin() + pos + _params.cp_length,
                          time_domain_frame.begin() + pos + _params.cp_length + _params.fft_size,
                          _fft_in.begin());
                
                fftwf_execute(_fft_plan);

                // Copy to output
                #pragma omp simd
                for (size_t j = 0; j < _params.fft_size; ++j) {
                    rx_symbols[i][j] = _fft_out[j] * scale;
                }
                pos += _params.fft_size + _params.cp_length;
            }
        }

        void _channel_estimation(const std::vector<AlignedVector>& rx_symbols, AlignedVector& H) {
            if (H.size() != _params.fft_size) H.resize(_params.fft_size);
            const auto& rx_sync = rx_symbols[_params.sync_pos];
            
            // H = Rx / Tx = Rx * conj(Tx) / |Tx|^2. Since |Tx| is const 1 (ZC)
            #pragma omp simd
            for (size_t i = 0; i < _params.fft_size; ++i) {
                std::complex<float> rx = rx_sync[i];
                std::complex<float> tx = _zc_freq[i];
                float denom = std::norm(tx);
                float inv = 1.0f/denom;
                H[i] = rx * std::conj(tx) * inv;
            }
        }

        std::pair<float, float> _cfo_sfo_estimation(const std::vector<AlignedVector>& rx_symbols) {
            std::vector<int> pilot_indices; 
            std::vector<float> weights; 
            std::vector<float> avg_phase_diff(_params.pilot_positions.size(), 0.0f);
            
            pilot_indices.reserve(_params.pilot_positions.size());
            weights.reserve(_params.pilot_positions.size());

            for (size_t j = 0; j < _params.pilot_positions.size(); j++) {
                auto pilot_index = _params.pilot_positions[j];
                std::complex<double> next_current_sum(0.0f, 0.0f);
                 #pragma omp simd
                 for (size_t i = _params.sync_pos; i < rx_symbols.size() - 1; i++) {
                    std::complex<float> current_pilot = rx_symbols[i][pilot_index];
                    std::complex<float> next_pilot = rx_symbols[i+1][pilot_index];
                    next_current_sum += std::conj(current_pilot) * (next_pilot);
                }
                avg_phase_diff[j] = std::arg(next_current_sum);
                int freq_index = static_cast<int>(pilot_index);
                if (freq_index >= static_cast<int>(_params.fft_size)/2) {
                    freq_index -= _params.fft_size;
                }
                pilot_indices.push_back(freq_index);
                weights.push_back(std::norm(next_current_sum));
            }
            
            DSP::unwrap(avg_phase_diff);
            // Returns [beta, alpha] -> [slope, constant]
            return DSP::weightedlinearRegression(pilot_indices, avg_phase_diff, weights);
        }

        void _equalize(const std::vector<AlignedVector>& rx_symbols, 
                       const AlignedVector& H, 
                       float alpha, float beta,
                       std::vector<AlignedVector>& eq_symbols) 
        {
            if (eq_symbols.size() != rx_symbols.size())
                eq_symbols.resize(rx_symbols.size(), AlignedVector(_params.fft_size));

            // Precompute H_inv
            #pragma omp simd
            for(size_t i=0; i<_params.fft_size; ++i) {
                float n = std::norm(H[i]);
                if (n < 1e-9f) _H_inv[i] = 0.0f;
                else _H_inv[i] = std::conj(H[i]) / n;
            }

            for (size_t i = 0; i < rx_symbols.size(); ++i) {
                const int relative_symbol_index = (i < _params.sync_pos) ? 
                    (static_cast<int>(i) - static_cast<int>(_params.sync_pos)) : 
                    (static_cast<int>(i) + 1 - static_cast<int>(_params.sync_pos));
                
                const float phase_diff_CFO = alpha * relative_symbol_index;
                const float beta_rel = beta * relative_symbol_index;

                #pragma omp simd simdlen(16)
                for(size_t j=0; j<_params.fft_size; ++j) {
                    const float phase = beta_rel * _actual_subcarrier_indices[j] + phase_diff_CFO;
                    
                    // First multiply by H_inv
                    std::complex<float> eq = rx_symbols[i][j] * _H_inv[j];
                    
                    // Derotate
                    const float c = std::cos(phase);
                    const float s = std::sin(phase);
                    // (re + i*im) * (c - i*s) = (re*c + im*s) + i(im*c - re*s)
                    eq_symbols[i][j] = std::complex<float>(
                        eq.real() * c + eq.imag() * s,
                        eq.imag() * c - eq.real() * s
                    );
                }
            }
        }

        float _estimate_noise(const std::vector<AlignedVector>& eq_symbols) {
            double err_power_acc = 0.0;
            size_t err_count = 0;
            for (const auto &sym : eq_symbols) {
                for (auto p : _params.pilot_positions) {
                    if (p < sym.size()) {
                        auto e = sym[p] - _zc_freq[p]; 
                        err_power_acc += std::norm(e);
                        err_count++;
                    }
                }
            }
            if (err_count > 0) return static_cast<float>(err_power_acc / err_count);
            return 0.001f;
        }

        void _calculate_llr(const std::vector<AlignedVector>& eq_symbols, float noise_var, AlignedFloatVector& llr) {
             const size_t data_sc_count = _params.fft_size - _params.pilot_positions.size();
             size_t total_llrs = eq_symbols.size() * data_sc_count * 2;
             if (llr.size() != total_llrs) llr.resize(total_llrs);

             if (noise_var < 1e-9f) noise_var = 1e-9f;
             float sigma2_dim = noise_var / 2.0f;
             float llr_scale = 2.0f / sigma2_dim;
             llr_scale = std::min(llr_scale * 0.70710678f, 500.0f); // 0.707 is for QPSK scaling? 
             // In original code: std::min(static_cast<float>(_llr_scale * M_SQRT1_2), 500.0f);
             // _llr_scale was 4/noise_var (which is 2/sigma2_dim). Correct.

             // We need to build data map once
             static std::vector<size_t> data_indices; // Making it static in header might be bad if multiple instances?
             // Better make it member or build locally. building locally is fast enough.
             if (data_indices.empty()) {
                 std::vector<char> is_pilot(_params.fft_size, 0);
                 for(auto p : _params.pilot_positions) if(p<_params.fft_size) is_pilot[p] = 1;
                 data_indices.reserve(data_sc_count);
                 for(size_t k=0; k<_params.fft_size; ++k) if(!is_pilot[k]) data_indices.push_back(k);
             }

             float* out_ptr = llr.data();
             
             for(size_t i=0; i<eq_symbols.size(); ++i) {
                 const auto& sym = eq_symbols[i];
                 #pragma omp simd simdlen(16)
                 for(size_t j=0; j<data_sc_count; ++j) {
                     size_t k = data_indices[j];
                     out_ptr[(i*data_sc_count + j)*2]     = sym[k].real() * llr_scale;
                     out_ptr[(i*data_sc_count + j)*2 + 1] = sym[k].imag() * llr_scale;
                 }
             }
        }
    };

} // namespace Core
} // namespace OpenISAC

#endif // OFDM_DEMODULATOR_CORE_HPP
