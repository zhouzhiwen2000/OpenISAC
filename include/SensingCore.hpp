#ifndef SENSING_CORE_HPP
#define SENSING_CORE_HPP

#include <vector>
#include <complex>
#include <fftw3.h>
#include <omp.h>
#include <Common.hpp>
#include <OFDMSignalProcessing.hpp>

namespace OpenISAC {
namespace Core {

    /**
     * @brief Sensing Core Processing Engine
     * 
     * Handles the Radar/ISAC processing pipeline:
     * 1. Channel Estimation (H = Rx / Tx)
     * 2. MTI Filtering (Clutter Removal)
     * 3. Range-Doppler Processing (2D FFT)
     */
    class SensingCore {
    public:
        struct Params {
            size_t fft_size;
            size_t range_fft_size;
            size_t doppler_fft_size;
            size_t sensing_symbol_num;
            bool enable_mti;
            bool enable_windowing;
        };

        SensingCore(const Params& params)
            : _params(params),
              _mti_filter(params.range_fft_size)
        {
            _init_memory();
            _init_fftw();
            _init_windows();
        }

        ~SensingCore() {
            if (_demod_fft_plan) fftwf_destroy_plan(_demod_fft_plan);
            if (_range_ifft_plan) fftwf_destroy_plan(_range_ifft_plan);
            if (_doppler_fft_plan) fftwf_destroy_plan(_doppler_fft_plan);
        }

        /**
         * @brief Update parameters (e.g. toggle MTI)
         */
        void update_params(const Params& new_params) {
            // Re-initialization needed if sizes change
            if (new_params.fft_size != _params.fft_size ||
                new_params.range_fft_size != _params.range_fft_size ||
                new_params.doppler_fft_size != _params.doppler_fft_size ||
                new_params.sensing_symbol_num != _params.sensing_symbol_num) 
            {
                _params = new_params;
                _init_memory();
                _init_fftw(); // Plans depend on sizes
                _init_windows();
                _mti_filter.resize(_params.range_fft_size);
            } else {
                // Just update flags
                _params.enable_mti = new_params.enable_mti;
                _params.enable_windowing = new_params.enable_windowing;
            }
        }
        
        void toggle_mti(bool enable) { _params.enable_mti = enable; }
        
        /**
         * @brief Process a full sensing frame
         * 
         * @param rx_symbols Received TIME-DOMAIN symbols (will be FFT'd internally)
         * @param tx_symbols Transmitted FREQUENCY-DOMAIN symbols (reference)
         * @return const AlignedVector& Reference to the processing result (Range-Doppler Map)
         */
        const AlignedVector& process(const std::vector<AlignedVector>& rx_symbols, 
                                     const std::vector<AlignedVector>& tx_symbols) 
        {
            // 0. FFT Demodulation: Convert time-domain RX symbols to frequency domain
            _fft_demodulation(rx_symbols);

            // 1. Channel Estimation (now using _rx_freq_symbols which are in frequency domain)
            _channel_estimation(_rx_freq_symbols, tx_symbols);

            // 2. MTI Filter
            if (_params.enable_mti) {
                _mti_filter.apply(_channel_response, _params.fft_size, _params.sensing_symbol_num);
            }

            // 3. Windowing & 2D FFT
            _range_doppler_processing();

            return _channel_response;
        }
        
        // Accessor for the result buffer
        const AlignedVector& get_results() const { return _channel_response; }

    private:
        Params _params;
        AlignedVector _channel_response;
        
        // FFT Demodulation (Time -> Frequency for RX symbols)
        AlignedVector _fft_in;                  // FFT input buffer
        AlignedVector _fft_out;                 // FFT output buffer
        std::vector<AlignedVector> _rx_freq_symbols;  // Demodulated RX symbols (frequency domain)
        fftwf_plan _demod_fft_plan = nullptr;   // Demodulation FFT plan
        
        // FFTW for Range-Doppler processing
        fftwf_plan _range_ifft_plan = nullptr;
        fftwf_plan _doppler_fft_plan = nullptr;

        // Windows
        AlignedFloatVector _range_window;
        AlignedFloatVector _doppler_window;

        // MTI
        MTIFilter _mti_filter;

        void _init_memory() {
            _channel_response.resize(_params.range_fft_size * _params.doppler_fft_size);
            std::fill(_channel_response.begin(), _channel_response.end(), std::complex<float>(0,0));
            
            // FFT demodulation buffers
            _fft_in.resize(_params.fft_size);
            _fft_out.resize(_params.fft_size);
            _rx_freq_symbols.resize(_params.sensing_symbol_num);
            for (auto& sym : _rx_freq_symbols) {
                sym.resize(_params.fft_size);
            }
        }

        void _init_fftw() {
            if (_demod_fft_plan) fftwf_destroy_plan(_demod_fft_plan);
            if (_range_ifft_plan) fftwf_destroy_plan(_range_ifft_plan);
            if (_doppler_fft_plan) fftwf_destroy_plan(_doppler_fft_plan);

            int demod_fft_size_int = static_cast<int>(_params.fft_size);
            int fft_size_int = static_cast<int>(_params.range_fft_size);
            int doppler_fft_size_int = static_cast<int>(_params.doppler_fft_size);

            // Demodulation FFT (Time -> Frequency for RX symbols)
            _demod_fft_plan = fftwf_plan_dft_1d(
                demod_fft_size_int,
                reinterpret_cast<fftwf_complex*>(_fft_in.data()),
                reinterpret_cast<fftwf_complex*>(_fft_out.data()),
                FFTW_FORWARD, FFTW_ESTIMATE
            );

            // Batch Range IFFT (Frequency -> Time)
            // n=fft_size, howmany=num_symbols
            _range_ifft_plan = fftwf_plan_many_dft(
                1, &fft_size_int, doppler_fft_size_int,
                reinterpret_cast<fftwf_complex*>(_channel_response.data()), 
                nullptr, 1, fft_size_int, // Input layout: contiguous symbols
                reinterpret_cast<fftwf_complex*>(_channel_response.data()), 
                nullptr, 1, fft_size_int, // Output layout: contiguous symbols
                FFTW_BACKWARD, FFTW_ESTIMATE
            );

            // Batch Doppler FFT (Time -> Frequency)
            // n=num_symbols, howmany=fft_size
            _doppler_fft_plan = fftwf_plan_many_dft(
                1, &doppler_fft_size_int, fft_size_int,
                reinterpret_cast<fftwf_complex*>(_channel_response.data()), 
                nullptr, fft_size_int, 1, // Input stride: jump fft_size to get next time sample of same bin
                reinterpret_cast<fftwf_complex*>(_channel_response.data()), 
                nullptr, fft_size_int, 1,
                FFTW_FORWARD, FFTW_ESTIMATE
            );
        }

        void _init_windows() {
            _range_window = DSP::generate_hanning_window(_params.fft_size);
            _doppler_window = DSP::generate_hanning_window(_params.sensing_symbol_num);
        }

        /**
         * @brief FFT Demodulation: Convert time-domain RX symbols to frequency domain
         */
        void _fft_demodulation(const std::vector<AlignedVector>& rx_time_symbols) {
            const size_t num_sym = std::min(_params.sensing_symbol_num, rx_time_symbols.size());
            
            for (size_t i = 0; i < num_sym; ++i) {
                // Copy time-domain symbol to FFT input buffer
                std::copy(rx_time_symbols[i].begin(), rx_time_symbols[i].end(), _fft_in.begin());
                
                // Execute FFT (Time -> Frequency)
                fftwf_execute(_demod_fft_plan);
                
                // Copy result to frequency-domain symbol storage
                std::copy(_fft_out.begin(), _fft_out.end(), _rx_freq_symbols[i].begin());
            }
        }

        void _channel_estimation(const std::vector<AlignedVector>& rx_symbols, 
                                 const std::vector<AlignedVector>& tx_symbols) 
        {
            const size_t num_sym = std::min({_params.sensing_symbol_num, rx_symbols.size(), tx_symbols.size()});
            const size_t fft_size = _params.fft_size;
            const size_t range_fft_size = _params.range_fft_size;
            const size_t half_size = fft_size / 2;

            for (size_t i = 0; i < num_sym; ++i) {
                auto* __restrict__ ch_curr = _channel_response.data() + i * range_fft_size;
                const auto* __restrict__ rx_curr = rx_symbols[i].data();
                const auto* __restrict__ tx_curr = tx_symbols[i].data();

                // Channel Est (H = Rx/Tx) + FFT Shift
                // Optimization: If Tx is unit circle (PSK/ZC), H = Rx * conj(Tx)
                // We assume Tx symbols are normalized.
                // FFT Shift: Store 2nd half of freq spectrum to 1st half of buffer, and vice-versa.
                
                // Process 1st half of spectrum -> store to 2nd half of buffer
                #pragma omp simd
                for (size_t k = 0; k < half_size; ++k) {
                    // Index k is in first half (0 to N/2-1) -> maps to (k + N/2)
                    std::complex<float> rx = rx_curr[k];
                    std::complex<float> tx = tx_curr[k];
                    // H = Rx * conj(Tx)
                    // (a+bi)(c-di) = (ac+bd) + i(bc-ad)
                    float est_real = rx.real() * tx.real() + rx.imag() * tx.imag();
                    float est_imag = rx.imag() * tx.real() - rx.real() * tx.imag();
                    
                    ch_curr[k + half_size] = std::complex<float>(est_real, est_imag);
                }

                // Process 2nd half of spectrum -> store to 1st half of buffer
                #pragma omp simd
                for (size_t k = half_size; k < fft_size; ++k) {
                    // Index k is in second half -> maps to (k - N/2)
                    std::complex<float> rx = rx_curr[k];
                    std::complex<float> tx = tx_curr[k];
                    
                    float est_real = rx.real() * tx.real() + rx.imag() * tx.imag();
                    float est_imag = rx.imag() * tx.real() - rx.real() * tx.imag();

                    ch_curr[k - half_size] = std::complex<float>(est_real, est_imag);
                }
            }
        }

        void _range_doppler_processing() {
            if (!_params.enable_windowing) return; // Should we strictly skip if disabled? Usually RD needs windowing/FFTs.
            // Actually, if we skip FFTs, we just have channel response.
            // But the method is named "process", implying RD map output.
            // We assume normally this runs.

            // Windowing
            for (size_t col = 0; col < _params.fft_size; ++col) {
                float w_r = _range_window[col];
                
                #pragma omp simd
                for (size_t row = 0; row < _params.sensing_symbol_num; ++row) {
                    size_t idx = row * _params.range_fft_size + col;
                    _channel_response[idx] *= w_r * _doppler_window[row];
                }
            }

            // 2D FFT
            fftwf_execute(_range_ifft_plan);
            fftwf_execute(_doppler_fft_plan);
        }
    };

} // namespace Core
} // namespace OpenISAC

#endif // SENSING_CORE_HPP
