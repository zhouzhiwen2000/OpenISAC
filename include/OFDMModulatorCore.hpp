#ifndef OFDM_MODULATOR_CORE_HPP
#define OFDM_MODULATOR_CORE_HPP

#include <vector>
#include <complex>
#include <fftw3.h>
#include <omp.h>
#include <cstring>
#include <Common.hpp>
#include <OFDMSignalProcessing.hpp>

namespace OpenISAC {
namespace Core {

    /**
     * @brief OFDM Modulator Core
     * 
     * Handles the Transmission processing pipeline:
     * 1. Resource Grid Mapping (Data + Pilots + ZC Sync)
     * 2. IFFT (Frequency -> Time)
     * 3. Cyclic Prefix (CP) Addition
     */
    class OFDMModulatorCore {
    public:
        struct Params {
            size_t fft_size;
            size_t cp_length;
            size_t num_symbols;
            size_t sync_pos;
            size_t zc_root;
            std::vector<size_t> pilot_positions;
            float scaling_factor; // e.g. 1.0/sqrt(N)/4
        };

        OFDMModulatorCore(const Params& params) : _params(params) {
            _init_memory();
            _init_fftw();
            _init_resources();
        }

        ~OFDMModulatorCore() {
            if (_ifft_plan) fftwf_destroy_plan(_ifft_plan);
        }

        /**
         * @brief Generate a full OFDM frame
         * 
         * @param data_payload Raw QPSK symbol indices (0-3) to be modulated.
         * @param frame_buffer Output buffer for Time-Domain samples (flattened).
         * @param symbols_buffer Output buffer for Frequency-Domain symbols (for sensing reference).
         */
        void generate_frame(const AlignedIntVector& data_payload, 
                            AlignedVector& frame_buffer, 
                            std::vector<AlignedVector>& symbols_buffer) 
        {
            // Ensure inputs are sized correctly
            size_t expected_size = _params.num_symbols * (_params.fft_size + _params.cp_length);
            if (frame_buffer.size() != expected_size) {
                frame_buffer.resize(expected_size);
            }
            if (symbols_buffer.size() != _params.num_symbols) {
                symbols_buffer.resize(_params.num_symbols);
                for(auto& s : symbols_buffer) s.resize(_params.fft_size);
            }

            size_t data_pool_pos = 0;
            size_t pool_size = data_payload.size();
            const int* pool_ptr = data_payload.data();

            for (size_t i = 0; i < _params.num_symbols; ++i) {
                const size_t pos = i * (_params.fft_size + _params.cp_length);
                auto* buf_ptr = frame_buffer.data() + pos;

                if (i == _params.sync_pos) {
                    // Sync symbol: Copy ZC sequence
                    std::memcpy(_fft_in.data(), _zc_seq.data(), _params.fft_size * sizeof(std::complex<float>));
                } else {
                    // Data symbol
                    _fill_data_symbol(pool_ptr, pool_size, data_pool_pos); 
                }

                // Save Frequency Domain (for sensing)
                std::memcpy(symbols_buffer[i].data(), _fft_in.data(), _params.fft_size * sizeof(std::complex<float>));

                // IFFT
                fftwf_execute(_ifft_plan);

                // Add CP & Scaling (Time Domain)
                _add_cp_and_scale(buf_ptr);
            }
        }

    private:
        Params _params;

        // FFTW
        AlignedVector _fft_in;
        AlignedVector _fft_out;
        fftwf_plan _ifft_plan = nullptr;

        // Resources
        AlignedVector _zc_seq;
        std::vector<char> _is_pilot;
        std::vector<int> _data_subcarriers;

        // Helpers
        void _init_memory() {
            _fft_in.resize(_params.fft_size);
            _fft_out.resize(_params.fft_size);
        }

        void _init_fftw() {
             if (_ifft_plan) fftwf_destroy_plan(_ifft_plan);
             _ifft_plan = fftwf_plan_dft_1d(
                static_cast<int>(_params.fft_size),
                reinterpret_cast<fftwf_complex*>(_fft_in.data()),
                reinterpret_cast<fftwf_complex*>(_fft_out.data()),
                FFTW_BACKWARD,
                FFTW_MEASURE // Measure for best performance since this is core loop
            );
        }

        void _init_resources() {
            // Pre-calculate ZC sequence
            _zc_seq = DSP::generate_zc_sequence(_params.fft_size, _params.zc_root);

            // Pilot Map
            _is_pilot.assign(_params.fft_size, 0);
            for (auto pos : _params.pilot_positions) {
                if (pos < _params.fft_size) _is_pilot[pos] = 1;
            }

            // Data Subcarrier Map
            _data_subcarriers.reserve(_params.fft_size - _params.pilot_positions.size());
            for (size_t k = 0; k < _params.fft_size; ++k) {
                if (!_is_pilot[k]) _data_subcarriers.push_back((int)k);
            }
        }

        void _fill_data_symbol(const int* pool_ptr, size_t pool_size, size_t& data_pool_pos) {
             // 1. Fill Pilots (from ZC)
             for (size_t k : _params.pilot_positions) {
                 if (k < _params.fft_size) _fft_in[k] = _zc_seq[k];
             }

             // 2. Fill Data
             const size_t ds_count = _data_subcarriers.size();
             const int* __restrict__ ds_ptr = _data_subcarriers.data();
             auto* __restrict__ fft_ptr = _fft_in.data();

             #pragma omp simd
             for (size_t di = 0; di < ds_count; ++di) {
                 const int k = ds_ptr[di];
                 int sym_idx = 0; 
                 // If we have data, use it; otherwise 0 (padding) or random?
                 // Original code used fallback to random pregen.
                 // Here we will just use 0 (QPSK 00) if exhausted, or wrap around?
                 // Let's assume strict pool usage. If exhausted, maybe just 0.
                 if (data_pool_pos + di < pool_size) {
                     sym_idx = pool_ptr[data_pool_pos + di];
                 }
                 fft_ptr[k] = DSP::QPSKMapper::map(sym_idx);
             }
             
             // Advance pool position
             size_t used = std::min(ds_count, (pool_size > data_pool_pos) ? pool_size - data_pool_pos : 0);
             data_pool_pos += used;
        }

        void _add_cp_and_scale(std::complex<float>* buf_ptr) {
            float scale = _params.scaling_factor;
            size_t N = _params.fft_size;
            size_t CP = _params.cp_length;

            // CP Part (Last CP samples of IFFT output -> Beginning of symbol)
            #pragma omp simd
            for (size_t j = 0; j < CP; ++j) {
                buf_ptr[j] = _fft_out[N - CP + j] * scale;
            }

            // Body Part (Full IFFT output)
            #pragma omp simd
            for (size_t j = 0; j < N; ++j) {
                buf_ptr[CP + j] = _fft_out[j] * scale;
            }
        }
    };

} // namespace Core
} // namespace OpenISAC

#endif // OFDM_MODULATOR_CORE_HPP
