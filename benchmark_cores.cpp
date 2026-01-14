
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <complex>
#include "OFDMModulatorCore.hpp"
#include "OFDMDemodulatorCore.hpp"
#include "SensingCore.hpp"

using namespace OpenISAC;
using namespace OpenISAC::Core;

// Helper to generate random bytes
std::vector<uint8_t> generate_random_data(size_t size) {
    std::vector<uint8_t> data(size);
    for(size_t i=0; i<size; ++i) data[i] = rand() % 256;
    return data;
}

int main() {
    std::cout << "Starting Core Benchmark..." << std::endl;

    // Parameters
    size_t fft_size = 1024;
    size_t cp_length = 128;
    size_t num_symbols = 14;
    size_t sample_rate = 50000000;
    
    // 1. Initialize Modulator
    OFDMModulatorCore::Params mod_params;
    mod_params.fft_size = fft_size;
    mod_params.cp_length = cp_length;
    mod_params.num_symbols = num_symbols;
    mod_params.pilot_positions = {100, 200, 300, 400, 500, 600, 700, 800}; // Example pilots
    mod_params.sync_pos = 1;
    mod_params.zc_root = 29;
    mod_params.scaling_factor = 1.0f / sqrt(fft_size) / 4.0f; 
    
    auto modulator = std::make_unique<OFDMModulatorCore>(mod_params);

    // 2. Initialize Demodulator
    OFDMDemodulatorCore::Params demod_params;
    demod_params.fft_size = fft_size;
    demod_params.cp_length = cp_length;
    demod_params.num_symbols = num_symbols;
    demod_params.sample_rate = sample_rate;
    demod_params.center_freq = 2.4e9;
    demod_params.pilot_positions = mod_params.pilot_positions;
    demod_params.sync_pos = mod_params.sync_pos;
    demod_params.zc_root = mod_params.zc_root;
    demod_params.sync_samples = fft_size + cp_length;
    
    auto demodulator = std::make_unique<OFDMDemodulatorCore>(demod_params);

    // 3. Initialize Sensing Core
    SensingCore::Params sensing_params;
    sensing_params.fft_size = fft_size;
    sensing_params.sensing_symbol_num = num_symbols;
    sensing_params.range_fft_size = fft_size;
    sensing_params.doppler_fft_size = 64;
    sensing_params.enable_mti = true;
    sensing_params.enable_windowing = true;

    auto sensing_core = std::make_unique<SensingCore>(sensing_params);

    // Prepare Data
    size_t payload_len = 1000;
    AlignedIntVector payload_indices(payload_len); 
    for(size_t k=0; k<payload_len; ++k) payload_indices[k] = rand() % 4;

    // Benchmark Loop
    int num_iterations = 100;
    
    std::cout << "Running " << num_iterations << " iterations..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    AlignedVector frame_buffer;
    std::vector<AlignedVector> symbols_buffer;

    for(int i=0; i<num_iterations; ++i) {
        // Modulate
        modulator->generate_frame(payload_indices, frame_buffer, symbols_buffer);
        
        // Channel Simulation (Identity)
        // frame_buffer is ready
        
        // Demodulate
        OFDMDemodulatorCore::DemodResult demod_res;
        demodulator->process_frame(frame_buffer, demod_res);
        
        // Sensing
        std::vector<AlignedVector> sensing_rx = demod_res.rx_symbols;
        // Since we have perfect loopback, RX symbols ~= TX symbols (phase rotated maybe)
        // We use symbols_buffer (TX freq domain) as ref
        
        sensing_core->process(sensing_rx, symbols_buffer);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    std::cout << "Total time: " << duration / 1000.0 << " ms" << std::endl;
    std::cout << "Avg time per frame: " << duration / (double)num_iterations << " us" << std::endl;

    return 0;
}
