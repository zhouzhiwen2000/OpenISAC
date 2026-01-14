
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <complex>
#include "OFDMModulatorCore.hpp"
#include "OFDMDemodulatorCore.hpp"
#include "SensingCore.hpp"

using namespace Core;

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
    mod_params.sample_rate = sample_rate;
    mod_params.center_freq = 2.4e9;
    mod_params.pilot_positions = {100, 200, 300, 400, 500, 600, 700, 800}; // Example pilots
    mod_params.sync_pos = 1;
    mod_params.zc_root = 29;
    
    auto modulator = std::make_unique<OFDMModulatorCore>(mod_params);

    // 2. Initialize Demodulator
    OFDMDemodulatorCore::Params demod_params = mod_params; // Similar params
    demod_params.sync_samples = fft_size + cp_length; // Just for init
    
    auto demodulator = std::make_unique<OFDMDemodulatorCore>(demod_params);

    // 3. Initialize Sensing Core
    SensingCore::Params sensing_params;
    sensing_params.fft_size = fft_size;
    sensing_params.cp_length = cp_length;
    sensing_params.sensing_symbol_num = num_symbols;
    sensing_params.range_fft_size = fft_size;
    sensing_params.doppler_fft_size = 64;
    sensing_params.frame_count_for_process = num_symbols;

    auto sensing_core = std::make_unique<SensingCore>(sensing_params);

    // Prepare Data
    size_t payload_size = 1000;
    auto payload_data = generate_random_data(payload_size);

    // Benchmark Loop
    int num_iterations = 100;
    
    std::cout << "Running " << num_iterations << " iterations..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for(int i=0; i<num_iterations; ++i) {
        // Modulate
        OFDMModulatorCore::ModResult mod_res;
        modulator->modulate(payload_data, mod_res);
        
        // Channel Simulation (Identity)
        std::vector<std::complex<float>> rx_data = mod_res.frame_data;
        
        // Demodulate
        OFDMDemodulatorCore::DemodResult demod_res;
        demodulator->process_frame(rx_data, demod_res);
        
        // Sensing
        // Feed RX and TX symbols to sensing core
        // Note: SensingCore expects accumulated symbols or frame?
        // It takes rx_symbols and tx_symbols vectors.
        std::vector<AlignedVector> sensing_rx = demod_res.rx_symbols;
        // Mock TX symbols (using modulator's intermediate if available, or just use RX for perfect loopback)
        std::vector<AlignedVector> sensing_tx = sensing_rx; 
        
        sensing_core->process(sensing_rx, sensing_tx);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    std::cout << "Total time: " << duration / 1000.0 << " ms" << std::endl;
    std::cout << "Avg time per frame: " << duration / (double)num_iterations << " us" << std::endl;

    return 0;
}
