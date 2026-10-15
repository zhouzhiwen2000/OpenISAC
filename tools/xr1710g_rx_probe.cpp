#include "Xr1710gBackend.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

// Acceptance probe for the same radio backend used by UE. Hardware control is
// enabled only when an explicit PLL_COMMAND is supplied.
int main(int argc, char** argv) {
    if (argc < 5 || argc > 7) {
        std::cerr << "usage: Xr1710gRxProbe SHM SAMPLE_RATE SOURCE_LO SECONDS [CFO_HZ [PLL_COMMAND]]\n";
        return 1;
    }
    try {
        radio::DeviceConfig cfg;
        cfg.backend = "xr1710g";
        cfg.xr1710g = {argv[1], std::stod(argv[2]), std::stod(argv[3]), 5.0};
        if (argc == 7) cfg.xr1710g.pll_command = argv[6];
        const double seconds = std::stod(argv[4]);
        if (!std::isfinite(seconds) || seconds <= 0 || seconds > 600)
            throw std::invalid_argument("SECONDS must be within (0, 600]");
        auto dev = radio::make_xr1710g_device(cfg);
        if (argc >= 6) dev->set_rx_freq_correction(std::stod(argv[5]));
        auto rx = dev->get_rx_stream(radio::StreamArgs("fc32", "sc16"));
        std::vector<radio::sample_t> samples(65536);
        rx->issue_stream_cmd(radio::StreamCmd(radio::StreamMode::StartContinuous));
        const auto begin = std::chrono::steady_clock::now();
        uint64_t total = 0, errors = 0, timeouts = 0, nonzero = 0;
        uint64_t overflows = 0, broken_chains = 0, bad_packets = 0;
        double peak = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count() < seconds) {
            radio::RxMetadata md;
            const auto n = rx->recv(samples.data(), samples.size(), md, .1);
            if (md.error_code == radio::RxError::Timeout) ++timeouts;
            else if (md.error_code != radio::RxError::None) ++errors;
            overflows += md.error_code == radio::RxError::Overflow;
            broken_chains += md.error_code == radio::RxError::BrokenChain;
            bad_packets += md.error_code == radio::RxError::BadPacket;
            total += n;
            // Sparse statistics keep the probe inexpensive at 320 MS/s.
            for (size_t i = 0; i < n; i += 256) {
                const double value = std::norm(samples[i]);
                if (!std::isfinite(value)) throw std::runtime_error("non-finite samples");
                nonzero += value != 0;
                peak = std::max(peak, value);
            }
        }
        rx->issue_stream_cmd(radio::StreamCmd(radio::StreamMode::StopContinuous));
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        std::cout << std::setprecision(12)
                  << "{\"samples\":" << total << ",\"seconds\":" << elapsed
                  << ",\"pll_control\":" << (cfg.xr1710g.pll_command.empty() ? "false" : "true")
                  << ",\"actual_correction_hz\":" << dev->rx_freq_correction()
                  << ",\"samples_per_second\":" << total/elapsed
                  << ",\"errors\":" << errors << ",\"timeouts\":" << timeouts
                  << ",\"overflows\":" << overflows << ",\"broken_chains\":" << broken_chains
                  << ",\"bad_packets\":" << bad_packets
                  << ",\"nonzero_probes\":" << nonzero << ",\"peak_power\":" << peak << "}\n";
        return total && nonzero && errors == 0 && timeouts == 0 ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "XR1710G: " << e.what() << '\n'; return 1;
    }
}
