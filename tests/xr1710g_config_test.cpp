#include "Common.hpp"
#include "Xr1710gBackend.hpp"
#include <filesystem>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("missing repository path");
        Config cfg = make_default_ue_config();
        if (!load_ue_config_from_yaml(cfg, std::string(argv[1])+"/config/UE_XR1710G.yaml"))
            throw std::runtime_error("XR1710G preset did not load");
        radio::validate_xr1710g_config(cfg.radio.xr1710g);
        if (cfg.radio.radio_backend != "xr1710g" || cfg.radio.xr1710g.sample_rate != 40e6 ||
            cfg.radio.xr1710g.center_freq != 2442e6 || cfg.radio.xr1710g.shm_name != "/xr1710g-live" ||
            cfg.radio.xr1710g.attach_timeout_s != 5 || uplink_enabled(cfg))
            throw std::runtime_error("XR1710G YAML fields did not reach the device config");
        auto pll_yaml = YAML::Load("radio: {radio_backend: xr1710g, xr1710g_pll_command: 'python3 helper.py'}");
        Config pll_cfg = make_default_ue_config();
        load_simulation_config(pll_yaml, pll_cfg);
        if (pll_cfg.radio.xr1710g.pll_command != "python3 helper.py")
            throw std::runtime_error("PLL helper command was not parsed");
        DemodControlTimeGates gates;
        gates.mark_freq_adjust(100000);
        if (gates.allow_freq_adjust(99999) || gates.allow_freq_adjust(100000) || !gates.allow_freq_adjust(100001))
            throw std::runtime_error("existing stale-CFO time gate regressed");
        for (int invalid = 0; invalid < 6; ++invalid) {
            auto bad = cfg;
            if (invalid == 0) { bad.uplink.enabled = true; bad.uplink.duplex.mode = DuplexMode::FDD; }
            if (invalid == 1) bad.sync_tracking.predictive_delay = true;
            if (invalid == 2) bad.sync_tracking.hardware_sync = true;
            if (invalid == 3) bad.network_output.arq_enabled = true;
            if (invalid == 4) bad.downlink.rx_channel = 1;
            if (invalid == 5) bad.clock_time.clock_source = "external";
            bool rejected = false;
            try { validate_xr1710g_ue_config(bad); } catch (const std::exception&) { rejected = true; }
            if (!rejected) throw std::runtime_error("unsupported XR1710G UE mode accepted");
        }
        for (const auto& path : std::filesystem::directory_iterator(std::string(argv[1])+"/config")) {
            if (path.path().filename().string().rfind("UE_", 0) != 0 || path.path().extension() != ".yaml") continue;
            Config legacy = make_default_ue_config();
            if (!load_ue_config_from_yaml(legacy, path.path().string()))
                throw std::runtime_error("existing UE preset failed: " + path.path().string());
        }
        Config bs = make_default_bs_config();
        if (load_bs_config_from_yaml(bs, std::string(argv[1])+"/config/UE_XR1710G.yaml"))
            throw std::runtime_error("BS accepted receive-only backend");
        std::cout << "XR1710G config and existing UE presets passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
