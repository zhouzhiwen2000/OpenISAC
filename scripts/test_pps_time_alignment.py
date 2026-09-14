#!/usr/bin/env python3
"""Build and run deterministic PPS device-time alignment helper tests."""

import subprocess
import sys
import tempfile
from pathlib import Path


TEST_SOURCE = r"""
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "PpsTimeAlignment.hpp"

static int passed = 0;
static int failed = 0;

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL: %s\n", message); \
        ++failed; \
    } else { \
        ++passed; \
    } \
} while (0)

class FakeDevice final : public radio::IDevice {
public:
    bool pps_supported = true;
    bool advance_edge = true;
    bool apply_alignment = true;
    mutable size_t last_pps_reads = 0;
    size_t set_next_pps_calls = 0;
    mutable size_t now_reads_after_arm = 0;
    mutable int64_t last_pps_ticks = 100;
    mutable int64_t now_ticks = 400'000'000;
    int64_t aligned_now_ticks = 25;
    double tick_rate = 200.0e6;

    bool supports(radio::Capability cap) const override {
        return cap == radio::Capability::PpsTimeSync && pps_supported;
    }

    radio::TimeSpec time_now() const override {
        if (set_next_pps_calls > 0) {
            ++now_reads_after_arm;
            if (now_reads_after_arm == 2) {
                if (apply_alignment) {
                    last_pps_ticks = 0;
                    now_ticks = aligned_now_ticks;
                }
            } else if (now_reads_after_arm > 2) {
                now_ticks += 1;
            }
        }
        return radio::TimeSpec::from_ticks(now_ticks, tick_rate);
    }

    radio::TimeSpec time_last_pps() const override {
        ++last_pps_reads;
        if (advance_edge && set_next_pps_calls == 0 && last_pps_reads >= 2) {
            last_pps_ticks = 200;
        }
        return radio::TimeSpec::from_ticks(last_pps_ticks, tick_rate);
    }

    void set_time_next_pps(const radio::TimeSpec&) override {
        ++set_next_pps_calls;
    }

    double master_clock_rate() const override { return tick_rate; }
    double get_tx_rate(size_t = 0) const override { return 50.0e6; }

    radio::TuneResult set_tx_freq(
        const radio::TuneRequest&, size_t = 0) override {
        return {};
    }
    radio::TuneResult set_rx_freq(
        const radio::TuneRequest&, size_t = 0) override {
        return {};
    }
    radio::ITxStreamPtr get_tx_stream(
        const radio::StreamArgs&) override {
        return {};
    }
    radio::IRxStreamPtr get_rx_stream(
        const radio::StreamArgs&) override {
        return {};
    }
};

static radio::PpsTimeAlignmentOptions fast_options() {
    radio::PpsTimeAlignmentOptions options;
    options.edge_timeout = std::chrono::milliseconds(5);
    options.apply_timeout = std::chrono::milliseconds(5);
    options.poll_interval = std::chrono::milliseconds(0);
    options.max_device_time_skew = std::chrono::milliseconds(1);
    return options;
}

static void test_same_edge_alignment_and_dedup() {
    auto first = std::make_shared<FakeDevice>();
    auto second = std::make_shared<FakeDevice>();
    const auto result = radio::align_device_times_to_next_pps(
        {
            {"BS TX", first, "external"},
            {"BS TX duplicate", first, "external"},
            {"BS RX", second, "external"},
        },
        fast_options());

    CHECK(result.devices.size() == 2,
          "duplicate device pointers are armed only once");
    CHECK(first->set_next_pps_calls == 1 && second->set_next_pps_calls == 1,
          "all distinct devices receive one next-PPS command");
    CHECK(first->now_reads_after_arm >= 2 && second->now_reads_after_arm >= 2,
          "alignment is accepted only after the armed PPS edge resets device time");
    CHECK(result.devices[0].last_pps_ticks == 0 &&
              result.devices[1].last_pps_ticks == 0,
          "all aligned devices verify target PPS at time zero");
    CHECK(result.devices[0].now_ticks == 25 &&
              result.devices[0].tick_rate == 200.0e6,
          "post-alignment diagnostics preserve native tick units");
}

static void test_missing_capability_rejected() {
    auto device = std::make_shared<FakeDevice>();
    device->pps_supported = false;
    bool rejected = false;
    try {
        radio::align_device_times_to_next_pps(
            {{"unsupported", device, "internal"}}, fast_options());
    } catch (const std::runtime_error& e) {
        rejected = std::string(e.what()).find("unsupported") != std::string::npos;
    }
    CHECK(rejected, "device without PPS capability is rejected");
}

static void test_missing_edge_rejected() {
    auto device = std::make_shared<FakeDevice>();
    device->advance_edge = false;
    bool rejected = false;
    try {
        radio::align_device_times_to_next_pps(
            {{"no PPS", device, "internal"}}, fast_options());
    } catch (const std::runtime_error& e) {
        rejected = std::string(e.what()).find("timed out") != std::string::npos;
    }
    CHECK(rejected, "missing PPS transition times out instead of using set_time_now");
}

static void test_failed_zero_verification_rejected() {
    auto device = std::make_shared<FakeDevice>();
    device->apply_alignment = false;
    bool rejected = false;
    try {
        radio::align_device_times_to_next_pps(
            {{"missed target", device, "internal"}}, fast_options());
    } catch (const std::runtime_error& e) {
        rejected = std::string(e.what()).find("timed out verifying") !=
            std::string::npos;
    }
    CHECK(rejected, "nonzero target PPS timestamp fails verification");
}

static void test_cross_device_epoch_mismatch_rejected() {
    auto first = std::make_shared<FakeDevice>();
    auto second = std::make_shared<FakeDevice>();
    second->aligned_now_ticks = 200'000'025;
    bool rejected = false;
    try {
        radio::align_device_times_to_next_pps(
            {{"epoch zero", first, "external"},
             {"epoch one", second, "external"}},
            fast_options());
    } catch (const std::runtime_error& e) {
        rejected = std::string(e.what()).find("timed out verifying") !=
            std::string::npos;
    }
    CHECK(rejected, "devices one second apart fail time_now consistency check");
}

static void test_multi_device_internal_time_source_rejected() {
    auto first = std::make_shared<FakeDevice>();
    auto second = std::make_shared<FakeDevice>();
    bool rejected = false;
    try {
        radio::align_device_times_to_next_pps(
            {{"internal TX", first, "internal"},
             {"external RX", second, "external"}},
            fast_options());
    } catch (const std::runtime_error& e) {
        rejected = std::string(e.what()).find("shared time_source") !=
            std::string::npos;
    }
    CHECK(rejected,
          "distinct devices reject internal PPS before alignment is armed");
    CHECK(first->set_next_pps_calls == 0 && second->set_next_pps_calls == 0,
          "multi-device time-source validation runs before PPS device I/O");
}

static void test_clock_source_alignment_enable() {
    CHECK(radio::clock_source_enables_pps_alignment("internal"),
          "internal clock enables startup PPS alignment");
    CHECK(radio::clock_source_enables_pps_alignment("external"),
          "external clock enables startup PPS alignment");
    CHECK(radio::clock_source_enables_pps_alignment("gpsdo"),
          "gpsdo clock enables startup PPS alignment");
    CHECK(!radio::clock_source_enables_pps_alignment(""),
          "empty clock source does not enable startup PPS alignment");
    bool rejected = false;
    try {
        (void)radio::clock_source_enables_pps_alignment("externl");
    } catch (const std::runtime_error& e) {
        rejected = std::string(e.what()).find("Invalid clock_source") !=
            std::string::npos;
    }
    CHECK(rejected, "misspelled clock source is rejected with a clear error");
}

int main() {
    test_same_edge_alignment_and_dedup();
    test_missing_capability_rejected();
    test_missing_edge_rejected();
    test_failed_zero_verification_rejected();
    test_cross_device_epoch_mismatch_rejected();
    test_multi_device_internal_time_source_rejected();
    test_clock_source_alignment_enable();

    std::printf("PPS time-alignment tests: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
"""


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="openisac-pps-align-test-") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "test_pps_time_alignment.cpp"
        binary = tmp_path / "test_pps_time_alignment"
        source.write_text(TEST_SOURCE)

        build_cmd = [
            "g++",
            "-std=c++17",
            "-O2",
            f"-I{repo_root / 'include'}",
            "-o",
            str(binary),
            str(source),
            "-lpthread",
        ]
        print(f"Building: {' '.join(build_cmd)}")
        result = subprocess.run(
            build_cmd, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"BUILD FAILED:\n{result.stderr}")
            return 1

        result = subprocess.run(
            [str(binary)], capture_output=True, text=True, timeout=30
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
