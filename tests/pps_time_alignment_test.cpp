#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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
    struct Reading {
        int64_t now;
        int64_t last_pps;
    };
    // Each time_now poll advances to the next reading; a stalled device
    // repeats its final reading. Values use this device's own tick rate.
    std::vector<Reading> after_arm;
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
            if (!after_arm.empty()) {
                const auto& reading = after_arm[std::min(now_reads_after_arm, after_arm.size()) - 1];
                now_ticks = reading.now;
                last_pps_ticks = reading.last_pps;
            } else if (now_reads_after_arm == 2) {
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

static std::shared_ptr<FakeDevice> scripted_device(
    double rate, std::vector<FakeDevice::Reading> readings) {
    auto device = std::make_shared<FakeDevice>();
    device->tick_rate = rate;
    device->after_arm = std::move(readings);
    return device;
}

static void test_pre_load_latches() {
    // The old timestamp falls below time_now before the first post-load PPS.
    auto n210 = scripted_device(1000, {{400, 100}, {25, 500}, {550, 500}, {1025, 999}});
    n210->now_ticks = 300;
    auto result = radio::align_device_times_to_next_pps({{"N210", n210, "external"}}, fast_options());
    CHECK(result.devices[0].last_pps_ticks == 999 && result.devices[0].now_ticks == 1025,
          "N210 returns a fresh latch even when the old value looks plausible");

    auto x310 = scripted_device(2000, {{50, 0}, {2050, 2000}, {4050, 4000}});
    n210 = scripted_device(1000, {{25, 999}, {1025, 999}, {2025, 1999}});
    result = radio::align_device_times_to_next_pps(
        {{"X310", x310, "external"}, {"N210", n210, "external"}}, fast_options());
    CHECK(result.devices[0].last_pps_ticks == 4000 && result.devices[1].last_pps_ticks == 1999,
          "mixed devices wait through an identical first post-load latch at native clock rates");

    auto first = scripted_device(1000, {{25, 12000}, {1025, 999}});
    auto second = scripted_device(1000, {{25, 9000}, {1025, 1000}});
    result = radio::align_device_times_to_next_pps(
        {{"N210 A", first, "external"}, {"N210 B", second, "gpsdo"}}, fast_options());
    CHECK(result.devices[0].last_pps_ticks == 999 && result.devices[1].last_pps_ticks == 1000,
          "two pre-load devices return timestamps from the aligned epoch");
}

static void expect_alignment_failure(const std::vector<radio::PpsTimeAlignmentTarget>& targets,
                                     const std::string& error, const char* message) {
    try {
        radio::align_device_times_to_next_pps(targets, fast_options());
        CHECK(false, message);
    } catch (const std::runtime_error& e) {
        CHECK(std::string(e.what()).find(error) != std::string::npos, message);
    }
}

static void test_post_load_failures() {
    auto ready = scripted_device(1000, {{5000, 4000}, {25, 0}, {1025, 1000}});
    auto stuck = scripted_device(1000, {{450, 100}, {25, 500}, {1025, 500}});
    ready->now_ticks = 4900;
    stuck->now_ticks = 400;
    expect_alignment_failure({{"ready", ready, "external"}, {"stale", stuck, "external"}},
                             "post-load PPS", "one stuck latch prevents mixed-device success");

    auto ahead = scripted_device(1000, {{25, 5000}, {1150, 999}});
    auto behind = scripted_device(1000, {{25, 6000}, {1025, 999}});
    expect_alignment_failure({{"ahead", ahead, "external"}, {"behind", behind, "external"}},
                             "post-load PPS", "post-load device skew is rechecked before return");

    auto late_pps = scripted_device(1000, {{25, 4000}, {1150, 1140}});
    auto on_time = scripted_device(1000, {{25, 5000}, {1150, 999}});
    expect_alignment_failure({{"late", late_pps, "external"}, {"on time", on_time, "external"}},
                             "post-load PPS", "matching counters cannot hide skewed PPS anchors");
    auto expired = scripted_device(1000, {{25, 5000}, {2025, 999}});
    expect_alignment_failure({{"expired", expired, "external"}}, "post-load PPS",
                             "a post-load timestamp older than one PPS period is rejected");
    auto future = scripted_device(1000, {{25, 3500}, {1025, 2000}});
    expect_alignment_failure({{"future", future, "external"}}, "post-load PPS",
                             "a changed latch ahead of device time is rejected");
    auto restarted = scripted_device(1000, {{25, 3500}, {550, 3500}, {5, 0}});
    expect_alignment_failure({{"restarted", restarted, "external"}}, "reset again",
                             "a second reset invalidates the armed epoch");
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
    test_pre_load_latches();
    test_post_load_failures();
    test_missing_capability_rejected();
    test_missing_edge_rejected();
    test_failed_zero_verification_rejected();
    test_cross_device_epoch_mismatch_rejected();
    test_multi_device_internal_time_source_rejected();
    test_clock_source_alignment_enable();

    std::printf("PPS time-alignment tests: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
