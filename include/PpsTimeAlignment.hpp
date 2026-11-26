#ifndef PPS_TIME_ALIGNMENT_HPP
#define PPS_TIME_ALIGNMENT_HPP

/**
 * @file PpsTimeAlignment.hpp
 * @brief Non-real-time USRP device-time alignment on a shared PPS edge.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "RadioBackend.hpp"

namespace radio {

struct PpsTimeAlignmentTarget {
    std::string label;
    IDevicePtr device;
    std::string time_source;
};

struct PpsTimeAlignmentOptions {
    std::chrono::milliseconds edge_timeout{1500};
    std::chrono::milliseconds apply_timeout{1500};
    std::chrono::milliseconds poll_interval{10};
    std::chrono::milliseconds max_device_time_skew{100};
};

struct PpsTimeAlignmentSnapshot {
    std::string label;
    int64_t last_pps_ticks = 0;
    int64_t now_ticks = 0;
    double tick_rate = 0.0;
};

struct PpsTimeAlignmentResult {
    std::vector<PpsTimeAlignmentSnapshot> devices;
};

/**
 * @brief Set all distinct device timekeepers to zero on the same next PPS.
 *
 * This helper is for startup/control threads only. It first observes one PPS
 * transition on the first device, then arms every device immediately after
 * that edge so all commands have almost one full second to reach the common
 * next edge. It then polls until every device time counter is observed to reset
 * at that edge and verifies that the resulting device-time values agree.
 * USRP2/N210 latches the old time on the load edge. In that case, wait for
 * a subsequent PPS transition before giving BS/UE a timestamp in the new epoch.
 *
 * @param targets Devices to align; duplicate pointers are collapsed while
 *                retaining their first label. Distinct devices must resolve
 *                to external or gpsdo so they share a common PPS epoch.
 * @param options Poll/timeout durations, exposed mainly for deterministic tests.
 * @return Per-device post-alignment timestamps in native master-clock ticks.
 * @throws std::runtime_error if distinct devices do not share a common PPS
 *         epoch, PPS is unsupported or absent, or verification fails.
 */
inline PpsTimeAlignmentResult align_device_times_to_next_pps(
    const std::vector<PpsTimeAlignmentTarget>& targets,
    const PpsTimeAlignmentOptions& options = {})
{
    std::vector<PpsTimeAlignmentTarget> unique;
    unique.reserve(targets.size());
    for (const auto& target : targets) {
        if (!target.device) {
            continue;
        }
        bool duplicate = false;
        for (const auto& existing : unique) {
            if (existing.device.get() == target.device.get()) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            unique.push_back(target);
        }
    }
    if (unique.empty()) {
        throw std::runtime_error("PPS time alignment has no devices");
    }
    if (unique.size() > 1) {
        for (const auto& target : unique) {
            if (target.time_source != "external" &&
                target.time_source != "gpsdo") {
                const std::string resolved_source =
                    target.time_source.empty() ? "<empty>" : target.time_source;
                throw std::runtime_error(
                    "Multi-device PPS time alignment requires every distinct "
                    "USRP to use a shared time_source of external or gpsdo; "
                    "target '" + target.label + "' resolved to '" +
                    resolved_source + "'");
            }
        }
    }
    for (const auto& target : unique) {
        if (!target.device->supports(Capability::PpsTimeSync)) {
            throw std::runtime_error(
                "PPS time alignment is unsupported by " + target.label);
        }
    }

    std::vector<double> tick_rates;
    tick_rates.reserve(unique.size());
    for (const auto& target : unique) {
        const double tick_rate = target.device->master_clock_rate();
        if (!(tick_rate > 0.0) || !std::isfinite(tick_rate)) {
            throw std::runtime_error(
                "PPS time alignment has invalid master clock rate on " +
                target.label);
        }
        tick_rates.push_back(tick_rate);
    }
    const double authority_tick_rate = tick_rates.front();
    const int64_t initial_last_pps =
        unique.front().device->time_last_pps().to_ticks(authority_tick_rate);
    const auto edge_deadline =
        std::chrono::steady_clock::now() + options.edge_timeout;
    bool observed_edge = false;
    while (std::chrono::steady_clock::now() < edge_deadline) {
        if (options.poll_interval.count() > 0) {
            std::this_thread::sleep_for(options.poll_interval);
        }
        const int64_t current_last_pps =
            unique.front().device->time_last_pps().to_ticks(authority_tick_rate);
        if (current_last_pps != initial_last_pps) {
            observed_edge = true;
            break;
        }
        if (options.poll_interval.count() <= 0) {
            std::this_thread::yield();
        }
    }
    if (!observed_edge) {
        throw std::runtime_error(
            "timed out waiting for PPS transition on " + unique.front().label);
    }

    // Sample before arming: the load can happen while a control call is in flight.
    std::vector<int64_t> previous_now_ticks;
    previous_now_ticks.reserve(unique.size());
    for (size_t i = 0; i < unique.size(); ++i) {
        previous_now_ticks.push_back(unique[i].device->time_now().to_ticks(tick_rates[i]));
    }
    for (const auto& target : unique) {
        target.device->set_time_next_pps(TimeSpec(0.0));
    }

    std::vector<bool> observed_time_reset(unique.size(), false);
    auto verify_deadline =
        std::chrono::steady_clock::now() + options.apply_timeout;
    PpsTimeAlignmentResult load_edge;
    PpsTimeAlignmentResult last_result;
    while (std::chrono::steady_clock::now() < verify_deadline) {
        if (options.poll_interval.count() > 0) {
            std::this_thread::sleep_for(options.poll_interval);
        }

        PpsTimeAlignmentResult current;
        current.devices.reserve(unique.size());
        bool all_reset = true;
        bool all_target_pps_zero = true;
        const bool waiting_for_post_load = !load_edge.devices.empty();
        bool all_post_load_pps = waiting_for_post_load;
        double min_pps_s = std::numeric_limits<double>::infinity();
        double max_pps_s = -std::numeric_limits<double>::infinity();
        double min_now_s = std::numeric_limits<double>::infinity();
        double max_now_s = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < unique.size(); ++i) {
            const auto& target = unique[i];
            const double tick_rate = tick_rates[i];
            PpsTimeAlignmentSnapshot snapshot;
            snapshot.label = target.label;
            snapshot.tick_rate = tick_rate;
            snapshot.now_ticks = target.device->time_now().to_ticks(tick_rate);
            if (snapshot.now_ticks < previous_now_ticks[i]) {
                if (observed_time_reset[i]) {
                    throw std::runtime_error(
                        "device time reset again during PPS alignment on " + target.label);
                }
                observed_time_reset[i] = true;
            }
            previous_now_ticks[i] = snapshot.now_ticks;
            snapshot.last_pps_ticks =
                target.device->time_last_pps().to_ticks(tick_rate);
            all_reset = all_reset && observed_time_reset[i];
            all_target_pps_zero =
                all_target_pps_zero && snapshot.last_pps_ticks == 0;
            if (waiting_for_post_load) {
                const auto latch = snapshot.last_pps_ticks;
                all_post_load_pps = all_post_load_pps &&
                    latch != load_edge.devices[i].last_pps_ticks &&
                    latch > 0 && latch <= snapshot.now_ticks &&
                    static_cast<double>(snapshot.now_ticks - latch) < tick_rate;
                const double pps_s = static_cast<double>(latch) / tick_rate;
                min_pps_s = std::min(min_pps_s, pps_s);
                max_pps_s = std::max(max_pps_s, pps_s);
            }
            const double now_s =
                static_cast<double>(snapshot.now_ticks) / tick_rate;
            min_now_s = std::min(min_now_s, now_s);
            max_now_s = std::max(max_now_s, now_s);
            current.devices.push_back(std::move(snapshot));
        }

        const double max_skew_s = std::chrono::duration<double>(
            options.max_device_time_skew).count();
        if (all_reset && max_now_s <= min_now_s + max_skew_s) {
            if ((!waiting_for_post_load && all_target_pps_zero) ||
                (all_post_load_pps && max_pps_s <= min_pps_s + max_skew_s)) {
                return current;
            }
            if (!waiting_for_post_load && min_now_s >= 0.0 && max_now_s < 1.0) {
                load_edge = current;
                // If the old latch equals the first post-load latch, only the
                // second post-load edge produces an observable change.
                verify_deadline = std::chrono::steady_clock::now() + 2 * options.edge_timeout;
            }
        }
        last_result = std::move(current);
        if (options.poll_interval.count() <= 0) {
            std::this_thread::yield();
        }
    }

    std::string detail;
    for (size_t i = 0; i < last_result.devices.size(); ++i) {
        const auto& snapshot = last_result.devices[i];
        detail += (i == 0 ? ": " : ", ") + snapshot.label +
            "{reset=" + (observed_time_reset[i] ? "yes" : "no") +
            ", last_pps_ticks=" + std::to_string(snapshot.last_pps_ticks) +
            ", now_ticks=" + std::to_string(snapshot.now_ticks) + "}";
    }
    throw std::runtime_error(
        (load_edge.devices.empty() ? "timed out verifying PPS time alignment"
                                   : "timed out verifying post-load PPS transition") + detail);
}

/**
 * @brief Return whether radio startup should align device time on PPS.
 *
 * Clock selection is the alignment enable: internal, external, and gpsdo
 * clocks all have a PPS epoch that must be established before streaming.
 * The time source selects which PPS drives the device, but does not disable
 * startup alignment.
 *
 * @throws std::runtime_error if clock_source is not empty and is not one of
 *         internal, external, or gpsdo.
 */
inline bool clock_source_enables_pps_alignment(const std::string& clock_source) {
    if (clock_source.empty()) {
        return false;
    }
    if (clock_source == "internal" || clock_source == "external" ||
        clock_source == "gpsdo") {
        return true;
    }
    throw std::runtime_error(
        "Invalid clock_source='" + clock_source +
        "'. Supported values: internal, external, gpsdo");
}

}  // namespace radio

#endif  // PPS_TIME_ALIGNMENT_HPP
