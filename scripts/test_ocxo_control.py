#!/usr/bin/env python3
"""Build and run deterministic OCXO controller helper tests."""

import subprocess
import sys
import tempfile
from pathlib import Path


TEST_SOURCE = r"""
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

#include "OFDMCore.hpp"

static int passed = 0;
static int failed = 0;

#define CHECK_NEAR(actual, expected, tol, message) do { \
    const double actual_value = static_cast<double>(actual); \
    const double expected_value = static_cast<double>(expected); \
    if (std::abs(actual_value - expected_value) > (tol)) { \
        std::fprintf(stderr, "FAIL: %s: actual=%.12g expected=%.12g\n", \
                     message, actual_value, expected_value); \
        ++failed; \
    } else { \
        ++passed; \
    } \
} while (0)

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL: %s\n", message); \
        ++failed; \
    } else { \
        ++passed; \
    } \
} while (0)

void test_ewma_control_step_polarity() {
    CfoErrorEwma::Params params;
    params.enable = true;
    params.alpha = 0.02;
    CfoErrorEwma ewma(params);

    CHECK_NEAR(ewma.update(1.0).filtered_error_ppm, 1.0, 1e-12,
               "EWMA initializes from residual");
    ewma.notify_control_action(+0.1);
    CHECK_NEAR(ewma.update(0.9).filtered_error_ppm, 0.9, 1e-12,
               "positive actuator step predicts negative residual step");
    ewma.notify_control_action(-0.2);
    CHECK_NEAR(ewma.update(1.1).filtered_error_ppm, 1.1, 1e-12,
               "negative actuator step predicts positive residual step");
}

void test_normal_cfo_dealiasing() {
    CHECK_NEAR(rx_tune_system_cfo_hz(100.0, 98.0, 3.0), 5.0, 1e-12,
               "positive RX DSP uses target - (RF - DSP)");
    CHECK_NEAR(rx_tune_system_cfo_hz(100.0, 98.0, -3.0), -1.0, 1e-12,
               "negative RX DSP uses target - (RF - DSP)");
    const auto positive = dealias_normal_cfo_hz(-100.0, 1000.0, 900.0);
    CHECK(positive.valid && positive.alias_index == 1,
          "normal CFO selects positive adjacent branch");
    CHECK_NEAR(positive.dealiased_hz, 900.0, 1e-12,
               "normal CFO positive branch value");
    const auto negative = dealias_normal_cfo_hz(100.0, 1000.0, -900.0);
    CHECK(negative.valid && negative.alias_index == -1,
          "normal CFO selects negative adjacent branch");
    CHECK_NEAR(negative.dealiased_hz, -900.0, 1e-12,
               "normal CFO negative branch value");

    NormalCfoQualityGate gate;
    NormalCfoDealiasResult estimate;
    estimate.valid = true;
    estimate.dealiased_hz = 100.0;
    CHECK(!gate.update(estimate, 1000.0),
          "CFO quality gate rejects the first cold-start sample");
    estimate.dealiased_hz = 101.0;
    CHECK(!gate.update(estimate, 1000.0),
          "CFO quality gate keeps warming on the second sample");
    estimate.dealiased_hz = 99.0;
    CHECK(gate.update(estimate, 1000.0),
          "CFO quality gate accepts a continuous three-sample run");
    estimate.dealiased_hz = 500.0;
    CHECK(!gate.update(estimate, 1000.0),
          "branch-scale CFO jump restarts the quality gate");
}

void test_cfo_acquisition_to_tracking() {
    constexpr double period = 100e6 / 1152.0;
    const auto modulo = [](double hz) { return std::remainder(hz, period); };
    // Reproduce the observed failure: after correcting -142807 Hz, the old
    // tune-only reference adds two alias periods to a 309 Hz residual.
    const auto old = dealias_normal_cfo_hz(309.0, period, 142807.0);
    CHECK_NEAR(old.dealiased_hz, 309.0 + 2.0 * period, 1e-7,
               "tune-only reference reproduces erroneous 173.9 kHz correction");

    for (const double physical_cfo : {-142807.0, 142807.0, -1861000.0, 1861000.0}) {
        NormalCfoTracker tracker;
        double tune = -4020.0; // nonzero initial UHD tune quantization
        double physical = physical_cfo;
        const double acquired = physical + tune;
        tracker.acquire(acquired, tune);
        // Before any correction, retain the full CFO (hardware-only tracking).
        auto estimate = tracker.resolve(modulo(acquired), period, tune, true);
        CHECK(estimate.valid, "full acquisition initializes normal tracking");
        CHECK_NEAR(estimate.dealiased_hz, acquired, 1e-7,
                   "uncorrected large CFO retains its acquired branch");
        tune -= acquired; // initial DSP correction
        physical += 309.0;
        // Old pipeline results must not move the post-retune reference.
        CHECK(!tracker.resolve(period * .49, period, tune, false).valid,
              "stale result is rejected without changing the CFO anchor");
        estimate = tracker.resolve(modulo(physical + tune), period, tune, true);
        CHECK_NEAR(estimate.dealiased_hz, 309.0, 1e-7,
                   "post-acquisition residual stays near zero for both CFO signs");
        for (int i = 0; i < 30; ++i) {
            // Actual retune may differ from the requested value. The next
            // observation must use the returned tune, not assume exact zero.
            tune -= estimate.dealiased_hz - 0.25;
            physical += (i % 2 ? -37.0 : 51.0);
            estimate = tracker.resolve(modulo(physical + tune), period, tune, true);
            CHECK_NEAR(estimate.dealiased_hz, physical + tune, 1e-7,
                       "repeated corrections track actual small residuals");
        }
        // Drift across multiple wraps with software correction disabled.
        for (int i = 0; i < 20; ++i) {
            tracker.acquire(estimate.dealiased_hz, tune);
            physical += period * .2;
            estimate = tracker.resolve(modulo(physical + tune), period, tune, true);
            CHECK_NEAR(estimate.dealiased_hz, physical + tune, 1e-7,
                       "continuous oscillator drift crosses modulo boundaries");
        }
        tracker.reset();
        CHECK(!tracker.resolve(0.0, period, tune, true).valid,
              "reset requires a new full acquisition");
        tracker.acquire(-physical, 0.0);
        estimate = tracker.resolve(modulo(-physical), period, 0.0, true);
        CHECK_NEAR(estimate.dealiased_hz, -physical, 1e-7,
                   "reacquisition discards the old sign and branch");
    }

    NormalCfoTracker noisy;
    noisy.acquire(0.0, 0.0);
    for (int i = 0; i < 20; ++i) {
        const double residual = (i % 2 ? -.49 : .49) * period;
        const auto sample = noisy.resolve(residual, period, 0.0, true);
        CHECK_NEAR(sample.dealiased_hz, residual, 1e-7,
                   "noisy fresh frames cannot walk the alias reference");
    }

    NormalCfoTracker gated;
    gated.acquire(0.0, 0.0);
    DemodControlTimeGates gates;
    gates.mark_freq_adjust(1000);
    CHECK(!gated.resolve(period * .49, period, 0.0,
                        gates.allow_freq_adjust(999)).valid,
          "timestamp gate rejects pre-retune IQ");
    const auto fresh = gated.resolve(-period * .1, period, 0.0,
                                    gates.allow_freq_adjust(1001));
    CHECK_NEAR(fresh.dealiased_hz, -period * .1, 1e-7,
               "stale IQ cannot drag a fresh observation onto another branch");
}

void test_ocxo_debug_packet_layout() {
    ocxo_debug_protocol::Channels channels{};
    for (size_t i = 0; i < channels.size(); ++i) {
        channels[i] = static_cast<float>(i) + 0.25f;
    }
    ocxo_debug_protocol::Diagnostics diagnostics;
    diagnostics.r_bs_samples = 12.25;
    diagnostics.r_ue_samples = 34.5;
    diagnostics.bs_last_pps_ticks = 9876543210LL;
    diagnostics.bs_t0_to_pps_ticks = 123456;
    diagnostics.ue_last_pps_ns = 8765432100LL;
    diagnostics.bs_tick_rate = 200.0e6;
    diagnostics.sample_rate = 50.0e6;
    const auto packet = ocxo_debug_protocol::make_packet(
        channels, diagnostics, 42, 123456789);
    CHECK(packet.size() == ocxo_debug_protocol::kPacketSize,
          "OCXO debug packet has the fixed protocol size");
    CHECK(std::equal(
              ocxo_debug_protocol::kMagic.begin(),
              ocxo_debug_protocol::kMagic.end(),
              packet.begin()),
          "OCXO debug packet carries the OCXODBG3 magic");
    const auto load_u32_le = [&packet](size_t offset) {
        uint32_t value = 0;
        for (size_t i = 0; i < sizeof(value); ++i) {
            value |= static_cast<uint32_t>(packet[offset + i]) << (8 * i);
        }
        return value;
    };
    const auto load_u64_le = [&packet](size_t offset) {
        uint64_t value = 0;
        for (size_t i = 0; i < sizeof(value); ++i) {
            value |= static_cast<uint64_t>(packet[offset + i]) << (8 * i);
        }
        return value;
    };
    const auto load_i64_le = [&load_u64_le](size_t offset) {
        return static_cast<int64_t>(load_u64_le(offset));
    };
    const auto load_f64_le = [&load_u64_le](size_t offset) {
        const uint64_t bits = load_u64_le(offset);
        double value = 0.0;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };
    CHECK(load_u32_le(8) == ocxo_debug_protocol::kVersion,
          "OCXO debug packet version is encoded little-endian");
    CHECK(load_u32_le(12) == ocxo_debug_protocol::kChannelCount,
          "OCXO debug packet declares all 16 channels");
    CHECK(load_u64_le(16) == 42 && load_u64_le(24) == 123456789,
          "OCXO debug packet sequence and timestamp match the source");
    const uint32_t first_bits = load_u32_le(ocxo_debug_protocol::kHeaderSize);
    float first_value = 0.0f;
    std::memcpy(&first_value, &first_bits, sizeof(first_value));
    CHECK_NEAR(first_value, 0.25, 1e-7,
               "OCXO debug channel payload is encoded as little-endian float32");
    const size_t diag_off = ocxo_debug_protocol::kHeaderSize +
        ocxo_debug_protocol::kChannelCount * sizeof(float);
    CHECK_NEAR(load_f64_le(diag_off), 12.25, 1e-12,
               "OCXO debug r_B keeps float64 precision");
    CHECK_NEAR(load_f64_le(diag_off + 8), 34.5, 1e-12,
               "OCXO debug r_U keeps float64 precision");
    CHECK(load_i64_le(diag_off + 16) == 9876543210LL &&
              load_i64_le(diag_off + 24) == 123456 &&
              load_i64_le(diag_off + 32) == 8765432100LL,
          "OCXO debug PPS timestamps keep exact int64 values");
    CHECK_NEAR(load_f64_le(diag_off + 40), 200.0e6, 1e-6,
               "OCXO debug carries the BS tick rate");
    CHECK_NEAR(load_f64_le(diag_off + 48), 50.0e6, 1e-6,
               "OCXO debug carries the sample rate");
}

void test_fll_velocity_pi_polarity() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Freq;
    params.max_abs_u_ppm = 10.0;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 0.1;
    params.fll_fast_kp = 0.0;
    params.fll_fast_ki = 1.0;
    params.fll_fast_max_slew_ppm_per_s = 100.0;
    params.fll_slow_enter_abs_ppm = 0.0;

    const auto command_for_residual = [&params](double residual_ppm) {
        OCXODiscipliner loop(params);
        OCXODiscipliner::Observation obs;
        obs.allow_control = true;
        obs.cfo_valid = true;
        obs.cfo_ppm = residual_ppm;
        const auto t0 = std::chrono::steady_clock::time_point{};
        return loop.update_at(obs, t0);
    };

    const auto positive = command_for_residual(1.0);
    CHECK_NEAR(positive.desired_delta_ppm, 0.1, 1e-12,
               "Mode A CFO positive residual produces positive OCXO step");
    const auto negative = command_for_residual(-1.0);
    CHECK_NEAR(negative.desired_delta_ppm, -0.1, 1e-12,
               "Mode A CFO negative residual produces negative OCXO step");
}

void test_fll_fast_half_step_control() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Freq;
    params.max_abs_u_ppm = 10.0;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 0.5;
    params.fll_fast_control_mode =
        OCXODiscipliner::FllFastControlMode::HalfStep;
    params.fll_fast_kp = 100.0;
    params.fll_fast_ki = 100.0;
    params.fll_fast_max_slew_ppm_per_s = 100.0;
    params.fll_slow_enter_abs_ppm = 0.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    obs.cfo_ppm = 0.2;
    const auto t0 = std::chrono::steady_clock::time_point{};
    const auto first = loop.update_at(obs, t0);
    CHECK_NEAR(first.desired_delta_ppm, 0.1, 1e-12,
               "Fast half-step requests half the positive frequency error");
    loop.commit_applied_delta(first.desired_delta_ppm);

    const auto early = loop.update_at(
        obs, t0 + std::chrono::milliseconds(499));
    CHECK(!early.request_control,
          "Fast half-step waits for its configured half-second period");
    const auto second = loop.update_at(
        obs, t0 + std::chrono::milliseconds(500));
    CHECK_NEAR(second.desired_delta_ppm, 0.1, 1e-12,
               "Fast half-step does not use PI error history");

    OCXODiscipliner negative_loop(params);
    obs.cfo_ppm = -0.2;
    const auto negative = negative_loop.update_at(obs, t0);
    CHECK_NEAR(negative.desired_delta_ppm, -0.1, 1e-12,
               "Fast half-step preserves frequency-control polarity");

    CHECK(normalize_ocxo_fll_fast_control_mode_string("HALF_STEP") ==
              kOCXOFllFastControlHalfStep,
          "Fast control mode parsing is case-insensitive");
    bool rejected = false;
    try {
        (void)normalize_ocxo_fll_fast_control_mode_string("pid");
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    CHECK(rejected, "unknown Fast control modes are rejected");
}

void test_mode_b_fll_uses_to_slope_not_cfo() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Phase;
    params.max_abs_u_ppm = 10.0;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 0.1;
    params.fll_fast_kp = 0.0;
    params.fll_fast_ki = 1.0;
    params.fll_fast_max_slew_ppm_per_s = 100.0;
    params.fll_slow_enter_abs_ppm = 0.0;
    params.samples_per_frame = 1000.0;
    params.phase_jump_gate_samples = 1.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    obs.cfo_ppm = -5.0;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_ue_samples = 0.0;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);
    obs.to_ue_samples = 0.0005;
    obs.to_frame_seq = 2;
    const auto out = loop.update_at(obs, t0 + std::chrono::milliseconds(100));
    CHECK(out.frequency_source == OCXODiscipliner::FrequencySource::ToSlope,
          "Mode B selects TO_clock slope as its SFO observation");
    // Control uses -to_slope so positive measured slope (UE TO walking up)
    // produces a negative OCXO step — matches bench UE-local OCXO plant.
    CHECK_NEAR(out.desired_delta_ppm, -0.05, 1e-12,
               "Mode B control uses inverted TO_clock slope polarity");
    CHECK_NEAR(out.to_slope_ppm, 0.5, 1e-12,
               "debug to_slope_ppm remains the raw regression slope");
    CHECK_NEAR(out.frequency_error_ppm, -0.5, 1e-12,
               "frequency_error_ppm is the plant-oriented Mode B control error");
}

void test_mode_b_fll_closed_loop_converges_with_positive_plant_gain() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Phase;
    params.max_abs_u_ppm = 10.0;
    params.coarse_enable = false;
    params.samples_per_frame = 1.0e6;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 1.0;
    params.fll_fast_kp = 0.0;
    params.fll_fast_ki = 0.25;
    params.fll_fast_max_slew_ppm_per_s = 10.0;
    params.fll_slow_enter_abs_ppm = 0.0;
    params.fll_switch_hold_s = 1000.0;
    params.phase_jump_gate_samples = 10.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    obs.to_ue_samples = 0.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);

    // Positive OCXO command raises the UE clock frequency and therefore raises
    // raw TO_clock slope. The negative Mode B control error must converge to
    // the command that cancels the uncontrolled positive clock offset.
    constexpr double uncontrolled_clock_error_ppm = 1.0;
    OCXODiscipliner::UpdateResult out;
    for (uint64_t second = 1; second <= 40; ++second) {
        const double raw_to_slope_ppm =
            uncontrolled_clock_error_ppm + loop.command_ppm();
        obs.to_ue_samples += raw_to_slope_ppm;
        obs.to_frame_seq = second + 1;
        out = loop.update_at(obs, t0 + std::chrono::seconds(second));
        if (out.request_control) {
            loop.commit_applied_delta(out.desired_delta_ppm);
        }
    }

    CHECK(loop.stage() == OCXODiscipliner::Stage::FllFast,
          "Mode B FLL closed-loop test stays in fast FLL");
    CHECK_NEAR(loop.command_ppm(), -uncontrolled_clock_error_ppm, 2e-5,
               "Mode B FLL cancels a positive-gain UE clock offset");
    CHECK_NEAR(out.to_slope_ppm, 0.0, 3e-5,
               "Mode B FLL drives raw TO_clock slope to zero");
}

void test_to_slope_uses_window_regression() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Phase;
    params.coarse_enable = false;
    params.samples_per_frame = 1000.0;
    params.fll_fast_obs_window_s = 2.0;
    params.fll_fast_control_period_s = 0.1;
    params.fll_slow_enter_abs_ppm = 0.0;
    params.phase_jump_gate_samples = 1.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    obs.to_ue_samples = 0.0;
    obs.pps_phase_valid = true;
    obs.pps_phase_error_samples = 0.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);

    obs.to_frame_seq = 101;
    obs.to_ue_samples = 0.1;
    const auto warming = loop.update_at(obs, t0 + std::chrono::seconds(1));
    CHECK(!warming.to_slope_valid,
          "TO slope waits for the configured regression window");

    obs.to_frame_seq = 201;
    obs.to_ue_samples = 0.1;
    const auto estimated = loop.update_at(obs, t0 + std::chrono::seconds(2));
    CHECK(estimated.to_slope_valid,
          "TO slope becomes valid after full window coverage");
    CHECK_NEAR(estimated.to_slope_ppm, 0.5, 1e-12,
               "TO slope uses window regression instead of the last zero delta");

    obs.to_frame_seq = 401;
    obs.to_ue_samples = 0.2;
    const auto irregular = loop.update_at(obs, t0 + std::chrono::seconds(4));
    CHECK_NEAR(irregular.to_slope_ppm, 0.5, 1e-12,
               "TO regression weights irregular frame gaps by frame sequence");
}

void test_phase_velocity_pi_closed_loop_converges() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::PpsPhase;
    params.max_abs_u_ppm = 10.0;
    params.coarse_enable = false;
    params.samples_per_frame = 1.0e6;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 1.0;
    params.fll_fast_kp = 0.0;
    params.fll_fast_ki = 0.0;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 1.0;
    params.fll_slow_kp = 0.0;
    params.fll_slow_ki = 0.0;
    params.fll_slow_enter_abs_ppm = 10.0;
    params.fll_fast_enter_abs_ppm = 20.0;
    params.fll_switch_hold_s = 0.0;
    params.phase_enter_sfo_abs_ppm = 10.0;
    params.phase_enter_hold_s = 0.0;
    params.phase_exit_sfo_abs_ppm = 100.0;
    params.phase_exit_hold_s = 1.0;
    params.phase_control_period_s = 1.0;
    params.phase_jump_gate_samples = 10.0;
    params.phase_kp = 0.5;
    params.phase_ki = 0.1;
    params.phase_max_slew_ppm_per_s = 10.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    obs.to_ue_samples = 0.0;
    obs.pps_phase_valid = true;
    obs.pps_phase_error_samples = 0.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);

    double to_clock = 0.0;
    constexpr double uncontrolled_clock_error_ppm = 1.0;
    for (uint64_t second = 1; second <= 2; ++second) {
        const double clock_error_ppm =
            uncontrolled_clock_error_ppm + loop.command_ppm();
        to_clock += clock_error_ppm;
        obs.to_ue_samples = to_clock;
        obs.pps_phase_error_samples = -to_clock;
        obs.to_frame_seq = second + 1;
        loop.update_at(obs, t0 + std::chrono::seconds(second));
    }
    CHECK(loop.stage() == OCXODiscipliner::Stage::Phase,
          "quiet clock slope reaches Phase before closed-loop simulation");

    // Model the one-shot AP coarse placement at Phase entry. TO_clock remains
    // untouched; only the external PPS phase error is brought near zero.
    double pps_phase_error = 0.0;
    double max_abs_error = 0.0;
    OCXODiscipliner::UpdateResult out;
    for (uint64_t second = 3; second <= 32; ++second) {
        const double clock_error_ppm =
            uncontrolled_clock_error_ppm + loop.command_ppm();
        to_clock += clock_error_ppm;
        pps_phase_error -= clock_error_ppm;
        obs.to_ue_samples = to_clock;
        obs.pps_phase_error_samples = pps_phase_error;
        obs.to_frame_seq = second + 1;
        out = loop.update_at(obs, t0 + std::chrono::seconds(second));
        max_abs_error = std::max(max_abs_error, std::abs(out.to_err));
        if (out.request_control) {
            loop.commit_applied_delta(out.desired_delta_ppm);
        }
    }
    CHECK(max_abs_error < 2.0,
          "Phase velocity PI remains bounded without growing oscillation");
    CHECK_NEAR(loop.command_ppm(), -uncontrolled_clock_error_ppm, 2e-4,
               "Phase velocity PI converges to the plant frequency bias");
    CHECK_NEAR(out.to_err, 0.0, 5e-4,
               "Phase velocity PI drives PPS phase error toward zero");
}

void test_phase_default_gain_tuning() {
    const OCXODiscipliner::Params defaults;
    CHECK_NEAR(defaults.fll_slow_kp, 0.1, 1e-15,
               "controller default uses the tuned slow FLL proportional gain");
    CHECK_NEAR(defaults.fll_slow_ki, 0.007, 1e-15,
               "controller default uses the tuned slow FLL integral gain");
    CHECK_NEAR(defaults.fll_slow_max_slew_ppm_per_s, 0.005, 1e-15,
               "controller default uses the tuned slow FLL slew limit");
    CHECK_NEAR(defaults.fll_slow_enter_abs_ppm, 0.0005, 1e-15,
               "controller default uses the tuned fast-to-slow threshold");
    CHECK_NEAR(defaults.phase_kp, 2.0e-4, 1e-15,
               "controller default uses the measured Phase damping gain");
    CHECK_NEAR(defaults.fll_slow_deadzone_ppm, 5.0e-5, 1e-15,
               "controller default keeps slow FLL active below Phase entry");
    CHECK_NEAR(defaults.phase_enter_sfo_abs_ppm, 2.0e-4, 1e-15,
               "controller default tightens the Phase entry slope");
    const Config cfg = make_default_ue_config();
    CHECK_NEAR(cfg.sync_tracking.ocxo_fll_slow_kp, 0.1, 1e-15,
               "UE config default uses the tuned slow FLL proportional gain");
    CHECK_NEAR(cfg.sync_tracking.ocxo_fll_slow_ki, 0.007, 1e-15,
               "UE config default uses the tuned slow FLL integral gain");
    CHECK_NEAR(cfg.sync_tracking.ocxo_fll_slow_max_slew_ppm_per_s, 0.005, 1e-15,
               "UE config default uses the tuned slow FLL slew limit");
    CHECK_NEAR(cfg.sync_tracking.ocxo_fll_slow_enter_abs_ppm, 0.0005, 1e-15,
               "UE config default uses the tuned fast-to-slow threshold");
    CHECK(cfg.sync_tracking.ocxo_pps_initial_ap_enable,
          "UE config enables the initial PPS AP correction by default");
    CHECK(cfg.sync_tracking.ocxo_pps_initial_ap_polarity == 1,
          "UE config uses the bench-confirmed PPS AP polarity");
    CHECK(cfg.sync_tracking.ocxo_dac_state_file ==
              ".openisac_ocxo_dac_state.yaml",
          "UE config persists the last acknowledged DAC word by default");
    CHECK_NEAR(cfg.sync_tracking.ocxo_phase_kp, 2.0e-4, 1e-15,
               "UE config default uses the measured Phase damping gain");
    CHECK_NEAR(cfg.sync_tracking.ocxo_fll_slow_deadzone_ppm, 5.0e-5, 1e-15,
               "UE config default lowers the slow FLL deadzone");
    CHECK_NEAR(cfg.sync_tracking.ocxo_phase_enter_sfo_abs_ppm, 2.0e-4, 1e-15,
               "UE config default tightens the Phase entry slope");
    const auto mapped = make_ocxo_discipliner_params(cfg);
    CHECK_NEAR(mapped.fll_slow_kp, 0.1, 1e-15,
               "UE config maps the tuned slow FLL proportional gain");
    CHECK_NEAR(mapped.fll_slow_ki, 0.007, 1e-15,
               "UE config maps the tuned slow FLL integral gain");
    CHECK_NEAR(mapped.fll_slow_max_slew_ppm_per_s, 0.005, 1e-15,
               "UE config maps the tuned slow FLL slew limit");
    CHECK_NEAR(mapped.fll_slow_enter_abs_ppm, 0.0005, 1e-15,
               "UE config maps the tuned fast-to-slow threshold");
    CHECK_NEAR(mapped.phase_kp, 2.0e-4, 1e-15,
               "UE config maps the tuned Phase gain into the controller");
    CHECK(mapped.fll_slow_deadzone_ppm < mapped.phase_enter_sfo_abs_ppm,
          "slow FLL deadzone stays below the Phase entry threshold");
    const auto ap = make_ocxo_pps_initial_aligner_params(cfg);
    CHECK(ap.enable && ap.command_polarity == 1,
          "UE config maps the enabled bench-confirmed initial AP polarity");
}

void test_discipliner_reset_retains_dac_command() {
    OCXODiscipliner loop;
    loop.commit_applied_delta(0.123);
    loop.reset();
    CHECK_NEAR(loop.command_ppm(), 0.123, 1e-12,
               "default tracking reset retains the current DAC command");
    CHECK(loop.stage() == OCXODiscipliner::Stage::Coarse,
          "tracking reset still restarts the observation stages");
}

void test_dac_word_persists_across_controller_restart() {
    const int master_fd = posix_openpt(O_RDWR | O_NOCTTY | O_NONBLOCK);
    CHECK(master_fd >= 0, "allocate pseudo-terminal for DAC persistence test");
    if (master_fd < 0) {
        return;
    }
    const bool pty_ready =
        grantpt(master_fd) == 0 && unlockpt(master_fd) == 0;
    CHECK(pty_ready, "initialize DAC persistence pseudo-terminal");
    if (!pty_ready) {
        close(master_fd);
        return;
    }
    char* slave_name = ptsname(master_fd);
    CHECK(slave_name != nullptr, "resolve DAC persistence slave path");
    if (slave_name == nullptr) {
        close(master_fd);
        return;
    }

    char state_template[] = "/tmp/openisac-ocxo-state-XXXXXX";
    const int state_fd = mkstemp(state_template);
    CHECK(state_fd >= 0, "allocate DAC persistence state path");
    if (state_fd < 0) {
        close(master_fd);
        return;
    }
    close(state_fd);
    unlink(state_template);
    const std::string state_path(state_template);

    std::atomic<bool> stop{false};
    std::atomic<size_t> sf_count{0};
    std::atomic<size_t> zero_word_count{0};
    std::atomic<int32_t> last_sf_word{-1};
    std::atomic<bool> rejected_unacknowledged_word{false};
    constexpr int32_t kRetainedWord = 593367;
    constexpr int32_t kUnacknowledgedWord = 643367;
    std::thread responder([&] {
        std::string pending;
        char buffer[128];
        while (!stop.load(std::memory_order_acquire)) {
            const ssize_t n = read(master_fd, buffer, sizeof(buffer));
            if (n > 0) {
                pending.append(buffer, static_cast<size_t>(n));
                size_t newline = std::string::npos;
                while ((newline = pending.find('\n')) != std::string::npos) {
                    const std::string command = pending.substr(0, newline + 1);
                    pending.erase(0, newline + 1);
                    if (command.rfind("SF", 0) == 0 && command.size() >= 9) {
                        const int32_t word = static_cast<int32_t>(
                            std::strtol(command.substr(2, 7).c_str(), nullptr, 10));
                        last_sf_word.store(word, std::memory_order_relaxed);
                        sf_count.fetch_add(1, std::memory_order_relaxed);
                        if (word == 493367) {
                            zero_word_count.fetch_add(1, std::memory_order_relaxed);
                        }
                        if (word == kUnacknowledgedWord &&
                            !rejected_unacknowledged_word.exchange(true)) {
                            continue;
                        }
                    }
                    const char response[] = "OK\n";
                    const ssize_t written =
                        write(master_fd, response, sizeof(response) - 1);
                    (void)written;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    });

    const auto wait_for = [](const auto& predicate) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!predicate() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return predicate();
    };

    {
        HardwareSyncController controller(slave_name);
        controller.configure_plant(
            1.0e-6, 493367, 0.45, 0.0, state_path);
        CHECK(wait_for([&] { return sf_count.load() >= 1; }),
              "initial DAC word is acknowledged");
        CHECK_NEAR(controller.apply_delta_ppm(0.1), 0.1, 1e-12,
                   "DAC persistence test applies the requested command");
        CHECK(wait_for([&] {
                  if (sf_count.load() < 2 || access(state_path.c_str(), F_OK) != 0) {
                      return false;
                  }
                  try {
                      return YAML::LoadFile(state_path)["word"].as<int32_t>() ==
                          kRetainedWord;
                  } catch (...) {
                      return false;
                  }
              }),
              "last acknowledged DAC word is persisted");
        CHECK_NEAR(controller.apply_delta_ppm(0.05), 0.05, 1e-12,
                   "queue a newer DAC command for shutdown filtering");
        CHECK(wait_for([&] {
                  return rejected_unacknowledged_word.load();
              }),
              "simulate a final DAC command without an OK response");
    }

    CHECK(YAML::LoadFile(state_path)["word"].as<int32_t>() == kRetainedWord,
          "shutdown saves the last acknowledged stable word, not an unconfirmed command");

    const size_t zero_count_before_restart = zero_word_count.load();
    const size_t sf_count_before_restart = sf_count.load();
    {
        HardwareSyncController controller(slave_name);
        controller.configure_plant(
            1.0e-6, 493367, 0.45, 0.0, state_path);
        CHECK(controller.current_word() == kRetainedWord,
              "controller restart restores the last acknowledged DAC word");
        CHECK_NEAR(controller.current_offset_ppm(), 0.1, 1e-12,
                   "controller restart restores the matching ppm command");
        CHECK(wait_for([&] {
                  return sf_count.load() > sf_count_before_restart;
              }),
              "controller restart reasserts the retained DAC word");
        CHECK(last_sf_word.load() == kRetainedWord,
              "restart sends the retained DAC word instead of starting over");
        CHECK(zero_word_count.load() == zero_count_before_restart,
              "restart never emits the calibrated zero word");
    }

    const auto write_state = [&](const std::string& content) {
        std::ofstream output(state_path, std::ios::out | std::ios::trunc);
        output << content;
        output.close();
        return static_cast<bool>(output);
    };
    const auto read_state = [&] {
        std::ifstream input(state_path);
        return std::string(
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>());
    };
    const auto verify_invalid_state_is_preserved = [&]
        (const std::string& content, const char* setup_message,
         const char* fallback_message, const char* preserve_message) {
        CHECK(write_state(content), setup_message);
        const size_t before = sf_count.load();
        {
            HardwareSyncController controller(slave_name);
            controller.configure_plant(
                1.0e-6, 493367, 0.45, 0.0, state_path);
            CHECK(controller.current_word() == 493367, fallback_message);
            CHECK(wait_for([&] { return sf_count.load() > before; }),
                  "invalid state fallback word is acknowledged");
        }
        CHECK(read_state() == content, preserve_message);
    };

    verify_invalid_state_is_preserved(
        "version: [\n",
        "write corrupt retained DAC YAML",
        "corrupt retained DAC YAML falls back to calibrated zero",
        "corrupt retained DAC YAML is not overwritten after fallback OK");

    const std::string valid_metadata =
        "version: 1\n"
        "device: \"" + std::string(slave_name) + "\"\n";
    verify_invalid_state_is_preserved(
        "version: 1\n"
        "device: \"/dev/not-this-ocxo\"\n"
        "word: 593367\n"
        "zero_word: 493367\n"
        "ppm_per_lsb: 1.0e-6\n",
        "write mismatched-device retained DAC state",
        "device mismatch falls back to calibrated zero",
        "device mismatch state is not overwritten after fallback OK");
    verify_invalid_state_is_preserved(
        valid_metadata +
            "word: 1048576\n"
            "zero_word: 493367\n"
            "ppm_per_lsb: 1.0e-6\n",
        "write out-of-range retained DAC state",
        "out-of-range retained word falls back to calibrated zero",
        "out-of-range retained word is not overwritten after fallback OK");
    verify_invalid_state_is_preserved(
        valid_metadata +
            "word: 593367\n"
            "zero_word: 493368\n"
            "ppm_per_lsb: 1.0e-6\n",
        "write calibration-mismatched retained DAC state",
        "zero-word mismatch falls back to calibrated zero",
        "zero-word mismatch state is not overwritten after fallback OK");
    verify_invalid_state_is_preserved(
        valid_metadata +
            "word: 593367\n"
            "zero_word: 493367\n"
            "ppm_per_lsb: 2.0e-6\n",
        "write gain-mismatched retained DAC state",
        "ppm-per-LSB mismatch falls back to calibrated zero",
        "ppm-per-LSB mismatch state is not overwritten after fallback OK");

    const size_t sf_count_before_slew_test = sf_count.load();
    {
        HardwareSyncController controller(slave_name);
        controller.configure_plant(1.0e-6, 493367, 0.45, 0.01, "");
        CHECK(wait_for([&] {
                  return sf_count.load() > sf_count_before_slew_test;
              }),
              "long-gap slew test initializes the DAC word");
        const auto t0 = std::chrono::steady_clock::time_point{};
        controller.rebase_frequency_control_slew();
        CHECK_NEAR(
            controller.apply_delta_ppm_at(0.0, t0 + std::chrono::hours(1)),
            0.0, 1e-15,
            "zero request does not consume the pending slew rebase");
        CHECK_NEAR(
            controller.apply_delta_ppm_at(0.1, t0 + std::chrono::hours(2)),
            1.0e-5, 1e-12,
            "first nonzero correction after a long search uses minimum slew dt");
        CHECK_NEAR(
            controller.apply_delta_ppm_at(
                0.1, t0 + std::chrono::hours(2) + std::chrono::seconds(1)),
            0.01, 1e-12,
            "subsequent correction uses actual elapsed control time");
    }

    stop.store(true, std::memory_order_release);
    responder.join();
    close(master_fd);
    unlink(state_path.c_str());
    unlink((state_path + ".tmp." + std::to_string(getpid())).c_str());
}

void test_pps_ap_is_one_shot_per_phase_entry() {
    OCXOPpsInitialAligner::Params params;
    params.enable = true;
    params.max_abs_adjust_ns = 500.0e6;
    params.settle_s = 2.0;
    OCXOPpsInitialAligner aligner(params);
    OCXOPpsInitialAligner::Observation obs;
    obs.phase_stage = true;
    obs.phase_valid = true;
    obs.phase_error_samples = 25.0;
    obs.sample_rate = 1.0e6;
    obs.publication_generation = 10;
    const auto t0 = std::chrono::steady_clock::time_point{};

    const auto requested = aligner.update_at(obs, t0);
    CHECK(requested.request_adjust && requested.adjustment_ns == 25'000,
          "positive UE-minus-BS PPS error uses the default positive AP polarity");
    CHECK(requested.request_hardware_rearm && requested.phase_entry_rearm,
          "first FllSlow-to-Phase entry rearms the PPS AP transport");
    CHECK(!requested.phase_control_ready,
          "Phase SF waits while the one-shot AP command is pending");

    obs.command_status = OCXOPpsInitialAligner::CommandStatus::Succeeded;
    const auto settling = aligner.update_at(
        obs, t0 + std::chrono::milliseconds(100));
    CHECK(settling.state == OCXOPpsInitialAligner::State::Settling &&
              !settling.phase_control_ready,
          "AP success waits for PPS settling and a fresh phase publication");

    const auto old_publication = aligner.update_at(
        obs, t0 + std::chrono::seconds(3));
    CHECK(old_publication.state == OCXOPpsInitialAligner::State::Settling,
          "elapsed time alone cannot accept the pre-AP phase publication");
    obs.publication_generation = 11;
    const auto done = aligner.update_at(obs, t0 + std::chrono::seconds(3));
    CHECK(done.state == OCXOPpsInitialAligner::State::Done &&
              done.phase_control_ready,
          "post-AP phase publication releases the PLL for this Phase residence");
    obs.phase_error_samples = -100.0;
    obs.publication_generation = 12;
    const auto remains_done = aligner.update_at(obs, t0 + std::chrono::seconds(4));
    CHECK(!remains_done.request_adjust &&
              remains_done.state == OCXOPpsInitialAligner::State::Done,
          "later PPS phase changes within one Phase residence emit no second AP");

    obs.phase_stage = false;
    const auto left_phase = aligner.update_at(
        obs, t0 + std::chrono::milliseconds(4500));
    CHECK(!left_phase.request_adjust && !left_phase.request_hardware_rearm &&
              left_phase.state == OCXOPpsInitialAligner::State::Done,
          "leaving Phase does not itself emit or rearm AP");
    obs.phase_stage = true;
    obs.command_status = OCXOPpsInitialAligner::CommandStatus::Succeeded;
    obs.publication_generation = 13;
    const auto rearmed = aligner.update_at(obs, t0 + std::chrono::seconds(5));
    CHECK(rearmed.request_hardware_rearm && rearmed.phase_entry_rearm &&
              !rearmed.request_adjust &&
              rearmed.state == OCXOPpsInitialAligner::State::Armed,
          "next FllSlow-to-Phase entry rearms the completed hardware transaction");
    obs.command_status = OCXOPpsInitialAligner::CommandStatus::Idle;
    const auto reentry_request = aligner.update_at(
        obs, t0 + std::chrono::milliseconds(5100));
    CHECK(!reentry_request.phase_entry_rearm && reentry_request.request_adjust &&
              reentry_request.adjustment_ns == -100'000 &&
              reentry_request.state ==
                  OCXOPpsInitialAligner::State::WaitingForCommand,
          "rearmed Phase entry emits exactly one AP from the fresh residual at Idle");

    auto opposite_params = params;
    opposite_params.command_polarity = -1;
    OCXOPpsInitialAligner opposite_polarity(opposite_params);
    obs.phase_error_samples = 25.0;
    obs.publication_generation = 20;
    obs.command_status = OCXOPpsInitialAligner::CommandStatus::Idle;
    const auto opposite = opposite_polarity.update_at(obs, t0);
    CHECK(opposite.request_adjust && opposite.adjustment_ns == -25'000,
          "AP polarity remains explicitly configurable for other hardware");

    OCXOPpsInitialAligner queue_failure(params);
    const auto failed_request = queue_failure.update_at(obs, t0);
    CHECK(failed_request.request_adjust &&
              queue_failure.notify_command_queue_result(
                  OCXOPpsInitialAligner::QueueResult::Failed) &&
              queue_failure.state() == OCXOPpsInitialAligner::State::Failed &&
              queue_failure.phase_control_ready(),
          "AP queue failure releases Phase SF instead of deadlocking");

    OCXOPpsInitialAligner queue_busy(params);
    const auto busy_request = queue_busy.update_at(obs, t0);
    CHECK(busy_request.request_adjust &&
              queue_busy.notify_command_queue_result(
                  OCXOPpsInitialAligner::QueueResult::Busy) &&
              queue_busy.state() == OCXOPpsInitialAligner::State::Armed &&
              !queue_busy.phase_control_ready(),
          "in-flight AP collision returns to Armed instead of failing the epoch");
    obs.command_status = OCXOPpsInitialAligner::CommandStatus::InFlight;
    const auto still_busy = queue_busy.update_at(
        obs, t0 + std::chrono::milliseconds(100));
    CHECK(!still_busy.request_adjust &&
              still_busy.state == OCXOPpsInitialAligner::State::Armed,
          "Armed aligner waits while the previous AP remains in flight");
    obs.command_status = OCXOPpsInitialAligner::CommandStatus::Idle;
    const auto idle_retry = queue_busy.update_at(
        obs, t0 + std::chrono::milliseconds(200));
    CHECK(idle_retry.request_adjust &&
              idle_retry.state == OCXOPpsInitialAligner::State::WaitingForCommand,
          "Armed aligner requests AP after hardware returns to Idle");

    params.command_timeout_s = 0.5;
    OCXOPpsInitialAligner timeout(params);
    timeout.update_at(obs, t0);
    const auto timed_out = timeout.update_at(obs, t0 + std::chrono::seconds(1));
    CHECK(timed_out.state == OCXOPpsInitialAligner::State::Failed &&
              timed_out.phase_control_ready,
          "AP command timeout releases Phase SF instead of waiting forever");

    OCXOPpsInitialAligner::Params settle_timeout_params = params;
    settle_timeout_params.settle_s = 0.5;
    settle_timeout_params.command_timeout_s = 0.5;
    settle_timeout_params.max_retries = 1;
    OCXOPpsInitialAligner settle_timeout(settle_timeout_params);
    obs.publication_generation = 30;
    obs.command_status = OCXOPpsInitialAligner::CommandStatus::Idle;
    settle_timeout.update_at(obs, t0);
    obs.command_status = OCXOPpsInitialAligner::CommandStatus::Succeeded;
    settle_timeout.update_at(obs, t0 + std::chrono::milliseconds(100));
    const auto retry_armed = settle_timeout.update_at(
        obs, t0 + std::chrono::milliseconds(1100));
    CHECK(retry_armed.state == OCXOPpsInitialAligner::State::Armed &&
              retry_armed.request_hardware_rearm &&
              retry_armed.retry_count == 1 &&
              !retry_armed.phase_control_ready,
          "settling deadline rearms one bounded AP retry");
    obs.command_status = OCXOPpsInitialAligner::CommandStatus::Idle;
    const auto stale_retry = settle_timeout.update_at(
        obs, t0 + std::chrono::milliseconds(1200));
    CHECK(!stale_retry.request_adjust,
          "settling retry never reuses the pre-AP publication");
    obs.publication_generation = 31;
    obs.phase_error_samples = 10.0;
    const auto fresh_retry = settle_timeout.update_at(
        obs, t0 + std::chrono::milliseconds(1300));
    CHECK(fresh_retry.request_adjust &&
              fresh_retry.adjustment_ns == 10'000,
          "settling retry recomputes AP from the fresh residual phase");
    obs.command_status = OCXOPpsInitialAligner::CommandStatus::Succeeded;
    settle_timeout.update_at(obs, t0 + std::chrono::milliseconds(1400));
    const auto retry_exhausted = settle_timeout.update_at(
        obs, t0 + std::chrono::milliseconds(2400));
    CHECK(retry_exhausted.state == OCXOPpsInitialAligner::State::Failed &&
              retry_exhausted.phase_control_ready,
          "exhausted settling retry releases SF instead of blocking forever");

    params.deadzone_ns = 30'000.0;
    OCXOPpsInitialAligner deadzone(params);
    obs.command_status = OCXOPpsInitialAligner::CommandStatus::Idle;
    const auto already_aligned = deadzone.update_at(obs, t0);
    CHECK(!already_aligned.request_adjust &&
              !already_aligned.request_hardware_rearm &&
              !already_aligned.phase_entry_rearm &&
              already_aligned.state == OCXOPpsInitialAligner::State::Done &&
              already_aligned.phase_control_ready,
          "near-zero PPS phase skips AP and settling delay");
    obs.phase_stage = false;
    deadzone.update_at(obs, t0 + std::chrono::milliseconds(100));
    obs.phase_stage = true;
    obs.command_status = OCXOPpsInitialAligner::CommandStatus::Succeeded;
    const auto aligned_reentry = deadzone.update_at(
        obs, t0 + std::chrono::milliseconds(200));
    CHECK(!aligned_reentry.request_adjust &&
              !aligned_reentry.request_hardware_rearm &&
              !aligned_reentry.phase_entry_rearm &&
              !aligned_reentry.state_changed &&
              aligned_reentry.state == OCXOPpsInitialAligner::State::Done &&
              aligned_reentry.phase_control_ready,
          "deadzone Phase re-entry stays done without hardware rearm noise");

    params.enable = false;
    OCXOPpsInitialAligner disabled(params);
    const auto bypassed = disabled.update_at(obs, t0);
    CHECK(!bypassed.request_adjust && bypassed.phase_control_ready &&
              bypassed.state == OCXOPpsInitialAligner::State::Done,
          "disabled initial AP directly releases PPS Phase SF tracking");
}

void test_pps_worker_jamsync_respects_wall_deadline() {
    const int master_fd = posix_openpt(O_RDWR | O_NOCTTY | O_NONBLOCK);
    CHECK(master_fd >= 0, "allocate pseudo-terminal for PPS worker test");
    if (master_fd < 0) {
        return;
    }
    const bool pty_ready =
        grantpt(master_fd) == 0 && unlockpt(master_fd) == 0;
    CHECK(pty_ready, "initialize pseudo-terminal slave");
    if (!pty_ready) {
        close(master_fd);
        return;
    }
    char* slave_name = ptsname(master_fd);
    CHECK(slave_name != nullptr, "resolve pseudo-terminal slave path");
    if (slave_name == nullptr) {
        close(master_fd);
        return;
    }

    std::atomic<bool> stop{false};
    std::atomic<size_t> ap_count{0};
    std::thread responder([&] {
        std::string pending;
        char buffer[128];
        while (!stop.load(std::memory_order_acquire)) {
            const ssize_t n = read(master_fd, buffer, sizeof(buffer));
            if (n > 0) {
                pending.append(buffer, static_cast<size_t>(n));
                size_t newline = std::string::npos;
                while ((newline = pending.find('\n')) != std::string::npos) {
                    const std::string command = pending.substr(0, newline + 1);
                    pending.erase(0, newline + 1);
                    if (command.rfind("AP", 0) == 0) {
                        ap_count.fetch_add(1, std::memory_order_relaxed);
                        const char response[] = "JAMSYNCING\n";
                        const ssize_t written =
                            write(master_fd, response, sizeof(response) - 1);
                        (void)written;
                    } else {
                        const char response[] = "OK\n";
                        const ssize_t written =
                            write(master_fd, response, sizeof(response) - 1);
                        (void)written;
                    }
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    });

    {
        HardwareSyncController controller(slave_name);
        controller.configure_pps_phase_adjust_timeout(0.35);
        const auto queued = controller.request_pps_phase_adjust_ns(10'000);
        CHECK(queued == HardwareSyncController::PpsPhaseAdjustQueueResult::Queued,
              "PPS worker accepts AP before wall deadline");
        const auto wait_deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (controller.pps_phase_adjust_status() !=
                   HardwareSyncController::PpsPhaseAdjustStatus::Failed &&
               std::chrono::steady_clock::now() < wait_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        CHECK(controller.pps_phase_adjust_status() ==
                  HardwareSyncController::PpsPhaseAdjustStatus::Failed,
              "JAMSYNCING AP reaches terminal Failed within wall deadline");
        const size_t count_at_failure = ap_count.load(std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::milliseconds(450));
        CHECK(ap_count.load(std::memory_order_relaxed) == count_at_failure,
              "PPS worker sends no AP copy after its wall deadline");
    }

    stop.store(true, std::memory_order_release);
    responder.join();
    close(master_fd);
}

void test_legacy_and_pps_phase_modes_remain_distinct() {
    Config cfg = make_default_ue_config();
    cfg.clock_time.time_source.clear();
    CHECK(resolve_ue_time_source(cfg) == "internal",
          "empty UE time_source resolves to internal, not clock_source");
    cfg.sync_tracking.ocxo_mode = kOCXOModePhase;
    cfg.sync_tracking.ocxo_phase_setpoint_mode = kOCXOPhaseSetpointFixed;
    cfg.sync_tracking.ocxo_phase_setpoint_fixed_samples = 3.5;
    const auto legacy = make_ocxo_discipliner_params(cfg);
    CHECK(legacy.policy == OCXODiscipliner::ModePolicy::Phase &&
              legacy.phase_setpoint_mode == OCXODiscipliner::SetpointMode::Fixed,
          "legacy phase mode keeps its TO_clock setpoint policy");
    CHECK_NEAR(legacy.phase_setpoint_fixed_samples, 3.5, 1e-12,
               "legacy fixed TO_clock setpoint remains configurable");

    cfg.sync_tracking.ocxo_mode = kOCXOModePpsPhase;
    const auto pps = make_ocxo_discipliner_params(cfg);
    CHECK(pps.policy == OCXODiscipliner::ModePolicy::PpsPhase,
          "pps_phase maps to the explicit PPS zero-target policy");
    bool rejected_missing_hardware = false;
    try {
        validate_ue_pps_phase_config(cfg);
    } catch (const std::runtime_error&) {
        rejected_missing_hardware = true;
    }
    CHECK(rejected_missing_hardware,
          "pps_phase rejects disabled hardware sync");
    cfg.sync_tracking.hardware_sync = true;
    bool rejected_missing_ertm = false;
    try {
        validate_ue_pps_phase_config(cfg);
    } catch (const std::runtime_error&) {
        rejected_missing_ertm = true;
    }
    CHECK(rejected_missing_ertm,
          "pps_phase rejects disabled eRTM TO publication");
    cfg.uplink.ertm_to_enable = true;
    cfg.clock_time.clock_source = "external";
    bool rejected_internal_time = false;
    try {
        validate_ue_pps_phase_config(cfg);
    } catch (const std::runtime_error&) {
        rejected_internal_time = true;
    }
    CHECK(rejected_internal_time,
          "pps_phase rejects an implicit internal UE time source");
    cfg.clock_time.time_source = "external";
    validate_ue_pps_phase_config(cfg);
    CHECK(true, "pps_phase accepts external clock and external time sources");
    cfg.sync_tracking.ocxo_pps_initial_ap_max_retries = 4;
    bool rejected_unbounded_retry = false;
    try {
        validate_ue_pps_phase_config(cfg);
    } catch (const std::runtime_error&) {
        rejected_unbounded_retry = true;
    }
    CHECK(rejected_unbounded_retry,
          "pps_phase rejects more than three post-AP settling retries");
}

void test_legacy_phase_uses_to_setpoint_without_pps() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Phase;
    params.phase_setpoint_mode = OCXODiscipliner::SetpointMode::Zero;
    params.max_abs_u_ppm = 10.0;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 0.1;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 0.1;
    params.fll_slow_enter_abs_ppm = 1.0;
    params.fll_switch_hold_s = 0.0;
    params.phase_enter_sfo_abs_ppm = 1.0;
    params.phase_enter_hold_s = 0.0;
    params.phase_control_period_s = 0.1;
    params.phase_kp = 1.0;
    params.phase_ki = 1.0;
    params.phase_max_slew_ppm_per_s = 100.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_ue_samples = 1.0;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    obs.pps_phase_valid = false;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);
    obs.to_frame_seq = 2;
    loop.update_at(obs, t0 + std::chrono::milliseconds(100));
    obs.to_frame_seq = 3;
    const auto entered = loop.update_at(
        obs, t0 + std::chrono::milliseconds(200));
    CHECK(entered.stage == OCXODiscipliner::Stage::Phase,
          "legacy phase enters without a PPS observation");
    obs.to_frame_seq = 4;
    const auto controlled = loop.update_at(
        obs, t0 + std::chrono::milliseconds(300));
    CHECK_NEAR(controlled.to_err, -1.0, 1e-12,
               "legacy phase error is setpoint minus TO_clock (Mode B sign)");
    CHECK_NEAR(controlled.desired_delta_ppm, -1.1, 1e-12,
               "legacy phase PI uses inverted TO setpoint error");
}

void test_legacy_phase_closed_loop_converges_with_positive_plant_gain() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Phase;
    params.phase_setpoint_mode = OCXODiscipliner::SetpointMode::Zero;
    params.max_abs_u_ppm = 10.0;
    params.coarse_enable = false;
    params.samples_per_frame = 1.0e6;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 1.0;
    params.fll_fast_kp = 0.0;
    params.fll_fast_ki = 0.0;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 1.0;
    params.fll_slow_kp = 0.0;
    params.fll_slow_ki = 0.0;
    params.fll_slow_enter_abs_ppm = 10.0;
    params.fll_fast_enter_abs_ppm = 20.0;
    params.fll_switch_hold_s = 0.0;
    params.phase_enter_sfo_abs_ppm = 10.0;
    params.phase_enter_hold_s = 0.0;
    params.phase_exit_sfo_abs_ppm = 100.0;
    params.phase_exit_hold_s = 1.0;
    params.phase_control_period_s = 1.0;
    params.phase_jump_gate_samples = 10.0;
    params.phase_kp = 0.5;
    params.phase_ki = 0.1;
    params.phase_max_slew_ppm_per_s = 10.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    obs.to_ue_samples = 0.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);

    constexpr double uncontrolled_clock_error_ppm = 1.0;
    OCXODiscipliner::UpdateResult out;
    for (uint64_t second = 1; second <= 40; ++second) {
        const double raw_to_slope_ppm =
            uncontrolled_clock_error_ppm + loop.command_ppm();
        obs.to_ue_samples += raw_to_slope_ppm;
        obs.to_frame_seq = second + 1;
        out = loop.update_at(obs, t0 + std::chrono::seconds(second));
        if (out.request_control) {
            loop.commit_applied_delta(out.desired_delta_ppm);
        }
    }

    CHECK(loop.stage() == OCXODiscipliner::Stage::Phase,
          "legacy positive-gain plant remains in Phase");
    CHECK_NEAR(loop.command_ppm(), -uncontrolled_clock_error_ppm, 2e-4,
               "legacy Phase cancels a positive-gain UE clock offset");
    CHECK_NEAR(out.to_err, 0.0, 5e-4,
               "legacy Phase drives setpoint-minus-TO_clock error to zero");
    CHECK_NEAR(out.frequency_error_ppm, -out.to_slope_ppm, 1e-12,
               "Phase frequency_error_ppm keeps the plant-oriented Mode B sign");
}

void test_auto_switch_cancels_cfo_coarse_target() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Auto;
    params.coarse_obs_window_s = 0.0;
    params.coarse_control_period_s = 0.1;
    params.coarse_max_slew_ppm_per_s = 10.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    obs.cfo_ppm = 0.2;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);
    const auto cfo_target = loop.update_at(
        obs, t0 + std::chrono::milliseconds(100));
    CHECK(cfo_target.request_control,
          "Mode A establishes a CFO coarse target before TO is available");
    loop.commit_applied_delta(0.0);

    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_ue_samples = 0.0;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    const auto switched = loop.update_at(
        obs, t0 + std::chrono::milliseconds(200));
    CHECK(switched.frequency_source == OCXODiscipliner::FrequencySource::ToSlope,
          "auto selects TO-derived SFO when Mode B TO becomes fresh");
    CHECK(switched.stage == OCXODiscipliner::Stage::FllFast &&
              !switched.request_control,
          "Mode switch cancels the outstanding CFO coarse target");
    CHECK_NEAR(switched.coarse_outstanding_ppm, 0.0, 1e-12,
               "cancelled CFO coarse target has no outstanding remainder");
}

void test_coarse_applied_delta_feedback() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Freq;
    params.max_abs_u_ppm = 1.0;
    params.coarse_enable = true;
    params.coarse_obs_window_s = 0.0;
    params.coarse_control_period_s = 0.1;
    params.coarse_max_abs_ppm = 0.2;
    params.coarse_max_slew_ppm_per_s = 10.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    obs.cfo_ppm = 0.2;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);
    const auto request = loop.update_at(obs, t0 + std::chrono::milliseconds(100));
    CHECK_NEAR(request.desired_delta_ppm, 0.2, 1e-12,
               "coarse requests its one target");
    loop.commit_applied_delta(0.05);
    const auto debug = loop.debug_state();
    CHECK_NEAR(debug.command_ppm, 0.05, 1e-12,
               "command advances only by hardware-applied delta");
    CHECK_NEAR(debug.coarse_outstanding_ppm, 0.15, 1e-12,
               "coarse target retains unapplied remainder");
}

void test_coarse_overshoot_cancels_target() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Freq;
    params.max_abs_u_ppm = 1.0;
    params.coarse_obs_window_s = 0.0;
    params.coarse_control_period_s = 0.1;
    params.coarse_max_slew_ppm_per_s = 10.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    obs.cfo_ppm = 0.2;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);
    const auto request = loop.update_at(
        obs, t0 + std::chrono::milliseconds(100));
    CHECK(request.request_control, "coarse target requests its first step");
    loop.commit_applied_delta(0.05);

    obs.cfo_ppm = -0.01;
    const auto aborted = loop.update_at(
        obs, t0 + std::chrono::milliseconds(200));
    CHECK(aborted.stage == OCXODiscipliner::Stage::FllFast &&
              aborted.transition_reason ==
                  OCXODiscipliner::TransitionReason::CoarseOvershoot,
          "coarse aborts when the residual reverses across its target");
    CHECK_NEAR(aborted.coarse_outstanding_ppm, 0.0, 1e-12,
               "coarse overshoot leaves no orphan target");
}

void test_coarse_to_jump_cancels_target() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Phase;
    params.coarse_obs_window_s = 0.0;
    params.coarse_control_period_s = 0.1;
    params.phase_jump_gate_samples = 0.5;
    params.samples_per_frame = 1000.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_ue_samples = 0.0;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    obs.pps_phase_valid = true;
    obs.pps_phase_error_samples = 0.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);
    obs.to_ue_samples = 0.0002;
    obs.to_frame_seq = 2;
    const auto latched = loop.update_at(
        obs, t0 + std::chrono::milliseconds(100));
    CHECK(std::abs(latched.coarse_outstanding_ppm) > 0.0,
          "Mode B coarse target is latched before TO jump");

    obs.to_ue_samples = 1.0;
    obs.to_frame_seq = 3;
    const auto jumped = loop.update_at(
        obs, t0 + std::chrono::milliseconds(200));
    CHECK(jumped.stage == OCXODiscipliner::Stage::FllFast &&
              jumped.transition_reason == OCXODiscipliner::TransitionReason::ToJump,
          "TO jump leaves Coarse through fast FLL");
    CHECK_NEAR(jumped.coarse_outstanding_ppm, 0.0, 1e-12,
               "TO jump clears the outstanding coarse target");
}

void test_fll_retains_real_window_until_control_period() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Freq;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.1;
    params.fll_fast_control_period_s = 1.0;
    params.fll_fast_kp = 0.0;
    params.fll_fast_ki = 0.0;
    params.fll_slow_enter_abs_ppm = 0.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    const auto t0 = std::chrono::steady_clock::time_point{};
    obs.cfo_ppm = 0.0;
    loop.update_at(obs, t0);
    loop.update_at(obs, t0 + std::chrono::milliseconds(100));

    obs.cfo_ppm = 1.0;
    loop.update_at(obs, t0 + std::chrono::milliseconds(200));
    obs.cfo_ppm = 3.0;
    const auto early = loop.update_at(
        obs, t0 + std::chrono::milliseconds(300));
    CHECK(!early.request_control,
          "ready FLL window waits for the control period");
    obs.cfo_ppm = 5.0;
    const auto consumed = loop.update_at(
        obs, t0 + std::chrono::milliseconds(1100));
    CHECK_NEAR(consumed.frequency_error_ppm, 3.0, 1e-12,
               "FLL consumes the three real samples without synthetic reseed");
}

void test_alignment_continuous_to_clock() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Phase;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 0.1;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 0.1;
    params.fll_slow_enter_abs_ppm = 1.0;
    params.fll_switch_hold_s = 0.0;
    params.phase_enter_sfo_abs_ppm = 0.05;
    params.phase_enter_hold_s = 0.0;
    params.phase_control_period_s = 0.1;
    params.phase_jump_gate_samples = 0.5;
    params.phase_max_report_age_frames = 100;
    params.samples_per_frame = 1000.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    obs.cfo_ppm = 0.0;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_report_age_frames = 0;
    obs.to_timeline_epoch = 7;
    obs.to_sync_generation = 3;
    obs.pps_phase_valid = true;
    obs.pps_phase_error_samples = 0.0;

    const auto t0 = std::chrono::steady_clock::time_point{};
    obs.to_ue_samples = 1.0;
    obs.to_alignment_samples = 0;
    obs.to_frame_seq = 10;
    loop.update_at(obs, t0);

    obs.to_ue_samples = -1.0;
    obs.to_alignment_samples = -2;
    obs.to_frame_seq = 11;
    obs.cfo_ppm = 1.0;
    const auto aligned = loop.update_at(obs, t0 + std::chrono::milliseconds(100));
    CHECK_NEAR(aligned.to_clock, 1.0, 1e-12,
               "known Alignment edge is removed from TO_clock");
    CHECK_NEAR(aligned.to_slope_ppm, 0.0, 1e-12,
               "Alignment edge does not create TO slope");
    CHECK_NEAR(aligned.alignment_feedforward_samples, 2.0, 1e-12,
               "Alignment feedforward has the cancelling sign");
    CHECK_NEAR(aligned.desired_delta_ppm, 0.0, 1e-12,
               "Phase loop does not react to pure Alignment");
    CHECK(aligned.stage == OCXODiscipliner::Stage::FllSlow,
          "quiet TO slope advances Mode B to slow FLL");

    obs.to_ue_samples = 1.0;
    obs.to_alignment_samples = 0;
    obs.to_frame_seq = 12;
    const auto reverse_aligned = loop.update_at(
        obs, t0 + std::chrono::milliseconds(200));
    CHECK_NEAR(reverse_aligned.to_clock, 1.0, 1e-12,
               "reverse Alignment edge also preserves TO_clock");
    CHECK_NEAR(reverse_aligned.alignment_feedforward_samples, -2.0, 1e-12,
               "reverse Alignment feedforward has the cancelling sign");
    CHECK(reverse_aligned.stage == OCXODiscipliner::Stage::Phase,
          "Doppler-only CFO does not affect Mode B Phase entry");

    obs.to_ue_samples = 2.0;
    obs.to_frame_seq = 13;
    const auto jumped = loop.update_at(obs, t0 + std::chrono::milliseconds(300));
    CHECK(jumped.to_jump_rejected &&
              jumped.stage == OCXODiscipliner::Stage::Phase &&
              !jumped.stage_changed,
          "a transient TO_clock jump is rejected without an immediate downgrade");
}

void test_phase_velocity_pi_does_not_repeat_proportional_step() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::PpsPhase;
    params.max_abs_u_ppm = 10.0;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 0.1;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 0.1;
    params.fll_slow_enter_abs_ppm = 1.0;
    params.fll_switch_hold_s = 0.0;
    params.phase_enter_sfo_abs_ppm = 1.0;
    params.phase_enter_hold_s = 0.0;
    params.phase_control_period_s = 0.1;
    params.phase_jump_gate_samples = 1.0;
    params.phase_kp = 1.0;
    params.phase_ki = 1.0;
    params.phase_max_slew_ppm_per_s = 100.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    obs.cfo_ppm = 0.0;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_ue_samples = 1.0;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    obs.pps_phase_valid = true;
    obs.pps_phase_error_samples = 1.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);

    obs.to_frame_seq = 2;
    const auto slowed = loop.update_at(
        obs, t0 + std::chrono::milliseconds(100));
    CHECK(slowed.stage == OCXODiscipliner::Stage::FllSlow,
          "second stable TO report advances to slow FLL");

    obs.to_frame_seq = 3;
    const auto entered = loop.update_at(
        obs, t0 + std::chrono::milliseconds(200));
    CHECK(entered.stage == OCXODiscipliner::Stage::Phase,
          "third stable TO report admits Phase");

    obs.to_frame_seq = 4;
    const auto first = loop.update_at(obs, t0 + std::chrono::milliseconds(300));
    CHECK_NEAR(first.desired_delta_ppm, 1.1, 1e-12,
               "phase velocity PI applies proportional and integral increments");
    loop.commit_applied_delta(0.05);

    obs.to_frame_seq = 5;
    const auto second = loop.update_at(obs, t0 + std::chrono::milliseconds(400));
    CHECK_NEAR(second.desired_delta_ppm, 0.1, 1e-12,
               "constant phase error does not repeat the proportional step");
}

void test_phase_exit_uses_sfo_only() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Phase;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 0.1;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 0.1;
    params.fll_slow_enter_abs_ppm = 1.0;
    params.fll_switch_hold_s = 0.0;
    params.phase_enter_sfo_abs_ppm = 0.05;
    params.phase_enter_hold_s = 0.0;
    params.phase_control_period_s = 0.1;
    params.phase_exit_sfo_abs_ppm = 0.1;
    params.phase_exit_hold_s = 0.1;
    params.samples_per_frame = 1000.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    obs.cfo_ppm = -100.0;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_ue_samples = 0.0;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    obs.pps_phase_valid = true;
    obs.pps_phase_error_samples = 10.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    const auto anchored = loop.update_at(obs, t0);
    CHECK(anchored.stage == OCXODiscipliner::Stage::FllFast,
          "single TO anchor waits in fast FLL for a slope estimate");
    obs.to_frame_seq = 2;
    const auto slowed = loop.update_at(
        obs, t0 + std::chrono::milliseconds(100));
    CHECK(slowed.stage == OCXODiscipliner::Stage::FllSlow,
          "quiet TO slope advances to slow FLL before Phase");
    obs.to_frame_seq = 3;
    const auto entered = loop.update_at(
        obs, t0 + std::chrono::milliseconds(200));
    CHECK(entered.stage == OCXODiscipliner::Stage::Phase,
          "large Doppler-like CFO does not block Mode B Phase entry");

    obs.to_ue_samples = 0.0002;
    obs.to_frame_seq = 4;
    const auto pending = loop.update_at(
        obs, t0 + std::chrono::milliseconds(300));
    CHECK(pending.stage == OCXODiscipliner::Stage::Phase,
          "Phase SFO fault starts the ten-second downgrade dwell");
    obs.to_ue_samples = 0.0198;
    obs.to_frame_seq = 102;
    const auto held = loop.update_at(
        obs, t0 + std::chrono::milliseconds(10100));
    CHECK(held.stage == OCXODiscipliner::Stage::Phase,
          "Phase SFO fault is retained for less than ten seconds");
    obs.to_ue_samples = 0.0200;
    obs.to_frame_seq = 103;
    const auto exited = loop.update_at(
        obs, t0 + std::chrono::milliseconds(10200));
    CHECK(exited.stage == OCXODiscipliner::Stage::FllFast &&
              exited.transition_reason ==
                  OCXODiscipliner::TransitionReason::PhaseSfoConfirmed,
          "Phase exits after ten seconds of sustained TO-derived SFO");
}

void test_phase_entry_requires_stable_to_slope() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Phase;
    params.coarse_enable = false;
    params.samples_per_frame = 1000.0;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 0.1;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 0.1;
    params.fll_slow_enter_abs_ppm = 1.0;
    params.fll_switch_hold_s = 0.0;
    params.phase_enter_sfo_abs_ppm = 1.0;
    params.phase_enter_hold_s = 0.0;
    params.phase_jump_gate_samples = 1.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    obs.to_ue_samples = 0.0;
    obs.pps_phase_valid = true;
    obs.pps_phase_error_samples = 0.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    const auto anchored = loop.update_at(obs, t0);
    CHECK(anchored.stage == OCXODiscipliner::Stage::FllFast &&
              !anchored.to_slope_valid,
          "a re-anchor alone cannot enter Phase");

    obs.to_frame_seq = 2;
    obs.to_ue_samples = 0.01;
    const auto moving = loop.update_at(
        obs, t0 + std::chrono::milliseconds(100));
    CHECK(moving.stage == OCXODiscipliner::Stage::FllFast &&
              moving.to_slope_valid,
          "an unstable TO slope keeps the loop in fast FLL");

    obs.to_frame_seq = 3;
    const auto slowed = loop.update_at(
        obs, t0 + std::chrono::milliseconds(200));
    CHECK(slowed.stage == OCXODiscipliner::Stage::FllSlow,
          "a quiet TO slope advances to slow FLL first");
    obs.to_frame_seq = 4;
    const auto stable = loop.update_at(
        obs, t0 + std::chrono::milliseconds(300));
    CHECK(stable.stage == OCXODiscipliner::Stage::Phase,
          "a valid quiet TO slope admits Phase");
}

void test_freeze_pauses_fast_fll_dwell() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Freq;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 1.0;
    params.fll_slow_enter_abs_ppm = 1.0;
    params.fll_switch_hold_s = 3.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    obs.cfo_ppm = 0.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);
    loop.update_at(obs, t0 + std::chrono::seconds(1));
    obs.allow_control = false;
    const auto frozen = loop.update_at(
        obs, t0 + std::chrono::milliseconds(1500));
    CHECK(frozen.stage == OCXODiscipliner::Stage::Freeze &&
              loop.stage() == OCXODiscipliner::Stage::FllFast,
          "freeze is an overlay on fast FLL");

    obs.allow_control = true;
    const auto resumed = loop.update_at(
        obs, t0 + std::chrono::milliseconds(101500));
    CHECK(resumed.stage == OCXODiscipliner::Stage::FllFast,
          "wall time spent frozen does not complete fast-FLL dwell");
    const auto completed = loop.update_at(
        obs, t0 + std::chrono::seconds(102));
    CHECK(completed.stage == OCXODiscipliner::Stage::FllSlow,
          "fast-FLL dwell resumes from its pre-freeze progress");
}

void test_fll_unstable_requires_ten_continuous_seconds() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Freq;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 1.0;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 1.0;
    params.fll_slow_enter_abs_ppm = 0.1;
    params.fll_fast_enter_abs_ppm = 0.2;
    params.fll_switch_hold_s = 0.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    obs.cfo_ppm = 0.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    const auto slow = loop.update_at(obs, t0);
    CHECK(slow.stage == OCXODiscipliner::Stage::FllSlow,
          "stable fast FLL enters slow FLL for downgrade testing");

    obs.cfo_ppm = 0.3;
    const auto started = loop.update_at(obs, t0 + std::chrono::seconds(1));
    CHECK(started.stage == OCXODiscipliner::Stage::FllSlow,
          "unstable FLL starts a downgrade dwell");
    const auto held = loop.update_at(
        obs, t0 + std::chrono::milliseconds(10999));
    CHECK(held.stage == OCXODiscipliner::Stage::FllSlow,
          "unstable FLL shorter than ten seconds retains slow mode");

    obs.cfo_ppm = 0.0;
    loop.update_at(obs, t0 + std::chrono::seconds(11));
    loop.update_at(obs, t0 + std::chrono::seconds(12));
    obs.cfo_ppm = 0.3;
    loop.update_at(obs, t0 + std::chrono::seconds(13));
    const auto recovered_dwell = loop.update_at(
        obs, t0 + std::chrono::milliseconds(22999));
    CHECK(recovered_dwell.stage == OCXODiscipliner::Stage::FllSlow,
          "a healthy observation clears the previous unstable-FLL dwell");
    const auto downgraded = loop.update_at(
        obs, t0 + std::chrono::seconds(23));
    CHECK(downgraded.stage == OCXODiscipliner::Stage::FllFast &&
              downgraded.transition_reason ==
                  OCXODiscipliner::TransitionReason::FllUnstable,
          "slow FLL downgrades after ten new continuous unstable seconds");
}

void test_frequency_source_change_requires_ten_continuous_seconds() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Auto;
    params.coarse_enable = false;
    params.samples_per_frame = 1000.0;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 1.0;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 1.0;
    params.fll_slow_enter_abs_ppm = 0.1;
    params.fll_switch_hold_s = 0.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    obs.cfo_ppm = 0.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    const auto slow = loop.update_at(obs, t0);
    CHECK(slow.stage == OCXODiscipliner::Stage::FllSlow &&
              slow.frequency_source == OCXODiscipliner::FrequencySource::Cfo,
          "Auto mode starts slow FLL on CFO without TO");

    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_ue_samples = 0.0;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    const auto started = loop.update_at(obs, t0 + std::chrono::seconds(1));
    CHECK(started.stage == OCXODiscipliner::Stage::FllSlow &&
              started.frequency_source == OCXODiscipliner::FrequencySource::Cfo,
          "TO availability starts a source-change downgrade dwell");
    obs.to_frame_seq = 2;
    const auto held = loop.update_at(
        obs, t0 + std::chrono::milliseconds(10999));
    CHECK(held.stage == OCXODiscipliner::Stage::FllSlow &&
              held.frequency_source == OCXODiscipliner::FrequencySource::Cfo,
          "source change shorter than ten seconds retains slow FLL");
    obs.to_frame_seq = 3;
    const auto downgraded = loop.update_at(
        obs, t0 + std::chrono::seconds(11));
    CHECK(downgraded.stage == OCXODiscipliner::Stage::FllFast &&
              downgraded.transition_reason ==
                  OCXODiscipliner::TransitionReason::FrequencySourceChanged &&
              downgraded.frequency_source ==
                  OCXODiscipliner::FrequencySource::ToSlope,
          "persistent source change downgrades after ten seconds");
}

void test_concurrent_degradation_dwells_are_independent() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Auto;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 1.0;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 1.0;
    params.fll_slow_enter_abs_ppm = 0.1;
    params.fll_fast_enter_abs_ppm = 0.2;
    params.fll_switch_hold_s = 0.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.cfo_valid = true;
    obs.cfo_ppm = 0.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);

    obs.cfo_ppm = 0.3;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_ue_samples = 0.0;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    loop.update_at(obs, t0 + std::chrono::seconds(1));

    obs.to_valid = false;
    obs.to_is_new = false;
    loop.update_at(obs, t0 + std::chrono::milliseconds(10500));
    const auto downgraded = loop.update_at(
        obs, t0 + std::chrono::seconds(11));
    CHECK(downgraded.stage == OCXODiscipliner::Stage::FllFast &&
              downgraded.transition_reason ==
                  OCXODiscipliner::TransitionReason::FllUnstable,
          "recovering one fault does not erase another downgrade dwell");
}

void test_freeze_defers_jump_and_stale_transitions() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::PpsPhase;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 0.1;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 0.1;
    params.fll_slow_enter_abs_ppm = 1.0;
    params.fll_switch_hold_s = 0.0;
    params.phase_enter_sfo_abs_ppm = 1.0;
    params.phase_enter_hold_s = 0.0;
    params.phase_jump_gate_samples = 0.5;

    const auto enter_phase = [&](OCXODiscipliner& loop,
                                 OCXODiscipliner::Observation& obs) {
        const auto t0 = std::chrono::steady_clock::time_point{};
        loop.update_at(obs, t0);
        obs.to_frame_seq = 2;
        loop.update_at(obs, t0 + std::chrono::milliseconds(100));
        obs.to_frame_seq = 3;
        loop.update_at(obs, t0 + std::chrono::milliseconds(200));
    };

    OCXODiscipliner::Observation obs;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_ue_samples = 0.0;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    obs.pps_phase_valid = true;
    obs.pps_phase_error_samples = 0.0;
    const auto t0 = std::chrono::steady_clock::time_point{};

    OCXODiscipliner jump_loop(params);
    enter_phase(jump_loop, obs);
    obs.allow_control = false;
    obs.to_ue_samples = 1.0;
    obs.to_frame_seq = 4;
    const auto frozen_jump = jump_loop.update_at(
        obs, t0 + std::chrono::milliseconds(300));
    CHECK(frozen_jump.stage == OCXODiscipliner::Stage::Freeze &&
              jump_loop.stage() == OCXODiscipliner::Stage::Phase &&
              !frozen_jump.stage_changed,
          "TO jump cannot change the internal stage while frozen");
    obs.allow_control = true;
    obs.to_is_new = false;
    const auto applied_jump = jump_loop.update_at(
        obs, t0 + std::chrono::seconds(10));
    CHECK(applied_jump.stage == OCXODiscipliner::Stage::Phase &&
              !applied_jump.stage_changed,
          "a deferred TO jump starts its downgrade dwell after control resumes");

    OCXODiscipliner stale_loop(params);
    obs.allow_control = true;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_ue_samples = 0.0;
    obs.to_frame_seq = 1;
    enter_phase(stale_loop, obs);
    obs.allow_control = false;
    obs.to_valid = false;
    obs.to_is_new = false;
    const auto frozen_stale = stale_loop.update_at(
        obs, t0 + std::chrono::milliseconds(200));
    CHECK(frozen_stale.stage == OCXODiscipliner::Stage::Freeze &&
              stale_loop.stage() == OCXODiscipliner::Stage::Phase,
          "stale TO cannot change the internal stage while frozen");
    obs.allow_control = true;
    const auto stale_started = stale_loop.update_at(
        obs, t0 + std::chrono::seconds(10));
    CHECK(stale_started.stage == OCXODiscipliner::Stage::Phase,
          "stale TO starts its downgrade dwell after control resumes");
    const auto stale_held = stale_loop.update_at(
        obs, t0 + std::chrono::milliseconds(19999));
    CHECK(stale_held.stage == OCXODiscipliner::Stage::Phase,
          "stale TO shorter than ten seconds retains Phase");
    const auto applied_stale = stale_loop.update_at(
        obs, t0 + std::chrono::seconds(20));
    CHECK(applied_stale.stage == OCXODiscipliner::Stage::FllFast &&
              applied_stale.transition_reason ==
                  OCXODiscipliner::TransitionReason::ToStale,
          "stale TO downgrades only after ten continuous seconds");

    OCXODiscipliner phase_loop(params);
    obs.allow_control = true;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_ue_samples = 0.0;
    obs.to_frame_seq = 1;
    enter_phase(phase_loop, obs);
    obs.to_ue_samples = 1.0;
    obs.pps_phase_error_samples = 1.0;
    obs.to_frame_seq = 4;
    phase_loop.update_at(obs, t0 + std::chrono::milliseconds(300));
    obs.to_frame_seq = 5;
    const auto reanchored = phase_loop.update_at(
        obs, t0 + std::chrono::milliseconds(400));
    CHECK_NEAR(reanchored.to_clock, 1.0, 1e-12,
               "post-jump report restores the absolute TO_clock offset");
    CHECK_NEAR(reanchored.to_err, 1.0, 1e-12,
               "PPS phase error remains the zero-target Phase variable");
}

void test_freeze_pauses_phase_exit_dwell() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::Phase;
    params.coarse_enable = false;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 1.0;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 1.0;
    params.fll_slow_enter_abs_ppm = 1.0;
    params.fll_switch_hold_s = 0.0;
    params.phase_enter_sfo_abs_ppm = 1.0;
    params.phase_enter_hold_s = 0.0;
    params.phase_control_period_s = 1.0;
    params.phase_exit_sfo_abs_ppm = 0.1;
    params.phase_exit_hold_s = 3.0;
    params.samples_per_frame = 1000.0;

    OCXODiscipliner loop(params);
    OCXODiscipliner::Observation obs;
    obs.to_valid = true;
    obs.to_is_new = true;
    obs.to_ue_samples = 0.0;
    obs.to_timeline_epoch = 1;
    obs.to_sync_generation = 1;
    obs.to_frame_seq = 1;
    obs.pps_phase_valid = true;
    obs.pps_phase_error_samples = 0.0;
    const auto t0 = std::chrono::steady_clock::time_point{};
    loop.update_at(obs, t0);
    obs.to_frame_seq = 2;
    loop.update_at(obs, t0 + std::chrono::seconds(1));
    obs.to_frame_seq = 3;
    loop.update_at(obs, t0 + std::chrono::seconds(2));
    obs.to_ue_samples = 0.0002;
    obs.to_frame_seq = 4;
    loop.update_at(obs, t0 + std::chrono::seconds(3));

    obs.allow_control = false;
    obs.to_is_new = false;
    const auto frozen = loop.update_at(
        obs, t0 + std::chrono::milliseconds(3500));
    CHECK(frozen.stage == OCXODiscipliner::Stage::Freeze &&
              loop.stage() == OCXODiscipliner::Stage::Phase,
          "freeze is an overlay on Phase");
    obs.allow_control = true;
    obs.to_is_new = false;
    const auto resumed = loop.update_at(
        obs, t0 + std::chrono::milliseconds(103500));
    CHECK(resumed.stage == OCXODiscipliner::Stage::Phase,
          "wall time spent frozen does not complete Phase exit dwell");
}

void test_reacquisition_preserves_mode_for_thirty_seconds() {
    OCXODiscipliner::Params params;
    params.policy = OCXODiscipliner::ModePolicy::PpsPhase;
    params.coarse_enable = false;
    params.samples_per_frame = 1000.0;
    params.fll_fast_obs_window_s = 0.0;
    params.fll_fast_control_period_s = 1.0;
    params.fll_slow_obs_window_s = 0.0;
    params.fll_slow_control_period_s = 1.0;
    params.fll_slow_enter_abs_ppm = 1.0;
    params.fll_switch_hold_s = 0.0;
    params.phase_enter_sfo_abs_ppm = 1.0;
    params.phase_enter_hold_s = 0.0;
    params.phase_control_period_s = 1.0;

    const auto enter_phase = [&params](OCXODiscipliner& loop) {
        OCXODiscipliner::Observation obs;
        obs.to_valid = true;
        obs.to_is_new = true;
        obs.to_ue_samples = 0.0;
        obs.to_timeline_epoch = 1;
        obs.to_sync_generation = 1;
        obs.to_frame_seq = 1;
        obs.pps_phase_valid = true;
        obs.pps_phase_error_samples = 0.0;
        const auto t0 = std::chrono::steady_clock::time_point{};
        loop.update_at(obs, t0);
        obs.to_frame_seq = 2;
        loop.update_at(obs, t0 + std::chrono::seconds(1));
        obs.to_frame_seq = 3;
        loop.update_at(obs, t0 + std::chrono::seconds(2));
        return obs;
    };

    const auto hold_start =
        std::chrono::steady_clock::time_point{} + std::chrono::seconds(2);
    OCXODiscipliner short_gap(params);
    auto obs = enter_phase(short_gap);
    CHECK(short_gap.stage() == OCXODiscipliner::Stage::Phase,
          "reacquisition test reaches Phase before sync loss");
    short_gap.commit_applied_delta(0.123);
    short_gap.begin_reacquisition_hold_at(hold_start);
    CHECK(short_gap.reacquisition_hold_active() &&
              short_gap.stage() == OCXODiscipliner::Stage::Phase,
          "sync loss freezes the Phase loop without changing its stage");

    OCXODiscipliner::Observation missing_obs;
    const auto waiting = short_gap.prepare_reacquisition_resume_at(
        missing_obs,
        OCXODiscipliner::kReacquisitionModeHoldS,
        hold_start + std::chrono::seconds(29));
    CHECK(waiting.action == OCXODiscipliner::ReacquisitionAction::Waiting &&
              short_gap.stage() == OCXODiscipliner::Stage::Phase,
          "Phase remains held while the recovered timeline has no TO anchor");

    obs.allow_control = false;
    obs.to_timeline_epoch = 2;
    obs.to_sync_generation = 2;
    obs.to_frame_seq = 1;
    short_gap.update_at(obs, hold_start + std::chrono::seconds(29));
    obs.allow_control = true;
    obs.to_is_new = false;
    const auto boundary = short_gap.prepare_reacquisition_resume_at(
        obs,
        OCXODiscipliner::kReacquisitionModeHoldS,
        hold_start + std::chrono::seconds(30));
    CHECK(boundary.action == OCXODiscipliner::ReacquisitionAction::Preserved,
          "exactly 30 seconds remains inside the mode-preserve window");
    const auto resumed = short_gap.update_at(
        obs, hold_start + std::chrono::seconds(30));
    CHECK(resumed.stage == OCXODiscipliner::Stage::Phase &&
              short_gap.stage() == OCXODiscipliner::Stage::Phase,
          "short reacquisition resumes in the retained Phase stage");
    CHECK_NEAR(short_gap.command_ppm(), 0.123, 1e-12,
               "short reacquisition retains the DAC command");

    OCXODiscipliner long_gap(params);
    enter_phase(long_gap);
    long_gap.commit_applied_delta(0.123);
    long_gap.begin_reacquisition_hold_at(hold_start);
    const auto expired = long_gap.prepare_reacquisition_resume_at(
        missing_obs,
        OCXODiscipliner::kReacquisitionModeHoldS,
        hold_start + std::chrono::milliseconds(30001));
    CHECK(expired.action == OCXODiscipliner::ReacquisitionAction::Reset &&
              long_gap.stage() == OCXODiscipliner::Stage::FllFast,
          "reacquisition beyond 30 seconds restarts fast FLL");
    CHECK_NEAR(long_gap.command_ppm(), 0.123, 1e-12,
               "long reacquisition reset still retains the DAC command");

    auto legacy_params = params;
    legacy_params.policy = OCXODiscipliner::ModePolicy::Phase;
    legacy_params.phase_setpoint_mode =
        OCXODiscipliner::SetpointMode::FreezeOnLock;
    OCXODiscipliner legacy_gap(legacy_params);
    auto legacy_obs = enter_phase(legacy_gap);
    legacy_gap.begin_reacquisition_hold_at(hold_start);
    legacy_obs.allow_control = false;
    legacy_obs.to_timeline_epoch = 2;
    legacy_obs.to_sync_generation = 2;
    legacy_obs.to_frame_seq = 1;
    legacy_gap.update_at(
        legacy_obs, hold_start + std::chrono::seconds(5));
    legacy_obs.allow_control = true;
    legacy_obs.to_is_new = false;
    const auto legacy_resume = legacy_gap.prepare_reacquisition_resume_at(
        legacy_obs,
        OCXODiscipliner::kReacquisitionModeHoldS,
        hold_start + std::chrono::seconds(6));
    const auto legacy_out = legacy_gap.update_at(
        legacy_obs, hold_start + std::chrono::seconds(6));
    CHECK(legacy_resume.action ==
              OCXODiscipliner::ReacquisitionAction::Preserved &&
              legacy_out.stage == OCXODiscipliner::Stage::Phase &&
              legacy_gap.stage() == OCXODiscipliner::Stage::Phase,
          "legacy FreezeOnLock Phase reanchors its setpoint and retains mode");
}

int main() {
    test_ewma_control_step_polarity();
    test_normal_cfo_dealiasing();
    test_cfo_acquisition_to_tracking();
    test_ocxo_debug_packet_layout();
    test_fll_velocity_pi_polarity();
    test_fll_fast_half_step_control();
    test_mode_b_fll_uses_to_slope_not_cfo();
    test_mode_b_fll_closed_loop_converges_with_positive_plant_gain();
    test_to_slope_uses_window_regression();
    test_phase_velocity_pi_closed_loop_converges();
    test_phase_default_gain_tuning();
    test_discipliner_reset_retains_dac_command();
    test_dac_word_persists_across_controller_restart();
    test_pps_ap_is_one_shot_per_phase_entry();
    test_pps_worker_jamsync_respects_wall_deadline();
    test_legacy_and_pps_phase_modes_remain_distinct();
    test_legacy_phase_uses_to_setpoint_without_pps();
    test_legacy_phase_closed_loop_converges_with_positive_plant_gain();
    test_auto_switch_cancels_cfo_coarse_target();
    test_coarse_applied_delta_feedback();
    test_coarse_overshoot_cancels_target();
    test_coarse_to_jump_cancels_target();
    test_fll_retains_real_window_until_control_period();
    test_alignment_continuous_to_clock();
    test_phase_velocity_pi_does_not_repeat_proportional_step();
    test_phase_exit_uses_sfo_only();
    test_phase_entry_requires_stable_to_slope();
    test_freeze_pauses_fast_fll_dwell();
    test_fll_unstable_requires_ten_continuous_seconds();
    test_frequency_source_change_requires_ten_continuous_seconds();
    test_concurrent_degradation_dwells_are_independent();
    test_freeze_defers_jump_and_stale_transitions();
    test_freeze_pauses_phase_exit_dwell();
    test_reacquisition_preserves_mode_for_thirty_seconds();
    std::printf("OCXO control helper tests: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
"""


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="openisac-ocxo-test-") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "test_ocxo_control.cpp"
        binary = tmp_path / "test_ocxo_control"
        source.write_text(TEST_SOURCE)
        build_cmd = [
            "g++",
            "-std=c++17",
            "-O2",
            f"-I{repo_root / 'include'}",
            "-o",
            str(binary),
            str(source),
            str(repo_root / "src" / "AsyncLogger.cpp"),
            "-lfftw3f",
            "-lyaml-cpp",
            "-lpthread",
        ]
        print(f"Building: {' '.join(build_cmd)}")
        result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"BUILD FAILED:\n{result.stderr}")
            return result.returncode

        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
