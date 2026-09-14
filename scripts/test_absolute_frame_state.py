#!/usr/bin/env python3
"""Build and run deterministic shared absolute-frame state tests."""

import subprocess
import sys
import tempfile
from pathlib import Path


TEST_SOURCE = r"""
#include <cmath>
#include <atomic>
#include <cstdio>
#include <limits>
#include <thread>
#include <vector>

#include "AirFrameTxGate.hpp"
#include "ErtmAbsoluteFrameState.hpp"

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

static std::vector<uint8_t> make_report(
    uint32_t seq, uint32_t epoch, uint64_t measure_frame)
{
    AlignedVector channel(4, std::complex<float>(1.0f, 0.0f));
    std::vector<uint8_t> payload;
    std::string error;
    CHECK(ErtmTimingPayload::pack(
              seq, 4, 1.0e6, 3, epoch, measure_frame,
              9'000'000, 125'000, 10'000, 1.0e6,
              channel, payload, &error),
          "pack ERTM5 report");
    return payload;
}

static void test_air_frame_tx_gate() {
    AirFrameTxGate gate(100, 10);
    gate.configure_timing(true, 1.0e6, 1.0e6);

    AirFrameIdentity stamp;
    stamp.timeline_epoch = 7;
    stamp.frame_seq = 10;
    stamp.frame_start_symbol = 100;
    stamp.ue_frame_start_time_ns = 1'000'000'000;
    stamp.valid = true;
    const auto update = gate.store_stamp(stamp);
    CHECK(update.generation_bumped && update.generation == 2,
          "valid stamp bumps generation");
    CHECK(update.next_claim_seq == 11,
          "first claim follows completed RX frame");

    const auto claim = gate.claim_period();
    CHECK(claim.identity.valid && claim.identity.frame_seq == 11,
          "claim uses shared absolute cursor");
    const auto expected = gate.identity_for_grid_ticks(1'000'100, 0);
    CHECK(expected.valid && expected.frame_seq == 11,
          "timed grid derives expected absolute frame");
    const auto commit = gate.validate_for_send(
        claim.identity, claim.generation, 1'000'100, 0, true);
    CHECK(commit.decision == AirFrameTxGate::Decision::Match,
          "claimed frame passes final hard gate");

    AirFrameIdentity stale = claim.identity;
    stale.frame_seq = 10;
    CHECK(AirFrameTxGate::classify(
              stale, claim.generation, expected, claim.generation, true) ==
              AirFrameTxGate::Decision::Stale,
          "hard gate classifies stale frame");
    AirFrameIdentity future = claim.identity;
    future.frame_seq = 12;
    CHECK(AirFrameTxGate::classify(
              future, claim.generation, expected, claim.generation, true) ==
              AirFrameTxGate::Decision::Future,
          "hard gate classifies future frame");
    CHECK(AirFrameTxGate::classify(
              claim.identity, claim.generation - 1, expected,
              claim.generation, true) ==
              AirFrameTxGate::Decision::CancelledGeneration,
          "hard gate rejects cancelled generation");

    AirFrameClock clock;
    clock.configure(100, 10, 1.0e6, 1.0e6);
    clock.force_next_seq(5);
    CHECK(clock.advance_next_seq_past(10) == 11,
          "blank recovery advances cursor atomically");
    CHECK(clock.claim_next().frame_seq == 11,
          "claim observes advanced cursor");
    CHECK(clock.advance_next_seq_past(10) == 12,
          "blank recovery never rewinds a newer claim");
}

static void test_air_frame_claim_advance_concurrency() {
    AirFrameClock clock;
    clock.configure(100, 10, 1.0e6, 1.0e6);
    clock.ensure_epoch_started();

    bool invariant_held = true;
    constexpr int kIterations = 1000;
    for (int i = 0; i < kIterations; ++i) {
        clock.force_next_seq(0);
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        uint64_t claimed_seq = std::numeric_limits<uint64_t>::max();

        std::thread claimant([&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            claimed_seq = clock.claim_next().frame_seq;
        });
        std::thread advancer([&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            clock.advance_next_seq_past(0);
        });
        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        go.store(true, std::memory_order_release);
        claimant.join();
        advancer.join();

        const uint64_t cursor = clock.next_seq();
        if (claimed_seq > 1 || cursor != claimed_seq + 1) {
            invariant_held = false;
            break;
        }
    }
    CHECK(invariant_held,
          "claim_next and advance_next_seq_past share one atomic cursor critical section");
}

static void test_air_frame_gate_concurrency() {
    AirFrameTxGate gate(100, 10);
    gate.configure_timing(true, 1.0e6, 1.0e6);

    bool claim_advance_invariant = true;
    constexpr int kIterations = 500;
    for (int i = 0; i < kIterations; ++i) {
        gate.store_stamp(AirFrameIdentity{});
        AirFrameIdentity stamp;
        stamp.timeline_epoch = 7;
        stamp.frame_seq = 0;
        stamp.frame_start_symbol = 0;
        stamp.ue_frame_start_time_ns = 1'000'000'000;
        stamp.valid = true;
        const auto update = gate.store_stamp(stamp);

        AirFrameIdentity elapsed;
        elapsed.timeline_epoch = 7;
        elapsed.frame_seq = 1;
        elapsed.valid = true;
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        AirFrameTxGate::Claim raced_claim;
        std::thread claimant([&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            raced_claim = gate.claim_period();
        });
        std::thread advancer([&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            gate.advance_claim_past(elapsed, update.generation);
        });
        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        go.store(true, std::memory_order_release);
        claimant.join();
        advancer.join();

        const auto following_claim = gate.claim_period();
        if (!raced_claim.identity.valid || !following_claim.identity.valid ||
            raced_claim.generation != update.generation ||
            following_claim.generation != update.generation ||
            following_claim.identity.frame_seq !=
                raced_claim.identity.frame_seq + 1) {
            claim_advance_invariant = false;
            break;
        }
    }
    CHECK(claim_advance_invariant,
          "AirFrameTxGate claim and advance share one cursor critical section");

    bool stamp_advance_invariant = true;
    for (int i = 0; i < kIterations; ++i) {
        AirFrameIdentity old_stamp;
        old_stamp.timeline_epoch = 7;
        old_stamp.frame_seq = 0;
        old_stamp.ue_frame_start_time_ns = 1'000'000'000;
        old_stamp.valid = true;
        const auto old_update = gate.store_stamp(old_stamp);

        AirFrameIdentity old_elapsed;
        old_elapsed.timeline_epoch = 7;
        old_elapsed.frame_seq = 5;
        old_elapsed.valid = true;
        AirFrameIdentity new_stamp;
        new_stamp.timeline_epoch = 8;
        new_stamp.frame_seq = 10;
        new_stamp.frame_start_symbol = 100;
        new_stamp.ue_frame_start_time_ns = 1'000'001'000;
        new_stamp.valid = true;

        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        AirFrameTxGate::StampUpdate new_update;
        std::thread publisher([&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            new_update = gate.store_stamp(new_stamp);
        });
        std::thread advancer([&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            gate.advance_claim_past(old_elapsed, old_update.generation);
        });
        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        go.store(true, std::memory_order_release);
        publisher.join();
        advancer.join();

        const auto claim = gate.claim_period();
        if (!new_update.generation_bumped || !claim.identity.valid ||
            claim.identity.timeline_epoch != 8 ||
            claim.identity.frame_seq != 11 ||
            claim.generation != new_update.generation) {
            stamp_advance_invariant = false;
            break;
        }
    }
    CHECK(stamp_advance_invariant,
          "AirFrameTxGate stamp reanchor wins over old-generation advance");
}

static void test_ertm_absolute_frame_state() {
    ErtmAbsoluteFrameState state;
    state.prepare_channel_buffers(4);

    AirFrameIdentity dl;
    dl.timeline_epoch = 7;
    dl.frame_seq = 10;
    dl.valid = true;

    CHECK(state.ingest(make_report(1, 7, 12), dl, 4).accepted(),
          "cache accepts future report");
    CHECK(state.ingest(make_report(2, 7, 8), dl, 4).accepted(),
          "cache accepts usable report");
    size_t future_count = 0;
    auto* selected = state.select_for_downlink(dl, future_count);
    CHECK(selected != nullptr && selected->view.ul_measure_frame_seq == 8,
          "selection chooses newest M not newer than N");
    CHECK(future_count == 1, "selection counts future reports");
    CHECK(state.is_unused(*selected), "new report starts unused");

    uint64_t compute_generation = 99;
    CHECK(state.begin_compute(compute_generation) && compute_generation == 0,
          "compute freezes applied clear generation");
    const auto publish = state.publish(
        compute_generation,
        1.25,
        7,
        selected->view.ul_measure_frame_seq,
        dl,
        -4,
        9,
        true,
        -12.5,
        250.0,
        300.0,
        9'000'000,
        125'000,
        8'000'000'000LL,
        1.0e6,
        1.0e6);
    CHECK(publish.accepted && !publish.had_previous,
          "first TO publication succeeds");
    const auto snapshot = state.snapshot();
    CHECK(snapshot.available && std::abs(snapshot.to_ue_samples - 1.25) < 1e-12,
          "TO snapshot is internally consistent");
    CHECK(snapshot.pub_gen == 1 && snapshot.clear_gen == snapshot.request_gen,
          "TO snapshot publishes generations together");
    CHECK(snapshot.pps_phase_available &&
              std::abs(snapshot.pps_phase_error_samples - (-12.5)) < 1e-12 &&
              std::abs(snapshot.r_bs_samples - 250.0) < 1e-12 &&
              std::abs(snapshot.r_ue_samples - 300.0) < 1e-12 &&
              snapshot.bs_last_pps_ticks == 9'000'000 &&
              snapshot.bs_t0_to_pps_ticks == 125'000 &&
              snapshot.ue_last_pps_ns == 8'000'000'000LL &&
              snapshot.bs_tick_rate == 1.0e6 &&
              snapshot.sample_rate == 1.0e6,
          "PPS phase and both frame/PPS distances publish atomically with TO");
    CHECK(snapshot.rx_alignment_shift_samples == -4 &&
              snapshot.dl_identity.valid &&
              snapshot.dl_identity.timeline_epoch == dl.timeline_epoch &&
              snapshot.dl_identity.frame_seq == dl.frame_seq &&
              snapshot.sync_generation == 9,
          "TO snapshot publishes same-frame Alignment and identity together");
    CHECK(!state.is_unused(*selected), "successful TO marks M used");

    dl.frame_seq = 12;
    selected = state.select_for_downlink(dl, future_count);
    CHECK(selected != nullptr && selected->view.ul_measure_frame_seq == 12,
          "future report activates when N catches M");
    size_t history_miss_callbacks = 0;
    const auto state_m = state.select_with_state_m(
        dl,
        [](uint32_t, uint64_t frame, UlTxTimingStateEntry& timing_state) {
            timing_state.effective_from_frame = frame;
            return frame == 8;
        },
        [&history_miss_callbacks](const ErtmTimingPayloadView&) {
            ++history_miss_callbacks;
        });
    CHECK(state_m.report != nullptr &&
              state_m.report->view.ul_measure_frame_seq == 8 &&
              state_m.state_m.effective_from_frame == 8,
          "state(M) selection falls back to older cached report");
    CHECK(state_m.history_miss_count == 1 && history_miss_callbacks == 1,
          "state(M) selection reports rolled-out timing history");

    const uint64_t clear_generation = state.request_clear();
    CHECK(clear_generation == 1 && !state.snapshot().available,
          "clear immediately invalidates published TO");
    CHECK(state.enqueue(make_report(3, 7, 13)),
          "post-clear report enters shared generation queue");
    ErtmAbsoluteFrameState::QueuedPayload queued;
    CHECK(state.try_pop(queued) && queued.clear_generation == clear_generation,
          "queued report carries clear generation");
    const auto clear = state.apply_pending_clear();
    CHECK(clear.applied && clear.generation == clear_generation &&
              clear.invalidated_reports == 1,
          "consumer clear wipes cache exactly once");
    CHECK(!state.publish(0, 2.0, 7, 12, dl, 3, 10).accepted,
          "pre-clear computation cannot resurrect TO");
}

int main() {
    test_air_frame_tx_gate();
    test_air_frame_claim_advance_concurrency();
    test_air_frame_gate_concurrency();
    test_ertm_absolute_frame_state();
    std::printf("Absolute-frame state tests: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
"""


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="openisac-frame-state-test-") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "test_absolute_frame_state.cpp"
        binary = tmp_path / "test_absolute_frame_state"
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
