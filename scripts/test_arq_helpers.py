#!/usr/bin/env python3
"""
Self-contained protocol helper logic test.
Exercises ARQ window/framing behavior and ERTM5 payload serialization via a
C++ test binary that links against the shared helpers in OFDMCore.hpp.

Build & run:
    g++ -std=c++17 -I../include -o test_arq_helpers test_arq_helpers.cpp -lfftw3f
    ./test_arq_helpers
"""
import subprocess
import sys
import tempfile
from pathlib import Path

TEST_SOURCE = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <chrono>
#include <cassert>
#include <functional>
#include <utility>
#include "OFDMCore.hpp"
#include "ErtmAbsoluteFrameState.hpp"

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

// ---- Test ArqFeedback pack/unpack ----
void test_feedback_pack_unpack() {
    ArqFeedback fb;
    fb.direction = 1;
    fb.ack_base = 100;
    fb.ack_bitmap = 0xDEADBEEFCAFEBABEULL;

    std::vector<uint8_t> packed;
    fb.pack(packed);
    CHECK(packed.size() == ArqFeedback::kPayloadSize, "feedback pack size");
    CHECK(std::memcmp(packed.data(), "ARQ1", 4) == 0, "feedback magic");
    CHECK(packed[4] == 1, "feedback direction");

    ArqFeedback fb2;
    CHECK(ArqFeedback::try_unpack(packed.data(), packed.size(), fb2), "feedback unpack");
    CHECK(fb2.direction == 1, "feedback direction roundtrip");
    CHECK(fb2.ack_base == 100, "feedback ack_base roundtrip");
    CHECK(fb2.ack_bitmap == 0xDEADBEEFCAFEBABEULL, "feedback ack_bitmap roundtrip");

    CHECK(ArqFeedback::is_feedback(packed.data(), packed.size()), "is_feedback true");
    CHECK(!ArqFeedback::is_feedback(packed.data(), 3), "is_feedback short");
    std::vector<uint8_t> not_fb = {'N', 'O', 'P', 'E', 0};
    CHECK(!ArqFeedback::is_feedback(not_fb.data(), not_fb.size()), "is_feedback false");

    ArqFeedback fb3;
    CHECK(!ArqFeedback::try_unpack(packed.data(), 10, fb3), "feedback unpack short");
    printf("  test_feedback_pack_unpack: OK\n");
}

// ---- Test mini-header flags for ARQ feedback classification ----
void test_mini_header_flags() {
    LdpcMiniHeader data_hdr{
        LdpcPacketFraming::kVersion,
        LdpcPacketFraming::kFlags,
        15,
        LdpcPacketFraming::payload_blocks_field_for_len(15),
        7,
    };
    LdpcMiniHeader feedback_hdr = data_hdr;
    feedback_hdr.flags = LdpcPacketFraming::kFlagArqFeedback;
    LdpcMiniHeader ertm_hdr = data_hdr;
    ertm_hdr.flags = LdpcPacketFraming::kFlagErtmTiming;

    CHECK(!LdpcPacketFraming::is_arq_feedback(data_hdr), "data header not feedback");
    CHECK(LdpcPacketFraming::is_arq_feedback(feedback_hdr), "feedback header flagged");
    CHECK(!LdpcPacketFraming::is_ertm_timing(data_hdr), "data header not ertm");
    CHECK(LdpcPacketFraming::is_ertm_timing(ertm_hdr), "ertm header flagged");
    CHECK(LdpcPacketFraming::flags_are_known(LdpcPacketFraming::kFlags), "data flags known");
    CHECK(LdpcPacketFraming::flags_are_known(LdpcPacketFraming::kFlagArqFeedback), "feedback flags known");
    CHECK(LdpcPacketFraming::flags_are_known(LdpcPacketFraming::kFlagErtmTiming), "ertm flags known");
    // Identity SIB uses 0x04 (word0) and 0x08 (word1 / IdentityHi); both are known.
    CHECK(LdpcPacketFraming::flags_are_known(LdpcPacketFraming::kFlagAirFrameIdentity),
          "identity word0 flags known");
    CHECK(LdpcPacketFraming::flags_are_known(LdpcPacketFraming::kFlagAirFrameIdentityHi),
          "identity word1 flags known");
    CHECK(!LdpcPacketFraming::flags_are_known(0x10), "unknown flag rejected");

    const uint64_t packed_header = LdpcPacketFraming::pack_header(feedback_hdr);
    LdpcMiniHeader decoded;
    CHECK(LdpcPacketFraming::unpack_header(packed_header, decoded), "decode flagged header");
    CHECK(decoded.flags == LdpcPacketFraming::kFlagArqFeedback, "flag roundtrip");
    CHECK(decoded.seq == feedback_hdr.seq, "seq roundtrip");

    const uint64_t packed_ertm_header = LdpcPacketFraming::pack_header(ertm_hdr);
    CHECK(LdpcPacketFraming::unpack_header(packed_ertm_header, decoded), "decode ertm header");
    CHECK(decoded.flags == LdpcPacketFraming::kFlagErtmTiming, "ertm flag roundtrip");

    // Expanded Identity SIB: epoch32 + frame_seq32 via dual BCH words.
    {
        std::vector<int> identity_qpsk(LdpcPacketFraming::kIdentityControlSymbols);
        const uint32_t epoch32 = 0xA1B2C3D4u;
        const uint32_t seq32 = 0x11223344u;
        LdpcPacketFraming::write_identity_control_qpsk(
            epoch32, seq32, identity_qpsk.data());
        // Convert QPSK ints to soft LLRs for decode path (high confidence).
        std::vector<float> identity_llrs(identity_qpsk.size() * 2);
        for (size_t i = 0; i < identity_qpsk.size(); ++i) {
            const int q = identity_qpsk[i] & 3;
            identity_llrs[i * 2] = ((q >> 1) & 1) ? -8.0f : 8.0f;
            identity_llrs[i * 2 + 1] = (q & 1) ? -8.0f : 8.0f;
        }
        LdpcMiniHeader identity_hdr{};
        CHECK(LdpcPacketFraming::decode_identity_control_llrs(
                  identity_llrs.data() + LdpcPacketFraming::kMarkerBits,
                  identity_hdr),
              "decode identity control");
        CHECK(LdpcPacketFraming::is_air_frame_identity(identity_hdr),
              "identity word0 flag set");
        CHECK(identity_hdr.timeline_epoch == epoch32, "identity epoch roundtrip");
        CHECK(identity_hdr.air_frame_seq == seq32, "identity seq roundtrip");
    }

    LdpcMiniHeader bad_hdr = data_hdr;
    bad_hdr.flags = 0x10;  // outside kKnownFlagsMask
    bool threw = false;
    try {
        LdpcPacketFraming::pack_header(bad_hdr);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "pack rejects unknown flags");

    std::vector<uint8_t> user_payload = {
        'A', 'R', 'Q', '1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    CHECK(ArqFeedback::is_feedback(user_payload.data(), user_payload.size()),
          "legacy payload sniff would match ARQ1");
    CHECK(!LdpcPacketFraming::is_arq_feedback(data_hdr),
          "unflagged ARQ1 user payload remains data by header");

    printf("  test_mini_header_flags: OK\n");
}

// ---- Test ERTM5 absolute-frame + raw PPS-anchor payload pack/unpack ----
void test_ertm5_payload() {
    AlignedVector channel_freq = {
        {1.0f, -0.5f},
        {-2.25f, 3.5f},
        {0.0f, 4.0f},
        {-1.0f, -1.0f},
    };
    constexpr uint32_t kSeq = 0x89ABCDEFu;
    constexpr uint32_t kEpoch = 0xA1B2C3D4u;
    constexpr uint64_t kFrameSeq = 0x12345678ABCDEF01ULL;

    std::vector<uint8_t> packed;
    std::string error;
    CHECK(ErtmTimingPayload::pack(
              kSeq,
              static_cast<uint32_t>(channel_freq.size()),
              15.36e6,
              -37,
              kEpoch,
              kFrameSeq,
              987654321,
              123456,
              200000,
              200.0e6,
              channel_freq,
              packed,
              &error),
          "ERTM5 pack");
    CHECK(ErtmTimingPayload::looks_like(packed.data(), packed.size()),
          "ERTM5 magic");

    ErtmTimingPayloadView decoded;
    error.clear();
    CHECK(ErtmTimingPayload::unpack(
              packed.data(),
              packed.size(),
              static_cast<uint32_t>(channel_freq.size()),
              decoded,
              &error),
          "ERTM5 unpack");
    CHECK(decoded.seq == kSeq, "ERTM5 seq roundtrip");
    CHECK(decoded.timeline_epoch == kEpoch, "ERTM5 epoch roundtrip");
    CHECK(decoded.ul_measure_frame_seq == kFrameSeq,
          "ERTM5 64-bit frame roundtrip");
    CHECK(decoded.duti_samples == -37, "ERTM5 DUTI roundtrip");
    CHECK(decoded.sample_rate == 15.36e6, "ERTM5 sample rate roundtrip");
    CHECK(decoded.bs_pps_anchor_valid, "ERTM5 PPS anchor valid");
    CHECK(decoded.bs_last_pps_ticks == 987654321,
          "ERTM5 raw BS PPS ticks roundtrip");
    CHECK(decoded.bs_t0_to_pps_ticks == 123456, "ERTM5 PPS t0 roundtrip");
    CHECK(decoded.bs_frame_ticks == 200000, "ERTM5 frame ticks roundtrip");
    CHECK(decoded.bs_tick_rate == 200.0e6, "ERTM5 tick rate roundtrip");
    CHECK(decoded.channel_freq == channel_freq, "ERTM5 channel roundtrip");

    error.clear();
    CHECK(!ErtmTimingPayload::unpack(
              packed.data(), packed.size(), 8, decoded, &error),
          "ERTM5 rejects wrong FFT size");
    CHECK(error.find("fft_size mismatch") != std::string::npos,
          "ERTM5 wrong FFT error");

    const std::vector<uint8_t> legacy = {'E', 'R', 'T', 'M', '2'};
    error.clear();
    CHECK(!ErtmTimingPayload::unpack(
              legacy.data(), legacy.size(), 4, decoded, &error),
          "ERTM5 rejects legacy ERTM2");
    CHECK(error.find("legacy ERTM2/3/4") != std::string::npos,
          "ERTM2 rejection error");

    const std::vector<uint8_t> legacy_v3 = {'E', 'R', 'T', 'M', '3'};
    error.clear();
    CHECK(!ErtmTimingPayload::unpack(
              legacy_v3.data(), legacy_v3.size(), 4, decoded, &error),
          "ERTM5 rejects legacy ERTM3 without a PPS anchor");
    CHECK(error.find("legacy ERTM2/3/4") != std::string::npos,
          "ERTM3 rejection error");

    const std::vector<uint8_t> legacy_v4 = {'E', 'R', 'T', 'M', '4'};
    error.clear();
    CHECK(!ErtmTimingPayload::unpack(
              legacy_v4.data(), legacy_v4.size(), 4, decoded, &error),
          "ERTM5 rejects legacy ERTM4 without raw BS PPS ticks");
    CHECK(error.find("legacy ERTM2/3/4") != std::string::npos,
          "ERTM4 rejection error");

    // Convert the valid ERTM5 packet into the complete former ERTM4 wire
    // layout by changing the magic and removing only bs_last_pps_ticks.
    std::vector<uint8_t> complete_legacy_v4 = packed;
    complete_legacy_v4[4] = '4';
    constexpr size_t kRawBsPpsOffset = 5 + 4 + 8 + 4 + 4 + 4 + 8 + 4;
    complete_legacy_v4.erase(
        complete_legacy_v4.begin() + kRawBsPpsOffset,
        complete_legacy_v4.begin() + kRawBsPpsOffset + sizeof(int64_t));
    CHECK(complete_legacy_v4.size() == packed.size() - sizeof(int64_t),
          "complete ERTM4 payload has the former wire size");
    AirFrameIdentity dl_identity;
    dl_identity.timeline_epoch = kEpoch;
    dl_identity.frame_seq = kFrameSeq;
    dl_identity.valid = true;
    ErtmAbsoluteFrameState frame_state;
    frame_state.prepare_channel_buffers(channel_freq.size());
    const auto legacy_ingest = frame_state.ingest(
        std::move(complete_legacy_v4),
        dl_identity,
        static_cast<uint32_t>(channel_freq.size()));
    CHECK(legacy_ingest.status ==
              ErtmAbsoluteFrameState::IngestStatus::LegacyPayload &&
              frame_state.cache_size() == 0,
          "complete ERTM4 ingest is classified as legacy without caching");

    printf("  test_ertm5_payload: OK\n");
}

void test_ertm_pps_phase_formula() {
    ErtmTimingPayloadView report;
    report.sample_rate = 1.0e6;
    report.bs_pps_anchor_valid = true;
    report.bs_last_pps_ticks = 9'750'000;
    report.bs_t0_to_pps_ticks = 2'000'000; // r_B(0) = 0.2 s
    report.bs_frame_ticks = 100'000;        // frame = 0.01 s
    report.bs_tick_rate = 10.0e6;

    // N=5 gives r_B=0.25 s; UE boundary is 0.30 s after its PPS.
    const auto phase = compute_ertm_pps_phase(
        report,
        5,
        10'300'000'000LL,
        10'000'000'000LL,
        100.0,
        50.0);
    CHECK(phase.valid, "PPS phase formula accepts complete same-frame inputs");
    CHECK(std::abs(phase.r_bs_samples - 250'000.0) < 1e-9,
          "PPS phase formula reconstructs BS PPS-to-frame distance");
    CHECK(std::abs(phase.r_ue_samples - 300'000.0) < 1e-9,
          "PPS phase formula reconstructs UE PPS-to-frame distance");
    CHECK(std::abs(phase.phase_error_samples - (-49'950.0)) < 1e-9,
          "P_U-P_B follows tau_DL-TO_UE+r_B-r_U");
    CHECK(phase.bs_last_pps_ticks == report.bs_last_pps_ticks &&
              phase.bs_t0_to_pps_ticks == report.bs_t0_to_pps_ticks &&
              phase.ue_last_pps_ns == 10'000'000'000LL &&
              phase.bs_tick_rate == report.bs_tick_rate &&
              phase.sample_rate == report.sample_rate,
          "PPS phase diagnostics pass exact BS/UE anchors and rates through");

    CHECK(std::abs(wrap_pps_phase_samples(800'000.0, 1.0e6) - (-200'000.0)) < 1e-9,
          "PPS phase error wraps into the centered one-second interval");
    CHECK(std::abs(wrap_pps_phase_samples(500'000.0, 1.0e6) - (-500'000.0)) < 1e-9,
          "positive half-second maps to the negative interval endpoint");
    CHECK(std::abs(wrap_pps_phase_samples(-500'000.0, 1.0e6) - (-500'000.0)) < 1e-9,
          "negative half-second stays at the interval endpoint");
    CHECK(std::abs(wrap_pps_phase_samples(1'000'000.0, 1.0e6)) < 1e-9,
          "a whole second wraps to zero");

    const auto shifted_epoch = compute_ertm_pps_phase(
        report,
        5,
        110'300'000'000LL,
        110'000'000'000LL,
        100.0,
        50.0);
    CHECK(shifted_epoch.valid &&
              std::abs(shifted_epoch.phase_error_samples - phase.phase_error_samples) < 1e-9,
          "whole-second UE timestamp epoch changes do not alter PPS phase");

    const auto huge_frame = compute_ertm_pps_phase(
        report,
        std::numeric_limits<uint64_t>::max() - 5u,
        10'300'000'000LL,
        10'000'000'000LL,
        0.0,
        0.0);
    CHECK(huge_frame.valid &&
              std::abs(huge_frame.r_bs_samples - 300'000.0) < 1e-9,
          "BS PPS distance stays exact beyond the double integer range");

    report.bs_pps_anchor_valid = false;
    CHECK(!compute_ertm_pps_phase(
               report, 5, 10'300'000'000LL, 10'000'000'000LL, 100.0, 50.0).valid,
          "missing BS PPS anchor invalidates the complete phase observation");
    printf("  test_ertm_pps_phase_formula: OK\n");
}

// ---- Test UL-TX timing state boundary semantics ----
void test_ul_tx_timing_effective_frame() {
    constexpr uint64_t kFrame = 1000;
    CHECK(ul_tx_timing_effective_from_frame(kFrame, false, true) == kFrame,
          "new burst timing state applies to current frame");
    CHECK(ul_tx_timing_effective_from_frame(kFrame, true, true) == kFrame + 1,
          "continuous aggregate shift applies to next frame");
    CHECK(ul_tx_timing_effective_from_frame(kFrame, true, false) == kFrame,
          "canceling TADV and Alignment deltas apply to current frame");

    printf("  test_ul_tx_timing_effective_frame: OK\n");
}

// ---- Test ArqTxWindow ----
void test_tx_window() {
    NetworkOutputConfig net;
    net.arq_window_packets = 8;
    net.arq_retransmit_timeout_ms = 50;
    net.arq_max_retries = 3;
    net.arq_ack_bitmap_bits = 64;

    ArqTxWindow tx;
    tx.configure(net);
    tx.set_direction(0);

    const int64_t t0 = 1000;
    // Insert packets
    std::vector<uint8_t> payload = {1, 2, 3, 4, 5};
    CHECK(tx.try_insert(0, payload.data(), payload.size(), t0), "insert seq 0");
    CHECK(tx.try_insert(1, payload.data(), payload.size(), t0), "insert seq 1");
    CHECK(tx.try_insert(2, payload.data(), payload.size(), t0), "insert seq 2");
    CHECK(tx.outstanding_count() == 3, "outstanding count 3");

    // Check entry exists
    CHECK(tx.has_entry(0), "has_entry 0");
    CHECK(tx.has_entry(1), "has_entry 1");
    CHECK(!tx.has_entry(99), "entry 99 missing");

    // No retransmit yet (not timed out)
    std::vector<uint16_t> rtx;
    tx.get_retransmit(rtx, t0 + 10);
    CHECK(rtx.empty(), "no retransmit before RTO");

    // Retransmit after RTO
    tx.get_retransmit(rtx, t0 + 60);
    CHECK(rtx.size() == 3, "3 retransmit after RTO");

    // Mark transmitted
    tx.mark_transmitted(0, t0 + 60);
    ArqTxEntry copied;
    CHECK(tx.get_entry_copy(0, copied), "copy entry 0");
    CHECK(copied.retry_count == 1, "retry count 1");

    // Process ACK for seq 0 and 1
    ArqFeedback ack;
    ack.direction = 0;
    ack.ack_base = 2; // cumulative ack for seq < 2
    ack.ack_bitmap = 0;
    size_t released = tx.process_ack(ack, t0 + 70);
    CHECK(released == 2, "released 2 from cumulative ack");
    CHECK(tx.outstanding_count() == 1, "1 outstanding after ack");
    CHECK(!tx.has_entry(0), "seq 0 acked");
    CHECK(!tx.has_entry(1), "seq 1 acked");
    CHECK(tx.has_entry(2), "seq 2 still pending");

    // Selective ACK for seq 2
    ArqFeedback sack;
    sack.direction = 0;
    sack.ack_base = 2;
    sack.ack_bitmap = 1; // bit 0 = seq 2
    released = tx.process_ack(sack, t0 + 80);
    CHECK(released == 1, "released 1 from selective ack");
    CHECK(tx.outstanding_count() == 0, "0 outstanding after all acked");

    // Window full test
    for (int i = 0; i < 8; i++) {
        tx.try_insert(static_cast<uint16_t>(10 + i), payload.data(), payload.size(), t0 + 100);
    }
    CHECK(!tx.has_room(), "window full");
    CHECK(!tx.try_insert(99, payload.data(), payload.size(), t0 + 100), "insert rejects when full");

    printf("  test_tx_window: OK\n");
}

// ---- Test ArqRxWindow: unordered delivery ----
void test_rx_window_unordered() {
    NetworkOutputConfig net;
    net.arq_window_packets = 16;
    net.arq_ordered_delivery = false;
    net.arq_ack_bitmap_bits = 64;
    net.arq_feedback_interval_ms = 0;

    ArqRxWindow rx;
    rx.configure(net);
    rx.set_direction(1);

    std::vector<uint8_t> p1 = {1}, p2 = {2}, p3 = {3};

    // Receive packets out of order: 0, 2, 1
    CHECK(rx.process_received(0, p1.data(), p1.size()), "accept seq 0");
    CHECK(rx.process_received(2, p3.data(), p3.size()), "accept seq 2");
    CHECK(rx.process_received(1, p2.data(), p2.size()), "accept seq 1");

    // Duplicate
    CHECK(!rx.process_received(0, p1.data(), p1.size()), "reject dup seq 0");
    CHECK(!rx.process_received(2, p3.data(), p3.size()), "reject dup seq 2");

    CHECK(rx.accepted_count() == 3, "3 accepted");
    CHECK(rx.dup_count() == 2, "2 duplicates");

    // ACK should reflect all received
    ArqFeedback ack = rx.generate_ack();
    CHECK(ack.ack_base == 3, "ack_base advanced to 3");
    CHECK(ack.ack_bitmap == 0, "bitmap clear after cumulative ack");

    printf("  test_rx_window_unordered: OK\n");
}

// ---- Test ArqRxWindow: ordered delivery ----
void test_rx_window_ordered() {
    NetworkOutputConfig net;
    net.arq_window_packets = 16;
    net.arq_ordered_delivery = true;
    net.arq_ack_bitmap_bits = 64;
    net.arq_feedback_interval_ms = 0;

    ArqRxWindow rx;
    rx.configure(net);
    rx.set_direction(0);

    std::vector<uint8_t> p0 = {10}, p1 = {11}, p2 = {12}, p3 = {13};

    // Receive out of order: 0, 2, 3, 1
    CHECK(rx.process_received(0, p0.data(), p0.size()), "accept seq 0");
    CHECK(rx.process_received(2, p2.data(), p2.size()), "accept seq 2");
    CHECK(rx.process_received(3, p3.data(), p3.size()), "accept seq 3");
    CHECK(rx.process_received(1, p1.data(), p1.size()), "accept seq 1");

    // Get deliverable: should be 0, 1, 2, 3 in order
    std::vector<std::vector<uint8_t>> deliverable;
    rx.get_deliverable(deliverable);
    CHECK(deliverable.size() == 4, "4 deliverable after all received");
    if (deliverable.size() == 4) {
        CHECK(deliverable[0][0] == 10, "deliverable[0] = 10");
        CHECK(deliverable[1][0] == 11, "deliverable[1] = 11");
        CHECK(deliverable[2][0] == 12, "deliverable[2] = 12");
        CHECK(deliverable[3][0] == 13, "deliverable[3] = 13");
    }

    // Test gap: receive 4, 6, 7 (skip 5)
    std::vector<uint8_t> p4 = {14}, p6 = {16}, p7 = {17};
    CHECK(rx.process_received(4, p4.data(), p4.size()), "accept seq 4");
    CHECK(rx.process_received(6, p6.data(), p6.size()), "accept seq 6");
    CHECK(rx.process_received(7, p7.data(), p7.size()), "accept seq 7");

    // Only seq 4 is deliverable (contiguous from expected_seq)
    deliverable.clear();
    rx.get_deliverable(deliverable);
    CHECK(deliverable.size() == 1, "1 deliverable (seq 4)");
    if (!deliverable.empty()) {
        CHECK(deliverable[0][0] == 14, "deliverable[0] = 14");
    }

    // Now receive 5 - should unlock 5, 6, 7
    std::vector<uint8_t> p5 = {15};
    CHECK(rx.process_received(5, p5.data(), p5.size()), "accept seq 5");
    deliverable.clear();
    rx.get_deliverable(deliverable);
    CHECK(deliverable.size() == 3, "3 deliverable (seq 5, 6, 7)");
    if (deliverable.size() == 3) {
        CHECK(deliverable[0][0] == 15, "deliverable[0] = 15");
        CHECK(deliverable[1][0] == 16, "deliverable[1] = 16");
        CHECK(deliverable[2][0] == 17, "deliverable[2] = 17");
    }

    printf("  test_rx_window_ordered: OK\n");
}

// ---- Test ArqRxWindow: first accepted packet can start at nonzero seq ----
void test_rx_window_nonzero_first_seq() {
    NetworkOutputConfig net;
    net.arq_window_packets = 16;
    net.arq_ordered_delivery = false;
    net.arq_ack_bitmap_bits = 64;
    net.arq_feedback_interval_ms = 0;

    ArqRxWindow rx;
    rx.configure(net);
    rx.set_direction(0);

    std::vector<uint8_t> p10 = {10}, p11 = {11};
    CHECK(rx.process_received(965, p10.data(), p10.size()), "accept nonzero first seq");
    ArqFeedback ack = rx.generate_ack();
    CHECK(ack.ack_base == 966, "nonzero first seq advances ack_base");
    CHECK(ack.ack_bitmap == 0, "nonzero first seq cumulative bitmap clear");

    CHECK(rx.process_received(966, p11.data(), p11.size()), "accept next nonzero seq");
    ack = rx.generate_ack();
    CHECK(ack.ack_base == 967, "next nonzero seq advances ack_base");

    net.arq_ordered_delivery = true;
    ArqRxWindow ordered_rx;
    ordered_rx.configure(net);
    ordered_rx.set_direction(0);
    CHECK(ordered_rx.process_received(1200, p10.data(), p10.size()),
          "ordered accept nonzero first seq");
    std::vector<std::vector<uint8_t>> deliverable;
    ordered_rx.get_deliverable(deliverable);
    CHECK(deliverable.size() == 1, "ordered deliver nonzero first seq");
    CHECK(deliverable[0][0] == 10, "ordered nonzero payload");

    printf("  test_rx_window_nonzero_first_seq: OK\n");
}

// ---- Test ArqRxWindow: skip old gaps when future seq exceeds RX window ----
void test_rx_window_skip_ahead() {
    NetworkOutputConfig net;
    net.arq_window_packets = 4;
    net.arq_ordered_delivery = false;
    net.arq_ack_bitmap_bits = 64;
    net.arq_feedback_interval_ms = 0;

    ArqRxWindow rx;
    rx.configure(net);
    rx.set_direction(0);

    std::vector<uint8_t> p0 = {0}, p1 = {1}, p6 = {6};
    CHECK(rx.process_received(0, p0.data(), p0.size()), "skip accept seq 0");
    CHECK(rx.process_received(1, p1.data(), p1.size()), "skip accept seq 1");
    CHECK(rx.process_received(6, p6.data(), p6.size()),
          "skip accepts future seq outside window");
    CHECK(rx.skip_count() == 1, "skip count increments");
    ArqFeedback ack = rx.generate_ack();
    CHECK(ack.ack_base == 3, "skip advances ack_base to include future seq");
    CHECK(ack.ack_bitmap == (1ULL << 3), "skip ACKs future seq selectively");

    net.arq_ordered_delivery = true;
    ArqRxWindow ordered_rx;
    ordered_rx.configure(net);
    ordered_rx.set_direction(0);
    CHECK(ordered_rx.process_received(0, p0.data(), p0.size()), "ordered skip accept 0");
    std::vector<std::vector<uint8_t>> deliverable;
    ordered_rx.get_deliverable(deliverable);
    CHECK(deliverable.size() == 1, "ordered skip delivers initial seq");
    CHECK(ordered_rx.process_received(6, p6.data(), p6.size()),
          "ordered skip accepts future seq outside window");
    CHECK(ordered_rx.expected_seq() == 3, "ordered skip advances expected seq");
    deliverable.clear();
    ordered_rx.get_deliverable(deliverable);
    CHECK(deliverable.empty(), "ordered skip waits for new base after skip");

    printf("  test_rx_window_skip_ahead: OK\n");
}

// ---- Test feedback frames do not create data-window gaps ----
void test_feedback_frame_skips_data_window() {
    NetworkOutputConfig net;
    net.arq_window_packets = 64;
    net.arq_ordered_delivery = true;
    net.arq_ack_bitmap_bits = 64;
    net.arq_feedback_interval_ms = 0;

    ArqRxWindow rx;
    rx.configure(net);
    rx.set_direction(0);

    std::vector<uint8_t> p0 = {10}, p1 = {11}, p2 = {12};
    CHECK(rx.process_received(0, p0.data(), p0.size()), "accept data seq 0");
    // A flagged feedback frame may carry any feedback seq, but callers must not
    // register it in the data RX window. Data seq 1 should remain next.
    LdpcMiniHeader feedback_hdr{
        LdpcPacketFraming::kVersion,
        LdpcPacketFraming::kFlagArqFeedback,
        ArqFeedback::kPayloadSize,
        LdpcPacketFraming::payload_blocks_field_for_len(ArqFeedback::kPayloadSize),
        500,
    };
    CHECK(LdpcPacketFraming::is_arq_feedback(feedback_hdr), "interleaved feedback flagged");
    CHECK(rx.process_received(1, p1.data(), p1.size()), "accept data seq 1 after feedback");
    CHECK(rx.process_received(2, p2.data(), p2.size()), "accept data seq 2 after feedback");

    ArqFeedback ack = rx.generate_ack();
    CHECK(ack.ack_base == 3, "ack_base advances across data only");
    CHECK(ack.ack_bitmap == 0, "bitmap clear after contiguous data");

    std::vector<std::vector<uint8_t>> deliverable;
    rx.get_deliverable(deliverable);
    CHECK(deliverable.size() == 3, "ordered delivery not blocked by feedback");
    if (deliverable.size() == 3) {
        CHECK(deliverable[0][0] == 10, "feedback skip deliver 0");
        CHECK(deliverable[1][0] == 11, "feedback skip deliver 1");
        CHECK(deliverable[2][0] == 12, "feedback skip deliver 2");
    }

    printf("  test_feedback_frame_skips_data_window: OK\n");
}

// ---- Test ARQ window is clamped below the 16-bit sequence half-space ----
void test_window_clamp_sequence_half_space() {
    NetworkOutputConfig defaults;
    CHECK(defaults.arq_window_packets == 256, "default ARQ window remains 256");

    NetworkOutputConfig net;
    net.arq_window_packets = 65536;
    net.arq_ack_bitmap_bits = 64;
    normalize_arq_config(net);
    CHECK(net.arq_window_packets == 32767, "normalize clamps ARQ window to 32767");

    ArqTxWindow tx;
    tx.configure(net);
    CHECK(tx.window_size() == 32767, "tx window size 32767");

    ArqRxWindow rx;
    rx.configure(net);
    std::vector<uint8_t> payload = {1};
    CHECK(rx.process_received(0, payload.data(), payload.size()), "rx accepts seq 0");
    CHECK(rx.process_received(64, payload.data(), payload.size()), "rx accepts seq 64 at bitmap edge");
    CHECK(rx.process_received(65, payload.data(), payload.size()), "rx skips to accept seq 65 outside old bitmap");
    CHECK(rx.skip_count() == 1, "rx skip count after bitmap edge");
    ArqFeedback ack = rx.generate_ack();
    CHECK(ack.ack_base == 2, "rx skip keeps ACK bitmap within 64 bits");

    printf("  test_window_clamp_sequence_half_space: OK\n");
}

// ---- Test sequence diff helpers ----
void test_seq_helpers() {
    CHECK(arq_seq_diff(5, 3) == 2, "seq_diff 5-3");
    CHECK(arq_seq_diff(3, 5) == -2, "seq_diff 3-5");
    CHECK(arq_seq_diff(0, 65535) == 1, "seq_diff wrap forward");
    CHECK(arq_seq_diff(65535, 0) == -1, "seq_diff wrap backward");
    CHECK(arq_seq_leq(3, 5), "seq_leq 3<=5");
    CHECK(arq_seq_leq(5, 5), "seq_leq 5<=5");
    CHECK(!arq_seq_leq(6, 5), "seq_leq 6>5");

    printf("  test_seq_helpers: OK\n");
}

// ---- Test TX window with encoded QPSK ----
void test_tx_window_encoded() {
    NetworkOutputConfig net;
    net.arq_window_packets = 4;
    net.arq_retransmit_timeout_ms = 100;
    net.arq_max_retries = 0;

    ArqTxWindow tx;
    tx.configure(net);

    AlignedIntVector encoded;
    encoded.push_back(0);
    encoded.push_back(1);
    encoded.push_back(2);
    encoded.push_back(3);

    std::vector<uint8_t> payload = {0xAA, 0xBB};
    CHECK(tx.try_insert_encoded(0, payload.data(), payload.size(),
                                 AlignedIntVector(encoded), 1000), "insert encoded");

    ArqTxEntry e;
    CHECK(tx.get_entry_copy(0, e), "entry exists");
    CHECK(e.encoded_qpsk.size() == 4, "encoded qpsk size");
    CHECK(e.raw_payload.size() == 2, "raw payload size");
    CHECK(e.encoded_qpsk[0] == 0, "encoded qpsk[0]");

    printf("  test_tx_window_encoded: OK\n");
}

// ---- Test drop_abandoned ----
void test_drop_abandoned() {
    NetworkOutputConfig net;
    net.arq_window_packets = 8;
    net.arq_retransmit_timeout_ms = 50;
    net.arq_max_retries = 2;

    ArqTxWindow tx;
    tx.configure(net);

    std::vector<uint8_t> p = {1};
    tx.try_insert(0, p.data(), p.size(), 1000);
    tx.try_insert(1, p.data(), p.size(), 1000);

    // Exhaust retries
    tx.mark_transmitted(0, 1000);
    tx.mark_transmitted(0, 1060);
    tx.mark_transmitted(1, 1000);
    tx.mark_transmitted(1, 1060);

    // drop_abandoned should remove them
    size_t dropped = tx.drop_abandoned(1120);
    CHECK(dropped == 2, "dropped 2 abandoned");
    CHECK(tx.outstanding_count() == 0, "0 outstanding after drop");

    printf("  test_drop_abandoned: OK\n");
}

int main() {
    printf("Running ARQ helper tests...\n");
    test_seq_helpers();
    test_feedback_pack_unpack();
    test_mini_header_flags();
    test_ertm5_payload();
    test_ertm_pps_phase_formula();
    test_ul_tx_timing_effective_frame();
    test_tx_window();
    test_tx_window_encoded();
    test_rx_window_unordered();
    test_rx_window_ordered();
    test_rx_window_nonzero_first_seq();
    test_rx_window_skip_ahead();
    test_feedback_frame_skips_data_window();
    test_window_clamp_sequence_half_space();
    test_drop_abandoned();

    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
"""

def main():
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="openisac-arq-test-") as tmp:
        tmp_path = Path(tmp)
        test_cpp = tmp_path / "test_arq_helpers.cpp"
        test_bin = tmp_path / "test_arq_helpers"
        test_cpp.write_text(TEST_SOURCE)

        build_cmd = [
            "g++", "-std=c++17", "-O2",
            f"-I{repo_root / 'include'}",
            "-o", str(test_bin), str(test_cpp), str(repo_root / "src" / "AsyncLogger.cpp"),
            "-lfftw3f", "-lpthread"
        ]
        print(f"Building: {' '.join(build_cmd)}")
        result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"BUILD FAILED:\n{result.stderr}")
            return 1

        print(f"\nRunning: {test_bin}")
        result = subprocess.run([str(test_bin)], capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")

        return result.returncode

if __name__ == "__main__":
    sys.exit(main())
