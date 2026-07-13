#!/usr/bin/env python3
"""
Self-contained ARQ helper logic test.
Exercises: duplicate suppression, ACK bitmap, retransmit timeout selection,
ordered/unordered delivery - all via a C++ test binary that links against
the shared ARQ helpers in OFDMCore.hpp.

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
#include "OFDMCore.hpp"

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
    CHECK(!LdpcPacketFraming::flags_are_known(0x08), "unknown flag rejected");

    const uint64_t packed_header = LdpcPacketFraming::pack_header(feedback_hdr);
    LdpcMiniHeader decoded;
    CHECK(LdpcPacketFraming::unpack_header(packed_header, decoded), "decode flagged header");
    CHECK(decoded.flags == LdpcPacketFraming::kFlagArqFeedback, "flag roundtrip");
    CHECK(decoded.seq == feedback_hdr.seq, "seq roundtrip");

    const uint64_t packed_ertm_header = LdpcPacketFraming::pack_header(ertm_hdr);
    CHECK(LdpcPacketFraming::unpack_header(packed_ertm_header, decoded), "decode ertm header");
    CHECK(decoded.flags == LdpcPacketFraming::kFlagErtmTiming, "ertm flag roundtrip");

    LdpcMiniHeader bad_hdr = data_hdr;
    bad_hdr.flags = 0x08;
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
