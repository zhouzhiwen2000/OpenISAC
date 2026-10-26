#include "UdpLinkTest.hpp"
#include <iostream>

using namespace udp_link_test;
static void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        std::vector<uint8_t> original(1316), received(1316);
        make_packet(original.data(), original.size(), 0x123456789ULL, 83);
        Header header;
        check(read_header(original.data(), original.size(), header), "valid packet rejected");
        check(header.session == 0x123456789ULL && header.sequence == 83, "metadata mismatch");
        received = original;
        received.back() ^= 0x81;
        received[96] ^= 0xff;
        check(read_header(received.data(), received.size(), header), "payload errors rejected");
        check(bit_errors(received.data(), original.data(), original.size()) == 10, "payload BER count");
        received = original;
        received[0] ^= 1; received[33] ^= 2; received[66] ^= 4;
        check(read_header(received.data(), received.size(), header), "majority header recovery failed");
        check(bit_errors(received.data(), original.data(), original.size()) == 3, "header BER count");
        received = original;
        for (size_t i = 0; i < kCopies; ++i) received[i * kHeaderBytes] ^= 1;
        check(!read_header(received.data(), received.size(), header), "unrecoverable header accepted");
        check(!read_header(original.data(), 90, header), "short datagram accepted");

        SequenceStats stats(8);
        check(stats.record(0) && stats.record(2), "initial sequence rejected");
        stats.confirm(4);
        check(stats.lost() == 2, "leading/interior/tail loss count");
        check(stats.record(1) && stats.lost() == 1 && stats.late == 1, "late recovery count");
        check(!stats.record(2) && stats.duplicates == 1, "duplicate included");
        check(stats.record(4) && stats.record(20), "large gap handling");
        stats.confirm(21);
        check(stats.unique == 5 && stats.lost() == 16, "bounded window lost accounting");
        check(!stats.record(1) && stats.too_old == 1, "expired duplicate accepted");
        check(stats.record(19) && stats.lost() == 15, "late packet inside retained window");
        SequenceStats empty(8);
        empty.confirm(1000000);
        check(!empty.started() && empty.expected() == 0 && empty.lost() == 0,
              "loss must wait for the first received packet");
        empty.reset_loss(1000000);
        check(!empty.started(), "reset must not start loss accounting");
        check(empty.record(999999), "first received packet after sender telemetry rejected");
        empty.confirm(1000000);
        check(empty.expected() == 1 && empty.lost() == 0, "leading loss was included");
        empty.confirm(1000005);
        check(empty.expected() == 6 && empty.lost() == 5, "blackout after first RX not counted");

        SequenceStats reset(32);
        reset.record(10); reset.record(12); reset.confirm(14);
        check(reset.expected() == 4 && reset.lost() == 2, "first RX sequence is not the origin");
        reset.reset_loss(16);
        check(reset.expected() == 0 && reset.lost() == 0 && reset.unique == 2,
              "loss reset changed cumulative receive statistics");
        reset.confirm(15);
        check(reset.record(11) && reset.lost() == 0, "old late packet changed reset loss");
        check(!reset.record(12), "loss reset forgot duplicate history");
        reset.record(16); reset.record(18); reset.confirm(20);
        check(reset.expected() == 4 && reset.lost() == 2, "post-reset loss denominator");
        reset.record(17);
        check(reset.lost() == 1, "post-reset late recovery");
        reset.reset_loss(); reset.confirm(23);
        check(reset.expected() == 3 && reset.lost() == 3, "reset after blackout moved origin backwards");
        std::cout << "UDP packet, bit-error and sequence accounting tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
