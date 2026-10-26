#include "UdpLinkTest.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <yaml-cpp/yaml.h>
#include <zmq.hpp>

using Clock = std::chrono::steady_clock;
using namespace udp_link_test;
static volatile std::sig_atomic_t running = 1;
static void stop(int) { running = 0; }

struct Options {
    bool tx = false, quiet = false;
    std::string host = "127.0.0.1", telemetry, control, sender_telemetry;
    int port = 0;
    size_t packet_bytes = 1316;
    double rate_mbps = 50, duration_s = 0, grace_s = 1;
    uint64_t session = 0, count = 0;
};

static std::string hex_id(uint64_t id) {
    std::ostringstream s;
    s << std::hex << std::setw(16) << std::setfill('0') << id;
    return s.str();
}

static Options parse(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help") {
        std::cout << "UdpLinkTest tx|rx [options]\n"
          "  --host HOST             UDP destination (tx) or bind address (rx)\n"
          "  --port PORT             Default: tx 50000, rx 50001\n"
          "  --rate-mbps RATE        UDP datagram bit rate, default 50; 0 pauses TX\n"
          "  --packet-bytes BYTES    Whole UDP datagram size, default 1316 (104..65507)\n"
          "  --telemetry ENDPOINT    PUB bind: tx tcp://127.0.0.1:18100, rx :18101\n"
          "  --control ENDPOINT      REP bind: TX :18102 (RATE), RX :18103 (RESET_LOSS/RESET_ERRORS), localhost by default\n"
          "  --sender-telemetry EP   RX SUB connect, default tcp://127.0.0.1:18100\n"
          "  --session HEX          Shared session ID; default TX random, RX automatic\n"
          "  --loss-grace SEC       Reordering grace, default 1 (0.1..10)\n"
          "  --count PACKETS        Stop TX after this many successful datagrams\n"
          "  --duration SEC         Stop after elapsed seconds (0 = unlimited)\n"
          "  --quiet                Suppress periodic JSON on stdout\n";
        std::exit(0);
    }
    Options o;
    const std::string role = argv[1];
    if (role != "tx" && role != "rx") throw std::invalid_argument("Expected tx or rx");
    o.tx = role == "tx";
    o.port = o.tx ? 50000 : 50001;
    o.telemetry = o.tx ? "tcp://127.0.0.1:18100" : "tcp://127.0.0.1:18101";
    o.control = o.tx ? "tcp://127.0.0.1:18102" : "tcp://127.0.0.1:18103";
    o.sender_telemetry = "tcp://127.0.0.1:18100";
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--quiet") { o.quiet = true; continue; }
        if (++i == argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[i];
        if (key == "--host") o.host = value;
        else if (key == "--port") o.port = std::stoi(value);
        else if (key == "--rate-mbps") o.rate_mbps = std::stod(value);
        else if (key == "--packet-bytes") o.packet_bytes = std::stoul(value);
        else if (key == "--telemetry") o.telemetry = value;
        else if (key == "--control") o.control = value;
        else if (key == "--sender-telemetry") o.sender_telemetry = value;
        else if (key == "--session") o.session = std::stoull(value, nullptr, 16);
        else if (key == "--loss-grace") o.grace_s = std::stod(value);
        else if (key == "--count") o.count = std::stoull(value);
        else if (key == "--duration") o.duration_s = std::stod(value);
        else throw std::invalid_argument("Unknown option " + key);
    }
    if (o.port < 1 || o.port > 65535 || o.packet_bytes < kPrefixBytes + 8 ||
        o.packet_bytes > 65507 || !std::isfinite(o.rate_mbps) || o.rate_mbps < 0 ||
        o.rate_mbps > 10000 || !std::isfinite(o.duration_s) || o.duration_s < 0 ||
        !std::isfinite(o.grace_s) || o.grace_s < .1 || o.grace_s > 10)
        throw std::invalid_argument("Option outside supported range");
    if (o.tx && !o.session) {
        std::random_device random;
        o.session = (uint64_t(random()) << 32) ^ random();
        if (!o.session) o.session = 1;
    }
    return o;
}

class UdpSocket {
public:
    explicit UdpSocket(const Options& o) {
        addrinfo hints{};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_DGRAM;
        addrinfo* result = nullptr;
        const int error = getaddrinfo(o.host.c_str(), std::to_string(o.port).c_str(), &hints, &result);
        if (error) throw std::runtime_error(gai_strerror(error));
        std::memcpy(&address, result->ai_addr, sizeof(address));
        freeaddrinfo(result);
        fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (fd < 0) throw std::runtime_error("Cannot create UDP socket");
        const int buffer = 16 * 1024 * 1024;
        setsockopt(fd, SOL_SOCKET, o.tx ? SO_SNDBUF : SO_RCVBUF, &buffer, sizeof(buffer));
        if (!o.tx && bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address))) {
            const std::string message = std::strerror(errno);
            close(fd); fd = -1;
            throw std::runtime_error("UDP bind: " + message);
        }
    }
    ~UdpSocket() { if (fd >= 0) close(fd); }
    UdpSocket(const UdpSocket&) = delete;
    UdpSocket& operator=(const UdpSocket&) = delete;
    int fd = -1;
    sockaddr_in address{};
};

class Telemetry {
public:
    explicit Telemetry(const Options& o) : socket(context, zmq::socket_type::pub), quiet(o.quiet) {
        socket.set(zmq::sockopt::linger, 0);
        socket.set(zmq::sockopt::sndhwm, 4);
        socket.bind(o.telemetry);
    }
    void publish(const std::string& data) {
        (void)socket.send(zmq::buffer(data), zmq::send_flags::dontwait);
        if (!quiet) std::cout << data << std::endl;
    }
    zmq::context_t context{1};
    zmq::socket_t socket;
    bool quiet;
};

static double seconds(Clock::duration delta) { return std::chrono::duration<double>(delta).count(); }

static int transmit(Options o) {
    UdpSocket udp(o);
    Telemetry telemetry(o);
    zmq::socket_t control(telemetry.context, zmq::socket_type::rep);
    control.set(zmq::sockopt::linger, 0);
    control.bind(o.control);
    std::vector<uint8_t> packet(o.packet_bytes);
    const auto began = Clock::now();
    auto next_send = began, last_publish = began;
    uint64_t sent = 0, bytes = 0, errors = 0, last_bytes = 0;
    double rate = 0;
    auto report = [&](bool active) {
        std::ostringstream s;
        s << std::setprecision(12) << "{\"version\":1,\"role\":\"tx\",\"session\":\"" << hex_id(o.session)
          << "\",\"running\":" << (active ? "true" : "false") << ",\"sent_packets\":" << sent
          << ",\"sent_bytes\":" << bytes << ",\"send_errors\":" << errors
          << ",\"packet_bytes\":" << o.packet_bytes << ",\"target_mbps\":" << o.rate_mbps
          << ",\"tx_mbps\":" << rate << ",\"elapsed_s\":" << seconds(Clock::now() - began) << '}';
        return s.str();
    };
    telemetry.publish(report(true));
    while (running) {
        auto now = Clock::now();
        if ((o.duration_s > 0 && seconds(now - began) >= o.duration_s) || (o.count && sent >= o.count)) break;
        zmq::message_t request;
        if (control.recv(request, zmq::recv_flags::dontwait)) {
            std::istringstream input(request.to_string());
            std::string command, extra;
            double target = -1;
            const bool valid = bool(input >> command >> target) && !(input >> extra) && command == "RATE" &&
                               std::isfinite(target) && target >= 0 && target <= 10000;
            if (valid) { o.rate_mbps = target; next_send = now; }
            const std::string response = valid ? "{\"ok\":true}" : "{\"ok\":false}";
            (void)control.send(zmq::buffer(response), zmq::send_flags::none);
        }
        if (o.rate_mbps > 0 && now >= next_send) {
            make_packet(packet.data(), packet.size(), o.session, sent);
            const auto n = sendto(udp.fd, packet.data(), packet.size(), MSG_DONTWAIT,
                                 reinterpret_cast<const sockaddr*>(&udp.address), sizeof(udp.address));
            if (n == static_cast<ssize_t>(packet.size())) { ++sent; bytes += packet.size(); }
            else ++errors;
            const auto interval = std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(packet.size() * 8.0 / (o.rate_mbps * 1e6)));
            next_send += interval;
            // Do not release a large catch-up burst after descheduling.
            if (now - next_send > 2 * interval) next_send = now + interval;
        }
        now = Clock::now();
        if (seconds(now - last_publish) >= .2) {
            rate = (bytes - last_bytes) * 8.0 / seconds(now - last_publish) / 1e6;
            telemetry.publish(report(true));
            last_publish = now; last_bytes = bytes;
        }
        if (o.rate_mbps <= 0) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        else if (next_send > now)
            std::this_thread::sleep_until(std::min(next_send, now + std::chrono::milliseconds(5)));
    }
    for (int i = 0; i < 3; ++i) {
        telemetry.publish(report(false));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}

static int receive(const Options& o) {
    UdpSocket udp(o);
    Telemetry telemetry(o);
    zmq::socket_t control(telemetry.context, zmq::socket_type::rep);
    control.set(zmq::sockopt::linger, 0);
    control.bind(o.control);
    zmq::socket_t sender(telemetry.context, zmq::socket_type::sub);
    sender.set(zmq::sockopt::subscribe, "");
    sender.set(zmq::sockopt::linger, 0);
    sender.set(zmq::sockopt::rcvhwm, 32);
    sender.connect(o.sender_telemetry);
    std::array<uint8_t, 65536> packet{}, expected{};
    SequenceStats sequence;
    uint64_t session = o.session, received_bytes = 0, compared_bits = 0, errors = 0;
    uint64_t invalid = 0, foreign = 0, all_packets = 0, last_bytes = 0;
    bool sender_seen = false;
    uint64_t sender_count = 0, loss_resets = 0, error_resets = 0;
    auto began = Clock::now(), last_publish = began;
    std::deque<std::pair<Clock::time_point, uint64_t>> pending;
    double rate = 0;
    auto report = [&](bool active) {
        std::ostringstream s;
        s << std::setprecision(12) << "{\"version\":1,\"role\":\"rx\",\"session\":\"" << hex_id(session)
          << "\",\"running\":" << (active ? "true" : "false")
          << ",\"received_packets\":" << all_packets << ",\"unique_packets\":" << sequence.unique
          << ",\"received_bytes\":" << received_bytes << ",\"duplicate_packets\":" << sequence.duplicates
          << ",\"reordered_packets\":" << sequence.reordered << ",\"late_packets\":" << sequence.late
          << ",\"too_old_packets\":" << sequence.too_old << ",\"invalid_packets\":" << invalid
          << ",\"foreign_packets\":" << foreign << ",\"confirmed_sent_packets\":" << sequence.confirmed_sent
          << ",\"loss_started\":" << (sequence.started() ? "true" : "false")
          << ",\"loss_expected_packets\":" << sequence.expected()
          << ",\"loss_resets\":" << loss_resets << ",\"error_resets\":" << error_resets
          << ",\"lost_packets\":" << sequence.lost() << ",\"loss_percent\":"
          << (sequence.expected() ? 100.0 * sequence.lost() / sequence.expected() : 0)
          << ",\"bit_errors\":" << errors << ",\"compared_bits\":" << compared_bits
          << ",\"ber\":" << (compared_bits ? double(errors) / compared_bits : 0)
          << ",\"rx_mbps\":" << rate << ",\"sender_seen\":" << (sender_seen ? "true" : "false")
          << ",\"elapsed_s\":" << seconds(Clock::now() - began) << '}';
        return s.str();
    };
    while (running) {
        auto now = Clock::now();
        if (o.duration_s > 0 && seconds(now - began) >= o.duration_s) break;
        pollfd item{udp.fd, POLLIN, 0};
        (void)poll(&item, 1, 5);
        for (int batch = 0; batch < 256; ++batch) {
            const ssize_t size = recv(udp.fd, packet.data(), packet.size(), MSG_DONTWAIT);
            if (size < 0) {
                if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                    throw std::runtime_error(std::string("UDP receive: ") + std::strerror(errno));
                break;
            }
            ++all_packets;
            Header header;
            if (!read_header(packet.data(), size_t(size), header)) {
                ++invalid;
                received_bytes += uint64_t(size);
                continue;
            }
            if (!session) session = header.session;
            if (header.session != session) { ++foreign; continue; }
            if (!sequence.started() && sender_seen)
                pending.emplace_back(Clock::now() + std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(o.grace_s)), sender_count);
            if (!sequence.record(header.sequence)) continue;
            make_packet(expected.data(), size_t(size), header.session, header.sequence);
            errors += bit_errors(packet.data(), expected.data(), size_t(size));
            compared_bits += uint64_t(size) * 8;
            received_bytes += uint64_t(size);
        }
        zmq::message_t message;
        while (sender.recv(message, zmq::recv_flags::dontwait)) {
            try {
                const auto data = YAML::Load(message.to_string());
                if (data["role"].as<std::string>() != "tx") continue;
                const auto incoming = std::stoull(data["session"].as<std::string>(), nullptr, 16);
                if (!session) session = incoming;
                if (session != incoming) continue;
                sender_seen = true;
                sender_count = std::max(sender_count, data["sent_packets"].as<uint64_t>());
                if (sequence.started())
                    pending.emplace_back(Clock::now() + std::chrono::duration_cast<Clock::duration>(
                        std::chrono::duration<double>(o.grace_s)), sender_count);
            } catch (const std::exception&) { /* Ignore unrelated telemetry. */ }
        }
        zmq::message_t request;
        if (control.recv(request, zmq::recv_flags::dontwait)) {
            const auto command = request.to_string();
            const bool valid = command == "RESET_LOSS" || command == "RESET_ERRORS";
            if (command == "RESET_LOSS") {
                sequence.reset_loss(sender_count);
                pending.clear();
                ++loss_resets;
            } else if (command == "RESET_ERRORS") {
                errors = compared_bits = 0;
                ++error_resets;
            }
            const std::string response = valid ? "{\"ok\":true}" : "{\"ok\":false}";
            (void)control.send(zmq::buffer(response), zmq::send_flags::none);
            if (valid) telemetry.publish(report(true));
        }
        now = Clock::now();
        while (!pending.empty() && pending.front().first <= now) {
            sequence.confirm(pending.front().second);
            pending.pop_front();
        }
        if (seconds(now - last_publish) >= .2) {
            // A missing sender monitor still permits in-stream gap accounting.
            if (!sender_seen && sequence.started()) pending.emplace_back(now + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(o.grace_s)), sequence.highest);
            rate = (received_bytes - last_bytes) * 8.0 / seconds(now - last_publish) / 1e6;
            telemetry.publish(report(true));
            last_publish = now; last_bytes = received_bytes;
        }
    }
    telemetry.publish(report(false));
    return 0;
}

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        return options.tx ? transmit(options) : receive(options);
    } catch (const zmq::error_t& e) {
        if (!running && e.num() == EINTR) return 0;
        std::cerr << "UdpLinkTest: " << e.what() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "UdpLinkTest: " << e.what() << '\n';
    }
    return 1;
}
