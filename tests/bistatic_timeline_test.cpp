#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <zmq.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "SensingChannel.hpp"

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int free_tcp_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    require(fd >= 0, "TCP socket failed");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t length = sizeof(address);
    const bool ok = bind(fd, reinterpret_cast<sockaddr*>(&address), length) == 0 &&
        getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) == 0;
    close(fd);
    require(ok, "TCP port allocation failed");
    return ntohs(address.sin_port);
}

struct Subscriber {
    void* context = zmq_ctx_new();
    void* socket = zmq_socket(context, ZMQ_SUB);

    explicit Subscriber(int port) {
        require(context && socket, "ZeroMQ subscriber failed");
        const int linger = 0;
        zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
        require(zmq_setsockopt(socket, ZMQ_SUBSCRIBE, "", 0) == 0, "SUBSCRIBE failed");
        const std::string endpoint = "tcp://127.0.0.1:" + std::to_string(port);
        require(zmq_connect(socket, endpoint.c_str()) == 0, "SUB connect failed");
    }
    ~Subscriber() {
        zmq_close(socket);
        zmq_ctx_term(context);
    }
    AlignedVector receive(int timeout_ms) {
        zmq_pollitem_t item{socket, 0, ZMQ_POLLIN, 0};
        require(zmq_poll(&item, 1, timeout_ms) >= 0, "SUB poll failed");
        if (!(item.revents & ZMQ_POLLIN)) return {};
        zmq_msg_t message;
        zmq_msg_init(&message);
        require(zmq_msg_recv(&message, socket, 0) >= 0, "SUB receive failed");
        const size_t bytes = zmq_msg_size(&message);
        require(bytes == 2 * 64 * sizeof(std::complex<float>), "Unexpected dense payload size");
        AlignedVector result(bytes / sizeof(std::complex<float>));
        std::memcpy(result.data(), zmq_msg_data(&message), bytes);
        zmq_msg_close(&message);
        return result;
    }
};

Config test_config() {
    Config cfg;
    cfg.ofdm.fft_size = 64;
    cfg.ofdm.cp_length = 8;
    cfg.ofdm.num_symbols = 10;
    cfg.ofdm.sensing_symbol_num = 2;
    cfg.sensing.range_fft_size = 64;
    cfg.sensing.doppler_fft_size = 2;
    cfg.sensing.backend_processing_enabled = false;
    return cfg;
}

SensingFrame frame(float amplitude) {
    SensingFrame result{};
    result.rx_symbols.assign(10, AlignedVector(64, {amplitude, 0.0f}));
    result.tx_symbols.assign(10, AlignedVector(64, {1.0f, 0.0f}));
    return result;
}

void check_amplitude(const AlignedVector& data, float expected) {
    require(!data.empty(), "Bistatic stream did not resume on the new timeline");
    for (const auto value : data) {
        require(std::abs(value - std::complex<float>(expected, 0.0f)) < 1e-4f,
                "Output mixed old and new timeline samples");
    }
}

void test_timeline(bool pending_update) {
    const Config cfg = test_config();
    std::atomic<bool> running{true};
    const int port = free_tcp_port();
    SensingChannel channel(cfg, {}, SensingChannel::SensingRole::Bistatic,
                          "127.0.0.1", port, 123456, running,
                          nullptr, nullptr, {}, {}, {});
    Subscriber subscriber(port);
    SharedSensingRuntime params;
    params.enable_mti = false;
    params.skip_sensing_fft = true;
    params.sensing_symbol_stride = 10;
    params.generation = 2;
    channel.apply_shared_cfg(params);
    channel.start_bistatic();

    // Wait for the real PUB/SUB subscription through actual processing/output.
    uint64_t old_start = 100000;
    AlignedVector output;
    for (int attempt = 0; attempt < 20 && output.empty(); ++attempt) {
        channel.process_bistatic_frame(frame(1.0f), old_start);
        old_start += 10;
        output = subscriber.receive(100);
    }
    check_amplitude(output, 1.0f);
    channel.process_bistatic_frame(frame(1.0f), old_start); // An old partial batch.
    require(subscriber.receive(50).empty(), "Partial old batch produced output");
    if (pending_update) {
        params.generation++;
        params.sensing_symbol_stride = 20;
        params.apply_symbol_index = old_start + 1000;
        channel.apply_shared_cfg(params);
    }

    channel.process_bistatic_frame(frame(3.0f), 0);
    require(subscriber.receive(50).empty(), "Old partial batch survived the restart");
    channel.process_bistatic_frame(frame(3.0f), 10);
    if (pending_update) {
        require(subscriber.receive(50).empty(), "Pending stride was not rebased to the new timeline");
        channel.process_bistatic_frame(frame(3.0f), 20);
    }
    check_amplitude(subscriber.receive(1000), 3.0f);
    channel.stop_bistatic();
}

void test_stride_larger_than_frame() {
    const Config cfg = test_config();
    std::atomic<bool> running{true};
    const int port = free_tcp_port();
    SensingChannel channel(cfg, {}, SensingChannel::SensingRole::Bistatic,
                          "127.0.0.1", port, 123457, running,
                          nullptr, nullptr, {}, {}, {});
    Subscriber subscriber(port);
    SharedSensingRuntime params;
    params.enable_mti = false;
    params.sensing_symbol_stride = 25;
    params.generation = 2;
    channel.apply_shared_cfg(params);
    channel.start_bistatic();
    AlignedVector output;
    uint64_t start = 100000;
    for (int attempt = 0; attempt < 30 && output.empty(); ++attempt) {
        channel.process_bistatic_frame(frame(1.0f), start);
        start += 10;
        output = subscriber.receive(100);
    }
    check_amplitude(output, 1.0f);
    channel.process_bistatic_frame(frame(3.0f), 0);
    channel.process_bistatic_frame(frame(9.0f), 10); // No scheduled sample here.
    require(subscriber.receive(50).empty(), "Forward cursor caused a false timeline reset");
    channel.process_bistatic_frame(frame(3.0f), 20);
    check_amplitude(subscriber.receive(1000), 3.0f);
    channel.stop_bistatic();
}

void test_mti_history() {
    SensingProcessor used({64, 64, 2, 2});
    SensingProcessor fresh({64, 64, 2, 2});
    const auto symbols = frame(3.0f).rx_symbols;
    used.copy_symbols_to_buffer(symbols, 2);
    used.apply_mti(true, 2);
    used.reset_history();
    used.copy_symbols_to_buffer(symbols, 2);
    fresh.copy_symbols_to_buffer(symbols, 2);
    used.apply_mti(true, 2);
    fresh.apply_mti(true, 2);
    for (size_t i = 0; i < used.channel_buffer().size(); ++i) {
        require(std::abs(used.channel_buffer()[i] - fresh.channel_buffer()[i]) < 1e-6f,
                "MTI retained history from the previous timeline");
    }
}
} // namespace

int main() {
    try {
        test_timeline(false);
        test_timeline(true);
        test_stride_larger_than_frame();
        test_mti_history();
        std::cout << "Bistatic timeline restart, pending controls, sparse stride and MTI checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
