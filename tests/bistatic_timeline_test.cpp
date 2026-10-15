#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <zmq.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <limits>
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

void test_timeline(bool pending_update, bool forward_gap = false) {
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

    const uint64_t restart = forward_gap ? old_start + 20 : 0;
    channel.process_bistatic_frame(frame(3.0f), restart);
    require(subscriber.receive(50).empty(), "Old partial batch survived a timeline gap/restart");
    channel.process_bistatic_frame(frame(3.0f), restart + 10);
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

void test_phase_reference() {
    constexpr size_t bins = 64, rows = 128, stride = 96;
    SensingPhaseReference reference(bins, rows);
    AlignedVector data(rows * stride), expected(rows * stride);
    for (size_t m = 0; m < rows; ++m) {
        const auto common = std::polar(1.0f, float(0.03 * m * m + 2.1 * (m % 7)));
        const auto moving = std::polar(0.2f, float(2 * M_PI * 7 * m / rows));
        for (size_t k = 0; k < bins; ++k) {
            expected[m * stride + k] = std::complex<float>(0.8f + 0.4f * m / rows, 0) +
                moving * std::polar(1.0f, float(-2 * M_PI * 12 * k / bins));
            data[m * stride + k] = expected[m * stride + k] * common;
        }
    }
    const auto before = data;
    float coherence;
    require(reference.apply(data.data(), rows, stride, 0.05f, coherence), "Valid LoS was rejected");
    require(coherence > 0.9f, "Unexpected reference coherence");
    std::complex<double> target_prev;
    for (size_t m = 0; m < rows; ++m) {
        std::complex<double> target;
        for (size_t k = 0; k < bins; ++k) {
            const auto i = m * stride + k;
            require(std::abs(data[i] - expected[i]) < 0.001f, "Common phase was not removed");
            require(std::abs(std::abs(data[i]) - std::abs(before[i])) < 2e-6f, "Amplitude changed");
            target += std::complex<double>(data[i]) * std::polar(1.0, 2 * M_PI * 12 * k / bins) / double(bins);
        }
        require(std::abs(std::abs(target) - 0.2) < 0.0001, "Moving echo amplitude changed");
        if (m) require(std::abs(std::arg(target * std::conj(target_prev)) - 2 * M_PI * 7 / rows) < 0.001,
                       "Moving target Doppler was removed or biased");
        target_prev = target;
        for (size_t k = bins; k < stride; ++k)
            require(data[m * stride + k] == std::complex<float>{}, "Padding was modified");
    }
    for (int mode = 0; mode < 4; ++mode) {
        data.assign(rows * stride, {1.0f, 0.0f});
        for (size_t k = 0; k < bins; ++k) {
            if (mode == 0) data[stride + k] = {};
            if (mode == 1) data[stride + k] = {(k % 2) ? 1.0f : -1.0f, 0.0f};
            if (mode == 2) data[stride + k] = {std::numeric_limits<float>::quiet_NaN(), 0};
            if (mode == 3) data[stride + k] = {std::numeric_limits<float>::infinity(), 0};
        }
        const auto original = data;
        require(!reference.apply(data.data(), rows, stride, 0.05f, coherence), "Invalid reference accepted");
        require(std::memcmp(data.data(), original.data(), data.size() * sizeof(data[0])) == 0,
                "Rejected batch was partially rotated");
    }

    // Display translations, including fractional/negative and FFT-period wrapping,
    // must move the output while preserving the same physical reference phase.
    for (float delay : {3.0f, -3.0f, 0.375f, -11.25f, 67.0f}) {
        data = before;
        for (size_t m = 0; m < rows; ++m) for (size_t k = 0; k < bins; ++k)
            data[m * stride + k] *= std::polar(1.0f, float(2 * M_PI * (double(k) - bins / 2) * delay / bins));
        require(reference.set_user_delay_offset(delay), "Valid display shift was rejected");
        require(reference.apply(data.data(), rows, stride, 0.05f, coherence), "Display shift lost the phase reference");
        for (size_t m = 0; m < rows; ++m) for (size_t k = 0; k < bins; ++k) {
            const auto shifted = expected[m * stride + k] * std::polar(1.0f, float(2 * M_PI * (double(k) - bins / 2) * delay / bins));
            require(std::abs(data[m * stride + k] - shifted) < 0.001f, "Display shift changed reference phase or target Doppler");
        }
    }
    std::cout << "Phase reference: arbitrary jumps, relative moving Doppler, amplitude and invalid data passed\n";
}

void test_phase_pipeline(bool enabled) {
    Config cfg = test_config();
    cfg.sensing.bistatic_phase_reference = enabled;
    std::atomic<bool> running{true};
    const int port = free_tcp_port();
    SensingChannel channel(cfg, {}, SensingChannel::SensingRole::Bistatic,
                          "127.0.0.1", port, 123458, running, nullptr, nullptr, {}, {}, {});
    Subscriber subscriber(port);
    SharedSensingRuntime params;
    params.enable_mti = false;
    params.sensing_symbol_stride = 10;
    params.generation = 2;
    channel.apply_shared_cfg(params);
    channel.start_bistatic();
    AlignedVector output;
    for (int attempt = 0; attempt < 20 && output.empty(); ++attempt) {
        auto input = frame(2.3f);
        for (auto& row : input.rx_symbols) for (auto& value : row) value *= std::complex<float>(0, 1);
        channel.process_bistatic_frame(input, attempt * 10);
        output = subscriber.receive(100);
    }
    require(!output.empty(), "Phase pipeline produced no output");
    for (const auto value : output)
        require(std::abs(value - (enabled ? std::complex<float>(2.3f, 0) : std::complex<float>(0, 2.3f))) < 1e-4f,
                "Phase correction placement/default-off behavior is wrong");

    if (enabled) {
        channel.process_bistatic_frame(frame(9.0f), 1000); // Partial old-coordinate batch.
        require(subscriber.receive(50).empty(), "Partial display batch produced output");
        auto shifted = frame(2.3f);
        shifted.user_delay_offset = -3.0f;
        shifted.delay_offset = -3.0f;
        for (auto& row : shifted.rx_symbols) for (auto& value : row) value *= std::complex<float>(0, 1);
        channel.process_bistatic_frame(shifted, 1010);
        require(subscriber.receive(50).empty(), "Display change mixed old and new coordinates");
        channel.process_bistatic_frame(shifted, 1020);
        output = subscriber.receive(1000);
        require(!output.empty(), "Display translation stopped sensing output");
        for (size_t m = 0; m < 2; ++m) for (size_t k = 0; k < 64; ++k) {
            const auto expected = std::polar(2.3f, float(-2 * M_PI * (double(k) - 32) * 3 / 64));
            require(std::abs(output[m * 64 + k] - expected) < 1e-4f, "Display translation/reference sign mismatch");
        }
    }
    channel.stop_bistatic();
}

void test_phase_config() {
    char path[] = "/tmp/openisac-phase-config-XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "Temporary config failed");
    close(fd);
    struct Cleanup { const char* path; ~Cleanup() { unlink(path); } } cleanup{path};
    for (int mode = 0; mode < 7; ++mode) {
        std::ofstream file(path);
        file << "sensing:\n  bi_enabled: " << (mode == 3 ? "false" : "true")
             << "\n  bistatic_phase_reference: true\n  phase_reference_min_coherence: "
             << (mode == 1 ? "0" : mode == 2 ? "1.1" : mode == 6 ? ".nan" : "0.1")
             << "\n  output_mode: " << (mode == 4 ? "compact_mask" : "dense")
             << "\n  sensing_delay_correction_mode: " << (mode == 5 ? "ertm_absolute" : "los_tracking") << '\n';
        file.close();
        Config cfg = make_default_ue_config();
        const bool loaded = load_ue_config_from_yaml(cfg, path);
        if (loaded != (mode == 0)) throw std::runtime_error("Phase config validation mismatch mode=" + std::to_string(mode));
        if (loaded) require(cfg.sensing.bistatic_phase_reference && cfg.sensing.phase_reference_min_coherence == 0.1f,
                            "Phase config fields were not parsed");
    }
}

} // namespace

int main() {
    try {
        test_timeline(false);
        test_timeline(true);
        test_timeline(false, true);
        test_phase_reference();
        test_phase_pipeline(false);
        test_phase_pipeline(true);
        test_phase_config();
        test_stride_larger_than_frame();
        test_mti_history();
        std::cout << "Bistatic timeline restart, pending controls, sparse stride and MTI checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
