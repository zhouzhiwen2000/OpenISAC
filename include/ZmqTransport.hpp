#pragma once

/**
 * @file ZmqTransport.hpp
 * @brief ZeroMQ transport plumbing for Backend <-> Frontend communication.
 *
 * This header replaces the previous raw-UDP transport. It is pure plumbing
 * (no DSP), sitting alongside Common.hpp's IO vocabulary.
 *
 * Two roles are provided:
 *  - SharedPubSocket / bind_pub(): XPUB sockets for one-way data streams
 *    (sensing frames + debug plots). Backend binds, frontend SUB connects.
 *    Sends are non-blocking and XPUB_NODROP turns high-water-mark overflow
 *    into EAGAIN, allowing the caller to warn instead of losing data silently.
 *  - ControlRouter: a ROUTER socket for the bidirectional control/params
 *    channel. Backend binds ROUTER, frontend connects DEALER.
 *
 * Important ZeroMQ constraints honoured here:
 *  - A ZMQ endpoint can only be bound once. Unlike UDP `sendto`, several
 *    logical senders cannot independently target the same port. bind_pub()
 *    therefore returns a process-wide shared, mutex-guarded PUB socket keyed
 *    by endpoint, so multiple SensingDataSenders to the same port share one
 *    bound XPUB socket.
 *  - ZMQ sockets are NOT thread-safe. ControlRouter owns its socket on the
 *    single poll thread and accepts outbound messages via a thread-safe
 *    queue (post_send), which the poll loop drains. Never call the socket
 *    directly from another thread.
 */

#include <zmq.hpp>

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace zmq_transport {

/// Process-wide ZeroMQ context (single IO thread is plenty for our streams).
inline zmq::context_t& global_context() {
    static zmq::context_t ctx{1};
    return ctx;
}

/// Build a TCP endpoint string. An empty / 0.0.0.0 IP maps to the wildcard
/// interface "*" for bind side. The frontend connects to the same IP:port.
inline std::string make_tcp_endpoint(const std::string& ip, int port) {
    std::string host = ip;
    if (host.empty() || host == "0.0.0.0") {
        host = "*";
    }
    return "tcp://" + host + ":" + std::to_string(port);
}

/// A view into a contiguous byte range to send as one ZMQ message part.
struct MsgPart {
    const void* data = nullptr;
    size_t size = 0;
};

/**
 * @brief Shared XPUB socket bound to a single endpoint.
 *
 * Multipart sends are atomic (a multipart message is only delivered to
 * subscribers once complete) and never block. High-water-mark overflow is
 * reported to the caller as a failed send instead of being dropped silently.
 */
class SharedPubSocket {
public:
    explicit SharedPubSocket(const std::string& endpoint)
        : _socket(global_context(), zmq::socket_type::xpub) {
        _socket.set(zmq::sockopt::sndhwm, 8);  // small: keep only recent frames
        _socket.set(zmq::sockopt::xpub_nodrop, 1);
        _socket.set(zmq::sockopt::linger, 0);
        try {
            _socket.bind(endpoint);
        } catch (const zmq::error_t& e) {
            throw std::runtime_error(
                "ZeroMQ PUB bind failed on " + endpoint + ": " + e.what() +
                " (is another backend instance already using this port?)");
        }
    }

    /// Send a multipart message. `parts` with a single entry sends one frame.
    /// Empty parts are allowed (e.g. a zero-length data part followed by
    /// metadata). Thread-safe; never blocks the caller.
    bool send_multipart(const std::vector<MsgPart>& parts) {
        if (parts.empty()) {
            return true;
        }
        std::lock_guard<std::mutex> lock(_mutex);
        _drain_subscription_events();
        try {
            const size_t n = parts.size();
            for (size_t i = 0; i < n; ++i) {
                zmq::message_t msg(parts[i].data, parts[i].size);
                const zmq::send_flags flags =
                    ((i + 1 < n) ? zmq::send_flags::sndmore : zmq::send_flags::none) |
                    zmq::send_flags::dontwait;
                if (!_socket.send(msg, flags)) {
                    return false;
                }
            }
        } catch (const zmq::error_t&) {
            return false;
        }
        return true;
    }

    bool send_single(const void* data, size_t size) {
        return send_multipart({MsgPart{data, size}});
    }

private:
    void _drain_subscription_events() {
        while (true) {
            zmq::message_t event;
            try {
                if (!_socket.recv(event, zmq::recv_flags::dontwait)) {
                    return;
                }
            } catch (const zmq::error_t& e) {
                if (e.num() == EAGAIN) {
                    return;
                }
                return;
            }
        }
    }

    zmq::socket_t _socket;
    std::mutex _mutex;
};

/**
 * @brief Return a process-wide shared XPUB socket bound to `endpoint`.
 *
 * The socket is reference-counted: it is created on first request and
 * destroyed once the last holder releases it, allowing a later rebind after
 * reconfiguration without an "address already in use" error.
 */
inline std::shared_ptr<SharedPubSocket> bind_pub(const std::string& endpoint) {
    static std::mutex reg_mutex;
    static std::unordered_map<std::string, std::weak_ptr<SharedPubSocket>> registry;
    std::lock_guard<std::mutex> lock(reg_mutex);
    auto it = registry.find(endpoint);
    if (it != registry.end()) {
        if (auto existing = it->second.lock()) {
            return existing;
        }
    }
    auto sock = std::make_shared<SharedPubSocket>(endpoint);
    registry[endpoint] = sock;
    return sock;
}

/// Opaque handle identifying a connected control peer (the DEALER routing id).
using PeerId = std::string;

/**
 * @brief ROUTER socket for the bidirectional control/params channel.
 *
 * The socket is owned exclusively by the polling thread. Outbound replies
 * (params, heartbeat) are posted from arbitrary threads via post_send() and
 * flushed by the poll loop, so no socket call ever crosses threads.
 */
class ControlRouter {
public:
    explicit ControlRouter(const std::string& endpoint, size_t max_outq = 1024)
        : _socket(global_context(), zmq::socket_type::router)
        , _max_outq(max_outq) {
        _socket.set(zmq::sockopt::linger, 0);
        _socket.set(zmq::sockopt::rcvtimeo, 5);  // ms; bounds poll latency
        // Report unknown/disconnected peers as EHOSTUNREACH so the owner can
        // emit a warning instead of silently losing the control reply.
        _socket.set(zmq::sockopt::router_mandatory, 1);
        try {
            _socket.bind(endpoint);
        } catch (const zmq::error_t& e) {
            throw std::runtime_error(
                "ZeroMQ ROUTER bind failed on " + endpoint + ": " + e.what() +
                " (is another backend instance already using this control port?)");
        }
    }

    /// Receive one request: [identity][payload]. Returns false on timeout.
    /// Must be called only from the owning poll thread.
    bool recv(PeerId& identity, std::vector<uint8_t>& payload) {
        zmq::message_t id_msg;
        auto r = _socket.recv(id_msg, zmq::recv_flags::none);
        if (!r) {
            return false;  // RCVTIMEO elapsed
        }
        identity.assign(static_cast<const char*>(id_msg.data()), id_msg.size());

        zmq::message_t payload_msg;
        auto r2 = _socket.recv(payload_msg, zmq::recv_flags::none);
        if (!r2) {
            return false;
        }
        const auto* p = static_cast<const uint8_t*>(payload_msg.data());
        payload.assign(p, p + payload_msg.size());
        return true;
    }

    /// Queue a reply to a specific peer. Thread-safe.
    bool post_send(const PeerId& identity, const void* data, size_t size) {
        if (identity.empty()) {
            return false;
        }
        const auto* p = static_cast<const uint8_t*>(data);
        std::vector<uint8_t> payload;
        if (p != nullptr && size > 0) {
            payload.assign(p, p + size);
        }
        bool evicted = false;
        std::lock_guard<std::mutex> lock(_out_mutex);
        while (_max_outq > 0 && _outq.size() >= _max_outq) {
            _outq.pop_front();
            evicted = true;
        }
        _outq.emplace_back(identity, std::move(payload));
        return !evicted;
    }

    /// Flush queued replies. Must be called only from the owning poll thread.
    size_t flush() {
        std::deque<std::pair<PeerId, std::vector<uint8_t>>> pending;
        {
            std::lock_guard<std::mutex> lock(_out_mutex);
            pending.swap(_outq);
        }
        size_t failed = 0;
        for (auto& item : pending) {
            if (!_send_reply_multipart(item.first, item.second)) {
                ++failed;
            }
        }
        return failed;
    }

private:
    bool _send_reply_multipart(const PeerId& identity, const std::vector<uint8_t>& payload) {
        const std::vector<MsgPart> parts{
            MsgPart{identity.data(), identity.size()},
            MsgPart{payload.data(), payload.size()},
        };
        return _send_multipart_blocking(parts);
    }

    bool _send_multipart_blocking(const std::vector<MsgPart>& parts) {
        if (parts.empty()) {
            return true;
        }
        try {
            for (size_t i = 0; i < parts.size(); ++i) {
                zmq::message_t msg(parts[i].data, parts[i].size);
                const zmq::send_flags flags =
                    (i + 1 < parts.size()) ? zmq::send_flags::sndmore : zmq::send_flags::none;
                if (!_socket.send(msg, flags)) {
                    return false;
                }
            }
        } catch (const zmq::error_t&) {
            return false;
        }
        return true;
    }

    zmq::socket_t _socket;
    std::mutex _out_mutex;
    std::deque<std::pair<PeerId, std::vector<uint8_t>>> _outq;
    size_t _max_outq = 1024;
};

}  // namespace zmq_transport
