#ifndef XR1710G_PLL_CONTROL_HPP
#define XR1710G_PLL_CONTROL_HPP

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <locale>
#include <mutex>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char** environ;

namespace radio::xr1710g {

/** @brief Synchronous, ordered PLL control on a persistent helper connection.
 * No receive-path work or sample flushing. The trusted configured command is
 * started once; numeric requests travel over its stdin, never through a shell.
 */
class PllControl {
public:
    PllControl(const std::string& command, double center, double rate) {
        int fd[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fd))
            throw std::runtime_error("XR1710G PLL socketpair failed");
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, fd[1], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, fd[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions, fd[0]);
        posix_spawn_file_actions_addclose(&actions, fd[1]);
        std::string exec = "exec " + command;
        char* argv[] = {const_cast<char*>("sh"), const_cast<char*>("-c"), exec.data(), nullptr};
        const int error = posix_spawn(&_pid, "/bin/sh", &actions, nullptr, argv, environ);
        posix_spawn_file_actions_destroy(&actions);
        close(fd[1]);
        if (error) { close(fd[0]); throw std::runtime_error("XR1710G PLL spawn: " + std::string(std::strerror(error))); }
        _fd = fd[0];
        try { transact("OPEN", center, rate); }
        catch (...) { shutdown_child(); throw; }
    }
    ~PllControl() { shutdown_child(); }
    PllControl(const PllControl&) = delete;
    PllControl& operator=(const PllControl&) = delete;

    /** @brief Set total positive-LO correction; return applied quantized Hz. */
    double set(double hz) {
        std::lock_guard<std::mutex> guard(_mutex);
        if (!std::isfinite(hz) || std::abs(hz) > 100000.0)
            throw std::invalid_argument("XR1710G PLL correction must be within +/-100000 Hz");
        if (_failed) throw std::runtime_error("XR1710G PLL control failed; reopen the backend");
        try { return transact("SET", hz); }
        catch (...) { _failed = true; shutdown(_fd, SHUT_WR); throw; }
    }

private:
    using Clock = std::chrono::steady_clock;
    void ready(short events, Clock::time_point deadline) {
        for (;;) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            if (left <= 0) throw std::runtime_error("XR1710G PLL control timed out");
            pollfd p{_fd, events, 0};
            const int result = poll(&p, 1, static_cast<int>(left));
            if (result < 0 && errno == EINTR) continue;
            if (result > 0 && (p.revents & events)) return;
            throw std::runtime_error("XR1710G PLL helper disconnected or timed out");
        }
    }
    double transact(const char* op, double a, double b = 0) {
        const uint64_t id = ++_sequence;
        std::ostringstream request;
        request.imbue(std::locale::classic());
        request << op << ' ' << id << ' ' << std::setprecision(17) << a;
        if (std::strcmp(op, "OPEN") == 0) request << ' ' << b;
        request << '\n';
        const std::string data = request.str();
        const auto deadline = Clock::now() + std::chrono::seconds(3);
        for (size_t offset = 0; offset < data.size();) {
            ready(POLLOUT, deadline);
            const ssize_t n = send(_fd, data.data()+offset, data.size()-offset, MSG_NOSIGNAL | MSG_DONTWAIT);
            if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            if (n <= 0) throw std::runtime_error("XR1710G PLL request failed");
            offset += static_cast<size_t>(n);
        }
        std::string line;
        bool complete = false;
        while (line.size() < 1024) {
            ready(POLLIN, deadline);
            char c;
            const ssize_t n = recv(_fd, &c, 1, MSG_DONTWAIT);
            if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            if (n <= 0) throw std::runtime_error("XR1710G PLL helper closed its reply");
            if (c == '\n') { complete = true; break; }
            line += c;
        }
        if (!complete) throw std::runtime_error("XR1710G PLL reply exceeds line limit");
        std::istringstream reply(line); reply.imbue(std::locale::classic());
        std::string status, extra;
        uint64_t reply_id = 0;
        double applied = 0, step = 0;
        if (!(reply >> status >> reply_id) || reply_id != id)
            throw std::runtime_error("XR1710G PLL malformed or stale reply");
        if (status == "ERR") throw std::runtime_error("XR1710G PLL: " + line);
        if (status != "OK" || !(reply >> applied >> step) || (reply >> extra) ||
            !std::isfinite(applied) || !std::isfinite(step) || step <= 0 || step > 15 ||
            std::abs(applied) > 100008 || std::abs(applied - (std::strcmp(op, "OPEN") ? a : 0)) > step*.5001)
            throw std::runtime_error("XR1710G PLL invalid applied-frequency reply");
        if (std::strcmp(op, "OPEN") == 0 && applied != 0)
            throw std::runtime_error("XR1710G PLL initial correction must be zero");
        return applied;
    }
    void shutdown_child() noexcept {
        if (_fd >= 0) { shutdown(_fd, SHUT_WR); close(_fd); _fd = -1; }
        if (_pid <= 0) return;
        for (int i = 0; i < 400; ++i) {
            const auto result = waitpid(_pid, nullptr, WNOHANG);
            if (result == _pid || (result < 0 && errno == ECHILD)) { _pid = -1; return; }
            if (i == 200) kill(_pid, SIGTERM);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        kill(_pid, SIGKILL);
        while (waitpid(_pid, nullptr, 0) < 0 && errno == EINTR) {}
        _pid = -1;
    }
    int _fd = -1;
    pid_t _pid = -1;
    uint64_t _sequence = 0;
    bool _failed = false;
    std::mutex _mutex;
};
} // namespace radio::xr1710g
#endif
