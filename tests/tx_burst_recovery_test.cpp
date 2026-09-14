#include "TxBurstRecovery.hpp"
#include "TimedStreamRestart.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_contains(
    const std::string& value,
    const std::string& expected,
    const char* message)
{
    require(value.find(expected) != std::string::npos, message);
}

} // namespace

int main()
{
    try {
        constexpr int64_t kCooldownNs = 200'000'000;
        TxAsyncRestartGate gate(kCooldownNs);

        require(gate.request_at(TxRestartReason::Underflow, 1'000'000'000),
                "first underflow was not accepted");
        require(!gate.request_at(TxRestartReason::TimeError, 1'500'000'000),
                "pending restart was not coalesced");
        require(gate.consume() == TxRestartReason::Underflow,
                "pending underflow reason was lost");
        require(gate.consume_coalesced_count() == 1,
                "pending event coalescing was not counted");

        gate.set_recovery_target_time_ns(3'000'000'000);
        require(!gate.request_at(TxRestartReason::TimeError, 1'199'999'999),
                "event during recovery was accepted");
        require(!gate.request_event_at(
                    TxRestartReason::Underflow,
                    1'500'000'000,
                    2'999'999'999),
                "old-burst event during recovery was accepted");
        require(!gate.request_event_at(
                    TxRestartReason::TimeError,
                    1'600'000'000,
                    3'000'000'000),
                "new-anchor event bypassed the active recovery");
        gate.finish_recovery(3'000'000'000);
        require(gate.consume() == TxRestartReason::TimeError,
                "new-anchor error was not deferred to the next recovery");
        gate.set_recovery_target_time_ns(4'000'000'000);
        gate.finish_recovery(4'000'000'000);
        require(!gate.request_event_at(
                    TxRestartReason::Underflow,
                    2'000'000'000,
                    3'500'000'000),
                "stale event before the accepted restart floor was accepted");
        require(gate.consume_coalesced_count() == 4,
                "recovery event coalescing count was incorrect");

        gate.reset();
        require(gate.consume() == TxRestartReason::None,
                "reset left a pending restart");
        require(gate.request_at(TxRestartReason::Underflow, 0),
                "reset did not clear cooldown state");

        require(serialized_tx_restart_target_ticks(200, 100, 250) == 350,
                "restart target did not stay one lead interval after prior burst");
        require(serialized_tx_restart_target_ticks(500, 100, 250) == 600,
                "restart target did not honor current device time");

        TimedStreamRestartAck restart_ack;
        const uint64_t armed_request = restart_ack.begin_request();
        require(restart_ack.status(armed_request) == TimedStreamRestartStatus::Pending,
                "new timed restart request was not pending");
        restart_ack.mark_armed(armed_request);
        require(restart_ack.status(armed_request) == TimedStreamRestartStatus::Armed,
                "timed restart arm was not published");
        const uint64_t failed_request = restart_ack.begin_request();
        restart_ack.mark_failed(failed_request);
        require(restart_ack.status(failed_request) == TimedStreamRestartStatus::Failed,
                "timed restart failure was not published");

        TxHotPathProfiler profiler(/*frame_duration_ns=*/1'000, /*start_ns=*/10'000);
        profiler.observe_queue(4);
        profiler.note_gate_drop();
        const auto blank_send = profiler.begin_send_at(11'000);
        profiler.complete_send_at(
            blank_send,
            /*modulated=*/false,
            /*held_future=*/false,
            /*requested_samples=*/100,
            /*sent_samples=*/90,
            /*air_seq=*/7,
            /*end_ns=*/13'000);
        profiler.observe_queue(2);
        const auto data_send = profiler.begin_send_at(13'500);
        profiler.complete_send_at(
            data_send,
            /*modulated=*/true,
            /*held_future=*/false,
            /*requested_samples=*/100,
            /*sent_samples=*/100,
            /*air_seq=*/8,
            /*end_ns=*/14'500);
        const std::string profile = profiler.report_and_reset_at(
            "test", /*current_queue_depth=*/3, /*next_air_seq=*/9, /*now_ns=*/15'000);
        require_contains(profile, "trigger=test", "profile trigger was omitted");
        require_contains(profile, "frames=2", "profile frame count was incorrect");
        require_contains(profile, "data=1", "profile data frame count was incorrect");
        require_contains(profile, "blank_empty=1", "profile blank count was incorrect");
        require_contains(profile, "gate_drops=1", "profile gate drop count was incorrect");
        require_contains(profile, "short_calls=1", "profile short send count was incorrect");
        require_contains(profile, "short_samples=10", "profile short sample count was incorrect");
        require_contains(profile, "@7:blank-empty(90/100)",
                         "profile slowest-send context was incorrect");
        require_contains(profile, "over_budget=1", "profile deadline miss count was incorrect");
        require_contains(profile, "queue(min/avg/max/current)=2/3.00/4/3",
                         "profile queue watermarks were incorrect");

        std::cout << "TX burst recovery test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
