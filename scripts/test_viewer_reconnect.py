"""Exercise the fast viewer's actual control loop across a ZeroMQ server restart."""
import ast
import logging
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from types import SimpleNamespace
import unittest

import zmq
from zmq.utils.monitor import recv_monitor_message

from sensing_runtime_protocol import (
    CTRL_HEADER, PARAMS_COMMAND, READY_COMMAND, PARAMS_PACKET_STRUCT,
    ViewerRuntimeParams, build_control_command, build_params_request,
    make_control_dealer, parse_params_packet,
)


def load_control_functions(namespace):
    # Importing fast_viewer creates the Qt application. Execute its unchanged
    # control functions so this transport regression can run without a display.
    path = Path(__file__).parent / "sensing_viewer/fast_viewer.py"
    names = {"_clear_queue", "_clear_control_tx_queue", "queue_startup_backend_sync",
             "_reset_backend_viewer_state", "_invalidate_backend_session",
             "_flush_control_tx_queue", "control_receiver", "send_startup_backend_sync"}
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)


class ViewerReconnectTest(unittest.TestCase):
    def setUp(self):
        self.router = zmq.Context.instance().socket(zmq.ROUTER)
        port = self.router.bind_to_random_port("tcp://127.0.0.1")
        self.endpoint = f"tcp://127.0.0.1:{port}"
        self.channel = SimpleNamespace(
            viewer_params=ViewerRuntimeParams(version=0), last_frame_id=123,
            current_display_data={"old": 1}, last_param_error=None,
            display_queue=Queue(), range_time_lock=threading.Lock(), range_time_data=1,
            micro_lock=threading.Lock(), micro_doppler_buffer=[1])
        self.ns = dict(zmq=zmq, recv_monitor_message=recv_monitor_message,
                       threading=threading, time=time, Full=Full, Empty=Empty,
                       LOGGER=logging.getLogger(__name__), CHANNELS=[self.channel],
                       CONTROL_ENDPOINT=self.endpoint,
                       control_socket=make_control_dealer(self.endpoint),
                       control_socket_lock=threading.Lock(), running=True,
                       connection_generation=0, backend_session_generation=0,
                       startup_sync_queued_generation=None, startup_sync_lock=threading.Lock(),
                       startup_sync_queue=Queue(8), control_tx_queue=Queue(256),
                       aggregate_frame_queue=Queue(), radio_params_queue=Queue(),
                       PENDING_VIEWER_PARAMS=ViewerRuntimeParams(version=0),
                       CTRL_HEADER=CTRL_HEADER, PARAMS_COMMAND=PARAMS_COMMAND,
                       READY_COMMAND=READY_COMMAND, parse_params_packet=parse_params_packet,
                       IS_BI_MODE=True, _warn_drop=lambda *args: None)
        self.ns["get_viewer_params"] = lambda ch: ch.viewer_params
        self.ns["set_viewer_params"] = lambda ch, params: setattr(ch, "viewer_params", params)
        self.ns["_handle_viewer_params"] = lambda channels, params, prefix: [
            setattr(ch, "viewer_params", params) for ch in channels]
        self.ns["request_viewer_params"] = lambda value=0: self.enqueue(build_params_request(value or 0))
        for name, command in [("send_skip_command", b"SKIP"), ("send_mti_command", b"MTI "),
                              ("send_strd_command", b"STRD")]:
            self.ns[name] = lambda value=1, command=command: self.enqueue(build_control_command(command, int(value)))
        load_control_functions(self.ns)
        self.controls = []
        self.last_peer = None
        self.thread = None

    def enqueue(self, packet):
        self.ns["control_tx_queue"].put_nowait((packet, self.endpoint, "test"))

    def start_receiver(self):
        self.thread = threading.Thread(target=self.ns["control_receiver"], daemon=True)
        self.thread.start()
        self.ns["request_viewer_params"]()

    def pump_until(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.router is not None and self.router.poll(10):
                parts = self.router.recv_multipart()
                self.last_peer, payload = parts[0], parts[-1]
                if payload[:4] == b"REQ ":
                    packet = PARAMS_PACKET_STRUCT.pack(
                        b"CTRL", b"PARM", 10, 16, 0, 100, 1024, 100, 1024,
                        100, 1024, 100, 0, 0, 1, 1, 7500, 2, 2,
                        31000000, 1000000, 0, 1024, 128, 100, 1, 1)
                    self.router.send_multipart([self.last_peer, packet])
                elif payload[:4] == b"CMD ":
                    self.controls.append((payload[4:8], int.from_bytes(payload[8:12], "big", signed=True)))
            else:
                time.sleep(.005)
            if predicate():
                return
        self.fail("control loop did not reach expected state")

    def apply_saved_controls(self):
        token = self.ns["startup_sync_queue"].get_nowait()
        self.assertEqual(token, (self.ns["connection_generation"], self.ns["backend_session_generation"]))
        self.ns["send_startup_backend_sync"]("100", False, "20", "30")

    def test_restart_replays_saved_controls_once_and_refreshes_params(self):
        self.start_receiver()
        self.pump_until(lambda: not self.ns["startup_sync_queue"].empty())
        self.apply_saved_controls()
        self.pump_until(lambda: len(self.controls) == 3)
        self.assertEqual(self.controls, [(b"SKIP", 1), (b"MTI ", 0), (b"STRD", 100)])
        self.router.close(0)
        self.router = None
        self.pump_until(lambda: self.ns["backend_session_generation"] == 1)
        self.assertEqual(self.channel.viewer_params.version, 0)
        self.assertIsNone(self.channel.last_frame_id)
        self.assertIsNone(self.channel.range_time_data)
        self.assertEqual(self.channel.micro_doppler_buffer, [])
        self.router = zmq.Context.instance().socket(zmq.ROUTER)
        self.router.bind(self.endpoint)
        self.controls = []
        self.pump_until(lambda: not self.ns["startup_sync_queue"].empty())
        self.apply_saved_controls()
        self.pump_until(lambda: len(self.controls) == 3)
        self.assertEqual(self.channel.viewer_params.version, 10)
        self.assertEqual(self.controls, [(b"SKIP", 1), (b"MTI ", 0), (b"STRD", 100)])
        for _ in range(5):
            self.router.send_multipart([self.last_peer, b"CTRL" + READY_COMMAND])
            time.sleep(.025)
        self.assertTrue(self.ns["startup_sync_queue"].empty())

    def test_full_startup_queue_can_retry_and_disconnect_drops_old_controls(self):
        for _ in range(8):
            self.ns["startup_sync_queue"].put_nowait("stale")
        self.ns["queue_startup_backend_sync"]()
        self.assertIsNone(self.ns["startup_sync_queued_generation"])
        self.ns["startup_sync_queue"].get_nowait()
        self.ns["queue_startup_backend_sync"]()
        self.assertEqual(self.ns["startup_sync_queued_generation"], (0, 0))
        self.enqueue(build_control_command(b"MTI ", 1))
        self.ns["_invalidate_backend_session"]()
        self.assertTrue(self.ns["control_tx_queue"].empty())
        self.assertTrue(self.ns["startup_sync_queue"].empty())
        self.ns["queue_startup_backend_sync"]()
        self.assertEqual(self.ns["startup_sync_queue"].get_nowait(), (0, 1))

    def tearDown(self):
        self.ns["running"] = False
        if self.thread is not None:
            self.thread.join(2)
            self.assertFalse(self.thread.is_alive())
        else:
            self.ns["control_socket"].close(0)
        if self.router is not None:
            self.router.close(0)


if __name__ == "__main__":
    unittest.main()
