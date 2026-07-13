from __future__ import annotations

from matplotlib.widgets import Button, TextBox

from sensing_runtime_protocol import make_debug_sub_conflate, make_tcp_endpoint


class DebugZmqConnector:
    """Small Matplotlib control strip for debug PUB/SUB viewers."""

    def __init__(self, fig, port: int, initial_host: str = "127.0.0.1") -> None:
        self.fig = fig
        self.port = int(port)
        self.host = self._normalize_host(initial_host)
        self.sock = None

        self.host_box = TextBox(
            fig.add_axes([0.12, 0.025, 0.46, 0.035]),
            "Host",
            initial=self.host,
        )
        self.connect_button = Button(
            fig.add_axes([0.62, 0.025, 0.16, 0.035]),
            "Connect",
        )
        self.status_text = fig.text(0.80, 0.035, "", va="center", fontsize=9)

        self.host_box.on_submit(self.connect)
        self.connect_button.on_clicked(lambda _event: self.connect())
        self.connect(self.host)

    @staticmethod
    def _normalize_host(host: str) -> str:
        return (host or "").strip() or "127.0.0.1"

    def connect(self, host: str | None = None) -> None:
        next_host = self._normalize_host(host if host is not None else self.host_box.text)
        next_sock = make_debug_sub_conflate(make_tcp_endpoint(next_host, self.port))
        old_sock = self.sock
        self.sock = next_sock
        self.host = next_host
        if old_sock is not None:
            old_sock.close(linger=0)
        self.status_text.set_text(f"Connected {self.host}:{self.port}")
        self.fig.canvas.draw_idle()

    def recv(self, flags: int = 0):
        return self.sock.recv(flags=flags)
