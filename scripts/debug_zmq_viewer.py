from __future__ import annotations

from matplotlib.widgets import Button, TextBox

from sensing_runtime_protocol import make_debug_sub_conflate, make_tcp_endpoint


_CONTROL_Y = 0.024
_CONTROL_H = 0.038
_STATUS_Y = 0.070
_HOST_AX = [0.12, _CONTROL_Y, 0.58, _CONTROL_H]
_BUTTON_AX = [0.73, _CONTROL_Y, 0.17, _CONTROL_H]
_STATUS_AX = [0.12, _STATUS_Y, 0.78, 0.028]


class DebugZmqConnector:
    """Small Matplotlib control strip for debug PUB/SUB viewers."""

    def __init__(self, fig, port: int, initial_host: str = "127.0.0.1") -> None:
        self.fig = fig
        self.port = int(port)
        self.host = self._normalize_host(initial_host)
        self.sock = None

        self.host_box = TextBox(
            fig.add_axes(_HOST_AX),
            "Host",
            initial=self.host,
            color="#f8fafc",
            hovercolor="#eef2ff",
            label_pad=0.015,
        )
        self.connect_button = Button(
            fig.add_axes(_BUTTON_AX),
            "Connect",
            color="#e8eef8",
            hovercolor="#dbeafe",
        )
        self.status_ax = fig.add_axes(_STATUS_AX)
        self.status_ax.set_axis_off()
        self.status_text = self.status_ax.text(
            0.0,
            0.5,
            "",
            va="center",
            ha="left",
            fontsize=9,
            clip_on=True,
            transform=self.status_ax.transAxes,
        )

        self.host_box.on_submit(self.connect)
        self.connect_button.on_clicked(lambda _event: self.connect())
        self.connect(self.host)

    @staticmethod
    def _normalize_host(host: str) -> str:
        return (host or "").strip() or "127.0.0.1"

    @staticmethod
    def _compact_endpoint(host: str, port: int) -> str:
        endpoint = f"{host}:{port}"
        if len(endpoint) <= 34:
            return endpoint
        return f"...{endpoint[-31:]}"

    def connect(self, host: str | None = None) -> None:
        next_host = self._normalize_host(host if host is not None else self.host_box.text)
        next_sock = make_debug_sub_conflate(make_tcp_endpoint(next_host, self.port))
        old_sock = self.sock
        self.sock = next_sock
        self.host = next_host
        if old_sock is not None:
            old_sock.close(linger=0)
        endpoint = self._compact_endpoint(self.host, self.port)
        self.status_text.set_text(f"Connected {endpoint}")
        self.fig.canvas.draw_idle()

    def recv(self, flags: int = 0):
        return self.sock.recv(flags=flags)
