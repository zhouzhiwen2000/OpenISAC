from __future__ import annotations

from matplotlib.widgets import Button, TextBox

from sensing_runtime_protocol import make_debug_sub_conflate, make_tcp_endpoint
from viewer_endpoint_store import default_settings_key, load_endpoint, save_endpoint


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
        self.default_port = int(port)
        self.settings_key = default_settings_key()
        saved_host, saved_port = load_endpoint(self.settings_key, initial_host, self.default_port)
        self.host, self.port = self._parse_endpoint(
            self._format_endpoint(saved_host, saved_port),
            self.default_port,
        )
        self.sock = None

        self.host_box = TextBox(
            fig.add_axes(_HOST_AX),
            "Endpoint",
            initial=self._format_endpoint(self.host, self.port),
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
        self.connect(self._format_endpoint(self.host, self.port))

    @staticmethod
    def _normalize_host(host: str) -> str:
        return (host or "").strip() or "127.0.0.1"

    @staticmethod
    def _parse_port(port_text: str) -> int:
        try:
            port = int(port_text.strip())
        except ValueError as exc:
            raise ValueError("Port must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("Port must be in 1..65535")
        return port

    @classmethod
    def _parse_endpoint(cls, endpoint: str, default_port: int) -> tuple[str, int]:
        value = (endpoint or "").strip()
        if value.startswith("tcp://"):
            value = value[len("tcp://"):]
        if not value:
            return "127.0.0.1", int(default_port)

        host = value
        port = int(default_port)
        if value.startswith("["):
            end = value.find("]")
            if end >= 0:
                host = value[1:end]
                rest = value[end + 1:].strip()
                if rest.startswith(":") and rest[1:]:
                    port = cls._parse_port(rest[1:])
                return cls._normalize_host(host), port

        if value.count(":") == 1:
            host_part, port_part = value.rsplit(":", 1)
            if port_part.strip():
                host = host_part
                port = cls._parse_port(port_part)

        return cls._normalize_host(host), port

    @staticmethod
    def _format_endpoint(host: str, port: int) -> str:
        if ":" in host and not (host.startswith("[") and host.endswith("]")):
            return f"[{host}]:{port}"
        return f"{host}:{port}"

    @staticmethod
    def _compact_endpoint(host: str, port: int) -> str:
        endpoint = f"{host}:{port}"
        if len(endpoint) <= 34:
            return endpoint
        return f"...{endpoint[-31:]}"

    def connect(self, host: str | None = None) -> None:
        endpoint_text = host if host is not None else self.host_box.text
        try:
            next_host, next_port = self._parse_endpoint(endpoint_text, self.default_port)
        except ValueError as exc:
            self.status_text.set_text(f"Invalid endpoint: {exc}")
            self.fig.canvas.draw_idle()
            return

        next_sock = make_debug_sub_conflate(make_tcp_endpoint(next_host, next_port))
        old_sock = self.sock
        self.sock = next_sock
        self.host = next_host
        self.port = next_port
        if old_sock is not None:
            old_sock.close(linger=0)
        endpoint = self._compact_endpoint(self.host, self.port)
        save_endpoint(self.settings_key, self.host, self.port)
        self.status_text.set_text(f"Connected {endpoint}")
        self.fig.canvas.draw_idle()

    def recv(self, flags: int = 0):
        return self.sock.recv(flags=flags)
