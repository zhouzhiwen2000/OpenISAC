import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import zmq
from debug_zmq_viewer import DebugZmqConnector

# Backend host and default debug port. The Endpoint box also accepts host:port,
# for example 127.0.0.1:12359 for the BS uplink delay-profile stream.
HOST = "127.0.0.1"
UDP_PORT = 12349
FFT_SIZE = 1024


def main() -> None:
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.18)
    connector = DebugZmqConnector(fig, UDP_PORT, HOST)
    x = np.arange(FFT_SIZE)
    line, = ax.plot(x, np.zeros(FFT_SIZE))
    vline = ax.axvline(0, color="r", linestyle="--", alpha=0)
    max_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, va="top")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Delay Tap")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Channel Delay Profile with Peak Marker")
    plt.grid(True)

    def update(_frame):
        try:
            data = connector.recv(flags=zmq.NOBLOCK)
            if len(data) == FFT_SIZE * np.dtype(np.complex64).itemsize:
                cdata = np.frombuffer(data, dtype=np.complex64)
                power = 10 * np.log10(np.abs(cdata) ** 2 + 1e-60)

                max_index = int(np.argmax(power))
                max_value = power[max_index]
                adjusted_index = max_index if max_index < FFT_SIZE // 2 else max_index - FFT_SIZE

                line.set_ydata(power)
                vline.set_xdata([max_index, max_index])
                vline.set_alpha(1)
                vline.set_ydata([ax.get_ylim()[0], ax.get_ylim()[1]])

                max_text.set_text(
                    f"Peak: {max_value:.1f} dB\n"
                    f"Index: {max_index} ({adjusted_index})"
                )

                current_max = np.max(power)
                ax.set_ylim(current_max - 60, current_max + 5)

        except zmq.Again:
            pass
        return line, vline, max_text

    ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
    plt.show()
    _ = ani


if __name__ == "__main__":
    main()
