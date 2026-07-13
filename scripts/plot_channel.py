from matplotlib.pylab import fftshift
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import zmq
from debug_zmq_viewer import DebugZmqConnector

# Backend host and default debug port. The Endpoint box also accepts host:port,
# for example 127.0.0.1:12358 for the BS uplink channel stream.
HOST = "127.0.0.1"
UDP_PORT = 12348
FFT_SIZE = 1024


def _positive_span(value: float, fallback: float = 1.0) -> float:
    return max(float(value), fallback)


def main() -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 13))
    fig.subplots_adjust(bottom=0.15, hspace=0.45)
    connector = DebugZmqConnector(fig, UDP_PORT, HOST)
    x = np.arange(FFT_SIZE)

    ax1.grid(True)
    line1, = ax1.plot(x, np.zeros(FFT_SIZE), "b")
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel("Frequency Bin")
    ax1.set_ylabel("Real Part")
    ax1.set_title("Channel Response - Real Part")

    ax2.grid(True)
    line2, = ax2.plot(x, np.zeros(FFT_SIZE), "r")
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel("Frequency Bin")
    ax2.set_ylabel("Imaginary Part")
    ax2.set_title("Channel Response - Imaginary Part")

    ax3.grid(True)
    line3, = ax3.plot(x, np.zeros(FFT_SIZE), "g")
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Frequency Bin")
    ax3.set_ylabel("Magnitude")
    ax3.set_title("Channel Response - Magnitude")

    def update(_frame):
        try:
            data = connector.recv(flags=zmq.NOBLOCK)
            if len(data) == FFT_SIZE * np.dtype(np.complex64).itemsize:
                cdata = np.frombuffer(data, dtype=np.complex64)
                cdata = fftshift(cdata)

                real_part = np.real(cdata)
                imag_part = np.imag(cdata)
                magnitude = np.abs(cdata)

                line1.set_ydata(real_part)
                line2.set_ydata(imag_part)
                line3.set_ydata(magnitude)

                real_max = _positive_span(np.max(np.abs(real_part)))
                ax1.set_ylim(-real_max * 1.1, real_max * 1.1)

                imag_max = _positive_span(np.max(np.abs(imag_part)))
                ax2.set_ylim(-imag_max * 1.1, imag_max * 1.1)

                mag_max = _positive_span(np.max(magnitude))
                ax3.set_ylim(0, mag_max * 1.1)

        except zmq.Again:
            pass
        return line1, line2, line3

    ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
    plt.show()
    _ = ani


if __name__ == "__main__":
    main()
