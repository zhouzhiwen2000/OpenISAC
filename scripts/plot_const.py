import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import zmq
from debug_zmq_viewer import DebugZmqConnector

# Receiver configuration. The Endpoint box also accepts host:port, for example
# 127.0.0.1:12356 for the BS uplink constellation stream.
HOST = "127.0.0.1"
UDP_PORT = 12346
FFT_SIZE = 1024
SHOW_GUARD_BAND = False

pilot_indices = [571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451]
all_active_indices = np.concatenate([np.arange(1, 490), np.arange(535, 1024)])
data_indices = np.setdiff1d(all_active_indices, pilot_indices)
guard_band_indices = np.arange(490, 534)


def main() -> None:
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.18)
    connector = DebugZmqConnector(fig, UDP_PORT, HOST)
    ax.grid(True)
    ax.set_title("OFDM Constellation")
    ax.set_xlabel("Real (I)")
    ax.set_ylabel("Imaginary (Q)")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    data_plot = ax.scatter([], [], alpha=0.6, c="blue", label="Data Subcarriers", s=10)
    guard_plot = ax.scatter(
        [],
        [],
        alpha=0.6,
        c="red",
        marker="*",
        label="Guard Band" if SHOW_GUARD_BAND else "_nolegend_",
        s=40,
    )
    pilot_plot = ax.scatter([], [], alpha=0.6, c="orange", marker="x", label="Pilot Subcarriers", s=50)
    guard_plot.set_visible(SHOW_GUARD_BAND)
    ax.legend(loc="upper right")

    current_symbol = np.zeros(FFT_SIZE, dtype=np.complex64)

    def update(_frame):
        nonlocal current_symbol
        try:
            data = connector.recv(flags=zmq.NOBLOCK)
            if len(data) != FFT_SIZE * np.dtype(np.complex64).itemsize:
                return data_plot, guard_plot, pilot_plot

            symbol = np.frombuffer(data, dtype=np.complex64)
            if symbol.size != FFT_SIZE:
                return data_plot, guard_plot, pilot_plot

            current_symbol = symbol

        except zmq.Again:
            pass
        except Exception as exc:
            print(f"Error: {exc}")

        data_sc = current_symbol[data_indices]
        pilot_sc = current_symbol[pilot_indices]

        data_plot.set_offsets(np.column_stack([data_sc.real, data_sc.imag]))
        if SHOW_GUARD_BAND:
            guard_sc = current_symbol[guard_band_indices]
            guard_plot.set_offsets(np.column_stack([guard_sc.real, guard_sc.imag]))
        else:
            guard_plot.set_offsets(np.empty((0, 2)))
        pilot_plot.set_offsets(np.column_stack([pilot_sc.real, pilot_sc.imag]))

        return data_plot, guard_plot, pilot_plot

    ani = FuncAnimation(
        fig,
        update,
        interval=10,
        blit=True,
        cache_frame_data=False,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()
    _ = ani


if __name__ == "__main__":
    main()
