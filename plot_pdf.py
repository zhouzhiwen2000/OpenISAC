import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

UDP_IP = "0.0.0.0"
UDP_PORT = 12349
FFT_SIZE = 1024

# Configure UDP receiver
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.2)

# Create plotting window
fig, ax = plt.subplots()
x = np.arange(FFT_SIZE)
line, = ax.plot(x, np.zeros(FFT_SIZE))
vline = ax.axvline(0, color='r', linestyle='--', alpha=0)
max_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top')
ax.set_ylim(0, 1)
ax.set_xlabel('Delay Tap')
ax.set_ylabel('Power (dB)')
ax.set_title('Channel Delay Profile with Peak Marker')
plt.grid(True)
def update(frame):
    try:
        data, _ = sock.recvfrom(FFT_SIZE * 8)
        if len(data) == FFT_SIZE * 8:
            # Data processing
            cdata = np.frombuffer(data, dtype=np.complex64)
            power = 10*np.log10(np.abs(cdata)**2 + 1e-60)  # Log power
            
            # Find maximum value
            max_index = np.argmax(power)
            max_value = power[max_index]
            
            # Adjust index display range
            adjusted_index = max_index if max_index < FFT_SIZE//2 else max_index - FFT_SIZE
            
            # Update graph
            line.set_ydata(power)
            vline.set_xdata([max_index, max_index])
            vline.set_alpha(1)  # Show marker line
            vline.set_ydata([ax.get_ylim()[0], ax.get_ylim()[1]])  # Full height
            
            # Update text
            max_text.set_text(f'Peak: {max_value:.1f} dB\n'
                             f'Index: {max_index} ({adjusted_index})')
            
            # Auto-adjust Y axis
            current_max = np.max(power)
            ax.set_ylim(current_max - 60, current_max + 5)
            
    except socket.timeout:
        pass
    return line, vline, max_text

# Update every 50ms
ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.show()