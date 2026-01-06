import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

UDP_IP = "0.0.0.0"
UDP_PORT = 12348
FFT_SIZE = 1024

# Configure UDP receiver
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.2)

# Create plotting window
fig, ax = plt.subplots()
plt.grid(True)
x = np.arange(FFT_SIZE)
line, = ax.plot(x, np.zeros(FFT_SIZE))
ax.set_ylim(0, 1)
ax.set_xlabel('Subcarrier Index')
ax.set_ylabel('Power (dB)')
ax.set_title('Channel Response')

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
            
            # Auto-adjust Y axis
            current_max = np.max(power)
            ax.set_ylim(current_max - 60, current_max + 5)
            
    except socket.timeout:
        pass
    return line,

# Update every 50ms
ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.show()