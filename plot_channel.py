import socket
from matplotlib.pylab import fftshift
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

# Create plotting window with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.3)
x = np.arange(FFT_SIZE)

# Real part subplot
ax1.grid(True)
line1, = ax1.plot(x, np.zeros(FFT_SIZE), 'b')
ax1.set_ylim(-1, 1)
ax1.set_xlabel('Frequency Bin')
ax1.set_ylabel('Real Part')
ax1.set_title('Channel Response - Real Part')

# Imaginary part subplot
ax2.grid(True)
line2, = ax2.plot(x, np.zeros(FFT_SIZE), 'r')
ax2.set_ylim(-1, 1)
ax2.set_xlabel('Frequency Bin')
ax2.set_ylabel('Imaginary Part')
ax2.set_title('Channel Response - Imaginary Part')

# Magnitude subplot
ax3.grid(True)
line3, = ax3.plot(x, np.zeros(FFT_SIZE), 'g')
ax3.set_ylim(0, 1)
ax3.set_xlabel('Frequency Bin')
ax3.set_ylabel('Magnitude')
ax3.set_title('Channel Response - Magnitude')

def update(frame):
    try:
        data, _ = sock.recvfrom(FFT_SIZE * 8)
        if len(data) == FFT_SIZE * 8:
            # Data processing
            cdata = np.frombuffer(data, dtype=np.complex64)
            cdata = fftshift(cdata)
            
            # Extract real and imaginary parts
            real_part = np.real(cdata)
            imag_part = np.imag(cdata)
            magnitude = np.abs(cdata)
            
            # Update graphs
            line1.set_ydata(real_part)
            line2.set_ydata(imag_part)
            line3.set_ydata(magnitude)
            
            # Auto-adjust Y axis for real part
            real_max = np.max(np.abs(real_part))
            ax1.set_ylim(-real_max * 1.1, real_max * 1.1)
            
            # Auto-adjust Y axis for imaginary part
            imag_max = np.max(np.abs(imag_part))
            ax2.set_ylim(-imag_max * 1.1, imag_max * 1.1)
            
            # Auto-adjust Y axis for magnitude
            mag_max = np.max(magnitude)
            ax3.set_ylim(0, mag_max * 1.1)
            
    except socket.timeout:
        pass
    return line1, line2, line3

# Update every 50ms
ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.show()