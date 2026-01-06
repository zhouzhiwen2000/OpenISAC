import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import struct

# UDP receiver configuration
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 12346    # Port number
FFT_SIZE = 1024     # OFDM symbol FFT size (number of subcarriers)
MAX_PACKET_SIZE = 8192  # Maximum packet size (1024 points * 8 bytes/complex)

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.001)  # Set timeout for non-blocking behavior

# Define subcarrier indices:
# - pilot_indices: Pilot subcarrier indices
# - all_active_indices: All non-guard band subcarriers
# - data_indices: Data subcarriers (excluding pilots)
pilot_indices = [571, 631, 692, 752, 812, 872, 933, 993, 29, 89, 150, 210, 270, 330, 391, 451]
all_active_indices = np.concatenate([np.arange(1, 490), np.arange(535, 1024)])
data_indices = np.setdiff1d(all_active_indices, pilot_indices)

# Create figure and axes
fig, ax = plt.subplots()
ax.grid(True)
ax.set_title("OFDM Constellation")
ax.set_xlabel("Real (I)")
ax.set_ylabel("Imaginary (Q)")
ax.set_xlim([-2, 2])  # Set axis range
ax.set_ylim([-2, 2])

# Initialize scatter plots for three types of subcarriers
data_plot = ax.scatter([], [], alpha=0.6, c='blue', label='Data Subcarriers', s=10)
guard_plot = ax.scatter([], [], alpha=0.6, c='red', marker='*', label='Guard Band', s=40)
pilot_plot = ax.scatter([], [], alpha=0.6, c='orange', marker='x', label='Pilot Subcarriers', s=50)
ax.legend(loc='upper right')  # Add legend

# Global variable: stores current OFDM symbol
current_symbol = np.zeros(FFT_SIZE, dtype=np.complex64)

def update(frame):
    """Animation update function, called periodically by FuncAnimation"""
    global current_symbol  # Reference global variable
    
    # Try to receive new data
    try:
        # Receive UDP packet
        data, addr = sock.recvfrom(MAX_PACKET_SIZE)
        
        # Check if packet size is valid (FFT_SIZE * 8 bytes)
        if len(data) != FFT_SIZE * 8:
            return data_plot, guard_plot, pilot_plot
            
        # Convert byte data to complex array
        symbol = np.frombuffer(data, dtype=np.complex64)
        if symbol.size != FFT_SIZE:
            return data_plot, guard_plot, pilot_plot
            
        current_symbol = symbol  # Update current symbol
        
    except socket.timeout:  # Skip if no data
        pass
    except Exception as e:  # Catch other exceptions
        print(f"Error: {e}")
    
    # Update constellation with current symbol
    if current_symbol is not None:
        # Extract subcarrier types
        data_sc = current_symbol[data_indices]    # Data subcarriers
        guard_sc = current_symbol[490:534]        # Guard band subcarriers (490-533)
        pilot_sc = current_symbol[pilot_indices]  # Pilot subcarriers
        
        # Update scatter plot data:
        # Combine real and imaginary parts into (N, 2) array
        data_plot.set_offsets(np.column_stack([data_sc.real, data_sc.imag]))
        guard_plot.set_offsets(np.column_stack([guard_sc.real, guard_sc.imag]))
        pilot_plot.set_offsets(np.column_stack([pilot_sc.real, pilot_sc.imag]))
    
    # Return graphics elements that need updating
    return data_plot, guard_plot, pilot_plot

# Create animation object
ani = FuncAnimation(
    fig, 
    update, 
    interval=10,  # Update interval (milliseconds)
    blit=True,     # Enable efficient rendering
    cache_frame_data=False  # Don't cache frame data to save memory
)

plt.tight_layout()  # Adjust layout
plt.show()  # Display figure