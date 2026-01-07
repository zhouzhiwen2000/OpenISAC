import numpy as np
# Try to import CuPy; fall back to CPU if unavailable or no CUDA device
USE_GPU = False
cp = None
try:
    import cupy as cp  # type: ignore
    try:
        _gpu_count = cp.cuda.runtime.getDeviceCount()
        if _gpu_count > 0:
            USE_GPU = True
            print(f"Backend: Using GPU via CuPy (devices: {_gpu_count})")
        else:
            print("Backend: CuPy found but no CUDA device detected; using CPU (NumPy)")
    except Exception as _e:
        print(f"Backend: CuPy available but CUDA init failed: {_e}; using CPU (NumPy)")
        USE_GPU = False
except Exception:
    print("Backend: CuPy not available; using CPU (NumPy)")

# Try to import Numba for CPU acceleration
HAVE_NUMBA = False
try:
    from numba import njit, prange
    HAVE_NUMBA = True
    print("Backend: Numba available for CPU acceleration")
except ImportError:
    print("Backend: Numba not available")
import socket
import struct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox, Slider
import matplotlib.patches as patches
import scipy.io as sio
import datetime
try:
    from scipy.signal import stft  # type: ignore
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False
    def stft(x, fs=1.0, window='hamming', nperseg=256, noverlap=192, nfft=256, return_onesided=False):
        """Minimal STFT fallback using NumPy if SciPy is unavailable."""
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.ravel()
        step = nperseg - noverlap
        if step <= 0:
            raise ValueError("noverlap must be < nperseg")
        if isinstance(window, str):
            if window.lower() == 'hamming':
                w = np.hamming(nperseg).astype(np.float32)
            else:
                w = np.ones(nperseg, dtype=np.float32)
        else:
            w = np.asarray(window).astype(np.float32)
        n_frames = 1 + max(0, (len(x) - nperseg) // step)
        if n_frames <= 0:
            return np.array([]), np.array([]), np.zeros((0, 0), dtype=np.complex64)
        Zxx = np.empty((nfft, n_frames), dtype=np.complex64)
        for i in range(n_frames):
            start = i * step
            seg = x[start:start + nperseg]
            if len(seg) < nperseg:
                seg = np.pad(seg, (0, nperseg - len(seg)))
            segw = (seg * w).astype(np.complex64)
            buf = np.zeros(nfft, dtype=np.complex64)
            buf[:nperseg] = segw
            Zxx[:, i] = np.fft.fft(buf)
        f = np.fft.fftfreq(nfft, d=1.0 / fs)
        t = (np.arange(n_frames) * step) / fs
        if return_onesided:
            keep = f >= 0
            return f[keep], t, Zxx[keep, :]
        return f, t, Zxx
import sys
import os
import time
import threading
from queue import Queue
from collections import deque

# ====== Global Configuration ======
UDP_IP = "0.0.0.0"
UDP_PORT = 8888
CONTROL_PORT = 9999  # Port for sending control commands
FFT_SIZE = 1024
NUM_SYMBOLS = 100
MAX_CHUNK_SIZE = 60000
HEADER_SIZE = 12  # 3*uint32 (frame_id, total_chunks, chunk_id)
MAX_QUEUE_SIZE = 5  # Pre-render frame queue size (prevent memory overflow)
SOCKET_BUFFER_SIZE = 8 * 1024 * 1024  # 8MB UDP buffer, prevent packet loss
MAX_RANGE_BIN = 1000  # Max displayed range bin
MAX_DOPPLER_BINS = 1000  # Max displayed Doppler bin

# Zero padding parameters
RANGE_FFT_SIZE = 10240   # Points after zero padding in range dimension
DOPPLER_FFT_SIZE = 1000  # Points after zero padding in Doppler dimension
enable_mti = True  # Enable MTI (Moving Target Indication)

# ====== Thread-safe Data Structures ======
RAW_QUEUE_SIZE = 20
DISPLAY_QUEUE_SIZE = 5
DISPLAY_DOWNSAMPLE = 2 # Downsample factor for display (2 = 4x fewer pixels)

raw_frame_queue = Queue(maxsize=RAW_QUEUE_SIZE)  # Raw frequency domain data from UDP
display_queue = Queue(maxsize=DISPLAY_QUEUE_SIZE) # Processed data ready for display

buffer_lock = threading.Lock()  # Frame buffer lock

# For storing sender IP
sender_ip = None
sender_lock = threading.Lock()  # Sender IP lock

# Micro-Doppler analysis related variables
selected_range_bin = 0  # Default selected range bin
show_micro_doppler = True  # Whether to show micro-Doppler spectrum
range_time_data = None  # Store complex range-time spectrum data
range_time_lock = threading.Lock()  # Data storage lock
current_display_data = {} # Latest frame data for saving

# Micro-Doppler accumulation buffer
BUFFER_LENGTH = 5000  # Accumulate 5000 time points
micro_doppler_buffer = deque(maxlen=BUFFER_LENGTH)  # Use fixed-length deque
buffer_lock = threading.Lock()  # Buffer lock


class FrameBuffer:
    def __init__(self):
        self.frame_id = 0
        self.total_chunks = 0
        self.buffer = [None] * 1024  # Initial size
        self.received_chunks = 0
    
    def init(self, frame_id, total_chunks):
        self.frame_id = frame_id
        self.total_chunks = total_chunks
        self.buffer = [None] * total_chunks
        self.received_chunks = 0
    
    def add_chunk(self, chunk_id, data):
        if chunk_id < self.total_chunks and self.buffer[chunk_id] is None:
            self.buffer[chunk_id] = data
            self.received_chunks += 1
            return self.received_chunks == self.total_chunks
        return False
    
    def assemble_frame(self):
        byte_data = b''.join(self.buffer[:self.total_chunks])
        complex_array = np.frombuffer(byte_data, dtype=np.complex64)
        return complex_array.reshape((NUM_SYMBOLS, FFT_SIZE))

# Global Frame Buffer
current_frame = FrameBuffer()

# Send SKIP command
def send_skip_command():
    global sender_ip
    
    # Wait until sender IP is acquired
    for _ in range(100):  # Try 100 times
        with sender_lock:
            if sender_ip is not None:
                break
        time.sleep(0.1)
    
    if sender_ip is None:
        print("Error: Cannot send command - sender IP not detected.")
        return
        
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Construct command structure
        command_data = struct.pack("!4s4si", 
                                  b"CMD ",      # Header
                                  b"SKIP",      # Command ID: SKIP means skip sensing FFT
                                  1)           # Skip FFT processing
        
        sock.sendto(command_data, (sender_ip, CONTROL_PORT))
        print(f"Sent skip FFT command to {sender_ip}: SKIP 1")
        
        sock.close()
    except Exception as e:
        print(f"Failed to send skip FFT command: {str(e)}")

# ====== Independent Network Receiver Thread ======
def udp_receiver():
    global sender_ip
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Set large buffer to reduce packet loss risk
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_SIZE)
    except:
        pass  # Some systems may limit buffer size
    
    try:
        sock.bind((UDP_IP, UDP_PORT))
        print(f"Listening on UDP port {UDP_PORT}")
    except Exception as e:
        print(f"Socket bind error: {e}")
        sys.exit(1)
    
    last_log_time = time.time()
    processed_frames = 0
    
    while True:
        try:
            # Receive packet
            data, addr = sock.recvfrom(MAX_CHUNK_SIZE + HEADER_SIZE)
            
            # Record sender IP on first receive
            with sender_lock:
                if sender_ip is None:
                    sender_ip = addr[0]
                    print(f"Detected data sender IP: {sender_ip}")
            
            # Check for Control Heartbeat (NAT Traversal)
            if len(data) >= 8 and data[:4] == b'CTRL':
                global CONTROL_PORT
                # Update control port to punch through NAT
                if CONTROL_PORT != addr[1]:
                    CONTROL_PORT = addr[1]
                    print(f"Control Port Updated via Heartbeat (NAT Traversal): {sender_ip}:{CONTROL_PORT}")
                continue

            if len(data) < HEADER_SIZE:
                continue
            
            # Parse header
            try:
                frame_id, total_chunks, chunk_id = struct.unpack("!III", data[:HEADER_SIZE])
            except struct.error:
                continue
                
            chunk_data = data[HEADER_SIZE:]
            
            with buffer_lock:
                # Detect if new frame
                if current_frame.frame_id != frame_id:
                    current_frame.init(frame_id, total_chunks)
                
                # Add data chunk
                if current_frame.add_chunk(chunk_id, chunk_data):
                    # Frame assembly complete
                    processed_frames += 1
                    
                    # Calculate current FPS
                    current_time = time.time()
                    if current_time - last_log_time >= 1.0:  # Output once per second
                        fps = processed_frames / (current_time - last_log_time)
                        print(f"Processing FPS: {fps:.1f} | Frame {frame_id} complete")
                        last_log_time = current_time
                        processed_frames = 0
                    
                    # Get complete frame data - this is frequency domain data after channel estimation
                    frame_data = current_frame.assemble_frame()
                    
                    # Put frame data into queue for processing
                    if raw_frame_queue.full():
                        try:
                            raw_frame_queue.get_nowait()  # Discard oldest frame to avoid blocking receiver
                        except:
                            pass
                    raw_frame_queue.put(frame_data)
                    
        except socket.error as e:
            # Common non-critical errors (e.g. resource temporarily unavailable)
            if e.errno == 10054:  # Windows WSAECONNRESET
                pass
            else:
                print(f"Socket error: {e}")
        except Exception as e:
            print(f"Unexpected error in receiver: {str(e)}")

# ====== Start Receiver Thread ======
receiver_thread = threading.Thread(target=udp_receiver, daemon=True)
receiver_thread.start()

# ====== FFT Processing Function ======
if USE_GPU:
    range_window = cp.array(np.hamming(FFT_SIZE).reshape(1, FFT_SIZE), dtype=cp.float32)
    doppler_window = cp.array(np.hamming(NUM_SYMBOLS).reshape(NUM_SYMBOLS, 1), dtype=cp.float32)
else:
    range_window = np.hamming(FFT_SIZE).reshape(1, FFT_SIZE).astype(np.float32)
    doppler_window = np.hamming(NUM_SYMBOLS).reshape(NUM_SYMBOLS, 1).astype(np.float32)

if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True)
    def cpu_prep_range_fft(frame_data, window, fft_size, range_fft_size):
        # frame_data: (NUM_SYMBOLS, FFT_SIZE)
        # Output: padded_data (NUM_SYMBOLS, RANGE_FFT_SIZE)
        
        num_symbols = frame_data.shape[0]
        half_n = fft_size // 2
        
        out = np.zeros((num_symbols, range_fft_size), dtype=np.complex64)
        
        # Flattened window for faster access if possible, or just index nicely
        # window is (1, FFT_SIZE)
        
        for i in prange(num_symbols):
            # FFT shift axis 1 + Window + Pad
            
            # Destination 0..half_n-1 gets Source half_n..FFT_SIZE-1
            for j in range(half_n):
                val = frame_data[i, j + half_n]
                out[i, j] = val * window[0, j]
            
            # Destination half_n..FFT_SIZE-1 gets Source 0..half_n-1
            for j in range(half_n):
                val = frame_data[i, j]
                out[i, j + half_n] = val * window[0, j + half_n]
                
        return out

    @njit(parallel=True, fastmath=True)
    def cpu_prep_doppler_fft(range_time, window, num_symbols, doppler_fft_size, range_fft_size):
        out = np.zeros((doppler_fft_size, range_fft_size), dtype=np.complex64)
        
        for i in prange(num_symbols):
            win_val = window[i, 0]
            for j in range(range_fft_size):
                out[i, j] = range_time[i, j] * win_val
                
        return out

    @njit(parallel=True, fastmath=True)
    def cpu_calc_mag_db(doppler_fft_data, num_symbols, fft_size):
        # doppler_fft_data: (DOPPLER_FFT_SIZE, RANGE_FFT_SIZE)
        rows = doppler_fft_data.shape[0]
        cols = doppler_fft_data.shape[1]
        half_rows = rows // 2
        
        out = np.empty((rows, cols), dtype=np.float32)
        norm_factor = 1.0 / np.sqrt(num_symbols * fft_size)
        
        for j in prange(cols):
            # FFT shift axis 0 and Mag dB
            
            # Dest 0..half_rows-1 gets Source half_rows..rows-1
            for i in range(half_rows):
                val = doppler_fft_data[i + half_rows, j]
                mag = np.abs(val) * norm_factor
                out[i, j] = 20.0 * np.log10(mag + 1e-12)
                
            # Dest half_rows..rows-1 gets Source 0..half_rows-1
            for i in range(half_rows):
                val = doppler_fft_data[i, j]
                mag = np.abs(val) * norm_factor
                out[i + half_rows, j] = 20.0 * np.log10(mag + 1e-12)
                
        return out

def process_range_doppler(frame_data, max_view_range_bins=None):
    """
    Range and Doppler FFT processing (Automatically select GPU/CUDA or CPU/NumPy)
    """
    global range_time_data

    # Use global MAX_RANGE_BIN if not specified
    if max_view_range_bins is None:
        max_view_range_bins = MAX_RANGE_BIN

    if USE_GPU:
        # Transfer input data to GPU
        frame_data_gpu = cp.array(frame_data)

        # 1. Range dimension processing (GPU)
        shifted_data = cp.fft.fftshift(frame_data_gpu, axes=1)
        windowed_data = shifted_data * range_window
        padded_data = cp.zeros((NUM_SYMBOLS, RANGE_FFT_SIZE), dtype=cp.complex64)
        padded_data[:, :FFT_SIZE] = windowed_data
        range_time = cp.fft.ifft(padded_data, axis=1) * RANGE_FFT_SIZE

        # OPTIMIZATION: Crop range bins EARLY (before Doppler processing)
        # We only process the range bins we actually want to display/analyze
        if max_view_range_bins < RANGE_FFT_SIZE:
             range_time_view = range_time[:, :max_view_range_bins]
        else:
             range_time_view = range_time

        range_time_cpu = cp.asnumpy(range_time_view)

        # 2. Doppler dimension processing (GPU on reduced data)
        range_time_gpu = cp.array(range_time_cpu, dtype=cp.complex64)
        # Note: doppler_window must match dimension 0 (NUM_SYMBOLS), which is unchanged
        doppler_windowed = range_time_gpu * doppler_window
        
        # Padded doppler size depends on cropped range bins
        view_width = range_time_view.shape[1]
        padded_doppler = cp.zeros((DOPPLER_FFT_SIZE, view_width), dtype=cp.complex64)
        padded_doppler[:NUM_SYMBOLS, :] = doppler_windowed
        
        doppler_fft = cp.fft.fft(padded_doppler, axis=0)
        doppler_shifted = cp.fft.fftshift(doppler_fft, axes=0)
        magnitude = cp.abs(doppler_shifted) / cp.sqrt(NUM_SYMBOLS * FFT_SIZE)
        magnitude_db = 20 * cp.log10(magnitude + 1e-12)

        # Store complex range-time spectrum (CPU) - using the cropped view
        with range_time_lock:
            range_time_data = range_time_cpu

        return cp.asnumpy(magnitude_db)

    elif HAVE_NUMBA:
        # Optimized CPU path with Numba
        
        # 1. Range dimension prep (Shift -> Window -> Pad)
        padded_data = cpu_prep_range_fft(frame_data, range_window, FFT_SIZE, RANGE_FFT_SIZE)
        
        # Range IFFT
        range_time = np.fft.ifft(padded_data, axis=1) * RANGE_FFT_SIZE
        
        # OPTIMIZATION: Crop range bins EARLY
        if max_view_range_bins < RANGE_FFT_SIZE:
             range_time_view = range_time[:, :max_view_range_bins]
        else:
             range_time_view = range_time
             
        # Update view width
        view_width = range_time_view.shape[1]

        # 2. Doppler dimension prep (Window -> Pad)
        # We pass the REDUCED view_width to the padding function
        padded_doppler = cpu_prep_doppler_fft(range_time_view, doppler_window, NUM_SYMBOLS, DOPPLER_FFT_SIZE, view_width)
        
        # Doppler FFT
        doppler_fft = np.fft.fft(padded_doppler, axis=0)
        
        # 3. Post processing (Shift -> Abs -> Log10)
        magnitude_db = cpu_calc_mag_db(doppler_fft, NUM_SYMBOLS, FFT_SIZE)
        
        # Store complex range-time spectrum (CPU)
        with range_time_lock:
            range_time_data = range_time_view

        return magnitude_db

    else:
        # 1. Range dimension processing (CPU)
        shifted_data = np.fft.fftshift(frame_data, axes=1)
        windowed_data = shifted_data * range_window
        padded_data = np.zeros((NUM_SYMBOLS, RANGE_FFT_SIZE), dtype=np.complex64)
        padded_data[:, :FFT_SIZE] = windowed_data
        range_time = np.fft.ifft(padded_data, axis=1) * RANGE_FFT_SIZE

        # OPTIMIZATION: Crop range bins EARLY
        if max_view_range_bins < RANGE_FFT_SIZE:
             range_time_view = range_time[:, :max_view_range_bins]
        else:
             range_time_view = range_time
        
        view_width = range_time_view.shape[1]

        # 2. Doppler dimension processing (CPU)
        doppler_windowed = range_time_view * doppler_window
        padded_doppler = np.zeros((DOPPLER_FFT_SIZE, view_width), dtype=np.complex64)
        padded_doppler[:NUM_SYMBOLS, :] = doppler_windowed
        doppler_fft = np.fft.fft(padded_doppler, axis=0)
        doppler_shifted = np.fft.fftshift(doppler_fft, axes=0)
        magnitude = np.abs(doppler_shifted) / np.sqrt(NUM_SYMBOLS * FFT_SIZE)
        magnitude_db = 20 * np.log10(magnitude + 1e-12)

        # Store complex range-time spectrum (CPU)
        with range_time_lock:
            range_time_data = range_time_view

        return magnitude_db

def accumulate_range_time_data():
    """
    Add current frame's range_time_data to accumulation buffer
    """
    with range_time_lock:
        if range_time_data is not None:
            # Extract data for selected range bin
            selected_data = range_time_data[:, selected_range_bin]
            
            # Add to accumulation buffer
            with buffer_lock:
                # Optimized: Use extend instead of loop
                micro_doppler_buffer.extend(selected_data)
            
            # Trigger worker thread
            # md_event.set() # Removed in DSP refactor

def calculate_micro_doppler():
    """
    Calculate the micro-Doppler spectrum using 5000 accumulated time points from the buffer
    """
    # Check buffer size
    if len(micro_doppler_buffer) < 256:  # At least 256 points are required to calculate STFT
        return None, None, None
    
    # Convert buffer to NumPy array
    with buffer_lock:
        complex_signal = np.array(micro_doppler_buffer)
    
    # Parameter settings
    fs = 1.0  # Normalized sampling rate
    nfft = 256  # FFT size
    nperseg = 256  # Points per segment
    noverlap = 192  # Overlap points
    
    # Calculate STFT
    f, t, Zxx = stft(complex_signal, fs=fs, window='hamming', 
                    nperseg=nperseg, noverlap=noverlap, nfft=nfft, 
                    return_onesided=False)
    
    # Calculate magnitude and convert to dB
    Pxx = np.abs(Zxx)
    Pxx_db = 20 * np.log10(Pxx + 1e-12)
    
    # FFT shift to move zero frequency to the center
    Pxx_db_shifted = np.fft.fftshift(Pxx_db, axes=0)
    f_shifted = np.fft.fftshift(f)
    
    # Keep zero Doppler in the center
    max_freq = 0.5  # Maximum display frequency (normalized)
    f_idx = (f_shifted > -max_freq) & (f_shifted < max_freq)
    
    return f_shifted[f_idx], t, Pxx_db_shifted[f_idx, :]

    return f_shifted[f_idx], t, Pxx_db_shifted[f_idx, :]

def dsp_worker():
    """
    Dedicated DSP Worker Thread
    Consumes raw frames, performs Range-Doppler & Micro-Doppler processing,
    and pushes results to display_queue.
    """
    global running, range_time_data
    
    print("DSP Worker started")
    
    frame_count_local = 0
    
    while True:
        try:
            # Get raw frame (Block with timeout to allow exit check)
            try:
                raw_frame = raw_frame_queue.get(timeout=0.1)
            except:
                # Timeout, check if we should still run
                # Note: We don't have a clean global 'running' check here easily accessible or reliable
                # But since it's daemon, it's fine.
                continue

            t_start_dsp = time.time()

            # 1. Range-Doppler Processing
            # Note: process_range_doppler updates global 'range_time_data' internally
            rd_spectrum = process_range_doppler(raw_frame)
            
            # Crop Display Range (same logic as before)
            if rd_spectrum.shape[1] > MAX_RANGE_BIN:
                rd_spectrum = rd_spectrum[:, :MAX_RANGE_BIN]
            
            # Crop Doppler Range
            center_idx = rd_spectrum.shape[0] // 2
            start_idx = max(0, center_idx - MAX_DOPPLER_BINS // 2)
            end_idx = min(rd_spectrum.shape[0], start_idx + MAX_DOPPLER_BINS)
            rd_spectrum = rd_spectrum[start_idx:end_idx, :]
            
            # Transpose for plotting
            rd_spectrum_plot = np.transpose(rd_spectrum)
            
            # OPTIMIZATION: Downsample for display to speed up set_data/rendering
            if DISPLAY_DOWNSAMPLE > 1:
                rd_spectrum_plot = rd_spectrum_plot[::DISPLAY_DOWNSAMPLE, ::DISPLAY_DOWNSAMPLE]

            # 2. Micro-Doppler Processing
            md_spectrum = None
            md_extent = None
            
            if show_micro_doppler:
                # Accumulate data
                accumulate_range_time_data()
                f, t, Pxx = calculate_micro_doppler()
                if Pxx is not None:
                    md_spectrum = Pxx
                    md_extent = [t[0], t[-1], f[0], f[-1]]
            
            frame_count_local += 1
            
            t_end_dsp = time.time()
            dsp_time = t_end_dsp - t_start_dsp
            
            # 3. Push to Display Queue
            result = {
                'raw': raw_frame,
                'rd': rd_spectrum_plot,
                'md': md_spectrum,
                'md_extent': md_extent,
                'dsp_time': dsp_time
            }
            
            if display_queue.full():
                try:
                    display_queue.get_nowait() # Drop oldest display frame to keep latency low
                except:
                    pass
            display_queue.put(result)
            
        except Exception as e:
            print(f"DSP Worker Error: {e}")
            import traceback
            traceback.print_exc()

# ====== Control Command Functions ======
def send_alignment_command(adjustment):
    global sender_ip
    
    # Wait until sender IP is acquired
    for _ in range(100):  # Try 100 times
        with sender_lock:
            if sender_ip is not None:
                break
        time.sleep(0.1)
    
    if sender_ip is None:
        print("Error: Cannot send command - sender IP not detected.")
        return
    
    try:
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Construct command structure
        command_data = struct.pack("!4s4si", 
                                  b"CMD ",      # Header
                                  b"ALGN",      # Command ID: ALGN (Alignment)
                                  int(adjustment))  # Alignment samples
        
        # Send to C++ side using detected IP
        sock.sendto(command_data, (sender_ip, CONTROL_PORT))
        print(f"Sent alignment command to {sender_ip}: ALGN {adjustment}")

        # Close socket
        sock.close()
    except Exception as e:
        print(f"Failed to send alignment command: {str(e)}")

def send_strd_command(val):
    """Send STRD configuration command"""
    try:
        strd_val = int(val)
        # Ensure value is within reasonable range
        if strd_val < 1: strd_val = 1
        if strd_val > NUM_SYMBOLS: strd_val = NUM_SYMBOLS
        
        # Send command
        global sender_ip
        if sender_ip is None:
            print("Error: Cannot send command - sender IP not detected.")
            return
            
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            command_data = struct.pack("!4s4si", 
                                      b"CMD ",      # Header
                                      b"STRD",      # Command ID: STRD (Sparse sampling stride)
                                      int(strd_val))
            sock.sendto(command_data, (sender_ip, CONTROL_PORT))
            print(f"Sent STRD command to {sender_ip}: {strd_val}")
            sock.close()
        except Exception as e:
            print(f"Failed to send STRD command: {str(e)}")
    except ValueError:
        print(f"Invalid STRD value: {val}")

def send_mti_command(enabled):
    """Send MTI control command to C++"""
    global sender_ip
    
    if sender_ip is None:
        print("Error: Cannot send command - sender IP not detected.")
        return
        
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Construct command structure
        command_data = struct.pack("!4s4si", 
                                  b"CMD ",      # Header
                                  b"MTI ",      # Command ID: MTI 
                                  1 if enabled else 0)
        
        sock.sendto(command_data, (sender_ip, CONTROL_PORT))
        print(f"Sent MTI command to {sender_ip}: {'Enable' if enabled else 'Disable'}")
        
        sock.close()
    except Exception as e:
        print(f"Failed to send MTI command: {str(e)}")

def update_range_bin(val):
    global selected_range_bin
    selected_range_bin = int(val)
    
    # Clear accumulation buffer when changing range bin
    with buffer_lock:
        micro_doppler_buffer.clear()
    
    print(f"Selected range bin updated to {selected_range_bin}, buffer cleared")

def toggle_micro_doppler(event):
    global show_micro_doppler
    show_micro_doppler = not show_micro_doppler
    btn_md.color = 'lightgreen' if show_micro_doppler else 'lightgray'
    print(f"Micro-Doppler display {'enabled' if show_micro_doppler else 'disabled'}")

# ====== GUI Setup ======
print("Setting up visualization...")
fig = plt.figure(figsize=(15, 8))  # Increase width for micro-Doppler spectrum

# Create grid layout
grid = plt.GridSpec(6, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1]*6)
ax_rd = plt.subplot(grid[:5, 0])  # Range-Doppler spectrum
ax_md = plt.subplot(grid[:5, 1])  # Micro-Doppler spectrum

# Initial empty image
init_data_rd = np.zeros((MAX_RANGE_BIN, MAX_DOPPLER_BINS))
rt_static = np.zeros((NUM_SYMBOLS,RANGE_FFT_SIZE))
init_data_md = np.zeros((64, 100))

# Range-Doppler spectrum image
doppler_bins = np.linspace(-0.5, 0.5, MAX_DOPPLER_BINS)
img_rd = ax_rd.imshow(init_data_rd, 
                    aspect='auto', 
                    origin='lower',
                    extent=[doppler_bins[0], doppler_bins[-1], 0, MAX_RANGE_BIN],
                    cmap=cm.jet,
                    interpolation='nearest', # Validation: 'nearest' is faster
                    vmin=0, vmax=60)
ax_rd.set_xlabel('Doppler Frequency')
ax_rd.set_ylabel('Range Bin')
ax_rd.set_title('Range-Doppler Spectrum')

# Micro-Doppler spectrum image
img_md = ax_md.imshow(init_data_md, 
                    aspect='auto', 
                    origin='lower',
                    cmap=cm.jet,
                    interpolation='nearest', # Validation: 'nearest' is faster
                    vmin=0, vmax=60)
ax_md.set_xlabel('Time (Symbol Index)')
ax_md.set_ylabel('Doppler Frequency')
ax_md.set_title('Micro-Doppler Spectrum')

# Create selection rectangle
rect = patches.Rectangle((-MAX_DOPPLER_BINS//2, selected_range_bin), MAX_DOPPLER_BINS, 3, 
                        linewidth=1, edgecolor='r', facecolor='r', alpha=0.3)
ax_rd.add_patch(rect)

# Add colorbars
cbar_rd = fig.colorbar(img_rd, ax=ax_rd, pad=0.01)
cbar_rd.set_label('Magnitude (dB)')
cbar_md = fig.colorbar(img_md, ax=ax_md, pad=0.01)
cbar_md.set_label('Magnitude (dB)')

# FPS and status display
fps_text = ax_rd.text(0.98, 0.95, "FPS: 0.0", transform=ax_rd.transAxes,
                    color='white', ha='right', va='top', fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.5))

queue_text = ax_rd.text(0.98, 0.83, "Queue: 0", transform=ax_rd.transAxes,
                      color='white', ha='right', va='top', fontsize=10,
                      bbox=dict(facecolor='black', alpha=0.5))

ip_text = ax_rd.text(0.98, 0.78, "Sender: Detecting...", transform=ax_rd.transAxes,
                  color='white', ha='right', va='top', fontsize=10,
                  bbox=dict(facecolor='black', alpha=0.5))

buffer_text = ax_md.text(0.02, 0.95, "Buffer: 0/5000", transform=ax_md.transAxes,
                     color='white', ha='left', va='top', fontsize=10,
                     bbox=dict(facecolor='black', alpha=0.5))

# Range bin selector
ax_range_textbox = plt.axes([0.15, 0.1, 0.08, 0.03])
range_text_box = TextBox(ax_range_textbox, 'Range Bin:', initial=str(selected_range_bin))

ax_range_submit = plt.axes([0.235, 0.1, 0.05, 0.03])
range_submit_btn = Button(ax_range_submit, 'Set')
range_submit_btn.label.set_fontsize(8)

def set_range_bin(text):
    try:
        bin_val = int(text)
        # Ensure value is within reasonable range
        if bin_val < 0: bin_val = 0
        if bin_val >= MAX_RANGE_BIN: bin_val = MAX_RANGE_BIN - 1
        
        update_range_bin(bin_val)
        print(f"Range bin set to: {bin_val}")
    except ValueError:
        print(f"Invalid range bin value: {text}")

range_submit_btn.on_clicked(lambda event: set_range_bin(range_text_box.text))

# Micro-Doppler toggle
ax_md_btn = plt.axes([0.29, 0.1, 0.1, 0.03])
btn_md = Button(ax_md_btn, 'Micro-Doppler', color='lightgreen')
btn_md.on_clicked(toggle_micro_doppler)

# Alignment control
ax_textbox = plt.axes([0.45, 0.02, 0.1, 0.03])
text_box = TextBox(ax_textbox, 'Delay:', initial="0")
alignment_value = 0

ax_submit = plt.axes([0.56, 0.02, 0.07, 0.03])
submit_btn = Button(ax_submit, 'Apply')
submit_btn.label.set_fontsize(8)

def apply_alignment(event):
    global alignment_value
    try:
        val = text_box.text.strip()
        if val:
            alignment_value = int(val)
        send_alignment_command(alignment_value)
        print(f"Sent alignment command: {alignment_value} samples")
    except ValueError:
        print(f"Invalid alignment value: {val}")
submit_btn.on_clicked(apply_alignment)

# Quick adjustment buttons
ax_btn_plus10 = plt.axes([0.64, 0.02, 0.06, 0.03])
btn_plus10 = Button(ax_btn_plus10, '+10')
btn_plus10.on_clicked(lambda event: send_alignment_command(10))

ax_btn_minus10 = plt.axes([0.71, 0.02, 0.06, 0.03])
btn_minus10 = Button(ax_btn_minus10, '-10')
btn_minus10.on_clicked(lambda event: send_alignment_command(-10))

ax_btn_plus = plt.axes([0.78, 0.02, 0.05, 0.03])
btn_plus = Button(ax_btn_plus, '+1')
btn_plus.on_clicked(lambda event: send_alignment_command(1))

ax_btn_minus = plt.axes([0.84, 0.02, 0.05, 0.03])
btn_minus = Button(ax_btn_minus, '-1')
btn_minus.on_clicked(lambda event: send_alignment_command(-1))

# STRD control
ax_strd_textbox = plt.axes([0.1, 0.02, 0.06, 0.03])
strd_text_box = TextBox(ax_strd_textbox, 'STRD:', initial="20")

ax_strd_submit = plt.axes([0.18, 0.02, 0.06, 0.03])
strd_submit_btn = Button(ax_strd_submit, 'Set')
strd_submit_btn.label.set_fontsize(8)
strd_submit_btn.on_clicked(lambda event: send_strd_command(strd_text_box.text))

sparse_text = fig.text(0.175, 0.05, 
                      "STRD: Sparse Sampling Stride for OFDM Symbols", 
                      ha='center', va='bottom', fontsize=9, color='gray')

# Add info text
plt.subplots_adjust(bottom=0.15)  # Leave space for buttons
info_text = fig.text(0.5, 0.05, 
                    "Set delay then click 'Apply' or use quick buttons", 
                    ha='center', va='bottom', fontsize=9, color='gray')
ax_mti_btn = plt.axes([0.40, 0.1, 0.07, 0.03])
btn_mti = Button(ax_mti_btn, 'MTI', color='lightgreen' if enable_mti else 'lightgray')

def toggle_mti(event):
    global enable_mti
    enable_mti = not enable_mti
    btn_mti.color = 'lightgreen' if enable_mti else 'lightgray'
    print(f"MTI {'enabled' if enable_mti else 'disabled'}")
    send_mti_command(enable_mti)
    
btn_mti.on_clicked(toggle_mti)

# ====== Save Functions ======
def save_raw_frame(event):
    if 'raw' in current_display_data:
        data = current_display_data['raw']
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"./capture/capture_raw_{ts}.mat"
        try:
            os.makedirs("./capture", exist_ok=True)
            sio.savemat(fname, {'raw_frame': data})
            print(f"Saved Raw frame to {fname} (Shape: {data.shape})")
        except Exception as e:
            print(f"Error saving Raw frame: {e}")
    else:
        print("No Raw data available to save")

def save_rd_map(event):
    if 'rd' in current_display_data:
        data = current_display_data['rd']
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"./capture/capture_rd_{ts}.mat"
        try:
            os.makedirs("./capture", exist_ok=True)
            sio.savemat(fname, {'rd_map': data})
            print(f"Saved RD Map to {fname} (Shape: {data.shape})")
        except Exception as e:
            print(f"Error saving RD Map: {e}")
    else:
        print("No RD Map data available to save")

def save_md_map(event):
    if 'md' in current_display_data:
        data = current_display_data['md']
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"./capture/capture_md_{ts}.mat"
        try:
            os.makedirs("./capture", exist_ok=True)
            sio.savemat(fname, {'md_map': data})
            print(f"Saved Micro-Doppler Map to {fname} (Shape: {data.shape})")
        except Exception as e:
            print(f"Error saving MD Map: {e}")
    else:
        print("No MD Map data available to save")

# Save Buttons
btn_width = 0.07
btn_height = 0.03
btn_y = 0.1

ax_save_raw = plt.axes([0.55, btn_y, btn_width, btn_height])
btn_save_raw = Button(ax_save_raw, 'Save Raw')
btn_save_raw.label.set_fontsize(8)
btn_save_raw.on_clicked(save_raw_frame)

ax_save_rd = plt.axes([0.63, btn_y, btn_width, btn_height])
btn_save_rd = Button(ax_save_rd, 'Save RD')
btn_save_rd.label.set_fontsize(8)
btn_save_rd.on_clicked(save_rd_map)

ax_save_md = plt.axes([0.71, btn_y, btn_width, btn_height])
btn_save_md = Button(ax_save_md, 'Save MD')
btn_save_md.label.set_fontsize(8)
btn_save_md.on_clicked(save_md_map)

# Send SKIP command
send_skip_command()

# ====== Animation Update Function ======
last_update_time = time.time()
frame_count = 0
total_dsp_time = 0.0  # Accumulate DSP processing time
total_python_time = 0.0 # Accumulate Total processing time
fps_update_interval = 0.5  # FPS update interval (seconds)
running = True  # Animation running status

def update(frame_num):
    global last_update_time, frame_count, running, sender_ip, selected_range_bin, total_dsp_time, total_python_time
    
    t_start_update = time.time() # Start Total Timer
    
    if not running:
        return []
    
    try:
        # Update sender IP display
        with sender_lock:
            if sender_ip:
                ip_text.set_text(f"Sender: {sender_ip}")
        
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - last_update_time
        
        # Update FPS display
        if elapsed > fps_update_interval:
            fps = frame_count / elapsed
            avg_dsp = (total_dsp_time / frame_count * 1000) if frame_count > 0 else 0
            avg_total = (total_python_time / frame_count * 1000) if frame_count > 0 else 0
            
            fps_text.set_text(f"FPS: {fps:.1f}\nDSP: {avg_dsp:.1f}ms\nTotal: {avg_total:.1f}ms")
            
            frame_count = 0
            total_dsp_time = 0.0
            total_python_time = 0.0
            last_update_time = current_time
        
        # Update queue status display
        queue_text.set_text(f"RawQ: {raw_frame_queue.qsize()} | DispQ: {display_queue.qsize()}")
        
        # Get result from Display Queue
        updated = False
        latest_result = None
        
        # Retrieve all available results, keep the latest one (Skip frames if drawing is slow)
        while not display_queue.empty():
            latest_result = display_queue.get()
            updated = True
        
        # Update Plots
        if updated and latest_result is not None:
            global current_display_data
            current_display_data = latest_result

            # RD Plot
            img_rd.set_data(latest_result['rd'])
            
            # MD Plot
            if latest_result['md'] is not None:
                img_md.set_data(latest_result['md'])
                # Only set extent if it changes or just set it (cheap)
                if latest_result['md_extent']:
                    img_md.set_extent(latest_result['md_extent'])
                    ax_md.set_xlabel('Time (Sample Index)')
                    ax_md.set_ylabel('Doppler Frequency')
            
            # Update selection rectangle
            rect.set_y(selected_range_bin)
            
            # Buffer status
            if show_micro_doppler:
                buffer_text.set_text(f"Buffer: {len(micro_doppler_buffer)}/{BUFFER_LENGTH}")
            
            # Accumulate DSP time from worker
            if 'dsp_time' in latest_result:
                total_dsp_time += latest_result['dsp_time']
                
            frame_count += 1
        
        # Return objects to update
        t_end_update = time.time()
        total_python_time += (t_end_update - t_start_update)
        return [img_rd, img_md, fps_text, queue_text, ip_text, rect, buffer_text]
    except Exception as e:
        print(f"Update error: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# ====== Close Event Handling ======
def on_close(event):
    global running
    print("Closing visualization...")
    running = False  # Stop animation loop
    print("Visualization closed. Receiver thread will exit automatically.")

# Connect close event
fig.canvas.mpl_connect('close_event', on_close)

# ====== Start Animation ======
print("Starting animation...")

# Start DSP Worker Thread
dsp_thread = threading.Thread(target=dsp_worker, daemon=True)
dsp_thread.start()

try:
    # Use short interval for high refresh rate
    ani = FuncAnimation(fig, update, 
                        interval=1,  # Try to update as fast as possible (was 50)
                        blit=True,
                        save_count=50)
    
    plt.show()
except Exception as e:
    print(f"Animation error: {str(e)}")
finally:
    running = False
    print("Animation stopped.")
