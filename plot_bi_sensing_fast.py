
import numpy as np
import sys
import os
import time
import threading
from queue import Queue
from collections import deque
import socket
import struct
import datetime
import scipy.io as sio

# PyQt6 + PyQtGraph Imports
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# ====== Backend Detection ======
USE_NVIDIA_GPU = False
USE_INTEL_GPU = False
cp = None
xp = np

try:
    import cupy as _cp
    try:
        _gpu_count = _cp.cuda.runtime.getDeviceCount()
        if _gpu_count > 0:
            USE_NVIDIA_GPU = True
            cp = _cp
            xp = _cp
            print(f"Backend: Using NVIDIA GPU via CuPy (devices: {_gpu_count})")
    except Exception as _e:
        print(f"Backend: CuPy available but CUDA init failed: {_e}")
except ImportError:
    pass

if not USE_NVIDIA_GPU:
    def _try_intel_gpu():
        global USE_INTEL_GPU, cp, xp
        try:
            import dpnp as _dpnp
            import dpctl
            try:
                _intel_devices = [d for d in dpctl.get_devices() if d.is_gpu]
            except:
                _intel_devices = []
            if _intel_devices:
                return _dpnp, _intel_devices[0].name
            return None, None
        except:
            return None, None
    
    import threading
    _intel_result = [None, None]
    def _intel_init_thread():
        _intel_result[0], _intel_result[1] = _try_intel_gpu()
    
    _init_thread = threading.Thread(target=_intel_init_thread, daemon=True)
    _init_thread.start()
    _init_thread.join(timeout=5.0)
    
    if not _init_thread.is_alive() and _intel_result[0] is not None:
        USE_INTEL_GPU = True
        cp = _intel_result[0]
        xp = _intel_result[0]
        print(f"Backend: Using Intel GPU via dpnp ({_intel_result[1]})")

USE_GPU = USE_NVIDIA_GPU or USE_INTEL_GPU
if not USE_GPU:
    print("Backend: No GPU acceleration available; using CPU")

HAVE_NUMBA = False
try:
    from numba import njit, prange
    HAVE_NUMBA = True
    print("Backend: Numba available for CPU acceleration")
except ImportError:
    print("Backend: Numba not available")

def to_numpy(arr):
    if hasattr(arr, 'get'):
        return arr.get()
    elif hasattr(arr, 'asnumpy'):
        return arr.asnumpy()
    elif hasattr(arr, '__array__'):
        return np.asarray(arr)
    return arr

try:
    from scipy.signal import stft
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False
    def stft(x, fs=1.0, window='hamming', nperseg=256, noverlap=192, nfft=256, return_onesided=False):
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.ravel()
        step = nperseg - noverlap
        if isinstance(window, str):
            w = np.hamming(nperseg).astype(np.float32) if window.lower() == 'hamming' else np.ones(nperseg, dtype=np.float32)
        else:
            w = np.asarray(window).astype(np.float32)
        n_frames = 1 + max(0, (len(x) - nperseg) // step)
        Zxx = np.empty((nfft, n_frames), dtype=np.complex64)
        for i in range(n_frames):
            start = i * step
            seg = x[start:start + nperseg]
            if len(seg) < nperseg:
                seg = np.pad(seg, (0, nperseg - len(seg)))
            buf = np.zeros(nfft, dtype=np.complex64)
            buf[:nperseg] = (seg * w).astype(np.complex64)
            Zxx[:, i] = np.fft.fft(buf)
        f = np.fft.fftfreq(nfft, d=1.0 / fs)
        t = (np.arange(n_frames) * step) / fs
        return f, t, Zxx

# ====== Global Configuration ======
UDP_IP = "0.0.0.0"
UDP_PORT = 8889  # Bi-sensing uses port 8889
CONTROL_PORT = 9999
FFT_SIZE = 1024
NUM_SYMBOLS = 100
MAX_CHUNK_SIZE = 60000
HEADER_SIZE = 12
SOCKET_BUFFER_SIZE = 8 * 1024 * 1024
MAX_RANGE_BIN = 1000
MAX_DOPPLER_BINS = 1000
RANGE_FFT_SIZE = 10240
DOPPLER_FFT_SIZE = 1000
enable_mti = True
enable_range_window = True
enable_doppler_window = True

RAW_QUEUE_SIZE = 20
DISPLAY_QUEUE_SIZE = 5
DISPLAY_DOWNSAMPLE = 2

raw_frame_queue = Queue(maxsize=RAW_QUEUE_SIZE)
display_queue = Queue(maxsize=DISPLAY_QUEUE_SIZE)
buffer_lock = threading.Lock()
sender_ip = None
sender_lock = threading.Lock()
selected_range_bin = 0
show_micro_doppler = True
range_time_data = None
range_time_lock = threading.Lock()
current_display_data = {}
BUFFER_LENGTH = 5000
micro_doppler_buffer = deque(maxlen=BUFFER_LENGTH)

class FrameBuffer:
    def __init__(self):
        self.frame_id = 0
        self.total_chunks = 0
        self.buffer = [None] * 1024
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

current_frame = FrameBuffer()

def send_skip_command():
    global sender_ip
    for _ in range(100):
        with sender_lock:
            if sender_ip is not None:
                break
        time.sleep(0.1)
    if sender_ip is None:
        print("Error: Cannot send command - sender IP not detected.")
        return
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command_data = struct.pack("!4s4si", b"CMD ", b"SKIP", 1)
        sock.sendto(command_data, (sender_ip, CONTROL_PORT))
        print(f"Sent skip FFT command to {sender_ip}: SKIP 1")
        sock.close()
    except Exception as e:
        print(f"Failed to send skip FFT command: {str(e)}")

def udp_receiver():
    global sender_ip, CONTROL_PORT
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_SIZE)
    except:
        pass
    try:
        sock.bind((UDP_IP, UDP_PORT))
        print(f"Listening on UDP port {UDP_PORT}")
    except Exception as e:
        print(f"Socket bind error: {e}")
        sys.exit(1)
    
    while True:
        try:
            data, addr = sock.recvfrom(MAX_CHUNK_SIZE + HEADER_SIZE)
            with sender_lock:
                if sender_ip is None:
                    sender_ip = addr[0]
                    print(f"Detected data sender IP: {sender_ip}")
            if len(data) >= 8 and data[:4] == b'CTRL':
                if CONTROL_PORT != addr[1]:
                    CONTROL_PORT = addr[1]
                continue
            if len(data) < HEADER_SIZE:
                continue
            try:
                frame_id, total_chunks, chunk_id = struct.unpack("!III", data[:HEADER_SIZE])
            except struct.error:
                continue
            chunk_data = data[HEADER_SIZE:]
            with buffer_lock:
                if current_frame.frame_id != frame_id:
                    current_frame.init(frame_id, total_chunks)
                if current_frame.add_chunk(chunk_id, chunk_data):
                    frame_data = current_frame.assemble_frame()
                    if raw_frame_queue.full():
                        try:
                            raw_frame_queue.get_nowait()
                        except:
                            pass
                    raw_frame_queue.put(frame_data)
        except socket.error as e:
            if e.errno != 10054:
                print(f"Socket error: {e}")
        except Exception as e:
            print(f"Unexpected error in receiver: {str(e)}")

receiver_thread = threading.Thread(target=udp_receiver, daemon=True)
receiver_thread.start()

# FFT Processing
if USE_GPU:
    range_window = cp.array(np.hamming(FFT_SIZE).reshape(1, FFT_SIZE), dtype=cp.float32)
    doppler_window = cp.array(np.hamming(NUM_SYMBOLS).reshape(NUM_SYMBOLS, 1), dtype=cp.float32)
else:
    range_window = np.hamming(FFT_SIZE).reshape(1, FFT_SIZE).astype(np.float32)
    doppler_window = np.hamming(NUM_SYMBOLS).reshape(NUM_SYMBOLS, 1).astype(np.float32)

if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True)
    def cpu_prep_range_fft(frame_data, window, fft_size, range_fft_size):
        num_symbols = frame_data.shape[0]
        half_n = fft_size // 2
        out = np.zeros((num_symbols, range_fft_size), dtype=np.complex64)
        for i in prange(num_symbols):
            for j in range(half_n):
                out[i, j] = frame_data[i, j + half_n] * window[0, j]
            for j in range(half_n):
                out[i, j + half_n] = frame_data[i, j] * window[0, j + half_n]
        return out

    @njit(parallel=True, fastmath=True)
    def cpu_prep_doppler_fft(range_time, window, num_symbols, doppler_fft_size, range_fft_size):
        out = np.zeros((doppler_fft_size, range_fft_size), dtype=np.complex64)
        for i in prange(num_symbols):
            for j in range(range_fft_size):
                out[i, j] = range_time[i, j] * window[i, 0]
        return out

    @njit(parallel=True, fastmath=True)
    def cpu_calc_mag_db(doppler_fft_data, num_symbols, fft_size):
        rows = doppler_fft_data.shape[0]
        cols = doppler_fft_data.shape[1]
        half_rows = rows // 2
        out = np.empty((rows, cols), dtype=np.float32)
        norm_factor = 1.0 / np.sqrt(num_symbols * fft_size)
        for j in prange(cols):
            for i in range(half_rows):
                out[i, j] = 20.0 * np.log10(np.abs(doppler_fft_data[i + half_rows, j]) * norm_factor + 1e-12)
            for i in range(half_rows):
                out[i + half_rows, j] = 20.0 * np.log10(np.abs(doppler_fft_data[i, j]) * norm_factor + 1e-12)
        return out

def process_range_doppler(frame_data, max_view_range_bins=None):
    global range_time_data
    if max_view_range_bins is None:
        max_view_range_bins = MAX_RANGE_BIN

    range_win = range_window if enable_range_window else (cp.ones((1, FFT_SIZE), dtype=cp.float32) if USE_GPU else np.ones((1, FFT_SIZE), dtype=np.float32))
    doppler_win = doppler_window if enable_doppler_window else (cp.ones((NUM_SYMBOLS, 1), dtype=cp.float32) if USE_GPU else np.ones((NUM_SYMBOLS, 1), dtype=np.float32))

    if USE_GPU:
        frame_data_gpu = cp.array(frame_data)
        shifted_data = cp.fft.fftshift(frame_data_gpu, axes=1)
        windowed_data = shifted_data * range_win
        padded_data = cp.zeros((NUM_SYMBOLS, RANGE_FFT_SIZE), dtype=cp.complex64)
        padded_data[:, :FFT_SIZE] = windowed_data
        range_time = cp.fft.ifft(padded_data, axis=1) * RANGE_FFT_SIZE
        range_time_view = range_time[:, :max_view_range_bins] if max_view_range_bins < RANGE_FFT_SIZE else range_time
        range_time_cpu = to_numpy(range_time_view)
        range_time_gpu = cp.array(range_time_cpu, dtype=cp.complex64)
        doppler_windowed = range_time_gpu * doppler_win
        view_width = range_time_view.shape[1]
        padded_doppler = cp.zeros((DOPPLER_FFT_SIZE, view_width), dtype=cp.complex64)
        padded_doppler[:NUM_SYMBOLS, :] = doppler_windowed
        doppler_fft = cp.fft.fft(padded_doppler, axis=0)
        doppler_shifted = cp.fft.fftshift(doppler_fft, axes=0)
        magnitude = cp.abs(doppler_shifted) / np.sqrt(NUM_SYMBOLS * FFT_SIZE)
        magnitude_db = 20 * cp.log10(magnitude + 1e-12)
        with range_time_lock:
            range_time_data = range_time_cpu
        return to_numpy(magnitude_db)
    elif HAVE_NUMBA:
        padded_data = cpu_prep_range_fft(frame_data, range_win, FFT_SIZE, RANGE_FFT_SIZE)
        range_time = np.fft.ifft(padded_data, axis=1) * RANGE_FFT_SIZE
        range_time_view = range_time[:, :max_view_range_bins] if max_view_range_bins < RANGE_FFT_SIZE else range_time
        view_width = range_time_view.shape[1]
        padded_doppler = cpu_prep_doppler_fft(range_time_view, doppler_win, NUM_SYMBOLS, DOPPLER_FFT_SIZE, view_width)
        doppler_fft = np.fft.fft(padded_doppler, axis=0)
        magnitude_db = cpu_calc_mag_db(doppler_fft, NUM_SYMBOLS, FFT_SIZE)
        with range_time_lock:
            range_time_data = range_time_view
        return magnitude_db
    else:
        shifted_data = np.fft.fftshift(frame_data, axes=1)
        windowed_data = shifted_data * range_win
        padded_data = np.zeros((NUM_SYMBOLS, RANGE_FFT_SIZE), dtype=np.complex64)
        padded_data[:, :FFT_SIZE] = windowed_data
        range_time = np.fft.ifft(padded_data, axis=1) * RANGE_FFT_SIZE
        range_time_view = range_time[:, :max_view_range_bins] if max_view_range_bins < RANGE_FFT_SIZE else range_time
        view_width = range_time_view.shape[1]
        doppler_windowed = range_time_view * doppler_win
        padded_doppler = np.zeros((DOPPLER_FFT_SIZE, view_width), dtype=np.complex64)
        padded_doppler[:NUM_SYMBOLS, :] = doppler_windowed
        doppler_fft = np.fft.fft(padded_doppler, axis=0)
        doppler_shifted = np.fft.fftshift(doppler_fft, axes=0)
        magnitude_db = 20 * np.log10(np.abs(doppler_shifted) / np.sqrt(NUM_SYMBOLS * FFT_SIZE) + 1e-12)
        with range_time_lock:
            range_time_data = range_time_view
        return magnitude_db

def accumulate_range_time_data():
    with range_time_lock:
        if range_time_data is not None:
            with buffer_lock:
                micro_doppler_buffer.extend(range_time_data[:, selected_range_bin])

def calculate_micro_doppler():
    if len(micro_doppler_buffer) < 256:
        return None, None, None
    with buffer_lock:
        complex_signal = np.array(micro_doppler_buffer)
    f, t, Zxx = stft(complex_signal, fs=1.0, window='hamming', nperseg=256, noverlap=192, nfft=256, return_onesided=False)
    Pxx_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    Pxx_db_shifted = np.fft.fftshift(Pxx_db, axes=0)
    f_shifted = np.fft.fftshift(f)
    f_idx = (f_shifted > -0.5) & (f_shifted < 0.5)
    return f_shifted[f_idx], t, Pxx_db_shifted[f_idx, :]

def dsp_worker():
    print("DSP Worker started")
    # Power calculation for bi-sensing
    accumulated_power = 0.0
    power_frame_count = 0
    MAX_POWER_FRAMES = 100
    
    while True:
        try:
            try:
                raw_frame = raw_frame_queue.get(timeout=0.1)
            except:
                continue
            
            # Power calculation
            current_power = np.mean(np.abs(raw_frame)**2)
            accumulated_power += current_power
            power_frame_count += 1
            if power_frame_count >= MAX_POWER_FRAMES:
                avg_power = accumulated_power / MAX_POWER_FRAMES
                power_dbm = 10 * np.log10(avg_power + 1e-12)
                print(f"Average Power (last {MAX_POWER_FRAMES} frames): {power_dbm:.2f} dBm")
                accumulated_power = 0.0
                power_frame_count = 0
            
            t_start = time.time()
            rd_spectrum = process_range_doppler(raw_frame)
            if rd_spectrum.shape[1] > MAX_RANGE_BIN:
                rd_spectrum = rd_spectrum[:, :MAX_RANGE_BIN]
            center_idx = rd_spectrum.shape[0] // 2
            start_idx = max(0, center_idx - MAX_DOPPLER_BINS // 2)
            end_idx = min(rd_spectrum.shape[0], start_idx + MAX_DOPPLER_BINS)
            rd_spectrum = rd_spectrum[start_idx:end_idx, :]
            rd_spectrum_plot = rd_spectrum  # Keep as (doppler, range): Y=Doppler, X=Range
            if DISPLAY_DOWNSAMPLE > 1:
                rd_spectrum_plot = rd_spectrum_plot[::DISPLAY_DOWNSAMPLE, ::DISPLAY_DOWNSAMPLE]
            md_spectrum = None
            if show_micro_doppler:
                accumulate_range_time_data()
                f, t, Pxx = calculate_micro_doppler()
                if Pxx is not None:
                    md_spectrum = Pxx.T  # Transpose: Y=Time, X=Frequency
            dsp_time = time.time() - t_start
            result = {'raw': raw_frame, 'rd': rd_spectrum_plot, 'md': md_spectrum, 'dsp_time': dsp_time}
            if display_queue.full():
                try:
                    display_queue.get_nowait()
                except:
                    pass
            display_queue.put(result)
        except Exception as e:
            print(f"DSP Worker Error: {e}")
            import traceback
            traceback.print_exc()

dsp_thread = threading.Thread(target=dsp_worker, daemon=True)
dsp_thread.start()

# Command Functions
def send_command(header, cmd_id, value):
    global sender_ip
    if sender_ip is None:
        print("Error: Cannot send command - sender IP not detected.")
        return
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(struct.pack("!4s4si", header, cmd_id, int(value)), (sender_ip, CONTROL_PORT))
        print(f"Sent command {cmd_id} to {sender_ip}: {value}")
        sock.close()
    except Exception as e:
        print(f"Failed to send command: {str(e)}")

def send_alignment_command(val):
    send_command(b"CMD ", b"ALGN", val)

def send_strd_command(val):
    try:
        strd_val = max(1, min(NUM_SYMBOLS, int(val)))
        send_command(b"CMD ", b"STRD", strd_val)
    except ValueError:
        print(f"Invalid STRD value: {val}")

def send_mti_command(enabled):
    send_command(b"CMD ", b"MTI ", 1 if enabled else 0)

# ====== MainWindow with PyQt6 + PyQtGraph ======
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenISAC Bi-Sensing - PyQtGraph")
        self.resize(1600, 900)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Plot Area
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_widget)
        main_layout.addWidget(plot_widget, stretch=4)

        # Range-Doppler Plot
        self.rd_plot = pg.PlotWidget(title="Range-Doppler Spectrum (Bi-Sensing)")
        self.rd_img = pg.ImageItem()
        self.rd_plot.addItem(self.rd_img)
        self.rd_plot.setLabel('left', 'Doppler Bin')
        self.rd_plot.setLabel('bottom', 'Range Bin')
        # Set colormap
        self.rd_img.setLookupTable(pg.colormap.get('turbo').getLookupTable())
        # Add color bar
        self.rd_colorbar = pg.ColorBarItem(values=(-40, 20), colorMap=pg.colormap.get('turbo'), interactive=False)
        self.rd_colorbar.setImageItem(self.rd_img, insert_in=self.rd_plot.plotItem)
        plot_layout.addWidget(self.rd_plot)

        # Micro-Doppler Plot
        self.md_plot = pg.PlotWidget(title="Micro-Doppler Spectrum")
        self.md_img = pg.ImageItem()
        self.md_plot.addItem(self.md_img)
        self.md_plot.setLabel('left', 'Time')
        self.md_plot.setLabel('bottom', 'Frequency')
        self.md_img.setLookupTable(pg.colormap.get('turbo').getLookupTable())
        # Add color bar
        self.md_colorbar = pg.ColorBarItem(values=(-30, 30), colorMap=pg.colormap.get('turbo'), interactive=False)
        self.md_colorbar.setImageItem(self.md_img, insert_in=self.md_plot.plotItem)
        plot_layout.addWidget(self.md_plot)

        # Control Panel
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(control_panel, stretch=1)

        # Status Labels
        self.lbl_fps = QtWidgets.QLabel("FPS: 0.0")
        self.lbl_queue = QtWidgets.QLabel("Queue: 0")
        self.lbl_sender = QtWidgets.QLabel("Sender: Detecting...")
        self.lbl_buffer = QtWidgets.QLabel("MD Buffer: 0/5000")
        for lbl in [self.lbl_fps, self.lbl_queue, self.lbl_sender, self.lbl_buffer]:
            control_layout.addWidget(lbl)
        control_layout.addSpacing(20)

        # Range Bin
        rb_layout = QtWidgets.QHBoxLayout()
        rb_layout.addWidget(QtWidgets.QLabel("Range Bin:"))
        self.txt_range_bin = QtWidgets.QLineEdit(str(selected_range_bin))
        btn_set_rb = QtWidgets.QPushButton("Set")
        btn_set_rb.clicked.connect(self.set_range_bin)
        rb_layout.addWidget(self.txt_range_bin)
        rb_layout.addWidget(btn_set_rb)
        control_layout.addLayout(rb_layout)

        # Micro-Doppler Toggle
        self.btn_md = QtWidgets.QPushButton("Micro-Doppler: ON")
        self.btn_md.setCheckable(True)
        self.btn_md.setChecked(True)
        self.btn_md.clicked.connect(self.toggle_micro_doppler)
        self.btn_md.setStyleSheet("QPushButton:checked { background-color: lightgreen; }")
        control_layout.addWidget(self.btn_md)
        control_layout.addSpacing(20)

        # Alignment
        align_layout = QtWidgets.QHBoxLayout()
        align_layout.addWidget(QtWidgets.QLabel("Delay:"))
        self.txt_delay = QtWidgets.QLineEdit("0")
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.clicked.connect(self.apply_alignment)
        align_layout.addWidget(self.txt_delay)
        align_layout.addWidget(btn_apply)
        control_layout.addLayout(align_layout)

        quick_layout = QtWidgets.QHBoxLayout()
        for label, val in [('+10', 10), ('-10', -10), ('+1', 1), ('-1', -1)]:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda ch, v=val: send_alignment_command(v))
            quick_layout.addWidget(btn)
        control_layout.addLayout(quick_layout)
        control_layout.addSpacing(20)

        # STRD
        strd_layout = QtWidgets.QHBoxLayout()
        strd_layout.addWidget(QtWidgets.QLabel("STRD:"))
        self.txt_strd = QtWidgets.QLineEdit("20")
        btn_strd = QtWidgets.QPushButton("Set")
        btn_strd.clicked.connect(self.set_strd)
        strd_layout.addWidget(self.txt_strd)
        strd_layout.addWidget(btn_strd)
        control_layout.addLayout(strd_layout)
        control_layout.addSpacing(20)

        # Toggles
        self.btn_mti = QtWidgets.QPushButton("MTI")
        self.update_toggle_style(self.btn_mti, enable_mti)
        self.btn_mti.clicked.connect(self.toggle_mti)
        control_layout.addWidget(self.btn_mti)

        self.btn_range_win = QtWidgets.QPushButton("Range Window")
        self.update_toggle_style(self.btn_range_win, enable_range_window)
        self.btn_range_win.clicked.connect(self.toggle_range_win)
        control_layout.addWidget(self.btn_range_win)

        self.btn_doppler_win = QtWidgets.QPushButton("Doppler Window")
        self.update_toggle_style(self.btn_doppler_win, enable_doppler_window)
        self.btn_doppler_win.clicked.connect(self.toggle_doppler_win)
        control_layout.addWidget(self.btn_doppler_win)
        control_layout.addSpacing(20)

        # Save
        save_layout = QtWidgets.QHBoxLayout()
        for name, fn in [("Save Raw", self.save_raw), ("Save RD", self.save_rd), ("Save MD", self.save_md)]:
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(fn)
            save_layout.addWidget(btn)
        control_layout.addLayout(save_layout)
        control_layout.addStretch()

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(1)  # Max FPS

        send_skip_command()
        self.last_update_time = time.time()
        self.frame_count = 0
        self.total_dsp_time = 0.0

    def update_toggle_style(self, btn, state):
        btn.setStyleSheet("background-color: lightgreen;" if state else "background-color: lightgray;")

    def set_range_bin(self):
        global selected_range_bin
        try:
            val = max(0, min(MAX_RANGE_BIN - 1, int(self.txt_range_bin.text())))
            selected_range_bin = val
            with buffer_lock:
                micro_doppler_buffer.clear()
            print(f"Range bin set to: {val}")
        except ValueError:
            print("Invalid range bin")

    def toggle_micro_doppler(self):
        global show_micro_doppler
        show_micro_doppler = self.btn_md.isChecked()
        self.btn_md.setText(f"Micro-Doppler: {'ON' if show_micro_doppler else 'OFF'}")

    def apply_alignment(self):
        try:
            send_alignment_command(int(self.txt_delay.text()))
        except ValueError:
            print("Invalid alignment value")

    def set_strd(self):
        send_strd_command(self.txt_strd.text())

    def toggle_mti(self):
        global enable_mti
        enable_mti = not enable_mti
        self.update_toggle_style(self.btn_mti, enable_mti)
        send_mti_command(enable_mti)

    def toggle_range_win(self):
        global enable_range_window
        enable_range_window = not enable_range_window
        self.update_toggle_style(self.btn_range_win, enable_range_window)

    def toggle_doppler_win(self):
        global enable_doppler_window
        enable_doppler_window = not enable_doppler_window
        self.update_toggle_style(self.btn_doppler_win, enable_doppler_window)

    def save_raw(self):
        if 'raw' in current_display_data:
            self._save_mat('raw_frame', current_display_data['raw'], "raw")

    def save_rd(self):
        if 'rd' in current_display_data:
            self._save_mat('rd_map', current_display_data['rd'], "rd")

    def save_md(self):
        if 'md' in current_display_data:
            self._save_mat('md_map', current_display_data['md'], "md")

    def _save_mat(self, key, data, suffix):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"./capture/capture_{suffix}_{ts}.mat"
        try:
            os.makedirs("./capture", exist_ok=True)
            sio.savemat(fname, {key: data})
            print(f"Saved {suffix} to {fname}")
        except Exception as e:
            print(f"Error saving {suffix}: {e}")

    def update_plots(self):
        global current_display_data, sender_ip
        with sender_lock:
            if sender_ip:
                self.lbl_sender.setText(f"Sender: {sender_ip}")

        latest = None
        while not display_queue.empty():
            latest = display_queue.get()
        
        if latest:
            current_display_data = latest
            rd_data = latest['rd']
            self.rd_img.setImage(rd_data, autoLevels=False)

            if latest['md'] is not None:
                self.md_img.setImage(latest['md'], autoLevels=False)

            if 'dsp_time' in latest:
                self.total_dsp_time += latest['dsp_time']
            self.frame_count += 1
            self.lbl_buffer.setText(f"MD Buffer: {len(micro_doppler_buffer)}/{BUFFER_LENGTH}")

        current_time = time.time()
        elapsed = current_time - self.last_update_time
        if elapsed > 0.5:
            fps = self.frame_count / elapsed
            avg_dsp = (self.total_dsp_time / self.frame_count * 1000) if self.frame_count > 0 else 0
            self.lbl_fps.setText(f"FPS: {fps:.1f} | DSP: {avg_dsp:.1f}ms")
            self.lbl_queue.setText(f"RawQ: {raw_frame_queue.qsize()} | DispQ: {display_queue.qsize()}")
            self.frame_count = 0
            self.total_dsp_time = 0
            self.last_update_time = current_time

    def closeEvent(self, event):
        print("Closing application...")
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    send_skip_command()
    sys.exit(app.exec())
