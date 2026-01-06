import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ====== Configuration ======
# Set the filename to inspect here. 
# If None, it will try to find the latest capture_rd_*.mat in the current directory.
TARGET_FILE = 'capture_rd_20251224_104750.mat' 

# Constants based on System Parameters
B = 50e6           # Bandwidth: 50 MHz
N_fft = 1024       # OFDM FFT Size
N_cp = 128         # Cyclic Prefix
Ms = 100           # Number of Symbols
Md = 20            # Multiplexing Factor (Stride)
FONT_SIZE = 14

def main():
    # 1. Select File
    filename = TARGET_FILE
    if filename is None:
        files = glob.glob('capture_rd_*.mat')
        if not files:
            print("No 'capture_rd_*.mat' files found.")
            return
        # Get latest file
        filename = max(files, key=os.path.getctime)
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    print(f"Inspecting file: {filename}")

    # 2. Load Data
    try:
        data = sio.loadmat(filename)
        if 'rd_map' not in data:
            print(f"Error: 'rd_map' variable not found in {filename}")
            return
        rd_map = data['rd_map']
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 3. Calculate Axes
    # Derived Physical Parameters
    T_sample = 1.0 / B
    T_sym = (N_fft + N_cp) * T_sample
    PRF_eff = 1.0 / (Md * T_sym)      # Approx 2170 Hz
    
    num_range_bins, num_doppler_bins = rd_map.shape
    
    # Range Axis in Nanoseconds (ns)
    # Range Resolution (Physical): 1/B = 20 ns
    range_res_ns = (1.0 / B) * 1e9         
    oversample_range = 10.0
    dt_bin_ns = range_res_ns / oversample_range # 2 ns
    
    range_max_ns = (num_range_bins - 1) * dt_bin_ns
    
    # Doppler Axis in Hz
    dop_min_hz = -PRF_eff / 2.0
    dop_max_hz = PRF_eff / 2.0

    # 4. Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # extent=[left, right, bottom, top]
    # Origin 'lower' means index (0,0) is at (left, bottom).
    im = ax.imshow(rd_map, 
                   origin='lower',
                   aspect='auto',
                   cmap='jet',
                   extent=[dop_min_hz, dop_max_hz, 0, range_max_ns],
                   vmin=-20, vmax=60)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Magnitude (dB)', fontsize=FONT_SIZE)
    
    ax.set_title(f'Inspect RD Map: {filename}', fontsize=FONT_SIZE)
    ax.set_xlabel('Doppler Frequency (Hz)', fontsize=FONT_SIZE)
    ax.set_ylabel('Delay (ns)', fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)

    # 5. Add Interactive Cursor (Mouse Hover)
    # Standard matplotlib format_coord is good, but we can make it better
    # or use an annotation text that follows the mouse.
    
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(event):
        x, y = event.xdata, event.ydata
        annot.xy = (x, y)
        
        # Find nearest index in data
        # x is Doppler (Hz), mapped from [dop_min, dop_max] to [0, cols-1]
        col_idx = int((x - dop_min_hz) / (dop_max_hz - dop_min_hz) * (num_doppler_bins - 1))
        
        # y is Delay (ns), mapped from [0, range_max] to [0, rows-1]
        row_idx = int(y / range_max_ns * (num_range_bins - 1))
        
        if 0 <= col_idx < num_doppler_bins and 0 <= row_idx < num_range_bins:
            val = rd_map[row_idx, col_idx]
            text = f"Doppler: {x:.1f} Hz\nDelay: {y:.1f} ns\nMag: {val:.1f} dB"
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.8)
            return True
        return False

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont = update_annot(event)
            if cont:
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

    # Connect event
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    # Also override standard toolbar coordinate formatter for convenience
    def format_coord(x, y):
        col_idx = int((x - dop_min_hz) / (dop_max_hz - dop_min_hz) * (num_doppler_bins - 1))
        row_idx = int(y / range_max_ns * (num_range_bins - 1))
        if 0 <= col_idx < num_doppler_bins and 0 <= row_idx < num_range_bins:
            z = rd_map[row_idx, col_idx]
            return f"Doppler={x:.1f} Hz, Delay={y:.1f} ns, Mag={z:.1f} dB"
        return f"x={x:.1f}, y={y:.1f}"
    
    ax.format_coord = format_coord

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
