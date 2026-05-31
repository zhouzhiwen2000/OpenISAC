import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib
# Use Agg backend for non-interactive plotting, safer for multiprocessing
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from concurrent.futures import ProcessPoolExecutor

# Global Font Size
FONT_SIZE = 32

# Constants based on System Parameters
B = 50e6           # Bandwidth: 50 MHz
N_fft = 1024       # OFDM FFT Size
N_cp = 128         # Cyclic Prefix
Ms = 100           # Number of Symbols
Md = 20            # Multiplexing Factor (Stride)

# Derived Physical Parameters
T_sample = 1.0 / B
T_sym = (N_fft + N_cp) * T_sample # Symbol Duration (approx 23.04 us)
T_coh = Ms * Md * T_sym           # Coherent Processing Interval (approx 46.08 ms)

# Effective PRF for Doppler (1 / Effective Symbol Period)
# Effective Symbol Period = Md * T_sym
PRF_eff = 1.0 / (Md * T_sym)      # Approx 2170 Hz

# Output directory
output_dir = 'plot_results'

def process_rd_file(current_file):
    # Set font size for this process
    plt.rcParams.update({'font.size': FONT_SIZE})
    
    print(f'Processing RD Map: {current_file}')
    try:
        data = sio.loadmat(current_file)
        if 'rd_map' in data:
            rd_map = data['rd_map']
            
            num_range_bins, num_doppler_bins = rd_map.shape
            
            # --- AXIS CALCULATION ---
            # Range Resolution (Physical): 1/B = 20 ns
            range_res_ns = (1.0 / B) * 1e9         # 20 ns
            oversample_range = 10.0
            dt_bin_ns = range_res_ns / oversample_range # 2 ns
            
            # Range Axis in Nanoseconds (ns)
            range_max_ns = (num_range_bins - 1) * dt_bin_ns
            
            # DOPPLER AXIS
            dop_min_hz = -PRF_eff / 2.0
            dop_max_hz = PRF_eff / 2.0
            
            # Create Invisible Figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(rd_map, 
                           origin='lower',
                           aspect='auto',
                           cmap='jet',
                           extent=[dop_min_hz, dop_max_hz, 0, range_max_ns],
                           vmin=-40, vmax=20)
            
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Magnitude (dB)', fontsize=FONT_SIZE)
            
            ax.set_xlabel('Doppler Frequency (Hz)', fontsize=FONT_SIZE)
            ax.set_ylabel('Delay (ns)', fontsize=FONT_SIZE)
            
            # Construct Output Filenames
            base_name = os.path.splitext(os.path.basename(current_file))[0]
            out_png = os.path.join(output_dir, f"{base_name}.png")
            out_pdf = os.path.join(output_dir, f"{base_name}.pdf")
            
            # Save Figure
            fig.savefig(out_png, dpi=300, bbox_inches='tight')
            fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
            
            plt.close(fig)
        else:
            print(f"Variable 'rd_map' not found in {current_file}")
    except Exception as e:
        print(f"Error processing {current_file}: {e}")

def process_md_file(current_file):
    # Set font size for this process
    plt.rcParams.update({'font.size': FONT_SIZE})
    
    print(f'Processing Micro-Doppler Map: {current_file}')
    try:
        data = sio.loadmat(current_file)
        if 'md_map' in data:
            md_map = data['md_map']
            
            num_freq_bins, num_time_bins = md_map.shape
            
            # Axis vectors
            dop_min_hz = -PRF_eff / 2.0
            dop_max_hz = PRF_eff / 2.0
            
            # X-Axis: Time (Seconds)
            stft_step = 64.0
            dt_step = stft_step / PRF_eff
            
            time_max_s = (num_time_bins - 1) * dt_step
            
            # Create Invisible Figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(md_map, 
                           origin='lower',
                           aspect='auto',
                           cmap='jet',
                           extent=[0, time_max_s, dop_min_hz, dop_max_hz],
                           vmin=-40, vmax=20)
            
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Magnitude (dB)', fontsize=FONT_SIZE)
            
            ax.set_xlabel('Time (s)', fontsize=FONT_SIZE)
            ax.set_ylabel('Doppler Frequency (Hz)', fontsize=FONT_SIZE)
            
            # Construct Output Filenames
            base_name = os.path.splitext(os.path.basename(current_file))[0]
            out_png = os.path.join(output_dir, f"{base_name}.png")
            out_pdf = os.path.join(output_dir, f"{base_name}.pdf")
            
            # Save Figure
            fig.savefig(out_png, dpi=300, bbox_inches='tight')
            fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
            
            plt.close(fig)
        else:
            print(f"Variable 'md_map' not found in {current_file}")
    except Exception as e:
        print(f"Error processing {current_file}: {e}")

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rd_files = glob.glob('capture_rd_*.mat')
    md_files = glob.glob('capture_md_*.mat')

    # Use ProcessPoolExecutor for parallel processing
    # max_workers defaults to the number of processors on the machine
    with ProcessPoolExecutor() as executor:
        if rd_files:
            print(f'--- Processing {len(rd_files)} RD Map files in parallel ---')
            # Submit all RD tasks
            executor.map(process_rd_file, rd_files)
        else:
            print('No RD Map files found (capture_rd_*.mat).')

        if md_files:
            print(f'--- Processing {len(md_files)} Micro-Doppler Map files in parallel ---')
            # Submit all MD tasks
            executor.map(process_md_file, md_files)

    print('All processing complete.')
