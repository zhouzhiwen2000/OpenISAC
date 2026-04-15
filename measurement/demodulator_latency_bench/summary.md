# Demodulator Latency Benchmark Summary

## Scope

- Benchmark script: `scripts/bench_demodulator_latency.py`
- Sweep range:
  - `sample_rate`: `50e6`, `100e6`, `200e6`
  - `fft_size`: `1024`
  - `num_symbols`: `50`, `100`, `200`
- Payload size: `1024` bytes

## Data Source

The current `latency_summary.csv` only reflects the most recently rerun point, because the benchmark script rewrites that CSV when invoked on a subset of the sweep.

This summary is rebuilt from the latest `demodulator.log` under each run directory in `measurement/demodulator_latency_bench/lat_*`.

Parsed metrics:

- `RX frame queue wait`
- `Dequeue + FFT/EQ/LLR queue`
- `Bit queue + LDPC/UDP out`
- `TOTAL E2E (excl. RX wait)`

Latency definitions used below:

- `rx_queue_ms`: wait time inside `frame_queue_`
- `demod_ms`: dequeue to demod/LLR enqueue
- `bit_ms`: bit queue plus LDPC decode plus UDP output
- `e2e_ms`: `demod_ms + bit_ms`
- `e2e_ms` does not include `rx_queue_ms`

The table below uses sample-count-weighted averages over all valid latency blocks in each run log.

## Latest Reconstructed Results

| run_id | valid blocks | total samples | rx_queue_ms | demod_ms | bit_ms | e2e_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lat_sr50000000_fft1024_sym50` | 107 | 45169 | 0.010 | 0.395 | 0.293 | 0.689 |
| `lat_sr50000000_fft1024_sym100` | 51 | 22008 | 0.050 | 0.786 | 0.295 | 1.081 |
| `lat_sr50000000_fft1024_sym200` | 27 | 11717 | 0.051 | 1.652 | 0.310 | 1.962 |
| `lat_sr100000000_fft1024_sym50` | 210 | 58048 | 0.015 | 0.439 | 4.478 | 4.916 |
| `lat_sr100000000_fft1024_sym100` | 100 | 42671 | 0.011 | 0.851 | 1.803 | 2.654 |
| `lat_sr100000000_fft1024_sym200` | 54 | 23433 | 0.011 | 1.797 | 1.972 | 3.768 |
| `lat_sr200000000_fft1024_sym50` | 242 | 9429 | 15.573 | 0.492 | 1.686 | 2.178 |
| `lat_sr200000000_fft1024_sym100` | 114 | 2938 | 31.379 | 0.991 | 0.833 | 1.824 |
| `lat_sr200000000_fft1024_sym200` | 60 | 2093 | 66.028 | 2.090 | 0.781 | 2.871 |

## Main Findings

- `50e6` remains the clean region in this sweep. `rx_queue_ms` is negligible and `e2e_ms` grows roughly with `num_symbols`.
- `100e6` is workable at `sym100` and `sym200`, but `bit_ms` becomes much larger than at `50e6`. The worst case here is `sym50`, where `bit_ms` dominates the latency budget.
- `200e6` is still a saturated region. Under the new latency definition, `e2e_ms` itself is only about `1.8-2.9 ms`, but the real receive path is dominated by `frame_queue_` backlog:
  - `sym50`: `~15.6 ms`
  - `sym100`: `~31.4 ms`
  - `sym200`: `~66.0 ms`
- Because of that backlog, the `200e6` points should be treated as queue-saturation cases, not as stable low-latency operating points.

## Region Notes

- `50e6`
  - Stable.
  - Best point in the current sweep: `lat_sr50000000_fft1024_sym50` with `~0.689 ms` E2E.
  - `sym100` was rerun with the new retry logic. `attempt1` exited early with `AssertionError: buff_size > 1` in UHD/RFNoC deserialization, and `attempt2` completed successfully.

- `100e6`
  - `sym100` and `sym200` are the more representative operating points.
  - `sym50` is still abnormal because `bit_ms` is much larger than the other two symbol settings.

- `200e6`
  - Not queue-stable with the current CPU demodulator path and current queue sizing.
  - The pure processing path is still only a few milliseconds, but queue buildup makes the run-level behavior unacceptable.

## Recommended Takeaway

If the goal is to characterize CPU demodulator processing latency under non-saturated conditions, the most representative points from the current measurement set are:

- `50e6 / 1024 / sym50`: `~0.689 ms`
- `50e6 / 1024 / sym100`: `~1.081 ms`
- `50e6 / 1024 / sym200`: `~1.962 ms`
- `100e6 / 1024 / sym100`: `~2.654 ms`
- `100e6 / 1024 / sym200`: `~3.768 ms`

If the goal is to evaluate end-to-end receive behavior including backlog pressure, the `200e6` points must be discussed together with `rx_queue_ms`, not `e2e_ms` alone.
