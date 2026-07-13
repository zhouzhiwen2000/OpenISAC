# OCXO DAC to PPM Mapping

## 3.1 GHz Ref3 Calibration

| OCXO raw DAC word | Measured ppm |
|---:|---:|
| 0 | -0.4731 |
| 262144 | -0.2275 |
| 524288 | +0.0234 |
| 786431 | +0.2832 |
| 1048575 | +0.5441 |

## OCXO Set Commands

| Target ppm | OCXO raw DAC word | Command |
|---:|---:|---|
| -0.50 | 0 | `python3 scripts/set_ocxo.py --word 0` |
| -0.25 | 238128 | `python3 scripts/set_ocxo.py --word 238128` |
| 0.00 | 499839 | `python3 scripts/set_ocxo.py --word 499839` |
| +0.25 | 752932 | `python3 scripts/set_ocxo.py --word 752932` |
| +0.50 | 994619 | `python3 scripts/set_ocxo.py --word 994619` |

```bash
python3 scripts/set_ocxo.py --word 499839 --tty /dev/ttyUSB0
```
