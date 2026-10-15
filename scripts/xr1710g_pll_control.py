#!/usr/bin/env python3
"""Persistent PLL control helper for Xr1710gBackend (run on the IQ receiver host).

The backend supplies total offsets, never increments. No IQ samples are touched.
stdin/stdout: OPEN id center_hz sample_rate; SET id offset_hz; CLOSE id.
Replies: OK id actual_offset_hz step_hz, or ERR id explanation.
Only one RF controller may change the radio while this helper owns it.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys

MAX_OFFSET_HZ = 100_000.0
FRAC_MASK = 0xffffff
INT_MASK = 0x3ff00000


class PllControl:
    """Own a saved divider; set absolute corrections and restore on close."""
    def __init__(self, unit, wm_factory, rf, radio, center, rate):
        if not math.isfinite(center) or not math.isfinite(rate):
            raise ValueError('center and rate must be finite')
        self.unit, self.wm_factory, self.rf, self.radio = unit, wm_factory, rf, radio
        self.w = None
        self.saved = None
        self.possible = set()
        self.current = None
        self.offset = 0.0
        self.center, self.rate = center, rate
        self.boot, state = self._state()
        self.band = state['band']
        self._check_state(state)
        self.num, self.den, self.pfd = radio.lo_divider(self.band, round(center))
        self.step = 2 * self.pfd * self.num / self.den / (1 << 24)
        self.regs = [rf.reg(self.band, 0 if self.band == 0 else 4, r)
                     for r in (0xa4b, 0xa4a, 0xa40, 0xa01, 0xca4)]
        try:
            self.w = wm_factory(unit)
            self._check_wm()
            words = self._read()
            q = ((words[0] >> 20 & 0x3ff) << 24) | (words[1] & FRAC_MASK)
            nominal = q * self.pfd * self.num / self.den / (1 << 24)
            if q & 1 or abs(nominal - center) > .51 or words[4] & 3:
                raise RuntimeError('PLL divider/lock disagrees with configured source LO')
            self.base_q = q
            self.saved = words
            self.current = (words[0], words[1] & FRAC_MASK)
            self.possible.add(self.current)
        except BaseException:
            if self.w: self.w.close()
            raise

    def _state(self):
        text = self.unit.check('cat /proc/sys/kernel/random/boot_id; cat /tmp/w1700k-state.json')
        boot, _, raw = text.partition('\n')
        return boot.strip(), json.loads(raw)

    def _check_state(self, state):
        b = state['band']
        if (not state.get('loaded') or b != self.band or
                state['bands'][str(b)]['sample_rate'] != self.rate or
                abs(state['bands'][str(b)]['lo_hz'] - self.center) > .51 or
                state.get('boot_id') != self.boot):
            raise RuntimeError('source band, LO, rate or boot changed; reopen the backend')

    def _check_wm(self):
        status = self.w.status()
        if status['version'] != 6 or status['traps']:
            raise RuntimeError('PLL control requires healthy WM v6')

    def _read(self):
        return self.w.run([self.rf.ar(r) for r in self.regs])

    def _check_current(self):
        boot, state = self._state()
        if boot != self.boot: raise RuntimeError('router rebooted')
        self._check_state(state)
        self._check_wm()
        words = self._read()
        if (words[0], words[1] & FRAC_MASK) != self.current or words[2:4] != self.saved[2:4]:
            raise RuntimeError('PLL changed outside this backend')
        if words[4] & 3: raise RuntimeError('PLL is unlocked')

    def set_offset(self, hz):
        """Apply a total LO offset, positive for a higher LO; return quantized Hz."""
        if not math.isfinite(hz) or abs(hz) > MAX_OFFSET_HZ:
            raise ValueError('PLL correction must be finite and within +/-100000 Hz')
        self._check_current()
        delta = round(hz / self.step) * 2
        q = self.base_q + delta
        if not 0 <= q < (1 << 34): raise ValueError('PLL divider overflow')
        actual = delta * self.step / 2
        if self.radio.lo_divider(self.band, round(self.center + actual)) != (self.num, self.den, self.pfd):
            raise ValueError('correction crosses a PLL ratio boundary; use normal RF tuning')
        target = ((self.saved[0] & ~INT_MASK) | ((q >> 24) << 20), q & FRAC_MASK)
        if target == self.current: return self.offset
        # If a two-write transaction is interrupted, only our possible partial
        # states may be restored. Never overwrite an unrelated external retune.
        self.possible = {self.current, (target[0], self.current[1]), target}
        self.w.run([self.rf.aw(self.regs[0], target[0]),
                    self.rf.aw(self.regs[1], 0x80000000 | target[1])])
        words = self._read()
        self._check_wm()
        if (words[0], words[1] & FRAC_MASK) != target or words[2:4] != self.saved[2:4] or words[4] & 3:
            raise RuntimeError('PLL update verification failed')
        self.current, self.offset = target, actual
        self.possible = {target}
        return actual

    def close(self):
        """Restore the saved divider using a fresh mailbox connection."""
        if self.w: self.w.close(); self.w = None
        if self.saved is None: return
        try:
            boot, state = self._state()
            if boot != self.boot: raise RuntimeError('cannot restore PLL after router reboot')
            self._check_state(state)
            self.w = self.wm_factory(self.unit)
            self._check_wm()
            words = self._read()
            if (words[0], words[1] & FRAC_MASK) not in self.possible or words[2:4] != self.saved[2:4]:
                raise RuntimeError('external PLL change: refusing to overwrite it during restore')
            original = (self.saved[0], self.saved[1] & FRAC_MASK)
            if (words[0], words[1] & FRAC_MASK) != original:
                self.w.run([self.rf.aw(self.regs[0], original[0]),
                            self.rf.aw(self.regs[1], 0x80000000 | original[1])])
            words = self._read()
            if (words[0], words[1] & FRAC_MASK) != original or words[4] & 3:
                raise RuntimeError('PLL restoration failed')
            self.saved = None
        finally:
            if self.w: self.w.close(); self.w = None


def serve(factory, inp, out):
    control = None
    last_id = 0
    try:
        while True:
            line = inp.readline(1025)
            if not line: break
            request_id = 0
            try:
                fields = line.split()
                if len(line) > 1024 or len(fields) < 2: raise ValueError('malformed request')
                op, request_id = fields[0], int(fields[1])
                if request_id <= last_id: raise ValueError('stale request id')
                last_id = request_id
                if op == 'OPEN' and len(fields) == 4 and control is None:
                    control = factory(float(fields[2]), float(fields[3]))
                    actual = 0.0
                elif op == 'SET' and len(fields) == 3 and control is not None:
                    actual = control.set_offset(float(fields[2]))
                elif op == 'CLOSE' and len(fields) == 2 and control is not None:
                    control.close()
                    out.write(f'OK {request_id} 0 {control.step:.17g}\n'); out.flush()
                    break
                else: raise ValueError('invalid operation or session state')
                out.write(f'OK {request_id} {actual:.17g} {control.step:.17g}\n'); out.flush()
            except Exception as exc:
                # Restore before replying to failure: no delayed write is queued.
                if control:
                    try: control.close()
                    except Exception as restore: exc = RuntimeError(f'{exc}; restore failed: {restore}')
                    control = None
                out.write(f'ERR {request_id} {str(exc).replace(chr(10), " ")}\n'); out.flush()
                return 1
    finally:
        if control: control.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdr-repo', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    if not args.config.is_file(): parser.error('--config must exist')
    sys.path.insert(0, str(args.sdr_repo / 'host'))
    from w1700k.config import load
    from w1700k.unit import Unit
    from w1700k.wm import Wm
    from w1700k import rf, radio
    unit = Unit(load(args.config).unit)
    # Same-host backend owners serialize without closing the shared SSH master.
    lock_dir = Path(os.environ.get('XDG_RUNTIME_DIR', f'/tmp/xr1710g-{os.getuid()}'))
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = hashlib.sha256(unit.target.encode()).hexdigest()[:24]
    with open(lock_dir / f'xr1710g-pll-{name}.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        return serve(lambda center, rate: PllControl(unit, Wm, rf, radio, center, rate), sys.stdin, sys.stdout)


if __name__ == '__main__':
    try: sys.exit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f'XR1710G PLL: {exc}', file=sys.stderr)
        sys.exit(1)
