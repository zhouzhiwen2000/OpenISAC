import io
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xr1710g_pll_control import PllControl, serve


class Hardware:
    def __init__(self):
        self.words = [0x03205f51, 0xd55556, 0x100015, 0xa3010cff, 0]
        self.original = self.words[:]
        self.writes = []
        self.failure_after_int = False
        self.band = 2
        self.center = 6100000005
        self.closed = 0
        self.rf = types.SimpleNamespace(reg=lambda b,k,r:r, ar=lambda r:('r',r), aw=lambda r,v:('w',r,v))
        self.radio = types.SimpleNamespace(lo_divider=lambda b,h:(3,4,160000000) if h>=5926000000 else (5,8,160000000))

    def check(self, command):
        import json
        return 'boot\n'+json.dumps({'boot_id':'boot','loaded':True,'band':self.band,
                                   'bands':{'2':{'lo_hz':self.center,'sample_rate':320000000}}})

    def wm(self, unit):
        hardware = self
        class WM:
            def status(self): return {'version':6,'traps':0}
            def close(self): hardware.closed += 1
            def run(self, ops):
                result=[]
                for op in ops:
                    i = [0xa4b,0xa4a,0xa40,0xa01,0xca4].index(op[1])
                    if op[0]=='w':
                        hardware.writes.append(op)
                        hardware.words[i]=op[2] & (0xffffff if i==1 else 0xffffffff)
                        if hardware.failure_after_int and i==0:
                            hardware.failure_after_int=False
                            raise RuntimeError('interrupted write')
                    result.append(hardware.words[i])
                return result
        return WM()

    def control(self, center=None, rate=320000000):
        return PllControl(self, self.wm, self.rf, self.radio, self.center if center is None else center, rate)


class PllTests(unittest.TestCase):
    def test_total_quantized_offsets_and_zero_restore(self):
        h=Hardware(); c=h.control()
        try:
            self.assertAlmostEqual(c.set_offset(100),100.13580322265625)
            self.assertEqual(h.words[1],h.original[1]+14)
            count=len(h.writes)
            self.assertAlmostEqual(c.set_offset(100),100.13580322265625)
            self.assertEqual(len(h.writes),count)
            self.assertAlmostEqual(c.set_offset(-100),-100.13580322265625)
            self.assertEqual(h.words[1],h.original[1]-14)
            self.assertEqual(c.set_offset(0),0)
            self.assertEqual(h.words,h.original)
        finally:c.close()

    def test_invalid_offsets_never_write(self):
        h=Hardware(); c=h.control()
        try:
            for value in [float('nan'),float('inf'),100001,-100001]:
                with self.assertRaises(ValueError):c.set_offset(value)
            self.assertEqual(h.writes,[])
        finally:c.close()

    def test_reject_wrong_source_before_write(self):
        for center,rate in [(6100000105,320000000),(6100000005,40000000)]:
            h=Hardware()
            with self.assertRaises(RuntimeError):h.control(center,rate)
            self.assertEqual(h.writes,[])

    def test_partial_update_restores_original(self):
        h=Hardware()
        # Cross an integer-divider carry, so failure after INT leaves a real
        # mixed old/new divider that must be rolled back.
        h.words[1]=0xfffff8;h.original=h.words[:]
        h.center=round((((h.words[0]>>20&0x3ff)<<24)|h.words[1])*120000000/(1<<24))
        c=h.control();h.failure_after_int=True
        with self.assertRaisesRegex(RuntimeError,'interrupted'):c.set_offset(100)
        self.assertNotEqual(h.words[0],h.original[0])
        c.close()
        self.assertEqual(h.words,h.original)

    def test_external_retune_is_not_overwritten(self):
        h=Hardware();c=h.control();c.set_offset(100)
        h.words[1]+=2000
        changed=h.words[:]
        with self.assertRaisesRegex(RuntimeError,'outside'):c.set_offset(200)
        with self.assertRaisesRegex(RuntimeError,'external'):c.close()
        self.assertEqual(h.words,changed)

    def test_eof_restores_and_stale_requests_are_rejected(self):
        h=Hardware();out=io.StringIO()
        self.assertEqual(serve(h.control,io.StringIO('OPEN 1 6100000005 320000000\nSET 2 100\n'),out),0)
        self.assertEqual(h.words,h.original)
        self.assertIn('OK 2 100.13580322265625',out.getvalue())
        h=Hardware();out=io.StringIO()
        self.assertEqual(serve(h.control,io.StringIO('OPEN 1 6100000005 320000000\nSET 1 100\n'),out),1)
        self.assertIn('stale request',out.getvalue())
        self.assertEqual(h.writes,[])


if __name__=='__main__':unittest.main()
