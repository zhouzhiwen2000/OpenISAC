# XR1710G stream ABI

`w1700k_stream.h` is the unmodified MIT-licensed host ABI header from
[zhouzhiwen2000/xr1710g-sdr](https://github.com/zhouzhiwen2000/xr1710g-sdr/blob/fca9883b2a33dcc28514aa31e5aee557a028a4f6/host/stream/include/w1700k_stream.h),
commit `fca9883b2a33dcc28514aa31e5aee557a028a4f6`.

Only this interface header is included here. OpenISAC implements its own
read-only consumer, mapping validation and timeouts in `src/Xr1710gBackend.cpp`.
The external host service remains responsible for UDP reordering, unpacking
and optional filter correction. The interface describes one RX channel;
experimental dual-path interleaved captures must not be attached to this backend.
