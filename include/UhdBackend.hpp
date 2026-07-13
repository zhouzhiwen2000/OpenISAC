#ifndef UHD_BACKEND_HPP
#define UHD_BACKEND_HPP

// UHD radio backend — implements the radio:: HAL over a real USRP via UHD.
//
// This is the ONLY engine-side translation unit that includes <uhd/...>. It wraps
// uhd::usrp::multi_usrp + uhd::tx_streamer / uhd::rx_streamer and translates
// radio:: <-> uhd:: value types at the boundary. Concrete classes (UhdDevice,
// UhdTxStream, UhdRxStream) live in src/UhdBackend.cpp; the public surface is just
// make_device() / set_thread_priority() declared in RadioBackend.hpp.
//
// make_device() also owns a small per-process device registry so a device shared
// across roles (BS TX reused for uplink-RX / sensing-RX) is the same IDevice
// instance (pointer identity preserved), with a clock/time-source conflict check.

#include "RadioBackend.hpp"

#endif  // UHD_BACKEND_HPP
