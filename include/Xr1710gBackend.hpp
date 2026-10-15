#ifndef XR1710G_BACKEND_HPP
#define XR1710G_BACKEND_HPP

#include "RadioBackend.hpp"

namespace radio {

/** @brief Validate a preconfigured XR1710G source without accessing hardware. */
void validate_xr1710g_config(const Xr1710gConfig& cfg);

/** @brief Create a receive-only device reading w1700k-stream ABI v1.
 * The stream is mapped read-only. Start/stop controls this consumer only.
 * Timestamps are a local sample timeline, not hardware/PPS timestamps.
 */
IDevicePtr make_xr1710g_device(const DeviceConfig& cfg);

} // namespace radio
#endif // XR1710G_BACKEND_HPP
