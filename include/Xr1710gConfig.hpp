#ifndef XR1710G_CONFIG_HPP
#define XR1710G_CONFIG_HPP

#include <string>

namespace radio {

/** @brief Attach settings for a single-channel XR1710G host stream.
 * The external controller establishes the source. A configured PLL helper
 * permits fine LO correction through the existing frequency-tune interface.
 */
struct Xr1710gConfig {
    std::string shm_name = "/xr1710g-live";
    double sample_rate = 0.0;
    double center_freq = 0.0;
    double attach_timeout_s = 5.0;
    std::string pll_command = {};  ///< Empty: software NCO; otherwise trusted persistent PLL helper command.
};

} // namespace radio
#endif // XR1710G_CONFIG_HPP
