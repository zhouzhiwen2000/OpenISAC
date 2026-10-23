#ifndef SENSING_FRAME_DECIMATOR_HPP
#define SENSING_FRAME_DECIMATOR_HPP

#include <algorithm>
#include <cstdint>

/** @brief Select complete CPIs by their original symbol time, consistently across channels.
 * The caller must retain continuous channel estimation/MTI before this gate.
 * A pending factor takes effect only for a CPI starting at/after its apply boundary.
 */
class SensingFrameDecimator {
public:
    /** @brief Schedule a factor (1 keeps all frames) at an absolute OFDM symbol. */
    void schedule(uint32_t factor, uint64_t apply_symbol) noexcept {
        _pending_factor = std::max(uint32_t{1}, factor);
        _apply_symbol = apply_symbol;
        _pending = true;
    }

    /** @brief Decide using the CPI start and nominal span, never an arrival counter. */
    bool keep(uint64_t first_symbol, uint64_t span) noexcept {
        if (_have_last && first_symbol < _last_symbol && _pending) {
            _apply_symbol = first_symbol; // Rebase a scheduled update after reacquisition.
        }
        _last_symbol = first_symbol;
        _have_last = true;
        if (_pending && first_symbol >= _apply_symbol) {
            _factor = _pending_factor;
            _pending = false;
        }
        return (first_symbol / std::max(uint64_t{1}, span)) % _factor == 0;
    }

private:
    uint32_t _factor = 1;
    uint32_t _pending_factor = 1;
    uint64_t _apply_symbol = 0;
    bool _pending = false;
    bool _have_last = false;
    uint64_t _last_symbol = 0;
};

#endif // SENSING_FRAME_DECIMATOR_HPP
