#ifndef ERTM_ABSOLUTE_FRAME_STATE_HPP
#define ERTM_ABSOLUTE_FRAME_STATE_HPP

/**
 * @file ErtmAbsoluteFrameState.hpp
 * @brief Shared host-side absolute-frame state for CPU/CUDA eRTM processing.
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "OFDMCore.hpp"

/**
 * @brief Own eRTM report queue/cache, clear generations and TO publication.
 *
 * The decode thread is the sole queue producer. The sensing/process path is
 * the sole queue consumer and cache owner. Other threads may only request a
 * clear or read a mutex-consistent published TO snapshot.
 */
class ErtmAbsoluteFrameState {
public:
    static constexpr size_t kReportCacheCapacity = 8;

    struct QueuedPayload {
        uint64_t clear_generation = 0;
        std::vector<uint8_t> raw;
    };

    struct CachedReport {
        ErtmTimingPayloadView view;
        bool valid = false;
    };

    struct ClearResult {
        bool applied = false;
        uint64_t generation = 0;
        size_t invalidated_reports = 0;
    };

    enum class IngestStatus {
        Accepted,
        Empty,
        LegacyPayload,
        InvalidPayload,
        MissingEpoch,
        CacheRejected,
    };

    struct IngestResult {
        IngestStatus status = IngestStatus::Empty;
        std::string error;
        uint32_t payload_seq = 0;

        bool accepted() const { return status == IngestStatus::Accepted; }
    };

    struct PublishSnapshot {
        bool available = false;
        double to_ue_samples = 0.0;
        bool pps_phase_available = false;
        double pps_phase_error_samples = 0.0;
        double r_bs_samples = 0.0;
        double r_ue_samples = 0.0;
        int64_t bs_last_pps_ticks = -1;
        int64_t bs_t0_to_pps_ticks = -1;
        int64_t ue_last_pps_ns = -1;
        double bs_tick_rate = 0.0;
        double sample_rate = 0.0;
        int32_t rx_alignment_shift_samples = 0;
        AirFrameIdentity dl_identity;
        uint64_t sync_generation = 0;
        uint64_t pub_gen = 0;
        uint64_t clear_gen = 0;
        uint64_t request_gen = 0;
    };

    struct PublishResult {
        bool accepted = false;
        bool had_previous = false;
        double previous_to_ue_samples = 0.0;
        int32_t previous_rx_alignment_shift_samples = 0;
        uint64_t request_generation = 0;
        uint64_t applied_generation = 0;
        uint64_t publication_generation = 0;
    };

    struct StateMSelection {
        CachedReport* report = nullptr;
        UlTxTimingStateEntry state_m;
        size_t future_count = 0;
        size_t history_miss_count = 0;
    };

    ErtmAbsoluteFrameState() = default;
    ErtmAbsoluteFrameState(const ErtmAbsoluteFrameState&) = delete;
    ErtmAbsoluteFrameState& operator=(const ErtmAbsoluteFrameState&) = delete;

    void prepare_channel_buffers(size_t fft_size) {
        for (auto& slot : _cache) {
            slot.view.channel_freq.resize(fft_size);
            slot.valid = false;
        }
        _cache_count = 0;
    }

    bool enqueue(std::vector<uint8_t>&& payload) {
        QueuedPayload queued;
        queued.clear_generation =
            _clear_request_generation.load(std::memory_order_acquire);
        queued.raw = std::move(payload);
        return _payload_queue.try_push(std::move(queued));
    }

    bool try_pop(QueuedPayload& payload) {
        return _payload_queue.try_pop(payload);
    }

    /** Immediately invalidate TO and publish a new queue/cache generation. */
    uint64_t request_clear() {
        std::lock_guard<std::mutex> lock(_publish_mutex);
        const uint64_t generation =
            _clear_request_generation.fetch_add(
                1, std::memory_order_acq_rel) +
            1;
        _published_available = false;
        return generation;
    }

    /** Consumer-only: apply a requested cache clear while retaining capacity. */
    ClearResult apply_pending_clear() {
        ClearResult result;
        const uint64_t requested =
            _clear_request_generation.load(std::memory_order_acquire);
        if (requested == _clear_applied_generation) {
            return result;
        }
        for (auto& slot : _cache) {
            result.invalidated_reports += slot.valid ? 1u : 0u;
            slot.valid = false;
        }
        _cache_count = 0;
        _last_used_measure_epoch = 0;
        _last_used_measure_frame = std::numeric_limits<uint64_t>::max();
        _clear_applied_generation = requested;
        {
            std::lock_guard<std::mutex> lock(_publish_mutex);
            _published_clear_generation = requested;
            _published_available = false;
        }
        result.applied = true;
        result.generation = requested;
        return result;
    }

    uint64_t applied_clear_generation() const {
        return _clear_applied_generation;
    }

    size_t cache_size() const { return _cache_count; }

    IngestResult ingest(
        std::vector<uint8_t>&& payload,
        const AirFrameIdentity& dl_identity,
        uint32_t expected_fft_size)
    {
        IngestResult result;
        if (payload.empty()) {
            result.status = IngestStatus::Empty;
            return result;
        }
        if (ErtmTimingPayload::looks_like_v2(payload.data(), payload.size()) &&
            !ErtmTimingPayload::looks_like(payload.data(), payload.size())) {
            result.status = IngestStatus::LegacyPayload;
            return result;
        }
        if (ErtmTimingPayload::looks_like_v3(payload.data(), payload.size()) &&
            !ErtmTimingPayload::looks_like(payload.data(), payload.size())) {
            result.status = IngestStatus::LegacyPayload;
            return result;
        }
        if (ErtmTimingPayload::looks_like_v4(payload.data(), payload.size()) &&
            !ErtmTimingPayload::looks_like(payload.data(), payload.size())) {
            result.status = IngestStatus::LegacyPayload;
            return result;
        }

        ErtmTimingPayloadView header;
        if (!ErtmTimingPayload::peek(
                payload.data(),
                payload.size(),
                expected_fft_size,
                header,
                &result.error)) {
            result.status = IngestStatus::InvalidPayload;
            return result;
        }
        result.payload_seq = header.seq;
        if (header.timeline_epoch == 0) {
            result.status = IngestStatus::MissingEpoch;
            return result;
        }

        const size_t replace_idx = choose_replacement(header, dl_identity);
        if (replace_idx == kReportCacheCapacity) {
            result.status = IngestStatus::CacheRejected;
            return result;
        }

        CachedReport& slot = _cache[replace_idx];
        if (!ErtmTimingPayload::unpack(
                payload.data(),
                payload.size(),
                expected_fft_size,
                slot.view,
                &result.error)) {
            slot.valid = false;
            result.status = IngestStatus::InvalidPayload;
            return result;
        }
        slot.valid = true;
        result.status = IngestStatus::Accepted;
        return result;
    }

    CachedReport* select_for_downlink(
        const AirFrameIdentity& dl_identity,
        size_t& future_count)
    {
        CachedReport* selected = nullptr;
        future_count = 0;
        if (!dl_identity.valid || dl_identity.timeline_epoch == 0) {
            return nullptr;
        }
        const uint32_t epoch =
            AirFrameClock::ota_epoch32(dl_identity.timeline_epoch);
        for (size_t i = 0; i < _cache_count; ++i) {
            CachedReport& slot = _cache[i];
            if (!slot.valid || slot.view.timeline_epoch != epoch) {
                continue;
            }
            if (slot.view.ul_measure_frame_seq > dl_identity.frame_seq) {
                ++future_count;
                continue;
            }
            if (selected == nullptr ||
                slot.view.ul_measure_frame_seq >
                    selected->view.ul_measure_frame_seq) {
                selected = &slot;
            }
        }
        return selected;
    }

    /**
     * @brief Select newest usable report M and its historical UL-TX state(M).
     *
     * A cache entry whose timing history has already rolled out is invalidated,
     * then the next older M is tried. Query and diagnostic policy stay in the
     * caller through the two lightweight callbacks.
     */
    template <typename QueryStateM, typename OnHistoryMiss>
    StateMSelection select_with_state_m(
        const AirFrameIdentity& dl_identity,
        QueryStateM&& query_state_m,
        OnHistoryMiss&& on_history_miss)
    {
        StateMSelection selection;
        while ((selection.report = select_for_downlink(
                    dl_identity, selection.future_count)) != nullptr) {
            if (query_state_m(
                    selection.report->view.timeline_epoch,
                    selection.report->view.ul_measure_frame_seq,
                    selection.state_m)) {
                return selection;
            }
            on_history_miss(selection.report->view);
            invalidate(*selection.report);
            selection.report = nullptr;
            ++selection.history_miss_count;
        }
        return selection;
    }

    void invalidate(CachedReport& report) {
        report.valid = false;
        // Retain channel_freq capacity for the next unpack into this slot.
    }

    bool is_unused(const CachedReport& report) const {
        return report.view.timeline_epoch != _last_used_measure_epoch ||
            report.view.ul_measure_frame_seq != _last_used_measure_frame;
    }

    /** Consumer-only: freeze the clear generation used by a TO computation. */
    bool begin_compute(uint64_t& clear_generation) const {
        clear_generation =
            _clear_request_generation.load(std::memory_order_acquire);
        return clear_generation == _clear_applied_generation;
    }

    PublishResult publish(
        uint64_t compute_clear_generation,
        double to_ue_samples,
        uint32_t measure_epoch,
        uint64_t measure_frame,
        const AirFrameIdentity& dl_identity,
        int32_t rx_alignment_shift_samples,
        uint64_t sync_generation,
        bool pps_phase_available = false,
        double pps_phase_error_samples = 0.0,
        double r_bs_samples = 0.0,
        double r_ue_samples = 0.0,
        int64_t bs_last_pps_ticks = -1,
        int64_t bs_t0_to_pps_ticks = -1,
        int64_t ue_last_pps_ns = -1,
        double bs_tick_rate = 0.0,
        double sample_rate = 0.0)
    {
        std::lock_guard<std::mutex> lock(_publish_mutex);
        PublishResult result;
        result.request_generation =
            _clear_request_generation.load(std::memory_order_acquire);
        result.applied_generation = _clear_applied_generation;
        if (result.request_generation != compute_clear_generation ||
            compute_clear_generation != _clear_applied_generation) {
            return result;
        }

        result.had_previous = _published_available;
        result.previous_to_ue_samples = _published_to_ue_samples;
        result.previous_rx_alignment_shift_samples =
            _published_rx_alignment_shift_samples;
        _published_to_ue_samples = to_ue_samples;
        _published_pps_phase_available = pps_phase_available;
        _published_pps_phase_error_samples = pps_phase_error_samples;
        _published_r_bs_samples = r_bs_samples;
        _published_r_ue_samples = r_ue_samples;
        _published_bs_last_pps_ticks = bs_last_pps_ticks;
        _published_bs_t0_to_pps_ticks = bs_t0_to_pps_ticks;
        _published_ue_last_pps_ns = ue_last_pps_ns;
        _published_bs_tick_rate = bs_tick_rate;
        _published_sample_rate = sample_rate;
        _published_rx_alignment_shift_samples = rx_alignment_shift_samples;
        _published_dl_identity = dl_identity;
        _published_sync_generation = sync_generation;
        _published_clear_generation = compute_clear_generation;
        _published_publication_generation = ++_publication_counter;
        _published_available = true;
        _last_used_measure_epoch = measure_epoch;
        _last_used_measure_frame = measure_frame;
        result.publication_generation = _published_publication_generation;
        result.accepted = true;
        return result;
    }

    void invalidate_to() {
        std::lock_guard<std::mutex> lock(_publish_mutex);
        _published_available = false;
    }

    PublishSnapshot snapshot() const {
        std::lock_guard<std::mutex> lock(_publish_mutex);
        PublishSnapshot snapshot;
        snapshot.available = _published_available;
        snapshot.to_ue_samples = _published_to_ue_samples;
        snapshot.pps_phase_available =
            _published_available && _published_pps_phase_available;
        snapshot.pps_phase_error_samples = _published_pps_phase_error_samples;
        snapshot.r_bs_samples = _published_r_bs_samples;
        snapshot.r_ue_samples = _published_r_ue_samples;
        snapshot.bs_last_pps_ticks = _published_bs_last_pps_ticks;
        snapshot.bs_t0_to_pps_ticks = _published_bs_t0_to_pps_ticks;
        snapshot.ue_last_pps_ns = _published_ue_last_pps_ns;
        snapshot.bs_tick_rate = _published_bs_tick_rate;
        snapshot.sample_rate = _published_sample_rate;
        snapshot.rx_alignment_shift_samples =
            _published_rx_alignment_shift_samples;
        snapshot.dl_identity = _published_dl_identity;
        snapshot.sync_generation = _published_sync_generation;
        snapshot.pub_gen =
            _published_publication_generation;
        snapshot.clear_gen = _published_clear_generation;
        snapshot.request_gen =
            _clear_request_generation.load(std::memory_order_relaxed);
        return snapshot;
    }

private:
    size_t choose_replacement(
        const ErtmTimingPayloadView& header,
        const AirFrameIdentity& dl_identity)
    {
        size_t first_invalid = kReportCacheCapacity;
        for (size_t i = 0; i < _cache_count; ++i) {
            CachedReport& slot = _cache[i];
            if (!slot.valid) {
                if (first_invalid == kReportCacheCapacity) {
                    first_invalid = i;
                }
                continue;
            }
            if (slot.view.timeline_epoch == header.timeline_epoch &&
                slot.view.ul_measure_frame_seq ==
                    header.ul_measure_frame_seq) {
                return i;
            }
        }
        if (first_invalid != kReportCacheCapacity) {
            return first_invalid;
        }
        if (_cache_count < kReportCacheCapacity) {
            return _cache_count++;
        }

        uint32_t current_epoch = 0;
        uint64_t current_frame = 0;
        if (dl_identity.valid && dl_identity.timeline_epoch != 0) {
            current_epoch =
                AirFrameClock::ota_epoch32(dl_identity.timeline_epoch);
            current_frame = dl_identity.frame_seq;
        }

        size_t protected_idx = kReportCacheCapacity;
        if (current_epoch != 0) {
            for (size_t i = 0; i < kReportCacheCapacity; ++i) {
                const CachedReport& cached = _cache[i];
                if (!cached.valid ||
                    cached.view.timeline_epoch != current_epoch ||
                    cached.view.ul_measure_frame_seq > current_frame) {
                    continue;
                }
                if (protected_idx == kReportCacheCapacity ||
                    cached.view.ul_measure_frame_seq >
                        _cache[protected_idx].view.ul_measure_frame_seq) {
                    protected_idx = i;
                }
            }
        }

        size_t replacement = kReportCacheCapacity;
        if (current_epoch != 0) {
            for (size_t i = 0; i < kReportCacheCapacity; ++i) {
                if (_cache[i].valid &&
                    _cache[i].view.timeline_epoch != current_epoch) {
                    replacement = i;
                    break;
                }
            }
        }

        if (replacement == kReportCacheCapacity) {
            for (size_t i = 0; i < kReportCacheCapacity; ++i) {
                const CachedReport& cached = _cache[i];
                if (!cached.valid || i == protected_idx ||
                    current_epoch == 0 ||
                    cached.view.timeline_epoch != current_epoch ||
                    cached.view.ul_measure_frame_seq > current_frame) {
                    continue;
                }
                if (replacement == kReportCacheCapacity ||
                    cached.view.ul_measure_frame_seq <
                        _cache[replacement].view.ul_measure_frame_seq) {
                    replacement = i;
                }
            }
        }

        if (replacement == kReportCacheCapacity) {
            for (size_t i = 0; i < kReportCacheCapacity; ++i) {
                const CachedReport& cached = _cache[i];
                if (!cached.valid || current_epoch == 0 ||
                    cached.view.timeline_epoch != current_epoch ||
                    cached.view.ul_measure_frame_seq <= current_frame) {
                    continue;
                }
                if (replacement == kReportCacheCapacity ||
                    cached.view.ul_measure_frame_seq >
                        _cache[replacement].view.ul_measure_frame_seq) {
                    replacement = i;
                }
            }
        }

        if (replacement == kReportCacheCapacity && current_epoch == 0) {
            for (size_t i = 0; i < kReportCacheCapacity; ++i) {
                if (replacement == kReportCacheCapacity ||
                    _cache[i].view.ul_measure_frame_seq <
                        _cache[replacement].view.ul_measure_frame_seq) {
                    replacement = i;
                }
            }
        }
        if (replacement == kReportCacheCapacity) {
            return replacement;
        }

        const ErtmTimingPayloadView& victim = _cache[replacement].view;
        const bool incoming_is_future =
            current_epoch != 0 && header.timeline_epoch == current_epoch &&
            header.ul_measure_frame_seq > current_frame;
        const bool victim_is_future =
            current_epoch != 0 && victim.timeline_epoch == current_epoch &&
            victim.ul_measure_frame_seq > current_frame;
        if (incoming_is_future && victim_is_future &&
            header.ul_measure_frame_seq >= victim.ul_measure_frame_seq) {
            return kReportCacheCapacity;
        }
        return replacement;
    }

    SPSCRingBuffer<QueuedPayload> _payload_queue{8};
    std::array<CachedReport, kReportCacheCapacity> _cache{};
    size_t _cache_count = 0;
    std::atomic<uint64_t> _clear_request_generation{0};
    uint64_t _clear_applied_generation = 0;  // consumer-only
    uint32_t _last_used_measure_epoch = 0;   // consumer-only
    uint64_t _last_used_measure_frame =
        std::numeric_limits<uint64_t>::max();

    mutable std::mutex _publish_mutex;
    bool _published_available = false;
    double _published_to_ue_samples = 0.0;
    bool _published_pps_phase_available = false;
    double _published_pps_phase_error_samples = 0.0;
    double _published_r_bs_samples = 0.0;
    double _published_r_ue_samples = 0.0;
    int64_t _published_bs_last_pps_ticks = -1;
    int64_t _published_bs_t0_to_pps_ticks = -1;
    int64_t _published_ue_last_pps_ns = -1;
    double _published_bs_tick_rate = 0.0;
    double _published_sample_rate = 0.0;
    int32_t _published_rx_alignment_shift_samples = 0;
    AirFrameIdentity _published_dl_identity;
    uint64_t _published_sync_generation = 0;
    uint64_t _publication_counter = 0;
    uint64_t _published_publication_generation = 0;
    uint64_t _published_clear_generation = 0;
};

#endif // ERTM_ABSOLUTE_FRAME_STATE_HPP
