#ifndef SENSING_CHANNEL_HPP
#define SENSING_CHANNEL_HPP

#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/types/tune_request.hpp>
#include <fftw3.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "Common.hpp"
#include "OFDMCore.hpp"

struct SharedSensingRuntime {
    size_t sensing_symbol_stride = 20;
    bool enable_mti = true;
    bool skip_sensing_fft = true;
    uint64_t generation = 1;
    uint64_t apply_symbol_index = 0;
};

class SensingChannel {
public:
    using HeartbeatCallback = std::function<void(const std::string&, int)>;
    using CoreResolver = std::function<size_t(size_t)>;
    using BatchResetRequester = std::function<void()>;

    SensingChannel(
        const Config& cfg,
        const SensingRxChannelConfig& channel_cfg,
        const std::string& output_ip,
        int output_port,
        uint32_t logical_id,
        std::atomic<bool>& running_ref,
        std::shared_ptr<AggregatedSensingDataSender> aggregated_sender,
        std::shared_ptr<std::atomic<uint64_t>> batch_reset_symbol,
        BatchResetRequester batch_reset_requester,
        HeartbeatCallback heartbeat_sender,
        CoreResolver core_resolver
    );

    ~SensingChannel();

    void start(const uhd::time_spec_t& start_time);
    void stop();
    void join();
    void start_bistatic();
    void stop_bistatic();

    bool enqueue_tx_symbols(
        const std::shared_ptr<const SymbolVector>& symbols,
        uint64_t frame_start_symbol_index,
        uint64_t frame_seq
    );
    void process_bistatic_frame(const SensingFrame& frame, uint64_t frame_start_symbol_index);

    void set_target_alignment(int32_t samples);
    void set_alignment(int32_t samples);
    bool set_rx_gain(double requested_gain_db, double* applied_gain_db = nullptr);
    void apply_shared_cfg(const SharedSensingRuntime& snapshot);

    uint32_t logical_id() const;
    int32_t target_alignment() const;
    const SensingRxChannelConfig& channel_cfg() const;

    static void initialize_rx_hardware_and_sync(
        const Config& cfg,
        const uhd::tune_request_t& tune_req,
        const std::string& tx_device_args,
        const std::string& tx_clock_source,
        const std::string& tx_time_source,
        const uhd::usrp::multi_usrp::sptr& tx_usrp,
        std::vector<std::unique_ptr<SensingChannel>>& channels
    );

    SensingChannel(const SensingChannel&) = delete;
    SensingChannel& operator=(const SensingChannel&) = delete;
    SensingChannel(SensingChannel&&) = delete;
    SensingChannel& operator=(SensingChannel&&) = delete;

private:
    enum class RxState { ALIGNMENT, NORMAL };

    struct TxSymbolsFrame {
        std::shared_ptr<const SymbolVector> symbols;
        uint64_t frame_start_symbol_index = 0;
        uint64_t frame_seq = 0;
    };

    struct RxSymbolsFrame {
        AlignedVector samples;
        uint64_t frame_seq = 0;
    };

    class PairedFrameQueue;

    struct RxIoContext {
        uint32_t logical_id = 0;
        SensingRxChannelConfig channel_cfg;
        uhd::usrp::multi_usrp::sptr rx_usrp;
        uhd::rx_streamer::sptr rx_stream;
        uhd::time_spec_t stream_start_time{0.0};
        ObjectPool<AlignedVector> rx_frame_pool;
        std::unique_ptr<PairedFrameQueue> paired_queue;
        std::atomic<RxState> rx_state{RxState::ALIGNMENT};
        std::atomic<int32_t> target_alignment{0};
        std::atomic<int32_t> discard_samples{0};
        uint64_t next_rx_frame_seq = 0;
        std::thread rx_thread;

        RxIoContext(const Config& cfg, const SensingRxChannelConfig& c, uint32_t logical_id);
        ~RxIoContext();

        RxIoContext(const RxIoContext&) = delete;
        RxIoContext& operator=(const RxIoContext&) = delete;
        RxIoContext(RxIoContext&&) = delete;
        RxIoContext& operator=(RxIoContext&&) = delete;
    };

    struct SensingComputeContext {
        std::vector<AlignedVector> accumulated_rx_symbols;
        std::vector<AlignedVector> accumulated_tx_symbols;
        SensingMaskLayout sensing_mask_layout;
        CompactSensingMaskAnalysis compact_mask_analysis;
        AlignedVector compact_channel_output;
        std::vector<AlignedVector> compact_selected_tx_symbols;
        uint64_t next_symbol_to_sample = 0;
        uint64_t current_batch_first_symbol = 0;
        bool batch_has_first_symbol = false;
        bool bistatic_active = false;

        size_t active_stride = 20;
        bool active_enable_mti = true;
        bool active_skip_sensing_fft = true;
        bool delay_estimation_enabled = false;
        uint64_t next_delay_estimation_frame_seq = 0;
        bool sensing_pipeline_disabled_by_mode = false;
        bool compact_mask_local_delay_doppler_supported = false;
        uint64_t applied_generation = 0;
        std::unique_ptr<SyncProcessor> system_delay_sync;
        size_t system_delay_symbol_len = 0;
        size_t system_delay_expected_sync_pos = 0;

        AlignedVector demod_fft_in;
        AlignedVector demod_fft_out;
        fftwf_plan demod_fft_plan = nullptr;
        SensingProcessor sensing_core;
        std::unique_ptr<SensingProcessor> compact_sensing_core;
        SensingOutputDispatcher sensing_sender;
        AlignedFloatVector range_window;
        AlignedFloatVector compact_range_window;
        AlignedFloatVector doppler_window;
        AlignedFloatVector compact_doppler_window;
        std::vector<int> actual_subcarrier_indices;
        std::vector<int> compact_shifted_subcarrier_indices;
        std::vector<float> subcarrier_phases_unit_delay;
        std::chrono::steady_clock::time_point next_hb_time;
        double pending_batch_gather_us = 0.0;
        double prof_gather_total_us = 0.0;
        double prof_prep_total_us = 0.0;
        double prof_chest_shift_total_us = 0.0;
        double prof_mti_total_us = 0.0;
        double prof_windows_fft_total_us = 0.0;
        double prof_send_total_us = 0.0;
        uint64_t prof_batch_count = 0;
        std::thread sensing_thread;

        SensingComputeContext(
            const Config& cfg,
            const SensingRxChannelConfig& c,
            uint32_t logical_id,
            std::shared_ptr<AggregatedSensingDataSender> aggregated_sender,
            const std::string& output_ip,
            int output_port);
        ~SensingComputeContext();

        SensingComputeContext(const SensingComputeContext&) = delete;
        SensingComputeContext& operator=(const SensingComputeContext&) = delete;
        SensingComputeContext(SensingComputeContext&&) = delete;
        SensingComputeContext& operator=(SensingComputeContext&&) = delete;
    };

    size_t _resolve_core(size_t hint) const;
    void _rx_loop(const uhd::time_spec_t& start_time);
    void _handle_alignment();
    void _handle_normal_rx();
    void _sensing_loop();
    void _send_heartbeat_if_due(const std::chrono::steady_clock::time_point& now);
    void _estimate_system_delay(const AlignedVector& rx_frame_data, uint64_t frame_seq);
    void _process_compact_monostatic_frame(const AlignedVector& rx_frame_data, const TxSymbolsFrame& tx_frame);
    void _process_compact_bistatic_frame(const SensingFrame& frame, uint64_t frame_start_symbol_index);
    void _process_regular_compact_monostatic_frame(const AlignedVector& rx_frame_data, const TxSymbolsFrame& tx_frame);
    void _process_regular_compact_bistatic_frame(const SensingFrame& frame, uint64_t frame_start_symbol_index);
    void _process_regular_compact_buffer(
        const std::vector<AlignedVector>& tx_symbols,
        uint64_t frame_start_symbol_index
    );
    void _sensing_process(const SensingFrame& frame, uint64_t first_symbol_index, double gather_us);
    void _sensing_process_freq(const SensingFrame& frame, uint64_t first_symbol_index, double gather_us);
    void _sensing_process_finalize(
        const std::vector<AlignedVector>& tx_symbols,
        uint64_t first_symbol_index,
        double gather_us,
        double prep_us,
        size_t symbol_count
    );
    void _apply_shared_sensing_if_due(uint64_t symbol_index);
    void _request_shared_batch_reset();
    void _apply_batch_reset_if_due(uint64_t frame_start_symbol_index);

    const Config& _cfg;
    std::atomic<bool>& _running_ref;
    std::string _output_ip;
    int _output_port = 0;
    std::shared_ptr<std::atomic<uint64_t>> _shared_batch_reset_symbol;
    BatchResetRequester _batch_reset_requester;
    HeartbeatCallback _heartbeat_sender;
    CoreResolver _core_resolver;
    uint64_t _applied_batch_reset_symbol = 0;

    RxIoContext _rx_io;
    SensingComputeContext _compute;

    std::mutex _shared_cfg_mutex;
    SharedSensingRuntime _pending_shared_cfg;
    bool _has_pending_shared_cfg = false;
    std::atomic<bool> _stop_requested{false};
};

#endif
