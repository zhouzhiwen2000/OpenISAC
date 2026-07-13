/* OpenISAC Config Console — application script (extracted from config_web_editor.html). */
const APP = window.__APP_STATE__;

    const tabs = Array.from(document.querySelectorAll('.tab'));
    const flash = document.getElementById('flash');
    const plannerFlash = document.getElementById('plannerFlash');
    const saveBtn = document.getElementById('saveBtn');
    const reloadBtn = document.getElementById('reloadBtn');
    const presetSelect = document.getElementById('presetSelect');
    const commandInput = document.getElementById('commandInput');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const resetIsolationBtn = document.getElementById('resetIsolationBtn');
    const refreshBtn = document.getElementById('refreshBtn');
    const configPathLabel = document.getElementById('configPathLabel');
    const buildDirLabel = document.getElementById('buildDirLabel');
    const editorTitle = document.getElementById('editorTitle');
    const runtimeTitle = document.getElementById('runtimeTitle');
    const mtimeLabel = document.getElementById('mtimeLabel');
    const fileStateLabel = document.getElementById('fileStateLabel');
    const processMeta = document.getElementById('processMeta');
    const statusPill = document.getElementById('statusPill');
    const statusText = document.getElementById('statusText');
    const logBox = document.getElementById('logBox');
    const configSections = document.getElementById('configSections');
    const isolateToggle = document.getElementById('isolateToggle');
    const overrideIsolateToggle = document.getElementById('overrideIsolateToggle');
    const overrideIsolateInput = document.getElementById('overrideIsolateInput');
    const sudoPasswordInput = document.getElementById('sudoPasswordInput');
	    const configRuntimeView = document.getElementById('configRuntimeView');
	    const plannerView = document.getElementById('plannerView');
	    const plannerTitle = document.getElementById('plannerTitle');
	    const plannerIntro = document.getElementById('plannerIntro');
	    const plannerSummary = document.getElementById('plannerSummary');
	    const plannerHost = document.getElementById('plannerHost');
	    const loadPlannerTxBtn = document.getElementById('loadPlannerTxBtn');
	    const loadPlannerRxBtn = document.getElementById('loadPlannerRxBtn');
	    const applyPlannerTxBtn = document.getElementById('applyPlannerTxBtn');
	    const applyPlannerRxBtn = document.getElementById('applyPlannerRxBtn');
	    const plannerTargetNote = document.getElementById('plannerTargetNote');

	    let currentTab = 'bs';
	    const cache = {};
	    const specialPlannerTabs = new Set(['planner', 'sensingPlanner']);

    function setFlash(message, kind = '') {
      for (const node of [flash, plannerFlash]) {
        node.textContent = message || '';
        node.className = 'flash' + (kind ? ' ' + kind : '');
      }
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
      });
      const text = await response.text();
      let payload = {};
      try {
        payload = text ? JSON.parse(text) : {};
      } catch {
        payload = { ok: false, error: text || 'Invalid JSON response' };
      }
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || response.statusText || 'Request failed');
      }
      return payload;
    }

	    function modelForTab(tabName) {
	      return cache[tabName]?.config || null;
	    }

	    function plannerSpec(tabName = currentTab) {
	      if (tabName === 'sensingPlanner') {
	        return {
	          cacheKey: 'sensingPlanner',
	          fieldKey: 'sensing_mask_blocks',
		          title: 'Sensing Resource Map',
	          intro: 'Plan sensing time-frequency resources on a large canvas, choose whether to load from the BS or UE config, then apply back to either side.',
		          summaryTitle: 'Sensing Resource Map',
		          loadedFromLabel: 'current sensing resource map tab',
	          metricLabel: 'sensing RE',
	          missingMode: 'disabled',
	          note: 'Selections are snapped to integer RE cells on drag release. STRD-based mode writes `sensing_output_mode=dense`. Custom Blocks writes `sensing_output_mode=compact_mask` plus `sensing_mask_blocks`. If every selected symbol uses the same subcarrier set and symbols are equally spaced across the frame ring, compact sensing can enable MTI and local Delay-Doppler; in that mode `range_fft_size` must cover the selected subcarrier count and `doppler_fft_size` must cover the selected symbol count. Load pulls `sensing_mask_blocks` from BS.yaml or UE.yaml. Apply writes the current sensing resource plan back to either file.',
	        };
	      }
	      return {
	        cacheKey: 'planner',
	        fieldKey: 'data_resource_blocks',
		        title: 'Resource Map',
	        intro: 'Plan `data_resource_blocks` on a large time-frequency canvas, choose whether each block carries payload or sensing pilot, then load/apply the result to the BS or UE config.',
		        summaryTitle: 'Resource Map',
		        loadedFromLabel: 'current resource map tab',
	        metricLabel: 'payload RE',
	        missingMode: 'legacy',
	        note: 'Selections are snapped to integer RE cells on drag release. Each `data_resource_blocks` entry can be `payload` or `sensing_pilot`. Sensing-pilot RE transmit the sync-symbol sequence on the same subcarriers, but are excluded from payload extraction and payload modulation. Load pulls `data_resource_blocks` from BS.yaml or UE.yaml. Apply writes the current planner state back to either file.',
	      };
	    }

	    function updateTabVisuals() {
	      for (const tabButton of tabs) {
	        const isActive = tabButton.dataset.target === currentTab;
	        tabButton.classList.toggle('active', isActive);
	        tabButton.setAttribute('aria-selected', isActive ? 'true' : 'false');
	      }
	      const plannerModeActive = specialPlannerTabs.has(currentTab);
	      configRuntimeView.classList.toggle('hidden', plannerModeActive);
	      plannerView.classList.toggle('hidden', !plannerModeActive);
	      if (plannerModeActive) {
	        buildDirLabel.textContent = APP.build_dir;
	        return;
      }
      const label = APP.tabs[currentTab].label;
      editorTitle.textContent = label + ' Config';
      runtimeTitle.textContent = label + ' Runtime';
      buildDirLabel.textContent = APP.build_dir;
      updateRuntimeOptionControls();
    }

    function populatePresets(tab) {
      presetSelect.innerHTML = '';
      for (const preset of APP.tabs[tab].presets) {
        const option = document.createElement('option');
        option.value = preset.command;
        option.textContent = preset.label;
        presetSelect.appendChild(option);
      }
      const draftCommand = cache[tab]?.draftCommand || APP.tabs[tab].default_command;
      const matched = APP.tabs[tab].presets.find((preset) => preset.command === draftCommand);
      presetSelect.value = matched ? matched.command : APP.tabs[tab].presets[0].command;
    }

    function parseIntOrNull(text) {
      if (text === null || text === undefined) return null;
      const trimmed = String(text).trim();
      if (!trimmed) return null;
      const value = Number.parseInt(trimmed, 10);
      return Number.isFinite(value) ? value : null;
    }

    function currentModel() {
      return modelForTab(currentTab);
    }

    function radioBackendValue(model = currentModel()) {
      const field = findField(model, 'radio_backend');
      return String(field?.value_text ?? field?.value ?? '').trim().toLowerCase();
    }

    const SIM_HARDWARE_SECTION_TITLES = new Set([
      'USRP Device Args',
      'Clock / Time Sources',
      'Wire Format',
      'USRP Device / Link',
    ]);

    const SIM_HARDWARE_FIELD_KEYS = new Set([
      'device_args',
      'clock_source',
      'time_source',
      'wire_format_tx',
      'uplink_rx_wire_format',
      'sensing_rx_wire_format',
      'downlink_rx_wire_format',
      'tx_gain',
      'tx_channel',
      'tx_device_args',
      'tx_clock_source',
      'tx_time_source',
      'rx_gain',
      'rx_device_args',
      'rx_clock_source',
      'rx_time_source',
      'rx_channel',
      'uplink_rx_channel',
      'uplink_rx_device_args',
      'uplink_rx_clock_source',
      'uplink_rx_time_source',
      'rx_agc_enable',
      'rx_agc_low_threshold_db',
      'rx_agc_high_threshold_db',
      'rx_agc_max_step_db',
      'rx_agc_update_frames',
      'hardware_sync',
      'hardware_sync_tty',
      'ocxo_pi_kp_fast',
      'ocxo_pi_ki_fast',
      'ocxo_pi_kp_slow',
      'ocxo_pi_ki_slow',
      'ocxo_pi_switch_abs_error_ppm',
      'ocxo_pi_switch_hold_s',
      'ocxo_pi_max_step_fast_ppm',
      'ocxo_pi_max_step_slow_ppm',
      'akf_enable',
      'akf_bootstrap_frames',
      'akf_innovation_window',
      'akf_max_lag',
      'akf_adapt_interval',
      'akf_gate_sigma',
      'akf_tikhonov_lambda',
      'akf_update_smooth',
      'akf_q_wf_min',
      'akf_q_wf_max',
      'akf_q_rw_min',
      'akf_q_rw_max',
      'akf_r_min',
      'akf_r_max',
    ]);

    const SIM_HARDWARE_MAPPING_ITEM_KEYS = new Set([
      'device_args',
      'clock_source',
      'time_source',
      'wire_format',
      'rx_gain',
      'rx_antenna',
    ]);

    function shouldHideFieldInSim(section, field, showSimulation) {
      if (!showSimulation) return false;
      if (field.type === 'simulation_mapping') return false;
      if (SIM_HARDWARE_SECTION_TITLES.has(section.title)) return true;
      return SIM_HARDWARE_FIELD_KEYS.has(field.key);
    }

    function visibleMappingItemFields(field, showSimulation) {
      const itemFields = Array.isArray(field.item_fields) ? field.item_fields : [];
      if (!showSimulation) return itemFields;
      if (field.key !== 'sensing_rx_channels') return itemFields;
      return itemFields.filter((sub) => !SIM_HARDWARE_MAPPING_ITEM_KEYS.has(sub.key));
    }

    // self_channel_* live in Network but stay gated by the uplink
    // mapping's debug_self_channel checkbox (and enable_uplink/TDD).
    function uplinkSelfChannelVisible(model) {
      if (!fieldBoolValue(model, 'enable_uplink', false)) return false;
      if (duplexModeValue(model) === 'fdd') return false;
      const uplinkField = findField(model, 'uplink');
      return uplinkField ? structuredBoolValue(uplinkField, 'debug_self_channel', false) : false;
    }

    function duplexModeValue(model = currentModel()) {
      const raw = String(findField(model, 'duplex_mode')?.value_text ?? 'tdd').trim().toLowerCase();
      return raw === 'fdd' ? 'fdd' : 'tdd';
    }

    const DEPENDENT_FIELD_RULES = {
      cfo_training_period_samples: (model) => fieldBoolValue(model, 'enable_cfo_training_sequence', false),
      duplex_mode: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink_idle_waveform: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink: (model) => fieldBoolValue(model, 'enable_uplink', false),
      self_channel_ip: uplinkSelfChannelVisible,
      self_channel_port: uplinkSelfChannelVisible,
      self_pdf_ip: uplinkSelfChannelVisible,
      self_pdf_port: uplinkSelfChannelVisible,
      equalizer_mode: (model) => !fieldIsInSection(model, 'equalizer_mode', 'Uplink') || fieldBoolValue(model, 'enable_uplink', false),
      channel_tracking_mode: (model) => !fieldIsInSection(model, 'channel_tracking_mode', 'Uplink') || fieldBoolValue(model, 'enable_uplink', false),
      equalizer_mag_floor: (model) => !fieldIsInSection(model, 'equalizer_mag_floor', 'Uplink') || fieldBoolValue(model, 'enable_uplink', false),
      channel_tracking_min_pilot_snr: (model) => !fieldIsInSection(model, 'channel_tracking_min_pilot_snr', 'Uplink') || fieldBoolValue(model, 'enable_uplink', false),
      rx_gain: (model) => !fieldIsInSection(model, 'rx_gain', 'Uplink') || fieldBoolValue(model, 'enable_uplink', false),
      tx_gain: (model) => !fieldIsInSection(model, 'tx_gain', 'Uplink') || fieldBoolValue(model, 'enable_uplink', false),
      tx_channel: (model) => !fieldIsInSection(model, 'tx_channel', 'Uplink') || fieldBoolValue(model, 'enable_uplink', false),
      wire_format_tx: (model) => !fieldIsInSection(model, 'wire_format_tx', 'Uplink') || fieldBoolValue(model, 'enable_uplink', false),
      uplink_rx_channel: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink_rx_wire_format: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink_rx_device_args: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink_rx_clock_source: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink_rx_time_source: (model) => fieldBoolValue(model, 'enable_uplink', false),
      bs_dl_ul_timing_diff: (model) => fieldBoolValue(model, 'enable_uplink', false),
      ue_timing_advance: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink_cpu_cores: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink_channel_ip: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink_channel_port: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink_pdf_ip: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink_pdf_port: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink_constellation_ip: (model) => fieldBoolValue(model, 'enable_uplink', false),
      uplink_constellation_port: (model) => fieldBoolValue(model, 'enable_uplink', false),
      measurement_mode: (model) => fieldBoolValue(model, 'measurement_enable', false),
      measurement_run_id: (model) => fieldBoolValue(model, 'measurement_enable', false),
      measurement_output_dir: (model) => fieldBoolValue(model, 'measurement_enable', false),
      measurement_payload_bytes: (model) => fieldBoolValue(model, 'measurement_enable', false),
      measurement_prbs_seed: (model) => fieldBoolValue(model, 'measurement_enable', false),
      measurement_packets_per_point: (model) => fieldBoolValue(model, 'measurement_enable', false),
      measurement_max_packets_per_frame: (model) => fieldBoolValue(model, 'measurement_enable', false),
      rx_agc_low_threshold_db: (model) => fieldBoolValue(model, 'rx_agc_enable', false),
      rx_agc_high_threshold_db: (model) => fieldBoolValue(model, 'rx_agc_enable', false),
      rx_agc_max_step_db: (model) => fieldBoolValue(model, 'rx_agc_enable', false),
      rx_agc_update_frames: (model) => fieldBoolValue(model, 'rx_agc_enable', false),
      hardware_sync_tty: (model) => fieldBoolValue(model, 'hardware_sync', false),
      ocxo_pi_kp_fast: (model) => fieldBoolValue(model, 'hardware_sync', false),
      ocxo_pi_ki_fast: (model) => fieldBoolValue(model, 'hardware_sync', false),
      ocxo_pi_kp_slow: (model) => fieldBoolValue(model, 'hardware_sync', false),
      ocxo_pi_ki_slow: (model) => fieldBoolValue(model, 'hardware_sync', false),
      ocxo_pi_switch_abs_error_ppm: (model) => fieldBoolValue(model, 'hardware_sync', false),
      ocxo_pi_switch_hold_s: (model) => fieldBoolValue(model, 'hardware_sync', false),
      ocxo_pi_max_step_fast_ppm: (model) => fieldBoolValue(model, 'hardware_sync', false),
      ocxo_pi_max_step_slow_ppm: (model) => fieldBoolValue(model, 'hardware_sync', false),
      akf_enable: (model) => fieldBoolValue(model, 'hardware_sync', false),
      akf_bootstrap_frames: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      akf_innovation_window: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      akf_max_lag: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      akf_adapt_interval: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      akf_gate_sigma: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      akf_tikhonov_lambda: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      akf_update_smooth: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      akf_q_wf_min: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      akf_q_wf_max: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      akf_q_rw_min: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      akf_q_rw_max: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      akf_r_min: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      akf_r_max: (model) => fieldBoolValue(model, 'hardware_sync', false) && fieldBoolValue(model, 'akf_enable', true),
      bi_sensing_output_enabled: (model) => fieldBoolValue(model, 'enable_bi_sensing', false),
      bi_sensing_ip: (model) => fieldBoolValue(model, 'enable_bi_sensing', false),
      bi_sensing_port: (model) => fieldBoolValue(model, 'enable_bi_sensing', false) && fieldBoolValue(model, 'bi_sensing_output_enabled', true),
      sensing_view_range_bins: (model) => fieldBoolValue(model, 'enable_backend_sensing_processing', false),
      sensing_view_doppler_bins: (model) => fieldBoolValue(model, 'enable_backend_sensing_processing', false),
      sensing_rx_channels: (model) => fieldIntValue(model, 'sensing_rx_channel_count', 0) > 0,
    };

    function shouldHideFieldByDependency(model, field) {
      const rule = DEPENDENT_FIELD_RULES[field.key];
      return Boolean(rule && !rule(model));
    }

    function structuredBoolValue(field, key, fallback = false) {
      const sub = (field.scalar_fields || []).find((item) => item.key === key);
      if (!sub) return fallback;
      if (typeof sub.value === 'boolean') return sub.value;
      const raw = sub.value_text ?? sub.value ?? '';
      if (typeof raw === 'string') {
        const normalized = raw.trim().toLowerCase();
        if (normalized === 'true') return true;
        if (normalized === 'false') return false;
      }
      return Boolean(raw);
    }

    function shouldHideStructuredScalar(field, sub, model = currentModel()) {
      if (field.key === 'simulation' && sub.key === 'enable_uplink') {
        return !fieldBoolValue(model, 'enable_uplink', false);
      }
      if (field.key === 'simulation' && sub.key === 'target_snr_db') {
        return !structuredBoolValue(field, 'snr_control_enable', false);
      }
      if (field.key === 'uplink') {
        const mode = duplexModeValue(model);
        if (['symbol_start', 'symbol_count', 'guard_symbols'].includes(sub.key)) {
          return mode !== 'tdd';
        }
        if (sub.key === 'center_freq') {
          return mode !== 'fdd';
        }
        if (sub.key === 'debug_self_channel') {
          return mode === 'fdd';
        }
      }
      return false;
    }

    function currentRuntimePrefs(tab = currentTab) {
      if (!cache[tab]) cache[tab] = {};
      if (!cache[tab].runtimePrefs) {
        cache[tab].runtimePrefs = { isolateCpu: true, overrideIsolate: false, customCpuSpec: '' };
      }
      return cache[tab].runtimePrefs;
    }

    function defaultCpuSpec() {
      const model = currentModel();
      if (!model) return '';
      const unique = [...new Set(moduleCpuValues(model).filter((value) => value >= 0))].sort((a, b) => a - b);
      return unique.join(',');
    }

    function parseCpuListText(text) {
      return String(text || '')
        .split(',')
        .map((value) => Number.parseInt(value.trim(), 10))
        .filter((value) => Number.isFinite(value));
    }

    function parseCpuScalar(value) {
      const parsed = Number.parseInt(String(value ?? '').trim(), 10);
      return Number.isFinite(parsed) ? parsed : -1;
    }

    function moduleCpuValues(model) {
      const values = [];
      values.push(...parseCpuListText(findField(model, 'downlink_cpu_cores')?.value_text));
      values.push(...parseCpuListText(findField(model, 'uplink_cpu_cores')?.value_text));
      for (const key of ['main_cpu_core']) {
        const field = findField(model, key);
        if (field) values.push(parseCpuScalar(field.value_text ?? field.value));
      }
      const sensingField = findField(model, 'sensing_rx_channels');
      for (const item of (sensingField?.items || [])) {
        values.push(parseCpuScalar(item.rx_cpu_core));
        values.push(parseCpuScalar(item.processing_cpu_core));
      }
      return values;
    }

    function updateRuntimeOptionControls() {
      const prefs = currentRuntimePrefs(currentTab);
      const defaultSpec = defaultCpuSpec();
      isolateToggle.checked = prefs.isolateCpu;
      overrideIsolateToggle.checked = prefs.overrideIsolate;
      if (!prefs.overrideIsolate) {
        overrideIsolateInput.value = defaultSpec;
      } else {
        if (!prefs.customCpuSpec) {
          prefs.customCpuSpec = defaultSpec;
        }
        overrideIsolateInput.value = prefs.customCpuSpec;
      }
      overrideIsolateInput.disabled = !prefs.isolateCpu || !prefs.overrideIsolate;
    }

    function findField(model, key) {
      if (!model?.sections) return null;
      for (const section of model.sections) {
        for (const field of section.fields) {
          if (field.key === key) return field;
        }
      }
      return null;
    }

    function fieldIsInSection(model, key, title) {
      if (!model?.sections) return false;
      for (const section of model.sections) {
        if (section.title !== title) continue;
        if (section.fields.some((field) => field.key === key)) return true;
      }
      return false;
    }

    function parseFlowListInts(field) {
      if (!field) return [];
      const raw = field.value_text ?? field.value ?? '';
      if (Array.isArray(raw)) {
        return raw
          .map((item) => parseIntOrNull(item))
          .filter((value) => value !== null);
      }
      return String(raw)
        .split(',')
        .map((item) => parseIntOrNull(item))
        .filter((value) => value !== null);
    }

    function fieldBoolValue(model, key, fallback = false) {
      const field = findField(model, key);
      if (!field) return fallback;
      if (typeof field.value === 'boolean') return field.value;
      const raw = field.value_text ?? field.value ?? '';
      if (typeof raw === 'string') {
        const normalized = raw.trim().toLowerCase();
        if (normalized === 'true') return true;
        if (normalized === 'false') return false;
      }
      return Boolean(raw);
    }

    function fieldIntValue(model, key, fallback = 0) {
      const field = findField(model, key);
      if (!field) return fallback;
      const raw = field.value_text ?? field.value ?? '';
      return parseIntOrNull(raw) ?? fallback;
    }

    function mappingScalarText(field, key) {
      if (!field || !Array.isArray(field.scalar_fields)) return '';
      const sub = field.scalar_fields.find((item) => item.key === key);
      if (!sub) return '';
      return sub.value_text ?? sub.value ?? '';
    }

    function uniqueSortedInts(values, minInclusive, maxExclusive) {
      return [...new Set(values
        .filter((value) => Number.isInteger(value) && value >= minInclusive && value < maxExclusive))]
        .sort((left, right) => left - right);
    }

    function symbolRowsToBlocks(symbols, config) {
      return symbols.map((symbol) => ({
        symbol_start: symbol,
        symbol_count: 1,
        subcarrier_start: 0,
        subcarrier_count: Math.max(1, config.fftSize),
      }));
    }

	    function plannerConfig(tabName = specialPlannerTabs.has(currentTab) ? 'bs' : currentTab) {
      const model = modelForTab(tabName);
      if (!model) {
        return {
          fftSize: 1024,
          numSymbols: 100,
          syncPos: 1,
          secSyncPos: null,
          cfoTrainingPos: null,
          zcSyncSymbols: [1],
          cfoTrainingSymbols: [],
          tddGuardSymbols: [],
          tddUplinkSymbols: [],
          tddSymbols: [],
          tddConflictLabels: [],
          midframePilotSymbols: [],
          reservedSymbolSet: new Set([1]),
          midframePilotSymbolSet: new Set(),
          pilotPositions: [],
          pilotSet: new Set(),
          dataSymbols: Array.from({ length: 99 }, (_value, index) => index + 1),
          actualToDataSymbol: [],
          nonPilotSubcarriers: Array.from({ length: 1024 }, (_value, index) => index),
          subcarrierToNonPilot: Array.from({ length: 1024 }, (_value, index) => index),
          nonPilotPerSymbol: 1024,
          totalNonPilotRe: 99 * 1024,
          rangeFftSize: 1024,
          dopplerFftSize: 100,
          sensingSymbolNum: 100,
        };
      }
      const fftSize = Math.max(1, parseIntOrNull(findField(model, 'fft_size')?.value_text ?? '') ?? 1024);
      const numSymbols = Math.max(1, parseIntOrNull(findField(model, 'num_symbols')?.value_text ?? '') ?? 100);
      const rangeFftSize = Math.max(1, parseIntOrNull(findField(model, 'range_fft_size')?.value_text ?? '') ?? fftSize);
      const dopplerFftSize = Math.max(1, parseIntOrNull(findField(model, 'doppler_fft_size')?.value_text ?? '') ?? 100);
      const sensingSymbolNum = Math.max(1, parseIntOrNull(findField(model, 'sensing_symbol_num')?.value_text ?? '') ?? numSymbols);
      const syncPos = Math.min(
        numSymbols - 1,
        Math.max(0, parseIntOrNull(findField(model, 'sync_pos')?.value_text ?? '') ?? 0),
      );
      const enableSecSyncSymbol = fieldBoolValue(model, 'enable_sec_sync_symbol', false);
      const enableCfoTrainingSequence = fieldBoolValue(model, 'enable_cfo_training_sequence', false);
      const secSyncPos = enableSecSyncSymbol && syncPos > 0 ? syncPos - 1 : null;
      const cfoTrainingPos = enableCfoTrainingSequence && syncPos + 1 < numSymbols ? syncPos + 1 : null;
	      const pilotPositions = parseFlowListInts(findField(model, 'pilot_positions'))
	        .filter((value) => value >= 0 && value < fftSize);
      const midframePilotSymbols = uniqueSortedInts(
        parseFlowListInts(findField(model, 'midframe_pilot_symbols')),
        0,
        numSymbols,
      );
        const dataResourceField = findField(model, 'data_resource_blocks');
        const sensingPilotBlocks = (!dataResourceField || plannerMode(dataResourceField) !== 'custom')
          ? []
          : plannerBlocks(dataResourceField)
            .filter((block) => payloadBlockKind(block) === 'sensing_pilot');
	      const pilotSet = new Set(pilotPositions);
      const zcSyncSymbols = uniqueSortedInts(
        [syncPos, secSyncPos].filter((value) => value !== null),
        0,
        numSymbols,
      );
      const cfoTrainingSymbols = uniqueSortedInts(
        cfoTrainingPos === null ? [] : [cfoTrainingPos],
        0,
        numSymbols,
      );
      const duplexModeRaw = String(findField(model, 'duplex_mode')?.value_text ?? '').trim().toLowerCase();
      const duplexMode = duplexModeRaw || 'tdd';
      const uplinkField = findField(model, 'uplink');
      let tddGuardSymbols = [];
      let tddUplinkSymbols = [];
      if (duplexMode === 'tdd' && uplinkField) {
        const ulStart = Math.max(0, parseIntOrNull(mappingScalarText(uplinkField, 'symbol_start')) ?? 0);
        const ulCount = Math.max(0, parseIntOrNull(mappingScalarText(uplinkField, 'symbol_count')) ?? 0);
        const ulGuard = Math.max(0, parseIntOrNull(mappingScalarText(uplinkField, 'guard_symbols')) ?? 0);
        const ulEnd = Math.min(numSymbols, ulStart + ulCount);
        const guardEnd = Math.min(ulEnd, ulStart + ulGuard);
        if (ulCount > 0 && ulStart < numSymbols) {
          tddGuardSymbols = Array.from({ length: Math.max(0, guardEnd - ulStart) }, (_value, index) => ulStart + index);
          tddUplinkSymbols = Array.from({ length: Math.max(0, ulEnd - guardEnd) }, (_value, index) => guardEnd + index);
        }
      }
      const tddSymbols = uniqueSortedInts([...tddGuardSymbols, ...tddUplinkSymbols], 0, numSymbols);
      const tddSymbolSet = new Set(tddSymbols);
      const tddConflictLabels = [];
      if (tddSymbolSet.has(syncPos)) {
        tddConflictLabels.push(`sync symbol ${syncPos}`);
      }
      if (secSyncPos !== null && tddSymbolSet.has(secSyncPos)) {
        tddConflictLabels.push(`second sync symbol ${secSyncPos}`);
      }
      if (cfoTrainingPos !== null && tddSymbolSet.has(cfoTrainingPos)) {
        tddConflictLabels.push(`CFO field ${cfoTrainingPos}`);
      }
      const reservedSymbolSet = new Set(zcSyncSymbols);
      const midframePilotSymbolSet = new Set(midframePilotSymbols);
      const actualToDataSymbol = new Array(numSymbols).fill(-1);
      const dataSymbols = [];
      for (let sym = 0; sym < numSymbols; sym += 1) {
        if (reservedSymbolSet.has(sym)) continue;
        if (cfoTrainingSymbols.includes(sym)) continue;
        if (tddSymbolSet.has(sym)) continue;
        if (midframePilotSymbolSet.has(sym)) continue;
        actualToDataSymbol[sym] = dataSymbols.length;
        dataSymbols.push(sym);
      }
      const subcarrierToNonPilot = new Array(fftSize).fill(-1);
      const nonPilotSubcarriers = [];
      for (let sc = 0; sc < fftSize; sc += 1) {
        if (pilotSet.has(sc)) continue;
        subcarrierToNonPilot[sc] = nonPilotSubcarriers.length;
        nonPilotSubcarriers.push(sc);
      }
      return {
        fftSize,
        numSymbols,
        syncPos,
        secSyncPos,
        cfoTrainingPos,
        zcSyncSymbols,
        cfoTrainingSymbols,
        tddGuardSymbols,
        tddUplinkSymbols,
        tddSymbols,
        tddSymbolSet,
        tddConflictLabels,
        midframePilotSymbols,
        reservedSymbolSet,
        midframePilotSymbolSet,
        pilotPositions,
        pilotSet,
        dataSymbols,
        actualToDataSymbol,
        nonPilotSubcarriers,
        subcarrierToNonPilot,
	        nonPilotPerSymbol: nonPilotSubcarriers.length,
	        totalNonPilotRe: nonPilotSubcarriers.length * dataSymbols.length,
	        rangeFftSize,
	        dopplerFftSize,
	        sensingSymbolNum,
          sensingPilotBlocks,
	      };
	    }

    function defaultPlannerBlock(config) {
      const symbolStart = config.dataSymbols.length ? config.dataSymbols[0] : 0;
      return {
        kind: 'payload',
        symbol_start: symbolStart,
        symbol_count: Math.max(1, Math.min(4, config.dataSymbols.length || 1)),
        subcarrier_start: 0,
        subcarrier_count: Math.max(1, Math.min(128, config.fftSize)),
      };
    }

    function syncSymbolsOnlyPlannerBlocks(config) {
      return symbolRowsToBlocks([config.syncPos], config);
    }

	    function knownSymbolsOnlyPlannerBlocks(config) {
	      const blocks = syncSymbolsOnlyPlannerBlocks(config);
      const extraSyncSymbols = (config.zcSyncSymbols || [])
        .filter((symbol) => symbol !== config.syncPos);
      blocks.push(...symbolRowsToBlocks(extraSyncSymbols, config));
      blocks.push(...symbolRowsToBlocks(config.midframePilotSymbols || [], config));
	      for (const pilot of config.pilotPositions) {
	        blocks.push({
	          symbol_start: 0,
          symbol_count: Math.max(1, config.numSymbols),
          subcarrier_start: pilot,
	          subcarrier_count: 1,
	        });
	      }
        for (const block of (config.sensingPilotBlocks || [])) {
          blocks.push({
            symbol_start: block.symbol_start,
            symbol_count: block.symbol_count,
            subcarrier_start: block.subcarrier_start,
            subcarrier_count: block.subcarrier_count,
          });
        }
	      return blocks;
	    }

      function sensingPilotsOnlyPlannerBlocks(config) {
        return (config.sensingPilotBlocks || []).map((block) => ({
          symbol_start: block.symbol_start,
          symbol_count: block.symbol_count,
          subcarrier_start: block.subcarrier_start,
          subcarrier_count: block.subcarrier_count,
        }));
      }

    function guardBandPlannerBlocks(config) {
      const blocks = [];
      const symbolCount = Math.max(1, config.numSymbols);
      const firstStart = 1;
      const firstEnd = Math.min(config.fftSize - 1, 489);
      if (firstEnd >= firstStart) {
        blocks.push({
          kind: 'payload',
          symbol_start: 0,
          symbol_count: symbolCount,
          subcarrier_start: firstStart,
          subcarrier_count: firstEnd - firstStart + 1,
        });
      }
      const secondStart = 535;
      const secondEnd = config.fftSize - 1;
      if (secondEnd >= secondStart) {
        blocks.push({
          kind: 'payload',
          symbol_start: 0,
          symbol_count: symbolCount,
          subcarrier_start: secondStart,
          subcarrier_count: secondEnd - secondStart + 1,
        });
      }
      if (!blocks.length && config.fftSize > 1) {
        blocks.push({
          kind: 'payload',
          symbol_start: 0,
          symbol_count: symbolCount,
          subcarrier_start: 1,
          subcarrier_count: config.fftSize - 1,
        });
      }
      return blocks;
    }

	    function plannerKind(field) {
      if (field?.key === 'sensing_mask_blocks') return 'sensing_mask';
      if (field?.key === 'data_resource_blocks') return 'payload';
	      return field?.planner_kind || 'payload';
	    }

    function plannerMode(field) {
      if (field.mode) return field.mode;
      if (plannerKind(field) === 'payload' && field.allow_omit && !field.items?.length) return 'legacy';
      if (plannerKind(field) === 'sensing_mask') return 'strd';
      return field.items?.length ? 'custom' : 'disabled';
    }

    function plannerBlocks(field) {
      return Array.isArray(field.items) ? field.items : [];
    }

    function payloadBlockKind(block) {
      const raw = String(block?.kind || 'payload').trim().toLowerCase().replace(/[\s-]+/g, '_');
      return raw === 'sensing_pilot' ? 'sensing_pilot' : 'payload';
    }

    function plannerBlockToRect(block, config) {
      const symbolStart = parseIntOrNull(block.symbol_start);
      const symbolCount = parseIntOrNull(block.symbol_count);
      const subcarrierStart = parseIntOrNull(block.subcarrier_start);
      const subcarrierCount = parseIntOrNull(block.subcarrier_count);
      if (
        symbolStart === null || symbolCount === null || symbolCount <= 0 ||
        subcarrierStart === null || subcarrierCount === null || subcarrierCount <= 0
      ) {
        return null;
      }
      if (symbolStart >= config.numSymbols || subcarrierStart >= config.fftSize) {
        return null;
      }
      return {
        symbolStart,
        symbolEnd: Math.min(config.numSymbols - 1, symbolStart + symbolCount - 1),
        subcarrierStart,
        subcarrierEnd: Math.min(config.fftSize - 1, subcarrierStart + subcarrierCount - 1),
      };
    }

    function analyzeSensingCompactMask(blocks, config) {
      const totalGridRe = config.numSymbols * config.fftSize;
      const mask = new Uint8Array(totalGridRe);
      for (const block of blocks) {
        const rect = plannerBlockToRect(block, config);
        if (!rect) continue;
        for (let sym = rect.symbolStart; sym <= rect.symbolEnd; sym += 1) {
          if (config.tddSymbolSet?.has(sym)) {
            continue;
          }
          if (config.cfoTrainingSymbols?.includes(sym)) {
            continue;
          }
          const rowBase = sym * config.fftSize;
          for (let sc = rect.subcarrierStart; sc <= rect.subcarrierEnd; sc += 1) {
            mask[rowBase + sc] = 1;
          }
        }
      }

      const selectedSymbols = [];
      const perSymbolSubcarriers = [];
      for (let sym = 0; sym < config.numSymbols; sym += 1) {
        const row = [];
        const rowBase = sym * config.fftSize;
        for (let sc = 0; sc < config.fftSize; sc += 1) {
          if (mask[rowBase + sc]) {
            row.push(sc);
          }
        }
        if (row.length) {
          selectedSymbols.push(sym);
          perSymbolSubcarriers.push(row);
        }
      }

      const result = {
        regularCompatible: false,
        localDelayDopplerSupported: false,
        selectedSymbolCount: selectedSymbols.length,
        commonSubcarrierCount: perSymbolSubcarriers[0]?.length || 0,
        implicitStride: 0,
        selectedSymbols,
        commonSubcarriers: perSymbolSubcarriers[0] ? [...perSymbolSubcarriers[0]] : [],
        reason: '',
        requiredRangeBins: perSymbolSubcarriers[0]?.length || 0,
        requiredDopplerBins: selectedSymbols.length,
      };

      if (!selectedSymbols.length) {
        result.reason = 'no sensing RE selected';
        return result;
      }
      if (!result.commonSubcarrierCount) {
        result.reason = 'selected symbols have no subcarriers';
        return result;
      }

      for (let row = 1; row < perSymbolSubcarriers.length; row += 1) {
        const current = perSymbolSubcarriers[row];
        if (current.length !== result.commonSubcarrierCount) {
          result.reason = 'selected subcarrier count differs across symbols';
          return result;
        }
        for (let idx = 0; idx < current.length; idx += 1) {
          if (current[idx] !== result.commonSubcarriers[idx]) {
            result.reason = 'selected subcarrier set differs across symbols';
            return result;
          }
        }
      }

      let expectedGap = config.numSymbols;
      if (selectedSymbols.length > 1) {
        expectedGap = selectedSymbols[1] - selectedSymbols[0];
      }
      if (expectedGap <= 0) {
        result.reason = 'selected symbols must be strictly increasing';
        return result;
      }
      for (let idx = 1; idx < selectedSymbols.length; idx += 1) {
        if (selectedSymbols[idx] - selectedSymbols[idx - 1] !== expectedGap) {
          result.reason = 'selected symbols are not equally spaced';
          return result;
        }
      }
      const wrapGap = config.numSymbols + selectedSymbols[0] - selectedSymbols[selectedSymbols.length - 1];
      if (wrapGap !== expectedGap) {
        result.reason = 'selected symbols are not equally spaced across frame wrap-around';
        return result;
      }

      result.regularCompatible = true;
      result.implicitStride = expectedGap;
      if (config.rangeFftSize < result.requiredRangeBins) {
        result.reason = `range_fft_size ${config.rangeFftSize} < required ${result.requiredRangeBins}`;
        return result;
      }
      if (config.dopplerFftSize < result.requiredDopplerBins) {
        result.reason = `doppler_fft_size ${config.dopplerFftSize} < required ${result.requiredDopplerBins}`;
        return result;
      }
      result.localDelayDopplerSupported = true;
      return result;
    }

    function plannerStats(field, config) {
      const mode = plannerMode(field);
      const blocks = plannerBlocks(field);
      const kind = plannerKind(field);
      const totalGridRe = config.numSymbols * config.fftSize;
      if (mode === 'legacy') {
        return {
          mode,
          kind,
          payloadCount: config.totalNonPilotRe,
          totalCount: config.totalNonPilotRe,
          strippedSync: 0,
          strippedCfoTraining: 0,
          strippedTdd: 0,
          strippedPilot: 0,
          overlaps: 0,
          blocks,
        };
      }
      if (mode === 'disabled') {
        return {
          mode,
          kind,
          payloadCount: 0,
          totalCount: kind === 'payload' ? config.totalNonPilotRe : totalGridRe,
          strippedSync: 0,
          strippedCfoTraining: 0,
          strippedTdd: 0,
          strippedPilot: 0,
          overlaps: 0,
          blocks,
        };
      }

      if (kind === 'sensing_mask' && mode === 'strd') {
        return {
          mode,
          kind,
          payloadCount: 0,
          totalCount: totalGridRe,
          strippedSync: 0,
          strippedPilot: 0,
          overlaps: 0,
          blocks,
          compactAnalysis: null,
        };
      }

      if (kind === 'sensing_mask') {
        const mask = new Uint8Array(totalGridRe);
        let selectedCount = 0;
        let overlaps = 0;
        let rejectedCfoTraining = 0;
        let rejectedTdd = 0;
        for (const block of blocks) {
          const rect = plannerBlockToRect(block, config);
          if (!rect) continue;
          for (let sym = rect.symbolStart; sym <= rect.symbolEnd; sym += 1) {
            if (config.tddSymbolSet?.has(sym)) {
              rejectedTdd += rect.subcarrierEnd - rect.subcarrierStart + 1;
              continue;
            }
            if (config.cfoTrainingSymbols?.includes(sym)) {
              rejectedCfoTraining += rect.subcarrierEnd - rect.subcarrierStart + 1;
              continue;
            }
            const rowBase = sym * config.fftSize;
            for (let sc = rect.subcarrierStart; sc <= rect.subcarrierEnd; sc += 1) {
              const flatIndex = rowBase + sc;
              if (mask[flatIndex]) {
                overlaps += 1;
                continue;
              }
              mask[flatIndex] = 1;
              selectedCount += 1;
            }
          }
        }
        return {
          mode,
          kind,
          payloadCount: selectedCount,
          totalCount: totalGridRe,
          strippedSync: 0,
          strippedPilot: 0,
          rejectedCfoTraining,
          rejectedTdd,
          overlaps,
          blocks,
          compactAnalysis: analyzeSensingCompactMask(blocks, config),
        };
      }

      const payloadMask = new Uint8Array(config.totalNonPilotRe);
      const sensingPilotMask = new Uint8Array(config.totalNonPilotRe);
      let payloadCount = 0;
      let sensingPilotCount = 0;
      let strippedSync = 0;
      let strippedCfoTraining = 0;
      let strippedTdd = 0;
      let strippedMidframePilot = 0;
      let strippedPilot = 0;
      let overlaps = 0;
      let crossKindOverlap = 0;
      for (const block of blocks) {
        const rect = plannerBlockToRect(block, config);
        if (!rect) continue;
        const blockKind = payloadBlockKind(block);
	        for (let sym = rect.symbolStart; sym <= rect.symbolEnd; sym += 1) {
          if (config.tddSymbolSet?.has(sym)) {
            strippedTdd += rect.subcarrierEnd - rect.subcarrierStart + 1;
            continue;
          }
	          if (config.zcSyncSymbols?.includes(sym)) {
              if (blockKind === 'payload') {
	                strippedSync += rect.subcarrierEnd - rect.subcarrierStart + 1;
              }
	            continue;
	          }
          if (config.cfoTrainingSymbols?.includes(sym)) {
            strippedCfoTraining += rect.subcarrierEnd - rect.subcarrierStart + 1;
            continue;
          }
          if (config.midframePilotSymbolSet?.has(sym)) {
            if (blockKind === 'payload') {
              strippedMidframePilot += rect.subcarrierEnd - rect.subcarrierStart + 1;
            }
            continue;
          }
          const dataSymbolIndex = config.actualToDataSymbol[sym];
          if (dataSymbolIndex < 0) continue;
          const rowBase = dataSymbolIndex * config.nonPilotPerSymbol;
          for (let sc = rect.subcarrierStart; sc <= rect.subcarrierEnd; sc += 1) {
            if (config.pilotSet.has(sc)) {
              strippedPilot += 1;
              continue;
            }
            const nonPilotIndex = config.subcarrierToNonPilot[sc];
            if (nonPilotIndex < 0) continue;
            const flatIndex = rowBase + nonPilotIndex;
            if (blockKind === 'sensing_pilot') {
              if (sensingPilotMask[flatIndex]) {
                overlaps += 1;
                continue;
              }
              if (payloadMask[flatIndex]) {
                crossKindOverlap += 1;
              }
              sensingPilotMask[flatIndex] = 1;
              continue;
            }
            if (payloadMask[flatIndex]) {
              overlaps += 1;
              continue;
            }
            if (sensingPilotMask[flatIndex]) {
              crossKindOverlap += 1;
            }
            payloadMask[flatIndex] = 1;
          }
        }
      }
      for (let idx = 0; idx < payloadMask.length; idx += 1) {
        if (sensingPilotMask[idx]) {
          sensingPilotCount += 1;
          payloadMask[idx] = 0;
        }
        if (payloadMask[idx]) {
          payloadCount += 1;
        }
      }
        return {
          mode,
          kind,
          payloadCount,
          sensingPilotCount,
          totalCount: config.totalNonPilotRe,
          strippedSync,
          strippedCfoTraining,
          strippedTdd,
          strippedMidframePilot,
          strippedPilot,
          overlaps,
          crossKindOverlap,
          blocks,
          compactAnalysis: null,
        };
      }

    function clonePlannerItems(items) {
      return Array.isArray(items) ? structuredClone(items) : [];
    }

	    function plannerFieldForTab(tabName, spec = plannerSpec()) {
	      const model = modelForTab(tabName);
	      return model ? findField(model, spec.fieldKey) : null;
	    }

	    function plannerSnapshotForTab(tabName, spec = plannerSpec()) {
	      const field = plannerFieldForTab(tabName, spec);
	      if (!field) {
	        return { mode: spec.missingMode, items: [] };
	      }
	      return {
	        mode: plannerMode(field),
	        items: clonePlannerItems(plannerBlocks(field)),
	      };
	    }

    function plannerSnapshotEquals(left, right) {
      return JSON.stringify(left) === JSON.stringify(right);
    }

	    function ensurePlannerState(force = false, spec = plannerSpec()) {
	      if (!cache._plannerStates) {
	        cache._plannerStates = {};
	      }
	      const txSnapshot = plannerSnapshotForTab('bs', spec);
	      const rxSnapshot = plannerSnapshotForTab('ue', spec);
	      const mismatch = !plannerSnapshotEquals(txSnapshot, rxSnapshot);
	      const state = cache._plannerStates[spec.cacheKey];
	      if (!state || force || !state.dirty) {
	        const useRxSnapshot = txSnapshot.mode === spec.missingMode && !txSnapshot.items.length &&
	          (rxSnapshot.mode !== spec.missingMode || rxSnapshot.items.length);
	        const source = useRxSnapshot ? APP.tabs.ue.label : APP.tabs.bs.label;
	        const chosen = useRxSnapshot ? rxSnapshot : txSnapshot;
	        cache._plannerStates[spec.cacheKey] = {
	          key: spec.fieldKey,
		          planner_kind: spec.fieldKey === 'sensing_mask_blocks' ? 'sensing_mask' : 'payload',
		          comment: spec.fieldKey === 'sensing_mask_blocks'
		            ? 'Optional compact sensing RE rectangles'
		            : 'Optional payload / sensing-pilot RE rectangles',
		          display_comment: spec.fieldKey === 'sensing_mask_blocks'
		            ? 'Plan compact sensing rectangles on a dedicated grid, then write them to TX or RX.'
		            : 'Plan payload or sensing-pilot rectangles on a dedicated grid, then apply them to TX or RX.',
	          allow_omit: true,
	          mode: chosen.mode,
	          items: clonePlannerItems(chosen.items),
	          baseTab: 'bs',
	          loadedFrom: mismatch ? source : 'shared',
          dirty: false,
	          txSnapshot,
	          rxSnapshot,
	          mismatch,
	        };
	      } else {
	        cache._plannerStates[spec.cacheKey].txSnapshot = txSnapshot;
	        cache._plannerStates[spec.cacheKey].rxSnapshot = rxSnapshot;
	        cache._plannerStates[spec.cacheKey].mismatch = mismatch;
	      }
	      return cache._plannerStates[spec.cacheKey];
	    }

	    function refreshPlannerComparisonState(spec = plannerSpec()) {
	      const state = cache._plannerStates?.[spec.cacheKey];
	      if (!state) return null;
	      state.txSnapshot = plannerSnapshotForTab('bs', spec);
	      state.rxSnapshot = plannerSnapshotForTab('ue', spec);
	      state.mismatch = !plannerSnapshotEquals(state.txSnapshot, state.rxSnapshot);
	      return state;
	    }

    function validatePlannerForTab(tabName, plannerField) {
      const config = plannerConfig(tabName);
      if (!plannerField) return null;
      if (plannerMode(plannerField) !== 'custom') {
        return null;
      }
      const items = plannerBlocks(plannerField);
      for (let index = 0; index < items.length; index += 1) {
        const block = items[index];
        const symbolStart = parseIntOrNull(block.symbol_start);
        const symbolCount = parseIntOrNull(block.symbol_count);
        const subcarrierStart = parseIntOrNull(block.subcarrier_start);
        const subcarrierCount = parseIntOrNull(block.subcarrier_count);
        if (
          symbolStart === null || symbolCount === null ||
          subcarrierStart === null || subcarrierCount === null
        ) {
          return `Block ${index + 1} has incomplete coordinates.`;
        }
	        if (symbolStart < 0 || subcarrierStart < 0 || symbolCount <= 0 || subcarrierCount <= 0) {
	          return `Block ${index + 1} must use non-negative starts and positive counts.`;
	        }
        if (plannerKind(plannerField) === 'payload') {
          const blockKind = payloadBlockKind(block);
          if (!['payload', 'sensing_pilot'].includes(blockKind)) {
            return `Block ${index + 1} must use payload or sensing_pilot type.`;
          }
        }
	        if (symbolStart + symbolCount > config.numSymbols) {
	          return `Block ${index + 1} exceeds ${APP.tabs[tabName].label} symbol range (${config.numSymbols}).`;
	        }
        if (subcarrierStart + subcarrierCount > config.fftSize) {
          return `Block ${index + 1} exceeds ${APP.tabs[tabName].label} FFT range (${config.fftSize}).`;
        }
      }
      return null;
    }

    function clipPlannerBlocksAroundSymbols(items, config, symbols, preserveKind = false) {
      const blockedSymbols = Array.isArray(symbols) ? symbols : [];
      if (!blockedSymbols.length || !Array.isArray(items) || !items.length) {
        return { items: clonePlannerItems(items), removedRe: 0, removedSymbols: [] };
      }
      const clipped = [];
      let removedRe = 0;
      const removedSymbols = new Set();
      for (const block of items) {
        const rect = plannerBlockToRect(block, config);
        if (!rect) {
          clipped.push(structuredClone(block));
          continue;
        }
        let ranges = [[rect.symbolStart, rect.symbolEnd]];
        for (const blockedSymbol of blockedSymbols) {
          if (blockedSymbol < rect.symbolStart || blockedSymbol > rect.symbolEnd) continue;
          removedSymbols.add(blockedSymbol);
          removedRe += rect.subcarrierEnd - rect.subcarrierStart + 1;
          const nextRanges = [];
          for (const [start, end] of ranges) {
            if (blockedSymbol < start || blockedSymbol > end) {
              nextRanges.push([start, end]);
              continue;
            }
            if (start <= blockedSymbol - 1) nextRanges.push([start, blockedSymbol - 1]);
            if (blockedSymbol + 1 <= end) nextRanges.push([blockedSymbol + 1, end]);
          }
          ranges = nextRanges;
        }
        for (const [start, end] of ranges) {
          clipped.push({
            ...(preserveKind ? { kind: payloadBlockKind(block) } : {}),
            symbol_start: start,
            symbol_count: end - start + 1,
            subcarrier_start: rect.subcarrierStart,
            subcarrier_count: rect.subcarrierEnd - rect.subcarrierStart + 1,
          });
        }
      }
      return {
        items: clipped,
        removedRe,
        removedSymbols: [...removedSymbols].sort((left, right) => left - right),
      };
    }

    function sanitizePlannerFieldForTab(tabName, field) {
      if (!field || plannerMode(field) !== 'custom') {
        return null;
      }
      const config = plannerConfig(tabName);
      const kind = plannerKind(field);
      const blockedSymbols = kind === 'sensing_mask'
        ? [...(config.tddSymbols || []), ...(config.cfoTrainingSymbols || [])]
        : (config.tddSymbols || []);
      const result = clipPlannerBlocksAroundSymbols(
        plannerBlocks(field),
        config,
        blockedSymbols,
        kind === 'payload',
      );
      if (!result.removedRe) {
        return null;
      }
      field.items = result.items;
      field.dirty = true;
      result.fieldKey = field.key;
      result.kind = kind;
      return result;
    }

    function makeSensingChannelItem(field, index, existingItems) {
      const baseSource = existingItems[0] && typeof existingItems[0] === 'object'
        ? structuredClone(existingItems[0])
        : structuredClone(field.default_item || {});
      const item = {};
      for (const sub of (field.item_fields || [])) {
        if (Object.prototype.hasOwnProperty.call(baseSource, sub.key)) {
          item[sub.key] = baseSource[sub.key];
        } else {
          item[sub.key] = sub.kind === 'bool' ? false : '';
        }
      }
      if (Object.prototype.hasOwnProperty.call(item, 'usrp_channel')) {
        const baseChannel = parseIntOrNull(baseSource.usrp_channel);
        item.usrp_channel = baseChannel === null ? index : baseChannel + index;
      }
      return item;
    }

    function ensureSensingChannelItems() {
      const model = currentModel();
      if (!model) return;
      const field = findField(model, 'sensing_rx_channels');
      if (!field) return;
      const countField = findField(model, 'sensing_rx_channel_count');
      const countRaw = countField ? (countField.value_text ?? countField.value ?? '') : '';
      const count = Math.max(0, parseIntOrNull(countRaw) ?? 0);
      const items = Array.isArray(field.items) ? field.items : [];
      field.items = items;
      while (items.length < count) {
        items.push(makeSensingChannelItem(field, items.length, items));
      }
      while (items.length > count) {
        items.pop();
      }
    }

	    function renderDataResourcePlanner(field, options = {}) {
	      const sourceTab = options.sourceTab || currentTab;
	      const standalone = Boolean(options.standalone);
      const readOnly = Boolean(options.readOnly);
	      const onChange = typeof options.onChange === 'function' ? options.onChange : () => {};
	      const kind = plannerKind(field);
	      const isPayloadPlanner = kind === 'payload';
	      const isSensingPlanner = kind === 'sensing_mask';
	      if (isPayloadPlanner && !field.new_block_kind) {
	        field.new_block_kind = 'payload';
	      }
	      const plannerTitle = isPayloadPlanner ? 'Resource Map' : 'Sensing Resource Map';
	      const plannerModeLabel = isPayloadPlanner ? 'Resource Mapping Mode' : 'Sensing Resource Mode';
	      const legacyOption = isPayloadPlanner
	        ? '<option value="legacy">Legacy Full Grid</option>'
	        : '';
	      const customOptionLabel = isPayloadPlanner ? 'Custom Blocks' : 'Custom Blocks';
	      const disabledOptionLabel = isPayloadPlanner ? 'Disable Payload' : 'Disable Compact Mask';
	      const sensingModeOptions = isSensingPlanner
	        ? '<option value="strd">STRD-based</option><option value="custom">Custom Blocks</option>'
	        : '';
	      const presetOptions = isPayloadPlanner
	        ? '<option value="guard_band_grid">Guard Band Grid</option>'
	        : '<option value="sync_symbols_only">Sync symbols only</option><option value="known_symbols_only">Known symbols only</option><option value="sensing_pilots_only">Sensing pilots only</option>';
	      field.items = plannerBlocks(field);
	      field.mode = plannerMode(field);

      const holder = document.createElement('div');
      holder.className = 'kv-row';
      if (standalone) {
        holder.classList.add('planner-standalone-row');
      }
      holder.innerHTML = `
        <div class="key-col">
          <code>${plannerTitle}</code>
          <div class="hint">${field.display_comment || field.comment || ''}</div>
        </div>
        <div class="value-col"></div>
      `;
      const valueCol = holder.querySelector('.value-col');
      const card = document.createElement('div');
      card.className = 'planner-card';

      let modeSelect = null;
      let presetSelect = null;
      let loadPresetBtn = null;
      let addBtn = null;
      let clearBtn = null;
      if (!readOnly) {
	        const toolbar = document.createElement('div');
	        toolbar.className = 'planner-toolbar';
	        const modeField = document.createElement('div');
	        modeField.className = 'field';
	        modeField.innerHTML = `
	          <label>${plannerModeLabel}</label>
	          <select>
	            ${legacyOption}
	            ${isSensingPlanner ? sensingModeOptions : `<option value="custom">${customOptionLabel}</option><option value="disabled">${disabledOptionLabel}</option>`}
	          </select>
	        `;
	        modeSelect = modeField.querySelector('select');
	        const presetField = document.createElement('div');
	        presetField.className = 'field';
	        presetField.innerHTML = `
	          <label>Preset</label>
	          <select>
	            <option value="">Custom / Current</option>
	            ${presetOptions}
	          </select>
	        `;
	        presetSelect = presetField.querySelector('select');
      let blockKindField = null;
      let blockKindSelect = null;
        if (isPayloadPlanner) {
        blockKindField = document.createElement('div');
        blockKindField.className = 'field';
        blockKindField.innerHTML = `
          <label>New Block Type</label>
          <select>
            <option value="payload">payload</option>
            <option value="sensing_pilot">sensing_pilot</option>
          </select>
        `;
        blockKindSelect = blockKindField.querySelector('select');
        blockKindSelect.value = field.new_block_kind || 'payload';
        blockKindSelect.addEventListener('change', () => {
          field.new_block_kind = blockKindSelect.value;
        });
      }
	        const actions = document.createElement('div');
	        actions.className = 'actions';
	        addBtn = document.createElement('button');
	        addBtn.className = 'btn';
	        addBtn.textContent = 'Add Block';
	        loadPresetBtn = document.createElement('button');
	        loadPresetBtn.className = 'btn';
	        loadPresetBtn.textContent = 'Load Preset';
	        clearBtn = document.createElement('button');
	        clearBtn.className = 'btn warn';
	        clearBtn.textContent = 'Clear Blocks';
	        actions.appendChild(loadPresetBtn);
	        actions.appendChild(addBtn);
	        actions.appendChild(clearBtn);
	        toolbar.appendChild(modeField);
	        toolbar.appendChild(presetField);
        if (blockKindField) {
          toolbar.appendChild(blockKindField);
        }
	        toolbar.appendChild(actions);
        card.appendChild(toolbar);

        modeField._modeSelect = modeSelect;
        presetField._presetSelect = presetSelect;
        actions._addBtn = addBtn;
        actions._loadPresetBtn = loadPresetBtn;
        actions._clearBtn = clearBtn;
        toolbar._blockKindSelect = blockKindSelect;
	      }
      if (readOnly) {
        const note = document.createElement('div');
        note.className = 'subtle';
        note.textContent = isPayloadPlanner
          ? 'Read-only preview. Edit payload / sensing-pilot mapping in the Resource Map tab.'
          : 'Read-only preview. Edit sensing resource mapping in the Sensing Resource Map tab.';
        card.appendChild(note);
      }
	      const statsHost = document.createElement('div');
	      statsHost.className = 'planner-meta';

	      const legend = document.createElement('div');
	      legend.className = 'planner-legend';
	      legend.innerHTML = `
	        <span><i style="background:#d8e8ff;"></i>${isPayloadPlanner ? 'payload area' : 'sensing resource area'}</span>
	        ${isPayloadPlanner ? '<span><i style="background:#d7f5d1;"></i>sensing pilot area</span>' : ''}
	        <span><i style="background:#ffddc7;"></i>ZC sync symbol</span>
	        <span><i style="background:#ffd6ea;"></i>CFO field</span>
	        <span><i style="background:#eadcff;"></i>TDD guard</span>
	        <span><i style="background:#ffcdd2;"></i>TDD uplink</span>
	        <span><i style="background:#fff1ad;"></i>mid-frame pilot</span>
	        <span><i style="background:#d7dce2;"></i>pilot subcarriers</span>
	        <span><i style="background:#87c4ff;"></i>drag preview / block</span>
	      `;

      const canvasWrap = document.createElement('div');
      canvasWrap.className = 'planner-canvas-wrap';
      const canvas = document.createElement('canvas');
      canvas.className = 'planner-canvas';
      canvasWrap.appendChild(canvas);

      const axis = document.createElement('div');
      axis.className = 'planner-axis';
      axis.innerHTML = `
        <span>subcarrier: 0</span>
        <span class="planner-coord">hover: -</span>
        <span>subcarrier: -</span>
      `;
      const coordLabel = axis.querySelector('.planner-coord');
      const axisRight = axis.lastElementChild;

      const blocksHost = document.createElement('div');
      blocksHost.className = 'planner-block-list';
	      card.appendChild(statsHost);
	      card.appendChild(legend);
	      card.appendChild(canvasWrap);
	      card.appendChild(axis);
      if (!readOnly) {
	        card.appendChild(blocksHost);
      }
	      valueCol.appendChild(card);

      let dragStart = null;
      let dragCurrent = null;

      function refreshStats() {
        const config = plannerConfig(sourceTab);
        const stats = plannerStats(field, config);
        const currentPreset = presetSelect ? presetSelect.value : '';
        if (modeSelect) {
	          modeSelect.value = field.mode;
        }
	        if (presetSelect && !presetSelect.dataset.keepValue) {
	          presetSelect.value = '';
	        }
        axisRight.textContent = `subcarrier: ${config.fftSize - 1}`;
        statsHost.innerHTML = '';
	        const payloadPct = stats.totalCount
	          ? ((stats.payloadCount / stats.totalCount) * 100).toFixed(1)
	          : '0.0';
        const sensingPilotPct = stats.totalCount
          ? (((stats.sensingPilotCount || 0) / stats.totalCount) * 100).toFixed(1)
          : '0.0';
	        const primaryLabel = isPayloadPlanner ? 'payload RE' : 'sensing RE';
	        const statItems = [
	          `${primaryLabel} ${stats.payloadCount}/${stats.totalCount} (${payloadPct}%)`,
	          `overlap ignored ${stats.overlaps}`,
	        ];
	        if (isPayloadPlanner) {
          statItems.push(`sensing pilot RE ${stats.sensingPilotCount || 0}/${stats.totalCount} (${sensingPilotPct}%)`);
	          statItems.push(`sync stripped ${stats.strippedSync}`);
          statItems.push(`CFO field stripped ${stats.strippedCfoTraining || 0}`);
          statItems.push(`TDD stripped ${stats.strippedTdd || 0}`);
          statItems.push(`midframe pilot stripped ${stats.strippedMidframePilot || 0}`);
	          statItems.push(`pilot stripped ${stats.strippedPilot}`);
          if (stats.crossKindOverlap) {
            statItems.push(`payload/sensing overlap ${stats.crossKindOverlap} (sensing pilot wins)`);
          }
	        }
        if (isSensingPlanner && stats.rejectedCfoTraining) {
          statItems.push(`CFO field removed on save ${stats.rejectedCfoTraining}`);
        }
        if (isSensingPlanner && stats.rejectedTdd) {
          statItems.push(`TDD removed on save ${stats.rejectedTdd}`);
        }
        if (config.tddSymbols?.length) {
          statItems.push(`TDD symbols ${config.tddSymbols[0]}..${config.tddSymbols[config.tddSymbols.length - 1]}`);
        }
        if (config.tddConflictLabels?.length) {
          statItems.push(`TDD conflict: ${config.tddConflictLabels.join(', ')}`);
        }
	        if (currentPreset === 'guard_band_grid') {
	          statItems.push(
	            isPayloadPlanner && config.fftSize >= 535
	              ? 'preset: bins 1-489 and 535-end'
	              : 'preset: bins 1-489 (or FFT end) from plot_const active set'
	          );
	        } else if (currentPreset === 'sync_symbols_only') {
	          statItems.push(`preset: sync symbol row only`);
	        } else if (currentPreset === 'known_symbols_only') {
	          statItems.push(`preset: sync symbol plus pilots plus sensing pilots`);
          } else if (currentPreset === 'sensing_pilots_only') {
            statItems.push(`preset: sensing pilot blocks only`);
	        }
	        if (field.mode !== 'custom' && field.items.length) {
	          statItems.push(`stored blocks ${field.items.length} (currently ignored)`);
	        } else {
	          statItems.push(`blocks ${field.items.length}`);
	        }
	        if (isSensingPlanner) {
	          statItems.push(`mode ${field.mode === 'custom' ? 'Custom Blocks' : 'STRD-based'}`);
	          if (stats.compactAnalysis) {
	            const analysis = stats.compactAnalysis;
	            if (analysis.localDelayDopplerSupported) {
	              statItems.push('MTI/local DD available');
	              statItems.push(`implicit stride ${analysis.implicitStride}`);
	              statItems.push(`used symbols ${analysis.selectedSymbolCount}`);
	              statItems.push(`used subcarriers ${analysis.commonSubcarrierCount}`);
	            } else if (analysis.regularCompatible) {
	              statItems.push('regular mask but FFT sizes too small');
	              statItems.push(`need range_fft_size >= ${analysis.requiredRangeBins}`);
	              statItems.push(`need doppler_fft_size >= ${analysis.requiredDopplerBins}`);
	            } else {
	              statItems.push(`not compatible: ${analysis.reason || 'mask is not regular-sampling compatible'}`);
	            }
	          }
	        }
	        for (const text of statItems) {
          const pill = document.createElement('span');
          pill.className = 'planner-stat';
          pill.textContent = text;
          statsHost.appendChild(pill);
        }
      }

      function renderBlockList() {
        blocksHost.innerHTML = '';
        if (!field.items.length) {
          const empty = document.createElement('div');
	          empty.className = 'planner-empty';
	          empty.textContent =
	            field.mode === 'legacy'
	              ? 'Legacy full-grid mode is active. Switch to Custom Blocks and drag on the planner to create payload rectangles.'
	              : isSensingPlanner && field.mode === 'strd'
	                ? 'Traditional STRD-based sensing mode is active. Switch to Custom Blocks to explicitly plan sensing time-frequency resources.'
	                : isPayloadPlanner
	                ? 'No custom payload blocks yet. Drag on the planner or use Add Block.'
	                : 'No custom sensing resource blocks yet. Drag on the planner or use Add Block.';
	          blocksHost.appendChild(empty);
	          return;
	        }
        field.items.forEach((block, index) => {
          const cardEl = document.createElement('div');
          cardEl.className = 'planner-block';
	          const head = document.createElement('div');
	          head.className = 'planner-block-head';
	          const title = document.createElement('strong');
          const blockKind = isPayloadPlanner ? payloadBlockKind(block) : '';
	          title.textContent = isPayloadPlanner
            ? `Block ${index + 1} · ${blockKind}`
            : `Block ${index + 1}`;
          const removeBtn = document.createElement('button');
          removeBtn.className = 'btn danger';
          removeBtn.textContent = 'Delete';
          removeBtn.addEventListener('click', () => {
            field.items.splice(index, 1);
            refreshAll();
          });
          head.appendChild(title);
          head.appendChild(removeBtn);
          cardEl.appendChild(head);

	          const grid = document.createElement('div');
	          grid.className = 'planner-block-grid';
          const specs = [];
          if (isPayloadPlanner) {
            specs.push(['kind', 'Type']);
          }
          specs.push(
            ['symbol_start', 'Symbol Start'],
            ['symbol_count', 'Symbol Count'],
            ['subcarrier_start', 'Subcarrier Start'],
            ['subcarrier_count', 'Subcarrier Count'],
          );
	          for (const [key, label] of specs) {
	            const fieldWrap = document.createElement('div');
	            const labelEl = document.createElement('label');
	            labelEl.textContent = label;
            let input;
            if (key === 'kind') {
              input = document.createElement('select');
              input.innerHTML = `
                <option value="payload">payload</option>
                <option value="sensing_pilot">sensing_pilot</option>
              `;
              input.value = payloadBlockKind(block);
              input.addEventListener('change', () => {
                block[key] = input.value;
                field.dirty = true;
                refreshAll();
              });
            } else {
	              input = document.createElement('input');
	              input.type = 'number';
	              input.step = '1';
	              input.min = '0';
	              input.value = block[key] ?? '';
	              input.addEventListener('input', () => {
	                block[key] = input.value;
	                field.dirty = true;
	                refreshStats();
	                drawPlanner();
	              });
	              input.addEventListener('change', () => {
	                const numeric = parseIntOrNull(input.value);
	                block[key] = numeric === null ? '' : numeric;
	                input.value = block[key] === '' ? '' : String(block[key]);
	                field.dirty = true;
	                refreshAll();
	              });
            }
	            fieldWrap.appendChild(labelEl);
	            fieldWrap.appendChild(input);
	            grid.appendChild(fieldWrap);
          }
          cardEl.appendChild(grid);
          blocksHost.appendChild(cardEl);
        });
      }

      function eventToCoord(event) {
        const config = plannerConfig(sourceTab);
        const rect = canvas.getBoundingClientRect();
        const x = Math.min(Math.max(event.clientX - rect.left, 0), rect.width - 1);
        const y = Math.min(Math.max(event.clientY - rect.top, 0), rect.height - 1);
        const subcarrier = Math.min(config.fftSize - 1, Math.max(0, Math.round((x / rect.width) * config.fftSize - 0.5)));
        const symbol = Math.min(config.numSymbols - 1, Math.max(0, Math.round((y / rect.height) * config.numSymbols - 0.5)));
        return { x, y, subcarrier, symbol };
      }

      function dragRect(config) {
        if (!dragStart || !dragCurrent) return null;
        const symbolStart = Math.min(dragStart.symbol, dragCurrent.symbol);
        const symbolEnd = Math.max(dragStart.symbol, dragCurrent.symbol);
        const subcarrierStart = Math.min(dragStart.subcarrier, dragCurrent.subcarrier);
        const subcarrierEnd = Math.max(dragStart.subcarrier, dragCurrent.subcarrier);
        return {
          symbolStart,
          symbolEnd: Math.min(config.numSymbols - 1, symbolEnd),
          subcarrierStart,
          subcarrierEnd: Math.min(config.fftSize - 1, subcarrierEnd),
        };
      }

      function drawBlockRect(ctx, rect, config, fillStyle, strokeStyle, dashed = false) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        const x0 = (rect.subcarrierStart / config.fftSize) * width;
        const x1 = ((rect.subcarrierEnd + 1) / config.fftSize) * width;
        const y0 = (rect.symbolStart / config.numSymbols) * height;
        const y1 = ((rect.symbolEnd + 1) / config.numSymbols) * height;
        ctx.fillStyle = fillStyle;
        ctx.fillRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
        ctx.save();
        ctx.strokeStyle = strokeStyle;
        ctx.lineWidth = 2;
        if (dashed) ctx.setLineDash([8, 4]);
        ctx.strokeRect(x0 + 1, y0 + 1, Math.max(0, x1 - x0 - 2), Math.max(0, y1 - y0 - 2));
        ctx.restore();
      }

      function drawableRectSegments(rect, config) {
        const blockedSymbols = new Set(config.tddSymbols || []);
        if (isSensingPlanner) {
          for (const symbol of config.cfoTrainingSymbols || []) {
            blockedSymbols.add(symbol);
          }
        }
        if (!blockedSymbols.size) return [rect];
        let ranges = [[rect.symbolStart, rect.symbolEnd]];
        for (const blockedSymbol of [...blockedSymbols].sort((left, right) => left - right)) {
          if (blockedSymbol < rect.symbolStart || blockedSymbol > rect.symbolEnd) continue;
          const nextRanges = [];
          for (const [start, end] of ranges) {
            if (blockedSymbol < start || blockedSymbol > end) {
              nextRanges.push([start, end]);
              continue;
            }
            if (start <= blockedSymbol - 1) nextRanges.push([start, blockedSymbol - 1]);
            if (blockedSymbol + 1 <= end) nextRanges.push([blockedSymbol + 1, end]);
          }
          ranges = nextRanges;
        }
        return ranges.map(([start, end]) => ({
          symbolStart: start,
          symbolEnd: end,
          subcarrierStart: rect.subcarrierStart,
          subcarrierEnd: rect.subcarrierEnd,
        }));
      }

      function drawPlanner() {
        const config = plannerConfig(sourceTab);
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(240, Math.floor(canvas.clientWidth * dpr));
        const height = Math.max(180, Math.floor(canvas.clientHeight * dpr));
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
        }
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);

        for (let sym = 0; sym <= config.numSymbols; sym += Math.max(1, Math.floor(config.numSymbols / 10))) {
          const y = (sym / config.numSymbols) * height;
          ctx.strokeStyle = 'rgba(95, 111, 130, 0.12)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(width, y);
          ctx.stroke();
        }
        for (let sc = 0; sc <= config.fftSize; sc += Math.max(1, Math.floor(config.fftSize / 8))) {
          const x = (sc / config.fftSize) * width;
          ctx.strokeStyle = 'rgba(95, 111, 130, 0.08)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
        }

	        if (field.mode === 'legacy') {
	          drawBlockRect(
	            ctx,
	            { symbolStart: 0, symbolEnd: config.numSymbols - 1, subcarrierStart: 0, subcarrierEnd: config.fftSize - 1 },
	            config,
	            'rgba(0, 86, 179, 0.12)',
	            'rgba(0, 86, 179, 0.0)',
	          );
	        } else if (field.mode === 'custom') {
	          field.items.forEach((block, index) => {
	            const rect = plannerBlockToRect(block, config);
	            if (!rect) return;
            const blockKind = isPayloadPlanner ? payloadBlockKind(block) : 'payload';
	            const hue = blockKind === 'sensing_pilot'
              ? (120 + (index * 19) % 40)
              : (205 + (index * 27) % 90);
            for (const segment of drawableRectSegments(rect, config)) {
	              drawBlockRect(
	                ctx,
	                segment,
	                config,
	                `hsla(${hue}, 85%, ${blockKind === 'sensing_pilot' ? '74' : '72'}%, 0.32)`,
	                `hsla(${hue}, 80%, ${blockKind === 'sensing_pilot' ? '32' : '40'}%, 0.95)`,
	              );
            }
	          });
	        }

        for (const pilot of config.pilotPositions) {
          const x0 = (pilot / config.fftSize) * width;
          const x1 = ((pilot + 1) / config.fftSize) * width;
          ctx.fillStyle = 'rgba(120, 134, 152, 0.55)';
          ctx.fillRect(x0, 0, Math.max(1, x1 - x0), height);
        }

        function drawSymbolRows(symbols, fillStyle) {
          for (const symbol of symbols || []) {
            const y0 = (symbol / config.numSymbols) * height;
            const y1 = ((symbol + 1) / config.numSymbols) * height;
            ctx.fillStyle = fillStyle;
            ctx.fillRect(0, y0, width, Math.max(1, y1 - y0));
          }
        }

        drawSymbolRows(config.zcSyncSymbols || [config.syncPos], 'rgba(255, 145, 77, 0.4)');
        drawSymbolRows(config.cfoTrainingSymbols || [], 'rgba(244, 114, 182, 0.35)');
        drawSymbolRows(config.tddGuardSymbols || [], 'rgba(126, 87, 194, 0.28)');
        drawSymbolRows(config.tddUplinkSymbols || [], 'rgba(220, 38, 38, 0.24)');
        drawSymbolRows(config.midframePilotSymbols || [], 'rgba(245, 196, 0, 0.34)');

        const preview = dragRect(config);
        if (preview) {
          drawBlockRect(
            ctx,
            preview,
            config,
            'rgba(70, 162, 255, 0.18)',
            'rgba(0, 86, 179, 0.95)',
            true,
          );
        }

        ctx.fillStyle = '#2c3e50';
        ctx.font = `${12 * dpr}px var(--ui-font)`;
        ctx.fillText(`symbols 0..${config.numSymbols - 1}`, 12 * dpr, 18 * dpr);
        ctx.fillText(`fft bins 0..${config.fftSize - 1}`, 12 * dpr, 36 * dpr);
      }

	      function refreshAll() {
	        refreshStats();
        if (!readOnly) {
	          renderBlockList();
        }
	        drawPlanner();
	        onChange(field);
	      }

        if (!readOnly) {
	      modeSelect.addEventListener('change', () => {
	        field.mode = modeSelect.value;
	        field.dirty = true;
	        refreshAll();
	      });

      loadPresetBtn.addEventListener('click', () => {
	        const preset = presetSelect.value;
	        if (!preset) {
	          return;
	        }
	        if (preset === 'guard_band_grid') {
          field.mode = 'custom';
          field.items = guardBandPlannerBlocks(plannerConfig(sourceTab));
          field.dirty = true;
          presetSelect.dataset.keepValue = '1';
	          refreshAll();
	          delete presetSelect.dataset.keepValue;
	          return;
	        }
	        if (preset === 'sync_symbols_only') {
	          field.mode = 'custom';
	          field.items = syncSymbolsOnlyPlannerBlocks(plannerConfig(sourceTab));
	          field.dirty = true;
	          presetSelect.dataset.keepValue = '1';
	          refreshAll();
	          delete presetSelect.dataset.keepValue;
	          return;
	        }
	        if (preset === 'known_symbols_only') {
	          field.mode = 'custom';
	          field.items = knownSymbolsOnlyPlannerBlocks(plannerConfig(sourceTab));
	          field.dirty = true;
	          presetSelect.dataset.keepValue = '1';
	          refreshAll();
	          delete presetSelect.dataset.keepValue;
            return;
	        }
          if (preset === 'sensing_pilots_only') {
            field.mode = 'custom';
            field.items = sensingPilotsOnlyPlannerBlocks(plannerConfig(sourceTab));
            field.dirty = true;
            presetSelect.dataset.keepValue = '1';
            refreshAll();
            delete presetSelect.dataset.keepValue;
	        }
	      });

	      addBtn.addEventListener('click', () => {
	        if (field.mode !== 'custom') field.mode = 'custom';
        const nextBlock = defaultPlannerBlock(plannerConfig(sourceTab));
        if (isPayloadPlanner) {
          nextBlock.kind = field.new_block_kind || 'payload';
        }
	        field.items.push(nextBlock);
	        field.dirty = true;
	        refreshAll();
	      });

      clearBtn.addEventListener('click', () => {
        field.items = [];
        field.dirty = true;
        if (field.mode === 'custom') {
          refreshAll();
          return;
        }
        refreshAll();
      });

      canvas.addEventListener('pointerdown', (event) => {
        if (event.button !== 0) return;
        if (field.mode !== 'custom') field.mode = 'custom';
        dragStart = eventToCoord(event);
        dragCurrent = dragStart;
        coordLabel.textContent = `drag: sym ${dragStart.symbol}, sc ${dragStart.subcarrier}`;
        drawPlanner();
      });

      canvas.addEventListener('pointermove', (event) => {
        const point = eventToCoord(event);
        if (dragStart) {
          dragCurrent = point;
          const preview = dragRect(plannerConfig(sourceTab));
          if (preview) {
            coordLabel.textContent =
              `drag: sym ${preview.symbolStart}..${preview.symbolEnd}, sc ${preview.subcarrierStart}..${preview.subcarrierEnd}`;
          }
          drawPlanner();
          return;
        }
        coordLabel.textContent = `hover: sym ${point.symbol}, sc ${point.subcarrier}`;
      });

      function finishDrag() {
        if (!dragStart || !dragCurrent) {
          dragStart = null;
          dragCurrent = null;
          drawPlanner();
          return;
        }
        const preview = dragRect(plannerConfig(sourceTab));
	        if (preview) {
            const nextBlock = {
              ...(isPayloadPlanner ? { kind: field.new_block_kind || 'payload' } : {}),
	            symbol_start: preview.symbolStart,
	            symbol_count: preview.symbolEnd - preview.symbolStart + 1,
	            subcarrier_start: preview.subcarrierStart,
	            subcarrier_count: preview.subcarrierEnd - preview.subcarrierStart + 1,
	          };
	          field.items.push(nextBlock);
	          field.dirty = true;
	        }
        dragStart = null;
        dragCurrent = null;
        refreshAll();
      }

      canvas.addEventListener('pointerup', finishDrag);
	      canvas.addEventListener('pointerleave', () => {
	        if (dragStart) {
	          finishDrag();
	          return;
	        }
	        coordLabel.textContent = 'hover: -';
	      });
        } else {
          canvas.style.cursor = 'default';
        }

      if (window.ResizeObserver) {
        const observer = new ResizeObserver(() => drawPlanner());
        observer.observe(canvas);
      }

      refreshAll();
      return holder;
    }

	    function fieldHintText(field) {
      const base = field.display_comment || field.comment || '';
      if (!field.optional) return base;
      if (field.default_text) {
        return base
          ? `${base}<br>Default: ${field.default_text}.`
          : `Optional.<br>Default: ${field.default_text}.`;
      }
      return base ? `${base} Optional; blank omits this key.` : 'Optional; blank omits this key.';
    }

    function renderScalarControl(host, field, onChange = () => {}) {
      if (field.kind === 'bool') {
        const wrapper = document.createElement('label');
        wrapper.className = 'checkbox-row';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = Boolean(field.value);
        input.addEventListener('change', () => {
          field.value = input.checked;
          field.value_text = input.checked ? 'true' : 'false';
          field.is_set = true;
          onChange(field);
        });
        wrapper.appendChild(input);
        host.appendChild(wrapper);
        return;
      }
      if (Array.isArray(field.options) && field.options.length) {
        const select = document.createElement('select');
        if (field.optional) {
          const blank = document.createElement('option');
          blank.value = '';
          blank.textContent = '';
          select.appendChild(blank);
        }
        for (const optionValue of field.options) {
          const option = document.createElement('option');
          option.value = String(optionValue);
          option.textContent = String(optionValue);
          select.appendChild(option);
        }
        select.value = field.value_text ?? '';
        if (!field.optional && !field.options.map(String).includes(select.value) && field.options.length) {
          select.value = String(field.options[0]);
          field.value_text = select.value;
        }
        select.addEventListener('change', () => {
          field.value_text = select.value;
          field.value = select.value;
          field.is_set = Boolean(select.value) || !field.optional;
          onChange(field);
        });
        host.appendChild(select);
        return;
      }
      const input = document.createElement(field.type === 'flow_list' ? 'textarea' : 'input');
      if (field.type !== 'flow_list') input.type = 'text';
      input.value = field.value_text ?? '';
      if (field.default_text) {
        input.placeholder = field.default_text;
      }
      input.addEventListener('input', () => {
        field.value_text = input.value;
        field.value = input.value;
        field.is_set = Boolean(input.value.trim()) || !field.optional;
        onChange(field);
      });
      host.appendChild(input);
      if (field.display_unit) {
        const unit = document.createElement('div');
        unit.className = 'hint';
        unit.textContent = `Display unit: ${field.display_unit}`;
        host.appendChild(unit);
      }
    }

    function renderMappingListItems(field, valueCol) {
      field.items = Array.isArray(field.items) ? field.items : [];
      field.items.forEach((item, index) => {
        const channel = document.createElement('div');
        channel.className = 'channel-card';
        const heading = document.createElement('h4');
        heading.textContent = `${field.key}[${index}]`;
        channel.appendChild(heading);
        const inner = document.createElement('div');
        inner.className = 'kv';
        field.item_fields.forEach((sub) => {
          const row = document.createElement('div');
          row.className = 'kv-row';
          const value = item[sub.key] ?? '';
          row.innerHTML = `
            <div class="key-col">
              <code>${sub.key}</code>
              <div class="hint">${sub.display_comment || sub.comment || ''}</div>
            </div>
            <div class="value-col"></div>
          `;
          const controlHost = row.querySelector('.value-col');
          if (sub.kind === 'bool') {
            const wrapper = document.createElement('label');
            wrapper.className = 'checkbox-row';
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = Boolean(value);
            input.addEventListener('change', () => {
              item[sub.key] = input.checked;
            });
            wrapper.appendChild(input);
            controlHost.appendChild(wrapper);
          } else if (Array.isArray(sub.options) && sub.options.length) {
            const select = document.createElement('select');
            for (const optionValue of sub.options) {
              const option = document.createElement('option');
              option.value = String(optionValue);
              option.textContent = String(optionValue);
              select.appendChild(option);
            }
            select.value = value === null || value === undefined ? '' : String(value);
            if (!sub.options.map(String).includes(select.value) && sub.options.length) {
              select.value = String(sub.options[0]);
              item[sub.key] = select.value;
            }
            select.addEventListener('change', () => {
              item[sub.key] = select.value;
            });
            controlHost.appendChild(select);
          } else {
            const input = document.createElement('input');
            input.type = 'text';
            input.value = value === null || value === undefined ? '' : String(value);
            input.addEventListener('input', () => {
              item[sub.key] = input.value;
            });
            controlHost.appendChild(input);
          }
          inner.appendChild(row);
        });
        channel.appendChild(inner);
        valueCol.appendChild(channel);
      });
    }

    function renderStructuredMapping(field) {
      const isUplink = field.type === 'uplink_mapping';
      const holder = document.createElement('div');
      holder.className = 'kv-row';
      holder.innerHTML = `
        <div class="key-col">
          <code>${field.key}</code>
          <div class="hint">${fieldHintText(field)}</div>
        </div>
        <div class="value-col"></div>
      `;
      const valueCol = holder.querySelector('.value-col');
      const card = document.createElement('div');
      card.className = 'simulation-card';

      const scalarGrid = document.createElement('div');
      scalarGrid.className = 'simulation-grid';
      for (const sub of (field.scalar_fields || [])) {
        const row = document.createElement('div');
        row.className = 'field';
        row.innerHTML = `
          <label>${sub.key}</label>
          <div class="hint">${fieldHintText(sub)}</div>
        `;
        renderScalarControl(row, sub);
        scalarGrid.appendChild(row);
      }
      card.appendChild(scalarGrid);

      for (const listField of (field.list_fields || [])) {
        listField.items = Array.isArray(listField.items) ? listField.items : [];
        const listBlock = document.createElement('div');
        listBlock.className = 'simulation-list';
        const head = document.createElement('div');
        head.className = 'simulation-list-head';
        head.innerHTML = `
          <div>
            <code>${listField.key}</code>
            <div class="hint">${listField.display_comment || listField.comment || ''}</div>
          </div>
        `;
        const addBtn = document.createElement('button');
        addBtn.className = 'btn';
        addBtn.textContent = 'Add';
        addBtn.addEventListener('click', () => {
          listField.items.push({ ...(listField.default_item || {}) });
          listField.is_set = true;
          renderSections();
        });
        head.appendChild(addBtn);
        listBlock.appendChild(head);

        const itemsHost = document.createElement('div');
        itemsHost.className = 'planner-block-list';
        listField.items.forEach((item, index) => {
          const block = document.createElement('div');
          block.className = 'planner-block';
          const blockHead = document.createElement('div');
          blockHead.className = 'planner-block-head';
          const strong = document.createElement('strong');
          strong.textContent = `${listField.key}[${index}]`;
          const removeBtn = document.createElement('button');
          removeBtn.className = 'btn danger';
          removeBtn.textContent = 'Delete';
          removeBtn.addEventListener('click', () => {
            listField.items.splice(index, 1);
            listField.is_set = true;
            renderSections();
          });
          blockHead.appendChild(strong);
          blockHead.appendChild(removeBtn);
          block.appendChild(blockHead);
          const grid = document.createElement('div');
          grid.className = 'planner-block-grid';
          for (const sub of (listField.item_fields || [])) {
            const cell = document.createElement('div');
            cell.innerHTML = `<label>${sub.key}</label>`;
            const input = document.createElement('input');
            input.type = 'text';
            input.value = item[sub.key] === null || item[sub.key] === undefined ? '' : String(item[sub.key]);
            input.addEventListener('input', () => {
              item[sub.key] = input.value;
              listField.is_set = true;
            });
            cell.appendChild(input);
            if (sub.display_comment || sub.comment) {
              const hint = document.createElement('div');
              hint.className = 'hint';
              hint.textContent = sub.display_comment || sub.comment;
              cell.appendChild(hint);
            }
            grid.appendChild(cell);
          }
          block.appendChild(grid);
          itemsHost.appendChild(block);
        });
        if (!listField.items.length) {
          const empty = document.createElement('div');
          empty.className = 'planner-empty';
          empty.textContent = 'No entries';
          itemsHost.appendChild(empty);
        }
        listBlock.appendChild(itemsHost);
        card.appendChild(listBlock);
      }

      const extraRow = document.createElement('div');
      extraRow.className = 'kv-row';
      extraRow.innerHTML = `
        <div class="key-col">
          <code>extra</code>
          <div class="hint">Additional ${isUplink ? 'uplink' : 'simulation'} keys not represented above.</div>
        </div>
        <div class="value-col"></div>
      `;
      const extraHost = extraRow.querySelector('.value-col');
      const extraTextarea = document.createElement('textarea');
      extraTextarea.value = field.extra_text || '';
      extraTextarea.rows = Math.max(3, Math.min(10, extraTextarea.value.split('\n').length + 1));
      extraTextarea.addEventListener('input', () => {
        field.extra_text = extraTextarea.value;
      });
      extraHost.appendChild(extraTextarea);
      card.appendChild(extraRow);

      valueCol.appendChild(card);
      return holder;
    }

    function renderUplinkMappingRows(field) {
      const fragment = document.createDocumentFragment();
      for (const sub of (field.scalar_fields || [])) {
        if (shouldHideStructuredScalar(field, sub)) continue;
        const row = document.createElement('div');
        row.className = 'kv-row';
        row.innerHTML = `
          <div class="key-col">
            <code>${sub.key}</code>
            <div class="hint">${fieldHintText(sub)}</div>
          </div>
          <div class="value-col"></div>
        `;
        renderScalarControl(row.querySelector('.value-col'), sub, (changedField) => {
          if (changedField.key === 'debug_self_channel') {
            renderSections();
          }
        });
        fragment.appendChild(row);
      }
      if (field.extra_text) {
        const row = document.createElement('div');
        row.className = 'kv-row';
        row.innerHTML = `
          <div class="key-col">
            <code>extra</code>
            <div class="hint">Additional uplink keys not represented above.</div>
          </div>
          <div class="value-col"></div>
        `;
        const textarea = document.createElement('textarea');
        textarea.value = field.extra_text || '';
        textarea.rows = Math.max(3, Math.min(10, textarea.value.split('\n').length + 1));
        textarea.addEventListener('input', () => {
          field.extra_text = textarea.value;
        });
        row.querySelector('.value-col').appendChild(textarea);
        fragment.appendChild(row);
      }
      return fragment;
    }

    function renderSimulationMappingRows(field) {
      const fragment = document.createDocumentFragment();
      for (const sub of (field.scalar_fields || [])) {
        if (shouldHideStructuredScalar(field, sub)) continue;
        const row = document.createElement('div');
        row.className = 'kv-row';
        row.innerHTML = `
          <div class="key-col">
            <code>simulation.${sub.key}</code>
            <div class="hint">${fieldHintText(sub)}</div>
          </div>
          <div class="value-col"></div>
        `;
        renderScalarControl(row.querySelector('.value-col'), sub, (changedField) => {
          if (changedField.key === 'snr_control_enable') {
            renderSections();
          }
        });
        fragment.appendChild(row);
      }

      for (const listField of (field.list_fields || [])) {
        listField.items = Array.isArray(listField.items) ? listField.items : [];
        const row = document.createElement('div');
        row.className = 'kv-row';
        row.innerHTML = `
          <div class="key-col">
            <code>simulation.${listField.key}</code>
            <div class="hint">${listField.display_comment || listField.comment || ''}</div>
          </div>
          <div class="value-col"></div>
        `;
        const valueCol = row.querySelector('.value-col');
        const head = document.createElement('div');
        head.className = 'simulation-list-head';
        const addBtn = document.createElement('button');
        addBtn.className = 'btn';
        addBtn.textContent = 'Add';
        addBtn.addEventListener('click', () => {
          listField.items.push({ ...(listField.default_item || {}) });
          listField.is_set = true;
          renderSections();
        });
        head.appendChild(addBtn);
        valueCol.appendChild(head);

        const itemsHost = document.createElement('div');
        itemsHost.className = 'planner-block-list';
        listField.items.forEach((item, index) => {
          const block = document.createElement('div');
          block.className = 'planner-block';
          const blockHead = document.createElement('div');
          blockHead.className = 'planner-block-head';
          const strong = document.createElement('strong');
          strong.textContent = `${listField.key}[${index}]`;
          const removeBtn = document.createElement('button');
          removeBtn.className = 'btn danger';
          removeBtn.textContent = 'Delete';
          removeBtn.addEventListener('click', () => {
            listField.items.splice(index, 1);
            listField.is_set = true;
            renderSections();
          });
          blockHead.appendChild(strong);
          blockHead.appendChild(removeBtn);
          block.appendChild(blockHead);
          const grid = document.createElement('div');
          grid.className = 'planner-block-grid';
          for (const sub of (listField.item_fields || [])) {
            const cell = document.createElement('div');
            cell.innerHTML = `<label>${sub.key}</label>`;
            const input = document.createElement('input');
            input.type = 'text';
            input.value = item[sub.key] === null || item[sub.key] === undefined ? '' : String(item[sub.key]);
            input.addEventListener('input', () => {
              item[sub.key] = input.value;
              listField.is_set = true;
            });
            cell.appendChild(input);
            if (sub.display_comment || sub.comment) {
              const hint = document.createElement('div');
              hint.className = 'hint';
              hint.textContent = sub.display_comment || sub.comment;
              cell.appendChild(hint);
            }
            grid.appendChild(cell);
          }
          block.appendChild(grid);
          itemsHost.appendChild(block);
        });
        if (!listField.items.length) {
          const empty = document.createElement('div');
          empty.className = 'planner-empty';
          empty.textContent = 'No entries';
          itemsHost.appendChild(empty);
        }
        valueCol.appendChild(itemsHost);
        fragment.appendChild(row);
      }

      if (field.extra_text) {
        const row = document.createElement('div');
        row.className = 'kv-row';
        row.innerHTML = `
          <div class="key-col">
            <code>simulation.extra</code>
            <div class="hint">Additional simulation keys not represented above.</div>
          </div>
          <div class="value-col"></div>
        `;
        const textarea = document.createElement('textarea');
        textarea.value = field.extra_text || '';
        textarea.rows = Math.max(3, Math.min(10, textarea.value.split('\n').length + 1));
        textarea.addEventListener('input', () => {
          field.extra_text = textarea.value;
        });
        row.querySelector('.value-col').appendChild(textarea);
        fragment.appendChild(row);
      }
      return fragment;
    }

	    function renderSections() {
	      const model = currentModel();
	      if (!model) return;
	      ensureSensingChannelItems();
	      configSections.innerHTML = '';

        const fieldSortOrder = {
          enable_uplink: 0,
          duplex_mode: 1,
          uplink_idle_waveform: 2,
          uplink: 3,
          equalizer_mode: 4,
          channel_tracking_mode: 5,
          equalizer_mag_floor: 6,
          channel_tracking_min_pilot_snr: 7,
          rx_gain: 8,
          uplink_rx_channel: 9,
          uplink_rx_wire_format: 10,
          uplink_rx_device_args: 11,
          uplink_rx_clock_source: 12,
          uplink_rx_time_source: 13,
          bs_dl_ul_timing_diff: 14,
          ue_timing_advance: 15,
          uplink_cpu_cores: 16,
          data_resource_blocks: 0,
          sensing_mask_blocks: 1,
        };

      const showSimulation = radioBackendValue(model) === 'sim';
	      for (const section of model.sections) {
        const visibleFields = section.fields.filter((field) => {
          if (field.type === 'simulation_mapping') return showSimulation;
          if (shouldHideFieldInSim(section, field, showSimulation)) return false;
          return !shouldHideFieldByDependency(model, field);
        });
        if (!visibleFields.length) continue;
	        const card = document.createElement('div');
	        card.className = 'section-card';
	        const title = document.createElement('h3');
        title.textContent = section.title;
        card.appendChild(title);

        const kv = document.createElement('div');
        kv.className = 'kv';

          const orderedFields = [...visibleFields].sort((lhs, rhs) => {
            const lhsOrder = fieldSortOrder[lhs.key] ?? 100;
            const rhsOrder = fieldSortOrder[rhs.key] ?? 100;
            if (lhsOrder !== rhsOrder) return lhsOrder - rhsOrder;
            return 0;
          });

		        for (const field of orderedFields) {
	          if (field.key === 'data_resource_blocks' || field.key === 'sensing_mask_blocks') {
	            kv.appendChild(renderDataResourcePlanner(field, {
	              sourceTab: currentTab,
                readOnly: true,
	            }));
	            continue;
	          }

          if (field.type === 'mapping_list') {
            const holder = document.createElement('div');
            holder.className = 'kv-row';
            holder.innerHTML = `
              <div class="key-col">
                <code>${field.key}</code>
                <div class="hint">${fieldHintText(field)}</div>
              </div>
              <div class="value-col"></div>
            `;
            const valueCol = holder.querySelector('.value-col');
            field.items.forEach((item, index) => {
              const channel = document.createElement('div');
              channel.className = 'channel-card';
              const heading = document.createElement('h4');
              heading.textContent = `${field.key}[${index}]`;
              channel.appendChild(heading);
              const inner = document.createElement('div');
              inner.className = 'kv';
              visibleMappingItemFields(field, showSimulation).forEach((sub) => {
                const row = document.createElement('div');
                row.className = 'kv-row';
                const value = item[sub.key] ?? '';
                row.innerHTML = `
                  <div class="key-col">
                    <code>${sub.key}</code>
                    <div class="hint">${sub.display_comment || sub.comment || ''}</div>
                  </div>
                  <div class="value-col"></div>
                `;
                const controlHost = row.querySelector('.value-col');
                if (sub.kind === 'bool') {
                  const wrapper = document.createElement('label');
                  wrapper.className = 'checkbox-row';
                  const input = document.createElement('input');
                  input.type = 'checkbox';
                  input.checked = Boolean(value);
                  input.addEventListener('change', () => {
                    item[sub.key] = input.checked;
                  });
                  wrapper.appendChild(input);
                  controlHost.appendChild(wrapper);
                } else if (Array.isArray(sub.options) && sub.options.length) {
                  const select = document.createElement('select');
                  for (const optionValue of sub.options) {
                    const option = document.createElement('option');
                    option.value = String(optionValue);
                    option.textContent = String(optionValue);
                    select.appendChild(option);
                  }
                  select.value = value === null || value === undefined ? '' : String(value);
                  if (!sub.options.map(String).includes(select.value) && sub.options.length) {
                    select.value = String(sub.options[0]);
                    item[sub.key] = select.value;
                  }
                  select.addEventListener('change', () => {
                    item[sub.key] = select.value;
                  });
                  controlHost.appendChild(select);
                } else {
                  const input = document.createElement('input');
                  input.type = 'text';
                  input.value = value === null || value === undefined ? '' : String(value);
                  input.addEventListener('input', () => {
                    item[sub.key] = input.value;
                  });
                  controlHost.appendChild(input);
                }
                inner.appendChild(row);
              });
              channel.appendChild(inner);
              valueCol.appendChild(channel);
            });
            kv.appendChild(holder);
            continue;
          }

          if (field.type === 'uplink_mapping') {
            kv.appendChild(renderUplinkMappingRows(field));
            continue;
          }

          if (field.type === 'simulation_mapping') {
            kv.appendChild(renderSimulationMappingRows(field));
            continue;
          }

          if (field.type === 'mapping') {
            const row = document.createElement('div');
            row.className = 'kv-row';
            row.innerHTML = `
              <div class="key-col">
                <code>${field.key}</code>
                <div class="hint">${fieldHintText(field)}</div>
              </div>
              <div class="value-col"></div>
            `;
            const controlHost = row.querySelector('.value-col');
            const textarea = document.createElement('textarea');
            textarea.value = field.value_text ?? '';
            textarea.rows = Math.max(5, Math.min(16, textarea.value.split('\n').length + 1));
            if (field.default_text) {
              textarea.placeholder = field.default_text;
            }
            textarea.addEventListener('input', () => {
              field.value_text = textarea.value;
            });
            controlHost.appendChild(textarea);
            kv.appendChild(row);
            continue;
          }

          if (field.type === 'profiling_modules') {
            const row = document.createElement('div');
            row.className = 'kv-row';
            row.innerHTML = `
              <div class="key-col">
                <code>${field.key}</code>
                <div class="hint">${fieldHintText(field)}</div>
              </div>
              <div class="value-col"></div>
            `;
            const controlHost = row.querySelector('.value-col');
            const checklist = document.createElement('div');
            checklist.className = 'checklist-grid';
            const selected = new Set(Array.isArray(field.selected) ? field.selected : []);
            for (const optionItem of (field.options || [])) {
              const option = typeof optionItem === 'string' ? optionItem : optionItem.key;
              const description = typeof optionItem === 'string' ? '' : (optionItem.description || '');
              const label = document.createElement('label');
              label.className = 'checklist-item';
              const input = document.createElement('input');
              input.type = 'checkbox';
              input.checked = selected.has(option);
              input.addEventListener('change', () => {
                const current = new Set(Array.isArray(field.selected) ? field.selected : []);
                if (option === 'all') {
                  field.selected = input.checked ? ['all'] : [];
                } else {
                  current.delete('all');
                  if (input.checked) {
                    current.add(option);
                  } else {
                    current.delete(option);
                  }
                  field.selected = (field.options || [])
                    .map((item) => (typeof item === 'string' ? item : item.key))
                    .filter((item) => current.has(item) && item !== 'all');
                }
                renderSections();
              });
              label.appendChild(input);
              const copy = document.createElement('div');
              copy.className = 'checklist-copy';
              const code = document.createElement('code');
              code.textContent = option;
              copy.appendChild(code);
              if (description) {
                const hint = document.createElement('div');
                hint.className = 'hint';
                hint.textContent = description;
                copy.appendChild(hint);
              }
              label.appendChild(copy);
              checklist.appendChild(label);
            }
            controlHost.appendChild(checklist);
            kv.appendChild(row);
            continue;
          }

          const row = document.createElement('div');
          row.className = 'kv-row';
          row.innerHTML = `
            <div class="key-col">
              <code>${field.key}</code>
              <div class="hint">${fieldHintText(field)}</div>
            </div>
            <div class="value-col"></div>
          `;
          const controlHost = row.querySelector('.value-col');
          if (field.kind === 'bool') {
            const wrapper = document.createElement('label');
            wrapper.className = 'checkbox-row';
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = Boolean(field.value);
            input.addEventListener('change', () => {
              field.value = input.checked;
              field.value_text = input.checked ? 'true' : 'false';
              if ([
                'enable_uplink',
                'enable_sec_sync_symbol',
                'enable_cfo_training_sequence',
                'measurement_enable',
                'rx_agc_enable',
                'hardware_sync',
                'akf_enable',
                'enable_bi_sensing',
                'enable_backend_sensing_processing',
              ].includes(field.key)) {
                renderSections();
                return;
              }
	              if (field.key === 'sensing_rx_channel_count') {
	                renderSections();
	                updateRuntimeOptionControls();
	              }
            });
            wrapper.appendChild(input);
            controlHost.appendChild(wrapper);
          } else if (Array.isArray(field.options) && field.options.length) {
            const select = document.createElement('select');
            if (field.optional) {
              const blank = document.createElement('option');
              blank.value = '';
              blank.textContent = '';
              select.appendChild(blank);
            }
            for (const optionValue of field.options) {
              const option = document.createElement('option');
              option.value = String(optionValue);
              option.textContent = String(optionValue);
              select.appendChild(option);
            }
            select.value = field.value_text ?? '';
            if (!field.optional && !field.options.map(String).includes(select.value) && field.options.length) {
              select.value = String(field.options[0]);
              field.value_text = select.value;
            }
            select.addEventListener('change', () => {
              field.value_text = select.value;
	              if (field.key === 'sensing_rx_channel_count') {
	                field.value = select.value;
	                renderSections();
	                updateRuntimeOptionControls();
	                return;
	              }
              if (field.key === 'radio_backend') {
                field.value = select.value;
                renderSections();
                return;
              }
              if (['fft_size', 'num_symbols', 'sync_pos', 'enable_uplink', 'enable_sec_sync_symbol', 'enable_cfo_training_sequence', 'measurement_enable', 'rx_agc_enable', 'hardware_sync', 'akf_enable', 'enable_bi_sensing', 'enable_backend_sensing_processing', 'midframe_pilot_symbols', 'pilot_positions', 'duplex_mode'].includes(field.key)) {
                field.value = select.value;
                renderSections();
              }
            });
            controlHost.appendChild(select);
          } else {
            const input = document.createElement(field.type === 'flow_list' ? 'textarea' : 'input');
            if (field.type !== 'flow_list') input.type = 'text';
            input.value = field.value_text ?? '';
            if (field.default_text) {
              input.placeholder = field.default_text;
            }
            input.addEventListener('input', () => {
              field.value_text = input.value;
	              if (field.key === 'sensing_rx_channel_count') {
	                field.value = input.value;
	                renderSections();
	                updateRuntimeOptionControls();
	                return;
	              }
              if (field.key === 'radio_backend') {
                field.value = input.value;
                renderSections();
                return;
              }
              if (['fft_size', 'num_symbols', 'sync_pos', 'enable_uplink', 'enable_sec_sync_symbol', 'enable_cfo_training_sequence', 'measurement_enable', 'rx_agc_enable', 'hardware_sync', 'akf_enable', 'enable_bi_sensing', 'enable_backend_sensing_processing', 'midframe_pilot_symbols', 'pilot_positions', 'duplex_mode'].includes(field.key)) {
                field.value = input.value;
                renderSections();
              }
            });
            controlHost.appendChild(input);
            if (field.display_unit) {
              const unit = document.createElement('div');
              unit.className = 'hint';
              unit.textContent = `Display unit: ${field.display_unit}`;
              controlHost.appendChild(unit);
            }
          }
          kv.appendChild(row);
        }

        card.appendChild(kv);
        configSections.appendChild(card);
      }

	      updateRuntimeOptionControls();
	    }

    function gatherSavePayloadForTab(tabName) {
      const model = modelForTab(tabName);
      const scalars = {};
      const mappings = {};
      const mappingLists = {};
      if (!model) {
        throw new Error(`Missing config model for ${tabName}.`);
      }
      for (const section of model.sections) {
        for (const field of section.fields) {
          if (field.type === 'mapping_list') {
            if (field.key === 'data_resource_blocks' || field.key === 'sensing_mask_blocks') {
              mappingLists[field.key] = {
                mode: plannerMode(field),
                items: field.items,
              };
            } else {
              mappingLists[field.key] = field.items;
            }
          } else if (field.type === 'simulation_mapping' || field.type === 'uplink_mapping') {
            const structuredScalars = {};
            for (const sub of (field.scalar_fields || [])) {
              if (field.type === 'uplink_mapping' && shouldHideStructuredScalar(field, sub, model)) {
                continue;
              }
              structuredScalars[sub.key] = {
                value: sub.kind === 'bool' ? Boolean(sub.value) : (sub.value_text ?? ''),
                is_set: Boolean(sub.is_set),
              };
            }
            const structuredLists = {};
            for (const listField of (field.list_fields || [])) {
              structuredLists[listField.key] = {
                items: listField.items || [],
                is_set: Boolean(listField.is_set),
              };
            }
            mappings[field.key] = {
              scalars: structuredScalars,
              lists: structuredLists,
              extra_text: field.extra_text || '',
            };
          } else if (field.type === 'mapping') {
            mappings[field.key] = field.value_text ?? '';
          } else if (field.type === 'profiling_modules') {
            scalars[field.key] = Array.isArray(field.selected) ? field.selected : [];
          } else if (field.kind === 'bool') {
            scalars[field.key] = field.value;
          } else {
            scalars[field.key] = field.value_text;
          }
        }
      }
	      return {
        name: tabName,
        scalars,
        mappings,
        mapping_lists: mappingLists,
      };
    }

    function sanitizeBeforeSave(tabName) {
      const model = modelForTab(tabName);
      if (!model) return [];
      return ['data_resource_blocks', 'sensing_mask_blocks']
        .map((key) => sanitizePlannerFieldForTab(tabName, findField(model, key)))
        .filter(Boolean);
    }

    function clipMessage(result) {
      if (!result || !result.removedRe) return '';
      const symbols = result.removedSymbols?.length
        ? ` symbol ${result.removedSymbols.join(', ')}`
        : '';
      if (result.fieldKey === 'data_resource_blocks') {
        return `TDD uplink/guard overlap removed from Resource Map${symbols} (${result.removedRe} RE).`;
      }
      return `Reserved TDD/CFO overlap removed from Sensing Resource Map${symbols} (${result.removedRe} RE).`;
    }

    function clipMessages(results) {
      const list = Array.isArray(results) ? results : [results];
      return list.map(clipMessage).filter(Boolean).join(' ');
    }

    function saveWarningMessage(result) {
      if (Array.isArray(result?.warnings)) {
        return result.warnings
          .map((warning) => warning?.message)
          .filter(Boolean)
          .join(' ');
      }
      const warning = result?.warning;
      if (warning && typeof warning.message === 'string') {
        return warning.message;
      }
      return '';
    }

    function gatherSavePayload() {
      return gatherSavePayloadForTab(currentTab);
    }

	    function renderPlannerSummary(plannerField, spec = plannerSpec()) {
	      const txConfig = plannerConfig('bs');
	      const rxConfig = plannerConfig('ue');
	      const txStats = plannerStats(plannerField, txConfig);
	      const rxStats = plannerStats(plannerField, rxConfig);
	      plannerSummary.innerHTML = '';

	      const overview = document.createElement('div');
	      overview.className = 'card';
	      const loadedFrom = plannerField.dirty
	        ? (plannerField.loadedFrom && plannerField.loadedFrom !== 'shared'
	          ? plannerField.loadedFrom
	          : spec.loadedFromLabel)
	        : (plannerField.mismatch ? plannerField.loadedFrom : 'shared');
	      overview.innerHTML = `
	        <h3>${spec.summaryTitle}</h3>
	        <div class="mono">Canvas reference: BS grid (${txConfig.numSymbols} symbols x ${txConfig.fftSize} subcarriers)
Loaded from: ${loadedFrom}
TX/RX block configs currently ${plannerField.mismatch ? 'differ' : 'match'}.</div>
	      `;
      plannerSummary.appendChild(overview);

	      const targets = document.createElement('div');
	      targets.className = 'card';
	      const txError = validatePlannerForTab('bs', plannerField);
	      const rxError = validatePlannerForTab('ue', plannerField);
        const summaryHeading = 'Resource Summary';
        if (spec.fieldKey === 'sensing_mask_blocks') {
	        targets.innerHTML = `
	          <h3>${summaryHeading}</h3>
	          <div class="mono">BS ${spec.metricLabel}: ${txStats.payloadCount}/${txStats.totalCount}
BS CFO field removed on save: ${txStats.rejectedCfoTraining || 0}
BS TDD removed on save: ${txStats.rejectedTdd || 0}
BS TDD conflicts: ${txConfig.tddConflictLabels?.length ? txConfig.tddConflictLabels.join(', ') : 'none'}${txError ? `\nBS validate: ${txError}` : ''}
UE ${spec.metricLabel}: ${rxStats.payloadCount}/${rxStats.totalCount}
UE CFO field removed on save: ${rxStats.rejectedCfoTraining || 0}
UE TDD removed on save: ${rxStats.rejectedTdd || 0}
UE TDD conflicts: ${rxConfig.tddConflictLabels?.length ? rxConfig.tddConflictLabels.join(', ') : 'none'}${rxError ? `\nUE validate: ${rxError}` : ''}</div>
	        `;
        } else {
          const resourceError = txError || rxError;
	        targets.innerHTML = `
	          <h3>${summaryHeading}</h3>
	          <div class="mono">payload RE: ${txStats.payloadCount}/${txStats.totalCount}
sensing pilot RE: ${txStats.sensingPilotCount || 0}/${txStats.totalCount}
overlap ignored: ${txStats.overlaps}
payload/sensing overlap: ${txStats.crossKindOverlap || 0} (sensing pilot wins)
sync stripped: ${txStats.strippedSync}
CFO field stripped: ${txStats.strippedCfoTraining || 0}
TDD stripped: ${txStats.strippedTdd || 0}
midframe pilot stripped: ${txStats.strippedMidframePilot || 0}
pilot stripped: ${txStats.strippedPilot}
TDD conflicts: ${txConfig.tddConflictLabels?.length ? txConfig.tddConflictLabels.join(', ') : 'none'}${resourceError ? `\nvalidate: ${resourceError}` : ''}</div>
	        `;
        }
	      plannerSummary.appendChild(targets);
	    }

	    function renderPlannerPage() {
	      const spec = plannerSpec();
	      const plannerField = ensurePlannerState(false, spec);
	      plannerTitle.textContent = spec.title;
	      plannerIntro.innerHTML = spec.intro;
	      plannerHost.innerHTML = '';
	      plannerHost.appendChild(renderDataResourcePlanner(plannerField, {
	        standalone: true,
	        sourceTab: plannerField.baseTab || 'bs',
	        onChange: () => renderPlannerSummary(plannerField, spec),
	      }));
	      renderPlannerSummary(plannerField, spec);
	      plannerTargetNote.textContent = spec.note;
	    }

	    async function loadPlannerFromTab(tabName) {
	      const spec = plannerSpec();
	      await loadConfig(tabName, true);
	      const sourceField = plannerFieldForTab(tabName, spec);
	      if (!sourceField) {
	        throw new Error(`Missing ${spec.fieldKey} field in ${tabName} config model.`);
	      }
	      const plannerField = ensurePlannerState(true, spec);
	      plannerField.mode = plannerMode(sourceField);
	      plannerField.items = clonePlannerItems(plannerBlocks(sourceField));
	      plannerField.baseTab = tabName;
	      plannerField.loadedFrom = APP.tabs[tabName].label;
	      plannerField.dirty = true;
	      refreshPlannerComparisonState(spec);
	      renderPlannerPage();
	      setFlash(`Loaded ${spec.fieldKey} from ${APP.tabs[tabName].label}.`, 'ok');
	    }

	    async function applyPlannerToTab(tabName) {
	      const spec = plannerSpec();
	      await loadConfig(tabName, true);
	      const plannerField = ensurePlannerState(false, spec);
	      const validationError = validatePlannerForTab(tabName, plannerField);
	      if (validationError) {
	        throw new Error(validationError);
	      }
	      const targetField = plannerFieldForTab(tabName, spec);
	      if (!targetField) {
	        throw new Error(`Missing ${spec.fieldKey} field in ${tabName} config model.`);
	      }
	      targetField.mode = plannerField.mode;
	      targetField.items = clonePlannerItems(plannerField.items);
      const clipResult = sanitizePlannerFieldForTab(tabName, targetField);
	      const result = await api('/api/config/save', {
	        method: 'POST',
        body: JSON.stringify(gatherSavePayloadForTab(tabName)),
	      });
	      cache[tabName] = { ...cache[tabName], config: result };
      const savedTargetField = plannerFieldForTab(tabName, spec);
      if (savedTargetField) {
        plannerField.mode = plannerMode(savedTargetField);
        plannerField.items = clonePlannerItems(plannerBlocks(savedTargetField));
      }
	      plannerField.dirty = true;
	      refreshPlannerComparisonState(spec);
	      renderPlannerPage();
      const clipNote = clipMessages(clipResult) || saveWarningMessage(result);
	      setFlash(
	        `Applied ${spec.fieldKey} to ${APP.tabs[tabName].label}.${clipNote ? ` ${clipNote}` : ''}`,
	        'ok',
	      );
	    }

    async function loadConfig(tab, preserveCommand = false) {
      const data = await api(`/api/config?name=${encodeURIComponent(tab)}`);
      cache[tab] = {
        ...cache[tab],
        config: data,
      };
      if (tab !== currentTab) return;

      configPathLabel.textContent = data.path;
      mtimeLabel.textContent = data.mtime ? `mtime: ${data.mtime}` : 'mtime: -';
      fileStateLabel.textContent = data.exists ? 'file: present' : 'file: missing';
      renderSections();
      if (!preserveCommand) {
        const draftCommand = cache[tab]?.draftCommand || cache[tab]?.runtime?.command || APP.tabs[tab].default_command;
        cache[tab].draftCommand = draftCommand;
        commandInput.value = draftCommand;
      }
    }

    function renderRuntime(tab) {
      const runtime = cache[tab]?.runtime;
      if (!runtime || tab !== currentTab) return;

      const running = runtime.running;
      statusPill.classList.toggle('running', running);
      statusPill.classList.toggle('stopped', !running);
      statusText.textContent = running ? 'running' : 'stopped';
      processMeta.textContent =
        `pid: ${runtime.pid ?? '-'}\n` +
        `return code: ${runtime.returncode ?? '-'}\n` +
        `cwd: ${runtime.cwd}\n` +
        `unit: ${runtime.unit_name || '-'}\n` +
        `isolate cpu: ${runtime.isolate_enabled ? 'on' : 'off'}\n` +
        `isolated CPUs: ${runtime.cpu_spec || '-'}\n` +
        `command: ${runtime.command || '-'}`;
      logBox.textContent = runtime.logs.length ? runtime.logs.join('\n') : '(no log output yet)';
      logBox.scrollTop = logBox.scrollHeight;
      if (!running && !commandInput.matches(':focus')) {
        const draftCommand = cache[tab]?.draftCommand || runtime.command || APP.tabs[tab].default_command;
        commandInput.value = draftCommand;
      }
    }

    async function refreshRuntime(tab) {
      const data = await api(`/api/process?name=${encodeURIComponent(tab)}`);
      cache[tab] = { ...cache[tab], runtime: data };
      renderRuntime(tab);
    }

    async function saveCurrentTab(showMessage = true) {
      const clipResult = sanitizeBeforeSave(currentTab);
      const result = await api('/api/config/save', {
        method: 'POST',
        body: JSON.stringify(gatherSavePayload()),
      });
      cache[currentTab] = { ...cache[currentTab], config: result };
      renderSections();
      if (showMessage) {
        const clipNote = clipMessages(clipResult) || saveWarningMessage(result);
        setFlash(clipNote ? `Saved successfully. ${clipNote}` : 'Saved successfully.', 'ok');
      }
    }

	    async function switchTab(tab) {
	      currentTab = tab;
	      updateTabVisuals();
	      if (specialPlannerTabs.has(tab)) {
	        await Promise.all([
	          loadConfig('bs', true),
	          loadConfig('ue', true),
	        ]);
	        ensurePlannerState(false, plannerSpec(tab));
	        renderPlannerPage();
	        setFlash('');
	        return;
      }
      populatePresets(tab);
      await loadConfig(tab);
      await refreshRuntime(tab);
      setFlash('');
    }

    presetSelect.addEventListener('change', () => {
      commandInput.value = presetSelect.value;
      cache[currentTab] = {
        ...cache[currentTab],
        draftCommand: presetSelect.value,
      };
    });

    commandInput.addEventListener('input', () => {
      cache[currentTab] = {
        ...cache[currentTab],
        draftCommand: commandInput.value,
      };
    });

    isolateToggle.addEventListener('change', () => {
      const prefs = currentRuntimePrefs(currentTab);
      prefs.isolateCpu = isolateToggle.checked;
      updateRuntimeOptionControls();
    });

    overrideIsolateToggle.addEventListener('change', () => {
      const prefs = currentRuntimePrefs(currentTab);
      prefs.overrideIsolate = overrideIsolateToggle.checked;
      if (prefs.overrideIsolate) {
        prefs.customCpuSpec = defaultCpuSpec();
      }
      updateRuntimeOptionControls();
    });

    overrideIsolateInput.addEventListener('input', () => {
      currentRuntimePrefs(currentTab).customCpuSpec = overrideIsolateInput.value;
    });

    saveBtn.addEventListener('click', async () => {
      saveBtn.disabled = true;
      try {
        await saveCurrentTab(true);
      } catch (error) {
        setFlash(error.message, 'err');
      } finally {
        saveBtn.disabled = false;
      }
    });

    reloadBtn.addEventListener('click', async () => {
      reloadBtn.disabled = true;
      try {
        await loadConfig(currentTab, true);
        setFlash('Reloaded from disk.', 'ok');
      } catch (error) {
        setFlash(error.message, 'err');
      } finally {
        reloadBtn.disabled = false;
      }
    });

    startBtn.addEventListener('click', async () => {
      startBtn.disabled = true;
      try {
        await saveCurrentTab(false);
        const runtime = await api('/api/process/start', {
          method: 'POST',
          body: JSON.stringify({
            name: currentTab,
            command: commandInput.value,
            isolate_cpu: isolateToggle.checked,
            override_isolate: overrideIsolateToggle.checked,
            isolate_cpu_spec: overrideIsolateInput.value,
            sudo_password: sudoPasswordInput.value,
          }),
        });
        cache[currentTab] = { ...cache[currentTab], runtime, draftCommand: commandInput.value };
        renderRuntime(currentTab);
        setFlash('Process started.', 'ok');
      } catch (error) {
        setFlash(error.message, 'err');
      } finally {
        startBtn.disabled = false;
      }
    });

    stopBtn.addEventListener('click', async () => {
      stopBtn.disabled = true;
      try {
        const runtime = await api('/api/process/stop', {
          method: 'POST',
          body: JSON.stringify({
            name: currentTab,
            sudo_password: sudoPasswordInput.value,
          }),
        });
        cache[currentTab] = { ...cache[currentTab], runtime };
        renderRuntime(currentTab);
        setFlash('Process stopped.', 'ok');
      } catch (error) {
        setFlash(error.message, 'err');
      } finally {
        stopBtn.disabled = false;
      }
    });

    resetIsolationBtn.addEventListener('click', async () => {
      resetIsolationBtn.disabled = true;
      try {
        const runtime = await api('/api/process/reset-isolation', {
          method: 'POST',
          body: JSON.stringify({
            name: currentTab,
            sudo_password: sudoPasswordInput.value,
          }),
        });
        cache[currentTab] = { ...cache[currentTab], runtime };
        renderRuntime(currentTab);
        setFlash('CPU isolation reset.', 'ok');
      } catch (error) {
        setFlash(error.message, 'err');
      } finally {
        resetIsolationBtn.disabled = false;
      }
    });

    refreshBtn.addEventListener('click', async () => {
      refreshBtn.disabled = true;
      try {
        await refreshRuntime(currentTab);
        setFlash('Runtime status refreshed.', 'ok');
      } catch (error) {
        setFlash(error.message, 'err');
      } finally {
        refreshBtn.disabled = false;
      }
    });

	    applyPlannerTxBtn.addEventListener('click', async () => {
	      applyPlannerTxBtn.disabled = true;
	      try {
	        await applyPlannerToTab('bs');
      } catch (error) {
        setFlash(error.message, 'err');
      } finally {
        applyPlannerTxBtn.disabled = false;
	      }
	    });

	    loadPlannerTxBtn.addEventListener('click', async () => {
	      loadPlannerTxBtn.disabled = true;
	      try {
	        await loadPlannerFromTab('bs');
	      } catch (error) {
	        setFlash(error.message, 'err');
	      } finally {
	        loadPlannerTxBtn.disabled = false;
	      }
	    });

	    loadPlannerRxBtn.addEventListener('click', async () => {
	      loadPlannerRxBtn.disabled = true;
	      try {
	        await loadPlannerFromTab('ue');
	      } catch (error) {
	        setFlash(error.message, 'err');
	      } finally {
	        loadPlannerRxBtn.disabled = false;
	      }
	    });

	    applyPlannerRxBtn.addEventListener('click', async () => {
	      applyPlannerRxBtn.disabled = true;
	      try {
	        await applyPlannerToTab('ue');
      } catch (error) {
        setFlash(error.message, 'err');
      } finally {
        applyPlannerRxBtn.disabled = false;
      }
    });

    tabs.forEach((tabButton) => {
      tabButton.addEventListener('click', () => {
        switchTab(tabButton.dataset.target).catch((error) => setFlash(error.message, 'err'));
      });
    });

	    setInterval(() => {
	      if (specialPlannerTabs.has(currentTab)) return;
	      refreshRuntime(currentTab).catch(() => {});
	    }, 1500);

    switchTab(currentTab).catch((error) => setFlash(error.message, 'err'));

    /* ================================================================
       UI ergonomics layer (theme, toasts, field search, collapsible
       sections, dirty indicator, keyboard shortcuts, persisted tab).
       Self-contained and additive — does not alter existing behavior.
       ================================================================ */
    (function uiEnhancements() {
      const LS = {
        theme: 'oisac.theme',
        tab: 'oisac.tab',
        collapsed: (tab) => `oisac.collapsed.${tab}`,
      };
      const store = {
        get(key) { try { return localStorage.getItem(key); } catch (e) { return null; } },
        set(key, val) { try { localStorage.setItem(key, val); } catch (e) {} },
      };

      /* ---- Theme toggle ---- */
      const themeToggle = document.getElementById('themeToggle');
      function currentTheme() {
        return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
      }
      function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        store.set(LS.theme, theme);
        if (themeToggle) {
          themeToggle.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
          const txt = themeToggle.querySelector('.theme-toggle-text');
          if (txt) txt.textContent = theme === 'dark' ? 'Dark' : 'Light';
        }
      }
      applyTheme(currentTheme());
      if (themeToggle) {
        themeToggle.addEventListener('click', () => {
          applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
        });
      }

      /* ---- Toast notifications ---- */
      const toastStack = document.getElementById('toastStack');
      let lastToast = { text: '', at: 0 };
      function showToast(message, kind = 'info') {
        if (!toastStack || !message) return;
        const now = (performance && performance.now) ? performance.now() : 0;
        if (message === lastToast.text && now - lastToast.at < 600) return;
        lastToast = { text: message, at: now };
        const toast = document.createElement('div');
        toast.className = `toast ${kind}`;
        toast.setAttribute('role', kind === 'err' ? 'alert' : 'status');
        const icon = document.createElement('span');
        icon.className = 'toast-icon';
        icon.textContent = kind === 'ok' ? '✓' : kind === 'err' ? '!' : 'i';
        const body = document.createElement('div');
        body.className = 'toast-body';
        body.textContent = message;
        toast.append(icon, body);
        toastStack.appendChild(toast);
        requestAnimationFrame(() => toast.classList.add('show'));
        const ttl = kind === 'err' ? 6000 : 3200;
        setTimeout(() => {
          toast.classList.remove('show');
          setTimeout(() => toast.remove(), 220);
        }, ttl);
      }

      /* ---- Dirty / unsaved-changes indicator ---- */
      const saveBtn = document.getElementById('saveBtn');
      const configSections = document.getElementById('configSections');
      function markDirty() { if (saveBtn) saveBtn.classList.add('is-dirty'); }
      function clearDirty() { if (saveBtn) saveBtn.classList.remove('is-dirty'); }
      if (configSections) {
        configSections.addEventListener('input', markDirty, true);
        configSections.addEventListener('change', markDirty, true);
      }

      /* ---- Route ok/err flashes to toasts + manage dirty state ---- */
      if (typeof setFlash === 'function') {
        const origSetFlash = setFlash;
        setFlash = function (message, kind = '') {
          origSetFlash(message, kind);
          if (message && (kind === 'ok' || kind === 'err')) showToast(message, kind);
          if (kind === 'ok' && /\b(saved|reload)/i.test(message)) clearDirty();
        };
      }

      /* ---- Collapsible config sections (persisted per tab) ---- */
      function collapsedSet(tab) {
        try { return new Set(JSON.parse(store.get(LS.collapsed(tab)) || '[]')); }
        catch (e) { return new Set(); }
      }
      function saveCollapsed(tab, set) {
        store.set(LS.collapsed(tab), JSON.stringify([...set]));
      }
      function decorateSections() {
        if (!configSections) return;
        const tab = currentTab;
        const collapsed = collapsedSet(tab);
        const cards = configSections.querySelectorAll('.section-card');
        cards.forEach((card) => {
          const h3 = card.querySelector('h3');
          if (!h3 || h3.dataset.collapsible === '1') {
            // already wired; just re-sync collapsed state
            if (h3) {
              const key = h3.textContent.trim();
              card.classList.toggle('collapsed', collapsed.has(key));
              h3.setAttribute('aria-expanded', collapsed.has(key) ? 'false' : 'true');
            }
            return;
          }
          h3.dataset.collapsible = '1';
          const key = h3.textContent.trim();
          h3.setAttribute('role', 'button');
          h3.setAttribute('tabindex', '0');
          h3.setAttribute('aria-expanded', collapsed.has(key) ? 'false' : 'true');
          card.classList.toggle('collapsed', collapsed.has(key));
          const toggle = () => {
            const set = collapsedSet(tab);
            const isCollapsed = card.classList.toggle('collapsed');
            if (isCollapsed) set.add(key); else set.delete(key);
            h3.setAttribute('aria-expanded', isCollapsed ? 'false' : 'true');
            saveCollapsed(tab, set);
          };
          h3.addEventListener('click', toggle);
          h3.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
          });
        });
      }

      /* ---- Field search / filter ---- */
      const searchInput = document.getElementById('configSearch');
      const searchCount = document.getElementById('configSearchCount');
      function applySearch() {
        if (!configSections) return;
        const q = (searchInput ? searchInput.value : '').trim().toLowerCase();
        const cards = configSections.querySelectorAll('.section-card');
        let total = 0, shown = 0;
        // remove any prior "no results" placeholder
        const existingEmpty = configSections.querySelector('.search-empty');
        if (existingEmpty) existingEmpty.remove();
        cards.forEach((card) => {
          const h3 = card.querySelector('h3');
          const title = h3 ? h3.textContent.toLowerCase() : '';
          const titleHit = q && title.includes(q);
          const rows = card.querySelectorAll('.kv-row');
          let cardShown = 0;
          rows.forEach((row) => {
            total += 1;
            const hit = !q || titleHit || row.textContent.toLowerCase().includes(q);
            row.classList.toggle('hidden', !hit);
            row.classList.toggle('search-hit', !!q && hit && !titleHit);
            if (hit) { cardShown += 1; shown += 1; }
          });
          // when searching, reveal collapsed sections that have matches
          if (q) {
            card.classList.toggle('collapsed', cardShown === 0 && !titleHit);
            card.classList.toggle('hidden', cardShown === 0 && !titleHit);
          } else {
            card.classList.remove('hidden');
            row_restore(card);
          }
        });
        if (searchCount) {
          searchCount.textContent = q ? `${shown} of ${total} fields` : '';
        }
        if (q && shown === 0) {
          const empty = document.createElement('div');
          empty.className = 'search-empty';
          empty.textContent = `No fields match “${searchInput.value.trim()}”`;
          configSections.appendChild(empty);
        }
      }
      // restore collapsed state from storage when clearing the search
      function row_restore(card) {
        const h3 = card.querySelector('h3');
        if (!h3) return;
        card.querySelectorAll('.kv-row').forEach((row) => {
          row.classList.remove('hidden', 'search-hit');
        });
        const collapsed = collapsedSet(currentTab);
        card.classList.toggle('collapsed', collapsed.has(h3.textContent.trim()));
      }
      if (searchInput) {
        searchInput.addEventListener('input', applySearch);
        searchInput.addEventListener('search', applySearch);
      }

      /* ---- React to config form re-renders ---- */
      if (configSections && 'MutationObserver' in window) {
        let scheduled = false;
        const observer = new MutationObserver(() => {
          if (scheduled) return;
          scheduled = true;
          requestAnimationFrame(() => {
            scheduled = false;
            decorateSections();
            applySearch();
          });
        });
        observer.observe(configSections, { childList: true, subtree: true });
      }

      /* ---- Persisted active tab + clear search/dirty on switch ---- */
      if (typeof switchTab === 'function') {
        const origSwitchTab = switchTab;
        switchTab = function (tab) {
          store.set(LS.tab, tab);
          clearDirty();
          if (searchInput) searchInput.value = '';
          if (searchCount) searchCount.textContent = '';
          return origSwitchTab(tab);
        };
      }
      const validTabs = new Set(['bs', 'ue', 'planner', 'sensingPlanner']);
      let initialTab = null;
      try {
        const qTab = new URLSearchParams(location.search).get('tab');
        if (qTab && validTabs.has(qTab)) initialTab = qTab;
      } catch (e) {}
      if (!initialTab) {
        const storedTab = store.get(LS.tab);
        if (storedTab && validTabs.has(storedTab)) initialTab = storedTab;
      }
      if (initialTab && initialTab !== currentTab) {
        switchTab(initialTab).catch((e) => setFlash(e.message, 'err'));
      }

      /* ---- Keyboard shortcuts ---- */
      document.addEventListener('keydown', (e) => {
        const k = e.key.toLowerCase();
        // Ctrl/Cmd+S -> save current config (when the config view is visible)
        if ((e.ctrlKey || e.metaKey) && k === 's') {
          const view = document.getElementById('configRuntimeView');
          if (view && !view.classList.contains('hidden') && saveBtn && !saveBtn.disabled) {
            e.preventDefault();
            saveBtn.click();
          }
          return;
        }
        // "/" focuses the field search (unless already typing in a field)
        if (k === '/' && !e.ctrlKey && !e.metaKey && !e.altKey) {
          const t = e.target;
          const typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
          const view = document.getElementById('configRuntimeView');
          if (!typing && searchInput && view && !view.classList.contains('hidden')) {
            e.preventDefault();
            searchInput.focus();
          }
        }
      });
    })();
