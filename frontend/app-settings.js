(function attachAppSettings(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DiTingAppSettings = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createAppSettings() {
  const ASR_PROFILES = ['low-latency', 'balanced', 'meeting', 'quality'];
  const ASR_PROVIDERS = ['xasr', 'qwen3'];
  const QWEN3_DEVICES = ['auto', 'cuda:0', 'cpu'];
  const QWEN3_DTYPES = ['auto', 'bfloat16', 'float16', 'float32'];
  const LIVE_PROFILES = ['meeting', 'dictation', 'oncall'];

  function number(value, fallback, minimum, maximum, integer = false) {
    const parsed = Number(value);
    const valid = Number.isFinite(parsed) ? parsed : fallback;
    const clamped = Math.min(maximum, Math.max(minimum, valid));
    return integer ? Math.round(clamped) : Math.round(clamped * 1000) / 1000;
  }

  function choice(value, choices, fallback) {
    const selected = String(value || '').trim().toLowerCase();
    return choices.includes(selected) ? selected : fallback;
  }

  function normalizeAppSettings(payload) {
    const source = payload && typeof payload === 'object' ? payload : {};
    const recognition = source.recognition || {};
    const microphone = source.microphone || {};
    return {
      recognition: {
        asr_provider: choice(recognition.asr_provider, ASR_PROVIDERS, 'xasr'),
        qwen3_model_path: String(recognition.qwen3_model_path || '').trim().slice(0, 1024),
        qwen3_device: choice(recognition.qwen3_device, QWEN3_DEVICES, 'auto'),
        qwen3_dtype: choice(recognition.qwen3_dtype, QWEN3_DTYPES, 'auto'),
        live_asr_profile: choice(recognition.live_asr_profile, ASR_PROFILES, 'meeting'),
        final_asr_profile: choice(recognition.final_asr_profile, ASR_PROFILES, 'meeting'),
        final_transcription_enabled: recognition.final_transcription_enabled !== false,
        file_vad_provider: 'silero',
        file_vad_threshold: number(recognition.file_vad_threshold, 0.5, 0.05, 0.95),
        file_vad_min_silence: number(recognition.file_vad_min_silence, 0.5, 0.1, 3),
        file_vad_min_speech: number(recognition.file_vad_min_speech, 0.2, 0.05, 2),
        file_vad_pre_padding_ms: number(recognition.file_vad_pre_padding_ms, 250, 0, 2000, true),
        file_vad_post_padding_ms: number(recognition.file_vad_post_padding_ms, 450, 0, 3000, true),
      },
      microphone: {
        device_id: String(microphone.device_id || '').slice(0, 512),
        live_profile: choice(microphone.live_profile, LIVE_PROFILES, 'meeting'),
        vad_gating: microphone.vad_gating === true,
        echo_cancellation: microphone.echo_cancellation !== false,
        noise_suppression: microphone.noise_suppression === true,
        auto_gain_control: microphone.auto_gain_control === true,
        pre_roll_ms: number(microphone.pre_roll_ms, 700, 0, 3000, true),
        endpoint_grace_ms: number(microphone.endpoint_grace_ms, 800, 0, 5000, true),
        tail_pad_ms: number(microphone.tail_pad_ms, 1000, 0, 3000, true),
        vad_threshold: number(microphone.vad_threshold, 0.5, 0.05, 0.95),
        vad_min_silence: number(microphone.vad_min_silence, 0.5, 0.1, 3),
        vad_min_speech: number(microphone.vad_min_speech, 0.2, 0.05, 2),
      },
      hotwords: source.hotwords || null,
      models: source.models || {},
    };
  }

  function buildMediaConstraints(microphone) {
    const settings = normalizeAppSettings({ microphone }).microphone;
    const audio = {
      channelCount: 1,
      echoCancellation: settings.echo_cancellation,
      noiseSuppression: settings.noise_suppression,
      autoGainControl: settings.auto_gain_control,
    };
    if (settings.device_id) audio.deviceId = { exact: settings.device_id };
    return { audio };
  }

  return {
    ASR_PROFILES,
    ASR_PROVIDERS,
    buildMediaConstraints,
    normalizeAppSettings,
    number,
  };
}));
