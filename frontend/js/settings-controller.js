(function attachSettingsController(root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuSettingsController = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createSettingsController() {
  function modelStatusSummary(models = {}) {
    const profiles = models.profiles || models;
    const live = profiles.live || {};
    const finalModel = profiles.final || {};
    const diarization = models.diarization || {};
    return {
      live_profile: live.effective_profile || '',
      final_profile: finalModel.effective_profile || live.effective_profile || '',
      diarization_mode: diarization.mode || (diarization.available ? 'diarization' : 'asr_only'),
      qwen3_mode: models.providers?.qwen3?.mode || 'final_transcription_only',
    };
  }
  return { modelStatusSummary };
}));
