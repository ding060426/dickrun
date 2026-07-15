(function attachManagementTranscription(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DiTingManagementTranscription = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function createApi() {
  function resolveBackend(locationLike) {
    const params = new URLSearchParams(locationLike.search || '');
    const host = params.get('apiHost') || locationLike.hostname || 'localhost';
    const port = params.get('apiPort') || '8765';
    const secure = locationLike.protocol === 'https:';
    return {
      apiBase: `${secure ? 'https:' : 'http:'}//${host}:${port}`,
      wsBase: `${secure ? 'wss:' : 'ws:'}//${host}:${port}`,
    };
  }

  function buildUploadUrl(apiBase, fileId, options = {}) {
    const params = new URLSearchParams({
      file_id: fileId,
      enable_diarization: options.enableDiarization === false ? 'false' : 'true',
    });
    if (options.numSpeakers) params.set('num_speakers', String(options.numSpeakers));
    return `${apiBase}/api/audio/upload?${params.toString()}`;
  }

  function buildAnalysisPayload({
    meetingId,
    title,
    segments = [],
  }) {
    const startOf = (segment) => Number(segment.start_sec ?? segment.start ?? 0);
    const endOf = (segment) => Number(segment.end_sec ?? segment.end ?? 0);
    const countItems = (field) => segments.reduce(
      (sum, segment) => sum + (Array.isArray(segment[field]) ? segment[field].length : 0),
      0,
    );
    return {
      meeting_id: meetingId || undefined,
      title: title || 'Untitled',
      transcript_json: segments.map((segment) => ({
        text: segment.display_text || segment.text || '',
        start: startOf(segment),
        end: endOf(segment),
        speaker: segment.speaker_name || segment.speaker_id || segment.speaker || '',
        speaker_id: segment.speaker_id || '',
        snr_db: segment.snr_db,
      })),
      segments_count: segments.length,
      duration_sec: segments.reduce(
        (sum, segment) => sum + Math.max(0, endOf(segment) - startOf(segment)),
        0,
      ),
      logic_flags_count: countItems('logic_flags'),
      low_confidence_count: 0,
      corrections_count: countItems('corrections'),
      overall_confidence: 0,
      hotwords: [],
      summary_json: {},
    };
  }

  return { buildAnalysisPayload, buildUploadUrl, resolveBackend };
});
