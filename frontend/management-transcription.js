(function attachManagementTranscription(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuManagementTranscription = api;
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

  function buildRecordDetailPath(recordId) {
    return `/api/records/${encodeURIComponent(recordId)}?include_audio=true`;
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

  function buildRecordPayload({
    recordId,
    meetingId,
    title,
    sourceType = 'manual',
    sourceFilename = '',
    sourceMimeType = '',
    sourceSizeBytes = 0,
    segments = [],
    speakers = [],
  }) {
    const normalizedSegments = segments.filter(Boolean).map((segment, position) => ({
      ...segment,
      index: Number(segment.index ?? segment.segment_index ?? position + 1),
      text: segment.display_text || segment.text || '',
      raw_text: segment.raw_text || '',
      start_sec: Number(segment.start_sec ?? segment.start ?? 0),
      end_sec: Number(segment.end_sec ?? segment.end ?? 0),
      speaker_id: segment.speaker_id || '',
      speaker_name: segment.speaker_name || segment.speaker || '',
      audio_wav_base64: segment.audio_wav_base64 || null,
    }));
    const fullText = normalizedSegments.map((segment) => {
      const speaker = segment.speaker_name || segment.speaker_id || '未区分说话人';
      return segment.text ? `[${speaker}] ${segment.text}` : '';
    }).filter(Boolean).join('\n');
    const durationSec = normalizedSegments.reduce(
      (maximum, segment) => Math.max(maximum, segment.end_sec),
      0,
    );
    return {
      id: recordId || undefined,
      meeting_id: meetingId || undefined,
      title: title || '未命名会议记录',
      source_type: sourceType,
      source_filename: sourceFilename,
      source_mime_type: sourceMimeType,
      source_size_bytes: Number(sourceSizeBytes) || 0,
      full_text: fullText,
      duration_sec: durationSec,
      speakers,
      segments: normalizedSegments,
    };
  }

  return {
    buildAnalysisPayload,
    buildRecordDetailPath,
    buildRecordPayload,
    buildUploadUrl,
    resolveBackend,
  };
});
