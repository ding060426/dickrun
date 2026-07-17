const assert = require('node:assert/strict');
const { createApiClient } = require('../js/api-client.js');

(async () => {
  const calls = [];
  const client = createApiClient({
    apiBase: 'http://api',
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return { ok: true, json: async () => ({ ok: true, url }) };
    },
  });
  const status = await client.uploadStatus('abc');
  assert.equal(status.url, 'http://api/api/audio/upload/abc/status');
  await client.cancelUpload('abc');
  assert.equal(calls[1].options.method, 'POST');
  console.log('api client tests passed');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
