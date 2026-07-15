const assert = require('node:assert/strict');
const { encodePcmFrame } = require('../live-protocol.js');

const pcm = new Int16Array([1, -2, 32767]);
const frame = encodePcmFrame(7, pcm.buffer);
const bytes = new Uint8Array(frame);
const view = new DataView(frame);

assert.equal(new TextDecoder().decode(bytes.slice(0, 4)), 'DTP2');
assert.equal(view.getUint32(4, true), 7);
assert.deepEqual(Array.from(new Int16Array(frame, 8)), [1, -2, 32767]);
console.log('live-protocol tests passed');
