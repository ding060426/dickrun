const assert = require('node:assert/strict');
const { LiveMicClient } = require('../js/live-mic-client.js');

class FakeWebSocket {
  static instances = [];
  constructor(url) {
    this.url = url;
    this.readyState = 1;
    this.sent = [];
    FakeWebSocket.instances.push(this);
  }
  send(payload) { this.sent.push(payload); }
  close() { this.readyState = 3; }
}

const encodedFrames = [];
const client = new LiveMicClient({
  wsBase: 'ws://127.0.0.1:8765',
  WebSocketImpl: FakeWebSocket,
  encodePcmFrame: (seq, payload) => { encodedFrames.push({ seq, payload }); return `frame-${seq}`; },
  setTimeoutImpl: () => 1,
  clearTimeoutImpl: () => {},
});
const configured = [];
client.on('configured', data => configured.push(data));
client.connect({ browser_sample_rate: 48000, profile: 'meeting', stream_id: 's1' });
const ws = FakeWebSocket.instances[0];
assert.equal(ws.url, 'ws://127.0.0.1:8765/ws/live');
ws.onmessage({ data: JSON.stringify({ type: 'ready' }) });
assert.equal(JSON.parse(ws.sent[0]).action, 'configure');
ws.onmessage({ data: JSON.stringify({ type: 'configured', data: { stream_id: 's2' } }) });
assert.equal(configured[0].stream_id, 's2');
assert.equal(client.sendFrame(new Uint8Array([1, 2]).buffer), true);
assert.equal(encodedFrames[0].seq, 0);
assert.equal(ws.sent.at(-1), 'frame-0');
assert.equal(client.stop(), true);
assert.equal(JSON.parse(ws.sent.at(-1)).action, 'stop');

console.log('live mic client tests passed');
