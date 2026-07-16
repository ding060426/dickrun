(function attachLiveProtocol(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuLiveProtocol = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createLiveProtocol() {
  const MAGIC = [0x44, 0x54, 0x50, 0x32]; // DTP2

  function encodePcmFrame(sequence, payload) {
    if (!Number.isInteger(sequence) || sequence < 0 || sequence > 0xffffffff) {
      throw new RangeError('sequence must be an unsigned 32-bit integer');
    }
    const pcm = payload instanceof ArrayBuffer
      ? new Uint8Array(payload)
      : new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
    const frame = new ArrayBuffer(8 + pcm.byteLength);
    const bytes = new Uint8Array(frame);
    bytes.set(MAGIC, 0);
    new DataView(frame).setUint32(4, sequence, true);
    bytes.set(pcm, 8);
    return frame;
  }

  return { encodePcmFrame };
}));
