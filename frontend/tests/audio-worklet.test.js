const assert = require('node:assert/strict');


let RegisteredProcessor = null;
global.sampleRate = 48000;
global.AudioWorkletProcessor = class {
  constructor() {
    this.messages = [];
    this.port = {
      onmessage: null,
      postMessage: message => this.messages.push(message),
    };
  }
};
global.registerProcessor = (_name, Processor) => {
  RegisteredProcessor = Processor;
};

require('../audio-worklet.js');

const processor = new RegisteredProcessor();
processor.process([[new Float32Array(2048).fill(0.25)]]);
const level = processor.messages.find(message => message.type === 'audio.level');

assert.ok(level, 'worklet should publish a throttled audio level measurement');
assert.ok(Math.abs(level.rms - 0.25) < 0.001, `unexpected RMS: ${level.rms}`);
assert.ok(Math.abs(level.peak - 0.25) < 0.001, `unexpected peak: ${level.peak}`);
console.log('audio-worklet tests passed');
