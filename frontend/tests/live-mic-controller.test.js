const assert = require('node:assert/strict');
const { MicState, createTransition, LiveMicController } = require('../js/live-mic-controller.js');

const transition = createTransition();
assert.equal(transition.state, MicState.IDLE);
transition.set(MicState.CONNECTING);
transition.set(MicState.RECORDING);
assert.deepEqual(transition.history, [MicState.IDLE, MicState.CONNECTING, MicState.RECORDING]);
assert.equal(transition.isActive(), true);

const events = new Map();
const client = {
  stopped: false,
  on(name, handler) { events.set(name, handler); },
  stop() { this.stopped = true; },
};
const states = [];
const controller = new LiveMicController({ client, onState: state => states.push(state) });
const first = controller.start();
assert.equal(states.at(-1), MicState.CONNECTING);
assert.equal(controller.markRecording(first), true);
assert.equal(states.at(-1), MicState.RECORDING);
assert.equal(controller.stop(first), true);
assert.equal(client.stopped, true);
const second = controller.start();
assert.equal(controller.cleanup(first), false, 'old session cleanup must not affect new session');
assert.equal(controller.cleanup(second), true);
assert.equal(controller.cleanupCalls, 1);

console.log('live mic controller tests passed');
