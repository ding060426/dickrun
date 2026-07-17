const assert = require('node:assert/strict');
const { clampPanelPosition, detectDockSide } = require('../js/mic-orb.js');

assert.deepEqual(
  clampPanelPosition(900, -10, { width: 800, height: 600 }, { width: 240, height: 150 }),
  { x: 560, y: 0 },
);
assert.equal(detectDockSide(5, { width: 800 }, 24), 'left');
assert.equal(detectDockSide(790, { width: 800 }, 24), 'right');
assert.equal(detectDockSide(400, { width: 800 }, 24), null);

console.log('mic orb tests passed');
