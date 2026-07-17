const assert = require('node:assert/strict');
const { recordTitle } = require('../js/records-controller.js');

assert.equal(recordTitle({ title: '会议' }), '会议');
assert.equal(recordTitle({ filename: 'a.wav' }), 'a.wav');
assert.equal(recordTitle({}), 'Untitled meeting');
console.log('records controller tests passed');
