const assert = require('node:assert/strict');

const { getVisibleLogicFlags } = require('../logic-flags.js');


const structuredConflict = { type: 'data_conflict', message: '结构化数据冲突' };
const visible = getVisibleLogicFlags({
  logic_flags: [
    { type: 'self_contradiction', message: '同一说话人数据可能存在矛盾 (数值差异 304%)' },
    structuredConflict,
  ],
});

assert.deepEqual(visible, [structuredConflict]);
assert.deepEqual(getVisibleLogicFlags({}), []);

console.log('logic-flags tests passed');
