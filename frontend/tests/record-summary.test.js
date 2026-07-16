const assert = require('node:assert/strict');

const { parseDownloadFilename } = require('../record-summary.js');

assert.equal(
  parseDownloadFilename(
    'attachment; filename="meeting-summary.md"; filename*=UTF-8\'\'%E7%A7%A6%E6%9C%9D%E5%88%B6%E5%BA%A6%E8%AF%BE%E7%A8%8B%E6%91%98%E8%A6%81.md',
  ),
  '秦朝制度课程摘要.md',
);
assert.equal(
  parseDownloadFilename('attachment; filename="weekly-summary.md"'),
  'weekly-summary.md',
);
assert.equal(parseDownloadFilename(''), 'meeting-summary.md');

console.log('record-summary tests passed');
