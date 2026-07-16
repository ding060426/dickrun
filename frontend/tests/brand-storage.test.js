const assert = require('node:assert/strict');

const {
  AUTH_TOKEN_KEY,
  getAuthToken,
  setAuthToken,
  clearAuthToken,
} = require('../brand-storage.js');

const values = new Map([['diting_auth_token', 'legacy-token']]);
const storage = {
  getItem: key => values.get(key) ?? null,
  setItem: (key, value) => values.set(key, value),
  removeItem: key => values.delete(key),
};

assert.equal(AUTH_TOKEN_KEY, 'huiwu:auth-token');
assert.equal(getAuthToken(storage), 'legacy-token');
assert.equal(values.get(AUTH_TOKEN_KEY), 'legacy-token');
assert.equal(values.has('diting_auth_token'), false);

setAuthToken('new-token', storage);
assert.equal(getAuthToken(storage), 'new-token');
clearAuthToken(storage);
assert.equal(getAuthToken(storage), '');

console.log('brand-storage tests passed');
