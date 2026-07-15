const assert = require('node:assert/strict');

const { createI18n } = require('../i18n.js');

const saved = new Map([['diting:language', 'en-US']]);
const storage = {
  getItem: key => saved.get(key) ?? null,
  setItem: (key, value) => saved.set(key, value),
};

const textNode = { dataset: { i18n: 'controls.demo' }, textContent: '' };
const parameterNode = {
  dataset: { i18n: 'upload.complete', i18nCount: '4' },
  textContent: '',
};
const placeholderNode = {
  dataset: { i18nPlaceholder: 'settings.search_placeholder' },
  placeholder: '',
};
const titleNode = {
  dataset: { i18nTitle: 'segment.rename_title' },
  title: '',
};
const ariaNode = {
  dataset: { i18nAriaLabel: 'language.label' },
  attributes: {},
  setAttribute(name, value) { this.attributes[name] = value; },
};
const document = {
  documentElement: { lang: '' },
  querySelectorAll(selector) {
    if (selector === '[data-i18n]') return [textNode, parameterNode];
    if (selector === '[data-i18n-placeholder]') return [placeholderNode];
    if (selector === '[data-i18n-title]') return [titleNode];
    if (selector === '[data-i18n-aria-label]') return [ariaNode];
    return [];
  },
};

const i18n = createI18n({ storage, document, navigatorLanguage: 'zh-CN' });
assert.equal(i18n.language, 'en-US');
i18n.apply();
assert.equal(textNode.textContent, 'Demo');
assert.equal(parameterNode.textContent, 'Complete: 4 segments');
assert.equal(placeholderNode.placeholder, 'Search settings...');
assert.equal(titleNode.title, 'Click to rename speaker');
assert.equal(ariaNode.attributes['aria-label'], 'Language');
assert.equal(document.documentElement.lang, 'en-US');

i18n.setLanguage('zh');
assert.equal(i18n.language, 'zh-CN');
assert.equal(saved.get('diting:language'), 'zh-CN');
assert.equal(textNode.textContent, '演示');
assert.equal(placeholderNode.placeholder, '搜索设置...');
assert.equal(document.documentElement.lang, 'zh-CN');
assert.equal(i18n.t('upload.complete', { count: 3 }), '完成：3 个片段');
assert.equal(i18n.t('missing.key', {}, '安全回退'), '安全回退');

console.log('i18n tests passed');
