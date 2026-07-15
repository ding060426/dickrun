(function attachHotwordSettings(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DiTingHotwordSettings = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createHotwordSettings() {
  const DOMAIN_PRESETS = Object.freeze([
    {
      id: 'technology',
      name: '技术研发',
      description: 'AI、模型、研发协作与技术评审',
      words: ['人工智能', '大语言模型', '机器学习', '深度学习', 'Transformer', 'BERT', 'OpenAI', '微调', '推理', '向量数据库', '多模态', 'API'],
    },
    {
      id: 'product',
      name: '产品运营',
      description: '需求、增长、数据指标与版本迭代',
      words: ['产品需求', '用户增长', '转化率', '留存率', '活跃用户', '用户画像', 'A/B测试', '埋点', '复盘', '迭代', '上线', 'OKR'],
    },
    {
      id: 'business',
      name: '商务会议',
      description: '客户、合同、报价与项目交付',
      words: ['商业模式', '合同', '报价', '交付', '客户需求', '招投标', '预算', '成本', '营收', '毛利率', '合作伙伴', '回款'],
    },
    {
      id: 'finance',
      name: '金融财务',
      description: '财务指标、投资分析与风险控制',
      words: ['资产配置', '现金流', '净利润', '市盈率', '收益率', '风险敞口', '利率', '汇率', '债券', '基金', '合规', '风控'],
    },
    {
      id: 'medical',
      name: '医疗健康',
      description: '临床、诊疗、用药与患者随访',
      words: ['电子病历', '临床试验', '诊断', '治疗方案', '医学影像', '检验指标', '用药', '不良反应', '随访', '临床指南', '医保', '患者'],
    },
  ]);

  const clamp = (value, fallback = 5) => {
    const parsed = Number(value);
    const safe = Number.isFinite(parsed) ? parsed : fallback;
    return Math.round(Math.min(20, Math.max(0.1, safe)) * 1000) / 1000;
  };

  const containsCjk = value => /[\u3400-\u9fff\uf900-\ufaff]/.test(value || '');

  function scoreForNewWord(text, defaultScore) {
    const score = clamp(defaultScore);
    return containsCjk(text) ? score : Math.min(score, 2.5);
  }

  function normalizeSettings(payload = {}) {
    const defaultScore = clamp(payload.default_score);
    const words = Array.isArray(payload.words) ? payload.words : [];
    return {
      enabled: payload.enabled !== false,
      fuzzy_pinyin_enabled: payload.fuzzy_pinyin_enabled !== false,
      default_score: defaultScore,
      words: words.map(item => ({
        text: String(item?.text || '').trim(),
        score: clamp(item?.score, scoreForNewWord(item?.text, defaultScore)),
        enabled: item?.enabled !== false,
      })).filter(item => item.text),
    };
  }

  function buildPayload(settings, rows) {
    return normalizeSettings({
      ...settings,
      words: rows,
    });
  }

  function getDomainPresets() {
    return DOMAIN_PRESETS.map(preset => ({
      ...preset,
      words: [...preset.words],
      word_count: preset.words.length,
    }));
  }

  function applyDomainPreset(rows, presetId, defaultScore) {
    const preset = DOMAIN_PRESETS.find(item => item.id === presetId);
    const words = (Array.isArray(rows) ? rows : []).map(item => ({ ...item }));
    if (!preset) {
      return { preset: null, words, added_count: 0, existing_count: 0 };
    }

    const existing = new Set(
      words.map(item => String(item?.text || '').trim().toLocaleLowerCase()).filter(Boolean),
    );
    let addedCount = 0;
    let existingCount = 0;
    preset.words.forEach(text => {
      const key = text.toLocaleLowerCase();
      if (existing.has(key)) {
        existingCount += 1;
        return;
      }
      words.push({
        text,
        score: scoreForNewWord(text, defaultScore),
        enabled: true,
      });
      existing.add(key);
      addedCount += 1;
    });

    return {
      preset: { id: preset.id, name: preset.name },
      words,
      added_count: addedCount,
      existing_count: existingCount,
    };
  }

  return {
    clamp,
    containsCjk,
    scoreForNewWord,
    normalizeSettings,
    buildPayload,
    getDomainPresets,
    applyDomainPreset,
  };
}));
