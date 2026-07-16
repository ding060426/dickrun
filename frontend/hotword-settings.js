(function attachHotwordSettings(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuHotwordSettings = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createHotwordSettings() {
  const DOMAIN_PRESETS = Object.freeze([
    {
      id: 'technology',
      name: '技术研发',
      description: 'AI、模型、研发协作与技术评审',
      words: [
        '人工智能', '大语言模型', '机器学习', '深度学习', 'Transformer', 'BERT', 'OpenAI',
        '微调', '推理', '向量数据库', '多模态', 'API', 'RAG', 'Agent', '智能体', '提示词工程',
        '知识库', '语音识别', '说话人分离', '前端', '后端', '数据库', '接口', '部署',
        '性能优化', '代码评审', '技术债', '版本发布',
      ],
    },
    {
      id: 'product',
      name: '产品运营',
      description: '需求、增长、数据指标与版本迭代',
      words: [
        '产品需求', '用户增长', '转化率', '留存率', '活跃用户', '用户画像', 'A/B测试',
        '埋点', '复盘', '迭代', '上线', 'OKR', '需求评审', '产品路线图', '用户体验',
        '原型', '交互设计', '漏斗分析', '日活', '月活', '获客成本', '用户生命周期',
        '功能优先级', '灰度发布', '竞品分析', '数据看板', '北极星指标', '版本规划',
      ],
    },
    {
      id: 'business',
      name: '商务会议',
      description: '客户、合同、报价与项目交付',
      words: [
        '商业模式', '合同', '报价', '交付', '客户需求', '招投标', '预算', '成本',
        '营收', '毛利率', '合作伙伴', '回款', '销售线索', '商机', '客户关系', '采购',
        '供应商', '项目周期', '里程碑', '验收', '服务协议', '解决方案', '市场份额',
        '渠道', '谈判', '决策人', '合规', '风险',
      ],
    },
    {
      id: 'finance',
      name: '金融财务',
      description: '财务指标、投资分析与风险控制',
      words: [
        '资产配置', '现金流', '净利润', '市盈率', '收益率', '风险敞口', '利率', '汇率',
        '债券', '基金', '合规', '风控', '预算', '资产负债表', '利润表', '现金流量表',
        '融资', '估值', '审计', '税务', '应收账款', '应付账款', '资本成本', '投资回报率',
        '业绩指引', '财务预测', '减值', '流动性',
      ],
    },
    {
      id: 'medical',
      name: '医疗健康',
      description: '临床、诊疗、用药与患者随访',
      words: [
        '电子病历', '临床试验', '诊断', '治疗方案', '医学影像', '检验指标', '用药',
        '不良反应', '随访', '临床指南', '医保', '患者', '病史', '主诉', '体征',
        '检查报告', '手术', '处方', '剂量', '禁忌症', '并发症', '预后', '门诊',
        '住院', '科室', '专家会诊', '康复', '知情同意',
      ],
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

  const wordKey = value => String(value || '').trim().toLocaleLowerCase();

  function getSelectedDomainPresetIds(rows) {
    const enabledWords = new Set(
      (Array.isArray(rows) ? rows : [])
        .filter(item => item?.enabled !== false)
        .map(item => wordKey(item?.text))
        .filter(Boolean),
    );
    return DOMAIN_PRESETS
      .filter(preset => preset.words.every(text => enabledWords.has(wordKey(text))))
      .map(preset => preset.id);
  }

  function setDomainPresetSelection(rows, presetId, selected, selectedPresetIds, defaultScore) {
    const preset = DOMAIN_PRESETS.find(item => item.id === presetId);
    const words = (Array.isArray(rows) ? rows : []).map(item => ({ ...item }));
    if (!preset) {
      return { preset: null, words, selected: Boolean(selected), added_count: 0, changed_count: 0 };
    }

    const existing = new Map(
      words.map(item => [wordKey(item?.text), item]).filter(([key]) => key),
    );
    const protectedWords = new Set();
    if (!selected) {
      const activePresetIds = new Set(Array.isArray(selectedPresetIds) ? selectedPresetIds : []);
      DOMAIN_PRESETS.forEach(item => {
        if (item.id === presetId || !activePresetIds.has(item.id)) return;
        item.words.forEach(text => protectedWords.add(wordKey(text)));
      });
    }
    let addedCount = 0;
    let changedCount = 0;
    preset.words.forEach(text => {
      const key = wordKey(text);
      if (!selected && protectedWords.has(key)) return;
      const existingWord = existing.get(key);
      if (existingWord) {
        if (existingWord.enabled !== Boolean(selected)) {
          existingWord.enabled = Boolean(selected);
          changedCount += 1;
        }
        return;
      }
      if (!selected) return;
      words.push({
        text,
        score: scoreForNewWord(text, defaultScore),
        enabled: true,
      });
      existing.set(key, words[words.length - 1]);
      addedCount += 1;
      changedCount += 1;
    });

    return {
      preset: { id: preset.id, name: preset.name },
      words,
      selected: Boolean(selected),
      added_count: addedCount,
      changed_count: changedCount,
    };
  }

  return {
    clamp,
    containsCjk,
    scoreForNewWord,
    normalizeSettings,
    buildPayload,
    getDomainPresets,
    getSelectedDomainPresetIds,
    setDomainPresetSelection,
  };
}));
