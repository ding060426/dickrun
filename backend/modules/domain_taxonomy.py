"""
谛听 DiTing - 领域分类体系
============================================================================
定义会议领域分类、关键词映射和子领域体系。

用于:
  1. 热词 → 领域反向推断 (DomainInferrer)
  2. 领域感知的热词加载
  3. 前端领域筛选

每个领域包含:
  - keywords: 该领域的典型术语/关键词列表
  - sub_domains: 子领域列表
  - description: 领域描述
"""

from typing import List, Dict, Optional

# ======================================================================
# 领域分类体系
# ======================================================================

DOMAIN_TAXONOMY: Dict[str, dict] = {
    "互联网产品": {
        "description": "互联网产品的设计、运营、增长和数据分析",
        "sub_domains": ["增长运营", "数据分析", "广告投放", "用户研究", "产品设计"],
        "keywords": [
            # 产品指标
            "转化率", "留存率", "日活", "月活", "DAU", "MAU", "用户粘性",
            "获客成本", "ARPU", "LTV", "ROI", "点击率", "CTR", "跳出率",
            "复购率", "客单价", "GMV", "渗透率", "覆盖率",
            # 产品运营
            "用户画像", "用户分层", "增长黑客", "拉新", "促活", "留存", "变现",
            "裂变", "漏斗", "转化漏斗", "AARRR", "RFM",
            # 实验与测试
            "A/B测试", "灰度发布", "MVP", "PMF", "NPS", "CSAT",
            # 渠道
            "信息流", "SEM", "SEO", "ASA", "ASO", "DSP", "SSP",
            "私域", "公域", "社群运营", "KOL", "KOC", "直播带货",
        ],
    },

    "人工智能": {
        "description": "AI/ML 研究、模型训练、推理部署",
        "sub_domains": ["NLP", "CV", "推荐系统", "大模型", "语音识别", "MLOps"],
        "keywords": [
            # 模型技术
            "BERT", "Transformer", "GPT", "CNN", "RNN", "LSTM", "Attention",
            "ResNet", "ViT", "CLIP", "Diffusion", "GAN", "VAE",
            "Encoder", "Decoder", "Tokenizer", "Embedding",
            # 训练
            "微调", "Fine-tuning", "预训练", "蒸馏", "量化", "剪枝",
            "LoRA", "P-Tuning", "RLHF", "SFT", "Prompt",
            "学习率", "Batch Size", "Epoch", "Loss", "梯度", "优化器",
            # 评估
            "准确率", "召回率", "F1", "BLEU", "ROUGE", "困惑度", "Perplexity",
            "AUC", "ROC", "混淆矩阵", "SOTA",
            # 部署
            "GPU", "TPU", "CUDA", "ONNX", "TensorRT", "推理", "延迟",
            "QPS", "吞吐", "显存", "V100", "A100", "H100",
            # 应用
            "RAG", "Agent", "Function Calling", "向量数据库", "Embedding",
        ],
    },

    "金融投资": {
        "description": "风险投资、二级市场、企业财务",
        "sub_domains": ["一级市场", "二级市场", "风控", "量化交易", "企业财务"],
        "keywords": [
            # 投资
            "估值", "尽调", "DD", "LP", "GP", "退出", "IPO", "并购", "M&A",
            "天使轮", "A轮", "B轮", "C轮", "Pre-IPO", "独角兽",
            "赛道", "市场规模", "TAM", "SAM", "SOM", "竞争格局",
            # 金融指标
            "ROE", "ROA", "PE", "PB", "PS", "EBITDA", "毛利率", "净利率",
            "现金流", "负债率", "周转率", "营收", "净利润",
            # 二级市场
            "指数", "ETF", "多头", "空头", "对冲", "套利", "波动率",
            "Alpha", "Beta", "夏普比率", "最大回撤", "年化收益",
            # 风控
            "风控模型", "信用评分", "反欺诈", "KYC", "AML", "巴塞尔",
        ],
    },

    "学术科研": {
        "description": "学术研究、论文、实验方法论",
        "sub_domains": ["计算机科学", "生物医药", "材料科学", "社会科学", "数学"],
        "keywords": [
            "论文", "实验", "基准", "Benchmark", "SOTA", "消融实验",
            "Ablation", "引用", "同行评议", "影响因子", "H指数",
            "数据集", "预处理", "Baseline", "对比实验", "显著性",
            "P值", "假设检验", "对照组", "双盲", "随机化",
            "预印本", "ArXiv", "审稿", "Rebuttal", "Camera Ready",
        ],
    },

    "企业运营": {
        "description": "企业日常管理、项目管理、人力资源",
        "sub_domains": ["项目管理", "人力资源", "财务", "战略规划", "供应链"],
        "keywords": [
            # 管理方法
            "OKR", "KPI", "ROI", "预算", "排期", "复盘", "里程碑",
            "交付", "对齐", "拉通", "闭环", "颗粒度", "抓手",
            "Sprint", "Scrum", "Kanban", "站会", "回顾", "Planning",
            # 人力资源
            "招聘", "入职", "离职", "绩效", "晋升", "职级", "薪酬",
            "期权", "ESOP", "人才盘点", "360评估",
            # 财务
            "预算", "成本中心", "利润中心", "摊销", "折旧", "应收账款",
            "应付账款", "现金流", "损益表", "资产负债表",
        ],
    },
}


# ======================================================================
# 停用词表 (TF-IDF 热词提取时过滤)
# ======================================================================

STOP_WORDS: set = {
    # 中文停用词
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "如果", "因为", "所以", "但是", "然后", "可以", "这个",
    "那个", "还是", "只是", "不过", "已经", "而且", "虽然", "应该", "觉得",
    "知道", "可能", "需要", "比较", "一下", "一点", "一个", "一种",
    "进行", "使用", "通过", "对于", "根据", "关于", "以及", "或者",
    "目前", "现在", "今天", "昨天", "明天", "之前", "之后", "以后",
    "我们", "你们", "他们", "大家", "这边", "那边", "里面", "外面",
    "时候", "地方", "问题", "情况", "方法", "方式", "方面", "部分",
    "刚才", "刚刚", "正在", "一直", "已经", "曾经", "经常", "总是",
    "嗯", "呃", "啊", "哦", "嘛", "吧", "呢", "吗", "呀", "呗",
    "就是", "就是说", "所以说", "对吧", "对", "行", "好", "嗯嗯",
    # 英文停用词
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "here", "there",
    "and", "but", "or", "nor", "for", "so", "yet", "with", "without",
    "in", "on", "at", "to", "from", "by", "of", "for", "about",
    "not", "no", "very", "just", "only", "also", "then", "now",
}


def get_domain_keywords(domain: str) -> List[str]:
    """获取指定领域的全部关键词。"""
    if domain in DOMAIN_TAXONOMY:
        return DOMAIN_TAXONOMY[domain].get("keywords", [])
    return []


def get_all_domains() -> List[str]:
    """获取所有领域名称。"""
    return list(DOMAIN_TAXONOMY.keys())


def get_sub_domains(domain: str) -> List[str]:
    """获取某个领域的子领域列表。"""
    if domain in DOMAIN_TAXONOMY:
        return DOMAIN_TAXONOMY[domain].get("sub_domains", [])
    return []


def match_domain(hotwords: List[str]) -> List[tuple]:
    """
    基于热词列表匹配最可能的领域 (纯规则)。

    返回: [(domain, score), ...] 按得分降序排列
    """
    hotwords_lower = set(w.lower() for w in hotwords)
    if not hotwords_lower:
        return []

    scores = []
    for domain, info in DOMAIN_TAXONOMY.items():
        domain_keywords = set(k.lower() for k in info.get("keywords", []))
        matched = hotwords_lower & domain_keywords
        if matched:
            score = len(matched) / len(hotwords_lower)
            scores.append((domain, round(score, 3), list(matched)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def merge_all_domain_keywords() -> List[str]:
    """合并所有领域关键词（去重），用于通用热词库。"""
    all_kw = set()
    for info in DOMAIN_TAXONOMY.values():
        all_kw.update(w.lower() for w in info.get("keywords", []))
    return sorted(all_kw)
