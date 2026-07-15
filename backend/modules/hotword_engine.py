"""
谛听 DiTing - 热词引擎
============================================================================
自动热词提取 + 拼音模糊纠偏引擎。

依赖:
  - jieba (中文分词)
  - pypinyin (拼音转换)

用法:
    from modules.hotword_engine import HotwordExtractor, PinyinFuzzyMatcher

    extractor = HotwordExtractor()
    hotwords = extractor.extract(["会议转写文本..."], top_n=30)

    matcher = PinyinFuzzyMatcher({"转化率", "BERT", "A/B测试"})
    corrections = matcher.find_matches("今天讨论了zhuanhualv的问题")
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple

logger = logging.getLogger("hotword_engine")

# 尝试导入可选依赖
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    logger.info("jieba not installed. Hotword extraction will use n-gram fallback.")
    logger.info("Install: pip install jieba")

try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    logger.info("pypinyin not installed. Fuzzy matching will use string similarity fallback.")
    logger.info("Install: pip install pypinyin")


# ======================================================================
# 中文分词工具
# ======================================================================

def _segment_text(text: str) -> List[str]:
    """
    中文分词。优先 jieba，不可用时降级为 n-gram。
    """
    if HAS_JIEBA:
        words = jieba.lcut(text)
        return [w.strip() for w in words if len(w.strip()) >= 2]

    # Fallback: 基于正则和 n-gram 的分词
    # 分离中英文
    segments = re.findall(r'[一-鿿]+|[a-zA-Z0-9/+\-]+', text)

    words = []
    for seg in segments:
        if re.match(r'[一-鿿]', seg):
            # 中文 2-gram 和 3-gram
            for i in range(len(seg) - 1):
                words.append(seg[i:i+2])
            for i in range(len(seg) - 2):
                words.append(seg[i:i+3])
        else:
            # 英文/数字/符号 → 保持原样
            if len(seg) >= 2:
                words.append(seg.lower())

    return words


# ======================================================================
# HotwordExtractor: 自动热词提取
# ======================================================================

class HotwordExtractor:
    """
    从 ASR 转写文本中自动提取领域热词。

    管线:
      jieba 分词 → 去停用词 → TF-IDF 加权 → 得分排序 → TOP-N
    """

    def __init__(self, stop_words: Set[str] = None):
        """
        Args:
            stop_words: 自定义停用词集。默认使用 domain_taxonomy.STOP_WORDS。
        """
        if stop_words is None:
            try:
                from .domain_taxonomy import STOP_WORDS
                stop_words = STOP_WORDS
            except ImportError:
                stop_words = set()
        self.stop_words = stop_words

    def extract(self, texts: List[str], top_n: int = 30,
                min_count: int = 2) -> List[Dict]:
        """
        从文本列表中提取热词。

        Args:
            texts: 文本列表 (每段 ASR 输出一句)
            top_n: 返回前 N 个热词
            min_count: 最少出现次数

        Returns:
            [{"word": "转化率", "score": 3.5, "count": 5}, ...]
        """
        if not texts:
            return []

        # 统计词频
        word_count: Dict[str, int] = {}
        doc_count: Dict[str, int] = {}
        total_docs = 0

        for text in texts:
            words = set()
            for w in _segment_text(text):
                w_clean = w.strip().strip('，。！？、；：""''（）【】 \t\n\r.,!?;:')
                if len(w_clean) < 2:
                    continue
                if w_clean in self.stop_words:
                    continue
                if re.match(r'^[\d.]+$', w_clean):  # 纯数字
                    continue

                word_count[w_clean] = word_count.get(w_clean, 0) + 1
                words.add(w_clean)

            if words:
                total_docs += 1
                for w in words:
                    doc_count[w] = doc_count.get(w, 0) + 1

        if not word_count:
            return []

        # TF-IDF 风格得分
        import math
        scored = []
        for word, count in word_count.items():
            if count < min_count:
                continue
            # TF: 词频归一化
            tf = count / max(word_count.values())
            # IDF: 逆文档频率
            doc_freq = doc_count.get(word, 1)
            idf = math.log((total_docs + 1) / (doc_freq + 1)) + 1
            # 长度惩罚: 太短的词通常信息量低
            len_bonus = math.log(len(word)) / math.log(4) if len(word) >= 2 else 1
            score = tf * idf * len_bonus

            scored.append({"word": word, "score": round(score, 2), "count": count})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]

    def extract_technical_terms(self, text: str) -> List[str]:
        """
        提取英文技术术语 (缩写、复合词等)。
        """
        terms = []
        # 英文大写缩写: BERT, CNN, GPU
        abbreviations = re.findall(r'\b[A-Z]{2,}(?:\s?[A-Z]{2,})*\b', text)
        terms.extend(a.strip() for a in abbreviations if len(a.strip()) >= 2)

        # 混合术语: A/B测试, Q3, iPhone15
        mixed = re.findall(r'\b[A-Za-z]+[/+\-]?[\dA-Za-z]*[一-鿿]+\b|'
                          r'\b[一-鿿]+[/+\-]?[A-Za-z\d]+\b', text)
        terms.extend(m.strip() for m in mixed if len(m.strip()) >= 2)

        # CamelCase: Transformer, FineTuning
        camel = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
        terms.extend(c.strip() for c in camel if len(c.strip()) >= 3)

        return list(set(terms))


# ======================================================================
# PinyinFuzzyMatcher: 拼音模糊匹配
# ======================================================================

class PinyinFuzzyMatcher:
    """
    拼音模糊匹配引擎。

    将 ASR 输出中的疑似错误词转为拼音，与热词库中所有词的拼音
    计算编辑距离，找到最相似的匹配。

    替代之前硬编码的 12 条 _build_pinyin_map() 映射表。
    """

    def __init__(self, hotwords: Set[str], threshold: float = 0.75):
        """
        Args:
            hotwords: 热词集合
            threshold: 相似度阈值 (0-1)，越高越严格
        """
        self.hotwords = hotwords
        self.threshold = threshold
        self._hotword_pinyin: Dict[str, List[str]] = {}
        self._build_pinyin_index()

    def _build_pinyin_index(self):
        """构建热词拼音索引。"""
        if not HAS_PYPINYIN:
            return
        for word in self.hotwords:
            try:
                py_list = pinyin(word, style=Style.TONE3)
                # 展平: [['zhuan'], ['hua'], ['lv']] → ['zhuan', 'hua', 'lv']
                flat = [p[0] for p in py_list if p]
                self._hotword_pinyin[word] = flat
            except Exception:
                self._hotword_pinyin[word] = [word.lower()]

    def _to_pinyin(self, text: str) -> List[str]:
        """将文本转为拼音列表。"""
        if not HAS_PYPINYIN:
            return [text.lower()]

        # 分离中英文
        result = []
        for char in text:
            if '一' <= char <= '鿿':
                try:
                    py = pinyin(char, style=Style.TONE3)
                    result.append(py[0][0] if py else char)
                except Exception:
                    result.append(char)
            elif char.isalpha():
                result.append(char.lower())
            elif char.isspace():
                result.append(' ')
        return result

    def _edit_distance(self, s1: List[str], s2: List[str]) -> int:
        """计算编辑距离 (Levenshtein) 在词级别。"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

        return dp[m][n]

    def _segment_compound_pinyin(self, token: str) -> List[str]:
        """
        尝试分割连在一起的拼音字符串 (如 'zhuanhualv' → ['zhuan', 'hua', 'lv'])。

        简单启发式: 在可能的音节边界处分割 (辅音+元音组合)。
        """
        # 常见拼音声母
        initials = 'b p m f d t n l g k h j q x zh ch sh r z c s y w'.split()
        # 如果 token 看起来像连续拼音 (全小写字母)
        if not re.match(r'^[a-z]+$', token):
            return [token]

        segments = []
        i = 0
        while i < len(token):
            # 尝试匹配声母 (2 字符)
            matched = False
            for init in sorted(initials, key=len, reverse=True):
                if token[i:].startswith(init) and len(init) >= 2:
                    # 看看后面跟了什么
                    remaining = token[i + len(init):]
                    # 找下一个声母的位置
                    next_init_pos = len(remaining)
                    for init2 in initials:
                        pos = remaining.find(init2)
                        if pos > 0 and pos < next_init_pos:
                            next_init_pos = pos
                    seg = token[i:i + len(init) + next_init_pos]
                    if seg:
                        segments.append(seg)
                    i += len(init) + next_init_pos
                    matched = True
                    break
            if not matched:
                # 单字符声母
                for init in sorted(initials, key=len, reverse=True):
                    if token[i:].startswith(init):
                        remaining = token[i + len(init):]
                        next_init_pos = len(remaining)
                        for init2 in initials:
                            pos = remaining.find(init2)
                            if pos > 0 and pos < next_init_pos:
                                next_init_pos = pos
                        seg = token[i:i + len(init) + next_init_pos]
                        if seg:
                            segments.append(seg)
                        i += len(init) + max(0, next_init_pos)
                        matched = True
                        break
            if not matched:
                i += 1

        return segments if segments else [token]

    def find_matches(self, asr_text: str) -> List[dict]:
        """
        在 ASR 文本中查找模糊匹配的热词。

        Args:
            asr_text: ASR 原始输出文本

        Returns:
            [{"original": "zhuanhualv", "corrected": "转化率",
              "similarity": 0.85, "position": 5}, ...]
        """
        corrections = []

        if not self.hotwords:
            return corrections

        if not HAS_PYPINYIN:
            # 降级: 直接字符串包含匹配
            return self._find_matches_fallback(asr_text)

        # 分词
        words = _segment_text(asr_text)
        # 也包含原始分词（因为 ASR 可能输出奇怪的英文音译）
        raw_tokens = re.findall(r'[a-zA-Z]+', asr_text)

        checked = set()

        for i, token in enumerate(words + raw_tokens):
            token_clean = token.lower().strip()
            if token_clean in checked or len(token_clean) < 3:
                continue
            checked.add(token_clean)

            # 如果已经是正确热词，跳过
            if token_clean in self.hotwords:
                continue

            # 转为拼音
            token_pinyin = self._to_pinyin(token_clean)
            if not token_pinyin:
                continue

            # 如果 token 全是小写字母且较长，尝试分割复合拼音
            if (re.match(r'^[a-z]+$', token_clean) and len(token_clean) > 4
                    and len(token_pinyin) == 1):
                segmented = self._segment_compound_pinyin(token_clean)
                if len(segmented) > 1:
                    token_pinyin = segmented

            # 与热词库中所有词比较
            best_word = None
            best_sim = 0.0

            for hw, hw_pinyin in self._hotword_pinyin.items():
                if not hw_pinyin:
                    continue

                max_len = max(len(token_pinyin), len(hw_pinyin))
                if max_len == 0:
                    continue

                dist = self._edit_distance(token_pinyin, hw_pinyin)
                sim = 1.0 - (dist / max_len)

                if sim > best_sim:
                    best_sim = sim
                    best_word = hw

            if best_word and best_sim >= self.threshold:
                position = 0
                if token in words:
                    # 在 words 中找位置
                    for j, w in enumerate(words):
                        if w == token:
                            position = j
                            break

                corrections.append({
                    "original": token,
                    "corrected": best_word,
                    "similarity": round(best_sim, 3),
                    "position": position,
                    "method": "pinyin_fuzzy",
                })

        corrections.sort(key=lambda x: x["similarity"], reverse=True)
        return corrections

    def _find_matches_fallback(self, asr_text: str) -> List[dict]:
        """无 pypinyin 时的降级匹配 (基于编辑距离 + 子串)。"""
        corrections = []
        words = asr_text.lower().split()

        for i, word in enumerate(words):
            if word in self.hotwords:
                continue

            best_word = None
            best_sim = 0.0

            for hw in self.hotwords:
                hw_lower = hw.lower()
                # 子串匹配
                if hw_lower in word or word in hw_lower:
                    sim = min(len(word), len(hw_lower)) / max(len(word), len(hw_lower))
                    if sim > best_sim:
                        best_sim = sim
                        best_word = hw
                    continue

                # 编辑距离
                max_len = max(len(word), len(hw_lower))
                if max_len == 0:
                    continue
                dist = self._edit_distance(list(word), list(hw_lower))
                sim = 1.0 - (dist / max_len)
                if sim > best_sim:
                    best_sim = sim
                    best_word = hw

            if best_word and best_sim >= self.threshold:
                corrections.append({
                    "original": word,
                    "corrected": best_word,
                    "similarity": round(best_sim, 3),
                    "position": i,
                    "method": "string_fuzzy",
                })

        return corrections


# ======================================================================
# 便捷函数
# ======================================================================

def extract_and_match(texts: List[str], hotwords: Set[str],
                       top_n: int = 30) -> Dict:
    """
    一站式：提取热词 + 模糊匹配。

    Returns:
        {
            "extracted": [{"word": ..., "score": ..., "count": ...}],
            "matched": [{"original": ..., "corrected": ..., "similarity": ...}],
            "new_hotwords": ["word1", "word2", ...],  # 可加入热词库的新词
        }
    """
    extractor = HotwordExtractor()
    matcher = PinyinFuzzyMatcher(hotwords)

    extracted = extractor.extract(texts, top_n=top_n)

    # 对全文做模糊匹配
    full_text = " ".join(texts)
    matched = matcher.find_matches(full_text)

    new_hotwords = [h["word"] for h in extracted if h["score"] > 2.0
                    and h["word"] not in hotwords]

    return {
        "extracted": extracted,
        "matched": matched,
        "new_hotwords": new_hotwords,
    }
