"""
谛听 DiTing - 文本后处理器
============================================================================
ASR 转写文本的后处理管线：

  raw ASR text → ① 语气词过滤 → ② 标点恢复(模型优先, 规则兜底)
               → ③ 强制断句 → ④ 规范化

标点恢复：
  - 优先使用 sherpa-onnx CT-Transformer 模型 (高精度)
  - 模型不可用时自动降级到规则引擎
  - 零外部依赖 (punct 模型可选)
"""

import os
import re
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger("text_postproc")

# ── 懒加载标点模型 ──────────────────────────────────────────────
_punct_restorer = None
_punct_load_attempted = False


def _get_punct_restorer():
    """懒加载标点恢复模型 (首次调用时初始化)。"""
    global _punct_restorer, _punct_load_attempted
    if _punct_load_attempted:
        return _punct_restorer
    _punct_load_attempted = True
    try:
        from .punctuation_model import get_punctuation_restorer
        _punct_restorer = get_punctuation_restorer()
        if _punct_restorer.is_available:
            logger.info("Punctuation model loaded — using ML-based restoration")
        else:
            logger.info("Punctuation model not available — using rule-based fallback")
    except Exception as e:
        logger.debug(f"Punctuation model init skipped: {e}")
    return _punct_restorer


# ======================================================================
# ① 语气词 / 填充词过滤
# ======================================================================

# 中文口语填充词 (句首/句尾删除，句中保留)
_FILLER_SENTENCE_START = re.compile(
    r'^[\s]*'
    r'(嗯[啊呀呢嘛吧]?|呃|哦[哦]?|欸|唉|嘶|啧|咝|诶[诶]?)'
    r'[\s,.，。！？!?]*'
)

_FILLER_SENTENCE_END = re.compile(
    r'[\s]*'
    r'(嗯[啊呀呢嘛吧]?|呃|啊[啊啊]?|哦[哦]?|嘛|呗|吧|啦|哈|嘿|呵)'
    r'[\s]*$'
)

# 常见口头禅/冗余词组 (句中可安全删除，不影响语义)
_FILLER_PHRASES = [
    ('就是说', ''),
    ('然后呢', '然后'),
    ('那个那个', '那个'),
    ('这个这个', '这个'),
    ('对对对', '对的'),
    ('是是是', '是的'),
    ('好好好', '好的'),
    ('行行行', '行的'),
    ('呃呃呃', ''),
    ('啊啊啊', ''),
    ('嗯嗯嗯', ''),
    ('就是说呢', ''),
    ('那么就是说', '那么'),
    ('咱们就是说', '咱们'),
    ('其实吧', '其实'),
    ('我个人觉得吧', '我觉得'),
    ('怎么说呢', ''),
    ('怎么说吧', ''),
    ('说白了就是', '就是'),
    ('说白了', ''),
    ('毋庸置疑', ''),
    ('众所周知', ''),
]


def remove_fillers(text: str) -> str:
    """移除句首/句尾语气词和句中口头禅。"""
    # 句首语气词
    text = _FILLER_SENTENCE_START.sub('', text)
    # 句尾语气词
    text = _FILLER_SENTENCE_END.sub('', text)
    # 句中冗余词组
    for phrase, replacement in _FILLER_PHRASES:
        text = text.replace(phrase, replacement)
    return text.strip()


# ======================================================================
# ② 标点恢复 (规则)
# ======================================================================

# 句间停顿词 → 前插句号或逗号
_SENTENCE_BREAK_WORDS = [
    # (pattern, 替换, 类型)
    # 强断句 — 句号
    (re.compile(r'(?<=。)但是(?=\S)'), '。但是', '。'),
    (re.compile(r'(?<=。)所以(?=\S)'), '。所以', '。'),
    (re.compile(r'(?<=。)然后(?=\S)'), '。然后', '。'),
    (re.compile(r'(?<=。)不过(?=\S)'), '。不过', '。'),
    (re.compile(r'(?<=。)而且(?=\S)'), '。而且', '。'),
    (re.compile(r'(?<=。)另外(?=\S)'), '。另外', '。'),
    (re.compile(r'(?<=。)因此(?=\S)'), '。因此', '。'),
    (re.compile(r'(?<=。)接下来(?=\S)'), '。接下来', '。'),
    (re.compile(r'(?<=。)首先(?=\S)'), '。首先', '。'),
    (re.compile(r'(?<=。)其次(?=\S)'), '。其次', '。'),
    (re.compile(r'(?<=。)最后(?=\S)'), '。最后', '。'),
    (re.compile(r'(?<=。)总之(?=\S)'), '。总之', '。'),
    (re.compile(r'(?<=。)同时(?=\S)'), '。同时', '。'),
    (re.compile(r'(?<=。)好了(?=\S)'), '。好了', '。'),
    (re.compile(r'(?<=。)行(?=\S)'), '。行', '。'),
    (re.compile(r'(?<=。)好(?=\S)'), '。好', '。'),
    # 中停顿 → 逗号
    (re.compile(r'(?<=。)比如(?=\S)'), '，比如', '，'),
    (re.compile(r'(?<=。)包括(?=\S)'), '，包括', '，'),
    (re.compile(r'(?<=。)还有(?=\S)'), '，还有', '，'),
    (re.compile(r'(?<=。)以及(?=\S)'), '，以及', '，'),
]


# 疑问/感叹结尾词
_QUERY_ENDING = re.compile(r'(吗|呢|吧|啊|呀|哇|哦|哈|呗|嘛)[\s]*$')
_EXCLAIM_ENDING = re.compile(r'(啊|呀|哇|啦|哦|喽|咯)[\s]*$')


# 连续无标点的最大汉字数 (超过则强制插入)
_MAX_CHARS_WITHOUT_PUNCT = 40

# 弱语义边界 — 在这些字后面可以安全插入逗号
_WEAK_BOUNDARY = re.compile(r'(的|了|和|与|及|或|等|啊|呀|呢|嘛|吧|吗)[\s]*(?!$)')

# 已有标点
_HAS_PUNCT = re.compile(r'[，。！？、；：\.,!?;:]')


def restore_punctuation(text: str, use_model: bool = True) -> str:
    """
    标点恢复。优先使用 sherpa-onnx CT-Transformer 模型，不可用时降级为规则。

    Args:
        text: 无标点的原始文本
        use_model: 是否尝试使用 ML 模型 (False 则强制使用规则)
    """
    if not text.strip():
        return text

    # 清理已有的半角标点，统一为中文全角
    text = text.replace(',', '，').replace('.', '。').replace('!', '！').replace('?', '？')
    text = text.replace(';', '；').replace(':', '：')

    # ── 尝试 ML 模型 ──
    # CT-Punc expects raw, unpunctuated ASR text. Feeding it transducer output
    # that already contains punctuation commonly produces "。。" / "。，".
    if use_model and not _HAS_PUNCT.search(text):
        restorer = _get_punct_restorer()
        if restorer and restorer.is_available:
            try:
                result = restorer.add_punctuation(text)
                if result and result.strip():
                    return result.strip()
            except Exception as e:
                logger.debug(f"ML punctuation failed, using rules: {e}")

    # ── 规则兜底 ──
    # 断句关键词
    for pattern, replacement, _ in _SENTENCE_BREAK_WORDS:
        text = pattern.sub(replacement, text)

    # 疑问/感叹结尾
    if _QUERY_ENDING.search(text) and not text.rstrip().endswith(('？', '?')):
        text = text.rstrip() + '？'
    elif _EXCLAIM_ENDING.search(text) and not text.rstrip().endswith(('！', '!')):
        text = text.rstrip() + '。'
    elif not text.rstrip().endswith(('。', '！', '？', '，', '；', '：', '.', '!', '?', ',', ';', ':')):
        text = text.rstrip() + '。'

    return text


def force_split_long_sentence(text: str, max_chars: int = None) -> str:
    """
    如果一句话太长且中间没有标点，在弱语义边界插入逗号。
    """
    if max_chars is None:
        max_chars = _MAX_CHARS_WITHOUT_PUNCT

    # 按已有标点分段
    segments = re.split(r'([，。！？、；：])', text)
    # segments 交替为 [文本, 标点, 文本, 标点, ...]

    result_parts = []
    for i, seg in enumerate(segments):
        if i % 2 == 1:
            # 这是标点符号本身
            result_parts.append(seg)
            continue

        # 这是一个文本段
        chars = re.findall(r'[一-鿿]', seg)
        if len(chars) <= max_chars:
            result_parts.append(seg)
            continue

        # 需要插入逗号
        # 找到所有弱边界位置
        positions = []
        for m in _WEAK_BOUNDARY.finditer(seg):
            chars_before = len(re.findall(r'[一-鿿]', seg[:m.end()]))
            if chars_before > max_chars * 0.5:  # 不要太靠前
                positions.append((m.end(), chars_before))

        if not positions:
            result_parts.append(seg)
            continue

        # 每 max_chars 个汉字找一个最接近的弱边界
        splits = []
        target = max_chars
        while target < len(chars):
            best = None
            best_dist = float('inf')
            for pos, char_count in positions:
                if pos in splits:
                    continue
                dist = abs(char_count - target)
                if dist < best_dist and target - max_chars * 0.4 < char_count < target + max_chars * 0.4:
                    best_dist = dist
                    best = (pos, char_count)
            if best:
                splits.append(best[0])
                target = best[1] + max_chars
            else:
                target += max_chars

        # 在 split 位置插入逗号
        if splits:
            result = ''
            last = 0
            for sp in sorted(splits):
                result += seg[last:sp] + '，'
                last = sp
            result += seg[last:]
            result_parts.append(result)
        else:
            result_parts.append(seg)

    return ''.join(result_parts)


# ======================================================================
# ④ 规范化
# ======================================================================

# 连续重复字 (3 次及以上)
_REPEATED_CHAR = re.compile(r'(.)\1{2,}')
_PUNCTUATION_RUN = re.compile(r'[，。！？、；：]{2,}')


def _collapse_punctuation(match: re.Match) -> str:
    run = match.group(0)
    for punctuation in ('？', '！', '。', '；', '：', '，', '、'):
        if punctuation in run:
            return punctuation
    return run[-1]


def normalize_text(text: str) -> str:
    """
    规范化 ASR 输出：
    - 合并连续重复字
    - 清理多余空格
    - 中英文数字间加适当间距
    """
    if not text:
        return text

    # 连续重复 3 次以上 → 保留 2 次
    text = _REPEATED_CHAR.sub(r'\1\1', text)

    # ASR 自带标点与恢复模型叠加时，只保留语义最强的边界。
    text = _PUNCTUATION_RUN.sub(_collapse_punctuation, text)

    # 英文单词前后与中文之间加空格 (可选，让排版更美观)
    # 数字前后加空格
    text = re.sub(r'(?<=[一-鿿])(?=\d)', ' ', text)
    text = re.sub(r'(?<=\d)(?=[一-鿿])', ' ', text)

    # 清理多余空格
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([，。！？、；：])', r'\1', text)  # 标点前不要空格
    text = re.sub(r'([，。！？、；：])\s+', r'\1', text)  # 标点后可选保留

    return text.strip()


# ======================================================================
# 组合管线
# ======================================================================

def process_asr_text(
    raw_text: str,
    enable_filler_filter: bool = True,
    enable_punctuation: bool = True,
    enable_force_split: bool = True,
    enable_normalize: bool = True,
    max_chars_per_segment: int = None,
) -> str:
    """
    ASR 文本后处理主入口。

    Args:
        raw_text: 原始 ASR 识别文本
        enable_filler_filter: 是否过滤语气词
        enable_punctuation: 是否恢复标点
        enable_force_split: 是否强制分割长句
        enable_normalize: 是否规范化
        max_chars_per_segment: 单句最大汉字数 (默认 40)

    Returns:
        处理后的文本
    """
    text = raw_text.strip()
    if not text:
        return text

    if enable_filler_filter:
        text = remove_fillers(text)

    if enable_punctuation:
        text = restore_punctuation(text)

    if enable_force_split:
        text = force_split_long_sentence(text, max_chars=max_chars_per_segment)

    if enable_normalize:
        text = normalize_text(text)

    return text


def process_transcript_segments(
    segments: List[str],
    **kwargs,
) -> List[str]:
    """批量处理多个转写片段。"""
    return [process_asr_text(seg, **kwargs) for seg in segments]


# ======================================================================
# Structured result + optional MacBERT layer (adapted from tep/be69440)
# ======================================================================

_macbert_corrector = None
_macbert_load_attempted = False


def _get_macbert_corrector():
    """Lazy-load pycorrector only when explicitly enabled."""
    global _macbert_corrector, _macbert_load_attempted
    if _macbert_corrector is not None:
        return _macbert_corrector
    if _macbert_load_attempted:
        return None
    _macbert_load_attempted = True
    try:
        from pycorrector import MacBertCorrector

        logger.info("Loading optional MacBERT text corrector")
        _macbert_corrector = MacBertCorrector()
    except Exception as error:
        logger.warning("MacBERT unavailable; continuing with rule pipeline: %s", error)
    return _macbert_corrector


def _macbert_correct(text: str) -> Tuple[str, List[dict]]:
    corrector = _get_macbert_corrector()
    if corrector is None:
        return text, []
    try:
        results = corrector.correct_batch([text])
        if not results:
            return text, []
        item = results[0]
        corrections = [
            {
                "original": error[0],
                "corrected": error[1],
                "position": error[2],
                "source": "macbert",
            }
            for error in item.get("errors", [])
            if len(error) >= 3
        ]
        return item.get("target", text), corrections
    except Exception as error:
        logger.warning("MacBERT correction failed: %s", error)
        return text, []


def _collect_cleanup_details(text: str) -> Tuple[List[str], List[str]]:
    fillers: List[str] = []
    for phrase, _ in _FILLER_PHRASES:
        if phrase and phrase in text and phrase not in fillers:
            fillers.append(phrase)
    for pattern in (_FILLER_SENTENCE_START, _FILLER_SENTENCE_END):
        match = pattern.search(text)
        if match:
            value = match.group(0).strip(" ,.，。！？!?")
            if value and value not in fillers:
                fillers.append(value)
    repetitions = [match.group(0) for match in _REPEATED_CHAR.finditer(text)]
    return fillers, repetitions


def process_asr_text_with_details(raw_text: str, **kwargs) -> Tuple[str, dict]:
    """Run the established pipeline and expose user-visible edit metadata."""
    fillers, repetitions = _collect_cleanup_details(raw_text)
    rule_text = process_asr_text(raw_text, **kwargs)
    final_text = rule_text
    corrections: List[dict] = []
    if os.getenv("DITING_ENABLE_MACBERT", "").strip().lower() in {"1", "true", "yes"}:
        final_text, corrections = _macbert_correct(rule_text)
    return final_text, {
        "original_text": raw_text,
        "rule_cleaned": rule_text,
        "final_text": final_text,
        "fillers_removed": fillers,
        "repetitions_merged": repetitions,
        "corrections": corrections,
    }
