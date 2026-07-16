"""
ASR Text Post-Processor: Layer 1 (Rules) + Layer 2 (MacBERT Correction)

Layer 1: Rule-based cleanup — filler word removal, repetition merge, punctuation normalization (<1ms)
Layer 2: MacBERT neural correction — homophone/typo fixing (50-200ms, cached model)
"""

import re
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger("diting")

# ============================================================
# Layer 1: Rule-based post-processing
# ============================================================

# Filler words and disfluencies to remove
FILLER_WORDS = [
    # Single-character fillers
    '嗯', '额', '啊', '呃', '诶', '唉', '哦', '噢', '哇', '呀', '哈',
    # Multi-character fillers
    '那个', '这个', '就是', '然后', '等于', '就是说', '的话',
    '对吧', '对啊', '是啊', '是吧', '那什么', '怎么说呢',
    '你知道吧', '你懂的', '对不对', '是不是',
    # English fillers
    'um', 'uh', 'er', 'ah', 'like', 'you know', 'I mean', 'so yeah',
]

# Common repetition patterns (character repeated 3+ times → 1)
REPETITION_PATTERN = re.compile(r'(.)\1{2,}')

# Patterns for filler words at sentence boundaries
FILLER_BOUNDARY = re.compile(
    r'^[嗯额啊呃诶唉哦噢哇呀哈]+[，,。.！!？?\s]*|'
    r'[，,。.！!？?\s]*[嗯额啊呃诶唉哦噢哇呀哈]+$'
)

# Multi-character filler patterns (whole-word, avoid breaking real words)
MULTI_FILLER_PATTERN = re.compile(
    r'(?:^|(?<=[，,。.！!？?\s]))'
    r'(?:那个|这个|就是说|等于说|怎么说呢|你知道吧|你懂的|那什么)'
    r'(?=[，,。.！!？?\s]|$)'
)

# Punctuation cleanup
MULTI_PUNCT = re.compile(r'([，,。.！!？?；;])\1+')
EXTRA_SPACES = re.compile(r'\s{2,}')
CJK_SPACE = re.compile(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])')

# Leading/trailing punctuation
LEADING_PUNCT = re.compile(r'^[，,。.！!？?；;\s]+')
TRAILING_PUNCT = re.compile(r'[，,；;\s]+$')


def remove_filler_words(text: str) -> Tuple[str, List[str]]:
    """Remove filler words and disfluencies from text."""
    removed = []
    original = text

    # Remove leading/trailing single-char fillers
    text = FILLER_BOUNDARY.sub('', text)

    # Remove multi-character fillers
    for match in MULTI_FILLER_PATTERN.finditer(original):
        removed.append(match.group())

    text = MULTI_FILLER_PATTERN.sub('', text)

    # Remove standalone single-char fillers (surrounded by punctuation or spaces)
    for filler in ['嗯', '额', '啊', '呃', '诶', '唉', '哦', '噢']:
        # Only remove if standalone (between punctuation/spaces), not inside words
        pattern = re.compile(
            rf'(?:^|(?<=[，,。.！!？?\s])){filler}+(?=[，,。.！!？?\s]|$)'
        )
        if pattern.search(text):
            removed.append(filler)
        text = pattern.sub('', text)

    return text, removed


def merge_repetitions(text: str) -> Tuple[str, List[str]]:
    """Merge repeated characters and words.
    
    '是是是' → '是'
    '好的好的' → '好的'
    '我们我们' → '我们'
    """
    removed = []
    original = text

    # Single character repetition: '是是是' → '是'
    def replace_rep(m):
        removed.append(m.group())
        return m.group(1)

    text = REPETITION_PATTERN.sub(replace_rep, text)

    # Two-char word repetition: '好的好的' → '好的'
    # Match 2-char patterns repeated 2+ times
    two_char_rep = re.compile(r'([\u4e00-\u9fff]{2})\1+')
    
    def replace_two_rep(m):
        word = m.group(1)
        # Avoid merging real repeated words like '天天' (which are valid)
        # Only merge if the word appears 3+ times total
        if len(m.group()) >= len(word) * 2:
            removed.append(m.group())
            return word
        return m.group()

    text = two_char_rep.sub(replace_two_rep, text)

    return text, removed


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation: merge duplicates, fix spacing."""
    # Merge multiple punctuation
    text = MULTI_PUNCT.sub(r'\1', text)

    # Remove spaces between CJK characters
    text = CJK_SPACE.sub(r'\1\2', text)

    # Collapse multiple spaces
    text = EXTRA_SPACES.sub(' ', text)

    # Clean leading/trailing punctuation
    text = LEADING_PUNCT.sub('', text)
    text = TRAILING_PUNCT.sub('', text)

    # Ensure sentence ends with punctuation
    if text and text[-1] not in '。.！!？?；;':
        text += '。'

    return text


def rule_postprocess(text: str) -> Tuple[str, dict]:
    """
    Layer 1: Rule-based post-processing.
    
    Returns (cleaned_text, info_dict)
    """
    if not text or not text.strip():
        return text, {'fillers_removed': [], 'repetitions_merged': []}

    original = text

    # Step 1: Remove filler words
    text, fillers = remove_filler_words(text)

    # Step 2: Merge repetitions
    text, reps = merge_repetitions(text)

    # Step 3: Normalize punctuation
    text = normalize_punctuation(text)

    info = {
        'fillers_removed': fillers,
        'repetitions_merged': reps,
        'original_text': original,
    }

    return text, info


# ============================================================
# Layer 2: MacBERT neural correction
# ============================================================

_macbert_corrector = None
_macbert_loaded = False


def get_macbert_corrector():
    """Lazy-load MacBertCorrector (first call ~65s, subsequent calls instant)."""
    global _macbert_corrector, _macbert_loaded
    
    if _macbert_corrector is not None:
        return _macbert_corrector
    
    if _macbert_loaded and _macbert_corrector is None:
        # Already tried and failed
        return None
    
    _macbert_loaded = True
    
    try:
        import sys
        import importlib
        
        # Check if torch is already importable
        try:
            import torch
        except ImportError:
            # Add alternate torch location if not in default path
            for p in ['C:\\pylibs']:
                if p not in sys.path:
                    sys.path.append(p)
            try:
                import torch
            except ImportError:
                logger.warning("torch not available, MacBERT disabled")
                return None
        
        # Verify torch is fully functional
        if not hasattr(torch, 'Tensor'):
            logger.warning("torch incomplete, MacBERT disabled")
            return None
        
        from pycorrector import MacBertCorrector
        logger.info("Loading MacBERT correction model (first load ~65s)...")
        _macbert_corrector = MacBertCorrector()
        logger.info("MacBERT model loaded successfully")
        return _macbert_corrector
    except Exception as e:
        logger.warning(f"MacBERT model not available: {e}")
        return None


def macbert_correct(text: str) -> Tuple[str, List[dict]]:
    """
    Layer 2: MacBERT neural correction for homophones and typos.
    
    Returns (corrected_text, corrections_list)
    """
    if not text or not text.strip():
        return text, []

    corrector = get_macbert_corrector()
    if corrector is None:
        return text, []

    try:
        results = corrector.correct_batch([text])
        if results and len(results) > 0:
            result = results[0]
            corrected = result.get('target', text)
            errors = result.get('errors', [])
            
            corrections = []
            for err in errors:
                if len(err) >= 3:
                    corrections.append({
                        'original': err[0],
                        'corrected': err[1],
                        'position': err[2],
                    })
            
            return corrected, corrections
    except Exception as e:
        logger.warning(f"MacBERT correction failed: {e}")
    
    return text, []


# ============================================================
# Combined post-processor
# ============================================================

def postprocess_text(text: str) -> Tuple[str, dict]:
    """
    Full post-processing pipeline: Layer 1 (rules) → Layer 2 (MacBERT).
    
    Args:
        text: Raw ASR output text
        
    Returns:
        (final_text, info_dict)
        info_dict contains:
            - original_text: raw input
            - rule_cleaned: after Layer 1
            - final_text: after Layer 2
            - fillers_removed: list of removed filler words
            - repetitions_merged: list of merged repetitions
            - corrections: list of MacBERT corrections
    """
    if not text or not text.strip():
        return text, {
            'original_text': text,
            'rule_cleaned': text,
            'final_text': text,
            'fillers_removed': [],
            'repetitions_merged': [],
            'corrections': [],
        }

    # Layer 1: Rule-based cleanup
    rule_text, rule_info = rule_postprocess(text)

    # Layer 2: MacBERT correction
    final_text, corrections = macbert_correct(rule_text)

    info = {
        'original_text': text,
        'rule_cleaned': rule_text,
        'final_text': final_text,
        'fillers_removed': rule_info['fillers_removed'],
        'repetitions_merged': rule_info['repetitions_merged'],
        'corrections': corrections,
    }

    return final_text, info


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    test_cases = [
        "嗯嗯，那个咱们今天就是针对产品进行一下研讨会",
        "是是是，好的好的，我们我们知道这个情况",
        "诶，他真的在听吗，嗯，应该是在听吧",
        "那个这个产品的话，就是就是说要等于说要改进",
        "咱们今天真对产品进行研讨会",
        "大家好今天跟大家说一件事情那个就是关于那个留宿的问题",
    ]

    print("=" * 70)
    print("ASR Text Post-Processor Test")
    print("=" * 70)

    for i, text in enumerate(test_cases):
        print(f"\n--- Test {i+1} ---")
        print(f"原始: {text}")
        
        final, info = postprocess_text(text)
        
        print(f"规则后: {info['rule_cleaned']}")
        print(f"最终: {final}")
        if info['fillers_removed']:
            print(f"停顿词: {info['fillers_removed']}")
        if info['repetitions_merged']:
            print(f"重复词: {info['repetitions_merged']}")
        if info['corrections']:
            for c in info['corrections']:
                print(f"纠错: '{c['original']}' → '{c['corrected']}' (pos={c['position']})")
