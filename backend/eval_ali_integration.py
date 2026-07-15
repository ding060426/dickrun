"""
谛听 会悟 - Eval_Ali Dataset Integration Module
==========================================================================
Integrates the AliMeeting Eval dataset and AliMeeting4MUG dataset
to improve 会悟's ASR quality and cognitive enhancement.

Capabilities:
  1. Parse TextGrid annotations for ground-truth transcriptions
  2. Evaluate ASR accuracy against ground truth (CER/WER)
  3. Extract domain hotwords from meeting transcripts
  4. Build meeting-level summaries from AliMeeting4MUG annotations
  5. Calibrate uncertainty estimates using eval results

Dataset paths:
  - Eval_Ali:  D:\dickrun\Eval_Ali\Eval_Ali\
  - 4MUG:      D:\HACKERMarathon\Project\dataset\AliMeeting4MUG\
"""

import os
import re
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger("eval_ali")


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class EvalConfig:
    """Configuration for Eval_Ali dataset integration."""
    eval_ali_root: str = r"D:\dickrun\Eval_Ali\Eval_Ali"
    mug_root: str = r"D:\HACKERMarathon\Project\dataset\AliMeeting4MUG"
    use_far_field: bool = True   # Far-field (more realistic)
    use_near_field: bool = True  # Near-field (clean reference)
    hotword_min_count: int = 3   # Min frequency for hotwords
    hotword_max_count: int = 200 # Max frequency (filter stopwords)


# ===========================================================================
# TextGrid Parser
# ===========================================================================

def parse_textgrid(file_path: str) -> List[Dict]:
    """
    Parse a Praat TextGrid file and extract utterance-level annotations.

    Returns list of {speaker, start_sec, end_sec, text}.
    """
    if not os.path.exists(file_path):
        logger.warning(f"TextGrid not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    utterances = []

    # TextGrid tier structure: each "intervals" block contains time + text
    # Pattern: intervals [N]: xmin = A, xmax = B, text = "C"
    pattern = re.compile(
        r'intervals\s*\[\d+\]\s*:\s*\n'
        r'\s*xmin\s*=\s*([\d.]+)\s*\n'
        r'\s*xmax\s*=\s*([\d.]+)\s*\n'
        r'\s*text\s*=\s*"([^"]*)"',
        re.MULTILINE
    )

    matches = pattern.findall(content)

    for xmin, xmax, text in matches:
        text = text.strip()
        if not text:
            continue
        # Skip silence markers
        if text in ('', ' ', 'sil', 'sp', 'spn', '<UNK>'):
            continue

        utterances.append({
            'start_sec': float(xmin),
            'end_sec': float(xmax),
            'text': text,
            'duration': float(xmax) - float(xmin),
        })

    return utterances


# ===========================================================================
# Eval Dataset Scanner
# ===========================================================================

def scan_eval_dataset(root: str = None, use_far: bool = True, use_near: bool = True) -> Dict:
    """
    Scan the Eval_Ali dataset and map audio files to their annotations.

    Returns:
        {
            'meetings': {meeting_id: {
                'far_wavs': [...],
                'near_wavs': {...},  # speaker_id -> wav path
                'far_textgrid': path,
                'near_textgrids': {...},
                'far_transcript': [utterances from TextGrid],
            }},
            'total_meetings': N,
            'total_audio_hours': H,
        }
    """
    if root is None:
        root = EvalConfig.eval_ali_root

    base = Path(root)
    result = {'meetings': {}, 'total_meetings': 0, 'total_audio_hours': 0.0}

    # Scan far-field
    far_audio_dir = base / "Eval_Ali_far" / "audio_dir"
    far_textgrid_dir = base / "Eval_Ali_far" / "textgrid_dir"

    if use_far and far_audio_dir.exists():
        far_wavs = list(far_audio_dir.glob("*.wav"))
        far_tgs = {tg.stem: tg for tg in far_textgrid_dir.glob("*.TextGrid")}

        for wav in far_wavs:
            meeting_id = wav.stem  # e.g. "R8001_M8004_MS801"
            base_mtg = meeting_id.rsplit('_', 1)[0]  # "R8001_M8004"

            if base_mtg not in result['meetings']:
                result['meetings'][base_mtg] = {
                    'far_wavs': [], 'near_wavs': {},
                    'far_textgrid': None, 'near_textgrids': {},
                    'far_transcript': [],
                }

            result['meetings'][base_mtg]['far_wavs'].append(str(wav))
            result['total_audio_hours'] += wav.stat().st_size / (16000 * 2 * 3600)  # rough

            # Link TextGrid
            if base_mtg in far_tgs:
                result['meetings'][base_mtg]['far_textgrid'] = str(far_tgs[base_mtg])
                if not result['meetings'][base_mtg]['far_transcript']:
                    result['meetings'][base_mtg]['far_transcript'] = parse_textgrid(
                        str(far_tgs[base_mtg])
                    )

    # Scan near-field
    near_audio_dir = base / "Eval_Ali_near" / "audio_dir"
    near_textgrid_dir = base / "Eval_Ali_near" / "textgrid_dir"

    if use_near and near_audio_dir.exists():
        near_wavs = list(near_audio_dir.glob("*.wav"))
        near_tgs = {tg.stem: tg for tg in near_textgrid_dir.glob("*.TextGrid")}

        for wav in near_wavs:
            stem = wav.stem  # e.g. "R8001_M8004_N_SPK8013"
            parts = stem.rsplit('_N_', 1)
            if len(parts) == 2:
                base_mtg = parts[0]
                speaker = parts[1]

                if base_mtg not in result['meetings']:
                    result['meetings'][base_mtg] = {
                        'far_wavs': [], 'near_wavs': {},
                        'far_textgrid': None, 'near_textgrids': {},
                        'far_transcript': [],
                    }

                result['meetings'][base_mtg]['near_wavs'][speaker] = str(wav)
                result['total_audio_hours'] += wav.stat().st_size / (16000 * 2 * 3600)

                # Link TextGrid
                if stem in near_tgs:
                    result['meetings'][base_mtg]['near_textgrids'][speaker] = str(near_tgs[stem])

    result['total_meetings'] = len(result['meetings'])
    return result


# ===========================================================================
# Hotword Extraction from AliMeeting4MUG
# ===========================================================================

def extract_hotwords_from_mug(
    mug_root: str = None,
    min_count: int = 3,
    max_count: int = 200,
    max_hotwords: int = 200,
) -> List[Dict]:
    """
    Extract domain hotwords from AliMeeting4MUG meeting transcripts.

    Uses TF-IDF-like scoring: high within-meeting frequency, low across-meeting
    (standard TF-IDF isn't used - we compute domain specificity).

    Returns list of {word, count, score}.
    """
    if mug_root is None:
        mug_root = EvalConfig.mug_root

    train_csv = Path(mug_root) / "extracted" / "train" / "train.csv"
    if not train_csv.exists():
        logger.warning(f"AliMeeting4MUG train.csv not found at {train_csv}")
        return []

    logger.info(f"Extracting hotwords from {train_csv}...")

    # Collect all sentences
    all_texts = []
    meeting_texts = []  # Per-meeting word sets for IDF

    with open(train_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                item = json.loads(row.get('content', '{}'))
                sentences = item.get('sentences', [])
                words_in_meeting = set()

                meeting_tokens = []
                for s in sentences:
                    text = s.get('s', '')
                    meeting_tokens.append(text)
                    # Segment words (Chinese: 2-char n-grams)
                    words = _segment_chinese_words(text)
                    words_in_meeting.update(words)

                all_texts.extend(meeting_tokens)
                meeting_texts.append(words_in_meeting)
            except (json.JSONDecodeError, KeyError):
                continue

    # Count word frequencies
    word_counter = Counter()
    meeting_counter = Counter()  # How many meetings a word appears in

    for text in all_texts:
        words = _segment_chinese_words(text)
        word_counter.update(words)

    for words_in_meeting in meeting_texts:
        meeting_counter.update(words_in_meeting)

    total_meetings = max(1, len(meeting_texts))

    # Score: high freq * low meeting dispersion (domain-specific)
    scored = []
    for word, count in word_counter.most_common(5000):
        if count < min_count or count > max_count:
            continue
        if len(word) < 2:
            continue

        # IDF-like: words appearing in fewer meetings are more specific
        meeting_freq = meeting_counter.get(word, 1)
        idf = total_meetings / max(1, meeting_freq)
        score = count * min(idf, 10.0)  # Cap IDF influence

        scored.append({'word': word, 'count': count, 'score': round(score, 1)})

    # Sort by score
    scored.sort(key=lambda x: x['score'], reverse=True)
    hotwords = scored[:max_hotwords]

    logger.info(f"Extracted {len(hotwords)} hotwords from {total_meetings} meetings")
    if hotwords:
        top5 = [h['word'] for h in hotwords[:5]]
        logger.info(f"  Top hotwords: {', '.join(top5)}")

    return hotwords


def _segment_chinese_words(text: str) -> List[str]:
    """
    Segment Chinese text into word candidates.
    Uses 2-gram and 3-gram extraction (no jieba dependency).
    Filters punctuation and pure numbers.
    """
    # Remove punctuation
    cleaned = re.sub(r'[^一-鿿㐀-䶿a-zA-Z0-9]', '', text)

    words = []

    # Chinese 2-grams
    chinese_chars = ''.join(c for c in cleaned if '一' <= c <= '鿿')
    for i in range(len(chinese_chars) - 1):
        words.append(chinese_chars[i:i+2])
    for i in range(len(chinese_chars) - 2):
        words.append(chinese_chars[i:i+3])

    # English words
    eng_matches = re.findall(r'[a-zA-Z]{2,}', cleaned)
    words.extend(eng_matches)

    # Alphanumeric terms
    alnum_matches = re.findall(r'[a-zA-Z0-9/+\-]{2,}', cleaned)
    words.extend(alnum_matches)

    return [w.lower() for w in words if len(w) >= 2]


# ===========================================================================
# ASR Evaluation
# ===========================================================================

def evaluate_against_textgrid(
    asr_results: List[Dict],
    ground_truth: List[Dict],
    max_segments: int = 100,
) -> Dict:
    """
    Evaluate ASR output against TextGrid ground truth.

    Computes:
      - CER (Character Error Rate)
      - Match rate (segments with correct text)

    Args:
        asr_results: List of {text, start_sec, end_sec} from ASR
        ground_truth: List of {text, start_sec, end_sec} from TextGrid

    Returns:
        Evaluation metrics dict
    """
    if not ground_truth:
        return {'error': 'No ground truth available', 'cer': None}

    # Simple approach: concatenate all texts and compute CER
    asr_full = ''.join(r.get('text', '') for r in asr_results)
    gt_full = ''.join(g.get('text', '') for g in ground_truth)

    cer = _compute_cer(gt_full, asr_full)

    # Also compute word-level metrics
    # Align segments by time overlap
    matched = 0
    total_gt = min(len(ground_truth), max_segments)

    for gt_seg in ground_truth[:max_segments]:
        gt_mid = (gt_seg.get('start_sec', 0) + gt_seg.get('end_sec', 0)) / 2
        for asr_seg in asr_results:
            asr_start = asr_seg.get('start_sec', 0)
            asr_end = asr_seg.get('end_sec', 0)
            if asr_start <= gt_mid <= asr_end:
                # Compute CER for this pair
                pair_cer = _compute_cer(
                    gt_seg.get('text', ''),
                    asr_seg.get('text', '')
                )
                if pair_cer < 0.5:
                    matched += 1
                break

    return {
        'cer': round(cer, 4),
        'cer_pct': f'{cer*100:.1f}%',
        'asr_chars': len(asr_full),
        'gt_chars': len(gt_full),
        'segment_match_rate': round(matched / max(1, total_gt), 3),
        'total_gt_segments': len(ground_truth),
        'total_asr_segments': len(asr_results),
    }


def _compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate using Levenshtein distance."""
    if not reference:
        return 1.0 if hypothesis else 0.0

    # Levenshtein distance at character level
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_chars[i-1] == hyp_chars[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost, # substitution
            )

    return dp[m][n] / max(1, m)


# ===========================================================================
# Quick check
# ===========================================================================

def check_dataset_status() -> Dict:
    """Check the status of all available datasets."""
    return {
        'eval_ali': {
            'path': EvalConfig.eval_ali_root,
            'exists': os.path.exists(EvalConfig.eval_ali_root),
            'scan': scan_eval_dataset(use_far=True, use_near=True),
        },
        'mug': {
            'path': EvalConfig.mug_root,
            'exists': os.path.exists(EvalConfig.mug_root),
        },
    }


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.logger import init_logging
    init_logging(console_level="DEBUG")

    print("\n=== Eval_Ali Dataset Check ===\n")
    status = check_dataset_status()
    print(f"Eval_Ali exists: {status['eval_ali']['exists']}")
    print(f"Meetings: {status['eval_ali']['scan']['total_meetings']}")
    print(f"Audio hours (est): {status['eval_ali']['scan']['total_audio_hours']:.1f}")
    print(f"MUG exists: {status['mug']['exists']}")

    # Test TextGrid parsing
    far_dir = Path(EvalConfig.eval_ali_root) / "Eval_Ali_far" / "textgrid_dir"
    if far_dir.exists():
        sample_tg = list(far_dir.glob("*.TextGrid"))[0]
        utts = parse_textgrid(str(sample_tg))
        print(f"\nSample TextGrid: {sample_tg.name}")
        print(f"  Utterances: {len(utts)}")
        if utts:
            print(f"  First: [{utts[0]['start_sec']:.1f}s-{utts[0]['end_sec']:.1f}s] {utts[0]['text'][:80]}")

    # Test hotword extraction
    print("\n=== Hotword Extraction ===")
    hotwords = extract_hotwords_from_mug(max_hotwords=30)
    for hw in hotwords[:15]:
        print(f"  {hw['word']:20s} count={hw['count']:4d}  score={hw['score']:.1f}")
