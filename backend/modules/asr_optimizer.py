"""
谛听 DiTing - ASR 优化器
============================================================================
围绕现有 sherpa-onnx X-ASR 管线做轻量、可插拔的前后处理优化：

  输入音频 → 音频标准化/质量画像 → 自适应 VAD 参数 → ASR
          → 输入法式候选纠错/文本稳定化 → 质量解释

设计目标：
- 不替换现有 ASR 模型，不影响队友实时转录对接；
- 无新增重型模型依赖，只复用 numpy、现有热词/后处理模块；
- 所有优化均可解释，并通过 report 返回前端展示。
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import cn2an
    HAS_CN2AN = True
except ImportError:
    cn2an = None
    HAS_CN2AN = False

logger = logging.getLogger("asr_optimizer")


ACRONYM_MAP = {
    "asr": "ASR", "vad": "VAD", "llm": "LLM", "nlp": "NLP",
    "api": "API", "sdk": "SDK", "ui": "UI", "ux": "UX",
    "gpu": "GPU", "cpu": "CPU", "ram": "RAM", "fps": "FPS",
    "okr": "OKR", "kpi": "KPI", "roi": "ROI", "dau": "DAU", "mau": "MAU",
    "ctr": "CTR", "cvr": "CVR", "uv": "UV", "pv": "PV",
    "bert": "BERT", "gpt": "GPT", "rag": "RAG", "cnn": "CNN", "rnn": "RNN",
    "transformer": "Transformer", "lora": "LoRA",
}

ACRONYM_EXPLANATIONS = {
    "ASR": "自动语音识别",
    "VAD": "语音活动检测",
    "LLM": "大语言模型",
    "NLP": "自然语言处理",
    "API": "应用程序接口",
    "SDK": "软件开发工具包",
    "UI": "用户界面",
    "UX": "用户体验",
    "GPU": "图形处理器",
    "CPU": "中央处理器",
    "RAM": "内存",
    "FPS": "每秒帧数",
    "OKR": "目标与关键结果",
    "KPI": "关键绩效指标",
    "ROI": "投资回报率",
    "DAU": "日活跃用户",
    "MAU": "月活跃用户",
    "CTR": "点击率",
    "CVR": "转化率",
    "UV": "独立访客",
    "PV": "页面浏览量",
    "BERT": "双向编码器表示模型",
    "GPT": "生成式预训练模型",
    "RAG": "检索增强生成",
    "CNN": "卷积神经网络",
    "RNN": "循环神经网络",
    "LoRA": "低秩适配",
}


@dataclass
class ASROptimizerReport:
    """一次音频处理的 ASR 优化报告。"""
    enabled: bool = True
    profile: Dict = field(default_factory=dict)
    vad_config: Dict = field(default_factory=dict)
    audio_actions: List[str] = field(default_factory=list)
    text_actions: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "profile": self.profile,
            "vad_config": self.vad_config,
            "audio_actions": self.audio_actions,
            "text_actions": self.text_actions,
            "suggestions": self.suggestions,
        }


class ASROptimizer:
    """
    ASR 优化器：聚焦“输入更干净、切分更合理、输出更稳定”。

    当前版本采用轻量规则，不引入会破坏部署的额外模型；后续可把
    WebRTC VAD、RNNoise、FunASR 标点/逆文本规范化等接到同一接口。
    """

    _CN_DIGITS = {
        "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    }
    _CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}

    def __init__(self, hotwords: Optional[List[str]] = None, sample_rate: int = 16000):
        self.hotwords = set(hotwords or [])
        self.sample_rate = sample_rate
        self.last_report = ASROptimizerReport()

    # ------------------------------------------------------------------
    # Audio front-end
    # ------------------------------------------------------------------

    def prepare_audio(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, ASROptimizerReport]:
        """
        对整段音频做模型前优化：单声道/幅度规范化/DC 去除/轻度降噪画像。
        """
        report = ASROptimizerReport(enabled=True)
        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(data) == 0:
            report.suggestions.append("音频为空，无法优化")
            self.last_report = report
            return data, report

        # 1. 去除 DC 偏移
        mean_abs = float(abs(np.mean(data)))
        if mean_abs > 1e-4:
            data = data - np.mean(data)
            report.audio_actions.append("移除直流偏移")

        # 2. 峰值归一化，但避免把纯噪声过度放大
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data ** 2) + 1e-12))
        if peak > 1.0:
            data = data / peak
            report.audio_actions.append("削峰归一化")
        elif 0.02 < peak < 0.6:
            gain = min(3.0, 0.85 / peak)
            data = np.clip(data * gain, -1.0, 1.0)
            report.audio_actions.append(f"增益补偿 x{gain:.1f}")

        # 3. 极低能量段提示，不强行放大
        if rms < 0.005:
            report.suggestions.append("输入音量偏低，建议靠近麦克风或提升录音增益")

        # 4. 质量画像 + 自适应 VAD 参数
        profile = self.profile_audio(data, sr)
        report.profile = profile
        report.vad_config = self.suggest_vad_config(profile)
        report.suggestions.extend(self._quality_suggestions(profile))

        if not report.audio_actions:
            report.audio_actions.append("音频幅度正常，保持原始动态")

        self.last_report = report
        return data.astype(np.float32), report

    def profile_audio(self, audio: np.ndarray, sr: int) -> Dict:
        """生成轻量音频画像，供 VAD 和前端解释使用。"""
        if len(audio) == 0:
            return {"duration_sec": 0.0, "rms": 0.0, "peak": 0.0, "silence_ratio": 1.0}

        frame_len = max(1, int(sr * 0.02))
        n_frames = max(1, len(audio) // frame_len)
        frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        noise_floor = float(np.percentile(frame_rms, 20))
        speech_floor = float(np.percentile(frame_rms, 80))
        dynamic_range = float((speech_floor + 1e-8) / (noise_floor + 1e-8))
        silence_threshold = max(0.003, noise_floor * 1.5)
        silence_ratio = float(np.mean(frame_rms < silence_threshold))
        clipping_ratio = float(np.mean(np.abs(audio) > 0.98))

        return {
            "duration_sec": round(len(audio) / sr, 2),
            "rms": round(rms, 5),
            "peak": round(peak, 4),
            "noise_floor": round(noise_floor, 5),
            "dynamic_range": round(dynamic_range, 2),
            "silence_ratio": round(silence_ratio, 3),
            "clipping_ratio": round(clipping_ratio, 4),
        }

    def suggest_vad_config(self, profile: Dict) -> Dict:
        """根据音频画像给现有 energy VAD 生成自适应参数。"""
        silence_ratio = profile.get("silence_ratio", 0.4)
        dynamic_range = profile.get("dynamic_range", 3.0)
        clipping_ratio = profile.get("clipping_ratio", 0.0)

        cfg = {
            "energy_threshold_ratio": 0.06,
            "min_speech_frames": 15,
            "min_silence_frames": 30,
            "pre_padding_ms": 220,
            "post_padding_ms": 260,
            "min_segment_duration": 0.8,
        }

        if dynamic_range < 2.0:
            # 噪声和语音区分度低：提高阈值、拉长最短语音，减少噪声误触发
            cfg.update({
                "energy_threshold_ratio": 0.085,
                "min_speech_frames": 20,
                "min_silence_frames": 36,
                "min_segment_duration": 1.0,
            })
        elif silence_ratio > 0.65:
            # 静音较多：降低切分迟滞，避免长时间等待
            cfg.update({
                "energy_threshold_ratio": 0.05,
                "min_silence_frames": 24,
                "post_padding_ms": 220,
            })
        elif silence_ratio < 0.18:
            # 连续讲话/环境声较多：增加静音帧，避免过碎切分
            cfg.update({
                "energy_threshold_ratio": 0.075,
                "min_silence_frames": 42,
                "min_segment_duration": 1.2,
            })

        if clipping_ratio > 0.01:
            cfg["energy_threshold_ratio"] = max(cfg["energy_threshold_ratio"], 0.08)

        return cfg

    # ------------------------------------------------------------------
    # Text back-end
    # ------------------------------------------------------------------

    def enhance_text(self, text: str, hotwords: Optional[List[str]] = None) -> Tuple[str, List[dict], List[str]]:
        """
        对 ASR 文本做输入法式候选纠错和规范化。

        返回: (优化文本, corrections, actions)
        """
        if not text:
            return text, [], []

        actions = []
        corrections: List[dict] = []
        enhanced = text
        active_hotwords = set(hotwords or []) | self.hotwords

        # 1. 复用现有拼音模糊匹配引擎，替代纯硬编码纠错
        if active_hotwords:
            try:
                from .hotword_engine import PinyinFuzzyMatcher
            except ImportError:
                try:
                    from modules.hotword_engine import PinyinFuzzyMatcher
                except ImportError:
                    PinyinFuzzyMatcher = None

            if PinyinFuzzyMatcher:
                matcher = PinyinFuzzyMatcher(active_hotwords, threshold=0.78)
                fuzzy = matcher.find_matches(enhanced)
                for c in fuzzy[:8]:
                    orig = c.get("original", "")
                    corr = c.get("corrected", "")
                    if orig and corr and orig != corr and orig in enhanced:
                        enhanced = enhanced.replace(orig, corr, 1)
                        corrections.append({
                            "original": orig,
                            "corrected": corr,
                            "method": c.get("method", "pinyin_fuzzy"),
                            "confidence": c.get("similarity", 0.78),
                        })
                if fuzzy:
                    actions.append("输入法式拼音候选纠错")

        # 2. 中文数字百分比规范化：百分之十五 → 15%
        percent_text, percent_corrections = self._normalize_chinese_percent(enhanced)
        if percent_corrections:
            enhanced = percent_text
            corrections.extend(percent_corrections)
            actions.append("中文数字百分比规范化")

        # 3. 常见会议/指标表达与中英文混排规范化
        normalized, norm_corrections = self._normalize_meeting_terms(enhanced)
        if norm_corrections:
            enhanced = normalized
            corrections.extend(norm_corrections)
            actions.append("会议指标与中英混排规范化")

        # 4. 英文缩写本地解释：ASR → ASR（自动语音识别）
        expanded, acronym_corrections = self._expand_acronyms(enhanced)
        if acronym_corrections:
            enhanced = expanded
            corrections.extend(acronym_corrections)
            actions.append("英文缩写本地解释")

        # 5. 去除 ASR 抖动导致的重复短片段
        deduped = self._dedupe_repeated_phrases(enhanced)
        if deduped != enhanced:
            enhanced = deduped
            actions.append("移除重复识别片段")

        return enhanced, corrections, actions

    def get_status(self) -> Dict:
        return {
            "enabled": True,
            "name": "DiTing ASR Optimizer",
            "version": "v0.2-text-normalizer",
            "capabilities": [
                "音频幅度标准化",
                "自适应 VAD 参数",
                "噪声/静音/削波质量画像",
                "拼音候选纠错",
                "中文数字百分比规范化",
                "中英文混排标准化",
                "英文缩写本地解释",
                "会议数字与术语规范化",
                "重复识别片段压缩",
            ],
            "last_report": self.last_report.to_dict() if self.last_report else None,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _quality_suggestions(self, profile: Dict) -> List[str]:
        tips = []
        if profile.get("dynamic_range", 3.0) < 2.0:
            tips.append("语音与背景噪声区分度偏低，已提高 VAD 阈值以减少误切分")
        if profile.get("silence_ratio", 0.0) > 0.7:
            tips.append("静音占比较高，已缩短断句等待以提升响应速度")
        if profile.get("silence_ratio", 0.0) < 0.15:
            tips.append("连续声源较多，已延长静音判定以避免片段过碎")
        if profile.get("clipping_ratio", 0.0) > 0.01:
            tips.append("检测到削波，建议降低麦克风输入增益")
        return tips

    def _normalize_chinese_percent(self, text: str) -> Tuple[str, List[dict]]:
        """百分之十五 / 百分之三点五 → 15% / 3.5%。"""
        corrections = []
        pattern = re.compile(r'百分之\s*([零〇一二两三四五六七八九十百千万亿点\d\.]+)')

        def repl(match):
            raw_num = match.group(1)
            value = self._parse_cn_number(raw_num)
            if value is None:
                return match.group(0)
            if abs(value - int(value)) < 1e-9:
                value_str = str(int(value))
            else:
                value_str = (f"{value:.6f}").rstrip('0').rstrip('.')
            corrected = f"{value_str}%"
            corrections.append({
                "original": match.group(0),
                "corrected": corrected,
                "method": "chinese_percent_normalize",
                "confidence": 0.96,
                "label": "中文百分比格式",
            })
            return corrected

        return pattern.sub(repl, text), corrections

    def _parse_cn_number(self, value: str) -> Optional[float]:
        value = value.strip()
        if not value:
            return None
        value = value.replace('两', '二').replace('〇', '零')
        if re.fullmatch(r'\d+(?:\.\d+)?', value):
            return float(value)

        if '点' in value:
            int_part, dec_part = value.split('点', 1)
            int_value = self._parse_cn_integer(int_part) if int_part else 0
            if int_value is None:
                return None
            decimals = []
            for ch in dec_part:
                if ch.isdigit():
                    decimals.append(ch)
                elif ch in self._CN_DIGITS:
                    decimals.append(str(self._CN_DIGITS[ch]))
                else:
                    return None
            return float(f"{int(int_value)}.{''.join(decimals) or '0'}")

        parsed = self._parse_cn_integer(value)
        return float(parsed) if parsed is not None else None

    def _parse_cn_integer(self, value: str) -> Optional[int]:
        if not value:
            return 0
        if re.fullmatch(r'\d+', value):
            return int(value)
        if HAS_CN2AN:
            try:
                return int(cn2an.cn2an(value, "smart"))
            except Exception:
                pass

        total = 0
        section = 0
        number = 0
        for ch in value:
            if ch in self._CN_DIGITS:
                number = self._CN_DIGITS[ch]
            elif ch in self._CN_UNITS:
                unit = self._CN_UNITS[ch]
                if unit >= 10000:
                    section = (section + number) * unit
                    total += section
                    section = 0
                else:
                    if number == 0:
                        number = 1
                    section += number * unit
                number = 0
            else:
                return None
        return total + section + number

    def _normalize_meeting_terms(self, text: str) -> Tuple[str, List[dict]]:
        corrections = []
        rules = [
            (r'(?i)(?<![A-Za-z])q\s*([1-4一二三四])(?![A-Za-z])', lambda m: f"Q{self._quarter_digit(m.group(1))}", "季度格式"),
            (r'(?i)(?<![A-Za-z])a\s*/?\s*b\s*测试', lambda m: "A/B 测试", "A/B 测试格式"),
        ]

        out = text
        for pattern, repl, label in rules:
            for m in list(re.finditer(pattern, out)):
                old = m.group(0)
                new = repl(m)
                if old != new:
                    corrections.append({
                        "original": old,
                        "corrected": new,
                        "method": "meeting_normalize",
                        "confidence": 0.9,
                        "label": label,
                    })
            out = re.sub(pattern, repl, out)

        # 统一常见英文缩写大小写，并在中英文之间补空格。
        def acronym_repl(match):
            old = match.group(0)
            canonical = ACRONYM_MAP.get(old.lower(), old)
            if canonical != old:
                corrections.append({
                    "original": old,
                    "corrected": canonical,
                    "method": "mixed_alnum_normalize",
                    "confidence": 0.92,
                    "label": "英文缩写标准化",
                })
            return canonical

        out = re.sub(r'(?<![A-Za-z])([A-Za-z][A-Za-z0-9]{1,14})(?![A-Za-z])', acronym_repl, out)
        spaced = self._normalize_mixed_spacing(out)
        if spaced != out:
            corrections.append({
                "original": out,
                "corrected": spaced,
                "method": "mixed_spacing_normalize",
                "confidence": 0.88,
                "label": "中英文混排空格",
            })
            out = spaced
        return out, corrections

    def _quarter_digit(self, raw: str) -> str:
        return str({"一": 1, "二": 2, "三": 3, "四": 4}.get(raw, raw))

    def _normalize_mixed_spacing(self, text: str) -> str:
        text = re.sub(r'(?<=[一-鿿])(?=[A-Za-z0-9])', ' ', text)
        text = re.sub(r'(?<=[A-Za-z0-9%])(?=[一-鿿])', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([，。！？、；：,.!?;:])', r'\1', text)
        return text.strip()

    def _expand_acronyms(self, text: str, explain_once: bool = True) -> Tuple[str, List[dict]]:
        corrections = []
        explained = set()
        pattern = re.compile(r'(?<![A-Za-z])([A-Z][A-Z0-9]{1,8}|BERT|GPT|LoRA|Transformer)(?![A-Za-z])')

        def repl(match):
            term = match.group(1)
            explanation = ACRONYM_EXPLANATIONS.get(term)
            if not explanation:
                return term
            # 如果后面已经有括号解释，不重复添加。
            tail = text[match.end():match.end() + 16]
            if tail.startswith('（') or tail.startswith('('):
                return term
            if explain_once and term in explained:
                return term
            explained.add(term)
            corrected = f"{term}（{explanation}）"
            corrections.append({
                "original": term,
                "corrected": corrected,
                "method": "local_acronym_explain",
                "confidence": 0.95,
                "label": "英文缩写本地解释",
            })
            return corrected

        return pattern.sub(repl, text), corrections

    def _dedupe_repeated_phrases(self, text: str) -> str:
        # 字符级连续重复：今天今天讨论 → 今天讨论
        text = re.sub(r'([一-鿿]{2,6})\1+', r'\1', text)
        # 英文/数字 token 连续重复
        text = re.sub(r'\b([A-Za-z0-9][A-Za-z0-9/\-]{1,})\s+\1\b', r'\1', text, flags=re.I)
        return text
