"""
谛听 DiTing - 音频处理模块
模拟说话人分离、环境感知、热词纠偏和逻辑校验
在实际部署中，这些会连接真实的 Whisper + pyannote 模型
"""

import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# 1. 声学环境感知
# ============================================================

@dataclass
class AcousticEnv:
    snr_db: float
    rt60: float  # 混响时间
    scene_type: str  # 声学场景类型
    quality_score: float  # 0-1 综合质量分


def estimate_snr(audio: np.ndarray, sr: int = 16000) -> float:
    """
    WADA SNR 估计算法
    基于波形幅度分布分析，不需要模型
    """
    if len(audio) == 0:
        return 30.0

    # 分帧
    frame_len = int(sr * 0.02)  # 20ms
    n_frames = len(audio) // frame_len
    if n_frames < 2:
        return 30.0

    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    energy = np.mean(np.abs(frames), axis=1)

    # 基于能量中位数分离信号帧和噪声帧
    threshold = np.median(energy)
    speech_frames = frames[energy > threshold]
    noise_frames = frames[energy <= threshold]

    if len(speech_frames) == 0 or len(noise_frames) == 0:
        return 30.0

    signal_power = np.mean(speech_frames ** 2)
    noise_power = np.mean(noise_frames ** 2)

    if noise_power < 1e-10:
        return 30.0

    snr = 10 * np.log10(signal_power / noise_power)
    return max(0.0, min(60.0, snr))


def estimate_rt60(audio: np.ndarray, sr: int = 16000) -> float:
    """
    基于衰减率估计混响时间 RT60
    使用 Schroeder 反向积分法简化版
    """
    if len(audio) < sr:
        return 0.3  # 默认干声

    # 简化实现: 基于能量衰减曲线
    frame_len = int(sr * 0.01)  # 10ms
    n_frames = len(audio) // frame_len
    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    energy = np.mean(frames ** 2, axis=1)

    # 找到能量峰值位置
    peak_idx = np.argmax(energy)

    # 从峰值开始, 找到能量下降到 -60dB 的时间
    peak_energy = energy[peak_idx]
    if peak_energy < 1e-10:
        return 0.3

    # 反向累积能量 (Schroeder积分)
    schroeder = np.cumsum(energy[peak_idx:][::-1])[::-1]
    schroeder_db = 10 * np.log10(np.maximum(schroeder, 1e-10))
    schroeder_db -= schroeder_db[0]  # 归一化到 0dB

    # 找到 -60dB 点
    idx_60db = np.argmax(schroeder_db < -60)
    if idx_60db == 0:
        idx_60db = len(schroeder_db) - 1

    rt60 = idx_60db * 0.01  # 10ms per frame
    return max(0.1, min(3.0, rt60))


def classify_scene(snr_db: float, rt60: float) -> str:
    """简化的声学场景分类（实际部署用BEATs）"""
    if rt60 > 1.0:
        if snr_db < 10:
            return "大型会议室(高混响+噪声)"
        return "报告厅(高混响)"
    elif rt60 > 0.5:
        if snr_db < 12:
            return "中型会议室(中混响+噪声)"
        return "中型会议室"
    else:
        if snr_db < 8:
            return "嘈杂环境(低SNR)"
        elif snr_db < 18:
            return "普通室内(中SNR)"
        return "安静室内"


def acoustic_quality_score(snr_db: float, rt60: float, overlap_ratio: float = 0.0) -> float:
    """
    综合声学质量评分 (0-1)
    SNR权重 50%, 混响权重 30%, 重叠权重 20%
    """
    # SNR评分: 0dB->0, 30dB->1
    snr_score = min(1.0, max(0.0, snr_db / 30.0))

    # 混响评分: 2s->0, 0.2s->1
    rt60_score = min(1.0, max(0.0, 1.0 - (rt60 - 0.2) / 1.8))

    # 重叠评分
    overlap_score = 1.0 - overlap_ratio

    return 0.5 * snr_score + 0.3 * rt60_score + 0.2 * overlap_score


# ============================================================
# 2. 说话人嵌入 (v3.1 — 真实声学特征)
# ============================================================

# 委托给 speaker_diarization 模块的真实实现
# 保留此类作为向后兼容的别名
try:
    from .speaker_diarization import SpeakerIdentifier as SpeakerEmbedder
except ImportError:
    # Fallback: 如果 speaker_diarization.py 不可用，保留最小实现
    class SpeakerEmbedder:
        """向后兼容的说话人嵌入包装器 (请使用 speaker_diarization.SpeakerIdentifier)"""

        def __init__(self):
            self.registered_speakers: Dict[str, np.ndarray] = {}

        def extract_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
            if len(audio) == 0:
                return np.zeros(256)
            energy = np.log1p(np.mean(np.abs(audio)))
            zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / len(audio)
            np.random.seed(int(energy * 1000) % (2**31))
            embedding = np.random.randn(256) * 0.1
            embedding[0] = energy * 0.5
            embedding[1] = zcr * 10
            return embedding / (np.linalg.norm(embedding) + 1e-8)

        def verify_speaker(self, embedding: np.ndarray, threshold: float = 0.6) -> Tuple[Optional[str], float]:
            best_id = None
            best_sim = 0.0
            for spk_id, ref_emb in self.registered_speakers.items():
                sim = np.dot(embedding, ref_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = spk_id
            if best_sim > threshold:
                return best_id, best_sim
            return None, best_sim


# ============================================================
# 3. 热词纠偏
# ============================================================

class HotwordCorrector:
    """
    领域热词纠偏引擎
    方法1: 拼音相似度匹配 (确定性热词)
    方法2: LLM 上下文推理 (歧义热词)
    """

    def __init__(self, hotwords: List[str]):
        self.hotwords = set(hotwords)
        self.pinyin_map = self._build_pinyin_map()

    def _build_pinyin_map(self) -> Dict[str, List[str]]:
        """构建简化拼音相似度映射表"""
        # 常见 ASR 错误映射
        return {
            "bat": ["BERT", "BAT"],
            "bate": ["BERT"],
            "bot": ["BERT"],
            "chuansi": ["Transformer"],
            "chuan si former": ["Transformer"],
            "ab": ["A/B"],
            "ab测试": ["A/B测试"],
            "weidiao": ["微调"],
            "duomotai": ["多模态"],
            "zhuanhualv": ["转化率"],
            "OKR": ["OKR"],
            "okr": ["OKR"],
            "q3": ["Q3"],
            "Q3": ["Q3"],
        }

    def pinyin_correct(self, text: str) -> List[dict]:
        """基于拼音相似度的热词纠偏"""
        corrections = []
        words = text.lower().split()

        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w/]', '', word)
            if clean_word in self.pinyin_map:
                candidates = self.pinyin_map[clean_word]
                # 选择在热词表中的候选
                for c in candidates:
                    if c in self.hotwords:
                        corrections.append({
                            'position': i,
                            'original': word,
                            'corrected': c,
                            'method': 'pinyin_match',
                            'confidence': 0.85
                        })
                        break

        return corrections

    def llm_context_correct(self, text: str, low_conf_spans: List[dict]) -> List[dict]:
        """
        基于 LLM 上下文的纠偏
        实际部署: Qwen2-1.5B 推理
        这里提供基于规则的模拟
        """
        corrections = []

        for span in low_conf_spans:
            span_text = span.get('text', '')
            context = span.get('context', text)

            # 模拟 LLM 推理: 基于上下文关键词选择最佳候选
            candidates = span.get('candidates', [])
            if not candidates:
                continue

            # 简单的上下文匹配
            best_candidate = candidates[0]  # 默认第一个

            if '用户' in context and '全量' in candidates:
                best_candidate = '全量用户'
            elif '用户' in context and '新增' in candidates:
                best_candidate = '新增用户'
            elif '训练' in context and '模型' in context:
                if 'BERT' in candidates:
                    best_candidate = 'BERT'
                elif 'Transformer' in candidates:
                    best_candidate = 'Transformer'

            corrections.append({
                'position': span.get('start', 0),
                'original': span_text,
                'corrected': best_candidate,
                'method': 'llm_context',
                'confidence': 0.72
            })

        return corrections


# ============================================================
# 4. 逻辑校验引擎
# ============================================================

class LogicValidator:
    """
    会议场景逻辑校验引擎
    校验维度:
      1. 跨发言数据矛盾
      2. 数学计算一致性
      3. 时序逻辑
      4. 术语一致性
    """

    def __init__(self):
        self.claimed_data: List[dict] = []  # 已声明数据点
        self.speaker_statements: Dict[str, List[str]] = {}  # 各说话人发言

    def add_statement(self, speaker_id: str, text: str, data_points: List[dict],
                      timestamp: float) -> List[dict]:
        """添加一条发言，返回检测到的逻辑冲突"""
        flags = []

        # 记录说话人发言
        if speaker_id not in self.speaker_statements:
            self.speaker_statements[speaker_id] = []
        self.speaker_statements[speaker_id].append(text)

        # 校验1: 数据矛盾检测
        for dp in data_points:
            value = dp.get('value', '')
            dtype = dp.get('type', '')

            # 提取数值
            numeric_val = self._parse_numeric(value)
            if numeric_val is None:
                continue

            # 检查是否与已有数据冲突
            for prev in self.claimed_data:
                prev_val = self._parse_numeric(prev.get('value', ''))
                if prev_val is None:
                    continue

                # 同类型数据对比
                if prev.get('type') == dtype and dtype in ('result', 'total', 'calculated_total'):
                    diff_pct = abs(numeric_val - prev_val) / max(prev_val, 1) * 100
                    if diff_pct > 20:
                        flags.append({
                            'type': 'data_conflict',
                            'severity': 'warning',
                            'message': f'数据矛盾: 当前值 {value} vs 已有值 {prev["value"]} (差异 {diff_pct:.0f}%)',
                            'calculation': f'|{value} - {prev["value"]}| / {prev["value"]} = {diff_pct:.0f}%'
                        })

                # 预算类型: 检查是否超额
                if dtype == 'dept_budget' and prev.get('type') == 'total_budget':
                    dept = dp.get('dept', '未知')
                    # 收集所有部门预算
                    all_dept_budgets = [self._parse_numeric(d['value'])
                                       for d in self.claimed_data
                                       if d.get('type') == 'dept_budget']
                    all_dept_budgets = [b for b in all_dept_budgets if b is not None]
                    all_dept_budgets.append(numeric_val)

                    total_budget = prev_val
                    claimed_total = sum(all_dept_budgets)

                    if claimed_total > total_budget:
                        flags.append({
                            'type': 'budget_overflow',
                            'severity': 'warning',
                            'message': f'⚠️ 预算超额: 申报总额{claimed_total:.0f}万 > 总预算{total_budget:.0f}万 (超出{((claimed_total-total_budget)/total_budget*100):.0f}%)',
                            'calculation': f'各部门预算之和 = {claimed_total:.0f}万 > {total_budget:.0f}万'
                        })
                    elif claimed_total == total_budget:
                        flags.append({
                            'type': 'budget_resolved',
                            'severity': 'resolved',
                            'message': f'✅ 预算达成一致: 各部门合计{claimed_total:.0f}万 = 总预算{total_budget:.0f}万',
                            'calculation': f'预算平衡 ✓'
                        })

            self.claimed_data.append(dp)

        # 校验2: 自相矛盾检测 (同一说话人)
        if len(self.speaker_statements[speaker_id]) >= 2:
            recent = self.speaker_statements[speaker_id][-2:]
            # 简化: 检查最近两句是否包含互相矛盾的数值
            nums1 = self._extract_numbers(recent[0])
            nums2 = self._extract_numbers(recent[1])
            for n1 in nums1:
                for n2 in nums2:
                    if n1 > 0 and n2 > 0:
                        diff = abs(n2 - n1) / max(n1, 1)
                        if diff > 0.5:  # 50% 差异
                            flags.append({
                                'type': 'self_contradiction',
                                'severity': 'warning',
                                'message': f'同一说话人数据可能存在矛盾 (数值差异 {diff*100:.0f}%)',
                                'calculation': f'{n1} vs {n2}'
                            })

        return flags

    def _parse_numeric(self, value_str: str) -> Optional[float]:
        """从字符串中提取数值 (支持 万/亿/%)"""
        if not value_str:
            return None
        match = re.search(r'[\d.]+', str(value_str))
        if match:
            return float(match.group())
        return None

    def _extract_numbers(self, text: str) -> List[float]:
        """提取文本中的所有数值"""
        numbers = re.findall(r'[\d.]+', text)
        return [float(n) for n in numbers if float(n) > 0]

    def reset(self):
        self.claimed_data = []
        self.speaker_statements = {}


# ============================================================
# 5. 不确定性估计与校准
# ============================================================

class UncertaintyEstimator:
    """
    多源不确定性估计器
    - 数据不确定性 (Aleatoric): 来自输入质量
    - 模型不确定性 (Epistemic): 来自分布外检测
    """

    def estimate(self, snr_db: float, rt60: float, overlap_ratio: float,
                 asr_confidence: float, is_ood: bool = False) -> dict:
        """
        综合不确定性估计
        返回校准后的不确定性得分 + 分级
        """
        # 数据不确定性 (输入质量驱动)
        aleatoric = 0.0

        # SNR 因素
        if snr_db < 5:
            aleatoric += 0.40
        elif snr_db < 12:
            aleatoric += 0.25
        elif snr_db < 20:
            aleatoric += 0.10

        # 混响因素
        if rt60 > 1.2:
            aleatoric += 0.15
        elif rt60 > 0.7:
            aleatoric += 0.08

        # 重叠因素
        aleatoric += overlap_ratio * 0.20

        # 模型不确定性
        epistemic = 0.20 if is_ood else 0.0

        # ASR 不确定性
        asr_uncertainty = max(0.0, 1.0 - asr_confidence) * 0.5

        # 联合不确定性
        total = min(1.0, aleatoric + epistemic + asr_uncertainty)

        # 分级
        if total < 0.15:
            level = "L0_高确信"
            action = "静默接受"
        elif total < 0.40:
            level = "L1_中确信"
            action = "接受并标记"
        elif total < 0.65:
            level = "L2_低确信"
            action = "主动确认"
        else:
            level = "L3_极低确信"
            action = "请求复述/拒绝"

        return {
            'total_uncertainty': total,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'asr_uncertainty': asr_uncertainty,
            'level': level,
            'action': action,
            'calibrated_confidence': max(0.0, 1.0 - total)
        }


# ============================================================
# 6. 联合会议处理管道
# ============================================================

class MeetingProcessor:
    """
    谛听核心处理管道
    串联: 环境感知 → 说话人分离 → ASR → 热词纠偏 → 逻辑校验 → 不确定性估计
    """

    def __init__(self, hotwords: List[str] = None):
        self.hotword_corrector = HotwordCorrector(hotwords or [])
        self.logic_validator = LogicValidator()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.speaker_embedder = SpeakerEmbedder()

    def process_segment(self, segment: dict) -> dict:
        """
        处理单个语音片段
        输入: {'audio': np.ndarray, 'speaker': str, 'asr_text': str, ...}
        输出: 全面增强后的结果
        """
        audio = segment.get('audio', np.array([]))
        sr = segment.get('sr', 16000)
        asr_text = segment.get('asr_text', '')
        asr_confidence = segment.get('asr_confidence', 0.8)

        # Step 1: 声学环境评估
        snr_db = estimate_snr(audio, sr)
        rt60 = estimate_rt60(audio, sr)
        overlap_ratio = segment.get('overlap_ratio', 0.0)
        quality = acoustic_quality_score(snr_db, rt60, overlap_ratio)

        # Step 2: 热词纠偏
        corrections = self.hotword_corrector.pinyin_correct(asr_text)
        corrected_text = asr_text
        for c in reversed(corrections):
            # Apply corrections
            pass

        # Step 3: 逻辑校验
        logic_flags = self.logic_validator.add_statement(
            segment.get('speaker', 'unknown'),
            asr_text,
            segment.get('data_points', []),
            segment.get('start', 0)
        )

        # Step 4: 不确定性估计
        uncertainty = self.uncertainty_estimator.estimate(
            snr_db, rt60, overlap_ratio, asr_confidence
        )

        return {
            'snr_db': round(snr_db, 1),
            'rt60': round(rt60, 2),
            'quality_score': round(quality, 2),
            'quality_label': 'high' if quality > 0.7 else ('medium' if quality > 0.4 else 'low'),
            'corrections': corrections,
            'logic_flags': logic_flags,
            'uncertainty': uncertainty,
            'corrected_text': corrected_text,
        }

    def reset(self):
        self.logic_validator.reset()
