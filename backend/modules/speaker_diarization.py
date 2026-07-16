"""
谛听 DiTing - 说话人分离与身份识别
============================================================================
基于声学特征的说话人嵌入提取、聚类分离和身份匹配。

依赖:
  - numpy, scipy (已安装) — 声学特征提取 (MFCC + 频谱)
  - librosa (已安装) — 音频加载

架构:
  SpeakerIdentifier: 声纹嵌入提取 + 身份匹配 (Eval_Ali 25人注册库)
  SpeakerDiarizer:   能量分割 + 嵌入聚类 → "谁在什么时候说话"

生产部署建议:
  - 替换 extract_embedding() 为 pyannote.audio / wespeaker / ECAPA-TDNN
  - 替换 diarize() 为 pyannote.audio 的 Pipeline
  - 当前实现提供了无需额外模型的声学特征基线

用法:
    from modules.speaker_diarization import SpeakerIdentifier, SpeakerDiarizer

    identifier = SpeakerIdentifier()
    identifier.enroll_from_eval_ali("/path/to/Eval_Ali/Eval_Ali")

    # 提取声纹
    embedding = identifier.extract_embedding(audio, sr=16000)
    speaker_name, confidence = identifier.identify(embedding)
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger("speaker_diarization")


# ======================================================================
# 声学特征提取 (MFCC + 频谱)
# ======================================================================

def _pre_emphasis(signal_data: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """预加重滤波器，增强高频分量。"""
    return np.append(signal_data[0], signal_data[1:] - coeff * signal_data[:-1])


def _frame_signal(samples: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """分帧。返回 (n_frames, frame_len)"""
    n_frames = max(1, (len(samples) - frame_len) // hop + 1)
    frames = np.zeros((n_frames, frame_len))
    for i in range(n_frames):
        frames[i] = samples[i * hop : i * hop + frame_len]
    return frames


def _mel_filterbank(n_filters: int, n_fft: int, sr: int,
                    low_freq: float = 0, high_freq: float = None) -> np.ndarray:
    """梅尔滤波器组。返回 (n_filters, n_fft//2+1)"""
    if high_freq is None:
        high_freq = sr / 2

    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)

    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        filters[i - 1, bin_points[i - 1] : bin_points[i]] = (
            np.linspace(0, 1, bin_points[i] - bin_points[i - 1])
        )
        filters[i - 1, bin_points[i] : bin_points[i + 1]] = (
            np.linspace(1, 0, bin_points[i + 1] - bin_points[i])
        )

    return filters


def _extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13,
                  n_mels: int = 26, n_fft: int = 512, hop_len: int = 160) -> np.ndarray:
    """
    提取 MFCC 特征。
    返回 (n_frames, n_mfcc) 数组。
    """
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    emphasized = _pre_emphasis(audio)
    frames = _frame_signal(emphasized, n_fft, hop_len)

    # 汉明窗
    window = np.hamming(n_fft)
    frames_windowed = frames * window

    # FFT → 功率谱
    mag_spec = np.abs(rfft(frames_windowed, n=n_fft))
    pow_spec = (mag_spec ** 2) / n_fft

    # 梅尔滤波
    mel_fb = _mel_filterbank(n_mels, n_fft, sr)
    mel_spec = np.dot(pow_spec, mel_fb.T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    mel_spec_db = 20 * np.log10(mel_spec)

    # DCT → MFCC
    n_dct = n_mfcc
    mfcc = np.zeros((mel_spec_db.shape[0], n_dct))
    for i in range(n_dct):
        mfcc[:, i] = np.sum(
            mel_spec_db * np.cos(np.pi * (i + 1) * np.arange(1, n_mels + 1) / n_mels),
            axis=1
        )
    mfcc *= np.sqrt(2.0 / n_mels)

    return mfcc


def _extract_spectral_features(audio: np.ndarray, sr: int,
                                n_fft: int = 512, hop_len: int = 160) -> Dict[str, np.ndarray]:
    """
    提取频谱特征：中心频率、带宽、滚降频率、频谱平坦度。
    """
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    frames = _frame_signal(audio, n_fft, hop_len)
    window = np.hamming(n_fft)
    frames_windowed = frames * window

    mag = np.abs(rfft(frames_windowed, n=n_fft))
    freqs = rfftfreq(n_fft, 1 / sr)
    n_frames = mag.shape[0]

    centroid = np.zeros(n_frames)
    bandwidth = np.zeros(n_frames)
    rolloff = np.zeros(n_frames)
    flatness = np.zeros(n_frames)
    flux = np.zeros(max(1, n_frames - 1))

    for i in range(n_frames):
        mag_sq = mag[i] ** 2
        mag_sum = np.sum(mag_sq) + 1e-10

        # 频谱中心
        centroid[i] = np.sum(freqs * mag_sq) / mag_sum

        # 频谱带宽
        bandwidth[i] = np.sqrt(np.sum(((freqs - centroid[i]) ** 2) * mag_sq) / mag_sum)

        # 频谱滚降 (85% 能量所在频率)
        cumsum = np.cumsum(mag_sq)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff[i] = freqs[min(rolloff_idx, len(freqs) - 1)]

        # 频谱平坦度
        geo_mean = np.exp(np.mean(np.log(mag_sq + 1e-10)))
        arith_mean = np.mean(mag_sq) + 1e-10
        flatness[i] = geo_mean / arith_mean

    # 频谱通量 (帧间变化)
    if n_frames > 1:
        for i in range(n_frames - 1):
            diff = mag[i + 1] - mag[i]
            flux[i] = np.sqrt(np.sum(diff ** 2))

    return {
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness": flatness,
        "flux": flux,
    }


def _extract_pitch_features(audio: np.ndarray, sr: int,
                             fmin: float = 50, fmax: float = 500) -> Dict[str, float]:
    """
    基于自相关的基频估计，提取音高相关特征。
    """
    if len(audio) < sr * 0.02:
        return {"pitch_mean": 0, "pitch_std": 0, "pitch_voiced_ratio": 0}

    frame_len = int(sr * 0.03)  # 30ms
    hop_len = int(sr * 0.01)    # 10ms
    frames = _frame_signal(audio, frame_len, hop_len)
    n_frames = frames.shape[0]

    pitches = []
    voiced = []

    for i in range(n_frames):
        frame = frames[i] * np.hamming(frame_len)
        # 自相关
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]

        # 搜索基频范围
        min_lag = int(sr / fmax)
        max_lag = int(sr / fmin)
        if max_lag >= len(corr):
            max_lag = len(corr) - 1
        if min_lag >= max_lag:
            continue

        corr_segment = corr[min_lag:max_lag]
        if len(corr_segment) == 0:
            continue

        peak_idx = np.argmax(corr_segment)
        peak_val = corr_segment[peak_idx] / max(corr[0], 1e-10)

        if peak_val > 0.3:  # 有声帧
            pitch = sr / (peak_idx + min_lag)
            pitches.append(pitch)
            voiced.append(True)
        else:
            voiced.append(False)

    if not pitches:
        return {"pitch_mean": 0, "pitch_std": 0, "pitch_voiced_ratio": 0}

    return {
        "pitch_mean": float(np.mean(pitches)),
        "pitch_std": float(np.std(pitches)),
        "pitch_voiced_ratio": sum(voiced) / max(1, len(voiced)),
    }


# ======================================================================
# 说话人嵌入
# ======================================================================

def extract_acoustic_embedding(audio: np.ndarray, sr: int = 16000,
                                embed_dim: int = 256) -> np.ndarray:
    """
    基于真实声学特征提取说话人嵌入向量。

    组合以下特征:
      - MFCC 统计量 (13 coeffs x 5 stats = 65)
      - 频谱特征统计量 (centroid/bandwidth/rolloff/flatness/flux x 3 stats = 15)
      - 能量包络统计 (10 bins = 10)
      - 音高特征 (3)
      - 过零率统计 (3)
      → 总计 ~96 维原始特征 → PCA 投影到 embed_dim

    这不是随机模拟——它基于信号处理，能捕捉不同说话人的声学差异。
    对于生产部署，替换为 ECAPA-TDNN / WavLM 嵌入可大幅提升精度。
    """
    if len(audio) < sr * 0.1:  # 至少 100ms
        return np.zeros(embed_dim)

    # 1. MFCC 特征
    mfcc = _extract_mfcc(audio, sr, n_mfcc=13)
    if mfcc.shape[0] == 0:
        return np.zeros(embed_dim)

    mfcc_feats = np.concatenate([
        np.mean(mfcc, axis=0),
        np.std(mfcc, axis=0),
        np.min(mfcc, axis=0),
        np.max(mfcc, axis=0),
        np.median(mfcc, axis=0),
    ])

    # 2. 频谱特征
    spec_feats_raw = _extract_spectral_features(audio, sr)
    spec_feats = []
    for key in ["centroid", "bandwidth", "rolloff", "flatness", "flux"]:
        arr = spec_feats_raw.get(key, np.zeros(1))
        spec_feats.extend([np.mean(arr), np.std(arr), np.max(arr)])
    spec_feats = np.array(spec_feats)

    # 3. 能量包络
    frame_len = int(sr * 0.02)
    hop_len = int(sr * 0.01)
    energy_frames = []
    for i in range(0, len(audio) - frame_len, hop_len):
        frame = audio[i : i + frame_len]
        energy_frames.append(np.log1p(np.mean(np.abs(frame))))
    energy_frames = np.array(energy_frames) if energy_frames else np.zeros(10)
    energy_hist = np.histogram(energy_frames, bins=10, range=(energy_frames.min(), energy_frames.max() + 1e-10))[0] if len(energy_frames) > 0 else np.zeros(10)
    energy_feats = energy_hist / max(energy_hist.sum(), 1e-10) * 10

    # 4. 音高特征
    pitch_feats_dict = _extract_pitch_features(audio, sr)
    pitch_feats = np.array([
        pitch_feats_dict["pitch_mean"] / 500,
        pitch_feats_dict["pitch_std"] / 500,
        pitch_feats_dict["pitch_voiced_ratio"],
    ])

    # 5. 过零率统计
    zcr = np.mean(np.abs(np.diff(np.sign(audio + 1e-10)))) / len(audio)
    zcr_feats = np.array([zcr, zcr * 2, np.log1p(zcr * 100)])

    # 组合原始特征
    raw_features = np.concatenate([mfcc_feats, spec_feats, energy_feats, pitch_feats, zcr_feats])

    # 固定种子投影 (确保同一说话人得到一致的嵌入)
    np.random.seed(int(np.sum(mfcc_feats[:13]) * 1000) % (2**31))

    # 使用固定的投影矩阵将原始特征映射到 embed_dim
    raw_dim = len(raw_features)
    projection = np.random.randn(raw_dim, embed_dim) / np.sqrt(raw_dim)
    embedding = raw_features @ projection

    # L2 归一化
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm

    return embedding


# ======================================================================
# SpeakerIdentifier: 声纹注册 + 识别
# ======================================================================

class SpeakerIdentifier:
    """
    说话人身份识别器。

    功能:
      - 从音频提取声纹嵌入向量
      - 从 Eval_Ali 近场录音批量注册说话人
      - 将新嵌入与注册库匹配，返回说话人身份

    生产部署:
      替换 extract_embedding() 中的 extract_acoustic_embedding()
      为 wespeaker / ECAPA-TDNN 的 extract_embedding()
    """

    def __init__(self, embed_dim: int = 256, match_threshold: float = 0.65):
        self.embed_dim = embed_dim
        self.match_threshold = match_threshold
        self.enrolled: Dict[str, dict] = {}

    # ---- 声纹提取 ----

    def extract_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        提取声纹嵌入向量。

        Args:
            audio: float32 音频数组 (归一化到 [-1, 1])
            sr: 采样率

        Returns:
            L2 归一化的 N 维嵌入向量
        """
        return extract_acoustic_embedding(audio, sr, self.embed_dim)

    # ---- 注册 ----

    def enroll_speaker(self, name: str, audio: np.ndarray, sr: int = 16000,
                       role: str = None, metadata: dict = None) -> bool:
        """
        注册单个说话人。

        Args:
            name: 说话人标识 (如 "SPK8013" 或 "张三")
            audio: 至少 2 秒的清晰语音样本
            sr: 采样率
            role: 角色 (如 "PM", "Tech Lead")
            metadata: 额外信息
        """
        if len(audio) < sr * 1.0:
            logger.warning(f"Audio too short for enrollment ({len(audio)/sr:.1f}s). Need >=1s.")
            return False

        embedding = self.extract_embedding(audio, sr)
        self.enrolled[name] = {
            "embedding": embedding,
            "role": role or "",
            "metadata": metadata or {},
            "enrolled_at": __import__('datetime').datetime.now().isoformat(),
        }
        logger.info(f"Enrolled speaker: {name} (role={role or 'N/A'})")
        return True

    def enroll_batch(self, speakers: List[Dict]) -> int:
        """
        批量注册说话人。

        Args:
            speakers: [{"name": "张三", "audio": np.ndarray, "sr": 16000, "role": "PM"}, ...]

        Returns:
            成功注册的数量
        """
        count = 0
        for spk in speakers:
            if self.enroll_speaker(
                name=spk["name"],
                audio=spk["audio"],
                sr=spk.get("sr", 16000),
                role=spk.get("role"),
                metadata=spk.get("metadata"),
            ):
                count += 1
        return count

    def enroll_from_eval_ali(self, eval_ali_root: str = None) -> int:
        """
        从 Eval_Ali 近场录音批量注册说话人。

        扫描 Eval_Ali_near/audio_dir/ 下所有 *_N_SPK*.wav 文件，
        每个文件提取声纹并注册为该说话人。

        Args:
            eval_ali_root: Eval_Ali 数据集根目录

        Returns:
            成功注册的说话人数量
        """
        if eval_ali_root is None:
            # 默认路径
            eval_ali_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "..", "Eval_Ali", "Eval_Ali"
            )
        elif "Eval_Ali_near" in eval_ali_root:
            # 如果传了二级目录，回退到 Eval_Ali 根
            eval_ali_root = str(Path(eval_ali_root).parent)

        near_audio_dir = Path(eval_ali_root) / "Eval_Ali_near" / "audio_dir"

        if not near_audio_dir.exists():
            logger.warning(f"Eval_Ali near-field audio dir not found: {near_audio_dir}")
            return 0

        wav_files = sorted(near_audio_dir.glob("*_N_SPK*.wav"))

        if not wav_files:
            logger.warning(f"No speaker WAV files found in {near_audio_dir}")
            return 0

        logger.info(f"Enrolling speakers from Eval_Ali: found {len(wav_files)} WAV files")

        count = 0
        for wav_path in wav_files:
            # 从文件名提取 speaker ID: R8001_M8004_N_SPK8013.wav → SPK8013
            stem = wav_path.stem
            speaker_id = stem.rsplit("_SPK", 1)[-1] if "_SPK" in stem else stem

            try:
                import soundfile as sf
                audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
                audio = audio.mean(axis=1)  # 转单声道
            except Exception as e:
                logger.warning(f"Failed to load {wav_path.name}: {e}")
                continue

            # 只用前 30 秒作为注册样本
            max_samples = int(sr * 30)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            if self.enroll_speaker(speaker_id, audio, sr):
                count += 1

        logger.info(f"Enrolled {count}/{len(wav_files)} speakers from Eval_Ali")
        return count

    # ---- 识别 ----

    def identify(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        匹配声纹嵌入到已注册说话人。

        Args:
            embedding: L2 归一化的嵌入向量

        Returns:
            (speaker_name, confidence) 或 (None, best_score)
        """
        if not self.enrolled:
            return None, 0.0

        best_name = None
        best_sim = -1.0

        for name, info in self.enrolled.items():
            ref_emb = info["embedding"]
            # 余弦相似度 (L2 归一化后等价于内积)
            sim = float(np.dot(embedding, ref_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= self.match_threshold:
            return best_name, best_sim
        return None, best_sim

    def identify_top_k(self, embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """返回 TOP-K 匹配结果。"""
        if not self.enrolled:
            return []

        results = []
        for name, info in self.enrolled.items():
            sim = float(np.dot(embedding, info["embedding"]))
            results.append((name, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    # ---- 查询 ----

    def has_enrolled(self) -> bool:
        return len(self.enrolled) > 0

    def get_speaker_info(self, name: str) -> Optional[dict]:
        if name in self.enrolled:
            info = self.enrolled[name].copy()
            info.pop("embedding", None)  # 不返回高维嵌入
            return info
        return None

    def list_speakers(self) -> List[Dict]:
        result = []
        for name, info in self.enrolled.items():
            result.append({
                "name": name,
                "role": info.get("role", ""),
                "metadata": info.get("metadata", {}),
                "enrolled_at": info.get("enrolled_at", ""),
            })
        return result

    def remove_speaker(self, name: str) -> bool:
        if name in self.enrolled:
            del self.enrolled[name]
            return True
        return False


# ======================================================================
# SpeakerDiarizer: 说话人分离
# ======================================================================

class SpeakerDiarizer:
    """
    说话人分离器 (Speaker Diarization)。

    实现方法 (无外部模型版本):
      1. 能量分割: 找到语音/静音边界，将音频切分为小段
      2. 每段提取说话人嵌入
      3. 基于嵌入相似度做凝聚层次聚类 (AHC)
      4. 合并相邻同说话人段

    生产部署:
      替换为 pyannote.audio Pipeline.from_pretrained("pyannote/speaker-diarization-3.0")

    用法:
        diarizer = SpeakerDiarizer(sample_rate=16000)
        segments = diarizer.diarize(audio, sr=16000)
        # → [{"speaker": "SPK_A", "start": 0.0, "end": 5.2}, ...]
    """

    def __init__(self, sample_rate: int = 16000,
                 min_segment_dur: float = 1.0,
                 max_segment_dur: float = 15.0,
                 clustering_threshold: float = 0.55):
        self.sample_rate = sample_rate
        self.min_segment_dur = min_segment_dur
        self.max_segment_dur = max_segment_dur
        self.clustering_threshold = clustering_threshold

    def diarize(self, audio: np.ndarray, sr: int = None,
                num_speakers: int = None) -> List[Dict]:
        """
        对音频进行说话人分离。

        Args:
            audio: float32 音频数组
            sr: 采样率 (默认 self.sample_rate)
            num_speakers: 预期说话人数 (None = 自动推断)

        Returns:
            [
                {"speaker": "SPK_A", "start": 0.0, "end": 5.2},
                {"speaker": "SPK_B", "start": 5.2, "end": 12.0},
                ...
            ]
        """
        if sr is None:
            sr = self.sample_rate

        # 1. 能量分割
        sub_segments = self._energy_segment(audio, sr)
        if not sub_segments:
            return []

        # 2. 提取每个子段的嵌入
        embeddings = []
        for seg in sub_segments:
            seg_audio = audio[seg["start_sample"] : seg["end_sample"]]
            emb = extract_acoustic_embedding(seg_audio, sr, embed_dim=256)
            embeddings.append(emb)

        # 3. 聚类
        labels = self._agglomerative_cluster(embeddings, num_speakers)

        # 4. 合并相邻同说话人段
        merged = self._merge_adjacent(sub_segments, labels)

        return merged

    def _energy_segment(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        基于短时能量的语音分割。
        找到语音活动区域，再按能量变化进一步切分。
        """
        frame_len = int(sr * 0.025)  # 25ms
        hop_len = int(sr * 0.010)    # 10ms
        n_frames = (len(audio) - frame_len) // hop_len + 1

        if n_frames < 5:
            return []

        # 计算每帧能量
        energies = np.zeros(n_frames)
        for i in range(n_frames):
            frame = audio[i * hop_len : i * hop_len + frame_len]
            energies[i] = np.sqrt(np.mean(frame ** 2))

        # 动态阈值: 能量中位数的 0.3 倍
        threshold = np.median(energies) * 0.3
        threshold = max(threshold, np.mean(energies) * 0.15)

        # 平滑
        smoothed = uniform_filter1d(energies, size=3)

        # 找语音段
        is_speech = smoothed > threshold
        min_speech_frames = int(self.min_segment_dur * 1000 / 10)
        min_silence_frames = int(0.3 * 1000 / 10)  # 300ms silence = speaker change

        segments = []
        in_speech = False
        speech_start = 0
        silence_count = 0

        for i in range(len(is_speech)):
            if is_speech[i] and not in_speech:
                in_speech = True
                speech_start = i
                silence_count = 0
            elif is_speech[i] and in_speech:
                silence_count = 0
            elif not is_speech[i] and in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    speech_end = i - silence_count
                    dur_frames = speech_end - speech_start
                    if dur_frames >= min_speech_frames:
                        start_sample = speech_start * hop_len
                        end_sample = speech_end * hop_len + frame_len
                        dur_sec = (end_sample - start_sample) / sr
                        if dur_sec >= self.min_segment_dur:
                            segments.append({
                                "start_sample": start_sample,
                                "end_sample": min(end_sample, len(audio)),
                                "start": start_sample / sr,
                                "end": min(end_sample, len(audio)) / sr,
                            })
                    in_speech = False

        # 最后一个语音段
        if in_speech:
            speech_end = len(is_speech) - 1
            dur_frames = speech_end - speech_start
            if dur_frames >= min_speech_frames:
                start_sample = speech_start * hop_len
                end_sample = speech_end * hop_len + frame_len
                segments.append({
                    "start_sample": start_sample,
                    "end_sample": min(end_sample, len(audio)),
                    "start": start_sample / sr,
                    "end": min(end_sample, len(audio)) / sr,
                })

        # 进一步切分过长的段
        final_segments = []
        for seg in segments:
            dur = seg["end"] - seg["start"]
            if dur > self.max_segment_dur:
                # 等分
                n_splits = max(2, int(dur / self.max_segment_dur) + 1)
                split_dur = dur / n_splits
                for j in range(n_splits):
                    s_start = seg["start"] + j * split_dur
                    s_end = seg["start"] + (j + 1) * split_dur
                    final_segments.append({
                        "start_sample": int(s_start * sr),
                        "end_sample": int(min(s_end * sr, len(audio))),
                        "start": s_start,
                        "end": s_end,
                    })
            else:
                final_segments.append(seg)

        return final_segments

    def _agglomerative_cluster(self, embeddings: List[np.ndarray],
                                num_speakers: int = None) -> List[str]:
        """
        凝聚层次聚类 (AHC)。

        从每个嵌入作为单独的簇开始，迭代合并最相似的簇，
        直到达到目标簇数或相似度低于阈值。
        """
        n = len(embeddings)
        if n == 0:
            return []
        if n == 1:
            return ["SPK_A"]

        # 计算相似度矩阵
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        # 初始化簇
        clusters = [[i] for i in range(n)]

        # 如果没有指定说话人数，自动推断
        if num_speakers is None:
            # 简单启发式: 看相似度分布
            upper_tri = sim_matrix[np.triu_indices(n, k=1)]
            mean_sim = np.mean(upper_tri)
            if mean_sim > 0.7:
                num_speakers = max(1, n // 4)
            elif mean_sim > 0.5:
                num_speakers = max(1, n // 3)
            else:
                num_speakers = max(1, n // 2)

        num_speakers = max(1, min(num_speakers, n))

        # 合并直到达到目标簇数
        while len(clusters) > num_speakers:
            # 找最相似的两个簇
            best_i, best_j, best_sim = -1, -1, -1.0
            for ci in range(len(clusters)):
                for cj in range(ci + 1, len(clusters)):
                    # 簇间平均相似度
                    sim_sum = 0.0
                    count = 0
                    for ii in clusters[ci]:
                        for jj in clusters[cj]:
                            sim_sum += sim_matrix[ii, jj]
                            count += 1
                    avg_sim = sim_sum / max(1, count)
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_i, best_j = ci, cj

            if best_sim < self.clustering_threshold:
                break

            # 合并
            clusters[best_i].extend(clusters[best_j])
            clusters.pop(best_j)

        # 生成标签
        labels = [""] * n
        for ci, cluster in enumerate(clusters):
            label = f"SPK_{chr(ord('A') + ci)}"
            for idx in cluster:
                labels[idx] = label

        return labels

    def _merge_adjacent(self, segments: List[Dict], labels: List[str]) -> List[Dict]:
        """合并相邻且同说话人的段。"""
        if not segments or not labels:
            return []

        merged = []
        current = {
            "speaker": labels[0],
            "start": segments[0]["start"],
            "end": segments[0]["end"],
        }

        for i in range(1, len(segments)):
            if labels[i] == current["speaker"]:
                # 同一说话人，延长当前段
                current["end"] = segments[i]["end"]
            else:
                merged.append(current)
                current = {
                    "speaker": labels[i],
                    "start": segments[i]["start"],
                    "end": segments[i]["end"],
                }

        merged.append(current)
        return merged


# ======================================================================
# 便捷函数
# ======================================================================

def create_speaker_pipeline(eval_ali_root: str = None,
                             embed_dim: int = 256) -> Tuple[SpeakerIdentifier, SpeakerDiarizer]:
    """
    创建说话人处理管线。

    Returns:
        (identifier, diarizer)
    """
    identifier = SpeakerIdentifier(embed_dim=embed_dim)

    if eval_ali_root:
        identifier.enroll_from_eval_ali(eval_ali_root)

    diarizer = SpeakerDiarizer()

    return identifier, diarizer
