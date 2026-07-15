"""
谛听 DiTing - 认知引擎
============================================================================
内容预测 + 领域推断 + 会议摘要生成。

这是 DiTing v3.1 的核心增值模块，将 ASR 系统从"听写工具"
升级为"会议认知系统"。

依赖 LLM API (llm_client.py) 和领域分类 (domain_taxonomy.py)。
所有 LLM 调用都有规则降级路径。

用法:
    from modules.cognitive_engine import (
        ContentPredictor, DomainInferrer, MeetingSummarizer
    )

    # 内容预测
    predictor = ContentPredictor(llm_client)
    upcoming_terms = predictor.predict_upcoming_terms(
        speaker_name="张三", speaker_role="PM",
        recent_context=["本季度转化率从12%下降到了..."]
    )

    # 领域推断
    inferrer = DomainInferrer(taxonomy=DOMAIN_TAXONOMY)
    result = inferrer.infer(["转化率", "A/B测试", "DAU"])

    # 会议摘要
    summarizer = MeetingSummarizer(llm_client)
    summary = summarizer.summarize(meeting_data)
"""

import logging
from typing import List, Dict, Tuple, Optional

from .domain_taxonomy import DOMAIN_TAXONOMY, match_domain, get_domain_keywords

logger = logging.getLogger("cognitive_engine")


# ======================================================================
# ContentPredictor: 内容预测
# ======================================================================

class ContentPredictor:
    """
    基于说话人身份和上下文的术语预测。

    用途:
      - 预测说话人接下来可能提到的关键词
      - 提前加载预测词为热词，提升后续 ASR 准确率
      - 上下文消歧：判断歧义词的真实含义

    工作模式:
      - LLM 可用: 用 LLM 做语义推理
      - LLM 不可用: 用规则引擎从 domain_taxonomy 做关键词匹配
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLMClient 实例 (默认从 llm_client.get_llm_client() 获取)
        """
        if llm_client is None:
            try:
                from .llm_client import get_llm_client
                llm_client = get_llm_client()
            except ImportError:
                llm_client = None
        self.llm = llm_client

    def predict_upcoming_terms(
        self,
        speaker_name: str = "",
        speaker_role: str = "",
        recent_context: List[str] = None,
        meeting_topic: str = None,
    ) -> List[str]:
        """
        预测说话人接下来可能提到的术语。

        Args:
            speaker_name: 说话人姓名
            speaker_role: 说话人角色
            recent_context: 最近 3-5 句发言
            meeting_topic: 会议主题 (可选)

        Returns:
            预测的术语列表 (5-10 个)
        """
        if not recent_context:
            return []

        context_text = " ".join(recent_context)
        if len(context_text) < 10:
            return []

        # 尝试 LLM
        if self.llm and self.llm.is_available:
            try:
                return self._predict_with_llm(
                    speaker_name, speaker_role, context_text, meeting_topic
                )
            except Exception as e:
                logger.debug(f"LLM content prediction failed, using rules: {e}")

        # 规则降级
        return self._predict_with_rules(context_text)

    def _predict_with_llm(
        self, speaker_name: str, speaker_role: str,
        context_text: str, meeting_topic: str = None,
    ) -> List[str]:
        """用 LLM 预测即将提到的术语。"""
        role_info = f"{speaker_name}" + (f"（{speaker_role}）" if speaker_role else "")
        topic_info = f"当前会议主题是{meeting_topic}。" if meeting_topic else ""

        prompt = f"""你是会议助手。{topic_info}
说话人{role_info}最近说:
{context_text}

请预测该说话人接下来可能提到的 5-10 个专业术语或关键词。
只返回术语列表，每行一个，不要编号和解释。"""

        result = self.llm.chat(prompt, system="你是会议分析和术语预测专家。",
                               temperature=0.1, max_tokens=256)
        if not result:
            return []

        # 解析：每行一个术语
        terms = [line.strip().strip('-,.，。1234567890.、*"\'')
                 for line in result.split("\n")]
        return [t for t in terms if len(t) >= 2][:10]

    def _predict_with_rules(self, context_text: str) -> List[str]:
        """规则引擎：从 taxonomy 中找与上下文最相关的领域关键词。"""
        # 从 domain_taxonomy 做简单的关键词匹配
        candidates = []
        for domain, info in DOMAIN_TAXONOMY.items():
            keywords = info.get("keywords", [])
            matched = [kw for kw in keywords if kw.lower() in context_text.lower()]
            if matched:
                # 返回同领域其他可能的关键词
                other = [kw for kw in keywords if kw not in matched]
                candidates.extend(other[:5])  # 每个领域取 5 个

        # 去重并限制数量
        seen = set()
        result = []
        for c in candidates:
            if c not in seen and c.lower() not in context_text.lower():
                seen.add(c)
                result.append(c)
        return result[:10]

    def contextual_disambiguation(
        self, ambiguous_term: str, context: str, candidates: List[str],
    ) -> Optional[str]:
        """
        上下文消歧。

        例如: "这个 bat 模型" → 根据上下文 (模型/BERT/Transformer)
        判断 bat = BERT 而非 BAT (百度/阿里/腾讯)

        Args:
            ambiguous_term: 歧义词 (如 "bat")
            context: 上下文文本
            candidates: 候选正确词 (如 ["BERT", "BAT"])

        Returns:
            最可能的正确词，或 None
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # 尝试 LLM
        if self.llm and self.llm.is_available:
            try:
                prompt = f"""请根据上下文判断歧义词的正确含义。

上下文: {context}
歧义词: "{ambiguous_term}"
候选含义: {candidates}

请只返回最可能的一个候选词，不要解释。"""
                result = self.llm.chat(prompt, temperature=0.0, max_tokens=32)
                result = result.strip().strip("'\"")
                for c in candidates:
                    if c in result:
                        return c
            except Exception:
                pass

        # 规则降级：选与上下文重叠最多的候选
        context_lower = context.lower()
        best = None
        best_score = 0
        for c in candidates:
            score = sum(1 for ch in c.lower() if ch in context_lower) / max(1, len(c))
            if score > best_score:
                best_score = score
                best = c

        return best if best_score > 0.3 else candidates[0]


# ======================================================================
# DomainInferrer: 反向领域推断
# ======================================================================

class DomainInferrer:
    """
    热词 → 领域反向推断。

    从会议中提取的热词列表推断会议所属领域/子领域，
    并加载该领域的专业术语词典。

    工作模式:
      - LLM 可用: LLM 语义推理 (更准确，能发现隐含领域)
      - LLM 不可用: 规则匹配 domain_taxonomy (快速，基于关键词交集)
    """

    def __init__(self, taxonomy: dict = None, llm_client=None):
        """
        Args:
            taxonomy: 领域分类 dict (默认 DOMAIN_TAXONOMY)
            llm_client: LLMClient 实例
        """
        self.taxonomy = taxonomy or DOMAIN_TAXONOMY

        if llm_client is None:
            try:
                from .llm_client import get_llm_client
                llm_client = get_llm_client()
            except ImportError:
                llm_client = None
        self.llm = llm_client

    def infer(self, hotwords: List[str]) -> dict:
        """
        推断会议领域。

        Args:
            hotwords: 从会议中提取的热词列表

        Returns:
            {
                "domain": "互联网产品",
                "confidence": 0.85,
                "sub_domains": ["增长运营"],
                "matched_terms": ["转化率", "DAU", ...],
                "method": "llm" | "rule",
                "candidates": [{"domain": "...", "score": 0.xx}, ...],
            }
        """
        if not hotwords:
            return {
                "domain": None, "confidence": 0, "sub_domains": [],
                "matched_terms": [], "method": "none", "candidates": [],
            }

        # 尝试 LLM
        if self.llm and self.llm.is_available:
            try:
                return self._infer_with_llm(hotwords)
            except Exception as e:
                logger.debug(f"LLM domain inference failed, using rules: {e}")

        # 规则降级
        return self._infer_with_rules(hotwords)

    def _infer_with_llm(self, hotwords: List[str]) -> dict:
        """用 LLM 推断领域。"""
        domain_list = "\n".join(f"- {d}: {self.taxonomy[d].get('description', '')}"
                                for d in self.taxonomy)
        hotword_str = ", ".join(hotwords[:30])

        prompt = f"""请根据以下会议中提取的关键术语，判断会议最可能属于哪个领域。

关键术语: {hotword_str}

可选领域:
{domain_list}

请返回 JSON 格式 (只返回 JSON):
{{
  "domain": "最可能的领域名",
  "confidence": 0.0-1.0,
  "sub_domains": ["子领域1", "子领域2"],
  "reason": "判断理由 (一句话)"
}}"""

        result = self.llm.chat_json(prompt, system="你是领域分类专家。", temperature=0.1)
        if result and "domain" in result:
            domain = result["domain"]
            # 验证领域名是否在 taxonomy 中
            if domain not in self.taxonomy:
                # 模糊匹配
                for d in self.taxonomy:
                    if d in domain or domain in d:
                        domain = d
                        break
                else:
                    domain = list(self.taxonomy.keys())[0]

            return {
                "domain": domain,
                "confidence": round(result.get("confidence", 0.7), 2),
                "sub_domains": result.get("sub_domains", []),
                "matched_terms": hotwords[:10],
                "method": "llm",
                "reason": result.get("reason", ""),
                "candidates": [],
            }

        # LLM JSON 解析失败，降级
        return self._infer_with_rules(hotwords)

    def _infer_with_rules(self, hotwords: List[str]) -> dict:
        """规则引擎：关键词交集匹配。"""
        matches = match_domain(hotwords)

        if not matches:
            return {
                "domain": None, "confidence": 0, "sub_domains": [],
                "matched_terms": [], "method": "rule", "candidates": [],
            }

        top_domain, top_score, matched = matches[0]
        sub_domains = self.taxonomy.get(top_domain, {}).get("sub_domains", [])

        return {
            "domain": top_domain,
            "confidence": min(1.0, top_score),
            "sub_domains": sub_domains[:3],
            "matched_terms": matched[:10],
            "method": "rule",
            "candidates": [
                {"domain": d, "score": s} for d, s, _ in matches[:3]
            ],
        }

    def load_domain_hotwords(self, domain: str, max_count: int = 100) -> List[str]:
        """加载指定领域的专业术语词典。"""
        keywords = get_domain_keywords(domain)
        return keywords[:max_count]


# ======================================================================
# MeetingSummarizer: 会议摘要生成
# ======================================================================

class MeetingSummarizer:
    """
    基于 LLM 的会议摘要生成。

    从完整的会议转写中提取:
      - summary: 会议总体概述
      - topics: 讨论主题列表
      - decisions: 做出的决策
      - action_items: 行动项 (谁做什么、截止时间)
      - key_metrics: 关键数据指标
      - data_conflicts: 数据矛盾及解决情况

    无 LLM 时从 LogicValidator 提取已有标记作为降级摘要。
    """

    def __init__(self, llm_client=None):
        if llm_client is None:
            try:
                from .llm_client import get_llm_client
                llm_client = get_llm_client()
            except ImportError:
                llm_client = None
        self.llm = llm_client

    def summarize(self, meeting_data: dict) -> dict:
        """
        生成会议摘要。

        Args:
            meeting_data: {
                "title": str,
                "participants": [{"id": str, "name": str, "role": str}, ...],
                "domain": {"domain": str, ...} or None,
                "segments": [
                    {"speaker": str, "text": str, "start": float, "end": float},
                    ...
                ],
                "logic_flags": [...],
            }

        Returns:
            {
                "summary": str,
                "topics": [str],
                "decisions": [{"decision": str, "by": str}],
                "action_items": [{"task": str, "assignee": str, "deadline": str}],
                "key_metrics": [{"metric": str, "value": str, "context": str}],
                "data_conflicts": [{"description": str, "resolution": str}],
                "next_meeting": str,
            }
        """
        # 尝试 LLM
        if self.llm and self.llm.is_available:
            try:
                return self._summarize_with_llm(meeting_data)
            except Exception as e:
                logger.debug(f"LLM summarization failed, using rules: {e}")

        # 规则降级
        return self._summarize_with_rules(meeting_data)

    def _summarize_with_llm(self, meeting_data: dict) -> dict:
        """用 LLM 生成结构化摘要。"""
        # 委托给 llm_client 的 summarize 方法
        result = self.llm.summarize(
            transcript=meeting_data.get("segments", []),
            meeting_title=meeting_data.get("title", ""),
        )

        # 确保返回结构完整
        defaults = {
            "summary": "", "topics": [], "decisions": [],
            "action_items": [], "key_metrics": [],
            "data_conflicts": [], "next_meeting": "",
        }
        for key in defaults:
            if key not in result:
                result[key] = defaults[key]

        return result

    def _summarize_with_rules(self, meeting_data: dict) -> dict:
        """
        规则引擎降级摘要。

        从已有信息中提取:
          - segments 中的 terms (热词) → topics
          - logic_flags → data_conflicts
          - 最长的 segments → 拼接成 summary
        """
        segments = meeting_data.get("segments", [])
        logic_flags = meeting_data.get("logic_flags", [])
        domain = meeting_data.get("domain", {})

        # 从 segments 提取 topics
        all_terms = set()
        for seg in segments:
            if isinstance(seg, dict):
                all_terms.update(seg.get("terms", []))
            elif hasattr(seg, 'terms'):
                all_terms.update(seg.terms)

        topics = list(all_terms)[:5]

        # 从 logic_flags 提取 conflicts
        conflicts = []
        for flag in logic_flags:
            if isinstance(flag, dict):
                conflicts.append({
                    "description": flag.get("message", ""),
                    "resolution": flag.get("resolution", "Pending"),
                })

        # summary: 拼接前 3 句最长发言
        sorted_segs = sorted(
            [s.get("text", "") if isinstance(s, dict) else getattr(s, 'text', '')
             for s in segments],
            key=len, reverse=True
        )
        summary = " | ".join(sorted_segs[:3]) if sorted_segs else ""

        # domain info
        domain_str = (
            f"领域: {domain.get('domain', '未知')}"
            if domain else "领域: 未识别"
        )

        return {
            "summary": f"[{domain_str}]\n{summary}" if summary else domain_str,
            "topics": topics,
            "decisions": [],
            "action_items": self._extract_actions(segments),
            "key_metrics": [],
            "data_conflicts": conflicts,
            "next_meeting": "",
        }

    def _extract_actions(self, segments: list) -> list:
        """使用 ActionExtractor 从 segments 中提取行动项。"""
        try:
            from .action_extractor import extract_action_items
            return extract_action_items(segments)
        except Exception as e:
            logger.debug(f"Action extraction failed: {e}")
            return []


# ======================================================================
# 便捷函数
# ======================================================================

def create_cognitive_pipeline(llm_client=None) -> dict:
    """
    创建完整的认知管线。

    Returns:
        {
            "predictor": ContentPredictor,
            "inferrer": DomainInferrer,
            "summarizer": MeetingSummarizer,
        }
    """
    return {
        "predictor": ContentPredictor(llm_client),
        "inferrer": DomainInferrer(llm_client=llm_client),
        "summarizer": MeetingSummarizer(llm_client),
    }
