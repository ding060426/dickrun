"""
Action Item Extractor: 从会议转写中自动提取行动项/TODO。

提取维度：
  - assignee: 谁负责
  - task: 做什么
  - deadline: 截止时间
  - priority: 优先级 (high/medium/low)
  - source_text: 相关原文片段
  - speaker: 发言人

工作模式：
  - LLM 可用: 用 LLM 做语义理解提取（更准确）
  - LLM 不可用: 用规则引擎模式匹配（零延迟）

规则引擎基于中文会议常用表达模式：
  - "XX负责..."
  - "XX来做..."
  - "需要XX去..."
  - "下周/周五前/月底前完成"
  - "XX跟进一下"
"""

import re
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("diting")


class ActionItem:
    """行动项数据结构。"""

    def __init__(
        self,
        task: str,
        assignee: str = "",
        deadline: str = "",
        priority: str = "medium",
        source_text: str = "",
        speaker: str = "",
        segment_index: int = -1,
    ):
        self.task = task
        self.assignee = assignee
        self.deadline = deadline
        self.priority = priority
        self.source_text = source_text
        self.speaker = speaker
        self.segment_index = segment_index

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "assignee": self.assignee,
            "deadline": self.deadline,
            "priority": self.priority,
            "source_text": self.source_text,
            "speaker": self.speaker,
            "segment_index": self.segment_index,
        }

    def __repr__(self):
        return f"ActionItem(task='{self.task}', assignee='{self.assignee}', deadline='{self.deadline}', priority='{self.priority}')"


class ActionExtractor:
    """
    行动项提取器。

    用法:
        extractor = ActionExtractor()
        actions = extractor.extract(segments)
        # segments: [{"speaker": "张三", "text": "...", "start": 0.0, "end": 5.0}, ...]
    """

    # ── 责任人模式 ──────────────────────────────────────────────
    # 匹配 "张三负责" / "李四来做" / "王五跟进" 等
    ASSIGN_PATTERNS = [
        # "张三负责整理报告"
        re.compile(r'([\u4e00-\u9fff]{2,4})\s*(?:负责|来做|去做|来跟进|跟进|去跟进|主导|牵头|来做一下|去做一下)'),
        # "让张三来处理" / "请李四跟进"
        re.compile(r'(?:让|请|叫|让|要)\s*([\u4e00-\u9fff]{2,4})\s*(?:来|去)?\s*(?:处理|跟进|负责|做|完成|整理|优化|测试|准备)'),
        # "这个交给张三" / "任务分配给李四"
        re.compile(r'(?:交给|分配给|由|归)\s*([\u4e00-\u9fff]{2,4})\s*(?:来|去)?(?:做|处理|跟进|负责|完成)'),
        # "张三你来弄" / "李四你去搞"
        re.compile(r'([\u4e00-\u9fff]{2,4})\s*(?:你)?(?:来|去)\s*(?:弄|搞|做|弄一下|搞一下|处理|负责)'),
        # English: "John will handle" / "assigned to Mary"
        re.compile(r'(\b[A-Z][a-z]+)\s+(?:will|should|needs to|is going to)\s+(\w+)', re.IGNORECASE),
    ]

    # ── 截止时间模式 ────────────────────────────────────────────
    DEADLINE_PATTERNS = [
        # "周五前" / "下周三前" / "月底前"
        re.compile(r'((?:下?周|本周|这周)?[一二三四五六日天末]\w*前|月底前?|年底前?|本周内|下周内|这个月|下个月|今天|明天|后天|下周一|下周二|下周三|下周四|下周五|这周五|这周末)'),
        # "X月X号/日前" / "X月X日完成"
        re.compile(r'(\d{1,2}月\d{1,2}[号日](?:前)?(?:完成|交付)?)'),
        # "三天内" / "一周内" / "两周内"
        re.compile(r'(\d?[一二三四五六七八九十两]+\s*(?:天|周|个月)内(?:完成)?)'),
        # "before Friday" / "by next Monday" / "end of this week"
        re.compile(r'((?:by|before|until|due)\s+\w+|end\s+of\s+\w+)', re.IGNORECASE),
        # "下次会议" / "下次同步"
        re.compile(r'(下次(?:会议|同步|讨论|评审)|今日内|明日)'),
    ]

    # ── 优先级关键词 ────────────────────────────────────────────
    PRIORITY_HIGH = ['紧急', '马上', '立即', '尽快', '今天必须', '优先', '加急', 'urgent', 'asap', 'critical', 'high priority']
    PRIORITY_LOW = ['有空', '不急', '方便时', '之后', '回头', '有空再看', 'low priority', 'when free']

    # ── 行动触发词 ──────────────────────────────────────────────
    ACTION_TRIGGERS = [
        '负责', '来做', '去做', '跟进', '处理', '完成', '整理', '优化', '测试',
        '准备', '安排', '确认', '检查', '修复', '更新', '提交', '发送', '部署',
        '调研', '分析', '编写', '修改', '评审', '同步', '推进', '落实',
    ]

    # ── 任务描述模式 ────────────────────────────────────────────
    # 匹配 "负责XXX" / "来做XXX" 后面的任务内容
    TASK_PATTERNS = [
        # "负责整理测试报告" → task="整理测试报告"
        re.compile(r'(?:负责|来|去|要)?\s*(整理|优化|测试|准备|安排|确认|检查|修复|更新|提交|发送|部署|调研|分析|编写|修改|评审|同步|推进|落实|跟进|处理|完成)([\u4e00-\u9fff\w\s，、的]+?)(?:[，。；,!？?]|$)'),
        # "需要做XXX" / "要做XXX"
        re.compile(r'(?:需要|要|得)\s*(做|去弄|处理|整理|完成|跟进|准备|检查|提交|发送)([\u4e00-\u9fff\w\s，、的]+?)(?:[，。；,!？?]|$)'),
    ]

    def __init__(self, llm_client=None):
        if llm_client is None:
            try:
                from .llm_client import get_llm_client
                llm_client = get_llm_client()
            except ImportError:
                llm_client = None
        self.llm = llm_client

    def extract(self, segments: List[dict]) -> List[dict]:
        """
        从会议 segments 中提取行动项。

        Args:
            segments: [{"speaker": "张三", "text": "...", "start": 0.0, "end": 5.0}, ...]

        Returns:
            List of action item dicts:
            [{
                "task": "整理测试报告",
                "assignee": "张三",
                "deadline": "周五前",
                "priority": "high",
                "source_text": "张三负责整理测试报告，周五前完成",
                "speaker": "李四",
                "segment_index": 5,
            }, ...]
        """
        if not segments:
            return []

        # Try LLM first
        if self.llm and self.llm.is_available:
            try:
                return self._extract_with_llm(segments)
            except Exception as e:
                logger.debug(f"LLM action extraction failed, using rules: {e}")

        # Rule-based extraction
        return self._extract_with_rules(segments)

    def _extract_with_rules(self, segments: List[dict]) -> List[dict]:
        """规则引擎：模式匹配提取行动项。"""
        actions = []
        seen_tasks = set()  # 去重

        for idx, seg in enumerate(segments):
            text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
            speaker = seg.get("speaker", "") if isinstance(seg, dict) else getattr(seg, "speaker", "")

            if not text or len(text) < 8:
                continue

            # Check if this segment contains action triggers
            has_trigger = any(trigger in text for trigger in self.ACTION_TRIGGERS)
            if not has_trigger:
                continue

            # Extract assignee
            assignee = ""
            for pattern in self.ASSIGN_PATTERNS:
                m = pattern.search(text)
                if m:
                    assignee = m.group(1).strip()
                    break

            # Extract task
            task = ""
            for pattern in self.TASK_PATTERNS:
                m = pattern.search(text)
                if m:
                    # Combine verb + object
                    verb = m.group(1)
                    obj = m.group(2).strip().rstrip('的，。；') if m.lastindex >= 2 else ""
                    task = f"{verb}{obj}" if obj else verb
                    break

            # If no task pattern matched but has trigger, use surrounding text
            if not task:
                for trigger in self.ACTION_TRIGGERS:
                    pos = text.find(trigger)
                    if pos >= 0:
                        # Take trigger + next 10 chars as task
                        end = min(pos + len(trigger) + 15, len(text))
                        # Stop at punctuation
                        for p in ['，', '。', '；', ',', '!', '？', '?']:
                            p_pos = text.find(p, pos + len(trigger))
                            if p_pos >= 0:
                                end = min(end, p_pos)
                        task = text[pos:end].strip()
                        break

            if not task or len(task) < 2:
                continue

            # Deduplicate
            task_key = task[:10]
            if task_key in seen_tasks:
                continue
            seen_tasks.add(task_key)

            # Extract deadline
            deadline = ""
            for pattern in self.DEADLINE_PATTERNS:
                m = pattern.search(text)
                if m:
                    deadline = m.group(1).strip()
                    break

            # Determine priority
            text_lower = text.lower()
            priority = "medium"
            if any(kw in text or kw.lower() in text_lower for kw in self.PRIORITY_HIGH):
                priority = "high"
            elif any(kw in text or kw.lower() in text_lower for kw in self.PRIORITY_LOW):
                priority = "low"

            # If has deadline but no priority indicator, default to medium-high
            if deadline and priority == "medium":
                priority = "medium"

            # Extract source text (the sentence containing the action)
            source_text = self._extract_source_sentence(text, task)

            action = ActionItem(
                task=task,
                assignee=assignee,
                deadline=deadline,
                priority=priority,
                source_text=source_text,
                speaker=speaker,
                segment_index=idx,
            )
            actions.append(action.to_dict())

        logger.info(f"ActionExtractor (rules): extracted {len(actions)} action items from {len(segments)} segments")
        return actions

    def _extract_source_sentence(self, full_text: str, task: str) -> str:
        """Extract the sentence containing the action item."""
        # Split by sentence-ending punctuation
        sentences = re.split(r'[。！？!?\n]', full_text)
        for sent in sentences:
            if task[:5] in sent or any(t in sent for t in self.ACTION_TRIGGERS if t in full_text):
                return sent.strip()
        # Fallback: return first 80 chars
        return full_text[:80].strip()

    def _extract_with_llm(self, segments: List[dict]) -> List[dict]:
        """用 LLM 提取行动项（更准确）。"""
        # Build transcript text
        transcript_parts = []
        for idx, seg in enumerate(segments):
            text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
            speaker = seg.get("speaker", f"说话人{idx+1}") if isinstance(seg, dict) else getattr(seg, "speaker", f"说话人{idx+1}")
            transcript_parts.append(f"[{speaker}]: {text}")

        transcript = "\n".join(transcript_parts)

        # Truncate if too long
        if len(transcript) > 8000:
            transcript = transcript[:8000] + "...(truncated)"

        prompt = f"""请从以下会议转写中提取所有行动项（TODO/Action Items）。

会议内容：
{transcript}

对于每个行动项，提取以下信息：
1. task: 具体任务描述（动词+宾语，简洁明了）
2. assignee: 负责人（如果文本中明确提到）
3. deadline: 截止时间（如果文本中提到）
4. priority: 优先级 (high/medium/low)
5. source_text: 包含该行动项的原文片段（完整一句话）
6. speaker: 提出该行动项的发言人

请以 JSON 数组格式返回，每个元素格式如下：
{{
  "task": "任务描述",
  "assignee": "负责人",
  "deadline": "截止时间",
  "priority": "high|medium|low",
  "source_text": "原文片段",
  "speaker": "发言人"
}}

如果某项信息未在文本中明确提及，请留空字符串。
只返回 JSON 数组，不要其他文字。"""

        result = self.llm.chat_json(prompt, system="你是会议分析和行动项提取专家。", temperature=0.1)

        if not result or not isinstance(result, list):
            # LLM might return {"action_items": [...]} format
            if isinstance(result, dict) and "action_items" in result:
                result = result["action_items"]
            else:
                logger.warning("LLM action extraction returned unexpected format, falling back to rules")
                return self._extract_with_rules(segments)

        # Validate and clean
        actions = []
        for item in result:
            if not isinstance(item, dict):
                continue
            task = item.get("task", "").strip()
            if not task or len(task) < 2:
                continue
            actions.append({
                "task": task,
                "assignee": item.get("assignee", "").strip(),
                "deadline": item.get("deadline", "").strip(),
                "priority": item.get("priority", "medium").strip().lower(),
                "source_text": item.get("source_text", "").strip(),
                "speaker": item.get("speaker", "").strip(),
                "segment_index": -1,
            })

        logger.info(f"ActionExtractor (LLM): extracted {len(actions)} action items")
        return actions


# ============================================================
# Convenience function
# ============================================================

def extract_action_items(segments: List[dict]) -> List[dict]:
    """
    便捷函数：从 segments 提取行动项。

    Args:
        segments: [{"speaker": "张三", "text": "...", "start": 0.0, "end": 5.0}, ...]

    Returns:
        List of action item dicts
    """
    extractor = ActionExtractor()
    return extractor.extract(segments)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    test_segments = [
        {"speaker": "主持人", "text": "好，我们今天讨论了产品优化方案，接下来安排一下后续工作。", "start": 0.0, "end": 5.0},
        {"speaker": "张三", "text": "张三负责整理测试报告，周五前完成，这个比较紧急。", "start": 5.0, "end": 10.0},
        {"speaker": "李四", "text": "李四来跟进VAD参数优化，下周三同步结果给大家。", "start": 10.0, "end": 15.0},
        {"speaker": "王五", "text": "那个部署的事情，让王五去处理一下，不急，有空弄就行。", "start": 15.0, "end": 20.0},
        {"speaker": "张三", "text": "还需要准备下个版本的PRD文档，月底前提交。", "start": 20.0, "end": 25.0},
        {"speaker": "李四", "text": "对，PRD文档交给李四来写，我会在这周内完成。", "start": 25.0, "end": 30.0},
        {"speaker": "主持人", "text": "好的，那我们下次会议同步进展。", "start": 30.0, "end": 33.0},
    ]

    print("=" * 70)
    print("Action Item Extractor Test")
    print("=" * 70)

    extractor = ActionExtractor()
    actions = extractor.extract(test_segments)

    print(f"\n提取到 {len(actions)} 个行动项:\n")
    for i, a in enumerate(actions):
        print(f"  行动项 {i+1}:")
        print(f"    任务: {a['task']}")
        print(f"    负责人: {a['assignee'] or '(未指定)'}")
        print(f"    截止时间: {a['deadline'] or '(未指定)'}")
        print(f"    优先级: {a['priority']}")
        print(f"    原文: {a['source_text'][:60]}...")
        print(f"    发言人: {a['speaker']}")
        print()
