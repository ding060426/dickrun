"""JSON Schema definitions for LLM-generated meeting summaries.

LLM returns structured JSON; markdown_renderer renders it to Markdown.
This keeps the LLM output constrained and the .md format stable.
"""

# ── Single-meeting summary schema ───────────────────────────────

SINGLE_SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["overview", "topics", "decisions", "action_items", "risks", "open_questions"],
    "properties": {
        "overview": {
            "type": "string",
            "description": "2-4 sentence executive summary of the meeting",
        },
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["topic", "summary"],
                "properties": {
                    "topic": {"type": "string"},
                    "summary": {"type": "string"},
                    "speakers": {"type": "array", "items": {"type": "string"}},
                },
            },
            "maxItems": 10,
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["content"],
                "properties": {
                    "content": {"type": "string"},
                    "owner": {"type": "string", "default": ""},
                    "rationale": {"type": "string", "default": ""},
                },
            },
            "maxItems": 20,
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["task"],
                "properties": {
                    "task": {"type": "string"},
                    "assignee": {"type": "string", "default": ""},
                    "deadline": {"type": "string", "default": ""},
                    "priority": {"type": "string", "default": "medium"},
                    "status": {"type": "string", "default": "pending"},
                },
            },
            "maxItems": 20,
        },
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["description"],
                "properties": {
                    "description": {"type": "string"},
                    "impact": {"type": "string", "default": ""},
                    "mitigation": {"type": "string", "default": ""},
                },
            },
            "maxItems": 10,
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
        "speaker_contributions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["speaker"],
                "properties": {
                    "speaker": {"type": "string"},
                    "contribution": {"type": "string"},
                },
            },
            "maxItems": 15,
        },
    },
}


# ── Multi-meeting comprehensive summary schema ──────────────────

MULTI_SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["executive_summary", "meeting_summaries", "common_topics", "decision_changes",
                 "open_action_items", "resolved_items", "new_risks", "recommendations"],
    "properties": {
        "executive_summary": {
            "type": "string",
            "description": "2-5 sentence high-level summary across all meetings",
        },
        "meeting_summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["record_id", "title", "summary"],
                "properties": {
                    "record_id": {"type": "string"},
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "summary": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "decisions": {"type": "array", "items": {"type": "string"}},
                    "action_items": {"type": "array", "items": {"type": "string"}},
                },
            },
            "maxItems": 20,
        },
        "common_topics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["topic"],
                "properties": {
                    "topic": {"type": "string"},
                    "description": {"type": "string"},
                    "mentioned_in": {"type": "array", "items": {"type": "string"}},
                },
            },
            "maxItems": 10,
        },
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["date", "event"],
                "properties": {
                    "date": {"type": "string"},
                    "event": {"type": "string"},
                    "record_id": {"type": "string"},
                },
            },
            "maxItems": 30,
        },
        "decision_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["decision", "change"],
                "properties": {
                    "decision": {"type": "string"},
                    "change": {"type": "string"},
                    "original_record": {"type": "string"},
                    "latest_record": {"type": "string"},
                },
            },
            "maxItems": 15,
        },
        "progress_changes": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
        "open_action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["task"],
                "properties": {
                    "task": {"type": "string"},
                    "assignee": {"type": "string", "default": ""},
                    "first_raised": {"type": "string"},
                    "latest_status": {"type": "string"},
                },
            },
            "maxItems": 20,
        },
        "resolved_items": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 15,
        },
        "new_risks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["description"],
                "properties": {
                    "description": {"type": "string"},
                    "impact": {"type": "string", "default": ""},
                    "first_seen": {"type": "string"},
                },
            },
            "maxItems": 10,
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
    },
}
