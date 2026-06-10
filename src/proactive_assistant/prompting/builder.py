import json

from proactive_assistant.prompting.contracts import PromptGenerationRequest
from proactive_assistant.prompting.enforcer import GlassesLengthLimits


SYSTEM_INSTRUCTIONS = """You generate PRD-fit proactive assistant prompts for smart glasses and the companion app.

You must decide whether a prompt should be shown and return only schema-valid JSON.
Optimize jointly for prompt timing and content granularity. Keep glasses text concise.
Respect privacy constraints. Prefer no prompt when help value is low, redundancy is high, or privacy risk is high.
Do not invent facts that are not present in transcript, memory, persona, or provided context.
For smart-glasses popup cards, HARD LIMITS apply. Never exceed the provided prd_glasses_card limits in glasses_title or glasses_text.
If the useful answer cannot fit inside those limits, return content_granularity=1 and put the full answer in app_detail_text.
Do not squeeze long explanations into glasses_text. Use glasses_text only for the shortest useful answer.
For person/fact recall, use memory_context when available and cite source_refs; otherwise say that verification is needed.
For unfamiliar terms or concepts, give a one-sentence explanation first, then optional app detail.
""".strip()


def build_prompt_generation_input(request: PromptGenerationRequest) -> str:
    """Build a compact JSON prompt payload for structured model generation."""

    glasses_limits = _glasses_card_limits()
    payload = {
        "session": {
            "session_id": request.session_id,
            "scenario_id": request.scenario_id,
            "locale": request.locale,
            "prd_surface": request.prd_surface,
            "display_mode": request.display_mode,
            "duration_policy": request.duration_policy,
        },
        "target": {
            "prompt_category_candidate": request.prompt_category_candidate,
            "target_content_granularity": request.target_content_granularity,
        },
        "persona_context": request.persona_context,
        "session_context": request.session_context,
        "memory_context": request.memory_context,
        "privacy_constraints": request.privacy_constraints,
        "recent_transcript": [item.model_dump(mode="json") for item in request.transcript_window],
        "output_rules": {
            "content_granularity_range": "0-4",
            "glasses_should_prefer": "levels 1-3 unless user manually opens detail",
            "app_detail_can_use": "level 4 when appropriate",
            "source_refs_required_when_prompting": True,
            "prd_glasses_card": {
                "hard_limits": glasses_limits,
                "title_field": "glasses_title",
                "answer_field": "glasses_text",
                "interaction": "user can confirm answer as useful or mark it not useful",
                "duration_policy": request.duration_policy,
                "overflow_policy": (
                    "If glasses_title or glasses_text cannot fit the hard limits, "
                    "return content_granularity=1 and move the full answer to app_detail_text."
                ),
                "detail_destination": "app_detail_text",
            },
            "focus_tasks": [
                "capture explicit or implicit questions",
                "explain unfamiliar terms and concepts",
                "recall long-term memory when relevant",
                "surface memory update-worthy facts only when supported by transcript",
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _glasses_card_limits(limits: GlassesLengthLimits | None = None) -> dict[str, object]:
    resolved = limits or GlassesLengthLimits()
    return {
        "zh": {
            "length_mode": "cjk_chars",
            "glasses_title_max": resolved.title_cjk_chars,
            "glasses_text_max": resolved.text_cjk_chars,
            "unit": "visible CJK/ASCII characters excluding whitespace and punctuation",
        },
        "en": {
            "length_mode": "en_words",
            "glasses_title_max": resolved.title_en_words,
            "glasses_text_max": resolved.text_en_words,
            "unit": "words",
        },
        "mixed_text_policy": (
            "prefer CJK counting when text contains CJK characters, matching PromptResultEnforcer"
            if resolved.prefer_cjk_when_mixed
            else "use locale-based counting"
        ),
        "hard_limit": True,
    }
