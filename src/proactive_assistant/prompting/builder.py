import json

from proactive_assistant.prompting.contracts import PromptGenerationRequest


SYSTEM_INSTRUCTIONS = """You generate PRD-fit proactive assistant prompts for smart glasses and the companion app.

You must decide whether a prompt should be shown and return only schema-valid JSON.
Optimize jointly for prompt timing and content granularity. Keep glasses text concise.
Respect privacy constraints. Prefer no prompt when help value is low, redundancy is high, or privacy risk is high.
Do not invent facts that are not present in transcript, memory, persona, or provided context.
""".strip()


def build_prompt_generation_input(request: PromptGenerationRequest) -> str:
    """Build a compact JSON prompt payload for structured model generation."""

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
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
