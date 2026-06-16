// 对接 proactive_assistant 的 product 后端（FastAPI, 默认 :8001，经 Vite proxy）。
// 契约见 src/proactive_assistant/product/api.py + contracts.py。

// 建会话：POST /sessions -> { session: {session_id,...}, meeting_state }
export async function createSession(title = "Web Demo 会话") {
  const r = await fetch("/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: `web-${Date.now()}`,
      config: { title, pre_context: "", privacy_constraints: [], metadata: { participants: ["我", "对方"], scheduled_end_ms: 1800000 } },
      start: true,
    }),
  });
  if (!r.ok) throw new Error(`createSession ${r.status}`);
  const d = await r.json();
  return d.session.session_id;
}

// 喂一句转写：POST /sessions/{id}/transcript -> { prompts:[ProductPromptPayload], meeting_gaps, ... }
export async function postTranscript(sessionId, text, speaker = "对方") {
  const segmentId = `seg-${Date.now()}`;
  const r = await fetch(`/sessions/${sessionId}/transcript`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      segment_id: segmentId,
      segment: { speaker, start_ms: 0, end_ms: Math.max(1000, [...text].length * 120), text, asr_confidence: 0.9 },
    }),
  });
  if (!r.ok) throw new Error(`postTranscript ${r.status}`);
  return { segmentId, data: await r.json() };
}

// 反馈：POST /prompt-decisions/{decision_id}/feedback  signal_type ∈ accept/dismiss/...
export async function postFeedback(decisionId, signalType) {
  return fetch(`/prompt-decisions/${decisionId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ signal_type: signalType }),
  }).then((r) => r.json());
}

export const health = () => fetch("/health").then((r) => r.json());
