import { useEffect, useState } from "react";
import { glass } from "../theme.js";

export default function HudCard({ card, onClose }) {
  const [left, setLeft] = useState(card.countdown || 5);
  useEffect(() => {
    setLeft(card.countdown || 5);
    if (card.streaming) return; // 流式输出答案时不倒计时，等答案输完(onDone)再开始
    const id = setInterval(() => setLeft((s) => (s <= 1 ? 0 : s - 1)), 1000);
    return () => clearInterval(id);
  }, [card.id, card.countdown, card.streaming]);

  // 倒计时归零 → 自动关闭（通知父组件清掉卡片）
  useEffect(() => {
    if (left <= 0 && !card.streaming) onClose?.();
  }, [left, card.streaming]);

  if (left <= 0 && !card.streaming) return null;
  return (
    <div style={{ border: `1px solid ${glass.greenDim}`, borderRadius: 12, padding: "18px 22px", background: "rgba(93,255,122,0.06)", color: glass.green, fontFamily: glass.mono, width: "88%", maxWidth: 560, boxShadow: `0 0 18px ${glass.greenGlow}33`, position: "relative" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, marginBottom: 10 }}>
        <div style={{ fontSize: 13, opacity: 0.75 }}>❓ {card.question}</div>
        <button
          onClick={onClose}
          aria-label="关闭"
          style={{ flexShrink: 0, width: 22, height: 22, borderRadius: "50%", border: `1px solid ${glass.greenDim}`, background: "transparent", color: glass.green, cursor: "pointer", fontSize: 12, lineHeight: 1, display: "flex", alignItems: "center", justifyContent: "center" }}
        >✕</button>
      </div>
      <div style={{ fontSize: 19, lineHeight: 1.55, minHeight: 26 }}>{card.answer}{card.streaming && <span style={{ opacity: 0.6 }}>▍</span>}</div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 10, fontSize: 12 }}>
        <span style={{ opacity: 0.6 }}>{card.scene} · {card.latencyMs ? `${(card.latencyMs / 1000).toFixed(1)}s` : "…"}</span>
        {!card.streaming && <span style={{ opacity: 0.5 }}>{left}s</span>}
      </div>
    </div>
  );
}
