import { glass, hudBackground } from "../theme.js";
import HudToast from "./HudToast.jsx";
import HudCard from "./HudCard.jsx";

// 镜端 HUD：横长竖窄的眼镜显示条
export default function Glasses({ on, card, thinking, interim, onClose }) {
  return (
    <div style={{ ...hudBackground, width: "100%", height: 240, borderRadius: 18, padding: "12px 24px", display: "flex", flexDirection: "column", gap: 8, border: `1px solid ${glass.greenDim}`, boxSizing: "border-box" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span style={{ color: glass.greenDim, fontFamily: glass.mono, fontSize: 12 }}>镜端 HUD</span>
        <HudToast on={on} />
      </div>
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", minHeight: 0 }}>
        {card ? (
          <HudCard card={card} onClose={onClose} />
        ) : thinking ? (
          <span style={{ color: glass.green, fontFamily: glass.mono, fontSize: 14 }}>💭 思考中…<span style={{ opacity: 0.6 }}>▍</span></span>
        ) : on && interim ? (
          <span style={{ color: glass.green, fontFamily: glass.mono, fontSize: 14, opacity: 0.85 }}>🎙 {interim}<span style={{ opacity: 0.6 }}>▍</span></span>
        ) : (
          <span style={{ color: glass.greenDim, fontFamily: glass.mono, fontSize: 13, opacity: 0.6 }}>{on ? "聆听中…" : "未开启"}</span>
        )}
      </div>
    </div>
  );
}
