import { glass } from "../theme.js";
export default function HudToast({ on }) {
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "8px 14px", borderRadius: 8, background: "rgba(93,255,122,0.12)", border: `1px solid ${glass.greenDim}`, color: glass.green, fontFamily: glass.mono, fontSize: 13 }}>
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: glass.green, boxShadow: `0 0 8px ${glass.greenGlow}` }} />
      {on ? "已开启 对话助手" : "已关闭 对话助手"}
    </div>
  );
}
