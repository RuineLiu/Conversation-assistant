import { app } from "../theme.js";

// 底部上滑控制面板：收听语言 + 收听方向 + 启动按钮
const LANGS = [
  ["zh-CN", "简体中文"], ["en-US", "English"],
  // PRD 实际支持 10 种（英/法/德/西/意/简中/繁粤/日/韩/波兰），demo 仅中英可真识别
];
const DIRECTIONS = [["voice", "人声凸显"], ["wide", "宽谱收音（默认）"]];

export default function ControlSheet({ open, onClose, lang, onLang, direction, onDirection, listening, onStart, onStop }) {
  return (
    <>
      {/* 遮罩 */}
      <div onClick={onClose} style={{ position: "absolute", inset: 0, background: open ? "rgba(0,0,0,0.35)" : "transparent", pointerEvents: open ? "auto" : "none", transition: "background .2s", borderRadius: 24, zIndex: 5 }} />
      {/* 上滑面板 */}
      <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, transform: open ? "translateY(0)" : "translateY(110%)", transition: "transform .25s cubic-bezier(.3,.8,.4,1)", background: "#3a3f45", color: "#fff", borderTopLeftRadius: 20, borderTopRightRadius: 20, padding: 18, zIndex: 6 }}>
        <div onClick={onClose} style={{ width: 44, height: 4, borderRadius: 2, background: "rgba(255,255,255,0.4)", margin: "0 auto 16px", cursor: "pointer" }} />

        <div style={{ fontSize: 13, opacity: 0.7, marginBottom: 8 }}>收听语言（默认 App 语言 · 实际支持 10 种）</div>
        <div style={{ display: "flex", gap: 8, marginBottom: 18 }}>
          {LANGS.map(([k, label]) => (
            <button key={k} onClick={() => onLang(k)} style={chip(lang === k)}>{label}</button>
          ))}
        </div>

        <div style={{ fontSize: 13, opacity: 0.7, marginBottom: 8 }}>收听方向</div>
        <div style={{ display: "flex", gap: 8, marginBottom: 22 }}>
          {DIRECTIONS.map(([k, label]) => (
            <button key={k} onClick={() => onDirection(k)} style={chip(direction === k)}>{label}</button>
          ))}
        </div>

        <button onClick={listening ? onStop : onStart} style={{ width: "100%", padding: "12px 0", borderRadius: 24, border: "none", cursor: "pointer", fontSize: 15, color: "#fff", background: listening ? app.red : app.green }}>
          {listening ? "停止" : "启动"}
        </button>
      </div>
    </>
  );
}

const chip = (active) => ({ flex: 1, padding: "8px 0", borderRadius: 8, border: "none", cursor: "pointer", fontSize: 13, background: active ? "#fff" : "rgba(255,255,255,0.12)", color: active ? "#1a1a1a" : "#fff" });
