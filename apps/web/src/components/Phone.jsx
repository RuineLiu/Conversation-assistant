import { useState } from "react";
import { app, glass } from "../theme.js";
import AssistDetail from "./AssistDetail.jsx";
import ControlSheet from "./ControlSheet.jsx";
import PhoneRecords from "./PhoneRecords.jsx";

// RayNeo app 外壳：首页宫格 → 对话助手详情页 → 上滑控制面板
export default function Phone({ on, listening, interim, lang, onLang, onStart, onStop, records, onFeedback }) {
  const [view, setView] = useState("home"); // home | detail | records
  const [sheetOpen, setSheetOpen] = useState(false);
  const [direction, setDirection] = useState("wide");

  return (
    <div style={{ position: "relative", width: 340, background: app.bg, color: app.text, borderRadius: 28, border: `1px solid ${app.line}`, padding: 18, height: 620, boxShadow: "0 8px 30px rgba(0,0,0,0.08)", overflow: "hidden", display: "flex", flexDirection: "column" }}>
      {/* 顶栏 */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, fontSize: 14 }}>
        <span style={{ color: app.sub }}>RayNeo ›</span>
        <span style={{ width: 30, height: 30, borderRadius: "50%", border: `1px solid ${app.line}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: app.sub }}>头像</span>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
        {view === "home" && (
          <Home on={on} onOpenAssist={() => setView("detail")} />
        )}
        {view === "detail" && (
          <AssistDetail
            onBack={() => setView("home")}
            onRecords={() => setView("records")}
            listening={listening} interim={interim} lang={lang} direction={direction}
            onOpenSheet={() => setSheetOpen(true)}
          />
        )}
        {view === "records" && (
          <Sub title="使用记录" onBack={() => setView("detail")}>
            <PhoneRecords records={records} onFeedback={onFeedback} />
          </Sub>
        )}
      </div>

      {/* 上滑控制面板（覆盖在手机内） */}
      <ControlSheet
        open={sheetOpen} onClose={() => setSheetOpen(false)}
        lang={lang} onLang={onLang} direction={direction} onDirection={setDirection}
        listening={listening}
        onStart={() => { onStart(); setSheetOpen(false); }}
        onStop={() => { onStop(); setSheetOpen(false); }}
      />
    </div>
  );
}

// 首页：状态卡 + 功能宫格
function Home({ on, onOpenAssist }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ border: `1px solid ${app.line}`, borderRadius: 14, padding: "18px 16px", textAlign: "center" }}>
        <div style={{ fontSize: 13, color: on ? app.green : app.sub }}>{on ? "● 对话助手运行中" : "对话助手 Demo"}</div>
        <div style={{ fontSize: 12, color: app.sub, marginTop: 4 }}>点「对话助手」进入</div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <Tile label="通知" disabled />
        <Tile label="录音" disabled big />
        <Tile label="看板" disabled />
        <Tile label="对话助手" active={on} onClick={onOpenAssist} badge />
      </div>
    </div>
  );
}

function Tile({ label, disabled, active, onClick, big, badge }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{ gridRow: big ? "span 2" : undefined, minHeight: big ? 132 : 60, border: "none", borderRadius: 12, cursor: disabled ? "default" : "pointer", background: "#e9ebed", color: disabled ? "#9aa0a6" : app.text, fontSize: 14, display: "flex", alignItems: "center", justifyContent: "center", gap: 6, opacity: disabled ? 0.7 : 1 }}>
      {label}
      {badge && (
        <span style={{ width: 18, height: 18, borderRadius: "50%", border: `2px solid ${active ? glass.green : "#fff"}`, color: active ? glass.green : "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, background: active ? "rgba(93,255,122,0.12)" : "transparent" }}>?</span>
      )}
    </button>
  );
}

// 子页通用外壳（返回 + 标题）
function Sub({ title, onBack, children }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <button onClick={onBack} style={{ border: "none", background: "none", fontSize: 18, cursor: "pointer", color: app.text }}>‹</button>
        <b>{title}</b>
      </div>
      {children}
    </div>
  );
}
