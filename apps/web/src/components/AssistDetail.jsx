import { app } from "../theme.js";

// 对话助手详情页：待机「Ready to assist」/ 启动后「∞ 辅助中」+ 底部上滑控制条
export default function AssistDetail({ onBack, onRecords, listening, interim, lang, direction, onOpenSheet }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
        <button onClick={onBack} style={{ border: "none", background: "none", fontSize: 18, cursor: "pointer", color: app.text }}>‹</button>
        <b>对话助手</b>
        <button onClick={onRecords} style={{ border: "none", background: "none", fontSize: 12, cursor: "pointer", color: app.sub }}>记录</button>
      </div>

      {/* 主体：待机 / 辅助中 */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 14 }}>
        {listening ? (
          <>
            <div style={{ fontSize: 40, color: app.text, animation: "jx-pulse 1.6s ease-in-out infinite" }}>∞</div>
            <div style={{ color: app.text, fontSize: 15 }}>辅助中</div>
            <div style={{ color: app.sub, fontSize: 12, minHeight: 18, padding: "0 16px", textAlign: "center" }}>{interim || "聆听中，说话或放播客…"}</div>
          </>
        ) : (
          <>
            <div style={{ fontSize: 40, color: "#cfd3d6" }}>∞</div>
            <div style={{ color: app.sub, fontSize: 14 }}>待机中 · 启动后开始聆听</div>
          </>
        )}
      </div>

      {/* 底部收起态控制条：点开上滑面板 */}
      <div onClick={onOpenSheet} style={{ cursor: "pointer", background: "#f5f6f7", borderRadius: 16, padding: "8px 18px 12px" }}>
        <div style={{ width: 34, height: 4, borderRadius: 2, background: "#d4d7da", margin: "0 auto 10px" }} />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", alignItems: "center" }}>
          <span style={{ fontSize: 13, color: app.sub, textAlign: "left" }}>{direction === "voice" ? "人声凸显" : "宽谱收音"}</span>
          <span style={{ fontSize: 13, color: app.sub, textAlign: "center" }}>{lang === "zh-CN" ? "中文" : "English"}</span>
          <span style={{ fontSize: 14, fontWeight: 600, color: listening ? app.red : app.green, textAlign: "right" }}>{listening ? "停止" : "启动"}</span>
        </div>
      </div>
    </div>
  );
}
