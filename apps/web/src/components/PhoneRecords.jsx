import { useState } from "react";
import { app } from "../theme.js";
export default function PhoneRecords({ records, onFeedback }) {
  const [openId, setOpenId] = useState(null);
  if (!records.length) return <div style={{ color: app.sub, textAlign: "center", padding: 40 }}>未产生提示记录</div>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {records.map((r) => (
        <div key={r.id} style={{ border: `1px solid ${app.line}`, borderRadius: 10, padding: 12 }}>
          <div style={{ fontSize: 12, color: app.sub }}>{r.scene} · {new Date(r.ts).toLocaleTimeString()}</div>
          <div style={{ fontSize: 13, color: app.sub, marginTop: 4 }}>❓ {r.question}</div>
          <div style={{ fontSize: 15, marginTop: 2 }}>{r.answer}</div>
          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
            <button onClick={() => onFeedback(r.id, r.feedback === "up" ? null : "up")} style={fb(r.feedback === "up", app.green)}>👍</button>
            <button onClick={() => { onFeedback(r.id, "down"); setOpenId(r.id); }} style={fb(r.feedback === "down", app.red)}>👎</button>
          </div>
          {openId === r.id && r.feedback === "down" && (
            <div style={{ marginTop: 8, fontSize: 12, color: app.sub }}>已收集反馈 · 感谢反馈</div>
          )}
        </div>
      ))}
    </div>
  );
}
const fb = (active, color) => ({ background: active ? color : "#fff", color: active ? "#fff" : color, border: `1px solid ${color}`, borderRadius: 6, cursor: "pointer", padding: "2px 10px" });
