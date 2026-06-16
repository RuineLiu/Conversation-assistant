import { useRef, useState } from "react";
import { useAliyunAsr } from "./asr/useAliyunAsr.js";
import { createSession, postTranscript, postFeedback } from "./api.js";
import Phone from "./components/Phone.jsx";
import Glasses from "./components/Glasses.jsx";

// duration_policy(auto/3s/5s/8s) -> 倒计时秒数；auto 按字数估
function durationToSec(dp, text) {
  if (dp === "3s") return 3;
  if (dp === "5s") return 5;
  if (dp === "8s") return 8;
  const n = [...(text || "")].length;
  return n <= 30 ? 6 : n <= 45 ? 10 : 15;
}

export default function App() {
  const [on, setOn] = useState(false);
  const [lang, setLang] = useState("zh-CN");
  const [card, setCard] = useState(null);
  const [thinking, setThinking] = useState(false);
  const [records, setRecords] = useState([]);
  const sessionId = useRef(null);
  const busy = useRef(false);

  // ASR 出一句 final -> 喂给后端 -> 取镜端提示 -> 弹卡
  const handleFinal = async (text) => {
    if (!sessionId.current || busy.current) return;
    busy.current = true;
    const t0 = Date.now();
    setCard(null);       // 清掉上一条的残留卡
    setThinking(true);   // 进入「思考中…」加载态（后端约十几秒）
    try {
      const { segmentId, data } = await postTranscript(sessionId.current, text);
      // 只显示「针对当前这句」的镜端卡，避免上一句残留的缺口卡被当成对当前输入的回答。
      // 会议缺口卡的 source_refs 指向本 segment（transcript:<id>）；常识问答 / 记忆类答案
      // 的来源是 general_knowledge / memory:* 而非转写段 —— 这类没有引用「别的」转写段，
      // 同样属于针对本句的回答，应当显示。只过滤掉明确引用了其它转写段的陈旧卡。
      const ref = `transcript:${segmentId}`;
      const glasses = (data.prompts || []).find((p) => {
        if (!p.should_display || !String(p.prd_surface || "").startsWith("glasses")) return false;
        const otherTranscriptRef = (p.source_refs || []).some(
          (r) => String(r).startsWith("transcript:") && r !== ref
        );
        return !otherTranscriptRef;
      });
      if (glasses) {
        const rec = {
          id: glasses.decision_id,
          ts: Date.now(),
          scene: glasses.prompt_category,
          question: glasses.glasses_title,
          answer: glasses.glasses_text,
          countdown: durationToSec(glasses.duration_policy, glasses.glasses_text),
          latencyMs: Date.now() - t0,
          feedback: null,
        };
        setCard({ ...rec, streaming: false });
        setRecords((r) => [rec, ...r]);
      }
    } catch (e) {
      console.error(e);
    } finally {
      busy.current = false;
      setThinking(false);
    }
  };

  // 从一组 prompts 里挑出可显示的镜端卡，做成 record;没有则 null。
  const glassesRecFrom = (step) => {
    const prompts = step?.prompts || [];
    const g = prompts.find(
      (p) => p.should_display && String(p.prd_surface || "").startsWith("glasses")
    );
    if (!g) return null;
    return {
      id: g.decision_id,
      ts: Date.now(),
      scene: g.prompt_category,
      question: g.glasses_title,
      answer: g.glasses_text,
      countdown: durationToSec(g.duration_policy, g.glasses_text),
      latencyMs: null,
      feedback: null,
    };
  };

  // partial 提前触发的"快卡"(prompt_preview):立刻弹卡(快),但不记入记录，
  // 等 final 再正式记录(final 会 reconcile，按 id 去重，避免重复)。
  const handleWsPreview = (text, step) => {
    console.debug("[asr] preview 产出:", (step?.prompts || []).map((p) => ({ display: p.should_display, surface: p.prd_surface, title: p.glasses_title })));
    const rec = glassesRecFrom(step);
    if (rec) setCard({ ...rec, streaming: false, preview: true });
  };

  // final 事件:正式结果，弹卡 + 记入记录(按 id 去重)。
  const handleWsFinal = (text, step) => {
    console.debug("[asr] final 产出:", (step?.prompts || []).map((p) => ({ display: p.should_display, surface: p.prd_surface, title: p.glasses_title })));
    const rec = glassesRecFrom(step);
    if (!rec) {
      console.debug("[asr] 未弹镜端卡 —— 没有 (should_display && glasses_*) 的 prompt(可能被去重/抑制)。");
      return;
    }
    setCard({ ...rec, streaming: false });
    setRecords((r) => (r.some((x) => x.id === rec.id) ? r : [rec, ...r]));
  };

  const asr = useAliyunAsr({
    language: lang,
    speaker: "我",
    onPreview: handleWsPreview,
    onFinalStep: handleWsFinal,
    onError: (e) => { console.error(e); alert(`ASR 出错：${e.message || e}`); setOn(false); },
  });

  const onStart = async () => {
    try {
      // 每次开聆听都开「新会话」：后端同会话里 segment 计数会从 stream_0001 重来，
      // 复用旧会话会让 decision_id 撞上一轮 → 后端 add_decision 崩(prompts=-1)。
      sessionId.current = await createSession();
      setOn(true);
      await asr.start(sessionId.current); // 阿里云 WS 需要先有 session
    } catch (e) {
      console.error("启动失败：", e);
      setOn(false);
      alert("启动失败：确认 product 后端在 :8001、已授权麦克风、且用 Chrome");
    }
  };
  const onStop = () => { setOn(false); asr.stop(); };

  const onFeedback = (id, fb) => {
    const signal = fb === "up" ? "accept" : fb === "down" ? "dismiss" : "ignore";
    postFeedback(id, signal).catch((e) => console.error(e));
    setRecords((r) => r.map((x) => (x.id === id ? { ...x, feedback: fb } : x)));
    if (card?.id === id) setCard((c) => ({ ...c, fb }));
  };

  // 文字输入（测试用）：自动建会话再喂转写，绕过麦克风
  const [draft, setDraft] = useState("");
  const submitText = async (text) => {
    const t = text.trim();
    if (!t) return;
    try {
      // 测试用：每句开全新会话，每句独立、互不干扰
      // （也绕开后端多轮里 decision_id 重复导致的 500 —— 那是后端 bug，已记录待搭子修）
      sessionId.current = await createSession();
      setOn(true);
      handleFinal(t);
      setDraft("");
    } catch (e) {
      console.error(e);
      alert("连接后端失败，确认 product 后端已在 :8001 启动");
    }
  };

  return (
    <div style={{ maxWidth: 1080, margin: "0 auto", padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h2 style={{ margin: "0 0 16px" }}>金星 · 对话助手 Web（接 product 后端）</h2>
      {!asr.supported && <div style={{ color: "#d94040", marginBottom: 12, fontSize: 14 }}>当前浏览器不支持语音识别，请用 Chrome。</div>}
      <form onSubmit={(e) => { e.preventDefault(); submitText(draft); }} style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input value={draft} onChange={(e) => setDraft(e.target.value)}
          placeholder="测试输入：打一句会议对话回车，如「这个问题谁负责，下周五deadline前能不能定？」"
          style={{ flex: 1, padding: "10px 14px", border: "1px solid #ececec", borderRadius: 10, fontSize: 14, outline: "none" }} />
        <button type="submit" style={{ border: "none", borderRadius: 10, padding: "0 18px", cursor: "pointer", background: "#000", color: "#fff", fontSize: 14 }}>发送</button>
      </form>
      <div style={{ display: "flex", flexDirection: "column", gap: 20, alignItems: "center" }}>
        <Glasses on={on} card={card} thinking={thinking} interim={asr.interim} onClose={() => setCard(null)} />
        <Phone on={on} listening={asr.listening} interim={asr.interim} lang={lang} onLang={setLang} onStart={onStart} onStop={onStop} records={records} onFeedback={onFeedback} />
      </div>
    </div>
  );
}
