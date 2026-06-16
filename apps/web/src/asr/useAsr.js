import { useEffect, useRef, useState, useCallback } from "react";

// 浏览器流式 ASR：interim 实时 / final 触发回调 / 60s 自动重连 / 语种切换
export function useAsr({ lang, onFinal }) {
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [supported, setSupported] = useState(true);
  const recRef = useRef(null);
  const wantRef = useRef(false);
  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

  useEffect(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { setSupported(false); return; }
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = lang;
    rec.onresult = (e) => {
      let itm = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) { const t = r[0].transcript.trim(); if (t) onFinalRef.current?.(t); }
        else itm += r[0].transcript;
      }
      setInterim(itm);
    };
    rec.onend = () => { setInterim(""); if (wantRef.current) { try { rec.start(); } catch {} } else setListening(false); };
    rec.onerror = (ev) => { if (ev.error === "not-allowed") { wantRef.current = false; setListening(false); } };
    recRef.current = rec;
    return () => { wantRef.current = false; try { rec.stop(); } catch {} };
  }, [lang]);

  const start = useCallback(() => { wantRef.current = true; try { recRef.current?.start(); setListening(true); } catch {} }, []);
  const stop = useCallback(() => { wantRef.current = false; try { recRef.current?.stop(); } catch {} setListening(false); }, []);
  return { listening, interim, supported, start, stop };
}
