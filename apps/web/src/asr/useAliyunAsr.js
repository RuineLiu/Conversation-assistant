import { useCallback, useRef, useState } from "react";

// 后端流式 ASR(阿里云):麦克风 -> 16kHz/16bit/单声道 PCM -> WS /sessions/{id}/asr/stream
//
// 与浏览器 Web Speech(useAsr)不同:这里把原始音频发给后端,后端用 .env 里配置的
// 阿里云 ASR 转写。后端在 final_transcript 时已经跑完产品流程,事件里直接带 prompts,
// 所以前端拿到 final 就能弹卡,无需再 POST /transcript。
//
// 重要:点「停止聆听」后,后端还需要几秒(阿里云出 final + 跑 Claude)才把结果发回。
// 所以 stop 时只关麦克风、发 {type:stop},但保持 WS 打开,等 session_stopped(或超时)再关,
// 否则会把还在路上的答案丢掉。
//
// 服务端事件:
//   {type:"stream_opened"} / {type:"session_started"}
//   {type:"partial_transcript", transcription:{text}}            // 实时,仅展示
//   {type:"final_transcript", transcription:{text}, transcript_step:{prompts:[...]}}
//   {type:"session_stopped"}

const GRACE_CLOSE_MS = 15000; // stop 后最多再等这么久收尾 final

function floatTo16BitPCM(input) {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

// 线性降采样到 16k(若 AudioContext 实际采样率不是 16000)
function downsampleTo16k(input, inRate) {
  if (inRate === 16000) return input;
  const ratio = inRate / 16000;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) out[i] = input[Math.floor(i * ratio)];
  return out;
}

export function useAliyunAsr({ language = "zh-CN", speaker = "我", onFinalStep, onPreview, onError }) {
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const wsRef = useRef(null);
  const ctxRef = useRef(null);
  const streamRef = useRef(null);
  const procRef = useRef(null);
  const wantRef = useRef(false);
  const graceTimerRef = useRef(null);

  // 只停麦克风/音频图,不动 WS
  const stopMic = useCallback(() => {
    try {
      if (procRef.current) {
        if (procRef.current.port) procRef.current.port.onmessage = null;
        procRef.current.disconnect();
      }
    } catch {}
    try { streamRef.current?.getTracks().forEach((t) => t.stop()); } catch {}
    try { ctxRef.current && ctxRef.current.state !== "closed" && ctxRef.current.close(); } catch {}
    procRef.current = null; streamRef.current = null; ctxRef.current = null;
  }, []);

  const closeWs = useCallback(() => {
    if (graceTimerRef.current) { clearTimeout(graceTimerRef.current); graceTimerRef.current = null; }
    const ws = wsRef.current;
    wsRef.current = null;
    try { ws && ws.readyState <= WebSocket.OPEN && ws.close(); } catch {}
  }, []);

  // 停止聆听:停麦克风 + 发 stop,但保持 WS,等 session_stopped(或超时)再关
  const stop = useCallback(() => {
    wantRef.current = false;
    stopMic();
    setListening(false);
    setInterim("");
    const ws = wsRef.current;
    try { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" })); } catch {}
    if (graceTimerRef.current) clearTimeout(graceTimerRef.current);
    graceTimerRef.current = setTimeout(() => { console.debug("[asr] grace timeout, closing ws"); closeWs(); }, GRACE_CLOSE_MS);
  }, [stopMic, closeWs]);

  // sessionId 必须先建好(WS 路径需要它)
  const start = useCallback(async (sessionId) => {
    if (!sessionId) throw new Error("missing sessionId for ASR stream");
    wantRef.current = true;

    // 1) 开 WS —— 同源(走前端域名 + Vite/反代的 ws 转发)。这样本地和隧道/线上都能连:
    //    本地 ws://localhost:5175 -> 代理到 8001;隧道 wss://xxx.trycloudflare.com -> 代理到 8001。
    //    可用 VITE_ASR_WS_HOST 覆盖(指向独立后端域名)。
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const wsHost = import.meta.env.VITE_ASR_WS_HOST || location.host;
    const url = `${proto}://${wsHost}/sessions/${encodeURIComponent(sessionId)}/asr/stream`
      + `?speaker=${encodeURIComponent(speaker)}&language=${encodeURIComponent(language)}&audio_format=pcm16k`;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(typeof ev.data === "string" ? ev.data : ""); } catch { return; }
      if (!msg || !msg.type) return;
      const t = msg.transcription?.text;
      if (msg.type !== "partial_transcript") console.debug("[asr]", msg.type, t ?? "");
      if (msg.type === "partial_transcript") {
        setInterim(t || "");
      } else if (msg.type === "prompt_preview") {
        // partial 提前触发的"快卡":一边说一边就出，不等句末
        console.debug("[asr] preview:", t, "| prompts:", msg.prompt_preview?.prompts?.length ?? 0);
        onPreview?.(t || "", msg.prompt_preview || null);
      } else if (msg.type === "final_transcript") {
        setInterim("");
        console.debug("[asr] final:", t, "| prompts:", msg.transcript_step?.prompts?.length ?? 0);
        onFinalStep?.(t || "", msg.transcript_step || null);
      } else if (msg.type === "session_stopped") {
        setListening(false);
        closeWs(); // 收尾完成,关连接
      } else if (msg.type === "error") {
        onError?.(new Error(msg.detail || msg.message || "ASR stream error"));
      }
    };
    ws.onerror = () => onError?.(new Error("WS 连接失败(确认后端 8001 + vite 代理 ws:true)"));
    ws.onclose = () => { stopMic(); setListening(false); };

    await new Promise((resolve, reject) => {
      ws.addEventListener("open", resolve, { once: true });
      ws.addEventListener("error", () => reject(new Error("WS open failed")), { once: true });
    });

    // 2) 麦克风 -> PCM,用 AudioWorklet 在独立音频线程采集(不被主线程 React 重渲染/DevTools
    //    饿死,避免音频喂入比实时慢、延迟滚雪球)。worklet 内部降采样到 16k 并按 ~100ms 一包回传。
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    ctxRef.current = ctx;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    streamRef.current = stream;
    await ctx.audioWorklet.addModule(new URL("./pcm16k-worklet.js", import.meta.url));
    const src = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "pcm16k-processor");
    procRef.current = node;
    node.port.onmessage = (e) => {
      if (!wantRef.current || ws.readyState !== WebSocket.OPEN) return;
      ws.send(e.data); // ArrayBuffer: Int16 PCM @16k
    };
    src.connect(node);
    node.connect(ctx.destination); // worklet 不输出音频,接 destination 仅为驱动图(静默)

    setListening(true);
    console.debug("[asr] listening started (AudioWorklet), session", sessionId);
  }, [language, speaker, onFinalStep, onPreview, onError, stopMic, closeWs]);

  const supported = !!(navigator.mediaDevices?.getUserMedia && window.WebSocket && (window.AudioContext || window.webkitAudioContext));
  return { listening, interim, supported, start, stop };
}
