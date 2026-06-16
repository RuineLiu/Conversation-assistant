// AudioWorklet:在独立音频线程采集 PCM,降采样到 16kHz/16-bit,按 ~100ms 一包 postMessage。
// 用它替代已废弃的 ScriptProcessorNode —— 后者跑在主线程,会被 React 重渲染/DevTools 抢占
// 而漏采音频,导致喂给 ASR 的音频比实时慢、延迟滚雪球。Worklet 不受主线程卡顿影响。
class PCM16kProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._ratio = sampleRate / 16000; // sampleRate 是本上下文真实采样率(全局变量)
    this._acc = new Int16Array(1600); // ~100ms @16k
    this._n = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const ch = input[0]; // Float32Array(128)
      const ratio = this._ratio;
      const count = Math.floor(ch.length / ratio);
      for (let i = 0; i < count; i++) {
        let s = ch[Math.floor(i * ratio)];
        s = s < -1 ? -1 : s > 1 ? 1 : s;
        this._acc[this._n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
        if (this._n === this._acc.length) {
          this.port.postMessage(this._acc.buffer.slice(0));
          this._n = 0;
        }
      }
    }
    return true; // keep processor alive
  }
}

registerProcessor("pcm16k-processor", PCM16kProcessor);
