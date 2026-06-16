# Proactive Assistant · Web 前端

接 `product` 后端（FastAPI）的 React/Vite web 客户端，复用金星对话助手 demo 的双端 UI（手机端 + 镜端 HUD）。与 `apps/mobile`（Flutter）、`apps/sandbox-viewer`（Phaser）并列。

## 运行

```bash
# 1. 先起 product 后端（在仓库根目录）
PROACTIVE_PROMPT_MODE=rules uv run uvicorn proactive_assistant.product.api:app --port 8001
#   - rules 模式：确定性，无需 API key
#   - 用真实模型：设 OPENAI_* 环境变量 + PROACTIVE_PROMPT_MODE=llm_with_rule_fallback

# 2. 起 web 前端
cd apps/web
npm install
npm run dev          # http://localhost:5174  (Chrome)
```

Vite 把后端路由（`/sessions`、`/prompt-decisions`、`/health` 等）代理到 `:8001`，避免 CORS。后端地址可用 `PROACTIVE_API` 覆盖。

## 它怎么接后端（契约映射）

| 前端动作 | 后端调用 | 说明 |
|---|---|---|
| 手机端「启动」 | `POST /sessions` | 建会话（带 `scheduled_end_ms`，否则会议缺口检测不触发）|
| ASR 出一句 final | `POST /sessions/{id}/transcript` | 返回 `prompts: [ProductPromptPayload]` |
| 镜端弹卡 | 取 `should_display && prd_surface 以 "glasses" 开头` 的 prompt | 见下方字段映射 |
| 卡片/记录反馈 | `POST /prompt-decisions/{decision_id}/feedback` | 👍→`accept`，👎→`dismiss` |

`ProductPromptPayload` → 镜端卡片字段映射（见 `src/api.js` / `src/App.jsx`）：

```
glasses_title     -> 卡片问题行（❓）
glasses_text      -> 卡片答案
prompt_category   -> 场景标签
duration_policy   -> 倒计时（3s/5s/8s；auto 按字数估）
decision_id       -> 反馈用
```

## ASR

当前用**浏览器 Web Speech API**（`src/asr/useAsr.js`）转写 → POST 文本给 `/transcript`。
后端已有 **Azure / 阿里云流式 ASR**（`WS /sessions/{id}/asr/stream`，PCM 16k），中文识别更准 —— 后续可把输入层换成 WS 发 PCM，替换浏览器 ASR。这是已知的升级点，不是这一版做的。

## 触发行为说明（rules 模式，实测正常）

- **每个全新 session 都会稳定触发**：3 个全新 session 喂同一句缺口型对话（如 `这个问题谁负责，下周五deadline前能不能定？`），都出 `summary_gap_check` 镜端卡（"待明确 / 建议确认：负责人、截止时间。"）。
- **同一 session 内重复同一句** → 路由到 `app_prompt_tab` 而非再弹镜端卡。这是**正确的去重**（同一未解决缺口不重复弹镜端），by design。
- 请求体注意：`segment.end_ms` 必须 `> 0` 且 `> start_ms`，否则后端返回 **422**（前端 `api.js` 已按字数估 `end_ms`）。

> 接口链路已端到端验证：建会话 + transcript + 字段映射 + Vite proxy 全通，浏览器实测出真实镜端卡片。
