# 实时链路延迟优化方案 (Latency Optimization Plan)

## 1. 问题描述

实测从 ASR 出 FINAL 转录到眼镜端实际内容输出,平均 **12 秒以上**,极端情况 **33 秒**。
TTFT(首字时间)= 整条产品流水线总时长,体验不可用。

目标:把 TTFT 压到 **~5s 以内**,内容流式呈现。

---

## 2. 根因分析(逐层拆解)

### 2.1 关键路径:每条 FINAL 转录触发的处理

```
FINAL 转录到达
│
├─ 1. base 记忆检索            prompt_category=None → 门控关闭 → 0 LLM
│
├─ 2. select_opportunities → detector.detect()      ★原为全串行★
│     ├─ 规则检测              0 LLM(本地,快)
│     ├─ UnknownTermDetector   1 LLM  fast model
│     └─ OpportunityDetector   1 LLM  fast model
│
├─ 3. _run_memory_aware_orchestration   ★逐 opportunity 串行循环★
│     for each opportunity(原最多 3 个):
│       ├─ per-opportunity 记忆检索(question/person 类 +1 LLM query understanding)
│       └─ generate_candidate → PromptGenerationService  1 LLM(gpt-5.5)
│
└─ 4. 输出:整个 transcript_step 全跑完才一次性 send_json
```

### 2.2 延迟来源(按贡献度排序)

| # | 来源 | 说明 | 可控性 |
|---|---|---|---|
| 1 | **端点不稳定(503)+ SDK 重试** | 代理端点返回 503,OpenAI SDK 默认重试 2 次,延迟 ×3,每次失败调用 ~16s | 部分(降重试)|
| 2 | **单次 LLM 调用慢** | gpt-5.5 在完整产品输入(记忆+会议状态+schema)下单次 7s+,大输入更慢 | 可控(换模型/裁输入)|
| 3 | **LLM 调用次数多** | 一条话最多 5-6 次 LLM 往返(检测 2 + 每 opportunity 生成 1 + query understanding) | 可控(降数量)|
| 4 | **全串行执行** | 检测两臂串行;opportunity 生成串行循环 | 可控(并发)|
| 5 | **无流式输出** | WebSocket 只在整条流水线跑完才推回,TTFT = 全流程时长 | 需架构改(流式)|

### 2.3 核心判断

- **ASR 是流式的** ✅
- **但 FINAL 之后整条决策/生成流水线是 batch + 全串行 + 多次 LLM 调用,WebSocket 只在全部完成后一次性推回。**
- 因此 **TTFT = 整条流水线总时长**,这是 12s 的根本原因。
- **不是流式架构**(在决策/生成层)。

---

## 3. 优化方案

### Phase 1 — A 类(低风险,不动架构,已完成)

| 编号 | 优化项 | 方法 | 状态 |
|---|---|---|---|
| **A1** | 检测两臂并发 | `UnknownTermDetector` 与 `OpportunityDetector` 是独立网络调用,用 `ThreadPoolExecutor` 并发(单臂时退化为串行无开销);各臂自吞异常返回 `[]`,future 不抛 | ✅ 完成 |
| **A4** | 眼镜端单条 prompt | realtime 路径在生成前把 opportunities 截断到 top-1(已按 priority/category/confidence 排序),生成调用 3→1;新增 `max_prompts` 参数默认 1 | ✅ 完成 |
| **A3** | 生成模型可配置 | `PromptOrchestrator` 加 `generation_model`;`PROACTIVE_PROMPT_GENERATION_MODEL=<name>` 或 `PROACTIVE_PROMPT_GENERATION_FAST=on` 切 fast model;默认保留 gpt-5.5 保质量 | ✅ 完成 |
| **A2** | per-opportunity 生成并发 | A4 后只生成 1 条,自动满足(无需并发) | ✅ 自动满足 |

#### A4 内容策略(眼镜端只显示 1 条,按类别分形态)

| 类别 | 形态 | 显示示例 |
|---|---|---|
| `concept_explanation`(通识/陌生词) | 直接回答 | "GMV=商品交易总额" |
| `person_or_fact`(记忆检索) | 直接回答 | "上次张三负责" |
| `question_answer`(提问) | 直接回答 | 直接给答案 |
| `summary_gap_check`(日程/缺口) | 关键信息 | "负责人未定·周五" |
| `suggestion`(建议) | 关键信息 | 一句话建议 |

### Phase 1.5 — 端点容错(新增,已完成)

| 编号 | 优化项 | 方法 | 状态 |
|---|---|---|---|
| **R1** | fail-fast 重试 | `max_retries` 由 SDK 默认 2 降为可配置(`OPENAI_MAX_RETRIES`,默认 1);端点 503 时延迟从 ×3 降到 ×2,33s→21s。可设 0 纯 fail-fast | ✅ 完成 |

### Phase 2 — B 类(架构级,真流式,待端点恢复后做)

| 编号 | 优化项 | 方法 | 状态 |
|---|---|---|---|
| **B2** | 真流式输出 | `ModelClient` 加流式接口 → `PromptGenerationService` 流式变体 → 编排流式 → WebSocket 增量推送 token;眼镜端 `glasses_title` 先到、文本边生成边显示。TTFT = 首 token 时间(~1-2s) | ⏳ 待做 |

> 说明:用户已批准 A 类全套 + B2(真流式),**未选** B1(合并检测+生成为单次调用)。保留 `检测 → 生成` 结构,A1/A4/B2 叠加不冲突。

---

## 4. 预期 vs 实测效果

| 阶段 | 检测 | 生成 | 端到端(实测) |
|---|---|---|---|
| 优化前(端口错→503 + gpt-5.5) | 串行 + 失败重试 | 失败回退规则 | **21-33s** |
| Phase 1 + 端口修正 + gpt-4o-mini | 并发 ~1.5s | 单次 ~1.3s | **✅ 实测 5.5-6.2s** |
| Phase 2(B2 流式)后 | 并发 ~1.5s | 流式首 token ~0.5s | 目标 TTFT **~2-3s** |

实测命令(产品流水线真实调用):
```
这个项目的 ddl 还没定，谁来跟进一下？
→ run0: 6238ms | summary_gap_check | 待跟进 | 这个项目的 ddl 还没定。
→ run1: 5475ms | summary_gap_check | 跟进未定 | 这个项目的 ddl 还没定，谁来跟进？
```
LLM 真实成功(非规则兜底),眼镜端内容正确、≤30 字。

---

## 5. 根因复盘:503 是配置错,不是端点挂

实测两个端口:
- `8.130.129.37:53245`(公司文档端口):**gpt-4o-mini 2.2s 正常** ✅
- `8.130.129.37:58081`(`.env` 旧端口):**503** ❌

503 的真凶是 **`.env` 里 base_url 端口写成了已停用的 58081**,公司文档(6/12 更新)已迁到 53245。修正端口后端点完全正常。

候选模型延迟对比(53245):
| 模型 | 延迟 | 备注 |
|---|---|---|
| deepseek-v4-flash | 978ms | ⚠️ 返回空内容,暂不用 |
| **gpt-4o-mini(选用)** | 1350ms | 中文好,兼容结构化输出管线 |
| gpt-5.5 | 2614ms | 慢一倍 |

### 已落地配置变更
- `OPENAI_BASE_URL`: `:58081` → `:53245`
- `OPENAI_MODEL`: `gpt-5.5` → `gpt-4o-mini`(默认==fast,全链路用小模型)

### 后续动作
1. ✅ 12s 问题已解决(端到端 ~6s,LLM 正常)。
2. ⏳ B2 流式(可选):把 TTFT 进一步压到 ~2-3s,体验"边说边出"。当前 6s 已可用,B2 为锦上添花。

---

## 6. 配置开关汇总

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `PROACTIVE_PROMPT_GENERATION_MODEL` | (空) | 指定生成模型 |
| `PROACTIVE_PROMPT_GENERATION_FAST` | off | 生成用 fast model |
| `OPENAI_MAX_RETRIES` | 1 | LLM 调用重试次数(0=纯 fail-fast) |
| `PROACTIVE_GLASSES_PROMPT_TIMEOUT_SECONDS` | 35 | 眼镜端生成硬超时 |
| `PROACTIVE_ASR_STREAM_FINAL_TIMEOUT_SECONDS` | 30 | 流式 FINAL 收尾超时 |
