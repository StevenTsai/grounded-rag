# GroundedRAG 评测

两种评测模式：

## 1. Verify 模式（确定性校验门）

测试 claim-level 校验门的三条指标，**不调用 LLM**：

```bash
# 内置 eval set
python -m groundedrag.eval

# 真实场景 eval set
python -m groundedrag.eval --docs examples/real_seed_docs.jsonl \
  --rules examples/real_seed_rules.json \
  --eval examples/real_eval_set.jsonl

# JSON 输出
python -m groundedrag.eval --eval examples/real_eval_set.jsonl --json
```

### 指标

| 指标 | 含义 |
|------|------|
| citation_completeness_rate | 期望 pass 的主张中引用检查通过的占比 |
| surface_consistency_rate | 期望 pass 的主张最终状态为 pass 的占比 |
| refusal_correctness_rate | 期望 refuse 的主张最终状态为 refuse 的占比 |

## 2. E2E 端到端模式

调用完整 `pipeline.ask()` 流程（retrieval → LLM 生成 → claim 解析 → 校验门验证）。

### 配置 LLM

支持三种方式（优先级：CLI > .env > 环境变量）：

**方式 1：.env 文件（推荐）**
```bash
cp .env.example .env
# 编辑 .env，填入 API Key
python -m groundedrag.eval --e2e
```

**方式 2：环境变量**
```bash
export LLM_API_KEY=sk-xxx
python -m groundedrag.eval --e2e
```

**方式 3：命令行参数**
```bash
python -m groundedrag.eval --e2e --llm-api-key sk-xxx
```

### 多模型 Failover

参考壹鹿康行的多 provider 架构，支持主备自动切换：

| Provider | 环境变量 | 默认 Base URL | 默认模型 |
|----------|---------|--------------|---------|
| xiaomi | `XIAOMI_API_KEY` | token-plan-cn.xiaomimimo.com/v1 | mimo-v2.5-pro |
| deepseek | `DEEPSEEK_API_KEY` | api.deepseek.com/v1 | deepseek-chat |
| doubao | `DOUBAO_API_KEY` | ark.cn-beijing.volces.com/api/v3 | doubao-seed-2-0-pro |

统一覆盖变量（优先于 provider-specific）：`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`

```bash
# 指定主 provider，其他 provider 有 key 则自动作为备用
python -m groundedrag.eval --e2e --llm-provider xiaomi

# .env 配置示例
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxx
XIAOMI_API_KEY=xxx     # 自动成为备用
```

### 运行

```bash
# 全量评测
python -m groundedrag.eval --e2e

# 限制题数（快速验证）
python -m groundedrag.eval --e2e --e2e-max 3

# JSON 输出
python -m groundedrag.eval --e2e --json
```

### E2E 指标

| 指标 | 含义 |
|------|------|
| match_rate | actual_status == expected_status 的比例 |
| llm_usage_rate | 有 LLM 参与的比例（非规则直出） |
| rule_direct_rate | 规则直出比例 |
| verdict_distribution | PASS / REFUSE / ANNOTATE 各占比 |
| avg_latency_ms | 平均耗时 |

### E2E Eval Set 格式

```json
{
  "question": "EGFR突变的晚期肺癌一线应该推荐什么方案？",
  "context": {"cancer_type": "肺癌", "treatment_line": "一线", "biomarker": "EGFR"},
  "evidence_docs": [],
  "expected_verdicts": "pass",
  "expect_refusal": false
}
```

- `expected_verdicts`: 整体预期状态（"pass" / "refuse"），可选
- `evidence_docs`: 固定证据 ID（空 = 自动检索）
- 无手写 claims：LLM 自己生成主张，校验门自动验证

## Eval Sets

| 文件 | 模式 | 用例数 | 说明 |
|------|------|--------|------|
| eval_set.jsonl | verify | 19 | 内置合成数据 |
| real_eval_set.jsonl | verify | 15 | 真实场景（壹鹿康行数据） |
| e2e_eval_set.jsonl | e2e | 13 | 端到端（需 LLM） |
