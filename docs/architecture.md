# GroundedRAG 架构设计

本文档说明 GroundedRAG 的三层架构设计、核心数据流、关键决策点与扩展接口。

## 设计原则

1. **机制开源，数据可闭源**（Open Core）—— 框架代码 MIT 开源，业务方的领域词典/规则库/真实数据可闭源。
2. **零 ORM 耦合** —— 规则引擎通过 `RuleProvider` 抽象接口加载规则，可对接任意数据源（JSON / 数据库 / API）。
3. **确定性优先** —— 校验门核心逻辑（引用完整性/表面一致性/冲突裁定/充分性）纯逻辑判定，零 LLM 依赖，可离线测试。
4. **无 LLM 可演示** —— 规则命中走"规则直出"（权威背书），未命中走结构化拒答模板；Gradio demo 无需配置 API Key 即可完整运行。
5. **领域无关** —— 框架本身不绑定医疗，`examples/` 合成示例可替换为金融/法律/教育领域数据。

---

## 三层架构总览

```
┌─────────────────────────── pipeline.py ───────────────────────────┐
│  编排：检索 → 规则匹配 → 受约束生成 → 主张解析 → 校验门 → 降级/拒答 │
├────────────────┬──────────────────────────┬───────────────────────┤
│  retriever/     │       guardrail/ ★        │         llm/          │
│                │  （核心差异化）            │                       │
│  BM25           │  规则引擎（纯Python）      │  LLMService 策略基类  │
│  实体增强        │  证据溯源 EvidenceId      │  OpenAI 兼容客户端    │
│  查询扩展        │  声明解析 AnswerClaim     │  模板回退             │
│  (词典可注入)     │  ★ 声明级校验门 verifier  │  主备降级 FailoverLLM │
└────────────────┴──────────────────────────┴───────────────────────┘
```

### 各层职责

| 层 | 输入 | 输出 | 核心类 |
|----|------|------|--------|
| **retriever** | 问题 + 文档语料 | 召回文档列表（按 BM25 相关度排序） | `BM25` / `Retriever` |
| **guardrail** | 问题 + 召回文档 + 规则库 | 证据集 + 规则决策 + 校验报告 | `GuidelineEngine` / `Verifier` |
| **llm** | 问题 + 证据 + 规则约束 | 结构化主张列表（带引用锚点） | `LLMService` / `FailoverLLM` |
| **pipeline** | 问题 | 可信回答 + 溯源链路 + 校验报告 | `Pipeline` / `PipelineResult` |

---

## 核心数据流

### 1. 用户提问 → 检索与规则匹配

```python
# 输入
question = "EGFR 突变的晚期肺癌一线推荐什么方案？"

# retriever 召回（BM25 + 实体增强）
retrieved = [
    RetrievedDoc(document=Document(doc_id="doc-lung-egfr-1st", ...), score=0.85),
    ...
]

# guardrail/engine 规则匹配
matched = [
    RuleDecision(
        rule_id="r-lung-egfr-1st",
        cancer_type="肺癌",
        treatment_line="一线",
        biomarker="EGFR",
        recommendations=[
            RuleRecommendation(plan_name="奥希替尼", grade="A", ...)
        ],
        source_version="CSCO-2026"
    )
]

# guardrail/evidence 证据归一化
evidence_ids = [
    EvidenceId(
        evidence_id="doc-lung-egfr-1st-ev1",
        doc_id="doc-lung-egfr-1st",
        source_type="guideline",
        grade="A",
        updated_at="2026-01-15",
        text_span="奥希替尼为三代EGFR-TKI，用于EGFR突变阳性晚期肺癌一线标准治疗..."
    )
]
```

### 2. 受约束生成 → 结构化主张

```python
# llm 层生成（带引用约束的 prompt）
prompt = f"""
根据以下证据与规则回答问题，每条主张必须标注 [证据n] 或 [规则n] 锚点：

证据：
- [证据1] 奥希替尼为三代EGFR-TKI...

规则：
- [规则1] 肺癌 一线 EGFR → 奥希替尼 (A级)

问题：{question}
"""

# LLM 输出（结构化）
raw_answer = """
推荐方案：奥希替尼 [规则1] [证据1]
推荐剂量：80mg 每日一次 [证据1]
"""

# guardrail/claims 解析为主张列表
claims = [
    AnswerClaim(
        text="推荐方案：奥希替尼",
        type="indication",  # LLM 自标
        evidence_refs=["doc-lung-egfr-1st-ev1"],
        rule_refs=["r-lung-egfr-1st"],
        critical=True
    ),
    AnswerClaim(
        text="推荐剂量：80mg 每日一次",
        type="factual",
        evidence_refs=["doc-lung-egfr-1st-ev1"],
        rule_refs=[],
        critical=True
    )
]
```

### 3. 声明级校验门 → 三态判定

```python
# guardrail/verifier 逐条校验
verdicts = [
    ClaimVerdict(
        claim=claims[0],
        status=PASS,  # ✅ 有据可依
        reason="rule_authoritative",  # 规则直出，权威背书
        overridden_type="indication",  # verifier 确认类型
        checks={
            "citation": {"complete": True},
            "claim_support": {"ratio": 1.0, "reason": "pass"},
            "conflict": None,
            "sufficiency": {"sufficient": True, "grade": "A"}
        }
    ),
    ClaimVerdict(
        claim=claims[1],
        status=PASS,  # ✅ 数值与证据一致
        reason="pass",
        overridden_type="factual",
        checks={
            "citation": {"complete": True},
            "claim_support": {"ratio": 1.0, "reason": "pass", "surface_tokens": ["80mg", "每日", "一次"]},
            "sufficiency": {"sufficient": True, "grade": "A"}
        }
    )
]

# 整体判定
report = VerificationReport(
    verdicts=verdicts,
    overall_status=PASS,  # 所有主张通过
    overall_reason="pass"
)
```

### 4. 最终交付

```python
result = PipelineResult(
    question=question,
    answer_text="推荐方案：奥希替尼 [规则1] [证据1]\n推荐剂量：80mg 每日一次 [证据1]",
    status="pass",  # pass / annotate / refuse
    report=report,
    evidence=evidence_ids,
    matched_rules=matched,
    retrieved=retrieved,
    used_llm="deepseek-chat"  # 或 None（规则直出）
)
```

---

## 关键设计决策

### D1: 主张类型"确定性覆盖"（防 LLM 自报绕过）

**问题**：LLM 可能把「推荐使用 A」错误标注为 `factual`，表面要素命中证据后通过校验，
即使证据写的是「**不可**使用 A」。

**方案**：`verifier.py` 用表层信号（否定词/比较词/因果词正则）**强制覆盖** LLM 标注：

```python
def classify_claim_type(text: str) -> str:
    """确定性分类，优先级：negation > comparison > causal > indication > factual"""
    if re.search(r"不|不可|禁忌|禁用|慎用|避免", text):
        return "negation"
    if re.search(r"优于|差于|一线|二线|首选|先用", text):
        return "comparison"
    if re.search(r"导致|激活|抑制|引起|由于", text):
        return "causal"
    if re.search(r"推荐|使用|选择|考虑", text) and has_entity(text):
        return "indication"
    return "factual"
```

`AnswerClaim.from_dict()` 只对**非法** type 值重判，合法但失真的 type 保留待 verifier 覆盖。

### D2: 表面要素一致性 ≠ 真实支持

**问题**：用户/评委可能误读"一致性检查"为"验证了语义正确性"。

**方案**：
1. **明确命名** —— `_check_surface_consistency`（不叫 `_check_support`）
2. **指标口径** —— metrics.md 显式声明"表面一致率 ≠ 真实支持率"
3. **关系型拒答** —— 含否定/比较/因果的主张在确定性档下**统一拒答或标注**，不冒充验证过

```python
# verifier.py:183
def _check_surface_consistency(claim: AnswerClaim, evidence: EvidenceId, ...) -> dict:
    """表面要素一致性（实体/数值 token overlap），NOT 语义真实支持。"""
    claim_tokens = surface_tokens(claim.text)
    evidence_tokens = surface_tokens(evidence.text_span)
    overlap = len(claim_tokens & evidence_tokens)
    ratio = overlap / len(claim_tokens) if claim_tokens else 0
    # 纯事实型 ratio ≥ 0.6 通过；关系型主张由决策表拒答
    ...
```

### D3: 规则冲突裁定（stale → grade → time → 分歧声明）

**问题**：两条规则推荐不同方案（如 A 药 vs B 药），如何裁决？

**方案**：确定性裁定顺序（`verifier.py:309` `arbitrate_conflict`）：

1. **过滤 stale** —— 证据 `updated_at` 距今超过阈值（指南 2 年/试验 5 年）→ 退出竞争
2. **按 grade 取最高** —— A > B > C > D
3. **grade 相同 → 按 updated_at 取最新** —— 缺失 `updated_at` 排在有时间戳者之后
4. **仍无胜出者（同级同时但内容矛盾）→ 分歧声明** —— 不静默二选一，输出"两种推荐均存在，请结合临床判断"

```python
# 冲突定义：同条件域（癌种+线次+biomarker）+ 方案互斥 + 均非 stale
def _is_conflict(r1: RuleDecision, r2: RuleDecision) -> bool:
    if r1.cancer_type != r2.cancer_type or r1.treatment_line != r2.treatment_line:
        return False  # 不同域，互补而非冲突
    plans1 = {normalize_drug_name(r.plan_name) for r in r1.recommendations}
    plans2 = {normalize_drug_name(r.plan_name) for r in r2.recommendations}
    return plans1.isdisjoint(plans2)  # 方案集合互斥
```

### D4: 无 LLM 可演示（规则直出 + 模板回退）

**问题**：评审现场/演示视频录制时，可能无外网或 API Key 过期。

**方案**：
- **规则命中** → `pipeline.py` 直接拼装答案（无需 LLM），标记 `used_llm=None` + `reason=rule_authoritative`
- **未命中 + 证据不足** → `llm/template.py` 结构化拒答文案（"该问题缺乏充分证据，建议携带病历线下就诊"）
- **Gradio 对照** → 裸 RAG 模式在无 LLM 时用 eval_set 幻觉变体注入展示，明确标注"模拟注入，仅作对照"

---

## 扩展接口

### 接入自定义 LLM

```python
from groundedrag.llm import LLMService

class MyCustomLLM(LLMService):
    def generate(self, prompt: str, context: dict | None = None) -> str:
        # 调用你的 API
        return my_api_call(prompt)

pipe = Pipeline.build_from_json(..., llm=FailoverLLM([MyCustomLLM()]))
```

### 接入数据库规则

```python
from groundedrag.guardrail import RuleProvider, Rule

class DatabaseRuleProvider(RuleProvider):
    def load_pathway_rules(self) -> list[Rule]:
        # 从 MySQL/PostgreSQL 查询
        rows = db.execute("SELECT * FROM treatment_pathway_rules")
        return [Rule.from_dict(row) for row in rows]

engine = GuidelineEngine(provider=DatabaseRuleProvider())
```

### 自定义分词器

```python
from groundedrag.retriever import BM25

def my_tokenizer(text: str) -> list[str]:
    # 英文用 nltk，日文用 mecab
    return nltk.word_tokenize(text)

bm25 = BM25(corpus, tokenizer=my_tokenizer)
```

---

## 性能考量

| 模块 | 时间复杂度 | 典型耗时（73 文档 + 22 规则） |
|------|-----------|---------------------------|
| BM25 构建 | O(D × avg_len) | ~20ms |
| BM25 检索 | O(D) | ~5ms / query |
| 规则匹配 | O(R × C) R=规则数, C=条件数 | ~2ms |
| 校验门 | O(N × E) N=主张数, E=证据数 | ~10ms（5 条主张 × 3 证据） |
| LLM 调用 | 外部 API 延迟 | 500-2000ms（主要瓶颈） |

**优化建议**：
- 生产环境预构建 BM25 索引（序列化为 JSON 缓存）
- 规则引擎按条件域建索引（hash table），避免全量遍历
- 多问题并发调用 LLM（asyncio / batch API）

---

## 测试策略

- **单元测试**（187 个，覆盖 5 档校验门全部分支）：`tests/test_*.py`
- **端到端测试**：`test_pipeline.py`（规则直出 + 分歧拒答 + LLM 结构化）
- **评测集回归**：`test_eval_runner.py`（三指标 ≥ 0.95）
- **合规扫描**：`test_leak_scan.py`（专利草稿 / 凭据泄漏自检）

---

## 后续演进

见 [README Roadmap](../README.md#roadmap)：
- v1.1 语义档（NLI 蕴含判定）
- v1.2 OncoKG 图谱证据链
- v1.3 多 LLM 对齐验证
- v2.0 多领域泛化（金融/法律/教育）
