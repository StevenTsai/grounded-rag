# 领域适配指南

GroundedRAG 的核心校验门是**领域无关**的——它只检查主张是否有可解析的锚点、表面要素是否一致、规则是否冲突。要适配新领域，你只需要准备两样东西：**证据文档**和**规则库**。

## 快速开始

```bash
# 1. 安装
pip install -e .

# 2. 生成模板文件
groundedrag init --dir my_domain/

# 3. 编辑模板（填入你的文档和规则）
vim my_domain/my_seed_docs.jsonl
vim my_domain/my_seed_rules.json

# 4. 提问
groundedrag ask "你的问题" --docs my_domain/my_seed_docs.jsonl --rules my_domain/my_seed_rules.json
```

## 证据文档格式（JSONL）

每行一个 JSON 对象，字段说明：

| 字段 | 必填 | 说明 |
|------|------|------|
| `doc_id` | ✅ | 文档唯一 ID（字符串） |
| `content` | ✅ | 文档正文（校验门从此文本中提取实体/数值） |
| `title` | 可选 | 文档标题 |
| `source_type` | 可选 | 来源类型：`guideline` / `literature` / `regulation` / `manual` |
| `source_version` | 可选 | 来源版本（如 `"CSCO-LUNG-2026"`） |
| `grade` | 可选 | 证据等级：`A` / `B` / `C` / `D` |
| `updated_at` | 可选 | 更新日期（`YYYY-MM-DD`），用于时效性校验 |
| `entities` | 可选 | 关键实体列表（辅助检索） |

### 示例：金融领域

```json
{"doc_id": "doc-mica-001", "content": "货币市场基金投资组合的平均剩余到期期限不得超过120天。", "source_type": "regulation", "source_version": "证监会-2026", "grade": "A", "updated_at": "2026-01-01", "entities": ["货币市场基金", "到期期限", "120天"]}
```

### 示例：法律领域

```json
{"doc_id": "doc-labor-001", "content": "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。", "source_type": "regulation", "source_version": "劳动合同法-2024", "grade": "A", "updated_at": "2024-01-01", "entities": ["劳动合同", "二倍工资", "一个月"]}
```

## 规则库格式（JSON）

单个 JSON 文件，包含 `pathway_rules` 数组：

```json
{
  "pathway_rules": [
    {
      "rule_id": "rr-finance-mmf-001",
      "name": "货币市场基金期限限制",
      "conditions": [
        {"field": "fund_type", "op": "eq", "value": "货币市场基金"}
      ],
      "recommendations": [
        {
          "plan_name": "平均剩余到期期限≤120天",
          "grade": "A",
          "note": "依据证监会规定",
          "source_version": "证监会-2026"
        }
      ],
      "source_version": "证监会-2026",
      "updated_at": "2026-01-01"
    }
  ]
}
```

### 规则字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `rule_id` | ✅ | 规则唯一 ID |
| `name` | ✅ | 规则名称 |
| `conditions` | ✅ | 匹配条件（见下文） |
| `recommendations` | ✅ | 推荐方案列表 |
| `source_version` | 可选 | 权威来源版本 |
| `updated_at` | 可选 | 更新日期 |

### 条件系统

条件是通用的 `field / op / value` 结构：

| 操作符 | 含义 | 示例 |
|--------|------|------|
| `eq` | 等于 | `{"field": "fund_type", "op": "eq", "value": "货币市场基金"}` |
| `neq` | 不等于 | `{"field": "status", "op": "neq", "value": "已废止"}` |
| `in` | 在列表中 | `{"field": "region", "op": "in", "value": ["北京", "上海"]}` |
| `contains` | 包含 | `{"field": "keywords", "op": "contains", "value": "利率"}` |
| `all` | 全部满足 | `{"field": "tags", "op": "all", "value": ["合规", "优先"]}` |

### 领域字段映射

默认的规则引擎使用 `cancer_type` / `treatment_line` / `biomarker` 作为上下文字段（医疗领域命名）。其他领域需要覆盖上下文提取逻辑：

```python
from groundedrag.pipeline import Pipeline
from groundedrag.guardrail.engine import GuidelineEngine

# 自定义上下文提取
class MyEngine(GuidelineEngine):
    def _extract_context(self, question: str, evidence_ids=None):
        ctx = super()._extract_context(question, evidence_ids)
        # 从 question 中提取你领域的关键字段
        ctx["fund_type"] = self._detect_fund_type(question)
        ctx["region"] = self._detect_region(question)
        return ctx

# 使用自定义引擎
pipe = Pipeline.build(docs, rules)
pipe.guideline_engine = MyEngine(...)
```

## 评测集格式（JSONL）

用于验证你的领域适配是否正确：

```json
{
  "question": "货币市场基金的期限限制是什么？",
  "context": {"fund_type": "货币市场基金"},
  "evidence_docs": ["doc-mica-001"],
  "claims": [
    {"text": "货币市场基金平均剩余到期期限不得超过120天", "evidence_refs": ["doc-mica-001"], "expected": "pass"}
  ],
  "expect_refusal": false
}
```

运行评测：

```bash
python -m groundedrag.eval --docs my_docs.jsonl --rules my_rules.json --eval my_eval.jsonl
```

## 完整示例：金融合规

### 1. 准备文档

```bash
cat > finance_docs.jsonl << 'EOF'
{"doc_id": "doc-mica-001", "content": "货币市场基金投资组合的平均剩余到期期限不得超过120天。", "source_type": "regulation", "source_version": "证监会-2026", "grade": "A", "updated_at": "2026-01-01", "entities": ["货币市场基金", "120天"]}
{"doc_id": "doc-mica-002", "content": "单只货币市场基金持有的同一机构发行的证券市值不得超过基金资产净值的10%。", "source_type": "regulation", "source_version": "证监会-2026", "grade": "A", "updated_at": "2026-01-01", "entities": ["货币市场基金", "10%"]}
EOF
```

### 2. 准备规则

```bash
cat > finance_rules.json << 'EOF'
{
  "pathway_rules": [
    {
      "rule_id": "rr-mmf-term",
      "name": "货币市场基金期限限制",
      "conditions": [{"field": "fund_type", "op": "eq", "value": "货币市场基金"}],
      "recommendations": [{"plan_name": "平均剩余到期期限≤120天", "grade": "A", "source_version": "证监会-2026"}],
      "source_version": "证监会-2026",
      "updated_at": "2026-01-01"
    }
  ]
}
EOF
```

### 3. 提问

```bash
groundedrag ask "货币市场基金的期限限制是什么？" \
  --docs finance_docs.jsonl \
  --rules finance_rules.json
```

输出：`- 货币市场基金 推荐方案：平均剩余到期期限≤120天`
