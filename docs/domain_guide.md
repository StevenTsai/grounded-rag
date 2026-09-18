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
| `source_type` | 可选 | 来源类型：`guideline` / `clinical_trial` / `insurance` / `variant` / `regulation` / `literature` / `manual` / `generic`（默认）。非法值会导致 schema 校验失败 → 主张被拒 |
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

规则条件使用通用的 `field / op / value`，任意字段都可直接写入规则库。上下文提取分两层：

1. **医疗四字段**（`cancer_type` / `treatment_line` / `biomarker` / `gene`）走内置启发式；
2. **其余字段**（如金融的 `fund_type`、法律的 `region`）走**通用兜底**：只要规则条件里
   出现的取值在问题中被命中（原文子串 / 别名 / 字母数字 token 全含），即自动填入上下文。
   因此上面金融示例的 `fund_type` 无需改框架即可命中规则。

只有当内置启发式不满足需求（例如同一字段需要词形归一、上下文来自多轮对话而非问题文本）时，
才需要继承 `GuidelineEngine` 覆盖 `extract_context`（注意是公开方法，无下划线前缀）：

```python
from groundedrag.pipeline import Pipeline
from groundedrag.guardrail.engine import GuidelineEngine

class MyEngine(GuidelineEngine):
    def extract_context(self, question, synonym_map=None):
        ctx = super().extract_context(question, synonym_map)
        # 在通用兜底之上，补充你领域的专用提取 / 归一
        ctx["region"] = self._detect_region(question)
        return ctx

# 构造后替换引擎（Pipeline 暴露的引擎属性名是 engine）
pipe = Pipeline.build(docs, rules)
pipe.engine = MyEngine(pipe.engine.pathway_rules, pipe.engine.resistance_rules)
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

输出：`- 推荐方案：平均剩余到期期限≤120天`

> 无 API Key 时走「规则直出」确定性路径；配置了 LLM（`.env`）时会先调用模型，
> 由校验门逐条核验其输出。上例的规则命中与规则直出不依赖任何外部模型。
