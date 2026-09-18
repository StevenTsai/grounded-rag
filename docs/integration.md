# 集成指南（Integration Guide）

GroundedRAG 以 **Python 依赖包**形式集成，无内置服务、数据库或网络调用。按需选择三种接入深度：

| 方式 | 适用场景 | 入口 |
|------|---------|------|
| 1. 整条流水线 | 新项目 / 快速验证 | `Pipeline.build_from_json(...)` |
| 2. 只挂校验门 | **已有 RAG，想加一道确定性校验** | `Verifier().verify(...)` |
| 3. 单组件复用 | 只用检索 / 规则匹配 / LLM 降级 | `Retriever` / `GuidelineEngine` / `FailoverLLM` |

## 0. 安装与依赖引用

> `groundedrag` 尚未发布 PyPI。请引用**固定 tag**（不要引用分支），保证构建可复现。

```txt
# requirements.txt
groundedrag @ git+https://gitee.com/miniclaw27/grounded-rag.git@v1.0.0
```

```toml
# pyproject.toml
dependencies = [
  "groundedrag @ git+https://gitee.com/miniclaw27/grounded-rag.git@v1.0.0",
]
```

```toml
# poetry
groundedrag = { git = "https://gitee.com/miniclaw27/grounded-rag.git", tag = "v1.0.0" }
```

也可构建 wheel 后分发（企业环境建议放入内部制品库）：

```bash
python -m build
pip install dist/groundedrag-1.0.0-py3-none-any.whl
```

要求 Python 3.10+；核心运行时依赖仅 `jieba`（MIT）。

## 1. 整条流水线（新项目）

```python
from groundedrag import Pipeline

pipe = Pipeline.build_from_json("docs.jsonl", "rules.json")   # 或 Pipeline.build(docs, rules)
result = pipe.ask("EGFR 突变肺癌一线推荐什么方案？")

print(result.status)        # PASS / ANNOTATE / REFUSE
print(result.answer_text)   # 已通过校验门的回答
```

- 无 API Key 也能跑：命中规则 → 规则直出；未命中 → 结构化拒答
- 接入真实 LLM：`Pipeline.build_from_json(..., llm=FailoverLLM([...]))`（见 [LLM 配置](#4-llm-配置)）

## 2. 只挂校验门（已有 RAG 的项目，推荐）

保留你自己的检索与生成，只在交付前增加确定性校验：

```python
from groundedrag.guardrail import (
    EvidenceId, EvidenceRegistry, parse_claims, Verifier,
)

# 1) 你的检索结果 → 证据锚点
evidences = [
    EvidenceId(
        evidence_id="ev1",
        doc_id="doc-1",
        source_type="guideline",      # 取值见 VALID_SOURCE_TYPES
        source_version="CSCO-LUNG-2026",
        grade="A",                    # A/B/C/D
        updated_at="2026-02-01",      # ISO 日期，缺失视为未知版本
        text_span="奥希替尼推荐剂量 80mg，每日口服一次。",
    ),
]
registry = EvidenceRegistry(evidences)

# 2) 你的 LLM 输出 → 原子主张（支持 JSON，或带 [证据N]/[规则N] 锚点的文本行）
claims = parse_claims(llm_output, evidence_ids=[e.evidence_id for e in evidences])

# 3) 确定性校验（纯逻辑、零 LLM）
report = Verifier().verify(claims, registry, matched_rules=[])
print(report.overall_status)            # PASS / ANNOTATE / REFUSE
for v in report.verdicts:
    print(v.status, v.reason, "|", v.claim.text)
```

完整可运行示例（无需 API Key、无需网络）：[`examples/integrate_verifier.py`](../examples/integrate_verifier.py)

注意：

- 主张必须携带可解析锚点；无锚点时确定性绑定器会尝试回绑，绑定失败 → 拒绝（设计使然：无据不放行）
- 只做证据校验时可传 `matched_rules=[]`；规则冲突裁定与充分性门槛需要规则命中结果（`GuidelineEngine.match(...)`）
- 校验门是同步 API；如需跨语言/微服务，请自行包一层 HTTP（例如 FastAPI）

## 3. 单组件复用

| 组件 | 用途 | 导入 |
|------|------|------|
| `Retriever` | BM25 + 实体增强 + 查询扩展 | `from groundedrag.retriever import Retriever` |
| `GuidelineEngine` | 规则匹配 / `RuleDecision` | `from groundedrag.guardrail import GuidelineEngine` |
| `EvidenceRegistry` | 证据锚点注册与解析 | `from groundedrag.guardrail import EvidenceRegistry` |
| `FailoverLLM` / `OpenAICompatibleLLM` | 多 provider 降级 / OpenAI 兼容调用 | `from groundedrag.llm import FailoverLLM` |

## 4. LLM 配置

```bash
cp .env.example .env    # 配置 API Key（支持多 provider 自动降级）
```

优先级：CLI 参数 > `.env` > 环境变量；无 Key 时自动走规则直出 / 模板拒答，不影响校验门运行。

## 5. 领域适配

校验门与领域无关，只需准备证据文档（JSONL）与规则库（JSON），详见
[domain_guide.md](domain_guide.md)；金融示例已纳入可执行测试
（`tests/test_domain_guide.py`）。

## 6. 版本、发布与引用

- **版本策略**：SemVer。PyPI 发布前请引用固定 tag；发布后推荐 `groundedrag>=1,<2`
- **学术/材料引用**：见仓库根目录 [`CITATION.cff`](../CITATION.cff)
- **发布清单**（维护者）：
  1. 更新 `CHANGELOG.md` 与 `pyproject.toml` 版本号
  2. 合并到 master，确认 CI 全绿
  3. 打 tag（`git tag -a vX.Y.Z -m "..."`）并推送两个远程
  4. 创建 GitHub / Gitee Release，附 wheel 与 release notes
  5. （可选）发布 PyPI：
     ```bash
     pip install build twine
     python -m build
     python -m twine check dist/*
     python -m twine upload dist/*    # 用户名填 __token__，密码填 PyPI API token（勿写入脚本/仓库）
     ```
     发布后用 `pip install groundedrag` 即可安装（推荐约束 `groundedrag>=1,<2`）；
     随后把 README / 本文档的安装示例切换到 PyPI 写法。
