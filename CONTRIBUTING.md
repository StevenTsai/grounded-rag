# 贡献指南

感谢对 GroundedRAG 的关注。这是一个**轻量开源 RAG 框架**，目标是让 LLM 回答做到
"有据可依、无据可拒"。欢迎提交 Issue、文档修订与代码贡献。

## 开发环境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,demo]"
pytest                    # 测试
ruff check                # lint（配置见 pyproject.toml，规则集 E4/E7/E9/F/I/W）
python -m groundedrag.eval           # 评测（verify 模式）
python -m groundedrag.eval --e2e     # 端到端评测（需 .env 配置 LLM）
python tools/leak_scan.py # 开源合规自检（push 前必跑）
```

## 代码约定

- 后端风格：Python 3.10+，类型注解，`from __future__ import annotations`，
  `typing.List/Dict/Optional` 风格（勿混用小写内置泛型）。
- 中文字符串/注释/docstring，英文变量名与标识符；行长 ≤100。
- 包内导入用绝对导入（`from groundedrag.…`）。
- 新增/修改接口前先看对应模块 docstring 与既有 schema，避免重复定义。

## 分层职责

| 目录 | 职责 | 红线 |
|------|------|------|
| `src/groundedrag/retriever/` | BM25 检索、实体增强、查询扩展 | 中文切词只允许 jieba 或 CJK char-bigram 回退，**绝不 `str.split()`** |
| `src/groundedrag/guardrail/` | 规则引擎 / EvidenceId 溯源 / AnswerClaim 解析 / **verifier 校验门** | 校验门必须纯确定性、零 LLM 依赖；不 import ORM/数据库 |
| `src/groundedrag/llm/` | 策略基类 / OpenAI 兼容客户端 / 模板回退 / 主备降级 | 降级链保证"永不因模型故障抛给用户" |
| `src/groundedrag/pipeline.py` | 编排 | 无 API Key 也必须可跑（规则直出 / 结构化拒答） |
| `src/groundedrag/eval/` | 评测（verify + e2e 两种模式） | 指标口径见 docs/metrics.md，不得改称"真实支持率" |
| `examples/` | 自研合成演示数据 + demo | **禁止放入真实临床/患者/第三方受版权数据** |
| `tools/` | 合规与工具 | `leak_scan.py` 内不写具体私有词，词表走 `.leak_markers.json` |

## 提交前检查清单（compliance）

开源仓库出现在公网前，以下任何一项不满足都不应 push：

1. [ ] `pytest` 全绿；`ruff check` 无告警。
2. [ ] `python -m groundedrag.eval --json` 三指标为预期值（1.0 / 高值）。
3. [ ] `python tools/leak_scan.py` 退出码 0。
4. [ ] 若有疑似参考的**内部文档/草稿**，先跑 `tools/leak_scan.py --refs <内部目录> --threshold 0.5`
       核对相似度 —— **改写≠清库**，高度相似段落不要进开源仓库。
5. [ ] 维护方如有私有词（内部代号/客户名/未公开术语），写入**不入库**的
       `.leak_markers.json`（模板见 `.leak_markers.example.json`）后再跑一次 scan。
6. [ ] 演示数据只增不减地自检：新 `examples/` 条目应为**自研合成**，不含真实病例/
       指南原文/受版权数据；新增依赖先核对许可证并更新 THIRD_PARTY_NOTICES.md。
7. [ ] 涉及行为语义的改动，同步 README / docs 中对应口径说明与示例。

## 分支与 PR

- 从 `main` 切特性分支；PR 描述说明改动动机、影响面与自测结果。
- 保持单 PR 单一主题；文档与代码分开更易审。

## 提问与 Issue

- Bug 报告请附：复现问题、期望行为、实际输出、`groundedrag` 版本。
- 特性提议请先说明使用场景与不动摇的三条边界：
  ① 校验门确定性（无 LLM 依赖）；② 核心运行时依赖最小化；
  ③ 无 API Key 也能端到端可跑。

## 维护者须知（仅仓库维护者）

本仓库面向公网开源，**独立成篇**：机制与代码自洽、合成演示数据自足、不携带任何
未授权内容。若你在维护另一处内部项目/数据，切勿以任何形式（含 git 历史、issue、
文档链接）将内部代码、数据或未公开材料带入本开源仓库 —— `git log` 同样是公网可见内容。
