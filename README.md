# GroundedRAG

**轻量开源 RAG 框架** —— 以「声明级校验门（claim-level verifier）」可复现地减少无证据输出，
让 LLM 回答做到「**有据可依，无据可拒**」。

GroundedRAG 面向医疗等高风险领域的检索增强问答设计：回答不是"生成即交付"，而是先拆成
原子主张（`AnswerClaim`），每条绑定可溯源证据（`EvidenceId`）或命中的权威规则
（`RuleDecision`），逐条经过确定性校验门后，才决定**原样输出 / 标注降级 / 拒绝回答**。

> ⚠️ 本项目内数据（`examples/`）为**自研合成示例**，用于演示与评测机制，**不构成真实诊疗建议**。
> 请勿用于任何真实医疗决策。

## 为什么是"声明级"

普通 RAG 用"整段相关"打分，模型可以编造一段开头像检索结果的话，段内却混入幻觉。GroundedRAG
把回答降维到 **一条主张 = 一条可核查断言**，校验门逐条判定：

```
AnswerClaim[]  ──引用──▶  EvidenceId[]  /  RuleDecision[]
      │
      ▼  （纯逻辑，零 LLM 依赖）
 引用完整性 → 表面要素一致性 → 规则冲突裁定 → 证据充分性
      │
      ▼
  PASS（原样输出）/ ANNOTATE（附标注）/ REFUSE（拒答/删除）
```

确定性档只做**表面要素一致性**（实体/数值/单位是否真实出现在证据原文），不声称"语义真实支持"；
含否定/比较/因果等关系语义的主张在语义档关闭时统一拒答或标注 —— 详见 [docs/metrics.md](docs/metrics.md)
（指标口径）与各模块 docstring。

## 快速开始

```bash
pip install -e ".[demo]"     # 或最小安装 pip install -e .
python examples/demo.py      # CLI 端到端 demo
python examples/app.py       # Gradio 可视化 demo（评审演示主界面）
```

无 API Key 也能跑通全链路：规则命中走"规则直出"（有据可答），未命中则结构化拒答。

```python
from groundedrag.pipeline import Pipeline

pipe = Pipeline.build_from_json(
    "examples/seed_docs.jsonl", "examples/seed_rules.json"
)
result = pipe.ask("EGFR 突变的晚期肺癌一线推荐什么方案？")
print(result.answer_text)
# - 肺癌 一线 EGFR 推荐方案：奥希替尼
```

启用真实 LLM（OpenAI 兼容端点，如 DeepSeek / 小米 MiMo / 豆包）：

```python
from groundedrag.llm import FailoverLLM, OpenAICompatibleLLM

llm = FailoverLLM([OpenAICompatibleLLM(base_url="https://api.deepseek.com/v1",
                                        api_key="sk-...", model="deepseek-chat")])
pipe = Pipeline.build_from_json("examples/seed_docs.jsonl",
                                "examples/seed_rules.json", llm=llm)
```

## 评测（三项指标）

```bash
python -m groundedrag.eval.runner                  # 读 examples/ 默认三文件
python -m groundedrag.eval.runner --json           # 只输出 JSON 摘要
```

内置可重复评测集 `examples/eval_set.jsonl`（19 组 24 条，含正例、数值幻觉、关系型反例、
证据侧否定翻转、类型自报绕过等），输出：

| 指标 | 口径（README / docs 完整说明） |
|------|------|
| **引用完整性率** | 期望通过的主张中，确实带上了完整可解析引用锚点的占比 |
| **表面一致率** | 期望通过的主张中，最终状态为 pass 的占比（好主张没有被误拒） |
| **拒答正确率** | 期望拒答的主张中，校验门确实给出 refuse 的占比（**防幻觉关键指标**） |

对照实验（GroundedRAG vs 裸 prompt RAG）说明见 [docs/comparison.md](docs/comparison.md)。

## 架构

```
┌────────────────────────── pipeline.py ─────────────────────────┐
│  编排：检索 → 规则匹配 → 受约束生成 → 声明解析 → 校验门 → 降级/拒答 │
├───────────────┬───────────────────────┬────────────────────────┤
│   retriever/   │      guardrail/ ★      │          llm/          │
│  BM25 检索      │  规则引擎 RuleDecision │   策略基类 + 多模型降级  │
│  实体增强       │  证据溯源 EvidenceId   │   OpenAI 兼容客户端     │
│  查询扩展       │  声明解析 AnswerClaim  │   模板回退（无据拒答文案）│
│  (词典可注入)    │  ★声明级校验门 verifier│                        │
└───────────────┴───────────────────────┴────────────────────────┘
```

- `retriever/`：BM25（jieba 分词；缺失时回退 **CJK char-bigram**，绝不回退 `split()`）+
  实体增强 + 查询扩展，领域同义词词典可注入（业务方可闭源自己的词典）。
- `guardrail/`（★ 核心差异化）：纯 Python 规则引擎（无 ORM，可替换业务侧数据库 Provider）、
  EvidenceId 溯源、AnswerClaim 解析、verifier 校验门（引用完整性 / 表面一致 / 冲突裁定 / 充分性）。
- `llm/`：`LLMService` 策略基类 + OpenAI 兼容原生 HTTP 客户端（零 SDK）+ 模板回退 + 主备降级。
- `eval/`：内置评测集跑分，输出三项指标。
- `pipeline.py`：编排流水线（`Pipeline.build_from_json(...).ask(...)`）。

## 目录

```
grounded-rag/
├── src/groundedrag/
│   ├── retriever/      # bm25.py + retriever.py
│   ├── guardrail/      # models/engine/provider/evidence/claims/verifier ★
│   ├── llm/            # base/openai_compat/template/failover
│   ├── eval/runner.py  # 三项指标评测
│   └── pipeline.py     # 可信问答编排
├── examples/           # 合成种子数据 + demo.py + app.py
├── tests/              # pytest 单测
├── tools/leak_scan.py  # 开源合规自检（扫描仓库内是否混入受限/私有内容）
└── docs/               # 指标口径 / 对照实验 / 架构设计
```

## Roadmap

### v1.1（2026 Q4）— 语义档增强

- [ ] 接入 NLI 模型（deberta-mnli / bge-reranker）实现真实蕴含判定
- [ ] 关系型主张（否定/比较/因果）从"拒答"升级为"语义校验通过"
- [ ] 表面一致率 → 语义支持率指标升级
- [ ] 可选语义档开关，兼容确定性档（医疗等高风险场景保留保守拒答策略）

### v1.2（2027 Q1）— OncoKG 图谱证据链

- [ ] 独立 `GraphProvider` 接口 + 图数据标准格式（节点/边 schema）
- [ ] 知识图谱路径推理增强证据链路（实体-关系-实体多跳溯源）
- [ ] 证据溯源从文档级升级为实体-关系级（`EvidenceId` 扩展 `entity_path` 字段）
- [ ] 图谱可视化工具（证据链路交互式探索）

### v1.3（2027 Q2）— 多 LLM 对齐验证

- [ ] 多模型并行生成 + 主张级 consensus voting（3 模型各自生成 → 逐条投票 → 分歧标注）
- [ ] 自洽性校验门（Self-Consistency Verifier）：同一问题多次采样 → 高方差主张降级
- [ ] LLM-as-judge 可选增强（GPT-4 / Claude 作第三方裁判）

### v2.0（2027 Q3）— 多领域泛化

- [ ] 金融/法律/教育垂直领域适配（规则 DSL + 领域无关引擎）
- [ ] 社区贡献的领域规则库生态（公开规则仓库 + 贡献者认证）
- [ ] 领域插件系统（一键切换医疗/金融/法律规则集 + 词典 + 评测集）
- [ ] 企业版：私有规则库托管 + 审计日志 + SSO

### 长期愿景

成为高风险垂直领域的「**可信 RAG 事实标准**」—— 让 LLM 在医疗/金融/法律等需要可溯源、
可审计的场景下做到「**有据可依，无据可拒**」；沉淀一套"约束生成 + 声明级校验"的工程范式，
推动 RAG 从"检索拼接"向"可信交付"演进。

我们相信：**在高风险场景下，一个诚实拒答的 AI 比一个流畅编造的 AI 更有价值。**

## 参考应用

GroundedRAG 由医疗数据平台 **onco-hub** 的生产实践演化而来（47,000+ 条医疗数据 + CSCO 指南规则），
参考应用线上运行见 https://onco.ylkang.cn/（旧问答管线，未含声明级校验门）；本框架将校验门机制
独立开源，使用合成示例数据（`examples/`），供任何垂直领域复用。

## 合规与第三方依赖

- 核心运行时仅依赖 **jieba**（MIT）；Gradio 仅存在于 `demo` extra。
- 第三方许可证清单见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- 贡献指南见 [CONTRIBUTING.md](CONTRIBUTING.md)（含 7 点合规清单 + 红线表）。

## 社区

- **GitHub 镜像**（规划中）: https://github.com/groundedrag/groundedrag
- **Gitee 主仓库**: https://gitee.com/miniclaw27/grounded-rag
- **反馈与讨论**: [Gitee Issues](https://gitee.com/miniclaw27/grounded-rag/issues)
- **贡献者**: 见 [CONTRIBUTING.md](CONTRIBUTING.md)

欢迎贡献代码、规则库、评测用例或新领域适配！特别欢迎：
- 金融/法律领域的规则引擎适配
- 多语言分词器支持（英文/日文/韩文）
- 语义档 NLI 模型集成
- 企业级部署案例

## License

MIT © 2026 GroundedRAG Contributors。见 [LICENSE](LICENSE)。
