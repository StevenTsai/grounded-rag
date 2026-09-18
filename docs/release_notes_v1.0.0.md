# GroundedRAG v1.0.0 Release Notes

**发布日期**：2026-09-18 ｜ **协议**：MIT ｜ **Python**：3.10+

一句话：给 RAG 加一道**声明级确定性校验门**——把回答拆成逐条可核查主张，绑定证据/规则锚点后纯逻辑校验，让 LLM 回答「有据可依，无据可拒」。

## 亮点

- **确定性校验**（零 LLM）：引用完整性 → 表面要素一致性 → 规则冲突裁定 → 证据充分性，同一输入永远同一输出
- **三态判定**：PASS / ANNOTATE / REFUSE，替代一刀切的 pass/fail
- **规则直出**：命中权威规则时跳过 LLM，断网可用、零幻觉风险
- **反跨证据拼装**：主张中的数字与实体必须来自同源证据
- **领域无关**：`field/op/value` 条件系统 + 可注入同义词词典，医疗/金融/法律均可适配
- **可复现评测**：verify 三指标 1.0（19 组 24 条主张）、binding 绑定准确率 1.0

## 安装

```bash
# 固定 tag 引用（PyPI 尚未发布）
pip install "groundedrag @ git+https://gitee.com/miniclaw27/grounded-rag.git@v1.0.0"
```

## 快速开始

```bash
groundedrag ask "EGFR突变肺癌一线推荐什么方案？"   # 无 API Key 也可运行
python -m groundedrag.eval --json                  # 确定性校验门跑分（零 LLM）
python examples/integrate_verifier.py              # 已有 RAG「只挂校验门」示例
```

## 集成方式

三种接入深度（详见 [`docs/integration.md`](integration.md)）：

1. 整条流水线：`Pipeline.build_from_json(...).ask(...)`
2. 只挂校验门：`Verifier().verify(claims, EvidenceRegistry(evidences))`
3. 单组件复用：`Retriever` / `GuidelineEngine` / `FailoverLLM`

## 质量

| 指标 | 值 |
|------|-----|
| 单元测试 | 247（全部通过） |
| 测试覆盖率 | 95% |
| 静态检查 | ruff 通过（CI 门禁） |
| CI | GitHub Actions：Python 3.10 / 3.11 / 3.12 全绿 |

## 引用

```bibtex
@software{groundedrag2026,
  title  = {GroundedRAG: 声明级确定性校验的 RAG 可验证交付层},
  author = {蔡德勋 and 杨章敏 and 杨章伦},
  year   = {2026},
  version = {1.0.0},
  license = {MIT},
  url    = {https://gitee.com/miniclaw27/grounded-rag}
}
```

或使用仓库根目录的 [`CITATION.cff`](../CITATION.cff)。

## 完整变更

见 [`CHANGELOG.md`](../CHANGELOG.md)。
