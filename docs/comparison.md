# 对照实验：GroundedRAG vs 裸 prompt RAG

这份对照回答评审最关心的实证问题：**校验门相比"检索增强后直接让 LLM 回答"，
到底可复现地拦掉了多少幻觉？**

## 方法论（控制变量）

两条流水线喂**同一批召回文档与同一组问题**，只差在是否经过 GroundedRAG 的约束与校验：

| 维度 | 裸 prompt RAG（对照组） | GroundedRAG（实验组） |
|------|------|------|
| 召回 | 同一 BM25 top-k | 同一 BM25 top-k + 规则匹配 |
| 提示词 | "根据资料直接回答"（无约束） | 要求逐条主张 + `[证据n]`/`[规则n]` 锚点 |
| 校验 | 无 | 声明级校验门（引用完整性/表面一致/冲突裁定/充分性） |
| 输出门 | 模型说什么就是什么 | PASS 原样 / ANNOTATE 标注 / REFUSE 拒答或删除 |
| 指标口径 | 整段"会/不会拒答" | 主张级三指标（见 [metrics.md](metrics.md)） |

**评测不走真实 LLM**：实验组的候选主张由 `examples/eval_set.jsonl` 显式注入
（既含正确输出也含幻觉变体），被测对象是确定性校验链路；对照组用
`groundedrag.eval.runner.baseline_questions()` 导出同源问题清单后，在任意
裸模型上逐题问"会/不会拒答"，再用 `score_baseline()` 打分。这样实验组离线可复现，
对照组只需一个裸模型，两侧共享同一批"幻觉高危题"。

## 预期结论（示意数值）

内置评测集构造了 19 组 24 条主张，其中约半数反例为"裸 RAG 极易直出"的幻觉高危题
（数值虚构、证据侧否定翻转、禁忌联用等）。实测（本仓库自检）实验组三指标全绿：

| 指标 | 裸 prompt RAG | GroundedRAG |
|------|------|------|
| 拒答正确率 `refusal_expected_caught` | **≈ 0**（有资料就直出，从不拒答） | **1.0**（幻觉高危题全部拦截） |
| 误拒 `false_refusal` | —（它从不拒答，谈不上误拒） | 低（好主张不被误伤） |
| 引用完整性率 | —（无锚点概念） | 1.0 |
| 表面一致率 | —（无主张粒度） | 1.0 |

> ⚠️ 数值随评测集演化而变化，README/本文件的表格需与 `python -m groundedrag.eval.runner --json`
> 的实际输出同步更新；勿在评审材料里用过期的声称数值。

一句话叙事：**裸 RAG 在"这份资料里没有的事"上会面不改色地编；GroundedRAG 在同一批
资料上，把这类回答拆成一条条可核查主张，逐条过确定性校验门后拒掉——整段拒答用
就医指引文案兜底，绝不把幻觉当答案交付。**

## 如何复现

```bash
# 1) 实验组：确定性校验链路（无需 LLM）
python -m groundedrag.eval.runner --json

# 2) 对照组：导出问题清单 → 在你的裸模型上逐题问"会/不会拒答"
python - <<'PY'
import json
from groundedrag.eval.runner import load_eval_set, baseline_questions
for q in baseline_questions(load_eval_set("examples/eval_set.jsonl")):
    print(q["index"], json.dumps({"question": q["question"]}, ensure_ascii=False))
PY

# 3) 把裸模型"会/不会拒答"的 bool 按 question 键写回 responses → score_baseline
```

跑真实演示对照（同一问题两条流水线并排输出），打开 Gradio 界面的**裸 RAG 对照开关**：

```bash
pip install -e ".[demo]"
python examples/app.py     # 切换「GroundedRAG 可信模式 / 裸 RAG 对照」
```

## 边界与诚实声明

- 表面一致率 ≠ "真实支持率"，关系型主张以拒答正确率为准（见 [metrics.md](metrics.md)）。
- 本对照证明的是**校验机制的拦截有效性**，不是某个模型在医学问题上的总准确率；
  语义档（NLI/更强模型二次核验）关闭时，方向性主张一律拒答或标注，不冒充验证过。
- `examples/` 数据为自研合成示例，不构成真实诊疗建议，不用于训练或医学结论。
