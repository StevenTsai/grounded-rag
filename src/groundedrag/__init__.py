"""GroundedRAG —— 轻量开源 RAG 框架。

以「声明级校验门」可复现地减少无证据输出，让 LLM 回答「有据可依、无据可拒」。

快速开始：:

    from groundedrag import Pipeline

    pipe = Pipeline.build_from_json(
        "examples/seed_docs.jsonl", "examples/seed_rules.json"
    )
    result = pipe.ask("EGFR 突变的晚期肺癌一线推荐什么方案？")
    print(result.answer_text)
"""

__version__ = "1.0.0"

# 便捷导入：from groundedrag import Pipeline, PASS, REFUSE, ...
from groundedrag.guardrail import ANNOTATE, PASS, REFUSE
from groundedrag.pipeline import Pipeline

__all__ = [
    "ANNOTATE",
    "PASS",
    "Pipeline",
    "REFUSE",
    "__version__",
]
