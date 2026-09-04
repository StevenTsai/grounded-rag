"""GroundedRAG —— 轻量开源 RAG 框架。

以「声明级校验门」可复现地减少无证据输出，让 LLM 回答「有据可依、无据可拒」。

组成：
- ``retriever``  检索层：BM25 + 实体增强 + 查询扩展（领域词典/同义词可注入）
- ``guardrail``  约束生成层（★核心）：规则引擎 + 证据溯源(EvidenceId) +
                 声明解析(AnswerClaim) + 声明级校验门(verifier)
- ``llm``        多模型层：策略基类 + OpenAI 兼容客户端 + 模板回退 + 主备降级
- ``eval``       评测集跑分：引用完整性率 / 表面一致率 / 拒答正确率
- ``pipeline``   可信问答编排流水线

示例：:

    from groundedrag.pipeline import Pipeline

    pipe = Pipeline.build_from_json(
        "examples/seed_docs.jsonl", "examples/seed_rules.json"
    )
    result = pipe.ask("EGFR 突变的晚期肺癌一线推荐什么方案？")
    print(result.answer_text)
"""

__version__ = "1.0.0"
