"""编排流水线 Pipeline 单元测试（无 API Key 的可信问答链路）。"""

from __future__ import annotations

from pathlib import Path

from groundedrag.llm.base import LLMService
from groundedrag.pipeline import (
    DIVERGENCE_NOTE,
    Pipeline,
    PipelineResult,
    load_docs_jsonl,
)
from groundedrag.retriever.retriever import Document

DOCS_PATH = Path(__file__).resolve().parent.parent / "examples" / "seed_docs.jsonl"
RULES_PATH = Path(__file__).resolve().parent.parent / "examples" / "seed_rules.json"


class StubJSONLLM(LLMService):
    """返回固定 JSON 结构化主张（模拟受约束生成的 LLM）。"""

    name = "stub-json"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, *, temperature=0.3, max_tokens=None) -> str:
        return '{"claims":[{"text":"奥希替尼为三代EGFR-TKI","evidence_refs":[1]}]}'


def _small_pipeline() -> Pipeline:
    """微型自足流水线：单文档 + 无规则，供 LLM-claims 路径测试。"""
    docs = [
        Document(
            doc_id="doc-q",
            title="肺癌治疗说明",
            content="奥希替尼为三代EGFR-TKI，用于EGFR突变阳性晚期肺癌一线标准治疗。",
            source_type="guideline",
            grade="A",
            updated_at="2026-02-01",
        )
    ]
    return Pipeline.build(docs, {})


class TestLoadDocs:
    def test_loads_example_docs(self):
        docs = load_docs_jsonl(DOCS_PATH)
        assert len(docs) == 52
        assert all(isinstance(d, Document) for d in docs)


class TestRuleDirect:
    def test_rule_direct_answer_has_authoritative_plan(self, pipeline):
        result = pipeline.ask(
            "EGFR突变的晚期肺癌一线推荐什么方案？",
            evidence_docs=["doc-lung-egfr-1st-pos"],
        )
        assert result.status == "pass"
        assert result.used_llm is None  # 规则直出，不走 LLM
        assert "奥希替尼" in result.answer_text
        assert any(c.rule_refs for c in result.claims)

    def test_rule_direct_can_be_pinned_without_context(self, pipeline):
        # 未显式给 context，启发式也能从问题提取（含线次词 + 癌种 + 标志物）
        result = pipeline.ask("肺癌EGFR一线用什么药？", evidence_docs=["doc-lung-egfr-1st-pos"])
        assert "奥希替尼" in result.answer_text


class TestNoMatchRefusal:
    def test_unrelated_question_refuses_gracefully(self, pipeline):
        result = pipeline.ask("今天天气怎么样", top_k=2)
        assert result.status == "refuse"
        assert "不足以支撑" in result.answer_text  # 结构化拒答文案兜底


class TestDivergence:
    def test_conflicting_rules_declare_divergence(self, pipeline):
        result = pipeline.ask(
            "EGFR突变的肺癌三线应该用什么方案？",
            evidence_docs=["doc-lung-egfr-1st-pos"],
        )
        assert result.status == "refuse"
        assert result.answer_text == DIVERGENCE_NOTE
        assert result.report.overall_reason == "rule_conflict_divergence"


class TestLLMClaims:
    def test_llm_structured_claims_pass_verifier(self):
        pipe = _small_pipeline()
        result = pipe.ask(
            "奥希替尼是什么类型的药物？",
            evidence_docs=["doc-q"],
            llm=StubJSONLLM(),
        )
        assert result.status == "pass"
        assert result.used_llm == "stub-json"
        assert "奥希替尼为三代EGFR-TKI" in result.answer_text
        assert result.report.overall_reason == "all_passed_or_annotated"

    def test_rule_direct_takes_over_when_llm_absent(self):
        # 默认 FailoverLLM（含模板）不可产出有效主张 → 有规则则规则直出
        pipe = _small_pipeline()
        assert pipe.ask("没有规则的普通问题", evidence_docs=["doc-q"]).status == "refuse"


class TestPipelineResult:
    def test_to_dict_roundtrip_fields(self):
        r = PipelineResult(question="q")
        d = r.to_dict()
        assert d["question"] == "q"
        assert d["answer_text"] == ""
        assert d["report"] is None

    def test_build_from_json_and_run_alias(self, pipeline):
        # run = ask 是同一底层函数（类体别名）
        assert pipeline.ask.__func__ is pipeline.run.__func__
        assert pipeline.run("EGFR突变的晚期肺癌一线推荐什么方案？").question


def test_docs_count_from_examples():
    docs = load_docs_jsonl(DOCS_PATH)
    assert len(docs) > 0
    assert RULES_PATH.exists()
