"""规则引擎 GuidelineEngine / 条件求值 / provider 单元测试。"""

from __future__ import annotations

from pathlib import Path

from groundedrag.guardrail.engine import GuidelineEngine, _is_match
from groundedrag.guardrail.models import RuleRecommendation
from groundedrag.guardrail.provider import DictRuleProvider

REPO_ROOT = Path(__file__).resolve().parent.parent

# 两条同域、同癌种/线次/标志物的规则 + 一条耐药规则
_PAYLOAD = {
    "pathway_rules": [
        {
            "rule_id": "r1",
            "name": "肺癌EGFR一线",
            "cancer_type": "肺癌",
            "treatment_line": "一线",
            "biomarker": "EGFR",
            "conditions": [
                {"field": "cancer_type", "op": "eq", "value": "肺癌"},
                {"field": "biomarker", "op": "eq", "value": "EGFR"},
                {"field": "treatment_line", "op": "eq", "value": "一线"},
            ],
            "recommendations": [
                {"plan_name": "奥希替尼", "grade": "A"},
            ],
        },
        {
            "rule_id": "r2",
            "name": "肺癌ALK一线",
            "cancer_type": "肺癌",
            "treatment_line": "一线",
            "biomarker": "ALK",
            "conditions": [
                {"field": "cancer_type", "op": "eq", "value": "肺癌"},
                {"field": "biomarker", "op": "eq", "value": "ALK"},
            ],
            "recommendations": [
                {"plan_name": "阿来替尼", "grade": "A"},
            ],
        },
        {
            "rule_id": "r3",
            "name": "多条件 all/in 规则",
            "conditions": [
                {"field": "excluded", "op": "neq", "value": "EGFR"},
                {"field": "history", "op": "all", "value": ["化疗", "免疫"]},
            ],
            "recommendations": [{"plan_name": "示例方案", "grade": "D"}],
        },
    ],
    "resistance_rules": [
        {
            "rule_id": "rr1",
            "gene": "T790M",
            "conditions": [{"field": "biomarker", "op": "eq", "value": "T790M"}],
            "recommendations": [{"plan_name": "奥希替尼", "grade": "A"}],
        },
    ],
}


def _engine() -> GuidelineEngine:
    p = DictRuleProvider(_PAYLOAD)
    return GuidelineEngine(
        pathway_rules=p.load_pathway_rules(),
        resistance_rules=p.load_resistance_rules(),
    )


class TestConditionOps:
    def test_eq(self):
        assert _is_match("eq", "肺癌", "肺癌")
        assert not _is_match("eq", "肺癌", "胃癌")

    def test_eq_case_insensitive(self):
        assert _is_match("eq", "EGFR", "egfr")

    def test_neq(self):
        assert _is_match("neq", "EGFR", "ALK")
        assert not _is_match("neq", "EGFR", "EGFR")

    def test_in_list_membership(self):
        assert _is_match("in", "肺癌", ["肺癌", "胃癌"])
        assert not _is_match("in", "肺癌", ["胃癌"])

    def test_contains_list(self):
        assert _is_match("contains", "已接受含铂化疗", ["化疗"])

    def test_all(self):
        assert _is_match("all", "既往化疗与免疫治疗均失败", ["化疗", "免疫"])
        assert not _is_match("all", "仅化疗", ["化疗", "免疫"])

    def test_none_context_value(self):
        assert not _is_match("eq", None, "肺癌")


class TestGuidelineEngine:
    def test_match_single_domain(self):
        engine = _engine()
        hits = engine.match({"cancer_type": "肺癌", "treatment_line": "一线", "biomarker": "ALK"})
        assert [d.rule_id for d in hits] == ["r2"]

    def test_match_all_ops_rule(self):
        engine = _engine()
        hits = engine.match({"excluded": "ALK", "history": "既往化疗与免疫失败"})
        assert "r3" in [d.rule_id for d in hits]

    def test_no_match(self):
        engine = _engine()
        assert engine.match({"cancer_type": "胃癌"}) == []

    def test_resistance_matches_gene(self):
        engine = _engine()
        hits = engine.match({"biomarker": "T790M"})
        assert [d.rule_id for d in hits] == ["rr1"]
        assert hits[0].biomarker == "T790M"

    def test_match_resistance_separate(self):
        engine = _engine()
        assert engine.match_resistance({"biomarker": "T790M"})
        assert engine.match_resistance({"biomarker": "ALK"}) == []

    def test_extract_context_from_question(self):
        engine = _engine()
        ctx = engine.extract_context("EGFR突变的晚期肺癌一线推荐什么方案？")
        assert ctx.get("cancer_type") == "肺癌"
        assert ctx.get("biomarker") == "EGFR"
        assert ctx.get("treatment_line") == "一线"

    def test_extract_context_cancer_only_when_biomarker_absent(self):
        # 未提供 synonym_map 且候选仅覆盖首条规则时，启发式保守不硬配 biomarker
        engine = _engine()
        ctx = engine.extract_context("晚期ALK阳性肺癌用什么？")
        assert ctx.get("cancer_type") == "肺癌"
        assert "biomarker" not in ctx

    def test_sort_by_grade(self):
        low = RuleDecision_factory("low", "C")
        high = RuleDecision_factory("high", "A")
        assert GuidelineEngine.sort_by_grade([low, high])[0].rule_id == "high"


def RuleDecision_factory(rule_id: str, grade: str):
    from groundedrag.guardrail.models import RuleDecision

    return RuleDecision(
        rule_id=rule_id,
        recommendations=[RuleRecommendation(plan_name="P", grade=grade)],
    )


class TestProviders:
    def test_dict_provider_loads(self):
        p = DictRuleProvider(_PAYLOAD)
        assert len(p.load_pathway_rules()) == 3
        assert len(p.load_resistance_rules()) == 1

    def test_json_provider_loads_examples(self):
        from groundedrag.guardrail.models import Rule
        from groundedrag.guardrail.provider import JsonRuleProvider

        provider = JsonRuleProvider(REPO_ROOT / "examples" / "seed_rules.json")
        rules = provider.load_pathway_rules()
        assert rules and all(isinstance(r, Rule) for r in rules)
        assert any(r.rule_id == "r-lung-egfr-1st" for r in rules)


class TestEngineInstantiatesEmpty:
    def test_no_rules_ok(self):
        engine = GuidelineEngine()
        assert engine.match({"cancer_type": "肺癌"}) == []
        assert engine.match_resistance({}) == []
