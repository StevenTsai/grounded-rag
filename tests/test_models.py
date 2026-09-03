"""guardrail.models 纯数据模型单元测试。"""

from __future__ import annotations

from groundedrag.guardrail.models import (
    GRADE_ORDER,
    ResistanceRule,
    Rule,
    RuleCondition,
    RuleDecision,
    RuleRecommendation,
    normalize_line,
    plan_components,
)


class TestNormalizeLine:
    def test_aliases(self):
        assert normalize_line("一线") == "一线"
        assert normalize_line("1线") == "一线"
        assert normalize_line("2线") == "二线"
        assert normalize_line("3线") == "三线"
        assert normalize_line("first") == "一线"
        assert normalize_line("后线") == "后线"

    def test_unknown_kept_trimmed(self):
        assert normalize_line("新辅助") == "新辅助"

    def test_none(self):
        assert normalize_line(None) is None


class TestPlanComponents:
    def test_single_plan(self):
        assert plan_components("奥希替尼") == ["奥希替尼"]

    def test_combination_splits_and_lowercases(self):
        parts = plan_components("A+贝伐珠单抗")
        assert parts == ["a", "贝伐珠单抗"]

    def test_various_separators(self):
        assert "紫杉醇" in plan_components("紫杉醇+卡铂")
        assert "卡铂" in plan_components("紫杉醇，卡铂")
        assert "奥希替尼" in plan_components("奥希替尼/吉非替尼")[:1]

    def test_empty(self):
        assert plan_components("") == []


class TestRuleCondition:
    def test_roundtrip(self):
        c = RuleCondition("cancer_type", "eq", "肺癌")
        assert RuleCondition.from_dict(c.to_dict()) == c

    def test_str_list_value(self):
        assert "A / B" in str(RuleCondition("x", "in", ["A", "B"]))


class TestRule:
    def _rule(self, **kw) -> Rule:
        return Rule(
            rule_id="r1",
            name="肺癌一线",
            cancer_type="肺癌",
            treatment_line="一线",
            biomarker="EGFR",
            conditions=[RuleCondition("cancer_type", "eq", "肺癌")],
            recommendations=[
                RuleRecommendation(plan_name="奥希替尼", grade="A"),
                RuleRecommendation(plan_name="奥希替尼+化疗", grade="A"),
            ],
            **kw,
        )

    def test_from_dict_id_fallback_and_line_normalize(self):
        rule = Rule.from_dict({"id": "rX", "treatment_line": "2线"})
        assert rule.rule_id == "rX"
        assert rule.treatment_line == "二线"

    def test_top_plans_dedup_by_component(self):
        top = self._rule().top_plans
        assert "奥希替尼" in top
        assert "化疗" in top
        assert top.count("奥希替尼") == 1

    def test_max_grade(self):
        assert self._rule().max_grade == "A"

    def test_normalized_line_property(self):
        assert self._rule().normalized_line == "一线"

    def test_roundtrip_to_from_dict(self):
        rule = self._rule()
        assert Rule.from_dict(rule.to_dict()) == rule

    def test_updated_at_none_from_dict(self):
        r = Rule.from_dict({"rule_id": "x"})
        assert r.updated_at is None


class TestRuleDecision:
    def _decision(self, plans=("奥希替尼", "化疗"), grade="A") -> RuleDecision:
        return RuleDecision(
            rule_id="d1",
            recommendations=[
                RuleRecommendation(plan_name=p, grade=grade) for p in plans
            ],
            cancer_type="肺癌",
            treatment_line="一线",
            biomarker="EGFR",
        )

    def test_plan_components_merge(self):
        comps = self._decision(["奥希替尼", "贝伐珠单抗"]).plan_components
        assert set(comps) == {"奥希替尼", "贝伐珠单抗"}

    def test_max_grade(self):
        d = RuleDecision(
            rule_id="d",
            recommendations=[
                RuleRecommendation("A方案", "C"),
                RuleRecommendation("B方案", "B"),
            ],
        )
        assert d.max_grade == "B"

    def test_to_dict_contains_max_grade(self):
        d = self._decision()
        assert d.to_dict()["max_grade"] == "A"
        assert d.to_dict()["rule_id"] == "d1"


class TestResistanceRule:
    def test_from_dict_maps_gene(self):
        r = ResistanceRule.from_dict({"rule_id": "rr1", "gene": "T790M"})
        assert r.gene == "T790M"

    def test_roundtrip(self):
        r = ResistanceRule(rule_id="rr1", gene="T790M")
        restored = ResistanceRule.from_dict(r.to_dict())
        assert restored == r


def test_grade_order_consistency():
    assert GRADE_ORDER["A"] > GRADE_ORDER["B"] > GRADE_ORDER["C"] > GRADE_ORDER["D"]
