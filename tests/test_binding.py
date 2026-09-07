"""确定性锚点绑定器（Pipeline._bind_anchors）单元测试。"""

from __future__ import annotations

from groundedrag.guardrail.claims import AnswerClaim
from groundedrag.guardrail.evidence import EvidenceId
from groundedrag.guardrail.models import RuleDecision, RuleRecommendation
from groundedrag.pipeline import Pipeline


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ev(eid: str, span: str) -> EvidenceId:
    return EvidenceId(
        evidence_id=eid,
        doc_id=f"doc-{eid}",
        source_type="guideline",
        source_version="2026",
        grade="A",
        updated_at="2026-01-01",
        text_span=span,
    )


def _rule(rid: str, *, cancer: str = "", line: str = "", bio: str = "",
          plans: list[str] | None = None) -> RuleDecision:
    recs = [RuleRecommendation(plan_name=p) for p in (plans or [])]
    return RuleDecision(
        rule_id=rid,
        cancer_type=cancer or None,
        treatment_line=line or None,
        biomarker=bio or None,
        recommendations=recs,
    )


# ---------------------------------------------------------------------------
# 证据绑定
# ---------------------------------------------------------------------------

class TestBindEvidence:
    def test_exact_text_match_binds(self):
        """主张原文出现在证据 span 中 → 绑定。"""
        c = AnswerClaim(text="奥希替尼推荐剂量 80mg")
        ev = _ev("ev1", "奥希替尼推荐剂量 80mg，每日口服一次")
        result = Pipeline._bind_anchors([c], [ev], ["ev1"], [])
        assert len(result) == 1
        assert result[0].evidence_refs == ["ev1"]

    def test_number_overlap_binds(self):
        """主张含数字 80mg，证据也含 80mg → 绑定。"""
        c = AnswerClaim(text="奥希替尼剂量为80mg")
        ev = _ev("ev1", "EGFR突变肺癌一线推荐奥希替尼 80mg 每日一次")
        result = Pipeline._bind_anchors([c], [ev], ["ev1"], [])
        assert len(result) == 1
        assert result[0].evidence_refs == ["ev1"]

    def test_no_match_discards(self):
        """主张与所有证据均无交集 → 丢弃。"""
        c = AnswerClaim(text="某某药物推荐剂量 500mg")
        ev = _ev("ev1", "奥希替尼推荐剂量 80mg")
        result = Pipeline._bind_anchors([c], [ev], ["ev1"], [])
        assert len(result) == 0

    def test_already_anchored_preserved(self):
        """已有锚点的主张不参与绑定，直接保留。"""
        c = AnswerClaim(text="奥希替尼 80mg", evidence_refs=["ev1"])
        result = Pipeline._bind_anchors([c], [], [], [])
        assert len(result) == 1
        assert result[0].evidence_refs == ["ev1"]

    def test_multiple_evidence_binds_best(self):
        """多条证据可匹配时，绑定得分最高的那条。"""
        c = AnswerClaim(text="奥希替尼推荐剂量 80mg")
        ev1 = _ev("ev1", "奥希替尼用于 EGFR 突变")
        ev2 = _ev("ev2", "奥希替尼推荐剂量 80mg，每日一次口服")
        result = Pipeline._bind_anchors([c], [ev1, ev2], ["ev1", "ev2"], [])
        assert len(result) == 1
        assert result[0].evidence_refs == ["ev2"]  # ev2 得分更高（exact match +5）


# ---------------------------------------------------------------------------
# 规则绑定
# ---------------------------------------------------------------------------

class TestBindRule:
    def test_rule_bind_by_fields(self):
        """主张含规则的 cancer_type + biomarker → 绑定。"""
        c = AnswerClaim(text="EGFR突变肺癌一线推荐奥希替尼")
        r = _rule("r1", cancer="肺癌", line="一线", bio="EGFR", plans=["奥希替尼"])
        result = Pipeline._bind_anchors([c], [], [], [r])
        assert len(result) == 1
        assert result[0].rule_refs == ["r1"]

    def test_rule_bind_by_plan_name(self):
        """主张含规则推荐方案名 → 绑定。"""
        c = AnswerClaim(text="推荐方案：奥希替尼")
        r = _rule("r1", cancer="肺癌", plans=["奥希替尼"])
        result = Pipeline._bind_anchors([c], [], [], [r])
        assert len(result) == 1
        assert result[0].rule_refs == ["r1"]

    def test_enum_prefix_stripped(self):
        """枚举前缀 '1. ' 被剥离，不干扰验证。"""
        c = AnswerClaim(text="1. 奥希替尼推荐剂量 80mg")
        ev = _ev("ev1", "奥希替尼推荐剂量 80mg，每日口服一次")
        result = Pipeline._bind_anchors([c], [ev], ["ev1"], [])
        assert len(result) == 1
        assert result[0].evidence_refs == ["ev1"]
        # 枚举前缀已剥离
        assert result[0].text == "奥希替尼推荐剂量 80mg"

    def test_rule_no_match_discards(self):
        """主张与规则字段无交集 → 丢弃。"""
        c = AnswerClaim(text="某某药物推荐剂量 500mg")
        r = _rule("r1", cancer="肺癌", plans=["奥希替尼"])
        result = Pipeline._bind_anchors([c], [], [], [r])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 组合：证据 + 规则
# ---------------------------------------------------------------------------

class TestBindCombined:
    def test_evidence_bind_preferred_over_rule(self):
        """同时有证据和规则时，优先绑定证据。"""
        c = AnswerClaim(text="奥希替尼推荐剂量 80mg")
        ev = _ev("ev1", "奥希替尼推荐剂量 80mg")
        r = _rule("r1", cancer="肺癌", plans=["奥希替尼"])
        result = Pipeline._bind_anchors([c], [ev], ["ev1"], [r])
        assert len(result) == 1
        # 证据绑定优先（_bind_anchors 先处理证据，再处理规则）
        assert result[0].evidence_refs == ["ev1"]

    def test_empty_input(self):
        """空主张列表 → 返回空。"""
        assert Pipeline._bind_anchors([], [], [], []) == []

    def test_mixed_anchored_and_unanchored(self):
        """混合有锚/无锚主张：有锚的保留，无锚的尝试绑定。"""
        c1 = AnswerClaim(text="奥希替尼 80mg", evidence_refs=["ev1"])
        c2 = AnswerClaim(text="奥希替尼推荐剂量 80mg")
        ev = _ev("ev1", "奥希替尼推荐剂量 80mg")
        result = Pipeline._bind_anchors([c1, c2], [ev], ["ev1"], [])
        assert len(result) == 2
        assert result[0].evidence_refs == ["ev1"]
        assert result[1].evidence_refs == ["ev1"]
