"""★声明级校验门 verifier 单元测试（防幻觉的核心逻辑）。

覆盖引用完整性 / 语义档 / 关系型拒答 / 表面一致性 / 数值不符 /
证据侧否定翻转 / 规则冲突裁定 / 充分性门槛 / 整组状态。
"""

from __future__ import annotations

from groundedrag.guardrail.claims import AnswerClaim
from groundedrag.guardrail.evidence import EvidenceId, EvidenceRegistry
from groundedrag.guardrail.models import RuleDecision, RuleRecommendation
from groundedrag.guardrail.verifier import (
    ANNOTATE,
    PASS,
    REFUSE,
    Verifier,
    VerifierConfig,
)


def ev(eid: str, span: str, grade: str = "A", updated_at: str = "2026-02-01") -> EvidenceId:
    return EvidenceId(
        evidence_id=eid,
        doc_id=eid,
        source_type="guideline",
        source_version="CSCO-LUNG-2026",
        grade=grade,
        updated_at=updated_at,
        text_span=span,
    )


def claim(text: str, **kw) -> AnswerClaim:
    kw.setdefault("critical", False)
    return AnswerClaim(text=text, **kw)


def decision(
    rule_id: str, plans, grade: str = "A", updated_at: str = "2026-02-01",
    cancer_type: str = "肺癌", treatment_line: str = "三线", biomarker: str = "EGFR",
) -> RuleDecision:
    return RuleDecision(
        rule_id=rule_id,
        rule_name=rule_id,
        recommendations=[
            RuleRecommendation(plan_name=p, grade=grade, source_version="SYN-2026")
            for p in plans
        ],
        source_version="SYN-2026",
        updated_at=updated_at,
        cancer_type=cancer_type,
        treatment_line=treatment_line,
        biomarker=biomarker,
    )


POS_SPAN = "奥希替尼为三代EGFR-TKI，用于EGFR突变阳性晚期肺癌一线标准治疗，推荐剂量80mg，每日口服一次。"


class TestCitationGate:
    def test_no_refs_critical_refuses(self):
        verifier = Verifier()
        c = claim("推荐奥希替尼用于一线治疗", critical=True)
        rep = verifier.verify([c], EvidenceRegistry([ev("e1", POS_SPAN)]))
        assert rep.verdicts[0].status == REFUSE
        assert rep.verdicts[0].reason == "citation_incomplete"

    def test_no_refs_noncritical_annotates(self):
        verifier = Verifier()
        c = claim("奥希替尼属于口服小分子靶向药")  # 无关键信号
        rep = verifier.verify([c], EvidenceRegistry())
        assert rep.verdicts[0].status == ANNOTATE
        assert rep.verdicts[0].reason == "citation_incomplete"

    def test_unresolvable_evidence_ref_incomplete(self):
        verifier = Verifier()
        c = claim("推荐奥希替尼[证据1]", evidence_refs=["ghost"], critical=True)
        rep = verifier.verify([c], EvidenceRegistry())
        assert rep.verdicts[0].reason == "citation_incomplete"
        assert rep.verdicts[0].status == REFUSE


class TestRelationalGate:
    def test_negation_refuses_even_with_evidence(self):
        verifier = Verifier()
        c = claim("EGFR突变患者不应使用吉非替尼", evidence_refs=["e1"])
        rep = verifier.verify([c], EvidenceRegistry([ev("e1", "吉非替尼不推荐用于T790M阳性患者")]))
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "relational_unverifiable"
        assert v.effective_critical is True  # 否定主张恒关键

    def test_causal_noncritical_annotates(self):
        verifier = Verifier()
        c = claim("EGFR突变激活下游信号通路", evidence_refs=["e1"])  # 无关键提示词
        rep = verifier.verify([c], EvidenceRegistry([ev("e1", POS_SPAN)]))
        v = rep.verdicts[0]
        assert v.status == ANNOTATE
        assert v.reason == "relational_unverifiable"


class TestSurfaceGate:
    def test_factual_pass(self):
        verifier = Verifier()
        c = claim("奥希替尼属于三代EGFR-TKI药物", evidence_refs=["e1"])
        rep = verifier.verify([c], EvidenceRegistry([ev("e1", POS_SPAN)]))
        assert rep.verdicts[0].status == PASS

    def test_entity_mismatch_critical_refuses(self):
        verifier = Verifier()
        # 主张引用的证据里完全没有吉非替尼 → 表面不一致
        c = claim("推荐使用吉非替尼", evidence_refs=["e1"], critical=True)
        rep = verifier.verify([c], EvidenceRegistry([ev("e1", POS_SPAN)]))
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "entity_mismatch"

    def test_number_mismatch_refuses(self):
        verifier = Verifier()
        c = claim("推荐剂量为120mg，每日一次", evidence_refs=["e1"], critical=True)
        rep = verifier.verify([c], EvidenceRegistry([ev("e1", POS_SPAN)]))
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "number_mismatch"

    def test_number_match_pass(self):
        verifier = Verifier()
        c = claim("推荐剂量为80mg，每日一次", evidence_refs=["e1"])
        rep = verifier.verify([c], EvidenceRegistry([ev("e1", POS_SPAN)]))
        assert rep.verdicts[0].status == PASS

    def test_evidence_negation_flip_refuses(self):
        verifier = Verifier()
        neg_span = "携带EGFR T790M的患者不应使用吉非替尼，不推荐用于此类患者。"
        c = claim("推荐使用吉非替尼", evidence_refs=["e1"], critical=True)
        rep = verifier.verify([c], EvidenceRegistry([ev("e1", neg_span)]))
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "evidence_negates_claim"


class TestSemanticGate:
    def _stub(self, result):
        class Stub:
            def __init__(self, r):
                self.r = r

            def check(self, claim, evidences):
                return self.r

        return Stub(result)

    def test_semantic_refute(self):
        verifier = Verifier(
            VerifierConfig(semantic_enabled=True, semantic_verifier=self._stub(False))
        )
        c = claim("奥希替尼属于三代EGFR-TKI药物", evidence_refs=["e1"], critical=True)
        rep = verifier.verify([c], EvidenceRegistry([ev("e1", POS_SPAN)]))
        v = rep.verdicts[0]
        assert v.semantic_used is True
        assert v.reason == "semantic_refute"
        assert v.status == REFUSE

    def test_semantic_support_passes(self):
        verifier = Verifier(
            VerifierConfig(semantic_enabled=True, semantic_verifier=self._stub(True))
        )
        c = claim("奥希替尼属于三代EGFR-TKI药物", evidence_refs=["e1"])
        rep = verifier.verify([c], EvidenceRegistry([ev("e1", POS_SPAN)]))
        assert rep.verdicts[0].status == PASS


class TestRuleDirectAndConflict:
    def test_rule_direct_pass_authoritative(self):
        verifier = Verifier()
        rules = [decision("r-div-a", ["甲磺酸阿帕替尼"], grade="A")]
        c = claim(
            "肺癌 三线 EGFR 推荐方案：甲磺酸阿帕替尼",
            rule_refs=["r-div-a"], critical=True,
        )
        rep = verifier.verify([c], EvidenceRegistry(), rules)
        v = rep.verdicts[0]
        assert v.status == PASS
        assert v.reason == "pass"
        assert v.checks["claim_support"]["reason"] == "rule_authoritative"

    def test_rule_direct_content_mismatch(self):
        verifier = Verifier()
        rules = [decision("r-div-a", ["甲磺酸阿帕替尼"], grade="A")]
        # 挂靠权威规则 id，却换了主张中的方案 → 规则内容比对不过
        c = claim("肺癌 三线 EGFR 推荐方案：安罗替尼", rule_refs=["r-div-a"], critical=True)
        rep = verifier.verify([c], EvidenceRegistry(), rules)
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "rule_content_mismatch"

    def test_conflict_loser_refuses(self):
        verifier = Verifier()
        rules = [
            decision("r-a", ["甲磺酸阿帕替尼"], grade="C", updated_at="2026-01-01"),
            decision("r-b", ["安罗替尼"], grade="C", updated_at="2026-01-01"),
        ]
        c = claim(
            "肺癌 三线 EGFR 推荐方案：甲磺酸阿帕替尼",
            rule_refs=["r-a"], critical=True,
        )
        rep = verifier.verify([c], EvidenceRegistry(), rules)
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "rule_conflict_lost"

    def test_conflict_higher_grade_wins(self):
        verifier = Verifier()
        rules = [
            decision("r-a", ["甲磺酸阿帕替尼"], grade="A", updated_at="2026-01-01"),
            decision("r-b", ["安罗替尼"], grade="C", updated_at="2026-01-01"),
        ]
        c = claim(
            "肺癌 三线 EGFR 推荐方案：甲磺酸阿帕替尼",
            rule_refs=["r-a"], critical=True,
        )
        rep = verifier.verify([c], EvidenceRegistry(), rules)
        assert rep.verdicts[0].status == PASS

    def test_diff_domain_rules_no_false_conflict(self):
        verifier = Verifier()
        lung = decision("r-lung", ["甲磺酸阿帕替尼"], grade="A", cancer_type="肺癌")
        breast = decision("r-breast", ["吡咯替尼"], grade="A", cancer_type="乳腺癌")
        c = claim(
            "肺癌 三线 EGFR 推荐方案：甲磺酸阿帕替尼",
            rule_refs=["r-lung"], critical=True,
        )
        rep = verifier.verify([c], EvidenceRegistry(), [lung, breast])
        assert rep.verdicts[0].status == PASS

    def test_stale_rule_loses_arbitration(self):
        verifier = Verifier()  # guideline 默认 2 年时效
        stale = decision("r-old", ["甲磺酸阿帕替尼"], updated_at="2019-01-01")
        c = claim(
            "肺癌 三线 EGFR 推荐方案：甲磺酸阿帕替尼",
            rule_refs=["r-old"], critical=True,
        )
        rep = verifier.verify([c], EvidenceRegistry(), [stale])
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "rule_conflict_lost"


class TestSufficiencyGate:
    def test_treatment_needs_high_grade(self):
        verifier = Verifier()
        c = claim("推荐方案为X方案", evidence_refs=["e1"], critical=True)
        rep = verifier.verify(
            [c],
            EvidenceRegistry([ev("e1", "X方案为一类新药。", grade="D")]),
        )
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "insufficient_evidence"

    def test_stale_dose_refuses(self):
        verifier = Verifier()
        c = claim("推荐剂量为80mg", evidence_refs=["e1"])
        rep = verifier.verify(
            [c],
            EvidenceRegistry([ev("e1", "推荐剂量为80mg。", grade="A", updated_at="2019-01-01")]),
        )
        assert rep.verdicts[0].reason == "insufficient_evidence"

    def test_stale_association_noncritical_annotates(self):
        verifier = Verifier()
        c = claim("该突变与PFS延长相关", evidence_refs=["e1"])
        rep = verifier.verify(
            [c],
            EvidenceRegistry([ev("e1", "该突变与PFS延长相关", grade="B", updated_at="2019-01-01")]),
        )
        v = rep.verdicts[0]
        assert v.status == ANNOTATE
        assert v.reason == "insufficient_evidence"


class TestOverallStatus:
    def _good(self):
        return claim("奥希替尼属于三代EGFR-TKI药物", evidence_refs=["e1"])

    def test_pass_all(self):
        verifier = Verifier()
        rep = verifier.verify(
            [self._good()], EvidenceRegistry([ev("e1", POS_SPAN)])
        )
        assert rep.overall_status == PASS

    def test_refuse_when_critical_fails(self):
        verifier = Verifier()
        bad = claim("EGFR突变患者不应使用吉非替尼", evidence_refs=["e1"])
        rep = verifier.verify(
            [self._good(), bad], EvidenceRegistry([ev("e1", POS_SPAN)])
        )
        assert rep.overall_status == REFUSE
        assert rep.any_refused is True

    def test_all_annotate(self):
        verifier = Verifier()
        c = claim("EGFR突变激活下游信号通路", evidence_refs=["e1"])
        rep = verifier.verify([c], EvidenceRegistry([ev("e1", POS_SPAN)]))
        assert rep.overall_status == ANNOTATE

    def test_empty_claims_no_rule_refuses(self):
        verifier = Verifier()
        rep = verifier.verify([], EvidenceRegistry())
        assert rep.overall_status == REFUSE
        assert rep.overall_reason == "no_evidence_no_rule"

    def test_empty_claims_with_rules_pass_rule_only(self):
        verifier = Verifier()
        rules = [decision("r1", ["奥希替尼"])]
        rep = verifier.verify([], EvidenceRegistry(), rules)
        assert rep.overall_status == PASS
        assert rep.overall_reason == "rule_only_no_claim"
