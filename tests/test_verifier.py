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

    def test_rule_direct_grade_c_pass(self):
        # 规则直出的 C 级推荐（推荐强度低，非充分性不足）→ 不应被误拒为
        # "insufficient_evidence"（#1）：规则的 grade 是推荐强度，不是拒答门槛。
        verifier = Verifier()
        rules = [decision("r-div-c", ["甲磺酸阿帕替尼"], grade="C")]
        c = claim(
            "肺癌 三线 EGFR 推荐方案：甲磺酸阿帕替尼",
            rule_refs=["r-div-c"], critical=True,
        )
        rep = verifier.verify([c], EvidenceRegistry(), rules)
        v = rep.verdicts[0]
        assert v.status == PASS
        assert v.reason == "pass"
        assert v.checks["claim_support"]["reason"] == "rule_authoritative"

    def test_rule_direct_stale_grade_c_still_refuses(self):
        # 规则直出的时效要求保留：C 级但已过期的规则 → 裁定判负（不因 #1 放宽时效）
        verifier = Verifier()
        rules = [decision("r-div-old", ["甲磺酸阿帕替尼"], grade="C", updated_at="2019-01-01")]
        c = claim(
            "肺癌 三线 EGFR 推荐方案：甲磺酸阿帕替尼",
            rule_refs=["r-div-old"], critical=True,
        )
        rep = verifier.verify([c], EvidenceRegistry(), rules)
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "rule_conflict_lost"

    def test_rule_direct_content_mismatch(self):
        verifier = Verifier()
        rules = [decision("r-div-a", ["甲磺酸阿帕替尼"], grade="A")]
        # 挂靠权威规则 id，却换了主张中的方案 → 规则内容比对不过
        c = claim("肺癌 三线 EGFR 推荐方案：安罗替尼", rule_refs=["r-div-a"], critical=True)
        rep = verifier.verify([c], EvidenceRegistry(), rules)
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "rule_content_mismatch"

    def test_conflict_divergence_when_tied_refuses(self):
        # 同域、同 grade、同时间的互斥规则 → 无胜出者 → 分歧声明（旧实现死代码，
        # lost 分支先命中，分歧分支永远不可达 —— 修复为分歧优先于"逐条判负"）
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
        assert v.reason == "rule_conflict_divergence"

    def test_conflict_loser_refuses(self):
        # 引用的规则在裁定中输给更高等级规则 → lost
        verifier = Verifier()
        rules = [
            decision("r-a", ["甲磺酸阿帕替尼"], grade="A", updated_at="2026-01-01"),
            decision("r-b", ["安罗替尼"], grade="C", updated_at="2026-01-01"),
        ]
        c = claim(
            "肺癌 三线 EGFR 推荐方案：安罗替尼",
            rule_refs=["r-b"], critical=True,
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

    def test_missing_date_dose_refuses(self):
        # 回归：dose/treatment 高门槛要求**可解析日期**——无 updated_at（未知时效）
        # 不豁免，即使 grade=A。修复前缺失时间戳会被当作"未过期"而放行。
        verifier = Verifier()
        c = claim("推荐剂量为80mg每日一次", evidence_refs=["e1"], critical=True)
        rep = verifier.verify(
            [c],
            EvidenceRegistry([ev("e1", "本品推荐剂量为80mg，每日口服一次。", updated_at=None)]),
        )
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "insufficient_evidence"

    def test_missing_date_treatment_refuses(self):
        verifier = Verifier()
        c = claim("肺癌一线推荐方案为阿帕替尼", evidence_refs=["e1"], critical=True)
        rep = verifier.verify(
            [c],
            EvidenceRegistry(
                [ev("e1", "肺癌一线患者推荐方案为阿帕替尼。", grade="A", updated_at=None)]
            ),
        )
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "insufficient_evidence"

    def test_missing_date_association_passes_low_gate(self):
        # 对照：低门槛（association/background）仅要求未过期；无日期=未知但未过期 → 仍通过
        verifier = Verifier()
        c = claim("该突变与PFS延长相关", evidence_refs=["e1"])
        rep = verifier.verify(
            [c],
            EvidenceRegistry([ev("e1", "携带该突变的患者PFS显著延长。", updated_at=None)]),
        )
        assert rep.verdicts[0].status == PASS


class TestAntiCrossEvidenceStitch:
    """回归：防跨证据拼装（round-2 code review 修复项）。

    旧实现表面门逐 token 全局取并集：实体命中一条证据、数值命中另一条证据也能
    "拼装通过"；充分性又取全局证据里最高等级者凑门槛 —— 弱证据供字面、强证据
    供等级。修复后：表面要素与数值必须落在**同一条**证据；充分性只在**锚定子集**
    （surface 通过的那几条）上判定。
    """

    def test_number_and_entity_must_share_one_evidence(self):
        # 药物实体只在 e-plan、剂量数值只在 e-dose → 谁都救不了对方
        # 数字匹配但实体不匹配 → entity_mismatch（防跨证据拼装）
        verifier = Verifier()
        e_plan = ev("e-plan", "肺癌三线EGFR复发患者推荐方案为阿帕替尼。")
        e_dose = ev("e-dose", "本品给药剂量为80mg每日一次，口服。")
        c = claim(
            "肺癌三线推荐方案为阿帕替尼，剂量80mg每日一次",
            evidence_refs=["e-plan", "e-dose"],
            critical=True,
        )
        rep = verifier.verify([c], EvidenceRegistry([e_plan, e_dose]))
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "entity_mismatch"

    def test_grade_stitch_irrelevant_high_grade_cannot_prop(self):
        # 充分性只认锚定证据：弱证据（D 级）供字面，无关 A 级证据不能给它凑等级门槛
        verifier = Verifier()
        weak = ev("e-weak", "肺癌三线EGFR复发患者推荐方案为阿帕替尼。", grade="D")
        strong = ev(
            "e-strong",
            "奥希替尼为三代EGFR-TKI，用于EGFR突变阳性晚期肺癌一线标准治疗。",
            grade="A",
        )
        c = claim(
            "肺癌三线推荐方案为阿帕替尼",
            evidence_refs=["e-weak", "e-strong"],
            critical=True,
        )
        rep = verifier.verify([c], EvidenceRegistry([weak, strong]))
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "insufficient_evidence"
        assert "e-strong" not in v.evidence_used


class TestMixedEvidenceRuleSufficiency:
    """回归：混合引用（证据 + 规则）时，被引规则的等级参与充分性判定。

    锚定修复把非规则直出主张的充分性池收窄为「锚定证据」，一度把被引规则的
    等级也一并剔除 —— 导致"弱证据（D 级）供字面 + 权威 A 级规则背书"的好主张
    被误拒为 insufficient_evidence。修复后：被引规则若**内容真正支持主张**
    （过 0.75 高门槛），其等级重新进入充分性池；内容不符（只复用条件域、偷换药物）
    的规则不得凑门槛。
    """

    def test_matching_rule_grade_counts(self):
        verifier = Verifier()
        weak = ev("e-weak", "肺癌三线EGFR复发患者推荐方案为奥希替尼。", grade="D")
        rule_a = decision("r-a", ["奥希替尼"], grade="A")
        c = claim(
            "肺癌 三线 EGFR 推荐方案：奥希替尼",
            evidence_refs=["e-weak"],
            rule_refs=["r-a"],
            critical=True,
        )
        rep = verifier.verify([c], EvidenceRegistry([weak]), [rule_a])
        v = rep.verdicts[0]
        assert v.status == PASS
        assert "r-a" in v.evidence_used

    def test_mismatched_rule_grade_cannot_prop(self):
        # 规则推荐奥希替尼、主张却写吉非替尼 → 内容不符，A 级规则不进入充分性池
        verifier = Verifier()
        weak = ev("e-weak", "肺癌三线EGFR复发患者推荐方案为吉非替尼。", grade="D")
        rule_a = decision("r-a", ["奥希替尼"], grade="A")
        c = claim(
            "肺癌 三线 EGFR 推荐方案：吉非替尼",
            evidence_refs=["e-weak"],
            rule_refs=["r-a"],
            critical=True,
        )
        rep = verifier.verify([c], EvidenceRegistry([weak]), [rule_a])
        v = rep.verdicts[0]
        assert v.status == REFUSE
        assert v.reason == "insufficient_evidence"
        assert "r-a" not in v.evidence_used


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
