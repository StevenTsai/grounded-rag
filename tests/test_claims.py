"""AnswerClaim 模型 / 解析 / 表层分类 / 表面要素提取 单元测试。"""

from __future__ import annotations

from groundedrag.guardrail.claims import (
    TYPE_CAUSAL,
    TYPE_COMPARISON,
    TYPE_FACTUAL,
    TYPE_INDICATION,
    TYPE_NEGATION,
    AnswerClaim,
    classify_claim_type,
    is_critical_claim,
    parse_anchor_text,
    parse_claims,
    parse_json_claims,
    surface_tokens,
)


class TestClassifyClaimType:
    def test_priority_negation_first(self):
        # 即使含比较词，否定词命中 → negation 覆盖
        assert classify_claim_type("该患者不应使用吉非替尼一线治疗") == TYPE_NEGATION

    def test_comparison(self):
        assert classify_claim_type("奥希替尼优于吉非替尼") == TYPE_COMPARISON
        assert classify_claim_type("三线推荐安罗替尼") == TYPE_COMPARISON

    def test_causal(self):
        assert classify_claim_type("EGFR突变激活下游信号通路") == TYPE_CAUSAL

    def test_indication(self):
        assert classify_claim_type("推荐方案：奥希替尼") == TYPE_INDICATION

    def test_factual_fallback(self):
        assert classify_claim_type("该药物的半衰期约为48小时") == TYPE_FACTUAL

    def test_empty(self):
        assert classify_claim_type("") == TYPE_FACTUAL


class TestIsCriticalClaim:
    def test_negation_comparison_always_critical(self):
        assert is_critical_claim("不应使用", TYPE_NEGATION)
        assert is_critical_claim("优于", TYPE_COMPARISON)

    def test_dose_hint_critical(self):
        assert is_critical_claim("每日剂量80mg", TYPE_FACTUAL)

    def test_background_factual_not_critical(self):
        assert not is_critical_claim("属于口服药物", TYPE_FACTUAL)


class TestAnswerClaim:
    def test_from_dict_invalid_type_reclassified(self):
        # 非法 type 值才触发表层重判；合法但失真的 type（如 LLM 自报 factual）
        # 原样保留，交由 verifier 在校验前确定性覆盖
        c = AnswerClaim.from_dict({"text": "不推荐使用吉非替尼", "type": "unknown"})
        assert c.type == TYPE_NEGATION
        kept = AnswerClaim.from_dict({"text": "不推荐使用吉非替尼", "type": "factual"})
        assert kept.type == TYPE_FACTUAL

    def test_from_dict_refs_aliases(self):
        c = AnswerClaim.from_dict(
            {"text": "x", "evidence": ["e1"], "rules": ["r1"], "critical": True}
        )
        assert c.evidence_refs == ["e1"]
        assert c.rule_refs == ["r1"]
        assert c.critical is True

    def test_to_dict(self):
        c = AnswerClaim.from_dict({"text": "x"})
        d = c.to_dict()
        assert d["text"] == "x"
        assert d["evidence_refs"] == []


class TestParseJsonClaims:
    def test_dict_wrapper(self):
        claims = parse_json_claims(
            '{"claims": [{"text": "推荐方案：奥希替尼[证据1]", "evidence_refs": ["ev1"]}]}',
            evidence_ids=["ev1"],
        )
        assert len(claims) == 1
        assert claims[0].text == "推荐方案：奥希替尼[证据1]"

    def test_bare_list_with_positional_refs(self):
        claims = parse_json_claims(
            '[{"text": "x", "evidence_refs": [1]}]',
            evidence_ids=["ev-a", "ev-b"],
        )
        assert claims[0].evidence_refs == ["ev-a"]

    def test_code_fence_stripped(self):
        claims = parse_json_claims(
            '```json\n[{"text": "y", "rule_refs": [2]}]\n```',
            rule_ids=["r1", "r2"],
        )
        assert claims[0].rule_refs == ["r2"]

    def test_invalid_json_returns_empty(self):
        assert parse_json_claims("not json") == []

    def test_skips_non_dict_items_and_blank(self):
        assert parse_json_claims('[{"text": ""}, "x"]') == []


class TestParseAnchorText:
    def test_bare_line_no_refs(self):
        claims = parse_anchor_text("奥希替尼是标准方案")
        assert len(claims) == 1
        assert claims[0].evidence_refs == []
        assert claims[0].rule_refs == []

    def test_positional_anchors_resolved(self):
        claims = parse_anchor_text(
            "肺癌一线推荐方案：奥希替尼[证据1][规则1]",
            evidence_ids=["ev-x"],
            rule_ids=["r-x"],
        )
        assert claims[0].evidence_refs == ["ev-x"]
        assert claims[0].rule_refs == ["r-x"]
        assert "奥希替尼" in claims[0].text
        assert "[证据1]" not in claims[0].text

    def test_out_of_range_position_with_pool_dropped(self):
        # 数字位置超出证据池范围时引用被丢弃（留给校验门判不完整）
        claims = parse_anchor_text("x[证据9]", evidence_ids=["ev1"])
        assert claims[0].evidence_refs == []

    def test_no_pool_raw_id_kept(self):
        # 未给池时，非数字 id 与数字都按原文保留
        claims = parse_anchor_text("x[证据9]", evidence_ids=None)
        assert claims[0].evidence_refs == ["9"]

    def test_comments_skipped(self):
        assert parse_anchor_text("# 注释\n// 行注释\n正文") == [
            AnswerClaim(text="正文")
        ]


class TestParseClaimsDispatch:
    def test_json_first(self):
        claims = parse_claims('[{"text": "a", "evidence_refs": [1]}]', evidence_ids=["e"])
        assert claims[0].evidence_refs == ["e"]

    def test_json_broken_falls_back_to_text(self):
        claims = parse_claims("一句话主张", evidence_ids=["e1"])
        assert len(claims) == 1
        assert claims[0].text == "一句话主张"


class TestSurfaceTokens:
    def test_digits_kept(self):
        assert any("120" in t for t in surface_tokens("剂量为120mg"))

    def test_function_words_dropped(self):
        toks = surface_tokens("推荐使用奥希替尼")
        assert "推荐" not in toks
        assert "使用" not in toks

    def test_short_tokens_dropped(self):
        assert "的" not in surface_tokens("奥希替尼的剂量")

    def test_entity_kept(self):
        assert any("EGFR" in t for t in surface_tokens("EGFR突变肺癌"))
