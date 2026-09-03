"""EvidenceId / 时效策略 / 证据注册表 / 检索结果归一 单元测试。"""

from __future__ import annotations

from datetime import date

from groundedrag.guardrail.evidence import (
    EvidenceId,
    EvidenceRegistry,
    StalenessPolicy,
    from_retrieved_docs,
    parse_iso_date,
)
from groundedrag.retriever.retriever import Document, RetrievedDocument


def _ev(evidence_id="ev1", **kw) -> EvidenceId:
    base = dict(
        evidence_id=evidence_id,
        doc_id="doc1",
        source_type="guideline",
        source_version="CSCO-2026",
        grade="A",
        updated_at="2026-02-01",
        text_span="奥希替尼一线标准方案。",
    )
    base.update(kw)
    return EvidenceId(**base)


class TestEvidenceId:
    def test_schema_complete(self):
        assert _ev().schema_complete()

    def test_schema_incomplete_fields(self):
        empty = EvidenceId(
            evidence_id="", doc_id="d", source_type="", source_version="",
            grade="", updated_at=None, text_span="",
        )
        assert not empty.schema_complete()

    def test_schema_invalid_source_type(self):
        assert not _ev(source_type="私有来源").schema_complete()

    def test_schema_invalid_grade(self):
        assert not _ev(grade="X").schema_complete()

    def test_grade_uppercased_in_from_dict(self):
        e = EvidenceId.from_dict({**_ev().to_dict(), "grade": "a"})
        assert e.grade == "A"

    def test_roundtrip(self):
        assert EvidenceId.from_dict(_ev().to_dict()) == _ev()

    def test_grade_rank(self):
        assert _ev(grade="A").grade_rank > _ev(grade="C").grade_rank

    def test_stale_candidate(self):
        assert _ev(updated_at=None).is_stale_candidate is False
        assert _ev().is_stale_candidate is True


class TestParseIsoDate:
    def test_formats(self):
        assert parse_iso_date("2026-02-01") == date(2026, 2, 1)
        assert parse_iso_date("2026-02") == date(2026, 2, 1)
        assert parse_iso_date("2026") == date(2026, 1, 1)

    def test_datetime_and_none(self):
        from datetime import datetime

        assert parse_iso_date(datetime(2026, 2, 3, 4, 5)) == date(2026, 2, 3)
        assert parse_iso_date(None) is None

    def test_garbage(self):
        assert parse_iso_date("not-a-date") is None


class TestStalenessPolicy:
    def test_stale_after_years(self):
        now = date(2026, 9, 1)
        policy = StalenessPolicy(now=now)  # guideline 默认 2 年
        old = _ev(updated_at="2023-01-01")  # >3 年
        new = _ev(updated_at="2026-01-01")
        assert policy.is_stale(old)
        assert not policy.is_stale(new)

    def test_no_updated_at_not_stale_but_time_unknown(self):
        policy = StalenessPolicy(now=date(2026, 9, 1))
        e = _ev(updated_at=None, source_version="")
        assert not policy.is_stale(e)
        assert policy.time_unknown(e)
        assert not policy.time_unknown(_ev())  # 有 source_version → 非 time_unknown

    def test_custom_years(self):
        policy = StalenessPolicy(years={"guideline": 1}, now=date(2026, 9, 1))
        assert policy.is_stale(_ev(updated_at="2025-06-01"))  # 超过 1 年


class TestEvidenceRegistry:
    def test_add_get_resolve(self):
        e1, e2 = _ev("ev1"), _ev("ev2", doc_id="doc2")
        reg = EvidenceRegistry([e1, e2])
        assert reg.get("ev1") == e1
        assert reg.resolve(["ev1", "missing"]) == [e1]
        assert len(reg) == 2

    def test_empty(self):
        reg = EvidenceRegistry()
        assert len(reg) == 0
        assert reg.resolve(["x"]) == []
        assert reg.all() == []


class TestFromRetrievedDocs:
    def _retrieved(self):
        return [
            RetrievedDocument(
                Document(
                    doc_id="d1",
                    title="标题",
                    content="内容片段" * 500,  # 超出 text_span_limit 截断
                    source_type="clinical_trial",
                    grade="b",
                    claims_supported=["c1"],
                ),
                1.0,
            )
        ]

    def test_mapping_and_fallback(self):
        evs = from_retrieved_docs(self._retrieved())
        assert len(evs) == 1
        e = evs[0]
        assert e.doc_id == "d1"
        assert e.evidence_id == "d1"
        assert e.source_type == "clinical_trial"
        assert e.grade == "B"  # 归一大写
        assert e.claims_supported == ["c1"]

    def test_text_span_limited(self):
        evs = from_retrieved_docs(self._retrieved(), text_span_limit=100)
        assert len(evs[0].text_span) <= 100

    def test_index_fallback_when_no_doc_id(self):
        rd = RetrievedDocument(Document(doc_id="", content="x"), 1.0)
        evs = from_retrieved_docs([rd], start_index=1)
        assert evs[0].evidence_id == "ev1"
