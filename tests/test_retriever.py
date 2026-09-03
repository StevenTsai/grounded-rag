"""Retriever（BM25 + 实体增强 + 查询扩展）单元测试。"""

from __future__ import annotations

from groundedrag.retriever.bm25 import tokenize
from groundedrag.retriever.retriever import Document, Retriever


def _docs():
    return [
        Document(
            doc_id="d1",
            title="肺癌EGFR一线治疗",
            content="奥希替尼用于EGFR突变晚期非小细胞肺癌的一线治疗，口服80mg每日一次。",
            source_type="guideline",
            grade="A",
            entities=["奥希替尼", "EGFR"],
        ),
        Document(
            doc_id="d2",
            title="结肠癌RAS野生型治疗",
            content="RAS野生型转移性结直肠癌的一线方案参考西妥昔单抗联合化疗。",
            source_type="guideline",
            grade="B",
            entities=["西妥昔单抗"],
        ),
        Document(
            doc_id="d3",
            title="无关背景资料",
            content="这里是一段与肿瘤治疗无关的普通介绍文本。",
            source_type="generic",
            grade="D",
        ),
    ]


class TestDocument:
    def test_roundtrip(self):
        doc = _docs()[0]
        restored = Document.from_dict(doc.to_dict())
        assert restored == doc

    def test_from_dict_id_fallback(self):
        d = Document.from_dict({"id": "x1", "content": "内容", "text": ""})
        assert d.doc_id == "x1"
        assert d.content == "内容"

    def test_search_text_concat(self):
        doc = _docs()[0]
        assert doc.title in doc.search_text and doc.content in doc.search_text

    def test_defaults(self):
        d = Document.from_dict({"doc_id": "a"})
        assert d.source_type == "generic"
        assert d.grade == "D"
        assert d.updated_at is None


class TestRetriever:
    def test_retrieve_orders_by_relevance(self):
        r = Retriever(_docs())
        hits = r.retrieve("EGFR 突变 肺癌 一线", top_k=3)
        assert hits[0].document.doc_id == "d1"

    def test_top_k_respected(self):
        r = Retriever(_docs())
        assert len(r.retrieve("治疗", top_k=2)) <= 2

    def test_synonym_expansion_enriches_query(self):
        r = Retriever(
            _docs(),
            synonym_map={"肺癌": ["非小细胞肺癌"], "奥希替尼": ["泰瑞沙"]},
        )
        toks = r.expand_query(tokenize("肺癌"))
        assert "非小细胞肺癌" in toks  # 词典别名被补充进查询词面

    def test_expand_disabled_passthrough(self):
        r = Retriever(_docs(), synonym_map={"肺癌": ["非小细胞肺癌"]})
        base = tokenize("肺癌")
        expanded = r.expand_query(base)
        assert len(expanded) > len(base)

    def test_entity_boost_method(self):
        r = Retriever(_docs(), entity_weight=2.0)
        # d1.entities 含 "EGFR"，查询 token 命中 → 加成 = 权重 * 重叠数
        boost = r._entity_boost(_docs()[0], tokenize("EGFR"))
        assert boost == 2.0
        # 无该实体的 d3 → 0
        assert r._entity_boost(_docs()[2], tokenize("EGFR")) == 0.0

    def test_entity_boost_disabled_by_zero_weight(self):
        r = Retriever(_docs(), entity_weight=0.0)
        assert r._entity_boost(_docs()[0], tokenize("EGFR")) == 0.0


class TestRetrieveDicts:
    def test_returns_plain_dicts(self):
        r = Retriever(_docs())
        out = r.retrieve_dicts("肺癌", top_k=2)
        assert isinstance(out, list)
        assert all(isinstance(x, dict) and "score" in x for x in out)
