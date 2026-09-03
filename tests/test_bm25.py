"""BM25 检索器与中文分词的单元测试。"""

from __future__ import annotations

import pytest

from groundedrag.retriever.bm25 import BM25, _char_bigram_fallback, tokenize

try:  # 与 bm25.py 同一探测方式
    import jieba  # noqa: F401

    HAS_JIEBA = True
except Exception:  # pragma: no cover
    HAS_JIEBA = False


class TestCharBigramFallback:
    """CJK char-bigram 回退路径（use_jieba=False / jieba 缺失）。"""

    def test_empty(self):
        assert tokenize("", use_jieba=False) == []
        assert _char_bigram_fallback("") == []

    def test_ascii_kept_as_word(self):
        assert tokenize("EGFR", use_jieba=False) == ["egfr"]
        assert tokenize("ADC 药物", use_jieba=False) == ["adc", "药物"]

    def test_two_char_cjk_is_single_bigram(self):
        # 两个相邻汉字产出唯一二元组"肺癌"（连续片段 len==2 → 整串一个 bigram）
        assert tokenize("肺癌", use_jieba=False) == ["肺癌"]

    def test_longer_cjk_produces_bigrams(self):
        # 三个汉字 → 两个相邻二元组
        assert tokenize("奥希替尼", use_jieba=False) == ["奥希", "希替", "替尼"]

    def test_mixed_cjk_ascii(self):
        toks = tokenize("EGFR突变的肺癌", use_jieba=False)
        assert "egfr" in toks
        assert "突变" in toks
        assert "肺癌" in toks

    def test_punctuation_skipped(self):
        assert _char_bigram_fallback("肺癌, 胃癌") == ["肺癌", "胃癌"]


class TestTokenizer:
    def test_jieba_missing_raises_when_forced(self):
        # 有 jieba 环境强制 use_jieba=True 不出错；语义校验交给回退分支测试
        if HAS_JIEBA:
            assert tokenize("肺癌", use_jieba=True)
        else:  # pragma: no cover
            with pytest.raises(ImportError):
                tokenize("肺癌", use_jieba=True)

    def test_auto_prefers_jieba_when_available(self):
        if HAS_JIEBA:
            toks = tokenize("EGFR突变的晚期肺癌")
            assert "EGFR" in toks
            assert "肺癌" in toks


class TestBM25:
    def _corpus(self):
        return [
            "奥希替尼用于EGFR突变晚期肺癌一线治疗",
            "吉非替尼用于EGFR突变肺癌的治疗",
            "帕博利珠单抗用于黑色素瘤治疗",
        ]

    def test_ranks_exact_match_first(self):
        bm = BM25(self._corpus())
        hits = bm.search("肺癌一线奥希替尼")
        assert hits and hits[0] == 0

    def test_search_with_scores_ordering(self):
        bm = BM25(self._corpus())
        scored = bm.search_with_scores("EGFR 肺癌 治疗", top_k=10)
        assert len(scored) == 3
        scores = [s for _, s in scored]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0 for s in scores)

    def test_zero_score_filtered(self):
        bm = BM25(self._corpus())
        # 完全无关的词不命中任何文档 → 空结果
        assert bm.search("苹果香蕉西瓜", top_k=3) == []

    def test_empty_query(self):
        bm = BM25(self._corpus())
        assert bm.search("", top_k=3) == []
        assert bm.search_with_scores("  ", top_k=3) == []

    def test_score_out_of_range_raises(self):
        bm = BM25(self._corpus())
        with pytest.raises(IndexError):
            bm.score(["肺癌"], 99)

    def test_custom_tokenizer(self):
        # 注入把整句当单 token 的分词器：命中即全分。
        # 注意：BM25.search 的查询分词独立于构建分词器，需显式传入 tokenizer。
        bm = BM25(["hello world"], tokenizer=lambda s: [s])
        assert bm.search("hello world", top_k=1, tokenizer=lambda s: [s]) == [0]
        # 查询用默认 jieba 分词（hello/world）时不会命中整句 token
        assert bm.search("hello world", top_k=1) == []

    def test_short_corpus_safe(self):
        assert BM25([]).search("anything") == []
        assert BM25([]).scores(["x"]) == []
