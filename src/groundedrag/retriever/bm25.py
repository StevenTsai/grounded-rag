"""BM25 中文检索算法。

设计要点：
- **jieba 为核心依赖**（MIT）：中文分词是 BM25 / 实体增强 / 证据比对的前提。
- jieba 缺失时回退到 **CJK char-bigram**（相邻字符二元组），**绝不回退到 ``split()``**
  （对中文约等于不分词）。
- 本模块为独立可单用的 BM25 实现（无其他依赖），也可嵌入 `Retriever` 使用。

线程安全：构建后只读，score/search 均可并发调用。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence

try:  # jieba 为核心依赖，但保留回退以支持无网络/受限环境
    import jieba

    _HAS_JIEBA = True
except Exception:  # pragma: no cover - 回退路径仅在无 jieba 时触发
    jieba = None
    _HAS_JIEBA = False


# CJK 统一表意文字区（含扩展 A 常用块）
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
_ALNUM_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str, *, use_jieba: bool | None = None) -> List[str]:
    """对文本做中文分词，返回 token 列表。

    - ``use_jieba=None``：自动选择（有 jieba 用 jieba，否则 char-bigram）
    - ``use_jieba=True``：强制 jieba（若 jieba 缺失抛 ImportError）
    - ``use_jieba=False``：强制 CJK char-bigram 回退
    """
    if not text:
        return []
    if use_jieba is None:
        use_jieba = _HAS_JIEBA
    if use_jieba:
        if jieba is None:  # pragma: no cover
            raise ImportError("jieba 未安装，且 use_jieba=True")
        # jieba 返回词语；过滤空白 token
        tokens = [t.strip() for t in jieba.lcut(text) if t and t.strip()]
        return tokens
    return _char_bigram_fallback(text)


def _char_bigram_fallback(text: str) -> List[str]:
    """CJK char-bigram 回退：对连续 CJK 文本产生相邻二元组，对 ASCII 按单词切分。

    例：`EGFR突变的肺癌` → 对连续中文片段产生 (突变)(变的)(的肺)(肺癌)，
    对 `EGFR` 保留为小写词。二元组保留了相邻性信息，比单字检索质量高得多。
    """
    tokens: List[str] = []
    text = text.lower()
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if _CJK_RE.match(ch):
            # 收集连续中文片段，产出 char-bigram
            j = i
            while j < n and _CJK_RE.match(text[j]):
                j += 1
            run = text[i:j]
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[k : k + 2] for k in range(len(run) - 1))
            i = j
        elif _ALNUM_RE.match(ch):
            m = _ALNUM_RE.match(text, i)
            tokens.append(m.group(0).lower())  # type: ignore[union-attr]
            i = m.end()  # type: ignore[union-attr]
        else:
            i += 1
    return tokens


class BM25:
    """Okapi BM25 检索器（独立可单用）。

    参数：
        corpus: 文档列表（已分词前，本类内部会自行 tokenize）
        k1/b:   BM25 超参数（默认 1.5 / 0.75）
        tokenizer: 自定义分词函数；默认用 `groundedrag.retriever.bm25.tokenize`
    """

    def __init__(
        self,
        corpus: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer=None,
    ) -> None:
        if tokenizer is None:
            tokenizer = tokenize
        self.k1 = k1
        self.b = b
        self.corpus = list(corpus)
        self._doc_tokens: List[List[str]] = []
        self._doc_freqs: List[Counter] = []
        self._df: Counter[str] = Counter()
        self._doc_lens: List[int] = []
        self._avgdl = 0.0
        self._build(tokenizer)

    def _build(self, tokenizer) -> None:
        total_len = 0
        for text in self.corpus:
            toks = tokenizer(text)
            self._doc_tokens.append(toks)
            self._doc_lens.append(len(toks))
            freq = Counter(toks)
            self._doc_freqs.append(freq)
            for term in freq:
                self._df[term] += 1
            total_len += len(toks)
        n = len(self.corpus)
        self._avgdl = total_len / n if n else 0.0

    # -- 内部：单个 term 的 idf ------------------------------------------
    def _idf(self, term: str) -> float:
        n = len(self.corpus)
        df = self._df.get(term, 0)
        # 平滑 idf，避免除零；保证与标准实现一致的数值行为
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: Sequence[str], doc_index: int) -> float:
        """单篇文档 BM25 得分。"""
        if doc_index < 0 or doc_index >= len(self.corpus):
            raise IndexError(f"doc_index {doc_index} 越界")
        dl = self._doc_lens[doc_index]
        if dl == 0:
            return 0.0
        freq = self._doc_freqs[doc_index]
        total = 0.0
        for term in query_tokens:
            tf = freq.get(term, 0)
            if not tf:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            total += self._idf(term) * (tf * (self.k1 + 1)) / denom
        return total

    def scores(self, query_tokens: Sequence[str]) -> List[float]:
        """全库逐文档得分（顺序与 corpus 一致）。"""
        return [self.score(query_tokens, i) for i in range(len(self.corpus))]

    def search(
        self,
        query: str,
        top_k: int = 5,
        tokenizer=None,
    ) -> List[int]:
        """返回命中文档下标列表（按得分降序）。"""
        if tokenizer is None:
            tokenizer = tokenize
        q_tokens = tokenizer(query)
        if not q_tokens:
            return []
        scored = self.scores(q_tokens)
        ranked = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)
        # 过滤 0 分
        return [i for i in ranked if scored[i] > 0][:top_k]

    def search_with_scores(
        self,
        query: str,
        top_k: int = 5,
        tokenizer=None,
    ) -> List[tuple[int, float]]:
        """返回 (下标, 得分) 列表，供上层排序/截断。"""
        if tokenizer is None:
            tokenizer = tokenize
        q_tokens = tokenizer(query)
        if not q_tokens:
            return []
        scored = self.scores(q_tokens)
        ranked = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)
        return [(i, scored[i]) for i in ranked if scored[i] > 0][:top_k]
