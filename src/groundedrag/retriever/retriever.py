"""检索服务：BM25 检索 + 实体增强 + 查询扩展。

领域词典/同义词**可注入**（``synonym_map``），None 时不启用扩展。
开源版内置通用示例词典；业务方可在不修改框架代码的前提下注入自己的
领域词典（词典本身可闭源，作为私有数据交付）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from groundedrag.retriever.bm25 import BM25, tokenize


@dataclass
class Document:
    """检索文档（承载后续证据归一所需的 EvidenceId 元数据字段）。

    字段语义对齐 `guardrail.evidence.EvidenceId`：
    source_type ∈ guideline|clinical_trial|insurance|variant|generic；
    grade ∈ A/B/C/D；updated_at 为 ISO 日期。
    """

    doc_id: str
    title: str = ""
    content: str = ""
    source_type: str = "generic"
    source_version: str = ""
    grade: str = "D"
    updated_at: Optional[str] = None
    claims_supported: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)  # 实体名（实体增强用）
    tags: List[str] = field(default_factory=list)

    @property
    def search_text(self) -> str:
        return f"{self.title}\n{self.content}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "source_type": self.source_type,
            "source_version": self.source_version,
            "grade": self.grade,
            "updated_at": self.updated_at,
            "claims_supported": list(self.claims_supported),
            "entities": list(self.entities),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Document":
        return cls(
            doc_id=str(data.get("doc_id") or data.get("id") or ""),
            title=str(data.get("title", "") or ""),
            content=str(data.get("content", "") or data.get("text", "") or ""),
            source_type=str(data.get("source_type", "generic") or "generic"),
            source_version=str(data.get("source_version", "") or ""),
            grade=str(data.get("grade", "D") or "D"),
            updated_at=(
                str(data["updated_at"]) if data.get("updated_at") is not None else None
            ),
            claims_supported=[
                str(x) for x in (data.get("claims_supported") or [])
            ],
            entities=[str(x) for x in (data.get("entities") or [])],
            tags=[str(x) for x in (data.get("tags") or [])],
        )


@dataclass
class RetrievedDocument:
    """一次检索命中：文档 + 检索得分。"""

    document: Document
    score: float

    def to_dict(self) -> Dict[str, object]:
        return {**self.document.to_dict(), "score": round(self.score, 4)}


class Retriever:
    """RAG 检索服务（BM25 基座 + 实体增强 + 查询扩展）。

    参数：
        documents: 待检索文档
        synonym_map: 领域同义词词典 ``{规范名: [别名...]}``；None 不启用扩展。
        entity_weight: 实体直接命中加权系数（0 关闭实体增强）。
        expand_depth: 查询扩展最多向每个命中词条补充的同义词数。
    """

    def __init__(
        self,
        documents: Sequence[Document],
        synonym_map: Optional[Dict[str, List[str]]] = None,
        *,
        entity_weight: float = 1.5,
        expand_depth: int = 3,
    ) -> None:
        self.documents = list(documents)
        self.synonym_map = {k: list(v) for k, v in (synonym_map or {}).items()}
        self.entity_weight = entity_weight
        self.expand_depth = expand_depth
        corpus = [d.search_text for d in self.documents]
        self._bm25 = BM25(corpus)
        self._id_to_idx = {d.doc_id: i for i, d in enumerate(self.documents)}

    # -- 查询扩展 ----------------------------------------------------------
    def expand_query(self, tokens: List[str]) -> List[str]:
        """基于同义词词典做查询扩展：词条命中则补充别名。

        词典键可整体短语（含空格）或单个 token；扩展深度受限防词面爆炸。
        """
        if not self.synonym_map:
            return tokens
        text_joined = "".join(tokens)
        extra: List[str] = []
        for canonical, aliases in self.synonym_map.items():
            # 词典键命中查询词面（短语优先整串匹配，否则逐 token）
            key_tokens = tokenize(canonical) if canonical else []
            hit = canonical in text_joined or (
                bool(key_tokens) and any(t in tokens for t in key_tokens)
            )
            if not hit:
                continue
            for alias in aliases[: self.expand_depth]:
                # 只补充真正带来新词面的别名
                if alias and alias not in tokens and alias not in extra:
                    extra.append(alias)
        return tokens + extra

    # -- 实体增强 ----------------------------------------------------------
    def _entity_boost(self, doc: Document, query_tokens: List[str]) -> float:
        """标题/实体与查询 token 重叠的加权加成（直接命中受重视）。"""
        if self.entity_weight <= 0:
            return 0.0
        pool = set(tokenize(doc.title)) | set(doc.entities) | set(doc.tags)
        qset = set(query_tokens)
        if not pool or not qset:
            return 0.0
        overlap = len(pool & qset)
        return self.entity_weight * overlap if overlap else 0.0

    # -- 检索入口 ----------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        expand: bool = True,
        entity_boost: bool = True,
    ) -> List[RetrievedDocument]:
        """检索 top_k 文档。

        打分 = BM25 得分 + 实体直接命中加权。返回按得分降序。
        """
        base_tokens = tokenize(query)
        if not base_tokens:
            return []
        q_tokens = self.expand_query(base_tokens) if expand else base_tokens
        scored: List[tuple[int, float]] = self._bm25.search_with_scores(
            " ".join(q_tokens), top_k=len(self.documents)
        )
        results: List[RetrievedDocument] = []
        for idx, bm25_score in scored:
            doc = self.documents[idx]
            total = bm25_score
            if entity_boost:
                total += self._entity_boost(doc, q_tokens)
            results.append(RetrievedDocument(doc, total))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # 便捷：直接返回 dict（示例/评测用）
    def retrieve_dicts(self, query: str, top_k: int = 5, **kw) -> List[Dict[str, object]]:
        return [r.to_dict() for r in self.retrieve(query, top_k, **kw)]
