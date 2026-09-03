"""证据溯源：EvidenceId 归一与证据注册表。

一次可信回答的三层链式对象之一为 ``EvidenceId[]`` —— 每条 AnswerClaim 绑定
的支持证据。schema 字段必须完整（见设计文档 §3.3），否则校验门无法编码/测试。

``text_span`` 为证据原文片段，供 verifier 做**表面要素一致性**比对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional

from groundedrag.guardrail.models import GRADE_ORDER, VALID_GRADES
from groundedrag.retriever.retriever import RetrievedDocument

VALID_SOURCE_TYPES = ("guideline", "clinical_trial", "insurance", "variant", "generic")


@dataclass
class EvidenceId:
    """证据标识（verifier 的一切判定只读这些字段）。"""

    evidence_id: str          # 全局唯一
    doc_id: str               # 召回文档 id
    source_type: str          # guideline | clinical_trial | insurance | variant | generic
    source_version: str       # 来源版本；缺失视为未知版本
    grade: str                # A/B/C/D
    updated_at: Optional[str]  # ISO 日期；缺失视为无时间信息
    text_span: str            # 证据原文片段（表面要素一致性比对用）
    claims_supported: List[str] = field(default_factory=list)

    @property
    def is_stale_candidate(self) -> bool:
        """具备时效判定条件：有 updated_at。"""
        return bool(self.updated_at)

    @property
    def grade_rank(self) -> int:
        return GRADE_ORDER.get(self.grade.upper(), -1)

    def schema_complete(self) -> bool:
        """schema 完整性：核心字段齐全且取值合法。"""
        return bool(
            self.evidence_id
            and self.doc_id
            and self.source_type in VALID_SOURCE_TYPES
            and self.grade.upper() in VALID_GRADES
            and bool(self.text_span)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "source_type": self.source_type,
            "source_version": self.source_version,
            "grade": self.grade.upper(),
            "updated_at": self.updated_at,
            "text_span": self.text_span,
            "claims_supported": list(self.claims_supported),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceId":
        return cls(
            evidence_id=str(data.get("evidence_id", "")),
            doc_id=str(data.get("doc_id", "")),
            source_type=str(data.get("source_type", "")),
            source_version=str(data.get("source_version", "") or ""),
            grade=str(data.get("grade", "")).upper(),
            updated_at=(str(data["updated_at"]) if data.get("updated_at") is not None else None),
            text_span=str(data.get("text_span", "")),
            claims_supported=[str(x) for x in (data.get("claims_supported") or [])],
        )


def parse_iso_date(value: Optional[str]) -> Optional[date]:
    """解析 ISO 日期（支持 ``2026-01-01`` / ``2026-01`` / datetime 对象）。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # 尝试 datetime 全量格式
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        return None


class StalenessPolicy:
    """过期规则（按来源类型设定年限）。

    语义（对齐设计文档 §3.4.2）：
    - 证据带 updated_at 且距今 > 设定年限 → stale（不能作为关键门槛满足证据）
    - updated_at 缺失 + source_version 缺失 → 无法确认时效，仅作低门槛证据
    - 缺失 updated_at（但有 source_version）→ 视作未知时间，不豁免过期判定
    """

    DEFAULT_YEARS = {
        "guideline": 2,
        "clinical_trial": 5,
        "insurance": 5,
        "variant": 5,
        "generic": 5,
    }

    def __init__(self, years: Optional[Mapping[str, int]] = None, now: Optional[date] = None) -> None:
        self.years = {**self.DEFAULT_YEARS, **(years or {})}
        self.now = now or date.today()

    def years_for(self, source_type: str) -> int:
        return self.years.get(source_type, self.DEFAULT_YEARS["generic"])

    def is_stale(self, evidence: EvidenceId) -> bool:
        """是否过期。无 updated_at → 不判 stale，但会被"未知时效"规则降级处理。"""
        d = parse_iso_date(evidence.updated_at)
        if d is None:
            return False
        age_days = (self.now - d).days
        max_days = self.years_for(evidence.source_type) * 365
        return age_days > max_days

    def time_unknown(self, evidence: EvidenceId) -> bool:
        """无法确认时效：updated_at 缺失 且 source_version 缺失。"""
        return not evidence.updated_at and not evidence.source_version


class EvidenceRegistry:
    """证据注册表：以 evidence_id 为键，提供解析与过滤。"""

    def __init__(self, evidences: Optional[Iterable[EvidenceId]] = None) -> None:
        self._store: Dict[str, EvidenceId] = {}
        for ev in evidences or []:
            self.add(ev)

    def add(self, evidence: EvidenceId) -> None:
        if evidence.evidence_id:
            self._store[evidence.evidence_id] = evidence

    def get(self, evidence_id: str) -> Optional[EvidenceId]:
        return self._store.get(evidence_id)

    def resolve(self, refs: Iterable[str]) -> List[EvidenceId]:
        out = []
        for r in refs:
            ev = self._store.get(r)
            if ev is not None:
                out.append(ev)
        return out

    def all(self) -> List[EvidenceId]:
        return list(self._store.values())

    def __len__(self) -> int:
        return len(self._store)


def from_retrieved_docs(
    docs: Iterable[RetrievedDocument],
    *,
    start_index: int = 1,
    text_span_limit: int = 600,
) -> List[EvidenceId]:
    """把检索命中归一为 EvidenceId 集。

    - evidence_id 取文档自带 id 或 ``ev{start_index}``（保证在评测中可对应锚点）
    - text_span 截断到 text_span_limit 字符（表面要素比对足够）
    """
    out: List[EvidenceId] = []
    idx = start_index
    for rd in docs:
        doc = rd.document
        ev_id = doc.doc_id or f"ev{idx}"
        span = f"{doc.title}\n{doc.content}"[:text_span_limit] if doc.content else doc.title
        out.append(
            EvidenceId(
                evidence_id=ev_id,
                doc_id=doc.doc_id,
                source_type=doc.source_type,
                source_version=doc.source_version,
                grade=doc.grade.upper() if doc.grade else "D",
                updated_at=doc.updated_at,
                text_span=span,
                claims_supported=list(doc.claims_supported),
            )
        )
        idx += 1
    return out
