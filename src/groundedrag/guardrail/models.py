"""纯 Python 数据模型（无 ORM 依赖）。

把「诊疗路径 / 耐药规则」表示为轻量 dataclass，是约束层解耦的核心：
规则引擎只依赖本模块的纯数据类型与 RuleProvider，不 import 任何
SQLAlchemy / 数据库模型；业务方接入真实数据库时只需实现 Provider
做字段映射，本层无需改动。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 证据等级确定性排序：A > B > C > D
GRADE_ORDER = {"A": 4, "B": 3, "C": 2, "D": 1}
VALID_GRADES = ("A", "B", "C", "D")

# 方案名组合分隔符（用于把「A+贝伐」拆成组件，判断互补 vs 互斥）
_PLAN_SEP_RE = re.compile(r"[+＋,，、/；;\s和与以及]")

# 治疗线次归一化
_LINE_ALIASES = {
    "一线": "一线", "1线": "一线", "first": "一线", "初始": "一线",
    "二线": "二线", "2线": "二线", "second": "二线",
    "三线": "三线", "3线": "三线", "后线": "后线",
}


def normalize_line(value: Optional[str]) -> Optional[str]:
    """治疗线次归一化（未知名词保持原样）。"""
    if value is None:
        return None
    return _LINE_ALIASES.get(value.strip(), value.strip())


def plan_components(plan_name: str) -> List[str]:
    """把方案名拆成规范化组件集合（用于冲突/互补判定）。"""
    parts = [p.strip() for p in _PLAN_SEP_RE.split(plan_name) if p.strip()]
    if not parts:
        return []
    return [p.lower() for p in parts]


@dataclass
class RuleCondition:
    """单条规则前件条件。

    - ``field``：上下文字段名，如 cancer_type / treatment_line / biomarker / gene
    - ``op``：eq | neq | in | contains | all
    - ``value``：str 或 list[str]；eq/neq 取字符串，in/all/contains 取列表
    """

    field: str
    op: str = "eq"
    value: Any = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "op": self.op, "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleCondition":
        return cls(field=str(data.get("field", "")), op=str(data.get("op", "eq")), value=data.get("value", ""))

    def __str__(self) -> str:
        if isinstance(self.value, (list, tuple)):
            v = " / ".join(str(x) for x in self.value)
        else:
            v = str(self.value)
        return f"{self.field} {self.op} {v}"


@dataclass
class RuleRecommendation:
    """规则的一条推荐（主推方案或可选项）。"""

    plan_name: str
    grade: str = "D"
    note: str = ""
    source_version: str = ""

    @property
    def components(self) -> List[str]:
        return plan_components(self.plan_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "grade": self.grade,
            "note": self.note,
            "source_version": self.source_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleRecommendation":
        return cls(
            plan_name=str(data.get("plan_name", "")),
            grade=str(data.get("grade", "D") or "D"),
            note=str(data.get("note", "") or ""),
            source_version=str(data.get("source_version", "") or ""),
        )


@dataclass
class Rule:
    """诊疗路径规则。

    ``domain`` 为规则的条件域速写（同癌种 + 同线次 + 同标志物/基因），
    与 ``conditions`` 冗余但专用于「冲突」的可编码定义与裁定。
    """

    rule_id: str
    name: str = ""
    cancer_type: Optional[str] = None
    treatment_line: Optional[str] = None
    biomarker: Optional[str] = None          # 标志物（含突变），如 "EGFR" / "EGFR 突变"
    conditions: List[RuleCondition] = field(default_factory=list)
    recommendations: List[RuleRecommendation] = field(default_factory=list)
    source_version: str = ""
    updated_at: Optional[str] = None          # ISO 日期
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_plans(self) -> List[str]:
        """主推方案（按 components 拆分、去重）。"""
        seen: List[str] = []
        for rec in self.recommendations:
            for comp in rec.components:
                if comp not in seen:
                    seen.append(comp)
        return seen

    @property
    def max_grade(self) -> str:
        """推荐中的最高证据等级（A 最高）。"""
        best = "D"
        best_rank = -1
        for rec in self.recommendations:
            g = rec.grade.upper()
            rank = GRADE_ORDER.get(g, -1)
            if rank > best_rank:
                best_rank, best = rank, g
        return best

    @property
    def normalized_line(self) -> Optional[str]:
        return normalize_line(self.treatment_line)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "cancer_type": self.cancer_type,
            "treatment_line": self.treatment_line,
            "biomarker": self.biomarker,
            "conditions": [c.to_dict() for c in self.conditions],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "source_version": self.source_version,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        return cls(
            rule_id=str(data.get("rule_id", "") or data.get("id", "")),
            name=str(data.get("name", "") or ""),
            cancer_type=_opt_str(data.get("cancer_type")),
            treatment_line=normalize_line(_opt_str(data.get("treatment_line"))),
            biomarker=_opt_str(data.get("biomarker")),
            conditions=[
                RuleCondition.from_dict(c) for c in (data.get("conditions") or [])
            ],
            recommendations=[
                RuleRecommendation.from_dict(r) for r in (data.get("recommendations") or [])
            ],
            source_version=str(data.get("source_version", "") or ""),
            updated_at=_opt_str(data.get("updated_at")),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class RuleDecision:
    """规则引擎的一次决策产物（命中规则 + 匹配条件 + 推荐方案 + 等级）。"""

    rule_id: str
    rule_name: str = ""
    matched_conditions: List[str] = field(default_factory=list)
    recommendations: List[RuleRecommendation] = field(default_factory=list)
    source_version: str = ""
    updated_at: Optional[str] = None
    cancer_type: Optional[str] = None
    treatment_line: Optional[str] = None
    biomarker: Optional[str] = None

    @property
    def plan_components(self) -> List[str]:
        seen: List[str] = []
        for rec in self.recommendations:
            for comp in rec.components:
                if comp not in seen:
                    seen.append(comp)
        return seen

    @property
    def max_grade(self) -> str:
        best = "D"
        best_rank = -1
        for rec in self.recommendations:
            rank = GRADE_ORDER.get(rec.grade.upper(), -1)
            if rank > best_rank:
                best_rank, best = rank, rec.grade.upper()
        return best

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "matched_conditions": list(self.matched_conditions),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "source_version": self.source_version,
            "updated_at": self.updated_at,
            "cancer_type": self.cancer_type,
            "treatment_line": self.treatment_line,
            "biomarker": self.biomarker,
            "max_grade": self.max_grade,
        }


@dataclass
class ResistanceRule:
    """耐药/替代规则（与 Rule 结构平行，供 provider 返回）。"""

    rule_id: str
    name: str = ""
    gene: Optional[str] = None          # 耐药相关基因/突变
    prior_regimen: Optional[str] = None  # 既往方案上下文
    conditions: List[RuleCondition] = field(default_factory=list)
    recommendations: List[RuleRecommendation] = field(default_factory=list)
    source_version: str = ""
    updated_at: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "gene": self.gene,
            "prior_regimen": self.prior_regimen,
            "conditions": [c.to_dict() for c in self.conditions],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "source_version": self.source_version,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResistanceRule":
        return cls(
            rule_id=str(data.get("rule_id", "") or data.get("id", "")),
            name=str(data.get("name", "") or ""),
            gene=_opt_str(data.get("gene")),
            prior_regimen=_opt_str(data.get("prior_regimen")),
            conditions=[
                RuleCondition.from_dict(c) for c in (data.get("conditions") or [])
            ],
            recommendations=[
                RuleRecommendation.from_dict(r) for r in (data.get("recommendations") or [])
            ],
            source_version=str(data.get("source_version", "") or ""),
            updated_at=_opt_str(data.get("updated_at")),
            extra=dict(data.get("extra") or {}),
        )


def _opt_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v)
    return s if s else None
