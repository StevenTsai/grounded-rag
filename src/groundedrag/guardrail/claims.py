"""AnswerClaim 数据模型与解析。

LLM 被要求以结构化块输出主张及其引用锚点（如 ``[证据1]`` / ``[规则1]``），
由本模块解析为 :class:`AnswerClaim` 列表，再交给 verifier 逐条校验。

⚠️ 类型语义：``type`` 只是 LLM 提示信号 —— verifier 在进入决策表前会用
确定性表层信号**重判并覆盖**（防"自报绕过"，见设计文档 §3.4.1）。
本模块提供 :func:`classify_claim_type` 作为权威表层信号分类函数，
解析标注与 verifier 覆盖共用，避免两处规则漂移。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from groundedrag.retriever.bm25 import tokenize

# 主张类型（与设计文档 schema 一致）
TYPE_FACTUAL = "factual"
TYPE_NEGATION = "negation"
TYPE_COMPARISON = "comparison"
TYPE_CAUSAL = "causal"
TYPE_INDICATION = "indication"

ALL_TYPES = (TYPE_FACTUAL, TYPE_NEGATION, TYPE_COMPARISON, TYPE_CAUSAL, TYPE_INDICATION)

# ---------------------------------------------------------------------------
# 表层信号词表（verifier 确定性覆盖用；纯字符串集合，可审计）
# ---------------------------------------------------------------------------
# 否定/禁忌信号
NEGATION_WORDS = [
    "不可", "不能", "不应", "不推荐", "不建议", "不适用", "禁忌", "禁用",
    "慎用", "避免", "禁止", "切勿", "勿用", "不得",
]
# 比较/方向信号（一线/二线/三线/后线等**线次定位**词不在此列：
# "X 为一线标准治疗"是可用表面要素核验的定位主张，而非方向性比较；
# 线次词仍是主张表面要素与关键性提示，见 CRITICAL_HINTS。）
COMPARISON_WORDS = [
    "优于", "差于", "劣于", "首选", "先用", "替代", "更优",
]
# 因果/机制信号
CAUSAL_WORDS = [
    "导致", "激活", "抑制", "引起", "由于", "促使", "诱导", "上调", "下调",
    "过表达",
]
# 治疗建议句式信号（命中 → 至少 indication，不可降级为纯事实型）
INDICATION_WORDS = ["推荐", "建议使用", "使用", "治疗", "用药", "适用于", "应用", "给予", "方案为"]
# 关键主张提示词（治疗/剂量/适应证/线次定位/比较/禁忌 → 未通过校验需拒答）
CRITICAL_HINTS = [
    "剂量", "适应症", "适应证", "推荐", "一线", "二线", "三线", "后线",
    "禁忌", "禁用", "给药",
]


def classify_claim_type(text: str) -> str:
    """确定性表层信号 → 主张类型（LLM 标注将被覆盖）。

    覆盖优先级：negation > comparison > causal > indication > factual。
    - 命中否定/禁忌词 → negation（强制）
    - 命中比较/方向/线次词 → comparison
    - 命中因果/机制词 → causal
    - 治疗建议句式 → 至少 indication
    - 无任何上述信号 → factual（纯事实型）
    """
    if not text:
        return TYPE_FACTUAL
    t = text.strip()
    for w in NEGATION_WORDS:
        if w in t:
            return TYPE_NEGATION
    for w in COMPARISON_WORDS:
        if w in t:
            return TYPE_COMPARISON
    for w in CAUSAL_WORDS:
        if w in t:
            return TYPE_CAUSAL
    for w in INDICATION_WORDS:
        if w in t:
            return TYPE_INDICATION
    return TYPE_FACTUAL


def is_critical_claim(text: str, claim_type: str) -> bool:
    """主张是否关键（治疗/剂量/适应证/比较/禁忌）。

    关键主张未通过校验 → 拒答；非关键（背景/机制/预后）→ 标注降级展示。
    """
    if claim_type in (TYPE_NEGATION, TYPE_COMPARISON):
        return True
    low = text.lower()
    for w in CRITICAL_HINTS:
        if w in low:
            return True
    return False


# ---------------------------------------------------------------------------
# AnswerClaim
# ---------------------------------------------------------------------------
@dataclass
class AnswerClaim:
    """原子主张。字段语义与设计文档 schema 完全一致；verifier 只读这些字段。"""

    text: str
    type: str = TYPE_FACTUAL          # LLM 提示信号，verifier 会确定性覆盖
    evidence_refs: List[str] = field(default_factory=list)
    rule_refs: List[str] = field(default_factory=list)
    critical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "evidence_refs": list(self.evidence_refs),
            "rule_refs": list(self.rule_refs),
            "critical": self.critical,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnswerClaim":
        text = str(data.get("text", "")).strip()
        typ = str(data.get("type", TYPE_FACTUAL) or TYPE_FACTUAL)
        if typ not in ALL_TYPES:
            typ = classify_claim_type(text)
        return cls(
            text=text,
            type=typ,
            evidence_refs=[
                str(x) for x in (data.get("evidence_refs") or data.get("evidence") or [])
            ],
            rule_refs=[str(x) for x in (data.get("rule_refs") or data.get("rules") or [])],
            critical=bool(data.get("critical", False)),
        )


# ---------------------------------------------------------------------------
# 解析：结构化块 → AnswerClaim[]
# ---------------------------------------------------------------------------
_ANCHOR_RE = re.compile(r"\[(证据|规则|evidence|rule)\s*[:：]?\s*([0-9A-Za-z_-]+)\]")


def parse_json_claims(
    text: str,
    *,
    evidence_ids: Optional[Sequence[str]] = None,
    rule_ids: Optional[Sequence[str]] = None,
) -> List[AnswerClaim]:
    """解析 JSON 结构化块。支持 ``{"claims":[...]}`` 或裸 ``[...]``。"""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        payload = json.loads(s)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        raw_list = payload.get("claims", payload.get("answer_claims", []))
    elif isinstance(payload, list):
        raw_list = payload
    else:
        return []
    if not isinstance(raw_list, list):
        return []
    claims: List[AnswerClaim] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        claim = AnswerClaim.from_dict(item)
        if not claim.text:
            continue
        claim.evidence_refs = _resolve_positional(claim.evidence_refs, evidence_ids)
        claim.rule_refs = _resolve_positional(claim.rule_refs, rule_ids)
        claims.append(claim)
    return claims


def _resolve_positional(refs: List[str], pool: Optional[Sequence[str]]) -> List[str]:
    if not pool:
        return refs
    resolved: List[str] = []
    for r in refs:
        if r.isdigit():
            idx = int(r) - 1
            if 0 <= idx < len(pool):
                resolved.append(pool[idx])
            else:
                resolved.append(r)
        else:
            resolved.append(r)
    return resolved


def parse_anchor_text(
    text: str,
    *,
    evidence_ids: Optional[Sequence[str]] = None,
    rule_ids: Optional[Sequence[str]] = None,
) -> List[AnswerClaim]:
    """解析文本行式结构化块。

    约定格式（每行一条主张）：
        ``主张文本[证据1][规则2]``
    ``[证据n]``/``[规则n]`` 的 n 为证据/规则列表位置（1 起），或直接写 id。
    无锚点的裸句 → 作为"无引用主张"交给 verifier 判不完整。
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    claims: List[AnswerClaim] = []
    for line in lines:
        if line.startswith("#") or line.startswith("//"):
            continue
        if "证据" not in line and "规则" not in line and "[evidence" not in line:
            claims.append(AnswerClaim(text=line))
            continue
        anchors = _ANCHOR_RE.findall(line)
        ev_refs: List[str] = []
        rule_refs: List[str] = []
        for kind, val in anchors:
            if kind in ("证据", "evidence"):
                if val.isdigit() and evidence_ids:
                    idx = int(val) - 1
                    if 0 <= idx < len(evidence_ids):
                        ev_refs.append(evidence_ids[idx])
                else:
                    ev_refs.append(val)
            elif kind in ("规则", "rule"):
                if val.isdigit() and rule_ids:
                    idx = int(val) - 1
                    if 0 <= idx < len(rule_ids):
                        rule_refs.append(rule_ids[idx])
                else:
                    rule_refs.append(val)
        clean = _ANCHOR_RE.sub("", line).strip(" \t-•*")
        if clean:
            claims.append(AnswerClaim(text=clean, evidence_refs=ev_refs, rule_refs=rule_refs))
    return claims


def parse_claims(
    text: str,
    *,
    evidence_ids: Optional[Sequence[str]] = None,
    rule_ids: Optional[Sequence[str]] = None,
) -> List[AnswerClaim]:
    """统一解析入口：优先 JSON，退化到锚点文本行。"""
    if text.strip().startswith(("{", "[")):
        claims = parse_json_claims(text, evidence_ids=evidence_ids, rule_ids=rule_ids)
        if claims:
            return claims
    return parse_anchor_text(text, evidence_ids=evidence_ids, rule_ids=rule_ids)


def surface_tokens(text: str) -> List[str]:
    """提取主张中的可核查表面要素 token（实体/数字/单位）。

    verifier 的表面要素一致性比对以本函数为准：过滤短词/纯功能词，
    保留药品、标志物、癌种等实体词与数字。纯功能词（如"推荐/使用/治疗/
    应该"）整词由常用字构成，在主张与证据两侧都会出现，不携带可核查信息，
    剔除后可避免以通用词凑够命中比例的误放行。
    """
    toks = tokenize(text)
    # 常用功能字（逐字判定；整词全由这些字组成则视为功能词）
    stop = set("的了在和与及对于并且通过可能可以应该建议推荐使用治疗给予用药进行需要相关")
    out: List[str] = []
    for t in toks:
        if t.isdigit():
            out.append(t)
            continue
        if len(t) < 2:
            continue
        if all(ch in stop for ch in t):
            continue
        out.append(t)
    return out
