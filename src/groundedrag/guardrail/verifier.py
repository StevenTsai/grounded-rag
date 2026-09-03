"""★声明级校验门（guardrail/verifier.py）。

回答生成后不是直接返回，而是逐条经过本模块：**引用完整性 / 主张-证据一致性 /
规则冲突 / 证据充分性** 四项校验（对应设计文档 §3.4）。全部为纯逻辑判定，
零 LLM 依赖，可直接被单测覆盖。

能力边界（对外口径）：
- 确定性档只做**表面要素一致性**（实体/数值/单位是否出现在证据 text_span），
  不做"真实支持"判定；否定/比较/因果等关系型主张在语义档关闭时一律拒答或标注。
- 语义档（NLI/LLM 蕴含）为可选增强，启用时输出标注「语义核验」，不静默混入。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

from groundedrag.guardrail.claims import (
    TYPE_CAUSAL,
    TYPE_COMPARISON,
    TYPE_FACTUAL,
    TYPE_NEGATION,
    AnswerClaim,
    classify_claim_type,
    is_critical_claim,
    surface_tokens,
)
from groundedrag.guardrail.evidence import (
    EvidenceId,
    EvidenceRegistry,
    StalenessPolicy,
    parse_iso_date,
)
from groundedrag.guardrail.models import GRADE_ORDER, RuleDecision

# 判定状态
PASS = "pass"
REFUSE = "refuse"          # 拒答（关键主张失败或无法验证）
ANNOTATE = "annotate"      # 标注降级（非关键主张，展示时附"未经本地证据核验"）

# 数字/剂量模式：剂量主张要求证据 span 内含同一数值
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_DOSE_RE = re.compile(r"\d+(?:\.\d+)?\s*(mg|ml|g|片|粒|支|天|日|周|次|疗程|月|年)")
# 证据侧表层否定信号（用于"证据说不可用、主张说推荐 A"这类翻转的确定性拦截）
_EVIDENCE_NEGATION = [
    "不可", "不能", "不应", "不推荐", "不建议", "禁忌", "禁用", "慎用",
    "避免", "禁止", "切勿", "不得", "不作为", "不使用",
]
# 正向推荐动词（主张为正例时才触发证据否定检查）
_POSITIVE_VERBS = ["推荐", "建议使用", "使用", "治疗", "用药", "给予", "给药", "应用", "可用于", "适用于"]


# ---------------------------------------------------------------------------
# 语义档（可选增强）接口
# ---------------------------------------------------------------------------
class SemanticSupportVerifier(Protocol):
    """语义档校验器：判定主张是否被证据真实支持（NLI/LLM 蕴含）。"""

    def check(self, claim: AnswerClaim, evidences: Sequence[EvidenceId]) -> Optional[bool]:
        """返回 True=支持 / False=不支持 / None=无法判定（弃权）。"""
        ...


def _kind_of(text: str) -> str:
    """主张"子类型"（治疗/剂量/关联/背景），供充分性门槛使用。"""
    low = text.lower()
    if _DOSE_RE.search(text) or any(h in low for h in ("剂量", "用量", "每日", "给药", "mg")):
        return "dose"
    if any(w in low for w in ("推荐", "治疗", "使用", "用药", "给予", "给药", "方案为", "适应证", "适应症", "应用")):
        return "treatment"
    if any(w in low for w in ("相关", "关联", "阳性", "表达", "突变", "标志物", "预后", "生存", "风险")):
        return "association"
    return "background"


def _claim_is_dose(text: str) -> bool:
    return _DOSE_RE.search(text) is not None


def _extract_numbers(text: str) -> List[str]:
    return [m.group() for m in _NUMBER_RE.finditer(text)]


@dataclass
class ClaimVerdict:
    """单条主张的校验结果。"""

    claim: AnswerClaim
    status: str = ANNOTATE
    reason: str = ""
    message: str = ""
    overridden_type: str = TYPE_FACTUAL
    kind: str = "background"
    effective_critical: bool = False
    checks: Dict[str, Dict[str, object]] = field(default_factory=dict)
    evidence_used: List[str] = field(default_factory=list)
    semantic_used: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.claim.text,
            "status": self.status,
            "reason": self.reason,
            "message": self.message,
            "overridden_type": self.overridden_type,
            "kind": self.kind,
            "critical": self.effective_critical,
            "checks": self.checks,
            "evidence_used": self.evidence_used,
            "semantic_used": self.semantic_used,
        }


@dataclass
class VerifyReport:
    """一次回答的整体验证报告。"""

    verdicts: List[ClaimVerdict] = field(default_factory=list)
    overall_status: str = PASS
    overall_reason: str = ""
    matched_rules: List[RuleDecision] = field(default_factory=list)
    message: str = ""

    @property
    def any_refused(self) -> bool:
        return any(v.status == REFUSE for v in self.verdicts)

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall_status": self.overall_status,
            "overall_reason": self.overall_reason,
            "message": self.message,
            "claim_verdicts": [v.to_dict() for v in self.verdicts],
            "matched_rules": [r.to_dict() for r in self.matched_rules],
        }


@dataclass
class VerifierConfig:
    """校验门配置。"""

    element_ratio: float = 0.5            # 表面要素命中比例阈值（0~1）
    semantic_enabled: bool = False
    staleness_years: Optional[Dict[str, int]] = None
    semantic_verifier: Optional[SemanticSupportVerifier] = None
    synonym_map: Optional[Dict[str, List[str]]] = None  # 别名归一（可选，用于要素比对）

    def staleness(self) -> StalenessPolicy:
        return StalenessPolicy(years=self.staleness_years)


# ---------------------------------------------------------------------------
# 规则冲突裁定
# ---------------------------------------------------------------------------
def _rules_conflict(a: RuleDecision, b: RuleDecision) -> bool:
    """两条命中规则是否冲突（设计文档 §3.4.3 可编码定义）。

    当且仅当同时满足：① 同条件域（癌种+线次+标志物/基因）；② 主推方案集合互斥
    （组件集合不相交）；③ 两者均非 stale（stale 过滤在裁定前完成，此处仅判定域与方案）。
    """
    same_domain = (
        (a.cancer_type == b.cancer_type or a.cancer_type is None or b.cancer_type is None)
        and (a.treatment_line == b.treatment_line or a.treatment_line is None or b.treatment_line is None)
        and (a.biomarker == b.biomarker or a.biomarker is None or b.biomarker is None)
    )
    if not same_domain:
        return False
    pa = set(a.plan_components)
    pb = set(b.plan_components)
    if not pa or not pb:
        return False
    return pa.isdisjoint(pb)  # 互斥 = 组件集合不相交


def arbitrate_conflict(
    decisions: Sequence[RuleDecision],
    policy: StalenessPolicy,
    now=None,
) -> tuple[List[RuleDecision], List[str]]:
    """冲突裁定：过滤 stale → 最高 grade → 最新 updated_at → 声明分歧。

    返回 (胜出者列表, 说明列表)。胜出者为空表示"声明分歧，不静默二选一"。
    """
    # 仅对存在实际冲突对的集合进行裁定
    notes: List[str] = []
    pool = list(decisions)
    # 1) 过滤 stale：规则视为 guideline 类时效
    fresh: List[RuleDecision] = []
    for d in pool:
        ev = EvidenceId(
            evidence_id=d.rule_id,
            doc_id=d.rule_id,
            source_type="guideline",
            source_version=d.source_version,
            grade=d.max_grade,
            updated_at=d.updated_at,
            text_span="",
        )
        if not policy.is_stale(ev):
            fresh.append(d)
    if len(fresh) < len(pool):
        notes.append(f"已过滤 {len(pool) - len(fresh)} 条过期规则")
    if not fresh:
        return [], notes + ["全部命中规则已过期，无有效规则"]
    # 2) 按最高 grade
    best_grade = max(GRADE_ORDER.get(d.max_grade, 0) for d in fresh)
    graded = [d for d in fresh if GRADE_ORDER.get(d.max_grade, 0) == best_grade]
    # 3) grade 相同 → 最新 updated_at（缺失视为未知，排在有时间戳者之后）
    def _ts(d: RuleDecision) -> tuple[int, int]:
        dt = parse_iso_date(d.updated_at)
        if dt is None:
            return (0, 0)
        return (1, dt.toordinal())

    graded.sort(key=_ts, reverse=True)
    newest_ts = _ts(graded[0])
    newest = [d for d in graded if _ts(d) == newest_ts]
    # 4) 仍无法裁决且内容矛盾 → 声明分歧
    if len(newest) > 1 and any(_rules_conflict(newest[0], x) for x in newest[1:]):
        return [], notes + ["两种推荐均存在且等级/时间相同，请结合临床判断（分歧声明）"]
    return newest, notes


# ---------------------------------------------------------------------------
# 校验门主体
# ---------------------------------------------------------------------------
class Verifier:
    """声明级校验门：逐条验证 AnswerClaim。"""

    def __init__(self, config: Optional[VerifierConfig] = None) -> None:
        self.config = config or VerifierConfig()
        self._policy = self.config.staleness()

    # -- 要素一致（确定性档） ----------------------------------------------
    def _normalize_token(self, tok: str) -> str:
        """可选别名归一（同义词词典注入）。"""
        if not self.config.synonym_map:
            return tok
        for canonical, aliases in self.config.synonym_map.items():
            if tok in aliases:
                return canonical
        return tok

    def _evidence_tokens(self, span: str) -> set[str]:
        return {self._normalize_token(t) for t in surface_tokens(span)}

    def _evidence_negates_claim(self, span: str, claim_toks: set[str]) -> bool:
        """单条证据是否在语义上否定主张的实体。

        判定：对主张每个实体 token 在证据里的出现位置，取最近的关系词
        （否定词 / 正向动词）。若最近的是否定词 → 认为证据对该实体"禁用"。

        这样处理临床里最常见的写法：证据同时写"不应使用 A，推荐使用 B"时，
        B 附近的最近词是正向动词，不会被 A 的否定误伤（旧实现把整篇一票否决）。
        """
        low = (span or "").lower()
        markers = []  # (index, kind)
        for w in _EVIDENCE_NEGATION:
            start = 0
            while True:
                i = low.find(w, start)
                if i < 0:
                    break
                markers.append((i, "neg"))
                start = i + 1
        for w in _POSITIVE_VERBS:
            start = 0
            while True:
                i = low.find(w, start)
                if i < 0:
                    break
                markers.append((i, "pos"))
                start = i + 1
        if not markers:
            return False
        # 只考虑出现在主张实体附近的词（窗口），跨段落的否定不误伤
        window = 24
        for tok in claim_toks:
            if any(ch.isdigit() for ch in tok):
                continue  # 数值 token 不参与否定方向判定
            start = 0
            while True:
                p = low.find(tok.lower(), start)
                if p < 0:
                    break
                near = [(i, k) for i, k in markers if abs(i - p) <= window]
                if near:
                    nearest = min(near, key=lambda x: abs(x[0] - p))
                    if nearest[1] == "neg":
                        return True
                start = p + 1
        return False

    def surface_support(self, claim: AnswerClaim, evidences: Sequence[EvidenceId]) -> Dict[str, object]:
        """确定性档表面要素一致性：主张要素是否出现在**同一条**证据 text_span。

        关键不变量（防跨证据拼装）：
        - 实体要素与数字必须落在**同一条**证据里才算通过 —— 不允许"实体来自文档 A、
          数字来自文档 B、等级来自文档 C"的拼装；
        - 数值 token（含数字）不计入实体比例，避免"80mg"这类 token 抬高命中比。
        """
        claim_toks = {self._normalize_token(t) for t in surface_tokens(claim.text)}
        claim_nums = set(_extract_numbers(claim.text))
        if not claim_toks and not claim_nums:
            return {"passed": False, "ratio": 0.0, "reason": "no_surface_element"}
        entity_toks = {t for t in claim_toks if not any(ch.isdigit() for ch in t)}

        threshold = self.config.element_ratio
        best_ratio = 0.0
        num_ok_any = False
        support_ids: List[str] = []
        for ev in evidences:
            span_toks = self._evidence_tokens(ev.text_span)
            span_nums = set(_extract_numbers(ev.text_span))
            num_ok = (not claim_nums) or claim_nums.issubset(span_nums)
            if num_ok:
                num_ok_any = True
            if entity_toks:
                hit = entity_toks & span_toks
                ent_ratio = len(hit) / len(entity_toks)
            else:
                # 无实体要素（纯数字/背景主张）：数字命中视为表面通过
                ent_ratio = 1.0 if num_ok else 0.0
            best_ratio = max(best_ratio, ent_ratio)
            if ent_ratio >= threshold and num_ok:
                support_ids.append(ev.evidence_id)

        passed = bool(support_ids)
        reason = ""
        # 证据侧否定翻转：正向推荐主张，其**锚定证据**（提供表面命中的那几条）对其实体表达否定
        positive = any(v in (claim.text or "") for v in _POSITIVE_VERBS)
        negated = False
        if passed and positive:
            anchored = [ev for ev in evidences if ev.evidence_id in support_ids]
            negation_flags = [self._evidence_negates_claim(ev.text_span, entity_toks) for ev in anchored]
            if anchored and negation_flags and all(negation_flags):
                support_ids = []
                passed = False
                negated = True

        if negated:
            reason = "evidence_negates_claim"
        elif not passed:
            if claim_nums and not num_ok_any:
                reason = "number_mismatch"
            elif claim_nums:
                reason = "number_mismatch"  # 数字命中了，但与实体不在同一条证据（跨证据拼装）
            elif entity_toks:
                reason = "entity_mismatch"
            else:
                reason = "no_surface_element"
        else:
            reason = "surface_hit"
        return {
            "passed": passed,
            "ratio": round(best_ratio, 3),
            "reason": reason,
            "support_ids": support_ids,
        }

    def _arbitrate_matched(
        self, decisions: Sequence[RuleDecision]
    ) -> tuple[set[str], List[str]]:
        """按条件域分组做冲突裁定，返回 (全局胜出 rule_id 集合, 说明列表)。

        同一问答上下文可能命中多条不同域（癌种/线次/标志物）规则，它们彼此
        不构成冲突；冲突只可能发生在**同域**规则之间（§3.4.3）。因此先按
        (cancer_type, treatment_line, biomarker) 分组，组内各自仲裁，
        再把各组的胜出者并入全局集合。任一组出现分歧（无胜出者）会在说明里标注。
        """
        groups: Dict[tuple, List[RuleDecision]] = {}
        for d in decisions:
            key = (d.cancer_type, d.treatment_line, d.biomarker)
            groups.setdefault(key, []).append(d)
        winner_ids: set[str] = set()
        notes: List[str] = []
        for group in groups.values():
            winners, ns = arbitrate_conflict(group, self._policy)
            notes.extend(ns)
            winner_ids.update(w.rule_id for w in winners)
        return winner_ids, notes

    def _rule_evidence(self, d: RuleDecision) -> EvidenceId:
        """把规则决策临时归一为 EvidenceId，供时效/等级判定与规则直出比对复用。

        ``text_span`` 拼接规则的条件域（癌种/线次/标志物）与推荐方案 ——
        规则直出主张（如「肺癌 一线 EGFR 推荐方案：奥希替尼」）的表面比对
        以该 span 为"权威原文"。
        """
        plans = " / ".join(rec.plan_name for rec in d.recommendations)
        scope = " ".join(
            x for x in (d.cancer_type or "", d.treatment_line or "", d.biomarker or "") if x
        )
        return EvidenceId(
            evidence_id=d.rule_id,
            doc_id=d.rule_id,
            source_type="guideline",
            source_version=d.source_version,
            grade=d.max_grade,
            updated_at=d.updated_at,
            text_span=f"{scope} {plans}".strip(),
        )

    def _matching_rule_evs(
        self, claim: AnswerClaim, rule_evs: Sequence[EvidenceId]
    ) -> List[EvidenceId]:
        """返回内容真正支持主张（过 0.75 高门槛表面比对）的规则侧证据。

        与规则直出同一门槛（``max(0.75, element_ratio)``）——防「只复用规则条件域、
        偷换药物」的挂靠包装：混合引用时，仅当主张确实复述了规则推荐内容，该规则的
        等级才可参与充分性判定。
        """
        support = self.surface_support(claim, rule_evs)
        if not support.get("passed"):
            return []
        ratio = float(support.get("ratio", 0.0))
        if ratio < max(0.75, self.config.element_ratio):
            return []
        ids = set(support.get("support_ids") or [])
        return [r for r in rule_evs if r.evidence_id in ids]

    def _fresh_for(self, ev: EvidenceId, kind: str) -> bool:
        """证据时效是否足以支撑该门槛。

        - dose / treatment：高门槛，要求有明确日期且未过期。**无 updated_at 不豁免**
          （模块口径：缺失时间戳（即便有 source_version）按"未知时效"处理，
          不视为新鲜 A 级证据）。
        - association / background：低门槛，仅过期的不可用；未知时间可用。
        """
        if kind in ("dose", "treatment"):
            if parse_iso_date(ev.updated_at) is None:
                return False
            return not self._policy.is_stale(ev)
        return not self._policy.is_stale(ev)

    def _sufficient(self, verdict: ClaimVerdict, evidences: Sequence[EvidenceId]) -> bool:
        """证据充分性门槛（设计文档 §3.4.3）。返回 False 表示证据不足→拒答/降级。

        ``evidences`` 是**已经通过表面一致性**的锚定证据（由调用方传入），
        充分性只在该子集上判定 —— 不允许拿一条无关高等级证据"凑门槛"。
        """
        text = verdict.claim.text
        kind = verdict.kind
        numbers = set(_extract_numbers(text))
        for ev in evidences:
            rank = ev.grade_rank
            if rank < 0:
                continue
            if not self._fresh_for(ev, kind):
                continue
            if kind == "dose":
                if _claim_is_dose(text) and numbers:
                    if rank >= GRADE_ORDER["C"] and numbers.intersection(_extract_numbers(ev.text_span)):
                        verdict.evidence_used.append(ev.evidence_id)
                        return True
                else:
                    # 主张看起来是剂量但无数字：按低门槛不通过
                    continue
            elif kind == "treatment":
                if rank < GRADE_ORDER["B"]:
                    continue
                verdict.evidence_used.append(ev.evidence_id)
                return True
            elif kind == "association":
                if rank >= GRADE_ORDER["D"]:
                    verdict.evidence_used.append(ev.evidence_id)
                    return True
            else:  # background / mechanism 低门槛
                if rank >= GRADE_ORDER["D"]:
                    verdict.evidence_used.append(ev.evidence_id)
                    return True
        return False

    # -- 单条主张校验 -------------------------------------------------------
    def _verify_claim(
        self,
        claim: AnswerClaim,
        registry: EvidenceRegistry,
        matched_rules: Sequence[RuleDecision],
    ) -> ClaimVerdict:
        verdict = ClaimVerdict(claim=claim)

        # 0) 类型确定性覆盖（LLM 自标不可信）
        verdict.overridden_type = classify_claim_type(claim.text)
        verdict.kind = _kind_of(claim.text)
        # 覆盖后关键性重判
        verdict.effective_critical = claim.critical or is_critical_claim(claim.text, verdict.overridden_type)

        # 1) 引用完整性：证据引用必须可解析且 schema 完整；规则引用必须命中
        ev_refs: List[EvidenceId] = registry.resolve(claim.evidence_refs)
        matched_ids = {m.rule_id for m in matched_rules}
        unresolved_rules = [r for r in claim.rule_refs if r not in matched_ids]
        # 规则直出：主张仅绑定已命中的权威规则（不带任何证据引用）
        rule_only = bool(claim.rule_refs) and not claim.evidence_refs and not unresolved_rules
        incomplete = False
        if not claim.evidence_refs and not claim.rule_refs:
            incomplete = True
        elif len(ev_refs) != len(claim.evidence_refs):
            incomplete = True  # 有引用无法解析
        elif unresolved_rules:
            incomplete = True  # 规则引用未命中任何匹配规则
        else:
            for ev in ev_refs:
                if not ev.schema_complete():
                    incomplete = True
                    break
        verdict.checks["citation"] = {
            "passed": not incomplete,
            "evidence_refs": claim.evidence_refs,
            "rule_refs": claim.rule_refs,
        }
        if incomplete:
            verdict.status = REFUSE if verdict.effective_critical else ANNOTATE
            verdict.reason = "citation_incomplete"
            verdict.message = "该条无引用或引用不完整（无证据/规则锚点）"
            return verdict

        # 被引规则的规则侧证据（其内容即主张要匹配的"权威原文"）。
        # 不仅规则直出（rule_only）需要 —— 混合引用（证据+规则）时规则等级也参与充分性。
        rule_evs: List[EvidenceId] = []
        if claim.rule_refs:
            rule_evs = [
                self._rule_evidence(m)
                for m in matched_rules
                if m.rule_id in claim.rule_refs
            ]

        # 2) 语义档开启时先咨询语义校验器
        if self.config.semantic_enabled and self.config.semantic_verifier is not None:
            semantic = self.config.semantic_verifier.check(claim, ev_refs)
            verdict.semantic_used = True
            if semantic is False:
                verdict.status = REFUSE if verdict.effective_critical else ANNOTATE
                verdict.reason = "semantic_refute"
                verdict.message = "语义核验：主张被绑定证据否定"
                return verdict

        # 3) 关系型主张：确定性档只能判"表面一致但语义未验证"
        #    例外：规则直出（rule_only）——主张内容是命中的权威规则本身，
        #    规则的否定/线次/比较语义由规则的构建方背书，无需再猜方向；
        #    但它仍要过第 4 步"主张 vs 规则内容"的表面比对，防挂靠权威规则包装幻觉。
        if not rule_only and verdict.overridden_type in (TYPE_NEGATION, TYPE_COMPARISON, TYPE_CAUSAL):
            verdict.status = REFUSE if verdict.effective_critical else ANNOTATE
            verdict.reason = "relational_unverifiable"
            verdict.message = (
                "该主张含否定/比较/因果等关系语义，确定性档无法验证方向，"
                "语义档未开启 → 拒答/标注"
            )
            verdict.checks["claim_support"] = {"passed": False, "reason": verdict.reason}
            return verdict

        # 4) 确定性档表面要素一致性（factual / indication）
        #    规则直出主张与「其绑定的规则内容」做表面比对（规则即权威原文），
        #    门槛更高（≈复述规则内容，防止只挂靠规则的条件域、换掉药物的包装幻觉）；
        #    其余主张与证据 text_span 做表面比对。
        if rule_only:
            support = self.surface_support(claim, rule_evs)
            ratio = float(support.get("ratio", 0.0))
            high_bar = max(0.75, self.config.element_ratio)
            if support.get("passed") and ratio < high_bar:
                support = {
                    "passed": False,
                    "ratio": ratio,
                    "reason": "rule_content_mismatch",
                    "support_ids": support.get("support_ids") or [],
                }
            elif support.get("passed"):
                support = {
                    "passed": True,
                    "ratio": ratio,
                    "reason": "rule_authoritative",
                    "support_ids": support.get("support_ids") or [],
                }
        else:
            support = self.surface_support(claim, ev_refs)
        support_ids = support.get("support_ids") or []
        verdict.checks["claim_support"] = support
        if not support["passed"]:
            verdict.status = REFUSE if verdict.effective_critical else ANNOTATE
            verdict.reason = str(support["reason"])
            verdict.message = "表面要素一致性不通过（实体/数值未出现在证据或所引规则中）"
            return verdict

        # 5) 证据充分性门槛 —— 只在**锚定该主张的证据**（通过表面一致的那几条）上判定，
        #    不允许拿一条无关高等级证据"凑门槛"（跨证据拼装：弱文档供字面、强文档供等级）。
        if rule_only:
            anchored: List[EvidenceId] = [r for r in rule_evs if r.evidence_id in support_ids]
        else:
            anchored = [r for r in ev_refs if r.evidence_id in support_ids]
            # 混合引用（证据+规则）：被引规则是显式权威来源，其等级应参与充分性判定；
            # 但须先过「主张 vs 规则内容」的高门槛表面比对 —— 防止挂靠一条 A 级但
            # 内容不符的规则凑门槛（内容不符的规则不进入充分性池，由证据等级单独决定）。
            if rule_evs:
                anchored_ids = {a.evidence_id for a in anchored}
                for re_ in self._matching_rule_evs(claim, rule_evs):
                    if re_.evidence_id not in anchored_ids:
                        anchored.append(re_)
                        anchored_ids.add(re_.evidence_id)
        # 治疗类主张（含规则直出）：先对**完整命中集**做按域分组的冲突裁定
        # （防止只对被引用子集仲裁而漏判"两条规则各被不同主张引用"的跨主张冲突），
        # 再要求主张引用的规则是胜出者，否则拒答。
        # 注意：规则直出主张的表层类型可能被重判为 comparison（主张文本含线次词，
        # 如"三线推荐方案"），但其语义由命中的权威规则背书 —— 冲突裁定依然必须执行，
        # 故不以 overridden_type == indication 为前提，而看是否携带规则引用 + 治疗类。
        if claim.rule_refs and verdict.kind == "treatment":
            winner_ids, arbitral_notes = self._arbitrate_matched(matched_rules)
            referenced = [m for m in matched_rules if m.rule_id in claim.rule_refs]
            if referenced:
                # ① 引用的规则本身已过期 → 直接判负（过期是硬性失效，不是"同级分歧"）
                stale_lost = any(
                    self._policy.is_stale(self._rule_evidence(m)) for m in referenced
                )
                if stale_lost:
                    verdict.status = REFUSE if verdict.effective_critical else ANNOTATE
                    verdict.reason = "rule_conflict_lost"
                    verdict.message = "主张引用的规则已过期，失去效力"
                    return verdict
                # ② 整组无胜出者（同级同时间互斥）→ 分歧声明，不静默二选一
                if not winner_ids and arbitral_notes:
                    verdict.status = REFUSE if verdict.effective_critical else ANNOTATE
                    verdict.reason = "rule_conflict_divergence"
                    verdict.message = "".join(arbitral_notes) or "两种推荐均存在，请结合临床判断"
                    return verdict
                # ③ 有胜出者，但本主张引用的规则被淘汰（更低等级 / 更旧版本）
                if any(m.rule_id not in winner_ids for m in referenced):
                    verdict.status = REFUSE if verdict.effective_critical else ANNOTATE
                    verdict.reason = "rule_conflict_lost"
                    verdict.message = "主张引用的规则在冲突裁定中被淘汰"
                    return verdict

        if not self._sufficient(verdict, anchored):
            verdict.status = REFUSE if verdict.effective_critical else ANNOTATE
            verdict.reason = "insufficient_evidence"
            verdict.message = "证据充分性不足（等级过低/已过期/数值不符）"
            return verdict

        verdict.status = PASS
        verdict.reason = "pass"
        verdict.message = "引用完整 + 表面要素一致 + 证据充分"
        return verdict

    # -- 整体验证 -----------------------------------------------------------
    def verify(
        self,
        claims: Sequence[AnswerClaim],
        registry: EvidenceRegistry,
        matched_rules: Optional[Sequence[RuleDecision]] = None,
    ) -> VerifyReport:
        matched_rules = list(matched_rules or [])
        verdicts = [self._verify_claim(c, registry, matched_rules) for c in claims]

        report = VerifyReport(verdicts=verdicts, matched_rules=matched_rules)
        # 整体状态：存在关键拒答 → refuse；无 claims 且无规则 → refuse；否则 pass/annotate
        refused_critical = [v for v in verdicts if v.status == REFUSE and v.effective_critical]
        if refused_critical:
            report.overall_status = REFUSE
            report.overall_reason = "critical_claim_failed"
            report.message = "存在未通过校验的关键主张（治疗/剂量/适应证/禁忌）"
            return report
        passed = [v for v in verdicts if v.status == PASS]
        if passed:
            report.overall_status = PASS
            report.overall_reason = "all_passed_or_annotated"
            report.message = "回答通过校验门（含标注降级项）"
            return report
        if not verdicts:
            if not matched_rules:
                report.overall_status = REFUSE
                report.overall_reason = "no_evidence_no_rule"
                report.message = "未检索到证据或规则，拒绝作答"
            else:
                report.overall_status = PASS
                report.overall_reason = "rule_only_no_claim"
                report.message = "存在命中规则但无额外主张，交由规则直出"
            return report
        # 全部为 ANNOTATE（非关键）
        report.overall_status = ANNOTATE
        report.overall_reason = "all_annotated"
        report.message = "所有主张均为标注降级（未经本地证据核验）"
        return report
