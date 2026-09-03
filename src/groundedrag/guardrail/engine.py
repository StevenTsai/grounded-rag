"""规则匹配 / 求值 / 推荐机制。

GuidelineEngine 只依赖 RuleProvider 与纯数据模型（models.py），
不依赖任何 ORM / 数据库连接 —— 任何实现 RuleProvider 接口的数据源
（文件 / 数据库 / 远端同步）都可无缝接入。
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Sequence

from groundedrag.guardrail.models import (
    GRADE_ORDER,
    LINE_ALIASES,
    ResistanceRule,
    Rule,
    RuleCondition,
    RuleDecision,
    normalize_line,
)

# 问题中识别治疗线次的触发词 —— 直接取 models.LINE_ALIASES 的键集，
# 与归一表同源，避免"检测词表 / 归一词表"双份漂移（长短语如 初始治疗/后线治疗
# 必须能被 normalize_line 归一，否则上下文里残留原文、线次条件永不命中）。
_LINE_KEYWORDS = tuple(LINE_ALIASES)
# 耐药 / 进展语境词：命中说明问的是"前序方案失败后"而非"初始该线次"
_PROGRESSION_TERMS = ("耐药", "进展", "继发", "复发", "后线", "失败后")
# biomarker/gene 后缀 → 中文极性词的对应（用于"RAS 野生型"这类问题命中 "RAS-WT" 候选）
_POLARITY_BY_SUFFIX = {
    "-WT": ("野生", "野生型"),
    "-MT": ("突变", "突变型"),
    "-POS": ("阳性",),
    "-NEG": ("阴性",),
}


def _is_match(op: str, ctx_value: object, cond_value: object) -> bool:
    """单个条件求值。

    eq:      相等（忽略大小写、首尾空白）
    neq:     不相等
    in:      上下文值 ∈ 条件取值列表；或条件值子串出现在上下文值
    contains:条件取值列表任一成员是上下文值的子串（或反向）
    all:     条件取值列表全部成员都出现于上下文值
    """
    if ctx_value is None:
        return False
    if isinstance(ctx_value, (list, tuple, set)):
        # 多值上下文：任一成员满足即视为命中
        return any(_is_match(op, v, cond_value) for v in ctx_value)
    ctx_str = str(ctx_value).strip()
    if not ctx_str:
        return False

    def norm(s: str) -> str:
        return s.strip().lower()

    cond_str = norm(str(cond_value)) if cond_value is not None else ""
    if op == "eq":
        return norm(ctx_str) == cond_str
    if op == "neq":
        return norm(ctx_str) != cond_str
    if isinstance(cond_value, (list, tuple, set)):
        members = [norm(str(x)) for x in cond_value]
        if op == "in":
            return norm(ctx_str) in members
        if op == "contains":
            return any(m and m in norm(ctx_str) for m in members)
        if op == "all":
            return all(m and m in norm(ctx_str) for m in members)
        return False
    # 标量条件值
    if op == "in":
        return cond_str in norm(ctx_str) or norm(ctx_str) in cond_str
    if op == "contains":
        return cond_str in norm(ctx_str) or norm(ctx_str) in cond_str
    return False


def _conditions_match(conditions: Sequence[RuleCondition], context: Mapping[str, object]) -> bool:
    for cond in conditions:
        ctx_value = context.get(cond.field)
        if not _is_match(cond.op, ctx_value, cond.value):
            return False
    return True


# ---------------------------------------------------------------------------
# 自由文本上下文抽取辅助（启发式：仅 demo / CLI；评测请传结构化 context）
# ---------------------------------------------------------------------------
def _alnum_tokens(text: str) -> set[str]:
    """取文本中的字母/数字 token（小写），供"候选词的全部字母都出现在问题里"判定。"""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _candidate_hit(candidate: str, names: Sequence[str], question: str) -> tuple[bool, str]:
    """候选词在问题中的命中强度。

    返回 (命中, 命中的名字)。规则：
    1. 规范名/别名的原文子串命中 → 强命中，取最长者；
    2. 规范名的全部字母数字 token 都出现在问题里（如 "EGFR-T790M" vs "EGFR T790M"）→ 弱命中；
    3. 带极性后缀的候选（RAS-WT / RAS-MT 等），核心字母命中且问题含对应中文极性词 → 强命中。
    """
    q = question
    # 1) 原文子串（含别名），最长优先
    best_name = ""
    for n in names:
        if n and n in q and len(n) > len(best_name):
            best_name = n
    if best_name:
        return True, best_name
    # 2) 全部字母数字 token 都出现（容忍连字符/空格差异）
    c_upper = candidate.upper()
    toks = _alnum_tokens(candidate)
    if toks and toks.issubset(_alnum_tokens(q)):
        return True, candidate
    # 3) 极性后缀：如 "RAS-WT"，问题写 "RAS 野生型"
    for suffix, polarities in _POLARITY_BY_SUFFIX.items():
        if c_upper.endswith(suffix):
            core = candidate[: -len(suffix)].lower()
            core_toks = _alnum_tokens(core)
            if core_toks and core_toks.issubset(_alnum_tokens(q)) and any(p in q for p in polarities):
                return True, candidate
    return False, ""


def _pick_candidate(field_candidates: Sequence[str], aliases: Mapping[str, str], question: str) -> Optional[str]:
    """在某个 field 的候选词里，取在问题中出现的最合适一个。

    优先级：强命中（子串）内最长 → 弱命中（token 全含）内最长。
    """
    strong: list[tuple[str, str]] = []
    weak: list[tuple[str, str]] = []
    for cand in field_candidates:
        names = [cand] + [a for a, c in aliases.items() if c == cand]
        hit, matched = _candidate_hit(cand, names, question)
        if not hit:
            continue
        # 命中方式是子串 → 强；否则弱
        if matched in [cand] + [a for a, c in aliases.items() if c == cand] and matched in question:
            strong.append((matched, cand))
        else:
            weak.append((matched, cand))
    if strong:
        strong.sort(key=lambda t: len(t[0]), reverse=True)
        return strong[0][1]
    if weak:
        weak.sort(key=lambda t: len(t[0]), reverse=True)
        return weak[0][1]
    return None


class GuidelineEngine:
    """诊疗路径规则引擎（纯 Python，无 ORM）。

    参数：
        pathway_rules: 诊疗路径规则列表（可由 RuleProvider 加载）
        resistance_rules: 耐药规则列表
        rules_provider: 可选；若给出则优先从 provider 加载
    """

    def __init__(
        self,
        pathway_rules: Optional[Sequence[Rule]] = None,
        resistance_rules: Optional[Sequence[ResistanceRule]] = None,
        *,
        rules_provider=None,
    ) -> None:
        if rules_provider is not None:
            pathway_rules = rules_provider.load_pathway_rules()
            resistance_rules = rules_provider.load_resistance_rules()
        self.pathway_rules: List[Rule] = list(pathway_rules or [])
        self.resistance_rules: List[ResistanceRule] = list(resistance_rules or [])

    # ------------------------------------------------------------------
    # 上下文构造辅助（demo / CLI 用：从自然语言问题提取可匹配条件）
    # ------------------------------------------------------------------
    def extract_context(self, question: str, synonym_map: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
        """从问题文本启发式提取 (cancer_type, treatment_line, biomarker / gene)。

        启发式规则：
        - 治疗线次：命中 `一线/二线/后线` 关键词并按 models.normalize_line 归一
        - 癌种 / 标志物 / 基因：收集**全部**路径规则与耐药规则的 domain + conditions 取值
          作为每字段候选集，取问题里出现的最长匹配（含别名 / token 全含 / 极性后缀映射）
        - 问句含耐药/进展语境且能命中基因候选 → 只给 (cancer_type, gene)，
          并**抑制治疗线次与一线路径 biomarker**，避免"一线进展后"被误配到初始一线规则
        说明：启发式仅用于 demo / CLI；评测与测试请直接传入结构化 context。
        """
        context: Dict[str, str] = {}
        q = question or ""
        # 1) 治疗线次
        for kw in _LINE_KEYWORDS:
            if kw in q:
                context["treatment_line"] = normalize_line(kw)
                break

        # 2) 候选词表（field -> 候选规范词列表；路径 + 耐药 + conditions 全部并入）
        candidates: Dict[str, List[str]] = {
            "cancer_type": [], "biomarker": [], "gene": [],
        }
        for rule in list(self.pathway_rules) + list(self.resistance_rules):
            values: Dict[str, object] = {
                "cancer_type": getattr(rule, "cancer_type", None),
                "biomarker": getattr(rule, "biomarker", None),
                "gene": getattr(rule, "gene", None),
            }
            for field, val in values.items():
                if isinstance(val, (list, tuple)) and val:
                    val = val[0]
                if isinstance(val, str) and val and val not in candidates[field]:
                    candidates[field].append(val)
            for cond in rule.conditions:
                if cond.field in candidates:
                    val = cond.value
                    if isinstance(val, (list, tuple)) and val:
                        val = val[0]
                    if isinstance(val, str) and val and val not in candidates[cond.field]:
                        candidates[cond.field].append(val)

        # 3) 别名表
        synonyms: Dict[str, str] = {}
        for canonical, aliases in (synonym_map or {}).items():
            for alias in aliases:
                synonyms[alias] = canonical

        progressed = any(t in q for t in _PROGRESSION_TERMS)
        gene = _pick_candidate(candidates["gene"], synonyms, q)
        cancer = _pick_candidate(candidates["cancer_type"], synonyms, q)
        # 耐药/进展语境 → 只留 (cancer_type, gene)，抑制一线路径匹配
        # （"一线进展后 T790M" 不应命中初始一线规则，应交由耐药规则裁定）
        if progressed and gene:
            if cancer:
                context["cancer_type"] = cancer
            context["gene"] = gene
            context["biomarker"] = gene
            context.pop("treatment_line", None)
            return context

        if cancer:
            context["cancer_type"] = cancer
        biom = _pick_candidate(candidates["biomarker"], synonyms, q)
        if biom:
            context["biomarker"] = biom
        # 非常规语境：问题只提到基因/耐药子串但无进展词（如 "T790M 阳性二线"），
        # 把基因也带上，让耐药规则有机会命中；不抑制线次。
        if gene and gene not in candidates["biomarker"] and "biomarker" not in context:
            context["gene"] = gene
        return context

    # ------------------------------------------------------------------
    # 路径规则匹配
    # ------------------------------------------------------------------
    def match_pathway(self, context: Mapping[str, object]) -> List[RuleDecision]:
        """返回所有命中的路径规则决策（按规则声明顺序稳定排序）。"""
        decisions: List[RuleDecision] = []
        ctx = dict(context)
        for rule in self.pathway_rules:
            if not _conditions_match(rule.conditions, ctx):
                continue
            matched = [str(c) for c in rule.conditions]
            decisions.append(
                RuleDecision(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    matched_conditions=matched,
                    recommendations=list(rule.recommendations),
                    source_version=rule.source_version,
                    updated_at=rule.updated_at,
                    cancer_type=rule.cancer_type,
                    treatment_line=rule.treatment_line,
                    biomarker=rule.biomarker,
                )
            )
        return decisions

    def match_resistance(self, context: Mapping[str, object]) -> List[RuleDecision]:
        """耐药规则匹配：将 ResistanceRule 归一为 RuleDecision 返回。"""
        decisions: List[RuleDecision] = []
        ctx = dict(context)
        for r in self.resistance_rules:
            ok = True
            for cond in r.conditions:
                ctx_value = ctx.get(cond.field)
                if not _is_match(cond.op, ctx_value, cond.value):
                    ok = False
                    break
            if not ok:
                continue
            decisions.append(
                RuleDecision(
                    rule_id=r.rule_id,
                    rule_name=r.name,
                    matched_conditions=[str(c) for c in r.conditions],
                    recommendations=list(r.recommendations),
                    source_version=r.source_version,
                    updated_at=r.updated_at,
                    biomarker=r.gene,
                )
            )
        return decisions

    def match(self, context: Mapping[str, object]) -> List[RuleDecision]:
        """合并返回路径 + 耐药命中。"""
        return self.match_pathway(context) + self.match_resistance(context)

    # ------------------------------------------------------------------
    # 排序/去重辅助
    # ------------------------------------------------------------------
    @staticmethod
    def sort_by_grade(decisions: Sequence[RuleDecision]) -> List[RuleDecision]:
        """按最高证据等级降序（A 最高）。"""
        return sorted(
            decisions,
            key=lambda d: GRADE_ORDER.get(d.max_grade, 0),
            reverse=True,
        )
