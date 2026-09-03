"""规则匹配 / 求值 / 推荐机制。

GuidelineEngine 只依赖 RuleProvider 与纯数据模型（models.py），
不依赖任何 ORM / 数据库连接 —— 任何实现 RuleProvider 接口的数据源
（文件 / 数据库 / 远端同步）都可无缝接入。
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from groundedrag.guardrail.models import (
    GRADE_ORDER,
    ResistanceRule,
    Rule,
    RuleCondition,
    RuleDecision,
)

_LINE_KEYWORDS = {
    "一线": "一线", "1线": "一线", "初始治疗": "一线", "初始": "一线",
    "二线": "二线", "2线": "二线", "后线治疗": "后线", "后线": "后线",
    "三线": "三线", "3线": "三线",
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
        """从问题文本启发式提取 (cancer_type, treatment_line, biomarker)。

        启发式规则：
        - 治疗线次：优先命中 `一线/二线/后线` 关键词
        - 癌种 / 标志物：在规则条件取值中收集候选词，取在问题里出现的最长匹配
        - 提供 synonym_map 时可把别名归一到规范名再匹配
        说明：启发式仅用于 demo / CLI；评测与测试请直接传入结构化 context。
        """
        context: Dict[str, str] = {}
        # 线次
        for kw, line in _LINE_KEYWORDS.items():
            if kw in question:
                context["treatment_line"] = line
                break
        # 候选词表：所有规则的 domain + conditions 取值
        candidates: Dict[str, str] = {}  # field -> 规范候选词
        for rule in self.pathway_rules:
            if rule.cancer_type:
                candidates.setdefault("cancer_type", rule.cancer_type)
            if rule.biomarker:
                candidates.setdefault("biomarker", rule.biomarker)
        # 并入 conditions 里 cancer_type/biomarker 字段
        for rule in self.pathway_rules:
            for cond in rule.conditions:
                if cond.field in ("cancer_type", "biomarker", "gene") and isinstance(cond.value, str):
                    candidates.setdefault(cond.field, cond.value)
                elif cond.field in ("cancer_type", "biomarker", "gene") and isinstance(cond.value, list):
                    candidates.setdefault(cond.field, cond.value[0])

        synonyms: Dict[str, str] = {}
        for canonical, aliases in (synonym_map or {}).items():
            for alias in aliases:
                synonyms[alias] = canonical

        def pick(canonical: str) -> bool:
            """问题文本是否出现规范词或其任一别名（取任一命中即可）。"""
            names = [canonical] + [a for a, c in synonyms.items() if c == canonical]
            return any(n and n in question for n in names)

        if "cancer_type" in candidates and pick(candidates["cancer_type"]):
            context["cancer_type"] = candidates["cancer_type"]
        if "biomarker" in candidates and pick(candidates["biomarker"]):
            context["biomarker"] = candidates["biomarker"]
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
