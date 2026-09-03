#!/usr/bin/env python3
"""CLI 端到端可信问答 demo（对应设计文档 §3.5 核心流程）。

数据流逐段打印，评审/快速自测两用：

    用户问题 → BM25 检索（召回文档）
            → 规则引擎匹配（RuleDecision）
            → 结构化生成（无 API Key 时命中规则走"规则直出"）
            → 声明级校验门（逐条 PASS / ANNOTATE / REFUSE + 原因）
            → 降级 / 拒答后的可信回答

运行：
    python examples/demo.py                       # 内置演示主线
    python examples/demo.py --question "……"       # 自定义问题
    python examples/demo.py --case 3              # 校验门分诊：跑 eval 集某用例的主张

可选真实 LLM（OpenAI 兼容端点；缺省走模板回退 / 规则直出，无需网络）：
    LLM_API_KEY=sk-… LLM_BASE_URL=https://api.deepseek.com/v1 LLM_MODEL=deepseek-chat
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from groundedrag.eval.cases import load_eval_set, parse_case_claims
from groundedrag.guardrail import (
    ANNOTATE,
    PASS,
    REFUSE,
    ClaimVerdict,
)
from groundedrag.llm import FailoverLLM, OpenAICompatibleLLM
from groundedrag.pipeline import Pipeline, PipelineResult

HERE = Path(__file__).resolve().parent
SEED_DOCS = HERE / "seed_docs.jsonl"
SEED_RULES = HERE / "seed_rules.json"
EVAL_SET = HERE / "eval_set.jsonl"

# 逐条校验 reason → 对外中文说明（供校验门面板）
REASON_LABELS: Dict[str, str] = {
    "pass": "校验通过（引用完整 + 表面要素一致 + 证据充分）",
    "citation_incomplete": "无引用或引用不完整（缺证据/规则锚点）",
    "semantic_refute": "语义核验：主张被绑定证据否定",
    "relational_unverifiable": "含否定/比较/因果等关系语义，确定性档无法验证方向",
    "rule_conflict_lost": "主张引用的规则在冲突裁定中被淘汰",
    "rule_conflict_divergence": "同域规则冲突无法裁定，不静默二选一",
    "insufficient_evidence": "证据充分性不足（等级过低 / 已过期 / 数值不符）",
    "rule_content_mismatch": "与所引规则内容不符（疑似挂靠规则包装幻觉）",
    "rule_authoritative": "内容与命中的权威规则一致（规则直出，无需 LLM）",
    "surface_hit": "表面要素一致",
    "no_surface_element": "主张不含可核查的表面要素（实体/数值/单位）",
    "number_mismatch": "数值与证据不符",
    "entity_mismatch": "实体/要素未出现在证据或所引规则中",
    "evidence_negates_claim": "证据侧否定翻转：证据写不可用/禁忌，主张却作正向推荐",
}
STATUS_ICON = {PASS: "✅ PASS", ANNOTATE: "🟡 ANNOTATE", REFUSE: "✗ REFUSE"}
STATUS_TITLE = {
    PASS: "有据可依 · 原样输出",
    ANNOTATE: "未经本地证据核验 · 标注降级",
    REFUSE: "无据可依 · 拒绝输出 / 整段拒答",
}
_TYPE_CN = {
    "factual": "事实",
    "negation": "否定",
    "comparison": "比较",
    "causal": "因果",
    "indication": "治疗建议",
}


# ---------------------------------------------------------------------------
# 管线构造
# ---------------------------------------------------------------------------
def llm_from_env() -> Optional[FailoverLLM]:
    """从环境变量构造真实 LLM；未配置返回 None（demo 走模板回退/规则直出）。"""
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return FailoverLLM(
        [
            OpenAICompatibleLLM(
                base_url=os.getenv(
                    "LLM_BASE_URL", "https://api.deepseek.com/v1"
                ).rstrip("/"),
                api_key=api_key,
                model=os.getenv("LLM_MODEL", "deepseek-chat"),
            )
        ]
    )


def build_pipeline() -> Pipeline:
    return Pipeline.build_from_json(SEED_DOCS, SEED_RULES, llm=llm_from_env())


def load_eval_cases() -> List[Dict[str, Any]]:
    # 读取统一收口到 eval.cases（demo/app/runner 共用）
    return load_eval_set(EVAL_SET)


# ---------------------------------------------------------------------------
# 渲染：校验门面板
# ---------------------------------------------------------------------------
def claim_card(verdict: ClaimVerdict, index: int) -> List[str]:
    """单条主张卡（多行文本，demo 面板用）。"""
    v = verdict
    vtype = _TYPE_CN.get(v.overridden_type, v.overridden_type)
    crit = "关键" if v.effective_critical else "非关键"
    lines: List[str] = []
    lines.append(f"── 主张 #{index}  「{v.claim.text}」")
    lines.append(f"    判定：{STATUS_ICON[v.status]}  [{STATUS_TITLE[v.status]}]")
    rule_only = bool(v.claim.rule_refs) and not v.claim.evidence_refs
    override_line = (
        f"    类型：LLM标注={_TYPE_CN.get(v.claim.type, v.claim.type)} "
        f"→ 表层信号重判={vtype}（{crit}）"
    )
    if rule_only:
        override_line += "  【规则直出·权威背书：跳过关系型拒答，改与规则内容做高门槛表面比对】"
    lines.append(override_line)
    checks = v.checks
    citation = checks.get("citation") or {}
    support = checks.get("claim_support") or {}
    refs: List[str] = []
    refs += [f"证据:{x}" for x in (citation.get("evidence_refs") or v.claim.evidence_refs)]
    refs += [f"规则:{x}" for x in (citation.get("rule_refs") or v.claim.rule_refs)]
    lines.append(f"    锚点：{('，'.join(refs)) if refs else '（无）'}")
    if support and "ratio" in support:
        lines.append(
            f"    表面一致：ratio={support['ratio']:.2f} → {REASON_LABELS.get(str(support.get('reason')), support.get('reason'))}"
        )
    if v.reason:
        reason_label = REASON_LABELS.get(v.reason, v.reason)
        lines.append(f"    原因：{reason_label}")
        if v.message and v.message != reason_label:
            lines.append(f"    说明：{v.message}")
    if v.evidence_used:
        lines.append(f"    引用证据：{', '.join(v.evidence_used)}")
    return lines


def render_verdict_panel(verdicts: Sequence[ClaimVerdict]) -> List[str]:
    out: List[str] = []
    if not verdicts:
        out.append("（无主张 —— 判定无据可答，整段拒答）")
    for i, v in enumerate(verdicts, 1):
        out.extend(claim_card(v, i))
    return out


def render_evidence(result: PipelineResult) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for e in result.evidence:
        if e.evidence_id in seen:
            continue
        seen.add(e.evidence_id)
        out.append(
            f"  · [{e.evidence_id}] {e.source_type} {e.grade}级 "
            f"「{(e.text_span or '')[:60]}…」"
        )
    for d in result.retrieved:
        if d.document.doc_id in seen:
            continue
        seen.add(d.document.doc_id)
        out.append(
            f"  · [{d.document.doc_id}] 检索命中 score={d.score:.3f} "
            f"「{(d.document.title or '')[:40]}」"
        )
    return out


def render_rules(result: PipelineResult) -> List[str]:
    out: List[str] = []
    for d in result.matched_rules:
        plans = " / ".join(
            f"{r.plan_name}（{r.grade}级）" for r in d.recommendations
        )
        scope = " ".join(
            x for x in (d.cancer_type or "", d.treatment_line or "", d.biomarker or "") if x
        )
        out.append(f"  · [{d.rule_id}] {scope} → {plans}（{d.source_version}）")
    return out


# ---------------------------------------------------------------------------
# 主流程演示
# ---------------------------------------------------------------------------
def show_question(pipe: Pipeline, question: str, *, top_k: int = 5) -> PipelineResult:
    """跑一次完整可信问答并分段打印数据流。"""
    print("\n" + "=" * 74)
    print(f"❓ 问题：{question}")
    print("=" * 74)

    result = pipe.ask(question, top_k=top_k)

    print("\n▍检索召回 / 证据溯源（EvidenceId）：")
    for line in render_evidence(result) or ["  （无）"]:
        print(line)
    print("\n▍命中权威规则（RuleDecision）：")
    for line in render_rules(result) or ["  （无）"]:
        print(line)
    print("\n▍声明级校验门（逐条主张面板）：")
    for line in render_verdict_panel(result.report.verdicts):
        print(line)
    print("\n▍最终对外回答：")
    print(f"  {result.answer_text}")
    if result.note:
        print(f"  （注：{result.note}）")
    print(f"  [整体 status={result.status} · used_llm={result.used_llm or '规则直出'}]")
    return result


def show_eval_case(pipe: Pipeline, case: Dict[str, Any]) -> None:
    """校验门分诊：把 eval 用例中（含幻觉变体的）主张逐条过校验门。

    无 LLM 也能跑 —— 被测对象是「检索→规则→校验门」确定性链路，
    主张由 eval 集显式给出（既含正确输出也含幻觉变体）。
    """
    question = str(case.get("question", ""))
    print("\n" + "=" * 74)
    print(f"❓ [eval 用例] {question}")
    print("=" * 74)
    note = case.get("_note") or ""
    if note:
        print(f"   用例说明：{note}")

    from groundedrag.guardrail import EvidenceRegistry

    info = pipe.retrieve_and_match(
        question,
        context=case.get("context") or None,
        evidence_docs=case.get("evidence_docs") or None,
    )
    # 主张解析（含 expected 保留）统一收口到 eval.cases，与 runner/app 一致
    claims, expected_by_id = parse_case_claims(
        case,
        evidence_ids=info["evidence_ids"],
        rule_ids=info["rule_ids"],
    )
    registry = EvidenceRegistry(info["registry"].all())
    report = pipe.verifier.verify(claims, registry, info["matched"])
    for i, v in enumerate(report.verdicts, 1):
        exp = expected_by_id.get(id(v.claim), "-")
        ok = "✓" if (exp == "-" or exp == v.status) else "✗"
        lines = claim_card(v, i)
        lines[0] = f"{lines[0]}   期望={exp} {ok}"
        print("\n".join(lines))
        print()
    print(f"  整体 status={report.overall_status} · {report.message}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
DEMO_QUESTIONS = [
    "EGFR 突变的晚期肺癌一线推荐什么方案？",
    "RAS 突变的结直肠癌一线可以用西妥昔单抗吗？",
    "骨质疏松应该补什么钙？",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GroundedRAG CLI demo")
    parser.add_argument("--question", help="自定义问题（跑一次完整可信问答）")
    parser.add_argument("--case", type=int, help="跑 eval_set.jsonl 第 N 条用例（校验门分诊）")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    print("GroundedRAG — 声明级校验门 RAG · 端到端 demo")
    print("种子数据为自研合成示例，不构成真实诊疗建议。\n")

    pipe = build_pipeline()
    llm_ready = pipe.llm.is_available()

    if args.question:
        show_question(pipe, args.question, top_k=args.top_k)
        return 0
    if args.case is not None:
        cases = load_eval_cases()
        if not 1 <= args.case <= len(cases):
            print(f"用例号需在 1..{len(cases)} 之间", file=sys.stderr)
            return 2
        show_eval_case(pipe, cases[args.case - 1])
        return 0

    print(f"LLM：{'已配置（OpenAI 兼容端点）' if llm_ready else '未配置（模板回退 / 规则直出）'}")
    print("\n—— 主流程演示：三个典型问题 ——")
    for q in DEMO_QUESTIONS:
        show_question(pipe, q, top_k=args.top_k)

    cases = load_eval_cases()
    print("\n\n" + "#" * 74)
    print("# 校验门分诊演示：把 eval 集中的幻觉/反例主张逐条过校验门")
    print("# （编号对应 eval_set.jsonl 行号，可用 --case N 单独跑）")
    print("#" * 74)
    for n in (3, 6, 18):
        if n <= len(cases):
            show_eval_case(pipe, cases[n - 1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
