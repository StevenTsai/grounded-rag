#!/usr/bin/env python3
"""Gradio 可视化 demo —— ★评审演示主界面（设计文档 §3.6）。

特性：
- **逐条主张面板**：每条主张一张卡 —— ✅ 有据可依（引用完整 + 锚点）/
  ✗ 无据拒答（展示保守拒答文案与原因）/ 🟡 标注降级；规则命中显示 RuleDecision + 证据等级。
- **裸 RAG 对照开关**：同一问题切换「GroundedRAG 可信模式 / 裸 RAG 对照」，
  直观展示"无校验门直出"与"逐条校验后降级/拒答"的差异（评审视频 90 秒核心镜头）。
- **检索来源面板**：召回的文档/证据/命中规则可展开，呼应"有据可依、有源可溯"。
- **eval 用例点选**：内置 ``eval_set.jsonl`` 题目一键换题，demo 可重复、可复现。

运行：
    pip install -e ".[demo]"      # 安装 gradio
    python examples/app.py        # 打开 http://127.0.0.1:7860

无 API Key 也能完整演示：可信模式走「规则直出 / 结构化拒答」，完全本地、无网络。
配置真实 LLM（OpenAI 兼容端点）后，裸 RAG 对照会调用模型直出，对比更有说服力：
    LLM_API_KEY=sk-… LLM_BASE_URL=https://api.deepseek.com/v1 LLM_MODEL=deepseek-chat python examples/app.py
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from groundedrag.guardrail import (
    ANNOTATE,
    PASS,
    REFUSE,
    AnswerClaim,
    ClaimVerdict,
)
from groundedrag.guardrail.claims import parse_claims
from groundedrag.llm import FailoverLLM, OpenAICompatibleLLM
from groundedrag.pipeline import Pipeline, PipelineResult

HERE = Path(__file__).resolve().parent
SEED_DOCS = HERE / "seed_docs.jsonl"
SEED_RULES = HERE / "seed_rules.json"
EVAL_SET = HERE / "eval_set.jsonl"

# 校验 reason → 中文说明（与 demo.py 保持一致）
REASON_LABELS: Dict[str, str] = {
    "pass": "引用完整 + 表面要素一致 + 证据充分",
    "citation_incomplete": "无引用或引用不完整（缺证据/规则锚点）",
    "semantic_refute": "语义核验：主张被绑定证据否定",
    "relational_unverifiable": "含否定/比较/因果关系语义，确定性档无法验证方向",
    "rule_conflict_lost": "主张引用的规则在冲突裁定中被淘汰",
    "rule_conflict_divergence": "同域规则冲突无法裁定，不静默二选一",
    "insufficient_evidence": "证据充分性不足（等级过低 / 已过期 / 数值不符）",
    "rule_content_mismatch": "与所引规则内容不符（疑似挂靠规则包装幻觉）",
    "rule_authoritative": "内容与命中的权威规则一致（规则直出，无需 LLM）",
    "number_mismatch": "数值与证据不符",
    "entity_mismatch": "实体/要素未出现在证据或所引规则中",
    "evidence_negates_claim": "证据侧否定翻转：证据写不可用/禁忌，主张却作正向推荐",
}
STATUS_CN = {PASS: "有据可依 · 原样输出", ANNOTATE: "标注降级 · 未经本地核验", REFUSE: "无据可依 · 拒绝输出"}
_TYPE_CN = {
    "factual": "事实",
    "negation": "否定",
    "comparison": "比较",
    "causal": "因果",
    "indication": "治疗建议",
}


# ---------------------------------------------------------------------------
# 管线 / 数据
# ---------------------------------------------------------------------------
def _real_llm() -> Optional[FailoverLLM]:
    """环境变量配置的真实 LLM；未配置返回 None。"""
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
        ],
        append_template=False,
    )


PIPE = Pipeline.build_from_json(SEED_DOCS, SEED_RULES)
REAL_LLM = _real_llm()


def load_eval_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(EVAL_SET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


EVAL_CASES = load_eval_cases()
CASE_QUESTIONS = [str(c.get("question", "")) for c in EVAL_CASES]


def find_case(question: str) -> Optional[Dict[str, Any]]:
    for c in EVAL_CASES:
        if str(c.get("question", "")).strip() == question.strip():
            return c
    return None


# ---------------------------------------------------------------------------
# 逻辑：一次运行
# ---------------------------------------------------------------------------
def _inject_case_claims(
    case: Dict[str, Any], question: str
) -> Dict[str, Any]:
    """把 eval 用例中（含幻觉变体的）主张逐条过校验门 → 逐条判定面板 + 整体判定。

    无 LLM 也可跑：被测对象是"检索→规则→校验门"确定性链路。
    """
    result = PIPE.ask(
        question,
        context=case.get("context") or None,
        evidence_docs=case.get("evidence_docs") or None,
    )
    info = PIPE.retrieve_and_match(
        question,
        context=case.get("context") or None,
        evidence_docs=case.get("evidence_docs") or None,
    )
    claims: List[AnswerClaim] = []
    for item in case.get("claims", []) or []:
        if isinstance(item, str):
            claims.extend(
                parse_claims(
                    item,
                    evidence_ids=info["evidence_ids"],
                    rule_ids=info["rule_ids"],
                )
            )
        else:
            claims.append(AnswerClaim.from_dict(item))
    expected_of = {
        i: str(c.get("expected", ""))
        for i, c in enumerate(case.get("claims", []))
        if isinstance(c, dict) and c.get("expected")
    }
    report = PIPE.verifier.verify(claims, info["registry"], info["matched"])
    return {
        "report": report,
        "verdicts": report.verdicts,
        "expected_of": expected_of,
        "result": result,
        "expect_refusal": bool(case.get("expect_refusal", False)),
    }


def trusted_run(
    question: str,
) -> Dict[str, Any]:
    """GroundedRAG 可信模式：完整流水线 + 逐条主张面板。"""
    case = find_case(question)
    out: Dict[str, Any] = {}
    if case:
        run = _inject_case_claims(case, question)
        report = run["report"]
        out["claims_html"] = _claims_html(run["verdicts"], run["expected_of"], "eval")
        # 整组级判定 = 评测 expect_refusal 口径（runner.py）：整体 REFUSE **且**无任何
        # 主张通过校验门才算"整组拒答"；有通过主张（哪怕含被拒变体）→ 交付部分保留答案。
        any_passed = any(v.status == PASS for v in report.verdicts)
        refused = report.overall_status == REFUSE and not any_passed
        expected_refuse = run["expect_refusal"]
        verdict_cn = "拒答（无据可依 / 矛盾）" if refused else "通过（保留有据主张）"
        hit = "✓ 符合用例期望" if refused == expected_refuse else "✗ 与用例期望不符"
        body = Pipeline._assemble_answer(report)
        banner = (
            f"### 本用例整体判定：**{verdict_cn}**　{hit}\n\n"
            f"**逐条结果见下方主张面板。**\n\n"
            f"> 依据：overall_reason=`{report.overall_reason}` · "
            f"期望 = {'拒答' if expected_refuse else '通过'}"
        )
        if body and body != "":
            banner += f"\n\n---\n\n{body}"
        out["answer_md"] = banner
        out["result"] = run["result"]  # 仅用于来源面板（检索/规则/证据一致）
    else:
        result = PIPE.ask(question)
        out["claims_html"] = _claims_html(result.report.verdicts, {}, "pipeline")
        out["result"] = result
        out["answer_md"] = _answer_md(result, bare=False)
    out["sources_html"] = _sources_html(out["result"])
    return out


def bare_run(question: str) -> Dict[str, Any]:
    """裸 RAG 对照：无校验门直出。

    配置了真实 LLM → 无约束生成（同一批召回文档，不做主张解析/校验）；
    未配置 → 无法诚实地产出"幻觉" —— 若该问题在 eval 集中携带幻觉变体主张，
    则把它们作为"典型无约束直出"**明确标注为模拟注入**展示对照；
    否则提示需配置 LLM 才能真正跑裸 RAG。
    """
    case = find_case(question)
    if REAL_LLM is not None:
        result = PIPE.retrieve_and_match(question)
        ctx = "\n".join(
            f"- {r.document.title}\n{r.document.content[:400]}"
            for r in result["retrieved"][:3]
        )
        prompt = (
            "请根据以下资料直接回答用户问题，不要分点，直接给出答案：\n\n"
            f"{ctx}\n\n问题：{question}"
        )
        try:
            raw = REAL_LLM.generate(prompt)
        except Exception as exc:  # noqa: BLE001 —— 对照失败也要把异常呈现出来
            raw = f"（裸 RAG 调用失败：{exc}）"
        banner = (
            f"> ⚠️ **裸 RAG 对照（真实 LLM · {REAL_LLM.services[0].name}）**"
            f"：无引用校验、无拒答门，模型直接生成。\n\n---\n\n{raw}"
        )
        return {
            "claims_html": _bare_notice_html(),
            "answer_md": banner,
            "result": PIPE.ask(question),
            "sources_html": "",
        }
    hallucinated = [
        c.get("text", "")
        for c in (case or {}).get("claims", [])
        if isinstance(c, dict) and c.get("expected") != PASS
    ]
    if hallucinated:
        sim = "\n\n".join(f"- {t}" for t in hallucinated)
        banner = (
            f"> ⚠️ **裸 RAG 对照（模拟注入 · 无校验门直出）**\n>\n"
            f"> 未配置真实 LLM，无法由模型现场生成；以下候选输出取自 eval 集中"
            f"代表*无约束模型典型错误*的变体主张，**仅作对照示意**，非本次运行生成。\n\n"
            f"{sim}"
        )
    else:
        banner = (
            "> ⚠️ **裸 RAG 对照**：本机未配置 LLM（`LLM_API_KEY`/`OPENAI_API_KEY`），"
            "无法诚实现场演示无约束生成。\n>\n> 配置 OpenAI 兼容端点后重开本界面，"
            "即可对比「直出 vs 逐条校验」。在可信模式下本问题由规则直出/结构化拒答完成。"
        )
    return {
        "claims_html": _bare_notice_html(),
        "answer_md": banner,
        "result": PIPE.ask(question),
        "sources_html": "",
    }


# ---------------------------------------------------------------------------
# HTML 渲染
# ---------------------------------------------------------------------------
def _badge(status: str) -> str:
    icons = {PASS: "✅", ANNOTATE: "🟡", REFUSE: "✗"}
    colors = {PASS: "#137333", ANNOTATE: "#b06000", REFUSE: "#c5221f"}
    bg = {PASS: "#e6f4ea", ANNOTATE: "#fef7e0", REFUSE: "#fce8e6"}
    return (
        f'<span style="background:{bg[status]};color:{colors[status]};'
        f'font-weight:700;border-radius:4px;padding:1px 8px;">'
        f'{icons[status]} {status.upper()}</span>'
    )


def _claims_html(
    verdicts: Sequence[ClaimVerdict],
    expected_of: Optional[Dict[int, str]] = None,
    mode: str = "pipeline",
) -> str:
    expected_of = expected_of or {}
    cards: List[str] = []
    for i, v in enumerate(verdicts, 1):
        rule_only = bool(v.claim.rule_refs) and not v.claim.evidence_refs
        tag = (
            " · <i>规则直出·权威背书</i>"
            if rule_only
            else ""
        )
        exp = expected_of.get(i - 1, "")
        exp_html = (
            f'<span style="color:#666;margin-left:8px;">期望 <b>{exp}</b> '
            f'{"<b style=color:#137333>✓</b>" if exp == v.status else "<b style=color:#c5221f>✗</b>"}</span>'
            if exp
            else ""
        )
        crit = "关键" if v.effective_critical else "非关键"
        over = _TYPE_CN.get(v.overridden_type, v.overridden_type)
        anchors = []
        for x in v.claim.evidence_refs:
            anchors.append(f'<code style="color:#1a73e8">证据 {x}</code>')
        for x in v.claim.rule_refs:
            anchors.append(f'<code style="color:#9334e6">规则 {x}</code>')
        anchor_html = ("锚点：" + "&nbsp;".join(anchors)) if anchors else "锚点：（无）"
        support = (v.checks or {}).get("claim_support") or {}
        ratio_html = ""
        if "ratio" in support:
            rlabel = REASON_LABELS.get(str(support.get("reason")), support.get("reason"))
            ratio_html = (
                f'<div style="color:#666;font-size:.92em;">表面一致 ratio='
                f'{support["ratio"]:.2f} → {html.escape(str(rlabel))}</div>'
            )
        reason_html = ""
        if v.reason:
            reason_html = (
                f'<div style="margin-top:4px;"><b>原因</b>：'
                f'{html.escape(REASON_LABELS.get(v.reason, v.reason))}</div>'
            )
        cards.append(
            f"""
            <div style="border:1px solid #dadce0;border-left:5px solid { {'pass':'#137333','annotate':'#b06000','refuse':'#c5221f'}[v.status] };border-radius:8px;padding:10px 14px;margin:8px 0;background:#fff;">
              <div style="display:flex;align-items:center;flex-wrap:wrap;">
                {_badge(v.status)}
                <span style="margin-left:10px;font-size:.9em;color:#444;">
                  {html.escape(str(crit))} · LLM标注={html.escape(_TYPE_CN.get(v.claim.type, v.claim.type))}
                  → 表层重判={html.escape(over)}{tag}
                </span>
                {exp_html}
              </div>
              <div style="margin:6px 0;font-size:1.02em;font-weight:600;">「{html.escape(v.claim.text)}」</div>
              <div style="color:#666;font-size:.92em;">{anchor_html}</div>
              {ratio_html}
              {reason_html}
            </div>
            """
        )
    if not cards:
        head = "**整段拒答**" if mode == "pipeline" else "（无主张）"
        cards.append(
            f'<div style="border:1px dashed #dadce0;border-radius:8px;padding:12px;'
            f'color:#444;">{head} —— 检索/规则未提供可校验主张，无可逐条展示。</div>'
        )
    return "\n".join(cards)


def _answer_md(result: PipelineResult, *, bare: bool = False) -> str:
    status = result.status
    color = {"pass": "#137333", "annotate": "#b06000", "refuse": "#c5221f"}[status]
    status_cn = {
        "pass": "✅ 回答通过校验门",
        "annotate": "🟡 部分内容未经本地证据核验（已标注）",
        "refuse": "✗ 无据可依 · 已拒答 / 保守回退",
    }[status]
    llm_line = f"生成：`{result.used_llm}`" if result.used_llm else "生成：**规则直出（无需 LLM）**"
    header = f"<h4 style='color:{color};margin:2px 0;'>{status_cn}</h4><div style='color:#666;font-size:.9em;'>{llm_line} · 整体 reason=`{result.report.overall_reason if result.report else '-'}`</div>"
    note = f"\n\n> 📌 {result.note}" if result.note else ""
    return f"{header}\n\n{result.answer_text}{note}"


def _bare_notice_html() -> str:
    return (
        '<div style="border:1px solid #dadce0;border-radius:8px;padding:12px;color:#555;'
        'background:#f8f9fa;">裸 RAG 对照模式**不经过校验门**，因此没有"逐条主张面板"。'
        '将开关切到 **GroundedRAG 可信模式**查看校验门逐条判定。</div>'
    )


def _sources_html(result: PipelineResult) -> str:
    blocks: List[str] = []
    seen: set[str] = set()
    rules = ""
    for d in result.matched_rules:
        plans = " / ".join(
            f"{r.plan_name}（{r.grade}级）" for r in d.recommendations
        )
        scope = " ".join(
            x for x in (d.cancer_type or "", d.treatment_line or "", d.biomarker or "") if x
        )
        rules += (
            f'<li><b>{html.escape(d.rule_id)}</b> {html.escape(scope)} → '
            f'{html.escape(plans)} <span style="color:#666">({html.escape(d.source_version)})</span></li>'
        )
    if rules:
        blocks.append(
            "<details><summary style='cursor:pointer;'>🎯 命中权威规则（RuleDecision）</summary>"
            f"<ul>{rules}</ul></details>"
        )
    evs: List[str] = []
    for e in result.evidence:
        if e.evidence_id in seen:
            continue
        seen.add(e.evidence_id)
        evs.append(
            f'<li><b>{html.escape(e.evidence_id)}</b> · {html.escape(e.source_type)} '
            f'<span style="color:#666">{html.escape(e.grade)}级 · {html.escape(e.source_version or "-")}'
            f' · {html.escape(e.updated_at or "无时间")}</span><br/>'
            f'<span style="color:#555;font-size:.92em;">{html.escape((e.text_span or "")[:160])}</span></li>'
        )
    for r in result.retrieved:
        d = r.document
        if d.doc_id in seen:
            continue
        seen.add(d.doc_id)
        evs.append(
            f'<li><b>{html.escape(d.doc_id)}</b>（检索命中 score={r.score:.3f}）'
            f'<span style="color:#666"> · {html.escape(d.title or "")}</span></li>'
        )
    if evs:
        blocks.append(
            "<details open><summary style='cursor:pointer;'>📚 证据 / 召回文档（EvidenceId）</summary>"
            f"<ul>{''.join(evs)}</ul></details>"
        )
    return "\n".join(blocks) or "<div style='color:#888'>（未召回文档 / 未命中规则）</div>"


# ---------------------------------------------------------------------------
# 裸模式回答文本（供对照组入口复用）
# ---------------------------------------------------------------------------
def main() -> int:
    # 演示/评审现场常无外网：关闭 gradio 的遥测与 HF Hub 在线拉取，避免 import 挂起。
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        import gradio as gr
    except ImportError:
        print(
            "缺少 gradio —— 请先安装 demo 依赖：\n"
            '    pip install -e ".[demo]"\n'
            "然后重新运行：python examples/app.py"
        )
        return 1

    if REAL_LLM is not None:
        llm_note = f"已配置真实 LLM：`{REAL_LLM.services[0].name}` —— 裸 RAG 对照将现场调用模型直出。"
    else:
        llm_note = (
            "未配置 LLM：可信模式走规则直出/结构化拒答（全本地）；裸 RAG 对照将以 eval 集"
            "注入的典型无约束输出作示意。配置 `LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL` 可获得真实对照。"
        )

    with gr.Blocks(title="GroundedRAG — 声明级校验门 RAG 演示") as demo:
        gr.Markdown(
            "# GroundedRAG · 有据可依，无据可拒\n"
            "轻量开源 RAG 框架 —— 回答先拆成原子主张，逐条绑定可溯源证据/权威规则，"
            "过声明级校验门后才输出。*种子数据为自研合成示例，不构成真实诊疗建议。*\n\n"
            f"> {llm_note}"
        )
        with gr.Row():
            mode = gr.Radio(
                ["GroundedRAG 可信模式", "裸 RAG（对照）"],
                value="GroundedRAG 可信模式",
                label="运行模式",
            )
        with gr.Row():
            case_picker = gr.Dropdown(
                ["（自由提问）"] + CASE_QUESTIONS,
                value="（自由提问）",
                label="内置 eval 用例（一键换题）",
            )
        with gr.Row():
            question = gr.Textbox(
                label="问题",
                value="EGFR 突变的晚期肺癌一线推荐什么方案？",
                scale=6,
            )
            run_btn = gr.Button("运行", variant="primary", scale=1)

        answer = gr.Markdown(label="回答 / 拒答结果")
        with gr.Accordion("逐条主张面板（校验门判定）", open=True):
            claims = gr.HTML()
        with gr.Accordion("检索来源 / 证据溯源", open=True):
            sources = gr.HTML()

        def on_pick(choice: str) -> Dict[str, Any]:
            return gr.update(value=choice if choice != "（自由提问）" else "")

        case_picker.change(on_pick, case_picker, question)

        def on_run(q: str, m: str) -> tuple[str, str, str]:
            q = (q or "").strip()
            if not q:
                q = "EGFR 突变的晚期肺癌一线推荐什么方案？"
            if "可信" in m:
                out = trusted_run(q)
            else:
                out = bare_run(q)
            return out["answer_md"], out["claims_html"], out["sources_html"]

        run_btn.click(on_run, [question, mode], [answer, claims, sources])
        question.submit(on_run, [question, mode], [answer, claims, sources])

    demo.launch(server_name="127.0.0.1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
