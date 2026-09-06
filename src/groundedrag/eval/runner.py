"""评测集跑分：引用完整性率 / 表面一致率 / 拒答正确率。

`eval_set.jsonl` 的每组用例表达"一次问答 + 模型可能输出的若干条主张 + 期望结论"：

.. code-block:: json

    {
      "question": "EGFR 突变的晚期肺癌一线推荐什么方案？",
      "context": {"cancer_type": "肺癌", "treatment_line": "一线", "biomarker": "EGFR"},
      "evidence_docs": ["doc-lung-egfr-01"],          // 钉死可用证据（null=自动检索）
      "claims": [
        {"text": "推荐方案：奥希替尼[证据1]", "expected": "pass"},    // 有据正例
        {"text": "该患者不应使用吉非替尼[证据1]", "expected": "refuse"} // 关系型反例
      ],
      "expect_refusal": false
    }

三条指标口径（README 亦说明，**不声称"真实支持率"**）：

- **引用完整性率** = 期望 ``pass`` 的主张中，引用检查（citation）通过的占比
  —— 衡量"有据主张确实带上了完整、可解析的引用锚点"。
- **表面一致率** = 期望 ``pass`` 的主张中，最终状态为 ``pass`` 的占比
  —— 衡量"证据充分、要素一致的好主张没有被误拒"。
- **拒答正确率** = 期望 ``refuse`` 的主张中，校验门确实给出 ``refuse`` 的占比
  —— 衡量"无证据/自相矛盾/关系型主张被成功拦截"。**这是防幻觉的关键指标。**

评测**不走真实 LLM**：主张由 eval_set 显式给出（既含正确输出也含幻觉变体），
被测对象是"检索→规则→校验门"整条确定性链路。裸 prompt RAG 对照组可用
:func:`baseline_questions` 导出问题清单后在任意模型上跑，再用
:func:`score_baseline` 与 expect_refusal 对比打分。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from groundedrag.eval.cases import load_eval_set, parse_case_claims
from groundedrag.guardrail.verifier import ANNOTATE, PASS, REFUSE, VerifyReport
from groundedrag.pipeline import Pipeline


@dataclass
class ClaimRow:
    """单条主张的评测记录。"""

    text: str
    expected: str
    actual: str
    correct: bool
    reason: str = ""
    message: str = ""
    citation_passed: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "expected": self.expected,
            "actual": self.actual,
            "correct": self.correct,
            "reason": self.reason,
            "message": self.message,
            "citation_passed": self.citation_passed,
        }


@dataclass
class CaseRow:
    """一组用例的评测记录。"""

    index: int
    question: str
    claim_rows: List[ClaimRow] = field(default_factory=list)
    expect_refusal: bool = False
    actual_refusal: bool = False
    answer_refusal_correct: Optional[bool] = None
    matched_rules: int = 0

    @property
    def claims_correct(self) -> bool:
        return bool(self.claim_rows) and all(r.correct for r in self.claim_rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "question": self.question,
            "expect_refusal": self.expect_refusal,
            "actual_refusal": self.actual_refusal,
            "answer_refusal_correct": self.answer_refusal_correct,
            "matched_rules": self.matched_rules,
            "claims_correct": self.claims_correct,
            "claims": [r.to_dict() for r in self.claim_rows],
        }


@dataclass
class EvalReport:
    """整体评测结果：三条指标 + 逐组明细。"""

    cases: List[CaseRow] = field(default_factory=list)

    # -- 三条指标 -----------------------------------------------------
    def _rows(self, expected: str) -> List[ClaimRow]:
        return [r for c in self.cases for r in c.claim_rows if r.expected == expected]

    @property
    def citation_rate(self) -> Optional[float]:
        """引用完整性率：期望 pass 的主张中 citation 检查通过的占比。"""
        rows = self._rows(PASS)
        if not rows:
            return None
        ok = [r for r in rows if r.citation_passed is True]
        return len(ok) / len(rows)

    @property
    def surface_rate(self) -> Optional[float]:
        """表面一致率：期望 pass 的主张最终状态为 pass 的占比。"""
        rows = self._rows(PASS)
        if not rows:
            return None
        ok = [r for r in rows if r.actual == PASS]
        return len(ok) / len(rows)

    @property
    def refusal_rate(self) -> Optional[float]:
        """拒答正确率：期望 refuse 的主张最终状态为 refuse 的占比。"""
        rows = self._rows(REFUSE)
        if not rows:
            return None
        ok = [r for r in rows if r.actual == REFUSE]
        return len(ok) / len(rows)

    # -- 辅助聚合 -----------------------------------------------------
    @property
    def annotate_rate(self) -> Optional[float]:
        rows = self._rows(ANNOTATE)
        if not rows:
            return None
        return len([r for r in rows if r.actual == ANNOTATE]) / len(rows)

    @property
    def claim_accuracy(self) -> Optional[float]:
        """整体主张级正确率（含 annotate 期望），供参考非头条指标。"""
        rows = [r for c in self.cases for r in c.claim_rows]
        if not rows:
            return None
        return len([r for r in rows if r.correct]) / len(rows)

    @property
    def answer_refusal_accuracy(self) -> Optional[float]:
        """整组级拒答判定正确率（期望拒答 vs 实际拒答）。"""
        rows = [c for c in self.cases if c.answer_refusal_correct is not None]
        if not rows:
            return None
        return len([c for c in rows if c.answer_refusal_correct]) / len(rows)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "citation_completeness_rate": round(self.citation_rate, 4)
            if self.citation_rate is not None else None,
            "surface_consistency_rate": round(self.surface_rate, 4)
            if self.surface_rate is not None else None,
            "refusal_correctness_rate": round(self.refusal_rate, 4)
            if self.refusal_rate is not None else None,
            "annotate_rate": round(self.annotate_rate, 4)
            if self.annotate_rate is not None else None,
            "claim_accuracy": round(self.claim_accuracy, 4)
            if self.claim_accuracy is not None else None,
            "answer_refusal_accuracy": round(self.answer_refusal_accuracy, 4)
            if self.answer_refusal_accuracy is not None else None,
            "case_count": len(self.cases),
            "claim_count": sum(len(c.claim_rows) for c in self.cases),
        }

    def render(self) -> str:
        """人类可读摘要（含逐组明细行）。"""
        lines = ["===== GroundedRAG eval 结果 =====", json.dumps(self.summary_dict(), ensure_ascii=False, indent=2)]
        lines.append("—— 逐组明细 ——")
        for c in self.cases:
            status = []
            for r in c.claim_rows:
                mark = "✓" if r.correct else "✗"
                status.append(
                    f"    [{mark}] expected={r.expected} actual={r.actual} "
                    f"reason={r.reason} | {r.text}"
                )
            lines.append(f"#{c.index} {c.question}")
            lines.extend(status)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# E2E 端到端评测
# ---------------------------------------------------------------------------
@dataclass
class E2EResult:
    """单条 E2E 评测结果（完整 ask() 流程）。"""

    case_id: int
    question: str
    status: str  # Answer.status: pass / refuse / annotate / partial_pass
    used_llm: Optional[str] = None
    answer_text: str = ""
    verdicts: List[Dict[str, Any]] = field(default_factory=list)
    expected_status: Optional[str] = None
    matched: Optional[bool] = None  # actual vs expected; None = 无期望值
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "status": self.status,
            "used_llm": self.used_llm,
            "answer_text": self.answer_text,
            "verdicts": self.verdicts,
            "expected_status": self.expected_status,
            "matched": self.matched,
            "latency_ms": self.latency_ms,
        }


@dataclass
class E2EReport:
    """E2E 评测汇总报告。"""

    total: int = 0
    matched_count: int = 0
    mismatched_count: int = 0
    llm_used_count: int = 0
    rule_direct_count: int = 0
    verdict_distribution: Dict[str, int] = field(default_factory=dict)
    results: List[E2EResult] = field(default_factory=list)

    @property
    def match_rate(self) -> Optional[float]:
        judged = [r for r in self.results if r.matched is not None]
        if not judged:
            return None
        return len([r for r in judged if r.matched]) / len(judged)

    @property
    def llm_usage_rate(self) -> Optional[float]:
        if not self.total:
            return None
        return self.llm_used_count / self.total

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "match_rate": round(self.match_rate, 4) if self.match_rate is not None else None,
            "llm_usage_rate": round(self.llm_usage_rate, 4) if self.llm_usage_rate is not None else None,
            "rule_direct_rate": round(self.rule_direct_count / self.total, 4) if self.total else None,
            "verdict_distribution": self.verdict_distribution,
            "avg_latency_ms": round(sum(r.latency_ms for r in self.results) / len(self.results)) if self.results else 0,
        }

    def render(self) -> str:
        lines = ["===== GroundedRAG E2E 评测结果 =====", json.dumps(self.summary_dict(), ensure_ascii=False, indent=2)]
        lines.append("—— 逐条明细 ——")
        for r in self.results:
            mark = "✓" if r.matched else ("✗" if r.matched is False else "?")
            llm_tag = f"llm={r.used_llm}" if r.used_llm else "规则直出"
            lines.append(
                f"  [{mark}] #{r.case_id} status={r.status} {llm_tag} "
                f"latency={r.latency_ms}ms"
            )
            lines.append(f"       Q: {r.question}")
            if r.answer_text:
                lines.append(f"       A: {r.answer_text[:120]}")
            if r.verdicts:
                for v in r.verdicts:
                    lines.append(f"         claim: [{v.get('status')}] {v.get('text', '')[:80]}")
        return "\n".join(lines)


def run_e2e(
    eval_path: Union[str, Path],
    pipeline_fn: Callable[[str], Any],
    *,
    expected_verdicts: Optional[Dict[int, str]] = None,
) -> E2EReport:
    """跑 E2E 端到端评测：对每个 question 调用完整 pipeline.ask()。

    ``pipeline_fn``: 接收 question 字符串，返回 ``Answer`` 对象。
    ``expected_verdicts``: {case_id (0-based): expected_status} 可选。
    """
    cases = load_eval_set(eval_path)
    report = E2EReport()

    for i, case in enumerate(cases):
        question = case.get("question", "")
        expected = (expected_verdicts or {}).get(i)

        t0 = time.time()
        answer = pipeline_fn(question)
        latency = int((time.time() - t0) * 1000)

        verdicts = answer.report.verdicts if answer.report else []
        actual_verdicts = [
            {"text": v.claim.text, "status": v.status, "reason": v.reason}
            for v in verdicts
        ]
        matched = (answer.status == expected) if expected is not None else None

        result = E2EResult(
            case_id=i + 1,
            question=question,
            status=answer.status,
            used_llm=getattr(answer, "used_llm", None),
            answer_text=getattr(answer, "answer_text", ""),
            verdicts=actual_verdicts,
            expected_status=expected,
            matched=matched,
            latency_ms=latency,
        )
        report.results.append(result)
        report.total += 1

        if matched is True:
            report.matched_count += 1
        elif matched is False:
            report.mismatched_count += 1

        if result.used_llm:
            report.llm_used_count += 1
        else:
            report.rule_direct_count += 1

        report.verdict_distribution[result.status] = (
            report.verdict_distribution.get(result.status, 0) + 1
        )

    return report


def _run_case(
    pipeline: Pipeline,
    case: Mapping[str, Any],
    index: int,
    *,
    top_k: int = 5,
) -> CaseRow:
    question = str(case.get("question", ""))
    info = pipeline.retrieve_and_match(
        question,
        context=case.get("context") or None,
        evidence_docs=case.get("evidence_docs") or None,
        top_k=top_k,
    )
    matched = info["matched"]
    ev_ids = info["evidence_ids"]
    rule_ids = info["rule_ids"]

    # 逐条解析主张，并保留每条携带的 expected（仅 dict 形式可携带；
    # 解析与 demo/app 共用 eval.cases.parse_case_claims，避免三份漂移实现）
    claim_items, expected_of = parse_case_claims(
        case, evidence_ids=ev_ids, rule_ids=rule_ids
    )

    report: VerifyReport = pipeline.verifier.verify(
        claim_items, info["registry"], matched
    )
    verdict_by_claim = {id(v.claim): v for v in report.verdicts}

    raw_rows: List[ClaimRow] = []
    for c in claim_items:
        expected = expected_of.get(id(c))
        if expected is None:
            continue  # 字符串主张不参与统计（无期望结论）
        verdict = verdict_by_claim.get(id(c))
        if verdict is None:
            continue
        citation = verdict.checks.get("citation")
        raw_rows.append(
            ClaimRow(
                text=c.text,
                expected=expected,
                actual=verdict.status,
                correct=(expected == verdict.status),
                reason=verdict.reason,
                message=verdict.message,
                citation_passed=bool(citation.get("passed")) if citation else None,
            )
        )

    expect_refusal = bool(case.get("expect_refusal", case.get("expect_answer") == "refuse"))
    # 实际"整组拒答"口径：整体 REFUSE 且无任何主张通过校验门
    # （部分主张被拒但有通过主张 → 交付的是部分保留答案，不算整组拒答）
    any_passed = any(v.status == PASS for v in report.verdicts)
    actual_refusal = report.overall_status == REFUSE and not any_passed
    row = CaseRow(
        index=index,
        question=question,
        claim_rows=raw_rows,
        expect_refusal=expect_refusal,
        actual_refusal=actual_refusal,
        answer_refusal_correct=(
            expect_refusal == actual_refusal
            if (raw_rows or expect_refusal)
            else None
        ),
        matched_rules=len(matched),
    )
    return row


def run_evaluation(
    eval_path: Union[str, Path],
    *,
    pipeline: Optional[Pipeline] = None,
    docs_path: Optional[Union[str, Path]] = None,
    rules_path: Optional[Union[str, Path]] = None,
    top_k: int = 5,
) -> EvalReport:
    """跑完整套 eval_set，返回 EvalReport（含三条指标）。

    需提供已构造的 ``pipeline``，或同时给 ``docs_path`` + ``rules_path``
    让其内部 ``Pipeline.build_from_json``。
    """
    if pipeline is None:
        if docs_path is None or rules_path is None:
            raise ValueError("pipeline 与 (docs_path, rules_path) 至少给一组")
        pipeline = Pipeline.build_from_json(docs_path, rules_path)
    cases = load_eval_set(eval_path)
    report = EvalReport()
    for i, case in enumerate(cases):
        report.cases.append(_run_case(pipeline, case, i + 1, top_k=top_k))
    return report


# ---------------------------------------------------------------------------
# 裸 prompt RAG 对照组辅助
# ---------------------------------------------------------------------------
def baseline_questions(eval_set: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """导出对照组问题清单：裸 prompt 模型只需回答"会/不会拒答"。

    每项含 ``question`` 与 ``expect_refusal``；期望拒答的题目即"幻觉高危题"，
    裸 RAG（直接输出、无校验门）天然答错这些题 → 拒答正确率 ~0。
    """
    out = []
    for i, case in enumerate(eval_set):
        out.append(
            {
                "index": i + 1,
                "question": case.get("question", ""),
                "expect_refusal": bool(
                    case.get("expect_refusal", case.get("expect_answer") == "refuse")
                ),
            }
        )
    return out


def score_baseline(
    responses: Mapping[str, bool],
    baseline: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """给裸 RAG 对照组打分：每组 'refused' ∈ {0,1} 与 expect_refusal 比对。

    ``responses`` 键为 question；值为"该模型是否拒绝回答"。
    返回含三项口径：期望拒答题的拦截率（=GroundRAG 的拒答正确率对照）。
    """
    tp = tn = fp = fn = total = 0
    for item in baseline:
        q = item.get("question", "")
        if q not in responses:
            continue
        total += 1
        refused = bool(responses[q])
        expect = bool(item.get("expect_refusal"))
        if expect and refused:
            tp += 1
        elif expect and not refused:
            fn += 1
        elif not expect and refused:
            fp += 1
        else:
            tn += 1
    return {
        "refusal_expected_caught": (tp / (tp + fn)) if (tp + fn) else None,
        "false_refusal": (fp / (fp + tn)) if (fp + tn) else None,
        "answered_count": total,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="groundedrag-eval",
        description="GroundedRAG 评测：引用完整性率 / 表面一致率 / 拒答正确率 / E2E 端到端",
    )
    parser.add_argument("--docs", default="examples/seed_docs.jsonl")
    parser.add_argument("--rules", default="examples/seed_rules.json")
    parser.add_argument("--eval", default="examples/eval_set.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="只输出 JSON 摘要")

    # E2E 端到端模式
    parser.add_argument("--e2e", action="store_true", help="启用 E2E 端到端评测（需 API Key）")
    parser.add_argument("--llm-provider", default=None, help="主 LLM provider：xiaomi / deepseek / doubao")
    parser.add_argument("--llm-api-key", default=None, help="LLM API Key（覆盖 .env）")
    parser.add_argument("--llm-base-url", default=None, help="LLM Base URL（覆盖 .env）")
    parser.add_argument("--llm-model", default=None, help="LLM 模型名（覆盖 .env）")
    parser.add_argument("--e2e-max", type=int, default=0, help="E2E 最多跑几题（0=全部）")
    args = parser.parse_args(argv)

    if args.e2e:
        return _run_e2e_cli(args)

    report = run_evaluation(
        args.eval, docs_path=args.docs, rules_path=args.rules, top_k=args.top_k
    )
    if args.json:
        print(json.dumps(report.summary_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.render())
    return 0


def _run_e2e_cli(args: argparse.Namespace) -> int:
    """E2E 模式 CLI 入口（支持 .env + 多 provider failover）。"""
    import os
    import tempfile

    from groundedrag.llm.failover import FailoverLLM
    from groundedrag.llm.openai_compat import OpenAICompatibleLLM

    # 1. 加载 .env（不覆盖已有环境变量）
    _load_dotenv()

    # 2. Provider 预设（base_url / model 默认值）
    _PRESETS: Dict[str, Dict[str, str]] = {
        "xiaomi": {
            "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
            "model": "mimo-v2.5-pro",
            "key_env": "XIAOMI_API_KEY",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "key_env": "DEEPSEEK_API_KEY",
        },
        "doubao": {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model": "doubao-seed-2-0-pro",
            "key_env": "DOUBAO_API_KEY",
        },
    }

    # 3. 确定主 provider
    provider = args.llm_provider or os.getenv("LLM_PROVIDER", "deepseek")
    if provider not in _PRESETS:
        print(f"错误：不支持的 provider '{provider}'，可选：{', '.join(_PRESETS)}")
        return 1
    preset = _PRESETS[provider]

    # 4. 解析配置（CLI > 统一 env > provider-specific > preset 默认值）
    unified_key = args.llm_api_key or os.getenv("LLM_API_KEY")
    unified_url = args.llm_base_url or os.getenv("LLM_BASE_URL")
    unified_model = args.llm_model or os.getenv("LLM_MODEL")

    primary_key = unified_key or os.getenv(preset["key_env"], "")
    primary_url = unified_url or preset["base_url"]
    primary_model = unified_model or preset["model"]

    if not primary_key:
        print("错误：未找到 API Key。请配置以下任一方式：")
        print(f"  1. .env 文件：LLM_API_KEY=sk-xxx 或 {preset['key_env']}=sk-xxx")
        print("  2. 环境变量：export LLM_API_KEY=sk-xxx")
        print("  3. 命令行：  --llm-api-key sk-xxx")
        return 1

    # 5. 构建 primary + fallback services
    services = []
    primary = OpenAICompatibleLLM(
        base_url=primary_url,
        api_key=primary_key,
        model=primary_model,
    )
    services.append(primary)
    if not args.json:
        print(f"主模型：{provider} / {primary_model}")

    # 其他 provider 作为 fallback
    for fb_name, fb_preset in _PRESETS.items():
        if fb_name == provider:
            continue
        fb_key = unified_key or os.getenv(fb_preset["key_env"], "")
        if not fb_key:
            continue
        fb_url = unified_url or fb_preset["base_url"]
        fb_model = unified_model or fb_preset["model"]
        fb_svc = OpenAICompatibleLLM(
            base_url=fb_url,
            api_key=fb_key,
            model=fb_model,
        )
        if fb_svc.is_available():
            services.append(fb_svc)
            if not args.json:
                print(f"备用模型：{fb_name} / {fb_model}")

    llm = FailoverLLM(services=services)
    pipeline = Pipeline.build_from_json(args.docs, args.rules, llm=llm)

    # 6. 加载 eval set
    cases = load_eval_set(args.eval)
    if args.e2e_max > 0:
        cases = cases[: args.e2e_max]

    expected: Dict[int, str] = {}
    for i, case in enumerate(cases):
        ev = case.get("expected_verdicts")
        if ev:
            expected[i] = ev

    eval_path = args.eval
    if args.e2e_max > 0:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix="e2e_subset_"
        )
        for case in cases:
            tmp.write(json.dumps(case, ensure_ascii=False) + "\n")
        tmp.close()
        eval_path = tmp.name

    report = run_e2e(eval_path, pipeline.ask, expected_verdicts=expected or None)

    if eval_path != args.eval:
        os.unlink(eval_path)

    if args.json:
        print(json.dumps(report.summary_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.render())
    return 0


def _load_dotenv(path: str = ".env") -> None:
    """轻量 .env 加载器（不覆盖已有环境变量，零依赖）。

    支持行内注释（值后 `` # ...`` 被剥离）。
    """
    import os
    import re

    p = os.path.join(os.getcwd(), path)
    if not os.path.isfile(p):
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            # 去掉行内注释（空格/制表符 + # + 后续内容）
            value = re.sub(r"\s+#.*$", "", value).strip()
            value = value.strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


if __name__ == "__main__":  # pragma: no cover —— python -m 入口，由集成测试覆盖
    raise SystemExit(main())
