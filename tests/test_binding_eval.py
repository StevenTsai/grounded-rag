"""锚点绑定持续评测（examples/binding_eval_set.jsonl）与 E2E 锚点绑定率指标。

这两项把"LLM 不吐锚点"这一不可复现的 E2E 问题，转成可离线回归的确定性指标。
"""

from __future__ import annotations

from pathlib import Path

from groundedrag.eval.runner import (
    BIND_EVIDENCE,
    BIND_NONE,
    BIND_RULE,
    E2EReport,
    E2EResult,
    run_binding_evaluation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BINDING_EVAL = REPO_ROOT / "examples" / "binding_eval_set.jsonl"


class TestBindingEvaluation:
    def test_fixture_metrics(self, pipeline):
        report = run_binding_evaluation(BINDING_EVAL, pipeline=pipeline)
        summary = report.summary_dict()
        assert summary["case_count"] == 3
        assert summary["claim_count"] == 5
        # 绑定应当全部命中期望来源（证据/规则/不绑定）
        assert summary["bind_accuracy"] == 1.0
        # 三例中两例成功绑定（证据、规则），第三例两条应丢弃
        assert summary["bind_rate"] == 3 / 5

    def test_rows_cover_all_outcomes(self, pipeline):
        report = run_binding_evaluation(BINDING_EVAL, pipeline=pipeline)
        outcomes = {r.actual for r in report.rows}
        assert outcomes == {BIND_EVIDENCE, BIND_RULE, BIND_NONE}
        assert all(r.correct for r in report.rows)

    def test_requires_pipeline_or_paths(self):
        import pytest

        with pytest.raises(ValueError):
            run_binding_evaluation(BINDING_EVAL)


class TestE2EAnchorBindingMetric:
    def _result(self, reasons):
        return E2EResult(
            case_id=1,
            question="q",
            status="refuse",
            verdicts=[{"text": f"c{i}", "status": "pass", "reason": r} for i, r in enumerate(reasons)],
        )

    def test_none_when_no_verdicts(self):
        report = E2EReport(total=1, results=[self._result([])])
        assert report.anchor_binding_rate is None
        assert report.summary_dict()["anchor_binding_rate"] is None

    def test_rate_excludes_citation_incomplete(self):
        report = E2EReport(
            total=1,
            results=[self._result(["pass", "citation_incomplete", "rule_authoritative"])],
        )
        assert report.anchor_binding_rate == 2 / 3
        assert report.summary_dict()["anchor_binding_rate"] == round(2 / 3, 4)
