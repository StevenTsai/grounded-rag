"""评测 runner 单元测试：加载 / 三指标 / 对照打分 / CLI 退出码。"""

from __future__ import annotations

from pathlib import Path

from groundedrag.eval.runner import (
    baseline_questions,
    load_eval_set,
    run_evaluation,
    score_baseline,
)

EVAL_PATH = Path(__file__).resolve().parent.parent / "examples" / "eval_set.jsonl"
DOCS_PATH = Path(__file__).resolve().parent.parent / "examples" / "seed_docs.jsonl"
RULES_PATH = Path(__file__).resolve().parent.parent / "examples" / "seed_rules.json"


class TestLoadEvalSet:
    def test_loads_nonempty(self):
        cases = load_eval_set(EVAL_PATH)
        assert len(cases) >= 1
        assert all("question" in c for c in cases)


class TestRunEvaluation:
    def test_example_metrics_are_high(self, eval_report):
        s = eval_report.summary_dict()
        # 三条头条指标按内置评测集构造应稳定全绿
        assert s["citation_completeness_rate"] >= 0.95
        assert s["surface_consistency_rate"] >= 0.95
        assert s["refusal_correctness_rate"] >= 0.95
        assert s["case_count"] == len(load_eval_set(EVAL_PATH))

    def test_run_evaluation_api_direct(self):
        # 直接走 runner.run_evaluation（自建 pipeline 亦可），保证公开入口可用
        report = run_evaluation(EVAL_PATH, docs_path=DOCS_PATH, rules_path=RULES_PATH)
        assert len(report.cases) == len(load_eval_set(EVAL_PATH))

    def test_case_rows_carry_expected_actual(self, eval_report):
        rows = [r for c in eval_report.cases for r in c.claim_rows]
        assert rows
        assert all(r.expected in ("pass", "refuse", "annotate") for r in rows)
        assert all(r.actual in ("pass", "refuse", "annotate") for r in rows)

    def test_refusal_accuracy_computed(self, eval_report):
        # 整组级拒答判定正确率：内置集全部正确
        assert eval_report.answer_refusal_accuracy == 1.0


class TestBaseline:
    def test_baseline_questions_count(self):
        cases = load_eval_set(EVAL_PATH)
        base = baseline_questions(cases)
        assert len(base) == len(cases)
        assert all("question" in b and "expect_refusal" in b for b in base)

    def test_score_baseline_all_caught(self):
        cases = load_eval_set(EVAL_PATH)
        base = baseline_questions(cases)
        responses = {b["question"]: True for b in base}  # 裸模型全拒
        out = score_baseline(responses, base)
        assert out["refusal_expected_caught"] == 1.0
        assert out["false_refusal"] == 1.0  # 连好题也拒

    def test_score_baseline_never_refuses(self):
        cases = load_eval_set(EVAL_PATH)
        base = baseline_questions(cases)
        responses = {b["question"]: False for b in base}  # 裸模型从不拒
        out = score_baseline(responses, base)
        assert out["refusal_expected_caught"] == 0.0

    def test_score_baseline_ignores_missing(self):
        base = [{"question": "q1", "expect_refusal": True}]
        out = score_baseline({}, base)
        assert out["answered_count"] == 0
        assert out["refusal_expected_caught"] is None


class TestCLI:
    def test_main_json_exits_zero(self):
        from groundedrag.eval.runner import main

        rc = main(["--eval", str(EVAL_PATH), "--docs", str(DOCS_PATH),
                   "--rules", str(RULES_PATH), "--json"])
        assert rc == 0
