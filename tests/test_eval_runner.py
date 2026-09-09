"""评测 runner 单元测试：加载 / 三指标 / 对照打分 / CLI 退出码。"""

from __future__ import annotations

from pathlib import Path

from groundedrag.eval.runner import (
    baseline_questions,
    load_eval_set,
    run_evaluation,
    score_baseline,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = REPO_ROOT / "examples" / "eval_set.jsonl"
DOCS_PATH = REPO_ROOT / "examples" / "seed_docs.jsonl"
RULES_PATH = REPO_ROOT / "examples" / "seed_rules.json"


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


# ---------------------------------------------------------------------------
# E2E 端到端（用 stub answer 跑 run_e2e，不触网）
# ---------------------------------------------------------------------------
class _StubVerdict:
    def __init__(self, text="t", status="pass", reason="r"):
        self.claim = type("C", (), {"text": text})()
        self.status = status
        self.reason = reason


class _StubReport:
    def __init__(self, verdicts=None):
        self.verdicts = verdicts or []


class _StubAnswer:
    def __init__(self, status="pass", verdicts=None, used_llm=None, text="a"):
        self.status = status
        self.report = _StubReport(verdicts)
        self.used_llm = used_llm
        self.answer_text = text


class TestRunE2E:
    def test_basic_flow_matches(self, tmp_path):
        from groundedrag.eval.runner import E2EReport, run_e2e

        p = tmp_path / "cases.jsonl"
        p.write_text(
            '{"question": "q1", "expected_verdicts": "pass"}\n'
            '{"question": "q2", "expected_verdicts": "refuse"}\n'
            '{"question": "q3", "expected_verdicts": "pass"}\n',
            encoding="utf-8",
        )

        def fake_pipeline(q):
            status = "pass" if q != "q2" else "refuse"
            return _StubAnswer(status=status, used_llm="stub" if q == "q1" else None)

        report = run_e2e(p, fake_pipeline, expected_verdicts={0: "pass", 1: "refuse", 2: "pass"})
        assert isinstance(report, E2EReport)
        assert report.total == 3
        assert report.matched_count == 3
        assert report.mismatched_count == 0
        assert report.llm_used_count == 1
        assert report.rule_direct_count == 2
        assert report.match_rate == 1.0
        assert report.llm_usage_rate == 1 / 3
        assert report.summary_dict()["total"] == 3

    def test_mismatch_and_unknown_expected(self, tmp_path):
        from groundedrag.eval.runner import run_e2e

        p = tmp_path / "cases.jsonl"
        p.write_text(
            '{"question": "q1", "expected_verdicts": "pass"}\n'
            '{"question": "q2"}\n',  # 无 expected → matched=None
            encoding="utf-8",
        )

        def fake_pipeline(q):
            return _StubAnswer(status="refuse", verdicts=[_StubVerdict()])

        report = run_e2e(p, fake_pipeline)
        # q1 无 expected_verdicts 映射（caller 未传）→ matched None；q2 同
        assert report.total == 2
        judged = [r for r in report.results if r.matched is not None]
        assert not judged  # 全部未判定
        assert report.match_rate is None

    def test_report_render_has_detail_lines(self, tmp_path):
        from groundedrag.eval.runner import run_e2e

        p = tmp_path / "cases.jsonl"
        p.write_text('{"question": "q1"}\n', encoding="utf-8")
        report = run_e2e(p, lambda q: _StubAnswer(status="pass", verdicts=[_StubVerdict(text="主张A", status="pass")]))
        text = report.render()
        assert "E2E" in text
        assert "主张A" in text


class TestRunEvaluationErrors:
    def test_requires_docs_or_pipeline(self):
        from groundedrag.eval.runner import run_evaluation

        try:
            run_evaluation("nope.jsonl")
        except ValueError as e:
            assert "pipeline" in str(e) or "docs" in str(e)
        else:
            raise AssertionError("expected ValueError")


class TestE2ECLI:
    """_run_e2e_cli 的健全性分支：未知 provider / 无 API key → 非零退出（不触网）。"""

    def test_unknown_provider_returns_1(self, monkeypatch, tmp_path, capsys):
        from groundedrag.eval.runner import main

        monkeypatch.setenv("LLM_API_KEY", "sk-x")
        monkeypatch.chdir(tmp_path)  # 无 .env
        rc = main(["--e2e", "--llm-provider", "nope",
                   "--docs", str(DOCS_PATH), "--rules", str(RULES_PATH),
                   "--eval", str(EVAL_PATH)])
        assert rc == 1
        assert "不支持" in capsys.readouterr().out

    def test_no_api_key_returns_1(self, monkeypatch, tmp_path, capsys):
        from groundedrag.eval.runner import main

        for k in ("LLM_API_KEY", "XIAOMI_API_KEY", "DEEPSEEK_API_KEY", "DOUBAO_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.chdir(tmp_path)
        rc = main(["--e2e", "--llm-provider", "deepseek",
                   "--docs", str(DOCS_PATH), "--rules", str(RULES_PATH),
                   "--eval", str(EVAL_PATH)])
        assert rc == 1
        assert "API Key" in capsys.readouterr().out

    def test_render_eval_report(self, eval_report):
        text = eval_report.render()
        assert "eval 结果" in text
        assert "逐组明细" in text

    def test_e2e_success_path_with_fake_llm(self, monkeypatch, tmp_path, capsys):
        """完整 E2E CLI 成功路径（mock LLM，不触网）：主 provider + fallback 循环 +
        e2e_max 临时文件裁剪 + JSON 摘要输出。"""
        import json

        from groundedrag.eval.runner import main

        class FakeOpenAILLM:
            """替代 OpenAICompatibleLLM：is_available 恒真，generate 返回规则直出文案。"""

            def __init__(self, base_url="", api_key="", model="", **kwargs):
                self.base_url = base_url
                self.api_key = api_key
                self.model = model

            def is_available(self):
                return True

            def generate(self, prompt, *, temperature=0.3, max_tokens=None):
                return "- 肺癌 一线 EGFR 推荐方案：奥希替尼[规则1]"

        import groundedrag.llm.openai_compat as oci_mod

        monkeypatch.setattr(oci_mod, "OpenAICompatibleLLM", FakeOpenAILLM)
        # 主 + 备用都有 key（同一统一 key 即可覆盖 fallback 循环）
        monkeypatch.setenv("LLM_API_KEY", "sk-fake")
        monkeypatch.setenv("LLM_PROVIDER", "xiaomi")
        monkeypatch.chdir(tmp_path)

        e2e_set = REPO_ROOT / "examples" / "e2e_eval_set.jsonl"
        rc = main(["--e2e", "--llm-provider", "deepseek",
                   "--docs", str(DOCS_PATH), "--rules", str(RULES_PATH),
                   "--eval", str(e2e_set), "--json", "--e2e-max", "1"])
        out = capsys.readouterr().out
        assert rc == 0
        payload = json.loads(out)
        assert payload["total"] == 1
        assert "verdict_distribution" in payload

    def test_e2e_text_mode_render(self, monkeypatch, tmp_path, capsys):
        """非 --json 模式的 E2E CLI 输出渲染。"""
        import groundedrag.llm.openai_compat as oci_mod
        from groundedrag.eval.runner import main

        class FakeLLM:
            def __init__(self, *a, **k):
                self.base_url = k.get("base_url", "")
                self.api_key = k.get("api_key", "")
                self.model = k.get("model", "")

            def is_available(self):
                return True

            def generate(self, prompt, **kwargs):
                return "- 肺癌 一线 EGFR 推荐方案：奥希替尼[规则1]"

        monkeypatch.setattr(oci_mod, "OpenAICompatibleLLM", FakeLLM)
        monkeypatch.setenv("LLM_API_KEY", "sk-fake")
        monkeypatch.setenv("LLM_PROVIDER", "xiaomi")
        monkeypatch.chdir(tmp_path)

        e2e_set = REPO_ROOT / "examples" / "e2e_eval_set.jsonl"
        rc = main(["--e2e", "--llm-provider", "xiaomi",
                   "--docs", str(DOCS_PATH), "--rules", str(RULES_PATH),
                   "--eval", str(e2e_set)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "E2E" in out
