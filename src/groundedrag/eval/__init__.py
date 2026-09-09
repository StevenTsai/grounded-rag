"""评测：内置 eval_set 跑分（引用完整性率 / 表面一致率 / 拒答正确率 + E2E 端到端）。"""

from groundedrag.eval.runner import (
    E2EReport,
    E2EResult,
    EvalReport,
    baseline_questions,
    load_eval_set,
    run_e2e,
    run_evaluation,
    score_baseline,
)

__all__ = [
    "E2EReport",
    "E2EResult",
    "EvalReport",
    "baseline_questions",
    "load_eval_set",
    "run_e2e",
    "run_evaluation",
    "score_baseline",
]
