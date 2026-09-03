"""评测：内置 eval_set 跑分（引用完整性率 / 表面一致率 / 拒答正确率）。"""

from groundedrag.eval.runner import (
    EvalReport,
    baseline_questions,
    load_eval_set,
    run_evaluation,
    score_baseline,
)

__all__ = [
    "EvalReport",
    "baseline_questions",
    "load_eval_set",
    "run_evaluation",
    "score_baseline",
]
