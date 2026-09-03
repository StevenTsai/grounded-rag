"""测试共享夹具：路径与内置示例流水线。

所有 examples/ 路径都以本文件所在仓库根为基准解析，测试从任意 CWD 运行均可靠。
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
DOCS_PATH = EXAMPLES_DIR / "seed_docs.jsonl"
RULES_PATH = EXAMPLES_DIR / "seed_rules.json"
EVAL_PATH = EXAMPLES_DIR / "eval_set.jsonl"


@pytest.fixture(scope="session")
def pipeline() -> "object":
    """用内置种子文档/规则构建一个只读 Pipeline（模块间复用）。"""
    from groundedrag.pipeline import Pipeline

    return Pipeline.build_from_json(DOCS_PATH, RULES_PATH)


@pytest.fixture(scope="session")
def eval_report() -> "object":
    """跑完整套内置评测集，返回 EvalReport（验证示例自身口径恒定）。"""
    from groundedrag.eval.runner import run_evaluation

    return run_evaluation(EVAL_PATH, docs_path=DOCS_PATH, rules_path=RULES_PATH)
