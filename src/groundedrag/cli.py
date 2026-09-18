"""GroundedRAG CLI —— 一行命令即可问答。

用法::

    groundedrag ask "EGFR突变肺癌一线推荐什么方案？"
    groundedrag ask "..." --docs my_docs.jsonl --rules my_rules.json
    groundedrag ask "..." --provider deepseek   # 读 .env 配置
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence


def _load_dotenv(path: str = ".env") -> None:
    """轻量 .env 加载器（委托 llm.providers，保持 CLI 内可调用）。"""
    from groundedrag.llm.providers import load_dotenv

    load_dotenv(path)


def _build_llm(provider: Optional[str] = None):
    """从 .env / 环境变量构造 LLM（无 key / 未知 provider 返回 None）。"""
    from groundedrag.llm.providers import PROVIDER_PRESETS, build_failover, resolve_provider

    prov = resolve_provider(provider)
    if prov not in PROVIDER_PRESETS:
        print(
            f"不支持的 provider: {prov}，可选: {', '.join(PROVIDER_PRESETS)}",
            file=sys.stderr,
        )
        return None
    return build_failover(prov)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="groundedrag",
        description="GroundedRAG —— 有据可依、无据可拒的可信问答",
    )
    sub = parser.add_subparsers(dest="command")

    # groundedrag ask "question"
    ask_p = sub.add_parser("ask", help="提问（基于本地规则 + 证据）")
    ask_p.add_argument("question", help="问题")
    ask_p.add_argument("--docs", default="examples/seed_docs.jsonl", help="证据文档（JSONL）")
    ask_p.add_argument("--rules", default="examples/seed_rules.json", help="规则库（JSON）")
    ask_p.add_argument("--provider", default=None, help="LLM provider（xiaomi/deepseek/doubao）")
    ask_p.add_argument("--top-k", type=int, default=5, help="检索条数")
    ask_p.add_argument("--json", action="store_true", help="输出 JSON 格式")

    # groundedrag init
    init_p = sub.add_parser("init", help="生成示例 docs + rules 模板")
    init_p.add_argument("--dir", default=".", help="输出目录")

    args = parser.parse_args(argv)

    if args.command == "ask":
        return _cmd_ask(args)
    elif args.command == "init":
        return _cmd_init(args)
    else:
        parser.print_help()
        return 1


def _cmd_ask(args: argparse.Namespace) -> int:
    _load_dotenv()

    from groundedrag.pipeline import Pipeline

    llm = _build_llm(args.provider)
    pipe = Pipeline.build_from_json(args.docs, args.rules, llm=llm)

    result = pipe.ask(args.question, top_k=args.top_k)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.answer_text)
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    """生成示例模板文件。"""
    import pathlib

    d = pathlib.Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)

    docs_path = d / "my_seed_docs.jsonl"
    rules_path = d / "my_seed_rules.json"

    if not docs_path.exists():
        docs_path.write_text(
            '{"doc_id": "doc-example-001", "title": "示例文档", "content": "奥希替尼（Osimertinib）是第三代 EGFR-TKI，推荐剂量 80mg 每日一次。", '
            '"source_type": "guideline", "source_version": "CSCO-2026", "grade": "A", '
            '"updated_at": "2026-01-01", "entities": ["奥希替尼", "EGFR", "肺癌"]}\n',
            encoding="utf-8",
        )
        print(f"已生成 {docs_path}")

    if not rules_path.exists():
        rules_path.write_text(
            json.dumps(
                {
                    "pathway_rules": [
                        {
                            "rule_id": "rr-example-001",
                            "name": "示例规则",
                            "cancer_type": "肺癌",
                            "treatment_line": "一线",
                            "biomarker": "EGFR",
                            "conditions": [
                                {"field": "cancer_type", "op": "eq", "value": "肺癌"},
                                {"field": "biomarker", "op": "eq", "value": "EGFR"},
                            ],
                            "recommendations": [
                                {"plan_name": "奥希替尼", "grade": "A", "note": "推荐剂量80mg每日一次", "source_version": "CSCO-2026"}
                            ],
                            "source_version": "CSCO-2026",
                            "updated_at": "2026-01-01",
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"已生成 {rules_path}")

    print("\n下一步：")
    print(f"  1. 编辑 {docs_path} 填入你的证据文档")
    print(f"  2. 编辑 {rules_path} 填入你的规则")
    print(f"  3. groundedrag ask \"你的问题\" --docs {docs_path} --rules {rules_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover —— python -m 入口，由集成测试覆盖
    raise SystemExit(main())
