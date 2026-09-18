#!/usr/bin/env python3
"""只挂校验门的最小集成示例（无需 API Key、无需网络）。

场景：你已有自己的检索与 LLM 生成，只想在输出前加一道确定性校验门。

三步：
    1) 你的检索结果 → ``EvidenceId`` + ``EvidenceRegistry``
    2) 你的 LLM 输出 → ``parse_claims``（支持 JSON，或 [证据N]/[规则N] 锚点文本）
    3) ``Verifier().verify(...)`` → PASS / ANNOTATE / REFUSE（逐条 verdict 带原因）

运行：
    python examples/integrate_verifier.py

完整指南见 docs/integration.md。
"""

from __future__ import annotations

import json

from groundedrag.guardrail import (
    EvidenceId,
    EvidenceRegistry,
    Verifier,
    parse_claims,
)

# 1) 已有 RAG 检索回来的证据（此处为演示用合成片段）
EVIDENCES = [
    EvidenceId(
        evidence_id="ev1",
        doc_id="demo-guideline-1",
        source_type="guideline",
        source_version="CSCO-LUNG-2026",
        grade="A",
        updated_at="2026-02-01",
        text_span="对于 EGFR 突变阳性的晚期非小细胞肺癌，一线治疗推荐奥希替尼，推荐剂量 80mg，每日一次。",
    ),
]

# 2) 已有 RAG 的 LLM 输出（JSON 主张；也可用 "[证据1]" 锚点文本）
OUTPUTS = {
    "有据主张（期望 PASS）": json.dumps(
        [{"text": "EGFR 突变肺癌一线推荐奥希替尼，剂量 80mg", "evidence_refs": ["ev1"]}],
        ensure_ascii=False,
    ),
    "数值编造（期望 REFUSE）": json.dumps(
        [{"text": "EGFR 突变肺癌一线推荐奥希替尼，剂量 120mg", "evidence_refs": ["ev1"]}],
        ensure_ascii=False,
    ),
    "无锚点治疗建议（期望 REFUSE）": json.dumps(
        [{"text": "肺癌一线推荐使用某新药"}],
        ensure_ascii=False,
    ),
}


def verify_output(raw: str) -> None:
    registry = EvidenceRegistry(EVIDENCES)
    claims = parse_claims(raw, evidence_ids=[e.evidence_id for e in EVIDENCES])
    report = Verifier().verify(claims, registry)
    print(f"  overall = {report.overall_status}  （{report.message}）")
    for v in report.verdicts:
        print(f"  - [{v.status}] {v.claim.text}  |  reason = {v.reason or '-'}")


def main() -> int:
    print("GroundedRAG 校验门 · 已有 RAG 输出接入演示")
    for title, raw in OUTPUTS.items():
        print(f"\n[{title}]")
        verify_output(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
