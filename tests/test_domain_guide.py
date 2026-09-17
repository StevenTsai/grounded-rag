"""领域适配指南（docs/domain_guide.md）可执行性测试 —— 防止文档与实现漂移。

覆盖：
- 指南的金融示例确实可跑通（非医疗字段 fund_type 自动提取 + 规则直出）；
- 指南引用的扩展点 API 名正确（`extract_context` / `pipe.engine`）；
- 文档中出现的所有 source_type 取值都在实现白名单内（否则 schema 校验必失败）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from groundedrag.guardrail.engine import GuidelineEngine
from groundedrag.guardrail.evidence import VALID_SOURCE_TYPES
from groundedrag.llm.template import TemplateLLM
from groundedrag.pipeline import Pipeline
from groundedrag.retriever.retriever import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
GUIDE_PATH = REPO_ROOT / "docs" / "domain_guide.md"

# 与 docs/domain_guide.md「完整示例：金融合规」保持一致
FINANCE_DOC = {
    "doc_id": "doc-mica-001",
    "content": "货币市场基金投资组合的平均剩余到期期限不得超过120天。",
    "source_type": "regulation",
    "source_version": "证监会-2026",
    "grade": "A",
    "updated_at": "2026-01-01",
    "entities": ["货币市场基金", "120天"],
}
FINANCE_RULES = {
    "pathway_rules": [
        {
            "rule_id": "rr-mmf-term",
            "name": "货币市场基金期限限制",
            "conditions": [{"field": "fund_type", "op": "eq", "value": "货币市场基金"}],
            "recommendations": [
                {
                    "plan_name": "平均剩余到期期限≤120天",
                    "grade": "A",
                    "source_version": "证监会-2026",
                }
            ],
            "source_version": "证监会-2026",
            "updated_at": "2026-01-01",
        }
    ]
}


class TestFinanceExampleRuns:
    def _pipeline(self) -> Pipeline:
        docs = [Document.from_dict(FINANCE_DOC)]
        # TemplateLLM = 无 API Key 时的确定性兜底，规则命中即规则直出
        return Pipeline.build(docs, FINANCE_RULES, llm=TemplateLLM())

    def test_fund_type_extracted_without_custom_engine(self):
        pipe = self._pipeline()
        ctx = pipe.engine.extract_context("货币市场基金的期限限制是什么？")
        assert ctx.get("fund_type") == "货币市场基金"

    def test_rule_direct_passes_with_regulation_source(self):
        pipe = self._pipeline()
        result = pipe.ask("货币市场基金的期限限制是什么？")
        assert result.status == "pass"
        assert "平均剩余到期期限≤120天" in result.answer_text
        assert [m.rule_id for m in result.matched_rules] == ["rr-mmf-term"]


class TestDocumentedExtensionPoint:
    def test_extract_context_override_and_engine_swap(self):
        """指南给出的扩展点：子类覆盖 extract_context + 替换 pipe.engine。"""

        class MyEngine(GuidelineEngine):
            def extract_context(self, question, synonym_map=None):
                ctx = super().extract_context(question, synonym_map)
                ctx["region"] = "默认区"
                return ctx

        pipe = Pipeline.build(
            [Document.from_dict(FINANCE_DOC)], FINANCE_RULES, llm=TemplateLLM()
        )
        pipe.engine = MyEngine(
            pipe.engine.pathway_rules, pipe.engine.resistance_rules
        )
        assert pipe.engine.extract_context("任意问题").get("region") == "默认区"


class TestGuideNoDrift:
    def _guide_text(self) -> str:
        return GUIDE_PATH.read_text(encoding="utf-8")

    def test_all_documented_source_types_are_valid(self):
        text = self._guide_text()
        # 文档中明确以 source_type 形式出现的取值必须都在实现白名单里
        values = set(re.findall(r'"source_type"\s*:\s*"([^"]+)"', text))
        assert values, "领域指南应至少包含一个 source_type 示例"
        assert values <= set(VALID_SOURCE_TYPES), (
            f"文档出现非法 source_type: {values - set(VALID_SOURCE_TYPES)}"
        )

    def test_no_stale_api_names(self):
        text = self._guide_text()
        # 旧实现/口误的 API 名不得再出现在文档中
        assert "_extract_context" not in text
        assert "guideline_engine" not in text

    def test_documented_conditions_field_is_generic(self):
        # 金融示例使用非医疗字段 fund_type —— 通用兜底提取必须支持
        rules = json.loads(json.dumps(FINANCE_RULES))
        cond_fields = {
            c["field"]
            for r in rules["pathway_rules"]
            for c in r["conditions"]
        }
        assert cond_fields == {"fund_type"}
