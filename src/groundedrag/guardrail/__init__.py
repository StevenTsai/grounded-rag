"""约束生成层：规则引擎 + 证据溯源 + 声明解析 + 声明级校验门。"""

from groundedrag.guardrail.claims import (
    AnswerClaim,
    classify_claim_type,
    is_critical_claim,
    parse_anchor_text,
    parse_claims,
    parse_json_claims,
    surface_tokens,
)
from groundedrag.guardrail.engine import GuidelineEngine
from groundedrag.guardrail.evidence import (
    EvidenceId,
    EvidenceRegistry,
    StalenessPolicy,
    from_retrieved_docs,
)
from groundedrag.guardrail.models import (
    ResistanceRule,
    Rule,
    RuleCondition,
    RuleDecision,
    RuleRecommendation,
)
from groundedrag.guardrail.provider import DictRuleProvider, JsonRuleProvider, RuleProvider
from groundedrag.guardrail.verifier import (
    ANNOTATE,
    PASS,
    REFUSE,
    ClaimVerdict,
    Verifier,
    VerifierConfig,
    VerifyReport,
)

__all__ = [
    "ANNOTATE",
    "AnswerClaim",
    "ClaimVerdict",
    "DictRuleProvider",
    "EvidenceId",
    "EvidenceRegistry",
    "GuidelineEngine",
    "JsonRuleProvider",
    "PASS",
    "REFUSE",
    "ResistanceRule",
    "Rule",
    "RuleCondition",
    "RuleDecision",
    "RuleProvider",
    "RuleRecommendation",
    "StalenessPolicy",
    "VerifyReport",
    "Verifier",
    "VerifierConfig",
    "classify_claim_type",
    "from_retrieved_docs",
    "is_critical_claim",
    "parse_anchor_text",
    "parse_claims",
    "parse_json_claims",
    "surface_tokens",
]
