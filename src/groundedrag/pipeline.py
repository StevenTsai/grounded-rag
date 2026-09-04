"""可信问答编排流水线（pipeline.py）。

数据流（对应设计文档 §3.1 架构图）：:

    检索 → 规则匹配 → 约束生成 → 声明解析 → 声明级校验 → 降级 / 拒答
    ─────────────┐   ──────────────────────────────────────────────┘
     retriever/  │              guardrail/ (★核心) + llm/
                 └── 命中规则可作"规则直出"，无需 LLM 也有据可答

对外主入口为 :class:`Pipeline`：``build_from_json(...)`` 加载种子数据后
``ask(...)`` 得到一次可信回答（:class:`PipelineResult`）。

无 API Key 时（LLM 全部不可用）pipeline 仍可工作：
- 问题命中规则 → 规则直出（把胜出的 RuleDecision 合成为有锚点主张并过校验门）；
- 未命中规则 → 结构化拒答模板，绝不编造。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from groundedrag.guardrail.claims import (
    TYPE_INDICATION,
    AnswerClaim,
    parse_claims,
)
from groundedrag.guardrail.evidence import (
    EvidenceId,
    EvidenceRegistry,
    from_retrieved_docs,
)
from groundedrag.guardrail.models import RuleDecision
from groundedrag.guardrail.provider import DictRuleProvider, JsonRuleProvider
from groundedrag.guardrail.verifier import (
    ANNOTATE,
    PASS,
    REFUSE,
    Verifier,
    VerifierConfig,
    VerifyReport,
)
from groundedrag.llm.base import LLMService
from groundedrag.llm.failover import FailoverLLM
from groundedrag.llm.template import (
    DIVERGENCE_NOTE,
    REFUSAL_TEMPLATE,
    UNVERIFIED_NOTE,
    TemplateLLM,
)
from groundedrag.retriever.retriever import (
    Document,
    RetrievedDocument,
    Retriever,
)

__all__ = [
    "DIVERGENCE_NOTE",
    "REFUSAL_TEMPLATE",
    "UNVERIFIED_NOTE",
    "Pipeline",
    "PipelineResult",
    "load_docs_jsonl",
]

# 模板拒答文案开头的识别（LLM 回退到模板时，parse 出的"主张"应被丢弃）
_TEMPLATE_HEADS = (
    "抱歉",
    "现有检索证据不足以支撑",
)

# 部分拒答提示：有通过校验的主张保留输出，被拒答的关键主张仅提示不展示
REFUSE_DROPPED_NOTE = "（部分主张因证据不足或相互矛盾已被校验门拒绝，未在上文呈现）"


@dataclass
class PipelineResult:
    """一次可信问答的完整产物（演示 / 评测 / 测试共用）。"""

    question: str
    answer_text: str = ""                       # 最终对外文本（已按校验结果降级/拒答）
    claims: List[AnswerClaim] = field(default_factory=list)
    report: Optional[VerifyReport] = None
    matched_rules: List[RuleDecision] = field(default_factory=list)
    evidence: List[EvidenceId] = field(default_factory=list)
    retrieved: List[RetrievedDocument] = field(default_factory=list)
    used_llm: Optional[str] = None              # 实际完成生成的模型名（None=规则直出）
    note: str = ""                              # 附加说明（如规则分歧声明）

    @property
    def status(self) -> str:
        return self.report.overall_status if self.report else REFUSE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer_text": self.answer_text,
            "status": self.status,
            "claims": [c.to_dict() for c in self.claims],
            "report": self.report.to_dict() if self.report else None,
            "matched_rules": [r.to_dict() for r in self.matched_rules],
            "evidence": [e.to_dict() for e in self.evidence],
            "retrieved": [r.to_dict() for r in self.retrieved],
            "used_llm": self.used_llm,
            "note": self.note,
        }


def load_docs_jsonl(path: Union[str, Path]) -> List[Document]:
    """从 JSONL 加载检索文档（每行一个 Document 的 dict 表示）。"""
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict) and (data.get("content") or data.get("text")):
                docs.append(Document.from_dict(data))
    return docs


class Pipeline:
    """可信问答编排流水线。

    组装四要素并统一其数据流：
    ``Retriever``（检索）→ ``GuidelineEngine``（规则匹配）→ ``Verifier``（★校验门）
    + ``LLMService``（受约束生成，主备降级 + 模板回退）。
    """

    def __init__(
        self,
        documents: Sequence[Document],
        pathway_rules: Sequence[Any] = (),
        resistance_rules: Sequence[Any] = (),
        *,
        synonym_map: Optional[Dict[str, List[str]]] = None,
        verifier_config: Optional[VerifierConfig] = None,
        llm: Optional[LLMService] = None,
    ) -> None:
        self.documents = list(documents)
        self.synonym_map = dict(synonym_map or {})
        self.retriever = Retriever(self.documents, synonym_map=self.synonym_map)
        from groundedrag.guardrail.engine import GuidelineEngine

        self.engine = GuidelineEngine(
            list(pathway_rules), list(resistance_rules)
        )
        self.verifier = Verifier(
            verifier_config
            or VerifierConfig(synonym_map=self.synonym_map or None)
        )
        self.llm = llm or FailoverLLM()
        self._docs_by_id: Dict[str, Document] = {
            d.doc_id: d for d in self.documents if d.doc_id
        }

    # ------------------------------------------------------------------
    # 构造入口
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        rules_payload: Optional[Mapping[str, Any]] = None,
        *,
        synonym_map: Optional[Dict[str, List[str]]] = None,
        verifier_config: Optional[VerifierConfig] = None,
        llm: Optional[LLMService] = None,
    ) -> "Pipeline":
        """从文档列表与规则 dict 构造（不依赖文件系统）。"""
        provider = DictRuleProvider(dict(rules_payload or {}))
        return cls(
            documents,
            pathway_rules=provider.load_pathway_rules(),
            resistance_rules=provider.load_resistance_rules(),
            synonym_map=synonym_map,
            verifier_config=verifier_config,
            llm=llm,
        )

    @classmethod
    def build_from_json(
        cls,
        docs_path: Union[str, Path],
        rules_path: Union[str, Path],
        *,
        synonym_map: Optional[Dict[str, List[str]]] = None,
        verifier_config: Optional[VerifierConfig] = None,
        llm: Optional[LLMService] = None,
    ) -> "Pipeline":
        """从 ``seed_docs.jsonl`` + ``seed_rules.json`` 构造（demo/评测主入口）。"""
        documents = load_docs_jsonl(docs_path)
        provider = JsonRuleProvider(rules_path)
        return cls(
            documents,
            pathway_rules=provider.load_pathway_rules(),
            resistance_rules=provider.load_resistance_rules(),
            synonym_map=synonym_map,
            verifier_config=verifier_config,
            llm=llm,
        )

    # ------------------------------------------------------------------
    # 检索 + 规则匹配（两个来源统一到一个上下文）
    # ------------------------------------------------------------------
    def context_for(
        self, question: str, context: Optional[Mapping[str, str]] = None
    ) -> Dict[str, str]:
        """上下文：显式传入优先，缺失字段逐字段用启发式从问题补齐。

        原先仅在三个字段全部缺失时才抽取，导致调用方只给部分 context
        （如只给了 treatment_line）时，cancer_type/biomarker 永远不会从问题补全，
        规则匹配静默失败。现在始终抽取，仅 setdefault 填充缺失字段。
        """
        ctx: Dict[str, str] = dict(context or {})
        extracted = self.engine.extract_context(question, self.synonym_map)
        for k, v in extracted.items():
            ctx.setdefault(k, v)
        return ctx

    def retrieve_and_match(
        self,
        question: str,
        *,
        context: Optional[Mapping[str, str]] = None,
        top_k: int = 5,
        evidence_docs: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """检索文档 + 匹配规则，归一为 (matched, registry, evidence_ids, rule_ids, retrieved)。

        ``evidence_docs`` 可钉死证据文档 id 列表（评测用：保证可复现），
        其余槽位用检索结果补齐并去重；锚点编号按返回顺序 1 起。
        """
        ctx = self.context_for(question, context)
        matched = self.engine.match(ctx)

        if evidence_docs is not None:
            pinned: List[RetrievedDocument] = []
            pinned_ids: List[str] = []
            for doc_id in evidence_docs:
                doc = self._docs_by_id.get(doc_id)
                if doc is not None:
                    pinned.append(RetrievedDocument(doc, 0.0))
                    pinned_ids.append(doc_id)
            extra = [
                r
                for r in self.retriever.retrieve(question, top_k=top_k)
                if r.document.doc_id not in pinned_ids
            ]
            retrieved = pinned + extra
        else:
            retrieved = self.retriever.retrieve(question, top_k=top_k)

        evidences = from_retrieved_docs(retrieved)
        registry = EvidenceRegistry(evidences)
        return {
            "matched": matched,
            "registry": registry,
            "evidence_ids": [e.evidence_id for e in evidences],
            "rule_ids": [d.rule_id for d in matched],
            "retrieved": retrieved,
            "context": ctx,
        }

    # ------------------------------------------------------------------
    # 规则直出（LLM 未产出有效主张时，命中的权威规则本身即答案）
    # ------------------------------------------------------------------
    def _claims_from_winners(
        self, matched: Sequence[RuleDecision]
    ) -> tuple[List[AnswerClaim], List[str]]:
        """把裁定胜出的规则推荐合成为有锚点主张（type=indication，绑定 rule_refs）。

        返回 (claims, notes)；notes 记录分歧/过期等说明，供上层附注。
        裁定失败（无胜出者 = 分歧或全部过期）→ 返回空 claims + 说明。
        """
        winner_ids, notes = self.verifier._arbitrate_matched(matched)  # noqa: SLF001 —— 复用同一裁定逻辑
        claims: List[AnswerClaim] = []
        for dec in matched:
            if dec.rule_id not in winner_ids:
                continue
            scope = " ".join(
                x for x in (dec.cancer_type or "", dec.treatment_line or "", dec.biomarker or "") if x
            )
            for rec in dec.recommendations:
                if not rec.plan_name:
                    continue
                prefix = f"{scope} 推荐方案" if scope else "推荐方案"
                text = f"{prefix}：{rec.plan_name}"
                claims.append(
                    AnswerClaim(
                        text=text,
                        type=TYPE_INDICATION,
                        rule_refs=[dec.rule_id],
                        critical=True,
                    )
                )
        return claims, notes

    # ------------------------------------------------------------------
    # 生成（受约束）：构造 prompt → 调 LLM → 解析出主张
    # ------------------------------------------------------------------
    def _generate(
        self,
        llm: LLMService,
        question: str,
        *,
        evidences: Sequence[EvidenceId],
        rules: Sequence[RuleDecision],
    ) -> tuple[str, str]:
        payload: Dict[str, Any] = {
            "evidences": [e.to_dict() for e in evidences],
            "rules": [r.to_dict() for r in rules],
        }
        context_block = llm.render_context(payload)
        prompt = (
            "请基于以下「可引用证据」与「命中的权威规则」回答问题。\n\n"
            f"{context_block}\n\n"
            "输出要求（只输出主张列表，不要解释）：\n"
            "1) 把回答拆成一条条可核查的原子主张；\n"
            "2) 每条主张后标注引用锚点：有证据支持写 [证据N]（N 为编号）；"
            "命中规则时优先写 [规则N]；\n"
            "3) 证据/规则未覆盖的内容不要写，不确定就只写『无法确认』；\n"
            f"问题：{question}"
        )
        raw = llm.generate(prompt)
        # FailoverLLM 记录最近一次实际完成生成的子服务（真实模型名）；其余服务用其自身 name。
        used = getattr(llm, "last_used_name", None) or getattr(llm, "name", None) or "llm"
        return raw, used

    @staticmethod
    def _drop_template_claims(claims: Sequence[AnswerClaim]) -> List[AnswerClaim]:
        """丢弃模板拒答文案被误解析出的"无引用主张"。"""
        out = []
        for c in claims:
            if not c.text:
                continue
            if any(c.text.startswith(h) for h in _TEMPLATE_HEADS):
                continue
            out.append(c)
        return out

    # ------------------------------------------------------------------
    # 回答组装
    # ------------------------------------------------------------------
    @staticmethod
    def _assemble_answer(report: VerifyReport) -> str:
        """按逐条校验结果组装对外文本（声明级粒度降级）。

        - PASS 主张 → 原样输出；
        - ANNOTATE 主张（非关键、未过校验）→ 附「未经本地证据核验」标注后输出；
        - REFUSE 关键主张 → 一律不输出。

        关键主张被拒且**无任何**通过主张 → 整段保守拒答（校验门判定"无据可答"）；
        仍有通过主张 → 保留通过项并附"部分主张被拒"提示（声明级删除，不误伤好主张）。
        """
        refused_critical = [
            v for v in report.verdicts if v.status == REFUSE and v.effective_critical
        ]
        passed = [v for v in report.verdicts if v.status == PASS]
        if refused_critical and not passed:
            return REFUSAL_TEMPLATE
        parts: List[str] = []
        for v in passed:
            parts.append(v.claim.text)
        if not refused_critical:
            for v in report.verdicts:
                if v.status == ANNOTATE:
                    parts.append(f"{v.claim.text}（{UNVERIFIED_NOTE}）")
        if refused_critical and parts:
            parts.append(REFUSE_DROPPED_NOTE)
        return "\n".join(f"- {p}" for p in parts)

    # ------------------------------------------------------------------
    # 主入口：可信问答
    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        *,
        context: Optional[Mapping[str, str]] = None,
        top_k: int = 5,
        evidence_docs: Optional[Sequence[str]] = None,
        llm: Optional[LLMService] = None,
        rule_fallback: bool = True,
    ) -> PipelineResult:
        """一次可信问答：检索 → 规则匹配 → 受约束生成 → 声明级校验 → 降级/拒答。

        ``evidence_docs`` 可钉死证据文档 id（评测/演示用例可复现用）；
        为 None 时按检索结果自动取回。
        """
        info = self.retrieve_and_match(
            question, context=context, top_k=top_k, evidence_docs=evidence_docs
        )
        matched: List[RuleDecision] = info["matched"]
        evidences: List[EvidenceId] = info["registry"].all()
        # 注意：锚点编号依赖检索返回顺序 → 用有序 evidence_ids 传给解析
        service = llm or self.llm

        if service.is_available():
            raw, used_name = self._generate(
                service, question, evidences=evidences, rules=matched
            )
        else:
            raw, used_name = "", None

        claims = parse_claims(
            raw,
            evidence_ids=info["evidence_ids"],
            rule_ids=info["rule_ids"],
        )
        claims = self._drop_template_claims(claims)

        # LLM 未产出有效主张：有命中规则 → 规则直出；否则交给 verifier 判无据拒答。
        note = ""
        if not claims and matched and rule_fallback:
            syn_claims, notes = self._claims_from_winners(matched)
            if syn_claims:
                claims = syn_claims
                used_name = None
                if notes:
                    note = "；".join(notes)
            else:
                # 全部规则裁定为分歧/过期 → 声明分歧，不静默二选一
                report = VerifyReport(
                    overall_status=REFUSE,
                    overall_reason="rule_conflict_divergence",
                    message="".join(notes) or "两种推荐均存在，请结合临床判断",
                    matched_rules=matched,
                )
                return PipelineResult(
                    question=question,
                    answer_text=DIVERGENCE_NOTE,
                    report=report,
                    matched_rules=matched,
                    evidence=evidences,
                    retrieved=info["retrieved"],
                    used_llm=None,
                    note="；".join(notes) or DIVERGENCE_NOTE,
                )

        report = self.verifier.verify(claims, info["registry"], matched)
        answer_text = self._assemble_answer(report)
        # 整体 REFUSE 时的对外文案，含证据不足的一般性就医指引（而非纯拒绝）
        if report.overall_status == REFUSE and not answer_text:
            answer_text = TemplateLLM().insufficient()

        return PipelineResult(
            question=question,
            answer_text=answer_text,
            claims=claims,
            report=report,
            matched_rules=matched,
            evidence=evidences,
            retrieved=info["retrieved"],
            used_llm=used_name,
            note=note,
        )

    # 别名：与 __init__.py docstring 中的示例保持一致
    run = ask
