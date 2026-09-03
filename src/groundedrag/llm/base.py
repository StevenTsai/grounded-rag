"""LLMService 抽象基类。

GroundedRAG 的 LLM 层职责是"受约束的生成"：不直接输出自由文本，而是输出
可解析的结构化块（AnswerClaim[] + 锚点）。策略基类提供统一调用面，子类
（OpenAI 兼容 / 模板回退）各自实现 generate。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMError(Exception):
    """LLM 调用异常基类。"""


class LLMService(ABC):
    """统一 LLM 调用抽象。"""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """生成文本。失败应抛 LLMError（由 failover 捕获切换）。"""
        raise NotImplementedError

    def is_available(self) -> bool:
        """是否可调用（如未配置 key 返回 False）。"""
        return True

    def render_context(self, payload: Dict[str, Any]) -> str:
        """把检索/规则上下文渲染成 prompt 片段（子类可定制）。"""
        return _default_context_render(payload)


def _default_context_render(payload: Dict[str, Any]) -> str:
    """默认上下文渲染：证据 + 规则 各带锚点。"""
    parts: list[str] = []
    evidences = payload.get("evidences") or []
    if evidences:
        parts.append("【可引用证据】")
        for i, ev in enumerate(evidences, 1):
            src = ev.get("source_type", "generic")
            ver = ev.get("source_version") or "版本未知"
            grade = ev.get("grade", "D")
            span = (ev.get("text_span") or "").strip().replace("\n", " ")
            parts.append(f"[证据{i}] (来源={src}, 版本={ver}, 等级={grade}) {span}")
    rules = payload.get("rules") or []
    if rules:
        parts.append("【命中的权威规则】")
        for i, r in enumerate(rules, 1):
            rec = "；".join(f"{x.get('plan_name','')}（{x.get('grade','')}级）" for x in (r.get("recommendations") or []))
            parts.append(f"[规则{i}] (id={r.get('rule_id')}, 更新={r.get('updated_at') or '未知'}) {rec}")
    return "\n".join(parts)
