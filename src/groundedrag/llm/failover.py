"""主备切换降级：按序尝试多个 LLMService，全部失败回退模板。"""

from __future__ import annotations

from typing import List, Optional, Sequence

from groundedrag.llm.base import LLMError, LLMService
from groundedrag.llm.template import TemplateLLM


class FailoverLLM(LLMService):
    """多模型主备降级封装。

    参数：
        services: 有序候选（第一个为主模型，后续为备用）
        append_template: 是否自动在尾部追加模板回退（默认 True，保证永不抛错）
    """

    name: str = "failover"

    def __init__(
        self,
        services: Optional[Sequence[LLMService]] = None,
        *,
        append_template: bool = True,
    ) -> None:
        self.services: List[LLMService] = list(services or [])
        self.append_template = append_template
        if append_template and not any(isinstance(s, TemplateLLM) for s in self.services):
            self.services.append(TemplateLLM())

    def add(self, service: LLMService) -> "FailoverLLM":
        self.services.append(service)
        return self

    @property
    def last_error(self) -> Optional[LLMError]:
        return getattr(self, "_last_error", None)

    def is_available(self) -> bool:
        return any(s.is_available() for s in self.services)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        last_exc: Optional[Exception] = None
        for svc in self.services:
            if not svc.is_available():
                continue
            try:
                return svc.generate(
                    prompt, temperature=temperature, max_tokens=max_tokens
                )
            except LLMError as exc:
                last_exc = exc
                continue
            except Exception as exc:  # noqa: BLE001 —— failover 吞掉所有异常继续降级
                last_exc = exc
                continue
        self._last_error = (
            last_exc if isinstance(last_exc, LLMError) else LLMError(str(last_exc) if last_exc else "无可用模型")
        )
        # append_template=True 时必有 TemplateLLM，这里仅为防御
        raise self._last_error
