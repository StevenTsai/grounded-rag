"""多模型层：策略基类 + OpenAI 兼容客户端 + 模板回退 + 主备降级。"""

from groundedrag.llm.base import LLMError, LLMService
from groundedrag.llm.failover import FailoverLLM
from groundedrag.llm.openai_compat import OpenAICompatibleLLM
from groundedrag.llm.template import TemplateLLM

__all__ = [
    "FailoverLLM",
    "LLMError",
    "LLMService",
    "OpenAICompatibleLLM",
    "TemplateLLM",
]
