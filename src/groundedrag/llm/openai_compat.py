"""OpenAI 兼容客户端（原生 HTTP，零 SDK 依赖）。

通过标准库 urllib 调用 ``{base_url}/chat/completions``，兼容小米 MiMo /
DeepSeek / 豆包等任何 OpenAI 兼容端点。不引入 httpx/openai SDK 以收窄
核心依赖面（见 THIRD_PARTY_NOTICES.md / 设计文档 §5.2）。
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from groundedrag.llm.base import LLMError, LLMService


class OpenAICompatibleLLM(LLMService):
    """OpenAI Chat Completions 兼容调用。

    参数：
        base_url: 端点根，如 ``https://api.deepseek.com/v1``
        api_key: API Key（缺省读 ``OPENAI_API_KEY`` / 对应环境变量）
        model: 模型名
        api_key_env: 未显式传 key 时读取的环境变量名
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        api_key_env: str = "OPENAI_API_KEY",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv(api_key_env, "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_headers = headers or {}

    @property
    def name(self) -> str:
        """对外名称为实际配置的模型名（供 used_llm / 界面标注，替代笼统的 openai-compatible）。"""
        return self.model

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _chat_url(self) -> str:
        # 兼容端点已经带 /chat/completions 与只给根地址两种情况
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.is_available():
            raise LLMError(f"{self.name}: 未配置 API Key")
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个严谨的医疗知识库问答助手。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        req = urllib.request.Request(
            self._chat_url(),
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                **self.extra_headers,
            },
            method="POST",
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, socket.timeout, json.JSONDecodeError) as exc:
            raise LLMError(f"{self.name}: 网络/解析错误 {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"{self.name}: 未知错误 {exc}") from exc

        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name}: 响应结构异常 {payload!r:.200}") from exc
