"""LLM provider 预设与配置解析（CLI / eval / tools 共用单一事实来源）。

原先 CLI 与 eval runner 各维护一份 provider 预设，易漂移。本模块把预设、
``.env`` 加载与 OpenAI 兼容服务构造收敛到一处；上层只决定"打印什么 / 用不用
failover"，不再复制 base_url / model / key_env 三元组。
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterator, List, Optional, Tuple

from groundedrag.llm.base import LLMService
from groundedrag.llm.failover import FailoverLLM
from groundedrag.llm.openai_compat import OpenAICompatibleLLM

# provider 预设：base_url / model 默认值 + 专属 API Key 环境变量名
PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "xiaomi": {
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "model": "mimo-v2.5-pro",
        "key_env": "XIAOMI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-2-0-pro",
        "key_env": "DOUBAO_API_KEY",
    },
}

DEFAULT_PROVIDER = "deepseek"


def load_dotenv(path: str = ".env") -> None:
    """轻量 ``.env`` 加载器（不覆盖已有环境变量，零依赖，支持行内注释）。"""
    p = os.path.join(os.getcwd(), path)
    if not os.path.isfile(p):
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = re.sub(r"\s+#.*$", "", value).strip()
            value = value.strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_provider(provider: Optional[str] = None) -> str:
    """解析 provider 名（CLI 参数 > ``LLM_PROVIDER`` > 默认）。"""
    return provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)


def build_service(
    provider: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[OpenAICompatibleLLM]:
    """按 provider 构造单个服务；缺 Key 或未知 provider → None。

    优先级：显式参数 > 统一覆盖（LLM_API_KEY/LLM_BASE_URL/LLM_MODEL）
    > provider 专属变量 > 预设默认值。
    """
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        return None
    key = api_key or os.getenv("LLM_API_KEY") or os.getenv(preset["key_env"], "")
    if not key:
        return None
    url = base_url or os.getenv("LLM_BASE_URL") or preset["base_url"]
    mdl = model or os.getenv("LLM_MODEL") or preset["model"]
    return OpenAICompatibleLLM(base_url=url, api_key=key, model=mdl)


def iter_services(
    provider: str,
    *,
    include_fallbacks: bool = True,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Iterator[Tuple[str, OpenAICompatibleLLM]]:
    """产出 (provider 名, 服务)：主 provider 在前，其余有 Key 的作备用。"""
    primary = build_service(
        provider, api_key=api_key, base_url=base_url, model=model
    )
    if primary is None:
        return
    yield provider, primary
    if not include_fallbacks:
        return
    for name in PROVIDER_PRESETS:
        if name == provider:
            continue
        svc = build_service(name, api_key=api_key, base_url=base_url, model=model)
        if svc is not None and svc.is_available():
            yield name, svc


def build_failover(
    provider: Optional[str] = None,
    *,
    include_fallbacks: bool = True,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[FailoverLLM]:
    """构造多 provider failover；无任何可用服务 → None。"""
    prov = resolve_provider(provider)
    if prov not in PROVIDER_PRESETS:
        return None
    services: List[LLMService] = [
        svc
        for _, svc in iter_services(
            prov,
            include_fallbacks=include_fallbacks,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    ]
    if not services:
        return None
    return FailoverLLM(services=services)
