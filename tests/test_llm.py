"""llm/ 层单元测试：策略基类 / 模板回退 / 主备降级 / OpenAI 兼容 URL 构造。"""

from __future__ import annotations

import pytest

from groundedrag.llm.base import LLMError, LLMService
from groundedrag.llm.failover import FailoverLLM
from groundedrag.llm.openai_compat import OpenAICompatibleLLM
from groundedrag.llm.template import (
    INSUFFICIENT_TEMPLATE,
    REFUSAL_TEMPLATE,
    TemplateLLM,
)


class RaisingService(LLMService):
    name = "raising"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, *, temperature=0.3, max_tokens=None) -> str:
        raise LLMError("模拟模型故障")


class UnavailableService(LLMService):
    name = "offline"

    def is_available(self) -> bool:
        return False

    def generate(self, prompt, *, temperature=0.3, max_tokens=None) -> str:
        return "不应被调用"


class OkService(LLMService):
    name = "ok"

    def generate(self, prompt, *, temperature=0.3, max_tokens=None) -> str:
        return "ok-回答"


class TestLLMServiceBase:
    def test_generate_is_abstract(self):
        with pytest.raises(TypeError):
            LLMService()  # type: ignore[abstract]

    def test_default_render_context(self):
        svc = OkService()
        out = svc.render_context(
            {
                "evidences": [
                    {"source_type": "guideline", "source_version": "v1",
                     "grade": "A", "text_span": "内容甲"}
                ],
                "rules": [
                    {"rule_id": "r1", "recommendations": [
                        {"plan_name": "方案X", "grade": "A"}], "updated_at": "2026-01-01"}
                ],
            }
        )
        assert "[证据1]" in out
        assert "[规则1]" in out


class TestTemplateLLM:
    def test_always_available_and_returns_refusal(self):
        t = TemplateLLM()
        assert t.is_available()
        assert t.generate("任何提示") == REFUSAL_TEMPLATE
        assert t.refusal() == REFUSAL_TEMPLATE
        assert t.insufficient() == INSUFFICIENT_TEMPLATE


class TestFailoverLLM:
    def test_falls_back_to_appended_template(self):
        llm = FailoverLLM([RaisingService()])  # append_template 默认 True
        assert llm.generate("p") == REFUSAL_TEMPLATE

    def test_returns_secondary_service(self):
        llm = FailoverLLM([RaisingService(), OkService()], append_template=False)
        assert llm.generate("p") == "ok-回答"

    def test_skips_unavailable_and_uses_ok(self):
        llm = FailoverLLM(
            [UnavailableService(), OkService()], append_template=False
        )
        assert llm.generate("p") == "ok-回答"

    def test_raises_when_all_fail_and_no_template(self):
        llm = FailoverLLM([RaisingService()], append_template=False)
        with pytest.raises(LLMError):
            llm.generate("p")
        assert isinstance(llm.last_error, LLMError)

    def test_all_unavailable_raises(self):
        llm = FailoverLLM([UnavailableService()], append_template=False)
        with pytest.raises(LLMError):
            llm.generate("p")

    def test_is_available_any(self):
        assert FailoverLLM([UnavailableService(), OkService()]).is_available()
        assert not FailoverLLM([UnavailableService()], append_template=False).is_available()

    def test_add_chains(self):
        llm = FailoverLLM(append_template=False)
        llm.add(OkService())
        assert llm.generate("p") == "ok-回答"


class TestOpenAICompatible:
    def test_no_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("GROUNDEDRAG_TEST_KEY", raising=False)
        llm = OpenAICompatibleLLM(
            "https://example.com/v1",
            api_key_env="GROUNDEDRAG_TEST_KEY",
        )
        assert not llm.is_available()
        with pytest.raises(LLMError):
            llm.generate("p")

    def test_with_key_available(self):
        llm = OpenAICompatibleLLM("https://example.com/v1", api_key="sk-abc")
        assert llm.is_available()

    def test_chat_url_joins(self):
        llm = OpenAICompatibleLLM("https://example.com/v1", api_key="sk-abc")
        assert llm._chat_url() == "https://example.com/v1/chat/completions"

    def test_chat_url_no_double_suffix(self):
        llm = OpenAICompatibleLLM(
            "https://example.com/v1/chat/completions", api_key="sk-abc"
        )
        assert llm._chat_url() == "https://example.com/v1/chat/completions"

    def test_default_model_and_env(self, monkeypatch):
        monkeypatch.setenv("GROUNDEDRAG_TEST_KEY", "sk-xyz")
        llm = OpenAICompatibleLLM(
            "https://example.com/v1", api_key_env="GROUNDEDRAG_TEST_KEY"
        )
        assert llm.is_available()
        assert llm.model == "gpt-4o-mini"
