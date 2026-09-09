"""CLI（groundedrag ask / init）单元测试：参数解析、.env 加载、LLM 构造、命令执行。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from groundedrag import cli

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = REPO_ROOT / "examples" / "seed_docs.jsonl"
RULES_PATH = REPO_ROOT / "examples" / "seed_rules.json"


# 清除会触发真实 LLM 调用的环境变量（保证测试确定性、无网络）
_LLM_ENV_KEYS = (
    "LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
    "XIAOMI_API_KEY", "DEEPSEEK_API_KEY", "DOUBAO_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    for k in _LLM_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


class TestLoadDotenv:
    def test_missing_file_is_noop(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        cli._load_dotenv()  # 无 .env → 不报错
        assert "GROUNDEDRAG_FAKE" not in os.environ

    def test_parses_key_value(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(
            "AUTHOR=test\nTHING=value\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        cli._load_dotenv()
        assert os.environ["AUTHOR"] == "test"
        assert os.environ["THING"] == "value"

    def test_strips_inline_comment_and_quotes(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(
            'LLM_PROVIDER=xiaomi # 主模型\nLLM_API_KEY="sk-abc"  # 带引号\n',
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        cli._load_dotenv()
        assert os.environ["LLM_PROVIDER"] == "xiaomi"
        assert os.environ["LLM_API_KEY"] == "sk-abc"

    def test_does_not_override_existing_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        (tmp_path / ".env").write_text("LLM_PROVIDER=xiaomi\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        cli._load_dotenv()
        assert os.environ["LLM_PROVIDER"] == "deepseek"

    def test_skips_comments_and_blank(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(
            "# 注释行\n\nNOEQUALS_LINE\nA=1\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        cli._load_dotenv()
        assert os.environ.get("A") == "1"
        assert "NOEQUALS_LINE" not in os.environ


class TestBuildLLM:
    def test_returns_none_without_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        assert cli._build_llm() is None

    def test_none_with_unknown_provider(self, monkeypatch, capsys):
        monkeypatch.setenv("LLM_API_KEY", "sk-x")
        llm = cli._build_llm("nope")
        assert llm is None
        assert "不支持" in capsys.readouterr().err

    def test_uses_unified_env_vars(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-unified")
        monkeypatch.setenv("LLM_BASE_URL", "https://custom/v1")
        monkeypatch.setenv("LLM_MODEL", "my-model")
        llm = cli._build_llm("deepseek")
        assert llm is not None
        assert llm.services[0].model == "my-model"

    def test_falls_back_to_provider_key_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        llm = cli._build_llm()
        assert llm is not None
        assert "deepseek" in llm.services[0].base_url


class TestMainDispatch:
    def test_no_command_prints_help_returns_1(self, capsys):
        rc = cli.main([])
        assert rc == 1
        assert "usage" in capsys.readouterr().out.lower()


class TestCmdInit:
    def test_generates_docs_and_rules(self, tmp_path):
        rc = cli._cmd_init(type("A", (), {"dir": str(tmp_path)})())
        assert rc == 0
        docs = tmp_path / "my_seed_docs.jsonl"
        rules = tmp_path / "my_seed_rules.json"
        assert docs.exists()
        assert rules.exists()
        # docs 每行是合法 JSON
        for line in docs.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            assert obj["doc_id"]
        # rules 可解析且含 pathway_rules
        payload = json.loads(rules.read_text(encoding="utf-8"))
        assert payload["pathway_rules"]

    def test_does_not_overwrite_existing(self, tmp_path):
        d = tmp_path / "sub"
        d.mkdir()
        rc = cli._cmd_init(type("A", (), {"dir": str(d)})())
        assert rc == 0
        docs = d / "my_seed_docs.jsonl"
        docs.write_text("original\n", encoding="utf-8")
        rc = cli._cmd_init(type("A", (), {"dir": str(d)})())
        assert rc == 0
        assert docs.read_text(encoding="utf-8") == "original\n"

    def test_main_init_subcommand(self, tmp_path):
        rc = cli.main(["init", "--dir", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "my_seed_rules.json").exists()


class TestCmdAsk:
    """chdir 到空目录再跑 —— 保证 `_load_dotenv()` 读不到仓库根真实 .env，
    无 API key → llm=None → 规则直出，测试不触网、确定性。"""

    @pytest.fixture(autouse=True)
    def _no_repo_dotenv(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)

    def test_rule_direct_answer_no_llm(self, capsys):
        rc = cli.main([
            "ask", "EGFR突变肺癌一线推荐什么方案？",
            "--docs", str(DOCS_PATH), "--rules", str(RULES_PATH),
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "奥希替尼" in out

    def test_refusal_when_no_rule(self, capsys):
        rc = cli.main([
            "ask", "埃博拉病毒治疗的首选方案是什么？",
            "--docs", str(DOCS_PATH), "--rules", str(RULES_PATH),
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "不足以支撑" in out or "无法确认" in out or "就医" in out

    def test_json_output_shape(self, capsys):
        rc = cli.main([
            "ask", "EGFR突变肺癌一线推荐什么方案？", "--json",
            "--docs", str(DOCS_PATH), "--rules", str(RULES_PATH),
        ])
        out = capsys.readouterr().out
        assert rc == 0
        payload = json.loads(out)
        assert payload["question"]
        assert payload["status"] in ("pass", "refuse", "annotate")
