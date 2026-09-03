"""tools/leak_scan.py 合规自检工具的单元测试。

工具本身是通用扫描器（不内置私有词），测试同样只用通用内容验证判定逻辑。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEAK_SCAN_PATH = REPO_ROOT / "tools" / "leak_scan.py"


def _load_tool():
    # 从 tools/ 下的单文件脚本加载（非包）：需先注册进 sys.modules，
    # 否则模块内 @dataclass / typing 解析按 __module__ 查 sys.modules 会失败。
    spec = importlib.util.spec_from_file_location("leak_scan", LEAK_SCAN_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


leak = _load_tool()


class TestSecretScan:
    # 蜜罐内容在运行时拼接 —— 仓库自扫描（含 tests/ 本文件）时这些字面量
    # 不落在源码里，避免"自检脚本被自己的测试样本误伤"。
    SECRET_HEAD = "-----BEGIN " + "PRIVATE KEY-----"
    DUMMY_TOKEN = "sk-" + "live-abcdef123456"  # 变量名避开 secret/api_key 词面

    def test_literal_secret_flagged(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text(f"api_key = '{self.DUMMY_TOKEN}'\n", encoding="utf-8")
        findings = []
        leak.scan_secret(f, "x.py", findings)
        assert findings and findings[0].severity == "high"

    def test_placeholder_not_flagged(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("api_key = 'your_api_key_here'\n", encoding="utf-8")
        findings = []
        leak.scan_secret(f, "x.py", findings)
        assert findings == []

    def test_os_getenv_not_flagged(self, tmp_path):
        # 引号外取值（读环境变量）不构成字面量密钥，避免误报
        f = tmp_path / "x.py"
        f.write_text("api_key = os.getenv('SECRET_KEY')\n", encoding="utf-8")
        findings = []
        leak.scan_secret(f, "x.py", findings)
        assert findings == []

    def test_private_key_header_flagged(self, tmp_path):
        f = tmp_path / "key.pem"
        f.write_text(self.SECRET_HEAD + "\nMIIB\n", encoding="utf-8")
        findings = []
        leak.scan_secret(f, "key.pem", findings)
        assert findings and findings[0].severity == "high"


class TestMarkers:
    def test_load_markers_file(self, tmp_path):
        mfile = tmp_path / "m.json"
        mfile.write_text(json.dumps({"terms": ["私有项目X"], "patterns": ["REG\\d+"]}))
        m = leak.Markers.load(mfile)
        assert m.active
        assert m.terms == ["私有项目X"]
        assert m.compiled

    def test_missing_markers_inactive(self):
        m = leak.Markers.load(None)
        assert not m.active

    def test_invalid_regex_ignored(self, tmp_path):
        mfile = tmp_path / "m.json"
        mfile.write_text(json.dumps({"patterns": ["[invalid"]}))
        m = leak.Markers.load(mfile)
        assert m.compiled == []

    def test_term_hit_and_miss(self, tmp_path):
        hit = tmp_path / "hit.md"
        hit.write_text("这里提到了 私有项目X。\n", encoding="utf-8")
        miss = tmp_path / "miss.md"
        miss.write_text("这里是干净文本。\n", encoding="utf-8")
        markers = leak.Markers.load(None)
        markers.terms = ["私有项目X"]
        f1, f2 = [], []
        leak.scan_markers(hit, "hit.md", markers, f1)
        leak.scan_markers(miss, "miss.md", markers, f2)
        assert any(x.severity == "high" for x in f1)
        assert f2 == []


class TestRefsSimilarity:
    def test_dice_of_identical_text(self):
        g = leak._bigrams("今天天气很好")
        assert leak._dice(g, g) == 1.0

    def test_disjoint_texts(self):
        a = leak._bigrams("今天天气很好")
        b = leak._bigrams("完全不同的内容")
        assert leak._dice(a, b) < 0.5

    def test_scan_refs_flags_copied_paragraph(self, tmp_path):
        ref = tmp_path / "ref.txt"
        ref.write_text("这是一段需要核对的私有参考文档正文内容示例文本。", encoding="utf-8")
        cand = tmp_path / "cand.md"
        cand.write_text("这是一段需要核对的私有参考文档正文内容示例文本。", encoding="utf-8")
        ref_grams = [leak._bigrams(ref.read_text(encoding="utf-8"))]
        findings = []
        leak.scan_refs(cand, "cand.md", ref_grams, 0.6, findings)
        assert findings and findings[0].severity == "warning"


class TestIterTextFiles:
    def test_skips_venv_and_cache(self, tmp_path):
        (tmp_path / "keep.md").write_text("a", encoding="utf-8")
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "x.py").write_text("b", encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "y.pyc").write_text("c", encoding="utf-8")
        found = {p.name for p in leak.iter_text_files(tmp_path)}
        assert found == {"keep.md"}


class TestDataFile:
    def test_large_and_data_suffix_warning(self, tmp_path):
        big = tmp_path / "big.txt"
        big.write_text("x" * 2_000_000, encoding="utf-8")
        sql = tmp_path / "dump.sql"
        sql.write_text("SELECT 1;", encoding="utf-8")
        f1, f2 = [], []
        leak.scan_datafile(big, "big.txt", f1)
        leak.scan_datafile(sql, "dump.sql", f2)
        assert f1 and f2


class TestMainExitCodes:
    def test_clean_repo_exit_zero(self):
        # 在仓库自身跑一遍默认门禁：当前应无 high / warning
        rc = leak.main(["--root", str(REPO_ROOT), "--json"])
        assert rc == 0
