#!/usr/bin/env python3
"""开源合规自检工具：防开源仓库混入受限/私有内容。

设计原则：本工具是**通用**扫描器，自身体内不内置任何具体公司/项目名。
维护方把私有词表放到不入库的 ``.leak_markers.json``（见 .leak_markers.example.json），
工具按需读取 —— 从而"扫描脚本随仓库开源、私有词表始终闭源"。

三档检查：

1. **凭据扫描（secret，high）**：``api_key/secret/token/password`` 等字面量赋值，
   自动过滤占位符（change_me/xxx/<...>/…）；含私钥块头也判 high。
2. **私有标记（markers，high）**：外部词表命中（词条/正则/路径子串），
   默认读 ``./.leak_markers.json``（如存在）。
3. **相似度比对（refs，warning）**：``--refs <目录>`` 指向"参考文档"（草稿/内部文档），
   对仓库文本做字符 bigram 重叠粗比对 —— 用于捕获"改写仍高度相似"的段落搬运。

用法：
    python tools/leak_scan.py                          # 凭据 + 本地 markers（若有）
    python tools/leak_scan.py --markers .leak_markers.json
    python tools/leak_scan.py --refs /path/internal/docs --threshold 0.5

退出码：0=通过；1=存在 high（凭据/私有词）；2=仅 warning（相似/疑似数据文件）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# 扫描时跳过的目录/文件
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".ruff_cache", ".pytest_cache", ".mypy_cache", "build", "dist",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".egg-info"}
# 疑似"成包数据/二进制"后缀（仅 warning；入库前应人工确认是否是合法数据资产）
DATA_SUFFIXES = {
    ".sql", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".pdf",
    ".zip", ".gz", ".7z", ".pkl", ".npy", ".npz", ".parquet", ".onnx",
    ".pem", ".key", ".crt", ".jpg", ".jpeg", ".png", ".woff", ".woff2",
}
MAX_TEXT_BYTES = 2_000_000  # >2MB 不参与文本扫描（可能为二进制/数据资产）

# 凭据赋值：只认「引号包住的字面量」，避免把 os.getenv(...) / Optional[...] 误报。
_SECRET_ASSIGN = re.compile(
    r"(?P<key>api[_-]?key|secret|token|password|passwd|private[_-]?key|access[_-]?key)"
    r"\s*[:=]\s*(?:['\"])(?P<val>[A-Za-z0-9_\-\./+]{8,})(?:['\"])",
    re.IGNORECASE,
)
_SECRET_HEAD = re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")
_PLACEHOLDER_SUBSTR = ("xxx", "change_me", "changeme", "<your", "<...", "your_", "example")


@dataclass
class Finding:
    severity: str            # high | warning | info
    check: str               # secret | marker | refs | datafile
    file: str
    line: int = 0
    detail: str = ""

    def __str__(self) -> str:
        loc = f"{self.file}:{self.line}" if self.line else self.file
        return f"[{self.severity.upper():7}] {self.check:<8} {loc}  {self.detail}"


@dataclass
class Markers:
    terms: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    paths: List[str] = field(default_factory=list)
    compiled: List[re.Pattern] = field(default_factory=list)

    @classmethod
    def load(cls, path: Optional[Path]) -> "Markers":
        m = cls()
        if path is None or not path.exists():
            return m
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"⚠️  markers 文件读取失败，忽略：{path}（{exc}）")
            return m
        m.terms = [str(t) for t in (data.get("terms") or [])]
        m.patterns = [str(p) for p in (data.get("patterns") or [])]
        m.paths = [str(p) for p in (data.get("paths") or [])]
        for p in m.patterns:
            try:
                m.compiled.append(re.compile(p))
            except re.error as exc:
                print(f"⚠️  忽略非法正则 <{p}>：{exc}")
        return m

    @property
    def active(self) -> bool:
        return bool(self.terms or self.patterns or self.paths)


def iter_text_files(root: Path) -> Iterable[Path]:
    """遍历仓库文本候选文件（跳过 venv/.git/缓存与超大文件）。"""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_TEXT_BYTES:
                continue
        except OSError:
            continue
        yield path


def _is_placeholder(val: str) -> bool:
    low = val.lower()
    return any(s in low for s in _PLACEHOLDER_SUBSTR) or len(val) < 8


def scan_secret(path: Path, rel: str, findings: List[Finding]) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    if _SECRET_HEAD.search(text):
        findings.append(
            Finding("high", "secret", rel, 1, "含私钥块头（-----BEGIN … PRIVATE KEY-----）")
        )
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in _SECRET_ASSIGN.finditer(line):
            val = m.group("val")
            if _is_placeholder(val):
                continue
            key = m.group("key")
            # 进一步降低误报：值与 key 同名且很短也算占位
            if val.lower() in (key.lower(), key.lower() + "123", "changeme"):
                continue
            findings.append(
                Finding(
                    "high", "secret", rel, lineno,
                    f"疑似密钥赋值：{key} = {val[:6]}…{val[-2:]}（长度 {len(val)}）",
                )
            )


def scan_markers(path: Path, rel: str, markers: Markers, findings: List[Finding]) -> None:
    if not markers.active:
        return
    if any(sub in rel for sub in markers.paths):
        findings.append(Finding("high", "marker", rel, 0, "路径命中私有标记"))
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    for lineno, line in enumerate(text.splitlines(), 1):
        for term in markers.terms:
            if term and term in line:
                findings.append(Finding("high", "marker", rel, lineno, f"命中私有词「{term}」"))
        for pat in markers.compiled:
            if pat.search(line):
                findings.append(Finding("high", "marker", rel, lineno, f"命中私有正则 /{pat.pattern}/"))


def _bigrams(text: str) -> Dict[str, int]:
    """字符 bigram 计数（用于与参考文档的相似度粗判）。"""
    norm = re.sub(r"\s+", "", text)
    grams: Dict[str, int] = {}
    for i in range(len(norm) - 1):
        g = norm[i : i + 2]
        grams[g] = grams.get(g, 0) + 1
    return grams


def _dice(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    common = 0
    for g, n in a.items():
        common += min(n, b.get(g, 0))
    sa = sum(a.values())
    sb = sum(b.values())
    return 2 * common / (sa + sb)


def scan_refs(
    path: Path, rel: str, ref_grams: List[Dict[str, int]], threshold: float,
    findings: List[Finding],
) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    grams = _bigrams(text)
    if not grams:
        return
    best = max((_dice(grams, rg) for rg in ref_grams), default=0.0)
    if best >= threshold:
        findings.append(
            Finding(
                "warning", "refs", rel, 1,
                f"与参考文档最高 bigram 重叠 {best:.2f}（阈值 {threshold}）—— 疑似段落级相似",
            )
        )


def scan_datafile(path: Path, rel: str, findings: List[Finding]) -> None:
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size > 1_000_000 or path.suffix.lower() in DATA_SUFFIXES:
        findings.append(
            Finding(
                "warning", "datafile", rel, 0,
                f"疑似数据/二进制资产（{size / 1e6:.1f}MB）—— 入库前确认是否应随开源仓库分发",
            )
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="leak_scan",
        description="开源合规自检：凭据 / 私有标记 / 参考文档相似度",
    )
    parser.add_argument("--root", default=None, help="仓库根目录（默认 tools/ 上两级）")
    parser.add_argument(
        "--markers", default=".leak_markers.json",
        help="私有词表 JSON（不入库；不存在则跳过该项）",
    )
    parser.add_argument("--refs", default=None, help="参考文档目录/文件（相似度比对）")
    parser.add_argument("--threshold", type=float, default=0.6, help="相似度阈值 0~1")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    markers = Markers.load(Path(args.markers).expanduser())
    if not markers.active:
        print("（未发现 .leak_markers.json —— 凭据与相似度检查照常执行）")

    ref_grams: List[Dict[str, int]] = []
    if args.refs:
        ref_path = Path(args.refs)
        ref_files = (
            [ref_path]
            if ref_path.is_file()
            else sorted(p for p in ref_path.rglob("*") if p.is_file())
        )
        for rf in ref_files:
            try:
                ref_grams.append(_bigrams(rf.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                continue
        print(f"加载 {len(ref_grams)} 个参考文档用于相似度比对")

    findings: List[Finding] = []
    files_scanned = 0
    for path in iter_text_files(root):
        rel = str(path.relative_to(root))
        files_scanned += 1
        scan_secret(path, rel, findings)
        scan_markers(path, rel, markers, findings)
        scan_datafile(path, rel, findings)
        if ref_grams:
            scan_refs(path, rel, ref_grams, args.threshold, findings)

    highs = [f for f in findings if f.severity == "high"]
    warnings = [f for f in findings if f.severity == "warning"]
    if args.json:
        print(
            json.dumps(
                {
                    "root": str(root),
                    "files_scanned": files_scanned,
                    "markers_loaded": markers.active,
                    "high": [f.__dict__ for f in highs],
                    "warning": [f.__dict__ for f in warnings],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"== 泄漏扫描报告（root={root}，扫描 {files_scanned} 个文件）==")
        if not findings:
            print("  未发现问题 ✓")
        for f in findings:
            print("  " + str(f))
        print(
            f"  小结：high={len(highs)}，warning={len(warnings)}，"
            f"参考文档数={len(ref_grams)}"
        )
    if highs:
        print("❌ 发现 high 级问题 —— 修复后再提交/push", file=sys.stderr)
        return 1
    if warnings:
        print("⚠️ 存在 warning —— 建议人工复核（不影响硬性门禁）", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
