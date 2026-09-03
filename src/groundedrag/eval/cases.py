"""eval_set 读取 + 用例主张解析（runner / demo / app 三处共用的唯一实现）。

原先 runner._parse_case_claim、demo.show_eval_case、app._inject_case_claims
各自实现了一遍"把 eval 用例里的主张（字符串带锚点 / dict 显式引用 / dict 文本带锚点）
转成 AnswerClaim"并维护 expected 结论 —— 三份逻辑易漂移（一处修了锚点剥离、
另一处忘了）。本模块收敛为 :func:`load_eval_set` + :func:`parse_case_claims`，
供评测 runner 与两个示例入口复用。

语义约束（保持与原实现逐条一致，避免评测指标漂移）：
- dict 带 ``evidence_refs``/``rule_refs`` → 显式引用（数字锚点按池解析，
  文本中的 ``[证据N]`` 锚点一并剥除）；
- dict 仅带 ``text`` / 字符串 → 退化到 claims.parse_claims（JSON 或锚点行）；
- expected 只对 dict 项按 ``id(claim)`` 挂载；字符串项不参与统计。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from groundedrag.guardrail.claims import AnswerClaim, parse_claims
from groundedrag.guardrail.verifier import ANNOTATE, PASS, REFUSE

# expected 合法取值
_EXPECTED = (PASS, REFUSE, ANNOTATE)

_ANCHOR_STRIP_RE = re.compile(r"\[(?:证据|规则|evidence|rule)\s*[:：]?\s*[0-9A-Za-z_-]+\]")


def load_eval_set(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """读取 eval_set.jsonl → 用例 dict 列表（跳过空行 / 注释行）。"""
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))
    return cases


def _resolve_refs(refs: Sequence[Any], pool: Sequence[str]) -> List[str]:
    """数字锚点 → 池中实际 id（与 claims 模块约定一致，1 起）。"""
    out: List[str] = []
    for r in refs:
        s = str(r)
        if s.isdigit():
            idx = int(s) - 1
            out.append(pool[idx] if 0 <= idx < len(pool) else s)
        else:
            out.append(s)
    return out


def parse_case_claim(
    item: Any, *, evidence_ids: Sequence[str], rule_ids: Sequence[str]
) -> List[AnswerClaim]:
    """解析 eval 用例里的一条主张（支持 字符串带锚点 / dict 显式引用 / dict 文本带锚点）。"""
    if isinstance(item, dict):
        raw_text = str(item.get("text", "")).strip()
        has_explicit_refs = bool(
            item.get("evidence_refs")
            or item.get("evidence")
            or item.get("rule_refs")
            or item.get("rules")
        )
        if has_explicit_refs:
            if not raw_text:
                return []
            text = _ANCHOR_STRIP_RE.sub("", raw_text).strip(" \t-•*")
            claim = AnswerClaim(
                text=text or raw_text,
                evidence_refs=_resolve_refs(
                    item.get("evidence_refs") or item.get("evidence") or [], evidence_ids
                ),
                rule_refs=_resolve_refs(
                    item.get("rule_refs") or item.get("rules") or [], rule_ids
                ),
                critical=bool(item.get("critical", False)),
            )
            return [claim]
        item_text = raw_text  # 显式引用缺失 → 退化到文本锚点解析
    else:
        item_text = str(item)
    if not item_text.strip():
        return []
    return parse_claims(item_text, evidence_ids=evidence_ids, rule_ids=rule_ids)


def parse_case_claims(
    case: Mapping[str, Any],
    *,
    evidence_ids: Sequence[str],
    rule_ids: Sequence[str],
) -> Tuple[List[AnswerClaim], Dict[int, str]]:
    """把 ``case['claims']`` 展开为 (主张列表, {id(claim): expected})。

    expected 只挂 dict 项（无 / 非法值时取 ``PASS``，与原 runner 一致）；
    字符串项解析出的主张不参与统计（不在 expected 表里）。
    """
    claims: List[AnswerClaim] = []
    expected_of: Dict[int, str] = {}
    for item in case.get("claims") or []:
        parsed = parse_case_claim(item, evidence_ids=evidence_ids, rule_ids=rule_ids)
        for c in parsed:
            claims.append(c)
            if isinstance(item, dict):
                exp = str(item.get("expected", PASS))
                expected_of[id(c)] = exp if exp in _EXPECTED else PASS
    return claims, expected_of
