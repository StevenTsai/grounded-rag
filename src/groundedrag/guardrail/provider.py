"""RuleProvider 抽象接口 + 内置文件实现。

解耦目标：``GuidelineEngine`` 只依赖 ``RuleProvider`` 与纯数据模型，
不 import 任何 ORM。业务方可实现自己的 ``DatabaseRuleProvider``
（例如从权威指南库实时加载规则，实现可闭源）并沿用同一接口无缝替换。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Protocol, Union

from groundedrag.guardrail.models import ResistanceRule, Rule


class RuleProvider(Protocol):
    """规则数据访问抽象。"""

    def load_pathway_rules(self) -> List[Rule]: ...
    def load_resistance_rules(self) -> List[ResistanceRule]: ...


class JsonRuleProvider:
    """从 ``seed_rules.json`` 加载合成示例规则。

    构造时一次性读取并缓存 payload（load_pathway_rules / load_resistance_rules
    各自消费同一份内存数据，避免同一文件被重复读两次）。
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f:
            self._payload = json.load(f)

    def load_pathway_rules(self) -> List[Rule]:
        raw = self._payload.get("pathway_rules", self._payload.get("rules", []))
        return [Rule.from_dict(item) for item in raw]

    def load_resistance_rules(self) -> List[ResistanceRule]:
        raw = self._payload.get("resistance_rules", [])
        return [ResistanceRule.from_dict(item) for item in raw]


class DictRuleProvider:
    """从内存 dict 加载规则（测试友好）。"""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload

    def load_pathway_rules(self) -> List[Rule]:
        raw = self.payload.get("pathway_rules", self.payload.get("rules", []))
        return [Rule.from_dict(item) for item in raw]

    def load_resistance_rules(self) -> List[ResistanceRule]:
        raw = self.payload.get("resistance_rules", [])
        return [ResistanceRule.from_dict(item) for item in raw]
