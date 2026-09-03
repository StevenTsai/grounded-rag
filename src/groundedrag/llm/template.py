"""模板回退实现（含「证据不足·就医建议」专用模板）。

当主模型 / 备用模型全部失败或未配置时，回退到本模板：**结构化拒答**，
绝不编造无证据内容。校验门逻辑本身零 LLM 依赖，可独立工作。
"""

from __future__ import annotations

from typing import Optional

from groundedrag.llm.base import LLMService

# 证据不足专用拒答模板（LLM 降级链的最终兜底：结构化拒答而非编造）
REFUSAL_TEMPLATE = (
    "抱歉，针对该问题目前缺乏充分且可核查的证据。"
    "建议携带病历前往正规医院线下就诊，由医生结合完整病史判断。"
    "（GroundedRAG 已拒绝输出无证据支持的内容）"
)

# 证据不足但可给一般性就医指引的变体（供非关键问题使用）
INSUFFICIENT_TEMPLATE = (
    "现有检索证据不足以支撑精确回答。以下为一般性提示："
    "请以主治医生当面评估为准，不建议仅凭网络信息自行决策。"
)

# 非关键主张降级展示时的标注
UNVERIFIED_NOTE = "（该条基于模型知识，未经本地证据核验）"

# 分歧声明（规则冲突无法裁定时）
DIVERGENCE_NOTE = "两种推荐均存在，请结合临床判断。"


class TemplateLLM(LLMService):
    """兜底模板服务：永远可调用，返回结构化拒答文案。"""

    name: str = "template"

    def __init__(
        self,
        refusal_text: str = REFUSAL_TEMPLATE,
        insufficient_text: str = INSUFFICIENT_TEMPLATE,
    ) -> None:
        self.refusal_text = refusal_text
        self.insufficient_text = insufficient_text

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # 模板回退不产出任何"主张"，返回空 claims 信号由 pipeline 判定拒答。
        # 返回固定文案，但 pipeline 会检测到无结构化 claims → 走拒答分支。
        return self.refusal_text

    def refusal(self) -> str:
        return self.refusal_text

    def insufficient(self) -> str:
        return self.insufficient_text
