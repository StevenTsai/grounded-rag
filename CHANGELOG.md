# Changelog

本文件记录 GroundedRAG 的主要变更。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### Fixed
- 模板拒答检测：LLM 超时/失败时跳过 `parse_claims`，让 rule-direct fallback 接管（之前伪主张会阻止规则直出）
- 确定性锚点绑定器（`_bind_anchors`）：LLM 未输出锚点时，自动将裸主张绑定到检索证据/匹配规则

### Added
- E2E 评测集（13 组用例，覆盖正例/负例/规则冲突/混合场景）
- 确定性锚点绑定器：`surface_tokens` + 子串匹配（证据）/ 字段匹配（规则）
- 枚举前缀剥离（`_ENUM_RE`）：防止 "1. xxx" 中的 "1" 被当作数值导致 number_mismatch

## [1.0.0] - 2026-09

### Added
- 核心校验门：四步确定性校验（引用完整性 → 表面要素一致性 → 规则冲突裁定 → 证据充分性）
- 三态判定：PASS / ANNOTATE / REFUSE
- 规则直接路径：规则命中 → claims 从规则合成 → 纯逻辑验证，零 LLM 依赖
- 反跨证据拼装：数字/实体必须来自同源证据
- BM25 检索 + 实体增强 + 查询扩展（jieba 分词）
- 多 LLM failover：OpenAI 兼容 + 模板回退
- 内置评测集：verify 模式（34 组）+ e2e 模式（13 组）
- CLI 工具：`groundedrag ask` / `groundedrag eval` / `groundedrag demo`
- 领域适配：`groundedrag init` 一键生成领域模板
- 236 单元测试，覆盖率 83%–95%
- CI：GitHub Actions，Python 3.10/3.11/3.12 矩阵测试
