# Changelog

本文件记录 GroundedRAG 的主要变更。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [1.0.0] - 2026-09-18

### Added

- 核心校验门：四步确定性校验（引用完整性 → 表面要素一致性 → 规则冲突裁定 → 证据充分性）
- 三态判定：PASS / ANNOTATE / REFUSE
- 规则直接路径：规则命中 → claims 从规则合成 → 纯逻辑验证，零 LLM 依赖
- 反跨证据拼装：数字/实体必须来自同源证据
- BM25 检索 + 实体增强 + 查询扩展（jieba 分词；缺失时 CJK char-bigram 回退）
- 多 LLM failover：OpenAI 兼容（零 SDK）+ 模板回退
- 内置评测集：verify 模式（34 组）、e2e 模式（13 组）、锚点绑定模式（binding，离线）
- E2E 评测集（13 组用例，覆盖正例/负例/规则冲突/混合场景）
- 确定性锚点绑定器：`surface_tokens` + 子串匹配（证据）/ 字段匹配（规则）
- 枚举前缀剥离（`_ENUM_RE`）：防止 "1. xxx" 中的 "1" 被当作数值导致 number_mismatch
- E2E `anchor_binding_rate` 指标（度量 LLM 是否按约束吐锚点）
- 领域适配：`source_type` 扩充 `regulation` / `literature` / `manual`；非医疗条件字段通用兜底提取
- 领域适配文档可执行测试（防文档与实现漂移）
- CLI 工具：`groundedrag ask` / `groundedrag init`（评测入口为 `python -m groundedrag.eval`）
- 外部集成指南 `docs/integration.md` + 可运行示例 `examples/integrate_verifier.py`（整条流水线 / 只挂校验门 / 单组件三种接入）
- 引用规范 `CITATION.cff`
- 247 单元测试，整体覆盖率 95%

### Fixed

- 全新克隆无法运行：`examples/seed_rules.json` 强制入库（此前被本地全局 gitignore 误伤）
- 模板拒答检测：LLM 超时/失败时跳过 `parse_claims`，让 rule-direct fallback 接管（之前伪主张会阻止规则直出）
- 领域适配指南：金融/法律示例因非法 `source_type`（`regulation` 未登记）与错误的扩展点 API 名无法运行
- CI：test job 补装 ruff，修复矩阵测试因缺少 ruff 全部失败

### Changed

- LLM provider 预设与配置解析收敛为 `llm/providers.py` 单一事实来源（CLI / eval / tools 共用）
- 数据声明如实化：合成示例与公开指南整理的评测数据分列；真实评测数据不入库、不随仓库分发
- 开发状态 classifier `Alpha` → `Beta`；README 补充 GitHub 镜像 URL 与集成/引用章节
- CI：GitHub Actions，Python 3.10/3.11/3.12 矩阵测试（全绿）
