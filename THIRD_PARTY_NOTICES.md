# Third-Party Notices

本项目（GroundedRAG）自身采用 **MIT License**（见 [LICENSE](LICENSE)）。运行与开发
所依赖的第三方组件及其许可证清单如下。许可证全文以各组件自身发行物为准；本清单仅供
快速核对。

> 生成方式：`pip install -e ".[demo,test,dev]"` 后在干净的虚拟环境执行
> `pip-licenses > third_party_licenses.txt` 可再生成完整快照（该快照不入库，见 .gitignore）。
> 本文件随直接依赖变更人工同步；大版本升级后请重新核对。

## 1. 核心运行时依赖（安装即引入）

| 包 | 版本 | 许可证 | 用途 |
|------|------|------|------|
| [jieba](https://github.com/fxsjy/jieba) | 0.42.1 | MIT | 中文分词（BM25 / 实体增强 / 证据要素比对的前提） |

核心运行时仅此一项 —— BM25 在 jieba 缺失时提供 CJK char-bigram 回退，绝不回退到
对中文无效的 `split()`。

## 2. demo 可视化 extra（可选安装）

`pip install -e ".[demo]"` 引入 **Gradio**（Apache-2.0）及其完整依赖树。主要组件：

| 包 | 许可证 |
|------|------|
| gradio / gradio_client | Apache-2.0 |
| fastapi / starlette / uvicorn / h11 | MIT / BSD-3-Clause |
| pydantic / annotated-types / typing-inspection | MIT |
| httpx / httpcore / anyio | BSD-3-Clause / MIT |
| huggingface_hub / filelock / fsspec / safehttpx | Apache-2.0 / BSD-3-Clause / MIT |
| numpy / pandas / orjson / PyYAML / python-dateutil / pytz | BSD / MIT / MPL 等（见下） |
| pillow / pydub / markdown-it-py / rich / click / typer | MIT-CMU / MIT / MIT / MIT / BSD-3-Clause / MIT |
| brotli / hf-xet / hf-gradio / groovy / semantic-version | MIT / Apache-2.0 / MIT / MIT / BSD |
| Jinja2 / MarkupSafe / Pygments / mdurl / certifi | BSD / BSD / BSD-2-Clause / MIT / MPL-2.0 |
| six / shellingham / tqdm / wcwidth / prettytable | MIT / ISC / MPL-2.0/MIT / MIT / BSD |

> numpy 标注为 `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`；orjson 为
> `MPL-2.0 AND (Apache-2.0 OR MIT)`；完整逐包清单见生成的 `third_party_licenses.txt`。

## 3. 开发 / 测试依赖（不随发行物分发）

| 包 | 许可证 |
|------|------|
| pytest / pytest-cov / coverage | MIT / MIT / Apache-2.0 |
| ruff | MIT |
| pip-licenses | MIT |
| iniconfig / pluggy / packaging | MIT / MIT / Apache-2.0/BSD-2-Clause |

## 4. 模型 API 服务条款提示

`groundedrag.llm` 仅封装 **OpenAI 兼容 HTTP 接口**，框架本身不内置任何模型权重。
接入第三方模型 API（如 DeepSeek / 小米 MiMo / 豆包）时，请遵守各服务商的
《服务条款》与《数据处理条款》；不得将患者/用户隐私数据直接上送未获授权的外部模型。
GroundedRAG 建议高风险场景默认以**模板回退 / 规则直出**运行（无需任何模型 API），
或经本地部署的合规端点接入。

## 5. 数据声明与来源标注

随仓库分发的 `examples/` 数据为**自研合成示例与派生评测用例**（`seed_docs.jsonl` /
`seed_rules.json` / `eval_set.jsonl` / `e2e_eval_set.jsonl` / `binding_eval_set.jsonl` /
`real_eval_set.jsonl`）：虚构药品/标志物组合、虚构数值与研究结论，用于演示与机制
评测，不含任何真实患者信息或第三方作品原文。

真实场景评测数据（`examples/real_seed_docs.jsonl` / `real_seed_rules.json`）由公开
渠道发布/流通的临床指南与文献整理摘录而成（来源以中国临床肿瘤学会 CSCO 系列指南
为主，另含 NCCN / NICE / ESMO / ACR / AIRO 及公开医学平台条目，逐条见记录
`title` / `source_version` 字段）。因涉及第三方内容版权，该数据**不入库、不随仓库
或发行物分发**，仅由维护方本地留存用于离线评测，可应评审/研究者的合理要求单独提供；
相关著作权归原作者/机构所有，本仓库不对其主张任何权利。

请勿将任何数据用于真实诊疗决策。
