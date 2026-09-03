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

## 5. 数据声明

本仓库 `examples/` 数据均为**自研合成示例**（虚构药品/标志物组合、虚构数值与
研究结论），用于演示与评测机制；不含任何真实患者信息、真实临床指南原文或其他
第三方受版权数据。请勿将其内容用于真实诊疗决策。
