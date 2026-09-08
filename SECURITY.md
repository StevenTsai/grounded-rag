# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ |

## Reporting a Vulnerability

如果你发现了安全漏洞，请通过以下方式报告：

**不要在公开 Issue 中报告安全漏洞。**

1. 发送邮件至项目维护者（请在 README 中查找联系方式）
2. 或通过 GitHub 的 [Private vulnerability reporting](https://github.com/StevenTsai/grounded-rag/security/advisories/new) 功能提交

### 响应时间

- **确认收到**：48 小时内
- **初步评估**：7 个工作日内
- **修复发布**：根据严重程度，Critical 级别 7 天内，其他 30 天内

### 漏洞范围

以下情况属于安全漏洞：

- API Key / 凭据泄漏（通过代码或日志）
- 依赖包已知漏洞（CVE）
- 校验门绕过（可导致无证据内容被放行）
- 规则注入（可通过恶意规则篡改校验结果）

### 致谢

我们会在修复发布后，在 CHANGELOG 中致谢漏洞报告者（除非报告者要求匿名）。
