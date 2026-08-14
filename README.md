# ResumeMind · 让面试官"对话"你的简历

<div align="center">

**一个可信优先的个人简历 RAG 问答 Agent** · 基于证据回答，宁可说"材料未记录"，也不编造一个数字

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](backend/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688?logo=fastapi&logoColor=white)](backend/main.py)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](frontend/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20DB-DC244C?logo=qdrant&logoColor=white)](docker-compose.yml)
[![Tests](https://img.shields.io/badge/tests-436%20passed-2ea44f)](backend/app/tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](#)

*Resume · Certificates · Projects → 检索增强生成 → 面试官自然语言提问，证据式回答*

</div>

---

## ✨ 为什么是 ResumeMind

把简历做成一个**可以追问到底的 Agent**。面试官问"那超卖是怎么防的？"，系统知道"那"指代哪个项目；问"REV 是你一个人做的吗"，系统答得出合作单位与分工；问到简历没写的东西（薪资、联系方式、QPS），系统**明确说材料未记录**，而不是替你编一个。

> 核心信念：面试官能接受"材料没有记录"，但不能接受一个错误的项目数字或虚构的经历。

## 🧩 功能全景

| 能力 | 说明 |
|---|---|
| 🔍 **七类意图路由** | 个人硬事实 / 项目深挖 / 通用技术 / HR·行为 / 简历问答 / 寒暄 / 越界——每类一条独立证据策略：硬事实证据不足**确定性拒答**；项目数字必须来自对应项目片段；通用技术区分"通用知识"与"本人实际做法"；HR 问题优先 persona 材料 |
| ⚖️ **事实台账（Fact Ledger）** | 结构化「实体—属性—值」台账（`fact_ledger` 表，65 条种子），生成后校验事实归属——**"A 项目的指标安到 B 项目"会被捕获并强制降级**，比 presence-only 校验更进一步 |
| 🚀 **分级检索链路** | 多角度问题合并为**一次**重排；融合分分差 / 词面锚定命中即**跳过 CPU 重排**；单问**硬时间预算**贯穿检索→重排→生成（预算不足自动摘录兜底）。实测 p50 端到端 **4.1s** |
| 💬 **多轮追问记忆** | 会话级记忆 + 指代消解（"那怎么解决的？"→ 自动补全前文对象），8 轮 × 24h TTL |
| 🔐 **访客访问码闸** | 简历附 6 位访问码，输入一次签发 24h httpOnly cookie；输错阶梯锁定（3 次→1min→5→10→30→60min 封顶）+ 每 IP 累计配额 + 全局每日预算保险丝 |
| 📎 **安全出处** | 匿名访客看到 `citations`（来源文件名 + 章节 + 80 字摘录 + 事实状态），内部 Prompt 与检索全文一律不外泄 |
| 📊 **端到端评测体系** | 30 道 AI 应用后端面试题集 + Recall@5 / MRR / 上下文精确率 / 事实关联 / 拒答正确率 / TTFT / 重排耗时 等 10+ 指标，`python scripts/eval_interview_set.py --data scripts/eval_cases_ai_interview.jsonl --stream` |

## 🏗️ 架构

```
面试官提问
   │
   ▼
访问码闸 / 限流 / 每日预算 ──► 问答缓存（秒回）
   │
   ▼
意图路由（7 类，fast path 确定性锚点 + LLM 兜底）
   │
   ▼
检索规划（枚举问句按文档清单确定性拆对象）
   │
   ▼
融合召回：dense RRF + 关键词精确召回（对象文档必入池）
   │
   ├── 跳重排分级：分差够大 / 词面锚定 / 预算不足 → 按融合分直接出
   └── 歧义问题：BGE reranker 交叉编码重排（候选 ≤6）
   │
   ▼
单次 LLM 生成 + 自评 {answer, evidence_sufficiency}
   │
   ├── grounding 硬事实校验（数字/日期/专名必须落在证据）
   └── 事实台账校验（实体—属性—值归属）
   │
   ▼
置信度分级：answered / hedged（推测标注）/ 拒答（材料未记录）
   │
   ▼
公开 citations（文件名+章节+摘录）｜qa_logs 审计落库
```

## 🎯 实测（30 道 AI 面试题 · `eval_interview_set.py`）

| 指标 | 数值 |
|---|---|
| 端到端延迟 | **p50 4.08s** · p95 7.26s（改造前 p50 13.1s） |
| LLM 调用次数 | p50 **1** 次（fast path） |
| 期望事实命中 | **91.6%**（76/83） |
| Recall@5 | **1.00** |
| 事实关联正确率 | 2/2 · 拒答正确率 1/1 · 幻觉禁止词 0 |
| 30 题通过率 | **100%**（0 失败） |

## 🚀 快速开始

```bash
# 1. 配置 .env（LLM_API_KEY、ADMIN_PASSWORD、可选 QA_ACCESS_CODE）
cp .env.example .env

# 2. 构建并启动（纯 CPU 镜像，2C4G 即可）
docker compose up -d --build

# 3. 上传知识库（docs/ 下的简历/证书/项目介绍 → 解析分块 → 索引）
python scripts/upload_knowledge_base.py --sync

# 4. 打开 http://127.0.0.1:8000 提问；左下角登录进入管理后台
```

上线前先跑预检：

```bash
python scripts/preflight_deploy.py          # 知识库边界 / 配置格式 / 依赖检查
python scripts/seed_fact_ledger.py          # 导入事实台账（幂等）
```

## 🧪 开发验证

```bash
python -m pytest -q                        # 后端 436 项测试（模型全 mock）
cd frontend && npm ci && npm run build     # 前端类型检查与构建
```

## 📦 技术栈

- **后端**：Python 3.13 / FastAPI · SQLAlchemy · SQLite（任务持久化 + 审计）
- **检索**：Qdrant（512 维 dense）+ 应用层关键词精确召回 · RRF 融合 · BGE reranker 本地交叉编码
- **模型**：BAAI/bge-small-zh-v1.5（embedding，本地 ONNX/PyTorch）+ BAAI/bge-reranker-base（rerank）
- **LLM**：OpenAI 兼容 Chat Completions API（默认 DeepSeek）
- **前端**：React 19 + TypeScript + Vite，构建产物由 FastAPI 单容器托管
- **部署**：docker-compose（app + qdrant），2C4G 档默认双 worker + blocking 预热

## 🗂️ 目录结构

```text
Resume-Agent/
  backend/app/
    api/               # 路由（qa / documents / auth / audit / health）
    services/          # 意图路由 / 规划 / 召回 / 重排 / 生成 / 台账 / 预算 / 缓存
    core/              # 配置 / 安全（JWT·访问码）/ 数据库（含手写迁移）
    models/            # SQLAlchemy 模型（含 fact_ledger）
  frontend/            # React + TS + Vite
  scripts/             # 知识库上传 / 台账导入 / 部署预检 / 面试题评测
  docs/                # 知识库素材（简历/证书/荣誉/项目介绍，一主题一文件）
  docs-guide/          # 部署指南 / 评测报告 / 修订清单
  docker-compose.yml   # app + qdrant（纯 CPU）
```

## 🛣️ Roadmap

- [x] 可信问答底座：证据式回答 + grounding + 事实台账
- [x] 七类意图路由与分级检索（跳重排 / 硬预算）
- [x] 30 题面试评测集与 10+ 指标体系
- [ ] **任意简历适配**：任何人的资料上传后，经 LLM 加工（提取结构化事实 + 生成材料主题 + 改写为检索友好的 Markdown）自动建库，无需手写知识库文档
- [ ] 规划 fast path 覆盖复合列举问题，p95 收敛至 6s 内
- [ ] 枚举问题的答案完整性（列举类问题把对象文档全部匹配块纳入 prompt）

## ⚠️ 安全

- 管理接口 JWT 鉴权（缺失 ADMIN_PASSWORD 拒绝启动，fail-closed）
- 匿名视角剥离内部 Prompt 与检索全文，仅暴露安全 citations
- 上传防护：50MB 上限 / 魔数校验 / OOXML 压缩炸弹限制 / 文件名路径穿越消毒 / URL 导入 SSRF 拦截
- 提示词注入防御声明（用户问题/知识片段/对话上下文一律视为数据）
- 更多细节见 `docs-guide/知识库修订清单.md` 与安全审计记录

## 📄 License

MIT © 2026 [iam-epiphany](https://github.com/iam-epiphany)
