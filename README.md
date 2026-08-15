# ResumeMind · 基于 RAG 的个人简历问答 Agent

<div align="center">

将简历、项目资料和个人经历整理为可检索的结构化知识，让面试官可以通过自然语言继续追问简历中的内容。

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](backend/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688?logo=fastapi&logoColor=white)](backend/main.py)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](frontend/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20DB-DC244C?logo=qdrant&logoColor=white)](docker-compose.yml)
[![Tests](https://img.shields.io/badge/tests-480%20passed-2ea44f)](backend/app/tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](#)

</div>

---

## ✨ 项目简介

ResumeMind 是一个面向个人简历场景的 RAG 问答系统。

和传统的「上传 PDF 后直接问答」不同，ResumeMind 会先把简历、项目说明、证书和个人经历等材料整理为结构化知识，再进行检索和生成。

系统重点处理几个简历问答中比较容易出问题的场景：

- 多轮追问中的指代，例如「那这个问题最后怎么解决的？」
- 不同项目之间的事实归属，避免把 A 项目的指标回答到 B 项目上
- 简历未记录的问题，例如薪资、联系方式、QPS 等，证据不足时明确拒答
- 项目数字、时间、合作单位等硬事实的 grounding 校验
- 多份个人资料上传后的统一整理和自动入库

目标不是让模型「更会说」，而是尽可能让回答与用户实际提供的材料保持一致。

## 🧩 核心能力

| 能力 | 实现 |
|---|---|
| **意图路由** | 将问题划分为个人事实、项目深挖、通用技术、HR/行为、简历问答、寒暄和越界等类型，不同类型采用不同的证据要求 |
| **Fact Ledger** | 使用结构化的「实体—属性—值」保存关键事实，生成后再次检查事实所属对象，降低跨项目事实混淆 |
| **混合检索** | Dense Retrieval 与关键词精确召回结合，通过 RRF 融合候选结果 |
| **分级重排** | 对高置信检索结果直接进入生成；存在歧义时再使用 BGE reranker，减少不必要的重排开销 |
| **多轮问答** | 保存会话上下文，并对「这个项目」「那怎么解决的」等省略表达进行补全 |
| **Grounding 校验** | 对数字、日期、专有名词等硬事实检查其是否存在于检索证据中 |
| **证据展示** | 对外仅返回来源文件、章节、摘录和事实状态，不暴露内部 Prompt 与完整检索上下文 |
| **人物工坊** | 将简历、项目资料等原始材料加工为 Markdown 知识文档、人物档案和结构化事实，并自动写入知识库 |
| **离线评测** | 提供面试问答评测集，统计 Recall@K、MRR、事实关联、拒答正确率、TTFT 和端到端延迟等指标 |

## 🧠 Skill 驱动的资料加工

项目将「如何处理个人材料」这部分规则独立为 Agent Skill：

```text
.agents/skills/resume-materials-workshop/
```

Skill 中维护资料加工流程、Prompt、输出契约、参考规则和版本信息，后端运行时读取同一套定义完成资料处理。

整体流程如下：

```text
简历 / 项目介绍 / 个人资料
            │
            ▼
resume-materials-workshop
            │
      解析、清洗、整理
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
 Markdown  Persona  Facts
     │              │
     └──────┬───────┘
            ▼
        RAG 知识库
```

原始材料不会直接作为最终知识块使用。系统会先按照主题重新组织为 Markdown，例如：

```text
education.md
skills.md
echoguide.md
resumemind.md
competition.md
```

再进行分块和向量化，使知识库中的内容更适合后续检索。

Skill 负责定义「资料应该如何加工」，RAG 知识库负责保存「这个人有哪些信息」，两部分相互独立。

### 人物 Skill 导出

人物工坊还支持将已经整理完成的资料导出为一个独立的人物 Skill 包：

```text
persona-{name}/
├── SKILL.md
├── facts.json
└── references/
    ├── profile.md
    ├── education.md
    └── projects/
```

其中 `references/` 保存个人资料，`facts.json` 保存结构化事实，`SKILL.md` 定义回答边界和资料使用方式。

这个能力主要用于将已经整理过的人物知识迁移到其他支持 Skills 的 Agent 环境中；ResumeMind 自身的在线问答仍然以 RAG 知识库为主要数据来源。

## 🏗️ 系统架构

```text
面试官提问
   │
   ▼
访问码 / 限流 / 每日预算
   │
   ├──────────────► 问答缓存
   │
   ▼
意图路由
fast path + LLM fallback
   │
   ▼
检索规划
   │
   ▼
Dense Retrieval ──┐
                  ├──► RRF 融合
关键词精确召回 ────┘
                       │
             ┌─────────┴─────────┐
             │                   │
       高置信结果              歧义结果
             │                   │
         跳过重排          BGE Reranker
             │                   │
             └─────────┬─────────┘
                       ▼
                 LLM 单次生成
                       │
              ┌────────┴────────┐
              ▼                 ▼
      Grounding 校验       Fact Ledger 校验
              │                 │
              └────────┬────────┘
                       ▼
            answered / hedged / refused
                       │
                       ▼
              citations + QA logs
```

系统为单次问答设置统一时间预算。检索、重排和生成共享这一预算，在剩余时间不足时会跳过部分非必要步骤，避免某一个环节拖慢整条链路。

## 🎯 评测

项目提供一套 30 题的 AI 应用后端面试评测集：

```bash
python scripts/eval_interview_set.py \
  --data scripts/eval_cases_ai_interview.jsonl \
  --stream
```

当前测试结果：

| 指标 | 结果 |
|---|---|
| 端到端延迟 | p50 **4.08s** · p95 **7.26s**（改造前 p50 13.1s） |
| LLM 调用次数 | fast path p50 **1 次** |
| 期望事实命中 | **91.6%**（76 / 83） |
| Recall@5 | **1.00** |
| 评测集通过率 | **30 / 30** |

除检索指标外，评测脚本还记录：

- MRR
- 上下文精确率
- 事实关联正确率
- 拒答正确率
- 幻觉禁止词
- TTFT
- Rerank 耗时
- 端到端耗时

这些数据主要用于比较不同检索策略和参数调整前后的效果，而不是只依赖人工体验判断系统是否变好。

## 🚀 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
```

至少配置：

```text
LLM_API_KEY
ADMIN_PASSWORD
```

如需启用访客访问码，再配置：

```text
QA_ACCESS_CODE
```

### 2. 构建并启动

```bash
docker compose up -d --build
```

项目采用 CPU 部署，默认配置面向 2C4G 云服务器。

网络环境不稳定时，可以提前下载 PyTorch CPU wheel：

```bash
python scripts/download_torch_wheel.py
```

### 3. 导入知识库

```bash
python scripts/upload_knowledge_base.py --sync
```

该脚本会处理 `docs/` 下的简历、证书和项目介绍等资料，并完成解析、分块和索引。

### 4. 访问系统

```text
http://127.0.0.1:8000
```

左下角登录入口可进入管理后台。

服务端口可通过 `.env` 中的 `RESUME_APP_PORT` 修改。

## 🔍 部署前检查

```bash
python scripts/preflight_deploy.py
python scripts/seed_fact_ledger.py
```

`preflight_deploy.py` 用于检查配置、知识库边界和运行依赖。

`seed_fact_ledger.py` 用于初始化事实台账，支持重复执行。

## 🧪 开发与测试

后端测试：

```bash
python -m pytest -q
```

当前包含 480 项测试，模型调用均使用 Mock。

前端检查与构建：

```bash
cd frontend
npm ci
npm run build
```

## 📦 技术栈

- **后端**：Python 3.13 / FastAPI / SQLAlchemy / SQLite
- **检索（RAG）**：Qdrant · BAAI/bge-small-zh-v1.5（embedding）· BAAI/bge-reranker-base（rerank）· Dense Retrieval + 关键词精确召回 · RRF 融合
- **LLM**：OpenAI 兼容 Chat Completions API（默认 DeepSeek）
- **前端**：React 19 / TypeScript / Vite
- **部署**：Docker / Docker Compose（纯 CPU 推理）

## 🗂️ 目录结构

```text
Resume-Agent/
├── backend/app/
│   ├── api/               # QA、文档、认证、审计、健康检查
│   ├── services/          # 路由、检索、重排、生成、Fact Ledger、缓存
│   ├── core/              # 配置、安全、数据库
│   └── models/            # SQLAlchemy Models
├── frontend/              # React + TypeScript + Vite
├── scripts/               # 知识库、评测、部署相关脚本
├── .agents/skills/        # resume-materials-workshop：个人资料加工 Skill
├── docs/                  # 简历、证书、项目介绍等知识库材料
├── docs-guide/            # 部署与评测文档
└── docker-compose.yml
```

## 🛣️ Roadmap

- [x] RAG 简历问答
- [x] Grounding 与 Fact Ledger
- [x] 七类意图路由
- [x] Dense + Keyword 混合检索
- [x] 分级 Rerank 与统一时间预算
- [x] 多轮问答与指代消解
- [x] 面试问答评测集
- [x] 人物工坊：个人资料 → Markdown / Persona / Facts
- [x] Skill 驱动的资料加工
- [ ] 提升复合列举问题的 fast path 覆盖率
- [ ] 优化列举类问题的答案完整性
- [ ] 继续降低 p95 延迟

## ⚠️ 安全

系统目前包含以下基础防护：

- 管理接口 JWT 鉴权
- 未配置 `ADMIN_PASSWORD` 时拒绝启动
- 匿名用户无法获取内部 Prompt 和完整检索上下文
- 上传文件大小限制与文件类型检查
- OOXML 压缩炸弹检查
- 文件名路径穿越处理
- URL 导入 SSRF 防护
- 用户问题、知识片段和历史上下文均按不可信输入处理
- 访问码错误次数限制和 IP 配额
- 每日调用预算限制

更详细的知识库和安全检查记录位于 `docs-guide/`。

## 📄 License

MIT © 2026 [iam-epiphany](https://github.com/iam-epiphany)
