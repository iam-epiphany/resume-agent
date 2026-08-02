# ResumeMind · 个人简历问答 Agent

一个面向"面试官 × 简历主人公"场景的 RAG（检索增强生成）问答系统：把简历、证书、荣誉、项目介绍等文档放进知识库，面试官对简历任意提问（自我介绍、项目深挖、技术八股、HR 素质、简历细节），系统以求职者第一人称自然作答。

**核心设计是"宽松推理 + 诚实标注"**：回答可以基于知识库适度组织与推理（面试回答的完整性优先），但硬事实（数字/日期/机构/证书名）必须来自检索内容；证据不足时不是拒答，而是走**检索兜底链**（降阈重试 → 查询改写重试 → 直接生成）并强制加"根据现有知识库推测"标注——面试官一眼区分"简历实答"与"合理推测"。

本项目由个人竞赛项目 ReguMate-Agent（可信 RAG 问答系统，详见知识库《项目介绍_ReguMate.md》）改造而来，聚焦个人简历问答场景，专为 2 核 4G CPU 云服务器设计。

## 功能特性

- **置信度分级回答**：LLM 生成时同步自评证据充足度（Self-RAG 思路）——证据充分直接答、证据不足强制"根据现有知识库推测"前缀、寒暄/无关话题礼貌转移（零 LLM）
- **面试官意图路由**：规则 + LLM 两级分类 7 类面试意图，每类用不同检索策略（自我介绍锚定文档、项目深挖多查询召回、寒暄/无关跳过检索）
- **多轮追问记忆**：会话级指代消解（"那怎么解决超卖的？"自动补全上下文），面试官连续追问不丢失前文
- **检索兜底链**：标准检索 → 降阈重选 → LLM 查询改写重试 → 直接生成，证据不足永不拒答
- **轻量本地模型**：BGE-Base-ZH 中文 embedding + BGE reranker（本地推理，约 1.5GB 常驻内存）
- **LLM 走 API**：DeepSeek 等 OpenAI 兼容接口，纯 HTTP 调用，不占本地资源
- **单容器托管前端**：React 前端构建产物由 FastAPI 静态托管，无需 nginx
- **前台/后台权限分离**：访客免登录使用智能问答与问答日志；知识库管理、系统状态、完整日志需管理员密码（JWT 认证）
- **防刷限流**：问答接口按 IP 限流 + 全局并发上限，防止恶意刷 LLM token 烧钱

## 技术架构

```
面试官提问
   │
   ▼
① 意图路由（规则 + LLM 分类 7 类意图）
   │
   ▼
② 多轮追问记忆（会话指代消解）
   │
   ▼
③ 检索（query planner 拆解 → Qdrant 向量检索 → BGE reranker 重排）
   │    └─ 兜底链：降阈重选 → 查询改写重试 → 直接生成
   ▼
④ 单次 LLM 生成 + 自评 {answer, evidence_sufficiency}
   │
   ▼
⑤ 置信度分级：直接答 / "根据现有知识库推测"标注 / 礼貌转移
```

- 后端：Python 3.13 / FastAPI，`api / schemas / models / services / core` 分层
- 前端：React 19 + TypeScript + Vite
- 向量库：Qdrant（单容器）
- 模型：BAAI/bge-base-zh-v1.5（embedding）+ BAAI/bge-reranker-base（rerank）
- LLM：OpenAI 兼容 Chat Completions API（默认 DeepSeek）

## 快速开始（Docker）

要求：Docker Engine（含 Compose v2）、4GB 以上内存、约 15GB 磁盘。

```bash
# 1. 配置（确认 .env 中 DeepSeek API Key 与 ADMIN_PASSWORD 管理员密码）
# 2. 构建并启动（纯 CPU 镜像）
docker compose up -d --build

# 3. 等待模型下载并预热（首次启动，国内使用 hf-mirror 镜像加速；warmup 为后台自动执行，
#    也可登录后手动触发：curl -X POST -H "Authorization: Bearer <token>" .../api/health/warmup）
# 4. 上传知识库（扫描 docs/ 目录下的简历、证书、项目介绍等；需管理员密码）
python scripts/upload_knowledge_base.py --admin-password "$ADMIN_PASSWORD"

# 5. 打开网页提问（访客免登录）
#    页面: http://127.0.0.1:8000  —— 智能问答 + 问答日志（前台）
#    后台: 页面左下角「登录」输入管理员密码 → 知识库管理 + 系统状态 + 完整日志
#    API 文档: http://127.0.0.1:8000/docs
```

## 知识库维护

简历、证书、项目介绍等材料修改后，重跑上传脚本即可**自动更新**（脚本会比对文件哈希，内容变化的文件自动覆盖重新上传并重新索引）：

```bash
# 管理员密码默认从环境变量 ADMIN_PASSWORD 读取，也可用 --admin-password 显式传入
python scripts/upload_knowledge_base.py        # 新增 + 更新 + 重新排队一次搞定
python scripts/upload_knowledge_base.py --delete "文件名.md"   # 按文件名删除某个文档
python scripts/upload_knowledge_base.py --purge               # 清空整个知识库（重建用）
```

- 修改 `docs/` 下任一文件后直接重跑，脚本自动完成"检测变化 → 覆盖上传 → 重新索引"
- 文件内容未变时自动跳过，不会重复上传
- 删除/清空后重跑上传脚本即可重建知识库

## 知识库内容

知识库默认收录仓库 `docs/` 目录下的文字材料（每个主题一个 md 文档，全部平铺）：

| 内容 | 文档 |
|---|---|
| 简历 | `docs/简历_张三.pdf` + `docs/简历文字版.md`（口径基线） |
| 自我介绍 | `docs/自我介绍.md`（30 秒/1 分钟/3 分钟版 + FAQ） |
| 个人特质 | `docs/个人特质与兴趣爱好.md` |
| 求职动机 | `docs/求职动机与职业规划.md` |
| 竞赛奖项 | `docs/竞赛奖项.md`（蓝桥杯/算法/数学竞赛） |
| 个人荣誉 | `docs/个人荣誉.md`（奖学金/三好学生） |
| 证书 | `docs/证书说明.md`（软考/CET-6） |
| 教育背景 | `docs/教育背景.md` |
| 技能专长 | `docs/技能专长.md` |
| 求职意向 | `docs/求职意向.md` |
| 项目介绍 | `docs/项目介绍_*.md`（每个项目一个） |

**内容规则**：证书、奖状等原件是图片/扫描件时无需放入（系统无 OCR 能力），把证书内容写成 md 即可；项目不放源代码，放文字介绍。新增材料只需在 `docs/` 下建一个 md 并重跑上传脚本。支持的扩展名：pdf / doc / docx / md / txt / html / htm（xls/xlsx/csv 表格场景已移除）。

## 配置说明

环境变量见 `.env`，关键项：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` | DeepSeek | LLM API 地址与密钥（OpenAI 兼容） |
| `MODEL_DEVICE` | `cpu` | 推理设备（本项目面向 CPU 部署） |
| `RESUME_PERFORMANCE_MODE` | `cpu_low_resource` | 低资源模式（线程/批大小自动收敛） |
| `RESUME_OFFLINE_MODE` | `false` | 模型下载完成后可改 `true` 离线运行 |
| `RERANK_PROMPT_THRESHOLD` / `MIN_CORE_RERANK_SCORE` | `0.20` | 主检索门槛；低于门槛的弱证据由兜底链降阈补选，最终由 LLM 自评决定是否加推测标注 |
| `FALLBACK_*` | `true` | 检索兜底链三级开关（降阈 / 查询改写 / 直接生成） |
| `HEDGE_PREFIX` | `根据现有知识库推测` | 证据不足时回答的强制前缀 |
| `CONVERSATION_MEMORY_*` | 开 | 多轮追问记忆（TTL 24h、最多 8 轮） |
| `ADMIN_PASSWORD` | 必填 | 管理员密码（后台登录；缺失时后端拒绝启动） |
| `AUTH_REQUIRED` | `true` | 管理接口鉴权开关（生产必须 `true`，开发期可临时关闭） |
| `ADMIN_JWT_SECRET` | 空 | JWT 签名密钥；为空时由 ADMIN_PASSWORD 派生（改密码即吊销旧 token） |
| `RATE_LIMIT_ENABLED` | `true` | 问答/登录接口 IP 限流总开关 |
| `QA_IP_RATE_LIMIT_PER_MINUTE` / `QA_IP_DAILY_LIMIT` | `30` / `500` | 每 IP 每分钟 / 每日问答次数上限 |
| `QA_GLOBAL_CONCURRENCY` | `4` | 全局同时执行问答的最大并发（2C4G 建议保持 4） |
| `LOGIN_RATE_LIMIT_PER_MINUTE` | `10` | 登录接口每 IP 每分钟次数上限 |
| `RATE_LIMIT_TRUST_PROXY` | `false` | 部署在可信反向代理（Cloudflare Tunnel）后置 `true`，按 X-Forwarded-For 取 IP |

## 前台 / 后台与安全

- **前台（访客免登录）**：智能问答页 + 操作日志页（仅问答类记录）
- **后台（管理员密码登录）**：知识库管理（上传/删除/索引）、系统状态面板、完整操作日志与历史归档
- **后端强制权限**：`/api/documents/**`、`/api/health/rag` 等管理接口全部要求 JWT；前端隐藏导航只是体验，安全边界在后端
- **防刷限流**：问答接口每 IP 限流 + 全局并发上限；登录接口独立限流；429 响应带 `Retry-After` 与 `X-Request-ID`
- **上线建议**：为项目单独申请 DeepSeek API Key 并设置消费上限（保险丝）；公网走 Cloudflare Tunnel（自带 DDoS 防护）；在 Cloudflare 配置简单 WAF 规则

本地模型加载路径（默认从 HuggingFace 下载，国内走 `HF_ENDPOINT=https://hf-mirror.com`）：
- embedding: `BAAI/bge-base-zh-v1.5` → `data/models/bge-base-zh-v1.5`
- reranker: `BAAI/bge-reranker-base` → `data/models/bge-reranker-base`

## 部署到 2C4G 云服务器

完整步骤见 [docs-guide/部署指南-2C4G-CPU.md](docs-guide/部署指南-2C4G-CPU.md)，包括：Ubuntu 安装 Docker、2GB swap、模型预下载、公网访问（安全组 / Cloudflare Tunnel / HTTPS）。

## 开发验证

```bash
# 后端测试（模型全部 mock，无需本地模型；288 个用例：意图路由/会话记忆/置信度分级/兜底链全覆盖）
python -m pytest -q

# 前端构建与测试（44 个用例）
cd frontend && npm ci && npm run build && npm test -- --run
```

## 面试官问题集（回归语料）

`docs-guide/interview_question_set.md`：6 类面试问题 + 寒暄 + 4 组多轮追问链，用于架构改造前后回答质量对比与回归验证（评分维度：事实错误数、推测标注合规、转移自然度、追问链连贯性、端到端耗时）。

## 技术亮点

`docs-guide/tech-highlights.md`：置信度分级回答 + 答案自评（Self-RAG）、面试官意图路由 + 查询改写（RAGFlow 语义路由思路）、多轮追问记忆（Dify 会话记忆思路）、检索兜底链——四个亮点的实现要点与"为什么比普通 RAG 强"的论证。

## 目录结构

```text
Resume-Agent/
  backend/                 # FastAPI 后端（api/schemas/models/services/core 分层）
  frontend/                # React + TypeScript + Vite 前端
  scripts/                 # 运维脚本（知识库上传、模型检查、预检）
  docs/                    # 知识库素材（简历/证书/荣誉/项目介绍）——不参与构建
  docs-guide/              # 部署指南等文档
  data/                    # 运行数据（模型、Qdrant、上传文件），由 docker-compose 挂载
  docker-compose.yml       # 标准启动配置（纯 CPU）
  Dockerfile               # 两阶段构建（前端 + 后端）
  requirements.txt         # 生产依赖锁
  requirements-cpu.txt     # CPU 版 torch
```
