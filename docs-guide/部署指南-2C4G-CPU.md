# ResumeMind 部署指南（2 核 4G CPU 云服务器，Ubuntu）

本指南面向：Ubuntu 22.04/24.04 云服务器（2C4G、40GB 磁盘），把 ResumeMind 简历问答 Agent 部署为公网可访问的服务。

## 0. 前置准备

- 云服务器一台（2C4G，磁盘 40GB+），Ubuntu 系统
- 一个域名（可选但推荐，用于 HTTPS 与简历展示）
- DeepSeek API Key（或任意 OpenAI 兼容 LLM API）

## 1. 安装 Docker

```bash
# 官方一键脚本（安装 Docker Engine + Compose 插件，并开机自启）
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker

# 验证
docker version
docker compose version
```

## 2. 系统调优（关键：防 OOM）

2C4G 上应用常驻约 2.2GB（embedding 已换 bge-small-zh-v1.5；2026-08-12 起问答双 worker 并行，实测进程峰值约 1.3GB、容器内预计 2-2.4GB），建议开 **2GB swap**，防止内存峰值触发 OOM kill：

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 验证
free -h
```

## 3. 上传代码

```bash
# 方式一：git clone（推荐）
git clone <你的仓库地址> resume-agent
cd resume-agent

# 方式二：scp 打包上传
# 本地：tar czf resume-agent.tar.gz --exclude='data' --exclude='.git' --exclude='node_modules' .
# scp resume-agent.tar.gz user@server:/opt/ && cd /opt && tar xzf resume-agent.tar.gz
```

## 4. 配置 .env

```bash
cd resume-agent
nano .env
```

关键项确认：

```dotenv
LLM_API_KEY=sk-你的DeepSeekKey          # 填入真实 Key
LLM_BASE_URL=https://api.deepseek.com
MODEL_DEVICE=cpu
RESUME_PERFORMANCE_MODE=cpu_low_resource
RESUME_OFFLINE_MODE=false            # 首次在线下载模型；下载完成后可改 true
HF_ENDPOINT=https://hf-mirror.com      # 国内下载镜像
TORCH_NUM_THREADS=2
RERANK_INPUT_MODE=compact
RERANK_CANDIDATE_LIMIT=12
RERANK_TOP_K=12

# 前台/后台权限分离（必填）：管理员密码，进入后台（知识库管理/系统状态）使用
ADMIN_PASSWORD=改成强密码
ADMIN_JWT_SECRET=                     # 可选；留空时由 ADMIN_PASSWORD 派生
AUTH_REQUIRED=true                    # 生产必须 true
# 限流（防恶意刷 token）：问答每 IP 每分钟 30 次、每日 500 次，全局并发 4
RATE_LIMIT_ENABLED=true
QA_IP_RATE_LIMIT_PER_MINUTE=30
QA_IP_DAILY_LIMIT=500
QA_GLOBAL_CONCURRENCY=4
LOGIN_RATE_LIMIT_PER_MINUTE=10
# 使用 Cloudflare Tunnel 后置 true（按 X-Forwarded-For 取真实访客 IP）
RATE_LIMIT_TRUST_PROXY=true
```

**建议**：为本项目单独申请一个 DeepSeek API Key 并设置消费上限（保险丝）——即使被恶意刷量，也只会烧到上限自动停止。

## 5. 构建并启动

```bash
# 构建镜像（纯 CPU，约 1.5GB，首次构建 10-20 分钟）
docker compose build app

# 启动
docker compose up -d
docker compose ps          # 等待 app 变 healthy（模型加载慢，start_period 180s）
docker compose logs -f app
```

## 6. 模型下载与预热

首次启动会自动从 HuggingFace（hf-mirror 镜像）下载两个模型（embedding bge-small-zh-v1.5 约 100MB + reranker bge-reranker-base 约 1.1GB，合计约 1.2GB），下载到 `data/model_cache/`。模型默认后台自动预热（`MODEL_WARMUP_POLICY=background`），也可登录后手动触发：

```bash
# 1. 登录拿 token
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"password\":\"你的ADMIN_PASSWORD\"}" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
# 2. 手动预热
curl -X POST -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/health/warmup
# 期望: {"warmed": true, ...}
```

确认服务存活（公开接口，无需登录；容器 healthcheck 也用它）：

```bash
curl http://127.0.0.1:8000/api/health
# 期望: {"status":"ok", ...}
```

> 就绪状态（模型/向量库/文档数）属后台信息：登录后在前台侧栏系统状态面板查看；访客前台轮询轻量 `/api/qa/status`。容器 healthy 不等于模型就绪，模型就绪以系统状态面板为准。

> 若下载过慢或失败：也可在本地下载模型后 scp 到服务器 `data/models/bge-small-zh-v1.5/` 和 `data/models/bge-reranker-base/`（含 config.json、tokenizer 文件、*.safetensors），再把 `RESUME_OFFLINE_MODE` 设为 `true` 完全离线运行。

## 7. 上传知识库

在服务器上（或本地指向公网地址）运行上传脚本，扫描 `docs/` 下的个人材料：

```bash
# 先做静态与运行库审计；首次部署没有 app.db 时会跳过运行库核对
python3 scripts/audit_knowledge_base.py

# 服务器本地执行（需要服务器上有 docs/ 内容，或本地执行后指向公网地址）
# 管理员密码默认从环境变量 ADMIN_PASSWORD 读取，也可用 --admin-password 显式传入
python3 scripts/upload_knowledge_base.py

# 或者本地 Windows 指向服务器：
python scripts/upload_knowledge_base.py --base-url http://你的域名:8000 --admin-password "你的ADMIN_PASSWORD"
```

脚本会自动：扫描 → 上传 → 等待索引完成 → 打印向量数。图片（荣誉 jpg）无法解析，需先转 PDF。

不要从开发机复制 `data/qdrant/` 到服务器：个人知识库重建很快，而历史 segment/WAL 可能达到数 GB。只同步 `docs/` 与可选的 `data/models/`，然后在服务器重新上传建库。若旧环境已同时索引简历 PDF 和 `简历文字版.md`，先删除 PDF 文档，只保留文字版口径基线。

## 8. 公网访问

### 方式一：安全组放行（最快，无 HTTPS）

云控制台安全组放行 TCP 8000 端口，访问 `http://服务器IP:8000`。

### 方式二：Cloudflare Tunnel（免费，推荐，自带 HTTPS 免备案）

```bash
# 1. 安装 cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared

# 2. 登录并创建 Tunnel（会得到一段 Tunnel Token）
cloudflared tunnel login
cloudflared tunnel create resume-agent

# 3. 配置 /etc/cloudflared/config.yml
# tunnel: <TUNNEL_ID>
# credentials-file: /root/.cloudflared/<TUNNEL_ID>.json
# ingress:
#   - hostname: resume.yourdomain.com   # 需要已在 Cloudflare 解析
#     service: http://127.0.0.1:8000
#   - service: http_status:404

# 4. 运行
cloudflared tunnel run resume-agent
```

之后 `https://resume.yourdomain.com` 即可访问，HTTPS 自动处理，**国内服务器无需 ICP 备案**（Cloudflare 代理不要求绑定备案域名）。

> Cloudflare Tunnel 模式下记得把 `.env` 的 `RATE_LIMIT_TRUST_PROXY` 设为 `true`，限流中间件才会按 `X-Forwarded-For` 取真实访客 IP（否则所有访客被当成同一 IP）。

## 8.5 安全加固清单（上线前检查）

1. ✅ `AUTH_REQUIRED=true`（生产必须；`false` 只用于开发）
2. ✅ `ADMIN_PASSWORD` 为强密码（后台 = 知识库删除权限）
3. ✅ 为本项目单独申请 DeepSeek API Key 并设置**消费上限**（防恶意刷量烧钱）
4. ✅ 限流已开启（`RATE_LIMIT_ENABLED=true`，问答 30 次/分/IP + 全局并发 4）
5. ✅ 公网走 Cloudflare Tunnel（自带 DDoS 防护 + HTTPS），可再加 WAF 规则拦截非浏览器 UA
6. ✅ 知识库接口全部需要登录 token——访客（面试官）只能问答和看问答日志，无法上传/删除/看管理日志

### 方式三：nginx 反代 + HTTPS（域名已备案时）

```bash
apt install nginx
# /etc/nginx/sites-enabled/resume-agent
# server {
#   listen 80;
#   server_name resume.yourdomain.com;
#   location / { proxy_pass http://127.0.0.1:8000; proxy_set_header Host $host; }
# }
# 再用 certbot 签发 HTTPS 证书
```

## 9. 知识库维护（更新 / 删除）

简历、证书、项目介绍等材料修改后，重跑上传脚本即可**自动更新**（脚本比对文件哈希，内容变化的文件自动覆盖上传并重新索引）：

```bash
# 服务器上执行（或本地指向公网地址）
python3 scripts/upload_knowledge_base.py

# 按文件名删除某个文档
python3 scripts/upload_knowledge_base.py --delete "文件名.md"

# 清空整个知识库（内容大改后重建用）
python3 scripts/upload_knowledge_base.py --purge
python3 scripts/upload_knowledge_base.py        # 重建
```

说明：
- 修改 `docs/` 下任一文件后直接重跑，脚本自动完成"检测变化 → 覆盖上传 → 重新索引"，无需手动删除
- 内容未变的文件自动跳过，不会重复上传
- 图片（如荣誉 jpg）无 OCR 能力，修改后需先转 PDF 或更新对应说明文档
- 上传脚本在项目根目录运行（依赖 scripts/ 与 .run-state/）

## 10. 日常运维

```bash
docker compose logs -f app        # 查看日志
docker compose restart app        # 重启（模型重新加载约 30-90s）
docker compose down               # 停止（不删数据）
docker compose up -d              # 再次启动

# 内存观察
docker stats
free -h

# 调参（修改 .env 后 docker compose up -d 生效，无需重新构建镜像）
# RERANK_PROMPT_THRESHOLD / MIN_CORE_RERANK_SCORE：证据过滤阈值，回答过严/过松时调整
```

## 11. 常见问题

| 问题 | 处理 |
|---|---|
| 容器 OOMKilled | 确认已开 2GB swap；`docker inspect app --format '{{.State.OOMKilled}}'` 排查 |
| 模型下载失败/超时 | 检查 `HF_ENDPOINT=https://hf-mirror.com`；或本地下好模型 scp 上传后转离线模式 |
| 首次问答很慢（1-2 分钟） | 模型正在后台加载；先执行一次 `/api/health/warmup` |
| 回答找不到依据 | 知识库还没上传/索引完成；或问题太笼统，换更具体问法 |
| healthcheck 一直 unhealthy | 知识库为空时 ready 返回 503 属预期；上传文档索引后恢复 |
