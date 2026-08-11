# Embedding 模型 A/B 评测报告：bge-base-zh-v1.5 → bge-small-zh-v1.5（2026-08-11）

> 材料主题：系统说明

## 背景与目标

部署目标从 4C8G 收紧到 2C4G。本地模型栈中最重的是 embedding（bge-base-zh-v1.5，391MB fp32）与 reranker（bge-reranker-base，1.1GB fp32）。Reranker 已是 BAAI 官方中文最小档（无更小官方替代，ONNX INT8 因 parity 门禁不过被禁用于生产），唯一可降的是 embedding：bge-small-zh-v1.5（24M 参数、512 维、92MB），官方 C-MTEB 检索单项 61.8 vs base 69.5（-7.7）。本报告用 23 问黄金评测集（`scripts/eval_cases.jsonl`）实测切换前后的端到端质量。

## 方法与控制变量

- 评测集：23 问（A 自我介绍 / B 项目深挖 / C 技术栈 / D HR / E 简历细节 / F 无关话题 / G 寒暄 / H 多轮追问），含期望事实、禁止词、期望文档（Recall@文档）。
- 三组对照：① base + 旧分块代码（部署现状，119 点）；② base + 当前分块代码（111 点）；③ small + 当前分块代码（110 点）。
- ② 组用于排除"分块代码变更"这一混杂变量（旧索引是 08-03 前分块代码建的，与当前代码不一致）；② 与 ③ 是**唯一变量 = embedding 模型**的严格对照。
- 评测为真实端到端（本地模型 + Qdrant + DeepSeek API），详见 `docs-guide/evaluation.md`。

## 结果

| 运行 | 事实命中 | Recall@文档 | answered/hedged/redirected | p50 | p95 | 错误 |
| --- | --- | --- | --- | --- | --- | --- |
| ① base + 旧索引（部署现状） | 49/51 (96.1%) | 19/20 (95.0%) | 15/5/3 | 11.6s | 17.2s | 0 |
| ② base + 当前代码 | 45/46 (97.8%) | 16/18 (88.9%) | 12/6/3 | 15.1s | 60.0s | 2（API 超时） |
| ③ **small + 当前代码** | **50/51 (98.0%)** | **18/20 (90.0%)** | **16/4/3** | 13.1s | 17.3s | 0 |

> ② 组 E-15/E-16 两问因 DeepSeek API 单次超时（60s 评测超时）未完成，其期望事实未计入分母（46 而非 51），p95 被超时拉高；两问在 ① ③ 组均正常回答。排除该噪声后 ② ③ 组的事实命中与文档召回实质持平。

## 逐题差异（② vs ③，仅 2/23 问不同）

- **B-9（EchoGuide 多 Agent 路由）**：hedged → **answered**（③ 改善；检索到 项目介绍_EchoGuide.md 等正确文档）。
- **E-17（证书/六级）**：事实命中 516 → 516+软件设计师（③ 改善）；但 `证书说明.md` 未进检索上下文（② 同样未进 → **该漏检是当前分块代码引入，与模型无关**），答案仍从 简历文字版.md 正确给出全部事实。

## 已知检索缺口（两组共有，与模型无关）

- C-10（Java 技术栈）：`技能专长.md` 未进上下文——三组运行均漏检，属既有问题，不在本次切换范围内。
- E-17：`证书说明.md` 漏检同上。

## 资源变化（2C4G 部署价值）

| 项目 | base | small | 变化 |
| --- | --- | --- | --- |
| embedding 权重内存 | ~391MB | ~92MB | **-300MB** |
| Qdrant 向量维度 | 768 | 512 | 索引体积减半、查询更快 |
| 磁盘占用（模型） | 391MB | 92MB | -300MB |
| 下载/SCP 时间 | 391MB | 92MB | 显著缩短 |

切换后 app 常驻估算从 ~2.0-2.5GB 降到 ~1.7-2.2GB；配合 2GB swap，2C4G 单访客展示档稳定性余量明显改善。Reranker 未动（仍为 PyTorch bge-reranker-base），全部证据门槛阈值（MIN_CORE_RERANK_SCORE 等）保持有效。

## 结论与建议

1. **可以切换**：同代码严格对照下，small 的事实命中（98.0%）与文档召回（90.0%）均不劣于 base（97.8% / 88.9%），回答模式（answered 16 vs 12）更好；唯一差异 B-9 为改善项。
2. 切换即：`EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5`、`EMBEDDING_MODEL_PATH=…/bge-small-zh-v1.5`、`EMBEDDING_DIMENSION=512`、`INDEX_VERSION=bge-small-zh-dense-v6-resume`，删除 Qdrant 集合后执行 `python scripts/rebuild_vector_index.py`（本项目已按此配置落地为默认值）。
3. 回退方法：三个环境变量改回 base 值（768 维 + 旧 INDEX_VERSION）+ 重建索引；`data/models/bge-base-zh-v1.5/` 保留未删。
4. 后续可选：排查 C-10 / E-17 的 `技能专长.md`、`证书说明.md` 漏检（关键词召回或候选数微调），与本次模型切换独立。
