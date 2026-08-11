# 2C4G / 4C8G 稳定运行手册

## 结论

2C4G 可以作为低成本展示档，但它是**单访客优先**配置：同一时刻的本地模型推理会串行，第二位访客排队是刻意的稳定性保护。若目标是未来 2～3 个月在面试时稳定可用，优先选择 4C8G；它不是为了提高并发数字，而是为模型、Qdrant、Docker 和系统页缓存留下确定余量。

| 档位 | 适用场景 | 不应突破的边界 |
| --- | --- | --- |
| 2C4G | 个人演示、预约面试、低并发访问 | `QA_GLOBAL_CONCURRENCY=2`、仅保留一个新建索引、禁止边面试边批量上传 |
| 4C8G | 连续 2～3 个月开放链接、面试官可能同时访问 | 保留 2 个并发槽；优先改善首问和峰值余量，而不是盲目提高并发 |

## 已验证的 reranker 决策

同一 `BAAI/bge-reranker-base` 已完成三条路径的实测：

| 后端 | 排序 / 分数一致性 | 结论 |
| --- | --- | --- |
| PyTorch FP32 | 基线 | 当前生产默认 |
| ONNX FP32 | 23 个评测问题 Top-1、Top-5 均 100% 一致 | 可作为受控实验；加速有限 |
| ONNX 动态 INT8 | Top-1 一致率 86.96%，未达到 95% 门槛 | **禁止用于生产** |

不应仅因 INT8 文件更小就切换；本系统用 rerank 分数作为证据门槛，分数漂移会直接影响“有证据/拒答”的判断。

## 部署方式

1. 新服务器只上传当前 `docs/`、两个本地模型、Docker 镜像构建所需代码；不要复制开发机的 `data/qdrant/` 和 `data/app.db` 历史文件。
2. 选择一个配置文件：`deploy/2c4g.env.example` 或 `deploy/4c8g.env.example`，将其中条目合并到服务器根目录 `.env`，再补齐密钥和域名配置。
3. 首次启动后重建索引，并在管理员后台确认 `/api/health/ready` 的 embedding、reranker、Qdrant、SQLite 均为 ready，且 warmup 为 warmed。
4. 用 `scripts/eval_interview_set.py` 跑回归；上线前至少手动提问 10 个评测问题，确认引用材料正确。

`MODEL_WARMUP_POLICY=blocking` 会让容器启动更慢，但可避免面试官的第一问触发模型加载。Docker healthcheck 已为此预留 180 秒。

## 3 个月守护线

- 磁盘：40 GiB 系统盘至少保留 10 GiB；每周检查 `docker system df`、`data/qdrant` 和日志体积。
- 内存：后台资源面板持续出现接近容器上限或重启，即升级至 4C8G，不要通过关闭 reranker 解决。
- 数据：仅在非面试时段上传、审计、重建索引；一次只保留一个当前 collection。
- 发布：先执行 `docker compose up -d --build`，待 warmup 完成后才切换访问链接；保留上一份 `.env` 和镜像标签以便回滚。

## 阿里云学生算力包选择

阿里云当前帮助文档说明，毕设包和科创包可叠加 300 元学生优惠券；科创包有 2000 元 ECS 按量额度、6 个月有效期，支持的规格最高到 8 vCPU / 16 GiB。购买页的实时规格与小时价格才是最终依据。

若你看到的科创包实付是约 89 元（常见为活动价减去 300 元券后的结果），它是这次项目的推荐备选：在购买页选**4 vCPU / 8 GiB、Linux、40 GiB 系统盘、按量付费**，并确认所选规格的小时单价不高于 `2000 / (90 × 24) ≈ 0.93 元/小时`，这样额度可覆盖至少 3 个月。若你只部署两个月，小时预算上限为约 1.39 元。

这些规则和额度以阿里云的[学生专属算力包常见问题](https://help.aliyun.com/zh/ecs/faqs-on-student-specific-computing-packages)与[学生权益说明](https://help.aliyun.com/zh/document_detail/2881975.html)为准；下单前以购买页展示为准。
