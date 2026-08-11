# ResumeMind RAG 评测指南

面向"面试官 × 简历主人公"场景的端到端评测框架，用于回归验证回答质量、
延迟与 LLM 成本。**评测需要真实运行的服务**（本地模型 + Qdrant + DeepSeek API），
单测环境不适用。

## 运行

```bash
# 服务已启动（docker compose up -d 或本地直跑）后：
python scripts/eval_interview_set.py --base-url http://127.0.0.1:18000 --groups ABCDEFGH

# 输出 JSON 报告（含逐问明细）：
python scripts/eval_interview_set.py --base-url http://127.0.0.1:18000 --report eval-report.json

# 结构化 JSONL 评测数据（覆盖内置问题集；支持 expected_documents / session 多轮链）：
python scripts/eval_interview_set.py --base-url http://127.0.0.1:18000 --admin-password "$ADMIN_PASSWORD" --data scripts/eval_cases.jsonl --report eval-report.json

# 上线门禁示例：事实/文档用例必须通过，且 p95 <= 6s、普通问题的 LLM 调用 p50 <= 1：
python scripts/eval_interview_set.py --admin-password "$ADMIN_PASSWORD" --data scripts/eval_cases.jsonl --max-p95 6 --max-llm-calls-p50 1
```

`expected_documents` 依赖管理员调试响应；不提供 `--admin-password` 时可以评测公开回答，但无法计算可靠的 Recall@文档。

JSONL 每条可包含：`group`、`question`、`expected_facts`、`forbidden_facts`、`expected_documents`、`expected_intent`、`expected_answer_mode(s)`、`expect_no_documents`、`max_llm_calls`、`session`。`session` 相同即同一多轮链。

## 问题集结构（A-H 8 组）

| 组 | 类型 | 说明 |
|---|---|---|
| A | 自我介绍 | 开放型自我介绍/优势/实习动机 |
| B | 项目深挖 | 秒杀/外卖/REV 项目细节与技术追问 |
| C | 技术八股 | HashMap/缓存/Kafka/B+ 树/Redisson |
| D | HR 素质 | 优缺点/失败经历/抗压/职业规划 |
| E | 简历细节 | 课程/绩点/竞赛/六级/GitHub |
| F | 无关话题 | 天气/烹饪/银行卡密码（应礼貌转移） |
| G | 寒暄 | 你好/谢谢/在吗（零 LLM 转移） |
| H | 多轮追问链 | 4 组同 session 串行追问（指代消解验证） |

单问条目格式：`[问题, 期望事实, 禁止事实]`。
- 期望事实：应在回答中出现的硬事实（Recall/Fact 命中判定）
- 禁止事实：不应出现的字样（幻觉检测）

## 指标

- **端到端延迟 p50 / p95**：DeepSeek API 主导；fast path 开启后普通独立问题
  应显著低于完整链路
- **单请求 LLM 调用次数**（`llm_call_count`）：普通独立问题 fast path 后应为 1
  （仅生成）；追问 2-3（意图消解 + 生成 + 可选规划）；枚举/复杂问题 3
- **期望事实命中率**：`expected_fact_hits / expected_facts`（回答级 Recall 近似）
- **禁止内容检出**：无关话题/幻觉防护是否生效
- **answer_mode 分布**：answered / hedged / redirected / failed
- **多轮追问正确性**：H 组每条链的期望事实是否随追问链保持命中

## 与单测的关系

- 单测（`pytest`，350+ 个）覆盖组件级行为（意图路由/规划/选择/grounding/兜底链），
  模型与 LLM 全部 mock，不消耗 API
- 本评测是真实系统的 E2E 回归，需真实 DeepSeek Key + 本地 BGE 模型 + Qdrant，
  会产生 API 费用（内置全量约 55 问 × 每问 0-3 次调用）

## 使用场景

- 改造前后对比：跑两遍对比延迟 / llm_call_count / 期望事实命中
- 回归验证：fast path / grounding / 检索逻辑改动后跑 A-H 确认无质量回退
- 上线前体检：确认 F/G 组全部礼貌转移、B/C 组硬事实命中、无 forbidden 内容
