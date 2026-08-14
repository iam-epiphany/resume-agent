---
name: resume-materials-workshop
description: >-
  把求职者的任意原始资料（聊天记录、PDF、DOCX、简历文本、课程作业说明等）加工成检索友好的
  知识库 Markdown 文档（一主题一文件，以 `> 材料主题：类别` 开头），同时产出人物档案与
  结构化事实；忠于材料、不补写不拔高、自动隐私清洗。当用户上传/粘贴简历、自我介绍、
  项目经历等原始材料并要求「整理成知识库」「做成 Markdown 材料」「写入 docs/」时使用；
  后端人物工坊（POST /api/workshop/transform）运行时也按本 skill 的提示词与输出契约
  驱动执行，本 skill 是转换规则的单一事实来源。即使没有明说 "skill" 也要考虑使用。
metadata:
  version: "1.0.0"
  author: ResumeMind
  license: Proprietary
---

# 简历材料加工（任意资料 → 检索友好知识库 Markdown）

## 何时使用

用户给出某求职者的**原始资料**（简历文件、聊天记录、PDF/DOCX 文本、课程作业、零散笔记），
并要求整理为知识库、生成材料文档、或入库供简历问答系统检索。识别特征是「原始、杂乱、
未经整理」，输出是**结构化、面向检索、忠于原文**的 Markdown 知识库。

## 工作流

1. **收集与解析**：读取全部原始材料（多文件时逐份解析取文本），剔除空材料；解析失败要
   报告具体文件与原因，不静默跳过。
2. **分批**：单批输入不超过 `WORKSHOP_MAX_INPUT_CHARS`（默认 12000 字符），优先按段落
   边界切分；每批调用一次 LLM。
3. **LLM 加工**：以 `references/transform-prompt.md` 全文作为 System Prompt（填入
   `{batch_index}/{total_batches}/{input_text}` 占位符），要求 `response_format=json_object`、
   temperature=0。LLM 只做加工，**不要**让模型把 Markdown 直接输出到文件系统。
4. **契约校验**：LLM 返回的 JSON 必须通过 `references/output-schema.json` 的校验
   （`jsonschema`）。不合契约时报告校验错误路径，不降级入库。
5. **隐私清洗**：对每篇文档内容做 PII 正则清洗（身份证/手机号/邮箱/银行卡 → `[已脱敏]`）。
6. **入库**：每篇 Markdown 作为独立文档入库并索引（persona_id=当前人物）；人物档案存为
   draft（人工确认后生效）；结构化事实入事实台账（默认 pending，只告警不硬校验）。
7. **留痕**：记录任务（skill 版本、LLM 调用次数、生成文档清单），失败可一键回滚。

## 质量红线（为什么：简历问答会被追问，错误必须来自材料本身）

- **忠于材料**：姓名、时间、角色、技术、数字、成果不得补写或拔高；材料没有的信息标
  `[待确认]`，合理推断标 `[推断]`；资料冲突时保留冲突并指出，不擅自选一个版本。
- **事实与观点分离**：压测数字、排名、奖项等硬事实照抄原文数字，禁止估算；
  「熟悉/精通/显著提升/零故障」等措辞必须能被材料支撑。
- **一篇一主题**：每篇文档只讲一个主题，文件名即主题（`主题_文件名.md`）；
  content 首行必须为 `> 材料主题：类别`，类别限：项目经历/技能专长/教育背景/竞赛奖项/
  荣誉奖励/证书资格/求职意向/个人特质/自我介绍/综合简历。
- **面向检索**：每个段落只陈述一个完整事实或一个「场景—方案—结果」，用具体名词，
  避免「它/这个/上述」等跨段指代；首次出现的缩写给出中文含义。
- **隐私**：身份证号、银行卡、手机号、邮箱、家庭住址等一律删除或替换为 `[已脱敏]`，
  绝不进入知识库。
- **去噪**：删除招聘套话、重复结论、无关源码与通用技术科普；保留能回答
  「你做了什么、为什么这样做、结果如何、还能怎么改」的信息。

## 输出契约

LLM 必须返回一个 JSON 对象，结构见 `references/output-schema.json`（机器可读的唯一契约）：

```json
{
  "persona_profile": { "name": "…", "summary": "…", "skills": ["…"] },
  "knowledge_documents": [
    { "filename": "项目经历_xxx.md", "content": "> 材料主题：项目经历\n\n…" }
  ],
  "facts": [
    { "subject": "…", "predicate": "…", "value": "…", "status": "pending" }
  ]
}
```

示例：`assets/sample-input.txt`（原始材料）→ `assets/sample-output.json`（黄金样例）。

## 自测

```bash
python scripts/self_test.py
```

离线自测：校验契约文件本身合法、黄金样例通过、几类非法输出被正确拒绝。
契约或提示词改动后必须跑通自测，再修改后端版本号（SKILL.md `metadata.version`）。

## 与系统集成

- 后端「规范驱动」：`backend/app/services/skill_loader.py` 运行时读取本目录的
  提示词、契约与版本；`WORKSHOP_SKILL_DIR` 环境变量可覆盖目录位置。
- 版本以本文件 frontmatter `metadata.version` 为唯一来源，随每次任务记录到
  `workshop_jobs.skill_version`，前端任务表展示。
- 修改规范后同步提升版本号；后端无需改代码即可切换加工规则。
