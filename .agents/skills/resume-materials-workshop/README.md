# resume-materials-workshop

**任意求职者原始资料 → 检索友好知识库 Markdown** 的 LLM 加工 skill。

把聊天记录、PDF、DOCX、简历文本、课程作业说明等杂乱原始材料，加工为简历问答系统
可直接检索的知识库：一主题一文件的 Markdown 文档（`> 材料主题：类别` 开头）、
人物档案、结构化事实。忠于材料、不补写不拔高、自动隐私清洗，并带有机器可读的输出契约
与离线自测——既是 Agent 可执行的 skill，也是后端「人物工坊」的规范驱动来源。

## 能力

- **解析**：多格式原始材料（txt / md / pdf / docx / doc / html / jsonl）解析为文本
- **加工**：LLM 批量转换为结构化 JSON（人物档案 + 知识库 Markdown 文档 + 事实）
- **契约化**：输出由 `references/output-schema.json`（JSON Schema）校验，不合规即失败
- **可信**：事实/观点分离、无法确认标 `[待确认]`、硬数字照抄原文、资料冲突保留并指出
- **隐私**：身份证/手机号/邮箱/银行卡等 PII 一律脱敏为 `[已脱敏]`，不进入知识库
- **可验证**：`scripts/self_test.py` 离线自测契约与黄金样例
- **可追溯**：版本号记录到每次加工任务，失败可一键回滚
- **可交付**：加工产物二次封装为可独立调用的人物 Skill 包（`persona-{人名}/SKILL.md +
  references/ + facts.json`，渐进式加载），下载即用

## 目录结构

```
resume-materials-workshop/
├── SKILL.md                      # 使用说明与质量红线（版本号在此 frontmatter）
├── README.md                     # 本文件（对外宣传）
├── references/
│   ├── transform-prompt.md       # 「材料加工师」System Prompt 原文（后端运行时加载）
│   ├── output-schema.json        # LLM 输出 JSON 契约（机器校验）
│   └── persona-skill-template.md # 人物 Skill 包 SKILL.md 模板（确定性组装，不额外调 LLM）
├── assets/
│   ├── sample-input.txt          # 黄金样例：原始材料
│   └── sample-output.json        # 黄金样例：加工结果
└── scripts/
    └── self_test.py              # 离线自测（契约合法性 + 黄金样例 + 负例 + 模板检查）
```

## 快速开始

```bash
# 自测（离线，无需 LLM）
python scripts/self_test.py

# 由 Agent 使用：把本 SKILL.md 交给 Agent，或放入 .agents/skills/ 自动发现；
# 按 SKILL.md「工作流」处理用户材料，输出为知识库 Markdown 文档。

# 由系统接入（规范驱动）：后端运行时从本目录加载提示词与契约，
# 无需复制规则到业务代码。见 SKILL.md「与系统集成」。
```

## 输出契约要点

```json
{
  "persona_profile": { "name": "…", "summary": "…", "skills": ["…"] },
  "knowledge_documents": [
    { "filename": "项目经历_xxx.md", "content": "> 材料主题：项目经历\n\n…" }
  ],
  "facts": [ { "subject": "…", "predicate": "…", "value": "…",
    "evidence_status": "explicit", "source_file": "简历.pdf" } ]
}
```

- 每篇文档一个主题；content 首行 `> 材料主题：类别`（类别限 10 种，见契约）
- 每个段落一个完整事实或一个「场景—方案—结果」，避免跨段指代
- facts 必须带 `evidence_status` 证据分级（explicit/inferred/conflict/missing）与必填的 `source_file` 来源文件名（只能来自输入材料标注）

## 版本

| 版本 | 日期 | 变更 |
| ---- | ---- | ---- |
| 1.0.0 | 2026-08-14 | 从 `backend/app/services/materials_workshop_service.py` 的硬编码提示词迁出，形成独立 skill；新增输出 JSON 契约与离线自测；修正提示词中残留的双花括号转义（`{{"filename"}}` → `{"filename"}`） |
| 1.1.0 | 2026-08-15 | 新增「人物 Skill 包产物」：加工产物（知识文档 + 人物档案 + 事实）确定性组装为可独立调用的人物 Skill（`references/persona-skill-template.md` 模板，零额外 LLM 调用）；后端在转换完成/档案确认/任务回滚后自动重建，zip 下载即用 |
| 1.2.0 | 2026-08-15 | 安全与工程化：System/User 消息分离（材料声明「不可信数据」，防 prompt injection）；PII 解析期本地预清洗 + 生成后二次清洗（三出口统一，补家庭住址正则）；facts 契约强制 `source_file` + `evidence_status`（旧 status 拆为证据/审核两列）；新增 `scripts/validate_output.py` 确定性业务校验与 Reduce 归并（文档去重/冲突重命名、事实冲突标记，冲突随任务落库）；入库原子化（校验先行 + 增量台账 + 失败自动回滚）；`skill_loader` 升级为通用 Skill Registry；新增 `evals/` 评测套件（with-skill vs baseline） |

## 许可

Proprietary（ResumeMind 项目内部资产）。对外展示请先完成脱敏审查。
