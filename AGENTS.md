# ResumeMind 协作规则

## 本地执行

- 本项目在 PowerShell 环境中执行。文件发现使用 `Get-ChildItem -Recurse -File`，文本检索使用 `Select-String`。
- Python 输出含中文路径或数学符号前，设置 `$env:PYTHONIOENCODING='utf-8'`。
- 不改写或删除用户已有资料；生成物与原始资料分开保存。知识库成品只放入 `docs/`，系统说明放入 `docs-guide/`。

## 把个人资料整理为知识库 Markdown

目标是让问答系统准确复述本人经历，并能应对面试追问；不是把资料改写成宣传稿。
加工规则（忠于来源 / 一篇一主题 / 事实与观点分离 / 隐私 / 去噪）的**单一事实来源**是
`.agents/skills/resume-materials-workshop/references/transform-rules.md`（LLM 批量加工时
由 skill_loader 拼接进 System Prompt）；`SKILL.md` 是流程与资源导航，`scripts/validate_output.py`
是确定性校验，`scripts/self_test.py` 是离线自测。规则冲突时以 skill 目录为准。

- 忠于来源：姓名、时间、角色、技术、指标和成果不得补写或拔高；无法确认的内容标为
  `[待确认]`，合理推断标为 `[推断]`；资料冲突时保留冲突并指出，不擅自选一个版本。
- 隐私红线：身份证号、银行卡、口令、密钥、家庭住址等不得进入 `docs/`；联系方式仅在本人
  明确要求公开时保留（加工产物中一律替换为 `[已脱敏]`）。

完成前检查：文件为 UTF-8、只有一个一级标题、材料主题正确、没有新增事实、关键数字有依据、
与其他文档不存在未说明的冲突。最后向用户列出 `[待确认]` 与冲突项，不替用户作事实决定。
