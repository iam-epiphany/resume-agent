# Resume-Agent 改造进度记录

## ✅ 访客问答访问码闸 + 全局预算保险丝（2026-08-04，后端 301 测试全绿 + 前端 35 全绿）

**需求（用户拍板）**：简历分享给面试官后，如何让面试官正常使用问答、又防止陌生人/脚本恶意消耗 DeepSeek 额度。只实施访问码闸（Wbz_123，一天 JWT 存 cookie）；预算超限时前端弹窗提醒。

**改动**：
- **访问码闸（后端）**：`security.py` 新增访客 JWT（独立 sub=qa_visitor，与管理员 token 互不通用）、常量时间校验访问码；`api/qa.py` 新增 `POST /api/qa/access`（校验码→签发 24h httpOnly cookie）与 `GET /api/qa/access/status`；`require_qa_access` 依赖覆盖全部问答端点（ask/tasks/stream/retrieve），管理员 token 旁路；`QA_ACCESS_CODE` 为空 = 闸门关闭（开发/测试默认），部署时 .env 设置
- **全局预算保险丝**：`QA_GLOBAL_DAILY_LIMIT`（默认 300）跨 IP 统计当日 qa_logs；剩余 ≤ max(30, 15%) 时响应带 `X-QA-Budget-Warning` 头 + 前端弹窗"今日预算即将超限"（每会话一次）；用尽后新问答 429"今日问答预算已用完，请明天再试"（防换 IP 刷爆）
- **前端**：RagPage 挂载时拉取访问闸状态 + 每 60s 轮询预算；未授权显示访问码门（输入框 + 校验 + 错误提示），授权后正常问答；预算提醒横幅可关闭
- **配置**：.env 写入 `QA_ACCESS_CODE=Wbz_123`、`QA_GLOBAL_DAILY_LIMIT=300`、`COOKIE_SECURE=false`（HTTPS 部署时置 true）

**测试**：新增 `test_qa_access.py`（10 用例：闸门默认关闭/未授权 401/错误码 403/正确码签发 cookie/管理员旁路/状态端点/预算告警头/预算用尽 429/任务创建告警头/当日计数只算今天）；隔离方案——独立 SQLite 引擎 + patch 各模块 SessionLocal 引用（Depends 在路由注册时已捕获 get_db，patch 模块属性无效）；全量 301 通过

**遗留**：①docker 容器仍运行旧代码，需 `docker compose up -d --build`；②DeepSeek 平台侧仍建议设置余额预警（系统内预算与平台预算互为双保险）

## ✅ 检索规划确定性化 + 多路召回 + 配额式选择 + 生成纪律（2026-08-03，后端 291 测试全绿 + 真实系统复测通过）

**需求（用户拍板）**：追问"除了这三个还有哪些项目"答错（把已介绍的 REV/ReguMate 又答一遍、声称"知识库没有详细记录"），根因是四层系统缺陷叠加：规划层靠 LLM 猜（臆测排除名单）、召回层只有 dense 单路（列举对象 rank 40+）、选择层多 pass 特判（词法门挤掉 0.61 分对象块）、生成层无纪律（前缀重复/断言缺失/直生自由发挥）。要求**系统级改造而非个案补丁**。

**核心改动**：
- **新增 `question_analyzer.py`（确定性问句分析器）**：枚举/补集问句识别（"还有哪些其他X/除了…之外/剩下的X"，排除"还有什么想问的"误判）、排除实体提取（显式实体 vs "这三个/你刚才提到的三个项目"指代→会话上下文解析）、对象类别识别（项目/技能/奖项→文档清单归类）、已知实体解析（上轮回答与对象文档名重叠匹配）
- **`query_planner_service`**：枚举/补集问句命中时从文档清单**确定性拆 aspect**（每对象文档一个、文档名锚定查询），LLM 只做查询措辞增强（失败降级不影响结构）；`plan_query` 新增 `memory_context`（上轮 Q/A 贯穿 planner，解析"这三个"指代）；QueryPlan 新增 `excluded_documents`（补集排除项，选择层软降权）
- **关键词精确召回（`_keyword_recall_candidates`）**：小库全量扫 chunk 快照，对象名子串命中注入候选池（dense 对列举对象弱，这是可靠锚点），相关性仍由 rerank 裁决
- **枚举问句融合检索（`_retrieve_aspects_fused`）**：多对象 aspect 的查询一次 dense 召回 + 一次 BGE rerank（rerank 查询=原问题+对象名），按查询命中切分回各 aspect——枚举问句端到端 81.9s → 17.7s（省 4/5 重排计算）；关键词候选无条件进重排、保留全部重排结果供选择层取舍
- **`_select_prompt_chunks` 重写为配额式选择**：统一 rerank 分数为唯一相关性信号；阶段 1 每 aspect 至少 1 块（词法/锚点优先）；阶段 2 每文档配额（`PER_DOCUMENT_PROMPT_CAP=2`）+ 对象文档优先级（排除文档最后）；阶段 3 单文档小库兜底；终检保留非枚举关键词门（防"综合成绩"召回"项目经历"章节），枚举放行
- **兜底链第 1 级不再整链重跑**：relaxed 复用同一 plan/候选池仅放宽选择门槛（省一次 LLM 规划 + 10s+）
- **生成纪律**：推测前缀完全由系统管理（剥 LLM 自写前缀后精确 prepend 一次，治双前缀）；prompt 新增"不得断言知识库中不存在某信息"；level-3 直生限制为 persona 软信息、硬事实必须说明"知识库未收录"；生成 prompt 注入对话上下文（上轮 Q/A）
- **可观测性**：`qa_logs` 新增 answer_mode/evidence_sufficiency/fallback_level/used_chunks 列（含 SQLite 迁移），替换恒 0 的假日志

**数据治理（P0，用户拍板）**：删除 `docs/项目/*/项目介绍_*.md` 5 个重复文件（root 版为唯一事实源，Qdrant 索引即 root 版无需重建）——此前两版分叉导致"回答与知识库真实情况不符"

**实测（真实系统，dev 实例）**：①"你的项目经历"→answered，秒杀(2026.02-06)+REV+外卖+EchoGuide 全列出；②"除了这三个还有哪些项目"→answered/sufficient/fallback=0，excluded_docs 正确解析出上轮介绍的秒杀项目，回答正确列出外卖/ReguMate/EchoGuide/REV（不再重复已介绍项目、不再伪称"知识库无记录"）；③离题/寒暄 1s 内礼貌转移（零 LLM）；④秒杀深挖/超卖追问/技术栈/爱好全部 answered；⑤前缀唯一（无双前缀）

**测试**：新增 `test_question_analyzer.py`（33 用例：枚举/补集/指代/实体/确定性规划/LLM 增强合并/失败降级）；`test_fallback_chain`/`test_rag_api`/`test_trust_enhancements` 适配新选择器与 `_plan_and_retrieve` 拆分；全量 291 通过

**遗留**：①"你还有什么想问的"类情景问句仍会 hedged 推测（面试官反问题，超出本次范围）；②docker 容器仍运行旧代码，需 `docker compose up -d --build` 重新构建部署；③枚举问句 ~18s、普通问句 8-16s（CPU rerank + DeepSeek API 主导，与基线持平）

## ✅ 去硬编码重构：LLM 意图理解层 + 检索净化（2026-08-03，后端 256 + 前端 35 全绿 + A-H 8 组端到端评测通过）

**需求**：多轮专项修改积累了大量针对当前简历材料写死的"取巧"逻辑，绕过 RAG 检索——关键词表判意图、身份快通道拼字符串、文件名强插项目枚举、词法子串伪造 rerank 分、邻块/兄弟文档词法补充。用户拍板：加 LLM 认知中间层替代硬编码，并彻查所有类似情况。

**核心决策（用户拍板）**：① 知识库 13 篇文档内嵌"面试追问 FAQ"段全部删除；② 意图理解纯 LLM（删光关键词表，失败保守默认 resume_qa）；③ 意图粒度精简为 3 类（resume_qa/greeting/off_topic）；④ 证书"真实性/有效期"确定性快通道删除，统一走 RAG。

**阶段 0 知识库净化**：13 篇文档删 `## .*FAQ` 段（技能专长 72 行八股问答在内）→ `upload_knowledge_base.py` 重建索引（16 文档，sha256 自动覆盖）→ 无 FAQ 基线评测：C 组八股（Kafka 不丢消息/B+ 树索引）从"精确"变 hedged/partial"根据现有知识库推测"合规标注，B 组技术追问（秒杀防超卖/Redis 预扣/Redisson 锁失效）仍 answered/sufficient（项目文档事实章节本身含答案）。

**阶段 1 意图层**：
- `intent_router_service.py` 整文件重写：7 类 → 3 类常量；删 `_RULE_ORDER`/`_RULE_TERMS`/`_rule_classify`/RetrievalStrategy 未消费字段；新增 `classify_and_resolve(question, previous_turn)`——一次 LLM 调用合并「3 类分类 + 追问补全」（输出 intent/confidence/reason/rewritten_question/needs_context），失败/坏 JSON 保守回退 resume_qa 原问；分类 prompt 不内嵌项目名单，寒暄+实质问题混合归 resume_qa，无法确定归 resume_qa
- `conversation_memory_service.py` 瘦身：删 `_INTERPOLATION_MARKERS`/`_needs_disambiguation`/`_llm_resolve`/`_DISAMBIGUATION_PROMPT`/`resolve_question`（消解职责移交意图路由），保留 recent_turns/record_turn/purge_expired 持久化
- `document_identity_query_service.py` 整文件删除（+ rag_service import/调用块/qa_identity_answered 审计分支）
- `rag_service.answer_question` 新流程：① classify_and_resolve（有上一轮才附上轮摘录）→ ② greeting/off_topic 礼貌转移（话术不变，触发源从规则表变 LLM）→ ③ resume_qa 全 RAG 主链 → ④ 单次自评生成
- config `INTENT_ROUTER_MAX_TOKENS` 160→300（rewritten_question 需要）
- 实测回归：`你好，介绍下你的项目` 走完整 RAG（旧规则整题转移寒暄）；`软考证书是真的吗还有效吗` 走 RAG+推测标注（旧快通道拼字符串）；寒暄/无关 LLM 触发转移正确；H 组追问链补全连贯（"那 Redis 呢？"正确补全）；H4"会打篮球吗？"由硬转移改进为合理推测回答

**阶段 2 检索净化（最风险）**：
- rag_service 删 A5 枚举强插（`_ensure_project_documents_covered` 12 短语+文件名直取）、A6 词法逃生通道（`_bounded_document_lexical_support_matches` 等 6 函数，字符子串伪造 rerank 0.88+ 注入）、A6b 邻块扩展整链（`_expand_neighbor_matches`/`_neighbor_candidates`/`_neighbor_relevance`/`_match_from_chunk` 等 5 函数）、A7 兄弟文档词法补充（`_recover_missing_aspects_from_sibling_documents`）；共删 22 个函数；`_normalize_exact_support_text` 删项目名别名表只留空白/标点归一化；`_prompt_aspect_coverage_terms` 删词法命中动词表；三处角色放行名单只留 `dynamic_table_evidence`（顺带清 formula_target_support 死角色）
- retrieval_service：question_terms 70+ 简历领域词表 + `_expanded_domain_terms` 同义扩展表 → **通用分词器**（书名号/引号整体保留 + CJK 2-gram + 字母数字 token + 停用词表，零同义替换）；`_normalize_for_match` 只留空白折叠；`_should_enforce_diversity` 删除（多样性无条件生效）；删 `_is_comparison_question` 死码；**`_rank_key` 移除 direct_evidence 权重 3.0**（rerank 主导——测试暴露词表删除后 3.0 放大噪声，验证后拍板）
- planner 枚举规则：prompt 增加"对象未知也要按对象逐个拆 aspect + 每 aspect 至少 1 条 document_style_statement 查询"；`plan_query_budget` 对枚举问句（对象名词+哪些/有什么/介绍下你的 X）放宽 max_aspects；`QueryPlan.enumerative` 标志贯通
- 检索端枚举配套：枚举问句**全量 rerank**（70 候选不截断，否则秒杀/REV 被 fusion 截断挤出）、prompt 容量放宽（forced_minimum 补到 12）、补选门槛降为 RERANK_PROMPT_THRESHOLD*0.5（0.2 绝对门槛挡掉 rerank 0.05-0.19 的项目简介块）、终检 `_filter_prompt_relevance` 枚举放行
- 环境修复：**tokenizers 0.20.3 → 0.22+**（transformers 4.57.6 要求 tokenizers>=0.22，FlagModel 导入失败导致 embedding 全 503）
- 实测：列举类问题（"你参与过哪些项目"/"介绍一下你的项目经历"/"你做过哪些项目"）5 个项目全部列出（fallback=0 单次检索足够）；A/B/E 组复测正常；兜底链人工用例正常

**阶段 3 Prompt 泛化 + 收尾**：prompt_builder CORE_RAG_RULES 第 5 条泛化（"把检索到的全部相关材料一并列出；只列检索到的内容，不得补充知识库外的项目"——去 `项目介绍_*.md` 文件名约定）；`_REWRITE_PROMPT` 示例改通用（去秒杀/乐观锁/Redisson/Lua）；删 `_infer_evidence_type`（fallback 一律"相关材料依据"）；`plan_query` 删 options 死参数

**阶段 4 全量回归**：后端 pytest 256 全绿；前端 35 全绿 + tsc + build 通过；A-H 8 组端到端评测复跑（43 问 0 异常：answered 31/hedged 5/redirected 7，3 类意图分类全对，H 组 4 条追问链补全连贯，枚举列举 5 项目全列出）；tech-highlights.md/README/PROGRESS.md/模块 docstring 更新（"规则优先/锚定文档"描述全部失效）。**延迟审计**：43 问中 11 问 <6s、32 问 6-41s——未达 <6s 目标，但与无 FAQ 基线持平（基线微信支付回调 34.4s/缓存穿透 35.2s），属 DeepSeek API 延迟主导（每问 3 次 LLM 调用：意图+planner+生成），非本次改造引入（合并消解使每问 LLM 调用从 4 次减到 3 次）；可调项：`INTENT_ROUTER_MODEL` 独立指向更快小模型（config 已支持）

**测试改造**：test_intent_router_service 整文件重写（≈17 个：3 类分类/混合归 resume_qa/LLM 失败 fallback/合并消解）；test_conversation_memory_service 删消解 5 个（持久化保留，intent 值改 resume_qa）；test_trust_enhancements 删 3 个（别名归一化/bounded lexical/身份快通道）；test_rag_api intent 断言 resume_detail→resume_qa（8 处）+ 邻块扩展 2 测试改为新行为断言；test_retrieval_service 排序断言适配 rerank 主导

**动态文档清单注入（2026-08-03 追加，用户拍板）**：planner/改写 prompt 注入运行时知识库文档清单（`rag_service._document_catalog_summary` 查 Document 表拼"文件名（标题）"，动态数据非硬编码）——planner 枚举规则强化"对照清单逐项拆 aspect + 查询引用文档名/主题锚点"，prompt 显式约束清单不得输出到回答。**效果**：枚举问句 planner 从 1 个宽泛 aspect 升级为 5 个精准 aspect（每项目独立检索查询带项目名），fallback=0、5 项目全列出、文件名不泄漏；测试 258 全绿（新增 catalog 注入/空清单兼容 2 用例，mock 签名适配 catalog 参数）

**遗留**：① `.env` 的 RERANK_PROMPT_THRESHOLD=0.20 对宽泛问句偏严（枚举问句已自动降半，普通问句如觉召回不足可再校准）；② scripts/eval_interview_set.py 新增评测脚本（A-H 8 组一键复跑）；③ 延迟 <6s 未达标（DeepSeek API 主导，见阶段 4 说明，`INTENT_ROUTER_MODEL` 可独立降配）

## ✅ 架构改造：从"银行可信/精确"到"简历面试宽松推理"（2026-08-02，后端 288 + 前端 44 全绿 + 端到端实测通过）

**需求**：系统是简历问答（面试官↔系统），不是金融场景——删除表格取数/加减计算/选择题判断/grounding 强制校验等银行机制；LLM 可适度修饰；证据不足不拒答，放宽检索让 LLM 推理并强制"根据现有知识库推测"标注；完全无关问题礼貌转移；必须有技术亮点。分支 `refactor/lenient-rag`。

**删除（P1）**：7 个银行机制服务文件（spreadsheet_retrieval/spreadsheet_parser/spreadsheet_cell_index/formula_parser/grounding_validation/material_semantic_grounding/retrieval_metadata_filter）+ SpreadsheetCell 模型；确定性答案路径（`_deterministic_table_answer`/`_formula_answer`/`_choice_recovery`/`_mixed_table_answer`/`_calculation_trace_valid` 等）、grounding 校验链（6 类校验/语义框架核验/修复/二次生成/摘录降级）、planner 表格/MCQ 路由、元数据硬过滤与零命中禁回退、7 阶段进度（改 5 阶段）。代码规模：answer_generation 3017→~480、rag_service 4849→~3100、planner 1915→~870。测试 401→218（裁剪后绿）。

**新增（P2/P3，四大技术亮点）**：
1. **置信度分级回答 + 答案自评（Self-RAG）**：单次 LLM 生成 `{answer, evidence_sufficiency, reason}`；sufficient→直接答、partial/insufficient→强制"根据现有知识库推测"前缀（后端兜底补）、greeting/off_topic→礼貌转移模板（零 LLM）。`answer_mode/evidence_sufficiency` 随 API 透出，前端显示"基于知识库推测"徽标。
2. **面试官意图路由 + 查询改写**：`intent_router_service.py` 规则（零成本词表）+ LLM 兜底分类 7 类意图（self_intro/project_deep_dive/tech_quiz/hr_quality/resume_detail/greeting/off_topic）→ RetrievalStrategy 映射（锚定文档/改写/跳过检索）；`query_planner.rewrite_search_queries` LLM 改写口语问题为检索查询（失败回退词表+锚点）。
3. **多轮追问记忆**：`conversation_memory_service.py` 前端 localStorage session_id + SQLite conversation_turns（进程内 LRU、24h TTL）；规则预检指代词→LLM 指代消解（"为什么不用数据库锁？"→"为什么不用数据库锁来防止超卖？"）；relevant=false 放弃消解。
4. **检索兜底链**：`_evidence_grade` 评估（none/weak/strong）→ level1 降阈补选（relaxed 选择按 rerank 补足 6 块）→ level2 查询改写重检索 → level3 空上下文直接生成（强制推测标注）；`retrieval_fallback_level` 透出。

**适配**：schemas/qa.py 新契约（去 citations/refused/claims/grounding_validation，增 answer_mode/evidence_sufficiency/session_id/5 阶段）；prompt_builder 宽松规则（可适度展开、硬事实不编造、推测必标注、无 [1] 引用）；身份快通道保留（QAResponse 适配）；身份台账 16 字段→7 字段（P4）；引用标注完全去掉（前端 EvidencePanel 删除，AnswerResult 加 answer_mode 徽标）；xls/xlsx/csv 从支持扩展名移除。

**工程**：DB 迁移（qa_tasks.session_id 列、DROP spreadsheet_cells、conversation_turns 建表）；INDEX_VERSION→v5 触发全量重索引（rebuild_vector_index.py 修复 ORM 删除+插入 flush 撞唯一约束的 bug）；.env 阈值放宽（RERANK_PROMPT_THRESHOLD/MIN_CORE_RERANK_SCORE=0.20，新增 FALLBACK_*/HEDGE_PREFIX/CONVERSATION_MEMORY_* 配置）；QDRANT_URL 默认从 RESUME_QDRANT_HTTP_PORT 推导（修复直跑连错旧栈 qdrant 的隐患）；README/PROGRESS/docs-guide（tech-highlights.md、interview_question_set.md）更新。

**验证**：后端 288 全绿（新增意图路由 31、会话记忆 23、置信度分级 12、兜底链 7）；前端 44 全绿 + tsc + build；端到端实测 14 问（自我介绍/优缺点/项目深挖/选型/简历细节/HR/无关×3/寒暄/追问链×3）：意图分类全对、礼貌转移全对、追问消解生效、回答口语化无引用标注、弱证据走兜底链。

**遗留**：① 规则词表边界（"介绍一下你的秒杀项目"命中 self_intro 规则——回答正确仅标签略偏，可优化"介绍一下你"后接宾语判定）；② identity 快通道与宽松哲学可进一步评估并入 resume_detail 意图（P5 未做，影响小）；③ 前端 QuestionHistoryDrawer 死标签已清理；④ api/qa.py terminal_statuses 仍含 "refused"（无害超集）。

## ✅ 全面审查修复（2026-08-01，后端 401 + 前端 44 全绿 + lint/tsc/build 通过）

**需求**：上线前全面审查（问答质量、安全、残留、健壮性）并按计划修复。实测确认 3 处信息泄露后逐一修复。

**安全（实测确认后修复）**：
1. **匿名泄露内部上下文（3 处）**：`/api/qa/retrieve` 匿名返回完整 llm_prompt+检索全文 → 仅管理员可用（匿名 403）；`/api/qa/ask/stream` final 事件携带完整 context_package → 匿名剥离；`/api/qa/ask` 匿名带 `include_debug=true` 可绕过剥离 → 匿名一律剥离（调试参数只对管理员生效）。统一复用 `security.is_admin_request`（新增公共函数，替换 qa.py 私有 `_is_admin_request`）
2. **匿名问答历史隐私**：按用户决策，问答历史（问题+答案全文）仅管理员可见——`GET /api/qa/tasks` 匿名返回空列表、`/api/audit/qa-logs` 匿名返回空；前端匿名隐藏"问答历史"按钮、操作日志导航、日志页显示"仅管理员可见"提示（后端为安全边界，前端隐藏只是体验）
3. **部署配置**：`.env` 弱密码 `Wbz_123456` → 强随机密码 `hxbMHxbw0kG4luZDHbg`；`ADMIN_JWT_SECRET` 独立随机 64hex（不再依赖密码派生）；`AUTH_REQUIRED=false→true`；`RATE_LIMIT_TRUST_PROXY=false→true`（上线前建议再生成一份新值）

**银行残留（活跃代码）**：
4. `answer_generation_service._relevant_extractive_excerpt` 精算加分（折现率曲线/现金流现值）→ 简历时间/列举语义；`_deterministic_table_output_unit` 保单/户数 → 人/件/项/门（保留"件数"——项目交付件数在简历场景有效）
5. `document_identity_query_service` 银行身份台账（现行有效/已废止/法律效力）→ 简历"材料真实性/有效期"语义（找到材料→说明颁发机构/时间/有效期，明确系统不鉴定真伪，真伪以官方渠道核验为准）
6. `retrieval_service.question_terms` 监管报送词（表述口径/合并判断/优先检查/四舍五入/重复汇总等 18 个）→ 简历领域词（毕业时间/绩点/实习经历/竞赛/软考/CET 等）
7. `query_planner_service` 行标签"全国合计"与省份枚举（银行区域统计）→ 简历技能/课程枚举；`_question_prefers_child_section` 银行词→简历追问词；`_missing_aspect_note` 外汇/资产差异分支删除
8. 前端 `DocumentIdentityCard` 监管 UI（发文机关/文号/版本状态"已废止/已被替代"/法律效力声明）→ 简历"材料信息卡"（颁发机构/证书编号/颁发日期/失效日期/内容主题），删除版本状态/替代关系/业务领域/生效日期；免责声明改为"不对证书真伪作出鉴定"
9. 测试残留：test_query_planner_service 5 处"全国合计/北京/天津/上海"问句、test_retrieval_service 3 处银行素材（普惠小微/绿色信贷/逾期不良）全部简历化

**健壮性**：
10. `answer_generation_service:317` 顶层异常全吞（无日志）→ `logger.warning` 记录失败原因再降级
11. `qa_task_service` 每个 progress 事件一次 SQLite commit → 0.25s 节流（失败事件必落库）
12. 取消检查扩展：非流式重试路径、语义校验器（`validate_material_semantics→_verify_risky_claims`）、LLM planner（`plan_query`）均透传 `cancellation_checker`（此前仅流式读取检查，用户取消后仍跑完 15-30s）
13. 前端：AuditPage/RagPage 2 个 lint warning 修复；`reloadTask` 加 catch（避免 unhandled rejection）；copyAnswer 定时器卸载清理；拒答时展示 `refusal_reason` 中文说明（拒答原因可读化）

**死代码清理**：
14. 后端 rag_service 删除 172 行零引用死代码：`QuestionAspect` 体系（`_expected_aspects`/`_match_chunk_to_aspects`/`_covered_aspect_ids`，含外汇残留载体）、`_mcq_reliability_question`（+测试用例）、`_corpus_lexical_support_matches`；坑：删除时误删 AspectRetrieval 的 `@dataclass` 装饰器（测试 TypeError 暴露，已修复）
15. 前端：删除 CitationList/EmptyState/ExpandableText 死组件、qa.ts 的 askQuestion/retrieveQuestionContext/askQuestionStream 死 API（其中 askQuestion 硬编码 include_debug=true 是泄露隐患）、system.ts getHealth、2 张 regumate 品牌图片、styles.css 约 200 行死样式（dashboard/workflow/capability/diagnostic/trust/readiness/evidence-rail 等，逐类验证 TSX 引用后按花括号配平删除；补回误删的 `.evidence-panel__chevron` 基础规则）
16. 保留项：旧检索链 `retrieve_citations` 等（生产不可达、仅测试 monkeypatch 触发）——连带删除大量测试，风险收益比不划算，暂留

**脚本/配置/文档**：
17. `check_offline_models.py` 仍检查旧模型名（bge-m3/bge-reranker-v2-m3）→ bge-base-zh-v1.5/bge-reranker-base（部署后误报"模型缺失"的 bug，已实测通过）
18. `.gitignore` 补 `data/app.db-wal`/`data/app.db-shm`（SQLite WAL 会提交进仓库）；删除孤儿文件 `.qdrant-initialized`、无关残留 `.github/modernize/java-upgrade/`
19. 品牌残留 7 处 ResumeMate → ResumeMind（scripts 两文件 + stop.sh）；main.py "docker-run.bat" 文案 → run.sh
20. README `MIN_RERANK_SCORE` 默认值 0.30 → 0.004（bge-base 分数分布）；config.py 硬编码 `D:\AI-Cache` Windows 个人路径删除（统一 DATA_DIR）
21. index.html 补 meta description / OG 标签 / favicon.svg（复用 BrandMark 渐变蓝紫 Logo，分享到微信/浏览器有预览卡片）

**测试**：后端 401 全绿（新增 4：匿名任务列表为空、retrieve 403、ask include_debug 不泄露、qa-logs 匿名空+管理员可见）；前端 44 全绿 + lint 0 warning + tsc + build 通过

**实测**（本地 docker 重建后）：匿名 /api/qa/retrieve → 403；匿名 /api/qa/ask/stream → context_package=None；匿名 /api/qa/tasks → []；管理员均正常

## ✅ 银行场景内容全面清理（2026-08-01 完成，后端 398 + 前端 44 全绿）

**需求**：系统内所有银行/监管场景内容全部合理替换为简历场景内容。保留项：`docs/项目介绍_ReguMate.md`、`docs/竞赛奖项.md` 等知识库素材（简历内容，面试官会问）。

**已替换（服务/测试/前端/根目录，残留 0）**：
1. **标识符全局替换**：`REGUMATE_*`→`RESUME_*`（.env/compose/Dockerfile/run.sh/config 同步）；`regumate`→`resumemind`（线程名/临时目录/key/集合名/uuid 命名空间/容器名）；`regulatory_topic`→`material_topic`（DB 列+payload+过滤链+前端 TS 类型）；`regulatory_basis`→`material_basis`（claim role+prompt 枚举）；`validate_regulatory_semantics`→`validate_material_semantics`；`regulatory_semantic_grounding_service.py` 改名 `material_semantic_grounding_service.py`
2. **场景概念**：制度/制度侧/报表侧→材料/材料侧/表格侧；制度依据/制度原文→材料依据/材料原文；填报口径/报送要求→表述口径/表述要求；监管主题→材料主题；发文机关/文号→颁发机构/编号；法律效力→真实性/有效性；合规/违规→相符/不符；商业银行主要指标分机构类表/保险业经营情况表→技能掌握情况表；人身险表→笔试成绩表、财产险表→机试成绩表；原保险保费收入→课程成绩/综合成绩
3. **分类与名单**：监管主题规则→竞赛奖项/荣誉奖励/证书资格/教育背景/技能专长/求职意向/项目经历；金融机构名单→河南大学/教育部/人社部/软考办/蓝桥杯组委会
4. **语义校验**：SUBJECT/ACTION 模式→简历主体（本人/项目/团队）+ 简历动词（负责/参与/完成/开发/荣获…）；LLM 提示词"监管语义证据校验器"→"简历材料语义证据校验器"
5. **表格/检索**：商业银行表识别→技能/成绩表族；保险指标→掌握程度/成绩排名等；`_expanded_domain_terms`、`question_terms` 领域词表→简历术语（综合成绩/技能/证书/获奖…）；预热文案"监管制度口径预热"→"简历材料问答预热"
6. **测试**：10 个测试文件全部改写（3 子代理并行 + 手工），银行问句→简历问句、断言同步；函数名含 regulation/insurance/policy/compliance 的 16 个改名
7. **前端**：localStorage key `resumemind.qa.*`、组件 id、CSS 注释、测试 mock 文件名；dist 已重建
8. **根目录**：README 来源叙述改为"个人竞赛项目 ReguMate-Agent 改造，详见知识库《项目介绍_ReguMate.md》"；.gitignore/.dockerignore 忽略规则统一

**坑**：`rag_service._expected_aspects` 与 `retrieval_service.retrieval_queries` 硬编码的"资产合计/外币折算"银行词（子代理 C 发现）→ 统一为"综合成绩/分项加总"；`test_retrieval_service` 两个用例（不在初次扫描词表内）漏改，改问句后需 `question_terms` 词表同步含"综合成绩"才能命中 direct_evidence。

## ✅ 跑题问题相关性门槛与礼貌拒答（2026-08-01，测试通过）

**问题**：问"你爱弹吉他吗？"（简历范围外）时系统答"为保证准确性，以下内容直接摘自知识库原文…"+ 奖学金摘录。实测记录 rerank 最高分仅 0.012（绝对阈值 0.45）。

**根因**：core 选择阶段（`_select_prompt_chunks` 最佳候选路径）无条件选入候选，绕过 `RERANK_PROMPT_THRESHOLD`（仅 generic 补选阶段生效）→ 无关 chunk 进 Prompt → LLM 校验失败两次 → 走"原文摘录"兜底而非拒答。

**修复**：
1. `rag_service._passes_core_relevance`（共享门槛 `_passes_prompt_relevance_bar`）：core/query/generic 三个无绝对门槛的选择路径统一要求 rerank ≥ `max(RERANK_PROMPT_THRESHOLD, MIN_CORE_RERANK_SCORE=0.1)`，或与问题有词法重叠（`_aspect_lexical_score > 0`）才入选；不达标则 covered=False + 记录 `relevance_gate` 诊断 → `has_sufficient_context=False` → 拒答
2. `answer_generation_service` insufficient_context 拒答文案改为礼貌引导（"抱歉，这个问题在我的简历内容中未找到足够依据…可以改问教育背景/项目经历/专业技能/荣誉奖项/求职意向"）
3. 回归测试 ×2：`test_qa_refuses_off_topic_question_with_zero_relevance_candidates`（默认配置）与 `test_qa_refuses_off_topic_question_even_with_deployed_loose_thresholds`（模拟 .env 宽松阈值 0.01/0）

**部署配置适配（重要）**：实测 rerank 分数为 0-1 分布（跑题≈0.012、真实匹配≥0.48），而 .env 按旧分布把 `RERANK_PROMPT_THRESHOLD` 调到 0.01、`MIN_EVIDENCE_COVERAGE=0`（恒真）——门槛在部署配置下会失效。故新增 `MIN_CORE_RERANK_SCORE=0.1`（env 可调）作下限，并去掉恒真的 coverage 逃逸（中文宽泛问题词汇覆盖不可靠，依赖 rerank 语义排序）。建议后续复核 .env 中 RERANK_PROMPT_THRESHOLD/MIN_RERANK_SCORE/STRONG_SEMANTIC_RERANK_SCORE 是否按真实 0-1 分布重新校准。

**测试**：全套件 399/399 全绿（基线 397 + 2 个新回归用例）；带真实 .env 部署配置端到端验证通过（跑题问题 → 礼貌拒答）

## ✅ 问答质量修复与管理员依据查看（2026-08-01，实测通过）

**问题**："你参与过哪些项目"等回答降级为"可核验摘录"；"介绍你自己"答成系统；管理员看不到检索依据。

**根因与修复**（4 个）：
1. **DeepSeek 推理模式未禁用（最隐蔽、影响最大）**：deepseek-v4-flash 默认思考，流式响应先输出 `reasoning_content`（content 为 null），思考长时 content 迟迟不出现 → 被误判"空响应" → 降级。修复 `llm_client.build_chat_payload`：对 DeepSeek 服务显式发 `thinking:{type:disabled}`（include_thinking=false 时）；原逻辑发反了。**修复后"你参与过哪些项目"3/3 稳定完整回答**
2. **query planner 监管残留**：planner 提示词仍按"制度/填报"风格拆查询（如"参与项目记录 填报说明 支持材料"），导致项目介绍文档检索不到。已改写为简历问答场景提示词 + 简历场景示例
3. **README.md 混入知识库**：上传脚本没排除根 README（系统自述），"介绍你自己"检索命中它 → 代词混淆。已删库中 README + 脚本排除根 README
4. **取消检查每流式行查库**：`cancellation_checker` 每行 DB 往返拖慢生成，已节流（25 行/0.5s）

**管理员查看检索依据**：`context_package`（检索 chunks 全文）随答案持久化（rag_service 总是带）；API 层按鉴权剥离——匿名任务接口/SSE 返回 None，带管理员 token 的 GET `/api/qa/tasks/{id}` 返回完整依据；前端 RagPage 管理员完成问答后自动拉取并展示"本次回答的检索依据"折叠面板（EvidencePanel）。测试：test_auth 新增依据剥离用例（后端 397 全绿）

**其他**：降级回答文案改为用户友好（"为保证准确性，以下内容直接摘自知识库原文…"）；摘录"- - "双重破折号修复；`.env` 放宽 LLM 超时（ANSWER_GENERATION_TIMEOUT_SECONDS=30 / planner 25 / grounding 15）

**实测**（任务接口=前端路径，真实 DeepSeek）：介绍你自己✓、你参与过哪些项目✓(3/3)、荣誉奖项✓、技术栈✓、教育背景✓、求职意向✓——全部 llm_grounded、校验通过、引用完整；匿名依据剥离✓

## ✅ 前台/后台权限分离（2026-08-01 完成，本地验证通过）

**需求**：上线后访客（面试官）只能智能问答 + 问答日志；知识库管理/系统状态/完整日志仅管理员。

**已实现**：
1. **认证**：单管理员 JWT（PyJWT 2.10.1），密码 `.env` 的 `ADMIN_PASSWORD`（缺失则 fail-closed），token 12h 有效期，无用户表
2. **权限矩阵**：`/api/documents/**`、`/api/health/rag|ready|warmup`、`/api/audit/logs|archives*`、`/api/auth/me` → admin；`/api/qa/**`、`/api/health`、新增 `/api/qa/status`（轻量就绪）与 `/api/audit/qa-logs`（仅问答类日志）→ 公开
3. **限流**：`backend/app/middleware/rate_limit.py` 纯 ASGI 中间件——问答每 IP 30 次/分 + 500 次/日，登录 10 次/分，全局并发 4（2C4G），SSE 豁免，429 带 Retry-After + X-Request-ID
4. **前端**：AuthProvider + LoginPage（/login）+ 路由守卫；匿名只见问答/日志导航 + "登录"入口；登录后见知识库/系统状态/退出；RagPage 改用公开就绪状态（readinessContext）；AuditPage 匿名只读 qa-logs
5. **中间件顺序坑**：Starlette add_middleware 后注册者在最外层——限流须注册在 CORS 内层、request-id 外层（详见 main.py 注释），否则 429 无 CORS 头导致浏览器读不到
6. **docker-compose**：healthcheck 从 `/api/health/ready` 降级为公开 `/api/health`（容器 healthy ≠ 模型就绪，就绪看后台系统状态面板）；新变量全部透传
7. **测试**：后端 393 全绿（新增 test_auth.py 13 个：401/200、token 过期/篡改、qa-logs 过滤、IP 限流、并发上限）；前端 38 全绿（AppShell/AuditPage/systemStatus 适配 AuthProvider + 匿名用例）；`npm run build` 通过
8. **文档**：README 新增"前台/后台与安全"章节；部署指南新增 .env 认证配置、安全加固清单、warmup 需登录说明；upload 脚本需 `--admin-password`/环境变量登录

**上线注意**：`.env` 的 `AUTH_REQUIRED` 当前为 `false`（本地开发），**部署前置 `true`**；Cloudflare Tunnel 后置 `RATE_LIMIT_TRUST_PROXY=true`；建议单独 DeepSeek key + 消费上限。

## ✅ 系统已完成并本地验证通过（基础改造）

### 改造内容
1. **代码复制**：ReguMate → `d:\Agent-Project\Resume-Agent`（原项目保持只读）
2. **模型层**：bge-m3 → bge-base-zh-v1.5（768 维纯 dense）；reranker v2-m3 → bge-reranker-base；FlagEmbedding.FlagModel/FlagReranker
3. **提示词**：ResumeMate 个人简历问答场景（prompt_builder/query_planner/audit 全改）
4. **品牌**：ResumeMate · 简历问答助手（前端/后端/脚本全改）
5. **清理**：竞赛主题内容（数据/脚本/测试/GPU/Windows 开发脚本）全部删除
6. **部署导向**：纯 CPU Dockerfile、docker-compose 默认值、.env 最终值、run.sh/stop.sh

### 关键调参（bge-reranker-base 分数分布适配）
| 参数 | 值 | 原因 |
|---|---|---|
| RERANK_MAX_LENGTH | 512 | FlagEmbedding 1.4.0 在 768/1024 有编码 bug（IndexError） |
| MIN_RERANK_SCORE | 0.004 | base 的 normalize 分数分布 0.001-0.03（v2-m3 是 0-1） |
| RERANK_PROMPT_THRESHOLD | 0.01 | 同上 |
| MIN_EVIDENCE_COVERAGE | 0 | 中文宽泛问题的词汇覆盖不可靠，依赖 rerank 语义排序 |
| STRONG_SEMANTIC_RERANK_SCORE | 0.02 | 强语义匹配阈值（原 0.55 是 v2-m3 分布） |

### 本地验证结果（docker，端口 18000/16333）
- ✅ 镜像 970MB；health 返回 ResumeMate
- ✅ 模型下载（download_models.py 从 hf-mirror 直下，绕过 huggingface_hub Xet 兼容问题）→ 离线模式
- ✅ 知识库 10 文档上传+索引（软考证书扫描件 PDF 无法解析属预期，有说明 md 替代）
- ✅ 问答验证：项目经历/外卖平台职责/荣誉/技能 4 问全部正确回答+引用
- ✅ 前端页面 200（标题 ResumeMate · 简历问答助手）
- ✅ pytest 380 通过、前端测试 34 通过

### 部署脚本（scripts/ 只剩 5 个）
- `upload_knowledge_base.py`：知识库一键上传（扫描 docs/ → 上传 → 等索引）
  - **内容变更自动更新**：本地 manifest（.run-state/kb_manifest.json）记录 filename→sha256，文件修改后重跑脚本自动"覆盖上传+重新索引"
  - `--delete "文件名"` 按名删除；`--purge` 清空知识库（重建用）
  - 同名 README.md 自动加父目录前缀（README_ReguMate.md）
  - 端口探测读 RESUME_QDRANT_HTTP_PORT 环境变量
- `download_models.py`：模型直下脚本（服务器上同样可用）
- `check_offline_models.py`、`rebuild_vector_index.py`、`scan_secrets.py`

### 知识库维护流程（已验证）
| 操作 | 命令 |
|---|---|
| 修改简历/项目文档后更新 | 直接重跑 `upload_knowledge_base.py`（自动检测变化→覆盖→重新索引） |
| 删除单个文档 | `--delete "文件名.md"` |
| 清空重建 | `--purge` 后重跑上传 |
| 注意 | 图片无 OCR；扫描件 PDF 无法解析，用文字 md 替代 |

### docs/ 内容体系（2026-08-01 重构，用户拍板）
- **全部平铺在 docs/ 根**：竞赛奖项.md、个人荣誉.md、证书说明.md、教育背景.md、技能专长.md、求职意向.md、项目介绍_*.md（5 个）+ 简历 PDF
- **规则**：证书/奖状图片不放（无 OCR），写成文字 md；项目不放源码放介绍文字
- **源码已移至** `d:/Agent-Project/Archive-项目源码/`（EchoMind/REV/外卖/秒杀/ReguMate 原项目；用户确认后可直接删该目录）
- **EchoMind 改名 XDU EchoGuide**（2026-08-01）：docs/项目介绍_EchoGuide.md（标题"XDU EchoGuide：基于多 Agent 与 RAG 的西电校园智慧助手"，注明原名）；Archive 源码目录已改名；代码内部 12+ 文件仍有 echomind 字符串（容器名/配置），待用户需要时再改
- 上传脚本已移除 ReguMate 特殊逻辑（源码移走不再需要）

### 改名 ResumeMind + 品牌 Logo（2026-08-01 完成）
- 全局改名：ResumeMate → **ResumeMind**（前后端 19 个文件：config/security/prompt_builder/compose/.env/README/部署指南等）
- QDRANT_COLLECTION → resumemind_chunks（知识库已重建）
- **BrandMark SVG Logo**：渐变蓝紫圆角方块 + 简历文档 + 对话气泡（新组件 frontend/src/components/BrandMark.tsx，替换 Sparkles）
- 管理员认证配置已补：.env ADMIN_PASSWORD（生成值 RM-SA6RLp2lLpIj）+ AUTH_REQUIRED=false（后端 security.py 由前后台分离改造引入，fail-closed；前端后台页面完成后置 true）
- 验证：前端 35 测试、后端 380 测试、docker 重建（resumemind-app）、页面 title/问答正常

### 问答页"去可信化"简化（2026-08-01 完成）
- **RagPage 重构**：删 TaskProgress/processing-note/StreamingAnswerPreview/TechnicalDetails/证据案卷侧栏 → 处理中显示"思考中..."三点跳动动画（ThinkingIndicator）；回答单栏纯文本
- **AnswerResult 简化**：删可信状态标签/核查结果/[1] 引用标注（stripInlineCitationLabels 段落感知清洗）；拒答只显示一句提示
- **QuestionComposer**：删"返回检索技术详情"设置项
- **删除组件**：TaskProgress.tsx、StreamingAnswerPreview.tsx（含测试）
- **CSS**：新增 .thinking-dots 跳动动画（prefers-reduced-motion 降级）
- **文案**：全局"可信"→"智能问答"（SystemStatusPanel/AuditPage）
- 后端零改动（前端不传 includeDebug，context_package 自动为 null）
- 验证：前端 34 测试通过、后端 380 通过、docker 重建页面 200

## 下一步（云服务器部署）
1. 买 2C4G 服务器（建议香港免备案）+ Docker Engine + 2GB swap
2. 上传代码 → `.env` 填 DeepSeek key → `RESUME_APP_PORT=8000`/`QDRANT_HTTP_PORT=6333`（删本地端口覆盖）
3. `docker compose up -d --build`
4. `python scripts/download_models.py`（或服务器直接下载）→ 离线模式
5. `python scripts/upload_knowledge_base.py` 上传知识库
6. 公网：安全组 / Cloudflare Tunnel（详见 docs-guide/部署指南-2C4G-CPU.md）

## 注意事项
- `.env` 本地端口覆盖 18000/16333（为避免与旧 resumemind 栈冲突）；**服务器部署时删除这两行**
- 旧 regumate 容器（D:\Agent-Project\ReguMate Agent）已禁用自启，但用户本地还有旧栈可能占用 8000/6333（可选清理）
- 简历 PDF 解析有文本重复问题（"河南大学奖学金"重复出现），后续可优化简历 PDF 质量
- 知识库素材：docs/ 下有 简历/蓝桥杯说明/软考说明/5 份项目介绍（文件名唯一）

## ✅ 全面工程修复：安全/配置一致性/历史遗留清理/fast path/grounding/评测（2026-08-08，后端 314 + 前端 35 全绿）

**需求**：面向 2C4G 部署，解决安全、配置/文档不一致、ReguMate/监管历史遗留、RAG 调用链延迟、grounding 可靠性、评测体系与工程清理 10 项问题。

**P1 安全与配置**
- 创建 `.env.example`（脱敏模板：所有密钥置空 + 中文注释），`.gitignore` 移除 `.env.example` 忽略（模板应入库）；确认 `.env` 从未被 git 跟踪（index 无记录）
- 清理 `.env`/`docker-compose.yml` 已删除功能的残留配置：`SEMANTIC_GROUNDING_*`（grounding 校验链 2026-08-02 已删）、`RESUME_ALLOW_UNREADY`、`MAX_SPREADSHEET_LOGICAL_CELLS`；保留 `MAX_OOXML_*`（docx zip 防护活功能）
- README/tech-highlights 修正：测试数 256→314/前端 44→35；`RESUME_OFFLINE_MODE` 默认 true；`MIN_CORE_RERANK_SCORE` 0.20；删除死配置 `MIN_RERANK_SCORE`/`STRONG_SEMANTIC_RERANK_SCORE`（config 零消费）；README 增交付检查清单；tech-highlights 更新 fast path/grounding/hybrid 真实描述

**P1 历史遗留清理（ReguMate/监管→简历）**
- 删死服务：`loader_evaluation.py`、`qdrant_admin_service.py`（仅测试引用）、`question_preprocessing_service.py`（MCQ 选项提取器，主链不用）
- 删死函数：`parse_document_text`/`CsvDocumentLoader`/`parse_document_with_loader`、`build_chunks`/`_split_*`、`save_original_document`（非 stream）/xls/csv 分支、`convert_with_libreoffice`（非 detailed）、`_chunk_type` GBK 乱码表格前缀
- vector_store payload 删 25 个无数据流银行/表格字段（sheet_name/year/month/quarter/table_id/business_domain/version_status/article_number 等）+ 索引/过滤/`_period_value`；**保留**简历化活字段（issuing_authority 颁发机构/document_number 证书编号/material_topic 内容主题/external_doc_id/source_url 等）
- Document 模型/DB/schemas 删 `business_domain/source_column/effective_date/version_status/supersedes_document_id/version_label` 列与 `VersionStatus` 类型（repealed/superseded/draft）；`_query_anchor_boost` 删监管文号正则；rerank compact 删表格标签；url-import 删 xls/csv；requirements 删 openpyxl/xlrd
- 前端：删 `HealthResponse`/`LLMContextPackage` 银行字段/`Citation.version_status`/`listQaAuditLogs`/`TableChunkMeta`/表格证据渲染/表格角色；styles.css 删 658 行死样式；测试素材（资产合计/报表.xlsx/NFRA/bge-m3）全部简历化

**P2-A fast path（普通问题 LLM 3→1 次）**
- 意图层确定性旁路（`INTENT_FAST_PATH_ENABLED`）：无上一轮时整句问候词表→greeting、简历锚点→resume_qa，跳过 LLM；有上一轮仍走 LLM 消解
- 规划层确定性旁路（`PLANNER_FAST_PATH_ENABLED`）：非枚举/非补集/无复合结构问题直接 `_fallback_aspects` 单/少方面，跳过 LLM；枚举/补集/复杂问题保留完整链路
- 效果：普通独立问题 intent(1)+planner(1)+generation(1)=3 → 1 次 LLM；追问/枚举/复杂维持原链路；两开关可关回退

**P2-B grounding 确定性硬事实校验（`GROUNDING_VERIFY_ENABLED`）**
- 新 `grounding_verification_service.py`（零 LLM 纯规则）：从答案抽数字/年份/日期/书名号专名，与检索证据归一化匹配；LLM 自评 sufficient 但硬事实缺失 → 强制 hedged + 记录缺失项到 `extra`；`llm_call_count` 随 QAResponse 透出供评测

**P2-C Resume 领域检索**
- `_retrieve_aspect_matches`：aspect 锚定的对象文档名（anchor_documents）解析为 document_id 集合，作为 Qdrant metadata_filter 传入——枚举对象不依赖 embedding 相似度；解析失败退化为全库检索（安全）

**P2-D hybrid 命名澄清**
- `hybrid_search` 注释说明真实实现：dense 单路 + 应用层关键词补充 + rerank（无 Qdrant sparse），函数名保留避免 churn；README/tech-highlights 同步

**P2-E 评测体系**
- `eval_interview_set.py` 扩展为评测框架：结构化问题集（期望事实/禁止事实）、延迟 p50/p95、单问 LLM 调用次数、期望事实命中率、禁止内容检测、JSON 报告；`docs-guide/evaluation.md` 使用说明

**验证**：后端 314 全绿（新增 fast path 9 + grounding 8 + anchor 2）；前端 35 全绿 + build 通过；复扫 `regumate/资产合计/报表.xlsx/NFRA/SEMANTIC_GROUNDING/spreadsheet` 零残留（docs/ 知识库素材与 PROGRESS 历史记录除外）

**遗留**：① 评测脚本需真实 DeepSeek + 本地模型（单测全 mock 不覆盖 E2E）；② rag_service 2690 行未拆分（职责混杂确认存在，但测试耦合深、收益低，仅清理死代码）；③ fast path 效果需在真实 2C4G 上用 eval 脚本验证延迟/LLM 次数；④ `.env` 含真实密钥仅本机，交付前必须用 `.env.example` 重建

## ✅ 二轮工程修复：Fast Path 上下文判断/complement 硬排除/material_topic 检索/实体 grounding/真实 LLM 计数/评测修复（2026-08-08，后端 357 + 前端 35 全绿）

**需求**：解决 8 项代码逻辑问题——Fast Path 有历史即失效、complement 排除不生效、material_topic 未参与检索、grounding 实体校验缺失、LLM 计数靠推测、评测 forbidden 统计 bug、测试 monkeypatch 隔离、历史残留。

**1. Fast Path 上下文判断（intent/planner）**
- 新增 `context_dependency.py`：`needs_context_resolution(question)` 判断问题是否真依赖上一轮（指代代词/序数/省略主语/衔接词）；`is_greeting_phrase` 只认整句问候
- 核心修复：intent/planner fast path 从"无历史才走"改为"问题自足才走"——有历史但问题独立（Q1 项目 → Q2 HashMap）仍走 fast path 跳过 LLM；追问（"第二个项目""那 Redis 呢"）才走 LLM 消解
- 修 `len<=2` 问候误判："爱好/籍贯/薪资/项目"等短词不再判 greeting（交 LLM）；锚点表单一事实源移到 context_dependency

**2. complement/enumeration 硬排除**
- 修复 `_doc_key` 返回 document_id 但 excluded_documents/anchor_documents 存 filename 的身份混用 → 统一 filename
- `_select` 与阶段 3 兜底都加硬排除：被排除文档的 chunk 绝不进最终 prompt（即使 coverage 逻辑倾向选它）
- 测试验证**最终 prompt**（非 QueryPlan）：`test_complement_question_excludes_documents_from_final_prompt`、`test_complement_exclusion_survives_force_min_chunks`

**3. material_topic 驱动枚举检索**
- catalog 从 `(filename, title)` 扩展为 `(filename, title, material_topic)`（动态读 Document 表）
- `classify_object_docs` 优先按 material_topic 匹配对象文档（"你有哪些项目"→ 项目经历 文档集合），文件名 pattern 仅作旧数据 fallback；`CATEGORY_TO_MATERIAL_TOPIC` 对齐内部类别与简历领域值

**4. Grounding Verifier 增强**
- 列表编号误杀修复：`1. EchoGuide 2. ReguMate` 的编号不作为数字事实
- 实体校验：`known_entities`（知识库 catalog 提取的学校/项目/奖项/证书/技术栈）——答案出现但证据缺失的已知实体 → hedged
- normalization：全角数字/百分号转半角、日期格式统一

**5. LLM 调用计数移到请求层**
- `llm_client` 用 ContextVar 在真正 urlopen 前计数（`reset_llm_call_count`/`llm_call_count`）——统计真实 API 调用，含超时/失败后 fallback；fast path 跳过 LLM 时不计
- `rag_service.answer_question` 开头 reset、末尾读取真实计数写入 `llm_call_count`；删除基于 pipeline 状态的推测函数

**6. E2E Evaluation 修复**
- forbidden 变量覆盖 bug：形参 `forbidden` 被命中列表覆盖导致 `0/0` 统计 → 分开命名，分母=期望检查数
- expected fact 缺失超半数 → 记为失败（不再静默 exit 0）；新增 expected_documents 检索命中统计（Recall@文档）
- 支持 `--data` JSONL 评测数据（group/question/expected_facts/forbidden_facts/expected_documents/session 多轮链）；`scripts/eval_cases.jsonl` 示例

**7. 测试 monkeypatch 审计**
- 审计确认：test_qa_access/test_auth/test_rag_api 均 patch 模块级绑定（`backend.app.api.qa.answer_question`、`qa_task_service.answer_question`），与 `from ... import` 绑定一致，patch 有效
- 新增 `test_qa_api_answer_patch_isolates_real_rag`：证明 patch 后 QA API 走 fake、真实 RAG（模型链）零调用

**8. 历史残留清理**
- `requirements.in` 删 openpyxl/xlrd；Document 模型删 `effective_date`（生产零消费）
- `document_url_import_service` 删 text/csv 死分支（content_type 集合已排除 csv）
- **保留** `spreadsheet_cells DROP`：旧库（ReguMate 时代遗留表）幂等清理，模型已删、新库零开销；`document_manifest_service` 的 csv 导入是活功能（manifest 清单）

**验证**：后端 357 全绿（净增 43）；前端 35 全绿 + tsc + build；评测脚本 py_compile + JSONL 加载验证通过

**遗留**：① fast path 的 `needs_context_resolution` 规则需在真实问答观察（"为什么不用数据库锁"独立问会保守走 LLM，可接受）；② grounding 实体校验的阈值需真实校准（长实体/模糊措辞可能误报）；③ rag_service/planner 仍未拆分（用户本轮明确不做）；④ E2E 评测需真实 DeepSeek + 本地模型运行（本环境未实调，未假装通过）

## ✅ Embedding 模型降档：bge-base-zh-v1.5 → bge-small-zh-v1.5（2026-08-11）

**目标**：2C4G 部署（用户拟从 4C8G 降档）。评估结论：embedding 是唯一可降档项（base → small，391MB → 92MB）；reranker base 已是官方中文最小档，ONNX INT8 因 parity 门禁（Top-1 86.96% < 95%）维持禁用，未动。

**改动**
- `config.py`：默认 `EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5`、`DEFAULT_EMBEDDING_MODEL_DIR=bge-small-zh-v1.5`、`EMBEDDING_DIMENSION=512`；`INDEX_VERSION` 改为环境变量可覆盖，默认 `bge-small-zh-dense-v6-resume`（SQLite/Qdrant 双端按版本过滤，换模型即失效旧索引）
- `model_path_resolver.py`：hub 缓存名 → `models--BAAI--bge-small-zh-v1.5`
- `docker-compose.yml` / `Dockerfile` / `download_models.py` / `check_offline_models.py` / `.env.example` / `deploy/*.env.example`：全部换 small
- 旧 `data/models/bge-base-zh-v1.5/` 保留未删，环境变量可随时回退

**A/B 评测（23 问黄金集，真实 DeepSeek + 本地模型，报告：docs-guide/embedding模型A-B评测报告.md）**
- 三组对照控制"分块代码"混杂变量：① base+旧索引 49/51 事实 (96.1%)、19/20 文档 (95.0%)；② base+当前代码 45/46 (97.8%)、16/18 (88.9%)（2 问 API 超时未完成）；③ small+当前代码 **50/51 (98.0%)、18/20 (90.0%)**
- ② vs ③ 唯一变量=模型：事实与文档召回实质持平，small 略优；B-9 由 hedged 转 answered（改善）；E-17 漏检 `证书说明.md` 是当前分块代码引入（② 同漏），与模型无关
- 已知漏检（三组共有）：C-10 `技能专长.md`、E-17 `证书说明.md`——遗留待排查，独立于本次切换

**验证**：后端 361 单测全绿（修 1 个 hub 缓存路径夹具 + 1 个预算时区 bug，见下）；索引重建 16 篇全绿（110 chunk，512 维，SQLite/Qdrant 对账一致）；本地服务 ready 全绿

**顺带修复（预算熔断时区 bug，2026-08-12 00:10 发现）**：`_global_daily_qa_remaining` 用 `func.date(created_at)`（UTC 日期）对比 `date.today()`（本地日期），本地 00:00-08:00 窗口会漏计/多计"今日"预算。改为 `func.date(func.datetime(created_at, "localtime"))` 按本地日统计；单测 `test_budget_exhausted_returns_429` / `test_count_only_today_rows` 因此复绿

## ✅ 系统压力指示灯（绿/黄/红）+ 繁忙提示横幅（2026-08-12）

**需求**：2C4G 部署下向访客展示系统压力——压力小绿色；勉强可承受黄色；黄色时提醒当前与未来用户"人流量较多，回复可能较慢"。

**容量结论（2C4G，实测校准）**：问答任务管线单 worker 串行 + 模型推理全局锁串行 → **1 人提问流畅（绿）、2 人勉强正常（黄，正是 QA_GLOBAL_CONCURRENCY=2 的设计档）、3 人及以上排队变慢（红）**；纯浏览访客（静态页+轻量轮询）不构成压力。硬上限：任务队列 16、IP 30 次/分、全局日预算。

**后端**（全部信号来自内存/DB 快照，公开接口零额外成本）
- `load_status_service.py`（新）：`classify_load(cpu_ratio, mem_ratio, in_flight)` 纯函数分级——CPU 比≥0.70 或 in_flight≥2 → 黄；≥0.90 或 in_flight≥3 → 红；内存仅作红色兜底（防 OOM）。CPU 取最近 30s 均值（rerank 突刺不误报），归一化到核数；in_flight = running + queued（qa_task_status_counts）
- `config.py`：LOAD_YELLOW_CPU_RATIO / LOAD_RED_CPU_RATIO / LOAD_RED_MEM_RATIO / LOAD_YELLOW_INFLIGHT / LOAD_RED_INFLIGHT / LOAD_MEMORY_REFERENCE_BYTES 六项 env 可调
- `readiness_service.public_qa_status()` 增加 `load: {level, signals}`；`schemas/qa.py` 新增 `QaPublicStatusResponse`（修复 FastAPI 按返回注解校验导致 500 的问题）
- 降级：负载采集异常 → 绿，不阻塞状态接口

**前端**
- `readinessContext.tsx`：PublicQaStatus 增加 load 字段，context 暴露 `loadLevel`（30s 轮询复用，新访客挂载即见）
- `RagPage.tsx`：标题栏负载指示点（复用 .status-dot：绿 ok/黄默认/红 error）+ 文案；黄色/红色弹"当前访问人数较多，系统负载较高，回复可能较慢，请耐心等待。"横幅（可关闭；回绿后再次转黄重新提醒，红色文案提示排队较久）
- `styles.css`：.load-indicator / .load-banner（复用 --warning-soft/--danger-soft 变量）

**验证**：后端 372 单测全绿（新增 11 个 classify_load 边界用例）；前端 41 测试全绿（新增 RagPage 6 个用例）+ tsc 通过；实测：2 并发任务 → yellow、1 任务 → green、全部完成 → green

## ✅ 2C4G 容量提升：问答任务双 worker（2026-08-12，目标"2 稳 3 勉"）

**结论**：不需要 4C8G。原任务队列单 worker 串行（每问 ~90% 时间是外部 DeepSeek API，本地推理仅 1-2s 且已有全局锁串行）——双 worker 后 2 人同时提问各自并行处理，第 3 人排队等待。

**改动**
- `config.py`：新增 `QA_TASK_WORKERS`（默认 2，1=回退单 worker）；负载阈值联动调整 `LOAD_YELLOW_INFLIGHT` 2→3、`LOAD_RED_INFLIGHT` 3→4（新语义：1-2 人绿、3 人黄、4+ 红）
- `qa_task_service.start_qa_task_worker()`：启动 QA_TASK_WORKERS 个线程（带序号命名），`_STARTED`/恢复/去重逻辑不变——队列 `queue.Queue` 线程安全、`_PENDING` 有锁去重、SQLite 已是 WAL+busy_timeout、ByteLRUCache 有 RLock、模型推理由 MODEL_INFERENCE_LOCK 跨线程串行（并发安全项全部核实）
- `docker-compose.yml`/`.env.example`/`deploy/2c4g.env.example`(2)/`deploy/4c8g.env.example`(3) 加 QA_TASK_WORKERS

**实测（本地，真实 DeepSeek）**：3 任务同时提交 → 最大 running=2（并行）、in_flight=3（1 排队）；完成时刻 3.2s/4.6s（并行，对比单 worker 串行时第 2 个任务要等第 1 个）/25.8s（排队）；负载分级 3 并发时 yellow、2 并发 green；uvicorn 进程峰值 RSS 1.29GB（远低于 mem_limit 2800m，容器内预计 ~2-2.4GB，余量充足）

**验证**：后端 374 单测全绿（新增 worker 启动数量/幂等 2 用例 + 负载阈值 3/4 边界更新）；前端不受影响

**遗留**：① 容器内 RSS 以服务器后台资源面板复核；② 可选二期：重复问题回答缓存（面试高频问题可再降 30-50% LLM 调用）

## ✅ 问答答案缓存：面试高频问题相似语义复用（2026-08-12）

**需求**：用户输入问题后用向量找"相似语义"的缓存答案直接复用。核心难点：语义相似 ≠ 答案可复用（"介绍一下秒杀项目"与"秒杀怎么防超卖"语义相关但答案不同）→ 两层保守判定。

**设计**
- ① 精确匹配（最安全）：问题归一化（NFKC 全角→半角、去标点空白；**不去停用词**——"怎么/为什么/哪些/多少"是关键差异词）完全相等 → 命中
- ② 语义匹配：问题 embedding 与缓存向量余弦 top-1 ≥ `QA_CACHE_SEMANTIC_THRESHOLD=0.93`（保守，宁缺毋滥）→ 命中
- 只缓存独立问题（无 session）的 answered+sufficient 答案；hedged/redirected/failed 不写；追问链不查不写
- 存储：SQLite 表 `qa_answer_cache`（重启不丢）+ 进程内 numpy 向量（300 条上限 LRU 按 updated_at 淘汰）；缓存键隐含 INDEX_VERSION+embedding+LLM 签名，换模型自然失效
- 失效：知识库变更（上传/删除/重建索引）→ `document_indexing_service`/`document_lifecycle_service` 的 invalidate 挂点旁整体清空

**接入**：`rag_service.answer_question` 开头查缓存（命中：llm_call_count=0、发 cache 进度事件、照常写 qa_logs 审计、响应 `cached=true`）；结尾写缓存。/ask 与任务路径全覆盖，前端零改动（progress stage 加 "cache"，前端仅透传不校验）。

**验证**
- 后端 388 单测全绿（新增 14 个：归一化/精确命中/语义命中/阈值 miss/签名失效/禁用/清空/LRU/覆盖写/命中分支零 LLM+审计）
- 端到端实测（真实 DeepSeek）：第 1 问 13.5s llm_calls=2（写入缓存）→ **第 2 问同一问题 cached=true、llm_calls=0、0.0s 秒回**；同义改写未过 0.93 阈值（保守设计，精确命中已覆盖"面试官重复问同一问题"主场景）；不同问题正常全链路无误命中
- 修测试坑：Windows 文件锁（临时文件库连接池未释放）→ 测试库改内存 StaticPool；fixture clear 指向测试库不碰真实 app.db

**配置**：`QA_CACHE_ENABLED=true`、`QA_CACHE_SEMANTIC_THRESHOLD=0.93`、`QA_CACHE_MAX_ITEMS=300`
