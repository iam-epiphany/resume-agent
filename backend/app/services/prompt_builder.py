from backend.app.schemas.qa import RetrievalResult


# 宽松版回答规则：面试回答的完整性优先于逐字忠实，硬事实不编造，推测必须标注。
CORE_RAG_RULES = (
    "回答要自然、完整、口语化，像一个真实求职者在面试中回答。",
    "可以基于检索到的知识片段适度组织、补充衔接与展开说明，但数字、日期、机构名、证书名等硬事实必须来自检索片段，不得编造。",
    "不得断言知识库中不存在某信息或没有某记录；检索片段未包含所需信息时，只能表述为“检索到的材料中未包含”或“暂未检索到相关记录”，不得说“知识库中没有”。",
    "不要输出任何引用标注（如[1]或“来源”字样）。",
    "当问题要求列举项目、技能或奖项时，把检索到的全部相关材料一并列出（说明名称、时间与职责/等级）；"
    "只列检索到的内容，不得补充知识库外的项目、技能或奖项。",
    "如果问题与简历内容完全无关，礼貌说明并引导到简历话题（教育背景、项目经历、专业技能、荣誉奖项、求职意向）。",
)

# 角色与代词消歧：面试官提问中的“你/他/自己”等称呼，一律指知识库的主人公（简历所属求职者），
# 而不是 ResumeMind 系统本身。例如“介绍你自己”应输出主人公的第一人称自我介绍。
PERSONA_AND_PRONOUN_RULES = (
    "代词消歧（必须遵守）：用户问题中的“你”“您”“自己”“本人”“他”“她”“这个人”“求职者”“候选人”等称呼，"
    "一律指知识库的主人公——即简历所属的求职者本人，而不是 ResumeMind 系统。\n"
    "用户问“介绍你自己”“你是谁”“你的情况”等时，以主人公的第一人称视角回答"
    "（如“我是张三，……”，内容必须基于知识库证据）；用户以“他/她”提问时，用第三人称介绍主人公。\n"
    "不得介绍 ResumeMind 系统本身的功能或定位。\n"
)

SELF_ASSESSMENT_OUTPUT_RULES = (
    "只输出JSON对象，不要输出任何其他内容。",
    '输出结构：{"answer": "完整回答", "evidence_sufficiency": "sufficient|partial|insufficient", "reason": "一句话说明依据情况"}',
    "evidence_sufficiency 取值：sufficient=检索内容足以直接回答；partial=检索内容部分相关、需要合理推断；insufficient=几乎没有相关检索内容、主要靠推断。",
    "回答中不得自行添加“根据现有知识库推测”等前缀（前缀由系统统一处理）。",
)


class RAGPromptBuilder:
    def build(self, query: str, chunks: list[RetrievalResult]) -> str:
        chunk_blocks = "\n\n".join(self._chunk_block(index, chunk) for index, chunk in enumerate(chunks, start=1))
        if not chunk_blocks:
            chunk_blocks = "（未检索到相关材料）"
        rule_block = "\n".join(f"{index}. {rule}" for index, rule in enumerate(CORE_RAG_RULES, start=1))

        return (
            "你是 ResumeMind，一个基于个人简历、证书、荣誉与项目文档的简历问答助手。\n\n"
            f"{PERSONA_AND_PRONOUN_RULES}\n"
            "请根据【检索到的知识片段】回答用户问题。\n"
            "要求：\n"
            f"{rule_block}\n\n"
            "【用户问题】\n"
            f"{query}\n\n"
            "【检索到的知识片段】\n"
            f"{chunk_blocks}\n\n"
            "【请输出】"
        )

    def build_generation_messages(
        self,
        query: str,
        chunks: list[RetrievalResult],
        *,
        llm_prompt: str | None = None,
        correction: str | None = None,
        no_evidence: bool = False,
    ) -> list[dict[str, str]]:
        prompt = llm_prompt or self.build(query, chunks)
        structured_rule_block = "\n".join(
            f"{index}. {rule}" for index, rule in enumerate(SELF_ASSESSMENT_OUTPUT_RULES, start=1)
        )
        no_evidence_block = (
            "当前未检索到任何知识库材料。只允许基于对话上下文作答，且只能使用性格特质、"
            "兴趣爱好、求职动机等个人软性信息；时间、机构、项目名、数字、成果等硬事实"
            "必须明确说明“知识库未收录”，不得编造。"
            if no_evidence
            else ""
        )
        return [
            {
                "role": "system",
                "content": (
                    "你是 ResumeMind 简历问答助手，回答必须符合面试场景的自然口语风格。"
                    f"{PERSONA_AND_PRONOUN_RULES}"
                    + "".join(f"{index}. {rule}\n" for index, rule in enumerate(CORE_RAG_RULES, start=1))
                    + structured_rule_block
                    + (no_evidence_block if no_evidence_block else "")
                    + (f"\n上次生成存在的问题（请修正）：{correction}" if correction else "")
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    def _chunk_block(self, index: int, chunk: RetrievalResult) -> str:
        section_title = chunk.section_title or "-"
        return (
            f"【知识片段 {index}】来源：{chunk.source_doc} / {section_title}\n"
            f"{chunk.text}"
        )
