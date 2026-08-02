from backend.app.schemas.qa import RetrievalResult


# 宽松版回答规则：面试回答的完整性优先于逐字忠实，硬事实不编造，推测必须标注。
CORE_RAG_RULES = (
    "回答要自然、完整、口语化，像一个真实求职者在面试中回答。",
    "可以基于检索到的知识片段适度组织、补充衔接与展开说明，但数字、日期、机构名、证书名等硬事实必须来自检索片段，不得编造。",
    "检索到的内容不足或只能推断时，必须使用“根据现有知识库推测”的措辞，不得把推测表述为事实。",
    "不要输出任何引用标注（如[1]或“来源”字样）。",
    "知识库中的《项目介绍_*.md》文档均为简历主人公参与或主导的项目；回答“参与过哪些项目”等列举类问题时，"
    "应把检索到的项目介绍文档中的项目一并列出（说明项目名称、时间与职责）。",
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
    "当 evidence_sufficiency 为 partial 或 insufficient 时，answer 必须以“根据现有知识库推测”开头。",
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
    ) -> list[dict[str, str]]:
        prompt = llm_prompt or self.build(query, chunks)
        structured_rule_block = "\n".join(
            f"{index}. {rule}" for index, rule in enumerate(SELF_ASSESSMENT_OUTPUT_RULES, start=1)
        )
        return [
            {
                "role": "system",
                "content": (
                    "你是 ResumeMind 简历问答助手，回答必须符合面试场景的自然口语风格。"
                    f"{PERSONA_AND_PRONOUN_RULES}"
                    + "".join(f"{index}. {rule}\n" for index, rule in enumerate(CORE_RAG_RULES, start=1))
                    + structured_rule_block
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
