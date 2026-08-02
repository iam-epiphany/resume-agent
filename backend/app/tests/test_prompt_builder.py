"""提示词构建测试：角色与代词消歧规则必须进入最终提示词。

背景：访客问“介绍你自己/他是谁”等代词问题时，应指知识库的主人公（简历求职者），
而不是 ResumeMind 系统本身。此规则定义在 PERSONA_AND_PRONOUN_RULES。
"""

from backend.app.schemas.qa import RetrievalResult
from backend.app.services.prompt_builder import RAGPromptBuilder


def _chunk() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-1",
        rank=1,
        score=0.9,
        source_doc="简历_张三.pdf",
        section_title="基本信息",
        text="张三，河南大学本科在读。",
        citation_label="[1]",
        metadata={"rerank_score": 0.9, "document_id": "DOC-TEST"},
    )


def test_prompt_includes_pronoun_disambiguation() -> None:
    prompt = RAGPromptBuilder().build("介绍你自己", [_chunk()])

    assert "代词消歧" in prompt
    assert "知识库的主人公" in prompt
    assert "而不是 ResumeMind 系统" in prompt
    assert "第一人称" in prompt
    assert "第三人称" in prompt


def test_generation_system_message_includes_pronoun_rules() -> None:
    messages = RAGPromptBuilder().build_generation_messages("他是谁？", [_chunk()])
    system_content = messages[0]["content"]

    assert "代词消歧" in system_content
    assert "一律指知识库的主人公" in system_content
    assert "不得介绍 ResumeMind 系统本身" in system_content
