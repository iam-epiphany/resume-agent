from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PreprocessedQuestion:
    question: str
    options: list[str]
    extracted_options: bool = False
    option_source: str | None = None
    option_labels: list[str | None] | None = None


_CIRCLED_LABELS = "①②③④⑤⑥⑦⑧"
_CHINESE_LABELS = "一二三四五六七八"
_LABEL_PREFIX = (
    r"(?:(?<=^)|(?<=[\n\r\t\s。；;，,：:？?]))"
    r"(?P<label>"
    r"[①②③④⑤⑥⑦⑧]"
    r"|[A-Ha-h]\s*(?:[\.．、:：\)）]|\s+)"
    r"|[1-8一二三四五六七八]\s*(?:[\.．、:：\)）]|\s+)"
    r")"
)
_LABELED_OPTION_PATTERN = re.compile(_LABEL_PREFIX)
_LEADING_LABEL_PATTERN = re.compile(r"^\s*" + _LABEL_PREFIX)


def preprocess_qa_request(question: str, options: list[str] | None = None) -> PreprocessedQuestion:
    cleaned_question = _normalize_spaces(question)
    explicit_options, explicit_labels = _clean_options_with_labels(options or [])
    if explicit_options:
        return PreprocessedQuestion(
            question=cleaned_question,
            options=explicit_options,
            option_labels=explicit_labels,
        )

    extracted = extract_inline_options(cleaned_question)
    if extracted is not None:
        stem, extracted_options, source, option_labels = extracted
        return PreprocessedQuestion(
            question=stem or cleaned_question,
            options=extracted_options,
            extracted_options=True,
            option_source=source,
            option_labels=option_labels,
        )

    return PreprocessedQuestion(question=cleaned_question, options=[], option_labels=[])


def extract_inline_options(question: str) -> tuple[str, list[str], str, list[str | None]] | None:
    """Extract 2-8 multiple-choice options embedded in a user question."""

    text = str(question or "").strip()
    if not text:
        return None

    labeled = _extract_labeled_options(text)
    if labeled is not None:
        return labeled

    unlabeled = _extract_unlabeled_option_block(text)
    if unlabeled is not None:
        return unlabeled

    tabular = _extract_tabular_choice_options(text)
    if tabular is not None:
        return tabular

    return None


def _extract_labeled_options(text: str) -> tuple[str, list[str], str, list[str | None]] | None:
    matches = list(_LABELED_OPTION_PATTERN.finditer(text))
    if len(matches) < 2:
        return None

    raw_options: list[str] = []
    raw_labels: list[str | None] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        option = text[start:end].strip(" \t\r\n。；;")
        option = _strip_trailing_stem(option)
        label = _normalize_option_label(match.group("label"))
        if option:
            raw_options.append(option)
            raw_labels.append(label)

    options, labels = _dedupe_options(raw_options, raw_labels)
    if len(options) < 2:
        return None

    stem = text[: matches[0].start()].strip(" \t\r\n：:")
    trailing = _trailing_question_after_last_option(text[matches[-1].end() :])
    if trailing and trailing not in stem:
        stem = f"{stem} {trailing}".strip()
    return stem or text, options, "inline_labeled", labels


def _extract_unlabeled_option_block(text: str) -> tuple[str, list[str], str, list[str | None]] | None:
    marker = re.search(r"(?:选项|候选项|备选项|如下|分别为)\s*[：:]", text)
    if not marker:
        return None

    stem = text[: marker.start()].strip(" \t\r\n：:")
    block = text[marker.end() :].strip()
    if not block:
        return None

    trailing_question = ""
    question_match = re.search(r"(?:请问|问：|判断|选择).{0,80}[？?]$", block)
    if question_match:
        trailing_question = question_match.group(0).strip()
        block = block[: question_match.start()].strip()

    parts = re.split(r"(?:\r?\n+|\t+|\s{2,}|[|｜/])", block)
    options, labels = _clean_options_with_labels(_strip_leading_option_word(part) for part in parts)
    if len(options) < 2:
        return None
    if trailing_question and trailing_question not in stem:
        stem = f"{stem} {trailing_question}".strip()
    return stem or text, options, "inline_unlabeled_block", labels


def _extract_tabular_choice_options(text: str) -> tuple[str, list[str], str, list[str | None]] | None:
    question_match = re.search(r"[？?]", text)
    if not question_match:
        return None
    stem = text[: question_match.end()].strip()
    if not re.search(r"(?:下列|哪一组|哪组|选项|表述均属于|正确)", stem):
        return None

    tail = text[question_match.end() :].strip()
    if "\t" not in tail and "\n" not in tail and "\r" not in tail:
        return None

    parts = re.split(r"(?:\t+|\r?\n+)", tail)
    options, labels = _clean_options_with_labels(parts)
    if len(options) < 2:
        return None
    return stem, options, "inline_tabular_options", labels


def _clean_options_with_labels(options: object) -> tuple[list[str], list[str | None]]:
    raw_options: list[str] = []
    raw_labels: list[str | None] = []
    for option in options:
        value = _strip_leading_option_word(str(option or ""))
        label, content = _split_leading_label(value)
        content = content.strip(" \t\r\n。；;")
        if content:
            raw_options.append(content)
            raw_labels.append(label)
    options_deduped, labels_deduped = _dedupe_options(raw_options, raw_labels)
    return options_deduped[:8], labels_deduped[:8]


def _dedupe_options(options: list[str], labels: list[str | None]) -> tuple[list[str], list[str | None]]:
    cleaned: list[str] = []
    cleaned_labels: list[str | None] = []
    seen: set[str] = set()
    for option, label in zip(options, labels, strict=False):
        normalized = re.sub(r"\s+", "", option)
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(option)
            cleaned_labels.append(label)
    return cleaned, cleaned_labels


def _split_leading_label(value: str) -> tuple[str | None, str]:
    match = _LEADING_LABEL_PATTERN.match(value)
    if not match:
        return None, value
    return _normalize_option_label(match.group("label")), value[match.end() :]


def _normalize_option_label(value: str | None) -> str | None:
    if not value:
        return None
    compact = re.sub(r"\s+", "", value).strip(".．、:：)）")
    if len(compact) == 1 and compact in "ABCDEFGHabcdefgh":
        return compact.upper()
    return compact or None


def _normalize_spaces(value: str) -> str:
    text = str(value or "").replace("\u3000", " ")
    text = re.sub(r" +", " ", text)
    return text.strip()


def _strip_leading_option_word(value: str) -> str:
    return re.sub(r"^\s*(?:选项|候选项|备选项)\s*[：:]\s*", "", value).strip()


def _strip_trailing_stem(option: str) -> str:
    return re.sub(r"(?:请问|问：|判断|选择).{0,80}[？?]\s*$", "", option).strip()


def _trailing_question_after_last_option(last_option_tail: str) -> str:
    match = re.search(r"(?:请问|问：|判断|选择).{0,80}[？?]\s*$", last_option_tail.strip())
    return match.group(0).strip() if match else ""
