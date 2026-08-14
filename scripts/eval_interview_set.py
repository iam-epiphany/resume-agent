"""ResumeMind RAG 端到端评测框架（面试官问题集 A-H + 指标聚合）。

用法:
    python scripts/eval_interview_set.py [--base-url http://127.0.0.1:18000]
                                         [--groups ABCDEFGH]
                                         [--timeout 60]
                                         [--report eval-report.json]

逐条调用 POST /api/qa/ask，采集：
  - 意图 / answer_mode / evidence_sufficiency / retrieval_fallback_level
  - 端到端耗时（p50 / p95 聚合）
  - 单请求 LLM 调用次数（llm_call_count，fast path 后普通问题应为 1）
  - 期望文档命中（context_package.retrieval_summary 或 answer 文本中检索到）
  - forbidden fact（幻觉检测：答案中不应出现的字样）
  - 多轮追问链正确性（H 组同 session 串行）

H 组追问链使用同一 session_id 串行调用。
输出控制台摘要 + 可选 JSON 报告（--report）。
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
# 问题集：单问 [question] 或 [question, expected_facts, forbidden_facts]
# expected_facts：期望在回答中出现的硬事实（用于 Recall/Fact 命中判定）
# forbidden_facts：禁止出现的内容（幻觉检测）
QUESTION_SETS = {
    "A": [
        ["请介绍一下你自己", ["简历", "自我介绍"]],
        ["用三个词形容你自己", [], ["银行"]],
        ["你最大的优势是什么"],
        ["你还没毕业，为什么现在找实习"],
        ["你什么时候可以到岗"],
    ],
    "B": [
        ["介绍一下你的 AI 开放平台项目", ["Token", "Redis"]],
        ["AI 开放平台怎么防止超卖", ["Redis", "超卖"]],
        ["为什么用 Redis Lua 和 Stream 异步发放", ["Redis", "Stream"]],
        ["AI 开放平台跑过多少并发", ["10"]],
        ["AI 开放平台的 API Key 如何保存", ["SHA-256"]],
        ["模拟计费如何防止重复扣费", ["幂等", "账本"]],
        ["什么是 grounding 校验"],
        ["多 Agent 是怎么编排的", ["Agent"]],
        ["NTT 数论变换是干什么的", ["NTT"]],
        ["你的简历写了 Kafka，项目里是怎么用的", ["Kafka"]],
    ],
    "C": [
        ["你的 Java 水平怎么样", ["Java"]],
        ["HashMap 的原理是什么", ["HashMap"]],
        ["缓存穿透、击穿、雪崩怎么解决", ["缓存"]],
        ["Kafka 怎么保证消息不丢", ["Kafka"]],
        ["为什么 B+ 树适合做索引", ["B+"]],
        ["Redisson 分布式锁会失效吗", ["Redisson"]],
    ],
    "D": [
        ["你的优点和缺点分别是什么"],
        ["讲一次你失败的经历"],
        ["你抗压能力怎么样"],
        ["你的职业规划是什么"],
        ["为什么选择在西安工作", ["西安"]],
        ["你期望的薪资是多少"],
        ["你的兴趣爱好是什么"],
    ],
    "E": [
        ["你本科都学了哪些课程", ["课程"]],
        ["你的绩点和专业排名是多少", ["绩点"]],
        ["为什么考研选择网络与信息安全", ["网络与信息安全"]],
        ["你本科参加过什么活动"],
        ["蓝桥杯一等奖是什么水平", ["蓝桥杯"]],
        ["你的英语六级成绩", ["六级"]],
        ["你的联系方式是什么", ["联系方式"]],
        ["你的 GitHub 是什么", ["GitHub"]],
    ],
    "F": [
        ["你会做红烧肉吗", [], ["红烧肉"]],
        ["今天天气怎么样", [], ["天气"]],
        ["你的银行卡密码是什么", [], ["银行卡密码"]],
        ["帮我写一首诗", [], ["诗"]],
    ],
    "G": [
        ["你好", [], []],
        ["谢谢", [], []],
        ["在吗", [], []],
    ],
    "H": [
        # 链 1
        [["介绍一下你的 AI 开放平台项目", "那超卖怎么防的？", "为什么还需要数据库兜底？"], ["Token", "超卖", "数据库"]],
        # 链 2
        [["你的技术栈是什么", "Java 这块水平怎么样", "那 Redis 呢？"], ["Java", "Redis"]],
        # 链 3
        [["你本科在哪个学校", "学了哪些课", "哪门学得最好？"], ["学校", "课程"]],
        # 链 4
        [["你平时有什么爱好", "会打篮球吗？"], ["爱好"]],
    ],
}

CHAIN_KEYS = {"H"}

_AUTH_HEADERS: dict[str, str] = {}


def admin_login(base_url: str, password: str, timeout: float) -> None:
    """Use an admin token so retrieval evidence is present in evaluation responses."""

    global _AUTH_HEADERS
    payload = json.dumps({"password": password}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/auth/login",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    token = body.get("token")
    if not token:
        raise RuntimeError("管理员登录未返回 token")
    _AUTH_HEADERS = {"Authorization": f"Bearer {token}"}


def ask(base_url: str, question: str, session_id: str | None, timeout: float) -> dict:
    payload = json.dumps(
        {
            "question": question,
            "session_id": session_id,
            # 检索文档只对管理员返回；匿名评测无法计算 Recall@文档。
            "include_debug": bool(_AUTH_HEADERS),
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/qa/ask",
        data=payload,
        headers={"Content-Type": "application/json", **_AUTH_HEADERS},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        body["_elapsed_s"] = round(elapsed, 2)
        return body
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        return {"_error": f"HTTP {e.code}: {detail}", "_elapsed_s": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}", "_elapsed_s": round(time.time() - t0, 2)}


def ask_stream(base_url: str, question: str, session_id: str | None, timeout: float) -> dict:
    """SSE 流式提问：额外采集 TTFT（首帧事件到达）与重排耗时（progress 事件）。

    返回体与 ask() 同构（final 事件即 QAResponse），附加 _ttft_s / _rerank_max_ms。
    重排耗时取自 progress 事件：summary 含 rerank_call_count 时该事件的
    elapsed_ms 即重排（或含重排的检索）耗时，取全程最大值。
    """
    payload = json.dumps(
        {
            "question": question,
            "session_id": session_id,
            "include_debug": bool(_AUTH_HEADERS),
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/qa/ask/stream",
        data=payload,
        headers={"Content-Type": "application/json", **_AUTH_HEADERS},
        method="POST",
    )
    t0 = time.time()
    try:
        body: dict = {}
        ttft: float | None = None
        rerank_max_ms = 0.0
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    if ttft is None:
                        ttft = time.time() - t0
                    continue
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[len("data:"):].strip())
                except json.JSONDecodeError:
                    continue
                if data.get("answer_mode") is not None and data.get("answer") is not None:
                    body = data
                    continue
                summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
                if summary.get("rerank_call_count") and data.get("elapsed_ms") is not None:
                    rerank_max_ms = max(rerank_max_ms, float(data["elapsed_ms"]))
        elapsed = time.time() - t0
        if not body:
            return {"_error": "流式响应未返回最终答案", "_elapsed_s": round(elapsed, 2)}
        body["_elapsed_s"] = round(elapsed, 2)
        body["_ttft_s"] = round(ttft, 2) if ttft is not None else None
        body["_rerank_max_ms"] = round(rerank_max_ms, 1)
        return body
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        return {"_error": f"HTTP {e.code}: {detail}", "_elapsed_s": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}", "_elapsed_s": round(time.time() - t0, 2)}


def summarize_answer(answer: str | None, max_len: int = 120) -> str:
    if not answer:
        return "(empty)"
    answer = answer.replace("\n", " ")
    return answer[:max_len] + ("…" if len(answer) > max_len else "")


def normalize_match_text(value: str) -> str:
    return (
        value.casefold()
        .replace("～", "-")
        .replace("~", "-")
        .replace("—", "-")
        .replace("–", "-")
        .replace("／", "/")
        .replace("（", "(")
        .replace("）", ")")
        .replace(" ", "")
        .replace(",", "")
        .replace("-", "")
    )


def item_matches(answer: str, item: str) -> bool:
    normalized_answer = normalize_match_text(answer)
    alternatives = [part.strip() for part in item.split("|") if part.strip()]
    return any(normalize_match_text(part) in normalized_answer for part in alternatives)


def fact_hits(answer: str | None, facts: list[str]) -> list[str]:
    if not answer:
        return []
    return [fact for fact in facts if fact and item_matches(answer, fact)]


def forbidden_hits(answer: str | None, forbidden: list[str]) -> list[str]:
    if not answer:
        return []
    return [item for item in forbidden if item and item_matches(answer, item)]


def retrieved_documents(body: dict) -> list[str]:
    """从响应 context_package 提取实际检索到的文档名（source_doc，保持顺序）。"""
    package = body.get("context_package") or {}
    chunks = package.get("context_chunks") or []
    return list(dict.fromkeys(str(chunk.get("source_doc") or "") for chunk in chunks if chunk.get("source_doc")))


def retrieved_chunks(body: dict) -> list[dict]:
    """提取检索上下文块（管理员视角；匿名视角为空）。"""
    package = body.get("context_package") or {}
    return list(package.get("context_chunks") or [])


def recall_at_k(body: dict, expected_documents: list[str], k: int = 5) -> float:
    """Recall@K：前 K 个检索文档中期望文档的占比（无期望文档时为 None）。"""
    if not expected_documents:
        return None
    top_k = retrieved_documents(body)[:k]
    hits = sum(1 for doc in expected_documents if any(doc in item for item in top_k))
    return hits / len(expected_documents)


def mrr(body: dict, expected_documents: list[str]) -> float:
    """MRR：第一个期望文档在检索文档序列中的倒数排名（无期望文档时为 None）。"""
    if not expected_documents:
        return None
    for rank, doc in enumerate(retrieved_documents(body), start=1):
        if any(expected in doc for expected in expected_documents):
            return 1.0 / rank
    return 0.0


def context_precision(body: dict, expected_facts: list[str]) -> float | None:
    """上下文精确率：命中至少一个期望事实的 chunk 占全部上下文 chunk 的比例。

    无期望事实或上下文为空时返回 None（不计入聚合）。
    """
    if not expected_facts:
        return None
    chunks = retrieved_chunks(body)
    if not chunks:
        return None
    relevant = sum(
        1
        for chunk in chunks
        if any(item_matches(str(chunk.get("text") or "") + str(chunk.get("section_title") or ""), fact) for fact in expected_facts)
    )
    return relevant / len(chunks)


def attribution_hits(answer: str | None, pairs: list[list[str]]) -> list[list[str]]:
    """事实关联：期望「实体, 值」对同时出现在回答中才计命中（presence-only 的弱关联版）。"""
    if not answer:
        return []
    hits: list[list[str]] = []
    for pair in pairs or []:
        subject, value = (str(pair[0]) if len(pair) > 0 else ""), (str(pair[1]) if len(pair) > 1 else "")
        if subject and value and item_matches(answer, subject) and item_matches(answer, value):
            hits.append(pair)
    return hits


def expected_document_hits(body: dict, expected_documents: list[str]) -> list[str]:
    """期望文档命中：检索上下文（context_chunks.source_doc）中包含期望文档名。"""
    if not expected_documents:
        return []
    retrieved = retrieved_documents(body)
    return [doc for doc in expected_documents if doc and any(doc in item for item in retrieved)]


def load_jsonl_cases(path: str) -> list[dict]:
    """从 JSONL 加载评测用例（group/question/expected_facts/forbidden_facts/expected_documents/session）。"""
    cases: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cases.append(record)
    return cases


def record_latency(records: list[float], body: dict) -> None:
    if "_elapsed_s" in body and body["_elapsed_s"] is not None:
        records.append(body["_elapsed_s"])


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * pct / 100)))
    return round(ordered[index], 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="ResumeMind 面试问题集 RAG 评测")
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--groups", default="ABCDEFGH")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--report", default=None, help="输出 JSON 报告路径")
    parser.add_argument(
        "--admin-password",
        default=os.getenv("ADMIN_PASSWORD", ""),
        help="管理员密码；提供后才能返回检索依据并计算 Recall@文档",
    )
    parser.add_argument("--max-p95", type=float, default=None, help="端到端 p95 秒数门槛")
    parser.add_argument("--max-llm-calls-p50", type=float, default=None, help="LLM 调用次数 p50 门槛")
    parser.add_argument("--max-ttft-p50", type=float, default=None, help="TTFT p50 秒数门槛（需 --stream）")
    parser.add_argument("--max-rerank-p50", type=float, default=None, help="重排耗时 p50 毫秒门槛（需 --stream）")
    parser.add_argument(
        "--stream", action="store_true",
        help="用 SSE 流式接口提问，额外采集 TTFT 与重排耗时指标",
    )
    parser.add_argument(
        "--data", default=None,
        help="JSONL 评测数据路径（覆盖内置问题集；每条含 group/question/expected_facts/"
             "forbidden_facts/expected_documents/session，可含 followup 多轮链）",
    )
    args = parser.parse_args()

    if args.admin_password:
        admin_login(args.base_url.rstrip("/"), args.admin_password, args.timeout)

    failures: list[str] = []
    latencies: list[float] = []
    llm_calls: list[int] = []
    mode_counts: dict[str, int] = {}
    fact_checked = 0
    fact_missed = 0
    forbidden_checked = 0
    forbidden_found = 0
    doc_checked = 0
    doc_hit = 0
    ttft_records: list[float] = []
    rerank_records: list[float] = []
    recall_k_records: list[float] = []
    mrr_records: list[float] = []
    precision_records: list[float] = []
    attribution_checked = 0
    attribution_hit = 0
    refusal_checked = 0
    refusal_correct = 0
    report_items: list[dict] = []

    def evaluate_question(
        group: str, label: str, q: str, session_id: str | None,
        expected: list[str], forbidden: list[str],
        expected_documents: list[str] | None = None,
        expected_intent: str | None = None,
        expected_answer_modes: list[str] | None = None,
        expect_no_documents: bool = False,
        max_llm_calls: int | None = None,
        expected_attributions: list[list[str]] | None = None,
        expect_refusal: bool = False,
        use_stream: bool = False,
    ) -> None:
        nonlocal fact_checked, fact_missed, forbidden_checked, forbidden_found, doc_checked, doc_hit
        nonlocal attribution_checked, attribution_hit, refusal_checked, refusal_correct
        if use_stream:
            body = ask_stream(args.base_url, q, session_id, args.timeout)
        else:
            body = ask(args.base_url, q, session_id, args.timeout)
        record_latency(latencies, body)
        if body.get("_ttft_s") is not None:
            ttft_records.append(float(body["_ttft_s"]))
        if body.get("_rerank_max_ms"):
            rerank_records.append(float(body["_rerank_max_ms"]))
        if "_error" in body:
            print(f"  [{label}] {q}")
            print(f"      ERROR {body['_error']}")
            failures.append(f"{group}-{label}: {q}")
            report_items.append({"label": f"{group}-{label}", "question": q, "error": body["_error"]})
            return
        mode = body.get("answer_mode")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        answer = body.get("answer")
        calls = body.get("llm_call_count")
        if calls is not None:
            llm_calls.append(int(calls))
        if expected_intent and body.get("intent") != expected_intent:
            failures.append(
                f"{group}-{label}: intent={body.get('intent')}，期望 {expected_intent}"
            )
        if expected_answer_modes and mode not in expected_answer_modes:
            failures.append(
                f"{group}-{label}: answer_mode={mode}，期望 {'/'.join(expected_answer_modes)}"
            )
        # 拒答正确率：expect_refusal 的题目必须拒答/转移，不得硬编
        if expect_refusal:
            refusal_checked += 1
            if mode in {"failed", "redirected"}:
                refusal_correct += 1
            else:
                failures.append(f"{group}-{label}: 应拒答/转移，实际 mode={mode}")
        if max_llm_calls is not None and (calls is None or int(calls) > max_llm_calls):
            failures.append(
                f"{group}-{label}: llm_call_count={calls}，上限 {max_llm_calls}"
            )
        hits = fact_hits(answer, expected)
        fact_checked += len(expected)
        fact_missed += len(expected) - len(hits)
        # 修复：forbidden 形参被同名局部变量覆盖的 bug——期望禁止词列表与
        # 实际命中的禁止词分开命名，统计分母用"期望检查数"而非"命中数"
        forbidden_hits_found = forbidden_hits(answer, forbidden)
        forbidden_checked += len(forbidden)
        forbidden_found += len(forbidden_hits_found)
        if forbidden_hits_found:
            failures.append(f"{group}-{label}: 出现禁止内容 {forbidden_hits_found}")
        # 质量阈值：期望事实缺失超过一半 → 记为失败（明显的事实错误不应被静默）
        if expected and len(hits) < max(1, len(expected) / 2):
            failures.append(
                f"{group}-{label}: 期望事实缺失 {len(expected) - len(hits)}/{len(expected)} "
                f"({', '.join(str(item) for item in expected if item not in hits)})"
            )
        # 期望文档命中（Retrieval Recall@文档）：从 context_package 提取检索文档核对
        doc_hits = expected_document_hits(body, expected_documents or [])
        doc_checked += len(expected_documents or [])
        doc_hit += len(doc_hits)
        if expected_documents and len(doc_hits) < len(expected_documents):
            failures.append(
                f"{group}-{label}: 期望文档未检索到 {len(doc_hits)}/{len(expected_documents)} "
                f"({', '.join(str(item) for item in expected_documents if item not in doc_hits)})"
            )
        actual_documents = retrieved_documents(body)
        if expect_no_documents:
            if not _AUTH_HEADERS:
                failures.append(f"{group}-{label}: expect_no_documents 需要管理员调试响应")
            elif actual_documents:
                failures.append(
                    f"{group}-{label}: 本应零检索却返回文档 {', '.join(actual_documents)}"
                )
        # 检索质量：Recall@5 / MRR / 上下文精确率（管理员视角才可计算）
        r5 = recall_at_k(body, expected_documents or [], k=5)
        rr = mrr(body, expected_documents or [])
        precision = context_precision(body, expected)
        if r5 is not None:
            recall_k_records.append(r5)
        if rr is not None:
            mrr_records.append(rr)
        if precision is not None:
            precision_records.append(precision)
        # 事实关联：实体—值对同时出现（presence-only 弱关联版；服务端台账做强校验）
        attribution_hits_found = attribution_hits(answer, expected_attributions)
        attribution_checked += len(expected_attributions or [])
        attribution_hit += len(attribution_hits_found)
        if expected_attributions:
            missing = [pair for pair in expected_attributions if pair not in attribution_hits_found]
            if missing:
                failures.append(f"{group}-{label}: 事实关联缺失 {missing}")
        print(f"  [{label}] {q}")
        print(
            f"      intent={body.get('intent')} mode={mode} "
            f"sufficiency={body.get('evidence_sufficiency')} "
            f"fallback={body.get('retrieval_fallback_level')} "
            f"llm_calls={calls} elapsed={body['_elapsed_s']}s"
            + (f" ttft={body.get('_ttft_s')}s rerank={body.get('_rerank_max_ms')}ms" if body.get("_ttft_s") is not None else "")
        )
        print(f"      -> {summarize_answer(answer)}")
        if hits:
            print(f"      期望事实命中: {hits}")
        elif expected:
            fact_missed_note = ", ".join(str(item) for item in expected)
            print(f"      期望事实缺失: {fact_missed_note}")
        if forbidden_hits_found:
            print(f"      ⚠ 禁止内容: {forbidden_hits_found}")
        report_items.append(
            {
                "label": f"{group}-{label}",
                "question": q,
                "intent": body.get("intent"),
                "answer_mode": mode,
                "evidence_sufficiency": body.get("evidence_sufficiency"),
                "fallback_level": body.get("retrieval_fallback_level"),
                "llm_call_count": calls,
                "elapsed_s": body["_elapsed_s"],
                "ttft_s": body.get("_ttft_s"),
                "rerank_max_ms": body.get("_rerank_max_ms"),
                "expected_fact_hits": hits,
                "expected_facts": expected,
                "forbidden_found": forbidden_hits_found,
                "expected_document_hits": doc_hits,
                "expected_documents": expected_documents or [],
                "retrieved_documents": actual_documents,
                "recall_at_5": r5,
                "mrr": rr,
                "context_precision": precision,
                "attribution_hits": attribution_hits_found,
                "expected_intent": expected_intent,
                "expected_answer_modes": expected_answer_modes or [],
                "expect_no_documents": expect_no_documents,
                "expect_refusal": expect_refusal,
                "max_llm_calls": max_llm_calls,
                "answer": answer,
            }
        )

    if args.data:
        # JSONL 评测数据（结构化，替代内置问题集）
        jsonl_cases = load_jsonl_cases(args.data)
        sessions: dict[str, str] = {}
        print(f"\n=== JSONL 评测（{len(jsonl_cases)} 条，{args.data}）===")
        for index, case in enumerate(jsonl_cases, start=1):
            session_key = case.get("session")
            session_id = None
            if session_key:
                if session_key not in sessions:
                    sessions[session_key] = f"eval-{session_key}-" + hex(int(time.time()))[2:]
                session_id = sessions[session_key]
            evaluate_question(
                str(case.get("group") or "X"),
                str(index),
                str(case.get("question") or ""),
                session_id,
                list(case.get("expected_facts") or []),
                list(case.get("forbidden_facts") or []),
                expected_documents=list(case.get("expected_documents") or []),
                expected_intent=case.get("expected_intent"),
                expected_answer_modes=(
                    list(case.get("expected_answer_modes") or [])
                    or ([str(case["expected_answer_mode"])] if case.get("expected_answer_mode") else [])
                ),
                expect_no_documents=bool(case.get("expect_no_documents")),
                max_llm_calls=(int(case["max_llm_calls"]) if case.get("max_llm_calls") is not None else None),
                expected_attributions=[
                    list(pair) for pair in (case.get("expected_attributions") or [])
                ],
                expect_refusal=bool(case.get("expect_refusal")),
                use_stream=bool(args.stream),
            )
    else:
        for group in args.groups:
            if group in CHAIN_KEYS:
                for chain_idx, (chain, chain_facts) in enumerate(QUESTION_SETS[group], start=1):
                    session_id = f"eval-{group.lower()}{chain_idx}-" + hex(int(time.time()))[2:]
                    print(f"\n=== {group}{chain_idx}（追问链，session={session_id}）===")
                    for turn, q in enumerate(chain, start=1):
                        evaluate_question(group, f"{chain_idx}-{turn}", q, session_id, chain_facts, [])
                continue

            print(f"\n=== {group} 组 ===")
            for index, entry in enumerate(QUESTION_SETS[group], start=1):
                q = entry[0]
                expected = entry[1] if len(entry) > 1 else []
                forbidden = entry[2] if len(entry) > 2 else []
                evaluate_question(group, str(index), q, None, expected, forbidden)

    latency_p95 = percentile(latencies, 95)
    llm_p50 = percentile(llm_calls, 50) if llm_calls else None
    ttft_p50 = percentile(ttft_records, 50) if ttft_records else None
    rerank_p50 = percentile(rerank_records, 50) if rerank_records else None
    if args.max_p95 is not None and latency_p95 > args.max_p95:
        failures.append(f"端到端延迟 p95={latency_p95}s，超过门槛 {args.max_p95}s")
    if args.max_llm_calls_p50 is not None and llm_p50 is not None and llm_p50 > args.max_llm_calls_p50:
        failures.append(f"LLM 调用次数 p50={llm_p50}，超过门槛 {args.max_llm_calls_p50}")
    if args.max_ttft_p50 is not None and ttft_p50 is not None and ttft_p50 > args.max_ttft_p50:
        failures.append(f"TTFT p50={ttft_p50}s，超过门槛 {args.max_ttft_p50}s")
    if args.max_rerank_p50 is not None and rerank_p50 is not None and rerank_p50 > args.max_rerank_p50:
        failures.append(f"重排耗时 p50={rerank_p50}ms，超过门槛 {args.max_rerank_p50}ms")

    recall_k_avg = (sum(recall_k_records) / len(recall_k_records)) if recall_k_records else None
    mrr_avg = (sum(mrr_records) / len(mrr_records)) if mrr_records else None
    precision_avg = (sum(precision_records) / len(precision_records)) if precision_records else None

    print("\n================ 汇总 ================")
    print(f"错误/异常: {len(failures)} 条")
    print(f"延迟: p50={percentile(latencies, 50)}s  p95={percentile(latencies, 95)}s  共 {len(latencies)} 问")
    if llm_calls:
        print(f"LLM 调用次数: p50={percentile(llm_calls, 50)}  p95={percentile(llm_calls, 95)}  平均={sum(llm_calls) / len(llm_calls):.2f}")
    if ttft_records:
        print(f"TTFT（首帧）: p50={percentile(ttft_records, 50)}s  p95={percentile(ttft_records, 95)}s")
    if rerank_records:
        print(f"重排耗时: p50={percentile(rerank_records, 50)}ms  p95={percentile(rerank_records, 95)}ms")
    print(f"answer_mode 分布: {mode_counts or '(无成功响应)'}")
    print(
        f"期望事实: 命中 {fact_checked - fact_missed}/{fact_checked} "
        f"({round((fact_checked - fact_missed) / max(fact_checked, 1) * 100, 1)}%)"
    )
    print(
        f"禁止内容: 检出 {forbidden_found}/{forbidden_checked} "
        f"({'通过' if forbidden_found == 0 else '有幻觉风险'})"
    )
    if doc_checked:
        print(
            f"期望文档命中(Recall@文档): {doc_hit}/{doc_checked} "
            f"({round(doc_hit / max(doc_checked, 1) * 100, 1)}%)"
        )
    if recall_k_avg is not None:
        print(f"Recall@5 平均: {round(recall_k_avg, 3)}")
    if mrr_avg is not None:
        print(f"MRR 平均: {round(mrr_avg, 3)}")
    if precision_avg is not None:
        print(f"上下文精确率平均: {round(precision_avg, 3)}")
    if attribution_checked:
        print(
            f"事实关联正确率: {attribution_hit}/{attribution_checked} "
            f"({round(attribution_hit / max(attribution_checked, 1) * 100, 1)}%)"
        )
    if refusal_checked:
        print(
            f"拒答正确率: {refusal_correct}/{refusal_checked} "
            f"({round(refusal_correct / max(refusal_checked, 1) * 100, 1)}%)"
        )
    for f in failures:
        print(f"  FAIL: {f}")

    if args.report:
        report = {
            "base_url": args.base_url,
            "groups": args.groups,
            "stream": bool(args.stream),
            "latency_p50_s": percentile(latencies, 50),
            "latency_p95_s": percentile(latencies, 95),
            "llm_calls_p50": percentile(llm_calls, 50) if llm_calls else None,
            "llm_calls_p95": percentile(llm_calls, 95) if llm_calls else None,
            "ttft_p50_s": percentile(ttft_records, 50) if ttft_records else None,
            "ttft_p95_s": percentile(ttft_records, 95) if ttft_records else None,
            "rerank_p50_ms": percentile(rerank_records, 50) if rerank_records else None,
            "rerank_p95_ms": percentile(rerank_records, 95) if rerank_records else None,
            "answer_mode_counts": mode_counts,
            "expected_fact_hits": fact_checked - fact_missed,
            "expected_fact_total": fact_checked,
            "forbidden_found": forbidden_found,
            "expected_document_hits": doc_hit,
            "expected_document_total": doc_checked,
            "recall_at_5_avg": round(recall_k_avg, 4) if recall_k_avg is not None else None,
            "mrr_avg": round(mrr_avg, 4) if mrr_avg is not None else None,
            "context_precision_avg": round(precision_avg, 4) if precision_avg is not None else None,
            "attribution_hits": attribution_hit,
            "attribution_total": attribution_checked,
            "refusal_correct": refusal_correct,
            "refusal_total": refusal_checked,
            "passed": not failures,
            "failures": failures,
            "items": report_items,
        }
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n报告已写入: {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
