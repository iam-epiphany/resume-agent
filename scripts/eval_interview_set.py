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
        ["请介绍一下你自己", ["张三", "简历"]],
        ["用三个词形容你自己", [], ["银行"]],
        ["你最大的优势是什么"],
        ["你还没毕业，为什么现在找实习"],
        ["你什么时候可以到岗"],
    ],
    "B": [
        ["介绍一下你的秒杀项目", ["秒杀"]],
        ["秒杀怎么防止超卖", ["Redis", "超卖"]],
        ["为什么用 Redis 预扣库存不用数据库锁", ["Redis"]],
        ["这个秒杀项目跑过多少并发", ["并发"]],
        ["外卖平台的微信支付流程是怎样的", ["微信支付"]],
        ["微信支付回调怎么防止重复处理", ["微信支付", "回调"]],
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
        [["介绍一下你的秒杀项目", "那超卖怎么防的？", "为什么不用数据库锁？"], ["秒杀", "超卖", "数据库锁"]],
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
    """从响应 context_package 提取实际检索到的文档名（source_doc）。"""
    package = body.get("context_package") or {}
    chunks = package.get("context_chunks") or []
    return list(dict.fromkeys(str(chunk.get("source_doc") or "") for chunk in chunks if chunk.get("source_doc")))


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
    report_items: list[dict] = []

    def evaluate_question(
        group: str, label: str, q: str, session_id: str | None,
        expected: list[str], forbidden: list[str],
        expected_documents: list[str] | None = None,
        expected_intent: str | None = None,
        expected_answer_modes: list[str] | None = None,
        expect_no_documents: bool = False,
        max_llm_calls: int | None = None,
    ) -> None:
        nonlocal fact_checked, fact_missed, forbidden_checked, forbidden_found, doc_checked, doc_hit
        body = ask(args.base_url, q, session_id, args.timeout)
        record_latency(latencies, body)
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
        print(f"  [{label}] {q}")
        print(
            f"      intent={body.get('intent')} mode={mode} "
            f"sufficiency={body.get('evidence_sufficiency')} "
            f"fallback={body.get('retrieval_fallback_level')} "
            f"llm_calls={calls} elapsed={body['_elapsed_s']}s"
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
                "expected_fact_hits": hits,
                "expected_facts": expected,
                "forbidden_found": forbidden_hits_found,
                "expected_document_hits": doc_hits,
                "expected_documents": expected_documents or [],
                "retrieved_documents": actual_documents,
                "expected_intent": expected_intent,
                "expected_answer_modes": expected_answer_modes or [],
                "expect_no_documents": expect_no_documents,
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
    if args.max_p95 is not None and latency_p95 > args.max_p95:
        failures.append(f"端到端延迟 p95={latency_p95}s，超过门槛 {args.max_p95}s")
    if args.max_llm_calls_p50 is not None and llm_p50 is not None and llm_p50 > args.max_llm_calls_p50:
        failures.append(f"LLM 调用次数 p50={llm_p50}，超过门槛 {args.max_llm_calls_p50}")

    print("\n================ 汇总 ================")
    print(f"错误/异常: {len(failures)} 条")
    print(f"延迟: p50={percentile(latencies, 50)}s  p95={percentile(latencies, 95)}s  共 {len(latencies)} 问")
    if llm_calls:
        print(f"LLM 调用次数: p50={percentile(llm_calls, 50)}  p95={percentile(llm_calls, 95)}  平均={sum(llm_calls) / len(llm_calls):.2f}")
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
    for f in failures:
        print(f"  FAIL: {f}")

    if args.report:
        report = {
            "base_url": args.base_url,
            "groups": args.groups,
            "latency_p50_s": percentile(latencies, 50),
            "latency_p95_s": percentile(latencies, 95),
            "llm_calls_p50": percentile(llm_calls, 50) if llm_calls else None,
            "llm_calls_p95": percentile(llm_calls, 95) if llm_calls else None,
            "answer_mode_counts": mode_counts,
            "expected_fact_hits": fact_checked - fact_missed,
            "expected_fact_total": fact_checked,
            "forbidden_found": forbidden_found,
            "expected_document_hits": doc_hit,
            "expected_document_total": doc_checked,
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
