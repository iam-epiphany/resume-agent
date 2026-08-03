"""面试官问题集 A-H 8 组端到端评测脚本。

用法:
    python scripts/eval_interview_set.py [--base-url http://127.0.0.1:18000]
                                         [--groups ABCDEFGH]
                                         [--timeout 60]

逐条调用 POST /api/qa/ask，输出 5 维度观测：intent / answer_mode /
evidence_sufficiency / retrieval_fallback_level / 端到端耗时 + 回答摘要。

H 组追问链使用同一 session_id 串行调用。
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

QUESTION_SETS = {
    "A": [
        "请介绍一下你自己",
        "用三个词形容你自己",
        "你最大的优势是什么",
        "你还没毕业，为什么现在找实习",
        "你什么时候可以到岗",
    ],
    "B": [
        "介绍一下你的秒杀项目",
        "秒杀怎么防止超卖",
        "为什么用 Redis 预扣库存不用数据库锁",
        "这个秒杀项目跑过多少并发",
        "外卖平台的微信支付流程是怎样的",
        "微信支付回调怎么防止重复处理",
        "什么是 grounding 校验",
        "多 Agent 是怎么编排的",
        "NTT 数论变换是干什么的",
        "你的简历写了 Kafka，项目里是怎么用的",
    ],
    "C": [
        "你的 Java 水平怎么样",
        "HashMap 的原理是什么",
        "缓存穿透、击穿、雪崩怎么解决",
        "Kafka 怎么保证消息不丢",
        "为什么 B+ 树适合做索引",
        "Redisson 分布式锁会失效吗",
    ],
    "D": [
        "你的优点和缺点分别是什么",
        "讲一次你失败的经历",
        "你抗压能力怎么样",
        "你的职业规划是什么",
        "为什么选择在西安工作",
        "你期望的薪资是多少",
        "你的兴趣爱好是什么",
    ],
    "E": [
        "你本科都学了哪些课程",
        "你的绩点和专业排名是多少",
        "为什么考研选择网络与信息安全",
        "你本科参加过什么活动",
        "蓝桥杯一等奖是什么水平",
        "你的英语六级成绩",
        "你的联系方式是什么",
        "你的 GitHub 是什么",
    ],
    "F": [
        "你会做红烧肉吗",
        "今天天气怎么样",
        "你的银行卡密码是什么",
        "帮我写一首诗",
    ],
    "G": [
        "你好",
        "谢谢",
        "在吗",
    ],
    "H": [
        # 链 1
        ["介绍一下你的秒杀项目", "那超卖怎么防的？", "为什么不用数据库锁？"],
        # 链 2
        ["你的技术栈是什么", "Java 这块水平怎么样", "那 Redis 呢？"],
        # 链 3
        ["你本科在哪个学校", "学了哪些课", "哪门学得最好？"],
        # 链 4
        ["你平时有什么爱好", "会打篮球吗？"],
    ],
}


def ask(base_url: str, question: str, session_id: str | None, timeout: float) -> dict:
    payload = json.dumps(
        {"question": question, "session_id": session_id}, ensure_ascii=False
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/qa/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
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


def main() -> int:
    parser = argparse.ArgumentParser(description="面试官问题集 A-H 端到端评测")
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--groups", default="ABCDEFGH")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    failures = []
    for group in args.groups:
        if group == "H":
            for chain_idx, chain in enumerate(QUESTION_SETS["H"], start=1):
                session_id = f"eval-h{chain_idx}-" + hex(int(time.time()))[2:]
                print(f"\n=== H{chain_idx}（追问链，session={session_id}）===")
                for turn, q in enumerate(chain, start=1):
                    body = ask(args.base_url, q, session_id, args.timeout)
                    if "_error" in body:
                        print(f"  [{turn}] {q}\n      ERROR {body['_error']}")
                        failures.append(f"H{chain_idx}-{turn}: {q}")
                        continue
                    print(f"  [{turn}] {q}")
                    print(
                        f"      intent={body.get('intent')} mode={body.get('answer_mode')} "
                        f"sufficiency={body.get('evidence_sufficiency')} "
                        f"fallback={body.get('retrieval_fallback_level')} "
                        f"elapsed={body['_elapsed_s']}s"
                    )
                    print(f"      -> {summarize_answer(body.get('answer'))}")
            continue

        print(f"\n=== {group} 组 ===")
        for q in QUESTION_SETS[group]:
            body = ask(args.base_url, q, None, args.timeout)
            if "_error" in body:
                print(f"  [{q}] ERROR {body['_error']}")
                failures.append(f"{group}: {q}")
                continue
            print(f"  [{q}]")
            print(
                f"      intent={body.get('intent')} mode={body.get('answer_mode')} "
                f"sufficiency={body.get('evidence_sufficiency')} "
                f"fallback={body.get('retrieval_fallback_level')} "
                f"elapsed={body['_elapsed_s']}s"
            )
            print(f"      -> {summarize_answer(body.get('answer'))}")

    print("\n================ 汇总 ================")
    print(f"评测完成。错误/异常 {len(failures)} 条")
    for f in failures:
        print(f"  FAIL: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
