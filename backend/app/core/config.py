import os
import re
from pathlib import Path

from dotenv import load_dotenv

# 本地直接运行（python -m uvicorn）时加载项目根 .env；
# docker compose 已把 .env 注入环境变量（load_dotenv 不覆盖已有值），互不影响。
# pytest 环境（conftest 设置 RESUMEMIND_SKIP_DOTENV）跳过：测试依赖默认配置，避免 .env 调参值干扰。
if not os.getenv("RESUMEMIND_SKIP_DOTENV"):
    load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer; received {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}; received {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number; received {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}; received {value}")
    return value


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    value = (os.getenv(name) or default).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise RuntimeError(f"{name} must be one of {allowed}; received {value!r}")
    return value


APP_NAME = "ResumeMind"
API_TITLE = "ResumeMind API"
BUILD_ID = os.getenv("RESUME_BUILD_ID", "dev").strip() or "dev"
APP_DESCRIPTION = "基于个人简历、证书与项目文档的可信 RAG 问答"
CORS_ORIGINS = [
    value.strip()
    for value in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174",
    ).split(",")
    if value.strip()
]
FRONTEND_DEV_SERVER = os.getenv("RESUME_FRONTEND_DEV_SERVER", "").strip().rstrip("/")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("RESUME_DATA_DIR", PROJECT_ROOT / "data"))
DATABASE_PATH = DATA_DIR / "app.db"
DOCUMENT_DIR = DATA_DIR / "documents" / "originals"
QDRANT_STORAGE_DIR = DATA_DIR / "qdrant"
AUDIT_ARCHIVE_DIR = DATA_DIR / "audit_archives"

# 模型缓存默认落在项目 data/ 下（RESUME_MODEL_CACHE_DIR 可覆盖）
MODEL_CACHE_DIR = Path(os.getenv("RESUME_MODEL_CACHE_DIR") or DATA_DIR / "model_cache")
MODEL_DIR = DATA_DIR / "models"
DEFAULT_EMBEDDING_MODEL_DIR = MODEL_DIR / "bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL_DIR = MODEL_DIR / "bge-reranker-base"
DEFAULT_RERANKER_ONNX_INT8_DIR = MODEL_DIR / "bge-reranker-base-onnx-int8"
HF_HOME = Path(os.getenv("HF_HOME") or MODEL_CACHE_DIR / "huggingface")
HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE") or HF_HOME / "hub")
SENTENCE_TRANSFORMERS_HOME = Path(
    os.getenv("SENTENCE_TRANSFORMERS_HOME") or MODEL_CACHE_DIR / "sentence_transformers"
)
TORCH_HOME = Path(os.getenv("TORCH_HOME") or MODEL_CACHE_DIR / "torch")
RESUME_OFFLINE_MODE = _env_bool("RESUME_OFFLINE_MODE", True)
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH")
RERANKER_ONNX_MODEL_PATH = Path(
    os.getenv("RERANKER_ONNX_MODEL_PATH") or DEFAULT_RERANKER_ONNX_INT8_DIR / "model.onnx"
)
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "auto").strip().lower()
MODEL_GPU_MIN_FREE_MEMORY_GB = _env_float("MODEL_GPU_MIN_FREE_MEMORY_GB", 1.0)
RESUME_PERFORMANCE_MODE = _env_choice(
    "RESUME_PERFORMANCE_MODE",
    "auto",
    {"auto", "gpu", "cpu_balanced", "cpu_low_resource"},
)
MODEL_BACKEND = _env_choice(
    "MODEL_BACKEND",
    "pytorch",
    {"pytorch", "onnx", "openvino"},
)
MODEL_WARMUP_POLICY = _env_choice(
    "MODEL_WARMUP_POLICY",
    "background",
    {"background", "blocking", "lazy"},
)
RERANK_BATCH_SIZE = _env_int("RERANK_BATCH_SIZE", 0)
RERANK_MAX_LENGTH = _env_int("RERANK_MAX_LENGTH", 1024, minimum=1)
RERANK_INPUT_MODE = _env_choice(
    "RERANK_INPUT_MODE",
    "embedding",
    {"embedding", "compact"},
)
TORCH_NUM_THREADS = _env_int("TORCH_NUM_THREADS", 0)
TORCH_NUM_INTEROP_THREADS = _env_int("TORCH_NUM_INTEROP_THREADS", 0)
RERANK_ONNX_INTRA_OP_THREADS = _env_int("RERANK_ONNX_INTRA_OP_THREADS", 0)
RERANK_ONNX_INTER_OP_THREADS = _env_int("RERANK_ONNX_INTER_OP_THREADS", 1)
RERANK_ONNX_ENABLE_CPU_MEM_ARENA = _env_bool("RERANK_ONNX_ENABLE_CPU_MEM_ARENA", True)
RERANK_ONNX_ENABLE_PREPACKING = _env_bool("RERANK_ONNX_ENABLE_PREPACKING", True)
QUERY_EMBEDDING_CACHE_BYTES = _env_int(
    "QUERY_EMBEDDING_CACHE_BYTES", 64 * 1024 * 1024, minimum=0
)
RERANK_SCORE_CACHE_BYTES = _env_int(
    "RERANK_SCORE_CACHE_BYTES", 32 * 1024 * 1024, minimum=0
)
QUERY_EMBEDDING_CACHE_ITEMS = _env_int("QUERY_EMBEDDING_CACHE_ITEMS", 2048, minimum=0)
RERANK_SCORE_CACHE_ITEMS = _env_int("RERANK_SCORE_CACHE_ITEMS", 50_000, minimum=0)

if TORCH_NUM_THREADS > 0:
    os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(TORCH_NUM_THREADS))

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(SENTENCE_TRANSFORMERS_HOME))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
if RESUME_OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".txt", ".md", ".doc", ".docx", ".pdf", ".jsonl", ".html", ".htm",
}
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
CHUNK_TARGET_TOKENS = 512
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 80
SEMANTIC_BREAK_THRESHOLD = 0.62

# 索引版本号：模型/维度变化时必须换新值并重建索引（SQLite 与 Qdrant 双端过滤旧版本）。
# 2026-08-11 起默认 bge-small-zh-v1.5（512 维，2C4G 轻量档）；可用环境变量覆盖回旧模型。
INDEX_VERSION = (
    os.getenv("INDEX_VERSION", "bge-small-zh-dense-v6-resume").strip()
    or "bge-small-zh-dense-v6-resume"
)
# 直跑后端默认连 RESUME_QDRANT_HTTP_PORT 映射的 qdrant（本地 .env 为 16333，服务器默认 6333）；
# docker-compose 内显式传 QDRANT_URL=http://qdrant:6333 覆盖此默认值
RESUME_QDRANT_HTTP_PORT = int(os.getenv("RESUME_QDRANT_HTTP_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL", f"http://127.0.0.1:{RESUME_QDRANT_HTTP_PORT}")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "resumemind_chunks")
QDRANT_AUTO_CREATE_COLLECTION = _env_bool("QDRANT_AUTO_CREATE_COLLECTION", True)
QDRANT_UPSERT_BATCH_SIZE = _env_int("QDRANT_UPSERT_BATCH_SIZE", 128, minimum=1)
QDRANT_DENSE_VECTOR_NAME = "dense"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
EMBEDDING_DIMENSION = _env_int("EMBEDDING_DIMENSION", 512, minimum=1)
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 8, minimum=1)
EMBEDDING_MAX_BATCH_SIZE = _env_int("EMBEDDING_MAX_BATCH_SIZE", 16, minimum=1)
RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 50, minimum=1)
# 十余篇个人资料库中，cross-encoder 是 CPU 热路径；默认控制候选数，歧义问题可通过环境变量提高。
RERANK_TOP_K = _env_int("RERANK_TOP_K", 6, minimum=1)
# 2026-08-14 评测校准 8→6：重排 ~1.2s/对，少 2 对省 ~2.4s，p50 逼近 6s 目标
RERANK_CANDIDATE_LIMIT = _env_int("RERANK_CANDIDATE_LIMIT", 6, minimum=1)
MAX_PROMPT_CHUNKS = 12
MAX_PROMPT_TOKENS = _env_int("MAX_PROMPT_TOKENS", 3600, minimum=1)
MIN_PROMPT_CHUNKS = _env_int("MIN_PROMPT_CHUNKS", 6, minimum=0)
FORCE_MIN_CHUNKS = _env_bool("FORCE_MIN_CHUNKS", True)
RERANK_PROMPT_THRESHOLD = _env_float("RERANK_PROMPT_THRESHOLD", 0.20)
# 跳重排分差比例调参入口（2026-08-14 激活：跳重排判定以 SKIP_RERANK_MARGIN_RATIO
# 为准，本参数保留供旧配置兼容；改这里不影响默认行为）
RELATIVE_SCORE_RATIO = _env_float("RELATIVE_SCORE_RATIO", 0.40)
# 每个文档在最终 prompt 中的最大片段数（多样性约束，防止单一文档霸屏挤掉其他对象）
PER_DOCUMENT_PROMPT_CAP = _env_int("PER_DOCUMENT_PROMPT_CAP", 2, minimum=1)
# 关键词精确召回：每个 aspect 最多注入的关键词候选数（小库全量扫，0=关闭）
KEYWORD_RECALL_LIMIT = _env_int("KEYWORD_RECALL_LIMIT", 6, minimum=0)
FINAL_CITATION_LIMIT = MAX_PROMPT_CHUNKS
MIN_EVIDENCE_COVERAGE = _env_float("MIN_EVIDENCE_COVERAGE", 0.25)
DIRECT_EVIDENCE_COVERAGE = _env_float("DIRECT_EVIDENCE_COVERAGE", 0.45)
# core 选择的绝对相关性下限：实际部署中 rerank 分数为 0-1 分布（跑题≈0.012、真实匹配≥0.48），
# 而 RERANK_PROMPT_THRESHOLD 可能被调得很低（如 0.01），取两者最大值保证跑题问题必被拦截
MIN_CORE_RERANK_SCORE = _env_float("MIN_CORE_RERANK_SCORE", 0.20)
# 词法逃生通道收紧（2026-08-02 拒答校准）：
# 低于主门槛的 chunk 仅当“词法命中数 ≥ MIN_LEXICAL_SCORE 且 rerank ≥ MIN_LEXICAL_RERANK_SCORE”才放行，
# 拦截“银行卡密码”类（rerank 0.004-0.03，仅关键词命中）与“爱弹吉他”类（语义相邻但无引号锚点）跑题内容
MIN_LEXICAL_SCORE = _env_float("MIN_LEXICAL_SCORE", 2.0)
MIN_LEXICAL_RERANK_SCORE = _env_float("MIN_LEXICAL_RERANK_SCORE", 0.05)
DOCUMENT_SNAPSHOT_CACHE_TTL_SECONDS = _env_float(
    "DOCUMENT_SNAPSHOT_CACHE_TTL_SECONDS", 600.0, minimum=1.0
)
DOCUMENT_SNAPSHOT_CACHE_MAX_DOCUMENTS = _env_int(
    "DOCUMENT_SNAPSHOT_CACHE_MAX_DOCUMENTS", 12, minimum=1
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai_compatible").strip() or "openai_compatible"
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
LLM_BASE_URL = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("DEEPSEEK_BASE_URL")
    or "https://api.deepseek.com"
)
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_INCLUDE_THINKING = _env_bool("LLM_INCLUDE_THINKING", False)
LLM_RESPONSE_FORMAT = os.getenv("LLM_RESPONSE_FORMAT", "json_object").strip() or "json_object"
LLM_STREAM = _env_bool("LLM_STREAM", True)

QUERY_PLANNER_ENABLED = _env_bool("QUERY_PLANNER_ENABLED", True)
QUERY_PLANNER_PROVIDER = os.getenv("QUERY_PLANNER_PROVIDER", LLM_PROVIDER)
QUERY_PLANNER_API_KEY = os.getenv("QUERY_PLANNER_API_KEY") or LLM_API_KEY
QUERY_PLANNER_BASE_URL = os.getenv("QUERY_PLANNER_BASE_URL", LLM_BASE_URL)
QUERY_PLANNER_MODEL = os.getenv("QUERY_PLANNER_MODEL", LLM_MODEL)
QUERY_PLANNER_TIMEOUT_SECONDS = _env_float("QUERY_PLANNER_TIMEOUT_SECONDS", 20.0, minimum=0.1)
QUERY_PLANNER_MAX_ASPECTS = _env_int("QUERY_PLANNER_MAX_ASPECTS", 12, minimum=1)
QUERY_PLANNER_MAX_SEARCH_QUERIES = _env_int("QUERY_PLANNER_MAX_SEARCH_QUERIES", 3, minimum=1)
QUERY_PLANNER_INCLUDE_THINKING = _env_bool("QUERY_PLANNER_INCLUDE_THINKING", LLM_INCLUDE_THINKING)
QUERY_PLANNER_RESPONSE_FORMAT = os.getenv("QUERY_PLANNER_RESPONSE_FORMAT", LLM_RESPONSE_FORMAT).strip() or LLM_RESPONSE_FORMAT

ANSWER_GENERATION_ENABLED = _env_bool("ANSWER_GENERATION_ENABLED", True)
ANSWER_GENERATION_PROVIDER = os.getenv("ANSWER_GENERATION_PROVIDER", LLM_PROVIDER)
ANSWER_GENERATION_API_KEY = os.getenv("ANSWER_GENERATION_API_KEY") or LLM_API_KEY
ANSWER_GENERATION_BASE_URL = os.getenv("ANSWER_GENERATION_BASE_URL", LLM_BASE_URL)
ANSWER_GENERATION_MODEL = os.getenv("ANSWER_GENERATION_MODEL", LLM_MODEL)
ANSWER_GENERATION_TIMEOUT_SECONDS = _env_float("ANSWER_GENERATION_TIMEOUT_SECONDS", 18.0, minimum=0.1)
ANSWER_GENERATION_MAX_TOKENS = _env_int("ANSWER_GENERATION_MAX_TOKENS", 900, minimum=1)
ANSWER_GENERATION_INCLUDE_THINKING = _env_bool("ANSWER_GENERATION_INCLUDE_THINKING", LLM_INCLUDE_THINKING)
ANSWER_GENERATION_RESPONSE_FORMAT = os.getenv("ANSWER_GENERATION_RESPONSE_FORMAT", LLM_RESPONSE_FORMAT).strip() or LLM_RESPONSE_FORMAT
ANSWER_GENERATION_STREAM = _env_bool("ANSWER_GENERATION_STREAM", LLM_STREAM)

# ---- 简历面试场景：意图路由 / 查询改写 / 多轮记忆 / 置信度分级 ----
INTENT_ROUTER_ENABLED = _env_bool("INTENT_ROUTER_ENABLED", True)
INTENT_ROUTER_PROVIDER = os.getenv("INTENT_ROUTER_PROVIDER", LLM_PROVIDER)
INTENT_ROUTER_API_KEY = os.getenv("INTENT_ROUTER_API_KEY") or LLM_API_KEY
INTENT_ROUTER_BASE_URL = os.getenv("INTENT_ROUTER_BASE_URL", LLM_BASE_URL)
INTENT_ROUTER_MODEL = os.getenv("INTENT_ROUTER_MODEL", LLM_MODEL)
INTENT_ROUTER_TIMEOUT_SECONDS = _env_float("INTENT_ROUTER_TIMEOUT_SECONDS", 8.0, minimum=0.1)
INTENT_ROUTER_MAX_TOKENS = _env_int("INTENT_ROUTER_MAX_TOKENS", 300, minimum=1)
INTENT_ROUTER_RESPONSE_FORMAT = os.getenv("INTENT_ROUTER_RESPONSE_FORMAT", "json_object").strip() or "json_object"

REWRITE_ENABLED = _env_bool("REWRITE_ENABLED", True)
REWRITE_MAX_QUERIES = _env_int("REWRITE_MAX_QUERIES", 3, minimum=1)

# ---- fast path（2026-08-08）：普通独立问题减少 LLM 调用 ----# 简历面试的大多数问题是"明确、独立、单对象"的普通提问：逐个走
# 「意图 LLM → 规划 LLM → 生成 LLM」成本高（DeepSeek API 延迟主导）。
# 开启后：无上一轮对话 + 无指代/追问标记时，意图层用极简规则跳过 LLM；
# 非枚举/非补集/无拆解需求的普通问题，规划层直接构造单方面检索跳过 LLM。
# 追问（指代消解）、枚举/补集、复杂问题仍走完整链路。两开关可独立关闭回退。
INTENT_FAST_PATH_ENABLED = _env_bool("INTENT_FAST_PATH_ENABLED", True)
PLANNER_FAST_PATH_ENABLED = _env_bool("PLANNER_FAST_PATH_ENABLED", True)

CONVERSATION_MEMORY_ENABLED = _env_bool("CONVERSATION_MEMORY_ENABLED", True)
CONVERSATION_MEMORY_MAX_TURNS = _env_int("CONVERSATION_MEMORY_MAX_TURNS", 8, minimum=1)
CONVERSATION_MEMORY_TTL_HOURS = _env_float("CONVERSATION_MEMORY_TTL_HOURS", 24.0, minimum=0.1)

# ---- grounding 确定性硬事实校验（2026-08-08）----
# 生成后校验答案中的数字/日期/书名号专名是否能在检索证据中找到；
# LLM 自评 sufficient 但硬事实校验失败 → 强制降级 hedged（防"自己生成、自己判断"盲区）。
GROUNDING_VERIFY_ENABLED = _env_bool("GROUNDING_VERIFY_ENABLED", True)

# ---- 访客问答访问码闸（2026-08-04）：看过简历的面试官凭访问码提问 ----
# 简历/分享说明中附带访问码，访客输入一次后签发短期 JWT 存 httpOnly cookie；
# QA_ACCESS_CODE 为空 = 闸门关闭（开发/测试默认），部署时在 .env 设置。
# 访问码必须为 6 位数字+大写英文字母（如 A7K2M9）：印在简历上不便更换，
# 高熵码 + 输错阶梯锁定（qa_access_guard）配合抵御暴力破解。格式不合法启动即报错。
QA_ACCESS_CODE = os.getenv("QA_ACCESS_CODE", "").strip().upper()
if QA_ACCESS_CODE and not re.fullmatch(r"[0-9A-Z]{6}", QA_ACCESS_CODE):
    raise RuntimeError(
        "QA_ACCESS_CODE 必须为空或 6 位数字+大写英文字母（如 A7K2M9）；"
        f"当前值 {QA_ACCESS_CODE!r} 不合法。"
    )
QA_ACCESS_TOKEN_TTL_HOURS = _env_float("QA_ACCESS_TOKEN_TTL_HOURS", 24.0, minimum=0.1)
# 访问码 cookie 的 Secure 标记：HTTPS 部署（Cloudflare Tunnel）时置 true
COOKIE_SECURE = _env_bool("COOKIE_SECURE", False)
# ---- 全局每日提问预算（跨 IP 保险丝，防换 IP 刷爆 DeepSeek 额度）----
QA_GLOBAL_DAILY_LIMIT = _env_int("QA_GLOBAL_DAILY_LIMIT", 300, minimum=1)
# 预算告警：剩余 ≤ max(该值, 预算×比例) 时前端弹窗提醒"今日预算即将超限"
QA_BUDGET_WARNING_REMAINING = _env_int("QA_BUDGET_WARNING_REMAINING", 30, minimum=0)
QA_BUDGET_WARNING_RATIO = _env_float("QA_BUDGET_WARNING_RATIO", 0.15, minimum=0.0)

# ---- 检索兜底链（P3）：证据不足时不拒答，逐级放宽 ----
FALLBACK_LOWER_THRESHOLD_ENABLED = _env_bool("FALLBACK_LOWER_THRESHOLD_ENABLED", True)
# 面试问答优先忠实性：默认不为弱证据再触发一轮 LLM+检索，也不在无证据时虚构个人经历。
FALLBACK_REWRITE_RETRY_ENABLED = _env_bool("FALLBACK_REWRITE_RETRY_ENABLED", False)
FALLBACK_DIRECT_GENERATION_ENABLED = _env_bool("FALLBACK_DIRECT_GENERATION_ENABLED", False)
# 降阈重试：rerank 分数低于该值视为"弱证据"；重试时不过滤直接取 top-N
RELAXED_RERANK_THRESHOLD = _env_float("RELAXED_RERANK_THRESHOLD", 0.05)
RELAXED_MIN_PROMPT_CHUNKS = _env_int("RELAXED_MIN_PROMPT_CHUNKS", 6, minimum=1)

# ---- 速度分级链路（2026-08-14）：跳重排 + 单问硬时间预算 ----
# 重排是 2C4G 热路径（每次约 8-9s）。分级策略：融合分分差足够大（实体明确的单对象问题）
# 或关键词精确命中锚定时跳过重排，只有歧义问题才重排；多角度问题合并后只重排一次。
SKIP_RERANK_ENABLED = _env_bool("SKIP_RERANK_ENABLED", True)
# top1 与 top2 融合分的分差比例（(top1-top2)/top1）达到该值 → 排序决定性，跳过重排。
# 2026-08-14 评测校准为 0.15（0.40/0.25 均过于保守，重排 p50 仍 4s+）；
# 融合分是 RRF 加权累加的小数值，分差天然偏大，0.15 为保守起步值。
SKIP_RERANK_MARGIN_RATIO = _env_float("SKIP_RERANK_MARGIN_RATIO", 0.15, minimum=0.0)
# top1 融合分绝对下限：低于该值的候选池信号不够强，不允许跳重排
SKIP_RERANK_TOP_FUSION_MIN = _env_float("SKIP_RERANK_TOP_FUSION_MIN", 0.015, minimum=0.0)
# top1 候选的关键词精确命中数达到该值 → 词面锚定强信号，跳过重排。
# 2026-08-14 起召回术语已扩充（锚点+专名），1 次精确命中即可信
SKIP_RERANK_MIN_KEYWORD_HITS = _env_int("SKIP_RERANK_MIN_KEYWORD_HITS", 1, minimum=1)
# 单问硬时间预算（秒，自意图路由起算）：重排前剩余预算不足即跳过重排，
# 生成前不足即跳过 LLM 生成改用摘录兜底。P95 ≤ 6s 目标的总预算上限；
# 2026-08-14 评测校准为 10（跳重排生效后重排 ~0-3s，给生成留足 5s+）。
QA_HARD_BUDGET_SECONDS = _env_float("QA_HARD_BUDGET_SECONDS", 10.0, minimum=0.5)
# 重排预估成本（秒）：剩余预算低于该值时跳过重排，按融合分排序直接输出
RERANK_BUDGET_SECONDS = _env_float("RERANK_BUDGET_SECONDS", 3.0, minimum=0.1)
# LLM 规划的最小预算（秒）：剩余预算低于该值时跳过规划 LLM，用零 LLM 单方面计划
PLANNER_MIN_BUDGET_SECONDS = _env_float("PLANNER_MIN_BUDGET_SECONDS", 1.5, minimum=0.1)
# LLM 生成的最小预算（秒）：剩余预算低于该值时跳过生成，摘录知识库原文（hedged）
GENERATION_MIN_BUDGET_SECONDS = _env_float("GENERATION_MIN_BUDGET_SECONDS", 1.5, minimum=0.1)

# ---- 负载指示灯（公开状态 /api/qa/status 的 load 字段，2026-08-12）----
# 按 2C4G 校准（QA_TASK_WORKERS=2，任务管线双 worker 并行）：1-2 人提问绿、3 人黄、4 人及以上红。
# in_flight = 运行中 + 排队中的问答任务数；CPU 比 = 进程 CPU% /（核数 × 100），
# 取最近 30s 均值平滑（rerank 突刺不误报）；内存仅作红色兜底（防 OOM），不参与黄色。
LOAD_YELLOW_CPU_RATIO = _env_float("LOAD_YELLOW_CPU_RATIO", 0.70)
LOAD_RED_CPU_RATIO = _env_float("LOAD_RED_CPU_RATIO", 0.90)
LOAD_RED_MEM_RATIO = _env_float("LOAD_RED_MEM_RATIO", 0.90)
LOAD_YELLOW_INFLIGHT = _env_int("LOAD_YELLOW_INFLIGHT", 3, minimum=1)
LOAD_RED_INFLIGHT = _env_int("LOAD_RED_INFLIGHT", 4, minimum=1)
# 内存基准兜底：容器内优先读 cgroup 上限；裸机/本地读不到时按服务器物理内存估算，再兜底该值
LOAD_MEMORY_REFERENCE_BYTES = _env_int("LOAD_MEMORY_REFERENCE_BYTES", 4 * 1024**3, minimum=1)

MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 50 * 1024 * 1024, minimum=1)
MAX_BATCH_UPLOAD_FILES = _env_int("MAX_BATCH_UPLOAD_FILES", 20, minimum=1)
INDEX_QUEUE_CAPACITY = _env_int("INDEX_QUEUE_CAPACITY", 8, minimum=1)
INDEX_TASK_MAX_RETRIES = _env_int("INDEX_TASK_MAX_RETRIES", 3)
QA_QUEUE_CAPACITY = _env_int("QA_QUEUE_CAPACITY", 16, minimum=1)
QA_TASK_MAX_RETRIES = _env_int("QA_TASK_MAX_RETRIES", 1)
# 问答任务并行 worker 数（2026-08-12 起 2C4G 档默认 2）：
# 每问约 90% 时间是外部 LLM API，本地推理仅 1-2s（全局锁串行）——
# 双 worker 让 2 人同时提问各自并行处理，第 3 人排队；设 1 回退旧单 worker 行为。
QA_TASK_WORKERS = _env_int("QA_TASK_WORKERS", 2, minimum=1)

# ---- 问答答案缓存（2026-08-12）：面试高频问题相似语义复用 ----
# 两层判定：① 问题归一化后完全相等（精确命中，最安全）；② embedding 余弦
# top-1 ≥ QA_CACHE_SEMANTIC_THRESHOLD（默认 0.93 保守——"语义相似"与"答案可复用"
# 不同，阈值宁高勿低防张冠李戴）。只缓存独立问题（无 session）的 answered+sufficient
# 答案；知识库变更（上传/删除/重建）时整体清空。
QA_CACHE_ENABLED = _env_bool("QA_CACHE_ENABLED", True)
QA_CACHE_SEMANTIC_THRESHOLD = _env_float("QA_CACHE_SEMANTIC_THRESHOLD", 0.93)
QA_CACHE_MAX_ITEMS = _env_int("QA_CACHE_MAX_ITEMS", 300, minimum=1)
MAX_OOXML_ENTRIES = _env_int("MAX_OOXML_ENTRIES", 20_000, minimum=1)
MAX_OOXML_UNCOMPRESSED_BYTES = _env_int(
    "MAX_OOXML_UNCOMPRESSED_BYTES", 500 * 1024 * 1024, minimum=1
)
OFFICE_CONVERSION_TIMEOUT_SECONDS = _env_int("OFFICE_CONVERSION_TIMEOUT_SECONDS", 120, minimum=1)
OFFICE_CONVERSION_MAX_BYTES = _env_int(
    "OFFICE_CONVERSION_MAX_BYTES", 200 * 1024 * 1024, minimum=1
)

# ---- 人物工坊（2026-08-14）：任意简历上传 → LLM 加工为检索友好 Markdown ----
# 加工师 LLM 配置（默认复用主 LLM）；转换按批调用，受全局每日预算与超时约束。
WORKSHOP_ENABLED = _env_bool("WORKSHOP_ENABLED", True)
WORKSHOP_PROVIDER = os.getenv("WORKSHOP_PROVIDER", LLM_PROVIDER)
WORKSHOP_API_KEY = os.getenv("WORKSHOP_API_KEY") or LLM_API_KEY
WORKSHOP_BASE_URL = os.getenv("WORKSHOP_BASE_URL", LLM_BASE_URL)
WORKSHOP_MODEL = os.getenv("WORKSHOP_MODEL", LLM_MODEL)
WORKSHOP_TIMEOUT_SECONDS = _env_float("WORKSHOP_TIMEOUT_SECONDS", 60.0, minimum=1.0)
# 单次转换最多输入字符数（超出按此分批）；单任务最多文件数
WORKSHOP_MAX_INPUT_CHARS = _env_int("WORKSHOP_MAX_INPUT_CHARS", 12_000, minimum=500)
WORKSHOP_MAX_FILES_PER_JOB = _env_int("WORKSHOP_MAX_FILES_PER_JOB", 10, minimum=1)
# 加工 skill 目录（规范驱动，2026-08-14）：加工提示词 / 输出 JSON 契约 / 版本
# 以 .agents/skills/resume-materials-workshop/ 为单一事实来源，后端运行时读取；
# Docker 镜像内通过 ENV 指向 /app/agents-skills/...，本地默认仓库内路径。
WORKSHOP_SKILL_DIR = os.getenv(
    "WORKSHOP_SKILL_DIR",
    str(PROJECT_ROOT / ".agents" / "skills" / "resume-materials-workshop"),
)
# Skill 注册表根目录（2026-08-15）：扫描 <dir>/*/SKILL.md 自动发现全部 skill；
# 默认取工坊 skill 目录的父目录（本地 .agents/skills，Docker /app/agents-skills），
# 与 WORKSHOP_SKILL_DIR 单点指向兼容。
SKILLS_DIR = os.getenv("SKILLS_DIR", str(Path(WORKSHOP_SKILL_DIR).parent))

# 默认人物姓名（2026-08-14）：部署者自行配置；留空 = 不注入姓名，
# 提示词与文案使用中性"我/求职者"表述（系统绝不内置任何人的姓名）
DEFAULT_PERSONA_NAME = os.getenv("DEFAULT_PERSONA_NAME", "").strip()


SUPPORTED_DOCUMENT_MIME_TYPES = {    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/plain"},
    ".doc": {"application/msword", "application/octet-stream"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ".pdf": {"application/pdf"},
    ".jsonl": {"application/x-ndjson", "application/jsonl", "application/json", "text/plain"},
    ".html": {"text/html", "application/xhtml+xml"},
    ".htm": {"text/html", "application/xhtml+xml"},
}

DOCUMENT_LOADER_ORDER = {
    ".txt": ["text", "unstructured"],
    ".md": ["markdown", "unstructured"],
    ".doc": ["libreoffice-doc", "antiword-doc"],
    ".docx": ["python-docx", "docling", "unstructured"],
    ".pdf": ["pymupdf4llm", "docling", "unstructured", "pypdf"],
    ".jsonl": ["jsonl"],
    ".html": ["html"],
    ".htm": ["html"],
}


# ---------------------------------------------------------------------------
# 认证与限流（前台/后台权限分离）
# ---------------------------------------------------------------------------
# 管理员密码必须通过环境变量提供；缺失时由 security 模块导入期 raise（fail-closed）。
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
# 可选：JWT 签名密钥。为空时用 sha256(ADMIN_PASSWORD) 派生（改密码即吊销全部旧 token）。
ADMIN_JWT_SECRET = os.getenv("ADMIN_JWT_SECRET", "")
ADMIN_TOKEN_EXPIRY_HOURS = _env_int("ADMIN_TOKEN_EXPIRY_HOURS", 12, minimum=1)
# 开发期 kill-switch；生产部署必须保持 true。
AUTH_REQUIRED = _env_bool("AUTH_REQUIRED", True)

RATE_LIMIT_ENABLED = _env_bool("RATE_LIMIT_ENABLED", True)
# 问答接口（/api/qa/*）每 IP 限流：每分钟次数（0 = 不限制，仅保留全局并发）。
# 提问总量改用 QA_IP_MAX_QUESTIONS 按 qa_logs 累计计数（不按天，防拿到访问码的人刷）。
QA_IP_RATE_LIMIT_PER_MINUTE = _env_int("QA_IP_RATE_LIMIT_PER_MINUTE", 30, minimum=0)
# 每个 IP 累计可提问的总数上限（写进 qa_logs 即消耗 1 次，含缓存命中与寒暄/无关转移）；
# 用尽后该 IP 提问返回 429，调大此值或清理 qa_logs 恢复。管理员请求不受限。
QA_IP_MAX_QUESTIONS = _env_int("QA_IP_MAX_QUESTIONS", 20, minimum=1)
# 问题长度上限（字符数）：超限返回 400。schema 的 max_length=8000 保留为最后防线——
# schema 校验失败返回 422，API 层显式检查才能返回约定的 400。
QA_MAX_QUESTION_CHARS = _env_int("QA_MAX_QUESTION_CHARS", 500, minimum=1)
# 全局同时执行问答（跑 LLM 的路径）的最大并发数；2C4G 保守稳定档建议 2。
QA_GLOBAL_CONCURRENCY = _env_int("QA_GLOBAL_CONCURRENCY", 2, minimum=1)
LOGIN_RATE_LIMIT_PER_MINUTE = _env_int("LOGIN_RATE_LIMIT_PER_MINUTE", 10, minimum=1)
# 部署在可信反向代理（Cloudflare Tunnel）后置 true，用 X-Forwarded-For 首段作为 IP。
RATE_LIMIT_TRUST_PROXY = _env_bool("RATE_LIMIT_TRUST_PROXY", False)


def ensure_runtime_dirs() -> None:
    """Create local runtime directories without downloading models."""

    for path in [
        DATA_DIR,
        DOCUMENT_DIR,
        QDRANT_STORAGE_DIR,
        AUDIT_ARCHIVE_DIR,
        MODEL_DIR,
        MODEL_CACHE_DIR,
        HF_HOME,
        HF_HUB_CACHE,
        SENTENCE_TRANSFORMERS_HOME,
        TORCH_HOME,
    ]:
        path.mkdir(parents=True, exist_ok=True)
