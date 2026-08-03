import os
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
DEFAULT_EMBEDDING_MODEL_DIR = MODEL_DIR / "bge-base-zh-v1.5"
DEFAULT_RERANKER_MODEL_DIR = MODEL_DIR / "bge-reranker-base"
HF_HOME = Path(os.getenv("HF_HOME") or MODEL_CACHE_DIR / "huggingface")
HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE") or HF_HOME / "hub")
SENTENCE_TRANSFORMERS_HOME = Path(
    os.getenv("SENTENCE_TRANSFORMERS_HOME") or MODEL_CACHE_DIR / "sentence_transformers"
)
TORCH_HOME = Path(os.getenv("TORCH_HOME") or MODEL_CACHE_DIR / "torch")
RESUME_OFFLINE_MODE = _env_bool("RESUME_OFFLINE_MODE", True)
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH")
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
    {"background", "lazy"},
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

INDEX_VERSION = "bge-base-zh-dense-v5-resume"
# 直跑后端默认连 RESUME_QDRANT_HTTP_PORT 映射的 qdrant（本地 .env 为 16333，服务器默认 6333）；
# docker-compose 内显式传 QDRANT_URL=http://qdrant:6333 覆盖此默认值
RESUME_QDRANT_HTTP_PORT = int(os.getenv("RESUME_QDRANT_HTTP_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL", f"http://127.0.0.1:{RESUME_QDRANT_HTTP_PORT}")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "resumemind_chunks")
QDRANT_AUTO_CREATE_COLLECTION = _env_bool("QDRANT_AUTO_CREATE_COLLECTION", True)
QDRANT_UPSERT_BATCH_SIZE = _env_int("QDRANT_UPSERT_BATCH_SIZE", 128, minimum=1)
QDRANT_DENSE_VECTOR_NAME = "dense"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-zh-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
EMBEDDING_DIMENSION = _env_int("EMBEDDING_DIMENSION", 768, minimum=1)
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 8, minimum=1)
EMBEDDING_MAX_BATCH_SIZE = _env_int("EMBEDDING_MAX_BATCH_SIZE", 16, minimum=1)
RETRIEVAL_TOP_K = 50
RERANK_TOP_K = 20
RERANK_CANDIDATE_LIMIT = 24
MAX_PROMPT_CHUNKS = 12
MAX_PROMPT_TOKENS = _env_int("MAX_PROMPT_TOKENS", 3600, minimum=1)
MIN_PROMPT_CHUNKS = _env_int("MIN_PROMPT_CHUNKS", 6, minimum=0)
FORCE_MIN_CHUNKS = _env_bool("FORCE_MIN_CHUNKS", True)
RERANK_PROMPT_THRESHOLD = _env_float("RERANK_PROMPT_THRESHOLD", 0.20)
RELATIVE_SCORE_RATIO = _env_float("RELATIVE_SCORE_RATIO", 0.40)
FINAL_CITATION_LIMIT = MAX_PROMPT_CHUNKS
MIN_RERANK_SCORE = _env_float("MIN_RERANK_SCORE", 0.30)
MIN_EVIDENCE_COVERAGE = _env_float("MIN_EVIDENCE_COVERAGE", 0.25)
DIRECT_EVIDENCE_COVERAGE = _env_float("DIRECT_EVIDENCE_COVERAGE", 0.45)
# 强语义匹配的 rerank 分数下限：bge-reranker-base 的 normalize 分数分布为 0.001-0.03，
# 旧模型（v2-m3）分布是 0-1，此阈值须按实际模型分布调整
STRONG_SEMANTIC_RERANK_SCORE = _env_float("STRONG_SEMANTIC_RERANK_SCORE", 0.02)
# core 选择的绝对相关性下限：实际部署中 rerank 分数为 0-1 分布（跑题≈0.012、真实匹配≥0.48），
# 而 RERANK_PROMPT_THRESHOLD 可能被调得很低（如 0.01），取两者最大值保证跑题问题必被拦截
MIN_CORE_RERANK_SCORE = _env_float("MIN_CORE_RERANK_SCORE", 0.1)
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

CONVERSATION_MEMORY_ENABLED = _env_bool("CONVERSATION_MEMORY_ENABLED", True)
CONVERSATION_MEMORY_MAX_TURNS = _env_int("CONVERSATION_MEMORY_MAX_TURNS", 8, minimum=1)
CONVERSATION_MEMORY_TTL_HOURS = _env_float("CONVERSATION_MEMORY_TTL_HOURS", 24.0, minimum=0.1)

# 推测标注前缀（证据不足时后端强制给回答加的前缀，保证措辞统一）
HEDGE_PREFIX = os.getenv("HEDGE_PREFIX", "根据现有知识库推测").strip() or "根据现有知识库推测"

# ---- 检索兜底链（P3）：证据不足时不拒答，逐级放宽 ----
FALLBACK_LOWER_THRESHOLD_ENABLED = _env_bool("FALLBACK_LOWER_THRESHOLD_ENABLED", True)
FALLBACK_REWRITE_RETRY_ENABLED = _env_bool("FALLBACK_REWRITE_RETRY_ENABLED", True)
FALLBACK_DIRECT_GENERATION_ENABLED = _env_bool("FALLBACK_DIRECT_GENERATION_ENABLED", True)
# 降阈重试：rerank 分数低于该值视为"弱证据"；重试时不过滤直接取 top-N
RELAXED_RERANK_THRESHOLD = _env_float("RELAXED_RERANK_THRESHOLD", 0.05)
RELAXED_MIN_PROMPT_CHUNKS = _env_int("RELAXED_MIN_PROMPT_CHUNKS", 6, minimum=1)

MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 50 * 1024 * 1024, minimum=1)
MAX_BATCH_UPLOAD_FILES = _env_int("MAX_BATCH_UPLOAD_FILES", 20, minimum=1)
INDEX_QUEUE_CAPACITY = _env_int("INDEX_QUEUE_CAPACITY", 8, minimum=1)
INDEX_TASK_MAX_RETRIES = _env_int("INDEX_TASK_MAX_RETRIES", 3)
QA_QUEUE_CAPACITY = _env_int("QA_QUEUE_CAPACITY", 16, minimum=1)
QA_TASK_MAX_RETRIES = _env_int("QA_TASK_MAX_RETRIES", 1)
MAX_OOXML_ENTRIES = _env_int("MAX_OOXML_ENTRIES", 20_000, minimum=1)
MAX_OOXML_UNCOMPRESSED_BYTES = _env_int(
    "MAX_OOXML_UNCOMPRESSED_BYTES", 500 * 1024 * 1024, minimum=1
)
OFFICE_CONVERSION_TIMEOUT_SECONDS = _env_int("OFFICE_CONVERSION_TIMEOUT_SECONDS", 120, minimum=1)
OFFICE_CONVERSION_MAX_BYTES = _env_int(
    "OFFICE_CONVERSION_MAX_BYTES", 200 * 1024 * 1024, minimum=1
)

SUPPORTED_DOCUMENT_MIME_TYPES = {
    ".txt": {"text/plain"},
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
# 问答接口（/api/qa/*）每 IP 限流：每分钟次数 + 每日次数。
QA_IP_RATE_LIMIT_PER_MINUTE = _env_int("QA_IP_RATE_LIMIT_PER_MINUTE", 30, minimum=1)
QA_IP_DAILY_LIMIT = _env_int("QA_IP_DAILY_LIMIT", 500, minimum=1)
# 全局同时执行问答（跑 LLM 的路径）的最大并发数；2C4G 建议保持 4。
QA_GLOBAL_CONCURRENCY = _env_int("QA_GLOBAL_CONCURRENCY", 4, minimum=1)
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
