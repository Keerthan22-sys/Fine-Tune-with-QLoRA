"""
Indian Tax Law AI — Production FastAPI Application
===================================================
Day 5 of the Upscale_with_AI journey.

Stack:
- Fine-tuned Llama 3.1 8B (Day 2: SFT on Indian Tax Law)
- DPO-aligned for CA-quality responses (Day 3)
- RAG over Income Tax Act 1961 + GST Acts (Day 4)
- FastAPI + Uvicorn serving layer (Day 5)

Run locally:
  pip install fastapi uvicorn slowapi pydantic chromadb
               sentence-transformers rank-bm25 pymupdf unsloth

  export TAX_API_KEYS="dev-key-123,prod-key-456"
  export MODEL_PATH="./indian-tax-expert-lora"
  export CHROMA_PATH="./chroma_db"
  export ENV="development"

  uvicorn day5-fastapi-app:app --host 0.0.0.0 --port 8000 --reload

Deploy to HuggingFace Spaces:
  - Docker SDK Space
  - See Dockerfile comments at bottom of this file

Disclaimer: This API provides information for educational purposes only.
It is NOT a substitute for advice from a qualified Chartered Accountant.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
import time
import hashlib
import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

# ─── Third-Party ──────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import torch

# ─── Logging Setup ────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": %(message)s}',
    datefmt="%Y-%m-%dT%H:%M:%SZ"
)
logger = logging.getLogger("indian-tax-api")

# ─── Configuration ────────────────────────────────────────────────────────────
ENV             = os.getenv("ENV", "development")
MODEL_PATH      = os.getenv("MODEL_PATH", "./indian-tax-expert-lora")
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.1"))
TOP_K_CHUNKS    = int(os.getenv("TOP_K_CHUNKS", "4"))

# API Keys — comma-separated list in environment variable
_raw_keys = os.getenv("TAX_API_KEYS", "dev-key-123")
VALID_API_KEYS: set[str] = {k.strip() for k in _raw_keys.split(",") if k.strip()}

LEGAL_DISCLAIMER = (
    "⚠️ This response is for informational and educational purposes only. "
    "It does not constitute legal or tax advice. Always verify information "
    "against the current Income Tax Act, GST Acts, and official CBDT notifications. "
    "For decisions involving tax liability, consult a qualified Chartered Accountant (CA)."
)

MODEL_VERSION = "llama-3.1-8b-indian-tax-dpo-v1"

# ─── In-memory session history (use Redis in production) ──────────────────────
session_history: dict[str, list[dict]] = {}


# ════════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ════════════════════════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The Indian tax law question to answer",
        example="What is the TDS rate on professional fees under Section 194J?",
    )
    context: Optional[str] = Field(
        None,
        max_length=500,
        description="Additional context (e.g., payment amount, type of taxpayer)",
        example="Payment of ₹45,000 to a freelance Chartered Accountant.",
    )
    session_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Optional session ID for conversation history",
    )

    @validator("question")
    def sanitize_question(cls, v):
        # Strip potential prompt injection attempts
        forbidden = ["ignore previous", "system:", "<<SYS>>", "[INST]"]
        v_lower = v.lower()
        for phrase in forbidden:
            if phrase in v_lower:
                raise ValueError("Invalid question content detected.")
        return v.strip()


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved_sections: list[str]
    confidence: float
    rag_used: bool
    model_version: str
    disclaimer: str
    latency_ms: int
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    rag_index_loaded: bool
    gpu_available: bool
    vram_used_gb: Optional[float]
    environment: str
    version: str


class HistoryItem(BaseModel):
    timestamp: str
    question: str
    answer: str
    latency_ms: int


# ════════════════════════════════════════════════════════════════════════════════
# MODEL & RAG STATE (loaded once on startup)
# ════════════════════════════════════════════════════════════════════════════════

class AppState:
    """Holds model, tokenizer, and RAG index — loaded once at startup."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.rag_query_engine = None
        self.model_loaded = False
        self.rag_loaded = False


app_state = AppState()

ALPACA_PROMPT = """Below is an instruction that describes a tax law question. \
Write a response that accurately and completely answers the question.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def load_model():
    """Load fine-tuned + DPO-aligned Llama 3.1 8B model."""
    logger.info('"Loading fine-tuned Indian Tax model..."')
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        logger.info('"Model loaded successfully"')
        return model, tokenizer

    except ImportError:
        # Fallback: load via transformers directly (slower, no Unsloth optimisation)
        logger.warning('"Unsloth not available — using HuggingFace transformers directly"')
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model.eval()
        return model, tokenizer

    except Exception as e:
        logger.error(f'"Failed to load model: {str(e)}"')
        return None, None


def load_rag_index():
    """Load ChromaDB vector index + BM25 for hybrid retrieval."""
    logger.info('"Loading RAG index (Income Tax Act + GST Acts)..."')
    try:
        import chromadb
        from llama_index.core import StorageContext, load_index_from_storage
        from llama_index.vector_stores.chroma import ChromaVectorStore

        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        chroma_collection = chroma_client.get_or_create_collection("indian_tax_law")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine(similarity_top_k=TOP_K_CHUNKS)
        logger.info('"RAG index loaded successfully"')
        return query_engine
    except Exception as e:
        logger.warning(f'"RAG index not available: {str(e)}. Falling back to model-only."')
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + RAG index once on startup, release on shutdown."""
    logger.info('"=== Indian Tax API Starting Up ==="')
    startup_start = time.time()

    app_state.model, app_state.tokenizer = load_model()
    app_state.model_loaded = app_state.model is not None

    app_state.rag_query_engine = load_rag_index()
    app_state.rag_loaded = app_state.rag_query_engine is not None

    startup_elapsed = time.time() - startup_start
    logger.info(f'"Startup complete in {startup_elapsed:.1f}s | model={app_state.model_loaded} | rag={app_state.rag_loaded}"')

    yield  # App is running

    # Shutdown
    logger.info('"=== Indian Tax API Shutting Down ==="')
    if app_state.model is not None:
        del app_state.model
        del app_state.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════════

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Indian Tax Law AI Assistant",
    description=(
        "A production API powered by a fine-tuned Llama 3.1 8B model, "
        "DPO-aligned for CA-quality responses, with RAG over the Income Tax Act 1961 and GST Acts. "
        "For educational purposes only — not a substitute for CA advice."
    ),
    version="1.0.0",
    docs_url="/docs" if ENV != "production" else None,  # Disable docs in prod
    redoc_url="/redoc" if ENV != "production" else None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ENV == "development" else [
        "https://your-frontend.com",
        "https://your-space.hf.space",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ════════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ════════════════════════════════════════════════════════════════════════════════

security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Validate Bearer token API key. Returns the key on success."""
    token = credentials.credentials
    if token not in VALID_API_KEYS:
        logger.warning(f'"Invalid API key attempt: {token[:8]}..."')
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key. Obtain a valid key to use this service.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# ════════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ════════════════════════════════════════════════════════════════════════════════

def run_rag_retrieval(question: str) -> tuple[str, list[str]]:
    """Retrieve relevant sections from the Indian Tax Law corpus."""
    if not app_state.rag_loaded:
        return "", []

    try:
        response = app_state.rag_query_engine.query(question)
        context_text = str(response)

        # Extract source node metadata for citations
        source_sections = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                metadata = node.node.metadata
                section = metadata.get("section_header", metadata.get("file_name", "Unknown"))
                if section not in source_sections:
                    source_sections.append(section)

        return context_text, source_sections

    except Exception as e:
        logger.error(f'"RAG retrieval error: {str(e)}"')
        return "", []


def generate_answer(
    question: str,
    context: str = "",
    retrieved_context: str = "",
) -> tuple[str, float]:
    """
    Generate answer using the fine-tuned model.
    Returns (answer_text, confidence_score).
    """
    if not app_state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later.",
        )

    # Build the instruction with RAG context if available
    if retrieved_context:
        instruction = (
            f"Based on the following provisions of Indian tax law:\n\n"
            f"{retrieved_context}\n\n"
            f"Answer this question accurately and completely: {question}"
        )
    else:
        instruction = question

    input_text = context if context else "(No additional context provided)"

    prompt = ALPACA_PROMPT.format(instruction=instruction, input=input_text)

    try:
        inputs = app_state.tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = app_state.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=0.9,
                do_sample=True,
                pad_token_id=app_state.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode answer
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        answer = app_state.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Simple confidence heuristic: based on sequence probability
        # In production, use a proper calibration layer
        if outputs.scores:
            import torch.nn.functional as F
            avg_logprob = sum(
                F.log_softmax(score, dim=-1).max().item()
                for score in outputs.scores
            ) / len(outputs.scores)
            # Map logprob (-∞ to 0) to confidence (0 to 1)
            confidence = float(min(1.0, max(0.0, 1.0 + avg_logprob / 10.0)))
        else:
            confidence = 0.75  # Default when scores not available

        return answer, round(confidence, 2)

    except Exception as e:
        logger.error(f'"Inference error: {str(e)}"')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(e)}",
        )


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint. No authentication required.
    Returns model and GPU status.
    """
    vram_used = None
    if torch.cuda.is_available():
        vram_used = round(torch.cuda.memory_allocated() / 1e9, 2)

    return HealthResponse(
        status="healthy" if app_state.model_loaded else "degraded",
        model_loaded=app_state.model_loaded,
        rag_index_loaded=app_state.rag_loaded,
        gpu_available=torch.cuda.is_available(),
        vram_used_gb=vram_used,
        environment=ENV,
        version=MODEL_VERSION,
    )


@app.post(
    "/api/v1/ask",
    response_model=AskResponse,
    tags=["Tax Assistant"],
    summary="Ask an Indian tax law question",
    description=(
        "Submit an Indian tax law question. The model uses RAG over the Income Tax Act 1961 "
        "and GST Acts to provide accurate, section-cited answers. Rate limited to 60 req/min."
    ),
)
@limiter.limit("60/minute")
async def ask_question(
    request: Request,
    body: AskRequest,
    api_key: str = Depends(verify_api_key),
):
    """Main inference endpoint. Authenticated + rate limited."""
    start_time = time.time()

    # Hash question for logging (don't log raw PII)
    q_hash = hashlib.sha256(body.question.encode()).hexdigest()[:12]
    logger.info(f'"request | key={api_key[:8]}... | q_hash={q_hash} | rag={app_state.rag_loaded}"')

    # 1. RAG retrieval
    retrieved_context, source_sections = "", []
    rag_used = False
    if app_state.rag_loaded:
        retrieved_context, source_sections = run_rag_retrieval(body.question)
        rag_used = bool(retrieved_context)

    # 2. Model inference
    answer, confidence = generate_answer(
        question=body.question,
        context=body.context or "",
        retrieved_context=retrieved_context,
    )

    # 3. Session history
    if body.session_id:
        if body.session_id not in session_history:
            session_history[body.session_id] = []
        session_history[body.session_id].append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": body.question,
            "answer": answer,
            "latency_ms": int((time.time() - start_time) * 1000),
        })
        # Keep last 20 turns per session
        session_history[body.session_id] = session_history[body.session_id][-20:]

    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(f'"response | q_hash={q_hash} | latency={latency_ms}ms | confidence={confidence} | rag={rag_used}"')

    return AskResponse(
        answer=answer,
        sources=[f"Income Tax Act, {s}" for s in source_sections] if source_sections else ["Llama 3.1 8B Indian Tax Model (trained knowledge)"],
        retrieved_sections=source_sections,
        confidence=confidence,
        rag_used=rag_used,
        model_version=MODEL_VERSION,
        disclaimer=LEGAL_DISCLAIMER,
        latency_ms=latency_ms,
        session_id=body.session_id,
    )


@app.get(
    "/api/v1/history/{session_id}",
    response_model=list[HistoryItem],
    tags=["Tax Assistant"],
    summary="Get conversation history for a session",
)
async def get_history(
    session_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Return the last 20 turns for a session ID."""
    history = session_history.get(session_id, [])
    return [HistoryItem(**item) for item in history]


@app.delete(
    "/api/v1/history/{session_id}",
    tags=["Tax Assistant"],
    summary="Clear conversation history for a session",
)
async def clear_history(
    session_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Clear all history for a session ID."""
    if session_id in session_history:
        del session_history[session_id]
    return {"status": "cleared", "session_id": session_id}


@app.get("/api/v1/topics", tags=["Tax Assistant"])
async def get_topics():
    """Return the supported Indian tax law topic categories (no auth required)."""
    return {
        "topics": [
            {"id": "income_tax", "label": "Income Tax", "examples": ["Tax slabs", "Section 80C", "HRA exemption", "Capital gains"]},
            {"id": "gst", "label": "GST", "examples": ["GST rates", "Input Tax Credit", "GSTR filing", "Composition scheme", "RCM"]},
            {"id": "tds", "label": "TDS / TCS", "examples": ["TDS rates by section", "Form 26AS", "TDS on salary", "TDS on rent"]},
            {"id": "itr", "label": "ITR Filing", "examples": ["Which ITR form to use", "Filing deadlines", "Late filing penalty", "Advance tax"]},
            {"id": "capital_gains", "label": "Capital Gains", "examples": ["LTCG on equity", "STCG on property", "Indexation", "Section 54 exemption"]},
        ],
        "disclaimer": LEGAL_DISCLAIMER,
    }


# ════════════════════════════════════════════════════════════════════════════════
# EXCEPTION HANDLERS
# ════════════════════════════════════════════════════════════════════════════════

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "disclaimer": LEGAL_DISCLAIMER if exc.status_code < 500 else None,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f'"Unhandled exception: {str(exc)}"')
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred. Please try again.", "status_code": 500},
    )


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "day5-fastapi-app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=1,          # One worker per GPU — never share GPU across workers
        log_level=LOG_LEVEL.lower(),
        reload=ENV == "development",
    )


# ════════════════════════════════════════════════════════════════════════════════
# DOCKERFILE (save as Dockerfile in same directory to deploy to HF Spaces)
# ════════════════════════════════════════════════════════════════════════════════
#
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
#
# RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
#
# WORKDIR /app
# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt
#
# COPY . .
#
# ENV ENV=production
# ENV PORT=7860
# ENV LOG_LEVEL=INFO
#
# EXPOSE 7860
#
# CMD ["python3", "-m", "uvicorn", "day5-fastapi-app:app",
#      "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
#
# ════════════════════════════════════════════════════════════════════════════════
# REQUIREMENTS (save as requirements.txt)
# ════════════════════════════════════════════════════════════════════════════════
#
# fastapi==0.111.0
# uvicorn[standard]==0.29.0
# slowapi==0.1.9
# pydantic==2.7.1
# torch==2.3.0
# transformers==4.41.2
# peft==0.11.1
# bitsandbytes==0.43.1
# unsloth @ git+https://github.com/unslothai/unsloth.git
# llama-index==0.10.43
# llama-index-vector-stores-chroma==0.1.9
# chromadb==0.5.0
# sentence-transformers==2.7.0
# rank-bm25==0.2.2
# pymupdf==1.24.4
# python-multipart==0.0.9
