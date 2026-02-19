# ==========================================================
#   STREAMLIT FRONTEND FOR AVIATION ACCIDENTS RAG
#   - Supports Ollama and Google Generative AI providers
#   - Most settings configurable live from the sidebar
# ==========================================================

import os
import re
import json
import hashlib
import logging
from typing import List, Optional
import numpy as np

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# ENV DEFAULTS  (sidebar overrides these at runtime)
# ==========================================================

DATA_DIR = "./data"
PERSIST_DIR = "./vector_db"
REGISTRY_FILE = "./doc_registry.json"

# Read .env once for defaults — sidebar widgets will override
_ENV = {
    "LLM_PROVIDER":       os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
    "OLLAMA_MODEL":       os.getenv("OLLAMA_MODEL", "phi4-mini:latest"),
    "GOOGLE_MODEL":       os.getenv("GOOGLE_MODEL", "gemma-3-27b-it"),
    "GOOGLE_API_KEY":     os.getenv("GOOGLE_API_KEY", ""),
    "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5"),
    "CROSS_ENCODER_MODEL": os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    "DENSE_K":            int(os.getenv("DENSE_K", "15")),
    "BM25_K":             int(os.getenv("BM25_K", "15")),
    "RERANK_TOP_K":       int(os.getenv("RERANK_TOP_K", "4")),
    "CONFIDENCE_THRESHOLD": float(os.getenv("CONFIDENCE_THRESHOLD", "-5.0")),
    "MAX_CONTEXT_TOKENS": int(os.getenv("MAX_CONTEXT_TOKENS", "5000")),
    "LLM_NUM_PREDICT":    int(os.getenv("LLM_NUM_PREDICT", "512")),
    "LLM_NUM_CTX":        int(os.getenv("LLM_NUM_CTX", "8192")),
    "CHUNK_SIZE":         int(os.getenv("CHUNK_SIZE", "800")),
    "CHUNK_OVERLAP":      int(os.getenv("CHUNK_OVERLAP", "100")),
    "EMBED_BATCH_SIZE":   int(os.getenv("EMBED_BATCH_SIZE", "500")),
    "DOMINANCE_THRESHOLD": float(os.getenv("DOMINANCE_THRESHOLD", "0.5")),
    "MIN_SECONDARY_SLOTS": int(os.getenv("MIN_SECONDARY_SLOTS", "1")),
    "HYDE_ENABLED":       os.getenv("HYDE_ENABLED", "true").strip().lower() == "true",
    "MIN_CHARS_PER_PAGE": int(os.getenv("MIN_CHARS_PER_PAGE", "50")),
}

AVG_CHARS_PER_TOKEN = 4

STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "and", "or",
    "of", "for", "with", "this", "that", "was", "are", "be", "by", "as",
    "from", "but", "not", "have", "had", "has", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "what",
    "how", "when", "where", "who", "which",
}

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# Helper to read current sidebar settings (fallback to env defaults)
def cfg(key: str):
    return st.session_state.get(f"cfg_{key}", _ENV[key])

# ==========================================================
# HELPERS (query / tokenisation)
# ==========================================================

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return [t for t in text.split() if t not in STOPWORDS]

def _preprocess_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"[^\w\s]", "", query)
    tokens = [t for t in query.split() if t not in STOPWORDS]
    return " ".join(tokens) if tokens else query

def _query_entropy(query: str) -> float:
    tokens = query.lower().split()
    if not tokens:
        return 0.0
    freq: dict = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens)
    return -sum((c / total) * np.log2(c / total) for c in freq.values())

def _keyword_density(query: str) -> float:
    tokens = query.lower().split()
    if not tokens:
        return 0.0
    return len([t for t in tokens if t not in STOPWORDS]) / len(tokens)

def _should_use_hyde(query: str) -> bool:
    if not cfg("HYDE_ENABLED"):
        return False
    return _keyword_density(query) < 0.4 or _query_entropy(query) < 1.0

def _build_context(docs, max_tokens: int | None = None) -> str:
    if max_tokens is None:
        max_tokens = cfg("MAX_CONTEXT_TOKENS")
    parts, total_chars = [], 0
    budget_chars = max_tokens * AVG_CHARS_PER_TOKEN
    for doc in docs:
        chunk = doc.page_content
        if total_chars + len(chunk) > budget_chars:
            remaining = budget_chars - total_chars
            if remaining > 50:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total_chars += len(chunk)
    return "\n\n".join(parts)

# ==========================================================
# INGESTION HELPERS (mirrors main.py exactly)
# ==========================================================

def _file_hash(filepath: str) -> Optional[str]:
    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed hashing {filepath}: {e}")
        return None

def _load_registry() -> dict:
    try:
        if os.path.exists(REGISTRY_FILE):
            with open(REGISTRY_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Registry load failed: {e}")
    return {}

def _save_registry(registry: dict) -> None:
    try:
        with open(REGISTRY_FILE, "w") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        logger.warning(f"Registry save failed: {e}")

def _is_digital_pdf(filepath: str) -> bool:
    import fitz
    try:
        doc = fitz.open(filepath)
        total_chars = sum(len(page.get_text()) for page in doc)
        avg = total_chars / max(len(doc), 1)
        doc.close()
        return avg >= cfg("MIN_CHARS_PER_PAGE")
    except Exception:
        return False

def _delete_document_chunks(vectorstore, filepath: str) -> None:
    try:
        results = vectorstore.get(where={"source": filepath})
        if results and results["ids"]:
            vectorstore.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} old chunks for: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to delete old chunks for {filepath}: {e}")

def _load_single_document(filepath: str) -> list:
    from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
    from langchain_core.documents import Document
    import fitz

    ext = os.path.splitext(filepath)[1].lower()
    if ext != ".pdf":
        try:
            return TextLoader(filepath, encoding="utf-8").load()
        except Exception as e:
            logger.warning(f"TextLoader failed for {filepath}: {e}")
            return []

    if _is_digital_pdf(filepath):
        try:
            return PyMuPDFLoader(filepath).load()
        except Exception as e:
            logger.warning(f"PyMuPDFLoader failed for {filepath}: {e}")
            return []
    else:
        ocr_path = os.path.splitext(filepath)[0] + ".txt"
        if os.path.exists(ocr_path):
            try:
                docs = TextLoader(ocr_path, encoding="utf-8").load()
                for doc in docs:
                    doc.metadata["source"] = filepath
                    doc.metadata["ocr_source"] = ocr_path
                return docs
            except Exception as e:
                logger.warning(f"Failed loading OCR file {ocr_path}: {e}")
                return []
        else:
            logger.warning(f"Scanned PDF with no paired .txt found: {filepath}")
            return []

def _get_new_documents(vectorstore) -> List:
    from langchain_core.documents import Document

    registry = _load_registry()
    updated_registry = registry.copy()
    new_docs = []

    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            all_files.append(os.path.join(root, fname))

    ocr_companions = {
        os.path.splitext(f)[0] + ".txt"
        for f in all_files if f.lower().endswith(".pdf")
    }

    for filepath in all_files:
        if filepath in ocr_companions:
            continue
        current_hash = _file_hash(filepath)
        if current_hash is None:
            continue
        if filepath not in registry or registry[filepath]["hash"] != current_hash:
            if filepath in registry:
                _delete_document_chunks(vectorstore, filepath)
            docs = _load_single_document(filepath)
            if docs:
                new_docs.extend(docs)
                updated_registry[filepath] = {
                    "hash": current_hash,
                    "embedding_model": cfg("EMBEDDING_MODEL_NAME"),
                }

    _save_registry(updated_registry)
    return new_docs

def _build_bm25(vectorstore):
    from langchain_core.documents import Document
    from rank_bm25 import BM25Okapi
    try:
        raw = vectorstore.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(raw["documents"], raw["metadatas"])
        ]
        corpus = [_tokenize(doc.page_content) for doc in docs]
        bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built over {len(docs)} chunks.")
        return bm25, docs
    except Exception as e:
        logger.warning(f"BM25 init failed: {e}")
        return None, []

# ==========================================================
# CACHED INIT — split into resources (heavy) + LLM (light)
# ==========================================================

@st.cache_resource(show_spinner=False)
def initialize_resources():
    """
    Runs once. Loads the heavy, rarely-changing components:
      1. Embedding model
      2. Chroma vectorstore (+ incremental ingestion)
      3. BM25 index
      4. Cross-encoder reranker
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import CrossEncoder

    status = st.status("Initializing RAG resources…", expanded=True)

    # 1. Embedding model
    emb_name = cfg("EMBEDDING_MODEL_NAME")
    status.write("⏳ Loading embedding model…")
    embedding_model = HuggingFaceEmbeddings(model_name=emb_name)
    status.write("✅ Embedding model loaded.")

    # 2. Vectorstore
    status.write("⏳ Connecting to vector store…")
    try:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model,
        )
        status.write("✅ Vector store connected.")
    except Exception as e:
        status.error(f"Vector DB load failed: {e}")
        raise

    # 3. Incremental ingestion
    status.write("🔍 Checking for new or changed documents…")
    new_docs = _get_new_documents(vectorstore)

    chunk_size = cfg("CHUNK_SIZE")
    chunk_overlap = cfg("CHUNK_OVERLAP")
    batch_size = cfg("EMBED_BATCH_SIZE")

    if new_docs:
        status.write(f"📄 Embedding {len(new_docs)} new/changed document(s)…")
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = splitter.split_documents(new_docs)
        try:
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                vectorstore.add_documents(batch)
                status.write(f"  ✅ Batch {i // batch_size + 1} added ({len(batch)} chunks).")
            status.write("✅ Vector store updated.")
        except Exception as e:
            status.warning(f"Embedding failed: {e}")
    else:
        status.write("✅ No new documents — vector store is up to date.")

    # 4. BM25
    status.write("⏳ Building BM25 index…")
    bm25, bm25_docs = _build_bm25(vectorstore)
    status.write(f"✅ BM25 index built over {len(bm25_docs)} chunks.")

    # 5. Cross-encoder
    ce_model = cfg("CROSS_ENCODER_MODEL")
    status.write(f"⏳ Loading cross-encoder ({ce_model})…")
    try:
        cross_encoder = CrossEncoder(ce_model)
        status.write("✅ Cross-encoder loaded.")
    except Exception as e:
        status.warning(f"Cross-encoder load failed: {e}")
        cross_encoder = None

    status.update(label="✅ RAG resources ready!", state="complete", expanded=False)
    return vectorstore, bm25, bm25_docs, cross_encoder, embedding_model


def initialize_llm(provider: str, model_name: str, num_predict: int, num_ctx: int):
    """
    Creates a new LLM instance. Not cached — these are lightweight API client
    objects (no model download), so re-creation on param change is cheap.
    """
    from langchain_community.llms import Ollama
    from langchain_google_genai import ChatGoogleGenerativeAI

    if provider == "google":
        api_key = _ENV["GOOGLE_API_KEY"]
        if not api_key:
            raise SystemExit("GOOGLE_API_KEY is not set in .env — cannot use Google provider.")
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                max_output_tokens=num_predict,
            )
            return llm
        except Exception as e:
            raise RuntimeError(f"Google GenAI load failed: {e}")
    else:
        try:
            llm = Ollama(
                model=model_name,
                num_predict=num_predict,
                num_ctx=num_ctx,
            )
            return llm
        except Exception as e:
            raise RuntimeError(f"Ollama load failed: {e}")

# ==========================================================
# RETRIEVAL + RERANKING
# ==========================================================

def _bm25_retrieve(query: str, bm25, bm25_docs, k: int | None = None):
    if k is None:
        k = cfg("BM25_K")
    if bm25 is None:
        return []
    try:
        preprocessed = _preprocess_query(query)
        scores = bm25.get_scores(preprocessed.split())
        top_idx = np.argsort(scores)[-k:][::-1]
        return [bm25_docs[i] for i in top_idx if scores[i] > 0]
    except Exception:
        return []

def _rerank(question: str, docs, cross_encoder, top_k: int | None = None):
    if top_k is None:
        top_k = cfg("RERANK_TOP_K")
    threshold = cfg("CONFIDENCE_THRESHOLD")
    if not docs:
        return [], []
    if cross_encoder is None:
        return docs[:top_k], [None] * min(top_k, len(docs))
    pairs = [(question, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    confident = [(doc, s) for doc, s in ranked if s >= threshold]
    if not confident:
        return [], []
    top = confident[:top_k]
    return [d for d, _ in top], [s for _, s in top]

def _source_weighted_context(reranked_docs, scores, top_k: int | None = None):
    """
    Allocate context slots with source-diversity awareness.
    Uses DOMINANCE_THRESHOLD and MIN_SECONDARY_SLOTS from sidebar config.
    If either tier is underfilled, remaining slots spill to the other.
    """
    from collections import Counter
    if top_k is None:
        top_k = cfg("RERANK_TOP_K")
    if not reranked_docs:
        return []

    dominance_thresh = cfg("DOMINANCE_THRESHOLD")
    min_secondary = cfg("MIN_SECONDARY_SLOTS")

    source_counts = Counter(doc.metadata.get("source", "unknown") for doc in reranked_docs)
    dominant_source, dominant_count = source_counts.most_common(1)[0]
    dominance_ratio = dominant_count / len(reranked_docs)

    if dominance_ratio > dominance_thresh:
        secondary_slots = max(min_secondary, 1)
        primary_slots = top_k - secondary_slots
    else:
        primary_slots = (top_k + 1) // 2
        secondary_slots = top_k - primary_slots

    primary_docs = [d for d in reranked_docs if d.metadata.get("source") == dominant_source]
    other_docs   = [d for d in reranked_docs if d.metadata.get("source") != dominant_source]

    selected_primary   = primary_docs[:primary_slots]
    selected_secondary = other_docs[:secondary_slots]

    # Spillover
    remaining = top_k - len(selected_primary) - len(selected_secondary)
    if remaining > 0:
        unused_primary   = primary_docs[len(selected_primary):]
        unused_secondary = other_docs[len(selected_secondary):]
        extras = (unused_secondary + unused_primary)[:remaining]
        selected_secondary += extras

    return selected_primary + selected_secondary

def run_rag(question: str, vectorstore, bm25, bm25_docs, cross_encoder, llm) -> tuple:
    """Returns (answer_str, source_list, debug_info)."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    debug = {}
    use_hyde = _should_use_hyde(question)
    debug["hyde_used"] = use_hyde
    query_for_search = question

    if use_hyde:
        try:
            hyde_prompt = ChatPromptTemplate.from_template(
                "Write a detailed answer to the following question as if you were an expert:\n\n{question}"
            )
            hyp = (hyde_prompt | llm | StrOutputParser()).invoke({"question": question})
            if hyp:
                query_for_search = hyp
        except Exception:
            pass

    dense_k = cfg("DENSE_K")
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": dense_k})
    try:
        dense_docs = dense_retriever.invoke(query_for_search)
    except Exception:
        dense_docs = []

    bm25_results = _bm25_retrieve(query_for_search, bm25, bm25_docs)

    merged: dict = {doc.page_content: doc for doc in dense_docs}
    for doc in bm25_results:
        merged.setdefault(doc.page_content, doc)

    debug["dense_count"] = len(dense_docs)
    debug["bm25_count"]  = len(bm25_results)
    debug["merged_count"] = len(merged)

    reranked, scores = _rerank(question, list(merged.values()), cross_encoder)
    debug["reranked_count"] = len(reranked)

    if not reranked:
        return "I don't have enough information to answer that.", [], debug

    weighted = _source_weighted_context(reranked, scores)
    context  = _build_context(weighted)
    sources  = list({os.path.basename(d.metadata.get("source", "unknown")) for d in weighted})
    debug["context_chunks"] = len(weighted)
    debug["top_score"] = f"{scores[0]:.3f}" if scores and scores[0] is not None else "N/A"

    rag_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question using the context provided below.

- If the context fully answers the question, give a complete answer.
- If the context partially answers the question, answer what you can and clearly note what is missing.
- If the context contains no relevant information, say: "I don't have enough information to answer that."
- if the answer can be made descriptive make it dexcriptive from the source

Context:
{context}

Question:
{question}

Answer:"""
    )
    try:
        answer = (rag_prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    except Exception as e:
        answer = f"Model failed to generate a response: {e}"

    return answer, sources, debug

# ==========================================================
# STREAMLIT UI
# ==========================================================

st.set_page_config(
    page_title="Aviation Accidents RAG",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
/* ── Page background ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f3e; }
[data-testid="stHeader"] { display: none; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 6px;
}

/* ── Source pill badges ── */
.source-pill {
    display: inline-block;
    background: #1e2535;
    border: 1px solid #2e3650;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.76rem;
    color: #8ab4f8;
    margin: 3px 3px 3px 0;
    font-family: monospace;
}

/* ── Suggestion cards ── */
.suggestion-card {
    background: #1a1f2e;
    border: 1px solid #2a3045;
    border-radius: 10px;
    padding: 12px 16px;
    cursor: pointer;
    transition: border-color 0.2s;
    font-size: 0.88rem;
    color: #c8cdd8;
    margin-bottom: 8px;
}
.suggestion-card:hover { border-color: #4a6fa5; color: #fff; }

/* ── Sidebar metric box ── */
.metric-box {
    background: #1a1f2e;
    border: 1px solid #2a3045;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.metric-label { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 1rem; color: #e2e8f0; font-weight: 600; margin-top: 2px; }

/* ── Hero header ── */
.hero { text-align: center; padding: 32px 0 8px 0; }
.hero h1 { font-size: 2.2rem; font-weight: 700; color: #e2e8f0; margin-bottom: 6px; }
.hero p  { color: #6b7280; font-size: 1rem; }

/* ── Divider ── */
.custom-divider { border: none; border-top: 1px solid #2a2f3e; margin: 16px 0; }

/* ── Debug expander ── */
.debug-row { font-size: 0.8rem; color: #8b95a5; font-family: monospace; }

/* ── Provider badge ── */
.provider-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.provider-ollama { background: #1a332a; color: #6ee7b7; border: 1px solid #2d5a44; }
.provider-google { background: #332a1a; color: #fbbf24; border: 1px solid #5a4a2d; }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# SIDEBAR — Settings & Controls
# ==========================================================

with st.sidebar:
    st.markdown("## ✈️ Aviation RAG")
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Provider & Model ──────────────────────
    st.markdown("#### 🔌 LLM Provider")

    provider_options = ["ollama", "google"]
    provider = st.selectbox(
        "Provider",
        provider_options,
        index=provider_options.index(_ENV["LLM_PROVIDER"]),
        key="cfg_LLM_PROVIDER",
        help="Switching provider reinitializes the LLM.",
    )

    if provider == "ollama":
        model_name = st.text_input(
            "Ollama Model", value=_ENV["OLLAMA_MODEL"], key="cfg_OLLAMA_MODEL",
            help="e.g. gemma3:4b, phi4-mini:latest, llama3:8b",
        )
        badge_class, badge_label = "provider-ollama", f"Ollama · {model_name}"
    else:
        model_name = st.text_input(
            "Google Model", value=_ENV["GOOGLE_MODEL"], key="cfg_GOOGLE_MODEL",
            help="e.g. gemma-3-27b-it, gemini-2.0-flash",
        )
        badge_class, badge_label = "provider-google", f"Google · {model_name}"

    st.markdown(f'<span class="provider-badge {badge_class}">{badge_label}</span>', unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Retrieval Settings ────────────────────
    st.markdown("#### 🔍 Retrieval")

    col1, col2 = st.columns(2)
    col1.number_input("Dense K", min_value=1, max_value=100, value=_ENV["DENSE_K"],
                      key="cfg_DENSE_K", help="Chunks from vector (dense) retrieval")
    col2.number_input("BM25 K", min_value=1, max_value=100, value=_ENV["BM25_K"],
                      key="cfg_BM25_K", help="Chunks from keyword (BM25) retrieval")

    st.number_input("Rerank Top-K", min_value=1, max_value=30, value=_ENV["RERANK_TOP_K"],
                    key="cfg_RERANK_TOP_K", help="How many chunks survive cross-encoder reranking")
    st.slider("Confidence Threshold", min_value=-10.0, max_value=5.0,
              value=_ENV["CONFIDENCE_THRESHOLD"], step=0.5, key="cfg_CONFIDENCE_THRESHOLD",
              help="Cross-encoder score cutoff (ms-marco scale). Lower = more permissive")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Context Assembly ──────────────────────
    st.markdown("#### 📦 Context Assembly")

    st.number_input("Max Context Tokens", min_value=500, max_value=64000, step=500,
                    value=_ENV["MAX_CONTEXT_TOKENS"], key="cfg_MAX_CONTEXT_TOKENS",
                    help="Token budget for context sent to the LLM")
    st.slider("Dominance Threshold", min_value=0.0, max_value=1.0, step=0.05,
              value=_ENV["DOMINANCE_THRESHOLD"], key="cfg_DOMINANCE_THRESHOLD",
              help="Ratio above which a single source is considered dominant")
    st.number_input("Min Secondary Slots", min_value=0, max_value=10,
                    value=_ENV["MIN_SECONDARY_SLOTS"], key="cfg_MIN_SECONDARY_SLOTS",
                    help="Minimum chunk slots reserved for non-dominant sources")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Generation ────────────────────────────
    st.markdown("#### ✍️ Generation")

    st.number_input("Max Output Tokens", min_value=64, max_value=8192, step=64,
                    value=_ENV["LLM_NUM_PREDICT"], key="cfg_LLM_NUM_PREDICT",
                    help="Maximum tokens the LLM can generate per answer")
    if provider == "ollama":
        st.number_input("Context Window (Ollama)", min_value=2048, max_value=131072, step=1024,
                        value=_ENV["LLM_NUM_CTX"], key="cfg_LLM_NUM_CTX",
                        help="num_ctx passed to Ollama")

    st.checkbox("Enable HyDE", value=_ENV["HYDE_ENABLED"], key="cfg_HYDE_ENABLED",
                help="Generate a hypothetical answer first to improve retrieval on vague queries")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Actions ───────────────────────────────
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.session_state.debug_info = {}
        st.rerun()

    st.caption("Indian aviation accident investigation reports — AAIB/DGCA")


# ==========================================================
# INITIALIZE
# ==========================================================

# Heavy resources — cached once, never re-created from sidebar changes
vectorstore, bm25, bm25_docs, cross_encoder, embedding_model = initialize_resources()

# LLM — re-created when provider, model, or generation params change
active_provider = cfg("LLM_PROVIDER")
active_model = cfg("OLLAMA_MODEL") if active_provider == "ollama" else cfg("GOOGLE_MODEL")
active_num_predict = cfg("LLM_NUM_PREDICT")
active_num_ctx = cfg("LLM_NUM_CTX")

llm = initialize_llm(active_provider, active_model, active_num_predict, active_num_ctx)

# ── Sidebar: System Info (after init so we can read chunk count) ──
with st.sidebar:
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown("#### 📊 System Info")
    try:
        chunk_count = vectorstore._collection.count()
    except Exception:
        chunk_count = "—"

    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Active LLM</div>
        <div class="metric-value" style="font-size:0.88rem">{active_model}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Embedding Model</div>
        <div class="metric-value" style="font-size:0.78rem">{cfg("EMBEDDING_MODEL_NAME")}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Indexed Chunks</div>
        <div class="metric-value">{chunk_count}</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = {}
if "debug_info" not in st.session_state:
    st.session_state.debug_info = {}


# ==========================================================
# MAIN AREA
# ==========================================================

# --- Hero header (only when chat is empty) ---
if not st.session_state.messages:
    st.markdown("""
    <div class="hero">
        <h1>✈️ Aviation Accidents RAG</h1>
        <p>Ask questions about Indian aviation accident investigation reports</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown("#### 💡 Try asking")

    SUGGESTIONS = [
        "What were the causes of the Air India Express Mangalore crash?",
        "What did the CVR recordings reveal in the Kozhikode accident?",
        "Which accidents involved CFIT (controlled flight into terrain)?",
        "What mechanical failures led to accidents in Indian aviation?",
        "Summarise the findings of the VT-EHY accident report.",
    ]

    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        if cols[i % 2].button(suggestion, key=f"sug_{i}", use_container_width=True):
            st.session_state.pending_prompt = suggestion
            st.rerun()

    st.markdown("")

else:
    # Compact header when chat has messages
    st.markdown("<h3 style='color:#e2e8f0;margin-bottom:0'>✈️ Aviation Accidents RAG</h3>", unsafe_allow_html=True)
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)


# --- Render chat history ---
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Source pills
            if i in st.session_state.sources:
                srcs = st.session_state.sources[i]
                if srcs:
                    pills = "".join(f'<span class="source-pill">📄 {s}</span>' for s in srcs)
                    st.markdown(f"<div style='margin-top:8px'>{pills}</div>", unsafe_allow_html=True)
            # Debug expander
            if i in st.session_state.debug_info:
                dbg = st.session_state.debug_info[i]
                with st.expander("🔎 Retrieval details", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Dense", dbg.get("dense_count", "—"))
                    c2.metric("BM25", dbg.get("bm25_count", "—"))
                    c3.metric("Merged", dbg.get("merged_count", "—"))
                    c4.metric("Reranked", dbg.get("reranked_count", "—"))
                    st.caption(
                        f"Context chunks: {dbg.get('context_chunks', '—')} · "
                        f"Top score: {dbg.get('top_score', '—')} · "
                        f"HyDE: {'Yes' if dbg.get('hyde_used') else 'No'}"
                    )


# --- Handle suggestion click or chat input ---
prompt = st.session_state.pop("pending_prompt", None)
typed  = st.chat_input("Ask about an aviation accident…")
prompt = prompt or typed

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            answer, sources, debug = run_rag(prompt, vectorstore, bm25, bm25_docs, cross_encoder, llm)
        st.markdown(answer)
        if sources:
            pills = "".join(f'<span class="source-pill">📄 {s}</span>' for s in sources)
            st.markdown(f"<div style='margin-top:8px'>{pills}</div>", unsafe_allow_html=True)
        with st.expander("🔎 Retrieval details", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dense", debug.get("dense_count", "—"))
            c2.metric("BM25", debug.get("bm25_count", "—"))
            c3.metric("Merged", debug.get("merged_count", "—"))
            c4.metric("Reranked", debug.get("reranked_count", "—"))
            st.caption(
                f"Context chunks: {debug.get('context_chunks', '—')} · "
                f"Top score: {debug.get('top_score', '—')} · "
                f"HyDE: {'Yes' if debug.get('hyde_used') else 'No'}"
            )

    msg_index = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.sources[msg_index] = sources
    st.session_state.debug_info[msg_index] = debug
    st.rerun()
