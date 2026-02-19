# ==========================================================
#   STREAMLIT FRONTEND FOR AVIATION ACCIDENTS RAG
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
# CONFIG (mirrors main.py)
# ==========================================================

DATA_DIR = "./data"
PERSIST_DIR = "./vector_db"
REGISTRY_FILE = "./doc_registry.json"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini:latest")

RERANK_TOP_K  = int(os.getenv("RERANK_TOP_K", "4"))
DENSE_K       = int(os.getenv("DENSE_K", "15"))
BM25_K        = int(os.getenv("BM25_K", "15"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "5000"))
AVG_CHARS_PER_TOKEN = 4
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "-5.0"))
HYDE_ENABLED  = os.getenv("HYDE_ENABLED", "true").strip().lower() == "true"
MIN_CHARS_PER_PAGE = 50
BATCH_SIZE    = 500

STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "and", "or",
    "of", "for", "with", "this", "that", "was", "are", "be", "by", "as",
    "from", "but", "not", "have", "had", "has", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "what",
    "how", "when", "where", "who", "which",
}

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

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
    if not HYDE_ENABLED:
        return False
    return _keyword_density(query) < 0.4 or _query_entropy(query) < 1.0

def _build_context(docs, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
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
        return avg >= MIN_CHARS_PER_PAGE
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
                    "embedding_model": EMBEDDING_MODEL_NAME,
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
# SINGLE CACHED INIT — mirrors main.py module-level startup
# ==========================================================

@st.cache_resource(show_spinner=False)
def initialize_rag():
    """
    Runs once per session. Mirrors main.py startup exactly:
      1. Load embedding model
      2. Load / create Chroma vectorstore
      3. Detect new/changed files → ingest into vectorstore
      4. Build BM25 index
      5. Load cross-encoder
      6. Load LLM
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import CrossEncoder
    from langchain_community.llms import Ollama

    status = st.status("Initializing RAG system...", expanded=True)

    # 1. Embedding model
    status.write("⏳ Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    status.write("✅ Embedding model loaded.")

    # 2. Vectorstore
    status.write("⏳ Connecting to vector store...")
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
    status.write("🔍 Checking for new or changed documents...")
    new_docs = _get_new_documents(vectorstore)

    if new_docs:
        status.write(f"📄 Embedding {len(new_docs)} new/changed document(s)...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = splitter.split_documents(new_docs)
        try:
            for i in range(0, len(splits), BATCH_SIZE):
                batch = splits[i:i + BATCH_SIZE]
                vectorstore.add_documents(batch)
                status.write(f"  ✅ Batch {i // BATCH_SIZE + 1} added ({len(batch)} chunks).")
            status.write("✅ Vector store updated.")
        except Exception as e:
            status.warning(f"Embedding failed: {e}")
    else:
        status.write("✅ No new documents — vector store is up to date.")

    # 4. BM25
    status.write("⏳ Building BM25 index...")
    bm25, bm25_docs = _build_bm25(vectorstore)
    status.write(f"✅ BM25 index built over {len(bm25_docs)} chunks.")

    # 5. Cross-encoder
    status.write("⏳ Loading cross-encoder...")
    try:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        status.write("✅ Cross-encoder loaded.")
    except Exception as e:
        status.warning(f"Cross-encoder load failed: {e}")
        cross_encoder = None

    # 6. LLM
    status.write(f"⏳ Connecting to Ollama ({OLLAMA_MODEL})...")
    try:
        llm = Ollama(model=OLLAMA_MODEL)
        status.write(f"✅ LLM ready ({OLLAMA_MODEL}).")
    except Exception as e:
        status.error(f"Ollama load failed: {e}")
        raise

    status.update(label="✅ RAG system ready!", state="complete", expanded=False)
    return vectorstore, bm25, bm25_docs, cross_encoder, llm, embedding_model

# ==========================================================
# RETRIEVAL + RERANKING
# ==========================================================

def _bm25_retrieve(query: str, bm25, bm25_docs, k: int = BM25_K):
    if bm25 is None:
        return []
    try:
        preprocessed = _preprocess_query(query)
        scores = bm25.get_scores(preprocessed.split())
        top_idx = np.argsort(scores)[-k:][::-1]
        return [bm25_docs[i] for i in top_idx if scores[i] > 0]
    except Exception:
        return []

def _rerank(question: str, docs, cross_encoder, top_k: int = RERANK_TOP_K):
    if not docs:
        return [], []
    if cross_encoder is None:
        return docs[:top_k], [None] * min(top_k, len(docs))
    pairs = [(question, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    confident = [(doc, s) for doc, s in ranked if s >= CONFIDENCE_THRESHOLD]
    if not confident:
        return [], []
    top = confident[:top_k]
    return [d for d, _ in top], [s for _, s in top]

def _source_weighted_context(reranked_docs, scores, top_k: int = RERANK_TOP_K):
    from collections import Counter
    if not reranked_docs:
        return []
    source_counts = Counter(doc.metadata.get("source", "unknown") for doc in reranked_docs)
    dominant_source, dominant_count = source_counts.most_common(1)[0]
    dominance_ratio = dominant_count / len(reranked_docs)
    primary_slots = (top_k - 1) if dominance_ratio > 0.5 else top_k // 2
    secondary_slots = top_k - primary_slots
    primary   = [d for d in reranked_docs if d.metadata.get("source") == dominant_source]
    secondary = [d for d in reranked_docs if d.metadata.get("source") != dominant_source
                 and source_counts[d.metadata.get("source")] > 1]
    fallback  = [d for d in reranked_docs if d.metadata.get("source") != dominant_source
                 and source_counts[d.metadata.get("source")] == 1]
    secondary_sel = secondary[:secondary_slots]
    if len(secondary_sel) < secondary_slots:
        secondary_sel += fallback[:secondary_slots - len(secondary_sel)]
    return primary[:primary_slots] + secondary_sel

def run_rag(question: str, vectorstore, bm25, bm25_docs, cross_encoder, llm) -> tuple:
    """Returns (answer_str, source_list)."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    use_hyde = _should_use_hyde(question)
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

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": DENSE_K})
    try:
        dense_docs = dense_retriever.invoke(query_for_search)
    except Exception:
        dense_docs = []

    bm25_results = _bm25_retrieve(query_for_search, bm25, bm25_docs)

    merged: dict = {doc.page_content: doc for doc in dense_docs}
    for doc in bm25_results:
        merged.setdefault(doc.page_content, doc)

    reranked, scores = _rerank(question, list(merged.values()), cross_encoder)
    if not reranked:
        return "I don't have enough information to answer that.", []

    weighted = _source_weighted_context(reranked, scores)
    context  = _build_context(weighted)
    sources  = list({os.path.basename(d.metadata.get("source", "unknown")) for d in weighted})

    rag_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question using the context provided below.

- If the context fully answers the question, give a complete answer.
- If the context partially answers the question, answer what you can and clearly note what is missing.
- If the context contains no relevant information, say: "I don't have enough information to answer that."

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

    return answer, sources

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
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a2f3e;
}

/* ── Hide default header ── */
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
</style>
""", unsafe_allow_html=True)

# --- Initialize (runs once) ---
vectorstore, bm25, bm25_docs, cross_encoder, llm, embedding_model = initialize_rag()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ✈️ Aviation RAG")
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # Metrics
    try:
        chunk_count = vectorstore._collection.count()
    except Exception:
        chunk_count = "—"

    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">LLM Model</div>
        <div class="metric-value">{OLLAMA_MODEL}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Embedding Model</div>
        <div class="metric-value" style="font-size:0.82rem">{EMBEDDING_MODEL_NAME}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Indexed Chunks</div>
        <div class="metric-value">{chunk_count}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Dense K", DENSE_K)
    col2.metric("BM25 K", BM25_K)
    col1.metric("Rerank K", RERANK_TOP_K)
    col2.metric("Threshold", CONFIDENCE_THRESHOLD)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    hyde_state = "✅ On" if HYDE_ENABLED else "❌ Off"
    st.markdown(f"**HyDE:** {hyde_state}")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources  = {}
        st.rerun()

    st.caption("Indian aviation accident investigation reports — AAIB/DGCA")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = {}

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
        if msg["role"] == "assistant" and i in st.session_state.sources:
            srcs = st.session_state.sources[i]
            if srcs:
                pills = "".join(f'<span class="source-pill">📄 {s}</span>' for s in srcs)
                st.markdown(f"<div style='margin-top:8px'>{pills}</div>", unsafe_allow_html=True)

# --- Handle suggestion click or chat input ---
prompt = st.session_state.pop("pending_prompt", None)
typed  = st.chat_input("Ask about an aviation accident...")
prompt = prompt or typed

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            answer, sources = run_rag(prompt, vectorstore, bm25, bm25_docs, cross_encoder, llm)
        st.markdown(answer)
        if sources:
            pills = "".join(f'<span class="source-pill">📄 {s}</span>' for s in sources)
            st.markdown(f"<div style='margin-top:8px'>{pills}</div>", unsafe_allow_html=True)

    msg_index = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.sources[msg_index] = sources
    st.rerun()
