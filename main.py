# ==========================================================
#   PRODUCTION RAG (PERSISTENT + HYBRID + HyDE + SAFE)
#   Improvements:
#     1. HyDE triggers on query entropy/keyword density, not length
#     2. vectorstore.persist() removed (Chroma auto-persists)
#     3. BM25 retains source metadata via Document objects
#     4. get_relevant_documents() replaced with .invoke()
#     5. Query preprocessing for BM25 (lowercase + stopword removal)
#     6. Context window guard based on token estimate
#     7. Softer prompt for partial-context answers
#     8. Confidence threshold — low-score results bail early
#     9. Retrieval logging with chunk content + scores
# ==========================================================

import os
import re
import json
import hashlib
import logging
from typing import List, Optional
import numpy as np

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

import fitz  # pip install pymupdf
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
)

from dotenv import load_dotenv
load_dotenv()




# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ==========================================================
# CONFIG
# ==========================================================

DATA_DIR = "./data"
PERSIST_DIR = "./vector_db"
REGISTRY_FILE = "./doc_registry.json"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini:latest")

RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "4"))
DENSE_K = int(os.getenv("DENSE_K", "15"))
BM25_K = int(os.getenv("BM25_K", "15"))

# Minimum extractable characters per page to consider a PDF "digital"
MIN_CHARS_PER_PAGE = 50



# Approximate token budget for context (leave room for prompt + answer).
# Mistral context window ~8k tokens; reserve ~3k for prompt + answer.
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "5000"))

AVG_CHARS_PER_TOKEN = 4  # rough estimate

# Cross-encoder confidence threshold: below this, consider the doc irrelevant.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "-5.0"))
  # ms-marco scores; tune to your use case

# HyDE toggle — reads "true"/"false" from env, defaults to True 
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "true").strip().lower() == "true"

# BM25 stopwords (simple English set — extend as needed)
STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "and", "or",
    "of", "for", "with", "this", "that", "was", "are", "be", "by", "as",
    "from", "but", "not", "have", "had", "has", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "what",
    "how", "when", "where", "who", "which",
}

# ==========================================================
# SAFE HELPERS
# ==========================================================

def file_hash(filepath: str) -> Optional[str]:
    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed hashing {filepath}: {e}")
        return None

def load_registry() -> dict:
    try:
        if os.path.exists(REGISTRY_FILE):
            with open(REGISTRY_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Registry load failed: {e}")
    return {}

def save_registry(registry: dict) -> None:
    try:
        with open(REGISTRY_FILE, "w") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        logger.warning(f"Registry save failed: {e}")

# ==========================================================
# QUERY PREPROCESSING  (Fix #4 + #5)
# ==========================================================

def preprocess_query(query: str) -> str:
    """Lowercase, strip punctuation, remove stopwords for BM25."""
    query = query.lower()
    query = re.sub(r"[^\w\s]", "", query)
    tokens = [t for t in query.split() if t not in STOPWORDS]
    return " ".join(tokens) if tokens else query  # fallback to original if all tokens removed

def tokenize(text: str) -> List[str]:
    """Tokenize for BM25 index with the same preprocessing."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return [t for t in text.split() if t not in STOPWORDS]

# ==========================================================
# HyDE TRIGGER — keyword density / entropy  (Fix #1)
# ==========================================================

def _query_entropy(query: str) -> float:
    """Compute token-level entropy as a proxy for query specificity."""
    tokens = query.lower().split()
    if not tokens:
        return 0.0
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens)
    entropy = -sum((c / total) * np.log2(c / total) for c in freq.values())
    return entropy

def _keyword_density(query: str) -> float:
    """Ratio of non-stopword tokens to total tokens."""
    tokens = query.lower().split()
    if not tokens:
        return 0.0
    keywords = [t for t in tokens if t not in STOPWORDS]
    return len(keywords) / len(tokens)

def should_use_hyde(query: str) -> bool:
    """
    Use HyDE when the query is vague or abstract:
      - Low keyword density  (few content-bearing words)
      - Low entropy          (repetitive or very short vocabulary)
    These signal that the raw query is unlikely to match document language well.
    """
    if not HYDE_ENABLED:
        logger.info("HyDE is disabled via environment variable.")
        return False
    
    density = _keyword_density(query)
    entropy = _query_entropy(query)
    vague = density < 0.4 or entropy < 1.0
    logger.info(f"HyDE decision: density={density:.2f}, entropy={entropy:.2f}, use_hyde={vague}")
    return vague

# ==========================================================
# LOAD NEW DOCUMENTS SAFELY
# ==========================================================

def is_digital_pdf(filepath: str) -> bool:
    """
    Returns True if the PDF has a real text layer.
    Heuristic: average extractable characters per page >= MIN_CHARS_PER_PAGE.
    Falls back to False (treat as scanned) on any error.
    """
    try:
        doc = fitz.open(filepath)
        total_chars = sum(len(page.get_text()) for page in doc)
        avg = total_chars / max(len(doc), 1)
        doc.close()
        logger.info(f"PDF text check: {filepath} — avg chars/page={avg:.1f}")
        return avg >= MIN_CHARS_PER_PAGE
    except Exception as e:
        logger.warning(f"Could not inspect PDF {filepath}, treating as scanned: {e}")
        return False


def delete_document_chunks(filepath: str) -> None:
    """Remove all chunks belonging to a file from Chroma."""
    try:
        results = vectorstore.get(where={"source": filepath})
        if results and results["ids"]:
            vectorstore.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} old chunks for: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to delete old chunks for {filepath}: {e}")
        
        
def load_single_document(filepath: str) -> List[Document]:
    """
    Load one file using the appropriate strategy:
      - .txt / .md / etc  → TextLoader
      - .pdf (digital)    → PyMuPDFLoader
      - .pdf (scanned)    → look for paired .txt OCR file; if missing, warn and skip
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext != ".pdf":
        try:
            return TextLoader(filepath, encoding="utf-8").load()
        except Exception as e:
            logger.warning(f"TextLoader failed for {filepath}: {e}")
            return []

    # --- PDF path ---
    if is_digital_pdf(filepath):
        logger.info(f"Loading as digital PDF: {filepath}")
        try:
            return PyMuPDFLoader(filepath).load()
        except Exception as e:
            logger.warning(f"PyMuPDFLoader failed for {filepath}: {e}")
            return []
    else:
        # Scanned PDF — look for paired OCR .txt
        ocr_path = os.path.splitext(filepath)[0] + ".txt"
        if os.path.exists(ocr_path):
            logger.info(f"Loading scanned PDF via OCR txt: {ocr_path}")
            try:
                docs = TextLoader(ocr_path, encoding="utf-8").load()
                # Keep source metadata pointing at the PDF, not the .txt
                for doc in docs:
                    doc.metadata["source"] = filepath
                    doc.metadata["ocr_source"] = ocr_path
                return docs
            except Exception as e:
                logger.warning(f"Failed loading OCR file {ocr_path}: {e}")
                return []
        else:
            # Fallback: attempt OCR via pymupdf's built-in if available,
            # otherwise warn clearly so the user knows what's missing
            logger.warning(
                f"Scanned PDF with no paired .txt found: {filepath}\n"
                f"  Expected OCR file at: {ocr_path}\n"
                f"  Either run OCR and place output at that path, "
                f"or install tesseract + pymupdf[ocr] for automatic fallback."
            )
            try:
                # pymupdf can call tesseract internally if installed
                import pytesseract  # noqa: F401 — just checking it's available
                logger.info(f"Attempting automatic OCR via tesseract for: {filepath}")
                doc_fitz = fitz.open(filepath)
                full_text = ""
                for page in doc_fitz:
                    full_text += page.get_textpage_ocr().extractText()
                doc_fitz.close()
                if full_text.strip():
                    return [Document(
                        page_content=full_text,
                        metadata={"source": filepath, "ocr_method": "tesseract_auto"}
                    )]
            except ImportError:
                logger.warning("pytesseract not installed — automatic OCR fallback unavailable.")
            except Exception as e:
                logger.warning(f"Automatic OCR failed for {filepath}: {e}")
            return []


def build_bm25_index():
    global bm25, bm25_docs
    try:
        raw = vectorstore.get(include=["documents", "metadatas"])
        bm25_docs = [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(raw["documents"], raw["metadatas"])
        ]
        bm25_corpus = [tokenize(doc.page_content) for doc in bm25_docs]
        bm25 = BM25Okapi(bm25_corpus)
        logger.info(f"BM25 index built over {len(bm25_docs)} chunks.")
    except Exception as e:
        logger.warning(f"BM25 initialization failed: {e}")
        bm25 = None
        bm25_docs = []


def get_new_documents() -> List[Document]:
    registry = load_registry()
    updated_registry = registry.copy()
    new_docs = []

    # Collect all files manually so we can apply per-file logic
    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            all_files.append(os.path.join(root, fname))

    # Skip .txt files that are OCR companions to a PDF —
    # they get loaded through the PDF path, not directly
    ocr_companions = {
        os.path.splitext(f)[0] + ".txt"
        for f in all_files
        if f.lower().endswith(".pdf")
    }

    for filepath in all_files:
        if filepath in ocr_companions:
            logger.debug(f"Skipping OCR companion (handled via PDF): {filepath}")
            continue

        current_hash = file_hash(filepath)
        if current_hash is None:
            continue

        if filepath not in registry or registry[filepath]["hash"] != current_hash:
            if filepath in registry:  # it's an update, not a new file
                 delete_document_chunks(filepath)
            docs = load_single_document(filepath)
            if docs:
                new_docs.extend(docs)
                updated_registry[filepath] = {
                    "hash": current_hash,
                    "embedding_model": EMBEDDING_MODEL_NAME,
                }

    save_registry(updated_registry)
    return new_docs
# ==========================================================
# EMBEDDINGS + VECTOR DB
# ==========================================================

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

try:
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
    )
    logger.info("Vector DB loaded.")
except Exception as e:
    logger.critical(f"Vector DB load failed: {e}")
    raise SystemExit("Cannot continue without vector DB.")

# ==========================================================
# INCREMENTAL UPDATE
# ==========================================================

new_docs = get_new_documents()

if new_docs:
    logger.info(f"Embedding {len(new_docs)} new/changed documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(new_docs)

    try:
        BATCH_SIZE = 500  # safe number

        for i in range(0, len(splits), BATCH_SIZE):
            batch = splits[i:i+BATCH_SIZE]
            vectorstore.add_documents(batch)
            logger.info(f"Added batch {i//BATCH_SIZE + 1}")
        # NOTE: No vectorstore.persist() — Chroma auto-persists in v0.4+  (Fix #2)
        logger.info("Vector DB updated.")
        build_bm25_index()
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        build_bm25_index()

# ==========================================================
# RETRIEVAL SETUP
# ==========================================================

dense_retriever = vectorstore.as_retriever(search_kwargs={"k": DENSE_K})

# Fix #3: Build BM25 from Document objects, preserving metadata
try:
    raw = vectorstore.get(include=["documents", "metadatas"])
    bm25_docs: List[Document] = [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(raw["documents"], raw["metadatas"])
    ]
    bm25_corpus = [tokenize(doc.page_content) for doc in bm25_docs]
    bm25 = BM25Okapi(bm25_corpus)
    logger.info(f"BM25 index built over {len(bm25_docs)} chunks.")
except Exception as e:
    logger.warning(f"BM25 initialization failed: {e}")
    bm25 = None
    bm25_docs = []

def bm25_retrieve(query: str, k: int = BM25_K) -> List[Document]:
    if bm25 is None:
        return []
    try:
        preprocessed = preprocess_query(query)
        scores = bm25.get_scores(preprocessed.split())
        top_idx = np.argsort(scores)[-k:][::-1]
        results = [bm25_docs[i] for i in top_idx if scores[i] > 0]
        logger.debug(f"BM25 returned {len(results)} docs (top score: {scores[top_idx[0]]:.3f})")
        return results
    except Exception as e:
        logger.warning(f"BM25 retrieval failed: {e}")
        return []

# ==========================================================
# CROSS ENCODER + CONFIDENCE  (Fix #8)
# ==========================================================

try:
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    logger.info("Cross-encoder loaded.")
except Exception as e:
    logger.warning(f"Cross-encoder load failed: {e}")
    cross_encoder = None


def source_weighted_context(reranked_docs: List[Document], scores: List[float], top_k: int = RERANK_TOP_K) -> List[Document]:
    """
    If one source dominates the top results, allocate more slots to it.
    Single-appearance sources are used only as last resort fallback.
    """
    from collections import Counter

    source_counts = Counter(doc.metadata.get("source", "unknown") for doc in reranked_docs)
    total = len(reranked_docs)
    dominant_source, dominant_count = source_counts.most_common(1)[0]
    dominance_ratio = dominant_count / total

    logger.info(f"Dominant source: {dominant_source} ({dominant_count}/{total} chunks, ratio={dominance_ratio:.2f})")

    if dominance_ratio > 0.5:
        primary_slots = top_k - 1
        secondary_slots = 1
    else:
        primary_slots = top_k // 2
        secondary_slots = top_k // 2

    # Split into three tiers
    primary_docs = [doc for doc in reranked_docs
                    if doc.metadata.get("source") == dominant_source]

    secondary_docs = [doc for doc in reranked_docs
                      if doc.metadata.get("source") != dominant_source
                      and source_counts[doc.metadata.get("source")] > 1]  # multi-chunk sources only

    fallback_docs = [doc for doc in reranked_docs
                     if doc.metadata.get("source") != dominant_source
                     and source_counts[doc.metadata.get("source")] == 1]  # single-appearance sources

    # Fill secondary slots — prefer multi-chunk sources, fall back to single-appearance
    secondary_selected = secondary_docs[:secondary_slots]
    if len(secondary_selected) < secondary_slots:
        remaining = secondary_slots - len(secondary_selected)
        secondary_selected += fallback_docs[:remaining]
        logger.info(f"Used {remaining} fallback (single-appearance) chunk(s) to fill secondary slots.")

    selected = primary_docs[:primary_slots] + secondary_selected

    logger.info(f"Context assembly: {len(primary_docs[:primary_slots])} chunks from dominant source + "
                f"{len(secondary_selected)} from others.")

    return selected



def rerank(question: str, docs: List[Document], top_k: int = RERANK_TOP_K):
    """
    Returns (reranked_docs, scores) for the top_k results.
    Docs scoring below CONFIDENCE_THRESHOLD are discarded.
    """
    if not docs:
        return [], []
    if cross_encoder is None:
        logger.warning("No cross-encoder; returning unscored top-k.")
        return docs[:top_k], [None] * min(top_k, len(docs))

    try:
        pairs = [(question, doc.page_content) for doc in docs]
        scores = cross_encoder.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        # Log all retrieved chunks with scores
        logger.info("=== Retrieved Chunks (pre-filter) ===")
        for i, (doc, score) in enumerate(ranked):
            source = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:120].replace("\n", " ")
            logger.info(f"  [{i+1}] score={score:.3f} | source={source} | {snippet}...")

        # Filter by confidence threshold
        confident = [(doc, s) for doc, s in ranked if s >= CONFIDENCE_THRESHOLD]

        if not confident:
            logger.warning(f"All chunks scored below threshold ({CONFIDENCE_THRESHOLD}). No confident results.")
            return [], []

        top = confident[:top_k]
        return [doc for doc, _ in top], [s for _, s in top]

    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        return docs[:top_k], [None] * min(top_k, len(docs))

# ==========================================================
# CONTEXT WINDOW GUARD  (Fix #6)
# ==========================================================

def build_context(docs: List[Document], max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """
    Concatenate doc chunks up to the token budget.
    Logs a warning if truncation occurs.
    """
    parts = []
    total_chars = 0
    budget_chars = max_tokens * AVG_CHARS_PER_TOKEN

    for i, doc in enumerate(docs):
        chunk = doc.page_content
        if total_chars + len(chunk) > budget_chars:
            logger.warning(
                f"Context budget exceeded at chunk {i+1}/{len(docs)}. "
                f"Truncating to stay within ~{max_tokens} tokens."
            )
            remaining = budget_chars - total_chars
            if remaining > 50:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total_chars += len(chunk)

    return "\n\n".join(parts)

# ==========================================================
# OLLAMA MODEL
# ==========================================================

try:
    llm = Ollama(model=OLLAMA_MODEL)
    logger.info(f"LLM loaded: {OLLAMA_MODEL}")
except Exception as e:
    logger.critical(f"Ollama load failed: {e}")
    raise SystemExit("LLM failed to initialize.")

# ==========================================================
# HyDE MODULE  (Fix #1 — triggered by should_use_hyde)
# ==========================================================

hyde_prompt = ChatPromptTemplate.from_template(
    "Write a detailed answer to the following question as if you were an expert:\n\n{question}"
)
hyde_chain = hyde_prompt | llm | StrOutputParser()

def generate_hypothetical_doc(question: str) -> Optional[str]:
    try:
        result = hyde_chain.invoke({"question": question})
        logger.info(f"HyDE document generated ({len(result)} chars).")
        return result
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return None

# ==========================================================
# HYBRID RETRIEVAL
# ==========================================================

def hybrid_retrieve(question: str, use_hyde: bool = False) -> List[Document]:
    query_for_search = question

    if use_hyde:
        hypothetical = generate_hypothetical_doc(question)
        if hypothetical:
            query_for_search = hypothetical

    # Fix #4: Use .invoke() instead of deprecated get_relevant_documents()
    try:
        dense_docs = dense_retriever.invoke(query_for_search)
        logger.info(f"Dense retrieval: {len(dense_docs)} docs.")
    except Exception as e:
        logger.warning(f"Dense retrieval failed: {e}")
        dense_docs = []

    bm25_results = bm25_retrieve(query_for_search)
    logger.info(f"BM25 retrieval: {len(bm25_results)} docs.")

    # Merge by content (dedup), preserving metadata
    merged: dict[str, Document] = {doc.page_content: doc for doc in dense_docs}
    for doc in bm25_results:
        merged.setdefault(doc.page_content, doc)

    logger.info(f"Merged pool: {len(merged)} unique chunks.")
    return list(merged.values())

# ==========================================================
# FINAL RAG PROMPT — softer, handles partial context  (Fix #7)
# ==========================================================

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

rag_chain = rag_prompt | llm | StrOutputParser()

# ==========================================================
# FINAL PIPELINE
# ==========================================================

def production_rag(question: str) -> str:
    logger.info(f"Question: {question!r}")

    use_hyde = should_use_hyde(question)
    retrieved = hybrid_retrieve(question, use_hyde=use_hyde)
    reranked, scores = rerank(question, retrieved)

    if not reranked:
        return "I don't have enough information to answer that."

    # Use source-weighted context instead of raw reranked
    weighted = source_weighted_context(reranked, scores, top_k=RERANK_TOP_K)
    context = build_context(weighted)

    top_score_str = f"{scores[0]:.3f}" if scores[0] is not None else "N/A"
    logger.info(f"Sending {len(weighted)} chunks to LLM (top score: {top_score_str}).")

    try:
        return rag_chain.invoke({"context": context, "question": question})
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return "Model failed to generate a response."
    

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    question = "what info was retrieved from the cockpit recordings in the mangalore crash"
    answer = production_rag(question)
    print("\nFINAL ANSWER:\n")
    print(answer)