"""
topic_model.py — Shared topic space over tweets and bills.

The topic model is fit on **tweets only**, then bill text is projected into
that space. Fitting jointly does not work: bill summaries run to a median of
173 words (mean 1,375) against ~25 for tweets, so a joint fit produces
separate legalese clusters that never match tweet clusters.

Topic distributions are computed from **persisted topic centroids** rather
than BERTopic internals (`topic_embeddings_`), so the retriever does not
depend on BERTopic being installed at inference time, and does not break
across BERTopic versions. Centroids are the L2-normalised mean embedding of
each topic's assigned documents, excluding the -1 outlier topic.

Heavy dependencies (sentence_transformers, bertopic, umap, hdbscan) are
imported lazily so that importing this module is cheap and safe on machines
where they are not installed.

Public API:
    get_device()                                   -> str
    encode(texts, ...)                             -> np.ndarray  (L2-normalised)
    build_vote_query_texts(title, summary, question) -> list[str]
    TopicSpace.load()                              -> TopicSpace
    load_topic_space()                             -> TopicSpace  (cached)
    fit_topic_space(...)                           -> TopicSpace  (server-side)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

_EMBEDDER = None
_TOPIC_SPACE = None

# Artifact filenames inside config.TOPIC_DIR
_CENTROIDS_NPY = "topic_centroids.npy"
_TOPIC_IDS_NPY = "topic_ids.npy"
_LABELS_JSON   = "topic_labels.json"
_META_JSON     = "topic_meta.json"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_device(preferred: Optional[str] = None) -> str:
    """
    Resolve the torch device: explicit > cuda > mps > cpu.

    Falling back to CPU is loud, not silent. Embedding a 5M-tweet corpus on
    CPU takes on the order of 200 hours versus ~30 minutes on an H100, so a
    silent fallback looks like a hang rather than a misconfiguration.
    """
    if preferred:
        return preferred
    if config.EMBEDDING_DEVICE:
        return config.EMBEDDING_DEVICE
    try:
        import torch
    except ImportError:
        logger.error("torch is not installed — embedding will not work.")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"

    logger.warning(
        "=" * 72 + "\n"
        "NO GPU DETECTED — falling back to CPU.\n"
        "  torch=%s  compiled_cuda=%s  cuda.is_available()=False\n"
        "Embedding throughput on CPU is roughly 10 tweets/s versus several\n"
        "thousand on an H100. A full-corpus run will take days.\n"
        "Check that the kernel has a GPU allocated and that torch is a CUDA\n"
        "build (torch.version.cuda must not be None). To override explicitly,\n"
        "set config.EMBEDDING_DEVICE or pass --device cuda.\n"
        + "=" * 72,
        torch.__version__, torch.version.cuda,
    )
    return "cpu"


def get_embedder(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return a cached SentenceTransformer. Imported lazily."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    from sentence_transformers import SentenceTransformer

    name = model_name or config.EMBEDDING_MODEL
    dev = get_device(device)
    logger.info("Loading embedding model %s on %s …", name, dev)
    _EMBEDDER = SentenceTransformer(name, device=dev)

    # fp16 roughly doubles throughput on CUDA at no measurable cost to
    # cosine-similarity quality. Not supported on CPU/MPS.
    if config.EMBEDDING_FP16 and str(dev).startswith("cuda"):
        _EMBEDDER = _EMBEDDER.half()
        logger.info("Embedding model running in fp16.")

    logger.info(
        "Embedder ready on %s (max_seq_length=%s).",
        getattr(_EMBEDDER, "device", dev),
        getattr(_EMBEDDER, "max_seq_length", "?"),
    )
    return _EMBEDDER


def encode(
    texts: Sequence[str],
    batch_size: Optional[int] = None,
    show_progress: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    """
    Embed texts and L2-normalise, so cosine similarity is a plain dot product.

    Returns shape (len(texts), dim). An empty input returns an empty array.
    """
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=dtype)

    model = get_embedder()
    emb = model.encode(
        list(texts),
        batch_size=batch_size or config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# Vote-side text
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_words: Optional[int] = None,
    max_chunks: Optional[int] = None,
) -> List[str]:
    """
    Split text into word-count chunks. Long bill summaries are chunked rather
    than embedded whole, because a single vector over 1,000+ words averages
    the specific policy content away.
    """
    chunk_words = chunk_words or config.TOPIC_SUMMARY_CHUNK_WORDS
    max_chunks = max_chunks or config.TOPIC_MAX_SUMMARY_CHUNKS

    words = str(text).split()
    if not words:
        return []
    chunks = [
        " ".join(words[i : i + chunk_words])
        for i in range(0, len(words), chunk_words)
    ]
    return chunks[:max_chunks]


def build_vote_query_texts(
    bill_title: str,
    bill_summary: str = "",
    question: str = "",
) -> List[str]:
    """
    Build the list of texts representing a vote in the topic space.

    The title (plus vote question for context) is always the first chunk, so
    votes with no summary — treaties, and any vote where the congress.gov
    lookup failed — still get a usable, if weaker, representation.
    """
    title = str(bill_title or "").strip()
    q = str(question or "").strip()
    head = f"{title}. {q}".strip(". ").strip()

    texts: List[str] = []
    if head:
        texts.append(head)

    summary = str(bill_summary or "").strip()
    if summary:
        texts.extend(chunk_text(summary))

    return texts or [title or q or ""]


# ---------------------------------------------------------------------------
# Topic space
# ---------------------------------------------------------------------------

class TopicSpace:
    """
    A fitted topic space: topic centroids in embedding space plus labels.

    centroids : (n_topics, dim) L2-normalised, outlier topic -1 excluded
    topic_ids : (n_topics,) the BERTopic topic id for each centroid row
    labels    : topic_id -> human-readable label
    """

    def __init__(
        self,
        centroids: np.ndarray,
        topic_ids: np.ndarray,
        labels: Optional[Dict[int, str]] = None,
        meta: Optional[dict] = None,
    ):
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.topic_ids = np.asarray(topic_ids)
        self.labels = labels or {}
        self.meta = meta or {}

    # -- scoring ---------------------------------------------------------

    def topic_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Cosine similarity of each embedding to each topic centroid."""
        if embeddings.size == 0 or self.centroids.size == 0:
            return np.zeros((len(embeddings), len(self.centroids)), dtype=np.float32)
        return np.asarray(embeddings, dtype=np.float32) @ self.centroids.T

    def topic_distribution(
        self, embeddings: np.ndarray, temperature: float = 0.05
    ) -> np.ndarray:
        """
        Soft topic distribution: softmax over centroid cosine similarities.

        A distribution rather than a hard label, so a tweet that sits between
        "immigration" and "border security" still matches a bill on either.
        Lower temperature -> peakier, closer to a hard assignment.
        """
        sim = self.topic_similarity(embeddings)
        if sim.size == 0:
            return sim
        scaled = sim / max(temperature, 1e-6)
        scaled -= scaled.max(axis=1, keepdims=True)   # numerical stability
        exp = np.exp(scaled)
        return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)

    def assign(self, embeddings: np.ndarray, min_similarity: Optional[float] = None) -> np.ndarray:
        """Hard topic assignment — nearest centroid's topic id, or -1 if below min_similarity."""
        sim = self.topic_similarity(embeddings)
        if sim.size == 0:
            return np.full(len(embeddings), -1)
        best_idx = sim.argmax(axis=1)
        result = self.topic_ids[best_idx]
        if min_similarity is not None:
            best_sim = sim[np.arange(len(sim)), best_idx]
            result = np.where(best_sim >= min_similarity, result, -1)
        return result

    def label(self, topic_id: int) -> str:
        return self.labels.get(int(topic_id), f"topic_{topic_id}")

    # -- persistence -----------------------------------------------------

    def save(self, directory: Optional[Path] = None) -> None:
        d = Path(directory or config.TOPIC_DIR)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / _CENTROIDS_NPY, self.centroids)
        np.save(d / _TOPIC_IDS_NPY, self.topic_ids)
        (d / _LABELS_JSON).write_text(
            json.dumps({str(k): v for k, v in self.labels.items()}, indent=2),
            encoding="utf-8",
        )
        (d / _META_JSON).write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        logger.info(
            "Saved topic space to %s (%d topics, dim=%d).",
            d, len(self.topic_ids), self.centroids.shape[1] if self.centroids.size else 0,
        )

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "TopicSpace":
        d = Path(directory or config.TOPIC_DIR)
        cpath = d / _CENTROIDS_NPY
        if not cpath.exists():
            raise FileNotFoundError(
                f"No topic space found at {d}. Fit one first:\n"
                f"    python fit_topic_model.py"
            )
        centroids = np.load(cpath)
        topic_ids = np.load(d / _TOPIC_IDS_NPY)

        labels: Dict[int, str] = {}
        lpath = d / _LABELS_JSON
        if lpath.exists():
            labels = {int(k): v for k, v in json.loads(lpath.read_text()).items()}

        meta = {}
        mpath = d / _META_JSON
        if mpath.exists():
            meta = json.loads(mpath.read_text())

        logger.info("Loaded topic space from %s (%d topics).", d, len(topic_ids))
        return cls(centroids, topic_ids, labels, meta)


def load_topic_space(directory: Optional[Path] = None) -> TopicSpace:
    """Cached TopicSpace loader."""
    global _TOPIC_SPACE
    if _TOPIC_SPACE is None:
        _TOPIC_SPACE = TopicSpace.load(directory)
    return _TOPIC_SPACE


# ---------------------------------------------------------------------------
# Fitting (server-side; needs bertopic + umap-learn + hdbscan)
# ---------------------------------------------------------------------------

def load_topic_corpus(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the cleaned tweet corpus used for topic modelling.

    Deliberately reads config.TWEETS_TOPIC_PKL, never config.TWEETS_PKL.
    Fails loudly on a missing required column rather than silently dropping
    rows the way build_tweet_index does.
    """
    p = Path(path or config.TWEETS_TOPIC_PKL)
    if not p.exists():
        raise FileNotFoundError(
            f"Topic corpus not found: {p}\n"
            "Build the cleaned corpus first and point config.TWEETS_TOPIC_PKL at it."
        )
    logger.info("Loading topic corpus from %s …", p)
    df = pd.read_pickle(p)

    required = {"id", "bioguide_id", "created_at", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Topic corpus {p.name} is missing required columns: {sorted(missing)}. "
            f"Present: {sorted(df.columns)}"
        )

    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    logger.info("Topic corpus: %d tweets, %d with bioguide_id.",
                len(df), df["bioguide_id"].notna().sum())
    return df


def _fit_subsample(df: pd.DataFrame, n: int, min_words: int, seed: int) -> pd.DataFrame:
    """
    Pick the documents used to fit UMAP/HDBSCAN.

    Short tweets and exact duplicates are excluded *from fitting only* — they
    add noise to clustering, but they stay fully retrievable at inference
    time (a four-word "vote yes on HR1" can be the most relevant tweet there is).
    """
    n_before = len(df)
    word_counts = df["text"].str.split().str.len()
    fit_pool = df[word_counts >= min_words]
    logger.info(
        "Fit pool: dropped %d tweets with < %d words (%d → %d).",
        n_before - len(fit_pool), min_words, n_before, len(fit_pool),
    )

    n_before = len(fit_pool)
    fit_pool = fit_pool.drop_duplicates(subset=["text"])
    logger.info(
        "Fit pool: dropped %d duplicate texts (%d → %d).",
        n_before - len(fit_pool), n_before, len(fit_pool),
    )

    if n and len(fit_pool) > n:
        fit_pool = fit_pool.sample(n=n, random_state=seed)
        logger.info("Fit pool: subsampled to %d documents.", len(fit_pool))

    return fit_pool


def fit_topic_space(
    corpus: Optional[pd.DataFrame] = None,
    fit_sample: Optional[int] = None,
    min_fit_words: Optional[int] = None,
    min_cluster_size: Optional[int] = None,
    nr_topics=None,
    seed: int = 42,
    save: bool = True,
) -> "TopicSpace":
    """
    Fit BERTopic on tweets, assign every tweet a topic, and derive centroids.

    Writes to config.TOPIC_DIR:
        topic_centroids.npy, topic_ids.npy, topic_labels.json, topic_meta.json
        tweet_topics.parquet   (id, bioguide_id, created_at, topic)
        tweet_embeddings.npy   (float16, row-aligned with tweet_topics.parquet)
        topic_info.csv         (BERTopic's topic overview)

    Run this on the server — it needs a GPU and `pip install bertopic
    umap-learn hdbscan`.
    """
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    fit_sample = config.TOPIC_FIT_SAMPLE if fit_sample is None else fit_sample
    min_fit_words = config.TOPIC_MIN_FIT_WORDS if min_fit_words is None else min_fit_words
    min_cluster_size = (
        config.TOPIC_MIN_CLUSTER_SIZE if min_cluster_size is None else min_cluster_size
    )
    nr_topics = config.TOPIC_NR_TOPICS if nr_topics is None else nr_topics

    df = corpus if corpus is not None else load_topic_corpus()
    fit_df = _fit_subsample(df, fit_sample, min_fit_words, seed)

    # Embed the fit pool.
    logger.info("Embedding %d fit documents …", len(fit_df))
    fit_emb = encode(fit_df["text"].tolist(), show_progress=True)

    # @mentions stay in the text the embedder sees (they carry sentence
    # context) but are suppressed here so topic *labels* describe policy
    # rather than who talks to whom.
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=10,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b",
    )

    logger.info("Fitting BERTopic (min_cluster_size=%d, nr_topics=%s) …",
                min_cluster_size, nr_topics)
    try:
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            prediction_data=True,
        )
    except ImportError:
        clusterer = None
        logger.warning("hdbscan not available — using BERTopic's default clusterer.")

    topic_model = BERTopic(
        embedding_model=get_embedder(),
        vectorizer_model=vectorizer,
        hdbscan_model=clusterer,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )
    fit_topics, _ = topic_model.fit_transform(fit_df["text"].tolist(), fit_emb)
    fit_topics = np.asarray(fit_topics)

    n_topics = len({t for t in fit_topics if t != -1})
    outlier_rate = float((fit_topics == -1).mean())
    logger.info("Fitted %d topics; outlier rate %.1f%%.", n_topics, outlier_rate * 100)

    # Centroids from the fit pool, excluding outliers.
    topic_ids = np.array(sorted({int(t) for t in fit_topics if t != -1}))
    dim = fit_emb.shape[1]
    centroids = np.zeros((len(topic_ids), dim), dtype=np.float32)
    for i, tid in enumerate(topic_ids):
        vecs = fit_emb[fit_topics == tid]
        if len(vecs) == 0:
            continue
        c = vecs.mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[i] = c / norm if norm > 0 else c

    # Data-driven floor for full-corpus assignment: the p-th percentile of
    # how similar the fit pool's own confidently-clustered documents are to
    # their assigned centroid. A tweet the model has never seen that falls
    # below this bar is less well-matched than the worst fit-time member of
    # any real cluster, so it's routed to -1 ("unclassified") instead of
    # being forced into the nearest topic (see TopicSpace.assign).
    assigned_mask = fit_topics != -1
    if assigned_mask.any():
        idx_of = {tid: i for i, tid in enumerate(topic_ids)}
        own_centroid = centroids[[idx_of[t] for t in fit_topics[assigned_mask]]]
        self_sim = np.einsum("ij,ij->i", fit_emb[assigned_mask], own_centroid)
        assign_min_sim = float(np.percentile(self_sim, config.TOPIC_ASSIGN_MIN_SIM_PERCENTILE))
    else:
        assign_min_sim = None
    logger.info("Assign similarity floor (p%d of fit self-similarity): %s",
                config.TOPIC_ASSIGN_MIN_SIM_PERCENTILE, assign_min_sim)

    labels = {-1: "unclassified (below similarity floor)"}
    for tid in topic_ids:
        words = [w for w, _ in (topic_model.get_topic(int(tid)) or [])[:5]]
        labels[int(tid)] = ", ".join(words) if words else f"topic_{tid}"

    space = TopicSpace(
        centroids=centroids,
        topic_ids=topic_ids,
        labels=labels,
        meta={
            "embedding_model": config.EMBEDDING_MODEL,
            "n_fit_docs": int(len(fit_df)),
            "n_topics": int(len(topic_ids)),
            "fit_outlier_rate": outlier_rate,
            "min_cluster_size": int(min_cluster_size),
            "min_fit_words": int(min_fit_words),
            "seed": int(seed),
            "assign_min_similarity": assign_min_sim,
            "assign_min_sim_percentile": config.TOPIC_ASSIGN_MIN_SIM_PERCENTILE,
        },
    )

    if not save:
        return space

    config.TOPIC_DIR.mkdir(parents=True, exist_ok=True)
    space.save()

    topic_model.get_topic_info().to_csv(config.TOPIC_INFO_CSV, index=False)
    logger.info("Wrote topic overview to %s", config.TOPIC_INFO_CSV)

    try:
        topic_model.save(
            str(config.TOPIC_MODEL_DIR),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=config.EMBEDDING_MODEL,
        )
        logger.info("Saved BERTopic model to %s", config.TOPIC_MODEL_DIR)
    except Exception as exc:   # non-fatal: centroids are what inference needs
        logger.warning("Could not save the BERTopic model itself (%s).", exc)

    # Assign a topic to every tweet in the corpus, in chunks.
    _assign_full_corpus(df, space)

    return space


def _assign_full_corpus(df: pd.DataFrame, space: "TopicSpace", chunk: int = 200_000) -> None:
    """Embed and topic-assign the whole corpus, persisting both artifacts."""
    logger.info("Assigning topics to all %d tweets …", len(df))
    n = len(df)
    dim = space.centroids.shape[1]
    all_emb = np.zeros((n, dim), dtype=np.float16)
    topics = np.zeros(n, dtype=np.int32)
    min_sim = space.meta.get("assign_min_similarity")

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        emb = encode(df["text"].iloc[start:end].tolist(), show_progress=True)
        all_emb[start:end] = emb.astype(np.float16)
        topics[start:end] = space.assign(emb, min_similarity=min_sim)
        logger.info("  assigned %d/%d", end, n)

    logger.info("Full-corpus assignment: %.1f%% unclassified (floor=%s).",
                (topics == -1).mean() * 100, min_sim)

    # utc=True forces a single dtype even if the source has mixed
    # timezone-awareness (naive + tz-aware values); without it, to_datetime
    # falls back to object dtype, which is exactly what crashes fastparquet.
    created_at = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    created_at = created_at.dt.tz_localize(None)

    out = pd.DataFrame({
        "id":          df["id"].values,
        "bioguide_id": df["bioguide_id"].values,
        "created_at":  created_at,
        "topic":       topics,
    })
    out.to_parquet(config.TWEET_TOPICS_PARQUET, index=False)
    np.save(config.TWEET_EMBEDDINGS_NPY, all_emb)
    logger.info(
        "Wrote %s and %s (float16, %.1f GB).",
        config.TWEET_TOPICS_PARQUET.name, config.TWEET_EMBEDDINGS_NPY.name,
        all_emb.nbytes / 1e9,
    )
