#!/usr/bin/env python3
"""
fit_topic_model.py — One-time script to fit the tweet topic space.

Run this on the server (needs a GPU and `pip install bertopic umap-learn hdbscan`).

The model is fit on tweets only; bill text is projected into the resulting
space at retrieval time. Short tweets and duplicates are excluded from the
fit but remain fully retrievable afterwards.

Usage:
    python fit_topic_model.py
    python fit_topic_model.py --fit-sample 1000000 --min-cluster-size 150
    python fit_topic_model.py --nr-topics 200          # force reduction

Outputs (into config.TOPIC_DIR):
    topic_centroids.npy / topic_ids.npy / topic_labels.json / topic_meta.json
    tweet_topics.parquet, tweet_embeddings.npy, topic_info.csv
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fit_topic_model")

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from pipeline import config          # noqa: E402
from pipeline import topic_model     # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the BERTopic topic space over the tweet corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=config.TWEETS_TOPIC_PKL,
        help="Cleaned tweet corpus pickle (needs id, bioguide_id, created_at, text).",
    )
    parser.add_argument(
        "--fit-sample",
        type=int,
        default=config.TOPIC_FIT_SAMPLE,
        dest="fit_sample",
        help="Documents used to fit UMAP/HDBSCAN. 0 = use the whole corpus.",
    )
    parser.add_argument(
        "--min-fit-words",
        type=int,
        default=config.TOPIC_MIN_FIT_WORDS,
        dest="min_fit_words",
        help="Exclude tweets shorter than this from FITTING (they stay retrievable).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=config.TOPIC_MIN_CLUSTER_SIZE,
        dest="min_cluster_size",
        help="HDBSCAN min_cluster_size — larger means fewer, broader topics.",
    )
    parser.add_argument(
        "--nr-topics",
        default=config.TOPIC_NR_TOPICS,
        dest="nr_topics",
        help="'auto', an int to force reduction, or 'none' to skip reduction.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force the embedding device, e.g. 'cuda' or 'cuda:0'. "
             "Omit to auto-detect (a CPU fallback is warned about loudly).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.EMBEDDING_BATCH_SIZE,
        dest="batch_size",
        help="Embedding batch size. Raise it on a large GPU.",
    )
    parser.add_argument(
        "--embedding-model",
        default=config.EMBEDDING_MODEL,
        dest="embedding_model",
        help="Sentence-transformer model. all-MiniLM-L6-v2 is ~3-5x faster.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        dest="no_fp16",
        help="Disable fp16 on CUDA (fp16 is on by default and ~2x faster).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    nr_topics = args.nr_topics
    if isinstance(nr_topics, str):
        if nr_topics.lower() == "none":
            nr_topics = None
        elif nr_topics.lower() != "auto":
            nr_topics = int(nr_topics)

    # Apply embedding overrides before anything touches the model.
    config.EMBEDDING_MODEL = args.embedding_model
    config.EMBEDDING_BATCH_SIZE = args.batch_size
    if args.device:
        config.EMBEDDING_DEVICE = args.device
    if args.no_fp16:
        config.EMBEDDING_FP16 = False

    logger.info(
        "Fitting topic space | corpus=%s | fit_sample=%s | min_cluster_size=%d | nr_topics=%s",
        args.corpus, args.fit_sample or "all", args.min_cluster_size, nr_topics,
    )
    logger.info(
        "Embedding | model=%s | batch=%d | device=%s | fp16=%s",
        config.EMBEDDING_MODEL, config.EMBEDDING_BATCH_SIZE,
        topic_model.get_device(), config.EMBEDDING_FP16,
    )

    corpus = topic_model.load_topic_corpus(args.corpus)

    space = topic_model.fit_topic_space(
        corpus=corpus,
        fit_sample=args.fit_sample,
        min_fit_words=args.min_fit_words,
        min_cluster_size=args.min_cluster_size,
        nr_topics=nr_topics,
        seed=args.seed,
        save=True,
    )

    logger.info("Done. %d topics fitted.", len(space.topic_ids))
    logger.info("Top topics:")
    for tid in space.topic_ids[:20]:
        logger.info("  %4d  %s", tid, space.label(int(tid)))


if __name__ == "__main__":
    main()
