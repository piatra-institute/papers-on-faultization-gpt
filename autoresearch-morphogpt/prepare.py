# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Data preparation and evaluation for autoresearch-morphogpt experiments.
Downloads names.txt, provides dataset loading, tokenization, and evaluation.

This file is READ-ONLY for the agent. Do not modify.

Usage:
    uv run --script prepare.py          # download data, print stats
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16          # context length
TIME_BUDGET = 30         # training time budget in seconds
VOCAB_SIZE = 27          # 26 letters + BOS
VAL_FRACTION = 0.1       # fraction of docs held out for validation

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "input.txt")
NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"


def download_data():
    """Download names.txt to data/input.txt if not present."""
    if os.path.exists(DATA_PATH):
        print(f"Data: already exists at {DATA_PATH}")
        return
    import urllib.request
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading {NAMES_URL} ...")
    urllib.request.urlretrieve(NAMES_URL, DATA_PATH)
    print(f"Data: saved to {DATA_PATH}")


# ---------------------------------------------------------------------------
# Dataset loading and tokenization
# ---------------------------------------------------------------------------

def load_dataset():
    """
    Load names dataset.
    Returns (docs, uchars, BOS, vocab_size).
    """
    if not os.path.exists(DATA_PATH):
        download_data()
    docs = [line.strip() for line in open(DATA_PATH) if line.strip()]
    uchars = sorted(set("".join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    return docs, uchars, BOS, vocab_size


def split_dataset(docs, seed=42):
    """
    Deterministic 90/10 train/val split.
    Returns (train_docs, val_docs).
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(docs))
    n_val = max(1, int(len(docs) * VAL_FRACTION))
    val_indices = set(indices[:n_val])
    train_docs = [docs[i] for i in range(len(docs)) if i not in val_indices]
    val_docs = [docs[i] for i in range(len(docs)) if i in val_indices]
    return train_docs, val_docs


def tokenize(doc, uchars, BOS):
    """Tokenize a name: [BOS] + char indices + [BOS]."""
    return [BOS] + [uchars.index(ch) for ch in doc] + [BOS]


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE -- this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_val_loss(forward_fn, state_dict, config, val_docs, uchars, BOS):
    """
    Evaluate mean cross-entropy loss over all validation documents.

    Args:
        forward_fn: callable(tokens, n, state_dict, config) -> (loss, grads)
            The training script provides this. Only loss is used.
        state_dict: model weights dict
        config: model config dict
        val_docs: list of validation document strings
        uchars: character vocabulary
        BOS: BOS token id

    Returns:
        float: mean cross-entropy loss over all val docs
    """
    block_size = config["block_size"]
    total_loss = 0.0
    total_positions = 0

    for doc in val_docs:
        tokens = tokenize(doc, uchars, BOS)
        n = min(block_size, len(tokens) - 1)
        if n == 0:
            continue
        loss, _ = forward_fn(tokens, n, state_dict, config)
        total_loss += loss * n
        total_positions += n

    if total_positions == 0:
        return float("inf")
    return total_loss / total_positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_data()
    docs, uchars, BOS, vocab_size = load_dataset()
    train_docs, val_docs = split_dataset(docs)

    print()
    print(f"Dataset stats:")
    print(f"  Total docs:    {len(docs)}")
    print(f"  Train docs:    {len(train_docs)}")
    print(f"  Val docs:      {len(val_docs)}")
    print(f"  Vocab size:    {vocab_size}")
    print(f"  Characters:    {''.join(uchars)}")
    print(f"  Block size:    {BLOCK_SIZE}")
    print(f"  Time budget:   {TIME_BUDGET}s")

    # Token stats
    total_tokens = sum(len(tokenize(d, uchars, BOS)) - 1 for d in docs)
    val_tokens = sum(len(tokenize(d, uchars, BOS)) - 1 for d in val_docs)
    print(f"  Total tokens:  {total_tokens}")
    print(f"  Val tokens:    {val_tokens}")
