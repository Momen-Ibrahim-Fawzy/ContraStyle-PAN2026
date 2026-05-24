"""
Stylometric feature extraction for style change detection.

For each sentence we compute a vector of style-level features that are
topic-agnostic: punctuation densities, word-length statistics, vocabulary
richness, function-word ratios, etc.

For each consecutive pair (sentence_i, sentence_{i+1}) we compute:
  - Absolute differences in all per-sentence features
  - Ratio features (max/min for non-negative features)
  - SBERT cosine distance (semantic/style distance)
  - Optional: window-level SBERT similarity (left-window vs. right-window)

These features feed a LightGBM classifier which acts as a complementary
signal to the DeBERTa neural model.
"""
import math
import re
import string
from collections import Counter

import numpy as np
from tqdm import tqdm

# ─── Word lists ───────────────────────────────────────────────────────────────
# Common English function words — their frequency is style-driven, not topic-driven
_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for", "of",
    "to", "in", "on", "at", "by", "with", "from", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "then", "once", "i", "me", "my",
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "not", "no", "nor",
    "both", "either", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "only", "own", "same",
    "than", "too", "very", "just", "because", "if", "while", "although",
    "since", "unless", "until", "when", "where", "whether", "how",
})

# Top-50 function words ordered by approximate corpus frequency.
# Their PER-WORD frequencies form a distribution fingerprint that is
# highly style-discriminative and completely topic-independent.
# We compute abs-difference of these frequency vectors between sentence pairs,
# giving the classifier information about WHICH function words differ, not
# just how many.
_TOP50_FW = [
    "the", "a", "and", "of", "to", "in", "is", "it", "that", "was",
    "for", "on", "are", "with", "as", "at", "be", "this", "by", "or",
    "an", "from", "but", "not", "have", "had", "has", "he", "she", "they",
    "we", "you", "i", "his", "her", "their", "its", "our", "my", "your",
    "what", "which", "who", "will", "would", "can", "could", "should",
    "so", "if",
]
_TOP50_FW_SET = {w: i for i, w in enumerate(_TOP50_FW)}   # word → index

_CONTRACTIONS = re.compile(
    r"\b(don't|can't|won't|isn't|aren't|wasn't|weren't|haven't|hasn't"
    r"|hadn't|doesn't|didn't|wouldn't|shouldn't|couldn't|i'm|i've|i'll"
    r"|i'd|you're|you've|you'll|you'd|he's|she's|it's|we're|we've|we'll"
    r"|we'd|they're|they've|they'll|they'd|that's|who's|what's|there's"
    r"|here's|let's|how's|n't)\b",
    re.IGNORECASE,
)


def _words(text: str) -> list:
    return re.findall(r"\b[a-zA-Z']+\b", text)


def _approx_syllables(word: str) -> int:
    word = word.lower().rstrip(".,!?;:")
    if not word:
        return 0
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in "aeiouy"
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def extract_fw_frequencies(sentence: str) -> np.ndarray:
    """
    Return a 50-dim vector of per-function-word frequencies (count / word_count).

    Each dimension corresponds to one word in _TOP50_FW.  The difference
    between these vectors for a sentence pair captures which specific
    function words change at a boundary — a strong style signal that is
    completely topic-agnostic.
    """
    words = _words(sentence.lower())
    total = max(len(words), 1)
    freq  = np.zeros(len(_TOP50_FW), dtype=np.float32)
    for w in words:
        idx = _TOP50_FW_SET.get(w)
        if idx is not None:
            freq[idx] += 1
    return freq / total


def extract_sentence_features(sentence: str) -> np.ndarray:
    """
    Compute a style feature vector for a single sentence.

    All features are topic-agnostic: they measure how text is written,
    not what it is about.

    Returns a 1-D float32 array of length N_FEATURES (see FEATURE_NAMES).
    """
    text = sentence.strip()
    char_count = max(len(text), 1)

    words = _words(text)
    word_count = max(len(words), 1)
    word_lengths = [len(w) for w in words] if words else [0]

    lower = [w.lower() for w in words]
    unique = set(lower)
    freq = Counter(lower)

    # Punctuation counts
    n_comma   = text.count(",")
    n_period  = text.count(".")
    n_exclaim = text.count("!")
    n_question = text.count("?")
    n_semi    = text.count(";")
    n_colon   = text.count(":")
    n_dash    = text.count("—") + text.count("–") + text.count(" - ")
    n_quote   = text.count('"') + text.count("'") + text.count("“") + text.count("”")
    n_paren   = text.count("(") + text.count(")")
    n_ellipsis = text.count("...")
    n_punct   = sum(1 for c in text if c in string.punctuation)

    # Vocabulary richness
    hapax = sum(1 for c in freq.values() if c == 1)
    dis   = sum(1 for c in freq.values() if c == 2)

    # Syllables
    syllables = [_approx_syllables(w) for w in words] if words else [1]

    # First / third person pronouns
    n_first  = sum(1 for w in lower if w in {"i","me","my","myself","mine","we","us","our","ours","ourselves"})
    n_third  = sum(1 for w in lower if w in {"he","she","they","him","her","them","his","hers","their","theirs"})

    feats = [
        # Length stats
        char_count,
        word_count,
        np.mean(word_lengths),
        np.std(word_lengths) if len(word_lengths) > 1 else 0.0,
        # Vocabulary richness
        len(unique) / word_count,                  # TTR
        hapax / word_count,                        # hapax ratio
        dis / word_count,                          # dis-legomena ratio
        # Punctuation densities (per word)
        n_comma    / word_count,
        n_period   / word_count,
        n_exclaim  / word_count,
        n_question / word_count,
        n_semi     / word_count,
        n_colon    / word_count,
        n_dash     / word_count,
        n_quote    / word_count,
        n_paren    / word_count,
        n_ellipsis / word_count,
        n_punct    / char_count,
        # Character ratios
        sum(1 for c in text if c.isupper()) / char_count,  # uppercase ratio
        sum(1 for c in text if c.isdigit()) / char_count,  # digit ratio
        # Lexical
        sum(1 for w in lower if w in _FUNCTION_WORDS) / word_count,  # function word ratio
        len(_CONTRACTIONS.findall(text)) / word_count,                # contraction density
        # Syllabic complexity
        np.mean(syllables),
        sum(1 for s in syllables if s >= 3) / word_count,  # complex word ratio
        # Pronoun usage (style signal)
        n_first  / word_count,
        n_third  / word_count,
        # Long / short word ratio
        sum(1 for l in word_lengths if l > 8) / word_count,
        sum(1 for l in word_lengths if l < 4) / word_count,
    ]
    return np.array(feats, dtype=np.float32)


# Number of features in extract_sentence_features output
N_SENT_FEATURES = 28

FEATURE_NAMES_SENT = [
    "char_count", "word_count", "avg_word_len", "std_word_len",
    "ttr", "hapax_ratio", "dis_ratio",
    "comma_per_word", "period_per_word", "exclaim_per_word", "question_per_word",
    "semi_per_word", "colon_per_word", "dash_per_word", "quote_per_word",
    "paren_per_word", "ellipsis_per_word", "punct_per_char",
    "uppercase_ratio", "digit_ratio",
    "function_word_ratio", "contraction_density",
    "avg_syllables", "complex_word_ratio",
    "first_person_ratio", "third_person_ratio",
    "long_word_ratio", "short_word_ratio",
]


N_TOP50_FW = len(_TOP50_FW)   # 50


def extract_pair_features(
    sent_i: str,
    sent_j: str,
    left_window: str = "",
    right_window: str = "",
) -> np.ndarray:
    """
    Compute pairwise features for (sent_i, sent_j).

    Features:
      - Absolute differences of all per-sentence features (N_SENT_FEATURES values)
      - Ratio max/min for positive-definite features (N_SENT_FEATURES values)
      - SBERT cosine distance between the two sentences (1 value)
      - SBERT cosine distance between left-window and right-window (1 value)
      - Absolute diff of top-50 function-word frequency vectors (N_TOP50_FW values)

    Total: 2 * N_SENT_FEATURES + 2 + N_TOP50_FW = 108 features.
    """
    fi = extract_sentence_features(sent_i)
    fj = extract_sentence_features(sent_j)

    diff  = np.abs(fi - fj)
    eps   = 1e-6
    ratio = np.minimum(np.maximum(fi, fj) / (np.minimum(fi, fj) + eps), 100.0)

    sbert_pair   = _sbert_cosine_distance(sent_i, sent_j)
    sbert_window = (_sbert_cosine_distance(left_window, right_window)
                    if left_window and right_window else sbert_pair)

    fw_i   = extract_fw_frequencies(sent_i)
    fw_j   = extract_fw_frequencies(sent_j)
    fw_diff = np.abs(fw_i - fw_j)

    return np.concatenate(
        [diff, ratio, [sbert_pair, sbert_window], fw_diff]
    ).astype(np.float32)


N_PAIR_FEATURES = 2 * N_SENT_FEATURES + 2 + N_TOP50_FW   # 108


# ─── SBERT model (lazily loaded, process-level singleton) ────────────────────
_sbert_model = None
_SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_sbert():
    global _sbert_model
    if _sbert_model is not None:
        return _sbert_model
    try:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(_SBERT_MODEL_NAME)
        return _sbert_model
    except ImportError:
        return None


def _sbert_cosine_distance(text_a: str, text_b: str) -> float:
    """
    1 - cosine_similarity between SBERT embeddings.
    Returns 0.5 as a neutral fallback when SBERT is unavailable.
    """
    sbert = _get_sbert()
    if sbert is None:
        return 0.5
    try:
        embs = sbert.encode([text_a, text_b], convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        sim = float(np.dot(embs[0], embs[1]))
        return float(1.0 - sim)
    except Exception:
        return 0.5


def build_pair_feature_matrix(records, window_size: int = 1) -> np.ndarray:
    """
    Build feature matrix for a list of PairRecord objects.

    Returns (N, N_PAIR_FEATURES) float32 array.
    """
    from .data import get_pair_texts

    if not records:
        return np.zeros((0, N_PAIR_FEATURES), dtype=np.float32)

    # ── Pre-batch all SBERT encodings ────────────────────────────────────────
    # Collect all unique texts first, encode in one batch, then look up by text.
    # This replaces 2-sentence-at-a-time calls (1.26M tiny SBERT calls → 1 batch).
    sbert = _get_sbert()
    sent_embs = {}  # text → embedding (np.ndarray)
    if sbert is not None:
        print("  Pre-computing SBERT embeddings...")
        all_texts = []
        for rec in records:
            left_text, right_text = get_pair_texts(rec, window_size=window_size)
            i = rec.pair_idx
            all_texts.append(rec.sentences[i])
            all_texts.append(rec.sentences[i + 1])
            all_texts.append(left_text)
            all_texts.append(right_text)

        unique_texts = list(dict.fromkeys(all_texts))  # deduplicate, preserve order
        print(f"  Encoding {len(unique_texts):,} unique texts with SBERT...")
        embs = sbert.encode(
            unique_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=512,
        )
        sent_embs = dict(zip(unique_texts, embs))

    # ── Compute stylometric features using cached embeddings ─────────────────
    rows = []
    for rec in tqdm(records, desc="Extracting features", unit="pair", dynamic_ncols=True):
        left_text, right_text = get_pair_texts(rec, window_size=window_size)
        i = rec.pair_idx
        sents = rec.sentences
        sent_i = sents[i]
        sent_j = sents[i + 1]

        fi = extract_sentence_features(sent_i)
        fj = extract_sentence_features(sent_j)
        diff  = np.abs(fi - fj)
        ratio = np.minimum(np.maximum(fi, fj) / (np.minimum(fi, fj) + 1e-6), 100.0)

        if sent_embs:
            ei, ej = sent_embs[sent_i], sent_embs[sent_j]
            sbert_pair = float(1.0 - np.dot(ei, ej))
            el, er = sent_embs[left_text], sent_embs[right_text]
            sbert_window = float(1.0 - np.dot(el, er)) if (left_text and right_text) else sbert_pair
        else:
            sbert_pair = sbert_window = 0.5

        fw_diff = np.abs(extract_fw_frequencies(sent_i) - extract_fw_frequencies(sent_j))

        rows.append(np.concatenate(
            [diff, ratio, [sbert_pair, sbert_window], fw_diff]
        ).astype(np.float32))

    return np.vstack(rows).astype(np.float32)


def get_labels(records) -> np.ndarray:
    """Extract labels from a list of PairRecord objects."""
    return np.array([r.label for r in records], dtype=np.int32)