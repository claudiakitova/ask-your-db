"""Find the passages that answer a question. Each passage is searched two ways, by meaning
(embeddings) and by exact words (BM25), and the two rankings are merged."""

import math
import re
from collections import Counter, defaultdict

import numpy as np
from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # small and free; runs inside the app, no key needed
PASSAGE_WORDS = 350  # about 512 tokens, the most the embedding model reads at once
OVERLAP_WORDS = 50   # each passage repeats the end of the one before, so a fact isn't cut in half
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
WORD = re.compile(r"\w+")


def embedder():
    return TextEmbedding(model_name=EMBEDDING_MODEL)


def sentences(text):
    """Split after . ! ? or … followed by a space, keeping the spaces and line breaks."""
    parts, start = [], 0
    for match in SENTENCE_END.finditer(text):
        parts.append(text[start:match.end()])
        start = match.end()
    if start < len(text):
        parts.append(text[start:])
    return parts


def split(text, size=PASSAGE_WORDS, overlap=OVERLAP_WORDS):
    """Cut text into passages of about `size` words that end at sentence ends where possible.
    A sentence longer than a passage (a table, say) is cut by words."""
    units = []
    for sentence in sentences(text):
        words = re.findall(r"\S+\s*", sentence)
        units += [words[i:i + size] for i in range(0, len(words), size)]
    passages, current = [], []
    for unit in units:
        if current and len(current) + len(unit) > size:
            passages.append("".join(current).strip())
            current = current[-overlap:] if overlap else []
        current += unit
    if current:
        passages.append("".join(current).strip())
    return [p for p in passages if p]


def words(text):
    return WORD.findall(text.lower())


class BM25:
    """The standard word-matching score: rare words count more, and a word repeated
    in a short passage counts more than in a long one."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(words(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.average = (sum(self.lengths) / len(self.lengths)) or 1
        count = len(self.docs)
        seen_in = Counter(w for d in self.docs for w in d)
        self.idf = {w: math.log(1 + (count - n + 0.5) / (n + 0.5)) for w, n in seen_in.items()}

    def scores(self, query):
        terms = words(query)
        result = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            for term in terms:
                if term in doc:
                    tf = doc[term]
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.average))
            result.append(score)
        return result


def fuse(rankings, k=60):
    """Reciprocal rank fusion: a passage near the top of either list rises, one near the top of both rises most."""
    score = defaultdict(float)
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            score[int(i)] += 1 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


class Index:
    def __init__(self, parts, model):
        """`parts` come from readers.read: dicts with the text, the file name and where in the file."""
        self.model = model
        self.passages = [{"text": t, "file": p["file"], "where": p["where"]} for p in parts for t in split(p["text"])]
        texts = [p["text"] for p in self.passages]
        self.vectors = unit(list(model.embed(texts)))
        self.bm25 = BM25(texts)

    def search(self, question, top=8):
        top = min(top, len(self.passages))
        query = unit(next(iter(self.model.query_embed(question))))
        by_meaning = np.argsort(-(self.vectors @ query))[:top]
        word_scores = self.bm25.scores(question)
        by_words = [i for i in np.argsort(word_scores)[::-1][:top] if word_scores[i] > 0]
        return [self.passages[i] for i in fuse([by_meaning, by_words])[:top]]
