"""Context Retrieval Agent (CRA), Section 3.1.

Turns the weather observed so far into a short rainfall narrative and retrieves Package-of-Practices
guidance with a hybrid BM25 + dense retriever. The result is the context bundle passed to the ABA.
"""
import os
import re
import json
import math
import logging
from collections import Counter
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from agri_scholar.simulation import weather as wa

log = logging.getLogger("agri_sigra")

# location -> region tag (see scripts/build_rag.py)
LOCATION_TO_REGION = {
    "mandya": "karnataka", "dharwad": "karnataka",
    "davanagere": "karnataka", "haveri": "karnataka",
}


class HybridRetriever:
    """BM25 + dense-embedding hybrid retriever over the agronomic PoP corpus."""

    def __init__(self, corpus_path: str, embed_model_name: str, alpha: float = 0.5,
                 cache_dir: str = None, device: str = None, location_boost: float = 0.25,
                 region_boost: float = 0.10, max_chunks_per_file: int = 2,
                 mmr_lambda: float = 0.7, mmr_pool_size: int = 20):
        self.alpha = alpha
        # score bonus when a chunk names the season's district
        self.location_boost = location_boost
        # smaller bonus for a matching region tag
        self.region_boost = region_boost
        # max chunks per source file in the top-k
        self.max_chunks_per_file = max_chunks_per_file
        # MMR trade-off between relevance and diversity (1.0 = plain top-k)
        self.mmr_lambda = mmr_lambda
        # MMR runs over the top mmr_pool_size candidates only
        self.mmr_pool_size = mmr_pool_size
        with open(corpus_path, "r") as f:
            self.chunks = json.load(f)
        self.corpus_size = len(self.chunks)
        self.texts = [c.get("text", "") or c.get("rule_text", "") for c in self.chunks]
        self.chunk_files = [c.get("file", "") for c in self.chunks]
        self.chunk_locations = [
            {loc.lower() for loc in (c.get("locations") or [])} for c in self.chunks
        ]
        self.chunk_region = [(c.get("region") or "").lower() for c in self.chunks]

        def tokenize(text):
            return re.findall(r"\w+", text.lower())

        # IDF over source files rather than chunks, so one large PDF does not dominate
        self.doc_freqs, self.doc_len = [], []
        word_file_sets = {}
        for text, file in zip(self.texts, self.chunk_files):
            tokens = tokenize(text)
            self.doc_len.append(len(tokens))
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for word in freqs:
                word_file_sets.setdefault(word, set()).add(file)
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 1
        n_files = len(set(self.chunk_files)) or 1
        self.idf = {
            word: math.log(1 + (n_files - len(files) + 0.5) / (len(files) + 0.5))
            for word, files in word_file_sets.items()
        }

        self.embed_model = SentenceTransformer(
            embed_model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        cache_dir = cache_dir or os.path.dirname(os.path.abspath(corpus_path))
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "rag_corpus_embeddings.npy")
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            self.doc_embeddings = cached if cached.shape[0] == self.corpus_size else self._embed_and_cache(cache_path)
        else:
            self.doc_embeddings = self._embed_and_cache(cache_path)

        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self._doc_embeddings_normed = self.doc_embeddings / norms

    def _embed_and_cache(self, cache_path):
        embeddings = self.embed_model.encode(self.texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        np.save(cache_path, embeddings)
        return embeddings

    def _bm25_scores(self, query: str) -> np.ndarray:
        tokens = re.findall(r"\w+", query.lower())
        scores = np.zeros(self.corpus_size)
        k1, b = 1.5, 0.75
        for word in tokens:
            if word not in self.idf:
                continue
            for i in range(self.corpus_size):
                f = self.doc_freqs[i].get(word, 0)
                if f > 0:
                    scores[i] += self.idf[word] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * self.doc_len[i] / self.avgdl))
        return scores

    def _dense_scores(self, query: str) -> np.ndarray:
        q = self.embed_model.encode([query], convert_to_numpy=True)[0]
        norm = np.linalg.norm(q)
        if norm == 0:
            return np.zeros(self.corpus_size)
        return self._doc_embeddings_normed @ (q / norm)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        return np.zeros_like(scores) if hi - lo < 1e-8 else (scores - lo) / (hi - lo)

    def retrieve(self, query: str, top_k: int = 3, location: str = None) -> List[str]:
        sparse = self._normalize(self._bm25_scores(query))
        dense = self._normalize(self._dense_scores(query))
        fused = self.alpha * dense + (1 - self.alpha) * sparse

        if location:
            loc_norm = location.strip().lower()
            boost_mask = np.array(
                [1.0 if loc_norm in locs else 0.0 for locs in self.chunk_locations]
            )
            fused = fused + self.location_boost * boost_mask

            region_norm = LOCATION_TO_REGION.get(loc_norm)
            if region_norm:
                region_mask = np.array(
                    [1.0 if r == region_norm else 0.0 for r in self.chunk_region]
                )
                fused = fused + self.region_boost * region_mask

        order = np.argsort(-fused)

        # greedy MMR selection with the per-file cap
        pool = order[:max(self.mmr_pool_size, top_k)]
        selected, file_counts = [], {}

        def pick_best(candidates, respect_cap):
            best_idx, best_score = None, -np.inf
            for idx in candidates:
                if respect_cap and file_counts.get(self.chunk_files[idx], 0) >= self.max_chunks_per_file:
                    continue
                sim_to_selected = (
                    max(float(self._doc_embeddings_normed[idx] @ self._doc_embeddings_normed[j])
                        for j in selected)
                    if selected else 0.0
                )
                mmr_score = self.mmr_lambda * fused[idx] - (1 - self.mmr_lambda) * sim_to_selected
                if mmr_score > best_score:
                    best_idx, best_score = idx, mmr_score
            return best_idx

        remaining = list(pool)
        while remaining and len(selected) < top_k:
            best_idx = pick_best(remaining, respect_cap=True)
            if best_idx is None:
                # all remaining candidates are at the cap, relax it
                best_idx = pick_best(remaining, respect_cap=False)
            selected.append(best_idx)
            file_counts[self.chunk_files[best_idx]] = file_counts.get(self.chunk_files[best_idx], 0) + 1
            remaining.remove(best_idx)

        # pool too small, fill from the full ranking
        if len(selected) < top_k:
            selected_set = set(selected)
            for idx in order:
                if idx in selected_set:
                    continue
                selected.append(idx)
                if len(selected) >= top_k:
                    break

        return [self.texts[i] for i in selected]


class ContextRetrievalAgent:
    """CRA: numeric weather -> language, plus hybrid retrieval -> a single context bundle string."""

    def __init__(self, rag_corpus_path: str, embed_model_name: str, rag_alpha: float,
                 top_k: int, device: str = None, location_boost: float = 0.25,
                 region_boost: float = 0.10, max_chunks_per_file: int = 2):
        self.retriever = HybridRetriever(rag_corpus_path, embed_model_name, alpha=rag_alpha,
                                          device=device, location_boost=location_boost,
                                          region_boost=region_boost,
                                          max_chunks_per_file=max_chunks_per_file)
        self.top_k = top_k

    def summarize_weather(self, met_path: str, year: int, as_of_day: int) -> dict:
        """Causal narrative: only weather observed through `as_of_day`."""
        return wa.summarize_kharif_season(met_path, year, as_of_day)

    def retrieve_agronomic_context(self, query: str, location: str = None) -> List[str]:
        docs = self.retriever.retrieve(query, top_k=self.top_k, location=location)
        log.info(f"CRA retrieval query='{query}' location='{location}' -> {len(docs)} chunks")
        return docs

    def build_context_bundle(self, location: str, year: int, week_idx: int,
                              weather_narrative: str, extra_query_terms: str = "",
                              mean_radn: float = 20.0, total_recent_rain: float = 0.0,
                              drainage_mm: float = 0.0, is_overcast: bool = False) -> str:
        """Weather narrative plus retrieved guidance for the current week, as passed to the ABA."""
        weather_cues = []
        if is_overcast or mean_radn < 12.0:
            weather_cues.append("low solar radiation cloudy weather kernel set photosynthesis")
        if total_recent_rain > 100.0 or drainage_mm > 50.0:
            weather_cues.append("excess rainfall nitrate leaching supplemental nitrogen top dressing")

        base_query = f"maize cultivation {location} irrigation fertilizer urea nitrogen stress management week {week_idx} {extra_query_terms}"
        query = base_query + " " + " ".join(weather_cues)

        docs = self.retrieve_agronomic_context(query, location=location)
        rag_block = "\n---\n".join(docs) if docs else "(no matching agronomic guidance found)"

        bundle = (
            f"LOCATION: {location} | YEAR: {year} | DECISION WEEK: {week_idx}\n"
            f"WEATHER CONTEXT:\n{weather_narrative}\n\n"
            f"RETRIEVED AGRONOMIC GUIDANCE (hybrid sparse+dense retrieval, top {self.top_k}):\n{rag_block}"
        )
        return bundle
