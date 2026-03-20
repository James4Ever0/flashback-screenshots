"""BM25 text search for flashback."""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from flashback.core.config import Config
from flashback.core.database import Database
from flashback.search.tokenizer import get_tokenizer


class BM25Search:
    """BM25 ranking for OCR text search."""

    def __init__(self, config: Config = None, db: Database = None):
        self.config = config or Config()
        self.db = db or Database(self.config.db_path)
        self.k1 = self.config.get("search.bm25.k1", 1.5)
        self.b = self.config.get("search.bm25.b", 0.75)

        # Initialize tokenizer from config
        tokenizer_config = self.config.get("search.bm25.tokenizer", {})
        self.tokenizer = get_tokenizer(tokenizer_config)

        # Index data
        self.doc_lengths: Dict[int, int] = {}
        self.avg_dl = 0.0
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.N = 0

        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms using configured tokenizer."""
        return self.tokenizer.tokenize(text)

    def _build_index(self):
        """Build inverted index from database."""
        records = self.db.get_all_ocr_text()
        total_length = 0

        for doc_id, text in records:
            if not text:
                continue

            tokens = self._tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # Count term frequencies
            term_counts: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_counts[token] += 1

            # Add to inverted index
            for term, freq in term_counts.items():
                self.inverted_index[term].append((doc_id, freq))
                self.doc_freqs[term] += 1

        self.N = len(self.doc_lengths)
        self.avg_dl = total_length / self.N if self.N > 0 else 0

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Search for query and return ranked document IDs with scores."""
        query_terms = self._tokenize(query)
        scores: Dict[int, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            df = self.doc_freqs[term]
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in self.inverted_index[term]:
                dl = self.doc_lengths.get(doc_id, 0)
                if dl == 0:
                    continue

                # BM25 formula
                denom = self.k1 * (1 - self.b + self.b * (dl / self.avg_dl)) + tf
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[doc_id] += score

        # Sort by score descending
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def refresh(self):
        """Rebuild the index from database."""
        self.doc_lengths.clear()
        self.doc_freqs.clear()
        self.inverted_index.clear()
        self._build_index()
