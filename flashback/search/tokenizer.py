"""Tokenizer backends for search (BM25)."""

import re
from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    """Base class for tokenizers."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of terms."""
        pass


class SimpleTokenizer(BaseTokenizer):
    """Simple regex tokenizer (ASCII only)."""

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"[a-zA-Z0-9]+", text.lower())


class NLTKTokenizer(BaseTokenizer):
    """NLTK word tokenizer for English."""

    def __init__(self, auto_download: bool = True):
        self.auto_download = auto_download
        self._ready = False

    def _ensure_data(self):
        if self._ready:
            return
        try:
            import nltk
            nltk.word_tokenize("test")
            self._ready = True
        except LookupError:
            if self.auto_download:
                import nltk
                nltk.download('punkt', quiet=True)
                self._ready = True

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        try:
            self._ensure_data()
            import nltk
            return nltk.word_tokenize(text.lower())
        except Exception:
            # Fallback to simple tokenizer
            return SimpleTokenizer().tokenize(text)


class JiebaTokenizer(BaseTokenizer):
    """Jieba tokenizer for Chinese."""

    def __init__(self, mode: str = "accurate"):
        self.mode = mode
        self._jieba = None

    def _ensure_jieba(self):
        if self._jieba is None:
            import jieba
            self._jieba = jieba
        return self._jieba

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        try:
            jieba = self._ensure_jieba()
            if self.mode == "search":
                return list(jieba.cut_for_search(text))
            elif self.mode == "full":
                return list(jieba.cut(text, cut_all=True))
            else:  # accurate
                return list(jieba.cut(text, cut_all=False))
        except Exception:
            # Fallback to simple tokenizer
            return SimpleTokenizer().tokenize(text)


class AutoTokenizer(BaseTokenizer):
    """Auto-detect language and use appropriate tokenizer."""

    def __init__(self, config: dict):
        self.config = config
        self.nltk = NLTKTokenizer(
            auto_download=config.get("nltk", {}).get("auto_download", True)
        )
        self.jieba = JiebaTokenizer(
            mode=config.get("jieba", {}).get("mode", "accurate")
        )
        self.simple = SimpleTokenizer()
        self.threshold = config.get("language_confidence_threshold", 0.7)

    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Chinese or English."""
        if not text:
            return "simple"

        # Count Chinese characters (CJK range)
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.strip())

        if total_chars == 0:
            return "simple"

        ratio = chinese_chars / total_chars
        if ratio > self.threshold:
            return "chinese"
        elif ratio < 0.1:  # Less than 10% Chinese, assume English
            return "english"
        else:
            return "mixed"

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        lang = self._detect_language(text)

        if lang == "chinese":
            return self.jieba.tokenize(text)
        elif lang == "english":
            return self.nltk.tokenize(text)
        else:  # mixed or unknown
            # For mixed text, try jieba first (handles both)
            try:
                return self.jieba.tokenize(text)
            except:
                return self.nltk.tokenize(text)


def get_tokenizer(config: dict) -> BaseTokenizer:
    """Factory function to get tokenizer based on config.

    Args:
        config: Tokenizer configuration dict

    Returns:
        Tokenizer instance
    """
    backend = config.get("backend", "auto")

    if backend == "nltk":
        return NLTKTokenizer(
            auto_download=config.get("nltk", {}).get("auto_download", True)
        )
    elif backend == "jieba":
        return JiebaTokenizer(
            mode=config.get("jieba", {}).get("mode", "accurate")
        )
    elif backend == "simple":
        return SimpleTokenizer()
    else:  # auto
        return AutoTokenizer(config)
