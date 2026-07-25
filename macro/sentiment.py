"""Sentiment scoring — FinBERT for financial text analysis."""

from __future__ import annotations

from typing import Any

from utils.logger import get_logger

log = get_logger("macro.sentiment")

_model = None
_tokenizer = None


def _load_model():
    """Lazy-load FinBERT model. Only loads once."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_name = "ProsusAI/finbert"
        log.info(f"Loading sentiment model: {model_name}")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.eval()
        log.info("FinBERT loaded successfully")
        return _model, _tokenizer
    except Exception as e:
        log.error(f"Failed to load FinBERT: {e}")
        return None, None


def score_text(text: str) -> dict[str, float]:
    """Score a single text. Returns {positive, negative, neutral} probabilities."""
    model, tokenizer = _load_model()
    if model is None:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    labels = ["positive", "negative", "neutral"]
    return {label: round(prob.item(), 4) for label, prob in zip(labels, probs)}


def score_batch(texts: list[str]) -> list[dict[str, float]]:
    """Score multiple texts efficiently."""
    model, tokenizer = _load_model()
    if model is None:
        return [{"positive": 0.33, "negative": 0.33, "neutral": 0.34}] * len(texts)

    import torch
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    labels = ["positive", "negative", "neutral"]
    results = []
    for i in range(len(texts)):
        results.append({label: round(probs[i][j].item(), 4) for j, label in enumerate(labels)})
    return results


def directional_score(sentiment: dict[str, float]) -> float:
    """Convert sentiment dict to a single directional score: -1 (bearish) to +1 (bullish)."""
    return sentiment.get("positive", 0) - sentiment.get("negative", 0)


def score_news_batch(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score a batch of news articles and attach sentiment."""
    texts = [a.get("title", "") + ". " + a.get("summary", "") for a in articles]
    if not texts:
        return articles

    scores = score_batch(texts)
    for article, score in zip(articles, scores):
        article["sentiment"] = score
        article["direction"] = directional_score(score)
    return articles
