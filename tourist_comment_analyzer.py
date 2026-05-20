#!/usr/bin/env python3
"""CLI pipeline for offline analysis of tourist comments."""

from __future__ import annotations

import argparse
import html
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
from nltk.stem.snowball import SnowballStemmer
try:
    from bertopic import BERTopic
    from umap import UMAP
    BERTOPIC_AVAILABLE = True
except Exception:
    BERTopic = None
    UMAP = None
    BERTOPIC_AVAILABLE = False
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


SUPPORTED_PALETTES = {
    "viridis": px.colors.sequential.Viridis,
    "cividis": px.colors.sequential.Cividis,
    "plasma": px.colors.sequential.Plasma,
    "inferno": px.colors.sequential.Inferno,
    "magma": px.colors.sequential.Magma,
    "plotly": px.colors.qualitative.Plotly,
    "safe": px.colors.qualitative.Safe,
}

LANGUAGE_MAP = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
}

STOPWORDS = {
    "en": {
        "a", "about", "after", "all", "also", "an", "and", "are", "as", "at", "be", "because",
        "been", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
        "have", "he", "her", "here", "him", "his", "hotel", "i", "if", "in", "is", "it", "its",
        "just", "me", "my", "not", "of", "on", "or", "our", "out", "place", "room", "she",
        "so", "stay", "that", "the", "their", "there", "they", "this", "to", "too", "very",
        "was", "we", "were", "with", "would", "you", "your",
    },
    "es": {
        "a", "al", "algo", "alli", "allí", "ante", "antes", "como", "con", "cuando", "de",
        "del", "desde", "donde", "durante", "e", "el", "él", "ella", "ellas", "ellos", "en",
        "era", "eran", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estado",
        "estan", "están", "estar", "este", "esto", "estos", "fue", "fueron", "ha", "habia",
        "había", "han", "hasta", "hay", "hotel", "la", "las", "le", "les", "lo", "los", "mas",
        "más", "me", "mi", "mis", "muy", "no", "nos", "o", "para", "pero", "por", "que",
        "qué", "se", "ser", "si", "sí", "sin", "sobre", "son", "su", "sus", "tambien",
        "también", "te", "tiene", "todo", "todos", "tu", "un", "una", "unas", "uno", "unos",
        "y", "ya",
    },
    "fr": {
        "a", "afin", "ai", "ainsi", "alors", "au", "aucun", "aussi", "autre", "aux", "avec",
        "avoir", "ce", "ces", "cet", "cette", "comme", "dans", "de", "des", "du", "elle",
        "elles", "en", "est", "et", "etre", "être", "eu", "fait", "hotel", "hôtel", "il",
        "ils", "je", "la", "le", "les", "leur", "lui", "mais", "me", "meme", "même", "mes",
        "mon", "ne", "nos", "notre", "nous", "on", "ou", "où", "par", "pas", "plus", "pour",
        "que", "qui", "sa", "se", "ses", "son", "sont", "sur", "ta", "te", "tes", "toi",
        "ton", "tres", "très", "tu", "un", "une", "vos", "votre", "vous",
    },
}

SENTIMENT_LEXICONS = {
    "en": {
        "positive": {
            "amazing", "beautiful", "best", "clean", "comfortable", "delicious", "enjoyed", "excellent",
            "friendly", "good", "great", "happy", "helpful", "incredible", "love", "lovely", "nice",
            "perfect", "pleasant", "recommend", "relaxing", "safe", "spectacular", "wonderful",
        },
        "negative": {
            "awful", "bad", "broken", "dirty", "disappointing", "expensive", "horrible", "late",
            "loud", "noisy", "poor", "problem", "rude", "slow", "small", "terrible", "uncomfortable",
            "unsafe", "worst",
        },
    },
    "es": {
        "positive": {
            "agradable", "amable", "barato", "bello", "bien", "bonito", "calido", "cálido", "comodo",
            "cómodo", "delicioso", "excelente", "facil", "fácil", "feliz", "genial", "hermoso",
            "increible", "increíble", "limpio", "maravilloso", "mejor", "organizado", "perfecto",
            "profesional", "rapido", "rápido", "recomendable", "rico", "seguro", "volveria", "volvería",
        },
        "negative": {
            "alto", "caro", "decepcionante", "demasiado", "desagradable", "desastre", "dificil",
            "difícil", "expectativas", "feo", "horrible", "humedad", "impuestos", "incomodo",
            "incómodo", "lento", "mal", "malo", "molesto", "peor", "problema", "ruido", "ruidoso",
            "sucio", "terrible", "tarde",
        },
    },
    "fr": {
        "positive": {
            "agreable", "agréable", "beau", "bien", "bon", "calme", "confortable", "delicieux",
            "délicieux", "excellent", "gentil", "joli", "magnifique", "meilleur", "parfait",
            "propre", "recommande", "super",
        },
        "negative": {
            "bruyant", "cher", "decevant", "décevant", "lent", "mal", "mauvais", "pire", "probleme",
            "problème", "sale", "terrible", "trop",
        },
    },
}

PRICE_TERMS = {
    "en": "price value cost expensive cheap fee worth money budget rate",
    "es": "precio valor costo caro barato tarifa dinero presupuesto vale cuesta",
    "fr": "prix valeur cout coût cher pas cher tarif argent budget vaut",
}


@dataclass
class AnalysisResult:
    frame: pd.DataFrame
    topic_summaries: list[dict[str, object]]
    topic_diagnostics: list[dict[str, object]]
    outlier_ngrams: dict[str, list[tuple[str, int]]]
    word_clouds: dict[str, list[dict[str, object]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tourist comments from a CSV and generate offline interactive reports."
    )
    parser.add_argument("csv_path", help="Path to the CSV file.")
    parser.add_argument("text_column", help="Name of the column that contains comments.")
    parser.add_argument("language", help="Target language code, for example: es, en, fr.")
    parser.add_argument("report_title", help="Title used in output reports.")
    parser.add_argument(
        "palette",
        help=f"Plotly color palette. Options: {', '.join(sorted(SUPPORTED_PALETTES))}.",
    )
    parser.add_argument("--output-dir", default="reports", help="Directory for generated files.")
    parser.add_argument("--min-topic-docs", type=int, default=8, help="Minimum docs needed to train topics.")
    parser.add_argument("--topics", type=int, default=4, help="Maximum number of topics per sentiment group.")
    parser.add_argument("--bertopic", action="store_true", help="Enable BERTopic/UMAP candidate evaluation.")
    return parser.parse_args()


def safe_report_slug(title: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", strip_accents(title.lower())).strip("_") or "reporte"


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def get_stemmer(language: str) -> SnowballStemmer | None:
    snowball_language = LANGUAGE_MAP.get(language)
    if not snowball_language:
        return None
    try:
        return SnowballStemmer(snowball_language)
    except ValueError:
        return None


def preprocess_text(text: str, language: str, stemmer: SnowballStemmer | None) -> tuple[str, list[str]]:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = strip_accents(text)
    tokens = re.findall(r"[a-záéíóúüñç]+", text, flags=re.IGNORECASE)
    stops = {strip_accents(w.lower()) for w in STOPWORDS.get(language, STOPWORDS["en"])}
    clean_tokens = [token for token in tokens if len(token) > 2 and token not in stops]
    if stemmer:
        clean_tokens = [stemmer.stem(token) for token in clean_tokens]
    return " ".join(clean_tokens), clean_tokens


def sentiment_score(tokens: Iterable[str], language: str, stemmer: SnowballStemmer | None) -> float:
    lexicon = SENTIMENT_LEXICONS.get(language, SENTIMENT_LEXICONS["en"])
    token_list = list(tokens)
    if not token_list:
        return 0.0

    def normalize_lexicon(words: Iterable[str]) -> set[str]:
        clean_words = [strip_accents(word.lower()) for word in words]
        if stemmer:
            return {stemmer.stem(word) for word in clean_words}
        return set(clean_words)

    positives = normalize_lexicon(lexicon["positive"])
    negatives = normalize_lexicon(lexicon["negative"])
    pos_count = sum(token in positives for token in token_list)
    neg_count = sum(token in negatives for token in token_list)
    return (pos_count - neg_count) / max(1, pos_count + neg_count)


def detect_outliers(texts: pd.Series) -> np.ndarray:
    if len(texts) < 6 or texts.str.strip().eq("").all():
        return np.zeros(len(texts), dtype=bool)

    vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
    matrix = vectorizer.fit_transform(texts)
    contamination = min(0.15, max(0.03, 2 / len(texts)))
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(matrix)
    return labels == -1


def ngram_counts(texts: Iterable[str], n: int, top_k: int = 15) -> list[tuple[str, int]]:
    vectorizer = CountVectorizer(ngram_range=(n, n), min_df=1)
    docs = [text for text in texts if text.strip()]
    if not docs:
        return []
    matrix = vectorizer.fit_transform(docs)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = counts.argsort()[::-1][:top_k]
    return [(terms[i], int(counts[i])) for i in order]


def word_cloud_terms(frame: pd.DataFrame, top_k: int = 55) -> list[dict[str, object]]:
    tokens: list[str] = []
    for row_tokens in frame.get("tokens", []):
        if isinstance(row_tokens, list):
            tokens.extend(row_tokens)
    counts = Counter(token for token in tokens if len(token) > 2)
    if not counts:
        return []
    max_count = counts.most_common(1)[0][1]
    return [
        {
            "term": term,
            "count": int(count),
            "weight": round(count / max_count, 4),
        }
        for term, count in counts.most_common(top_k)
    ]


def build_word_clouds(frame: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    return {
        "global": word_cloud_terms(frame),
        "positivo": word_cloud_terms(frame[frame["sentiment"] == "positivo"]),
        "negativo": word_cloud_terms(frame[frame["sentiment"] == "negativo"]),
        "outlier": word_cloud_terms(frame[frame["sentiment"] == "outlier"]),
    }


def topic_silhouette(matrix, labels: np.ndarray) -> float:
    mask = labels >= 0
    if mask.sum() != len(labels):
        matrix = matrix[mask]
        labels = labels[mask]
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= matrix.shape[0]:
        return 0.0
    try:
        return float(silhouette_score(matrix, labels, metric="cosine"))
    except ValueError:
        return 0.0


def topic_balance(labels: np.ndarray) -> float:
    labels = labels[labels >= 0]
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels.astype(int))
    counts = counts[counts > 0]
    if len(counts) < 2:
        return 0.0
    return float(counts.min() / counts.max())


def topic_keywords(matrix, terms: np.ndarray, labels: np.ndarray, topic_idx: int, top_k: int = 10) -> list[str]:
    rows = matrix[labels == topic_idx]
    if rows.shape[0] == 0:
        return []
    weights = np.asarray(rows.mean(axis=0)).ravel()
    return [terms[i] for i in weights.argsort()[::-1][:top_k] if weights[i] > 0]


def bertopic_candidate(
    docs: list[str],
    matrix,
    terms: np.ndarray,
    dense_projection: np.ndarray,
    topic_count: int,
) -> dict[str, object] | None:
    if not BERTOPIC_AVAILABLE or BERTopic is None or UMAP is None:
        return None
    try:
        topic_count = max(2, min(topic_count, len(docs) - 1))
        umap_components = max(2, min(topic_count, dense_projection.shape[1], len(docs) - 2))
        umap_model = UMAP(
            n_neighbors=min(10, max(2, len(docs) - 1)),
            n_components=umap_components,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        cluster_model = KMeans(n_clusters=topic_count, n_init=25, random_state=42)
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), min_df=1)
        model = BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=False,
            verbose=False,
        )
        labels, _ = model.fit_transform(docs, embeddings=dense_projection)
        labels = np.asarray(labels, dtype=int)
        unique_labels = sorted(label for label in np.unique(labels) if label >= 0)
        if len(unique_labels) < 2:
            return None

        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        mapped_labels = np.asarray([label_map.get(label, 0) for label in labels], dtype=int)
        centroids = np.vstack([
            dense_projection[mapped_labels == idx].mean(axis=0)
            for idx in range(len(unique_labels))
        ])
        weights = cosine_similarity(normalize(dense_projection), normalize(centroids))
        keywords_by_topic: dict[int, list[str]] = {}
        for original_label, mapped_label in label_map.items():
            topic_terms = model.get_topic(original_label) or []
            keywords = [term for term, _ in topic_terms[:10] if term]
            keywords_by_topic[mapped_label] = keywords or topic_keywords(matrix, terms, mapped_labels, mapped_label)

        return {
            "name": "BERTopic",
            "labels": mapped_labels,
            "weights": weights,
            "silhouette": topic_silhouette(dense_projection, mapped_labels),
            "balance": topic_balance(mapped_labels),
            "topic_count": len(unique_labels),
            "keywords_by_topic": keywords_by_topic,
        }
    except Exception:
        return None


def topic_model(
    frame: pd.DataFrame,
    sentiment: str,
    max_topics: int,
    min_topic_docs: int,
    enable_bertopic: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    result = frame.copy()
    result["topic_id"] = "frecuencia"
    result["topic_label"] = f"{sentiment}_frecuencia"
    docs = result["clean_text"].fillna("").tolist()
    non_empty_docs = [doc for doc in docs if doc.strip()]

    if len(non_empty_docs) < min_topic_docs:
        words = Counter(" ".join(non_empty_docs).split()).most_common(10)
        result["representative_score"] = result["clean_text"].str.len()
        result["topic_model"] = "frequency"
        result["topic_quality"] = 0.0
        result["topic_balance"] = 0.0
        representative = result.sort_values("representative_score", ascending=False).head(1)
        summary = [{
            "sentiment": sentiment,
            "method": "frequency",
            "topic": f"{sentiment}_frecuencia",
            "keywords": ", ".join(word for word, _ in words),
            "quality": 0.0,
            "balance": 0.0,
            "documents": int(len(result)),
            "representative_comment": representative["comment"].iloc[0] if not representative.empty else "",
        }]
        diagnostics = [{"sentiment": sentiment, "method": "frequency", "quality": 0.0, "balance": 0.0, "selected": True}]
        return result, summary, diagnostics

    topic_count = max(2, min(max_topics, len(non_empty_docs) // 3))
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, max_features=2000)
    matrix = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()
    if matrix.shape[1] < 2:
        words = Counter(" ".join(non_empty_docs).split()).most_common(10)
        result["representative_score"] = result["clean_text"].str.len()
        result["topic_model"] = "frequency"
        result["topic_quality"] = 0.0
        result["topic_balance"] = 0.0
        representative = result.sort_values("representative_score", ascending=False).head(1)
        summary = [{
            "sentiment": sentiment,
            "method": "frequency",
            "topic": f"{sentiment}_frecuencia",
            "keywords": ", ".join(word for word, _ in words),
            "quality": 0.0,
            "balance": 0.0,
            "documents": int(len(result)),
            "representative_comment": representative["comment"].iloc[0] if not representative.empty else "",
        }]
        diagnostics = [{"sentiment": sentiment, "method": "frequency", "quality": 0.0, "balance": 0.0, "selected": True}]
        return result, summary, diagnostics
    dense_projection = TruncatedSVD(
        n_components=min(max(2, topic_count * 2), matrix.shape[1]),
        random_state=42,
    ).fit_transform(matrix)

    nmf_model = NMF(n_components=topic_count, random_state=42, init="nndsvda", max_iter=300)
    nmf_weights = nmf_model.fit_transform(matrix)
    nmf_labels = nmf_weights.argmax(axis=1)
    nmf_score = topic_silhouette(dense_projection, nmf_labels)

    kmeans_model = KMeans(n_clusters=topic_count, n_init=10, random_state=42)
    kmeans_labels = kmeans_model.fit_predict(normalize(dense_projection))
    kmeans_score = topic_silhouette(dense_projection, kmeans_labels)

    candidates = [
        {
            "name": "NMF",
            "labels": nmf_labels,
            "weights": nmf_weights,
            "silhouette": nmf_score,
            "balance": topic_balance(nmf_labels),
            "topic_count": topic_count,
            "keywords_by_topic": {},
        },
        {
            "name": "LSA-KMeans",
            "labels": kmeans_labels,
            "weights": cosine_similarity(normalize(dense_projection), kmeans_model.cluster_centers_),
            "silhouette": kmeans_score,
            "balance": topic_balance(kmeans_labels),
            "topic_count": topic_count,
            "keywords_by_topic": {},
        },
    ]
    bertopic_result = bertopic_candidate(docs, matrix, terms, dense_projection, topic_count) if enable_bertopic else None
    if bertopic_result:
        candidates.append(bertopic_result)
    selected = max(candidates, key=lambda item: (item["silhouette"] * 0.78) + (item["balance"] * 0.22))

    topic_ids = selected["labels"].astype(int)
    weights = selected["weights"]
    selected_topic_count = int(selected.get("topic_count", topic_count))
    result["topic_id"] = topic_ids
    result["topic_label"] = [f"{sentiment}_topico_{idx + 1}" for idx in topic_ids]
    result["topic_model"] = selected["name"]
    result["topic_quality"] = round(float(selected["silhouette"]), 4)
    result["topic_balance"] = round(float(selected["balance"]), 4)
    result["representative_score"] = weights.max(axis=1)

    summaries = []
    for topic_idx in range(selected_topic_count):
        if selected["name"] == "NMF":
            component = nmf_model.components_[topic_idx]
            top_terms = [terms[i] for i in component.argsort()[::-1][:10]]
        elif selected["name"] == "BERTopic":
            top_terms = selected.get("keywords_by_topic", {}).get(topic_idx, [])
        else:
            top_terms = topic_keywords(matrix, terms, topic_ids, topic_idx)
        topic_rows = result[result["topic_id"] == topic_idx].sort_values(
            "representative_score", ascending=False
        )
        representative = topic_rows["comment"].iloc[0] if not topic_rows.empty else ""
        summaries.append({
            "sentiment": sentiment,
            "method": selected["name"],
            "topic": f"{sentiment}_topico_{topic_idx + 1}",
            "keywords": ", ".join(top_terms),
            "quality": round(float(selected["silhouette"]), 4),
            "balance": round(float(selected["balance"]), 4),
            "documents": int((topic_ids == topic_idx).sum()),
            "representative_comment": representative,
        })
    diagnostics = [
        {
            "sentiment": sentiment,
            "method": candidate["name"],
            "quality": round(float(candidate["silhouette"]), 4),
            "balance": round(float(candidate["balance"]), 4),
            "topics": int(candidate.get("topic_count", topic_count)),
            "selected": candidate["name"] == selected["name"],
        }
        for candidate in candidates
    ]
    if not bertopic_result:
        diagnostics.append({
            "sentiment": sentiment,
            "method": "BERTopic",
            "quality": 0.0,
            "balance": 0.0,
            "topics": 0,
            "selected": False,
            "status": "disabled" if not enable_bertopic else "not_available",
        })
    return result, summaries, diagnostics


def add_projection(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    docs = result["clean_text"].fillna("").tolist()
    if len(result) < 2 or all(not doc.strip() for doc in docs):
        result["x"] = np.arange(len(result))
        result["y"] = 0
        return result

    matrix = TfidfVectorizer(max_features=2500, min_df=1).fit_transform(docs)
    if matrix.shape[1] < 2:
        result["x"] = np.arange(len(result))
        result["y"] = 0
        return result
    coords = TruncatedSVD(n_components=2, random_state=42).fit_transform(matrix)
    result["x"] = coords[:, 0]
    result["y"] = coords[:, 1]
    return result


def analyze_price_concept(frame: pd.DataFrame, language: str) -> pd.DataFrame:
    result = frame.copy()
    concept = PRICE_TERMS.get(language, PRICE_TERMS["en"])
    docs = result["clean_text"].fillna("").tolist() + [concept]
    vectorizer = TfidfVectorizer(max_features=2500, min_df=1)
    matrix = vectorizer.fit_transform(docs)
    doc_matrix = matrix[:-1]
    concept_vector = matrix[-1]
    result["price_similarity"] = cosine_similarity(doc_matrix, concept_vector).ravel()

    if doc_matrix.shape[1] >= 2 and len(result) >= 2:
        normalized = normalize(doc_matrix)
        coords = TruncatedSVD(n_components=2, random_state=42).fit_transform(normalized)
        result["price_x"] = coords[:, 0]
        result["price_y"] = coords[:, 1]
    else:
        result["price_x"] = np.arange(len(result))
        result["price_y"] = result["price_similarity"]
    return result


def build_scatter(frame: pd.DataFrame, title: str, palette: str, output_path: Path) -> None:
    colors = SUPPORTED_PALETTES[palette]
    fig = px.scatter(
        frame,
        x="x",
        y="y",
        color="topic_label",
        symbol="sentiment",
        color_discrete_sequence=colors,
        hover_data={
            "comment": True,
            "sentiment": True,
            "sentiment_score": ":.2f",
            "topic_label": True,
            "x": False,
            "y": False,
        },
        title=f"{title}: comentarios por topico y sentimiento",
    )
    fig.update_traces(marker={"size": 10, "opacity": 0.78, "line": {"width": 0.5, "color": "#222"}})
    fig.update_layout(template="plotly_white", legend_title_text="Topico")
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)


def build_price_scatter(frame: pd.DataFrame, title: str, palette: str, output_path: Path) -> None:
    colors = SUPPORTED_PALETTES[palette]
    fig = px.scatter(
        frame,
        x="price_x",
        y="price_y",
        color="price_similarity",
        symbol="sentiment",
        color_continuous_scale=colors,
        hover_data={
            "comment": True,
            "sentiment": True,
            "price_similarity": ":.3f",
            "price_x": False,
            "price_y": False,
        },
        title=f'{title}: similitud con "precio / valor / costo"',
    )
    fig.update_traces(marker={"size": 10, "opacity": 0.82, "line": {"width": 0.5, "color": "#222"}})
    fig.update_layout(template="plotly_white")
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)


def write_report(
    title: str,
    frame: pd.DataFrame,
    topic_summaries: list[dict[str, object]],
    topic_diagnostics: list[dict[str, object]],
    outlier_ngrams: dict[str, list[tuple[str, int]]],
    word_clouds: dict[str, list[dict[str, object]]],
    output_path: Path,
    scatter_name: str,
    price_name: str,
) -> None:
    top_price = frame.sort_values("price_similarity", ascending=False).head(5)
    summary_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['sentiment']))}</td>"
        f"<td>{html.escape(str(row['method']))}</td>"
        f"<td>{html.escape(str(row['topic']))}</td>"
        f"<td>{html.escape(str(row['keywords']))}</td>"
        f"<td>{html.escape(str(row.get('quality', 0)))}</td>"
        f"<td>{html.escape(str(row.get('documents', 0)))}</td>"
        f"<td>{html.escape(str(row['representative_comment']))}</td>"
        "</tr>"
        for row in topic_summaries
    )
    price_rows = "\n".join(
        "<tr>"
        f"<td>{idx + 1}</td>"
        f"<td>{row.price_similarity:.3f}</td>"
        f"<td>{html.escape(str(row.comment))}</td>"
        "</tr>"
        for idx, row in enumerate(top_price.itertuples())
    )
    diagnostic_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('sentiment', '')))}</td>"
        f"<td>{html.escape(str(row.get('method', '')))}</td>"
        f"<td>{html.escape(str(row.get('quality', 0)))}</td>"
        f"<td>{html.escape(str(row.get('balance', 0)))}</td>"
        f"<td>{html.escape(str(row.get('topics', 0)))}</td>"
        f"<td>{'Si' if row.get('selected') else 'No'}</td>"
        "</tr>"
        for row in topic_diagnostics
    )
    ngram_sections = "\n".join(
        f"<h3>{label}</h3><ol>"
        + "".join(f"<li>{html.escape(term)} ({count})</li>" for term, count in values)
        + "</ol>"
        for label, values in outlier_ngrams.items()
    )
    word_cloud_sections = "\n".join(
        f"<section class=\"word-cloud-group\"><h3>{html.escape(label.title())}</h3><div class=\"word-cloud\">"
        + "".join(
            f"<span style=\"--weight:{float(item['weight'])};\">"
            f"{html.escape(str(item['term']))}<small>{int(item['count'])}</small></span>"
            for item in values
        )
        + "</div></section>"
        for label, values in word_clouds.items()
        if values
    )
    html_doc = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px; vertical-align: top; }}
    th {{ background: #f6f8fa; text-align: left; }}
    iframe {{ width: 100%; height: 720px; border: 1px solid #d0d7de; }}
    .metric {{ display: inline-block; margin-right: 22px; font-weight: 700; }}
    .word-cloud-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin: 16px 0 28px; }}
    .word-cloud-group {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 14px; background: #fbfdff; }}
    .word-cloud {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; line-height: 1.1; }}
    .word-cloud span {{ color: #0f766e; font-weight: 800; font-size: calc(13px + (var(--weight) * 24px)); }}
    .word-cloud small {{ margin-left: 3px; color: #64748b; font-size: 10px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="metric">Comentarios analizados: {len(frame)}</p>
  <p class="metric">Outliers: {int(frame["is_outlier"].sum())}</p>
  <p class="metric">Normales: {int((~frame["is_outlier"]).sum())}</p>

  <h2>Topicos y comentarios representativos</h2>
  <table>
    <thead><tr><th>Sentimiento</th><th>Metodo</th><th>Topico</th><th>Palabras clave</th><th>Calidad</th><th>Docs</th><th>Comentario representativo</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>

  <h2>Diagnostico de modelos de topicos</h2>
  <table>
    <thead><tr><th>Sentimiento</th><th>Modelo</th><th>Silueta</th><th>Balance</th><th>Topicos</th><th>Seleccionado</th></tr></thead>
    <tbody>{diagnostic_rows}</tbody>
  </table>

  <h2>N-gramas en comentarios atipicos</h2>
  {ngram_sections}

  <h2>Nube de palabras</h2>
  <div class="word-cloud-grid">{word_cloud_sections}</div>

  <h2>Visualizacion interactiva por topico</h2>
  <iframe src="{html.escape(scatter_name)}"></iframe>

  <h2>Analisis precio / valor / costo</h2>
  <table>
    <thead><tr><th>#</th><th>Similitud</th><th>Comentario</th></tr></thead>
    <tbody>{price_rows}</tbody>
  </table>
  <iframe src="{html.escape(price_name)}"></iframe>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def analyze_csv(
    csv_path: Path,
    text_column: str,
    language: str,
    max_topics: int = 4,
    min_topic_docs: int = 8,
    enable_bertopic: bool = False,
) -> AnalysisResult:
    language = language.lower().strip()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available columns: {', '.join(df.columns)}")

    stemmer = get_stemmer(language)
    comments = df[text_column].fillna("").astype(str)
    processed = comments.apply(lambda value: preprocess_text(value, language, stemmer))

    analysis = pd.DataFrame({
        "comment": comments,
        "clean_text": processed.apply(lambda item: item[0]),
        "tokens": processed.apply(lambda item: item[1]),
    })
    analysis = analysis[analysis["comment"].str.strip().ne("")].reset_index(drop=True)
    analysis["is_outlier"] = detect_outliers(analysis["clean_text"])

    outlier_docs = analysis.loc[analysis["is_outlier"], "clean_text"].tolist()
    outlier_ngrams = {
        "Unigramas": ngram_counts(outlier_docs, 1),
        "Bigramas": ngram_counts(outlier_docs, 2),
        "Trigramas": ngram_counts(outlier_docs, 3),
    }

    normal = analysis[~analysis["is_outlier"]].copy()
    normal["sentiment_score"] = normal["tokens"].apply(lambda tokens: sentiment_score(tokens, language, stemmer))
    normal["sentiment"] = np.where(normal["sentiment_score"] >= 0, "positivo", "negativo")

    modeled_parts = []
    topic_summaries: list[dict[str, object]] = []
    topic_diagnostics: list[dict[str, object]] = []
    for sentiment in ["positivo", "negativo"]:
        part = normal[normal["sentiment"] == sentiment].copy()
        if part.empty:
            continue
        modeled, summaries, diagnostics = topic_model(part, sentiment, max_topics, min_topic_docs, enable_bertopic)
        modeled_parts.append(modeled)
        topic_summaries.extend(summaries)
        topic_diagnostics.extend(diagnostics)

    modeled_normal = pd.concat(modeled_parts, ignore_index=True) if modeled_parts else normal
    outliers = analysis[analysis["is_outlier"]].copy()
    outliers["sentiment_score"] = 0.0
    outliers["sentiment"] = "outlier"
    outliers["topic_id"] = "outlier"
    outliers["topic_label"] = "outlier"
    outliers["topic_model"] = "outlier"
    outliers["topic_quality"] = 0.0
    outliers["topic_balance"] = 0.0
    outliers["representative_score"] = 0.0

    final = pd.concat([modeled_normal, outliers], ignore_index=True)
    final = add_projection(final)
    final = analyze_price_concept(final, language)
    word_clouds = build_word_clouds(final)
    return AnalysisResult(final, topic_summaries, topic_diagnostics, outlier_ngrams, word_clouds)


def write_analysis_outputs(
    result: AnalysisResult,
    report_title: str,
    palette: str,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_title = safe_report_slug(report_title)

    csv_output = output_dir / f"{safe_title}_analysis.csv"
    scatter_output = output_dir / f"{safe_title}_topics.html"
    price_output = output_dir / f"{safe_title}_price.html"
    report_output = output_dir / f"{safe_title}_report.html"

    export_frame = result.frame.drop(columns=["tokens"], errors="ignore")
    export_frame.to_csv(csv_output, index=False)
    build_scatter(result.frame, report_title, palette, scatter_output)
    build_price_scatter(result.frame, report_title, palette, price_output)
    write_report(
        report_title,
        result.frame,
        result.topic_summaries,
        result.topic_diagnostics,
        result.outlier_ngrams,
        result.word_clouds,
        report_output,
        scatter_output.name,
        price_output.name,
    )
    return {
        "analysis_csv": csv_output,
        "topics_html": scatter_output,
        "price_html": price_output,
        "report_html": report_output,
    }


def main() -> int:
    args = parse_args()
    palette = args.palette.lower().strip()
    csv_path = Path(args.csv_path)

    if palette not in SUPPORTED_PALETTES:
        print(f"Palette '{args.palette}' is not supported. Use one of: {', '.join(SUPPORTED_PALETTES)}", file=sys.stderr)
        return 2

    try:
        result = analyze_csv(
            csv_path,
            args.text_column,
            args.language,
            args.topics,
            args.min_topic_docs,
            args.bertopic,
        )
        outputs = write_analysis_outputs(result, args.report_title, palette, Path(args.output_dir))
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Analysis completed: {outputs['report_html']}")
    print(f"Detailed CSV: {outputs['analysis_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
