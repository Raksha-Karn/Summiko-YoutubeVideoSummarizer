import re
import sys
import spacy
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from sklearn.feature_extraction.text import TfidfVectorizer
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

DATA_FOLDER = Path("data")
MODEL_PATH = DATA_FOLDER / "sentence_importance_model.joblib"
THRESHOLD_PATH = DATA_FOLDER / "sentence_importance_threshold.joblib"

ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC"}
FEATURE_COLS = [
    "relative_position",
    "sentence_length",
    "tfidf_score",
    "named_entity_count",
    "first_sentence_overlap",
]

print("Loading models")
nlp = spacy.load("en_core_web_sm")
punctuation_model = pipeline(
    "ner",
    model="oliverguhr/fullstop-punctuation-multilang-large",
    aggregation_strategy="simple"  
)
sentence_model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)
print("Models loaded.\n")

def extract_video_id(video_url: str) -> str:
    parsed = urlparse(video_url)

    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/")

    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            params = parse_qs(parsed.query)
            if "v" not in params:
                raise ValueError("No video ID found in URL.")
            return params["v"][0]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/")[1].split("/")[0]

    raise ValueError(f"Could not extract YouTube video ID from: {video_url}")

def get_transcript(video_url: str) -> str:
    video_id = extract_video_id(video_url)
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id)

    if not fetched:
        raise ValueError("No transcript found for this video.")

    return " ".join(snippet.text.strip() for snippet in fetched if snippet.text.strip())

def add_punctuation(transcript_text: str) -> str:
    words = transcript_text.split()
    chunk_size = 200
    chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
    
    result_words = []
    for chunk in chunks:
        chunk_text = " ".join(chunk)
        predictions = punctuation_model(chunk_text)
        
        for pred in predictions:
            word = pred["word"].replace("▁", "").strip()
            label = pred["entity_group"]
            if not word:
                continue
            if label == ".":
                result_words.append(word + ".")
            elif label == ",":
                result_words.append(word + ",")
            elif label == "?":
                result_words.append(word + "?")
            else: 
                result_words.append(word)
    
    return " ".join(result_words)

def split_sentences(text: str, min_words: int = 5) -> list[str]:
    doc = nlp(text)
    return [
        sent.text.strip() for sent in doc.sents
        if len([t for t in sent if not t.is_punct and not t.is_space]) >= min_words
    ]

def clean_words(text: str) -> set:
    return set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))

def compute_features(sentences: list[str]) -> pd.DataFrame:
    n = len(sentences)
    df = pd.DataFrame({"sentence": sentences})
    df["sentence_id"] = range(n)
    df["relative_position"] = df["sentence_id"] / n
    df["sentence_length"] = df["sentence"].apply(
        lambda x: len([
            t for t in nlp(str(x))
            if not t.is_punct and not t.is_space
        ])
    )

    if n == 1:
        df["tfidf_score"] = 0.0
    else:
        try:
            vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(df["sentence"].fillna(""))
            df["tfidf_score"] = np.asarray(tfidf_matrix.mean(axis=1)).ravel()
        except ValueError:
            df["tfidf_score"] = 0.0

    def count_entities(text):
        doc = nlp(str(text))
        return sum(1 for ent in doc.ents if ent.label_ in ENTITY_TYPES)

    df["named_entity_count"] = df["sentence"].apply(count_entities)

    first_words = clean_words(sentences[0]) if sentences else set()
    df["first_sentence_overlap"] = df["sentence"].apply(
        lambda x: len(first_words.intersection(clean_words(x)))
    )

    return df[FEATURE_COLS]

def summarize(video_url: str, min_sentences: int = 3, max_sentences: int = 10) -> list[str]:
    print("Fetching transcript")
    raw_text = get_transcript(video_url)
    print(raw_text)
    print()

    print("Restoring punctuation")
    punctuated_text = add_punctuation(raw_text)
    print(punctuated_text)
    print()

    print("Splitting into sentences")
    sentences = split_sentences(punctuated_text)
    print(sentences)
    print()

    if len(sentences) < 3:
        raise ValueError(
            f"Too few sentences found ({len(sentences)}). "
            "The video may be too short or have a poor transcript."
        )

    print(f"Found {len(sentences)} sentences. Computing features now")

    features = compute_features(sentences)
    scores = sentence_model.predict_proba(features)[:, 1]
    selected_indices = [i for i, score in enumerate(scores) if score >= threshold]

    if len(selected_indices) < min_sentences:
        selected_indices = np.argsort(scores)[::-1][:min_sentences].tolist()
    elif len(selected_indices) > max_sentences:
        selected_scores = [(i, scores[i]) for i in selected_indices]
        selected_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [i for i, _ in selected_scores[:max_sentences]]

    selected_indices = sorted(selected_indices)
    return [sentences[i] for i in selected_indices]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <youtube_url>")
        print('Example: python summarizer.py "https://www.youtube.com/watch?v=CMrHM8a3hqw"')
        sys.exit(1)

    url = sys.argv[1]

    try:
        summary = summarize(url)
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for i, sentence in enumerate(summary, 1):
            print(f"\n{i}. {sentence}")
        print("\n" + "=" * 60)

    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)