import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

sentence_df = pd.read_csv("semantic_sentence_labels.csv")

nlp = spacy.load("en_core_web_sm")
ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC"}

df = sentence_df.copy()
df["total_sentences"] = df.groupby("article_id")["sentence_id"].transform("count")
df["relative_position"] = df["sentence_id"] / df["total_sentences"]
print(df[["article_id", "sentence_id", "total_sentences", "relative_position", "label"]].head(20))
print(df.groupby("label")["relative_position"].mean())

df["sentence_length"] = df["sentence"].apply(
    lambda x: len([
        token for token in nlp(str(x))
        if not token.is_punct and not token.is_space
    ])
)
print(df[["article_id", "sentence", "sentence_length", "label"]].head())
print(df.groupby("label")["sentence_length"].mean())

def add_tfidf_scores(group, article_id):
    group = group.copy()
    group["article_id"] = article_id

    sentences = group["sentence"].fillna("").tolist()

    if len(sentences) == 1:
        group["tfidf_score"] = 0.0
        return group
    
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = tfidf_matrix.mean(axis=1)
        group["tfidf_score"] = np.asarray(sentence_scores).ravel()
    except ValueError:
        group["tfidf_score"] = 0.0
    
    return group

print(df.columns)
tfidf_groups = []

for article_id, group in df.groupby("article_id", sort=False):
    group = add_tfidf_scores(group, article_id)
    tfidf_groups.append(group)

df = pd.concat(tfidf_groups, ignore_index=True)
print(df.columns)
print(df[["article_id", "sentence_id", "tfidf_score", "label"]].head(20))
print(df.groupby("label")["tfidf_score"].mean())

def count_named_entities(sentence):
    count = 0
    doc = nlp(str(sentence))
    for ent in doc.ents:
        if ent.label_ in ENTITY_TYPES:
            count += 1
    return count

df["named_entity_count"] = df["sentence"].apply(count_named_entities)
print(df[["article_id", "sentence", "named_entity_count", "label"]].head(20))
print(df.groupby("label")["named_entity_count"].mean())

def clean_words(text):
    text = str(text).lower()
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    return set(words)

first_sentences = (
    df[df["sentence_id"] == 0]
    .set_index("article_id")["sentence"]
    .to_dict()
)

def first_sentence_overlap(row):
    first_sentence = first_sentences.get(row["article_id"], "")

    first_words = clean_words(first_sentence)
    sentence_words = clean_words(row["sentence"])

    overlap = first_words.intersection(sentence_words)

    return len(overlap)


df["first_sentence_overlap"] = df.apply(first_sentence_overlap, axis=1)
print(df.groupby("label")["first_sentence_overlap"].mean())

feature_cols = [
    "article_id", "sentence_id", "sentence", "label",
    "relative_position", "sentence_length",
    "tfidf_score", "named_entity_count", "first_sentence_overlap"
]
df[feature_cols].to_csv("features.csv", index=False)
print("Saved features.csv")
print(df[feature_cols].head())