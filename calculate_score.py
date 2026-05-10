from pathlib import Path
import pandas as pd
from datasets import load_from_disk
import spacy
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


DATASET_PATH = Path("./cnn_dailymail")
dataset = load_from_disk(DATASET_PATH)

train_df = dataset["train"].select(range(1000)).to_pandas()
test_df = dataset["test"].to_pandas()
val_df = dataset["validation"].to_pandas()

nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer("all-MiniLM-L6-v2")

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def label_article_semantic_top_k(article, highlights, k=3):
    article_sentences = split_sentences(article)
    highlight_sentences = split_sentences(highlights)

    if len(article_sentences) == 0:
        return []
    
    if len(highlight_sentences) == 0:
        return [
            {
                "sentence_id": i,
                "sentence": sentence,
                "score": 0.0,
                "label": 0
            }
            for i, sentence in enumerate(article_sentences)
        ]
    article_embeddings = model.encode(article_sentences, convert_to_tensor=True)
    highlight_embeddings = model.encode(highlight_sentences, convert_to_tensor=True)
    similarity_matrix = util.cos_sim(article_embeddings, highlight_embeddings)

    sentence_scores = []
    for i, article_sentence in enumerate(article_sentences):
        best_score = similarity_matrix[i].max().item()
        sentence_scores.append({
            "sentence_id": i,
            "sentence": article_sentence,
            "score": best_score,
            "label": 0
        }) 

    top_k = min(k, len(sentence_scores))
    top_sentences = sorted(
        sentence_scores,
        key=lambda x:x["score"],
        reverse=True
    )[:top_k]
    top_ids = {item["sentence_id"] for item in top_sentences}
    
    for item in sentence_scores:
        if item["sentence_id"] in top_ids:
            item["label"] = 1
        
    return sentence_scores

all_rows = []

for article_id, row in tqdm(train_df.iterrows(), total=len(train_df)):
    num_highlights = len(split_sentences(row["highlights"]))

    labeled_sentences = label_article_semantic_top_k(
        article=row["article"],
        highlights=row["highlights"],
        k=num_highlights
    )

    for item in labeled_sentences:
        all_rows.append({
            "article_id": article_id,
            "sentence_id": item["sentence_id"],
            "sentence": item["sentence"],
            "score": item["score"],
            "label": item["label"]
        })

sentence_df = pd.DataFrame(all_rows)
print(sentence_df.head())
print(sentence_df["label"].value_counts())

sentence_df.to_csv("semantic_sentence_labels.csv", index=False)