
<div align="center">

<img 
  src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=900&color=FF7EB6&center=true&vCenter=true&width=700&lines=Summiko+%F0%9F%8C%B8;NLP+Based+YouTube+Video+Summarizer;Extractive+NLP+Sentence+Ranking;Built+with+spaCy+%2B+scikit-learn" 
  alt="Typing SVG"
/>

<br />

<img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/NLP-spaCy-green?style=for-the-badge" />
<img src="https://img.shields.io/badge/ML-scikit--learn-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/Project-YouTube%20Summarizer-ff69b4?style=for-the-badge" />

<br />
<br />

<h1>🌸 Summiko</h1>

<p>

</p>

</div>

An extractive YouTube video summarizer built from scratch using hand-engineered NLP features and a trained machine learning classifier — no LLM APIs, no black boxes.


## Demo

```
$ python summarizer.py "https://www.youtube.com/watch?v=CMrHM8a3hqw"

Loading models...
Fetching transcript...
Restoring punctuation...
Splitting into sentences...
Found 49 sentences. Computing features...

════════════════════════════════════════════════════════════
SUMMARY
════════════════════════════════════════════════════════════

1. While star wars might be set in a galaxy far away, the
   reality of having machines talk and respond to us in a
   human-like manner is already a reality.

2. Natural language processing or NLP refers to the branch
   of artificial intelligence that gives machines the ability
   to read, understand and derive meaning from human languages.

3. NLP combines the field of linguistics and computer science
   to decipher language structure and guidelines and to make
   models which can comprehend, break down and separate
   significant details from text and speech.

...
════════════════════════════════════════════════════════════
```

---

## How It Works

### The Full Pipeline

```
YouTube URL
    │
    ▼
Transcript (youtube-transcript-api)
    │
    ▼
Punctuation Restoration (oliverguhr/fullstop-punctuation-multilang-large)
    │
    ▼
Sentence Splitting (spaCy en_core_web_sm)
    │
    ▼
Feature Engineering ──────────────────────────────────────┐
    │   • Relative position in transcript                  │
    │   • Sentence length (word count)                     │
    │   • TF-IDF score                                     │
    │   • Named entity count (PERSON, ORG, GPE, LOC)       │
    │   • Word overlap with first sentence                 │
    ▼                                                      │
Logistic Regression Classifier (trained on CNN/DailyMail) ◄┘
    │
    ▼
Threshold Filtering (threshold = 0.45)
    │
    ▼
Top sentences returned in original order
```



## Training

### Dataset

The model was trained on 1,000 articles from the [CNN/DailyMail dataset](https://huggingface.co/datasets/cnn_dailymail). Each article's sentences were labeled using semantic similarity — sentences most similar to the article's highlight bullet points (using `all-MiniLM-L6-v2` embeddings) were labeled as important.

**Label distribution:** 89% unimportant / 11% important — handled with `class_weight="balanced"`.

### Model Comparison (5-fold StratifiedGroupKFold cross-validation)

| Model | F1 | Precision | Recall |
|---|---|---|---|
| Logistic Regression | **0.3425** | 0.2286 | 0.6828 |
| Linear SVM | 0.3429 | 0.2292 | 0.6805 |
| Random Forest | 0.3385 | 0.5075 | 0.2539 |
| Gaussian Naive Bayes | 0.2639 | 0.3390 | 0.2164 |
| Dummy (majority class) | 0.0000 | 0.0000 | 0.0000 |

Logistic Regression was chosen for its `predict_proba` support, which enables threshold tuning.

### Threshold Tuning

Rather than using the default 0.5 threshold, thresholds from 0.10 to 0.90 were evaluated on the held-out test set. **0.45** was chosen to prioritize recall — for a summarizer, missing an important sentence is worse than including a slightly irrelevant one.

At threshold 0.45: **Precision 0.199 / Recall 0.722 / F1 0.311**



## Setup

**Requirements:** Python 3.12+

```bash
git clone https://github.com/Raksha-Karn/Summiko-YoutubeVideoSummarizer
cd yt-video-summarizer

uv venv && uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## What I Learned

This project was built as a structured introduction to applied ML. The key things it covers hands-on:

- Creating labeled training data from an unlabeled dataset using semantic similarity
- Feature engineering with interpretability checks (discriminability per feature)
- Handling class imbalance without resampling
- Group-aware train/test splitting to prevent data leakage
- Threshold tuning and why accuracy is the wrong metric for imbalanced problems
- Building a reusable inference pipeline that matches training-time feature computation exactly

---
