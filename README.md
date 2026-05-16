<div align="center">

<img
  src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=900&color=FF7EB6&center=true&vCenter=true&width=760&lines=Summiko;YouTube+Video+Summarizer;Extractive+NLP+Sentence+Ranking;Built+with+spaCy+%2B+scikit-learn"
  alt="Summiko typing banner"
/>

<br />

<img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python" alt="Python 3.12+" />
<img src="https://img.shields.io/badge/NLP-spaCy-09A3D5?style=for-the-badge" alt="spaCy" />
<img src="https://img.shields.io/badge/ML-scikit--learn-F7931E?style=for-the-badge" alt="scikit-learn" />
<img src="https://img.shields.io/badge/Transcripts-YouTube-red?style=for-the-badge&logo=youtube" alt="YouTube transcripts" />

<br />
<br />


Summiko is an extractive summarization project for YouTube videos. Instead of generating new text, it identifies the most important sentences from a video's transcript and returns them as a clean summary.

## Features

- **YouTube URL support** for standard `youtube.com/watch` links, `youtu.be` short links, and YouTube Shorts URLs.
- **Transcript extraction** using `youtube-transcript-api`.
- **Punctuation restoration** with `oliverguhr/fullstop-punctuation-multilang-large`.
- **Sentence segmentation** with spaCy.
- **Feature engineering** for sentence position, length, TF-IDF signal, named entities, and overlap with the opening sentence.
- **Machine-learning ranking** using a saved scikit-learn classifier.
- **Threshold-based selection** with minimum and maximum summary sentence limits.
- **Training workflow** based on the CNN/DailyMail dataset and semantic sentence labeling.

## How It Works

```text
YouTube URL
    |
    v
Extract video ID
    |
    v
Fetch transcript
    |
    v
Restore punctuation
    |
    v
Split into sentences
    |
    v
Compute sentence features
    |
    v
Predict sentence importance
    |
    v
Return selected sentences in original order
```

The final summary keeps selected sentences in their original order so the output still follows the flow of the video.

## Requirements

- Python `3.12+`
- `uv` for dependency management


## Installation

Clone the repository and install the Python environment:

```bash
git clone git@github.com:Raksha-Karn/Summiko-YoutubeVideoSummarizer.git
cd yt_video_summarizer
uv sync
```

Install the spaCy English model:

```bash
uv run python -m spacy download en_core_web_sm
```

## Training Pipeline

The project includes a complete training workflow for creating the sentence-importance model.

### 1. Download Dataset

```bash
uv run python download_dataset.py
```

This downloads the CNN/DailyMail dataset and saves it locally as:

```text
cnn_dailymail/
```

### 2. Create Semantic Labels

```bash
uv run python calculate_score.py
```

This script:

- Loads the local CNN/DailyMail dataset.
- Splits articles and highlights into sentences.
- Uses `all-MiniLM-L6-v2` sentence embeddings.
- Compares article sentences with reference highlights.
- Labels the top matching article sentences as important.
- Saves the output as `semantic_sentence_labels.csv`.

### 3. Build Features

```bash
uv run python feature_engineering.py
```

This script creates sentence-level features and saves:

```text
features.csv
```

The notebook currently reads from:

```text
data/features.csv
```

So after generating features, keep a copy of the file in `data/` before training.

### 4. Train and Save Model

Open the notebook:

```bash
uv run jupyter notebook training_data.ipynb
```

The notebook compares several models:

- Dummy majority baseline
- Logistic regression
- Random forest
- Gaussian naive Bayes
- Linear SVM

It also tunes the probability threshold used to decide whether a sentence is important.

The runtime summarizer expects the final files here:

```text
data/sentence_importance_model.joblib
data/sentence_importance_threshold.joblib
```

If the notebook saves the files in the project root, move or copy them into `data/` before running `summarizer.py`.

## Model Features

The classifier uses five sentence-level features:

| Feature | Description |
| --- | --- |
| `relative_position` | Sentence position divided by total number of sentences. |
| `sentence_length` | Number of non-punctuation, non-space tokens. |
| `tfidf_score` | Mean TF-IDF score for the sentence within the document. |
| `named_entity_count` | Count of `PERSON`, `ORG`, `GPE`, and `LOC` entities. |
| `first_sentence_overlap` | Word overlap with the first sentence of the document. |

These features are intentionally lightweight so inference stays simple and fast after transcript and punctuation models are loaded.

Example output format:

```text
============================================================
SUMMARY
============================================================

1. First selected sentence from the transcript.

2. Second selected sentence from the transcript.

3. Third selected sentence from the transcript.

============================================================
```
