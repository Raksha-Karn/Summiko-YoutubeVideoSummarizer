import spacy
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from deepmultilingualpunctuation import PunctuationModel

DATA_FOLDER = Path("data")
MODEL_PATH = DATA_FOLDER / "sentence_importance_model.joblib"
THRESHOLD_PATH = DATA_FOLDER / "sentence_importance_threshold.joblib"

nlp = spacy.load("en_core_web_sm")
model = PunctuationModel()
ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC"}

def extract_video_id(video_url: str) -> str:
    parsed = urlparse(video_url)

    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/")

    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            return parse_qs(parsed.query)["v"][0]

        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/")[1].split("/")[0]

    raise ValueError("Could not extract YouTube video ID from URL.")

def get_transcript(video_url: str) -> str:
    video_id = extract_video_id(video_url)
    ytt_api = YouTubeTranscriptApi()

    fetched_transcript = ytt_api.fetch(video_id)
    transcipt_text = ""
    if fetched_transcript:
        transcript_text = " ".join(snippet.text for snippet in fetched_transcript)
    else:
        raise ValueError("Missing transcript for the video")

    return transcript_text

def add_punctuation(transcript_text):
    result = model.restore_punctuation(transcript_text)
    return result

def compute_features(segments: list[str]) -> pd.DataFrame:
    pass


# compute_features(get_transcript("https://www.youtube.com/watch?v=CMrHM8a3hqw"))
# print(get_transcript("https://www.youtube.com/watch?v=CMrHM8a3hqw"))
print(add_punctuation(get_transcript("https://www.youtube.com/watch?v=CMrHM8a3hqw")))
