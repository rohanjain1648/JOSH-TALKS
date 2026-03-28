"""
01_preprocess.py
────────────────────────────────────────────────────────────────────────────────
Download audio + transcriptions from GCS (upload_goai bucket), segment each
long-form recording into ≤30 s chunks aligned to word-level timestamps, and
save as a Hugging Face Dataset ready for fine-tuning Whisper-small on Hindi.

Usage:
    python 01_preprocess.py \
        --manifest FT_Data_-_data.csv \
        --output_dir ./hindi_asr_dataset \
        --max_chunk_sec 29.0 \
        --min_chunk_sec 0.5

Requirements:
    pip install datasets torchaudio soundfile pandas requests indicnlp
"""

import os, re, json, logging, argparse, hashlib
from pathlib import Path
import pandas as pd
import requests
import torchaudio
import torch
from datasets import Dataset, DatasetDict, Audio
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── URL helpers ─────────────────────────────────────────────────────────────

GCS_BASE = os.environ.get("GCS_BUCKET_BASE", "https://storage.googleapis.com/upload_goai")

def transform_url(old_url: str, suffix: str) -> str:
    """
    Transform legacy joshtalks-data-collection URL to upload_goai format.
    OLD: .../joshtalks-data-collection/hq_data/hi/{folder}/{rec_id}_audio.wav
    NEW: .../upload_goai/{folder}/{rec_id}_{suffix}
    """
    m = re.search(r"/(\d+)/(\d+)_", old_url)
    if not m:
        raise ValueError(f"Cannot parse folder/rec_id from URL: {old_url}")
    folder_id, rec_id = m.group(1), m.group(2)
    return f"{GCS_BASE}/{folder_id}/{rec_id}_{suffix}"


# ─── Download helpers ────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, timeout: int = 120) -> Path:
    """Download a file if not already cached."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
    return dest


# ─── Transcript cleaning ─────────────────────────────────────────────────────

def clean_transcript(text: str) -> str:
    """
    Clean a raw Hindi transcript string:
      - Strip whitespace
      - Remove SSML / markup tags
      - Collapse multiple spaces
      - Remove stray ASCII punctuation (keep Devanagari punct: ।?!)
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)          # strip markup
    text = re.sub(r"[^\u0900-\u097F\u0030-\u0039\u0966-\u096F ?!।\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_transcription(json_path: Path) -> list[dict]:
    """
    Parse a transcription JSON file into a list of word-level dicts:
      [{"word": str, "start": float (sec), "end": float (sec)}, ...]
    Handles both common JSON schemas (words list, segments list).
    """
    with open(json_path) as f:
        data = json.load(f)

    words = []
    # Schema 1: top-level "words" key
    if "words" in data:
        for w in data["words"]:
            words.append({"word": w.get("word", w.get("text", "")),
                           "start": float(w.get("start", w.get("start_time", 0))),
                           "end":   float(w.get("end",   w.get("end_time",   0)))})
    # Schema 2: "segments" with nested "words"
    elif "segments" in data:
        for seg in data["segments"]:
            for w in seg.get("words", []):
                words.append({"word": w.get("word", w.get("text", "")),
                               "start": float(w.get("start", 0)),
                               "end":   float(w.get("end",   0))})
    # Fallback: single segment with full text — create one chunk
    elif "transcript" in data or "text" in data:
        text = data.get("transcript", data.get("text", ""))
        dur  = data.get("duration", 30.0)
        words = [{"word": text, "start": 0.0, "end": float(dur)}]

    return words


# ─── Chunking ────────────────────────────────────────────────────────────────

def segment_words_into_chunks(
    words: list[dict], max_dur: float = 29.0, min_dur: float = 0.5
) -> list[dict]:
    """
    Greedily pack consecutive words into chunks of ≤ max_dur seconds.
    Returns: [{"text": str, "start": float, "end": float}, ...]
    """
    chunks = []
    cur_words, cur_start = [], None

    for w in words:
        word_text  = w["word"].strip()
        word_start = w["start"]
        word_end   = w["end"]
        if not word_text:
            continue
        if cur_start is None:
            cur_start = word_start

        # Would adding this word exceed the limit?
        if cur_words and (word_end - cur_start) > max_dur:
            # flush current chunk
            text = " ".join(x["word"] for x in cur_words)
            dur  = cur_words[-1]["end"] - cur_start
            if dur >= min_dur and text.strip():
                chunks.append({"text": clean_transcript(text),
                                "start": cur_start,
                                "end":   cur_words[-1]["end"]})
            cur_words = []
            cur_start = word_start

        cur_words.append(w)

    # flush last chunk
    if cur_words:
        text = " ".join(x["word"] for x in cur_words)
        dur  = cur_words[-1]["end"] - cur_start
        if dur >= min_dur and text.strip():
            chunks.append({"text": clean_transcript(text),
                            "start": cur_start,
                            "end":   cur_words[-1]["end"]})
    return chunks


# ─── Audio processing ────────────────────────────────────────────────────────

TARGET_SR = 16_000

def load_and_normalise_audio(wav_path: Path) -> tuple[torch.Tensor, int]:
    """Load WAV, convert to mono float32, resample to 16 kHz."""
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # stereo → mono
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)
    waveform = waveform.squeeze(0)  # [T]
    # amplitude normalise
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform, TARGET_SR


def slice_audio(waveform: torch.Tensor, start_sec: float, end_sec: float) -> torch.Tensor:
    """Slice waveform at 16 kHz."""
    s = int(start_sec * TARGET_SR)
    e = int(end_sec   * TARGET_SR)
    return waveform[s:e]


# ─── Main pipeline ───────────────────────────────────────────────────────────

def build_examples(
    df: pd.DataFrame,
    cache_dir: Path,
    max_chunk_sec: float,
    min_chunk_sec: float,
) -> list[dict]:
    """
    For each row in the manifest: download audio + transcription, segment,
    and return a flat list of {"audio": ndarray, "text": str, "user_id": int}.
    """
    examples = []
    total = len(df)

    for idx, row in df.iterrows():
        rec_id  = row["recording_id"]
        user_id = row["user_id"]
        log.info(f"[{idx+1}/{total}] Processing rec {rec_id} (user {user_id})")

        audio_url   = transform_url(row["rec_url_gcp"], "audio.wav")
        transc_url  = transform_url(row["rec_url_gcp"], "transcription.json")

        audio_path  = cache_dir / "audio"  / f"{rec_id}.wav"
        transc_path = cache_dir / "transc" / f"{rec_id}.json"

        try:
            download_file(audio_url,  audio_path)
            download_file(transc_url, transc_path)
        except Exception as e:
            log.warning(f"  Download failed for {rec_id}: {e} — skipping")
            continue

        # Load audio
        try:
            waveform, _ = load_and_normalise_audio(audio_path)
        except Exception as e:
            log.warning(f"  Audio load failed for {rec_id}: {e} — skipping")
            continue

        # Parse transcription & chunk
        try:
            words  = load_transcription(transc_path)
            chunks = segment_words_into_chunks(words, max_chunk_sec, min_chunk_sec)
        except Exception as e:
            log.warning(f"  Transcription parse failed for {rec_id}: {e} — skipping")
            continue

        for chunk in chunks:
            audio_slice = slice_audio(waveform, chunk["start"], chunk["end"])
            if audio_slice.shape[0] < int(min_chunk_sec * TARGET_SR):
                continue
            examples.append({
                "audio":      audio_slice.numpy(),
                "sampling_rate": TARGET_SR,
                "text":       chunk["text"],
                "user_id":    int(user_id),
                "recording_id": int(rec_id),
                "chunk_start": chunk["start"],
                "chunk_end":   chunk["end"],
            })

        log.info(f"  → {len(chunks)} chunks from {row['duration']}s recording")

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",      default="FT_Data_-_data.csv")
    parser.add_argument("--output_dir",    default="./hindi_asr_dataset")
    parser.add_argument("--cache_dir",     default="./gcs_cache")
    parser.add_argument("--max_chunk_sec", type=float, default=29.0)
    parser.add_argument("--min_chunk_sec", type=float, default=0.5)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    df        = pd.read_csv(args.manifest)
    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Manifest: {len(df)} recordings, {df['duration'].sum()/3600:.2f} hrs total")

    examples = build_examples(df, cache_dir, args.max_chunk_sec, args.min_chunk_sec)
    log.info(f"Total examples after segmentation: {len(examples)}")

    # Speaker-stratified 90/10 split
    user_ids = list({e["user_id"] for e in examples})
    train_users, val_users = train_test_split(
        user_ids, test_size=0.10, random_state=args.seed
    )
    train_user_set = set(train_users)

    train_examples = [e for e in examples if e["user_id"] in train_user_set]
    val_examples   = [e for e in examples if e["user_id"] not in train_user_set]

    log.info(f"Train: {len(train_examples)}  |  Val: {len(val_examples)}")

    def make_hf_dataset(exs):
        return Dataset.from_list([
            {"audio": {"array": e["audio"], "sampling_rate": e["sampling_rate"]},
             "sentence": e["text"],
             "user_id":  e["user_id"]}
            for e in exs
        ])

    dataset = DatasetDict({
        "train":      make_hf_dataset(train_examples),
        "validation": make_hf_dataset(val_examples),
    })
    dataset.save_to_disk(str(out_dir))
    log.info(f"Dataset saved to {out_dir}")


if __name__ == "__main__":
    main()
