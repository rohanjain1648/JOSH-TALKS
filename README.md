# Hindi ASR with Whisper

This repository contains a complete pipeline for data preprocessing, fine-tuning, evaluating, and error-handling for Hindi Automatic Speech Recognition (ASR) using OpenAI's `whisper-small` architecture. It includes tools for data ingestion from Google Cloud Storage, model fine-tuning via Hugging Face Transformers, baseline evaluation on the FLEURS dataset, error analysis, and a custom Hindi text normaliser.

## Features

- **Data Preprocessing Pipeline (`01_preprocess.py`)**: Downloads audio and transcription pairs from Google Cloud Storage, normalises audio (16kHz mono), segments long-form recordings to ≤ 30s chunks aligned to word-level timestamps, cleans transcripts, and compiles a Hugging Face Dataset ready for fine-tuning.
- **Model Fine-Tuning (`02_finetune.py`)**: Fine-tunes `openai/whisper-small` using the Hugging Face `Seq2SeqTrainer` optimized for speech tasks, with automatic evaluation using WER (Word Error Rate).
- **Baseline Evaluation (`03_eval_baseline.py`)**: Evaluates the zero-shot performance of the base `whisper-small` model on the `hi_in` split of the FLEURS test dataset.
- **Error Sampling (`05_error_sample.py`)**: Facilitates robust error analysis by applying stratified interval sampling across low, medium, and high WER tiers from evaluation outputs.
- **Hindi Text Normalisation (`06_hindi_normaliser.py`)**: A custom post-processing module addressing common Hindi orthography variations. It handles Devanagari to Arabic numeral canonicalization, ordinal text mappings, numeric word mappings, and schwa deletion variants to improve WER metric consistency during evaluation.

## Installation

Ensure you have Python 3.9+ installed and a working CUDA environment if training on GPU.

Install the necessary dependencies via pip:

```bash
pip install -r requirements.txt
```

### Main Dependencies
- `transformers`, `datasets`, `evaluate`, `accelerate` for model training and data handling.
- `torch`, `torchaudio` for tensor and audio operations.
- `jiwer` for Word Error Rate (WER) computation.
- `indic-nlp-library` for text analysis.

## Usage

### 1. Preprocess Dataset
Downloads the audio files and transcripts specified in a manifest CSV, processes them, and saves a built Hugging Face Dataset to disk.

```bash
python 01_preprocess.py \
    --manifest FT_Data_-_data.csv \
    --output_dir ./hindi_asr_dataset \
    --max_chunk_sec 29.0 \
    --min_chunk_sec 0.5
```

### 2. Fine-tune Whisper Model
Fine-tunes the `whisper-small` model on the generated local dataset.

```bash
python 02_finetune.py \
    --dataset_dir ./hindi_asr_dataset \
    --output_dir ./whisper-small-hi-finetuned \
    --epochs 5 \
    --batch_size 16
```

### 3. Evaluate Baseline Model
Evaluates the baseline `openai/whisper-small` model on the FLEURS Hindi test set to establish an initial benchmark.

```bash
python 03_eval_baseline.py --output_csv baseline_results.csv
```

### 4. Sample Errors for Analysis
Extracts a stratified sample of errors from the evaluation results (e.g., from `finetuned_results.csv`) for manual inspection.

```bash
python 05_error_sample.py \
    --results_csv finetuned_results.csv \
    --output_csv sample_errors.csv \
    --n_per_tier 10
```

### 5. Normalise Hindi Transcripts
Run the standalone script to calculate the impact of the Hindi text normaliser on WER metrics by comparing metrics before and after text cleaning.

```bash
python 06_hindi_normaliser.py \
    --input_csv sample_errors.csv \
    --output_csv sample_errors_normalised.csv
```

This normaliser can also be imported into other python evaluation scripts to normalize reference and hypothesis strings prior to calculating the WER.

```python
import jiwer
from 06_hindi_normaliser import normalise

wer_score = jiwer.wer(normalise(reference), normalise(hypothesis))
```

## Documentation

For a comprehensive explanation of methodologies, modeling decisions, text normalization implementations, and final results, please refer to the accompanying report file: `Hindi_ASR_Whisper_Report.docx`.
