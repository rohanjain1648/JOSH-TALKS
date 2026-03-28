"""
03_eval_baseline.py
────────────────────────────────────────────────────────────────────────────────
Evaluate the pretrained Whisper-small baseline on FLEURS hi_in test set.

Usage:
    python 03_eval_baseline.py --output_csv baseline_results.csv
"""

import argparse, logging
import torch
import evaluate
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer as compute_wer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_ID = "openai/whisper-small"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


def run_eval(model_id_or_path: str, output_csv: str):
    log.info(f"Loading model: {model_id_or_path}")
    processor = WhisperProcessor.from_pretrained(model_id_or_path)
    model     = WhisperForConditionalGeneration.from_pretrained(model_id_or_path).to(DEVICE)
    model.eval()

    # Set Hindi language & transcription task
    forced_ids = processor.get_decoder_prompt_ids(language="hindi", task="transcribe")
    model.generation_config.forced_decoder_ids = forced_ids

    log.info("Loading FLEURS hi_in test set …")
    fleurs = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)

    metric = evaluate.load("wer")
    all_refs, all_hyps = [], []
    results = []

    for i, sample in enumerate(fleurs):
        audio      = sample["audio"]["array"]
        sr         = sample["audio"]["sampling_rate"]
        reference  = sample["transcription"]

        inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(DEVICE)
        with torch.no_grad():
            predicted_ids = model.generate(inputs, max_new_tokens=448)
        hypothesis = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        utt_wer = compute_wer(reference, hypothesis) * 100
        all_refs.append(reference)
        all_hyps.append(hypothesis)
        results.append({
            "idx":        i,
            "reference":  reference,
            "hypothesis": hypothesis,
            "wer":        round(utt_wer, 2),
        })

        if (i + 1) % 50 == 0:
            log.info(f"  Processed {i+1}/{len(fleurs)} samples …")

    overall_wer = metric.compute(predictions=all_hyps, references=all_refs) * 100
    log.info(f"Overall WER: {overall_wer:.2f}%")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    log.info(f"Per-utterance results saved to {output_csv}")
    return overall_wer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=MODEL_ID)
    parser.add_argument("--output_csv", default="baseline_results.csv")
    args = parser.parse_args()
    run_eval(args.model, args.output_csv)


if __name__ == "__main__":
    main()
