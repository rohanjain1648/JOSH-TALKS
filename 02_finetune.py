"""
02_finetune.py
────────────────────────────────────────────────────────────────────────────────
Fine-tune openai/whisper-small on the preprocessed Hindi ASR dataset using
Hugging Face Seq2SeqTrainer.

Usage:
    python 02_finetune.py \
        --dataset_dir ./hindi_asr_dataset \
        --output_dir  ./whisper-small-hi-finetuned \
        --epochs 5

Requirements:
    pip install transformers datasets evaluate jiwer torchaudio wandb accelerate
"""

import os, argparse, logging
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
from datasets import load_from_disk, DatasetDict
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_ID  = "openai/whisper-small"
LANGUAGE  = "Hindi"
TASK      = "transcribe"

# ─── Data Collator ───────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate audio and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so it is ignored in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip BOS token if it was prepended
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ─── Preprocessing ───────────────────────────────────────────────────────────

def prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


# ─── Metrics ─────────────────────────────────────────────────────────────────

def build_compute_metrics(processor):
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",   default="./hindi_asr_dataset")
    parser.add_argument("--output_dir",    default="./whisper-small-hi-finetuned")
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--grad_accum",    type=int,   default=2)
    parser.add_argument("--lr",            type=float, default=1e-5)
    parser.add_argument("--warmup_steps",  type=int,   default=500)
    parser.add_argument("--fp16",          action="store_true", default=True)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    # Load processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)

    # Load dataset
    dataset = load_from_disk(args.dataset_dir)
    log.info(f"Dataset: {dataset}")

    # Feature extraction (map)
    def _prepare(batch):
        return prepare_dataset(batch, feature_extractor, tokenizer)

    dataset = dataset.map(
        _prepare,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
        desc="Extracting features",
    )

    # Model
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.generation_config.language = LANGUAGE.lower()
    model.generation_config.task     = TASK
    model.generation_config.forced_decoder_ids = None  # handled by tokenizer settings

    # Data collator
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        warmup_steps                = args.warmup_steps,
        fp16                        = args.fp16,
        evaluation_strategy         = "steps",
        eval_steps                  = 500,
        save_strategy               = "steps",
        save_steps                  = 500,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        predict_with_generate       = True,
        generation_max_length       = 448,
        logging_steps               = 25,
        report_to                   = ["tensorboard"],
        push_to_hub                 = False,
        seed                        = args.seed,
        dataloader_num_workers      = 4,
        max_grad_norm               = 1.0,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        args           = training_args,
        model          = model,
        train_dataset  = dataset["train"],
        eval_dataset   = dataset["validation"],
        data_collator  = collator,
        compute_metrics= build_compute_metrics(processor),
        tokenizer      = processor.feature_extractor,
    )

    log.info("Starting training …")
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    log.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
