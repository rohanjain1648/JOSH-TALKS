"""
05_error_sample.py
────────────────────────────────────────────────────────────────────────────────
Stratified interval sampling of ≥25 error utterances from evaluation results.

Usage:
    python 05_error_sample.py \
        --results_csv finetuned_results.csv \
        --output_csv  sample_errors.csv \
        --n_per_tier  10

Strategy:
  1. Load per-utterance WER from CSV produced by 04_eval_finetuned.py
  2. Bin utterances: Low (1–15%), Medium (16–50%), High (>50%)
  3. Within each tier, select every Nth row (systematic/interval sampling)
     until n_per_tier examples are collected.
  4. Sort final sample by idx to remove selection bias.
  5. Save to CSV with: idx, reference, hypothesis, wer, tier, severity_rank
"""

import argparse
import pandas as pd


def assign_tier(wer: float) -> str:
    if wer <= 0:
        return "correct"
    elif wer <= 15:
        return "low"
    elif wer <= 50:
        return "medium"
    else:
        return "high"


def interval_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Every Nth row from a sorted DataFrame until n rows collected."""
    if len(df) == 0:
        return df
    step = max(1, len(df) // n)
    indices = list(range(0, len(df), step))[:n]
    return df.iloc[indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", default="finetuned_results.csv")
    parser.add_argument("--output_csv",  default="sample_errors.csv")
    parser.add_argument("--n_per_tier",  type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    df["tier"] = df["wer"].apply(assign_tier)

    # Drop correct utterances
    errors_df = df[df["tier"] != "correct"].copy()
    print(f"Total error utterances: {len(errors_df)}")
    print(df["tier"].value_counts())

    samples = []
    for tier in ["low", "medium", "high"]:
        tier_df = errors_df[errors_df["tier"] == tier].reset_index(drop=True)
        sampled = interval_sample(tier_df, args.n_per_tier)
        sampled["severity_rank"] = {"low": 1, "medium": 2, "high": 3}[tier]
        samples.append(sampled)
        print(f"  Tier '{tier}': {len(tier_df)} errors → sampled {len(sampled)}")

    final = pd.concat(samples).sort_values("idx").reset_index(drop=True)
    final["sample_id"] = range(1, len(final) + 1)
    cols = ["sample_id", "idx", "tier", "severity_rank", "wer", "reference", "hypothesis"]
    final[cols].to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(final)} sampled errors to {args.output_csv}")


if __name__ == "__main__":
    main()
