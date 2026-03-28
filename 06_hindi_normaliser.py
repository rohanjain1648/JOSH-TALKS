"""
06_hindi_normaliser.py
────────────────────────────────────────────────────────────────────────────────
Post-processing normaliser implementing Fix F.3 (numeral normalisation) and a
lightweight component of Fix F.1 (schwa variant restoration).

Apply to BOTH reference and hypothesis strings before calling jiwer.wer().
This module is also the "implemented fix" for Section G of the report.

Usage as a module:
    from hindi_normaliser import normalise
    wer_score = jiwer.wer(normalise(ref), normalise(hyp))

Usage standalone (shows before/after on a test set CSV):
    python 06_hindi_normaliser.py \
        --input_csv  sample_errors.csv \
        --output_csv sample_errors_normalised.csv
"""

import re
import argparse
import pandas as pd
from jiwer import wer as compute_wer

# ─── 1. Devanagari ↔ Arabic digit mapping ────────────────────────────────────

DEVA_TO_ARABIC = str.maketrans("०१२३४५६७८९", "0123456789")
# We canonicalise to Arabic digits for uniformity.

# ─── 2. Ordinal normalisation ────────────────────────────────────────────────
# Map common mixed ordinal forms to their written-out Hindi equivalents.

ORDINAL_MAP: dict[str, str] = {
    "1st":  "पहली",   "1ला":  "पहला",   "1ली":  "पहली",
    "2nd":  "दूसरी",  "2रा":  "दूसरा",  "2री":  "दूसरी",
    "3rd":  "तीसरी",  "3रा":  "तीसरा",
    "4th":  "चौथी",   "4था":  "चौथा",
    "5th":  "पाँचवीं", "5वाँ": "पाँचवाँ", "5वीं": "पाँचवीं",
    "6th":  "छठी",
    "7th":  "सातवीं",
    "8th":  "आठवीं",
    "9th":  "नौवीं",
    "10th": "दसवीं",
}

# ─── 3. Numeral word forms ────────────────────────────────────────────────────
# Spoken Hindi uses words for many round numbers; map both directions to digit.

NUMBER_WORD_MAP: dict[str, str] = {
    "एक":     "1",   "दो":      "2",   "तीन":    "3",
    "चार":    "4",   "पाँच":    "5",   "छह":     "6",
    "सात":    "7",   "आठ":      "8",   "नौ":     "9",
    "दस":     "10",  "ग्यारह":  "11",  "बारह":   "12",
    "तेरह":   "13",  "चौदह":    "14",  "पंद्रह": "15",
    "सोलह":   "16",  "सत्रह":   "17",  "अठारह":  "18",
    "उन्नीस": "19",  "बीस":     "20",
    "तीस":    "30",  "चालीस":   "40",  "पचास":   "50",
    "साठ":    "60",  "सत्तर":   "70",  "अस्सी":  "80",
    "नब्बे":  "90",  "सौ":      "100", "हज़ार":  "1000",
}

# ─── 4. Schwa variant map ─────────────────────────────────────────────────────
# ~500 entries in production; this excerpt covers the most frequent forms.
# Key = schwa-deleted (as spoken), Value = canonical orthographic form.

SCHWA_MAP: dict[str, str] = {
    # Infinitives ending in -ना
    "करन":   "करना",   "बोलन":  "बोलना",  "देखन":  "देखना",
    "समझन":  "समझना",  "पढ़न":  "पढ़ना",   "लिखन":  "लिखना",
    "सुनन":  "सुनना",  "जान":   "जाना",   "आन":    "आना",
    "खान":   "खाना",   "पीन":   "पीना",   "सोन":   "सोना",
    "उठन":   "उठना",   "बैठन":  "बैठना",  "चलन":   "चलना",
    "दौड़न":  "दौड़ना", "सीखन":  "सीखना",  "सिखान": "सिखाना",
    "बतान":  "बताना",  "दिखान": "दिखाना", "मिलन":  "मिलना",
    "भेजन":  "भेजना",  "लेन":   "लेना",   "देन":   "देना",
    "पान":   "पाना",   "रखन":   "रखना",   "छोड़न": "छोड़ना",
    "मारन":  "मारना",  "बचान":  "बचाना",  "बनान":  "बनाना",
    "जीतन":  "जीतना",  "हारन":  "हारना",  "कहन":   "कहना",
    "सोचन":  "सोचना",  "समझान": "समझाना", "पहुँचन": "पहुँचना",
    "रुकन":  "रुकना",  "शुरू करन": "शुरू करना",
    # Gerundive / stem forms
    "होन":   "होना",   "रहन":   "रहना",   "जीन":   "जीना",
}

# ─── Normaliser function ──────────────────────────────────────────────────────

def normalise(text: str, *, convert_number_words: bool = False) -> str:
    """
    Normalise a Hindi ASR transcript string for WER evaluation.

    Steps applied in order:
      1. Strip leading/trailing whitespace.
      2. Collapse multiple spaces.
      3. Convert Devanagari digits → Arabic digits (canonical form).
      4. Apply ordinal normalisation.
      5. Optionally convert number words → Arabic digits.
      6. Apply schwa variant restoration.

    Parameters
    ----------
    text : str
        Raw transcript (reference or hypothesis).
    convert_number_words : bool
        If True, also convert spoken number words (एक→1, दो→2, …).
        Default False — only applies to digit-script and ordinal mismatches.

    Returns
    -------
    str
        Normalised transcript.
    """
    if not isinstance(text, str):
        return ""

    # 1–2: whitespace
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    # 3: Devanagari → Arabic digits
    text = text.translate(DEVA_TO_ARABIC)

    # 4: ordinal normalisation (match whole tokens)
    for variant, canon in ORDINAL_MAP.items():
        text = re.sub(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", canon, text)

    # 5: number word → digit (optional)
    if convert_number_words:
        for variant, canon in NUMBER_WORD_MAP.items():
            text = re.sub(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", canon, text)

    # 6: schwa restoration (whole-token match)
    for variant, canon in SCHWA_MAP.items():
        text = re.sub(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", canon, text)

    return text.strip()


# ─── Standalone evaluation ────────────────────────────────────────────────────

def evaluate_before_after(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    assert "reference" in df.columns and "hypothesis" in df.columns

    before_wers, after_wers = [], []

    for _, row in df.iterrows():
        ref = str(row["reference"])
        hyp = str(row["hypothesis"])
        w_before = compute_wer(ref, hyp) * 100
        w_after  = compute_wer(normalise(ref), normalise(hyp)) * 100
        before_wers.append(round(w_before, 2))
        after_wers.append(round(w_after,  2))

    df["wer_before_norm"] = before_wers
    df["wer_after_norm"]  = after_wers
    df["wer_delta"]       = [round(b - a, 2) for b, a in zip(before_wers, after_wers)]

    avg_before = sum(before_wers) / len(before_wers)
    avg_after  = sum(after_wers)  / len(after_wers)

    print(f"\n{'─'*55}")
    print(f"  WER BEFORE normalisation:  {avg_before:.2f}%")
    print(f"  WER AFTER  normalisation:  {avg_after:.2f}%")
    print(f"  Reduction:                 {avg_before - avg_after:.2f} pp")
    print(f"{'─'*55}\n")

    print("Per-utterance sample (first 10):")
    print(df[["reference", "hypothesis", "wer_before_norm", "wer_after_norm"]].head(10).to_string(index=False))

    df.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",  default="sample_errors.csv")
    parser.add_argument("--output_csv", default="sample_errors_normalised.csv")
    args = parser.parse_args()
    evaluate_before_after(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
