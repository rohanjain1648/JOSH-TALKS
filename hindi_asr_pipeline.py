"""
Hindi ASR Cleanup Pipeline
==========================
Operations:
  (a) Number Normalization  – spoken Hindi number-words → digits
  (b) English Word Detection – tag English-origin words in Hindi transcripts

Network is unavailable, so we use realistic ASR-style examples that mirror
what whisper-small produces on Hindi conversational audio.
"""

import re
import unicodedata
from typing import List, Tuple, Dict


# ──────────────────────────────────────────────────────────────────────────────
# SECTION A: NUMBER NORMALISATION
# ──────────────────────────────────────────────────────────────────────────────

# ── 1. Atomic word → digit mappings ──────────────────────────────────────────
ONES = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16, "सत्रह": 17,
    "अठारह": 18, "उन्नीस": 19,
}

TENS = {
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29,
    "तीस": 30, "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34,
    "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44,
    "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49,
    "पचास": 50, "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54,
    "पचपन": 55, "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    "साठ": 60, "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64,
    "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69,
    "सत्तर": 70, "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74,
    "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उन्यासी": 79,
    "अस्सी": 80, "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84,
    "पचासी": 85, "छियासी": 86, "सतासी": 87, "अट्ठासी": 88, "नवासी": 89,
    "नब्बे": 90, "इक्यानबे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94,
    "पचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}

MULTIPLIERS = {
    "सौ": 100, "हज़ार": 1000, "हजार": 1000,
    "लाख": 100_000, "करोड़": 10_000_000,
}

ALL_NUM_WORDS = {**ONES, **TENS, **MULTIPLIERS}


def _is_number_word(w: str) -> bool:
    return w in ALL_NUM_WORDS


# ── 2. Idiom / frozen-phrase guard list ──────────────────────────────────────
# These are set phrases where converting number-words to digits distorts meaning.
IDIOM_PHRASES = [
    "दो-चार",          # "a few"
    "दो चार",
    "चार-छह",          # "a handful"
    "सात-आठ",
    "पाँच-सात",
    "पांच-सात",
    "दो-तीन",
    "दो तीन",
    "तीन-चार",
    "तीन चार",
    "चार-पाँच",
    "चार पांच",
    "एक-दो",
    "एक दो",
    "दस-बीस",
    "बीस-पच्चीस",
    "सौ-दो सौ",
    "हज़ार-दो हज़ार",
    "सात आठ",          # approximate range "7 or 8" (like "a few years")
    "पाँच छह",
    "छह सात",
    "नौ दस",
]

# Regex patterns for idiom detection (compiled once)
IDIOM_PATTERNS = [re.compile(r'\b' + re.escape(p) + r'\b') for p in IDIOM_PHRASES]


def _contains_idiom(span: str) -> bool:
    for pat in IDIOM_PATTERNS:
        if pat.search(span):
            return True
    return False


# ── 3. Core converter: list of consecutive number-words → int ─────────────────
def _words_to_int(words: List[str]) -> int:
    """
    Convert a sequence of Hindi number words to an integer.
    Handles: एक हज़ार तीन सौ चौवन → 1354
    """
    total = 0
    current = 0

    for w in words:
        if w in ONES:
            current += ONES[w]
        elif w in TENS:
            current += TENS[w]
        elif w in MULTIPLIERS:
            m = MULTIPLIERS[w]
            if m >= 1000:
                total += (current if current > 0 else 1) * m
                current = 0
            else:  # सौ (100)
                current = (current if current > 0 else 1) * m
    total += current
    return total


# ── 4. Tokenise, detect spans, convert ───────────────────────────────────────
def normalize_numbers(text: str) -> Tuple[str, List[Dict]]:
    """
    Returns (normalized_text, list_of_changes).
    Each change: {original, replacement, reason}
    """
    changes = []

    # First, protect idioms by tagging their positions
    protected_spans = []
    for pat in IDIOM_PATTERNS:
        for m in pat.finditer(text):
            protected_spans.append((m.start(), m.end()))

    def _in_protected(start: int, end: int) -> bool:
        for ps, pe in protected_spans:
            if start < pe and end > ps:
                return True
        return False

    # Tokenise on whitespace, keeping positions
    tokens = list(re.finditer(r'\S+', text))
    result_tokens = [t.group() for t in tokens]
    i = 0

    while i < len(tokens):
        tok = tokens[i].group()
        # Strip trailing punctuation for lookup
        clean = re.sub(r'[,।!?]+$', '', tok)

        if not _is_number_word(clean):
            i += 1
            continue

        # Gather the longest contiguous run of number words
        j = i
        span_words = []
        while j < len(tokens):
            c = re.sub(r'[,।!?]+$', '', tokens[j].group())
            if _is_number_word(c):
                span_words.append(c)
                j += 1
            else:
                break

        span_start = tokens[i].start()
        span_end   = tokens[j - 1].end()

        if _in_protected(span_start, span_end):
            i = j
            continue

        original_phrase = ' '.join(tokens[k].group() for k in range(i, j))
        number = _words_to_int(span_words)
        replacement = str(number)

        # Replace in result_tokens
        result_tokens[i] = replacement
        for k in range(i + 1, j):
            result_tokens[k] = ''

        changes.append({
            "original": original_phrase,
            "replacement": replacement,
            "position": i,
        })
        i = j

    # Rebuild text preserving spacing
    parts = []
    for orig_tok, new_tok in zip(tokens, result_tokens):
        if new_tok != '':
            parts.append(new_tok)

    # Reconstruct with original spacing
    out = text
    for change in reversed(changes):
        # We rebuild from the token list instead
        pass

    # Simpler rebuild: rejoin with single spaces (acceptable for pipeline output)
    out_tokens = [t for t in result_tokens if t != '']
    return ' '.join(out_tokens), changes


# ──────────────────────────────────────────────────────────────────────────────
# SECTION B: ENGLISH WORD DETECTION
# ──────────────────────────────────────────────────────────────────────────────

# Common English loanwords that appear in Hindi speech, written in Devanagari
# (per transcription guideline: English spoken words → Devanagari script)
ENGLISH_LOANWORDS_DEVANAGARI = {
    # Tech / professional
    "इंटरव्यू", "इंटरव्यूअर", "जॉब", "ऑफिस", "मैनेजर", "टीम", "प्रोजेक्ट",
    "मीटिंग", "प्रेजेंटेशन", "फीडबैक", "टारगेट", "सेल्स", "मार्केट",
    "कंपनी", "बिज़नेस", "सैलरी", "बोनस", "प्रमोशन", "ट्रेनिंग", "इंटर्नशिप",
    "रिज्यूमे", "प्रोफाइल", "लिंक्डइन", "ईमेल", "लैपटॉप", "कंप्यूटर",
    "सॉफ्टवेयर", "ऐप", "वेबसाइट", "ऑनलाइन", "ऑफलाइन", "डिजिटल",
    "इंटरनेट", "वाई-फाई", "फोन", "स्मार्टफोन", "चार्जर",
    # Finance / commerce
    "लोन", "ईएमआई", "क्रेडिट", "डेबिट", "बैंक", "अकाउंट", "ट्रांसफर",
    "पेमेंट", "कैश", "चेक", "इनवॉइस", "रिसीट",
    # Education
    "कोर्स", "क्लास", "टेस्ट", "एग्जाम", "रिजल्ट", "मार्क्स", "ग्रेड",
    "सर्टिफिकेट", "डिग्री", "कॉलेज", "यूनिवर्सिटी", "स्कूल", "टीचर",
    # Daily life
    "टाइम", "डेट", "प्लान", "स्टेशन", "बस", "ट्रेन", "फ्लाइट",
    "होटल", "रूम", "टिकट", "बुकिंग", "चेक-इन", "चेक-आउट",
    # Soft skills / HR
    "स्किल", "एक्सपीरियंस", "पोटेंशियल", "परफॉर्मेंस", "रिव्यू",
    "प्रॉब्लम", "सॉल्यूशन", "चैलेंज", "गोल", "टास्क", "रिस्पॉन्सिबिलिटी",
    # Attitude / emotions (colloquial)
    "कूल", "फाइन", "ओके", "ओके", "सॉरी", "थैंक्यू", "प्लीज़", "हेलो",
    "हाय", "बाय", "गुड", "बेस्ट", "ग्रेट", "नाइस", "परफेक्ट",
    # Roman-script English words (in case ASR outputs them)
    "interview", "job", "office", "manager", "team", "project",
    "meeting", "salary", "company", "email", "laptop", "computer",
    "software", "app", "website", "online", "phone", "bank",
    "account", "payment", "course", "class", "test", "exam",
    "result", "marks", "degree", "college", "university", "skill",
    "experience", "problem", "solution", "goal", "task", "ok", "okay",
    "sorry", "thanks", "please", "hello", "hi", "bye", "good", "best",
    "great", "nice", "perfect", "cool", "fine",
}


def _is_roman_script(word: str) -> bool:
    """True if word is written in Latin (Roman) script."""
    latin_count = sum(1 for c in word if unicodedata.name(c, '').startswith('LATIN'))
    return latin_count > len(word) * 0.5 if word else False


def _is_devanagari(word: str) -> bool:
    return any('\u0900' <= c <= '\u097F' for c in word)


def detect_and_tag_english(text: str) -> Tuple[str, List[str]]:
    """
    Returns (tagged_text, list_of_detected_english_words).
    Tags: [EN]word[/EN]
    """
    # Tokenise preserving punctuation attachment
    tokens = re.findall(r'\S+', text)
    detected = []
    tagged_tokens = []

    for token in tokens:
        # Strip punctuation for lookup
        clean = re.sub(r'^[।,!?"\'-]+|[।,!?"\'-]+$', '', token)
        prefix = token[:len(token) - len(token.lstrip('।,!?"\'- '))]
        suffix = token[len(clean) + len(prefix):]

        is_english = False

        # Check 1: Roman script word
        if _is_roman_script(clean) and len(clean) >= 2:
            is_english = True

        # Check 2: Known Devanagari loanword
        elif clean in ENGLISH_LOANWORDS_DEVANAGARI:
            is_english = True

        # Check 3: Mixed word (e.g. starts with Latin)
        elif any(_is_roman_script(c) for c in clean if c.strip()):
            is_english = True

        if is_english:
            detected.append(clean)
            tagged_tokens.append(f"{prefix}[EN]{clean}[/EN]{suffix}")
        else:
            tagged_tokens.append(token)

    return ' '.join(tagged_tokens), detected


# ──────────────────────────────────────────────────────────────────────────────
# REALISTIC SYNTHETIC EXAMPLES
# (These mirror what whisper-small produces on Hindi conversational audio)
# ──────────────────────────────────────────────────────────────────────────────

# Each entry: (recording_id, raw_asr, human_reference)
SYNTHETIC_EXAMPLES = [
    # ── Number normalisation cases ──
    {
        "id": "825780_seg_04",
        "raw_asr": "मेरी सैलरी तीन लाख पचास हज़ार रुपये सालाना है",
        "reference": "मेरी सैलरी तीन लाख पचास हज़ार रुपये सालाना है",
        "note": "Compound number: तीन लाख पचास हज़ार",
    },
    {
        "id": "825780_seg_07",
        "raw_asr": "मेरा इंटरव्यू दो बजे है और मुझे पाँच सौ रुपये टैक्सी के देने होंगे",
        "reference": "मेरा इंटरव्यू दो बजे है और मुझे पाँच सौ रुपये टैक्सी के देने होंगे",
        "note": "Time reference (दो बजे) + currency (पाँच सौ)",
    },
    {
        "id": "825727_seg_02",
        "raw_asr": "कंपनी में एक हज़ार दो सौ पचास लोग काम करते हैं",
        "reference": "कंपनी में एक हज़ार दो सौ पचास लोग काम करते हैं",
        "note": "Large compound number: 1,250",
    },
    {
        "id": "825727_seg_09",
        "raw_asr": "उसने मुझसे दो-चार बातें कीं और चला गया",
        "reference": "उसने मुझसे दो-चार बातें कीं और चला गया",
        "note": "IDIOM: दो-चार = 'a few' — must NOT convert",
    },
    {
        "id": "988596_seg_03",
        "raw_asr": "यह प्रोजेक्ट तीन महीने में पूरा होगा और बजट पचास लाख है",
        "reference": "यह प्रोजेक्ट तीन महीने में पूरा होगा और बजट पचास लाख है",
        "note": "Mixed sentence: तीन महीने (duration) + पचास लाख (budget)",
    },
    {
        "id": "988596_seg_11",
        "raw_asr": "मेरे पास एक-दो ऑप्शन हैं",
        "reference": "मेरे पास एक-दो ऑप्शन हैं",
        "note": "IDIOM: एक-दो = 'one or two / a couple' — must NOT convert",
    },
    {
        "id": "103445_seg_05",
        "raw_asr": "उसने सात आठ साल पहले यह काम शुरू किया था",
        "reference": "उसने सात आठ साल पहले यह काम शुरू किया था",
        "note": "IDIOM: सात आठ = approximate range, not literal 56",
    },
    # ── English detection cases ──
    {
        "id": "825780_seg_12",
        "raw_asr": "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
        "reference": "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
        "note": "English loanwords in Devanagari: इंटरव्यू, जॉब",
    },
    {
        "id": "825727_seg_14",
        "raw_asr": "यह problem solve नहीं हो रहा है मुझे manager से बात करनी होगी",
        "reference": "यह प्रॉब्लम सॉल्व नहीं हो रहा है मुझे मैनेजर से बात करनी होगी",
        "note": "ASR outputs Roman-script English: problem, solve, manager",
    },
    {
        "id": "988596_seg_07",
        "raw_asr": "हमें अगले meeting में presentation देनी है और feedback लेना है",
        "reference": "हमें अगले मीटिंग में प्रेजेंटेशन देनी है और फीडबैक लेना है",
        "note": "Mixed Roman+Devanagari in raw ASR output",
    },
    {
        "id": "103445_seg_08",
        "raw_asr": "उसका performance बहुत अच्छा है और salary भी ठीक है",
        "reference": "उसका परफॉर्मेंस बहुत अच्छा है और सैलरी भी ठीक है",
        "note": "Roman English in ASR: performance, salary",
    },
    {
        "id": "103445_seg_15",
        "raw_asr": "ये course online है और इसमें तीन सौ रुपये की fees है",
        "reference": "यह कोर्स ऑनलाइन है और इसमें तीन सौ रुपये की फीस है",
        "note": "Both operations: number + English detection",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(examples):
    results = []
    for ex in examples:
        norm_text, num_changes = normalize_numbers(ex["raw_asr"])
        tagged_text, en_words = detect_and_tag_english(norm_text)

        results.append({
            "id": ex["id"],
            "raw_asr": ex["raw_asr"],
            "after_num_norm": norm_text,
            "after_en_tagging": tagged_text,
            "reference": ex["reference"],
            "num_changes": num_changes,
            "en_words": en_words,
            "note": ex["note"],
        })
    return results


if __name__ == "__main__":
    results = run_pipeline(SYNTHETIC_EXAMPLES)
    for r in results:
        print(f"\n{'='*70}")
        print(f"ID        : {r['id']}")
        print(f"NOTE      : {r['note']}")
        print(f"RAW ASR   : {r['raw_asr']}")
        print(f"NUM NORM  : {r['after_num_norm']}")
        print(f"EN TAGGED : {r['after_en_tagging']}")
        print(f"REFERENCE : {r['reference']}")
        if r['num_changes']:
            print(f"NUMBERS   : {[(c['original'], '→', c['replacement']) for c in r['num_changes']]}")
        if r['en_words']:
            print(f"ENGLISH   : {r['en_words']}")
