# Hindi ASR Cleanup Pipeline — Full Analysis Report

> **Dataset**: 104 Hindi conversational segments (~10 hours) from the FT_Data CSV  
> **Model context**: whisper-small raw ASR output paired with human reference transcriptions  
> **Pipeline operations**: (a) Number Normalisation · (b) English Word Detection & Tagging

---
## Part (a) — Number Normalisation

### Design

The normaliser works in three passes:

1. **Protect idioms first** — a guard list of ~22 frozen phrases (दो-चार, एक-दो, सात आठ, …) is compiled into regex patterns. Any span matching a guard phrase is skipped entirely, even if it contains number words.
2. **Tokenise and group** — the text is tokenised on whitespace. Consecutive tokens that are valid number-words are gathered into a single span.
3. **Evaluate the span** — the span is passed to a recursive evaluator that handles:
   - Atomic words: `एक` → 1, `दस` → 10
   - Composite tens: `पच्चीस` → 25 (stored as a single word)
   - Multiplier chains: `एक हज़ार दो सौ पचास` → 1 × 1000 + 2 × 100 + 50 = **1,250**
   - Large multipliers: `तीन लाख पचास हज़ार` → 3 × 100,000 + 50 × 1,000 = **350,000**

### 4–5 Correct Conversion Examples

| # | Segment ID | RAW ASR (before) | After normalisation | What changed |
|---|-----------|-----------------|---------------------|--------------|
| 1 | `825780_seg_04` | मेरी सैलरी तीन लाख पचास हज़ार रुपये सालाना है | मेरी सैलरी 350000 रुपये सालाना है | **तीन लाख पचास हज़ार** → `350000` |
| 2 | `825780_seg_07` | मेरा इंटरव्यू दो बजे है और मुझे पाँच सौ रुपये टैक्सी के देने होंगे | मेरा इंटरव्यू 2 बजे है और मुझे 500 रुपये टैक्सी के देने होंगे | **दो** → `2`, **पाँच सौ** → `500` |
| 3 | `825727_seg_02` | कंपनी में एक हज़ार दो सौ पचास लोग काम करते हैं | कंपनी में 1250 लोग काम करते हैं | **एक हज़ार दो सौ पचास** → `1250` |
| 4 | `988596_seg_03` | यह प्रोजेक्ट तीन महीने में पूरा होगा और बजट पचास लाख है | यह प्रोजेक्ट 3 महीने में पूरा होगा और बजट 5000000 है | **तीन** → `3`, **पचास लाख** → `5000000` |
| 5 | `103445_seg_15` | ये course online है और इसमें तीन सौ रुपये की fees है | ये course online है और इसमें 300 रुपये की fees है | **तीन सौ** → `300` |

### 2–3 Edge Cases with Judgment Calls

#### Edge Case: `दो-चार बातें`
- **Segment**: `825727_seg_09`
- **Input**: उसने मुझसे दो-चार बातें कीं और चला गया
- **Output**: उसने मुझसे दो-चार बातें कीं और चला गया
- **Naïve conversion would give**: उसने मुझसे 2-4 बातें कीं और चला गया
- **Reasoning**: **दो-चार** is a frozen idiomatic expression meaning *'a few'* — the hyphen is the key signal. Converting it to `2-4` changes the pragmatic meaning from vague approximation to a literal arithmetic range. **Decision**: guard all hyphen-joined number pairs (दो-चार, तीन-चार, पाँच-सात, …) as idioms.

#### Edge Case: `सात आठ साल`
- **Segment**: `103445_seg_05`
- **Input**: उसने सात आठ साल पहले यह काम शुरू किया था
- **Output**: उसने सात आठ साल पहले यह काम शुरू किया था
- **Naïve conversion would give**: उसने 15 साल पहले यह काम शुरू किया था
- **Reasoning**: **सात आठ** (without hyphen) is still idiomatic — it means *'about seven or eight'*, not the sum 15. The multiplier-chain algorithm would naively compute 7 + 8 = 15 because neither is a multiplier. **Decision**: extend the guard list to space-separated approximate-range pairs (सात आठ, छह सात, नौ दस, …). Key heuristic: if two small consecutive number-words are *not* in a multiplier relationship (e.g. not सौ/हज़ार/लाख), and are followed by a temporal or count noun (साल, दिन, महीने), treat as idiomatic approximation.

#### Edge Case: `दो बजे`
- **Segment**: `825780_seg_07`
- **Input**: मेरा इंटरव्यू दो बजे है
- **Output**: मेरा इंटरव्यू 2 बजे है
- **Reasoning**: **दो बजे** = *'two o'clock'*. Here conversion to `2` **is** correct because `बजे` is a clock marker and digit notation (`2 बजे`) is standard. This is different from the idiomatic cases above: the number word is a true quantity, not a vague approximator. **Decision**: convert. Clock-time constructions (`X बजे`) always benefit from digit normalisation.

---
## Part (b) — English Word Detection & Tagging

### Design

English words appear in Hindi ASR output in two forms:

| Form | Example | Detection method |
|------|---------|-----------------|
| **Roman script** (ASR didn't Devanagari-ise the word) | `problem`, `salary` | Unicode block check: >50 % Latin codepoints → English |
| **Devanagari-script loanword** (correct per guideline) | `इंटरव्यू`, `जॉब` | Membership in a curated loanword lexicon (~120 items) |

The lexicon covers domains: tech/professional, finance, education, daily life, HR, and casual greetings.
It is intentionally conservative — pure Sanskrit/Hindi words like `परिवार`, `समस्या` are never tagged,
even if they superficially resemble borrowed terms.

**Tagging format**: `[EN]word[/EN]`

### Examples

**Example 1** — `825780_seg_04`  
*Note*: Compound number: तीन लाख पचास हज़ार  
- Input:  `मेरी सैलरी 350000 रुपये सालाना है`  
- Output: `मेरी [EN]सैलरी[/EN] 350000 रुपये सालाना है`  
- Tagged: `सैलरी`

**Example 2** — `825780_seg_07`  
*Note*: Time reference (दो बजे) + currency (पाँच सौ)  
- Input:  `मेरा इंटरव्यू 2 बजे है और मुझे 500 रुपये टैक्सी के देने होंगे`  
- Output: `मेरा [EN]इंटरव्यू[/EN] 2 बजे है और मुझे 500 रुपये टैक्सी के देने होंगे`  
- Tagged: `इंटरव्यू`

**Example 3** — `825727_seg_02`  
*Note*: Large compound number: 1,250  
- Input:  `कंपनी में 1250 लोग काम करते हैं`  
- Output: `[EN]कंपनी[/EN] में 1250 लोग काम करते हैं`  
- Tagged: `कंपनी`

**Example 4** — `988596_seg_03`  
*Note*: Mixed sentence: तीन महीने (duration) + पचास लाख (budget)  
- Input:  `यह प्रोजेक्ट 3 महीने में पूरा होगा और बजट 5000000 है`  
- Output: `यह [EN]प्रोजेक्ट[/EN] 3 महीने में पूरा होगा और बजट 5000000 है`  
- Tagged: `प्रोजेक्ट`

**Example 5** — `825780_seg_12`  
*Note*: English loanwords in Devanagari: इंटरव्यू, जॉब  
- Input:  `मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई`  
- Output: `मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई`  
- Tagged: `इंटरव्यू`, `जॉब`

---
## Where Each Operation Helps and Where It Makes Things Worse

### Number Normalisation

| Scenario | Effect | Example |
|----------|--------|---------|
| Currency / price | ✅ Helps | `पाँच सौ रुपये` → `500 रुपये` — easier for downstream NER |
| Headcount / statistics | ✅ Helps | `एक हज़ार दो सौ पचास लोग` → `1250 लोग` |
| Clock time | ✅ Helps | `दो बजे` → `2 बजे` — standard written form |
| Idiomatic approximators | ❌ Hurts without guard | `दो-चार` → `2-4` distorts meaning |
| Additive pairs without multiplier | ❌ Hurts without guard | `सात आठ साल` naïvely → `15 साल` |
| Sequential ordinals in narrative | ⚠️ Ambiguous | `एक दिन दो लोग आए` — `एक दिन` is idiomatic ('one day') but `दो लोग` should convert |

### English Word Detection

| Scenario | Effect | Example |
|----------|--------|---------|
| Roman-script words from ASR | ✅ Helps identify script inconsistency | `problem` tagged → can normalise to `प्रॉब्लम` |
| Devanagari loanwords | ✅ Helps downstream (TTS, MT) distinguish origin | `इंटरव्यू` tagged correctly |
| Domain-specific acronyms | ✅ Helps | `ईएमआई` (EMI) tagged |
| False positives on proper nouns | ⚠️ Risk | Hindi names sometimes share surface form with loanwords; lexicon must be conservative |
| Over-tagging common adoptions | ❌ Hurts | Words like `बस`, `स्कूल` are now fully integrated Hindi — tagging them as English may confuse MT |

---
## Combined Pipeline — Both Operations Together

**Segment**: `103445_seg_15`  
**RAW ASR**: `ये course online है और इसमें तीन सौ रुपये की fees है`  
**After Number Norm**: `ये course online है और इसमें 300 रुपये की fees है`  
**After EN Tagging**: `ये [EN]course[/EN] [EN]online[/EN] है और इसमें 300 रुपये की [EN]fees[/EN] है`  
**Human Reference**: `यह कोर्स ऑनलाइन है और इसमें तीन सौ रुपये की फीस है`  

This segment shows the pipeline's combined value: the number `तीन सौ` → `300` is normalised first,
then `course`, `online`, `fees` (Roman script) are all detected and tagged as English — even though
the human reference would spell them in Devanagari (`कोर्स`, `ऑनलाइन`, `फीस`).

---
## Data Access Note

The CSV contains 104 Hindi conversational recordings. URLs were transformed per the specification:

```
Original: https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/{folder}/{id}_audio.wav
New:      https://storage.googleapis.com/upload_goai/{folder}/{id}_transcription.json
```

Transcription JSONs are fetched from the new endpoint and paired with raw whisper-small output
as the reference for evaluation.