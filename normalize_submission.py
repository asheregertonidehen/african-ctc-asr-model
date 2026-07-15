"""Conservative WER normalizer for a hackathon submission CSV.

Only removes things that are UNAMBIGUOUSLY noise for these 6 Latin-script
languages (swa, kik, luo, som, mas, kln):
  * characters from foreign scripts the model hallucinated (Devanagari, Thaana,
    Arabic, Cyrillic, CJK, ...) -- detected as "not a LATIN-named letter"
  * stray combining marks / odd punctuation
  * duplicate / leading / trailing whitespace

DELIBERATELY KEEPS (removing these would hurt WER):
  * Latin diacritics (ĩ ũ ...) -- meaningful orthography, precomposed after NFC
  * apostrophes and hyphens -- legit (e.g. Swahili ng', hyphenated compounds)
  * the "..." placeholder rows (kept non-empty so the grader never sees a null)

Usage:  python normalize_submission.py IN.csv OUT.csv
"""
import re
import sys
import unicodedata

import pandas as pd

KEEP_PUNCT = set(" '-")
PLACEHOLDER = "..."


def _keep_char(ch: str) -> bool:
    if ch in KEEP_PUNCT or ch.isspace():
        return True
    if ch.isdigit():
        return True
    cat = unicodedata.category(ch)
    if cat[0] == "L":  # a letter -- keep only if it's LATIN (drops Devanagari/Thaana/etc.)
        try:
            return unicodedata.name(ch).startswith("LATIN")
        except ValueError:
            return False
    return False  # combining marks (M*), symbols, other punctuation -> drop


def normalize_text(t) -> str:
    t = "" if t is None else str(t)
    if t.strip() == PLACEHOLDER:
        return PLACEHOLDER
    t = unicodedata.normalize("NFC", t)
    t = "".join(ch for ch in t if _keep_char(ch))
    t = re.sub(r"\s+", " ", t).strip()
    return t if t else PLACEHOLDER  # never emit an empty cell


def main(inp: str, out: str) -> None:
    df = pd.read_csv(inp)
    col = "transcription"
    before = df[col].fillna("").astype(str)
    df[col] = before.map(normalize_text)
    after = df[col]

    changed = int((before.map(lambda s: unicodedata.normalize("NFC", s)) != after).sum())
    # count how many rows had a foreign-script char removed
    foreign = int(before.map(
        lambda s: any(c.isalpha() and not unicodedata.name(c, "").startswith("LATIN")
                      for c in unicodedata.normalize("NFC", str(s)))
    ).sum())
    df = df[["id", "language", "transcription"]]
    df.to_csv(out, index=False)

    empty = int((after.str.strip() == "").sum())
    print(f"wrote {out}")
    print(f"  rows: {len(df)}  languages: {sorted(df['language'].unique())}")
    print(f"  rows changed by normalization: {changed}")
    print(f"  rows that had foreign-script (Devanagari/Thaana/...) chars removed: {foreign}")
    print(f"  nulls: {int(df[col].isna().sum())}  empty: {empty}   -> {'OK' if empty==0 else 'PROBLEM'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
