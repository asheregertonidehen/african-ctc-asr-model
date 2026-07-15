"""Run a (finetuned) CTC checkpoint over the Kaggle test set and write a
submission CSV. Single-GPU, greedy decoding. Loads the model via an installed
asset card (see submit_from_checkpoint.sh, which writes the card + calls this).

Backfills any missing/undecodable id with a non-empty placeholder so the grader
never sees a null (the exact bug the first submission hit)."""
import argparse
import ctypes
import glob as globmod
import io
import time
from itertools import groupby

for _lib in globmod.glob("/usr/local/lib/python*/*-packages/nvidia/*/lib/libcudart.so*"):
    try:
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torchaudio
from fairseq2.data._memory import MemoryBlock
from fairseq2.data.audio import AudioDecoder
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 35
MIN_CHUNK_SAMPLES = SAMPLE_RATE // 4
ROW_BATCH_SIZE = 32
FLUSH_EVERY_CHUNKS = 64
EXPECTED_LANGS = {"swa", "kik", "luo", "som", "mas", "kln"}
EXPECTED_TOTAL_ROWS = 41_733
PLACEHOLDER = "..."
NA_LIKE = {"", "nan", "null", "none", "na", "n/a", "<na>"}

_dec = AudioDecoder(dtype=torch.float32)


def to_mono_1d(w):
    if w.ndim > 1:
        w = w.mean(dim=int(np.argmin(w.shape)))
    return w.reshape(-1)


def bytes_to_16k(b):
    """Decode audio bytes to a 16kHz mono numpy waveform, trying three backends
    so a single decoder's format gap never silently drops a clip (skipped rows
    score ~100% WER -- a decode failure is pure, avoidable loss):
      1. fairseq2 AudioDecoder (native)
      2. torchaudio (ffmpeg backend -- webm/mp3/m4a/...)
      3. soundfile (libsndfile -- flac/wav/ogg)
    Returns None only if ALL three fail (genuinely corrupt bytes)."""
    w = sr = None
    try:
        d = _dec(MemoryBlock(b))
        w, sr = to_mono_1d(d["waveform"]), int(d["sample_rate"])
    except Exception:
        pass
    if w is None:
        try:
            wav, sr = torchaudio.load(io.BytesIO(b))
            w = to_mono_1d(wav)
        except Exception:
            pass
    if w is None:
        try:
            import soundfile as sf
            arr, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
            w = to_mono_1d(torch.from_numpy(np.asarray(arr)))
        except Exception:
            pass
    if w is None or sr in (None, 0):
        return None
    if sr != SAMPLE_RATE:
        w = torchaudio.functional.resample(w, sr, SAMPLE_RATE)
    return w.numpy()


def chunk(w):
    n = CHUNK_SECONDS * SAMPLE_RATE
    cs = [w[i:i + n] for i in range(0, len(w), n)]
    if len(cs) > 1 and len(cs[-1]) < MIN_CHUNK_SAMPLES:
        cs = cs[:-1]
    return cs


def group_by_clip(meta, texts):
    for k, g in groupby(zip(meta, texts), key=lambda x: x[0]):
        yield k, [t for _, t in g]


def scrub(series):
    s = series.fillna(PLACEHOLDER).astype(str)
    return s.mask(s.str.strip().str.lower().isin(NA_LIKE), PLACEHOLDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True, help="installed asset card name for the checkpoint")
    ap.add_argument("--glob", required=True, help="test parquet glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: transcribe only the first N clips (missing ids get backfilled with placeholders)")
    args = ap.parse_args()

    dtype = torch.bfloat16 if (torch.cuda.is_available()
                               and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"loading '{args.card}' (dtype={dtype})...", flush=True)
    t0 = time.time()
    pipe = ASRInferencePipeline(model_card=args.card, dtype=dtype)
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    paths = sorted(globmod.glob(args.glob, recursive=True))
    print(f"{len(paths)} test parquet file(s)", flush=True)
    rows, batch_dicts, batch_meta = [], [], []
    seen = 0
    n_decode_fail = [0]   # list so the inner loop can mutate it
    run_start = time.time()

    def flush():
        if not batch_dicts:
            return
        texts = pipe.transcribe(batch_dicts, batch_size=args.batch_size)
        for (cid, lang), grp in group_by_clip(batch_meta, texts):
            rows.append({"id": cid, "language": lang, "transcription": " ".join(grp)})
        batch_dicts.clear()
        batch_meta.clear()

    done = False
    for pi, p in enumerate(paths, 1):
        if done:
            break
        pf = pq.ParquetFile(p)
        for rb in pf.iter_batches(batch_size=ROW_BATCH_SIZE):
            if done:
                break
            for row in rb.to_pylist():
                if args.limit and seen >= args.limit:
                    done = True
                    break
                seen += 1
                try:
                    wav = bytes_to_16k(row["audio"]["bytes"])
                except Exception as e:
                    wav = None
                    print(f"decode error id={row['id']}: {e}", flush=True)
                if wav is None or len(wav) == 0:
                    # undecodable by all 3 backends -> count it so we KNOW how many
                    # rows are lost to decode (vs model-blank); still backfilled later
                    n_decode_fail[0] += 1
                    continue
                for c in chunk(wav):
                    batch_dicts.append({"waveform": c, "sample_rate": SAMPLE_RATE})
                    batch_meta.append((row["id"], row["language"]))
                if len(batch_dicts) >= FLUSH_EVERY_CHUNKS:
                    flush()
        flush()
        el = (time.time() - run_start) / 60
        print(f"[{pi}/{len(paths)}] {len(rows)} rows | {seen} clips | {el:.1f} min", flush=True)
    flush()

    df = pd.DataFrame(rows).drop_duplicates(subset="id", keep="first")

    # Diagnose WHERE the empty/backfilled rows come from -- this tells you whether
    # to fix decoding (avoidable loss) or the model (needs finetune/KenLM).
    empty_model = int((df["transcription"].fillna("").astype(str).str.strip() == "").sum())
    print(f"\n=== SKIPPED-ROW BREAKDOWN ===", flush=True)
    print(f"  decode failures (all 3 backends failed): {n_decode_fail[0]}", flush=True)
    print(f"  model-blank (decoded fine, model output empty): {empty_model}", flush=True)
    if empty_model:
        bylang = (df[df["transcription"].fillna("").astype(str).str.strip() == ""]
                  .groupby("language").size().to_dict())
        print(f"  model-blank by language: {bylang}", flush=True)
    print(f"  (decode fails are avoidable loss; model-blanks need a better model / KenLM)\n", flush=True)

    # backfill every test id, scrub empties, validate
    id_frames = []
    for p in paths:
        try:
            id_frames.append(pq.read_table(p, columns=["id", "language"]).to_pandas())
        except Exception as e:
            print(f"WARN ids from {p}: {e}")
    allids = pd.concat(id_frames, ignore_index=True).drop_duplicates(subset="id")
    missing = allids[~allids["id"].isin(df["id"])].copy()
    if len(missing):
        missing["transcription"] = PLACEHOLDER
        df = pd.concat([df, missing], ignore_index=True)
        print(f"backfilled {len(missing)} missing id(s)")
    df["transcription"] = scrub(df["transcription"])
    df = df[["id", "language", "transcription"]]
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}", flush=True)

    problems = []
    if len(df) != EXPECTED_TOTAL_ROWS:
        problems.append(f"row count {len(df)} != {EXPECTED_TOTAL_ROWS}")
    if set(df["language"].unique()) - EXPECTED_LANGS:
        problems.append(f"bad langs: {set(df['language'].unique()) - EXPECTED_LANGS}")
    if int(df["transcription"].isna().sum()) or int((df["transcription"].str.strip() == "").sum()):
        problems.append("null/empty transcription cells present")
    print("FORMAT PROBLEMS: " + "; ".join(problems) if problems else "Format checks passed.")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
