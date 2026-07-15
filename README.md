
Finetune Meta's **omnilingual-asr** CTC models (300M) on six African languages —
**Swahili (swa), Kikuyu (kik), Dholuo (luo), Somali (som), Maasai (mas), Kalenjin (kln)** —
and produce a competition-ready submission CSV. 

```bash
bash setup_pod.sh                 # environment (deps + asset cards + verify)
SMOKE=1 bash finetune.sh          # ~10 min end-to-end validation
bash finetune.sh                  # ingest -> stats -> card -> train
bash submit_from_checkpoint.sh    # checkpoint -> submission.csv
```

Runs on a single card up to multi-GPU nodes — batch size
and gradient accumulation are auto-tuned to the detected GPU(s).

```
setup_pod.sh                  fresh-box bootstrap: deps, version pins, asset cards, import verify
finetune.sh                   one-command pipeline: ingest -> stats -> dataset card -> train (idempotent, resumable)
submit_from_checkpoint.sh     newest checkpoint -> Kaggle test set -> submission CSV (greedy decode)
infer_testset.py              the inference worker (3-backend audio decode, chunking, null-proof backfill)
normalize_submission.py       conservative submission cleaner (strips foreign-script hallucinations, keeps diacritics)
asr_recipe/                   fairseq2 training recipe (from omnilingual-asr git) + the two working configs
  configs/hackathon-ctc-v1-finetune.yaml     <- default (scored better on this competition)
  configs/hackathon-ctc-v2-finetune.yaml     <- v2 "written" orthography variant
dataprep/
  ingest_hackathon_datasets.py   HF -> mixture-parquet for all 7 source repos (3 ingestion paths)
  audio_tools.py                 audio decode/encode processors (parallelized, crash-hardened)
  compute_stats.py               language-hours TSV the sampler needs
cards/                        dataset asset cards (v1 + v2 tokenizer variants)
```

- CUDA GPU (24 GB+ recommended), Python 3.10–3.12 (NOT 3.13+ — fairseq2 wheels)
- A HuggingFace token with access to the gated source datasets
  (`Anv-ke/*`, `DigitalUmuganda/Afrivoice*`) — `export HF_TOKEN=hf_...`
- Kaggle API creds (`~/.kaggle/kaggle.json`) for the test-set download
- ~60 GB free disk on `$WORKSPACE` (default `/workspace`): subset data ~15 GB,
  test set ~37 GB, checkpoints ~4 GB each

1. **Ingest** (`dataprep/ingest_hackathon_datasets.py`) — streams a *subset*
   (`TAKE`, default 2000 train clips/language) of the five Anv-ke repos plus a
   capped slice of the two Afrivoice repos into hive-partitioned
   mixture-parquet (`corpus=/split=/language=`). Streaming a subset matters:
   the full Anv-ke repos are **~850 GB**. Seeded-shuffle + skip/take gives
   deterministic, disjoint chunks if you ingest more later.
2. **Stats** (`dataprep/compute_stats.py`) — writes the language-hours TSV;
   training temperature-samples languages by it (`beta_language: 0.5`).
3. **Dataset card** — installed into the omnilingual package's card dir so
   fairseq2 resolves `dataset.name: hackathon_asr`.
4. **Train** (`asr_recipe`) — CTC finetune from `omniASR_CTC_300M`, bf16,
   lr 1e-5, encoder frozen for the first 200 steps, validation WER + a
   checkpoint every 500 steps. 
5. **Submission** — greedy CTC over the test parquets, chunked at 35 s, every
   test id guaranteed present and non-null in the CSV. `--limit N` smoke-tests
   the whole path in minutes. Optionally post-process with
   `normalize_submission.py`
   
## Credits

- Model, recipe skeleton, and data tooling: [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)
- Datasets: Anv-Ke and DigitalUmuganda
