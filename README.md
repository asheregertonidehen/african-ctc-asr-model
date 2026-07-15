# African CTC ASR Model

Finetune Meta's **omnilingual-asr** CTC models (300M) on six African languages —
**Swahili (swa), Kikuyu (kik), Dholuo (luo), Somali (som), Maasai (mas), Kalenjin (kln)** —
and produce a competition-ready submission CSV. Built for the DigitalUmuganda /
Anv-Ke multilingual ASR challenge, but the pipeline generalizes to any
mixture-parquet CTC finetune.

The whole chain is **three commands** on a fresh GPU box:

```bash
bash setup_pod.sh                 # environment (deps + asset cards + verify)
SMOKE=1 bash finetune.sh          # ~10 min end-to-end validation
bash finetune.sh                  # ingest -> stats -> card -> train
bash submit_from_checkpoint.sh    # checkpoint -> submission.csv
```

Runs on a single 24 GB card (RTX 3090/4090) up to multi-GPU nodes — batch size
and gradient accumulation are auto-tuned to the detected GPU(s).

## What's in here

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

## Requirements

- CUDA GPU (24 GB+ recommended), Python 3.10–3.12 (NOT 3.13+ — fairseq2 wheels)
- A HuggingFace token with access to the gated source datasets
  (`Anv-ke/*`, `DigitalUmuganda/Afrivoice*`) — `export HF_TOKEN=hf_...`
- Kaggle API creds (`~/.kaggle/kaggle.json`) for the test-set download
- ~60 GB free disk on `$WORKSPACE` (default `/workspace`): subset data ~15 GB,
  test set ~37 GB, checkpoints ~4 GB each

## How the pipeline works

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
   checkpoint every 500 steps. **Stop when val WER plateaus** (typically
   1500–3000 steps) — the best checkpoint is usually not the last.
5. **Submission** — greedy CTC over the test parquets, chunked at 35 s, every
   test id guaranteed present and non-null in the CSV. `--limit N` smoke-tests
   the whole path in minutes. Optionally post-process with
   `normalize_submission.py` (removes hallucinated foreign-script characters;
   keeps meaningful Latin diacritics like ĩ/ũ and apostrophes).

### v1 vs v2 checkpoints (important)

The v2 models (`*_v2`, `omniASR_tokenizer_written_v2`) emit a different
*written-form orthography* than v1. On this competition's references, **v1
scored materially better than v2** (both zeroshot and finetuned). v1 is the
default everywhere; the v2 config/cards are included for experiments. Never
mix a v1 checkpoint with the v2 tokenizer or vice versa — vocab sizes differ
(9812 vs 10288) and training will silently target garbage.

## Hard-won gotchas this repo already handles

| Problem | Handling |
|---|---|
| `huggingface_hub` 1.x breaks transformers/ray import chain | pinned `<1.0` in setup |
| PyPI `omnilingual-asr` 0.1.0 ships no v2 cards / archs and no `omniASR_tokenizer_v1` card | setup installs cards from git + grafts the v2-arch `config.py` |
| Full `rc_models_v1.yaml` + `rc_models_v2.yaml` collide (duplicate W2V cards) | v1 installed as a minimal supplement |
| `torchaudio.save(BytesIO, format="ogg")` segfaults Ray workers (torchaudio 2.9) | encode via `soundfile` FLAC everywhere |
| Some clips "decode" with `sample_rate<=0` / empty waveform, then crash resample | dropped at ingest; null `audio_size` rows filtered before write |
| A null `audio_size` reaching training crashes validation (`'length' ... is of type float`) | rows never written; see also the in-place cleaner pattern in git history |
| Parallel per-repo ingest OOMs (each 2000-clip read is a ~20 GB Ray block) | sequential ingest, capped object store, spill to disk |
| Afrivoice webm decode is serial and looks "frozen" for 30+ min | `--max_clips_per_shard` caps |
| Flaky pod networks (`RemoteDisconnected`) abort ingest | per-repo retry w/ backoff + longer HF timeout + per-repo failure isolation |
| CUDA OOM with "13 GB reserved but unallocated" | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + VRAM-tiered batch |
| fairseq2 CLI rejects bare overrides | overrides passed after `--config` |
| `validate_every_n_steps` must be a multiple of `publish_metrics_every_n_steps` | smoke overrides set all three |
| fairseq2 checkpoints are nested (`ws_*/checkpoints/step_N/model/pp_00/...`) | submit script auto-discovers the layout |
| Empty/null transcription cells get a submission rejected | every id backfilled with `"..."`, scrubbed, format-validated |
| Model hallucinates Devanagari/Thaana on hard clips | `normalize_submission.py` strips non-Latin letters, keeps diacritics |

## Improving WER further

Greedy CTC leaves a large gap between character accuracy and word accuracy
(one wrong character = one wrong word). The proven next steps, in order of
impact:

1. **KenLM shallow fusion** — a 4-gram KenLM per language over the training
   transcripts, decoded with `pyctcdecode`, dropped WER from ~0.65 to ~0.52
   on this competition (zeroshot). It applies unchanged to a finetuned v1
   checkpoint (same vocab).
2. **More data** — raise `TAKE` (ingest time scales linearly; training time
   doesn't).
3. **Quantization** — int8 quantization of the acoustic model slightly
   *improved* WER in our runs while also satisfying edge-device constraints
   (<8 GB RAM, 1–2x RTF on CPU).

## Credits

- Model, recipe skeleton, and data tooling: [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)
  (`asr_recipe/` and `dataprep/audio_tools.py` derive from it; see the
  copyright headers).
- Datasets: Anv-Ke and DigitalUmuganda (gated on HuggingFace; request access).
