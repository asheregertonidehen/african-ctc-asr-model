#!/usr/bin/env bash
# =============================================================================
# One-command finetune of omniASR_CTC_300M (v1) on the 6 hackathon languages
# (swa, kik, luo, som, mas, kln). Works on any CUDA box from a single 24GB
# card (RTX 3090/4090) up to multi-GPU nodes -- batch size and grad-accum are
# auto-tuned to the detected GPU(s).
#
# Does the WHOLE chain, and is idempotent -- safe to re-run; each phase is
# skipped if already done, and training RESUMES from its checkpoints:
#   1. ingest the HF data -> mixture-parquet        (skipped if data exists)
#   2. compute the language-distribution stats TSV  (always, cheap)
#   3. install the dataset asset card               (always, cheap)
#   4. train (torchrun if >1 GPU)                   (resumes if interrupted)
#
# USAGE (inside tmux, so it survives disconnects):
#   export HF_TOKEN=hf_...            # the source repos are gated
#   SMOKE=1 bash finetune.sh          # ~10 min end-to-end validation FIRST
#   bash finetune.sh                  # the real run
#
# Variants (override via env):
#   TAKE=4000 bash finetune.sh                # more clips per language
#   CONFIG=asr_recipe/configs/hackathon-ctc-v2-finetune.yaml \
#     CARD_SRC=cards/hackathon_asr.yaml \
#     RUN_DIR=/workspace/runs/ctc300m_v2 bash finetune.sh    # v2 checkpoint
#     (NOTE: on this competition's references the v2 "written" orthography
#      scored WORSE than v1 -- benchmark before committing to v2.)
# =============================================================================
set -euo pipefail

# ---- config (override via env) ----------------------------------------------
WORKSPACE="${WORKSPACE:-/workspace}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/data/hackathon_asr/version=0}"
STATS="${STATS:-$WORKSPACE/data/hackathon_asr/language_distribution_0.tsv}"
RUN_DIR="${RUN_DIR:-$WORKSPACE/runs/ctc300m_v1_hackathon}"   # STABLE name -> resumable
HF_CACHE="${HF_CACHE:-$WORKSPACE/hf_cache}"
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
SMOKE="${SMOKE:-0}"                                          # 1 => tiny end-to-end validation
CONFIG="${CONFIG:-asr_recipe/configs/hackathon-ctc-v1-finetune.yaml}"
CARD_SRC="${CARD_SRC:-cards/hackathon_asr_v1.yaml}"

FT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # repo root
cd "$FT_DIR"
mkdir -p "$WORKSPACE/data/hackathon_asr" "$HF_CACHE" "$(dirname "$RUN_DIR")"
LOG="$WORKSPACE/finetune_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
banner(){ echo; echo "=======================================================================" ; echo " $*"; echo "======================================================================="; }

banner "PREFLIGHT"
echo "workspace=$WORKSPACE  data_root=$DATA_ROOT  run_dir=$RUN_DIR  NGPU=$NGPU  smoke=$SMOKE"
# Some pod templates set HF_HUB_ENABLE_HF_TRANSFER=1 without shipping the
# package -- every hf_hub_download then raises. Disable unless importable.
python -c "import hf_transfer" 2>/dev/null || { export HF_HUB_ENABLE_HF_TRANSFER=0; echo "hf_transfer not installed -> HF_HUB_ENABLE_HF_TRANSFER=0"; }
# Flaky-network resilience: longer HF timeout (default 10s) so a slow response
# doesn't count as a drop. The ingest also retries transient failures per-repo.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
nvidia-smi -L || { echo "no GPUs visible"; exit 1; }
python -c "import omnilingual_asr, fairseq2; print('omnilingual_asr + fairseq2 import OK')" \
  || { echo "deps missing -- run setup_pod.sh first"; exit 1; }
[ -n "${HF_TOKEN:-}" ] || echo "WARNING: HF_TOKEN not set -- phase 1 will 401 on gated repos"

# ---- 1. ingest -------------------------------------------------------------
# Streamed subset (TAKE train clips/lang), all 5 Anv-ke repos SEQUENTIALLY in
# ONE process. These are long recordings (~10MB each): one repo's 2000-clip read
# is a ~20GB Ray block, so running 5 in parallel OOMs a 124GB box. Sequential
# holds ONE block at a time (~25GB peak) -- safe -- while a high-concurrency
# encoder + capped/spilling object store keep it fast. Full download is off the
# table anyway: the 5 repos total ~848GB.
TAKE="${TAKE:-2000}"
[ "$SMOKE" = "1" ] && TAKE=20 && echo "SMOKE mode: TAKE=20 clips/lang"
banner "PHASE 1/4  INGEST DATA  (streamed subset: TAKE=$TAKE/lang, sequential)"
if [ -d "$DATA_ROOT" ] && [ -n "$(ls -A "$DATA_ROOT" 2>/dev/null)" ]; then
  echo "data already present at $DATA_ROOT -> skipping ingestion"
  echo "(for a CLEAN run:  rm -rf $DATA_ROOT)"
else
  # Object store capped so it spills to disk instead of racing the read block
  # for RAM. concurrency = audio encoder actors within the process.
  INGEST_RAY_OBJ_GB="${INGEST_RAY_OBJ_GB:-8}" \
    python dataprep/ingest_hackathon_datasets.py anvke "$DATA_ROOT" \
      --take="$TAKE" --concurrency="${INGEST_CONCURRENCY:-4}" \
    || { echo "ANVKE INGEST FAILED"; exit 1; }

  # afrivoice decodes webm SERIALLY (no Ray) -- cap clips/shard or one shard is
  # thousands of clips and dominates ingest. smoke: tiny; full: ~TAKE/shard.
  if [ "$SMOKE" = "1" ]; then AFRI="--max_shards=1 --max_clips_per_shard=10"
  else                        AFRI="--max_shards=4 --max_clips_per_shard=$(( TAKE / 2 ))"; fi
  python dataprep/ingest_hackathon_datasets.py afrivoice_swahili "$DATA_ROOT" --cache_dir "$HF_CACHE" $AFRI \
    || echo "WARN: swahili ingest failed; continuing (Anv-ke data still trains)"
  python dataprep/ingest_hackathon_datasets.py afrivoice_somali  "$DATA_ROOT" --cache_dir "$HF_CACHE" $AFRI \
    || echo "WARN: somali ingest failed; continuing"
  python - <<PY
import pyarrow.dataset as ds
d = ds.dataset("$DATA_ROOT", partitioning="hive")
langs = sorted(set(str(f).split("language=")[-1].split("/")[0] for f in d.files))
print(f"ingested {d.count_rows()} rows across languages: {langs}")
PY
fi

# ---- 2. language-distribution stats ----------------------------------------
banner "PHASE 2/4  COMPUTE STATS TSV"
# ALWAYS recompute: a stale TSV from an earlier partial ingest silently skews
# the language sampling weights.
python dataprep/compute_stats.py "$DATA_ROOT" "$STATS"

# ---- 3. install the dataset asset card -------------------------------------
banner "PHASE 3/4  INSTALL DATASET CARD"
CARD_DIR="$(python -c "import omnilingual_asr, os; print(os.path.join(os.path.dirname(omnilingual_asr.__file__), 'cards', 'datasets'))")"
sed "s#/workspace/data/hackathon_asr/version=0#$DATA_ROOT#g" "$CARD_SRC" > "$CARD_DIR/hackathon_asr.yaml"
echo "installed dataset card from $CARD_SRC -> $CARD_DIR/hackathon_asr.yaml (data=$DATA_ROOT)"

# ---- 4. train (auto-tuned to the detected GPU(s); resumes from $RUN_DIR) -----
banner "PHASE 4/4  TRAIN  (validation WER prints every 500 steps)"
# Auto-tune to whatever GPU we landed on:
#   - max_num_elements by VRAM  (bigger batch only if the card can hold it)
#   - grad_accumulation scaled so the EFFECTIVE batch stays ~constant
# expandable_segments: a 4090 OOM'd with 8.5GB allocated but 13GB "reserved but
# unallocated" -- this reclaims allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 24000)
if   [ "$VRAM_MB" -ge 40000 ]; then MAXEL=7_680_000; BASE_ACCUM=8    # 48GB (L40S/A6000)
elif [ "$VRAM_MB" -ge 20000 ]; then MAXEL=1_920_000; BASE_ACCUM=16   # 24GB (3090/4090): 3.84M OOMs; halve batch, double accum
else                                MAXEL=960_000;   BASE_ACCUM=32   # smaller
fi
ACCUM=$(( BASE_ACCUM / (NGPU>0 ? NGPU : 1) )); [ "$ACCUM" -lt 1 ] && ACCUM=1
echo "auto-tune: VRAM=${VRAM_MB}MB NGPU=$NGPU -> max_num_elements=$MAXEL grad_accum=$ACCUM"

FREEZE=200
SMOKE_OVERRIDES=()
if [ "$SMOKE" = "1" ]; then
  RUN_DIR="$WORKSPACE/runs/smoke"          # never pollutes the real run's resume state
  FREEZE=0
  # publish_metrics must divide validate/checkpoint intervals (fairseq2 check)
  SMOKE_OVERRIDES=(regime.num_steps=20 regime.checkpoint_every_n_steps=10
                   regime.validate_every_n_steps=10
                   regime.publish_metrics_every_n_steps=10)
  echo "SMOKE mode: 20 training steps -> $RUN_DIR"
fi

LAUNCH=(python -m asr_recipe)
[ "${NGPU:-1}" -gt 1 ] && LAUNCH=(torchrun --standalone --nproc_per_node="$NGPU" -m asr_recipe)
echo "launch: ${LAUNCH[*]}"
# Inline overrides go AFTER a single --config flag (fairseq2 recipe CLI);
# passing them bare makes argparse reject them as "unrecognized arguments".
"${LAUNCH[@]}" "$RUN_DIR" \
  --config-file "$CONFIG" \
  --config \
  dataset.mixture_parquet_storage_config.dataset_summary_path="$STATS" \
  dataset.asr_task_config.max_num_elements=$MAXEL \
  trainer.grad_accumulation.num_batches=$ACCUM \
  trainer.freeze_encoder_for_n_steps=$FREEZE \
  "${SMOKE_OVERRIDES[@]}"

banner "DONE"
touch "$WORKSPACE/FINETUNE_DONE"
cat <<EOF
Training finished. Artifacts:
  checkpoints + metrics : $RUN_DIR   (checkpoints live under ws_*/checkpoints/step_*)
  full log              : $LOG

Next:
  - Pick the best checkpoint by validation WER (grep the log for 'wer') --
    it is usually NOT the last step; watch for the plateau.
  - Generate a submission CSV from it:   bash submit_from_checkpoint.sh
  - Copy the chosen checkpoint off-box before you kill the pod/instance.
EOF
