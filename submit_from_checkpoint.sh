#!/usr/bin/env bash
# =============================================================================
# Turn a finetuned checkpoint into a Kaggle submission CSV.
# Run after finetune.sh finishes (or any time a checkpoint exists).
#
#   1. downloads the Kaggle test set (if not already present)
#   2. finds the newest checkpoint in the run dir (fairseq2 ws_*/ layout)
#   3. installs a local asset card pointing at it
#   4. runs greedy inference -> $WORKSPACE/submission_finetuned.csv
#
# USAGE (needs Kaggle creds for step 1):
#   mkdir -p ~/.kaggle && printf '{"username":"U","key":"K"}' > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
#   bash submit_from_checkpoint.sh
#
# Overrides:
#   RUN_DIR=...                        # which training run to submit from
#   CKPT=/path/to/step_N               # skip auto-discovery
#   LIMIT=40                           # smoke test: only 40 clips transcribed
#   MODEL_ARCH=300m_v2 TOKENIZER_REF=omniASR_tokenizer_written_v2   # v2 runs
# =============================================================================
set -euo pipefail
shopt -s globstar nullglob   # `**` must span the anv_test/<lang>/<Scripted|Unscripted>/ nesting

WORKSPACE="${WORKSPACE:-/workspace}"
RUN_DIR="${RUN_DIR:-$WORKSPACE/runs/ctc300m_v1_hackathon}"
TEST_DIR="${TEST_DIR:-$WORKSPACE/data/anv_test}"
OUT="${OUT:-$WORKSPACE/submission_finetuned.csv}"
# Card fields MUST match the checkpoint's version: v1 -> 300m + tokenizer_v1;
# v2 -> 300m_v2 + tokenizer_written_v2. Mixing them decodes garbage.
MODEL_ARCH="${MODEL_ARCH:-300m}"
TOKENIZER_REF="${TOKENIZER_REF:-omniASR_tokenizer_v1}"
FT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$FT_DIR"

# ---- 1. Kaggle test set ----------------------------------------------------
GLOB="$TEST_DIR/**/test_*.parquet"
if ! ls $GLOB >/dev/null 2>&1; then
  echo "downloading Kaggle test set -> $TEST_DIR ..."
  mkdir -p "$TEST_DIR"
  kaggle datasets download -d digitalumuganda/anv-test-data-nt -p "$TEST_DIR" --unzip
fi
ls $GLOB >/dev/null 2>&1 || { echo "no test parquet found at $GLOB"; exit 1; }

# ---- 2. find the checkpoint ------------------------------------------------
CKPT="${CKPT:-}"
if [ -z "$CKPT" ]; then
  # fairseq2 nests checkpoints under a ws_<hash>/ dir and saves them SHARDED
  # (step_N/model/pp_00/tp_00/sdp_00.pt), so the checkpoint is the step_N DIR,
  # not a flat model.pt. Search both the ws_* layout and the legacy flat one.
  STEP_DIR="$(ls -d "$RUN_DIR"/ws_*/checkpoints/step_* "$RUN_DIR"/checkpoints/step_* 2>/dev/null \
              | sort -t_ -k2 -n | tail -1 || true)"
  if [ -z "$STEP_DIR" ]; then
    echo "ERROR: no step_* checkpoints under $RUN_DIR (searched ws_*/checkpoints and checkpoints)"
    find "$RUN_DIR" -maxdepth 4 -type d 2>/dev/null | head -40
    exit 1
  fi
  CKPT="$STEP_DIR"   # fairseq2 native checkpoint = the step_N directory itself
fi
echo "using checkpoint: $CKPT  (arch=$MODEL_ARCH, tokenizer=$TOKENIZER_REF)"

# ---- 3. install a local asset card pointing at it --------------------------
CARD_DIR="$(python -c "import omnilingual_asr, os; print(os.path.join(os.path.dirname(omnilingual_asr.__file__),'cards','models'))")"
cat > "$CARD_DIR/ctc300m_finetuned.yaml" <<YAML
name: ctc300m_finetuned
model_family: wav2vec2_asr
model_arch: $MODEL_ARCH
checkpoint: $CKPT
tokenizer_ref: $TOKENIZER_REF
YAML
echo "installed card 'ctc300m_finetuned' -> $CARD_DIR"

# ---- 4. inference ----------------------------------------------------------
# LIMIT=40 -> smoke test: only 40 clips transcribed (rest backfilled with "...")
python infer_testset.py --card ctc300m_finetuned --glob "$GLOB" --out "$OUT" \
  ${LIMIT:+--limit "$LIMIT"}

echo
echo "SUBMISSION READY -> $OUT"
echo "Optionally clean it further:  python normalize_submission.py $OUT ${OUT%.csv}_normalized.csv"
echo "Then scp it down and upload to Kaggle."
