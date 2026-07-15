#!/usr/bin/env bash
# =============================================================================
# Fresh GPU-box bootstrap (RunPod / EC2 / any Ubuntu+CUDA container).
# Idempotent -- safe to re-run. Captures every environment fix this pipeline
# hit the hard way:
#   * install omnilingual-asr[data] (brings fairseq2, ray, datasets, polars...)
#     + torchcodec, soundfile, fire, hf_transfer, tensorboard, kaggle
#   * RE-PIN huggingface_hub <1.0  -- a stray upgrade pulls hub 1.x, which
#     transformers (imported by ray.data.from_huggingface) rejects
#   * install the v1/v2 asset cards + v2 arch registrations that the PyPI
#     0.1.0 release of omnilingual-asr doesn't ship
#   * put pip/tmp/HF caches on the big volume so the small root FS doesn't
#     hit "disk quota exceeded" mid-download
#   * verify the WHOLE import chain (this is what proves the box works)
#
# USAGE (on the fresh box, inside tmux):
#   bash setup_pod.sh
# then follow the printed next steps.
# =============================================================================
set -uo pipefail   # NOT -e: we want the verify block to run even if a pip warns

WORKSPACE="${WORKSPACE:-/workspace}"
export TMPDIR="$WORKSPACE/tmp" PIP_CACHE_DIR="$WORKSPACE/pipcache" HF_HOME="$WORKSPACE/hf_home"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME"

echo "== 1/3  install omnilingual-asr + deps (few min) =="
pip install "omnilingual-asr[data]" torchcodec soundfile fire hf_transfer tensorboard kaggle

echo "== 2/3  re-pin huggingface_hub <1.0 (the transformers/ray conflict) =="
pip install "huggingface_hub>=0.34,<1.0"

echo "== 2b/3  install v1+v2 asset cards (PyPI 0.1.0 ships neither fully) =="
CARD_DIR=$(python -c "import omnilingual_asr,os;print(os.path.join(os.path.dirname(omnilingual_asr.__file__),'cards','models'))")
# v2 cards (models + written_v2 tokenizer + shared W2V cards):
curl -fsSL "https://raw.githubusercontent.com/facebookresearch/omnilingual-asr/main/src/omnilingual_asr/cards/models/rc_models_v2.yaml" \
  -o "$CARD_DIR/rc_models_v2.yaml" && echo "installed rc_models_v2.yaml -> $CARD_DIR"
# v1 SUPPLEMENT ONLY -- NOT the full rc_models_v1.yaml, which duplicates the
# shared W2V cards already provided by rc_models_v2.yaml ("same name ... W2V_300M").
cat > "$CARD_DIR/z_hackathon_v1.yaml" <<'YAML'
name: omniASR_tokenizer_v1
tokenizer_family: char_tokenizer
tokenizer: https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model
YAML
# add the v1 CTC-300M model card only if the package doesn't already ship it
if ! python -m fairseq2.assets list --kind model 2>/dev/null | grep -qw omniASR_CTC_300M; then
cat >> "$CARD_DIR/z_hackathon_v1.yaml" <<'YAML'

---

name: omniASR_CTC_300M
model_family: wav2vec2_asr
model_arch: 300m
checkpoint: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M.pt
tokenizer_ref: omniASR_tokenizer_v1
YAML
fi
echo "installed v1 supplement (tokenizer_v1 [+ CTC_300M if missing]) -> $CARD_DIR"
# ...and the v2 ARCH registrations (300m_v2 etc.): 0.1.0's config.py only has
# v1 archs; git-main's is byte-identical except the appended v2 blocks
# (verified by diff -- imports and registrar identical), so it's a safe drop-in.
PKG=$(python -c "import omnilingual_asr,os;print(os.path.dirname(omnilingual_asr.__file__))")
curl -fsSL https://raw.githubusercontent.com/facebookresearch/omnilingual-asr/main/src/omnilingual_asr/models/wav2vec2_asr/config.py \
  -o "$PKG/models/wav2vec2_asr/config.py" && echo "installed v2-arch config.py -> $PKG/models/wav2vec2_asr/"

echo "== 3/3  verify the full import chain =="
python - <<'PY'
import sys
ok = True
def check(label, fn):
    global ok
    try:
        fn(); print(f"  OK  {label}")
    except Exception as e:
        ok = False; print(f"  FAIL {label}: {type(e).__name__}: {e}")

import torch
check("torch CUDA", lambda: (_ for _ in ()).throw(RuntimeError("no CUDA")) if not torch.cuda.is_available() else print(f"      {torch.cuda.get_device_name(0)}", end=""))
check("omnilingual_asr", lambda: __import__("omnilingual_asr"))
check("fairseq2", lambda: __import__("fairseq2"))
check("ray.data.from_huggingface", lambda: __import__("ray.data", fromlist=["from_huggingface"]).from_huggingface)
check("soundfile", lambda: __import__("soundfile"))
check("torchaudio", lambda: __import__("torchaudio"))
import huggingface_hub, datasets
print(f"  hub={huggingface_hub.__version__}  datasets={datasets.__version__}")
sys.exit(0 if ok else 1)
PY
VERIFY=$?

echo
if [ "$VERIFY" -eq 0 ]; then
  cat <<EOF
ENV OK. Next:
  export HF_TOKEN=hf_...                       # your gated-repo token
  cd "\$(dirname "\$0")"
  SMOKE=1 bash finetune.sh                     # ~10 min end-to-end validation
  # then the real run:  rm -rf $WORKSPACE/data/hackathon_asr && bash finetune.sh
EOF
else
  cat <<EOF
ENV HAS A FAILURE above. Most likely fixes:
  * torchaudio/torch/CUDA mismatch (libcudart error): reinstall torchaudio matched --
      python - <<'EOF2'
import subprocess, torch
v=torch.__version__.split("+")[0]; cu=(torch.version.cuda or "").replace(".","")
subprocess.run(["pip","install","--force-reinstall","--no-deps",f"torchaudio=={v}","--index-url",f"https://download.pytorch.org/whl/cu{cu}"])
EOF2
  * huggingface_hub still 1.x: pip install "huggingface_hub>=0.34,<1.0"
  Re-run this script after.
EOF
fi
