"""
Ingest the 7 hackathon finetuning dataset repos into the omnilingual mixture-parquet
schema (text, audio_bytes, audio_size, language, split, corpus), partitioned by
corpus/split/language -- ready to point a dataset asset card at (see
models/finetune/asr_recipe README / dataprep README "Automatic Dataset Generation").

Schemas below were confirmed by actually loading/listing each repo, not guessed --
see memory `hackathon-hf-dataset-schemas` for how. Three distinct ingestion paths:

Group A: Anv-ke/{kikuyu,Dholuo,Somali,Maasai,Kalenjin}
    Standard `datasets.load_dataset` works fine; has real train/validation/test splits.

Group B: DigitalUmuganda/Afrivoice_Swahili
    `load_dataset` is broken (legacy dataset script, unsupported by datasets>=4.x).
    Bypassed via huggingface_hub.list_repo_files/hf_hub_download: each
    <domain>_swahili_<split>/ folder has audio/audio_N.tar.xz shards paired with
    manifest_N.jsonl (fields include audio_filepath, transcription, locale).

Group C: DigitalUmuganda/Afrivoice, filtered to the Somali/* path (Mogadishu variant)
    Same bypass as Group B (Somali/audio_shards/audio_N.tar.xz + Somali/manifest_N.json,
    which is actually JSONL despite the extension). ~79% of rows have
    transcription=null and are dropped. No source split -- we carve our own
    train/dev/test here.

Usage:
    python ingest_hackathon_datasets.py anvke /path/to/output
    python ingest_hackathon_datasets.py afrivoice_swahili /path/to/output --max_shards=2
    python ingest_hackathon_datasets.py afrivoice_somali /path/to/output --max_shards=2
    python ingest_hackathon_datasets.py all /path/to/output
"""

import io
import json
import random
import tarfile
import time
import uuid
from functools import partial

import fire
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download, list_repo_files

# Group A (anvke) needs fairseq2 (via audio_tools.AudioTableProcessor) + ray;
# Groups B/C only need torch/torchaudio/huggingface_hub, so these are imported
# lazily inside _ingest_anvke_repo/anvke() to keep B/C runnable without fairseq2.

CORPUS = "hackathon_2026"
SAMPLE_RATE = 16_000


# ============================================================
# Group A: Anv-ke -- standard load_dataset path
# ============================================================

ANVKE_REPOS = {
    "Anv-ke/kikuyu": "kik",
    "Anv-ke/Dholuo": "luo",
    "Anv-ke/Somali": "som",
    "Anv-ke/Maasai": "mas",
    "Anv-ke/Kalenjin": "kln",
}

SPLIT_RENAME = {"validation": "dev"}


class _AnvkeTextProcessor:
    """Overrides the source's full-name language column ("Kikuyu", "Dholuo", ...)
    with the hackathon's 3-letter competition code."""

    def __init__(self, competition_code: str):
        self.competition_code = competition_code

    def __call__(self, batch: pa.Table) -> pa.Table:
        # The Anv-ke schema carries its own "split" column; drop it or
        # map_to_target_schema's append_column("split", ...) creates a duplicate
        # column name and the final select() fails. The real split comes from
        # the load_dataset(split=...) argument.
        for col in ("language", "split"):
            if col in batch.column_names:
                batch = batch.drop([col])
        batch = batch.append_column(
            "language",
            pa.array([self.competition_code] * len(batch), type=pa.string()),
        )
        return batch


def _ingest_anvke_repo(
    repo: str,
    competition_code: str,
    output_dir: str,
    skip: int = 0,
    take: int = 0,
    stream: bool = True,
    concurrency: int = 2,
):
    import ray
    from audio_tools import AudioTableProcessor, map_to_target_schema
    from datasets import Audio, load_dataset

    def _load_retry(repo, split, streaming, tries=6):
        """HF connections on some pods are flaky (RemoteDisconnected / connection
        reset mid file-resolution). Retry transient failures with backoff instead
        of aborting the whole ingest on one dropped request."""
        last = None
        for i in range(tries):
            try:
                return load_dataset(repo, split=split, streaming=streaming)
            except Exception as e:
                last = e
                wait = min(60, 5 * (i + 1))
                print(f"[retry] load_dataset({repo}, {split}) failed "
                      f"({type(e).__name__}); attempt {i+1}/{tries}, waiting {wait}s",
                      flush=True)
                time.sleep(wait)
        raise last

    # Two ingest modes:
    #  * stream=True  -> row-by-row over ONE HTTP connection. Good for a small,
    #    disjoint subset (skip/take on a seeded shuffle) in the ingest->train loop,
    #    but ReadHuggingFace is a single serial task => only ~1 CPU, ~1 clip/s.
    #  * stream=False -> download the parquet files in PARALLEL (visible % bars),
    #    then ray.data.from_huggingface() splits them into MANY blocks so
    #    ReadHuggingFace runs as N parallel tasks and saturates all CPUs. This is
    #    the right mode for a FULL ingest -- far faster, and skip/take are ignored
    #    (you're taking everything).
    #
    # skip/take (stream mode only) carve deterministic, DISJOINT chunks out of the
    # seeded shuffle -- that's what lets the loop add fresh data each round without
    # re-downloading or duplicating. When resuming a later round (skip>0), only
    # train grows; dev/test stay fixed so WER stays comparable.
    splits = ["train", "validation", "test"] if skip == 0 else ["train"]
    for split in splits:
        if stream:
            hf_ds = _load_retry(repo, split=split, streaming=True)
            # Keep audio as a raw {"bytes", "path"} struct instead of letting
            # `datasets` decode it (needs torchcodec and is wasted work --
            # AudioTableProcessor decodes from audio.bytes itself).
            hf_ds = hf_ds.cast_column("audio", Audio(decode=False))
            if split == "train":
                # Seeded shuffle ONLY on train: it's the disjointness mechanism
                # for skip/take across rounds. Buffer capped at 1000 -- these are
                # long recordings (~10MB each), so a 2000-item buffer is ~20GB of
                # RAM held in the HF layer ON TOP of Ray's read block. 1000 keeps
                # good randomness at half the memory, and scales down for smoke.
                buf = min(1000, max(2 * take, 100)) if take else 1000
                hf_ds = hf_ds.shuffle(seed=123, buffer_size=buf)
                if skip:
                    hf_ds = hf_ds.skip(skip)
                if take:
                    hf_ds = hf_ds.take(take)
            elif take:
                # Eval splits: NO shuffle (a shuffle buffer would download 2000
                # rows to yield 500). First-N is fine for a fixed dev/test.
                hf_ds = hf_ds.take(min(take, 500))
        else:
            # Parallel download of the whole split -> local arrow. Progress bars
            # print automatically. ray then reads it as many parallel blocks.
            # WARNING: the 5 Anv-ke repos total ~848GB -- this does NOT fit a
            # 500GB volume. Only use stream=False with plenty of disk.
            hf_ds = _load_retry(repo, split=split, streaming=False)
            hf_ds = hf_ds.cast_column("audio", Audio(decode=False))
            print(f"[anvke] {repo} split={split}: {hf_ds.num_rows} rows downloaded, encoding...")
        ray_ds = ray.data.from_huggingface(hf_ds)

        ray_ds = ray_ds.map_batches(
            _AnvkeTextProcessor,
            fn_constructor_kwargs={"competition_code": competition_code},
            batch_size=100,
            batch_format="pyarrow",
        )
        ray_ds = ray_ds.map_batches(
            AudioTableProcessor,
            fn_constructor_kwargs={"audio_column": "audio.bytes", "audio_format": "ogg"},
            batch_size=50,
            batch_format="pyarrow",
            concurrency=concurrency,  # actors x nb_threads(10) decode/encode threads
        )
        out_split = SPLIT_RENAME.get(split, split)
        ray_ds = ray_ds.map_batches(
            partial(map_to_target_schema, split=out_split, corpus=CORPUS),
            batch_size=100,
            batch_format="pyarrow",
        )
        ray_ds.write_parquet(
            output_dir,
            partition_cols=["corpus", "split", "language"],
            min_rows_per_file=1_000,
            row_group_size=100,
        )


def _ray_init_capped():
    """ray.init with optional caps so SEVERAL ingest processes can coexist on
    one pod (each Ray head otherwise grabs all CPUs and reserves ~30% of RAM
    for its object store -- 5 heads would over-reserve both).

    Env: INGEST_RAY_CPUS (int), INGEST_RAY_OBJ_GB (float)."""
    import os

    import ray

    if ray.is_initialized():
        return
    kwargs = {}
    if os.environ.get("INGEST_RAY_CPUS"):
        kwargs["num_cpus"] = int(os.environ["INGEST_RAY_CPUS"])
    if os.environ.get("INGEST_RAY_OBJ_GB"):
        kwargs["object_store_memory"] = int(float(os.environ["INGEST_RAY_OBJ_GB"]) * 1e9)
    ray.init(**kwargs)
    # Tolerate a handful of undecodable blocks instead of aborting a multi-
    # thousand-clip ingest over one corrupt clip (belt-and-suspenders on top of
    # the per-clip guards in audio_tools._post_process / _wav_to_bytes).
    try:
        import ray.data

        ray.data.DataContext.get_current().max_errored_blocks = 50
    except Exception:
        pass


def anvke(
    output_dir: str,
    skip: int = 0,
    take: int = 0,
    stream: bool = True,
    concurrency: int = 2,
):
    """Ingest all 5 Anv-ke repos (Group A), SEQUENTIALLY. Needs fairseq2 + ray.

    For speed, prefer 5 parallel `anvke_one` processes (one per repo) -- each
    HF stream is a single serial HTTP connection, so cross-repo parallelism is
    the only real streaming speedup. finetune.sh does this.

    stream=True  (default): streamed subset via skip/take.
    stream=False           : FULL parallel download -- ~848GB for all 5 repos;
                             needs way more than a 500GB volume. skip/take ignored.

    skip/take (stream mode) carve a chunk out of the seeded-shuffle stream:
    round 1 = --take=2000, round 2 = --skip=2000 --take=2000, ...
    """
    import ray

    _ray_init_capped()
    failed = []
    try:
        for repo, code in ANVKE_REPOS.items():
            print(
                f"[anvke] ingesting {repo} -> language={code} "
                f"(stream={stream}, skip={skip}, take={take}, concurrency={concurrency})"
            )
            try:
                _ingest_anvke_repo(
                    repo, code, output_dir,
                    skip=skip, take=take, stream=stream, concurrency=concurrency,
                )
            except Exception as e:
                # On a flaky-network pod, don't let one repo abort all 5 -- record
                # it and continue so the other languages still ingest. Re-run anvke
                # later to pick up the failures (finetune.sh skips if data exists,
                # so use anvke_one for the missing repo).
                print(f"WARN: {repo} FAILED after retries: {type(e).__name__}: {e} "
                      f"-- continuing with the other languages", flush=True)
                failed.append(repo)
    finally:
        if ray.is_initialized():
            ray.shutdown()
    if failed:
        print(f"\n[anvke] DONE with FAILURES: {failed}\n"
              f"  re-ingest each with:  python dataprep/ingest_hackathon_datasets.py "
              f"anvke_one <name> {output_dir} --take={take}", flush=True)


def anvke_one(
    repo: str,
    output_dir: str,
    skip: int = 0,
    take: int = 0,
    stream: bool = True,
    concurrency: int = 2,
):
    """Ingest ONE Anv-ke repo. `repo` is the full id ("Anv-ke/kikuyu") or just
    the short name ("kikuyu", case-insensitive). Run 5 of these as parallel
    processes (with INGEST_RAY_CPUS/INGEST_RAY_OBJ_GB set) for ~5x ingest speed:

        for r in kikuyu Dholuo Somali Maasai Kalenjin; do
          INGEST_RAY_CPUS=5 INGEST_RAY_OBJ_GB=2 \\
            python ingest_hackathon_datasets.py anvke_one "$r" OUT --take=2000 &
        done; wait
    """
    import ray

    matches = {k: v for k, v in ANVKE_REPOS.items()
               if k.lower() == repo.lower() or k.split("/")[1].lower() == repo.lower()}
    if not matches:
        raise ValueError(f"unknown Anv-ke repo {repo!r}; choose from {list(ANVKE_REPOS)}")
    ((full_repo, code),) = matches.items()

    _ray_init_capped()
    try:
        print(
            f"[anvke_one] ingesting {full_repo} -> language={code} "
            f"(stream={stream}, skip={skip}, take={take}, concurrency={concurrency})"
        )
        _ingest_anvke_repo(
            full_repo, code, output_dir,
            skip=skip, take=take, stream=stream, concurrency=concurrency,
        )
    finally:
        if ray.is_initialized():
            ray.shutdown()


# ============================================================
# Groups B & C: manifest + tar-shard bypass (load_dataset unusable)
# ============================================================


def _decode_audio_bytes(raw_bytes: bytes, suffix: str) -> torch.Tensor:
    """torchaudio auto-picks a backend here (unlike audio_tools.py's
    backend="soundfile") because .webm needs ffmpeg, which soundfile can't do."""
    wav, sr = torchaudio.load(io.BytesIO(raw_bytes), format=suffix.lstrip("."))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def _wav_to_flac_int8_bytes(wav: torch.Tensor) -> np.ndarray:
    """Encode to FLAC via soundfile (libsndfile). Bypasses torchaudio.save():
    on torchaudio 2.9 the ffmpeg/libvorbis ogg encoder rejects the s16 tensor
    ("libvorbis does not support s16 ... Supported values are; fltp"), and the
    torchcodec path can't write BytesIO. FLAC via libsndfile is stable, takes a
    float32 mono 1-D array, and fairseq2's AudioDecoder reads it downstream just
    like ogg. int8 VIEW (not a value cast) matches the omnilingual audio_bytes
    schema -- see dataprep README "list_(pa.int8()) instead of binary()"."""
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)  # mono, 1-D
    if wav.size == 0 or not np.isfinite(wav).all():
        raise ValueError("empty or non-finite waveform")
    buffer = io.BytesIO()
    sf.write(buffer, wav, SAMPLE_RATE, format="FLAC")
    return np.frombuffer(buffer.getvalue(), dtype=np.int8)


def _wav_to_arrow_row(wav: torch.Tensor, text: str, language: str, split: str) -> dict:
    audio_bytes: np.ndarray = _wav_to_flac_int8_bytes(wav)
    return {
        "text": text,
        "audio_bytes": audio_bytes,
        "audio_size": wav.shape[-1],
        "language": language,
        "split": split,
        "corpus": CORPUS,
    }


def _write_rows(rows: list, output_dir: str):
    if not rows:
        return
    table = pa.table(
        {
            "text": pa.array([r["text"] for r in rows], type=pa.string()),
            "audio_bytes": pa.array(
                [r["audio_bytes"] for r in rows], type=pa.list_(pa.int8())
            ),
            "audio_size": pa.array([r["audio_size"] for r in rows], type=pa.int64()),
            "language": pa.array([r["language"] for r in rows], type=pa.string()),
            "split": pa.array([r["split"] for r in rows], type=pa.string()),
            "corpus": pa.array([r["corpus"] for r in rows], type=pa.string()),
        }
    )
    pa_ds.write_dataset(
        table,
        output_dir,
        format="parquet",
        partitioning=["corpus", "split", "language"],
        partitioning_flavor="hive",
        existing_data_behavior="overwrite_or_ignore",
        # Unique basename per call: with the default "part-{i}" template, every
        # shard's write would land on part-0.parquet in the same partition dir
        # and silently overwrite the previous shard's rows.
        basename_template=f"part-{uuid.uuid4().hex}-{{i}}.parquet",
        # The recipe's parquet reader streams per row group and the repo writes
        # 100-row groups everywhere else (see dataprep README) -- match it, or
        # one giant row group per file blows up memory during training.
        max_rows_per_group=100,
        min_rows_per_group=100,
    )


def afrivoice_swahili(
    output_dir: str,
    cache_dir: str | None = None,
    max_shards: int = 0,
    max_clips_per_shard: int = 0,
):
    """Group B: DigitalUmuganda/Afrivoice_Swahili.

    max_clips_per_shard caps the SERIAL webm decode per shard (0 = whole shard).
    Without it a single shard is thousands of clips decoded one-at-a-time (~ms
    each) with no output until done -- looks hung and dominates ingest time."""
    repo = "DigitalUmuganda/Afrivoice_Swahili"
    domains = ["agriculture", "education", "financial", "government", "health"]
    splits = ["train", "dev", "test"]
    all_files = list_repo_files(repo, repo_type="dataset")

    total = 0
    shards_done = 0
    for domain in domains:
        for split in splits:
            prefix = f"{domain}_swahili_{split}"
            manifest_files = sorted(
                f for f in all_files if f.startswith(f"{prefix}/manifest_")
            )
            for manifest_path in manifest_files:
                if max_shards and shards_done >= max_shards:
                    print(f"[afrivoice_swahili] max_shards={max_shards} reached, stopping (partial run)")
                    print(f"[afrivoice_swahili] total labeled clips written: {total}")
                    return
                shard_idx = manifest_path.split("manifest_")[1].split(".")[0]
                audio_shard_path = f"{prefix}/audio/audio_{shard_idx}.tar.xz"

                local_manifest = hf_hub_download(repo, manifest_path, repo_type="dataset", cache_dir=cache_dir)
                with open(local_manifest) as f:
                    entries = [json.loads(line) for line in f if line.strip()]
                entries = [e for e in entries if e.get("transcription")]
                if not entries:
                    shards_done += 1
                    continue

                local_shard = hf_hub_download(repo, audio_shard_path, repo_type="dataset", cache_dir=cache_dir)
                rows = []
                with tarfile.open(local_shard, mode="r:xz") as tar:
                    members = {m.name.split("/")[-1]: m for m in tar.getmembers()}
                    for entry in entries:
                        member = members.get(entry["audio_filepath"])
                        if member is None:
                            continue
                        if max_clips_per_shard and len(rows) >= max_clips_per_shard:
                            break
                        raw = tar.extractfile(member).read()
                        try:
                            wav = _decode_audio_bytes(raw, suffix=".webm")
                            row = _wav_to_arrow_row(wav, entry["transcription"], "swa", split)
                        except Exception as e:
                            print(f"  skip {entry['audio_filepath']}: {e}")
                            continue
                        rows.append(row)
                print(f"[afrivoice_swahili] {prefix} shard {shard_idx}: {len(rows)} labeled clips")
                _write_rows(rows, output_dir)
                total += len(rows)
                shards_done += 1

    print(f"[afrivoice_swahili] done. total labeled clips written: {total}")


def afrivoice_somali(
    output_dir: str,
    cache_dir: str | None = None,
    max_shards: int = 0,
    max_clips_per_shard: int = 0,
    dev_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 123,
):
    """Group C: DigitalUmuganda/Afrivoice, Somali/* path only.
    No source split -- carved into train/dev/test here.
    max_clips_per_shard caps the serial decode per shard (0 = whole shard)."""
    repo = "DigitalUmuganda/Afrivoice"
    rng = random.Random(seed)

    manifest_files = sorted(
        f for f in list_repo_files(repo, repo_type="dataset")
        if f.startswith("Somali/manifest_")
    )

    total = 0
    for i, manifest_path in enumerate(manifest_files):
        if max_shards and i >= max_shards:
            print(f"[afrivoice_somali] max_shards={max_shards} reached, stopping (partial run)")
            break
        shard_idx = manifest_path.split("manifest_")[1].split(".")[0]
        audio_shard_path = f"Somali/audio_shards/audio_{shard_idx}.tar.xz"

        local_manifest = hf_hub_download(repo, manifest_path, repo_type="dataset", cache_dir=cache_dir)
        with open(local_manifest) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        entries = [e for e in entries if e.get("transcription")]
        if not entries:
            continue

        local_shard = hf_hub_download(repo, audio_shard_path, repo_type="dataset", cache_dir=cache_dir)
        rows = []
        with tarfile.open(local_shard, mode="r:xz") as tar:
            members = {m.name.split("/")[-1]: m for m in tar.getmembers()}
            for entry in entries:
                if max_clips_per_shard and len(rows) >= max_clips_per_shard:
                    break
                member = members.get(entry["audio_filepath"])
                if member is None:
                    continue
                raw = tar.extractfile(member).read()
                r = rng.random()
                split = "test" if r < test_frac else ("dev" if r < test_frac + dev_frac else "train")
                try:
                    wav = _decode_audio_bytes(raw, suffix=".wav")
                    row = _wav_to_arrow_row(wav, entry["transcription"], "som", split)
                except Exception as e:
                    print(f"  skip {entry['audio_filepath']}: {e}")
                    continue
                rows.append(row)
        print(f"[afrivoice_somali] shard {shard_idx}: {len(rows)} labeled clips")
        _write_rows(rows, output_dir)
        total += len(rows)

    print(f"[afrivoice_somali] done. total labeled clips written: {total}")


def all(output_dir: str, cache_dir: str | None = None):
    anvke(output_dir)
    afrivoice_swahili(output_dir, cache_dir=cache_dir)
    afrivoice_somali(output_dir, cache_dir=cache_dir)


if __name__ == "__main__":
    fire.Fire(
        {
            "anvke": anvke,
            "anvke_one": anvke_one,
            "afrivoice_swahili": afrivoice_swahili,
            "afrivoice_somali": afrivoice_somali,
            "all": all,
        }
    )
