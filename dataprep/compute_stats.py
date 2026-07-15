"""Compute the language-distribution stats TSV the training recipe needs.

The mixture-parquet dataset backend temperature-samples languages by their
total hours; this writes the (corpus, language, hours) table it reads.

Usage:  python dataprep/compute_stats.py DATA_ROOT OUT_TSV
"""
import sys

import polars as pl
import pyarrow.dataset as pa_ds


def compute_stats(parquet_dataset_root: str, output_path: str) -> str:
    table = pa_ds.dataset(
        parquet_dataset_root, partitioning="hive", exclude_invalid_files=True
    ).to_table(columns=["language", "corpus", "audio_size"])
    pl_table = pl.from_arrow(table.combine_chunks())
    assert isinstance(pl_table, pl.DataFrame)
    stats = pl_table.group_by(["corpus", "language"]).agg(
        (pl.col("audio_size").sum() / 3600 / 16_000).alias("hours")
    )
    stats.write_csv(output_path, separator="\t")
    return output_path


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    print(f"Computing stats for: {root}")
    print(f"Statistics saved to: {compute_stats(root, out)}")
