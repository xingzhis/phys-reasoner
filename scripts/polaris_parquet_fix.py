"""Rewrite POLARIS stage1 parquet to add `data_source='math_dapo'` so verl's
default_compute_score routes correctly (POLARIS's original parquet has empty
data_source, which our verl doesn't know how to route).

Run INSIDE the apptainer container (newer pyarrow handles the parquet's
repetition-level histogram metadata that login-node pyarrow can't read).
"""
from __future__ import annotations
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/polaris/qwen3-4b-s1.parquet"
DST = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/polaris/qwen3-4b-s1_dapo.parquet"

tbl = ds.dataset(SRC).to_table()
print(f"src rows={tbl.num_rows} columns={tbl.column_names}")

# Set every row's data_source to 'math_dapo' so verl's default_compute_score routes
# to verl.utils.reward_score.math_dapo.compute_score (same family as POLARIS's
# internal scorer: last-boxed extraction + sympy-equivalence).
ds_col = pa.array(["math_dapo"] * tbl.num_rows)
idx = tbl.column_names.index("data_source")
new_tbl = tbl.set_column(idx, "data_source", ds_col)
pq.write_table(new_tbl, DST)
print(f"wrote {DST}  rows={new_tbl.num_rows}")
print(f"sample data_source: {new_tbl.column('data_source')[0].as_py()!r}")
