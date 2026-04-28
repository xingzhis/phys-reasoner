"""Inspect multiple parquets' schemas + data_source values from inside container."""
import pyarrow.dataset as ds

paths = [
    ("polaris_dapo", "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/polaris/qwen3-4b-s1_dapo.parquet"),
    ("dapo17k_train", "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/processed_dapo17k_cot/data/train.parquet"),
    ("deepscaler", "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/deepscaler/train.parquet"),
]

for name, p in paths:
    print(f"\n=== {name}: {p} ===")
    try:
        tbl = ds.dataset(p).to_table()
        print(f"  rows={tbl.num_rows}  columns={tbl.column_names}")
        if "data_source" in tbl.column_names:
            vals = set()
            col = tbl.column("data_source").to_pylist()
            for v in col[:100]:
                vals.add(str(v))
            print(f"  data_source (first 100 unique): {sorted(vals)}")
        row = tbl.slice(0, 1).to_pandas().iloc[0]
        for c in tbl.column_names:
            v = row[c]
            s = str(v)
            print(f"  {c} ({type(v).__name__}): {s[:120]}{'...' if len(s)>120 else ''}")
    except Exception as e:
        print(f"  ERROR: {e}")
