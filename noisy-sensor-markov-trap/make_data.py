#!/usr/bin/env python3
import argparse
import os

import numpy as np

try:
    # Support running as a script
    from .utils import set_seed  # type: ignore
except Exception:
    import sys

    sys.path.append(os.path.dirname(__file__))
    from utils import set_seed  # type: ignore


def drag(v: np.ndarray, c: float) -> np.ndarray:
    return v / (1.0 + c * v)


def generate_data(mode: str, seed: int, sources: int, count: int):
    set_seed(seed)

    # Per-source parameters
    mus = np.random.uniform(90.0, 100.0, size=sources)
    sigmas = np.random.uniform(2.5, 3.5, size=sources)
    c0 = 0.003
    s = 0.3
    cs = np.exp(np.random.normal(np.log(c0), s, size=sources))
    process_sigma = 0.3

    # Rows
    src = np.random.randint(0, sources, size=count)
    ids = np.arange(count, dtype=np.int64)

    if mode == "clean":
        upstream_latent = np.random.normal(mus[src], sigmas[src])
        downstream_latent = drag(upstream_latent, cs[src]) + np.random.normal(0.0, process_sigma, size=count)
        upstream_obs = upstream_latent
        downstream_obs = downstream_latent
    elif mode == "noisy":
        z = (np.random.rand(count) < 0.5).astype(np.int32)
        sensor_sigma = np.where(z == 0, 0.3, 1.2)
        upstream_latent = np.random.normal(mus[src], sigmas[src])
        upstream_obs = upstream_latent + np.random.normal(0.0, sensor_sigma)
        downstream_latent = drag(upstream_latent, cs[src]) + np.random.normal(0.0, process_sigma, size=count)
        downstream_obs = downstream_latent + np.random.normal(0.0, sensor_sigma)
    else:
        raise ValueError("mode must be 'clean' or 'noisy'")

    import pandas as pd

    df = pd.DataFrame(
        {
            "id": ids,
            "source": src.astype(np.int32),
            "upstream_speed": upstream_obs.astype(np.float64),
            "downstream_speed": downstream_obs.astype(np.float64),
        }
    )
    return df


def to_parquet(df, out_path: str):
    # Try PyArrow first
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401

        df.to_parquet(out_path, index=False)
        return
    except Exception as e:
        # Try via duckdb CLI if available
        try:
            import shutil

            if shutil.which("duckdb") is not None:
                tmp_csv = out_path + ".tmp.csv"
                df.to_csv(tmp_csv, index=False)
                os.system(
                    f"duckdb -c \"COPY (SELECT * FROM read_csv_auto('{tmp_csv}')) TO '{out_path}' (FORMAT PARQUET)\""
                )
                os.remove(tmp_csv)
                return
        except Exception:
            pass
        raise RuntimeError("Failed to write parquet. Install pyarrow or duckdb CLI. Original error: %r" % (e,))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["clean", "noisy"], default="clean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sources", type=int, default=10)
    p.add_argument("--count", type=int, default=10000)
    p.add_argument("--out", required=True, help="Output parquet filename")
    args = p.parse_args()

    df = generate_data(args.mode, args.seed, args.sources, args.count)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    to_parquet(df, out_path)
    print(f"Wrote {out_path} with {len(df)} rows.")


if __name__ == "__main__":
    main()
