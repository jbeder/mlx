import argparse
import os

import numpy as np
import pandas as pd
import torch


def drag(v: np.ndarray, c: float) -> np.ndarray:
    return v / (1.0 + c * v)


def generate_data(mode: str, seed: int, sources: int, count: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    
    # normalize both upstream and downstream
    upstream_mean = upstream_obs.mean()
    upstream_std = upstream_obs.std()
    downstream_mean = downstream_obs.mean()
    downstream_std = downstream_obs.std()
    upstream_z = (upstream_obs - upstream_mean) / upstream_std
    downstream_z = (downstream_obs - downstream_mean) / downstream_std

    df = pd.DataFrame(
        {
            "id": ids,
            "source": src.astype(np.int32),
            "upstream_speed": upstream_z.astype(np.float64),
            "downstream_speed": downstream_z.astype(np.float64),
        }
    )
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["clean", "noisy"], default="clean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sources", type=int, default=10)
    p.add_argument("--count", type=int, default=10000)
    # New: allow a data directory to be provided; default to package data dir
    default_data_dir = os.path.join(os.path.dirname(__file__), "data")
    p.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,
        help=f"Directory to write parquet to when --out is not provided (default: {default_data_dir})",
    )
    # Make --out optional; if not provided, we write to <data_dir>/<mode>.parquet
    p.add_argument("--out", type=str, default=None, help="Output parquet filename (overrides --data_dir)")
    args = p.parse_args()

    df = generate_data(args.mode, args.seed, args.sources, args.count)
    out_path = args.out or os.path.join(args.data_dir, f"{args.mode}.parquet")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows.")


if __name__ == "__main__":
    main()