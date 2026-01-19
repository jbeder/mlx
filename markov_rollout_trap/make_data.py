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
    # Per-source mixture weight pi_k ~ Uniform(0.1, 0.9)
    pis = np.random.uniform(0.1, 0.9, size=sources)

    # Rows
    src = np.random.randint(0, sources, size=count)
    ids = np.arange(count, dtype=np.int64)

    # Per-row latent physics state: efficiency/drag multiplier z >= 0
    # We model z as LogNormal(mean=0, std=s_row), so E[z] > 1 and multiplicative on c
    # This introduces row-level variability in the transition u -> v.
    s_row = 0.25
    drag_z = np.exp(np.random.normal(0.0, s_row, size=count))

    if mode == "clean":
        upstream_latent = np.random.normal(mus[src], sigmas[src])
        # Transition uses per-source c scaled by per-row latent drag_z
        c_eff = cs[src] * drag_z
        downstream_latent = drag(upstream_latent, c_eff) + np.random.normal(0.0, process_sigma, size=count)
        upstream_obs = upstream_latent
        downstream_obs = downstream_latent
    elif mode == "noisy":
        # Independent sensor errors for upstream and downstream conditional on latents
        # Each stream has its own regime with per-source mixture weight pi(source)
        pi_row = pis[src]

        # Regimes for upstream/downstream are independent Bernoulli(pi_row)
        z_up = (np.random.rand(count) < pi_row).astype(np.int32)  # 1=bad regime, 0=good regime
        z_down = (np.random.rand(count) < pi_row).astype(np.int32)

        # Bad regime is biased and higher-variance (bimodal: +/- B with 50/50), per stream
        B = 2.0
        sign_up = np.where(np.random.rand(count) < 0.5, 1.0, -1.0)
        sign_down = np.where(np.random.rand(count) < 0.5, 1.0, -1.0)
        bias_up = np.where(z_up == 1, sign_up * B, 0.0)
        bias_down = np.where(z_down == 1, sign_down * B, 0.0)

        sigma0 = 0.3
        sigma1 = 1.2
        sensor_sigma_up = np.where(z_up == 0, sigma0, sigma1)
        sensor_sigma_down = np.where(z_down == 0, sigma0, sigma1)

        upstream_latent = np.random.normal(mus[src], sigmas[src])
        # Transition uses per-source c scaled by per-row latent drag_z
        c_eff = cs[src] * drag_z
        downstream_latent = drag(upstream_latent, c_eff) + np.random.normal(0.0, process_sigma, size=count)

        upstream_obs = upstream_latent + bias_up + np.random.normal(0.0, sensor_sigma_up)
        downstream_obs = downstream_latent + bias_down + np.random.normal(0.0, sensor_sigma_down)
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
            # Not for training; keep for analysis.
            "drag_z": drag_z.astype(np.float64),
            "sensor_pi": pis[src].astype(np.float64),
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
