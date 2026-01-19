# Demonstrates rollout quality with a Markov model and noisy sensors

## Data

python -m markov_rollout_trap.make_data [--mode=clean|noisy] --seed=<seed> --sources=<K> --count=<N> --out=<filename.parquet>

Defaults: seed=42, sources=10, count=10000

Generates N example rows of data as follows:

Schema: id, source, upstream_speed, downstream_speed

Set seed to <seed>.

- For each source 0..K-1, fix a random mu uniformly in [90, 100], sigma uniformly in [2.5, 3.5].
- For each source 0..K-1, fix a hidden drag coefficient c := exp(ξ), where ξ ~ Normal(log(c0), s^2), with c0 = 0.003 and s=0.3.
- Fix a small physics/process noise process_sigma = 0.3.
- In mode=noisy only, sensor noise is a 2-regime mixture per row:
  - Sample z ~ Bernoulli(0.5).
  - If z=0 ("good sensor"), use sensor_sigma = 0.3.
  - If z=1 ("bad sensor"), use sensor_sigma = 1.2.

In mode=clean:

- Generate source (uniform random in 0..K-1)
- Generate upstream_speed ~ Normal(mu, sigma^2) from that source
- Generate downstream_speed by applying a drag of v/(1+c\*v) to upstream_speed (using that source’s c), then adding Normal(0, process_sigma^2)

In mode=noisy

- Generate source (uniform random in 0..K-1)
- Sample z ~ Bernoulli(0.5) and set sensor_sigma as above
- Generate a latent upstream speed ~ Normal(mu, sigma^2) from that source
- Generate upstream_speed by adding noise ~ Normal(0, sensor_sigma^2) to the latent upstream speed
- Generate a latent downstream speed by applying a drag of v/(1+c\*v) on the latent upstream speed (using that source’s c), then adding Normal(0, process_sigma^2)
- Generate downstream_speed by adding noise ~ Normal(0, sensor_sigma^2) to the latent downstream speed

## Model

```
python -m markov_rollout_trap.train \
  --config markov_rollout_trap/config.yaml \
  --model gmm|markov|latent \
  --data <filename.parquet> \
  --out_dir <model_dir> \
  [--seed 42] [--device auto]
```

Train on rows `(source, upstream_speed, downstream_speed)`.

- `source`: conditioning feature (categorical)
- `upstream_speed`: observed upstream sensor value
- `downstream_speed`: observed downstream sensor value

All models should provide:

- `log_prob(source, upstream_speed, downstream_speed)` for evaluation
- `sample(source)` for rollout (returns `(upstream_speed, downstream_speed)`)

### gmm: joint Gaussian (full covariance)

What it models:

- Directly models the joint distribution of `(upstream_speed, downstream_speed)` conditioned on `source` as a single 2D Gaussian.
- Uses a full 2x2 covariance so it can represent correlation between upstream and downstream.

Training:

- Maximum likelihood on the joint `(upstream_speed, downstream_speed)` given `source`.

Rollout:

- Sample `(upstream_speed, downstream_speed)` in one shot from the joint Gaussian.

### markov: teacher-forced chain (observed upstream -> downstream)

What it models:

- A 2-step Markov factorization:
  - `p(upstream_speed | source)` as a 1D diagonal Gaussian
  - `p(downstream_speed | source, upstream_speed)` as a 1D diagonal Gaussian

Training:

- Teacher-forced MLE:
  - maximize log-likelihood of upstream conditioned on source, using the observed upstream_speed
  - maximize log-likelihood of downstream conditioned on source and the observed upstream_speed

Rollout:

1. Sample `upstream_speed` from `p(upstream_speed | source)`
2. Sample `downstream_speed` from `p(downstream_speed | source, sampled_upstream_speed)`

### latent: latent true speeds + sensor noise (separates process vs measurement)

Latents:

- `u`: latent upstream true speed (unobserved)
- `v`: latent downstream true speed (unobserved)

What it models:

- A process model on latents:
  - `p(u | source)` as a 1D diagonal Gaussian
  - `p(v | source, u)` as a 1D diagonal Gaussian
- Two sensor likelihoods that map latent true values to observed readings, as a shared 2-component Gaussian mixture (same component selection for upstream and downstream on a given row):
  - `p(upstream_speed | u)` = π(source) Normal(u, sensor_sigma0^2) + (1-π(source)) Normal(u, sensor_sigma1^2)
  - `p(downstream_speed | v)` = π(source) Normal(v, sensor_sigma0^2) + (1-π(source)) Normal(v, sensor_sigma1^2)

Training:

- Latent-variable maximum likelihood via variational inference (ELBO):
  - introduce an encoder / approximate posterior `q(u, v | source, upstream_speed, downstream_speed)`
  - optimize generative params and encoder params jointly
- Sensor mixture parameters are learned (two sigmas sensor_sigma0, sensor_sigma1, plus a mixing logit for π; conditioned on source).

Rollout:

1. Sample latent `u` from `p(u | source)`
2. Sample latent `v` from `p(v | source, u)`
3. Sample observed `upstream_speed` from `p(upstream_speed | u)`
4. Sample observed `downstream_speed` from `p(downstream_speed | v)`

## Eval

python -m markov_rollout_trap.eval --data=<filename.parquet> --model=<model_dir>

Compute metrics for each `(data_mode, model_config)` pair. **All primary metrics must be computed from rollouts**, i.e. samples produced by `sample(source)` (no conditioning on observed `upstream_speed`). Writes `/eval/metrics.json` to the model directory.

### Rollout sampling protocol

For each row `i` with `source_i`, draw `S` rollout samples:

- `(u_hat[i,s], d_hat[i,s]) ~ sample(source_i)` for `s = 1..S`

All metrics below use only:

- observed `(upstream_speed_i, downstream_speed_i)`
- rollout samples `(u_hat[i,*], d_hat[i,*])`

### Primary rollout metrics

1. **Upstream CRPS (rollout)**
   Mean CRPS of `upstream_speed_i` against samples `{u_hat[i,*]}`.

2. **Downstream CRPS (rollout)** _(the headline)_
   Mean CRPS of `downstream_speed_i` against samples `{d_hat[i,*]}`.

3. **Per-source downstream mean MAE (rollout)**
   For each source `k`:
   - `mu_data[k] = mean(downstream_speed | source=k)`
   - `mu_roll[k] = mean(d_hat | source=k)` using all rollout samples for rows with `source=k`
     Report `mean_k |mu_roll[k] - mu_data[k]|`.

4. **Per-source downstream std log-error (rollout)**
   For each source `k`:
   - `sd_data[k] = std(downstream_speed | source=k)`
   - `sd_roll[k] = std(d_hat | source=k)` (over all rollout samples)
     Report `mean_k |log(sd_roll[k] / sd_data[k])|`.

5. **Within-source correlation error (rollout)**
   For each source `k`, compute Pearson correlation:
   - `rho_data[k] = corr(upstream_speed, downstream_speed | source=k)`
   - `rho_roll[k] = corr(u_hat, d_hat | source=k)` (use one rollout sample per row, e.g. `s=1`, to avoid overweighting rows)
     Report `mean_k |rho_roll[k] - rho_data[k]|`.

6. **Joint energy distance (rollout, 2D)**
   Energy distance between the empirical 2D clouds of observed pairs `(upstream_speed, downstream_speed)` and rollout pairs `(u_hat, d_hat)` (use one rollout sample per row so the clouds have matching size).

### Output schema

Write `metrics.json` as:

- `meta`:
  - `num_samples`
  - `num_rows`
- `rollout`:
  - `crps_upstream`
  - `crps_downstream`
  - `downstream_mean_mae_by_source`
  - `downstream_std_logerr_by_source`
  - `corr_err_by_source`
  - `energy_2d`
