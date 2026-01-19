# Analysis-4: Rollout-Focused Evaluation (code-verified)

This analysis reads the rollout metrics in runs/{gmm,markov,latent}/{clean,noisy}/eval/metrics.json and the README, and cross-checks key hypotheses directly against the code (make_data.py, model.py, eval.py, train.py). The major change since prior analyses is that all primary metrics are rollout-based only (no teacher-forcing), per README and eval.py.

References consulted

- Metrics: six JSONs under markov_rollout_trap/runs/.../eval/metrics.json
- Code: make_data.py, model.py, eval.py, train.py, config.yaml
- Specs: README (rollout protocol and metric definitions)
- Prior write-ups: ANALYSIS.md, ANALYSIS-1.md, ANALYSIS-2.md, ANALYSIS-3.md

Rollout metrics (headline numbers)

- Common meta: num_rows = 10,000; num_samples = 128
- Clean
  - gmm: crps_up = 0.3954, crps_down = 0.1960, corr_err_by_source = 0.00116, energy_2d ≈ -2.22e-4, mean_mae_by_source = 0.752, std_logerr_by_source = 1.076
  - markov: crps_up = 0.3954, crps_down = 0.1960, corr_err_by_source = 0.00085, energy_2d ≈ -1.96e-4, mean_mae_by_source = 0.752, std_logerr_by_source = 1.079
  - latent: crps_up = 0.3952, crps_down = 0.2194, corr_err_by_source = 0.985, energy_2d ≈ 0.0157, mean_mae_by_source = 0.751, std_logerr_by_source = 1.191
- Noisy
  - gmm: crps_up = 0.4054, crps_down = 0.2139, corr_err_by_source = 0.00743, energy_2d ≈ -2.10e-4, mean_mae_by_source = 0.741, std_logerr_by_source = 0.983
  - markov: crps_up = 0.4048, crps_down = 0.2139, corr_err_by_source = 0.00724, energy_2d ≈ 2.13e-4, mean_mae_by_source = 0.740, std_logerr_by_source = 0.988
  - latent: crps_up = 0.4059, crps_down = 0.2328, corr_err_by_source = 0.839, energy_2d ≈ 0.0124, mean_mae_by_source = 0.740, std_logerr_by_source = 1.091

All metrics above are rollout-only and computed as in eval.py::_aggregate_rollout_metrics (confirmed):
- CRPS via Monte Carlo on rollout samples against observations (_crps_mc)
- Per-source mean and std errors using all rollout samples (d_s flattened across S) vs observed per-source aggregates
- Correlation error per source using one rollout sample per row
- Joint 2D energy distance using one rollout sample per row

What we expected (from README)

- Clean: gmm and markov should do well. latent (with explicit process and sensor model) should be competitive.
- Noisy: latent should benefit from separating process and measurement noise, potentially outperforming on downstream rollout fidelity. gmm/markov may degrade somewhat since they do not explicitly separate sensor noise.

What actually happened (from rollout metrics)

1) gmm and markov are nearly identical and overall strong on global rollout fidelity
- Very small energy distance (≈ ±2e-4) indicates the pooled 2D cloud of rollout pairs (u_hat, d_hat) closely matches the pooled data cloud.
- CRPS_downstream increases from clean (~0.196) to noisy (~0.214), a modest degradation consistent with noisier data.
- corr_err_by_source is tiny on clean (~1e-3) and small on noisy (~7e-3), indicating within-source correlation is captured reasonably well.
- However, two per-source distribution metrics are large for both models: downstream_mean_mae_by_source ≈ 0.74–0.75 and downstream_std_logerr_by_source ≈ 0.98–1.08.

2) latent underperforms on multiple rollout axes
- CRPS_downstream is clearly worse than gmm/markov on both clean (0.219 vs 0.196) and noisy (0.233 vs 0.214).
- corr_err_by_source is extremely large (≈0.98 clean; ≈0.84 noisy), indicating the sampled upstream/downstream correlation within a source is far from the data’s.
- energy_2d is notably larger and positive (~1.2e-2 to 1.6e-2), signaling a perceptible mismatch in the joint 2D distribution compared to gmm/markov.
- Per-source mean and std errors are also large (similar scale to gmm/markov), so latent does not recover per-source aggregates either.

3) Clean vs noisy trends are consistent
- All models show slightly worse CRPS_downstream and corr_err_by_source on noisy vs clean, as expected.
- The per-source std log-error decreases for gmm/markov moving to noisy (from ~1.07→~0.98), consistent with larger within-source empirical std in the noisier data (bringing sd_roll/sd_data closer to 1 in log space if sd_roll is roughly pooled-like).

Reconciling an apparent contradiction: near-zero energy vs large per-source errors (gmm/markov)

- The pooled 2D energy distance being near zero, while per-source mean and std errors are large (~0.74 MAE in means, ~1.0 in |log std ratio|), suggests the models’ rollouts match the overall (pooled across sources) joint distribution very well but do not specialize by source.
  - Intuition: If a model effectively ignores the source during rollout and samples from a (nearly) source-agnostic pooled distribution that matches the global data mix, global metrics (like energy across all rows) can look excellent while per-source aggregates differ substantially from the true per-source distributions.
  - This is consistent with the numbers: in normalized units (make_data.py z-scores upstream and downstream across the dataset, not per source), per-source downstream means typically differ from the global mean by O(0.5–1.0). If the model’s rollout mean for every source stays near the global mean, the average |mu_roll[k] - mu_data[k]| will be ~0.7–0.8, matching the observed ~0.74–0.75.
  - Similarly, within-source empirical std in the data is often < 1 (because the dataset-level standardization inflates variation when sources are pooled). If the model outputs a pooled-like std ~1 for every source, |log(sd_roll/sd_data)| averages near |log(1 / <1)| ≈ 1, matching the ~0.98–1.08 reported.

Why would gmm/markov become source-agnostic in rollout?

- Code check (model.py):
  - JointGaussianModel.sample(source): ctx = Embedding(source); GMM outputs a 2D Gaussian mixture conditioned on ctx (has full 2×2 covariance). It can specialize by source, but it does not have to.
  - MarkovModel.sample(source): samples u ~ p(u|source) then d ~ p(d|source, u) using GMM heads conditioned on [Embedding(source), u]. It can specialize by source and upstream, but again, it does not have to.
- Training (train.py): both are trained with source as input (teacher-forced for markov), so the capacity to learn per-source differences is present. There is no explicit regularizer enforcing source invariance.
- Given the metrics, the most parsimonious explanation is that both models learned parameters that closely match the pooled distribution and under-utilize the source embedding for per-source shifts. This fits all four facts: (1) near-zero pooled energy, (2) small CRPS, (3) large per-source mean and std errors, (4) very similar numbers for gmm and markov across both modes.

Why does latent have such large correlation error and higher energy distance?

- Code check (model.py::LatentModel.sample):
  - u is sampled from p(u|source); v from p(v|source,u) via a GMM conditioned on [Embedding(source), u]. Then observed upstream/downstream are produced by adding independent Gaussian noise with a shared regime k per row.
- The very large corr_err_by_source implies rho_roll[k] is far from rho_data[k]. A plausible, code-consistent cause is that the learned p(v | source, u) underutilizes the dependence on u, rendering v nearly independent of u in rollout. With independent sensor noises, this drives low upstream–downstream correlation within each source.
  - This is further supported by the larger energy_2d vs gmm/markov: if the latent rollout captures less of the upstream→downstream coupling, the 2D cloud becomes less aligned with data’s curved/paired structure.
- Additionally, latent’s sensor noise sigmas are global scalars shared across sources (sensor_log_std is a learned 2-vector), with only the mixing weight π(source) varying by source. This limits per-source flexibility compared to gmm/markov heads and can exacerbate the mismatch.

Clean vs noisy expectations vs results (recap)

- Expected: latent to benefit on noisy data due to explicit measurement modeling.
- Observed (rollout): gmm/markov remain best on global match (energy) and CRPS; latent is worse on both clean and noisy, and much worse on within-source correlation. The models’ relative ranking under rollout-only metrics differs from earlier teacher-forced conclusions in ANALYSIS-2/3.

Code confirmations (no speculation without checks)

- Rollout-only metrics: Verified in eval.py that metrics are computed strictly from model.sample(source) rollouts (no conditioning on observed upstream), matching the README’s stated protocol.
- Per-source metrics: Verified definitions in eval.py (downstream_mean_mae_by_source, downstream_std_logerr_by_source) use observed downstream by source vs rollout downstream by source, with samples flattened across S. The magnitude patterns above are consistent with models matching pooled distributions but not per-source distributions.
- Data normalization: make_data.py standardizes upstream and downstream per dataset (global), not per source; hence per-source means and stds differ from 0 and 1, making per-source discrepancies meaningful.

Answers to the three questions

1) How did the models do in each case?
- gmm and markov: Very good pooled 2D match (energy ~ 0), modest CRPS, low correlation error; but large per-source mean and std errors suggest they do not reproduce per-source distributions in rollout.
- latent: Worse than gmm/markov on CRPS and energy, and dramatically worse on within-source correlation; also exhibits large per-source mean/std errors.

2) How did we expect the models to do?
- Clean: gmm/markov competitive; latent plausible competitor. Noisy: latent expected to gain via explicit sensor-noise modeling and process separation.

3) What might cause the discrepancy?
- For gmm/markov: The models appear to have learned (or default to using) a near source-agnostic rollout distribution that matches the pooled data very well, which keeps global metrics strong and CRPS modest, but fails per-source aggregate checks. This is consistent with their architectures allowing—but not requiring—source specialization.
- For latent: The learned p(v | source, u) likely underutilizes u, breaking the within-source coupling and raising correlation error and energy distance; global sensor sigmas (shared across sources) further limit per-source fidelity.

Next steps (if we want per-source rollout fidelity)

- Diagnostics: Report per-source rollout means/stds alongside data aggregates to confirm the source-agnostic pattern directly (the metrics already strongly imply it).
- Encourage source specialization: increase emb_dim or mixture components; add mild regularization pushing distinct source embeddings; or train with per-source calibration loss terms (e.g., penalties on |mu_roll[k]-mu_data[k]| and |log sd_roll[k]/sd_data[k]|) computed from held-out validation batches.
- For latent: increase capacity of p(v | s, u) and consider conditioning sensor sigmas on source (or learning per-source scalings) to improve within-source correlation and variance alignment.

Notes vs prior analyses

- ANALYSIS-2/3 included teacher-forced metrics and (post-fix) showed very strong downstream fits for gmm/markov. In this rollout-only evaluation, the key difference is that per-source rollout calibration is now explicitly measured and reveals that gmm/markov are matching pooled distributions more than per-source distributions. The latent model’s rollout weaknesses (higher energy, poor correlation) are also clearer here.
