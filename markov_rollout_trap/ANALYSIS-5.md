# Analysis-5: Rollout-only evaluation after introducing biased, per-source sensor noise

This pass reads the six rollout metrics JSONs under runs/{gmm,markov,latent}/{clean,noisy}/eval/metrics.json, the README, and the current code (make_data.py, model.py, eval.py, train.py, config.py). It focuses on:

- How each model did (rollout-only metrics)
- How we expected them to do (given the new biased/per-source-weight sensor noise)
- What might explain discrepancies, verified against code (no speculation without code checks)

Key change since prior analyses (verified in code):

- Data generator now uses per-source mixture weights and a biased “bad sensor” regime in noisy mode.
  - make_data.py: for mode="noisy", draws per-source mixture weights pis[k] ~ Uniform(0.1, 0.9), samples z ~ Bernoulli(pi[source]), and when z==1 (bad) adds a bimodal bias b ∈ {+B, -B} with B=2.0, plus higher-variance noise (sigma1=1.2). Good regime uses sigma0=0.3 with zero bias. The same bias and regime apply to both upstream and downstream on a row.
  - This differs from README (which still describes Bernoulli(0.5) and no bias); the README is outdated relative to code.

- Latent model was updated to explicitly match this generator:
  - model.py::LatentModel uses per-source sensor mixing weight via an embedding (sensor_logit), global learned sigmas for good/bad regimes (sensor_log_std), and a learned positive bias magnitude B via softplus. For k=1 (bad), it adds ±B bias with 50/50 sign to both upstream and downstream emissions on a row.

- Evaluation is rollout-only (no teacher forcing):
  - eval.py loads the parquet, calls model.sample(source, S), and computes the rollout metrics defined in README (CRPS up/down, per-source downstream mean MAE and std log-error, within-source correlation error using one rollout sample per row, and 2D joint energy distance). This matches the rollout protocol and definitions in README.

Rollout metrics (from metrics.json)

- Common meta: num_rows = 10,000, num_samples = 128

- Clean
  - gmm: crps_up=0.3940, crps_down=0.1950, corr_err=0.00084, energy_2d≈-2.49e-4, mean_MAE_by_source=0.7520, std_logerr_by_source=1.0802
  - markov: crps_up=0.3953, crps_down=0.1957, corr_err=0.00305, energy_2d≈-9.67e-6, mean_MAE_by_source=0.7514, std_logerr_by_source=1.0804
  - latent: crps_up=0.4357, crps_down=0.3029, corr_err=0.3937, energy_2d≈0.0925, mean_MAE_by_source=0.7508, std_logerr_by_source=1.5623

- Noisy (with per-source π and biased bad regime)
  - gmm: crps_up=0.4228, crps_down=0.2525, corr_err=0.00886, energy_2d≈-2.06e-4, mean_MAE_by_source=0.7149, std_logerr_by_source=0.8114
  - markov: crps_up=0.4230, crps_down=0.2526, corr_err=0.01010, energy_2d≈-2.52e-4, mean_MAE_by_source=0.7140, std_logerr_by_source=0.8150
  - latent: crps_up=0.4605, crps_down=0.3389, corr_err=0.2661, energy_2d≈0.0806, mean_MAE_by_source=0.7138, std_logerr_by_source=1.2654

What we expected (given the new noise)

- The biased, per-source-weighted sensor noise should advantage the latent model, which explicitly separates process (u→v) from measurement with:
  - a shared per-row regime k for both sensors,
  - per-source mixing weights π(source), and
  - a learned bias magnitude B for the bad regime.
- By contrast, gmm and markov do not model the sensor mixture or bias explicitly; we expected their rollout fidelity—especially downstream—to degrade more on noisy data.

What actually happened

1) gmm and markov remain nearly identical and best on pooled rollout fidelity
- CRPS_downstream increases from ~0.195 (clean) to ~0.253 (noisy), a modest degradation consistent with noisier/more complex observed data.
- Joint 2D energy distance is very close to zero (small negative due to finite-sample estimation noise), indicating the pooled 2D distribution of rollout pairs (u_hat, d_hat) closely matches the pooled data cloud.
- Within-source correlation error remains small (clean ~1e-3; noisy ~1e-2), meaning the sampled upstream/downstream relationship is plausible.

2) Per-source distribution metrics are large for all models (gmm/markov especially)
- downstream_mean_MAE_by_source is ~0.75 on clean and ~0.71 on noisy for gmm/markov.
- downstream_std_logerr_by_source is ~1.08 on clean, dropping to ~0.81 on noisy.
- Interpretation (consistent with prior analyses): models match the pooled distribution well but do not calibrate per-source distributions. Global z-scoring (make_data.py) means per-source downstream means deviate from 0 and per-source stds are typically <1; if a model’s rollout stays near global mean and unit variance across sources, the per-source errors land near the observed magnitudes. This matches the metrics pattern and does not require any code bug.

3) Latent underperforms on multiple rollout axes despite modeling the new noise
- CRPS_downstream is notably worse than gmm/markov on both clean (0.303 vs ~0.195) and noisy (0.339 vs ~0.253).
- corr_err_by_source is very large (0.394 clean; 0.266 noisy), indicating poor recovery of within-source upstream–downstream coupling.
- energy_2d is much larger (0.08–0.09 vs ~0), showing a visible mismatch in the pooled 2D joint distribution.
- Per-source mean and std errors are also large for latent (std log-error is the worst of the three), so it does not recover per-source aggregates either.

Why the discrepancy? (code-checked explanations)

1) The generator change is present and the latent model is capable of representing it
- Data: per-source π, biased bad regime, shared across sensors (make_data.py). Confirmed.
- Latent: samples k ~ Bernoulli(π(source)) and adds ±B bias and regime-specific sigmas to both sensors, matching the generator’s structure (model.py::LatentModel.sample and _sensor_joint_log_prob). Confirmed.

2) Yet gmm/markov still win on pooled metrics
- Capacity: Both gmm and markov use GMM heads (2 components by default). A 2-component full-covariance joint Gaussian (gmm) can approximate the mixture structure in normalized space surprisingly well, especially because the ±B bias is symmetric and the per-source π just reweights modes by source—something a source-conditioned GMM can emulate. Markov adds conditional structure but shows nearly identical rollout metrics to gmm here.
- Global standardization: Because upstream and downstream are z-scored across the entire dataset (not per source), pooled structure dominates. Matching the pooled distribution tightly (which gmm/markov do) yields near-zero energy distance and competitive CRPS, even if per-source calibration is poor.

3) Why is latent’s within-source correlation so far off?
- Process dependence may be underutilized: p(v | source, u) is a 1D GMM conditioned on [embedding(source), u]. If the learned dependence on u is weak, the latent rollout will produce v that is only loosely coupled to u, driving down within-source corr(u_hat, d_hat). The large corr_err_by_source values are consistent with this failure mode.
- Encoder factorization: The amortized encoder is fully factorized (q(u|·) q(v|·)). This is explicit in model.py::LatentModel._encode: it emits [mu_u, log_std_u, mu_v, log_std_v] and constructs independent Normals for q_u and q_v. While common, a factorized q can make it harder to learn strong u↔v dependencies in the generative conditional p(v|u), especially when measurement noise adds a shared biased component.
- Sensor parameterization: The latent model uses per-source π via an embedding (matches the generator), but sigmas are global scalars across sources. That matches the generator as well (sigma0=0.3, sigma1=1.2 globally), so this is unlikely the limiting factor; the dominant miss appears to be the weak u→v mapping learned under the current training setup.

4) Interpreting the per-source metrics trend (clean→noisy)
- The drop in std_logerr_by_source from ~1.08 (clean) to ~0.81 (noisy) for gmm/markov is expected if per-source empirical std increases under noisier sensors (bringing sd_roll ≈ 1 closer to sd_data). Meanwhile, corr_err_by_source and CRPS degrade modestly as noise increases.

Answers to the guiding questions

1) How did the models do?
- gmm and markov are nearly tied and best on pooled rollout fidelity (lowest CRPS_downstream, near-zero energy_2d, low corr_err). All three models have poor per-source calibration (large per-source mean MAE and std log-errors).
- latent is worse on all pooled rollout metrics and exhibits dramatically higher within-source correlation error.

2) How did we expect them to do?
- We expected the biased, per-source-weighted sensor noise to hurt gmm/markov (which do not model the sensor mixture) and favor latent (which does). That did not materialize in these rollout metrics.

3) What might cause the discrepancy? (confirmed/ruled-in by code)
- gmm/markov can approximate the pooled mixture structure sufficiently with source-conditioned GMM heads (2 components) in globally standardized space, keeping pooled metrics strong.
- The latent model’s learned conditional p(v|s,u) appears to underutilize u, producing weak within-source coupling and a poorer joint 2D match, despite correctly modeling the shared biased sensor regime.
- Models broadly ignore per-source calibration in rollout, as indicated by large per-source mean and std errors; this is consistent with global z-scoring and training objectives that reward pooled fit.

Notes vs previous analyses

- Earlier analyses (e.g., Analysis-4) reached similar conclusions about pooled vs per-source fidelity under rollout-only evaluation. This pass confirms those findings under the new generator with biased/per-source-weighted sensor noise. The README’s data description is outdated; the code reflects the new noise and the latent model matches it structurally.

Actionable follow-ups (if we want latent to improve and per-source fidelity to matter)

- Strengthen u→v dependence learning in latent (e.g., richer conditional head or more components for p(v|s,u); consider a less factorized encoder, or KL annealing, to better capture the dependency).
- Encourage per-source calibration during training (e.g., add auxiliary per-source calibration losses measured on validation batches; or increase capacity of source embeddings / mixture components so models don’t collapse to pooled behavior).
- Keep rollout-only metrics (as done here) and consider adding explicit per-source calibration diagnostics to training logs to prevent regressions.

---

Postscript: Are gmm/markov “too good,” or is latent “not good enough” on noisy?

Question

- Goal reminder: demonstrate that gmm/markov cannot match a latent model on the noisy dataset.
- Observation from noisy metrics (S=128, N=10k):
  - gmm: crps_down=0.2525, corr_err=0.0089, energy_2d≈-2.06e-4
  - markov: crps_down=0.2526, corr_err=0.0101, energy_2d≈-2.52e-4
  - latent: crps_down=0.3389, corr_err=0.2661, energy_2d≈0.0806

Answer (based on code and metrics):

- The issue is primarily that the latent model is not good enough on the noisy dataset, not that gmm/markov are “too good.”
  - Code confirms the generator change (per-source π and biased bad regime) and latent implements it correctly (shared k, learned ±B, global sigmas). There is no evidence of evaluation bias favoring gmm/markov: eval.py computes rollout-only metrics uniformly across models.
  - gmm/markov performing well on pooled metrics is plausible: source-conditioned GMMs with 2 components can approximate the global mixture in normalized space, keeping pooled energy/CRPS small. Their within-source correlation errors are also small, indicating they capture the upstream→downstream coupling better than latent under current training.
  - Latent’s weaknesses are specific: very large within-source correlation error and higher energy distance. This points to an underutilized p(v|s,u): the learned conditional doesn’t couple v tightly to u, despite correct sensor-regime modeling. The encoder is factorized (q(u) q(v)), which can make learning a strong dependency harder.

Implication for the demonstration

- To make the intended point (latent > gmm/markov on noisy), we should focus on improving the latent model rather than weakening gmm/markov further. Concretely:
  1) Strengthen p(v|s,u) capacity (more components or a richer conditional head) and consider a less-factorized encoder (e.g., joint q(u,v) or coupling via shared layers), possibly with KL annealing to encourage informative posteriors.
  2) Add training diagnostics/auxiliary losses that reflect the rollout metrics, especially within-source correlation (e.g., minimize |corr_roll[k] − corr_data[k]| on a validation split) and per-source calibration (mean/std) to prevent collapse to pooled behavior.
  3) If necessary, increase S in training/eval for latent (more samples in ELBO/IS) and report effective sample sizes to rule out estimator variance as a culprit.

Net: The current evidence says latent underperforms relative to its capacity, so the path to the demo is improving latent’s conditional/process learning rather than assuming gmm/markov are artificially advantaged.