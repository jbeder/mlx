# Analysis-7: Rollout-only evaluation after adding per-row latent drag and independent sensor regimes

Scope and method

- I read the six rollout metrics under runs/{gmm,markov,latent}/{clean,noisy}/eval/metrics.json, the resolved configs, and the code in make_data.py, model.py, eval.py, train.py, and config.py.
- I verified hypotheses directly in code (no speculation). Notably, the current data generator introduces (a) a per-row latent drag multiplier and (b) independent sensor-noise regimes for upstream vs downstream in the noisy variant. The README’s noisy description (shared regime) is now outdated relative to the code.

What changed since the last analysis (code-verified)

- Per-row latent: make_data.py multiplies each row’s effective drag coefficient by a LogNormal factor drag_z (z ≥ 0), adding row-level variability to the u → v mapping.
- Noise regimes: In mode="noisy", upstream and downstream sensors choose regimes independently per row with per-source mixture weights π(source). Bad regime uses higher σ1 and a ±B bias with 50/50 sign; good regime uses σ0 and zero bias. This is implemented independently for upstream and downstream (z_up, z_down). The README still describes a single shared regime k across both sensors; that is no longer accurate.

Rollout metrics (S=128, N=10k)

- gmm/clean:
  - crps_up=0.3952, crps_down=0.3702, corr_err=0.0236, energy_2d≈-1.71e-4,
    downstream mean MAE by source=0.6040, downstream std log-error by source=0.4331
- gmm/noisy:
  - crps_up=0.4255, crps_down=0.3832, corr_err=0.0230, energy_2d≈-1.96e-4,
    mean MAE by source=0.5847, std log-error by source=0.3972

- markov/clean:
  - crps_up=0.3956, crps_down=0.3694, corr_err=0.0218, energy_2d≈-2.33e-4,
    mean MAE by source=0.6022, std log-error by source=0.4339
- markov/noisy:
  - crps_up=0.4254, crps_down=0.3835, corr_err=0.0167, energy_2d≈-1.33e-4,
    mean MAE by source=0.5831, std log-error by source=0.3933

- latent/clean:
  - crps_up=0.4490, crps_down=0.4286, corr_err=0.0890, energy_2d≈0.0627,
    mean MAE by source=0.6051, std log-error by source=0.8924
- latent/noisy:
  - crps_up=0.4731, crps_down=0.4391, corr_err=0.1492, energy_2d≈0.0591,
    mean MAE by source=0.5860, std log-error by source=0.8548

1) How did the models do?

- gmm and markov are nearly identical across clean and noisy and clearly best on pooled rollout fidelity:
  - Low CRPS (latent is higher on both variables); very small joint 2D energy distance (near 0; tiny negatives are sampling noise in the estimator); and small within-source correlation error (≈0.017–0.024).
  - Per-source calibration is moderate: downstream per-source mean MAE ≈0.58–0.60 and std log-error ≈0.39–0.43.

- latent underperforms on all pooled rollout metrics and per-source std alignment:
  - Higher CRPS (both upstream and downstream), much larger 2D energy distance (~0.06 vs ~0.0002), and substantially larger within-source correlation error (0.089 clean; 0.149 noisy).
  - Per-source downstream std log-error is notably worse (≈0.85–0.89), indicating a ≈2.3× mismatch in per-source spread on average (exp(0.85) ≈ 2.34).
  - Downstream per-source mean MAE is similar to gmm/markov (~0.59–0.61), so the main latent weakness is in shape/coupling rather than the mean alone.

2) How did we expect them to do?

- Clean: Expect gmm and markov to do well; latent to be competitive.
- Noisy (with the new changes meant to “break” gmm/markov): We would hope the latent model—which explicitly separates process vs measurement—to gain an advantage. However, because the current generator uses independent noise regimes for upstream and downstream while the latent model assumes a shared regime across both sensors, we should expect some degradation in latent’s joint fit unless the model is updated to match the new structure.

3) What might cause the discrepancy? (code-verified)

Primary, confirmed mismatch (generator vs latent model):

- Data (make_data.py, noisy mode): upstream and downstream noise regimes are independent Bernoulli(π(source)); each stream has its own ±B bias sign and its own σ (σ0 or σ1). See z_up / z_down and bias_up / bias_down.
- Latent model (model.py::LatentModel): uses a shared regime k ~ Bernoulli(π(source)) for both sensors on the row and a single shared bias b ∈ {+B, -B} added to both upstream and downstream.

Implications:

- With independent regimes in the data, the observed upstream/downstream pair contains less “shared measurement” correlation than a shared-regime model would inject. The latent model’s shared k and shared b couple the two observed sensors more strongly than the data does, mis-shaping the joint distribution and inflating within-source correlation error and energy distance. This exactly matches the metrics: latent’s corr_err_by_source and energy_2d are an order of magnitude worse than gmm/markov.
- The std log-error by source being much larger for latent is also consistent: a shared ±B across both sensors per row concentrates mass into two biased lobes more than the data’s independent per-stream biases, misaligning the per-source downstream spread when aggregating over many rows.

Secondary, verified context:

- Per-row latent drag multiplier (drag_z) introduces additional variability in the u → v mapping. All three models have enough conditional capacity (source-conditioned GMMs) to approximate this in normalized space, which likely explains why gmm/markov still achieve near-zero pooled energy distance and modest CRPS. The latent model’s conditional p(v | s, u) may not be leveraging u as effectively (as seen in larger corr_err), and the regime mismatch compounds the issue.
- Evaluation protocol (eval.py) is rollout-only and uniform across models: CRPS via MC on samples; per-source aggregates computed with all rollout samples; correlation and 2D energy computed using one rollout sample per row. This rules out evaluation bias as the cause of the ranking.
- Config parity: All runs use the same training schedule/hyperparameters family (emb_dim=8; markov components=2; latent components_u=2, components_v=4; elbo_samples=1; 50 epochs). There’s no evidence of a training schedule discrepancy between clean/noisy or across models.

Cross-check vs prior analyses

- Earlier write-ups noted latent’s joint/regime structure once matched a shared-k generator. The current code purposely “breaks up” the noise regimes (independent per stream). The latent model was not updated to reflect that change, which plausibly and demonstrably degrades its rollout fit relative to gmm/markov. This explains the reversal where gmm/markov remain strong while latent degrades on noisy.

Conclusions

- gmm and markov: Best pooled rollout fidelity on both clean and noisy; small CRPS, near-zero 2D energy, and low correlation error. Per-source calibration is only moderate, but better than latent on spread.
- latent: Worse pooled rollout fidelity and coupling metrics; per-source spread is notably miscalibrated. The dominant, code-verified reason is a structural mismatch: the model assumes a shared sensor regime and bias across both sensors, but the generator now uses independent regimes per stream.

Recommendations (if we want latent to benefit from the new noisy setup)

- Update the latent sensor model to mirror the generator:
  - Use independent regimes k_up and k_down with per-source Bernoulli(π(source)), separate ±B signs per stream, and separate σ selections (σ0/σ1) per stream in both the likelihood and the sampler.
  - Keep π(source) learned via an embedding, as in the current code.
- After aligning the regime structure, re-run training/eval. We expect corr_err_by_source and energy_2d to drop significantly for latent, CRPS to improve, and downstream std log-error by source to move closer to gmm/markov levels.
- Optional: strengthen p(v | s, u) capacity (e.g., deeper context MLP or more components) if coupling remains weak after regime alignment.

Files and lines consulted (non-exhaustive)

- Data: markov_rollout_trap/make_data.py — independent z_up/z_down and biases, per-row drag_z, global z-scoring.
- Model: markov_rollout_trap/model.py — LatentModel shared k and shared ±B across sensors; GMM/Markov sample paths; parameter capacities.
- Eval: markov_rollout_trap/eval.py — rollout-only sampling and metrics implementations.
- Config/Train: markov_rollout_trap/{config.py, train.py} — shared hyperparameters and training loop across runs.
