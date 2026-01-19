# Analysis-2: Updated Metrics and Code-Verified Findings (post-normalization and fixes)

Scope

- Read all metrics under runs/{gmm,markov,latent}/{clean,noisy}/eval/metrics.json
- Cross-check README and the current code (make_data.py, model.py, eval.py, train.py)
- Answer: (1) How the models did, (2) how we expected them to do, (3) why discrepancies exist
- Validate hypotheses directly in code. Incorporate the note that upstream/downstream inputs were normalized.

Key code changes verified

- Input normalization: make_data.py z-scores the observed upstream and downstream per dataset before writing parquet.
  - Code: make_data.py
    - upstream_z = (upstream_obs - upstream_mean) / upstream_std
    - downstream_z = (downstream_obs - downstream_mean) / downstream_std
  - Effect: All models train/eval on standardized units (per-variable, dataset-level normalization).

- GMM marginal NLL bug fixed: \_gmm_1d_log_prob now includes mixture normalization.
  - Code: model.py::\_gmm_1d_log_prob
    - Uses logsumexp(logits + log N) − logsumexp(logits)
  - This corrects the optimistic NLL seen earlier.

- Estimator harmonization for latent downstream CRPS: now uses standard MC CRPS via importance resampling.
  - Code: model.py::LatentModel.eval_fit_arrays
    - \_downstream_conditional_samples_is(...) + multinomial resampling → \_crps_mc(...)
    - Downstream NLL still via self-normalized IS (nll_is), but CRPS is harmonized.

- Sampling parameterization: all sampling paths use std (sigma), not variance.
  - Code: model.py across JointGaussianModel/MarkovModel/LatentModel sample() and rollout()

What the new metrics show (normalized data)

Numbers below are from the current JSONs; units are z-score space.

- gmm
  - Clean
    - Upstream: nll_mean 1.048, crps_mean 0.398
    - Downstream_tf: nll_mean -1.443, crps_mean 0.033
    - Rollout: upstream_var_ratio 1.006, downstream_var_ratio 0.998, joint_energy ≈ -0.0009
  - Noisy
    - Upstream: nll_mean 1.075, crps_mean 0.408
    - Downstream_tf: nll_mean -0.270, crps_mean 0.108
    - Rollout: upstream_var_ratio 1.015, downstream_var_ratio 0.992, joint_energy ≈ -0.0013

- markov
  - Clean
    - Upstream: nll_mean 1.048, crps_mean 0.397
    - Downstream_tf: nll_mean -1.466, crps_mean 0.032
    - Rollout: upstream_var_ratio 0.994, downstream_var_ratio 0.997, joint_energy ≈ -0.0010
  - Noisy
    - Upstream: nll_mean 1.074, crps_mean 0.409
    - Downstream_tf: nll_mean -0.267, crps_mean 0.108
    - Rollout: upstream_var_ratio 1.028, downstream_var_ratio 1.006, joint_energy ≈ -0.0004

- latent
  - Clean
    - Upstream (mc): nll_mean 1.060, crps_mean 0.398
    - Downstream_tf (is): nll_mean 0.593, crps_mean 0.227
    - Rollout: upstream_var_ratio 0.995, downstream_var_ratio 1.246, joint_energy ≈ 0.0151
  - Noisy
    - Upstream (mc): nll_mean 1.087, crps_mean 0.410
    - Downstream_tf (is): nll_mean 0.628, crps_mean 0.241
    - Rollout: upstream_var_ratio 0.979, downstream_var_ratio 1.228, joint_energy ≈ 0.0117

1. How did the models do?

- Per-speed fits (teacher-forced)
  - Upstream: All three models are very similar in normalized space (nll_mean ≈ 1.05–1.09, crps ≈ 0.397–0.410). No material separation.
  - Downstream (teacher-forced): gmm and markov clearly outperform latent. On clean, gmm/markov achieve very low nll (≈ -1.44 to -1.47) and CRPS ≈ 0.032–0.033; on noisy, they degrade to nll ≈ -0.27 and CRPS ≈ 0.108 as expected. Latent is worse on both clean and noisy (nll_is ≈ 0.59–0.63, CRPS ≈ 0.227–0.241).

- Rollout (one sample per row)
  - gmm/markov: Excellent alignment with data; both upstream and downstream variance ratios ~1.0 on both clean and noisy; joint energy near zero; very small q90/mean errors.
  - latent: Upstream variance ratio ~1.0, but downstream variance is slightly inflated (≈1.23–1.25) and joint energy modestly worse than gmm/markov, though still small in absolute terms.

Bottom line: On normalized data, gmm and markov provide the best downstream teacher-forced fits and the best rollouts. Latent underperforms downstream (wider predictive) and rolls out with slightly inflated downstream variance.

2. How did we expect the models to do?

From README and prior reasoning:

- Clean
  - Expect gmm and markov to do well; latent should be competitive but not necessarily superior (since measurement-noise modeling is less critical).

- Noisy
  - Expect latent to gain an advantage by modeling shared two-regime sensor noise and separating process vs measurement; gmm/markov should degrade relative to clean.

3. What might cause the discrepancies?

Discrepancy A: Latent is worse than gmm/markov on downstream (both clean and noisy), contrary to expectation.

- Code-confirmed mechanics
  - markov/gmm condition directly on the observed upstream_speed during both training and teacher-forced evaluation.
    - Code: model.py::MarkovModel.forward/eval_fit_arrays and JointGaussianModel conditional via \_gmm_conditional_1d
  - latent’s downstream teacher-forced likelihood integrates out latents via a proposal p(u|s)p(v|s,u) and reweights by p(xu|u,k). This is a principled estimator but not as sharp a conditional as a direct parameterization may achieve, especially with finite samples.
    - Code: model.py::LatentModel.\_log_prob_down_conditional_is (self-normalized IS)
    - CRPS is computed from resampled IS draws: LatentModel.eval_fit_arrays → \_downstream_conditional_samples_is + \_crps_mc

- Likely cause (grounded in code structure)
  - Direct conditional modeling (markov/gmm) can produce a tighter p(downstream | source, upstream_obs) than latent’s IS-based approximate conditional over (u,v,k). Even with correct estimation, latent places uncertainty on both u and v plus the shared sensor regime, which naturally yields broader conditionals than a model that maps upstream_obs → downstream directly.
  - The sensor noise parameters in latent are global scalars per regime (sensor_log_std is global), not per-source; this may limit flexibility relative to gmm/markov, which learn per-source conditionals via GMM heads.
    - Code: model.py::LatentModel.sensor_log_std is a global 2-vector (shared across sources); sensor regime mixing weight depends on source, but sigmas are shared.

Discrepancy B: Earlier runs (ANALYSIS.md / ANALYSIS-1.md) showed catastrophic rollout variance; current runs do not.

- Code and data changes explain the reversal:
  - Input normalization now z-scores both variables per dataset (make_data.py). Working in normalized space stabilizes scale and eliminates earlier scale-mismatch artifacts.
  - GMM marginal NLL formula is corrected (model.py::\_gmm_1d_log_prob), removing the optimistic NLL bias and aligning fit metrics with observed rollouts.
  - Estimator harmonization for latent’s downstream CRPS removes apples-to-oranges comparisons seen previously.
  - Sampling throughout uses std correctly (no std-vs-var bug found in current code paths).

Clean vs noisy, revisited (normalized units)

- gmm/markov: Clear but modest degradation on noisy downstream teacher-forced metrics (CRPS increases from ~0.032→~0.108; NLL increases from ~-1.46→~-0.27), while rollout variance ratios remain ~1.0.
- latent: Also degrades slightly on noisy downstream (CRPS ~0.227→~0.241; NLL_is ~0.593→~0.628) and shows small downstream variance inflation in rollout (~1.23–1.25) on both datasets.

Answers to the three questions

1. How did the models do?
   - All models fit upstream similarly. For downstream (teacher-forced) and rollout, gmm and markov are best and nearly identical; latent is noticeably worse downstream and shows mild downstream variance inflation in rollout.

2. How did we expect them to do?
   - Clean: gmm/markov good; latent competitive. Noisy: latent expected to win due to explicit sensor-noise modeling. Post-normalization, gmm/markov remain best on downstream metrics; latent is not winning.

3. What might cause the discrepancy?
   - Working in normalized space removes scale pathologies seen earlier and may reduce the advantage of explicit sensor-noise modeling; direct conditionals (gmm/markov) can better exploit the observed upstream reading. In latent, downstream conditionals are computed by integrating over (u,v,k) with global sensor sigmas, naturally yielding broader predictions. These points are supported by the current code paths cited above; no std/var sampling bug is present in the current code.

Actionable next checks (if pursuing latent gains)

- Consider per-source sensor sigmas (instead of global), or richer conditioning of sensor parameters on source; this is closer to the data generation where noise regime probability depends on source and regime, and might sharpen conditionals.
- Increase IS sample size and report effective sample size for latent downstream metrics to rule out estimator variance as a contributor. [DONE]
- Explore conditioning p(v | source, u) with higher capacity or more mixture components to better capture the downstream mapping in normalized space.

References

- README: model/task definitions and evaluation protocol
- make_data.py: per-variable z-score normalization (verified)
- model.py: corrected GMM marginal NLL; harmonized latent CRPS estimation; std-based sampling across models
- eval.py: consistent aggregation of metrics and rollout diagnostics
- Prior passes: ANALYSIS.md, ANALYSIS-1.md (earlier, pre-normalization findings; now superseded by current code and runs)
