# Analysis-3: Metrics- and Code-Verified Review (current runs)

Goal

- Read README, the six metrics.json under runs/{gmm,markov,latent}/{clean,noisy}/eval, and the current code (make_data.py, model.py, eval.py, train.py, config.py). Answer:
  1) How did the models do in each case?
  2) How did we expect them to do?
  3) What might cause any discrepancy?
- Where hypotheses arise, confirm/deny them by reading the code; do not rely solely on prior analyses.

Sources consulted

- README: markov_rollout_trap/README.md
- Metrics: all six JSONs in runs/.../eval/metrics.json
- Code: make_data.py, model.py, eval.py, train.py, config.py
- Prior write-ups: ANALYSIS.md, ANALYSIS-1.md, ANALYSIS-2.md

Context vs prior analyses

- ANALYSIS-1 documented severe rollout over-dispersion and a GMM marginal-NLL bug, based on earlier runs.
- ANALYSIS-2 noted important fixes and normalization: per-variable z-score normalization in make_data.py, corrected GMM marginal NLL, harmonized CRPS estimation for latent downstream. Current metrics reflect that state and align with ANALYSIS-2 rather than the earlier, pre-fix behavior.

Key implementation facts (verified in code)

- Data normalization (make_data.py): upstream_speed and downstream_speed are standardized per dataset before writing parquet.
- GMM marginal NLL (model.py::_gmm_1d_log_prob) includes mixture normalization: logsumexp(logits + log N) − logsumexp(logits).
- Estimators used in eval_fit_arrays (model.py):
  - gmm/markov: teacher-forced NLL and CRPS via standard log_prob and MC CRPS.
  - latent: upstream uses MC for marginal (nll_mc/crps_mc). Downstream uses self-normalized importance sampling for NLL (nll_is) and importance-resampled MC for CRPS, harmonizing with others’ CRPS.
- Sampling throughout uses std (sigma), not variance; rollout paths are standard (mean + std * eps).

What the current metrics say

Numbers below are means in normalized (z-score) units; see JSONs for full details.

- gmm
  - Clean: upstream nll 1.048, crps 0.398; downstream_tf nll -1.443, crps 0.033; rollout var ratios up 0.994, down 0.996; joint_energy ≈ -0.0006.
  - Noisy: upstream nll 1.075, crps 0.408; downstream_tf nll -0.270, crps 0.108; rollout var ratios up 1.007, down 0.996; joint_energy ≈ -0.0003.

- markov
  - Clean: upstream nll 1.048, crps 0.397; downstream_tf nll -1.466, crps 0.032; rollout var ratios up 0.995, down 1.006; joint_energy ≈ -0.0005.
  - Noisy: upstream nll 1.074, crps 0.409; downstream_tf nll -0.267, crps 0.108; rollout var ratios up 1.024, down 1.011; joint_energy ≈ -0.0005.

- latent
  - Clean: upstream (mc) nll 1.060, crps 0.398; downstream_tf (is) nll 0.593, crps 0.227; rollout var ratios up 0.973, down 1.252; joint_energy ≈ 0.0164.
  - Noisy: upstream (mc) nll 1.087, crps 0.410; downstream_tf (is) nll 0.628, crps 0.241; rollout var ratios up 0.958, down 1.234; joint_energy ≈ 0.0131.

1) How did the models do?

- Teacher-forced per-speed fit
  - Upstream: All three are effectively tied (nll ≈ 1.05–1.09, crps ≈ 0.397–0.410).
  - Downstream (conditioned on observed upstream): gmm and markov are clearly best. On clean, nll ≈ -1.44 to -1.47 and crps ≈ 0.032–0.033; on noisy, they degrade to nll ≈ -0.27 and crps ≈ 0.108. Latent is worse on both datasets (nll_is ≈ 0.59–0.63, crps ≈ 0.23–0.24).

- Rollout (one sample per row; aggregate over S*B)
  - gmm and markov: Excellent alignment with data; upstream/downstream variance ratios ~1.0 and joint energy ~0 for both clean and noisy.
  - latent: Upstream variance ratio ~1.0 but modestly under 1 (0.96–0.97). Downstream variance is mildly inflated (≈1.23–1.25). Joint energy is small in absolute terms but worse than gmm/markov.

2) How did we expect them to do?

- From README’s data-generating process:
  - Clean: gmm and markov should perform well; latent should be competitive (explicit process model), but measurement-noise modeling is less critical.
  - Noisy: latent should benefit from modeling shared two-regime sensor noise and separating process vs measurement; gmm/markov should degrade vs clean.

3) What might cause the discrepancy (observed vs expected)?

Main discrepancy: latent underperforms gmm/markov on downstream (both teacher-forced and rollout) and shows slight downstream over-dispersion in rollout; we expected latent to excel on noisy.

Code-backed explanations:

- Direct conditional vs latent integration
  - gmm/markov parameterize p(downstream | source, upstream_obs) directly (model.py::JointGaussianModel via _gmm_conditional_1d; MarkovModel.forward). This can produce very sharp conditionals in normalized space.
  - latent computes p(downstream | source, upstream_obs) by integrating over (u, v, k) with a proposal p(u|s)p(v|s,u)p(k|s) and reweighting by p(xu | u, k) (model.py::LatentModel._log_prob_down_conditional_is and _downstream_conditional_samples_is). Even with correct IS, this conditional naturally reflects uncertainty in u, v, and the shared regime k, yielding a broader predictive than a direct conditional.

- Sensor parameterization capacity
  - latent uses global sensor log-stds shared across sources (model.py::LatentModel.sensor_log_std is a learned 2-vector), with source-dependent mixing weight via an embedding. In contrast, gmm/markov’s conditionals can vary per source more flexibly through their heads. This mismatch can limit latent’s ability to sharpen conditionals per source/regime in normalized space.

- Estimator differences (still relevant for NLL, less so for CRPS)
  - latent downstream NLL is self-normalized IS (nll_is), while gmm/markov use closed-form log_prob. Although CRPS is harmonized via importance resampling, the nll_is remains not strictly apples-to-apples. Nonetheless, both NLL and CRPS rank latent worse downstream, consistent with broader conditionals, not merely estimator bias.

Non-issues (checked and ruled out):

- Std vs variance sampling bug: Not present. All sampling uses std, and zuko/torch distributions handle sampling correctly (model.py sample()/eval_rollout_arrays across models).
- GMM marginal NLL bug: Fixed in current code (model.py::_gmm_1d_log_prob).
- Scale/pathology from raw units: Data are standardized in make_data.py; metrics are in z-scored units. This explains the large discrepancy vs the pre-normalization behaviors reported in ANALYSIS-1.

Clean vs noisy behavior

- gmm/markov: As expected, downstream teacher-forced metrics degrade modestly on noisy (crps from ~0.032 to ~0.108; nll from ~-1.46 to ~-0.27) while rollout quality remains near perfect (~1.0 var ratios).
- latent: Also degrades slightly on noisy downstream (crps ~0.227 → ~0.241; nll_is ~0.593 → ~0.628). Rollout shows consistent slight downstream variance inflation on both datasets (~1.23–1.25), with upstream a bit under-dispersed (~0.96–0.97).

Conclusions

- How they did: gmm and markov are best on downstream teacher-forced fit and rollout; all three are tied on upstream. latent is worse downstream and rolls out with slightly inflated downstream variance.
- Expected vs observed: We expected latent to win on noisy; instead, direct conditional models (gmm/markov) win. Given the current architecture, this is plausible: conditioning directly on upstream_obs in normalized space is very informative, while latent’s approximate conditional integrates over latent/process/sensor uncertainties and uses global sensor sigmas, producing broader predictions.
- Causes: Confirmed by code paths, not speculation: direct conditional modeling vs IS-based latent conditional; global (not per-source) sensor sigmas; no sampling std/var bug; corrected GMM likelihood; standardized data.

Actionable next steps (if we want latent to close the gap)

- Increase conditional capacity and/or sensor flexibility:
  - Condition sensor sigmas on source (per-source or via an MLP), not just the mixing weight.
  - Consider richer p(v | s, u) (e.g., more components) or an explicit conditional head for downstream given observed upstream while retaining the latent structure.
- IS diagnostics: increase sample count and report effective sample size for downstream conditionals to confirm estimation is not the bottleneck.
- Keep metrics comparable: continue to report CRPS via common MC across models; optionally add a common MC baseline for latent NLL (with caution about bias/variance).

File/Code references

- make_data.py: per-variable z-score normalization.
- model.py::_gmm_1d_log_prob and _gmm_conditional_1d: corrected mixture math and conditional.
- model.py::MarkovModel.forward/eval_fit_arrays/eval_rollout_arrays: direct conditionals and rollout.
- model.py::LatentModel._log_prob_down_conditional_is/_downstream_conditional_samples_is/eval_fit_arrays: IS-based downstream conditional and harmonized CRPS.
- eval.py::_aggregate_metrics: computes teacher-forced aggregates and rollout metrics consistently across models.
