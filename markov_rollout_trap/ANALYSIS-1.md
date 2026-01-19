# Analysis-1: Markov Rollout Trap — Metrics- and Code-Backed Review

Goal

- Read all run metrics under runs/{gmm,markov,latent}/{clean,noisy}/eval/metrics.json and the README to assess: (1) how each model performed, (2) how we expected them to perform, (3) what causes discrepancies. Where needed, verify hypotheses by reading the implementation in model.py and eval.py rather than speculating.

Sources consulted

- README: markov_rollout_trap/README.md
- Metrics: six files under runs/.../eval/metrics.json (clean/noisy × gmm/markov/latent)
- Code: model.py, eval.py, train.py, make_data.py, model_test.py
- Prior write-up: markov_rollout_trap/ANALYSIS.md (earlier pass — numbers differ from current runs; see notes below)

Summary (TL;DR)

- Teacher-forced per-speed fits
  - Upstream: gmm reports lowest NLL but see a bug in its marginal NLL calculation (missing mixture normalization); CRPS values for all models are very large (~55–58), signaling highly over-dispersed predictive distributions relative to data.
  - Downstream (teacher-forced): markov has slightly better NLL/CRPS than gmm. latent reports extremely low CRPS (~3) but higher NLL (~7–7.8); its estimator differs (self-normalized IS and weighted CRPS), so numbers are not directly comparable.
- Rollout (one sample per row)
  - Upstream variance ratio is enormous across all models: ~450–670× the data. This is consistent with the large upstream CRPS and indicates the learned upstream predictive distributions are far too wide, not a sampling std/var bug.
  - Downstream variance ratio explodes via chain effects: gmm ~206–207×, latent ~4.38e5 (clean) to ~3.23e6 (noisy), markov catastrophic ~1.3e18–1.4e18.
- Causes (code-backed)
  1) The gmm upstream/downstream marginal NLL routine omits the mixture normalization term, biasing its NLL downward (too optimistic). 2) All three models learn overly large predictive variances (seen via CRPS and rollout), and markov’s conditional step compounds upstream dispersion into the downstream. 3) latent uses different estimators for teacher-forced metrics (IS/weighted CRPS), making direct comparisons misleading.

Important note vs earlier ANALYSIS.md

- The earlier ANALYSIS.md cites different numbers (e.g., latent downstream NLL < 0, downstream variance ratios 1e4–5e4). Current runs’ metrics are: latent downstream nll_is ≈ 7–7.8 (positive), downstream variance ratios ≈ 4.38e5 (clean) and 3.23e6 (noisy). This analysis uses the current JSONs and treats the earlier file as historical context only.

1) What the metrics say (by model and data)

- gmm
  - Clean
    - Upstream: nll_mean 2.309, nll_q90 2.717, crps_mean 58.097
    - Downstream_tf: nll_mean 5.742, nll_q90 5.864, crps_mean 43.641
    - Rollout: upstream_var_ratio 587.47, downstream_var_ratio 205.98, joint_energy 141.78
  - Noisy
    - Upstream: nll_mean 2.358, nll_q90 2.762, crps_mean 58.297
    - Downstream_tf: nll_mean 5.730, nll_q90 5.903, crps_mean 43.465
    - Rollout: upstream_var_ratio 562.97, downstream_var_ratio 207.39, joint_energy 141.96

- markov
  - Clean
    - Upstream: nll_mean 6.023, nll_q90 6.309, crps_mean 58.104
    - Downstream_tf: nll_mean 5.665, nll_q90 5.751, crps_mean 42.225
    - Rollout: upstream_var_ratio 669.09, downstream_var_ratio 1.3146e18, joint_energy 477.48
  - Noisy
    - Upstream: nll_mean 6.024, nll_q90 6.311, crps_mean 58.132
    - Downstream_tf: nll_mean 5.668, nll_q90 5.760, crps_mean 42.257
    - Rollout: upstream_var_ratio 648.34, downstream_var_ratio 1.4332e18, joint_energy 486.67

- latent
  - Clean
    - Upstream: nll_mc_mean 12.441, nll_q90 24.826, crps_mc_mean 54.787
    - Downstream_tf: nll_is_mean 7.009, nll_q90 16.708, crps_is_mean 2.921
    - Rollout: upstream_var_ratio 470.07, downstream_var_ratio 4.3802e5, joint_energy 110.11
  - Noisy
    - Upstream: nll_mc_mean 12.478, nll_q90 25.798, crps_mc_mean 54.814
    - Downstream_tf: nll_is_mean 7.762, nll_q90 18.715, crps_is_mean 3.176
    - Rollout: upstream_var_ratio 454.83, downstream_var_ratio 3.2342e6, joint_energy 109.13

Takeaways from the metrics

- Upstream per-speed fit appears poor for all models when judged by CRPS (≈55–58), indicating predictive distributions far too wide for the data scale.
- gmm shows “good” upstream NLL (~2.31–2.36), but see code bug below: the marginal mixture NLL omits a normalization and is too optimistic; CRPS tells a different story and aligns with the rollout over-dispersion.
- markov has teacher-forced downstream fit slightly better than gmm, but its rollout is catastrophically unstable downstream (variance ratio ~1e18) driven by the over-dispersed upstream sample propagating into the conditional.
- latent’s downstream teacher-forced CRPS is extremely low (~3) while its NLL_is is higher than gmm/markov; estimators differ (importance sampling and weighted CRPS), so direct comparison is not apples-to-apples. Its rollout is still drastically over-dispersed, especially downstream.
- Clean vs noisy: gmm and markov are surprisingly similar; latent degrades further on noisy downstream rollout (variance inflates by ~7.4× from clean).

2) What we expected (from README and modeling assumptions)

- Clean data
  - gmm: a full-covariance joint Gaussian should capture correlation and yield decent per-speed fits and plausible rollouts (though it cannot model the exact nonlinearity).
  - markov: with teacher-forcing during training, expect competitive per-speed fits and reasonable rollouts.
  - latent: on clean (no measurement mixture), expect performance similar to markov/gmm, maybe slightly better rollouts thanks to explicit process modeling.

- Noisy data
  - latent: should clearly outperform on teacher-forced metrics by modeling the shared two-regime sensor noise and separating process vs measurement; rollout should align closer to data variance.
  - gmm/markov: should degrade vs clean because they conflate process and measurement noise (fatter predictive tails, some rollout over-dispersion), but not orders of magnitude.

Reality vs expectation

- Instead of modest differences, we see severe over-dispersion for upstream across all models and catastrophic downstream rollouts for markov and latent. The teacher-forced downstream advantage expected for latent is not clearly reflected in NLL (it is actually worse than gmm/markov) and its very low CRPS is not directly comparable due to a different estimator. This suggests implementation issues (estimation and/or training behavior), not only modeling limits.

3) Discrepancy analysis — code-backed findings

A. gmm marginal NLL bug (confirmed in code)

- File: model.py, function _gmm_1d_log_prob
  - Current implementation: return logsumexp(logits + log N(x; mu_k, std_k))
  - Correct formula: logsumexp(logits + log N) − logsumexp(logits)
  - Because logits are unnormalized mixture logits, omitting the −logsumexp(logits) term inflates log p(x) by a constant per row, making NLL too small. This aligns with gmm’s “good-looking” upstream NLL (~2.31) despite very large CRPS and rollout variance.

B. Sampling parameterization (std vs var) — no bug found

- For all distributions, sampling uses mean + std * eps, not variance:
  - gmm and markov use zuko GMMs whose .sample() is handled by torch MixtureSameFamily; no custom sampling math present.
  - latent.sample() computes upstream/downstream as u + sigma * eps with sigma = exp(sensor_log_std) (std, not var).
- Tests in model_test.py validate that empirical sampling variances match the theoretical mixture variances for: joint gmm marginals and conditionals; markov upstream and downstream heads. This further supports that sampling itself isn’t squaring stds or similar.

C. Why are rollouts massively over-dispersed, especially upstream (all models)?

- The per-speed CRPS numbers (≈55–58) already indicate the learned predictive distributions are extremely wide vs the data. Rollout draws a single sample per row from those same distributions, so the aggregate variance across rows becomes large — matching the observed upstream variance ratios (~450–670×). This points to training converging to large-variance solutions rather than a sampling arithmetic bug.

D. Markov chain amplification (consistent with code structure)

- markov.rollout() samples upstream u ~ p(up|source), then conditions downstream on [source, u]. Any upstream over-dispersion directly propagates. If the learned conditional mean responds strongly to u (large effective slope) and/or the conditional noise is large, downstream variance inflates dramatically. The observed ~1e18× ratio is consistent with this compounding, given already-large upstream dispersion.

E. latent teacher-forced estimators differ (affecting comparability)

- Upstream: _log_prob_up_marginal uses Monte Carlo over u and sensor mixture; CRPS via unweighted samples.
- Downstream (teacher-forced): uses self-normalized importance sampling in _log_prob_down_conditional_is and a weighted CRPS estimator (_crps_weighted). These are not directly comparable to the closed-form/standard MC metrics used by gmm/markov and can behave differently (e.g., low CRPS_is despite higher NLL_is).

F. Clean vs noisy similarity (gmm/markov)

- The metrics are very similar across data modes. Given both models conflate process and measurement noise, we expected some degradation on noisy. The dominant over-dispersion in the learned distributions likely masks more subtle clean/noisy distinctions.

What likely caused the discrepancies?

- Primary: Learned predictive variances are much too large across models. Evidence: consistently high CRPS for upstream (~55–58) and huge upstream rollout variance ratios across gmm/markov/latent. No sampling std/var bug is present.
- Secondary: Markov conditional step turns upstream over-dispersion into catastrophic downstream dispersion (code structure confirms this flow).
- Measurement: gmm’s marginal NLL is biased (too low) due to missing mixture normalization; latent uses different estimator families for downstream, hindering direct comparisons.

Actionable checks (grounded in code)

1) Fix gmm marginal NLL
   - In _gmm_1d_log_prob, change return to:
     logsumexp(logits + comp_lp, dim=-1) − logsumexp(logits, dim=-1)
   - This will make gmm’s NLLs comparable to markov’s and to reality.

2) Add diagnostics for learned scales
   - Log/inspect the per-row predicted component stds for upstream heads (gmm/markov) and the prior stds (latent). Given the very large CRPS and rollout variance, these are likely inflated. Model code exposes the needed parameters via zuko distributions.

3) Harmonize estimators
   - Where closed forms exist (e.g., Gaussian CRPS), prefer analytic or consistent MC estimators across models. For latent, keep IS but also report a common MC baseline to aid comparability, and include effective sample sizes.

4) Sanity-check conditional sensitivity (markov)
   - Probe how the downstream mean changes with upstream by sweeping u while holding source fixed to estimate effective slope. If >1 or large noise, it explains the explosion when upstream is over-dispersed.

5) Training regularization/constraints
   - Consider constraining or regularizing predicted stds (e.g., penalties on very large log-stds, or priors) to avoid pathological wide solutions that fit via flat likelihoods.

Appendix: Code references

- _gmm_1d_log_prob (model.py): missing −logsumexp(logits) normalization when computing 1D marginal log-prob of a mixture.
- markov.rollout (model.py): samples upstream, then conditions downstream on sampled upstream — structurally amplifies any upstream dispersion.
- latent.sample (model.py): adds Gaussian noise with std = exp(sensor_log_std); sampling math uses std correctly.
- model_test.py: validates that sampling variances match theoretical variances of the heads/conditionals, supporting absence of a std/var arithmetic bug.
