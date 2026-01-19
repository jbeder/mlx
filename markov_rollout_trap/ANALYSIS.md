# Analysis: Markov Rollout Trap

This file summarizes how each model (gmm, markov, latent) performed on the clean vs noisy datasets, what we expected a priori, and likely causes of the observed discrepancies. Sources: README and the six metrics.json files under markov_rollout_trap/runs/*/*/eval/metrics.json.

## TL;DR

- Teacher-forced per-speed fit:
  - Downstream: latent is extremely strong (very low CRPS, negative NLL via IS estimator), gmm/markov are similar and much worse than latent.
  - Upstream: gmm best, markov worse, latent appears very poor (very high NLL_mc), despite having a sensor-noise model.
- Rollout: all models massively over-dispersed for upstream; downstream is also over-dispersed with gmm (~200×), latent (10k–48k×), and markov catastrophically unstable (≈1e18× variance ratio).
- Clean vs noisy: results are surprisingly similar for gmm and markov; latent degrades a bit on noisy downstream rollout (variance inflation grows).
- Most likely root cause: sampling/parameterization bug(s) that inflate standard deviation to variance at sampling time, then the Markov chain compounds the error. Latent’s evaluation uses different estimators (IS vs MC), likely making its upstream vs downstream metrics not directly comparable to the others.

---

## What we expected

Given the data-generating process in README:

- Clean data:
  - gmm: Should do reasonably on both per-speed fits and rollout—though it cannot model the nonlinear drag exactly, the joint 2D Gaussian often captures correlation and yields realistic rollouts.
  - markov: Teacher-forced training should produce good per-speed downstream fit and reasonable rollout (sample upstream, then downstream | upstream). It should be competitive with gmm on clean.
  - latent: On clean data (little measurement noise), latent separation of process vs measurement is less critical; we’d expect similar performance to markov/gmm and potentially slightly better rollout due to an explicit process model.

- Noisy data:
  - latent: Should clearly outperform others on per-speed fit due to explicit sensor noise mixture and separation of process vs measurement. Rollout should be best-aligned with data variance (close to Var(data)), since it samples latents then noisy observations.
  - gmm and markov: Should degrade vs clean because they conflate process and sensor noise; we’d expect fatter predictive tails and somewhat over-dispersed rollouts—but not orders of magnitude off.

---

## What actually happened (key numbers)

Per-speed fit (teacher-forced):

- gmm (clean vs noisy)
  - upstream: NLL_mean ≈ 2.31 vs 2.36; CRPS ≈ 58.10 vs 58.30 (stable)
  - downstream_tf: NLL_mean ≈ 5.74 vs 5.73; CRPS ≈ 43.64 vs 43.46

- markov (clean vs noisy)
  - upstream: NLL_mean ≈ 6.02 vs 6.02; CRPS ≈ 58.10 vs 58.13 (worse than gmm in NLL)
  - downstream_tf: NLL_mean ≈ 5.67 vs 5.67; CRPS ≈ 42.22 vs 42.26 (comparable/slightly better than gmm)

- latent (clean vs noisy)
  - upstream: NLL_mean ≈ 12.44 vs 12.48 (nll_mc), CRPS ≈ 54.79 vs 54.81 — looks very poor in NLL compared to others
  - downstream_tf: NLL_mean ≈ -4.62 vs -4.44 (nll_is), CRPS ≈ 2.89 vs 3.14 — dramatically better than gmm/markov

Rollout metrics (one sample per row):

- gmm: upstream_var_ratio ≈ 587 (clean) / 563 (noisy); downstream_var_ratio ≈ 206–207; joint_energy ≈ 142; downstream_mean_err ≈ 70–71
- markov: upstream_var_ratio ≈ 669 / 648; downstream_var_ratio ≈ 1.3e18–1.4e18; joint_energy ≈ 477–487; downstream_mean_err ≈ 3.4e7–3.6e7 (catastrophic)
- latent: upstream_var_ratio ≈ 467 / 452; downstream_var_ratio ≈ 1.0e4 (clean) / 4.8e4 (noisy); joint_energy ≈ 112–113; downstream_mean_err ≈ 54–67

Observations:

- All models produce upstream rollouts with variance ~450–670× the data. This is a strong signal of a systematic sampling/scale issue for upstream across implementations.
- Downstream rollout inflation is model-dependent: gmm (~200×) < latent (1e4–5e4×) << markov (≈1e18×). Markov’s chain amplifies the upstream explosion into astronomical downstream variance.
- Clean vs noisy is surprisingly similar for gmm and markov. Latent shows some degradation in noisy rollout (downstream variance ratio increases).

---

## Likely causes of the discrepancies

1) Sampling parameterization bug (std vs var)
- Symptom: Upstream rollout variance is ~500–650× data for all models, even though gmm and markov report moderate teacher-forced NLLs. If the sampler uses variance where std was intended (e.g., sample = mu + variance * eps instead of mu + std * eps), the variance inflates by roughly std^2; squaring a std of ~20 would yield ~400× variance—close to what we see.
- Consequence: gmm and latent show large—but bounded—downstream variance inflation; markov, which samples downstream conditioned on the (already-bloated) upstream, can explode to ~1e18× if the conditional mapping further scales variance.

2) Markov chain amplification
- Even without a bug, if p(downstream | source, upstream) has a strong linear dependence on upstream (slope > 1) or a large conditional noise term, any upstream over-dispersion gets amplified. With a sampling-scale bug, this becomes catastrophic (observed ~1e18×).

3) Inconsistent/biasy evaluation estimators for latent
- The latent metrics use different estimators and names: downstream (nll_is, crps_is) vs upstream (nll_mc, crps_mc). The negative downstream NLL and ultra-low CRPS compared to others suggest the IS estimator may be biased/optimistic or not directly comparable. Conversely, the upstream MC NLL is extremely large (≈12.5) relative to others, which may reflect estimator variance or a mis-specification in the upstream predictive likelihood calculation (e.g., integrating latents incorrectly or using too few samples).

4) Clean vs noisy similarity for gmm/markov
- Since both models conflate process and measurement noise, we expected some degradation on noisy vs clean. The near-identical metrics hint that either the noise regimes are under-reflected in the learned variances, or that the evaluation set/path was similar across runs. It’s also consistent with a dominant sampling bug that masks the more subtle clean-vs-noisy distinctions.

5) Possible feature scaling / space mismatch
- If training/eval are performed in one (normalized) space but sampling emits in another (unnormalized) without the inverse transform, or vice versa, the resulting rollout variance can be severely mis-scaled while teacher-forced likelihoods remain “reasonable” in the trained space.

---

## How each model did vs expectations

- gmm
  - Expected: Reasonable per-speed fit and decent rollout, especially on clean.
  - Observed: Per-speed fit is stable and moderate, but rollout is heavily over-dispersed (≈200× downstream var, ≈560× upstream var). Indicates sampling scale issue rather than inherent model limitation.

- markov
  - Expected: Strong downstream per-speed fit (teacher-forced) and reasonable rollout.
  - Observed: Downstream teacher-forced fit is indeed slightly better than gmm, but rollout is catastrophically unstable (downstream var ratio ~1e18). This is consistent with upstream over-dispersion plus chain amplification, likely due to the same sampling scale bug compounded in the conditional step.

- latent
  - Expected: Clear win on noisy data (per-speed and rollout), and competitive on clean.
  - Observed: Downstream teacher-forced fit looks exceptionally strong (nll_is < 0; crps_is ~3), but upstream per-speed NLL_mc is far worse than others. Rollout still suffers large over-dispersion (10k–48k× downstream; ~450× upstream). Mixed estimator choices (IS vs MC) and potential evaluation bias likely distort per-speed comparisons; rollout results still point to the same sampling-scale pathology.

---

## Recommended checks and fixes

1) Validate sampling math per model
- For all models, verify that sampling uses std (sigma) and not variance:
  - Should be: sample = mean + std * eps, eps ~ Normal(0, 1)
  - Not: mean + variance * eps, or mean + std^2 * eps
- Unit test: For a fixed source, draw many samples; compare empirical variance to the parameterized variance from the model head. They should match within Monte Carlo error.

2) Markov conditional stability
- Inspect p(downstream | source, upstream) parameterization. If downstream mean is an affine function of upstream, check learned slope magnitude; overly large slopes will amplify upstream variance.
- As a diagnostic, compute rollout with teacher-forced upstream (i.e., sample only downstream | observed upstream). If downstream variance ratio ≈ 1 in this setting, the upstream step is the main culprit.

3) Harmonize evaluation estimators
- Use the same estimator across models for NLL/CRPS where closed forms exist (e.g., CRPS for Gaussian has an analytic form). For latent, ensure IS is well-normalized, uses sufficient samples, and report standard errors.
- Recompute latent upstream predictive likelihood carefully; verify the mixture-of-Gaussians around the latent with shared component selection is integrated correctly.

4) Check feature scaling consistency
- Confirm any normalization applied during training is inverted consistently during sampling/eval. A mismatch can preserve “OK” teacher-forced likelihoods (computed in normalized space) but yield mis-scaled raw rollouts.

5) Clean vs noisy dataset plumbing
- Double-check that each run evaluated on the intended dataset split (clean vs noisy) and that the run directories indeed point to their corresponding data files.

---

## Conclusion

- The central issue is severe rollout over-dispersion across all models, with markov compounding it to catastrophic levels. This strongly suggests a shared sampling-scale bug (std vs var) and/or a space-mismatch during sampling, rather than a fundamental modeling limitation.
- Once sampling is corrected and evaluators are harmonized, we expect:
  - gmm to produce reasonable rollouts on clean and somewhat over-dispersed on noisy (but within a small factor).
  - markov to be competitive on clean and acceptable on noisy, with downstream rollout variance much closer to 1×.
  - latent to win on noisy (and be competitive on clean), with per-speed metrics and rollout variance aligned with the data, and without the current estimator artifacts.
