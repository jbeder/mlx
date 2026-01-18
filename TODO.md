# Code Review TODO — markov_rollout_trap/eval.py

Goal: Capture issues found while reviewing eval.py against markov_rollout_trap/README.md and the model/train contracts, organized into three categories: (1) fundamental correctness bugs, (2) code health/duplication WTFs, and (3) minor polish (top 3 only). Each item includes a concise fix suggestion.

## 1) Fundamental correctness issues / bugs

~ Energy distance uses self-pairs (biased within terms) — FIXED
  - Implemented unbiased within-set terms by excluding diagonals and averaging over m·(m−1) in both `_aggregate_metrics` and the recomputation path in `main`.
  - Verified via running eval successfully on markov/noisy run.

- CLI parameter `--energy_k` not threaded through metric computation (double sampling, inconsistent pipeline)
  - Where: `_aggregate_metrics` has `energy_k=2000` default and is called from the per-kind compute functions without passing the CLI value. Then `main` re-samples a fresh rollout and re-computes the entire `rollout` block to “patch” in the requested subset size.
  - Impact: (a) Redundant compute and extra RNG draws; (b) The final `rollout` metrics come from a second rollout sample, not the one produced inside the per-kind compute; (c) harder reproducibility/debugging.
  - Fix: Thread `energy_k` from `args` down into `_compute_metrics_*` and `_aggregate_metrics`, and compute the rollout metrics exactly once using a single rollout sample; return that in the final JSON without the patch step.
~ Status — FIXED: Threaded `energy_k` through `_compute_metrics_*` and `_aggregate_metrics` and removed the patch/recompute block from `main`. Metrics are now computed once from a single rollout sample, honoring the CLI subset size.

- Latent evaluation: regime coupling is lost in per-dim MC sampling helpers
  - Where: `_latent_samples_up` and `_latent_samples_down` each sample a sensor regime `k` independently per (S,B) call. The generative model shares the same regime `k` for both sensors on a row.
  - Impact: For the current usage (per-dimension CRPS only), this is acceptable; however, if these helpers were reused to construct joint metrics from CRPS samples, they would fail to capture the intended upstream/downstream coupling.
  - Fix: Provide a joint sampler that draws a single `k` per row (per sample) and reuses it for both upstream and downstream when joint samples are needed; keep per-dim samplers or share a common routine that can toggle “shared_k=True/False”.
~ Status — FIXED: Added `_latent_samples_joint(model, source, S, shared_k=True)` that draws a single regime k per row when requested and returns joint (xu, xd) samples for potential future joint metrics.

- Model architecture not faithfully reconstructed at eval time
  - Where: `_load_model` only reads `model_kind` and `num_sources` from `model.pt`, then rebuilds models with constructor defaults. Training, however, may have used different `emb_dim`, `num_components`, or `encoder_hidden` (for latent). This can cause `state_dict` shape mismatches or silently incorrect architectures if shapes happen to align.
  - Impact: Evaluations can fail or, worse, succeed with a mismatched architecture.
  - Fix: Save all necessary architecture hyperparameters in `model.pt` at train time (or load and parse `config.resolved.json`) and pass them into the model constructors in `_load_model`.
~ Status — FIXED: `_load_model` now reads `config.resolved.json` when present and reconstructs `markov` and `latent` architectures with the saved `emb_dim`, `num_components`, and `encoder_hidden`.

## 2) Code health / duplication WTFs (don’t change behavior but raise eyebrows)

~ Duplicate rollout/energy computations and JSON patching — FIXED
  - Removed the recomputation/patch block from `main`; `energy_k` is now threaded into `_compute_metrics_*` and `_aggregate_metrics`, so rollout metrics are computed once.

~ Two different energy-distance implementations sprinkled in two places — FIXED
  - With the patching removed from `main`, there is now a single energy-distance implementation inside `_aggregate_metrics`, eliminating duplication.

~ Repeated device copies of `sensor_log_std` — FIXED
  - Removed redundant `.to(device)` calls; we now read `model.sensor_log_std` directly (already on the model’s device).

~ Inconsistent seeding helpers across train/eval — FIXED
  - `eval._seed_all` now also seeds CUDA generators when available, mirroring train.

## 3) Minor polish (top 3 only)

- Unify and expose a single energy distance helper
  - Extract the energy-distance calculation into a single function (e.g., `_energy_distance(X, Y, unbiased=True)`), reuse it in both `_aggregate_metrics` and `main`.

- Reduce RNG overhead in MC log-likelihoods
  - Cache `log_num_samples = np.log(S)` or `torch.log(torch.tensor(S, device=...))` once per call rather than reconstructing a new tensor; or just use a Python float in the log-subtract (no grad under `@torch.no_grad()`).

- Naming and docstrings for clarity
  - `_gmm_1d_log_prob`: consider `dim_name="upstream|downstream"` or an Enum instead of `dim_index`; expand the doc to clarify the expected shapes for zuko’s `GMM` component/mixture distributions.

---

Suggested refactor contour (non-breaking):

1) Thread `energy_k` to `_compute_metrics_*` and `_aggregate_metrics`.
2) Have each `_compute_metrics_*` return both metrics and the single rollout arrays `(up_hat, down_hat)` it actually used.
3) Centralize energy distance in one helper that supports unbiased within-set terms and accepts `energy_k`.
4) Remove the “patch rollout metrics” block in `main`.
5) Optionally add a joint latent sampler that shares `k` across both sensors per row when joint samples are needed.