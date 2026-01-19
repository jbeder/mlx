import math

import torch

from markov_rollout_trap.model import (
    JointGaussianModel,
    LatentModel,
    MarkovModel,
    _gmm_conditional_1d,
)


def _mixture_marginal_var_from_base_logits(logits: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """
    Compute mixture marginal variance for a 1D mixture given component logits, means, and variances.

    Args:
        logits: (B, K)
        mu:     (B, K)
        var:    (B, K)
    Returns:
        (B,) variance
    """
    w = torch.softmax(logits, dim=-1)
    m = (w * mu).sum(dim=-1)  # (B,)
    second = (w * (var + mu**2)).sum(dim=-1)
    return second - m**2


def _mixture_marginal_var_1d(dist) -> torch.Tensor:
    """
    Compute marginal variance for a 1D mixture over Normal components.
    """
    # Torch MixtureSameFamily path
    if hasattr(dist, "mixture_distribution") and hasattr(dist, "component_distribution"):
        mix = dist.mixture_distribution
        if hasattr(mix, "probs") and mix.probs is not None:
            w = mix.probs
        else:
            w = torch.softmax(mix.logits, dim=-1)
        comp = dist.component_distribution
        mu = getattr(comp, "loc", getattr(comp, "mean", None))
        std = getattr(comp, "scale", getattr(comp, "stddev", None))
        if mu is None or std is None:
            raise AttributeError("Component distribution missing loc/scale attributes")
        if mu.ndim == 3 and mu.shape[-1] == 1:
            mu = mu.squeeze(-1)
            std = std.squeeze(-1)
        var = std**2
        m = (w * mu).sum(dim=-1)
        second = (w * (var + mu**2)).sum(dim=-1)
        return second - m**2

    # zuko Mixture path (has .logits and .base)
    if hasattr(dist, "logits") and hasattr(dist, "base"):
        w = torch.softmax(dist.logits, dim=-1)
        base = dist.base
        mu = getattr(base, "loc", getattr(base, "mean", None))
        std = getattr(base, "scale", getattr(base, "stddev", None))
        if mu is None or std is None:
            # Fallback to covariance if exposed
            cov = getattr(base, "covariance_matrix", None)
            if cov is None:
                raise AttributeError("Base distribution missing parameters to compute variance")
            mu = getattr(base, "loc", getattr(base, "mean"))
            std = cov.sqrt()
        if mu.ndim == 3 and mu.shape[-1] == 1:
            mu = mu.squeeze(-1)
            std = std.squeeze(-1)
        var = std**2
        m = (w * mu).sum(dim=-1)
        second = (w * (var + mu**2)).sum(dim=-1)
        return second - m**2

    raise TypeError("Unsupported mixture distribution type for variance extraction")


def _assert_close_variance(emp: torch.Tensor, theo: torch.Tensor, rtol: float = 0.15, atol: float = 1e-4):
    """Assert empirical vs theoretical variance are close per batch element.

    Uses relative tolerance, with a fallback absolute tolerance for very small values.
    """
    emp = emp.detach()
    theo = theo.detach()
    # Avoid divide-by-zero; where theo is tiny, use absolute tolerance only
    rel_err = (emp - theo).abs() / (theo.abs() + 1e-12)
    # A match if either relative or absolute criteria satisfied
    ok_mask = (rel_err <= rtol) | ((emp - theo).abs() <= atol)
    if not torch.all(ok_mask):
        # Provide helpful debugging info
        idx = (~ok_mask).nonzero(as_tuple=False).squeeze(-1)
        details = [
            f"idx={i.item()}, emp={emp[i].item():.6f}, theo={theo[i].item():.6f}, rel_err={rel_err[i].item():.3f}"
            for i in idx
        ]
        raise AssertionError("Empirical vs theoretical variance mismatch:\n" + "\n".join(details))


def test_joint_gaussian_sampling_matches_marginals_and_conditional():
    torch.manual_seed(0)
    num_sources = 3
    model = JointGaussianModel(num_sources=num_sources, emb_dim=8, num_components=3).eval()

    # Use a single source id repeated to create a batch
    B = 16
    source = torch.zeros(B, dtype=torch.long)
    dist = model(source)

    # Marginal variances for upstream (dim=0) and downstream (dim=1)
    base = dist.base
    logits = dist.logits  # (B,K)
    mu = base.loc  # (B,K,2)
    cov = base.covariance_matrix  # (B,K,2,2)

    # Theoretical variances per dim
    var_up = _mixture_marginal_var_from_base_logits(logits, mu[..., 0], cov[..., 0, 0])
    var_down = _mixture_marginal_var_from_base_logits(logits, mu[..., 1], cov[..., 1, 1])

    # Empirical from many samples
    S = 8000
    samples = dist.sample((S,))  # (S,B,2)
    emp_up = samples[..., 0].var(dim=0, unbiased=True)
    emp_down = samples[..., 1].var(dim=0, unbiased=True)

    _assert_close_variance(emp_up, var_up)
    _assert_close_variance(emp_down, var_down)

    # Conditional p(down | up)
    xu = samples[0, :, 0].detach()  # fix a particular upstream observation per row
    cond = _gmm_conditional_1d(dist, xu, given_dim=0, target_dim=1)
    # Theoretical conditional variance from the conditional mixture
    cond_var = _mixture_marginal_var_1d(cond)
    # Empirical by sampling conditional repeatedly
    cond_samples = cond.sample((S,))  # (S,B)
    emp_cond = cond_samples.var(dim=0, unbiased=True)
    _assert_close_variance(emp_cond, cond_var, rtol=0.2)  # slightly looser due to conditioning variability


def test_markov_model_sampling_matches_heads():
    torch.manual_seed(1)
    num_sources = 4
    model = MarkovModel(num_sources=num_sources, emb_dim=8, num_components=3).eval()

    B = 16
    source = torch.full((B,), 2, dtype=torch.long)

    # Upstream head
    ctx = model.source_emb(source)
    up_dist = model.upstream(ctx)
    # Theoretical variance directly from MixtureSameFamily API
    var_up = _mixture_marginal_var_1d(up_dist)
    S = 8000
    up_samples = up_dist.sample((S,)).squeeze(-1)
    emp_up = up_samples.var(dim=0, unbiased=True)
    _assert_close_variance(emp_up, var_up)

    # Downstream teacher-forced head at a fixed upstream value
    xu = torch.randn(B)
    down_dist = model.downstream(torch.cat([ctx, xu.unsqueeze(-1)], dim=-1))
    var_down = _mixture_marginal_var_1d(down_dist)
    down_samples = down_dist.sample((S,)).squeeze(-1)
    emp_down = down_samples.var(dim=0, unbiased=True)
    _assert_close_variance(emp_down, var_down)


def test_latent_model_sensor_noise_variance_matches():
    torch.manual_seed(2)
    num_sources = 5
    model = LatentModel(num_sources=num_sources, emb_dim=8, num_components=2).eval()

    B = 12
    # Use a single source id so pi is shared and easy to evaluate
    src_id = 3
    source = torch.full((B,), src_id, dtype=torch.long)

    # Expected emission variance (for residual upstream - u or downstream - v): E_k[sigma_k^2]
    logit = model.sensor_logit(source).squeeze(-1)  # (B,)
    pi = torch.sigmoid(logit)[0]  # scalar (same across batch)
    std = torch.exp(model.sensor_log_std)  # (2,)
    sigma2 = (pi * (std[0] ** 2) + (1.0 - pi) * (std[1] ** 2)).item()

    # Draw many samples and compute residuals
    S = 3000
    res_up = []
    res_down = []
    for _ in range(S):
        ro = model.sample(source)
        res_up.append((ro.upstream - ro.u))
        res_down.append((ro.downstream - ro.v))
    res_up = torch.stack(res_up, dim=0)  # (S,B)
    res_down = torch.stack(res_down, dim=0)

    emp_var_up = res_up.var(dim=0, unbiased=True).mean()  # average across batch rows
    emp_var_down = res_down.var(dim=0, unbiased=True).mean()

    # Compare against expected sigma^2 with modest tolerance
    assert math.isclose(emp_var_up.item(), sigma2, rel_tol=0.2, abs_tol=1e-3), (
        f"latent upstream residual variance {emp_var_up.item():.6f} vs expected {sigma2:.6f}"
    )
    assert math.isclose(emp_var_down.item(), sigma2, rel_tol=0.2, abs_tol=1e-3), (
        f"latent downstream residual variance {emp_var_down.item():.6f} vs expected {sigma2:.6f}"
    )
