from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli
from zuko.distributions import MultivariateNormal, Normal


def _as_long(x: Tensor) -> Tensor:
    return x.to(dtype=torch.long)


def _ensure_1d(x: Tensor) -> Tensor:
    if x.ndim == 0:
        return x.view(1)
    return x


class GMMModel(nn.Module):
    """Per-source 2D Gaussian over (upstream_speed, downstream_speed).

    Parameters are maintained as:
      - mean: (S, 2)
      - raw_L: (S, 2, 2), lower-triangular with positive diagonal enforced via softplus

    This allows full covariance per source.
    """

    def __init__(self, num_sources: int, init_mean: Optional[Tensor] = None, init_cov: Optional[Tensor] = None):
        super().__init__()
        S = int(num_sources)
        if init_mean is None:
            init_mean = torch.zeros(S, 2, dtype=torch.float32)
        if init_cov is None:
            init_cov = torch.stack([torch.eye(2, dtype=torch.float32) for _ in range(S)], dim=0)  # (S,2,2)

        # Convert cov to cholesky
        with torch.no_grad():
            L0 = torch.linalg.cholesky(init_cov + 1e-6 * torch.eye(2, dtype=init_cov.dtype))

        self.mean = nn.Parameter(init_mean.clone())  # (S,2)
        self.raw_L = nn.Parameter(L0.clone())        # (S,2,2)
        self._softplus = nn.Softplus()

    @property
    def num_sources(self) -> int:
        return int(self.mean.shape[0])

    def _scale_tril(self) -> Tensor:
        L = torch.tril(self.raw_L)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag = self._softplus(diag) + 1e-6
        L = L.clone()
        L[..., 0, 0] = diag[..., 0]
        L[..., 1, 1] = diag[..., 1]
        return L

    def log_prob(self, source: Tensor, upstream: Tensor, downstream: Tensor) -> Tensor:
        """Joint log p([upstream, downstream] | source).

        Inputs must be tensors on the same device as parameters.
        Shapes: source: (N,), upstream: (N,), downstream: (N,)
        Returns: (N,) log probabilities.
        """
        src = _as_long(_ensure_1d(source))
        up = _ensure_1d(upstream)
        down = _ensure_1d(downstream)
        x = torch.stack([up, down], dim=-1)
        m = self.mean[src]
        L = self._scale_tril()[src]
        dist = MultivariateNormal(m, scale_tril=L)
        return dist.log_prob(x)

    @torch.no_grad()
    def sample(self, source: Tensor, n: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Rollout sample(s).

        If n is None, returns one sample per source row. Otherwise, returns n samples per row.
        Returns (upstream, downstream) tensors.
        """
        src = _as_long(_ensure_1d(source))
        m = self.mean[src]
        L = self._scale_tril()[src]
        dist = MultivariateNormal(m, scale_tril=L)
        if n is None:
            xy = dist.sample()
        else:
            xy = dist.sample((n,))  # (n, N, 2)
        return xy[..., 0], xy[..., 1]


class MarkovModel(nn.Module):
    """Teacher-forced Markov chain:

    p(up | s) = Normal(mu_u[s], std_u[s])
    p(down | s, up) = Normal(a[s] * up + b[s], std_d[s])
    """

    def __init__(
        self,
        num_sources: int,
        init_mu_u: Optional[Tensor] = None,
        init_std_u: Optional[Tensor] = None,
        init_a: Optional[Tensor] = None,
        init_b: Optional[Tensor] = None,
        init_std_d: Optional[Tensor] = None,
    ):
        super().__init__()
        S = int(num_sources)
        if init_mu_u is None:
            init_mu_u = torch.zeros(S, dtype=torch.float32)
        if init_std_u is None:
            init_std_u = torch.ones(S, dtype=torch.float32)
        if init_a is None:
            init_a = torch.zeros(S, dtype=torch.float32)
        if init_b is None:
            init_b = torch.zeros(S, dtype=torch.float32)
        if init_std_d is None:
            init_std_d = torch.ones(S, dtype=torch.float32)

        self.mu_u = nn.Parameter(init_mu_u.clone())
        self.log_std_u = nn.Parameter(init_std_u.log().clone())
        self.a = nn.Parameter(init_a.clone())
        self.b = nn.Parameter(init_b.clone())
        self.log_std_d = nn.Parameter(init_std_d.log().clone())

    @property
    def num_sources(self) -> int:
        return int(self.mu_u.shape[0])

    def log_prob(self, source: Tensor, upstream: Tensor, downstream: Tensor) -> Tensor:
        src = _as_long(_ensure_1d(source))
        up = _ensure_1d(upstream)
        down = _ensure_1d(downstream)

        std_u = torch.exp(self.log_std_u)
        std_d = torch.exp(self.log_std_d)

        mu_u_b = self.mu_u[src]
        std_u_b = std_u[src]
        a_b = self.a[src]
        b_b = self.b[src]
        std_d_b = std_d[src]

        up_dist = Normal(mu_u_b, std_u_b)
        down_mu = a_b * up + b_b
        down_dist = Normal(down_mu, std_d_b)
        return up_dist.log_prob(up) + down_dist.log_prob(down)

    @torch.no_grad()
    def sample(self, source: Tensor, n: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        src = _as_long(_ensure_1d(source))
        std_u = torch.exp(self.log_std_u)
        std_d = torch.exp(self.log_std_d)

        mu_u_b = self.mu_u[src]
        std_u_b = std_u[src]
        a_b = self.a[src]
        b_b = self.b[src]
        std_d_b = std_d[src]

        if n is None:
            up = Normal(mu_u_b, std_u_b).sample()
            down = Normal(a_b * up + b_b, std_d_b).sample()
        else:
            up = Normal(mu_u_b, std_u_b).sample((n,))  # (n, N)
            down = Normal(a_b * up + b_b, std_d_b).sample()
        return up, down


class LatentSensorModel(nn.Module):
    """Latent true speeds with shared 2-regime sensor noise mixture.

    Latent process (per source s):
      u ~ Normal(mu_u[s], var_u[s])
      v | u ~ Normal(a[s] * u + b[s], var_v[s])

    Sensor model (shared across upstream and downstream, same regime z per row):
      z ~ Bernoulli(pi[s])
      sensor_sigma = sigma0 if z=1 else sigma1 (with sigma1 >= sigma0)
      x | u, z ~ Normal(u, sensor_sigma^2)
      y | v, z ~ Normal(v, sensor_sigma^2)

    log_prob integrates out u and v, yielding a mixture of two 2D Gaussians with
    means m_s = [mu_u, a*mu_u + b] and covariances P_s + sigma_k^2 I, where
    P_s = [[var_u, a var_u], [a var_u, a^2 var_u + var_v]].
    """

    def __init__(
        self,
        num_sources: int,
        init_mu_u: Optional[Tensor] = None,
        init_var_u: Optional[Tensor] = None,
        init_a: Optional[Tensor] = None,
        init_b: Optional[Tensor] = None,
        init_var_v: Optional[Tensor] = None,
        init_pi: Optional[Tensor] = None,
        init_sigma0: float = 0.3,
        init_sigma1: float = 1.2,
    ):
        super().__init__()
        S = int(num_sources)
        if init_mu_u is None:
            init_mu_u = torch.zeros(S, dtype=torch.float32)
        if init_var_u is None:
            init_var_u = torch.ones(S, dtype=torch.float32)
        if init_a is None:
            init_a = torch.zeros(S, dtype=torch.float32)
        if init_b is None:
            init_b = torch.zeros(S, dtype=torch.float32)
        if init_var_v is None:
            init_var_v = 0.1 * torch.ones(S, dtype=torch.float32)
        if init_pi is None:
            init_pi = 0.5 * torch.ones(S, dtype=torch.float32)

        self.mu_u = nn.Parameter(init_mu_u.clone())        # (S,)
        self.log_var_u = nn.Parameter(init_var_u.clone().log())  # (S,)
        self.a = nn.Parameter(init_a.clone())              # (S,)
        self.b = nn.Parameter(init_b.clone())              # (S,)
        self.log_var_v = nn.Parameter(init_var_v.clone().log())  # (S,)
        # store pi via logits
        logits = torch.log(init_pi.clamp(1e-6, 1 - 1e-6)) - torch.log(1 - init_pi.clamp(1e-6, 1 - 1e-6))
        self.raw_pi = nn.Parameter(logits.clone())         # (S,)

        # Global/shared sensor sigmas with constraint sigma1 >= sigma0
        self.raw_sigma0 = nn.Parameter(torch.tensor(init_sigma0).log())
        # positive increment via softplus on delta
        # initialize so that sigma1 ~= init_sigma1
        delta = torch.tensor(max(init_sigma1 - init_sigma0, 1e-4))
        self.raw_delta_sigma = nn.Parameter(delta.log())
        self._softplus = nn.Softplus()

    @property
    def num_sources(self) -> int:
        return int(self.mu_u.shape[0])

    def _sensor_sigmas(self) -> Tuple[Tensor, Tensor]:
        sigma0 = self._softplus(self.raw_sigma0) + 1e-6
        sigma1 = sigma0 + self._softplus(self.raw_delta_sigma)
        return sigma0, sigma1

    def _prior_mean_cov(self) -> Tuple[Tensor, Tensor]:
        """Return m: (S,2) and P: (S,2,2) from current parameters."""
        var_u = torch.exp(self.log_var_u).clamp_min(1e-9)
        var_v = torch.exp(self.log_var_v).clamp_min(1e-9)
        m_u = self.mu_u
        m_v = self.a * self.mu_u + self.b
        m = torch.stack([m_u, m_v], dim=-1)  # (S,2)

        P = torch.zeros(self.num_sources, 2, 2, dtype=m.dtype, device=m.device)
        P[:, 0, 0] = var_u
        P[:, 0, 1] = self.a * var_u
        P[:, 1, 0] = self.a * var_u
        P[:, 1, 1] = (self.a * self.a) * var_u + var_v
        return m, P

    def log_prob(self, source: Tensor, upstream: Tensor, downstream: Tensor) -> Tensor:
        src = _as_long(_ensure_1d(source))
        up = _ensure_1d(upstream)
        down = _ensure_1d(downstream)
        xy = torch.stack([up, down], dim=-1)

        m, P = self._prior_mean_cov()
        m_b = m[src]
        P_b = P[src]
        pi = torch.sigmoid(self.raw_pi)[src]
        sig0, sig1 = self._sensor_sigmas()
        I = torch.eye(2, dtype=P_b.dtype, device=P_b.device)
        cov0 = P_b + (sig0 ** 2) * I
        cov1 = P_b + (sig1 ** 2) * I

        dist0 = MultivariateNormal(m_b, covariance_matrix=cov0 + 1e-12 * I)
        dist1 = MultivariateNormal(m_b, covariance_matrix=cov1 + 1e-12 * I)
        l0 = torch.log(pi + 1e-12) + dist0.log_prob(xy)
        l1 = torch.log(1 - pi + 1e-12) + dist1.log_prob(xy)
        return torch.logsumexp(torch.stack([l0, l1], dim=-1), dim=-1)

    @torch.no_grad()
    def sample(self, source: Tensor, n: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        src = _as_long(_ensure_1d(source))
        device = self.mu_u.device
        var_u = torch.exp(self.log_var_u)
        var_v = torch.exp(self.log_var_v)
        pi = torch.sigmoid(self.raw_pi)[src]
        sig0, sig1 = self._sensor_sigmas()

        if n is None:
            # Latents
            u = Normal(self.mu_u[src], torch.sqrt(var_u[src])).sample()
            v = Normal(self.a[src] * u + self.b[src], torch.sqrt(var_v[src])).sample()
            # Regime
            z = Bernoulli(probs=pi).sample().to(dtype=torch.bool)
            sig = torch.where(z, sig0.expand_as(pi), sig1.expand_as(pi))
            up = Normal(u, sig).sample()
            down = Normal(v, sig).sample()
        else:
            # Broadcast to (n, N)
            mu_u_b = self.mu_u[src].expand(n, -1)
            std_u_b = torch.sqrt(var_u[src]).expand(n, -1)
            a_b = self.a[src].expand(n, -1)
            b_b = self.b[src].expand(n, -1)
            std_v_b = torch.sqrt(var_v[src]).expand(n, -1)

            u = Normal(mu_u_b, std_u_b).sample()  # (n, N)
            v = Normal(a_b * u + b_b, std_v_b).sample()
            # Regime per-row (same regime for both sensors)
            z = Bernoulli(probs=pi).sample((n,)) > 0.5  # (n, N)
            sig = torch.where(z, sig0.expand_as(z), sig1.expand_as(z)).to(device=device, dtype=mu_u_b.dtype)
            up = Normal(u, sig).sample()
            down = Normal(v, sig).sample()

        return up, down


__all__ = ["GMMModel", "MarkovModel", "LatentSensorModel"]