from __future__ import annotations

from typing import NamedTuple

import pandas as pd
import torch
from torch import nn
from torch.distributions import Distribution
from torch.utils.data import Dataset
from zuko.mixtures import GMM


class SpeedBatch(NamedTuple):
    source: torch.Tensor
    upstream_speed: torch.Tensor
    downstream_speed: torch.Tensor


class SpeedDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.source = torch.as_tensor(data["source"].to_numpy(), dtype=torch.long)
        self.upstream_speed = torch.as_tensor(data["upstream_speed"].to_numpy(), dtype=torch.float32)
        self.downstream_speed = torch.as_tensor(data["downstream_speed"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx) -> SpeedBatch:
        return SpeedBatch(
            source=self.source[idx],
            upstream_speed=self.upstream_speed[idx],
            downstream_speed=self.downstream_speed[idx],
        )


def _ensure_2d_samples(x: torch.Tensor) -> torch.Tensor:
    """Convert tensors of shape (S,B,1) or (S,B) or (B,1) to (S,B).

    If x has no sample dimension, assumes S=1 and adds it.
    """
    if x.dim() == 1:
        # (B,) -> (1,B)
        return x.unsqueeze(0)
    if x.dim() == 2:
        # Could be (S,B) or (B,1). If it's (B,1), treat as (1,B)
        if x.size(1) == 1 and x.size(0) != 1:
            return x.transpose(0, 1)
        return x
    if x.dim() == 3:
        # (S,B,1) -> (S,B)
        if x.size(-1) == 1:
            return x.squeeze(-1)
    return x


class JointGaussianModel(nn.Module):
    def __init__(self, num_sources: int, emb_dim: int = 8, num_components: int = 2):
        super().__init__()
        self.source_emb = nn.Embedding(num_sources, emb_dim)
        self.gmm = GMM(
            features=2,
            context=emb_dim,
            components=num_components,
            covariance_type="full",
            epsilon=1e-6,
        )

    def forward(self, source: torch.Tensor) -> Distribution:
        ctx = self.source_emb(source)  # (B, emb_dim)
        return self.gmm(ctx)

    @torch.no_grad()
    def sample(self, source: torch.Tensor, num_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rollout sampling from the joint 2D Gaussian mixture.

        Returns (upstream_hat, downstream_hat) with shape (S, B).
        """
        dist = self(source)
        y = dist.sample((num_samples,))  # (S,B,2)
        u = y[..., 0]
        d = y[..., 1]
        return _ensure_2d_samples(u), _ensure_2d_samples(d)


class MarkovRollout(NamedTuple):
    upstream: Distribution
    upstream_sample: torch.Tensor  # (B, 1)
    downstream: Distribution
    downstream_sample: torch.Tensor  # (B, 1)


class MarkovModel(nn.Module):
    def __init__(self, num_sources: int, emb_dim: int = 8, num_components: int = 2):
        super().__init__()
        self.source_emb = nn.Embedding(num_sources, emb_dim)

        self.upstream = GMM(
            features=1,
            context=emb_dim,
            components=num_components,
            covariance_type="diagonal",
            epsilon=1e-6,
        )
        self.downstream = GMM(
            features=1,
            context=emb_dim + 1,
            components=num_components,
            covariance_type="diagonal",
            epsilon=1e-6,
        )

    def forward(self, source: torch.Tensor, upstream_speed: torch.Tensor):
        """
        Downstream is conditioned on observed upstream (teacher forcing).

        Args:
            source: (B,)
            upstream_speed: (B,)
        """
        ctx = self.source_emb(source)  # (B, emb_dim)
        up_dist = self.upstream(ctx)

        u = upstream_speed.unsqueeze(-1)  # (B, 1)
        down_dist = self.downstream(torch.cat([ctx, u], dim=-1))
        return up_dist, down_dist

    @torch.no_grad()
    def rollout(self, source: torch.Tensor) -> MarkovRollout:
        """
        Rollout by sampling upstream speed.

        Args:
            source: (B,)
        """
        ctx = self.source_emb(source)
        up = self.upstream(ctx)
        u = up.sample()
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        down = self.downstream(torch.cat([ctx, u], dim=-1))
        d = down.sample()
        if d.ndim == 1:
            d = d.unsqueeze(-1)  # (B, 1)
        return MarkovRollout(
            upstream=up,
            upstream_sample=u,
            downstream=down,
            downstream_sample=d,
        )

    @torch.no_grad()
    def sample(self, source: torch.Tensor, num_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rollout sampling using the Markov factorization.

        Returns (upstream_hat, downstream_hat) with shape (S, B).
        """
        ctx = self.source_emb(source)  # (B,E)
        up_dist = self.upstream(ctx)
        u_s = up_dist.sample((num_samples,))  # (S,B,1) or (S,B)
        if u_s.dim() == 2:
            u_s = u_s.unsqueeze(-1)

        S, B = u_s.shape[0], u_s.shape[1]
        ctx_rep = ctx.unsqueeze(0).expand(S, B, -1)  # (S,B,E)
        down_ctx = torch.cat([ctx_rep, u_s], dim=-1)  # (S,B,E+1)
        down_ctx_flat = down_ctx.reshape(S * B, -1)  # (S*B,E+1)
        down_dist = self.downstream(down_ctx_flat)
        d_flat = down_dist.sample()  # (S*B,1) or (S*B,)
        if d_flat.dim() == 1:
            d_flat = d_flat.unsqueeze(-1)
        d_s = d_flat.reshape(S, B, 1)

        u = u_s.squeeze(-1)  # (S,B)
        d = d_s.squeeze(-1)  # (S,B)
        return u, d


class LatentRollout(NamedTuple):
    u: torch.Tensor
    v: torch.Tensor
    upstream: torch.Tensor
    downstream: torch.Tensor


class LatentModel(nn.Module):
    """
    Latent true speeds + shared sensor-noise regime with per-source mixture weight
    and biased bad-sensor component.

    Generative:
      u ~ p(u | source)
      v ~ p(v | source, u)
      k ~ Bernoulli(pi(source))   # shared for both sensors on the row
      If k=0 (good):
        upstream_speed   ~ Normal(u + 0,      sigma0^2)
        downstream_speed ~ Normal(v + 0,      sigma0^2)
      If k=1 (bad, biased & high-variance):
        b ~ {+B, -B} with 50/50
        upstream_speed   ~ Normal(u + b,      sigma1^2)
        downstream_speed ~ Normal(v + b,      sigma1^2)

    Training:
      ELBO with factorized Gaussian encoder q(u|s,xu,xd) q(v|s,xu,xd).
    """

    def __init__(
        self,
        num_sources: int,
        emb_dim: int = 8,
        num_components: int = 2,
        encoder_hidden: int = 64,
    ):
        super().__init__()
        self.source_emb = nn.Embedding(num_sources, emb_dim)

        # Latent process model
        self.prior_u = GMM(
            features=1,
            context=emb_dim,
            components=num_components,
            covariance_type="diagonal",
            epsilon=1e-6,
        )
        self.prior_v = GMM(
            features=1,
            context=emb_dim + 1,
            components=num_components,
            covariance_type="diagonal",
            epsilon=1e-6,
        )

        # Encoder q(u, v | source, upstream_speed, downstream_speed)
        # Outputs: mu_u, log_std_u, mu_v, log_std_v
        self.encoder = nn.Sequential(
            nn.Linear(emb_dim + 2, encoder_hidden),
            nn.SiLU(),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.SiLU(),
            nn.Linear(encoder_hidden, 4),
        )

        # Sensor mixture parameters:
        #   pi(source) via an embedding -> logit
        #   sigma0, sigma1 are global learned scalars (log-stds)
        #   B (bias magnitude for bad regime) learned as positive via softplus
        self.sensor_logit = nn.Embedding(num_sources, 1)
        self.sensor_log_std = nn.Parameter(torch.zeros(2))  # (2,) for k in {0,1}
        self._bias_mag_raw = nn.Parameter(torch.tensor(2.0))  # initialize near generator's B

    def _encode(self, source: torch.Tensor, upstream_speed: torch.Tensor, downstream_speed: torch.Tensor):
        ctx = self.source_emb(source)  # (B, emb_dim)

        xu = upstream_speed.unsqueeze(-1)  # (B, 1)
        xd = downstream_speed.unsqueeze(-1)  # (B, 1)
        h = torch.cat([ctx, xu, xd], dim=-1)  # (B, emb_dim+2)

        params = self.encoder(h)  # (B, 4)
        mu_u, log_std_u, mu_v, log_std_v = params.chunk(4, dim=-1)

        std_u = torch.nn.functional.softplus(log_std_u) + 1e-4
        std_v = torch.nn.functional.softplus(log_std_v) + 1e-4

        q_u = torch.distributions.Normal(mu_u, std_u)  # event dim 1
        q_v = torch.distributions.Normal(mu_v, std_v)  # event dim 1
        return ctx, q_u, q_v

    def _sensor_joint_log_prob(
        self,
        source: torch.Tensor,
        u: torch.Tensor,  # (B,1)
        v: torch.Tensor,  # (B,1)
        upstream_speed: torch.Tensor,  # (B,)
        downstream_speed: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        # shared k for upstream + downstream:
        # log sum_k [ w_k(source) * p(xu, xd | k) ]
        # with:
        #   k=0: xu ~ N(u + 0,  sigma0^2), xd ~ N(v + 0,  sigma0^2)
        #   k=1: b in {+B, -B} 50/50, xu ~ N(u + b, sigma1^2), xd ~ N(v + b, sigma1^2)
        logit = self.sensor_logit(source).squeeze(-1)  # (B,)
        # Define pi = P(k=1 is bad). Then k=0 has weight (1-pi).
        log_w1 = torch.nn.functional.logsigmoid(logit)  # log pi (bad)
        log_w0 = torch.nn.functional.logsigmoid(-logit)  # log (1-pi) (good)

        xu = upstream_speed.unsqueeze(-1)  # (B,1)
        xd = downstream_speed.unsqueeze(-1)  # (B,1)

        log_std = self.sensor_log_std  # (2,)
        std0 = torch.exp(log_std[0])
        std1 = torch.exp(log_std[1])
        Bmag = torch.nn.functional.softplus(self._bias_mag_raw) + 1e-6

        # k = 0 (good): zero bias
        lp0 = torch.distributions.Normal(u + 0.0, std0).log_prob(xu) + torch.distributions.Normal(
            v + 0.0, std0
        ).log_prob(xd)  # (B,1)

        # k = 1 (bad): mixture over +/- B with equal weights
        lp1_plus = torch.distributions.Normal(u + Bmag, std1).log_prob(xu) + torch.distributions.Normal(
            v + Bmag, std1
        ).log_prob(xd)  # (B,1)
        lp1_minus = torch.distributions.Normal(u - Bmag, std1).log_prob(xu) + torch.distributions.Normal(
            v - Bmag, std1
        ).log_prob(xd)  # (B,1)
        lp1_mix = torch.logsumexp(
            torch.stack([lp1_plus, lp1_minus], dim=-1).squeeze(-2)  # (B,2)
            + torch.log(torch.tensor(0.5, device=u.device)),
            dim=-1,
        )  # (B,)

        lp0 = lp0.squeeze(-1) + log_w0  # (B,)
        lp1 = lp1_mix + log_w1  # (B,)

        return torch.logsumexp(torch.stack([lp0, lp1], dim=-1), dim=-1)  # (B,)

    def forward(self, source: torch.Tensor, upstream_speed: torch.Tensor, downstream_speed: torch.Tensor):
        ctx, q_u, q_v = self._encode(source, upstream_speed, downstream_speed)

        p_u = self.prior_u(ctx)  # p(u|source)
        return ctx, p_u, q_u, q_v

    def elbo(
        self,
        source: torch.Tensor,
        upstream_speed: torch.Tensor,
        downstream_speed: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        ctx, q_u, q_v = self._encode(source, upstream_speed, downstream_speed)

        p_u = self.prior_u(ctx)
        log_p_terms = []
        log_q_terms = []
        log_px_terms = []

        for _ in range(num_samples):
            u = q_u.rsample()  # (B,1)
            v = q_v.rsample()  # (B,1)

            p_v = self.prior_v(torch.cat([ctx, u], dim=-1))

            log_pu = p_u.log_prob(u).squeeze(-1)  # (B,)
            log_pv = p_v.log_prob(v).squeeze(-1)  # (B,)
            log_px = self._sensor_joint_log_prob(source, u, v, upstream_speed, downstream_speed)  # (B,)

            log_qu = q_u.log_prob(u).squeeze(-1)
            log_qv = q_v.log_prob(v).squeeze(-1)

            log_p_terms.append(log_pu + log_pv)
            log_q_terms.append(log_qu + log_qv)
            log_px_terms.append(log_px)

        log_p = torch.stack(log_p_terms, dim=0)  # (S,B)
        log_q = torch.stack(log_q_terms, dim=0)  # (S,B)
        log_px = torch.stack(log_px_terms, dim=0)  # (S,B)

        elbo = (log_p + log_px - log_q).mean(dim=0).mean()
        return elbo

    def loss(
        self,
        source: torch.Tensor,
        upstream_speed: torch.Tensor,
        downstream_speed: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        return -self.elbo(source, upstream_speed, downstream_speed, num_samples=num_samples)

    @torch.no_grad()
    def log_prob(
        self,
        source: torch.Tensor,
        upstream_speed: torch.Tensor,
        downstream_speed: torch.Tensor,
        num_samples: int = 64,
    ) -> torch.Tensor:
        # IWAE estimate: log p(x) ~= log mean_s exp(log p(u,v,x) - log q(u,v|x))
        ctx, q_u, q_v = self._encode(source, upstream_speed, downstream_speed)
        p_u = self.prior_u(ctx)

        ws = []
        for _ in range(num_samples):
            u = q_u.rsample()
            v = q_v.rsample()
            p_v = self.prior_v(torch.cat([ctx, u], dim=-1))

            log_pu = p_u.log_prob(u).squeeze(-1)
            log_pv = p_v.log_prob(v).squeeze(-1)
            log_px = self._sensor_joint_log_prob(source, u, v, upstream_speed, downstream_speed)

            log_qu = q_u.log_prob(u).squeeze(-1)
            log_qv = q_v.log_prob(v).squeeze(-1)

            ws.append((log_pu + log_pv + log_px) - (log_qu + log_qv))

        w = torch.stack(ws, dim=0)  # (S,B)
        return torch.logsumexp(w, dim=0) - torch.log(torch.tensor(float(num_samples), device=w.device))  # (B,)

    @torch.no_grad()
    def sample(self, source: torch.Tensor, num_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rollout samples of observed sensors using the latent process + shared sensor regime.

        Returns (upstream_hat, downstream_hat) with shape (S, B).
        """
        ctx = self.source_emb(source)  # (B,E)

        # Sample latents
        p_u = self.prior_u(ctx)
        u_s = p_u.sample((num_samples,))  # (S,B,1) or (S,B)
        if u_s.dim() == 2:
            u_s = u_s.unsqueeze(-1)

        S, B = u_s.shape[0], u_s.shape[1]
        ctx_rep = ctx.unsqueeze(0).expand(S, B, -1)  # (S,B,E)
        pv_ctx = torch.cat([ctx_rep, u_s], dim=-1)
        pv_ctx_flat = pv_ctx.reshape(S * B, -1)
        p_v = self.prior_v(pv_ctx_flat)
        v_flat = p_v.sample()  # (S*B,1) or (S*B,)
        if v_flat.dim() == 1:
            v_flat = v_flat.unsqueeze(-1)
        v_s = v_flat.reshape(S, B, 1)

        # Shared sensor regime per (s, b)
        logit = self.sensor_logit(source).squeeze(-1)  # (B,)
        pi = torch.sigmoid(logit)
        k = torch.bernoulli(pi.unsqueeze(0).expand(S, -1)).long()  # (S,B)

        std = torch.exp(self.sensor_log_std)  # (2,)
        sigma = std[k]  # (S,B)

        # Bias magnitude for bad regime, with 50/50 sign
        Bmag = torch.nn.functional.softplus(self._bias_mag_raw) + 1e-6
        sign = torch.where(torch.rand_like(sigma) < 0.5, 1.0, -1.0)
        bias = torch.where(k == 1, sign * Bmag, torch.zeros_like(sigma))  # (S,B)

        eps_up = torch.randn_like(sigma)
        eps_down = torch.randn_like(sigma)
        up_s = u_s.squeeze(-1) + bias + sigma * eps_up  # (S,B)
        down_s = v_s.squeeze(-1) + bias + sigma * eps_down  # (S,B)

        return up_s, down_s
