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


class EvalArrays(NamedTuple):
    nll_up: torch.Tensor  # (B,)
    nll_down: torch.Tensor  # (B,)
    crps_up: torch.Tensor  # (B,)
    crps_down: torch.Tensor  # (B,)
    up_hat: torch.Tensor  # (B,)
    down_hat: torch.Tensor  # (B,)


class FitArrays(NamedTuple):
    # Teacher-forced fit arrays
    nll_up: torch.Tensor  # (B,)
    nll_down_tf: torch.Tensor  # (B,)
    crps_up: torch.Tensor  # (B,)
    crps_down_tf: torch.Tensor  # (B,)
    # Names for metric estimators (e.g., "nll", "nll_mc", "nll_is")
    nll_up_name: str
    nll_down_tf_name: str
    crps_up_name: str
    crps_down_tf_name: str


class RolloutArrays(NamedTuple):
    up_hat: torch.Tensor  # (B,)
    down_hat: torch.Tensor  # (B,)


def _crps_mc(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Monte Carlo CRPS for 1D predictive distribution.

    Args:
      samples: (S, B) tensor of samples
      target: (B,) tensor of observed values

    Returns:
      (B,) tensor with CRPS per row
    """
    mean_abs = (samples - target.unsqueeze(0)).abs().mean(dim=0)  # (B,)

    S = samples.size(0)
    s_sorted, _ = samples.sort(dim=0)
    idx = torch.arange(1, S + 1, device=samples.device, dtype=samples.dtype).unsqueeze(1)  # (S,1)
    w = 2 * idx - (S + 1)
    pairwise_mean = (2.0 / (S * S)) * (w * s_sorted).sum(dim=0)  # (B,)
    crps = mean_abs - 0.5 * pairwise_mean
    return crps


def _gmm_1d_log_prob(dist: Distribution, x: torch.Tensor, dim_index: int) -> torch.Tensor:
    """
    1D marginal log-prob for a zuko/torch MixtureSameFamily whose components have event dim D.
      dist: MixtureSameFamily (e.g., from zuko.mixtures.GMM(...)(...))
      x: (B,)
      dim_index: dimension to marginalize (0..D-1)
    Returns:
      (B,)
    """
    base = dist.base
    logit_w = dist.logits  # (B,K)
    mu = base.loc[..., dim_index]  # (B,K)
    var = base.covariance_matrix[..., dim_index, dim_index]
    std = var.clamp_min(1e-12).sqrt()  # (B,K)

    x_ = x.unsqueeze(-1)  # (B,1)
    comp_lp = torch.distributions.Normal(mu, std).log_prob(x_)  # (B,K)
    return torch.logsumexp(logit_w + comp_lp, dim=-1)  # (B,)


def _gmm_conditional_1d(
    dist: Distribution,
    given_x: torch.Tensor,
    *,
    given_dim: int,
    target_dim: int,
) -> Distribution:
    """
    Build the conditional 1D mixture p(x_target | x_given) for a 2D full-covariance GMM.

    Args:
      dist: MixtureSameFamily with MultivariateNormal components (event dim=2)
      given_x: (B,) observed value for the given_dim
      given_dim: index of the dimension to condition on (0 or 1)
      target_dim: index of the target dimension (0 or 1), target_dim != given_dim

    Returns:
      MixtureSameFamily over Normal with batch (B,) and event dim scalar representing
      the conditional distribution.
    """
    base = dist.base
    logit_w = dist.logits
    mu = base.loc  # (B,K,2)
    cov = base.covariance_matrix  # (B,K,2,2)

    mu_u = mu[..., given_dim]  # (B,K)
    mu_d = mu[..., target_dim]  # (B,K)
    S_uu = cov[..., given_dim, given_dim]  # (B,K)
    S_dd = cov[..., target_dim, target_dim]  # (B,K)
    S_du = cov[..., target_dim, given_dim]  # (B,K)
    # S_ud = cov[..., given_dim, target_dim]

    x = given_x.unsqueeze(-1)  # (B,1)
    eps = 1e-8
    inv_Suu = 1.0 / (S_uu + eps)

    # Component-wise conditional parameters
    mu_cond = mu_d + S_du * inv_Suu * (x - mu_u)  # (B,K)
    var_cond = (S_dd - S_du * S_du * inv_Suu).clamp_min(1e-12)  # (B,K)
    std_cond = var_cond.sqrt()

    # Posterior component weights w'(k | x_given)
    like_log = torch.distributions.Normal(mu_u, S_uu.clamp_min(1e-12).sqrt()).log_prob(given_x.unsqueeze(-1))  # (B,K)
    # Posterior over components: softmax(logits + log_like)
    w_post = torch.softmax(logit_w + like_log, dim=-1)

    comp_1d = torch.distributions.Normal(mu_cond, std_cond)  # (B,K)
    mix_post = torch.distributions.Categorical(probs=w_post)
    return torch.distributions.MixtureSameFamily(mix_post, comp_1d)


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
    def eval_fit_arrays(
        self,
        source: torch.Tensor,
        upstream_speed: torch.Tensor,
        downstream_speed: torch.Tensor,
        *,
        crps_samples: int = 64,
    ) -> FitArrays:
        dist = self(source)

        # Upstream marginal
        nll_up = -_gmm_1d_log_prob(dist, upstream_speed, dim_index=0)
        samples = dist.sample((crps_samples,))  # (S,B,2)
        up_s = samples[..., 0]  # (S,B)
        crps_up = _crps_mc(up_s, upstream_speed)

        # Downstream conditional p(xd | s, xu)
        cond = _gmm_conditional_1d(dist, upstream_speed, given_dim=0, target_dim=1)
        nll_down_tf = -cond.log_prob(downstream_speed)
        down_s = cond.sample((crps_samples,))  # (S,B)
        crps_down_tf = _crps_mc(down_s, downstream_speed)

        return FitArrays(
            nll_up=nll_up,
            nll_down_tf=nll_down_tf,
            crps_up=crps_up,
            crps_down_tf=crps_down_tf,
            nll_up_name="nll",
            nll_down_tf_name="nll",
            crps_up_name="crps",
            crps_down_tf_name="crps",
        )

    @torch.no_grad()
    def eval_rollout_arrays(
        self,
        source: torch.Tensor,
        *,
        num_samples: int = 1,
    ) -> RolloutArrays:
        dist = self(source)
        y_hat = dist.sample()  # (B,2)
        up_hat = y_hat[..., 0]
        down_hat = y_hat[..., 1]
        return RolloutArrays(up_hat=up_hat, down_hat=down_hat)

    @torch.no_grad()
    def eval_arrays(
        self,
        source: torch.Tensor,
        upstream_speed: torch.Tensor,
        downstream_speed: torch.Tensor,
        *,
        crps_samples: int = 64,
    ) -> EvalArrays:
        dist = self(source)

        lp_up = _gmm_1d_log_prob(dist, upstream_speed, dim_index=0)
        lp_down = _gmm_1d_log_prob(dist, downstream_speed, dim_index=1)
        nll_up = -lp_up
        nll_down = -lp_down

        samples = dist.sample((crps_samples,))  # (S,B,2)
        up_s = samples[..., 0]  # (S,B)
        down_s = samples[..., 1]
        crps_up = _crps_mc(up_s, upstream_speed)
        crps_down = _crps_mc(down_s, downstream_speed)

        y_hat = dist.sample()  # (B,2)
        up_hat = y_hat[..., 0]
        down_hat = y_hat[..., 1]

        return EvalArrays(
            nll_up=nll_up,
            nll_down=nll_down,
            crps_up=crps_up,
            crps_down=crps_down,
            up_hat=up_hat,
            down_hat=down_hat,
        )


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
    def eval_fit_arrays(
        self,
        source: torch.Tensor,
        upstream_speed: torch.Tensor,
        downstream_speed: torch.Tensor,
        *,
        crps_samples: int = 64,
    ) -> FitArrays:
        up_dist, down_dist = self(source, upstream_speed)

        nll_up = -up_dist.log_prob(upstream_speed.unsqueeze(-1)).squeeze(-1)
        nll_down_tf = -down_dist.log_prob(downstream_speed.unsqueeze(-1)).squeeze(-1)

        up_s = up_dist.sample((crps_samples,)).squeeze(-1)  # (S,B)
        down_s = down_dist.sample((crps_samples,)).squeeze(-1)
        crps_up = _crps_mc(up_s, upstream_speed)
        crps_down_tf = _crps_mc(down_s, downstream_speed)

        return FitArrays(
            nll_up=nll_up,
            nll_down_tf=nll_down_tf,
            crps_up=crps_up,
            crps_down_tf=crps_down_tf,
            nll_up_name="nll",
            nll_down_tf_name="nll",
            crps_up_name="crps",
            crps_down_tf_name="crps",
        )

    @torch.no_grad()
    def eval_rollout_arrays(
        self,
        source: torch.Tensor,
        *,
        num_samples: int = 1,
    ) -> RolloutArrays:
        ro = self.rollout(source)
        up_hat = ro.upstream_sample.squeeze(-1)
        down_hat = ro.downstream_sample.squeeze(-1)
        return RolloutArrays(up_hat=up_hat, down_hat=down_hat)

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
    def eval_arrays(
        self,
        source: torch.Tensor,
        upstream_speed: torch.Tensor,
        downstream_speed: torch.Tensor,
        *,
        crps_samples: int = 64,
    ) -> EvalArrays:
        up_dist, down_dist = self(source, upstream_speed)

        nll_up = -up_dist.log_prob(upstream_speed.unsqueeze(-1)).squeeze(-1)
        nll_down = -down_dist.log_prob(downstream_speed.unsqueeze(-1)).squeeze(-1)

        up_s = up_dist.sample((crps_samples,)).squeeze(-1)  # (S,B)
        down_s = down_dist.sample((crps_samples,)).squeeze(-1)
        crps_up = _crps_mc(up_s, upstream_speed)
        crps_down = _crps_mc(down_s, downstream_speed)

        ro = self.rollout(source)
        up_hat = ro.upstream_sample.squeeze(-1)
        down_hat = ro.downstream_sample.squeeze(-1)

        return EvalArrays(
            nll_up=nll_up,
            nll_down=nll_down,
            crps_up=crps_up,
            crps_down=crps_down,
            up_hat=up_hat,
            down_hat=down_hat,
        )


class LatentRollout(NamedTuple):
    u: torch.Tensor
    v: torch.Tensor
    upstream: torch.Tensor
    downstream: torch.Tensor


class LatentModel(nn.Module):
    """
    Latent true speeds + shared sensor-noise regime.

    Generative:
      u ~ p(u | source)
      v ~ p(v | source, u)
      k ~ Bernoulli(pi(source))   # shared for both sensors on the row
      upstream_speed   ~ Normal(u, sigma_k^2)
      downstream_speed ~ Normal(v, sigma_k^2)

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
        self.sensor_logit = nn.Embedding(num_sources, 1)
        self.sensor_log_std = nn.Parameter(torch.zeros(2))  # (2,) for k in {0,1}

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
        # log sum_k [ w_k(source) * N(xu; u, s_k^2) * N(xd; v, s_k^2) ]
        logit = self.sensor_logit(source).squeeze(-1)  # (B,)
        log_w0 = torch.nn.functional.logsigmoid(logit)  # log pi
        log_w1 = torch.nn.functional.logsigmoid(-logit)  # log (1-pi)

        xu = upstream_speed.unsqueeze(-1)  # (B,1)
        xd = downstream_speed.unsqueeze(-1)  # (B,1)

        log_std = self.sensor_log_std  # (2,)
        std0 = torch.exp(log_std[0])
        std1 = torch.exp(log_std[1])

        lp0 = torch.distributions.Normal(u, std0).log_prob(xu) + torch.distributions.Normal(v, std0).log_prob(
            xd
        )  # (B,1)
        lp1 = torch.distributions.Normal(u, std1).log_prob(xu) + torch.distributions.Normal(v, std1).log_prob(
            xd
        )  # (B,1)

        lp0 = lp0.squeeze(-1) + log_w0
        lp1 = lp1.squeeze(-1) + log_w1

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
    def sample(self, source: torch.Tensor) -> LatentRollout:
        ctx = self.source_emb(source)

        p_u = self.prior_u(ctx)
        u = p_u.sample()
        if u.ndim == 1:
            u = u.unsqueeze(-1)

        p_v = self.prior_v(torch.cat([ctx, u], dim=-1))
        v = p_v.sample()
        if v.ndim == 1:
            v = v.unsqueeze(-1)

        logit = self.sensor_logit(source).squeeze(-1)  # (B,)
        pi = torch.sigmoid(logit)
        k = torch.bernoulli(pi).long()  # (B,)

        std = torch.exp(self.sensor_log_std)  # (2,)
        sigma = std[k].unsqueeze(-1)  # (B,1)

        upstream = (u + sigma * torch.randn_like(u)).squeeze(-1)  # (B,)
        downstream = (v + sigma * torch.randn_like(v)).squeeze(-1)  # (B,)

        return LatentRollout(
            u=u.squeeze(-1),
            v=v.squeeze(-1),
            upstream=upstream,
            downstream=downstream,
        )

    @torch.no_grad()
    def _log_prob_up_marginal(self, source: torch.Tensor, xu: torch.Tensor, *, num_samples: int = 64) -> torch.Tensor:
        ctx = self.source_emb(source)
        p_u = self.prior_u(ctx)

        u = p_u.sample((num_samples,))  # (S,B,1) or (S,B)
        if u.dim() == 2:
            u = u.unsqueeze(-1)

        logit = self.sensor_logit(source).squeeze(-1)  # (B,)
        log_w0 = torch.nn.functional.logsigmoid(logit)
        log_w1 = torch.nn.functional.logsigmoid(-logit)

        std = torch.exp(self.sensor_log_std)  # (2,)
        s0, s1 = std[0], std[1]

        xu_ = xu.unsqueeze(0).unsqueeze(-1)  # (1,B,1)
        lp0 = torch.distributions.Normal(u, s0).log_prob(xu_)  # (S,B,1)
        lp1 = torch.distributions.Normal(u, s1).log_prob(xu_)
        lp_mix = torch.logsumexp(
            torch.stack([lp0.squeeze(-1) + log_w0, lp1.squeeze(-1) + log_w1], dim=-1),
            dim=-1,
        )  # (S,B)

        return torch.logsumexp(lp_mix, dim=0) - torch.log(torch.tensor(float(num_samples), device=xu.device))

    @torch.no_grad()
    def _log_prob_down_marginal(self, source: torch.Tensor, xd: torch.Tensor, *, num_samples: int = 64) -> torch.Tensor:
        ctx = self.source_emb(source)
        p_u = self.prior_u(ctx)

        u = p_u.sample((num_samples,))  # (S,B,1) or (S,B)
        if u.dim() == 2:
            u = u.unsqueeze(-1)

        S, B = u.shape[0], u.shape[1]
        ctx_rep = ctx.unsqueeze(0).expand(S, B, -1)  # (S,B,E)
        pv_ctx = torch.cat([ctx_rep, u], dim=-1)  # (S,B,E+1)
        pv_ctx_flat = pv_ctx.reshape(S * B, -1)
        p_v = self.prior_v(pv_ctx_flat)
        v = p_v.sample().reshape(S, B, 1)

        logit = self.sensor_logit(source).squeeze(-1)  # (B,)
        log_w0 = torch.nn.functional.logsigmoid(logit)
        log_w1 = torch.nn.functional.logsigmoid(-logit)

        std = torch.exp(self.sensor_log_std)
        s0, s1 = std[0], std[1]

        xd_ = xd.unsqueeze(0).unsqueeze(-1)  # (1,B,1)
        lp0 = torch.distributions.Normal(v, s0).log_prob(xd_)  # (S,B,1)
        lp1 = torch.distributions.Normal(v, s1).log_prob(xd_)
        lp_mix = torch.logsumexp(
            torch.stack([lp0.squeeze(-1) + log_w0, lp1.squeeze(-1) + log_w1], dim=-1),
            dim=-1,
        )  # (S,B)

        return torch.logsumexp(lp_mix, dim=0) - torch.log(torch.tensor(float(num_samples), device=xd.device))

    @torch.no_grad()
    def _samples_up_marginal(self, source: torch.Tensor, *, num_samples: int) -> torch.Tensor:
        ctx = self.source_emb(source)
        p_u = self.prior_u(ctx)
        u = p_u.sample((num_samples,))  # (S,B,1) or (S,B)
        if u.dim() == 2:
            u = u.unsqueeze(-1)

        logit = self.sensor_logit(source).squeeze(-1)  # (B,)
        pi = torch.sigmoid(logit)
        k = torch.bernoulli(pi.unsqueeze(0).expand(num_samples, -1)).long()  # (S,B)

        std = torch.exp(self.sensor_log_std)  # (2,)
        sigma = std[k]  # (S,B)
        eps = torch.randn_like(sigma)
        return u.squeeze(-1) + sigma * eps  # (S,B)

    @torch.no_grad()
    def _samples_down_marginal(self, source: torch.Tensor, *, num_samples: int) -> torch.Tensor:
        ctx = self.source_emb(source)
        p_u = self.prior_u(ctx)
        u = p_u.sample((num_samples,))  # (S,B,1) or (S,B)
        if u.dim() == 2:
            u = u.unsqueeze(-1)

        S, B = u.shape[0], u.shape[1]
        ctx_rep = ctx.unsqueeze(0).expand(S, B, -1)  # (S,B,E)
        pv_ctx = torch.cat([ctx_rep, u], dim=-1)  # (S,B,E+1)
        pv_ctx_flat = pv_ctx.reshape(S * B, -1)
        p_v = self.prior_v(pv_ctx_flat)
        v = p_v.sample().reshape(S, B)  # (S,B)

        logit = self.sensor_logit(source).squeeze(-1)  # (B,)
        pi = torch.sigmoid(logit)
        k = torch.bernoulli(pi.unsqueeze(0).expand(num_samples, -1)).long()  # (S,B)

        std = torch.exp(self.sensor_log_std)
        sigma = std[k]
        eps = torch.randn_like(sigma)
        return v + sigma * eps  # (S,B)

    @torch.no_grad()
    def _crps_weighted(self, samples: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Weighted Monte Carlo CRPS for 1D predictive distribution with normalized weights.

        Args:
          samples: (S,B)
          target: (B,)
          weights: (S,B) non-negative, will be normalized across S
        Returns:
          (B,)
        """
        w = weights / (weights.sum(dim=0, keepdim=True) + 1e-12)  # (S,B)
        mean_abs = (w * (samples - target.unsqueeze(0)).abs()).sum(dim=0)  # (B,)
        # Pairwise term: 0.5 * E_w|X - X'|
        # Compute using broadcasting with care (S,S,B)
        S = samples.size(0)
        a = samples.unsqueeze(1)  # (S,1,B)
        b = samples.unsqueeze(0)  # (1,S,B)
        w_i = w.unsqueeze(1)  # (1,S,B) after broadcast
        w_j = w.unsqueeze(0)  # (S,1,B) after broadcast
        pair = (w_i * w_j * (a - b).abs()).sum(dim=(0, 1))  # (B,)
        return mean_abs - 0.5 * pair

    @torch.no_grad()
    def _downstream_conditional_samples_is(
        self,
        source: torch.Tensor,
        upstream_speed: torch.Tensor,
        *,
        num_samples: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Importance sampling for p(xd | s, xu):
          draw (u,v,k) ~ p(u|s) p(v|s,u) p(k|s),
          weight by p(xu | u, k),
          then draw xd ~ Normal(v, sigma_k).

        Returns:
          samples: (S,B)
          weights: (S,B) (unnormalized)
        """
        ctx = self.source_emb(source)

        p_u = self.prior_u(ctx)
        u = p_u.sample((num_samples,))  # (S,B,1) or (S,B)
        if u.dim() == 2:
            u = u.unsqueeze(-1)

        S_, B = u.shape[0], u.shape[1]
        ctx_rep = ctx.unsqueeze(0).expand(S_, B, -1)  # (S,B,E)
        pv_ctx = torch.cat([ctx_rep, u], dim=-1)
        pv_ctx_flat = pv_ctx.reshape(S_ * B, -1)
        p_v = self.prior_v(pv_ctx_flat)
        v = p_v.sample().reshape(S_, B, 1)  # (S,B,1)

        logit = self.sensor_logit(source).squeeze(-1)  # (B,)
        pi = torch.sigmoid(logit)
        k = torch.bernoulli(pi.unsqueeze(0).expand(S_, -1)).long()  # (S,B)

        std = torch.exp(self.sensor_log_std)
        sigma = std[k]  # (S,B)

        xu = upstream_speed.unsqueeze(0).unsqueeze(-1)  # (1,B,1)
        log_w = torch.distributions.Normal(u, sigma.unsqueeze(-1)).log_prob(xu).squeeze(-1)  # (S,B)
        w = torch.exp(log_w - log_w.max(dim=0, keepdim=True).values)  # stabilize

        xd_samples = v.squeeze(-1) + sigma * torch.randn_like(sigma)  # (S,B)
        return xd_samples, w

    @torch.no_grad()
    def eval_fit_arrays(
        self,
        source: torch.Tensor,
        upstream_speed: torch.Tensor,
        downstream_speed: torch.Tensor,
        *,
        crps_samples: int = 64,
    ) -> FitArrays:
        # Upstream marginal (MC)
        lp_up = self._log_prob_up_marginal(source, upstream_speed, num_samples=crps_samples)
        nll_up = -lp_up
        up_s = self._samples_up_marginal(source, num_samples=crps_samples)  # (S,B)
        crps_up = _crps_mc(up_s, upstream_speed)

        # Downstream conditional (IS via joint - marginal trick)
        lp_joint = self.log_prob(source, upstream_speed, downstream_speed, num_samples=crps_samples)
        lp_up_m = lp_up  # reuse same S for simplicity
        lp_down_cond = lp_joint - lp_up_m
        nll_down_tf = -lp_down_cond

        # CRPS for conditional via IS weighting
        xd_s, w = self._downstream_conditional_samples_is(
            source, upstream_speed, num_samples=crps_samples
        )  # (S,B), (S,B)
        crps_down_tf = self._crps_weighted(xd_s, downstream_speed, w)

        return FitArrays(
            nll_up=nll_up,
            nll_down_tf=nll_down_tf,
            crps_up=crps_up,
            crps_down_tf=crps_down_tf,
            nll_up_name="nll_mc",
            nll_down_tf_name="nll_is",
            crps_up_name="crps_mc",
            crps_down_tf_name="crps_is",
        )

    @torch.no_grad()
    def eval_rollout_arrays(
        self,
        source: torch.Tensor,
        *,
        num_samples: int = 1,
    ) -> RolloutArrays:
        ro = self.sample(source)
        return RolloutArrays(up_hat=ro.upstream, down_hat=ro.downstream)

    @torch.no_grad()
    def eval_arrays(
        self,
        source: torch.Tensor,
        upstream_speed: torch.Tensor,
        downstream_speed: torch.Tensor,
        *,
        crps_samples: int = 64,
    ) -> EvalArrays:
        lp_up = self._log_prob_up_marginal(source, upstream_speed, num_samples=crps_samples)
        lp_down = self._log_prob_down_marginal(source, downstream_speed, num_samples=crps_samples)
        nll_up = -lp_up
        nll_down = -lp_down

        up_s = self._samples_up_marginal(source, num_samples=crps_samples)  # (S,B)
        down_s = self._samples_down_marginal(source, num_samples=crps_samples)  # (S,B)
        crps_up = _crps_mc(up_s, upstream_speed)
        crps_down = _crps_mc(down_s, downstream_speed)

        ro = self.sample(source)
        up_hat = ro.upstream
        down_hat = ro.downstream

        return EvalArrays(
            nll_up=nll_up,
            nll_down=nll_down,
            crps_up=crps_up,
            crps_down=crps_down,
            up_hat=up_hat,
            down_hat=down_hat,
        )
