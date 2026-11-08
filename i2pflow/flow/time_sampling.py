"""Training-time distributions over tau (tau=1 is data)."""
import torch


def sample_tau(shape, device, kind="mixture", t_min=0.01, t_max=0.99, uniform_frac=0.2,
               beta_a=1.9, beta_b=1.0, logit_mean=0.0, logit_std=1.0):
    shape = tuple(shape)
    if kind == "uniform":
        tau = torch.rand(shape, device=device)
    elif kind == "mixture":  # Proteina-style: uniform + Beta(1.9, 1) biased toward data
        beta = torch.distributions.Beta(torch.tensor(beta_a, device=device), torch.tensor(beta_b, device=device))
        tau_b = beta.sample(shape)
        use_u = torch.rand(shape, device=device) < uniform_frac
        tau = torch.where(use_u, torch.rand(shape, device=device), tau_b)
    elif kind == "logit_normal":  # SD3-style
        tau = torch.sigmoid(torch.randn(shape, device=device) * logit_std + logit_mean)
    else:
        raise ValueError(kind)
    return tau.clamp(t_min, t_max)
