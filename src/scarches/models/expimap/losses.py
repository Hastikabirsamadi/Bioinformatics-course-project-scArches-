from scipy.special import gamma
import torch

# Adapted from
# Title: Information Constraints on Auto-Encoding Variational Bayes
# Authors: Romain Lopez, Jeffrey Regier, Nir Yosef, Michael I. Jordan
# Code: https://github.com/romain-lopez/HCV/blob/master/hcv.py

def kernel_matrix(x: torch.Tensor, sigma):
    x1  = torch.unsqueeze(x, 0)
    x2  = torch.unsqueeze(x, 1)

    return torch.exp( -sigma * torch.sum(torch.pow(x1-x2, 2), axis=2) )

def bandwidth(d):
    """
    in the case of Gaussian random variables and the use of a RBF kernel,
    this can be used to select the bandwidth according to the median heuristic
    """
    gz = 2 * gamma(0.5 * (d+1)) / gamma(0.5 * d)
    return 1. / (2. * gz**2)

def hsic(z, s):
    d_z = z.shape[1]
    d_s = s.shape[1]

    zz = kernel_matrix(z, bandwidth(d_z))
    ss = kernel_matrix(s, bandwidth(d_s))

    h  = (zz * ss).mean() + zz.mean() * ss.mean() - 2 * torch.mean(zz.mean(1) * ss.mean(1))
    return h.sqrt()


def correlation_penalty(z: torch.Tensor) -> torch.Tensor:
    # Sum of squared off-diagonal entries of the covariance matrix of z ([batch_size, n_latent]).
    n, d = z.shape
    if n < 2 or d < 2:
        return z.new_zeros(())
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / (n - 1)
    off = cov - torch.diag(cov.diagonal())
    return (off ** 2).sum()
