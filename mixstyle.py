import random
import torch
import torch.nn as nn

class MixStyle(nn.Module):
    """MixStyle: Domain Generalization with MixStyle. ICLR 2021.
    Adapted from: https://github.com/KaiyangZhou/Dassl.pytorch
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x, labels=None):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)

        if self.mix == "random":
            perm = torch.randperm(B)

        elif self.mix == "crossdomain":
            if B % 2 != 0:
                raise ValueError("Batch size must be even for crossdomain mixing.")
            perm = torch.arange(B - 1, -1, -1)
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(len(perm_b))]
            perm_a = perm_a[torch.randperm(len(perm_a))]
            perm = torch.cat([perm_b, perm_a], dim=0)

        elif self.mix == "crosssample":
            if labels is None:
                raise ValueError("Labels are required for crosssample mixing.")
            contrast_bf = (labels.long() == 1).nonzero(as_tuple=True)[0]
            contrast_attack = (labels.long() == 0).nonzero(as_tuple=True)[0]
            perm_idx_bf = contrast_bf[torch.randperm(len(contrast_bf))]
            perm_idx_attack = contrast_attack[torch.randperm(len(contrast_attack))]
            old_idx = torch.cat([contrast_bf, contrast_attack], 0)
            perm = torch.cat([perm_idx_bf, perm_idx_attack], 0)
            perm = perm[torch.argsort(old_idx)]

        else:
            raise NotImplementedError(f"Mix method '{self.mix}' is not implemented.")

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed*sig_mix + mu_mix
    
