import torch, torch.nn as nn, torch.nn.functional as F

class HFAM(nn.Module):
    def __init__(self, dim: int, reduction: int = 16, init_gauss=True):
        super().__init__()
        self.k_lp = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        if init_gauss:
            g = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32)
            g = F.pad(g, (1,1,1,1)) / g.sum()
            self.k_lp.weight.data.copy_(g.expand(dim,1,5,5))
        else:
            nn.init.dirac_(self.k_lp.weight)

        self.hpm = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1), nn.ReLU(True),
            nn.Conv2d(dim // reduction, dim * 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lf = self.k_lp(x)
        hf = x - lf
        γβ = self.hpm(hf.mean([2, 3], keepdim=True))
        γ, β = γβ.chunk(2, dim=1)
        return x + γ * hf + β
