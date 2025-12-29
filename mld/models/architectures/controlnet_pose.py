import torch
import torch.nn as nn


class ControlNetPose(nn.Module):
    def __init__(self, njoints, latent_dim, hidden_dim=512, scale=1.0):
        super().__init__()
        if not isinstance(latent_dim, int):
            latent_dim = int(latent_dim[-1])
        self.njoints = njoints
        self.latent_dim = latent_dim
        self.scale = scale
        input_dim = njoints * 3
        self.pose_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, hint):
        if hint is None:
            return None
        # hint: [bs, 2, njoints, 3] or [bs, 2, njoints*3]
        if hint.dim() == 4:
            hint = hint.reshape(hint.shape[0], hint.shape[1], -1)
        if hint.dim() != 3:
            raise ValueError("hint must be [bs, 2, njoints, 3] or [bs, 2, njoints*3]")
        embeds = self.pose_mlp(hint)
        embeds = embeds * self.scale
        return embeds.permute(1, 0, 2)
