import torch
import torch.nn as nn
from mld.models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from mld.models.operator import PositionalEncoding
from mld.models.operator.cross_attention import (
    SkipTransformerEncoder,
    TransformerEncoderLayer,
)
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask


class ControlNetDenoiser(nn.Module):
    def __init__(
        self,
        ablation,
        nfeats: int = 263,
        condition: str = "text",
        latent_dim: list = [1, 256],
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        normalize_before: bool = False,
        activation: str = "gelu",
        flip_sin_to_cos: bool = True,
        position_embedding: str = "learned",
        arch: str = "trans_enc",
        freq_shift: int = 0,
        guidance_scale: float = 7.5,
        guidance_uncondp: float = 0.1,
        text_encoded_dim: int = 768,
        nclasses: int = 10,
        **kwargs
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim[-1]
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.abl_plus = False
        self.ablation_skip_connection = ablation.SKIP_CONNECT
        self.diffusion_only = ablation.VAE_TYPE == "no"
        self.arch = arch
        self.pe_type = ablation.DIFF_PE_TYPE

        if self.diffusion_only:
            self.pose_embd = nn.Linear(nfeats, self.latent_dim)

        if self.condition in ["text", "text_uncond"]:
            self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos, freq_shift)
            self.time_embedding = TimestepEmbedding(text_encoded_dim, self.latent_dim)
            if text_encoded_dim != self.latent_dim:
                self.emb_proj = nn.Sequential(nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))
        elif self.condition in ["action"]:
            self.time_proj = Timesteps(self.latent_dim, flip_sin_to_cos, freq_shift)
            self.time_embedding = TimestepEmbedding(self.latent_dim, self.latent_dim)
            self.emb_proj = EmbedAction(
                nclasses,
                self.latent_dim,
                guidance_scale=guidance_scale,
                guidance_uncodp=guidance_uncondp,
            )
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        if self.pe_type == "actor":
            self.query_pos = PositionalEncoding(self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.query_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding
            )
        else:
            raise ValueError("Not Support PE type")

        if self.arch != "trans_enc":
            raise ValueError("ControlNetDenoiser only supports trans_enc")

        self.control_proj = nn.Linear(self.latent_dim, self.latent_dim)

        if self.ablation_skip_connection:
            encoder_layer = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            encoder_norm = nn.LayerNorm(self.latent_dim)
            self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.num_layers = num_layers

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        lengths=None,
        controlnet_cond=None,
        **kwargs
    ):
        sample = sample.permute(1, 0, 2)
        if lengths not in [None, []]:
            mask = lengths_to_mask(lengths, sample.device)
        else:
            mask = None

        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        if self.condition in ["text", "text_uncond"]:
            encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
            text_emb = encoder_hidden_states
            if self.text_encoded_dim != self.latent_dim:
                text_emb_latent = self.emb_proj(text_emb)
            else:
                text_emb_latent = text_emb
            if self.abl_plus:
                emb_latent = time_emb + text_emb_latent
            else:
                emb_latent = torch.cat((time_emb, text_emb_latent), 0)
        elif self.condition in ["action"]:
            action_emb = self.emb_proj(encoder_hidden_states)
            if self.abl_plus:
                emb_latent = action_emb + time_emb
            else:
                emb_latent = torch.cat((time_emb, action_emb), 0)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        if controlnet_cond is not None:
            if controlnet_cond.dim() != 3:
                raise ValueError("controlnet_cond must be [B,2,D] or [2,B,D]")
            # normalize to [n_tokens, B, D]
            if controlnet_cond.shape[0] == sample.shape[1]:
                controlnet_cond = controlnet_cond.permute(1, 0, 2)
            elif controlnet_cond.shape[1] != sample.shape[1]:
                raise ValueError("controlnet_cond first two dims must match control tokens and batch size")
            control_tokens = self.control_proj(controlnet_cond)
            emb_latent = torch.cat((emb_latent, control_tokens), 0)

        if self.diffusion_only:
            sample = self.pose_embd(sample)
            xseq = torch.cat((emb_latent, sample), axis=0)
            motion_first = False
        else:
            xseq = torch.cat((sample, emb_latent), axis=0)
            motion_first = True

        # zero padded positions so residuals do not leak padding info
        valid_mask = None
        if mask is not None:
            pad_len = xseq.shape[0] - emb_latent.shape[0]
            if pad_len == mask.T.shape[0]:
                # emb_latent tokens are always valid (ones), mask covers motion tokens
                ones = torch.ones(emb_latent.shape[0], mask.T.shape[1], device=mask.device, dtype=mask.dtype)
                if motion_first:
                    valid_mask = torch.cat((mask.T, ones), dim=0)
                else:
                    valid_mask = torch.cat((ones, mask.T), dim=0)
                xseq = xseq * valid_mask.unsqueeze(-1)

        if valid_mask is not None:
            attn_pad_mask = (valid_mask.T == 0).bool()
        else:
            attn_pad_mask = None
        xseq = self.query_pos(xseq)

        if self.ablation_skip_connection:
            x = xseq
            xs = []
            residuals = []
            for module in self.encoder.input_blocks:
                x = module(x, src_key_padding_mask=attn_pad_mask)
                if valid_mask is not None:
                    x = x * valid_mask.unsqueeze(-1)
                residuals.append(x)
                xs.append(x)
            x = self.encoder.middle_block(x, src_key_padding_mask=attn_pad_mask)
            if valid_mask is not None:
                x = x * valid_mask.unsqueeze(-1)
            residuals.append(x)
            for module, linear in zip(self.encoder.output_blocks, self.encoder.linear_blocks):
                x = torch.cat([x, xs.pop()], dim=-1)
                x = linear(x)
                x = module(x, src_key_padding_mask=attn_pad_mask)
                if valid_mask is not None:
                    x = x * valid_mask.unsqueeze(-1)
                residuals.append(x)
            if self.encoder.norm is not None:
                x = self.encoder.norm(x)
        else:
            x = xseq
            residuals = []
            for layer in self.encoder.layers:
                x = layer(x, src_key_padding_mask=attn_pad_mask)
                if valid_mask is not None:
                    x = x * valid_mask.unsqueeze(-1)
                residuals.append(x)
            if self.encoder.norm is not None:
                x = self.encoder.norm(x)

        if valid_mask is not None:
            residuals = [r * valid_mask.unsqueeze(-1) for r in residuals]

        return residuals


class EmbedAction(nn.Module):
    def __init__(
        self,
        num_actions,
        latent_dim,
        guidance_scale=7.5,
        guidance_uncodp=0.1,
        force_mask=False,
    ):
        super().__init__()
        self.nclasses = num_actions
        self.guidance_scale = guidance_scale
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))
        self.guidance_uncodp = guidance_uncodp
        self.force_mask = force_mask
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):
        idx = input[:, 0].to(torch.long)
        output = self.action_embedding[idx]
        output = self.mask_cond(output)
        return output.unsqueeze(0)

    def mask_cond(self, cond, force=False):
        if force or (self.training and torch.rand(()) < self.guidance_uncodp):
            cond = torch.zeros_like(cond)
        return cond
