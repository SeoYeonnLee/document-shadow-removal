import os, sys
import time
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.vit import Block
from networks.Patch_embed import PatchEmbed
from networks.FProLite import FProLiteBlock


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales: int = 4, ct_channels: int = 1):
        super().__init__()
        if num_scales == 4:
            scales = (4, 8, 16, 32)
        elif num_scales == 3:
            scales = (4, 8, 16)
        else:
            raise ValueError("num_scales must be 3 or 4")

        self.stages = nn.ModuleList([
            self._make_stage(in_channels, s, ct_channels) for s in scales
        ])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv  = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu  = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat(
            [F.interpolate(stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats],
            dim=1
        )
        return self.relu(self.bottleneck(priors))


class MaskedAutoencoderViT(nn.Module):
    """MatteViT backbone with a single FPro‑Lite block to amplify HF details."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        out_chans: int = 3,
        fea_chans: int = 16,
        num_scales: int = 4,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.,
        norm_layer=nn.LayerNorm,
        norm_pix_loss: bool = False,
        global_residual: bool = False,
    ):
        super().__init__()
        self.global_residual = global_residual

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.fpro        = FProLiteBlock(embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_embed_for_unselected = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * fea_chans)

        self.norm_pix_loss = norm_pix_loss

        self.pyramid_module = PyramidPooling(
            fea_chans, fea_chans, num_scales=num_scales, ct_channels=fea_chans // 4
        )
        self.last_conv = nn.Conv2d(fea_chans, out_chans, kernel_size=3, padding=1, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.flatten(1))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x, H: int, W: int):
        p = self.patch_embed.patch_size[0]
        h, w = H // p, W // p
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, -1)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], -1, h * p, w * p)

    def forward_encoder(self, imgs):
        feat = self.patch_embed.proj(imgs)
        feat = self.fpro(feat)  # HF amplification

        x = feat.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
        
    def forward_decoder(self, x):
        x = self.decoder_embed(x)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)

    def forward(self, imgs):
        _, _, ori_H, ori_W = imgs.size()
        latent = self.forward_encoder(imgs)
        pred   = self.forward_decoder(latent)
        pred_img = self.unpatchify(pred, ori_H, ori_W)
        pred_img = self.last_conv(self.pyramid_module(pred_img))
        if self.global_residual:
            pred_img = pred_img + imgs
        return pred_img

def mae_vit_small_patch16_dec128d4b(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=8, embed_dim=256, depth=6, num_heads=8,
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

if __name__ == "__main__":
    model = mae_vit_small_patch16_dec128d4b(img_size=256, in_chans=4)
    print('#parameters:', sum(p.numel() for p in model.parameters()))
    x = torch.randn(1, 4, 256, 256)
    y = model(x)
    print('output:', y.shape)
