import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba
from timm.models.layers import DropPath

NEG_INF = -1000000


class Fusion_block(nn.Module):

    def __init__(self, channels=64, r=2):
        super(Fusion_block, self).__init__()

        inter_channels = channels // r
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, depth, mask, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        depth = depth.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        mask = mask.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        xa = x + mask
        xb = x + depth
        xl = self.local_att(xa)
        xg = self.global_att(xb)
        wei1 = self.sigmoid(xl)
        wei2 = self.sigmoid(xg)

        xo = x * wei1 + x * wei2
        return xo.permute(0, 2, 3, 1).view(B, L, C).contiguous()


class ECA(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16, use_cam=False):
        super(ECA, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())
        self.use_cam = use_cam
        self.dim = num_feat


    def forward(self, x, cam=None):

        y = self.attention(x)
        x = x * y
        if self.use_cam:
            cam = cam.permute(0, 2, 1).contiguous()
            cam = cam.unsqueeze(-1)
            x = x * cam
            x = x + cam
        return x



class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=2, squeeze_factor=2, use_cam=False):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1)
            )
        self.eca = ECA(num_feat, squeeze_factor=squeeze_factor, use_cam=use_cam)

    def forward(self, x, cam=None):
        if cam != None:
            x = self.eca(self.cab(x), cam)
        else:
            x = self.eca(self.cab(x))
        return x


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            use_cam=False,
            **kwargs,
    ):
        super().__init__()
        self.use_cam=use_cam
        self.ln_1 = norm_layer(hidden_dim)
        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=4,
            expand=expand,
        )
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, use_cam=self.use_cam)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input, x_size, cam=None):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = self.mamba(x.view(B, L, C).contiguous())
        x = input*self.skip_scale + self.drop_path(x.view(B, *x_size, C).contiguous())
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous(), cam).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # print(emb.shape)
        # emb = emb.permute(0, 2, 1).contiguous().unsqueeze(-1)

        return emb



class bokeh(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=16,
                 num_blocks=[4, 4, 4, 6],
                 mlp_ratio=4.,
                 num_refinement_blocks=6,
                 drop_path_rate=0.,
                 bias=False,
                 use_cam=False,
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(bokeh, self).__init__()
        if use_cam:
            self.embed = SinusoidalPosEmb(dim)

        self.mlp_ratio = mlp_ratio
        self.use_cam = use_cam
        self.patch_embed1 = OverlapPatchEmbed(inp_channels, dim//2)
        self.patch_embed2 = OverlapPatchEmbed(inp_channels, dim//2)
        self.patch_embed3 = OverlapPatchEmbed(inp_channels, dim//2)
        self.down1 = Downsample(dim//2)  ## From Level 0 to Level 1
        self.fusion = Fusion_block(dim//2, 2)
        self.skip_scale= nn.Parameter(torch.ones(dim//2))
        base_d_state = 4
        self.encoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=base_d_state,
                use_cam=use_cam,
            )
            for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
                use_cam=use_cam,
            )
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
                use_cam=use_cam,
            )
            for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 3),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 3),
                use_cam=use_cam,
            )
            for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
                use_cam=use_cam,
            )
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
                use_cam=use_cam,
            )
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
                use_cam=use_cam,
            )
            for i in range(num_blocks[0])])

        self.refinement = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
                use_cam=use_cam,
            )
            for i in range(num_refinement_blocks)])
        self.up1 = Upsample(dim*2)  ## From Level 1 to Level 0


        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, depth_img, mask, cam=None):
        if self.use_cam:
            cam = self.embed(cam)
            cam1 = torch.cat([cam, cam], 2)
            cam2 = torch.cat([cam1, cam1], 2)
            cam3 = torch.cat([cam2, cam2], 2)
        _, _, H, W = inp_img.shape
        H, W = H // 2, W // 2
        inp_enc_level1_1 = self.patch_embed1(inp_img)
        depth_img = self.patch_embed2(depth_img)
        mask = self.patch_embed3(mask)
        inp_enc_level1_1 = self.fusion(inp_enc_level1_1, depth_img, mask, 2*H, 2*W) + inp_enc_level1_1*self.skip_scale
        inp_enc_level1 = self.down1(inp_enc_level1_1, 2*H, 2*W)  # b, hw, c
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W], cam=cam)

        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)  # b, hw//4, 2c
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            if self.use_cam:
                out_enc_level2 = layer(out_enc_level2, [H // 2, W // 2], cam=cam1)
            else:
                out_enc_level2 = layer(out_enc_level2, [H // 2, W // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2)  # b, hw//16, 4c
        out_enc_level3 = inp_enc_level3
        for layer in self.encoder_level3:
            if self.use_cam:
                out_enc_level3 = layer(out_enc_level3, [H // 4, W // 4], cam=cam2)
            else:
                out_enc_level3 = layer(out_enc_level3, [H // 4, W // 4])

        inp_enc_level4 = self.down3_4(out_enc_level3, H // 4, W // 4)  # b, hw//64, 8c
        latent = inp_enc_level4
        for layer in self.latent:
            if self.use_cam:
                latent = layer(latent, [H // 8, W // 8], cam=cam3)
            else:
                latent = layer(latent, [H // 8, W // 8])

        inp_dec_level3 = self.up4_3(latent, H // 8, W // 8)  # b, hw//16, 4c
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            if self.use_cam:
                out_dec_level3 = layer(out_dec_level3, [H // 4, W // 4], cam=cam2)
            else:
                out_dec_level3 = layer(out_dec_level3, [H // 4, W // 4])

        inp_dec_level2 = self.up3_2(out_dec_level3, H // 4, W // 4)  # b, hw//4, 2c
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            if self.use_cam:
                out_dec_level2 = layer(out_dec_level2, [H // 2, W // 2], cam=cam1)
            else:
                out_dec_level2 = layer(out_dec_level2, [H // 2, W // 2])

        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2)  # b, hw, c
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            if self.use_cam:
                out_dec_level1 = layer(out_dec_level1, [H, W], cam=cam1)
            else:
                out_dec_level1 = layer(out_dec_level1, [H, W])

        for layer in self.refinement:
            if self.use_cam:
                out_dec_level1 = layer(out_dec_level1, [H, W], cam=cam1)
            else:
                out_dec_level1 = layer(out_dec_level1, [H, W])

        out_dec_level1 = self.up1(out_dec_level1, H, W)
        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=2*H, w=2*W).contiguous()

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
if __name__ == '__main__':
    model = bokeh(use_cam=True)
    inp_img = torch.randn(2, 3, 256, 256)
    depth_img = torch.randn(2, 3, 256, 256)
    mask = torch.randn(2, 3, 256, 256)
    cam = torch.FloatTensor([1.8])
    out = model(inp_img, depth_img, mask, cam)
    print(out.shape)