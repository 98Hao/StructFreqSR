from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
from torch import nn
from torch.nn import functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, act=True):
        super().__init__()
        if padding is None:
            padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.pointwise(self.depthwise(x)))


def build_conv(in_channels, out_channels, conv_type="depthwise", kernel_size=3, act=True):
    padding = kernel_size // 2
    if conv_type == "normal":
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=True)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    if conv_type == "depthwise":
        return DepthwiseSeparableConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, act=act)
    raise ValueError(f"Unsupported conv_type: {conv_type}")


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, conv_type="normal"):
        super().__init__()
        self.conv1 = build_conv(channels, channels, conv_type=conv_type, act=False)
        self.conv2 = build_conv(channels, channels, conv_type=conv_type, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

def closest_larger_multiple(size, multiple):
    return int(math.ceil(size / multiple) * multiple)


class SizeAdapter:
    def __init__(self, minimum_size=1):
        self.minimum_size = max(1, int(minimum_size))
        self.original_height = None
        self.original_width = None

    def pad(self, x):
        h, w = x.shape[-2:]
        self.original_height, self.original_width = h, w
        pad_h = closest_larger_multiple(h, self.minimum_size) - h
        pad_w = closest_larger_multiple(w, self.minimum_size) - w
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    def unpad(self, x):
        return x[..., :self.original_height, :self.original_width]


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="depthwise"):
        super().__init__()
        self.block = nn.Sequential(build_conv(in_channels, out_channels, conv_type=conv_type), build_conv(out_channels, out_channels, conv_type=conv_type))

    def forward(self, x):
        return self.block(F.avg_pool2d(x, 2, stride=2))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, conv_type="depthwise"):
        super().__init__()
        self.pre = build_conv(in_channels, out_channels, conv_type=conv_type)
        self.fuse = build_conv(out_channels + skip_channels, out_channels, conv_type=conv_type)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.pre(x), skip], dim=1))


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, num_levels=2, base_channels=16, max_channels=96, conv_type="depthwise", residual=True):
        super().__init__()
        if num_levels < 0:
            raise ValueError("num_levels must be >= 0.")
        self.num_levels = int(num_levels)
        self.residual = bool(residual)
        self.size_adapter = SizeAdapter(minimum_size=2 ** self.num_levels)
        widths = [min(base_channels * (2 ** i), max_channels) for i in range(self.num_levels + 1)]
        self.stem = nn.Sequential(build_conv(in_channels, widths[0], conv_type=conv_type), build_conv(widths[0], widths[0], conv_type=conv_type))
        self.downs = nn.ModuleList([DownBlock(widths[i], widths[i + 1], conv_type=conv_type) for i in range(self.num_levels)])
        self.ups = nn.ModuleList([UpBlock(widths[i + 1], widths[i], widths[i], conv_type=conv_type) for i in reversed(range(self.num_levels))])
        self.out_conv = nn.Conv2d(widths[0], out_channels, 3, 1, 1)
        self.residual_proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x_pad = self.size_adapter.pad(x)
        skips = []
        y = self.stem(x_pad)
        skips.append(y)
        for down in self.downs:
            y = down(y)
            skips.append(y)
        if self.num_levels > 0:
            for i, up in enumerate(self.ups):
                y = up(y, skips[-2 - i])
        y = self.out_conv(y)
        if self.residual:
            y = y + self.residual_proj(x_pad)
        return self.size_adapter.unpad(y)


class SemanticEncoder(nn.Module):

    def __init__(self, in_channels=3, semantic_channels=24):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, semantic_channels, 3, 1, 1), nn.ReLU(inplace=True))

        self.coarse = nn.Sequential(
            DepthwiseSeparableConv(semantic_channels, semantic_channels, 5, padding=2),
            ResidualConvBlock(semantic_channels, "normal"),
            DepthwiseSeparableConv(semantic_channels, semantic_channels, 5, padding=2)
        )

        self.fine_fusion = nn.Conv2d(semantic_channels * 2, semantic_channels, 1, 1, 0)
        self.fine = nn.Sequential(
            DepthwiseSeparableConv(semantic_channels, semantic_channels, 3, padding=1),
            ResidualConvBlock(semantic_channels, "normal")
        )

    def forward(self, x):
        base = self.stem(x)
        coarse = self.coarse(base)
        fine = self.fine(self.fine_fusion(torch.cat([base, coarse], dim=1)))
        return coarse, fine

class StructureTensor(nn.Module):

    def __init__(self, smooth_kernel=3, semantic_channels = 24,eps=1e-6):
        super().__init__()
        self.semantic_proj = nn.Conv2d(semantic_channels, 1, 1, 1, 0)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        sobel_y = sobel_x.t().contiguous()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.smooth_kernel = int(smooth_kernel)
        self.eps = float(eps)

    def _smooth(self, x):
        k = self.smooth_kernel
        return F.avg_pool2d(x, k, stride=1, padding=k // 2) if k > 1 else x

    def forward(self, semantic):
        original_dtype = semantic.dtype

        with torch.amp.autocast(device_type=semantic.device.type, enabled=False):
            s = self.semantic_proj(semantic.float())
            gx = F.conv2d(s, self.sobel_x.float(), padding=1)
            gy = F.conv2d(s, self.sobel_y.float(), padding=1)

            jxx = self._smooth(gx * gx)
            jyy = self._smooth(gy * gy)
            jxy = self._smooth(gx * gy)

            trace = jxx + jyy
            delta = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square() + self.eps)

            rho = (delta / (trace + self.eps)).clamp(0.0, 1.0)
            dir_x = (jxx - jyy) / (delta + self.eps)
            dir_y = (2.0 * jxy) / (delta + self.eps)

            energy = trace / (trace.mean(dim=(-2, -1), keepdim=True) + self.eps)
            energy = torch.tanh(energy)

        return energy.to(original_dtype), rho.to(original_dtype), dir_x.to(original_dtype), dir_y.to(original_dtype)


def create_dct_matrix(n):
    mat = torch.zeros(n, n, dtype=torch.float32)
    factor = math.pi / (2.0 * n)
    for k in range(n):
        alpha = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
        for i in range(n):
            mat[k, i] = alpha * math.cos((2 * i + 1) * k * factor)
    return mat


class BlockDCT2D(nn.Module):
    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = int(block_size)
        dct = create_dct_matrix(self.block_size)
        self.register_buffer("dct", dct)
        self.register_buffer("idct", dct.t().contiguous())

    def _pad(self, x):
        b = self.block_size
        h, w = x.shape[-2:]
        ph = (b - h % b) % b
        pw = (b - w % b) % b
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return x, (h, w)

    def _to_blocks(self, x):
        b = self.block_size
        B, C, H, W = x.shape
        Hb, Wb = H // b, W // b
        return x.view(B, C, Hb, b, Wb, b).permute(0, 1, 2, 4, 3, 5).contiguous()

    def _from_blocks(self, x):
        b = self.block_size
        B, C, Hb, Wb, _, _ = x.shape
        return x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hb * b, Wb * b)

    def forward(self, x):
        x, original_size = self._pad(x)
        blocks = self._to_blocks(x)
        coeff = torch.matmul(self.dct, blocks)
        coeff = torch.matmul(coeff, self.idct)
        return coeff, original_size

    def inverse(self, coeff, original_size):
        x = torch.matmul(self.idct, coeff)
        x = torch.matmul(x, self.dct)
        x = self._from_blocks(x)
        h, w = original_size
        return x[..., :h, :w]

    def pack_frequency(self, coeff):

        B, C, Hb, Wb, b1, b2 = coeff.shape
        x = coeff.permute(0, 1, 4, 5, 2, 3).contiguous()
        return x.view(B * C, b1 * b2, Hb, Wb), (B, C, Hb, Wb)

    def unpack_frequency(self, x, shape_info):
        """
        [B*C,b*b,Hb,Wb] -> [B,C,Hb,Wb,b,b]
        """
        B, C, Hb, Wb = shape_info
        b = self.block_size
        x = x.view(B, C, b, b, Hb, Wb)
        return x.permute(0, 1, 4, 5, 2, 3).contiguous()


class DirectionalFrequencyEnhancer(nn.Module):
    def __init__(self, channels, semantic_channels, block_size=8):
        super().__init__()
        self.channels = channels
        self.block_size = int(block_size)
        self.num_freq = self.block_size * self.block_size

        self.dct = BlockDCT2D(block_size)
        self.structure = StructureTensor(smooth_kernel=3, semantic_channels=semantic_channels)

        self.semantic_proj = nn.Conv2d(semantic_channels, channels, 1, 1, 0)

        self.freq_net = nn.Sequential(
            nn.Conv2d(self.num_freq, self.num_freq, 1, 1, 0),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(self.num_freq, self.num_freq, 3),
            nn.Conv2d(self.num_freq, self.num_freq, 1, 1, 0)
        )

        self.gamma = nn.Parameter(torch.tensor(0.1))

        cos2phi, sin2phi, high = self._build_frequency_tables(block_size)
        self.register_buffer("cos2phi", cos2phi)
        self.register_buffer("sin2phi", sin2phi)
        self.register_buffer("high_freq", high)

    @staticmethod
    def _build_frequency_tables(block_size):
        u = torch.arange(block_size, dtype=torch.float32).view(block_size, 1)
        v = torch.arange(block_size, dtype=torch.float32).view(1, block_size)

        fx = v.expand(block_size, block_size)
        fy = u.expand(block_size, block_size)

        denom = fx.square() + fy.square()

        cos2phi = (fx.square() - fy.square()) / (denom + 1e-6)
        sin2phi = (2.0 * fx * fy) / (denom + 1e-6)

        high = torch.sqrt(denom) / (math.sqrt(2.0) * max(block_size - 1, 1))

        cos2phi[0, 0] = 0.0
        sin2phi[0, 0] = 0.0
        high[0, 0] = 0.0

        return (
            cos2phi.view(1, 1, 1, 1, block_size, block_size),
            sin2phi.view(1, 1, 1, 1, block_size, block_size),
            high.view(1, 1, 1, 1, block_size, block_size)
        )

    def _direction_weight(self, semantic, Hb, Wb):
        energy, rho, dir_x, dir_y = self.structure(semantic)

        b = self.block_size
        Hp, Wp = Hb * b, Wb * b

        ph = Hp - semantic.shape[-2]
        pw = Wp - semantic.shape[-1]

        maps = [
            F.pad(m, (0, pw, 0, ph), mode="replicate")
            for m in (energy, rho, dir_x, dir_y)
        ]

        energy_b, rho_b, dx_b, dy_b = [
            F.avg_pool2d(m, kernel_size=b, stride=b)
            for m in maps
        ]

        energy_b = energy_b.unsqueeze(-1).unsqueeze(-1)
        rho_b = rho_b.unsqueeze(-1).unsqueeze(-1)
        dx_b = dx_b.unsqueeze(-1).unsqueeze(-1)
        dy_b = dy_b.unsqueeze(-1).unsqueeze(-1)

        direction_match = 0.5 * (
            1.0
            + dx_b * self.cos2phi
            + dy_b * self.sin2phi
        )

        weight = (
            energy_b
            * rho_b
            * direction_match.clamp(0.0, 1.0)
            * self.high_freq
        ).clamp(0.0, 1.0)

        B = weight.shape[0]
        weight = weight.permute(0, 1, 4, 5, 2, 3).contiguous()
        weight = weight.view(B, self.num_freq, Hb, Wb)

        weight = weight.repeat_interleave(self.channels, dim=0)

        return weight

    def forward(self, feature, semantic):

        residual = self.freq_net(feature)

        enhanced_freq = feature + residual

        return enhanced_freq

class SpatialEnhanceStage(nn.Module):
    def __init__(self, channels, semantic_channels, dct_block=8, unet_levels=2, unet_base_channels=16, unet_max_channels=96, conv_type="depthwise"):
        super().__init__()
        self.freq_net = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(channels, channels, 3),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.unet = UNet(channels, channels, num_levels=unet_levels, base_channels=unet_base_channels, max_channels=unet_max_channels, conv_type=conv_type, residual=True)

    def forward(self, feature, semantic):
        return self.unet(self.freq_net(feature))


class OffsetEstimator(nn.Module):
    def __init__(self, semantic_channels, hidden_channels=24, max_offset=8.0):
        super().__init__()
        self.max_offset = float(max_offset)
        self.net = nn.Sequential(nn.Conv2d(semantic_channels * 2, hidden_channels, 1, 1, 0), nn.ReLU(inplace=True), DepthwiseSeparableConv(hidden_channels, hidden_channels, 3), nn.Conv2d(hidden_channels, 2, 1, 1, 0))

    def forward(self, previous_semantic, current_semantic):
        raw = self.net(torch.cat([previous_semantic, current_semantic], dim=1))
        return self.max_offset * torch.tanh(raw)


def warp_feature(feature, offset):

    B, _, H, W = feature.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=feature.device, dtype=feature.dtype), torch.arange(W, device=feature.device, dtype=feature.dtype), indexing="ij")
    grid_x = (2.0 * (xx + 0.5) / W - 1.0).unsqueeze(0).expand(B, -1, -1)
    grid_y = (2.0 * (yy + 0.5) / H - 1.0).unsqueeze(0).expand(B, -1, -1)
    grid_x = grid_x + 2.0 * offset[:, 0] / W
    grid_y = grid_y + 2.0 * offset[:, 1] / H
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(feature, grid, mode="bilinear", padding_mode="border", align_corners=False)


class SpatialSelectionMask(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or max(channels // 2, 8)
        self.net = nn.Sequential(nn.Conv2d(channels * 2, hidden_channels, 1, 1, 0), nn.ReLU(inplace=True), DepthwiseSeparableConv(hidden_channels, hidden_channels, 3), nn.Conv2d(hidden_channels, 1, 1, 1, 0), nn.Sigmoid())

    def forward(self, aligned_previous, current):
        return self.net(torch.cat([aligned_previous, current], dim=1))

class GumbelGroupSelector(nn.Module):
    """Select K channel groups from Q_{t-2}. Output is compact: K * (C/G) channels."""
    def __init__(self, channels, num_groups=8, topk=4, hidden_channels=32, tau=1.0):
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(f"channels={channels} must be divisible by num_groups={num_groups}.")
        if not 1 <= topk <= num_groups:
            raise ValueError("topk must satisfy 1 <= topk <= num_groups.")
        self.channels = channels
        self.num_groups = num_groups
        self.group_channels = channels // num_groups
        self.topk = topk
        self.tau = float(tau)
        self.score = nn.Sequential(nn.Linear(channels * 2, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, num_groups))

    def forward(self, memory, current):
        B, C, H, W = memory.shape
        descriptor = torch.cat(
            [F.adaptive_avg_pool2d(memory, 1).flatten(1), F.adaptive_avg_pool2d(current, 1).flatten(1)], dim=1)
        scores = self.score(descriptor)

        if self.training:
            scores_fp32 = scores.float()
            u = torch.rand(scores_fp32.shape, device=scores.device, dtype=torch.float32).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            routing_logits = (scores_fp32 + gumbel) / self.tau
        else:
            routing_logits = scores.float()

        indices = torch.topk(routing_logits, k=self.topk, dim=1, largest=True, sorted=True).indices
        groups = memory.view(B, self.num_groups, self.group_channels, H, W)
        gather_index = indices[:, :, None, None, None].expand(B, self.topk, self.group_channels, H, W)
        selected = torch.gather(groups, dim=1, index=gather_index)
        selected_scores = torch.gather(routing_logits, 1, indices)
        selected_weights = torch.softmax(selected_scores, dim=1).to(selected.dtype).view(B, self.topk, 1, 1, 1)
        selected = (selected * selected_weights).reshape(B, self.topk * self.group_channels, H, W)
        return selected, indices


class TemporalFusion(nn.Module):
    def __init__(self, channels, selected_memory_channels):
        super().__init__()
        in_channels = channels * 2 + selected_memory_channels
        self.fuse = nn.Sequential(nn.Conv2d(in_channels, channels, 1, 1, 0), nn.ReLU(inplace=True), ResidualConvBlock(channels, 'depthwise'), nn.ReLU(inplace=True), DepthwiseSeparableConv(channels, channels, 3), nn.Conv2d(channels, channels, 1, 1, 0))

    def forward(self, current, previous_selected, memory_selected):
        return current + self.fuse(torch.cat([current, previous_selected, memory_selected], dim=1))


class MemoryUpdater(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.update = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, 1, 0), nn.ReLU(inplace=True))

    def forward(self, old_memory, previous_feature):

        return self.update(torch.cat([old_memory, previous_feature], dim=1))


VSRState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class RenderVSR(nn.Module):
    def __init__(self, scale=4, channels=3, num_features=64, semantic_channels=64, dct_block=8, unet_levels=2, unet_base_channels=64, unet_max_channels=1024, conv_type="depthwise", memory_groups=8, memory_topk=4, max_offset=8.0, num_recon_blocks=2, use_aligned_prev=True):
        super().__init__()
        if scale < 2 or int(scale) != scale:
            raise ValueError(f"scale must be integer >= 2, got {scale}.")
        if num_features % memory_groups != 0:
            raise ValueError("num_features must be divisible by memory_groups.")
        self.scale = int(scale)
        self.channels = int(channels)
        self.num_features = int(num_features)
        self.semantic_channels = int(semantic_channels)
        self.use_aligned_prev = bool(use_aligned_prev)

        self.semantic_encoder = SemanticEncoder(channels, semantic_channels)

        self.shallow = nn.Sequential(nn.Conv2d(channels, num_features, 3, 1, 1), nn.ReLU(inplace=True))

        self.coarse_spatial = SpatialEnhanceStage(num_features, semantic_channels, dct_block, unet_levels, unet_base_channels, unet_max_channels, conv_type)
        self.fine_spatial = SpatialEnhanceStage(num_features, semantic_channels, int(dct_block/2), unet_levels, unet_base_channels, unet_max_channels, conv_type)

        self.offset_estimator = OffsetEstimator(semantic_channels, hidden_channels=semantic_channels, max_offset=max_offset)
        self.spatial_mask = SpatialSelectionMask(num_features)

        self.memory_selector = GumbelGroupSelector(num_features, num_groups=memory_groups, topk=memory_topk, hidden_channels=max(num_features // 2, 16))
        selected_memory_channels = memory_topk * (num_features // memory_groups)
        self.temporal_fusion = TemporalFusion(num_features, selected_memory_channels)
        self.memory_updater = MemoryUpdater(num_features)

        self.post_refine = nn.Sequential(*(ResidualConvBlock(num_features, conv_type=conv_type) for _ in range(num_recon_blocks))) if num_recon_blocks > 0 else nn.Identity()
        self.reconstruction = nn.Sequential(DepthwiseSeparableConv(num_features, num_features, 3), nn.Conv2d(num_features, channels * self.scale * self.scale, 3, 1, 1), nn.PixelShuffle(self.scale))

        final_conv = self.reconstruction[-2]
        nn.init.normal_(
            final_conv.weight,
            mean=0.0,
            std=1e-3
        )
        nn.init.zeros_(final_conv.bias)

    def spatial_forward(self, lr):
        coarse_semantic, fine_semantic = self.semantic_encoder(lr)
        feature = self.shallow(lr)
        feature = self.coarse_spatial(feature, coarse_semantic)
        feature = self.fine_spatial(feature, fine_semantic)
        return feature, fine_semantic, coarse_semantic

    def forward_step(self, current_lr, state: Optional[VSRState] = None, return_aux=False):
        if current_lr.ndim != 4 or current_lr.shape[1] != self.channels:
            raise ValueError(f"Expected [B,{self.channels},H,W], got {tuple(current_lr.shape)}.")

        current_feature, current_semantic, _ = self.spatial_forward(current_lr)
        B, _, H, W = current_feature.shape
        aux = {}

        if state is None:
            fused = current_feature
            new_memory = current_feature.new_zeros(B, self.num_features, H, W)
        else:
            previous_feature, previous_semantic, old_memory = state
            if previous_feature.shape != current_feature.shape or previous_semantic.shape != current_semantic.shape or old_memory.shape != current_feature.shape:
                raise ValueError("Temporal state shape does not match current frame. Reset state after resolution/batch-size changes.")

            offset = self.offset_estimator(previous_semantic, current_semantic)
            aligned_previous = warp_feature(previous_feature, offset)
            spatial_mask = self.spatial_mask(aligned_previous, current_feature)
            previous_for_fusion = aligned_previous if self.use_aligned_prev else previous_feature
            selected_previous = spatial_mask * previous_for_fusion
            selected_memory, selected_indices = self.memory_selector(old_memory, current_feature)
            fused = self.temporal_fusion(current_feature, selected_previous, selected_memory)
            new_memory = self.memory_updater(old_memory, previous_feature)

            if return_aux:
                aux.update({"offset": offset, "aligned_previous": aligned_previous, "spatial_mask": spatial_mask, "selected_memory_indices": selected_indices})

        refined = self.post_refine(fused)
        residual = self.reconstruction(refined)
        base = F.interpolate(current_lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        sr = base + residual
        new_state: VSRState = (current_feature, current_semantic, new_memory)
        return (sr, new_state, aux) if return_aux else (sr, new_state)

    def forward(self, lr_sequence, return_aux=False):
        """lr_sequence: [B,T,C,H,W]."""
        if lr_sequence.ndim != 5 or lr_sequence.shape[2] != self.channels:
            raise ValueError(f"Expected [B,T,{self.channels},H,W], got {tuple(lr_sequence.shape)}.")
        state = None
        outputs, all_aux = [], []
        for t in range(lr_sequence.shape[1]):
            if return_aux:
                sr, state, aux = self.forward_step(lr_sequence[:, t], state, return_aux=True)
                all_aux.append(aux)
            else:
                sr, state = self.forward_step(lr_sequence[:, t], state, return_aux=False)
            outputs.append(sr)
        outputs = torch.stack(outputs, dim=1)
        return (outputs, all_aux) if return_aux else outputs


if __name__ == "__main__":
    model = RenderVSR(scale=4, channels=3, num_features=64, semantic_channels=64, dct_block=8, unet_levels=2, unet_base_channels=64, conv_type="depthwise", memory_groups=8, memory_topk=4, max_offset=8.0, num_recon_blocks=2)
    x = torch.randn(2, 4, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)
    print("params:", sum(p.numel() for p in model.parameters()) / 1e3, "K")
