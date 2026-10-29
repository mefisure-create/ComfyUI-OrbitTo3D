from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small building blocks matching OpenMMLab / RealBasicVSR state_dict names.
# No mmcv / mmagic dependency is required.
# -----------------------------------------------------------------------------


class _ConvAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding, activate=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=True)
        self.activate = nn.ReLU(inplace=True) if activate else nn.Identity()

    def forward(self, x):
        return self.activate(self.conv(x))


class _ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out


def _make_layer(num_blocks: int, channels: int):
    return nn.Sequential(*[_ResidualBlockNoBN(channels) for _ in range(num_blocks)])


class _ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            _make_layer(num_blocks, out_channels),
        )

    def forward(self, feat):
        return self.main(feat)


class _PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel=3):
        super().__init__()
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            1,
            (upsample_kernel - 1) // 2,
            bias=True,
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.upsample_conv(x))


def _flow_warp(x, flow, padding_mode="zeros", align_corners=True):
    """Warp x [N,C,H,W] with flow [N,H,W,2] measured in pixels."""
    n, _, h, w = x.shape
    if flow.shape[1:3] != (h, w):
        raise ValueError(f"flow {tuple(flow.shape)} does not match feature {(h, w)}")

    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=x.dtype, device=x.device),
        torch.arange(w, dtype=x.dtype, device=x.device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    vgrid = grid + flow
    vx = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
    vy = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
    grid = torch.stack((vx, vy), dim=-1)
    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


class _SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvAct(8, 32, 7, 1, 3, True),
            _ConvAct(32, 64, 7, 1, 3, True),
            _ConvAct(64, 32, 7, 1, 3, True),
            _ConvAct(32, 16, 7, 1, 3, True),
            _ConvAct(16, 2, 7, 1, 3, False),
        )

    def forward(self, x):
        return self.basic_module(x)


class _SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([_SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.shape
        ref_pyr = [(ref - self.mean) / self.std]
        supp_pyr = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref_pyr.append(F.avg_pool2d(ref_pyr[-1], 2, 2, count_include_pad=False))
            supp_pyr.append(F.avg_pool2d(supp_pyr[-1], 2, 2, count_include_pad=False))
        ref_pyr.reverse()
        supp_pyr.reverse()

        flow = ref.new_zeros(n, 2, h // 32, w // 32)
        for level, (r, s) in enumerate(zip(ref_pyr, supp_pyr)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    flow, scale_factor=2, mode="bilinear", align_corners=True
                ) * 2.0
                if flow_up.shape[-2:] != r.shape[-2:]:
                    old_h, old_w = flow_up.shape[-2:]
                    new_h, new_w = r.shape[-2:]
                    flow_up = F.interpolate(
                        flow_up, size=(new_h, new_w), mode="bilinear", align_corners=True
                    )
                    flow_up[:, 0] *= new_w / max(old_w, 1)
                    flow_up[:, 1] *= new_h / max(old_h, 1)
            warped = _flow_warp(
                s, flow_up.permute(0, 2, 3, 1), padding_mode="border"
            )
            flow = flow_up + self.basic_module[level](
                torch.cat((r, warped, flow_up), dim=1)
            )
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[-2:]
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        ref_up = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp_up = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = self.compute_flow(ref_up, supp_up)
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / w_up
        flow[:, 1] *= h / h_up
        return flow


class _BasicVSRNet(nn.Module):
    """BasicVSR generator used inside RealBasicVSR."""

    def __init__(self, mid_channels=64, num_blocks=20):
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = _SPyNet()
        self.backward_resblocks = _ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks
        )
        self.forward_resblocks = _ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks
        )
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = _PixelShufflePack(mid_channels, mid_channels, 2, 3)
        self.upsample2 = _PixelShufflePack(mid_channels, 64, 2, 3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.shape
        lrs_1 = lrs[:, :-1].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:].reshape(-1, c, h, w)
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def forward_center(self, lrs):
        """Return only the x4 SR result for the temporal center frame."""
        n, t, _, h, w = lrs.shape
        center = t // 2
        flows_forward, flows_backward = self.compute_flow(lrs)

        # Backward-time propagation: only the state at `center` is required.
        feat_backward = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, center - 1, -1):
            if i < t - 1:
                flow = flows_backward[:, i]
                feat_backward = _flow_warp(
                    feat_backward, flow.permute(0, 2, 3, 1)
                )
            feat_backward = self.backward_resblocks(
                torch.cat((lrs[:, i], feat_backward), dim=1)
            )

        # Forward-time propagation up to the target frame.
        feat_forward = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(0, center + 1):
            if i > 0:
                flow = flows_forward[:, i - 1]
                feat_forward = _flow_warp(
                    feat_forward, flow.permute(0, 2, 3, 1)
                )
            feat_forward = self.forward_resblocks(
                torch.cat((lrs[:, i], feat_forward), dim=1)
            )

        lr_curr = lrs[:, center]
        out = self.lrelu(self.fusion(torch.cat((feat_backward, feat_forward), dim=1)))
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        return out + self.img_upsample(lr_curr)


class RealBasicVSRNet(nn.Module):
    """RealBasicVSR generator: image cleaning + BasicVSR."""

    def __init__(
        self,
        mid_channels=64,
        num_propagation_blocks=20,
        num_cleaning_blocks=20,
        dynamic_refine_thres=5.0,
    ):
        super().__init__()
        self.dynamic_refine_thres = float(dynamic_refine_thres) / 255.0
        self.image_cleaning = nn.Sequential(
            _ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
        )
        self.basicvsr = _BasicVSRNet(mid_channels, num_propagation_blocks)

    def clean_sequence(self, lqs):
        # Sequential cleaning saves VRAM and is mathematically equivalent to
        # applying the same convolutional cleaning network to the time dimension
        # as a batch.
        n, t, _, _, _ = lqs.shape
        lqs = lqs.clone()
        for _ in range(3):
            residues = []
            for i in range(t):
                residue = self.image_cleaning(lqs[:, i])
                lqs[:, i] = lqs[:, i] + residue
                residues.append(residue)
            mean_abs = torch.stack([r.abs().mean() for r in residues]).mean()
            if mean_abs < self.dynamic_refine_thres:
                break
        return lqs

    def forward_center(self, lqs):
        clean = self.clean_sequence(lqs)
        return self.basicvsr.forward_center(clean)


# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------


def _candidate_state_dicts(ckpt) -> Dict[str, Dict[str, torch.Tensor]]:
    sources = {}
    if isinstance(ckpt, dict):
        for name in ("params_ema", "params", "state_dict"):
            obj = ckpt.get(name)
            if isinstance(obj, dict):
                sources[name] = obj
    if not sources and isinstance(ckpt, dict):
        sources["root"] = ckpt

    candidates = {}
    prefixes = (
        "module.generator_ema.",
        "generator_ema.",
        "module.generator.",
        "generator.",
        "module.",
        "",
    )
    for source_name, source in sources.items():
        for prefix in prefixes:
            selected = {}
            for key, value in source.items():
                if not torch.is_tensor(value):
                    continue
                if key == "step_counter" or key.endswith(".step_counter"):
                    continue
                if prefix:
                    if not key.startswith(prefix):
                        continue
                    new_key = key[len(prefix):]
                else:
                    new_key = key
                selected[new_key] = value
            if selected:
                candidates[f"{source_name}:{prefix or '<none>'}"] = selected
    return candidates


def load_realbasicvsr(checkpoint_path: str, device: torch.device, dynamic_refine_thres=5.0):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RealBasicVSRNet(dynamic_refine_thres=dynamic_refine_thres)
    model_keys = set(model.state_dict().keys())

    best = None
    best_overlap = -1
    for name, state in _candidate_state_dicts(ckpt).items():
        overlap = len(model_keys.intersection(state.keys()))
        if overlap > best_overlap:
            best = (name, state)
            best_overlap = overlap

    if best is None:
        raise RuntimeError("Could not find a usable state_dict in the RealBasicVSR checkpoint")

    name, state = best
    filtered = {k: v for k, v in state.items() if k in model_keys}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        raise RuntimeError(
            "RealBasicVSR checkpoint mismatch.\n"
            f"Selected state: {name}\n"
            f"Matched: {len(filtered)}/{len(model_keys)} tensors\n"
            f"Missing ({len(missing)}): {missing[:40]}\n"
            f"Unexpected ({len(unexpected)}): {unexpected[:40]}"
        )

    print(
        f"[Turntable RealBasicVSR] checkpoint loaded via {name}; "
        f"matched {len(filtered)} tensors"
    )
    model.eval().requires_grad_(False).to(device)
    return model


# -----------------------------------------------------------------------------
# Spatial tiled center-frame inference
# -----------------------------------------------------------------------------


def _tile_starts(length: int, tile: int, overlap: int):
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, max(length - tile, 0) + 1, step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _blend_window(h: int, w: int, overlap_out: int, device, dtype):
    wy = torch.ones(h, device=device, dtype=dtype)
    wx = torch.ones(w, device=device, dtype=dtype)
    n = min(overlap_out, h // 2, w // 2)
    if n > 0:
        ramp = torch.linspace(0.05, 1.0, n, device=device, dtype=dtype)
        wy[:n] = ramp
        wy[-n:] = ramp.flip(0)
        wx[:n] = ramp
        wx[-n:] = ramp.flip(0)
    return (wy[:, None] * wx[None, :])[None, None]


class RealBasicVSRBackend:
    def __init__(
        self,
        checkpoint_path: str,
        device="cuda",
        output_scale: int = 2,
        tile_size: int = 384,
        tile_overlap: int = 48,
        dynamic_refine_thres: float = 5.0,
    ):
        self.device = torch.device(device)
        self.output_scale = int(output_scale)
        self.tile_size = int(tile_size)
        self.tile_overlap = int(tile_overlap)
        self.model = load_realbasicvsr(
            checkpoint_path,
            self.device,
            dynamic_refine_thres=dynamic_refine_thres,
        )
        print(f"[Turntable RealBasicVSR] loaded: {checkpoint_path}")

    @torch.inference_mode()
    def _run_tile(self, sequence):
        if sequence.ndim != 4 or sequence.shape[1] != 3:
            raise ValueError(f"Expected sequence [T,3,H,W], got {tuple(sequence.shape)}")
        if sequence.shape[0] < 3 or sequence.shape[0] % 2 == 0:
            raise ValueError(f"Temporal window must be odd and >=3, got T={sequence.shape[0]}")

        inp = sequence.unsqueeze(0).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        out4 = self.model.forward_center(inp)
        if self.output_scale == 4:
            out = out4
        elif self.output_scale == 2:
            h, w = sequence.shape[-2:]
            out = F.interpolate(
                out4,
                size=(h * 2, w * 2),
                mode="bicubic",
                align_corners=False,
            )
        else:
            raise ValueError(f"output_scale must be 2 or 4, got {self.output_scale}")
        return out.clamp_(0.0, 1.0)

    @torch.inference_mode()
    def enhance_center(self, sequence_rgb: torch.Tensor):
        if sequence_rgb.ndim != 4 or sequence_rgb.shape[1] != 3:
            raise ValueError(f"Expected sequence [T,3,H,W], got {tuple(sequence_rgb.shape)}")

        t, _, h, w = sequence_rgb.shape
        if t < 3 or t % 2 == 0:
            raise ValueError(f"Temporal window must be odd and >=3, got T={t}")

        s = self.output_scale
        tile = min(self.tile_size, h, w)
        overlap = min(self.tile_overlap, tile // 3)
        ys = _tile_starts(h, tile, overlap)
        xs = _tile_starts(w, tile, overlap)

        out = torch.zeros((1, 3, h * s, w * s), dtype=torch.float32, device="cpu")
        acc = torch.zeros((1, 1, h * s, w * s), dtype=torch.float32, device="cpu")

        for y in ys:
            for x in xs:
                crop = sequence_rgb[:, :, y:y + tile, x:x + tile]
                sr = self._run_tile(crop).float().cpu()
                oh, ow = sr.shape[-2:]
                win = _blend_window(oh, ow, overlap * s, sr.device, sr.dtype)
                oy, ox = y * s, x * s
                out[:, :, oy:oy + oh, ox:ox + ow] += sr * win
                acc[:, :, oy:oy + oh, ox:ox + ow] += win
                del sr, crop
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return out / acc.clamp_min_(1e-6)

    def close(self):
        if hasattr(self, "model"):
            self.model.cpu()
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
