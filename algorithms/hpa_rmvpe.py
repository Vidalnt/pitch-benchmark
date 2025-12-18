import urllib.request
from pathlib import Path
from typing import Tuple, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from librosa.filters import mel

from .base import ContinuousPitchAlgorithm

SAMPLE_RATE = 16000
N_CLASS = 360
N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 1024
DEFAULT_MODEL_URL = (
    "https://huggingface.co/vidalnt/hpa-rmvpe/resolve/main/exp1/model_124000.pt"
)


def get_model_path(model_path: str = None) -> str:
    if model_path is None:
        try:
            model_path = Path(__file__).parent / "hpa_rmvpe.pt"
        except NameError:
            model_path = Path("hpa_rmvpe.pt")
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        print(f"HPA-RMVPE model not found at {model_path}")
        print(f"Downloading from {DEFAULT_MODEL_URL}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(model_path))
            print(f"Model downloaded successfully to {model_path}")
        except Exception as e:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download model: {e}")

    return str(model_path)


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
                audio.device
            )

        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new

        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(
            c_in,
            c_in,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            groups=c_in,
            bias=bias,
        )
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(
                Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DSBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)
        self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k1=3, k2=5, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(
                DSBottleneck(c_, c_, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2)
                for _ in range(n)
            )
        )


class DSC3k2(C2f):
    def __init__(
        self, c1, c2, n=1, dsc3k=False, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k(
                    self.c,
                    self.c,
                    n=2,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2,
                )
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                DSBottleneck(
                    self.c, self.c, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2
                )
                for _ in range(n)
            )


class AdaHyperedgeGen(nn.Module):
    def __init__(
        self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)

        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)
        elif context == "both":
            self.context_net = nn.Linear(2 * node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(f"Unsupported context '{context}'.")

        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)
        else:
            avg_context = X.mean(dim=1)
            max_context, _ = X.max(dim=1)
            context_cat = torch.cat([avg_context, max_context], dim=-1)

        prototype_offsets = self.context_net(context_cat).view(
            B, self.num_hyperedges, D
        )
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets

        X_proj = self.pre_head_proj(X)
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(
            B, self.num_hyperedges, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(
            B * self.num_heads, self.num_hyperedges, self.head_dim
        ).transpose(1, 2)

        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1)
        logits = self.dropout(logits)

        return F.softmax(logits, dim=1)


class AdaHGConv(nn.Module):
    def __init__(
        self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"
    ):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(
            embed_dim, num_hyperedges, num_heads, dropout, context
        )
        self.edge_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

    def forward(self, X):
        A = self.edge_generator(X)
        He = torch.bmm(A.transpose(1, 2), X)
        He = self.edge_proj(He)
        X_new = torch.bmm(A, He)
        X_new = self.node_proj(X_new)
        return X_new + X


class AdaHGComputation(nn.Module):
    def __init__(
        self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(embed_dim, num_hyperedges, num_heads, dropout, context)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.hgnn(tokens)
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out


class C3AH(nn.Module):
    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 16 == 0, f"Dimension {c_} should be a multiple of 16."
        num_heads = c_ // 16
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(c_, num_hyperedges, num_heads, 0.1, context)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class FuseModule(nn.Module):
    def __init__(self, c_in, channel_adjust):
        super(FuseModule, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if channel_adjust:
            self.conv_out = Conv(4 * c_in, c_in, 1)
        else:
            self.conv_out = Conv(3 * c_in, c_in, 1)

    def forward(self, x):
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])
        x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
        out = self.conv_out(x_cat)
        return out


class HyperACE(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        num_hyperedges=8,
        dsc3k=True,
        shortcut=False,
        e1=0.5,
        e2=1,
        context="both",
        channel_adjust=True,
    ):
        super().__init__()
        self.c = int(c2 * e1)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            (
                DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7)
                if dsc3k
                else DSBottleneck(self.c, self.c, shortcut=shortcut)
            )
            for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))


class FullPAD_Tunnel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x[0] + self.gate * x[1]


class YOLO13Encoder(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.stem = Conv(in_channels, base_channels, k=3, s=1)

        self.p2 = nn.Sequential(
            DSConv(base_channels, base_channels * 2, k=3, s=(2, 2)),
            DSC3k2(base_channels * 2, base_channels * 2, n=1, dsc3k=True),
        )

        self.p3 = nn.Sequential(
            DSConv(base_channels * 2, base_channels * 4, k=3, s=(2, 2)),
            DSC3k2(base_channels * 4, base_channels * 4, n=2, dsc3k=True),
        )

        self.p4 = nn.Sequential(
            DSConv(base_channels * 4, base_channels * 8, k=3, s=(2, 2)),
            DSC3k2(base_channels * 8, base_channels * 8, n=2, dsc3k=True),
        )

        self.p5 = nn.Sequential(
            DSConv(base_channels * 8, base_channels * 16, k=3, s=(2, 2)),
            DSC3k2(base_channels * 16, base_channels * 16, n=1, dsc3k=True),
        )

        self.out_channels = [
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

    def forward(self, x):
        x = self.stem(x)
        p2 = self.p2(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p2, p3, p4, p5]


class YOLO13FullPADDecoder(nn.Module):
    def __init__(self, encoder_channels, hyperace_out_c, out_channels_final):
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = encoder_channels

        c_d5, c_d4, c_d3, c_d2 = c_p5, c_p4, c_p3, c_p2

        self.h_to_d5 = Conv(hyperace_out_c, c_d5, 1, 1)
        self.h_to_d4 = Conv(hyperace_out_c, c_d4, 1, 1)
        self.h_to_d3 = Conv(hyperace_out_c, c_d3, 1, 1)
        self.h_to_d2 = Conv(hyperace_out_c, c_d2, 1, 1)

        self.tunnel_d5 = FullPAD_Tunnel()
        self.tunnel_d4 = FullPAD_Tunnel()
        self.tunnel_d3 = FullPAD_Tunnel()
        self.tunnel_d2 = FullPAD_Tunnel()

        self.skip_p5 = Conv(c_p5, c_d5, 1, 1)
        self.skip_p4 = Conv(c_p4, c_d4, 1, 1)
        self.skip_p3 = Conv(c_p3, c_d3, 1, 1)
        self.skip_p2 = Conv(c_p2, c_d2, 1, 1)

        self.up_d5 = DSC3k2(c_d5, c_d4, n=1, dsc3k=True)
        self.up_d4 = DSC3k2(c_d4, c_d3, n=1, dsc3k=True)
        self.up_d3 = DSC3k2(c_d3, c_d2, n=1, dsc3k=True)

        self.final_d2 = DSC3k2(c_d2, c_d2, n=1, dsc3k=True)
        self.final_conv = Conv(c_d2, out_channels_final, 1, 1)

    def forward(self, enc_feats, h_ace):
        p2, p3, p4, p5 = enc_feats

        d5 = self.skip_p5(p5)

        h_d5 = self.h_to_d5(
            F.interpolate(
                h_ace, size=d5.shape[2:], mode="bilinear", align_corners=False
            )
        )
        d5 = self.tunnel_d5([d5, h_d5])

        d5_up = F.interpolate(
            d5, size=p4.shape[2:], mode="bilinear", align_corners=False
        )

        d4_pre = self.up_d5(d5_up)
        d4 = d4_pre + self.skip_p4(p4)

        h_d4 = self.h_to_d4(
            F.interpolate(
                h_ace, size=d4.shape[2:], mode="bilinear", align_corners=False
            )
        )
        d4 = self.tunnel_d4([d4, h_d4])

        d4_up = F.interpolate(
            d4, size=p3.shape[2:], mode="bilinear", align_corners=False
        )
        d3_pre = self.up_d4(d4_up)
        d3 = d3_pre + self.skip_p3(p3)

        h_d3 = self.h_to_d3(
            F.interpolate(
                h_ace, size=d3.shape[2:], mode="bilinear", align_corners=False
            )
        )
        d3 = self.tunnel_d3([d3, h_d3])

        d3_up = F.interpolate(
            d3, size=p2.shape[2:], mode="bilinear", align_corners=False
        )
        d2_pre = self.up_d3(d3_up)
        d2 = d2_pre + self.skip_p2(p2)

        h_d2 = self.h_to_d2(
            F.interpolate(
                h_ace, size=d2.shape[2:], mode="bilinear", align_corners=False
            )
        )
        d2 = self.tunnel_d2([d2, h_d2])

        d2 = self.final_d2(d2)
        return self.final_conv(d2)


class DeepUnet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        en_out_channels=16,
        base_channels=64,
        hyperace_k=2,
        hyperace_l=1,
        num_hyperedges=16,
        num_heads=8,
    ):
        super().__init__()

        self.encoder = YOLO13Encoder(in_channels, base_channels)
        enc_ch = self.encoder.out_channels

        c2, c3, c4, c5 = enc_ch

        self.hyperace = HyperACE(
            c1=c4,
            c2=c4,
            n=1,
            num_hyperedges=num_hyperedges,
            dsc3k=True,
            e1=0.5,
            e2=1.0,
            context="both",
            channel_adjust=True,
        )

        self.hyperace.fuse.conv_out = Conv(c3 + c4 + c5, c4, 1)

        self.decoder = YOLO13FullPADDecoder(
            encoder_channels=enc_ch,
            hyperace_out_c=c4,
            out_channels_final=en_out_channels,
        )

    def forward(self, x):
        original_size = x.shape[2:]

        feats = self.encoder(x)
        p2, p3, p4, p5 = feats

        h_ace_input = [p3, p4, p5]

        h_ace = self.hyperace(h_ace_input)

        x_dec = self.decoder(feats, h_ace)

        x_out = F.interpolate(
            x_dec, size=original_size, mode="bilinear", align_corners=False
        )

        return x_out


class E2E0(nn.Module):
    def __init__(
        self,
        hop_length,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(E2E0, self).__init__()
        self.mel = MelSpectrogram(
            N_MELS,
            SAMPLE_RATE,
            WINDOW_LENGTH,
            hop_length,
            None,
            MEL_FMIN,
            MEL_FMAX,
        )

        self.unet = DeepUnet(
            in_channels=in_channels,
            en_out_channels=en_out_channels,
            base_channels=64,
            hyperace_k=2,
            hyperace_l=1,
            num_hyperedges=16,
            num_heads=8,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))

        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
            )

    def forward(self, x):
        mel = self.mel(x.reshape(-1, x.shape[-1]))
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant")
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.unet(mel)
        if n_pad > 0:
            x = x[:, :, :-n_pad, :]
        x = self.cnn(x).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


def to_local_average_cents(salience, center=None, thred=0.0):
    if not hasattr(to_local_average_cents, "cents_mapping"):
        to_local_average_cents.cents_mapping = (
            np.linspace(0, 7180, 360) + 1997.3794084376191
        )

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum if np.max(salience) > thred else 0
    if salience.ndim == 2:
        return np.array(
            [
                to_local_average_cents(salience[i, :], None, thred)
                for i in range(salience.shape[0])
            ]
        )

    raise Exception("label should be either 1d or 2d ndarray")


class HPARMVPEPitchAlgorithm(ContinuousPitchAlgorithm):
    _name = "hpa_rmvpe"

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model_hop_length = 160

        model_path = get_model_path(model_path)
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = E2E0(self.model_hop_length, 4, 1, (2, 2))

        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if hasattr(checkpoint, "module"):
            state_dict = checkpoint.module.state_dict()
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = (
                checkpoint.state_dict()
                if hasattr(checkpoint, "state_dict")
                else checkpoint
            )

        model_dict = model.state_dict()

        filtered_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }

        if len(filtered_dict) == 0:
            print("WARNING: No matching keys found in checkpoint.")
            print(
                "Ensure you are loading a model trained with the HyperACE architecture."
            )

        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)

        model.to(self.device)
        model.eval()
        return model

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        if len(audio.shape) == 2:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

        if self.sample_rate != SAMPLE_RATE:
            try:
                from resampy import resample

                audio = resample(audio, self.sample_rate, SAMPLE_RATE)
            except ImportError:
                from scipy.signal import resample

                target_length = int(len(audio) * SAMPLE_RATE / self.sample_rate)
                audio = resample(audio, target_length).astype(np.float32)

        audio_max = np.max(np.abs(audio))
        if audio_max > 1.0:
            audio = audio / audio_max

        return audio

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        audio_processed = self._preprocess_audio(audio)

        audio_tensor = (
            torch.from_numpy(audio_processed).float().to(self.device).contiguous()
        )

        with torch.inference_mode():
            try:
                pitch_pred = self.model(audio_tensor).squeeze(0)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    audio_tensor = audio_tensor.cpu()
                    self.model = self.model.cpu()
                    pitch_pred = self.model(audio_tensor).squeeze(0)
                    self.model = self.model.cuda()
                else:
                    raise e

        pitch_pred_np = pitch_pred.cpu().numpy()

        del audio_tensor
        del pitch_pred
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        cents = to_local_average_cents(pitch_pred_np, thred=0.0)
        f0 = np.array([10 * (2 ** (cent / 1200)) if cent else 0 for cent in cents])

        periodicity = (
            np.max(pitch_pred_np, axis=1) if pitch_pred_np.ndim > 1 else pitch_pred_np
        )

        model_hopsize_seconds = self.model_hop_length / SAMPLE_RATE
        n_frames = len(f0)
        times = np.arange(n_frames) * model_hopsize_seconds

        return times, f0, periodicity

    def _get_default_threshold(self) -> float:
        return 0.03
