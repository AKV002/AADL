"""
3D DenseNet121 with Atlas-Guided Attention Gates
MaxPool expansion + AvgPool downsampling for gating signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, dropout_prob):
        super().__init__()
        mid = bn_size * growth_rate
        self.layers = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm3d(mid),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, growth_rate, kernel_size=3, padding=(1, 3, 3), dilation=(1, 3, 3), bias=False),
        )
        self.dropout = nn.Dropout3d(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        out = self.layers(x)
        if self.dropout:
            out = self.dropout(out)
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, dropout_prob):
        super().__init__()
        for i in range(num_layers):
            self.add_module(
                f"denselayer{i+1}",
                _DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, dropout_prob),
            )


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module("norm", nn.BatchNorm3d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2))


class MaxPoolExpand3D(nn.Module):
    def __init__(self, kernel_size=3, iterations=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.padding = kernel_size // 2

    def forward(self, x):
        for _ in range(self.iterations):
            x = F.max_pool3d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        return x


class AtlasAttentionGate3D(nn.Module):
    def __init__(self, in_channels, atlas_channels=1, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(in_channels // 4, 1)

        self.theta = nn.Conv3d(in_channels, inter_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.phi = nn.Conv3d(atlas_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.W = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(in_channels),
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        nn.init.constant_(self.psi.bias, 3.0)

    def forward(self, x, atlas_expanded):
        theta_x = self.theta(x)
        target_size = theta_x.shape[2:]

        g_down = F.adaptive_avg_pool3d(atlas_expanded, output_size=target_size)
        phi_g = self.phi(g_down)

        f = self.relu(theta_x + phi_g)
        alpha_low = self.sigmoid(self.psi(f))
        alpha = F.interpolate(alpha_low, size=x.shape[2:], mode="trilinear", align_corners=False)

        y = alpha * x
        out = self.W(y)
        return out, alpha


class DenseNet121WithAtlasAttention(nn.Module):
    def __init__(
        self,
        in_channels=1,
        atlas_channels=1,
        out_channels=2,
        init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        bn_size=4,
        dropout_prob=0.0,
        expand_kernel_size=3,
        expand_iterations=1,
    ):
        super().__init__()

        self.atlas_expander = MaxPoolExpand3D(
            kernel_size=expand_kernel_size,
            iterations=expand_iterations,
        )

        self.stem = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv3d(in_channels, init_features, kernel_size=(3, 7, 7), stride=2, padding=(1, 3, 3), bias=False)),
            ("norm0", nn.BatchNorm3d(init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        n = init_features

        self.denseblock1 = _DenseBlock(block_config[0], n, bn_size, growth_rate, dropout_prob)
        n += block_config[0] * growth_rate
        self.transition1 = _Transition(n, n // 2)
        n //= 2

        self.denseblock2 = _DenseBlock(block_config[1], n, bn_size, growth_rate, dropout_prob)
        n += block_config[1] * growth_rate
        self.transition2 = _Transition(n, n // 2)
        n //= 2

        self.denseblock3 = _DenseBlock(block_config[2], n, bn_size, growth_rate, dropout_prob)
        n3 = n + block_config[2] * growth_rate
        self.transition3 = _Transition(n3, n3 // 2)
        n = n3 // 2

        self.denseblock4 = _DenseBlock(block_config[3], n, bn_size, growth_rate, dropout_prob)
        n4 = n + block_config[3] * growth_rate
        self.norm5 = nn.BatchNorm3d(n4)

        self.n3 = n3
        self.n4 = n4

        self.att3 = AtlasAttentionGate3D(in_channels=n3, atlas_channels=atlas_channels, inter_channels=max(n3 // 4, 1))
        self.att4 = AtlasAttentionGate3D(in_channels=n4, atlas_channels=atlas_channels, inter_channels=max(n4 // 4, 1))

        self.norm_att3 = nn.BatchNorm3d(n3)

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        concat_dim = n4 + n3 + n4
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(concat_dim, out_channels),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.att3.psi.bias, 3.0)
        nn.init.constant_(self.att4.psi.bias, 3.0)

    def forward(self, ct, atlas):
        atlas = self.atlas_expander(atlas)

        x = self.stem(ct)

        x = self.denseblock1(x)
        x = self.transition1(x)

        x = self.denseblock2(x)
        x = self.transition2(x)

        x = self.denseblock3(x)
        feat_db3 = x

        x = self.transition3(x)

        x = self.denseblock4(x)
        x = self.norm5(x)
        feat_db4 = x

        y1, att3_map = self.att3(feat_db3, atlas)
        y1 = self.norm_att3(y1)

        y2, att4_map = self.att4(feat_db4, atlas)

        feat_db4_pooled = self.global_pool(feat_db4).flatten(1)
        y1_pooled = self.global_pool(y1).flatten(1)
        y2_pooled = self.global_pool(y2).flatten(1)

        concat_feat = torch.cat([feat_db4_pooled, y1_pooled, y2_pooled], dim=1)

        logits = self.classifier(concat_feat)

        return logits, att3_map, att4_map


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenseNet121WithAtlasAttention(
        in_channels=1,
        atlas_channels=1,
        out_channels=2,
        dropout_prob=0,
        expand_kernel_size=3,
        expand_iterations=1,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ct = torch.randn(1, 1, 256, 256, 64).to(device)
    atlas = (torch.rand(1, 1, 256, 256, 64) > 0.5).float().to(device)

    with torch.no_grad():
        logits, att3, att4 = model(ct, atlas)
        print(f"logits: {logits.shape}")
        print(f"att3:   {att3.shape}")
        print(f"att4:   {att4.shape}")
