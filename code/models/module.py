import torch
import torch.nn as nn
import torch.nn.functional as F

##################################
#              SAFE              #
##################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class MutualInformationRegularizer(nn.Module):
    """
    Shannon Information Theory 기반 정보 중복도(Redundancy) 축소 모듈
    세 가지 활성화 함수의 채널별 상관관계를 최소화하여 상호보완적 정보를 강제함
    """
    def __init__(self):
        super(MutualInformationRegularizer, self).__init__()

    def forward(self, f1, f2, f3):
        # f1, f2, f3 shape: (B, C, H, W)
        B, C, H, W = f1.shape
        N = H * W

        # Spatial Flatten 및 정규화
        z1 = F.normalize(f1.view(B, C, N), dim=2)
        z2 = F.normalize(f2.view(B, C, N), dim=2)
        z3 = F.normalize(f3.view(B, C, N), dim=2)

        # 교차 상관관계 (Cross-correlation) 행렬 계산
        corr_12 = torch.bmm(z1, z2.transpose(1, 2)) / N
        corr_23 = torch.bmm(z2, z3.transpose(1, 2)) / N
        corr_31 = torch.bmm(z3, z1.transpose(1, 2)) / N

        # 대각 요소를 제외한 비대각(Off-diagonal) 중복도 패널티 계산
        # 대각선은 자기 자신의 채널 상관관계이므로 1에 가깝게 둔다
        identity = torch.eye(C, device=f1.device).unsqueeze(0).expand(B, -1, -1)
        
        loss_mi = (corr_12 - identity).pow(2).mean() + \
                  (corr_23 - identity).pow(2).mean() + \
                  (corr_31 - identity).pow(2).mean()
                  
        return loss_mi

class EvidentialUncertaintyEstimator(nn.Module):
    """Subjective Logic 기반 공간적 불확실성 추정기"""
    def __init__(self, channels, K=10.0):
        super(EvidentialUncertaintyEstimator, self).__init__()
        self.K = K
        self.evidence_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        evidence = self.softplus(self.evidence_conv(x))
        S = torch.sum(evidence, dim=1, keepdim=True) + self.K
        uncertainty = self.K / S # (B, 1, H, W)
        return uncertainty

class DynamicGating(nn.Module):
    def __init__(self, channels):
        super(DynamicGating, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 3)
        )

    def forward(self, x1, x2, x3):
        context = x1 + x2 + x3
        B, C, _, _ = context.shape
        z = self.fc(self.gap(context).view(B, C))
        weights = F.softmax(z, dim=1)
        w1, w2, w3 = weights[:, 0:1, None, None], weights[:, 1:2, None, None], weights[:, 2:3, None, None]
        return (x1 * w1) + (x2 * w2) + (x3 * w3)

class KroneckerCrossAttentionFusion(nn.Module):
    def __init__(self, channels):
        super(KroneckerCrossAttentionFusion, self).__init__()
        self.q_conv = nn.Conv2d(channels, channels, 1)
        self.k_conv = nn.Conv2d(channels, channels, 1)
        self.v_conv = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.kron_compress = nn.Sequential(
            nn.Linear(channels * channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )

    def forward(self, local_feat, context_feat):
        B, C, H, W = local_feat.shape
        N = H * W
        
        # 1. Cross-Attention
        Q = self.q_conv(local_feat).view(B, C, N)
        K = self.k_conv(context_feat).view(B, C, N)
        V = self.v_conv(context_feat).view(B, C, N)

        attn = F.softmax(torch.bmm(Q.transpose(1, 2), K) * self.scale, dim=-1)
        ca_feat = torch.bmm(V, attn.transpose(1, 2)).view(B, C, H, W)
        
        # 2. Kronecker Bilinear Pooling
        z_loc = self.gap(local_feat).view(B, C, 1)
        z_ca = self.gap(ca_feat).view(B, 1, C)
        
        kron_prod = torch.bmm(z_loc, z_ca).view(B, -1)
        weights = self.kron_compress(kron_prod).view(B, C, 1, 1)
        
        return (local_feat * ca_feat) * weights + local_feat + ca_feat

class SAFEModule_Apex(nn.Module):
    """
    MI 극대화 및 불확실성(EDL) 제어가 완비된 최고 등급 SAFE 모듈
    """
    def __init__(self, in_channels, out_channels=64):
        super(SAFEModule_Apex, self).__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu_base = nn.GELU()

        # Uncertainty & MI Regularizer
        self.uncertainty_estimator = EvidentialUncertaintyEstimator(out_channels)
        self.mi_regularizer = MutualInformationRegularizer()

        # 3가지 이질적 비선형 분기 (GELU, ELU, ReLU)
        self.dil_conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1)
        self.dil_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2)
        self.dil_conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4)
        
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.relu = nn.ReLU() # 논문 설계 방향에 맞춘 ReLU 도입

        self.dynamic_gating = DynamicGating(out_channels)
        self.context_conv = nn.Conv2d(out_channels, out_channels, 3, padding=6, dilation=6)
        self.fusion = KroneckerCrossAttentionFusion(out_channels)
        
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        reduced = self.gelu_base(self.bn(self.reduce(x)))

        # 1. 공간적 불확실성 계산
        U = self.uncertainty_estimator(reduced)

        # 2. 3-Branch 비선형 특징 추출
        f_gelu = self.gelu(self.dil_conv1(reduced))
        f_elu = self.elu(self.dil_conv2(reduced))
        f_relu = self.relu(self.dil_conv4(reduced))

        # 3. 보조 손실(Auxiliary Loss) - MI Redundancy Penalty 계산
        # 학습 중에만 계산하여 연산 효율성 확보
        mi_loss = 0.0
        if self.training:
            mi_loss = self.mi_regularizer(f_gelu, f_elu, f_relu)

        # 4. Dynamic Gating & Cross-Attention Kronecker Fusion
        local_fused = self.dynamic_gating(f_gelu, f_elu, f_relu)
        context_feat = self.gelu_base(self.context_conv(reduced))
        complex_fusion = self.fusion(local_fused, context_feat)

        # 5. 베이지안 융합 제어 (Uncertainty-Aware Fusion)
        uncertainty_controlled_feat = (1 - U) * complex_fusion + U * reduced

        out = uncertainty_controlled_feat + self.skip(x)
        return self.gelu_base(out), mi_loss

def add_noise_safe(tensor, noise_std_ratio=0.6):
    std = torch.std(tensor)
    noise_std = std * noise_std_ratio
    noise = torch.randn_like(tensor) * noise_std
    return tensor + noise

class SAFEModule(nn.Module):
    def __init__(self, in_channels, out_channels=64, noise_std_ratio=0.6):
        super(SAFEModule, self).__init__()
        self.noise_std_ratio = noise_std_ratio

        # Depthwise Separable Conv
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        # Multi-scale dilated convs
        self.dil_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.dil_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dil_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)

        # Channel attention (Squeeze-Excitation)
        self.se_fc1 = nn.Linear(out_channels * 3, out_channels)
        self.se_fc2 = nn.Linear(out_channels, out_channels * 3)

        # Skip connection
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.elu = nn.ELU()
        self.celu = nn.CELU()
        self.gelu = nn.GELU()
        self.mish = nn.Mish()
        self.logsigmoid = nn.LogSigmoid()
        self.dil_conv1_lazy = nn.LazyConv2d(out_channels, kernel_size=3, padding=1, dilation=1)
        self.dil_conv2_lazy = nn.LazyConv2d(out_channels, kernel_size=3, padding=2, dilation=2)
        self.dil_conv4_lazy = nn.LazyConv2d(out_channels, kernel_size=3, padding=4, dilation=4)
        self.pwconv1 = nn.Linear(out_channels, 4 * out_channels)
        self.pwconv2 = nn.Linear(4*out_channels, out_channels)
        self.logsigmoid = nn.LogSigmoid()
        self.lrn = nn.LocalResponseNorm(1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(out_channels))
    def forward(self, x):
        # Depthwise + Pointwise
        out = self.depthwise(x)
        out = self.lrn(out)
        out = self.pointwise(out)
        out = self.gelu(out)

        dil1 = self.gelu(self.dil_conv1(out))
        dil1 = self.lrn(dil1)
        dil1 = self.gelu(dil1)

        dil2 = self.gelu(self.dil_conv2(out))
        dil2 = self.lrn(dil2)
        dil2 = self.gelu(dil2)

        dil4 = self.gelu(self.dil_conv4(out))
        dil4 = self.lrn(dil4)
        dil4 = self.gelu(dil4)

        multi_scale1 = torch.cat([dil1, dil2, dil4], dim=1)
        multi_scale1 = self.lrn(multi_scale1)
        multi_scale = self.gelu(multi_scale1)

        # Channel attention (SE block)
        se = F.adaptive_avg_pool2d(multi_scale, 1).view(multi_scale.size(0), -1)
        se = self.mish(self.se_fc1(se))
        se = self.logsigmoid(self.se_fc2(se))
        se = se.view(multi_scale.size(0), multi_scale.size(1), 1, 1)
        attended = multi_scale * se
        fused = self.mish(attended)
        return fused

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 차원 축소를 위한 1x1 Conv
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAMBlock(nn.Module):
    """표준 CBAM (Convolutional Block Attention Module)"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

import torch
import torch.nn as nn

class M_SAFE_Fusion_Module(nn.Module):
    """공간-채널 특징 병렬 융합 모듈"""
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        
        # 앞서 추가하신 CBAMBlock 사용
        self.cbam = CBAMBlock(in_channels)
        
        # 2. Parallel Feature Extraction
        # ConvNeXt는 64 채널 반환
        self.convnext_path = ConvNeXtStyleModule(in_channels, out_channels)
        # SAFEModule은 내부 다중 스케일 결합으로 인해 192 (64 * 3) 채널 반환
        self.safe_path = SAFEModule(in_channels, out_channels)
        
        # 3. Feature Fusion (특징 융합)
        # 64 + 192 = 256 채널을 입력으로 받아 다시 64 채널로 압축
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False), # 여기가 수정됨 (out_channels * 4)
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        # x shape: (B, 1280, H, W)
        x = self.cbam(x)
        
        feat_local = self.convnext_path(x)  # shape: (B, 64, H, W)
        feat_global = self.safe_path(x)     # shape: (B, 192, H, W)
        
        # 채널 방향 결합 (64 + 192 = 256)
        fused = torch.cat([feat_local, feat_global], dim=1) 
        
        # 융합 및 차원 축소 (256 -> 64)
        out = self.fusion_conv(fused)
        return out

class ConvNeXtStyleModule(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        
        # 1. 입력 차원이 다를 경우 맞춰주는 역할 (1x1 Conv)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # 2. Large Kernel Depthwise Conv (수용 영역 확장)
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels)
        
        # 3. LayerNorm (PyTorch의 LayerNorm은 마지막 차원을 기준으로 정규화함)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        
        # 4. Inverted Bottleneck (특징 추출 극대화)
        self.pwconv1 = nn.Linear(out_channels, 4 * out_channels) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * out_channels, out_channels)
        
        # 5. Layer Scale (학습 안정화)
        self.gamma = nn.Parameter(1e-6 * torch.ones((out_channels)), requires_grad=True)

    def forward(self, x):
        # 1. 차원 맞추기 및 잔차(Residual) 저장
        # x 차원: (B, in_channels, H, W) -> (B, out_channels, H, W)
        x = self.proj(x)
        res = x 

        # 2. Depthwise Conv
        # x 차원 유지: (B, out_channels, H, W)
        x = self.dwconv(x)

        # 3. LayerNorm 및 Linear 연산을 위한 차원 변경
        # (B, out_channels, H, W) -> (B, H, W, out_channels)
        x = x.permute(0, 2, 3, 1)
        
        # 이제 마지막 차원이 out_channels(예: 192)이므로 LayerNorm 정상 작동
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer Scale 적용
        x = self.gamma * x

        # 4. 원래 차원으로 복구
        # (B, H, W, out_channels) -> (B, out_channels, H, W)
        x = x.permute(0, 3, 1, 2)

        # 5. 잔차 연결
        return x + res
