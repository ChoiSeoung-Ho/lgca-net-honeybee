import torch
import torch.nn as nn
import torch.nn.functional as F

##################################
#       SAFE Ablation Modules    #
##################################
# 
# Purpose: Compare activation function strategies in SAFE module
# 
# Ablation configurations:
# 1. Baseline (No SAFE) - Model without SAFE module
# 2. GELU only - Single-path with GELU activation
# 3. ELU only - Single-path with ELU activation
# 4. GELU+ELU (Add) - Dual-path with addition
# 5. GELU⊙ELU (Multiply) - Dual-path with multiplication (Full SAFE)

class SAFEModule_GELU(nn.Module):
    """
    Ablation: GELU activation only (Single-path)
    Full SAFE structure but uses only GELU path
    """
    def __init__(self, in_channels, out_channels=64):
        super(SAFEModule_GELU, self).__init__()
        # Phase 1: Depthwise + Pointwise
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Phase 2: Multi-scale dilated convs
        self.dil_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.dil_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dil_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        
        # Phase 4: Channel attention
        self.se_fc1 = nn.Linear(out_channels * 3, out_channels)
        self.se_fc2 = nn.Linear(out_channels, out_channels * 3)
        
        # Phase 5: Fusion
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)
        
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Phase 1: Depthwise + Pointwise (GELU only)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.gelu(out)
        
        # Phase 2: Multi-scale dilated convs (GELU only)
        dil1 = self.gelu(self.dil_conv1(out))
        dil2 = self.gelu(self.dil_conv2(out))
        dil4 = self.gelu(self.dil_conv4(out))
        multi_scale = torch.cat([dil1, dil2, dil4], dim=1)
        
        # Phase 4: Channel attention
        se = F.adaptive_avg_pool2d(multi_scale, 1).view(multi_scale.size(0), -1)
        se = self.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        se = se.view(multi_scale.size(0), multi_scale.size(1), 1, 1)
        attended = multi_scale * se
        
        # Phase 5: Fusion
        fused = self.fuse(attended)
        return self.relu(fused)

class SAFEModule_ELU(nn.Module):
    """
    Ablation: ELU activation only (Single-path)
    Full SAFE structure but uses only ELU path
    """
    def __init__(self, in_channels, out_channels=64):
        super(SAFEModule_ELU, self).__init__()
        # Phase 1: Depthwise + Pointwise
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Phase 2: Multi-scale dilated convs
        self.dil_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.dil_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dil_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        
        # Phase 4: Channel attention
        self.se_fc1 = nn.Linear(out_channels * 3, out_channels)
        self.se_fc2 = nn.Linear(out_channels, out_channels * 3)
        
        # Phase 5: Fusion
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)
        
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Phase 1: Depthwise + Pointwise (ELU only)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.elu(out)
        
        # Phase 2: Multi-scale dilated convs (ELU only)
        dil1 = self.elu(self.dil_conv1(out))
        dil2 = self.elu(self.dil_conv2(out))
        dil4 = self.elu(self.dil_conv4(out))
        multi_scale = torch.cat([dil1, dil2, dil4], dim=1)
        
        # Phase 4: Channel attention
        se = F.adaptive_avg_pool2d(multi_scale, 1).view(multi_scale.size(0), -1)
        se = self.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        se = se.view(multi_scale.size(0), multi_scale.size(1), 1, 1)
        attended = multi_scale * se
        
        # Phase 5: Fusion
        fused = self.fuse(attended)
        return self.relu(fused)

class SAFEModule_DualPath_Add(nn.Module):
    """
    Ablation: Dual-path with Addition
    Uses both GELU and ELU paths, combines with addition (+)
    """
    def __init__(self, in_channels, out_channels=64):
        super(SAFEModule_DualPath_Add, self).__init__()
        # Phase 1: Depthwise + Pointwise
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Phase 2: Multi-scale dilated convs
        self.dil_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.dil_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dil_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        
        # Phase 4: Channel attention
        self.se_fc1 = nn.Linear(out_channels * 3, out_channels)
        self.se_fc2 = nn.Linear(out_channels, out_channels * 3)
        
        # Phase 5: Fusion
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)
        
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Phase 1: Path 1 (GELU)
        out1 = self.depthwise(x)
        out1 = self.pointwise(out1)
        out1 = self.gelu(out1)
        
        # Phase 2: Multi-scale (GELU path)
        dil1_g = self.gelu(self.dil_conv1(out1))
        dil2_g = self.gelu(self.dil_conv2(out1))
        dil4_g = self.gelu(self.dil_conv4(out1))
        multi_scale1 = torch.cat([dil1_g, dil2_g, dil4_g], dim=1)
        
        # Phase 1: Path 2 (ELU)
        out2 = self.depthwise(x)
        out2 = self.pointwise(out2)
        out2 = self.elu(out2)
        
        # Phase 2: Multi-scale (ELU path)
        dil1_e = self.elu(self.dil_conv1(out2))
        dil2_e = self.elu(self.dil_conv2(out2))
        dil4_e = self.elu(self.dil_conv4(out2))
        multi_scale2 = torch.cat([dil1_e, dil2_e, dil4_e], dim=1)
        
        # Phase 3: Combine with ADDITION (+)
        multi_scale = multi_scale1 + multi_scale2
        
        # Phase 4: Channel attention
        se = F.adaptive_avg_pool2d(multi_scale, 1).view(multi_scale.size(0), -1)
        se = self.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        se = se.view(multi_scale.size(0), multi_scale.size(1), 1, 1)
        attended = multi_scale * se
        
        # Phase 5: Fusion
        fused = self.fuse(attended)
        return self.relu(fused)

class SAFEModule_DualPath_Multiply(nn.Module):
    """
    Ablation: Dual-path with Multiplication (= Full SAFE)
    Uses both GELU and ELU paths, combines with multiplication (⊙)
    This is the full SAFE module with gating mechanism
    """
    def __init__(self, in_channels, out_channels=64):
        super(SAFEModule_DualPath_Multiply, self).__init__()
        # Phase 1: Depthwise + Pointwise
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Phase 2: Multi-scale dilated convs
        self.dil_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.dil_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dil_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        
        # Phase 4: Channel attention
        self.se_fc1 = nn.Linear(out_channels * 3, out_channels)
        self.se_fc2 = nn.Linear(out_channels, out_channels * 3)
        
        # Phase 5: Fusion
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)
        
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Phase 1: Path 1 (GELU)
        out1 = self.depthwise(x)
        out1 = self.pointwise(out1)
        out1 = self.gelu(out1)
        
        # Phase 2: Multi-scale (GELU path)
        dil1_g = self.gelu(self.dil_conv1(out1))
        dil2_g = self.gelu(self.dil_conv2(out1))
        dil4_g = self.gelu(self.dil_conv4(out1))
        multi_scale1 = torch.cat([dil1_g, dil2_g, dil4_g], dim=1)
        
        # Phase 1: Path 2 (ELU)
        out2 = self.depthwise(x)
        out2 = self.pointwise(out2)
        out2 = self.elu(out2)
        
        # Phase 2: Multi-scale (ELU path)
        dil1_e = self.elu(self.dil_conv1(out2))
        dil2_e = self.elu(self.dil_conv2(out2))
        dil4_e = self.elu(self.dil_conv4(out2))
        multi_scale2 = torch.cat([dil1_e, dil2_e, dil4_e], dim=1)
        
        # Phase 3: Gating with MULTIPLICATION (⊙)
        multi_scale = multi_scale1 * multi_scale2
        
        # Phase 4: Channel attention
        se = F.adaptive_avg_pool2d(multi_scale, 1).view(multi_scale.size(0), -1)
        se = self.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        se = se.view(multi_scale.size(0), multi_scale.size(1), 1, 1)
        attended = multi_scale * se
        
        # Phase 5: Fusion
        fused = self.fuse(attended)
        return self.relu(fused)
