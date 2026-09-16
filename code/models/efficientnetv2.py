import copy
from functools import partial
from collections import OrderedDict
import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    """Convolution-Normalization-Activation Module"""
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups, norm_layer, act, conv_layer=nn.Conv2d):
        super(ConvBNAct, self).__init__(
            conv_layer(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=groups, bias=False),
            norm_layer(out_channel),
            act()
        )


class SEUnit(nn.Module):
    """Squeeze-Excitation Unit

    paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper

    """
    def __init__(self, in_channel, reduction_ratio=4, act1=partial(nn.SiLU, inplace=True), act2=nn.Sigmoid):
        super(SEUnit, self).__init__()
        hidden_dim = in_channel // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channel, hidden_dim, (1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, in_channel, (1, 1), bias=True)
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x):
        return x * self.act2(self.fc2(self.act1(self.fc1(self.avg_pool(x)))))


class StochasticDepth(nn.Module):
    """StochasticDepth

    paper: https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39

    :arg
        - prob: Probability of dying
        - mode: "row" or "all". "row" means that each row survives with different probability
    """
    def __init__(self, prob, mode):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
            return x * torch.empty(shape).bernoulli_(self.survival).div_(self.survival).to(x.device)


class MBConvConfig:
    """EfficientNet Building block configuration"""
    def __init__(self, expand_ratio: float, kernel: int, stride: int, in_ch: int, out_ch: int, layers: int,
                 use_se: bool, fused: bool, act=nn.SiLU, norm_layer=nn.BatchNorm2d):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_layers = layers
        self.act = act
        self.norm_layer = norm_layer
        self.use_se = use_se
        self.fused = fused

    @staticmethod
    def adjust_channels(channel, factor, divisible=8):
        new_channel = channel * factor
        divisible_channel = max(divisible, (int(new_channel + divisible / 2) // divisible) * divisible)
        divisible_channel += divisible if divisible_channel < 0.9 * new_channel else 0
        return divisible_channel


class MBConv(nn.Module):
    """EfficientNet main building blocks

    :arg
        - c: MBConvConfig instance
        - sd_prob: stochastic path probability
    """
    def __init__(self, c, sd_prob=0.0):
        super(MBConv, self).__init__()
        inter_channel = c.adjust_channels(c.in_ch, c.expand_ratio)
        block = []

        if c.expand_ratio == 1:
            block.append(('fused', ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
        elif c.fused:
            block.append(('fused', ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
            block.append(('fused_point_wise', ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))
        else:
            block.append(('linear_bottleneck', ConvBNAct(c.in_ch, inter_channel, 1, 1, 1, c.norm_layer, c.act)))
            block.append(('depth_wise', ConvBNAct(inter_channel, inter_channel, c.kernel, c.stride, inter_channel, c.norm_layer, c.act)))
            block.append(('se', SEUnit(inter_channel, 4 * c.expand_ratio)))
            block.append(('point_wise', ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))

        self.block = nn.Sequential(OrderedDict(block))
        self.use_skip_connection = c.stride == 1 and c.in_ch == c.out_ch
        self.stochastic_path = StochasticDepth(sd_prob, "row")

    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out


class EfficientNetV2(nn.Module):
    """Pytorch Implementation of EfficientNetV2

    paper: https://arxiv.org/abs/2104.00298

    - reference 1 (pytorch): https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    - reference 2 (official): https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_configs.py

    :arg
        - layer_infos: list of MBConvConfig
        - out_channels: bottleneck channel
        - nlcass: number of class
        - dropout: dropout probability before classifier layer
        - stochastic depth: stochastic depth probability
    """
    def __init__(self, layer_infos, out_channels=1280, nclass=0, dropout=0.2, stochastic_depth=0.0,
                 block=MBConv, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super(EfficientNetV2, self).__init__()
        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer

        self.in_channel = layer_infos[0].in_ch
        self.final_stage_channel = layer_infos[-1].out_ch
        self.out_channels = out_channels

        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.stem = ConvBNAct(3, self.in_channel, 3, 2, 1, self.norm_layer, self.act)
        self.blocks = nn.Sequential(*self.make_stages(layer_infos, block))
        self.head = nn.Sequential(OrderedDict([
            ('bottleneck', ConvBNAct(self.final_stage_channel, out_channels, 1, 1, 1, self.norm_layer, self.act)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
            ('dropout', nn.Dropout(p=dropout, inplace=True)),
            ('classifier', nn.Linear(out_channels, nclass) if nclass else nn.Identity())
        ]))

    def make_stages(self, layer_infos, block):
        return [layer for layer_info in layer_infos for layer in self.make_layers(copy.copy(layer_info), block)]

    def make_layers(self, layer_info, block):
        layers = []
        for i in range(layer_info.num_layers):
            layers.append(block(layer_info, sd_prob=self.get_sd_prob()))
            layer_info.in_ch = layer_info.out_ch
            layer_info.stride = 1
        return layers

    def get_sd_prob(self):
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_block)
        self.cur_block += 1
        return sd_prob

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head.bottleneck(x)
        x = self.head.avgpool(x)
        x = self.head.flatten(x)
        return x

    def forward_spatial_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head.bottleneck(x)
        # avgpool과 flatten을 거치지 않고 2D 상태(B, C, H, W)로 반환
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head.dropout(x)
        x = self.head.classifier(x)
        return x

    def change_dropout_rate(self, p):
        self.head[-2] = nn.Dropout(p=p, inplace=True)


def efficientnet_v2_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)


def get_efficientnet_v2_structure(model_name):
    """Get EfficientNetV2 model structure configurations"""
    if 'efficientnet_v2_s' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 24, 24, 2, False, True),
            (4, 3, 2, 24, 48, 4, False, True),
            (4, 3, 2, 48, 64, 4, False, True),
            (4, 3, 2, 64, 128, 6, True, False),
            (6, 3, 1, 128, 160, 9, True, False),
            (6, 3, 2, 160, 256, 15, True, False),
        ]
    elif 'efficientnet_v2_m' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 24, 24, 3, False, True),
            (4, 3, 2, 24, 48, 5, False, True),
            (4, 3, 2, 48, 80, 5, False, True),
            (4, 3, 2, 80, 160, 7, True, False),
            (6, 3, 1, 160, 176, 14, True, False),
            (6, 3, 2, 176, 304, 18, True, False),
            (6, 3, 1, 304, 512, 5, True, False),
        ]
    elif 'efficientnet_v2_l' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 32, 32, 4, False, True),
            (4, 3, 2, 32, 64, 7, False, True),
            (4, 3, 2, 64, 96, 7, False, True),
            (4, 3, 2, 96, 192, 10, True, False),
            (6, 3, 1, 192, 224, 19, True, False),
            (6, 3, 2, 224, 384, 25, True, False),
            (6, 3, 1, 384, 640, 7, True, False),
        ]
    elif 'efficientnet_v2_xl' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 32, 32, 4, False, True),
            (4, 3, 2, 32, 64, 8, False, True),
            (4, 3, 2, 64, 96, 8, False, True),
            (4, 3, 2, 96, 192, 16, True, False),
            (6, 3, 1, 192, 256, 24, True, False),
            (6, 3, 2, 256, 512, 32, True, False),
            (6, 3, 1, 512, 640, 8, True, False),
        ]


def get_efficientnet_v2(model_name, pretrained=False, nclass=0, dropout=0.1, stochastic_depth=0.2, **kwargs):
    """Create EfficientNetV2 model"""
    residual_config = [MBConvConfig(*layer_config) for layer_config in get_efficientnet_v2_structure(model_name)]
    model = EfficientNetV2(residual_config, 1280, nclass, dropout=dropout, stochastic_depth=stochastic_depth, block=MBConv, act_layer=nn.SiLU)
    efficientnet_v2_init(model)
    return model


class EfficientNetV2ForImageClassification(nn.Module):
    """EfficientNetV2 wrapper for image classification compatible with the project"""
    
    def __init__(self, num_labels=7, img_size=224, patch_size=16, hidden_dim=512, 
                 model_variant='s'):
        super(EfficientNetV2ForImageClassification, self).__init__()
        
        # Model configurations for different variants
        configs = {
            's': {
                'model_name': 'efficientnet_v2_s',
                'dropout': 0.2,
                'stochastic_depth': 0.2
            },
            'm': {
                'model_name': 'efficientnet_v2_m',
                'dropout': 0.3,
                'stochastic_depth': 0.3
            },
            'l': {
                'model_name': 'efficientnet_v2_l',
                'dropout': 0.4,
                'stochastic_depth': 0.4
            },
            'xl': {
                'model_name': 'efficientnet_v2_xl',
                'dropout': 0.4,
                'stochastic_depth': 0.5
            }
        }
        
        config = configs.get(model_variant, configs['s'])
        
        # Create backbone without classification head
        self.backbone = get_efficientnet_v2(
            model_name=config['model_name'],
            pretrained=False,
            nclass=0,  # No classification head in backbone
            dropout=config['dropout'],
            stochastic_depth=config['stochastic_depth']
        )
        
        # Replace classifier with identity
        self.backbone.head.classifier = nn.Identity()
        
        # Calculate output dimension - EfficientNetV2 uses 1280 as bottleneck
        num_features = 1280
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, images, labels=None):
        # Extract features
        features = self.backbone.forward_features(images)
        
        # Classification
        logits = self.classifier(features)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits

from models.module import SAFEModule,ConvNeXtStyleModule,M_SAFE_Fusion_Module
from models.safe_ablation import (
    SAFEModule_GELU, SAFEModule_ELU,
    SAFEModule_DualPath_Add, SAFEModule_DualPath_Multiply
)

class EfficientNetV2SAFE_Ablation(nn.Module):
    """EfficientNetV2 with SAFE Ablation for different configurations"""
    
    def __init__(self, num_labels=2, img_size=224, patch_size=16, hidden_dim=512,
                 model_variant='s', safe_type='none'):
        """
        safe_type: 
            'none' - Baseline (No SAFE)
            'gelu' - GELU activation only
            'elu' - ELU activation only
            'add' - Dual-path with Addition
            'multiply' or 'full' - Dual-path with Multiplication (Full SAFE)
        """
        super(EfficientNetV2SAFE_Ablation, self).__init__()
        
        configs = {
            's': {'model_name': 'efficientnet_v2_s', 'dropout': 0.2, 'stochastic_depth': 0.2},
            'm': {'model_name': 'efficientnet_v2_m', 'dropout': 0.3, 'stochastic_depth': 0.3},
            'l': {'model_name': 'efficientnet_v2_l', 'dropout': 0.4, 'stochastic_depth': 0.4},
        }
        
        config = configs.get(model_variant, configs['s'])
        
        self.backbone = get_efficientnet_v2(
            model_name=config['model_name'],
            pretrained=False,
            nclass=0,
            dropout=config['dropout'],
            stochastic_depth=config['stochastic_depth']
        )
        
        self.backbone.head.classifier = nn.Identity()
        num_features = 1280
        
        self.safe_type = safe_type
        if safe_type == 'none':
            self.safe_module = None
        elif safe_type == 'gelu':
            self.safe_module = SAFEModule_GELU(num_features, out_channels=64)
        elif safe_type == 'elu':
            self.safe_module = SAFEModule_ELU(num_features, out_channels=64)
        elif safe_type == 'add':
            self.safe_module = SAFEModule_DualPath_Add(num_features, out_channels=64)
        elif safe_type in ['multiply', 'full']:
            self.safe_module = SAFEModule_DualPath_Multiply(num_features, out_channels=64)
        else:
            raise ValueError(f"Unknown safe_type: {safe_type}. Use 'none', 'gelu', 'elu', 'add', or 'multiply'/'full'")
        
        if self.safe_module is not None:
            self.bridge_conv = nn.Conv2d(64, num_features, 1)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )
    
    def forward(self, images, labels=None):
        x = self.backbone.stem(images)
        x = self.backbone.blocks(x)
        x = self.backbone.head.bottleneck(x)
        
        if self.safe_module is not None:
            x = self.safe_module(x)
            x = self.bridge_conv(x)
        
        x = self.backbone.head.avgpool(x)
        x = self.backbone.head.flatten(x)
        logits = self.classifier(x)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits

class EfficientNetV2ForImageClassification_v2__(nn.Module):
    """EfficientNetV2 wrapper for image classification compatible with the project"""

    def __init__(self, num_labels=7, img_size=224, patch_size=16, hidden_dim=512,
                 model_variant='s'):
        super(EfficientNetV2ForImageClassification_v2, self).__init__()

        # Model configurations for different variants
        configs = {
            's': {
                'model_name': 'efficientnet_v2_s',
                'dropout': 0.2,
                'stochastic_depth': 0.2
            },
            'm': {
                'model_name': 'efficientnet_v2_m',
                'dropout': 0.3,
                'stochastic_depth': 0.3
            },
            'l': {
                'model_name': 'efficientnet_v2_l',
                'dropout': 0.4,
                'stochastic_depth': 0.4
            },
            'xl': {
                'model_name': 'efficientnet_v2_xl',
                'dropout': 0.4,
                'stochastic_depth': 0.5
            }
        }

        config = configs.get(model_variant, configs['s'])

        # Create backbone without classification head
        self.backbone = get_efficientnet_v2(
            model_name=config['model_name'],
            pretrained=False,
            nclass=0,  # No classification head in backbone
            dropout=config['dropout'],
            stochastic_depth=config['stochastic_depth']
        )

        # Replace classifier with identity
        self.backbone.head.classifier = nn.Identity()

        # Calculate output dimension - EfficientNetV2 uses 1280 as bottleneck
        num_features = 1280

        # CBAM
        from models.deep_model import CBAMBlock
        self.cbam_channels = num_features
        self.cbam = CBAMBlock(self.cbam_channels)

        # SAFE module
        self.safe_module = SAFEModule(self.cbam_channels, out_channels=64)
        self.convnext_module = ConvNeXtStyleModule(self.cbam_channels, out_channels=64)
        self.m_safe_fusion = M_SAFE_Fusion_Module(in_channels=num_features, out_channels=64)

        # Calculate FC input dimension after SAFE
        with torch.no_grad():
            dummy_features = torch.zeros(1, self.cbam_channels)
            dummy_features_2d = dummy_features.unsqueeze(-1).unsqueeze(-1)
            x = self.cbam(dummy_features_2d)
            x = self.convnext_module(x)
            feature_dim_fc = x.shape[1]

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim_fc),
            nn.Linear(feature_dim_fc, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        
        )
    def forward(self, images, labels=None):
        # 1. 1차원 벡터가 아닌, 공간 정보가 살아있는 2D 피처맵 추출
        spatial_features = self.backbone.forward_spatial_features(images)

        # 2. 제안 모듈 통과 (이 과정에서 대형 커널과 팽창 합성곱이 제대로 작동함)
        x = self.m_safe_fusion(spatial_features)

        # 3. 마지막 단계에서 GAP 적용
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        features = torch.flatten(x, 1)

        # 4. 분류기 통과
        logits = self.classifier(features)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits
import torch
import torch.nn as nn

class EfficientNetV2ForImageClassification_v2_(nn.Module):
    def __init__(self, num_labels=7, img_size=224, patch_size=16, hidden_dim=512, model_variant='s'):
        super(EfficientNetV2ForImageClassification_v2, self).__init__()
        
        # 1. 백본(Backbone) 설정 (누락되었던 부분 복구)
        configs = {
            's': {'model_name': 'efficientnet_v2_s', 'dropout': 0.2, 'stochastic_depth': 0.2},
            'm': {'model_name': 'efficientnet_v2_m', 'dropout': 0.3, 'stochastic_depth': 0.3},
            'l': {'model_name': 'efficientnet_v2_l', 'dropout': 0.4, 'stochastic_depth': 0.4},
            'xl': {'model_name': 'efficientnet_v2_xl', 'dropout': 0.4, 'stochastic_depth': 0.5}
        }
        config = configs.get(model_variant, configs['s'])
        
        # 여기서 self.backbone이 생성됩니다. (안정적인 학습을 위해 pretrained=True 강력 권장)
        self.backbone = get_efficientnet_v2(
            model_name=config['model_name'],
            pretrained=True, 
            nclass=0,  
            dropout=config['dropout'],
            stochastic_depth=config['stochastic_depth']
        )
        self.backbone.head.classifier = nn.Identity()
        
        # 2. 제안 모듈(M-SAFE) 및 분류기(Classifier) 설정
        num_features = 1280
        fusion_out_channels = 128 
        
        # 제안 모듈
        self.m_safe_fusion = M_SAFE_Fusion_Module(in_channels=num_features, out_channels=fusion_out_channels)
        
        # 기존 1280채널 + 융합 모듈 128채널 = 1408채널
        feature_dim_fc = num_features + fusion_out_channels 

        # 최종 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim_fc),
            nn.Linear(feature_dim_fc, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, images, labels=None):
        # 1. 공간 정보가 살아있는 2D 피처맵 추출 (B, 1280, H, W)
        spatial_features = self.backbone.forward_spatial_features(images)

        # 2. 제안 모듈 통과 (보조 특징 추출) -> (B, 128, H, W)
        fusion_features = self.m_safe_fusion(spatial_features)

        # 3. GAP (Global Average Pooling)
        orig_gap = torch.nn.functional.adaptive_avg_pool2d(spatial_features, (1, 1))
        orig_flat = torch.flatten(orig_gap, 1) # (B, 1280)
        
        fusion_gap = torch.nn.functional.adaptive_avg_pool2d(fusion_features, (1, 1))
        fusion_flat = torch.flatten(fusion_gap, 1) # (B, 128)

        # 4. 특징 병합 (Concatenation) -> (B, 1408)
        features = torch.cat([orig_flat, fusion_flat], dim=1)

        # 5. 분류기 통과
        logits = self.classifier(features)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """CNN 특징과 Transformer 특징을 융합하는 핵심 모듈"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_cnn, x_trans):
        # x_cnn (Query): (B, N, C) / x_trans (Key, Value): (B, N, C)
        B, N, C = x_cnn.shape
        
        q = self.q(x_cnn).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x_trans).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_trans).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EfficientNetV2ForImageClassification_v2(nn.Module):
    """
    기존 파이프라인 호환성을 유지하기 위한 CAT-Net 래퍼(Wrapper) 클래스
    호출 규격: EfficientNetV2ForImageClassification_v2(num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s')
    """
    def __init__(self, num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s'):
        super().__init__()
        
        # 파라미터 매핑: variant에 따라 임베딩 차원 조절 가능 (기본 256)
        embed_dim = 256 if model_variant == 's' else 512
        
        # 1. CNN Branch (Local Stream)
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            # 다양한 img_size에도 Transformer 입력 시퀀스 길이(14x14=196)를 고정하기 위한 풀링
            nn.AdaptiveAvgPool2d((14, 14)) 
        )
        
        self.seq_len = 14 * 14 # 196
        
        # 2. Transformer Branch (Global Stream)
        self.trans_proj = nn.Linear(embed_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        self.trans_block = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4, batch_first=True
        )
        
        # 3. Hybrid Fusion Layer (Cross-Attention)
        self.cross_attn = CrossAttention(dim=embed_dim)
        self.norm_fusion = nn.LayerNorm(embed_dim)
        
        # 4. Final Classification Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), # 인자로 받은 hidden_dim 사용
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels) # 인자로 받은 num_labels 사용
        )

    def forward(self, images, labels=None):
        x = images
        
        # Local Features (CNN)
        feat_cnn = self.cnn_stem(x) # (B, embed_dim, 14, 14)
        feat_cnn_flat = feat_cnn.flatten(2).transpose(1, 2) # (B, 196, embed_dim)
        
        # Global Features (Transformer)
        feat_trans = self.trans_proj(feat_cnn_flat) + self.pos_embed
        feat_trans = self.trans_block(feat_trans) # (B, 196, embed_dim)
        
        # Cross-Attention Fusion
        fused_feat = self.cross_attn(feat_cnn_flat, feat_trans)
        fused_feat = self.norm_fusion(fused_feat + feat_cnn_flat) # Residual connection
        
        # Global Representation & Classifier
        out = self.avgpool(fused_feat.transpose(1, 2)).flatten(1)
        logits = self.classifier(out)
        
        # 학습 파이프라인에서 labels를 넘겨주는 경우 손실(Loss) 함께 반환
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
            
        return logits

import torch
import torch.nn as nn
# (CrossAttention 모듈은 외부에서 선언되어 있다고 가정합니다)

class Hybrid_CNN_Transformer_AblationModel(nn.Module):
    """
    하이브리드 CNN-Transformer 모델의 Ablation Study를 위한 클래스
    호출 시 플래그(use_transformer, use_cross_attn)를 통해 구조를 제어합니다.
    """
    def __init__(self, num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s',
                 use_transformer=True, use_cross_attn=True):
        super().__init__()
        
        self.use_transformer = use_transformer
        self.use_cross_attn = use_cross_attn
        embed_dim = 256 if model_variant == 's' else 512
        
        # 1. CNN Branch (Local Stream) - 항상 사용 (Backbone 역할)
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.AdaptiveAvgPool2d((14, 14)) 
        )
        self.seq_len = 14 * 14 # 196
        
        # 2. Transformer Branch (Global Stream) - 조건부 생성
        if self.use_transformer:
            self.trans_proj = nn.Linear(embed_dim, embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
            self.trans_block = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4, batch_first=True
            )
        
        # 3. Cross-Attention Fusion - 조건부 생성
        if self.use_cross_attn:
            self.cross_attn = CrossAttention(dim=embed_dim)
            
        self.norm_fusion = nn.LayerNorm(embed_dim)
        
        # 4. Final Classification Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, images, labels=None):
        x = images
        
        # 1. Local Features (CNN)
        feat_cnn = self.cnn_stem(x) 
        feat_cnn_flat = feat_cnn.flatten(2).transpose(1, 2) 
        
        fused_feat = feat_cnn_flat # 기본값 (Transformer 미사용 시)
        
        # 2 & 3. Transformer & Fusion 
        if self.use_transformer:
            feat_trans = self.trans_proj(feat_cnn_flat) + self.pos_embed
            feat_trans = self.trans_block(feat_trans) 
            
            if self.use_cross_attn:
                # 제안 구조: Cross-Attention을 통한 융합
                attn_out = self.cross_attn(feat_cnn_flat, feat_trans)
                fused_feat = self.norm_fusion(attn_out + feat_cnn_flat)
            else:
                # Ablation: Cross-Attention 제외, 특징 간 단순 요소별 합산(Addition) 처리
                fused_feat = self.norm_fusion(feat_trans + feat_cnn_flat)
        else:
            # Ablation: Transformer 제외 (CNN 특징만 정규화 후 통과)
            fused_feat = self.norm_fusion(feat_cnn_flat)
            
        # 4. Global Representation & Classifier
        out = self.avgpool(fused_feat.transpose(1, 2)).flatten(1)
        logits = self.classifier(out)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
            
        return logits

import torch
import torch.nn as nn

# (이전 파일들에 정의된 구성 요소들을 가져온다고 가정합니다)
from models.module import CBAMBlock, ConvNeXtStyleModule, SAFEModule
#from efficientnetv2 import get_efficientnet_v2

class M_SAFE_Ablation_Module(nn.Module):
    """
    Ablation Study를 위한 M-SAFE 모듈
    인자를 통해 특정 구성 요소(CBAM, Local Path, Global Path)를 끄거나 켤 수 있습니다.
    """
    def __init__(self, in_channels, out_channels=64, 
                 use_cbam=True, 
                 use_local_path=True, 
                 use_global_path=True):
        super().__init__()
        
        self.use_cbam = use_cbam
        self.use_local_path = use_local_path
        self.use_global_path = use_global_path
        
        if not (self.use_local_path or self.use_global_path):
            raise ValueError("Local Path와 Global Path 중 최소 하나는 활성화되어야 합니다.")

        # 1. CBAM (선택적)
        if self.use_cbam:
            self.cbam = CBAMBlock(in_channels)
        else:
            self.cbam = nn.Identity()
            
        # 2. Parallel Feature Extraction (선택적)
        concat_channels = 0
        
        if self.use_local_path:
            self.convnext_path = ConvNeXtStyleModule(in_channels, out_channels)
            concat_channels += out_channels  # 64
            
        if self.use_global_path:
            self.safe_path = SAFEModule(in_channels, out_channels)
            concat_channels += (out_channels * 3)  # SAFE 모듈은 192 (64 * 3) 반환
            
        # 3. Feature Fusion (활성화된 경로에 맞춰 채널 수 동적 조절)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        # 1. CBAM
        x = self.cbam(x)
        
        features = []
        
        # 2. Paths
        if self.use_local_path:
            feat_local = self.convnext_path(x)
            features.append(feat_local)
            
        if self.use_global_path:
            feat_global = self.safe_path(x)
            features.append(feat_global)
            
        # 3. Concatenate & Fusion
        fused = torch.cat(features, dim=1) 
        out = self.fusion_conv(fused)
        
        return out


class EfficientNetV2_MSAFE_AblationModel(nn.Module):
    """
    Ablation Study용 최종 분류 모델
    """
    def __init__(self, num_labels=2, model_variant='s', hidden_dim=512,
                 use_cbam=True, use_local_path=True, use_global_path=True):
        """
        논문 실험 세팅 가이드:
        1. Base: use_cbam=False, use_local_path=False, use_global_path=False (오류 발생 방지를 위해 하나는 켜야함, 보통 백본만 따로 테스트)
        2. w/o CBAM: use_cbam=False, use_local=True, use_global=True
        3. w/o Local (ConvNeXt): use_cbam=True, use_local=False, use_global=True
        4. w/o Global (SAFE): use_cbam=True, use_local=True, use_global=False
        5. Proposed (Full): use_cbam=True, use_local=True, use_global=True
        """
        super().__init__()
        
        configs = {
            's': {'model_name': 'efficientnet_v2_s', 'dropout': 0.2, 'stochastic_depth': 0.2},
            'm': {'model_name': 'efficientnet_v2_m', 'dropout': 0.3, 'stochastic_depth': 0.3},
            'l': {'model_name': 'efficientnet_v2_l', 'dropout': 0.4, 'stochastic_depth': 0.4},
        }
        config = configs.get(model_variant, configs['s'])
        
        # 백본 로드
        self.backbone = get_efficientnet_v2(
            model_name=config['model_name'], pretrained=True, nclass=0,
            dropout=config['dropout'], stochastic_depth=config['stochastic_depth']
        )
        self.backbone.head.classifier = nn.Identity()
        
        num_features = 1280
        fusion_out_channels = 64
        
        # Ablation 제어용 M-SAFE 모듈
        self.ablation_module = M_SAFE_Ablation_Module(
            in_channels=num_features, 
            out_channels=fusion_out_channels,
            use_cbam=use_cbam,
            use_local_path=use_local_path,
            use_global_path=use_global_path
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_out_channels),
            nn.Linear(fusion_out_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, images, labels=None):
        # 1. 2D 특징 맵 추출
        spatial_features = self.backbone.forward_spatial_features(images)
        
        # 2. Ablation 모듈 통과
        x = self.ablation_module(spatial_features)
        
        # 3. GAP 적용
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        features = torch.flatten(x, 1)
        
        # 4. 분류기 통과
        logits = self.classifier(features)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits
