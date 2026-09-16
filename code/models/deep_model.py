import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models
from models.module import SAFEModule 

#######################
# CBAM Block
#######################

class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction_ratio=8):
        super(CBAMBlock, self).__init__()
        self.reduction_ratio = reduction_ratio

        # ---- Channel Attention ----
        self.shared_mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False)
        )

        # ---- Spatial Attention ----
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # === Channel Attention ===
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)  # [B, C]
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)  # [B, C]

        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        channel_attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_attention

        # === Spatial Attention ===
        avg_pool_spatial = torch.mean(x, dim=1, keepdim=True)           # [B, 1, H, W]
        max_pool_spatial, _ = torch.max(x, dim=1, keepdim=True)         # [B, 1, H, W]
        concat = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1) # [B, 2, H, W]

        spatial_attention = self.sigmoid(self.conv_spatial(concat))
        refined_feature = x * spatial_attention

        return refined_feature
    
class DeepModel(nn.Module):
    def __init__(self, model_name='ResNet50', num_classes=2, input_channels=3):
        super(DeepModel, self).__init__()

        # === Base model 선택 ===
        model_dict = {
            'VGG16': models.vgg16,
            'ResNet50': models.resnet50,      
            'DenseNet121': models.densenet121,
            "EfficientNetV2":models.efficientnet_v2_s,
            "ConvNeXtTiny":models.convnext_tiny,
            "VisionTransformer":models.vit_b_16,
        }

        if model_name not in model_dict:
            raise ValueError(f"Unsupported model name: {model_name}")

        base_model = model_dict[model_name](weights=None)  # pretrained 불러오지 않음

        # === Feature extractor만 남기기 ===
        if model_name == 'VGG16':
            self.feature_extractor = nn.Sequential(*list(base_model.features))
            feature_dim = 512
        elif model_name == 'ResNet50':
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 2048
        elif model_name == 'DenseNet121':
            self.feature_extractor = nn.Sequential(*list(base_model.features.children()))
            feature_dim = 1024
        elif model_name == 'EfficientNetV2':
            self.feature_extractor = nn.Sequential(*list(base_model.features.children()))
            feature_dim = 1280
        elif model_name == 'ConvNeXtTiny':
            self.feature_extractor = nn.Sequential(*list(base_model.features.children()))
            feature_dim = 768

        # === CBAM Block (외부 정의 필요) ===
        self.cbam = CBAMBlock(feature_dim)

        # === Adaptive Global Average Pooling ===
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # === Fully Connected Layers ===
        self.fc1 = nn.Linear(feature_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.act2 = nn.ReLU()

        self.drop = nn.Dropout(0.5)

        # === Combined layer ===
        self.fc_combined = nn.Linear(512, 256)
        self.bn_combined = nn.BatchNorm1d(256)
        self.act_combined = nn.ReLU()

        # === Output layer ===
        if num_classes == 1:
            self.fc_out = nn.Linear(256, 1)
            self.activation = nn.Sigmoid()
        else:
            self.fc_out = nn.Linear(256, num_classes)
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        x = self.cbam(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)

        # Dense layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop(x)

        # Combined + Output
        x = self.fc_combined(x)
        x = self.bn_combined(x)
        x = self.act_combined(x)
        x = self.drop(x)

        x = self.fc_out(x)
        x = self.activation(x)
        return x
    
class DeepSAFEModel(nn.Module):
    def __init__(self, model_name='ResNet50', num_classes=2, input_channels=3):
        super(DeepSAFEModel, self).__init__()

        model_dict = {
            'VGG16': models.vgg16,
            'ResNet50': models.resnet50,
            'DenseNet121': models.densenet121,
            "EfficientNetV2":models.efficientnet_v2_s,
            "ConvNeXtTiny":models.convnext_tiny,
            "VisionTransformer":models.vit_b_16,
        }

        if model_name not in model_dict:
            raise ValueError(f"Unsupported model: {model_name}")

        base_model = model_dict[model_name](weights=None)

        # Feature extractor
        if model_name == 'VGG16':
            self.feature_extractor = nn.Sequential(*list(base_model.features))
        elif model_name == 'ResNet50':
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        elif model_name == 'DenseNet121':
            self.feature_extractor = nn.Sequential(*list(base_model.features.children()))
        elif model_name == 'EfficientNetV2':
            self.feature_extractor = nn.Sequential(*list(base_model.features.children()))
        elif model_name == 'ConvNeXtTiny':
            self.feature_extractor = nn.Sequential(*list(base_model.features.children()))
        
        # 채널 수 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 224, 224)
            x = self.feature_extractor(dummy_input)
            cbam_channels = x.shape[1]

        # CBAM & SAFE
        self.cbam = CBAMBlock(cbam_channels)
        self.safe_module = SAFEModule(cbam_channels)

        # GAP 후 FC 입력 차원
        with torch.no_grad():
            x = self.cbam(x)
            x = self.safe_module(x)
            x = nn.AdaptiveAvgPool2d((1,1))(x)
            feature_dim_fc = x.flatten(1).shape[1]

        # FC layers
        self.fc1 = nn.Linear(feature_dim_fc, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)

        self.fc_combined = nn.Linear(512, 256)
        self.bn_combined = nn.BatchNorm1d(256)
        self.act_combined = nn.ReLU()
        self.drop_combined = nn.Dropout(0.5)

        # Output
        if num_classes == 1:
            self.fc_out = nn.Linear(256, 1)
            self.activation = nn.Sigmoid()
        else:
            self.fc_out = nn.Linear(256, num_classes)
            self.activation = nn.Identity()  # CrossEntropyLoss용

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.cbam(x)
        x = self.safe_module(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)

        x = self.fc_combined(x)
        x = self.bn_combined(x)
        x = self.act_combined(x)
        x = self.drop_combined(x)

        x = self.fc_out(x)
        x = self.activation(x)
        return x
