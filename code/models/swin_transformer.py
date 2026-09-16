import torch
import torch.nn as nn 
from models.module import SAFEModule

#########################
# Window Attention    
#########################

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (B*num_windows, window_size*window_size, dim)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, dim_head)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out

#########################
# Swin Transformer Block
#########################

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=4, mlp_ratio=2.0, shift_size=0):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def window_partition(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size,
                   W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        return windows

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size,
                         self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x, H, W):
        B, C, _, _ = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = self.window_partition(x)
        attn_windows = self.attn(windows)
        x = self.window_reverse(attn_windows, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return x

########################
# Swin Transformer
########################

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=2, window_size=4, mlp_ratio=2.0, num_heads=2, input_shape=(3, 256, 256)):
        super(SwinTransformer, self).__init__()
        self.embed_dim = 32
        self.num_blocks = 2
        self.window_size = window_size

        # 패치 임베딩 (Conv4x4, stride=4)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(input_shape[0], self.embed_dim, kernel_size=4, stride=4, padding=0),
            nn.LayerNorm(self.embed_dim)
        )

        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            block = SwinTransformerBlock(self.embed_dim, num_heads, window_size, mlp_ratio, shift_size)
            self.blocks.append(block)

        # 분류기 헤드
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.embed_dim, num_classes)
        self.act = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        x = self.patch_embed[0](x)  # Conv2d
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.patch_embed[1](x)
        x = x.permute(0, 3, 1, 2)

        for block in self.blocks:
            x = block(x, H=H, W=W)

        x = self.gap(x).flatten(1)
        x = self.fc(x)
        x = self.act(x)
        return x

########################
# Swin Transformer(SAFE)
########################
    
class SwinTransformerSAFE(nn.Module):
    def __init__(self, num_classes=2, window_size=4, mlp_ratio=2.0, num_heads=2, input_shape=(3, 256, 256)):
        super(SwinTransformerSAFE, self).__init__()
        self.embed_dim = 32
        self.num_blocks = 2
        self.window_size = window_size

        # Patch Embedding
        self.patch_embed = nn.Conv2d(input_shape[0], self.embed_dim, kernel_size=4, stride=4, padding=0)
        self.ln = nn.LayerNorm(self.embed_dim)

        # SAFE 모듈: out_channels를 embed_dim으로 맞춤
        self.safe_module = SAFEModule(in_channels=self.embed_dim, out_channels=self.embed_dim)
        self.conv1x1 = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)

        # Swin Blocks
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            self.blocks.append(SwinTransformerBlock(self.embed_dim, num_heads, window_size, mlp_ratio, shift_size))

        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.embed_dim, num_classes)
        self.act = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)
        B, C, H, W = x.shape

        # LayerNorm 적용 (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)

        # SAFE + 1x1 Conv
        x = self.safe_module(x)
        x = self.conv1x1(x)

        # Swin blocks
        for blk in self.blocks:
            x = blk(x, H=H, W=W)

        # Head
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        x = self.act(x)
        return x
