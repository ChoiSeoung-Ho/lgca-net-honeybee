from transformers import BertModel, BertConfig
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

from models.module import SAFEModule

class BertForImageClassification_v2(nn.Module):
    def __init__(self, num_labels, img_size=224, patch_size=16, hidden_dim=512):
        super(BertForImageClassification_v2, self).__init__()

        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.config.num_labels = num_labels

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size,
                                         in_channels=3, embed_dim=self.config.hidden_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, self.config.hidden_size))

        self.transformer = BertModel(self.config)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            x = self.transformer(dummy_input)
            SAFEModule_channels = x.shape[1]

        self.safe_module = SAFEModule(SAFEModule_channels)


        self.mlp = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, images, labels=None):
        batch_size = images.shape[0]

        x = self.patch_embed(images)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed

        transformer_output = self.transformer(inputs_embeds=x)
        cls_output = transformer_output.last_hidden_state[:, 0]

        logits = self.mlp(cls_output)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits

class BertForImageClassification(nn.Module):
    def __init__(self, num_labels, img_size=224, patch_size=16, hidden_dim=512):
        super(BertForImageClassification, self).__init__()
        
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.config.num_labels = num_labels
        
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, 
                                         in_channels=3, embed_dim=self.config.hidden_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, self.config.hidden_size))
        
        self.transformer = BertModel(self.config)
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, images, labels=None):
        batch_size = images.shape[0]
        
        x = self.patch_embed(images)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        
        transformer_output = self.transformer(inputs_embeds=x)
        cls_output = transformer_output.last_hidden_state[:, 0]
        
        logits = self.mlp(cls_output)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits


