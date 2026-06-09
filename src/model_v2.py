import sys

sys.path.append('/Users/tom/Development/carbot_ai_model/Depth-Anything-V2')
from model_v1 import ResetNetBlock, ActionHead
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from depth_anything_v2.dpt import DepthAnythingV2


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = self.dim // 2
        self.emb_calc = math.log(10000) / (self.half_dim - 1)

    def forward(self, x):
        """
        x is the time embedding and is translated to a sinusoidal postional embedding.
        """
        device = x.device
        emb = torch.exp(torch.arange(self.half_dim, device=device) * -self.emb_calc)
        ## here we are unsquezee the vectors to get (B, 1) * (1, half_dim)
        emb = x[:, None] * emb[None, :] # [B, half_dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # [B, dim]
        return emb

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


class Encoder(nn.Module):
    def __init__(self, model_context, in_chans = 3, kernel_size=3, n_groups=8):
        super().__init__()
        self.levels = levels = model_context.levels
        self.blocks = nn.ModuleList()

        zipped_levels = zip([in_chans] + levels[:-1], levels)

        for (in_ch, out_ch) in zipped_levels:
            self.blocks.append(nn.Sequential(ResetNetBlock(in_ch, out_ch, kernel_size, n_groups),
                                             nn.MaxPool2d(kernel_size=2, stride=2)))
            
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(levels[-1], model_context.embedding_dim)
                                             

    def forward(self, x):
        for block in self.blocks:
            x = block(x) 
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm_b = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        x = self.norm_a(x + attn_output)
        ffn_output = self.ff(x)
        x = self.norm_b(x + ffn_output)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, in_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionBlock(embed_dim=in_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        return x


class DepthMap(nn.Module):
    def __init__(self, model_context, kernel_size=3):
        super().__init__()
        self.model_context = model_context
        encoder = 'vits' # or 'vits', 'vitb', 'vitg'
        depth_model = DepthAnythingV2(**model_configs[encoder])
        depth_model.load_state_dict(torch.load(f'/Users/tom/Development/carbot_ai_model/models/depth/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        depth_model = depth_model.to(model_context.device).to(model_context.data_type).eval()

        for param in depth_model.parameters():
            param.requires_grad = False

        self.depth_model = depth_model

        self.depth_encoder = Encoder(model_context=model_context, in_chans = 1, kernel_size=kernel_size)



    # Expects input shape (B, O, C, H, W)
    def forward(self, x):
        curr_x = x[:, 0, :, :, :]
        with torch.no_grad():
            depth_map = self.depth_model(curr_x)
            depth_map = F.interpolate(depth_map[:, None], (self.model_context.height, self.model_context.width), mode="bilinear", align_corners=True)
        depth_encoded = self.depth_encoder(depth_map)
        return depth_encoded
    

class ModelV2(nn.Module):
    def __init__(self, model_context, kernel_size=3, dropout=0.2):
        super().__init__()
        C, H, W = model_context.channels, model_context.height, model_context.width
        T = model_context.num_prev_observations + 1  # Current observation + previous observations
        self.embd_dim = model_context.embedding_dim

        self.depth_map = DepthMap(model_context=model_context).to(model_context.device)

        self.encoder = Encoder(model_context=model_context, in_chans = C, kernel_size=kernel_size)
        

        self.context_encoder = TransformerEncoder(in_dim=model_context.embedding_dim, num_layers=model_context.attention_blocks, num_heads=4, dropout=dropout)
        ## since the time intervals are constant we can precompute the time embeddings. 
        sin_emb = SinusoidalPosEmb(self.embd_dim) 
        tx = torch.linspace(0, T - 1, T).to(model_context.device).to(model_context.data_type)
        with torch.no_grad():
            self.time_emb = sin_emb(tx)


        self.action_head = ActionHead(embed_dim=self.embd_dim, waypoint_num=model_context.waypoint_num, dropout=dropout)
    ## expects input shape (B, O, C, H, W)
    def forward(self, x):
        B, T, C, H, W = x.shape

        depth_emb = self.depth_map(x).view(B, 1, -1)  # Shape (B, 1, embd_dim)

        x = x.view(B * T, C, H, W)
        x = self.encoder(x)
        x = x.view(B, T, -1)  # Reshape back to (B, O, context_emb_dim)

        x = x + self.time_emb[None, :, :]
        x = torch.cat([depth_emb, x], dim=1)  # Concatenate depth embedding

        # Expects input shape (B, T, context_emb_dim)
        x = self.context_encoder(x)

        x = torch.mean(x, dim=1)  # Aggregate over time dimension
        out = self.action_head(x)
        return out
