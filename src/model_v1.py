import torch.nn as nn
import torch
import torch.nn.functional as F
from src.model_context import ModelContext

class Convnet2DBlock(nn.Module):
    """"
    Block use Conv2D -> GroupNorm -> MISH
    """
    def __init__(self, in_chans: int, out_chans: int, kernel_size: int, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=kernel_size, padding=kernel_size//2),
            nn.GroupNorm(num_groups=n_groups, num_channels=out_chans),
            nn.Mish())

    def forward(self, x):
        return self.block(x)
    
class ResetNetBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, kernel_size: int, n_groups=8):
        super().__init__() 
        self.conv = Convnet2DBlock(in_chans=in_chans, out_chans=out_chans, kernel_size=kernel_size)
        self.resnet = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=kernel_size, padding=kernel_size//2) if in_chans != out_chans else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        res = self.resnet(x)
        return out + res
    
    
class ActionHead(nn.Module):
    def __init__(self, embed_dim: int, waypoint_num: int, dropout=0.2):
        super().__init__()

        self.waypoint_num = waypoint_num
        self.dropout = dropout
        self.waypoint_encoder = nn.Sequential(
            nn.Linear(waypoint_num * 2 - 2, embed_dim),
            nn.Mish(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, embed_dim))

        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim * 4, embed_dim), 
            nn.Mish(), 
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Mish(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim // 2, 2))

    def forward(self, x):
        B, D = x.shape
        waypoints = []

        # Auto-regressive waypoint prediction
        for i in range(self.waypoint_num):
            if i == 0:
                waypoint = self.forward_(x)
                waypoints.append(waypoint)
            else:
                prev_waypoints = torch.cat(waypoints, dim=1)  # Concatenate previous waypoints
                target_cols = self.waypoint_num * 2 - 2
                padded = prev_waypoints.new_zeros(B, target_cols)
                padded[:, :prev_waypoints.shape[1]] = prev_waypoints
                prev_waypoints = padded
                waypoint = self.forward_(x, prev_waypoints)
                waypoints.append(waypoint)

        out = torch.cat(waypoints, dim=1)
        out = out.view(out.size(0), -1, 2) # Reshape to (batch_size, waypoint_num, 2)
        return out

    def forward_(self, x, prev_waypoints = None):
        out = None
        if prev_waypoints is not None:
            prev_action_embed = self.waypoint_encoder(prev_waypoints)
            out = x + prev_action_embed
        out = self.action_decoder(out if out is not None else x)
        return out
    
class ModelV1(nn.Module):
    def __init__(self, model_context: ModelContext, kernel_size=3):
        super().__init__()
        self.encoder = Encoder(in_chans=model_context.channels, levels=model_context.levels, kernel_size=kernel_size, n_groups=8)
        C, H, W = model_context.channels, model_context.height, model_context.width
        hidden_dim = model_context.hidden_dim
        self.waypoint_num = model_context.waypoint_num
        with torch.no_grad():
            dummy = torch.zeros((1, C, H, W))
            encoded = self.encoder(dummy)
            flat_dim = encoded.view(1, -1).shape[1]

        print(f'Flattened dimension: {flat_dim}')

        self.flatten = nn.Flatten()
        self.action_head = ActionHead(in_dim=flat_dim, hidden_dim=hidden_dim, waypoint_num=self.waypoint_num)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        out = self.action_head(x)
        return out