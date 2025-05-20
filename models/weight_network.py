# Echo's implementation
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_dims=[512, 512], ctx_dim=2048, conf_dim=5):
        super().__init__()
        
        # confidence features
        self.conf_proj = nn.Sequential(
            nn.LayerNorm(conf_dim),
            nn.Linear(conf_dim, 128),
            nn.ReLU()
        )
        
        # hidden states
        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, 512),
            nn.ReLU()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(512 + 128, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], vocab_size), 
            nn.Sigmoid() 
        )
        
        self._init_weights()

    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, ctx_hidden, conf_feat):
        ctx_emb = self.ctx_proj(ctx_hidden)      # [B, 512]
        conf_emb = self.conf_proj(conf_feat)     # [B, 128]

        x = torch.cat([ctx_emb, conf_emb], dim=-1)  # [B, 640]

        return self.mlp(x)
