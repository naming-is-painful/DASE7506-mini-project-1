"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory change. Explain it, measure its cost and perform a mechanism ablation. Merely renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""
import torch
from torch import dropout, nn
from torch.nn import functional as F

# Directly copied from models.py
# GELU() changed to ReLU(), which has very little effect
class Block(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        # Each position attends only to itself and earlier input tokens.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # dropout only makes results worse?
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.dropout(self.mlp(self.norm2(x)))

        # x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        # return x + self.mlp(self.norm2(x))

class almost_the_original_GPT(nn.Module):
    # copied and not-yet-modified from GPT in model.py
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([Block(width, config['heads']) for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, with no access to targets.

        Override this for a strictly causal, within-window memory mechanism.
        A prediction at position t can use ids[:, :t+1] and nothing later.
        Reset all temporary state on every call; each evaluation window starts fresh.
        """
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return almost_the_original_GPT(config)


