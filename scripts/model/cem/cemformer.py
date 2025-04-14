# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn

from utils.dino_utils import trunc_normal_
import torch.nn.functional as F

def exists(val):
    return val is not None

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(4,patch_size,patch_size), stride=(4,patch_size,patch_size))
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        B,C,T,H,W = x.shape
   
        
        #using conv3d
        x = self.proj(x)    
        return x.flatten(2).transpose(1, 2)

        #using conv2d
        # x = x.transpose(1,2).reshape(B*T,C,H,W)
        # return self.proj(x).flatten(2).transpose(1, 2)

        #x = [self.proj(i.permute(1,0,2,3)).flatten(2).transpose(1, 2) for i in x]
        #return torch.stack(x, dim=0)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask = None, memories = None):
        x = self.norm(x)

        x_kv = x # input for key / values projection

        if exists(memories):
            # add memories to key / values if it is passed in
            memories = repeat(memories, 'n d -> b n d', b = x.shape[0]) if memories.ndim == 2 else memories
            x_kv = torch.cat((x_kv, memories), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(x_kv).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, attn_mask = None, memories = None):
       
        for ind, (attn, ff) in enumerate(self.layers):
            layer_memories = memories[ind] if exists(memories) else None

            x = attn(x, attn_mask = attn_mask, memories = layer_memories) + x
            x = ff(x) + x
        
        return x

