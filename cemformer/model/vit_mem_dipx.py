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
import math
from functools import partial

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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_mem(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask = None, memories = None):
        batch, B, N, C = x.shape
        #import pdb;pdb.set_trace()
        x_kv = x 

        if exists(memories):
           #import pdb;pdb.set_trace()
            memories = repeat(memories, 'n d -> batch b n d', batch=batch, b = x.shape[1]) if memories.ndim == 2 else memories
            x_kv = torch.cat((x_kv, memories), dim = 2)
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            

        qkv = (self.to_q(x), *self.to_kv(x_kv).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'batch b n (h d) -> batch b h n d', h = self.num_heads), qkv)


        #q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if exists(attn_mask):
            attn = attn.masked_fill(~attn_mask, -torch.finfo(attn.dtype).max)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch, B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #import pdb;pdb.set_trace()
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention_mem(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask, memories, return_attention=False):
        #import pdb;pdb.set_trace()
        y, attn = self.attn(self.norm1(x),memories = memories, attn_mask = attn_mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        batch, B, C, H, W = x.shape
        #import pdb;pdb.set_trace()
        x = [self.proj(i).flatten(2).transpose(1, 2) for i in x]
        return torch.stack(x, dim=0)


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=5, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, num_mem = 3,**kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.depth = depth

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        #self.mem_tokens = nn.Parameter(torch.zeros(1, self.num_mem, embed_dim))

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x, memories = None, attn_mask = None):

        #import pdb;pdb.set_trace()
        for blk in self.blocks:
            x = blk(x,attn_mask,memories)
        x = self.norm(x)
        #import pdb;pdb.set_trace()
        return x

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class Adapter(nn.Module):
    def __init__(
        self,
        *,
        patch_size=16,
        embed_dim=768,
        depth=12,
        in_chans=3,
        img_size=[224],
        num_memories_per_layer = 10,
        num_classes = 5,   
    ):
        super().__init__()
        


        dim = embed_dim
        self.patch_embed1 = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed2 = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed1.num_patches


        self.pos_drop = nn.Dropout(p=0.)
        self.memory_cls_token = nn.Parameter(torch.randn(dim))
       # import pdb;pdb.set_trace()
        self.memories_per_layer = nn.Parameter(torch.randn(num_memories_per_layer, dim))
       
        #self.pos_embed1 = nn.Parameter(torch.zeros(1, (num_patches), embed_dim))
        #self.pos_embed2 = nn.Parameter(torch.zeros(1, (num_patches), embed_dim))

        self.pos_embed1 = nn.Parameter(torch.zeros(embed_dim))
        self.pos_embed2 = nn.Parameter(torch.zeros(embed_dim))
        # specialized attention mask to preserve the output of the original ViT
        # it allows the memory CLS token to attend to all other tokens (and the learnable memory layer tokens), but not vice versa        

        attn_mask = torch.ones((num_patches*2, num_patches*2), dtype = torch.bool)
        attn_mask = F.pad(attn_mask, (1, num_memories_per_layer), value = False)  # main tokens cannot attend to learnable memories per layer
        attn_mask = F.pad(attn_mask, (0, 0, 1, 0), value = True)                  # memory CLS token can attend to everything
        self.attn_mask = attn_mask
        self.vit = VisionTransformer( patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_mem = 3)
        self.head_gaze = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 15)
        )

        self.head_ego = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 17)
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

        # Apply to specific learnable tokens and embeddings

        trunc_normal_(self.pos_embed1, std=0.02)
        trunc_normal_(self.pos_embed2, std=0.02)
        trunc_normal_(self.memory_cls_token, std=0.02)

    def interpolate_pos_encoding(self, x, w, h,pe):
        npatch = x.shape[1] - 1
        N = pe.shape[1] - 1
        if npatch == N and w == h:
            return pe
        class_pos_embed = pe[:, 0]
        patch_pos_embed = pe[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed1.patch_size
        h0 = h // self.patch_embed2.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self,img1,img2):
        batch, B, nc, w, h = img1.shape
 
        embed_img1= self.patch_embed1(img1) # shape - (2,16,196,768)
        embed_img2= self.patch_embed2(img2)
        #x = self.patch_embed(x)  # patch linear embedding
        self.pos_embed1_ = repeat(self.pos_embed1, 'd -> batch b num_patch d', batch = batch, num_patch = 196, b = B)
        self.pos_embed2_ = repeat(self.pos_embed2, 'd -> batch b num_patch d', batch = batch, num_patch = 196, b = B)
        # self.pos_embed1 = self.pos_embed1.unsqueeze(0).unsqueeze(0).expand(batch, B, 196, -1)
        # self.pos_embed1 = self.pos_embed1.unsqueeze(0).unsqueeze(0).expand(batch, B, 196, -1)
        # embed_img1 = embed_img1 + self.interpolate_pos_encoding(embed_img1, w, h,self.pos_embed1)
        # embed_img2 = embed_img2 + self.interpolate_pos_encoding(embed_img2, w, h,self.pos_embed2)
        embed_img1 = embed_img1 + self.pos_embed1_
        embed_img2 = embed_img2 + self.pos_embed2_

        # add the [CLS] token to the embed patch tokens
        x= torch.cat((embed_img1,embed_img2),dim=2)

        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens,concatenated_img), dim=1)

        # # add positional encoding to each token
        # x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)
    
    def forward(self, img1,img2):
        device = img1.device
        b = img1.shape[1]
        batch_size = img1.shape[0]

        # tokens1 = self.patch_embed(img1)
        # tokens2 = self.patch_embed(img2)

        # concatenated_img= torch.cat((tokens1,tokens2),dim=1)
  
        concatenated_img = self.prepare_tokens(img1,img2) # (2,16,392,768)

        # add task specific memory tokens

        memory_cls_tokens = repeat(self.memory_cls_token, 'd -> batch b 1 d', batch = batch_size,b = b)
        tokens = torch.cat((memory_cls_tokens, concatenated_img), dim = 2)        

        # pass memories along with image tokens through transformer for attending
        self.attn_mask = self.attn_mask.to(device)
        out = self.vit(tokens, memories = self.memories_per_layer, attn_mask = self.attn_mask)

        # extract memory CLS tokens
        
        memory_cls_tokens = out[:, :,0]

        # pass through task specific adapter head
        
        return self.mlp_head(memory_cls_tokens),self.head_gaze(memory_cls_tokens),self.head_ego(memory_cls_tokens)
        
def vit_mem_dipx(patch_size=16, num_memories_per_layer= 10, num_classes=5, **kwargs):

    #import pdb;pdb.set_trace()
    model = Adapter(patch_size=patch_size, num_memories_per_layer = num_memories_per_layer, num_classes = num_classes)
    return model
