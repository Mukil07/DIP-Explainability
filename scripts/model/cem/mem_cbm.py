from model.final_cbm import cbm_module,add_fc
from model.cem.cemformer import Transformer, PatchEmbed

from utils.dino_utils import trunc_normal_
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from functools import partial
from einops import rearrange, repeat

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
        drop_rate,
        attn_drop_rate  
    ):
        super().__init__()
        

        self.depth =depth
        dim = embed_dim
        self.patch_embed1 = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed2 = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed1.num_patches*4
        self.feat = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.memory_cls_token = nn.Parameter(torch.randn(dim))
       # import pdb;pdb.set_trace()
        self.memories_per_layer = nn.Parameter(torch.randn(self.depth, num_memories_per_layer, embed_dim))
       
        #self.pos_embed1 = nn.Parameter(torch.zeros(1, (num_patches), embed_dim))
        #self.pos_embed2 = nn.Parameter(torch.zeros(1, (num_patches), embed_dim))

        # self.pos_embed1 = nn.Parameter(torch.randn( num_patches+num_memories_per_layer, embed_dim))
        # self.pos_embed2 = nn.Parameter(torch.randn( num_patches+num_memories_per_layer, embed_dim))
        # for conv2d 
        #self.pos_embed = nn.Parameter(torch.randn( 1, 2*num_patches + 1, embed_dim))

        #for conv3d
        self.pos_embed = nn.Parameter(torch.randn( 1, 2*num_patches + 1, embed_dim))

        # specialized attention mask to preserve the output of the original ViT
        # it allows the memory CLS token to attend to all other tokens (and the learnable memory layer tokens), but not vice versa        

        attn_mask = torch.ones((num_patches*2, num_patches*2), dtype = torch.bool)
        attn_mask = F.pad(attn_mask, (1, num_memories_per_layer), value = False)  # main tokens cannot attend to learnable memories per layer
        attn_mask = F.pad(attn_mask, (0, 0, 1, 0), value = True)                  # memory CLS token can attend to everything
        self.attn_mask = attn_mask
        self.vit = Transformer(dim=embed_dim, depth=self.depth, heads=12, mlp_dim=embed_dim, dropout= attn_drop_rate)
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

        # trunc_normal_(self.pos_embed1, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
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
        #import pdb;pdb.set_trace()
        embed_img1= self.patch_embed1(img1) # shape - (2,16,196,768)
        embed_img2= self.patch_embed2(img2)

        # add the [CLS] token to the embed patch tokens
        x= torch.cat((embed_img1,embed_img2),dim=1)

        return self.pos_drop(x)
    
    def forward(self, img1,img2):
        device = img1.device
        T = img1.shape[2]
        batch_size = img1.shape[0]
        
        concatenated_img = self.prepare_tokens(img1,img2) # (2,16,392,768)
       
        # add task specific memory tokens
       
        memory_cls_tokens = repeat(self.memory_cls_token, 'd -> batch 1 d', batch=batch_size)
        pos_embedding = repeat(self.pos_embed, '1 num_tokens d -> batch num_tokens d', batch=batch_size)
        tokens = torch.cat((memory_cls_tokens, concatenated_img), dim = 1)        
        tokens = tokens + pos_embedding
        # pass memories along with image tokens through transformer for attending
        self.attn_mask = self.attn_mask.to(device)
        out = self.vit(tokens, memories = self.memories_per_layer, attn_mask = self.attn_mask)
        
        # extract memory CLS tokens
        
        memory_cls_tokens = out[:, 0]

        # pass through task specific adap   ter head
        #import pdb;pdb.set_trace()

        #for conv2d 
        # self.feat = memory_cls_tokens.mean(dim=0).unsqueeze(0)
        # x = memory_cls_tokens.reshape(batch_size, T, -1)
        # x = x.mean(dim=1)

        #for conv3d
        x = memory_cls_tokens
        self.feat = x

        #memory_cls_tokens = memory_cls_tokens.squeeze(2).permute(0, 2, 1)  # Shape: [1, 768, 16]
        #memory_cls_tokens = F.avg_pool1d(memory_cls_tokens, kernel_size=16) 

        return x
        #return memory_cls_tokens.mean(dim=1)
        #return [self.mlp_head(memory_cls_tokens.mean(dim=1))]
        #return self.mlp_head(memory_cls_tokens),self.head_gaze(memory_cls_tokens),self.head_ego(memory_cls_tokens)
        

def vit_mem_dipx(num_classes, multitask_classes, multitask, n_attributes, bottleneck, expand_dim,
                 use_relu, use_sigmoid,connect_CY, dropout, num_memories_per_layer):

    #import pdb;pdb.set_trace()
    patch_size=16
    embed_dim=768

    model = Adapter(patch_size=patch_size, num_memories_per_layer = num_memories_per_layer, num_classes = num_classes, drop_rate=dropout,attn_drop_rate=dropout)
    model = add_fc(model,embed_dim, num_classes, multitask_classes, multitask, n_attributes, bottleneck, expand_dim,
                 use_relu, use_sigmoid,connect_CY, dropout)
    return cbm_module(model,embed_dim, num_classes, multitask_classes, multitask, n_attributes, bottleneck, expand_dim,
                 use_relu, use_sigmoid,connect_CY, dropout)
