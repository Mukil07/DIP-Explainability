from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.transforms.functional import to_pil_image, to_grayscale
from torch.utils.data import Dataset, DataLoader, random_split

import torch
import torch.nn as nn

class CreatePatches(nn.Module):
    def __init__(
        self, channels=3, embed_dim=768, patch_size=16
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # Flatten along dim = 2 
        patches = self.proj(x).flatten(2).transpose(1, 2)
        return patches
    
class Attention(nn.Module):
   
    def __init__(self, embed_dim, num_heads=12, attn_p=0., proj_p=0.):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        
        self.linear = torch.nn.Linear(embed_dim, 3 * embed_dim)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
       
        n_samples, n_tokens, dim = x.shape


        
        x = self.linear(x)  # shape: (10, 397, 3 * 768)
        x = self.attn_drop(x)

       
        qkv=  x.reshape(n_samples, n_tokens, 3, self.num_heads, self.head_dim) # qkv= expand (10,397,3,12,64)
        qkv = qkv.permute(2, 0, 3, 1, 4)  
        q, k, v = qkv[0], qkv[1], qkv[2] # q without memory token, k and v are as usual

        k_t = k.transpose(-2, -1)  
        
        dp = (q @ k_t) * self.scale # (10, 12, 397,397)

        # have to apply attenion mask here - shape (b,num_heads,seq,seq)
        #if mem_mask is not None:
        #        mem_mask = mem_mask.unsqueeze(1)  # (batch_size, 1, seq_length, seq_length)
        #        mem_mask = mem_mask.repeat(1, self.transformer.nhead, 1, 1) 

        attention = dp.softmax(dim=-1)  #  (10, 12, 397,397)
        attention = self.attn_drop(attention)

        weighted_avg = attention @ v  # (10, 12, 397,64)
        weighted_avg = weighted_avg.transpose(1, 2)  # (10, 397, 12, 64)
        weighted_avg = weighted_avg.flatten(2)  # (10, 397, 768)

        x = self.proj(weighted_avg)  # (10, 397, 768)
        x = self.proj_drop(x)  # (10, 397, 768)

        return x
    
def build_attention_mask(patches: int, memory_tokens_list: list, extension: bool = False):

    class_tokens = len(memory_tokens_list) + 1
    input_tokens = patches + class_tokens
    total_tokens = input_tokens + sum(memory_tokens_list)

    with torch.no_grad():
        mask = torch.zeros(1, input_tokens, total_tokens)
        # Disable all interactions for all newly added class tokens and memory
        mask[:, :, (patches + 1):] = -math.inf
        # Enable interactions for each class token and corresponding memory
        previous_memory = 0
        for i, memory_tokens in enumerate(memory_tokens_list):
            # 16 patches + 1 default class token + index of this
            class_token = (patches + 1) + i
            memory_start_index = input_tokens + previous_memory
            memory_end_index = memory_start_index + memory_tokens
            if extension:
                # Class token can interact with itself
                mask[:, class_token:, class_token] = 0.0
                # Class token can interact with its memory tokens
                mask[:, class_token:, memory_start_index:memory_end_index] = 0.0

            previous_memory += memory_tokens

    return mask

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-06)

        self.attention= Attention(embed_dim= embed_dim, num_heads= num_heads, attn_p=0., proj_p=0.)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x_norm = self.pre_norm(x)


        x = x + self.attention(x_norm)
        x = x + self.MLP(self.norm(x))
        #x = torch.cat((x,self.eps_mem),dim=1)
        return x


class ViT(nn.Module):
    def __init__(
        self, 
        img_size=224,
        in_channels=3,
        patch_size=16,
        embed_dim=768,
        hidden_dim=3072,
        num_heads=12,
        num_layers=12,
        dropout=0.0,
        num_classes=5
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size//patch_size) ** 2
        self.patch_embed = CreatePatches(
            channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        # Postional encoding.
        self.pos_embedding = nn.Parameter(torch.randn(1, (num_patches*2)+1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn_layers = nn.ModuleList([])
        self.eps_mem= nn.Parameter(torch.randn(1, 4, embed_dim))
        self.embed_dim=embed_dim
        for _ in range(num_layers):
            self.attn_layers.append(
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout)
            )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-06)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self,img1,img2):
        
        #img1= torch.squeeze(img1,dim=0)
        #img2= torch.squeeze(img2,dim=0)
        b, n, h,w = img1.shape
        embed_img1= self.patch_embed(img1) # shape - (10,196,768)
        embed_img2= self.patch_embed(img2)
        
        concatenated_img= torch.cat((embed_img1,embed_img2),dim=1) # shape -(10,392,768)
        cls_tokens = self.cls_token.expand(b, -1, -1) # shape -(10,1,768)
        concatenated_img= torch.cat((concatenated_img,cls_tokens),dim=1) # shape -(10,393,768)
        concatenated_img+= self.pos_embedding  # shape -(10,393,768)
        
        # if eps_mem.shape[0]==1:
        #     eps_mem= self.eps_mem.expand(b, -1, -1) # - shape (10,4,768)
        #concatenated_img= torch.cat((concatenated_img,eps_mem),dim=1) # shape - (10,397,768)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #linear = torch.nn.Linear(self.embed_dim, 3 * self.embed_dim)
        #linear.to(device)
        #x = linear(concatenated_img)  # shape: (10, 397, 3 * 768)
        x = self.dropout(concatenated_img)
        #q, k, v = torch.split(t,768, dim=2) # q - (10,397,768)
            
        
        for layer in self.attn_layers:
            x = layer(x)
        
        #self.eps_mem= x[:,:4]
        
        x = self.ln(x)
        
        #z_final=x[:,:4]
        #x = z_final 
        #import pdb;pdb.set_trace()
        #return self.head(z_final)
        return self.head(x[:,-1,:])
