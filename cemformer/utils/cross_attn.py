import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    def __init__(self, query_dim=1536, key_dim=768, embed_dim=512, num_heads=8):
        """
        Args:
            query_dim (int): Dimension of the image embedding (query).
            key_dim (int): Dimension of the text embedding (key/value).
            embed_dim (int): Dimension of the common projected space.
            num_heads (int): Number of attention heads.
        """
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(key_dim, embed_dim)
        
        # Output projection.
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Scale factor for dot-product attention.
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, image_embedding, text_embedding):
        """
        Args:
            image_embedding (Tensor): Query tensor of shape (batch, n, query_dim).
            text_embedding (Tensor): Key/Value tensor of shape (batch, t, key_dim).
        Returns:
            out (Tensor): The result of cross attention with shape (batch, n, embed_dim).
        """
        B, n, _ = image_embedding.shape  
        B, t, _ = text_embedding.shape     

        Q = self.q_proj(image_embedding)
        K = self.k_proj(text_embedding)    
        V = self.v_proj(text_embedding) 
        

        #(B, num_heads, seq_length, head_dim)
        Q = Q.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)  #(B, num_heads, n, head_dim)
        K = K.view(B, t, self.num_heads, self.head_dim).transpose(1, 2)  #(B, num_heads, t, head_dim)
        V = V.view(B, t, self.num_heads, self.head_dim).transpose(1, 2)  #(B, num_heads, t, head_dim)
        
        #(B, num_heads, n, t)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = torch.matmul(attn_weights, V)
        
        #(B, n, embed_dim)
        context = context.transpose(1, 2).contiguous().view(B, n, self.embed_dim)

        out = self.out_proj(context)
        return out

