import math
from functools import partial
from model.cbm_models_gaze import MLP, End2EndModel
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn

from utils.dino_utils import trunc_normal_
from utils.cross_attn import CrossAttention
import torch.nn.functional as F
import clip
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, AutoConfig, VideoMAEConfig, VideoMAEForVideoClassification2

class Adapter(nn.Module):
    def __init__(
        self,
        *,
        num_classes = 5,dim = 1536,
        n_attributes=17, bottleneck=True, expand_dim=512, connect_CY=False,dropout=0.
    ):
        super().__init__()
        self.attn = CrossAttention(query_dim=1536, key_dim=768, embed_dim=512, num_heads=8)
        self.feat = None
        self.bottleneck = bottleneck
        self.n_attributes = n_attributes
        self.all_fc = nn.ModuleList()

        self.dim = dim

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None        
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(1536, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(1536, 1, expand_dim))
        else:
            self.all_fc.append(FC(1536, num_classes, expand_dim))

        
        config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base")
        config.image_size = 224          # Change spatial resolution
        config.patch_size = 16           # Patch size
        config.num_frames = 16           # Number of video frames
        config.hidden_size = 768         # Hidden layer size
        config.num_attention_heads = 12
        #config.num_labels = 5  # Attention heads
        config.num_labels = num_classes
        config.dropout = 0.4  


        label2id={"rturn": 0, "rchange": 1, "lturn": 2, "lchange": 3, "endaction": 4}
        id2label = {i: label for label, i in label2id.items()}
        model_ckpt = "MCG-NJU/videomae-base"

        self.model1 = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            #hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
            #config=config,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        self.model2 = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            #hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5,
            #config=config,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.apply(self._init_weights)

        
        self.tubelet_size = self.model1.config.tubelet_size
        self.patch_size = self.model1.config.patch_size
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # pushing the text embedding to vidoe space 
        self.text2vid = nn.Linear(512,768).to(device)
        self.final_mlp = nn.Linear(512,7).to(device)
        self.final_gaze = nn.Linear(512,15).to(device)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        # loading the clip model 
        # model_clip, preprocess = clip.load("ViT-B/32", device=device)
        # model_clip.eval()

        #  17 explanation texts
        # explanations = [
        # ]
        # assert len(explanations) == 17, "please bro, provide exactly 17 explanations."

        # tokenized_text = clip.tokenize(explanations).to(device)
        # with torch.no_grad():
        #     text_features = model_clip.encode_text(tokenized_text)


        self.text_feat = torch.load("/scratch/mukil/cemformer/weights/text_feat.pt")
        self.text_feat = self.text_feat / self.text_feat.norm(dim=-1, keepdim=True)
        
        self.text_feat = self.text2vid(self.text_feat.to(torch.float32)).detach() 
        print("Shape of text embeddings:", self.text_feat.shape) 

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

    def forward(self, img1,img2):
        
        B,T,c,H,W = img1['pixel_values'].shape
        T_emb = (T//self.tubelet_size)
        H_emb = W_emb = (H//self.patch_size)

        seq1,ori1 = self.model1(**img1)
        seq2,ori2 = self.model2(**img2)

        x = torch.cat((seq1,seq2),dim=-1)
        ori = torch.cat((ori1,ori2),dim=-1)
        C = self.dim

        ori_3d = ori.reshape(B, T_emb, H_emb, W_emb, C)

        ori_3d = ori_3d.permute((0,-1,1,2,3))
        ori_3d = self.pool(ori_3d).flatten(2).permute(0,2,1)

        # cross attention is performed here 
        text_feature = self.text_feat.unsqueeze(0).repeat(ori.shape[0],1,1)
        attn_value = self.attn(ori_3d,text_feature)

        x = self.final_mlp(attn_value)
        gaze = self.final_gaze(attn_value)
        x = x.mean(1)
        gaze = gaze.mean(1)
        return x,gaze


class FC(nn.Module):

    def __init__(self, input_dim, output_dim, expand_dim, stddev=None):
        """
        Extend standard Torch Linear layer to include the option of expanding into 2 Linear layers
        """
        super(FC, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim > 0:
            self.relu = nn.ReLU()
            self.fc_new = nn.Linear(input_dim, expand_dim)
            self.fc = nn.Linear(expand_dim, output_dim)
        else:
            self.fc = nn.Linear(input_dim, output_dim)
        if stddev:
            self.fc.stddev = stddev
            if expand_dim > 0:
                self.fc_new.stddev = stddev

    def forward(self, x):
        #import pdb;pdb.set_trace()
        if self.expand_dim > 0:
            x = self.fc_new(x)
            x = self.relu(x)
        x = self.fc(x)
        return x

def Multi_Mae_cross(num_classes, multitask_classes, multitask, n_attributes, bottleneck, expand_dim,
                 use_relu, use_sigmoid,connect_CY, dropout):

    model1 = Adapter(num_classes = num_classes,n_attributes=n_attributes,
                  bottleneck=bottleneck, expand_dim=expand_dim,connect_CY=connect_CY,dropout=dropout)

    return model1
