import math
from functools import partial
from model.cbm_models_gaze import MLP, End2EndModel
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn

from utils.dino_utils import trunc_normal_
import torch.nn.functional as F

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, AutoConfig, VideoMAEConfig, VideoMAEForVideoClassification2

class Adapter(nn.Module):
    def __init__(
        self,
        *,
        num_classes = 5,dim = 1536,
        n_attributes=17, bottleneck=True, expand_dim=512, connect_CY=False,dropout=0.
    ):
        super().__init__()

        self.feat = None
        self.bottleneck = bottleneck
        self.n_attributes = n_attributes
        self.all_fc = nn.ModuleList()
       
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

        image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        self.model1 = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
            #num_hidden_layers=6,
            #config=config,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        self.model2 = VideoMAEForVideoClassification2.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5,
            #num_hidden_layers=6,
            #config=config,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        self.classifier = nn.Sequential(
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

    def forward(self, img1,img2):
        #import pdb;pdb.set_trace()
        
        seq1 = self.model1(**img1)
        seq2 = self.model2(**img2)

        #import pdb;pdb.set_trace()
        x = torch.cat((seq1,seq2),dim=-1)
        self.feat = x
        out = []
        #x= x.permute((0,2,3,4,1))
        if self.n_attributes == 0:
            out.append(x)
            return out
        for fc in self.all_fc:
            out.append(fc(x))
        
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            out[0] += self.cy_fc(attr_preds)

        return out


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

def Multi_Mae(num_classes, multitask_classes, multitask, n_attributes, bottleneck, expand_dim,
                 use_relu, use_sigmoid,connect_CY, dropout):

    model1 = Adapter(num_classes = num_classes,n_attributes=n_attributes,
                  bottleneck=bottleneck, expand_dim=expand_dim,connect_CY=connect_CY,dropout=dropout)

    model2 = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)

    if n_attributes>0:
    
        if multitask:
            model3 = MLP(input_dim=n_attributes, num_classes=multitask_classes, expand_dim=expand_dim)
            model4 = None
            return End2EndModel(model1, model2, model3, model4, multitask,  n_attributes, use_relu, use_sigmoid)
        else:
            model3 = None
            model4= None
            return End2EndModel(model1, model2, model3, model4, multitask, n_attributes, use_relu, use_sigmoid)
    else:
        if multitask:
            model2 = MLP(input_dim=1536, num_classes=num_classes, expand_dim=expand_dim)
            model3 = MLP(input_dim=1536, num_classes=15, expand_dim=expand_dim) # gaze head 
            model4 = MLP(input_dim=1536, num_classes=17, expand_dim=expand_dim) # ego head 
            return End2EndModel(model1, model2, model3, model4, multitask, n_attributes, use_relu, use_sigmoid)

        else:
            model2 = MLP(input_dim=1536, num_classes=num_classes, expand_dim=expand_dim)
            model3 = None
            model4 = None
            return End2EndModel(model1, model2, model3, model4, multitask, n_attributes, use_relu, use_sigmoid)
