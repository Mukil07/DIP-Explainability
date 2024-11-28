import torch 
import torch.nn as nn 
import numpy as np

class Custom_criterion(nn.Module):
    
    def __init__(self):
        
        self.A={'0':[[1,0,0],[1,0,1],[0,0,0],[0,1,0],[1,1,0]],
                '1':[[0,1,0],[1,1,0],[0,1,1],[1,1,1]],
                '2':[[0,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,0]],
                '3':[[1,0,0],[1,0,1],[1,1,0],[1,1,1]],
                '4':None}
    
    def forward(self,c,output_logits):
        c = tuple(map(int,c.split(',')))

        ctx=np.zeros(3)

        if c[0] == 1:
            ctx[0]=1
        if c[0] == c[1]:
            ctx[1]=1
        if c[2] == 1:
            ctx[2]=1

        cc_loss=0 
        for i in self.A:
            if not int(i) == 4: 
                if ctx.tolist() in self.A[i]:
                    cc_loss+= -torch.log(1-output_logits[int[i]])
        
        return cc_loss
    
    
    
