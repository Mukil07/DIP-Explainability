import torch
import torch.nn as nn
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)

        focal_factor = (1 - pt) ** self.gamma
        loss = self.alpha * focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: 
            return loss
        
class cc_loss():

    def __init__(self):    

        self.A ={'0':[[1,0,0],[1,0,1],[0,0,0],[0,1,0],[1,1,0]],
                '1':[[0,1,0],[1,1,0],[0,1,1],[1,1,1]],
                '2':[[0,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,0]],
                '3':[[1,0,0],[1,0,1],[1,1,0],[1,1,1]],
                '4':None}

    
    def calc_loss(self,context,output_logits):

        output_logits = torch.nn.functional.softmax(output_logits, dim=1)

        cc_loss=0 
        count=0
        for c in context:
            ctx=np.zeros(3)
            if c[0].item() == 1:
                ctx[0]=1
            if c[0].item() == c[1].item():
                ctx[1]=1
            if c[2].item() == 1:
                ctx[2]=1

            
            for i in self.A:
                if not int(i) == 4: 
                    if ctx.tolist() in self.A[i]:

                        cc_loss+= -torch.log(1-output_logits[count][int(i)])
            count=count+1
            
        return cc_loss
    
