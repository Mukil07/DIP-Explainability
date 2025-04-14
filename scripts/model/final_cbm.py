
import torch.nn as nn 
import torch 
from model.cbm_models_gaze import MLP, End2EndModel

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

class add_fc(nn.Module):

    def __init__(self,model,embed_dim,num_classes, multitask_classes, multitask, n_attributes, bottleneck, expand_dim,
                 use_relu, use_sigmoid,connect_CY, dropout):
        
        super(add_fc, self).__init__()
        self.model = model
        self.embed_dim = embed_dim

        self.bottleneck = bottleneck
        self.n_attributes = n_attributes
        self.all_fc = nn.ModuleList()
       
        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None        
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(embed_dim, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(embed_dim, 1, expand_dim))
        else:
            self.all_fc.append(FC(embed_dim, num_classes, expand_dim))

    def forward(self,img1,img2):

        x = self.model(img1,img2)
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



def cbm_module(first_model,embed_dim, num_classes, multitask_classes, multitask, n_attributes, bottleneck, expand_dim,
                 use_relu, use_sigmoid,connect_CY, dropout):

    model1 = first_model

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
            model2 = MLP(input_dim=embed_dim, num_classes=num_classes, expand_dim=expand_dim)
            model3 = MLP(input_dim=embed_dim, num_classes=15, expand_dim=expand_dim) # gaze head 
            model4 = MLP(input_dim=embed_dim, num_classes=17, expand_dim=expand_dim) # ego head 
            return End2EndModel(model1, model2, model3, model4, multitask, n_attributes, use_relu, use_sigmoid)

        else:
            model2 = MLP(input_dim=embed_dim, num_classes=num_classes, expand_dim=expand_dim)
            model3 = None
            model4 = None
            return End2EndModel(model1, model2, model3, model4, multitask, n_attributes, use_relu, use_sigmoid)
