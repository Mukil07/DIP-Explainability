import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class End2EndModel(torch.nn.Module):
    def __init__(self, model1, model2, model3,model4, multitask, n_attributes, use_relu=False, use_sigmoid=False):
        super(End2EndModel, self).__init__()
        self.first_model = model1
        self.sec_model = model2
        self.third_model = model3
        self.fourth_model = model4
        self.multitask = multitask
        self.n_attributes = n_attributes
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid

    def forward_stage2(self, stage1_out):
        #import pdb;pdb.set_trace()
        if self.use_relu:
            attr_outputs = [nn.ReLU()(o) for o in stage1_out]
        elif self.use_sigmoid:
            attr_outputs = [torch.nn.Sigmoid()(o) for o in stage1_out]
        else:
            attr_outputs = stage1_out

        stage2_inputs = attr_outputs
        stage2_inputs = torch.cat(stage2_inputs, dim=1)
  
        if self.n_attributes >0: # for bottleneck 
            if self.multitask:
                all_out = [self.sec_model(stage2_inputs),self.third_model(stage2_inputs)]
                all_out.extend(stage1_out)
                return all_out
            else:
                all_out = [self.sec_model(stage2_inputs)]
                all_out.extend(stage1_out)
                return all_out
        else: 
            if self.multitask: # for multitask 

                all_out = [self.sec_model(stage2_inputs),self.third_model(stage2_inputs), self.fourth_model(stage2_inputs)]
            else: # for no bottleneck 
                all_out = [self.sec_model(stage2_inputs)]

            return all_out


    def forward(self, x1,x2):
        

        outputs = self.first_model(x1,x2)
        return self.forward_stage2(outputs)

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim:
            self.linear = nn.Linear(input_dim, expand_dim)
            self.activation = torch.nn.ReLU()
            self.linear2 = nn.Linear(expand_dim, num_classes) #softmax is automatically handled by loss function
        #$import pdb;pdb.set_trace()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        #import pdb;pdb.set_trace()
        x = self.linear(x)
        if hasattr(self, 'expand_dim') and self.expand_dim:
            x = self.activation(x)
            x = self.linear2(x)
        return x

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
                 use_relu, use_sigmoid,connect_CY):
        
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
            out.append(fc(x).mean(1)) #for fine model 
        
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            out[0] += self.cy_fc(attr_preds)

        return out



def cbm_module(first_model,embed_dim, num_classes, multitask_classes, multitask, n_attributes, bottleneck, expand_dim,
                 use_relu, use_sigmoid,connect_CY):

    model1 = first_model
    #import pdb;pdb.set_trace()
    
    
    if n_attributes>0:
        model2 = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
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
