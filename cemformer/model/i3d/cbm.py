from model.cbm_models_gaze import MLP, End2EndModel
from model.i3d.i3d_multi import InceptionI3d
import scipy.stats as stats
import torch 
import torch.nn as nn 

class CBM(InceptionI3d):

    def __init__(self, num_classes=5, n_attributes=17, bottleneck=True, expand_dim=512, connect_CY=False, dropout_keep_prob=0.45):

        super(CBM, self).__init__(num_classes=num_classes, dropout_keep_prob=dropout_keep_prob)
        self.bottleneck = bottleneck
        self.n_attributes = n_attributes
        self.all_fc = nn.ModuleList()
        self.feat = None
       
        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None
        
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(2048, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(2048, 1, expand_dim))
        else:
            self.all_fc.append(FC(2048, num_classes, expand_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1,x2):
        
        #import pdb;pdb.set_trace()
        for end_point in self.VALID_ENDPOINTS:
            # if end_point == 'Logits':
            #    # import pdb;pdb.set_trace()
            if end_point in self.end_points1:
                x1 = self._modules[end_point](x1) # use _modules to work with dataparallel

        for end_point in self.VALID_ENDPOINTS:
            # if end_point == 'Logits':
            #     #import pdb;pdb.set_trace()
            if end_point in self.end_points2:
                x2 = self._modules[end_point](x2) # use _modules to work with dataparallel

       # import pdb;pdb.set_trace()
        x = torch.cat((x1,x2),dim=1)               
        self.feat = self.avg_pool(x).flatten(1)
        x = self.dropout(self.avg_pool(x))[:,:,0,0,0]
        
        out = []
   #x= x.permute((0,2,3,4,1))
        if self.n_attributes ==0:
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


def ModelXtoCtoY(num_classes, multitask_classes, multitask, n_attributes, bottleneck, expand_dim,
                 use_relu, use_sigmoid,connect_CY, dropout):

    model1 = CBM(num_classes=num_classes,n_attributes=n_attributes,
                  bottleneck=bottleneck, expand_dim=expand_dim,connect_CY=connect_CY, dropout_keep_prob = dropout)

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
            model2 = MLP(input_dim=2048, num_classes=num_classes, expand_dim=expand_dim)
            model3 = MLP(input_dim=2048, num_classes=15, expand_dim=expand_dim) # gaze head 
            model4 = MLP(input_dim=2048, num_classes=17, expand_dim=expand_dim) # ego head 
            return End2EndModel(model1, model2, model3, model4, multitask, n_attributes, use_relu, use_sigmoid)

        else:
            model2 = MLP(input_dim=2048, num_classes=num_classes, expand_dim=expand_dim)
            model3 = None
            model4 = None
            return End2EndModel(model1, model2, model3, model4, multitask, n_attributes, use_relu, use_sigmoid)
