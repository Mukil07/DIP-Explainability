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
        self.all_lstm = nn.ModuleList()
        self.feat = None
        self.avgpool2d = nn.AvgPool2d(7)
        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None
        
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_lstm.append(LSTM(2048,num_classes,expand_dim))
            for i in range(self.n_attributes):
                self.all_lstm.append(LSTM(2048,1,expand_dim))
        else:
            self.all_lstm.append(LSTM(2048,num_classes,expand_dim))

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

        x = torch.cat((x1,x2),dim=1)     

        self.feat = self.avg_pool(x).flatten(1)
        
        if self.n_attributes ==0:
            x = self.dropout(self.avg_pool(x))[:,:,0,0,0]
            out.append(x)
            return out
        
        x = [self.avgpool2d(x[:,:,t]).view(x.size(0),-1) for t in range(x.size(2))]
        x = torch.stack(x) #(1,2048,2) (b,dim,frames)


        out = []
        
        # if bottleneck exists then, 
        
        for ls in self.all_lstm:
            out.append(ls(x))
       # import pdb;pdb.set_trace()
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            out[0] += self.cy_fc(attr_preds)

        return out


class LSTM(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Extend standard Torch Linear layer to include the option of expanding into 2 Linear layers
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # if self.expand_dim > 0:
        #     self.relu = nn.ReLU()
        #     self.fc_new = nn.Linear(input_dim, expand_dim)
        #     self.fc = nn.Linear(expand_dim, output_dim)
        # else:
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=512, num_layers=3) 
        self.fc = nn.Linear(512, output_dim)


    def forward(self, x):
        #import pdb;pdb.set_trace()
        hidden = None
        for ind in range(x.size(0)):
            output, hidden = self.lstm(x[ind],hidden)

        #import pdb;pdb.set_trace()
        x = self.fc(output)
        return x
    
def ModelXtoCtoY_lstm(num_classes, multitask_classes, multitask, n_attributes, bottleneck, expand_dim,
                 use_relu, use_sigmoid,connect_CY, dropout):
    #import pdb;pdb.set_trace()
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
