from model.CUB import MLP, End2EndModel
from model.i3d import InceptionI3d
import scipy.stats as stats
import torch 
import torch.nn as nn 

class CBM(InceptionI3d):

    def __init__(self, num_classes=5, n_attributes=3, bottleneck=False, expand_dim=0, connect_CY=False):

        super(CBM, self).__init__()
        self.bottleneck = bottleneck
        self.n_attributes = n_attributes
        self.all_fc = nn.ModuleList()

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(1024, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(1024, 1, expand_dim))
        else:
            self.all_fc.append(FC(1024, num_classes, expand_dim))

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

    def forward(self, x):

        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel
        import pdb;pdb.set_trace()
        out = []
        x= x.permute((0,2,3,4,1))
        for fc in self.all_fc:
            out.append(fc(x))
        import pdb;pdb.set_trace()
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


def ModelXtoCtoY(num_classes, n_attributes, expand_dim,
                 use_relu, use_sigmoid,connect_CY):

    model1 = CBM(num_classes=num_classes,n_attributes=n_attributes,
                  bottleneck=False, expand_dim=expand_dim,connect_CY=connect_CY)

    model2 = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)

    return End2EndModel(model1, model2, use_relu, use_sigmoid)
