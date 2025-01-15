
from utils.dip_model import generate_model


def extract_inside_vector(inside_model_path, input_data):
    # 모델 및 옵션 설정
    
    model, _ = generate_model(opt)
    checkpoint = torch.load(inside_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    with torch.no_grad():
        inputs = input_data.float().to(device)
        outputs = model(inputs)
        inside_vector = outputs.cpu().numpy()

    return inside_vector

def extract_outside_vector(outside_model_path, input_data):
    model = encoder(hidden_channels=[128, 64, 64, 32], sample_size=opt.sample_size, sample_duration=opt.sample_duration_outside).to(device)    
    checkpoint = torch.load(outside_model_path)
    state_dict = checkpoint['state_dict'] 
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)

    model.eval()
    with torch.no_grad():
        inputs = input_data.float().to(device)
        outputs = model(inputs) 
        # To Tensor
        outside_vector = outputs.cpu().numpy()

    return outside_vector

class Conv_Block(nn.Module):
    def __init__(self):
        super(Conv_Block, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
            
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        #flatten the output
        x = x.view(x.size(0),-1)
        return x
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # FC0: 3072 -> 2048
        
        self.Classifier_fc = nn.Sequential(
            nn.Linear(3072, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 5),
            # nn.BatchNorm1d(5),
            nn.ReLU(),
            nn.Softmax(dim=1) 
        )

    def forward(self, x):
        x = self.Classifier_fc(x)
        return x
