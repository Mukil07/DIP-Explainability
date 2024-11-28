import torch 
import torch.nn as nn 
import numpy as np 

import argparse

import torch.optim as optim

from torchvision import datasets, transforms

from tqdm.auto import tqdm
import os
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.transforms.functional import to_pil_image, to_grayscale
from torch.utils.data import Dataset, DataLoader, random_split

from cc_loss import Custom_criterion
from utils.Custom_dataloader_eval import CustomDataset
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--directory", help="Directory for home_dir", default = os.path.expanduser('~'))
    parser.add_argument("-e", "--epochs", type = int, help="Number of epochs", default = 1)
    parser.add_argument("--mem_per_layer", type = int, help="Number of memory tokens", default = 3)
    parser.add_argument("--model",  type = str, default = None)
    parser.add_argument("--num_classes",  type = int, default = 5)
    parser.add_argument("--debug",  type = str, default = None)

    args = parser.parse_args()

    home_dir = str(args.directory)
    cache_dir = os.path.join(home_dir, "mukil")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = CustomDataset(debug = args.debug, transform=transform)

    validation_loader = DataLoader(dataset, batch_size=1, num_workers=1)

    number_of_classes=5
    warmup_steps = 5


    model = build_model(args)

    print(model)
    
    checkpoint = "/scratch/mukil/cemformer/best_model_dir_dipx/best_model2.pth"
    #import pdb;pdb.set_trace()
    new_length = 393 
    num_channels = 768  

    model.load_state_dict(torch.load(checkpoint),strict=False)
    model = model.to(device)
    print("Started Evaluating")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for i, (img1,img2,cls) in tqdm(enumerate(validation_loader)): 

            
            # img1= torch.squeeze(img1,dim=0)
            # img2= torch.squeeze(img2,dim=0)

            # b,_,_,_=img1.shape
            
            img1 = img1.to(device)
            img2 = img2.to(device)

            #cls= cls.type(torch.LongTensor)
            label = cls.to(device)

            # Forward pass
            #import pdb;pdb.set_trace()
            img1=img1.type(torch.cuda.FloatTensor)
            img2=img2.type(torch.cuda.FloatTensor)
            outputs = model(img1,img2) #  size - (10,4,768)
            
           # import pdb;pdb.set_trace()
            outputs = outputs[0].mean(dim=1)
            #import pdb;pdb.set_trace()
            predicted = torch.argmax(outputs)
            print(predicted,label)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(label[0].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted') #'weighted' or 'macro' s
    print("accuracy and F1",accuracy,f1) 