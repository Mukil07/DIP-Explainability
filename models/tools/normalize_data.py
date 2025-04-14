from utils.standardization import OnlineMeanStd
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import argparse
import os
from sklearn.model_selection import KFold
from utils.DIPX import CustomDataset
from torch.utils.data import Dataset, DataLoader, random_split

def cross_validate_model(args, dataset, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch)

        #import pdb;pdb.set_trace()
        mean,std = OnlineMeanStd()(train_subset, batch_size=args.batch, method='strong')
        print("mean for gaze and go pro are",mean)
        print("std for gaze and go pro are",std)
        exit()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--directory", help="Directory for home_dir", default = os.path.expanduser('~'))
    parser.add_argument("-e", "--epochs", type = int, help="Number of epochs", default = 1)
    parser.add_argument("--mem_per_layer", type = int, help="Number of memory tokens", default = 3)
    parser.add_argument("--dataset",  type = str, default = None)
    parser.add_argument("--model",  type = str, default = None)
    parser.add_argument("--debug",  type = str, default = None)
    parser.add_argument("--technique",  type = str, default = None)
    parser.add_argument("--num_classes",  type = int, default = 5)
    parser.add_argument("--batch",  type = int, default = 1)
    parser.add_argument("--distributed",  type = bool, default = False)
    parser.add_argument("--n_attributes", type = int, default= None) # for bottleneck

    parser.add_argument("--connect_CY", type = bool, default= False)
    parser.add_argument("--expand_dim", type = int, default= 0)
    parser.add_argument("--use_relu", type = bool, default= False)
    parser.add_argument("--use_sigmoid", type = bool, default= False)
    parser.add_argument("--multitask_classes", type = int, default=None) # for final classification along with action classificaiton
    parser.add_argument("--dropout", type = float, default= 0.45)
    
    parser.add_argument("-bottleneck",  action="store_true", help="Enable bottleneck mode")
    parser.add_argument("-gaze_cbm", action="store_true", help="Enable gaze CBM mode")
    parser.add_argument("-ego_cbm", action="store_true", help="Enable ego CBM mode")
    parser.add_argument("-multitask", action="store_true", help="Enable multitask mode")
    parser.add_argument("-combined_bottleneck", action="store_true", help="Enable combined_bottleneck mode")
    args = parser.parse_args()

    home_dir = str(args.directory)
    cache_dir = os.path.join(home_dir, "mukil")

    dataset = CustomDataset(debug = args.debug)
    cross_validate_model(args,dataset)
