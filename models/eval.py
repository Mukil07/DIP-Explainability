import torch 
import argparse

from tqdm.auto import tqdm
import os
import torch 
import torch.nn as nn 
import numpy as np 

from utils.DIPX_350 import CustomDataset
from utils.gradcam import GradCAM
from utils.save_img import visualize
from sklearn.metrics import accuracy_score, f1_score

from model import build_model
from tools.engine import train, val

def trainer(args, valid_subset, n_splits=5):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(args)
    model.to(device)
    #import pdb;pdb.set_trace()
    #checkpoint = "weights/dino_vitbase16_pretrain.pth"
    ckp = torch.load(args.weights,map_location=device)
    

    model.load_state_dict(ckp,strict=True)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    val_loader = torch.utils.data.DataLoader(valid_subset, batch_size=args.batch,pin_memory=True,drop_last=True)

    for param in model.parameters():
        param.requires_grad = False

    # plot gradcam 
    loss, preds = val(args, val_loader, model, device)

    evaluate(args, loss, preds, val_loader)


def evaluate(args, running_loss,preds, valid_dataloader):

    
    val_loss = running_loss/len(valid_dataloader)



    if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:

        all_labels, all_preds, all_labels_gaze, all_preds_gaze, all_labels_ego, all_preds_ego = preds

        accuracy_val_gaze = accuracy_score(all_labels_gaze, all_preds_gaze)
        f1_val_gaze = f1_score(all_labels_gaze, all_preds_gaze, average='weighted') #'weighted' or 'macro' s
        
        accuracy_val_ego = accuracy_score(all_labels_ego, all_preds_ego)
        f1_val_ego = f1_score(all_labels_ego, all_preds_ego, average='weighted')
        f1_val_ego_micro = f1_score(all_labels_ego, all_preds_ego, average='micro')
        f1_val_ego_macro = f1_score(all_labels_ego, all_preds_ego, average='macro')

        #'weighted' or 'macro' s
        print("accuracy and F1(GAZE)",accuracy_val_gaze,f1_val_gaze) 
        print("accuracy and F1(EGO)",accuracy_val_ego,f1_val_ego) 
        print("EGO Macro F1 and Micro F1",f1_val_ego_macro, f1_val_ego_micro)
        # writer.add_scalar("Accuracy/Validation(Gaze)", accuracy_val_gaze, epoch)
        # writer.add_scalar("F1/Validation(Gaze)", f1_val_gaze, epoch)
        # writer.add_scalar("Accuracy/Validation(Ego)", accuracy_val_ego, epoch)
        # writer.add_scalar("F1/Validation(Ego)", f1_val_ego, epoch)
        accuracy_val = accuracy_score(all_labels, all_preds)
        f1_val = f1_score(all_labels, all_preds, average='weighted') #'weighted' or 'macro' s
    else:
        all_labels, all_preds = preds
        accuracy_val = accuracy_score(all_labels, all_preds)
        f1_val = f1_score(all_labels, all_preds, average='weighted') #'weighted' or 'macro' s


    print("accuracy and F1",accuracy_val,f1_val) 
if __name__ == '__main__':
    # seed = 37

    # np.random.seed(seed)
    # torch.manual_seed(seed) 
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
    parser.add_argument("--weights",  type = str, default = "weights/best_i3d_fine_dipx.pth")
    parser.add_argument("--connect_CY", type = bool, default= False)
    parser.add_argument("--expand_dim", type = int, default= 0)
    parser.add_argument("--use_relu", type = bool, default= False)
    parser.add_argument("--use_sigmoid", type = bool, default= False)
    parser.add_argument("--multitask_classes", type = int, default=None) # for final classification along with action classificaiton
    parser.add_argument("--dropout", type = float, default= 0.45)
    parser.add_argument("--clusters",default=5,type=int)
    parser.add_argument("-bottleneck",  action="store_true", help="Enable bottleneck mode")
    parser.add_argument("-gaze_cbm", action="store_true", help="Enable gaze CBM mode")
    parser.add_argument("-ego_cbm", action="store_true", help="Enable ego CBM mode")
    parser.add_argument("-multitask", action="store_true", help="Enable multitask mode")
    parser.add_argument("-combined_bottleneck", action="store_true", help="Enable combined_bottleneck mode")
    args = parser.parse_args()

    home_dir = str(args.directory)
    cache_dir = os.path.join(home_dir, "mukil_new")
    
    val_csv = "/scratch/mukil_new/dipx/val.csv"

    val_subset = CustomDataset(val_csv, debug=args.debug)

    trainer(args, val_subset)
