import torch 
import torch.nn as nn 
import numpy as np 

from utils.plot_confusion import confusion
import argparse
import torch.optim as optim
import torchvision
from tqdm.auto import tqdm
import os

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, f1_score



from utils.tsne import plot_tsne as TSNE
from utils.plot_confusion import confusion
from utils.loader import CustomDataset
from model import build_model

from tools.engine import train_one_epoch, val

def Trainer(args, train_subset, valid_subset ):

        
    log_dir = f"runs_{args.model}_DAAD_{args.technique}/DAAD"  # Separate log directory for each fold
    writer = SummaryWriter(log_dir)   


    # Initialize model, criterion, and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(args)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    model.to(device)

    #checkpoint = "weights/dino_vitbase16_pretrain.pth"
    ckp = torch.load(args.weights,map_location=device)
    # ckp = torch.load(checkpoint,map_location=device)
    del ckp['logits.conv3d.bias']
    del ckp['logits.conv3d.weight']
    #del ckp['pos_embed']
    model.first_model.load_state_dict(ckp,strict=False)

    if args.ego_cbm:
        for param in model.third_model.parameters():
            param.requires_grad = False

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch,pin_memory=True, shuffle= True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(valid_subset, batch_size=args.batch,drop_last=True)

    # weights = [4, 2, 4, 2, 1]
    # class_weights = torch.FloatTensor(weights).cuda()
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    if args.gaze_cbm or args.combined_bottleneck:
        criterion1 = nn.CrossEntropyLoss() # action classificaiton 
        criterion2 = nn.CrossEntropyLoss() # gaze classificaiotn (bottleneck)
        criterion3 = nn.BCEWithLogitsLoss()  # ego classification (multitask) 
    elif args.ego_cbm or args.multitask :       
        criterion1 = nn.CrossEntropyLoss() # action classification 
        criterion2 = nn.BCEWithLogitsLoss() # ego multilabel classsification (bottleneck)
        criterion3 = nn.CrossEntropyLoss() # gaze classification (multitask)

    else:

        criterion1 = nn.CrossEntropyLoss()
        criterion2=None
        criterion3=None
    
    criterion= [criterion1, criterion2, criterion3]
    ### OLDER Settings 

    #criterion=torch.nn.CrossEntropyLoss()
    #optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=5e-2)
    
    ### TESTING SGD 
    learning_rate = 0.008
    momentum = 0.9
    weight_decay = 0.001

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    ####

    # Train and evaluate the model
    if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:


        

        acc,f1,acc_gaze,f1_gaze,acc_ego,f1_ego,f1_ego_micro,f1_ego_macro = train_eval(args, train_loader, val_loader, model, criterion, optimizer, device,writer)
        print("Average Accuracy",acc)
        print("Average F1",f1)
        print("Average Accuracy(Gaze)",acc_gaze)
        print("Average F1(Gaze)",f1_gaze)
        print("Average Accuracy(Ego)",acc_ego)
        print("Average F1(Ego)",f1_ego)
        print("Average F1(EGO) micro",f1_ego_micro)
        print("Average F1(EGO) macro",f1_ego_macro)
    else:
        accuracy, f1 = train_eval(args, train_loader, val_loader, model, criterion, optimizer, device,writer)
        print("Average Accuracy",accuracy)
        print("Average F1",f1)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")



def train_eval(args, train_dataloader, valid_dataloader, model, criterion, optimizer, device,writer):
    
    T=1
    num_epochs=args.num_epochs
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, valid_accuracy = [], []

    if args.debug:
        patience =1
    else:
        patience =10
    #patience = 10 
    min_delta = 0.0001  
    best_acc = 0 
    counter = 0 
    lam1,lam2 = 0.5,0.5
    best_val_acc = 0
    if args.technique: 
        save_dir = f"best_{args.model}_{args.dataset}_{args.technique}_dir"

    else:
        save_dir = f"best_{args.model}_{args.dataset}_dir"
        
    patience_counter = 0
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    best_model_path = os.path.join(save_dir, f"best_{args.model}_{args.dataset}.pth") 
    print("Started Training")
    for epoch in range(num_epochs):

        train_loss, all_labels, all_preds = train_one_epoch(args,epoch, train_dataloader, model, criterion, optimizer,cosine_scheduler, device, writer)
        epoch_loss = train_loss/len(train_dataloader)
        print(f"Epoch {epoch} Loss: {epoch_loss:.4f}")

        train_losses.append(epoch_loss)

        all_preds = np.hstack(all_preds)
        all_labels = np.hstack(all_labels)

        accuracy_train = accuracy_score(all_labels, all_preds)
        f1_train = f1_score(all_labels, all_preds, average='weighted')

        print(f"Accuracy (Training): {accuracy_train:.4f}, F1 Score: {f1_train:.4f}")

        
        ### EVAL
        if epoch % 5 ==0: 

            print("Started Evaluating")
            val_loss_running, preds = val(args, valid_dataloader, model, device)
            val_loss = val_loss_running/len(valid_dataloader)


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

            del all_preds_gaze, all_labels_gaze, all_preds_ego, all_labels_ego

            # writer.add_scalar("Loss/Train", epoch_loss, epoch)
            # writer.add_scalar("Loss/Validation", val_loss, epoch)
            # writer.add_scalar("Accuracy/Validation", accuracy_val, epoch)
            # writer.add_scalar("Accuracy/Train", accuracy_train, epoch)

            # writer.add_scalar("F1/Validation", f1_val, epoch)
            # writer.add_scalar("F1/Train", f1_train, epoch)
            if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:
                final_acc = (accuracy_val + accuracy_val_ego + accuracy_val_gaze)/3
            else:
                final_acc = accuracy_val

            if  final_acc - best_acc  > min_delta:
                best_acc = final_acc
                if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:
                    best_ego_acc = accuracy_val_ego
                    best_ego_f1 = f1_val_ego

                    best_gaze_acc = accuracy_val_gaze
                    best_gaze_f1 = f1_val_gaze

                    best_ego_f1_macro = f1_val_ego_macro
                    best_ego_f1_micro = f1_val_ego_micro
                best_val_acc = accuracy_val
                best_val_f1 = f1_val

                counter = 0  
                if args.distributed:
                    torch.save(model.module.state_dict(),best_model_path)
                else:
                    torch.save(model.state_dict(),best_model_path)
                patience_counter=0
                print("best model is saved ")  
                print("patience is set to zero") 

            else:
                patience_counter += 1
                print(f"No improvement ... Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

    if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:
        return best_val_acc,best_val_f1,best_gaze_acc,best_gaze_f1,best_ego_acc,best_ego_f1,best_ego_f1_micro, best_ego_f1_macro
    else:

        return best_val_acc,best_val_f1
    



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
    parser.add_argument("--weights",  type = str, default = "models/weights/rgb_imagenet_modified.pt")
    parser.add_argument("--connect_CY", type = bool, default= False)
    parser.add_argument("--expand_dim", type = int, default= 0)
    parser.add_argument("--use_relu", type = bool, default= False)
    parser.add_argument("--use_sigmoid", type = bool, default= False)
    parser.add_argument("--multitask_classes", type = int, default=None) # for final classification along with action classificaiton
    parser.add_argument("--dropout", type = float, default= 0.0)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--clusters",default=5,type=int)
    parser.add_argument("-bottleneck",  action="store_true", help="Enable bottleneck mode")
    parser.add_argument("-gaze_cbm", action="store_true", help="Enable gaze CBM mode")
    parser.add_argument("-ego_cbm", action="store_true", help="Enable ego CBM mode")
    parser.add_argument("-multitask", action="store_true", help="Enable multitask mode")
    parser.add_argument("-combined_bottleneck", action="store_true", help="Enable combined_bottleneck mode")
    args = parser.parse_args()
    home_dir = str(args.directory)
    cache_dir = os.path.join(home_dir, "mukil_new")
    world_size = torch.cuda.device_count()

    train_csv = "DATA/train.csv"
    val_csv = "DATA/val.csv"

    # transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_subset = CustomDataset(train_csv, debug = args.debug)
    val_subset = CustomDataset(val_csv, debug=args.debug)

    Trainer(args, train_subset, val_subset)
