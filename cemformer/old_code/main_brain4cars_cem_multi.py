import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt

from utils.plot_confusion import confusion
import argparse

import torch.optim as optim

from torchvision import datasets, transforms

from tqdm.auto import tqdm
import os
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix



from utils.tsne import plot_tsne as TSNE
from utils.plot_confusion import confusion
from utils.Brain4Cars import CustomDataset
from utils.loss import cc_loss

from model import build_model


def cross_validate_model(args, dataset, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        log_dir = f"runs_{args.model}_{args.dataset}_{args.technique}/fold_brain_{fold}"  # Separate log directory for each fold
        writer = SummaryWriter(log_dir)   


        # Initialize model, criterion, and optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #import pdb;pdb.set_trace()
        model = build_model(args)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

        model.to(device)

        checkpoint = "weights/dino_vitbase16_pretrain.pth"
        #ckp = torch.load('/scratch/mukil/final/pytorch-i3d/models/rgb_imagenet_modified.pt',map_location=device)
        ckp = torch.load(checkpoint,map_location=device)
        # del ckp['logits.conv3d.bias']
        # del ckp['logits.conv3d.weight']
        del ckp['pos_embed']
        #model.first_model.load_state_dict(ckp,strict=False)
        #import pdb;pdb.set_trace()
        model.vit_1.load_state_dict(ckp,strict=False)
        model.vit_2.load_state_dict(ckp,strict=False)
        #import pdb;pdb.set_trace()
        # Creating data loaders for training and validation
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch)


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
        ### OLDER Settings 

        #criterion=torch.nn.CrossEntropyLoss()
        #optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=5e-2)
        
        ### TESTING SGD 
        learning_rate = 0.001
        momentum = 0.9
        weight_decay = 0.001

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        ####

        # Train and evaluate the model
        accuracy, f1 = train(args, train_loader, val_loader, model, criterion1, criterion2, criterion3, optimizer, device,writer,fold)
        accuracies.append(accuracy)
        f1_scores.append(f1)

        
    # Calculate the average and standard deviation of the metrics
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_f1_score = np.mean(f1_scores)
    std_f1_score = np.std(f1_scores)

    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1_score:.4f} ± {std_f1_score:.4f}")
    return avg_accuracy, std_accuracy, avg_f1_score, std_f1_score




def train(args, train_dataloader, valid_dataloader, model, criterion1, criterion2, criterion3, optimizer, device,writer,fold):
    model.train()
    # if args.distributed:
    #     for param in model.module.parameters():
    #         param.requires_grad = False
    #     #import pdb;pdb.set_trace()
    #     for block in model.module.vit.blocks:
    #         for param in block.attn.parameters():
    #             param.requires_grad = True

    #     for param in model.module.mlp_head.parameters():
    #         param.requires_grad = True
    # else:

    #     for param in model.parameters():
    #         param.requires_grad = False
    #     #import pdb;pdb.set_trace()
    #     for block in model.vit.blocks:
    #         for param in block.attn.parameters():
    #             param.requires_grad = True

    #     for param in model.mlp_head.parameters():
    #         param.requires_grad = True        

    #warmup_scheduler = LambdaLR(optimizer, warmup_linear)

    # Iterate through the dataloader
    T=1
    num_epochs=100
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, valid_accuracy = [], []
    print("Started Training")
    if args.debug:
        patience =1
    else:
        patience =10
    #patience = 10 
    min_delta = 0.0001  
    best_acc = 0 
    counter = 0 
    lam1,lam2 = 0.5,0.5

    cc_criterion = cc_loss()
    if args.technique: 
        save_dir = f"best_{args.model}_{args.dataset}_{args.technique}_dir"

    else:
        save_dir = f"best_{fold}_{args.model}_{args.dataset}_dir"
        
    patience_counter = 0
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    best_model_path = os.path.join(save_dir, f"best_{fold}_{args.model}_{args.dataset}.pth") 
    print("Started Training")
    for epoch in range(num_epochs):
        all_preds = []
        all_labels = []
        train_loss = 0.0
        ego=None
        gaze=None
        feat= None
        for i, (img1,img2,cls,context) in tqdm(enumerate(train_dataloader)):
            

            img1 = img1.to(device)
            img2 = img2.to(device)

            label = cls.to(device)

            # Forward pass

            img1=img1.type(torch.cuda.FloatTensor)
            img2=img2.type(torch.cuda.FloatTensor)
            #import pdb;pdb.set_trace()
            #import pdb;pdb.set_trace()
            outputs = model(img1,img2)
           # import pdb;pdb.set_trace()
            feat = model.feat
            
            loss1 = criterion1(outputs[0],label)  
            #import pdb;pdb.set_trace()
            if args.gaze_cbm:

                loss3 = lam2*criterion3(outputs[1],torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).to(device))
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),gaze.cuda())
                loss = loss1 + loss2 + loss3
            
            elif args.ego_cbm:
                    
                loss3 = lam2*criterion3(outputs[1],gaze.cuda())
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).to(device))
                loss = loss1 + loss2 + loss3

            elif args.multitask:

                loss3 = lam2*criterion3(outputs[1],gaze.cuda())
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).to(device))

                loss = loss1 + loss2 + loss3
            
            elif args.combined_bottleneck:
                #import pdb;pdb.set_trace()
                loss2 = lam1*criterion2(torch.hstack(outputs[1:16]),gaze.cuda())
                loss3 = lam2*criterion3(torch.hstack(outputs[16:33]),torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).to(device))
                loss = loss1 + loss2 + loss3
            else:

                
                context = [list(x) for x in zip(*context)]
                ccloss = cc_criterion.calc_loss(context,outputs[0])
                loss = loss1+ccloss

            #loss = criterion(outputs, label)


            # uncomment for exponential loss

            #time_frame= np.arange(0,T,T/b)
            #import pdb;pdb.set_trace()
            #loss_time = 0 
            # for t in time_frame:
            #     loss_time += 1*(np.exp(-1*(T-t)))
            # #import pdb;pdb.set_trace()
            # loss = loss_time*loss

            loss_val = loss.cpu()
            train_loss += loss_val

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            
            predicted = torch.argmax(outputs[0],dim=1)
            all_preds.append(predicted.cpu())
            all_labels.append(label.cpu())

            cosine_scheduler.step()


        epoch_loss = train_loss/len(train_dataloader)

        print(f"epoch {epoch} Loss: {epoch_loss:.4f}")

        train_losses.append(epoch_loss)

        all_preds = np.hstack(all_preds)
        all_labels = np.hstack(all_labels)

        accuracy_train = accuracy_score(all_labels, all_preds)
        f1_train = f1_score(all_labels, all_preds, average='weighted')

        print(f"Accuracy(Trainign): {accuracy_train:.4f}, F1 Score: {f1_train:.4f}")

        ### EVAL 
        print("Started Evaluating")
        model.eval()

        all_preds = []
        all_labels = []
        all_preds_gaze = []
        all_labels_gaze = []
        all_preds_ego = []
        all_labels_ego = []

        val_loss_running=0
        FEAT=[]
        LABEL=[]
        gaze=None
        ego=None
        with torch.no_grad():

            for i, (img1,img2,cls,context) in tqdm(enumerate(valid_dataloader)): 

                img1 = img1.to(device)
                img2 = img2.to(device)

                label = cls.to(device)

                # Forward pass

                img1=img1.type(torch.cuda.FloatTensor)
                img2=img2.type(torch.cuda.FloatTensor)
                outputs = model(img1,img2) 
                
                feat = model.feat


                loss1 = criterion1(outputs[0],label)  
                
                if args.gaze_cbm:

                    loss3 = lam2*criterion3(outputs[1],torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).to(device))
                    loss2 = lam1*criterion2(torch.hstack(outputs[2:]),gaze.cuda())
                    loss = loss1 + loss2 + loss3
                    predicted_gaze = torch.argmax(outputs[2],dim=1)
                    all_preds_gaze.append(predicted_gaze.cpu())
                    all_labels_gaze.append(gaze.cpu())          
                    predicted_ego = (torch.sigmoid(outputs[1]) > 0.5).float().cpu()
                    all_preds_ego.append(predicted_ego)
                    all_labels_ego.append(torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).cpu())

                elif args.ego_cbm:
                        
                    loss3 = lam2*criterion3(outputs[1],gaze.cuda())
                    loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).to(device))
                    loss = loss1 + loss2 + loss3
                    predicted_gaze = torch.argmax(outputs[1],dim=1)
                    all_preds_gaze.append(predicted_gaze.cpu())
                    all_labels_gaze.append(gaze.cpu())    
                    predicted_ego = (torch.sigmoid(torch.hstack(outputs[2:])) > 0.5).float().cpu()
                    all_preds_ego.append(predicted_ego)
                   #import pdb;pdb.set_trace()
                    all_labels_ego.append(torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).cpu())

                elif args.multitask:

                    loss3 = lam2*criterion3(outputs[1],gaze.cuda())
                    loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).to(device))

                    loss = loss1 + loss2 + loss3
                    #import pdb;pdb.set_trace()
                    predicted_gaze = torch.argmax(outputs[1],dim=1)
                    all_preds_gaze.append(predicted_gaze.cpu())
                    all_labels_gaze.append(gaze.cpu()) 
                    #import pdb;pdb.set_trace()
                    predicted_ego = (torch.sigmoid(outputs[2]) > 0.5).float().cpu()
                    all_preds_ego.append(predicted_ego)
                    all_labels_ego.append(torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).cpu())

                elif args.combined_bottleneck:
                    #import pdb;pdb.set_trace()
                    loss2 = lam1*criterion2(torch.hstack(outputs[1:16]),gaze.cuda())
                    loss3 = lam2*criterion3(torch.hstack(outputs[16:33]),torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).to(device))
                    loss = loss1 + loss2 + loss3
                    predicted_gaze = torch.argmax(torch.hstack(outputs[1:16]),dim=1)
                    all_preds_gaze.append(predicted_gaze.cpu())
                    all_labels_gaze.append(gaze.cpu())          
                    predicted_ego = (torch.sigmoid(torch.hstack(outputs[16:33])) > 0.5).float().cpu()
                    all_preds_ego.append(predicted_ego)
                    all_labels_ego.append(torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).cpu())    

                else:

                    
                    context = [list(x) for x in zip(*context)]
                    ccloss = cc_criterion.calc_loss(context,outputs[0])
                    loss = loss1+ccloss

                val_loss_running+=loss

                predicted = torch.argmax(outputs[0],dim=1)
                

                all_preds.append(predicted.cpu())
                all_labels.append(label.cpu())
 

                FEAT.append(feat.cpu())
                LABEL.append(label.cpu())
        #ssimport pdb;pdb.set_trace()
        tsne = TSNE()
        tsne_img = tsne.plot(FEAT,LABEL,args.dataset)

        all_labels = np.hstack(all_labels)
        all_preds = np.hstack(all_preds)

        if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:
            

            all_labels_gaze = np.hstack(all_labels_gaze)
            all_preds_gaze = np.hstack(all_preds_gaze)
            all_labels_ego = np.hstack(all_labels_ego)
            all_preds_ego = np.hstack(all_preds_ego)    
            #confusion(all_labels_ego, all_preds_ego,'ego',writer,epoch)
            confusion(all_labels_gaze, all_preds_gaze,'gaze',writer,epoch)
        # for Action Classification 

        confusion(all_labels, all_preds,'action',writer,epoch)

        writer.add_figure('TSNE', tsne_img,epoch)

        val_loss = val_loss_running/len(valid_dataloader)

        accuracy_val = accuracy_score(all_labels, all_preds)
        f1_val = f1_score(all_labels, all_preds, average='weighted') #'weighted' or 'macro' s

        if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:

            accuracy_val_gaze = accuracy_score(all_labels_gaze, all_preds_gaze)
            f1_val_gaze = f1_score(all_labels_gaze, all_preds_gaze, average='weighted') #'weighted' or 'macro' s
            
            accuracy_val_ego = accuracy_score(all_labels_ego, all_preds_ego)
            f1_val_ego = f1_score(all_labels_ego, all_preds_ego, average='weighted')

             #'weighted' or 'macro' s
            print("accuracy and F1(GAZE)",accuracy_val_gaze,f1_val_gaze) 
            print("accuracy and F1(GAZE)",accuracy_val_ego,f1_val_ego) 

            writer.add_scalar("Accuracy/Validation(Gaze)", accuracy_val_gaze, epoch)
            writer.add_scalar("F1/Validation(Gaze)", f1_val_gaze, epoch)
            writer.add_scalar("Accuracy/Validation(Ego)", accuracy_val_ego, epoch)
            writer.add_scalar("F1/Validation(Ego)", f1_val_ego, epoch)

        print("accuracy and F1",accuracy_val,f1_val) 

        

        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy_val, epoch)
        writer.add_scalar("Accuracy/Train", accuracy_train, epoch)

        writer.add_scalar("F1/Validation", f1_val, epoch)
        writer.add_scalar("F1/Train", f1_train, epoch)

        if  accuracy_val - best_acc  > min_delta:
            best_acc = accuracy_val
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



        
    return accuracy_val, f1_val
    


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
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--connect_CY", type = bool, default= False)
    parser.add_argument("--expand_dim", type = int, default= 0)
    parser.add_argument("--use_relu", type = bool, default= False)
    parser.add_argument("--use_sigmoid", type = bool, default= False)
    parser.add_argument("--multitask_classes", type = int, default=None) # for final classification along with action classificaiton
    # all the flags for different modes are here, 
    parser.add_argument("-gaze_cbm", action="store_true", help="Enable gaze CBM mode")
    parser.add_argument("-ego_cbm", action="store_true", help="Enable ego CBM mode")
    parser.add_argument("-multitask", action="store_true", help="Enable multitask mode")
    parser.add_argument("-combined_bottleneck", action="store_true", help="Enable combined_bottleneck mode")
    parser.add_argument("-bottleneck",  action="store_true", help="Enable bottleneck mode")

    args = parser.parse_args()

    home_dir = str(args.directory)
    cache_dir = os.path.join(home_dir, "mukil")

    dataset = CustomDataset(debug = args.debug)
    cross_validate_model(args,dataset)

