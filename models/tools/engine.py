
import torch 
import torch.nn as nn 
import numpy as np 

import argparse
import torch.optim as optim
import torchvision
from tqdm.auto import tqdm
import os

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, f1_score


def train_one_epoch(args,epoch, train_dataloader, model, criterion, optimizer, scheduler, device, writer):

    print("Epoch: ",epoch )
    model.train()

    criterion1, criterion2, criterion3 = criterion
    lam1,lam2 = 0.5,0.5


    all_preds = []
    all_labels = []
    train_loss = 0.0
    for i, (img1,img2,cls,gaze,ego) in enumerate(tqdm(train_dataloader)):
        

        img1 = img1.to(device)
        img2 = img2.to(device)

        label = cls.to(device)

        # import pdb;pdb.set_trace()
        # Forward pass

        img1=img1.type(torch.cuda.FloatTensor)
        img2=img2.type(torch.cuda.FloatTensor)
        #import pdb;pdb.set_trace()
        outputs = model(img1,img2)

        
        loss1 = criterion1(outputs[0],label)  
        #import pdb;pdb.set_trace()
        if args.gaze_cbm:

            loss3 = lam2*criterion3(outputs[1],torch.vstack(ego).to(dtype=torch.float).permute((1,0)).to(device))
            loss2 = lam1*criterion2(torch.hstack(outputs[2:]),gaze.cuda())
            loss = loss1 + loss2 + loss3
        
        elif args.ego_cbm:
                
            # loss3 = lam2*criterion3(outputs[1],gaze.cuda())
            loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((1,0)).to(device))
            loss = loss1 + loss2 #+ loss3
        
        elif args.combined_bottleneck:
            
            loss2 = lam1*criterion2(torch.hstack(outputs[1:16]),gaze.cuda())
            loss3 = lam2*criterion3(torch.hstack(outputs[16:33]),torch.vstack(ego).to(dtype=torch.float).permute((1,0)).to(device))
            loss = loss1 + loss2 + loss3

        elif args.multitask:

            loss3 = lam2*criterion3(outputs[1],gaze.cuda())
            loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((1,0)).to(device))

            loss = loss1 + loss2 + loss3
        else:

            loss = loss1

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

        scheduler.step()

        del img1
        del img2 
        del label
        del outputs
        torch.cuda.empty_cache()
    return train_loss, all_labels, all_preds 

def val(args, valid_dataloader, model, device):


    model.eval()

    lam1,lam2 = 0.5,0.5
    #with torch.no_grad():            
    all_preds = []
    all_labels = []
    all_preds_gaze = []
    all_labels_gaze = []
    all_preds_ego = []
    all_labels_ego = []

    val_loss_running=0
    FEAT=[]
    LABEL=[]

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

    with torch.no_grad():  
        for i, (img1,img2,cls,gaze,ego) in enumerate(tqdm(valid_dataloader)): 

            img1 = img1.to(device)
            img2 = img2.to(device)

            label = cls.to(device)

            # Forward pass

            img1=img1.type(torch.cuda.FloatTensor)
            img2=img2.type(torch.cuda.FloatTensor)
            outputs = model(img1,img2) 

            
            feat = model.first_model.feat


            loss1 = criterion1(outputs[0],label)  
            
            if args.gaze_cbm:

                loss3 = lam2*criterion3(outputs[1],torch.vstack(ego).to(dtype=torch.float).permute((1,0)).to(device))
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),gaze.cuda())
                loss = loss1 + loss2 + loss3
                predicted_gaze = torch.argmax(outputs[2],dim=1)
                all_preds_gaze.append(predicted_gaze.cpu())
                all_labels_gaze.append(gaze.cpu())          
                predicted_ego = (torch.sigmoid(outputs[1]) > 0.5).float().cpu()
                all_preds_ego.append(predicted_ego)
                all_labels_ego.append(torch.vstack(ego).to(dtype=torch.float).permute((1,0)).cpu())

            elif args.ego_cbm:
                    
                #loss3 = lam2*criterion3(outputs[1],gaze.cuda())
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((1,0)).to(device))
                loss = loss1 + loss2 #+ loss3
                predicted_gaze = torch.argmax(outputs[1],dim=1)
                all_preds_gaze.append(predicted_gaze.cpu())
                all_labels_gaze.append(gaze.cpu())    
                predicted_ego = (torch.sigmoid(torch.hstack(outputs[2:])) > 0.5).float().cpu()
                all_preds_ego.append(predicted_ego)
            #import pdb;pdb.set_trace()
                all_labels_ego.append(torch.vstack(ego).to(dtype=torch.float).permute((1,0)).cpu())


            elif args.combined_bottleneck:
                #import pdb;pdb.set_trace()
                loss2 = lam1*criterion2(torch.hstack(outputs[1:16]),gaze.cuda())
                loss3 = lam2*criterion3(torch.hstack(outputs[16:33]),torch.vstack(ego).to(dtype=torch.float).permute((1,0)).to(device))
                loss = loss1 + loss2 + loss3
                predicted_gaze = torch.argmax(torch.hstack(outputs[1:16]),dim=1)
                all_preds_gaze.append(predicted_gaze.cpu())
                all_labels_gaze.append(gaze.cpu())          
                predicted_ego = (torch.sigmoid(torch.hstack(outputs[16:33])) > 0.5).float().cpu()
                all_preds_ego.append(predicted_ego)
                all_labels_ego.append(torch.vstack(ego).to(dtype=torch.float).permute((1,0)).cpu())    

            elif args.multitask:

                loss3 = lam2*criterion3(outputs[1],gaze.cuda())
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((1,0)).to(device))

                loss = loss1 + loss2 + loss3
                #import pdb;pdb.set_trace()
                predicted_gaze = torch.argmax(outputs[1],dim=1)
                all_preds_gaze.append(predicted_gaze.cpu())
                all_labels_gaze.append(gaze.cpu()) 
                #import pdb;pdb.set_trace()
                predicted_ego = (torch.sigmoid(outputs[2]) > 0.5).float().cpu()
                all_preds_ego.append(predicted_ego)
                all_labels_ego.append(torch.vstack(ego).to(dtype=torch.float).permute((1,0)).cpu())

            else:

                loss = loss1

            val_loss_running+=loss

            predicted = torch.argmax(outputs[0],dim=1)
            

            all_preds.append(predicted.cpu())
            all_labels.append(label.cpu())

            del img1
            del img2 
            del label
            del outputs
    
    all_labels = np.hstack(all_labels)
    all_preds = np.hstack(all_preds)

    if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:
        

        all_labels_gaze = np.hstack(all_labels_gaze)
        all_preds_gaze = np.hstack(all_preds_gaze)
        # all_labels_ego = np.hstack(all_labels_ego)
        # all_preds_ego = np.hstack(all_preds_ego)    
        all_labels_ego = torch.cat(all_labels_ego, dim=0)  # Shape (N, 17)
        all_preds_ego = torch.cat(all_preds_ego, dim=0)    # Shape (N, 17)

        return val_loss_running, [all_labels, all_preds, all_labels_gaze, all_preds_gaze, all_labels_ego, all_preds_ego]
    else:
        return val_loss_running, [all_labels, all_preds]