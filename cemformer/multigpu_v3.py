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

from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.transforms.functional import to_pil_image, to_grayscale
from torch.utils.data import Dataset, DataLoader, random_split

from utils.tsne import plot_tsne as TSNE
from utils.plot_confusion import confusion
from utils.DIPX import CustomDataset
from model import build_model

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from collections import Counter
import videotransforms

import torch.multiprocessing as mp 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
def ddp_setup(args, rank,world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args.port)
    init_process_group(backend="nccl", rank=rank, world_size = world_size)

def cross_validate_model(rank, world_size, args, dataset, n_splits=5):
    

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []
      
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        ddp_setup(args, rank, world_size)
        print(f"Fold {fold + 1}/{n_splits}")
        
        log_dir = f"runs_{args.model}_DIPX_{args.technique}/fold_brain_{fold}"  # Separate log directory for each fold
        writer = SummaryWriter(log_dir)   

       # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        device = rank
        model = build_model(args)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        model.to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        

        #import pdb;pdb.set_trace()
        # Creating data loaders for training and validation
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch,pin_memory=True, shuffle= False, sampler = DistributedSampler(train_subset),drop_last=True)
        #val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch,pin_memory=True, shuffle= False, sampler = DistributedSampler(val_subset),drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch)

        if args.distributed:
            for param in model.module.parameters():
                param.requires_grad = False
            for layer in model.module.first_model.model1.videomae.encoder.layer:
                for param in layer.attention.parameters():
                    param.requires_grad = True
            for layer in model.module.first_model.model2.videomae.encoder.layer:
                for param in layer.attention.parameters():
                    param.requires_grad = True
            for layer in model.module.sec_model.parameters():
                param.requires_grad = True
        else:

            for param in model.parameters():
                param.requires_grad = False
            for layer in model.first_model.model1.videomae.encoder.layer:
                for param in layer.attention.parameters():
                    param.requires_grad = True
            for layer in model.first_model.model2.videomae.encoder.layer:
                for param in layer.attention.parameters():
                    param.requires_grad = True
            for layer in model.sec_model.parameters():
                param.requires_grad = True
            
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

        if args.gaze_cbm or args.combined_bottleneck:
            #criterion1 = nn.CrossEntropyLoss() 
            criterion1 = torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='FocalLoss',
                alpha=torch.tensor([.05, .104, .071, .122, .099, .098, .449]),
                gamma=2,
                force_reload=False
            ).to(device)# action classificaiton 
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
        # learning_rate = 0.001
        # momentum = 0.9
        # weight_decay = 0.001

        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        ####
        base_learning_rate = args.learning_rate
        weight_decay = 0.05
        if args.distributed:
            optimizer = optim.AdamW(model.module.parameters(), lr=base_learning_rate, weight_decay=weight_decay)
        else:

            optimizer = optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay)

        # Train and evaluate the model
        accuracy, f1 = train(args, train_loader, val_loader, model, criterion1, criterion2, criterion3, optimizer, device,writer,fold)
        accuracies.append(accuracy)
        f1_scores.append(f1)

        destroy_process_group()
        
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

    T=1
    num_epochs=100
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.00005, total_iters=100)

    train_losses, valid_accuracy = [], []
    print("Started Training")
    if args.debug:
        patience =1
    else:
        patience =1000
    #patience = 10 
    min_delta = 0.0001  
    best_acc = 0 
    counter = 0 
    lam1,lam2 = 0.5,0.5

    if args.technique: 
        save_dir = f"best_{args.model}_{args.dataset}_{args.technique}_dir"

    else:
        save_dir = f"best_{fold}_{args.model}_{args.dataset}_dir"
        
    patience_counter = 0
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    best_model_path = os.path.join(save_dir, f"best_{fold}_{args.model}_{args.dataset}.pth") 
    print("Started Training")
    for epoch in range(num_epochs):
        model.to(device)
        all_preds = []
        all_labels = []
        all_preds_local=[]
        all_labels_local=[]

        train_loss = 0.0
        for i, (img1,img2,clas,gaze,ego) in tqdm(enumerate(train_dataloader)):
            

            img1 = img1.to(device)
            img2 = img2.to(device)

            label = clas.to(device)

            # Forward pass

            img1=img1.type(torch.cuda.FloatTensor)
            img2=img2.type(torch.cuda.FloatTensor)
            #import pdb;pdb.set_trace()
            inputs1 = {"pixel_values": img1.permute((0,2,1,-2,-1)),"labels":label}
            inputs2 = {"pixel_values": img2.permute((0,2,1,-2,-1)),"labels":label}
            inputs1 = {k: v for k, v in inputs1.items()}
            inputs2 = {k: v for k, v in inputs2.items()}

            outputs = model(inputs1,inputs2)
            #import pdb;pdb.set_trace()
            if args.distributed:

                feat = model.module.first_model.feat
            else:
                feat = model.first_model.feat

            #import pdb;pdb.set_trace()
            loss1 = criterion1(outputs[0],label)  
            #import pdb;pdb.set_trace()
            if args.gaze_cbm:

                loss3 = lam2*criterion3(outputs[1],torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),gaze.cuda().to(device))
                loss = loss1 + loss2 + loss3
            
            elif args.ego_cbm:
                    
                loss3 = lam2*criterion3(outputs[1],gaze.cuda().to(device))
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))
                loss = loss1 + loss2 + loss3

            elif args.multitask:

                loss3 = lam2*criterion3(outputs[1],gaze.cuda().to(device))
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))

                loss = loss1 + loss2 + loss3
            
            elif args.combined_bottleneck:
                #import pdb;pdb.set_trace()
                loss2 = lam1*criterion2(torch.hstack(outputs[1:16]),gaze.cuda().to(device))
                loss3 = lam2*criterion3(torch.hstack(outputs[16:33]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))
                loss = loss1 + loss2 + loss3
            else:

                loss = loss1
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
            # all_preds.append(predicted.cpu())
            # all_labels.append(label.cpu())
            all_preds.append(predicted.to(device))
            all_labels.append(label.to(device))
            scheduler.step()

       
        # running_loss = torch.tensor([train_loss], device=device)         
        # # if torch.cuda.is_available():
        # #     dist.reduce(running_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
        #     #reduce(running_corrects, dst=0, op=torch.distributed.ReduceOp.SUM)
        running_loss = torch.tensor([train_loss], device=device) 
        print("training over loss is calculated ", running_loss)
        if dist.get_rank() == 0:
            epoch_loss = running_loss.item()/len(train_dataloader)
            print(f" epoch {epoch} Loss: {epoch_loss:.4f}")



        all_preds_local = torch.cat(all_preds).to(device)
        all_labels_local = torch.cat(all_labels).to(device)


        all_preds_gathered = [torch.zeros_like(all_preds_local) for _ in range(dist.get_world_size())]
        all_labels_gathered = [torch.zeros_like(all_labels_local) for _ in range(dist.get_world_size())]
        dist.all_gather(all_preds_gathered, all_preds_local)
        dist.all_gather(all_labels_gathered, all_labels_local)


        if dist.get_rank() == 0:
            all_preds = torch.cat(all_preds_gathered).cpu().numpy()
            all_labels = torch.cat(all_labels_gathered).cpu().numpy()

            accuracy_train = accuracy_score(all_labels, all_preds)
            f1_train = f1_score(all_labels, all_preds, average='weighted')

            print(f"Accuracy (Training): {accuracy_train:.4f}, F1 Score: {f1_train:.4f}")

        ### EVAL 
        dist.barrier()
        if dist.get_rank() == 0:
            single_gpu_device = torch.device("cuda:0")
            model.to(single_gpu_device)
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
            with torch.no_grad():

                for i, (img1,img2,clas,gaze,ego) in tqdm(enumerate(valid_dataloader)): 

                    img1 = img1.to(single_gpu_device)
                    img2 = img2.to(single_gpu_device)

                    label = clas.to(single_gpu_device)

                    # Forward pass

                    img1=img1.type(torch.cuda.FloatTensor)
                    img2=img2.type(torch.cuda.FloatTensor)

                    inputs1 = {"pixel_values": img1.permute((0,2,1,-2,-1)),"labels":label}
                    inputs2 = {"pixel_values": img2.permute((0,2,1,-2,-1)),"labels":label}
                    inputs1 = {k: v for k, v in inputs1.items()}
                    inputs2 = {k: v for k, v in inputs2.items()}

                    outputs = model(inputs1,inputs2)
                    #import pdb;pdb.set_trace()
                    feat = model.module.first_model.feat


                    loss1 = criterion1(outputs[0],label)  
                    
                    if args.gaze_cbm:

                        loss3 = lam2*criterion3(outputs[1],torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).to(single_gpu_device))
                        loss2 = lam1*criterion2(torch.hstack(outputs[2:]),gaze.to(single_gpu_device))
                        loss = loss1 + loss2 + loss3
                        predicted_gaze = torch.argmax(outputs[2],dim=1)
                        all_preds_gaze.append(predicted_gaze.cpu())
                        all_labels_gaze.append(gaze.cpu())          
                        predicted_ego = (torch.sigmoid(outputs[1]) > 0.5).float().cpu()
                        all_preds_ego.append(predicted_ego)
                        all_labels_ego.append(torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).cpu())

                    elif args.ego_cbm:
                            
                        loss3 = lam2*criterion3(outputs[1],gaze.to(single_gpu_device))
                        loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(single_gpu_device))
                        loss = loss1 + loss2 + loss3
                        predicted_gaze = torch.argmax(outputs[1],dim=1)
                        all_preds_gaze.append(predicted_gaze.cpu())
                        all_labels_gaze.append(gaze.cpu())    
                        predicted_ego = (torch.sigmoid(torch.hstack(outputs[2:])) > 0.5).float().cpu()
                        all_preds_ego.append(predicted_ego)
                    #import pdb;pdb.set_trace()
                        all_labels_ego.append(torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).cpu())

                    elif args.multitask:

                        loss3 = lam2*criterion3(outputs[1],gaze.to(single_gpu_device))
                        loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(single_gpu_device))

                        loss = loss1 + loss2 + loss3
                        #import pdb;pdb.set_trace()
                        predicted_gaze = torch.argmax(outputs[1],dim=1)
                        all_preds_gaze.append(predicted_gaze.cpu())
                        all_labels_gaze.append(gaze.cpu()) 
                        #import pdb;pdb.set_trace()
                        predicted_ego = (torch.sigmoid(outputs[2]) > 0.5).float().cpu()
                        all_preds_ego.append(predicted_ego)
                        all_labels_ego.append(torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).cpu())

                    elif args.combined_bottleneck:
                        #import pdb;pdb.set_trace()
                        loss2 = lam1*criterion2(torch.hstack(outputs[1:16]),gaze.to(single_gpu_device))
                        loss3 = lam2*criterion3(torch.hstack(outputs[16:33]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(single_gpu_device))
                        loss = loss1 + loss2 + loss3
                        predicted_gaze = torch.argmax(torch.hstack(outputs[1:16]),dim=1)
                        all_preds_gaze.append(predicted_gaze.cpu())
                        all_labels_gaze.append(gaze.cpu())          
                        predicted_ego = (torch.sigmoid(torch.hstack(outputs[16:33])) > 0.5).float().cpu()
                        all_preds_ego.append(predicted_ego)
                        all_labels_ego.append(torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).cpu())    

                    else:

                        loss = loss1

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
                print("accuracy and F1(EGO)",accuracy_val_ego,f1_val_ego) 

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
    parser.add_argument("--learning_rate",  type = float, default = 0.001)
    parser.add_argument("--weight_decay",  type = float, default = 0.001)
    parser.add_argument("--model",  type = str, default = None)
    parser.add_argument("--debug",  type = str, default = None)
    parser.add_argument("--technique",  type = str, default = None)
    parser.add_argument("--num_classes",  type = int, default = 5)
    parser.add_argument("--batch",  type = int, default = 1)
    parser.add_argument("--port",  type = int, default = 12345)
    parser.add_argument("-distributed",  action ="store_true")
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
    world_size = torch.cuda.device_count()
    dataset = CustomDataset(debug = args.debug)
    mp.spawn(cross_validate_model, args= [world_size, args,dataset], nprocs = world_size)
    #cross_validate_model(rank, world_size, args,dataset)

