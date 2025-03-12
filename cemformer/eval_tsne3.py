import torch 
import torch.nn as nn 
import numpy as np 

from utils.plot_confusion import confusion
import argparse
import torch.optim as optim
import torchvision
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from captum.attr import Occlusion,IntegratedGradients

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, f1_score
import pickle


from utils.tsne_v2 import plot_tsne as TSNE
from utils.plot_confusion import confusion
from utils.DIPX_350 import CustomDataset
from model import build_model


def forward_func(*x,model):
    # return the first element in the tuple
    return model(*x)[0]



def Eval(args, valid_subset ):


    log_dir = f"runs_{args.model}_{args.dataset}_{args.technique}/dipx"  # Separate log directory for each fold
    writer = SummaryWriter(log_dir)   


    # Initialize model, criterion, and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(args)

    if args.grad_cam:
        ig = IntegratedGradients(forward_func)
    else:
        ig= None
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    #import pdb;pdb.set_trace()
    model.to(device)

    #checkpoint = "weights/dino_vitbase16_pretrain.pth"
    ckp = torch.load('/scratch/mukilv2/cemformer/weights/best_cbm_dipx.pth',map_location=device)
               
    model.load_state_dict(ckp,strict=False)

    val_loader = torch.utils.data.DataLoader(valid_subset, batch_size=args.batch)

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

    if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:
        acc,f1,acc_gaze,f1_gaze,acc_ego,f1_ego = evaluate(args, val_loader, model, criterion1, criterion2, criterion3, device,writer, ig)
        print("Average Accuracy",acc)
        print("Average F1",f1)
        print("Average Accuracy(Gaze)",acc_gaze)
        print("Average F1(Gaze)",f1_gaze)
        print("Average Accuracy(Ego)",acc_ego)
        print("Average F1(Ego)",f1_ego)
    else:
        accuracy, f1 = evaluate(args, val_loader, model, criterion1, criterion2, criterion3, device, writer, ig)
        print("Average Accuracy",accuracy)
        print("Average F1",f1)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")


def evaluate(args, valid_dataloader, model, criterion1, criterion2, criterion3, device,writer,cap):
    model.eval()
    T=1

    lam1,lam2 = 0.5,0.5

    print("Started Evaluating")
    model.eval()

    all_preds = []
    all_labels = []
    all_preds_gaze = []
    all_labels_gaze = []
    all_preds_ego = []
    all_labels_ego = []

    val_loss_running=0
    final_flag = False
    anchor_FEAT= []
    # anchor_LABEL = (np.ones(17)*7).astype(np.uint8).tolist()
    # anchor_LABEL = [torch.tensor([num]) for num in anchor_LABEL]
    anchor_LABEL = []
    final_FEAT=[]
    for index in range(17):

        FEAT=[]
        LABEL=[]
        with torch.no_grad():
            
            for i, (img1,img2,cls,gaze,ego) in tqdm(enumerate(valid_dataloader)): 
                
                
                    
                
                img1 = img1.to(device)
                img2 = img2.to(device)

                label = cls.to(device)

                # Forward pass

                img1=img1.type(torch.cuda.FloatTensor)
                img2=img2.type(torch.cuda.FloatTensor)
                outputs = model(img1,img2) 
                pred_ego =  (torch.sigmoid(torch.stack(outputs[2:]).squeeze()) > 0.2).float().cpu()
                
                #import pdb;pdb.set_trace()
                if pred_ego[index] == 1:
                    feat = model.first_model.feat
                
                    if args.grad_cam:
                        attributions_ig  = cap.attribute(img1, target=label, n_steps=200)

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
                            
                        #loss3 = lam2*criterion3(outputs[1],gaze.cuda())
                        loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).to(device))
                        loss = loss1 + loss2 #+ loss3
                        predicted_gaze = torch.argmax(outputs[1],dim=1)
                        all_preds_gaze.append(predicted_gaze.cpu())
                        all_labels_gaze.append(gaze.cpu())    
                        predicted_ego = (torch.sigmoid(torch.hstack(outputs[2:])) > 0.5).float().cpu()
                        all_preds_ego.append(predicted_ego)
                    #import pdb;pdb.set_trace()
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

                    else:

                        loss = loss1

                    val_loss_running+=loss

                    predicted = torch.argmax(outputs[0],dim=1)
                    

                    all_preds.append(predicted.cpu())
                    all_labels.append(label.cpu())

                    FEAT.append(feat.cpu())
                    LABEL.append(label.cpu())

                    del img1
                    del img2 
                    del label
                    del outputs
                
        if len(FEAT) != 0:
            anchor_FEAT.append(torch.vstack(FEAT).mean(dim=0).unsqueeze(0))
            anchor_LABEL.append(torch.tensor(8))
################################

    FEAT=[]
    LABEL=[]
    with torch.no_grad():
        
        for i, (img1,img2,cls,gaze,ego) in tqdm(enumerate(valid_dataloader)): 
            
           
            #import pdb;pdb.set_trace()
            img1 = img1.to(device)
            img2 = img2.to(device)

            label = cls.to(device)

            # Forward pass

            img1=img1.type(torch.cuda.FloatTensor)
            img2=img2.type(torch.cuda.FloatTensor)
            outputs = model(img1,img2) 
            
            feat = model.first_model.feat
        
            if args.grad_cam:
                attributions_ig  = cap.attribute(img1, target=label, n_steps=200)

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
                    
                #loss3 = lam2*criterion3(outputs[1],gaze.cuda())
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.hstack(ego).to(dtype=torch.float).unsqueeze(0).to(device))
                loss = loss1 + loss2 #+ loss3
                predicted_gaze = torch.argmax(outputs[1],dim=1)
                all_preds_gaze.append(predicted_gaze.cpu())
                all_labels_gaze.append(gaze.cpu())    
                predicted_ego = (torch.sigmoid(torch.hstack(outputs[2:])) > 0.5).float().cpu()
                all_preds_ego.append(predicted_ego)
            #import pdb;pdb.set_trace()
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

            else:

                loss = loss1

            val_loss_running+=loss

            predicted = torch.argmax(outputs[0],dim=1)
            

            all_preds.append(predicted.cpu())
            all_labels.append(label.cpu())

            FEAT.append(feat.cpu())
            LABEL.append(label.cpu())

            del img1
            del img2 
            del label
            del outputs

################################

    final_FEAT = FEAT + anchor_FEAT
    LABEL = LABEL + anchor_LABEL

    with open("data_baseline.pkl", "wb") as f:
        pickle.dump({"embeddings": final_FEAT, "labels": LABEL}, f)

    tsne = TSNE()
    tsne_img = tsne.plot(final_FEAT,LABEL,args.dataset)

    plt.savefig('tsne_all_pred_baseline_2.png')
    
    all_labels = np.hstack(all_labels)
    all_preds = np.hstack(all_preds)

    if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:
        #import pdb;pdb.set_trace()

        all_labels_gaze = np.hstack(all_labels_gaze)
        all_preds_gaze = np.hstack(all_preds_gaze)
        all_labels_ego = torch.cat(all_labels_ego, dim=0)  # Shape (N, 17)
        all_preds_ego = torch.cat(all_preds_ego, dim=0)    # Shape (N, 17)
        #confusion(all_labels_gaze, all_preds_gaze,'gaze',writer,1)

    # for Action Classification 
    confusion(all_labels, all_preds,'action',writer,1)
    

    writer.add_figure('TSNE', tsne_img)

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

        writer.add_scalar("Accuracy/Validation(Gaze)", accuracy_val_gaze)
        writer.add_scalar("F1/Validation(Gaze)", f1_val_gaze)
        writer.add_scalar("Accuracy/Validation(Ego)", accuracy_val_ego)
        writer.add_scalar("F1/Validation(Ego)", f1_val_ego)

    print("accuracy and F1",accuracy_val,f1_val) 

    del all_preds_gaze, all_labels_gaze, all_preds_ego, all_labels_ego


    writer.add_scalar("Loss/Validation", val_loss)
    writer.add_scalar("Accuracy/Validation", accuracy_val)

    writer.add_scalar("F1/Validation", f1_val)


    if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:
        return accuracy_val,f1_val,accuracy_val_gaze,f1_val_gaze,accuracy_val_ego,f1_val_ego
    else:

        return accuracy_val,f1_val


if __name__ == '__main__':
    torch.manual_seed(0)
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
    parser.add_argument("--clusters", type = int, default= 5)
    parser.add_argument("-bottleneck",  action="store_true", help="Enable bottleneck mode")
    parser.add_argument("-gaze_cbm", action="store_true", help="Enable gaze CBM mode")
    parser.add_argument("-ego_cbm", action="store_true", help="Enable ego CBM mode")
    parser.add_argument("-multitask", action="store_true", help="Enable multitask mode")
    parser.add_argument("-grad_cam", action="store_true", help="enable grad cam")
    parser.add_argument("-combined_bottleneck", action="store_true", help="Enable combined_bottleneck mode")
    args = parser.parse_args()
    home_dir = str(args.directory)
    cache_dir = os.path.join(home_dir, "mukilv2")
    world_size = torch.cuda.device_count()

    val_csv = "/scratch/mukilv2/dipx/val.csv"
    

    val_subset = CustomDataset(val_csv, debug=args.debug)
    Eval(args, val_subset)