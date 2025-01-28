import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt

import argparse
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, AutoConfig, VideoMAEConfig
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
from utils.loss import cc_loss
from utils.Brain4Cars import CustomDataset
from model import build_model


def cross_validate_model(args, dataset, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        log_dir = f"runs_MAE/fold_brain_{fold}"  # Separate log directory for each fold
        writer = SummaryWriter(log_dir)   


        # Initialize model, criterion, and optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = build_model(args)

        model.to(device)
            
        for param in model.parameters():
            param.requires_grad = False
        for layer in model.model1.videomae.encoder.layer:
            for param in layer.attention.parameters():
                param.requires_grad = True
        for layer in model.model2.videomae.encoder.layer:
            for param in layer.attention.parameters():
                param.requires_grad = True
        for layer in model.classifier.parameters():
            param.requires_grad = True

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch)

        criterion = torch.nn.CrossEntropyLoss()
        
        ### TESTING SGD 
        # learning_rate = 0.01
        # momentum = 0.9
        # weight_decay = 0.001

        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        ### Testing Adam 
        base_learning_rate = 5e-5
        weight_decay = 0.05
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_learning_rate, weight_decay=weight_decay)

        ####
        accuracy, f1 = train(args, train_loader, val_loader, model, criterion, optimizer, device,writer)
        accuracies.append(accuracy)
        f1_scores.append(f1)

    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_f1_score = np.mean(f1_scores)
    std_f1_score = np.std(f1_scores)

    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1_score:.4f} ± {std_f1_score:.4f}")
    return avg_accuracy, std_accuracy, avg_f1_score, std_f1_score

def cc_loss(context,output_logits):
        A ={'0':[[1,0,0],[1,0,1],[0,0,0],[0,1,0],[1,1,0]],
                '1':[[0,1,0],[1,1,0],[0,1,1],[1,1,1]],
                '2':[[0,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,0]],
                '3':[[1,0,0],[1,0,1],[1,1,0],[1,1,1]],
                '4':None}
        
        #c = tuple(map(int,c.split(',')))
        #import pdb;pdb.set_trace() 
        output_logits = torch.nn.functional.softmax(output_logits, dim=1)
        
        cc_loss=0 
        count=0
        for c in context:
            ctx=np.zeros(3)
            if c[0].item() == 1:
                ctx[0]=1
            if c[0].item() == c[1].item():
                ctx[1]=1
            if c[2].item() == 1:
                ctx[2]=1

            
            for i in A:
                if not int(i) == 4: 
                    if ctx.tolist() in A[i]:

                        cc_loss+= -torch.log(1-output_logits[count][int(i)])
            count=count+1
            
        return cc_loss


def train(args, train_dataloader, valid_dataloader,model,criterion, optimizer, device,writer):
    model.train()


    #warmup_scheduler = LambdaLR(optimizer, warmup_linear)

    # Iterate through the dataloader
    T=1
    num_epochs=100
    #cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.00005, total_iters=100)
    train_losses, valid_accuracy = [], []
    print("Started Training")
    patience = 5 
    min_delta = 0.0001  
    best_acc = 0 
    counter = 0 
    save_dir = "best_model_dir"

    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    best_model_path = os.path.join(save_dir, f"best_mae.pth") 
    print("Started Training")
    for epoch in range(num_epochs):
        all_preds = []
        all_labels = []
        train_loss = 0.0
        for i, (img1,img2,cls,context) in tqdm(enumerate(train_dataloader)):


            b,_,_,_,_=img1.shape
           # import pdb;pdb.set_trace()
            img1 = img1.to(device)
            img2 = img2.to(device)

            label = cls.to(device)

            # Forward pass
            img1=img1.type(torch.cuda.FloatTensor)
            img2=img2.type(torch.cuda.FloatTensor)
            #import pdb;pdb.set_trace()

            inputs1 = {"pixel_values": img1.permute((0,2,1,-2,-1)),"labels":label}
            inputs2 = {"pixel_values": img2.permute((0,2,1,-2,-1)),"labels":label}
            inputs1 = {k: v for k, v in inputs1.items()}
            inputs2 = {k: v for k, v in inputs2.items()}

      
            outputs = model(inputs1,inputs2)
          
        

            loss = criterion(outputs, label)
            context = [list(x) for x in zip(*context)]
            ccloss = cc_loss(context,outputs)
            loss = loss+ccloss

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

            
            predicted = torch.argmax(outputs,dim=1)
            all_preds.append(predicted.cpu())
            all_labels.append(label.cpu())

            scheduler.step()
            ##print("iteration loss ",loss_val)#,torch.nn.functional.softmax(outputs,dim=0))

        epoch_loss = train_loss/len(train_dataloader)

        print(f"epoch {epoch} Loss: {epoch_loss:.4f}")

        train_losses.append(epoch_loss)
        #import pdb;pdb.set_trace()
        all_preds = np.hstack(all_preds)
        all_labels = np.hstack(all_labels)



        accuracy_train = accuracy_score(all_labels, all_preds)
        f1_train = f1_score(all_labels, all_preds, average='weighted')

        print(f"Accuracy(Trainign): {accuracy_train:.4f}, F1 Score: {f1_train:.4f}")

        if  accuracy_train - best_acc  > min_delta:
            best_acc = accuracy_train
            counter = 0  
            if args.distributed:
                torch.save(model.module.state_dict(),best_model_path)
            else:
                torch.save(model.state_dict(),best_model_path)
            print("best model is saved ")  

    # cleanup() 
        ### EVAL 
        print("Started Evaluating")
        model.eval()
        all_preds = []
        all_labels = []
        val_loss_running=0
        FEAT=[]
        LABEL=[]
        with torch.no_grad():

            for i, (img1,img2,cls,context) in tqdm(enumerate(valid_dataloader)): 

                b,_,_,_,_=img1.shape
                
                img1 = img1.to(device)
                img2 = img2.to(device)

                #cls= cls.type(torch.LongTensor)
                label = cls.to(device)

                # Forward pass
                #import pdb;pdb.set_trace()
                img1=img1.type(torch.cuda.FloatTensor)
                img2=img2.type(torch.cuda.FloatTensor)
                
                inputs1 = {"pixel_values": img1.permute((0,2,1,-2,-1)),"labels":label}
                inputs2 = {"pixel_values": img2.permute((0,2,1,-2,-1)),"labels":label}
                inputs1 = {k: v for k, v in inputs1.items()}
                inputs2 = {k: v for k, v in inputs2.items()}

        
                outputs = model(inputs1,inputs2)
          

                loss = criterion(outputs, label)
                context = [list(x) for x in zip(*context)]
                ccloss = cc_loss(context,outputs)

                loss = loss+ccloss                
                val_loss_running+=loss

                predicted = torch.argmax(outputs,dim=1)

                all_preds.append(predicted.cpu())
                all_labels.append(label.cpu())
                # FEAT.append(feat.cpu())
                # LABEL.append(label.cpu())
        #import pdb;pdb.set_trace()
        all_labels = np.hstack(all_labels)
        all_preds = np.hstack(all_preds)
        # tsne = TSNE()
        # tsne_img = tsne.plot(FEAT,LABEL)
        
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap='Blues')

        # Add color bar
        fig.colorbar(cax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix (Brain)')

        # Annotate the cells with the numeric values
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{val}', ha='center', va='center', color='white')

        # Log the confusion matrix as an image in TensorBoard
        writer.add_figure('Confusion Matrix (Brain)', fig,epoch)
        #writer.add_figure('TSNE', tsne_img,epoch)
        val_loss = val_loss_running/len(valid_dataloader)
        # Calculate accuracy and F1 score
        
        
        accuracy_val = accuracy_score(all_labels, all_preds)
        f1_val = f1_score(all_labels, all_preds, average='weighted') #'weighted' or 'macro' s
        print("accuracy and F1",accuracy_val,f1_val) 
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy_val, epoch)
        writer.add_scalar("Accuracy/Train", accuracy_train, epoch)
        
    return accuracy_val, f1_val
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--directory", help="Directory for home_dir", default = os.path.expanduser('~'))
    parser.add_argument("-e", "--epochs", type = int, help="Number of epochs", default = 1)
    parser.add_argument("--mem_per_layer", type = int, help="Number of memory tokens", default = 3)
    parser.add_argument("--model",  type = str, default = None)
    parser.add_argument("--debug",  type = str, default = None)
    parser.add_argument("--num_classes",  type = int, default = 5)
    parser.add_argument("--batch",  type = int, default = 1)
    parser.add_argument("--distributed",  type = bool, default = False)
    args = parser.parse_args()

    home_dir = str(args.directory)
    cache_dir = os.path.join(home_dir, "mukil")

    dataset = CustomDataset(debug = args.debug)
    cross_validate_model(args,dataset)

