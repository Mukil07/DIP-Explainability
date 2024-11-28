import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt

import argparse

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.transforms.functional import to_pil_image, to_grayscale
from torch.utils.data import Dataset, DataLoader, random_split

import pdb
from tqdm.auto import tqdm
import os
from copy import deepcopy

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from cc_loss import Custom_criterion
from utils.Custom_dataloader_DIPX import CustomDataset
from model import build_model

def setup():
    # Retrieve rank and world size from environment variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Initialize the process group for distributed training
    if not torch.distributed.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    return rank, world_size
def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def cross_validate_model(args, dataset, n_splits=2):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # rank, world_size = setup()
    # torch.cuda.set_device(rank)
    # device = torch.device("cuda", rank)
    rank, world_size = setup()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    accuracies = []
    f1_scores = []
    accuracies_gaze = []
    f1_scores_gaze = []
    accuracies_ego = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)): #1294/5 

        print(f"Fold {fold + 1}/{n_splits}")
        log_dir = f"runs/fold_dipx_{fold}"  # Separate log directory for each fold
        writer = SummaryWriter(log_dir)   
        model = build_model(args)
        model.to(device)
        model = DDP(model, device_ids=[rank],find_unused_parameters=True) 


        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        sampler_train = DistributedSampler(train_subset, shuffle=True)
        sampler_val = DistributedSampler(val_subset, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch, sampler= sampler_train)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch, sampler=sampler_val)

        print(model)
        checkpoint = "/scratch/mukil/cemformer/weights/dino_vitbase16_pretrain.pth"
        ckp = torch.load(checkpoint)

        # del ckp['mlp_head.0.bias']
        # del ckp['mlp_head.1.bias']
        # del ckp['mlp_head.0.weight']
        # del ckp['mlp_head.1.weight']
        del ckp['pos_embed']
        new_length = 393 
        num_channels = 768  
        model.module.load_state_dict(ckp,strict=False)
      

        class_weights = torch.as_tensor([0.2,0.6,1.2,1.5,1.5,1.7,2], dtype=torch.float32).to(device)
        criterion=torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion_gaze =torch.nn.CrossEntropyLoss()
        criterion_ego = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.module.parameters(), lr=0.00005, weight_decay=5e-2)

        # Train and evaluate the model
        accuracy, f1,accuracy_gaze,f1_gaze = train(args,train_loader, val_loader, model, criterion,criterion_gaze,criterion_ego, optimizer, device,writer)

        accuracies.append(accuracy)
        accuracies_gaze.append(accuracy_gaze)
        #accuracies_ego.append(acc_ego)

        f1_scores.append(f1)
        f1_scores_gaze.append(f1_gaze)
        
        
    # Calculate the average and standard deviation of the metrics
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_f1_score = np.mean(f1_scores)
    std_f1_score = np.std(f1_scores)

    avg_accuracy_g = np.mean(accuracies_gaze)
    std_accuracy_g = np.std(accuracies_gaze)
    avg_f1_score_g = np.mean(f1_scores_gaze)
    std_f1_score_g = np.std(f1_scores_gaze)

    # avg_accuracy_e = np.mean(accuracies_ego)
    # std_accuracy_e = np.std(accuracies_ego)

    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1_score:.4f} ± {std_f1_score:.4f}")

    print(f"Average Accuracy(gaze): {avg_accuracy_g:.4f} ± {std_accuracy_g:.4f}")
    print(f"Average F1 Score(gaze): {avg_f1_score_g:.4f} ± {std_f1_score_g:.4f}")

    
    #print(f"Average Accuracy(gaze): {avg_accuracy_e:.4f} ± {std_accuracy_e:.4f}")
    return model

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


def custom_acc(gt,pred):
    count=1
    for i in range(len(gt)):
        if gt == pred:
            print("ego")
            count+=1
    return count/len(gt)

def train(args, train_dataloader, valid_dataloader,model,criterion,criterion_gaze,criterion_ego, optimizer, device,writer):
    model.train()

    # remove parts 

    for param in model.module.parameters():
        param.requires_grad = False
    #import pdb;pdb.set_trace()
    for block in model.module.vit.blocks:
        for param in block.attn.parameters():
            param.requires_grad = True

    for param in model.module.mlp_head.parameters():
        param.requires_grad = True
    for param in model.module.head_gaze.parameters():
        param.requires_grad = True
    # for param in model.head_ego.parameters():
    #     param.requires_grad = True


    
    T=1
    num_epochs=50

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, valid_accuracy = [], []
    print("Started Training")
    patience = 5 
    min_delta = 0.0001  
    best_acc = 0
    counter = 0 
    save_dir = "best_model_dir_dipx"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    best_model_path = os.path.join(save_dir, f"best_model{args.mem_per_layer}.pth") 


    for epoch in range(num_epochs):
        all_preds = []
        all_labels = []
        train_loss = 0.0
        for i, (img1,img2,cls,gaze,ego) in tqdm(enumerate(train_dataloader)):
            

            gaze = gaze.to(device)
            ego = ego.to(device)

            
            img1 = img1.to(device)
            img2 = img2.to(device)

            #cls= cls.type(torch.LongTensor)
            label = cls.to(device)

            # Forward pass
            #
            img1=img1.type(torch.cuda.FloatTensor)
            img2=img2.type(torch.cuda.FloatTensor)

            outputs,out_gaze,out_ego = model(img1,img2) #  size - (10,4,768)

            #import pdb;pdb.set_trace()
            outputs = outputs.mean(dim=1)
            out_gaze = out_gaze.mean(dim=1)
            out_ego = out_ego.mean(dim=1)
            #import pdb;pdb.set_trace()
            loss = criterion(outputs, label)
            loss_gaze = criterion_gaze(out_gaze,gaze)
            #import pdb;pdb.set_trace()
            #loss_ego = criterion_ego(out_ego,ego.squeeze(0))
            loss = loss+loss_gaze#+loss_ego

            # Uncomment for exponential loss

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

            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            predicted = torch.argmax(outputs)

            all_preds.append(predicted.cpu())
            all_labels.append(label[0].cpu())


            cosine_scheduler.step()
            ##print("iteration loss ",loss_val)#,torch.nn.functional.softmax(outputs,dim=0))
        
        epoch_loss = train_loss/len(train_dataloader)
        dist.all_reduce(torch.as_tensor(epoch_loss).to(device), op=dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            epoch_loss /= dist.get_world_size()  
            print(f"epoch {epoch} Loss: {epoch_loss:.4f}")
            
            # Check for improvement\
        #import pdb;pdb.set_trace()
        all_preds_tesnor = torch.tensor(all_preds).to(device)
        all_labels_tesnor = torch.as_tensor(all_labels).to(device)

        gathered_preds = [torch.zeros_like(all_preds_tesnor) for _ in range(dist.get_world_size())]
        gathered_labels = [torch.zeros_like(all_labels_tesnor) for _ in range(dist.get_world_size())]
        
        dist.all_gather(gathered_preds, all_preds_tesnor)
        dist.all_gather(gathered_labels, all_labels_tesnor)
        
        if dist.get_rank() == 0:
            # Only on rank 0, calculate and print the full accuracy and F1 score
            all_preds_flat = torch.cat(gathered_preds, dim=0).cpu().numpy()
            all_labels_flat = torch.cat(gathered_labels, dim=0).cpu().numpy()

            accuracy_train = accuracy_score(all_labels_flat, all_preds_flat)
            f1_train = f1_score(all_labels_flat, all_preds_flat, average='weighted')

            print(f"Accuracy(Trainign): {accuracy_train:.4f}, F1 Score: {f1_train:.4f}")


        if  accuracy_train - best_acc  > min_delta:
            best_acc = accuracy_train
            counter = 0  
            if dist.get_rank() == 0:
                torch.save(model.module.state_dict(),best_model_path)
                print("best model is saved ")  

    #cleanup() 
        ### EVAL 
        print("Started Evaluating")
        model.eval()
        all_preds = []
        all_labels = []
        all_gaze=[]
        all_gaze_gt=[]
        all_ego=[]
        all_ego_gt=[]
        val_loss = 0.0
        with torch.no_grad():

            for i, (img1,img2,cls,gaze,ego) in tqdm(enumerate(valid_dataloader)): 
            # print(i)
                # img1 size - (1,10,3,224,224)
                #import pdb;pdb.set_trace()
                gaze = gaze.to(device)
                ego = ego.to(device)
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

                outputs,out_gaze,out_ego = model(img1,img2) #  size - (10,4,768)
                

                outputs = outputs.mean(dim=1)
                out_gaze = out_gaze.mean(dim=1)
                out_ego = out_ego.mean(dim=1)

                loss = criterion(outputs, label)
                loss_gaze = criterion_gaze(out_gaze,gaze)
                #import pdb;pdb.set_trace()
                #loss_ego = criterion_ego(out_ego,ego.squeeze(0))
                loss = loss+loss_gaze#+loss_ego
                val_loss+=loss
                # threshold = 0.5
                # pred_ego = probs > threshold
                #import pdb;pdb.set_trace()
                predicted = torch.argmax(outputs)
                pred_gaze = torch.argmax(out_gaze)
                print(predicted,label,pred_gaze)

                all_preds.append(predicted.cpu().numpy())
                all_labels.append(label[0].cpu().numpy())
                all_gaze.append(pred_gaze.cpu().numpy())
                all_gaze_gt.append(gaze[0].cpu().numpy())
        #import pdb;pdb.set_trace()
        cm = confusion_matrix(np.array(all_labels), np.array(all_preds))
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap='Blues')

        # Add color bar
        fig.colorbar(cax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix (DIPX)')

        # Annotate the cells with the numeric values
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{val}', ha='center', va='center', color='white')

        # Log the confusion matrix as an image in TensorBoard
        writer.add_figure('Confusion Matrix (DIPX)', fig)


        loss_val = val_loss/len(valid_dataloader)
        accuracy_val = accuracy_score(all_labels, all_preds)
        accuracy_val_gaze = accuracy_score(all_gaze_gt, all_gaze)
        #acc_ego = custom_acc(all_ego_gt,all_ego)

        f1_val = f1_score(all_labels, all_preds, average='weighted') #'weighted' or 'macro' s
        f1_val_gaze = f1_score(all_gaze_gt, all_gaze, average='weighted') 
        print("accuracy and F1",accuracy_val,f1_val) 
        print("accuracy and F1(gaze)",accuracy_val_gaze,f1_val_gaze) 
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Loss/Validation", loss_val, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy_val, epoch)
        writer.add_scalar("Accuracy/Train", accuracy_train, epoch)
        writer.add_scalar("Accuracy/Gaze", accuracy_val_gaze, epoch)
        writer.add_scalar("F1/Gaze", f1_val_gaze, epoch)
        writer.add_scalar("F1/Validation", f1_val, epoch)
        writer.add_scalar("F1/Train", f1_train, epoch)
    #print("accuracy of ego expl",acc_ego)
    return accuracy_val, f1_val, accuracy_val_gaze, f1_val_gaze#,acc_ego
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--directory", help="Directory for home_dir", default = os.path.expanduser('~'))
    parser.add_argument("-e", "--epochs", type = int, help="Number of epochs", default = 1)
    parser.add_argument("--mem_per_layer", type = int, help="Number of memory tokens", default = 3)
    parser.add_argument("--model",  type = str, default = None)
    parser.add_argument("--num_classes",  type = int, default = 5)
    parser.add_argument("--debug",  type = str, default = None)
    parser.add_argument("--batch",  type = int, default = 1)
    
    args = parser.parse_args()

    home_dir = str(args.directory)
    cache_dir = os.path.join(home_dir, "mukil")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = CustomDataset(debug = args.debug, transform=transform)

    n_splits = 5

    model = cross_validate_model(args, dataset, n_splits)
    torch.distributed.barrier()  # Synchronize before cleanup
    cleanup()

    #create models directory if not exists
    if not os.path.exists("models"):
        os.makedirs("models")
        
    torch.save(model.state_dict(), f"models/cemformer_DIPX_mem_{args.mem_per_layer}.pt")


        # all_preds.append(predicted.cpu())
        # all_labels.append(label[0].cpu())
        # all_gaze.append(pred_gaze.cpu())
        # all_gaze_gt.append(gaze[0].cpu())
        # all_ego.append(pred_ego.cpu())
        # all_ego_gt.append(ego.cpu())

# Calculate accuracy and F1 score

# all_preds_tesnor = torch.tensor(all_preds).to(device)
# all_labels_tesnor = torch.as_tensor(all_labels).to(device)
# all_gaze_tesnor = torch.tensor(all_gaze).to(device)
# all_gaze_gt_tesnor = torch.as_tensor(all_gaze_gt).to(device)
# # all_ego_tesnor = torch.tensor(all_ego).to(device)
# # all_ego_gt_tesnor = torch.as_tensor(all_ego_gt).to(device)

# gathered_preds = [torch.zeros_like(all_preds_tesnor) for _ in range(dist.get_world_size())]
# gathered_labels = [torch.zeros_like(all_labels_tesnor) for _ in range(dist.get_world_size())]
# gathered_gaze = [torch.zeros_like(all_gaze_tesnor) for _ in range(dist.get_world_size())]
# gathered_gaze_gt = [torch.zeros_like(all_gaze_gt_tesnor) for _ in range(dist.get_world_size())]
# # gathered_ego = [torch.zeros_like(all_ego_tesnor) for _ in range(dist.get_world_size())]
# # gathered_ego_gt = [torch.zeros_like(all_ego_gt_tesnor) for _ in range(dist.get_world_size())]

# dist.all_gather(gathered_preds, all_preds_tesnor)
# dist.all_gather(gathered_labels, all_labels_tesnor)
# dist.all_gather(gathered_gaze, all_gaze_tesnor)
# dist.all_gather(gathered_gaze_gt, all_gaze_gt_tesnor)
# # dist.all_gather(gathered_ego, all_ego_tesnor)
# # dist.all_gather(gathered_ego_gt, all_ego_gt_tesnor)

# if dist.get_rank() == 0:
#     # Only on rank 0, calculate and print the full accuracy and F1 score
#     all_preds_flat = torch.cat(gathered_preds, dim=0).cpu().numpy()
#     all_labels_flat = torch.cat(gathered_labels, dim=0).cpu().numpy()
#     all_preds_flat_g = torch.cat(gathered_gaze, dim=0).cpu().numpy()
#     all_labels_flat_g = torch.cat(gathered_gaze_gt, dim=0).cpu().numpy()
#     # all_preds_flat_e = torch.cat(gathered_ego, dim=0).cpu().numpy()
#     # all_labels_flat_e = torch.cat(gathered_ego_gt, dim=0).cpu().numpy()

#     accuracy = accuracy_score(all_labels_flat, all_preds_flat)
#     f1 = f1_score(all_labels_flat, all_preds_flat, average='weighted')
#     accuracy_g = accuracy_score(all_labels_flat_g, all_preds_flat_g)
#     f1_g = f1_score(all_labels_flat_g, all_preds_flat_g, average='weighted')

#     print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
#     print(f"Accuracy(Gaze): {accuracy_g:.4f}, F1 Score(Gaze): {f1_g:.4f}")