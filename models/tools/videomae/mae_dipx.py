import torch 
import torch.nn as nn 
import numpy as np 

from utils.plot_confusion import confusion
import argparse
import torch.optim as optim
import torchvision
from tqdm.auto import tqdm
import os
import psutil
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, f1_score



from utils.tsne import plot_tsne as TSNE
from utils.plot_confusion import confusion
from utils.loader import CustomDataset
from model import build_model



def Trainer(args, train_subset, valid_subset ):

    log_dir = f"runs_{args.model}_{args.dataset}_{args.technique}/fold_brain"  # Separate log directory for each fold
    writer = SummaryWriter(log_dir)   
    # Initialize model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    model.to(device)
    

    train_loader = torch.utils.data.DataLoader(train_subset,num_workers=args.workers, batch_size=args.batch,pin_memory=True, shuffle= True)
    val_loader = torch.utils.data.DataLoader(valid_subset,num_workers=args.workers, batch_size=args.batch)

    #import pdb;pdb.set_trace()


    # if args.distributed:
    #     for param in model.module.parameters():
    #         param.requires_grad = False
    #     for layer in model.module.first_model.model1.videomae.encoder.layer:
    #         for param in layer.attention.parameters():
    #             param.requires_grad = True
    #     for layer in model.module.first_model.model2.videomae.encoder.layer:
    #         for param in layer.attention.parameters():
    #             param.requires_grad = True
    #     for param in model.module.sec_model.parameters():
    #         param.requires_grad = True
        
    #     if model.module.third_model is not None :
    #         for param in model.module.third_model.parameters():
    #             param.requires_grad = True
    #     if model.module.fourth_model is not None:
    #         for param in model.module.third_model.parameters():
    #             param.requires_grad = True


    # else:

    for param in model.first_model.parameters():
        param.requires_grad = False
    for layer in model.first_model.model1.videomae.encoder.layer:
        for param in layer.attention.parameters():
            param.requires_grad = True
    for layer in model.first_model.model2.videomae.encoder.layer:
        for param in layer.attention.parameters():
            param.requires_grad = True

    if args.ego_cbm:
        for param in model.third_model.parameters():
            param.requires_grad= False

    #import pdb;pdb.set_trace()
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

    base_learning_rate = args.learning_rate
    weight_decay = 0.05
    if args.distributed:
        optimizer = optim.AdamW(model.module.parameters(), lr=base_learning_rate, weight_decay=weight_decay)
    else:

        optimizer = optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay)

    #scheduler = CosineAnnealingLR(optimizer, len(train_loader))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.000005)

    if args.resume:
        print("Resuming from the checkpoint")
        ckp = torch.load(args.ckp)
        model.load_state_dict(ckp['model_state_dict'])
        optimizer.load_state_dict(ckp['optimizer_state_dict'])


    # Train and evaluate the model
    if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:
        acc,f1,acc_gaze,f1_gaze,acc_ego,f1_ego,f1_ego_micro,f1_ego_macro = train(args, train_loader, val_loader, model,scheduler, criterion1, criterion2, criterion3, optimizer, device,writer)
        print("Average Accuracy",acc)
        print("Average F1",f1)
        print("Average Accuracy(Gaze)",acc_gaze)
        print("Average F1(Gaze)",f1_gaze)
        print("Average Accuracy(Ego)",acc_ego)
        print("Average F1(Ego)",f1_ego)
        print("Average F1(EGO) micro",f1_ego_micro)
        print("Average F1(EGO) macro",f1_ego_macro)
    else:
        accuracy, f1 = train(args, train_loader, val_loader, model, scheduler, criterion1, criterion2, criterion3, optimizer, device,writer)
        print("Average Accuracy",accuracy)
        print("Average F1",f1)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")



def train(args, train_dataloader, valid_dataloader, model,scheduler, criterion1, criterion2, criterion3, optimizer, device,writer):
    

    T=1
    num_epochs=args.num_epochs
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
        save_dir = f"best_{args.model}_{args.dataset}_dir"
        
    patience_counter = 0
    accumulation_steps = args.accumulation
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    best_model_path = os.path.join(save_dir, f"best_{args.model}_{args.dataset}.pth") 

    process = psutil.Process()

    print("Started Training")
    for epoch in range(num_epochs):
        model.train()
        model.to(device)

        all_preds = []
        all_labels = []
        train_loss = 0.0
        for i, batch in tqdm(enumerate(train_dataloader)):
            
            *images,cls,gaze,ego = batch

            images = [img.to(device) for img in images]
            images = [img.type(torch.cuda.FloatTensor) for img in images]
            label = cls.to(device)

           # import pdb;pdb.set_trace()
            inputs1 = {"pixel_values": images[0].permute((0,2,1,-2,-1)),"labels":label}
            inputs2 = {"pixel_values": images[1].permute((0,2,1,-2,-1)),"labels":label}

            inputs1 = {k: v for k, v in inputs1.items()}
            inputs2 = {k: v for k, v in inputs2.items()}


            outputs = model(inputs1,inputs2)
            #import pdb;pdb.set_trace()
            feat = model.first_model.feat
            
            loss1 = criterion1(outputs[0],label)  
            #import pdb;pdb.set_trace()
            if args.gaze_cbm:

                loss3 = lam2*criterion3(outputs[1],torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),gaze.cuda().to(device))
                loss = loss1 + loss2 + loss3
            
            elif args.ego_cbm:
                #import pdb;pdb.set_trace()
                #loss3 = lam2*criterion3(outputs[1],gaze.cuda().to(device))
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))
                loss = loss1 + loss2 #+ loss3


            
            elif args.combined_bottleneck:
                #import pdb;pdb.set_trace()
                loss2 = lam1*criterion2(torch.hstack(outputs[1:16]),gaze.cuda().to(device))
                loss3 = lam2*criterion3(torch.hstack(outputs[16:33]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))
                loss = loss1 + loss2 + loss3

            elif args.multitask:

                loss3 = lam2*criterion3(outputs[1],gaze.cuda().to(device))
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))

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

            ###################### grad accumulation ############################
            if args.accumulation > 1:

                loss_val = loss.cpu()
                train_loss += loss_val

                loss = loss / accumulation_steps    
                loss.backward()

                if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                    optimizer.step()                            # Now we can do an optimizer step
                    optimizer.zero_grad() 
            else:
                    
                loss_val = loss.cpu()
                train_loss += loss_val
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() 
            ###################### grad accumulation ############################
            # Gradient clipping



            clip_grad_norm_(model.parameters(), max_norm=1.0)


            
            predicted = torch.argmax(outputs[0],dim=1)
            all_preds.append(predicted.cpu())
            all_labels.append(label.cpu())

            del images, outputs, label
            torch.cuda.empty_cache()
            #import pdb;pdb.set_trace()
            memory_usage = process.memory_info().rss / (1024 ** 2)

        scheduler.step()


        epoch_loss = train_loss/len(train_dataloader)

        print(f"epoch {epoch} Loss: {epoch_loss:.4f}, memory {memory_usage} MB")

        train_losses.append(epoch_loss)

        all_preds = np.hstack(all_preds)
        all_labels = np.hstack(all_labels)

        accuracy_train = accuracy_score(all_labels, all_preds)
        f1_train = f1_score(all_labels, all_preds, average='weighted')

        print(f"Accuracy(Trainign): {accuracy_train:.4f}, F1 Score: {f1_train:.4f}")

        ### EVAL 
        if epoch % 5 ==0:
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

                for i, batch in tqdm(enumerate(valid_dataloader)): 

                    *images,cls,gaze,ego = batch

                    images = [img.to(device) for img in images]
                    images = [img.type(torch.cuda.FloatTensor) for img in images]
                    label = cls.to(device)

                # import pdb;pdb.set_trace()
                    inputs1 = {"pixel_values": images[0].permute((0,2,1,-2,-1)),"labels":label}
                    inputs2 = {"pixel_values": images[1].permute((0,2,1,-2,-1)),"labels":label}

                    inputs1 = {k: v for k, v in inputs1.items()}
                    inputs2 = {k: v for k, v in inputs2.items()}


                    outputs = model(inputs1,inputs2)

                    if args.grad_cam:
                        with torch.enable_grad():
                            import pdb;pdb.set_trace()
                            tar=["first_model/model1/videomae/encoder/layer","first_model/model2/videomae/encoder/layer"]                   
                            grad = GradCAM(model,tar,[0,0,0],[1,1,1])
                            img,_ = grad([inputs1,inputs2],label)
                            # visualize(img[0].squeeze(0))

                    feat = model.first_model.feat


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
                        all_labels_ego.append(torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).cpu())

                    elif args.ego_cbm:
                            
                        #loss3 = lam2*criterion3(outputs[1],gaze.cuda())
                        loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))
                        loss = loss1 + loss2 #+ loss3
                        predicted_gaze = torch.argmax(outputs[1],dim=1)
                        all_preds_gaze.append(predicted_gaze.cpu())
                        all_labels_gaze.append(gaze.cpu())    
                        predicted_ego = (torch.sigmoid(torch.hstack(outputs[2:])) > 0.5).float().cpu()
                        all_preds_ego.append(predicted_ego)
                    #import pdb;pdb.set_trace()
                        all_labels_ego.append(torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).cpu())



                    elif args.combined_bottleneck:
                        #import pdb;pdb.set_trace()
                        loss2 = lam1*criterion2(torch.hstack(outputs[1:16]),gaze.cuda())
                        loss3 = lam2*criterion3(torch.hstack(outputs[16:33]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))
                        loss = loss1 + loss2 + loss3
                        predicted_gaze = torch.argmax(torch.hstack(outputs[1:16]),dim=1)
                        all_preds_gaze.append(predicted_gaze.cpu())
                        all_labels_gaze.append(gaze.cpu())          
                        predicted_ego = (torch.sigmoid(torch.hstack(outputs[16:33])) > 0.5).float().cpu()
                        all_preds_ego.append(predicted_ego)
                        all_labels_ego.append(torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).cpu()) 
                           
                    elif args.multitask:

                        loss3 = lam2*criterion3(outputs[1],gaze.cuda())
                        loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))

                        loss = loss1 + loss2 + loss3
                        #import pdb;pdb.set_trace()
                        predicted_gaze = torch.argmax(outputs[1],dim=1)
                        all_preds_gaze.append(predicted_gaze.cpu())
                        all_labels_gaze.append(gaze.cpu()) 
                        #import pdb;pdb.set_trace()
                        predicted_ego = (torch.sigmoid(outputs[2]) > 0.5).float().cpu()
                        all_preds_ego.append(predicted_ego)
                        all_labels_ego.append(torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).cpu())
                    else:

                        loss = loss1

                    val_loss_running+=loss

                    predicted = torch.argmax(outputs[0],dim=1)
                    

                    all_preds.append(predicted.cpu())
                    all_labels.append(label.cpu())
    

            #         FEAT.append(feat.cpu())
            #         LABEL.append(label.cpu())

            # tsne = TSNE()
            # tsne_img = tsne.plot(FEAT,LABEL,args.dataset)

            all_labels = np.hstack(all_labels)
            all_preds = np.hstack(all_preds)

            if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:
                

                all_labels_gaze = np.hstack(all_labels_gaze)
                all_preds_gaze = np.hstack(all_preds_gaze)
                # all_labels_ego = np.hstack(all_labels_ego)
                # all_preds_ego = np.hstack(all_preds_ego)    
                all_labels_ego = torch.cat(all_labels_ego, dim=0)  # Shape (N, 17)
                all_preds_ego = torch.cat(all_preds_ego, dim=0)    # Shape (N, 17)

                #confusion(all_labels_ego, all_preds_ego,'ego',writer,epoch)
                #confusion(all_labels_gaze, all_preds_gaze,'gaze',writer,epoch)
            # for Action Classification 

            #confusion(all_labels, all_preds,'action',writer,epoch)
            
            #writer.add_figure('TSNE', tsne_img,epoch)

            val_loss = val_loss_running/len(valid_dataloader)

            accuracy_val = accuracy_score(all_labels, all_preds)
            f1_val = f1_score(all_labels, all_preds, average='weighted') #'weighted' or 'macro' s

            if args.multitask or args.gaze_cbm or args.ego_cbm or args.combined_bottleneck:

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
                #writer.add_scalar("Accuracy/Validation(Gaze)", accuracy_val_gaze, epoch)
                #writer.add_scalar("F1/Validation(Gaze)", f1_val_gaze, epoch)
                #writer.add_scalar("Accuracy/Validation(Ego)", accuracy_val_ego, epoch)
                #writer.add_scalar("F1/Validation(Ego)", f1_val_ego, epoch)

            print("accuracy and F1",accuracy_val,f1_val) 

            

           # writer.add_scalar("Loss/Train", epoch_loss, epoch)
            #writer.add_scalar("Loss/Validation", val_loss, epoch)
            #writer.add_scalar("Accuracy/Validation", accuracy_val, epoch)
            #writer.add_scalar("Accuracy/Train", accuracy_train, epoch)

            #writer.add_scalar("F1/Validation", f1_val, epoch)
            #writer.add_scalar("F1/Train", f1_train, epoch)

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

                checkpoint_final = {
                    'epoch': epoch,  
                    'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint_final,best_model_path)
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
    seed = 37

    np.random.seed(seed)
    torch.manual_seed(seed) 
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--directory", help="Directory for home_dir", default = os.path.expanduser('~'))
    parser.add_argument("-e", "--epochs", type = int, help="Number of epochs", default = 1)
    parser.add_argument("--mem_per_layer", type = int, help="Number of memory tokens", default = 3)
    parser.add_argument("--dataset",  type = str, default = None)
    parser.add_argument("--learning_rate",  type = float, default = 0.001)
    parser.add_argument("--weight_decay",  type = float, default = 0.001)
    parser.add_argument("--model",  type = str, default = None)
    parser.add_argument("--debug",  type = str, default = None)
    parser.add_argument("-grad_cam",action="store_true")
    parser.add_argument("--technique",  type = str, default = None)
    parser.add_argument("--num_classes",  type = int, default = 5)
    parser.add_argument("--batch",  type = int, default = 1)
    parser.add_argument("--workers",  type = int, default = 1)
    parser.add_argument("--port",  type = int, default = 12345)
    parser.add_argument("-distributed",  action ="store_true")
    parser.add_argument("--n_attributes", type = int, default= None) # for bottleneck
    parser.add_argument("--num_epochs", default = 100, type=int)
    parser.add_argument("--connect_CY", type = bool, default= False)
    parser.add_argument("--expand_dim", type = int, default= 0)
    parser.add_argument("--use_relu", type = bool, default= False)
    parser.add_argument("--use_sigmoid", type = bool, default= False)
    parser.add_argument("--multitask_classes", type = int, default=None) # for final classification along with action classificaiton
    parser.add_argument("--dropout", type = float, default= 0.0)
    parser.add_argument("--accumulation",type= int, default = 1)
    parser.add_argument("-bottleneck",  action="store_true", help="Enable bottleneck mode")
    parser.add_argument("-gaze_cbm", action="store_true", help="Enable gaze CBM mode")
    parser.add_argument("-ego_cbm", action="store_true", help="Enable ego CBM mode")
    parser.add_argument("-multitask", action="store_true", help="Enable multitask mode")
    parser.add_argument("-combined_bottleneck", action="store_true", help="Enable combined_bottleneck mode")
    parser.add_argument("-resume", action="store_true",help="auto resume from a checkpoint" )
    parser.add_argument("--ckp",type= str, default = None)
    args = parser.parse_args()

    home_dir = str(args.directory)
    cache_dir = os.path.join(home_dir, "mukil")
    world_size = torch.cuda.device_count()

    train_csv = "/scratch/mukil/dipx/train.csv"
    val_csv = "/scratch/mukil/dipx/val.csv"

    # transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # train_subset = CustomDataset(train_csv, debug = args.debug, transform=transform)
    # val_subset = CustomDataset(val_csv, debug=args.debug, transform=transform)
    train_subset = CustomDataset(train_csv, debug = args.debug)
    
    val_subset = CustomDataset(val_csv, debug=args.debug)
    Trainer(args, train_subset, val_subset)