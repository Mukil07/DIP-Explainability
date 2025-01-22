

import torch 
import torch.nn as nn 
import numpy as np 
import torch.optim as optim
import torchvision
from tqdm.auto import tqdm
import os

from torch.nn.utils import clip_grad_norm_

def train(args, train_dataloader, model,scheduler, criterion1, criterion2, criterion3, optimizer, device):
    model.train()

    T=1
    num_epochs=200
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
    print("Started Training")
    for epoch in range(num_epochs):
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
            inputs2 = {"pixel_values1": images[1].permute((0,2,1,-2,-1)),"pixel_values2":images[2].permute((0,2,1,-2,-1)),"labels":label}

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
                loss3 = lam2*criterion3(outputs[1],gaze.cuda().to(device))
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))
                loss = loss1 + loss2 + loss3


            
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

            loss_val = loss.cpu()
            train_loss += loss_val

            loss = loss / accumulation_steps    
            loss.backward()

            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()    

            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)


            
            predicted = torch.argmax(outputs[0],dim=1)
            all_preds.append(predicted.cpu())
            all_labels.append(label.cpu())

        scheduler.step()


        epoch_loss = train_loss/len(train_dataloader)

        print(f"epoch {epoch} Loss: {epoch_loss:.4f}")

        train_losses.append(epoch_loss)

        all_preds = np.hstack(all_preds)
        all_labels = np.hstack(all_labels)


def evaluate(args, model,valid_dataloader, criterion1, criterion2, criterion3, device):
    if args.debug:
        patience =1
    else:
        patience =1000
    #patience = 10 
    min_delta = 0.0001  
    best_acc = 0 
    counter = 0 
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
            inputs2 = {"pixel_values1": images[1].permute((0,2,1,-2,-1)),"pixel_values2":images[2].permute((0,2,1,-2,-1)),"labels":label}

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
                    
                loss3 = lam2*criterion3(outputs[1],gaze.cuda())
                loss2 = lam1*criterion2(torch.hstack(outputs[2:]),torch.vstack(ego).to(dtype=torch.float).permute((-1,-2)).to(device))
                loss = loss1 + loss2 + loss3
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