import torch 
import argparse

from tqdm.auto import tqdm
import os
import numpy as np 
from utils.DIPX_350 import CustomDataset
from utils.gradcam import GradCAM
from utils.save_img import visualize

from model import build_model


def trainer(args, train_subset, valid_subset, n_splits=5):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(args)
    model.to(device)
    #import pdb;pdb.set_trace()
    #checkpoint = "weights/dino_vitbase16_pretrain.pth"
    ckp = torch.load(args.weights,map_location=device)
    import pdb;pdb.set_trace()

    model.load_state_dict(ckp,strict=True)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valid_subset, batch_size=args.batch,pin_memory=True)

    for param in model.parameters():
        param.requires_grad = True

        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    # plot gradcam 
    val( val_loader, model, device)


def val( valid_dataloader, model, device):
    model.eval()

    #with torch.no_grad():

    for i, (img1,img2,cls,gaze,ego) in tqdm(enumerate(valid_dataloader)): 

            img1 = img1.to(device)
            img2 = img2.to(device)

            label = cls.to(device)

            # Forward pass

            img1=img1.type(torch.cuda.FloatTensor)
            img2=img2.type(torch.cuda.FloatTensor)
            outputs = model(img1,img2) 
            #import pdb;pdb.set_trace()
            #with torch.enable_grad():
            #import pdb;pdb.set_trace()
            print(torch.argmax(outputs[0]),label)
          
            #tar = ["first_model/MaxPool3d_5a_2x2","first_model/MaxPool3d_5a_2x2_2"]
            #tar = ["first_model/MaxPool3d_3a_3x3","first_model/MaxPool3d_3a_3x3_2"] # general feat 
            tar = ["first_model/MaxPool3d_4a_3x3", "first_model/MaxPool3d_4a_3x3_2"] # decent 
            #tar = ["first_model/MaxPool3d_5a_2x2_2", "first_model/MaxPool3d_5a_2x2_2"]
            #tar = ["first_model/Mixed_5c","first_model/Mixed_5c_2"]
            grad = GradCAM(model,tar,[0,0,0],[1,1,1])
            
            img,_ = grad([img1,img2],label)
            visualize(img[1].squeeze(0),i,"ego3") #for face image
            visualize(img[0].squeeze(0),i,"aria3")
            

            if i ==50:
                exit()
if __name__ == '__main__':
    seed = 37

    np.random.seed(seed)
    torch.manual_seed(seed) 
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
    train_csv = "/scratch/mukil_new/dipx/train_debug.csv"
    val_csv = "/scratch/mukil_new/dipx/val.csv"
    train_subset = CustomDataset(train_csv, debug = args.debug)
    val_subset = CustomDataset(val_csv, debug=args.debug)

    trainer(args,train_subset, val_subset)
