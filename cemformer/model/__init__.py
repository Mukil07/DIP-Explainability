from model.vit_dino import vit_base
from model.vit_mem import vit_mem
from model.vit_mem_dipx import vit_mem_dipx
from model.i3d import InceptionI3d
from model.videomae import vit_base_patch16_224
from model.cbm import ModelXtoCtoY
def build_model(args):

    if args.model == 'vit':
        return vit_base()
    elif args.model == 'memvit':
        return vit_mem(num_memories_per_layer= args.mem_per_layer, num_classes=args.num_classes)
    elif args.model == 'memvit_dipx':
        return vit_mem_dipx(num_memories_per_layer= args.mem_per_layer, num_classes=args.num_classes)    
    elif args.model == 'i3d':
        return InceptionI3d(5, in_channels=3)
    elif args.model == 'mae':
        return vit_base_patch16_224()
    elif args.model == 'cbm':
        return ModelXtoCtoY(args.num_classes, args.n_attributes, args.expand_dim,
                 args.use_relu, args.use_sigmoid,args.connect_CY)
    else :
        print("Original DINO VIT is being used here ")
        return vit_base()