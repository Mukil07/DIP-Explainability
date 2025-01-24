from model.vit_dino import vit_base
from model.vit_mem import vit_mem
from model.vit_mem_multi import vit_mem_multi
from model.mem_cbm import vit_mem_dipx
from model.i3d import InceptionI3d
from model.videomae import vit_base_patch16_224
from model.multi_mae import Multi_Mae
from model.multi_mae_test import Multi_Mae_test
#from model.cbm import ModelXtoCtoY
from model.cbm import ModelXtoCtoY
def build_model(args):

    if args.model == 'vit':
        return vit_base()
    elif args.model == 'memvit':
        return vit_mem(num_memories_per_layer= args.mem_per_layer, num_classes=args.num_classes, drop=args.dropout)
    elif args.model == 'memvit_multi':
        return vit_mem_multi(num_memories_per_layer= args.mem_per_layer, num_classes=args.num_classes, drop=args.dropout)
    elif args.model == 'memvit_dipx':
        return vit_mem_dipx(args.num_classes,args.multitask_classes, args.multitask, args.n_attributes, args.bottleneck, args.expand_dim,
                 args.use_relu, args.use_sigmoid, args.connect_CY, args.dropout,num_memories_per_layer= args.mem_per_layer)    
    elif args.model == 'i3d':
        return InceptionI3d(args.num_classes, in_channels=3)
    elif args.model == 'mae':
        return vit_base_patch16_224()
    # elif args.model == 'cbm':
    #     return ModelXtoCtoY(args.num_classes, args.n_attributes, args.bottleneck, args.expand_dim,
    #              args.use_relu, args.use_sigmoid, args.connect_CY)
    
    elif args.model == 'cbm':

        return ModelXtoCtoY(args.num_classes, args.multitask_classes, args.multitask, args.n_attributes, args.bottleneck, args.expand_dim,
                 args.use_relu, args.use_sigmoid, args.connect_CY, args.dropout)  
    
    elif args.model == 'multimae':

        return Multi_Mae(args.num_classes,args.multitask_classes, args.multitask, args.n_attributes, args.bottleneck, args.expand_dim,
                 args.use_relu, args.use_sigmoid, args.connect_CY, args.dropout)       

    elif args.model == 'multimae_test':

        return Multi_Mae_test(args.num_classes,args.multitask_classes, args.multitask, args.n_attributes, args.bottleneck, args.expand_dim,
                 args.use_relu, args.use_sigmoid, args.connect_CY, args.dropout)     
    else :
        print("Original DINO VIT is being used here ")
        return vit_base()