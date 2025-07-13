import torch


model_path = 'weights/rgb_imagenet.pt'
state_dict = torch.load(model_path)

duplicated_state = {}

for key, value in state_dict.items():

    
    if 'logits' not in key:
        
        #import pdb;pdb.set_trace()
        sub_key= key.split('.')
        sub_key[0] = sub_key[0]+'_2'
        new_key = '.'.join(sub_key)
        
        duplicated_state[new_key] = value
        duplicated_state[key] = value
    else:
        duplicated_state[key] = value


new_path = 'weights/rgb_imagenet_modified.pt'
torch.save(duplicated_state, new_path)
