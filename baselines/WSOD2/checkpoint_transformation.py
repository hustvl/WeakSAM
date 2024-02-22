import torch
new_state_dict = {}

# eva
"""
path = "/mnt/data01/fcy/WSOD2/pretrained/eva02_S_pt_in21k_p14.pt"
model = torch.load(path)['module']
new_state_dict = {}

for key, value in model.items():
    if "w12" in key:
        w1 = key.replace('w12', 'w1')
        w2 = key.replace('w12', 'w2')
        new_state_dict[w1] = value
        new_state_dict[w2] = value

    # elif key == "norm.weight":
    # new_state_dict["fc_norm.weight"] = value

    # elif key == "norm.bias":
    # new_state_dict["fc_norm.bias"] = value

    else:
    new_state_dict[key] =  value
torch.save(new_state_dict, "/mnt/data01/fcy/WSOD2/pretrained/eva02.pth")
"""

path = "/home/junweizhou/WeakSAM/WSOD2/pretrain/vanillanet_6.pth"  # To be transformed.
model = torch.load(path)['model']
new_state_dict = {}

for key, value in model.items():

    if "cls" in key:
        continue
    
    new_state_dict["backbone." + key] = value

torch.save(new_state_dict, "/home/junweizhou/WeakSAM/WSOD2/pretrain/vanilla_6.pth")