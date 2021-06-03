import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
from    torch.utils.data import DataLoader
import  random
from    mini_meta_feature import Meta_mini
from    mini_utils import get_config, save_model, name_path, load_model
import matplotlib.pyplot as plt
import pickle
from visualization_utils import metric2_cos, get_cross_covariance, shuffle, get_averaged_matrix, get_map


n_way = 5
k_shot = 20
k_qry = 20
maml_order = "first"
init_var = 1
seed_start = 222
seed_end = 232
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epoch = 500
task_num = 1
batchsz = 10
outer_lr, inner_lr = 0.001, 0.01
train_update_steps, test_update_steps = 5, 10

root = "./results/"
"""Please set the data_root here"""
data_root = "../data_miniImagenet/"

mini = MiniImagenet(data_root, mode='test', n_way=n_way, k_shot=k_shot, k_query=k_qry, batchsz=400, resize=84)
maml = Meta_mini(n_way, k_shot, k_qry, task_num, 
                train_update_steps, test_update_steps, 
                inner_lr, outer_lr, get_config(n_way), device).to(device)
maml.set_last_layer_variance(init_var)
if init_var == 0:
    maml.set_last_layer_to_zero()
db = DataLoader(mini, task_num, shuffle=True, num_workers=8, pin_memory=True)

"""Original FOMAML"""
for seed in range(seed_start, seed_end):
    """
    memory_all: used to collect the computed cosine similarity along training
    get_cross_covariance: used to compute the cosine similarity between features.
    """
    # Set the random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Initialize the meta-learning model
    maml = Meta_mini(n_way, k_shot, k_qry, task_num, 
                    train_update_steps, test_update_steps, 
                    inner_lr, outer_lr, get_config(n_way), device).to(device)
    
    # Initialize the data loader
    db = DataLoader(mini, task_num, shuffle=True, num_workers=8, pin_memory=True)
    
    memory_all = list()
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        # We consider a simplified scenario where there is only one task. 
        # So this for-loop does not iterate but break quickly
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        # The model undergoes 101 outer loop updates
        for i in range(101):
            # To avoid channel-memorization problem. we explicitly perform channel shuffling.
            # Please refer to visualization_utils.py for more details
            y_spt_s, y_qry_s =shuffle(y_spt, y_qry)
            # Forward the data
            accs = maml.forward_FOMAML(x_spt, y_spt_s, x_qry, y_qry_s)
            if i % 10 == 0:
                # The cosine similarity is computed. 
                # Please refer to visualization_utils.py for more details
                memory = get_cross_covariance(maml, x_spt, x_qry, y_spt, y_qry)
                
                memory_all.append(memory)   
        break
    with open('./pickles/RandInit_{}.pickle'.format(seed), 'wb') as handle:
        pickle.dump(memory_all, handle)
        
"""Original FOMAML + zero initialization"""
for seed in range(seed_start, seed_end):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    maml = Meta_mini(n_way, k_shot, k_qry, task_num, 
                    train_update_steps, test_update_steps, 
                    inner_lr, outer_lr, get_config(n_way), device).to(device)
    db = DataLoader(mini, task_num, shuffle=True, num_workers=8, pin_memory=True)
    
    maml.set_last_layer_to_zero()
    memory_all = list()
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        for i in range(101):
            y_spt_s, y_qry_s =shuffle(y_spt, y_qry)
            accs = maml.forward_FOMAML(x_spt, y_spt_s, x_qry, y_qry_s)
            if i % 10 == 0:
                memory = get_cross_covariance(maml, x_spt, x_qry, y_spt, y_qry)
                memory_all.append(memory)   
        break
    with open('./pickles/ZeroInit_{}.pickle'.format(seed), 'wb') as handle:
        pickle.dump(memory_all, handle)
        
"""Original FOMAML + zering trick"""
for seed in range(seed_start, seed_end):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    maml = Meta_mini(n_way, k_shot, k_qry, task_num, 
                    train_update_steps, test_update_steps, 
                    inner_lr, outer_lr, get_config(n_way), device).to(device)
    db = DataLoader(mini, task_num, shuffle=True, num_workers=8, pin_memory=True)

    memory_all = list()
    maml.set_last_layer_to_zero()
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        for i in range(101):
            
            y_spt_s, y_qry_s =shuffle(y_spt, y_qry)
            accs = maml.forward_FOMAML(x_spt, y_spt_s, x_qry, y_qry_s)
            if i % 10 == 0:
                memory = get_cross_covariance(maml, x_spt, x_qry, y_spt, y_qry)
                memory_all.append(memory)   
            if i % 1 == 0:
                maml.set_last_layer_to_zero()
        break
    with open('./pickles/ZeroIter1_{}.pickle'.format(seed), 'wb') as handle:
        pickle.dump(memory_all, handle)
        
"""Plot the results"""
cmap = "Spectral_r"
vmin, vmax = 0, 0.6
c = 3
print("Show the main results")
print("Note that the mid column is used to seperate the results. Please ignore it")
for task in ["RandInit", "ZeroInit","ZeroIter1"]:
    fig, axes = plt.subplots(1,c, figsize=(c*8+(c-1),4))
    matrix = np.abs(get_averaged_matrix(task))
    ax = axes[0]
    ax.set_title("\n1 outer loop updates", fontsize=(20))
    ax.pcolormesh(get_map(matrix[0]), vmin=vmin, vmax=vmax, cmap=cmap, edgecolors="white",linewidth=1.5)
    ax.axis('off')
    matrix = get_averaged_matrix(task)
    ax = axes[1]
    ax.set_title("\n10 outer loop updates", fontsize=(20))
    ax.pcolormesh(get_map(matrix[1]), vmin=vmin, vmax=vmax, cmap=cmap, edgecolors="white",linewidth=1.5)
    ax.axis('off')
    matrix = get_averaged_matrix(task)
    ax = axes[2]
    ax.set_title("\n100 outer loop updates", fontsize=(20))
    ax.pcolormesh(get_map(matrix[10]), vmin=vmin, vmax=vmax, cmap=cmap, edgecolors="white",linewidth=1.5)
    ax.axis('off')
    plt.savefig("./figures/{}.png".format(task))

print("Show the color bar")
a = np.array([[0,0.6]])
plt.figure(figsize=(8, 0.4))
img = plt.imshow(a, cmap="Spectral_r")
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.2, 0.8, 0.6])
plt.colorbar(orientation="horizontal", cax=cax)
plt.show()