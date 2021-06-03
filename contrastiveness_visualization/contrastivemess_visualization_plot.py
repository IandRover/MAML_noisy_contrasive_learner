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