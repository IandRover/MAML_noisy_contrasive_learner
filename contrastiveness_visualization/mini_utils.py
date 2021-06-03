import pickle
import torch
import os

def get_config(n_way):
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [n_way, 32 * 5 * 5])
    ]
    return config

def save_model(model, PATH):
    torch.save(model.state_dict(), PATH)
def load_model(model, PATH):
    model.load_state_dict(torch.load(PATH), strict=True)
    model.eval()
    return model

def save_obj(obj, name):
    with open(name+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open(name+'.pickle', 'rb') as handle:
        return pickle.load(handle)

def name_path(main_path, n_way, k_shot, k_qry, 
              order, zero, zero_interval, init_var, seed, build = True):
    
    if order == "first":
        maml_order = "FO"
    elif order == "second":
        maml_order = "SO"
    elif order == "first_reg":
        maml_order = "FOreg"
    elif order == "second_reg":
        maml_order = "SOreg"
    else: 
        print("variable name is invalid")
        
    if init_var >= 0:
        initvar = "initvar{}_".format(init_var)
    else:
        initvar = ""
        
    if zero:
        zero_str = "zero{}_".format(zero_interval)
    else:
        zero_str = ""
        
    path = "{}/mini_{}w{}s{}q_{}_{}{}seed{}".format(main_path, 
            n_way, k_shot, k_qry, 
            maml_order, zero_str, initvar, seed)
    
    if os.path.exists(path): 
        print("path already exists!!")        
    elif build:
        os.mkdir(path)
    return path

