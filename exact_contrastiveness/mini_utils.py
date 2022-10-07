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

def get_config_noBN(n_way):
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
#         ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [n_way, 32 * 5 * 5])
    ]
    return config

def name_path(main_path, args):
    
    args1, args2, args3 = "", "", ""

    if args.feature_norm == 1:
        args3 = "fn_"
        
    if args.scheme == "maml":
        
        if args.order == 1: maml_order = "FO"
        elif args.order == 2: maml_order = "SO"

        if args.head == "zero": maml_type = "zero_"
        elif args.initvar == 1: maml_type = "init1_"
        elif args.initvar == 0: maml_type = "init0_"
        else: print(args.head, args.initvar)
        
        path = "{}/mini_{}_{}_{}seed{}".format(main_path, 
            args.scheme, maml_order, maml_type, args.seed)

    elif args.scheme == "contrastive":
        
        if args.head == "zero":
            
            maml_type = "zero_"
            
            if args.order == 1: maml_order = "FO"
            elif args.order == 2: maml_order = "SO"
                
            path = "{}/mini_{}_{}_{}seed{}".format(main_path, 
            args.scheme, maml_order, maml_type, args.seed)
        
        else:
            
            maml_order = "FO"
      
            path = "{}/mini_{}_{}_seed{}".format(main_path, 
            args.scheme, maml_order, args.seed)
            
            if args.IFR == 1:
                path += "_IFR1"
            if args.IFR == 0:
                path += "_IFR0"
            
            if args.pos_contrast == 1:
                path += "_pos_contrast"
            elif args.zero_w == 1:
                path += "_zero_w" # 有zero_w，一定要有q_contrast
            elif args.q_contrast == 1:
                path += "_q_contrast"
                
    elif args.scheme == "upper_bound":
        
        assert args.head == "zero"
            
        maml_type = "zero_"

        if args.order == 1: maml_order = "FO"
        elif args.order == 2: maml_order = "SO"

        path = "{}/mini_{}_{}_{}seed{}".format(main_path, 
        args.scheme, maml_order, maml_type, args.seed)
    
    if os.path.exists(path): 
        print("path already exists!!")
        print(path)
    else:
        os.mkdir(path)
        print(path)
    return path

