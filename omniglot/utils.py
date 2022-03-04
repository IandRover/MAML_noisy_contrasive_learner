import os

def get_config(args):
    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]
    return config

def name_path(main_path, args):
    
    if args.order == 1:
        order_str = "FO_"
    elif args.order == 2:
        order_str = "SO_"
    else:
        print("variable name is invalid")
        
    initvar_str = "initvar{:.1f}_".format(args.initvar)
        
    zero_str = "zero{}_".format(args.zero)
        
    path = "{}/omni_{}w{}s_{}{}{}seed{}".format(main_path, 
            args.n_way, args.k_spt, 
            order_str, zero_str, initvar_str, args.seed)
    
    if os.path.exists(path): 
        print("path already exists!!")        
    else:
        os.mkdir(path)
    return path

