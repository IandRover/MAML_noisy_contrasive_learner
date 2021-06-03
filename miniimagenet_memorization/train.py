import  torch, os
import  numpy as np
from    MiniImagenet_memorization import MiniImagenet as MiniImagenet_fix
from    torch.utils.data import DataLoader
import  random, argparse
from    meta import Meta_mini
from    utils import get_config, save_model, name_path, load_model
import time
import pandas as pd


# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int)
parser.add_argument('--s', type=int)
parser.add_argument('--q', type=int)
parser.add_argument('--zero', type=int)
parser.add_argument('--zero_interval', type=int, default=1)
parser.add_argument('--order', type=str)
parser.add_argument('--inivar', type=float)
parser.add_argument('--seed', type=int)
parser.add_argument('--device', type=int)
parser.add_argument('--epoch', type=int, default=12)
parser.add_argument('--resume_epoch', type=int, default=-1)
args = parser.parse_args()

n_way = args.n
k_shot = args.s
k_qry = args.q
if args.zero == 1:
    apply_zero_trick = True
elif args.zero == 0:
    apply_zero_trick = False
maml_order = args.order
init_var = args.inivar
seed = args.seed
device = torch.device('cuda:{}'.format(args.device))
num_epoch = args.epoch
if args.s == 1: task_num, batchsz = 4, 10000
if args.s == 5: task_num, batchsz = 2, 5000
outer_lr, inner_lr = 0.001, 0.01
train_update_steps, test_update_steps = 5, 10

# Set seeds
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Directories
root = "./results/"               # The directory for saving files
data_root = "../data_miniImagenet/" # The directory for data

save_path = name_path(root, n_way, k_shot, k_qry, 
          maml_order, args.zero, args.zero_interval, init_var,
          seed)
print(save_path)

mini = MiniImagenet_fix(data_root, mode='train', n_way=n_way, k_shot=k_shot, k_query=k_qry, batchsz=batchsz, resize=84)
mini_test = MiniImagenet_fix(data_root, mode='test', n_way=n_way, k_shot=k_shot, k_query=k_qry, batchsz=400, resize=84)
maml = Meta_mini(n_way, k_shot, k_qry, task_num, 
                train_update_steps, test_update_steps, 
                inner_lr, outer_lr, get_config(n_way), device).to(device)

# Set initial norm
maml.set_last_layer_variance(init_var)
if init_var == 0:
    maml.set_last_layer_to_zero()

test_text = []
train_text = []

print("start running", args)
start = time.time()

count_steps = 0
test_zero_text = []
test_text = []
for epoch in range(num_epoch):
    db = DataLoader(mini, task_num, shuffle=True, num_workers=8, pin_memory=True)
    
    # Set resume training 
    if args.resume_epoch>epoch: 
        continue
    elif args.resume_epoch==epoch:
        maml.load_model(save_path, epoch)

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        
        # Apply zero trick or not
        if apply_zero_trick and (count_steps%args.zero_interval==0):
            maml.set_last_layer_to_zero()
            
        # Choose first order or second order MAML for training
        if maml_order == "first":
            accs = maml.forward_FOMAML(x_spt, y_spt, x_qry, y_qry)
        elif maml_order == "second":
            accs = maml.forward_SOMAML(x_spt, y_spt, x_qry, y_qry)
        
        # Finetuning
        if step % 200 == 0:            
            train_text.append(accs)
            print(time.time()-start)
            db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=8, pin_memory=True)
            accs_all_test = []
            accs_all_test_zero = []
            for x_spt, y_spt, x_qry, y_qry in db_test:
                x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                # Original finetune method 
                accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                accs_all_test.append(accs)
                # finetune with applying zeroing trick 
                accs = maml.finetunning_zero(x_spt, y_spt, x_qry, y_qry)
                accs_all_test_zero.append(accs)
            accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            test_text.append(accs)
            accs = np.array(accs_all_test_zero).mean(axis=0).astype(np.float16)
            test_zero_text.append(accs)
            print(time.time()-start)
        
        if (count_steps) % 200 == 0:
            maml.save_model(save_path, epoch, step)
            txt_path = os.path.join(save_path, "test_E{}S{}.csv".format(epoch, step))
            df = pd.DataFrame(test_text)
            df.to_csv(txt_path,index=False)
            txt_path = os.path.join(save_path, "test_zero_E{}S{}.csv".format(epoch, step))
            df = pd.DataFrame(test_zero_text)
            df.to_csv(txt_path,index=False)
            
        count_steps += 1
