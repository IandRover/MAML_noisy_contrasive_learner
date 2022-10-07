import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
from    mini_meta_exact_maml import Meta_mini
from    mini_utils import get_config, name_path

from    torch.utils.data import DataLoader
import  random, argparse
import time
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--n_way', type=int, default=5)
parser.add_argument('--k_shot', type=int, default=1)
parser.add_argument('--k_qry', type=int, default=15)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--epoch', type=int, default=12)

parser.add_argument('--loss', type=str, default="origin", choices=["origin", "bound"])
parser.add_argument('--zero_interval', type=int, default=1)
parser.add_argument('--update_steps', type=int, default=1)
parser.add_argument('--feature_norm', type=int, default=0)

# scheme
parser.add_argument('--scheme', type=str, default="maml", choices=["maml", "contrastive"])
# MAML
parser.add_argument('--order', type=int, default=1)
parser.add_argument('--initvar', type=int, default=1)
parser.add_argument('--head', type=str, default="origin", choices=["origin", "zero", "random"])
# Contrastive
parser.add_argument("--IFR", type=int, default=1, choices=[0,1])
parser.add_argument("--q_contrast", type=int, default=0, choices=[0,1])
args, _ = parser.parse_known_args()

if args.scheme == "maml":
    assert args.IFR == 1
    assert args.q_contrast == 0
    

if args.k_shot == 1: task_num, batchsz = 4, 10000
if args.k_shot == 5: task_num, batchsz = 2, 5000
outer_lr, inner_lr = 0.001, 0.1
train_update_steps, test_update_steps = args.update_steps, 10
device = torch.device('cuda:{}'.format(args.device))
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
root = "./results/"
if not os.path.exists(root): os.mkdir(root)
if os.path.exists("/data"): data_root = "/data/miniimagenet/"
save_path = name_path(root, args)

mini = MiniImagenet(args, data_root, mode='train', batchsz=batchsz, resize=84)
mini_test = MiniImagenet(args, data_root, mode='test', batchsz=200, resize=84)
maml = Meta_mini(args, task_num,
                train_update_steps, test_update_steps, 
                inner_lr, outer_lr, get_config(args.n_way), device).to(device)

maml.set_last_layer_variance(maml, args.initvar)

print("start running", args)
start = time.time()

test_text = []
test_zero_text = []
count_steps = 0
for epoch in range(args.epoch):
    db = DataLoader(mini, task_num, shuffle=True, num_workers=8, pin_memory=True)
    
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        
        if args.head == "zero" and (count_steps % args.zero_interval == 0):
            maml.set_last_layer_variance(maml, 0)
        elif args.head == "random":
            maml.set_last_layer_random()
            
        if args.scheme == "maml":
            maml.forward_maml_one_step(x_spt, y_spt, x_qry, y_qry)
        elif args.scheme == "contrastive":
            maml.forward_contrastive(x_spt, y_spt, x_qry, y_qry)
            
        if ((step+1) % 250 == 0) or (epoch == 0 and step == 0) :
            print("Epcoh: {}, step: {}".format(epoch, step))
            print("train time: {:.1f}".format(time.time()-start))
            start = time.time()
            db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=8, pin_memory=True)
            accs_all_test = []
            accs_all_test_zero = []
            for x_spt, y_spt, x_qry, y_qry in db_test:
                x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                accs_all_test.append(accs)
                accs = maml.finetunning_zero(x_spt, y_spt, x_qry, y_qry)
                accs_all_test_zero.append(accs)
            accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            test_text.append(accs)
            accs_ = np.array(accs_all_test_zero).mean(axis=0).astype(np.float16)
            test_zero_text.append(accs_)
            print("test time: {:.1f}".format(time.time()-start))
            print("Epoch {} Step {}: accs: {}".format(epoch, step, accs[:5]))
            print("Epoch {} Step {}: accs: {}".format(epoch, step, accs_[:5]))
            start = time.time()
        
        if (step+1)  ==  2500:
            txt_path = os.path.join(save_path, "test_E{}S{}.csv".format(epoch, step))
            df = pd.DataFrame(test_text)
            df.to_csv(txt_path,index=False)

            txt_path = os.path.join(save_path, "test_zero_E{}S{}.csv".format(epoch, step))
            df = pd.DataFrame(test_zero_text)
            df.to_csv(txt_path,index=False)

            model_path = os.path.join(save_path, "model_E{}.pt".format(epoch))
            maml.save_model(model_path)
            
        count_steps += 1
        
txt_path = os.path.join(save_path, "final.csv")
df = pd.DataFrame(test_text)
df.to_csv(txt_path,index=False)
txt_path = os.path.join(save_path, "final_zero.csv")
df = pd.DataFrame(test_zero_text)
df.to_csv(txt_path,index=False)

model_path = os.path.join(save_path, "final_model_E{}.pt".format(epoch))
maml.save_model(model_path)

