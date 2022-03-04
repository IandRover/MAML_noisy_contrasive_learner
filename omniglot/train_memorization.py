import  torch, os, time
import  numpy as np
from    omniglotNShot_memorization import OmniglotNShot_memorization
import  argparse
import  pandas as pd
from    meta import Meta
from    utils import get_config, name_path

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=3000)
argparser.add_argument('--n_way', type=int, help='n way', default=5)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
argparser.add_argument('--imgc', type=int, help='imgc', default=1)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=2)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)

argparser.add_argument('--seed', type=int, help='random seed', default=3)
argparser.add_argument('--device', type=int, help='cuda device index', default=0)
argparser.add_argument('--order', type=int, default=1)
## To apply the zeroing trick, set "zero" to 1.
argparser.add_argument('--zero', type=int, default=0)
## To scale the initial weight by "initvar".
argparser.add_argument('--initvar', type=float, default=1.0)

args, _ = argparser.parse_known_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

config = get_config(args)
main_path = "./results"
save_path = name_path(main_path, args)
device = torch.device('cuda:{}'.format(args.device))
maml = Meta(args, config).to(device)

def variables_scaling(scale):
    maml.net.vars[16].data *= scale
    maml.net.vars[17].data *= scale

db_train = OmniglotNShot_memorization('omniglot', batchsz=args.task_num, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, imgsz=args.imgsz)

variables_scaling(args.initvar)
memory = list()

start = time.time()
for step in range(args.epoch):

    x_spt, y_spt, x_qry, y_qry = db_train.next()
    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                 torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

    # the zeroing trick
    if args.zero == 1: variables_scaling(0)
    _ = maml(x_spt, y_spt, x_qry, y_qry, order=args.order)

    if step % 30 == 0:
        accs = [[], []]
        for _ in range(1000//args.task_num):
            # test
            x_spt, y_spt, x_qry, y_qry = db_train.next('test')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                         torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

            # split to single task each time
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one, zero=0)
                accs[0].append( test_acc )
                test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one, zero=1)
                accs[1].append( test_acc )

        acc = np.mean(accs, axis=1)
        memory.append(acc[0])
        memory.append(acc[1])
        print(step, np.round(acc[0], 3), np.round(acc[1], 3), str(int((time.time()-start)/(step+1)*(args.epoch-(step+1))/60)) + " mins")
        
        txt_path = os.path.join(save_path, "test_s{}.csv".format(step))
        df = pd.DataFrame(memory)
        df.to_csv(txt_path,index=False)