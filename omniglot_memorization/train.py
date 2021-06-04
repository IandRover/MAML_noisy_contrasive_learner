import torch
import math
import os
import time
import json
import logging
import argparse
import numpy as np
import copy

from torchmeta.utils.data import BatchMetaDataLoader

from maml.datasets import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning

import pandas as pd

parser = argparse.ArgumentParser('MAML')

# General
parser.add_argument('--dataset', type=str, 
    choices=['sinusoid', 'omniglot', 'miniimagenet'], default='omniglot',
    help='Name of the dataset (default: omniglot).')
parser.add_argument('--output-folder', type=str, default="./results",
    help='Path to the output folder to save the model.')

parser.add_argument('--num-shots-test', type=int, default=15,
    help='Number of test example per class. If negative, same as the number '
    'of training examples `--num-shots` (default: 15).')

# Model
parser.add_argument('--hidden-size', type=int, default=64,
    help='Number of channels in each convolution layer of the VGG network '
    '(default: 64).')

# Optimization
parser.add_argument('--batch-size', type=int, default=32,
    help='Number of tasks in a batch of tasks (default: 25).')
parser.add_argument('--num-epochs', type=int, default=600,
    help='Number of epochs of meta-training (default: 50).')
parser.add_argument('--num-batches', type=int, default=100,
    help='Number of batch of tasks per epoch (default: 100).')
parser.add_argument('--order', type=int, default = 1,
    help='Use the first order approximation if 1 and second order if 2')
parser.add_argument('--meta-lr', type=float, default=0.001,
    help='Learning rate for the meta-optimizer (optimization of the outer '
    'loss). The default optimizer is Adam (default: 1e-3).')

# Misc
parser.add_argument('--num-workers', type=int, default=8,
    help='Number of workers to use for data-loading (default: 1).')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--use-cuda', action='store_true', default=True)

parser.add_argument('--num-ways', type=int, default=5,)
parser.add_argument('--num-shots', type=int, default=1,)
parser.add_argument('--num-steps', type=int, default=1,)
parser.add_argument('--step-size', type=float, default=0.4,)
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--zero', type=int)
parser.add_argument('--initvar', type=float)
parser.add_argument('--seed', type=int)

args, _ = parser.parse_known_args()

if args.num_shots_test <= 0:
    args.num_shots_test = args.num_shots

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

if True:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    device = torch.device('cuda:{}'.format(args.device) if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logging.debug('Creating folder `{0}`'.format(args.output_folder))

    folder = os.path.join(args.output_folder,
                          time.strftime('%Y-%m-%d_%H%M%S'))
    
    if args.order == 1:
        use_first_order = True
        str_first = "FO"
        print("FO")
    elif args.order == 2: 
        use_first_order = False
        str_first = "SO"
        print("SO")
    folder = os.path.join(args.output_folder, "{}w{}s_{}_zero{}_initvar{}_seed{}_{}".format(args.num_ways, 
                                                                                     args.num_shots, 
                                                                                     str_first,
                                                                                     args.zero, 
                                                                                     args.initvar, 
                                                                                     args.seed,
                                                                                     time.strftime('%H%M%S')))
    os.makedirs(folder)
    logging.debug('Creating folder `{0}`'.format(folder))

    args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))
    # Save the configuration in a config.json file
    with open(os.path.join(folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    logging.info('Saving configuration file in `{0}`'.format(
                 os.path.abspath(os.path.join(folder, 'config.json'))))

    data_folder = "./omniglot_data/"
    benchmark = get_benchmark_by_name(args.dataset,
                                      data_folder,
                                      args.num_ways,
                                      args.num_shots,
                                      args.num_shots_test,
                                      hidden_size=args.hidden_size)

    meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    
    meta_val_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)
    
    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr)
    metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                            meta_optimizer,
                                            first_order=use_first_order,
                                            num_adaptation_steps=args.num_steps,
                                            step_size=args.step_size,
                                            loss_function=benchmark.loss_function,
                                            device=device)

    best_value = None

    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    start = time.time()
    text = []
    
    metalearner.model.classifier.weight.data = metalearner.model.classifier.weight.data * args.initvar
    print(folder)
    for epoch in range(args.num_epochs):
        
        if args.zero == 0: 
            metalearner.model.classifier.weight.data = torch.zeros_like(metalearner.model.classifier.weight.data)
            metalearner.model.classifier.bias.data = torch.zeros_like(metalearner.model.classifier.bias.data)
        
        metalearner.train(meta_train_dataloader,
                          max_batches=args.num_batches,
                          verbose=args.verbose,
                          desc='Training',
                          leave=False)
        
        # Get the testing accuracy
        meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          pin_memory=True)
        results = metalearner.evaluate(meta_test_dataloader,
                                       max_batches=500,
                                       verbose=args.verbose,
                                       desc=epoch_desc.format(epoch + 1))
        del meta_test_dataloader
        
        text.append(results["accuracies_after"])
        print("Epoch: "epoch, results, np.round(time.time()-start))
        start = time.time()

        # Get the testing accuracy after zeroing
        meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          pin_memory=True)
        metalearner__ = copy.deepcopy(metalearner)
        metalearner__.model.classifier.weight.data = torch.zeros_like(metalearner.model.classifier.weight.data)
        metalearner__.model.classifier.bias.data = torch.zeros_like(metalearner.model.classifier.bias.data)
        results = metalearner__.evaluate(meta_test_dataloader,
                                       max_batches=500,
                                       verbose=args.verbose,
                                       desc=epoch_desc.format(epoch + 1))
        text.append(results["accuracies_after"])
        print("Epoch: "epoch, results, np.round(time.time()-start))
        start = time.time()
        del meta_test_dataloader
        del metalearner__

        txt_path = os.path.join(folder, "test_E{}.csv".format(epoch))
        df = pd.DataFrame(text)
        df.to_csv(txt_path,index=False)

        # Save best model
        if 'accuracies_after' in results:
            if (best_value is None) or (best_value < results['accuracies_after']):
                best_value = results['accuracies_after']
                save_model = True
        elif (best_value is None) or (best_value > results['mean_outer_loss']):
            best_value = results['mean_outer_loss']
            save_model = True
        else:
            save_model = False

        if save_model and (args.output_folder is not None):
            with open(args.model_path, 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)
        if epoch % 50 == 0 or (epoch+1) % 50 == 0:
            model_path_temp = os.path.abspath(os.path.join(folder, 'model_E{}.th'.format(epoch)))
            with open(model_path_temp, 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)

    if hasattr(benchmark.meta_train_dataset, 'close'):
        benchmark.meta_train_dataset.close()
        benchmark.meta_val_dataset.close()
