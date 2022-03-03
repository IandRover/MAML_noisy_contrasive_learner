Title: MAML is a Noisy Contrastive Learner in Classification

| ICLR'22: [paper](https://openreview.net/forum?id=LDAwu17QaJz) | Arxiv: <span style="color:blue">preparing</span> | Talk: <span style="color:blue">preparing</span> |

## 1. Specification of dependencies

### 1.1 Setup
To avoid conflict with your current setup, please create and activate a virtual environment and install the required packages. For example:
```
conda create --name noisyMAML python=3.7
conda activate noisyMAML
pip install -r requirements.txt
```

## 2. Building up dataset

### 2.1 MiniImagenet
For experiments on miniImagenet dataset, please manually download the MiniImagenet dataset [here](https://drive.google.com/open?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk) to `./data_miniImagenet` folder and unzip it. ([ref1](https://github.com/dragen1860/MAML-Pytorch) and [ref2](https://github.com/dragen1860/LearningToCompare-Pytorch/issues/4))

```
cd ./data_miniImagenet
gdown https://drive.google.com/u/0/uc?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nukt
unzip mini-imagenet.zip
```

### 2.2 Omniglot
For experiments on Omniglot dataset, the dataset will be download automatically.

## 3.1 Cosine similarity analysis
To visualize the contrastiveness of the MAML algorithm, please go to ```./contrastiveness_visualization``` and run ```./contrastivemess_visualization.py``` to train models and calculate the cosine similarities. You can refer to the ipython notebook to directly visualize the results.

## 3.2 Training code
The folders below provide the code to reproduce the results in Sec.3.2 to Sec.3.5 in MiniImagenet and Omniglot datasets.
```
./omniglot_main
./miniimagenet_main
```

To explore how the zeroing trick mitigates the memorization effect, please check out at ```./miniimagenet_memorization``` and ```./omniglot_memorization```

To run the code, one can run ```experiment_command.txt``` inside each folders to get the results. For example:
```
cd ./miniimagenet
. experiment_command.txt
```

## 3.3 Experimental results
For reproducibility, we also provide our experimental results and our visualization code in ```./results_and_plots```.
