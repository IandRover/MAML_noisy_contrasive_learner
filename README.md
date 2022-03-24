# MAML Is a Noisy Contrastive Learner in Classification (ICLR 2022)

| [Openreview](https://openreview.net/forum?id=LDAwu17QaJz) | [Arxiv](https://arxiv.org/abs/2106.15367) | [poster](https://github.com/IandRover/IandRover.github.io/blob/gh-pages/data/papers/ICLR2022_poster.pdf) | 

<!-- [Talk](https://recorder-v3.slideslive.com/#/share?share=62802&s=f6598308-e36c-44d2-9840-58554d3a04c0) | -->

I also explain our paper in detail at [Medium](https://medium.com/@aaronkao/paper-maml-is-a-noisy-contrastive-learner-in-classification-de63001ad3e0).

## 1. Specification of dependencies

### 1.1 Setup
To avoid conflict with your current setup, please create and activate a virtual environment and install the required packages. For example:
```
conda create --name noisyMAML python=3.7
conda activate noisyMAML
pip install -r requirements.txt
```

## 2. Building up dataset

### 2.1 mini-ImageNet
For experiments on mini-ImageNet dataset, please manually download the mini-ImageNet dataset [here](https://drive.google.com/open?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk) to `./data/miniimagenet` folder and unzip it. ([ref1](https://github.com/dragen1860/MAML-Pytorch) and [ref2](https://github.com/dragen1860/LearningToCompare-Pytorch/issues/4))

```
cd ./data/miniimagenet
gdown https://drive.google.com/u/0/uc?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk
unzip mini-imagenet.zip
```

### 2.2 Omniglot
For experiments on Omniglot dataset, the dataset will be download automatically.

## 3.1 Cosine similarity analysis
To visualize the contrastiveness of the MAML algorithm, please go to ```./cos_sim_analysis``` and run ```./contrastivemess_visualization.py``` to train models and calculate the cosine similarities. You can refer to the ipython notebook to directly visualize the results.

## 3.2 Training code
In ```./omniglot``` and ```./miniimagenet``` folders, codes that reproduce the results are provided. 

To obtain the main results, please run ```script.txt```.

To explore how the zeroing trick mitigates the memorization problem, please run ```script_memorization.txt```.

## 3.3 Experimental results
For reproducibility, we also provide our experimental results and our visualization code in ```./figure_reproduction```.

## Acknowledgement
The codes are adapted from [this repository](https://github.com/dragen1860/MAML-Pytorch).

## Citation
```
@InProceedings{kao2022maml,
  title={MAML Is a Noisy Contrastive Learner in Classification},
  author={Kao, Chia-Hsiang and Chiu, Wei-Chen and Chen, Pin-Yu},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  year={2022}
}
```
