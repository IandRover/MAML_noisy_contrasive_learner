To reproduce the results, we offer the code for "MAML is a noisy contrastive learner in classification" accepted by ICLR 2022.

## 1. Specification of dependencies

### 1.1 Setup
To avoid conflict with your current setup, please create and activate a virtual environment. 
The author implements the code on Python 3.7 platform. Please install the required packages with ```pip install -r requirements.txt```.

## 2 Building up dataset

### 2.1 miniImagenet
For experiments on miniImagenet dataset, please manually download the miniImagenet dataset [here](https://drive.google.com/open?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk) to `./data_miniImagenet` folder and unzip it. ([ref1](https://github.com/dragen1860/MAML-Pytorch) and [ref2](https://github.com/dragen1860/LearningToCompare-Pytorch/issues/4))

```
cd ./data_miniImagenet
gdown https://drive.google.com/u/0/uc?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nukt
unzip mini-imagenet.zip
```

### 2.2 Omniglot
For experiments on Omniglot dataset, the dataset will be download automatically.

## 3. Training code
The four folders below provide the code to reproduce the results in Figure.3 ~ Figure.6.
```
./omniglot_main
./omniglot_memorization
./miniimagenet_main
./miniimagenet_memorization
```
To run the code, one can run ```experiment_command.txt``` inside each folders to get the results. To faithfully reproduce the results, it is worth noted that we use random seed of 222-225.
```
cd ./miniimagenet
. experiment_command.txt
```

To visualize the contrastiveness of the MAML algorithm, please go to ```./contrastiveness_visualization``` and run ```./contrastivemess_visualization.py``` to train models and calculate the cosine similarities. One can also refer to the ipython notebook to directly visualize the results.
