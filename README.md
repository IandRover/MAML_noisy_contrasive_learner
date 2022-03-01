Paper: MAML is a noisy contrastive learner in classification (ICLR'22 poster)
Talk: Preparing

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

## 3.1 Code for cosine similarity analysis
To visualize the contrastiveness of the MAML algorithm, please go to ```./contrastiveness_visualization``` and run ```./contrastivemess_visualization.py``` to train models and calculate the cosine similarities. One can also refer to the ipython notebook to directly visualize the results.

## 3.2 Training code
The four folders below provide the code to reproduce the results in Sec.3.2 to Sec.3.5.
```
./omniglot_main
./miniimagenet_main
```

For Fig.13 and Fig.17, please check out at ```./miniimagenet_memorization``` and ```./omniglot_memorization```

To run the code, one can run ```experiment_command.txt``` inside each folders to get the results.
```
cd ./miniimagenet
. experiment_command.txt
```


