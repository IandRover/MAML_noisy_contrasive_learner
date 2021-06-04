For reproduce the results, we offer the code for "MAML is a noisy contrastive learning" submitted for NeurIPS 2021.

## 1. Specification of dependencies

### 1.1 Setup
To avoid conflict with your current Python setup, please create and activate a virtual environment. The author implements the code on Python 3.7 platform
Then install the required packages with ```pip install -r requirements.txt```.

## 2 Build up dataset

### 2.1 miniImagenet
For experiments regarding miniImagenet, one have to manually download the data.
Please downlod the miniImagenet dataset [here](https://drive.google.com/open?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk)  to `data_miniImagenet` folder and unzip it. 
([ref1](https://github.com/dragen1860/MAML-Pytorch) and [ref2](https://github.com/dragen1860/LearningToCompare-Pytorch/issues/4))
```
cd ./data_miniImagenet
gdown https://drive.google.com/u/0/uc?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nukt
unzip mini-imagenet.zip
```

### 2.2 Omniglot
For experiments regarding Omniglot, the data will be download automatically.

## 3. Training code
The four folders below provide the code to reproduce the results in Figure. 3~Figure. 6.
```
./omniglot
./omniglot_memorization_
./miniimagenet_main
./miniimagenet_memorization
```
To run the code, one can run ```experiment_command.txt``` inside each folders to get the results. To faithfully reproduce the results, one has to change the random seed to 222-225.
```
cd ./miniimagenet
. experiment_command.txt
```

To visualize the contrastiveness of the MAML algorithm, please go to ```./contrastiveness_visualization``` and run ```./contrastivemess_visualization.py``` to train models and calculate the cosine similarities. One can also refer to the ipython notebook to directly visualize the results.


## 🎉 Additional awesome resources for releasing research code

### Hosting pretrained models files

1. [Zenodo](https://zenodo.org) - versioning, 50GB, free bandwidth, DOI, provides long-term preservation
2. [GitHub Releases](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository) - versioning, 2GB file limit, free bandwidth
3. [OneDrive](https://www.onedrive.com/) - versioning, 2GB (free)/ 1TB (with Office 365), free bandwidth
4. [Google Drive](https://drive.google.com) - versioning, 15GB, free bandwidth
5. [Dropbox](https://dropbox.com) - versioning, 2GB (paid unlimited), free bandwidth
6. [AWS S3](https://aws.amazon.com/s3/) - versioning, paid only, paid bandwidth
7. [huggingface_hub](https://github.com/huggingface/huggingface_hub) - versioning, no size limitations, free bandwidth
8. [DAGsHub](https://dagshub.com/) - versioning, no size limitations, free bandwith
9. [CodaLab Worksheets](https://worksheets.codalab.org/) - 10GB, free bandwith
 
### Managing model files

1. [RClone](https://rclone.org/) - provides unified access to many different cloud storage providers

### Standardized model interfaces

1. [PyTorch Hub](https://pytorch.org/hub/)
2. [Tensorflow Hub](https://www.tensorflow.org/hub)
3. [Hugging Face NLP models](https://huggingface.co/models)
4. [ONNX](https://onnx.ai/)

### Results leaderboards

1. [Papers with Code leaderboards](https://paperswithcode.com/sota) - with 4000+ leaderboards
2. [CodaLab Competitions](https://competitions.codalab.org/) - with 450+ leaderboards
3. [EvalAI](https://eval.ai/) - with 100+ leaderboards
4. [NLP Progress](https://nlpprogress.com/) - with 90+ leaderboards
5. [Collective Knowledge](https://cKnowledge.io/reproduced-results) - with 40+ leaderboards
6. [Weights & Biases - Benchmarks](https://www.wandb.com/benchmarks) - with 9+ leaderboards

### Making project pages

1. [GitHub pages](https://pages.github.com/)
2. [Fastpages](https://github.com/fastai/fastpages)

### Making demos, tutorials, executable papers

1. [Google Colab](https://colab.research.google.com/)
2. [Binder](https://mybinder.org/)
3. [Streamlit](https://github.com/streamlit/streamlit)
4. [CodaLab Worksheets](https://worksheets.codalab.org/)

## Contributing

If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at hello@paperswithcode.com or open an issue on this GitHub repository. 

All contributions welcome! All content in this repository is licensed under the MIT license.
'+
