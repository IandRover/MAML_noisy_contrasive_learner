For reproduce the results, we offer the code for "MAML is a noisy contrastive learning" submitted for NeurIPS 2021.

## ✓ ML Code Completeness Checklist

We compiled this checklist by looking at what's common to the most popular ML research repositories. In addition, we prioritized items that facilitate reproducibility and make it easier for others build upon research code.

The ML Code Completeness Checklist consists of five items:

1. **Specification of dependencies**
2. **Training code** 
3. **Evaluation code**
4. **Pre-trained models**
5. **README file including table of results accompanied by precise commands to run/produce those results**

We verified that repositories that check more items on the checklist also tend to have a higher number of GitHub stars. This was verified by analysing official NeurIPS 2019 repositories - more details in the [blog post](https://medium.com/paperswithcode/ml-code-completeness-checklist-e9127b168501). We also provide the [data](notebooks/code_checklist-neurips2019.csv) and [notebook](notebooks/code_checklist-analysis.pdf) to reproduce this analysis from the post. 

NeurIPS 2019 repositories that had all five of these components had the highest number of GitHub stars (median of 196 and mean of 2,664 stars). 

We explain each item on the checklist in detail blow. 

## 1. Specification of dependencies

### 1.1 Setup
To avoid conflict with your current Python setup, please create and activate a virtual environment. The author implements the code on Python 3.7 platform
Then install the required packages with ```pip install -r requirements.txt```.

## 1.2 Build up dataset

### 1.2.1 miniImagenet
For experiments regarding miniImagenet, one have to manually download the data.
Please downlod the miniImagenet dataset [here](https://drive.google.com/open?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk)  to `data_miniImagenet` folder and unzip it. 
([ref1](https://github.com/dragen1860/MAML-Pytorch) and [ref2](https://github.com/dragen1860/LearningToCompare-Pytorch/issues/4))
```
cd ./data_miniImagenet
gdown https://drive.google.com/u/0/uc?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nukt
unzip mini-imagenet.zip
```

### 1.2.2 Omniglot
For experiments regarding Omniglot, the data will be download automatically.

## 2. Training code
The four folders below provide the code to reproduce the results in Figure. 3~Figure. 6.
```
./omniglot
./omniglot_memorization_
./miniimagenet_main
./miniimagenet_memorization
```
To run the code, one can run ```experiment_command.txt``` to get the results. To faithfully reproduce the results, one has to change the random seed to 222-225.
```
cd ./miniimagenet
. experiment_command.txt
```

## 3. Evaluation code

Model evaluation and experiments often depend on subtle details that are not always possible to explain in the paper. This is why including the exact code you used to evaluate or run experiments is helpful to give a complete description of the procedure. In turn, this helps the user to trust, understand and build on your research.

You can provide a documented command line wrapper such as `eval.py` to serve as a useful entry point for your users.

## 4. Pre-trained models

Training a model from scratch can be time-consuming and expensive. One way to increase trust in your results is to provide a pre-trained model that the community can evaluate to obtain the end results. This means users can see the results are credible without having to train afresh.

Another common use case is fine-tuning for downstream task, where it's useful to release a pretrained model so others can build on it for application to their own datasets.

Lastly, some users might want to try out your model to see if it works on some example data. Providing pre-trained models allows your users to play around with your work and aids understanding of the paper's achievements.

## 5. README file includes table of results accompanied by precise command to run to produce those results

Adding a table of results into README.md lets your users quickly understand what to expect from the repository (see the [README.md template](templates/README.md) for an example). Instructions on how to reproduce those results (with links to any relevant scripts, pretrained models etc) can provide another entry point for the user and directly facilitate reproducibility. In some cases, the main result of a paper is a Figure, but that might be more difficult for users to understand without reading the paper. 

You can further help the user understand and contextualize your results by linking back to the full leaderboard that has up-to-date results from other papers. There are [multiple leaderboard services](#results-leaderboards) where this information is stored.  

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
