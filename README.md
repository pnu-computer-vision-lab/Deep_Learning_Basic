# Simple PyTorch Baseline for Training

## TODO
### utils
- [ ] F1
- [ ] Accuracy
- [x] config
- [x] config param save
### log
- [ ] tensorboard logging
- [ ] csv logging
### trainer
- [ ] logging pesudo code
- [x] train pesudo code
- [ ] validation pesudo code
- [ ] model save
- [ ] load model for continual learning
- [ ] early stopping
- [ ] metric save
### other 
- [ ] Pytorch lightning


## Setting
miniconda, anaconda, python venv

**my server setting**
```sh
conda create base -n python=3.9
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## How to use

1. clone the repository
2. make virtual enviornmnet
3. make custom dataset
4. make model
5. make own your transformer 
6. fill trainer setup
7. fill the train, _valid function


