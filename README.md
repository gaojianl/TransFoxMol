# KEFAT: 

This is a Pytorch implementation of the following paper: 

Weihua Hu*, Bowen Liu*, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, Jure Leskovec. Strategies for Pre-training Graph Neural Networks. ICLR 2020.
[arXiv](https://arxiv.org/abs/1905.12265) [OpenReview](https://openreview.net/forum?id=HJlWWJSFDH) 

If you make use of the code/experiment in your work, please cite our paper.

## Installation
You can just execute following command to create the conda environment.
'''
conda create --name kefat --file requirements.txt
'''

## Usage

#### 1. Dataset preparation
Put your raw csvfile(`DATASET_NAME.csv`) in `dataset/raw/`.
```
python molnetdata.py --moldata DATASET_NAME --task clas --ncpu 10
```
This will save the processed dataset in `dataset/processed/`.

#### 2. Training
```
python run.py --mode train \
               --moldata DATASET_NAME \
               --task clas \
               --device cuda:0 \
               --batch_size 32 \
               --train_epoch 50 \
               --lr 0.0005 \
               --valrate 0.1 
               --testrate 0.1 \
               --seed 426 \
               --fold 3 \
               --dropout 0.05 \
               --scaffold True \
               --attn_head 6 \
               --attn_layers 2 \
               --output_dim 256 \
               --D 4 \
               --disw 1.5 
```
This will save the resulting model in `log/checkpoint/`.

#### 3. Testing
```
python run.py --mode test \
               --moldata DATASET_NAME \
               --task clas \
               --device cuda:0 \
               --batch_size 32 \
               --seed 426 \
               --attn_head 6 \
               --attn_layers 2 \
               --output_dim 256 \
               --D 4 \
               --disw 1.5 \
               --pretrain log/checkpoint/XXXX.pkl
```
This will load the model in `log/checkpoint/` to make predictions and the results are saved in `log/`.

#### 3. Hyper-parameter searching
```
python run.py --mode search \
               --moldata DATASET_NAME \
               --task clas \
               --device cuda:0 \
               --train_epoch 50 \
               --valrate 0.1 \
               --testrate 0.1 \
               --seed 426 \
               --fold 3 \
               --scaffold True 
```
This will return the best hyper-params.
