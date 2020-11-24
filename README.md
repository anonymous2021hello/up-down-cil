## Up-down-cil branch

## Requirements
- Python 3.8
- Pytorch 1.6

## Prepare data
1. Please use **git clone --recurse-submodules** to clone this repository and remember to follow initialization steps in coco-caption/README.md. Remenber to download the "captions_robust_val_test.json" from [link](https://pan.baidu.com/s/1zt9LhEqrWM-dJkQ5mrG5VQ)(password:6666) and place it under coco-caption/annotations/
2. Download the preprocessd dataset from this [link](https://pan.baidu.com/s/1rGX-18JJGq9WmDCZ_saidw) 
(password:6666) and extract it to data/.
3. Please follow this [instruction](https://github.com/ruotianluo/self-critical.pytorch/blob/master/data/README.md#convert-from-peteanderson80s-original-file) to prepare the bottom-up features and place them under data/mscoco/.
4. Download the pretrained models from this [link](https://pan.baidu.com/s/19idYT3qynu8MzKLcULS9jg)(password:6666) and extract them to log/.

## Evaluation
To reproduce the results reported in the Table 1 for up-dowm model, just run

```
bash eval_up-down.sh
```

## Training
For example, training the up-down model with cexe_weight=0.5 on Robust-COCO dataset
```
python train.py  --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log/cexe-sup-kl-w0.5  --id  cexe-sup-kl-w0.5  --cexe_weight  0.5  --dataset  robust-coco
```

## Acknowledgements
This repository is built upon [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).
