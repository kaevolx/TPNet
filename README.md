<img width="700" height="300" alt="three" src="https://github.com/user-attachments/assets/f0cce8c4-3d0c-40db-93a6-28064a987568" />

## Setup
Clone the repository and set up the environment:

```python
conda create -n TPNet python=3.8
conda activate TPNet
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.3.1
conda install pytorch-lightning==2.0.3
pip install lanelet2==1.2.1
```

## Dataset
If you can provide proof of authorization for the exiD dataset, please contact us to obtain the dataset.

## Training
Data preprocessing may take several hours the first time you run this project. Training on a RTX 4090 GPU.
```python
python train.py --root /path/to/data_root/ --train_batch_size 8 --val_batch_size 8 --front_loss_weight 0.4 --devices 1
```

## Validation
```python
python val.py --root /path/to/data_root/ --val_batch_size 8 --devices 1
```

## Testing
```python
python test.py --root /path/to/data_root/ --test_batch_size 8 --devices 1
```


## Acknowladgement
Please also support the representation learning work on which this work is based:

HPNet：[HPNet: Dynamic Trajectory Forecasting with Historical Prediction Attention](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_HPNet_Dynamic_Trajectory_Forecasting_with_Historical_Prediction_Attention_CVPR_2024_paper.html)  
&emsp;&emsp;&emsp;&emsp;[code](https://github.com/XiaolongTang23/HPNet)
