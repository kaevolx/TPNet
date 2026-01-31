
## Setup
[MMRotate](https://github.com/open-mmlab/mmrotate) depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection). Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction. Below are quick steps for installation.

```python
conda create -n HPNet python=3.8
conda activate HPNet
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.3.1
conda install pytorch-lightning==2.0.3
```

## Acknowladgement
Please also support the representation learning work on which this work is based:

HPNet：&emsp;&emsp;&emsp;&emsp;[HPNet: Dynamic Trajectory Forecasting with Historical Prediction Attention](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_HPNet_Dynamic_Trajectory_Forecasting_with_Historical_Prediction_Attention_CVPR_2024_paper.html)  
        &emsp;&emsp;&emsp;&emsp;[code](https://github.com/XiaolongTang23/HPNet)
