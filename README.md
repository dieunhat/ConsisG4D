# Graph Anomaly Detection on Amazon dataset using ConsisGAD

This repository contains the implementation of the ConsisGAD model for graph anomaly detection on the Amazon dataset. The model is based on the paper [Consistency Training with Learnable Data Augmentation for Graph Anomaly Detection with Limited Supervision](https://openreview.net/pdf?id=elMKXvhhQ9).

## Set up and usage
Install required packages by running the following command:
```bash
pip3 install -r requirements.txt
``` 

Wandb setup:
```bash
wandb login
```
Enter the API key when prompted and replace the `entity` in the `wandb.init()` function with your username (in `main.py`).

To start training, execute the following command:
```bash
python3 main.py
```

## Results
Configuration details used for training the model can be found in `config.yaml`. The model is trained on CPU and the whole training process takes approximately 30 minutes, proving its efficiency and lightweightness. The comparison of the results with the paper is as follows:

| Metric | Paper | Ours (training on high-quality nodes) | Ours (training on full unlabeled nodes) |
| --- | --- | --- | --- |
| AUC-ROC | 93.91 | 91.897 | 91.95 |
| AUC-PR | 83.33 | 79.98 | 80.17 |
| Macro-F1 | 90.03 | 88.67 | 88.72 |

Plots of the training and validation loss curves and metrics can be found in folder `logs/`.




