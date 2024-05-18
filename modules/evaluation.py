import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, average_precision_score
from scikitplot.helpers import binary_ks_curve
import wandb


def find_best_f1(probs: np.ndarray, labels: np.ndarray):
    """ Find the best F1 score and the best threshold to achieve the best F1 score

    ---------
    Args:
        - `probs`: probability array
        - `labels`: label array

    ----------
    Returns:
        - `best_f1`: Best F1 score
        - `best_thre`: Best threshold to achieve the best F1 score
    """

    # Initialize the best F1 score and the best threshold
    best_f1, best_thre = -1., -1.

    # Create an array of thresholds
    thres_arr = np.linspace(0.05, 0.95, 19)

    # Iterate over the thresholds
    for thres in thres_arr:
        # Create one-hot encoded array of predictions
        preds = np.zeros_like(labels)

        # Set the values of the array to 1 if the probability is greater than the threshold
        preds[probs > thres] = 1

        # Calculate the F1 score
        mf1 = f1_score(labels, preds, average='macro')

        # Update the best F1 score and the best threshold
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres

    return best_f1, best_thre


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor):
    """ Evaluate the prediction of the model

    ---------
    Args:
        - `pred`: prediction tensor
        - `target`: target tensor

    ----------
    Returns:
        - `auc_roc`: Area Under the Receiver Operating Characteristic curve score
        - `auc_pr`: Area Under the Precision-Recall curve score
        - `ks_statistics`: Kolmogorov-Smirnov statistics, to determine if two distributions A and B are different
        - `accuracy`: Accuracy score
        - `recall`: Recall score
        - `precision`: Precision score
        - `best_f1`: Best F1 score
        - `best_thre`: Best threshold to achieve the best F1 score
    """

    # Convert the tensors to numpy arrays
    np_pred = pred.cpu().detach().numpy()
    np_target = target.cpu().detach().numpy()

    # Calculate auc_roc score
    auc_roc = roc_auc_score(np_target, np_pred)

    # Calculate auc_pr score
    auc_pr = average_precision_score(np_target, np_pred)

    # calculate ks_statistics
    ks_statistics = binary_ks_curve(np_target, np_pred)[3]

    # Find the best F1 score and the best threshold
    best_f1, best_threshold = find_best_f1(np_pred, np_target)
    predict_labels = (np_pred > best_threshold).astype(int)
    accuracy = accuracy_score(np_target, predict_labels)
    recall = recall_score(np_target, predict_labels)
    precision = precision_score(np_target, predict_labels)

    return auc_roc, auc_pr, ks_statistics, accuracy, recall, precision, best_f1, best_threshold
