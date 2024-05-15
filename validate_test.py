import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl

from modules.evaluation import calculate_metrics, accuracy_score, recall_score, precision_score, f1_score


def get_model_pred(model: nn.Module, graph: dgl.DGLGraph,
                   data_loader: DataLoader,
                   sampler: dgl.dataloading.MultiLayerFullNeighborSampler,
                   args):
    """
        Get the model prediction.

        Args:
            `model`: the model to be evaluated
            `graph`: the graph
            `data_loader`: the data loader
            `sampler`: the sampler
            `args`: the arguments

        Returns:
            `pred_list`: the prediction list
            `label_list`: the label list
    """

    model.eval()

    pred_list = []
    label_list = []

    with torch.no_grad():
        for node_idx in data_loader:
            # get blocks of nodes
            _, _, blocks = sampler.sample_blocks(
                graph, node_idx.to(args['device']))

            pred = model(blocks)
            target = blocks[-1].dstdata['label']

            pred_list.append(pred.detach())
            label_list.append(target.detach())

        pred_list = torch.cat(pred_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        # exponential the prediction: convert from log-probability to probability
        # but only for the second dimension: the positive class (fraud class)
        pred_list = pred_list.exp()[:, 1]

    return pred_list, label_list


def validate_and_test(epoch, model: nn.Module,  graph: dgl.DGLGraph,
                   valid_loader: DataLoader, test_loader: DataLoader,
                   sampler: dgl.dataloading.MultiLayerFullNeighborSampler, args):
    """
        Validate and test the model.
        Implementations of the `val_epoch` function in the ConsisGAD paper.

        Args:
            `epoch`: the epoch number
            `model`: the model to be evaluated
            `graph`: the graph
            `valid_loader`: the valid data loader
            `test_loader`: the test data loader
            `sampler`: the sampler
            `args`: the arguments

        Returns:
            `val_results`: the dictionary of validation results
            `test_results`: the dictionary of test results
    """

    val_results = {}
    val_pred, val_true = get_model_pred(model, graph,
                                         valid_loader, sampler, args)

    val_roc, val_pr, val_ks, val_acc, val_r, val_p, val_f1, val_thre = calculate_metrics(
        val_pred, val_true)
    val_results['auc-roc'] = val_roc
    val_results['auc-pr'] = val_pr
    val_results['ks-statistics'] = val_ks
    val_results['accuracy'] = val_acc
    val_results['recall'] = val_r
    val_results['precision'] = val_p
    val_results['macro-f1'] = val_f1


    test_results = {}
    test_pred, test_true = get_model_pred(model, graph,
                                           test_loader, sampler, args)

    test_roc, test_pr, test_ks, _, _, _, _, _ = calculate_metrics(
        test_pred, test_true)
    test_results['auc-roc'] = test_roc
    test_results['auc-pr'] = test_pr
    test_results['ks-statistics'] = test_ks

    test_pred = test_pred.cpu().numpy()
    test_true = test_true.cpu().numpy()
    # use the best threshold from the validation set 
    guessed_label = np.zeros_like(test_true)
    guessed_label[test_pred > val_thre] = 1

    test_results['accuracy'] = accuracy_score(test_true, guessed_label)
    test_results['recall'] = recall_score(test_true, guessed_label)
    test_results['precision'] = precision_score(test_true, guessed_label)
    test_results['macro-f1'] = f1_score(test_true,
                                        guessed_label, average='macro')

    return val_results, test_results