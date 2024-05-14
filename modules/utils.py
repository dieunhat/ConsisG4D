import torch

def high_quality_nodes(logits: torch.tensor, normal_th: float = 0.05, fraud_th: float = 0.85):
    # prediction of the model
    #u_pred_log.shape = (192, 2)
    u_pred_log = logits.log_softmax(dim=-1)

    #u_pred[i]: probability that node i-th is an abnormal node
    u_pred = u_pred_log.exp()[:, 1]

    pseudo_labels = torch.ones_like(u_pred).long()

    # neg_tar: nodes that have p(abnormal) <= 0.05 or p(normal) >= 0.95
    neg_tar = (u_pred <= (normal_th/100.)).bool()

    #pos_tar: nodes that have p(abnormal) >= 0.85
    pos_tar = (u_pred >= (fraud_th/100.)).bool()
    
    pseudo_labels[neg_tar] = 0
    pseudo_labels[pos_tar] = 1

    # u_mask: set of high-quality nodes
    # 1 node is considered high-quality if and only if p(normal) >= 0.95 or p(abnormal) >= 0.85
    u_mask = torch.logical_or(neg_tar, pos_tar)
    return pseudo_labels, u_mask