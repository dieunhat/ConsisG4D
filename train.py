from modules.augment import fixed_augmentation
from modules.utils import high_quality_nodes
from modules.regularization import l2_regularization

import torch
import torch.nn.functional as F


def nll_loss(pred, target, pos_w: float=1.0, reduction='mean', device='cuda'):
    """
    Args:
        pred: prediction tensor
        target: target tensor
        pos_w: positive weight
        reduction: reduction method
    Returns:
        loss_value: loss value
    """
    weight_tensor = torch.tensor([1., pos_w]).to(pred.device)
    loss_value = F.nll_loss(pred, target.long(), weight=weight_tensor,
                            reduction=reduction)

    if device == 'cuda' or device == 'mps':
        return loss_value
    else:
        loss_value = loss_value.cpu().detach().numpy()
        return loss_value


def training_paradigm(model, loss_func, graph,
                      label_loader, unlabel_loader,
                      optimizer, augmentor, config):
    """
    Notes: `config` argument is expected to contains the following to be used in this function:
        - device: device to run the model (cuda/cpu/mps)
        - num_iters: number of iterations
        - normal-th: normal threshold
        - fraud-th: fraud threshold
        - consis-weight: consistency weight
        - learnable-weight-decay: weight decay for learnable augmentation
        - weight-decay: weight decay
    """
    model.train()
    num_iters = config['num_iters']  # This equals to len(dl) / batch_size
    sampler, attn_drop, ad_optim = augmentor

    unlabel_loader_iter = iter(unlabel_loader)
    label_loader_iter = iter(label_loader)

    for idx in range(num_iters):  # iterate over batches of dataloader
        try:
            label_idx = label_loader_iter.__next__()
        except:
            label_loader_iter = iter(label_loader)
            label_idx = label_loader_iter.__next__()
        try:
            unlabel_idx = unlabel_loader_iter.__next__()
        except:
            unlabel_loader_iter = iter(unlabel_loader)
            unlabel_idx = unlabel_loader_iter.__next__()

            ###
            model.eval()
            with torch.no_grad():
                _, _, u_blocks = fixed_augmentation(graph, unlabel_idx.to(config['device']),
                                                    sampler, aug_type='none')  # get blocks of unlabeled nodes

                # (6) calculate the embeddings
                weak_inter_results = model(
                    u_blocks, update_bn=False, return_logits=True)
                weak_h = torch.stack(weak_inter_results, dim=1)
                weak_h = weak_h.reshape(weak_h.shape[0], -1)
                # (7) conduct logits prediction
                weak_logits = model.proj_out(weak_h)

            pseudo_labels, u_mask = high_quality_nodes(logits=weak_logits,
                                                       normal_th=config['normal-th'],
                                                       fraud_th=config['fraud-th'])  # (8) get high-quality nodes

            model.train()
            attn_drop.train()

            ### --- Optimize learnable data augmentation modules - Freeze the model optimization --- ###
            for param in model.parameters():
                param.requires_grad = False
            for param in attn_drop.parameters():
                param.requires_grad = True

            # add some stochastic noise at the input state
            _, _, u_blocks = fixed_augmentation(graph, unlabel_idx.to(
                config['device']), sampler, aug_type='drophidden')

            # calculate the embeddings
            inter_results = model(
                u_blocks, update_bn=False, return_logits=True)
            dropped_results = [inter_results[0]]
            for i in range(1, len(inter_results)):
                # apply the learnable data augmentation (masking)
                dropped_results.append(attn_drop(inter_results[i]))

            h = torch.stack(dropped_results, dim=1)
            h = h.reshape(h.shape[0], -1)
            logits = model.proj_out(h)  # (13) conduct logits prediction
            u_pred = logits.log_softmax(dim=-1)

            # (14) calculate the consistency loss
            consistency_loss = nll_loss(
                u_pred, pseudo_labels, pos_w=1.0, reduction='none', device=config['device'])
            consistency_loss = torch.mean(torch.tensor(consistency_loss) * u_mask)
            
            # (14) calculate the diversity loss
            diversity_loss = F.pairwise_distance(weak_h, h)
            diversity_loss = torch.mean(diversity_loss * u_mask)

            # (15) total loss of the learnable data augmentation module + regularization
            total_loss = config['consis-weight'] * consistency_loss - \
                diversity_loss + config['trainable-weight-decay'] * \
                l2_regularization(attn_drop)

            ad_optim.zero_grad()
            total_loss.backward()
            ad_optim.step()

            ### --- Optimize the model - Freeze the learnable data augmentation module --- ###
            for param in model.parameters():
                param.requires_grad = True
            for param in attn_drop.parameters():
                param.requires_grad = False

            # (17) calculate the embeddings
            inter_results = model(
                u_blocks, update_bn=False, return_logits=True)
            dropped_results = [inter_results[0]]

            # calculate embeddings of the augmented data
            for i in range(1, len(inter_results)):
                # freeze the optimization of the learnable data augmentation module
                dropped_results.append(
                    attn_drop(inter_results[i], in_eval=True))

            h = torch.stack(dropped_results, dim=1)
            h = h.reshape(h.shape[0], -1)
            logits = model.proj_out(h)
            u_pred = logits.log_softmax(dim=-1)

            unsup_loss = nll_loss(
                u_pred, pseudo_labels, pos_w=1.0, reduction='none', device=config['device'])
            # consistency loss (eq3) in the framework
            unsup_loss = torch.mean(torch.tensor(unsup_loss) * u_mask)

        else:
            unsup_loss = 0.0

        _, _, s_blocks = fixed_augmentation(graph, label_idx.to(
            config['device']), sampler, aug_type='none')  # sample blocks of labeled nodes
        s_pred = model(s_blocks)
        s_target = s_blocks[-1].dstdata['label']

        # (19) CE loss on labeled nodes
        sup_loss = loss_func(s_pred, s_target)

        loss = sup_loss + unsup_loss + \
            config['weight-decay'] * l2_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
