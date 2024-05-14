from modules.augment import fixed_augmentation
from modules.utils import high_quality_nodes
from modules.regularization import l2_regularization

import torch
import torch.nn.functional as F

def nll_loss(pred, target, pos_w, reduction='mean', device='cuda'):
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
        - weight-decay: weight decay
    """
    model.train()
    num_iters = config['num_iters'] # This equals to len(dl) / batch_size
    sampler, attn_drop, ad_optim = augmentor
    
    unlabel_loader_iter = iter(unlabel_loader) 
    label_loader_iter = iter(label_loader)
    
    for idx in range(num_iters): # iterate over batches of dataloader
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
                                                    sampler, aug_type='none') # get blocks of unlabeled nodes
                
                weak_inter_results = model(u_blocks, update_bn=False, return_logits=True) # (6) calculate the embeddings
                weak_h = torch.stack(weak_inter_results, dim=1)
                weak_h = weak_h.reshape(weak_h.shape[0], -1)
                weak_logits = model.proj_out(weak_h) # (7) conduct logits prediction

            pseudo_labels, u_mask = high_quality_nodes(logits=weak_logits,
                                                       normal_th=config['normal-th'],
                                                         fraud_th=config['fraud-th']) # (8) get high-quality nodes

            model.train()
            attn_drop.train()

            ### --- Optimize learnable data augmentation modules - Freeze the model optimization --- ###
            for param in model.parameters():
                param.requires_grad = False
            for param in attn_drop.parameters():
                param.requires_grad = True

            # add some stochastic noise at the input state
            _, _, u_blocks = fixed_augmentation(graph, unlabel_idx.to(config['device']), sampler, aug_type='drophidden')

            inter_results = model(u_blocks, update_bn=False, return_logits=True) # calculate the embeddings
            dropped_results = [inter_results[0]] 
            for i in range(1, len(inter_results)):
                dropped_results.append(attn_drop(inter_results[i])) # apply the learnable data augmentation (masking)

            h = torch.stack(dropped_results, dim=1)
            h = h.reshape(h.shape[0], -1)
            logits = model.proj_out(h) # (13) conduct logits prediction
            u_pred = logits.log_softmax(dim=-1)
            
            consistency_loss = nll_loss(u_pred, pseudo_labels, pos_w=1.0, reduction='none', device=config['device']) # (14) calculate the consistency loss
            consistency_loss = torch.mean(consistency_loss * u_mask)

            diversity_loss = F.pairwise_distance(weak_h, h) # (14) calculate the diversity loss
            
            # (15) total loss of the learnable data augmentation module + regularization
            total_loss = config['consis-weight'] * consistency_loss - diversity_loss + config['weight-decay'] * l2_regularization(attn_drop)
            
            ad_optim.zero_grad()
            total_loss.backward()
            ad_optim.step()
            
            ### --- Optimize the model - Freeze the learnable data augmentation module --- ###
            for param in model.parameters():
                param.requires_grad = True
            for param in attn_drop.parameters():
                param.requires_grad = False

            inter_results = model(u_blocks, update_bn=False, return_logits=True) # (17) calculate the embeddings
            dropped_results = [inter_results[0]]

            # calculate embeddings of the augmented data
            for i in range(1, len(inter_results)):
                dropped_results.append(attn_drop(inter_results[i], in_eval=True)) # freeze the optimization of the learnable data augmentation module

            h = torch.stack(dropped_results, dim=1)
            h = h.reshape(h.shape[0], -1)
            logits = model.proj_out(h)
            u_pred = logits.log_softmax(dim=-1)

            unsup_loss = nll_loss(u_pred, pseudo_labels, pos_w=1.0, reduction='none', device=config['device'])
            unsup_loss = torch.mean(unsup_loss * u_mask) # consistency loss (eq3) in the framework

        else:
            unsup_loss = 0.0

        _, _, s_blocks = fixed_augmentation(graph, label_idx.to(config['device']), sampler, aug_type='none') # sample blocks of labeled nodes
        s_pred = model(s_blocks)
        s_target = s_blocks[-1].dstdata['label'] 
            
        sup_loss, _ = loss_func(s_pred, s_target) # (19) CE loss on labeled nodes

        loss = sup_loss + unsup_loss + config['weight-decay'] * l2_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()     