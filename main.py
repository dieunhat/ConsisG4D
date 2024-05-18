import os
import dgl
import sys
import yaml
import time
import torch
import numpy as np
from torch import optim
from train import nll_loss, training_paradigm
from validate_test import validate_and_test
from modules.model import GNN_backbone
from modules.augment import SoftAttentionDrop
from modules.dataloader import get_index_loader_test
import wandb

def create_model(config, e_ts):
    if config['model'] == 'backbone':
        tmp_model = GNN_backbone(in_feats=config['node-in-dim'], hidden_feats=config['hidden-dim'], out_feats=config['node-out-dim'], 
                                 num_layers=config['num-layers'], e_types=e_ts, input_drop=config['input-drop'], hidden_drop=config['hidden-drop'], 
                                 mlp_drop=config['mlp-drop'], mlp12_dim=config['mlp12-dim'], mlp3_dim=config['mlp3-dim'], bn_type=config['bn-type'])
    else:
        raise
    tmp_model.to(config['device'])
            
    return tmp_model

def store_model(my_model, name, config):
    file_path = os.path.join(config['model-path'], f"{name}.pth")
    torch.save(my_model.state_dict(), file_path)

def run_model(config):
    # define dataset
    graph, label_loader, valid_loader, test_loader, unlabel_loader = get_index_loader_test(name=config['data-set'], 
                                                                                           batch_size=config['batch-size'], 
                                                                                           unlabel_ratio=config['unlabel-ratio'],
                                                                                           training_ratio=config['training-ratio'],
                                                                                           shuffle_train=config['shuffle-train'], 
                                                                                           to_homo=config['to-homo'])
    
    print("Done getting loader")
    graph = graph.to(config['device'])
    
    config['node-in-dim'] = graph.ndata['feature'].shape[1]
    config['node-out-dim'] = 2

    # define model
    my_model = create_model(config, graph.etypes)

    # define optimizer
    optimizer = optim.Adam(my_model.parameters(), lr=config['lr'], weight_decay=0.0)
    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(config['num-layers'])

    train_epoch = training_paradigm
    attn_drop = SoftAttentionDrop(config).to(config['device'])

    ad_optim = optim.Adam(attn_drop.parameters(), lr=config['trainable-lr'], weight_decay=0.0)
    augmentor = (sampler, attn_drop, ad_optim)

    task_loss = nll_loss
    
    # define the smallest floating point number
    best_val_roc = sys.float_info.min
    best_val_pr = sys.float_info.min
    best_val_f1 = sys.float_info.min
    
    for i in range(config['epochs']):
        print(f"Epoch {i+1}/{config['epochs']}\n")
        train_epoch(my_model, task_loss, graph, label_loader, unlabel_loader, optimizer, augmentor, config)
        val_results, test_results = validate_and_test(my_model, graph, valid_loader, test_loader, sampler, config)
        
        if val_results['auc-pr'] > best_val_pr:
            best_val_pr = val_results['auc-pr']
            best_test_pr = test_results
            
            if config['store-model']:
                store_model(my_model, "pr" ,config)

        if val_results['macro-f1'] > best_val_f1:
            best_val_f1 = val_results['macro-f1']
            best_test_f1 = test_results
            
            if config['store-model']:
                store_model(my_model, "f1" ,config)

        if val_results['auc-roc'] > best_val_roc:
            best_val_roc = val_results['auc-roc']
            best_test_roc = test_results
            
            if config['store-model']:
                store_model(my_model, "roc" ,config)
                
    return list(best_test_pr.values()), list(best_test_f1.values()), list(best_test_roc.values())

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_result(results):
    print(f"AUC-ROC: {results[0]}\nAUC-PR: {results[1]}\nMacro-F1: {results[-1]}\n")
          
if __name__ == "__main__":
    start_time = time.time()
        
    config = get_config()
    if torch.cuda.is_available():
        config['device'] = torch.device('cuda:%d'%(config['device']))
    else:
        config['device'] = torch.device('cpu')
    
    print("Device: ", config['device'])
    set_seed(config['seed'])

    # init wandb
    wandb.init(project='amazon-fraud-detection', 
               entity='dnhat',
               config={
                   "runs": config['runs'],
                    "epochs": config['epochs'],
                    "lr": config['lr'],
               }
               )
    
    final_result_pr, final_result_f1, final_result_roc = run_model(config)
    print(f"Test results (tuned by AUC-PR): {print_result(final_result_pr)}\n-----------------")
    print(f"Test results (tuned by Macro-F1): {print_result(final_result_f1)}\n-----------------")
    print(f"Test results (tuned by AUC-ROC): {print_result(final_result_roc)}\n-----------------")

    print('total time: ', time.time()-start_time)
    