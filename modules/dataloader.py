import torch
import dgl
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as torch_dataloader

def get_dataset(name: str, raw_dir: str, to_homo: bool = False):
    if name == "amazon":
        # raw_dir: directory that will store the downloaded data
        # random_seed: the seed in splitting the dataset
        # verbose: whether to print out progress information
        amazon_data = dgl.data.FraudAmazonDataset(raw_dir=raw_dir, random_seed=7537, verbose=False)
        graph = amazon_data[0]
        if to_homo:
            # Convert a heterogeneous graph to a homogeneous graph
            # Heterogeneous graph: contains either multiple types of objects or multiple types of links
            # Homogeneous graph: all the nodes represent instances of the same type
            graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])

            # Add self-loops for each node in the graph and return a new graph.
            graph = dgl.add_self_loop(graph)
        return graph

def get_index_loader_test(name: str, batch_size: int, unlabel_ratio: int = 1, training_ratio: float = 1,
                             shuffle_train: bool = True, to_homo: bool = False):
    assert name =='amazon', 'Invalid dataset name'
    
    graph = get_dataset(name, raw_dir = 'data/', to_homo = to_homo)

    # 1194 node
    # index = array([0, 1, 2, ..., 11941, 11942, 11943]) (1 x 11944)
    index = np.arange(graph.num_nodes())

    # labels = array([0, 1, 2, ..., 11941, 11942, 11943]) (1 x 11944)
    labels = graph.ndata['label']

    if name == 'amazon':
        # index = array([3305, 3306, ..., 11943]) (1 x 8639)
        index = np.arange(3305, graph.num_nodes())

    # stratify: Divide subjects into subgroups called strata based on the labels,
    #           each subgroup is randomly sampled using another probability sampling method.
    # train size: 1%
    # random_state: Controls the shuffling applied to the data before applying the split.
    # shuffle = True: Shuffle the data before splitting
    train_nids, valid_test_nids = train_test_split(index, stratify = labels[index],
                                                   train_size = training_ratio/100., random_state = 2, shuffle = True)
    
    # valid size: 32%
    # test size: 67%
    valid_nids, test_nids = train_test_split(valid_test_nids, stratify = labels[valid_test_nids],
                                             test_size = 0.67, random_state = 2, shuffle = True)
    
    train_mask = torch.zeros_like(labels).bool()
    val_mask = torch.zeros_like(labels).bool()
    test_mask = torch.zeros_like(labels).bool()

    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    labeled_nids = train_nids
    unlabeled_nids = np.concatenate([valid_nids, test_nids, train_nids])

    power = 10 if name == 'tfinance' else 16
    
    valid_loader = torch_dataloader(valid_nids, batch_size = 2**power, shuffle = False, drop_last = False, num_workers = 4)
    test_loader = torch_dataloader(test_nids, batch_size = 2**power, shuffle = False, drop_last = False, num_workers = 4)
    labeled_loader = torch_dataloader(labeled_nids, batch_size = batch_size, shuffle = shuffle_train, drop_last = True, num_workers = 0)
    unlabeled_loader = torch_dataloader(unlabeled_nids, batch_size = batch_size * unlabel_ratio, shuffle = shuffle_train, 
                                        drop_last = True, num_workers=0)

    return graph, labeled_loader, valid_loader, test_loader, unlabeled_loader
