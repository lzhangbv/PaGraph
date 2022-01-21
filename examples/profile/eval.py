import torch
import numpy as np
import dgl



def evaluation(args, graph, model, test_nid, labels, sample_infer=True):
    "Evaluation on one GPU"
    ctx = torch.device(0)
    
    # load sampler
    num_hops = args.n_layers if args.preprocess else args.n_layers + 1
    if sample_infer:  # sample neighbors for model inference
        sampler = dgl.contrib.sampling.NeighborSampler(graph,
            args.batch_size,
            args.num_neighbors,
            neighbor_type='in',
            shuffle=False,
            num_workers=4,
            num_hops=num_hops,
            seed_nodes=test_nid)
    else:
        batch_size = args.batch_size # non-sample mini-batch
        #batch_size = len(test_nid)   # non-sample full-batch
        sampler = dgl.contrib.sampling.NeighborSampler(graph,
            batch_size,
            graph.number_of_nodes(),
            neighbor_type='in',
            shuffle=False,
            num_workers=4,
            num_hops=num_hops,
            seed_nodes=test_nid)

    
    # call evaluation
    num_acc = 0.0
    model.eval()
    with torch.no_grad():
        for nf in sampler:
            nf.copy_from_parent(ctx=ctx)
            pred = model(nf)
            batch_nids = nf.layer_parent_nid(-1)
            batch_labels = labels[batch_nids].cuda(0)
            num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()

    model.train()
    return num_acc / len(test_nid)
