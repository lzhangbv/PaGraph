import os
import sys
import argparse, time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import dgl
from dgl import DGLGraph

from PaGraph.model.gcn_nssc import GCNSampling
import PaGraph.data as data
import PaGraph.storage as storage
from PaGraph.parallel import SampleLoader

exclude_GPU_compute = False

def init_process(rank, world_size, backend):
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29501'
  dist.init_process_group(backend, rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)
  torch.manual_seed(rank)
  print('rank [{}] process successfully launches'.format(rank))


def trainer(rank, world_size, args, backend='nccl'):
  # init multi process
  init_process(rank, world_size, backend)
  
  # load data
  dataname = os.path.basename(args.dataset)
  if rank == 0:
      print("before loading remote_g")
  remote_g = dgl.contrib.graph_store.create_graph_from_store(dataname, "shared_mem")
  if rank == 0:
      print("after loading remote_g")

  adj, t2fid = data.get_sub_train_graph(args.dataset, rank, world_size)
  g = DGLGraph(adj, readonly=True)
  n_classes = args.n_classes
  train_nid = data.get_sub_train_nid(args.dataset, rank, world_size)
  sub_labels = data.get_sub_train_labels(args.dataset, rank, world_size)
  labels = np.zeros(np.max(train_nid) + 1, dtype=np.int)
  labels[train_nid] = sub_labels

  # to torch tensor
  t2fid = torch.LongTensor(t2fid)
  labels = torch.LongTensor(labels)
  embed_names = ['features', 'norm']
  cacher = storage.GraphCacheServer(remote_g, adj.shape[0], t2fid, rank)
  cacher.init_field(embed_names)
  cacher.log = True

  # prepare model
  num_hops = args.n_layers if args.preprocess else args.n_layers + 1
  model = GCNSampling(args.feat_size,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.preprocess)
  loss_fcn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
  model.cuda(rank)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
  ctx = torch.device(rank)

  if rank == 0:
      print("before prepare sampler")
  if args.remote_sample:
    sampler = SampleLoader(g, rank, one2all=args.one2all)
  else:
    sampler = dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
      args.num_neighbors, neighbor_type='in',
      shuffle=True, num_workers=args.num_workers,
      num_hops=num_hops, seed_nodes=train_nid,
      prefetch=True
    )
  if rank == 0:
      print("after prepare sampler")

  # start training
  epoch_dur = []
  epoch_miss_rate = []
  epoch_loss = []
  tic = time.time()
  with torch.autograd.profiler.profile(enabled=(rank==0), use_cuda=True) as prof:
    for epoch in range(args.n_epochs):
      model.train()
      epoch_start_time = time.time()
      step = 0
      for nf in sampler:
        with torch.autograd.profiler.record_function('gpu-load'):
          cacher.fetch_data(nf)
          batch_nids = nf.layer_parent_nid(-1)
          label = labels[batch_nids]
          label = label.cuda(rank, non_blocking=True)
        #with torch.autograd.profiler.record_function('gpu-compute'):
        if not exclude_GPU_compute or not (epoch + step):
          pred = model(nf)
          loss = loss_fcn(pred, label)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        step += 1
        if epoch == 0 and step == 1:
          cache_time = time.time()
          cacher.auto_cache(g, embed_names)
          print('auto_cache time on gpu %s: %.4f' % (rank, time.time() - cache_time))
        if rank == 0 and step % 2 == 0:
          print('epoch [{}] step [{}]. Loss: {:.4f}'
                .format(epoch + 1, step, loss.item()))
      
      epoch_dur.append(time.time() - epoch_start_time)
      epoch_loss.append(loss.item())
      #print('Epoch average time: {:.4f}'.format(np.mean(np.array(epoch_dur[2:]))))
      if cacher.log:
        miss_rate = cacher.get_miss_rate()
        epoch_miss_rate.append(miss_rate)
    toc = time.time()
  
  if cacher.log and rank == 0:
    print('Epoch average hit rate: {:.4f}'.format(1-np.mean((epoch_miss_rate[2:]))))
  if rank == 0:
    print('Epoch average time: {:.4f}'.format(np.mean(np.array(epoch_dur[2:]))))
    print('Total Time: {:.4f}s'.format(toc - tic))
    print('Epoch train loss: %s' % epoch_loss)
    #print(prof.key_averages().table(sort_by='cuda_time_total'))



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GCN')

  parser.add_argument("--gpu", type=str, default='cpu',
                      help="gpu ids. such as 0 or 0,1,2")
  parser.add_argument("--dataset", type=str, default=None,
                      help="path to the dataset folder")
  # model arch
  parser.add_argument("--feat-size", type=int, default=600,
                      help='input feature size')
  parser.add_argument("--n-classes", type=int, default=60)
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="dropout probability")
  parser.add_argument("--n-hidden", type=int, default=32,
                      help="number of hidden gcn units")
  parser.add_argument("--n-layers", type=int, default=1,
                      help="number of hidden gcn layers")
  parser.add_argument("--preprocess", dest='preprocess', action='store_true')
  parser.set_defaults(preprocess=False)
  # training hyper-params
  parser.add_argument("--lr", type=float, default=3e-2,
                      help="learning rate")
  parser.add_argument("--n-epochs", type=int, default=10,
                      help="number of training epochs")
  parser.add_argument("--batch-size", type=int, default=6000,
                      help="batch size")
  parser.add_argument("--weight-decay", type=float, default=0,
                      help="Weight for L2 loss")
  # sampling hyper-params
  parser.add_argument("--num-neighbors", type=int, default=2,
                      help="number of neighbors to be sampled")
  parser.add_argument("--num-workers", type=int, default=16)
  parser.add_argument("--remote-sample", dest='remote_sample', action='store_true')
  parser.set_defaults(remote_sample=False)
  parser.add_argument("--one2all", dest='one2all', action='store_true')
  parser.set_defaults(one2all=False)
  
  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  gpu_num = len(args.gpu.split(','))

  mp.spawn(trainer, args=(gpu_num, args), nprocs=gpu_num, join=True)
  
