from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import copy
import time
from datetime import timedelta
import datetime
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize
import os
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(".")
from reid import datasets
from reid import models
from reid.models import convert_bn,convert_dsbn
from reid.models.xbm import XBM
from reid.trainer import Base_trainer
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import transforms as T
from reid.utils.data import *
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.rerank import compute_jaccard_distance
from reid.utils.plt import draw

from pdb import set_trace



def get_data(name,root):
    root = osp.join(root,name)
    return datasets.get_dataset(name=name,root=root)
def get_train_loader(data,height,width,batchsize,workers,num_instances,iters ,train_set=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    train_set = sorted(data.train) if train_set is None else sorted(train_set)
    sample = RandomMultipleGallerySampler(train_set,num_instances=num_instances)
    #root= None，根据dataset（list）的fname读取图片
    train_loader = IterLoader(
        DataLoader(Preprocessor(dataset=train_set,root=data.images_dir,transform=train_transformer),
                   batch_size=batchsize,sampler=sample,pin_memory=True,
                   shuffle=False,drop_last=True,num_workers=workers),length=iters
    )
    return train_loader

def get_test_loader(data,height,width,batchsize,workers,test_set =None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    test_set = list(set(data.query)|set(data.gallery)) if test_set is None else sorted(test_set)

    test_loader = DataLoader(
        Preprocessor(dataset=test_set,root=data.images_dir,transform=test_transformer)
        ,batch_size=batchsize,shuffle=False,num_workers=workers,pin_memory=True
    )
    return test_loader


def main(args):
    start_epoch = best_mAP= 0
    start_time = time.monotonic()
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    assert(args.batch_size%3)!=0
    # if torch.cuda.is_available():
    #     os.environ['CUDA_VISIBLE_DEVICES']='0'
    #     print('可用的Gpu数目为,',torch.cuda.device_count())
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    ##Create dataset
    print("==========Load source-domain dataset:",args.dataset_source)
    source_data = get_data(args.dataset_source,args.data_dir)

    ds_train_loader = get_train_loader(source_data,args.height,args.width,args.batch_size,
                                       args.workers,args.num_instances,args.iters)
    print("==========Load target-domain dataset:", args.dataset_target)
    target_data = get_data(args.dataset_target,args.data_dir)
    dt_test_loader  = get_test_loader(target_data,args.height,args.width,args.batch_size,args.workers)

    args.s_class = source_data.num_train_pids
    args.t_class = len(target_data.train)
    args.fc_class = source_data.num_train_pids+ len(target_data.train)




    ###create Model
    ##*****************************************************************************************

    print('now create model:',args.arch)
    model = models.create(name=args.arch,num_features = args.features,num_classes = args.fc_class)
   #convert_dsbn(model)
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print('start at {} epoch'.format(start_epoch))
    model = model.cuda()
    model = nn.DataParallel(model)

    evaluator = Evaluator(model)

    ##init optim
    ##****************************************************************************************

    paras = [{'params': [value]} for _, value in model.named_parameters() if value.requires_grad]
    optim = torch.optim.Adam(paras, lr=args.lr, weight_decay=args.weight_decay)

    warm_up_with_step_lr = lambda epoch:(epoch+1)/args.warm_up_epochs if epoch<args.warm_up_epochs \
                    else 0.1**((epoch-args.warm_up_epochs)//args.step_size)
    #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim,warm_up_with_step_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_size, gamma=0.1)


    ##createXBM
    datasetsize = len(source_data.train)+len(target_data.train)
    args.memorysize = int(datasetsize*args.ratio)
    if args.use_xbm:
        print('使用xbm:',args.memorysize)
        xbm  =XBM(args.memorysize,args.featureSize)
    else:
        xbm = None

    with torch.no_grad():

        ##get source domain and target domain centroids
        ds_class_loader = get_test_loader(source_data,args.height,args.width,args.batch_size,
                                          args.workers,test_set=source_data.train)

        sou_fea ,_ = extract_features(model,ds_class_loader)

        sour_cen_list = collections.defaultdict(list)
        for _,(i,pid,_) in enumerate(source_data.train):
            sour_cen_list[pid].append(sou_fea[i])

        sour_centers = [torch.stack(sour_cen_list[pid], 0).mean(0)
                        for pid in sour_cen_list.keys()]
        #    sour_centers = [torch.cat(sour_cen_list[pid],0).view(-1,args.featureSize).mean(0)
               #     for pid in sour_cen_list.keys()]

        sour_centers = torch.stack(sour_centers, 0)
        sour_centers = F.normalize(sour_centers, dim=1)  ###trainset class centroids

        model.module.classifier.weight.data[0:int(args.s_class)].copy_(sour_centers.cuda())
        del ds_class_loader,sour_cen_list,sour_centers,sou_fea


    ##init trainer
    trainer = Base_trainer(arg=args,model=model,num_classes=args.fc_class,xbm=xbm,margin=args.margin)
    for epoch in range(start_epoch,args.epochs):

        ###target_data cluster

        dt_class_loader = get_test_loader(target_data,args.height,args.width,args.batch_size,
                                 args.workers,test_set=sorted(target_data.train))
        dt_fea,_ = extract_features(model,dt_class_loader)
        dt_fea = torch.cat([dt_fea[fname].unsqueeze(0) for fname,_,_ in sorted(target_data.train)],0)


        print('==> Create pseudo labels for unlabeled target domain with DBSCAN clustering')
        ###accel the process of cluster,jaccard_distance return a distance metric

        rerank_dist = compute_jaccard_distance(dt_fea, k1=args.k1, k2=args.k2, use_gpu=False).numpy()
        eps = args.eps
        cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed',n_jobs=-1)
        labels = cluster.fit_predict(rerank_dist)
        num_id = len(set(labels))-(1 if -1 in labels else 0)
        args.t_class = num_id
        #genrate new dataset and calculate target centers

        print('第{}个epoch聚类得到了{}个类'.format(epoch, num_id))
        new_dataset = []
        cluster_centers = collections.defaultdict(list)

        for i,((fname,_,cid),label)in enumerate(zip(sorted(target_data.train),labels)):
            if label == -1 :continue  ###离群点不放进来
            new_dataset.append([fname,args.s_class+label,cid])
            cluster_centers[label].append(dt_fea[i])
        cluster_centers = [torch.stack(cluster_centers[pid],0).mean(0) for pid in sorted(cluster_centers.keys())]
        cluster_centers = torch.stack(cluster_centers,0)
        cluster_centers =F.normalize(cluster_centers,dim=1)
        model.module.classifier.weight.data[args.s_class:args.s_class+num_id].\
            copy_(cluster_centers).cuda()

        del cluster_centers,dt_fea,rerank_dist,dt_class_loader

        #init dt train loader
        print("load target train loader with presudo label ,the len of dataset is:{} ".format(len(new_dataset)))
        dt_train_loader = get_train_loader(target_data,args.height,args.width,args.batch_size,args.workers,
                                           args.num_instances,args.iters,train_set=new_dataset)

        ds_train_loader.new_epoch()
        dt_train_loader.new_epoch()

        trainer.train(epoch,ds_train_loader,dt_train_loader,optim,args.s_class,args.t_class,args.iters,
                      gamma = args.gamma,print_freq=args.print_freq)

        if ((epoch+1)%50==0) or (epoch == args.epochs-1) or epoch ==0 :
            test_time = time.monotonic()
            print('test on target',args.dataset_target)
            _,mAP = evaluator.evaluate(dt_test_loader,target_data.query,target_data.gallery,cmc_flag = True,rerank=args.rerank)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{},test_time:{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else '',timedelta(seconds=(test_time-time.monotonic()))))

        lr_scheduler.step()

    #画loss,lr曲线
    #draw(trainer.losslist,trainer.losslist_ce,args.logs_dir)

    print('==> Test with the best model on the target domain:')

    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(dt_test_loader, target_data.query, target_data.gallery, cmc_flag=True,rerank=args.rerank)

    end_time = time.monotonic()
    print('Total running time:', timedelta(seconds=end_time - start_time))
    print('end time:',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on UDA re-ID")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc')
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                            "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--nclass', type=int, default=1000,
                        help="number of classes (source+target)")
    parser.add_argument('--s-class', type=int, default=1000,
                        help="number of classes (source)")
    parser.add_argument('--t-class', type=int, default=1000,
                        help="number of classes (target)")
    # loss
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin for triplet loss")
    parser.add_argument('--mu1', type=float, default=0.5,
                        help="weight for loss_bridge_pred")
    parser.add_argument('--mu2', type=float, default=0.1,
                        help="weight for loss_bridge_feat")
    parser.add_argument('--gamma', type=float, default=0.0,
                        help="weight for loss_re")
    parser.add_argument('--mu3', type=float, default=1,
                        help="weight for loss_div")

    # models
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate',action= 'store_true',help='if True ,only evaluate model')
    parser.add_argument('--rerank',action= 'store_true',help='if True ,reranking when test')

    # xbm parameters
    parser.add_argument('--memorySize', type=int, default=8192,
                        help='meomory bank size')
    parser.add_argument('--ratio', type=float, default=1,
                        help='memorySize=ratio*data_size')
    parser.add_argument('--featureSize', type=int, default=2048)
    parser.add_argument('--use-xbm', action='store_true',help="if True: strong baseline; if False: naive baseline")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--warm_up_epochs', type=int, default=5)

    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=16)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=10)

    # path

    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    args = parser.parse_args()
    main(args)