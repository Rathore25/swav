import argparse
import os
import time
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torchsummary import summary

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

from datetime import datetime

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    accuracy,
)
import src.resnet50 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Feature visualization of ResNet50 model pre-trained on ImageNet")

#########################
#### main parameters ####
#########################
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--data_classes", default=1000, type=int,
                    help="number of data classes")

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")
parser.add_argument("--global_pooling", default=True, type=bool_flag,
                    help="if True, we use the resnet50 global average pooling")
parser.add_argument("--use_bn", default=False, type=bool_flag,
                    help="optionally add a batchnorm layer before the linear classifier")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", default=0.3, type=float, help="initial learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--nesterov", default=False, type=bool_flag, help="nesterov momentum")
parser.add_argument("--scheduler_type", default="cosine", type=str, choices=["step", "cosine"])
# for multi-step learning rate decay
parser.add_argument("--decay_epochs", type=int, nargs="+", default=[60, 80],
                    help="Epochs at which to decay learning rate.")
parser.add_argument("--gamma", type=float, default=0.1, help="decay factor")
# for cosine learning rate schedule
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

def main():
    global args, best_acc
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val"
    )

    # build test data
    test_dataset = datasets.ImageFolder(os.path.join(args.data_path, "test"))
    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )
    test_dataset.transform  = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    logger.info("Building test data done")


    # model = resnet_models.__dict__[args.arch](output_dim=0, eval_mode=True)

    # model = model.cuda()
    # model.eval()

    # If os.path.isfile(args.pretrained):
        # state_dict = torch.load(args.pretrained, map_location="cuda:" + str(args.gpu_to_work_on))
        # if "state_dict" in state_dict:
            # state_dict = state_dict["state_dict"]
        # State_dict = {k.replace("module.", ""): V for k, v in state_dict.items()}
        # for k, v in model.state_dict().items():
            # if k not in list(state_dict):
                # logger.info('key "{}" could not be found in provided state dict'.format(k))
            # elif state_dict[k].shape != V.shape:
                # logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                # state_dict[k] = v
        # msg = model.load_state_dict(state_dict, strict=false)
        # logger.info("load pretrained model with msg: {}".format(msg))
    # else:
        # logger.info("no pretrained weights found => training with random weights")

    # Cudnn.benchmark = true
    
    # Logger.info("finding outputs of load pretrained model with msg: {}".format(msg))
    
    test_targets    = None
    test_embeddings = None
    
    model       = models.resnet50(True, True)
    #model       = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    model.fc    = nn.Identity()
    model       = model.cuda()
    model.eval()
    
    print(summary(model, (3, 224, 224)))
    
    with torch.no_grad():
        for i, (inp, target) in enumerate(test_loader):
            inp     = inp.cuda(non_blocking=True)
            
            output  = model(inp)
            output  = output.cpu()
            output  = output.numpy()
            
            y       = target.cpu()
            y       = y.numpy()
            
            if test_embeddings is None:
                test_embeddings = output
            else:
                test_embeddings = np.concatenate((test_embeddings, output), axis=0)
            
            if test_targets is None:
                test_targets = y
            else:
                test_targets = np.concatenate((test_targets, y), axis=0)

    test_embeddings = np.array(test_embeddings)
    test_targets    = np.array(test_targets)
    
    print(test_embeddings.shape)
    print(test_embeddings[0].shape)
    print(test_targets.shape)
    print(test_targets[0].shape)
    
    logger.info("Completed preparing the embeddings")
        
    tsne            = TSNE(2, verbose=1)
    tsne_results    = tsne.fit_transform(test_embeddings)
    
    logger.info("Completed fitting into tSNE")
    
    df_tsne_results             = pd.DataFrame(tsne_results, columns=["tsne-2d-one","tsne-2d-two"])
    df_tsne_results["Cluster"]  = test_targets
    
    print(df_tsne_results.head(2))
    
    plt.figure(figsize=(8,8))
    
    sns.scatterplot(
        x       = "tsne-2d-one", 
        y       = "tsne-2d-two",
        hue     = "Cluster",
        palette = sns.color_palette("hls", args.data_classes),
        data    = df_tsne_results,
        legend  = False,
        alpha   = 0.5,
        s       = 5
    )
    
    now         = datetime.now()
    plotName    = "plot_" + now.strftime("%Y%m%d%H%M%S") + ".png"
    plotPath    = os.path.join(args.dump_path, plotName)
    logger.info("File path: " + plotPath)
    
    plt.savefig(plotPath, bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    main()
