import os
# from datetime import datetime
import random
import numpy as np
import argparse

import torch
from backbones import get_model
from dataset import ImageFolderDataset, LMDBDataset
from losses import CombinedMarginLoss, ArcFace, CosFace, NaiveFace
# from lr_scheduler import PolynomialLRWarmup
from lr_scheduler import MHLR
from torch.optim.lr_scheduler import ConstantLR, LinearLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, SequentialLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from partial_fc_v2 import PartialFC_V2, my_PFC, LoraFC, my_CE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from logger import logger
import importlib

def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def main(config_file):
    # get config
    config = importlib.import_module("configs."+config_file)
    cfg = config.cfg()
    
    device = torch.device(cfg.device)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    os.makedirs(cfg.output, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    log = logger(cfg=cfg, start_step = 0, writer=summary_writer)
    
    # Image Folder
    train_set = None
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    if cfg.rec.endswith(".pickle"):
        train_set = ImageFolderDataset(cfg.rec, transform)
    elif cfg.rec.endswith(".lmdb"):
        train_set = LMDBDataset(cfg.rec, transform)
    else:
        train_set = ImageFolder(cfg.rec, transform)
        
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)


    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
    backbone.train().to(device)

    margin_loss = CombinedMarginLoss(cfg.s, cfg.margin_list[0], cfg.margin_list[1], cfg.margin_list[2], cfg.interclass_filtering_threshold)
    # margin_loss = CosFace()
    # margin_loss = NaiveFace()
    
    # CE_loss = my_CE(margin_loss, cfg.embedding_size, cfg.num_classes, False)
    # CE_loss = LoraFC(margin_loss, cfg.embedding_size, cfg.num_classes, cfg.bottle_neck, False)
    CE_loss = my_PFC(margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False)
    CE_loss.train().to(device)
    
    opt = torch.optim.SGD(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    # opt = torch.optim.Adam(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    # opt = torch.optim.AdamW(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # lr_scheduler = MHLR(opt, cfg.total_step)
    # lr_scheduler = ConstantLR(opt, factor=1, total_iters=cfg.total_step)
    # warmup_scheduler = LinearLR(opt, start_factor=0.01, end_factor=1, total_iters=0.1*cfg.total_step)
    # lr_scheduler = MultiStepLR(opt, milestones=[0.3*cfg.total_step, 0.5*cfg.total_step, 0.6*cfg.total_step, 0.7*cfg.total_step, 0.8*cfg.total_step, 0.9*cfg.total_step], gamma=0.5)
    # lr_scheduler = ExponentialLR(opt, gamma=0.9999)
    lr_scheduler = CosineAnnealingLR(opt, T_max=cfg.total_step)
    # warmup_scheduler = LinearLR(opt, start_factor=0.01, end_factor=1, total_iters=0.1*cfg.total_step)
    # main_scheduler = CosineAnnealingLR(opt, T_max=0.9*cfg.total_step)
    # lr_scheduler = SequentialLR(opt, schedulers=[warmup_scheduler, main_scheduler], milestones=[0.1*cfg.total_step])



    global_step = 0
    amp = torch.amp.GradScaler("cuda", growth_interval=100)
    # amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    for epoch in range(0, cfg.num_epoch):
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img.to(device))
            loss, loss_ce, loss_p, avg_embedding_norm, avg_weight_norm = CE_loss(local_embeddings, local_labels.to(device))

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()
            # lr_scheduler.my_step(loss.item())

            with torch.no_grad():
                log(global_step, loss.detach().cpu().numpy(), loss_ce, loss_p, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp, avg_embedding_norm, avg_weight_norm)

                # if global_step % cfg.verbose == 0 and global_step > 0:
                #     callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.state_dict(),
                "state_dict_softmax_fc": CE_loss.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{epoch}.pt"))

        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.state_dict(), path_module)

    path_module = os.path.join(cfg.output, "model.pt")
    torch.save(backbone.state_dict(), path_module)
    log.loss2csv(cfg.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get configurations')
    parser.add_argument('--config', default="ms1mv3", help='the name of config file')
    args = parser.parse_args()
    main(args.config)
