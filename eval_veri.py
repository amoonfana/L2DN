# import pickle
import sys
from PIL import Image
import logging
import os
import datetime
from typing import List
# import cv2
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from utils.utils_logging import init_logging

import sklearn
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import argparse
from backbones import get_model
# import verification
import importlib
# from logger import logger

def init_logging(rank, models_root):
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
        log_root.info('rank_id: %d' % rank)

@torch.no_grad()
def load_bin(path, image_size):
    issame_list = np.load(os.path.join(path, "pair_label.npy"))
    img_path = os.path.join(path, "imgs")
    
    imgs = []
    for idx in range(len(issame_list)*2):
        # img = cv2.imread(os.path.join(img_path, f"{idx}.jpg"))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(Image.open(os.path.join(img_path, f"{idx}.jpg")))
        imgs.append(img)
    
    # with open(path, 'rb') as f:
    #     bins, issame_list = pickle.load(f, encoding='bytes')  # py3
        
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        img = imgs[idx]
    #     _bin = bins[idx]
    #     img = mx.image.imdecode(_bin)
        # if img.shape[1] != image_size[0]:
        #     img = mx.image.resize_short(img, image_size[0])
        img = np.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = np.flip(img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.copy())
        if idx % 1000 == 0:
            print('loading img', idx)
    print(data_list[0].shape)
    return data_list, issame_list

def init_dataset(val_targets, data_dir, image_size):
    for name in val_targets:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            data_set = load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)

class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
            
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                        embeddings1,
                                        embeddings2,
                                        np.asarray(actual_issame),
                                        nrof_folds=nrof_folds,
                                        pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5
            net_out: torch.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list

def ver_test(backbone: torch.nn.Module, global_step: int):
    results = []
    for i in range(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = test(
            ver_list[i], backbone, 10, 10)
        logging.info('[%s][%d]XNorm: %f' % (ver_name_list[i], global_step, xnorm))
        logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], global_step, acc2, std2))

        summary_writer.add_scalar(tag=ver_name_list[i], scalar_value=acc2, global_step=global_step, )

        if acc2 > highest_acc_list[i]:
            highest_acc_list[i] = acc2
        logging.info(
            '[%s][%d]Accuracy-Highest: %1.5f' % (ver_name_list[i], global_step, highest_acc_list[i]))
        results.append(acc2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get configurations')
    parser.add_argument('--config', default="webface12m", help='the name of config file')
    args = parser.parse_args()
    
    # cfg = get_config(args.config)
    config = importlib.import_module("configs."+args.config)
    cfg = config.cfg()
    
    rec_prefix = cfg.val
    model_path = cfg.output + "/model.pt"
    val_targets = cfg.val_targets
    network = cfg.network
    image_size = cfg.image_size
    embedding_size = cfg.embedding_size
    
    init_logging(0, cfg.output)
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    # log = logger(cfg=cfg, start_step = 0, writer=summary_writer)
    
    backbone = get_model(network, dropout=0, fp16=False).cuda()
    backbone.load_state_dict(torch.load(model_path))
    backbone = torch.nn.DataParallel(backbone)
    
    highest_acc_list: List[float] = [0.0] * len(val_targets)
    ver_list: List[object] = []
    ver_name_list: List[str] = []
    init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)
    
    backbone.eval()
    ver_test(backbone, 6666)
    # backbone.train()