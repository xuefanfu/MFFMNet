import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils_v_dsm import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.unetformer_dual import UNetFormer
from dice import DiceLoss
# export HF_ENDPOINT=https://hf-mirror.com

import logging
import os
# from tqdm.notebook import tqdm as tqdm_notebook
# import tqdm 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    ## Potsdam
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)
                dsm_patches = dsm_patches.unsqueeze(1)
                # Do the inference
                result, resultd, result_final, basefeature, detailfeature, basefeature_d, detailfeature_d = net(image_patches, dsm_patches)
                outs = result_final.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
            
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy

def train(net, epochs, save_path, weights=WEIGHTS):
    try:
        from urllib.request import URLopener
    except ImportError:
        from urllib import URLopener

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()
    params = 0
    for name, param in net.named_parameters():
        params += param.nelement()
    logging.info(params)
    # Load the datasets

    logging.info("training : {}".format(train_ids))
    logging.info("testing : {}".format(test_ids))
    logging.info("BATCH_SIZE: {}".format(BATCH_SIZE))
    logging.info("Stride Size: {}".format(Stride_Size))
    train_set = ISPRS_dataset(train_ids, cache=CACHE)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

    base_lr = 0.01
    # base_lr = 0.05
    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        if '_D' in key:
            # Decoder weights are trained at the nominal learning rate
            params += [{'params':[value],'lr': base_lr}]
        else:
            # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
            params += [{'params':[value],'lr': base_lr / 2}]


    lr = 1e-4
    # lr = 1e-5
    weight_decay = 0
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(
    #     net.parameters(), lr=lr, weight_decay=1e-5,betas=(0.9, 0.999))
    optim_step = 20 
    optim_gamma = 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)

    iter_ = 0
    acc_best = 0.905

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            
            dsm = dsm.unsqueeze(1)
            optimizer.zero_grad()
            result, result_d, result_final, basefeature, detailfeature, basefeature_d, detailfeature_d = net(data, dsm)

            loss1 = CrossEntropy2d(result, target, weight=weights)
            loss2 = CrossEntropy2d(result_d, target, weight=weights)
            loss3 = CrossEntropy2d(result_final, target, weight=weights)
            cc_loss_B = cc(basefeature, basefeature)
            cc_loss_D = cc(detailfeature, detailfeature_d)
            loss_decomp =   (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
            loss = 2*loss3+ 0.5*loss1 + 0.5*loss2 + 0.1*loss_decomp

            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(result_final.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                logging.info('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

            # if e % save_epoch == 0:
            if iter_ % 1000 == 0:
                net.eval()
                acc = test(net, test_ids, all=False, stride=Stride_Size)
                net.train()
                if acc > acc_best:
                    torch.save(net.state_dict(), save_path+'/segnet256_epoch{}_{}'.format(e, acc))
                    acc_best = acc
                else:
                    if acc > 0.91:
                        torch.save(net.state_dict(), save_path+'/segnet256_epoch{}_{}'.format(e, acc))
    logging.info('acc_best: ', acc_best)


if __name__ == '__main__':
    #####   train   ####
    seed=1
    set_global_seed(seed)
    save_path = "/MFFMNet-v-"+str(seed)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    time_start=time.time()
    # train(net, optimizer, 100, scheduler)
    net = UNetFormer().cuda()
    train(net, 100, save_path)
    time_end=time.time()
    logging.info('Total Time Cost: ',time_end-time_start)

