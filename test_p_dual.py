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
from utils_postdam import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.unetformer_dual import UNetFormer
from dice import DiceLoss
from save_img_script import save_image
#d

import logging
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
logging.basicConfig(filename='', level=logging.INFO, format='%(message)s')
logging.info("--------------------------------test-result----------------------------------------")
# 如果你也想在控制台上看到输出，可以添加 StreamHandler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
logging.info("Device :", nvmlDeviceGetName(handle))

net = UNetFormer().cuda()
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
logging.info(params)
# Load the datasets

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

# optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# # We define the scheduler
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
# # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40, 45], gamma=0.1)
# # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40, 60, 80], gamma=0.1)

# optimizer = optim.Adadelta(net.parameters(), lr=1, rho=0.95, eps=1e-8)


# lr = 1e-4
lr = 1e-4
weight_decay = 0
optimizer = torch.optim.Adam(
    net.parameters(), lr=lr, weight_decay=weight_decay)
optim_step = 20
optim_gamma = 0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)

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
save_path = "/{}.jpg"
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

def test(net, test_ids, all=False, stride=32, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
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
        number_sta = 0
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            img_name = test_ids[number_sta]
            print("number_sta",img_name)

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
                # logging.info("resultresultresult",type(result))

                outs = result_final.data.cpu().numpy()
                
                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            
            save_img_is = True
            if save_img_is:
                save_image(pred,save_path.format(img_name))

            number_sta = number_sta +1 

            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
            
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy

net.load_state_dict(torch.load(""))
net.eval()
acc, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
