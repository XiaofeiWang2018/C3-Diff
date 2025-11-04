from share import *
import config
import matplotlib.pyplot as plt
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import pandas as pd
import random
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector, nms
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy import interpolate
from PIL import Image
import os
from tutorial_dataset import np_norm,gray_value_of_gene
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import scipy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)
####################################################
##########################GT HR LR WSI generation
####################################################


data_root='/home/cbtil/ST_proj/data/Breast_cancer/'


class Xenium_dataset(Dataset):
    def __init__(self, data_root,SR_times,status,gene_num):

        if status=='Train':
            sample_name=['01220101', '01220102', 'NC1', 'NC2', '0418']
        elif status=='Test':
            sample_name = ['01220201', '01220202']
        self.gene_num=gene_num
        SR_ST_all=[]
        gene_order = np.load(data_root + 'gene_order.npy')[0:gene_num]
        self.gene_order = []
        self.WSI_name=[]

        gene_name_200 = pd.read_csv('gene_name_200.csv').values[:,0]
        for i in range(84):
            if i ==0:
                self.gene_name_200=gene_name_200
            else:
                self.gene_name_200=np.concatenate((self.gene_name_200,gene_name_200),axis=0)

        ### HR ST
        for sample_id in sample_name:
            sub_patches=os.listdir(data_root+'Xenium/HR_ST/extract/'+sample_id)
            for patch_id in sub_patches:
                if SR_times==10:
                    SR_ST=np.load(data_root+'Xenium/HR_ST/extract/'+sample_id+'/'+patch_id+'/HR_ST_256.npy')
                elif SR_times==5:
                    SR_ST = np.load(data_root + 'Xenium/HR_ST/extract/' + sample_id + '/' + patch_id + '/HR_ST_128.npy')
                SR_ST=np.transpose(SR_ST,axes=(2,0,1))
                SR_ST_all.append(SR_ST)
                self.gene_order.append(gene_order)
        SR_ST_all=np.array(SR_ST_all)
        self.gene_order = np.array(self.gene_order)
        self.SR_ST_all=SR_ST_all[:,gene_order,...].astype(np.float64) # (X,50,256,256)


        for ii in range(self.SR_ST_all.shape[0]):
            for jj in range(self.SR_ST_all.shape[1]):
                if np.sum(self.SR_ST_all[ii, jj]) != 0:
                    Max=np.max(self.SR_ST_all[ii,jj])
                    Min=np.min(self.SR_ST_all[ii,jj])
                    self.SR_ST_all[ii,jj]=(self.SR_ST_all[ii,jj]-Min)/(Max-Min)
        self.SR_ST_all=self.SR_ST_all.reshape((-1,self.SR_ST_all.shape[2],self.SR_ST_all.shape[3]))
        self.gene_order = self.gene_order.reshape((-1))
        ### spot ST
        spot_ST_all=[]
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/spot_ST/extract/' + sample_id)
            for patch_id in sub_patches:
                spot_ST = np.load(data_root + 'Xenium/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
                spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
                spot_ST_all.append(spot_ST)
        spot_ST_all = np.array(spot_ST_all)
        self.spot_ST_all = spot_ST_all[:, gene_order, ...].astype(np.float64)


        for ii in range(self.spot_ST_all.shape[0]):
            for jj in range(self.spot_ST_all.shape[1]):
                if np.sum(self.spot_ST_all[ii, jj]) != 0:
                    Max = np.max(self.spot_ST_all[ii,jj])
                    Min = np.min(self.spot_ST_all[ii,jj])
                    self.spot_ST_all[ii,jj] = (self.spot_ST_all[ii,jj] - Min) / (Max - Min)
        self.spot_ST_all = self.spot_ST_all.reshape((-1, self.spot_ST_all.shape[2], self.spot_ST_all.shape[3]))

        ## WSI 5120
        WSI_5120_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_5120 = np.load(data_root + 'Xenium/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
                # WSI_save = Image.fromarray(WSI_5120)
                # WSI_save.save('img_result/WSI/' + sample_id[-2:] + '-' + patch_id + '.png')
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
                self.WSI_name.append(sample_id[-2:]+'-'+patch_id)
                # if not os.path.exists('img_result/HR_GT/' + sample_id[-2:] +'-'+ patch_id + '/'):
                #     os.makedirs('img_result/HR_GT/' + sample_id[-2:] +'-'+ patch_id + '/')
                # if not os.path.exists('img_result/LR_GT/' + sample_id[-2:] +'-'+ patch_id + '/'):
                #     os.makedirs('img_result/LR_GT/' + sample_id[-2:]+'-' + patch_id + '/')

        self.WSI_5120_all = np.array(WSI_5120_all)
        self.WSI_name= np.array(self.WSI_name)
        self.WSI_5120_all=np.repeat(self.WSI_5120_all, gene_num, axis=0)
        self.WSI_name = np.repeat(self.WSI_name, gene_num, axis=0)
        a=1





    def __len__(self):
        return self.spot_ST_all.shape[0]

    def _getitem__(self, index):

        target=self.SR_ST_all[index]
        target = np.expand_dims(target, axis=2)
        target = np.repeat(target, 3, axis=2)  #[256,256,3]
        source_STmap = self.spot_ST_all[index]# [26,26]
        source_STmap=np.expand_dims(source_STmap,axis=2)  # [26,26,1]
        source_STmap = np.repeat(source_STmap, 3, axis=2) # [26,26,3]
        source_STmap=np.array(Image.fromarray(np.uint8(source_STmap*255)).resize((256,256)))/255 # [256,256,3]
        target = (target.astype(np.float32) / 0.5) - 1.0


        source_WSI=self.WSI_5120_all[index]
        source_WSI = np.transpose(source_WSI, axes=(1, 2, 0)) # [256,256,3]
        ########################################################### Normalize source images to [0, 1].
        source_WSI = source_WSI.astype(np.float32) / 255.0
        WSI_name=self.WSI_name[index]
        gene_name=self.gene_name_200[index]

        return dict(HRST=target, LRST=source_STmap, WSI=source_WSI,WSI_name=WSI_name,gene_name=gene_name)


testset = Xenium_dataset(data_root=data_root, SR_times=10, status='Test',
                             gene_num=200)

for img_idx in range(len(testset)):
    batch = testset._getitem__(img_idx)
    GT = batch['HRST']  # 256 256 3 [-1,1]
    GT_final = (GT * 127.5 + 127.5).clip(0, 255).astype(np.uint8) / 255  # 256 256 3 [0,1]
    LRST = batch['LRST']  # 256 256 3 [0,1]
    WSI_name = batch['WSI_name']
    gene_name = batch['gene_name']


    # plt.imsave('img_result/LR_GT/'+WSI_name+'/'+gene_name+'.png',LRST)
    # plt.imsave('img_result/HR_GT/' + WSI_name + '/' + gene_name + '.png', GT_final)
    a=plt.imread('img_result/HR_GT/'+WSI_name+'/'+gene_name+'.png')
    plt.imshow(a[...,2])
    plt.show()
    print(img_idx)
    a=1




