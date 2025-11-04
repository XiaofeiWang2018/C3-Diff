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

def get_ssim(ground_truth, generated_image,):
    """
    Calculate the structural similarity index (SSIM) between two images.
    Args:
        ground_truth (numpy.ndarray): Ground truth image.
        generated_image (numpy.ndarray): Generated image.
    Returns:
        float: SSIM value.
    """
    return structural_similarity(ground_truth, generated_image, multichannel=True, gaussian_weights=True, sigma=1.5)

class Xenium_dataset(Dataset):
    def __init__(self, data_root,SR_times,status,gene_num):

        if status == 'Train':
            sample_name = ['01220101', '01220102', 'NC1', 'NC2']
            # sample_name = ['NC2']
        elif status == 'Test':
            sample_name = ['01220201', '01220202']
            # sample_name = ['01220202']
        self.gene_num = gene_num
        SR_ST_all = []
        gene_order = np.load(data_root + 'gene_order.npy')[0:gene_num]
        self.gene_order = []
        gene_name_50 = pd.read_csv('gene_name_200.csv').values[:,0]
        for i in range(84):
            if i == 0:
                self.gene_name_50 = gene_name_50
            else:
                self.gene_name_50 = np.concatenate((self.gene_name_50, gene_name_50), axis=0)
        ### HR ST
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/HR_ST/extract/' + sample_id)
            for patch_id in sub_patches:
                if SR_times == 10:
                    SR_ST = np.load(data_root + 'Xenium/HR_ST/extract/' + sample_id + '/' + patch_id + '/HR_ST_256.npy')
                elif SR_times == 5:
                    SR_ST = np.load(data_root + 'Xenium/HR_ST/extract/' + sample_id + '/' + patch_id + '/HR_ST_128.npy')
                SR_ST = np.transpose(SR_ST, axes=(2, 0, 1))
                SR_ST_all.append(SR_ST)
                self.gene_order.append(gene_order)
        SR_ST_all = np.array(SR_ST_all)
        self.gene_order = np.array(self.gene_order)
        self.SR_ST_all = SR_ST_all[:, gene_order, ...].astype(np.float64)  # (X,50,256,256)

        for ii in range(self.SR_ST_all.shape[0]):
            for jj in range(self.SR_ST_all.shape[1]):
                if np.sum(self.SR_ST_all[ii, jj]) != 0:
                    Max = np.max(self.SR_ST_all[ii, jj])
                    Min = np.min(self.SR_ST_all[ii, jj])
                    self.SR_ST_all[ii, jj] = (self.SR_ST_all[ii, jj] - Min) / (Max - Min)
        self.SR_ST_all = self.SR_ST_all.reshape((-1, self.SR_ST_all.shape[2], self.SR_ST_all.shape[3]))
        self.gene_order = self.gene_order.reshape((-1))
        ### spot ST
        spot_ST_all = []
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
                    Max = np.max(self.spot_ST_all[ii, jj])
                    Min = np.min(self.spot_ST_all[ii, jj])
                    self.spot_ST_all[ii, jj] = (self.spot_ST_all[ii, jj] - Min) / (Max - Min)
        self.spot_ST_all = self.spot_ST_all.reshape((-1, self.spot_ST_all.shape[2], self.spot_ST_all.shape[3]))

        ## WSI 5120
        WSI_5120_all = []
        self.WSI_name = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_5120 = np.load(data_root + 'Xenium/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
                self.WSI_name.append(sample_id[-2:] + '-' + patch_id)

        self.WSI_5120_all = np.array(WSI_5120_all)
        self.WSI_5120_all = np.repeat(self.WSI_5120_all, gene_num, axis=0)
        self.WSI_name = np.array(self.WSI_name)
        self.WSI_name = np.repeat(self.WSI_name, gene_num, axis=0)
        a = 1



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
        ## Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 0.5) - 1.0


        source_WSI=self.WSI_5120_all[index]
        source_WSI = np.transpose(source_WSI, axes=(1, 2, 0)) # [256,256,3]
        # source_WSI=np_norm(source_WSI) ## can be annotated
        ########################################################### Normalize source images to [0, 1].
        source_WSI = source_WSI.astype(np.float32) / 255.0


        gene_class=self.gene_order[index]
        # gene_caption='spatial expression map of gene '+str(gene_class)
        gene_caption = ' '
        assert  self.gene_num<255
        Gene_code=gray_value_of_gene(gene_class,self.gene_order[0:self.gene_num])
        Gene_index_map=np.ones(shape=(256,256,1))*Gene_code / 255.0
        WSI_name = self.WSI_name[index]
        gene_name = self.gene_name_50[index]
        #
        return dict(HRST=target, LRST=source_STmap, WSI=source_WSI,gene_caption=gene_caption, gene_class=gene_class,
                    Gene_index_map=Gene_index_map,WSI_name=WSI_name,gene_name=gene_name)

if __name__ == '__main__':

    config = OmegaConf.load('./models/cldm_ours.yaml')
    testset = Xenium_dataset(data_root=config.data_root, SR_times=config.SR_times, status='Test',
                             gene_num=config.gene_num)
    model = instantiate_from_config(config.model).cpu()
    model.load_state_dict(load_state_dict(config.resume_path_test, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    seed_everything(config.seed)

    rmse_all = np.zeros(shape=(len(testset)))
    cc_all = np.zeros(shape=(len(testset)))
    ssim_all = np.zeros(shape=(len(testset)))
    for img_idx in range(len(testset)):

        batch=testset._getitem__(img_idx)
        GT = batch['HRST'] # 256 256 3 [-1,1]
        GT_final=(GT* 127.5 + 127.5).clip(0, 255).astype(np.uint8)[...,2]/255# 256 256

        LRST = batch['LRST']# 256 256 3 [0,1]
        WSI = batch['WSI']# 256 256 3 [0,1]
        gene_caption = batch['gene_caption']
        Gene_index_map = batch['Gene_index_map']# 256 256 1 [0,1]
        WSI_name = batch['WSI_name']
        gene_name = batch['gene_name']
        with torch.no_grad():
            H, W, C = WSI.shape
            control_LRST = torch.from_numpy(LRST.copy()).float().cuda()  # 256 256 3
            control_LRST=torch.unsqueeze(control_LRST,dim=0)# 1 256 256 3
            control_LRST = einops.rearrange(control_LRST, 'b h w c -> b c h w').clone() # 1 3 256 256

            control_WSI = torch.from_numpy(WSI.copy()).float().cuda()   # 256 256 3
            control_WSI = torch.unsqueeze(control_WSI, dim=0)  # 1 256 256 3
            control_WSI = einops.rearrange(control_WSI, 'b h w c -> b c h w').clone()  # 1 3 256 256

            control_Gene_index_map = torch.from_numpy(Gene_index_map.copy()).float().cuda()   # 256 256 3
            control_Gene_index_map = torch.unsqueeze(control_Gene_index_map, dim=0)  # 1 256 256 3
            control_Gene_index_map = einops.rearrange(control_Gene_index_map, 'b h w c -> b c h w').clone()  # 1 3 256 256


            # print(model.cond_stage_forward)
            cond = {"control_WSI": [control_WSI], "control_LRST": [control_LRST],
                    "control_Gene_index_map": [control_Gene_index_map], "gene_caption": [model.get_learned_conditioning([None])]}


            un_cond = {"control_WSI": [control_WSI], "control_LRST": [control_LRST],
                    "control_Gene_index_map": [control_Gene_index_map],"gene_caption": [model.get_learned_conditioning([""])]}

            shape = (3, H // 4, W // 4)

            model.control_scales = [1 * (0.825 ** float(12 - i)) for i in range(13)] if config.guess_mode else (
                        [1] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(20, 1,
                                                         shape, cond, verbose=False, eta=0,
                                                         unconditional_guidance_scale=9,
                                                         unconditional_conditioning=un_cond)

            x_samples = model.decode_first_stagedecode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') ).cpu().numpy()[0]
            x_samples = (x_samples * 127.5 + 127.5).clip(0, 255).astype(np.uint8) / 255  # 256 256
            # plt.imshow(x_samples)
            # plt.show()
            # plt.imsave('img_result/SR_9-72599/' + WSI_name + '/' + gene_name + '.png', x_samples)
            # a=plt.imread('img_result/SR_0-3299/'+WSI_name+'/'+gene_name+'.png')
            # plt.imshow(a[...,2])
            # plt.show()
            # print(img_idx / len(testset))
            a = 1
            x_samples=x_samples[...,0]
            ##### statistics
            denom = GT_final.max() - GT_final.min()
            mse = np.mean((x_samples - GT_final) ** 2, dtype=np.float64)
            rmse = np.sqrt(mse) / denom
            rmse_all[img_idx] = rmse
            ssim = get_ssim(np.uint8(x_samples * 255), np.uint8(GT_final * 255))
            ssim_all[img_idx] = ssim
            cc_ours = scipy.stats.pearsonr(np.reshape(x_samples, (-1)), np.reshape(GT_final, (-1)))
            cc_all[img_idx] = abs(cc_ours.statistic)

            if img_idx == 600:
                cc_all=cc_all[:600]
                ssim_all = ssim_all[:600]
                rmse_all = rmse_all[:600]
                cc=np.mean(cc_all)
                ssim = np.mean(ssim_all)
                rmse = np.mean(rmse_all)
                a = 2

