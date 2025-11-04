import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy import interpolate
from PIL import Image

def np_norm(inp):
    max_in=np.max(inp)
    min_in = np.min(inp)
    return (inp-min_in)/(max_in-min_in)

def gray_value_of_gene(gene_class,gene_order):
    gene_order=list(gene_order)
    Index=gene_order.index(gene_class)
    interval=255/len(gene_order)
    value=Index*interval
    return int(value)



class MyDataset(Dataset):
    def __init__(self,status):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        a=1
        self.status=status
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # print(self.status)
        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)
        new_dim = (256, 256)
        source = cv2.resize(source, new_dim)
        target = cv2.resize(target, new_dim)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        # if self.status=='Test':
        #     print('TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT')

        return dict(jpg=target, txt=prompt, hint=source)


class Melanoma_dataset(Dataset):
    def __init__(self, data_root,SR_times,status,gene_num):

        if status=='Train':
            sample_name=['20230918_train']
        elif status=='Test':
            sample_name = ['20230918_test']

        SR_ST_all=[]
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
        SR_ST_all=np.array(SR_ST_all)
        # Sum=np.sum(SR_ST_all,axis=(0,2,3))
        # gene_order=np.argsort(Sum)[::-1][0:gene_num]
        # np.save(data_root + 'gene_order.npy',gene_order)
        gene_order=np.load(data_root + 'gene_order.npy')[0:gene_num]
        self.SR_ST_all=SR_ST_all[:,gene_order,...].astype(np.float64) # (X,50,256,256)



        for ii in range(self.SR_ST_all.shape[0]):
            for jj in range(self.SR_ST_all.shape[1]):
                if np.sum(self.SR_ST_all[ii, jj]) != 0:
                    Max=np.max(self.SR_ST_all[ii,jj])
                    Min=np.min(self.SR_ST_all[ii,jj])
                    self.SR_ST_all[ii,jj]=(self.SR_ST_all[ii,jj]-Min)/(Max-Min)

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

        ## WSI 5120
        WSI_5120_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:

                WSI_5120 = np.load(data_root + 'Xenium/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)
        a=1




    def __len__(self):
        return self.WSI_5120_all.shape[0]

    def _getitem__(self, index):

        return self.SR_ST_all[index], self.spot_ST_all[index], self.WSI_5120_all[index]



class Xenium_dataset(Dataset):
    def __init__(self, data_root,SR_times,status,gene_num):

        if status=='Train':
            sample_name=['01220101', '01220102', 'NC1', 'NC2']
            # sample_name = ['NC2']
        elif status=='Test':
            sample_name = ['01220201', '01220202']
            # sample_name = ['01220202']
        self.gene_num=gene_num
        SR_ST_all=[]
        gene_order = np.load(data_root + 'gene_order.npy')[0:gene_num]
        self.gene_order = []

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
        # Sum=np.sum(SR_ST_all,axis=(0,2,3))
        # gene_order=np.argsort(Sum)[::-1][0:]
        # np.save(data_root + 'gene_order.npy',gene_order)
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
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)
        self.WSI_5120_all=np.repeat(self.WSI_5120_all, gene_num, axis=0)
        a=1




    def __len__(self):
        return self.spot_ST_all.shape[0]

    def __getitem__(self, index):

        target=self.SR_ST_all[index]
        target = np.expand_dims(target, axis=2)
        target = np.repeat(target, 3, axis=2)  #[256,256,3]
        source_STmap = self.spot_ST_all[index]# [26,26]
        source_STmap=np.expand_dims(source_STmap,axis=2)  # [26,26,1]
        source_STmap = np.repeat(source_STmap, 3, axis=2) # [26,26,3]
        source_STmap=np.array(Image.fromarray(np.uint8(source_STmap*255)).resize((256,256)))/255 # [256,256,3]
        # source_STmap = np.expand_dims(source_STmap, axis=2)  # [256,256,1]
        ## Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 0.5) - 1.0
        # source_STmap= (source_STmap.astype(np.float32) / 0.5) - 1.0 ## can be annotated


        source_WSI=self.WSI_5120_all[index]
        source_WSI = np.transpose(source_WSI, axes=(1, 2, 0)) # [256,256,3]
        # source_WSI=np_norm(source_WSI) ## can be annotated
        ########################################################### Normalize source images to [0, 1].
        source_WSI = source_WSI.astype(np.float32) / 255.0


        gene_class=self.gene_order[index]
        gene_caption='spatial expression map of gene '+str(gene_class)
        # gene_caption = ' '
        assert  self.gene_num<255
        Gene_code=gray_value_of_gene(gene_class,self.gene_order[0:self.gene_num])
        Gene_index_map=np.ones(shape=(256,256,1))*Gene_code / 255.0

        #
        return dict(HRST=target, LRST=source_STmap, WSI=source_WSI,gene_caption=gene_caption, gene_class=gene_class, Gene_index_map=Gene_index_map)
        # return


if __name__ == '__main__':
    data_root=r'/home/zeiler/ST_proj/data/Breast_cancer/'
    a=Xenium_dataset(data_root=data_root,SR_times=10,status='Train',gene_num=100)
    L=len(a)
    SR_ST_all, spot_ST_all, WSI_5120_all=a._getitem__(132)
    WSI_5120 = np.transpose(WSI_5120_all, axes=(1,2,0))
    SR_ST=SR_ST_all[0]
    spot_ST = spot_ST_all[0]
    plt.imshow(WSI_5120)
    plt.show()
    plt.imshow(SR_ST)
    plt.show()
    plt.imshow(spot_ST)
    plt.show()
    a=1
