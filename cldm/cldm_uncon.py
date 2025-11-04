import einops
import torch
import torch as th
import torch.nn as nn
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm_uncon import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim_uncon import DDIMSampler
import torch
import torch.nn.functional as F

import numpy as np
import random



def cross_entropy(logits,labels):
    """

    Args:
        logits: [2*BS, Cls_num] where Cls_num=2*BS
        labels: [2*BS, Cls_num] where Cls_num=2*BS
    Returns:

    """


    # Example inputs
    batch_size = logits.shape[0]

    # logits = torch.tensor([[1.2, 0.5, 2.1, 0.8, 1.7],
    #                        [1.1, 1.0, 1.3, 2.1, 0.9],
    #                        [2.2, 1.1, 0.9, 1.8, 0.7]])
    # labels = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])

    # Step 1: Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)

    # Step 2: Take the log of the probabilities
    log_probs = torch.log(probabilities)

    # Step 3: Gather the log-probabilities for the correct class labels
    # Use labels as index to pick the log probabilities for the correct classes

    correct_log_probs = []
    for i in range(batch_size):
        ce_persample = torch.sum(log_probs[i] * labels[i])
        correct_log_probs.append(ce_persample)
    correct_log_probs = torch.tensor(correct_log_probs, device=logits.device,requires_grad=True)
    # Step 4: Compute the negative log-likelihood
    negative_log_likelihood = -correct_log_probs

    # Step 5: Calculate the mean loss across the batch
    loss = negative_log_likelihood.mean()
    return loss


def CC_ContrastiveLoss(WSI_feture, LRST_feature,temperature=0.07):
    """
    WSI_feture: Embeddings from the WSI encoder (batch_size, embedding_dim)
    LRST_feature: Embeddings from the LRST encoder (batch_size, embedding_dim)
    """
    # Normalize the embeddings to unit vectors

    WSI_feture = F.normalize(WSI_feture, p=2, dim=-1)
    LRST_feature = F.normalize(LRST_feature, p=2, dim=-1)

    WSI_feture_new=torch.concat((WSI_feture,LRST_feature),dim=0)
    LRST_feture_new = torch.concat((LRST_feature, WSI_feture), dim=0)
    # Compute the similarity matrix (dot product between image and text embeddings)
    logits_new = torch.matmul(WSI_feture_new, LRST_feture_new.T) / temperature

    # Create labels (positive pairs are diagonal elements)
    batch_size = WSI_feture_new.shape[0]
    labels = torch.zeros(size=(batch_size,batch_size), device=WSI_feture.device)
    for i in range(batch_size):
        labels[i,i]=1
    for i in range(int(batch_size/2)):
        labels[i, i+int(batch_size/2)] = 1
    for i in range(int(batch_size/2)):
        labels[i+int(batch_size/2),i] = 1
    labels.requires_grad=True
    # Calculate cross-entropy loss
    loss = cross_entropy(logits_new, labels)


    return loss




def CM_ContrastiveLoss(WSI_feture, LRST_feature,temperature=0.07):
    """
    WSI_feture: Embeddings from the WSI encoder (batch_size, embedding_dim)
    LRST_feature: Embeddings from the LRST encoder (batch_size, embedding_dim)
    """
    # Normalize the embeddings to unit vectors

    WSI_feture = F.normalize(WSI_feture, p=2, dim=-1)
    LRST_feature = F.normalize(LRST_feature, p=2, dim=-1)

    WSI_feture_new=torch.concat((WSI_feture,LRST_feature),dim=0)
    LRST_feture_new = torch.concat((LRST_feature, WSI_feture), dim=0)
    # Compute the similarity matrix (dot product between image and text embeddings)
    logits_new = torch.matmul(WSI_feture_new, LRST_feture_new.T) / temperature

    # Create labels (positive pairs are diagonal elements)
    batch_size = WSI_feture_new.shape[0]
    labels = torch.zeros(size=(batch_size,batch_size), device=WSI_feture.device)

    for i in range(int(batch_size/2)):
        for j in range(int(batch_size / 2),batch_size):
            labels[i,j]=1
    for i in range(int(batch_size / 2),batch_size):
        for j in range(batch_size):
            labels[i,j]=1
    labels.requires_grad = True
    # Calculate cross-entropy loss
    loss = cross_entropy(logits_new, labels)


    return loss

def IS_ContrastiveLoss(WSI_C, WSI_M,temperature=0.07):
    """
    WSI_feture: Embeddings from the WSI encoder (batch_size, embedding_dim)
    LRST_feature: Embeddings from the LRST encoder (batch_size, embedding_dim)
    """
    # Normalize the embeddings to unit vectors

    WSI_C = F.normalize(WSI_C, p=2, dim=-1)
    WSI_M = F.normalize(WSI_M, p=2, dim=-1)


    # Compute the similarity matrix (dot product between image and text embeddings)
    logits_new = torch.matmul(WSI_C, WSI_M.T) / temperature

    # Create labels (positive pairs are diagonal elements)
    batch_size = WSI_C.shape[0]
    labels = torch.zeros(size=(batch_size,batch_size), device=WSI_C.device)
    for i in range(batch_size):
        labels[i,i]=1

    labels.requires_grad=True
    # Calculate cross-entropy loss
    loss = cross_entropy(logits_new, labels)


    return loss


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block_WSI_CC = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 64, 64, 3, padding=1),
            nn.SiLU(),
            # conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            # nn.SiLU(),
            # conv_nd(dims, 96, 96, 3, padding=1),
            # nn.SiLU(),
            conv_nd(dims, 64, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, 55, 3, padding=1))
        )
        self.input_hint_block_LRST_CC = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 64, 64, 3, padding=1),
            nn.SiLU(),
            # conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            # nn.SiLU(),
            # conv_nd(dims, 96, 96, 3, padding=1),
            # nn.SiLU(),
            conv_nd(dims, 64, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, 55, 3, padding=1))
        )
        self.input_hint_block_WSI_CM = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 64, 64, 3, padding=1),
            nn.SiLU(),
            # conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            # nn.SiLU(),
            # conv_nd(dims, 96, 96, 3, padding=1),
            # nn.SiLU(),
            conv_nd(dims, 64, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, 55, 3, padding=1))
        )
        self.input_hint_block_LRST_CM = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 64, 64, 3, padding=1),
            nn.SiLU(),
            # conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            # nn.SiLU(),
            # conv_nd(dims, 96, 96, 3, padding=1),
            # nn.SiLU(),
            conv_nd(dims, 64, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, 55, 3, padding=1))
        )
        self.input_hint_block_GeneCodeMap = TimestepEmbedSequential(
            zero_module(conv_nd(dims, 1, 1, 4, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint_WSI,hint_LRST,hint_GeneCodeMap, timesteps, context,train_flag, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint_WSI_CC = self.input_hint_block_WSI_CC(hint_WSI, emb, context) # b, 55, 64, 64
        guided_hint_LRST_CC = self.input_hint_block_LRST_CC(hint_LRST, emb, context)# b, 55, 64, 64
        guided_hint_WSI_CM = self.input_hint_block_WSI_CM(hint_WSI, emb, context)  # b, 55, 64, 64
        guided_hint_LRST_CM = self.input_hint_block_LRST_CM(hint_LRST, emb, context)  # b, 55, 64, 64
        guided_hint_GeneCodeMap = torch.nn.functional.interpolate(hint_GeneCodeMap, (64, 64))  # b, 1, 64, 64



        ######################## impute
        if train_flag:
            num=np.random.rand()
            if num<0.05:

                guided_hint_WSI_CC = self.input_hint_block_WSI_CC(hint_WSI, emb, context)  # b, 55, 64, 64
                temperature=0.07
                WSI_C = F.normalize(torch.reshape(guided_hint_WSI_CC, (guided_hint_WSI_CC.shape[0],-1)), p=2, dim=-1) # b, X
                logits_new = torch.matmul(WSI_C, WSI_C.T) / temperature
                probabilities = F.softmax(logits_new[:,:-1], dim=1)[0,:]
                guided_hint_LRST_CC[x.shape[0]-1]=0
                for bs in range(x.shape[0]-1):
                    guided_hint_LRST_CC+=probabilities[bs]*guided_hint_LRST_CC[bs]

                guided_hint_WSI_CM = self.input_hint_block_WSI_CM(hint_WSI, emb, context)  # b, 55, 64, 64
                temperature = 0.07
                WSI_M = F.normalize(torch.reshape(guided_hint_WSI_CM, (guided_hint_WSI_CM.shape[0], -1)), p=2,dim=-1)  # b, X
                logits_new = torch.matmul(WSI_M, WSI_M.T) / temperature
                probabilities = F.softmax(logits_new[:, :-1], dim=1)[0, :]
                guided_hint_LRST_CM[x.shape[0] - 1] = 0
                for bs in range(x.shape[0] - 1):
                    guided_hint_LRST_CM += probabilities[bs] * guided_hint_LRST_CM[bs]

        ######################## noise denoise
        std_dev = 0.03
        np.random.seed(512)
        random.seed(512)
        guided_hint_LRST_CC = guided_hint_LRST_CC + torch.randn_like(guided_hint_LRST_CC) * std_dev
        mean = guided_hint_LRST_CC.mean()
        std = guided_hint_LRST_CC.std()
        guided_hint_LRST_CC = (guided_hint_LRST_CC - mean) / std
        guided_hint_LRST_CM = guided_hint_LRST_CM + torch.randn_like(guided_hint_LRST_CM) * std_dev
        mean = guided_hint_LRST_CM.mean()
        std = guided_hint_LRST_CM.std()
        guided_hint_LRST_CM = (guided_hint_LRST_CM - mean) / std



        guided_hint=torch.concat((guided_hint_WSI_CC,guided_hint_LRST_CC,guided_hint_WSI_CM,guided_hint_LRST_CM
                                  ,guided_hint_GeneCodeMap,guided_hint_GeneCodeMap,guided_hint_GeneCodeMap,guided_hint_GeneCodeMap),dim=1)

        CC_CL=CC_ContrastiveLoss(WSI_feture=torch.reshape(guided_hint_WSI_CC, (guided_hint_WSI_CC.shape[0],-1)), LRST_feature=torch.reshape(guided_hint_LRST_CC, (guided_hint_LRST_CC.shape[0],-1)))
        CM_CL=CM_ContrastiveLoss(WSI_feture=torch.reshape(guided_hint_WSI_CM, (guided_hint_WSI_CM.shape[0],-1)), LRST_feature=torch.reshape(guided_hint_LRST_CM, (guided_hint_LRST_CM.shape[0],-1)))
        IS_CL=IS_ContrastiveLoss(WSI_C=torch.reshape(guided_hint_WSI_CC, (guided_hint_WSI_CC.shape[0],-1)), WSI_M=torch.reshape(guided_hint_WSI_CM, (guided_hint_WSI_CM.shape[0],-1)))


        outs = []
        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        return outs,CC_CL,CM_CL,IS_CL



class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_LRST,control_WSI,control_Gene_index_map, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config) #ControlNet
        self.control_LRST = control_LRST
        self.control_WSI = control_WSI
        self.control_Gene_index_map = control_Gene_index_map

        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        # self.control_scales =[1 * (0.825 ** float(12 - i)) for i in range(13)]

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control_LRST = batch[self.control_LRST]
        control_WSI = batch[self.control_WSI]
        control_Gene_index_map = batch[self.control_Gene_index_map]
        if bs is not None:
            control_LRST = control_LRST[:bs]
            control_WSI = control_WSI[:bs]
            control_Gene_index_map = control_Gene_index_map[:bs]
        control_LRST = control_LRST.to(self.device)
        control_WSI = control_WSI.to(self.device)
        control_Gene_index_map = control_Gene_index_map.to(self.device)

        control_LRST = einops.rearrange(control_LRST, 'b h w c -> b c h w') # b 3 256 256
        control_WSI = einops.rearrange(control_WSI, 'b h w c -> b c h w')  # b 3 256 256
        control_Gene_index_map = einops.rearrange(control_Gene_index_map, 'b h w c -> b c h w')  # b 1 256 256

        control_LRST = control_LRST.to(memory_format=torch.contiguous_format).float()
        control_WSI = control_WSI.to(memory_format=torch.contiguous_format).float()
        control_Gene_index_maplog_dict = control_Gene_index_map.to(memory_format=torch.contiguous_format).float()

        # return x, dict(c_crossattn=[c], c_concat=[control])
        return x, dict(control_LRST=[control_LRST], control_WSI=[control_WSI],control_Gene_index_map=[control_Gene_index_map])

    def apply_model(self, x_noisy, t, cond,train_flag=True, *args, **kwargs):

        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        # if cond['c_crossattn'][0] is not None:
        #     cond_txt = torch.cat(cond['c_crossattn'], 1)
        # else:
        cond_txt=None

        # if cond['c_concat'] is None:
        #     eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        # else:

        control,CC_CL,CM_CL,IS_CL = self.control_model(x=x_noisy, hint_WSI=torch.cat(cond['control_WSI'], 1),hint_LRST=torch.cat(cond['control_LRST'], 1),
                                     hint_GeneCodeMap=torch.cat(cond['control_Gene_index_map'], 1), timesteps=t, context=cond_txt,train_flag=train_flag)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps,CC_CL,CM_CL,IS_CL

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        # c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0]
        c_cat_WSI,c_cat_LRST,c_cat_GeneCodeMap, c = c["control_WSI"][0][:N],c["control_LRST"][0][:N],c["control_Gene_index_map"][0][:N], None
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control_WSI"] = c_cat_WSI * 2.0 - 1.0
        log["control_LRST"] = c_cat_LRST * 2.0 - 1.0
        # log["control_LRST"] = c_cat_LRST
        log["control_GeneCodeMap"] = c_cat_GeneCodeMap * 2.0 - 1.0


        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat_WSI = c_cat_WSI  # torch.zeros_like(c_cat)
            uc_cat_LRST = c_cat_LRST  # torch.zeros_like(c_cat)
            uc_cat_GeneCodeMap = c_cat_GeneCodeMap  # torch.zeros_like(c_cat)
            uc_full = {"control_WSI": [uc_cat_WSI],"control_LRST": [uc_cat_LRST],"control_Gene_index_map": [uc_cat_GeneCodeMap], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"control_WSI": [c_cat_WSI],"control_LRST": [c_cat_LRST],"control_Gene_index_map": [c_cat_GeneCodeMap], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)

            x_samples_cfg_0=torch.unsqueeze(x_samples_cfg[:, 0, ...],1)
            x_samples_cfg_1 = torch.unsqueeze(x_samples_cfg[:, 1, ...], 1)
            x_samples_cfg_2 = torch.unsqueeze(x_samples_cfg[:, 2, ...], 1)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}-d0"] = x_samples_cfg_0
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}-d1"] = x_samples_cfg_1
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}-d2"] = x_samples_cfg_2

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["control_WSI"][0].shape
        shape = (self.channels, h // 4, w // 4)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
