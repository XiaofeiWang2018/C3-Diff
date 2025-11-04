import torch
from share import *
from ldm.util import instantiate_from_config
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import *
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import datetime
from cldm.data_light import *
import numpy as np

config = OmegaConf.load('./models/cldm_ours.yaml')



model = instantiate_from_config(config.model).cpu()
model.load_state_dict(load_state_dict(config.resume_path, location='cpu'))
model.learning_rate = config.learning_rate
model.sd_locked = config.sd_locked
model.only_mid_control = config.only_mid_control
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_imgloger_path=now +'_'+config.modelname+'_'+config.dataset_choose
logdir='logs/'+save_imgloger_path+'/'
saveimgdir=os.path.join(logdir, "image")
ckptdir = os.path.join(logdir, "checkpoints")
cfgdir = os.path.join(logdir, "configs")
os.makedirs(ckptdir, exist_ok=True)
os.makedirs(cfgdir, exist_ok=True)
os.makedirs(saveimgdir, exist_ok=True)
seed_everything(config.seed)

data = DataModuleFromConfig(batch_size=config.batch_size,config=config)
data.setup()


logger = ImageLogger(batch_frequency=config.logger_freq,save_path=saveimgdir)
trainer = pl.Trainer(gpus=[0], precision=32, callbacks=[logger],accelerator='ddp')
OmegaConf.save(config,os.path.join(cfgdir, "{}-project.yaml".format(now)))
trainer.callbacks[4].dirpath=ckptdir
trainer.callbacks[4]._every_n_train_steps=int(99*config.gene_num/config.batch_size/3)
trainer.callbacks[4].save_top_k=-1

# a=trainer.callbacks[4]
# Train!
trainer.fit(model, data)
