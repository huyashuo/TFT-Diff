import os
import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))

import torch
import numpy as np

from engine.solver import Trainer
from Utils.metric_utils import visualization
from Data.build_dataloader import build_dataloader
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one, cond_fn

class Args_Example:
    def __init__(self) -> None:
        self.config_path = '../Config/chb.yaml'
        self.gpu = 0
        self.save_dir = '../chb_exp'
        os.makedirs(self.save_dir, exist_ok=True)

args =  Args_Example()
configs = load_yaml_config(args.config_path)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

dl_info = build_dataloader(configs, args)
model = instantiate_from_config(configs['model']).to(device)
classifier = instantiate_from_config(configs['classifier']).to(device)
trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)

trainer.train()

trainer.train_classfier(classifier)

dataset = dl_info['dataset']
seq_length, feature_dim = dataset.window, dataset.var_num

model_kwargs = {}
model_kwargs['classifier'] = trainer.classifier
model_kwargs['classifier_scale'] = 0.1

ori_data_1 = dataset.normalize(dataset.data_1)

model_kwargs['y'] = torch.ones((1001, )).long().to(device)

fake_data_1 = trainer.sample(num=len(ori_data_1), size_every=1001, shape=[seq_length, feature_dim],
                             model_kwargs=model_kwargs, cond_fn=cond_fn)
if dataset.auto_norm:
    ori_data_1 = unnormalize_to_zero_to_one(ori_data_1)
    fake_data_1 = unnormalize_to_zero_to_one(fake_data_1)

np.save(os.path.join(args.save_dir, f'ddpm_fake_1_eeg.npy'), fake_data_1)
np.save(os.path.join(args.save_dir, f'ori_data_1.npy'), ori_data_1)

visualization(ori_data=ori_data_1, generated_data=fake_data_1, analysis='pca', compare=ori_data_1.shape[0])

visualization(ori_data=ori_data_1, generated_data=fake_data_1, analysis='tsne', compare=ori_data_1.shape[0])

visualization(ori_data=ori_data_1, generated_data=fake_data_1, analysis='kernel', compare=ori_data_1.shape[0])